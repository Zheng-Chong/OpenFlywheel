"""Stage 3: Multi-view rendering, VLM / 2D segmentation back-projection, and semantic part tagging.

Augments Stage 1/2 `affordance.json` sidecars with fine-grained semantic `part_tag`
labels (`handle`, `body`, `rim`, `spout`, `lid`, `neck`, `cap`, `base`, etc.) and
sets `tag_source: "vlm"`.

Pipeline:
  1. Build an orbital multi-view pinhole camera rig around the object's local bounds.
  2. Render RGB + metric depth + foreground mask per view:
     - CPU pure-NumPy Z-buffer rasterizer (`render_mesh_views`), zero GPU dependency.
     - Isaac Sim RTX renderer (`render_isaac_views`) for photorealistic textured views on GPU Slurm.
  3. Obtain 2D semantic labels via either:
     - 2D segmentation masks (e.g. Grounded-SAM / Florence-2 / PartSlip / VLM region boxes)
       back-projected onto 3D surface points & grasp contact points with Z-buffer occlusion check.
     - Multi-view VLM grasp-marker prompting via any OpenAI-compatible multimodal API
       (`OPENFLYWHEEL_VLM_API_URL` / `OPENAI_BASE_URL`, `OPENFLYWHEEL_VLM_API_KEY` /
       `OPENAI_API_KEY`, `OPENFLYWHEEL_VLM_MODEL`).
  4. Fuse multi-view votes onto each grasp, update `part_tag` & `tag_source: "vlm"` (preserving
     `part_tag_geom`), and record `semantic_annotation` metadata in `affordance.json`.

CLI usage:
    python scripts/objects/affordance_vlm.py render path/to/object.obj -o /tmp/views
    python scripts/objects/affordance_vlm.py annotate affordance.json --mesh object.obj --masks-dir /tmp/masks
    python scripts/objects/affordance_vlm.py annotate affordance.json --mesh object.obj --backend openai
    python scripts/objects/affordance_vlm.py batch --manifest manifest.json --backend openai
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from scripts.objects.affordance import (
    _ensure_pxr,
    close_pxr_app,
    load_mesh,
    resolve_mesh_path,
    sample_surface_points,
    triangle_geometry,
)

# ---------------------------------------------------------------------------
# Canonical semantic part taxonomy
# ---------------------------------------------------------------------------

SEMANTIC_PART_TAGS: tuple[str, ...] = (
    "handle",
    "body",
    "rim",
    "spout",
    "lid",
    "neck",
    "cap",
    "base",
    "blade",
    "stem",
    "knob",
    "ear",
    "prong",
    "trigger",
    "unknown",
)

PART_TAG_ALIASES: dict[str, str] = {
    "grip": "handle",
    "ear_handle": "handle",
    "loop": "handle",
    "bail": "handle",
    "lip": "rim",
    "mouth": "rim",
    "opening": "rim",
    "edge": "rim",
    "brim": "rim",
    "nozzle": "spout",
    "pour_spout": "spout",
    "beak": "spout",
    "cover": "lid",
    "top_lid": "lid",
    "stopper": "cap",
    "bottle_cap": "cap",
    "screw_cap": "cap",
    "bottleneck": "neck",
    "collar": "neck",
    "throat": "neck",
    "bottom": "base",
    "foot": "base",
    "stand": "base",
    "wall": "body",
    "barrel": "body",
    "container": "body",
    "cup_body": "body",
    "bottle_body": "body",
    "body_top": "body",
    "body_side": "body",
    "body_low": "body",
    "side": "body",
    "shaft": "stem",
    "stalk": "stem",
    "tine": "prong",
    "button": "knob",
    "dial": "knob",
}


_CANONICAL_NON_UNKNOWN = frozenset(t for t in SEMANTIC_PART_TAGS if t != "unknown")


def normalize_part_tag(raw: Any) -> str:
    """Normalize a raw VLM or 2D segmentation label into the canonical part taxonomy.

    Uses exact match first, then right-to-left word-token matching (preferring canonical
    part nouns over aliases) so compound phrases like ``"side_handle"`` or ``"flip_lid"``
    resolve to ``"handle"`` / ``"lid"`` rather than colliding on substrings like ``"side"``
    or ``"lip"``.
    """
    if raw is None:
        return "unknown"
    text = str(raw).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if not text:
        return "unknown"
    if text in SEMANTIC_PART_TAGS:
        return text
    if text in PART_TAG_ALIASES:
        return PART_TAG_ALIASES[text]
    tokens = [tok for tok in text.split("_") if tok]
    # Prefer canonical nouns (right-to-left: head noun in English compounds comes last)
    for tok in reversed(tokens):
        if tok in _CANONICAL_NON_UNKNOWN:
            return tok
    # Check adjacent bigram aliases (e.g. "pour_spout", "bottle_cap") before single-token aliases
    for i in range(len(tokens) - 2, -1, -1):
        bigram = f"{tokens[i]}_{tokens[i + 1]}"
        if bigram in PART_TAG_ALIASES:
            return PART_TAG_ALIASES[bigram]
    for tok in reversed(tokens):
        if tok in PART_TAG_ALIASES:
            return PART_TAG_ALIASES[tok]
    return "unknown"


# ---------------------------------------------------------------------------
# Pinhole camera model & multi-view orbital rig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraView:
    """OpenCV-convention pinhole camera (+X right, +Y down, +Z forward)."""

    view_id: str
    width: int
    height: int
    K: np.ndarray  # (3, 3) float64 intrinsics
    cam_to_world: np.ndarray  # (4, 4) float64 rigid transform

    def world_to_cam(self) -> np.ndarray:
        """Inverse rigid transform: world -> camera coordinates."""
        rot = self.cam_to_world[:3, :3]
        trans = self.cam_to_world[:3, 3]
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rot.T
        out[:3, 3] = -(rot.T @ trans)
        return out

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project (N, 3) world points to (u, v, z_cam) arrays of shape (N,)."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        rot = self.cam_to_world[:3, :3]
        trans = self.cam_to_world[:3, 3]
        pts_cam = (pts - trans[None, :]) @ rot  # equivalent to R^T (p - t)
        z = pts_cam[:, 2]
        safe_z = np.where(np.abs(z) > 1e-9, z, 1e-9)
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        u = fx * (pts_cam[:, 0] / safe_z) + cx
        v = fy * (pts_cam[:, 1] / safe_z) + cy
        return u, v, z

    def unproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Unproject pixel coordinates (u, v) and metric depths (z_cam) to (N, 3) world points."""
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
        d_arr = np.asarray(depth, dtype=np.float64).reshape(-1)
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        x_cam = (u_arr - cx) * d_arr / fx
        y_cam = (v_arr - cy) * d_arr / fy
        pts_cam = np.column_stack([x_cam, y_cam, d_arr])
        rot = self.cam_to_world[:3, :3]
        trans = self.cam_to_world[:3, 3]
        return pts_cam @ rot.T + trans[None, :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "width": int(self.width),
            "height": int(self.height),
            "K": self.K.tolist(),
            "cam_to_world": self.cam_to_world.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CameraView:
        return cls(
            view_id=str(data["view_id"]),
            width=int(data["width"]),
            height=int(data["height"]),
            K=np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
            cam_to_world=np.asarray(data["cam_to_world"], dtype=np.float64).reshape(4, 4),
        )


@dataclass
class RenderedView:
    """One rendered multi-view observation (RGB, metric depth, foreground mask, camera)."""

    view: CameraView
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters along +Z_cam, np.inf for background
    mask: np.ndarray  # (H, W) bool, True where object is visible


def look_at_cam_to_world(
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Build a 4x4 OpenCV cam_to_world matrix (+X right, +Y down, +Z forward)."""
    eye_v = np.asarray(eye, dtype=np.float64).reshape(3)
    target_v = np.asarray(target, dtype=np.float64).reshape(3)
    up_v = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target_v - eye_v
    f_norm = float(np.linalg.norm(forward))
    if f_norm < 1e-9:
        raise ValueError("Camera eye and target cannot coincide")
    forward /= f_norm

    up_norm = float(np.linalg.norm(up_v))
    up_u = up_v / up_norm if up_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_u))) > 0.999:
        up_u = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(forward, up_u)
    right /= float(np.linalg.norm(right))
    down = np.cross(forward, right)
    down /= float(np.linalg.norm(down))

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = down
    mat[:3, 2] = forward
    mat[:3, 3] = eye_v
    return mat


def pinhole_intrinsics(width: int, height: int, fov_deg: float = 45.0) -> np.ndarray:
    """Build a centered pinhole intrinsic matrix K for vertical FOV `fov_deg`."""
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(fov_deg))
    fx = fy
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_multiview_cameras(
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    *,
    n_azimuth: int = 6,
    elevations_deg: Sequence[float] = (20.0, 50.0),
    include_top: bool = True,
    image_size: tuple[int, int] = (256, 256),
    fov_deg: float = 45.0,
    margin_factor: float = 1.35,
) -> list[CameraView]:
    """Construct an orbital multi-view camera rig framing the object's bounding sphere."""
    b_min = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    b_max = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    center = 0.5 * (b_min + b_max)
    radius = max(0.5 * float(np.linalg.norm(b_max - b_min)), 0.02)

    width, height = int(image_size[0]), int(image_size[1])
    K = pinhole_intrinsics(width, height, fov_deg=fov_deg)
    min_half_fov = min(
        math.atan(0.5 * width / float(K[0, 0])),
        math.atan(0.5 * height / float(K[1, 1])),
    )
    dist = margin_factor * radius / max(math.sin(min_half_fov), 1e-3)

    views: list[CameraView] = []
    idx = 0
    for elev_deg in elevations_deg:
        elev = math.radians(float(elev_deg))
        cos_e = math.cos(elev)
        sin_e = math.sin(elev)
        for k in range(max(1, int(n_azimuth))):
            az = 2.0 * math.pi * k / max(1, int(n_azimuth))
            direction = np.array([cos_e * math.cos(az), cos_e * math.sin(az), sin_e], dtype=np.float64)
            eye = center + dist * direction
            c2w = look_at_cam_to_world(eye, center)
            views.append(
                CameraView(
                    view_id=f"view_{idx:02d}_az{int(round(math.degrees(az))):03d}_el{int(round(elev_deg)):02d}",
                    width=width,
                    height=height,
                    K=K.copy(),
                    cam_to_world=c2w,
                )
            )
            idx += 1

    if include_top:
        eye = center + np.array([0.0, 0.0, dist], dtype=np.float64)
        c2w = look_at_cam_to_world(eye, center, up=(0.0, 1.0, 0.0))
        views.append(
            CameraView(
                view_id=f"view_{idx:02d}_top",
                width=width,
                height=height,
                K=K.copy(),
                cam_to_world=c2w,
            )
        )
    return views


# ---------------------------------------------------------------------------
# Pure-NumPy CPU Z-buffer renderer (RGB + metric depth + mask)
# ---------------------------------------------------------------------------


def render_mesh_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    view: CameraView,
    *,
    base_color: tuple[int, int, int] = (195, 200, 210),
    bg_color: tuple[int, int, int] = (245, 246, 248),
    near: float = 1e-3,
) -> RenderedView:
    """Render a triangle mesh to RGB + metric depth + mask using a pure-NumPy Z-buffer."""
    w, h = view.width, view.height
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = bg_color[0]
    rgb[:, :, 1] = bg_color[1]
    rgb[:, :, 2] = bg_color[2]
    depth = np.full((h, w), np.inf, dtype=np.float32)

    if len(vertices) == 0 or len(faces) == 0:
        return RenderedView(view=view, rgb=rgb, depth=depth, mask=np.zeros((h, w), dtype=bool))

    rot = view.cam_to_world[:3, :3]
    trans = view.cam_to_world[:3, 3]
    v_cam = (vertices.astype(np.float64) - trans[None, :]) @ rot

    fx, fy = float(view.K[0, 0]), float(view.K[1, 1])
    cx, cy = float(view.K[0, 2]), float(view.K[1, 2])

    tris_cam = v_cam[faces]  # (F, 3, 3)
    valid_tri = np.all(tris_cam[:, :, 2] > near, axis=1)
    if not np.any(valid_tri):
        return RenderedView(view=view, rgb=rgb, depth=depth, mask=np.zeros((h, w), dtype=bool))
    tris_cam = tris_cam[valid_tri]

    # Face normals in camera space for Lambertian shading (two-sided so thin-walled interiors shade well)
    e1 = tris_cam[:, 1] - tris_cam[:, 0]
    e2 = tris_cam[:, 2] - tris_cam[:, 0]
    fn = np.cross(e1, e2)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    non_deg = fn_norm[:, 0] > 1e-12
    tris_cam = tris_cam[non_deg]
    fn = fn[non_deg] / fn_norm[non_deg]
    if len(tris_cam) == 0:
        return RenderedView(view=view, rgb=rgb, depth=depth, mask=np.zeros((h, w), dtype=bool))

    # Key light slightly above-left of camera + gentle fill
    key_light = np.array([-0.35, -0.45, -0.82], dtype=np.float64)
    key_light /= float(np.linalg.norm(key_light))
    shade = 0.28 + 0.72 * np.abs(fn @ key_light)  # (F,)
    base_rgb = np.asarray(base_color, dtype=np.float64)
    tri_colors = np.clip(shade[:, None] * base_rgb[None, :], 0.0, 255.0).astype(np.uint8)

    # Project all triangle vertices
    zs = tris_cam[:, :, 2]  # (F, 3)
    us = fx * (tris_cam[:, :, 0] / zs) + cx
    vs = fy * (tris_cam[:, :, 1] / zs) + cy
    inv_zs = 1.0 / zs

    # Clip bounding boxes to screen
    u_min = np.maximum(0, np.floor(us.min(axis=1)).astype(np.int32))
    u_max = np.minimum(w - 1, np.ceil(us.max(axis=1)).astype(np.int32))
    v_min = np.maximum(0, np.floor(vs.min(axis=1)).astype(np.int32))
    v_max = np.minimum(h - 1, np.ceil(vs.max(axis=1)).astype(np.int32))
    on_screen = (u_max >= u_min) & (v_max >= v_min)

    for i in np.flatnonzero(on_screen):
        x0, y0 = float(us[i, 0]), float(vs[i, 0])
        x1, y1 = float(us[i, 1]), float(vs[i, 1])
        x2, y2 = float(us[i, 2]), float(vs[i, 2])
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue
        inv_denom = 1.0 / denom

        xs = np.arange(u_min[i], u_max[i] + 1, dtype=np.float64) + 0.5
        ys = np.arange(v_min[i], v_max[i] + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)  # (bh, bw)

        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) * inv_denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) * inv_denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue

        iz = w0 * float(inv_zs[i, 0]) + w1 * float(inv_zs[i, 1]) + w2 * float(inv_zs[i, 2])
        z_pix = np.where(inside & (iz > 1e-9), 1.0 / np.maximum(iz, 1e-9), np.inf).astype(np.float32)

        sub_depth = depth[v_min[i] : v_max[i] + 1, u_min[i] : u_max[i] + 1]
        closer = inside & (z_pix < sub_depth)
        if not np.any(closer):
            continue
        sub_depth[closer] = z_pix[closer]
        sub_rgb = rgb[v_min[i] : v_max[i] + 1, u_min[i] : u_max[i] + 1]
        sub_rgb[closer] = tri_colors[i]

    mask = np.isfinite(depth)
    return RenderedView(view=view, rgb=rgb, depth=depth, mask=mask)


def render_mesh_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: Sequence[CameraView],
    *,
    base_color: tuple[int, int, int] = (195, 200, 210),
    bg_color: tuple[int, int, int] = (245, 246, 248),
) -> list[RenderedView]:
    """Render all requested camera views of a mesh on CPU."""
    return [render_mesh_view(vertices, faces, v, base_color=base_color, bg_color=bg_color) for v in views]


def render_isaac_views(
    asset_dir: Path,
    views: Sequence[CameraView],
    *,
    device: str = "cuda:0",
    warmup_renders: int = 20,
) -> list[RenderedView]:
    """Render photorealistic RGB + metric depth views inside Isaac Sim (GPU Slurm).

    Spawns the asset USD at the origin with all rigid bodies disabled/kinematic and
    timeline paused so the object never moves under gravity while driving a reused
    `UsdGeom.Camera` + `omni.replicator.core` render product (`rgb` +
    `distance_to_image_plane`), matching `/shared/opt/robodojo/isaac_render_check.py`.
    Raises RuntimeError if Isaac Sim is not available in the current environment.
    """
    try:
        try:
            import omni.replicator.core as rep  # type: ignore[import-not-found]
        except ImportError:
            _ensure_pxr()
            import omni.replicator.core as rep  # type: ignore[import-not-found]
        import carb  # type: ignore[import-not-found]
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import omni.timeline  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Isaac Sim is not available in this environment; use CPU render_mesh_views() or run under isaac-sim."
        ) from exc

    asset_path = Path(asset_dir)
    usd_file = next(
        (asset_path / name for name in ("object.usdz", "object.usd", "object.usda") if (asset_path / name).is_file()),
        None,
    )
    if usd_file is None:
        raise FileNotFoundError(f"No object.usd[a|z] found in {asset_path}")

    if not views:
        return []

    carb.settings.get_settings().set_float("/rtx/translucency/worldEps", 0.0)
    if sim_utils.SimulationContext.instance() is not None:
        sim_utils.SimulationContext.clear_instance()
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Keep the timeline paused so PhysX never simulates gravity on rigid-body assets.
    timeline = omni.timeline.get_timeline_interface()
    if timeline is not None:
        timeline.pause()

    dome = sim_utils.DomeLightCfg(intensity=450.0)
    dome.func("/World/dome", dome)
    sun = sim_utils.DistantLightCfg(intensity=2200.0, angle=5.0)
    sun.func("/World/sun", sun, orientation=(0.9397, 0.2418, 0.2418, 0.0))

    obj_root_path = "/World/target_00000_0"
    obj_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_file))
    obj_cfg.func(obj_root_path, obj_cfg, translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

    # Belt-and-suspenders: freeze any authored RigidBodyAPI on the spawned asset subtree
    # so the object stays strictly at its local origin across every rendered view.
    obj_root = stage.GetPrimAtPath(obj_root_path)
    if obj_root and obj_root.IsValid():
        for prim in Usd.PrimRange(obj_root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                rb_api.CreateKinematicEnabledAttr(True)
                rb_api.CreateRigidBodyEnabledAttr(False)

    def _to_np(x: Any, dtype: Any = np.float64) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype)

    def _step_renderer() -> None:
        try:
            rep.orchestrator.step(delta_time=0.0, pause_timeline=True)
        except TypeError:
            rep.orchestrator.step()

    # Single UsdGeom.Camera + Replicator render product reused across views of equal resolution
    first_view = views[0]
    cam_path = "/World/mv_cam"
    usd_cam = UsdGeom.Camera.Define(stage, cam_path)
    horiz_ap = 20.955  # mm
    usd_cam.CreateHorizontalApertureAttr().Set(horiz_ap)
    usd_cam.CreateVerticalApertureAttr().Set(horiz_ap * float(first_view.height) / float(first_view.width))
    usd_cam.CreateFocalLengthAttr().Set(float(first_view.K[0, 0]) * horiz_ap / float(first_view.width))
    usd_cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.001, 50.0))
    xform_op = UsdGeom.Xformable(usd_cam.GetPrim()).AddTransformOp()

    current_res = (int(first_view.width), int(first_view.height))
    rp = rep.create.render_product(cam_path, current_res)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_ann.attach(rp)
    depth_ann.attach(rp)

    cv_to_usd = np.diag([1.0, -1.0, -1.0])
    rendered: list[RenderedView] = []
    for i, view in enumerate(views):
        view_res = (int(view.width), int(view.height))
        if view_res != current_res:
            rgb_ann.detach()
            depth_ann.detach()
            if hasattr(rp, "destroy"):
                rp.destroy()
            current_res = view_res
            rp = rep.create.render_product(cam_path, current_res)
            rgb_ann.attach(rp)
            depth_ann.attach(rp)

        # Convert OpenCV cam_to_world (+X right, +Y down, +Z forward) to USD (+X right, +Y up, -Z forward)
        c2w_usd = np.eye(4, dtype=np.float64)
        c2w_usd[:3, :3] = view.cam_to_world[:3, :3] @ cv_to_usd
        c2w_usd[:3, 3] = view.cam_to_world[:3, 3]
        # USD Gf.Matrix4d uses row-vector convention (transpose of column-vector 4x4)
        xform_op.Set(Gf.Matrix4d(*c2w_usd.T.reshape(-1).tolist()))
        usd_cam.GetVerticalApertureAttr().Set(horiz_ap * float(view.height) / float(view.width))
        usd_cam.GetFocalLengthAttr().Set(float(view.K[0, 0]) * horiz_ap / float(view.width))

        n_steps = max(3, int(warmup_renders) // 6) if i == 0 else 2
        for _ in range(n_steps):
            _step_renderer()

        rgba = _to_np(rgb_ann.get_data(), dtype=np.uint8)
        raw_depth = depth_ann.get_data()
        if rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError(
                f"Isaac Sim camera returned invalid RGB shape {rgba.shape} for {view.view_id}. "
                "Ensure Isaac Sim was launched with AppLauncher(headless=True, enable_cameras=True)."
            )
        rgb = rgba[:, :, :3].copy()
        depth_arr = _to_np(raw_depth, dtype=np.float32) if raw_depth is not None else np.array([], dtype=np.float32)
        if depth_arr.size != view.height * view.width:
            depth = np.full((view.height, view.width), np.inf, dtype=np.float32)
        else:
            depth = depth_arr.reshape(view.height, view.width)
            depth = np.where((depth > 1e-4) & (depth < 40.0) & np.isfinite(depth), depth, np.inf)
        mask = np.isfinite(depth)
        rendered.append(RenderedView(view=view, rgb=rgb, depth=depth, mask=mask))

    rgb_ann.detach()
    depth_ann.detach()
    if hasattr(rp, "destroy"):
        rp.destroy()
    return rendered


# ---------------------------------------------------------------------------
# Multi-view bundle I/O
# ---------------------------------------------------------------------------


def save_rendered_views(rendered_views: Sequence[RenderedView], output_dir: Path) -> Path:
    """Save RGB PNGs, float32 metric depth `.npy`, mask PNGs, and `cameras.json`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for rv in rendered_views:
        vid = rv.view.view_id
        rgb_name = f"{vid}_rgb.png"
        depth_name = f"{vid}_depth.npy"
        mask_name = f"{vid}_mask.png"
        Image.fromarray(rv.rgb).save(out / rgb_name)
        np.save(out / depth_name, rv.depth.astype(np.float32))
        Image.fromarray(rv.mask.astype(np.uint8) * 255).save(out / mask_name)
        entry = rv.view.to_dict()
        entry.update({"rgb": rgb_name, "depth": depth_name, "mask": mask_name})
        meta.append(entry)
    cam_path = out / "cameras.json"
    cam_path.write_text(json.dumps({"views": meta}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cam_path


def load_rendered_views(views_dir: Path) -> list[RenderedView]:
    """Load a previously saved multi-view bundle from `views_dir`."""
    vdir = Path(views_dir)
    cam_doc = json.loads((vdir / "cameras.json").read_text(encoding="utf-8"))
    out: list[RenderedView] = []
    for entry in cam_doc.get("views", []):
        view = CameraView.from_dict(entry)
        rgb = np.asarray(Image.open(vdir / entry["rgb"]).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(np.load(vdir / entry["depth"]), dtype=np.float32)
        mask_path = vdir / entry.get("mask", "")
        if entry.get("mask") and mask_path.is_file():
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        else:
            mask = np.isfinite(depth) & (depth > 0.0)
        out.append(RenderedView(view=view, rgb=rgb, depth=depth, mask=mask))
    return out


# ---------------------------------------------------------------------------
# 2D Mask / Region Back-Projection onto 3D Points & Grasps
# ---------------------------------------------------------------------------


def _sample_label_at_pixel(
    label_map: np.ndarray,
    depth_map: np.ndarray,
    u_int: int,
    v_int: int,
    z_cam: float,
    depth_tol: float,
    id_to_tag: Mapping[int, str] | None,
) -> str | None:
    """Read a semantic tag at (u_int, v_int) with 3x3 boundary fallback and depth visibility check."""
    h, w = label_map.shape[:2]

    def _decode(val: Any) -> str | None:
        if id_to_tag is not None:
            iv = int(val)
            if iv <= 0:
                return None
            raw = id_to_tag.get(iv)
        else:
            raw = val
        tag = normalize_part_tag(raw)
        return None if tag == "unknown" else tag

    d0 = float(depth_map[v_int, u_int])
    if math.isfinite(d0) and d0 > 0.0 and abs(z_cam - d0) <= depth_tol:
        tag = _decode(label_map[v_int, u_int])
        if tag is not None:
            return tag

    # 3x3 neighborhood fallback for grazing silhouette edges (still requires depth consistency)
    for dv in (-1, 0, 1):
        vv = v_int + dv
        if vv < 0 or vv >= h:
            continue
        for du in (-1, 0, 1):
            if dv == 0 and du == 0:
                continue
            uu = u_int + du
            if uu < 0 or uu >= w:
                continue
            dn = float(depth_map[vv, uu])
            if math.isfinite(dn) and dn > 0.0 and abs(z_cam - dn) <= depth_tol:
                tag = _decode(label_map[vv, uu])
                if tag is not None:
                    return tag
    return None


def backproject_2d_masks_to_points(
    points_3d: np.ndarray,
    rendered_views: Sequence[RenderedView],
    label_maps: Sequence[np.ndarray],
    *,
    id_to_tag: Mapping[int, str] | None = None,
    view_weights: Sequence[float] | None = None,
    depth_tol: float = 0.015,
    min_votes: int = 1,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Back-project multi-view 2D semantic label maps onto (N, 3) 3D points with Z-buffer visibility.

    Returns ``(tags, confidences, vote_counts)`` where:
      - ``tags``: list[str] of length N (`"unknown"` if fewer than `min_votes` visible views matched)
      - ``confidences``: (N,) float64 in [0, 1] (winning weight / total weight)
      - ``vote_counts``: (N,) int32 number of views that contributed a valid semantic vote
    """
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    n_pts = len(pts)
    if n_pts == 0 or not rendered_views:
        return [], np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)
    if len(rendered_views) != len(label_maps):
        raise ValueError(f"Mismatched rendered_views ({len(rendered_views)}) vs label_maps ({len(label_maps)})")

    weights = [float(w) for w in view_weights] if view_weights is not None else [1.0] * len(rendered_views)
    vote_buckets: list[dict[str, float]] = [{} for _ in range(n_pts)]
    vote_counts = np.zeros(n_pts, dtype=np.int32)

    for rv, lmap, vw in zip(rendered_views, label_maps, weights):
        if vw <= 0.0:
            continue
        u, v, z = rv.view.project(pts)
        # Pixel (ui, vi) is rasterized at center (ui + 0.5, vi + 0.5) in render_mesh_view
        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
        in_bounds = (z > 1e-4) & (ui >= 0) & (ui < rv.view.width) & (vi >= 0) & (vi < rv.view.height)
        for idx in np.flatnonzero(in_bounds):
            tag = _sample_label_at_pixel(
                lmap,
                rv.depth,
                int(ui[idx]),
                int(vi[idx]),
                float(z[idx]),
                depth_tol,
                id_to_tag,
            )
            if tag is not None:
                bucket = vote_buckets[idx]
                bucket[tag] = bucket.get(tag, 0.0) + vw
                vote_counts[idx] += 1

    tags: list[str] = []
    confidences = np.zeros(n_pts, dtype=np.float64)
    for idx in range(n_pts):
        bucket = vote_buckets[idx]
        if not bucket or int(vote_counts[idx]) < int(min_votes):
            tags.append("unknown")
            confidences[idx] = 0.0
        else:
            total = sum(bucket.values())
            best_tag, best_w = max(bucket.items(), key=lambda kv: (kv[1], kv[0]))
            tags.append(best_tag)
            confidences[idx] = best_w / max(total, 1e-9)
    return tags, confidences, vote_counts


def rasterize_regions_to_label_map(
    regions: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    """Rasterize 2D normalized bounding boxes or polygons (from VLM) onto the foreground mask.

    Each region dict accepts:
      - ``part`` / ``label``: semantic part name
      - ``bbox_norm``: ``[x1, y1, x2, y2]`` in ``[0, 1]`` (or ``polygon_norm``: list of ``[x, y]``)
      - ``confidence`` (optional, float): higher-confidence / smaller regions take priority
    """
    h, w = mask.shape[:2]
    label_map = np.zeros((h, w), dtype=np.int32)
    tag_to_id: dict[str, int] = {}
    id_to_tag: dict[int, str] = {}

    # Sort larger regions first so smaller fine parts (e.g. handle, spout, rim) paint on top of body
    scored: list[tuple[float, float, Mapping[str, Any]]] = []
    for reg in regions:
        area = 1.0
        if "polygon_norm" in reg and len(reg["polygon_norm"]) >= 3:
            pts_poly = [(float(pt[0]), float(pt[1])) for pt in reg["polygon_norm"]]
            shoelace = sum(
                pts_poly[i][0] * pts_poly[(i + 1) % len(pts_poly)][1]
                - pts_poly[(i + 1) % len(pts_poly)][0] * pts_poly[i][1]
                for i in range(len(pts_poly))
            )
            area = 0.5 * abs(shoelace)
        elif "bbox_norm" in reg and len(reg["bbox_norm"]) == 4:
            x1, y1, x2, y2 = [float(v) for v in reg["bbox_norm"]]
            area = abs(x2 - x1) * abs(y2 - y1)
        conf = float(reg.get("confidence", 0.5))
        scored.append((-area, conf, reg))
    scored.sort(key=lambda item: (item[0], item[1]))

    for _, _, reg in scored:
        tag = normalize_part_tag(reg.get("part") or reg.get("label"))
        if tag == "unknown":
            continue
        if tag not in tag_to_id:
            tid = len(tag_to_id) + 1
            tag_to_id[tag] = tid
            id_to_tag[tid] = tag
        tid = tag_to_id[tag]

        reg_mask = np.zeros((h, w), dtype=bool)
        if "polygon_norm" in reg and len(reg["polygon_norm"]) >= 3:
            img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(img)
            poly = [(float(pt[0]) * w, float(pt[1]) * h) for pt in reg["polygon_norm"]]
            draw.polygon(poly, fill=1)
            reg_mask = np.asarray(img, dtype=np.uint8) > 0
        elif "bbox_norm" in reg and len(reg["bbox_norm"]) == 4:
            x1, y1, x2, y2 = [float(v) for v in reg["bbox_norm"]]
            ix1 = max(0, min(w, int(round(min(x1, x2) * w))))
            ix2 = max(0, min(w, int(round(max(x1, x2) * w))))
            iy1 = max(0, min(h, int(round(min(y1, y2) * h))))
            iy2 = max(0, min(h, int(round(max(y1, y2) * h))))
            if ix2 > ix1 and iy2 > iy1:
                reg_mask[iy1:iy2, ix1:ix2] = True

        label_map[reg_mask & mask] = tid

    return label_map, id_to_tag


# ---------------------------------------------------------------------------
# Grasp visibility & marker overlay rendering
# ---------------------------------------------------------------------------


def visible_grasps_in_view(
    grasps: Sequence[Mapping[str, Any]],
    rv: RenderedView,
    *,
    depth_tol: float = 0.020,
) -> list[dict[str, Any]]:
    """Return grasps whose center or contact points are visible (unoccluded) in `rv`."""
    visible: list[dict[str, Any]] = []
    for idx, g in enumerate(grasps):
        pos = np.asarray(g["position"], dtype=np.float64).reshape(3)
        closing = np.asarray(g.get("closing_axis", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
        c_norm = float(np.linalg.norm(closing))
        closing = closing / c_norm if c_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        half_w = 0.5 * float(g.get("width", 0.02))
        probes = np.vstack([pos - half_w * closing, pos + half_w * closing, pos])

        u, v, z = rv.view.project(probes)
        best = None
        for k in range(3):
            ui, vi = int(math.floor(float(u[k]))), int(math.floor(float(v[k])))
            if z[k] <= 1e-4 or ui < 0 or ui >= rv.view.width or vi < 0 or vi >= rv.view.height:
                continue
            d_buf = float(rv.depth[vi, ui])
            if math.isfinite(d_buf) and d_buf > 0.0 and abs(float(z[k]) - d_buf) <= depth_tol:
                best = (ui, vi, float(z[k]))
                break
        if best is not None:
            visible.append(
                {
                    "index": idx,
                    "id": str(g.get("id", f"g{idx:03d}")),
                    "u": best[0],
                    "v": best[1],
                    "depth": best[2],
                }
            )
    return visible


def render_grasp_markers_image(
    rv: RenderedView,
    visible_grasps: Sequence[Mapping[str, Any]],
    *,
    radius: int = 8,
) -> Image.Image:
    """Draw numbered grasp markers onto a copy of `rv.rgb` for VLM visual prompting."""
    img = Image.fromarray(rv.rgb.copy())
    draw = ImageDraw.Draw(img)
    for item in visible_grasps:
        u, v = int(item["u"]), int(item["v"])
        label = str(item["index"])
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(230, 45, 45), outline=(255, 255, 255))
        draw.text((u - 3 * len(label), v - 5), label, fill=(255, 255, 255))
    return img


# ---------------------------------------------------------------------------
# Assign semantic tags to affordance.json grasps
# ---------------------------------------------------------------------------


def assign_semantic_tags_to_grasps(
    affordance_data: dict[str, Any],
    rendered_views: Sequence[RenderedView] = (),
    *,
    label_maps: Sequence[np.ndarray] | None = None,
    id_to_tag: Mapping[int, str] | None = None,
    surface_points: np.ndarray | None = None,
    surface_tags: Sequence[str] | None = None,
    grasp_marker_votes: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    gripper_types: Sequence[str] | None = None,
    verified_only: bool = False,
    depth_tol: float = 0.020,
    knn_radius: float = 0.025,
    knn_k: int = 7,
    model_name: str | None = None,
    method: str = "multiview_backprojection",
) -> dict[str, Any]:
    """Update grasp `part_tag` and `tag_source: "vlm"` in `affordance_data` (returns a new dict).

    Combines (when supplied):
      1. Direct multi-view 2D mask back-projection at each grasp's contact points (`pos +/- 0.5*w*closing`)
         and grasp center, gated by Z-buffer depth consistency.
      2. Nearest-neighbor voting from back-projected 3D labeled surface points within `knn_radius`.
      3. Direct VLM grasp-marker multi-view votes (`grasp_marker_votes[grasp_id]`).

    If no valid semantic vote is obtained for a grasp, its existing `part_tag` and `tag_source`
    are preserved unchanged (never overwritten with `"unknown"`).
    """
    updated = json.loads(json.dumps(affordance_data))
    grasps_by_gripper = updated.get("grasps") or {}
    target_grippers = list(gripper_types) if gripper_types is not None else list(grasps_by_gripper.keys())

    # Build optional 3D surface kNN index (excluding "unknown" points)
    surf_tree: cKDTree | None = None
    valid_surf_tags: list[str] = []
    if surface_points is not None and surface_tags is not None and len(surface_points) == len(surface_tags):
        keep_pts = []
        for pt, tg in zip(surface_points, surface_tags):
            canon = normalize_part_tag(tg)
            if canon != "unknown":
                keep_pts.append(pt)
                valid_surf_tags.append(canon)
        if keep_pts:
            surf_tree = cKDTree(np.asarray(keep_pts, dtype=np.float64))

    total_updated = 0
    total_considered = 0
    part_counts: dict[str, int] = {}

    for gtype in target_grippers:
        glist = grasps_by_gripper.get(gtype)
        if not isinstance(glist, list):
            continue
        for g in glist:
            if verified_only and not bool(g.get("verified", False)):
                continue
            total_considered += 1
            gid = str(g.get("id", ""))
            votes: dict[str, float] = {}
            n_views_hit = 0

            pos = np.asarray(g["position"], dtype=np.float64).reshape(3)
            closing = np.asarray(g.get("closing_axis", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
            c_norm = float(np.linalg.norm(closing))
            closing = closing / c_norm if c_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
            half_w = 0.5 * float(g.get("width", 0.02))
            probes = np.vstack([pos - half_w * closing, pos + half_w * closing, pos])

            # 1. Direct multi-view 2D mask back-projection at contact points + center
            if label_maps is not None and rendered_views:
                p_tags, p_conf, p_cnts = backproject_2d_masks_to_points(
                    probes,
                    rendered_views,
                    label_maps,
                    id_to_tag=id_to_tag,
                    depth_tol=depth_tol,
                    min_votes=1,
                )
                probe_weights = (1.0, 1.0, 0.6)
                for pt_tag, pt_conf, pt_cnt, pw in zip(p_tags, p_conf, p_cnts, probe_weights):
                    if pt_tag != "unknown" and int(pt_cnt) > 0:
                        votes[pt_tag] = votes.get(pt_tag, 0.0) + pw * float(pt_conf) * float(pt_cnt)
                        n_views_hit = max(n_views_hit, int(pt_cnt))

            # 2. 3D surface kNN voting around contact points + center
            if surf_tree is not None:
                rad = max(float(knn_radius), half_w + 0.008)
                k_eff = min(int(knn_k), len(valid_surf_tags))
                for probe in probes:
                    dists, idxs = surf_tree.query(probe, k=k_eff, distance_upper_bound=rad)
                    dists_arr = np.atleast_1d(dists)
                    idxs_arr = np.atleast_1d(idxs)
                    for d_val, ii in zip(dists_arr, idxs_arr):
                        if not math.isfinite(float(d_val)) or int(ii) >= len(valid_surf_tags):
                            continue
                        tg = valid_surf_tags[int(ii)]
                        w_knn = 0.5 / (1.0 + float(d_val) / max(rad, 1e-6))
                        votes[tg] = votes.get(tg, 0.0) + w_knn

            # 3. Direct VLM grasp-marker votes (prefer gripper-qualified key `gtype:gid`, fallback to `gid`)
            if grasp_marker_votes:
                comp_key = f"{gtype}:{gid}"
                mv_list = grasp_marker_votes.get(comp_key)
                if mv_list is None:
                    mv_list = grasp_marker_votes.get(gid)
                if mv_list:
                    for raw_tag, conf in mv_list:
                        canon = normalize_part_tag(raw_tag)
                        if canon != "unknown" and float(conf) > 0.0:
                            votes[canon] = votes.get(canon, 0.0) + float(conf)
                            n_views_hit += 1

            if votes:
                total_w = sum(votes.values())
                best_tag, best_w = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
                if "part_tag_geom" not in g:
                    g["part_tag_geom"] = str(g.get("part_tag", "unknown"))
                g["part_tag"] = best_tag
                g["tag_source"] = "vlm"
                g["part_tag_confidence"] = round(float(best_w / max(total_w, 1e-9)), 4)
                g["part_tag_views"] = int(max(n_views_hit, 1))
                total_updated += 1
                part_counts[best_tag] = part_counts.get(best_tag, 0) + 1

    updated["semantic_annotation"] = {
        "method": method,
        "model": model_name,
        "num_views": len(rendered_views),
        "verified_only": bool(verified_only),
        "grasps_considered": int(total_considered),
        "grasps_updated": int(total_updated),
        "part_counts": part_counts,
    }
    return updated


# ---------------------------------------------------------------------------
# OpenAI-compatible VLM client (stdlib only, no extra dependencies)
# ---------------------------------------------------------------------------


def _extract_json_block(text: str) -> Any:
    """Parse JSON from a VLM response, stripping optional markdown code fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1).strip()
    return json.loads(cleaned)


def _encode_png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class OpenAICompatibleVLMLabeler:
    """Multimodal VLM client supporting any OpenAI-compatible `/chat/completions` API."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
    ):
        base = (
            api_url
            or os.environ.get("OPENFLYWHEEL_VLM_API_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.api_key = api_key or os.environ.get("OPENFLYWHEEL_VLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("OPENFLYWHEEL_VLM_MODEL") or "gpt-4o-mini"
        self.timeout = float(timeout)

    def _post_chat(self, messages: list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing VLM API key: set OPENFLYWHEEL_VLM_API_KEY or OPENAI_API_KEY")
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.0,
                "messages": messages,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"VLM HTTP {exc.code}: {detail}") from exc
        return str(body["choices"][0]["message"]["content"])

    def predict_part_regions(self, rv: RenderedView, *, object_hint: str = "") -> list[dict[str, Any]]:
        """Ask the VLM for 2D normalized bounding boxes of semantic parts in one view."""
        allowed = ", ".join(t for t in SEMANTIC_PART_TAGS if t != "unknown")
        prompt = (
            f"You are annotating 3D robotic manipulation parts for an object ({object_hint or 'rigid object'}). "
            f"Identify visible semantic parts from this allowed vocabulary: [{allowed}]. "
            "Return ONLY a JSON object: "
            '{"parts": [{"part": "<tag>", "bbox_norm": [x1, y1, x2, y2], "confidence": 0.0-1.0}]} '
            "where coordinates are normalized in [0, 1] relative to image width/height."
        )
        data_url = _encode_png_data_url(Image.fromarray(rv.rgb))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        parsed = _extract_json_block(self._post_chat(messages))
        parts = parsed.get("parts", []) if isinstance(parsed, dict) else parsed
        return [p for p in parts if isinstance(p, dict)]

    def classify_grasp_markers(
        self,
        rv: RenderedView,
        visible_grasps: Sequence[Mapping[str, Any]],
        *,
        object_hint: str = "",
    ) -> dict[str, tuple[str, float]]:
        """Prompt the VLM with numbered grasp markers overlaid on `rv` and return `{grasp_id: (tag, conf)}`."""
        if not visible_grasps:
            return {}
        overlay = render_grasp_markers_image(rv, visible_grasps)
        allowed = ", ".join(t for t in SEMANTIC_PART_TAGS if t != "unknown")
        idx_to_id = {str(item["index"]): str(item["id"]) for item in visible_grasps}
        indices_str = ", ".join(idx_to_id.keys())
        prompt = (
            f"This image shows a rendered view of a 3D object ({object_hint or 'rigid object'}) "
            f"with red numbered grasp markers ({indices_str}). "
            f"For each numbered marker, classify which semantic part it lies on using ONLY: [{allowed}, unknown]. "
            'Return ONLY JSON: {"markers": {"<index>": {"part": "<tag>", "confidence": 0.0-1.0}}}.'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_png_data_url(overlay)}},
                ],
            }
        ]
        parsed = _extract_json_block(self._post_chat(messages))
        markers = parsed.get("markers", parsed) if isinstance(parsed, dict) else {}
        out: dict[str, tuple[str, float]] = {}
        if isinstance(markers, dict):
            for idx_str, val in markers.items():
                gid = idx_to_id.get(str(idx_str))
                if gid is None:
                    continue
                if isinstance(val, dict):
                    tag = normalize_part_tag(val.get("part") or val.get("label"))
                    conf = float(val.get("confidence", 0.8))
                else:
                    tag = normalize_part_tag(val)
                    conf = 0.8
                if tag != "unknown" and conf > 0.0:
                    out[gid] = (tag, conf)
        return out


# ---------------------------------------------------------------------------
# Loading external 2D segmentation masks (e.g. Grounded-SAM / PartSlip)
# ---------------------------------------------------------------------------


def load_2d_label_maps_from_dir(
    masks_dir: Path,
    rendered_views: Sequence[RenderedView],
) -> tuple[list[np.ndarray], dict[int, str] | None]:
    """Load per-view 2D segmentation masks from `masks_dir`.

    Supports two formats:
      1. `<view_id>_labels.npy` (int HxW) + `id_to_tag.json` (`{"1": "handle", "2": "body"}`).
      2. `<view_id>_regions.json` (`{"parts": [{"part": "handle", "bbox_norm": [...]}]}`),
         rasterized onto each view's foreground mask.
    """
    mdir = Path(masks_dir)
    id_map_path = mdir / "id_to_tag.json"
    if id_map_path.is_file():
        raw_map = json.loads(id_map_path.read_text(encoding="utf-8"))
        id_to_tag = {int(k): normalize_part_tag(v) for k, v in raw_map.items()}
        maps = []
        for rv in rendered_views:
            npy_path = mdir / f"{rv.view.view_id}_labels.npy"
            if npy_path.is_file():
                maps.append(np.asarray(np.load(npy_path), dtype=np.int32))
            else:
                maps.append(np.zeros((rv.view.height, rv.view.width), dtype=np.int32))
        return maps, id_to_tag

    # Fallback: per-view region JSON files
    tag_to_id: dict[str, int] = {}
    id_to_tag_all: dict[int, str] = {}
    maps = []
    for rv in rendered_views:
        reg_path = mdir / f"{rv.view.view_id}_regions.json"
        if not reg_path.is_file():
            maps.append(np.zeros((rv.view.height, rv.view.width), dtype=np.int32))
            continue
        doc = json.loads(reg_path.read_text(encoding="utf-8"))
        regions = doc.get("parts", doc) if isinstance(doc, dict) else doc
        lmap_local, local_id_to_tag = rasterize_regions_to_label_map(regions, rv.mask)
        remapped = np.zeros_like(lmap_local)
        for lid, tag in local_id_to_tag.items():
            if tag not in tag_to_id:
                gid = len(tag_to_id) + 1
                tag_to_id[tag] = gid
                id_to_tag_all[gid] = tag
            remapped[lmap_local == lid] = tag_to_id[tag]
        maps.append(remapped)
    return maps, id_to_tag_all


# ---------------------------------------------------------------------------
# High-level Stage 3 orchestration
# ---------------------------------------------------------------------------


def annotate_affordance_with_vlm(
    affordance_data: dict[str, Any],
    *,
    mesh_path: Path | None = None,
    rendered_views: Sequence[RenderedView] | None = None,
    masks_dir: Path | None = None,
    vlm_labeler: OpenAICompatibleVLMLabeler | None = None,
    vlm_mode: str = "markers",
    verified_only: bool = False,
    n_azimuth: int = 6,
    elevations_deg: Sequence[float] = (20.0, 50.0),
    image_size: tuple[int, int] = (256, 256),
    surface_samples: int = 1500,
    seed: int = 0,
) -> dict[str, Any]:
    """Run Stage 3 semantic part annotation on an affordance dict."""
    vertices = np.zeros((0, 3), dtype=np.float64)
    faces = np.zeros((0, 3), dtype=np.int64)
    if mesh_path is not None:
        resolved = resolve_mesh_path(Path(mesh_path))
        vertices, faces = load_mesh(resolved)

    views_list: list[RenderedView]
    if rendered_views is not None:
        views_list = list(rendered_views)
    else:
        bounds = affordance_data.get("bounds") or {}
        if "min" in bounds and "max" in bounds:
            b_min, b_max = bounds["min"], bounds["max"]
        elif len(vertices) > 0:
            b_min, b_max = vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()
        else:
            raise ValueError("Either rendered_views, mesh_path, or affordance bounds must be provided")
        cams = build_multiview_cameras(
            b_min,
            b_max,
            n_azimuth=n_azimuth,
            elevations_deg=elevations_deg,
            image_size=image_size,
        )
        views_list = render_mesh_views(vertices, faces, cams)

    object_hint = str(affordance_data.get("asset_id", ""))
    label_maps: list[np.ndarray] | None = None
    id_to_tag: dict[int, str] | None = None
    marker_votes: dict[str, list[tuple[str, float]]] | None = None
    model_name: str | None = None
    method = "multiview_backprojection"

    if masks_dir is not None:
        label_maps, id_to_tag = load_2d_label_maps_from_dir(Path(masks_dir), views_list)
        method = "2d_mask_backprojection"
    elif vlm_labeler is not None:
        model_name = vlm_labeler.model
        if vlm_mode == "regions":
            method = "vlm_region_backprojection"
            tag_to_id: dict[str, int] = {}
            id_to_tag = {}
            label_maps = []
            for rv in views_list:
                regions = vlm_labeler.predict_part_regions(rv, object_hint=object_hint)
                local_map, local_map_ids = rasterize_regions_to_label_map(regions, rv.mask)
                remapped = np.zeros_like(local_map)
                for lid, tg in local_map_ids.items():
                    if tg not in tag_to_id:
                        gid = len(tag_to_id) + 1
                        tag_to_id[tg] = gid
                        id_to_tag[gid] = tg
                    remapped[local_map == lid] = tag_to_id[tg]
                label_maps.append(remapped)
        else:
            method = "vlm_multiview_markers"
            marker_votes = {}
            all_grasps = []
            for gtype, glist in (affordance_data.get("grasps") or {}).items():
                if not isinstance(glist, list):
                    continue
                for idx_g, g in enumerate(glist):
                    if verified_only and not bool(g.get("verified", False)):
                        continue
                    gid = str(g.get("id", f"g{idx_g:03d}"))
                    g_tagged = dict(g)
                    g_tagged["id"] = f"{gtype}:{gid}"
                    all_grasps.append(g_tagged)
            for rv in views_list:
                vis = visible_grasps_in_view(all_grasps, rv)
                if not vis:
                    continue
                preds = vlm_labeler.classify_grasp_markers(rv, vis, object_hint=object_hint)
                for comp_id, vote in preds.items():
                    marker_votes.setdefault(comp_id, []).append(vote)

    surface_pts = None
    surface_tags = None
    if label_maps is not None and len(vertices) > 0 and len(faces) > 0 and surface_samples > 0:
        rng = np.random.default_rng(seed)
        tris, tri_normals, areas, _ = triangle_geometry(vertices, faces)
        if len(tris) > 0:
            surface_pts, _, _ = sample_surface_points(tris, tri_normals, areas, int(surface_samples), rng)
            surface_tags, _, _ = backproject_2d_masks_to_points(
                surface_pts,
                views_list,
                label_maps,
                id_to_tag=id_to_tag,
            )

    return assign_semantic_tags_to_grasps(
        affordance_data,
        views_list,
        label_maps=label_maps,
        id_to_tag=id_to_tag,
        surface_points=surface_pts,
        surface_tags=surface_tags,
        grasp_marker_votes=marker_votes,
        verified_only=verified_only,
        model_name=model_name,
        method=method,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenFlywheel Stage 3 VLM / 3D part-level semantic annotator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ren = sub.add_parser("render", help="Render multi-view RGB + metric depth + cameras.json")
    p_ren.add_argument("path", type=Path, help="Mesh file (.obj/.stl/.usd[a|z]) or asset directory")
    p_ren.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory for views bundle")
    p_ren.add_argument("--azimuths", type=int, default=6, help="Number of azimuth angles per elevation ring")
    p_ren.add_argument("--image-size", type=int, default=256, help="Square image resolution (pixels)")
    p_ren.add_argument("--isaac", action="store_true", help="Render with Isaac Sim RTX instead of CPU Z-buffer")
    p_ren.add_argument("--device", default="cuda:0", help="Isaac Sim device when --isaac is set")

    p_ann = sub.add_parser("annotate", help="Run Stage 3 semantic part tagging on an affordance.json")
    p_ann.add_argument("affordance", type=Path, help="Input affordance.json path")
    p_ann.add_argument("--mesh", type=Path, default=None, help="Mesh file or asset directory")
    p_ann.add_argument("--views-dir", type=Path, default=None, help="Pre-rendered views directory")
    p_ann.add_argument("--masks-dir", type=Path, default=None, help="Precomputed 2D segmentation masks directory")
    p_ann.add_argument("--backend", choices=("masks", "openai"), default="masks")
    p_ann.add_argument("--vlm-mode", choices=("markers", "regions"), default="markers")
    p_ann.add_argument("--model", default=None, help="Override VLM model name (OPENFLYWHEEL_VLM_MODEL)")
    p_ann.add_argument("--verified-only", action="store_true", help="Only update Stage-2 verified grasps")
    p_ann.add_argument("--output", "-o", type=Path, default=None, help="Output JSON (default: overwrite input)")

    p_bat = sub.add_parser("batch", help="Batch Stage 3 semantic tagging over a Stage 1/2 manifest.json")
    p_bat.add_argument("--manifest", type=Path, required=True, help="Stage 1/2 manifest.json path")
    p_bat.add_argument("--backend", choices=("masks", "openai"), default="openai")
    p_bat.add_argument(
        "--masks-root",
        type=Path,
        default=None,
        help="Root directory of per-asset 2D masks (required when --backend masks; looks up <masks-root>/<asset_id>)",
    )
    p_bat.add_argument("--vlm-mode", choices=("markers", "regions"), default="markers")
    p_bat.add_argument("--model", default=None)
    p_bat.add_argument("--verified-only", action="store_true")
    p_bat.add_argument("--limit", type=int, default=0)

    args = parser.parse_args(argv)

    try:
        if args.cmd == "render":
            mesh_path = resolve_mesh_path(args.path)
            verts, faces = load_mesh(mesh_path)
            if len(verts) == 0:
                raise ValueError(f"Empty mesh: {mesh_path}")
            cams = build_multiview_cameras(
                verts.min(axis=0),
                verts.max(axis=0),
                n_azimuth=args.azimuths,
                image_size=(args.image_size, args.image_size),
            )
            if args.isaac:
                asset_dir = args.path if args.path.is_dir() else args.path.parent
                views = render_isaac_views(asset_dir, cams, device=args.device)
            else:
                views = render_mesh_views(verts, faces, cams)
            cam_path = save_rendered_views(views, args.output_dir)
            print(json.dumps({"views": len(views), "cameras_json": str(cam_path)}, indent=2), flush=True)
            if args.isaac:
                os._exit(0)  # app.close() hangs in Isaac Sim 5.1 once a camera render product exists
            return 0

        if args.cmd == "annotate":
            if args.backend == "masks" and args.masks_dir is None:
                parser.error("annotate with --backend masks requires --masks-dir")
            data = json.loads(args.affordance.read_text(encoding="utf-8"))
            rendered = load_rendered_views(args.views_dir) if args.views_dir else None
            labeler = OpenAICompatibleVLMLabeler(model=args.model) if args.backend == "openai" else None
            out_data = annotate_affordance_with_vlm(
                data,
                mesh_path=args.mesh,
                rendered_views=rendered,
                masks_dir=args.masks_dir,
                vlm_labeler=labeler,
                vlm_mode=args.vlm_mode,
                verified_only=args.verified_only,
            )
            out_path = args.output or args.affordance
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(out_data["semantic_annotation"], indent=2, ensure_ascii=False))
            return 0

        if args.cmd == "batch":
            if args.backend == "masks" and args.masks_root is None:
                parser.error("batch with --backend masks requires --masks-root")
            entries = json.loads(args.manifest.read_text(encoding="utf-8"))
            if args.limit > 0:
                entries = entries[: args.limit]
            labeler = OpenAICompatibleVLMLabeler(model=args.model) if args.backend == "openai" else None
            ok, err = 0, 0
            for entry in entries:
                if entry.get("status") not in ("ok", "skipped"):
                    continue
                aff_path = Path(entry["affordance_path"])
                asset_dir = Path(entry["asset_dir"]) if entry.get("asset_dir") else None
                if not aff_path.is_file():
                    continue
                entry_masks_dir: Path | None = None
                if args.masks_root is not None:
                    aid = str(entry.get("asset_id", ""))
                    cand_a = args.masks_root / aid
                    cand_b = args.masks_root / aid.replace("/", "_")
                    if cand_a.is_dir():
                        entry_masks_dir = cand_a
                    elif cand_b.is_dir():
                        entry_masks_dir = cand_b
                    else:
                        print(f"[affordance_vlm batch] ERROR {aid}: masks dir not found under {args.masks_root}")
                        err += 1
                        continue
                try:
                    data = json.loads(aff_path.read_text(encoding="utf-8"))
                    out_data = annotate_affordance_with_vlm(
                        data,
                        mesh_path=asset_dir,
                        masks_dir=entry_masks_dir,
                        vlm_labeler=labeler,
                        vlm_mode=args.vlm_mode,
                        verified_only=args.verified_only,
                    )
                    aff_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    ok += 1
                except Exception as exc:
                    print(f"[affordance_vlm batch] ERROR {entry.get('asset_id')}: {type(exc).__name__}: {exc}")
                    err += 1
            print(f"[affordance_vlm batch] ok={ok} err={err}")
            return 0 if err == 0 else 2

        return 1
    finally:
        close_pxr_app()


if __name__ == "__main__":
    raise SystemExit(main())
