"""2D affordance projection, heatmap/BBox/keypoint generation, visualization, and VLM/VLA export.

Projects 3D actionable affordance sidecars (`affordance.json`, supporting any gripper/hand key
such as `dex1_1`, `aurora_lite`, `sharpa_wave`, `wuji_hand2_2`) into any calibrated camera view
to produce:
  - 2D Gaussian-splatted graspability heatmaps (float32 [0, 1] & uint8 [0, 255], plus optional
    per-`part_tag` heatmaps) weighted by `sim_score` (falling back to geometric `score`);
  - Tight 2D affordance BBoxes (`affordance_bbox_xyxy` / `affordance_bbox_xyxy_norm`);
  - Projected 3D object bounds BBoxes (`object_bbox_xyxy` / `object_bbox_xyxy_norm`);
  - Visible 2D grasp keypoints (`u, v, u_norm, v_norm, score, part_tag, grasp_id, grasp_type`);
  - Clean publication-grade RGB overlays (`render_affordance_2d_overlay`) in pure NumPy + Pillow.

CLI usage:
    python scripts/objects/affordance_2d.py render affordance.json --mesh object.obj -o /tmp/aff2d
    python scripts/objects/affordance_2d.py render affordance.json --views-dir /tmp/views -o /tmp/aff2d
    python scripts/objects/affordance_2d.py export-vlm --affordance affordance.json --views-dir /tmp/views -o /tmp/vlm
    python scripts/objects/affordance_2d.py export-vlm --hdf5 episode_000000.hdf5 -o /tmp/vlm_datagen
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Camera conventions & projection math
# ---------------------------------------------------------------------------

_USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

_AABB_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def usd_to_opencv_cam_to_world(cam_to_world_usd: np.ndarray) -> np.ndarray:
    """Convert a 4x4 USD/OpenGL cam_to_world (+X right, +Y up, -Z forward) to OpenCV (+X right, +Y down, +Z forward)."""
    mat = np.asarray(cam_to_world_usd, dtype=np.float64).reshape(4, 4)
    return mat @ _USD_TO_OPENCV


def _to_opencv_cam_to_world(cam_to_world: np.ndarray, camera_convention: str) -> np.ndarray:
    mat = np.asarray(cam_to_world, dtype=np.float64).reshape(4, 4)
    conv = str(camera_convention).strip().lower()
    if conv == "opencv":
        return mat
    if conv in ("usd", "opengl"):
        return mat @ _USD_TO_OPENCV
    raise ValueError(f"Unsupported camera_convention={camera_convention!r}; expected 'opencv', 'usd', or 'opengl'")


def _from_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError(f"Zero-norm quaternion: {quat_wxyz!r}")
    q = q / norm
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _normalize_scale(scale: float | Sequence[float], scale_ref: float = 1.0) -> np.ndarray:
    arr = np.asarray(scale, dtype=np.float64)
    vec = np.full(3, float(arr), dtype=np.float64) if arr.ndim == 0 else arr.reshape(3).astype(np.float64)
    if np.any(vec <= 0.0) or not np.isfinite(vec).all():
        raise ValueError(f"Scale must be positive and finite, got {scale!r}")
    ref = float(scale_ref)
    if ref <= 0.0 or not math.isfinite(ref):
        raise ValueError(f"scale_ref must be positive and finite, got {scale_ref!r}")
    return vec / ref


def _load_affordance_dict(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def project_world_points(
    points_world: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    *,
    camera_convention: str = "opencv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N, 3) world points into pixel coordinates (u, v, z_cam)."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty
    c2w = _to_opencv_cam_to_world(cam_to_world, camera_convention)
    k_mat = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]
    pts_cam = (pts - trans[None, :]) @ rot  # R^T (p - t)
    z = pts_cam[:, 2]
    safe_z = np.where(np.abs(z) > 1e-9, z, 1e-9)
    u = float(k_mat[0, 0]) * (pts_cam[:, 0] / safe_z) + float(k_mat[0, 2])
    v = float(k_mat[1, 1]) * (pts_cam[:, 1] / safe_z) + float(k_mat[1, 2])
    return u, v, z


def _symmetry_angles(symmetry: Mapping[str, Any] | None, continuous_samples: int) -> list[float]:
    if not symmetry:
        return [0.0]
    mode = str(symmetry.get("z_axis", "none"))
    if mode == "continuous":
        n = max(1, int(continuous_samples))
        return [2.0 * math.pi * k / n for k in range(n)]
    if mode == "discrete":
        fold = max(1, int(symmetry.get("fold", 1)))
        return [2.0 * math.pi * k / fold for k in range(fold)]
    return [0.0]


def project_object_bbox_2d(
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    obj_pos: Sequence[float],
    obj_quat_wxyz: Sequence[float],
    K: np.ndarray,
    cam_to_world: np.ndarray,
    width: int,
    height: int,
    *,
    scale: float | Sequence[float] = 1.0,
    scale_ref: float = 1.0,
    camera_convention: str = "opencv",
    near_clip: float = 0.02,
) -> tuple[list[float] | None, list[float] | None]:
    """Project a 3D local AABB into a clamped 2D pixel & normalized [x1, y1, x2, y2] BBox.

    Clips AABB edges against ``z_cam >= near_clip`` before perspective division so objects
    straddling the near plane project cleanly without sign inversion.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    b_min = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    b_max = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    scale_vec = _normalize_scale(scale, scale_ref)
    lo = b_min * scale_vec
    hi = b_max * scale_vec
    corners_local = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    p_obj = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    r_obj = _from_wxyz(obj_quat_wxyz)
    corners_world = corners_local @ r_obj.T + p_obj[None, :]

    c2w = _to_opencv_cam_to_world(cam_to_world, camera_convention)
    pts_cam = (corners_world - c2w[:3, 3][None, :]) @ c2w[:3, :3]

    clipped_cam: list[np.ndarray] = [pt for pt in pts_cam if pt[2] >= near_clip]
    for i, j in _AABB_EDGES:
        p0, p1 = pts_cam[i], pts_cam[j]
        in0, in1 = p0[2] >= near_clip, p1[2] >= near_clip
        if in0 != in1:
            t = (near_clip - p0[2]) / (p1[2] - p0[2])
            clipped_cam.append(p0 + t * (p1 - p0))
    if not clipped_cam:
        return None, None

    arr_cam = np.asarray(clipped_cam, dtype=np.float64)
    k_mat = np.asarray(K, dtype=np.float64).reshape(3, 3)
    us = float(k_mat[0, 0]) * (arr_cam[:, 0] / arr_cam[:, 2]) + float(k_mat[0, 2])
    vs = float(k_mat[1, 1]) * (arr_cam[:, 1] / arr_cam[:, 2]) + float(k_mat[1, 2])

    x1 = float(np.clip(np.min(us), 0.0, float(width)))
    y1 = float(np.clip(np.min(vs), 0.0, float(height)))
    x2 = float(np.clip(np.max(us), 0.0, float(width)))
    y2 = float(np.clip(np.max(vs), 0.0, float(height)))
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None, None
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None, None

    bbox_px = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    bbox_norm = [
        round(x1 / width, 6),
        round(y1 / height, 6),
        round(x2 / width, 6),
        round(y2 / height, 6),
    ]
    return bbox_px, bbox_norm


# ---------------------------------------------------------------------------
# Result container & core 3D -> 2D projection
# ---------------------------------------------------------------------------


@dataclass
class Affordance2DResult:
    """2D affordance annotations projected into a single camera view."""

    width: int
    height: int
    heatmap: np.ndarray  # (H_hm, W_hm) float32 in [0, 1]
    heatmap_uint8: np.ndarray  # (H_hm, W_hm) uint8 in [0, 255]
    affordance_bbox_xyxy: list[float] | None
    affordance_bbox_xyxy_norm: list[float] | None
    object_bbox_xyxy: list[float] | None
    object_bbox_xyxy_norm: list[float] | None
    keypoints: list[dict[str, Any]]
    asset_id: str = ""
    gripper_types: tuple[str, ...] = ()
    part_heatmaps: dict[str, np.ndarray] = field(default_factory=dict)
    selected_grasp_2d: dict[str, Any] | None = None

    def resize_heatmap_uint8(self, out_h: int, out_w: int) -> np.ndarray:
        """Resize the heatmap to ``(out_h, out_w)`` uint8 using bilinear resampling."""
        oh, ow = int(out_h), int(out_w)
        if oh <= 0 or ow <= 0:
            raise ValueError(f"Invalid target heatmap size: ({oh}, {ow})")
        if self.heatmap_uint8.shape == (oh, ow):
            return self.heatmap_uint8.copy()
        img = Image.fromarray(self.heatmap_uint8, mode="L")
        return np.asarray(img.resize((ow, oh), resample=Image.Resampling.BILINEAR), dtype=np.uint8)

    def to_dict(self, *, include_heatmap: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "asset_id": self.asset_id,
            "gripper_types": list(self.gripper_types),
            "width": int(self.width),
            "height": int(self.height),
            "heatmap_hw": [int(self.heatmap.shape[0]), int(self.heatmap.shape[1])],
            "object_bbox_xyxy": self.object_bbox_xyxy,
            "object_bbox_xyxy_norm": self.object_bbox_xyxy_norm,
            "affordance_bbox_xyxy": self.affordance_bbox_xyxy,
            "affordance_bbox_xyxy_norm": self.affordance_bbox_xyxy_norm,
            "keypoints": self.keypoints,
            "num_keypoints": len(self.keypoints),
        }
        if self.selected_grasp_2d is not None:
            payload["selected_grasp_2d"] = self.selected_grasp_2d
        if self.part_heatmaps:
            payload["part_tags"] = sorted(self.part_heatmaps)
        if include_heatmap:
            payload["heatmap_uint8"] = self.heatmap_uint8.tolist()
        return payload


def _grasp_score(g: Mapping[str, Any]) -> float:
    sim_score = g.get("sim_score")
    if isinstance(sim_score, int | float) and not isinstance(sim_score, bool) and math.isfinite(float(sim_score)):
        return max(0.0, float(sim_score))
    score = g.get("score", 0.0)
    if isinstance(score, int | float) and not isinstance(score, bool) and math.isfinite(float(score)):
        return max(0.0, float(score))
    return 0.0


def _grasp_type_label(g: Mapping[str, Any], gripper_type: str) -> str:
    if g.get("grasp_type"):
        return str(g["grasp_type"])
    if gripper_type in ("dex1_1", "aurora_lite"):
        return "parallel_jaw"
    return str(gripper_type)


def _extract_closing_axis(g: Mapping[str, Any]) -> np.ndarray | None:
    """Return unit local closing axis from `closing_axis` or `quaternion_wxyz` (+Y_ee), else None."""
    raw_ca = g.get("closing_axis")
    if raw_ca is not None:
        try:
            ca = np.asarray(raw_ca, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(ca))
            if np.isfinite(ca).all() and norm > 1e-6:
                return ca / norm
        except (TypeError, ValueError):
            pass
    raw_q = g.get("quaternion_wxyz")
    if raw_q is not None:
        try:
            rot = _from_wxyz(raw_q)
            ca = rot[:, 1]
            norm = float(np.linalg.norm(ca))
            if np.isfinite(ca).all() and norm > 1e-6:
                return ca / norm
        except (TypeError, ValueError):
            pass
    return None


def _rotate_z_around(pt: np.ndarray, axis_xy: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) <= 1e-9:
        return pt
    c, s = math.cos(angle), math.sin(angle)
    rel = pt - axis_xy
    return np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]], dtype=np.float64) + axis_xy


def _normalize_fg_mask(fg_mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(fg_mask)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    if mask.shape != (height, width):
        raise ValueError(f"fg_mask shape {mask.shape} does not match image size ({height}, {width})")
    return mask.astype(bool)


def _resize_mask_to_hw(mask: np.ndarray, hm_h: int, hm_w: int) -> np.ndarray:
    if mask.shape == (hm_h, hm_w):
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = np.asarray(img.resize((hm_w, hm_h), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
    return resized > 0


def _clip_and_renorm_heatmap(hm: np.ndarray, mask_hm: np.ndarray | None) -> np.ndarray:
    if mask_hm is None:
        return hm
    clipped = np.where(mask_hm, hm, 0.0).astype(np.float32)
    peak = float(clipped.max())
    if peak > 1e-12:
        clipped = np.clip(clipped / peak, 0.0, 1.0).astype(np.float32)
    return clipped


def _splat_gaussians(
    centers_u: np.ndarray,
    centers_v: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    hm_h: int,
    hm_w: int,
    src_w: int,
    src_h: int,
) -> np.ndarray:
    """Vectorized 2D Gaussian splatting onto a ``(hm_h, hm_w)`` float32 canvas normalized to ``[0, 1]``."""
    canvas = np.zeros((hm_h, hm_w), dtype=np.float64)
    if len(centers_u) == 0:
        return canvas.astype(np.float32)

    sx = float(hm_w) / float(src_w)
    sy = float(hm_h) / float(src_h)
    scale_sig = math.sqrt(sx * sy)
    eff_weights = np.asarray(weights, dtype=np.float64)
    if float(np.max(eff_weights)) <= 0.0:
        eff_weights = np.ones_like(eff_weights)

    for u, v, sig, w in zip(centers_u, centers_v, sigmas, eff_weights):
        if w <= 0.0:
            continue
        cu = float(u) * sx
        cv = float(v) * sy
        s = max(float(sig) * scale_sig, 0.75)
        rad = max(2, int(math.ceil(3.0 * s)))
        ic_u = int(round(cu))
        ic_v = int(round(cv))
        x0 = max(0, ic_u - rad)
        x1 = min(hm_w, ic_u + rad + 1)
        y0 = max(0, ic_v - rad)
        y1 = min(hm_h, ic_v + rad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1, dtype=np.float64) + 0.5 - cu
        ys = np.arange(y0, y1, dtype=np.float64) + 0.5 - cv
        inv_2s2 = 0.5 / (s * s)
        patch = np.exp(-((ys[:, None] ** 2) + (xs[None, :] ** 2)) * inv_2s2)
        canvas[y0:y1, x0:x1] += float(w) * patch

    peak = float(canvas.max())
    if peak > 1e-12:
        canvas /= peak
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def project_affordance_2d(
    affordance: Mapping[str, Any] | str | Path,
    obj_pos: Sequence[float],
    obj_quat_wxyz: Sequence[float],
    K: np.ndarray,
    cam_to_world: np.ndarray,
    width: int,
    height: int,
    *,
    scale: float | Sequence[float] = 1.0,
    depth: np.ndarray | None = None,
    fg_mask: np.ndarray | None = None,
    clip_to_foreground: bool = True,
    use_contact_points: bool = True,
    depth_tol: float = 0.03,
    gripper_types: Sequence[str] | None = None,
    part_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    verified_only: bool = False,
    skip_sim_failed: bool = True,
    expand_symmetry: bool = True,
    continuous_sym_samples: int = 8,
    sigma_px: float | None = None,
    min_sigma_px: float = 4.0,
    max_sigma_px: float = 36.0,
    bbox_heatmap_thresh: float = 0.25,
    per_part_heatmaps: bool = False,
    max_keypoints: int | None = None,
    selected_grasp_world: Sequence[float] | None = None,
    camera_convention: str = "opencv",
    heatmap_hw: tuple[int, int] | None = None,
    near_clip: float = 0.02,
) -> Affordance2DResult:
    """Project a 3D `affordance.json` into 2D heatmap, BBoxes, and keypoints for a camera view.

    When ``use_contact_points=True`` (default) and grasps carry ``closing_axis`` or
    ``quaternion_wxyz``, projects the actual opposing finger contact points
    ``position ± (width / 2) * closing_axis`` so heatmaps and Affordance BBoxes sit on the
    real graspable surface walls rather than empty hollow interiors. When ``fg_mask`` or
    ``depth`` is available and ``clip_to_foreground=True`` (default), strictly clips heatmaps
    and the derived Affordance BBox to the object's visible foreground mask.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    data = _load_affordance_dict(affordance)
    asset_id = str(data.get("asset_id", ""))
    scale_ref = float(data.get("scale_ref", 1.0))
    scale_vec = _normalize_scale(scale, scale_ref)
    mean_scale = float(np.mean(scale_vec))

    hm_h, hm_w = (int(heatmap_hw[0]), int(heatmap_hw[1])) if heatmap_hw is not None else (int(height), int(width))
    if hm_h <= 0 or hm_w <= 0:
        raise ValueError(f"Invalid heatmap_hw: {heatmap_hw!r}")

    p_obj = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    r_obj = _from_wxyz(obj_quat_wxyz)
    c2w_cv = _to_opencv_cam_to_world(cam_to_world, camera_convention)

    # 1. Object 2D BBox from local AABB bounds (+ camera-space near/far Z for depth foreground masking)
    bounds = data.get("bounds") or {}
    obj_bbox_px: list[float] | None = None
    obj_bbox_norm: list[float] | None = None
    obj_z_min: float | None = None
    obj_z_max: float | None = None
    if isinstance(bounds, Mapping) and bounds.get("min") is not None and bounds.get("max") is not None:
        obj_bbox_px, obj_bbox_norm = project_object_bbox_2d(
            bounds["min"],
            bounds["max"],
            obj_pos,
            obj_quat_wxyz,
            K,
            cam_to_world,
            width,
            height,
            scale=scale,
            scale_ref=scale_ref,
            camera_convention=camera_convention,
            near_clip=near_clip,
        )
        b_min = np.asarray(bounds["min"], dtype=np.float64).reshape(3) * scale_vec
        b_max = np.asarray(bounds["max"], dtype=np.float64).reshape(3) * scale_vec
        if np.isfinite(b_min).all() and np.isfinite(b_max).all() and np.isfinite(p_obj).all():
            corners_loc = np.array(
                [[x, y, z] for x in (b_min[0], b_max[0]) for y in (b_min[1], b_max[1]) for z in (b_min[2], b_max[2])],
                dtype=np.float64,
            )
            corners_world = corners_loc @ r_obj.T + p_obj[None, :]
            corners_cam = (corners_world - c2w_cv[:3, 3][None, :]) @ c2w_cv[:3, :3]
            fin_z = corners_cam[np.isfinite(corners_cam[:, 2]), 2]
            if len(fin_z) > 0:
                obj_z_min = float(np.min(fin_z))
                obj_z_max = float(np.max(fin_z))

    # 2. Collect candidate 3D grasp entries (with optional Z-axis symmetry expansion)
    grasps_map = data.get("grasps") or {}
    if not isinstance(grasps_map, Mapping):
        grasps_map = {}
    if gripper_types is None:
        selected_grippers = tuple(str(k) for k in grasps_map)
    else:
        selected_grippers = tuple(str(k) for k in gripper_types if k in grasps_map)

    allow_tags = {str(t) for t in part_tags} if part_tags is not None else None
    deny_tags = {str(t) for t in exclude_tags} if exclude_tags is not None else set()

    sym_meta = data.get("symmetry") if expand_symmetry else None
    sym_angles = _symmetry_angles(sym_meta if isinstance(sym_meta, Mapping) else None, continuous_sym_samples)
    raw_axis = sym_meta.get("axis_xy") if isinstance(sym_meta, Mapping) else None
    sym_axis = (
        np.array([float(raw_axis[0]), float(raw_axis[1]), 0.0], dtype=np.float64)
        if raw_axis is not None and len(raw_axis) >= 2
        else np.zeros(3, dtype=np.float64)
    )

    # Validate depth & fg_mask upfront
    depth_arr: np.ndarray | None = None
    if depth is not None:
        depth_arr = np.asarray(depth, dtype=np.float64)
        if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.shape != (height, width):
            raise ValueError(f"depth shape {depth_arr.shape} does not match image size ({height}, {width})")

    fg_mask_explicit: np.ndarray | None = None
    if fg_mask is not None:
        fg_mask_explicit = _normalize_fg_mask(fg_mask, height, width)
    fg_mask_full: np.ndarray | None = fg_mask_explicit if clip_to_foreground else None

    k_mat = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx = abs(float(k_mat[0, 0]))
    z_band_margin = max(0.08, 2.0 * float(depth_tol))

    def _is_point_visible(u: float, v: float, z: float, occ_tol: float, *, grazing_radius: int = 0) -> bool:
        if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(z)):
            return False
        if z <= near_clip or u < 0.0 or u >= float(width) or v < 0.0 or v >= float(height):
            return False
        iu = min(width - 1, max(0, int(round(u))))
        iv = min(height - 1, max(0, int(round(v))))
        if fg_mask_full is not None:
            r_m = max(2, grazing_radius)
            y0, y1 = max(0, iv - r_m), min(height, iv + r_m + 1)
            x0, x1 = max(0, iu - r_m), min(width, iu + r_m + 1)
            if not bool(fg_mask_full[y0:y1, x0:x1].any()):
                return False
        if depth_arr is not None:
            if grazing_radius > 0:
                y0, y1 = max(0, iv - grazing_radius), min(height, iv + grazing_radius + 1)
                x0, x1 = max(0, iu - grazing_radius), min(width, iu + grazing_radius + 1)
                patch_d = depth_arr[y0:y1, x0:x1]
                valid_d = np.isfinite(patch_d) & (patch_d > 0.0)
                if fg_mask_full is not None:
                    valid_d = valid_d & fg_mask_full[y0:y1, x0:x1]
                elif obj_z_max is not None:
                    valid_d = valid_d & (patch_d <= obj_z_max + z_band_margin)
                if bool(valid_d.any()):
                    d_obs = float(np.max(patch_d[valid_d]))
                    if z > d_obs + occ_tol:
                        return False
                else:
                    d_center = float(depth_arr[iv, iu])
                    if math.isfinite(d_center) and d_center > 0.0 and z > d_center + occ_tol:
                        return False
            else:
                d_obs = float(depth_arr[iv, iu])
                if math.isfinite(d_obs) and d_obs > 0.0 and z > d_obs + occ_tol:
                    return False
        return True

    # Build flat list of 3D points to project in one batch, grouped by grasp candidate
    pts_world_all: list[np.ndarray] = []
    # Each entry describes one grasp instance (or symmetry copy):
    # (start_idx, count, is_contact_mode, emit_keypoint, meta_dict)
    grasp_entries: list[tuple[int, int, bool, bool, dict[str, Any]]] = []

    for g_idx_type, gtype in enumerate(selected_grippers):
        glist = grasps_map.get(gtype) or []
        if not isinstance(glist, Sequence):
            continue
        for gi, g in enumerate(glist):
            if not isinstance(g, Mapping) or "position" not in g:
                continue
            tag = str(g.get("part_tag", "unknown"))
            if allow_tags is not None and tag not in allow_tags:
                continue
            if tag in deny_tags:
                continue
            is_verified = bool(g.get("verified", False))
            if verified_only and not is_verified:
                continue
            sim_tested = g.get("sim_score") is not None or g.get("sim_metrics") is not None
            if skip_sim_failed and sim_tested and not is_verified:
                continue

            pos_local = np.asarray(g["position"], dtype=np.float64).reshape(3)
            if not np.isfinite(pos_local).all():
                continue
            score = _grasp_score(g)
            width_local = max(0.005, float(g.get("width", 0.03)))
            width_m = width_local * mean_scale
            gid = str(g.get("id", f"{gtype}_{g_idx_type}_{gi}"))
            g_kind = _grasp_type_label(g, gtype)
            closing_axis = _extract_closing_axis(g) if use_contact_points else None

            rel_xy = math.hypot(float(pos_local[0] - sym_axis[0]), float(pos_local[1] - sym_axis[1]))
            on_axis = rel_xy < 1e-6
            kp_angles = [0.0] if on_axis else sym_angles

            if closing_axis is not None:
                p1_base = pos_local + 0.5 * width_local * closing_axis
                p2_base = pos_local - 0.5 * width_local * closing_axis
                ca_xy = math.hypot(float(closing_axis[0]), float(closing_axis[1]))
                if on_axis and ca_xy > 1e-6 and len(sym_angles) > 1:
                    # Expand contact ring around Z for heatmap while deduplicating symmetric 3D duplicates
                    # (e.g. p1(ang + pi) == p2(ang)) and emitting a single deduplicated keypoint.
                    start = len(pts_world_all)
                    ring_pts: list[np.ndarray] = []
                    for ang in sym_angles:
                        for base_pt in (p1_base, p2_base):
                            pw = r_obj @ (_rotate_z_around(base_pt, sym_axis, ang) * scale_vec) + p_obj
                            if not any(float(np.linalg.norm(pw - prev)) < 1e-5 for prev in ring_pts):
                                ring_pts.append(pw)
                    pts_world_all.extend(ring_pts)
                    meta = {
                        "grasp_id": gid,
                        "base_grasp_id": gid,
                        "is_sym_copy": False,
                        "gripper_type": gtype,
                        "grasp_type": g_kind,
                        "part_tag": tag,
                        "score": score,
                        "verified": is_verified,
                        "width_m": width_m,
                    }
                    grasp_entries.append((start, len(ring_pts), True, True, meta))
                else:
                    for s_idx, ang in enumerate(kp_angles):
                        start = len(pts_world_all)
                        p1_w = r_obj @ (_rotate_z_around(p1_base, sym_axis, ang) * scale_vec) + p_obj
                        p2_w = r_obj @ (_rotate_z_around(p2_base, sym_axis, ang) * scale_vec) + p_obj
                        pts_world_all.append(p1_w)
                        pts_world_all.append(p2_w)
                        meta = {
                            "grasp_id": gid if s_idx == 0 else f"{gid}#sym{s_idx}",
                            "base_grasp_id": gid,
                            "is_sym_copy": s_idx > 0,
                            "gripper_type": gtype,
                            "grasp_type": g_kind,
                            "part_tag": tag,
                            "score": score,
                            "verified": is_verified,
                            "width_m": width_m,
                        }
                        grasp_entries.append((start, 2, True, True, meta))
            else:
                for s_idx, ang in enumerate(kp_angles):
                    start = len(pts_world_all)
                    p_w = r_obj @ (_rotate_z_around(pos_local, sym_axis, ang) * scale_vec) + p_obj
                    pts_world_all.append(p_w)
                    meta = {
                        "grasp_id": gid if s_idx == 0 else f"{gid}#sym{s_idx}",
                        "base_grasp_id": gid,
                        "is_sym_copy": s_idx > 0,
                        "gripper_type": gtype,
                        "grasp_type": g_kind,
                        "part_tag": tag,
                        "score": score,
                        "verified": is_verified,
                        "width_m": width_m,
                    }
                    grasp_entries.append((start, 1, False, True, meta))

    # 3. Project all candidate 3D points & run depth occlusion checks
    splat_u: list[float] = []
    splat_v: list[float] = []
    splat_sig: list[float] = []
    splat_w: list[float] = []
    splat_tag: list[str] = []
    splat_ver: list[bool] = []
    vis_kp: list[dict[str, Any]] = []
    min_vis_z = float("inf")
    max_vis_z = 0.0

    if pts_world_all:
        pts_arr = np.asarray(pts_world_all, dtype=np.float64)
        us, vs, zs = project_world_points(pts_arr, k_mat, cam_to_world, camera_convention=camera_convention)
        for start, count, is_contact, emit_kp, m in grasp_entries:
            width_m = float(m["width_m"])
            if is_contact:
                vis_contacts: list[tuple[float, float, float, float]] = []
                for idx_pt in range(start, start + count):
                    u, v, z = float(us[idx_pt]), float(vs[idx_pt]), float(zs[idx_pt])
                    if not _is_point_visible(u, v, z, float(depth_tol), grazing_radius=2):
                        continue
                    contact_radius_m = float(np.clip(0.22 * width_m, 0.007 * mean_scale, 0.011 * mean_scale))
                    raw_sig = (fx * contact_radius_m / z) if fx > 1e-6 else min_sigma_px
                    sig = (
                        float(sigma_px) if sigma_px is not None else float(np.clip(raw_sig, min_sigma_px, max_sigma_px))
                    )
                    splat_u.append(u)
                    splat_v.append(v)
                    splat_sig.append(sig)
                    splat_w.append(float(m["score"]))
                    splat_tag.append(str(m["part_tag"]))
                    splat_ver.append(bool(m["verified"]))
                    vis_contacts.append((u, v, z, sig))
                    if z < min_vis_z:
                        min_vis_z = z
                    if z > max_vis_z:
                        max_vis_z = z
                if emit_kp and vis_contacts:
                    # Anchor keypoint on the front-most visible finger contact so it sits on the real surface
                    best_u, best_v, best_z, _ = min(vis_contacts, key=lambda c: c[2])
                    width_px = (fx * width_m / best_z) if fx > 1e-6 else float(min_sigma_px * 2.0)
                    vis_kp.append(
                        {
                            "u": round(best_u, 2),
                            "v": round(best_v, 2),
                            "u_norm": round(best_u / float(width), 6),
                            "v_norm": round(best_v / float(height), 6),
                            "score": round(float(m["score"]), 4),
                            "part_tag": m["part_tag"],
                            "grasp_id": m["grasp_id"],
                            "grasp_type": m["grasp_type"],
                            "gripper_type": m["gripper_type"],
                            "verified": bool(m["verified"]),
                            "z_cam": round(best_z, 4),
                            "width_px": round(float(width_px), 2),
                            "contact_points_2d": [[round(cu, 2), round(cv, 2)] for cu, cv, _, _ in vis_contacts],
                        }
                    )
            else:
                u, v, z = float(us[start]), float(vs[start]), float(zs[start])
                occ_tol = float(depth_tol) + 0.5 * width_m
                if not _is_point_visible(u, v, z, occ_tol):
                    continue
                width_px = (fx * width_m / z) if fx > 1e-6 else float(min_sigma_px * 2.0)
                sig = (
                    float(sigma_px)
                    if sigma_px is not None
                    else float(np.clip(0.45 * width_px, min_sigma_px, max_sigma_px))
                )
                splat_u.append(u)
                splat_v.append(v)
                splat_sig.append(sig)
                splat_w.append(float(m["score"]))
                splat_tag.append(str(m["part_tag"]))
                splat_ver.append(bool(m["verified"]))
                if z < min_vis_z:
                    min_vis_z = z
                if z > max_vis_z:
                    max_vis_z = z
                if emit_kp:
                    vis_kp.append(
                        {
                            "u": round(u, 2),
                            "v": round(v, 2),
                            "u_norm": round(u / float(width), 6),
                            "v_norm": round(v / float(height), 6),
                            "score": round(float(m["score"]), 4),
                            "part_tag": m["part_tag"],
                            "grasp_id": m["grasp_id"],
                            "grasp_type": m["grasp_type"],
                            "gripper_type": m["gripper_type"],
                            "verified": bool(m["verified"]),
                            "z_cam": round(z, 4),
                            "width_px": round(float(width_px), 2),
                        }
                    )

    # 4. Derive foreground mask (from explicit fg_mask or depth Z-band) and splat clipped heatmaps
    if fg_mask_full is None and depth_arr is not None and clip_to_foreground:
        fg_mask_full = np.isfinite(depth_arr) & (depth_arr > 0.0)
        z_ref_max = max(obj_z_max or 0.0, max_vis_z)
        if z_ref_max > 0.0:
            fg_mask_full = fg_mask_full & (depth_arr <= z_ref_max + z_band_margin)
        z_min_cands = [z for z in (obj_z_min, min_vis_z) if z is not None and math.isfinite(z) and z > 0.0]
        if z_min_cands:
            z_ref_min = min(z_min_cands)
            fg_mask_full = fg_mask_full & (depth_arr >= max(near_clip, z_ref_min - z_band_margin))
    fg_mask_hm = _resize_mask_to_hw(fg_mask_full, hm_h, hm_w) if fg_mask_full is not None else None

    u_arr = np.asarray(splat_u, dtype=np.float64)
    v_arr = np.asarray(splat_v, dtype=np.float64)
    sig_arr = np.asarray(splat_sig, dtype=np.float64)
    w_arr = np.asarray(splat_w, dtype=np.float64)

    heatmap = _clip_and_renorm_heatmap(
        _splat_gaussians(u_arr, v_arr, sig_arr, w_arr, hm_h, hm_w, width, height), fg_mask_hm
    )
    heatmap_u8 = np.clip(np.round(heatmap * 255.0), 0.0, 255.0).astype(np.uint8)

    part_maps: dict[str, np.ndarray] = {}
    if per_part_heatmaps and splat_tag:
        tags_present = sorted(set(splat_tag))
        for tag in tags_present:
            idx = [i for i, t in enumerate(splat_tag) if t == tag]
            part_maps[tag] = _clip_and_renorm_heatmap(
                _splat_gaussians(u_arr[idx], v_arr[idx], sig_arr[idx], w_arr[idx], hm_h, hm_w, width, height),
                fg_mask_hm,
            )

    # 5. Tight 2D affordance BBox derived from foreground-clipped heatmap (prefer verified grasps)
    aff_bbox_px: list[float] | None = None
    aff_bbox_norm: list[float] | None = None
    if len(u_arr) > 0:
        ver_idx = [i for i, is_v in enumerate(splat_ver) if is_v]
        if ver_idx and len(ver_idx) < len(u_arr):
            box_hm = _clip_and_renorm_heatmap(
                _splat_gaussians(
                    u_arr[ver_idx], v_arr[ver_idx], sig_arr[ver_idx], w_arr[ver_idx], hm_h, hm_w, width, height
                ),
                fg_mask_hm,
            )
            if float(box_hm.max()) < float(bbox_heatmap_thresh):
                box_hm = heatmap
        else:
            box_hm = heatmap

        ys, xs = np.nonzero(box_hm >= float(bbox_heatmap_thresh))
        if len(xs) > 0:
            sx_inv = float(width) / float(hm_w)
            sy_inv = float(height) / float(hm_h)
            x1 = float(np.clip(float(xs.min()) * sx_inv, 0.0, float(width)))
            y1 = float(np.clip(float(ys.min()) * sy_inv, 0.0, float(height)))
            x2 = float(np.clip(float(xs.max() + 1) * sx_inv, 0.0, float(width)))
            y2 = float(np.clip(float(ys.max() + 1) * sy_inv, 0.0, float(height)))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                aff_bbox_px = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                aff_bbox_norm = [
                    round(x1 / width, 6),
                    round(y1 / height, 6),
                    round(x2 / width, 6),
                    round(y2 / height, 6),
                ]

    # 6. Sort keypoints by (verified desc, score desc) and optionally cap to max_keypoints
    vis_kp.sort(key=lambda kp: (int(kp["verified"]), float(kp["score"]), -float(kp["z_cam"])), reverse=True)
    if max_keypoints is not None and max_keypoints >= 0:
        vis_kp = vis_kp[: int(max_keypoints)]

    # 7. Optional expert-selected 3D grasp point projection
    sel_2d: dict[str, Any] | None = None
    if selected_grasp_world is not None:
        sel_pt = np.asarray(selected_grasp_world, dtype=np.float64).reshape(1, 3)
        if np.isfinite(sel_pt).all():
            su, sv, sz = project_world_points(sel_pt, k_mat, cam_to_world, camera_convention=camera_convention)
            u0, v0, z0 = float(su[0]), float(sv[0]), float(sz[0])
            visible = (
                math.isfinite(u0)
                and math.isfinite(v0)
                and z0 > near_clip
                and 0.0 <= u0 < float(width)
                and 0.0 <= v0 < float(height)
            )
            sel_2d = {
                "u": round(u0, 2) if visible else None,
                "v": round(v0, 2) if visible else None,
                "u_norm": round(u0 / float(width), 6) if visible else None,
                "v_norm": round(v0 / float(height), 6) if visible else None,
                "z_cam": round(z0, 4) if math.isfinite(z0) else None,
                "visible": bool(visible),
            }

    return Affordance2DResult(
        width=int(width),
        height=int(height),
        heatmap=heatmap,
        heatmap_uint8=heatmap_u8,
        affordance_bbox_xyxy=aff_bbox_px,
        affordance_bbox_xyxy_norm=aff_bbox_norm,
        object_bbox_xyxy=obj_bbox_px,
        object_bbox_xyxy_norm=obj_bbox_norm,
        keypoints=vis_kp,
        asset_id=asset_id,
        gripper_types=selected_grippers,
        part_heatmaps=part_maps,
        selected_grasp_2d=sel_2d,
    )


# ---------------------------------------------------------------------------
# Pure-NumPy colormaps (Turbo / Jet) & publication-grade RGB overlay
# ---------------------------------------------------------------------------


def _build_turbo_lut() -> np.ndarray:
    """256x3 uint8 Turbo colormap LUT (Anton Mikhailov's polynomial approximation)."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    r = np.clip(
        0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))),
        0.0,
        1.0,
    )
    g = np.clip(
        0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))),
        0.0,
        1.0,
    )
    b = np.clip(
        0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))),
        0.0,
        1.0,
    )
    return np.round(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _build_jet_lut() -> np.ndarray:
    """256x3 uint8 classic Jet colormap LUT."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.round(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


COLORMAPS: dict[str, np.ndarray] = {
    "turbo": _build_turbo_lut(),
    "jet": _build_jet_lut(),
}

_PART_TAG_COLORS: dict[str, tuple[int, int, int]] = {
    "handle": (255, 107, 107),
    "handle_or_protrusion": (255, 107, 107),
    "body": (72, 219, 251),
    "body_top": (72, 219, 251),
    "body_side": (52, 172, 224),
    "rim": (254, 202, 87),
    "neck": (255, 159, 243),
    "cap": (29, 209, 161),
    "lid": (29, 209, 161),
    "spout": (255, 159, 67),
    "base": (162, 155, 254),
}


def colorize_heatmap(heatmap: np.ndarray, *, colormap: str = "turbo") -> np.ndarray:
    """Convert a 2D float [0, 1] or uint8 [0, 255] heatmap into an ``(H, W, 3)`` uint8 RGB image."""
    lut = COLORMAPS.get(colormap.lower())
    if lut is None:
        raise ValueError(f"Unknown colormap={colormap!r}; expected one of {sorted(COLORMAPS)}")
    arr = np.asarray(heatmap)
    if arr.dtype == np.uint8:
        idx = arr
    else:
        idx = np.clip(np.round(np.asarray(arr, dtype=np.float32) * 255.0), 0.0, 255.0).astype(np.uint8)
    return lut[idx]


def _draw_label_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    bg_rgb: tuple[int, int, int],
    fg_rgb: tuple[int, int, int] = (10, 10, 14),
    img_w: int = 0,
    img_h: int = 0,
    avoid: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 4, 2
    pw, ph = tw + 2 * pad_x, th + 2 * pad_y
    rx1 = max(0, min(int(round(x)), max(0, img_w - pw - 1))) if img_w else int(round(x))
    ry1 = max(0, int(round(y)) - ph - 2)
    if img_h and ry1 + ph >= img_h:
        ry1 = max(0, img_h - ph - 1)
    if avoid is not None:
        ax1, ay1, ax2, ay2 = avoid
        if rx1 < ax2 and rx1 + pw > ax1 and ry1 < ay2 and ry1 + ph > ay1:
            # Shift right of the avoided pill if room, otherwise place just below it
            if not img_w or ax2 + 4 + pw < img_w:
                rx1 = ax2 + 4
            else:
                ry1 = min(max(0, ay2 + 2), max(0, img_h - ph - 1)) if img_h else ay2 + 2
    rect = (rx1, ry1, rx1 + pw, ry1 + ph)
    draw.rectangle(rect, fill=bg_rgb)
    draw.text((rx1 + pad_x, ry1 + pad_y - bbox[1]), text, fill=fg_rgb)
    return rect


def render_affordance_2d_overlay(
    rgb: np.ndarray,
    result: Affordance2DResult,
    *,
    colormap: str = "turbo",
    alpha: float = 0.58,
    heatmap_thresh: float = 0.08,
    draw_object_bbox: bool = True,
    draw_affordance_bbox: bool = True,
    draw_keypoints: bool = True,
    top_k_keypoints: int = 12,
    draw_legend: bool = True,
) -> np.ndarray:
    """Render a publication-grade RGB overlay with heatmap, object/affordance BBoxes, and top-K keypoints."""
    base = np.asarray(rgb, dtype=np.uint8)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {base.shape}")
    h, w = int(base.shape[0]), int(base.shape[1])

    # Resize heatmap to match RGB canvas when needed (e.g. compact 64x48 datagen heatmaps)
    if result.heatmap.shape == (h, w):
        hm = np.asarray(result.heatmap, dtype=np.float32)
    else:
        hm = result.resize_heatmap_uint8(h, w).astype(np.float32) / 255.0

    colored = colorize_heatmap(hm, colormap=colormap).astype(np.float32)
    denom = max(1.0 - float(heatmap_thresh), 1e-6)
    mask_alpha = np.clip((hm - float(heatmap_thresh)) / denom, 0.0, 1.0) * float(alpha)
    blended = base.astype(np.float32) * (1.0 - mask_alpha[..., None]) + colored * mask_alpha[..., None]
    out_img = Image.fromarray(np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out_img)

    sx = float(w) / float(max(result.width, 1))
    sy = float(h) / float(max(result.height, 1))

    # Object 2D BBox (cyan)
    obj_pill: tuple[int, int, int, int] | None = None
    if draw_object_bbox and result.object_bbox_xyxy is not None:
        ox1, oy1, ox2, oy2 = result.object_bbox_xyxy
        ox1, oy1, ox2, oy2 = ox1 * sx, oy1 * sy, ox2 * sx, oy2 * sy
        obj_col = (0, 216, 255)
        draw.rectangle([ox1, oy1, ox2, oy2], outline=obj_col, width=2)
        obj_pill = _draw_label_pill(draw, ox1, oy1, "Object", obj_col, img_w=w, img_h=h)

    # Affordance 2D BBox (bright lime)
    if draw_affordance_bbox and result.affordance_bbox_xyxy is not None:
        ax1, ay1, ax2, ay2 = result.affordance_bbox_xyxy
        ax1, ay1, ax2, ay2 = ax1 * sx, ay1 * sy, ax2 * sx, ay2 * sy
        aff_col = (57, 255, 20)
        draw.rectangle([ax1, ay1, ax2, ay2], outline=aff_col, width=2)
        _draw_label_pill(draw, ax1, ay1, "Affordance", aff_col, img_w=w, img_h=h, avoid=obj_pill)

    # Top-K visible keypoints (with mild 2D pixel NMS so nearby points don't stack)
    drawn_parts: set[str] = set()
    if draw_keypoints and result.keypoints and top_k_keypoints > 0:
        r_dot = max(3, int(round(min(w, h) * 0.011)))
        min_sep_sq = float((2.2 * r_dot) ** 2)
        drawn_pts: list[tuple[float, float]] = []
        for kp in result.keypoints:
            if len(drawn_pts) >= int(top_k_keypoints):
                break
            u = float(kp["u"]) * sx
            v = float(kp["v"]) * sy
            if any((u - pu) ** 2 + (v - pv) ** 2 < min_sep_sq for pu, pv in drawn_pts):
                continue
            idx = len(drawn_pts)
            drawn_pts.append((u, v))
            ptag = str(kp.get("part_tag", ""))
            if ptag:
                drawn_parts.add(ptag)
            col = _PART_TAG_COLORS.get(ptag, (255, 230, 80))
            draw.ellipse([u - r_dot - 1, v - r_dot - 1, u + r_dot + 1, v + r_dot + 1], fill=(15, 15, 20))
            draw.ellipse([u - r_dot, v - r_dot, u + r_dot, v + r_dot], fill=col)
            if idx < 3:
                draw.text((u + r_dot + 2, v - r_dot - 2), f"#{idx + 1}", fill=(255, 255, 255))

    # Selected expert grasp point (magenta diamond)
    if result.selected_grasp_2d and result.selected_grasp_2d.get("visible"):
        su = float(result.selected_grasp_2d["u"]) * sx
        sv = float(result.selected_grasp_2d["v"]) * sy
        d = max(5, int(round(min(w, h) * 0.018)))
        poly = [(su, sv - d), (su + d, sv), (su, sv + d), (su - d, sv)]
        draw.polygon(poly, fill=(255, 45, 175), outline=(255, 255, 255))

    # Compact top-left legend bar
    if draw_legend:
        parts = sorted(drawn_parts) if drawn_parts else sorted({str(kp["part_tag"]) for kp in result.keypoints})
        label = result.asset_id or "affordance"
        info = f"{label} | kps={len(result.keypoints)}"
        if parts:
            info += f" | parts={','.join(parts[:3])}"
        _draw_label_pill(draw, 6, 20, info, (20, 24, 32), fg_rgb=(240, 244, 248), img_w=w, img_h=h)

    return np.asarray(out_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# VLM / VLA training export helpers
# ---------------------------------------------------------------------------


def export_views_to_vlm_jsonl(
    affordance_path: Path,
    views_dir: Path,
    output_dir: Path,
    *,
    gripper_types: Sequence[str] | None = None,
    verified_only: bool = False,
    save_overlays: bool = False,
    max_keypoints: int = 16,
    heatmap_hw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Export multi-view rendered images + `affordance.json` to VLM/VLA JSONL + heatmap PNGs."""
    from scripts.objects.affordance_vlm import load_rendered_views

    aff = _load_affordance_dict(affordance_path)
    rendered = load_rendered_views(views_dir)
    output_dir = Path(output_dir)
    hm_dir = output_dir / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = output_dir / "overlays"
    if save_overlays:
        ov_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "annotations.jsonl"
    records: list[dict[str, Any]] = []
    for rv in rendered:
        res = project_affordance_2d(
            aff,
            obj_pos=[0.0, 0.0, 0.0],
            obj_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            K=rv.view.K,
            cam_to_world=rv.view.cam_to_world,
            width=rv.view.width,
            height=rv.view.height,
            depth=rv.depth,
            fg_mask=rv.mask,
            gripper_types=gripper_types,
            verified_only=verified_only,
            max_keypoints=max_keypoints,
            heatmap_hw=heatmap_hw,
        )
        hm_rel = f"heatmaps/{rv.view.view_id}_heatmap.png"
        Image.fromarray(res.heatmap_uint8, mode="L").save(output_dir / hm_rel)
        rec = {
            "view_id": rv.view.view_id,
            "rgb_path": str((views_dir / f"{rv.view.view_id}_rgb.png").resolve()),
            "heatmap_path": hm_rel,
            **res.to_dict(include_heatmap=False),
        }
        if save_overlays:
            ov_rel = f"overlays/{rv.view.view_id}_overlay.png"
            ov = render_affordance_2d_overlay(rv.rgb, res)
            Image.fromarray(ov, mode="RGB").save(output_dir / ov_rel)
            rec["overlay_path"] = ov_rel
        records.append(rec)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"records": len(records), "jsonl": str(jsonl_path), "heatmaps_dir": str(hm_dir)}


def export_hdf5_to_vlm_jsonl(
    hdf5_paths: Sequence[Path],
    output_dir: Path,
    *,
    cameras: Sequence[str] | None = None,
    frame_stride: int = 1,
    save_heatmaps: bool = True,
    save_overlays: bool = False,
) -> dict[str, Any]:
    """Export datagen HDF5 `affordance_2d/` streams into VLM/VLA JSONL (+ optional heatmap/overlay PNGs)."""
    import io

    import h5py

    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hm_dir = output_dir / "heatmaps"
    if save_heatmaps:
        hm_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = output_dir / "overlays"
    if save_overlays:
        ov_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "annotations.jsonl"
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for ep_idx, h5_path in enumerate(hdf5_paths):
            h5_path = Path(h5_path)
            with h5py.File(h5_path, "r") as f:
                if "affordance_2d" not in f:
                    continue
                instr_raw = f["instruction"][()] if "instruction" in f else b""
                instruction = (
                    instr_raw.decode("utf-8", errors="replace")
                    if isinstance(instr_raw, bytes | np.bytes_)
                    else str(instr_raw)
                )
                aff_grp = f["affordance_2d"]
                avail_cams = sorted(aff_grp.keys())
                sel_cams = [c for c in (cameras or avail_cams) if c in aff_grp]
                for cam in sel_cams:
                    cg = aff_grp[cam]
                    obj_boxes = cg["object_bbox_xyxy_norm"][()]
                    aff_boxes = cg["affordance_bbox_xyxy_norm"][()]
                    kps_arr = cg["keypoints_xy_score"][()]
                    hms = cg["heatmap_uint8"][()]
                    sel_arr = cg["selected_grasp_xy_norm"][()] if "selected_grasp_xy_norm" in cg else None
                    raw_src_hw = cg.attrs.get("source_hw", "[480, 640]")
                    if isinstance(raw_src_hw, bytes | np.bytes_):
                        raw_src_hw = raw_src_hw.decode("utf-8", errors="replace")
                    src_hw = json.loads(raw_src_hw)
                    src_h, src_w = int(src_hw[0]), int(src_hw[1])
                    raw_asset = cg.attrs.get("asset_id", "")
                    asset_id = (
                        raw_asset.decode("utf-8", errors="replace")
                        if isinstance(raw_asset, bytes | np.bytes_)
                        else str(raw_asset)
                    )
                    raw_label = cg.attrs.get("target_label", "")
                    target_label = (
                        raw_label.decode("utf-8", errors="replace")
                        if isinstance(raw_label, bytes | np.bytes_)
                        else str(raw_label)
                    )
                    ep_tag = f"ep{ep_idx:05d}_{h5_path.stem}"
                    n_frames = len(hms)
                    has_rgb = save_overlays and f"vision/{cam}/colors" in f
                    for fi in range(0, n_frames, frame_stride):
                        ob = obj_boxes[fi]
                        ab = aff_boxes[fi]
                        ob_list = [round(float(x), 6) for x in ob] if np.isfinite(ob).all() else None
                        ab_list = [round(float(x), 6) for x in ab] if np.isfinite(ab).all() else None
                        kps_valid = []
                        for row in kps_arr[fi]:
                            if np.isfinite(row).all():
                                kps_valid.append(
                                    {
                                        "u_norm": round(float(row[0]), 6),
                                        "v_norm": round(float(row[1]), 6),
                                        "score": round(float(row[2]), 4),
                                    }
                                )
                        sel_pt = None
                        if sel_arr is not None and np.isfinite(sel_arr[fi]).all():
                            sel_pt = [round(float(sel_arr[fi][0]), 6), round(float(sel_arr[fi][1]), 6)]
                        rec: dict[str, Any] = {
                            "episode": h5_path.stem,
                            "episode_path": str(h5_path),
                            "episode_index": ep_idx,
                            "frame_index": fi,
                            "camera": cam,
                            "instruction": instruction,
                            "asset_id": asset_id,
                            "target_label": target_label,
                            "width": src_w,
                            "height": src_h,
                            "heatmap_hw": [int(hms.shape[1]), int(hms.shape[2])],
                            "object_bbox_xyxy_norm": ob_list,
                            "affordance_bbox_xyxy_norm": ab_list,
                            "keypoints": kps_valid,
                            "selected_grasp_xy_norm": sel_pt,
                        }
                        if save_heatmaps:
                            hm_rel = f"heatmaps/{ep_tag}_{cam}_f{fi:05d}.png"
                            Image.fromarray(hms[fi], mode="L").save(output_dir / hm_rel)
                            rec["heatmap_path"] = hm_rel
                        if has_rgb:
                            raw_buf = bytes(f[f"vision/{cam}/colors"][fi]).rstrip(b"\x00")
                            try:
                                rgb_img = np.asarray(Image.open(io.BytesIO(raw_buf)).convert("RGB"), dtype=np.uint8)
                                res_obj = Affordance2DResult(
                                    width=src_w,
                                    height=src_h,
                                    heatmap=hms[fi].astype(np.float32) / 255.0,
                                    heatmap_uint8=hms[fi],
                                    affordance_bbox_xyxy=(
                                        [ab_list[0] * src_w, ab_list[1] * src_h, ab_list[2] * src_w, ab_list[3] * src_h]
                                        if ab_list
                                        else None
                                    ),
                                    affordance_bbox_xyxy_norm=ab_list,
                                    object_bbox_xyxy=(
                                        [ob_list[0] * src_w, ob_list[1] * src_h, ob_list[2] * src_w, ob_list[3] * src_h]
                                        if ob_list
                                        else None
                                    ),
                                    object_bbox_xyxy_norm=ob_list,
                                    keypoints=[
                                        {
                                            "u": kp["u_norm"] * src_w,
                                            "v": kp["v_norm"] * src_h,
                                            "u_norm": kp["u_norm"],
                                            "v_norm": kp["v_norm"],
                                            "score": kp["score"],
                                            "part_tag": "grasp",
                                        }
                                        for kp in kps_valid
                                    ],
                                    asset_id=asset_id,
                                    selected_grasp_2d=(
                                        {
                                            "u": sel_pt[0] * src_w,
                                            "v": sel_pt[1] * src_h,
                                            "u_norm": sel_pt[0],
                                            "v_norm": sel_pt[1],
                                            "visible": True,
                                        }
                                        if sel_pt
                                        else None
                                    ),
                                )
                                ov_rel = f"overlays/{ep_tag}_{cam}_f{fi:05d}.png"
                                Image.fromarray(render_affordance_2d_overlay(rgb_img, res_obj), mode="RGB").save(
                                    output_dir / ov_rel
                                )
                                rec["overlay_path"] = ov_rel
                            except Exception:
                                pass
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        count += 1
    return {"records": count, "jsonl": str(jsonl_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_hw(spec: str) -> tuple[int, int]:
    """Parse 'WxH' into (H, W)."""
    parts = spec.lower().split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise argparse.ArgumentTypeError(f"Expected WxH (e.g. 64x48), got {spec!r}")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Heatmap dimensions must be positive, got {spec!r}")
    return (h, w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenFlywheel 2D affordance projection & VLM/VLA export")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ren = sub.add_parser("render", help="Render multi-view 2D affordance heatmap + BBox overlays")
    p_ren.add_argument("affordance", type=Path, help="Path to affordance.json")
    src_grp = p_ren.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--mesh", type=Path, help="Mesh file (.obj/.stl/.usd[a|z]) or asset directory")
    src_grp.add_argument("--views-dir", type=Path, help="Pre-rendered views dir from affordance_vlm.py render")
    p_ren.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    p_ren.add_argument("--gripper", action="append", dest="grippers", help="Filter gripper type(s)")
    p_ren.add_argument("--verified-only", action="store_true", help="Include only Stage-2 verified grasps")
    p_ren.add_argument("--colormap", default="turbo", choices=sorted(COLORMAPS))
    p_ren.add_argument("--image-size", type=int, default=512, help="Square render size when using --mesh")
    p_ren.add_argument("--n-azimuth", type=int, default=6, help="Azimuth views when using --mesh")
    p_ren.add_argument("--top-k", type=int, default=12, help="Top-K keypoints drawn on overlay")

    p_exp = sub.add_parser("export-vlm", help="Export VLM/VLA JSONL + heatmap PNGs from views or datagen HDF5")
    p_exp.add_argument("--affordance", type=Path, help="affordance.json (used with --views-dir)")
    p_exp.add_argument("--views-dir", type=Path, help="Pre-rendered views dir from affordance_vlm.py render")
    p_exp.add_argument("--hdf5", type=Path, nargs="+", help="Recorded datagen HDF5 episode(s) with affordance_2d/")
    p_exp.add_argument("--run-dir", type=Path, help="Datagen run dir containing seed*/episode_*.hdf5")
    p_exp.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    p_exp.add_argument("--gripper", action="append", dest="grippers", help="Filter gripper type(s) (views mode)")
    p_exp.add_argument("--verified-only", action="store_true", help="Only verified grasps (views mode)")
    p_exp.add_argument("--cameras", default="", help="Comma-separated camera filter (HDF5 mode)")
    p_exp.add_argument("--frame-stride", type=int, default=1, help="Subsample stride in frames (HDF5 mode)")
    p_exp.add_argument("--heatmap-size", type=_parse_hw, default=None, help="Optional WxH heatmap size (views mode)")
    p_exp.add_argument("--save-overlays", action="store_true", help="Also save RGB+heatmap+BBox overlay PNGs")

    args = parser.parse_args(argv)

    if args.cmd == "render":
        from scripts.objects.affordance import load_mesh, resolve_mesh_path
        from scripts.objects.affordance_vlm import build_multiview_cameras, load_rendered_views, render_mesh_views

        aff = _load_affordance_dict(args.affordance)
        if args.views_dir is not None:
            rendered = load_rendered_views(args.views_dir)
        else:
            mesh_path = resolve_mesh_path(args.mesh)
            verts, faces = load_mesh(mesh_path)
            b_min = verts.min(axis=0)
            b_max = verts.max(axis=0)
            cams = build_multiview_cameras(
                b_min,
                b_max,
                n_azimuth=args.n_azimuth,
                elevations_deg=(22.0, 48.0),
                include_top=True,
                image_size=(args.image_size, args.image_size),
            )
            rendered = render_mesh_views(verts, faces, cams)

        args.output.mkdir(parents=True, exist_ok=True)
        summaries = []
        for rv in rendered:
            res = project_affordance_2d(
                aff,
                obj_pos=[0.0, 0.0, 0.0],
                obj_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                K=rv.view.K,
                cam_to_world=rv.view.cam_to_world,
                width=rv.view.width,
                height=rv.view.height,
                depth=rv.depth,
                fg_mask=rv.mask,
                gripper_types=args.grippers,
                verified_only=args.verified_only,
            )
            ov = render_affordance_2d_overlay(
                rv.rgb,
                res,
                colormap=args.colormap,
                top_k_keypoints=args.top_k,
            )
            Image.fromarray(ov, mode="RGB").save(args.output / f"{rv.view.view_id}_overlay.png")
            Image.fromarray(res.heatmap_uint8, mode="L").save(args.output / f"{rv.view.view_id}_heatmap.png")
            summaries.append({"view_id": rv.view.view_id, **res.to_dict(include_heatmap=False)})

        (args.output / "affordance_2d_views.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"views": len(summaries), "output": str(args.output)}, indent=2))
        return 0

    if args.cmd == "export-vlm":
        if args.views_dir is not None:
            if args.affordance is None:
                parser.error("--affordance is required with --views-dir")
            summary = export_views_to_vlm_jsonl(
                args.affordance,
                args.views_dir,
                args.output,
                gripper_types=args.grippers,
                verified_only=args.verified_only,
                save_overlays=args.save_overlays,
                heatmap_hw=args.heatmap_size,
            )
            print(json.dumps(summary, indent=2))
            return 0

        h5_files: list[Path] = list(args.hdf5 or ())
        if args.run_dir is not None:
            h5_files.extend(sorted(args.run_dir.glob("seed*/episode_*.hdf5")))
            if not h5_files:
                h5_files.extend(sorted(args.run_dir.glob("episode_*.hdf5")))
        if not h5_files:
            parser.error("Provide either (--affordance + --views-dir) or (--hdf5 / --run-dir)")
        cam_filter = [c.strip() for c in args.cameras.split(",") if c.strip()] or None
        summary = export_hdf5_to_vlm_jsonl(
            h5_files,
            args.output,
            cameras=cam_filter,
            frame_stride=args.frame_stride,
            save_overlays=args.save_overlays,
        )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
