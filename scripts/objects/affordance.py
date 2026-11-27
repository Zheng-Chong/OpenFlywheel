"""Automated object affordance annotation for OpenFlywheel simulation assets.

Stage 1 (offline, CPU-only numpy/scipy + optional pxr):
  1. Loads an object's collision mesh (preferring UsdPhysics.CollisionAPI in USD,
     with pure-Python fallback loaders for .obj, .stl, and ASCII .usda).
  2. Extracts geometric metadata: AABB bounds, area-weighted CoM, functional frames
     (support_base, center_of_mass, top_center), and Z-axis rotational symmetry.
  3. Samples antipodal (force-closure) parallel-jaw grasp candidates for a target
     gripper (default: G1-D Dex1-1, matching scripts/g1d/scripted_demo.py frame
     conventions), filters them with 3D gripper-mesh collision checks, assigns
     heuristic part tags, scores them, and deduplicates via pose NMS.
  4. Emits a versioned sidecar `affordance.json` in the object's local frame at
     scale_ref=1.0.

Dex1-1 EE frame convention (matches scripts/g1d/scripted_demo.py):
  - Origin: grasp_point (between finger pads, PAD_REACH=0.12 m ahead of dex1_base_link)
  - +X_ee: approach_dir (palm -> fingertips; fingertips at +FINGER_TIPS=+0.023 m)
  - +Y_ee: closing_axis (jaws open/close along +/-Y_ee)
  - +Z_ee: dorsal_dir = approach_dir x closing_axis (wrist-camera side)

CLI usage:
    python scripts/objects/affordance.py annotate path/to/object.obj --output affordance.json
    python scripts/objects/affordance.py annotate Assets/Object/Real2Sim/Rigid/plastic_cup/00001
    python scripts/objects/affordance.py inspect path/to/affordance.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GripperSpec:
    """Physical dimensions of a parallel-jaw gripper in its EE grasp_point frame (metres)."""

    name: str
    pad_closed: float  # minimum jaw gap at grip=0
    pad_travel: float  # jaw travel from grip=0 to grip=1
    grasp_clear: float  # extra opening width over grasped side during approach
    min_width: float  # minimum graspable object thickness
    max_width: float  # maximum graspable object thickness (with grasp_clear)
    pad_reach: float  # ee_link origin -> grasp_point along +X_ee
    finger_tips: float  # grasp_point -> fingertips along +X_ee
    finger_back: float  # grasp_point -> palm face along -X_ee
    finger_thick: float  # finger + pad thickness along closing axis (Y_ee)
    finger_depth: float  # finger height perpendicular to approach & closing (Z_ee)
    palm_len: float  # palm housing extent behind finger_back along -X_ee
    palm_width: float  # palm housing width along Y_ee
    palm_height: float  # palm housing height along Z_ee

    @property
    def max_open(self) -> float:
        return self.pad_closed + self.pad_travel


GRIPPER_SPECS: dict[str, GripperSpec] = {
    # Matches scripts/g1d/scripted_demo.py (PAD_CLOSED=0.006, PAD_TRAVEL=0.089,
    # GRASP_CLEAR=0.025, PAD_REACH=0.12, FINGER_TIPS=0.023) and cup_handle.py FINGER.
    "dex1_1": GripperSpec(
        name="dex1_1",
        pad_closed=0.006,
        pad_travel=0.089,
        grasp_clear=0.025,
        min_width=0.008,
        max_width=0.070,
        pad_reach=0.120,
        finger_tips=0.023,
        finger_back=0.045,
        finger_thick=0.016,
        finger_depth=0.030,
        palm_len=0.075,
        palm_width=0.085,
        palm_height=0.050,
    ),
    "aurora_lite": GripperSpec(
        name="aurora_lite",
        pad_closed=0.004,
        pad_travel=0.086,
        grasp_clear=0.020,
        min_width=0.006,
        max_width=0.070,
        pad_reach=0.115,
        finger_tips=0.022,
        finger_back=0.042,
        finger_thick=0.014,
        finger_depth=0.026,
        palm_len=0.070,
        palm_width=0.075,
        palm_height=0.045,
    ),
}


# ---------------------------------------------------------------------------
# Mesh loading (OBJ, STL, ASCII USDA, and OpenUSD via pxr when available)
# ---------------------------------------------------------------------------


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh from a Wavefront .obj file (pure Python)."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx = []
            for tok in parts:
                v_str = tok.split("/")[0]
                if not v_str:
                    continue
                vi = int(v_str)
                idx.append(vi - 1 if vi > 0 else len(verts) + vi)
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    v_arr = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return v_arr, f_arr


def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh from a binary or ASCII .stl file (pure Python)."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n_tris = struct.unpack_from("<I", data, 80)[0]
        if 84 + n_tris * 50 == len(data):
            tris = np.empty((n_tris, 3, 3), dtype=np.float64)
            offset = 84
            for i in range(n_tris):
                vals = struct.unpack_from("<12f", data, offset)
                tris[i, 0] = vals[3:6]
                tris[i, 1] = vals[6:9]
                tris[i, 2] = vals[9:12]
                offset += 50
            verts = tris.reshape(-1, 3)
            faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
            return verts, faces
    # ASCII STL fallback
    verts_list: list[list[float]] = []
    for raw in data.decode("utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("vertex "):
            parts = line.split()
            if len(parts) >= 4:
                verts_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.asarray(verts_list, dtype=np.float64).reshape(-1, 3)
    n_tris = len(verts) // 3
    verts = verts[: n_tris * 3]
    faces = np.arange(n_tris * 3, dtype=np.int64).reshape(-1, 3)
    return verts, faces


def _load_usda_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight pure-Python ASCII USDA UsdGeom.Mesh parser (fallback when pxr is absent)."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    mpu_match = re.search(r"metersPerUnit\s*=\s*([0-9eE.+\-]+)", text)
    mpu = float(mpu_match.group(1)) if mpu_match else 1.0

    points_blocks = re.findall(r"point3f\[\]\s+points\s*=\s*\[(.*?)\]", text, flags=re.S)
    counts_blocks = re.findall(r"int\[\]\s+faceVertexCounts\s*=\s*\[(.*?)\]", text, flags=re.S)
    indices_blocks = re.findall(r"int\[\]\s+faceVertexIndices\s*=\s*\[(.*?)\]", text, flags=re.S)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    v_offset = 0
    for p_blk, c_blk, i_blk in zip(points_blocks, counts_blocks, indices_blocks):
        tuples = re.findall(r"\(\s*([0-9eE.+\-]+)\s*,\s*([0-9eE.+\-]+)\s*,\s*([0-9eE.+\-]+)\s*\)", p_blk)
        if not tuples:
            continue
        pts = np.asarray([[float(a), float(b), float(c)] for a, b, c in tuples], dtype=np.float64) * mpu
        counts = np.fromstring(c_blk.replace("\n", " "), sep=",", dtype=np.int64)
        indices = np.fromstring(i_blk.replace("\n", " "), sep=",", dtype=np.int64)
        if len(counts) == 0 or len(indices) == 0:
            continue
        tris: list[list[int]] = []
        cursor = 0
        for cnt in counts:
            c_int = int(cnt)
            poly = indices[cursor : cursor + c_int]
            cursor += c_int
            for k in range(1, c_int - 1):
                tris.append([int(poly[0]) + v_offset, int(poly[k]) + v_offset, int(poly[k + 1]) + v_offset])
        if tris:
            all_verts.append(pts)
            all_faces.append(np.asarray(tris, dtype=np.int64))
            v_offset += len(pts)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


_PXR_APP: Any = None


def _ensure_pxr() -> tuple[Any, Any, Any]:
    """Import (Usd, UsdGeom, UsdPhysics), bootstrapping headless Isaac Sim inside isaac-sim if needed."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        return Usd, UsdGeom, UsdPhysics
    except ImportError:
        global _PXR_APP
        if _PXR_APP is None:
            from isaacsim import SimulationApp  # type: ignore[import-not-found]

            _PXR_APP = SimulationApp({"headless": True})
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        return Usd, UsdGeom, UsdPhysics


def close_pxr_app() -> None:
    """Close the headless Isaac Sim SimulationApp if `_ensure_pxr()` launched one."""
    global _PXR_APP
    if _PXR_APP is not None:
        app = _PXR_APP
        _PXR_APP = None
        app.close()


def _has_collision_api(prim: Any, collision_api: Any) -> bool:
    """Return True if `prim` or any ancestor Xform/Scope carries UsdPhysics.CollisionAPI."""
    cur = prim
    while cur and cur.IsValid() and not cur.IsPseudoRoot():
        if cur.HasAPI(collision_api):
            return True
        cur = cur.GetParent()
    return False


def _load_usd_pxr(path: Path, prefer_collision: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load triangle mesh from .usd/.usdz/.usda using OpenUSD pxr (handles instances & CollisionAPI)."""
    Usd, UsdGeom, UsdPhysics = _ensure_pxr()

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {path}")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    inv = cache.GetLocalToWorldTransform(root).GetInverse() if root else None
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))

    def _collect(collision_only: bool) -> tuple[np.ndarray, np.ndarray]:
        verts_out: list[np.ndarray] = []
        faces_out: list[np.ndarray] = []
        v_offset = 0
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            has_col = _has_collision_api(prim, UsdPhysics.CollisionAPI)
            if collision_only and not has_col:
                continue
            mesh = UsdGeom.Mesh(prim)
            if not collision_only and not has_col:
                purpose = mesh.GetPurposeAttr().Get()
                if purpose == UsdGeom.Tokens.guide:
                    continue
            raw_pts = mesh.GetPointsAttr().Get()
            raw_cnts = mesh.GetFaceVertexCountsAttr().Get()
            raw_idxs = mesh.GetFaceVertexIndicesAttr().Get()
            if not raw_pts or not raw_cnts or not raw_idxs:
                continue
            pts = np.asarray(raw_pts, dtype=np.float64)
            xform = cache.GetLocalToWorldTransform(prim)
            if inv is not None:
                xform = xform * inv
            mat = np.asarray(xform, dtype=np.float64)
            pts = (np.c_[pts, np.ones(len(pts))] @ mat)[:, :3] * mpu
            counts = np.asarray(raw_cnts, dtype=np.int64)
            indices = np.asarray(raw_idxs, dtype=np.int64)
            if len(counts) == 0 or int(counts.max()) < 3:
                continue
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            tris_list = []
            for k in range(1, int(counts.max()) - 1):
                sel = counts > k + 1
                tris_list.append(indices[np.c_[starts[sel], starts[sel] + k, starts[sel] + k + 1]] + v_offset)
            if tris_list:
                tris_arr = np.concatenate(tris_list, axis=0)
                # Keep outward triangle winding consistent if the prim transform has a reflection
                if float(np.linalg.det(mat[:3, :3])) < 0.0:
                    tris_arr = tris_arr[:, [0, 2, 1]]
                verts_out.append(pts)
                faces_out.append(tris_arr)
                v_offset += len(pts)
        if not verts_out:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(verts_out, axis=0), np.concatenate(faces_out, axis=0)

    if prefer_collision:
        v, f = _collect(collision_only=True)
        if len(f) > 0:
            return v, f
    return _collect(collision_only=False)


def resolve_mesh_path(path: Path) -> Path:
    """Resolve an asset directory or mesh file path to a concrete mesh file."""
    p = Path(path)
    if p.is_dir():
        for name in ("object.usd", "object.usdz", "object.usda", "collision.obj", "mesh.obj", "object.obj"):
            cand = p / name
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No supported mesh/USD file found in directory: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Mesh path does not exist: {p}")
    return p


def load_mesh(path: Path, prefer_collision: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load (vertices [V,3], faces [F,3]) in metres from OBJ, STL, or USD/USDA/USDZ."""
    p = resolve_mesh_path(path)
    ext = p.suffix.lower()
    if ext == ".obj":
        return load_obj(p)
    if ext == ".stl":
        return load_stl(p)
    if ext in (".usd", ".usdz", ".usda"):
        try:
            return _load_usd_pxr(p, prefer_collision=prefer_collision)
        except ImportError:
            if ext == ".usda":
                return _load_usda_ascii(p)
            raise RuntimeError(
                f"Loading binary USD ({p.name}) requires OpenUSD `pxr` (available inside isaac-sim / Isaac Sim)."
            )
    raise ValueError(f"Unsupported mesh extension: {ext}")


def compute_mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Deterministic SHA-256 hash of quantized mesh geometry for cache invalidation."""
    h = hashlib.sha256()
    v_q = np.round(np.asarray(vertices, dtype=np.float64), 6)
    f_i = np.asarray(faces, dtype=np.int64)
    h.update(v_q.tobytes())
    h.update(f_i.tobytes())
    return f"sha256:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Geometry analysis: normals, areas, symmetry, functional frames
# ---------------------------------------------------------------------------


def triangle_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (tri_verts [F,3,3], normals [F,3], areas [F], centers [F,3]), dropping degenerate faces."""
    if len(vertices) == 0 or len(faces) == 0:
        empty3 = np.zeros((0, 3), dtype=np.float64)
        return np.zeros((0, 3, 3), dtype=np.float64), empty3, np.zeros((0,), dtype=np.float64), empty3
    tris = vertices[faces]
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    cross = np.cross(e1, e2)
    dbl_area = np.linalg.norm(cross, axis=1)
    valid = dbl_area > 1e-12
    tris = tris[valid]
    areas = 0.5 * dbl_area[valid]
    normals = cross[valid] / dbl_area[valid, None]
    centers = tris.mean(axis=1)
    return tris, normals, areas, centers


def sample_surface_points(
    tris: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted uniform sampling of (points [N,3], normals [N,3], tri_idx [N])."""
    if len(tris) == 0 or n_samples <= 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, np.zeros((0,), dtype=np.int64)
    probs = areas / areas.sum()
    tri_idx = rng.choice(len(tris), size=n_samples, p=probs)
    r1 = np.sqrt(rng.random(n_samples))
    r2 = rng.random(n_samples)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    chosen = tris[tri_idx]
    pts = w0[:, None] * chosen[:, 0] + w1[:, None] * chosen[:, 1] + w2[:, None] * chosen[:, 2]
    nrms = normals[tri_idx]
    return pts, nrms, tri_idx


def detect_symmetry(vertices: np.ndarray, sectors: int = 36) -> dict[str, Any]:
    """Detect Z-axis rotational symmetry of an object in its local frame."""
    if len(vertices) < 4:
        return {"z_axis": "none", "fold": 1, "axis_xy": [0.0, 0.0]}
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center_xy = 0.5 * (lo[:2] + hi[:2])
    axis_xy = [round(float(center_xy[0]), 5), round(float(center_xy[1]), 5)]
    rel_xy = vertices[:, :2] - center_xy
    radii = np.linalg.norm(rel_xy, axis=1)
    if float(radii.max()) < 1e-6:
        return {"z_axis": "none", "fold": 1, "axis_xy": axis_xy}

    z_min, z_max = float(lo[2]), float(hi[2])
    z_span = max(z_max - z_min, 1e-6)
    angles = np.arctan2(rel_xy[:, 1], rel_xy[:, 0]) % (2.0 * np.pi)
    sec_idx = np.floor(angles / (2.0 * np.pi) * sectors).astype(int) % sectors

    # Evaluate sector max-radius profile across 3 horizontal bands to avoid false positives
    # (e.g. a cup with a handle in the middle band is NOT rotationally symmetric).
    bands = [
        (z_min + 0.1 * z_span, z_min + 0.4 * z_span),
        (z_min + 0.35 * z_span, z_min + 0.65 * z_span),
        (z_min + 0.6 * z_span, z_min + 0.9 * z_span),
    ]
    profiles: list[np.ndarray] = []
    for z0, z1 in bands:
        mask = (vertices[:, 2] >= z0) & (vertices[:, 2] <= z1) & (radii > 1e-5)
        if mask.sum() < sectors:
            continue
        prof = np.full(sectors, np.nan)
        for s in range(sectors):
            sel = mask & (sec_idx == s)
            if sel.any():
                prof[s] = radii[sel].max()
        if np.isfinite(prof).sum() >= int(0.8 * sectors):
            med = float(np.nanmedian(prof))
            prof = np.where(np.isfinite(prof), prof, med)
            profiles.append(prof)

    if not profiles:
        # Fallback to full-height profile (e.g. sparse box with only 8 corner vertices)
        prof = np.full(sectors, np.nan)
        for s in range(sectors):
            sel = (sec_idx == s) & (radii > 1e-5)
            if sel.any():
                prof[s] = radii[sel].max()
        if np.isfinite(prof).sum() < 4:
            return {"z_axis": "none", "fold": 1, "axis_xy": axis_xy}
        # Sparse mesh (like an 8-vertex cube): check 4-fold / 2-fold directly on vertices
        res = _detect_discrete_vertex_symmetry(rel_xy, vertices[:, 2])
        res["axis_xy"] = axis_xy
        return res

    # Continuous Z-symmetry: radial variance small in every band
    is_continuous = all(float(np.std(p) / max(np.mean(p), 1e-6)) < 0.045 for p in profiles)
    if is_continuous:
        return {"z_axis": "continuous", "fold": 0, "axis_xy": axis_xy}

    # Check 4-fold (90 deg) and 2-fold (180 deg) rotational invariance
    for fold in (4, 2):
        shift = sectors // fold
        ok = True
        for p in profiles:
            diff = np.max(np.abs(p - np.roll(p, shift))) / max(float(np.mean(p)), 1e-6)
            if diff > 0.06:
                ok = False
                break
        if ok:
            return {"z_axis": "discrete", "fold": fold, "axis_xy": axis_xy}

    return {"z_axis": "none", "fold": 1, "axis_xy": axis_xy}


def _detect_discrete_vertex_symmetry(rel_xy: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Check 4-fold / 2-fold Z symmetry for sparse polyhedral meshes (e.g., 8-vertex boxes)."""
    pts = np.column_stack([rel_xy, z])
    scale = max(float(np.linalg.norm(rel_xy, axis=1).max()), 1e-6)
    for fold in (4, 2):
        angle = 2.0 * np.pi / fold
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotated = pts @ rot.T
        # Nearest-neighbor distance from every rotated vertex to original vertices
        dists = np.linalg.norm(rotated[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        if float(dists.max()) < 0.03 * scale:
            return {"z_axis": "discrete", "fold": fold}
    return {"z_axis": "none", "fold": 1}


def extract_functional_frames(
    vertices: np.ndarray,
    areas: np.ndarray,
    centers: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract canonical functional frames in the object local frame."""
    if len(vertices) == 0:
        return {}
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    z_span = max(float(hi[2] - lo[2]), 1e-6)

    # Support base: centroid of bottom-most vertices (within 3% of height)
    base_mask = vertices[:, 2] <= lo[2] + max(0.002, 0.03 * z_span)
    base_xy = vertices[base_mask, :2].mean(axis=0) if base_mask.any() else 0.5 * (lo[:2] + hi[:2])

    # Top center: centroid of top-most vertices
    top_mask = vertices[:, 2] >= hi[2] - max(0.002, 0.03 * z_span)
    top_xy = vertices[top_mask, :2].mean(axis=0) if top_mask.any() else 0.5 * (lo[:2] + hi[:2])

    # Area-weighted surface centroid (proxy for CoM)
    if len(areas) > 0 and areas.sum() > 0:
        com = (centers * areas[:, None]).sum(axis=0) / areas.sum()
    else:
        com = 0.5 * (lo + hi)

    frames: dict[str, Any] = {
        "support_base": {
            "position": [round(float(base_xy[0]), 5), round(float(base_xy[1]), 5), round(float(lo[2]), 5)],
            "normal": [0.0, 0.0, -1.0],
        },
        "center_of_mass": {
            "position": [round(float(com[0]), 5), round(float(com[1]), 5), round(float(com[2]), 5)],
        },
        "top_center": {
            "position": [round(float(top_xy[0]), 5), round(float(top_xy[1]), 5), round(float(hi[2]), 5)],
            "normal": [0.0, 0.0, 1.0],
        },
    }

    # Bridge existing articulated / handle metadata if present
    if metadata and isinstance(metadata, dict):
        if "articulation" in metadata and isinstance(metadata["articulation"], dict):
            frames["articulation"] = metadata["articulation"]
    return frames


# ---------------------------------------------------------------------------
# Vectorized antipodal ray-triangle intersection & 3D gripper collision check
# ---------------------------------------------------------------------------


def _ray_mesh_antipodal_hits(
    origin: np.ndarray,
    direction: np.ndarray,
    tris: np.ndarray,
    tri_normals: np.ndarray,
    tri_centers: np.ndarray,
    tri_radii: np.ndarray,
    min_dist: float,
    max_dist: float,
    cos_friction: float,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Find all antipodal opposing surface hits along ray (origin, direction) in [min_dist, max_dist]."""
    rel = tri_centers - origin
    t_proj = rel @ direction
    cand = (t_proj >= min_dist - tri_radii) & (t_proj <= max_dist + tri_radii)
    if not cand.any():
        return []
    # Perpendicular distance bound from ray to triangle center
    rel_c = rel[cand]
    tp_c = t_proj[cand]
    perp_sq = np.sum(rel_c * rel_c, axis=1) - tp_c * tp_c
    r_c = tri_radii[cand]
    near = perp_sq <= r_c * r_c + 1e-12
    if not near.any():
        return []

    idx = np.flatnonzero(cand)[near]
    # Also pre-filter by antipodal normal alignment: exiting surface normal aligns with ray direction
    align = tri_normals[idx] @ direction
    ok_n = align >= cos_friction
    if not ok_n.any():
        return []
    idx = idx[ok_n]

    # Möller-Trumbore intersection
    v0 = tris[idx, 0]
    e1 = tris[idx, 1] - v0
    e2 = tris[idx, 2] - v0
    h = np.cross(direction[None, :], e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > 1e-12
    if not valid.any():
        return []
    idx, v0, e1, e2, h, a = idx[valid], v0[valid], e1[valid], e2[valid], h[valid], a[valid]
    f = 1.0 / a
    s = origin[None, :] - v0
    u = f * np.einsum("ij,ij->i", s, h)
    m = (u >= -1e-6) & (u <= 1.0 + 1e-6)
    if not m.any():
        return []
    idx, e1, e2, f, s, u = idx[m], e1[m], e2[m], f[m], s[m], u[m]
    q = np.cross(s, e1)
    v = f * (q @ direction)
    m2 = (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
    if not m2.any():
        return []
    idx, e2, f, q = idx[m2], e2[m2], f[m2], q[m2]
    t = f * np.einsum("ij,ij->i", e2, q)
    m3 = (t >= min_dist) & (t <= max_dist)
    if not m3.any():
        return []

    hits: list[tuple[float, np.ndarray, np.ndarray]] = []
    for ti, face_i in zip(t[m3], idx[m3]):
        hit_pt = origin + float(ti) * direction
        hits.append((float(ti), hit_pt, tri_normals[face_i]))
    hits.sort(key=lambda item: item[0])
    return hits


def _build_hand_rotation(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Build orthonormal right-handed 3x3 rotation [approach (+X_ee), closing (+Y_ee), dorsal (+Z_ee)]."""
    x = approach / np.linalg.norm(approach)
    y = closing - np.dot(closing, x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    return np.column_stack([x, y, z])


def _rot_to_wxyz(rot_mat: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to canonical (w >= 0) [w, x, y, z] quaternion."""
    xyzw = Rotation.from_matrix(rot_mat).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if wxyz[0] < 0.0:
        wxyz = -wxyz
    return [round(float(v), 6) for v in wxyz]


def _gripper_collides(
    collision_pts: np.ndarray,
    grasp_pos: np.ndarray,
    rot_mat: np.ndarray,
    width: float,
    spec: GripperSpec,
    safety_margin: float = 0.002,
) -> bool:
    """Return True if any object collision point penetrates the open gripper's fingers or palm."""
    if len(collision_pts) == 0:
        return False
    # Transform object points into EE grasp_point frame: P_ee = (P - grasp_pos) @ R
    p_ee = (collision_pts - grasp_pos) @ rot_mat
    x, y, z = p_ee[:, 0], p_ee[:, 1], p_ee[:, 2]

    # Open finger inner surface sits at +/- open_half during approach
    open_half = min(0.5 * (width + spec.grasp_clear), 0.5 * spec.max_open)
    # Require a small inner clearance (safety_margin) outside the grasped half-width
    inner_y = max(0.5 * width + safety_margin, open_half - 0.004)
    outer_y = inner_y + spec.finger_thick
    half_z = 0.5 * spec.finger_depth + safety_margin

    in_finger_x = (x >= -spec.finger_back) & (x <= spec.finger_tips + safety_margin)
    in_finger_z = np.abs(z) <= half_z
    in_left_finger = in_finger_x & in_finger_z & (y >= -outer_y) & (y <= -inner_y)
    if in_left_finger.any():
        return True
    in_right_finger = in_finger_x & in_finger_z & (y >= inner_y) & (y <= outer_y)
    if in_right_finger.any():
        return True

    # Palm / wrist housing behind the fingers
    in_palm = (
        (x >= -(spec.finger_back + spec.palm_len))
        & (x <= -spec.finger_back + safety_margin)
        & (np.abs(y) <= 0.5 * spec.palm_width + safety_margin)
        & (np.abs(z) <= 0.5 * spec.palm_height + safety_margin)
    )
    return bool(in_palm.any())


def _heuristic_part_tag(
    pos: np.ndarray,
    approach: np.ndarray,
    width: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> str:
    """Geometry-based MVP part tag (can be refined later by Stage 3 VLM segmentation)."""
    ext = np.maximum(bounds_max - bounds_min, 1e-6)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    z_norm = float((pos[2] - bounds_min[2]) / ext[2])
    r_xy = float(np.linalg.norm(pos[:2] - center_xy))
    r_max = 0.5 * float(np.hypot(ext[0], ext[1]))

    if z_norm >= 0.55 and approach[2] < -0.5 and width < 0.35 * min(ext[0], ext[1]):
        return "rim"
    if r_max > 1e-6 and r_xy > 0.55 * r_max and width < 0.45 * min(ext[0], ext[1]):
        return "protrusion"
    if z_norm >= 0.78:
        return "body_top"
    if z_norm <= 0.25:
        return "body_low"
    if approach[2] < -0.5:
        return "body_top"
    return "body_side"


def _nms_grasps(
    candidates: list[dict[str, Any]],
    max_grasps: int,
    pos_thresh: float = 0.015,
    angle_thresh_deg: float = 25.0,
) -> list[dict[str, Any]]:
    """Greedy score-ordered NMS keeping diverse grasps separated by translation or orientation."""
    ordered = sorted(candidates, key=lambda g: g["score"], reverse=True)
    kept: list[dict[str, Any]] = []
    kept_pos: list[np.ndarray] = []
    kept_rot: list[np.ndarray] = []
    cos_ang = math.cos(math.radians(angle_thresh_deg))

    for cand in ordered:
        if len(kept) >= max_grasps:
            break
        p = np.asarray(cand["position"], dtype=np.float64)
        a = np.asarray(cand["approach_dir"], dtype=np.float64)
        c = np.asarray(cand["closing_axis"], dtype=np.float64)
        too_close = False
        for kp, (ka, kc) in zip(kept_pos, kept_rot):
            if np.linalg.norm(p - kp) < pos_thresh:
                # Parallel jaw is 180-deg symmetric in closing_axis
                if float(np.dot(a, ka)) >= cos_ang and abs(float(np.dot(c, kc))) >= cos_ang:
                    too_close = True
                    break
        if not too_close:
            kept.append(cand)
            kept_pos.append(p)
            kept_rot.append((a, c))

    for i, g in enumerate(kept):
        g["id"] = f"g{i:03d}"
    return kept


# ---------------------------------------------------------------------------
# Main Stage 1 antipodal grasp sampler & annotator
# ---------------------------------------------------------------------------


def sample_antipodal_grasps(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    gripper_type: str = "dex1_1",
    n_surface_samples: int = 250,
    n_approach_angles: int = 8,
    max_grasps: int = 24,
    friction_mu: float = 0.4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample collision-free antipodal grasps in the object local frame."""
    if gripper_type not in GRIPPER_SPECS:
        raise ValueError(f"Unknown gripper_type={gripper_type!r}; available: {sorted(GRIPPER_SPECS)}")
    spec = GRIPPER_SPECS[gripper_type]

    tris, tri_normals, areas, tri_centers = triangle_geometry(vertices, faces)
    if len(tris) == 0:
        return []

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    ext = bounds_max - bounds_min
    diag = max(float(np.linalg.norm(ext)), 1e-6)
    com = (tri_centers * areas[:, None]).sum(axis=0) / areas.sum()

    # Ensure normals point predominantly outward (Divergence Theorem: sum(A * ((c - com) . n)) == 3 * signed_volume)
    signed_vol_proxy = float(np.sum(areas * np.einsum("ij,ij->i", tri_centers - com, tri_normals)))
    if signed_vol_proxy < -1e-12:
        tri_normals = -tri_normals

    # Precompute triangle bounding spheres for fast ray culling
    tri_radii = np.linalg.norm(tris - tri_centers[:, None, :], axis=2).max(axis=1)

    rng = np.random.default_rng(seed)
    pts, nrms, _ = sample_surface_points(tris, tri_normals, areas, n_surface_samples, rng)

    # Dense point cloud for 3D gripper collision checking (vertices + sampled surface points)
    n_col_pts = min(max(600, len(vertices)), 2500)
    col_surface, _, _ = sample_surface_points(tris, tri_normals, areas, n_col_pts, rng)
    collision_pts = np.vstack([vertices, col_surface]) if len(vertices) <= 1500 else col_surface

    cos_friction = 1.0 / math.sqrt(1.0 + friction_mu * friction_mu)
    raw_candidates: list[dict[str, Any]] = []

    angles = np.linspace(0.0, 2.0 * np.pi, n_approach_angles, endpoint=False)

    for p1, n1 in zip(pts, nrms):
        ray_dir = -n1
        ray_origin = p1 + 1e-4 * ray_dir
        hits = _ray_mesh_antipodal_hits(
            ray_origin,
            ray_dir,
            tris,
            tri_normals,
            tri_centers,
            tri_radii,
            min_dist=spec.min_width,
            max_dist=spec.max_width,
            cos_friction=cos_friction,
        )
        if not hits:
            continue

        # Take the closest valid antipodal hit (and if there is a wider full-body hit, also test the farthest)
        selected_hits = [hits[0]]
        if len(hits) > 1 and hits[-1][0] - hits[0][0] > 0.01:
            selected_hits.append(hits[-1])

        for _t_hit, p2, n2 in selected_hits:
            width = float(np.linalg.norm(p2 - p1))
            if width < spec.min_width or width > spec.max_width:
                continue
            p_mid = 0.5 * (p1 + p2)
            closing = (p2 - p1) / width
            # Antipodal quality in [0, 1]
            c1 = float(np.dot(-n1, closing))
            c2 = float(np.dot(n2, closing))
            antipodal_q = min(c1, c2)
            if antipodal_q < cos_friction:
                continue

            # Construct orthonormal basis (u, v) in plane perpendicular to closing axis
            ref = np.array([0.0, 0.0, 1.0]) if abs(closing[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(closing, ref)
            u /= np.linalg.norm(u)
            v = np.cross(closing, u)

            # CoM proximity score (closer to CoM => lower gravity torque)
            com_dist = float(np.linalg.norm(p_mid - com))
            com_score = math.exp(-2.5 * (com_dist / diag) ** 2)

            # Width margin score (prefer comfortably inside [min_width, max_width])
            margin_hi = (spec.max_width - width) / max(spec.max_width - spec.min_width, 1e-6)
            width_score = 0.5 + 0.5 * float(np.clip(margin_hi, 0.0, 1.0))

            for ang in angles:
                approach = math.cos(ang) * u + math.sin(ang) * v
                # Reject approaches coming strongly from underneath (z > +0.35 in upright local frame)
                if approach[2] > 0.35:
                    continue
                rot_mat = _build_hand_rotation(approach, closing)
                grasp_pos = p_mid
                # For flat objects, shift top-down grasp_point slightly back along -approach so
                # fingertips clear the support_base plane while still overlapping the object >= 5 mm.
                if approach[2] < -0.5:
                    tip_z = float(p_mid[2] + spec.finger_tips * approach[2])
                    min_tip_z = float(bounds_min[2]) + 0.003
                    if tip_z < min_tip_z:
                        shift = (min_tip_z - tip_z) / float(-approach[2])
                        if shift > spec.finger_tips - 0.005:
                            continue
                        grasp_pos = p_mid - shift * approach
                if _gripper_collides(collision_pts, grasp_pos, rot_mat, width, spec):
                    continue

                # Tabletop friendliness: slight preference for top-down / tilted-down approaches
                top_score = float(np.clip(0.5 - 0.5 * approach[2], 0.0, 1.0))
                total_score = 0.40 * antipodal_q + 0.30 * com_score + 0.15 * width_score + 0.15 * top_score
                tag = _heuristic_part_tag(grasp_pos, approach, width, bounds_min, bounds_max)

                raw_candidates.append(
                    {
                        "id": "",
                        "position": [round(float(x), 5) for x in grasp_pos],
                        "quaternion_wxyz": _rot_to_wxyz(rot_mat),
                        "approach_dir": [round(float(x), 5) for x in rot_mat[:, 0]],
                        "closing_axis": [round(float(x), 5) for x in rot_mat[:, 1]],
                        "width": round(float(width), 5),
                        "part_tag": tag,
                        "tag_source": "geometry",
                        "score": round(float(total_score), 4),
                        "source": "antipodal_geom",
                        "verified": False,
                        "sim_score": None,
                    }
                )

    return _nms_grasps(raw_candidates, max_grasps=max_grasps)


def annotate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    asset_id: str = "unknown",
    gripper_types: tuple[str, ...] = ("dex1_1",),
    metadata: dict[str, Any] | None = None,
    n_surface_samples: int = 250,
    n_approach_angles: int = 8,
    max_grasps: int = 24,
    friction_mu: float = 0.4,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a full affordance.json dictionary from an in-memory triangle mesh."""
    tris, _, areas, centers = triangle_geometry(vertices, faces)
    if len(vertices) > 0:
        b_min = vertices.min(axis=0)
        b_max = vertices.max(axis=0)
        ext = b_max - b_min
    else:
        b_min = np.zeros(3)
        b_max = np.zeros(3)
        ext = np.zeros(3)

    grasps_by_gripper: dict[str, list[dict[str, Any]]] = {}
    for gtype in gripper_types:
        grasps_by_gripper[gtype] = sample_antipodal_grasps(
            vertices,
            faces,
            gripper_type=gtype,
            n_surface_samples=n_surface_samples,
            n_approach_angles=n_approach_angles,
            max_grasps=max_grasps,
            friction_mu=friction_mu,
            seed=seed,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "mesh_hash": compute_mesh_hash(vertices, faces),
        "scale_ref": 1.0,
        "mesh_stats": {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "valid_triangles": int(len(tris)),
        },
        "bounds": {
            "min": [round(float(x), 5) for x in b_min],
            "max": [round(float(x), 5) for x in b_max],
            "extents": [round(float(x), 5) for x in ext],
        },
        "symmetry": detect_symmetry(vertices),
        "functional_frames": extract_functional_frames(vertices, areas, centers, metadata=metadata),
        "grasps": grasps_by_gripper,
    }


def annotate_asset(
    path: Path,
    *,
    output: Path | None = None,
    asset_id: str | None = None,
    gripper_types: tuple[str, ...] = ("dex1_1",),
    n_surface_samples: int = 250,
    n_approach_angles: int = 8,
    max_grasps: int = 24,
    friction_mu: float = 0.4,
    seed: int = 0,
) -> dict[str, Any]:
    """Load an asset directory or mesh file, annotate affordances, and optionally save JSON."""
    p = Path(path)
    mesh_path = resolve_mesh_path(p)
    asset_dir = p if p.is_dir() else p.parent

    metadata = None
    meta_file = asset_dir / "metadata.json"
    if meta_file.is_file():
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None

    if asset_id is None:
        asset_id = str(asset_dir.name if p.is_dir() else mesh_path.stem)

    vertices, faces = load_mesh(mesh_path)
    result = annotate_mesh(
        vertices,
        faces,
        asset_id=asset_id,
        gripper_types=gripper_types,
        metadata=metadata,
        n_surface_samples=n_surface_samples,
        n_approach_angles=n_approach_angles,
        max_grasps=max_grasps,
        friction_mu=friction_mu,
        seed=seed,
    )
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


VALID_TAG_SOURCES = frozenset({"geometry", "vlm", "manual"})
VALID_SYMMETRY_MODES = frozenset({"continuous", "discrete", "none"})

SHARPA_WAVE_DOF = 22
SHARPA_WAVE_GRASP_TYPES = frozenset({"power_cylindrical", "power_spherical", "tripod_pinch", "handle_hook"})
SHARPA_WAVE_HAND_SIDES = frozenset({"left", "right"})
SHARPA_WAVE_EE_FRAMES = {"left": "left_hand_flange", "right": "right_hand_flange"}
SHARPA_WAVE_FINGERS = frozenset({"thumb", "index", "middle", "ring", "pinky"})
SHARPA_WAVE_MCP_AA_INDICES = (6, 10, 14, 19)
SHARPA_WAVE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "real2sim" / "robot" / "g1d_sharpa_wave" / "model_contract.json"
)

_SHARPA_LIMITS_CACHE: dict[str, tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]] = {}


def load_sharpa_wave_joint_limits(
    side: str = "left",
    contract_path: Path | None = None,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    """Return ``(joint_names, lower_limits, upper_limits)`` for Sharpa Wave from model_contract.json."""
    if side not in SHARPA_WAVE_HAND_SIDES:
        raise ValueError(f"Unknown hand side={side!r}; expected one of {sorted(SHARPA_WAVE_HAND_SIDES)}")
    path = Path(contract_path) if contract_path is not None else SHARPA_WAVE_CONTRACT_PATH
    cache_key = f"{path.resolve()}::{side}"
    if cache_key in _SHARPA_LIMITS_CACHE:
        return _SHARPA_LIMITS_CACHE[cache_key]
    raw = json.loads(path.read_text(encoding="utf-8"))
    hand = raw["hands"][side]
    joints = tuple(str(j) for j in hand["joints"])
    limits = hand["limits"]
    lowers = tuple(float(limits[j]["lower"]) for j in joints)
    uppers = tuple(float(limits[j]["upper"]) for j in joints)
    result = (joints, lowers, uppers)
    _SHARPA_LIMITS_CACHE[cache_key] = result
    return result


def _validate_sharpa_wave_grasp(
    g: dict[str, Any],
    prefix: str,
    is_finite_num: Any,
    errors: list[str],
    *,
    contract_path: Path | None = None,
) -> None:
    """Validate Sharpa Wave dexterous-hand specific grasp fields."""
    hand_side = g.get("hand_side")
    if hand_side not in SHARPA_WAVE_HAND_SIDES:
        errors.append(f"{prefix}.hand_side={hand_side!r} not in {sorted(SHARPA_WAVE_HAND_SIDES)}")
        side_for_limits = "left"
    else:
        side_for_limits = str(hand_side)
        expected_ee = SHARPA_WAVE_EE_FRAMES[side_for_limits]
        ee_frame = g.get("ee_frame")
        if ee_frame != expected_ee:
            errors.append(f"{prefix}.ee_frame={ee_frame!r} does not match hand_side={hand_side!r} ({expected_ee!r})")

    grasp_type = g.get("grasp_type")
    if grasp_type not in SHARPA_WAVE_GRASP_TYPES:
        errors.append(f"{prefix}.grasp_type={grasp_type!r} not in {sorted(SHARPA_WAVE_GRASP_TYPES)}")

    w_pos = g.get("wrist_position")
    if not isinstance(w_pos, list) or len(w_pos) != 3 or not all(is_finite_num(v) for v in w_pos):
        errors.append(f"{prefix}.wrist_position must be 3 finite numbers")

    w_quat = g.get("wrist_quaternion_wxyz")
    if not isinstance(w_quat, list) or len(w_quat) != 4 or not all(is_finite_num(v) for v in w_quat):
        errors.append(f"{prefix}.wrist_quaternion_wxyz must be 4 finite numbers")
    else:
        wnorm = math.sqrt(sum(float(v) * float(v) for v in w_quat))
        if abs(wnorm - 1.0) > 1e-2:
            errors.append(f"{prefix}.wrist_quaternion_wxyz not unit length (norm={wnorm:.4f})")

    for vec_key in ("approach_dir", "closing_axis", "thumb_dir"):
        if vec_key in g:
            vec = g.get(vec_key)
            if not isinstance(vec, list) or len(vec) != 3 or not all(is_finite_num(v) for v in vec):
                errors.append(f"{prefix}.{vec_key} must be 3 finite numbers")
            else:
                vnorm = math.sqrt(sum(float(v) * float(v) for v in vec))
                if abs(vnorm - 1.0) > 1e-2:
                    errors.append(f"{prefix}.{vec_key} not unit length (norm={vnorm:.4f})")

    joints, lowers, uppers = load_sharpa_wave_joint_limits(side=side_for_limits, contract_path=contract_path)
    tol = 1e-4
    for q_key in ("preshape_qpos", "target_qpos"):
        q = g.get(q_key)
        if not isinstance(q, list) or len(q) != SHARPA_WAVE_DOF or not all(is_finite_num(v) for v in q):
            errors.append(f"{prefix}.{q_key} must be {SHARPA_WAVE_DOF} finite numbers")
        else:
            for j_idx, (val, lo, hi, jname) in enumerate(zip(q, lowers, uppers, joints)):
                fv = float(val)
                if fv < lo - tol or fv > hi + tol:
                    errors.append(
                        f"{prefix}.{q_key}[{j_idx}] ({jname})={fv:.4f} outside contract limits [{lo:.4f}, {hi:.4f}]"
                    )

    sched = g.get("close_schedule")
    if not isinstance(sched, dict):
        errors.append(f"{prefix}.close_schedule must be an object")
    else:
        tfj = sched.get("thumb_first_joints")
        if not isinstance(tfj, int) or isinstance(tfj, bool) or tfj < 0 or tfj > SHARPA_WAVE_DOF:
            errors.append(f"{prefix}.close_schedule.thumb_first_joints={tfj!r} must be int in [0, {SHARPA_WAVE_DOF}]")
        tlr = sched.get("thumb_lead_ratio")
        if not is_finite_num(tlr) or float(tlr) < 0.0 or float(tlr) > 1.0:
            errors.append(f"{prefix}.close_schedule.thumb_lead_ratio={tlr!r} must be in [0.0, 1.0]")

    cf = g.get("contact_fingers")
    if (
        not isinstance(cf, list)
        or len(cf) == 0
        or not all(isinstance(f, str) and f in SHARPA_WAVE_FINGERS for f in cf)
        or len(set(cf)) != len(cf)
    ):
        errors.append(f"{prefix}.contact_fingers must be a non-empty unique list of {sorted(SHARPA_WAVE_FINGERS)}")


def validate_affordance(
    data: Any,
    *,
    expected_mesh_hash: str | None = None,
    require_verified: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate an affordance.json payload against schema_version=1 invariants.

    Returns ``(errors, warnings)``. Errors indicate structural or mathematical
    corruption (non-unit quaternions, NaN/Inf, missing required keys, stale
    ``mesh_hash`` when checked); warnings cover soft operational advisories
    (e.g. zero grasps on an oversized object, or unverified grasps when
    ``require_verified=True``).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return (["root payload must be a JSON object"], warnings)

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={data.get('schema_version')!r} (expected {SCHEMA_VERSION})")

    asset_id = data.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        errors.append("missing or empty asset_id")

    mesh_hash = data.get("mesh_hash")
    if not isinstance(mesh_hash, str) or not mesh_hash.startswith("sha256:") or len(mesh_hash) <= 7:
        errors.append(f"invalid mesh_hash={mesh_hash!r}")
    elif expected_mesh_hash is not None and mesh_hash != expected_mesh_hash:
        errors.append(f"stale mesh_hash: got {mesh_hash}, expected {expected_mesh_hash}")

    def _is_finite_num(v: Any) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))

    scale_ref = data.get("scale_ref")
    if not _is_finite_num(scale_ref) or float(scale_ref) <= 0.0:
        errors.append(f"invalid scale_ref={scale_ref!r}")

    bounds = data.get("bounds")
    if not isinstance(bounds, dict):
        errors.append("missing bounds object")
    else:
        b_min = bounds.get("min")
        b_max = bounds.get("max")
        ext = bounds.get("extents")
        for key, vec in (("min", b_min), ("max", b_max), ("extents", ext)):
            if not isinstance(vec, list) or len(vec) != 3 or not all(_is_finite_num(v) for v in vec):
                errors.append(f"bounds.{key} must be 3 finite numbers")
        if (
            isinstance(b_min, list)
            and isinstance(b_max, list)
            and len(b_min) == 3
            and len(b_max) == 3
            and all(_is_finite_num(v) for v in (*b_min, *b_max))
        ):
            if any(float(lo) > float(hi) + 1e-6 for lo, hi in zip(b_min, b_max)):
                errors.append(f"bounds.min {b_min} exceeds bounds.max {b_max}")

    sym = data.get("symmetry")
    if not isinstance(sym, dict):
        errors.append("missing symmetry object")
    elif sym.get("z_axis") not in VALID_SYMMETRY_MODES:
        errors.append(f"invalid symmetry.z_axis={sym.get('z_axis')!r}")

    grasps = data.get("grasps")
    if not isinstance(grasps, dict) or not grasps:
        errors.append("missing or empty grasps mapping")
    else:
        total_grasps = 0
        total_verified = 0
        for gtype, glist in grasps.items():
            if not isinstance(glist, list):
                errors.append(f"grasps.{gtype} must be a list")
                continue
            total_grasps += len(glist)
            for i, g in enumerate(glist):
                prefix = f"grasps.{gtype}[{i}]"
                if not isinstance(g, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                pos = g.get("position")
                if not isinstance(pos, list) or len(pos) != 3 or not all(_is_finite_num(v) for v in pos):
                    errors.append(f"{prefix}.position must be 3 finite numbers")
                quat = g.get("quaternion_wxyz")
                if not isinstance(quat, list) or len(quat) != 4 or not all(_is_finite_num(v) for v in quat):
                    errors.append(f"{prefix}.quaternion_wxyz must be 4 finite numbers")
                else:
                    qnorm = math.sqrt(sum(float(v) * float(v) for v in quat))
                    if abs(qnorm - 1.0) > 1e-2:
                        errors.append(f"{prefix}.quaternion_wxyz not unit length (norm={qnorm:.4f})")
                width = g.get("width")
                if not _is_finite_num(width) or float(width) <= 0.0:
                    errors.append(f"{prefix}.width must be a positive finite number")
                score = g.get("score")
                if not _is_finite_num(score):
                    errors.append(f"{prefix}.score must be a finite number")
                tag = g.get("part_tag")
                if not isinstance(tag, str) or not tag.strip():
                    errors.append(f"{prefix}.part_tag must be a non-empty string")
                tag_src = g.get("tag_source", "geometry")
                if tag_src not in VALID_TAG_SOURCES:
                    errors.append(f"{prefix}.tag_source={tag_src!r} not in {sorted(VALID_TAG_SOURCES)}")
                if not isinstance(g.get("verified", False), bool):
                    errors.append(f"{prefix}.verified must be a boolean")
                if gtype == "sharpa_wave":
                    _validate_sharpa_wave_grasp(g, prefix, _is_finite_num, errors)
                if g.get("verified"):
                    total_verified += 1
        if total_grasps == 0:
            warnings.append("no grasp candidates generated (object may exceed gripper max_width)")
        elif require_verified and total_verified == 0:
            warnings.append(f"0/{total_grasps} grasps are Stage-2 physics verified")

    return errors, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenFlywheel object affordance annotator (Stage 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Annotate a mesh file or asset directory")
    p_ann.add_argument("path", type=Path, help="Mesh file (.obj/.stl/.usd[a|z]) or asset directory")
    p_ann.add_argument("--output", "-o", type=Path, default=None, help="Output affordance.json path")
    p_ann.add_argument("--asset-id", default=None, help="Override asset_id recorded in JSON")
    p_ann.add_argument("--gripper", action="append", dest="grippers", help="Gripper type (default: dex1_1)")
    p_ann.add_argument("--samples", type=int, default=250, help="Surface points to sample")
    p_ann.add_argument("--angles", type=int, default=8, help="Approach angles per antipodal pair")
    p_ann.add_argument("--max-grasps", type=int, default=24, help="Maximum grasps per gripper after NMS")
    p_ann.add_argument("--mu", type=float, default=0.4, help="Friction coefficient for antipodal cone")
    p_ann.add_argument("--seed", type=int, default=0, help="Random seed")

    p_ins = sub.add_parser("inspect", help="Print a concise summary of an affordance.json file")
    p_ins.add_argument("path", type=Path, help="Path to affordance.json")
    p_ins.add_argument("--validate", action="store_true", help="Validate schema invariants (exit 1 on error)")

    p_bat = sub.add_parser("batch", help="Batch-annotate objects from a pool JSON (e.g. pick_anything_pool.json)")
    p_bat.add_argument("--pool", type=Path, default=Path(__file__).with_name("pick_anything_pool.json"))
    p_bat.add_argument("--assets-root", type=Path, default=Path("Assets"), help="Path to Assets directory")
    p_bat.add_argument(
        "--output-dir", "-o", type=Path, required=True, help="Output root (<dir>/<cat>/<idx:05d>/affordance.json)"
    )
    p_bat.add_argument(
        "--commercial-only", action="store_true", help="Exclude noncommercial categories (geniesim_, partnet_)"
    )
    p_bat.add_argument("--limit", type=int, default=0, help="Max number of objects to annotate (0 = all)")
    p_bat.add_argument("--offset", type=int, default=0, help="Start offset into pool")
    p_bat.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Output manifest JSON path (default: <output-dir>/manifest.json)",
    )
    p_bat.add_argument("--skip-existing", action="store_true", help="Skip objects whose affordance.json already exists")
    p_bat.add_argument("--gripper", action="append", dest="grippers", help="Gripper type (default: dex1_1)")
    p_bat.add_argument("--samples", type=int, default=250)
    p_bat.add_argument("--angles", type=int, default=8)
    p_bat.add_argument("--max-grasps", type=int, default=24)
    p_bat.add_argument("--mu", type=float, default=0.4)
    p_bat.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.cmd == "annotate":
        grippers = tuple(args.grippers) if args.grippers else ("dex1_1",)
        out = args.output
        if out is None and args.path.is_dir():
            out = args.path / "affordance.json"
        res = annotate_asset(
            args.path,
            output=out,
            asset_id=args.asset_id,
            gripper_types=grippers,
            n_surface_samples=args.samples,
            n_approach_angles=args.angles,
            max_grasps=args.max_grasps,
            friction_mu=args.mu,
            seed=args.seed,
        )
        summary = {
            "asset_id": res["asset_id"],
            "mesh_hash": res["mesh_hash"],
            "extents": res["bounds"]["extents"],
            "symmetry": res["symmetry"],
            "grasp_counts": {k: len(v) for k, v in res["grasps"].items()},
            "output": str(out) if out else None,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if _PXR_APP is not None:
            _PXR_APP.close()
        return 0

    if args.cmd == "inspect":
        data = json.loads(args.path.read_text(encoding="utf-8"))
        for gtype, glist in data.get("grasps", {}).items():
            verified = sum(1 for g in glist if g.get("verified"))
            tags: dict[str, int] = {}
            tag_sources: dict[str, int] = {}
            for g in glist:
                tags[g.get("part_tag", "unknown")] = tags.get(g.get("part_tag", "unknown"), 0) + 1
                src = g.get("tag_source", "geometry")
                tag_sources[src] = tag_sources.get(src, 0) + 1
            print(
                f"{data.get('asset_id')} [{gtype}]: {len(glist)} grasps ({verified} verified), "
                f"tags={tags}, tag_sources={tag_sources}"
            )
        if args.validate:
            errs, warns = validate_affordance(data)
            for w in warns:
                print(f"  warning: {w}")
            for e in errs:
                print(f"  error: {e}")
            return 1 if errs else 0
        return 0

    if args.cmd == "batch":
        rc = _run_batch(args)
        if _PXR_APP is not None:
            _PXR_APP.close()
        return rc

    return 1


def _resolve_pool_asset_dir(assets_root: Path, cat: str, idx: int) -> Path | None:
    """Resolve Rigid/<cat>/<idx:05d> across standard OpenFlywheel asset libraries."""
    idx_str = f"{int(idx):05d}"
    for lib in ("RoboDojo", "Real2Sim", "FMB", "Blocks"):
        cand = assets_root / "Object" / lib / "Rigid" / cat / idx_str
        if cand.is_dir() and any((cand / f).is_file() for f in ("object.usd", "object.usdz", "object.usda")):
            return cand
    return None


def _run_batch(args: argparse.Namespace) -> int:
    pool_data = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    objects = pool_data["objects"] if isinstance(pool_data, dict) else pool_data
    if args.commercial_only:
        objects = [o for o in objects if not str(o["cat"]).startswith(("geniesim_", "partnet_"))]
    if args.offset > 0:
        objects = objects[args.offset :]
    if args.limit > 0:
        objects = objects[: args.limit]

    grippers = tuple(args.grippers) if args.grippers else ("dex1_1",)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    ok_count = 0
    zero_count = 0
    err_count = 0
    skip_count = 0

    for obj in objects:
        cat = str(obj["cat"])
        idx = int(obj["idx"])
        asset_id = f"Rigid/{cat}/{idx:05d}"
        out_path = out_root / cat / f"{idx:05d}" / "affordance.json"
        asset_dir = _resolve_pool_asset_dir(Path(args.assets_root), cat, idx)
        if asset_dir is None:
            manifest.append({"asset_id": asset_id, "status": "missing_asset", "affordance_path": str(out_path)})
            err_count += 1
            continue
        if args.skip_existing and out_path.is_file():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            n_g = len((existing.get("grasps") or {}).get(grippers[0], []))
            manifest.append(
                {
                    "asset_id": asset_id,
                    "asset_dir": str(asset_dir),
                    "affordance_path": str(out_path),
                    "status": "skipped",
                    "grasps": n_g,
                }
            )
            skip_count += 1
            continue
        try:
            res = annotate_asset(
                asset_dir,
                output=out_path,
                asset_id=asset_id,
                gripper_types=grippers,
                n_surface_samples=args.samples,
                n_approach_angles=args.angles,
                max_grasps=args.max_grasps,
                friction_mu=args.mu,
                seed=args.seed,
            )
            n_g = len(res["grasps"].get(grippers[0], []))
            if n_g > 0:
                ok_count += 1
            else:
                zero_count += 1
            manifest.append(
                {
                    "asset_id": asset_id,
                    "asset_dir": str(asset_dir),
                    "affordance_path": str(out_path),
                    "status": "ok" if n_g > 0 else "zero_grasps",
                    "grasps": n_g,
                    "symmetry": res["symmetry"]["z_axis"],
                    "extents": res["bounds"]["extents"],
                    "vertices": res["mesh_stats"]["vertices"],
                    "faces": res["mesh_stats"]["faces"],
                }
            )
        except Exception as exc:
            err_count += 1
            manifest.append(
                {
                    "asset_id": asset_id,
                    "asset_dir": str(asset_dir),
                    "affordance_path": str(out_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest_path = Path(args.manifest_path) if getattr(args, "manifest_path", None) else (out_root / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[affordance batch] total={len(objects)} ok={ok_count} zero={zero_count} "
        f"skipped={skip_count} err={err_count} manifest={manifest_path}"
    )
    return 0 if err_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
