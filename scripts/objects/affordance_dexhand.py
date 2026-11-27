"""Stage 1 offline dexterous-hand (Sharpa Wave) affordance annotator (CPU-only).

Generates 22-DoF Sharpa Wave grasp candidates in the canonical object local frame
(`scale_ref = 1.0`, `left_hand_flange`) and validates them against:
  - `real2sim/robot/g1d_sharpa_wave/model_contract.json` joint limits
  - `|MCP_AA| <= 0.04 rad` inter-finger collision safety window
  - URDF FK 18-capsule (wrist + palm + 15 phalanges) zero-penetration & tabletop clearance
  - Bucketed pose NMS across 4 task-oriented grasp primitives:
      1. `power_cylindrical` (side wrap, `thumb_first_joints=5`)
      2. `power_spherical`   (top/oblique 5-finger dome wrap)
      3. `tripod_pinch`      (refitted wide-opening >= 75 mm pad clearance, ring/pinky tucked)
      4. `handle_hook`       (handle/loop hook wrap for mugs/baskets or Stage 3 `handle` tags)

Also provides deterministic left -> right hand mirroring locked to < 1 mm FK error
against `real2sim/robot/g1d_sharpa_wave/robot.urdf`.

CLI usage:
    python scripts/objects/affordance_dexhand.py annotate path/to/object.obj -o affordance.json
    python scripts/objects/affordance_dexhand.py annotate path/to/object.obj -o affordance.json --merge
    python scripts/objects/affordance_dexhand.py verify-pinch
    python scripts/objects/affordance_dexhand.py inspect affordance.json --validate
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dexhands import kinematics as dex_kin  # noqa: E402
from scripts.objects import affordance as aff  # noqa: E402

SHARPA_URDF_PATH = ROOT / "real2sim" / "robot" / "g1d_sharpa_wave" / "robot.urdf"
SHARPA_CONTRACT_PATH = aff.SHARPA_WAVE_CONTRACT_PATH

SHARPA_DOF = aff.SHARPA_WAVE_DOF
MCP_AA_INDICES = aff.SHARPA_WAVE_MCP_AA_INDICES  # (6, 10, 14, 19)
MCP_AA_SAFE_LIMIT = 0.04  # rad (hard inter-finger collision safety bound)

# Empirically verified against real2sim/robot/g1d_sharpa_wave/robot.urdf FK:
# right-hand joint axes in robot.urdf are already mirrored relative to right_hand_flange,
# so q_right = +1.0 * q_left reproduces diag(1, -1, 1) Y-reflection to ~1.2e-6 m.
RIGHT_JOINT_SIGNS = np.ones(SHARPA_DOF, dtype=np.float64)
HAND_MIRROR_Y = np.diag([1.0, -1.0, 1.0])

# Elastomer pad centers & outward contact normals in each `*_elastomer` link frame
# (derived from real2sim/robot/g1d_sharpa_wave/meshes/left/*_elastomer_*.STL).
THUMB_PAD_LOCAL = np.array([0.0, -0.0210, -0.0078], dtype=np.float64)
FINGER_PAD_LOCAL = np.array([0.0, -0.0172, -0.0069], dtype=np.float64)
PAD_NORMAL_LOCAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)

# Ring & pinky tucked pose used during tripod_pinch to clear tabletop & object body.
_TUCK_RING = (1.35, 0.0, 1.50, 1.15)
_TUCK_PINKY = (0.0, 1.35, 0.0, 1.50, 1.15)

# Refitted wide-opening tripod_pinch template (optimized via URDF FK with |MCP_AA| <= 0.04 rad):
#   - Open pad-to-pad distance: ~92.5 mm (true STL mesh-to-mesh clearance > 90 mm >= 75 mm)
#   - Closed pad-to-pad gap:    ~2.0 mm (smooth monotonic interpolation 0..90 mm)
#   - Index/middle MP joints stay > 60 mm clear of pinch corridor.
TRIPOD_PINCH_CLOSED_Q = np.array(
    [
        1.9199,
        0.166,
        -0.5236,
        0.337,
        0.05,
        1.321,
        0.003,
        0.14,
        0.10,
        1.350,
        0.003,
        0.14,
        0.10,
        *_TUCK_RING,
        *_TUCK_PINKY,
    ],
    dtype=np.float64,
)
TRIPOD_PINCH_OPEN_Q = np.array(
    [
        1.365,
        -0.123,
        -0.5236,
        -0.349,
        0.851,
        0.751,
        0.040,
        0.0,
        0.0,
        0.807,
        0.040,
        0.0,
        0.0,
        *_TUCK_RING,
        *_TUCK_PINKY,
    ],
    dtype=np.float64,
)

# Canonical left_hand_flange geometric anchors (matching cup_grasp_demo.py).
PALM_FINGERS_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
PALM_NORMAL_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
PALM_THUMB_AXIS = np.array([0.0, -1.0, 0.0], dtype=np.float64)
PALM_SURFACE_POINT = np.array([0.025, 0.0, 0.130], dtype=np.float64)

# Link-local capsule segments (p0, p1, radius) derived from Sharpa Wave collision STLs.
_CAPSULE_SPECS: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], float, bool], ...] = (
    # (link_suffix, p0, p1, radius, is_palm_or_wrist)
    ("hand_wrist", (0.0, 0.0, 0.004), (0.0, 0.0, 0.028), 0.019, True),
    ("hand_C_MC", (0.0, -0.020, 0.020), (0.0, -0.020, 0.072), 0.021, True),
    ("hand_C_MC", (0.0, 0.020, 0.020), (0.0, 0.020, 0.072), 0.021, True),
    ("thumb_MC", (0.012, -0.008, -0.008), (0.062, -0.010, -0.010), 0.013, False),
    ("thumb_PP", (0.012, 0.0, 0.0), (0.042, 0.0, 0.0), 0.010, False),
    ("thumb_DP", (0.005, -0.002, 0.0), (0.028, -0.002, 0.0), 0.009, False),
    ("index_PP", (0.012, 0.0, 0.0), (0.050, 0.0, 0.0), 0.0095, False),
    ("index_MP", (0.004, 0.0, 0.0), (0.033, 0.0, 0.0), 0.0085, False),
    ("index_DP", (0.004, -0.001, 0.0), (0.025, -0.001, 0.0), 0.0075, False),
    ("middle_PP", (0.012, 0.0, 0.0), (0.050, 0.0, 0.0), 0.0095, False),
    ("middle_MP", (0.004, 0.0, 0.0), (0.033, 0.0, 0.0), 0.0085, False),
    ("middle_DP", (0.004, -0.001, 0.0), (0.025, -0.001, 0.0), 0.0075, False),
    ("ring_PP", (0.012, 0.0, 0.0), (0.050, 0.0, 0.0), 0.0095, False),
    ("ring_MP", (0.004, 0.0, 0.0), (0.033, 0.0, 0.0), 0.0085, False),
    ("ring_DP", (0.004, -0.001, 0.0), (0.025, -0.001, 0.0), 0.0075, False),
    ("pinky_PP", (0.012, 0.0, 0.0), (0.050, 0.0, 0.0), 0.0095, False),
    ("pinky_MP", (0.004, 0.0, 0.0), (0.033, 0.0, 0.0), 0.0085, False),
    ("pinky_DP", (0.004, -0.001, 0.0), (0.025, -0.001, 0.0), 0.0075, False),
)

_PALM_CAPSULE_COUNT = sum(1 for spec in _CAPSULE_SPECS if spec[4])


@dataclass(frozen=True)
class _CompiledJoint:
    parent: str
    child: str
    origin: np.ndarray
    kind: str
    axis: np.ndarray
    q_idx: int | None


@dataclass(frozen=True)
class TripodPinchGeometry:
    """FK-derived reference geometry of the refitted tripod_pinch template in left_hand_flange."""

    open_qpos: np.ndarray
    closed_qpos: np.ndarray
    pinch_center: np.ndarray
    closing_axis: np.ndarray
    approach_dir: np.ndarray
    thumb_dir: np.ndarray
    open_pad_clearance_m: float
    closed_pad_gap_m: float
    mp_corridor_clearance_m: float


class SharpaWaveKinematics:
    """Fast pre-compiled CPU URDF forward kinematics for Sharpa Wave left & right hands."""

    def __init__(
        self,
        urdf_path: Path | None = None,
        contract_path: Path | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path is not None else SHARPA_URDF_PATH
        self.contract_path = Path(contract_path) if contract_path is not None else SHARPA_CONTRACT_PATH
        self._root = ET.parse(self.urdf_path).getroot()

        self.joints: dict[str, tuple[str, ...]] = {}
        self.lower: dict[str, np.ndarray] = {}
        self.upper: dict[str, np.ndarray] = {}
        self._chains: dict[str, list[_CompiledJoint]] = {}

        for side in ("left", "right"):
            names, lo, hi = aff.load_sharpa_wave_joint_limits(side=side, contract_path=self.contract_path)
            self.joints[side] = names
            self.lower[side] = np.asarray(lo, dtype=np.float64)
            self.upper[side] = np.asarray(hi, dtype=np.float64)
            self._chains[side] = self._compile_subtree(f"{side}_hand_flange", names)

        # Precompute exact per-link local reflection diagonal D_link = diag(R_l^T @ M_y @ R_r)
        # from the URDF so right-hand capsules match diag(1, -1, 1) Y-reflection to < 0.01 mm
        # (in robot.urdf palm links reflect Y, phalanges reflect Z, elastomer links reflect X).
        tf_l0 = self.fk(side="left")
        tf_r0 = self.fk(side="right")
        self._right_link_mirror: dict[str, np.ndarray] = {}
        for suffix, *_ in _CAPSULE_SPECS:
            if suffix in self._right_link_mirror:
                continue
            Rl = tf_l0[f"left_{suffix}"][:3, :3]
            Rr = tf_r0[f"right_{suffix}"][:3, :3]
            D = Rl.T @ HAND_MIRROR_Y @ Rr
            self._right_link_mirror[suffix] = np.sign(np.diag(D)).astype(np.float64)

        self._tripod_geom: TripodPinchGeometry | None = None

    def _compile_subtree(self, root_link: str, joint_names: tuple[str, ...]) -> list[_CompiledJoint]:
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        by_parent: dict[str, list[ET.Element]] = {}
        for j in self._root.findall("joint"):
            parent = j.find("parent").get("link")
            by_parent.setdefault(parent, []).append(j)

        chain: list[_CompiledJoint] = []
        queue = [root_link]
        visited = {root_link}
        while queue:
            cur = queue.pop(0)
            for j in by_parent.get(cur, ()):
                child = j.find("child").get("link")
                if child in visited:
                    continue
                visited.add(child)
                kind = str(j.get("type", "fixed"))
                axis_el = j.find("axis")
                axis = (
                    dex_kin.vector(axis_el.get("xyz"), (1.0, 0.0, 0.0))
                    if axis_el is not None
                    else np.array([1.0, 0.0, 0.0])
                )
                norm = float(np.linalg.norm(axis))
                if norm > 1e-12:
                    axis = axis / norm
                chain.append(
                    _CompiledJoint(
                        parent=cur,
                        child=child,
                        origin=dex_kin.origin_transform(j.find("origin")),
                        kind=kind,
                        axis=axis,
                        q_idx=name_to_idx.get(j.get("name")),
                    )
                )
                queue.append(child)
        return chain

    def clip_qpos(
        self,
        qpos: np.ndarray | list[float],
        *,
        side: str = "left",
        enforce_mcp_aa: bool = True,
    ) -> np.ndarray:
        """Clip 22-DoF joint vector to model_contract.json limits (and optional |MCP_AA| <= 0.04)."""
        q = np.asarray(qpos, dtype=np.float64).copy()
        if q.shape != (SHARPA_DOF,):
            raise ValueError(f"Expected qpos shape ({SHARPA_DOF},), got {q.shape}")
        q = np.clip(q, self.lower[side], self.upper[side])
        if enforce_mcp_aa:
            for idx in MCP_AA_INDICES:
                q[idx] = float(np.clip(q[idx], -MCP_AA_SAFE_LIMIT, MCP_AA_SAFE_LIMIT))
        return q

    def fk(self, qpos: np.ndarray | list[float] | None = None, *, side: str = "left") -> dict[str, np.ndarray]:
        """Compute 4x4 link transforms relative to `{side}_hand_flange`."""
        q = np.zeros(SHARPA_DOF, dtype=np.float64) if qpos is None else np.asarray(qpos, dtype=np.float64)
        flange = f"{side}_hand_flange"
        transforms: dict[str, np.ndarray] = {flange: np.eye(4, dtype=np.float64)}
        for step in self._chains[side]:
            T_parent = transforms[step.parent]
            if step.kind == "fixed" or step.q_idx is None:
                transforms[step.child] = T_parent @ step.origin
            else:
                angle = float(q[step.q_idx])
                if abs(angle) < 1e-12:
                    transforms[step.child] = T_parent @ step.origin
                else:
                    motion = np.eye(4, dtype=np.float64)
                    if step.kind in ("revolute", "continuous"):
                        motion[:3, :3] = dex_kin.rotation(step.axis, angle)
                    elif step.kind == "prismatic":
                        motion[:3, 3] = step.axis * angle
                    transforms[step.child] = T_parent @ step.origin @ motion
        return transforms

    def elastomer_pads(
        self, qpos: np.ndarray | list[float], *, side: str = "left"
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return `{finger: (pad_pos, pad_normal)}` in `{side}_hand_flange` frame."""
        tf = self.fk(qpos, side=side)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            T = tf[f"{side}_{finger}_elastomer"]
            local_p = THUMB_PAD_LOCAL if finger == "thumb" else FINGER_PAD_LOCAL
            p = T[:3, :3] @ local_p + T[:3, 3]
            n = T[:3, :3] @ PAD_NORMAL_LOCAL
            n_norm = float(np.linalg.norm(n))
            if n_norm > 1e-12:
                n = n / n_norm
            out[finger] = (p, n)
        return out

    def link_capsules(
        self, qpos: np.ndarray | list[float], *, side: str = "left"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(p0s, p1s, radii)` of 18 hand capsules in `{side}_hand_flange` frame."""
        tf = self.fk(qpos, side=side)
        n = len(_CAPSULE_SPECS)
        p0s = np.empty((n, 3), dtype=np.float64)
        p1s = np.empty((n, 3), dtype=np.float64)
        radii = np.empty(n, dtype=np.float64)
        is_right = side == "right"
        for i, (suffix, a_loc, b_loc, r, _is_palm) in enumerate(_CAPSULE_SPECS):
            T = tf[f"{side}_{suffix}"]
            a = np.asarray(a_loc, dtype=np.float64)
            b = np.asarray(b_loc, dtype=np.float64)
            if is_right:
                mirror = self._right_link_mirror[suffix]
                a = a * mirror
                b = b * mirror
            p0s[i] = T[:3, :3] @ a + T[:3, 3]
            p1s[i] = T[:3, :3] @ b + T[:3, 3]
            radii[i] = r
        return p0s, p1s, radii

    def tripod_pinch_geometry(self) -> TripodPinchGeometry:
        """Return cached FK-derived geometry for the refitted wide-opening tripod_pinch template."""
        if self._tripod_geom is not None:
            return self._tripod_geom
        q_closed = self.clip_qpos(TRIPOD_PINCH_CLOSED_Q, side="left", enforce_mcp_aa=True)
        q_open = self.clip_qpos(TRIPOD_PINCH_OPEN_Q, side="left", enforce_mcp_aa=True)

        pads_c = self.elastomer_pads(q_closed, side="left")
        p_th_c, n_th_c = pads_c["thumb"]
        p_f_c = 0.5 * (pads_c["index"][0] + pads_c["middle"][0])
        n_f_c = 0.5 * (pads_c["index"][1] + pads_c["middle"][1])
        pinch_center = 0.5 * (p_th_c + p_f_c)

        closing_axis = n_th_c - n_f_c
        closing_axis = closing_axis / max(float(np.linalg.norm(closing_axis)), 1e-12)

        # Approach direction: from wrist/palm toward pinch point, orthogonalized to closing_axis
        raw_app = pinch_center - np.array([0.02, -0.02, 0.08], dtype=np.float64)
        raw_app = raw_app - float(np.dot(raw_app, closing_axis)) * closing_axis
        approach_dir = raw_app / max(float(np.linalg.norm(raw_app)), 1e-12)
        thumb_dir = np.cross(approach_dir, closing_axis)
        thumb_dir = thumb_dir / max(float(np.linalg.norm(thumb_dir)), 1e-12)

        pads_o = self.elastomer_pads(q_open, side="left")
        p_th_o = pads_o["thumb"][0]
        p_ix_o = pads_o["index"][0]
        p_mi_o = pads_o["middle"][0]
        open_clear = float(min(np.linalg.norm(p_th_o - p_ix_o), np.linalg.norm(p_th_o - p_mi_o)))
        closed_gap = float(np.linalg.norm(p_th_c - p_f_c))

        tf_o = self.fk(q_open, side="left")
        mid_o = 0.5 * (p_th_o + 0.5 * (p_ix_o + p_mi_o))
        mp_clear = float(
            min(
                np.linalg.norm(tf_o["left_index_MP"][:3, 3] - mid_o),
                np.linalg.norm(tf_o["left_middle_MP"][:3, 3] - mid_o),
            )
        )

        self._tripod_geom = TripodPinchGeometry(
            open_qpos=q_open,
            closed_qpos=q_closed,
            pinch_center=pinch_center,
            closing_axis=closing_axis,
            approach_dir=approach_dir,
            thumb_dir=thumb_dir,
            open_pad_clearance_m=open_clear,
            closed_pad_gap_m=closed_gap,
            mp_corridor_clearance_m=mp_clear,
        )
        return self._tripod_geom


_DEFAULT_KIN: SharpaWaveKinematics | None = None


def get_sharpa_kinematics() -> SharpaWaveKinematics:
    """Return a lazily initialized singleton SharpaWaveKinematics instance."""
    global _DEFAULT_KIN
    if _DEFAULT_KIN is None:
        _DEFAULT_KIN = SharpaWaveKinematics()
    return _DEFAULT_KIN


def refit_tripod_pinch_template(
    kin: SharpaWaveKinematics | None = None,
    *,
    target_open_m: float = 0.092,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-solve wide-opening Sharpa Wave tripod_pinch (closed_q, open_q, open_clearance_m) via URDF FK."""
    kin = kin or get_sharpa_kinematics()
    lo = kin.lower["left"][:13].copy()
    hi = kin.upper["left"][:13].copy()
    for aa_idx in (6, 10):
        lo[aa_idx] = -MCP_AA_SAFE_LIMIT
        hi[aa_idx] = MCP_AA_SAFE_LIMIT
    tuck = np.array([*_TUCK_RING, *_TUCK_PINKY], dtype=np.float64)

    def _eval(q13: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        q22 = np.concatenate([q13, tuck])
        pads = kin.elastomer_pads(q22, side="left")
        tf = kin.fk(q22, side="left")
        return (
            pads["thumb"][0],
            pads["index"][0],
            pads["middle"][0],
            pads["thumb"][1],
            pads["index"][1],
            pads["middle"][1],
            tf,
        )

    def _res_closed(q13: np.ndarray) -> np.ndarray:
        p_th, p_ix, p_mi, n_th, n_ix, n_mi, _ = _eval(q13)
        p_f = 0.5 * (p_ix + p_mi)
        return np.concatenate(
            [
                (p_th - p_f) * 20.0,
                np.array([p_ix[0] - p_mi[0], p_ix[2] - p_mi[2]]) * 15.0,
                np.array([float(np.dot(n_th, n_ix)) + 1.0, float(np.dot(n_th, n_mi)) + 1.0]) * 2.0,
                np.array([max(0.0, 0.08 - p_f[0]) * 5.0, max(0.0, 0.14 - p_f[2]) * 5.0, q13[6], q13[10]]),
            ]
        )

    x0_c = np.clip(TRIPOD_PINCH_CLOSED_Q[:13], lo, hi)
    sol_c = least_squares(_res_closed, x0_c, bounds=(lo, hi), max_nfev=600)
    q_closed = kin.clip_qpos(np.concatenate([sol_c.x, tuck]), side="left", enforce_mcp_aa=True)
    p_th_c, p_ix_c, p_mi_c, n_th_c, n_ix_c, n_mi_c, _ = _eval(q_closed[:13])
    center = 0.5 * (p_th_c + 0.5 * (p_ix_c + p_mi_c))
    axis = n_th_c - 0.5 * (n_ix_c + n_mi_c)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    def _res_open(q13: np.ndarray) -> np.ndarray:
        p_th, p_ix, p_mi, n_th, n_ix, n_mi, tf = _eval(q13)
        p_f = 0.5 * (p_ix + p_mi)
        d = p_f - p_th
        proj = float(np.dot(d, axis))
        perp = d - proj * axis
        mid = 0.5 * (p_th + p_f)
        d_mp_ix = float(np.linalg.norm(tf["left_index_MP"][:3, 3] - mid))
        d_mp_mi = float(np.linalg.norm(tf["left_middle_MP"][:3, 3] - mid))
        return np.concatenate(
            [
                np.array([(proj - target_open_m) * 25.0]),
                perp * 12.0,
                (mid - center) * 8.0,
                np.array([float(np.dot(p_ix - p_mi, axis)), p_ix[2] - p_mi[2]]) * 15.0,
                np.array(
                    [
                        float(np.dot(n_th, axis)) - 0.85,
                        float(np.dot(n_ix, -axis)) - 0.85,
                        float(np.dot(n_mi, -axis)) - 0.85,
                    ]
                )
                * 1.5,
                np.array([max(0.0, 0.055 - d_mp_ix) * 20.0, max(0.0, 0.055 - d_mp_mi) * 20.0]),
            ]
        )

    x0_o = np.clip(TRIPOD_PINCH_OPEN_Q[:13], lo, hi)
    sol_o = least_squares(_res_open, x0_o, bounds=(lo, hi), max_nfev=800)
    q_open = kin.clip_qpos(np.concatenate([sol_o.x, tuck]), side="left", enforce_mcp_aa=True)
    pads_o = kin.elastomer_pads(q_open, side="left")
    open_clear = float(
        min(
            np.linalg.norm(pads_o["thumb"][0] - pads_o["index"][0]),
            np.linalg.norm(pads_o["thumb"][0] - pads_o["middle"][0]),
        )
    )
    return q_closed, q_open, open_clear


# ---------------------------------------------------------------------------
# Deterministic Left -> Right Mirroring
# ---------------------------------------------------------------------------


def _rot_to_quat_wxyz(rot_mat: np.ndarray) -> list[float]:
    q_xyzw = Rotation.from_matrix(rot_mat).as_quat()
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def _quat_wxyz_to_rot(q_wxyz: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def mirror_grasp_left_to_right(
    grasp: dict[str, Any],
    *,
    mirror_normal: np.ndarray | list[float] | None = None,
    mirror_origin: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Deterministically mirror a canonical `left` Sharpa Wave grasp to `right`.

    Reflects the left hand across the plane passing through `mirror_origin`
    (default: `grasp["position"]`) with unit normal `u = mirror_normal`
    (default: `closing_axis` for `tripod_pinch`/`power_spherical`, or
    `approach_dir` for `power_cylindrical`/`handle_hook`).

    Because `right_hand_flange` links in `robot.urdf` are exact Y-mirrors
    (`M_hand = diag(1, -1, 1)`) of `left_hand_flange` links when
    `q_right = RIGHT_JOINT_SIGNS * q_left`:
        R_wrist_R = M_u @ R_wrist_L @ M_hand    (det = (-1)*(+1)*(-1) = +1 in SO(3))
        p_wrist_R = c + M_u @ (p_wrist_L - c)
    Every link position in object frame satisfies:
        p_link_obj_R == c + M_u @ (p_link_obj_L - c)   (< 0.01 mm FK error).
    """
    if grasp.get("hand_side") != "left":
        raise ValueError(f"mirror_grasp_left_to_right requires hand_side='left', got {grasp.get('hand_side')!r}")
    out = dict(grasp)
    c = np.asarray(grasp["position"] if mirror_origin is None else mirror_origin, dtype=np.float64)
    if mirror_normal is None:
        gtype = str(grasp.get("grasp_type", ""))
        default_key = "approach_dir" if gtype in ("power_cylindrical", "handle_hook") else "closing_axis"
        u = np.asarray(grasp[default_key], dtype=np.float64)
    else:
        u = np.asarray(mirror_normal, dtype=np.float64)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-12:
        raise ValueError("mirror_normal must be non-zero")
    u = u / u_norm
    M_u = np.eye(3, dtype=np.float64) - 2.0 * np.outer(u, u)

    p_pos = np.asarray(grasp["position"], dtype=np.float64)
    p_pos_r = c + M_u @ (p_pos - c)
    R_pos_l = _quat_wxyz_to_rot(grasp["quaternion_wxyz"])
    R_pos_r = M_u @ R_pos_l @ HAND_MIRROR_Y

    p_w = np.asarray(grasp["wrist_position"], dtype=np.float64)
    p_w_r = c + M_u @ (p_w - c)
    R_w_l = _quat_wxyz_to_rot(grasp["wrist_quaternion_wxyz"])
    R_w_r = M_u @ R_w_l @ HAND_MIRROR_Y

    # Semantic frame columns [approach_dir, closing_axis, thumb_dir] match R_pos_r (right-handed SO(3))
    app_r = R_pos_r[:, 0]
    cls_r = R_pos_r[:, 1]
    thm_r = R_pos_r[:, 2]

    q_pre_r = RIGHT_JOINT_SIGNS * np.asarray(grasp["preshape_qpos"], dtype=np.float64)
    q_tgt_r = RIGHT_JOINT_SIGNS * np.asarray(grasp["target_qpos"], dtype=np.float64)

    out["hand_side"] = "right"
    out["ee_frame"] = "right_hand_flange"
    out["position"] = [round(float(v), 5) for v in p_pos_r]
    out["quaternion_wxyz"] = [round(float(v), 5) for v in _rot_to_quat_wxyz(R_pos_r)]
    out["wrist_position"] = [round(float(v), 5) for v in p_w_r]
    out["wrist_quaternion_wxyz"] = [round(float(v), 5) for v in _rot_to_quat_wxyz(R_w_r)]
    out["approach_dir"] = [round(float(v), 5) for v in (app_r / max(float(np.linalg.norm(app_r)), 1e-12))]
    out["closing_axis"] = [round(float(v), 5) for v in (cls_r / max(float(np.linalg.norm(cls_r)), 1e-12))]
    out["thumb_dir"] = [round(float(v), 5) for v in (thm_r / max(float(np.linalg.norm(thm_r)), 1e-12))]
    out["preshape_qpos"] = [round(float(v), 4) for v in q_pre_r]
    out["target_qpos"] = [round(float(v), 4) for v in q_tgt_r]
    return out


# ---------------------------------------------------------------------------
# Synergy Pose Parameterizations (all 4 primitives)
# ---------------------------------------------------------------------------


def power_cylindrical_qpos(radius_m: float, kin: SharpaWaveKinematics) -> tuple[np.ndarray, np.ndarray]:
    """Radius-adapted 22-DoF (preshape, target) for side cylindrical power wrap."""
    r = float(np.clip(radius_m, 0.0225, 0.0575))
    # Tightness factor t=0 for large cylinder (r=0.0575), t=1 for slim cylinder (r=0.0225)
    t = float(np.clip((0.0575 - r) / (0.0575 - 0.0225), 0.0, 1.0))

    preshape = np.array(
        [
            0.0,
            -0.34,
            1.35,
            -0.30,
            1.65,
            0.0,
            0.0,
            0.25,
            0.18,
            0.0,
            0.0,
            0.25,
            0.18,
            0.0,
            0.0,
            0.25,
            0.18,
            0.0,
            0.0,
            0.0,
            0.25,
            0.18,
        ],
        dtype=np.float64,
    )
    # Thumb: FK-calibrated opposition on the back wall (-z) across r in [0.0225, 0.0575] (~7-20 mm
    # inward preload without over-penetrating past the cylinder axis and tipping tall bottles).
    th_cmc_fe = 0.72 + 0.34 * t
    th_cmc_aa = -0.32 + 0.14 * t
    th_mcp_fe = 1.12 - 0.28 * t
    th_mcp_aa = -0.26 + 0.14 * t
    th_ip = 1.42 - 0.38 * t
    # Fingers: moderate calibrated wrap so all 4 distal elastomer pads oppose the thumb cleanly.
    mcp = 0.62 + 0.34 * t
    pip = 0.78 + 0.42 * t
    dip = 0.52 + 0.36 * t
    target = np.array(
        [
            th_cmc_fe,
            th_cmc_aa,
            th_mcp_fe,
            th_mcp_aa,
            th_ip,
            mcp,
            0.0,
            pip,
            dip,
            mcp,
            0.0,
            pip,
            dip,
            mcp,
            0.0,
            pip,
            dip,
            0.0,
            mcp,
            0.0,
            pip,
            dip,
        ],
        dtype=np.float64,
    )
    return (
        kin.clip_qpos(preshape, side="left", enforce_mcp_aa=True),
        kin.clip_qpos(target, side="left", enforce_mcp_aa=True),
    )


def power_spherical_qpos(radius_m: float, kin: SharpaWaveKinematics) -> tuple[np.ndarray, np.ndarray]:
    """Radius-adapted 22-DoF (preshape, target) for 5-finger spherical dome grasp (|MCP_AA| <= 0.04)."""
    r = float(np.clip(radius_m, 0.0225, 0.0525))
    t = float(np.clip((0.0525 - r) / (0.0525 - 0.0225), 0.0, 1.0))

    # Gentle symmetric abduction strictly inside [-0.04, +0.04] rad
    aa = (-0.03, -0.01, 0.01, 0.03)
    preshape = np.array(
        [
            0.25,
            -0.28,
            0.35,
            -0.20,
            0.40,
            0.15,
            aa[0],
            0.22,
            0.15,
            0.15,
            aa[1],
            0.22,
            0.15,
            0.15,
            aa[2],
            0.22,
            0.15,
            0.05,
            0.15,
            aa[3],
            0.22,
            0.15,
        ],
        dtype=np.float64,
    )
    mcp = 0.78 + 0.52 * t
    pip = 0.90 + 0.55 * t
    dip = 0.65 + 0.45 * t
    target = np.array(
        [
            0.95 + 0.40 * t,
            -0.20,
            0.65 + 0.35 * t,
            -0.15,
            0.85 + 0.40 * t,
            mcp,
            aa[0],
            pip,
            dip,
            mcp,
            aa[1],
            pip,
            dip,
            mcp,
            aa[2],
            pip,
            dip,
            0.08,
            mcp,
            aa[3],
            pip,
            dip,
        ],
        dtype=np.float64,
    )
    return (
        kin.clip_qpos(preshape, side="left", enforce_mcp_aa=True),
        kin.clip_qpos(target, side="left", enforce_mcp_aa=True),
    )


def tripod_pinch_qpos(
    width_m: float,
    kin: SharpaWaveKinematics,
    *,
    clear_m: float = 0.018,
    squeeze_m: float = 0.006,
) -> tuple[np.ndarray, np.ndarray]:
    """Width-adapted 22-DoF (preshape, target) from the refitted wide-opening tripod_pinch template."""
    geom = kin.tripod_pinch_geometry()
    span = max(geom.open_pad_clearance_m - geom.closed_pad_gap_m, 1e-3)
    w = float(np.clip(width_m, 0.008, 0.070))

    alpha_pre = float(np.clip((w + clear_m - geom.closed_pad_gap_m) / span, 0.15, 1.0))
    alpha_tgt = float(np.clip((w - squeeze_m - geom.closed_pad_gap_m) / span, 0.0, 0.95))

    q_pre = (1.0 - alpha_pre) * geom.closed_qpos + alpha_pre * geom.open_qpos
    q_tgt = (1.0 - alpha_tgt) * geom.closed_qpos + alpha_tgt * geom.open_qpos
    return (
        kin.clip_qpos(q_pre, side="left", enforce_mcp_aa=True),
        kin.clip_qpos(q_tgt, side="left", enforce_mcp_aa=True),
    )


def handle_hook_qpos(thickness_m: float, kin: SharpaWaveKinematics) -> tuple[np.ndarray, np.ndarray]:
    """22-DoF (preshape, target) for index+middle+ring handle hook + thumb brace."""
    w = float(np.clip(thickness_m, 0.012, 0.045))
    t = float(np.clip((0.045 - w) / (0.045 - 0.012), 0.0, 1.0))

    preshape = np.array(
        [
            0.10,
            -0.30,
            0.90,
            -0.25,
            1.10,
            0.10,
            0.0,
            0.25,
            0.15,
            0.10,
            0.0,
            0.25,
            0.15,
            0.10,
            0.0,
            0.25,
            0.15,
            0.0,
            1.10,
            0.0,
            1.35,
            0.95,
        ],
        dtype=np.float64,
    )
    mcp = 0.90 + 0.32 * t
    pip = 1.15 + 0.35 * t
    dip = 0.85 + 0.30 * t
    target = np.array(
        [
            1.05 + 0.35 * t,
            -0.22 + 0.10 * t,
            0.80 - 0.25 * t,
            -0.16 + 0.08 * t,
            1.00 - 0.30 * t,
            mcp,
            0.0,
            pip,
            dip,
            mcp,
            0.0,
            pip,
            dip,
            mcp,
            0.0,
            pip,
            dip,
            0.0,
            1.25,
            0.0,
            1.45,
            1.05,
        ],
        dtype=np.float64,
    )
    return (
        kin.clip_qpos(preshape, side="left", enforce_mcp_aa=True),
        kin.clip_qpos(target, side="left", enforce_mcp_aa=True),
    )


# ---------------------------------------------------------------------------
# Vectorized 3D Capsule Collision & Tabletop Clearance
# ---------------------------------------------------------------------------


def check_hand_capsules_collision(
    surface_pts: np.ndarray,
    cap_p0_hand: np.ndarray,
    cap_p1_hand: np.ndarray,
    cap_r: np.ndarray,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
    *,
    support_z: float,
    palm_pen_tol: float = 0.003,
    finger_pen_tol: float = 0.006,
    table_tol: float = 0.005,
) -> tuple[bool, float, float]:
    """Vectorized collision check of 18 hand capsules against object surface points & tabletop.

    Returns ``(collision_free, palm_min_clearance_m, table_clearance_m)``.
    """
    a = cap_p0_hand @ wrist_rot.T + wrist_pos  # (K, 3)
    b = cap_p1_hand @ wrist_rot.T + wrist_pos  # (K, 3)

    min_z = float(min(np.min(a[:, 2] - cap_r), np.min(b[:, 2] - cap_r)))
    table_clear = min_z - support_z
    if table_clear < -table_tol:
        return False, -1.0, table_clear

    if len(surface_pts) == 0:
        return True, float("inf"), table_clear

    ab = b - a  # (K, 3)
    ab2 = np.sum(ab * ab, axis=-1)  # (K,)
    ap = surface_pts[:, None, :] - a[None, :, :]  # (M, K, 3)
    t = np.sum(ap * ab[None, :, :], axis=-1) / np.maximum(ab2[None, :], 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dists = np.linalg.norm(surface_pts[:, None, :] - closest, axis=-1) - cap_r[None, :]  # (M, K)

    palm_min = float(np.min(dists[:, :_PALM_CAPSULE_COUNT]))
    if palm_min < -palm_pen_tol:
        return False, palm_min, table_clear
    finger_min = float(np.min(dists[:, _PALM_CAPSULE_COUNT:]))
    if finger_min < -finger_pen_tol:
        return False, palm_min, table_clear
    return True, palm_min, table_clear


# ---------------------------------------------------------------------------
# Candidate Generators (4 Primitives)
# ---------------------------------------------------------------------------


def _build_hand_frame(approach_dir: np.ndarray, closing_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build orthonormal (approach, closing, thumb) basis from approach & closing vectors."""
    x = approach_dir / max(float(np.linalg.norm(approach_dir)), 1e-12)
    y = closing_axis - float(np.dot(closing_axis, x)) * x
    yn = float(np.linalg.norm(y))
    if yn < 1e-9:
        ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = np.cross(ref, x)
        yn = float(np.linalg.norm(y))
    y = y / max(yn, 1e-12)
    z = np.cross(x, y)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    return x, y, z


def _make_grasp_entry(
    *,
    grasp_type: str,
    position: np.ndarray,
    approach_dir: np.ndarray,
    closing_axis: np.ndarray,
    thumb_dir: np.ndarray,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
    width: float,
    preshape_qpos: np.ndarray,
    target_qpos: np.ndarray,
    thumb_first_joints: int,
    thumb_lead_ratio: float,
    contact_fingers: list[str],
    part_tag: str,
    score: float,
) -> dict[str, Any]:
    R_sem = np.column_stack([approach_dir, closing_axis, thumb_dir])
    if float(np.linalg.det(R_sem)) < 0.0:
        R_sem[:, 2] = -R_sem[:, 2]
        thumb_dir = R_sem[:, 2]
    return {
        "id": "",
        "hand_side": "left",
        "ee_frame": "left_hand_flange",
        "grasp_type": grasp_type,
        "position": [round(float(v), 5) for v in position],
        "quaternion_wxyz": [round(float(v), 5) for v in _rot_to_quat_wxyz(R_sem)],
        "wrist_position": [round(float(v), 5) for v in wrist_pos],
        "wrist_quaternion_wxyz": [round(float(v), 5) for v in _rot_to_quat_wxyz(wrist_rot)],
        "approach_dir": [round(float(v), 5) for v in approach_dir],
        "closing_axis": [round(float(v), 5) for v in closing_axis],
        "thumb_dir": [round(float(v), 5) for v in thumb_dir],
        "width": round(float(width), 5),
        "preshape_qpos": [round(float(v), 4) for v in preshape_qpos],
        "target_qpos": [round(float(v), 4) for v in target_qpos],
        "close_schedule": {
            "thumb_first_joints": int(thumb_first_joints),
            "thumb_lead_ratio": round(float(thumb_lead_ratio), 3),
        },
        "contact_fingers": list(contact_fingers),
        "part_tag": part_tag,
        "tag_source": "geometry",
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "source": "dex_synergy_geom",
        "verified": False,
        "sim_score": None,
    }


def _generate_power_cylindrical(
    pts: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    com: np.ndarray,
    kin: SharpaWaveKinematics,
    *,
    n_azimuths: int = 12,
) -> list[dict[str, Any]]:
    ext = bounds_max - bounds_min
    height = float(ext[2])
    if height < 0.055:
        return []
    support_z = float(bounds_min[2])

    # Sample 3 candidate grasp heights along object body/neck, keeping thumb (+20 mm) & index (+30 mm)
    # below the top rim on short 7.5-9 cm cups while keeping pinky clear of the tabletop.
    z_lo = max(support_z + 0.038, support_z + 0.35 * height)
    z_hi = min(float(bounds_max[2]) - 0.032, support_z + 0.65 * height)
    if z_lo > z_hi:
        z_levels = [float(np.clip(support_z + 0.48 * height, support_z + 0.036, float(bounds_max[2]) - 0.024))]
    else:
        z_levels = list(np.linspace(z_lo, z_hi, 3))

    # Canonical left_hand_flange axes matrix: columns = [fingers(+Z), palm(+X), thumb(-Y)]
    H_local = np.column_stack([PALM_FINGERS_AXIS, PALM_NORMAL_AXIS, PALM_THUMB_AXIS])
    candidates: list[dict[str, Any]] = []

    for z_lvl in z_levels:
        band = np.abs(pts[:, 2] - z_lvl) <= max(0.018, 0.14 * height)
        if int(np.sum(band)) < 8:
            continue
        band_pts = pts[band]
        c_xy = np.mean(band_pts[:, :2], axis=0)
        radii_xy = np.linalg.norm(band_pts[:, :2] - c_xy[None, :], axis=-1)
        r_local = float(np.percentile(radii_xy, 80))
        width = 2.0 * r_local
        if width < 0.045 or width > 0.115:
            continue

        q_pre, q_tgt = power_cylindrical_qpos(r_local, kin)
        cap_p0, cap_p1, cap_r = kin.link_capsules(q_pre, side="left")
        part_tag = "neck" if (z_lvl > support_z + 0.65 * height and width < 0.068) else "body"
        pos = np.array([c_xy[0], c_xy[1], z_lvl], dtype=np.float64)

        for k in range(n_azimuths):
            theta = 2.0 * math.pi * k / n_azimuths
            palm_in = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
            thumb_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            fingers_dir = np.cross(thumb_up, palm_in)
            fingers_dir = fingers_dir / max(float(np.linalg.norm(fingers_dir)), 1e-12)

            # wrist_rot maps H_local columns [fingers, palm, thumb] -> [fingers_dir, palm_in, thumb_up]
            W_target = np.column_stack([fingers_dir, palm_in, thumb_up])
            wrist_rot = W_target @ H_local.T

            # Place palm_surface + (r_local + 0.001) along palm_in at pos (flush palm preload)
            reach_hand = PALM_SURFACE_POINT + (r_local + 0.001) * PALM_NORMAL_AXIS
            wrist_pos = pos - wrist_rot @ reach_hand

            ok, palm_clear, table_clear = check_hand_capsules_collision(
                pts, cap_p0, cap_p1, cap_r, wrist_pos, wrist_rot, support_z=support_z
            )
            if not ok:
                continue

            com_dist = float(np.linalg.norm(pos - com))
            com_score = math.exp(-((com_dist / max(float(np.linalg.norm(ext)), 0.05)) ** 2))
            clear_score = float(np.clip((palm_clear + 0.003) / 0.025, 0.0, 1.0))
            table_score = float(np.clip(table_clear / 0.03, 0.0, 1.0))
            width_opt = 1.0 - abs(width - 0.075) / 0.06
            score = 0.35 * com_score + 0.25 * clear_score + 0.20 * table_score + 0.20 * np.clip(width_opt, 0.0, 1.0)

            candidates.append(
                _make_grasp_entry(
                    grasp_type="power_cylindrical",
                    position=pos,
                    approach_dir=palm_in,
                    closing_axis=fingers_dir,
                    thumb_dir=thumb_up,
                    wrist_pos=wrist_pos,
                    wrist_rot=wrist_rot,
                    width=width,
                    preshape_qpos=q_pre,
                    target_qpos=q_tgt,
                    thumb_first_joints=5,
                    thumb_lead_ratio=0.35,
                    contact_fingers=["thumb", "index", "middle", "ring", "pinky"],
                    part_tag=part_tag,
                    score=score,
                )
            )
    return candidates


def _generate_power_spherical(
    pts: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    com: np.ndarray,
    kin: SharpaWaveKinematics,
    *,
    n_azimuths: int = 8,
) -> list[dict[str, Any]]:
    ext = bounds_max - bounds_min
    support_z = float(bounds_min[2])
    xy_span = float(0.5 * (ext[0] + ext[1]))
    if xy_span < 0.045 or xy_span > 0.105:
        return []

    r_est = 0.5 * xy_span
    q_pre, q_tgt = power_spherical_qpos(r_est, kin)
    cap_p0, cap_p1, cap_r = kin.link_capsules(q_pre, side="left")

    # Anchor near upper center of object
    pos = np.array([com[0], com[1], float(bounds_min[2] + 0.60 * ext[2])], dtype=np.float64)
    H_local = np.column_stack([PALM_FINGERS_AXIS, PALM_NORMAL_AXIS, PALM_THUMB_AXIS])
    candidates: list[dict[str, Any]] = []

    for tilt_deg in (0.0, 35.0, 55.0):
        tilt = math.radians(tilt_deg)
        az_count = 4 if tilt_deg == 0.0 else n_azimuths
        for k in range(az_count):
            theta = 2.0 * math.pi * k / az_count
            # Palm normal points downward/oblique into object center
            palm_in = np.array(
                [math.sin(tilt) * math.cos(theta), math.sin(tilt) * math.sin(theta), -math.cos(tilt)],
                dtype=np.float64,
            )
            palm_in = palm_in / max(float(np.linalg.norm(palm_in)), 1e-12)
            ref = np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=np.float64)
            fingers_dir = np.cross(palm_in, ref)
            fingers_dir = fingers_dir / max(float(np.linalg.norm(fingers_dir)), 1e-12)
            thumb_dir = np.cross(palm_in, fingers_dir)
            thumb_dir = thumb_dir / max(float(np.linalg.norm(thumb_dir)), 1e-12)

            W_target = np.column_stack([fingers_dir, palm_in, thumb_dir])
            wrist_rot = W_target @ H_local.T

            # Palm dome sits r_est + 0.012 above pos along -palm_in, centered at mid-palm (z=0.10)
            dome_anchor = np.array([0.025 + r_est + 0.012, 0.0, 0.105], dtype=np.float64)
            wrist_pos = pos - wrist_rot @ dome_anchor

            ok, palm_clear, table_clear = check_hand_capsules_collision(
                pts, cap_p0, cap_p1, cap_r, wrist_pos, wrist_rot, support_z=support_z
            )
            if not ok:
                continue

            top_pref = 1.0 - 0.25 * (tilt_deg / 60.0)
            clear_score = float(np.clip((palm_clear + 0.003) / 0.025, 0.0, 1.0))
            table_score = float(np.clip(table_clear / 0.03, 0.0, 1.0))
            score = 0.40 * top_pref + 0.30 * clear_score + 0.30 * table_score

            candidates.append(
                _make_grasp_entry(
                    grasp_type="power_spherical",
                    position=pos,
                    approach_dir=palm_in,
                    closing_axis=fingers_dir,
                    thumb_dir=thumb_dir,
                    wrist_pos=wrist_pos,
                    wrist_rot=wrist_rot,
                    width=xy_span,
                    preshape_qpos=q_pre,
                    target_qpos=q_tgt,
                    thumb_first_joints=5,
                    thumb_lead_ratio=0.30,
                    contact_fingers=["thumb", "index", "middle", "ring", "pinky"],
                    part_tag="body_top",
                    score=score,
                )
            )
    return candidates


def _generate_tripod_pinch(
    pts: np.ndarray,
    nrms: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    com: np.ndarray,
    kin: SharpaWaveKinematics,
    rng: np.random.Generator,
    *,
    max_pairs: int = 80,
    n_angles: int = 8,
) -> list[dict[str, Any]]:
    ext = bounds_max - bounds_min
    support_z = float(bounds_min[2])
    geom = kin.tripod_pinch_geometry()
    H_pinch = np.column_stack([geom.approach_dir, geom.closing_axis, geom.thumb_dir])

    # Vectorized antipodal pair search over sampled surface points
    n_pts = len(pts)
    if n_pts < 4:
        return []
    sub_n = min(n_pts, 180)
    idx_sub = rng.choice(n_pts, size=sub_n, replace=False) if n_pts > sub_n else np.arange(n_pts)
    p_sub = pts[idx_sub]
    n_sub = nrms[idx_sub]

    diff = p_sub[None, :, :] - p_sub[:, None, :]  # (S, S, 3): p_j - p_i
    dist = np.linalg.norm(diff, axis=-1)  # (S, S)
    valid_w = (dist >= 0.008) & (dist <= 0.070)
    iu, ju = np.where(np.triu(valid_w, k=1))
    if len(iu) == 0:
        return []

    dirs = diff[iu, ju] / dist[iu, ju, None]  # p_i -> p_j
    # Outward normals should oppose each other along dirs: n_i . dirs < -cos_cone and n_j . dirs > cos_cone
    cos_cone = 1.0 / math.sqrt(1.0 + 0.45 * 0.45)  # ~0.91 (mu=0.45)
    c1 = -np.einsum("ij,ij->i", n_sub[iu], dirs)
    c2 = np.einsum("ij,ij->i", n_sub[ju], dirs)
    anti = (c1 >= cos_cone) & (c2 >= cos_cone)
    iu, ju, dirs = iu[anti], ju[anti], dirs[anti]
    c1, c2 = c1[anti], c2[anti]
    if len(iu) == 0:
        return []

    widths = dist[iu, ju]
    mids = 0.5 * (p_sub[iu] + p_sub[ju])
    # Rank pairs by antipodal alignment & upper-half accessibility
    align = 0.5 * (c1 + c2)
    z_rel = (mids[:, 2] - support_z) / max(float(ext[2]), 1e-4)
    pair_score = 0.6 * align + 0.4 * np.clip(z_rel, 0.0, 1.0)
    order = np.argsort(-pair_score)[:max_pairs]

    candidates: list[dict[str, Any]] = []
    # Cache capsule sets by quantized width (2 mm buckets) for speed
    cap_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for idx in order:
        mid = mids[idx]
        w = float(widths[idx])
        cls_axis = dirs[idx]
        w_bin = int(round(w * 500.0))
        if w_bin not in cap_cache:
            q_pre, q_tgt = tripod_pinch_qpos(w, kin)
            cap_p0, cap_p1, cap_r = kin.link_capsules(q_pre, side="left")
            cap_cache[w_bin] = (q_pre, q_tgt, cap_p0, cap_p1, cap_r)
        q_pre, q_tgt, cap_p0, cap_p1, cap_r = cap_cache[w_bin]

        # Build orthonormal frame perpendicular to closing_axis and test top/angled approaches
        ref = np.array([0.0, 0.0, 1.0]) if abs(cls_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u1 = np.cross(cls_axis, ref)
        u1 = u1 / max(float(np.linalg.norm(u1)), 1e-12)
        u2 = np.cross(cls_axis, u1)
        u2 = u2 / max(float(np.linalg.norm(u2)), 1e-12)

        for a_i in range(n_angles):
            phi = 2.0 * math.pi * a_i / n_angles
            app = math.cos(phi) * u1 + math.sin(phi) * u2
            if app[2] > -0.15:
                continue  # reject upward/shallow approaches that collide with tabletop
            for cls_sign in (1.0, -1.0):
                app_u, cls_u, thm_u = _build_hand_frame(app, cls_sign * cls_axis)
                W_target = np.column_stack([app_u, cls_u, thm_u])
                wrist_rot = W_target @ H_pinch.T
                wrist_pos = mid - wrist_rot @ geom.pinch_center

                ok, palm_clear, table_clear = check_hand_capsules_collision(
                    pts, cap_p0, cap_p1, cap_r, wrist_pos, wrist_rot, support_z=support_z
                )
                if not ok:
                    continue

                z_norm = (mid[2] - support_z) / max(float(ext[2]), 1e-4)
                part_tag = "rim" if (z_norm > 0.78 and w < 0.025) else ("lid" if z_norm > 0.82 else "body_top")
                com_dist = float(np.linalg.norm(mid - com))
                com_s = math.exp(-((com_dist / max(float(np.linalg.norm(ext)), 0.05)) ** 2))
                top_s = float(np.clip(-app_u[2], 0.0, 1.0))
                clear_s = float(np.clip((palm_clear + 0.003) / 0.025, 0.0, 1.0))
                score = 0.35 * float(align[idx]) + 0.25 * top_s + 0.20 * com_s + 0.20 * clear_s

                candidates.append(
                    _make_grasp_entry(
                        grasp_type="tripod_pinch",
                        position=mid,
                        approach_dir=app_u,
                        closing_axis=cls_u,
                        thumb_dir=thm_u,
                        wrist_pos=wrist_pos,
                        wrist_rot=wrist_rot,
                        width=w,
                        preshape_qpos=q_pre,
                        target_qpos=q_tgt,
                        thumb_first_joints=0,
                        thumb_lead_ratio=0.0,
                        contact_fingers=["thumb", "index", "middle"],
                        part_tag=part_tag,
                        score=score,
                    )
                )
    return candidates


def _detect_handle_points(
    pts: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    com: np.ndarray,
) -> np.ndarray:
    """Detect off-center thin handle protrusion points (e.g. mug handle) from geometry."""
    ext = bounds_max - bounds_min
    if float(ext[2]) < 0.05 or max(float(ext[0]), float(ext[1])) < 0.06:
        return np.empty((0, 3), dtype=np.float64)
    # Estimate main-body radial envelope from median XY; handle points protrude radially
    c_xy = np.median(pts[:, :2], axis=0)
    r_xy = np.linalg.norm(pts[:, :2] - c_xy[None, :], axis=-1)
    r_med = float(np.median(r_xy))
    if r_med < 0.015:
        return np.empty((0, 3), dtype=np.float64)
    protrusion = pts[r_xy > r_med + max(0.016, 0.28 * r_med)]
    if len(protrusion) < 8:
        return np.empty((0, 3), dtype=np.float64)
    # Check angular concentration: a handle occupies a narrow angular sector (< 90 deg)
    ang = np.arctan2(protrusion[:, 1] - c_xy[1], protrusion[:, 0] - c_xy[0])
    mean_dir = np.array([float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))])
    concentration = float(np.linalg.norm(mean_dir))
    if concentration < 0.75:
        return np.empty((0, 3), dtype=np.float64)
    # Check handle bar cross-section is genuinely compact (not a wide box face/corner)
    out_dir = mean_dir / concentration
    tang_dir = np.array([-out_dir[1], out_dir[0]])
    rel_xy = protrusion[:, :2] - np.mean(protrusion[:, :2], axis=0)[None, :]
    if float(np.ptp(rel_xy @ tang_dir)) > 0.060 or float(np.ptp(rel_xy @ out_dir)) > 0.060:
        return np.empty((0, 3), dtype=np.float64)
    return protrusion


def _generate_handle_hook(
    pts: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    com: np.ndarray,
    kin: SharpaWaveKinematics,
    *,
    handle_pts: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    h_pts = (
        handle_pts
        if (handle_pts is not None and len(handle_pts) >= 4)
        else _detect_handle_points(pts, bounds_min, bounds_max, com)
    )
    if len(h_pts) < 4:
        return []

    support_z = float(bounds_min[2])
    h_center = np.mean(h_pts, axis=0)
    # Radial outward direction from object center to handle
    out_xy = h_center[:2] - com[:2]
    out_norm = float(np.linalg.norm(out_xy))
    if out_norm < 1e-4:
        return []
    out_dir = np.array([out_xy[0] / out_norm, out_xy[1] / out_norm, 0.0], dtype=np.float64)

    # Estimate handle bar thickness perpendicular to radial & vertical axes
    tang = np.array([-out_dir[1], out_dir[0], 0.0], dtype=np.float64)
    tang_proj = (h_pts - h_center[None, :]) @ tang
    rad_proj = (h_pts - h_center[None, :]) @ out_dir
    thickness = float(np.clip(min(np.ptp(tang_proj), np.ptp(rad_proj)), 0.012, 0.045))

    q_pre, q_tgt = handle_hook_qpos(thickness, kin)
    cap_p0, cap_p1, cap_r = kin.link_capsules(q_pre, side="left")
    H_local = np.column_stack([PALM_FINGERS_AXIS, PALM_NORMAL_AXIS, PALM_THUMB_AXIS])
    candidates: list[dict[str, Any]] = []

    # Approach handle from outside (-out_dir) with slight pitch variations and both tangent sides
    for sign in (1.0, -1.0):
        for pitch_deg in (0.0, 20.0):
            pitch = math.radians(pitch_deg)
            palm_in = -math.cos(pitch) * out_dir + np.array([0.0, 0.0, -math.sin(pitch)], dtype=np.float64)
            palm_in = palm_in / max(float(np.linalg.norm(palm_in)), 1e-12)
            fingers_dir = sign * tang
            thumb_dir = np.cross(palm_in, fingers_dir)
            thumb_dir = thumb_dir / max(float(np.linalg.norm(thumb_dir)), 1e-12)
            if thumb_dir[2] < 0.0:
                continue  # keep thumb up

            W_target = np.column_stack([fingers_dir, palm_in, thumb_dir])
            wrist_rot = W_target @ H_local.T
            reach_hand = PALM_SURFACE_POINT + (0.5 * thickness + 0.010) * PALM_NORMAL_AXIS
            wrist_pos = h_center - wrist_rot @ reach_hand

            ok, palm_clear, table_clear = check_hand_capsules_collision(
                pts, cap_p0, cap_p1, cap_r, wrist_pos, wrist_rot, support_z=support_z
            )
            if not ok:
                continue

            clear_s = float(np.clip((palm_clear + 0.003) / 0.025, 0.0, 1.0))
            table_s = float(np.clip(table_clear / 0.03, 0.0, 1.0))
            score = 0.50 + 0.25 * clear_s + 0.25 * table_s

            candidates.append(
                _make_grasp_entry(
                    grasp_type="handle_hook",
                    position=h_center,
                    approach_dir=palm_in,
                    closing_axis=fingers_dir,
                    thumb_dir=thumb_dir,
                    wrist_pos=wrist_pos,
                    wrist_rot=wrist_rot,
                    width=thickness,
                    preshape_qpos=q_pre,
                    target_qpos=q_tgt,
                    thumb_first_joints=0,
                    thumb_lead_ratio=0.0,
                    contact_fingers=["thumb", "index", "middle", "ring"],
                    part_tag="handle",
                    score=score,
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# Bucketed Pose NMS & Top-level Pipeline
# ---------------------------------------------------------------------------


def bucketed_nms_dexhand(
    candidates: list[dict[str, Any]],
    *,
    max_grasps: int = 28,
    pos_tol_m: float = 0.018,
    angle_tol_deg: float = 25.0,
) -> list[dict[str, Any]]:
    """Per-`grasp_type` pose NMS with balanced round-robin quota selection."""
    if not candidates:
        return []
    cos_tol = math.cos(math.radians(angle_tol_deg))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        by_type.setdefault(str(c["grasp_type"]), []).append(c)

    kept_by_type: dict[str, list[dict[str, Any]]] = {}
    for gtype, bucket in by_type.items():
        bucket_sorted = sorted(bucket, key=lambda g: -float(g["score"]))
        kept: list[dict[str, Any]] = []
        kept_pos: list[np.ndarray] = []
        kept_app: list[np.ndarray] = []
        for g in bucket_sorted:
            p = np.asarray(g["position"], dtype=np.float64)
            a = np.asarray(g["approach_dir"], dtype=np.float64)
            dup = False
            for kp, ka in zip(kept_pos, kept_app):
                if float(np.linalg.norm(p - kp)) < pos_tol_m and float(np.dot(a, ka)) > cos_tol:
                    dup = True
                    break
            if not dup:
                kept.append(g)
                kept_pos.append(p)
                kept_app.append(a)
        kept_by_type[gtype] = kept

    # Round-robin across non-empty buckets ordered by each bucket's top score
    active_types = sorted(kept_by_type, key=lambda t: -float(kept_by_type[t][0]["score"]))
    selected: list[dict[str, Any]] = []
    cursors = {t: 0 for t in active_types}
    while len(selected) < max_grasps:
        progressed = False
        for t in active_types:
            if len(selected) >= max_grasps:
                break
            idx = cursors[t]
            if idx < len(kept_by_type[t]):
                selected.append(kept_by_type[t][idx])
                cursors[t] = idx + 1
                progressed = True
        if not progressed:
            break

    selected.sort(key=lambda g: -float(g["score"]))
    for i, g in enumerate(selected):
        g["id"] = f"sw{i:03d}"
    return selected


HEAVY_OBJECT_MASS_KG = 0.30


def _extract_mass_kg(metadata: dict[str, Any] | None) -> float | None:
    if not isinstance(metadata, dict):
        return None
    for key in ("mass_kg", "mass"):
        val = metadata.get(key)
        if isinstance(val, int | float) and not isinstance(val, bool) and math.isfinite(float(val)) and float(val) > 0:
            return float(val)
    return None


def generate_sharpa_wave_grasps(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_grasps: int = 28,
    n_surface_samples: int = 400,
    seed: int = 0,
    kin: SharpaWaveKinematics | None = None,
    handle_points: np.ndarray | None = None,
    mass_kg: float | None = None,
) -> list[dict[str, Any]]:
    """Generate collision-free, contract-compliant Sharpa Wave grasps for a mesh."""
    kin = kin or get_sharpa_kinematics()
    tris, tri_normals, areas, tri_centers = aff.triangle_geometry(vertices, faces)
    if len(tris) == 0:
        return []

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    com = (tri_centers * areas[:, None]).sum(axis=0) / areas.sum()

    signed_vol_proxy = float(np.sum(areas * np.einsum("ij,ij->i", tri_centers - com, tri_normals)))
    if signed_vol_proxy < -1e-12:
        tri_normals = -tri_normals

    rng = np.random.default_rng(seed)
    pts, nrms, _ = aff.sample_surface_points(tris, tri_normals, areas, n_surface_samples, rng)

    power_raw: list[dict[str, Any]] = []
    power_raw.extend(_generate_power_cylindrical(pts, bounds_min, bounds_max, com, kin))
    power_raw.extend(_generate_power_spherical(pts, bounds_min, bounds_max, com, kin))
    power_raw.extend(_generate_handle_hook(pts, bounds_min, bounds_max, com, kin, handle_pts=handle_points))
    pinch_raw = _generate_tripod_pinch(pts, nrms, bounds_min, bounds_max, com, kin, rng)

    # Mass-aware taxonomy routing (plan §8.4): DIP torque limit is 0.189 Nm, so heavy objects (> 0.3 kg)
    # prioritize palm-supported power/handle grasps and restrict tripod_pinch to rim/lid/cap features.
    if mass_kg is not None and mass_kg > HEAVY_OBJECT_MASS_KG and power_raw:
        filtered_pinch: list[dict[str, Any]] = []
        for g in pinch_raw:
            if g.get("part_tag") in ("rim", "lid", "cap"):
                g_adj = dict(g)
                g_adj["score"] = round(float(g["score"]) * 0.7, 4)
                filtered_pinch.append(g_adj)
        pinch_raw = filtered_pinch

    raw = power_raw + pinch_raw
    return bucketed_nms_dexhand(raw, max_grasps=max_grasps)


def annotate_mesh_sharpa_wave(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    asset_id: str,
    metadata: dict[str, Any] | None = None,
    max_grasps: int = 28,
    n_surface_samples: int = 400,
    seed: int = 0,
    existing_payload: dict[str, Any] | None = None,
    kin: SharpaWaveKinematics | None = None,
) -> dict[str, Any]:
    """Build or merge a schema_version=1 affordance payload containing `grasps['sharpa_wave']`."""
    mesh_hash = aff.compute_mesh_hash(vertices, faces)
    sw_grasps = generate_sharpa_wave_grasps(
        vertices,
        faces,
        max_grasps=max_grasps,
        n_surface_samples=n_surface_samples,
        seed=seed,
        kin=kin,
        mass_kg=_extract_mass_kg(metadata),
    )
    if existing_payload is not None:
        payload = json.loads(json.dumps(existing_payload))
        payload["mesh_hash"] = mesh_hash
        payload.setdefault("grasps", {})["sharpa_wave"] = sw_grasps
        return payload

    tris, _, areas, centers = aff.triangle_geometry(vertices, faces)
    if len(vertices) > 0:
        b_min = vertices.min(axis=0)
        b_max = vertices.max(axis=0)
        ext = b_max - b_min
    else:
        b_min = np.zeros(3)
        b_max = np.zeros(3)
        ext = np.zeros(3)
    return {
        "schema_version": aff.SCHEMA_VERSION,
        "asset_id": asset_id,
        "mesh_hash": mesh_hash,
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
        "symmetry": aff.detect_symmetry(vertices),
        "functional_frames": aff.extract_functional_frames(vertices, areas, centers, metadata=metadata),
        "grasps": {"sharpa_wave": sw_grasps},
    }


def annotate_asset_sharpa_wave(
    path: Path | str,
    *,
    output: Path | str | None = None,
    asset_id: str | None = None,
    merge: bool = False,
    max_grasps: int = 28,
    n_surface_samples: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Load an object asset (.obj/.stl/.usd*) and annotate `grasps['sharpa_wave']`."""
    p = Path(path)
    mesh_path = aff.resolve_mesh_path(p)
    asset_dir = p if p.is_dir() else p.parent
    metadata = None
    meta_file = asset_dir / "metadata.json"
    if meta_file.is_file():
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None
    aid = asset_id or str(asset_dir.name if p.is_dir() else mesh_path.stem)
    verts, faces = aff.load_mesh(mesh_path)
    existing: dict[str, Any] | None = None
    if merge and output is not None and Path(output).is_file():
        existing = json.loads(Path(output).read_text(encoding="utf-8"))
    result = annotate_mesh_sharpa_wave(
        verts,
        faces,
        asset_id=aid,
        metadata=metadata,
        max_grasps=max_grasps,
        n_surface_samples=n_surface_samples,
        seed=seed,
        existing_payload=existing,
    )
    if output is not None:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Batch annotation & CLI
# ---------------------------------------------------------------------------


def _run_batch_dexhand(args: argparse.Namespace) -> int:
    pool_data = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    objects = pool_data["objects"] if isinstance(pool_data, dict) else pool_data
    if args.commercial_only:
        objects = [o for o in objects if not str(o["cat"]).startswith(("geniesim_", "partnet_"))]
    if args.offset > 0:
        objects = objects[args.offset :]
    if args.limit > 0:
        objects = objects[: args.limit]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    kin = get_sharpa_kinematics()

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
        asset_dir = aff._resolve_pool_asset_dir(Path(args.assets_root), cat, idx)
        if asset_dir is None:
            manifest.append({"asset_id": asset_id, "status": "missing_asset", "affordance_path": str(out_path)})
            err_count += 1
            continue
        if args.skip_existing and out_path.is_file():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            sw_existing = (existing.get("grasps") or {}).get("sharpa_wave") if isinstance(existing, dict) else None
            if sw_existing is not None:
                manifest.append(
                    {
                        "asset_id": asset_id,
                        "asset_dir": str(asset_dir),
                        "affordance_path": str(out_path),
                        "status": "skipped",
                        "grasps_sharpa": len(sw_existing),
                    }
                )
                skip_count += 1
                continue
        try:
            mesh_path = aff.resolve_mesh_path(asset_dir)
            meta_file = asset_dir / "metadata.json"
            metadata = None
            if meta_file.is_file():
                try:
                    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metadata = None
            verts, faces = aff.load_mesh(mesh_path)
            existing_payload = (
                json.loads(out_path.read_text(encoding="utf-8")) if (args.merge and out_path.is_file()) else None
            )
            res = annotate_mesh_sharpa_wave(
                verts,
                faces,
                asset_id=asset_id,
                metadata=metadata,
                max_grasps=args.max_grasps,
                n_surface_samples=args.samples,
                seed=args.seed,
                existing_payload=existing_payload,
                kin=kin,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            sw = res["grasps"].get("sharpa_wave", [])
            n_g = len(sw)
            if n_g > 0:
                ok_count += 1
            else:
                zero_count += 1
            breakdown: dict[str, int] = {}
            for g in sw:
                gt = str(g.get("grasp_type", ""))
                breakdown[gt] = breakdown.get(gt, 0) + 1
            manifest.append(
                {
                    "asset_id": asset_id,
                    "asset_dir": str(asset_dir),
                    "affordance_path": str(out_path),
                    "status": "ok" if n_g > 0 else "zero_grasps",
                    "grasps_sharpa": n_g,
                    "breakdown": breakdown,
                    "extents": res["bounds"]["extents"],
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

    manifest_path = Path(args.manifest_path) if args.manifest_path else (out_root / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[affordance-dexhand batch] total={len(objects)} ok={ok_count} zero={zero_count} "
        f"skipped={skip_count} err={err_count} manifest={manifest_path}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenFlywheel Sharpa Wave affordance annotator (Stage 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Annotate a mesh or asset directory for Sharpa Wave")
    p_ann.add_argument("path", type=Path, help="Mesh file (.obj/.stl/.usd[a|z]) or asset directory")
    p_ann.add_argument("--output", "-o", type=Path, default=None, help="Output affordance.json path")
    p_ann.add_argument("--asset-id", default=None, help="Override asset_id")
    p_ann.add_argument("--merge", action="store_true", help="Merge into existing output affordance.json if present")
    p_ann.add_argument("--max-grasps", type=int, default=28, help="Max Sharpa Wave grasps after bucketed NMS")
    p_ann.add_argument("--samples", type=int, default=400, help="Surface points to sample")
    p_ann.add_argument("--seed", type=int, default=0, help="Random seed")

    p_batch = sub.add_parser("batch", help="Batch annotate a pool JSON for Sharpa Wave")
    p_batch.add_argument("--pool", type=Path, required=True, help="Pool JSON (e.g. pick_anything_pool.json)")
    p_batch.add_argument("--assets-root", type=Path, required=True, help="Assets root directory")
    p_batch.add_argument("--output-dir", type=Path, required=True, help="Output directory for affordance.json files")
    p_batch.add_argument("--manifest-path", type=Path, default=None, help="Output manifest.json path")
    p_batch.add_argument("--commercial-only", action="store_true", help="Exclude geniesim_* and partnet_*")
    p_batch.add_argument("--offset", type=int, default=0, help="Start offset into pool")
    p_batch.add_argument("--limit", type=int, default=0, help="Max objects to annotate (0 = all)")
    p_batch.add_argument("--max-grasps", type=int, default=24, help="Max Sharpa Wave grasps per object")
    p_batch.add_argument("--samples", type=int, default=400, help="Surface points to sample")
    p_batch.add_argument("--seed", type=int, default=0, help="Random seed")
    p_batch.add_argument("--merge", action="store_true", help="Merge into existing affordance.json")
    p_batch.add_argument("--skip-existing", action="store_true", help="Skip if sharpa_wave already present")

    sub.add_parser("verify-pinch", help="Verify refitted Sharpa Wave tripod_pinch pad clearance & limits")

    p_ins = sub.add_parser("inspect", help="Inspect and optionally validate an affordance.json")
    p_ins.add_argument("path", type=Path, help="Path to affordance.json")
    p_ins.add_argument("--validate", action="store_true", help="Validate schema invariants (exit 1 on error)")

    args = parser.parse_args(argv)
    if args.cmd == "batch":
        return _run_batch_dexhand(args)
    if args.cmd == "verify-pinch":
        kin = get_sharpa_kinematics()
        geom = kin.tripod_pinch_geometry()
        print(
            f"tripod_pinch open_pad_clearance={geom.open_pad_clearance_m * 1000:.1f} mm "
            f"(target >= 75.0 mm), closed_gap={geom.closed_pad_gap_m * 1000:.1f} mm, "
            f"mp_corridor_clear={geom.mp_corridor_clearance_m * 1000:.1f} mm"
        )
        return 0 if geom.open_pad_clearance_m >= 0.075 else 1

    if args.cmd == "annotate":
        res = annotate_asset_sharpa_wave(
            args.path,
            output=args.output,
            asset_id=args.asset_id,
            merge=args.merge,
            max_grasps=args.max_grasps,
            n_surface_samples=args.samples,
            seed=args.seed,
        )
        sw = res["grasps"]["sharpa_wave"]
        counts: dict[str, int] = {}
        for g in sw:
            counts[g["grasp_type"]] = counts.get(g["grasp_type"], 0) + 1
        print(f"asset={res['asset_id']} sharpa_wave={len(sw)} breakdown={counts}")
        if args.output is not None:
            print(f"wrote {args.output}")
        return 0

    if args.cmd == "inspect":
        data = json.loads(args.path.read_text(encoding="utf-8"))
        sw = data.get("grasps", {}).get("sharpa_wave", [])
        counts: dict[str, int] = {}
        for g in sw:
            if isinstance(g, dict) and "grasp_type" in g:
                counts[g["grasp_type"]] = counts.get(g["grasp_type"], 0) + 1
        print(f"asset={data.get('asset_id')} sharpa_wave={len(sw)} breakdown={counts}")
        if args.validate:
            errs, warns = aff.validate_affordance(data)
            for w in warns:
                print(f"WARN: {w}")
            for e in errs:
                print(f"ERROR: {e}")
            return 1 if errs else 0
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
