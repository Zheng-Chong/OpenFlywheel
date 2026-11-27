"""Online Sharpa Wave dexterous-hand affordance planner & dual-mode pick execution.

Consumes Stage 1+2 `affordance.json["grasps"]["sharpa_wave"]` sidecars (canonical
`left_hand_flange`, `scale_ref = 1.0`) and provides:

  1. Pure-CPU `DexHandAffordancePlanner` (NumPy/SciPy only, zero Isaac dependency):
     - Z-axis rotational symmetry expansion (`continuous` / `discrete` around `axis_xy`)
     - Deterministic left -> right hand mirroring via `mirror_grasp_left_to_right` (< 1 mm FK error)
     - DR scale adaptation of grasp/wrist pose, width bounds, and 22-DoF closure targets
     - Semantic `part_tag` filtering (`part_tags`, `exclude_tags`, `prefer_tags`) & `grasp_types`
     - Tabletop clearance & distractor obstacle collision filtering (hand + approach corridor)
     - Optional whole-body `(lift, waist_pitch, arm_q7)` coordinated IK callback
  2. Dual-mode online execution (`pick_anything_sharpa_affordance_actions`) using
     `env_cfg/g1d_office_sharpa_wave.yml`:
     - `tripod_pinch`: virtual pinch-center bridge (`PinchBridgeSpec`)
     - `power_cylindrical` / `power_spherical` / `handle_hook`: staged 22-DoF closure
       (`open -> preshape -> slide_in -> thumb_first -> finger_wrap -> column_lift`)
     - Automatic fallback to `x_anything_sharpa` heuristic when no affordance applies
     - Per-episode logging of `grasp_type`, `grasp_source`, `grasp_id`, `hand_side`, and
       failure attribution.

`scripts/g1d/scripted_demo.py` and all existing 16-DoF Dex1-1 experts remain untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.objects import affordance_dexhand as dex_aff

SHARPA_GRIPPER_KEY = "sharpa_wave"
SHARPA_DOF = dex_aff.SHARPA_DOF
MCP_AA_INDICES = dex_aff.MCP_AA_INDICES
MCP_AA_SAFE_LIMIT = dex_aff.MCP_AA_SAFE_LIMIT

# Valid scaled width bounds per grasp_type (m)
GRASP_TYPE_WIDTH_BOUNDS: dict[str, tuple[float, float]] = {
    "power_cylindrical": (0.040, 0.120),
    "power_spherical": (0.040, 0.110),
    "tripod_pinch": (0.006, 0.075),
    "handle_hook": (0.010, 0.050),
}

# Whole-body reference parameters (matching cup_grasp_demo.py & pour_beans_sharpa.py)
LIFT_TABLE_REF = 0.75  # m desk height corresponding to lift = 0.0 m
WAIST_PITCH_POWER = 0.25  # rad forward waist pitch for low/side power grasps
WAIST_PITCH_PINCH = 0.23  # rad forward waist pitch so low 2-4 cm desk items stay in reach
WAIST_PITCH_LIFT = 0.16  # rad upright target pitch during lift
LIFT_COLUMN_RAISE = 0.13  # m vertical lift-column raise for power_* / handle_hook
PINCH_ARM_LIFT = 0.14  # m vertical lift target for tripod_pinch
WAIST_YAW_START_X = 0.18  # m lateral threshold for gentle waist yaw toward target
WAIST_YAW_GAIN = 0.40  # rad/m
WAIST_YAW_MAX = 0.20  # rad
WAIST_LIMITS = np.array([[-2.65, 2.65], [-0.04, 1.00]], dtype=np.float64)
LIFT_LIMITS = np.array([[0.0, 0.55]], dtype=np.float64)

# Execution mode routing
PINCH_BRIDGE_TYPES = frozenset({"tripod_pinch"})
POWER_STAGED_TYPES = frozenset({"power_cylindrical", "power_spherical", "handle_hook"})

# IK filter callback: (wrist_pos_w, wrist_rot_w, grasp_type, width_m) -> (ok, penalty)
DexIKFilterFn = Callable[[np.ndarray, Rotation, str, float], tuple[bool, float]]


@dataclass(frozen=True)
class PlannedDexGrasp:
    """A single world-frame Sharpa Wave grasp candidate emitted by DexHandAffordancePlanner."""

    grasp_id: str
    hand_side: str  # "left" | "right"
    ee_frame: str  # "left_hand_flange" | "right_hand_flange"
    grasp_type: str  # power_cylindrical | power_spherical | tripod_pinch | handle_hook
    position: np.ndarray  # (3,) world grasp center
    rotation: Rotation  # world semantic frame [approach, closing, thumb]
    wrist_position: np.ndarray  # (3,) world {side}_hand_flange position
    wrist_rotation: Rotation  # world {side}_hand_flange rotation
    approach_dir: np.ndarray  # (3,) unit vector in world
    closing_axis: np.ndarray  # (3,) unit vector in world
    thumb_dir: np.ndarray  # (3,) unit vector in world
    width: float  # DR-scaled grasp width (m)
    preshape_qpos: np.ndarray  # (22,) rad, DR-adapted & contract-clipped
    target_qpos: np.ndarray  # (22,) rad, DR-adapted & contract-clipped
    thumb_first_joints: int
    thumb_lead_ratio: float
    part_tag: str
    score: float
    base_score: float
    verified: bool
    source: str  # "affordance_verified" | "affordance_geom"
    execution_mode: str  # "pinch_bridge" | "dex22_staged"
    name: str


@dataclass(frozen=True)
class PinchBridgeSpec:
    """Virtual parallel-gripper abstraction for `tripod_pinch` (mirrors PinchArm in block_pattern_demo.py)."""

    side: str
    pinch_center_hand: np.ndarray  # (3,) pinch center in {side}_hand_flange frame
    open_qpos: np.ndarray  # (22,)
    preshape_qpos: np.ndarray  # (22,)
    target_qpos: np.ndarray  # (22,)

    def openness_to_qpos(self, openness: float) -> np.ndarray:
        """Map scalar openness in [0 (closed/target), 1 (preshape/open)] to 22-DoF joint targets."""
        a = float(np.clip(openness, 0.0, 1.0))
        return (1.0 - a) * self.target_qpos + a * self.preshape_qpos


@dataclass(frozen=True)
class WholeBodyPosture:
    """Coordinated (waist [yaw, pitch], lift [m]) posture for grasping and lifting."""

    waist_grasp: np.ndarray  # (2,) [yaw, pitch] rad
    lift_grasp: np.ndarray  # (1,) m
    waist_lift: np.ndarray  # (2,) [yaw, pitch] rad
    lift_up: np.ndarray  # (1,) m
    use_column_lift: bool


def from_wxyz(quat_wxyz: Sequence[float]) -> Rotation:
    """Convert [w, x, y, z] quaternion to scipy Rotation."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def load_affordance(source: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load an affordance dictionary from a dict or JSON file path."""
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _normalize_scale(scale: float | Sequence[float], scale_ref: float) -> np.ndarray:
    arr = np.asarray(scale, dtype=np.float64)
    if arr.ndim == 0:
        vec = np.full(3, float(arr), dtype=np.float64)
    else:
        vec = arr.reshape(3).astype(np.float64)
    if np.any(vec <= 0.0):
        raise ValueError(f"Scale must be positive, got {scale!r}")
    return vec / max(float(scale_ref), 1e-9)


def _symmetry_angles(symmetry: dict[str, Any] | None, continuous_samples: int) -> list[float]:
    if not symmetry:
        return [0.0]
    mode = symmetry.get("z_axis", "none")
    if mode == "continuous":
        n = max(1, int(continuous_samples))
        return [2.0 * math.pi * k / n for k in range(n)]
    if mode == "discrete":
        fold = max(1, int(symmetry.get("fold", 1)))
        return [2.0 * math.pi * k / fold for k in range(fold)]
    return [0.0]


def adapt_qpos_for_scale(
    preshape_qpos: np.ndarray | Sequence[float],
    target_qpos: np.ndarray | Sequence[float],
    width_raw: float,
    width_scaled: float,
    *,
    side: str = "left",
    kin: dex_aff.SharpaWaveKinematics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt 22-DoF `(preshape_qpos, target_qpos)` for DR object scaling within contract limits.

    When `width_scaled > width_raw` (larger object), relaxes closure slightly toward `preshape_qpos`.
    When `width_scaled < width_raw` (smaller object), tightens closure moderately past `target_qpos`.
    Always clips both vectors to `model_contract.json` joint limits and `|MCP_AA| <= 0.04 rad`.
    """
    kin = kin or dex_aff.get_sharpa_kinematics()
    q_pre = np.asarray(preshape_qpos, dtype=np.float64).reshape(SHARPA_DOF).copy()
    q_tgt = np.asarray(target_qpos, dtype=np.float64).reshape(SHARPA_DOF).copy()

    ratio = float(width_scaled) / max(float(width_raw), 1e-6)
    if abs(ratio - 1.0) > 1e-4:
        if ratio > 1.0:
            # Larger object: reduce closure depth monotonically toward preshape
            alpha = float(np.clip(1.0 - 0.55 * (ratio - 1.0), 0.55, 1.0))
        else:
            # Smaller object: gently tighten closure
            alpha = float(np.clip(1.0 + 0.45 * (1.0 - ratio), 1.0, 1.25))
        q_tgt = q_pre + alpha * (q_tgt - q_pre)

    return (
        kin.clip_qpos(q_pre, side=side, enforce_mcp_aa=True),
        kin.clip_qpos(q_tgt, side=side, enforce_mcp_aa=True),
    )


def scale_dex_grasp_local(
    grasp: Mapping[str, Any],
    scale_vec: np.ndarray,
    *,
    sym_theta: float = 0.0,
    sym_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Apply symmetry rotation (unscaled frame) then DR scaling to a local Sharpa Wave grasp.

    Returns `(pos_sc, rot_sem_sc, wrist_pos_sc, wrist_rot_sc, width_sc)` in object local frame,
    preserving rigid hand dimensions while adjusting palm standoff by `0.5 * (width_sc - width_raw)`
    for power/hook grasps and preserving exact pinch-center offset for `tripod_pinch`.
    """
    p_loc = np.asarray(grasp["position"], dtype=np.float64).reshape(3)
    r_sem_loc = from_wxyz(grasp["quaternion_wxyz"]).as_matrix()
    p_w_loc = np.asarray(grasp["wrist_position"], dtype=np.float64).reshape(3)
    r_w_loc = from_wxyz(grasp["wrist_quaternion_wxyz"]).as_matrix()
    w_raw = float(grasp["width"])
    gtype = str(grasp.get("grasp_type", ""))

    # 1. Optional symmetry rotation in unscaled local frame about sym_axis
    if abs(sym_theta) > 1e-9:
        axis_pt = np.zeros(3, dtype=np.float64) if sym_axis is None else np.asarray(sym_axis, dtype=np.float64)
        c, s = math.cos(sym_theta), math.sin(sym_theta)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        p_loc = axis_pt + rz @ (p_loc - axis_pt)
        r_sem_loc = rz @ r_sem_loc
        p_w_loc = axis_pt + rz @ (p_w_loc - axis_pt)
        r_w_loc = rz @ r_w_loc

    # Fixed local relationship between semantic frame and wrist frame (rigid hand property)
    r_sem_to_wrist = r_sem_loc.T @ r_w_loc
    wrist_offset_in_wrist = r_w_loc.T @ (p_w_loc - p_loc)

    # 2. Scale anchor position and semantic frame axes
    p_sc = p_loc * scale_vec
    app_raw = r_sem_loc[:, 0] * scale_vec
    cls_raw = r_sem_loc[:, 1] * scale_vec

    # Width scaling: along closing_axis for pinch/hook; mean cross-sectional scale for power_*
    if gtype in ("power_cylindrical", "power_spherical"):
        # Cross-section perpendicular to thumb_dir (or approach & closing axes)
        a_scale = float(np.linalg.norm(app_raw))
        c_scale = float(np.linalg.norm(cls_raw))
        width_sc = w_raw * 0.5 * (a_scale + c_scale)
    else:
        width_sc = w_raw * float(np.linalg.norm(cls_raw))

    if width_sc < 1e-6:
        return None

    c_norm = float(np.linalg.norm(cls_raw))
    if c_norm < 1e-6:
        return None
    cls_u = cls_raw / c_norm

    app_orth = app_raw - float(np.dot(app_raw, cls_u)) * cls_u
    a_norm = float(np.linalg.norm(app_orth))
    if a_norm < 1e-6:
        return None
    app_u = app_orth / a_norm

    thm_u = np.cross(app_u, cls_u)
    t_norm = float(np.linalg.norm(thm_u))
    if t_norm < 1e-6:
        return None
    thm_u /= t_norm

    r_sem_sc = np.column_stack([app_u, cls_u, thm_u])
    r_w_sc = r_sem_sc @ r_sem_to_wrist

    # 3. Reconstruct rigid-hand wrist position on the scaled object
    p_w_sc = p_sc + r_w_sc @ wrist_offset_in_wrist
    if gtype in POWER_STAGED_TYPES:
        # Palm sits at surface distance 0.5 * width from anchor along -approach_dir
        p_w_sc = p_w_sc - 0.5 * (width_sc - w_raw) * app_u

    return p_sc, r_sem_sc, p_w_sc, r_w_sc, float(width_sc)


def build_pinch_bridge_spec(
    grasp: PlannedDexGrasp,
    *,
    kin: dex_aff.SharpaWaveKinematics | None = None,
) -> PinchBridgeSpec:
    """Create a `PinchBridgeSpec` mapping `tripod_pinch` to a virtual gripper frame."""
    kin = kin or dex_aff.get_sharpa_kinematics()
    geom = kin.tripod_pinch_geometry()
    center = geom.pinch_center.copy()
    if grasp.hand_side == "right":
        center = dex_aff.HAND_MIRROR_Y @ center
    open_q = kin.clip_qpos(np.zeros(SHARPA_DOF, dtype=np.float64), side=grasp.hand_side, enforce_mcp_aa=True)
    return PinchBridgeSpec(
        side=grasp.hand_side,
        pinch_center_hand=center,
        open_qpos=open_q,
        preshape_qpos=grasp.preshape_qpos.copy(),
        target_qpos=grasp.target_qpos.copy(),
    )


def compute_wholebody_posture(
    grasp_type: str,
    target_xy: Sequence[float],
    table_z: float,
    waist0: Sequence[float],
    lift0: float = 0.0,
    *,
    wrist_z: float | None = None,
) -> WholeBodyPosture:
    """Compute coordinated `(waist_grasp, lift_grasp, waist_lift, lift_up, use_column_lift)`.

    - `power_cylindrical` / `power_spherical` / `handle_hook`: lowers lift column to
      `clip(table_z - 0.75, 0.0, 0.25)` and pitches waist forward (`~0.25 rad` + reach adjustment)
      so low/side grasps stay well inside arm workspace; sets `use_column_lift=True` so lifting
      raises the lift column while holding arm joints steady (preventing wrist-singularity slip).
    - `tripod_pinch`: adapts `lift_grasp` from `wrist_z` (when provided) so top-down pinches on
      tall bottles/tumblers raise the torso cleanly while short desk items stay low.
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    w0 = np.asarray(waist0, dtype=np.float64).reshape(2)
    t_z = float(table_z)

    is_power = grasp_type in POWER_STAGED_TYPES
    if is_power or wrist_z is None:
        lift_grasp_val = float(np.clip(t_z - LIFT_TABLE_REF, 0.0, 0.25))
    else:
        # Top-down tripod_pinch: keep lift=0.0 for short desk items (< 8 cm tall), raise for tall bottles
        lift_grasp_val = float(np.clip(float(wrist_z) - 0.94, 0.0, 0.35))

    far_y = max(0.0, ty + 0.40)
    low_dz = max(0.0, LIFT_TABLE_REF - t_z)
    high_dz = max(0.0, t_z - LIFT_TABLE_REF)
    pitch_extra = float(np.clip(2.4 * low_dz - 1.0 * high_dz + 1.2 * far_y, -0.10, 0.38))

    base_pitch = WAIST_PITCH_POWER if is_power else WAIST_PITCH_PINCH
    pitch = float(np.clip(base_pitch + pitch_extra, WAIST_LIMITS[1, 0], WAIST_LIMITS[1, 1]))

    excess_x = max(0.0, abs(tx) - WAIST_YAW_START_X)
    yaw_delta = float(np.clip(-np.sign(tx) * WAIST_YAW_GAIN * excess_x, -WAIST_YAW_MAX, WAIST_YAW_MAX))
    yaw = float(np.clip(float(w0[0]) + yaw_delta, WAIST_LIMITS[0, 0], WAIST_LIMITS[0, 1]))

    waist_grasp = np.array([yaw, pitch], dtype=np.float64)
    lift_grasp = np.array([lift_grasp_val], dtype=np.float64)

    lift_raise = LIFT_COLUMN_RAISE if is_power else 0.13
    lift_up = np.array([float(np.clip(lift_grasp_val + lift_raise, LIFT_LIMITS[0, 0], LIFT_LIMITS[0, 1]))])
    # Hold torso pitch constant during lift so lift_joint_state translates the hand purely vertically
    # (un-pitching the waist while holding arm joints fixed tilts the wrist ~10 deg and pries objects loose).
    waist_lift = waist_grasp.copy()

    return WholeBodyPosture(
        waist_grasp=waist_grasp,
        lift_grasp=lift_grasp,
        waist_lift=waist_lift,
        lift_up=lift_up,
        use_column_lift=is_power,
    )


def predict_shoulder_base_transform(
    base_now: np.ndarray,
    lift_now: float,
    waist_now: Sequence[float],
    lift_target: float,
    waist_target: Sequence[float],
    *,
    waist_pivot_z_offset: float = 0.28,
) -> np.ndarray:
    """Predict 4x4 world shoulder base transform after moving `(lift_now, waist_now) -> (lift_target, waist_target)`.

    Used by whole-body coordinated IK filtering so arm reachability is evaluated at the planned
    grasp torso posture before moving the robot.
    """
    base = np.asarray(base_now, dtype=np.float64).copy()
    w0 = np.asarray(waist_now, dtype=np.float64).reshape(2)
    w1 = np.asarray(waist_target, dtype=np.float64).reshape(2)

    dz_lift = float(lift_target) - float(lift_now)
    dyaw = float(w1[0] - w0[0])
    dpitch = float(w1[1] - w0[1])

    # Approximate waist pivot directly below shoulder centreline
    pivot = base[:3, 3].copy()
    pivot[0] = 0.0
    pivot[2] = float(base[2, 3]) - float(waist_pivot_z_offset)

    # Robot faces +Y: +yaw rotates about +Z, +pitch leans torso forward toward +Y (rotation about -X)
    r_delta = Rotation.from_euler("z", dyaw).as_matrix() @ Rotation.from_euler("x", -dpitch).as_matrix()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_delta @ base[:3, :3]
    out[:3, 3] = pivot + r_delta @ (base[:3, 3] - pivot) + np.array([0.0, 0.0, dz_lift], dtype=np.float64)
    return out


def compute_dex_pre_grasp_waypoints(
    grasp: PlannedDexGrasp,
    *,
    table_z: float | None = None,
    slide_dist: float = 0.060,
    hover_dist: float = 0.120,
    lift_dist: float = 0.140,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(hover_wrist, pre_grasp_wrist, grip_wrist, lift_wrist)` in world coordinates."""
    grip_w = grasp.wrist_position.copy()
    app = grasp.approach_dir / max(float(np.linalg.norm(grasp.approach_dir)), 1e-12)
    lift_w = grip_w + np.array([0.0, 0.0, float(lift_dist)], dtype=np.float64)

    if float(app[2]) <= -0.80:
        # Top-down approach (e.g. tripod_pinch / top power_spherical): descend along -approach_dir
        pre_w = grip_w - float(slide_dist) * app
        hover_w = grip_w + np.array([0.0, 0.0, max(float(hover_dist), float(slide_dist) + 0.04)], dtype=np.float64)
        return hover_w, pre_w, grip_w, lift_w

    pre_w = grip_w - float(slide_dist) * app
    if table_z is not None:
        pre_w[2] = max(float(pre_w[2]), float(table_z) + 0.025)
    hover_w = np.array(
        [pre_w[0], pre_w[1], max(float(pre_w[2]) + float(hover_dist), float(lift_w[2]))],
        dtype=np.float64,
    )
    return hover_w, pre_w, grip_w, lift_w


def _collides_obstacles(
    test_pts: Sequence[np.ndarray],
    obstacles: Sequence[Mapping[str, Any]],
    margin: float = 0.015,
) -> bool:
    """Check whether any hand or corridor point collides with obstacle spheres or AABBs."""
    for obs in obstacles:
        if "position" in obs and "radius" in obs:
            center = np.asarray(obs["position"], dtype=np.float64).reshape(3)
            rad = float(obs["radius"]) + margin
            rad_sq = rad * rad
            for pt in test_pts:
                diff = pt - center
                if float(np.dot(diff, diff)) < rad_sq:
                    return True
        elif "min" in obs and "max" in obs:
            lo = np.asarray(obs["min"], dtype=np.float64).reshape(3) - margin
            hi = np.asarray(obs["max"], dtype=np.float64).reshape(3) + margin
            for pt in test_pts:
                if bool(np.all((pt >= lo) & (pt <= hi))):
                    return True
    return False


class DexHandAffordancePlanner:
    """Pure-CPU Sharpa Wave grasp planner consuming `affordance.json['grasps']['sharpa_wave']`."""

    def __init__(
        self,
        *,
        continuous_sym_samples: int = 8,
        table_clear: float = 0.004,
        max_approach_z: float = 0.20,
        obstacle_margin: float = 0.015,
        slide_dist: float = 0.060,
        kin: dex_aff.SharpaWaveKinematics | None = None,
    ) -> None:
        self.continuous_sym_samples = int(continuous_sym_samples)
        self.table_clear = float(table_clear)
        self.max_approach_z = float(max_approach_z)
        self.obstacle_margin = float(obstacle_margin)
        self.slide_dist = float(slide_dist)
        self.kin = kin or dex_aff.get_sharpa_kinematics()

    def plan_grasps(
        self,
        affordance: dict[str, Any] | str | Path,
        obj_pos: Sequence[float],
        obj_quat_wxyz: Sequence[float],
        *,
        hand_side: str = "left",
        scale: float | Sequence[float] = 1.0,
        table_z: float | None = None,
        part_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        prefer_tags: Sequence[str] | Mapping[str, float] | None = None,
        grasp_types: Sequence[str] | None = None,
        verified_only: bool = False,
        skip_sim_failed: bool = True,
        obstacles: Sequence[Mapping[str, Any]] | None = None,
        ik_filter: DexIKFilterFn | None = None,
        top_k: int = 16,
        sample_temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[PlannedDexGrasp]:
        """Transform, mirror, DR-adapt, filter, and rank Sharpa Wave grasp candidates."""
        if hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

        data = load_affordance(affordance)
        raw_grasps = (data.get("grasps") or {}).get(SHARPA_GRIPPER_KEY) or []
        if not raw_grasps:
            return []

        scale_ref = float(data.get("scale_ref", 1.0))
        scale_vec = _normalize_scale(scale, scale_ref)
        sym_meta = data.get("symmetry") or {}
        sym_angles = _symmetry_angles(sym_meta, self.continuous_sym_samples)
        raw_axis = sym_meta.get("axis_xy")
        sym_axis = (
            np.array([float(raw_axis[0]), float(raw_axis[1]), 0.0], dtype=np.float64)
            if (raw_axis is not None and len(raw_axis) >= 2)
            else np.zeros(3, dtype=np.float64)
        )

        p_obj = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        r_obj = from_wxyz(obj_quat_wxyz).as_matrix()

        allow_tags = set(part_tags) if part_tags is not None else None
        deny_tags = set(exclude_tags) if exclude_tags is not None else set()
        allow_types = set(grasp_types) if grasp_types is not None else None
        obs_list = list(obstacles) if obstacles else []

        prefer_bonuses: dict[str, float] = {}
        if isinstance(prefer_tags, Mapping):
            prefer_bonuses = {str(k): float(v) for k, v in prefer_tags.items()}
        elif prefer_tags is not None:
            pref_list = [str(t) for t in prefer_tags]
            n_pref = len(pref_list)
            for rank, t_name in enumerate(pref_list):
                if t_name not in prefer_bonuses:
                    prefer_bonuses[t_name] = 0.35 if n_pref <= 1 else (0.35 - 0.30 * (rank / (n_pref - 1)))

        # Expected ergonomic horizontal approach direction: left arm approaches from -X (+X into obj),
        # right arm approaches from +X (-X into obj), and both prefer approaching from robot side (+Y).
        side_in_x = 1.0 if hand_side == "left" else -1.0
        candidates: list[PlannedDexGrasp] = []
        cap_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for g_raw in raw_grasps:
            gtype = str(g_raw.get("grasp_type", ""))
            if gtype not in GRASP_TYPE_WIDTH_BOUNDS:
                continue
            if allow_types is not None and gtype not in allow_types:
                continue
            tag = str(g_raw.get("part_tag", "unknown"))
            if allow_tags is not None and tag not in allow_tags:
                continue
            if tag in deny_tags:
                continue

            is_verified = bool(g_raw.get("verified", False))
            if verified_only and not is_verified:
                continue
            sim_score = g_raw.get("sim_score")
            sim_tested = sim_score is not None or g_raw.get("sim_metrics") is not None
            if skip_sim_failed and sim_tested and not is_verified:
                continue

            # Deterministic left -> right hand mirroring before symmetry/scale expansion
            g = (
                dex_aff.mirror_grasp_left_to_right(g_raw)
                if (hand_side == "right" and g_raw.get("hand_side", "left") == "left")
                else g_raw
            )
            w_raw = float(g["width"])
            w_lo, w_hi = GRASP_TYPE_WIDTH_BOUNDS[gtype]
            base_score = float(sim_score) if sim_score is not None else float(g.get("score", 0.5))
            gid = str(g.get("id", "sw"))
            sched = g.get("close_schedule") or {}
            thumb_first = int(sched.get("thumb_first_joints", 5 if gtype in POWER_STAGED_TYPES else 0))
            thumb_lead = float(sched.get("thumb_lead_ratio", 0.35 if thumb_first > 0 else 0.0))
            exec_mode = "pinch_bridge" if gtype in PINCH_BRIDGE_TYPES else "dex22_staged"
            source_tag = "affordance_verified" if is_verified else "affordance_geom"

            for sym_idx, theta in enumerate(sym_angles):
                scaled = scale_dex_grasp_local(g, scale_vec, sym_theta=theta, sym_axis=sym_axis)
                if scaled is None:
                    continue
                p_sc_base, r_sem_sc_base, p_w_sc_base, r_w_sc_base, width_sc = scaled
                if width_sc < w_lo or width_sc > w_hi:
                    continue

                # For non-symmetric objects, tripod_pinch is still 180-deg antipodally symmetric around
                # approach_dir at the pinch center, giving the arm both wrist-yaw orientations.
                flip_modes = (False, True) if (gtype == "tripod_pinch" and len(sym_angles) == 1) else (False,)
                for flipped in flip_modes:
                    if not flipped:
                        p_sc, r_sem_sc, p_w_sc, r_w_sc = p_sc_base, r_sem_sc_base, p_w_sc_base, r_w_sc_base
                    else:
                        # Rotate 180 deg about local approach_dir at p_sc_base: R_flip = 2 * a a^T - I
                        a_loc = r_sem_sc_base[:, 0]
                        r_flip = 2.0 * np.outer(a_loc, a_loc) - np.eye(3, dtype=np.float64)
                        p_sc = p_sc_base
                        r_sem_sc = r_flip @ r_sem_sc_base
                        p_w_sc = p_sc_base + r_flip @ (p_w_sc_base - p_sc_base)
                        r_w_sc = r_flip @ r_w_sc_base

                    p_world = p_obj + r_obj @ p_sc
                    r_sem_w = r_obj @ r_sem_sc
                    wrist_p_w = p_obj + r_obj @ p_w_sc
                    wrist_r_w = r_obj @ r_w_sc

                    app_w = r_sem_w[:, 0]
                    cls_w = r_sem_w[:, 1]
                    thm_w = r_sem_w[:, 2]

                    # Reject upward approaches (coming from below tabletop)
                    if float(app_w[2]) > self.max_approach_z:
                        continue

                    q_pre, q_tgt = adapt_qpos_for_scale(
                        g["preshape_qpos"],
                        g["target_qpos"],
                        w_raw,
                        width_sc,
                        side=hand_side,
                        kin=self.kin,
                    )

                    # Tabletop clearance check using exact URDF FK hand capsules at preshape_qpos
                    if table_z is not None:
                        cache_key = (gid, f"{hand_side}_{round(width_sc, 4)}")
                        if cache_key not in cap_cache:
                            cap_cache[cache_key] = self.kin.link_capsules(q_pre, side=hand_side)
                        cap_p0, cap_p1, cap_r = cap_cache[cache_key]
                        a_z = (cap_p0 @ wrist_r_w[2, :]) + float(wrist_p_w[2]) - cap_r
                        b_z = (cap_p1 @ wrist_r_w[2, :]) + float(wrist_p_w[2]) - cap_r
                        min_hand_z = float(min(np.min(a_z), np.min(b_z)))
                        if min_hand_z < float(table_z) - self.table_clear:
                            continue

                    # Distractor obstacle collision check (hand anchor + wrist + approach corridor)
                    if obs_list:
                        palm_w = wrist_p_w + wrist_r_w @ (
                            dex_aff.PALM_SURFACE_POINT
                            * (np.array([1.0, -1.0, 1.0]) if hand_side == "right" else np.ones(3))
                        )
                        pre_w = wrist_p_w - self.slide_dist * app_w
                        pre_p = p_world - self.slide_dist * app_w
                        test_pts = (p_world, wrist_p_w, palm_w, pre_w, pre_p)
                        if _collides_obstacles(test_pts, obs_list, margin=self.obstacle_margin):
                            continue

                    wrist_rot = Rotation.from_matrix(wrist_r_w)
                    ik_penalty = 0.0
                    if ik_filter is not None:
                        ok, pen = ik_filter(wrist_p_w, wrist_rot, gtype, width_sc)
                        if not ok:
                            continue
                        ik_penalty = float(pen)

                    # Composite score: verified bonus + arm-side ergonomic alignment + tag preference - ik_penalty
                    score = base_score + (0.15 if is_verified else 0.0)
                    # Prefer approaches coming from the active arm's side and from the robot (+Y)
                    score += 0.08 * float(app_w[0] * side_in_x)
                    score += 0.06 * float(np.clip(app_w[1], -0.5, 1.0))
                    # Prefer thumb pointing upward for power/hook grasps (avoids inverted wrist)
                    if gtype in POWER_STAGED_TYPES:
                        score += 0.08 * float(np.clip(thm_w[2], -1.0, 1.0))
                    else:
                        # For tripod_pinch, strongly reward vertical top-down approach and wrist clearance above desk
                        # so 45-deg tilted low-wrist grasps don't jam knuckles/forearm into the tabletop.
                        score += 0.28 * float(np.clip(-app_w[2], 0.0, 1.0))
                        if table_z is not None:
                            wrist_clear = float(wrist_p_w[2]) - float(table_z)
                            if wrist_clear < 0.125:
                                score -= 1.5 * (0.125 - wrist_clear)
                    if prefer_bonuses:
                        score += prefer_bonuses.get(tag, 0.0)
                    score -= ik_penalty

                    suffix = f"_s{sym_idx}" if sym_idx else ""
                    if flipped:
                        suffix += "_f"
                    candidates.append(
                        PlannedDexGrasp(
                            grasp_id=f"{gid}{suffix}",
                            hand_side=hand_side,
                            ee_frame=f"{hand_side}_hand_flange",
                            grasp_type=gtype,
                            position=p_world.copy(),
                            rotation=Rotation.from_matrix(r_sem_w),
                            wrist_position=wrist_p_w.copy(),
                            wrist_rotation=wrist_rot,
                            approach_dir=app_w.copy(),
                            closing_axis=cls_w.copy(),
                            thumb_dir=thm_w.copy(),
                            width=width_sc,
                            preshape_qpos=q_pre,
                            target_qpos=q_tgt,
                            thumb_first_joints=thumb_first,
                            thumb_lead_ratio=thumb_lead,
                            part_tag=tag,
                            score=round(float(score), 5),
                            base_score=round(float(base_score), 5),
                            verified=is_verified,
                            source=source_tag,
                            execution_mode=exec_mode,
                            name=f"sw:{gtype}:{tag}:{gid}{suffix}",
                        )
                    )

        if not candidates:
            return []

        # Deduplicate near-identical world wrist poses
        candidates.sort(key=lambda c: c.score, reverse=True)
        deduped: list[PlannedDexGrasp] = []
        cos_ang = math.cos(math.radians(15.0))
        for cand in candidates:
            if len(deduped) >= max(top_k * 2, 32):
                break
            dup = False
            for kept in deduped:
                if (
                    cand.grasp_type == kept.grasp_type
                    and np.linalg.norm(cand.wrist_position - kept.wrist_position) < 0.015
                ):
                    if float(np.dot(cand.approach_dir, kept.approach_dir)) >= cos_ang:
                        dup = True
                        break
            if not dup:
                deduped.append(cand)

        top = deduped[:top_k]
        if sample_temperature > 0.0 and len(top) > 1:
            gen = rng if rng is not None else np.random.default_rng()
            scores = np.array([c.score for c in top], dtype=np.float64)
            logits = (scores - scores.max()) / float(sample_temperature)
            probs = np.exp(logits)
            probs /= probs.sum()
            order = gen.choice(len(top), size=len(top), replace=False, p=probs)
            top = [top[int(i)] for i in order]
        return top


def classify_pick_failure(
    poses: np.ndarray,
    labels: Sequence[str],
    start: np.ndarray,
    *,
    lift_thresh: float = 0.08,
    stay_xy: float = 0.03,
    stay_z: float = 0.02,
    hold_frames: int = 30,
) -> str:
    """Attribute episode outcome to a deterministic failure category for smoke-test diagnostics."""
    poses_arr = np.asarray(poses, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64)
    if poses_arr.ndim != 3 or len(poses_arr) == 0:
        return "empty_trajectory"

    t_idx = list(labels).index("target")
    target_dz = poses_arr[:, t_idx, 2] - start_arr[t_idx, 2]
    max_dz = float(np.max(target_dz))
    tail_n = min(int(hold_frames), len(poses_arr))
    tail_min_dz = float(np.min(target_dz[-tail_n:]))

    # Check distractors in tail
    distractor_bad = False
    for i, lbl in enumerate(labels):
        if lbl == "target":
            continue
        move = poses_arr[-tail_n:, i, :3] - start_arr[i][None, :]
        if (
            float(np.max(np.linalg.norm(move[:, :2], axis=-1))) >= stay_xy
            or float(np.max(np.abs(move[:, 2]))) >= stay_z
        ):
            distractor_bad = True
            break

    if tail_min_dz >= lift_thresh and not distractor_bad:
        return "ok"
    if tail_min_dz >= lift_thresh and distractor_bad:
        return "distractor_moved"
    if max_dz >= lift_thresh and tail_min_dz < lift_thresh:
        return "dropped_during_lift"
    if max_dz >= 0.02:
        return "insufficient_lift"
    if distractor_bad:
        return "distractor_and_no_lift"
    return "no_lift"


# ---------------------------------------------------------------------------
# Online env sidecar resolution & dual-mode Sharpa Wave pick execution
# ---------------------------------------------------------------------------


def resolve_affordance_for_env(
    env: Any,
    label: str = "target",
    *,
    affordance_map: Mapping[str, dict[str, Any] | str | Path] | None = None,
    affordance_dir: str | Path | None = None,
    assets_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Locate `affordance.json` sidecar for `label` in `env` (supports `OPENFLYWHEEL_AFFORDANCE_DIR`)."""
    if affordance_map and label in affordance_map:
        return load_affordance(affordance_map[label])

    env_aff_dir = os.environ.get("OPENFLYWHEEL_AFFORDANCE_DIR", "").strip()
    aff_dir = Path(affordance_dir) if affordance_dir is not None else (Path(env_aff_dir) if env_aff_dir else None)

    entry: Mapping[str, Any] | None = None
    if hasattr(env, "entries"):
        try:
            entry = (env.entries(0) or {}).get(label)
        except Exception:
            entry = None
    if not isinstance(entry, Mapping):
        return None

    if isinstance(entry.get("affordance"), Mapping):
        return dict(entry["affordance"])
    if entry.get("affordance_path") and Path(entry["affordance_path"]).is_file():
        return load_affordance(entry["affordance_path"])

    cat = entry.get("category")
    idx = entry.get("category_idx")
    if cat is not None and idx is not None and aff_dir is not None:
        override = aff_dir / str(cat) / f"{int(idx):05d}" / "affordance.json"
        if override.is_file():
            return load_affordance(override)

    usd_path = entry.get("usd_path") or entry.get("path")
    if usd_path:
        sidecar = Path(usd_path).parent / "affordance.json"
        if sidecar.is_file():
            return load_affordance(sidecar)
        if assets_root is not None:
            sidecar2 = Path(assets_root) / Path(usd_path).parent / "affordance.json"
            if sidecar2.is_file():
                return load_affordance(sidecar2)

    if cat is not None and idx is not None:
        try:
            from env.global_configs import ASSETS_PATH, object_dir

            assets = str(assets_root) if assets_root is not None else ASSETS_PATH
            obj_dir = Path(object_dir(f"Rigid/{cat}/{int(idx):05d}", assets=assets))
            sidecar = obj_dir / "affordance.json"
            if sidecar.is_file():
                return load_affordance(sidecar)
        except Exception:
            pass
    return None


def _collect_scene_obstacles(env: Any, exclude_label: str = "target") -> list[dict[str, Any]]:
    """Build bounding-sphere obstacles for all non-target objects in the layout."""
    from scripts.g1d import scripted_demo as sd

    lm = env.scene_manager.layout_manager
    entries = env.entries(0) if hasattr(env, "entries") else {}
    obstacles: list[dict[str, Any]] = []
    dummy = type("_R", (), {"lm": lm})()
    for lbl in entries:
        if lbl == exclude_label:
            continue
        try:
            pos, _, ext, _, _ = sd.Robot.object(dummy, lbl)
            radius = 0.5 * float(np.linalg.norm(ext[:2]))
            obstacles.append({"position": np.asarray(pos, dtype=np.float64), "radius": max(radius, 0.02)})
        except Exception:
            continue
    return obstacles


def _make_wholebody_ik_filter(
    arm: Any,
    posture_fn: Callable[[str, float], WholeBodyPosture],
    lift0: float,
    waist0: np.ndarray,
    table_z: float,
    *,
    pos_tol_m: float = 0.012,
    rot_tol_deg: float = 12.0,
    live_base: bool = False,
) -> DexIKFilterFn:
    """Build a coordinated whole-body `(lift, waist_pitch, arm_q7)` IK filter for `arm`.

    When `live_base=True` (torso already positioned at grasp posture), evaluates exact incremental
    step-IK rollout (`_rollout_step_ik`) on the live shoulder base so selected candidates are
    guaranteed to converge cleanly under `Arm.ik`.
    """
    from scripts.g1d import affordance_grasp as aff_g, scripted_demo as sd

    q0 = arm.q()
    orig_reach = arm.REACH
    arm.REACH = np.zeros(3, dtype=np.float64)
    try:
        reach_solver = sd.Reach(arm)
    except Exception as exc:
        print(f"[sharpa_aff] warning: sd.Reach(arm) failed ({exc}); IK filter disabled", flush=True)
        return lambda _p, _r, _gt, _w: (True, 0.0)
    finally:
        arm.REACH = orig_reach

    base_orig = reach_solver.base.copy()

    def _filter(wrist_p: np.ndarray, wrist_r: Rotation, grasp_type: str, _width: float) -> tuple[bool, float]:
        if live_base:
            reach_solver.base = base_orig
        else:
            posture = posture_fn(grasp_type, float(wrist_p[2]))
            reach_solver.base = predict_shoulder_base_transform(
                base_orig,
                lift0,
                waist0,
                float(posture.lift_grasp[0]),
                posture.waist_grasp,
            )
        # 1. Fast global IK check at target grip pose
        q_grip, perr, rerr = reach_solver.solve(wrist_p, wrist_r, q0)
        if perr > pos_tol_m or rerr > rot_tol_deg:
            return False, 1.0
        # 2. Verify incremental step-IK path from q0 -> hover -> grip converges without stalling
        p_hover = wrist_p + np.array([0.0, 0.0, 0.07], dtype=np.float64)
        roll_pos, roll_deg = aff_g._rollout_step_ik(
            reach_solver,
            q0,
            [(p_hover, wrist_r, 35, False), (wrist_p, wrist_r, 30, True)],
            sd.waypoints,
            settle_steps=25,
            dq_max=0.026,
        )
        if roll_pos > pos_tol_m or roll_deg > rot_tol_deg:
            return False, 1.0
        dq = float(np.linalg.norm(q_grip - q0))
        penalty = 0.10 * (roll_pos / max(pos_tol_m, 1e-6)) + 0.05 * (roll_deg / max(rot_tol_deg, 1e-6)) + 0.03 * dq
        return True, penalty

    return _filter


def pick_anything_sharpa_affordance_actions(
    env: Any,
    *,
    affordance_map: Mapping[str, dict[str, Any] | str | Path] | None = None,
    affordance_dir: str | Path | None = None,
    verified_only: bool = False,
    part_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    prefer_tags: Sequence[str] | Mapping[str, float] | None = None,
    grasp_types: Sequence[str] | None = None,
    use_tactile_gate: bool = False,
    info_out: dict[str, Any] | None = None,
):
    """Yield 61D wholebody Sharpa Wave actions for `g1d_pick_anything` using `affordance.json`.

    Dual-mode execution:
      - `tripod_pinch` -> virtual pinch-center bridge (`PinchBridgeSpec`) + arm/hybrid lift
      - `power_cylindrical` / `power_spherical` / `handle_hook` -> staged 22-DoF closure
        (`open -> preshape -> slide_in -> thumb_first -> finger_wrap -> column_lift`)
    Falls back automatically to `x_anything_sharpa.sharpa_pick_anything_actions(env)` when
    no `affordance.json` sidecar or reachable candidate applies.
    """
    from scripts.dexhands import pour_beans_sharpa as pbs, x_anything_sharpa as xas
    from scripts.g1d import scripted_demo as sd

    info = info_out if info_out is not None else {}
    lm = env.scene_manager.layout_manager
    obj_pos_raw, obj_quat_raw = lm.get_instance_pose(0, label="target")
    obj_pos = sd.as_np(obj_pos_raw)
    obj_quat = sd.as_np(obj_quat_raw)
    centre, _, ext, bottom, mass = xas._target_geometry(env)
    entries = env.entries(0) if hasattr(env, "entries") else {}
    target_entry = entries.get("target") or {}
    scale_val = target_entry.get("scale", 1.0)
    table_z = float(bottom)

    aff_data = resolve_affordance_for_env(
        env,
        "target",
        affordance_map=affordance_map,
        affordance_dir=affordance_dir,
    )
    raw_sw = ((aff_data or {}).get("grasps") or {}).get(SHARPA_GRIPPER_KEY) or []
    info["candidates_raw"] = len(raw_sw)

    # Set up whole-body session with zero reach (direct {side}_hand_flange control)
    robots = [r for r in env.robot_manager.robot_list if r.type == "target"]
    sess = xas._WholeBodySession(env, [np.zeros(3, dtype=np.float64) for _ in robots])
    primary_side = "left" if float(centre[0]) <= 0.0 else "right"
    other_side = "right" if primary_side == "left" else "left"

    # 0. Pre-position torso (waist + lift) FIRST so sd.Reach(arm) & _rollout_step_ik run on the
    #    exact live shoulder base transform (zero approximation error!).
    est_wrist_z = float(bottom + ext[2]) + 0.16
    init_posture = compute_wholebody_posture(
        "tripod_pinch", centre[:2], table_z, sess.waist0, float(sess.lift0[0]), wrist_z=est_wrist_z
    )
    yield from sess.hold_steps(24, waist=init_posture.waist_grasp, lift=init_posture.lift_grasp)

    chosen: PlannedDexGrasp | None = None
    obstacles = _collect_scene_obstacles(env, exclude_label="target")

    if raw_sw:
        # Try strict 6 mm tabletop capsule slack first; relax once to 12 mm only if needed for flat 2 cm objects.
        for t_clear in (0.006, 0.012):
            planner = DexHandAffordancePlanner(table_clear=t_clear)
            for side in (primary_side, other_side):
                if side not in sess.by_side:
                    continue
                arm = sess.arms[sess.by_side[side]]

                def _posture_for(gt: str, wz: float) -> WholeBodyPosture:
                    return init_posture

                ik_fn = _make_wholebody_ik_filter(
                    arm, _posture_for, float(sess.cur_lift[0]), sess.cur_waist, table_z, live_base=True
                )
                cands = planner.plan_grasps(
                    aff_data,
                    obj_pos,
                    obj_quat,
                    hand_side=side,
                    scale=scale_val,
                    table_z=table_z,
                    part_tags=part_tags,
                    exclude_tags=exclude_tags,
                    prefer_tags=prefer_tags,
                    grasp_types=grasp_types,
                    verified_only=verified_only,
                    obstacles=obstacles,
                    ik_filter=ik_fn,
                    top_k=8,
                )
                info["candidates_after_filter"] = len(cands)
                if cands:
                    chosen = cands[0]
                    break
            if chosen is not None:
                break

    if chosen is None:
        # Automatic fallback: try refitted top-down tripod_pinch across narrow side for compact/unverified objects,
        # searching both arms and yaw orientations and requiring IK convergence; otherwise fall back to geometric
        # power_wrap with pure-vertical column lift.
        kin = dex_aff.get_sharpa_kinematics()
        narrow_w = float(min(ext[0], ext[1]))
        if narrow_w <= 0.072 and float(ext[2]) <= 0.14:
            geom = kin.tripod_pinch_geometry()
            H_pinch = np.column_stack([geom.approach_dir, geom.closing_axis, geom.thumb_dir])
            _, obj_rot, _, _, _ = xas._target_geometry(env)
            r_mat = obj_rot.as_matrix()
            cls_axis = r_mat[:, 0] if float(ext[0]) <= float(ext[1]) else r_mat[:, 1]
            cls_axis = np.array([float(cls_axis[0]), float(cls_axis[1]), 0.0], dtype=np.float64)
            cn = float(np.linalg.norm(cls_axis))
            cls_axis = cls_axis / cn if cn > 1e-6 else np.array([1.0, 0.0, 0.0])
            app_u = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            pinch_z = bottom + min(0.65 * float(ext[2]), max(0.004, float(ext[2]) - 0.008))
            pinch_pt = np.array([float(centre[0]), float(centre[1]), pinch_z])
            w_use = max(0.010, narrow_w)
            q_pre_l, q_tgt_l = dex_aff.tripod_pinch_qpos(w_use, kin)
            best_fb = None
            for fb_side in (primary_side, other_side):
                if fb_side not in sess.by_side:
                    continue
                f_arm_live = sess.arms[sess.by_side[fb_side]]
                f_ik = _make_wholebody_ik_filter(
                    f_arm_live,
                    lambda *_: init_posture,
                    float(sess.cur_lift[0]),
                    sess.cur_waist,
                    table_z,
                    live_base=True,
                )
                side_pen = 0.0 if fb_side == primary_side else 0.10
                for yaw_deg in (0.0, 180.0, 20.0, -20.0, 160.0, -160.0):
                    r_yaw = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
                    c_vec = r_yaw @ cls_axis
                    t_vec = np.cross(app_u, c_vec)
                    t_vec /= max(float(np.linalg.norm(t_vec)), 1e-12)
                    w_rot_l = np.column_stack([app_u, c_vec, t_vec]) @ H_pinch.T
                    if fb_side == "right":
                        w_rot_u = Rotation.from_matrix(
                            (np.eye(3) - 2.0 * np.outer(c_vec, c_vec)) @ w_rot_l @ dex_aff.HAND_MIRROR_Y
                        )
                        c_h = dex_aff.HAND_MIRROR_Y @ geom.pinch_center
                    else:
                        w_rot_u = Rotation.from_matrix(w_rot_l)
                        c_h = geom.pinch_center.copy()
                    w_p_u = pinch_pt - w_rot_u.apply(c_h)
                    ok_fb, pen_fb = f_ik(w_p_u, w_rot_u, "tripod_pinch", w_use)
                    if not ok_fb:
                        continue
                    yaw_pen = 0.0 if abs(yaw_deg % 180.0) < 1e-3 else 0.05
                    cand_fb = (side_pen + yaw_pen + pen_fb, fb_side, c_vec, t_vec, w_p_u, w_rot_u)
                    if best_fb is None or cand_fb[0] < best_fb[0]:
                        best_fb = cand_fb
                if best_fb is not None:
                    break
            if best_fb is not None:
                _, fb_side, cls_axis, thm_u, wrist_p_use, wrist_rot_use = best_fb
                q_pre_fb = dex_aff.RIGHT_JOINT_SIGNS * q_pre_l if fb_side == "right" else q_pre_l
                q_tgt_fb = dex_aff.RIGHT_JOINT_SIGNS * q_tgt_l if fb_side == "right" else q_tgt_l
                chosen = PlannedDexGrasp(
                    grasp_id="fallback_pinch",
                    hand_side=fb_side,
                    ee_frame=f"{fb_side}_hand_flange",
                    grasp_type="tripod_pinch",
                    position=pinch_pt,
                    rotation=Rotation.from_matrix(np.column_stack([app_u, cls_axis, thm_u])),
                    wrist_position=wrist_p_use,
                    wrist_rotation=wrist_rot_use,
                    approach_dir=app_u,
                    closing_axis=cls_axis,
                    thumb_dir=thm_u,
                    width=w_use,
                    preshape_qpos=kin.clip_qpos(q_pre_fb, side=fb_side, enforce_mcp_aa=True),
                    target_qpos=kin.clip_qpos(q_tgt_fb, side=fb_side, enforce_mcp_aa=True),
                    thumb_first_joints=0,
                    thumb_lead_ratio=0.0,
                    part_tag="body_top",
                    score=0.5,
                    base_score=0.5,
                    verified=False,
                    source="fallback_heuristic",
                    execution_mode="pinch_bridge",
                    name="fallback:tripod_pinch",
                )
                print(f"[sharpa_aff] fallback -> refitted tripod_pinch ({fb_side}) w={w_use:.3f}", flush=True)
        if chosen is None:
            plan = xas._grasp_plan(env)
            prim = plan["primitive"]
            info.update(
                {
                    "grasp_source": "fallback_heuristic",
                    "grasp_type": prim,
                    "grasp_id": "fallback",
                    "execution_mode": "fallback_x_anything_sharpa",
                    "hand_side": plan["side"],
                    "part_tag": "unknown",
                    "verified": False,
                }
            )
            print(
                f"[sharpa_aff] fallback -> {prim} ({plan['side']}) raw={info.get('candidates_raw', 0)}",
                flush=True,
            )
            f_idx = sess.by_side[plan["side"]]
            f_arm = sess.arms[f_idx]
            f_arm.REACH = np.asarray(plan["reach"], dtype=float)
            open_q, pre_q, thumb_q, closed_q = xas.primitive_hand_poses(prim, plan["half_width"], f_arm.hand_limits)
            w_grasp, l_grasp = xas.compute_wholebody_grasp_pose(plan["bottom"], plan["centre"][:2], sess.waist0)
            l_up = np.array([float(np.clip(l_grasp[0] + LIFT_COLUMN_RAISE, 0.0, 0.45))], dtype=float)
            start = plan["grip"] - xas.SLIDE * plan["palm_in"]
            yield from sess.move_arm(
                f_idx,
                start + np.array([0.0, 0.0, xas.HOVER]),
                plan["hand_rot"],
                hand_q=pre_q,
                waist=w_grasp,
                lift=l_grasp,
                steps=45,
            )
            yield from sess.move_arm(f_idx, start, plan["hand_rot"], hand_q=pre_q, steps=24)
            yield from sess.move_arm(f_idx, plan["grip"], plan["hand_rot"], hand_q=pre_q, steps=26)
            yield from sess.close_with_tactile_gate(f_idx, prim, pre_q, thumb_q, closed_q)
            yield from sess.hold_steps(8, loaded={f_arm.arm_key: True})
            yield from sess.hold_steps(36, waist=w_grasp, lift=l_up, loaded={f_arm.arm_key: True})
            yield from sess.hold_steps(35, loaded={f_arm.arm_key: True})
            return

    info.update(
        {
            "grasp_source": chosen.source,
            "grasp_type": chosen.grasp_type,
            "grasp_id": chosen.grasp_id,
            "execution_mode": chosen.execution_mode,
            "hand_side": chosen.hand_side,
            "part_tag": chosen.part_tag,
            "verified": bool(chosen.verified),
            "score": float(chosen.score),
        }
    )
    print(
        f"[sharpa_aff] selected {chosen.name} side={chosen.hand_side} mode={chosen.execution_mode} "
        f"verified={chosen.verified} score={chosen.score:.3f}",
        flush=True,
    )

    idx = sess.by_side[chosen.hand_side]
    arm = sess.arms[idx]
    robot = sess.robots[idx]
    # Keep the live torso posture that IK was solved against; lift vertically by +0.13 m
    posture = WholeBodyPosture(
        waist_grasp=init_posture.waist_grasp.copy(),
        lift_grasp=init_posture.lift_grasp.copy(),
        waist_lift=init_posture.waist_grasp.copy(),
        lift_up=np.array(
            [float(np.clip(init_posture.lift_grasp[0] + LIFT_COLUMN_RAISE, LIFT_LIMITS[0, 0], LIFT_LIMITS[0, 1]))],
            dtype=np.float64,
        ),
        use_column_lift=True,
    )

    def _track_peak(peak_now: float) -> float:
        f = xas.read_fingertip_forces(env, robot)
        return max(peak_now, float(np.max(f))) if f is not None else peak_now

    def _settle_arm(target_pos: np.ndarray, target_rot: Rotation, hand_q: np.ndarray, max_steps: int = 35):
        """Closed-loop DLS IK settle until position < 4 mm and orientation < 3.5 deg."""
        for _ in range(max_steps):
            perr = float(np.linalg.norm(arm.grasp_point() - target_pos))
            rerr = float(np.degrees((target_rot * arm.pose()[1].inv()).magnitude()))
            if perr < 0.004 and rerr < 3.5:
                break
            q_ik = arm.ik(target_pos, target_rot, smooth=False)
            yield from sess.emit(q_map={arm.arm_key: q_ik}, h_map={arm.grip_key: hand_q}, loaded={arm.arm_key: False})

    if chosen.execution_mode == "pinch_bridge":
        # Mode A: tripod_pinch virtual pinch-center bridge.
        # Move hover -> pre -> grip with arm.REACH = bridge.pinch_center_hand so fingertips track a clean
        # straight line over tall bottles/caps and straight down onto slim objects, then switch arm.REACH = 0
        # for closed-loop wrist settle (_settle_arm) so lever-arm Jacobian coupling never stalls convergence.
        bridge = build_pinch_bridge_spec(chosen)
        arm.REACH = bridge.pinch_center_hand
        hover_w, pre_w, grip_w, _ = compute_dex_pre_grasp_waypoints(
            chosen, table_z=table_z, slide_dist=0.045, hover_dist=0.075, lift_dist=PINCH_ARM_LIFT
        )
        off_w = chosen.wrist_rotation.apply(bridge.pinch_center_hand)
        hover_p, pre_p, grip_p = hover_w + off_w, pre_w + off_w, grip_w + off_w

        # 1. Hover above pinch point while adapting waist/lift and opening to preshape
        yield from sess.move_arm(
            idx,
            hover_p,
            chosen.wrist_rotation,
            hand_q=bridge.preshape_qpos,
            waist=posture.waist_grasp,
            lift=posture.lift_grasp,
            steps=45,
        )
        # 2. Descend to pre-grasp and pinch point, then settle closed-loop IK directly on wrist pose
        yield from sess.move_arm(idx, pre_p, chosen.wrist_rotation, hand_q=bridge.preshape_qpos, steps=22)
        yield from sess.move_arm(idx, grip_p, chosen.wrist_rotation, hand_q=bridge.preshape_qpos, steps=26)
        arm.REACH = np.zeros(3, dtype=np.float64)
        yield from _settle_arm(grip_w, chosen.wrist_rotation, bridge.preshape_qpos, max_steps=35)
        grip_err = float(np.linalg.norm(arm.grasp_point() - grip_w))
        info["grip_err_m"] = round(grip_err, 4)
        print(f"[sharpa_aff] pinch grip_err={grip_err:.4f} m lift={posture.lift_grasp[0]:.3f}", flush=True)

        # 3. Close pinch (simultaneous thumb + index + middle)
        if use_tactile_gate:
            gate = yield from sess.close_with_tactile_gate(
                idx,
                "tripod_pinch",
                bridge.preshape_qpos,
                bridge.preshape_qpos,
                bridge.target_qpos,
                thumb_steps=1,
                finger_steps=28,
            )
            info["tactile_trigger"] = gate.trigger_reason
            info["peak_force_n"] = round(float(gate.peak_force_n), 3)
        else:
            # Close smoothly to target_qpos + width-adapted compliant flexion preload (contract-clipped:
            # +0.11 rad for slim < 3.5 cm items like vials/pens, +0.08 rad for wider objects).
            kin = dex_aff.get_sharpa_kinematics()
            preload = 0.11 if chosen.width < 0.035 else 0.08
            q_firm = bridge.target_qpos.copy()
            for j_idx in (0, 2, 4, 5, 7, 8, 9, 11, 12):
                q_firm[j_idx] += preload
            q_firm = kin.clip_qpos(q_firm, side=chosen.hand_side, enforce_mcp_aa=True)
            peak_f = 0.0
            for h_t in pbs.interp_hand(bridge.preshape_qpos, q_firm, 30):
                yield from sess.emit(
                    h_map={arm.grip_key: h_t},
                    loaded={arm.arm_key: True},
                )
                peak_f = _track_peak(peak_f)
            info["tactile_trigger"] = "full_target_preload"
            info["peak_force_n"] = round(peak_f, 3)

        # 4. Settle and lift vertically via lift column (zero arm-IK wrist slip)
        yield from sess.hold_steps(8, loaded={arm.arm_key: True})
        yield from sess.hold_steps(
            36,
            waist=posture.waist_lift,
            lift=posture.lift_up,
            loaded={arm.arm_key: True},
        )
        yield from sess.hold_steps(35, loaded={arm.arm_key: True})
        return

    # Mode B: power_cylindrical / power_spherical / handle_hook staged 22-DoF trajectory
    hover_w, pre_w, grip_w, _ = compute_dex_pre_grasp_waypoints(
        chosen, table_z=table_z, slide_dist=0.055, hover_dist=0.11, lift_dist=LIFT_COLUMN_RAISE
    )
    q_pre = chosen.preshape_qpos
    q_tgt = chosen.target_qpos
    k_thumb = int(chosen.thumb_first_joints)
    q_thumb = q_pre.copy()
    if k_thumb > 0:
        q_thumb[:k_thumb] = q_tgt[:k_thumb]

    # 1. Hover above pre-grasp while adapting waist + lift + preshape
    yield from sess.move_arm(
        idx,
        hover_w,
        chosen.wrist_rotation,
        hand_q=q_pre,
        waist=posture.waist_grasp,
        lift=posture.lift_grasp,
        steps=45,
    )
    # 2. Lower to pre-grasp
    yield from sess.move_arm(idx, pre_w, chosen.wrist_rotation, hand_q=q_pre, steps=24)
    # 3. Slide into wrist grasp pose and settle closed-loop IK before closing
    yield from sess.move_arm(idx, grip_w, chosen.wrist_rotation, hand_q=q_pre, steps=26)
    yield from _settle_arm(grip_w, chosen.wrist_rotation, q_pre, max_steps=30)
    grip_err = float(np.linalg.norm(arm.grasp_point() - grip_w))
    info["grip_err_m"] = round(grip_err, 4)
    print(f"[sharpa_aff] power grip_err={grip_err:.4f} m lift={posture.lift_grasp[0]:.3f}", flush=True)

    # 4. Staged thumb-first -> finger wrap closure
    gate_prim = "power_wrap"
    if use_tactile_gate:
        thumb_steps = 14 if k_thumb > 0 else 1
        gate = yield from sess.close_with_tactile_gate(
            idx,
            gate_prim,
            q_pre,
            q_thumb,
            q_tgt,
            thumb_steps=thumb_steps,
            finger_steps=28,
        )
        info["tactile_trigger"] = gate.trigger_reason
        info["peak_force_n"] = round(float(gate.peak_force_n), 3)
    else:
        kin = dex_aff.get_sharpa_kinematics()
        q_firm = q_tgt.copy()
        for j_idx in (0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18, 20, 21):
            q_firm[j_idx] += 0.06
        q_firm = kin.clip_qpos(q_firm, side=chosen.hand_side, enforce_mcp_aa=True)
        q_thumb_firm = q_pre.copy()
        if k_thumb > 0:
            q_thumb_firm[:k_thumb] = q_firm[:k_thumb]
        peak_f = 0.0
        if k_thumb > 0:
            for h_t in pbs.interp_hand(q_pre, q_thumb_firm, 14):
                yield from sess.emit(h_map={arm.grip_key: h_t}, loaded={arm.arm_key: True})
                peak_f = _track_peak(peak_f)
        for h_t in pbs.interp_hand(q_thumb_firm, q_firm, 28):
            yield from sess.emit(h_map={arm.grip_key: h_t}, loaded={arm.arm_key: True})
            peak_f = _track_peak(peak_f)
        info["tactile_trigger"] = "full_target_preload"
        info["peak_force_n"] = round(peak_f, 3)

    # 5. Short settle hold
    yield from sess.hold_steps(8, loaded={arm.arm_key: True})
    # 6. Lift-column raise (holds arm joints steady to prevent IK slip)
    yield from sess.hold_steps(
        36,
        waist=posture.waist_lift,
        lift=posture.lift_up,
        loaded={arm.arm_key: True},
    )
    # 7. Hold lifted cleanly for acceptance window
    yield from sess.hold_steps(35, loaded={arm.arm_key: True})
