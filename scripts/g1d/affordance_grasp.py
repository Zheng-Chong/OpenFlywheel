"""Affordance-aware grasp planning and GraspRobot integration for OpenFlywheel.

Consumes offline-annotated `affordance.json` sidecars (produced by
`scripts/objects/affordance.py` and optionally physics-verified by
`scripts/objects/verify_affordance_sim.py`) during scripted expert planning:

  1. Transforms local-frame grasps to world frame with DR scale adaptation.
  2. Expands Z-axis rotational symmetry and 180-deg parallel-jaw wrist flip.
  3. Filters by task part_tags / exclude_tags, tabletop clearance, approach angle,
     nearby scene obstacles, and optional joint grasp+downstream IK callbacks.
  4. Ranks candidates by composite score (with optional Top-K softmax sampling for
     trajectory diversity) and integrates into `GraspRobot41` with automatic
     fallback to the existing bounding-box heuristic when no affordance applies.

`scripts/g1d/scripted_demo.py` is left untouched so registered expert sources
remain immutable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence as AbcSequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.objects.affordance import GRIPPER_SPECS, GripperSpec

IKFilterFn = Callable[[np.ndarray, Rotation, float], tuple[bool, float]]


@dataclass(frozen=True)
class PlannedGrasp:
    """A single world-frame grasp candidate emitted by AffordancePlanner."""

    grasp_id: str
    position: np.ndarray  # (3,) world grasp_point (between pads)
    rotation: Rotation  # world EE rotation: +X=approach, +Y=closing, +Z=dorsal
    approach_dir: np.ndarray  # (3,) unit vector in world
    closing_axis: np.ndarray  # (3,) unit vector in world
    width: float  # scaled object grasp width (m)
    part_tag: str
    score: float  # composite planning score
    base_score: float  # raw affordance score (sim_score or geom score)
    verified: bool
    name: str  # human-readable candidate tag for logs/sidecars


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


def _scale_local_grasp(
    pos_local: np.ndarray,
    rot_local: np.ndarray,
    width: float,
    scale_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Apply (possibly non-uniform) DR scaling to a local grasp pose and width."""
    p_scaled = pos_local * scale_vec
    approach_raw = rot_local[:, 0] * scale_vec
    closing_raw = (width * rot_local[:, 1]) * scale_vec
    new_width = float(np.linalg.norm(closing_raw))
    if new_width < 1e-6:
        return None
    closing_u = closing_raw / new_width
    # Re-orthogonalize approach against scaled closing axis
    approach_orth = approach_raw - float(np.dot(approach_raw, closing_u)) * closing_u
    a_norm = float(np.linalg.norm(approach_orth))
    if a_norm < 1e-6:
        return None
    approach_u = approach_orth / a_norm
    dorsal_u = np.cross(approach_u, closing_u)
    d_norm = float(np.linalg.norm(dorsal_u))
    if d_norm < 1e-6:
        return None
    dorsal_u /= d_norm
    rot_scaled = np.column_stack([approach_u, closing_u, dorsal_u])
    return p_scaled, rot_scaled, new_width


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


def _collides_obstacles(
    p_world: np.ndarray,
    p_tip: np.ndarray,
    p_palm: np.ndarray,
    obstacles: Sequence[dict[str, Any]],
    margin: float = 0.015,
    extra_pts: Sequence[np.ndarray] = (),
) -> bool:
    """Check whether key gripper & approach-corridor points collide with obstacle spheres or AABBs."""
    test_pts = (p_world, p_tip, p_palm, *extra_pts)
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


class AffordancePlanner:
    """Pure-NumPy/SciPy grasp planner consuming asset affordance.json metadata."""

    def __init__(
        self,
        *,
        gripper_type: str = "dex1_1",
        continuous_sym_samples: int = 8,
        table_clear: float = 0.004,
        palm_clear: float = 0.005,
        max_approach_z: float = 0.15,
        obstacle_margin: float = 0.015,
    ):
        if gripper_type not in GRIPPER_SPECS:
            raise ValueError(f"Unknown gripper_type={gripper_type!r}")
        self.gripper_type = gripper_type
        self.spec: GripperSpec = GRIPPER_SPECS[gripper_type]
        self.continuous_sym_samples = int(continuous_sym_samples)
        self.table_clear = float(table_clear)
        self.palm_clear = float(palm_clear)
        self.max_approach_z = float(max_approach_z)
        self.obstacle_margin = float(obstacle_margin)

    def plan_grasps(
        self,
        affordance: dict[str, Any] | str | Path,
        obj_pos: Sequence[float],
        obj_quat_wxyz: Sequence[float],
        *,
        scale: float | Sequence[float] = 1.0,
        table_z: float | None = None,
        part_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        prefer_tags: Sequence[str] | Mapping[str, float] | None = None,
        verified_only: bool = False,
        skip_sim_failed: bool = True,
        prefer_yaw: float | None = None,
        obstacles: Sequence[dict[str, Any]] | None = None,
        ik_filter: IKFilterFn | None = None,
        top_k: int = 16,
        sample_temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[PlannedGrasp]:
        """Transform, filter, and rank local affordance grasps into world-frame candidates."""
        data = load_affordance(affordance)
        raw_grasps = (data.get("grasps") or {}).get(self.gripper_type) or []
        if not raw_grasps:
            return []

        scale_ref = float(data.get("scale_ref", 1.0))
        scale_vec = _normalize_scale(scale, scale_ref)
        sym_meta = data.get("symmetry") or {}
        sym_angles = _symmetry_angles(sym_meta, self.continuous_sym_samples)
        raw_axis = sym_meta.get("axis_xy")
        if raw_axis is not None and len(raw_axis) >= 2:
            sym_axis = np.array([float(raw_axis[0]), float(raw_axis[1]), 0.0], dtype=np.float64)
        else:
            sym_axis = np.zeros(3, dtype=np.float64)

        p_obj = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        r_obj = from_wxyz(obj_quat_wxyz).as_matrix()

        allow_set = set(part_tags) if part_tags is not None else None
        deny_set = set(exclude_tags) if exclude_tags is not None else set()
        prefer_bonuses: dict[str, float] = {}
        if isinstance(prefer_tags, Mapping):
            prefer_bonuses = {str(k): float(v) for k, v in prefer_tags.items()}
        elif prefer_tags is not None:
            pref_list = [str(t) for t in prefer_tags]
            n_pref = len(pref_list)
            for rank, t_name in enumerate(pref_list):
                if t_name not in prefer_bonuses:
                    # Ranked preference bonus: +0.35 for top-ranked tag down to +0.05 for last
                    prefer_bonuses[t_name] = 0.35 if n_pref <= 1 else (0.35 - 0.30 * (rank / (n_pref - 1)))
        obs_list = list(obstacles) if obstacles else []

        bounds = data.get("bounds") or {}
        b_min_raw = bounds.get("min")
        b_max_raw = bounds.get("max")
        if b_min_raw is not None and b_max_raw is not None and len(b_min_raw) >= 3 and len(b_max_raw) >= 3:
            obj_z_min = float(b_min_raw[2]) * float(scale_vec[2])
            obj_z_max = float(b_max_raw[2]) * float(scale_vec[2])
            obj_height = max(0.0, obj_z_max - obj_z_min)
        else:
            obj_z_min = obj_z_max = obj_height = None

        flip_mat = np.diag([1.0, -1.0, -1.0])  # 180-deg roll around +X_ee (parallel-jaw symmetry)
        candidates: list[PlannedGrasp] = []

        for g in raw_grasps:
            tag = str(g.get("part_tag", "unknown"))
            if allow_set is not None and tag not in allow_set:
                continue
            if tag in deny_set:
                continue
            is_verified = bool(g.get("verified", False))
            if verified_only and not is_verified:
                continue
            sim_score = g.get("sim_score")
            sim_tested = sim_score is not None or g.get("sim_metrics") is not None
            if skip_sim_failed and sim_tested and not is_verified:
                continue

            p_loc = np.asarray(g["position"], dtype=np.float64).reshape(3)
            r_loc = from_wxyz(g["quaternion_wxyz"]).as_matrix()
            w_raw = float(g["width"])

            base_score = float(sim_score) if sim_score is not None else float(g.get("score", 0.5))
            gid = str(g.get("id", "g"))

            for sym_idx, theta in enumerate(sym_angles):
                # Rotate in the unscaled local frame about the detected symmetry axis FIRST,
                # then apply (possibly non-uniform) DR scaling so broken symmetries are re-validated.
                if abs(theta) > 1e-9:
                    c, s = math.cos(theta), math.sin(theta)
                    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                    p_sym = sym_axis + rz @ (p_loc - sym_axis)
                    r_sym = rz @ r_loc
                else:
                    p_sym = p_loc
                    r_sym = r_loc

                scaled = _scale_local_grasp(p_sym, r_sym, w_raw, scale_vec)
                if scaled is None:
                    continue
                p_sc, r_sc, width = scaled
                if width < self.spec.min_width or width > self.spec.max_width:
                    continue

                p_world = p_obj + r_obj @ p_sc
                r_world_base = r_obj @ r_sc
                approach_w = r_world_base[:, 0]

                # Reject upward approaches (approaching from below the table)
                if float(approach_w[2]) > self.max_approach_z:
                    continue

                open_half = min(0.5 * (width + self.spec.grasp_clear), 0.5 * self.spec.max_open)
                finger_half_y = open_half + self.spec.finger_thick

                for flip_idx, r_world in enumerate((r_world_base, r_world_base @ flip_mat)):
                    closing_w = r_world[:, 1]
                    dorsal_w = r_world[:, 2]

                    p_tip = p_world + self.spec.finger_tips * approach_w
                    p_palm = p_world - (self.spec.finger_back + 0.5 * self.spec.palm_len) * approach_w

                    if table_z is not None:
                        tip_low_z = (
                            float(p_tip[2])
                            - finger_half_y * abs(float(closing_w[2]))
                            - 0.5 * self.spec.finger_depth * abs(float(dorsal_w[2]))
                        )
                        if tip_low_z < table_z + self.table_clear:
                            continue
                        palm_half_z = 0.5 * (
                            self.spec.palm_len * abs(float(approach_w[2]))
                            + self.spec.palm_width * abs(float(closing_w[2]))
                            + self.spec.palm_height * abs(float(dorsal_w[2]))
                        )
                        palm_low_z = float(p_palm[2]) - palm_half_z
                        if palm_low_z < table_z + self.palm_clear:
                            continue

                    if obs_list:
                        fy_off = finger_half_y * closing_w
                        corridor_pts: list[np.ndarray] = [
                            p_world + fy_off,
                            p_world - fy_off,
                            p_tip + fy_off,
                            p_tip - fy_off,
                        ]
                        if float(approach_w[2]) <= SIDE_APPROACH_Z_THRESH:
                            for dz in (0.05, 0.10):
                                p_up = p_world + np.array([0.0, 0.0, dz], dtype=np.float64)
                                corridor_pts.extend((p_up, p_up + fy_off, p_up - fy_off))
                        else:
                            p_pre = p_world - SIDE_APPROACH_BACK * approach_w
                            corridor_pts.extend((p_pre, p_pre + fy_off, p_pre - fy_off))
                        if _collides_obstacles(
                            p_world, p_tip, p_palm, obs_list, margin=self.obstacle_margin, extra_pts=corridor_pts
                        ):
                            continue

                    hand_rot = Rotation.from_matrix(r_world)
                    ik_penalty = 0.0
                    if ik_filter is not None:
                        ok, pen = ik_filter(p_world, hand_rot, width)
                        if not ok:
                            continue
                        ik_penalty = float(pen)

                    # Composite score: affordance quality + verified bonus + wrist-down penalty + top-down + yaw
                    score = base_score + (0.10 if is_verified else 0.0)
                    # Only penalize wrist-down (dorsal_z < 0); never reward tilting away from vertical top-down
                    score += 0.08 * float(np.clip(dorsal_w[2], -1.0, 0.0))
                    # Strongly prefer top-down over horizontal side when both are valid (exceeds prefer_yaw=0.10)
                    score += 0.12 * float(np.clip(-approach_w[2], 0.0, 1.0))
                    # For top-down grasps on upright objects, ensure palm/wrist-camera shell clears the object top
                    # (real Dex1-1 wrist-camera housing sits 44 mm behind grasp_point, cf. scripted_demo.PLUS_PALM)
                    if (
                        obj_z_max is not None
                        and obj_z_min is not None
                        and obj_height is not None
                        and float(approach_w[2]) <= -0.70
                        and float(r_obj[2, 2]) > 0.70
                    ):
                        top_above_grip = obj_z_max - float(p_sc[2])
                        palm_top_clear = self.spec.finger_back - top_above_grip
                        palm_min_clear = (
                            0.006 if (obj_height > 0.12 and float(approach_w[2]) <= SIDE_APPROACH_Z_THRESH) else 0.0
                        )
                        if palm_top_clear < palm_min_clear:
                            # Hard-reject top-down grasps whose palm penetrates (or grazes <6 mm on tall >12 cm objects)
                            continue
                        if palm_top_clear < 0.012:
                            # Strong soft penalty so safe upper grasps win over tight-clearance deep grasps
                            score -= 0.50 * (0.012 - palm_top_clear) / 0.012
                        if obj_height > 0.08:
                            z_frac = float(np.clip((float(p_sc[2]) - obj_z_min) / max(obj_height, 1e-6), 0.0, 1.0))
                            score += 0.04 * z_frac
                    if prefer_yaw is not None:
                        # Compare closing axis yaw (mod pi) or approach horizontal heading with prefer_yaw
                        horiz = approach_w[:2]
                        h_norm = float(np.linalg.norm(horiz))
                        if h_norm > 0.15:
                            yaw_a = math.atan2(float(horiz[1]), float(horiz[0]))
                            # Hand forward at yaw=0 is +Y (GRASP_ROT), so preferred approach yaw is prefer_yaw + pi/2
                            score += 0.10 * math.cos(yaw_a - (prefer_yaw + math.pi / 2.0))
                        else:
                            # Near-vertical top-down: align closing axis with prefer_yaw
                            yaw_c = math.atan2(float(closing_w[1]), float(closing_w[0]))
                            score += 0.10 * abs(math.cos(yaw_c - prefer_yaw))
                    score -= ik_penalty
                    if prefer_bonuses:
                        score += prefer_bonuses.get(tag, 0.0)

                    suffix = ""
                    if sym_idx:
                        suffix += f"_s{sym_idx}"
                    if flip_idx:
                        suffix += "_f"
                    candidates.append(
                        PlannedGrasp(
                            grasp_id=f"{gid}{suffix}",
                            position=p_world,
                            rotation=hand_rot,
                            approach_dir=approach_w.copy(),
                            closing_axis=closing_w.copy(),
                            width=width,
                            part_tag=tag,
                            score=round(float(score), 5),
                            base_score=round(float(base_score), 5),
                            verified=is_verified,
                            name=f"aff:{tag}:{gid}{suffix}",
                        )
                    )

        if not candidates:
            return []

        # Deduplicate near-identical world poses (can happen after symmetry expansion)
        candidates.sort(key=lambda c: c.score, reverse=True)
        deduped: list[PlannedGrasp] = []
        cos_ang = math.cos(math.radians(15.0))
        for cand in candidates:
            if len(deduped) >= max(top_k * 2, 32):
                break
            dup = False
            for k in deduped:
                if np.linalg.norm(cand.position - k.position) < 0.01:
                    if (
                        float(np.dot(cand.approach_dir, k.approach_dir)) >= cos_ang
                        and float(np.dot(cand.closing_axis, k.closing_axis)) >= cos_ang
                    ):
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


# ---------------------------------------------------------------------------
# Task-specific semantic part_tag policies (pouring, placing, putting, picking)
# ---------------------------------------------------------------------------

TASK_PART_POLICIES: dict[str, dict[str, Any]] = {
    "pick": {
        "part_tags": None,
        "exclude_tags": None,
        "prefer_tags": None,
        "max_approach_z": -0.35,
    },
    "place": {
        "part_tags": None,
        "exclude_tags": ("base", "blade", "spout"),
        "prefer_tags": ("body_top", "body", "handle", "neck"),
        "max_approach_z": -0.35,
    },
    "put": {
        "part_tags": None,
        "exclude_tags": ("base", "blade", "spout"),
        "prefer_tags": ("body_top", "neck", "handle", "body"),
        "max_approach_z": -0.35,
    },
    "pour": {
        "part_tags": None,
        "exclude_tags": ("rim", "spout", "lid", "cap", "base"),
        "prefer_tags": ("handle", "body_side", "body", "neck"),
        "max_approach_z": 0.15,  # allow horizontal side grasps on cup/pitcher handles & bodies
    },
}


def make_task_planner(task_name: str, *, gripper_type: str = "dex1_1", **kwargs: Any) -> AffordancePlanner:
    """Instantiate an `AffordancePlanner` configured with task-appropriate approach limits."""
    policy = TASK_PART_POLICIES.get(task_name)
    if policy is None:
        raise ValueError(f"Unknown task_name={task_name!r}; expected one of {sorted(TASK_PART_POLICIES)}")
    params: dict[str, Any] = {"gripper_type": gripper_type, "max_approach_z": policy["max_approach_z"]}
    params.update(kwargs)
    return AffordancePlanner(**params)


def select_pour_grasp(
    affordance: dict[str, Any] | str | Path,
    obj_pos: Sequence[float],
    obj_quat_wxyz: Sequence[float],
    *,
    scale: float | Sequence[float] = 1.0,
    table_z: float | None = None,
    prefer_tags: Sequence[str] | Mapping[str, float] | None = None,
    exclude_tags: Sequence[str] | None = None,
    verified_only: bool = False,
    prefer_yaw: float | None = None,
    obstacles: Sequence[dict[str, Any]] | None = None,
    ik_filter: IKFilterFn | None = None,
    gripper_type: str = "dex1_1",
    top_k: int = 8,
) -> list[PlannedGrasp]:
    """Select pouring-safe grasps from `affordance.json` (handle preferred, rim/spout/lid excluded)."""
    policy = TASK_PART_POLICIES["pour"]
    planner = make_task_planner("pour", gripper_type=gripper_type)
    return planner.plan_grasps(
        affordance,
        obj_pos,
        obj_quat_wxyz,
        scale=scale,
        table_z=table_z,
        exclude_tags=policy["exclude_tags"] if exclude_tags is None else exclude_tags,
        prefer_tags=policy["prefer_tags"] if prefer_tags is None else prefer_tags,
        verified_only=verified_only,
        prefer_yaw=prefer_yaw,
        obstacles=obstacles,
        ik_filter=ik_filter,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# GraspRobot41 integration (lazy import of scripted_demo to keep planner standalone)
# ---------------------------------------------------------------------------

SIDE_APPROACH_Z_THRESH = -0.80  # approach_dir[2] > this (incl. 45-deg -0.707) => approach-axis pre-grasp
SIDE_APPROACH_BACK = 0.07  # m retract along -approach_dir before sliding in to grasp_point
STEP_IK_MID_ERR_LIMIT = 0.10  # m peak tracking error allowed during descent (hover/pre-grasp->grip) or grip->lift


def _is_affordance_name(name: str) -> bool:
    return name.startswith("aff:") or name.startswith("other hand, aff:")


def _rollout_step_ik(
    reach: Any,
    q0: np.ndarray,
    segments: Sequence[tuple[np.ndarray, Rotation, int, bool]],
    waypoints_fn: Callable[..., Any],
    *,
    damping: float = 0.02,
    settle_steps: int = 25,
    max_mid_err: float = STEP_IK_MID_ERR_LIMIT,
    dq_max: float | None = None,
) -> tuple[float, float]:
    """Simulate Arm.ik incremental 1-step DLS (+ optional Limiter dq_max) along interpolated segments on the URDF.

    Each segment is `(target_pos, target_rot, n_steps, check_mid_spike)`. Returns `(worst_pos_m, worst_rot_deg)`
    so `choose()` rejects candidates whose path stalls or whips at joint/velocity limits even if 200-iter
    `reach.solve()` finds a disconnected branch.
    """
    lim = np.asarray(reach.lim, dtype=np.float64)
    base = np.asarray(reach.base, dtype=np.float64)
    base_r = base[:3, :3]
    base_t = base[:3, 3]
    reach_vec = np.asarray(reach.reach, dtype=np.float64)
    q = np.clip(np.asarray(q0, dtype=np.float64).copy(), lim[:, 0], lim[:, 1])
    n_dof = len(q)
    damp_eye = (damping**2) * np.eye(6)

    def _fk_state(qv: np.ndarray, want_jac: bool = False):
        axes: list[tuple[np.ndarray, np.ndarray]] | None = [] if want_jac else None
        t = base @ reach.fk(qv, axes)
        pos = t[:3, 3]
        rot = Rotation.from_matrix(t[:3, :3])
        grip = pos + rot.apply(reach_vec)
        if not want_jac:
            return grip, rot, None
        jac = np.zeros((6, n_dof), dtype=np.float64)
        for i, (o, a) in enumerate(axes or ()):
            ow = base_r @ o + base_t
            aw = base_r @ a
            jac[:3, i] = np.cross(aw, grip - ow)
            jac[3:, i] = 0.3 * aw
        return grip, rot, jac

    def _ik_step(qv: np.ndarray, p_des: np.ndarray, r_des: Rotation) -> tuple[np.ndarray, float, float]:
        grip, rot, jac = _fk_state(qv, want_jac=True)
        dp = p_des - grip
        rv = (r_des * rot.inv()).as_rotvec()
        pos_e = float(np.linalg.norm(dp))
        ang_e = float(np.degrees(np.linalg.norm(rv)))
        err = np.r_[dp, 0.3 * rv]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damp_eye, err)
        saturated = ((qv + dq <= lim[:, 0]) & (dq < 0.0)) | ((qv + dq >= lim[:, 1]) & (dq > 0.0))
        if np.any(saturated):
            jac[:, saturated] = 0.0
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damp_eye, err)
        q_next = np.clip(qv + dq, lim[:, 0], lim[:, 1])
        if dq_max is not None:
            q_next = qv + np.clip(q_next - qv, -dq_max, dq_max)
        return q_next, pos_e, ang_e

    worst_pos = 0.0
    worst_rot = 0.0
    for pt_t, rot_t, n_steps, check_mid in segments:
        p0, r0, _ = _fk_state(q, want_jac=False)
        peak_mid = 0.0
        for p_w, r_w, _ in waypoints_fn(p0, r0, [(pt_t, rot_t, 0.0, n_steps)]):
            q, pos_e, _ = _ik_step(q, p_w, r_w)
            if check_mid and pos_e > peak_mid:
                peak_mid = pos_e
        for _ in range(settle_steps):
            q, pos_e, ang_e = _ik_step(q, pt_t, rot_t)
            if pos_e < 0.004 and ang_e < 2.0:
                break
        grip_end, rot_end, _ = _fk_state(q, want_jac=False)
        end_pos = float(np.linalg.norm(pt_t - grip_end))
        end_rot = float(np.degrees(np.linalg.norm((rot_t * rot_end.inv()).as_rotvec())))
        worst_pos = max(worst_pos, end_pos)
        worst_rot = max(worst_rot, end_rot)
        if check_mid and peak_mid > max_mid_err:
            worst_pos = max(worst_pos, peak_mid)
    return worst_pos, worst_rot


def compute_pre_grasp_waypoints(
    grip: np.ndarray,
    hand: Rotation,
    lift: float,
    table_z: float | None = None,
    approach_back: float = SIDE_APPROACH_BACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return (hover, pre_grasp, lift_pos, is_side) for a grasp candidate.

    Top-down grasps (-approach_z >= 0.80) descend vertically from `hover = grip + [0, 0, lift]`.
    Side/angled grasps retract along `-approach_dir` by `approach_back` (clamped above `table_z`),
    with a high transit `hover` above `pre_grasp` so the open fingers never sweep into the object.
    """
    grip_arr = np.asarray(grip, dtype=np.float64).reshape(3)
    approach = hand.as_matrix()[:, 0]
    lift_pos = grip_arr + np.array([0.0, 0.0, float(lift)], dtype=np.float64)
    if float(approach[2]) <= SIDE_APPROACH_Z_THRESH:
        return lift_pos, lift_pos, lift_pos, False

    pre_grasp = grip_arr - float(approach_back) * approach
    if table_z is not None:
        pre_grasp[2] = max(float(pre_grasp[2]), float(table_z) + 0.03)
    hover = np.array(
        [pre_grasp[0], pre_grasp[1], max(float(pre_grasp[2]) + 0.06, float(lift_pos[2]))],
        dtype=np.float64,
    )
    return hover, pre_grasp, lift_pos, True


def make_affordance_grasp_robot_class():
    """Build AffordanceGraspRobot(GraspRobot41) lazily to avoid circular imports."""
    from scripts.g1d import scripted_demo as demo
    from scripts.g1d.grasp_v41 import GraspRobot41

    class AffordanceGraspRobot(GraspRobot41):
        """GraspRobot41 that plans from object affordance.json first and falls back to BBox heuristics."""

        affordance_part_tags: Sequence[str] | None = None
        affordance_exclude_tags: Sequence[str] | None = None
        affordance_prefer_tags: Sequence[str] | Mapping[str, float] | None = None
        affordance_max_approach_z: float | None = -0.35
        affordance_verified_only: bool = False
        affordance_top_k: int = 8
        affordance_temperature: float = 0.0

        def __init__(
            self,
            env,
            *,
            affordance_map: dict[str, dict[str, Any] | str | Path] | None = None,
            affordance_dir: str | Path | None = None,
            assets_root: str | Path | None = None,
            planner: AffordancePlanner | None = None,
            label_policies: Mapping[str, Mapping[str, Any]] | None = None,
        ):
            super().__init__(env)
            self.affordance_map: dict[str, dict[str, Any] | str | Path] = dict(affordance_map or {})
            self.affordance_label_policies: dict[str, dict[str, Any]] = {
                str(k): dict(v) for k, v in (label_policies or {}).items()
            }
            env_aff_dir = os.environ.get("OPENFLYWHEEL_AFFORDANCE_DIR", "").strip()
            resolved_aff_dir = affordance_dir if affordance_dir is not None else (env_aff_dir or None)
            self.affordance_dir = Path(resolved_aff_dir) if resolved_aff_dir is not None else None
            self.assets_root = Path(assets_root) if assets_root is not None else None
            # Tabletop pick_anything: require >= 20 deg downward approach (excludes pure horizontal body_side
            # grasps that hit G1-D horizontal-wrist lift singularities on tabletop, while keeping 45-90 deg grasps).
            self.planner = planner or AffordancePlanner(
                gripper_type="dex1_1", max_approach_z=self.affordance_max_approach_z
            )
            self._last_table_z: float | None = None

        def _resolve_affordance(self, label: str) -> dict[str, Any] | None:
            if label in self.affordance_map:
                return load_affordance(self.affordance_map[label])
            try:
                entry = demo.layout_entry(self.env, label)
            except Exception:
                entry = None
            if isinstance(entry, Mapping):
                if isinstance(entry.get("affordance"), Mapping):
                    return dict(entry["affordance"])
                if entry.get("affordance_path") and Path(entry["affordance_path"]).is_file():
                    return load_affordance(entry["affordance_path"])
                cat = entry.get("category")
                idx = entry.get("category_idx")
                aff_dir = getattr(self, "affordance_dir", None)
                if cat is not None and idx is not None and aff_dir is not None:
                    override = Path(aff_dir) / str(cat) / f"{int(idx):05d}" / "affordance.json"
                    if override.is_file():
                        return load_affordance(override)
                usd_path = entry.get("usd_path") or entry.get("path")
                if usd_path:
                    sidecar = Path(usd_path).parent / "affordance.json"
                    if sidecar.is_file():
                        return load_affordance(sidecar)
                    if self.assets_root is not None:
                        sidecar2 = self.assets_root / Path(usd_path).parent / "affordance.json"
                        if sidecar2.is_file():
                            return load_affordance(sidecar2)
                # Standard OpenFlywheel layout entry: category + category_idx
                if cat is not None and idx is not None:
                    from env.global_configs import ASSETS_PATH, object_dir

                    assets = str(self.assets_root) if self.assets_root is not None else ASSETS_PATH
                    obj_dir = Path(object_dir(f"Rigid/{cat}/{int(idx):05d}", assets=assets))
                    sidecar = obj_dir / "affordance.json"
                    if sidecar.is_file():
                        return load_affordance(sidecar)
            return None

        def _object_scale(self, label: str) -> float | list[float]:
            try:
                entry = demo.layout_entry(self.env, label)
            except Exception:
                return 1.0
            if isinstance(entry, Mapping) and "scale" in entry and entry["scale"] is not None:
                sc = entry["scale"]
                return (
                    [float(x) for x in sc]
                    if isinstance(sc, AbcSequence) and not isinstance(sc, str | bytes)
                    else float(sc)
                )
            return 1.0

        def _iter_rigid_entries(self) -> list[Mapping[str, Any]]:
            # Canonical OpenFlywheel scene layout: env.scene_manager.config["env0"]["Rigid"] = {cat: [entry, ...]}
            # Note: scene_manager.config uses omegaconf DictConfig/ListConfig (Mapping/Sequence, NOT dict/list).
            sm = getattr(self.env, "scene_manager", None)
            cfg = getattr(sm, "config", None)
            if isinstance(cfg, Mapping):
                env0 = cfg.get("env0")
                if isinstance(env0, Mapping) and isinstance(env0.get("Rigid"), Mapping):
                    return [
                        e
                        for es in env0["Rigid"].values()
                        if isinstance(es, AbcSequence) and not isinstance(es, str | bytes)
                        for e in es
                        if isinstance(e, Mapping)
                    ]
            # Fallback for lightweight test mocks
            layout = getattr(getattr(self.env, "cfg", None), "layout", None)
            if isinstance(layout, Mapping) and isinstance(layout.get("objects"), AbcSequence):
                return [e for e in layout["objects"] if isinstance(e, Mapping)]
            return []

        def _scene_obstacles(self, exclude_label: str) -> list[dict[str, Any]]:
            obstacles: list[dict[str, Any]] = []
            lm = getattr(self, "lm", None)
            if lm is None or not hasattr(lm, "get_instance_pose"):
                return obstacles
            for entry in self._iter_rigid_entries():
                lbl = entry.get("label")
                if not lbl or lbl == exclude_label:
                    continue
                try:
                    pos, _ = lm.get_instance_pose(0, label=lbl)
                    p = demo.as_np(pos)
                    sz = (entry.get("physics") or {}).get("size")
                    sc = entry.get("scale")
                    if sc is not None:
                        sc_xy = (
                            max(float(sc[0]), float(sc[1]))
                            if isinstance(sc, AbcSequence) and not isinstance(sc, str | bytes) and len(sc) >= 2
                            else float(sc)
                        )
                    else:
                        sc_xy = 1.0
                    rad = 0.5 * float(math.hypot(sz[0], sz[1])) * sc_xy if sz and len(sz) >= 2 else 0.04
                    obstacles.append({"position": p.tolist(), "radius": rad})
                except Exception:
                    continue
            return obstacles

        def candidates(
            self,
            label,
            across,
            prefer,
            period,
            arm,
            *,
            part_tags: Sequence[str] | None = None,
            exclude_tags: Sequence[str] | None = None,
            prefer_tags: Sequence[str] | Mapping[str, float] | None = None,
        ):
            fallback = super().candidates(label, across, prefer, period, arm)
            aff = self._resolve_affordance(label)
            if aff is None:
                return fallback

            centre, _, _, bottom, _ = self.object(label)
            self._last_table_z = float(bottom)
            pos_raw, quat_raw = self.lm.get_instance_pose(0, label=label)
            obj_pos = demo.as_np(pos_raw)
            obj_quat = demo.as_np(quat_raw)

            first = arm or self.arm_for(centre[0])
            arms = [first] + [
                a for a in self.arms if a is not first and arm is None and abs(centre[0]) < demo.GRASP_MIDLINE
            ]
            scale = self._object_scale(label)
            obstacles = self._scene_obstacles(exclude_label=label)

            lbl_pol = (getattr(self, "affordance_label_policies", None) or {}).get(label, {})
            eff_part_tags = part_tags if part_tags is not None else lbl_pol.get("part_tags", self.affordance_part_tags)
            eff_exclude_tags = (
                exclude_tags if exclude_tags is not None else lbl_pol.get("exclude_tags", self.affordance_exclude_tags)
            )
            eff_prefer_tags = (
                prefer_tags if prefer_tags is not None else lbl_pol.get("prefer_tags", self.affordance_prefer_tags)
            )

            aff_candidates = []
            for a in arms:
                pref_yaw = prefer[a.arm_key] if (prefer is not None and a.arm_key in prefer) else None
                planned = self.planner.plan_grasps(
                    aff,
                    obj_pos,
                    obj_quat,
                    scale=scale,
                    table_z=float(bottom),
                    part_tags=eff_part_tags,
                    exclude_tags=eff_exclude_tags,
                    prefer_tags=eff_prefer_tags,
                    verified_only=self.affordance_verified_only,
                    prefer_yaw=pref_yaw,
                    obstacles=obstacles,
                    top_k=self.affordance_top_k,
                    sample_temperature=self.affordance_temperature,
                )
                own = "" if a is first else "other hand, "
                for pg in planned:
                    aff_candidates.append((a, pg.rotation, pg.position, pg.width, f"{own}{pg.name}"))

            print(
                f"[affordance_grasp] resolved {aff.get('asset_id')} for {label}: "
                f"{len(aff_candidates)} affordance cands, {len(fallback)} fallback cands"
            )
            return aff_candidates + fallback

        def miss(self, cand, lift):
            """Check incremental step-IK trackability across hover, [pre-grasp], grip, and lift."""
            a, hand, grip, _, name = cand
            if not _is_affordance_name(name):
                return super().miss(cand, lift)
            hover, pre_grasp, lift_pos, is_side = compute_pre_grasp_waypoints(
                grip, hand, lift, table_z=getattr(self, "_last_table_z", None)
            )
            reach = self.reach[a.arm_key]
            if hasattr(reach, "fk") and hasattr(reach, "base") and hasattr(reach, "lim") and hasattr(reach, "reach"):
                segments: list[tuple[np.ndarray, Rotation, int, bool]] = [(hover, hand, 54, False)]
                if is_side:
                    segments.append((pre_grasp, hand, max(16, self.DESCEND // 2), False))
                    segments.append((grip, hand, self.DESCEND, True))
                else:
                    segments.append((grip, hand, self.DESCEND, True))
                segments.append((lift_pos, hand, 30, True))
                return _rollout_step_ik(
                    reach,
                    a.q(),
                    segments,
                    demo.waypoints,
                    settle_steps=demo.PATTERN_SETTLE[1],
                    dq_max=demo.REAL_ARM_VMAX / 30.0,
                )
            # Fallback for lightweight mock Reach objects without fk/base/lim
            q, *hover_off = reach.solve(hover, hand, a.q())
            if is_side:
                q, *pre_off = reach.solve(pre_grasp, hand, q)
            else:
                pre_off = (0.0, 0.0)
            q, *grip_off = reach.solve(grip, hand, q)
            _, *lift_off = reach.solve(lift_pos, hand, q)
            return (
                max(hover_off[0], pre_off[0], grip_off[0], lift_off[0]),
                max(hover_off[1], pre_off[1], grip_off[1], lift_off[1]),
            )

        def choose(self, cands, lift):
            """Prefer the first URDF-reachable affordance candidate before falling back to v3."""
            misses = []
            for c in cands:
                if _is_affordance_name(c[4]):
                    m = self.miss(c, lift)
                    if m[0] <= demo.GRASP_TOL[0] and m[1] <= demo.GRASP_TOL[1]:
                        return c, m
                    misses.append((m, c))
            # No affordance candidate within tolerance: defer to GraspRobot41's default choice
            non_aff = [c for c in cands if not _is_affordance_name(c[4])]
            if non_aff:
                return super().choose(non_aff, lift)
            m, c = min(misses, key=lambda mc: mc[0][0])
            return c, m

        def pick(
            self,
            label,
            lift=demo.LIFT,
            close=False,
            across=None,
            prefer=None,
            period=np.pi,
            arm=None,
            *,
            part_tags: Sequence[str] | None = None,
            exclude_tags: Sequence[str] | None = None,
            prefer_tags: Sequence[str] | Mapping[str, float] | None = None,
        ):
            """Grasp `label` with approach-axis pre-grasp for side/angled affordance candidates."""
            tried = {}
            first = self.object(label)[0]
            for attempt in range(demo.GRASP_TRIES):
                centre, _, _, bottom, _ = self.object(label)
                if attempt and centre[2] < first[2] - demo.GRASP_GONE:
                    print(f"[affordance_grasp] {label} knocked off the desk ({centre.round(3)}): no more tries")
                    break
                cands = self.candidates(
                    label,
                    across,
                    prefer,
                    period,
                    arm,
                    part_tags=part_tags,
                    exclude_tags=exclude_tags,
                    prefer_tags=prefer_tags,
                )
                fresh = [c for c in cands if np.linalg.norm(tried.get(c[4], centre + 1.0) - centre) > demo.GRASP_MOVED]
                cand, urdf_miss = self.choose(fresh or cands[:1], lift)
                a, hand, grip, width, name = cand
                tried[name] = centre
                open_cmd = self.opening(width)
                is_aff = _is_affordance_name(name)
                if is_aff:
                    hover, pre_grasp, lift_pos, is_side = compute_pre_grasp_waypoints(
                        grip, hand, lift, table_z=float(bottom)
                    )
                else:
                    lift_pos = grip + np.array([0.0, 0.0, float(lift)], dtype=np.float64)
                    hover, pre_grasp, is_side = lift_pos, lift_pos, False

                entry = dict(label=label, attempt=attempt, grasp=name, arm=a.arm_key)
                if urdf_miss is not None:
                    entry["urdf_miss"] = [round(urdf_miss[0], 3), round(urdf_miss[1], 1)]
                self.log.append(entry)
                print(
                    f"[affordance_grasp] grasp try {attempt}: {label} by {a.arm_key} ({name}) at {grip.round(3)}"
                    f" width {width:.3f} open {open_cmd:.2f}, side={is_side}, URDF miss={entry.get('urdf_miss', '-')}"
                )
                last = attempt == demo.GRASP_TRIES - 1
                yield from self.move(a, [(hover, hand, open_cmd, 54 if attempt == 0 else 40)])
                if is_side:
                    yield from self.move(
                        a,
                        [
                            (pre_grasp, hand, open_cmd, max(16, self.DESCEND // 2)),
                            (grip, hand, open_cmd, self.DESCEND),
                        ],
                    )
                else:
                    slow, slow_n = demo.GRASP_SLOW
                    down = [(grip, hand, open_cmd, self.DESCEND)]
                    if attempt:
                        down = [
                            (grip + [0, 0, slow], hand, open_cmd, self.DESCEND - slow_n),
                            (grip, hand, open_cmd, slow_n),
                        ]
                    yield from self.move(a, down)

                point = a.grasp_point()
                if is_aff:
                    # Compare against planned grip (adjusted for any object displacement) so off-center grasps work
                    obj_shift = self.object(label)[0] - centre
                    target_pt = grip + obj_shift
                    aside = float(np.linalg.norm(point[:2] - target_pt[:2]))
                    above = float(point[2] - target_pt[2])
                else:
                    aside = float(np.linalg.norm(point[:2] - self.object(label)[0][:2]))
                    above = float(point[2] - grip[2])
                checked = attempt or self.first_try_grip_check
                if not last and checked and (aside > demo.GRIP_MISS[0] or above > demo.GRIP_MISS[1]):
                    entry["result"] = (
                        f"grip missed ({aside * 1000:.0f} mm beside the target, {above * 1000:.0f} mm high)"
                    )
                    retreat = (
                        [(pre_grasp, hand, open_cmd, 18), (hover, hand, open_cmd, 24)]
                        if is_side
                        else [(hover, hand, open_cmd, 30)]
                    )
                    yield from self.move(a, retreat)
                    continue

                yield from self.move(a, [(grip, hand, self.closing(width), 18)])
                jaw = demo.jaw_opening(self.env, a)
                if jaw < demo.GRASP_EMPTY:
                    entry["result"] = f"shut on nothing ({jaw:.2f})"
                    retreat = (
                        [(grip, hand, open_cmd, 10), (pre_grasp, hand, open_cmd, 16), (hover, hand, open_cmd, 24)]
                        if is_side
                        else [(grip, hand, open_cmd, 10), (hover, hand, open_cmd, 30)]
                    )
                    yield from self.move(a, retreat)
                    continue

                close_cmd = self.closing(width) if close else self.holding(a, width)
                before = self.object(label)[0]
                yield from self.move(a, [(lift_pos, hand, close_cmd, 30)])
                rose = self.object(label)[0][2] - before[2]
                if not last and rose < demo.TEST_RISE:
                    entry["result"] = f"did not rise ({rose * 1000:.0f} mm, jaw {jaw:.2f})"
                    low = np.r_[a.grasp_point()[:2], grip[2] + demo.TEST_LIFT]
                    yield from self.move(
                        a, [(low, hand, close_cmd, 20), (low, hand, open_cmd, 10), (hover, hand, open_cmd, 24)]
                    )
                    continue

                centre = self.object(label)[0]
                entry["result"] = "held" if rose >= demo.TEST_RISE else f"last try lifted anyway ({rose * 1000:.0f} mm)"
                print(f"[affordance_grasp] grasp try {attempt}: lifted {label} to {centre.round(3)}, jaw {jaw:.2f}")
                offset = a.pose()[1].inv().apply(centre - a.grasp_point())
                return a, hand, offset, grip[2] - bottom, open_cmd, close_cmd

            print(f"[affordance_grasp] {label}: no grasp held in {demo.GRASP_TRIES} tries")
            offset = a.pose()[1].inv().apply(self.object(label)[0] - a.grasp_point())
            return a, hand, offset, grip[2] - bottom, open_cmd, self.closing(width)

    return AffordanceGraspRobot


AffordanceGraspRobot = make_affordance_grasp_robot_class()


def pick_anything_affordance_actions(env):
    """g1d_pick_anything using AffordanceGraspRobot (affordance.json first, v4.1 BBox fallback)."""
    from scripts.g1d import scripted_demo as demo

    return demo.pick_anything_v4_actions(env, AffordanceGraspRobot)


def make_affordance_place_robot_class(task_name: str = "place"):
    """Build a PlaceRobot subclass that uses AffordanceGraspRobot with task `part_tag` policy + v4.2/v4.4 placing."""
    from scripts.g1d import place_v42, place_v44

    policy = TASK_PART_POLICIES[task_name]

    class AffordancePlaceRobot(place_v44.StallBreakMixin, AffordanceGraspRobot):
        """Combines affordance-based part_tag grasp selection with ik_within placing & stall-break settle."""

        placing = False
        affordance_part_tags = policy["part_tags"]
        affordance_exclude_tags = policy["exclude_tags"]
        affordance_prefer_tags = policy["prefer_tags"]
        affordance_max_approach_z = policy["max_approach_z"]

        def _command(self, arm, point, rot, grip):
            if not self.placing:
                return super()._command(arm, point, rot, grip)
            self.hold[arm.arm_key] = place_v42.ik_within(arm, point, rot)
            self.grip[arm.grip_key] = [float(grip)]
            self._advance(arm.arm_key)
            if self.step % 20 == 0:
                err = np.linalg.norm(arm.grasp_point() - point)
                print(f"[affordance_place] {self.step:3d} {arm.arm_key} err {err:.3f}")
            self.step += 1
            return dict(self.hold) | self.grip

    AffordancePlaceRobot.__name__ = f"Affordance{task_name.capitalize()}Robot"
    return AffordancePlaceRobot


def place_anything_affordance_actions(env):
    """g1d_place_anything using AffordancePlaceRobot (part_tag-aware affordance grasp + v4.4 placing)."""
    from scripts.g1d import place_v42

    return place_v42.place_anything_actions(env, make_affordance_place_robot_class("place"))


def put_anything_in_affordance_actions(env):
    """g1d_put_anything_in using AffordancePutRobot (part_tag-aware affordance grasp + v4.4 placing)."""
    from scripts.g1d import place_v42

    return place_v42.place_anything_actions(env, make_affordance_place_robot_class("put"))
