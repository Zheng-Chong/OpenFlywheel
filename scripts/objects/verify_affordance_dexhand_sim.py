"""Stage 2 physics & tactile closed-loop verification for Sharpa Wave affordance.json candidates.

Evaluates Stage 1 `grasps["sharpa_wave"]` candidates by running a lightweight single left-hand
Sharpa Wave articulation (IdealPD actuators, preserved `vendor_dynamics.json` & `collision_exclusions`,
6-DoF virtual wrist) at `dt = 1/240 s` through a 4-phase closed-loop test:
  1. Approach (`preshape_qpos`, retreating `0.07 m` along `-approach_dir`, early-abort on knock > 3 cm)
  2. Close (`thumb_first` staggered closure per `close_schedule` -> wrap close -> 0.3 s hold)
  3. Lift (+0.15 m vertical lift)
  4. Shake (+/-0.02 m translation & +/-8 deg rotation) + 0.5 s hold

Dual acceptance gate:
  - Kinematics: `lift_height_m >= 0.10 m`, `slip_m <= 0.025 m`, `slip_deg <= 20 deg`, finite states
  - PhysX elastomer opposing contact forces (`left_*_elastomer` filtered against the target object):
      * `power_*` / `handle_hook`: `thumb > 0.30 N` AND >= 2 opposing fingers `> 0.20 N`
      * `tripod_pinch`:            `thumb > 0.25 N` AND >= 1 of (`index`, `middle`) `> 0.25 N`

Results are written back idempotently into `affordance.json["grasps"]["sharpa_wave"]`
(`verified`, `sim_score`, `sim_metrics`, preserving `geom_score` and `grasps["dex1_1"]`).

Usage:
    python scripts/objects/verify_affordance_dexhand_sim.py path/to/affordance.json --dry-run
    isaac-sim python scripts/objects/verify_affordance_dexhand_sim.py path/to/affordance.json --asset obj.usdz --isaac
    isaac-sim python scripts/objects/verify_affordance_dexhand_sim.py --manifest manifest.json \
        --isaac --save-vis-dir vis/
"""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Protocol
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.objects import affordance as aff  # noqa: E402
from scripts.objects.verify_affordance_sim import (  # noqa: E402
    compute_relative_slip,
    quat_wxyz_to_rot,
    rot_to_quat_wxyz,
)

SHARPA_GRIPPER_KEY = "sharpa_wave"
SHARPA_ROBOT_DIR = ROOT / "real2sim" / "robot" / "g1d_sharpa_wave"
SHARPA_URDF_PATH = SHARPA_ROBOT_DIR / "robot.urdf"
SHARPA_CONTRACT_PATH = SHARPA_ROBOT_DIR / "model_contract.json"
SHARPA_VENDOR_DYNAMICS_PATH = SHARPA_ROBOT_DIR / "vendor_dynamics.json"

FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
LEFT_CONTACT_LINKS: tuple[str, ...] = tuple(f"left_{f}_elastomer" for f in FINGER_NAMES)
VIRTUAL_WRIST_JOINTS: tuple[str, ...] = ("vw_tx", "vw_ty", "vw_tz", "vw_rx", "vw_ry", "vw_rz")

DEFAULT_DT = 1.0 / 240.0
DEFAULT_APPROACH_DIST_M = 0.07
DEFAULT_MAX_APPROACH_DISP_M = 0.03
DEFAULT_LIFT_HEIGHT_M = 0.15
DEFAULT_MIN_LIFT_M = 0.10
DEFAULT_MAX_SLIP_M = 0.025
DEFAULT_MAX_SLIP_DEG = 20.0
DEFAULT_SHAKE_AMP_M = 0.02
DEFAULT_SHAKE_DEG = 8.0
DEFAULT_MAX_LIN_VEL_MPS = 2.0
DEFAULT_MAX_ANG_VEL_RPS = 15.0


@dataclass(frozen=True)
class DexHandVerificationResult:
    """Outcome of a single Sharpa Wave physics + tactile grasp trial."""

    grasp_id: str
    passed: bool
    sim_score: float  # in [0.0, 1.0]
    lift_height_m: float = 0.0
    slip_m: float = 0.0
    slip_deg: float = 0.0
    contact_fingers_n: int = 0
    thumb_force_n: float = 0.0
    opposing_force_n: float = 0.0
    finger_forces_n: dict[str, float] = field(default_factory=dict)
    tacmap_active_pixels: int | None = None
    reason: str = ""

    def to_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "lift_height_m": round(float(self.lift_height_m), 4),
            "slip_m": round(float(self.slip_m), 4),
            "slip_deg": round(float(self.slip_deg), 2),
            "contact_fingers_n": int(self.contact_fingers_n),
            "thumb_force_n": round(float(self.thumb_force_n), 3),
            "opposing_force_n": round(float(self.opposing_force_n), 3),
            "finger_forces_n": {k: round(float(v), 3) for k, v in self.finger_forces_n.items()},
            "reason": self.reason,
        }
        if self.tacmap_active_pixels is not None:
            metrics["tacmap_active_pixels"] = int(self.tacmap_active_pixels)
        return metrics


# ---------------------------------------------------------------------------
# Pure-Python / CPU-testable helpers (URDF extraction, scheduling, scoring)
# ---------------------------------------------------------------------------


def build_left_hand_urdf(
    source_urdf: Path | str = SHARPA_URDF_PATH,
    out_urdf: Path | str | None = None,
) -> Path:
    """Extract a standalone left Sharpa Wave URDF with a 6-DoF virtual wrist root.

    Preserves all 35 `left_*` hand links, 34 hand joints, inertial tensors, and STL meshes
    (rewritten to absolute paths so URDF import works from any working directory).
    """
    src_path = Path(source_urdf).resolve()
    src_dir = src_path.parent
    tree = ET.parse(src_path)
    root = tree.getroot()

    by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        if parent_el is not None and parent_el.get("link"):
            by_parent.setdefault(parent_el.get("link", ""), []).append(joint)

    link_by_name = {el.get("name", ""): el for el in root.findall("link")}
    if "left_hand_flange" not in link_by_name:
        raise ValueError(f"`left_hand_flange` not found in {src_path}")

    # BFS from left_hand_flange to collect all left-hand links & joints
    kept_links: list[str] = ["left_hand_flange"]
    kept_joints: list[ET.Element] = []
    queue = ["left_hand_flange"]
    seen = {"left_hand_flange"}
    while queue:
        cur = queue.pop(0)
        for j in by_parent.get(cur, ()):
            kept_joints.append(j)
            child_el = j.find("child")
            if child_el is None:
                continue
            cname = child_el.get("link", "")
            if cname and cname not in seen:
                seen.add(cname)
                kept_links.append(cname)
                queue.append(cname)

    out_root = ET.Element("robot", {"name": "sharpa_wave_left_standalone"})

    def _add_dummy_link(name: str) -> None:
        lk = ET.SubElement(out_root, "link", {"name": name})
        iner = ET.SubElement(lk, "inertial")
        ET.SubElement(iner, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(iner, "mass", {"value": "0.005"})
        ET.SubElement(
            iner, "inertia", {"ixx": "1e-5", "ixy": "0", "ixz": "0", "iyy": "1e-5", "iyz": "0", "izz": "1e-5"}
        )

    # 6-DoF virtual wrist chain: virtual_wrist_base -> tx -> ty -> tz -> rx -> ry -> rz -> left_hand_flange
    vw_links = ["virtual_wrist_base", "vw_tx_link", "vw_ty_link", "vw_tz_link", "vw_rx_link", "vw_ry_link"]
    for lname in vw_links:
        _add_dummy_link(lname)

    vw_specs = [
        ("vw_tx", "prismatic", "virtual_wrist_base", "vw_tx_link", "1 0 0", "-0.6", "0.6", "500.0", "5.0"),
        ("vw_ty", "prismatic", "vw_tx_link", "vw_ty_link", "0 1 0", "-0.6", "0.6", "500.0", "5.0"),
        ("vw_tz", "prismatic", "vw_ty_link", "vw_tz_link", "0 0 1", "-0.6", "0.6", "500.0", "5.0"),
        ("vw_rx", "revolute", "vw_tz_link", "vw_rx_link", "1 0 0", "-1.0", "1.0", "200.0", "10.0"),
        ("vw_ry", "revolute", "vw_rx_link", "vw_ry_link", "0 1 0", "-1.0", "1.0", "200.0", "10.0"),
        ("vw_rz", "revolute", "vw_ry_link", "left_hand_flange", "0 0 1", "-1.0", "1.0", "200.0", "10.0"),
    ]
    for jname, jtype, parent_n, child_n, axis_xyz, lo, hi, eff, vel in vw_specs:
        jel = ET.SubElement(out_root, "joint", {"name": jname, "type": jtype})
        ET.SubElement(jel, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(jel, "parent", {"link": parent_n})
        ET.SubElement(jel, "child", {"link": child_n})
        ET.SubElement(jel, "axis", {"xyz": axis_xyz})
        ET.SubElement(jel, "limit", {"lower": lo, "upper": hi, "effort": eff, "velocity": vel})

    # Copy all left-hand links with resolved absolute mesh paths
    for lname in kept_links:
        el = copy.deepcopy(link_by_name[lname])
        for mesh_el in el.findall(".//mesh"):
            fname = mesh_el.get("filename", "")
            if fname and not os.path.isabs(fname) and not fname.startswith("package://"):
                mesh_el.set("filename", str((src_dir / fname).resolve()))
        out_root.append(el)

    for jel in kept_joints:
        out_root.append(copy.deepcopy(jel))

    xml_bytes = ET.tostring(out_root, encoding="utf-8", xml_declaration=True)
    if out_urdf is None:
        digest = hashlib.sha256(xml_bytes).hexdigest()[:12]
        cache_dir = Path(tempfile.gettempdir()) / "ofw_sharpa_left_urdf"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"left_sharpa_wave_{digest}.urdf"
    else:
        out_path = Path(out_urdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.is_file() or out_path.read_bytes() != xml_bytes:
        out_path.write_bytes(xml_bytes)
    return out_path


def interpolate_close_qpos(
    preshape_q: np.ndarray | list[float],
    target_q: np.ndarray | list[float],
    progress: float,
    *,
    thumb_first_joints: int = 5,
    thumb_lead_ratio: float = 0.35,
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 22-DoF joint targets at normalized closure `progress` in [0, 1].

    When `thumb_first_joints > 0`, the thumb joints (`[:thumb_first_joints]`) close first
    over `[0, thumb_lead_ratio]` to establish opposition before the four fingers wrap over
    `[thumb_lead_ratio, 1.0]`, preventing objects from being knocked over toward the wrist.
    """
    q0 = np.asarray(preshape_q, dtype=np.float64).reshape(-1)
    q1 = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape:
        raise ValueError(f"preshape_q shape {q0.shape} != target_q shape {q1.shape}")

    p = float(np.clip(progress, 0.0, 1.0))
    n = q0.shape[0]
    k = int(max(0, min(thumb_first_joints, n)))
    lead = float(np.clip(thumb_lead_ratio, 0.0, 0.95))

    if k == 0 or lead <= 1e-6:
        q = (1.0 - p) * q0 + p * q1
    else:
        alpha_thumb = min(1.0, p / lead)
        alpha_fingers = max(0.0, (p - lead) / max(1.0 - lead, 1e-6))
        q = np.empty_like(q0)
        q[:k] = (1.0 - alpha_thumb) * q0[:k] + alpha_thumb * q1[:k]
        q[k:] = (1.0 - alpha_fingers) * q0[k:] + alpha_fingers * q1[k:]

    if lower_limits is not None and upper_limits is not None:
        q = np.clip(q, lower_limits, upper_limits)
    return q


def evaluate_contact_forces(
    finger_forces_n: dict[str, float],
    grasp_type: str,
) -> tuple[bool, str, int, float, float]:
    """Check PhysX elastomer opposing contact force gate for a grasp type.

    Returns `(contact_ok, reason, contact_fingers_n, thumb_force_n, opposing_force_n)`.
    """
    forces = {f: float(finger_forces_n.get(f, 0.0)) for f in FINGER_NAMES}
    if not all(math.isfinite(v) for v in forces.values()):
        return False, "unstable_contact_nan", 0, 0.0, 0.0

    thumb_f = forces["thumb"]
    opposing_sum = sum(forces[f] for f in ("index", "middle", "ring", "pinky"))

    if grasp_type == "tripod_pinch":
        thumb_thr = 0.25
        opp_thr = 0.25
        active_n = int(thumb_f > thumb_thr) + sum(
            1 for f in ("index", "middle", "ring", "pinky") if forces[f] > opp_thr
        )
        opp_pinch_n = sum(1 for f in ("index", "middle") if forces[f] > opp_thr)
        if thumb_f <= thumb_thr:
            return False, "insufficient_thumb_contact", active_n, thumb_f, opposing_sum
        if opp_pinch_n < 1:
            return False, "insufficient_opposing_contact", active_n, thumb_f, opposing_sum
        return True, "ok", active_n, thumb_f, opposing_sum

    if grasp_type == "handle_hook":
        # Hook grasps carry gravity load primarily on hooked index/middle/ring fingers;
        # accept either thumb brace (>0.20 N) + >=2 fingers (>0.20 N) or a strong multi-finger hook
        # (>=2 fingers >0.30 N and total opposing load >1.0 N).
        opp_hook_n = sum(1 for f in ("index", "middle", "ring", "pinky") if forces[f] > 0.20)
        opp_strong_n = sum(1 for f in ("index", "middle", "ring", "pinky") if forces[f] > 0.30)
        active_n = int(thumb_f > 0.20) + opp_hook_n
        if opp_hook_n < 2:
            return False, "insufficient_opposing_contact", active_n, thumb_f, opposing_sum
        if thumb_f > 0.20 or (opp_strong_n >= 2 and opposing_sum > 1.0):
            return True, "ok", active_n, thumb_f, opposing_sum
        return False, "insufficient_thumb_contact", active_n, thumb_f, opposing_sum

    # power_cylindrical / power_spherical
    thumb_thr = 0.30
    opp_thr = 0.20
    opp_active_n = sum(1 for f in ("index", "middle", "ring", "pinky") if forces[f] > opp_thr)
    active_n = int(thumb_f > thumb_thr) + opp_active_n
    if thumb_f <= thumb_thr:
        return False, "insufficient_thumb_contact", active_n, thumb_f, opposing_sum
    if opp_active_n < 2:
        return False, "insufficient_opposing_contact", active_n, thumb_f, opposing_sum
    return True, "ok", active_n, thumb_f, opposing_sum


def _is_valid_quat(q: np.ndarray) -> bool:
    """Return True iff `q` is a finite 4-vector with non-zero norm (guards SciPy Rotation)."""
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    return arr.shape == (4,) and bool(np.all(np.isfinite(arr))) and float(np.linalg.norm(arr)) > 1e-6


def _extract_contact_force_mag(fmat: Any) -> float:
    """Extract net 3D contact force magnitude (N) from `ContactSensor.data.force_matrix_w`.

    `force_matrix_w` has shape `(num_envs, num_bodies, num_filter_bodies, 3)`. Summing over
    all matched filter bodies handles both single-body and multi-body object USDs.
    """
    if fmat is None:
        return 0.0
    arr = np.asarray(
        fmat.detach().cpu() if hasattr(fmat, "detach") else (fmat.cpu() if hasattr(fmat, "cpu") else fmat),
        dtype=np.float64,
    )
    if arr.size < 3:
        return 0.0
    if not np.all(np.isfinite(arr)):
        return float("nan")
    if arr.ndim == 4:
        fvec = arr[0, 0].sum(axis=0)
    elif arr.ndim == 3:
        fvec = arr[0].sum(axis=0)
    else:
        fvec = arr.reshape(-1, 3).sum(axis=0)
    return float(np.linalg.norm(fvec))


def _vis_snapshot_key(res: DexHandVerificationResult) -> tuple[int, float, float]:
    """Ranking key for choosing the best per-asset visual verification snapshot across chunks."""
    return (1 if res.passed else 0, float(res.sim_score), float(res.lift_height_m))


def score_dexhand_trial(
    grasp_id: str,
    grasp_type: str,
    lift_height_m: float,
    slip_m: float,
    slip_deg: float,
    finger_forces_n: dict[str, float],
    *,
    approach_disp_m: float = 0.0,
    max_approach_disp_m: float = DEFAULT_MAX_APPROACH_DISP_M,
    target_lift_m: float = DEFAULT_LIFT_HEIGHT_M,
    min_lift_m: float = DEFAULT_MIN_LIFT_M,
    max_slip_m: float = DEFAULT_MAX_SLIP_M,
    max_slip_deg: float = DEFAULT_MAX_SLIP_DEG,
    obj_lin_vel_mps: float = 0.0,
    obj_ang_vel_rps: float = 0.0,
    max_lin_vel_mps: float = DEFAULT_MAX_LIN_VEL_MPS,
    max_ang_vel_rps: float = DEFAULT_MAX_ANG_VEL_RPS,
    tacmap_active_pixels: int | None = None,
) -> DexHandVerificationResult:
    """Evaluate a completed Sharpa Wave physics trial against dual kinematics + tactile gates."""
    forces = {f: float(finger_forces_n.get(f, 0.0)) for f in FINGER_NAMES}
    contact_ok, contact_reason, contact_n, thumb_f, opp_f = evaluate_contact_forces(forces, grasp_type)

    if not (
        math.isfinite(lift_height_m)
        and math.isfinite(slip_m)
        and math.isfinite(slip_deg)
        and math.isfinite(approach_disp_m)
        and math.isfinite(obj_lin_vel_mps)
        and math.isfinite(obj_ang_vel_rps)
    ):
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            finger_forces_n=forces,
            reason="unstable_nan",
        )

    if obj_lin_vel_mps > max_lin_vel_mps or obj_ang_vel_rps > max_ang_vel_rps:
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            contact_fingers_n=contact_n,
            thumb_force_n=thumb_f,
            opposing_force_n=opp_f,
            finger_forces_n=forces,
            tacmap_active_pixels=tacmap_active_pixels,
            reason="unstable_velocity",
        )

    if approach_disp_m > max_approach_disp_m:
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            contact_fingers_n=contact_n,
            thumb_force_n=thumb_f,
            opposing_force_n=opp_f,
            finger_forces_n=forces,
            tacmap_active_pixels=tacmap_active_pixels,
            reason="approach_collision",
        )

    if lift_height_m < min_lift_m:
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            contact_fingers_n=contact_n,
            thumb_force_n=thumb_f,
            opposing_force_n=opp_f,
            finger_forces_n=forces,
            tacmap_active_pixels=tacmap_active_pixels,
            reason="insufficient_lift",
        )

    if slip_m > max_slip_m:
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            contact_fingers_n=contact_n,
            thumb_force_n=thumb_f,
            opposing_force_n=opp_f,
            finger_forces_n=forces,
            tacmap_active_pixels=tacmap_active_pixels,
            reason="excessive_slip_pos",
        )

    if slip_deg > max_slip_deg:
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            contact_fingers_n=contact_n,
            thumb_force_n=thumb_f,
            opposing_force_n=opp_f,
            finger_forces_n=forces,
            tacmap_active_pixels=tacmap_active_pixels,
            reason="excessive_slip_rot",
        )

    if not contact_ok:
        return DexHandVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            contact_fingers_n=contact_n,
            thumb_force_n=thumb_f,
            opposing_force_n=opp_f,
            finger_forces_n=forces,
            tacmap_active_pixels=tacmap_active_pixels,
            reason=contact_reason,
        )

    lift_q = min(1.0, max(0.0, lift_height_m / max(target_lift_m, 1e-6)))
    slip_pos_q = max(0.0, 1.0 - slip_m / max(max_slip_m, 1e-6))
    slip_rot_q = max(0.0, 1.0 - slip_deg / max(max_slip_deg, 1e-6))
    denom_fingers = 3.0 if grasp_type == "tripod_pinch" else 5.0
    finger_q = min(1.0, contact_n / denom_fingers)
    balance_q = (2.0 * min(thumb_f, opp_f)) / max(thumb_f + opp_f, 1e-6)
    sim_score = 0.5 + 0.5 * (0.25 * lift_q + 0.25 * slip_pos_q + 0.20 * slip_rot_q + 0.15 * finger_q + 0.15 * balance_q)

    return DexHandVerificationResult(
        grasp_id=grasp_id,
        passed=True,
        sim_score=round(float(sim_score), 4),
        lift_height_m=lift_height_m,
        slip_m=slip_m,
        slip_deg=slip_deg,
        contact_fingers_n=contact_n,
        thumb_force_n=thumb_f,
        opposing_force_n=opp_f,
        finger_forces_n=forces,
        tacmap_active_pixels=tacmap_active_pixels,
        reason="ok",
    )


def compute_world_wrist_waypoints(
    obj_pos: np.ndarray,
    obj_rot: np.ndarray,
    grasp: dict[str, Any],
    *,
    approach_dist_m: float = DEFAULT_APPROACH_DIST_M,
    lift_height_m: float = DEFAULT_LIFT_HEIGHT_M,
) -> tuple[bool, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute `(ok, reason, p_pre, p_grasp, r_grasp, p_lift)` in world frame for `left_hand_flange`."""
    p_loc = np.asarray(grasp["wrist_position"], dtype=np.float64).reshape(3)
    r_loc = quat_wxyz_to_rot(grasp["wrist_quaternion_wxyz"]).as_matrix()
    app_loc = np.asarray(grasp["approach_dir"], dtype=np.float64).reshape(3)
    app_norm = float(np.linalg.norm(app_loc))
    if app_norm < 1e-8:
        return False, "invalid_approach_dir", np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)
    app_w = obj_rot @ (app_loc / app_norm)

    if float(app_w[2]) > 0.25:
        return False, "upward_approach", np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)

    p_grasp = obj_pos + obj_rot @ p_loc
    r_grasp = obj_rot @ r_loc
    p_pre = p_grasp - float(approach_dist_m) * app_w
    p_lift = p_grasp + np.array([0.0, 0.0, float(lift_height_m)], dtype=np.float64)
    return True, "ok", p_pre, p_grasp, r_grasp, p_lift


# ---------------------------------------------------------------------------
# Verifier protocol & implementations
# ---------------------------------------------------------------------------


class DexHandGraspVerifier(Protocol):
    """Pluggable physics verifier interface for Sharpa Wave Stage 2 verification."""

    def verify_batch(
        self,
        asset_path: Path | None,
        grasps: list[dict[str, Any]],
        *,
        asset_id: str = "",
    ) -> list[DexHandVerificationResult]: ...


class DryRunDexHandVerifier:
    """CPU-only dry-run verifier for testing the Stage 2 writeback & manifest pipeline."""

    def __init__(self, score_threshold: float = 0.50):
        self.score_threshold = float(score_threshold)

    def verify_batch(
        self,
        asset_path: Path | None,
        grasps: list[dict[str, Any]],
        *,
        asset_id: str = "",
    ) -> list[DexHandVerificationResult]:
        out: list[DexHandVerificationResult] = []
        for g in grasps:
            geom_score = float(g.get("geom_score", g.get("score", 0.0)))
            gtype = str(g.get("grasp_type", "power_cylindrical"))
            passed = geom_score >= self.score_threshold
            if passed:
                if gtype == "tripod_pinch":
                    forces = {"thumb": 1.4, "index": 1.2, "middle": 0.9, "ring": 0.0, "pinky": 0.0}
                else:
                    forces = {"thumb": 2.8, "index": 1.6, "middle": 1.5, "ring": 1.1, "pinky": 0.7}
                res = score_dexhand_trial(
                    str(g["id"]),
                    gtype,
                    lift_height_m=0.145,
                    slip_m=0.005,
                    slip_deg=2.2,
                    finger_forces_n=forces,
                )
                out.append(
                    DexHandVerificationResult(
                        grasp_id=res.grasp_id,
                        passed=True,
                        sim_score=round(geom_score, 4),
                        lift_height_m=res.lift_height_m,
                        slip_m=res.slip_m,
                        slip_deg=res.slip_deg,
                        contact_fingers_n=res.contact_fingers_n,
                        thumb_force_n=res.thumb_force_n,
                        opposing_force_n=res.opposing_force_n,
                        finger_forces_n=res.finger_forces_n,
                        reason="dry_run_ok",
                    )
                )
            else:
                out.append(
                    DexHandVerificationResult(
                        grasp_id=str(g["id"]),
                        passed=False,
                        sim_score=0.0,
                        lift_height_m=0.0,
                        slip_m=0.05,
                        slip_deg=35.0,
                        contact_fingers_n=0,
                        thumb_force_n=0.0,
                        opposing_force_n=0.0,
                        finger_forces_n={f: 0.0 for f in FINGER_NAMES},
                        reason="dry_run_below_threshold",
                    )
                )
        return out


class IsaacSimDexHandVerifier:
    """Isaac Sim 5.1 headless physics + tactile verifier for Sharpa Wave (runs via `isaac-sim`).

    Spawns lightweight standalone left Sharpa Wave articulations (IdealPD actuators, preserved
    `vendor_dynamics.json` gains/armature and `collision_exclusions`, 6-DoF virtual wrist) at
    `dt = 1/240 s`, executes `approach -> thumb_first close -> lift -> shake`, and enforces
    kinematics + PhysX `left_*_elastomer` opposing contact force gates.
    """

    def __init__(
        self,
        *,
        dt: float = DEFAULT_DT,
        device: str = "cuda:0",
        spacing: float = 0.85,
        settle_steps: int = 120,
        approach_dist_m: float = DEFAULT_APPROACH_DIST_M,
        approach_steps: int = 96,
        max_approach_disp_m: float = DEFAULT_MAX_APPROACH_DISP_M,
        close_steps: int = 168,
        close_hold_steps: int = 72,
        lift_height_m: float = DEFAULT_LIFT_HEIGHT_M,
        lift_steps: int = 120,
        shake_amp_m: float = DEFAULT_SHAKE_AMP_M,
        shake_deg: float = DEFAULT_SHAKE_DEG,
        shake_steps: int = 144,
        final_hold_steps: int = 120,
        min_lift_m: float = DEFAULT_MIN_LIFT_M,
        max_slip_m: float = DEFAULT_MAX_SLIP_M,
        max_slip_deg: float = DEFAULT_MAX_SLIP_DEG,
        max_parallel: int = 28,
        save_vis_dir: Path | str | None = None,
    ):
        self.dt = float(dt)
        self.device = str(device)
        self.spacing = float(spacing)
        self.settle_steps = int(settle_steps)
        self.approach_dist_m = float(approach_dist_m)
        self.approach_steps = int(approach_steps)
        self.max_approach_disp_m = float(max_approach_disp_m)
        self.close_steps = int(close_steps)
        self.close_hold_steps = int(close_hold_steps)
        self.lift_height_m = float(lift_height_m)
        self.lift_steps = int(lift_steps)
        self.shake_amp_m = float(shake_amp_m)
        self.shake_rad = math.radians(float(shake_deg))
        self.shake_steps = int(shake_steps)
        self.final_hold_steps = int(final_hold_steps)
        self.min_lift_m = float(min_lift_m)
        self.max_slip_m = float(max_slip_m)
        self.max_slip_deg = float(max_slip_deg)
        self.max_parallel = max(1, int(max_parallel))
        self.save_vis_dir = Path(save_vis_dir) if save_vis_dir is not None else None

        self._app: Any = None
        self._sim: Any = None
        self._left_urdf_path: Path | None = None
        self._contract = json.loads(SHARPA_CONTRACT_PATH.read_text(encoding="utf-8"))
        self._vendor_dynamics = json.loads(SHARPA_VENDOR_DYNAMICS_PATH.read_text(encoding="utf-8"))
        self._left_joints: tuple[str, ...] = tuple(self._contract["hands"]["left"]["joints"])
        self._left_limits: dict[str, dict[str, float]] = self._contract["hands"]["left"]["limits"]
        self._lower = np.array([float(self._left_limits[j]["lower"]) for j in self._left_joints], dtype=np.float64)
        self._upper = np.array([float(self._left_limits[j]["upper"]) for j in self._left_joints], dtype=np.float64)

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        try:
            from isaaclab.app import AppLauncher  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "IsaacSimDexHandVerifier requires Isaac Sim + IsaacLab (run via `isaac-sim` on GPU Slurm)."
            ) from exc
        launcher = AppLauncher(headless=True, enable_cameras=self.save_vis_dir is not None)
        self._app = launcher.app
        self._left_urdf_path = build_left_hand_urdf(SHARPA_URDF_PATH)

    def close(self) -> None:
        if self._app is not None:
            self._app.close()
            self._app = None
            self._sim = None

    @staticmethod
    def _resolve_asset_usd_and_meta(asset_path: Path) -> tuple[Path, dict[str, Any]]:
        p = Path(asset_path)
        if p.is_dir():
            asset_dir = p
            usd_file = None
            for name in ("object.usdz", "object.usd", "object.usda"):
                cand = asset_dir / name
                if cand.is_file():
                    usd_file = cand
                    break
            if usd_file is None:
                raise FileNotFoundError(f"No object.usd[z|a] found in {asset_dir}")
        else:
            usd_file = p
            asset_dir = p.parent

        meta: dict[str, Any] = {}
        meta_file = asset_dir / "metadata.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        return usd_file, meta

    @staticmethod
    def _as_np(x: Any) -> np.ndarray:
        return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=np.float64).reshape(-1)

    def _build_hand_articulation_cfg(self, prim_path: str, park_pos: tuple[float, float, float]) -> Any:
        """Build lightweight single left Sharpa Wave ArticulationCfg preserving vendor_dynamics & exclusions."""
        from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import ArticulationCfg  # type: ignore[import-not-found]
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        from isaaclab.sim.spawners.from_files import from_files  # type: ignore[import-not-found]
        from pxr import Gf, PhysxSchema, Sdf, UsdPhysics  # type: ignore[import-not-found]

        assert self._left_urdf_path is not None
        left_urdf = self._left_urdf_path
        contract = self._contract
        left_dyn = self._vendor_dynamics["left"]
        left_links = set(left_dyn["rigid_bodies"].keys())
        left_exclusions = [
            (a, b) for a, b in contract["collision_exclusions"] if a.startswith("left_") and b.startswith("left_")
        ]
        source_xml = ET.parse(left_urdf).getroot()

        @sim_utils.clone
        def _spawn_left_hand(p_path: str, cfg: Any, *args: Any, **kwargs: Any) -> Any:
            prim = from_files.spawn_from_urdf.__wrapped__(p_path, cfg, *args, **kwargs)
            stage = prim.GetStage()

            # Restore manufacturer hand RGB visual materials for clear verification renders
            materials: dict[str, str] = {}
            for link_el in source_xml.findall("link"):
                lname = link_el.get("name", "")
                if lname not in left_links:
                    continue
                child = prim.GetChild(lname)
                if not child:
                    continue
                visuals = child.GetChild("visuals")
                if not visuals:
                    continue
                color = link_el.find("visual/material/color")
                if color is None or not color.get("rgba"):
                    continue
                rgb = tuple(float(v) for v in color.get("rgba", "0.5 0.5 0.5").split()[:3])
                key = "hand_" + str(tuple(round(v, 3) for v in rgb)).replace(" ", "").replace(".", "_").replace(
                    ",", "_"
                ).replace("(", "").replace(")", "")
                mat_path = p_path + "/Looks/" + key
                if key not in materials:
                    mcfg = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=rgb,
                        metallic=0.65 if min(rgb) > 0.5 else 0.0,
                        roughness=0.55,
                    )
                    mcfg.func(mat_path, mcfg)
                    materials[key] = mat_path
                visuals.SetInstanceable(False)
                sim_utils.bind_visual_material(str(visuals.GetPath()), materials[key])

            # Preserve manufacturer self-collision exclusions
            for first, second in left_exclusions:
                a, b = prim.GetChild(first), prim.GetChild(second)
                if not a or not b:
                    raise ValueError(f"Unknown left-hand collision pair: {first}, {second}")
                UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(b.GetPath())

            # Preserve vendor query-frame masses so URDF import does not assign 1 kg fallbacks
            for lname, attrs in left_dyn["rigid_bodies"].items():
                if not lname.endswith(("_tip", "_fingertip", "_wrist_imu_frame")):
                    continue
                body = prim.GetChild(lname)
                if not body:
                    raise ValueError(f"Missing left-hand query frame: {lname}")
                mass = UsdPhysics.MassAPI.Apply(body)
                mass.CreateMassAttr(float(attrs["physics:mass"]))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*ast.literal_eval(attrs["physics:diagonalInertia"])))

            # Preserve vendor USD joint stiffness / damping / armature / friction
            for jname, attrs in left_dyn["joints"].items():
                if not attrs:
                    continue
                joint = stage.GetPrimAtPath(p_path + "/joints/" + jname)
                if not joint:
                    raise ValueError(f"Missing manufacturer joint: {jname}")
                PhysxSchema.PhysxJointAPI.Apply(joint)
                for key, value in attrs.items():
                    if not key.startswith(("drive:", "physxJoint:")) or key.endswith(
                        ("targetPosition", "targetVelocity")
                    ):
                        continue
                    attr = joint.GetAttribute(key)
                    if not attr:
                        attr = joint.CreateAttribute(
                            key, Sdf.ValueTypeNames.Token if isinstance(value, str) else Sdf.ValueTypeNames.Float
                        )
                    attr.Set(value)

            return prim

        usd_cache_dir = str(left_urdf.parent / "usd_cache")
        init_jpos = {j: 0.0 for j in VIRTUAL_WRIST_JOINTS}
        init_jpos.update({j: 0.0 for j in self._left_joints})

        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                func=_spawn_left_hand,
                asset_path=str(left_urdf),
                usd_dir=usd_cache_dir,
                fix_base=True,
                activate_contact_sensors=True,
                merge_fixed_joints=False,
                collider_type="convex_hull",
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=park_pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=init_jpos,
            ),
            actuators={
                "virtual_wrist_t": ImplicitActuatorCfg(
                    joint_names_expr=["vw_t[xyz]"],
                    stiffness=50000.0,
                    damping=1200.0,
                    effort_limit_sim=500.0,
                    velocity_limit_sim=5.0,
                ),
                "virtual_wrist_r": ImplicitActuatorCfg(
                    joint_names_expr=["vw_r[xyz]"],
                    stiffness=5000.0,
                    damping=150.0,
                    effort_limit_sim=200.0,
                    velocity_limit_sim=10.0,
                ),
                "dexhand": IdealPDActuatorCfg(
                    joint_names_expr=list(self._left_joints),
                    stiffness=None,
                    damping=None,
                    effort_limit=None,
                    effort_limit_sim={j: float(self._left_limits[j]["effort"]) for j in self._left_joints},
                    velocity_limit=None,
                    velocity_limit_sim={j: float(self._left_limits[j]["velocity"]) for j in self._left_joints},
                ),
            },
        )

    @staticmethod
    def _look_at_quat_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
        left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
        ln = float(np.linalg.norm(left))
        if ln < 1e-6:
            left = np.array([0.0, 1.0, 0.0])
        else:
            left /= ln
        up = np.cross(fwd, left)
        rot_m = np.column_stack([fwd, left, up])
        return rot_to_quat_wxyz(Rotation.from_matrix(rot_m))

    def _save_snapshot(
        self,
        cam: Any,
        trial: dict[str, Any],
        result: DexHandVerificationResult,
        out_path: Path,
    ) -> None:
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]

        p_obj, _ = map(self._as_np, trial["obj"].get_world_pose())
        if not np.all(np.isfinite(p_obj)):
            p_obj = trial["p_lift"]
        target = p_obj.copy()
        eye = target + np.array([0.36, -0.34, 0.20], dtype=np.float64)
        q_wxyz = self._look_at_quat_wxyz(eye, target)
        cam.set_world_pose(position=eye, orientation=q_wxyz, camera_axes="world")
        self._sim.step(render=True)
        for _ in range(5):
            self._sim.render()
        rgba = cam.get_rgba()
        if rgba is None or rgba.size == 0:
            return
        rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        status = "PASS" if result.passed else f"FAIL ({result.reason})"
        banner = (
            f"{trial['asset_id']} | {trial['grasp_id']} ({trial['grasp_type']}) | {status}\n"
            f"lift={result.lift_height_m:.3f}m slip={result.slip_m * 1000:.1f}mm/{result.slip_deg:.1f}deg "
            f"F_thumb={result.thumb_force_n:.2f}N F_opp={result.opposing_force_n:.2f}N "
            f"fingers={result.contact_fingers_n}"
        )
        draw.rectangle([(0, 0), (img.width, 40)], fill=(15, 18, 24))
        draw.text((8, 6), banner, fill=(90, 240, 140) if result.passed else (255, 120, 100))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    def _run_trials_chunk(
        self,
        trial_specs: list[tuple[str, Path, dict[str, Any], dict[str, Any]]],
        *,
        best_vis_scores: dict[str, tuple[int, float, float]] | None = None,
    ) -> list[DexHandVerificationResult]:
        """Run a chunk of `(asset_id, usd_file, meta, grasp)` trials in one SimulationContext."""
        from isaaclab.assets import Articulation  # type: ignore[import-not-found]
        from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore[import-not-found]
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]
        from pxr import UsdGeom  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]

        from env.scene_manager.objects.rigid import RigidObject  # type: ignore[import-not-found]

        if not trial_specs:
            return []

        n = len(trial_specs)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / cols)))
        span_x = max(20.0, cols * self.spacing + 4.0)
        span_y = max(20.0, rows * self.spacing + 4.0)

        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device=self.device)
        if self._sim is not None:
            self._sim.clear_instance()
        omni.usd.get_context().new_stage()
        self._sim = sim_utils.SimulationContext(sim_cfg)
        stage = omni.usd.get_context().get_stage()

        floor_cfg = sim_utils.CuboidCfg(
            size=(span_x, span_y, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.35, 0.40)),
        )
        floor_cfg.func(
            "/World/floor",
            floor_cfg,
            translation=(0.5 * (cols - 1) * self.spacing, 0.5 * (rows - 1) * self.spacing, -0.02),
        )
        if self.save_vis_dir is not None:
            dome = sim_utils.DomeLightCfg(intensity=1500.0)
            dome.func("/World/Light", dome)

        UsdGeom.Xform.Define(stage, "/World/trials")
        trials: list[dict[str, Any]] = []
        for idx, (asset_id, usd_file, meta, g) in enumerate(trial_specs):
            r_idx, c_idx = divmod(idx, cols)
            origin = np.array([c_idx * self.spacing, r_idx * self.spacing, 0.0], dtype=np.float64)
            t_root = f"/World/trials/t_{idx:04d}"
            UsdGeom.Xform.Define(stage, t_root)

            physics_cfg = dict(meta.get("physics") or {})
            if not physics_cfg.get("mass"):
                physics_cfg["mass"] = 0.15
            size = physics_cfg.get("size") or ((meta.get("geometry") or {}).get("aligned_bbox") or {}).get("extents")
            half_h = 0.5 * float(size[2]) if (size and len(size) >= 3) else 0.06

            obj_path = f"{t_root}/obj_0_{idx}"
            obj = RigidObject(
                prim_path=obj_path,
                usd_path=str(usd_file),
                inst_config={"physics": physics_cfg},
                env_origin=[0.0, 0.0, 0.0],
                default_pos=(float(origin[0]), float(origin[1]), float(half_h + 0.008)),
                default_ori=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            )

            hand_path = f"{t_root}/hand"
            park_pos = (float(origin[0]), float(origin[1]), 1.35)
            hand = Articulation(self._build_hand_articulation_cfg(hand_path, park_pos))

            sensors: dict[str, Any] = {}
            for finger, link_name in zip(FINGER_NAMES, LEFT_CONTACT_LINKS):
                sensors[finger] = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{hand_path}/{link_name}",
                        filter_prim_paths_expr=[obj_path],
                        update_period=0.0,
                        max_contact_data_count_per_prim=64,
                    )
                )

            close_sched = g.get("close_schedule") or {}
            trials.append(
                {
                    "idx": idx,
                    "asset_id": asset_id,
                    "grasp": g,
                    "grasp_id": str(g.get("id", f"sw{idx:03d}")),
                    "grasp_type": str(g.get("grasp_type", "power_cylindrical")),
                    "origin": origin,
                    "obj": obj,
                    "hand": hand,
                    "sensors": sensors,
                    "preshape_q": np.asarray(g["preshape_qpos"], dtype=np.float64).reshape(aff.SHARPA_WAVE_DOF),
                    "target_q": np.asarray(g["target_qpos"], dtype=np.float64).reshape(aff.SHARPA_WAVE_DOF),
                    "thumb_first": int(close_sched.get("thumb_first_joints", 5)),
                    "thumb_lead": float(close_sched.get("thumb_lead_ratio", 0.35)),
                    "active": True,
                    "result": None,
                }
            )

        cam = None
        if self.save_vis_dir is not None:
            from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

            cam = Camera(prim_path="/World/VisCamera", resolution=(640, 480))

        self._sim.reset()
        if cam is not None:
            cam.initialize()
            cam.prim.GetAttribute("focalLength").Set(16.0)
            cam.set_clipping_range(0.005, 10.0)

        for t in trials:
            t["obj"].initialize()
            hand = t["hand"]
            jnames = list(hand.joint_names)
            t["vw_idx"] = [jnames.index(j) for j in VIRTUAL_WRIST_JOINTS]
            t["hand_idx"] = [jnames.index(j) for j in self._left_joints]
            t["flange_body_idx"] = hand.body_names.index("left_hand_flange")
            q_init = hand.data.default_joint_pos.clone()
            q_init[0, t["hand_idx"]] = torch.tensor(t["preshape_q"], dtype=torch.float32, device=self._sim.device)
            hand.write_joint_state_to_sim(q_init, torch.zeros_like(q_init))
            hand.set_joint_position_target(q_init)
            hand.write_data_to_sim()

        def _step_all(targets: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None) -> None:
            if targets is not None:
                for t in active_trials:
                    dp_loc, dr_loc, q_hand = targets[t["idx"]]
                    hand = t["hand"]
                    q_cmd = hand.data.joint_pos_target.clone()
                    q_cmd[0, t["vw_idx"][0:3]] = torch.tensor(dp_loc, dtype=torch.float32, device=self._sim.device)
                    q_cmd[0, t["vw_idx"][3:6]] = torch.tensor(dr_loc, dtype=torch.float32, device=self._sim.device)
                    q_cmd[0, t["hand_idx"]] = torch.tensor(q_hand, dtype=torch.float32, device=self._sim.device)
                    hand.set_joint_position_target(q_cmd)
                    hand.write_data_to_sim()
            self._sim.step(render=False)
            for t in trials:
                t["hand"].update(self.dt)

        # Phase 0: Settle objects on the floor while hands stay parked at z=1.35 m
        active_trials: list[dict[str, Any]] = []
        for _ in range(self.settle_steps):
            _step_all(None)

        # Phase 1: Compute world waypoints from settled objects and teleport hand virtual_wrist_base to (p_pre, R_grasp)
        zero_rot = np.zeros(3, dtype=np.float64)
        for t in trials:
            pos0, quat0 = map(self._as_np, t["obj"].get_world_pose())
            if not np.all(np.isfinite(pos0)) or not _is_valid_quat(quat0):
                t["active"] = False
                t["result"] = DexHandVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason="settle_unstable"
                )
                continue
            r0 = quat_wxyz_to_rot(quat0).as_matrix()
            ok_wp, wp_reason, p_pre, p_grasp, r_grasp, p_lift = compute_world_wrist_waypoints(
                pos0,
                r0,
                t["grasp"],
                approach_dist_m=self.approach_dist_m,
                lift_height_m=self.lift_height_m,
            )
            if not ok_wp:
                t["active"] = False
                t["result"] = DexHandVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason=wp_reason
                )
                continue

            t["obj_pos_settled"] = pos0.copy()
            t["z_settled"] = float(pos0[2])
            t["p_pre"] = p_pre
            t["p_grasp"] = p_grasp
            t["r_grasp"] = r_grasp
            t["p_lift"] = p_lift
            t["dp_grasp_loc"] = r_grasp.T @ (p_grasp - p_pre)
            t["dp_lift_loc"] = r_grasp.T @ (p_lift - p_pre)

            q_grasp_wxyz = rot_to_quat_wxyz(Rotation.from_matrix(r_grasp))
            root_pose = torch.tensor(
                [[*p_pre.tolist(), *q_grasp_wxyz.tolist()]], dtype=torch.float32, device=self._sim.device
            )
            hand = t["hand"]
            hand.write_root_pose_to_sim(root_pose)
            q_init = torch.zeros_like(hand.data.default_joint_pos)
            q_init[0, t["hand_idx"]] = torch.tensor(t["preshape_q"], dtype=torch.float32, device=self._sim.device)
            hand.write_joint_state_to_sim(q_init, torch.zeros_like(q_init))
            hand.set_joint_position_target(q_init)
            hand.write_data_to_sim()

        active_trials = [t for t in trials if t["active"]]
        if not active_trials:
            return [t["result"] for t in trials]

        # Brief pre-approach settle at p_pre (12 steps = 0.05 s)
        for _ in range(12):
            _step_all({t["idx"]: (np.zeros(3), zero_rot, t["preshape_q"]) for t in active_trials})

        # Phase 2: Approach from p_pre -> p_grasp with preshape_qpos
        for step in range(1, self.approach_steps + 1):
            alpha = step / float(self.approach_steps)
            cmds = {t["idx"]: (alpha * t["dp_grasp_loc"], zero_rot, t["preshape_q"]) for t in active_trials}
            _step_all(cmds)

        # Check approach collision (object knocked > max_approach_disp_m)
        still_active: list[dict[str, Any]] = []
        for t in active_trials:
            p_obj_a, _ = map(self._as_np, t["obj"].get_world_pose())
            if not np.all(np.isfinite(p_obj_a)):
                t["active"] = False
                t["result"] = DexHandVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason="unstable_nan"
                )
                continue
            disp = float(np.linalg.norm(p_obj_a - t["obj_pos_settled"]))
            t["approach_disp_m"] = disp
            if disp > self.max_approach_disp_m:
                t["active"] = False
                t["result"] = score_dexhand_trial(
                    t["grasp_id"],
                    t["grasp_type"],
                    lift_height_m=0.0,
                    slip_m=disp,
                    slip_deg=0.0,
                    finger_forces_n={f: 0.0 for f in FINGER_NAMES},
                    approach_disp_m=disp,
                    max_approach_disp_m=self.max_approach_disp_m,
                )
                continue
            still_active.append(t)
        active_trials = still_active
        if not active_trials:
            return [t["result"] for t in trials]

        # Phase 3: Close fingers per close_schedule (thumb_first -> wrap) + 0.3 s hold
        for step in range(1, self.close_steps + 1):
            prog = step / float(self.close_steps)
            cmds = {}
            for t in active_trials:
                q_t = interpolate_close_qpos(
                    t["preshape_q"],
                    t["target_q"],
                    prog,
                    thumb_first_joints=t["thumb_first"],
                    thumb_lead_ratio=t["thumb_lead"],
                    lower_limits=self._lower,
                    upper_limits=self._upper,
                )
                cmds[t["idx"]] = (t["dp_grasp_loc"], zero_rot, q_t)
            _step_all(cmds)

        for _ in range(self.close_hold_steps):
            cmds = {t["idx"]: (t["dp_grasp_loc"], zero_rot, t["target_q"]) for t in active_trials}
            _step_all(cmds)

        # Record post-close relative pose between flange and object
        for t in active_trials:
            p_obj_c, q_obj_c = map(self._as_np, t["obj"].get_world_pose())
            fl_idx = t["flange_body_idx"]
            p_fl_c = self._as_np(t["hand"].data.body_pos_w[0, fl_idx])
            q_fl_c = self._as_np(t["hand"].data.body_quat_w[0, fl_idx])
            t["obj_pos_close"] = p_obj_c
            t["obj_rot_close"] = quat_wxyz_to_rot(q_obj_c).as_matrix() if _is_valid_quat(q_obj_c) else np.eye(3)
            t["fl_pos_close"] = p_fl_c
            t["fl_rot_close"] = quat_wxyz_to_rot(q_fl_c).as_matrix() if _is_valid_quat(q_fl_c) else t["r_grasp"]

        # Phase 4a: Lift vertically +lift_height_m
        for step in range(1, self.lift_steps + 1):
            alpha = step / float(self.lift_steps)
            cmds = {}
            for t in active_trials:
                dp = (1.0 - alpha) * t["dp_grasp_loc"] + alpha * t["dp_lift_loc"]
                cmds[t["idx"]] = (dp, zero_rot, t["target_q"])
            _step_all(cmds)

        # Phase 4b: Sinusoidal translation + rotational shake (+/-0.02 m, +/-8 deg)
        for step in range(1, self.shake_steps + 1):
            phase = 2.0 * math.pi * step / float(self.shake_steps)
            off_w = np.array(
                [self.shake_amp_m * math.sin(phase), 0.5 * self.shake_amp_m * math.sin(phase), 0.0],
                dtype=np.float64,
            )
            dr = np.array(
                [self.shake_rad * math.sin(phase), 0.6 * self.shake_rad * math.sin(2.0 * phase), 0.0],
                dtype=np.float64,
            )
            cmds = {}
            for t in active_trials:
                dp_off = t["r_grasp"].T @ off_w
                cmds[t["idx"]] = (t["dp_lift_loc"] + dp_off, dr, t["target_q"])
            _step_all(cmds)

        # Phase 4c: Return to p_lift and hold 0.5 s while sampling filtered PhysX elastomer forces
        force_samples: dict[int, dict[str, list[float]]] = {
            t["idx"]: {f: [] for f in FINGER_NAMES} for t in active_trials
        }
        sample_start = max(0, self.final_hold_steps - 48)
        for step in range(self.final_hold_steps):
            cmds = {t["idx"]: (t["dp_lift_loc"], zero_rot, t["target_q"]) for t in active_trials}
            _step_all(cmds)
            if step >= sample_start and (step % 4 == 0):
                for t in active_trials:
                    for finger, sensor in t["sensors"].items():
                        sensor.update(self.dt, force_recompute=True)
                        fmag = _extract_contact_force_mag(sensor.data.force_matrix_w)
                        force_samples[t["idx"]][finger].append(fmag)

        # Evaluate dual gate (kinematics + PhysX elastomer opposing forces)
        for t in active_trials:
            p_obj_f, q_obj_f = map(self._as_np, t["obj"].get_world_pose())
            fl_idx = t["flange_body_idx"]
            p_fl_f = self._as_np(t["hand"].data.body_pos_w[0, fl_idx])
            q_fl_f = self._as_np(t["hand"].data.body_quat_w[0, fl_idx])
            if not (
                np.all(np.isfinite(p_obj_f))
                and _is_valid_quat(q_obj_f)
                and np.all(np.isfinite(p_fl_f))
                and _is_valid_quat(q_fl_f)
            ):
                t["result"] = DexHandVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason="unstable_nan"
                )
                continue

            obj_lin_vel_mps = 0.0
            obj_ang_vel_rps = 0.0
            if hasattr(t["obj"], "get_linear_velocity"):
                v_lin = self._as_np(t["obj"].get_linear_velocity())
                obj_lin_vel_mps = float(np.linalg.norm(v_lin)) if np.all(np.isfinite(v_lin)) else float("nan")
            if hasattr(t["obj"], "get_angular_velocity"):
                v_ang = self._as_np(t["obj"].get_angular_velocity())
                obj_ang_vel_rps = float(np.linalg.norm(v_ang)) if np.all(np.isfinite(v_ang)) else float("nan")

            r_obj_f = quat_wxyz_to_rot(q_obj_f).as_matrix()
            r_fl_f = quat_wxyz_to_rot(q_fl_f).as_matrix()
            lift_m = float(p_obj_f[2] - t["z_settled"])
            slip_m, slip_deg = compute_relative_slip(
                t["fl_pos_close"],
                t["fl_rot_close"],
                t["obj_pos_close"],
                t["obj_rot_close"],
                p_fl_f,
                r_fl_f,
                p_obj_f,
                r_obj_f,
            )
            finger_forces = {f: float(np.median(vals)) if vals else 0.0 for f, vals in force_samples[t["idx"]].items()}
            t["result"] = score_dexhand_trial(
                t["grasp_id"],
                t["grasp_type"],
                lift_height_m=lift_m,
                slip_m=slip_m,
                slip_deg=slip_deg,
                finger_forces_n=finger_forces,
                approach_disp_m=float(t.get("approach_disp_m", 0.0)),
                max_approach_disp_m=self.max_approach_disp_m,
                target_lift_m=self.lift_height_m,
                min_lift_m=self.min_lift_m,
                max_slip_m=self.max_slip_m,
                max_slip_deg=self.max_slip_deg,
                obj_lin_vel_mps=obj_lin_vel_mps,
                obj_ang_vel_rps=obj_ang_vel_rps,
            )

        # Optional visual verification snapshot: save best passed (or highest-lift) trial per asset across all chunks
        if cam is not None and self.save_vis_dir is not None:
            by_asset: dict[str, list[dict[str, Any]]] = {}
            for t in trials:
                if t["result"] is not None:
                    by_asset.setdefault(t["asset_id"], []).append(t)
            for aid, tlist in by_asset.items():
                best = max(tlist, key=lambda x: _vis_snapshot_key(x["result"]))
                best_key = _vis_snapshot_key(best["result"])
                if best_vis_scores is not None:
                    prev_key = best_vis_scores.get(aid)
                    if prev_key is not None and best_key <= prev_key:
                        continue
                    best_vis_scores[aid] = best_key
                safe_name = aid.replace("/", "_") or f"asset_{best['idx']}"
                self._save_snapshot(cam, best, best["result"], self.save_vis_dir / f"{safe_name}.png")

        return [t["result"] for t in trials]

    def verify_multi_assets(
        self,
        items: list[tuple[str, Path, list[dict[str, Any]]]],
    ) -> list[list[DexHandVerificationResult]]:
        """Verify multiple `(asset_id, asset_path, grasps)` items chunked by `self.max_parallel`."""
        if not items:
            return []
        self._ensure_app()
        flat_specs: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
        counts: list[int] = []
        for asset_id, asset_path, grasps in items:
            usd_file, meta = self._resolve_asset_usd_and_meta(Path(asset_path))
            counts.append(len(grasps))
            for g in grasps:
                flat_specs.append((asset_id, usd_file, meta, g))

        flat_results: list[DexHandVerificationResult] = []
        best_vis_scores: dict[str, tuple[int, float, float]] = {}
        for start in range(0, len(flat_specs), self.max_parallel):
            chunk = flat_specs[start : start + self.max_parallel]
            flat_results.extend(self._run_trials_chunk(chunk, best_vis_scores=best_vis_scores))

        grouped: list[list[DexHandVerificationResult]] = []
        cursor = 0
        for cnt in counts:
            grouped.append(flat_results[cursor : cursor + cnt])
            cursor += cnt
        return grouped

    def verify_batch(
        self,
        asset_path: Path | None,
        grasps: list[dict[str, Any]],
        *,
        asset_id: str = "",
    ) -> list[DexHandVerificationResult]:
        if not grasps:
            return []
        if asset_path is None:
            raise ValueError("IsaacSimDexHandVerifier requires --asset / asset_path pointing to the object USD or dir.")
        return self.verify_multi_assets([(asset_id, Path(asset_path), grasps)])[0]


# ---------------------------------------------------------------------------
# Idempotent JSON writeback & CLI
# ---------------------------------------------------------------------------


def apply_dexhand_verification(
    affordance: dict[str, Any],
    verifier: DexHandGraspVerifier,
    *,
    asset_path: Path | None = None,
    gripper_type: str = SHARPA_GRIPPER_KEY,
    drop_failed: bool = False,
    sim_weight: float = 0.7,
) -> dict[str, Any]:
    """Run `verifier` on `affordance['grasps'][gripper_type]` and return idempotently updated dict."""
    updated = json.loads(json.dumps(affordance))
    grasps_map = updated.setdefault("grasps", {})
    grasps = list(grasps_map.get(gripper_type, []))
    if not grasps:
        return updated

    results = verifier.verify_batch(asset_path, grasps, asset_id=str(updated.get("asset_id", "")))
    by_id = {r.grasp_id: r for r in results}

    out_list: list[dict[str, Any]] = []
    for g in grasps:
        res = by_id.get(str(g.get("id")))
        if res is None:
            out_list.append(g)
            continue
        geom_score = float(g.get("geom_score", g.get("score", 0.0)))
        sim_score = float(max(0.0, min(1.0, res.sim_score)))
        g["geom_score"] = round(geom_score, 4)
        g["verified"] = bool(res.passed)
        g["sim_score"] = round(sim_score, 4)
        g["sim_metrics"] = res.to_metrics()
        if res.passed:
            g["score"] = round((1.0 - sim_weight) * geom_score + sim_weight * sim_score, 4)
        else:
            g["score"] = round(geom_score, 4)
        if not drop_failed or res.passed:
            out_list.append(g)

    out_list.sort(key=lambda item: (bool(item.get("verified")), float(item.get("score", 0.0))), reverse=True)
    grasps_map[gripper_type] = out_list
    return updated


def verify_dexhand_affordance_file(
    affordance_path: Path,
    verifier: DexHandGraspVerifier,
    *,
    asset_path: Path | None = None,
    output_path: Path | None = None,
    gripper_type: str = SHARPA_GRIPPER_KEY,
    drop_failed: bool = False,
) -> dict[str, Any]:
    """Load an affordance.json file, verify `grasps[gripper_type]`, and write back."""
    data = json.loads(Path(affordance_path).read_text(encoding="utf-8"))
    updated = apply_dexhand_verification(
        data,
        verifier,
        asset_path=asset_path,
        gripper_type=gripper_type,
        drop_failed=drop_failed,
    )
    dest = Path(output_path) if output_path is not None else Path(affordance_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Sharpa Wave Stage 1 affordance grasps in Isaac Sim")
    parser.add_argument("affordance", nargs="?", type=Path, default=None, help="Path to affordance.json")
    parser.add_argument("--manifest", type=Path, default=None, help="Batch manifest.json from Stage 1")
    parser.add_argument("--asset", type=Path, default=None, help="Path to object USD/dir (single-file mode)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path (defaults to in-place)")
    parser.add_argument(
        "--gripper", default=SHARPA_GRIPPER_KEY, help="Gripper key inside grasps (default: sharpa_wave)"
    )
    parser.add_argument("--drop-failed", action="store_true", help="Remove failed grasps from output")
    parser.add_argument(
        "--skip-verified",
        action="store_true",
        help="Skip manifest entries whose grasps already contain `sim_metrics` (resume support)",
    )
    parser.add_argument("--offset", type=int, default=0, help="Start offset into manifest")
    parser.add_argument("--limit", type=int, default=0, help="Max manifest items to verify (0 = all)")
    parser.add_argument("--max-parallel", type=int, default=28, help="Max concurrent hand trials per SimulationContext")
    parser.add_argument("--device", default="cuda:0", help="Isaac Sim physics device (default: cuda:0)")
    parser.add_argument("--save-vis-dir", type=Path, default=None, help="Directory to save visual verification PNGs")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Run CPU dry-run verifier")
    mode.add_argument("--isaac", action="store_true", help="Run Isaac Sim verifier (GPU Slurm)")
    args = parser.parse_args(argv)

    if args.affordance is None and args.manifest is None:
        parser.error("Provide either `affordance` path or `--manifest` path.")

    verifier: DexHandGraspVerifier = (
        DryRunDexHandVerifier()
        if args.dry_run
        else IsaacSimDexHandVerifier(
            device=args.device,
            max_parallel=args.max_parallel,
            save_vis_dir=args.save_vis_dir,
        )
    )

    exit_code = 0
    try:
        if args.manifest is not None:
            all_entries = json.loads(args.manifest.read_text(encoding="utf-8"))
            start_i = max(0, int(args.offset))
            end_i = start_i + args.limit if args.limit > 0 else len(all_entries)
            entries = all_entries[start_i:end_i]
            valid_items: list[tuple[dict[str, Any], Path, Path, dict[str, Any], list[dict[str, Any]]]] = []
            for entry in entries:
                if entry.get("status") not in ("ok", "skipped"):
                    continue
                aff_str = entry.get("affordance_path")
                asset_str = entry.get("asset_dir")
                if not aff_str or not asset_str:
                    continue
                aff_path = Path(aff_str)
                asset_dir = Path(asset_str)
                if not aff_path.is_file() or not asset_dir.exists():
                    continue
                data = json.loads(aff_path.read_text(encoding="utf-8"))
                grasps = list((data.get("grasps") or {}).get(args.gripper, []))
                if not grasps:
                    continue
                if args.skip_verified and all(isinstance(g.get("sim_metrics"), dict) for g in grasps):
                    continue
                valid_items.append((entry, aff_path, asset_dir, data, grasps))

            if isinstance(verifier, IsaacSimDexHandVerifier) and valid_items:
                grouped_results = verifier.verify_multi_assets(
                    [(str(e.get("asset_id", "")), asset_dir, grasps) for e, _, asset_dir, _, grasps in valid_items]
                )
            else:
                grouped_results = [
                    verifier.verify_batch(asset_dir, grasps, asset_id=str(e.get("asset_id", "")))
                    for e, _, asset_dir, _, grasps in valid_items
                ]

            total_verified_assets = 0
            total_passed = 0
            total_grasps = 0
            for (entry, aff_path, _asset_dir, data, _grasps), results in zip(valid_items, grouped_results):

                class _StaticVerifier:
                    def verify_batch(self, _a, _gl, *, asset_id=""):
                        return results

                updated = apply_dexhand_verification(
                    data,
                    _StaticVerifier(),
                    gripper_type=args.gripper,
                    drop_failed=args.drop_failed,
                )
                aff_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                glist = updated.get("grasps", {}).get(args.gripper, [])
                passed = sum(1 for g in glist if g.get("verified"))
                entry["sim_verified_sharpa"] = passed
                entry["sim_total_sharpa"] = len(glist)
                total_passed += passed
                total_grasps += len(glist)
                if passed > 0:
                    total_verified_assets += 1
                reasons: dict[str, int] = {}
                for g in glist:
                    r = str((g.get("sim_metrics") or {}).get("reason", ""))
                    if r:
                        reasons[r] = reasons.get(r, 0) + 1
                print(
                    f"[verify-dexhand] {entry.get('asset_id')}: {passed}/{len(glist)} passed reasons={reasons}",
                    flush=True,
                )

            args.manifest.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                f"[verify-dexhand batch] assets_with_verified={total_verified_assets}/{len(valid_items)} "
                f"grasps_passed={total_passed}/{total_grasps}",
                flush=True,
            )
        else:
            res = verify_dexhand_affordance_file(
                args.affordance,
                verifier,
                asset_path=args.asset,
                output_path=args.output,
                gripper_type=args.gripper,
                drop_failed=args.drop_failed,
            )
            glist = res.get("grasps", {}).get(args.gripper, [])
            passed = sum(1 for g in glist if g.get("verified"))
            print(f"Verified {args.affordance}: {passed}/{len(glist)} passed for {args.gripper}", flush=True)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        exit_code = 1
    finally:
        if isinstance(verifier, IsaacSimDexHandVerifier) and verifier._app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
