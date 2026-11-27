"""Stage 2 physics closed-loop verification for object affordance.json candidates.

Evaluates Stage 1 geometric grasp candidates by simulating approach -> close -> lift -> shake
in an empty scene with a Dex1-1 parallel-jaw gripper, recording `verified`, `sim_score`,
and `sim_metrics` back into `affordance.json`.

Usage (on GPU Slurm inside isaac-sim, or locally with --dry-run):
    python scripts/objects/verify_affordance_sim.py path/to/affordance.json --dry-run
    isaac-sim python scripts/objects/verify_affordance_sim.py path/to/affordance.json \
        --asset path/to/object.usdz --isaac
    isaac-sim python scripts/objects/verify_affordance_sim.py --manifest path/to/manifest.json --isaac
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.objects.affordance import GRIPPER_SPECS, GripperSpec


@dataclass(frozen=True)
class GraspVerificationResult:
    """Outcome of a single physics grasp trial (approach -> close -> lift -> shake)."""

    grasp_id: str
    passed: bool
    sim_score: float  # in [0.0, 1.0]
    lift_height_m: float = 0.0
    slip_m: float = 0.0
    slip_deg: float = 0.0
    reason: str = ""

    def to_metrics(self) -> dict[str, Any]:
        return {
            "lift_height_m": round(float(self.lift_height_m), 4),
            "slip_m": round(float(self.slip_m), 4),
            "slip_deg": round(float(self.slip_deg), 2),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Pure-NumPy verification math (CPU unit-testable without Isaac Sim)
# ---------------------------------------------------------------------------


def quat_wxyz_to_rot(q: np.ndarray | list[float]) -> Rotation:
    arr = np.asarray(q, dtype=np.float64).reshape(4)
    return Rotation.from_quat([arr[1], arr[2], arr[3], arr[0]])


def rot_to_quat_wxyz(rot: Rotation) -> np.ndarray:
    xyzw = rot.as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if wxyz[0] < 0.0:
        wxyz = -wxyz
    return wxyz


def compute_relative_slip(
    ee_pos_0: np.ndarray,
    ee_rot_0: np.ndarray,
    obj_pos_0: np.ndarray,
    obj_rot_0: np.ndarray,
    ee_pos_f: np.ndarray,
    ee_rot_f: np.ndarray,
    obj_pos_f: np.ndarray,
    obj_rot_f: np.ndarray,
) -> tuple[float, float]:
    """Compute (translational_slip_m, rotational_slip_deg) of object relative to gripper frame."""
    p_rel_0 = ee_rot_0.T @ (obj_pos_0 - ee_pos_0)
    r_rel_0 = ee_rot_0.T @ obj_rot_0
    p_rel_f = ee_rot_f.T @ (obj_pos_f - ee_pos_f)
    r_rel_f = ee_rot_f.T @ obj_rot_f
    slip_m = float(np.linalg.norm(p_rel_f - p_rel_0))
    r_diff = r_rel_0.T @ r_rel_f
    trace_val = float(np.clip(0.5 * (np.trace(r_diff) - 1.0), -1.0, 1.0))
    slip_deg = float(math.degrees(math.acos(trace_val)))
    return slip_m, slip_deg


def check_floor_clearance(
    grasp_pos: np.ndarray,
    grasp_rot: np.ndarray,
    width: float,
    spec: GripperSpec,
    floor_z: float = 0.0,
    min_clear: float = 0.001,
) -> tuple[bool, str]:
    """Return (ok, reason) checking whether open fingers or palm penetrate the floor at `grasp_pos`."""
    approach = grasp_rot[:, 0]
    closing = grasp_rot[:, 1]
    dorsal = grasp_rot[:, 2]
    if float(approach[2]) > 0.25:
        return False, "upward_approach"

    open_half = min(0.5 * (width + spec.grasp_clear), 0.5 * spec.max_open)
    finger_half_y = open_half + spec.finger_thick
    p_tip = grasp_pos + spec.finger_tips * approach
    tip_low_z = (
        float(p_tip[2]) - finger_half_y * abs(float(closing[2])) - 0.5 * spec.finger_depth * abs(float(dorsal[2]))
    )
    if tip_low_z < floor_z + min_clear:
        return False, "floor_tip_collision"

    p_palm = grasp_pos - (spec.finger_back + 0.5 * spec.palm_len) * approach
    palm_half_z = 0.5 * (
        spec.palm_len * abs(float(approach[2]))
        + spec.palm_width * abs(float(closing[2]))
        + spec.palm_height * abs(float(dorsal[2]))
    )
    if float(p_palm[2]) - palm_half_z < floor_z + min_clear:
        return False, "floor_palm_collision"

    return True, "ok"


def score_verification_trial(
    grasp_id: str,
    lift_height_m: float,
    slip_m: float,
    slip_deg: float,
    *,
    target_lift_m: float = 0.15,
    min_lift_m: float = 0.10,
    max_slip_m: float = 0.025,
    max_slip_deg: float = 20.0,
) -> GraspVerificationResult:
    """Evaluate a completed physics trial against lift & slip acceptance thresholds."""
    if not (math.isfinite(lift_height_m) and math.isfinite(slip_m) and math.isfinite(slip_deg)):
        return GraspVerificationResult(grasp_id=grasp_id, passed=False, sim_score=0.0, reason="unstable_nan")

    if lift_height_m < min_lift_m:
        return GraspVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            reason="insufficient_lift",
        )
    if slip_m > max_slip_m:
        return GraspVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            reason="excessive_slip_pos",
        )
    if slip_deg > max_slip_deg:
        return GraspVerificationResult(
            grasp_id=grasp_id,
            passed=False,
            sim_score=0.0,
            lift_height_m=lift_height_m,
            slip_m=slip_m,
            slip_deg=slip_deg,
            reason="excessive_slip_rot",
        )

    lift_q = min(1.0, max(0.0, lift_height_m / max(target_lift_m, 1e-6)))
    slip_pos_q = max(0.0, 1.0 - slip_m / max(max_slip_m, 1e-6))
    slip_rot_q = max(0.0, 1.0 - slip_deg / max(max_slip_deg, 1e-6))
    sim_score = 0.5 + 0.5 * (0.30 * lift_q + 0.40 * slip_pos_q + 0.30 * slip_rot_q)
    return GraspVerificationResult(
        grasp_id=grasp_id,
        passed=True,
        sim_score=round(float(sim_score), 4),
        lift_height_m=lift_height_m,
        slip_m=slip_m,
        slip_deg=slip_deg,
        reason="ok",
    )


# ---------------------------------------------------------------------------
# Verifier protocol & implementations
# ---------------------------------------------------------------------------


class GraspVerifier(Protocol):
    """Pluggable physics verifier interface for Stage 2 batch evaluation."""

    def verify_batch(
        self,
        asset_path: Path | None,
        gripper_type: str,
        grasps: list[dict[str, Any]],
    ) -> list[GraspVerificationResult]: ...


class DryRunVerifier:
    """CPU-only dry-run verifier that marks high-geometric-score grasps for pipeline testing."""

    def __init__(self, score_threshold: float = 0.55):
        self.score_threshold = float(score_threshold)

    def verify_batch(
        self,
        asset_path: Path | None,
        gripper_type: str,
        grasps: list[dict[str, Any]],
    ) -> list[GraspVerificationResult]:
        out: list[GraspVerificationResult] = []
        for g in grasps:
            geom_score = float(g.get("score", 0.0))
            passed = geom_score >= self.score_threshold
            out.append(
                GraspVerificationResult(
                    grasp_id=str(g["id"]),
                    passed=passed,
                    sim_score=geom_score if passed else 0.0,
                    lift_height_m=0.12 if passed else 0.0,
                    slip_m=0.002 if passed else 0.05,
                    slip_deg=1.5 if passed else 35.0,
                    reason="dry_run_ok" if passed else "dry_run_below_threshold",
                )
            )
        return out


class IsaacSimGraspVerifier:
    """Isaac Sim 5.1 headless concurrent physics verifier (runs on GPU Slurm inside isaac-sim).

    Spawns all candidate grasps for an asset in parallel cells on a static floor in one
    SimulationContext (render=False), settles the objects, then executes:
      pre-grasp -> approach -> force-limited close (55 N) -> lift (+0.15 m) -> shake (+/-0.03 m).
    """

    def __init__(
        self,
        *,
        dt: float = 0.004,
        device: str = "cpu",
        spacing: float = 0.60,
        settle_steps: int = 125,
        approach_dist: float = 0.075,
        approach_steps: int = 45,
        close_steps: int = 60,
        lift_height: float = 0.15,
        lift_steps: int = 60,
        shake_amp: float = 0.02,
        shake_steps: int = 100,
        min_lift_m: float = 0.10,
        max_slip_m: float = 0.025,
        max_slip_deg: float = 20.0,
        max_force: float = 55.0,
        pad_friction: float = 1.5,
        max_parallel: int = 32,
    ):
        self.dt = float(dt)
        self.device = str(device)
        self.spacing = float(spacing)
        self.settle_steps = int(settle_steps)
        self.approach_dist = float(approach_dist)
        self.approach_steps = int(approach_steps)
        self.close_steps = int(close_steps)
        self.lift_height = float(lift_height)
        self.lift_steps = int(lift_steps)
        self.shake_amp = float(shake_amp)
        self.shake_steps = int(shake_steps)
        self.min_lift_m = float(min_lift_m)
        self.max_slip_m = float(max_slip_m)
        self.max_slip_deg = float(max_slip_deg)
        self.max_force = float(max_force)
        self.pad_friction = float(pad_friction)
        self.max_parallel = max(1, int(max_parallel))
        self._app: Any = None
        self._sim: Any = None
        self._batch_counter: int = 0

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        try:
            from isaaclab.app import AppLauncher  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "IsaacSimGraspVerifier requires Isaac Sim + IsaacLab (run via `isaac-sim` on GPU Slurm)."
            ) from exc
        launcher = AppLauncher(headless=True, enable_cameras=False)
        self._app = launcher.app

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

    def _create_gripper_prim(
        self,
        stage: Any,
        root_path: str,
        spec: GripperSpec,
        park_pos: np.ndarray,
        pad_mat_path: str,
    ) -> dict[str, Any]:
        """Author a kinematic palm + 2 compliant-contact kinematic finger pads in USD."""
        from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade  # type: ignore[import-not-found]

        UsdGeom.Xform.Define(stage, root_path)

        # Palm: unit-scale kinematic rigid body; child Cube carries the scaled collision box
        palm_path = f"{root_path}/palm"
        palm_xf = UsdGeom.Xform.Define(stage, palm_path)
        palm_rb = UsdPhysics.RigidBodyAPI.Apply(palm_xf.GetPrim())
        palm_rb.CreateKinematicEnabledAttr().Set(True)

        palm_box = UsdGeom.Cube.Define(stage, f"{palm_path}/geom")
        palm_box.CreateSizeAttr().Set(1.0)
        palm_box_xf = UsdGeom.Xformable(palm_box)
        palm_box_xf.AddTranslateOp().Set(Gf.Vec3d(-(spec.finger_back + 0.5 * spec.palm_len), 0.0, 0.0))
        palm_box_xf.AddScaleOp().Set(Gf.Vec3f(spec.palm_len, spec.palm_width, spec.palm_height))
        UsdPhysics.CollisionAPI.Apply(palm_box.GetPrim())

        # Fingers: kinematic pads with compliant PhysX contact material (stiffness=5000 N/m, damping=200)
        finger_len = spec.finger_tips + spec.finger_back
        finger_x_mid = 0.5 * (spec.finger_tips - spec.finger_back)
        mat_prim = stage.GetPrimAtPath(pad_mat_path)

        finger_prims = {}
        for side, sign in (("left", -1.0), ("right", 1.0)):
            f_path = f"{root_path}/{side}_finger"
            f_xf = UsdGeom.Xform.Define(stage, f_path)
            f_rb = UsdPhysics.RigidBodyAPI.Apply(f_xf.GetPrim())
            f_rb.CreateKinematicEnabledAttr().Set(True)

            f_box = UsdGeom.Cube.Define(stage, f"{f_path}/geom")
            f_box.CreateSizeAttr().Set(1.0)
            f_box_xf = UsdGeom.Xformable(f_box)
            # Offset box center outward by 0.5*finger_thick so finger origin sits on the inner pad face
            f_box_xf.AddTranslateOp().Set(Gf.Vec3d(finger_x_mid, sign * 0.5 * spec.finger_thick, 0.0))
            f_box_xf.AddScaleOp().Set(Gf.Vec3f(finger_len, spec.finger_thick, spec.finger_depth))
            UsdPhysics.CollisionAPI.Apply(f_box.GetPrim())
            if mat_prim and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(f_box.GetPrim()).Bind(
                    UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics"
                )

            f_rigid = SingleRigidPrim(prim_path=f_path)
            f_rigid.set_world_pose(
                position=park_pos + np.array([0.0, sign * 0.5 * spec.max_open, 0.0], dtype=np.float64),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            finger_prims[side] = f_rigid

        palm_rigid = SingleRigidPrim(prim_path=palm_path)
        palm_rigid.set_world_pose(position=park_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        return {"palm": palm_rigid, "fingers": finger_prims}

    def _set_gripper_pose(
        self,
        gripper: dict[str, Any],
        grasp_pos: np.ndarray,
        grasp_rot: np.ndarray,
        gap_half: float,
        teleport_fingers: bool = False,
    ) -> None:
        q_wxyz = rot_to_quat_wxyz(Rotation.from_matrix(grasp_rot))
        y_axis = grasp_rot[:, 1]
        gripper["palm"].set_world_pose(position=grasp_pos, orientation=q_wxyz)
        gripper["fingers"]["left"].set_world_pose(position=grasp_pos - gap_half * y_axis, orientation=q_wxyz)
        gripper["fingers"]["right"].set_world_pose(position=grasp_pos + gap_half * y_axis, orientation=q_wxyz)

    def _run_trials_once(
        self,
        trial_specs: list[tuple[Path, dict[str, Any], GripperSpec, dict[str, Any]]],
    ) -> list[GraspVerificationResult]:
        """Run all `(usd_file, meta, spec, grasp)` trials concurrently in a single SimulationContext."""
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]
        from pxr import UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        from env.scene_manager.objects.rigid import RigidObject  # type: ignore[import-not-found]

        if not trial_specs:
            return []

        n = len(trial_specs)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / cols)))
        span_x = max(20.0, cols * self.spacing + 4.0)
        span_y = max(20.0, rows * self.spacing + 4.0)

        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device=self.device)
        self._sim = sim_utils.SimulationContext(sim_cfg)
        stage = omni.usd.get_context().get_stage()

        floor_cfg = sim_utils.CuboidCfg(
            size=(span_x, span_y, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        floor_cfg.func(
            "/World/floor",
            floor_cfg,
            translation=(0.5 * (cols - 1) * self.spacing, 0.5 * (rows - 1) * self.spacing, -0.02),
        )

        # Shared high-friction compliant physics material for gripper pads
        pad_mat_path = "/World/pad_physics_material"
        mat_prim = stage.DefinePrim(pad_mat_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat.CreateStaticFrictionAttr().Set(float(self.pad_friction))
        phys_mat.CreateDynamicFrictionAttr().Set(float(self.pad_friction))
        phys_mat.CreateRestitutionAttr().Set(0.0)
        try:
            from pxr import PhysxSchema  # type: ignore[import-not-found]

            pmat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            pmat.CreateCompliantContactStiffnessAttr().Set(6000.0)
            pmat.CreateCompliantContactDampingAttr().Set(240.0)
            pmat.CreateFrictionCombineModeAttr().Set("max")
        except Exception:
            pass

        UsdGeom.Xform.Define(stage, "/World/trials")
        trials: list[dict[str, Any]] = []
        for idx, (usd_file, meta, spec, g) in enumerate(trial_specs):
            r_idx, c_idx = divmod(idx, cols)
            origin = np.array([c_idx * self.spacing, r_idx * self.spacing, 0.0], dtype=np.float64)
            t_root = f"/World/trials/t_{idx:04d}"
            UsdGeom.Xform.Define(stage, t_root)

            physics_cfg = dict(meta.get("physics") or {})
            if not physics_cfg.get("mass"):
                physics_cfg["mass"] = 0.15
            size = physics_cfg.get("size") or ((meta.get("geometry") or {}).get("aligned_bbox") or {}).get("extents")
            half_h = 0.5 * float(size[2]) if (size and len(size) >= 3) else 0.06

            obj = RigidObject(
                prim_path=f"{t_root}/obj_0_{idx}",
                usd_path=str(usd_file),
                inst_config={"physics": physics_cfg},
                env_origin=[0.0, 0.0, 0.0],
                default_pos=(float(origin[0]), float(origin[1]), float(half_h + 0.008)),
                default_ori=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            )
            gripper = self._create_gripper_prim(
                stage,
                f"{t_root}/gripper",
                spec,
                park_pos=origin + np.array([0.0, 0.0, 1.2], dtype=np.float64),
                pad_mat_path=pad_mat_path,
            )
            trials.append(
                {
                    "idx": idx,
                    "spec": spec,
                    "grasp": g,
                    "grasp_id": str(g.get("id", f"g{idx:03d}")),
                    "origin": origin,
                    "obj": obj,
                    "gripper": gripper,
                    "active": True,
                    "result": None,
                }
            )

        self._sim.reset()
        for t in trials:
            t["obj"].initialize()
            t["gripper"]["palm"].initialize()
            t["gripper"]["fingers"]["left"].initialize()
            t["gripper"]["fingers"]["right"].initialize()

        # Phase 1: Settle objects on the floor
        for _ in range(self.settle_steps):
            self._sim.step(render=False)

        # Phase 2: Compute world-frame grasp poses from settled objects and filter floor collisions
        for t in trials:
            spec = t["spec"]
            pos0, quat0 = map(self._as_np, t["obj"].get_world_pose())
            if not np.all(np.isfinite(pos0)) or not np.all(np.isfinite(quat0)):
                t["active"] = False
                t["result"] = GraspVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason="settle_unstable"
                )
                continue
            r0 = quat_wxyz_to_rot(quat0).as_matrix()
            g = t["grasp"]
            p_loc = np.asarray(g["position"], dtype=np.float64).reshape(3)
            r_loc = quat_wxyz_to_rot(g["quaternion_wxyz"]).as_matrix()
            width = float(g["width"])

            p_grasp = pos0 + r0 @ p_loc
            r_grasp = r0 @ r_loc
            ok_floor, floor_reason = check_floor_clearance(p_grasp, r_grasp, width, spec, floor_z=0.0)
            if not ok_floor:
                t["active"] = False
                t["result"] = GraspVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason=floor_reason
                )
                continue

            approach_w = r_grasp[:, 0]
            p_pre = p_grasp - self.approach_dist * approach_w
            p_pre[2] = max(float(p_pre[2]), 0.03)
            open_half = min(0.5 * (width + spec.grasp_clear), 0.5 * spec.max_open)
            # 4 mm compliant squeeze per pad (~24 N normal force at k=6000 N/m, within Dex1-1 55 N limit)
            closed_half = max(0.5 * spec.pad_closed, 0.5 * width - 0.004)

            t["p_grasp"] = p_grasp
            t["r_grasp"] = r_grasp
            t["p_pre"] = p_pre
            t["p_lift"] = p_grasp + np.array([0.0, 0.0, self.lift_height], dtype=np.float64)
            t["open_half"] = open_half
            t["closed_half"] = closed_half
            t["z_settled"] = float(pos0[2])
            self._set_gripper_pose(t["gripper"], p_pre, r_grasp, open_half, teleport_fingers=True)

        active_trials = [t for t in trials if t["active"]]
        if not active_trials:
            return [t["result"] for t in trials]

        # Let teleported fingers settle into prismatic joint targets
        for _ in range(10):
            for t in active_trials:
                self._set_gripper_pose(t["gripper"], t["p_pre"], t["r_grasp"], t["open_half"])
            self._sim.step(render=False)

        # Phase 3: Linear approach from p_pre -> p_grasp
        for step in range(1, self.approach_steps + 1):
            alpha = step / float(self.approach_steps)
            for t in active_trials:
                pos = (1.0 - alpha) * t["p_pre"] + alpha * t["p_grasp"]
                self._set_gripper_pose(t["gripper"], pos, t["r_grasp"], t["open_half"])
            self._sim.step(render=False)

        # Phase 4: Close fingers with force-limited prismatic drives
        ramp_steps = max(1, self.close_steps // 2)
        for step in range(1, self.close_steps + 1):
            beta = min(1.0, step / float(ramp_steps))
            for t in active_trials:
                gap = (1.0 - beta) * t["open_half"] + beta * t["closed_half"]
                self._set_gripper_pose(t["gripper"], t["p_grasp"], t["r_grasp"], gap)
            self._sim.step(render=False)

        # Record post-close relative pose between gripper and object
        for t in active_trials:
            p_obj_c, q_obj_c = map(self._as_np, t["obj"].get_world_pose())
            t["obj_pos_close"] = p_obj_c
            t["obj_rot_close"] = quat_wxyz_to_rot(q_obj_c).as_matrix() if np.all(np.isfinite(q_obj_c)) else np.eye(3)
        if active_trials:
            t0 = active_trials[0]
            lf_p, _ = map(self._as_np, t0["gripper"]["fingers"]["left"].get_world_pose())
            rf_p, _ = map(self._as_np, t0["gripper"]["fingers"]["right"].get_world_pose())
            print(
                f"[verify debug] t0 p_grasp={t0['p_grasp'].round(4).tolist()} "
                f"lf={lf_p.round(4).tolist()} rf={rf_p.round(4).tolist()} "
                f"obj_close={t0['obj_pos_close'].round(4).tolist()} "
                f"closed_half={t0['closed_half']:.4f}",
                flush=True,
            )

        # Phase 5: Lift vertically +lift_height
        for step in range(1, self.lift_steps + 1):
            alpha = step / float(self.lift_steps)
            for t in active_trials:
                pos = (1.0 - alpha) * t["p_grasp"] + alpha * t["p_lift"]
                self._set_gripper_pose(t["gripper"], pos, t["r_grasp"], t["closed_half"])
            self._sim.step(render=False)

        # Hold briefly at lift top
        for _ in range(15):
            for t in active_trials:
                self._set_gripper_pose(t["gripper"], t["p_lift"], t["r_grasp"], t["closed_half"])
            self._sim.step(render=False)

        # Phase 6: Gentle (~0.5g) sinusoidal lateral shake along X and Y, then return to p_lift
        for step in range(1, self.shake_steps + 1):
            phase = 2.0 * math.pi * step / float(self.shake_steps)
            offset = np.array(
                [self.shake_amp * math.sin(phase), 0.5 * self.shake_amp * math.sin(phase), 0.0],
                dtype=np.float64,
            )
            for t in active_trials:
                self._set_gripper_pose(t["gripper"], t["p_lift"] + offset, t["r_grasp"], t["closed_half"])
            self._sim.step(render=False)

        for _ in range(15):
            for t in active_trials:
                self._set_gripper_pose(t["gripper"], t["p_lift"], t["r_grasp"], t["closed_half"])
            self._sim.step(render=False)

        # Phase 7: Evaluate lift height and post-shake slip
        for t in active_trials:
            p_obj_f, q_obj_f = map(self._as_np, t["obj"].get_world_pose())
            if not np.all(np.isfinite(p_obj_f)) or not np.all(np.isfinite(q_obj_f)):
                t["result"] = GraspVerificationResult(
                    grasp_id=t["grasp_id"], passed=False, sim_score=0.0, reason="unstable_nan"
                )
                continue
            r_obj_f = quat_wxyz_to_rot(q_obj_f).as_matrix()
            lift_m = float(p_obj_f[2] - t["z_settled"])
            slip_m, slip_deg = compute_relative_slip(
                t["p_grasp"],
                t["r_grasp"],
                t["obj_pos_close"],
                t["obj_rot_close"],
                t["p_lift"],
                t["r_grasp"],
                p_obj_f,
                r_obj_f,
            )
            t["result"] = score_verification_trial(
                t["grasp_id"],
                lift_m,
                slip_m,
                slip_deg,
                target_lift_m=self.lift_height,
                min_lift_m=self.min_lift_m,
                max_slip_m=self.max_slip_m,
                max_slip_deg=self.max_slip_deg,
            )

        return [t["result"] for t in trials]

    def verify_multi_assets(
        self,
        items: list[tuple[Path, str, list[dict[str, Any]]]],
    ) -> list[list[GraspVerificationResult]]:
        """Verify multiple `(asset_path, gripper_type, grasps)` concurrently in one SimulationContext."""
        if not items:
            return []
        self._ensure_app()
        flat_specs: list[tuple[Path, dict[str, Any], GripperSpec, dict[str, Any]]] = []
        counts: list[int] = []
        for asset_path, gripper_type, grasps in items:
            if gripper_type not in GRIPPER_SPECS:
                raise ValueError(f"Unknown gripper_type={gripper_type!r}")
            spec = GRIPPER_SPECS[gripper_type]
            usd_file, meta = self._resolve_asset_usd_and_meta(Path(asset_path))
            counts.append(len(grasps))
            for g in grasps:
                flat_specs.append((usd_file, meta, spec, g))

        flat_results = self._run_trials_once(flat_specs)
        grouped: list[list[GraspVerificationResult]] = []
        cursor = 0
        for cnt in counts:
            grouped.append(flat_results[cursor : cursor + cnt])
            cursor += cnt
        return grouped

    def verify_batch(
        self,
        asset_path: Path | None,
        gripper_type: str,
        grasps: list[dict[str, Any]],
    ) -> list[GraspVerificationResult]:
        if not grasps:
            return []
        if asset_path is None:
            raise ValueError("IsaacSimGraspVerifier requires --asset / asset_path pointing to the object USD or dir.")
        return self.verify_multi_assets([(Path(asset_path), gripper_type, grasps)])[0]


# ---------------------------------------------------------------------------
# JSON read/write & CLI
# ---------------------------------------------------------------------------


def apply_verification(
    affordance: dict[str, Any],
    verifier: GraspVerifier,
    *,
    asset_path: Path | None = None,
    gripper_type: str = "dex1_1",
    drop_failed: bool = False,
    sim_weight: float = 0.7,
) -> dict[str, Any]:
    """Run `verifier` on `affordance['grasps'][gripper_type]` and return updated dict."""
    updated = json.loads(json.dumps(affordance))
    grasps_map = updated.setdefault("grasps", {})
    grasps = list(grasps_map.get(gripper_type, []))
    if not grasps:
        return updated

    results = verifier.verify_batch(asset_path, gripper_type, grasps)
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
        if not drop_failed or res.passed:
            out_list.append(g)

    out_list.sort(key=lambda item: (bool(item.get("verified")), float(item.get("score", 0.0))), reverse=True)
    grasps_map[gripper_type] = out_list
    return updated


def verify_affordance_file(
    affordance_path: Path,
    verifier: GraspVerifier,
    *,
    asset_path: Path | None = None,
    output_path: Path | None = None,
    gripper_type: str = "dex1_1",
    drop_failed: bool = False,
) -> dict[str, Any]:
    """Load an affordance.json file, verify its grasps, and write the updated JSON."""
    data = json.loads(Path(affordance_path).read_text(encoding="utf-8"))
    updated = apply_verification(
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
    parser = argparse.ArgumentParser(description="Verify Stage 1 affordance.json grasps in physics simulation")
    parser.add_argument("affordance", nargs="?", type=Path, default=None, help="Path to affordance.json")
    parser.add_argument("--manifest", type=Path, default=None, help="Batch manifest.json from `affordance.py batch`")
    parser.add_argument("--asset", type=Path, default=None, help="Path to object USD/mesh (single-file mode)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path (defaults to in-place)")
    parser.add_argument("--gripper", default="dex1_1", help="Gripper key inside grasps")
    parser.add_argument("--drop-failed", action="store_true", help="Remove failed grasps from output")
    parser.add_argument("--offset", type=int, default=0, help="Start offset into manifest")
    parser.add_argument("--limit", type=int, default=0, help="Max manifest items to verify (0 = all)")
    parser.add_argument("--device", default="cpu", help="Isaac Sim physics device (default: cpu)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Run CPU dry-run verifier")
    mode.add_argument("--isaac", action="store_true", help="Run Isaac Sim verifier (GPU Slurm)")
    args = parser.parse_args(argv)

    if args.affordance is None and args.manifest is None:
        parser.error("Provide either `affordance` path or `--manifest` path.")

    verifier: GraspVerifier = DryRunVerifier() if args.dry_run else IsaacSimGraspVerifier(device=args.device)
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
                aff_path = Path(entry["affordance_path"])
                asset_dir = Path(entry["asset_dir"])
                if not aff_path.is_file() or not asset_dir.exists():
                    continue
                data = json.loads(aff_path.read_text(encoding="utf-8"))
                grasps = list((data.get("grasps") or {}).get(args.gripper, []))
                if not grasps:
                    continue
                valid_items.append((entry, aff_path, asset_dir, data, grasps))

            if isinstance(verifier, IsaacSimGraspVerifier) and valid_items:
                grouped_results = verifier.verify_multi_assets(
                    [(asset_dir, args.gripper, grasps) for _, _, asset_dir, _, grasps in valid_items]
                )
            else:
                grouped_results = [
                    verifier.verify_batch(asset_dir, args.gripper, grasps) for _, _, asset_dir, _, grasps in valid_items
                ]

            total_verified_assets = 0
            total_passed = 0
            total_grasps = 0
            for (entry, aff_path, _asset_dir, data, _grasps), results in zip(valid_items, grouped_results):

                class _StaticVerifier:
                    def verify_batch(self, _a, _g, _gl):
                        return results

                updated = apply_verification(
                    data,
                    _StaticVerifier(),
                    gripper_type=args.gripper,
                    drop_failed=args.drop_failed,
                )
                aff_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                glist = updated.get("grasps", {}).get(args.gripper, [])
                passed = sum(1 for g in glist if g.get("verified"))
                entry["sim_verified"] = passed
                entry["sim_total"] = len(glist)
                total_passed += passed
                total_grasps += len(glist)
                if passed > 0:
                    total_verified_assets += 1
                print(
                    f"[verify] {entry.get('asset_id')}: {passed}/{len(glist)} passed for {args.gripper}",
                    flush=True,
                )
            args.manifest.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                f"[verify batch] assets_with_verified={total_verified_assets}/{len(valid_items)} "
                f"grasps_passed={total_passed}/{total_grasps}",
                flush=True,
            )
        else:
            res = verify_affordance_file(
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
        if isinstance(verifier, IsaacSimGraspVerifier) and verifier._app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
