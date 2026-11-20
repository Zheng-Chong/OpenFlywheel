"""CPU unit tests for table horizontal resize (room_overrides), cup RigidObject, and G1-D finger/wrist contact."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _has_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_PXR = _has_spec("pxr")


def _install_isaac_stubs_if_needed() -> None:
    """Install lightweight CPU stubs for `isaacsim` / `isaaclab` / `omni` when running outside SimulationApp."""
    if _has_spec("omni.kit.app"):
        return

    def _ensure_mod(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub not in sys.modules:
                m = types.ModuleType(sub)
                m.__path__ = []  # type: ignore[attr-defined]
                sys.modules[sub] = m
                if i > 1:
                    setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], m)
        return sys.modules[name]

    if "torch" not in sys.modules and not _has_spec("torch"):
        torch_mod = _ensure_mod("torch")
        torch_mod.Tensor = np.ndarray  # type: ignore[attr-defined]
        torch_mod.float32 = np.float32  # type: ignore[attr-defined]
        torch_mod.tensor = lambda x, dtype=None, device=None: np.asarray(x, dtype=dtype or np.float32)  # type: ignore[attr-defined]
        torch_mod.device = lambda d="cpu": types.SimpleNamespace(type=str(d))  # type: ignore[attr-defined]

    if "omegaconf" not in sys.modules and not _has_spec("omegaconf"):
        oc = _ensure_mod("omegaconf")
        oc.DictConfig = dict  # type: ignore[attr-defined]

    # Stub stage holder for CPU tests that exercise RigidObject.__init__ on an in-memory pxr.Usd.Stage.
    stage_holder: dict[str, object] = {"stage": None}

    if not HAS_PXR:
        pxr_mod = _ensure_mod("pxr")
        for sub in ("Gf", "Pcp", "Sdf", "Usd", "UsdGeom", "UsdPhysics", "UsdShade", "Vt"):
            setattr(pxr_mod, sub, types.SimpleNamespace())

    omni_usd = _ensure_mod("omni.usd")
    omni_usd._stage_holder = stage_holder  # type: ignore[attr-defined]

    prims_utils = _ensure_mod("isaacsim.core.utils.prims")
    stage_utils = _ensure_mod("isaacsim.core.utils.stage")
    string_utils = _ensure_mod("isaacsim.core.utils.string")
    mat_mod = _ensure_mod("isaacsim.core.api.materials")
    phys_mat_mod = _ensure_mod("isaacsim.core.api.materials.physics_material")
    prims_mod = _ensure_mod("isaacsim.core.prims")
    sim_mgr_mod = _ensure_mod("isaacsim.core.simulation_manager")

    def _get_stage():
        return stage_holder["stage"]

    def _get_prim_at_path(prim_path: str):
        st = _get_stage()
        return st.GetPrimAtPath(prim_path) if st is not None else None

    def _is_prim_path_valid(prim_path: str) -> bool:
        p = _get_prim_at_path(prim_path)
        return bool(p and p.IsValid())

    def _add_reference_to_stage(usd_path: str, prim_path: str):
        st = _get_stage()
        prim = st.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_path)
        return prim

    def _find_unique_string_name(initial: str, is_unique_fn):
        cand = initial
        idx = 0
        while not is_unique_fn(cand):
            idx += 1
            cand = f"{initial}_{idx}"
        return cand

    class _StubPhysicsMaterial:
        def __init__(
            self,
            prim_path: str,
            static_friction: float = 0.5,
            dynamic_friction: float = 0.5,
            restitution: float = 0.0,
        ):
            self.prim_path = prim_path
            self.static_friction = static_friction
            self.dynamic_friction = dynamic_friction
            self.restitution = restitution
            st = _get_stage()
            if st is not None and HAS_PXR:
                from pxr import UsdPhysics, UsdShade

                mat = UsdShade.Material.Define(st, prim_path)
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr().Set(float(static_friction))
                api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
                api.CreateRestitutionAttr().Set(float(restitution))
                self.material = mat

    class _StubPreviewSurface:
        def __init__(self, prim_path: str, color=None, roughness: float = 0.5, metallic: float = 0.0):
            st = _get_stage()
            if st is not None and HAS_PXR:
                from pxr import UsdShade

                self.material = UsdShade.Material.Define(st, prim_path)
            else:
                self.material = None

    class _StubSingleGeometryPrim:
        def __init__(self, prim_path: str, name: str = "", **kwargs):
            self._geom_prim_path = prim_path

        def apply_physics_material(self, physics_material, weaker_than_descendants: bool = False):
            st = _get_stage()
            if st is not None and HAS_PXR and getattr(physics_material, "material", None) is not None:
                from pxr import UsdShade

                prim = st.GetPrimAtPath(self._geom_prim_path)
                strength = (
                    UsdShade.Tokens.weakerThanDescendants
                    if weaker_than_descendants
                    else UsdShade.Tokens.strongerThanDescendants
                )
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_material.material, strength, "physics")

    class _StubSingleRigidPrim:
        def __init__(self, prim_path: str, name: str = "", translation=None, orientation=None, scale=None, **kwargs):
            self._rigid_prim_path = prim_path

        def set_mass(self, mass: float) -> None:
            st = _get_stage()
            if st is not None and HAS_PXR:
                from pxr import UsdPhysics

                prim = st.GetPrimAtPath(self._rigid_prim_path)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass))

        def set_default_state(self, linear_velocity=None, angular_velocity=None) -> None:
            pass

    prims_utils.get_prim_at_path = _get_prim_at_path  # type: ignore[attr-defined]
    prims_utils.is_prim_path_valid = _is_prim_path_valid  # type: ignore[attr-defined]
    stage_utils.get_current_stage = _get_stage  # type: ignore[attr-defined]
    stage_utils.add_reference_to_stage = _add_reference_to_stage  # type: ignore[attr-defined]
    string_utils.find_unique_string_name = _find_unique_string_name  # type: ignore[attr-defined]
    mat_mod.PreviewSurface = _StubPreviewSurface  # type: ignore[attr-defined]
    phys_mat_mod.PhysicsMaterial = _StubPhysicsMaterial  # type: ignore[attr-defined]
    prims_mod.SingleGeometryPrim = _StubSingleGeometryPrim  # type: ignore[attr-defined]
    prims_mod.SingleRigidPrim = _StubSingleRigidPrim  # type: ignore[attr-defined]
    sim_mgr_mod.SimulationManager = types.SimpleNamespace(get_physics_sim_view=lambda: None)  # type: ignore[attr-defined]

    if "env.scene_manager.layout_manager" not in sys.modules:
        import env.scene_manager  # keep real PEP 420 namespace __path__ intact

        lm_mod = types.ModuleType("env.scene_manager.layout_manager")
        lm_mod.LayoutManager = types.SimpleNamespace(  # type: ignore[attr-defined]
            decode_inst_name=staticmethod(lambda inst_name: (inst_name, "0"))
        )
        sys.modules["env.scene_manager.layout_manager"] = lm_mod
        env.scene_manager.layout_manager = lm_mod  # type: ignore[attr-defined]

    # Minimal IsaacLab stubs for importing env.robot_manager.robot_config.g1d on CPU.
    art_mod = _ensure_mod("isaaclab.assets.articulation")
    act_mod = _ensure_mod("isaaclab.actuators")
    sim_mod = _ensure_mod("isaaclab.sim")
    ff_mod = _ensure_mod("isaaclab.sim.spawners.from_files")

    @dataclass
    class _CfgBase:
        def replace(self, **changes):
            return replace(self, **changes)

    @dataclass
    class _ImplicitActuatorCfg(_CfgBase):
        joint_names_expr: list[str] = field(default_factory=list)
        stiffness: float = 0.0
        damping: float = 0.0
        effort_limit_sim: float | None = None
        velocity_limit_sim: float | None = None
        armature: float | None = None

    @dataclass
    class _PreviewSurfaceCfg(_CfgBase):
        diffuse_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
        metallic: float = 0.0
        roughness: float = 0.5
        func: object = lambda path, cfg: None

    @dataclass
    class _PDGainsCfg(_CfgBase):
        stiffness: float = 0.0
        damping: float = 0.0

    class _UrdfConverterCfg:
        @dataclass
        class JointDriveCfg(_CfgBase):
            gains: object = None
            PDGainsCfg = _PDGainsCfg

    @dataclass
    class _RigidBodyPropertiesCfg(_CfgBase):
        disable_gravity: bool = False
        max_depenetration_velocity: float = 5.0

    @dataclass
    class _ArticulationRootPropertiesCfg(_CfgBase):
        enabled_self_collisions: bool = True
        solver_position_iteration_count: int = 8
        solver_velocity_iteration_count: int = 0
        fix_root_link: bool = True

    @dataclass
    class _UrdfFileCfg(_CfgBase):
        func: object = None
        asset_path: str = ""
        usd_dir: str = ""
        fix_base: bool = True
        activate_contact_sensors: bool = True
        merge_fixed_joints: bool = False
        joint_drive: object = None
        rigid_props: object = None
        articulation_props: object = None

    class _ArticulationCfg(_CfgBase):
        @dataclass
        class InitialStateCfg(_CfgBase):
            pos: tuple = (0.0, 0.0, 0.0)
            rot: tuple = (1.0, 0.0, 0.0, 0.0)
            joint_pos: dict = field(default_factory=dict)

        def __init__(self, spawn=None, init_state=None, actuators=None, prim_path: str = ""):
            self.spawn = spawn
            self.init_state = init_state
            self.actuators = actuators or {}
            self.prim_path = prim_path

    def _clone_decorator(fn):
        fn.__wrapped__ = fn
        return fn

    act_mod.ImplicitActuatorCfg = _ImplicitActuatorCfg  # type: ignore[attr-defined]
    art_mod.ArticulationCfg = _ArticulationCfg  # type: ignore[attr-defined]
    sim_mod.PreviewSurfaceCfg = _PreviewSurfaceCfg  # type: ignore[attr-defined]
    sim_mod.UrdfConverterCfg = _UrdfConverterCfg  # type: ignore[attr-defined]
    sim_mod.RigidBodyPropertiesCfg = _RigidBodyPropertiesCfg  # type: ignore[attr-defined]
    sim_mod.ArticulationRootPropertiesCfg = _ArticulationRootPropertiesCfg  # type: ignore[attr-defined]
    sim_mod.UrdfFileCfg = _UrdfFileCfg  # type: ignore[attr-defined]
    sim_mod.clone = _clone_decorator  # type: ignore[attr-defined]
    sim_mod.bind_visual_material = lambda *a, **kw: None  # type: ignore[attr-defined]
    ff_mod.from_files = types.SimpleNamespace()  # type: ignore[attr-defined]


_install_isaac_stubs_if_needed()

from env.robot_manager.robot_config import g1d, g1d_spec  # noqa: E402
from env.scene_manager.objects import rigid, room_overrides  # noqa: E402


class PureConfigResolutionTests(unittest.TestCase):
    """Pure-Python tests that run in every CPU environment even when pxr is not installed."""

    def test_rigid_material_defaults_and_friction_fallback(self):
        # Default backward-compatibility: 0.6 static / 1.5 dynamic / 0.0 restitution, no PhysxMaterialAPI extras.
        default_cfg = rigid.resolve_rigid_material_config({})
        self.assertAlmostEqual(default_cfg["static_friction"], 0.6)
        self.assertAlmostEqual(default_cfg["dynamic_friction"], 1.5)
        self.assertAlmostEqual(default_cfg["restitution"], 0.0)
        self.assertIsNone(default_cfg["friction_combine"])
        self.assertIsNone(default_cfg["compliant_stiffness"])
        self.assertIsNone(default_cfg["compliant_damping"])

        # Layout passing `"friction": 0.5` without static/dynamic_friction sets both to 0.5 (fixes ignored key bug).
        fb_cfg = rigid.resolve_rigid_material_config({"friction": 0.5})
        self.assertAlmostEqual(fb_cfg["static_friction"], 0.5)
        self.assertAlmostEqual(fb_cfg["dynamic_friction"], 0.5)

        # Explicit static_friction / dynamic_friction wins over fallback `friction`.
        explicit_cfg = rigid.resolve_rigid_material_config(
            {
                "friction": 0.5,
                "static_friction": 0.55,
                "dynamic_friction": 0.45,
                "restitution": 0.02,
                "friction_combine": "average",
                "compliant_stiffness": 5000.0,
                "compliant_damping": 50.0,
            }
        )
        self.assertAlmostEqual(explicit_cfg["static_friction"], 0.55)
        self.assertAlmostEqual(explicit_cfg["dynamic_friction"], 0.45)
        self.assertAlmostEqual(explicit_cfg["restitution"], 0.02)
        self.assertEqual(explicit_cfg["friction_combine"], "average")
        self.assertAlmostEqual(explicit_cfg["compliant_stiffness"], 5000.0)
        self.assertAlmostEqual(explicit_cfg["compliant_damping"], 50.0)

    def test_room_table_resize_resolution_and_eval_cfg_stack_fallback(self):
        # Direct Room keys
        sz, sc, off = room_overrides._resolve_table_resize({"table_size": [1.45, 0.74]})
        self.assertEqual(sz, (1.45, 0.74))
        self.assertIsNone(sc)
        self.assertEqual(off, (0.0, 0.0))

        # Nested under table_lift
        sz2, sc2, off2 = room_overrides._resolve_table_resize(
            {"table_lift": {"dz": 0.065, "table_xy_scale": [1.2, 1.1], "table_xy_offset": [0.01, -0.02]}}
        )
        self.assertIsNone(sz2)
        self.assertEqual(sc2, (1.2, 1.1))
        self.assertEqual(off2, (0.01, -0.02))

        # Fallback from EvalEnv / TaskEnv `eval_cfg["room_overrides"]` on the call stack
        class _FakeEvalEnv:
            def __init__(self):
                self.eval_cfg = {"room_overrides": {"table_size": [1.45, 0.74]}}

            def query(self):
                return room_overrides._resolve_table_resize({"table_lift": {"dz": 0.065}})

        sz3, sc3, off3 = _FakeEvalEnv().query()
        self.assertEqual(sz3, (1.45, 0.74))
        self.assertIsNone(sc3)
        self.assertEqual(off3, (0.0, 0.0))

    def test_g1d_default_actuators_unchanged_and_wrist_overrides_disjoint(self):
        default_groups = g1d_spec.actuators(mobile=False)
        self.assertEqual(set(default_groups), {"lift", "waist", "arm", "gripper"})
        self.assertEqual(
            default_groups["arm"].joint_names_expr,
            [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
        )
        self.assertEqual(default_groups["arm"].stiffness, 4400.0)
        self.assertEqual(default_groups["arm"].damping, 40.0)
        self.assertEqual(default_groups["arm"].effort_limit_sim, 100.0)

        # Wrist scalar overrides split arm and wrist into strictly disjoint groups.
        split_groups = g1d_spec.actuators(
            mobile=False,
            cfg={"wrist_stiffness": 240.0, "wrist_damping": 12.0, "wrist_effort_limit": 20.0},
        )
        self.assertEqual(set(split_groups), {"lift", "waist", "arm", "wrist", "gripper"})
        self.assertEqual(split_groups["arm"].joint_names_expr, [".*_shoulder_.*", ".*_elbow_joint"])
        self.assertEqual(split_groups["arm"].stiffness, 4400.0)
        self.assertEqual(split_groups["wrist"].joint_names_expr, [".*_wrist_.*"])
        self.assertEqual(split_groups["wrist"].stiffness, 240.0)
        self.assertEqual(split_groups["wrist"].damping, 12.0)
        self.assertEqual(split_groups["wrist"].effort_limit_sim, 20.0)

        # Regex override dicts partition all 14 arm joints with zero overlap across groups.
        regex_groups = g1d_spec.actuators(
            mobile=False,
            cfg={
                "arm_stiffness_overrides": {".*_elbow_joint": 1800.0, ".*_wrist_.*": 200.0},
                "arm_damping_overrides": {".*_wrist_.*": 10.0},
                "arm_effort_limit_overrides": {".*_wrist_.*": 18.0},
            },
        )
        matched_per_joint = {j: 0 for j in g1d_spec.ARM_JOINT_NAMES}
        resolved_kp = {}
        for gname, grp in regex_groups.items():
            if gname in ("lift", "waist", "gripper"):
                continue
            for j in g1d_spec.ARM_JOINT_NAMES:
                if any(re.fullmatch(p, j) for p in grp.joint_names_expr):
                    matched_per_joint[j] += 1
                    resolved_kp[j] = grp.stiffness
        self.assertTrue(all(cnt == 1 for cnt in matched_per_joint.values()), matched_per_joint)
        self.assertEqual(resolved_kp["left_shoulder_pitch_joint"], 4400.0)
        self.assertEqual(resolved_kp["right_elbow_joint"], 1800.0)
        self.assertEqual(resolved_kp["left_wrist_pitch_joint"], 200.0)

    def test_g1d_robot_manager_frame_forwarding_preserves_spawn_func(self):
        from unittest.mock import patch

        # Simulate RobotManager._attach_scene_cfg calling g1d.get_robot_config() with no arguments.
        def _attach_scene_cfg(robot_name: str, robots, idx: int, cfg: dict):
            return g1d.get_robot_config()

        yml_cfg = {
            "robot_name": "g1d",
            "wrist_stiffness": 300.0,
            "wrist_damping": 15.0,
            "wrist_effort_limit": 25.0,
            "finger_contact": {
                "static_friction": 0.9,
                "dynamic_friction": 0.8,
                "restitution": 0.0,
                "torsional_patch_radius": 0.008,
                "min_torsional_patch_radius": 0.002,
                "max_depenetration_velocity": 0.5,
            },
        }
        with patch.object(g1d_spec, "filtered_pairs", return_value=[]):
            art_cfg = _attach_scene_cfg("g1d", (), 0, yml_cfg)
            self.assertIs(art_cfg.spawn.func, g1d._spawn_g1d)
            self.assertIn("wrist", art_cfg.actuators)
            self.assertEqual(art_cfg.actuators["wrist"].stiffness, 300.0)
            fc = g1d._lookup_spawn_finger_contact("/World/envs/env_0/robot0", art_cfg.spawn)
            self.assertIsNotNone(fc)
            self.assertAlmostEqual(fc["static_friction"], 0.9)
            self.assertAlmostEqual(fc["torsional_patch_radius"], 0.008)

            # Calling get_robot_config() without finger_contact clears stale per-index state.
            clean_cfg = _attach_scene_cfg("g1d", (), 0, {"robot_name": "g1d"})
            self.assertNotIn("wrist", clean_cfg.actuators)
            self.assertIsNone(g1d._lookup_spawn_finger_contact("/World/envs/env_0/robot0", clean_cfg.spawn))


@unittest.skipUnless(HAS_PXR, "requires pxr (USD Python bindings)")
class UsdStagePhysicsTests(unittest.TestCase):
    """End-to-end USD stage tests verifying real_scene5 table geometry, RigidObject attrs, and G1-D finger contact."""

    def test_real_scene5_table_size_145x074_preserves_z_0815_and_normals(self):
        from pxr import Gf, Usd, UsdGeom

        scene_usda = REPO_ROOT / "real2sim/scenes/real_scene5_shenzhen_lab/usd/scene.usda"
        parts_json = REPO_ROOT / "real2sim/scenes/real_scene5_shenzhen_lab/usd/parts.json"
        self.assertTrue(scene_usda.is_file(), scene_usda)
        parts = json.loads(parts_json.read_text())

        stage = Usd.Stage.CreateInMemory()
        room_path = "/World/envs/env_0/Room"
        room_prim = stage.DefinePrim(room_path, "Xform")
        room_prim.GetReferences().AddReference(str(scene_usda))

        # Apply worst-case real520_sz@v2 Room placement (L08: [0.0131, -0.3862, 0.0]) + table_lift dz=0.065 + 1.45x0.74.
        xf = UsdGeom.Xformable(room_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0131, -0.3862, 0.0))

        room_cfg = {
            "table_size": [1.45, 0.74],
            "table_lift": {
                "dz": 0.065,
                "move": parts["lift"]["move"],
                "stretch": parts["lift"]["stretch"],
            },
        }
        room_overrides.apply_room_overrides(stage, room_path, room_cfg)
        # Idempotent when applied again:
        room_overrides.apply_room_overrides(stage, room_path, room_cfg)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )
        for rel in ("Table/Colliders/top", "Table/Top/slab"):
            prim = stage.GetPrimAtPath(f"{room_path}/{rel}")
            self.assertTrue(prim.IsValid(), rel)
            rng = bbox_cache.ComputeRelativeBound(prim, room_prim).ComputeAlignedRange()
            lo, hi = rng.GetMin(), rng.GetMax()
            self.assertAlmostEqual(lo[0], -0.725, places=5)
            self.assertAlmostEqual(hi[0], +0.725, places=5)
            self.assertAlmostEqual(lo[1], -0.370, places=5)
            self.assertAlmostEqual(hi[1], +0.370, places=5)
            self.assertAlmostEqual(lo[2], 0.795, places=5)
            self.assertAlmostEqual(hi[2], 0.815, places=6)

            # Verify top-face normals remain strictly (0, 0, 1) in world space.
            mesh = UsdGeom.Mesh(prim)
            world_tf = xf_mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            inv_t = world_tf.ExtractRotationMatrix().GetInverse().GetTranspose()
            for n in mesh.GetNormalsAttr().Get():
                if n[2] > 0.9:
                    wn = Gf.Vec3d(float(n[0]), float(n[1]), float(n[2])) * inv_t
                    wn /= wn.GetLength()
                    self.assertAlmostEqual(wn[0], 0.0, places=6)
                    self.assertAlmostEqual(wn[1], 0.0, places=6)
                    self.assertAlmostEqual(wn[2], 1.0, places=6)
            _ = xf_mat

        # Verify legs shifted cleanly in X without changing foot Z=[0, 0.035] or colliding with G1-D torso/AGV.
        pos_foot = stage.GetPrimAtPath(f"{room_path}/Table/Colliders/Legs_pos_foot")
        neg_foot = stage.GetPrimAtPath(f"{room_path}/Table/Colliders/Legs_neg_foot")
        pf_rng = bbox_cache.ComputeRelativeBound(pos_foot, room_prim).ComputeAlignedRange()
        nf_rng = bbox_cache.ComputeRelativeBound(neg_foot, room_prim).ComputeAlignedRange()
        self.assertAlmostEqual(pf_rng.GetMin()[2], 0.0, places=6)
        self.assertAlmostEqual(pf_rng.GetMax()[2], 0.035, places=6)
        self.assertAlmostEqual(0.5 * (pf_rng.GetMin()[0] + pf_rng.GetMax()[0]), 0.472 * (1.45 / 1.20), places=4)
        self.assertAlmostEqual(0.5 * (nf_rng.GetMin()[0] + nf_rng.GetMax()[0]), -0.472 * (1.45 / 1.20), places=4)

        # Worst overhanging cup in real520_sz@v2 (L05 target cup: p_room=[0.1588, -0.3139], r_base=0.0255) has >30 mm
        # margin (and >20 mm even with source cup r_base=0.0355).
        top_rng = bbox_cache.ComputeRelativeBound(
            stage.GetPrimAtPath(f"{room_path}/Table/Colliders/top"), room_prim
        ).ComputeAlignedRange()
        cup_x, cup_y, r_base = 0.1588, -0.3139, 0.0255
        margin_x = min((cup_x - r_base) - top_rng.GetMin()[0], top_rng.GetMax()[0] - (cup_x + r_base))
        margin_y = (cup_y - r_base) - top_rng.GetMin()[1]
        self.assertGreater(margin_x, 0.500)
        self.assertGreater(margin_y, 0.030)
        self.assertGreater((cup_y - 0.0355) - top_rng.GetMin()[1], 0.020)

        # World tabletop front Y vs G1-D torso_link (max world Y = -0.8290 at root [-0.03, -0.967, 0.1115]).
        top_world_rng = bbox_cache.ComputeWorldBound(
            stage.GetPrimAtPath(f"{room_path}/Table/Colliders/top")
        ).ComputeAlignedRange()
        self.assertGreater(top_world_rng.GetMin()[1] - (-0.8290), 0.050)

    def test_rigid_object_authors_com_depenetration_torsional_and_compliant_material(self):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        omni.usd._stage_holder["stage"] = stage  # type: ignore[attr-defined]
        try:
            cup_path = "/World/envs/env_0/cup/cup_0"
            stage.DefinePrim(cup_path, "Xform")
            for idx in range(2):
                hull = UsdGeom.Cube.Define(stage, f"{cup_path}/collision/hull_{idx}").GetPrim()
                UsdPhysics.CollisionAPI.Apply(hull)

            inst_cfg = {
                "physics": {
                    "mass": 0.024,
                    "friction": 0.5,
                    "static_friction": 0.55,
                    "center_of_mass": [0.0, 0.0, -0.018],
                    "max_depenetration_velocity": 0.5,
                    "linear_damping": 0.2,
                    "angular_damping": 0.4,
                    "torsional_patch_radius": 0.006,
                    "min_torsional_patch_radius": 0.002,
                    "compliant_stiffness": 4000.0,
                    "compliant_damping": 40.0,
                    "friction_combine": "average",
                }
            }
            obj = rigid.RigidObject(
                prim_path=cup_path,
                usd_path="",
                inst_config=inst_cfg,
                env_origin=np.zeros(3, dtype=np.float32),
                default_pos=(0.0, -0.5, 0.88),
                default_ori=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            )

            cup_prim = stage.GetPrimAtPath(cup_path)
            mass_api = UsdPhysics.MassAPI(cup_prim)
            self.assertAlmostEqual(mass_api.GetMassAttr().Get(), 0.024, places=6)
            com = mass_api.GetCenterOfMassAttr().Get()
            self.assertEqual(com, Gf.Vec3f(0.0, 0.0, -0.018))

            self.assertAlmostEqual(
                cup_prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get(), 0.5, places=6
            )
            self.assertAlmostEqual(cup_prim.GetAttribute("physxRigidBody:linearDamping").Get(), 0.2, places=6)
            self.assertAlmostEqual(cup_prim.GetAttribute("physxRigidBody:angularDamping").Get(), 0.4, places=6)

            for idx in range(2):
                hprim = stage.GetPrimAtPath(f"{cup_path}/collision/hull_{idx}")
                self.assertAlmostEqual(hprim.GetAttribute("physxCollision:torsionalPatchRadius").Get(), 0.006, places=6)
                self.assertAlmostEqual(
                    hprim.GetAttribute("physxCollision:minTorsionalPatchRadius").Get(), 0.002, places=6
                )

            mat_prim = stage.GetPrimAtPath(obj.physics_material_path)
            mat_api = UsdPhysics.MaterialAPI(mat_prim)
            self.assertAlmostEqual(mat_api.GetStaticFrictionAttr().Get(), 0.55, places=6)
            # dynamic_friction fell back to `friction=0.5` instead of 1.5!
            self.assertAlmostEqual(mat_api.GetDynamicFrictionAttr().Get(), 0.50, places=6)
            self.assertEqual(mat_prim.GetAttribute("physxMaterial:frictionCombineMode").Get(), "average")
            self.assertAlmostEqual(
                mat_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get(), 4000.0, places=5
            )
            self.assertAlmostEqual(mat_prim.GetAttribute("physxMaterial:compliantContactDamping").Get(), 40.0, places=5)
        finally:
            omni.usd._stage_holder["stage"] = None  # type: ignore[attr-defined]

    def test_g1d_apply_finger_contact_uninstances_colliders_and_binds_physics_material(self):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        # Prototype collider hierarchy matching g1d_dex1_physics.usd (instanced `collisions` on each link).
        stage.CreateClassPrim("/_ColliderProtos")
        stage.DefinePrim("/_ColliderProtos/pad_col", "Xform")
        mesh = UsdGeom.Cube.Define(stage, "/_ColliderProtos/pad_col/Link1_3/node_STL_BINARY_").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)

        robot_path = "/World/envs/env_0/robot0"
        robot_prim = stage.DefinePrim(robot_path, "Xform")
        all_links = [*g1d_spec.DEFAULT_FINGER_CONTACT_LINKS, "torso_link"]
        for lname in all_links:
            lprim = stage.DefinePrim(f"{robot_path}/{lname}", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(lprim)
            cprim = stage.DefinePrim(f"{robot_path}/{lname}/collisions", "Xform")
            cprim.GetInherits().AddInherit("/_ColliderProtos/pad_col")
            cprim.SetInstanceable(True)
            self.assertTrue(cprim.IsInstance())

        g1d.apply_finger_contact(
            robot_prim,
            {
                "static_friction": 0.9,
                "dynamic_friction": 0.8,
                "restitution": 0.0,
                "friction_combine": "average",
                "compliant_stiffness": 6000.0,
                "compliant_damping": 60.0,
                "torsional_patch_radius": 0.008,
                "min_torsional_patch_radius": 0.002,
                "max_depenetration_velocity": 0.5,
            },
        )

        # Non-finger link remains instanced and untouched.
        self.assertTrue(stage.GetPrimAtPath(f"{robot_path}/torso_link/collisions").IsInstance())

        expected_mat_path = f"{robot_path}/Looks/finger_physics_material"
        for lname in g1d_spec.DEFAULT_FINGER_CONTACT_LINKS:
            lprim = stage.GetPrimAtPath(f"{robot_path}/{lname}")
            self.assertAlmostEqual(lprim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get(), 0.5, places=6)
            cprim = lprim.GetChild("collisions")
            self.assertFalse(cprim.IsInstance(), lname)
            leaf_mesh = stage.GetPrimAtPath(f"{robot_path}/{lname}/collisions/Link1_3/node_STL_BINARY_")
            self.assertTrue(leaf_mesh.IsValid(), lname)
            bound_mat, _ = UsdShade.MaterialBindingAPI(leaf_mesh).ComputeBoundMaterial("physics")
            self.assertEqual(str(bound_mat.GetPath()), expected_mat_path)
            self.assertAlmostEqual(leaf_mesh.GetAttribute("physxCollision:torsionalPatchRadius").Get(), 0.008, places=6)
            self.assertAlmostEqual(
                leaf_mesh.GetAttribute("physxCollision:minTorsionalPatchRadius").Get(), 0.002, places=6
            )

        mat_prim = stage.GetPrimAtPath(expected_mat_path)
        mat_api = UsdPhysics.MaterialAPI(mat_prim)
        self.assertAlmostEqual(mat_api.GetStaticFrictionAttr().Get(), 0.9, places=6)
        self.assertAlmostEqual(mat_api.GetDynamicFrictionAttr().Get(), 0.8, places=6)
        self.assertAlmostEqual(mat_api.GetRestitutionAttr().Get(), 0.0, places=6)
        self.assertAlmostEqual(mat_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get(), 6000.0, places=5)
        self.assertAlmostEqual(mat_prim.GetAttribute("physxMaterial:compliantContactDamping").Get(), 60.0, places=5)


if __name__ == "__main__":
    unittest.main()
