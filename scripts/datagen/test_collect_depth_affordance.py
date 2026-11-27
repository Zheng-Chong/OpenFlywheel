"""CPU tests for Depth & Affordance wiring in obs['vision'] and scripts/datagen/collect.py."""

from __future__ import annotations

import importlib
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("env.seeding", ModuleType("env.seeding"))
sys.modules["env.seeding"].seed_everywhere = getattr(sys.modules["env.seeding"], "seed_everywhere", lambda *a: None)

from env.observation_manager.obs_manager import ObsManager  # noqa: E402
from env.observation_manager.vision_utils import parse_camera_selection, requested_depth_cameras  # noqa: E402
from scripts.datagen import collect, export  # noqa: E402
from scripts.objects.affordance_vlm import look_at_cam_to_world, pinhole_intrinsics  # noqa: E402


def _encode_jpeg(rgb):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _patch_xpolicylab():
    pkg = ModuleType("XPolicyLab")
    utils_mod = ModuleType("XPolicyLab.utils")
    proc_mod = ModuleType("XPolicyLab.utils.process_data")
    proc_mod.encode_image_bit = _encode_jpeg
    return patch.dict(
        sys.modules,
        {"XPolicyLab": pkg, "XPolicyLab.utils": utils_mod, "XPolicyLab.utils.process_data": proc_mod},
    )


def _sample_affordance(asset_id="Rigid/mug/00001") -> dict:
    return {
        "schema_version": 1,
        "asset_id": asset_id,
        "mesh_hash": "sha256:0123456789",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.05], "max": [0.06, 0.04, 0.05], "extents": [0.10, 0.08, 0.10]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_handle",
                    "position": [0.05, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.018,
                    "part_tag": "handle",
                    "score": 0.90,
                    "sim_score": 0.95,
                    "verified": True,
                },
                {
                    "id": "g_body",
                    "position": [0.0, 0.03, 0.01],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.065,
                    "part_tag": "body",
                    "score": 0.72,
                    "sim_score": 0.82,
                    "verified": True,
                },
            ]
        },
    }


def _dex1_arm(side):
    return SimpleNamespace(
        type="target",
        robot_type="arm",
        asset_name="g1d",
        robot_name="g1d",
        arm_name=f"{side}_arm",
        gripper_name=f"{side}_ee",
        ee_type="gripper",
        gripper_scale=[-0.02, 0.0245],
        gripper_move={"sign": -1.0, "base": f"{side}_finger_1", "mimic": [f"{side}_finger_2", 1.0, 0.0]},
        gripper_joints_name=[f"{side}_finger_1", f"{side}_finger_2"],
        arm_joints_name=[f"{side}_arm_j{i}" for i in range(7)],
        ee_link_name=f"{side}_hand",
        arm_joint_indices=range(7),
    )


class _Tensor(np.ndarray):
    def cpu(self):
        return np.asarray(self)


def _articulation(side):
    jac = np.zeros((1, 1, 6, 7))
    jac[0, 0, :, :6] = np.eye(6)
    return SimpleNamespace(
        body_names=["base", f"{side}_hand"],
        root_physx_view=SimpleNamespace(get_jacobians=lambda: jac.view(_Tensor)),
        data=SimpleNamespace(soft_joint_pos_limits=np.tile([-2.0, 2.0], (1, 7, 1)), joint_pos=np.zeros((1, 7))),
    )


class _StubCaptureManager:
    """Mimics TiledCaptureManager/MujocoCaptureManager annotator outputs for ObsManager tests."""

    def __init__(self, camera_names, *, h=120, w=160, include_depth=True, depth_annotator="distance_to_image_plane"):
        self.camera_names = camera_names
        self.h = h
        self.w = w
        self.include_depth = include_depth
        self.depth_annotator = depth_annotator
        self.t = 0
        self.depth_fn = None

    def step(self, env_ids=None, cam_ids=None):
        env_ids = list(range(len(self.camera_names))) if env_ids is None else list(env_ids)
        num_cams = len(self.camera_names[0])
        out = []
        for cam_id in range(num_cams):
            rgb_list = []
            depth_list = []
            for eid in env_ids:
                rgba = np.full((self.h, self.w, 4), 40 + (self.t * 10) % 150, dtype=np.uint8)
                rgba[..., 3] = 255
                rgb_list.append({"data": rgba, "info": {}})
                if self.include_depth:
                    if self.depth_fn is not None:
                        d = self.depth_fn(eid, cam_id, self.t)
                    else:
                        d = np.full((self.h, self.w, 1), 0.75 + 0.05 * cam_id, dtype=np.float32)
                    depth_list.append({"data": d, "info": {}})
            cam_dict = {"rgb": rgb_list}
            if self.include_depth:
                cam_dict[self.depth_annotator] = depth_list
            out.append(cam_dict)
        return out


class _StubEnvWithObsManager:
    """Minimal env exercising real ObsManager + collect.record_episode end-to-end."""

    def __init__(
        self,
        layout_file: Path,
        *,
        obs_vision_cfg: dict,
        aff_path: Path | None = None,
        env_origin=(0.0, 0.0, 0.0),
        include_depth_stream=True,
        depth_annotator="distance_to_image_plane",
    ):
        robots = [_dex1_arm("right"), _dex1_arm("left")]
        self.num_envs = 1
        self.task_name = "g1d_pick_anything"
        self.config_name = "g1d"
        self.dt = 1 / 300
        self.eval_seed = 77
        self.current_env_seed_map = {0: 0}
        self.success, self.end_flag, self.task_success = [True], [False], [True]
        self.end_on_success, self.step_lim, self.t = True, 100, 0
        self.target_label = "target"
        self.selected_grasp_world = None
        self.robot_manager = SimpleNamespace(
            robot_list=robots,
            robot_key=[_articulation(r.arm_name.split("_")[0]) for r in robots],
            target_arm_nums=2,
            process_name=lambda n: n + "_joint_state",
            get_joint=lambda robot, env_idx_list: {e: np.zeros(7, dtype=np.float32) for e in env_idx_list},
            get_gripper_force=lambda robot, env_idx_list: None,
            get_real_endpose=lambda robot, env_idx_list: {e: np.zeros(7, dtype=np.float32) for e in env_idx_list},
            restore_name=lambda key: key[: -len("_joint_state")],
            get_robot_by_gripper_name=lambda name: robots[0] if "right" in name else robots[1],
            control_manager=SimpleNamespace(
                prev_control=[
                    {
                        "left_ee_joint_state": {"position": [0.0, 0.0]},
                        "right_ee_joint_state": {"position": [0.0, 0.0]},
                    }
                ]
            ),
        )
        entry = {
            "label": "target",
            "category": "mug",
            "category_idx": 1,
            "default_pos": [0.45, 0.0, 0.85],
            "default_ori": [1.0, 0.0, 0.0, 0.0],
            "scale": 1.0,
        }
        if aff_path is not None:
            entry["affordance_path"] = str(aff_path)
        self._scene_info = {"Rigid": {"mug": [entry]}}
        self.seed_manager = SimpleNamespace(
            seed_info={0: {"scene_layout": str(layout_file)}},
            get_seed_scene_info=lambda lid: self._scene_info,
        )
        origin_arr = np.asarray(env_origin, dtype=np.float64)
        self.scene_manager = SimpleNamespace(
            env_origins=origin_arr[None, :],
            layout_manager=SimpleNamespace(get_instance_pose=self._get_instance_pose),
            config={"env0": self._scene_info},
        )

        c2w_cv = look_at_cam_to_world(eye=[0.0, 0.0, 1.35], target=[0.45, 0.0, 0.85])
        c2w_usd = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])
        c2w_usd[:3, 3] += origin_arr  # world pose includes env_origin, while get_instance_pose is env-relative
        K = pinhole_intrinsics(160, 120, fov_deg=55.0)
        cam_names = [["cam_head", "cam_left_wrist", "cam_right_wrist"]]
        self.camera_manager = SimpleNamespace(
            num_cams=3,
            camera_names=cam_names,
            get_camera_intrinsics=lambda cid, eid=0: K,
            get_camera_extrinsics=lambda cid, eid=0: c2w_usd.copy(),
        )
        self.capture_manager = _StubCaptureManager(
            cam_names,
            h=120,
            w=160,
            include_depth=include_depth_stream,
            depth_annotator=depth_annotator,
        )
        self.obs_manager = ObsManager(
            obs_config={
                "collect_freq": 30,
                "robot": {"joint_states": True, "world_ee_state": False, "ee_state": "command"},
                "vision": obs_vision_cfg,
            },
            num_envs=1,
            dt=1 / 300,
            task_name=self.task_name,
            description_cfg={},
        )
        self.obs_manager.initialize(self)
        self.obs_manager.instruction = ["pick up the mug"]

    def layout_entries(self, key):
        return [self._scene_info.get(key)]

    def _get_instance_pose(self, env_idx, label):
        return np.array([0.45, 0.015 * self.t, 0.85 + 0.01 * self.t], dtype=np.float64), np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

    def get_obs(self):
        return self.obs_manager.get_obs(env_idx_list=[0])[0]

    def take_action(self, action):
        self.t += 1
        self.capture_manager.t = self.t


def _fake_plugin(n_steps=4, *, set_selected_grasp=False):
    class Episode:
        reject = None

        def __init__(self, env):
            self.env = env
            self.target_label = "target"
            self.selected_grasp_world = np.array([0.50, 0.0, 0.85], dtype=np.float64) if set_selected_grasp else None

        def actions(self):
            for _ in range(n_steps):
                yield {
                    "left_arm_joint_state": np.zeros(7, dtype=np.float32),
                    "left_ee_joint_state": np.array([0.5], dtype=np.float32),
                    "right_arm_joint_state": np.zeros(7, dtype=np.float32),
                    "right_ee_joint_state": np.array([0.5], dtype=np.float32),
                }

        def step(self):
            return True

        def summary(self):
            return {}

        def privileged(self):
            return {}

    return SimpleNamespace(
        __name__="scripts.datagen.plugins.g1d_pick_anything_affordance",
        TASK="g1d_pick_anything",
        ACCEPTANCE="v1",
        HOLD=2,
        TAIL=0,
        STEP_LIM=100,
        RECORD_TO_END=True,
        Episode=Episode,
    )


def test_camera_selection_parsing_and_depth_helper():
    assert parse_camera_selection(False) is None
    assert parse_camera_selection(True) is True
    assert parse_camera_selection("1") is True
    assert parse_camera_selection("false") is None
    assert parse_camera_selection("cam_head, cam_left_wrist") == {"cam_head", "cam_left_wrist"}
    assert parse_camera_selection(["cam_head"]) == {"cam_head"}
    assert parse_camera_selection({"cam_head"}) == {"cam_head"}
    assert parse_camera_selection(["all"]) is True
    assert parse_camera_selection("cam_head,*") is True
    assert parse_camera_selection({"enabled": False, "cameras": ["cam_head"]}) is None
    assert parse_camera_selection({"enabled": True, "cameras": ["cam_head"]}) == {"cam_head"}

    with patch.dict(os.environ, {"OPENFLYWHEEL_COLLECT_DEPTH": "cam_head"}, clear=False):
        assert requested_depth_cameras(None) == {"cam_head"}
        assert requested_depth_cameras(["cam_left_wrist"]) == {"cam_head", "cam_left_wrist"}
        assert requested_depth_cameras(True) is True


def test_capture_managers_enable_depth_only_when_requested():
    try:
        from omegaconf import OmegaConf
    except ImportError:

        class _FakeDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

        class _FakeOmega:
            @staticmethod
            def create(obj):
                return _FakeDict(obj) if isinstance(obj, dict) else obj

            @staticmethod
            def to_container(obj, resolve=True):
                return dict(obj) if isinstance(obj, dict) else obj

        OmegaConf = _FakeOmega

    cam_cfg = OmegaConf.create(
        {
            "annotator": {
                "common": {"enabled": True, "rgb_capture": {"type": "rgb", "device": "cpu"}},
            }
        }
    )
    cam_mgr = SimpleNamespace(
        cameras=[
            [
                SimpleNamespace(prim_path="/cam0", _resolution=(160, 120)),
                SimpleNamespace(prim_path="/cam1", _resolution=(160, 120)),
            ]
        ],
        camera_names=[["cam_head", "cam_left_wrist"]],
        num_cams=2,
    )

    # 1. TiledCaptureManager without Isaac Sim installed: stub minimal kit modules
    fake_isaac_cam = ModuleType("isaacsim.sensors.camera")
    fake_isaac_cam.Camera = object
    fake_rep = ModuleType("omni.replicator.core")
    fake_ann = ModuleType("omni.replicator.core.scripts.annotators")
    fake_ann.Annotator = object
    fake_rl = ModuleType("env.environment.isaac.isaac_rl_env")
    fake_rl.IsaacRLEnv = object
    fake_cv = ModuleType("env.camera_manager.capture.camera_view")
    fake_cv.ANNOTATOR_SPEC = {
        "rgb": {"name": "rgba", "channels": 4, "dtype": "u8"},
        "distance_to_image_plane": {"name": "distance_to_image_plane", "channels": 1, "dtype": "f32"},
    }
    fake_cv.CameraView = lambda paths, camera_resolution, output_annotators: SimpleNamespace(
        _render_product=f"rp:{output_annotators}"
    )
    fake_wp = ModuleType("warp")
    fake_wp.zeros = lambda shape, dtype, device: np.zeros(shape, dtype=np.float32)
    fake_carb = ModuleType("carb")
    fake_carb.settings = SimpleNamespace(get_settings=lambda: SimpleNamespace(get=lambda k: 0))

    fake_oc = ModuleType("omegaconf")
    fake_oc.DictConfig = dict
    fake_oc.OmegaConf = OmegaConf
    fake_cm = ModuleType("env.camera_manager.camera_manager")
    fake_cm.CameraManager = object

    stubs = {
        "omegaconf": fake_oc,
        "env.camera_manager.camera_manager": fake_cm,
        "env.camera_manager.camera_pose": ModuleType("env.camera_manager.camera_pose"),
        "env.seeding": ModuleType("env.seeding"),
        "isaacsim": ModuleType("isaacsim"),
        "isaacsim.sensors": ModuleType("isaacsim.sensors"),
        "isaacsim.sensors.camera": fake_isaac_cam,
        "omni": ModuleType("omni"),
        "omni.replicator": ModuleType("omni.replicator"),
        "omni.replicator.core": fake_rep,
        "omni.replicator.core.scripts": ModuleType("omni.replicator.core.scripts"),
        "omni.replicator.core.scripts.annotators": fake_ann,
        "env.environment.isaac.isaac_rl_env": fake_rl,
        "env.camera_manager.capture.camera_view": fake_cv,
        "warp": fake_wp,
        "carb": fake_carb,
        "torch": ModuleType("torch"),
    }
    stubs["torch"].device = object
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("env.camera_manager.capture.tiled_capture_manager", None)
        tcm_mod = importlib.import_module("env.camera_manager.capture.tiled_capture_manager")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENFLYWHEEL_COLLECT_DEPTH", None)
            mgr_off = tcm_mod.TiledCaptureManager(1, cam_cfg, cam_mgr, "cpu")
            mgr_off.initialize(None)
            mgr_off.init_cameras()
            assert mgr_off.annotator_type == [["rgb"], ["rgb"]]

            mgr_cfg = tcm_mod.TiledCaptureManager(1, cam_cfg, cam_mgr, "cpu")
            mgr_cfg._requested_depth_cameras = ["cam_head"]
            mgr_cfg.initialize(None)
            mgr_cfg.init_cameras()
            assert mgr_cfg.annotator_type == [["rgb", "distance_to_image_plane"], ["rgb"]]

        with patch.dict(os.environ, {"OPENFLYWHEEL_COLLECT_DEPTH": "1"}):
            mgr_env = tcm_mod.TiledCaptureManager(1, cam_cfg, cam_mgr, "cpu")
            mgr_env.initialize(None)
            mgr_env.init_cameras()
            assert mgr_env.annotator_type == [["rgb", "distance_to_image_plane"], ["rgb", "distance_to_image_plane"]]

    # 2. MujocoCaptureManager
    try:
        from env.backends.mujoco.cameras import MujocoCaptureManager
    except ImportError:
        fake_mj = ModuleType("mujoco")
        cp_stub = stubs["env.camera_manager.camera_pose"]
        for attr in (
            "REAL_MAP",
            "intrinsics",
            "layout_offset",
            "pose_matrix",
            "sample_camera_pose",
            "split_camera_config",
        ):
            setattr(cp_stub, attr, lambda *a, **kw: None)
        stubs["env.seeding"].seed_everywhere = lambda *a: None
        with patch.dict(sys.modules, {**stubs, "mujoco": fake_mj}):
            sys.modules.pop("env.backends.mujoco.cameras", None)
            MujocoCaptureManager = importlib.import_module("env.backends.mujoco.cameras").MujocoCaptureManager
    if MujocoCaptureManager is not None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENFLYWHEEL_COLLECT_DEPTH", None)
            m_off = MujocoCaptureManager(1, cam_cfg, cam_mgr, "cpu")
            m_off.initialize(None)
            m_off.init_cameras()
            assert m_off.annotator_type == [["rgb"], ["rgb"]]

        with patch.dict(os.environ, {"OPENFLYWHEEL_COLLECT_DEPTH": "cam_left_wrist"}):
            m_env = MujocoCaptureManager(1, cam_cfg, cam_mgr, "cpu")
            m_env.initialize(None)
            m_env.init_cameras()
            assert m_env.annotator_type == [["rgb"], ["rgb", "distance_to_image_plane"]]


def test_obs_manager_vision_depth_and_affordance_with_env_origin_and_occlusion(tmp_path: Path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")
    aff_path = tmp_path / "affordance.json"
    aff_path.write_text(json.dumps(_sample_affordance()), encoding="utf-8")

    # 1. Default off: no depth, no affordance
    with patch.dict(os.environ, {}, clear=False):
        for k in (
            "OPENFLYWHEEL_COLLECT_DEPTH",
            "OPENFLYWHEEL_COLLECT_AFFORDANCE",
            "OPENFLYWHEEL_COLLECT_CAMERA_MATRICES",
        ):
            os.environ.pop(k, None)
        env_off = _StubEnvWithObsManager(
            layout_file,
            obs_vision_cfg={"approximate_depth": False, "depth": False, "affordance": False},
            aff_path=aff_path,
        )
        obs_off = env_off.get_obs()
        assert set(obs_off["vision"]["cam_head"]) == {"color", "shape"}

    # 2. Enable depth (for cam_head only) + affordance (for cam_head) via env vars, with non-zero env_origin
    with patch.dict(
        os.environ,
        {
            "OPENFLYWHEEL_COLLECT_DEPTH": "cam_head",
            "OPENFLYWHEEL_COLLECT_AFFORDANCE": "cam_head",
            "OPENFLYWHEEL_COLLECT_CAMERA_MATRICES": "1",
        },
        clear=False,
    ):
        env_on = _StubEnvWithObsManager(
            layout_file,
            obs_vision_cfg={"approximate_depth": False, "depth": False},
            aff_path=aff_path,
            env_origin=(5.0, -3.0, 0.0),
        )
        obs0 = env_on.get_obs()
        cam_h = obs0["vision"]["cam_head"]
        cam_l = obs0["vision"]["cam_left_wrist"]
        assert cam_h["depth"].shape == (120, 160) and cam_h["depth"].dtype == np.float32
        assert "depth" not in cam_l
        assert cam_h["intrinsic_matrix"].shape == (3, 3)
        assert cam_h["extrinsic_matrix"].shape == (4, 4)
        assert "affordance" in cam_h and "affordance" not in cam_l
        aff0 = cam_h["affordance"]
        assert aff0["heatmap"].shape == (48, 64) and aff0["heatmap"].dtype == np.float32
        assert aff0["heatmap_uint8"].shape == (48, 64) and aff0["heatmap_uint8"].dtype == np.uint8
        assert aff0["object_bbox_xyxy_norm"] is not None
        assert aff0["affordance_bbox_xyxy_norm"] is not None
        assert len(aff0["keypoints"]) == 2  # both grasps visible at depth=0.75 m (> 0.67 m cam-to-obj dist)

        # Move object and verify obs["vision"]["cam_head"]["affordance"] tracks motion
        env_on.take_action({})
        obs1 = env_on.get_obs()
        assert obs1["vision"]["cam_head"]["affordance"]["object_bbox_xyxy_norm"] != aff0["object_bbox_xyxy_norm"]

        # Place an occluding wall at depth 0.10 m in front of the camera -> depth occlusion filters all keypoints
        env_on.capture_manager.depth_fn = lambda eid, cid, t: np.full((120, 160, 1), 0.10, dtype=np.float32)
        obs_occ = env_on.get_obs()
        assert obs_occ["vision"]["cam_head"]["affordance"]["keypoints"] == []

    # 3. Also handles annotator named "depth" + approximate_depth with NaN/inf safely
    env_approx = _StubEnvWithObsManager(
        layout_file,
        obs_vision_cfg={"approximate_depth": True, "depth": True},
        aff_path=aff_path,
        depth_annotator="depth",
    )
    env_approx.capture_manager.depth_fn = lambda eid, cid, t: np.array(
        [[[np.nan], [np.inf]], [[0.5], [1.2]]], dtype=np.float32
    )
    env_approx.capture_manager.h = 2
    env_approx.capture_manager.w = 2
    obs_a = env_approx.get_obs()
    assert obs_a["vision"]["cam_head"]["depth"].shape == (2, 2)
    assert obs_a["vision"]["cam_head"]["approximate_depth"].dtype == np.uint16
    assert int(obs_a["vision"]["cam_head"]["approximate_depth"][1, 0]) == 500


def test_collect_end_to_end_consumes_obs_vision_depth_and_affordance(tmp_path: Path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")
    aff_path = tmp_path / "affordance.json"
    aff_path.write_text(json.dumps(_sample_affordance("Rigid/mug/00001")), encoding="utf-8")

    out_dir = tmp_path / "seed0"
    out_dir.mkdir()

    env = _StubEnvWithObsManager(
        layout_file,
        obs_vision_cfg={
            "approximate_depth": False,
            "depth": ["cam_head", "cam_left_wrist"],
            "intrinsic_matrix": True,
            "extrinsic_matrix": True,
            "affordance": {"enabled": True, "cameras": ["cam_head"], "heatmap_hw": [24, 32], "top_k": 8},
        },
        aff_path=aff_path,
    )
    # Note: cfg leaves record_depth/record_affordance_2d unset (auto-detect from obs["vision"]),
    # and requests 48x64 heatmap in collect.py to verify resize from ObsManager's 24x32!
    cfg = {
        "env_cfg": "g1d",
        "fps": 30,
        "trajectory": True,
        "early_abort": False,
        "affordance_2d_heatmap_hw": [48, 64],
        "affordance_2d_top_k": 4,
    }
    with _patch_xpolicylab():
        collect.record_episode(env, _fake_plugin(n_steps=4, set_selected_grasp=True), out_dir, cfg)

    h5_path = out_dir / "episode_000000.hdf5"
    sidecar_path = out_dir / "episode_000000.json"
    quality = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert quality["accepted"] is True
    assert quality["depth"]["cameras"] == ["cam_head", "cam_left_wrist"]
    assert quality["depth"]["shapes"] == {"cam_head": [120, 160], "cam_left_wrist": [120, 160]}
    assert quality["affordance_2d"]["cameras"] == ["cam_head"]
    assert quality["affordance_2d"]["heatmap_hw"] == [48, 64]
    assert quality["affordance_2d"]["has_selected_grasp"] is True

    with h5py.File(h5_path, "r") as f:
        d_head = f["vision/cam_head/depths"]
        assert d_head.shape == (4, 120, 160) and d_head.dtype == np.float32
        assert d_head.attrs["unit"] == "m"
        np.testing.assert_allclose(d_head[0], 0.75)
        assert f["vision/cam_head/intrinsic_matrix"].shape == (3, 3)
        assert f["vision/cam_head/extrinsic_matrices"].shape == (4, 4, 4)
        assert f["vision/cam_left_wrist/depths"].shape == (4, 120, 160)
        assert "depths" not in f["vision/cam_right_wrist"]
        assert f["affordance_2d/cam_head/heatmap_uint8"].shape == (4, 48, 64)
        assert f["affordance_2d/cam_head/keypoints_xy_score"].shape == (4, 4, 3)
        assert np.isfinite(f["affordance_2d/cam_head/selected_grasp_xy_norm"][()]).all()

    export.check_raw_episodes([(h5_path, quality)])
    card_md = export.card("ds", {"revision": {}}, [(h5_path, quality)], {"episodes": 1, "frames": 4, "fps": 30})
    assert "Depth (stored in raw HDF5" in card_md
    assert "2D affordance annotations" in card_md


def test_collect_depth_strict_error_and_export_checks(tmp_path: Path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")

    # 1. Explicit record_depth=True when obs['vision'] lacks depth raises clear ValueError
    out_missing = tmp_path / "missing_depth"
    out_missing.mkdir()
    env_no_depth = _StubEnvWithObsManager(
        layout_file,
        obs_vision_cfg={"approximate_depth": False, "depth": False},
        include_depth_stream=False,
    )
    with _patch_xpolicylab(), pytest.raises(ValueError, match="lacks 'depth'"):
        collect.record_episode(
            env_no_depth,
            _fake_plugin(n_steps=3),
            out_missing,
            {"env_cfg": "g1d", "fps": 30, "trajectory": True, "early_abort": False, "record_depth": True},
        )

    # 1b. Explicit depth_cameras=["cam_left_wrist"] when only cam_head has depth raises clear ValueError
    out_missing_cam = tmp_path / "missing_cam_depth"
    out_missing_cam.mkdir()
    env_head_only = _StubEnvWithObsManager(
        layout_file,
        obs_vision_cfg={"approximate_depth": False, "depth": ["cam_head"]},
    )
    with _patch_xpolicylab(), pytest.raises(ValueError, match="lacks 'depth'"):
        collect.record_episode(
            env_head_only,
            _fake_plugin(n_steps=3),
            out_missing_cam,
            {
                "env_cfg": "g1d",
                "fps": 30,
                "trajectory": True,
                "early_abort": False,
                "record_depth": True,
                "depth_cameras": ["cam_left_wrist"],
            },
        )

    # 2. Mixing episode with depth and episode without depth is rejected by export.select
    raw_dir = tmp_path / "raw"
    seed_dir = raw_dir / "seed10"
    seed_dir.mkdir(parents=True)

    env_d = _StubEnvWithObsManager(
        layout_file,
        obs_vision_cfg={"approximate_depth": False, "depth": ["cam_head"]},
    )
    with _patch_xpolicylab():
        collect.record_episode(
            env_d,
            _fake_plugin(n_steps=3),
            seed_dir,
            {"env_cfg": "g1d", "fps": 30, "trajectory": True, "early_abort": False, "record_depth": True},
        )
    # Second layout without depth
    env_nd = _StubEnvWithObsManager(
        layout_file,
        obs_vision_cfg={"approximate_depth": False, "depth": False},
        include_depth_stream=False,
    )
    env_nd.current_env_seed_map = {0: 1}
    env_nd.seed_manager.seed_info = {1: {"scene_layout": str(layout_file)}}
    env_nd._scene_info = {
        "Rigid": {"mug": [{"label": "target", "default_pos": [0.55, 0.0, 0.85], "default_ori": [1, 0, 0, 0]}]}
    }
    with _patch_xpolicylab():
        collect.record_episode(
            env_nd,
            _fake_plugin(n_steps=3),
            seed_dir,
            {"env_cfg": "g1d", "fps": 30, "trajectory": True, "early_abort": False, "record_depth": False},
        )

    with pytest.raises(ValueError, match="depth"):
        export.select([raw_dir], 2)

    # 3. Corrupting depth dtype/unit is caught by export.check_raw_episodes
    h5_0 = seed_dir / "episode_000000.hdf5"
    q_0 = json.loads((seed_dir / "episode_000000.json").read_text(encoding="utf-8"))
    with h5py.File(h5_0, "r+") as f:
        f["vision/cam_head/depths"].attrs["unit"] = "mm"
    with pytest.raises(ValueError, match="expected unit 'm'"):
        export.check_raw_episodes([(h5_0, q_0)])


def test_collect_cli_sets_depth_and_affordance_env_vars(tmp_path: Path):
    out_dir = tmp_path / "cli_out"
    layout_dir = tmp_path / "layouts"
    layout_dir.mkdir()
    (layout_dir / "t_0.json").write_text("{}", encoding="utf-8")
    recorded = {}

    import runpy

    def fake_run_path(path, run_name=None):
        recorded["policy_module"] = "scripts.datagen.collect"
        recorded["argv"] = list(sys.argv)
        recorded["OPENFLYWHEEL_COLLECT_DEPTH"] = os.environ.get("OPENFLYWHEEL_COLLECT_DEPTH")
        recorded["OPENFLYWHEEL_COLLECT_AFFORDANCE"] = os.environ.get("OPENFLYWHEEL_COLLECT_AFFORDANCE")
        recorded["OPENFLYWHEEL_COLLECT_CAMERA_MATRICES"] = os.environ.get("OPENFLYWHEEL_COLLECT_CAMERA_MATRICES")
        recorded["OPENFLYWHEEL_AFFORDANCE_HEATMAP_SIZE"] = os.environ.get("OPENFLYWHEEL_AFFORDANCE_HEATMAP_SIZE")
        recorded["OPENFLYWHEEL_AFFORDANCE_TOP_K"] = os.environ.get("OPENFLYWHEEL_AFFORDANCE_TOP_K")
        recorded["cfg"] = json.loads(os.environ[collect.ENV_KEY])

    with (
        patch.object(collect, "layout_files", return_value=[layout_dir / "t_0.json"]),
        patch.object(collect, "load_yaml", return_value={"observation": {"collect_freq": 30}}),
        patch.object(collect.experts, "record", return_value=None),
        patch.object(collect.importlib, "import_module", return_value=SimpleNamespace(__name__="p")),
        patch.object(runpy, "run_path", side_effect=fake_run_path),
        patch.dict(os.environ, {}, clear=False),
        patch.object(
            sys,
            "argv",
            [
                "collect.py",
                "--task",
                "t",
                "--env-cfg",
                "g1d",
                "--seed",
                "0",
                "--output",
                str(out_dir),
                "--record-depth",
                "--depth-cameras",
                "cam_head,cam_left_wrist",
                "--record-affordance",
                "--affordance-cameras",
                "cam_head",
                "--affordance-2d-heatmap-size",
                "32x24",
                "--affordance-2d-top-k",
                "10",
            ],
        ),
    ):
        collect.main()

    assert recorded["policy_module"] == "scripts.datagen.collect"
    assert recorded["OPENFLYWHEEL_COLLECT_DEPTH"] == "cam_head,cam_left_wrist"
    assert recorded["OPENFLYWHEEL_COLLECT_AFFORDANCE"] == "cam_head"
    assert recorded["OPENFLYWHEEL_COLLECT_CAMERA_MATRICES"] == "1"
    assert recorded["OPENFLYWHEEL_AFFORDANCE_HEATMAP_SIZE"] == "32x24"
    assert recorded["OPENFLYWHEEL_AFFORDANCE_TOP_K"] == "10"
    assert recorded["cfg"]["record_depth"] is True
    assert recorded["cfg"]["depth_cameras"] == ["cam_head", "cam_left_wrist"]
    assert recorded["cfg"]["record_affordance_2d"] is True
    assert recorded["cfg"]["affordance_2d_heatmap_hw"] == [24, 32]
    assert recorded["cfg"]["affordance_2d_top_k"] == 10


def test_depth_buffer_reuse_does_not_alias_and_late_affordance_pads_cleanly():
    buf = np.zeros((1, 4, 4, 1), dtype=np.float32)

    class _ReusedBufferCapture:
        def step(self, env_ids=None):
            buf[:] += 1.0
            return [
                {
                    "rgb": [{"data": np.zeros((4, 4, 4), dtype=np.uint8), "info": {}}],
                    "distance_to_image_plane": [{"data": buf[0], "info": {}}],
                }
            ]

    om = ObsManager(
        {
            "collect_freq": 30,
            "robot": {"joint_states": False, "world_ee_state": False},
            "vision": {"approximate_depth": False, "depth": True},
        },
        1,
        1 / 300,
        "t",
        {},
    )
    om.camera_manager = SimpleNamespace(num_cams=1, camera_names=[["cam_head"]])
    om.capture_manager = _ReusedBufferCapture()
    om.robot_manager = None
    om.instruction = ["x"]
    om.env = SimpleNamespace()

    drec = collect.DepthRecorder(["cam_head"], strict=True)
    drec.record(om.get_obs([0])[0])
    drec.record(om.get_obs([0])[0])
    d_payload, _ = drec.payload(2)
    np.testing.assert_allclose(d_payload["cam_head"]["depths"][0], 1.0)
    np.testing.assert_allclose(d_payload["cam_head"]["depths"][1], 2.0)

    # Late-appearing obs["vision"]["cam_head"]["affordance"] must pad leading empty frames cleanly
    arec = collect.Affordance2DRecorder(
        SimpleNamespace(),
        SimpleNamespace(),
        {"affordance_2d_cameras": None, "affordance_2d_heatmap_hw": [12, 16], "affordance_2d_top_k": 4},
        0,
        all_cameras=("cam_head", "cam_left_wrist"),
    )
    assert arec.cameras == ("cam_head", "cam_left_wrist")
    assert not arec.enabled
    arec.record(
        SimpleNamespace(),
        {"vision": {"cam_head": {"color": np.zeros((8, 8, 3), dtype=np.uint8)}}},
        SimpleNamespace(),
    )
    arec.record(
        SimpleNamespace(),
        {
            "vision": {
                "cam_head": {
                    "color": np.zeros((8, 8, 3), dtype=np.uint8),
                    "affordance": {"heatmap_uint8": np.full((12, 16), 200, dtype=np.uint8)},
                }
            }
        },
        SimpleNamespace(),
    )
    a_payload, _ = arec.payload(2)
    assert a_payload["cameras"]["cam_head"]["heatmap_uint8"].shape == (2, 12, 16)
    assert int(a_payload["cameras"]["cam_head"]["heatmap_uint8"][0].max()) == 0
    assert int(a_payload["cameras"]["cam_head"]["heatmap_uint8"][1].max()) == 200


def test_task_common_depth_propagation_preserves_approximate_depth_and_handles_disabled_dict():
    class _FakeBase:
        def __init__(self, sim_config, app=None, interactiveSceneCfg=None, **kwargs):
            self.env_seeds = [0]
            self.use_fabric = False

    class _FakeRobotMgr:
        def __init__(self, *a, **kw):
            pass

        def _get_SceneCfg(self, *a, **kw):
            return None

        def _load_camera_cfg(self, *a, **kw):
            pass

    class _FakeSceneMgr:
        def __init__(self, *a, **kw):
            pass

    class _FakeCamMgr:
        def __init__(self, *a, **kw):
            pass

    class _FakeCapMgr:
        def __init__(self, *a, **kw):
            self._requested_depth_cameras = None

    class _FakeDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    class _FakeOmega:
        DictConfig = _FakeDict

        @staticmethod
        def select(cfg, key):
            cur = cfg
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur

    fake_oc = ModuleType("omegaconf")
    fake_oc.OmegaConf = _FakeOmega
    fake_oc.open_dict = lambda d: SimpleNamespace(__enter__=lambda *a: None, __exit__=lambda *a: None)
    fake_dc = ModuleType("omegaconf.dictconfig")
    fake_dc.DictConfig = _FakeDict
    fake_cm_mod = ModuleType("env.robot_manager.control_manager")
    fake_cm_mod.ControlSeq = object

    with patch.dict(
        sys.modules,
        {"omegaconf": fake_oc, "omegaconf.dictconfig": fake_dc, "env.robot_manager.control_manager": fake_cm_mod},
    ):
        sys.modules.pop("env.environment.task_common", None)
        tc_mod = importlib.import_module("env.environment.task_common")

        class _Env(tc_mod.TaskEnvCommon, _FakeBase):
            @staticmethod
            def manager_classes():
                return (_FakeRobotMgr, _FakeSceneMgr, _FakeCamMgr, _FakeCapMgr)

        def _make_cfg(vision):
            return _FakeDict(
                sim=_FakeDict(scene=_FakeDict(num_envs=1, env_spacing=2.0), dt=1 / 300),
                scene=_FakeDict(),
                robot=_FakeDict(),
                camera=_FakeDict(),
                task_env=_FakeDict(),
                eval_cfg=_FakeDict(observation=_FakeDict(vision=_FakeDict(vision))),
            )

        e1 = _Env(_make_cfg({"depth": ["cam_head"], "approximate_depth": True}), app=None)
        assert e1.capture_manager._requested_depth_cameras is True

        e2 = _Env(_make_cfg({"depth": _FakeDict({"enabled": False}), "approximate_depth": True}), app=None)
        assert e2.capture_manager._requested_depth_cameras is True

        e3 = _Env(_make_cfg({"depth": ["cam_head"], "approximate_depth": False}), app=None)
        assert e3.capture_manager._requested_depth_cameras == {"cam_head"}


def test_collect_cli_all_cameras_env_all_and_no_trajectory(tmp_path: Path):
    import runpy

    out_dir = tmp_path / "cli_out"
    layout_dir = tmp_path / "layouts"
    layout_dir.mkdir()
    (layout_dir / "t_0.json").write_text("{}", encoding="utf-8")

    def _run_cli(argv, extra_env=None):
        recorded = {}

        def fake_run_path(path, run_name=None):
            recorded["OPENFLYWHEEL_COLLECT_DEPTH"] = os.environ.get("OPENFLYWHEEL_COLLECT_DEPTH")
            recorded["OPENFLYWHEEL_COLLECT_AFFORDANCE"] = os.environ.get("OPENFLYWHEEL_COLLECT_AFFORDANCE")
            recorded["cfg"] = json.loads(os.environ[collect.ENV_KEY])

        env_patch = dict(extra_env or {})
        with (
            patch.object(collect, "layout_files", return_value=[layout_dir / "t_0.json"]),
            patch.object(collect, "load_yaml", return_value={"observation": {"collect_freq": 30}}),
            patch.object(collect.experts, "record", return_value=None),
            patch.object(collect.importlib, "import_module", return_value=SimpleNamespace(__name__="p")),
            patch.object(runpy, "run_path", side_effect=fake_run_path),
            patch.dict(os.environ, env_patch, clear=False),
            patch.object(sys, "argv", argv),
        ):
            for k in (
                "OPENFLYWHEEL_COLLECT_DEPTH",
                "OPENFLYWHEEL_COLLECT_AFFORDANCE",
                "OPENFLYWHEEL_COLLECT_CAMERA_MATRICES",
            ):
                if k not in env_patch:
                    os.environ.pop(k, None)
            collect.main()
        return recorded

    # 1. --affordance-2d-cameras all and --depth-cameras all expand to None (all rendered cameras) + "1" env
    r_all = _run_cli(
        [
            "collect.py",
            "--task",
            "t",
            "--env-cfg",
            "g1d",
            "--seed",
            "0",
            "--output",
            str(out_dir / "all"),
            "--record-depth",
            "--depth-cameras",
            "all",
            "--record-affordance-2d",
            "--affordance-2d-cameras",
            "*",
        ]
    )
    assert r_all["OPENFLYWHEEL_COLLECT_DEPTH"] == "1"
    assert r_all["OPENFLYWHEEL_COLLECT_AFFORDANCE"] == "1"
    assert r_all["cfg"]["depth_cameras"] is None
    assert r_all["cfg"]["affordance_2d_cameras"] is None

    # 2. OPENFLYWHEEL_COLLECT_AFFORDANCE=1 without CLI flag keeps affordance_2d_cameras=None (all cameras)
    r_env = _run_cli(
        ["collect.py", "--task", "t", "--env-cfg", "g1d", "--seed", "0", "--output", str(out_dir / "env1")],
        extra_env={"OPENFLYWHEEL_COLLECT_AFFORDANCE": "1"},
    )
    assert r_env["cfg"]["record_affordance_2d"] is True
    assert r_env["cfg"]["affordance_2d_cameras"] is None

    # 3. --no-trajectory does not turn on OPENFLYWHEEL_COLLECT_DEPTH / OPENFLYWHEEL_COLLECT_AFFORDANCE
    r_notraj = _run_cli(
        [
            "collect.py",
            "--task",
            "t",
            "--env-cfg",
            "g1d",
            "--seed",
            "0",
            "--output",
            str(out_dir / "notraj"),
            "--no-trajectory",
            "--record-depth",
            "--record-affordance-2d",
        ]
    )
    assert r_notraj["OPENFLYWHEEL_COLLECT_DEPTH"] is None
    assert r_notraj["OPENFLYWHEEL_COLLECT_AFFORDANCE"] is None


def test_run_batch_forwards_and_resumes_depth_and_affordance_flags(tmp_path: Path):
    from scripts.datagen import run_batch

    out_dir = tmp_path / "batch_run"
    raw_dir = out_dir / "raw"
    seed_dir = raw_dir / "seed100"
    seed_dir.mkdir(parents=True)
    # Create an already-accepted layout=0 so the loop only needs to spawn 1 worker for layout=1,
    # and make fake Popen write layout=1 accepted so target=2 finishes immediately.
    (seed_dir / "episode_000000.json").write_text(json.dumps({"accepted": True}), encoding="utf-8")

    spawned_cmds = []

    class _FakeProc:
        pid = 12345
        returncode = 0

        def poll(self):
            return 0

    def _fake_popen(cmd, **kwargs):
        spawned_cmds.append(list(cmd))
        (seed_dir / "episode_000001.json").write_text(json.dumps({"accepted": True}), encoding="utf-8")
        return _FakeProc()

    with (
        patch.object(run_batch, "expert_version", return_value=None),
        patch.object(run_batch, "code_revision", return_value={"commit": "abc", "dirty": False}),
        patch.object(run_batch, "make_shards", return_value=[]),
        patch.object(run_batch, "source_hashes", return_value={}),
        patch.object(run_batch, "asset_hashes", return_value={}),
        patch.object(run_batch, "layout_files", return_value=[Path("a_0.json"), Path("a_1.json")]),
        patch.object(
            run_batch,
            "load_yaml",
            return_value={"config_name": "g1d", "config": {"sim": "s"}, "observation": {"collect_freq": 30}},
        ),
        patch.object(run_batch.time, "sleep", lambda s: None),
        patch.object(run_batch.subprocess, "Popen", side_effect=_fake_popen),
        patch.object(
            sys,
            "argv",
            [
                "run_batch.py",
                "--output",
                str(out_dir),
                "--task",
                "t",
                "--env-cfg",
                "g1d",
                "--candidates",
                "0",
                "--shards",
                "100",
                "--target",
                "2",
                "--record-depth",
                "--depth-cameras",
                "cam_head,cam_left_wrist",
                "--record-affordance-2d",
                "--affordance-2d-cameras",
                "cam_head",
                "--affordance-2d-heatmap-size",
                "32x24",
                "--affordance-2d-top-k",
                "12",
            ],
        ),
    ):
        # Need fresh non-existing out_dir for non-resume run_batch
        import shutil

        shutil.rmtree(out_dir)

        def _fake_make_shards(args):
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "episode_000000.json").write_text(json.dumps({"accepted": True}), encoding="utf-8")
            return []

        with patch.object(run_batch, "make_shards", side_effect=_fake_make_shards):
            run_batch.main()

    assert len(spawned_cmds) == 1
    cmd0 = spawned_cmds[0]
    for flag in (
        "--record-depth",
        "--depth-cameras",
        "cam_head,cam_left_wrist",
        "--record-affordance-2d",
        "--affordance-2d-cameras",
        "cam_head",
        "--affordance-2d-heatmap-size",
        "32x24",
        "--affordance-2d-top-k",
        "12",
    ):
        assert flag in cmd0

    cfg = json.loads((out_dir / "collection_config.json").read_text(encoding="utf-8"))
    assert cfg["record_depth"] is True
    assert cfg["depth_cameras"] == "cam_head,cam_left_wrist"
    assert cfg["record_affordance_2d"] is True
    assert cfg["affordance_2d_cameras"] == "cam_head"
    assert cfg["affordance_2d_heatmap_size"] == "32x24"
    assert cfg["affordance_2d_top_k"] == 12

    # Resume with additional depth override on an episode-incomplete run and verify config persists
    (seed_dir / "episode_000001.json").unlink()
    spawned_cmds.clear()
    with (
        patch.object(run_batch, "expert_version", return_value=None),
        patch.object(run_batch, "layout_files", return_value=[Path("a_0.json"), Path("a_1.json")]),
        patch.object(run_batch.time, "sleep", lambda s: None),
        patch.object(run_batch.subprocess, "Popen", side_effect=_fake_popen),
        patch.object(
            sys,
            "argv",
            [
                "run_batch.py",
                "--output",
                str(out_dir),
                "--resume",
                "--depth-cameras",
                "cam_head",
            ],
        ),
    ):
        run_batch.main()

    assert "--record-depth" in spawned_cmds[0]
    assert "--record-affordance-2d" in spawned_cmds[0]
    assert "--affordance-2d-heatmap-size" in spawned_cmds[0]
    cfg_resumed = json.loads((out_dir / "collection_config.json").read_text(encoding="utf-8"))
    assert cfg_resumed["record_depth"] is True
    assert cfg_resumed["record_affordance_2d"] is True
    assert cfg_resumed["affordance_2d_top_k"] == 12
