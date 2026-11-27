"""CPU unit tests for G1-D Sharpa Wave 61D wholebody pour-beans expert, plugin, and collector hooks."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts.datagen import collect, early_abort, export
from scripts.datagen.plugins import g1d_pour_beans_plus_sharpa as plugin
from scripts.dexhands import pour_beans_sharpa as expert

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "real2sim" / "robot" / "g1d_sharpa_wave" / "model_contract.json"


def _sharpa_limits(side="left"):
    data = json.loads(CONTRACT.read_text())["hands"][side]
    lim = data.get("effective_active_limits", data["limits"])
    return np.array([[lim[j]["lower"], lim[j]["upper"]] for j in data["joints"]], dtype=float)


@pytest.mark.parametrize("out", [-1.0, 1.0])
def test_side_grasp_frame_and_determinant(out):
    rot, reach = expert.side_grasp(out, 0.04)
    mat = rot.as_matrix()
    assert np.allclose(mat.T @ mat, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-6)
    mirror = np.array([1.0, -1.0, 1.0]) if out > 0 else np.ones(3)
    assert np.allclose(rot.apply(expert.FINGERS * mirror), [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(rot.apply(expert.PALM * mirror), [-out, 0.0, 0.0], atol=1e-6)
    assert np.allclose(rot.apply(expert.THUMB * mirror), [0.0, 0.0, 1.0], atol=1e-6)
    assert reach[0] == pytest.approx(0.025 + 0.04 + expert.PALM_CLEAR, abs=1e-6)


def test_level_rotation_aligns_tilted_cup_upright():
    from scipy.spatial.transform import Rotation

    tilted = Rotation.from_euler("xy", [22.0, -18.0], degrees=True)
    lvl = expert.level_rotation(tilted)
    up = (lvl * tilted).apply([0.0, 0.0, 1.0])
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("radius", [0.028, 0.035, 0.040, 0.050, 0.058])
def test_adapted_closed_pose_within_urdf_limits(side, radius):
    lim = _sharpa_limits(side)
    pose = expert.adapted_closed_pose(radius, lim)
    assert pose.shape == (22,)
    assert np.all(pose >= lim[:, 0])
    assert np.all(pose <= lim[:, 1])
    # Narrower cups curl index MCP_FE deeper than wider cups
    narrow = expert.adapted_closed_pose(0.030, lim)
    wide = expert.adapted_closed_pose(0.055, lim)
    assert narrow[5] > wide[5]


def test_tapered_cup_finger_radii_and_short_cup_grip():
    cavity = {
        "axis": [0.0, 0.0],
        "bottom": -0.046,
        "top": 0.046,
        "r": [0.023, 0.034],
        "body": {"z": [-0.046, 0.0, 0.046], "r": [0.024, 0.031, 0.038]},
    }
    grip_up, r_grip, _, _, height = expert.cup_grip_params(cavity, 1.0)
    assert height == pytest.approx(0.092, abs=1e-6)
    assert grip_up <= 0.046  # keeps thumb below rim on 9.2 cm short cups
    radii = expert.finger_radii(cavity, 1.0, grip_up)
    assert len(radii) == 5
    # Tapered cup: thumb/index see larger radius than ring/pinky
    assert radii[0] > radii[2] > radii[4]
    lim = _sharpa_limits("left")
    pose = expert.adapted_closed_pose(radii, lim)
    assert pose.shape == (22,)
    assert np.all(pose >= lim[:, 0]) and np.all(pose <= lim[:, 1])
    # Pinky MCP_FE (idx 18) curls deeper than index MCP_FE (idx 5) on upward-flaring cup
    assert pose[18] > pose[5]


def test_compute_waist_grasp_pitch_and_motion_audit_margins():
    w_high = expert.compute_waist_grasp_pitch(0.90, 0.0)
    w_nom = expert.compute_waist_grasp_pitch(0.75, 0.0)
    w_low = expert.compute_waist_grasp_pitch(0.64, 0.07)
    assert w_high < w_nom < w_low
    assert 0.14 <= w_high <= 0.20
    assert w_nom == pytest.approx(expert.WAIST_PITCH_GRASP, abs=1e-6)
    assert 0.50 <= w_low <= 0.65
    # Motion-audit compliance: zero speed/margin violations
    assert expert.ARM_VMAX_LOADED < 0.80
    assert expert.ARM_VMAX_EMPTY < 0.80
    assert expert.JOINT_MARGIN > 0.05


def test_wholebody_limiter_respects_speed_and_dim():
    lim_l = _sharpa_limits("left")
    lim_r = _sharpa_limits("right")
    arm_l = SimpleNamespace(arm_key="left_arm_joint_state", grip_key="left_ee_joint_state", hand_limits=lim_l)
    arm_r = SimpleNamespace(arm_key="right_arm_joint_state", grip_key="right_ee_joint_state", hand_limits=lim_r)
    arm_limits = {
        "left_arm_joint_state": np.tile([-2.0, 2.0], (7, 1)),
        "right_arm_joint_state": np.tile([-2.0, 2.0], (7, 1)),
    }
    limiter = expert.WholeBodyLimiter(
        [arm_l, arm_r],
        {"left_arm_joint_state": np.zeros(7), "right_arm_joint_state": np.zeros(7)},
        {"left_ee_joint_state": np.zeros(22), "right_ee_joint_state": np.zeros(22)},
        [0.0, 0.0],
        [0.07],
        arm_limits,
    )
    target = {
        "left_arm_joint_state": np.full(7, 0.4),
        "left_ee_joint_state": expert.adapted_closed_pose(0.04, lim_l),
        "right_arm_joint_state": np.full(7, -0.2),
        "right_ee_joint_state": expert.PRESHAPE_POSE,
        "waist_joint_state": [0.0, 0.25],
        "lift_joint_state": [0.0],
    }
    steps = list(limiter.step(target, loaded={"left_arm_joint_state": True, "right_arm_joint_state": False}))
    assert len(steps) >= 15
    prev_q = np.zeros(7)
    for s in steps:
        q = np.asarray(s["left_arm_joint_state"])
        assert np.abs(q - prev_q).max() <= expert.ARM_VMAX_LOADED / expert.FPS + 1e-9
        prev_q = q
        assert len(s["left_ee_joint_state"]) == 22
        assert len(s["right_ee_joint_state"]) == 22
        assert len(s["waist_joint_state"]) == 2
        assert len(s["lift_joint_state"]) == 1


def test_collect_state_keys_wholebody_and_cam_head_right():
    assert "cam_head_right" in collect.CAMERAS
    rb_l = SimpleNamespace(
        type="target", arm_name="left_arm", gripper_name="left_ee", body_dims=("waist_joint_state", "lift_joint_state")
    )
    rb_r = SimpleNamespace(type="target", arm_name="right_arm", gripper_name="right_ee", body_dims=())
    rm = SimpleNamespace(robot_list=[rb_r, rb_l], process_name=lambda n: f"{n}_joint_state")

    env_wb = SimpleNamespace(robot_manager=rm, obs_manager=SimpleNamespace(robot_cfg={"body_states": True}))
    assert collect.state_keys(env_wb) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
        "waist_joint_state",
        "lift_joint_state",
    ]
    env_no_wb = SimpleNamespace(robot_manager=rm, obs_manager=SimpleNamespace(robot_cfg={"body_states": False}))
    assert collect.state_keys(env_no_wb) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    ]


def test_plugin_audit_and_early_abort():
    assert plugin.__name__ in early_abort.PLUS
    cavity = {
        "axis": [0.0, 0.0],
        "bottom": 0.0,
        "top": 0.10,
        "z": [0.01, 0.095],
        "r": [0.035, 0.040],
        "body": {"z": [0.0, 0.10], "r": [0.038, 0.042]},
    }
    frames = 40
    n_beans = 20
    pose_a = np.array([-0.15, -0.35, 0.75, 1.0, 0.0, 0.0, 0.0])
    pose_b = np.array([0.15, -0.35, 0.75, 1.0, 0.0, 0.0, 0.0])
    cup_poses = np.tile(np.stack([pose_a, pose_b]), (frames, 1, 1))
    init_pts = np.tile(np.array([-0.15, -0.35, 0.79]), (n_beans, 1))
    end_pts = np.tile(np.array([0.15, -0.35, 0.79]), (n_beans, 1))
    particles = np.tile(end_pts, (frames, 1, 1))
    openness = np.full((frames, 2), 0.95, dtype=np.float32)
    privileged = dict(
        initial_particles=init_pts.astype(np.float32),
        particles=particles.astype(np.float32),
        cup_poses=cup_poses.astype(np.float32),
        distractor_poses=np.zeros((frames, 0, 7), dtype=np.float32),
        distractor_poses_start=np.zeros((0, 7), dtype=np.float32),
        hand_openness=openness,
    )
    quality = dict(
        frames=frames,
        initial_counts=dict(total=n_beans, source=n_beans, target=0, outside=0),
        cups={"source_cup": {"cavity": cavity, "scale": 1.0}, "target_cup": {"cavity": cavity, "scale": 1.0}},
        start_z=[0.75, 0.75],
    )
    metrics = plugin.audit(privileged, quality)
    assert metrics["in_target_share"] == 1.0
    assert metrics["min_tail_hand_openness"] >= 0.9

    # Closed hand in tail must fail audit
    bad = dict(privileged, hand_openness=np.full((frames, 2), 0.3, dtype=np.float32))
    with pytest.raises(AssertionError):
        plugin.audit(bad, quality)


def test_sharpa_wb_env_cfgs_resolve_in_robot_info():
    robot_info = json.loads((ROOT / "env_cfg" / "robot" / "_robot_info.json").read_text(encoding="utf-8"))
    for env_cfg_name, expected_robot in (("g1d_sharpa_beans_wb", "g1d_sharpa_wave"),):
        cfg = yaml.safe_load((ROOT / "env_cfg" / f"{env_cfg_name}.yml").read_text(encoding="utf-8"))
        assert cfg["config"]["robot"] == expected_robot
        assert cfg["observation"]["robot"]["body_states"] is True
        assert cfg["observation"]["vision"]["cameras"] == ["cam_head", "cam_head_right"]
        info = robot_info[expected_robot]
        assert info == {"arm_dim": [7, 7], "ee_dim": [22, 22]}
        total_dim = sum(info["arm_dim"]) + sum(info["ee_dim"]) + sum(export.BODY_STATE_DIMS.values())
        assert total_dim == 61


def test_export_camera_map_and_collisions():
    assert export.camera_map(None) == export.CAMERAS
    assert export.camera_map([]) == export.CAMERAS
    assert export.camera_map(["cam_head", "cam_left_wrist", "cam_right_wrist"]) == export.CAMERAS
    assert export.camera_map(["cam_head", "cam_head_right"]) == {
        "cam_high": "cam_head",
        "cam_high_right": "cam_head_right",
    }
    assert export.camera_map(["cam_head", "cam_third"]) == {"cam_high": "cam_head", "cam_third": "cam_third"}
    with pytest.raises(ValueError, match="Duplicate LeRobot camera key"):
        export.camera_map(["cam_head", "cam_high"])


def _fake_converter_module():
    def _get_nested(data, *keys, default=None):
        cur = data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _concat_state_parts(parts, name):
        arrs = [np.asarray(v, dtype=np.float32) for _, v in parts if v is not None]
        return np.concatenate(arrs, axis=1) if arrs else None

    def _dims_from_robot_action_info(info):
        return [a + e for a, e in zip(info["arm_dim"], info["ee_dim"], strict=True)]

    def _build_motor_names_from_dims(per_arm_dims):
        prefixes = (
            ["arm"]
            if len(per_arm_dims) == 1
            else (["left", "right"] if len(per_arm_dims) == 2 else [f"arm_{i}" for i in range(len(per_arm_dims))])
        )
        return [f"{p}_joint_{j}" for p, d in zip(prefixes, per_arm_dims, strict=True) for j in range(d)]

    return SimpleNamespace(
        CAMERA_CANDIDATES={"cam_high": [("vision", "cam_head", "colors")]},
        _get_nested=_get_nested,
        _concat_state_parts=_concat_state_parts,
        _dims_from_robot_action_info=_dims_from_robot_action_info,
        _build_motor_names_from_dims=_build_motor_names_from_dims,
        _extract_qpos=lambda _data: "orig_qpos",
        _extract_action=lambda _data: "orig_action",
    )


def test_export_patch_converter_stereo_and_wholebody_61d():
    # Standard 4 state keys: cameras updated, state extractors untouched
    conv_std = _fake_converter_module()
    export.patch_converter(
        conv_std, cameras=["cam_head", "cam_head_right"], state_keys=list(export.STANDARD_STATE_KEYS)
    )
    assert conv_std.CAMERA_CANDIDATES == {
        "cam_high": [("vision", "cam_head", "colors")],
        "cam_high_right": [("vision", "cam_head_right", "colors")],
    }
    assert conv_std._extract_qpos({}) == "orig_qpos"
    assert conv_std._extract_action({}) == "orig_action"

    # 61D wholebody state_keys: stereo cameras + body dims/names + full 61D extraction
    wb_keys = [*export.STANDARD_STATE_KEYS, "waist_joint_state", "lift_joint_state"]
    conv_wb = _fake_converter_module()
    export.patch_converter(conv_wb, cameras=["cam_head", "cam_head_right"], state_keys=wb_keys)
    dims = conv_wb._dims_from_robot_action_info({"arm_dim": [7, 7], "ee_dim": [22, 22]})
    assert dims == [29, 29, 2, 1]
    assert sum(dims) == 61
    motors = conv_wb._build_motor_names_from_dims(dims)
    assert len(motors) == 61
    assert motors[:2] == ["left_joint_0", "left_joint_1"]
    assert motors[29:31] == ["right_joint_0", "right_joint_1"]
    assert motors[58:] == ["waist_yaw", "waist_pitch", "lift_height"]

    t = 4
    sample = {
        group: {
            "left_arm_joint_states": np.ones((t, 7), dtype=np.float32),
            "left_ee_joint_states": np.full((t, 22), 2.0, dtype=np.float32),
            "right_arm_joint_states": np.full((t, 7), 3.0, dtype=np.float32),
            "right_ee_joint_states": np.full((t, 22), 4.0, dtype=np.float32),
            "waist_joint_states": np.full((t, 2), 5.0, dtype=np.float32),
            "lift_joint_states": np.full((t, 1), 6.0, dtype=np.float32),
        }
        for group in ("state", "action")
    }
    qpos = conv_wb._extract_qpos(sample)
    act = conv_wb._extract_action(sample)
    assert qpos.shape == (t, 61) and act.shape == (t, 61)
    assert qpos[0, 58] == pytest.approx(5.0) and qpos[0, 60] == pytest.approx(6.0)

    # Newer XPolicyLab converter with native `body_dim=3`: keep native state/dims/motor hooks intact
    orig_native_qpos = lambda _data, body_dim=0: ("native_qpos", body_dim)  # noqa: E731
    orig_native_action = lambda _data, body_dim=0: ("native_action", body_dim)  # noqa: E731
    orig_native_motors = lambda per_arm_dims, body_dim=0: (list(per_arm_dims), body_dim)  # noqa: E731
    conv_native = _fake_converter_module()
    conv_native._extract_qpos = orig_native_qpos
    conv_native._extract_action = orig_native_action
    conv_native._build_motor_names_from_dims = orig_native_motors
    export.patch_converter(conv_native, cameras=["cam_head", "cam_head_right"], state_keys=wb_keys)
    assert conv_native.CAMERA_CANDIDATES == {
        "cam_high": [("vision", "cam_head", "colors")],
        "cam_high_right": [("vision", "cam_head_right", "colors")],
    }
    assert conv_native._extract_qpos is orig_native_qpos
    assert conv_native._extract_action is orig_native_action
    assert conv_native._build_motor_names_from_dims is orig_native_motors
    assert conv_native._dims_from_robot_action_info({"arm_dim": [7, 7], "ee_dim": [22, 22]}) == [29, 29]
    assert conv_native._build_motor_names_from_dims([29, 29], body_dim=3) == ([29, 29], 3)

    # Unknown extra state key fails fast
    with pytest.raises(ValueError, match="Unsupported extra state_keys"):
        export.patch_converter(_fake_converter_module(), state_keys=[*export.STANDARD_STATE_KEYS, "base_velocity"])


def test_export_verify_and_card_stereo_61d_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")

    wb_keys = [*export.STANDARD_STATE_KEYS, "waist_joint_state", "lift_joint_state"]
    key_dims = {
        "left_arm_joint_state": 7,
        "left_ee_joint_state": 22,
        "right_arm_joint_state": 7,
        "right_ee_joint_state": 22,
        "waist_joint_state": 2,
        "lift_joint_state": 1,
    }
    n = 6
    hdf5_path = tmp_path / "episode_000000.hdf5"
    state_full = np.arange(n * 61, dtype=np.float32).reshape(n, 61) * 0.01
    action_full = state_full + 0.05
    # Minimal valid SOI..EOI fake JPEG payload encoding frame/camera value in middle byte
    cam_vals = {"cam_head": 80, "cam_head_right": 160}
    with h5py.File(hdf5_path, "w") as f:
        offset = 0
        for k in wb_keys:
            d = key_dims[k]
            f[f"state/{k}s"] = state_full[:, offset : offset + d]
            f[f"action/{k}s"] = action_full[:, offset : offset + d]
            offset += d
        for cam, val in cam_vals.items():
            f[f"vision/{cam}/colors"] = np.asarray(
                [b"\xff\xd8" + bytes([val + i]) + b"\xff\xd9" for i in range(n)], dtype="S5"
            )

    quality = dict(
        task="g1d_pour_beans_plus",
        plugin="scripts.datagen.plugins.g1d_pour_beans_plus_sharpa",
        acceptance_version="sharpa_plus_v1",
        hold_frames=2,
        stable_frames=2,
        frames=n,
        env_cfg="g1d_shenzhen_lab_sharpa_beans_wb",
        config_name="g1d_shenzhen_lab",
        physics_dt=1.0 / 600.0,
        fps=30,
        layout_seed=9931,
        instruction="pour the beans",
        cameras=["cam_head", "cam_head_right"],
        state_keys=wb_keys,
        hand_joint_names={"left_ee_joint_state": [f"lh_{i}" for i in range(22)]},
    )
    chosen = [(hdf5_path, quality)]
    export.check_raw_episodes(chosen)

    class _FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)

        @property
        def shape(self):
            return self._arr.shape

        def numpy(self):
            return self._arr

    def _make_dataset_cls(corrupt_right=False):
        class _FakeLeRobotDataset:
            def __init__(self, _name, root):
                self.num_episodes = 1
                self.num_frames = n
                self.fps = 30
                self.meta = SimpleNamespace(
                    episodes=[{"dataset_from_index": 0, "dataset_to_index": n, "tasks": ["pour the beans"]}]
                )

            def __getitem__(self, row):
                img_l = np.full((3, 32, 32), (cam_vals["cam_head"] + row) / 255.0, dtype=np.float32)
                img_r = np.full((3, 32, 32), (cam_vals["cam_head_right"] + row) / 255.0, dtype=np.float32)
                if corrupt_right and row == 0:
                    img_r[:, :16, :16] = 0.0  # 16x16 dark square artifact
                return {
                    "episode_index": 0,
                    "frame_index": row,
                    "observation.state": _FakeTensor(state_full[row]),
                    "action": _FakeTensor(action_full[row]),
                    "observation.images.cam_high": _FakeTensor(img_l),
                    "observation.images.cam_high_right": _FakeTensor(img_r),
                }

        return _FakeLeRobotDataset

    def _fake_decode(buf):
        val = bytes(buf)[2]
        return np.full((32, 32, 3), val, dtype=np.uint8)

    fake_lerobot_ds = ModuleType("lerobot.datasets.lerobot_dataset")
    fake_lerobot_ds.LeRobotDataset = _make_dataset_cls(corrupt_right=False)
    fake_xpl_proc = ModuleType("XPolicyLab.utils.process_data")
    fake_xpl_proc.decode_image_bit = _fake_decode

    fake_modules = {
        "lerobot": ModuleType("lerobot"),
        "lerobot.datasets": ModuleType("lerobot.datasets"),
        "lerobot.datasets.lerobot_dataset": fake_lerobot_ds,
        "XPolicyLab": ModuleType("XPolicyLab"),
        "XPolicyLab.utils": ModuleType("XPolicyLab.utils"),
        "XPolicyLab.utils.process_data": fake_xpl_proc,
    }
    with patch.dict(sys.modules, fake_modules):
        result = export.verify(tmp_path, "sharpa_test", chosen)
        assert result == dict(status="verified", episodes=1, frames=n, fps=30, max_block_diff=0.0)

        fake_lerobot_ds.LeRobotDataset = _make_dataset_cls(corrupt_right=True)
        with pytest.raises(ValueError, match="cam_high_right frame 0: 16x16 block diff"):
            export.verify(tmp_path, "sharpa_test", chosen)

    readme = export.card("sharpa_test", {"revision": {"commit": "abc1234", "dirty": False}}, chosen, result)
    assert "observation.images.cam_high" in readme
    assert "observation.images.cam_high_right" in readme
    assert "cam_left_wrist" not in readme
    assert "waist_joint_state" in readme and "lift_joint_state" in readme
    assert "Dex1-1" not in readme


def test_export_script_adds_repo_root_to_sys_path():
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('export_cli', {str(ROOT / 'scripts/datagen/export.py')!r}); "
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        f"assert str({str(ROOT)!r}) in sys.path"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT / "scripts"), env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
