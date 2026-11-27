"""Unit tests for 19D fixed-base wholebody (`g1d_wb19`: waist + lift + dual arms) pipeline, layout recipes,
X-Anything wholebody experts/plugins, and LeRobot v3 export compatibility."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.robot_manager.robot_class.g1d import LIFT_LIMITS, WAIST_LIMITS, G1DLeft  # noqa: E402
from scripts.datagen import (  # noqa: E402
    collect,
    early_abort,
    export as datagen_export,
    handover_anything_layouts as hoal,
    motion,
    pick_anything_layouts as pal,
    place_anything_layouts as plal,
)
from scripts.datagen.plugins import (  # noqa: E402
    g1d_handover_anything_wb,
    g1d_pick_anything_wb,
    g1d_place_anything_wb,
    g1d_put_anything_in_wb,
)
from scripts.g1d import wholebody_demo as wb, x_anything_wb  # noqa: E402


def _make_fake_env(body_states=True, waist=(0.0, 0.0), lift=(0.035,), table_height=0.80):
    class _FakeData:
        def __init__(self):
            self.joint_pos = np.zeros((1, 9), dtype=np.float32)
            self.soft_joint_pos_limits = np.tile(np.array([[-2.5, 2.5]], dtype=np.float32), (1, 9, 1))

    class _FakeKey:
        def __init__(self):
            self.joint_names = [f"j{i}" for i in range(7)] + ["f1", "f2"]
            self.body_names = ["link_ee"]
            self.data = _FakeData()

    class _FakeRobot:
        def __init__(self, side, has_body=False):
            self.type = "target"
            self.arm_name = f"{side}_arm"
            self.gripper_name = f"{side}_ee"
            self.arm_joints_name = [f"j{i}" for i in range(7)]
            self.arm_joint_indices = list(range(7))
            self.gripper_joint_indices = [7, 8]
            self.ee_link_name = "link_ee"
            if has_body:
                self.body_dims = {"waist_joint_state": 2, "lift_joint_state": 1}

    left = _FakeRobot("left", has_body=True)
    right = _FakeRobot("right", has_body=False)

    class _FakeRM:
        def __init__(self):
            self.robot_list = [left, right]
            self.robot_key = [_FakeKey(), _FakeKey()]

        def process_name(self, name):
            return f"{name}_joint_state"

        def get_body_state(self, robot, env_idx_list=None):
            assert robot is left
            assert env_idx_list == [0]
            return {0: {"waist_joint_state": list(waist), "lift_joint_state": list(lift)}}

    lm = types.SimpleNamespace(saved_layouts=[{"Room": {"table_height": table_height}}])
    return types.SimpleNamespace(
        robot_manager=_FakeRM(),
        obs_manager=types.SimpleNamespace(robot_cfg={"body_states": body_states}),
        scene_manager=types.SimpleNamespace(layout_manager=lm, env_origins=np.zeros((1, 3), dtype=np.float32)),
    )


def test_g1d_body_targets_clips_limits():
    r = object.__new__(G1DLeft)
    r.mobile = False
    pos, vel = r.body_targets(
        {"waist_joint_state": [10.0, -5.0], "lift_joint_state": [0.99]},
        np.zeros(4),
        np.zeros(4),
        1 / 30,
    )
    assert np.allclose(pos[:2], [WAIST_LIMITS[0, 1], WAIST_LIMITS[1, 0]])
    assert np.allclose(pos[2:], [LIFT_LIMITS[1] / 2, LIFT_LIMITS[1] / 2])
    assert np.allclose(vel, 0.0)


def test_state_keys_16d_vs_19d():
    env16 = _make_fake_env(body_states=False)
    env19 = _make_fake_env(body_states=True)
    assert collect.state_keys(env16) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    ]
    assert collect.state_keys(env19) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
        "waist_joint_state",
        "lift_joint_state",
    ]


def test_plan_body_posture_exercises_lift_and_pitch():
    # High table, near target -> lift increases, pitch stays ~0
    p_high = wb.plan_body_posture(target_y=-0.45, target_z=0.95, table_top_z=0.92, comfort_reach_y=0.32)
    assert p_high["waist_joint_state"][0] == 0.0
    assert p_high["lift_joint_state"][0] == pytest.approx(0.24, abs=1e-6)
    assert p_high["waist_joint_state"][1] == pytest.approx(0.0, abs=1e-6)

    # Far target on table -> positive waist pitch within max_pitch
    p_far = wb.plan_body_posture(
        target_y=-0.22,
        target_z=0.78,
        table_top_z=0.75,
        max_pitch=x_anything_wb.WB_MAX_PITCH,
        comfort_reach_y=x_anything_wb.WB_COMFORT_REACH_Y,
    )
    assert p_far["waist_joint_state"][0] == 0.0
    assert 0.15 < p_far["waist_joint_state"][1] <= x_anything_wb.WB_MAX_PITCH + 1e-9
    assert p_far["lift_joint_state"][0] == pytest.approx(0.07, abs=1e-6)


def test_wholebody_wdls_step_respects_speed_and_limits():
    jac_arm = np.eye(6, 7)
    jac_p = np.array([0.0, 0.45, 0.0, 0.0, 1.0, 0.0])
    jac_l = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    err = np.array([0.0, 0.2, 0.1, 0.0, 0.0, 0.0])
    q_arm = np.zeros(7)
    lim_arm = np.tile([-2.0, 2.0], (7, 1))
    qn, pn, ln = wb.wholebody_wdls_step(jac_arm, jac_p, jac_l, err, q_arm, 0.1, 0.1, lim_arm, fps=30.0)
    assert np.abs(qn - q_arm).max() <= wb.demo.REAL_ARM_VMAX / 30.0 + 1e-9
    assert abs(pn - 0.1) <= wb.REAL_WAIST_VMAX / 30.0 + 1e-9
    assert abs(ln - 0.1) <= wb.REAL_LIFT_VMAX / 30.0 + 1e-9


def test_wholebody_limiter_and_motion_summarize_zero_violations():
    env = _make_fake_env(body_states=True, waist=(0.0, 0.0), lift=(0.035,))
    lim = wb.WholebodyLimiter(env)
    fps = 30.0
    states = {k: [] for k in collect.state_keys(env)}
    actions = {k: [] for k in collect.state_keys(env)}
    cur_body = {"waist_joint_state": [0.0, 0.0], "lift_joint_state": [0.035]}
    # Jump command far away — WholebodyLimiter must rate-limit it so motion.summarize has 0 violations
    for _ in range(30):
        raw = {
            "left_arm_joint_state": [0.0] * 7,
            "left_ee_joint_state": [1.0],
            "right_arm_joint_state": [0.0] * 7,
            "right_ee_joint_state": [1.0],
            "waist_joint_state": [0.0, 0.35],
            "lift_joint_state": [0.25],
        }
        act = lim(raw)
        for k in states:
            if k in cur_body:
                states[k].append(list(cur_body[k]))
                cur_body[k] = list(act[k])
            else:
                states[k].append(list(act[k]))
            actions[k].append(list(act[k]))
    limits = {
        "left_arm_joint_state": [[-2.5, 2.5]] * 7,
        "right_arm_joint_state": [[-2.5, 2.5]] * 7,
        "waist_joint_state": wb.WAIST_URDF_LIMITS.tolist(),
        "lift_joint_state": wb.LIFT_URDF_LIMITS.tolist(),
    }
    summ = motion.summarize(states, actions, fps, ["left_arm_joint_state", "right_arm_joint_state"], limits)
    assert summ["violations"] == []
    w = motion.worst(summ)
    assert w["motion_violations"] == 0
    assert w["motion_waist_pitch_span"] > 0.2
    assert w["motion_lift_span"] > 0.05

    # Exceeding waist speed must be flagged as waist_speed violation
    bad_actions = dict(actions)
    bad_actions["waist_joint_state"] = [[0.0, 0.0]] + [[0.0, 0.5]] * (len(actions["waist_joint_state"]) - 1)
    rep = motion.report([(0, states, bad_actions)], fps, limits)
    assert rep["over_real_limits"]["episodes"] == 1
    assert rep["over_real_limits"]["waist_speed"] == 1


def test_export_wb19_layout_key_card_and_converter_patch():
    q1 = {"planned_poses": {"target": [[0, 0, 0], [1, 0, 0, 0]]}, "init_body_state": {"lift_joint_state": [0.05]}}
    q2 = {"planned_poses": {"target": [[0, 0, 0], [1, 0, 0, 0]]}, "init_body_state": {"lift_joint_state": [0.20]}}
    assert datagen_export.layout_key(q1) != datagen_export.layout_key(q2)
    assert datagen_export.is_wb19_state_keys(["left_arm_joint_state", "waist_joint_state", "lift_joint_state"])
    assert not datagen_export.is_wb19_state_keys(["left_arm_joint_state", "right_arm_joint_state"])

    # Mock XPolicyLab converter object
    def _concat_state_parts(parts, name):
        arrs = [np.asarray(v, dtype=np.float32) for _, v in parts if v is not None]
        return np.concatenate(arrs, axis=1)

    def _get_nested(data, *keys):
        cur = data
        for k in keys:
            cur = cur.get(k) if isinstance(cur, dict) else None
        return cur

    fake_conv = types.SimpleNamespace(
        _dims_from_robot_action_info=lambda info: [8, 8],
        _build_motor_names_from_dims=lambda dims: [f"m{i}" for i in range(sum(dims))],
        _concat_state_parts=_concat_state_parts,
        _get_nested=_get_nested,
    )
    datagen_export.patch_converter_wb19(fake_conv)
    dims = fake_conv._dims_from_robot_action_info({"arm_dim": [7, 7], "ee_dim": [1, 1]})
    assert dims == [8, 8, 3]
    names = fake_conv._build_motor_names_from_dims(dims)
    assert len(names) == 19
    assert names[-3:] == ["waist_yaw", "waist_pitch", "lift_height"]
    fake_h5 = {
        "state": {
            "left_arm_joint_states": np.zeros((4, 7)),
            "left_ee_joint_states": np.ones((4, 1)),
            "right_arm_joint_states": np.zeros((4, 7)),
            "right_ee_joint_states": np.ones((4, 1)),
            "waist_joint_states": np.full((4, 2), 0.1),
            "lift_joint_states": np.full((4, 1), 0.2),
        },
        "action": {
            "left_arm_joint_states": np.zeros((4, 7)),
            "left_ee_joint_states": np.ones((4, 1)),
            "right_arm_joint_states": np.zeros((4, 7)),
            "right_ee_joint_states": np.ones((4, 1)),
            "waist_joint_states": np.full((4, 2), 0.15),
            "lift_joint_states": np.full((4, 1), 0.25),
        },
    }
    assert fake_conv._extract_qpos(fake_h5).shape == (4, 19)
    assert fake_conv._extract_action(fake_h5).shape == (4, 19)


def test_env_cfg_wb_yaml_files_present():
    for name in (
        "g1d_wb.yml",
        "g1d_beans_rigid_wb.yml",
    ):
        text = (ROOT / "env_cfg" / name).read_text()
        assert "body_states: true" in text


def test_wholebody_plugins_and_early_abort_registration():
    assert g1d_pick_anything_wb.__name__ in early_abort.PICK
    assert g1d_place_anything_wb.__name__ in early_abort.PLACE
    assert g1d_put_anything_in_wb.__name__ in early_abort.PLACE
    assert g1d_handover_anything_wb.__name__ in early_abort.HANDOVER
    assert g1d_pick_anything_wb.STEP_LIM == 900
    assert g1d_place_anything_wb.STEP_LIM == 1800
    assert g1d_put_anything_in_wb.STEP_LIM == 1800
    assert g1d_handover_anything_wb.STEP_LIM == 1800


def test_wholebody_x_anything_layout_recipes(tmp_path):
    # Minimal mock asset metadata for pick/place/handover layout generation
    pool = [
        {"cat": "mug", "idx": 1, "family": "cup", "name": "mug", "size": [0.05, 0.05, 0.08], "mass": 0.2},
        {"cat": "box", "idx": 1, "family": "box", "name": "box", "size": [0.05, 0.12, 0.05], "mass": 0.2},
        {"cat": "cup", "idx": 2, "family": "cup", "name": "cup", "size": [0.05, 0.05, 0.07], "mass": 0.15},
        {"cat": "toy", "idx": 1, "family": "toy", "name": "toy", "size": [0.04, 0.06, 0.05], "mass": 0.1},
        {"cat": "can", "idx": 1, "family": "can", "name": "can", "size": [0.05, 0.05, 0.09], "mass": 0.2},
    ]
    rec = {
        "id": "tray/00001",
        "cat": "tray",
        "idx": 1,
        "family": "tray",
        "kinds": ["support", "container"],
        "split": {"support": "train", "container": "train"},
        "size": [0.25, 0.20, 0.03],
        "mass": 0.3,
        "geometry": {
            "support": {
                "bottom": -0.015,
                "top": 0.015,
                "rim": 0.015,
                "floor": 0.015,
                "region": {"center": [0.0, 0.0], "half": [0.10, 0.08], "radius": 0.08},
            },
            "container": {
                "bottom": -0.015,
                "top": 0.03,
                "rim": 0.03,
                "floor": 0.005,
                "region": {"center": [0.0, 0.0], "half": [0.10, 0.08], "radius": 0.08},
            },
        },
    }
    import json

    for obj in [*pool, rec]:
        d = tmp_path / "Object" / "RoboDojo" / "Rigid" / obj["cat"] / f"{obj['idx']:05d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({"physics": {"size": obj["size"], "mass": obj["mass"]}}))

    # Pick wholebody across all 3 scenes (including g1d_lab)
    gen_pick = pal.Generator(tmp_path, pool=pool, names={}, round_="pick-wb-test", wholebody=True)
    assert gen_pick.spec["target_y"] == pal.WHOLEBODY_DEFAULT["target_y"]
    for cfg in ("g1d", "g1d_shenzhen_lab", "g1d_lab"):
        lay = gen_pick.layout(cfg, 100, 0)
        assert "Robot" in lay and "init_joint_pos" in lay["Robot"]
        ijp = lay["Robot"]["init_joint_pos"]
        assert set(ijp) == {"LZ_mt_Joint", "LZ_it_Joint", "Yaw_Joint", "torso_Joint"}
        assert ijp["torso_Joint"] == 0.0
        wb_meta = lay["Generalization"]["wholebody"]
        assert 0.64 <= wb_meta["table_top_z"] <= 0.96
        assert lay["Room"]["table_height"] == wb_meta["table_top_z"]
        assert lay["Generalization"]["base"]["lift"] == 0.0
        assert lay["Generalization"]["base"]["table"] == 0.0

    # Place wholebody
    gen_place = plal.Generator(
        "g1d_place_anything", tmp_path, pool=pool, receptacles=[rec], names={}, round_="place-wb-test", wholebody=True
    )
    assert gen_place.spec["target_y"] == plal.WHOLEBODY_DEFAULT["target_y"]
    assert gen_place.spec["receptacle_y"] == plal.WHOLEBODY_DEFAULT["receptacle_y"]
    lay_p = gen_place.layout("g1d_shenzhen_lab", 200, 0)
    assert "wholebody" in lay_p["Generalization"] and "Robot" in lay_p

    # Handover wholebody (ho.SPEC uses x_y / y_y, not target_y / receptacle_y)
    gen_ho = hoal.Generator(
        tmp_path, pool=[pool[1]], distractors=pool, receptacles=[rec], names={}, round_="ho-wb-test", wholebody=True
    )
    assert gen_ho.spec["x_y"] == hoal.WHOLEBODY_DEFAULT["x_y"]
    assert gen_ho.spec["y_y"] == hoal.WHOLEBODY_DEFAULT["y_y"]
    ho_target_ys, ho_rec_ys = [], []
    for i in range(25):
        lay_h = gen_ho.layout("g1d_lab", 300, i)
        assert "wholebody" in lay_h["Generalization"] and "Robot" in lay_h
        ho_target_ys.append(lay_h["Generalization"]["target_xy"][1])
        ho_rec_ys.append(lay_h["Generalization"]["receptacle_xy"][1])
    # Expanded wholebody depth reaches beyond 16D ho.SPEC bounds (x_y max -0.32, y_y max -0.36)
    assert max(ho_target_ys) > -0.32
    assert max(ho_rec_ys) > -0.36


def test_wholebody_mixin_preserves_commanded_body_target():
    class _BaseRobot:
        def __init__(self, env):
            self.env = env

        def _command(self, arm, point, rot, grip):
            return {"left_arm_joint_state": [0.0] * 7}

        def finish(self):
            yield {"left_arm_joint_state": [0.0] * 7}

    class _WBRobot(wb.WholebodyMixin, _BaseRobot):
        pass

    # Measured env state has gravity sag (0.37 rad) vs commanded target (0.35 rad)
    env = _make_fake_env(body_states=True, waist=(0.0, 0.37), lift=(0.145,))
    commanded = {"waist_joint_state": [0.0, 0.35], "lift_joint_state": [0.15]}
    robot = _WBRobot(env, body=commanded)
    cmd = robot._command(None, None, None, None)
    assert cmd["waist_joint_state"] == [0.0, 0.35]
    assert cmd["lift_joint_state"] == [0.15]
    assert list(robot.finish())[0]["waist_joint_state"] == [0.0, 0.35]
    # Mutating the caller's dict must not mutate robot.body
    commanded["waist_joint_state"][1] = 0.0
    assert robot.body["waist_joint_state"] == [0.0, 0.35]
