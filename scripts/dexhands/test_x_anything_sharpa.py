"""Unit tests for 61D Sharpa Wave wholebody X-Anything (pool, primitives, tactile gating, plugins, env configs)."""

from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from scripts.datagen import early_abort
from scripts.datagen.plugins import (
    g1d_handover_anything_sharpa,
    g1d_pick_anything_sharpa,
    g1d_place_anything_sharpa,
    g1d_put_anything_in_sharpa,
)
from scripts.dexhands import pour_beans_sharpa as pbs, x_anything_sharpa as xas
from scripts.objects import pick_anything, sharpa_x_anything as sxa

ROOT = Path(__file__).resolve().parents[2]


class _FakeData:
    def __init__(self, n_arm=7):
        self.soft_joint_pos_limits = np.zeros((1, 20, 2), dtype=float)
        self.soft_joint_pos_limits[0, :, 0] = -2.5
        self.soft_joint_pos_limits[0, :, 1] = 2.5
        self.joint_pos = np.zeros((1, 20), dtype=float)
        self.body_link_pose_w = np.zeros((1, 24, 7), dtype=float)
        self.body_link_pose_w[0, :, 3] = 1.0  # identity quat wxyz


class _FakeKey:
    def __init__(self, side):
        self.data = _FakeData()
        self.body_names = [f"{side}_{n}" for n in g1d_handover_anything_sharpa.SHARPA_HAND_LINKS]


class _FakeRobot:
    def __init__(self, side, tactile_forces=None):
        self.type = "target"
        self.side = side
        self.arm_name = f"{side}_arm"
        self.gripper_name = f"{side}_gripper"
        self.body_dims = {"waist_joint_state": 2, "lift_joint_state": 1}
        self._limits = np.tile(np.array([[-0.5, 1.8]], dtype=float), (pbs.HAND_DIM, 1))
        self._tactile_forces = tactile_forces
        self._q = np.zeros(7, dtype=float)
        self._pos = np.array([-0.25 if side == "left" else 0.25, -0.35, 0.90], dtype=float)
        self._rot = Rotation.identity()

    def hand_limits(self):
        return self._limits

    def hand_openness(self, q):
        lim = self._limits
        q = np.asarray(q, dtype=float).reshape(-1)
        rest = np.clip(0.0, lim[:, 0], lim[:, 1])
        return float(1.0 - np.mean(np.abs(q - rest) / np.maximum(lim[:, 1] - lim[:, 0], 1e-6)))


class _FakeHandArm:
    """Lightweight replacement for pour_beans_sharpa.HandArm in CPU tests (no Isaac/CuRobo)."""

    def __init__(self, env, robot, reach):
        self.env = env
        self.robot = robot
        self.REACH = np.asarray(reach, dtype=float)
        self.arm_key = robot.arm_name
        self.grip_key = robot.gripper_name
        self.hand_limits = robot.hand_limits()
        self.cols = list(range(7))
        self.key = _FakeKey(robot.side)
        self._q = np.zeros(7, dtype=float)
        self._p = robot._pos.copy()
        self._r = robot._rot

    def q(self):
        return self._q.copy()

    def grasp_point(self):
        return self._p.copy()

    def pose(self):
        return self._p.copy(), self._r

    def ik(self, pos, rot, smooth=True):
        pos = np.asarray(pos, dtype=float)
        self._p = pos.copy()
        self._r = rot
        # Smooth deterministic synthetic joint target within limits
        target = np.clip(np.r_[pos - np.array([0.0, -0.40, 0.85]), rot.as_rotvec(), 0.0], -1.8, 1.8)
        self._q = target
        return target.copy()


def _make_fake_env(target_pos=(-0.12, -0.42, 0.80), target_size=(0.05, 0.05, 0.11), tactile_seq=None):
    left = _FakeRobot("left")
    right = _FakeRobot("right")
    call_count = {"n": 0}

    class _RM:
        robot_list = [left, right]

        @staticmethod
        def process_name(name):
            return name

        @staticmethod
        def get_body_state(rb, env_idx_list=None):
            return {0: {"waist_joint_state": np.array([0.0, 0.10]), "lift_joint_state": np.array([0.05])}}

        @staticmethod
        def get_tactile_contact(rb, env_idx_list=None):
            if tactile_seq is None:
                return None
            idx = min(call_count["n"], len(tactile_seq) - 1)
            call_count["n"] += 1
            return {0: {"normal_force_world_N": tactile_seq[idx]}}

    class _LM:
        @staticmethod
        def get_instance_pose(env_idx, label="target"):
            if label == "target":
                return np.asarray(target_pos, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            return np.array([0.08, -0.42, 0.75], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    env = SimpleNamespace(
        robot_manager=_RM(),
        scene_manager=SimpleNamespace(layout_manager=_LM()),
        entries=lambda _i: {
            "target": {"category": "bottle", "category_idx": 0, "physics": {"size": list(target_size), "mass": 0.18}},
            "receptacle": {
                "category": "plate",
                "category_idx": 0,
                "receptacle": {
                    "kind": "support",
                    "region": {"center": [0.0, 0.0], "half": [0.09, 0.09], "radius": 0.09},
                    "floor": 0.01,
                    "rim": 0.03,
                    "top": 0.03,
                    "bottom": 0.0,
                },
            },
        },
    )
    return env, target_pos, target_size


class SharpaPoolTests(unittest.TestCase):
    def test_committed_pool_size_commercial_and_primitives(self):
        pool = sxa.load_pool()
        self.assertGreaterEqual(len(pool), 80)
        self.assertLessEqual(len(pool), 120)
        prims = {o["primitive"] for o in pool}
        self.assertEqual(prims, set(sxa.PRIMITIVES))
        for obj in pool:
            self.assertFalse(obj["cat"].startswith(pick_anything.NONCOMMERCIAL), obj["cat"])
            self.assertGreaterEqual(obj["mass"], sxa.SHARPA_RULES["mass"][0])
            self.assertLessEqual(obj["mass"], sxa.SHARPA_RULES["mass"][1])
            self.assertEqual(sxa.select_primitive(obj["size"], obj["mass"]), obj["primitive"])

    def test_select_primitive_boundaries(self):
        self.assertEqual(sxa.select_primitive([0.05, 0.05, 0.12], 0.20), "power_wrap")
        self.assertEqual(sxa.select_primitive([0.04, 0.04, 0.05], 0.10), "tripod_pinch")
        self.assertEqual(sxa.select_primitive([0.04, 0.11, 0.03], 0.08), "lateral")
        self.assertIsNone(sxa.select_primitive([0.05, 0.05, 0.12], 0.60))  # overweight
        self.assertIsNone(sxa.select_primitive([0.01, 0.01, 0.05], 0.05))  # too narrow
        self.assertIsNone(sxa.select_primitive([], 0.10))

    def test_handover_pool_and_layout_hooks(self):
        ho_pool = sxa.handover_pool()
        self.assertGreater(len(ho_pool), 30)
        for obj in ho_pool:
            self.assertIn("shift", obj)
        rng = random.Random(42)
        pick_rigid = sxa.pick_layout(rng, index=0)
        labels = {e["label"] for es in pick_rigid.values() for e in es}
        self.assertIn("target", labels)
        place_rigid, _ = sxa.place_layout(rng, index=0, kind="support")
        place_entries = [e for es in place_rigid.values() for e in es]
        place_labels = {e["label"] for e in place_entries}
        self.assertIn("target", place_labels)
        self.assertIn("receptacle", place_labels)
        rec_entry = next(e for e in place_entries if e["label"] == "receptacle")
        self.assertEqual(rec_entry["receptacle"]["kind"], "support")
        for key in ("id", "region", "floor", "rim", "top"):
            self.assertIn(key, rec_entry["receptacle"])

        put_rigid, _ = sxa.place_layout(rng, index=0, kind="container")
        put_entries = [e for es in put_rigid.values() for e in es]
        put_labels = {e["label"] for e in put_entries}
        self.assertIn("target", put_labels)
        self.assertIn("receptacle", put_labels)
        put_rec = next(e for e in put_entries if e["label"] == "receptacle")
        self.assertEqual(put_rec["receptacle"]["kind"], "container")
        self.assertIn("region", put_rec["receptacle"])

        ho_rigid, _ = sxa.handover_layout(rng, index=0)
        ho_entries = [e for es in ho_rigid.values() for e in es]
        ho_labels = {e["label"] for e in ho_entries}
        self.assertIn("target", ho_labels)
        self.assertIn("receptacle", ho_labels)
        ho_rec = next(e for e in ho_entries if e["label"] == "receptacle")
        self.assertEqual(ho_rec["receptacle"]["kind"], "support")
        self.assertIn("region", ho_rec["receptacle"])


class SharpaPrimitiveAndTactileTests(unittest.TestCase):
    def test_all_primitives_poses_and_frames(self):
        limits = np.tile(np.array([[-0.5, 1.8]], dtype=float), (pbs.HAND_DIM, 1))
        for prim in sxa.PRIMITIVES:
            open_q, pre_q, thumb_q, closed_q = xas.primitive_hand_poses(prim, 0.025, limits)
            for arr in (open_q, pre_q, thumb_q, closed_q):
                self.assertEqual(arr.shape, (pbs.HAND_DIM,))
                self.assertTrue(np.all(arr >= limits[:, 0]))
                self.assertTrue(np.all(arr <= limits[:, 1]))
            # Thumb-first intermediate matches closed thumb + preshape fingers
            np.testing.assert_allclose(thumb_q[: pbs.THUMB_COUNT], closed_q[: pbs.THUMB_COUNT])
            np.testing.assert_allclose(thumb_q[pbs.THUMB_COUNT :], pre_q[pbs.THUMB_COUNT :])
            for out in (-1.0, 1.0):
                rot, reach, grip_up = xas.primitive_grasp_frame(prim, out, 0.025, 0.08)
                self.assertAlmostEqual(float(np.linalg.det(rot.as_matrix())), 1.0, places=6)
                self.assertEqual(reach.shape, (3,))
                self.assertGreater(grip_up, 0.025)

    def test_tactile_closure_gate_opposing_contact_and_cap(self):
        limits = np.tile(np.array([[-0.5, 1.8]], dtype=float), (pbs.HAND_DIM, 1))
        gate = xas.TactileClosureGate("tripod_pinch", limits=limits, hold_steps=2)
        q0 = np.full(pbs.HAND_DIM, 0.5, dtype=float)
        # None forces -> no trigger
        self.assertIsNone(gate.update(None, q0))
        # Single-finger contact -> no trigger
        self.assertIsNone(gate.update([0.8, 0.0, 0.0, 0.0, 0.0], q0))
        # First opposing contact step -> not locked yet
        self.assertIsNone(gate.update([0.5, 0.5, 0.0, 0.0, 0.0], q0))
        # Second consecutive opposing contact step -> locks with preload
        locked = gate.update([0.5, 0.6, 0.0, 0.0, 0.0], q0)
        self.assertIsNotNone(locked)
        self.assertEqual(gate.trigger_reason, "opposing_contact")
        self.assertGreater(locked[0], q0[0])  # thumb preloaded
        self.assertAlmostEqual(locked[15], q0[15])  # tucked ring untouched in tripod_pinch

        # Force cap triggers immediately without adding preload
        gate_cap = xas.TactileClosureGate("power_wrap", limits=limits, contact_cap_n=5.0)
        locked_cap = gate_cap.update([0.0, 5.5, 0.0, 0.0, 0.0], q0)
        self.assertIsNotNone(locked_cap)
        self.assertEqual(gate_cap.trigger_reason, "force_cap")
        np.testing.assert_allclose(locked_cap, q0)

    def test_adaptive_waist_and_lift(self):
        w_center, l_center = xas.compute_wholebody_grasp_pose(0.75, (0.05, -0.40), np.array([0.0, 0.0]))
        self.assertAlmostEqual(w_center[0], 0.0)
        self.assertAlmostEqual(l_center[0], 0.0)
        # Robot stands at y=-0.73 facing +y with torso_Joint about +z:
        # +x (right edge) requires negative waist yaw; -x (left edge) requires positive waist yaw.
        w_right, l_high = xas.compute_wholebody_grasp_pose(0.85, (0.28, -0.42), np.array([0.0, 0.0]))
        self.assertLess(w_right[0], 0.0)
        w_left, _ = xas.compute_wholebody_grasp_pose(0.75, (-0.28, -0.42), np.array([0.0, 0.0]))
        self.assertGreater(w_left[0], 0.0)
        self.assertAlmostEqual(l_high[0], 0.10, places=5)
        # Farther across desk (ty=-0.30) must pitch forward MORE than closer edge (ty=-0.48)
        w_far, _ = xas.compute_wholebody_grasp_pose(0.75, (0.05, -0.30), np.array([0.0, 0.0]))
        w_near, _ = xas.compute_wholebody_grasp_pose(0.75, (0.05, -0.48), np.array([0.0, 0.0]))
        self.assertGreater(w_far[1], w_near[1])
        self.assertTrue(pbs.WAIST_LIMITS[0, 0] <= w_right[0] <= pbs.WAIST_LIMITS[0, 1])
        self.assertTrue(pbs.WAIST_LIMITS[1, 0] <= w_right[1] <= pbs.WAIST_LIMITS[1, 1])

    def test_handover_plan_vertical_and_horizontal_consistency(self):
        # Tall narrow bottle -> vertical upper/lower handover at x=0 with >= 55 mm z separation
        g_rot, g_reach, g_up = xas.primitive_grasp_frame("power_wrap", -1.0, 0.025, 0.14)
        t_rot, t_reach, t_up = xas.primitive_grasp_frame("power_wrap", 1.0, 0.025, 0.14)
        g_plan = dict(out=-1.0, hand_rot=g_rot, reach=g_reach, grip_up=g_up)
        t_plan = dict(out=1.0, hand_rot=t_rot, reach=t_reach, grip_up=t_up)
        tall = xas._handover_plan(
            np.array([-0.14, -0.40, 0.82]), Rotation.identity(), np.array([0.05, 0.05, 0.14]), 0.75, g_plan, t_plan
        )
        self.assertEqual(tall["mode"], "vertical")
        self.assertGreaterEqual(tall["t_ho_offset"][2] - tall["g_ho_offset"][2], 0.055)
        self.assertLess(tall["taker_grip_up"], 0.14)

        # Horizontal elongated bar rotated along y -> off-center giver grasp + yaw-aligned opposite-end taker grasp
        horiz = xas._handover_plan(
            np.array([-0.14, -0.40, 0.77]), Rotation.identity(), np.array([0.04, 0.12, 0.04]), 0.75, g_plan, t_plan
        )
        self.assertEqual(horiz["mode"], "horizontal")
        # Both hands separated by >= 0.06 m > CLASH=0.03 m, and giver grasped off-center along y
        self.assertGreaterEqual(float(np.linalg.norm(horiz["t_ho_offset"] - horiz["g_ho_offset"])), 0.06)
        self.assertGreater(abs(horiz["g_grip"][1] - (-0.40)), 0.025)

    def test_stage1_thumb_force_cap_still_closes_opposing_fingers(self):
        # High force on thumb only during stage 1 should freeze thumb and still close opposing fingers in stage 2
        tactile_seq = [np.array([[8.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])] + [np.zeros((5, 3))] * 40
        env, target_pos, _ = _make_fake_env(target_size=(0.05, 0.05, 0.12), tactile_seq=tactile_seq)
        orig_hand_arm = pbs.HandArm
        orig_robot_obj = xas.sd.Robot.object
        try:
            pbs.HandArm = _FakeHandArm
            xas.sd.Robot.object = staticmethod(
                lambda _self, _label: (
                    np.asarray(target_pos, dtype=float),
                    Rotation.identity(),
                    np.array([0.05, 0.05, 0.12]),
                    float(target_pos[2] - 0.06),
                    float(target_pos[2] + 0.06),
                )
            )
            cmds = list(xas.sharpa_pick_anything_actions(env))
        finally:
            pbs.HandArm = orig_hand_arm
            xas.sd.Robot.object = orig_robot_obj
        # Final left hand opposing index PIP joint [7] must be closed well past preshape (0.30)
        final_left_hand = np.asarray(cmds[-1]["left_gripper"], dtype=float)
        self.assertGreater(final_left_hand[7], 0.60)

    def test_grasp_plan_uses_rotated_world_x_half_width(self):
        env, target_pos, _ = _make_fake_env(target_size=(0.03, 0.08, 0.05))
        orig_robot_obj = xas.sd.Robot.object
        try:
            # Rotated 90 deg around z: 0.08 m side now lies along world x -> half_width = 0.04 m (not 0.015 m)
            xas.sd.Robot.object = staticmethod(
                lambda _self, _label: (
                    np.asarray(target_pos, dtype=float),
                    Rotation.from_euler("z", np.pi / 2),
                    np.array([0.03, 0.08, 0.05]),
                    0.75,
                    0.80,
                )
            )
            plan = xas._grasp_plan(env)
        finally:
            xas.sd.Robot.object = orig_robot_obj
        self.assertAlmostEqual(plan["half_width"], 0.04, places=5)


class SharpaExpertRolloutAndPluginTests(unittest.TestCase):
    def _run_and_check_61d_limits(self, generator_fn, target_size=(0.05, 0.05, 0.11), tactile_seq=None):
        env, target_pos, _ = _make_fake_env(target_size=target_size, tactile_seq=tactile_seq)
        orig_hand_arm = pbs.HandArm
        orig_robot_obj = xas.sd.Robot.object
        try:
            pbs.HandArm = _FakeHandArm
            xas.sd.Robot.object = staticmethod(
                lambda _self, _label: (
                    np.asarray(target_pos, dtype=float),
                    Rotation.identity(),
                    np.asarray(target_size, dtype=float),
                    float(target_pos[2] - target_size[2] / 2),
                    float(target_pos[2] + target_size[2] / 2),
                )
            )
            cmds = list(generator_fn(env))
        finally:
            pbs.HandArm = orig_hand_arm
            xas.sd.Robot.object = orig_robot_obj

        self.assertGreater(len(cmds), 50)
        expected_keys = {
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
            "waist_joint_state",
            "lift_joint_state",
        }
        prev = None
        for cmd in cmds:
            self.assertEqual(set(cmd.keys()), expected_keys)
            dim = sum(len(np.asarray(cmd[k]).reshape(-1)) for k in expected_keys)
            self.assertEqual(dim, 61)
            if prev is not None:
                for k in ("left_arm", "right_arm"):
                    dq = float(np.max(np.abs(np.asarray(cmd[k]) - np.asarray(prev[k]))))
                    self.assertLessEqual(dq, pbs.ARM_VMAX_LOADED / pbs.FPS + 1e-6)
                dw = float(np.max(np.abs(np.asarray(cmd["waist_joint_state"]) - np.asarray(prev["waist_joint_state"]))))
                self.assertLessEqual(dw, pbs.WAIST_VMAX / pbs.FPS + 1e-6)
                dl = float(np.max(np.abs(np.asarray(cmd["lift_joint_state"]) - np.asarray(prev["lift_joint_state"]))))
                self.assertLessEqual(dl, pbs.LIFT_VMAX / pbs.FPS + 1e-6)
            prev = cmd
        return cmds

    def test_all_four_tasks_emit_valid_rate_limited_61d_trajectories(self):
        # Test tactile-gated pick (power_wrap), place (tripod_pinch), put_in (lateral), and handover (both modes)
        tactile_seq = [np.zeros((5, 3))] * 10 + [
            np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ] * 5
        self._run_and_check_61d_limits(xas.sharpa_pick_anything_actions, (0.05, 0.05, 0.12), tactile_seq)
        self._run_and_check_61d_limits(xas.sharpa_place_anything_actions, (0.04, 0.04, 0.05))
        self._run_and_check_61d_limits(xas.sharpa_put_anything_in_actions, (0.04, 0.10, 0.03))
        self._run_and_check_61d_limits(xas.sharpa_handover_anything_actions, (0.05, 0.12, 0.08), tactile_seq)
        self._run_and_check_61d_limits(xas.sharpa_handover_anything_actions, (0.05, 0.05, 0.14), tactile_seq)

    def test_handover_sharpa_pad_forces(self):
        tactile_seq = [np.array([[3.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.5, 0.0, 0.0], [0, 0, 0], [0, 0, 0]])]
        env, _, _ = _make_fake_env(tactile_seq=tactile_seq)
        f = g1d_handover_anything_sharpa.sharpa_pad_forces(env, env.robot_manager.robot_list[0])
        np.testing.assert_allclose(f, [3.0, 2.5], atol=1e-5)
        env_none, _, _ = _make_fake_env(tactile_seq=None)
        f_none = g1d_handover_anything_sharpa.sharpa_pad_forces(env_none, env_none.robot_manager.robot_list[0])
        np.testing.assert_allclose(f_none, [0.0, 0.0], atol=1e-5)

    def test_plugins_early_abort_and_env_cfgs(self):
        self.assertIn("scripts.datagen.plugins.g1d_pick_anything_sharpa", early_abort.PICK)
        self.assertIn("scripts.datagen.plugins.g1d_place_anything_sharpa", early_abort.PLACE)
        self.assertIn("scripts.datagen.plugins.g1d_put_anything_in_sharpa", early_abort.PLACE)
        self.assertIn("scripts.datagen.plugins.g1d_handover_anything_sharpa", early_abort.HANDOVER)
        self.assertEqual(g1d_pick_anything_sharpa.TASK, "g1d_pick_anything")
        self.assertEqual(g1d_place_anything_sharpa.TASK, "g1d_place_anything")
        self.assertEqual(g1d_put_anything_in_sharpa.TASK, "g1d_put_anything_in")
        self.assertEqual(g1d_handover_anything_sharpa.TASK, "g1d_handover_anything")

        for name in ("g1d_sharpa_x_anything_wb.yml",):
            cfg = yaml.safe_load((ROOT / "env_cfg" / name).read_text())
            self.assertEqual(cfg["config"]["sim"], "g1d_sim_config")
            self.assertTrue(cfg["observation"]["robot"]["body_states"])
            self.assertEqual(cfg["observation"]["vision"]["cameras"], ["cam_head", "cam_head_right"])


if __name__ == "__main__":
    unittest.main()
