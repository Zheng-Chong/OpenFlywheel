"""CPU checks of scripts/datagen/early_abort.py and its use in collect.py: python -m pytest scripts/datagen"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen import collect, early_abort  # noqa: E402
from scripts.datagen.test_datagen import FakeEnv, Plugin  # noqa: E402
from scripts.objects.test_cup_profile import straight_cup  # noqa: E402

UP = [1.0, 0.0, 0.0, 0.0]
A, B = np.array([-0.15, -0.33, 0.8]), np.array([0.15, -0.33, 0.8])  # cup roots, the straight cup's at its centre
DESK = 0.8 - 0.045  # the straight cup is 9 cm tall


class PourBeansPlusTests(unittest.TestCase):
    def setUp(self):
        self.cup = straight_cup()
        self.cups = {label: (self.cup, 1.0) for label in early_abort.LABELS}
        self.poses = np.array([[*A, *UP], [*B, *UP]])
        self.check = early_abort.PourBeansPlus(self.cups, self.poses, [0.8, 0.8], 0.02)
        self.beans = early_abort.cup_profile.column(self.cup, 50) + B

    def frames(self, beans, n, touched=False, moved=0.0):
        return [self.check(beans, self.poses, touched, moved) for _ in range(n)]

    def test_settled_beans_and_flying_beans_never_end_it(self):
        flying = self.beans.copy()
        flying[:5] = [0.0, -0.33, 0.95]  # mid-air between the cups, above the desk
        self.assertEqual(self.frames(self.beans, 50) + self.frames(flying, 50), [None] * 100)

    def test_a_bean_on_the_desk_away_from_both_cups_ends_it_after_persist_frames(self):
        spilled = self.beans.copy()
        spilled[0] = [0.0, -0.15, DESK + 0.003]
        self.assertEqual(self.frames(spilled, early_abort.PERSIST - 1), [None] * (early_abort.PERSIST - 1))
        self.assertEqual(self.check(spilled, self.poses, False, 0.0), "beans_lost")

    def test_partial_in_target_share_waits_until_budget_is_exceeded(self):
        # 50 beans at 0.95 require >= 47.5 -> 48 in B: 2 lost is recoverable, 3 lost is permanently doomed
        check = early_abort.PourBeansPlus(self.cups, self.poses, [0.8, 0.8], 0.02, in_target_share=0.95)
        two_lost = self.beans.copy()
        two_lost[:2] = [0.0, -0.15, DESK + 0.003]
        self.assertEqual([check(two_lost, self.poses, False, 0.0) for _ in range(30)], [None] * 30)
        three_lost = two_lost.copy()
        three_lost[2] = [0.0, -0.15, DESK + 0.003]
        self.assertEqual(
            [check(three_lost, self.poses, False, 0.0) for _ in range(early_abort.PERSIST - 1)],
            [None] * (early_abort.PERSIST - 1),
        )
        self.assertEqual(check(three_lost, self.poses, False, 0.0), "beans_lost")

    def test_the_lost_count_restarts_when_the_bean_leaves_the_desk(self):
        spilled = self.beans.copy()
        spilled[0] = [0.0, -0.15, DESK + 0.003]
        self.frames(spilled, early_abort.PERSIST - 1)
        self.assertIsNone(self.check(self.beans, self.poses, False, 0.0))
        self.assertEqual(self.frames(spilled, early_abort.PERSIST - 1), [None] * (early_abort.PERSIST - 1))

    def test_a_bean_on_the_desk_next_to_a_cup_or_its_place_does_not_end_it(self):
        beside = self.beans.copy()
        beside[0] = B + [0.05, 0.0, DESK + 0.003 - B[2]]  # against B's wall
        self.assertEqual(self.frames(beside, 3 * early_abort.PERSIST), [None] * 3 * early_abort.PERSIST)
        lifted = self.poses.copy()
        lifted[1, :3] = [0.0, -0.2, 1.0]  # B up in the air: its place on the desk is still near the bean
        self.assertIsNone([self.check(beside, lifted, False, 0.0) for _ in range(3 * early_abort.PERSIST)][-1])

    def test_touching_and_a_moved_distractor_end_it_at_once(self):
        self.assertEqual(self.check(self.beans, self.poses, True, 0.0), "touched")
        self.assertEqual(self.check(self.beans, self.poses, False, 0.03), "distractor_moved")
        self.assertIsNone(self.check(self.beans, self.poses, False, 0.02))

    def test_replay_finds_the_frame_a_recorded_episode_would_end_at(self):
        n = 40
        particles = np.tile(self.beans, (n, 1, 1))
        particles[10:, 0] = [0.0, -0.15, DESK + 0.003]
        quality = dict(
            cups={label: {"cavity": self.cup, "scale": 1.0} for label in early_abort.LABELS},
            start_z=[0.8, 0.8],
            moved_limit_m=0.02,
        )
        privileged = dict(
            particles=particles,
            cup_poses=np.tile(self.poses, (n, 1, 1)),
            distractor_poses=np.zeros((n, 0, 7)),
            distractor_poses_start=np.zeros((0, 7)),
        )
        self.assertEqual(early_abort.replay(quality, privileged), (10 + early_abort.PERSIST, "beans_lost"))
        privileged["particles"] = np.tile(self.beans, (n, 1, 1))
        self.assertIsNone(early_abort.replay(quality, privileged))
        touching = privileged["cup_poses"].copy()
        touching[7, 0, :3] = B - [0.069, 0, 0]
        self.assertEqual(early_abort.replay(quality, dict(privileged, cup_poses=touching)), (8, "touched"))


class CollectTests(unittest.TestCase):
    def record(self, doomed, early=True, to_end=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "layout.json").write_text("{}")
            env, plugin = FakeEnv(root / "layout.json"), Plugin()
            plugin.RECORD_TO_END = to_end
            cfg = dict(env_cfg="cfg", fps=30, revision=None, trajectory=False, early_abort=early)
            with patch.object(early_abort, "for_episode", return_value=doomed) as hook:
                collect.record_episode(env, plugin, root, cfg)
            return json.loads((root / "episode_000003.json").read_text()), len(env.actions), hook.called

    def test_a_doomed_episode_ends_rejected_with_its_reason(self):
        quality, actions, _ = self.record(lambda ep: "touched")
        self.assertEqual((quality["accepted"], quality["frames"], actions), (False, 1, 1))
        self.assertEqual(quality["aborted"], dict(reason="touched", frame=1))

    def test_an_episode_nothing_dooms_is_recorded_as_without_the_checks(self):
        plain, _, _ = self.record(None)
        checked, _, _ = self.record(lambda ep: None)
        off, _, called = self.record(lambda ep: "touched", early=False)
        self.assertFalse(called)
        self.assertEqual(plain, checked)
        self.assertEqual(plain, off)
        self.assertEqual((plain["accepted"], plain["frames"]), (True, 7))
        self.assertNotIn("aborted", plain)

    def test_a_frame_that_passes_the_acceptance_is_never_ended(self):
        # the fake plugin settles from its third step on: a check that fires from then on must not end it
        quality, _, _ = self.record(lambda ep: "beans_lost" if ep.steps >= 3 else None)
        self.assertEqual((quality["accepted"], quality["frames"]), (True, 7))
        self.assertNotIn("aborted", quality)


class XAnythingEarlyAbortTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["distractor_0", "receptacle", "target"]
        self.start = np.array(
            [
                [-0.2, -0.3, 0.75, 1.0, 0.0, 0.0, 0.0],
                [0.15, -0.3, 0.75, 1.0, 0.0, 0.0, 0.0],
                [-0.1, -0.3, 0.75, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def test_handover_sticky_flags_abort_immediately(self):
        check = early_abort.HandoverAnything(self.labels, self.start)
        self.assertIsNone(check(self.start))
        self.assertEqual(check(self.start, wrong=True), "wrong_hand")
        self.assertEqual(check(self.start, clash=True), "clash")
        self.assertEqual(check(self.start, dropped=True), "dropped")

    def test_handover_receptacle_and_distractor_require_persistence(self):
        check = early_abort.HandoverAnything(self.labels, self.start, persist=3)
        moved_rec = self.start.copy()
        moved_rec[1, 0] += 0.10  # pushed receptacle
        self.assertIsNone(check(moved_rec))
        self.assertIsNone(check(moved_rec))
        self.assertIsNone(check(self.start))  # transient bump resets counter
        self.assertIsNone(check(moved_rec))
        self.assertIsNone(check(moved_rec))
        self.assertEqual(check(moved_rec), "receptacle_moved")

        check2 = early_abort.HandoverAnything(self.labels, self.start, persist=2)
        moved_dist = self.start.copy()
        moved_dist[0, 0] += 0.10  # >= ABORT_STAY (0.08)
        self.assertIsNone(check2(moved_dist))
        self.assertEqual(check2(moved_dist), "distractor_moved")

    def test_pick_grasp_retry_cap_and_distractor_moved(self):
        labels = ["distractor_0", "target"]
        start = self.start[[0, 2], :3]
        poses = self.start[[0, 2]]
        check = early_abort.PickAnything(labels, start, max_grasp_tries=2, persist=2)
        self.assertIsNone(check(poses, []))
        self.assertIsNone(check(poses, [dict(attempt=0, result="shut on nothing (0.01)")]))
        self.assertIsNone(check(poses, [dict(attempt=0, result="shut on nothing (0.01)"), dict(attempt=1)]))
        self.assertIsNone(
            check(poses, [dict(attempt=0, result="shut on nothing (0.01)"), dict(attempt=1, result="held")])
        )
        self.assertEqual(
            check(
                poses,
                [
                    dict(attempt=0, result="shut on nothing (0.01)"),
                    dict(attempt=1, result="did not rise (5 mm, jaw 0.10)"),
                ],
            ),
            "grasp_retry_cap",
        )
        # 3rd attempt started even if 1st held and dropped
        self.assertEqual(
            check(poses, [dict(attempt=0, result="held"), dict(attempt=1, result="held"), dict(attempt=2)]),
            "grasp_retry_cap",
        )
        moved_dist = poses.copy()
        moved_dist[0, 2] -= 0.06  # fallen below start - ABORT_FALL_Z
        self.assertIsNone(check(moved_dist, []))
        self.assertEqual(check(moved_dist, []), "distractor_moved")

    def test_place_and_put_grasp_retry_cap_receptacle_and_distractor(self):
        check = early_abort.PlaceAnything(self.labels, self.start, max_grasp_tries=2, persist=2, tilt_persist=3)
        self.assertIsNone(check(self.start, [dict(attempt=0, result="grip missed")]))
        self.assertEqual(
            check(self.start, [dict(attempt=0, result="grip missed"), dict(attempt=1, result="did not rise")]),
            "grasp_retry_cap",
        )
        # Brief rocking / moderate tilt (< 75 deg) must not false-abort
        rocked = self.start.copy()
        rocked[1, 3:] = [0.9659, 0.2588, 0.0, 0.0]  # ~30 deg tilt
        for _ in range(10):
            self.assertIsNone(check(rocked, []))
        tipped_rec = self.start.copy()
        tipped_rec[1, 3:] = [0.7071, 0.7071, 0.0, 0.0]  # 90 deg tilt
        self.assertIsNone(check(tipped_rec, []))
        self.assertIsNone(check(tipped_rec, []))
        self.assertEqual(check(tipped_rec, []), "receptacle_moved")

    def test_default_pick_and_place_do_not_false_abort_on_third_grasp_or_regrasp(self):
        # GraspRobot.GRASP_TRIES=3 and place_v42.PLACE_REGRASPS=1 legitimately produce accepted episodes on try 3.
        third_try_ok = [
            dict(attempt=0, result="grip missed (77 mm beside the target, -2 mm high)"),
            dict(attempt=1, result="grip missed (251 mm beside the target, 248 mm high)"),
            dict(attempt=2, result="held"),
        ]
        regrasp_ok = [dict(attempt=0, result="held"), dict(attempt=1, result="held"), dict(attempt=2)]
        pick_check = early_abort.PickAnything(["distractor_0", "target"], self.start[[0, 2], :3])
        place_check = early_abort.PlaceAnything(self.labels, self.start)
        self.assertIsNone(pick_check(self.start[[0, 2]], third_try_ok))
        self.assertIsNone(place_check(self.start, third_try_ok))
        self.assertIsNone(place_check(self.start, regrasp_ok))

    def test_for_episode_wires_all_x_anything_plugins(self):
        from types import ModuleType, SimpleNamespace

        for mod_name in (*early_abort.PICK, *early_abort.PLACE, *early_abort.HANDOVER):
            mod = ModuleType(mod_name)
            is_pick = mod_name in early_abort.PICK
            base_poses = self.start[[0, 2]] if is_pick else self.start
            ep = SimpleNamespace(
                labels=["distractor_0", "target"] if is_pick else self.labels,
                meta=dict(start=(self.start[[0, 2], :3] if is_pick else self.start).tolist()),
                history=dict(object_poses=[base_poses]),
                env=SimpleNamespace(grasp_attempts=[]),
                tracker=SimpleNamespace(wrong=False, clash=True, dropped=False),
            )
            fn = early_abort.for_episode(mod, ep)
            self.assertIsNotNone(fn, mod_name)
            if mod_name in early_abort.HANDOVER:
                self.assertEqual(fn(ep), "clash")
            else:
                self.assertIsNone(fn(ep))
                # Two failed grasp attempts must NOT false-abort (3rd try can still succeed)
                ep.env.grasp_attempts = [dict(result="miss"), dict(result="miss")]
                self.assertIsNone(fn(ep))
                # Sustained unrecoverable distractor displacement must abort after PERSIST frames
                moved = base_poses.copy()
                moved[0, 0] += early_abort.ABORT_STAY + 0.02
                ep.history["object_poses"].append(moved)
                for _ in range(early_abort.PERSIST - 1):
                    self.assertIsNone(fn(ep))
                self.assertEqual(fn(ep), "distractor_moved")

    def test_scan_one_replays_x_anything_from_npz_privileged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "episode_000000.json"
            n = 25
            poses = np.tile(self.start, (n, 1, 1))
            poses[5:, 0, 0] += 0.10  # distractor moved >= ABORT_STAY from frame 5 onward
            quality = dict(
                task="g1d_place_anything",
                accepted=False,
                frames=n,
                labels=self.labels,
                start=self.start.tolist(),
            )
            sidecar.write_text(json.dumps(quality))
            collect.write_privileged(sidecar.with_suffix(".npz"), dict(object_poses=poses))
            row = early_abort.scan_one(sidecar)
        self.assertEqual(row["abort_reason"], "distractor_moved")
        self.assertEqual(row["abort_frame"], 5 + early_abort.PERSIST)

    def test_sort_anything_receptacle_moved_and_item_fallen(self):
        labels = ["item_0", "item_1", "rec_0", "rec_1"]
        items = [("item_0", "rec_0"), ("item_1", "rec_1")]
        receptacles = {"rec_0": {"kind": "support"}, "rec_1": {"kind": "container"}}
        start = np.array(
            [
                [-0.15, -0.30, 0.78, 1.0, 0.0, 0.0, 0.0],
                [0.15, -0.30, 0.78, 1.0, 0.0, 0.0, 0.0],
                [-0.15, -0.45, 0.75, 1.0, 0.0, 0.0, 0.0],
                [0.15, -0.45, 0.75, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        check = early_abort.SortAnything(labels, items, receptacles, start, persist=3)
        self.assertIsNone(check(start))
        moved_rec = start.copy()
        moved_rec[2, 0] += 0.10
        self.assertIsNone(check(moved_rec))
        self.assertIsNone(check(moved_rec))
        self.assertIsNone(check(start))  # resets counter
        self.assertIsNone(check(moved_rec))
        self.assertIsNone(check(moved_rec))
        self.assertEqual(check(moved_rec), "receptacle_moved")

        check2 = early_abort.SortAnything(labels, items, receptacles, start, persist=3)
        # Tall item tipping onto the desk (z drops 8 cm) must NOT false-abort
        tipped_on_desk = start.copy()
        tipped_on_desk[1, 2] -= 0.08
        for _ in range(10):
            self.assertIsNone(check2(tipped_on_desk))
        fallen = start.copy()
        fallen[1, 2] -= 0.25  # off the desk (below min(start_z) - 0.15)
        self.assertIsNone(check2(fallen))
        self.assertIsNone(check2(fallen))
        self.assertIsNone(check2(start))  # resets counter
        self.assertIsNone(check2(fallen))
        self.assertIsNone(check2(fallen))
        self.assertEqual(check2(fallen), "item_fallen")

    def test_for_episode_and_scan_one_wire_sort_anything(self):
        from types import ModuleType, SimpleNamespace

        labels = ["item_0", "rec_0"]
        items = [["item_0", "rec_0"]]
        receptacles = {"rec_0": {"kind": "support"}}
        start = np.array(
            [[-0.1, -0.3, 0.78, 1.0, 0.0, 0.0, 0.0], [0.1, -0.4, 0.75, 1.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        for mod_name in early_abort.SORT:
            mod = ModuleType(mod_name)
            ep = SimpleNamespace(
                labels=labels,
                meta=dict(items=items, receptacles=receptacles, start=start.tolist()),
                history=dict(object_poses=[start]),
            )
            fn = early_abort.for_episode(mod, ep)
            self.assertIsNotNone(fn, mod_name)
            self.assertIsNone(fn(ep))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "episode_000000.json"
            n = 25
            poses = np.tile(start, (n, 1, 1))
            poses[4:, 0, 2] -= 0.25  # item fell off desk from frame 4 onward
            quality = dict(
                task="g1d_sort_anything",
                accepted=False,
                frames=n,
                labels=labels,
                items=items,
                receptacles=receptacles,
                start=start.tolist(),
            )
            sidecar.write_text(json.dumps(quality))
            collect.write_privileged(sidecar.with_suffix(".npz"), dict(object_poses=poses))
            row = early_abort.scan_one(sidecar)
        self.assertEqual(row["abort_reason"], "item_fallen")
        self.assertEqual(row["abort_frame"], 4 + early_abort.PERSIST)


if __name__ == "__main__":
    unittest.main()
