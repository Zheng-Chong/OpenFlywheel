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


if __name__ == "__main__":
    unittest.main()
