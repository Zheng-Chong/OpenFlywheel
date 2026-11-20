"""scripts/datagen plugin: the grasp exam of g1d_pour_beans_plus (scripts/datagen/grasp_exam.py), recorded with
`collect.py --task g1d_pour_beans_plus --plugin g1d_pour_beans_plus_grasp`. The task's expert picks both cups and
pours as for the dataset, then ends with both cups still held (no putting back). Each frame also records the
holding hands' poses and gripper commands; the quality file gets each cup's grasp outcome ("grasp"), its cup id and
the grasp entry it was held by (grasp_exam.grip_name).
Not a dataset: acceptance grasp_v1 only says the expert ran to its end.

An episode ends early once a cup has clearly left its hand (EARLY_DROP_M from where it sat after the lift) or fell
over before it was grasped (EARLY_TILT_DEG), so a lost cup does not run the pour to the step limit.
"""

import json
from pathlib import Path

import numpy as np

from scripts.datagen import grasp_exam
from scripts.datagen.plugins import g1d_pour_beans_plus as plus
from scripts.g1d import scripted_demo as demo

TASK = plus.TASK
ACCEPTANCE = "grasp_v1"
STEP_LIM = plus.STEP_LIM
TAIL = HOLD = 15
LABELS = plus.LABELS
EARLY_DROP_M, EARLY_TILT_DEG = 0.08, 60.0


def hand_pose(arm):
    pos, rot = arm.pose()
    return np.r_[pos, np.roll(rot.as_quat(), 1)]


class Episode(plus.Episode):
    def __init__(self, env):
        super().__init__(env)
        arms = [demo.Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        source = min(arms, key=lambda a: abs(a.grasp_point()[0] - self.start[0, 0]))  # as the expert picks
        self.arms = (source, next(a for a in arms if a is not source))
        layout = json.loads(Path(env.seed_manager.seed_info[env.current_env_seed_map[0]]["scene_layout"]).read_text())
        gen = layout.get("Generalization", {})
        self.ids, self.scales = gen.get("cups", {}), gen.get("scale", {})
        entries = {e["label"]: e for es in layout.get("Rigid", {}).values() for e in es}
        self.grips = {
            label: grasp_exam.grip_name(demo.role_grip(entries.get(label, {}).get("grasp"), label.split("_")[0]))
            for label in LABELS
        }
        self.history.update(hand_poses=[], closed=[])
        self.command, self.done, self.stopped = None, False, None
        self.lifted = [None, None]  # per cup: its distance from the hand once lifted

    def actions(self):
        for action in demo.plus_actions(self.env, place=False):
            self.command = action
            yield action
        self.done = True

    def step(self):
        """After each action: the task's records plus both hands and grippers; True once the script has ended."""
        super().step()
        hands = np.array([hand_pose(a) for a in self.arms])
        closed = [float(self.command[a.grip_key][0]) < 0.01 if self.command else False for a in self.arms]
        self.history["hand_poses"].append(hands.astype(np.float32))
        self.history["closed"].append(closed)
        cups = self.history["cup_poses"][-1].astype(float)
        frame = len(self.history["closed"]) - 1
        for i, label in enumerate(LABELS):
            gap = float(np.linalg.norm(cups[i, :3] - hands[i, :3]))
            if not closed[i] and grasp_exam.tilt_deg(cups[i : i + 1])[0] > EARLY_TILT_DEG:
                self.stop(f"{label} fell over before it was grasped at frame {frame}")
            elif closed[i] and self.lifted[i] is None:
                start = next(k for k, c in enumerate(self.history["closed"]) if c[i])
                if hands[i, 2] - self.history["hand_poses"][start][i, 2] >= grasp_exam.LIFT_M:
                    self.lifted[i] = gap
            elif self.lifted[i] is not None and gap - self.lifted[i] > EARLY_DROP_M:
                self.stop(f"{label} left its hand at frame {frame}")
        return self.done

    def stop(self, reason):
        if self.stopped is None:
            self.stopped = reason
            self.env.success[0] = False

    def grasp(self):
        if not self.history["closed"]:
            return {
                label: dict(cup=self.ids.get(label), grip=self.grips[label], outcome="missed", frame=None)
                for label in LABELS
            }
        cups = np.asarray(self.history["cup_poses"], float)
        hands = np.asarray(self.history["hand_poses"], float)
        closed = np.asarray(self.history["closed"], bool)
        return {
            label: dict(
                cup=self.ids.get(label),
                scale=self.scales.get(label),
                grip=self.grips[label],
                **grasp_exam.classify(cups[:, i], hands[:, i], closed[:, i]),
            )
            for i, label in enumerate(LABELS)
        }

    def summary(self):
        return dict(super().summary(), grasp=self.grasp(), cup_ids=self.ids, stopped=self.stopped)

    def privileged(self):
        return dict(
            super().privileged(),
            hand_poses=np.asarray(self.history["hand_poses"], np.float32).reshape(-1, 2, 7),
            closed=np.asarray(self.history["closed"], bool).reshape(-1, 2),
        )
