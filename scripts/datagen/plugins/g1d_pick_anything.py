"""scripts/datagen plugin for g1d_pick_anything (按名称拿起任意物体): the scripted expert is
scripted_demo.pick_anything_actions (top grasp across the target's narrow side, lift 13 cm), through its Limiter.

Acceptance pick_anything_v1, held for HOLD frames in a row: the target's centre at least LIFT above where it lay at the
start; every distractor's centre within STAY of its start horizontally and within STAY_Z of it vertically (not pushed,
knocked over or carried along). The sidecar keeps every object's asset (category, index) and the target's names in
both languages (category, then simple to detailed) with the ones the instruction could use (pick_anything.usable_names),
so a dataset can be relabelled at any name level.
"""

import numpy as np

from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.g1d import scripted_demo as demo

TASK = "g1d_pick_anything"
ACCEPTANCE = "pick_anything_v1"
STEP_LIM = 600  # actions: the task's
TAIL = 30  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
LIFT, STAY, STAY_Z = 0.08, 0.03, 0.02  # m


def accepted(poses, labels, start):
    """Does the acceptance hold for poses (n, 7: xyz, wxyz) in `labels` order, against the start positions?"""
    start = np.asarray(start, dtype=float)
    for i, label in enumerate(labels):
        move = poses[i, :3] - start[i]
        if label == "target":
            if move[2] < LIFT:
                return False
        elif np.linalg.norm(move[:2]) >= STAY or abs(move[2]) >= STAY_Z:
            return False
    return True


class Episode:
    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        entries = env.entries(0)
        self.labels = sorted(entries)
        names = {lang: self.lm.get_label_descriptions(0, "target", lang=lang) for lang in ("zh", "en")}
        self.meta = dict(
            labels=self.labels,
            objects={k: [entries[k]["category"], int(entries[k]["category_idx"])] for k in self.labels},
            target_names=names,
            usable_names={lang: env.names(0, lang) for lang in ("zh", "en")},
            start=self.poses()[:, :3].round(5).tolist(),
        )
        self.reject = None
        self.last = None
        self.history = dict(object_poses=[], gripper_openness=[])

    def poses(self):
        return np.array(
            [np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=k)]) for k in self.labels]
        )

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.DEMOS[TASK](self.env))

    def step(self):
        """After each action: record every object and the grippers; True while the acceptance holds."""
        poses = self.poses()
        self.history["object_poses"].append(poses.astype(np.float32))
        self.history["gripper_openness"].append(np.asarray(gripper_openness(self.env), dtype=np.float32))
        self.last = accepted(poses, self.labels, self.meta["start"])
        return self.last

    def privileged(self):
        """Per frame after its action: every object's pose (T, n, 7: xyz, wxyz) in meta label order, gripper openness
        (T, 2)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        return dict(self.meta, final=bool(self.last))


def audit(privileged, quality):
    """Recompute the acceptance from the recorded poses in each of the last HOLD frames, and that the instruction
    names the target by a usable name. Raises AssertionError on a violation."""
    poses, labels = privileged["object_poses"], quality["labels"]
    assert poses.shape == (quality["frames"], len(labels), 7), poses.shape
    assert len(privileged["gripper_openness"]) == quality["frames"]
    for frame in range(quality["frames"] - HOLD, quality["frames"]):
        assert accepted(poses[frame], labels, quality["start"]), frame
    usable = [n for names in quality["usable_names"].values() for n in names]
    assert any(n in quality["instruction"] for n in usable), quality["instruction"]
    target = labels.index("target")
    lift = float(poses[-HOLD:, target, 2].min() - quality["start"][target][2])
    return dict(task=quality["task"], frames_checked=HOLD, lift_m=round(lift, 4), objects=len(labels))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
