"""Shared collection plugin of the atomic block tasks (scripts/objects/block_atomic.py); each task's plugin module
(g1d_<task>.py) takes TASK and STEP_LIM from here and adds its own. The scripted expert is
scripted_demo.DEMOS[task], through its Limiter.

Acceptance block_atomic_v1, the task's check made tighter, held for HOLD frames in a row (hands open: both grippers
open at least OPEN; STAY: where it lay at the start):
    place_block_on_mark   the block upright on the table within PLACE_RADIUS of the mark's centre; hands open
    stack_block_on_block  the top block's centre within STACK_XY of the base's, its bottom within STACK_Z of the
                          base's top, upright; the base within STAY; hands open
    pick_named_block      the named block's bottom at least LIFT above the table; the others within STAY
    put_block_in_tray     the named block's centre in the tray's cavity (footprint less the walls) and below its rim,
                          the others outside its footprint, the tray within STAY; hands open
    rotate_block_parallel the block's long side within ROTATE_DEG of X, upright on the table, within ROTATE_STAY;
                          hands open
    line_up_blocks        block_0..2 in a row left to right, neighbours ROW_GAP apart, y within ROW_Y, upright;
                          hands open
"""

import numpy as np

from env.reward_manager import block_checks
from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.g1d import scripted_demo as demo
from scripts.objects import block_patterns, block_sorting

ACCEPTANCE = "block_atomic_v1"
TAIL = 60  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
OPEN = 0.8
PLACE_RADIUS, STACK_XY, STACK_Z, LIFT, STAY = 0.015, 0.010, 0.006, 0.06, 0.02
ROTATE_DEG, ROTATE_STAY, ROW_GAP, ROW_Y, TILT = 6.0, 0.05, (0.035, 0.065), 0.010, 6.0
CAVITY = np.array([s / 2 - block_sorting.TRAY_WALL for s in block_sorting.BOX_SIZE[:2]])  # m, half sizes inside


def accepted(task, poses, openness, meta):
    """Does `task`'s acceptance hold for {label: (pos, rot)}, the grippers' openness and the episode's meta (sizes,
    table_z, start)?"""
    size, table, start = meta["sizes"], meta["table_z"], meta["start"]
    hands = min(openness) >= OPEN

    def near(label, tol):
        return float(np.linalg.norm(poses[label][0][:2] - np.asarray(start[label]))) < tol

    def upright(label):
        return block_checks.tilt_deg(poses[label][1]) < TILT

    if task == "g1d_place_block_on_mark":
        pos, rot = poses["block"]
        return hands and block_checks.placed_on(pos, rot, size["block"], poses["mark"][0], table, PLACE_RADIUS, TILT)
    if task == "g1d_stack_block_on_block":
        top, rot = poses["top_block"]
        on = block_checks.stacked_on(top, rot, size["top_block"], poses["base_block"][0], size["base_block"], STACK_XY,
                                     STACK_Z, TILT)  # fmt: skip
        return hands and on and near("base_block", STAY)
    if task == "g1d_pick_named_block":
        up = block_checks.lifted(poses["target_block"][0], size["target_block"], table, LIFT)
        return up and all(near(b, STAY) for b in meta["blocks"] if b != "target_block")
    if task == "g1d_put_block_in_tray":
        tray_pos, tray_rot = poses["tray"]

        def local(label):
            return np.asarray(tray_rot).T @ (poses[label][0] - tray_pos)

        inside = (
            np.all(np.abs(local("target_block")[:2]) <= CAVITY)
            and local("target_block")[2] < block_sorting.BOX_SIZE[2] / 2
        )
        out = all(np.any(np.abs(local(b)[:2]) > np.array(block_sorting.BOX_SIZE[:2]) / 2) for b in meta["blocks"]
                  if b != "target_block")  # fmt: skip
        return hands and bool(inside) and out and near("tray", STAY)
    if task == "g1d_rotate_block_parallel":
        pos, rot = poses["block"]
        resting = abs(pos[2] - table - size["block"][2] / 2) < 0.006
        aligned = block_checks.yaw_aligned(rot, 0.0, 180, ROTATE_DEG)
        return hands and aligned and resting and upright("block") and near("block", ROTATE_STAY)
    if task == "g1d_line_up_blocks":
        labels = ["block_0", "block_1", "block_2"]
        row = block_checks.in_row([poses[b][0] for b in labels], [poses[b][1] for b in labels], ROW_GAP, ROW_Y, TILT)
        return hands and row
    raise ValueError(task)


class Episode:
    def __init__(self, env):
        self.env, self.lm, self.task = env, env.scene_manager.layout_manager, env.task_name
        entries = env.entries(0)
        self.labels = sorted(entries)
        blocks = env.blocks(0)
        self.meta = dict(
            labels=self.labels,
            blocks=blocks,
            categories={k: entries[k]["category"] for k in self.labels},
            paints={k: entries[k]["category_idx"] for k in blocks},
            sizes={k: list(block_patterns.SIZES[entries[k]["category"]]) for k in blocks},
            table_z=float(env.table_z(0)),
            start={k: [float(v) for v in xy] for k, xy in env.start[0].items()},
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
        return (limiter(action) for action in demo.DEMOS[self.task](self.env))

    def step(self):
        """After each action: record every object and the grippers; True while the acceptance holds."""
        poses, openness = self.poses(), gripper_openness(self.env)
        self.history["object_poses"].append(poses.astype(np.float32))
        self.history["gripper_openness"].append(np.asarray(openness, dtype=np.float32))
        self.last = accepted(self.task, as_dict(self.labels, poses), openness, self.meta)
        return self.last

    def privileged(self):
        """Per frame after its action: every object's pose (T, n, 7: xyz, wxyz) in meta label order, gripper openness
        (T, 2)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        return dict(self.meta, final=bool(self.last))


def as_dict(labels, poses):
    return {
        k: (np.asarray(p[:3], float), demo.from_wxyz(np.asarray(p[3:], float)).as_matrix())
        for k, p in zip(labels, poses)
    }


def audit(privileged, quality):
    """Recompute the acceptance from the recorded poses and grippers in each of the last HOLD frames. Raises
    AssertionError on a violation."""
    poses, openness = privileged["object_poses"], privileged["gripper_openness"]
    assert poses.shape == (quality["frames"], len(quality["labels"]), 7), poses.shape
    assert len(openness) == quality["frames"]
    for frame in range(quality["frames"] - HOLD, quality["frames"]):
        ok = accepted(quality["task"], as_dict(quality["labels"], poses[frame]), openness[frame], quality)
        assert ok, (quality["task"], frame)
    return dict(task=quality["task"], frames_checked=HOLD)
