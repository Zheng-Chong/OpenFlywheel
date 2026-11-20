"""scripts/datagen plugin for g1d_sort_blocks_by_color (按卡片分拣积木，按颜色); g1d_sort_blocks_by_shape reuses it. The
scripted expert drops each block into the tray whose card matches it (scripted_demo.block_sort_actions, through its
Limiter).

Acceptance sorted_v1, the task's check (each block inside its tray's footprint and below its rim, both hands open) made
tighter and held for HOLD frames in a row: every block's centre inside its tray's cavity (the footprint less the walls,
in the tray's frame) and below the rim, the trays within TRAY_MOVE / TRAY_TILT_DEG of where they stood at the start
(neither dragged nor tipped), the cards within CARD_MOVE of theirs, and both grippers open at least OPEN.
"""

import numpy as np

from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.g1d import scripted_demo as demo
from scripts.objects import block_sorting

TASK = "g1d_sort_blocks_by_color"
ACCEPTANCE = "sorted_v1"
STEP_LIM = 1800  # actions: the task's 60 s
TAIL = 60  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
TRAY_MOVE, TRAY_TILT_DEG, CARD_MOVE = 0.02, 5.0, 0.01  # m, deg, m from the start
OPEN = 0.8  # gripper opening (0 closed .. 1 open), as the task's check
BOXES, CARDS = ("box_0", "box_1"), ("card_0", "card_1")
CAVITY = np.array([s / 2 - block_sorting.TRAY_WALL for s in block_sorting.BOX_SIZE[:2]])  # m, half sizes inside


def sorted_state(blocks, trays, cards, start, tray_of):
    """(blocks in their cavities, worst tray move m, worst tray tilt deg, worst card move m) of poses (n, 7: xyz, wxyz);
    `start`: the trays' and cards' poses at the start, `tray_of`: each block's tray index."""
    blocks, trays, cards = (np.asarray(p, dtype=float) for p in (blocks, trays, cards))
    inside = True
    for pose, t in zip(blocks, tray_of):
        local = demo.from_wxyz(trays[t, 3:]).inv().apply(pose[:3] - trays[t, :3])
        inside &= bool(np.all(np.abs(local[:2]) <= CAVITY) and local[2] < block_sorting.BOX_SIZE[2] / 2)
    t0, c0 = np.asarray(start["trays"], dtype=float), np.asarray(start["cards"], dtype=float)
    move = float(np.max(np.linalg.norm(trays[:, :2] - t0[:, :2], axis=1)))
    tilt = max(
        float(np.degrees(np.arccos(np.clip(demo.from_wxyz(q).apply([0, 0, 1])[2], -1, 1)))) for q in trays[:, 3:]
    )
    card = float(np.max(np.linalg.norm(cards[:, :2] - c0[:, :2], axis=1)))
    return inside, move, tilt, card


def accepted(state, openness):
    inside, move, tilt, card = state
    return inside and move < TRAY_MOVE and tilt < TRAY_TILT_DEG and card < CARD_MOVE and min(openness) >= OPEN


class Episode:
    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        groups = env.groups[0]
        self.labels = sorted(label for box in BOXES for label in groups[box])
        self.tray_of = [next(i for i, box in enumerate(BOXES) if label in groups[box]) for label in self.labels]
        self.groups = {box: list(groups[box]) for box in BOXES}
        self.start = dict(trays=self.poses(BOXES).tolist(), cards=self.poses(CARDS).tolist())
        self.reject = None
        self.last = None
        self.history = dict(block_poses=[], tray_poses=[], card_poses=[], gripper_openness=[])

    def poses(self, labels):
        return np.array(
            [np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=label)]) for label in labels]
        )

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.block_sort_actions(self.env))

    def step(self):
        """After each action: record blocks, trays, cards and grippers; True while all sorted, trays and cards in
        place, hands open."""
        blocks, trays, cards = self.poses(self.labels), self.poses(BOXES), self.poses(CARDS)
        openness = gripper_openness(self.env)
        for key, value in zip(self.history, (blocks, trays, cards, openness)):
            self.history[key].append(np.asarray(value, dtype=np.float32))
        self.last = sorted_state(blocks, trays, cards, self.start, self.tray_of)
        return accepted(self.last, openness)

    def privileged(self):
        """Per frame after its action: block poses (T, 4, 7) in label order, tray and card poses (T, 2, 7), gripper
        openness (T, 2)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        out = dict(by=self.env.BY, groups=self.groups, labels=self.labels, tray_of=self.tray_of, start=self.start)
        if self.last:
            out["final"] = dict(zip(("sorted", "tray_move_m", "tray_tilt_deg", "card_move_m"), self.last))
        return out


def audit(privileged, quality):
    """Recompute the acceptance from the recorded poses and grippers in each of the last HOLD frames. Raises
    AssertionError on a violation; returns the worst tray / card moves over those frames."""
    n = len(quality["labels"])
    blocks, trays, cards = privileged["block_poses"], privileged["tray_poses"], privileged["card_poses"]
    openness = privileged["gripper_openness"]
    assert blocks.shape == (quality["frames"], n, 7), blocks.shape
    assert trays.shape == cards.shape == (quality["frames"], 2, 7), (trays.shape, cards.shape)
    assert len(openness) == quality["frames"]
    worst = np.zeros(3)
    for frame in range(quality["frames"] - HOLD, quality["frames"]):
        state = sorted_state(blocks[frame], trays[frame], cards[frame], quality["start"], quality["tray_of"])
        assert accepted(state, openness[frame]), (state, openness[frame])
        worst = np.maximum(worst, state[1:])
    keys = ("max_tray_move_m", "max_tray_tilt_deg", "max_card_move_m")
    return dict(by=quality["by"], **{k: float(v) for k, v in zip(keys, worst)})
