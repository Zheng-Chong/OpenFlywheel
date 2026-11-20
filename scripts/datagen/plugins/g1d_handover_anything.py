"""scripts/datagen plugin for g1d_handover_anything (把 X 交给另一只手再放到 Y 上). The scripted expert is
scripted_demo.handover_anything_actions (the giver's off-centre top grasp, the level carry to the handover, the
taker's top grasp on the other end, the giver letting go, the taker setting X on the support); through its Limiter.

Acceptance handover_anything_v2 (scripts/objects/handover_anything.py thresholds), held for HOLD frames in a row:
    lifted by the giver   X rose LIFTED with the giver (the hand on X's side) holding it and the taker not
    handed over           both hands held X (handover_anything.held: a pad presses, the jaw not shut) for HANDED
                          frames in a row with X AIR above where it lay
    never dropped         from the lift until X is on Y, at most LOST frames in a row with no hand holding X and X
                          less than AIR up; the taker holds X as it first comes onto Y
    no clash              the two arms' hand links (elbow to finger tips) stay CLASH apart, every frame
    on Y, let go, settled, Y and the others kept: g1d_place_anything's acceptance (place_anything_v1)
The first four are sticky: once broken (or before they hold) no frame is accepted. The sidecar keeps every object's
asset, X's and Y's names in both languages with the ones the instruction could use, the hands and Y's geometry.
"""

import numpy as np

from scripts.datagen.plugins import g1d_place_anything as place
from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.g1d import scripted_demo as demo
from scripts.objects import handover_anything as ho, place_anything as pa

TASK = ho.TASK
KIND = ho.KIND
ACCEPTANCE = "handover_anything_v2"  # v2: a hand holds X with one pad pressing (v1: both)
STEP_LIM = 1500  # actions: the task's
TAIL = 30  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
# the links whose origins stand for each arm's hand in the clash check
HAND_LINKS = (
    "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "dex1_base_link", "dex1_finger_link_1",
    "dex1_finger_link_2", "dex1_finger_link_1_tip", "dex1_finger_link_2_tip",
)  # fmt: skip


class Tracker:
    """The acceptance frame by frame: update() returns the first condition the frame misses, or None."""

    def __init__(self, labels, start, receptacle, x_size, hands, giver):
        self.labels, self.start = labels, np.asarray(start, dtype=float)
        self.receptacle, self.x_size = receptacle, x_size
        self.g, self.t = hands.index(giver), hands.index(ho.other(giver))
        self.x = labels.index("target")
        self.lifted = self.wrong = self.clash = self.dropped = self.handed = False
        self.run = self.lost = 0
        self.prev = None
        self.frames = dict(lifted=None, handed=None, placed=None)
        self.min_gap, self.most_lost = float("inf"), 0

    def update(self, frame, poses, opening, forces, gap):
        rise = poses[self.x, 2] - self.start[self.x, 2]
        hold = [ho.held(float(o), f) for o, f in zip(opening, forces)]
        self.min_gap = min(self.min_gap, float(gap))
        self.clash |= gap < ho.CLASH
        if not self.lifted and rise > ho.LIFTED:
            self.lifted, self.frames["lifted"] = True, frame
            self.wrong = not hold[self.g] or hold[self.t]
        x, y = poses[self.x], poses[self.labels.index(pa.LABEL)]
        on_y = pa.placed(KIND, self.receptacle, 1.0, x[:3], x[3:], self.x_size, y[:3], y[3:])
        if on_y and self.frames["placed"] is None and self.lifted:
            self.frames["placed"] = frame
            self.dropped |= not hold[self.t]  # fell onto Y rather than set down by the taker
        if self.lifted and self.frames["placed"] is None:
            self.lost = 0 if any(hold) or rise >= ho.AIR else self.lost + 1
            self.most_lost = max(self.most_lost, self.lost)
            self.dropped |= self.lost > ho.LOST
        self.run = self.run + 1 if all(hold) and rise > ho.AIR else 0
        if not self.handed and self.run >= ho.HANDED:
            self.handed, self.frames["handed"] = True, frame
        fail = place.failing(
            poses, self.labels, self.start, self.receptacle, self.x_size, opening, self.prev, KIND
        )  # fmt: skip
        self.prev = poses
        for flag, why in (
            (not self.lifted, "not_lifted"),
            (self.wrong, "wrong_hand"),
            (self.clash, "clash"),
            (self.dropped, "dropped"),
            (not self.handed, "not_handed"),
        ):
            if flag:
                return why
        return fail

    def record(self):
        return dict(stage_frames=self.frames, min_hand_gap=round(self.min_gap, 4), most_lost=self.most_lost)


def hand_bodies(env):
    """[(articulation, [body index, ...]) per target arm], the arms in gripper_openness order, and their sides."""
    out, sides = [], []
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        arm = demo.Arm(env, robot)
        side = "left" if "left" in robot.gripper_name else "right"
        names = arm.key.body_names
        out.append((arm, [names.index(f"{side}_{n}") for n in HAND_LINKS if f"{side}_{n}" in names]))
        sides.append(side)
    return out, sides


def hand_gap(bodies):
    """m between the nearest hand link origins of the two arms."""
    pts = [demo.as_np(arm.key.data.body_link_pose_w[0, ids, :3]).reshape(-1, 3) for arm, ids in bodies]
    return float(np.min(np.linalg.norm(pts[0][:, None] - pts[1][None], axis=-1)))


class Episode(place.Episode):
    task, kind = TASK, KIND

    def __init__(self, env):
        super().__init__(env)
        self.bodies, hands = hand_bodies(env)
        giver = ho.hand_of(float(self.meta["start"][self.labels.index("target")][0]))
        self.meta.update(hands=hands, giver=giver, taker=ho.other(giver))
        self.tracker = Tracker(self.labels, self.meta["start"], self.meta["receptacle"], self.meta["target_size"],
                               hands, giver)  # fmt: skip
        self.history.update(pad_forces=[], hand_gap=[])
        self.frame = 0

    def step(self):
        """After each action: every object, the grippers, their pads and the hands' gap; True while the acceptance
        holds."""
        poses = self.poses()
        opening = np.asarray(gripper_openness(self.env), dtype=np.float32)
        rm = self.env.robot_manager
        forces = [rm.get_gripper_force(arm.robot, [0]) for arm, _ in self.bodies]
        forces = np.asarray([np.zeros(2) if f is None else f[0] for f in forces], dtype=np.float32)
        gap = hand_gap(self.bodies)
        for key, value in (("object_poses", poses), ("gripper_openness", opening), ("pad_forces", forces)):
            self.history[key].append(np.asarray(value, dtype=np.float32))
        self.history["hand_gap"].append(np.float32(gap))
        self.fail = self.tracker.update(self.frame, poses, opening, forces, gap)
        self.frame += 1
        self.last = self.fail is None
        return self.last

    def privileged(self):
        """Per frame after its action: every object's pose (T, n, 7: xyz, wxyz) in meta label order, gripper openness
        (T, 2) and pad forces (T, 2, 2: N) in meta hands order, the hands' gap (T: m)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        return dict(super().summary(), **self.tracker.record())


def audit(privileged, quality):
    """Recompute the acceptance from the recorded frames (the sticky conditions from the first one), that it holds
    in each of the last HOLD frames, and that the instruction names X, Y and the taker. Raises AssertionError on a
    violation."""
    poses, labels = privileged["object_poses"], quality["labels"]
    n = quality["frames"]
    assert poses.shape == (n, len(labels), 7), poses.shape
    for key in ("gripper_openness", "pad_forces", "hand_gap"):
        assert len(privileged[key]) == n, key
    tracker = Tracker(labels, quality["start"], quality["receptacle"], quality["target_size"], quality["hands"],
                      quality["giver"])  # fmt: skip
    fails = [
        tracker.update(i, poses[i], privileged["gripper_openness"][i], privileged["pad_forces"][i],
                       privileged["hand_gap"][i])
        for i in range(n)
    ]  # fmt: skip
    assert all(f is None for f in fails[n - HOLD :]), [f for f in fails[n - HOLD :] if f]
    for key in ("usable_names", "usable_receptacle_names"):
        usable = [x for names in quality[key].values() for x in names]
        assert any(x in quality["instruction"] for x in usable), (key, quality["instruction"])
    taker = quality["taker"]
    assert any(ho.NAMES[lang][taker] in quality["instruction"] for lang in ho.NAMES), quality["instruction"]
    rec = tracker.record()
    t = labels.index("target")
    moved = float(np.linalg.norm(poses[-1, t, :2] - np.asarray(quality["start"])[t, :2]))
    return dict(task=quality["task"], frames_checked=n, carried_m=round(moved, 4), objects=len(labels), **rec)


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
