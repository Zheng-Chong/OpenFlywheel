"""scripts/datagen plugin for G1-D Sharpa Wave 61D wholebody g1d_handover_anything (把 X 交给另一只手再放到 Y 上).

Uses `scripts.dexhands.x_anything_sharpa.sharpa_handover_anything_actions` and native Sharpa fingertip elastomer
tactile contact sensors (`get_tactile_contact`) for the handover `pad_forces` check (mapped to [thumb_N, max_opp_N]).

Collect with:
    --task g1d_handover_anything --plugin g1d_handover_anything_sharpa --env-cfg g1d_office_sharpa_x_anything_wb
"""

import numpy as np

from scripts.datagen.plugins import g1d_handover_anything as base
from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.dexhands import x_anything_sharpa as expert
from scripts.g1d import scripted_demo as demo

TASK = base.TASK
KIND = base.KIND
ACCEPTANCE = "handover_anything_sharpa_v1"
STEP_LIM = 1800  # 60 s at 30 Hz: giver grasp + mid-air handover + taker place + release
TAIL = base.TAIL
HOLD = base.HOLD
audit = base.audit

SHARPA_HAND_LINKS = (
    "elbow_link",
    "wrist_roll_link",
    "wrist_pitch_link",
    "wrist_yaw_link",
    "hand_flange",
    "thumb_fingertip",
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "pinky_fingertip",
    "thumb_elastomer",
    "index_elastomer",
    "middle_elastomer",
    "ring_elastomer",
    "pinky_elastomer",
)


def sharpa_hand_bodies(env):
    """[(arm, [body_idx, ...]) per target arm] using Sharpa Wave link names, in robot_list order."""
    out, sides = [], []
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        arm = demo.Arm(env, robot)
        side = "left" if "left" in robot.gripper_name else "right"
        names = arm.key.body_names
        ids = [names.index(f"{side}_{n}") for n in SHARPA_HAND_LINKS if f"{side}_{n}" in names]
        out.append((arm, ids))
        sides.append(side)
    return out, sides


def sharpa_pad_forces(env, robot):
    """Return [thumb_N, max_opposing_finger_N] from native Sharpa fingertip elastomer contact sensors."""
    f = expert.read_fingertip_forces(env, robot)
    if f is None:
        return np.zeros(2, dtype=np.float32)
    return np.array([float(f[0]), float(np.max(f[1:]))], dtype=np.float32)


class Episode(base.Episode):
    task, kind = TASK, KIND

    def __init__(self, env):
        super().__init__(env)
        bodies, hands = sharpa_hand_bodies(env)
        if all(ids for _, ids in bodies):
            self.bodies = bodies
            self.meta["hands"] = hands

    def actions(self):
        return expert.sharpa_handover_anything_actions(self.env)

    def step(self):
        poses = self.poses()
        opening = np.asarray(gripper_openness(self.env), dtype=np.float32)
        forces = np.asarray([sharpa_pad_forces(self.env, arm.robot) for arm, _ in self.bodies], dtype=np.float32)
        gap = base.hand_gap(self.bodies)
        for key, value in (("object_poses", poses), ("gripper_openness", opening), ("pad_forces", forces)):
            self.history[key].append(np.asarray(value, dtype=np.float32))
        self.history["hand_gap"].append(np.float32(gap))
        self.fail = self.tracker.update(self.frame, poses, opening, forces, gap)
        self.frame += 1
        self.last = self.fail is None
        return self.last


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
