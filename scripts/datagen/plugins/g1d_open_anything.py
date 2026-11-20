"""scripts/datagen plugin for g1d_open_anything (打开任意铰接物体): the scripted expert is
scripted_demo.open_anything_actions (take the handle or edge the pool's method names, follow the part's arc or slide,
hold it open),
through its Limiter.

Acceptance open_anything_v3, held for HOLD frames in a row: the target joint at least its goal past closed (the
layout's "open": 40 deg for a door or lid, 8 cm for a drawer, at most 60% of the joint's travel); every other openable
joint of the object within SIBLING_DEG / SIBLING_M of where it started (only the named part opened); and the hand
opened it: until a hand's grip point first came within TOUCH of the handle the part stayed under START_SHARE of the
goal (a door that falls open by itself is no success). An episode whose part is not closed at the start (more than
START_SHARE of the goal open once the scene settled) is rejected. v2 (11-18): TOUCH 3 -> 5 cm; a pinch holds a
handle with the pads' middle 1-2 cm in front of its centre, so v1 failed parts the hand opened (screen8: 15). v3
(11-18 evening): and the target joint never jumps more than JUMP from one frame to the next. The sidecar keeps the
object, the part, its names and the opening trace summary, so a dataset can be relabelled at any name level.
"""

import math

import numpy as np

from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.g1d import scripted_demo as demo

TASK = "g1d_open_anything"
ACCEPTANCE = "open_anything_v3"
STEP_LIM = 600  # actions: the task's
TAIL = 30  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
SIBLING_DEG, SIBLING_M = 10.0, 0.03
START_SHARE = 0.15
TOUCH = 0.05  # m between a hand's grip point and the handle's centre
# the target joint moves at most this much from one frame to the next (rad, m): a basket or door the physics threw
# out (an air-fryer basket 38-41 mm in one frame and back, a toaster-oven door 22 deg; pilot 910-00) is no
# demonstration
JUMP = {"revolute": math.radians(10.0), "prismatic": 0.015}


def by_hand(values, gaps, part):
    """Did the part stay under START_SHARE of its goal until a hand first came within TOUCH of the handle (per frame:
    joint `values` (T, n), the target first; `gaps` (T,))?"""
    opening = part["sign"] * (np.asarray(values)[:, 0] - part["closed"])
    near = np.flatnonzero(np.asarray(gaps) <= TOUCH)
    first = near[0] if len(near) else len(opening)
    return bool(np.all(opening[:first] <= START_SHARE * part["goal"]))


def jumped(values, part):
    """Did the target joint (`values` (T, n), the target first) move more than JUMP in one frame?"""
    trace = np.asarray(values)[:, 0]
    return bool(len(trace) > 1 and np.abs(np.diff(trace)).max() > JUMP[part["joint_type"]])


def accepted(values, start, part):
    """Does the acceptance hold for the joint `values` (the target first, then the part's siblings), against their
    `start`?"""
    if part["sign"] * (values[0] - part["closed"]) < part["goal"]:
        return False
    for value, value0, revolute in zip(values[1:], start[1:], part["revolute"][1:]):
        if abs(value - value0) >= (math.radians(SIBLING_DEG) if revolute else SIBLING_M):
            return False
    return True


class Episode:
    def __init__(self, env):
        self.env = env
        lm = env.scene_manager.layout_manager
        self.inst = lm.get_scene_object(0, lm.get_instance_name(0, "target"))
        part = env.target_part(0)
        others = [j for j in part["siblings"] if j != part["joint"] and j in self.inst.dof_names]
        self.joints = [part["joint"], *others]
        info = {name: self.inst.get_joint_info(name) for name in self.joints}
        names = {lang: list(part["names"][lang]) for lang in ("zh", "en")}
        self.meta = dict(
            object=part["id"],
            joint=part["joint"],
            role=part["role"],
            part=part["part"],
            joint_type=part["type"],
            closed=part["closed"],
            sign=part["sign"],
            goal=part["goal"],
            revolute=[part["siblings"].get(j, part)["type"] == "revolute" for j in self.joints],
            joints=self.joints,
            object_names=names,
            part_name=part["part_name"],
            split=part.get("split", "train"),
            limits={k: [float(v["lower"]), float(v["upper"])] for k, v in info.items()},
        )
        self.meta["start"] = self.values().round(6).tolist()
        self.meta["opening_start"] = round(part["sign"] * (self.meta["start"][0] - part["closed"]), 5)
        self.reject = None
        if self.meta["opening_start"] > START_SHARE * part["goal"]:
            self.reject = f"part not closed at the start ({self.meta['opening_start']:.3f} open)"
        self.last = None
        self.history = dict(joint_values=[], gripper_openness=[], hand_gap=[])
        self.part = demo.OpenPart(env)
        self.arms = [demo.Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        self.handle = np.asarray(part["grasp"])
        self.alone = False  # the part opened before a hand got to it
        self.jumped = False  # the target joint jumped more than JUMP in a frame

    def values(self):
        return np.array([float(self.inst.get_joint_info(j)["position"]) for j in self.joints])

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.DEMOS[TASK](self.env))

    def step(self):
        """After each action: record the object's openable joints and the grippers; True while the acceptance
        holds."""
        values, openness = self.values(), np.asarray(gripper_openness(self.env), dtype=np.float32)
        handle = self.part.to_env(self.handle, values[0])
        gap = min(float(np.linalg.norm(a.grasp_point() - handle)) for a in self.arms)
        self.history["joint_values"].append(values.astype(np.float32))
        self.history["gripper_openness"].append(openness)
        self.history["hand_gap"].append(np.float32(gap))
        if not self.alone:
            self.alone = not by_hand(self.history["joint_values"], self.history["hand_gap"], self.meta)
        trace = self.history["joint_values"]
        if len(trace) > 1 and abs(float(trace[-1][0] - trace[-2][0])) > JUMP[self.meta["joint_type"]]:
            self.jumped = True
        self.last = accepted(values, self.meta["start"], self.meta) and not self.alone and not self.jumped
        return self.last

    def privileged(self):
        """Per frame after its action: the openable joints (T, n) in meta joints order, the target first, the
        gripper openness (T, 2) and the nearest hand's grip point to the handle (T,)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        trace = np.asarray(self.history["joint_values"])
        opened = self.meta["sign"] * (trace[:, 0] - self.meta["closed"]) if len(trace) else np.zeros(1)
        return dict(
            self.meta,
            final=bool(self.last),
            opening_final=round(float(opened[-1]), 5),
            opening_max=round(float(opened.max()), 5),
            opened_alone=self.alone,
            jumped=self.jumped,
        )


def audit(privileged, quality):
    """Recompute the acceptance from the recorded joints and grippers in each of the last HOLD frames, and that the
    instruction names the object. Raises AssertionError on a violation."""
    values = privileged["joint_values"]
    assert values.shape == (quality["frames"], len(quality["joints"])), values.shape
    assert len(privileged["gripper_openness"]) == quality["frames"]
    for frame in range(quality["frames"] - HOLD, quality["frames"]):
        assert accepted(values[frame], quality["start"], quality), frame
    assert by_hand(values, privileged["hand_gap"], quality), "the part opened before a hand got to it"
    assert not jumped(values, quality), "the part jumped in one frame"
    names = [n for lang in quality["object_names"].values() for n in lang]
    assert any(n in quality["instruction"] for n in names), quality["instruction"]
    opened = quality["sign"] * (values[-HOLD:, 0] - quality["closed"])
    return dict(task=quality["task"], frames_checked=HOLD, opening=round(float(opened.min()), 4), role=quality["role"])


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
