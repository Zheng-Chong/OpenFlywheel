"""Privileged joint-command expert; objects move only through simulated contact."""

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1d.scripted_demo import GRASP_ROT, Robot


def actions(env):
    robot = Robot(env)
    robot.settle = (0.010, 75)
    arm = next(a for a in robot.arms if a.arm_key.startswith("right"))
    food, _, ext, bottom, _ = robot.object("food")
    table = bottom - 0.001
    hand = Rotation.from_euler("x", -35, degrees=True) * GRASP_ROT
    point = np.r_[food[:2], bottom + 0.030]
    opening = float(np.clip((min(ext[:2]) + 0.025 - 0.006) / 0.089, 0, 1))
    yield from robot.move(
        arm,
        [
            (point + [0, -0.09, 0.08], hand, opening, 90),
            (point + [0, -0.09, 0], hand, opening, 40),
            (point, hand, opening, 55),
            (point, hand, 0, 30),
            (point + [0, 0, 0.17], hand, 0, 65),
        ],
    )
    if robot.object("food")[3] - table < 0.020:
        print("[pick_bucket] stop: food pickup failed", flush=True)
        return
    centre, _, _, basket_bottom, _ = robot.object("basket")
    # Use the held object's CURRENT lowest point, including contact-induced rotation.
    offset = robot.object("food")[0] - arm.grasp_point()
    drop = np.r_[centre[:2] - offset[:2], arm.grasp_point()[2] + basket_bottom + 0.125 - robot.object("food")[3]]
    yield from robot.move(arm, [(drop, hand, 0, 150)])
    if robot.object("food")[3] < table + 0.02:
        print("[pick_bucket] stop: food slipped during transfer", flush=True)
        return
    yield from robot.move(
        arm,
        [
            (drop + [0, 0, -0.05], hand, 0, 45),
            (drop + [0, 0, -0.05], hand, 1, 30),
            (drop + [0, 0, 0.15], hand, 1, 55),
            (drop + [0, -0.10, 0.15], hand, 1, 40),
        ],
    )
    for _ in range(45):
        yield dict(robot.hold) | robot.grip
    if env.progress[0].stage < 2:
        print("[pick_bucket] stop: food not contained and released", flush=True)
        return
    centre, rot, _, _, _ = robot.object("basket")
    point = centre + rot.apply([0, 0, 0.122])
    hand = Rotation.from_euler("z", 45, degrees=True) * Rotation.from_euler("x", -60, degrees=True) * GRASP_ROT
    yield from robot.move(
        arm,
        [
            (point + [0.07, -0.07, 0.05], hand, 0.65, 75),
            (point, hand, 0.65, 60),
            (point, hand, 0, 35),
            (point + [0, 0, 0.065], hand, 0, 65),
            (point + [0, 0, 0.065], hand, 0, 30),
            (point + [0, 0, 0.003], hand, 0, 65),
            (point + [0, 0, 0.003], hand, 1, 35),
            (point + [0, -0.07, 0.12], hand, 1, 55),
        ],
    )
    for _ in range(45):
        yield dict(robot.hold) | robot.grip
