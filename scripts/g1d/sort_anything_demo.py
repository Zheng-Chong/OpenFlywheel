"""Scripted expert of g1d_sort_anything (按种类分拣任意物体), outside scripts/g1d/scripted_demo.py so the other tasks'
experts keep their fingerprints (scripts/datagen/experts.py): built from scripted_demo's robot, grasp and constants.

The items one by one, those for the receptacle farthest from the robot first, each by the hand on its receptacle's
side (sort_arm): g1d_pick_anything's grasp (the robot class's pick: v3's TopGripRobot, or pick expert v4.1's
GraspRobot41, which retries itself), tried `tries` times, then carried to its slot in its class's receptacle
as g1d_place_anything sets X down (place_held), at the slot's spot nearest the robot; the hand then waits above its own
side (PATTERN_ASIDE), in the background while the other hand starts on the next item. A grasp missed `tries` times
or a slip ends the script (the episode cannot be accepted).
"""

import numpy as np

from scripts.g1d.scripted_demo import (
    FINGER_TIPS,
    HOVER,
    PATTERN_ASIDE,
    PATTERN_ASIDE_ACTIONS,
    PATTERN_SETTLE,
    PICK_ANYTHING_LIFT,
    PLACE_CARRY,
    PLACE_DROP,
    PLACE_FIX,
    PLACE_GAP,
    PLACE_HOLD,
    PLACE_LIFTED,
    PLACE_SETTLE,
    PLACE_SLIPPED,
    PLACE_SPEED,
    SETTLED_SPEED,
    TURNED_RELEASE,
    TopGripRobot,
    as_np,
    from_wxyz,
    layout_entry,
    yaw_of,
)
from scripts.objects import pick_anything, place_anything, sort_anything

# an object within SORT_BLOCKED (m, footprint to footprint) of another still on the desk is picked after the free ones
# of its receptacle (its neighbour may stand in the way of the open fingers)
SORT_BLOCKED = 0.04
# v3's grasps of one item before the script gives up: a miss leaves the hand above the item, and pick opens it again on
# its way down to where the item now lies (with one try a single miss ends a layout of 3-4 items)
SORT_TRIES = 2
# m toward the robot from the receptacle the spot is drawn to: the hand lowers X on the receptacle's near side
# (lowered short of a spot on the far side, it pressed on the far rim and pushed the receptacle away, screen 8994)
NEAR = 1.0


def place_held(robot, grasp, label, receptacle, geom, start, hold=PLACE_HOLD):
    """scripted_demo.place_anything_actions' set-down of the object `label` held by `grasp` (arm, hand, offset, grip
    height, close command of pick()) on `geom` (its receptacle's geometry or a slot of it), at the allowed spot nearest
    the robot on the side X came from (`start`); held `hold` actions after the hand backs off. Returns False when it
    slipped on the way (given up, still held)."""
    arm, hand, offset, grip_height, close_cmd = grasp
    size = layout_entry(robot.env, label)["physics"]["size"]
    y_pos, y_quat = (as_np(v) for v in robot.lm.get_instance_pose(0, label=receptacle))
    y_rot = from_wxyz(y_quat)
    _, x_rot, _, x_bottom, _ = robot.object(label)
    rel = yaw_of(x_rot.as_matrix()) - yaw_of(y_rot.as_matrix())
    toward = y_rot.inv().apply([start[0] - y_pos[0], -NEAR, 0.0])[:2]
    spot = place_anything.spot_toward(geom, size, rel, toward)
    xy = (y_pos + y_rot.apply([spot[0], spot[1], 0.0]))[:2]
    face = y_pos[2] + y_rot.apply([0.0, 0.0, geom["floor"]])[2]
    top = y_pos[2] + y_rot.apply([0.0, 0.0, geom["top"]])[2]
    surface = face
    if geom["kind"] == "container":
        surface = max(face, top + PLACE_DROP + FINGER_TIPS - grip_height)
    hover = max(HOVER, top + PLACE_CARRY - surface, x_bottom - surface - PLACE_GAP)
    point = np.r_[xy - hand.apply(offset)[:2], surface + grip_height + PLACE_GAP]
    above = point + [0.0, 0.0, hover]
    carry = max(48, int(np.ceil(np.linalg.norm(above - arm.grasp_point()) / PLACE_SPEED * 30)))
    print(
        f"[sort_demo] {geom['kind']} {geom['id']}: spot {np.round(spot, 3)} (yaw {np.degrees(rel):.0f} deg off),"
        f" X bottom to {surface:.3f} (face {face:.3f}, top {top:.3f}), hover {hover:.3f}, carry {carry}"
    )
    yield from robot.move(arm, [(above, hand, close_cmd, carry)])
    robot.settle, before = None, robot.object(label)[0]
    for _ in range(PLACE_SETTLE):
        yield from robot.move(arm, [(above, hand, close_cmd, 1)])
        now = robot.object(label)[0]
        if np.linalg.norm(now - before) * 30 < SETTLED_SPEED:
            break
        before = now
    robot.settle = PATTERN_SETTLE
    held = arm.grasp_point() + arm.pose()[1].apply(offset)
    slipped = float(np.linalg.norm(now - held))
    if slipped > PLACE_SLIPPED:
        print(f"[sort_demo] {label} {slipped * 1000:.0f} mm off where it sat in the hand: slipped, giving up")
        return False
    fix = xy - now[:2]
    fix *= min(1.0, PLACE_FIX / max(float(np.linalg.norm(fix)), 1e-9))
    print(f"[sort_demo] {label} {np.linalg.norm(xy - now[:2]) * 1000:.0f} mm off over the spot, moved {fix.round(3)}")
    keys = [
        (point, hand, close_cmd, 30),
        (point, hand, TURNED_RELEASE, 12),  # release
        (point + [0, 0, hover], hand, TURNED_RELEASE, 24),
    ]
    yield from robot.move(arm, [(p + [*fix, 0.0], r, g, n) for p, r, g, n in keys])
    if hold:
        yield from robot.move(arm, [(arm.grasp_point(), arm.pose()[1], TURNED_RELEASE, hold)])
    end = robot.object(label)[0]
    print(f"[sort_demo] {label} at {end.round(3)}, receptacle at {robot.object(receptacle)[0].round(3)}")
    return True


def sort_slots(robot, items):
    """{item: geometry of its slot in its receptacle} (sort_anything.assign_slots: the items of one receptacle in
    slots along its long side, by where they lie), at their yaws as they lie."""
    by = {}
    for x, y in items:
        by.setdefault(y, []).append(x)
    out = {}
    for y, xs in by.items():
        geom = layout_entry(robot.env, y)["receptacle"]
        y_pos, y_quat = (as_np(v) for v in robot.lm.get_instance_pose(0, label=y))
        y_rot = from_wxyz(y_quat)
        parts = []
        for x in xs:
            centre, x_rot, _, _, _ = robot.object(x)
            rel = yaw_of(x_rot.as_matrix()) - yaw_of(y_rot.as_matrix())
            size = layout_entry(robot.env, x)["physics"]["size"]
            parts.append((size, rel, tuple(y_rot.inv().apply(centre - y_pos)[:2])))
        slots = sort_anything.assign_slots(geom, geom["kind"], parts)
        if slots is None:  # turned since the layout: slots by where they lie, the spots fall back to slot centres
            subs = sort_anything.split_region(geom, len(xs))
            axis = sort_anything.long_axis(geom)
            ranked = sorted(range(len(xs)), key=lambda i: parts[i][2][axis])
            slots = [subs[ranked.index(i)] for i in range(len(xs))]
        out.update(zip(xs, slots))
    return out


def sort_arm(robot, item, receptacle):
    """The hand that carries `item` to `receptacle`: the one on the receptacle's side (sort_anything.side_of), or on
    the item's side for a receptacle in the middle."""
    side = sort_anything.side_of(robot.object(receptacle)[0][0])
    return robot.arm_for(side * 0.1 if side else robot.object(item)[0][0])


def sort_next(robot, left, target, far):
    """The next item of `left` to pick: those for the receptacle farthest from the robot (`far`: receptacle -> its
    rank, 0 the farthest) first, so X is never carried past a receptacle already filled; of them those no other item
    still on the desk crowds (SORT_BLOCKED), then the one nearest the hand that carries it (sort_arm)."""

    def footprint(label):
        centre, rot, _, _, _ = robot.object(label)
        w, d = layout_entry(robot.env, label)["physics"]["size"][:2]
        return (centre[0], centre[1], w / 2, d / 2, yaw_of(rot.as_matrix()))

    def cost(x):
        centre = robot.object(x)[0]
        blocked = any(not place_anything.apart(footprint(x), footprint(o), SORT_BLOCKED) for o in left if o != x)
        reach = float(np.linalg.norm(centre[:2] - sort_arm(robot, x, target[x]).grasp_point()[:2]))
        return far[target[x]], blocked, reach

    return min(left, key=cost)


def carry(robot, x, receptacle, slot, start, arm, hold, tries):
    """Picks item `x` with `arm` (`tries` times at most) and sets it into its `slot` of `receptacle` (place_held).
    Returns whether it was set down."""
    before = robot.object(x)[0]
    for attempt in range(tries):
        hand_x = np.copysign(before[0], robot.home[arm.arm_key][0][0])
        prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(hand_x, before[1])))
        arm, hand, offset, grip_height, _, close_cmd = yield from robot.pick(
            x, lift=PICK_ANYTHING_LIFT, close=True, prefer=prefer, arm=arm
        )
        lifted = bool(robot.object(x)[0][2] - before[2] >= PLACE_LIFTED)
        robot.env.sort_picks.append(dict(item=x, receptacle=receptacle, tries=attempt + 1, lifted=lifted))
        if lifted:
            break
        print(f"[sort_demo] {x} not lifted: missed (try {attempt + 1} of {tries})")
        before = robot.object(x)[0]
    else:
        return False
    grasp = (arm, hand, offset, grip_height, close_cmd)
    return (yield from place_held(robot, grasp, x, receptacle, slot, start, hold=hold))


def sort_anything_actions(env, robot_class=TopGripRobot, tries=SORT_TRIES, carry=carry):
    """g1d_sort_anything: the module docstring's script, each item picked and set down by `carry` (its arguments).
    Each item's pick is recorded in env.sort_picks (item, its class's receptacle, tries made, lifted)."""
    robot = robot_class(env)
    env.sort_picks = []
    robot.settle = PATTERN_SETTLE
    items = env.items(0)
    target = dict(items)
    starts = {x: robot.object(x)[0] for x, _ in items}
    table = min(robot.object(x)[3] for x, _ in items)
    slots = sort_slots(robot, items)
    ys = sorted(set(target.values()), key=lambda y: -robot.object(y)[0][1])  # env +Y: away from the robot
    far = {y: i for i, y in enumerate(ys)}
    print(f"[sort_demo] sort {len(items)} items: {items}, receptacles far to near {ys}")
    left = [x for x, _ in items]
    while left:
        x = sort_next(robot, left, target, far)
        arm = sort_arm(robot, x, target[x])
        if arm.arm_key in robot.background:  # still on its way aside: get there first
            yield from robot.finish()
        left.remove(x)
        placed = yield from carry(robot, x, target[x], slots[x], starts[x], arm, 0 if left else PLACE_HOLD, tries)
        if not placed:
            return
        if left:
            aside_x, aside_y, up = PATTERN_ASIDE
            side = np.sign(robot.home[arm.arm_key][0][0])
            robot.send_aside(arm, np.array([side * aside_x, aside_y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()
