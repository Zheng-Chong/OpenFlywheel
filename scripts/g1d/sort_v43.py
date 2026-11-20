"""Sort anything expert v4.3: sort_anything_demo's order, hands, slots and spots (each item at its slot's spot nearest
the robot) with place anything expert v4.3's grasp and place (place_v43, docs/pick-expert-v4.md「v4.3」), item by
item. Before an item's first grasp, place_v43's Route replays on the URDF the route of each grasp of the hand that
carries it (sort_arm) through the lift to the set-down at its own slot's spot, and the first that reaches both the
grip and the set-down is taken: into a container a slender item deep first (as Put), onto a support v3's first (as
Place); none: v3's. The carry and set-down are place_v42's (ik_within, set down by where the item sits in the hand,
one that fell out of the hand on the way grasped again once). An item not lifted after pick's own tries, or fallen
off the desk, ends the script (the episode cannot be accepted). Kept out of sort_anything_demo so its v3 / v4.1
experts stay as they are.
"""

import numpy as np

from scripts.g1d import place_v42, place_v43, scripted_demo as demo, sort_anything_demo
from scripts.objects import pick_anything


class SlotSpot(place_v43.Spot):
    """place_v43.Spot for `item`'s slot `geom` of `receptacle`: the spot nearest the robot on the side the item came
    from (`start`), as sort_anything_demo.place_held."""

    def __init__(self, robot, item, receptacle, geom, start):
        self.geom = geom
        self.size = demo.layout_entry(robot.env, item)["physics"]["size"]
        pos, quat = (demo.as_np(v) for v in robot.lm.get_instance_pose(0, label=receptacle))
        self.y_pos, self.y_rot = pos, demo.from_wxyz(quat)
        self.face = pos[2] + self.y_rot.apply([0.0, 0.0, geom["floor"]])[2]
        self.top = pos[2] + self.y_rot.apply([0.0, 0.0, geom["top"]])[2]
        self.toward = self.y_rot.inv().apply([start[0] - pos[0], -sort_anything_demo.NEAR, 0.0])[:2]


class SortRobot43(place_v43.PlaceRobot43):
    """PlaceRobot43 whose routed item and spot are the ones `aim` set (place_v43 routes "target" to the receptacle)."""

    def __init__(self, env):
        place_v42.PlaceRobot.__init__(self, env)
        self.spot, self.item, self.routing = None, None, False
        env.place_route, env.place_log = [], []

    def aim(self, item, receptacle, geom, start):
        self.item, self.spot = item, SlotSpot(self, item, receptacle, geom, start)
        self.slender_deep = geom["kind"] == "container"

    def pick(self, label, *args, **kwargs):
        self.routing = label == self.item
        n = len(self.env.place_route)
        out = yield from place_v42.PlaceRobot.pick(self, label, *args, **kwargs)
        for r in self.env.place_route[n:]:
            r["item"] = label
        return out

    def route(self, cand, lift):
        """place_v43.PlaceRobot43.route for the aimed item."""
        a, hand, grip, _, _ = cand
        centre, x_rot, _, bottom, _ = self.object(self.item)
        r = place_v43.Route(self.reach[a.arm_key])
        hover = grip + [0.0, 0.0, lift]
        q, _ = r.move(a.q(), [(hover, hand, 0.0, 54)], place_v43.ARM_IK_MARGIN)
        q, off_grip = r.move(q, [(grip, hand, 0.0, self.DESCEND)], place_v43.ARM_IK_MARGIN)
        if off_grip > place_v43.ROUTE_TOL:
            return off_grip, 1.0, 1.0  # m: not replayed further
        q, _ = r.move(q, [(hover, hand, 0.0, 30)], place_v43.ARM_IK_MARGIN)
        offset = hand.inv().apply(centre - grip)
        point, above = self.spot.set_down(self.spot.xy(x_rot), hand, offset, grip[2] - bottom, bottom + lift)
        carry = max(48, int(np.ceil(np.linalg.norm(above - hover) / demo.PLACE_SPEED * 30)))
        q, off_above = r.move(q, [(above, hand, 0.0, carry)], place_v42.PLACE_JOINT_MARGIN)
        _, off_point = r.move(q, [(point, hand, 0.0, 30)], place_v42.PLACE_JOINT_MARGIN)
        return off_grip, off_above, off_point


def carry(robot, x, receptacle, slot, start, arm, hold, tries=None):
    """sort_anything_demo.carry with v4.3's grasp choice and place_v42's carry, set-down and regrasp (`tries`: pick's
    own). Returns whether `x` was set down."""
    env = robot.env
    first = robot.object(x)[0]
    for _ in range(1 + place_v42.PLACE_REGRASPS):
        robot.placing = False
        lying = robot.object(x)[0]
        if lying[2] < first[2] - place_v42.PLACE_GONE:
            print(f"[sort_demo] {x} off the desk ({lying.round(3)}): not grasped again")
            return False
        robot.aim(x, receptacle, slot, start)
        hand_x = np.copysign(lying[0], robot.home[arm.arm_key][0][0])
        prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(hand_x, lying[1])))
        n = len(env.grasp_attempts)
        arm, hand, offset, grip_height, open_cmd, close_cmd = yield from robot.pick(
            x, lift=demo.PICK_ANYTHING_LIFT, close=True, prefer=prefer, arm=arm
        )
        lifted = bool(robot.object(x)[0][2] - lying[2] >= demo.PLACE_LIFTED)
        env.sort_picks.append(dict(item=x, receptacle=receptacle, tries=len(env.grasp_attempts) - n, lifted=lifted))
        if not lifted:
            print(f"[sort_demo] {x} not lifted: missed, giving up")
            return False
        robot.placing = True
        _, x_rot, _, x_bottom, _ = robot.object(x)
        spot = robot.spot
        xy = spot.xy(x_rot)
        _, above = spot.set_down(xy, hand, offset, grip_height, x_bottom)
        n_carry = max(48, int(np.ceil(np.linalg.norm(above - arm.grasp_point()) / demo.PLACE_SPEED * 30)))
        print(f"[sort_demo] {slot['kind']} {slot['id']}: {x} to {xy.round(3)}, carry {n_carry}")
        yield from robot.move(arm, [(above, hand, close_cmd, n_carry)])
        robot.settle, before = None, robot.object(x)[0]
        now = before
        for _ in range(demo.PLACE_SETTLE):
            yield from robot.move(arm, [(above, hand, close_cmd, 1)])
            now = robot.object(x)[0]
            if np.linalg.norm(now - before) * 30 < demo.SETTLED_SPEED:
                break
            before = now
        robot.settle = demo.PATTERN_SETTLE
        slipped = float(np.linalg.norm(now - (arm.grasp_point() + arm.pose()[1].apply(offset))))
        if slipped > place_v42.PLACE_HELD or demo.jaw_opening(env, arm) < demo.GRASP_EMPTY:
            print(f"[sort_demo] {x} {slipped * 1000:.0f} mm off where it sat in the hand: fell out")
            yield from robot.move(arm, [(arm.grasp_point() + [0, 0, 0.03], arm.pose()[1], open_cmd, 20)])
            continue
        centre, x_rot, _, x_bottom, _ = robot.object(x)
        point, rot = arm.grasp_point(), arm.pose()[1]
        point, above2 = spot.set_down(xy, hand, rot.inv().apply(centre - point), point[2] - x_bottom, x_bottom)
        short = float(np.linalg.norm(arm.grasp_point() - above))
        env.place_log.append(dict(item=x, hand_short_mm=round(short * 1000), slipped_mm=round(slipped * 1000)))
        print(f"[sort_demo] {x} {slipped * 1000:.0f} mm off in the hand, hand {short * 1000:.0f} mm off over the spot")
        keys = [
            (above2, hand, close_cmd, 18),
            (point, hand, close_cmd, 30),
            (point, hand, demo.TURNED_RELEASE, 12),  # release
            (above2, hand, demo.TURNED_RELEASE, 24),
        ]
        yield from robot.move(arm, keys)
        if hold:
            yield from robot.move(arm, [(arm.grasp_point(), arm.pose()[1], demo.TURNED_RELEASE, hold)])
        end, rec = robot.object(x)[0], robot.object(receptacle)[0]
        print(f"[sort_demo] {x} at {end.round(3)}, {receptacle} at {rec.round(3)}")
        return True
    return False


def sort_anything_actions(env):
    """g1d_sort_anything with expert v4.3 (module docstring)."""
    return sort_anything_demo.sort_anything_actions(env, SortRobot43, carry=carry)
