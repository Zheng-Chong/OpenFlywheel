"""Handover anything expert v3: v2's handover (scripted_demo.handover_anything_actions: the giver's off-centre top
grasp, the level carry, the taker's top grasp on the other end, the giver letting go) with place v4.3's place stage for
the taker (scripts/g1d/place_v42.py, docs/pick-expert-v4.md「v4.2」「v4.3」). Kept out of scripted_demo.py so that the
registered experts' sources stay as they are.

- Once the giver has let go, the taker's moves step with place_v42.ik_within: Arm.ik switches a joint off only 0.025
  off its limit, inside the Limiter's clamp, so a clamped joint kept being asked to move and the arm stalled short of
  the spot (Place / Put: 23-42% of placements more than 6 cm short).
- Over the support, once X has stopped swinging (and v2's leveling), the set-down is planned from where X sits in the
  hand then; it is given up only when X fell out of the hand (PLACE_HELD off it, or the jaw shut on nothing). v2 gave
  up past PLACE_SLIPPED and else moved the set-down by at most PLACE_FIX.

Place v4.3's grasp choice (the route replayed on the URDF) is not taken: the giver's grasp is fixed by the handover (an
end of X, the fingers pointing at the taker) and the taker's by the giver's. A dropped X is not grasped again (a second
handover would not fit the 1500 actions).
"""

import zlib

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1d import place_v42, scripted_demo as demo
from scripts.objects import handover_anything, pick_anything, place_anything


class HandoverRobot(demo.TopGripRobot):
    """TopGripRobot whose moves step with place_v42.ik_within once `placing`."""

    placing = False

    def _command(self, arm, point, rot, grip):
        if not self.placing:
            return super()._command(arm, point, rot, grip)
        self.hold[arm.arm_key] = place_v42.ik_within(arm, point, rot)
        self.grip[arm.grip_key] = [float(grip)]
        self._advance(arm.arm_key)
        if self.step % 20 == 0:
            print(f"[scripted_demo] {self.step:3d} {arm.arm_key} err {np.linalg.norm(arm.grasp_point() - point):.3f}")
        self.step += 1
        return dict(self.hold) | self.grip


def handover_anything_actions(env):
    """g1d_handover_anything with expert v3 (module docstring)."""
    robot = HandoverRobot(env)
    robot.settle = demo.PATTERN_SETTLE
    entry = demo.layout_entry(env, "target")
    rng = np.random.default_rng(zlib.crc32(repr(entry["default_pos"]).encode()))
    start, rot0, ext, bottom0, top0 = robot.object("target")
    side = 1.0 if start[0] > 0 else -1.0
    giver = robot.arm_for(start[0])
    taker = next(a for a in robot.arms if a is not giver)
    toward = np.array([-side, 0.0, 0.0])  # giver -> taker
    shift = handover_anything.shift(entry["physics"]["size"])
    axes = rot0.as_matrix()
    up = int(np.argmax(np.abs(axes[2])))
    narrow = min((a for a in range(3) if a != up), key=lambda a: ext[a])
    long_ax = 3 - up - narrow
    # the giver: pick anything's grasp, shifted along the long side away from the fingertips
    yaw = np.arctan2(axes[1, narrow], axes[0, narrow])
    want = np.radians(pick_anything.grip_yaw(start[0], start[1]))
    hand = Rotation.from_euler("z", yaw + np.pi * round((want - yaw) / np.pi)) * demo.TOP_ROT
    fingers = hand.apply([1.0, 0.0, 0.0])[:2]
    fingers /= np.linalg.norm(fingers)
    grip = np.r_[start[:2] - shift * fingers, robot.grip_z(bottom0, top0)]
    open_cmd = robot.opening(ext[narrow])
    print(
        f"[scripted_demo] handover v3: {giver.arm_key} takes target at {start.round(3)} size {ext.round(3)}"
        f" {shift * 1000:.0f} mm off centre, {taker.arm_key} takes over"
    )
    yield from robot.move(
        giver,
        [
            (grip + [0, 0, demo.PICK_ANYTHING_LIFT], hand, open_cmd, 54),
            (grip, hand, open_cmd, robot.DESCEND),
            (grip, hand, 0.0, 18),
            (grip + [0, 0, demo.HO_LIFT], hand, 0.0, 30),
        ],
    )
    centre = robot.object("target")[0]
    if centre[2] - start[2] < demo.PLACE_LIFTED:
        print("[scripted_demo] target not lifted: missed, giving up")
        return
    offset = giver.pose()[1].inv().apply(centre - giver.grasp_point())
    # the handover
    g = max(rng.uniform(*demo.HO_GRIP), grip[2] - bottom0 + demo.HO_AIR)
    meet = np.array([rng.uniform(-demo.HO_X, demo.HO_X), rng.uniform(*demo.HO_Y)])
    hand_h = demo.top_hand(toward)
    grip_h = np.r_[meet - hand_h.apply(offset)[:2], bottom0 + g]
    carry = max(60, int(np.ceil(np.linalg.norm(grip_h - giver.grasp_point()) / demo.PLACE_SPEED * 30)))
    carry = max(carry, demo.turn_actions(hand, hand_h))
    print(f"[scripted_demo] handover at {meet.round(3)}, grips {g:.3f} above the desk, carry {carry}")
    yield from robot.move(
        giver, [(np.r_[giver.grasp_point()[:2], grip_h[2]], hand, 0.0, 30), (grip_h, hand_h, 0.0, carry)]
    )
    now = yield from demo.settle_in(robot, giver, grip_h, hand_h)
    slipped = float(np.linalg.norm(now - giver.grasp_point() - giver.pose()[1].apply(offset)))
    if slipped > demo.PLACE_SLIPPED:
        print(f"[scripted_demo] target {slipped * 1000:.0f} mm off where it sat in the giver: slipped, giving up")
        return
    # the taker: the other end from the top, fingers pointing back at the giver
    for attempt in range(demo.HO_TRIES):
        centre, rot, ext_now, _, _ = robot.object("target")
        m = rot.as_matrix()
        u = m[:, long_ax] * np.sign(m[:, long_ax] @ toward)
        upv = m[:, up] * np.sign(m[2, up])
        point = centre + shift * u + upv * (ext_now[up] / 2 - min(ext_now[up] / 2, demo.HO_DEPTH))
        out = np.r_[u[:2] / np.linalg.norm(u[:2]), 0.0]
        hand_t = demo.top_hand(-out[:2])
        open_t = robot.opening(ext_now[narrow])
        print(f"[scripted_demo] taker try {attempt}: grip {point.round(3)}")
        yield from robot.move(
            taker,
            [
                (point + [0, 0, demo.HO_HOVER] + demo.HO_OUT * out, hand_t, open_t, 75),
                (point, hand_t, open_t, 45),
                (point, hand_t, 0.0, 18),
            ],
        )
        yield from robot.move(taker, [(point, hand_t, 0.0, 6)])
        opening, forces = demo.jaw_state(env, taker)
        print(f"[scripted_demo] taker jaw {opening:.2f}, pads {forces.round(1)} N")
        if handover_anything.held(opening, forces):
            break
        yield from robot.move(
            taker,
            [(point, hand_t, open_t, 12), (point + [0, 0, demo.HO_HOVER] + demo.HO_OUT * out, hand_t, open_t, 30)],
        )
    else:
        print("[scripted_demo] taker never held the target: giving up")
        return
    # the giver lets go and backs off
    yield from robot.move(taker, [(point, hand_t, 0.0, demo.HO_WAIT)])
    yield from robot.move(giver, [(grip_h, hand_h, demo.TURNED_RELEASE, 15)])
    yield from robot.move(taker, [(point, hand_t, 0.0, demo.HO_WAIT)])
    yield from robot.move(giver, [(grip_h + [0, 0, demo.HO_RISE], hand_h, demo.TURNED_RELEASE, 20)])
    robot.send_aside(giver, grip_h + demo.HO_PARK * [side, 1.0, 1.0], 45)
    # the taker sets X on the support: place v4.2's place stage
    robot.placing = True
    geom = demo.layout_entry(env, place_anything.LABEL)["receptacle"]
    size = entry["physics"]["size"]
    y_pos, y_quat = (demo.as_np(v) for v in robot.lm.get_instance_pose(0, label=place_anything.LABEL))
    y_rot = demo.from_wxyz(y_quat)
    centre, x_rot, _, x_bottom, _ = robot.object("target")
    hand_now = taker.pose()[1]
    hand_p = Rotation.from_euler("z", np.radians(pick_anything.grip_yaw(y_pos[0], y_pos[1]))) * demo.TOP_ROT
    turn = Rotation.from_euler("z", demo.yaw_of(hand_p.as_matrix()) - demo.yaw_of(hand_now.as_matrix()))
    rel = demo.yaw_of((turn * x_rot).as_matrix()) - demo.yaw_of(y_rot.as_matrix())
    spot = place_anything.spot_toward(geom, size, rel, geom["region"]["center"])
    xy = (y_pos + y_rot.apply([spot[0], spot[1], 0.0]))[:2]
    face = y_pos[2] + y_rot.apply([0.0, 0.0, geom["floor"]])[2]
    top = y_pos[2] + y_rot.apply([0.0, 0.0, geom["top"]])[2]

    def set_down(offset, grip_height, x_bottom, hand_p):
        """(point, above) of the taker setting X (`offset` from the grasp point) down over the spot."""
        hover = max(demo.HOVER, top + demo.PLACE_CARRY - face, x_bottom - face - demo.PLACE_GAP)
        point = np.r_[xy - hand_p.apply(offset)[:2], face + grip_height + demo.PLACE_GAP]
        return point, point + [0.0, 0.0, hover]

    offset = hand_now.inv().apply(centre - taker.grasp_point())
    point, above = set_down(offset, taker.grasp_point()[2] - x_bottom, x_bottom, hand_p)
    carry = max(48, int(np.ceil(np.linalg.norm(above - taker.grasp_point()) / demo.PLACE_SPEED * 30)))
    carry = max(carry, demo.turn_actions(hand_now, hand_p))
    print(
        f"[scripted_demo] support {geom['id']}: spot {np.round(spot, 3)} (yaw {np.degrees(rel):.0f} deg off),"
        f" carry {carry}"
    )
    yield from robot.move(taker, [(above, hand_p, 0.0, carry)])
    now = yield from demo.settle_in(robot, taker, above, hand_p)
    # level X's long side (it sags about the closing axis)
    _, x_rot, _, _, _ = robot.object("target")
    level = demo.leveled(hand_p, x_rot.as_matrix()[:, long_ax])
    if level is not None:
        print(f"[scripted_demo] leveling X: hand turned {np.degrees((level * hand_p.inv()).magnitude()):.0f} deg")
        yield from robot.move(taker, [(above, level, 0.0, max(20, demo.turn_actions(hand_p, level)))])
        now = yield from demo.settle_in(robot, taker, above, level)
        hand_p = level
    slipped = float(np.linalg.norm(now - (taker.grasp_point() + taker.pose()[1].apply(offset))))
    if slipped > place_v42.PLACE_HELD or demo.jaw_opening(env, taker) < demo.GRASP_EMPTY:
        print(f"[scripted_demo] target {slipped * 1000:.0f} mm off where it sat in the taker: fell out, giving up")
        yield from robot.finish()
        return
    # set down by where X sits in the hand now
    _, offset, grip_height = place_v42.in_hand(robot, taker)
    point, above2 = set_down(offset, grip_height, robot.object("target")[3], hand_p)
    short = np.linalg.norm(taker.grasp_point() - above)
    env.place_log = dict(hand_short_mm=round(short * 1000), slipped_mm=round(slipped * 1000))
    print(
        f"[scripted_demo] target {slipped * 1000:.0f} mm off in the hand, hand {short * 1000:.0f} mm off over the"
        f" spot: set down by where X sits"
    )
    keys = [
        (above2, hand_p, 0.0, 18),
        (point, hand_p, 0.0, 30),
        (point, hand_p, demo.TURNED_RELEASE, 12),  # release
        (above2, hand_p, demo.TURNED_RELEASE, 24),
    ]
    yield from robot.move(taker, keys)
    yield from robot.finish()
    yield from robot.move(taker, [(taker.grasp_point(), taker.pose()[1], demo.TURNED_RELEASE, demo.PLACE_HOLD)])
    end = robot.object("target")[0]
    print(f"[scripted_demo] target at {end.round(3)}, support at {robot.object(place_anything.LABEL)[0].round(3)}")
