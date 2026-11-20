"""Place anything expert v4.2: pick anything expert v4.1's grasp (grasp_v41.GraspRobot41) and v3's place
(scripted_demo.place_anything_actions: the same spot, hand and moves) with three changes (docs/pick-expert-v4.md
「v4.2」). Kept out of scripted_demo.py so that the registered experts' sources stay as they are.

- The place stage's moves step with ik_within: most grasped-but-rejected episodes (A/B 11-19, rounds 1-2) were let go
  off the receptacle, the arm stalled 5-30 cm short of the spot with a joint (mostly shoulder roll, wrist roll or
  elbow) at the Limiter's clamp, REAL_JOINT_MARGIN off its limit, while Arm.ik switches a joint off only within 0.025
  and so kept asking the clamped joint to move.
- Over the spot, once X has stopped swinging, the set-down is planned from where X sits in the hand then (v3: given up
  past PLACE_SLIPPED, else moved by at most PLACE_FIX).
- An X that fell out of the hand on the way (PLACE_HELD off it, or the jaw shut on nothing) is grasped again
  (PLACE_REGRASPS) unless it fell off the desk (v3: the episode ran out holding nothing).

Turning the hand toward a spot the URDF (GraspRobot's Reach) reaches, and finishing stalled moves in joint space, were
tried and dropped (v42 / v42b / v42c, 11-19): the URDF rejected hand turns v3 placed with, the turned carries stalled
as well, and joint-space moves swept X into things or flung it out of the hand.
"""

import numpy as np

from scripts.g1d import scripted_demo as demo
from scripts.g1d.grasp_v41 import GraspRobot41
from scripts.objects import pick_anything, place_anything

PLACE_JOINT_MARGIN = demo.REAL_JOINT_MARGIN + 0.01  # rad off a limit ik_within stops a joint at (inside the Limiter's)
PLACE_REGRASPS = 1  # times an X that fell out of the hand on the way is grasped again
PLACE_HELD = 0.06  # m: X's centre this far off where it sat in the hand once lifted has fallen out of it
PLACE_GONE = 0.05  # m below where X lay at the start: off the desk, not grasped again


def ik_within(arm, point, rot_t, margin=PLACE_JOINT_MARGIN):
    """Arm.ik's damped least-squares step with each joint switched off `margin` off its limit (Arm.ik: 0.025, inside
    the Limiter's clamp, so a clamped joint kept being asked to move and the arm stalled)."""
    pos, rot = arm.pose()
    r = rot.apply(arm.REACH)
    err = np.r_[point - (pos + r), 0.3 * (rot_t * rot.inv()).as_rotvec()]
    jac = demo.as_np(demo.link_jacobians(arm.key)[0, arm.body - 1][:, arm.cols]).reshape(6, -1)
    jac = np.r_[jac[:3] - np.cross(r, jac[3:].T).T, 0.3 * jac[3:]]
    q = arm.q()
    lim = demo.as_np(arm.key.data.soft_joint_pos_limits[0, arm.cols]).reshape(-1, 2) + [margin, -margin]
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err)
    jac[:, ((q + dq <= lim[:, 0]) & (dq < 0)) | ((q + dq >= lim[:, 1]) & (dq > 0))] = 0.0
    return np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err), lim[:, 0], lim[:, 1])


class PlaceRobot(GraspRobot41):
    """GraspRobot41 whose moves step with ik_within once `placing` (the grasp stays v4.1's)."""

    placing = False

    def _command(self, arm, point, rot, grip):
        if not self.placing:
            return super()._command(arm, point, rot, grip)
        self.hold[arm.arm_key] = ik_within(arm, point, rot)
        self.grip[arm.grip_key] = [float(grip)]
        self._advance(arm.arm_key)
        if self.step % 20 == 0:
            print(f"[scripted_demo] {self.step:3d} {arm.arm_key} err {np.linalg.norm(arm.grasp_point() - point):.3f}")
        self.step += 1
        return dict(self.hold) | self.grip


def in_hand(robot, arm):
    """(X's centre, its offset from the grasp point in the hand frame, grasp point height above X's bottom)."""
    centre, _, _, bottom, _ = robot.object("target")
    point, rot = arm.grasp_point(), arm.pose()[1]
    return centre, rot.inv().apply(centre - point), point[2] - bottom


def place_anything_actions(env, robot_class=PlaceRobot):
    """g1d_place_anything, g1d_put_anything_in with expert v4.2 (module docstring)."""
    robot = robot_class(env)
    robot.settle = demo.PATTERN_SETTLE
    start = robot.object("target")[0]
    prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(start[0], start[1])))
    geom = demo.layout_entry(env, place_anything.LABEL)["receptacle"]
    size = demo.layout_entry(env, "target")["physics"]["size"]
    y_pos, y_quat = (demo.as_np(v) for v in robot.lm.get_instance_pose(0, label=place_anything.LABEL))
    y_rot = demo.from_wxyz(y_quat)
    face = y_pos[2] + y_rot.apply([0.0, 0.0, geom["floor"]])[2]
    top = y_pos[2] + y_rot.apply([0.0, 0.0, geom["top"]])[2]
    toward = y_rot.inv().apply(start - y_pos)[:2]
    for _ in range(1 + PLACE_REGRASPS):
        robot.placing = False
        lying = robot.object("target")[0]
        if lying[2] < start[2] - PLACE_GONE:
            print(f"[scripted_demo] target off the desk ({lying.round(3)}): not grasped again")
            return
        arm, hand, offset, grip_height, open_cmd, close_cmd = yield from robot.pick(
            "target", lift=demo.PICK_ANYTHING_LIFT, close=True, prefer=prefer
        )
        if robot.object("target")[0][2] - lying[2] < demo.PLACE_LIFTED:
            print("[scripted_demo] target not lifted: missed, giving up")
            return
        robot.placing = True
        _, x_rot, _, x_bottom, _ = robot.object("target")
        rel = demo.yaw_of(x_rot.as_matrix()) - demo.yaw_of(y_rot.as_matrix())
        spot = place_anything.spot_toward(geom, size, rel, toward)
        xy = (y_pos + y_rot.apply([spot[0], spot[1], 0.0]))[:2]

        def set_down(offset, grip_height, x_bottom):
            """(point, above) of the hand setting X (`offset` from the grasp point) down over the spot (v3's)."""
            surface = face
            if geom["kind"] == "container":
                surface = max(face, top + demo.PLACE_DROP + demo.FINGER_TIPS - grip_height)
            hover = max(demo.HOVER, top + demo.PLACE_CARRY - surface, x_bottom - surface - demo.PLACE_GAP)
            point = np.r_[xy - hand.apply(offset)[:2], surface + grip_height + demo.PLACE_GAP]
            return point, point + [0.0, 0.0, hover]

        point, above = set_down(offset, grip_height, x_bottom)
        carry = max(48, int(np.ceil(np.linalg.norm(above - arm.grasp_point()) / demo.PLACE_SPEED * 30)))
        print(f"[scripted_demo] {geom['kind']} {geom['id']}: spot {np.round(spot, 3)}, carry {carry}")
        yield from robot.move(arm, [(above, hand, close_cmd, carry)])
        robot.settle, before = None, robot.object("target")[0]
        now = before
        for _ in range(demo.PLACE_SETTLE):
            yield from robot.move(arm, [(above, hand, close_cmd, 1)])
            now = robot.object("target")[0]
            if np.linalg.norm(now - before) * 30 < demo.SETTLED_SPEED:
                break
            before = now
        robot.settle = demo.PATTERN_SETTLE
        slipped = float(np.linalg.norm(now - (arm.grasp_point() + arm.pose()[1].apply(offset))))
        if slipped > PLACE_HELD or demo.jaw_opening(env, arm) < demo.GRASP_EMPTY:
            print(f"[scripted_demo] target {slipped * 1000:.0f} mm off where it sat in the hand: fell out")
            yield from robot.move(arm, [(arm.grasp_point() + [0, 0, 0.03], arm.pose()[1], open_cmd, 20)])
            continue
        _, offset, grip_height = in_hand(robot, arm)
        point, above2 = set_down(offset, grip_height, robot.object("target")[3])
        short = np.linalg.norm(arm.grasp_point() - above)
        env.place_log = dict(hand_short_mm=round(short * 1000), slipped_mm=round(slipped * 1000))
        print(
            f"[scripted_demo] target {slipped * 1000:.0f} mm off in the hand, hand {short * 1000:.0f} mm off over the"
            f" spot: set down by where X sits"
        )
        keys = [
            (above2, hand, close_cmd, 18),
            (point, hand, close_cmd, 30),
            (point, hand, demo.TURNED_RELEASE, 12),  # release
            (above2, hand, demo.TURNED_RELEASE, 24),
        ]
        yield from robot.move(arm, keys)
        yield from robot.move(arm, [(arm.grasp_point(), arm.pose()[1], demo.TURNED_RELEASE, demo.PLACE_HOLD)])
        end, rec = robot.object("target")[0], robot.object(place_anything.LABEL)[0]
        print(f"[scripted_demo] target at {end.round(3)}, receptacle at {rec.round(3)}")
        return
