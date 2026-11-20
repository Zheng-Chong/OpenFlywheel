"""Place anything expert v4.3: place v4.2 (place_v42: v3's place with the place stage's IK stopping joints at the
Limiter's clamp, set down by where X sits in the hand, a dropped X grasped again) whose first grasp of X is the one the
whole route reaches (docs/pick-expert-v4.md「v4.3」). Kept out of scripted_demo.py so that the registered experts'
sources stay as they are.

v4.2's remaining off-spot episodes were spots the hand, turned as it grasped X, never reaches (A/B v42d: the hand ended
more than 6 cm short over the spot in 23% / 38% of Place / Put placements). The hand keeps its grasp turn all the way
(turning it on the way stalled and swung X out, v42 / v42c), so the turn is chosen at the grasp: before the first try
Route replays on the URDF the IK the sim runs (Arm.ik for the grasp, ik_within for the place, the Limiter's clamp and
speed) along hover, grip, lift, carry over the spot and set-down, and the first candidate whose route ends within
ROUTE_TOL is taken: v3's first (Place), a slender X taken deep first then v3's (Put: held by its top it swings and slips
in the hand), then the other candidates in pick v4's order; none: v3's. Later tries are pick v4.1's.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1d import place_v42, scripted_demo as demo
from scripts.objects import place_anything

ARM_IK_MARGIN = 0.025  # rad: Arm.ik switches a joint off this far off its limit (the grasp's IK)
# m the replayed route may end off the grip and off the set-down (over the spot it may end further off: in the far
# region v3's replayed hand ended 4-8 cm off over the spot yet reached the set-down, and v4.1 placed those, A/B v43b)
ROUTE_TOL = 0.015


class Route:
    """Replays the sim's step IK on an arm's URDF (GraspRobot's Reach): one damped least-squares step per action from
    where the arm is, joints switched off `margin` off their limits, the command clamped and rate-limited as the
    Limiter does, the arm taken to follow it."""

    def __init__(self, reach):
        self.reach = reach
        self.raw = reach.lim - [demo.REAL_JOINT_MARGIN, -demo.REAL_JOINT_MARGIN]

    def pose(self, q):
        """(hand rotation, grasp point, [(joint point, axis)]) at joints `q`, sim frame."""
        axes = []
        t = self.reach.base @ self.reach.fk(q, axes)
        b = self.reach.base
        axes = [(b[:3, :3] @ o + b[:3, 3], b[:3, :3] @ a) for o, a in axes]
        return Rotation.from_matrix(t[:3, :3]), t[:3, 3] + t[:3, :3] @ self.reach.reach, axes

    def step(self, q, point, rot_t, margin):
        rot, grip, axes = self.pose(q)
        err = np.r_[point - grip, 0.3 * (rot_t * rot.inv()).as_rotvec()]
        jac = np.zeros((6, len(q)))
        for i, (o, a) in enumerate(axes):
            jac[:3, i], jac[3:, i] = np.cross(a, grip - o), 0.3 * a
        lim = self.raw + [margin, -margin]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err)
        jac[:, ((q + dq <= lim[:, 0]) & (dq < 0)) | ((q + dq >= lim[:, 1]) & (dq > 0))] = 0.0
        cmd = np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err), lim[:, 0], lim[:, 1])
        cmd = np.clip(cmd, self.reach.lim[:, 0], self.reach.lim[:, 1])
        dq_max = demo.REAL_ARM_VMAX / 30
        return q + np.clip(cmd - q, -dq_max, dq_max)

    def move(self, q, keys, margin):
        """Robot.move with PATTERN_SETTLE replayed: (joints, m the grasp point ends off the last key)."""
        tol, most = demo.PATTERN_SETTLE
        for key in keys:
            rot, p, _ = self.pose(q)
            for point, r, _ in demo.waypoints(p, rot, [key]):
                q = self.step(q, point, r, margin)
            for _ in range(most):
                if np.linalg.norm(self.pose(q)[1] - key[0]) < tol:
                    break
                q = self.step(q, key[0], key[1], margin)
        return q, float(np.linalg.norm(self.pose(q)[1] - keys[-1][0]))


class Spot:
    """The receptacle and where v3's place sets X down over it (place_v42.place_anything_actions' set_down)."""

    def __init__(self, robot):
        env = robot.env
        self.geom = demo.layout_entry(env, place_anything.LABEL)["receptacle"]
        self.size = demo.layout_entry(env, "target")["physics"]["size"]
        pos, quat = (demo.as_np(v) for v in robot.lm.get_instance_pose(0, label=place_anything.LABEL))
        self.y_pos, self.y_rot = pos, demo.from_wxyz(quat)
        self.face = pos[2] + self.y_rot.apply([0.0, 0.0, self.geom["floor"]])[2]
        self.top = pos[2] + self.y_rot.apply([0.0, 0.0, self.geom["top"]])[2]
        self.toward = self.y_rot.inv().apply(robot.object("target")[0] - pos)[:2]

    def xy(self, x_rot):
        rel = demo.yaw_of(x_rot.as_matrix()) - demo.yaw_of(self.y_rot.as_matrix())
        spot = place_anything.spot_toward(self.geom, self.size, rel, self.toward)
        return (self.y_pos + self.y_rot.apply([spot[0], spot[1], 0.0]))[:2]

    def set_down(self, xy, hand, offset, grip_height, x_bottom):
        """(point, above) of the hand setting X (`offset` from the grasp point, hand frame) down at `xy`."""
        surface = self.face
        if self.geom["kind"] == "container":
            surface = max(self.face, self.top + demo.PLACE_DROP + demo.FINGER_TIPS - grip_height)
        hover = max(demo.HOVER, self.top + demo.PLACE_CARRY - surface, x_bottom - surface - demo.PLACE_GAP)
        point = np.r_[xy - hand.apply(offset)[:2], surface + grip_height + demo.PLACE_GAP]
        return point, point + [0.0, 0.0, hover]


class PlaceRobot43(place_v42.PlaceRobot):
    """place_v42.PlaceRobot whose first grasp of the target in each pick() is chosen by its whole route (Route).
    `slender_deep`: a slender target is tried deep first (Put)."""

    slender_deep = False

    def __init__(self, env):
        super().__init__(env)
        self.spot = Spot(self)
        self.routing = False
        env.place_route = []

    def pick(self, label, *args, **kwargs):
        self.routing = label == "target"
        return (yield from super().pick(label, *args, **kwargs))

    def order(self, cands):
        first = ["deep", "v3"] if self.slender_deep else ["v3"]
        return sorted(cands, key=lambda c: first.index(c[4]) if c[4] in first else len(first))

    def route(self, cand, lift):
        """(m off the grip, m off over the spot, m off the set-down) of the replayed route of grasp `cand`."""
        a, hand, grip, _, _ = cand
        centre, x_rot, _, bottom, _ = self.object("target")
        r = Route(self.reach[a.arm_key])
        hover = grip + [0.0, 0.0, lift]
        q, _ = r.move(a.q(), [(hover, hand, 0.0, 54)], ARM_IK_MARGIN)
        q, off_grip = r.move(q, [(grip, hand, 0.0, self.DESCEND)], ARM_IK_MARGIN)
        if off_grip > ROUTE_TOL:
            return off_grip, 1.0, 1.0  # m: not replayed further
        q, _ = r.move(q, [(hover, hand, 0.0, 30)], ARM_IK_MARGIN)
        offset = hand.inv().apply(centre - grip)
        point, above = self.spot.set_down(self.spot.xy(x_rot), hand, offset, grip[2] - bottom, bottom + lift)
        carry = max(48, int(np.ceil(np.linalg.norm(above - hover) / demo.PLACE_SPEED * 30)))
        q, off_above = r.move(q, [(above, hand, 0.0, carry)], place_v42.PLACE_JOINT_MARGIN)
        _, off_point = r.move(q, [(point, hand, 0.0, 30)], place_v42.PLACE_JOINT_MARGIN)
        return off_grip, off_above, off_point

    def choose(self, cands, lift):
        if not self.routing:
            return super().choose(cands, lift)
        self.routing = False
        for cand in self.order(cands):
            off = self.route(cand, lift)
            ok = off[0] <= ROUTE_TOL and off[2] <= ROUTE_TOL
            self.env.place_route.append(dict(grasp=cand[4], off_mm=[round(v * 1000) for v in off], ok=ok))
            print(f"[scripted_demo] route of {cand[4]}: off grip / over the spot / set-down {np.round(off, 3)} m")
            if ok:
                return cand, None
        print("[scripted_demo] no grasp's route reaches the set-down: v3's")
        return super().choose(cands, lift)


class PutRobot43(PlaceRobot43):
    slender_deep = True


def place_anything_actions(env):
    """g1d_place_anything with expert v4.3."""
    return place_v42.place_anything_actions(env, PlaceRobot43)


def put_anything_in_actions(env):
    """g1d_put_anything_in with expert v4.3: a slender X tried deep first."""
    return place_v42.place_anything_actions(env, PutRobot43)
