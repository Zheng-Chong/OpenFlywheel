"""Scripted g1d_build_block_pattern (照图搭积木) with a dexterous-hand G1-D (PINCHES: Wuji Hand2.2 / 2.1, Sharpa Wave,
Inspire RH56DFTP) in the office (env_cfg/g1d_office_<hand>.yml): scripts/g1d/scripted_demo.py's pattern expert,
the Dex1 jaw replaced by a tip pinch, thumb against the index and middle tips (ring and little fingers curled away).

Each arm reports a jaw-like frame at the pinch point instead of its ee link (PinchArm): x the approach, y the closing
axis (thumb to finger tips at the grasp width, flipped on the right hand so the same side of both hands is up),
z = x cross y; so the expert's top grasp (TOP_ROT, fingers 60 deg down, closing across)
and its yaws apply as they are. The gripper command stays scalar (and rate-limited as Dex1's): 0 the tips together ..
1 the pinch open, past 1 on to the open hand (all joints 0 rad) the episode ends with; hand_joints() turns it into the
hand's joint targets at the end.

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.:XPolicyLab EVAL_NUM=1 python scripts/dexhands/block_pattern_demo.py \\
        --env_cfg_type g1d_office_wuji_hand2_2 --task_name g1d_build_block_pattern --seed 70 --layout_ids 3
"""

import json
from pathlib import Path
import sys
from typing import NamedTuple
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.plugins import g1d_build_block_pattern as plugin  # noqa: E402
from scripts.dexhands.kinematics import fk, origin_transform, rotation, vector  # noqa: E402
from scripts.g1d import scripted_demo as sd  # noqa: E402


class Pinch(NamedTuple):
    """A tip pinch of one hand model, in its left ee link's frame (robot.urdf FK); the right hand mirrors it in y."""

    ee: str  # left ee link
    closed: tuple  # joint targets (rad, robot config order), the three tips together
    open: tuple  # 75-85 mm between the thumb tip and the finger tips, about the same point
    tips: tuple  # (link, offset in its frame) of the thumb, index and middle fingertips
    approach: tuple  # the top grasp's approach (made normal to the closing axis): with the fingers 60 deg down, the
    # one (mesh FK, every 10 deg about the closing axis) whose lowest part, open 30 .. 55 mm, is highest
    low: float  # m that part is below the pinch point
    # m the thumb's and fingers' pads lie inside their tip points along the closing axis, together (visual meshes
    # within 20 mm of the pinch point, at 30 .. 50 mm): the hand opens to the block's width plus this
    pad: float
    right: tuple = ()  # rad added to the right hand's targets so that it mirrors the left
    shoulder_yaw_in: float = None  # rad the shoulder yaw may turn the arm in (default: its limit)


# Joint poses fitted by FK (least squares): the thumb tip on the middle of the index and middle tips, 18 mm apart, the
# distal phalanges of thumb and fingers parallel (curled fingers held the middle phalanges 26 mm below the tips); open
# with the same pinch point and closing axis. Wuji: thumb cmc_flex, cmc_abd, mcp, ip; index .. pinky mcp_flex,
# mcp_abd, pip, dip.
WUJI = Pinch(
    ee="l_mount",
    closed=(
        0.84,
        -0.26,
        0.05,
        -0.34,
        1.24,
        -0.64,
        0.45,
        -0.84,
        1.36,
        -0.67,
        0.58,
        -0.85,
        1.3,
        0,
        1.6,
        1.2,
        1.3,
        0,
        1.6,
        1.2,
    ),
    open=(0.6, -0.77, -0.02, 0.5, 0.08, -0.68, 1.2, 0.45, 0.95, -0.65, 0.21, -0.77, 1.3, 0, 1.6, 1.2, 1.3, 0, 1.6, 1.2),
    tips=(("l_thumb_tip", (0, 0, 0)), ("l_index_finger_tip", (0, 0, 0)), ("l_middle_finger_tip", (0, 0, 0))),
    approach=(0.34, -0.8, -0.49),  # palm down, the tips under it
    low=0.004,
    pad=0.014,
    # to cubes 0.46-0.52 m out near the body, the shoulder yaw stopped at 0.39-0.48 rad in, 0.1-0.25 short of its
    # command, the arm against the body, and the hand never came down (sim, 11-06); within this the IK reaches
    # every block place and the build site (offline, 11-06)
    shoulder_yaw_in=0.45,
)
PINCHES = {
    "g1d_wuji_hand2_2": WUJI,
    "g1d_wuji_hand2_1": WUJI,  # same kinematics and joints, no tip sensor frames
    # Sharpa Wave: thumb CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP; index, middle, ring: MCP_FE, MCP_AA, PIP, DIP; pinky CMC,
    # MCP_FE, MCP_AA, PIP, DIP
    "g1d_sharpa_wave": Pinch(
        ee="left_hand_flange",
        closed=(
            1.03,
            -0.04,
            0.47,
            -0.2,
            0.85,
            0.66,
            -0.06,
            0.99,
            0.7,
            0.71,
            -0.09,
            0.94,
            0.7,
            1.4,
            0,
            1.5,
            1.2,
            0,
            1.4,
            0,
            1.5,
            1.2,
        ),
        open=(
            0.58,
            -0.27,
            0.73,
            0.33,
            1.19,
            1.29,
            0.26,
            0.05,
            0.03,
            1.29,
            0.24,
            0.05,
            0.03,
            1.4,
            0,
            1.5,
            1.2,
            0,
            1.4,
            0,
            1.5,
            1.2,
        ),
        tips=(
            ("left_thumb_fingertip", (0, 0, 0)),
            ("left_index_fingertip", (0, 0, 0)),
            ("left_middle_fingertip", (0, 0, 0)),
        ),
        approach=(0.13, -0.37, 0.92),
        low=0.013,
        # the tip frames are on the nail side: fully open, these poses leave 25 mm between the pads, under a cube's
        # 30, and the index's middle phalanx hangs over the pinch point, so the fingers come down on a cube's top
        # (11-06; a pose that opens the pads wider has not been fitted)
        pad=0.040,
    ),
    # Inspire RH56DFTP: little, ring, middle, index, thumb bend, thumb swing; no tip links: the distal meshes' far
    # vertices
    "g1d_inspire_rh56dftp": Pinch(
        ee="left_inspire_base_link",
        closed=(1.6, 1.6, 0.9, 0.9, 0.51, 1.7),
        open=(1.6, 1.6, 0.45, 0.45, 0.3, 1.7),
        tips=(
            ("left_thumb_3", (-0.034, 0.01, -0.012)),
            ("left_index_2", (0.004, 0.053, -0.011)),
            ("left_middle_2", (0.005, 0.057, -0.011)),
        ),
        # 50 deg on from the highest (-0.77, -0.62, 0.19), low 6 mm: with that, the left arm reached the cubes and the
        # build site only at its wrist limits, its IK flipped the wrist roll each few actions and the sim blew up; this
        # one reaches them all with the shoulder roll in SHOULDER_IN (offline IK, 11-06)
        approach=(0.0, -0.97, 0.26),
        low=0.008,
        pad=0.010,
        # the right thumb's joints are zeroed apart from the left's mirror: its thumb meshes match the mirrored left
        # ones within 1.3 mm at these offsets of bend and rotation (any pose, mesh FK), 6-13 mm off without (11-06)
        right=(0.0, 0.0, 0.0, 0.0, -0.125, -0.311),
    ),
}
WIDTH = 0.03  # m: every block is gripped across its 30 mm depth (local Y); the pinch point is taken at this aperture
SQUEEZE = 0.012  # m the closed command's aperture is under the block's width
CLEAR = 0.004  # m the hand's lowest part stays above the table
TIP_IN = 0.008  # m the pinch point stays under a block's top
REST_ACTIONS = 40  # actions to open the hand fully once the pattern stands
# rad the shoulder roll may bring the upper arm in past straight down: further, it pressed into the torso and stopped
# 0.3 rad short (Wuji Hand2.2, the left hand to a cube 0.25 m ahead of the robot, 11-06)
SHOULDER_IN = 0.05
# ik() solves from the last command; missing the target by over IK_TOL (m), it also solves with each joint pulled toward
# a nominal arm pose by IK_PULL (residual m per rad), then without from there, and keeps the closer: from the last
# command alone, the RH56DFTP's arm folded into its shoulder limits 7 to 10 cm from cubes it reaches from other starts,
# and with the pull alone Wuji's right hand stopped 12 mm beside a cube (11-06). The nominal pose: the pinch frame at
# NOMINAL (|x|, y, z, m, env frame; above the table on the arm's side) at PATTERN_PLACE_YAW, of NOMINAL_STARTS random
# starts and the start pose the one that reaches it nearest the start pose
IK_TOL, IK_PULL, NOMINAL, NOMINAL_STARTS = 0.003, 0.01, (0.16, -0.40, 0.835), 30
MIRROR = np.diag([1.0, -1.0, 1.0])


class Geometry(NamedTuple):
    frame: Rotation  # jaw-like frame in the ee link's
    point: np.ndarray  # pinch point at WIDTH, ee frame
    g: np.ndarray  # command samples, 0 .. 1
    aperture: np.ndarray  # m between the thumb tip and the finger tips at each


def geometry(robot, pinch):
    """The pinch's frame, point and aperture curve for `robot` (one arm), from its URDF: the left hand's FK, mirrored
    for the right."""
    path = Path(robot.robot_file)
    root = ET.parse(path / "robot.urdf").getroot()
    names = json.loads((path / "model_contract.json").read_text())["hands"]["left"]["joints"]
    closed, opened = pinch_joints(robot, pinch, mirror=False)

    def tips(g):
        """Thumb, index and middle tips, left ee frame."""
        t = fk(root, dict(zip(names, closed + g * (opened - closed))))
        inv = np.linalg.inv(t[pinch.ee])
        return [(inv @ t[link] @ np.r_[off, 1.0])[:3] for link, off in pinch.tips]

    gs = np.linspace(0.0, 1.0, 21)
    poses = [tips(g) for g in gs]
    centre = np.array([np.linalg.norm((i + m) / 2 - t) for t, i, m in poses])
    thumb, index, middle = tips(float(np.interp(WIDTH, centre, gs)))
    close = (index + middle) / 2 - thumb
    close /= np.linalg.norm(close)
    # the nearer finger tip along the closing axis: the index sits 10-15 mm nearer the thumb than the middle on Wuji
    # and Sharpa, and opened to the midpoint's width it came down on top of a cube (Sharpa Wave, 11-06)
    aperture = np.array([min((i - t) @ close, (m - t) @ close) for t, i, m in poses])
    thumb, index, middle = tips(float(np.interp(WIDTH, aperture, gs)))
    approach = np.asarray(pinch.approach, dtype=float)
    approach = approach - approach @ close * close
    approach /= np.linalg.norm(approach)
    point = (thumb + (index + middle) / 2) / 2
    point += close * (WIDTH / 2 - (point - thumb) @ close)
    if robot.config_key != "left":
        approach, close, point = MIRROR @ approach, -(MIRROR @ close), MIRROR @ point
    frame = Rotation.from_matrix(np.c_[approach, close, np.cross(approach, close)])
    return Geometry(frame, point, gs, aperture)


def pinch_joints(robot, pinch, mirror=True):
    """(closed, open) joint targets within this hand's limits (the right RH56DFTP's thumb bend starts at 0.035), moved
    by `pinch.right` on the right hand (mirror=False: the left hand's, for the FK in geometry())."""
    lim = robot.hand_limits()
    off = np.asarray(pinch.right) if mirror and pinch.right and robot.config_key != "left" else 0.0
    return tuple(np.clip(np.asarray(q, dtype=float) + off, lim[:, 0], lim[:, 1]) for q in (pinch.closed, pinch.open))


class PinchArm(sd.Arm):
    """An arm whose pose() is the pinch's jaw-like frame and whose grasp point is the pinch point. ik() solves each
    target exactly on the URDF chain (root to ee link), near the arm's last command (IK_PULL): the one damped step
    from the measured joints of sd.Arm.ik fell behind the rate-limited arm on the way down to a block, turned the
    wrist roll to its limit and left the hand 8 cm short, fingers up (Wuji Hand2.2, 11-06)."""

    def __init__(self, env, robot, geo):
        super().__init__(env, robot)
        self.geo = geo
        self.REACH = geo.frame.inv().apply(geo.point)
        root = ET.parse(Path(robot.robot_file) / "robot.urdf").getroot()
        parent = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], robot.ee_link_name
        while link in parent:
            chain.insert(0, parent[link])
            link = parent[link].find("parent").get("link")
        names = list(self.key.joint_names)
        arm = [names[c] for c in self.cols]
        q = sd.as_np(self.key.data.joint_pos[0])
        # (origin, axis or None, index in the arm's joints or the fixed value of a body joint)
        self.chain = [
            (
                origin_transform(j.find("origin")),
                None if j.get("type") == "fixed" else vector(j.find("axis").get("xyz"), (1, 0, 0)),
                ("arm", arm.index(j.get("name")))
                if j.get("name") in arm
                else ("fixed", q[names.index(j.get("name"))])
                if j.get("name") in names
                else ("fixed", 0.0),
            )
            for j in chain
        ]
        self.lim = sd.as_np(self.key.data.soft_joint_pos_limits[0, self.cols]).reshape(-1, 2) + [0.025, -0.025]
        roll = arm.index(next(n for n in arm if "shoulder_roll" in n))
        inward = -1 if robot.config_key == "left" else 1  # the roll that brings the upper arm in to the torso
        self.lim[roll, (1 + inward) // 2] = inward * SHOULDER_IN
        yaw_in = PINCHES[robot.asset_name].shoulder_yaw_in
        if yaw_in is not None:
            self.lim[arm.index(next(n for n in arm if "shoulder_yaw" in n)), (1 + inward) // 2] = inward * yaw_in
        self.last = None
        x, y, z = NOMINAL
        side = -1 if robot.config_key == "left" else 1
        err = self.error(
            np.array([side * x, y, z]), Rotation.from_euler("z", sd.PATTERN_PLACE_YAW[self.arm_key]) * sd.TOP_ROT
        )
        rng = np.random.default_rng(0)
        fits = [self.solve(err, s) for s in [self.q()] + [rng.uniform(*self.lim.T) for _ in range(NOMINAL_STARTS)]]
        reached = [f.x for f in fits if np.linalg.norm(f.fun[:3]) < 0.003] or [min(fits, key=lambda f: f.cost).x]
        self.nominal = min(reached, key=lambda q: np.linalg.norm(q - self.q()))
        pos, rot = super().pose()
        fk_pos, fk_rot = self.fk(self.q())
        root_rot = sd.from_wxyz(sd.as_np(self.key.data.root_quat_w[0]))
        fk_pos, fk_rot = self.root() + root_rot.apply(fk_pos), root_rot * fk_rot
        print(
            f"[block_pattern] {self.arm_key} URDF chain vs sim ee: {np.linalg.norm(fk_pos - pos) * 1000:.1f} mm,"
            f" {np.degrees((fk_rot * rot.inv()).magnitude()):.1f} deg (ik() takes up the difference)",
            flush=True,
        )

    def fk(self, q):
        """ee link pose (4x4) in the URDF root's frame at arm joints `q`."""
        t = np.eye(4)
        for origin, axis, (kind, v) in self.chain:
            t = t @ origin
            if axis is not None:
                m = np.eye(4)
                m[:3, :3] = rotation(axis, q[v] if kind == "arm" else v)
                t = t @ m
        return t[:3, 3], Rotation.from_matrix(t[:3, :3])

    def sim_pose(self):
        pos, rot = super().pose()
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = rot.as_matrix(), pos
        return out

    def pose(self):
        pos, rot = super().pose()
        return pos, rot * self.geo.frame

    def error(self, point, rot_t, pull=0.0):
        """Residual of arm joints q for the pinch frame at (`point`, `rot_t`), plus `pull` * (q - the nominal pose)."""
        pos, rot = self.fk(self.q())
        now = np.eye(4)
        now[:3, :3], now[:3, 3] = rot.as_matrix(), pos
        base = self.sim_pose() @ np.linalg.inv(now)  # URDF root -> env frame, as the sim has the ee link now

        def err(q):
            pos, rot = self.fk(q)
            pos, rot = base[:3, :3] @ pos + base[:3, 3], Rotation.from_matrix(base[:3, :3]) * rot * self.geo.frame
            out = np.r_[pos + rot.apply(self.REACH) - point, 0.1 * (rot_t * rot.inv()).as_rotvec()]
            return np.r_[out, pull * (q - self.nominal)] if pull else out

        return err

    def solve(self, err, seed):
        return least_squares(err, np.clip(seed, *self.lim.T), bounds=(self.lim[:, 0], self.lim[:, 1]), xtol=1e-6)

    def reach_error(self, point, rot_t):
        """How far the pinch frame misses (`point`, `rot_t`) at best, in IK_TOL (position) or 3 deg (rotation)
        units, from the last command, the nominal pose and a few fixed starts; the arm is not commanded."""
        err = self.error(point, rot_t)
        rng = np.random.default_rng(1)
        last = self.q() if self.last is None else self.last
        seeds = [last, self.nominal] + [rng.uniform(*self.lim.T) for _ in range(3)]
        out = np.inf
        for seed in seeds:
            f = self.solve(err, seed).fun
            out = min(out, max(np.linalg.norm(f[:3]) / IK_TOL, np.degrees(np.linalg.norm(f[3:]) / 0.1) / 3))
            if out <= 1.0:
                break
        return out

    def ik(self, point, rot_t):
        seed = self.q() if self.last is None else self.last
        err = self.error(point, rot_t)
        best = self.solve(err, seed)
        if np.linalg.norm(best.fun[:3]) > IK_TOL:
            near = self.solve(self.error(point, rot_t, IK_PULL), seed).x
            best = min(best, self.solve(err, near), key=lambda f: f.cost)
        self.last = best.x
        return self.last


class PinchRobot(sd.Robot):
    """scripted_demo.Robot with a tip pinch per hand for the Dex1 jaw."""

    def make_arm(self, env, robot):
        return PinchArm(env, robot, geometry(robot, PINCHES[robot.asset_name]))

    def _geo(self):
        return self.arms[0].geo

    def opening(self, width):
        geo = self._geo()
        pad = PINCHES[self.arms[0].robot.asset_name].pad
        return float(np.interp(width + pad + sd.GRASP_CLEAR, geo.aperture, geo.g))

    def grip_z(self, bottom, top):
        low = PINCHES[self.arms[0].robot.asset_name].low
        return max(min(bottom + low + CLEAR, top - TIP_IN), (bottom + top) / 2)

    def closing(self, width):
        geo = self._geo()
        return float(np.interp(max(width - SQUEEZE, 0.0), geo.aperture, geo.g))

    def holding(self, arm, width):
        return self.closing(width)

    def pick(self, label, lift=sd.LIFT, close=False, across=None, prefer=None, period=np.pi, arm=None):
        """scripted_demo.Robot.pick with the hand yaw `prefer`red among those the arm reaches: of the yaws that grip
        the block alike, the nearest that the URDF IK reaches at the grip and the lift above it, else the closest
        miss. Nearest to block_patterns.grip_yaw alone, the RH56DFTP's arm missed cubes near the body by 5-8 cm
        (y -0.46 .. -0.48) and Wuji's one at y -0.52; with the cube's other quarter turns every cube in the scatter
        range is reached by every hand (offline IK, |x| 0.10 .. 0.19, y -0.53 .. -0.37, 11-06). A block for a layer
        above the table is lifted that much higher: lifted HOVER only, the third cube of a tower hung under the top of
        the two below, the RH56DFTP's hand ran into them on the way over and the cube fell (sim, 11-06)."""
        pattern = self.env.pattern[0]
        layer = dict(zip(sd.block_patterns.labels(pattern), (z for _, _, z in pattern["pieces"]))).get(label, 0.0)
        lift += layer
        if prefer is not None and across is not None:
            centre, rot, _, bottom, top = self.object(label)
            arm = arm or self.arm_for(centre[0])
            want = prefer[arm.arm_key]
            axes = rot.as_matrix()
            yaw = np.arctan2(axes[1, across], axes[0, across])
            if period:
                cands = [sd.wrap(yaw + period * k) for k in range(int(round(2 * np.pi / period)))]
            else:
                cands = [want + np.radians(d) for d in range(-90, 91, 15)]
            grip = np.r_[centre[:2], self.grip_z(bottom, top)]
            scored = []
            for c in cands:
                rot_t = Rotation.from_euler("z", c) * sd.TOP_ROT
                miss = max(arm.reach_error(p, rot_t) for p in (grip, grip + [0, 0, lift]))
                scored.append((miss > 1.0, abs(sd.wrap(c - want)) if miss <= 1.0 else miss, c))
            best = min(scored)[2]
            print(f"[block_pattern] {label} grip yaw {np.degrees(best):.0f} (preferred {np.degrees(want):.0f})")
            prefer = {arm.arm_key: best}
        return (yield from super().pick(label, lift, close, across, prefer, period, arm))

    def yaw_off(self, arm, label, period):
        """As scripted_demo.place_turned: tips do not square a block, one stayed 21-27 deg turned in the hand and the
        tower missed the task's 15 deg (Wuji, 11-06)."""
        if not period:
            return 0.0
        in_hand = sd.yaw_of(self.object(label)[1].as_matrix()) - (arm.pose()[1] * sd.TOP_ROT.inv()).as_euler("zyx")[0]
        off = (sd.PATTERN_PLACE_YAW[arm.arm_key] + in_hand + period / 2) % period - period / 2
        print(f"[block_pattern] {label} sits {np.degrees(off):.1f} deg off in the hand", flush=True)
        return off

    def release(self):
        """Open both hands fully (command 1 -> 2) where they are."""
        for k in range(1, REST_ACTIONS + 1):
            self.grip = {a.grip_key: [1.0 + k / REST_ACTIONS] for a in self.arms}
            self.step += 1
            yield dict(self.hold) | self.grip

    def hand_joints(self, action):
        """`action` with each scalar gripper command turned into the hand's joint targets."""
        out = dict(action)
        for arm in self.arms:
            if arm.grip_key not in out:
                continue
            lim = arm.robot.hand_limits()
            closed, opened = pinch_joints(arm.robot, PINCHES[arm.robot.asset_name])
            rest = np.clip(0.0, lim[:, 0], lim[:, 1])
            g = float(np.clip(out[arm.grip_key][0], 0.0, 2.0))
            q = closed + g * (opened - closed) if g <= 1.0 else opened + (g - 1.0) * (rest - opened)
            out[arm.grip_key] = list(np.clip(q, lim[:, 0], lim[:, 1]))
        return out


def actions(env, robot):
    yield from sd.block_pattern_actions(env, robot)
    yield from robot.release()


def eval_one_episode(TASK_ENV, model_client):
    TASK_ENV.get_obs()
    robot = PinchRobot(TASK_ENV)
    robot.settle = sd.PATTERN_SETTLE
    for arm in robot.arms:
        print(
            f"[block_pattern] {arm.arm_key} {arm.robot.asset_name}: pinch point {arm.geo.point.round(3)}"
            f" aperture {arm.geo.aperture.min() * 1000:.0f}..{arm.geo.aperture.max() * 1000:.0f} mm",
            flush=True,
        )
    limiter = sd.Limiter(TASK_ENV)
    episode = plugin.Episode(TASK_ENV)
    for action in actions(TASK_ENV, robot):
        TASK_ENV.take_action(robot.hand_joints(limiter(action)))
        if TASK_ENV.is_episode_end():
            break
        TASK_ENV.get_obs()
    v2 = episode.step()
    xy, z, tilt, yaw = episode.last
    print(
        f"[block_pattern] end: task check {bool(TASK_ENV.success[0]) and bool(TASK_ENV.end_flag[0])},"
        f" v2 tolerances on the last frame {v2} (xy {xy * 1000:.0f} mm, z {z * 1000:.0f} mm, tilt {tilt:.0f},"
        f" yaw {yaw:.0f} deg; openness {np.round(plugin.gripper_openness(TASK_ENV), 2).tolist()})",
        flush=True,
    )
    if not TASK_ENV.end_flag[0]:
        TASK_ENV.success[0] = False
        TASK_ENV.end_flag[0] = True


if __name__ == "__main__":
    sd.main(policy_module="scripts.dexhands.block_pattern_demo")
