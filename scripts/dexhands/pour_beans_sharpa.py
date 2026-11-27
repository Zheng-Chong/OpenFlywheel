"""Scripted bimanual pour-beans expert for G1-D with Sharpa Wave dexterous hands (61D wholebody:
left_arm 7 + left_hand 22 + right_arm 7 + right_hand 22 + waist 2 + lift 1).

Designed for `g1d_pour_beans_plus` (rigid beans, arbitrary cup shapes from `cups.json` with 3D `cavity`/`body`
profiles). Both hands side-grasp their cups with palm inward, fingers forward (+y), thumb up (+z), using
radius-adapted thumb-first wrap (from cup_grasp_demo.py), lift via coordinated lift-column + arm motion,
pour by rolling the source cup about its forearm (+y) axis with closed-loop bean-flow feedback, and place
both cups back upright on the desk before releasing fingers-first and sliding outward.

Usage (GPU Slurm via isaac-sim, or direct GPU):
    PYTHONPATH=.:XPolicyLab python scripts/datagen/collect.py \
        --task g1d_pour_beans_plus --plugin g1d_pour_beans_plus_sharpa \
        --env-cfg g1d_shenzhen_lab_sharpa_beans_wb --seed <seed> \
        --layout-ids 0 --output output/datagen/sharpa-smoke --device cuda:0 --device_id 0
"""

import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d import scripted_demo as sd  # noqa: E402
from scripts.objects import cup_profile  # noqa: E402

# Sharpa Wave left_hand_flange frame (mirrors in y on right_hand_flange):
# fingers along +z (knuckles 0.13 m, tips 0.23 m), palm surface at x = 0.025 m, thumb toward -y.
FINGERS = np.array([0.0, 0.0, 1.0])
PALM = np.array([1.0, 0.0, 0.0])
THUMB = np.array([0.0, -1.0, 0.0])
PALM_POINT = np.array([0.025, 0.0, 0.13])
PALM_CLEAR = -0.002  # m 2 mm flush palm preload for reliable lift across empty/tapered cups

# 22D Sharpa Wave joint order:
# [0..4]   thumb:  CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
# [5..8]   index:  MCP_FE, MCP_AA, PIP, DIP
# [9..12]  middle: MCP_FE, MCP_AA, PIP, DIP
# [13..16] ring:   MCP_FE, MCP_AA, PIP, DIP
# [17..21] pinky:  CMC, MCP_FE, MCP_AA, PIP, DIP
HAND_DIM = 22
THUMB_COUNT = 5
OPEN_POSE = np.zeros(HAND_DIM, dtype=float)
PRESHAPE_POSE = np.r_[
    [0.0, -0.34, 1.39, -0.33, 1.74],
    [0.0, 0.0, 0.30, 0.20] * 3,
    [0.0, 0.0, 0.0, 0.30, 0.20],
]
# FK-calibrated reference wrap for R_REF=0.040 m (~8 mm inward preload on all 5 fingertips; prevents
# the -35 mm over-penetration that tilted cups ~40 deg during closing).
CLOSED_BASE = np.r_[
    [0.76, -0.34, 1.22, -0.27, 1.50],
    [0.70, 0.0, 0.87, 0.57] * 3,
    [0.0, 0.70, 0.0, 0.87, 0.57],
]
R_REF = 0.040  # m reference cup radius for CLOSED_BASE

# Wholebody posture & motion parameters
WAIST_PITCH_GRASP = 0.25  # rad forward pitch so both hands reach desk cups cleanly
WAIST_PITCH_POUR = 0.16  # rad upright torso pitch during pour so both arms reach MEET_X cleanly
LIFT_TABLE_REF = 0.75  # m tabletop height corresponding to lift = 0.0 m
LIFT_UP = 0.12  # m column raise after grasping both cups (avoids arm-only lift slip)
SLIDE = 0.050  # m compact horizontal approach/retreat (avoids nearby distractors & reach edge)
HOVER = 0.14  # m vertical hover above approach point (clears distractors)

# Mid-air pour geometry
MEET_X = 0.015  # m target cup offset from robot centreline (keeps pouring arm well within wrist-roll range)
MEET_Y = -0.37  # m world y of meet plane (comfortably in front of chest)
POUR_INWARD_YAW = np.deg2rad(30.0)  # yaw source hand inward at meet to align forearm with wrist-roll axis
LIP_CLEAR = 0.026  # m source lip height above target rim during pour (extra clearance against rim graze)
BODY_CLEAR = 0.025  # m minimum vertical clearance of source cup corners above target rim
LIP_NEAR_FRAC = 0.0  # aim dead-center over target mouth (avoids overshoot on narrow target cups)
RETURN_LIFT = 0.035  # m extra vertical lift while un-tilting source cup after pour (prevents return rim touch)
CARRY_SPEED = 0.005  # m per action (0.15 m/s at 30 Hz)

# Pour tilt control (pure wrist roll about forearm axis)
DRY_TILT = np.deg2rad(32.0)
DRY_RATE = 0.018  # rad/action before beans approach rim
POUR_RATE = 0.007  # rad/action during active pour (steady stream, fewer surge spills)
RETURN_RATE = 0.020  # rad/action returning upright
MAX_TILT = np.deg2rad(122.0)  # max commanded delta roll (empties tall cups completely)
MAX_CUP_TILT = np.deg2rad(125.0)  # stop increasing theta once actual cup tilt reaches this angle
POUR_DONE_SHARE = 0.985  # stop pouring once this share of beans has left source cup
STALL_MAX = 45  # actions without new beans leaving source at max tilt before returning
LIP_GAIN = 0.06  # closed-loop lip steering gain (gentle to avoid wrist jerks)
LIP_SLACK = 0.030  # m max closed-loop lip correction
HANDOVER_STEPS = 20  # smooth blend from meet pose into pour_target(theta)

# Real-speed safety limits (protects 5 N·m wrist torque limit with 1.31 kg Sharpa hand; zero motion violations)
FPS = 30.0
ARM_VMAX_LOADED = 0.78  # rad/s while holding cup (< 0.80 rad/s motion audit threshold)
ARM_VMAX_EMPTY = 0.78  # rad/s while empty-handed (< 0.80 rad/s motion audit threshold)
HAND_VMAX = 3.00  # rad/s finger curl/open
WAIST_VMAX = 0.35  # rad/s
LIFT_VMAX = 0.14  # m/s
WAIST_LIMITS = np.array([[-2.65, 2.65], [-0.04, 1.00]], dtype=float)
LIFT_LIMITS = np.array([[0.0, 0.55]], dtype=float)
JOINT_MARGIN = 0.052  # rad margin from soft arm joint limits (> 0.05 rad motion audit threshold)


def level_rotation(cup_rot):
    """Minimal world rotation that brings a tilted cup's +z axis back to world [0, 0, 1] upright."""
    up = cup_rot.apply(np.array([0.0, 0.0, 1.0]))
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-6:
        return Rotation.identity()
    angle = float(np.arctan2(sin_a, float(np.dot(up, z))))
    return Rotation.from_rotvec((axis / sin_a) * angle)


def side_grasp(out, radius):
    """Hand rotation (fingers +y, palm inward -out*x, thumb +z) and ee-frame cup-axis grasp point."""
    mirror = np.array([1.0, -1.0, 1.0]) if out > 0 else np.ones(3)
    fingers = FINGERS * mirror
    palm = PALM * mirror
    thumb = THUMB * mirror
    point = PALM_POINT * mirror
    world = np.array([[0.0, 1.0, 0.0], [-out, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
    rot = Rotation.from_matrix(world @ np.c_[fingers, palm, thumb].T)
    reach = point + (float(radius) + PALM_CLEAR) * palm
    return rot, reach


def adapted_closed_pose(radius, limits=None, extra_pip=0.0, extra_dip=0.0, extra_thumb=0.0, extra_mcp=0.0):
    """FK-calibrated radius-adapted 22D Sharpa Wave wrap pose (~6-10 mm uniform preload across r=0.028..0.055 m).
    `radius` may be a scalar or a 5-tuple (thumb, index, middle, ring, pinky) so tapered cups get per-finger contact.
    Extra PIP/DIP curl hooks the middle & distal phalanges around the +x side opposite the palm so the cup is cradled
    securely underneath when rolled 90-125 deg during pouring."""
    if np.ndim(radius) == 0:
        r_t = r_i = r_m = r_r = r_p = float(radius)
    else:
        r_t, r_i, r_m, r_r, r_p = [float(x) for x in radius]

    def _f(r):
        dr = float(np.clip(r - R_REF, -0.018, 0.020))
        mcp = float(np.clip(0.70 - 13.0 * dr + extra_mcp, 0.50, 1.15))
        pip = float(np.clip(0.87 - 13.0 * dr + extra_pip, 0.66, 1.70))
        dip = float(np.clip(0.57 - 9.0 * dr + extra_dip, 0.40, 1.40))
        return mcp, pip, dip

    dr_t = float(np.clip(r_t - R_REF, -0.018, 0.020))
    t_fe = float(np.clip(0.76 - 18.0 * dr_t + extra_thumb, 0.54, 1.35))
    i_m, i_p, i_d = _f(r_i)
    m_m, m_p, m_d = _f(r_m)
    r_m2, r_p2, r_d = _f(r_r)
    p_m, p_p, p_d = _f(r_p)
    pose = np.r_[
        [t_fe, -0.34, 1.22, -0.27, 1.50],
        [i_m, 0.0, i_p, i_d],
        [m_m, 0.0, m_p, m_d],
        [r_m2, 0.0, r_p2, r_d],
        [0.0, p_m, 0.0, p_p, p_d],
    ]
    if limits is not None:
        lo, hi = np.asarray(limits, dtype=float).T
        pose = np.clip(pose, lo + 1e-4, hi - 1e-4)
    return pose


def clip_hand(pose, limits=None):
    arr = np.asarray(pose, dtype=float).reshape(HAND_DIM)
    if limits is None:
        return arr
    lo, hi = np.asarray(limits, dtype=float).T
    return np.clip(arr, lo + 1e-4, hi - 1e-4)


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def quintic(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def cup_grip_params(cavity, scale):
    """Compute (grip_up, r_grip, r_mouth, r_rim, height) in meters from a cup's 3D cavity/body profile."""
    bottom = float(cavity["bottom"]) * scale
    top = float(cavity["top"]) * scale
    height = top - bottom
    # Keep thumb below rim on 9-10 cm short cups while keeping pinky clear of the desk.
    hi_grip = max(0.044, min(0.078, height - 0.025))
    grip_up = float(np.clip(0.48 * height, 0.044, hi_grip))
    if height < 0.069:
        grip_up = max(0.038, height - 0.014)
    z_local = float(cavity["bottom"]) + grip_up / scale
    r_grip = float(np.interp(z_local, cavity["body"]["z"], cavity["body"]["r"]) * scale)
    r_mouth = float(cavity["r"][-1] * scale)
    r_rim = float(cavity["body"]["r"][-1] * scale)
    return grip_up, r_grip, r_mouth, r_rim, height


def finger_radii(cavity, scale, grip_up):
    """Per-finger cup body radii (thumb, index, middle, ring, pinky) in meters for tapered-cup wrapping."""
    height = (float(cavity["top"]) - float(cavity["bottom"])) * scale
    offsets = (0.024, 0.018, 0.0, -0.018, -0.030)
    radii = []
    for dz in offsets:
        h_f = float(np.clip(grip_up + dz, 0.005, max(0.006, height - 0.005)))
        z_local = float(cavity["bottom"]) + h_f / scale
        radii.append(float(np.interp(z_local, cavity["body"]["z"], cavity["body"]["r"]) * scale))
    return tuple(radii)


def compute_waist_grasp_pitch(table_z, far_y):
    """Adaptive torso forward pitch for grasping cups across low/far tables (un-pitched during Phase 2 lift)."""
    low_dz = max(0.0, LIFT_TABLE_REF - float(table_z))
    high_dz = max(0.0, float(table_z) - LIFT_TABLE_REF)
    pitch_extra = float(np.clip(2.4 * low_dz - 1.0 * high_dz + 1.2 * max(0.0, float(far_y)), -0.10, 0.38))
    return WAIST_PITCH_GRASP + pitch_extra


class HandArm(sd.Arm):
    """sd.Arm with a custom ee-frame grasp point (cup axis offset along Sharpa palm normal)."""

    def __init__(self, env, robot, reach):
        super().__init__(env, robot)
        self.REACH = np.asarray(reach, dtype=float)
        self.hand_limits = robot.hand_limits() if hasattr(robot, "hand_limits") else None


class WholeBodyLimiter:
    """Synchronized real-speed rate limiter for 61D Sharpa Wave wholebody commands."""

    def __init__(self, arms, init_q, init_h, init_waist, init_lift, arm_limits):
        self.arms = arms
        self.q = {k: np.asarray(v, dtype=float).copy() for k, v in init_q.items()}
        self.h = {k: np.asarray(v, dtype=float).copy() for k, v in init_h.items()}
        self.waist = np.asarray(init_waist, dtype=float).copy()
        self.lift = np.asarray(init_lift, dtype=float).reshape(1).copy()
        self.arm_lim = {
            k: np.column_stack([lim[:, 0] + JOINT_MARGIN, lim[:, 1] - JOINT_MARGIN]) for k, lim in arm_limits.items()
        }
        self.hand_lim = {a.grip_key: a.hand_limits for a in arms}
        self.dq_loaded = ARM_VMAX_LOADED / FPS
        self.dq_empty = ARM_VMAX_EMPTY / FPS
        self.dh_max = HAND_VMAX / FPS
        self.dw_max = WAIST_VMAX / FPS
        self.dl_max = LIFT_VMAX / FPS

    def step(self, cmd, loaded=None):
        """Yield n >= 1 rate-limited synchronized actions reaching `cmd`."""
        loaded = loaded or {}
        ratios = [1.0]
        q_tgt, h_tgt = {}, {}
        for a in self.arms:
            if a.arm_key in cmd:
                q = np.clip(np.asarray(cmd[a.arm_key], dtype=float), *self.arm_lim[a.arm_key].T)
                q_tgt[a.arm_key] = q
                cap = self.dq_loaded if loaded.get(a.arm_key, True) else self.dq_empty
                ratios.append(float(np.abs(q - self.q[a.arm_key]).max()) / cap)
            if a.grip_key in cmd:
                h = clip_hand(cmd[a.grip_key], self.hand_lim[a.grip_key])
                h_tgt[a.grip_key] = h
                ratios.append(float(np.abs(h - self.h[a.grip_key]).max()) / self.dh_max)
        w_tgt = np.clip(
            np.asarray(cmd.get("waist_joint_state", self.waist), dtype=float),
            WAIST_LIMITS[:, 0],
            WAIST_LIMITS[:, 1],
        )
        l_tgt = np.clip(
            np.asarray(cmd.get("lift_joint_state", self.lift), dtype=float).reshape(1),
            LIFT_LIMITS[:, 0],
            LIFT_LIMITS[:, 1],
        )
        ratios.append(float(np.abs(w_tgt - self.waist).max()) / self.dw_max)
        ratios.append(float(np.abs(l_tgt - self.lift).max()) / self.dl_max)

        n = max(1, int(math.ceil(max(ratios) - 1e-9)))
        q0 = {k: self.q[k].copy() for k in q_tgt}
        h0 = {k: self.h[k].copy() for k in h_tgt}
        w0, l0 = self.waist.copy(), self.lift.copy()
        for k in range(1, n + 1):
            frac = k / n
            out = {}
            for a in self.arms:
                if a.arm_key in q_tgt:
                    self.q[a.arm_key] = q0[a.arm_key] + (q_tgt[a.arm_key] - q0[a.arm_key]) * frac
                out[a.arm_key] = self.q[a.arm_key].copy()
                if a.grip_key in h_tgt:
                    self.h[a.grip_key] = clip_hand(
                        h0[a.grip_key] + (h_tgt[a.grip_key] - h0[a.grip_key]) * frac,
                        self.hand_lim[a.grip_key],
                    )
                out[a.grip_key] = self.h[a.grip_key].tolist()
            self.waist = w0 + (w_tgt - w0) * frac
            self.lift = l0 + (l_tgt - l0) * frac
            out["waist_joint_state"] = self.waist.tolist()
            out["lift_joint_state"] = self.lift.tolist()
            yield out


def interp_hand(h0, h1, n):
    h0 = np.asarray(h0, dtype=float)
    h1 = np.asarray(h1, dtype=float)
    return [h0 + (h1 - h0) * smoothstep(i / n) for i in range(1, n + 1)]


def carry_steps(a, b, min_steps=16):
    return max(int(math.ceil(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / CARRY_SPEED)), min_steps)


def sharpa_pour_actions(env):
    """Yield 61D wholebody actions for G1-D Sharpa Wave pouring all beans from source_cup into target_cup."""
    lm = env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
    robots = [r for r in env.robot_manager.robot_list if r.type == "target"]
    base_arms = [sd.Arm(env, r) for r in robots]

    # Read cup profiles and initial world poses
    profiles, cup_info = {}, {}
    for label in ("source_cup", "target_cup"):
        obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
        scale = float(np.asarray(getattr(obj, "scale", (1.0, 1.0, 1.0)), float).reshape(-1)[0])
        cavity = obj.instance_config["cavity"]
        profiles[label] = (cavity, scale)
        pos, quat = lm.get_instance_pose(0, label=label)
        pos = sd.as_np(pos)
        rot = sd.from_wxyz(sd.as_np(quat))
        grip_up, r_grip, r_mouth, r_rim, height = cup_grip_params(cavity, scale)
        axis_local = np.r_[np.asarray(cavity["axis"], float) * scale, 0.0]
        axis_world = pos + rot.apply(axis_local)
        bottom_z = pos[2] + float(cavity["bottom"]) * scale
        cup_info[label] = dict(
            pos0=pos,
            rot0=rot,
            scale=scale,
            cavity=cavity,
            grip_up=grip_up,
            r_grip=r_grip,
            r_mouth=r_mouth,
            r_rim=r_rim,
            height=height,
            axis_local=axis_local,
            axis_xy=axis_world[:2].copy(),
            bottom_z=bottom_z,
        )

    src_i = cup_info["source_cup"]
    tgt_i = cup_info["target_cup"]
    side = float(np.sign(src_i["axis_xy"][0]))  # -1 if source on left, +1 if source on right
    if side == 0.0:
        side = -1.0 if src_i["axis_xy"][0] <= tgt_i["axis_xy"][0] else 1.0

    # Assign arms by x-side
    src_idx = min(range(len(base_arms)), key=lambda i: abs(base_arms[i].grasp_point()[0] - src_i["axis_xy"][0]))
    tgt_idx = next(i for i in range(len(base_arms)) if i != src_idx)
    src_out = -1.0 if base_arms[src_idx].arm_key.startswith("left") else 1.0
    tgt_out = -1.0 if base_arms[tgt_idx].arm_key.startswith("left") else 1.0

    s_hand_rot, s_reach = side_grasp(src_out, src_i["r_grip"])
    t_hand_rot, t_reach = side_grasp(tgt_out, tgt_i["r_grip"])
    src_arm = HandArm(env, robots[src_idx], s_reach)
    tgt_arm = HandArm(env, robots[tgt_idx], t_reach)
    arms = [src_arm, tgt_arm]

    # Hand poses (clipped to URDF limits; per-finger radii handle tapered cups cleanly)
    s_radii = finger_radii(src_i["cavity"], src_i["scale"], src_i["grip_up"])
    t_radii = finger_radii(tgt_i["cavity"], tgt_i["scale"], tgt_i["grip_up"])
    s_open = clip_hand(OPEN_POSE, src_arm.hand_limits)
    t_open = clip_hand(OPEN_POSE, tgt_arm.hand_limits)
    s_pre = clip_hand(PRESHAPE_POSE, src_arm.hand_limits)
    t_pre = clip_hand(PRESHAPE_POSE, tgt_arm.hand_limits)
    s_closed = adapted_closed_pose(s_radii, src_arm.hand_limits, extra_pip=0.12, extra_dip=0.14, extra_thumb=0.06)
    t_closed = adapted_closed_pose(t_radii, tgt_arm.hand_limits, extra_pip=0.12, extra_dip=0.14, extra_thumb=0.06)
    s_hold = adapted_closed_pose(
        s_radii, src_arm.hand_limits, extra_pip=0.50, extra_dip=0.60, extra_thumb=0.22, extra_mcp=0.10
    )
    t_hold = adapted_closed_pose(
        t_radii, tgt_arm.hand_limits, extra_pip=0.50, extra_dip=0.60, extra_thumb=0.22, extra_mcp=0.10
    )
    s_thumb = clip_hand(np.r_[s_closed[:THUMB_COUNT], s_pre[THUMB_COUNT:]], src_arm.hand_limits)
    t_thumb = clip_hand(np.r_[t_closed[:THUMB_COUNT], t_pre[THUMB_COUNT:]], tgt_arm.hand_limits)
    # Gentle thumb-first 3-stage wrap on table (avoid over-squeezing while cup rests on desk)
    s_light = clip_hand(
        np.r_[s_closed[:THUMB_COUNT], s_pre[THUMB_COUNT:] + 0.55 * (s_closed[THUMB_COUNT:] - s_pre[THUMB_COUNT:])],
        src_arm.hand_limits,
    )
    t_light = clip_hand(
        np.r_[t_closed[:THUMB_COUNT], t_pre[THUMB_COUNT:] + 0.55 * (t_closed[THUMB_COUNT:] - t_pre[THUMB_COUNT:])],
        tgt_arm.hand_limits,
    )

    # Body state (waist + lift) adapted to table height and cup forward reach
    body_rb = next(r for r in robots if hasattr(r, "body_dims"))
    body0 = env.robot_manager.get_body_state(body_rb, env_idx_list=[0])[0]
    waist0 = np.asarray(body0["waist_joint_state"], dtype=float)
    lift0 = np.asarray(body0["lift_joint_state"], dtype=float).reshape(1)
    table_z = float(min(src_i["bottom_z"], tgt_i["bottom_z"]))
    lift_grasp = np.array([float(np.clip(table_z - LIFT_TABLE_REF, 0.0, 0.25))])
    lift_pour = np.array([float(np.clip(lift_grasp[0] + LIFT_UP, 0.0, 0.45))])
    far_y = max(0.0, -float(min(src_i["axis_xy"][1], tgt_i["axis_xy"][1])) - 0.40)
    waist_grasp = np.array([waist0[0], compute_waist_grasp_pitch(table_z, far_y)])
    waist_pour = np.array([waist0[0], min(waist_grasp[1], WAIST_PITCH_POUR)])

    # Limiter setup
    init_q = {a.arm_key: a.q() for a in arms}
    init_h = {src_arm.grip_key: s_open.copy(), tgt_arm.grip_key: t_open.copy()}
    arm_limits = {a.arm_key: sd.as_np(a.key.data.soft_joint_pos_limits[0, a.cols]).reshape(-1, 2) for a in arms}
    limiter = WholeBodyLimiter(arms, init_q, init_h, waist0, lift0, arm_limits)
    last_q = {a.arm_key: a.q() for a in arms}

    def ik_smooth(arm, pos, rot):
        q = arm.ik(pos, rot, smooth=True)
        last_q[arm.arm_key] = q
        return q

    def cup_pose(label):
        pos, quat = lm.get_instance_pose(0, label=label)
        return sd.as_np(pos), sd.from_wxyz(sd.as_np(quat))

    def bean_points():
        pts = np.asarray(fluid.get_particle_positions()[0], dtype=float).reshape(-1, 3)
        return pts @ cup_profile.rotation(fluid.init_ori).T + np.asarray(fluid.init_pos, float)

    def inside(label):
        pos, rot = cup_pose(label)
        return cup_profile.in_cavity(bean_points(), pos, np.roll(rot.as_quat(), 1), *profiles[label])

    step = 0
    cur_waist = waist0.copy()
    cur_lift = lift0.copy()

    def emit(s_q, s_h, t_q, t_h, waist=None, lift=None, s_loaded=True, t_loaded=True):
        nonlocal step, cur_waist, cur_lift
        if waist is not None:
            cur_waist = np.asarray(waist, dtype=float).copy()
        if lift is not None:
            cur_lift = np.asarray(lift, dtype=float).reshape(1).copy()
        raw = {
            src_arm.arm_key: s_q,
            src_arm.grip_key: s_h,
            tgt_arm.arm_key: t_q,
            tgt_arm.grip_key: t_h,
            "waist_joint_state": cur_waist.tolist(),
            "lift_joint_state": cur_lift.tolist(),
        }
        for out_cmd in limiter.step(raw, loaded={src_arm.arm_key: s_loaded, tgt_arm.arm_key: t_loaded}):
            if step % 30 == 0:
                sp, _ = cup_pose("source_cup")
                tp, _ = cup_pose("target_cup")
                in_s = inside("source_cup").mean()
                in_t = inside("target_cup").mean()
                print(
                    f"[sharpa_pour] step {step:4d} lift={cur_lift[0]:.3f} waist={cur_waist[1]:.2f}"
                    f" src={sp.round(3)} tgt={tp.round(3)} in_s={in_s:.0%} in_t={in_t:.0%}",
                    flush=True,
                )
            step += 1
            yield out_cmd

    s_grip = np.r_[src_i["axis_xy"], src_i["bottom_z"] + src_i["grip_up"]]
    t_grip = np.r_[tgt_i["axis_xy"], tgt_i["bottom_z"] + tgt_i["grip_up"]]
    s_palm_in = np.array([-src_out, 0.0, 0.0])
    t_palm_in = np.array([-tgt_out, 0.0, 0.0])
    s_start = s_grip - SLIDE * s_palm_in
    t_start = t_grip - SLIDE * t_palm_in

    print(
        f"[sharpa_pour] table_z={table_z:.3f} lift {lift0[0]:.3f}->{lift_grasp[0]:.3f}->{lift_pour[0]:.3f}"
        f" | src({src_arm.arm_key}) r={src_i['r_grip']:.3f} grip={s_grip.round(3)}"
        f" | tgt({tgt_arm.arm_key}) r={tgt_i['r_grip']:.3f} grip={t_grip.round(3)}",
        flush=True,
    )

    # ── Phase 0: Whole-body adapt (lower lift, pitch waist forward) while hovering outside cups ──
    s_hover = s_start + np.array([0.0, 0.0, HOVER])
    t_hover = t_start + np.array([0.0, 0.0, HOVER])
    n_adapt = 45
    s_w_path = list(sd.waypoints(src_arm.grasp_point(), src_arm.pose()[1], [(s_hover, s_hand_rot, 0.0, n_adapt)], True))
    t_w_path = list(sd.waypoints(tgt_arm.grasp_point(), tgt_arm.pose()[1], [(t_hover, t_hand_rot, 0.0, n_adapt)], True))
    s_h_adapt = interp_hand(s_open, s_pre, n_adapt)
    t_h_adapt = interp_hand(t_open, t_pre, n_adapt)
    for k in range(n_adapt):
        u = quintic((k + 1) / n_adapt)
        w = waist0 + u * (waist_grasp - waist0)
        lf = lift0 + u * (lift_grasp - lift0)
        yield from emit(
            ik_smooth(src_arm, s_w_path[k][0], s_w_path[k][1]),
            s_h_adapt[k],
            ik_smooth(tgt_arm, t_w_path[k][0], t_w_path[k][1]),
            t_h_adapt[k],
            waist=w,
            lift=lf,
            s_loaded=False,
            t_loaded=False,
        )

    # ── Phase 1: Descend to approach, slide in, thumb-first wrap, then 4-finger wrap ──
    def run_dual_segment(s_keys, t_keys, s_h_seq, t_h_seq, s_loaded=False, t_loaded=False):
        sp = list(sd.waypoints(src_arm.grasp_point(), src_arm.pose()[1], s_keys, True))
        tp = list(sd.waypoints(tgt_arm.grasp_point(), tgt_arm.pose()[1], t_keys, True))
        for k in range(max(len(sp), len(tp))):
            s_pt, s_r, _ = sp[min(k, len(sp) - 1)]
            t_pt, t_r, _ = tp[min(k, len(tp) - 1)]
            sh = s_h_seq[min(k, len(s_h_seq) - 1)]
            th = t_h_seq[min(k, len(t_h_seq) - 1)]
            yield from emit(
                ik_smooth(src_arm, s_pt, s_r),
                sh,
                ik_smooth(tgt_arm, t_pt, t_r),
                th,
                s_loaded=s_loaded,
                t_loaded=t_loaded,
            )

    # Descend + slide in (re-solve from current pose after body adapt)
    yield from run_dual_segment(
        [(s_start, s_hand_rot, 0.0, 25), (s_grip, s_hand_rot, 0.0, 35)],
        [(t_start, t_hand_rot, 0.0, 25), (t_grip, t_hand_rot, 0.0, 35)],
        [s_pre] * 60,
        [t_pre] * 60,
        s_loaded=False,
        t_loaded=False,
    )
    # Settle palms flush against both cups before closing
    for _ in range(15):
        s_err = float(np.linalg.norm(src_arm.grasp_point() - s_grip))
        t_err = float(np.linalg.norm(tgt_arm.grasp_point() - t_grip))
        if max(s_err, t_err) < 0.008:
            break
        yield from emit(
            ik_smooth(src_arm, s_grip, s_hand_rot),
            s_pre,
            ik_smooth(tgt_arm, t_grip, t_hand_rot),
            t_pre,
            s_loaded=False,
            t_loaded=False,
        )
    # If lift bottomed out on a low/far table and either arm still cannot reach, pitch waist further forward
    if (
        max(
            float(np.linalg.norm(src_arm.grasp_point() - s_grip)),
            float(np.linalg.norm(tgt_arm.grasp_point() - t_grip)),
        )
        > 0.012
        and cur_lift[0] <= 0.001
        and cur_waist[1] < 0.70
    ):
        for _ in range(18):
            s_err = float(np.linalg.norm(src_arm.grasp_point() - s_grip))
            t_err = float(np.linalg.norm(tgt_arm.grasp_point() - t_grip))
            if max(s_err, t_err) < 0.008 or cur_waist[1] >= 0.70:
                break
            w_next = np.array([cur_waist[0], min(cur_waist[1] + 0.015, 0.70)])
            yield from emit(
                ik_smooth(src_arm, s_grip, s_hand_rot),
                s_pre,
                ik_smooth(tgt_arm, t_grip, t_hand_rot),
                t_pre,
                waist=w_next,
                s_loaded=False,
                t_loaded=False,
            )
        waist_grasp = cur_waist.copy()
    print(
        f"[sharpa_pour] pre-close err src={np.linalg.norm(src_arm.grasp_point() - s_grip):.3f}m"
        f" tgt={np.linalg.norm(tgt_arm.grasp_point() - t_grip):.3f}m waist={cur_waist[1]:.2f}",
        flush=True,
    )
    # Balanced 3-stage wrap: thumb contact (14) -> gentle fingers (16) -> firm wrap (24) + hold (10)
    s_q_grasp = src_arm.q()
    t_q_grasp = tgt_arm.q()
    s_close_seq = (
        interp_hand(s_pre, s_thumb, 14)
        + interp_hand(s_thumb, s_light, 16)
        + interp_hand(s_light, s_closed, 24)
        + [s_closed] * 10
    )
    t_close_seq = (
        interp_hand(t_pre, t_thumb, 14)
        + interp_hand(t_thumb, t_light, 16)
        + interp_hand(t_light, t_closed, 24)
        + [t_closed] * 10
    )
    for sh, th in zip(s_close_seq, t_close_seq):
        yield from emit(s_q_grasp, sh, t_q_grasp, th, s_loaded=True, t_loaded=True)

    # ── Phase 2: Lift both cups via lift column + un-pitch waist to WAIST_PITCH_POUR ──
    n_lift = 35
    for k in range(n_lift):
        u = quintic((k + 1) / n_lift)
        lf = lift_grasp + u * (lift_pour - lift_grasp)
        w = waist_grasp + u * (waist_pour - waist_grasp)
        yield from emit(s_q_grasp, s_closed, t_q_grasp, t_closed, waist=w, lift=lf, s_loaded=True, t_loaded=True)
    for _ in range(8):
        yield from emit(s_q_grasp, s_closed, t_q_grasp, t_closed, s_loaded=True, t_loaded=True)

    # Verify both cups lifted cleanly; abort early if a cup slipped or tipped during grasp
    s_pos_lifted, s_rot_lifted = cup_pose("source_cup")
    t_pos_lifted, t_rot_lifted = cup_pose("target_cup")
    s_dz_lift = float(s_pos_lifted[2] - src_i["pos0"][2])
    t_dz_lift = float(t_pos_lifted[2] - tgt_i["pos0"][2])
    s_tilt_lift = float(np.degrees(np.arccos(np.clip(s_rot_lifted.as_matrix()[2, 2], -1.0, 1.0))))
    t_tilt_lift = float(np.degrees(np.arccos(np.clip(t_rot_lifted.as_matrix()[2, 2], -1.0, 1.0))))
    print(
        f"[sharpa_pour] post-lift src dz={s_dz_lift:+.3f}m tilt={s_tilt_lift:.1f}deg"
        f" | tgt dz={t_dz_lift:+.3f}m tilt={t_tilt_lift:.1f}deg",
        flush=True,
    )
    if s_dz_lift < 0.04 or t_dz_lift < 0.04 or s_tilt_lift > 42.0 or t_tilt_lift > 42.0:
        print("[sharpa_pour] ABORT: cup failed to lift cleanly after grasp", flush=True)
        return

    # Mid-air firm squeeze: lock both cups securely now that they are clear of the desk
    for sh, th in zip(interp_hand(s_closed, s_hold, 15), interp_hand(t_closed, t_hold, 15)):
        yield from emit(s_q_grasp, sh, t_q_grasp, th, s_loaded=True, t_loaded=True)
    s_pos_lifted, s_rot_lifted = cup_pose("source_cup")
    t_pos_lifted, t_rot_lifted = cup_pose("target_cup")

    # ── Phase 3: Move both cups to mid-air meet pose ──
    # Measure actual cup pose in hand after lift
    s_grasp_c = s_rot_lifted.inv().apply(src_arm.grasp_point() - s_pos_lifted)
    t_grasp_c = t_rot_lifted.inv().apply(tgt_arm.grasp_point() - t_pos_lifted)
    s_hand_lifted = src_arm.pose()[1]
    t_hand_lifted = tgt_arm.pose()[1]

    # Target cup meet position: auto-level the target cup back to 0 deg upright so its full mouth area
    # and fill capacity face upward during pouring, and its base lands flat when placed back on the desk.
    t_level = level_rotation(t_rot_lifted)
    t_hand_level = t_level * t_hand_lifted
    t_rot_level = t_level * t_rot_lifted
    t_air_z = float(tgt_arm.grasp_point()[2])
    t_mouth_local = np.r_[
        np.asarray(tgt_i["cavity"]["axis"], float) * tgt_i["scale"],
        float(tgt_i["cavity"]["top"]) * tgt_i["scale"],
    ]
    t_air_axis_xy = np.array([-side * MEET_X, MEET_Y])
    t_air = np.r_[t_air_axis_xy + (t_rot_level.apply(t_grasp_c - tgt_i["axis_local"]))[:2], t_air_z]

    # Source cup pour geometry: yaw hand slightly inward along natural forearm direction so wrist roll stays
    # far from its +-1.97 rad limit throughout the 90-120 deg pour.
    s_hand_pour = Rotation.from_euler("z", side * POUR_INWARD_YAW) * s_hand_lifted
    s_rot_pour = Rotation.from_euler("z", side * POUR_INWARD_YAW) * s_rot_lifted
    roll_axis = s_hand_pour.apply(FINGERS)
    roll_axis /= max(float(np.linalg.norm(roll_axis)), 1e-6)
    pour_dir = -side * np.cross(roll_axis, np.array([0.0, 0.0, 1.0]))
    pour_dir /= max(float(np.linalg.norm(pour_dir)), 1e-6)
    s_cav = src_i["cavity"]
    s_sc = src_i["scale"]
    s_centre = np.r_[
        np.asarray(s_cav["axis"], float) * s_sc,
        0.5 * (float(s_cav["bottom"]) + float(s_cav["top"])) * s_sc,
    ]
    b_r = float(max(s_cav["body"]["r"])) * s_sc
    half_h = 0.5 * src_i["height"]
    corners_c = s_centre + np.array(
        [[dx, dy, dz] for dx in (-b_r, b_r) for dy in (-b_r, b_r) for dz in (-half_h, half_h)]
    )
    toward_local = s_rot_pour.inv().apply(pour_dir)[:2]
    toward_local /= max(float(np.linalg.norm(toward_local)), 1e-6)
    lip_c = np.r_[
        np.asarray(s_cav["axis"], float) * s_sc + src_i["r_rim"] * toward_local,
        float(s_cav["top"]) * s_sc,
    ]

    def pour_target(theta, mouth_xy, mouth_z):
        """(grasp_point, hand_rot, lip_target) rolling source cup about its pour lip over target mouth."""
        tilt = Rotation.from_rotvec(-side * theta * roll_axis)
        rot = tilt * s_rot_pour
        low = float(rot.apply(corners_c - lip_c)[:, 2].min())
        near_off = LIP_NEAR_FRAC * tgt_i["r_mouth"]
        lip = np.r_[mouth_xy - near_off * pour_dir[:2], mouth_z + max(LIP_CLEAR, BODY_CLEAR - low)]
        return lip + rot.apply(s_grasp_c - lip_c), tilt * s_hand_pour, lip

    est_t_mouth_z = t_air_z + float(t_rot_level.apply(t_mouth_local - t_grasp_c)[2])
    s_air_pos, s_air_rot, _ = pour_target(0.0, t_air_axis_xy, est_t_mouth_z)
    # Source rises straight up to pour height over its own spot before crossing over target rim
    s_cur = src_arm.grasp_point()
    s_high = np.r_[s_cur[:2], max(s_cur[2], s_air_pos[2] + 0.04)]
    t_cur = tgt_arm.grasp_point()
    n_s1, n_s2 = carry_steps(s_cur, s_high), carry_steps(s_high, s_air_pos)
    n_t = max(carry_steps(t_cur, t_air), n_s1)

    yield from run_dual_segment(
        [(s_high, s_hand_lifted, 0.0, n_s1), (s_air_pos, s_air_rot, 0.0, n_s2)],
        [(t_air, t_hand_level, 0.0, n_t)],
        [s_hold] * (n_s1 + n_s2),
        [t_hold] * n_t,
        s_loaded=True,
        t_loaded=True,
    )
    # Brief settle at meet
    s_q_meet, t_q_meet = src_arm.q(), tgt_arm.q()
    for _ in range(10):
        yield from emit(s_q_meet, s_hold, t_q_meet, t_hold, s_loaded=True, t_loaded=True)

    # Re-measure in-hand source cup pose right before tilting
    s_pos_lifted, s_rot_pour = cup_pose("source_cup")
    s_grasp_c = s_rot_pour.inv().apply(src_arm.grasp_point() - s_pos_lifted)
    s_hand_pour = src_arm.pose()[1]
    roll_axis = s_hand_pour.apply(FINGERS)
    roll_axis /= max(float(np.linalg.norm(roll_axis)), 1e-6)
    pour_dir = -side * np.cross(roll_axis, np.array([0.0, 0.0, 1.0]))
    pour_dir /= max(float(np.linalg.norm(pour_dir)), 1e-6)
    toward_local = s_rot_pour.inv().apply(pour_dir)[:2]
    toward_local /= max(float(np.linalg.norm(toward_local)), 1e-6)
    lip_c = np.r_[
        np.asarray(s_cav["axis"], float) * s_sc + src_i["r_rim"] * toward_local,
        float(s_cav["top"]) * s_sc,
    ]

    def t_rim_top_z(mouth, tr):
        """Highest world z of target cup rim, accounting for any residual cup tilt."""
        sin_tilt = float(np.sqrt(max(0.0, 1.0 - float(tr.as_matrix()[2, 2]) ** 2)))
        return float(mouth[2]) + tgt_i["r_rim"] * sin_tilt

    tp0, tr0 = cup_pose("target_cup")
    mouth0 = tp0 + tr0.apply(t_mouth_local)
    zero_pos, _, _ = pour_target(0.0, mouth0[:2], t_rim_top_z(mouth0, tr0))
    handover_off = src_arm.grasp_point() - zero_pos
    handover_idx = 0
    fix = np.zeros(3)

    def steer(theta, extra_z=0.0):
        nonlocal fix, handover_idx
        tp, tr = cup_pose("target_cup")
        mouth = tp + tr.apply(t_mouth_local)
        h_pos, h_rot, lip_tgt = pour_target(theta, mouth[:2], t_rim_top_z(mouth, tr) + extra_z)
        u = min(handover_idx / HANDOVER_STEPS, 1.0)
        ease = 1.0 - smoothstep(u)
        handover_idx = min(handover_idx + 1, HANDOVER_STEPS)
        if ease == 0.0:
            sp, sr = cup_pose("source_cup")
            lip_now = sp + sr.apply(lip_c)
            fix = np.clip(fix + LIP_GAIN * (lip_tgt - lip_now), -LIP_SLACK, LIP_SLACK)
            fix[2] = max(0.0, fix[2])  # never push source cup downward into target rim
        q_s = ik_smooth(src_arm, h_pos + fix + ease * handover_off, h_rot)
        return emit(q_s, s_hold, t_q_meet, t_hold, s_loaded=True, t_loaded=True)

    # ── Phase 4: Closed-loop pour ──
    theta = 0.0
    total_beans = max(len(inside("source_cup")), 1)
    # Dry tilt until beans approach rim or DRY_TILT reached
    while theta < DRY_TILT:
        theta = min(theta + DRY_RATE, DRY_TILT)
        yield from steer(theta)
        sp, sr = cup_pose("source_cup")
        if sp[2] < table_z + 0.06:
            print("[sharpa_pour] ABORT: source cup dropped during dry tilt", flush=True)
            return
        lip_z = float((sp + sr.apply(lip_c))[2])
        if float(bean_points()[:, 2].max()) > lip_z - 0.015:
            break

    prev_out = 1.0 - float(inside("source_cup").mean())
    stall = 0
    while True:
        sp, sr = cup_pose("source_cup")
        if sp[2] < table_z + 0.06:
            print("[sharpa_pour] ABORT: source cup dropped during pour", flush=True)
            return
        cup_tilt = float(np.arccos(np.clip(sr.as_matrix()[2, 2], -1.0, 1.0)))
        now_out = 1.0 - float(inside("source_cup").mean())
        flow = now_out - prev_out
        prev_out = now_out
        if now_out >= POUR_DONE_SHARE:
            break
        if theta >= MAX_TILT or cup_tilt >= MAX_CUP_TILT:
            stall = stall + 1 if flow < 0.002 else 0
            if stall >= STALL_MAX:
                break
        else:
            # Pause tilt when beans are actively flowing to avoid surges/spills
            if flow < 0.003:
                theta = min(theta + POUR_RATE, MAX_TILT)
        yield from steer(theta)

    # Brief dwell at peak tilt for final beans to drop
    for _ in range(15):
        yield from steer(theta)

    # Lift source cup clear of target rim while smoothly returning upright
    ret_idx = 0
    while theta > 0.0:
        theta = max(theta - RETURN_RATE, 0.0)
        ret_idx += 1
        extra_z = RETURN_LIFT * smoothstep(min(ret_idx / 12.0, 1.0))
        yield from steer(theta, extra_z=extra_z)

    # ── Phase 5: Carry both cups back above start spots, lower onto desk, release & retract ──
    s_pos_now, s_rot_now = cup_pose("source_cup")
    t_pos_now, t_rot_now = cup_pose("target_cup")
    s_off_xy = (src_arm.grasp_point() - (s_pos_now + s_rot_now.apply(src_i["axis_local"])))[:2]
    t_off_xy = (tgt_arm.grasp_point() - (t_pos_now + t_rot_now.apply(tgt_i["axis_local"])))[:2]

    # Auto-level source & target cups upright and return to exact LIFT_UP height above start_z
    s_level = level_rotation(s_rot_now)
    s_hand_land = s_level * src_arm.pose()[1]
    t_level_now = level_rotation(t_rot_now)
    t_hand_land = t_level_now * tgt_arm.pose()[1]
    s_over_z = float(src_arm.grasp_point()[2]) - (float(s_pos_now[2]) - (src_i["pos0"][2] + LIFT_UP))
    t_over_z = float(tgt_arm.grasp_point()[2]) - (float(t_pos_now[2]) - (tgt_i["pos0"][2] + LIFT_UP))
    s_over = np.r_[src_i["axis_xy"] + s_off_xy, s_over_z]
    t_over = np.r_[tgt_i["axis_xy"] + t_off_xy, t_over_z]
    n_back = max(carry_steps(src_arm.grasp_point(), s_over), carry_steps(tgt_arm.grasp_point(), t_over))
    yield from run_dual_segment(
        [(s_over, s_hand_land, 0.0, n_back)],
        [(t_over, t_hand_land, 0.0, n_back)],
        [s_hold] * n_back,
        [t_hold] * n_back,
        s_loaded=True,
        t_loaded=True,
    )
    # Re-level both cups in place over their landing spots (corrects any wrist sag accumulated during carry)
    _, s_rot_ov = cup_pose("source_cup")
    _, t_rot_ov = cup_pose("target_cup")
    s_hand_land = level_rotation(s_rot_ov) * src_arm.pose()[1]
    t_hand_land = level_rotation(t_rot_ov) * tgt_arm.pose()[1]
    yield from run_dual_segment(
        [(src_arm.grasp_point(), s_hand_land, 0.0, 15)],
        [(tgt_arm.grasp_point(), t_hand_land, 0.0, 15)],
        [s_hold] * 15,
        [t_hold] * 15,
        s_loaded=True,
        t_loaded=True,
    )

    # Lower both cups smoothly onto the desk using lift column + waist re-pitch (fixed arm joints -> straight down)
    s_pos_ov, _ = cup_pose("source_cup")
    t_pos_ov, _ = cup_pose("target_cup")
    min_above = min(float(s_pos_ov[2] - src_i["pos0"][2]), float(t_pos_ov[2] - tgt_i["pos0"][2]))
    lift_land = np.array([float(np.clip(cur_lift[0] - max(0.0, min_above - 0.008), 0.0, cur_lift[0]))])
    s_q_place = src_arm.q()
    t_q_place = tgt_arm.q()
    n_down = 42
    for k in range(n_down):
        s_pos_c, s_rot_c = cup_pose("source_cup")
        t_pos_c, t_rot_c = cup_pose("target_cup")
        s_ez = float(s_pos_c[2] - src_i["pos0"][2])
        t_ez = float(t_pos_c[2] - tgt_i["pos0"][2])
        s_td = float(np.degrees(np.arccos(np.clip(s_rot_c.as_matrix()[2, 2], -1.0, 1.0))))
        t_td = float(np.degrees(np.arccos(np.clip(t_rot_c.as_matrix()[2, 2], -1.0, 1.0))))
        if min(s_ez, t_ez) <= 0.008 or max(s_td, t_td) > 12.0:
            break
        u = quintic((k + 1) / n_down)
        lf = lift_pour + u * (lift_land - lift_pour)
        w = waist_pour + u * (waist_grasp - waist_pour)
        yield from emit(s_q_place, s_hold, t_q_place, t_hold, waist=w, lift=lf, s_loaded=True, t_loaded=True)
    # Gentle fine vertical touch-down; pitch waist slightly forward if lift bottomed out at 0 and cups still hover
    s_land_pt = src_arm.grasp_point().copy()
    t_land_pt = tgt_arm.grasp_point().copy()
    s_q_dn, t_q_dn = src_arm.q(), tgt_arm.q()
    _, s_rot_dn = cup_pose("source_cup")
    _, t_rot_dn = cup_pose("target_cup")
    s_hand_land = level_rotation(s_rot_dn) * src_arm.pose()[1]
    t_hand_land = level_rotation(t_rot_dn) * tgt_arm.pose()[1]
    for _ in range(65):
        s_pos_d, s_rot_d = cup_pose("source_cup")
        t_pos_d, t_rot_d = cup_pose("target_cup")
        s_err_z = float(s_pos_d[2] - src_i["pos0"][2])
        t_err_z = float(t_pos_d[2] - tgt_i["pos0"][2])
        s_tilt_d = float(np.degrees(np.arccos(np.clip(s_rot_d.as_matrix()[2, 2], -1.0, 1.0))))
        t_tilt_d = float(np.degrees(np.arccos(np.clip(t_rot_d.as_matrix()[2, 2], -1.0, 1.0))))
        s_move = s_err_z > 0.004 and s_tilt_d < 14.0
        t_move = t_err_z > 0.004 and t_tilt_d < 14.0
        if not s_move and not t_move:
            break
        w_cmd = None
        if cur_lift[0] <= 0.001 and cur_waist[1] < 0.68:
            s_stalled = s_move and (float(src_arm.grasp_point()[2]) > s_land_pt[2] + 0.005)
            t_stalled = t_move and (float(tgt_arm.grasp_point()[2]) > t_land_pt[2] + 0.005)
            if (s_stalled or t_stalled) and min(s_err_z, t_err_z) > 0.008:
                w_cmd = np.array([cur_waist[0], min(cur_waist[1] + 0.005, 0.68)])
        if s_move:
            s_land_pt[2] -= min(0.002, s_err_z - 0.002)
            s_q_dn = ik_smooth(src_arm, s_land_pt, s_hand_land)
        if t_move:
            t_land_pt[2] -= min(0.002, t_err_z - 0.002)
            t_q_dn = ik_smooth(tgt_arm, t_land_pt, t_hand_land)
        yield from emit(s_q_dn, s_hold, t_q_dn, t_hold, waist=w_cmd, s_loaded=True, t_loaded=True)

    # Release: 4 fingers open first (22 steps), then thumb opens (16 steps)
    s_q_rel, t_q_rel = src_arm.q(), tgt_arm.q()
    s_rel_seq = interp_hand(s_hold, s_thumb, 22) + interp_hand(s_thumb, s_pre, 16)
    t_rel_seq = interp_hand(t_hold, t_thumb, 22) + interp_hand(t_thumb, t_pre, 16)
    for sh, th in zip(s_rel_seq, t_rel_seq):
        yield from emit(s_q_rel, sh, t_q_rel, th, s_loaded=False, t_loaded=False)

    # Slide outward horizontally away from cups while opening to flat OPEN_POSE, then hover up
    s_ret = src_arm.grasp_point() - SLIDE * s_palm_in
    t_ret = tgt_arm.grasp_point() - SLIDE * t_palm_in
    s_up = s_ret + np.array([0.0, 0.0, HOVER])
    t_up = t_ret + np.array([0.0, 0.0, HOVER])
    n_ret = 30
    n_up = 20
    s_h_ret = interp_hand(s_pre, s_open, n_ret) + [s_open] * n_up
    t_h_ret = interp_hand(t_pre, t_open, n_ret) + [t_open] * n_up
    yield from run_dual_segment(
        [(s_ret, s_hand_land, 0.0, n_ret), (s_up, s_hand_land, 0.0, n_up)],
        [(t_ret, t_hand_land, 0.0, n_ret), (t_up, t_hand_land, 0.0, n_up)],
        s_h_ret,
        t_h_ret,
        s_loaded=False,
        t_loaded=False,
    )
    print(
        f"[sharpa_pour] done at step {step}: in_src={inside('source_cup').sum()}/{total_beans}"
        f" in_tgt={inside('target_cup').sum()}/{total_beans}",
        flush=True,
    )
