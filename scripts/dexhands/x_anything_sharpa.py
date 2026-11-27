"""Scripted 61D wholebody X-Anything experts (pick / place / put_in / handover) for G1-D with Sharpa Wave hands.

Action / state space (61D fixed-base wholebody, no mobile base):
    left_arm (7) + left_hand (22) + right_arm (7) + right_hand (22) + waist [yaw, pitch] (2) + lift (1)

Grasp primitives (selected automatically from object geometry, or overridden via `primitive=`):
    power_wrap    all-5-finger radius-adapted side wrap (tall cylinders/bottles/cups/boxes)
    tripod_pinch  thumb + index + middle precision side-angled pinch, ring/pinky tucked clear of desk
    lateral       thumb-opposed side-edge pinch angled downward, ring/pinky tucked (flat/elongated objects)

Closure uses thumb-first gentle interpolation gated by native Sharpa fingertip elastomer tactile contact forces
(`env.robot_manager.get_tactile_contact`), falling back cleanly to geometric closure if tactile sensors are absent.
All commands pass through `WholeBodyLimiter` so `motion_over_real_limits == 0`.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.dexhands import pour_beans_sharpa as pbs  # noqa: E402
from scripts.g1d import scripted_demo as sd  # noqa: E402
from scripts.objects import handover_anything as ho, place_anything as pa, sharpa_x_anything as sxa  # noqa: E402

HAND_DIM = pbs.HAND_DIM
THUMB_COUNT = pbs.THUMB_COUNT
OPEN_POSE = pbs.OPEN_POSE
PRESHAPE_POSE = pbs.PRESHAPE_POSE

# Primitive hand-pitch angles (rad): dip fingers slightly so tucked ring+pinky clear desk
TRIPOD_PITCH_RAD = float(np.deg2rad(12.0))
LATERAL_PITCH_RAD = float(np.deg2rad(18.0))

# Tactile contact gating parameters (native elastomer normal force magnitudes in N)
CONTACT_FORCE_N = 0.35  # N per fingertip elastomer to count as solid contact
CONTACT_CAP_N = 6.0  # N peak force ceiling (stops closing immediately to prevent crushing/ejection)
CONTACT_HOLD_STEPS = 2  # consecutive steps of opposing contact before locking closure
PRELOAD_DELTA = 0.08  # rad gentle compliant preload added to active joints upon tactile lock

# Approach / lift / carry geometry
SLIDE = 0.055  # m horizontal approach/retreat along palm normal
HOVER = 0.14  # m vertical clearance above distractors
LIFT_UP = 0.12  # m coordinated lift-column raise after grasp
HANDOVER_Y = -0.37  # m mid-air handover plane in front of chest
HANDOVER_Z_ABOVE_TABLE = 0.25  # m handover height above desk
HANDOVER_SEP_MIN = 0.075  # m minimum centre-to-centre hand separation during bimanual handover
WAIST_YAW_START_X = 0.18  # m lateral threshold beyond which torso gently yaws toward target
WAIST_YAW_GAIN = 0.40  # rad per meter beyond threshold
WAIST_YAW_MAX = 0.20  # rad max gentle waist yaw offset


def read_fingertip_forces(env, robot):
    """Return 5D normal contact force magnitudes [thumb, index, middle, ring, pinky] in N, or None if unavailable."""
    rm = getattr(env, "robot_manager", None)
    if rm is None or not hasattr(rm, "get_tactile_contact"):
        return None
    contact = rm.get_tactile_contact(robot, env_idx_list=[0])
    if not contact or 0 not in contact or contact[0] is None:
        return None
    vecs = np.asarray(contact[0].get("normal_force_world_N", ()), dtype=float)
    if vecs.shape != (5, 3):
        return None
    return np.linalg.norm(vecs, axis=-1)


def active_finger_mask(primitive):
    """Boolean mask over the 22D Sharpa hand indicating joints that actively close on the object."""
    mask = np.zeros(HAND_DIM, dtype=bool)
    mask[:THUMB_COUNT] = True  # thumb always active
    if primitive == "power_wrap":
        mask[THUMB_COUNT:] = True
    else:
        # tripod_pinch and lateral: index [5..8] + middle [9..12]; ring/pinky stay tucked
        mask[5:13] = True
    return mask


class TactileClosureGate:
    """Monitors 5-finger elastomer contact forces during gentle closure and locks with a small compliant preload."""

    def __init__(
        self,
        primitive,
        limits=None,
        contact_force_n=CONTACT_FORCE_N,
        contact_cap_n=CONTACT_CAP_N,
        hold_steps=CONTACT_HOLD_STEPS,
        preload_delta=PRELOAD_DELTA,
    ):
        self.primitive = primitive
        self.limits = limits
        self.contact_force_n = float(contact_force_n)
        self.contact_cap_n = float(contact_cap_n)
        self.hold_steps = int(hold_steps)
        self.preload_delta = float(preload_delta)
        self.mask = active_finger_mask(primitive)
        self.opp_slice = slice(1, 5) if primitive == "power_wrap" else slice(1, 3)
        self.contact_run = 0
        self.locked = False
        self.locked_pose = None
        self.trigger_reason = None
        self.peak_force_n = 0.0

    def update(self, forces, current_q):
        """Feed latest 5D fingertip forces (or None) and current 22D hand command; returns locked pose if triggered."""
        if self.locked:
            return self.locked_pose
        if forces is None:
            return None
        f = np.asarray(forces, dtype=float).reshape(5)
        peak = float(np.max(f))
        self.peak_force_n = max(self.peak_force_n, peak)
        thumb_ok = float(f[0]) >= self.contact_force_n
        opp_ok = float(np.max(f[self.opp_slice])) >= self.contact_force_n
        if peak >= self.contact_cap_n:
            self.locked = True
            self.trigger_reason = "force_cap"
            self.locked_pose = pbs.clip_hand(np.asarray(current_q, dtype=float), self.limits)
            return self.locked_pose
        if thumb_ok and opp_ok:
            self.contact_run += 1
        else:
            self.contact_run = 0
        if self.contact_run >= self.hold_steps:
            self.locked = True
            self.trigger_reason = "opposing_contact"
            preloaded = np.asarray(current_q, dtype=float).copy()
            # Add gentle compliant preload only to active flexion joints (thumb CMC_FE/MCP_FE/IP & opposing PIP/DIP)
            flex_idx = (
                [0, 2, 4, 7, 8, 11, 12] if self.primitive != "power_wrap" else [0, 2, 4, 7, 8, 11, 12, 15, 16, 20, 21]
            )
            for idx in flex_idx:
                preloaded[idx] += self.preload_delta
            self.locked_pose = pbs.clip_hand(preloaded, self.limits)
            return self.locked_pose
        return None


def primitive_hand_poses(primitive, half_width, limits=None):
    """Return (open_q, preshape_q, thumb_q, closed_q) 22D Sharpa hand targets for `primitive` and `half_width`."""
    r = float(np.clip(half_width, 0.012, 0.045))
    open_q = pbs.clip_hand(OPEN_POSE, limits)
    if primitive == "power_wrap":
        pre_q = pbs.clip_hand(PRESHAPE_POSE, limits)
        closed_q = pbs.adapted_closed_pose(r, limits, extra_pip=0.14, extra_dip=0.16, extra_thumb=0.08)
        thumb_q = pbs.clip_hand(np.r_[closed_q[:THUMB_COUNT], pre_q[THUMB_COUNT:]], limits)
        return open_q, pre_q, thumb_q, closed_q

    if primitive == "tripod_pinch":
        tucked_ring = [1.15, 0.0, 1.35, 0.95]
        tucked_pinky = [0.0, 1.15, 0.0, 1.35, 0.95]
        pre_raw = np.r_[
            [0.05, -0.32, 1.25, -0.28, 1.45],
            [0.08, 0.0, 0.25, 0.15] * 2,
            tucked_ring,
            tucked_pinky,
        ]
        base_closed = pbs.adapted_closed_pose(r, None, extra_pip=0.20, extra_dip=0.22, extra_thumb=0.12)
        closed_raw = np.r_[base_closed[:13], tucked_ring, tucked_pinky]
        pre_q = pbs.clip_hand(pre_raw, limits)
        closed_q = pbs.clip_hand(closed_raw, limits)
        thumb_q = pbs.clip_hand(np.r_[closed_q[:THUMB_COUNT], pre_q[THUMB_COUNT:]], limits)
        return open_q, pre_q, thumb_q, closed_q

    if primitive == "lateral":
        tucked_ring = [1.25, 0.0, 1.45, 1.00]
        tucked_pinky = [0.0, 1.25, 0.0, 1.45, 1.00]
        pre_raw = np.r_[
            [0.10, -0.28, 0.95, -0.20, 1.10],
            [0.25, 0.0, 0.35, 0.20] * 2,
            tucked_ring,
            tucked_pinky,
        ]
        base_closed = pbs.adapted_closed_pose(
            max(0.014, r - 0.004), None, extra_mcp=0.15, extra_pip=0.25, extra_dip=0.25, extra_thumb=0.18
        )
        closed_raw = np.r_[base_closed[:13], tucked_ring, tucked_pinky]
        pre_q = pbs.clip_hand(pre_raw, limits)
        closed_q = pbs.clip_hand(closed_raw, limits)
        thumb_q = pbs.clip_hand(np.r_[closed_q[:THUMB_COUNT], pre_q[THUMB_COUNT:]], limits)
        return open_q, pre_q, thumb_q, closed_q

    raise ValueError(f"Unknown Sharpa grasp primitive: {primitive!r}")


def primitive_grasp_frame(primitive, out, half_width, height):
    """Return (hand_rot, reach_ee, grip_up) for `primitive` on side `out` (-1 left, +1 right)."""
    r = float(np.clip(half_width, 0.012, 0.045))
    h = float(max(height, 0.024))
    rot, reach = pbs.side_grasp(out, r)
    if primitive == "power_wrap":
        hi_grip = max(0.044, min(0.078, h - 0.022))
        grip_up = float(np.clip(0.48 * h, 0.044, hi_grip))
        return rot, reach, grip_up
    if primitive == "tripod_pinch":
        # Pitch fingers slightly downward (-TRIPOD_PITCH_RAD around world +x) so thumb/index/middle reach compact items
        rot = Rotation.from_euler("x", -TRIPOD_PITCH_RAD) * rot
        grip_up = float(np.clip(0.54 * h, 0.036, max(0.036, h - 0.010)))
        return rot, reach, grip_up
    if primitive == "lateral":
        rot, reach = pbs.side_grasp(out, max(0.012, r - 0.004))
        rot = Rotation.from_euler("x", -LATERAL_PITCH_RAD) * rot
        grip_up = float(np.clip(0.58 * h, 0.032, max(0.032, h - 0.008)))
        return rot, reach, grip_up
    raise ValueError(f"Unknown Sharpa grasp primitive: {primitive!r}")


def resolve_primitive(size, mass=None, override=None):
    """Choose a valid Sharpa grasp primitive for any object geometry, falling back gracefully for out-of-pool items."""
    if override in sxa.PRIMITIVES:
        return override
    prim = sxa.select_primitive(size, mass=mass)
    if prim is not None:
        return prim
    h = float(size[2])
    if h >= 0.072:
        return "power_wrap"
    if h >= 0.036:
        return "tripod_pinch"
    return "lateral"


def compute_wholebody_grasp_pose(table_z, target_xy, waist0):
    """Compute adaptive (waist_grasp [yaw, pitch], lift_grasp [height]) for reaching target_xy on desk table_z."""
    tx, ty = float(target_xy[0]), float(target_xy[1])
    lift_grasp = np.array([float(np.clip(float(table_z) - pbs.LIFT_TABLE_REF, 0.0, 0.25))], dtype=float)
    # Robot base sits at y=-0.73 facing +y: larger ty (e.g. -0.32) is farther across the desk than -0.47.
    far_y = max(0.0, ty + 0.40)
    pitch = pbs.compute_waist_grasp_pitch(table_z, far_y)
    # Gentle waist yaw toward objects near lateral workspace edge: robot faces +y and torso_Joint turns about +z,
    # so +yaw turns left (-x) and -yaw turns right (+x).
    excess_x = max(0.0, abs(tx) - WAIST_YAW_START_X)
    yaw_delta = float(np.clip(-np.sign(tx) * WAIST_YAW_GAIN * excess_x, -WAIST_YAW_MAX, WAIST_YAW_MAX))
    yaw = float(np.clip(float(waist0[0]) + yaw_delta, pbs.WAIST_LIMITS[0, 0], pbs.WAIST_LIMITS[0, 1]))
    return np.array([yaw, pitch], dtype=float), lift_grasp


class _WholeBodySession:
    """Shared 61D Sharpa wholebody controller for pick / place / put_in / handover experts."""

    def __init__(self, env, reaches):
        self.env = env
        self.lm = env.scene_manager.layout_manager
        self.robots = [r for r in env.robot_manager.robot_list if r.type == "target"]
        self.arms = [pbs.HandArm(env, r, reaches[i]) for i, r in enumerate(self.robots)]
        self.by_side = {("left" if a.arm_key.startswith("left") else "right"): i for i, a in enumerate(self.arms)}
        body_rb = next(r for r in self.robots if hasattr(r, "body_dims"))
        body0 = env.robot_manager.get_body_state(body_rb, env_idx_list=[0])[0]
        self.waist0 = np.asarray(body0["waist_joint_state"], dtype=float)
        self.lift0 = np.asarray(body0["lift_joint_state"], dtype=float).reshape(1)
        self.cur_waist = self.waist0.copy()
        self.cur_lift = self.lift0.copy()
        self.q = {a.arm_key: a.q() for a in self.arms}
        self.h = {a.grip_key: pbs.clip_hand(OPEN_POSE, a.hand_limits) for a in self.arms}
        arm_limits = {
            a.arm_key: sd.as_np(a.key.data.soft_joint_pos_limits[0, a.cols]).reshape(-1, 2) for a in self.arms
        }
        self.limiter = pbs.WholeBodyLimiter(self.arms, self.q, self.h, self.waist0, self.lift0, arm_limits)

    def emit(self, q_map=None, h_map=None, waist=None, lift=None, loaded=None):
        if q_map:
            for k, v in q_map.items():
                self.q[k] = np.asarray(v, dtype=float).copy()
        if h_map:
            for k, v in h_map.items():
                self.h[k] = np.asarray(v, dtype=float).copy()
        if waist is not None:
            self.cur_waist = np.asarray(waist, dtype=float).copy()
        if lift is not None:
            self.cur_lift = np.asarray(lift, dtype=float).reshape(1).copy()
        raw = {
            **{a.arm_key: self.q[a.arm_key] for a in self.arms},
            **{a.grip_key: self.h[a.grip_key] for a in self.arms},
            "waist_joint_state": self.cur_waist.tolist(),
            "lift_joint_state": self.cur_lift.tolist(),
        }
        yield from self.limiter.step(raw, loaded=loaded)

    def move_arm(self, idx, target_pos, target_rot, hand_q=None, waist=None, lift=None, steps=None, loaded=False):
        arm = self.arms[idx]
        p0, r0 = arm.grasp_point(), arm.pose()[1]
        p1 = np.asarray(target_pos, dtype=float)
        n = pbs.carry_steps(p0, p1) if steps is None else max(1, int(steps))
        h0 = self.h[arm.grip_key].copy()
        h1 = h0 if hand_q is None else np.asarray(hand_q, dtype=float)
        w0 = self.cur_waist.copy()
        w1 = w0 if waist is None else np.asarray(waist, dtype=float)
        l0 = self.cur_lift.copy()
        l1 = l0 if lift is None else np.asarray(lift, dtype=float).reshape(1)
        slerp_rot = (target_rot * r0.inv()).as_rotvec()
        for i in range(1, n + 1):
            u = pbs.smoothstep(i / n)
            pt = p0 + (p1 - p0) * u
            rt = Rotation.from_rotvec(slerp_rot * u) * r0
            q_ik = arm.ik(pt, rt, smooth=True)
            yield from self.emit(
                q_map={arm.arm_key: q_ik},
                h_map={arm.grip_key: h0 + (h1 - h0) * u},
                waist=w0 + (w1 - w0) * u,
                lift=l0 + (l1 - l0) * u,
                loaded={arm.arm_key: loaded},
            )

    def close_with_tactile_gate(self, idx, primitive, pre_q, thumb_q, closed_q, thumb_steps=12, finger_steps=26):
        """Thumb-first gentle closure gated by fingertip tactile contact."""
        arm = self.arms[idx]
        robot = self.robots[idx]
        gate = TactileClosureGate(primitive, limits=arm.hand_limits)
        stage2_start = np.asarray(thumb_q, dtype=float).copy()
        stage2_end = np.asarray(closed_q, dtype=float).copy()
        # Stage 1: thumb-first onto the object wall
        for h_t in pbs.interp_hand(pre_q, thumb_q, thumb_steps):
            forces = read_fingertip_forces(self.env, robot)
            locked = gate.update(forces, h_t)
            if locked is not None:
                if gate.trigger_reason == "opposing_contact":
                    yield from self.emit(h_map={arm.grip_key: locked}, loaded={arm.arm_key: True})
                    return gate
                # Force cap on thumb alone: freeze thumb at current angle, then still close opposing fingers
                stage2_start = np.asarray(h_t, dtype=float).copy()
                stage2_end[:THUMB_COUNT] = stage2_start[:THUMB_COUNT]
                gate = TactileClosureGate(primitive, limits=arm.hand_limits)
                yield from self.emit(h_map={arm.grip_key: stage2_start}, loaded={arm.arm_key: True})
                break
            yield from self.emit(h_map={arm.grip_key: h_t}, loaded={arm.arm_key: True})
        # Stage 2: gentle opposing finger wrap until tactile lock or full geometric closure
        for h_t in pbs.interp_hand(stage2_start, stage2_end, finger_steps):
            forces = read_fingertip_forces(self.env, robot)
            locked = gate.update(forces, h_t)
            if locked is not None:
                yield from self.emit(h_map={arm.grip_key: locked}, loaded={arm.arm_key: True})
                return gate
            yield from self.emit(h_map={arm.grip_key: h_t}, loaded={arm.arm_key: True})
        return gate

    def hold_steps(self, n, waist=None, lift=None, loaded=None):
        w0, l0 = self.cur_waist.copy(), self.cur_lift.copy()
        w1 = w0 if waist is None else np.asarray(waist, dtype=float)
        l1 = l0 if lift is None else np.asarray(lift, dtype=float).reshape(1)
        for i in range(1, max(1, int(n)) + 1):
            u = pbs.smoothstep(i / max(1, int(n)))
            yield from self.emit(waist=w0 + (w1 - w0) * u, lift=l0 + (l1 - l0) * u, loaded=loaded)


def _target_geometry(env):
    """Read target world centre, rotation, local extents (w, d, h), bottom_z, and mass."""
    lm = env.scene_manager.layout_manager
    centre, rot, ext, bottom, _ = sd.Robot.object(type("R", (), {"lm": lm})(), "target")
    entries = env.entries(0) if hasattr(env, "entries") else {}
    phys = (entries.get("target") or {}).get("physics") or {}
    mass = phys.get("mass")
    return centre, rot, ext, float(bottom), (None if mass is None else float(mass))


def _grasp_plan(env, side=None, primitive=None):
    """Compute active side, primitive, half_width, hand frame, grip point, and table_z for target."""
    centre, rot, ext, bottom, mass = _target_geometry(env)
    prim = resolve_primitive(ext, mass=mass, override=primitive)
    # Projected half-extent along world x (palm normal) so rotated elongated objects don't penetrate the palm
    rmat = rot.as_matrix()
    hx = 0.5 * (abs(rmat[0, 0]) * float(ext[0]) + abs(rmat[0, 1]) * float(ext[1]) + abs(rmat[0, 2]) * float(ext[2]))
    half_width = float(np.clip(hx, 0.012, 0.045))
    if side is None:
        side = "left" if float(centre[0]) <= 0.0 else "right"
    out = -1.0 if side == "left" else 1.0
    hand_rot, reach, grip_up = primitive_grasp_frame(prim, out, half_width, float(ext[2]))
    grip = np.array([float(centre[0]), float(centre[1]), bottom + grip_up], dtype=float)
    palm_in = np.array([-out, 0.0, 0.0], dtype=float)
    return dict(
        centre=centre,
        rot=rot,
        ext=ext,
        bottom=bottom,
        mass=mass,
        primitive=prim,
        half_width=half_width,
        side=side,
        out=out,
        hand_rot=hand_rot,
        reach=reach,
        grip_up=grip_up,
        grip=grip,
        palm_in=palm_in,
    )


def sharpa_pick_anything_actions(env, primitive=None):
    """Yield 61D wholebody actions for g1d_pick_anything with Sharpa Wave hands."""
    plan = _grasp_plan(env, primitive=primitive)
    # Build two-arm reaches (active arm uses object-adapted reach; idle arm uses default side_grasp)
    robots = [r for r in env.robot_manager.robot_list if r.type == "target"]
    reaches = []
    for r in robots:
        side = "left" if "left" in r.arm_name else "right"
        if side == plan["side"]:
            reaches.append(plan["reach"])
        else:
            _, idle_reach = pbs.side_grasp(-1.0 if side == "left" else 1.0, pbs.R_REF)
            reaches.append(idle_reach)
    sess = _WholeBodySession(env, reaches)
    idx = sess.by_side[plan["side"]]
    arm = sess.arms[idx]

    open_q, pre_q, thumb_q, closed_q = primitive_hand_poses(plan["primitive"], plan["half_width"], arm.hand_limits)
    waist_grasp, lift_grasp = compute_wholebody_grasp_pose(plan["bottom"], plan["centre"][:2], sess.waist0)
    lift_up = np.array([float(np.clip(lift_grasp[0] + LIFT_UP, 0.0, 0.45))], dtype=float)
    waist_lifted = np.array([waist_grasp[0], min(waist_grasp[1], pbs.WAIST_PITCH_POUR)], dtype=float)

    start = plan["grip"] - SLIDE * plan["palm_in"]
    # 1. Hover above start while adapting waist + lift + preshaping hand
    yield from sess.move_arm(
        idx,
        start + np.array([0.0, 0.0, HOVER]),
        plan["hand_rot"],
        hand_q=pre_q,
        waist=waist_grasp,
        lift=lift_grasp,
        steps=45,
    )
    # 2. Lower to side approach point
    yield from sess.move_arm(idx, start, plan["hand_rot"], hand_q=pre_q, steps=24)
    # 3. Slide horizontally into grasp
    yield from sess.move_arm(idx, plan["grip"], plan["hand_rot"], hand_q=pre_q, steps=26)
    # 4. Thumb-first gentle closure with tactile contact gating
    yield from sess.close_with_tactile_gate(idx, plan["primitive"], pre_q, thumb_q, closed_q)
    # 5. Short settle hold before lifting
    yield from sess.hold_steps(8, loaded={arm.arm_key: True})
    # 6. Coordinated lift-column raise (keeps arm joints steady to avoid IK slip, then slight extra arm lift)
    yield from sess.hold_steps(36, waist=waist_lifted, lift=lift_up, loaded={arm.arm_key: True})
    # 7. Hold lifted cleanly for acceptance window
    yield from sess.hold_steps(35, loaded={arm.arm_key: True})


def _receptacle_target(env, kind, grip_up):
    """Compute world placement (x, y, z), safe hover_z above receptacle, and receptacle base z."""
    lm = env.scene_manager.layout_manager
    entries = env.entries(0)
    rec_geom = entries[pa.LABEL]["receptacle"]
    pos, quat = lm.get_instance_pose(0, label=pa.LABEL)
    pos = sd.as_np(pos)
    rot = sd.from_wxyz(sd.as_np(quat))
    center_local = np.array([float(rec_geom["region"]["center"][0]), float(rec_geom["region"]["center"][1]), 0.0])
    xy = (pos + rot.apply(center_local))[:2]
    floor_z = float(pos[2] + rec_geom["floor"])
    rim_z = float(pos[2] + rec_geom["rim"])
    top_z = float(pos[2] + rec_geom["top"])
    base_z = float(pos[2] + rec_geom.get("bottom", 0.0))
    if kind == "container":
        place_z = rim_z + max(0.015, grip_up - 0.008)
    else:
        place_z = floor_z + grip_up + 0.006
    hover_z = max(top_z + 0.12, place_z + 0.10)
    return np.array([float(xy[0]), float(xy[1]), place_z], dtype=float), float(hover_z), base_z


def sharpa_place_anything_actions(env, kind="support", primitive=None):
    """Yield 61D wholebody actions for g1d_place_anything (support) or g1d_put_anything_in (container)."""
    plan = _grasp_plan(env, primitive=primitive)
    robots = [r for r in env.robot_manager.robot_list if r.type == "target"]
    reaches = []
    for r in robots:
        side = "left" if "left" in r.arm_name else "right"
        if side == plan["side"]:
            reaches.append(plan["reach"])
        else:
            _, idle_reach = pbs.side_grasp(-1.0 if side == "left" else 1.0, pbs.R_REF)
            reaches.append(idle_reach)
    sess = _WholeBodySession(env, reaches)
    idx = sess.by_side[plan["side"]]
    arm = sess.arms[idx]

    open_q, pre_q, thumb_q, closed_q = primitive_hand_poses(plan["primitive"], plan["half_width"], arm.hand_limits)
    waist_grasp, lift_grasp = compute_wholebody_grasp_pose(plan["bottom"], plan["centre"][:2], sess.waist0)
    lift_up = np.array([float(np.clip(lift_grasp[0] + LIFT_UP, 0.0, 0.45))], dtype=float)
    waist_carry = np.array([sess.waist0[0], min(waist_grasp[1], pbs.WAIST_PITCH_POUR)], dtype=float)

    start = plan["grip"] - SLIDE * plan["palm_in"]
    # 1. Grasp & lift
    yield from sess.move_arm(
        idx,
        start + np.array([0.0, 0.0, HOVER]),
        plan["hand_rot"],
        hand_q=pre_q,
        waist=waist_grasp,
        lift=lift_grasp,
        steps=45,
    )
    yield from sess.move_arm(idx, start, plan["hand_rot"], hand_q=pre_q, steps=24)
    yield from sess.move_arm(idx, plan["grip"], plan["hand_rot"], hand_q=pre_q, steps=26)
    yield from sess.close_with_tactile_gate(idx, plan["primitive"], pre_q, thumb_q, closed_q)
    yield from sess.hold_steps(8, loaded={arm.arm_key: True})
    yield from sess.hold_steps(34, waist=waist_carry, lift=lift_up, loaded={arm.arm_key: True})

    # 2. Carry above receptacle & lower with receptacle-adapted waist/lift
    place_pt, hover_z, rec_z = _receptacle_target(env, kind, plan["grip_up"])
    waist_place, lift_place = compute_wholebody_grasp_pose(rec_z, place_pt[:2], sess.waist0)
    above_rec = np.array([place_pt[0], place_pt[1], hover_z], dtype=float)
    yield from sess.move_arm(idx, above_rec, plan["hand_rot"], loaded=True)
    yield from sess.move_arm(idx, place_pt, plan["hand_rot"], waist=waist_place, lift=lift_place, steps=36, loaded=True)
    yield from sess.hold_steps(8, loaded={arm.arm_key: True})

    # 3. Release fingers-first (preshape -> open) and retreat cleanly
    cur_h = sess.h[arm.grip_key].copy()
    for h_t in pbs.interp_hand(cur_h, pre_q, 14):
        yield from sess.emit(h_map={arm.grip_key: h_t}, loaded={arm.arm_key: False})
    for h_t in pbs.interp_hand(pre_q, open_q, 14):
        yield from sess.emit(h_map={arm.grip_key: h_t}, loaded={arm.arm_key: False})
    retreat = place_pt - SLIDE * plan["palm_in"]
    yield from sess.move_arm(idx, retreat, plan["hand_rot"], hand_q=open_q, steps=24, loaded=False)
    yield from sess.move_arm(idx, retreat + np.array([0.0, 0.0, HOVER]), plan["hand_rot"], hand_q=open_q, steps=24)
    yield from sess.hold_steps(35, loaded={arm.arm_key: False})


def sharpa_put_anything_in_actions(env, primitive=None):
    """Yield 61D wholebody actions for g1d_put_anything_in (container)."""
    yield from sharpa_place_anything_actions(env, kind="container", primitive=primitive)


def _handover_plan(centre, rot, ext, bottom, g_plan, t_plan):
    """Compute physically consistent giver/taker grasp points and mid-air handover frames.

    - Tall objects (h >= 0.095 m and long_xy < 0.09 m, or h >= 0.105 m): vertical upper/lower handover at x=0 so both
      hands wrap the actual object with >= 55 mm vertical separation (> CLASH = 30 mm).
    - Lengthwise horizontal objects (long_xy >= 0.09 m): giver grasps off-center at -s along the object's horizontal
      long axis and rotates in yaw to align that axis with world x at handover, placing object centre at x=0, giver end
      at g_out*s, and taker end at t_out*s (2*s >= 60 mm > CLASH = 30 mm, both hands on the object).
    """
    h = float(ext[2])
    long_side = float(max(ext[0], ext[1]))
    if h >= 0.095 and (long_side < 0.09 or h >= 0.105):
        grip_up_g = float(np.clip(0.32 * h, 0.036, max(0.036, h - 0.068)))
        dz_ho = float(np.clip(h - grip_up_g - 0.022, 0.055, 0.085))
        g_grip = np.array([float(centre[0]), float(centre[1]), bottom + grip_up_g], dtype=float)
        return dict(
            mode="vertical",
            g_grip=g_grip,
            g_ho_offset=np.array([0.0, 0.0, 0.0], dtype=float),
            t_ho_offset=np.array([0.0, 0.0, dz_ho], dtype=float),
            g_ho_rot=g_plan["hand_rot"],
            taker_grip_up=grip_up_g + dz_ho,
        )

    # Lengthwise horizontal handover
    local_long = np.array([1.0, 0.0, 0.0]) if ext[0] >= ext[1] else np.array([0.0, 1.0, 0.0])
    axis_w = rot.apply(local_long)
    axis_xy = np.array([float(axis_w[0]), float(axis_w[1])], dtype=float)
    norm = float(np.linalg.norm(axis_xy))
    axis_xy = axis_xy / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=float)
    # Orient axis_xy from giver end toward taker end so yaw alignment stays within [-pi/2, pi/2]
    target_dir = np.array([float(t_plan["out"]), 0.0], dtype=float)
    if float(np.dot(axis_xy, target_dir)) < 0.0:
        axis_xy = -axis_xy
    s = float(min(ho.shift(ext), max(0.030, long_side * 0.5 - 0.015)))
    g_grip = np.array(
        [float(centre[0]) - s * axis_xy[0], float(centre[1]) - s * axis_xy[1], bottom + g_plan["grip_up"]],
        dtype=float,
    )
    cur_yaw = float(np.arctan2(axis_xy[1], axis_xy[0]))
    tgt_yaw = float(np.arctan2(target_dir[1], target_dir[0]))
    delta_yaw = float((tgt_yaw - cur_yaw + np.pi) % (2.0 * np.pi) - np.pi)
    g_ho_rot = Rotation.from_euler("z", delta_yaw) * g_plan["hand_rot"]
    return dict(
        mode="horizontal",
        g_grip=g_grip,
        g_ho_offset=np.array([g_plan["out"] * s, 0.0, 0.0], dtype=float),
        t_ho_offset=np.array([t_plan["out"] * s, 0.0, 0.0], dtype=float),
        g_ho_rot=g_ho_rot,
        taker_grip_up=t_plan["grip_up"],
    )


def sharpa_handover_anything_actions(env, primitive=None):
    """Yield 61D wholebody bimanual handover actions for g1d_handover_anything."""
    centre, rot, ext, bottom, _mass = _target_geometry(env)
    giver_side = ho.hand_of(float(centre[0]))
    taker_side = ho.other(giver_side)
    g_plan = _grasp_plan(env, side=giver_side, primitive=primitive)
    t_plan = _grasp_plan(env, side=taker_side, primitive=g_plan["primitive"])
    ho_plan = _handover_plan(centre, rot, ext, bottom, g_plan, t_plan)

    robots = [r for r in env.robot_manager.robot_list if r.type == "target"]
    reaches = [g_plan["reach"] if ("left" in r.arm_name) == (giver_side == "left") else t_plan["reach"] for r in robots]
    sess = _WholeBodySession(env, reaches)
    g_idx = sess.by_side[giver_side]
    t_idx = sess.by_side[taker_side]
    g_arm, t_arm = sess.arms[g_idx], sess.arms[t_idx]

    g_open, g_pre, g_thumb, g_closed = primitive_hand_poses(
        g_plan["primitive"], g_plan["half_width"], g_arm.hand_limits
    )
    t_open, t_pre, t_thumb, t_closed = primitive_hand_poses(
        t_plan["primitive"], t_plan["half_width"], t_arm.hand_limits
    )

    waist_grasp, lift_grasp = compute_wholebody_grasp_pose(bottom, ho_plan["g_grip"][:2], sess.waist0)
    lift_ho = np.array([float(np.clip(lift_grasp[0] + LIFT_UP, 0.0, 0.45))], dtype=float)
    waist_ho = np.array([sess.waist0[0], min(waist_grasp[1], pbs.WAIST_PITCH_POUR)], dtype=float)

    # 1. Giver grasps target (lower half for tall items, off-center end for horizontal items) and lifts
    g_grip = ho_plan["g_grip"]
    g_start = g_grip - SLIDE * g_plan["palm_in"]
    yield from sess.move_arm(
        g_idx,
        g_start + np.array([0.0, 0.0, HOVER]),
        g_plan["hand_rot"],
        hand_q=g_pre,
        waist=waist_grasp,
        lift=lift_grasp,
        steps=45,
    )
    yield from sess.move_arm(g_idx, g_start, g_plan["hand_rot"], hand_q=g_pre, steps=24)
    yield from sess.move_arm(g_idx, g_grip, g_plan["hand_rot"], hand_q=g_pre, steps=26)
    yield from sess.close_with_tactile_gate(g_idx, g_plan["primitive"], g_pre, g_thumb, g_closed)
    yield from sess.hold_steps(8, loaded={g_arm.arm_key: True})
    yield from sess.hold_steps(30, waist=waist_ho, lift=lift_ho, loaded={g_arm.arm_key: True})

    # 2. Giver carries target to mid-air handover point with physically consistent object placement
    ho_z = max(bottom + HANDOVER_Z_ABOVE_TABLE, float(g_arm.grasp_point()[2]))
    ho_center = np.array([0.0, HANDOVER_Y, ho_z], dtype=float)
    g_ho_pt = ho_center + ho_plan["g_ho_offset"]
    t_ho_pt = ho_center + ho_plan["t_ho_offset"]
    yield from sess.move_arm(g_idx, g_ho_pt, ho_plan["g_ho_rot"], loaded=True)

    # 3. Taker approaches from opposite side and executes tactile-gated closure
    t_approach = t_ho_pt - (SLIDE + 0.03) * t_plan["palm_in"]
    yield from sess.move_arm(t_idx, t_approach, t_plan["hand_rot"], hand_q=t_pre, steps=35, loaded=False)
    yield from sess.move_arm(t_idx, t_ho_pt, t_plan["hand_rot"], hand_q=t_pre, steps=24, loaded=False)
    yield from sess.close_with_tactile_gate(t_idx, t_plan["primitive"], t_pre, t_thumb, t_closed)

    # 4. Dual-hold window (>= HANDED frames with both hands holding)
    yield from sess.hold_steps(12, loaded={g_arm.arm_key: True, t_arm.arm_key: True})

    # 5. Giver releases fingers-first and retreats clear
    for h_t in pbs.interp_hand(sess.h[g_arm.grip_key], g_pre, 12):
        yield from sess.emit(h_map={g_arm.grip_key: h_t}, loaded={t_arm.arm_key: True})
    for h_t in pbs.interp_hand(g_pre, g_open, 12):
        yield from sess.emit(h_map={g_arm.grip_key: h_t}, loaded={t_arm.arm_key: True})
    g_retreat = g_ho_pt - (SLIDE + 0.05) * g_plan["palm_in"]
    yield from sess.move_arm(g_idx, g_retreat, ho_plan["g_ho_rot"], hand_q=g_open, steps=24, loaded=False)

    # 6. Taker places target onto receptacle with receptacle-adapted waist/lift, releases, and retreats
    place_pt, hover_z, rec_z = _receptacle_target(env, "support", ho_plan["taker_grip_up"])
    waist_place, lift_place = compute_wholebody_grasp_pose(rec_z, place_pt[:2], sess.waist0)
    yield from sess.move_arm(t_idx, np.array([place_pt[0], place_pt[1], hover_z]), t_plan["hand_rot"], loaded=True)
    yield from sess.move_arm(
        t_idx, place_pt, t_plan["hand_rot"], waist=waist_place, lift=lift_place, steps=36, loaded=True
    )
    yield from sess.hold_steps(8, loaded={t_arm.arm_key: True})
    for h_t in pbs.interp_hand(sess.h[t_arm.grip_key], t_pre, 14):
        yield from sess.emit(h_map={t_arm.grip_key: h_t}, loaded={t_arm.arm_key: False})
    for h_t in pbs.interp_hand(t_pre, t_open, 14):
        yield from sess.emit(h_map={t_arm.grip_key: h_t}, loaded={t_arm.arm_key: False})
    t_retreat = place_pt - SLIDE * t_plan["palm_in"]
    yield from sess.move_arm(t_idx, t_retreat, t_plan["hand_rot"], hand_q=t_open, steps=24, loaded=False)
    yield from sess.move_arm(
        t_idx, t_retreat + np.array([0.0, 0.0, HOVER]), t_plan["hand_rot"], hand_q=t_open, steps=24
    )
    yield from sess.hold_steps(35, loaded={t_arm.arm_key: False})


SHARPA_DEMOS = {
    "g1d_pick_anything": sharpa_pick_anything_actions,
    "g1d_place_anything": sharpa_place_anything_actions,
    "g1d_put_anything_in": sharpa_put_anything_in_actions,
    "g1d_handover_anything": sharpa_handover_anything_actions,
}
