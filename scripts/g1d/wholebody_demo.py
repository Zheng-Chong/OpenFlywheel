"""19D fixed-base wholebody (waist + lift + dual-arm gripper, `g1d_wb19`) IK, posture planning, and limiter foundation.

Kept separate from `scripts/g1d/scripted_demo.py` so all registered 16D experts in `scripts/datagen/experts.json` keep
zero source hash drift.
"""

import numpy as np

from scripts.g1d import scripted_demo as demo

# Wholebody (waist + lift, fixed base) motion limits: conservative waist speed pending real-robot sysid; lift matches
# AgvClient.HeightAdjust (76.5 mm/s across the 0..0.42 m two-stage stroke); safe pitch keeps cam_head (47.6 deg down)
# seeing the table ahead without colliding torso/shoulders with the table edge.
BODY_KEYS = ("waist_joint_state", "lift_joint_state")
REAL_WAIST_VMAX = 0.5  # rad/s (WaistYaw, WaistPitch)
REAL_LIFT_VMAX = 0.0765  # m/s (2.55 mm per 30 Hz action)
REAL_WAIST_PITCH_MAX = float(np.deg2rad(35.0))  # rad, default safe active-vision / table-clearance bend limit
REAL_LIFT_MARGIN = 0.005  # m inside [0, 0.42] when soft-limited
WAIST_URDF_LIMITS = np.array([[-2.7053, 2.7053], [-0.04363, 2.3562]])  # [yaw, pitch] rad
LIFT_URDF_LIMITS = np.array([[0.0, 0.42]])  # [height] m


def body_robot(env):
    """The target robot entry carrying body_dims (waist + lift), or None if absent."""
    rm = getattr(env, "robot_manager", None)
    if rm is None:
        return None
    for r in rm.robot_list:
        if r.type == "target" and getattr(r, "body_dims", None):
            return r
    return None


def body_states_enabled(env):
    """Whether the env config enables body_states observation (19D g1d_wb19 fixed-base wholebody)."""
    om = getattr(env, "obs_manager", None)
    rcfg = getattr(om, "robot_cfg", None)
    return bool(rcfg and rcfg.get("body_states", False))


def current_body_state(env):
    """Current {waist_joint_state, lift_joint_state} of env 0 from robot_manager.get_body_state."""
    rb = body_robot(env)
    if rb is None:
        return {}
    raw = env.robot_manager.get_body_state(robot=rb, env_idx_list=[0])[0]
    return {k: list(raw[k]) for k in BODY_KEYS if k in raw}


def table_top_z_from_env(env, fallback_z=0.75):
    """Read the layout's table top height (m) from saved_layouts[0]['Room'], falling back to `fallback_z`."""
    lm = getattr(getattr(env, "scene_manager", None), "layout_manager", None)
    saved = (getattr(lm, "saved_layouts", None) or [None])[0] or {}
    room = saved.get("Room") or {}
    if room.get("table_height") is not None:
        return float(room["table_height"])
    choice = room.get("appearance_choice") or {}
    if choice.get("table_top_z") is not None:
        return float(choice["table_top_z"])
    return float(fallback_z)


def plan_body_posture(
    target_y,
    target_z,
    table_top_z,
    *,
    base_y=-0.73,
    max_pitch=REAL_WAIST_PITCH_MAX,
    lift_ref_offset=0.68,
    comfort_reach_y=0.45,
    torso_len=0.45,
    table_clear=0.05,
):
    """Macro wholebody posture planner for fixed-base G1-D (waist_yaw locked at 0):
    1. Prefer lift_joint_state [0..0.42 m] to match table_top_z (keeps cam_head 47.6 deg forward tilt unchanged).
    2. Allocate waist_pitch only when lift bottoms out on low targets or target depth exceeds comfortable arm reach,
       clamped to [0, max_pitch] and checked for torso-vs-table vertical clearance."""
    lift = float(np.clip(table_top_z - lift_ref_offset, LIFT_URDF_LIMITS[0, 0], LIFT_URDF_LIMITS[0, 1]))
    # Residual depth beyond comfortable arm reach and residual vertical reach below comfortable desk height
    depth_excess = max(0.0, (target_y - base_y) - comfort_reach_y)
    z_comfort = lift_ref_offset + lift - 0.05
    z_deficit = max(0.0, z_comfort - target_z)
    pitch_for_depth = float(np.arcsin(np.clip(depth_excess / torso_len, 0.0, 0.85))) if depth_excess > 0 else 0.0
    pitch_for_z = float(np.arccos(np.clip(1.0 - z_deficit / torso_len, 0.25, 1.0))) if z_deficit > 0 else 0.0
    pitch = min(float(max_pitch), max(pitch_for_depth, pitch_for_z))
    # Table-edge clearance: shoulder/torso height = (0.5355 + lift) + torso_len * cos(pitch)
    hip_z = 0.5355 + lift
    min_shoulder_z = table_top_z + table_clear
    if hip_z + torso_len * np.cos(pitch) < min_shoulder_z:
        cos_need = np.clip((min_shoulder_z - hip_z) / torso_len, -1.0, 1.0)
        pitch = min(pitch, float(np.arccos(cos_need)))
    pitch = float(np.clip(pitch, 0.0, min(max_pitch, WAIST_URDF_LIMITS[1, 1])))
    return {"waist_joint_state": [0.0, pitch], "lift_joint_state": [lift]}


def wholebody_wdls_step(
    jac_arm,
    jac_waist_pitch,
    jac_lift,
    err,
    q_arm,
    q_pitch,
    q_lift,
    lim_arm,
    lim_pitch=(0.0, REAL_WAIST_PITCH_MAX),
    lim_lift=(0.0, 0.42),
    *,
    w_pitch=0.15,
    w_lift=0.25,
    rest_arm=None,
    rest_pitch=0.0,
    rest_gain=0.02,
    fps=30.0,
):
    """Weighted Damped Least-Squares wholebody step for (waist_pitch, lift, 7-DOF arm).

    Weighting w_arm=1.0 > w_lift > w_pitch makes the arm do fast fine motion while waist_pitch and lift only contribute
    when the arm approaches its reach or joint limits. Returns (q_arm_next, q_pitch_next, q_lift_next) clipped to
    joint limits and real-robot per-step speed caps."""
    q_arm = np.asarray(q_arm, dtype=float)
    lim_arm = np.asarray(lim_arm, dtype=float)
    jac_arm = np.asarray(jac_arm, dtype=float)
    jac_p = np.asarray(jac_waist_pitch, dtype=float).reshape(len(err), 1)
    jac_l = np.asarray(jac_lift, dtype=float).reshape(len(err), 1)
    err = np.asarray(err, dtype=float)

    q_all = np.concatenate([[float(q_pitch), float(q_lift)], q_arm])
    lim_all = np.vstack([np.array([lim_pitch, lim_lift], dtype=float), lim_arm])
    jac_all = np.hstack([jac_p, jac_l, jac_arm])

    base_w = np.concatenate([[w_pitch, w_lift], np.ones(len(q_arm))])
    room = np.minimum(q_all - lim_all[:, 0], lim_all[:, 1] - q_all)
    zone = np.concatenate([[demo.PLUS_LIMIT_ZONE, 0.03], np.full(len(q_arm), demo.PLUS_LIMIT_ZONE)])
    w = base_w * np.clip(room / zone, demo.PLUS_LIMIT_FLOOR, 1.0)
    jw = jac_all * w
    low = np.linalg.svd(jw, compute_uv=False)[-1]
    damp = np.hypot(0.02, demo.PLUS_DAMP_MAX * np.sqrt(max(0.0, 1.0 - (low / demo.PLUS_SINGULAR) ** 2)))
    pinv_w = jw.T @ np.linalg.solve(jw @ jw.T + damp**2 * np.eye(len(err)), np.eye(len(err)))
    dq = w * (pinv_w @ err)
    if rest_arm is not None:
        q_rest = np.concatenate([[float(rest_pitch), float(q_lift)], np.asarray(rest_arm, dtype=float)])
        # Nullspace posture bias toward rest_arm / upright waist (lift left neutral)
        null_proj = np.eye(len(q_all)) - (w[:, None] * pinv_w) @ jac_all
        bias = rest_gain * (q_rest - q_all)
        bias[1] = 0.0
        dq = dq + null_proj @ bias

    step_cap = np.concatenate(
        [[REAL_WAIST_VMAX / fps, REAL_LIFT_VMAX / fps], np.full(len(q_arm), demo.REAL_ARM_VMAX / fps)]
    )
    dq = np.clip(dq, -step_cap, step_cap)
    q_next = np.clip(q_all + dq, lim_all[:, 0], lim_all[:, 1])
    return q_next[2:], float(q_next[0]), float(q_next[1])


def move_body_hold(arms, body, dst_waist=None, dst_lift=None, q_arms=None, g_arms=None, steps=None, fps=30.0):
    """Yield smootherstep commands interpolating `body` (`waist_joint_state` / `lift_joint_state`) at real-robot speed
    limits while holding `arms` at `q_arms` and grippers at `g_arms`. Mutates `body` in place."""
    if not body:
        return
    q_arms = {a.arm_key: a.q() for a in arms} if q_arms is None else q_arms
    g_arms = {a.grip_key: [1.0] for a in arms} if g_arms is None else g_arms
    cur_w = np.asarray(body.get("waist_joint_state", [0.0, 0.0]), dtype=float)
    cur_l = np.asarray(body.get("lift_joint_state", [0.035]), dtype=float)
    tgt_w = (
        np.clip(np.asarray(dst_waist, dtype=float), WAIST_URDF_LIMITS[:, 0], WAIST_URDF_LIMITS[:, 1])
        if dst_waist is not None
        else cur_w.copy()
    )
    tgt_l = (
        np.clip(
            np.asarray([dst_lift] if np.ndim(dst_lift) == 0 else dst_lift, dtype=float),
            LIFT_URDF_LIMITS[:, 0],
            LIFT_URDF_LIMITS[:, 1],
        )
        if dst_lift is not None
        else cur_l.copy()
    )
    if steps is None:
        dw = float(np.abs(tgt_w - cur_w).max()) / (REAL_WAIST_VMAX / fps)
        dl = float(np.abs(tgt_l - cur_l).max()) / (REAL_LIFT_VMAX / fps)
        if max(dw, dl) < 1e-4:
            return
        # Peak slope of smootherstep 6t^5 - 15t^4 + 10t^3 is 1.875 at t=0.5
        steps = max(1, int(np.ceil(1.875 * max(dw, dl))))
    for k in range(1, steps + 1):
        t = k / steps
        s = t * t * t * (t * (6 * t - 15) + 10)
        if "waist_joint_state" in body:
            body["waist_joint_state"] = (cur_w + s * (tgt_w - cur_w)).tolist()
        if "lift_joint_state" in body:
            body["lift_joint_state"] = (cur_l + s * (tgt_l - cur_l)).tolist()
        yield dict(q_arms) | dict(g_arms) | dict(body)


def adapt_body_to_scene(
    env,
    target_y,
    target_z,
    table_top_z,
    *,
    max_pitch=float(np.deg2rad(22.0)),
    comfort_reach_y=0.32,
    settle_steps=5,
    fps=30.0,
):
    """Pre-grasp wholebody posture adaptation: smoothly drive lift & waist_pitch to match `table_top_z` and `target_y`
    before constructing GraspRobot / PlaceRobot (so Reach() and home poses see the post-adaptation shoulder frame)."""
    if not body_states_enabled(env):
        return {}
    body = current_body_state(env)
    if not body:
        return {}
    arms = [demo.Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
    q_init = {a.arm_key: a.q() for a in arms}
    g_open = {a.grip_key: [1.0] for a in arms}
    posture = plan_body_posture(
        target_y=target_y,
        target_z=target_z,
        table_top_z=table_top_z,
        max_pitch=max_pitch,
        comfort_reach_y=comfort_reach_y,
    )
    print(
        f"[wholebody_demo] adapt: table_z={table_top_z:.3f} target_y={target_y:.3f} init={body} -> target={posture}",
        flush=True,
    )
    yield from move_body_hold(
        arms, body, posture["waist_joint_state"], posture["lift_joint_state"], q_init, g_open, fps=fps
    )
    for _ in range(settle_steps):
        yield dict(q_init) | dict(g_open) | dict(body)
    return body


class WholebodyMixin:
    """Mixin for `scripted_demo.Robot` subclasses: preserves fixed-base body_states keys in every yielded command
    (`_command()` and `finish()`) and provides `move_body()`."""

    def __init__(self, env, *args, body=None, **kwargs):
        super().__init__(env, *args, **kwargs)
        if body is not None:
            self.body = {k: list(v) for k, v in body.items()}
        else:
            self.body = current_body_state(env) if body_states_enabled(env) else {}
        self.body_home = {k: list(v) for k, v in self.body.items()}

    def _command(self, arm, point, rot, grip):
        cmd = super()._command(arm, point, rot, grip)
        if self.body:
            return dict(cmd) | self.body
        return cmd

    def finish(self):
        for cmd in super().finish():
            yield (dict(cmd) | self.body) if self.body else cmd

    def move_body(self, waist=None, lift=None, steps=None, fps=30.0):
        """Smoothly interpolate waist_joint_state and/or lift_joint_state while holding current arm & gripper poses."""
        if not self.body:
            return
        for cmd in move_body_hold(self.arms, self.body, waist, lift, self.hold, self.grip, steps=steps, fps=fps):
            self.step += 1
            yield cmd


class WholebodyLimiter(demo.Limiter):
    """Real-robot motion limits for 19D fixed-base wholebody: extends `scripted_demo.Limiter` with waist_joint_state
    (`REAL_WAIST_VMAX`) and lift_joint_state (`REAL_LIFT_VMAX`) clamping and rate limiting."""

    def __init__(self, env):
        super().__init__(env)
        self.body = {}
        self.body_dq_max = {
            "waist_joint_state": REAL_WAIST_VMAX / 30,
            "lift_joint_state": REAL_LIFT_VMAX / 30,
        }
        if body_robot(env) is not None:
            self.lim["waist_joint_state"] = WAIST_URDF_LIMITS + np.array(
                [[demo.REAL_JOINT_MARGIN, -demo.REAL_JOINT_MARGIN], [0.0, 0.0]]
            )
            self.lim["lift_joint_state"] = LIFT_URDF_LIMITS.copy()
            if body_states_enabled(env):
                raw_body = current_body_state(env)
                self.body = {k: np.asarray(raw_body[k], dtype=float) for k in BODY_KEYS if k in raw_body}
        self.asked_waist = self.asked_lift = 0.0

    def __call__(self, action):
        out = super().__call__(action)
        for key in BODY_KEYS:
            if key in out:
                lim = self.lim.get(key, WAIST_URDF_LIMITS if key == "waist_joint_state" else LIFT_URDF_LIMITS)
                target = np.clip(np.asarray(out[key], dtype=float).reshape(-1), *lim.T)
                prev = self.body.get(key, target.copy())
                step = target - prev
                dq = self.body_dq_max[key]
                if key == "waist_joint_state":
                    self.asked_waist = max(self.asked_waist, float(np.abs(step).max()))
                else:
                    self.asked_lift = max(self.asked_lift, float(np.abs(step).max()))
                self.body[key] = prev + np.clip(step, -dq, dq)
                out[key] = self.body[key].tolist()
        return out
