"""Motion plausibility of a recorded episode: what its commands ask of the real G1-D and how the simulated arm moved.

collect.py writes summarize() into the sidecar as `motion` and the per-frame Probe data into privileged/; audit.py
recomputes it from the HDF5. Per arm, from the joint positions (rad) at the collection rate, for the commands (action/,
from the first recorded state on) and the response (state/) alike: peak and 99th-percentile joint speed, peak
acceleration and jerk (finite differences) and, with the joint limits, the least distance to them. Per gripper: the
largest command change per action. From privileged/: each arm's least manipulability sqrt(det(J J^T)) of its 6 x n
hand Jacobian (0 at a singular pose) and each gripper's peak finger-pad contact force (N, obs["gripper_force"]).

`violations` names what goes past the real arm's limits that scripted_demo.Limiter enforces: arm commands faster than
REAL_ARM_VMAX or closer than REAL_JOINT_MARGIN to the joint limits, gripper commands faster than REAL_GRIP_RATE.
Acceleration, jerk, manipulability and force have no measured real limit yet and are only reported.

A released dataset without its raw episodes (LeRobot v3, the contract's 16-dim state and action; needs pyarrow):
    python scripts/datagen/motion.py --lerobot <dataset> [--limits <sidecar with joint_limits>] [--out report.json]
prints each worst-per-episode metric's spread over the episodes and how many episodes exceed the real limits, by kind.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.datagen.twitch import (  # noqa: E402
    TWITCH_REV_BURST,
    TWITCH_REV_SPEED,
    TWITCH_STATE_SPEED_MAX,
    TWITCH_TRACK_ERR_MAX,
    TWITCH_WINDOW_S,
    add_twitch_args,
    arm_twitch_metrics,
    reason_kind as _twitch_reason_kind,
    scan_lerobot as scan_lerobot_twitch,
    twitch_reasons,
)
from scripts.g1d.scripted_demo import (  # noqa: E402
    REAL_ARM_VMAX,
    REAL_GRIP_RATE,
    REAL_JOINT_MARGIN,
    Arm,
    as_np,
    link_jacobians,
)
from scripts.g1d.wholebody_demo import (  # noqa: E402
    BODY_KEYS,
    LIFT_URDF_LIMITS,
    REAL_LIFT_MARGIN,
    REAL_LIFT_VMAX,
    REAL_WAIST_VMAX,
    WAIST_URDF_LIMITS,
    body_states_enabled,
)

TOL = 1e-4  # rad (and gripper command) of float32 rounding in the recorded values
MANIPULABILITY, GRIP_FORCE = "motion_manipulability_", "motion_grip_force_"  # privileged/ key prefixes
LEROBOT = {  # the 16-dim vector's parts (CONTRACT.md)
    "left_arm_joint_state": slice(0, 7),
    "left_ee_joint_state": slice(7, 8),
    "right_arm_joint_state": slice(8, 15),
    "right_ee_joint_state": slice(15, 16),
}
LEROBOT_WB19 = {  # the 19-dim wholebody vector's parts (CONTRACT.md g1d_wb19)
    **LEROBOT,
    "waist_joint_state": slice(16, 18),
    "lift_joint_state": slice(18, 19),
}
KINDS = {  # violation text -> kind
    "waist rad/s": "waist_speed",
    "m/s": "lift_speed",
    "rad/s": "arm_speed",
    "joint limit": "limit_margin",
    "per action": "gripper_rate",
}


class Probe:
    """Per-frame privileged motion data while collecting: each arm's manipulability, each gripper's pad forces."""

    def __init__(self, env):
        self.arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        self.body_enabled = body_states_enabled(env)
        self.rows = {MANIPULABILITY + a.arm_key: [] for a in self.arms}

    def __call__(self, obs):
        for a in self.arms:
            jac = as_np(link_jacobians(a.key)[0, a.body - 1][:, a.cols]).reshape(6, -1)
            self.rows[MANIPULABILITY + a.arm_key].append(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))
        for name, force in (obs.get("gripper_force") or {}).items():
            self.rows.setdefault(GRIP_FORCE + name, []).append(as_np(force))

    def limits(self):
        """{arm/body key: [[low, high], ...]} soft joint limits (rad, or m for lift), in the recorded joint order."""
        lim = {a.arm_key: as_np(a.key.data.soft_joint_pos_limits[0, a.cols]).reshape(-1, 2).tolist() for a in self.arms}
        if self.body_enabled:
            lim["waist_joint_state"] = WAIST_URDF_LIMITS.tolist()
            lim["lift_joint_state"] = LIFT_URDF_LIMITS.tolist()
        return lim

    def privileged(self):
        return {key: np.asarray(rows, dtype=np.float32) for key, rows in self.rows.items() if rows}


def peak(x):
    return float(np.abs(x).max()) if np.size(x) else 0.0


def derivatives(q, fps):
    v = np.diff(q, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    return v, a, np.diff(a, axis=0) * fps


def margin(q, limits):
    low, high = np.asarray(limits, dtype=float).T
    return float(np.min(np.minimum(q - low, high - q)))


def summarize(
    states,
    actions,
    fps,
    arms,
    limits=None,
    privileged=None,
    *,
    window_s=TWITCH_WINDOW_S,
    rev_speed=TWITCH_REV_SPEED,
    max_rev_burst=TWITCH_REV_BURST,
    max_state_speed=TWITCH_STATE_SPEED_MAX,
    max_track_err=TWITCH_TRACK_ERR_MAX,
):
    """{arms, grippers, body, manipulability_min, grip_force_max, violations} of an episode. states / actions:
    {key: (T, D)}; arms: the arm keys; BODY_KEYS are checked as body (waist rad/s, lift m/s); the others are grippers;
    limits: {key: (D, 2)} or None (older sidecars have none)."""
    privileged = privileged or {}
    out = dict(arms={}, grippers={}, body={}, manipulability_min={}, grip_force_max={}, violations=[])
    for key in states:
        state, action = np.asarray(states[key], dtype=float), np.asarray(actions[key], dtype=float)
        if key in BODY_KEYS:
            v, a, _ = derivatives(np.vstack([state[:1], action]), fps)
            sv, _, _ = derivatives(state, fps)
            speed = np.abs(v).max(axis=1)
            rmin = np.min(state, axis=0)
            rmax = np.max(state, axis=0)
            m = dict(
                cmd_speed_max=peak(v),
                cmd_speed_p99=float(np.percentile(speed, 99)) if speed.size else 0.0,
                cmd_acc_max=peak(a),
                speed_max=peak(sv),
                range_min=rmin.tolist(),
                range_max=rmax.tolist(),
                range_span=(rmax - rmin).tolist(),
            )
            if key == "waist_joint_state":
                if m["cmd_speed_max"] > REAL_WAIST_VMAX + TOL * fps:
                    out["violations"].append(f"{key}: command {m['cmd_speed_max']:.3f} waist rad/s > {REAL_WAIST_VMAX}")
                if limits and key in limits:
                    m["cmd_limit_margin_min"] = margin(action, limits[key])
                    m["limit_margin_min"] = margin(state, limits[key])
                    if m["cmd_limit_margin_min"] < -TOL:
                        out["violations"].append(f"{key}: command {m['cmd_limit_margin_min']:.3f} rad past joint limit")
            else:  # lift_joint_state (m)
                if m["cmd_speed_max"] > REAL_LIFT_VMAX + TOL * fps:
                    out["violations"].append(f"{key}: command {m['cmd_speed_max']:.4f} m/s > {REAL_LIFT_VMAX}")
                if limits and key in limits:
                    m["cmd_limit_margin_min"] = margin(action, limits[key])
                    m["limit_margin_min"] = margin(state, limits[key])
                    if m["cmd_limit_margin_min"] < -REAL_LIFT_MARGIN - TOL:
                        out["violations"].append(f"{key}: command {m['cmd_limit_margin_min']:.4f} m past joint limit")
            out["body"][key] = m
            continue
        if key not in arms:
            rate = peak(np.diff(action, axis=0))
            out["grippers"][key] = dict(cmd_rate_max=rate)
            if rate > REAL_GRIP_RATE + TOL:
                out["violations"].append(f"{key}: command {rate:.4f} per action > {REAL_GRIP_RATE:.4f}")
            continue
        v, a, j = derivatives(np.vstack([state[:1], action]), fps)
        sv, sa, sj = derivatives(state, fps)
        tm = arm_twitch_metrics(state, action, fps, window_s=window_s, rev_speed=rev_speed)
        speed = np.abs(v).max(axis=1)
        m = dict(
            cmd_speed_max=peak(v),
            cmd_speed_p99=float(np.percentile(speed, 99)) if speed.size else 0.0,
            cmd_acc_max=peak(a),
            cmd_jerk_max=peak(j),
            speed_max=peak(sv),
            acc_max=peak(sa),
            jerk_max=peak(sj),
            rev_burst_max=tm["rev_burst_max"],
            rev_count=tm["rev_count"],
            rev_speed_max=tm["rev_speed_max"],
            osc_wiggle_max=tm["osc_wiggle_max"],
            track_err_max=tm["track_err_max"],
        )
        if m["cmd_speed_max"] > REAL_ARM_VMAX + TOL * fps:
            out["violations"].append(f"{key}: command {m['cmd_speed_max']:.3f} rad/s > {REAL_ARM_VMAX}")
        if limits and key in limits:
            m["cmd_limit_margin_min"], m["limit_margin_min"] = margin(action, limits[key]), margin(state, limits[key])
            if m["cmd_limit_margin_min"] < REAL_JOINT_MARGIN - TOL:
                out["violations"].append(
                    f"{key}: command {m['cmd_limit_margin_min']:.3f} rad from a joint limit < {REAL_JOINT_MARGIN}"
                )
        out["arms"][key] = m
    for name, rows in privileged.items():
        if name.startswith(MANIPULABILITY):
            out["manipulability_min"][name.removeprefix(MANIPULABILITY)] = float(np.min(rows))
        elif name.startswith(GRIP_FORCE):
            out["grip_force_max"][name.removeprefix(GRIP_FORCE)] = peak(rows)
    out["twitch"] = twitch_reasons(
        out,
        max_rev_burst=max_rev_burst,
        max_state_speed=max_state_speed,
        max_track_err=max_track_err,
    )
    return out


def worst(motion):
    """The episode's worst value of each metric over its arms, grippers and body, flat, for audit.py's summary."""
    out = {}
    for group in ("arms", "grippers"):
        for metrics in motion[group].values():
            for name, value in metrics.items():
                low = "margin" in name
                key = f"motion_{name}"
                out[key] = (min if low else max)(out.get(key, value), value)
    body = motion.get("body") or {}
    if "waist_joint_state" in body:
        w = body["waist_joint_state"]
        out["motion_waist_cmd_speed_max"] = w["cmd_speed_max"]
        span = w.get("range_span") or [0.0, 0.0]
        if len(span) >= 2:
            out["motion_waist_pitch_span"] = float(span[1])
    if "lift_joint_state" in body:
        lf = body["lift_joint_state"]
        out["motion_lift_cmd_speed_max"] = lf["cmd_speed_max"]
        span = lf.get("range_span") or [0.0]
        if span:
            out["motion_lift_span"] = float(span[0])
    if motion["manipulability_min"]:
        out["motion_manipulability_min"] = min(motion["manipulability_min"].values())
    if motion["grip_force_max"]:
        out["motion_grip_force_max"] = max(motion["grip_force_max"].values())
    out["motion_violations"] = len(motion["violations"])
    if "twitch" in motion:
        out["motion_twitch"] = len(motion["twitch"])
    return out


def lerobot_episodes(root):
    """(episode index, states, actions) of a LeRobot v3 dataset (16D or 19D g1d_wb19), split into recorder keys."""
    import pyarrow.parquet as pq

    columns = ["episode_index", "frame_index", "observation.state", "action"]
    tables = [pq.read_table(f, columns=columns) for f in sorted(Path(root).glob("data/chunk-*/file-*.parquet"))]
    data = {c: np.concatenate([t[c].to_numpy(zero_copy_only=False) for t in tables]) for c in columns}
    order = np.lexsort((data["frame_index"], data["episode_index"]))
    episodes = data["episode_index"][order]
    starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
    for begin, end in zip(starts, [*starts[1:], len(order)]):
        rows = order[begin:end]
        state, action = np.stack(data["observation.state"][rows]), np.stack(data["action"][rows])
        layout = LEROBOT_WB19 if state.shape[1] == 19 else LEROBOT
        yield int(episodes[begin]), *({k: x[:, s] for k, s in layout.items()} for x in (state, action))


def _violation_kind(text):
    for token, kind in KINDS.items():
        if token in text:
            return kind
    return "other"


def report(
    episodes,
    fps,
    limits=None,
    *,
    window_s=TWITCH_WINDOW_S,
    rev_speed=TWITCH_REV_SPEED,
    max_rev_burst=TWITCH_REV_BURST,
    max_state_speed=TWITCH_STATE_SPEED_MAX,
    max_track_err=TWITCH_TRACK_ERR_MAX,
):
    """Spread of each worst-per-episode metric (p5, p50, p95, min, max), real-limit violations, and twitch counts."""
    arms = [k for k in LEROBOT if "_arm_" in k]
    worsts, over, twitched = [], {}, {}
    for episode, states, actions in episodes:
        moved = summarize(
            states,
            actions,
            fps,
            arms,
            limits,
            window_s=window_s,
            rev_speed=rev_speed,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        )
        worsts.append(worst(moved))
        if moved["violations"]:
            over[episode] = moved["violations"]
        if moved["twitch"]:
            twitched[episode] = moved["twitch"]
    spread = {}
    for name in sorted({k for w in worsts for k in w}):
        values = np.percentile([w[name] for w in worsts if name in w], [5, 50, 95, 0, 100])
        spread[name] = dict(zip(("p5", "p50", "p95", "min", "max"), np.round(values, 5).tolist()))
    kinds = Counter(kind for violations in over.values() for kind in {_violation_kind(v) for v in violations})
    twitch_kinds = Counter(kind for reasons in twitched.values() for kind in {_twitch_reason_kind(r) for r in reasons})
    return dict(
        episodes=len(worsts),
        fps=fps,
        metrics=spread,
        over_real_limits=dict(episodes=len(over), **kinds),
        twitch=dict(episodes=len(twitched), **twitch_kinds),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot", type=Path, required=True, help="LeRobot v3 dataset folder")
    parser.add_argument("--limits", type=Path, help="a collection sidecar whose joint_limits to check against")
    parser.add_argument("--out", type=Path, help="write the report here as well")
    parser.add_argument("--twitch-out", type=Path, help="write per-episode twitch exclusion report JSON here")
    add_twitch_args(parser)
    args = parser.parse_args()
    fps = json.loads((args.lerobot / "meta/info.json").read_text())["fps"]
    limits = json.loads(args.limits.read_text())["joint_limits"] if args.limits else None
    result = report(
        lerobot_episodes(args.lerobot),
        fps,
        limits,
        window_s=args.window_s,
        rev_speed=args.rev_speed,
        max_rev_burst=args.max_rev_burst,
        max_state_speed=args.max_state_speed,
        max_track_err=args.max_track_err,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    if args.twitch_out:
        twitch_report = scan_lerobot_twitch(
            args.lerobot,
            window_s=args.window_s,
            rev_speed=args.rev_speed,
            max_rev_burst=args.max_rev_burst,
            max_state_speed=args.max_state_speed,
            max_track_err=args.max_track_err,
        )
        args.twitch_out.parent.mkdir(parents=True, exist_ok=True)
        args.twitch_out.write_text(json.dumps(twitch_report, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
