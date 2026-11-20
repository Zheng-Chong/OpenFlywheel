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
from scripts.g1d.scripted_demo import (  # noqa: E402
    REAL_ARM_VMAX,
    REAL_GRIP_RATE,
    REAL_JOINT_MARGIN,
    Arm,
    as_np,
    link_jacobians,
)

TOL = 1e-4  # rad (and gripper command) of float32 rounding in the recorded values
MANIPULABILITY, GRIP_FORCE = "motion_manipulability_", "motion_grip_force_"  # privileged/ key prefixes
LEROBOT = {  # the 16-dim vector's parts (CONTRACT.md)
    "left_arm_joint_state": slice(0, 7),
    "left_ee_joint_state": slice(7, 8),
    "right_arm_joint_state": slice(8, 15),
    "right_ee_joint_state": slice(15, 16),
}
KINDS = {"rad/s": "arm_speed", "joint limit": "limit_margin", "per action": "gripper_rate"}  # violation text -> kind


class Probe:
    """Per-frame privileged motion data while collecting: each arm's manipulability, each gripper's pad forces."""

    def __init__(self, env):
        self.arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        self.rows = {MANIPULABILITY + a.arm_key: [] for a in self.arms}

    def __call__(self, obs):
        for a in self.arms:
            jac = as_np(link_jacobians(a.key)[0, a.body - 1][:, a.cols]).reshape(6, -1)
            self.rows[MANIPULABILITY + a.arm_key].append(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))
        for name, force in (obs.get("gripper_force") or {}).items():
            self.rows.setdefault(GRIP_FORCE + name, []).append(as_np(force))

    def limits(self):
        """{arm key: [[low, high], ...]} soft joint limits (rad), in the recorded joint order."""
        return {
            a.arm_key: as_np(a.key.data.soft_joint_pos_limits[0, a.cols]).reshape(-1, 2).tolist() for a in self.arms
        }

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


def summarize(states, actions, fps, arms, limits=None, privileged=None):
    """{arms, grippers, manipulability_min, grip_force_max, violations} of an episode. states / actions: {key: (T, D)};
    arms: the arm keys (the others are grippers); limits: {arm key: (D, 2)} or None (older sidecars have none)."""
    privileged = privileged or {}
    out = dict(arms={}, grippers={}, manipulability_min={}, grip_force_max={}, violations=[])
    for key in states:
        state, action = np.asarray(states[key], dtype=float), np.asarray(actions[key], dtype=float)
        if key not in arms:
            rate = peak(np.diff(action, axis=0))
            out["grippers"][key] = dict(cmd_rate_max=rate)
            if rate > REAL_GRIP_RATE + TOL:
                out["violations"].append(f"{key}: command {rate:.4f} per action > {REAL_GRIP_RATE:.4f}")
            continue
        v, a, j = derivatives(np.vstack([state[:1], action]), fps)
        sv, sa, sj = derivatives(state, fps)
        speed = np.abs(v).max(axis=1)
        m = dict(
            cmd_speed_max=peak(v),
            cmd_speed_p99=float(np.percentile(speed, 99)) if speed.size else 0.0,
            cmd_acc_max=peak(a),
            cmd_jerk_max=peak(j),
            speed_max=peak(sv),
            acc_max=peak(sa),
            jerk_max=peak(sj),
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
    return out


def worst(motion):
    """The episode's worst value of each metric over its arms and grippers, flat, for audit.py's dataset summary."""
    out = {}
    for group in ("arms", "grippers"):
        for metrics in motion[group].values():
            for name, value in metrics.items():
                low = "margin" in name
                key = f"motion_{name}"
                out[key] = (min if low else max)(out.get(key, value), value)
    if motion["manipulability_min"]:
        out["motion_manipulability_min"] = min(motion["manipulability_min"].values())
    if motion["grip_force_max"]:
        out["motion_grip_force_max"] = max(motion["grip_force_max"].values())
    out["motion_violations"] = len(motion["violations"])
    return out


def lerobot_episodes(root):
    """(episode index, states, actions) of a LeRobot v3 dataset, split into the recorder's keys."""
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
        yield int(episodes[begin]), *({k: x[:, s] for k, s in LEROBOT.items()} for x in (state, action))


def report(episodes, fps, limits=None):
    """Spread of each worst-per-episode metric (p5, p50, p95, min, max) and the episodes over the real limits."""
    arms = [k for k in LEROBOT if "_arm_" in k]
    worsts, over = [], {}
    for episode, states, actions in episodes:
        moved = summarize(states, actions, fps, arms, limits)
        worsts.append(worst(moved))
        if moved["violations"]:
            over[episode] = moved["violations"]
    spread = {}
    for name in sorted({k for w in worsts for k in w}):
        values = np.percentile([w[name] for w in worsts if name in w], [5, 50, 95, 0, 100])
        spread[name] = dict(zip(("p5", "p50", "p95", "min", "max"), np.round(values, 5).tolist()))
    kinds = Counter(
        kind for violations in over.values() for kind in {KINDS[t] for v in violations for t in KINDS if t in v}
    )
    return dict(episodes=len(worsts), fps=fps, metrics=spread, over_real_limits=dict(episodes=len(over), **kinds))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot", type=Path, required=True, help="LeRobot v3 dataset folder")
    parser.add_argument("--limits", type=Path, help="a collection sidecar whose joint_limits to check against")
    parser.add_argument("--out", type=Path, help="write the report here as well")
    args = parser.parse_args()
    fps = json.loads((args.lerobot / "meta/info.json").read_text())["fps"]
    limits = json.loads(args.limits.read_text())["joint_limits"] if args.limits else None
    result = report(lerobot_episodes(args.lerobot), fps, limits)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
