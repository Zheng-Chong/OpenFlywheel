"""Detect high-frequency arm twitching (oscillation chatter) and physics-bounce spikes in G1-D trajectories.

Empirical diagnosis across 74,729 episodes (22 X-Anything datasets) showed two distinct physical twitch modes:
1. Closed-loop IK / Limiter bang-bang chatter: an arm joint alternates +/-REAL_ARM_VMAX (+/-0.8 rad/s) frame-to-frame
   at ~15 Hz, producing repeated velocity sign reversals (>= 4 reversals within 0.5 s at > 0.30 rad/s).
2. Physics contact-bounce / arm-snag flailing: simulated arm state diverges from commanded action (track_err > 0.25 rad,
   vs ~0.005 rad normal 1-step PD lag) or snaps at > 1.8 rad/s (> 2.25x REAL_ARM_VMAX).

Standalone scan of an exported LeRobot v3 dataset (requires only numpy and pyarrow):
    python scripts/datagen/twitch.py --lerobot /path/to/dataset --out excluded.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

TWITCH_WINDOW_S = 0.5  # sliding window duration (s); 15 frames at 30 Hz
TWITCH_REV_SPEED = 0.30  # rad/s: min(|v[t-1]|, |v[t]|) to count a velocity sign reversal (15 Hz chatter is ~0.55)
TWITCH_REV_BURST = 4  # reversals on one arm joint within TWITCH_WINDOW_S (>= 2 full oscillation cycles)
TWITCH_STATE_SPEED_MAX = 1.8  # rad/s: > 2.25x REAL_ARM_VMAX (0.8); catches physics contact-bounce / snag snap-back
TWITCH_TRACK_ERR_MAX = 0.25  # rad (~14.3 deg): ~50x normal 0.005 rad 1-step PD lag; catches arm snag / collision

LEROBOT_ARM_SLICES = {
    "left_arm_joint_state": slice(0, 7),
    "right_arm_joint_state": slice(8, 15),
}

REASON_KINDS = (
    ("rev_burst_max", "rev_burst"),
    ("state_speed_max", "state_speed"),
    ("track_err_max", "track_err"),
)


def arm_twitch_metrics(
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    *,
    window_s: float = TWITCH_WINDOW_S,
    rev_speed: float = TWITCH_REV_SPEED,
) -> dict[str, int | float]:
    """Compute twitch/oscillation metrics for one arm's (T, D) joint state and action trajectories."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")

    s = np.asarray(state, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if a.ndim == 1:
        a = a[:, None]
    if s.shape != a.shape or s.ndim != 2:
        raise ValueError(f"state and action must have identical 2D shape, got {s.shape} vs {a.shape}")
    if not (np.isfinite(s).all() and np.isfinite(a).all()):
        raise ValueError("state and action must be finite")

    t_len, dims = s.shape
    if t_len < 2 or dims == 0:
        return {
            "speed_max": 0.0,
            "rev_burst_max": 0,
            "rev_count": 0,
            "rev_speed_max": 0.0,
            "osc_wiggle_max": 0.0,
            "track_err_max": 0.0,
        }

    dq = np.diff(s, axis=0)  # (T-1, D)
    sv = dq * float(fps)
    speed_max = float(np.abs(sv).max())
    # action[t] is commanded after observing state[t] and drives the transition to state[t+1].
    track_err_max = float(np.abs(a[:-1] - s[1:]).max())

    if t_len < 3:
        return {
            "speed_max": speed_max,
            "rev_burst_max": 0,
            "rev_count": 0,
            "rev_speed_max": 0.0,
            "osc_wiggle_max": 0.0,
            "track_err_max": track_err_max,
        }

    rev_mask = (sv[:-1] * sv[1:]) < 0.0
    rev_amp = np.where(rev_mask, np.minimum(np.abs(sv[:-1]), np.abs(sv[1:])), 0.0)
    rev_speed_max = float(rev_amp.max())
    sig_rev = rev_amp > float(rev_speed)
    rev_count = int(sig_rev.any(axis=1).sum())

    win = max(2, round(float(window_s) * float(fps)))
    if sig_rev.shape[0] >= win - 1:
        rcsum = np.vstack([np.zeros((1, dims), dtype=np.int32), np.cumsum(sig_rev.astype(np.int32), axis=0)])
        burst_win = rcsum[win - 1 :] - rcsum[: -(win - 1)]  # (T - win, D)
        rev_burst_max = int(burst_win.max())
        csum = np.vstack([np.zeros((1, dims), dtype=np.float64), np.cumsum(np.abs(dq), axis=0)])
        wig_win = (csum[win:] - csum[:-win]) - np.abs(s[win:] - s[:-win])
        osc_wiggle_max = float(np.where(burst_win >= 3, wig_win, 0.0).max())
    else:
        burst_all = sig_rev.sum(axis=0)
        rev_burst_max = int(burst_all.max())
        wig_all = np.abs(dq).sum(axis=0) - np.abs(s[-1] - s[0])
        osc_wiggle_max = float(np.where(burst_all >= 3, wig_all, 0.0).max())

    return {
        "speed_max": speed_max,
        "rev_burst_max": rev_burst_max,
        "rev_count": rev_count,
        "rev_speed_max": rev_speed_max,
        "osc_wiggle_max": osc_wiggle_max,
        "track_err_max": track_err_max,
    }


def twitch_reasons(
    motion_or_arms: dict,
    *,
    max_rev_burst: int | None = TWITCH_REV_BURST,
    max_state_speed: float | None = TWITCH_STATE_SPEED_MAX,
    max_track_err: float | None = TWITCH_TRACK_ERR_MAX,
) -> list[str]:
    """Return human-readable twitch reasons for a `motion.summarize()` dict or `{arm_key: metrics}` mapping."""
    arms = motion_or_arms.get("arms", motion_or_arms) if isinstance(motion_or_arms, dict) else {}
    reasons: list[str] = []
    for key in sorted(arms):
        m = arms[key]
        if not isinstance(m, dict):
            continue
        if max_rev_burst is not None and "rev_burst_max" in m and m["rev_burst_max"] >= max_rev_burst:
            reasons.append(f"{key}: rev_burst_max {int(m['rev_burst_max'])} >= {max_rev_burst}")
        if max_state_speed is not None and "speed_max" in m and m["speed_max"] > max_state_speed:
            reasons.append(f"{key}: state_speed_max {float(m['speed_max']):.3f} rad/s > {max_state_speed}")
        if max_track_err is not None and "track_err_max" in m and m["track_err_max"] > max_track_err:
            reasons.append(f"{key}: track_err_max {float(m['track_err_max']):.3f} rad > {max_track_err}")
    return reasons


def has_twitch_metrics(motion_dict: dict | None) -> bool:
    """True if a sidecar's `motion` dict already contains arm twitch metrics."""
    if not isinstance(motion_dict, dict):
        return False
    arms = motion_dict.get("arms")
    if not isinstance(arms, dict) or not arms:
        return False
    return all(
        isinstance(m, dict) and "rev_burst_max" in m and "speed_max" in m and "track_err_max" in m
        for m in arms.values()
    )


def reason_kind(text: str) -> str:
    for token, kind in REASON_KINDS:
        if token in text:
            return kind
    return "other"


def check_hdf5_episode(
    path: Path,
    quality: dict,
    *,
    window_s: float = TWITCH_WINDOW_S,
    rev_speed: float = TWITCH_REV_SPEED,
    max_rev_burst: int | None = TWITCH_REV_BURST,
    max_state_speed: float | None = TWITCH_STATE_SPEED_MAX,
    max_track_err: float | None = TWITCH_TRACK_ERR_MAX,
) -> tuple[dict[str, dict[str, int | float]], list[str]]:
    """Compute arm twitch metrics and reasons directly from a raw trajectory HDF5 file."""
    import h5py

    fps = float(quality.get("fps", 30))
    joint_names = quality.get("joint_names") or {}
    state_keys = quality.get("state_keys") or list(joint_names) or list(LEROBOT_ARM_SLICES)
    arm_keys = [k for k in state_keys if k in joint_names or "_arm_" in k]
    arms: dict[str, dict[str, int | float]] = {}
    with h5py.File(path, "r") as f:
        for key in arm_keys:
            s_ds = f.get(f"state/{key}s")
            a_ds = f.get(f"action/{key}s")
            if s_ds is None or a_ds is None:
                continue
            arms[key] = arm_twitch_metrics(s_ds[()], a_ds[()], fps, window_s=window_s, rev_speed=rev_speed)
    return arms, twitch_reasons(
        arms,
        max_rev_burst=max_rev_burst,
        max_state_speed=max_state_speed,
        max_track_err=max_track_err,
    )


def scan_lerobot(
    root: Path,
    *,
    window_s: float = TWITCH_WINDOW_S,
    rev_speed: float = TWITCH_REV_SPEED,
    max_rev_burst: int | None = TWITCH_REV_BURST,
    max_state_speed: float | None = TWITCH_STATE_SPEED_MAX,
    max_track_err: float | None = TWITCH_TRACK_ERR_MAX,
    top_k: int = 20,
) -> dict:
    """Scan an exported LeRobot v3 dataset for twitching episodes and return a structured exclusion report."""
    import pyarrow.parquet as pq

    root = Path(root)
    info_path = root / "meta" / "info.json"
    fps = float(json.loads(info_path.read_text()).get("fps", 30)) if info_path.is_file() else 30.0
    pq_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    cols = ["episode_index", "frame_index", "observation.state", "action"]
    tables = [pq.read_table(f, columns=cols) for f in pq_files]
    ep_idx_all = np.concatenate([t["episode_index"].to_numpy(zero_copy_only=False) for t in tables])
    fr_idx_all = np.concatenate([t["frame_index"].to_numpy(zero_copy_only=False) for t in tables])
    st_all = np.concatenate([t["observation.state"].to_numpy(zero_copy_only=False) for t in tables])
    ac_all = np.concatenate([t["action"].to_numpy(zero_copy_only=False) for t in tables])
    order = np.lexsort((fr_idx_all, ep_idx_all))
    if len(order) == 0:
        ep_sorted = np.empty(0, dtype=np.int64)
        starts = np.empty(0, dtype=np.int64)
        ends = np.empty(0, dtype=np.int64)
    else:
        ep_sorted = ep_idx_all[order]
        starts = np.flatnonzero(np.r_[True, ep_sorted[1:] != ep_sorted[:-1]])
        ends = np.r_[starts[1:], len(order)]

    flagged: list[dict] = []
    by_reason: Counter[str] = Counter()
    for begin, end in zip(starts, ends):
        rows = order[begin:end]
        ep = int(ep_sorted[begin])
        state = np.stack(st_all[rows])
        action = np.stack(ac_all[rows])
        if state.shape[1] < 15:
            raise ValueError(f"Expected >= 15 state dims for dual-arm G1-D, got {state.shape[1]} in {root}")
        arms = {
            key: arm_twitch_metrics(state[:, sl], action[:, sl], fps, window_s=window_s, rev_speed=rev_speed)
            for key, sl in LEROBOT_ARM_SLICES.items()
        }
        reasons = twitch_reasons(
            arms,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        )
        if reasons:
            for kind in {reason_kind(r) for r in reasons}:
                by_reason[kind] += 1
            flagged.append(
                {
                    "episode_index": ep,
                    "frames": int(state.shape[0]),
                    "reasons": reasons,
                    "rev_burst_max": max(m["rev_burst_max"] for m in arms.values()),
                    "speed_max": round(max(m["speed_max"] for m in arms.values()), 4),
                    "track_err_max": round(max(m["track_err_max"] for m in arms.values()), 4),
                    "osc_wiggle_max": round(max(m["osc_wiggle_max"] for m in arms.values()), 4),
                }
            )

    total = len(starts)
    excluded = [item["episode_index"] for item in flagged]
    top_offenders = sorted(
        flagged,
        key=lambda x: (x["rev_burst_max"], x["speed_max"], x["track_err_max"]),
        reverse=True,
    )[:top_k]

    return {
        "dataset": root.name,
        "episodes": total,
        "fps": fps,
        "thresholds": {
            "window_s": window_s,
            "rev_speed": rev_speed,
            "max_rev_burst": max_rev_burst,
            "max_state_speed": max_state_speed,
            "max_track_err": max_track_err,
        },
        "twitch_episodes": len(flagged),
        "clean_episodes": total - len(flagged),
        "twitch_rate": round(len(flagged) / total, 5) if total else 0.0,
        "by_reason": dict(sorted(by_reason.items())),
        "excluded_episode_indices": excluded,
        "top_offenders": top_offenders,
        "flagged": flagged,
    }


def load_excluded_indices(path: Path) -> set[int]:
    """Load excluded episode indices from a twitch scan report JSON or a plain JSON integer list."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return {int(x) for x in raw}
    if isinstance(raw, dict):
        if isinstance(raw.get("excluded_episode_indices"), list):
            return {int(x) for x in raw["excluded_episode_indices"]}
        if isinstance(raw.get("excluded"), list):
            return {int(x) for x in raw["excluded"]}
    raise ValueError(f"Unrecognized exclusion list format in {path}")


def add_twitch_args(parser: argparse.ArgumentParser) -> None:
    """Register standard twitch threshold CLI flags on an ArgumentParser."""
    parser.add_argument(
        "--window-s",
        type=float,
        default=TWITCH_WINDOW_S,
        help=f"sliding window duration in seconds for reversal bursts (default {TWITCH_WINDOW_S})",
    )
    parser.add_argument(
        "--rev-speed",
        type=float,
        default=TWITCH_REV_SPEED,
        help=f"min velocity amplitude (rad/s) on both sides of a sign reversal (default {TWITCH_REV_SPEED})",
    )
    parser.add_argument(
        "--max-rev-burst",
        type=int,
        default=TWITCH_REV_BURST,
        help=f"flag episode when any arm joint has >= N reversals in window (default {TWITCH_REV_BURST})",
    )
    parser.add_argument(
        "--max-state-speed",
        type=float,
        default=TWITCH_STATE_SPEED_MAX,
        help=f"flag episode when arm joint speed exceeds this rad/s (default {TWITCH_STATE_SPEED_MAX})",
    )
    parser.add_argument(
        "--max-track-err",
        type=float,
        default=TWITCH_TRACK_ERR_MAX,
        help=f"flag episode when 1-step |action - state| exceeds this rad (default {TWITCH_TRACK_ERR_MAX})",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot", type=Path, required=True, help="LeRobot v3 dataset folder")
    parser.add_argument("--out", type=Path, help="write full twitch exclusion report JSON here")
    add_twitch_args(parser)
    args = parser.parse_args()
    result = scan_lerobot(
        args.lerobot,
        window_s=args.window_s,
        rev_speed=args.rev_speed,
        max_rev_burst=args.max_rev_burst,
        max_state_speed=args.max_state_speed,
        max_track_err=args.max_track_err,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("flagged", "excluded_episode_indices")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
