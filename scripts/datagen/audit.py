"""Independent audit of an exported dataset against its raw episodes; the last step before release.

Run with h5py, numpy and scipy (the Isaac env interpreter works, nothing is launched):
    python scripts/datagen/audit.py --dataset output/datagen/office-beans-v3/lerobot/g1d_office_beans_v3
Re-reads every episode's HDF5 (frame counts across state, action and cameras, finite values), recomputes the task's
acceptance from the recorded privileged data (the plugin's audit()) and the motion metrics (motion.py), checks that the
layouts are distinct and reports their coverage. Writes validation_raw.json, layouts.csv and coverage.png (with
matplotlib), then final_validation.json and sha256.json over the whole dataset folder; upload.py refuses a dataset
without them. Episodes whose commands exceed the real arm's limits are listed; --strict-motion fails on any.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.export import layout_key  # noqa: E402

BINS = 20


def check_episode(path, quality, plugin):
    """(the plugin's audit metrics with the worst motion metrics, the motion violations) of one raw episode."""
    import h5py

    from scripts.datagen import motion  # needs scipy; rsi_release.py imports this module in the lerobot env

    n = quality["frames"]
    data = {"state": {}, "action": {}}
    with h5py.File(path, "r") as f:
        for key in quality["state_keys"]:
            for group in data:
                rows = data[group][key] = f[f"{group}/{key}s"][()]
                assert rows.shape[0] == n and np.isfinite(rows).all(), (path, group, key)
        for camera in quality["cameras"]:
            assert f[f"vision/{camera}/colors"].shape == (n,), (path, camera)
        privileged = {k: f[f"privileged/{k}"][()] for k in f["privileged"]}
    moved = motion.summarize(
        data["state"], data["action"], quality["fps"], quality["joint_names"], quality.get("joint_limits"), privileged
    )
    return plugin.audit(privileged, quality) | motion.worst(moved), moved["violations"]


def coverage(qualities):
    """Per labelled object: planned x, y (m) and yaw (deg) ranges and how many of BINS equal bins they fill."""
    rows, report = [], {}
    for q in qualities:
        for label, (pos, wxyz) in sorted(q["planned_poses"].items()):
            yaw = np.degrees(2 * np.arctan2(wxyz[3], wxyz[0]))
            rows.append([q["layout_seed"], q["layout_id"], label, pos[0], pos[1], yaw])
    for label in sorted({r[2] for r in rows}):
        values = np.array([r[3:] for r in rows if r[2] == label])
        report[label] = {}
        for axis, column in zip(("x", "y", "yaw"), values.T):
            low, high = float(column.min()), float(column.max())
            bins = np.clip(((column - low) / (high - low or 1) * BINS).astype(int), 0, BINS - 1)
            report[label][axis] = dict(min=low, max=high, bins_filled=len(set(bins)), bins=BINS)
    return rows, report


def plot(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = sorted({r[2] for r in rows})
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), layout="constrained", squeeze=False)
    for ax, label in zip(axes[0], labels):
        xy = np.array([r[3:5] for r in rows if r[2] == label]) * 100
        ax.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.8)
        ax.set(title=label, xlabel="x (cm)", ylabel="y (cm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--strict-motion", action="store_true", help="fail if any episode exceeds the real limits")
    parser.add_argument("--strict-twitch", action="store_true", help="fail if any episode exhibits arm twitching")
    args = parser.parse_args()
    root = args.dataset
    verification = json.loads((root / "export_verification.json").read_text())
    if verification["status"] != "verified":
        raise SystemExit("export.py did not verify this dataset")
    qualities = json.loads((root / "collection_quality.json").read_text())
    plugin = importlib.import_module(qualities[0]["plugin"])
    metrics, over, twitched = [], {}, {}
    for i, q in enumerate(qualities):
        episode_metrics, violations = check_episode(Path(q["raw"]), q, plugin)
        metrics.append(episode_metrics)
        if violations:
            over[Path(q["raw"]).name] = violations
        if episode_metrics.get("motion_twitch", 0) > 0:
            raw_path = Path(q["raw"])
            ep_key = f"{i}:{raw_path.parent.name}/{raw_path.name}" if raw_path.parent.name else f"{i}:{raw_path.name}"
            twitched[ep_key] = {
                k: episode_metrics[k]
                for k in (
                    "motion_rev_burst_max",
                    "motion_speed_max",
                    "motion_track_err_max",
                    "motion_osc_wiggle_max",
                )
                if k in episode_metrics
            }
        if (i + 1) % 10 == 0:
            print(f"audit {i + 1}/{len(qualities)}: pass", flush=True)
    if len({layout_key(q) for q in qualities}) != len(qualities):
        raise SystemExit("Duplicate layouts in the dataset")
    if args.strict_motion and over:
        raise SystemExit(f"{len(over)} episodes exceed the real arm's limits: {json.dumps(over)}")
    if args.strict_twitch and twitched:
        raise SystemExit(f"{len(twitched)} episodes exhibit arm twitching: {json.dumps(twitched)}")
    rows, report = coverage(qualities)
    numeric = {k for m in metrics for k, v in m.items() if isinstance(v, int | float)}
    summary = {
        k: dict(min=min(m[k] for m in metrics if k in m), max=max(m[k] for m in metrics if k in m))
        for k in sorted(numeric)
    }
    validation = dict(
        episodes=len(qualities),
        frames=sum(q["frames"] for q in qualities),
        metrics=summary,
        coverage=report,
        motion_over_real_limits=over,
        motion_twitch=twitched,
    )
    (root / "validation_raw.json").write_text(json.dumps(validation, indent=2) + "\n")
    header = "layout_seed,layout_id,label,x,y,yaw_deg\n"
    (root / "layouts.csv").write_text(header + "".join(",".join(map(str, r)) + "\n" for r in rows))
    plotted = plot(rows, root / "coverage.png")
    final = dict(
        status="validated",
        acceptance_version=qualities[0]["acceptance_version"],
        export=verification,
        audit=dict(
            episodes=len(qualities),
            recomputed_acceptance=True,
            distinct_layouts=True,
            coverage_plot=plotted,
            motion_over_real_limits=len(over),
            motion_twitch=len(twitched),
        ),
    )
    (root / "final_validation.json").write_text(json.dumps(final, indent=2) + "\n")
    write_checksums(root)
    print(json.dumps(dict(final, metrics=summary)), flush=True)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_checksums(root):
    """sha256.json over every file of the dataset folder; upload.py verifies the remote copy against it."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "sha256.json")
    sums = {str(p.relative_to(root)): sha256(p) for p in files}
    (root / "sha256.json").write_text(json.dumps(sums, indent=2) + "\n")


if __name__ == "__main__":
    main()
