"""Early end of collection episodes that can no longer be accepted, so collect.py moves on instead of running the
script and its tail to the end. Only conditions the acceptance can never recover from end an episode; an episode that
would be accepted runs and records exactly as without this module, and collect.py only ends one on a frame that does
not pass the acceptance anyway (its stable count is 0).

g1d_pour_beans_plus (plugins in PLUS, acceptance plus_v1 / plus_v2: every bean in the target cup, plus_v3: 95% of
them; the cups never touching, no distractor moved):
- touched: the cups' bodies touched (the plugin's flag only ever turns on);
- distractor_moved: a distractor moved past the limit (the plugin keeps the largest shift so far);
- beans_lost: for PERSIST frames running a bean has lain outside both cavities below FLOOR_H over the lower cup's
  underside at the start (on the desk or the floor), farther than its bounding sphere plus REACH from both cups'
  centres now and at the start. Nothing in the script lifts a bean off the desk, and a cup cannot come down over one
  that far from where it stands and where it goes back.

    python scripts/datagen/early_abort.py scan --sidecars 'raw/*/batch/*/raw/seed*/episode_*.json' --out report.json
replays the checks on recorded episodes (privileged/ of the HDF5, or the .npz a rejected one was cut down to): the
frame each would have ended at, and any accepted episode one would have ended (there must be none).
"""

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.objects import cup_profile  # noqa: E402

PLUS = (
    "scripts.datagen.plugins.g1d_pour_beans_plus",
    "scripts.datagen.plugins.g1d_pour_beans_plus_v2",
    "scripts.datagen.plugins.g1d_pour_beans_plus_v3",
    "scripts.datagen.plugins.g1d_pour_beans_plus_limited",
)
LABELS = ("source_cup", "target_cup")
FLOOR_H = 0.015  # m over the lower cup's underside at the start
REACH = 0.03  # m past a cup's bounding sphere
PERSIST = 15  # frames a bean stays lost


def centres(poses, cups):
    """(2, 2) world xy of each cup's centre (mid-height on its axis) for poses (2, 7)."""
    out = []
    for (cup, scale), pose in zip(cups, poses):
        local = np.r_[np.asarray(cup["axis"]) * scale, (cup["bottom"] + cup["top"]) / 2 * scale]
        out.append((pose[:3] + cup_profile.rotation(pose[3:]) @ local)[:2])
    return np.array(out)


class PourBeansPlus:
    """Checks of one episode; call once per frame with the beans (N, 3) and cup poses (2, 7) after its action and the
    plugin's touched flag and largest distractor shift so far. Returns the reason to end the episode, or None."""

    def __init__(self, cups, start, start_z, moved_limit, in_target_share=1.0):
        self.cups = [(cups[label][0], float(cups[label][1])) for label in LABELS]
        self.floor = min(z + cup["bottom"] * scale for (cup, scale), z in zip(self.cups, start_z)) + FLOOR_H
        sphere = [np.hypot(max(cup["body"]["r"]), (cup["top"] - cup["bottom"]) / 2) * scale for cup, scale in self.cups]
        self.reach = np.array(sphere) + REACH
        self.start = centres(np.asarray(start, float), self.cups)
        self.moved_limit = moved_limit
        self.in_target_share = float(in_target_share)
        self.run = None

    def __call__(self, points, poses, touched, moved):
        if touched:
            return "touched"
        if moved > self.moved_limit:
            return "distractor_moved"
        points, poses = np.asarray(points, float), np.asarray(poses, float)
        inside = np.zeros(len(points), bool)
        for (cup, scale), pose in zip(self.cups, poses):
            inside |= cup_profile.in_cavity(points, pose[:3], pose[3:], cup, scale)
        lost = ~inside & (points[:, 2] < self.floor)
        for centre, reach in zip(np.r_[centres(poses, self.cups), self.start], np.r_[self.reach, self.reach]):
            lost &= np.hypot(*(points[:, :2] - centre).T) > reach
        self.run = np.where(lost, (self.run if self.run is not None else 0) + 1, 0)
        lost_count = int((self.run >= PERSIST).sum())
        return "beans_lost" if len(points) - lost_count < self.in_target_share * len(points) else None


def for_episode(plugin, episode):
    """The checks of a collect.py episode as a callable of the episode after each frame, or None for a plugin without
    any."""
    if plugin.__name__ not in PLUS:
        return None
    summary = episode.summary()
    cups = {label: episode.cups[label] for label in LABELS}
    check = PourBeansPlus(
        cups, episode.start, episode.start_z, summary["moved_limit_m"], summary.get("in_target_share", 1.0)
    )

    def after_frame(ep):
        h = ep.history
        return check(h["particles"][-1], h["cup_poses"][-1], ep.touched, ep.moved)

    return after_frame


def replay(quality, privileged):
    """(frame, reason) the checks end a recorded plus episode at (1-based, as collect.py counts frames), or None."""
    cups = {label: (quality["cups"][label]["cavity"], quality["cups"][label]["scale"]) for label in LABELS}
    poses = np.asarray(privileged["cup_poses"], float)
    check = PourBeansPlus(
        cups, poses[0], quality["start_z"], quality["moved_limit_m"], quality.get("in_target_share", 1.0)
    )
    others = np.asarray(privileged["distractor_poses"], float)
    first = np.asarray(privileged["distractor_poses_start"], float).reshape(-1, 7)
    touched, moved = False, 0.0
    for t, (points, frame) in enumerate(zip(privileged["particles"], poses)):
        (a, b), ((ca, sa), (cb, sb)) = frame, check.cups
        touched |= cup_profile.touching((a[:3], a[3:]), ca, (b[:3], b[3:]), cb, sa, sb)
        if len(first):
            moved = max(moved, float(np.linalg.norm(others[t].reshape(-1, 7)[:, :2] - first[:, :2], axis=1).max()))
        if reason := check(points, frame, touched, moved):
            return t + 1, reason
    return None


def load_privileged(sidecar):
    npz = sidecar.with_suffix(".npz")
    if npz.exists():
        return dict(np.load(npz))
    import h5py

    with h5py.File(sidecar.with_suffix(".hdf5")) as f:
        return {k: f[f"privileged/{k}"][()] for k in f["privileged"]}


def scan_one(path):
    sidecar = Path(path)
    quality = json.loads(sidecar.read_text())
    row = dict(path=str(sidecar), accepted=bool(quality.get("accepted")), frames=quality.get("frames"))
    if quality.get("reason") or quality.get("task") != "g1d_pour_beans_plus":
        return dict(row, skipped=quality.get("reason") or quality.get("task"))
    try:
        hit = replay(quality, load_privileged(sidecar))
    except (OSError, KeyError, ValueError) as err:
        return dict(row, skipped=f"{type(err).__name__}: {err}")
    return dict(row, abort_frame=hit and hit[0], abort_reason=hit and hit[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="replay the checks on recorded episodes")
    scan.add_argument("--sidecars", nargs="+", required=True, help="globs of episode_*.json")
    scan.add_argument("--out", type=Path, required=True)
    scan.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    paths = sorted({p for pattern in args.sidecars for p in glob.glob(pattern)})
    from multiprocessing import Pool

    with Pool(args.jobs) as pool:
        rows = pool.map(scan_one, paths, chunksize=4)
    done = [r for r in rows if "skipped" not in r]
    killed = [r for r in done if r["accepted"] and r["abort_frame"]]
    rejected = [r for r in done if not r["accepted"]]
    caught = [r for r in rejected if r["abort_frame"]]
    summary = dict(
        episodes=len(rows),
        skipped=len(rows) - len(done),
        accepted=len(done) - len(rejected),
        accepted_aborted=len(killed),
        rejected=len(rejected),
        rejected_aborted=len(caught),
        reasons={k: sum(r["abort_reason"] == k for r in caught) for k in ("touched", "distractor_moved", "beans_lost")},
        rejected_frames=sum(r["frames"] for r in rejected),
        frames_saved=sum(r["frames"] - r["abort_frame"] for r in caught),
        all_frames=sum(r["frames"] for r in done),
    )
    args.out.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    if killed:
        sys.exit(f"{len(killed)} accepted episodes would have been ended: {[r['path'] for r in killed[:5]]}")


if __name__ == "__main__":
    main()
