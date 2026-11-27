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
from scripts.objects import cup_profile, place_anything as pa  # noqa: E402

PLUS = (
    "scripts.datagen.plugins.g1d_pour_beans_plus",
    "scripts.datagen.plugins.g1d_pour_beans_plus_v2",
    "scripts.datagen.plugins.g1d_pour_beans_plus_v3",
    "scripts.datagen.plugins.g1d_pour_beans_plus_limited",
    "scripts.datagen.plugins.g1d_pour_beans_plus_wholebody",
    "scripts.datagen.plugins.g1d_pour_beans_plus_sharpa",
)
HANDOVER = (
    "scripts.datagen.plugins.g1d_handover_anything",
    "scripts.datagen.plugins.g1d_handover_anything_v43",
    "scripts.datagen.plugins.g1d_handover_anything_wb",
    "scripts.datagen.plugins.g1d_handover_anything_sharpa",
)
PICK = (
    "scripts.datagen.plugins.g1d_pick_anything",
    "scripts.datagen.plugins.g1d_pick_anything_v4",
    "scripts.datagen.plugins.g1d_pick_anything_v41",
    "scripts.datagen.plugins.g1d_pick_anything_wb",
    "scripts.datagen.plugins.g1d_pick_anything_sharpa",
)
PLACE = (
    "scripts.datagen.plugins.g1d_place_anything",
    "scripts.datagen.plugins.g1d_place_anything_v4",
    "scripts.datagen.plugins.g1d_place_anything_v41",
    "scripts.datagen.plugins.g1d_place_anything_v42",
    "scripts.datagen.plugins.g1d_place_anything_v43",
    "scripts.datagen.plugins.g1d_place_anything_v44",
    "scripts.datagen.plugins.g1d_place_anything_wb",
    "scripts.datagen.plugins.g1d_place_anything_sharpa",
    "scripts.datagen.plugins.g1d_put_anything_in",
    "scripts.datagen.plugins.g1d_put_anything_in_v4",
    "scripts.datagen.plugins.g1d_put_anything_in_v41",
    "scripts.datagen.plugins.g1d_put_anything_in_v42",
    "scripts.datagen.plugins.g1d_put_anything_in_v43",
    "scripts.datagen.plugins.g1d_put_anything_in_v44",
    "scripts.datagen.plugins.g1d_put_anything_in_wb",
    "scripts.datagen.plugins.g1d_put_anything_in_sharpa",
)
SORT = (
    "scripts.datagen.plugins.g1d_sort_anything",
    "scripts.datagen.plugins.g1d_sort_anything_v3",
    "scripts.datagen.plugins.g1d_sort_anything_v41",
    "scripts.datagen.plugins.g1d_sort_anything_v44",
)
LABELS = ("source_cup", "target_cup")
FLOOR_H = 0.015  # m over the lower cup's underside at the start
REACH = 0.03  # m past a cup's bounding sphere
PERSIST = 15  # frames a condition stays broken before aborting
# Disabled by default: GraspRobot.GRASP_TRIES=3 and place_v42.PLACE_REGRASPS=1 legitimately produce accepted episodes
# on a 3rd grasp attempt or regrasp (verified on OBS pick-r7/place-r3/put-r3). Callers may still opt in via kwarg.
MAX_GRASP_TRIES = None
STAY, STAY_Z = 0.03, 0.02  # m, distractor shift limits (g1d_pick_anything / g1d_place_anything)
# Conservative unrecoverable abort thresholds (strictly looser than acceptance so rocking/nudged objects that settle
# back within acceptance limits never cause a false abort; verified 0 false aborts on OBS scan):
ABORT_STAY = 0.08  # m horizontal distractor shift (acceptance STAY=0.03)
ABORT_FALL_Z = 0.05  # m below start z (fallen off the desk; matches place_v42.PLACE_GONE)
ABORT_REC_STAY = 0.10  # m horizontal receptacle shift (acceptance Y_STAY=0.04)
ABORT_REC_TILT_COS = float(np.cos(np.radians(75.0)))  # receptacle tipped flat (> 75 deg)
ABORT_REC_TILT_PERSIST = 45  # frames tipped flat before aborting (rocking bounces recover in < 30 frames)
ITEM_FALL_Z = 0.15  # m below the lowest start z (actually off the desk; a tall item tipping on the desk stays above)


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


def _grasp_retry_exceeded(attempts, max_tries=MAX_GRASP_TRIES):
    """True once `max_tries` grasp attempts have failed (non-'held' result) or a try past `max_tries` started."""
    if max_tries is None or not attempts:
        return False
    failed = sum(1 for a in attempts if isinstance(a, dict) and "result" in a and a["result"] != "held")
    return failed >= max_tries or len(attempts) > max_tries


def _distractor_moved(poses, labels, start, skip_labels=("target",)):
    for i, label in enumerate(labels):
        if label in skip_labels:
            continue
        move = poses[i, :3] - start[i, :3]
        if np.linalg.norm(move[:2]) >= ABORT_STAY or move[2] < -ABORT_FALL_Z:
            return True
    return False


def _receptacle_displaced(pose, start_xyz):
    """True if receptacle shifted far horizontally (>= ABORT_REC_STAY) or fell off the desk."""
    move = pose[:3] - start_xyz[:3]
    return bool(np.linalg.norm(move[:2]) >= ABORT_REC_STAY or move[2] < -ABORT_FALL_Z)


def _receptacle_tipped(quat_wxyz):
    """True if receptacle's local +z axis is tilted past 75 deg from world +z."""
    w, x, y, z = quat_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return bool(up_z < ABORT_REC_TILT_COS)


class HandoverAnything:
    """Early-abort checks for g1d_handover_anything: sticky wrong/clash/dropped flags abort immediately, and
    unrecoverable receptacle_moved / distractor_moved abort after persistence."""

    def __init__(self, labels, start, persist=PERSIST, tilt_persist=ABORT_REC_TILT_PERSIST):
        self.labels = list(labels)
        self.start = np.asarray(start, dtype=float)
        self.r = self.labels.index(pa.LABEL)
        self.persist = int(persist)
        self.tilt_persist = int(tilt_persist)
        self.receptacle_run = 0
        self.receptacle_tilt_run = 0
        self.distractor_run = 0

    def __call__(self, poses, *, wrong=False, clash=False, dropped=False):
        if wrong:
            return "wrong_hand"
        if clash:
            return "clash"
        if dropped:
            return "dropped"
        poses = np.asarray(poses, dtype=float)
        y = poses[self.r]
        self.receptacle_run = self.receptacle_run + 1 if _receptacle_displaced(y, self.start[self.r]) else 0
        self.receptacle_tilt_run = self.receptacle_tilt_run + 1 if _receptacle_tipped(y[3:]) else 0
        if self.receptacle_run >= self.persist or self.receptacle_tilt_run >= self.tilt_persist:
            return "receptacle_moved"
        moved = _distractor_moved(poses, self.labels, self.start, skip_labels=("target", pa.LABEL))
        self.distractor_run = self.distractor_run + 1 if moved else 0
        if self.distractor_run >= self.persist:
            return "distractor_moved"
        return None


class PickAnything:
    """Early-abort checks for g1d_pick_anything*: sustained distractor_moved (and optional grasp retry cap)."""

    def __init__(self, labels, start, max_grasp_tries=MAX_GRASP_TRIES, persist=PERSIST):
        self.labels = list(labels)
        self.start = np.asarray(start, dtype=float)
        self.max_grasp_tries = None if max_grasp_tries is None else int(max_grasp_tries)
        self.persist = int(persist)
        self.distractor_run = 0

    def __call__(self, poses, grasp_attempts=()):
        if _grasp_retry_exceeded(grasp_attempts, self.max_grasp_tries):
            return "grasp_retry_cap"
        poses = np.asarray(poses, dtype=float)
        moved = _distractor_moved(poses, self.labels, self.start, skip_labels=("target",))
        self.distractor_run = self.distractor_run + 1 if moved else 0
        if self.distractor_run >= self.persist:
            return "distractor_moved"
        return None


class PlaceAnything:
    """Early-abort checks for g1d_place_anything* and g1d_put_anything_in*: sustained unrecoverable
    receptacle_moved, sustained distractor_moved (and optional grasp retry cap)."""

    def __init__(
        self, labels, start, max_grasp_tries=MAX_GRASP_TRIES, persist=PERSIST, tilt_persist=ABORT_REC_TILT_PERSIST
    ):
        self.labels = list(labels)
        self.start = np.asarray(start, dtype=float)
        self.r = self.labels.index(pa.LABEL)
        self.max_grasp_tries = None if max_grasp_tries is None else int(max_grasp_tries)
        self.persist = int(persist)
        self.tilt_persist = int(tilt_persist)
        self.receptacle_run = 0
        self.receptacle_tilt_run = 0
        self.distractor_run = 0

    def __call__(self, poses, grasp_attempts=()):
        if _grasp_retry_exceeded(grasp_attempts, self.max_grasp_tries):
            return "grasp_retry_cap"
        poses = np.asarray(poses, dtype=float)
        y = poses[self.r]
        self.receptacle_run = self.receptacle_run + 1 if _receptacle_displaced(y, self.start[self.r]) else 0
        self.receptacle_tilt_run = self.receptacle_tilt_run + 1 if _receptacle_tipped(y[3:]) else 0
        if self.receptacle_run >= self.persist or self.receptacle_tilt_run >= self.tilt_persist:
            return "receptacle_moved"
        moved = _distractor_moved(poses, self.labels, self.start, skip_labels=("target", pa.LABEL))
        self.distractor_run = self.distractor_run + 1 if moved else 0
        if self.distractor_run >= self.persist:
            return "distractor_moved"
        return None


class SortAnything:
    """Early-abort checks for g1d_sort_anything*: sustained unrecoverable receptacle_moved or item_fallen."""

    def __init__(
        self,
        labels,
        items,
        receptacles,
        start,
        fall_z=ITEM_FALL_Z,
        persist=PERSIST,
        tilt_persist=ABORT_REC_TILT_PERSIST,
    ):
        self.labels = list(labels)
        self.at = {label: i for i, label in enumerate(self.labels)}
        self.item_idx = [self.at[x] for x, _ in items]
        self.rec_idx = [self.at[y] for y in receptacles]
        self.start = np.asarray(start, dtype=float)
        self.floor_z = float(np.min(self.start[:, 2])) - float(fall_z)
        self.persist = int(persist)
        self.tilt_persist = int(tilt_persist)
        self.receptacle_run = 0
        self.receptacle_tilt_run = 0
        self.fallen_run = 0

    def __call__(self, poses):
        poses = np.asarray(poses, dtype=float)
        rec_disp = any(_receptacle_displaced(poses[i], self.start[i]) for i in self.rec_idx)
        rec_tip = any(_receptacle_tipped(poses[i, 3:]) for i in self.rec_idx)
        self.receptacle_run = self.receptacle_run + 1 if rec_disp else 0
        self.receptacle_tilt_run = self.receptacle_tilt_run + 1 if rec_tip else 0
        if self.receptacle_run >= self.persist or self.receptacle_tilt_run >= self.tilt_persist:
            return "receptacle_moved"
        fallen = any(poses[i, 2] < self.floor_z for i in self.item_idx)
        self.fallen_run = self.fallen_run + 1 if fallen else 0
        if self.fallen_run >= self.persist:
            return "item_fallen"
        return None


def for_episode(plugin, episode):
    """The checks of a collect.py episode as a callable of the episode after each frame, or None for a plugin without
    any."""
    name = plugin.__name__
    if name in PLUS:
        summary = episode.summary()
        cups = {label: episode.cups[label] for label in LABELS}
        check = PourBeansPlus(
            cups, episode.start, episode.start_z, summary["moved_limit_m"], summary.get("in_target_share", 1.0)
        )

        def after_frame(ep):
            h = ep.history
            return check(h["particles"][-1], h["cup_poses"][-1], ep.touched, ep.moved)

        return after_frame
    if name in HANDOVER:
        check = HandoverAnything(episode.labels, episode.meta["start"])

        def after_handover(ep):
            tr = ep.tracker
            return check(ep.history["object_poses"][-1], wrong=tr.wrong, clash=tr.clash, dropped=tr.dropped)

        return after_handover
    if name in PICK:
        check = PickAnything(episode.labels, episode.meta["start"])

        def after_pick(ep):
            return check(ep.history["object_poses"][-1], getattr(ep.env, "grasp_attempts", ()))

        return after_pick
    if name in PLACE:
        check = PlaceAnything(episode.labels, episode.meta["start"])

        def after_place(ep):
            return check(ep.history["object_poses"][-1], getattr(ep.env, "grasp_attempts", ()))

        return after_place
    if name in SORT:
        check = SortAnything(episode.labels, episode.meta["items"], episode.meta["receptacles"], episode.meta["start"])

        def after_sort(ep):
            return check(ep.history["object_poses"][-1])

        return after_sort
    return None


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


def replay_x_anything(quality, privileged):
    """(frame, reason) the X-Anything early-abort checks end a recorded episode at (1-based), or None."""
    task = quality.get("task")
    poses = np.asarray(privileged["object_poses"], dtype=float)
    labels, start = quality["labels"], quality["start"]
    if task == "g1d_handover_anything":
        from scripts.datagen.plugins.g1d_handover_anything import Tracker

        tracker = Tracker(
            labels, start, quality["receptacle"], quality["target_size"], quality["hands"], quality["giver"]
        )
        check = HandoverAnything(labels, start)
        for t in range(len(poses)):
            tracker.update(
                t,
                poses[t],
                privileged["gripper_openness"][t],
                privileged["pad_forces"][t],
                privileged["hand_gap"][t],
            )
            if reason := check(poses[t], wrong=tracker.wrong, clash=tracker.clash, dropped=tracker.dropped):
                return t + 1, reason
        return None
    if task == "g1d_pick_anything":
        check = PickAnything(labels, start)
        for t in range(len(poses)):
            if reason := check(poses[t]):
                return t + 1, reason
        return None
    if task in ("g1d_place_anything", "g1d_put_anything_in"):
        check = PlaceAnything(labels, start)
        for t in range(len(poses)):
            if reason := check(poses[t]):
                return t + 1, reason
        return None
    if task == "g1d_sort_anything":
        check = SortAnything(labels, quality["items"], quality["receptacles"], start)
        for t in range(len(poses)):
            if reason := check(poses[t]):
                return t + 1, reason
        return None
    return None


def load_privileged(sidecar):
    npz = sidecar.with_suffix(".npz")
    if npz.exists():
        with np.load(npz) as f:
            return {k: f[k] for k in f.files}
    import h5py

    with h5py.File(sidecar.with_suffix(".hdf5")) as f:
        return {k: f[f"privileged/{k}"][()] for k in f["privileged"]}


SUPPORTED_TASKS = {
    "g1d_pour_beans_plus",
    "g1d_handover_anything",
    "g1d_pick_anything",
    "g1d_place_anything",
    "g1d_put_anything_in",
    "g1d_sort_anything",
}


def scan_one(path):
    sidecar = Path(path)
    quality = json.loads(sidecar.read_text())
    row = dict(path=str(sidecar), accepted=bool(quality.get("accepted")), frames=quality.get("frames"))
    task = quality.get("task")
    if quality.get("reason") or task not in SUPPORTED_TASKS:
        return dict(row, skipped=quality.get("reason") or task)
    try:
        priv = load_privileged(sidecar)
        hit = replay(quality, priv) if task == "g1d_pour_beans_plus" else replay_x_anything(quality, priv)
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
    reason_keys = sorted({"touched", "distractor_moved", "beans_lost"} | {r["abort_reason"] for r in caught})
    summary = dict(
        episodes=len(rows),
        skipped=len(rows) - len(done),
        accepted=len(done) - len(rejected),
        accepted_aborted=len(killed),
        rejected=len(rejected),
        rejected_aborted=len(caught),
        reasons={k: sum(r["abort_reason"] == k for r in caught) for k in reason_keys},
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
