"""Why place / put anything episodes whose grasp worked are still rejected: the place stage (carry, set down, let go,
back off) of every recorded episode, offline from its sidecar and HDF5, with pick_failures' grasp features.

Events on the last grasp (the closed stretch of the active hand's gripper command that ends in the release):
    t_shut     the jaw commanded shut          t_lift   X up HELD          t_over   the hand within OVER of where it
    t_release  commanded open (> RELEASE)                                           lets go (xy): carried over

Place outcome (`place`), first that applies:
    accepted        the acceptance held
    missed          the grasp failed (pick_failures' outcome is not lifted / carry_slip; X never up HELD)
    never_released  the hand never let go (the expert gave up: slipped in the hand, or the episode ran out)
    off_spot        X's centre outside the receptacle's region at the release: slipped (X moved > SLIP against the
                    hand since t_lift + HAND_REF) or aimed off
    tipped          X in the region at the end, tilted > X_TILT from how it lay (support), at the release already
                    (`tipped_at_release`) or after it
    slid_off        in the region at the release, out of it at the end (rolled, slid or bounced off)
    high            in the region and upright, X's bottom above the face / X's centre above the rim (resting on the
                    rim, an edge or another object)
    receptacle_moved / distractor_moved / moving / ...: the sidecar's final_fail otherwise, with when it moved
                    (`moved_in`: grasp, carry, place, after)

    python scripts/datagen/place_failures.py scan --runs '/obs/Employee/filler/raw/place-anything-r2-g1d/*' \
        --out /shared/tmp/$USER/pl42/r2-g1d.jsonl --sample 120 --sample-accepted 20
    python scripts/datagen/place_failures.py report a.jsonl b.jsonl --out report.md
"""

import argparse
from collections import Counter, defaultdict
import glob
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.datagen import pick_failures as pf  # noqa: E402
from scripts.objects import place_anything as pa  # noqa: E402

RELEASE, OVER, SLIP, HELD, HAND_REF = 0.5, 0.015, 0.02, pf.HELD, pf.HAND_REF
MOVE, MOVE_Z, Y_MOVE = 0.03, 0.02, pa.Y_STAY  # m: the plugin's STAY, STAY_Z; Y_STAY
STEP_LIMS = {"v4": 1100, "v41": 1400, "v42": 1400}  # plugin suffix -> STEP_LIM (v3: 900)
PLACES = ("accepted", "missed", "never_released", "off_spot", "tipped", "slid_off", "high")


def region_out(geom, xy):
    """m X's centre (receptacle frame) lies outside the region (0 inside)."""
    reg = geom["region"]
    dx, dy = (abs(v - c) for v, c in zip(xy, reg["center"]))
    rect = np.hypot(max(dx - reg["half"][0], 0.0), max(dy - reg["half"][1], 0.0))
    return float(max(min(rect, np.hypot(dx, dy) - reg["radius"]), 0.0))


def height(kind, geom, x, y, size):
    """support: X's bottom above the face; container: X's centre above the rim (both m, receptacle frame)."""
    if kind == "container":
        return float(pa.local(y[:3], y[3:], x[:3])[2] - geom["rim"])
    face = y[2] + pa.rotate(y[3:], (0.0, 0.0, geom["floor"]))[2]
    return float(pa.bottom_z(x[:3], x[3:], size) - face)


def last_grasp(c):
    """(t_shut, t_release or None) of the last closed stretch of gripper command `c` (before the final opening)."""
    shut = np.flatnonzero(c <= pf.SHUT)
    if not len(shut):
        return None, None
    opened = np.flatnonzero(c > RELEASE)
    ends = [t for t in opened if t > shut[0] and c[t - 1] <= RELEASE]
    t_release = int(ends[-1]) if ends else None
    stop = t_release if t_release is not None else len(c)
    closed = shut[shut < stop]
    t = int(closed[-1])
    while t > 0 and c[t - 1] <= pf.SHUT:
        t -= 1
    return t, t_release


def episode(sidecar):
    import h5py

    rec = pf.episode(sidecar)
    meta = json.loads(Path(sidecar).read_text())
    side = Path(sidecar)
    plugin = str(meta.get("plugin") or "")
    rec.update(
        round=side.parents[2].name.rsplit("-", 1)[0] if side.parents[3].name.startswith("out-") else rec.get("round"),
        final_fail=meta.get("final_fail"),
        kind=meta.get("kind"),
        receptacle_id=(meta.get("receptacle") or {}).get("id"),
        receptacle_family=(meta.get("objects") or {}).get(pa.LABEL, [None])[0],
        step_lim=STEP_LIMS.get(plugin.rsplit("_", 1)[-1], 900),
    )
    if rec.get("outcome") == "unknown":
        return rec
    if "t_close" not in rec:
        return dict(rec, place="accepted" if rec.get("accepted") else "missed")
    labels = meta["labels"]
    ti, ri = labels.index("target"), labels.index(pa.LABEL)
    geom, kind, size = meta["receptacle"], meta["kind"], meta["target_size"]
    start = np.asarray(meta["start"], dtype=float)
    with h5py.File(Path(sidecar).with_suffix(".hdf5"), "r") as f:
        poses = f["privileged/object_poses"][()].astype(float)
        cmd = {s: f[f"action/{s}_ee_joint_states"][()][:, 0] for s in ("left", "right")}
        q = {s: f[f"state/{s}_arm_joint_states"][()] for s in ("left", "right")}
    n = len(poses)
    arm = rec.get("arm") or min(cmd, key=lambda s: cmd[s].min())
    lift = pf.LIFT_JOINTS.get(rec["scene"], pf.LIFT_JOINTS["g1d"])
    t_shut, t_release = last_grasp(cmd[arm])
    rise = poses[:, ti, 2] - start[ti, 2]
    ev = dict(n=n, t_shut=t_shut, t_release=t_release)
    if t_shut is not None:
        up = np.flatnonzero(rise[t_shut:] >= HELD)
        ev["t_lift"] = t_lift = t_shut + int(up[0]) if len(up) else None
        stop = t_release if t_release is not None else n
        hand = np.array([pf.chain(arm).grasp_point(q[arm][t], lift) for t in range(t_shut, stop)])
        if t_lift is not None and len(hand):
            ref = min(t_lift - t_shut + HAND_REF, len(hand) - 1)
            rel = poses[t_shut:stop, ti, :3] - hand
            ev["slip"] = round(float(np.linalg.norm(rel[-1] - rel[ref])), 4)
            ev["slip_max"] = round(float(np.linalg.norm(rel[ref:] - rel[ref], axis=1).max()), 4)
            if t_release is not None:
                near = np.flatnonzero(np.linalg.norm(hand[ref:, :2] - hand[-1, :2], axis=1) < OVER)
                ev["t_over"] = t_shut + ref + int(near[0]) if len(near) else None
                ev["drop_height"] = round(
                    height(kind, geom, poses[t_release - 1, ti], poses[t_release - 1, ri], size), 4
                )
    for name, t in (("rel", t_release), ("end", n - 1)):
        if t is None:
            continue
        x, y = poses[t, ti], poses[t, ri]
        p = pa.local(y[:3], y[3:], x[:3])
        ev[f"out_{name}"] = round(region_out(geom, p[:2]), 4)
        ev[f"tilt_{name}"] = round(pa.tilt_deg(x[3:], start[ti, 3:]), 1)
        ev[f"height_{name}"] = round(height(kind, geom, x, y, size), 4)
        ev[f"ymove_{name}"] = round(float(np.linalg.norm(y[:2] - start[ri, :2])), 4)
    others = [i for i in range(len(labels)) if i not in (ti, ri)]
    moved = {}
    for i in others + [ri]:
        d = poses[:, i, :3] - start[i, :3]
        bad = (np.linalg.norm(d[:, :2], axis=1) >= (Y_MOVE if i == ri else MOVE)) | (
            (np.abs(d[:, 2]) >= MOVE_Z) if i != ri else False
        )
        hit = np.flatnonzero(bad)
        if len(hit):
            moved[labels[i]] = int(hit[0])
    ev["first_moved"] = moved
    rec.update(ev)
    rec["place"] = place_outcome(rec)
    return rec


def phase(t, r):
    """grasp / carry / place / after: which stretch of the episode frame `t` falls in."""
    if r.get("t_lift") is None or t < r["t_lift"]:
        return "grasp"
    if r.get("t_over") is not None and t < r["t_over"]:
        return "carry"
    if r.get("t_release") is not None and t < r["t_release"]:
        return "place" if r.get("t_over") is not None else "carry"
    return "after"


def place_outcome(r):
    if r.get("accepted"):
        return "accepted"
    if r.get("outcome") not in ("lifted", "carry_slip") or r.get("t_lift") is None:
        return "missed"
    if r.get("t_release") is None:
        return "never_released"
    fail = r.get("final_fail")
    if fail == "not_placed" or fail is None:
        if r["out_rel"] > 0:
            return "off_spot"
        if r["out_end"] == 0 and r["kind"] == "support" and r["tilt_end"] > pa.X_TILT:
            return "tipped"
        if r["out_end"] > 0:
            return "slid_off"
        return "high"
    return fail


def subtype(r):
    """A finer label within a place outcome."""
    p = r["place"]
    if p == "off_spot":
        return "slipped" if r.get("slip", 0) > SLIP else "aimed_off"
    if p == "tipped":
        return "at_release" if r["tilt_rel"] > pa.X_TILT else "after"
    if p == "never_released":
        if r.get("slip_max", 0) > 0.04:
            return "slipped"
        return "ran_out" if r.get("n", 0) >= r.get("step_lim", 900) else "other"
    if p in ("receptacle_moved", "distractor_moved"):
        moved = r.get("first_moved", {})
        hits = [t for k, t in moved.items() if (k == pa.LABEL) == (p == "receptacle_moved")]
        return phase(min(hits), r) if hits else "?"
    return ""


def scan(args):
    files = sorted(f for pat in args.runs for d in glob.glob(pat) for f in glob.glob(f"{d}/raw/seed*/episode_*.json"))
    print(f"{len(files)} sidecars", flush=True)
    with Pool(args.jobs) as pool:
        sides = pool.map(pf.side_info, files, chunksize=32)
    pick = files
    if args.sample:
        rng, pick, by = np.random.default_rng(0), [], defaultdict(lambda: ([], []))
        for r in sides:
            if r.get("outcome") != "unknown":
                by[r["round"]][r["accepted"]].append(r["sidecar"])
        for rejected, accepted in by.values():
            for group, k in ((rejected, args.sample), (accepted, args.sample_accepted)):
                pick += [group[i] for i in sorted(rng.permutation(len(group))[:k])]
    with open(args.out + ".sidecars.jsonl", "w") as out:
        out.writelines(json.dumps(r) + "\n" for r in sides)
    print(f"{len(pick)} HDF5", flush=True)
    with Pool(args.h5_jobs) as pool, open(args.out, "w") as out:
        for i, rec in enumerate(pool.imap_unordered(safe, pick, chunksize=2)):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 100 == 0:
                print(i, flush=True)


def safe(sidecar):
    try:
        return episode(sidecar)
    except Exception as e:  # noqa: BLE001 - a broken file is reported, not fatal
        return dict(sidecar=sidecar, outcome="unknown", why=f"{type(e).__name__}: {e}")


def med(values, fmt=".0f"):
    v = [x for x in values if x is not None]
    return format(float(np.median(v)), fmt) if v else "—"


def report(args):
    recs = [r for r in pf.load(args.inputs) if r.get("place")]
    lines = [f"# 放置段失败分析（{len(recs)} 条读了 HDF5）", ""]
    by = defaultdict(list)
    for r in recs:
        by[args.group_key and r.get(args.group_key) or r["round"]].append(r)
    cols = PLACES[2:] + ("receptacle_moved", "distractor_moved", "moving", "holding")
    lines += [
        "## 各集合（抽样，不合格按 place 结果分）",
        "",
        "| 集合 | 条数 | 合格 | 抓空 | " + " | ".join(cols) + " |",
    ]
    lines.append("|" + " --- |" * (4 + len(cols)))
    for k, rs in sorted(by.items()):
        c = Counter(r["place"] for r in rs)
        lines.append(
            f"| {k} | {len(rs)} | {c['accepted']} | {c['missed']} | " + " | ".join(str(c[x]) for x in cols) + " |"
        )
    lines += ["", "## 子类（抓住了但不合格）", ""]
    for k, rs in sorted(by.items()):
        c = Counter(f"{r['place']}:{subtype(r)}" for r in rs if r["place"] not in ("accepted", "missed"))
        lines.append(f"- {k}: " + "，".join(f"{a} {b}" for a, b in c.most_common()))
    lines += ["", "## 各阶段用时（帧，中位）", "", "| 集合 | place 结果 | 条数 | 抓取 t_lift | 搬运 t_over−t_lift | "
              "下放 t_release−t_over | 松手后 n−t_release | 全长 n | 松手时离面 cm | 松手时离区域 cm | 手中滑移 cm |",
              "|" + " --- |" * 11]  # fmt: skip
    for k, rs in sorted(by.items()):
        for p in PLACES:
            g = [r for r in rs if r["place"] == p]
            if not g or p == "missed":
                continue

            def d(a, b):
                return [r[b] - r[a] if r.get(a) is not None and r.get(b) is not None else None for r in g]

            lines.append(
                f"| {k} | {p} | {len(g)} | {med([r.get('t_lift') for r in g])} | {med(d('t_lift', 't_over'))} | "
                f"{med(d('t_over', 't_release'))} | {med(d('t_release', 'n'))} | {med([r.get('n') for r in g])} | "
                f"{med([r['drop_height'] * 100 if r.get('drop_height') is not None else None for r in g], '.1f')} | "
                f"{med([r['out_rel'] * 100 if r.get('out_rel') is not None else None for r in g], '.1f')} | "
                f"{med([r['slip'] * 100 if r.get('slip') is not None else None for r in g], '.1f')} |"
            )
    for title, key in (("形状", "shape"), ("物体大类", "family"), ("Y", "receptacle_family"), ("手", "arm")):
        lines += [
            "",
            f"## 按{title}（全部集合，抓住了的回合）",
            "",
            f"| {title} | 抓住 | 合格 | " + " | ".join(cols) + " |",
        ]
        lines.append("|" + " --- |" * (3 + len(cols)))
        g = defaultdict(list)
        for r in recs:
            if r["place"] != "missed":
                g[r.get(key)].append(r)
        for k, rs in sorted(g.items(), key=lambda kv: -len(kv[1]))[:25]:
            if len(rs) < args.min_n:
                continue
            c = Counter(r["place"] for r in rs)
            lines.append(f"| {k} | {len(rs)} | {c['accepted'] / len(rs):.0%} | " + " | ".join(
                f"{c[x] / len(rs):.0%}" for x in cols) + " |")  # fmt: skip
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--runs", nargs="+", required=True, help="unit dirs (globs) holding raw/seed*/episode_*.json")
    s.add_argument("--out", required=True)
    s.add_argument("-j", "--jobs", type=int, default=16)
    s.add_argument("--h5-jobs", type=int, default=8)
    s.add_argument("--sample", type=int, default=0, help="rejected episodes per collection (0: every episode)")
    s.add_argument("--sample-accepted", type=int, default=20)
    r = sub.add_parser("report")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--group-key", help="record field to group by instead of the collection (round)")
    r.add_argument("--min-n", type=int, default=15)
    args = p.parse_args(argv)
    {"scan": scan, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
