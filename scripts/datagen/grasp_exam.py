"""Grasp exam of the bean-pour expert (g1d_pour_beans_plus): did each cup stay in its hand from the grasp to the end
of the pour? Episodes are recorded by the exam plugin (scripts/datagen/plugins/g1d_pour_beans_plus_grasp.py, the
expert without putting the cups back), which writes each cup's outcome into the quality file; this module holds the
rules and the report.

Per cup, from its holding hand's frames (cup pose, hand pose, gripper command closed or not):
    knocked  before the grip had closed (SETTLE frames after the close command) the cup tipped KNOCK_DEG from how it
             stood or was pushed more than PUSH_M across the desk
    missed   the hand never rose LIFT_M above where it closed, or the cup rose less than half of that with it
    dropped  from the lift to the end the cup moved more than DROP_M from where it sat in the hand
    slipped  ... more than SLIP_M or turned more than SLIP_DEG in the hand, and stayed in it
    ok       none of these
    cut      ok, but the other cup ended the episode early (the plugin stops once a cup is lost): not a trial
    unrun    the episode was turned down before the expert moved (beans outside the source cup as spawned): no trial
A cup passes the exam, by the grasp entry it was held by (trials of the same entry pool across runs), when it is ok
in at least PASS of its trials as the source and as the target in each hand, and, with both cups held, POUR_OK of
the beans end in the target in at least PASS of the pours from it and of those into it.

Usage (repo root):
    python scripts/datagen/grasp_exam.py --run output/datagen/<batch> [--run ...] [--out DIR] [--grasps FILE]
writes <out>/grasp_exam.md and grasp_exam.json (default out: the first --run); with --grasps (the task's
real2sim/tasks/g1d_pour_beans_plus/grasps.json) the best grasp per cup and role (pick) goes into it and the chosen
grasps' table and the supported cups into <out>/grasp_pick.md.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

KNOCK_DEG, PUSH_M = 20.0, 0.03
SETTLE = 15  # frames: the close command (15 actions) until the pads hold
LIFT_M = 0.04
DROP_M, SLIP_M, SLIP_DEG = 0.04, 0.015, 15.0
PASS = 0.9
# beans in the target a pour needs: the task's success check (IN_CUP), not every bean (user, 2026-11-09); most pours
# short of every bean lost them through a cup wall, whatever the expert did
POUR_OK = 0.95
OUTCOMES = ("ok", "slipped", "dropped", "missed", "knocked", "cut", "unrun")
NO_TRIAL = ("cut", "unrun")
ROLES = ("source_cup", "target_cup")
HANDS = ("left", "right")
CELLS = tuple(f"{r}/{h}" for r in ROLES for h in HANDS)
POURS = ("pour", "pour_in")  # per role in ROLES: the pours from the cup, into it


def rotation(q):
    """(..., 3, 3) rotations of (..., 4) quaternions (w, x, y, z)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def tilt_deg(poses, up=(0.0, 0.0, 1.0)):
    """(T,) tilt of each pose's z axis from `up` (default vertical), degrees."""
    z = rotation(np.asarray(poses)[:, 3:])[:, :, 2]
    return np.degrees(np.arccos(np.clip(z @ np.asarray(up, float), -1, 1)))


def in_hand(cup, hand):
    """Cup poses in the hand frame: positions (T, 3) and rotations (T, 3, 3)."""
    r_hand = rotation(hand[:, 3:])
    pos = np.einsum("tji,tj->ti", r_hand, cup[:, :3] - hand[:, :3])
    return pos, np.einsum("tji,tjk->tik", r_hand, rotation(cup[:, 3:]))


def first(mask, offset=0):
    hits = np.flatnonzero(mask)
    return int(hits[0]) + offset if len(hits) else None


def classify(cup, hand, closed):
    """The grasp outcome of one cup: dict(outcome, frame, and the measures). cup, hand (T, 7) [x, y, z, qw, qx, qy,
    qz] per frame; closed (T,) the holding gripper's command is closed. Frames after the gripper opens again are
    left out (the cup is let go)."""
    cup, hand, closed = np.asarray(cup, float), np.asarray(hand, float), np.asarray(closed, bool)
    out = dict(outcome="missed", frame=None, push_m=0.0, tilt_deg=0.0, rise_m=0.0, slip_m=0.0, slip_deg=0.0)
    close = first(closed)
    if close is None:
        return out
    opened = first(~closed[close:], close)
    end = len(cup) if opened is None else opened
    grip = min(close + SETTLE, end - 1)
    tilt = tilt_deg(cup[: grip + 1], rotation(cup[0, 3:])[:, 2])
    push = np.linalg.norm(cup[: grip + 1, :2] - cup[0, :2], axis=1)
    out.update(push_m=round(float(push.max()), 4), tilt_deg=round(float(tilt.max()), 1))
    bad = (tilt > KNOCK_DEG) | (push > PUSH_M)
    if bad.any():
        return dict(out, outcome="knocked", frame=first(bad))
    lift = first(hand[grip:end, 2] - hand[grip, 2] >= LIFT_M, grip)
    if lift is None:
        return dict(out, frame=end - 1)
    rise = cup[lift, 2] - cup[grip, 2]
    out["rise_m"] = round(float(rise), 4)
    if rise < LIFT_M / 2:
        return dict(out, frame=lift)
    pos, rot = in_hand(cup[lift:end], hand[lift:end])
    moved = np.linalg.norm(pos - pos[0], axis=1)
    turned = np.degrees(np.arccos(np.clip((np.einsum("ij,tij->t", rot[0], rot) - 1) / 2, -1, 1)))
    out.update(slip_m=round(float(moved.max()), 4), slip_deg=round(float(turned.max()), 1))
    if (moved > DROP_M).any():
        return dict(out, outcome="dropped", frame=first(moved > DROP_M, lift))
    slip = (moved > SLIP_M) | (turned > SLIP_DEG)
    if slip.any():
        return dict(out, outcome="slipped", frame=first(slip, lift))
    return dict(out, outcome="ok", frame=None)


def grip_name(grip):
    """A cup's grasp entry for its role (scripted_demo.role_grip) as a short name: "handle", its keys, "default"."""
    if grip.get("mode") == "handle":
        return "handle"
    return ",".join(f"{k}={v}" for k, v in sorted(grip.items())) or "default"


def hand_of(q, role):
    """The hand that held the cup: the robot's right for a cup planned at x > 0 (the expert takes the nearer arm)."""
    return "right" if q["planned_poses"][role][0][0] > 0 else "left"


def tally(quality_files):
    """{"cup id [grip]": {"<role>/<hand>": Counter of outcomes, "pour" / "pour_in": Counter of in / short}} and the
    per-trial rows of the exam episodes' quality files: the trials of a cup held by the same grasp entry pool across
    runs. A pour counts when both cups stayed in hand, as "pour" for the source cup and "pour_in" for the target: "in"
    when POUR_OK of the beans ended in the target, else "short"."""
    cups = collections.defaultdict(lambda: {c: collections.Counter() for c in CELLS + POURS})
    rows = []
    for path in quality_files:
        q = json.loads(Path(path).read_text())
        grasp = q.get("grasp")
        if not grasp:
            continue
        counts = (q.get("initial_counts") or {}).get("total"), (q.get("final_counts") or {}).get("target")
        landed = round(counts[1] / counts[0], 4) if counts[0] and counts[1] is not None else None
        held = {}
        for role in ROLES:
            g = dict(grasp[role], grip=grasp[role].get("grip", "default"), hand=hand_of(q, role))
            if "rise_m" not in g:  # never classified: no frames
                g["outcome"] = "unrun"
            elif g["outcome"] == "ok" and q.get("stopped") and not q["stopped"].startswith(role):
                g["outcome"] = "cut"  # the other cup ended the episode before this one's pour was over
            key = held[role] = f"{g['cup']} [{g['grip']}]"
            cups[key][f"{role}/{g['hand']}"][g["outcome"]] += 1
            rows.append(dict(episode=str(path), role=role, key=key, landed=landed, **g))
        if landed is not None and all(r["outcome"] == "ok" for r in rows[-2:]):
            for role, pour in zip(ROLES, POURS):
                cups[held[role]][pour]["in" if landed >= POUR_OK else "short"] += 1
    return cups, rows


def trials(counts):
    """Trials that count: all but NO_TRIAL."""
    return sum(n for o, n in counts.items() if o not in NO_TRIAL)


def verdict(counts, good="ok"):
    n = trials(counts)
    return bool(n) and counts[good] / n >= PASS


def passes(cup):
    """Each role in each hand and the pours from and into it tried, all at PASS or better."""
    return all(verdict(cup[c]) for c in CELLS) and all(verdict(cup[p], "in") for p in POURS)


def holds(cup):
    """Each role in each hand tried and held in PASS of its trials or better (the pours left out)."""
    return all(verdict(cup[c]) for c in CELLS)


def parse_grip(name):
    """A grip_name back to its grasp entry ("default": none)."""
    if name == "default":
        return {}
    return {k: float(v) for k, v in (kv.split("=", 1) for kv in name.split(","))}


def role_score(counts, role):
    """((good + 1) / (tries + 2), tries) of a cup's trials in `role`: held in either hand. The pours do not count: 77%
    of the pours short of all beans lost them through a cup wall (V3b confirmation exam, 2026-11-07), whatever the
    grasp, and with them a grasp tried twice outscored one held 16 times in 16."""
    hands = [counts[f"{role}/{h}"] for h in HANDS]
    return (sum(c["ok"] for c in hands) + 1) / (sum(trials(c) for c in hands) + 2), sum(trials(c) for c in hands)


def empty():
    return {c: collections.Counter() for c in CELLS + POURS}


def pick(cups, drop=()):
    """Per cup the grasp per role that did best in `cups` (tally): the highest role_score of the default (the profile
    rule; untried in a role, still a candidate if the cup was tried with it at all: the exam leaves out a cup no scale
    of the default holds) and the grasps tried in that role, ties to the fewer keys (the default first). Returns the
    grasps.json entries {cup id: entry} (shared keys on top, the rest under "source" / "target"; none where both roles
    keep the default) and the chosen tally keys {cup id: {role: "cup [grip]"}}. The handle grasp and the cups in
    `drop` are left out."""
    by_cup = collections.defaultdict(dict)
    for key, counts in cups.items():
        cup, grip = key.rsplit(" [", 1)
        if grip[:-1] != "handle" and cup not in drop:
            by_cup[cup][grip[:-1]] = counts
    entries, chosen = {}, {}
    for cup, grips in sorted(by_cup.items()):
        default_tried = "default" in grips
        grips.setdefault("default", empty())
        names = sorted(grips, key=lambda g: (len(parse_grip(g)), g))
        best = {}
        for role in ROLES:
            tried = [g for g in names if role_score(grips[g], role)[1] or g == "default" and default_tried]
            tried = tried or ["default"]
            best[role] = max(tried, key=lambda g: role_score(grips[g], role)[0])
        src, tgt = (parse_grip(best[r]) for r in ROLES)
        entry = {k: v for k, v in src.items() if tgt.get(k) == v}
        for role, grip in (("source", src), ("target", tgt)):
            if own := {k: v for k, v in grip.items() if k not in entry}:
                entry[role] = own
        if entry:
            entries[cup] = entry
        chosen[cup] = {r: f"{cup} [{best[r]}]" for r in ROLES}
    return entries, chosen


def composite(cups, keys):
    """One cup's counts as held by its chosen grasp per role (pick): the role's cells and pours from its key."""
    out = {}
    for role, pour in zip(ROLES, POURS):
        counts = cups.get(keys[role]) or empty()
        out.update({c: counts[c] for c in CELLS if c.startswith(role)}, **{pour: counts[pour]})
    return out


def pick_report(cups, rows, chosen, drop):
    """The per-cup table of the chosen grasps (pick), the cups supported (passes, holds) and the ones dropped."""
    named = {}
    for cup, keys in chosen.items():
        grips = [keys[r].rsplit(" [", 1)[1][:-1] for r in ROLES]
        named[f"{cup} [{grips[0] if grips[0] == grips[1] else ' / '.join(grips)}]"] = composite(cups, keys)
    as_source = {keys["source_cup"]: name for name, keys in zip(named, chosen.values())}
    as_target = {keys["target_cup"] for keys in chosen.values()}
    rows = [dict(r, key=as_source[r["key"]]) for r in rows if r["role"] == "source_cup" and r["key"] in as_source] + [
        r for r in rows if r["role"] == "target_cup" and r["key"] in as_target
    ]
    full = [n for n, c in named.items() if passes(c)]
    held = [n for n, c in named.items() if holds(c) and not passes(c)]
    listed = [named[n][k] for n in full + held for k in CELLS] + [named[n][p] for n in full for p in POURS]
    least = min(map(trials, listed), default=0)
    median = sorted(map(trials, listed))[len(listed) // 2] if listed else 0
    lines = report(named, rows).splitlines()
    lines[0] = "# 抓取考试：逐杯选定的抓法（源杯 / 目标杯各自的最好抓法）"
    lines += [
        "",
        f"## 支持的杯子：{len(full)} 件（左右手、源杯和目标杯都抓稳，倒出和倒入的豆子都 ≥ {POUR_OK:.0%} 进目标杯，"
        f"每格 ≥ {PASS:.0%}）",
        "",
        *(f"- {n}" for n in sorted(full)),
        "",
        f"## 抓得稳但倾倒没达标的杯子：{len(held)} 件（四个抓取格都 ≥ {PASS:.0%}）",
        "",
        *(f"- {n}" for n in sorted(held)),
        "",
        f"上面两张清单里每格的次数很少（最少 {least} 次，中位 {median} 次）：一格 1/1 也算 ≥ {PASS:.0%}，"
        "所以这是“没见过失败”，不是 9/10 的证明。",
        "",
        f"## 放弃的杯子：{len(drop)} 件",
        "",
        *(f"- {c}：{why}" for c, why in sorted(drop.items())),
    ]
    return "\n".join(lines) + "\n"


def report(cups, rows):
    lines = ["# 抓取考试", ""]
    total = sum(trials(c[k]) for c in cups.values() for k in CELLS)
    ok = sum(c[k]["ok"] for c in cups.values() for k in CELLS)
    pours = sum(trials(c["pour"]) for c in cups.values())
    full = sum(c["pour"]["in"] for c in cups.values())
    cut, unrun = (sum(c[k][o] for c in cups.values() for k in CELLS) for o in NO_TRIAL)
    lines += [
        f"{len(cups)} 件杯子（按抓法分），{total} 次抓取，{ok} 次没滑没掉没碰倒（{ok / max(total, 1):.0%}）；"
        f"两杯都拿住的 {pours} 次倾倒里 {full} 次 ≥ {POUR_OK:.0%} 的豆子进了目标杯（{full / max(pours, 1):.0%}）；"
        f"{sum(passes(c) for c in cups.values())} 件左右手、两种角色和倾倒都达到 {PASS:.0%}。"
        f"另有 {cut} 次因另一只杯子先掉而没倒完、{unrun // 2} 个 episode 在专家动手前被拒（豆子生成时就在源杯外），"
        "不计。",
        "",
        f"| 杯子 [抓法] | 源杯左手 | 源杯右手 | 目标杯左手 | 目标杯右手 | 倒出 ≥ {POUR_OK:.0%} | 倒入 ≥ {POUR_OK:.0%} "
        "| 倒入（中位） | 失败 |",
        "| :-- | :-- | :-- | :-- | :-- | :-- | :-- | --: | :-- |",
    ]

    def share(c):
        return sum(cups[c][k]["ok"] for k in CELLS) / max(1, sum(trials(cups[c][k]) for k in CELLS))

    poured = collections.defaultdict(list)
    for r in rows:
        if r["role"] == "source_cup" and r["outcome"] == "ok" and r.get("landed") is not None:
            poured[r["key"]].append(r["landed"])

    def cell(counts, good="ok"):
        n = trials(counts)
        return f"{counts[good]}/{n}" if n else "—"

    for cup in sorted(cups, key=lambda c: (share(c), c)):
        cells = [cell(cups[cup][k]) for k in CELLS] + [cell(cups[cup][p], "in") for p in POURS]
        fails = collections.Counter(
            o for k in CELLS for o, n in cups[cup][k].items() for _ in range(n) if o not in ("ok", *NO_TRIAL)
        )
        landed = f"{np.median(poured[cup]):.0%}" if poured[cup] else "—"
        fail = ", ".join(f"{o} {k}" for o, k in fails.items())
        lines.append(f"| {cup} | {' | '.join(cells)} | {landed} | {fail} |")
    by = collections.Counter((r["role"], r["outcome"]) for r in rows)
    lines += ["", "| 结果 | 源杯 | 目标杯 |", "| :-- | --: | --: |"]
    lines += [f"| {o} | {by[('source_cup', o)]} | {by[('target_cup', o)]} |" for o in OUTCOMES]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--grasps",
        type=Path,
        help="a grasps.json: its cups replaced by the pick (its note and drop list kept); writes <out>/grasp_pick.md",
    )
    args = parser.parse_args()
    files = sorted(p for run in args.run for p in run.rglob("episode_*.json"))
    cups, rows = tally(files)
    out = args.out or args.run[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "grasp_exam.md").write_text(report(cups, rows))
    summary = {c: {k: dict(n) for k, n in v.items()} for c, v in sorted(cups.items())}
    (out / "grasp_exam.json").write_text(json.dumps(dict(cups=summary, trials=rows), indent=1) + "\n")
    print(out / "grasp_exam.md")
    if args.grasps:
        grasps = json.loads(args.grasps.read_text())
        drop = grasps.get("drop", {})
        grasps["cups"], chosen = pick(cups, drop)
        args.grasps.write_text(json.dumps(grasps, indent=1, ensure_ascii=False) + "\n")
        (out / "grasp_pick.md").write_text(pick_report(cups, rows, chosen, drop))
        print(args.grasps, out / "grasp_pick.md")


if __name__ == "__main__":
    main()
