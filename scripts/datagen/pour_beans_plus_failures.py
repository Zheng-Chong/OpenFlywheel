"""Failure analysis of a policy on the g1d_pour_beans_plus benchmark
(real2sim/tasks/g1d_pour_beans_plus/benchmark.json): every episode sidecar of scripts/g1d/remote_policy.py joined
with its layout's record (levels, cups, scales, beans, hand, instruction), the gripper trace next to it
(episode_<id>.npz: command and finger joints) and, optionally, the generalization records of the training episodes,
so a failure can be set against what the training data covered.

Each episode gets one primary failure class (CLASSES, first match in that order) and tags:

    reject       the start was rejected (beans not all in A)
    repoured     the task check passed at some point but under half of the beans are in B at the end (B poured
                 back or knocked over after the pour)
    missed       nothing in B and over 5% of the beans are out of both cups: poured beside B, or A dropped or
                 knocked over
    no_grasp     nothing in B, the source arm never closed on the cup (empty closes only)
    no_pour      nothing in B, the source arm held the cup
    big_spill    over 5% of the beans out of both cups
    leftover     beans left in A (none or few spilled)
    small_spill  1 bean to 5% out of both cups, none left in A
    touched      every bean in B, the cups' bodies touched
    pushed       every bean in B, a distractor moved over 2 cm
    not_placed   every bean in B, no touch, no push, yet not settled for 30 frames: a cup not upright or not back
                 on the desk (tag still_holding: a gripper closed at the end)
    ok           accepted (plus_v1)

    tags: regrasp (an arm closed on something more than once), still_holding, timeout (ran the full step limit),
          touched, pushed, empty_close (a close on nothing), target_not_held (the target arm never held its cup)

The summary also fits a ridge logistic regression of the strict outcome on the single-dimension diagnostics and G0
(one indicator per dimension level, plus scene and hand, which those sets do not hold fixed), and one on every
episode with the layout's factors, so a level's effect is read with the scene and hand taken out.

    python scripts/datagen/pour_beans_plus_failures.py --sidecars <dir> [<dir> ...] --model plus \
        --benchmark real2sim/tasks/g1d_pour_beans_plus/benchmark.json --train <train generalization json> --out <dir>

<dir> holds <eval_id>/<env_cfg>/<seed>/rep<r>/episode_<i>.json (+ .npz), eval ids owam-bp-<model>-<date> (one
evaluation of the whole benchmark, pour_beans_plus@v1) or, from before, owam-bp-<model>-<bench>-<date> (one per
group); an episode's group (G0..G3, diag_<dimension>) is the one whose job has its env_cfg and seed. --train: a
JSON list of the training layouts' "Generalization" records, one per training episode. Writes episodes.jsonl,
summary.json and summary.md.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIMS = ("spatial", "visual", "object", "camera", "language")
CLASSES = ("reject", "repoured", "missed", "no_grasp", "no_pour", "big_spill", "leftover", "small_spill", "touched",
           "pushed", "not_placed", "ok")  # fmt: skip
EVAL_ID = re.compile(r"owam-bp-(?P<model>[a-z0-9]+)-(?:(?:g[0-3]|d-[a-z]+)-)?\d+$")
CLOSE_BELOW, OPEN_RAW = 0.96, 5.4  # remote_policy: a gripper row under CLOSE_BELOW * OPEN_RAW closes the gripper
EMPTY_FINGER = -0.001  # m: fingers past this when closed hold nothing (open -0.02, on a cup about -0.012..-0.004)
MIN_CLOSE = 10  # frames: a close shorter than this is a twitch
SPILL = 0.05


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def closes(command, fingers):
    """Per arm (left, right): [(start, end, holds something)] of the closes at least MIN_CLOSE frames long."""
    out = []
    for arm in range(2):
        closed = command[:, 14 + arm] < CLOSE_BELOW * OPEN_RAW
        finger = fingers[:, arm].mean(axis=1)
        runs, start = [], None
        for t, c in enumerate(np.r_[closed, False]):
            if c and start is None:
                start = t
            elif not c and start is not None:
                if t - start >= MIN_CLOSE:
                    settled = finger[start + (t - start) // 2 : t]  # past the closing motion
                    runs.append((start, t, bool(np.median(settled) < EMPTY_FINGER)))
                start = None
        out.append(runs)
    return out


def classify(r, trace, hand):
    """(primary class, tags) of an episode sidecar `r`, its gripper `trace` (npz or None), source hand."""
    if r.get("reason"):
        return "reject", []
    c, total = r["final_counts"], r["final_counts"]["total"]
    tags = []
    holds = None
    if trace is not None:
        runs = closes(trace["command"], trace["fingers"])
        src, dst = (runs[0], runs[1]) if hand == "left" else (runs[1], runs[0])
        holds = any(h for _, _, h in src)
        if not any(h for _, _, h in dst):
            tags.append("target_not_held")
        if any(sum(h for _, _, h in arm) > 1 for arm in runs):
            tags.append("regrasp")
        if any(not h for arm in runs for _, _, h in arm):
            tags.append("empty_close")
        if any(arm and arm[-1][1] >= len(trace["command"]) for arm in runs):
            tags.append("still_holding")
    if r["frames"] >= 1500 + (60 if r.get("start_joints") else 0) and not r["accepted"]:
        tags.append("timeout")
    if r.get("touched"):
        tags.append("touched")
    if r.get("distractor_moved_m", 0) > r.get("moved_limit_m", 0.02):
        tags.append("pushed")
    if r["accepted"]:
        return "ok", tags
    spilled = c["outside"] / total
    if r["task_success"] and c["target"] < 0.5 * total:
        return "repoured", tags
    if c["target"] == 0:
        if spilled > SPILL:
            return "missed", tags
        return ("no_pour" if holds or holds is None else "no_grasp"), tags
    if spilled > SPILL:
        return "big_spill", tags
    if c["source"]:
        return "leftover", tags
    if c["outside"]:
        return "small_spill", tags
    if "touched" in tags:
        return "touched", tags
    if "pushed" in tags:
        return "pushed", tags
    return "not_placed", tags


def layouts(benchmark):
    """{(env_cfg, seed): (group, [layout record])} of the benchmark."""
    out = {}
    for name, group in json.loads(Path(benchmark).read_text())["groups"].items():
        for job in group["jobs"]:
            out[job["env_cfg"], job["seed"]] = name, job["layouts"]
    return out


def family(cup_id):
    return cup_id.split("/")[2]


def scale_bin(s):
    return "<0.85" if s < 0.85 else "0.85-0.95" if s < 0.95 else "0.95-1.05" if s <= 1.05 else "1.05-1.15" \
        if s <= 1.15 else ">1.15"  # fmt: skip


def fill_bin(f):
    return f"{math.floor(f * 10) / 10:.1f}-{math.floor(f * 10) / 10 + 0.1:.1f}"


def factors(lay, scene, eval_cups=frozenset()):
    """{factor: value} of a layout record (benchmark entry or training Generalization) for the tables."""
    cups, scale = lay["cups"], lay["scale"]
    return {
        "scene": scene,
        "hand": lay["hand"],
        "source_cup": cups["source_cup"],
        "target_cup": cups["target_cup"],
        "source_split": "eval" if cups["source_cup"] in eval_cups else "train",
        "target_split": "eval" if cups["target_cup"] in eval_cups else "train",
        "source_family": family(cups["source_cup"]),
        "target_family": family(cups["target_cup"]),
        "source_scale": scale_bin(scale["source_cup"]),
        "target_scale": scale_bin(scale["target_cup"]),
        "beans": lay["beans"]["kind"],
        "fill": fill_bin(lay["beans"]["fill"]),
        "bean_count": f"{lay['beans']['count'] // 200 * 200}-{lay['beans']['count'] // 200 * 200 + 199}",
        "distractor_cup": "yes" if cups.get("distractors") else "no",
        **{f"L_{d}": lay["levels"][d] for d in DIMS},
    }


def collect(dirs, model, table, eval_cups):
    """Episode records of `model` under `dirs` (the first copy of an episode found wins)."""
    seen, rows = set(), []
    for root in dirs:
        for f in sorted(Path(root).glob("*/*/*/rep*/episode_*.json")):
            rep_dir = f.parent
            seed_dir, env_dir, eval_dir = rep_dir.parent, rep_dir.parent.parent, rep_dir.parent.parent.parent
            m = EVAL_ID.match(eval_dir.name)
            if not m or m["model"] != model:
                continue
            seed, rep, idx = int(seed_dir.name), int(rep_dir.name[3:]), int(f.stem.split("_")[1])
            key = (env_dir.name, seed, rep, idx)
            if key in seen:
                continue
            seen.add(key)
            bench, lays = table[env_dir.name, seed]
            lay = lays[idx]
            r = json.loads(f.read_text())
            npz = f.with_suffix(".npz")
            trace = np.load(npz) if npz.exists() else None
            primary, tags = classify(r, trace, lay["hand"])
            scene = env_dir.name.removesuffix("_beans_rigid_collect")
            c = r.get("final_counts") or {}
            rows.append(dict(
                bench=bench, env_cfg=env_dir.name, seed=seed, rep=rep, index=idx, set=lay["set"],
                layout=f"{env_dir.name}/{seed}/{idx}", accepted=bool(r["accepted"]),
                task_success=bool(r.get("task_success")), frames=r.get("frames"), primary=primary, tags=tags,
                counts=c, touched=r.get("touched"), distractor_moved_m=r.get("distractor_moved_m"),
                instruction=lay["instruction"]["zh"], factors=factors(lay, scene, eval_cups),
                video=str(f.relative_to(root)),
            ))  # fmt: skip
    return rows


def tally(rows):
    n, a, t = len(rows), sum(r["accepted"] for r in rows), sum(r["task_success"] for r in rows)
    return dict(n=n, strict=a, task=t, strict_rate=round(a / n, 3) if n else None, ci95=wilson(a, n),
                task_rate=round(t / n, 3) if n else None)  # fmt: skip


def classes(rows):
    c = Counter(r["primary"] for r in rows)
    return {k: c[k] for k in CLASSES if c[k]}


def logistic(rows, features, outcome="accepted", ridge=1.0):
    """{feature: odds ratio, 95% CI, episodes with it} of a ridge logistic regression of the `outcome` (accepted:
    strict, task_success) on the 0/1 `features` {name: fn(row)} (Newton steps; the intercept is not penalized)."""
    names = [n for n, fn in features.items() if any(fn(r) for r in rows)]
    if not rows or not names:
        return {}
    x = np.array([[1.0] + [float(features[n](r)) for n in names] for r in rows])
    y = np.array([float(r[outcome]) for r in rows])
    pen = np.diag([0.0] + [ridge] * len(names))
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-x @ w))
        hess = x.T @ (x * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(hess, x.T @ (y - p) - pen @ w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    out = {"intercept_rate": round(float(1 / (1 + np.exp(-w[0]))), 3), "episodes": len(rows)}
    for i, n in enumerate(names, 1):
        lo, hi = w[i] - 1.96 * se[i], w[i] + 1.96 * se[i]
        odds = np.exp([w[i], lo, hi]).round(2).tolist()
        out[n] = dict(odds=odds[0], ci95=odds[1:], n=int(x[:, i].sum()))
    return out


def level_features():
    f = {f"{d}_L{k}": (lambda r, d=d, k=k: r["factors"][f"L_{d}"] == k) for d in DIMS for k in (1, 2, 3)}
    f["scene_shenzhen"] = lambda r: r["factors"]["scene"] == "g1d_shenzhen_lab"
    f["hand_right"] = lambda r: r["factors"]["hand"] == "right"
    return f


def factor_features():
    big = {"1.05-1.15", ">1.15"}
    return {
        "scene_shenzhen": lambda r: r["factors"]["scene"] == "g1d_shenzhen_lab",
        "scene_lab_heldout": lambda r: r["factors"]["scene"] == "g1d_lab",
        "hand_right": lambda r: r["factors"]["hand"] == "right",
        "source_eval_cup": lambda r: r["factors"]["source_split"] == "eval",
        "target_eval_cup": lambda r: r["factors"]["target_split"] == "eval",
        "source_not_twin": lambda r: not r["factors"]["source_cup"].endswith("plastic_cup/00000"),
        "target_not_twin": lambda r: not r["factors"]["target_cup"].endswith("plastic_cup/00000"),
        "source_scale_out_of_train": lambda r: r["factors"]["source_scale"] in ("<0.85", ">1.15"),
        "target_scale_out_of_train": lambda r: r["factors"]["target_scale"] in ("<0.85", ">1.15"),
        "target_smaller_0.95": lambda r: r["factors"]["target_scale"] in ("<0.85", "0.85-0.95"),
        "source_larger_1.05": lambda r: r["factors"]["source_scale"] in big,
        "distractor_cup": lambda r: r["factors"]["distractor_cup"] == "yes",
        "beans_mung": lambda r: r["factors"]["beans"] == "mung",
        "beans_black": lambda r: r["factors"]["beans"] == "black",
        "fill_below_0.3": lambda r: r["factors"]["fill"] == "0.2-0.3",
        "fill_above_0.4": lambda r: r["factors"]["fill"] in ("0.4-0.5", "0.5-0.6"),
        "spatial_L2+": lambda r: r["factors"]["L_spatial"] >= 2,
        "spatial_L3": lambda r: r["factors"]["L_spatial"] == 3,
        "visual_L1+": lambda r: r["factors"]["L_visual"] >= 1,
        "visual_L2+": lambda r: r["factors"]["L_visual"] >= 2,
        "camera_L1+": lambda r: r["factors"]["L_camera"] >= 1,
        "camera_L2+": lambda r: r["factors"]["L_camera"] >= 2,
        "camera_L3": lambda r: r["factors"]["L_camera"] == 3,
        "language_L1+": lambda r: r["factors"]["L_language"] >= 1,
        "language_L3": lambda r: r["factors"]["L_language"] == 3,
    }


def summarize(rows, train, bench_expected, eval_cups=frozenset()):
    out = {"episodes": len(rows), "overall": tally(rows), "classes": classes(rows)}
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r["bench"]].append(r)
    out["bench"] = {
        b: dict(
            tally(rs), classes=classes(rs), expected=bench_expected.get(b), missing=bench_expected.get(b, 0) - len(rs)
        )
        for b, rs in sorted(by_bench.items())
    }
    tags = Counter(t for r in rows if not r["accepted"] for t in r["tags"])
    out["tags_of_failures"] = dict(tags.most_common())
    # single-dimension diagnostics, with G0 as their L0 anchor
    diag = {}
    g0 = by_bench.get("G0", [])
    for d in DIMS:
        rs = by_bench.get(f"diag_{d}", [])
        diag[d] = {"L0 (G0)": dict(tally(g0), classes=classes(g0))}
        for lv in (1, 2, 3):
            sub = [r for r in rs if r["factors"][f"L_{d}"] == lv]
            diag[d][f"L{lv}"] = dict(tally(sub), classes=classes(sub))
    out["diagnostics"] = diag
    single = g0 + [r for d in DIMS for r in by_bench.get(f"diag_{d}", [])]
    out["regression"] = {
        f"{name} ({outcome})": logistic(sub, feats, outcome)
        for outcome in ("accepted", "task_success")
        for name, sub, feats in (("diagnostics_and_G0", single, level_features()),
                                 ("all_factors", rows, factor_features()))
    }  # fmt: skip
    # layout consistency: strict passes out of the reps of each layout
    per_layout = defaultdict(list)
    for r in rows:
        per_layout[r["bench"], r["layout"]].append(r["accepted"])
    out["layout_consistency"] = {
        b: dict(Counter(f"{sum(v)}/{len(v)}" for (bb, _), v in per_layout.items() if bb == b)) for b in sorted(by_bench)
    }
    # factors pooled over all benchmarks (confounded with the sets; read with the diagnostics), with training counts
    train_counts = defaultdict(Counter)
    for g in train or []:
        for k, v in factors(g, g["scene"], eval_cups).items():
            train_counts[k][v] += 1
    tables = {}
    for k in rows[0]["factors"] if rows else ():
        groups = defaultdict(list)
        for r in rows:
            groups[r["factors"][k]].append(r)
        tables[k] = {
            str(v): dict(tally(rs), classes=classes(rs), train_episodes=train_counts[k].get(v, 0))
            for v, rs in sorted(groups.items(), key=lambda x: str(x[0]))
        }
    out["factors"] = tables
    return out


def markdown(s):
    def row(name, t):
        cls = ", ".join(f"{k} {v}" for k, v in t.get("classes", {}).items() if k != "ok")
        return (f"| {name} | {t['n']} | {t['strict_rate']} {t['ci95']} | {t['task_rate']} | "
                f"{t.get('train_episodes', '')} | {cls} |")  # fmt: skip

    head = "| | n | strict [95% CI] | task | train eps | failure classes |\n| :-- | --: | :-- | --: | --: | :-- |"
    lines = [f"# Failure analysis ({s['episodes']} episodes)", "", "## Benchmarks", "", head]
    lines += [row(b, t) for b, t in s["bench"].items()]
    lines += ["", f"All: {row('all', dict(s['overall'], classes=s['classes']))}", "", "## Diagnostics (L0 = G0)", ""]
    for d, levels in s["diagnostics"].items():
        lines += [f"### {d}", "", head] + [row(lv, t) for lv, t in levels.items()] + [""]
    for name, fit in s["regression"].items():
        lines += [f"## Logistic regression: {name} ({fit.get('episodes', 0)} episodes, baseline rate "
                  f"{fit.get('intercept_rate')})", "", "| feature | episodes | odds ratio [95% CI] |",
                  "| :-- | --: | :-- |"]  # fmt: skip
        lines += [f"| {k} | {v['n']} | {v['odds']} {v['ci95']} |" for k, v in fit.items() if isinstance(v, dict)]
        lines += [""]
    lines += ["## Layout consistency (strict passes / reps)", ""]
    lines += [f"- {b}: {dict(sorted(v.items()))}" for b, v in s["layout_consistency"].items()]
    lines += ["", "## Failure tags", "", f"{s['tags_of_failures']}", "", "## Factors (pooled over the benchmarks)", ""]
    for k, t in s["factors"].items():
        lines += [f"### {k}", "", head] + [row(v, x) for v, x in t.items()] + [""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sidecars", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--train", type=Path, help="JSON list of the training episodes' Generalization records")
    parser.add_argument("--cups", type=Path, default=ROOT / "real2sim" / "tasks" / "g1d_pour_beans_plus" / "cups.json")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = layouts(args.benchmark)
    split = json.loads(args.cups.read_text())["split"]
    eval_cups = frozenset(split["eval"])
    rows = collect(args.sidecars, args.model, table, eval_cups)
    expected = Counter()
    for group, g in json.loads(args.benchmark.read_text())["groups"].items():
        expected[group] = sum(len(j["layouts"]) for j in g["jobs"]) * args.reps
    train = json.loads(args.train.read_text()) if args.train else None
    s = summarize(rows, train, expected, eval_cups)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n")
    (args.out / "summary.md").write_text(markdown(s))
    print(json.dumps(dict(episodes=len(rows), overall=s["overall"], classes=s["classes"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
