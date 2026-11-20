"""Expert feasibility filter and freezing of the g1d_pour_beans_plus graded benchmark (README §三 分级泛化评测集).

Candidates: `pour_beans_plus_layouts.py bench --reps N --seed <C>` writes N x 325 layouts over the scenes' folders
<C> with sets.json (file, global index, set, levels); replicate 0 is the benchmark as designed, the rest spares.
The filter runs in rounds: `round` copies the candidates of replicates --rep .. --rep + --reps - 1 whose sets still
lack accepted layouts (with --factor, per set only the first factor x missing / pass rate so far) into a new seed
folder per scene (same file names), each run with `run_batch.py --no-trajectory --candidates <it>
--shards <it + 1> --per-shard <its size>`. `status` and `freeze` read the runs' sidecars (through the shards'
split.json) and status.json "skipped" (unstable at reset). Per set, its candidates in global order up to the first
one without an outcome; the first TARGET accepted are its layouts, so a set fills from replicate 0 first and never
by which episode finished first.

Groups: G0-G3 and diag_<dimension> (L1-L3, DIAG each), one seed per group and one job per scene. `freeze` copies a
group's layouts unchanged into <scene>/<--base + group index>/g1d_pour_beans_plus_<j>.json with benchmark.json (no
split.json) and writes the manifest of all groups. On the platform the groups are the subsets of one benchmark
(`pour_beans_plus@v1`: a job per group and scene, `subset` = the group; G0-G3 scored, the diagnostics not).

`score` writes one evaluation's result.json for the platform: a run per job (task, env_cfg, seed, subset = group,
level "<group> · <scene>", strict success = plus_v1 accepted, task_success, Wilson 95% interval, by_level for the
diagnostics), from the sidecars of scripts/g1d/remote_policy.py (<dir>/<env_cfg>/<seed>/rep<r>/episode_<i>.json;
starts rejected as unstable are left out) or from earlier per-group result.json files (their runs are kept). With
--episodes / --expert-episodes (sidecar dirs of the policy / of the scripted expert's repeats) it adds "layouts" (per
layout: subset, set, the five levels, the expert's outcomes), "episodes" (per episode found: layout, rep, strict and
task outcome) and "episode_records" (how many of the runs' episodes and of the expert's have a record), which the
platform uses to cut the results by any dimension's level and to list the episodes behind a number.

    python scripts/datagen/pour_beans_plus_bench.py round --candidates 1341 --rep 0 --seed 1350 --runs output/datagen/b*
    python scripts/datagen/pour_beans_plus_bench.py status --candidates 1341 --runs output/datagen/b*
    python scripts/datagen/pour_beans_plus_bench.py freeze --candidates 1341 --runs output/datagen/b* --base 1320 \
        --manifest real2sim/tasks/g1d_pour_beans_plus/benchmark.json
    python scripts/datagen/pour_beans_plus_bench.py score --eval-id <id> --sidecars <eval dir> [...] --out <dir>

A new version of the benchmark from a frozen one: `pool` puts the frozen layouts (--manifest) back into a candidate
folder under their candidate names and global indices, optionally followed by fresh spares (`bench --reps`, indices
from replicate SPARE_REP on); after changing them (e.g. `pour_beans_plus_layouts.py capsule`) the rounds and the
freeze run on it as above, so each set keeps the frozen layouts that still pass and fills from the spares.

    python scripts/datagen/pour_beans_plus_bench.py pool --candidates 1360 --manifest <v1 benchmark.json> [--spares S]
"""

import argparse
import collections
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.collect import layout_dir, sha256  # noqa: E402
from scripts.datagen.pour_beans_plus_layouts import BENCH, DIAG, DIMS, TASK  # noqa: E402
from scripts.datagen.run_batch import span  # noqa: E402

LAYOUTS = ROOT / "Assets" / "Eval_Layout" / "RoboDojo"
GROUPS = (*BENCH, *(f"diag_{d}" for d in DIMS))
PER_REP = sum(BENCH.values()) + DIAG * 3 * len(DIMS)
SPARE_REP = 100  # pool: the spares' replicates start here, after every candidate a frozen benchmark came from


def group_of(set_name):
    return set_name if set_name in BENCH else set_name.rsplit("_", 1)[0]


def target(set_name):
    return BENCH.get(set_name, DIAG)


def candidates(seed):
    """[row] of every scene's sets.json in global order; row: file, index, set, levels, scene, path."""
    rows = []
    for meta in sorted(LAYOUTS.glob(f"*/{seed}/sets.json")):
        for row in json.loads(meta.read_text())["layouts"]:
            rows.append(dict(row, scene=meta.parent.parent.name, path=meta.parent / row["file"]))
    return sorted(rows, key=lambda r: r["index"])


def outcomes(runs):
    """({(scene, file): outcome}, {scene: env_cfg}) over the filter runs."""
    out, env_cfgs = {}, {}
    for run in runs:
        config = json.loads((run / "collection_config.json").read_text())
        scene, env_cfg = config["config_name"], config["env_cfg"]
        env_cfgs[scene] = env_cfg
        status = run / "status.json"
        skipped = set(json.loads(status.read_text()).get("skipped", [])) if status.exists() else set()
        for shard in span(config["shards"]):
            names = json.loads((layout_dir(env_cfg, shard) / "split.json").read_text())["files"]
            for i, name in enumerate(names):
                sidecar = run / "raw" / f"seed{shard}" / f"episode_{i:06d}.json"
                if sidecar.exists():
                    side = json.loads(sidecar.read_text())
                    reason = (
                        "accepted"
                        if side["accepted"]
                        else side.get("reason")
                        or ("task success, not plus_v1" if side.get("task_success") else "failed")
                    )
                    out[scene, name] = dict(
                        accepted=side["accepted"], outcome=reason, sha256=side["layout_sha256"],
                        acceptance=side.get("acceptance_version"), run=run.name,
                    )  # fmt: skip
                elif f"{shard}_{i}" in skipped:
                    out[scene, name] = dict(accepted=False, outcome="unstable at reset", run=run.name)
    return out, env_cfgs


def select(rows, outs):
    """{set: dict(chosen rows, consumed, outcomes Counter, done)} by global order."""
    by_set = collections.defaultdict(list)
    for row in rows:
        by_set[row["set"]].append(row)
    result = {}
    for name, items in by_set.items():
        chosen, counts, consumed = [], collections.Counter(), 0
        for row in items:
            if len(chosen) == target(name):
                break
            out = outs.get((row["scene"], row["file"]))
            if out is None:
                break
            consumed += 1
            counts[out["outcome"]] += 1
            if out["accepted"]:
                chosen.append(dict(row, **out))
        result[name] = dict(chosen=chosen, consumed=consumed, outcomes=counts, done=len(chosen) == target(name))
    return result


def report(sel):
    for name in sorted(sel, key=lambda s: (GROUPS.index(group_of(s)), s)):
        s = sel[name]
        rate = len(s["chosen"]) / s["consumed"] if s["consumed"] else 0
        got = f"{len(s['chosen']):3d}/{target(name):3d} from {s['consumed']:3d} ({rate:.0%})"
        print(f"{name:20s} {got} {dict(s['outcomes'])}")


def cmd_round(args, rows, sel, outs):
    want = {name for name, s in sel.items() if not s["done"]}
    reps = range(args.rep, args.rep + args.reps)
    taken = [
        r
        for r in rows
        if (r["scene"], r["file"]) not in outs
        and any((LAYOUTS / r["scene"] / str(seed) / r["file"]).exists() for seed in args.skip)
    ]
    busy = {(r["scene"], r["file"]) for r in taken}
    picked = [
        r
        for r in rows
        if r["index"] // PER_REP in reps
        and r["set"] in want
        and (r["scene"], r["file"]) not in outs
        and (r["scene"], r["file"]) not in busy
    ]
    if args.factor:
        cap = {}
        for name in want:
            s = sel[name]
            rate = max(len(s["chosen"]) / s["consumed"] if s["consumed"] else 0.5, 0.15)
            cap[name] = math.ceil(args.factor * (target(name) - len(s["chosen"])) / rate)
        kept, capped = collections.Counter(r["set"] for r in taken), []
        for row in picked:
            kept[row["set"]] += 1
            if kept[row["set"]] <= cap[row["set"]]:
                capped.append(row)
        picked = capped
    per_scene = collections.Counter()
    for row in picked:
        folder = LAYOUTS / row["scene"] / str(args.seed)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], folder / row["file"])
        per_scene[row["scene"]] += 1
    print(json.dumps(dict(reps=list(reps), seed=args.seed, layouts=dict(per_scene), sets=sorted(want))))


def cmd_pool(args):
    """Candidates <scene>/<--candidates>/ from a frozen benchmark (--manifest, its layouts in their seed folders):
    each layout under its candidate name and global index, so the filter tries them first and in the order they were
    chosen in; then the --spares candidates (bench --reps) past replicate 0 (the benchmark as designed, already in the
    frozen one), their indices from replicate SPARE_REP on."""
    manifest = json.loads(args.manifest.read_text())
    rows = collections.defaultdict(list)
    for group in manifest["groups"].values():
        for job in group["jobs"]:
            for lay in job["layouts"]:
                src = LAYOUTS / job["scene"] / str(job["seed"]) / lay["file"]
                if sha256(src) != lay["sha256"]:
                    sys.exit(f"{src}: not the frozen layout")
                row = dict(file=lay["candidate"], index=lay["index"], set=lay["set"], levels=lay["levels"])
                rows[job["scene"]].append((src, row))
    for spare in candidates(args.spares) if args.spares is not None else ():
        if spare["index"] < PER_REP:
            continue
        index = SPARE_REP * PER_REP + spare["index"]
        row = dict(file=f"{TASK}_{index}.json", index=index, set=spare["set"], levels=spare["levels"])
        rows[spare["scene"]].append((spare["path"], row))
    for scene, items in rows.items():
        folder = LAYOUTS / scene / str(args.candidates)
        folder.mkdir(parents=True, exist_ok=False)
        for src, row in items:
            shutil.copy2(src, folder / row["file"])
        layouts = sorted((row for _, row in items), key=lambda r: r["index"])
        meta = dict(task=TASK, split="eval", mode="pool", frozen=manifest["candidates_seed"], spares=args.spares)
        (folder / "sets.json").write_text(json.dumps(dict(meta, layouts=layouts), indent=1) + "\n")
    print(json.dumps({scene: len(items) for scene, items in rows.items()}))


def cmd_freeze(args, rows, sel, env_cfgs):
    short = [name for name, s in sel.items() if not s["done"]]
    if short:
        counts = ", ".join(f"{n} {len(sel[n]['chosen'])}/{target(n)}" for n in short)
        sys.exit(f"sets short of accepted layouts: {counts}")
    groups = {}
    for k, group in enumerate(GROUPS):
        chosen = sorted((r for name, s in sel.items() if group_of(name) == group for r in s["chosen"]),
                        key=lambda r: r["index"])  # fmt: skip
        seed, jobs = args.base + k, []
        for scene in sorted({r["scene"] for r in chosen}):
            folder = LAYOUTS / scene / str(seed)
            folder.mkdir(parents=True, exist_ok=False)
            layouts = []
            for j, row in enumerate(r for r in chosen if r["scene"] == scene):
                if sha256(row["path"]) != row["sha256"]:
                    sys.exit(f"{scene}/{row['file']}: the filter ran a different file")
                shutil.copy2(row["path"], folder / f"{TASK}_{j}.json")
                g = json.loads(row["path"].read_text())["Generalization"]
                layouts.append(dict(
                    file=f"{TASK}_{j}.json", candidate=row["file"], index=row["index"], set=row["set"],
                    sha256=row["sha256"], levels=row["levels"], cups=g["cups"], scale=g["scale"], beans=g["beans"],
                    hand=g["hand"], instruction=g["instruction"],
                ))  # fmt: skip
            job = dict(env_cfg=env_cfgs[scene], scene=scene, seed=seed, layouts=layouts)
            if args.plugin:
                job["plugin"] = args.plugin
            (folder / "benchmark.json").write_text(json.dumps(dict(task=TASK, group=group, **job), indent=1) + "\n")
            jobs.append(job)
        sets = {name: s for name, s in sel.items() if group_of(name) == group}
        groups[group] = dict(
            seed=seed,
            jobs=jobs,
            filter={
                name: dict(
                    target=target(name),
                    consumed=s["consumed"],
                    accepted=len(s["chosen"]),
                    pass_rate=round(len(s["chosen"]) / s["consumed"], 3),
                    outcomes=dict(s["outcomes"]),
                )  # fmt: skip
                for name, s in sorted(sets.items())
            },
        )
    acceptance = sorted({r["acceptance"] for s in sel.values() for r in s["chosen"]})
    manifest = dict(task=TASK, candidates_seed=args.candidates, acceptance=acceptance, groups=groups)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({g: {j["scene"]: len(j["layouts"]) for j in v["jobs"]} for g, v in groups.items()}))


MANIFEST = ROOT / "real2sim" / "tasks" / TASK / "benchmark.json"
SCENES = {"g1d": "办公室", "g1d_lab": "实验室", "g1d_shenzhen_lab": "深圳实验室"}
SCORED = tuple(BENCH)  # the grades count towards the headline; the single-axis diagnostics do not (README §三)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def tally(rows):
    n, a, t = len(rows), sum(r["accepted"] for r in rows), sum(r["task_success"] for r in rows)
    return dict(episodes=n, accepted=a, task_success=t, success_rate=round(a / n, 4) if n else 0.0,
                task_success_rate=round(t / n, 4) if n else 0.0, ci95=wilson(a, n))  # fmt: skip


def jobs_of(manifest):
    """[(group, job)] of the frozen benchmark in its order: the platform benchmark's jobs."""
    return [(name, job) for name, g in manifest["groups"].items() for job in g["jobs"]]


def episode_files(job, dirs, repeats):
    """(rep, layout index, sidecar) of one job under `dirs`; the first copy of an episode wins."""
    for rep in range(repeats):
        for i in range(len(job["layouts"])):
            name = Path(job["env_cfg"], str(job["seed"]), f"rep{rep}", f"episode_{i:06d}.json")
            f = next((d / name for d in dirs if (d / name).exists()), None)
            if f is not None:
                yield rep, i, json.loads(f.read_text())


def sidecar_run(group, job, dirs, repeats):
    """The run of one job from the sidecars under `dirs` (the first copy of an episode wins), or None."""
    rows, rejected = [], 0
    for _, i, r in episode_files(job, dirs, repeats):
        if r.get("reason"):  # the start was rejected (objects not settled): not an episode of the policy
            rejected += 1
            continue
        rows.append(dict(layout=i, accepted=bool(r.get("accepted")), task_success=bool(r.get("task_success")),
                         levels=job["layouts"][i]["levels"], chunk_s=r.get("mean_chunk_seconds")))  # fmt: skip
    if not rows and not rejected:
        return None
    run = dict(
        layouts=len(job["layouts"]), repeats=repeats, **tally(rows), expected_episodes=repeats * len(job["layouts"])
    )
    if group.startswith("diag_"):
        axis, by = group.removeprefix("diag_"), collections.defaultdict(list)
        for r in rows:
            by[f"L{r['levels'][axis]}"].append(r)
        run["by_level"] = {k: tally(v) for k, v in sorted(by.items())}
    run["layouts_solved_any"] = len({r["layout"] for r in rows if r["accepted"]})
    run["rejected_at_start"] = rejected
    chunks = [r["chunk_s"] for r in rows if r["chunk_s"]]
    run["mean_chunk_seconds"] = round(sum(chunks) / len(chunks), 3) if chunks else None
    run["eval_time"] = len(rows)
    return run


def outcomes_of(job, dirs, repeats):
    """{(rep, layout index): (accepted, task_success)} of one job's sidecars under `dirs`, rejected starts left out."""
    return {(rep, i): (bool(r.get("accepted")), bool(r.get("task_success")))
            for rep, i, r in episode_files(job, dirs, repeats) if not r.get("reason")}  # fmt: skip


def layout_records(manifest, episodes=(), expert_episodes=(), repeats=3):
    """Per layout its labels (subset, set, five levels) and the expert's repeats on it; per episode with a sidecar under
    `episodes` its strict and task outcome. What the platform filters by any dimension's level and drills down with."""
    layouts, eps = [], []
    for group, job in jobs_of(manifest):
        where = dict(env_cfg=job["env_cfg"], seed=job["seed"])
        expert = outcomes_of(job, [Path(d) for d in expert_episodes], repeats) if expert_episodes else {}
        for i, lay in enumerate(job["layouts"]):
            rec = dict(**where, scene=job["scene"], id=i, subset=group, set=lay["set"], levels=lay["levels"])
            mine = [v for (_, j), v in expert.items() if j == i]
            if mine:
                rec["expert"] = dict(
                    episodes=len(mine), accepted=sum(a for a, _ in mine), task_success=sum(t for _, t in mine)
                )
            layouts.append(rec)
        for (rep, i), (a, t) in sorted(outcomes_of(job, [Path(d) for d in episodes], repeats).items()):
            eps.append(dict(**where, layout=i, rep=rep, accepted=a, task_success=t))
    return layouts, eps


def score(manifest, eval_id, sidecars=(), results=(), repeats=3, expert=None, episodes=(), expert_episodes=()):
    """result.json of one evaluation over the benchmark's jobs (only those with episodes get a run); with `episodes`
    (and `expert_episodes`) sidecar dirs also its per-layout labels and per-episode outcomes (layout_records)."""
    earlier = {}
    for f in results:
        for r in json.loads(Path(f).read_text())["runs"]:
            earlier.setdefault((r["env_cfg"], r["seed"]), r)
    runs = []
    for group, job in jobs_of(manifest):
        key = (job["env_cfg"], job["seed"])
        run = dict(earlier[key]) if key in earlier else sidecar_run(group, job, [Path(d) for d in sidecars], repeats)
        if run is None:
            continue
        run.update(task=TASK, env_cfg=job["env_cfg"], seed=job["seed"], subset=group,
                   level=f"{group} · {SCENES.get(job['scene'], job['scene'])}")  # fmt: skip
        if expert is not None:
            run["expert_success_rate"] = expert
        runs.append(run)
    if not runs:
        sys.exit(f"{eval_id}: no episodes")
    by_group = collections.defaultdict(list)
    for r in runs:
        by_group[r["subset"]].append(r)

    def total(rs):
        n, a = sum(r["episodes"] for r in rs), sum(r["accepted"] for r in rs)
        return n, a, sum(r["task_success"] for r in rs)

    n, a, t = total([r for g in SCORED for r in by_group.get(g, [])])
    expected = sum(len(job["layouts"]) * repeats for _, job in jobs_of(manifest))
    done = sum(r["episodes"] + r.get("rejected_at_start", 0) for r in runs)
    parts = "；".join(
        f"{g} {sum(r['accepted'] for r in rs)}/{sum(r['episodes'] for r in rs)}" for g, rs in by_group.items()
    )
    lo, hi = wilson(a, n)
    summary = f"各子集严格成功：{parts}。"
    if n:
        summary = (f"计分子集（{'、'.join(SCORED)}）严格成功 {a}/{n}（{a / n:.0%}，95% 区间 {lo:.0%}–{hi:.0%}），"
                   f"任务判据 {t}/{n}（{t / n:.0%}）。{summary}")  # fmt: skip
    rejected = sum(r.get("rejected_at_start", 0) for r in runs)
    issues = ("布局按脚本专家的 plus_v1 严格判据筛过（专家在这些布局上都成功），专家对照为 100%，分数偏乐观、尤其 G3。"
              "严格判据要求每一颗豆子都在目标杯里、两杯直立放回且全程不相碰，任务判据（task_success）一并给出。"
              f"启动时物体不稳定被拒的 {rejected} 个回合不计入。诊断子集不计入总分。")  # fmt: skip
    if done < expected:
        issues += f" 只完成 {done}/{expected} 回合（部分评测）。"
    res = dict(evaluated_at=time.strftime("%Y-%m-%d %H:%M"), summary=summary, runs=runs, issues=issues,
               rejected_at_start=rejected)  # fmt: skip
    if episodes or expert_episodes:
        res["layouts"], res["episodes"] = layout_records(manifest, episodes, expert_episodes, repeats)
        res["episode_records"] = dict(
            recorded=len(res["episodes"]),
            episodes=sum(r["episodes"] for r in runs),
            expert=sum(lay.get("expert", {}).get("episodes", 0) for lay in res["layouts"]),
        )
    return res


def cmd_score(argv):
    parser = argparse.ArgumentParser(prog="pour_beans_plus_bench.py score")
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--sidecars", type=Path, nargs="*", default=[], help="eval dirs: <env_cfg>/<seed>/rep<r>/")
    parser.add_argument("--results", type=Path, nargs="*", default=[], help="earlier per-group result.json files")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expert", type=float, help="expert_success_rate of every run (1.0: expert-filtered)")
    parser.add_argument("--episodes", type=Path, nargs="*", default=[], help="sidecar dirs for per-episode records")
    parser.add_argument("--expert-episodes", type=Path, nargs="*", default=[], help="the expert's sidecar dirs")
    parser.add_argument("--out", type=Path, required=True, help="writes <out>/real2sim/evals/<eval id>/result.json")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    res = score(manifest, args.eval_id, args.sidecars, args.results, args.repeats, args.expert,
                args.episodes, args.expert_episodes)  # fmt: skip
    dest = args.out / "real2sim" / "evals" / args.eval_id / "result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(res["summary"])


def main():
    if sys.argv[1:2] == ["score"]:
        return cmd_score(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["round", "status", "freeze", "score", "pool"])
    parser.add_argument("--candidates", type=int, required=True, help="seed folder of bench --reps candidates")
    parser.add_argument("--spares", type=int, help="pool: seed folder of bench --reps candidates appended")
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="run_batch.py --output of the filter")
    parser.add_argument("--rep", type=int, help="round: the (first) replicate to run")
    parser.add_argument("--reps", type=int, default=1, help="round: replicates from --rep on")
    parser.add_argument("--seed", type=int, help="round: new seed folder for it")
    parser.add_argument("--skip", type=int, nargs="*", default=[], help="round: seed folders already being run")
    parser.add_argument("--factor", type=float, help="round: per set at most factor x missing / pass rate so far")
    parser.add_argument("--base", type=int, help="freeze: seed folder of G0; the other groups follow")
    parser.add_argument("--manifest", type=Path, help="freeze: where to write the manifest; pool: the frozen one")
    parser.add_argument("--plugin", help="freeze: the plugin each job is judged by (scripts/platform/eval_run.py)")
    args = parser.parse_args()
    if args.command == "pool":
        return cmd_pool(args)
    rows = candidates(args.candidates)
    outs, env_cfgs = outcomes(args.runs)
    sel = select(rows, outs)
    if args.command == "round":
        cmd_round(args, rows, sel, outs)
    elif args.command == "status":
        report(sel)
    else:
        report(sel)
        cmd_freeze(args, rows, sel, env_cfgs)


if __name__ == "__main__":
    main()
