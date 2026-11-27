"""Collect a batch of accepted expert episodes on several GPUs.

Shards a seed folder of candidate layouts into new seed folders marked as training layouts (split.json; the eval
client then refuses them without --allow_train_layouts), runs scripts/datagen/collect.py on chunks of --chunk layouts
(10) with --per-gpu processes per GPU (4 for a 32 GB 5090, which 8 run out of memory; 8 for a 96 GB H20), and stops
once --target episodes are accepted. A chunk whose process exits 0 before every layout is recorded is queued again;
a chunk that recorded nothing skips its first layout (status.json "skipped", retained on --resume). A layout the eval
client abandoned as PhysX-broken ("[PhysX] broken envs=... -> abandon seeds=[...]": no episode, exit 0) is tried again,
and skipped once that happened BROKEN_TRIES times ("physx_broken" counts them), so a unit ends with every layout
recorded or skipped.
Workers on one GPU start one at a time: each next one starts only after the previous one has taken its first sim step.
Started together on an H20, about half of them hang at the first camera capture; a worker that has not stepped within
--start-timeout is killed and its chunk queued again, up to 3 times.
Workers end an episode once it can no longer be accepted (scripts/datagen/early_abort.py) unless --no-early-abort;
collection_config.json records which ("early_abort"), and --resume of a run without it runs episodes to the end.

Usage (Isaac env python, repo root):
    python scripts/datagen/run_batch.py --task g1d_pour_beans_measuring_cups --env-cfg g1d_beans_collect \
        --candidates 9 --shards 100-107 --gpus 0-7 --target 100 --output output/datagen/office-beans-v3
    python scripts/datagen/run_batch.py --output output/datagen/office-beans-v3 --resume
Writes <output>/collection_config.json (code revision, source and asset hashes, layouts), raw/seed<N>/ episodes,
worker_<seed>_<first layout>.log and status.json. Then export with scripts/datagen/export.py and audit with
scripts/datagen/audit.py.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.global_configs import object_dir  # noqa: E402
from env.observation_manager.vision_utils import parse_heatmap_hw  # noqa: E402
from scripts.datagen import experts  # noqa: E402
from scripts.datagen.collect import SECTIONS, code_revision, layout_dir, layout_files, load_yaml, sha256  # noqa: E402
from scripts.datagen.layout_pool import LayoutPool  # noqa: E402

SOURCES = ("scripts/datagen", "scripts/g1d", "scripts/objects", "src/eval_client", "env", "task/RoboDojo", "utils")
THREADS = {k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}  # 8 Kit processes per box
STEPPED = b"env0 step: "  # src/eval_client/eval_env.py, once the scene and cameras are up
UNSTABLE_RE = re.compile(rb"Unstable nums:\s*([1-9]\d*)")
CRASH_MARKERS = (b"[Eval] unhandled exception", b"Traceback (most recent call last):")
START_TRIES = 3
ABANDON_RE = re.compile(rb"\[PhysX\] broken envs=\[[^\]]*\] -> abandon seeds=\[([0-9, ]*)\]")
BROKEN_TRIES = 2


def span(text):
    """'0-3,6' -> [0, 1, 2, 3, 6]"""
    out = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out += range(int(low), int(high or low) + 1)
    return out


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def source_hashes():
    files = [p for d in SOURCES for p in sorted((ROOT / d).rglob("*")) if p.suffix in (".py", ".yml", ".yaml", ".json")]
    files += sorted((ROOT / "env_cfg").rglob("*.yml"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.is_file() and "__pycache__" not in p.parts}


def asset_hashes(layouts, env_cfg):
    """Objects, rooms and the robot the candidate layouts load."""
    assets = ROOT / "Assets"
    files = set()
    for path in layouts:
        layout = json.loads(path.read_text())
        for section in SECTIONS:
            for key, entries in (layout.get(section) or {}).items():
                for e in entries:
                    kind = "Clutter" if e.get("type") == "cluttered" else section
                    folder = Path(object_dir(f"{kind}/{e.get('category', key)}/{e.get('category_idx', 0):05d}", assets))
                    files |= {folder / "object.usdz", folder / "object.usd", folder / "metadata.json"}
        room = (layout.get("Room") or {}).get("default")
        if room:
            files.add(assets / "Room" / room / f"{room}.usd")
    for robot in load_yaml(ROOT / "env_cfg/robot" / f"{env_cfg['config']['robot']}.yml")["robots"]:
        files |= set((assets / "Robots" / robot["robot_name"]).glob("*.urdf"))
    return {str(p.relative_to(assets)): sha256(p) for p in sorted(files) if p.is_file()}


def make_shards(args):
    files = layout_files(args.env_cfg, args.candidates, args.task)
    shards, per = span(args.shards), args.per_shard
    if len(files) < len(shards) * per:
        raise SystemExit(f"{len(files)} candidate layouts, {len(shards)} shards x {per} needs more")
    for i, seed in enumerate(shards):
        folder = layout_dir(args.env_cfg, seed)
        folder.mkdir(parents=True, exist_ok=False)
        chosen = files[i * per : (i + 1) * per]
        for j, path in enumerate(chosen):
            shutil.copy2(path, folder / f"{args.task}_{j}.json")
        write_json(
            folder / "split.json",
            dict(split="train", run=args.output.name, candidates=args.candidates, files=[p.name for p in chosen]),
        )
    candidates = layout_dir(args.env_cfg, args.candidates) / "split.json"
    if not candidates.exists():
        write_json(candidates, dict(split="train", run=args.output.name, note="sharded into seeds " + args.shards))
    return files[: len(shards) * per]


def assign(slots, active, queue, starting=()):
    """[(slot, job)] for free (gpu, k) slots in order, popping jobs off the queue until it runs out; at most one per
    GPU, and none on a GPU in starting."""
    busy, out = set(starting), []
    for slot in slots:
        if slot not in active and slot[0] not in busy and queue:
            busy.add(slot[0])
            out.append((slot, queue.pop(0)))
    return out


def worker_unstable(path, offset):
    """True only when this worker run cleanly logged at least one reset-unstable layout and no unhandled traceback."""
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    if any(marker in data for marker in CRASH_MARKERS):
        return False
    return bool(UNSTABLE_RE.search(data))


def retry_layouts(seed, ids, left, skipped, returncode, *, unstable=True):
    """Skip a reset-unstable layout only after a clean exit that proved UnStableError; crashes never skip."""
    left = list(left)
    if returncode == 0 and len(left) == len(ids):
        if not unstable:
            return []
        skipped.append(f"{seed}_{left.pop(0)}")
    return left


def abandoned(path, offset):
    """Layout ids this worker run abandoned as PhysX-broken."""
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    return sorted({int(x) for m in ABANDON_RE.finditer(data) for x in m.group(1).split(b",") if x.strip()})


def skip_broken(seed, ids, recorded, broken, skipped):
    """Count each of `ids` this run abandoned as PhysX-broken and not recorded; skip the ones abandoned BROKEN_TRIES
    times. Returns whether any was abandoned."""
    hit = [i for i in ids if i not in recorded and f"{seed}_{i}" not in skipped]
    for i in hit:
        name = f"{seed}_{i}"
        broken[name] = broken.get(name, 0) + 1
        if broken[name] >= BROKEN_TRIES:
            skipped.append(name)
    return bool(hit)


def stepped(path, offset):
    with path.open("rb") as f:
        f.seek(offset)
        return STEPPED in f.read()


def chunks(args, seed, ids):
    return [(seed, ids[i : i + args.chunk]) for i in range(0, len(ids), args.chunk)]


def dynamic_queue(todo, slots, raw, skipped, broken, stop):
    """Initialize one flock-protected LayoutPool per active shard and return persistent worker jobs:
    [(seed, ids, worker_idx, pool)]."""
    active_seeds = [s for s, ids in todo.items() if ids]
    if not active_seeds:
        return [], {}
    per_seed = max(1, (len(slots) + len(active_seeds) - 1) // len(active_seeds))
    queue, pools = [], {}
    for seed in active_seeds:
        ids = todo[seed]
        seed_dir = raw / f"seed{seed}"
        seed_skipped = [int(x.split("_", 1)[1]) for x in skipped if x.startswith(f"{seed}_")]
        seed_broken = {k.split("_", 1)[1]: v for k, v in broken.items() if k.startswith(f"{seed}_")}
        pool = LayoutPool(seed_dir / ".claim_pool.json", seed_dir, stop_file=stop, broken_tries=BROKEN_TRIES)
        pool.ensure_initialized(ids, skipped=seed_skipped, broken=seed_broken)
        pools[seed] = pool
        for w in range(min(per_seed, len(ids))):
            queue.append((seed, ids, w))
    return queue, pools


def should_retry_dynamic(hung_slot, returncode, hang_count, crash_count, max_tries=START_TRIES):
    """True when an exited dynamic worker should be re-queued (bounded separately for startup hangs and crashes)."""
    if hung_slot:
        return hang_count < max_tries
    return returncode != 0 and crash_count < max_tries


def sync_pool_state(seed, pool, skipped, broken):
    """Merge a shard LayoutPool's skipped/broken outcomes back into run_batch's status collections."""
    snap = pool.snapshot()
    for lid in snap["skipped"]:
        name = f"{seed}_{lid}"
        if name not in skipped:
            skipped.append(name)
    for lid_str, count in snap["broken"].items():
        name = f"{seed}_{lid_str}"
        broken[name] = max(broken.get(name, 0), int(count))


def accepted_count(raw):
    return sum(json.loads(p.read_text()).get("accepted", False) for p in raw.glob("seed*/episode_*.json"))


def remaining(args, seed):
    done = {p.stem for p in (args.output / "raw" / f"seed{seed}").glob("episode_*.json")}
    return [i for i in range(len(layout_files(args.env_cfg, seed, args.task))) if f"episode_{i:06d}" not in done]


def expert_version(args):
    """experts.record for the workers' code (collect.py checks it again); exits when --expert's code is not here."""
    try:
        if args.expert:
            args.plugin = experts.resolve(args.task, args.expert, args.plugin)["plugin"]
        return experts.record(args.task, args.expert, args.plugin or args.task)
    except experts.ExpertError as err:
        sys.exit(f"[run_batch] {err}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="relaunch on an existing --output with its config")
    parser.add_argument("--task")
    parser.add_argument("--env-cfg")
    parser.add_argument("--candidates", type=int, help="seed folder of candidate layouts")
    parser.add_argument("--shards", help="new seed folders for the shards, e.g. 100-107")
    parser.add_argument("--per-shard", type=int, default=50)
    parser.add_argument("--gpus", default="0", help="e.g. 0-7")
    parser.add_argument(
        "--per-gpu", type=int, default=4, help="worker processes per GPU (4 on a 32 GB 5090, 8 on a 96 GB H20)"
    )
    parser.add_argument(
        "--start-timeout", type=int, default=300, help="seconds for a worker to take its first sim step"
    )
    parser.add_argument("--chunk", type=int, default=10, help="layouts per worker process")
    parser.add_argument("--target", type=int, help="accepted episodes to collect")
    parser.add_argument("--python", default=sys.executable, help="Isaac env interpreter for the workers")
    parser.add_argument("--revision", help="code commit to record when this run dir is not a git checkout")
    parser.add_argument("--no-trajectory", action="store_true", help="sidecars only (expert feasibility filter)")
    parser.add_argument(
        "--keep-rejected-hdf5",
        action="store_true",
        help="collect.py --keep-rejected-hdf5: write full image HDF5 for rejected episodes too",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="share one flock-protected layout_id pool per shard across persistent workers instead of static --chunk",
    )
    parser.add_argument("--plugin", help="collect.py --plugin: another plugin of the task")
    parser.add_argument("--expert", help="collect.py --expert: expert version of the task (experts.json), e.g. v2")
    parser.add_argument("--no-early-abort", action="store_true", help="collect.py --no-early-abort")
    parser.add_argument("--record-depth", action="store_true", help="collect.py --record-depth")
    parser.add_argument("--depth-cameras", default="", help="collect.py --depth-cameras")
    parser.add_argument(
        "--record-affordance-2d",
        "--record-affordance",
        dest="record_affordance_2d",
        action="store_true",
        help="collect.py --record-affordance-2d",
    )
    parser.add_argument(
        "--affordance-2d-cameras",
        "--affordance-cameras",
        dest="affordance_2d_cameras",
        default="",
        help="collect.py --affordance-2d-cameras",
    )
    parser.add_argument("--affordance-2d-heatmap-size", default="", help="collect.py --affordance-2d-heatmap-size")
    parser.add_argument("--affordance-2d-top-k", type=int, default=0, help="collect.py --affordance-2d-top-k")
    args = parser.parse_args()
    if args.affordance_2d_heatmap_size:
        try:
            parse_heatmap_hw(args.affordance_2d_heatmap_size)
        except ValueError as err:
            parser.error(str(err))
    if args.affordance_2d_top_k < 0:
        parser.error(f"--affordance-2d-top-k must be >= 0, got {args.affordance_2d_top_k!r}")
    config_path = args.output / "collection_config.json"
    if args.resume:
        config = json.loads(config_path.read_text())
        for key in ("task", "env_cfg", "candidates", "shards", "per_shard", "target"):
            setattr(args, key, config[key])
        args.no_trajectory = config.get("no_trajectory", False)
        args.keep_rejected_hdf5 = args.keep_rejected_hdf5 or config.get("keep_rejected_hdf5", False)
        args.dynamic = args.dynamic or config.get("dynamic", False)
        config["keep_rejected_hdf5"] = args.keep_rejected_hdf5
        config["dynamic"] = args.dynamic
        args.plugin = config.get("plugin")
        args.no_early_abort = not config.get("early_abort", False)  # runs from before early_abort.py ran to the end
        args.expert = (config.get("expert") or {}).get("version") if config.get("expert_pinned") else None
        args.record_depth = args.record_depth or config.get("record_depth", False)
        args.depth_cameras = args.depth_cameras or config.get("depth_cameras", "")
        args.record_affordance_2d = args.record_affordance_2d or config.get("record_affordance_2d", False)
        args.affordance_2d_cameras = args.affordance_2d_cameras or config.get("affordance_2d_cameras", "")
        args.affordance_2d_heatmap_size = args.affordance_2d_heatmap_size or config.get(
            "affordance_2d_heatmap_size", ""
        )
        args.affordance_2d_top_k = args.affordance_2d_top_k or config.get("affordance_2d_top_k", 0)
        config["record_depth"] = args.record_depth
        config["depth_cameras"] = args.depth_cameras
        config["record_affordance_2d"] = args.record_affordance_2d
        config["affordance_2d_cameras"] = args.affordance_2d_cameras
        config["affordance_2d_heatmap_size"] = args.affordance_2d_heatmap_size
        config["affordance_2d_top_k"] = args.affordance_2d_top_k
        config.setdefault("resumed_unix", []).append(time.time())
        expert_version(args)
    else:
        if None in (args.task, args.env_cfg, args.candidates, args.shards, args.target):
            parser.error("--task, --env-cfg, --candidates, --shards and --target are required without --resume")
        expert = expert_version(args)
        args.output.mkdir(parents=True, exist_ok=False)
        env_cfg = load_yaml(ROOT / "env_cfg" / f"{args.env_cfg}.yml")
        config = dict(
            task=args.task,
            env_cfg=args.env_cfg,
            config_name=env_cfg["config_name"],
            sim=load_yaml(ROOT / "env_cfg/sim" / f"{env_cfg['config']['sim']}.yml"),
            fps=env_cfg["observation"]["collect_freq"],
            candidates=args.candidates,
            shards=args.shards,
            per_shard=args.per_shard,
            target=args.target,
            no_trajectory=args.no_trajectory,
            keep_rejected_hdf5=args.keep_rejected_hdf5,
            dynamic=args.dynamic,
            plugin=args.plugin,
            early_abort=not args.no_early_abort,
            expert_pinned=bool(args.expert),
            record_depth=args.record_depth,
            depth_cameras=args.depth_cameras,
            record_affordance_2d=args.record_affordance_2d,
            affordance_2d_cameras=args.affordance_2d_cameras,
            affordance_2d_heatmap_size=args.affordance_2d_heatmap_size,
            affordance_2d_top_k=args.affordance_2d_top_k,
            revision=code_revision() or (dict(commit=args.revision, dirty=None) if args.revision else None),
            started_unix=time.time(),
        )
        layouts = make_shards(args)
        config.update(source_sha256=source_hashes(), asset_sha256=asset_hashes(layouts, env_cfg))
        config["expert"] = expert
    write_json(config_path, config)
    raw, stop = args.output / "raw", args.output / "STOP"
    if stop.exists() and accepted_count(raw) < args.target:
        stop.unlink()
    status_path = args.output / "status.json"
    skipped = json.loads(status_path.read_text()).get("skipped", []) if args.resume and status_path.exists() else []
    todo = {s: [i for i in remaining(args, s) if f"{s}_{i}" not in skipped] for s in span(args.shards)}
    slots = [(gpu, k) for k in range(args.per_gpu) for gpu in span(args.gpus)]  # fill every GPU once, then again
    active, starting, hangs, exits, crashes, status = {}, {}, {}, {}, {}, {}
    broken = json.loads(status_path.read_text()).get("physx_broken", {}) if args.resume and status_path.exists() else {}
    if args.dynamic:
        queue, pools = dynamic_queue(todo, slots, raw, skipped, broken, stop)
    else:
        queue = [(s, ids, None) for s, ids_all in todo.items() for s, ids in chunks(args, s, ids_all)]
        pools = {}
    env = dict(os.environ, PYTHONPATH=".:XPolicyLab", OMNI_KIT_ACCEPT_EULA="YES", **THREADS)
    env.setdefault("OPENFLYWHEEL_NO_VIDEO", "1")
    env["OPENFLYWHEEL_REVISION"] = env["ROBODOJO_REVISION"] = json.dumps(config["revision"])
    while queue or active:
        accepted = accepted_count(raw)
        if accepted >= args.target:
            stop.touch()
            queue.clear()
        for slot, (seed, ids, worker_idx) in assign(slots, active, queue, {gpu for gpu, _ in starting}):
            gpu = slot[0]
            name = f"{seed}_w{worker_idx}" if worker_idx is not None else f"{seed}_{ids[0]}"
            log_path = args.output / f"worker_{name}.log"
            log = log_path.open("a")
            offset = log_path.stat().st_size
            cmd = [args.python, "-u", "scripts/datagen/collect.py", "--task", args.task, "--env-cfg", args.env_cfg]
            cmd += ["--seed", str(seed), "--output", str(raw / f"seed{seed}"), "--stop-file", str(stop)]
            cmd += ["--layout-ids", ",".join(map(str, ids)), "--device", "cuda:0", "--device_id", "0"]
            if worker_idx is not None:
                cmd += ["--claim-pool", str(pools[seed].path)]
            cmd += ["--no-trajectory"] if args.no_trajectory else []
            cmd += ["--keep-rejected-hdf5"] if args.keep_rejected_hdf5 else []
            cmd += ["--plugin", args.plugin] if args.plugin else []
            cmd += ["--expert", args.expert] if args.expert else []
            cmd += ["--no-early-abort"] if args.no_early_abort else []
            cmd += ["--record-depth"] if args.record_depth else []
            cmd += ["--depth-cameras", args.depth_cameras] if args.depth_cameras else []
            cmd += ["--record-affordance-2d"] if args.record_affordance_2d else []
            cmd += ["--affordance-2d-cameras", args.affordance_2d_cameras] if args.affordance_2d_cameras else []
            if args.affordance_2d_heatmap_size:
                cmd += ["--affordance-2d-heatmap-size", args.affordance_2d_heatmap_size]
            cmd += ["--affordance-2d-top-k", str(args.affordance_2d_top_k)] if args.affordance_2d_top_k else []
            # With every GPU visible each worker also keeps a CUDA context on all the others (~250 MB each), which
            # with 4 workers per GPU on 8 GPUs pushes a card past 32 GB.
            run_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            run_env["ROBODOJO_RUN_ID"] = f"{args.output.name}-seed{name}-{int(time.time())}"
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=run_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            active[slot] = (seed, ids, worker_idx, name, proc, log, log_path, offset)
            starting[slot] = (time.time(), log_path, offset)
            mode = f"dynamic worker {worker_idx}" if worker_idx is not None else f"layouts {ids[0]}-{ids[-1]}"
            print(f"[run_batch] seed {seed} {mode} on GPU {gpu}, pid {proc.pid}", flush=True)
        hung = set()
        for slot, (t0, log_path, offset) in list(starting.items()):
            seed, ids, worker_idx, name, proc, *_ = active[slot]
            if proc.poll() is not None or stepped(log_path, offset):
                del starting[slot]
            elif time.time() - t0 > args.start_timeout:
                # Kit ignores SIGTERM while it hangs.
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                del starting[slot]
                hung.add(slot)
                hangs[name] = hangs.get(name, 0) + 1
                mode = f"dynamic worker {worker_idx}" if worker_idx is not None else f"layouts {ids[0]}-{ids[-1]}"
                print(f"[run_batch] seed {seed} {mode} hung at startup, killed", flush=True)
        for slot, (seed, ids, worker_idx, name, proc, log, log_path, offset) in list(active.items()):
            if proc.poll() is not None:
                log.close()
                exits.setdefault(name, []).append(proc.returncode)
                del active[slot]
                starting.pop(slot, None)  # it can exit between the startup check above and here
                if worker_idx is not None:
                    pool = pools[seed]
                    sync_pool_state(seed, pool, skipped, broken)
                    if slot not in hung and proc.returncode != 0:
                        crashes[name] = crashes.get(name, 0) + 1
                        print(
                            f"[run_batch] seed {seed} dynamic worker {worker_idx} crashed (rc={proc.returncode})",
                            flush=True,
                        )
                    can_retry = should_retry_dynamic(
                        slot in hung, proc.returncode, hangs.get(name, 0), crashes.get(name, 0)
                    )
                    if can_retry and not stop.exists() and pool.has_pending():
                        queue.append((seed, ids, worker_idx))
                    continue
                todo_ids = set(remaining(args, seed))
                lost = [i for i in abandoned(log_path, offset) if i in ids]
                physx = skip_broken(seed, lost, {i for i in ids if i not in todo_ids}, broken, skipped)
                left = [i for i in todo_ids if i in ids and f"{seed}_{i}" not in skipped]
                left.sort()
                if physx and proc.returncode == 0:  # tried and failed, not a crash: the rest again
                    if left and not stop.exists():
                        queue.append((seed, left, None))
                    continue
                unstable = worker_unstable(log_path, offset) if proc.returncode == 0 and len(left) == len(ids) else True
                if slot not in hung and (proc.returncode != 0 or not unstable):
                    crashes[name] = crashes.get(name, 0) + 1
                    span_txt = f"{ids[0]}-{ids[-1]}"
                    print(f"[run_batch] seed {seed} layouts {span_txt} crashed (rc={proc.returncode})", flush=True)
                retry = proc.returncode == 0 or (slot in hung and hangs[name] < START_TRIES)
                if retry and not stop.exists() and left:
                    left = retry_layouts(seed, ids, left, skipped, proc.returncode, unstable=unstable)
                    if left:
                        queue.append((seed, left, None))
        status = dict(
            accepted=accepted,
            attempted=len(list(raw.glob("seed*/episode_*.json"))),
            target=args.target,
            active={name: p.pid for _, _, _, name, p, *_ in active.values()},
            queued=[f"{s}_w{w}" if w is not None else f"{s}_{ids[0]}" for s, ids, w in queue],
            starting=[active[slot][3] for slot in starting],
            startup_hangs=hangs,
            exit_codes=exits,
            crashes=crashes,
            physx_broken=broken,
            skipped=skipped,
            state="collecting" if accepted < args.target else "finishing",
            updated_unix=time.time(),
        )
        write_json(status_path, status)
        if queue or active:
            time.sleep(5)
    accepted = accepted_count(raw)
    # a resume with nothing left to run never fills `status`: keep the skipped layouts and PhysX counts it read
    final = dict(status, accepted=accepted, skipped=skipped, physx_broken=broken, state="done")
    final["updated_unix"] = time.time()
    write_json(args.output / "status.json", final)
    print(json.dumps(dict(accepted=accepted, target=args.target, exit_codes=exits)), flush=True)
    if accepted < args.target:
        sys.exit(f"Only {accepted}/{args.target} accepted: add candidate layouts and shards, then --resume")


if __name__ == "__main__":
    main()
