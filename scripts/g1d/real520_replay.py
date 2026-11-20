"""Open-loop replay of real G1-D episodes (LeRobot generalization_pour_beans_eps520_v3, ShenZhen stages B-D) in their
1:1 sim twins (benchmark pour_beans_real520_sz, layouts with Generalization.real_episode).

Each layout of --seed is the twin of one real episode: cups where the real hands closed on them, start joints of its
first frame. The episode's recorded actions (absolute joint targets [L7, R7], grippers in raw Dex1 units 0 closed ..
5.4 open, 30 Hz) go to the env one per step; the sim state, finger joints, both cup poses, bean counts and the head
camera are recorded per step, and the task's own success check (the benchmark's acceptance) at the end.

Gripper: --close-below 0 sends the recorded opening as is; > 0 sends 0 (close) whenever the recorded opening / 5.4 is
under it (the real Dex1 stalls on the cup at ~4.07 with ~3.89 commanded, i.e. a weak squeeze).

--diag-frames t1,t2,..: also dump, at those steps, the beans' poses as PhysX has them, as USD has them and as Fabric
has them (what RTX draws), with the head image (ep<N>_diag.npz).

Usage (Isaac env, repo root):
    PYTHONPATH=.:XPolicyLab python scripts/g1d/real520_replay.py --task g1d_pour_beans_real520 \
        --env-cfg g1d_shenzhen_lab_beans_rigid_real520_v2 --seed 5540 \
        --episodes real2sim/tasks/g1d_pour_beans_real520/episodes --output <dir> [--layout-ids 0,1] [--close-below 0.9]
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d.remote_policy import RAW_OPEN, arm_keys, finger_joints, joint16, place_arms, to_action  # noqa: E402

ENV_KEY = "ROBODOJO_REAL520_REPLAY"
SETTLE = 10
TAIL = 30  # steps the last action is held after the recording ends (the acceptance holds over the last frames)


class Video:
    def __init__(self, path, size, fps=30):
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{size[1]}x{size[0]}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "26",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def bean_poses(fluid):
    """(physx, usd, fabric, fabric attr name) positions (N, 3), world frame, of a GranularObject's beans."""
    import omni.usd
    from pxr import Usd, UsdGeom

    n = fluid.count
    physx = fluid.view.get_transforms()[:, :3].cpu().numpy().astype(float) if fluid.view is not None else None
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    paths = [f"{fluid.prim_path}/bean_{i}" for i in range(n)]
    usd = np.array([list(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation()) for p in paths])
    fabric, name = None, ""
    try:
        import usdrt

        rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        first = rt.GetPrimAtPath(paths[0])
        names = [str(a.GetName()) for a in first.GetAttributes()]
        print("[real520_replay] fabric attrs of bean_0:", names, flush=True)
        has_hull = rt.GetPrimAtPath(paths[0] + "/hull").IsValid()
        sub_suffix = "/hull" if has_hull else ""
        if has_hull:
            hull = rt.GetPrimAtPath(paths[0] + "/hull")
            print(
                "[real520_replay] fabric attrs of bean_0/hull:",
                [str(a.GetName()) for a in hull.GetAttributes()],
                flush=True,
            )

        def get(prim, attr):
            a = prim.GetAttribute(attr)
            return a.Get() if a and a.IsValid() else None

        for p in paths[:3]:
            for q in (p, p + "/hull") if has_hull else (p,):
                pr = rt.GetPrimAtPath(q)
                print(
                    "[real520_replay]",
                    q.rsplit("/", 2)[-2:],
                    {
                        k: get(pr, k)
                        for k in ("omni:rtx:skip", "_worldVisibility", "visibility", "omni:fabric:worldMatrix")
                    },
                    flush=True,
                )
        name, rows = f"bean{sub_suffix} omni:fabric:worldMatrix", []
        for p in paths:
            v = get(rt.GetPrimAtPath(p + sub_suffix), "omni:fabric:worldMatrix")
            rows.append([np.nan] * 3 if v is None else list(v.ExtractTranslation()))
        fabric = np.array(rows, float)
        skip = [bool(get(rt.GetPrimAtPath(p + sub_suffix), "omni:rtx:skip")) for p in paths]
        print(f"[real520_replay] {sub_suffix or 'bean'} rtx:skip {sum(skip)}/{len(skip)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[real520_replay] fabric read failed:", repr(exc), flush=True)
    return physx, usd, fabric, name


def run_episode(env, plugin, cfg):
    layout_id = env.current_env_seed_map[0]
    ep = cfg["episodes"][str(layout_id)]
    out = Path(cfg["output"])
    data = np.load(ROOT / cfg["episodes_dir"] / f"ep{ep}.npz")
    state, action = data["state"], data["action"]
    act16 = np.concatenate([action[:, :14], action[:, 17:19]], axis=1).astype(np.float64)
    real16 = np.concatenate([state[:, :14], state[:, 17:19]], axis=1)
    n = len(act16)
    env.end_on_success = False
    env.step_lim = n + SETTLE + TAIL + 5
    keys = arm_keys(env)
    place_arms(env, keys, real16[0, :14], cfg["close_below"], SETTLE)
    episode = plugin.Episode(env)
    obs = env.get_obs()
    trace = dict(state=[joint16(obs, keys)], fingers=[finger_joints(env)], command=[])
    head = obs["vision"]["cam_head"]["color"]
    video = Video(out / f"ep{ep}_sim_head.mp4", head.shape[:2])
    video.write(head)
    diag_frames = set(cfg.get("diag_frames") or [])
    diag = dict(frames=[], physx=[], usd=[], fabric=[], head=[])
    for t in range(n + TAIL):
        row = act16[min(t, n - 1)]
        trace["command"].append(
            [*row[:14], *[0.0 if row[g] / RAW_OPEN < cfg["close_below"] else row[g] for g in (14, 15)]]
        )
        env.take_action(to_action(row, keys, cfg["close_below"]))
        obs = env.get_obs()
        episode.step()
        trace["state"].append(joint16(obs, keys))
        trace["fingers"].append(finger_joints(env))
        video.write(obs["vision"]["cam_head"]["color"])
        if t in diag_frames:
            physx, usd, fabric, name = bean_poses(episode.fluid)
            diag["frames"].append(t)
            diag["physx"].append(physx)
            diag["usd"].append(usd)
            diag["fabric"].append(fabric if fabric is not None else np.full_like(usd, np.nan))
            diag["head"].append(np.asarray(obs["vision"]["cam_head"]["color"])[..., :3])
            diag["fabric_attr"] = name
            du = np.nanmax(np.abs(physx - usd)) if physx is not None else -1.0
            df = np.nanmax(np.abs(physx - diag["fabric"][-1])) if physx is not None else -1.0
            print(f"[real520_replay] diag t={t} physx-usd {du:.4f} physx-fabric {df:.4f}", flush=True)
        if env.end_flag[0]:
            break
    video.close()
    if diag["frames"]:
        np.savez_compressed(out / f"ep{ep}_diag.npz", **{k: np.asarray(v) for k, v in diag.items()})
    task_success = bool(env.task_success[0])
    hist = episode.history
    np.savez_compressed(
        out / f"ep{ep}_sim.npz",
        state=np.asarray(trace["state"]),
        fingers=np.asarray(trace["fingers"]),
        command=np.asarray(trace["command"]),
        real_state=real16,
        real_action=act16,
        cup_poses=np.asarray(hist["cup_poses"]),
        counts=np.asarray(hist["counts"]),
        cup_start=episode.start,
        initial_counts=np.asarray([episode.initial_counts[k] for k in plugin.COUNT_KEYS]),
    )
    s = np.asarray(trace["state"])
    m = min(len(s), n)
    err = np.abs(s[:m, :14] - real16[:m, :14])
    summary = dict(
        episode=ep,
        layout_id=layout_id,
        layout_seed=int(env.eval_seed),
        close_below=cfg["close_below"],
        frames=n,
        replayed=len(trace["command"]),
        arm_err_max=float(err.max()),
        arm_err_mean=float(err.mean()),
        task_success=task_success,
        **episode.summary(),
    )
    summary.pop("cups", None)
    (out / f"ep{ep}_sim.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    env.success[0], env.end_flag[0] = bool(task_success), True
    print(
        f"[real520_replay] ep {ep} layout {layout_id}: frames {len(trace['command'])}/{n} arm err max {err.max():.3f}"
        f" final {summary['final_counts']} task_success {task_success}",
        flush=True,
    )


def eval_one_episode(TASK_ENV, model_client):
    cfg = json.loads(os.environ[ENV_KEY])
    run_episode(TASK_ENV, importlib.import_module(cfg["plugin"]), cfg)


def main():
    from scripts.datagen.collect import layout_files, plugin_name

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="g1d_pour_beans_real520")
    parser.add_argument("--plugin-task", default="g1d_pour_beans_plus", help="datagen plugin that counts the beans")
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", required=True, help="dir (repo-relative) of ep<N>.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-ids", default="")
    parser.add_argument("--close-below", type=float, default=0.9)
    parser.add_argument("--diag-frames", default="")
    args, remaining = parser.parse_known_args()
    files = layout_files(args.env_cfg, args.seed, args.task)
    order = [json.loads(Path(f).read_text())["Generalization"]["real_episode"] for f in files]
    wanted = [int(i) for i in args.layout_ids.split(",") if i.strip()] or range(len(files))
    args.output.mkdir(parents=True, exist_ok=True)
    todo = [i for i in wanted if not (args.output / f"ep{order[i]}_sim.json").exists()]
    if not todo:
        print("[real520_replay] nothing left")
        return
    cfg = dict(
        plugin=plugin_name(args.plugin_task),
        output=str(args.output.resolve()),
        episodes_dir=args.episodes,
        episodes={str(i): e for i, e in enumerate(order)},
        close_below=args.close_below,
        diag_frames=[int(x) for x in args.diag_frames.split(",") if x.strip()],
    )
    os.environ[ENV_KEY] = json.dumps(cfg)
    os.environ["EVAL_NUM"] = str(len(todo))
    sys.argv = [str(ROOT / "src/eval_client/main.py"), "--env_cfg_type", args.env_cfg, "--task_name", args.task]
    sys.argv += [
        "--seed",
        str(args.seed),
        "--layout_ids",
        ",".join(map(str, todo)),
        "--allow_train_layouts",
        *remaining,
    ]
    if "--device_id" not in remaining:
        sys.argv += ["--device_id", "0"]
    sys.argv += ["--policy_name", "real520_replay", "--port", "0", "--policy_module", "scripts.g1d.real520_replay"]
    sys.argv += ["--additional_info", Path(args.output).name, "--headless", "--enable_cameras"]
    sys.argv += ["--kit_args", "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
