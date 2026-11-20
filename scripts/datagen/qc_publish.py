"""Show an exported datagen dataset in the Real2Sim platform's 数据质检 page, grouped under its dataset name.

Reads the dataset's collection_quality.json (one entry per exported episode, pointing at its raw HDF5), encodes each
episode's three camera streams (JPEG frames in vision/<cam>/colors) to H.264 MP4, and writes one run directory per
layout seed in the layout the platform scans (src/eval_client/eval_env.py; the dataset name in the policy slot):
    <out>/RoboDojo/<task>/<dataset>/<env_cfg>/<seed>_train/datagen/_result.json + episode_<layout>_cam_<cam>_success.mp4
_result.json carries "dataset" (and "round", the platform's collection round) for the page's dataset level.

Run it on the host that has the raw episodes, then copy <out>/ into an eval root of the platform (AGENTS.md):
    python scripts/datagen/qc_publish.py --dataset output/datagen/r1/lerobot/<name> --out qc_eval_result --round <id>
    scripts/platform/r2s publish qc_eval_result eval_result
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import h5py

CAMS = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
FFMPEG = shutil.which("ffmpeg", path="/usr/bin") or "ffmpeg"  # /usr/local/bin/ffmpeg on the servers has no libx264
EXTRA = ("pattern", "by", "groups", "target")  # task-specific episode fields shown with the episode, when present


def encode(job):
    raw, run_dir, layout, fps = job
    with h5py.File(raw) as f:
        for key, cam in CAMS.items():
            out = run_dir / f"episode_{layout}_cam_{cam}_success.mp4"
            if out.is_file() and out.stat().st_size:
                continue
            tmp = out.with_suffix(".tmp.mp4")
            cmd = [FFMPEG, "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg"]
            cmd += ["-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]
            cmd += ["-movflags", "+faststart", "-threads", "2", str(tmp)]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in f[f"vision/{key}/colors"]:
                p.stdin.write(bytes(frame))
            p.stdin.close()
            if p.wait():
                raise RuntimeError(f"ffmpeg failed: {raw} {cam}")
            tmp.rename(out)
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="an export.py output (with collection_quality.json)")
    ap.add_argument("--out", type=Path, required=True, help="an eval_result-shaped folder to write into")
    ap.add_argument("--round", help="the platform collection round the dataset came from, e.g. <task>-r1")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    name = a.dataset.name
    by_seed = defaultdict(list)
    for q in json.loads((a.dataset / "collection_quality.json").read_text()):
        by_seed[q["layout_seed"]].append(q)
    jobs, results = [], {}
    for seed, qs in sorted(by_seed.items()):
        run_dir = a.out / "RoboDojo" / qs[0]["task"] / name / qs[0]["env_cfg"] / f"{seed}_train" / "datagen"
        run_dir.mkdir(parents=True, exist_ok=True)
        details = {}
        for q in qs:
            ep = json.loads(Path(q["raw"]).with_suffix(".json").read_text())
            assert ep.get("accepted"), q["raw"]
            details[str(q["layout_id"])] = {
                "success": True,
                "score": 1.0,
                "layout_id": q["layout_id"],
                "instruction": ep.get("instruction"),
                "frames": ep.get("frames"),
                "acceptance_version": ep.get("acceptance_version"),
                "raw": q["raw"],
                **{k: ep[k] for k in EXTRA if k in ep},
            }
            jobs.append((q["raw"], run_dir, q["layout_id"], ep.get("fps", 30)))
        results[run_dir] = {
            "success_rate": 1.0,
            "eval_time": len(details),
            "source": "datagen (scripted expert): the accepted episodes of an exported dataset",
            "dataset": name,
            "round": a.round,
            "details": details,
        }
    with ProcessPoolExecutor(a.workers) as pool:
        for n, raw in enumerate(pool.map(encode, jobs), 1):
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {raw}", flush=True)
    for run_dir, result in results.items():  # written last: a run shows on the platform only once its videos exist
        (run_dir / "_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"{name}: {len(jobs)} episodes in {len(results)} runs under {a.out}", flush=True)


if __name__ == "__main__":
    main()
