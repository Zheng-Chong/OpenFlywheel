"""Show an exported datagen dataset's episodes and videos on the platform's 数据看板, grouped under its dataset name.

Reads the dataset's collection_quality.json (one entry per exported episode, pointing at its raw HDF5), encodes each
episode's three camera streams (JPEG frames in vision/<cam>/colors, or LeRobot v3 exported videos as fallback when raw
HDF5s are absent/inaccessible) to H.264 MP4, and writes one run directory per (env_cfg, layout_seed) in the layout the
platform scans (src/eval_client/eval_env.py; the dataset name in the policy slot):
    <out>/RoboDojo/<task>/<dataset>/<env_cfg>/<seed>_train/datagen/_result.json + episode_<layout>_cam_<cam>_success.mp4
_result.json carries "dataset" (and "round", the platform's collection round) for the page's dataset level.

Pass --limit N to uniformly sample up to N episodes across scenes and layout seeds (useful for quick dashboard previews
of multi-thousand-episode datasets). Omitting --limit preserves the full-dataset behavior.

Run it on a host with the raw episodes or the exported LeRobot v3 dataset, then publish <out>/ to the platform:
    python scripts/datagen/qc_publish.py --dataset output/datagen/r1/lerobot/<name> --out qc_eval_result --round <id>
    scripts/platform/r2s publish qc_eval_result eval_result
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from scripts.datagen import twitch
except ImportError:
    _DATAGEN_DIR = Path(__file__).resolve().parent
    if str(_DATAGEN_DIR) not in sys.path:
        sys.path.insert(0, str(_DATAGEN_DIR))
    import twitch  # noqa: E402

CAMS = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
LEROBOT_CAMS = {
    "observation.images.cam_high": "head",
    "observation.images.cam_head": "head",
    "observation.images.cam_left_wrist": "left_wrist",
    "observation.images.cam_right_wrist": "right_wrist",
}
FFMPEG = shutil.which("ffmpeg", path="/usr/bin") or "ffmpeg"  # /usr/local/bin/ffmpeg on the servers has no libx264
EXTRA = ("pattern", "by", "groups", "target")  # task-specific episode fields shown with the episode, when present


def _has_h5py() -> bool:
    return importlib.util.find_spec("h5py") is not None


def _is_readable_file(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def _raw_path(q: dict) -> Path | None:
    raw = q.get("raw")
    if not raw:
        return None
    p = Path(raw)
    return p if p.name else None


def _uniform_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return [i * (total - 1) // (count - 1) for i in range(count)]


def _allocate(capacities: list[int], total: int) -> list[int]:
    """Distribute `total` slots across buckets up to `capacities`, giving every non-empty bucket >= 1 when possible."""
    if total <= 0 or not capacities:
        return [0] * len(capacities)
    cap_sum = sum(capacities)
    if total >= cap_sum:
        return list(capacities)
    non_empty = [i for i, c in enumerate(capacities) if c > 0]
    alloc = [0] * len(capacities)
    if total <= len(non_empty):
        for pos in _uniform_indices(len(non_empty), total):
            alloc[non_empty[pos]] = 1
        return alloc
    for i in non_empty:
        alloc[i] = 1
    rem = total - len(non_empty)
    spare = [capacities[i] - alloc[i] for i in range(len(capacities))]
    spare_sum = sum(spare)
    if rem > 0 and spare_sum > 0:
        extra = [min(spare[i], (rem * spare[i]) // spare_sum) for i in range(len(capacities))]
        for i in range(len(capacities)):
            alloc[i] += extra[i]
        rem -= sum(extra)
    if rem > 0:
        candidates = [i for i in range(len(capacities)) if alloc[i] < capacities[i]]
        while rem > 0 and candidates:
            step_pick = _uniform_indices(len(candidates), min(rem, len(candidates)))
            for pos in step_pick:
                alloc[candidates[pos]] += 1
                rem -= 1
            candidates = [i for i in range(len(capacities)) if alloc[i] < capacities[i]]
    return alloc


def sample_episodes(entries: list[tuple[int, dict]], limit: int | None) -> list[tuple[int, dict]]:
    """Uniformly sample up to `limit` (episode_index, quality_row) pairs across scenes (env_cfg) and layout seeds."""
    if limit is None:
        return list(entries)
    if limit <= 0:
        raise ValueError(f"--limit must be a positive integer, got {limit}")
    if limit >= len(entries):
        return list(entries)

    by_scene: dict[str, dict[int, list[tuple[int, dict]]]] = defaultdict(lambda: defaultdict(list))
    for ep_idx, q in entries:
        by_scene[q.get("env_cfg") or "default"][q["layout_seed"]].append((ep_idx, q))

    scenes = sorted(by_scene.keys())
    scene_caps = [sum(len(eps) for eps in by_scene[s].values()) for s in scenes]
    scene_quotas = _allocate(scene_caps, limit)

    selected: list[tuple[int, dict]] = []
    for scene, s_quota in zip(scenes, scene_quotas):
        if s_quota <= 0:
            continue
        seeds = sorted(by_scene[scene].keys())
        seed_lists = [sorted(by_scene[scene][sd], key=lambda item: (item[1]["layout_id"], item[0])) for sd in seeds]
        seed_quotas = _allocate([len(eps) for eps in seed_lists], s_quota)
        for eps, k in zip(seed_lists, seed_quotas):
            for pos in _uniform_indices(len(eps), k):
                selected.append(eps[pos])
    return selected


def load_lerobot_episode_videos(
    dataset_dir: Path, episode_indices: set[int]
) -> dict[int, dict[str, tuple[Path, float, float]]]:
    """Load {episode_index: {cam_name: (mp4_path, from_ts, to_ts)}} from a LeRobot v3 dataset."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing LeRobot v3 metadata for fallback: {info_path}")
    info = json.loads(info_path.read_text())
    video_path_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    video_keys: dict[str, str] = {}
    for feat_key, feat in (info.get("features") or {}).items():
        if feat.get("dtype") == "video" and feat_key in LEROBOT_CAMS:
            video_keys[feat_key] = LEROBOT_CAMS[feat_key]
    if not video_keys:
        raise RuntimeError(f"no recognized camera video features in {info_path}")

    import pyarrow.parquet as pq

    ep_files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"no episode parquet files under {dataset_dir / 'meta' / 'episodes'}")

    needed_cols = ["episode_index"]
    for vkey in video_keys:
        needed_cols.extend(
            [
                f"videos/{vkey}/chunk_index",
                f"videos/{vkey}/file_index",
                f"videos/{vkey}/from_timestamp",
                f"videos/{vkey}/to_timestamp",
            ]
        )

    found: dict[int, dict[str, tuple[Path, float, float]]] = {}
    for pq_path in ep_files:
        table = pq.read_table(pq_path, columns=needed_cols)
        ep_col = table["episode_index"].to_pylist()
        for row_idx, ep_idx in enumerate(ep_col):
            if ep_idx not in episode_indices:
                continue
            cams: dict[str, tuple[Path, float, float]] = {}
            for vkey, cam_name in video_keys.items():
                chunk_idx = int(table[f"videos/{vkey}/chunk_index"][row_idx].as_py())
                file_idx = int(table[f"videos/{vkey}/file_index"][row_idx].as_py())
                from_ts = float(table[f"videos/{vkey}/from_timestamp"][row_idx].as_py())
                to_ts = float(table[f"videos/{vkey}/to_timestamp"][row_idx].as_py())
                src = dataset_dir / video_path_tpl.format(video_key=vkey, chunk_index=chunk_idx, file_index=file_idx)
                cams[cam_name] = (src, from_ts, to_ts)
            found[ep_idx] = cams
            if len(found) == len(episode_indices):
                return found
    return found


def _encode_hdf5(raw: str, run_dir: Path, layout: int, fps: int) -> str:
    import h5py

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
            try:
                for frame in f[f"vision/{key}/colors"]:
                    p.stdin.write(bytes(frame))
            except BrokenPipeError:
                pass
            finally:
                try:
                    p.stdin.close()
                except OSError:
                    pass
            if p.wait():
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg failed: {raw} {cam}")
            tmp.rename(out)
    return raw


def _encode_lerobot(label: str, cam_slices: dict[str, tuple[Path, float, float]], run_dir: Path, layout: int) -> str:
    for cam, (src_mp4, from_ts, to_ts) in cam_slices.items():
        out = run_dir / f"episode_{layout}_cam_{cam}_success.mp4"
        if out.is_file() and out.stat().st_size:
            continue
        if not _is_readable_file(src_mp4):
            raise FileNotFoundError(f"missing LeRobot video file: {src_mp4}")
        duration = to_ts - from_ts
        if duration <= 0:
            raise ValueError(f"invalid video slice duration: {label} {cam} ({from_ts}..{to_ts})")
        tmp = out.with_suffix(".tmp.mp4")
        cmd = [
            FFMPEG,
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{from_ts:.6f}",
            "-i",
            str(src_mp4),
            "-t",
            f"{duration:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            "2",
            str(tmp),
        ]
        if subprocess.run(cmd, check=False).returncode:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed: {label} {cam} ({src_mp4} {from_ts:.3f}..{to_ts:.3f})")
        tmp.rename(out)
    return label


def encode(job):
    kind = job[0]
    if kind == "hdf5":
        _, raw, run_dir, layout, fps = job
        return _encode_hdf5(raw, run_dir, layout, fps)
    if kind == "lerobot":
        _, label, cam_slices, run_dir, layout = job
        return _encode_lerobot(label, cam_slices, run_dir, layout)
    raise ValueError(f"unknown encode job kind: {kind!r}")


def _detect_twitch_excluded(
    dataset: Path,
    entries: list[tuple[int, dict]],
    *,
    window_s: float = twitch.TWITCH_WINDOW_S,
    rev_speed: float = twitch.TWITCH_REV_SPEED,
    max_rev_burst: int | None = twitch.TWITCH_REV_BURST,
    max_state_speed: float | None = twitch.TWITCH_STATE_SPEED_MAX,
    max_track_err: float | None = twitch.TWITCH_TRACK_ERR_MAX,
) -> set[int]:
    """Identify twitching episode indices from LeRobot v3 parquet when available, else sidecars / raw HDF5s."""
    if any(dataset.glob("data/chunk-*/file-*.parquet")):
        report = twitch.scan_lerobot(
            dataset,
            window_s=window_s,
            rev_speed=rev_speed,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        )
        if report["episodes"] != len(entries):
            raise ValueError(
                f"Episode count mismatch in {dataset}: parquet has {report['episodes']} episodes "
                f"vs collection_quality.json has {len(entries)}"
            )
        return set(report["excluded_episode_indices"])

    excluded: set[int] = set()
    has_h5py = _has_h5py()
    default_params = window_s == twitch.TWITCH_WINDOW_S and rev_speed == twitch.TWITCH_REV_SPEED
    for ep_idx, q in entries:
        if default_params and twitch.has_twitch_metrics(q.get("motion")):
            if twitch.twitch_reasons(
                q["motion"],
                max_rev_burst=max_rev_burst,
                max_state_speed=max_state_speed,
                max_track_err=max_track_err,
            ):
                excluded.add(ep_idx)
            continue
        raw_path = _raw_path(q)
        if has_h5py and raw_path is not None and _is_readable_file(raw_path):
            _, reasons = twitch.check_hdf5_episode(
                raw_path,
                q,
                window_s=window_s,
                rev_speed=rev_speed,
                max_rev_burst=max_rev_burst,
                max_state_speed=max_state_speed,
                max_track_err=max_track_err,
            )
            if reasons:
                excluded.add(ep_idx)
    return excluded


def run(
    dataset: Path,
    out: Path,
    round_id: str | None = None,
    workers: int = 12,
    limit: int | None = None,
    *,
    exclude: set[int] | Path | None = None,
    filter_twitch: bool = False,
    window_s: float = twitch.TWITCH_WINDOW_S,
    rev_speed: float = twitch.TWITCH_REV_SPEED,
    max_rev_burst: int | None = twitch.TWITCH_REV_BURST,
    max_state_speed: float | None = twitch.TWITCH_STATE_SPEED_MAX,
    max_track_err: float | None = twitch.TWITCH_TRACK_ERR_MAX,
) -> tuple[int, int]:
    name = dataset.name
    all_entries = list(enumerate(json.loads((dataset / "collection_quality.json").read_text())))
    excluded: set[int] = set()
    if exclude is not None:
        excluded |= twitch.load_excluded_indices(exclude) if isinstance(exclude, Path | str) else set(exclude)
    if filter_twitch:
        excluded |= _detect_twitch_excluded(
            dataset,
            all_entries,
            window_s=window_s,
            rev_speed=rev_speed,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        )
    if excluded:
        before = len(all_entries)
        all_entries = [(ep_idx, q) for ep_idx, q in all_entries if ep_idx not in excluded]
        print(
            f"{name}: excluded {before - len(all_entries)}/{before} episodes ({len(all_entries)} clean remaining)",
            flush=True,
        )
    selected = sample_episodes(all_entries, limit)

    has_h5py = _has_h5py()
    fallback_indices = {
        ep_idx
        for ep_idx, q in selected
        if not (has_h5py and (rp := _raw_path(q)) is not None and _is_readable_file(rp))
    }
    lerobot_videos = load_lerobot_episode_videos(dataset, fallback_indices) if fallback_indices else {}

    by_group: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for ep_idx, q in selected:
        by_group[(q.get("env_cfg") or "default", q["layout_seed"])].append((ep_idx, q))

    jobs, results = [], {}
    for (env_cfg, seed), qs in sorted(by_group.items()):
        run_dir = out / "RoboDojo" / qs[0][1]["task"] / name / env_cfg / f"{seed}_train" / "datagen"
        run_dir.mkdir(parents=True, exist_ok=True)
        details = {}
        for ep_idx, q in qs:
            raw_path = _raw_path(q)
            if raw_path is not None and _is_readable_file(raw_path.with_suffix(".json")):
                ep = json.loads(raw_path.with_suffix(".json").read_text())
                assert ep.get("accepted"), q["raw"]
            else:
                ep = q
                if "accepted" in ep:
                    assert ep["accepted"], q.get("raw") or f"episode_{ep_idx}"
            details[str(q["layout_id"])] = {
                "success": True,
                "score": 1.0,
                "layout_id": q["layout_id"],
                "instruction": ep.get("instruction"),
                "frames": ep.get("frames"),
                "acceptance_version": ep.get("acceptance_version"),
                "raw": q.get("raw"),
                **{k: ep[k] for k in EXTRA if k in ep},
            }
            if has_h5py and raw_path is not None and _is_readable_file(raw_path):
                jobs.append(("hdf5", str(raw_path), run_dir, q["layout_id"], ep.get("fps", 30)))
            else:
                cam_slices = lerobot_videos.get(ep_idx)
                if not cam_slices:
                    raise FileNotFoundError(
                        f"episode {ep_idx} raw HDF5 unavailable ({q.get('raw')}) and not found in LeRobot metadata"
                    )
                jobs.append(("lerobot", f"{name}#ep{ep_idx}", cam_slices, run_dir, q["layout_id"]))
        results[run_dir] = {
            "success_rate": 1.0,
            "eval_time": len(details),
            "source": "datagen (scripted expert): the accepted episodes of an exported dataset",
            "dataset": name,
            "round": round_id,
            "details": details,
        }

    if workers <= 1 or len(jobs) <= 1:
        for n, job in enumerate(jobs, 1):
            label = encode(job)
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {label}", flush=True)
    else:
        with ProcessPoolExecutor(workers) as pool:
            for n, label in enumerate(pool.map(encode, jobs), 1):
                if n % 25 == 0 or n == len(jobs):
                    print(f"{n}/{len(jobs)} {label}", flush=True)
    for run_dir, result in results.items():  # written last: a run shows on the platform only once its videos exist
        (run_dir / "_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"{name}: {len(jobs)} episodes in {len(results)} runs under {out}", flush=True)
    return len(jobs), len(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="an export.py output (with collection_quality.json)")
    ap.add_argument("--out", type=Path, required=True, help="an eval_result-shaped folder to write into")
    ap.add_argument("--round", help="the platform collection round the dataset came from, e.g. <task>-r1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="uniformly sample up to N episodes across scenes and seeds (default: all)",
    )
    ap.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="JSON exclusion list (e.g. from twitch.py --out) of episode indices to drop before sampling",
    )
    ap.add_argument(
        "--filter-twitch",
        action="store_true",
        help="detect and drop twitching episodes before sampling",
    )
    twitch.add_twitch_args(ap)
    a = ap.parse_args()
    if a.limit is not None and a.limit <= 0:
        ap.error("--limit must be a positive integer")
    run(
        a.dataset,
        a.out,
        round_id=a.round,
        workers=a.workers,
        limit=a.limit,
        exclude=a.exclude,
        filter_twitch=a.filter_twitch,
        window_s=a.window_s,
        rev_speed=a.rev_speed,
        max_rev_burst=a.max_rev_burst,
        max_state_speed=a.max_state_speed,
        max_track_err=a.max_track_err,
    )


if __name__ == "__main__":
    main()
