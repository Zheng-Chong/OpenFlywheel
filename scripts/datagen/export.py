"""Release accepted episodes as a LeRobot v3.0 dataset through XPolicyLab's official converter
(XPolicyLab/scripts/transform_lerobot_v30_format.py), verify it with the LeRobot loader and write the dataset card.

Run in the LeRobot env (lerobot==0.4.4 and h5py), repo root; nothing is uploaded:
    python scripts/datagen/export.py --run output/datagen/office-beans-v3 --episodes 100 --name g1d_office_beans_v3
Takes the first --episodes accepted episodes with distinct layouts (planned object poses) of the --run batches in
order (repeat --run for batches of other hosts or scenes), stages them as data/<name>/<task>/<env_cfg>/data/*.hdf5
(the converter's input; one folder per env config, converted in name order), converts into <first run>/lerobot/<name>,
then checks every episode's length, instruction, and first and last frame against its HDF5 (--shards N: N converters
on contiguous slices in parallel, joined by LeRobot's aggregate_datasets, for thousands of episodes). Fails instead of
shipping fewer or unverified episodes. Run scripts/datagen/audit.py on the result next.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "XPolicyLab/scripts/transform_lerobot_v30_format.py"
CAMERAS = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
SAME = ("task", "plugin", "acceptance_version", "physics_dt", "fps", "cameras", "state_keys", "expert")
DEFAULT_AV1_PRESET = 8  # LeRobot defaults to SVT-AV1 preset 12, which adds lone 16x16 dark squares on fast wrist cams
BLOCK_SIZE = 16
BLOCK_TOLERANCE = 35.0  # max 16x16-block mean abs diff (0-255); normal CRF 18 stays < 8, dark-block artifacts hit 40+


def valid_jpeg(buf):
    """True when an HDF5 NUL-padded byte buffer holds a complete SOI..EOI JPEG stream."""
    trimmed = bytes(buf).rstrip(b"\x00")
    return len(trimmed) >= 4 and trimmed.startswith(b"\xff\xd8") and trimmed.endswith(b"\xff\xd9")


def max_block_diff(decoded_rgb, raw_rgb, block=BLOCK_SIZE):
    """Largest mean absolute pixel difference (0-255) over `block` x `block` tiles between two HxWxC RGB frames."""
    a = np.asarray(decoded_rgb, dtype=np.float32)
    b = np.asarray(raw_rgb, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"Frame shape mismatch: {a.shape} vs {b.shape}")
    h, w, c = a.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return float(np.abs(a - b).mean())
    cropped = np.abs(a[: hb * block, : wb * block] - b[: hb * block, : wb * block])
    return float(cropped.reshape(hb, block, wb, block, c).mean(axis=(1, 3, 4)).max())


def check_raw_episodes(chosen):
    """Fail before multi-hour conversion if any chosen HDF5 is missing or has truncated camera JPEG buffers."""
    import h5py

    for path, quality in chosen:
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw trajectory HDF5: {path}")
        with h5py.File(path, "r") as f:
            for camera in quality.get("cameras") or CAMERAS.values():
                stream = f[f"vision/{camera}/colors"]
                if len(stream) < quality["frames"]:
                    raise ValueError(f"{path} {camera}: {len(stream)} frames, expected {quality['frames']}")
                for frame_idx, buf in enumerate(stream[: quality["frames"]]):
                    if not valid_jpeg(buf):
                        raise ValueError(f"{path} {camera} frame {frame_idx}: truncated or corrupt JPEG")


def patch_streaming_encoder(cls, av1_preset=DEFAULT_AV1_PRESET):
    """Configure LeRobot's StreamingVideoEncoder for lossless queue buffering and artifact-free SVT-AV1 preset."""
    if getattr(cls, "_ofw_orig_init", None) is None:
        cls._ofw_orig_init = cls.__init__
    orig_init = cls._ofw_orig_init

    def lossless_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.queue_maxsize = 0
        if getattr(self, "vcodec", None) == "libsvtav1" and av1_preset is not None:
            self.preset = av1_preset

    cls.__init__ = lossless_init


def layout_key(quality):
    return json.dumps({k: np.round(np.r_[p, q], 5).tolist() for k, (p, q) in sorted(quality["planned_poses"].items())})


def select(raws, count):
    """[(hdf5, quality)]: the first `count` accepted episodes (run, seed folder, then layout order) with distinct
    layouts, grouped by env config in name order (the converter's order)."""
    chosen, keys = [], set()
    for sidecar in (p for raw in raws for p in sorted(raw.glob("seed*/episode_*.json"))):
        quality = json.loads(sidecar.read_text())
        if not quality["accepted"]:
            continue
        if not quality["hold_frames"] <= quality["stable_frames"] <= quality["frames"]:
            raise ValueError(f"Inconsistent acceptance evidence: {sidecar}")
        key = (quality["env_cfg"], layout_key(quality))
        if key in keys:
            continue
        keys.add(key)
        chosen.append((sidecar.with_suffix(".hdf5"), quality))
        if len(chosen) == count:
            break
    else:
        raise ValueError(f"Only {len(chosen)} accepted episodes with distinct layouts in {raws}; need {count}")
    mixed = [k for k in SAME if len({json.dumps(q.get(k)) for _, q in chosen}) > 1]
    if mixed:
        raise ValueError(f"Episodes disagree on {mixed}")
    return sorted(chosen, key=lambda c: c[1]["env_cfg"])


def convert(pattern, name, episodes, output, av1_preset=DEFAULT_AV1_PRESET):
    """Run the official converter on data/<pattern>, writing <output>/<name>."""
    os.environ["HF_LEROBOT_HOME"] = str(output)  # read by lerobot at import
    spec = importlib.util.spec_from_file_location("transform_lerobot_v30_format", CONVERTER)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    # It roots data/ and env_cfg/ at its own resolved path, which is another checkout when XPolicyLab is a symlink
    # (as in server run dirs): pin both to this repo.
    converter.DATA_ROOT, converter.ENV_CFG_ROOT = ROOT / "data", ROOT / "env_cfg"
    converter.ROBOT_INFO_PATH = ROOT / "env_cfg/robot/_robot_info.json"
    # LeRobot's streaming encoder drops a frame whenever its 30-frame queue stays full for 0.1 s (it only logs), which
    # shifts every later frame of that video; with few CPUs (16 on a Slurm job, 11-10) AV1 falls behind. Unbounded
    # queues never drop, and preset 8 avoids SVT-AV1 preset 12's 16x16 dark-block wrist-camera artifacts.
    from lerobot.datasets.video_utils import StreamingVideoEncoder

    patch_streaming_encoder(StreamingVideoEncoder, av1_preset=av1_preset)
    sys.argv = [str(CONVERTER), pattern, "--repo_id", name, "--max_episode", str(episodes)]
    converter.main()


def stage(staging, task, chosen):
    """<staging>/<task>/<env_cfg>/data/episode_<i>.hdf5 links to `chosen`, the converter's input."""
    for env_cfg in sorted({q["env_cfg"] for _, q in chosen}):
        inputs = staging / task / env_cfg / "data"
        inputs.mkdir(parents=True)
        for index, (path, _) in enumerate(c for c in chosen if c[1]["env_cfg"] == env_cfg):
            (inputs / f"episode_{index:06d}.hdf5").symlink_to(path.resolve())


def shard_slices(count, shards):
    """Contiguous [start, end) slices of `count` episodes for `shards` converters; joined in order they keep
    `chosen`'s order (env configs in name order, as the single converter takes them)."""
    size = -(-count // shards)
    return [(i, min(i + size, count)) for i in range(0, count, size)]


def convert_shards(name, task, chosen, output, shards, workers, av1_preset=DEFAULT_AV1_PRESET):
    """The converter on contiguous slices of `chosen`, `workers` at a time, joined into <output>/<name> by LeRobot's
    aggregate_datasets (as rsi_release.convert). The converter's encoder queues are unbounded, so a converter's memory
    grows with its slice (5056 episodes in 48 slices on 96 cores passed 400 GB): keep slices short."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    work = output / f"{name}.shards"
    names, jobs = [], []
    for k, (start, end) in enumerate(shard_slices(len(chosen), shards)):
        names.append(f"{name}_shard{k:03d}")
        stage(ROOT / "data" / names[-1], task, chosen[start:end])
        jobs.append((f"{names[-1]}.{task}.*", names[-1], end - start, work, av1_preset))
    # Not multiprocessing.Pool: its daemonic workers cannot start LeRobot's image writer processes.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=spawn, max_tasks_per_child=1) as pool:
        list(pool.map(convert, *zip(*jobs, strict=True)))
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(names, name, roots=[work / n for n in names], aggr_root=output / name)
    for n in names:
        shutil.rmtree(ROOT / "data" / n)
    shutil.rmtree(work)


def verify(dataset_dir, name, chosen):
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from XPolicyLab.utils.process_data import decode_image_bit

    dataset = LeRobotDataset(name, root=dataset_dir)
    frames = sum(q["frames"] for _, q in chosen)
    if (dataset.num_episodes, dataset.num_frames) != (len(chosen), frames):
        raise ValueError(f"Converted {dataset.num_episodes} episodes / {dataset.num_frames} frames, expected {frames}")
    worst_block = 0.0
    for index, (path, quality) in enumerate(chosen):
        meta = dataset.meta.episodes[index]
        start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        n = quality["frames"]
        if end - start != n or list(meta["tasks"]) != [quality["instruction"]]:
            raise ValueError(f"Episode {index} ({path}) length or instruction differs")
        picks = sorted({int(i) for i in np.linspace(0, n - 1, min(n, 5)).round()} | {0, n // 2, n - 1})
        with h5py.File(path, "r") as f:
            state = np.concatenate([f[f"state/{k}s"][()] for k in quality["state_keys"]], axis=1)
            action = np.concatenate([f[f"action/{k}s"][()] for k in quality["state_keys"]], axis=1)
            jpegs = {cam: [f[f"vision/{cam}/colors"][i] for i in picks] for cam in CAMERAS.values()}
        for which, relative in enumerate(picks):
            row = start + relative
            frame = dataset[row]
            assert int(frame["episode_index"]) == index and int(frame["frame_index"]) == relative
            np.testing.assert_allclose(frame["observation.state"].numpy(), state[relative], atol=1e-6)
            np.testing.assert_allclose(frame["action"].numpy(), action[relative], atol=1e-6)
            for key, raw_cam in CAMERAS.items():
                tensor = frame[f"observation.images.{key}"]
                assert tensor.shape[0] == 3, key
                video_rgb = tensor.numpy().transpose(1, 2, 0) * 255.0
                raw_rgb = decode_image_bit(jpegs[raw_cam][which])
                bdiff = max_block_diff(video_rgb, raw_rgb)
                if bdiff > BLOCK_TOLERANCE:
                    msg = f"Episode {index} ({path}) {key} frame {relative}: 16x16 block diff {bdiff:.1f}"
                    raise ValueError(f"{msg} > {BLOCK_TOLERANCE}")
                worst_block = max(worst_block, bdiff)
    return dict(
        status="verified",
        episodes=dataset.num_episodes,
        frames=dataset.num_frames,
        fps=dataset.fps,
        max_block_diff=round(worst_block, 2),
    )


def expert_line(expert):
    if not expert:
        return "no registered version (collected before scripts/datagen/experts.json)"
    line = f"`{expert['task']}` {expert['version']} (scripts/datagen/experts.json, code of {expert['commit'][:9]})"
    return line + (f"; these files differed from it: {', '.join(expert['drift'])}" if expert["drift"] else "")


def card(name, config, chosen, verification):
    q = chosen[0][1]
    revision = config.get("revision") or {}
    scenes = {}
    for _, c in chosen:
        scenes.setdefault((c["env_cfg"], c["config_name"]), set()).add(c["layout_seed"])
    envs = "; ".join(f"`{e}` ({sum(c['env_cfg'] == e for _, c in chosen)} episodes)" for e, _ in scenes)
    layouts = "; ".join(f"seed folders {sorted(s)} of `Eval_Layout/RoboDojo/{n}`" for (_, n), s in scenes.items())
    instructions = sorted({c["instruction"] for _, c in chosen})
    instruction = (
        f'"{q["instruction"]}"'
        if len(instructions) == 1
        else f'{len(instructions)} distinct, e.g. "{q["instruction"]}"'
    )
    return f"""# {name}

Unitree G1-D (Dex1-1 grippers) `{q["task"]}` demonstrations generated in simulation by OpenFlywheel's scripted expert
(`scripts/datagen`), converted with XPolicyLab's official LeRobot v3.0 converter.

- Episodes / frames / fps: {verification["episodes"]} / {verification["frames"]} / {verification["fps"]}
- Instruction: {instruction}
- Simulation: env config {envs}, physics step {q["physics_dt"]:.6f} s. Evaluate policies trained on this data with
  the same `--env-cfg`.
- Acceptance: `{q["acceptance_version"]}` (`{q["plugin"]}`), held for {q["hold_frames"]} frames.
- Expert: {expert_line(q.get("expert"))}.
- Layouts: {layouts}, marked as training layouts; do not evaluate on them.
- Code: {revision.get("commit", "unknown")} (uncommitted changes: {revision.get("dirty", "unknown")}).

## Features

- `observation.state`, `action` (16): left arm joints 1-7 (rad), left gripper, right arm joints 1-7, right gripper
  (converter motor names `left_joint_0..7`, `right_joint_0..7`). Gripper state is the measured finger opening and the
  gripper action the command, both 0 = closed .. 1 = open; arm actions are joint position targets.
- `observation.images.cam_high` (head, left eye), `cam_left_wrist`, `cam_right_wrist`: 640 x 480 RGB video.

The full data contract, including the mapping to the real robot, is `scripts/datagen/CONTRACT.md` in OpenFlywheel.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", type=Path, action="append", required=True, help="run_batch.py output (raw/, collection_config.json)"
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--name", required=True, help="dataset name (LeRobot repo id)")
    parser.add_argument("--output", type=Path, help="default <first run>/lerobot; the dataset goes to <output>/<name>")
    parser.add_argument("--shards", type=int, default=1, help="slices converted separately, then joined")
    parser.add_argument("--workers", type=int, help="converters at a time (default --shards; each AV1-encodes)")
    parser.add_argument(
        "--av1-preset", type=int, default=DEFAULT_AV1_PRESET, help="SVT-AV1 preset (default 8; avoids preset 12 blocks)"
    )
    args = parser.parse_args()
    output = (args.output or args.run[0] / "lerobot").resolve()
    dataset_dir = output / args.name
    staging = ROOT / "data" / args.name
    if args.episodes < 1 or dataset_dir.exists() or staging.exists():
        parser.error(f"--episodes must be positive and {dataset_dir}, {staging} must not exist")
    configs = [json.loads((run / "collection_config.json").read_text()) for run in args.run]
    if len({json.dumps(c.get("revision")) for c in configs}) > 1:
        parser.error("the runs were collected with different code revisions")
    config = configs[0] if len(configs) == 1 else dict(revision=configs[0].get("revision"), runs=configs)
    chosen = select([run / "raw" for run in args.run], args.episodes)
    check_raw_episodes(chosen)
    task = chosen[0][1]["task"]
    if args.shards == 1:
        stage(staging, task, chosen)
        convert(f"{args.name}.{task}.*", args.name, args.episodes, output, av1_preset=args.av1_preset)
    else:
        convert_shards(
            args.name, task, chosen, output, args.shards, args.workers or args.shards, av1_preset=args.av1_preset
        )
    verification = verify(dataset_dir, args.name, chosen)
    shutil.rmtree(staging, ignore_errors=True)
    (dataset_dir / "collection_config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = [dict(raw=str(path.resolve()), **quality) for path, quality in chosen]
    (dataset_dir / "collection_quality.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (dataset_dir / "export_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (dataset_dir / "README.md").write_text(card(args.name, config, chosen, verification))
    print(f"Verified {verification['episodes']} episodes / {verification['frames']} frames: {dataset_dir}")


if __name__ == "__main__":
    main()
