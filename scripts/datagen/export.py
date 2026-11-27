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
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.datagen import twitch  # noqa: E402

CONVERTER = ROOT / "XPolicyLab/scripts/transform_lerobot_v30_format.py"
CAMERAS = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
RAW_TO_LEROBOT_CAMERA = {
    "cam_head": "cam_high",
    "cam_head_right": "cam_high_right",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
STANDARD_STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
BODY_STATE_DIMS = {"waist_joint_state": 2, "lift_joint_state": 1}
BODY_MOTOR_NAMES = {
    "waist_joint_state": ("waist_yaw", "waist_pitch"),
    "lift_joint_state": ("lift_height",),
}
NATIVE_WHOLEBODY_STATE_KEYS = (*STANDARD_STATE_KEYS, "waist_joint_state", "lift_joint_state")
SAME = ("task", "plugin", "acceptance_version", "physics_dt", "fps", "cameras", "state_keys", "expert")
DEFAULT_AV1_PRESET = 8  # LeRobot defaults to SVT-AV1 preset 12, which adds lone 16x16 dark squares on fast wrist cams
BLOCK_SIZE = 16
BLOCK_TOLERANCE = 35.0  # max 16x16-block mean abs diff (0-255); normal CRF 18 stays < 8, dark-block artifacts hit 40+


def camera_map(cameras=None):
    """Map LeRobot image feature key -> raw HDF5 vision/<cam>/colors key for `cameras`."""
    if not cameras:
        return dict(CAMERAS)
    mapping = {}
    for raw_cam in cameras:
        key = RAW_TO_LEROBOT_CAMERA.get(raw_cam, raw_cam)
        if key in mapping:
            raise ValueError(f"Duplicate LeRobot camera key {key!r} for raw cameras {cameras}")
        mapping[key] = raw_cam
    return mapping


def patch_converter(converter, cameras=None, state_keys=None):
    """Adapt XPolicyLab's converter module in-place for dynamic camera sets and optional body state keys."""
    cam_map = camera_map(cameras)
    converter.CAMERA_CANDIDATES = {key: [("vision", raw_cam, "colors")] for key, raw_cam in cam_map.items()}
    if not state_keys:
        return
    extra_keys = [k for k in state_keys if k not in STANDARD_STATE_KEYS]
    if not extra_keys:
        return
    unknown = [k for k in extra_keys if k not in BODY_STATE_DIMS]
    if unknown:
        raise ValueError(f"Unsupported extra state_keys for LeRobot export: {unknown}")
    ordered_keys = list(state_keys)
    # Newer XPolicyLab converters natively handle 3D wholebody (waist_joint_state + lift_joint_state) via `body_dim=3`.
    # Do not overwrite their (data, body_dim=0) / (per_arm_dims, body_dim=0) hooks when native support matches.
    native_body_dim = "body_dim" in inspect.signature(converter._extract_qpos).parameters
    if native_body_dim and tuple(ordered_keys) == NATIVE_WHOLEBODY_STATE_KEYS:
        return

    def _extract(group):
        def _fn(data, body_dim=0):
            parts = [(f"{k}s", converter._get_nested(data, group, f"{k}s")) for k in ordered_keys]
            arr = converter._concat_state_parts(parts, group)
            if arr is not None:
                return arr
            raise ValueError(f"Cannot find {group} data")

        return _fn

    orig_dims = converter._dims_from_robot_action_info
    extra_dims = [BODY_STATE_DIMS[k] for k in extra_keys]
    extra_motor_names = [name for k in extra_keys for name in BODY_MOTOR_NAMES[k]]

    def _dims_with_body(robot_action_dim_info):
        return orig_dims(robot_action_dim_info) + list(extra_dims)

    orig_motors = converter._build_motor_names_from_dims

    def _motors_with_body(per_arm_dims, body_dim=0):
        arm_count = len(per_arm_dims) - len(extra_dims)
        if arm_count <= 0:
            return orig_motors(per_arm_dims)
        motors = orig_motors(per_arm_dims[:arm_count])
        motors.extend(extra_motor_names)
        return motors

    converter._extract_qpos = _extract("state")
    converter._extract_action = _extract("action")
    if not native_body_dim:
        converter._dims_from_robot_action_info = _dims_with_body
        converter._build_motor_names_from_dims = _motors_with_body


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


def _check_depth_hdf5(h5file, path, quality):
    """Validate optional per-camera metric depth (and camera matrix) datasets under `vision/<camera>/`."""
    depth_meta = quality.get("depth")
    if not depth_meta:
        return
    n = int(quality["frames"])
    cams = list(depth_meta.get("cameras") or ())
    if not cams:
        raise ValueError(f"{path}: depth metadata declares no cameras")
    shapes = depth_meta.get("shapes") or {}
    for cam in cams:
        key = f"vision/{cam}/depths"
        if key not in h5file:
            raise ValueError(f"{path}: missing depth dataset {key}")
        ds = h5file[key]
        expected_hw = tuple(int(x) for x in shapes[cam]) if cam in shapes else None
        if len(ds) < n or ds.ndim != 3 or (expected_hw is not None and tuple(ds.shape[1:]) != expected_hw):
            raise ValueError(f"{path} {key}: shape {ds.shape}, expected (>={n}, {expected_hw})")
        if ds.dtype != np.float32:
            raise ValueError(f"{path} {key}: expected float32 dtype, got {ds.dtype}")
        unit = ds.attrs.get("unit")
        if unit != "m":
            raise ValueError(f"{path} {key}: expected unit 'm', got {unit!r}")
    for cam in depth_meta.get("has_intrinsics") or ():
        k_key = f"vision/{cam}/intrinsic_matrix"
        if k_key not in h5file or tuple(h5file[k_key].shape) != (3, 3):
            raise ValueError(f"{path}: missing or malformed {k_key}")
    for cam in depth_meta.get("has_extrinsics") or ():
        ext_key = f"vision/{cam}/extrinsic_matrices"
        if ext_key not in h5file or len(h5file[ext_key]) < n or tuple(h5file[ext_key].shape[1:]) != (4, 4):
            raise ValueError(f"{path}: missing or malformed {ext_key}")


def _check_affordance_2d_hdf5(h5file, path, quality):
    """Validate optional per-frame 2D affordance HDF5 datasets under `affordance_2d/<camera>/`."""
    aff2d = quality.get("affordance_2d")
    if not aff2d:
        return
    n = int(quality["frames"])
    cams = list(aff2d.get("cameras") or ())
    if not cams:
        raise ValueError(f"{path}: affordance_2d metadata declares no cameras")
    hm_h, hm_w = int(aff2d["heatmap_hw"][0]), int(aff2d["heatmap_hw"][1])
    top_k = int(aff2d["top_k"])
    for cam in cams:
        grp_key = f"affordance_2d/{cam}"
        if grp_key not in h5file:
            raise ValueError(f"{path}: missing affordance_2d group {grp_key}")
        grp = h5file[grp_key]
        for sub, expected_tail in (
            ("object_bbox_xyxy_norm", (4,)),
            ("affordance_bbox_xyxy_norm", (4,)),
            ("keypoints_xy_score", (top_k, 3)),
            ("selected_grasp_xy_norm", (2,)),
            ("heatmap_uint8", (hm_h, hm_w)),
        ):
            if sub not in grp:
                raise ValueError(f"{path}: missing dataset {grp_key}/{sub}")
            ds = grp[sub]
            if len(ds) < n or tuple(ds.shape[1:]) != expected_tail:
                raise ValueError(f"{path} {grp_key}/{sub}: shape {ds.shape}, expected (>={n}, *{expected_tail})")
            arr = ds[:n]
            if sub == "heatmap_uint8":
                if arr.dtype != np.uint8:
                    raise ValueError(f"{path} {grp_key}/{sub}: expected uint8 dtype, got {arr.dtype}")
                continue
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(f"{path} {grp_key}/{sub}: expected floating dtype, got {arr.dtype}")
            if np.isinf(arr).any():
                raise ValueError(f"{path} {grp_key}/{sub}: contains Inf")
            fin = np.isfinite(arr)
            all_fin = fin.all(axis=-1)
            any_fin = fin.any(axis=-1)
            if np.any(any_fin & ~all_fin):
                raise ValueError(f"{path} {grp_key}/{sub}: partially NaN row")
            if sub in ("object_bbox_xyxy_norm", "affordance_bbox_xyxy_norm"):
                if np.any(all_fin):
                    v_arr = arr[all_fin]
                    if np.any(v_arr < -1e-5) or np.any(v_arr > 1.0 + 1e-5):
                        raise ValueError(f"{path} {grp_key}/{sub}: normalized bbox outside [0, 1]")
                    if np.any(v_arr[:, 2] < v_arr[:, 0] - 1e-5) or np.any(v_arr[:, 3] < v_arr[:, 1] - 1e-5):
                        raise ValueError(f"{path} {grp_key}/{sub}: invalid bbox ordering (x2 < x1 or y2 < y1)")
            elif sub == "keypoints_xy_score":
                if np.any(all_fin):
                    v_arr = arr[all_fin]
                    coords = v_arr[:, :2]
                    if np.any(coords < -1e-5) or np.any(coords > 1.0 + 1e-5):
                        raise ValueError(f"{path} {grp_key}/{sub}: normalized keypoint coords outside [0, 1]")
                    if np.any(v_arr[:, 2] < -1e-5):
                        raise ValueError(f"{path} {grp_key}/{sub}: negative keypoint score")
            elif sub == "selected_grasp_xy_norm":
                if np.any(all_fin):
                    v_arr = arr[all_fin]
                    if np.any(v_arr < -1e-5) or np.any(v_arr > 1.0 + 1e-5):
                        raise ValueError(f"{path} {grp_key}/{sub}: normalized selected grasp coords outside [0, 1]")


def export_affordance_2d_jsonl(
    chosen,
    output_dir,
    *,
    cameras=None,
    frame_stride=1,
    save_heatmaps=True,
    save_overlays=False,
):
    """Export 2D affordance annotations from selected raw HDF5 episodes to VLM/VLA JSONL + heatmap PNGs."""
    from scripts.objects.affordance_2d import export_hdf5_to_vlm_jsonl

    return export_hdf5_to_vlm_jsonl(
        [path for path, _ in chosen],
        Path(output_dir),
        cameras=cameras,
        frame_stride=frame_stride,
        save_heatmaps=save_heatmaps,
        save_overlays=save_overlays,
    )


def check_raw_episodes(chosen):
    """Fail before multi-hour conversion if any chosen HDF5 is missing, has corrupt JPEGs, or invalid side streams."""
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
            _check_depth_hdf5(f, path, quality)
            _check_affordance_2d_hdf5(f, path, quality)


def patch_streaming_encoder(cls, av1_preset=DEFAULT_AV1_PRESET):
    """Configure LeRobot's StreamingVideoEncoder for lossless queue buffering and artifact-free SVT-AV1 preset."""
    if getattr(cls, "_bzf_orig_init", None) is None:
        cls._bzf_orig_init = cls.__init__
    orig_init = cls._bzf_orig_init

    def lossless_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.queue_maxsize = 0
        if getattr(self, "vcodec", None) == "libsvtav1" and av1_preset is not None:
            self.preset = av1_preset

    cls.__init__ = lossless_init


def is_wb19_state_keys(state_keys):
    """True when `state_keys` is the 19D fixed-base wholebody layout (`g1d_wb19`: waist + lift)."""
    keys = set(state_keys or ())
    return "waist_joint_state" in keys and "lift_joint_state" in keys


def patch_converter_wb19(converter):
    """Extend XPolicyLab's loaded LeRobot v3.0 converter in-memory to append 3D fixed-base body keys
    (`waist_joint_states` [2] + `lift_joint_states` [1]) after the 16D dual-arm vector (`g1d_wb19`)."""
    orig_dims = converter._dims_from_robot_action_info

    def dims_wb19(robot_action_dim_info):
        return [*orig_dims(robot_action_dim_info), 3]

    def motor_names_wb19(per_arm_dims):
        if len(per_arm_dims) == 3 and per_arm_dims[2] == 3:
            left = [f"left_joint_{i}" for i in range(per_arm_dims[0])]
            right = [f"right_joint_{i}" for i in range(per_arm_dims[1])]
            return [*left, *right, "waist_yaw", "waist_pitch", "lift_height"]
        return converter._orig_build_motor_names_from_dims(per_arm_dims)

    def extract_wb19(group):
        def _extract(data):
            vec = converter._concat_state_parts(
                [
                    ("left_arm_joint_states", converter._get_nested(data, group, "left_arm_joint_states")),
                    ("left_ee_joint_states", converter._get_nested(data, group, "left_ee_joint_states")),
                    ("right_arm_joint_states", converter._get_nested(data, group, "right_arm_joint_states")),
                    ("right_ee_joint_states", converter._get_nested(data, group, "right_ee_joint_states")),
                    ("waist_joint_states", converter._get_nested(data, group, "waist_joint_states")),
                    ("lift_joint_states", converter._get_nested(data, group, "lift_joint_states")),
                ],
                group,
            )
            if vec is not None:
                return vec
            raise ValueError(f"Cannot find {group} data")

        return _extract

    converter._orig_build_motor_names_from_dims = converter._build_motor_names_from_dims
    converter._dims_from_robot_action_info = dims_wb19
    converter._build_motor_names_from_dims = motor_names_wb19
    converter._extract_qpos = extract_wb19("state")
    converter._extract_action = extract_wb19("action")


def layout_key(quality):
    poses = {k: np.round(np.r_[p, q], 5).tolist() for k, (p, q) in sorted(quality["planned_poses"].items())}
    if quality.get("init_body_state"):
        poses["_init_body_state"] = {
            k: np.round(np.asarray(v, dtype=float), 4).tolist() for k, v in sorted(quality["init_body_state"].items())
        }
    return json.dumps(poses)


def _episode_twitch_reasons(
    hdf5_path,
    quality,
    *,
    window_s=twitch.TWITCH_WINDOW_S,
    rev_speed=twitch.TWITCH_REV_SPEED,
    max_rev_burst=twitch.TWITCH_REV_BURST,
    max_state_speed=twitch.TWITCH_STATE_SPEED_MAX,
    max_track_err=twitch.TWITCH_TRACK_ERR_MAX,
):
    motion_dict = quality.get("motion")
    default_params = window_s == twitch.TWITCH_WINDOW_S and rev_speed == twitch.TWITCH_REV_SPEED
    if default_params and twitch.has_twitch_metrics(motion_dict):
        return twitch.twitch_reasons(
            motion_dict,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        )
    if hdf5_path.is_file():
        _, reasons = twitch.check_hdf5_episode(
            hdf5_path,
            quality,
            window_s=window_s,
            rev_speed=rev_speed,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        )
        return reasons
    return []


def _depth_schema_signature(quality):
    depth = quality.get("depth")
    if not depth:
        return None
    return {
        "cameras": list(depth.get("cameras") or ()),
        "unit": depth.get("unit", "m"),
        "dtype": depth.get("dtype", "float32"),
    }


def _affordance_2d_schema_signature(quality):
    aff = quality.get("affordance_2d")
    if not aff:
        return None
    return {
        "cameras": list(aff.get("cameras") or ()),
        "heatmap_hw": list(aff.get("heatmap_hw") or ()),
        "top_k": int(aff.get("top_k", 0)),
    }


def select(
    raws,
    count,
    *,
    filter_twitch=False,
    window_s=twitch.TWITCH_WINDOW_S,
    rev_speed=twitch.TWITCH_REV_SPEED,
    max_rev_burst=twitch.TWITCH_REV_BURST,
    max_state_speed=twitch.TWITCH_STATE_SPEED_MAX,
    max_track_err=twitch.TWITCH_TRACK_ERR_MAX,
):
    """[(hdf5, quality)]: the first `count` accepted episodes (run, seed folder, then layout order) with distinct
    layouts, grouped by env config in name order (the converter's order)."""
    chosen, keys, skipped_twitch = [], set(), 0
    for sidecar in (p for raw in raws for p in sorted(raw.glob("seed*/episode_*.json"))):
        quality = json.loads(sidecar.read_text())
        if not quality["accepted"]:
            continue
        if not quality["hold_frames"] <= quality["stable_frames"] <= quality["frames"]:
            raise ValueError(f"Inconsistent acceptance evidence: {sidecar}")
        hdf5_path = sidecar.with_suffix(".hdf5")
        if filter_twitch and _episode_twitch_reasons(
            hdf5_path,
            quality,
            window_s=window_s,
            rev_speed=rev_speed,
            max_rev_burst=max_rev_burst,
            max_state_speed=max_state_speed,
            max_track_err=max_track_err,
        ):
            skipped_twitch += 1
            continue
        key = (quality["env_cfg"], layout_key(quality))
        if key in keys:
            continue
        keys.add(key)
        chosen.append((hdf5_path, quality))
        if len(chosen) == count:
            break
    else:
        extra = f" (skipped {skipped_twitch} twitching)" if skipped_twitch else ""
        raise ValueError(f"Only {len(chosen)} accepted episodes with distinct layouts in {raws}{extra}; need {count}")
    if skipped_twitch:
        print(f"export.select: skipped {skipped_twitch} twitching episodes", flush=True)
    mixed = [k for k in SAME if len({json.dumps(q.get(k)) for _, q in chosen}) > 1]
    if len({json.dumps(_depth_schema_signature(q), sort_keys=True) for _, q in chosen}) > 1:
        mixed.append("depth")
    if len({json.dumps(_affordance_2d_schema_signature(q), sort_keys=True) for _, q in chosen}) > 1:
        mixed.append("affordance_2d")
    if mixed:
        raise ValueError(f"Episodes disagree on {mixed}")
    return sorted(chosen, key=lambda c: c[1]["env_cfg"])


def convert(pattern, name, episodes, output, av1_preset=DEFAULT_AV1_PRESET, cameras=None, state_keys=None, wb19=False):
    """Run the official converter on data/<pattern>, writing <output>/<name>."""
    os.environ["HF_LEROBOT_HOME"] = str(output)  # read by lerobot at import
    spec = importlib.util.spec_from_file_location("transform_lerobot_v30_format", CONVERTER)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    # It roots data/ and env_cfg/ at its own resolved path, which is another checkout when XPolicyLab is a symlink
    # (as in server run dirs): pin both to this repo.
    converter.DATA_ROOT, converter.ENV_CFG_ROOT = ROOT / "data", ROOT / "env_cfg"
    converter.ROBOT_INFO_PATH = ROOT / "env_cfg/robot/_robot_info.json"
    if wb19 and not state_keys:
        state_keys = list(NATIVE_WHOLEBODY_STATE_KEYS)
    patch_converter(converter, cameras=cameras, state_keys=state_keys)
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


def convert_shards(name, task, chosen, output, shards, workers, av1_preset=DEFAULT_AV1_PRESET, wb19=False):
    """The converter on contiguous slices of `chosen`, `workers` at a time, joined into <output>/<name> by LeRobot's
    aggregate_datasets (as rsi_release.convert). The converter's encoder queues are unbounded, so a converter's memory
    grows with its slice (5056 episodes in 48 slices on 96 cores passed 400 GB): keep slices short."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    work = output / f"{name}.shards"
    cameras = chosen[0][1].get("cameras")
    state_keys = chosen[0][1].get("state_keys")
    names, jobs = [], []
    for k, (start, end) in enumerate(shard_slices(len(chosen), shards)):
        names.append(f"{name}_shard{k:03d}")
        stage(ROOT / "data" / names[-1], task, chosen[start:end])
        jobs.append((f"{names[-1]}.{task}.*", names[-1], end - start, work, av1_preset, cameras, state_keys, wb19))
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
        cam_map = camera_map(quality.get("cameras"))
        picks = sorted({int(i) for i in np.linspace(0, n - 1, min(n, 5)).round()} | {0, n // 2, n - 1})
        with h5py.File(path, "r") as f:
            state = np.concatenate([f[f"state/{k}s"][()] for k in quality["state_keys"]], axis=1)
            action = np.concatenate([f[f"action/{k}s"][()] for k in quality["state_keys"]], axis=1)
            jpegs = {cam: [f[f"vision/{cam}/colors"][i] for i in picks] for cam in cam_map.values()}
        for which, relative in enumerate(picks):
            row = start + relative
            frame = dataset[row]
            assert int(frame["episode_index"]) == index and int(frame["frame_index"]) == relative
            np.testing.assert_allclose(frame["observation.state"].numpy(), state[relative], atol=1e-6)
            np.testing.assert_allclose(frame["action"].numpy(), action[relative], atol=1e-6)
            for key, raw_cam in cam_map.items():
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
    cam_map = camera_map(q.get("cameras"))
    cam_desc = ", ".join(f"`observation.images.{k}` (`{raw}`)" for k, raw in cam_map.items())
    state_keys = q.get("state_keys") or list(STANDARD_STATE_KEYS)
    if state_keys == list(STANDARD_STATE_KEYS) and not q.get("hand_joint_names"):
        robot_label = "Unitree G1-D (Dex1-1 grippers)"
        state_desc = (
            "`observation.state`, `action` (16): left arm joints 1-7 (rad), left gripper, right arm joints 1-7, right"
            " gripper\n"
            "  (converter motor names `left_joint_0..7`, `right_joint_0..7`). Gripper state is the measured finger"
            " opening and the\n"
            "  gripper action the command, both 0 = closed .. 1 = open; arm actions are joint position targets."
        )
    elif is_wb19_state_keys(state_keys) and not q.get("hand_joint_names"):
        robot_label = "Unitree G1-D (Dex1-1 grippers)"
        state_desc = (
            "`observation.state`, `action` (19, `g1d_wb19`): left arm joints 1-7 (rad), left gripper, right arm "
            "joints 1-7, right gripper, waist `[yaw, pitch]` (rad), lift height (m, 0..0.42) (converter motor names "
            "`left_joint_0..7`, `right_joint_0..7`, `waist_yaw`, `waist_pitch`, `lift_height`). Gripper state is the "
            "measured finger opening and the gripper action the command, both 0 = closed .. 1 = open; arm and body "
            "actions are joint position targets (fixed base, no `base_velocity`)."
        )
    else:
        robot_label = "Unitree G1-D"
        state_desc = f"`observation.state`, `action`: concatenated `{', '.join(state_keys)}`."
    depth_line = ""
    if q.get("depth"):
        d = q["depth"]
        cams = ", ".join(f"`{c}`" for c in (d.get("cameras") or ()))
        depth_line = f"- Depth (stored in raw HDF5 `vision/<cam>/depths`, float32 m): cameras {cams}.\n"
    aff2d_line = ""
    if q.get("affordance_2d"):
        a2 = q["affordance_2d"]
        cams = ", ".join(f"`{c}`" for c in (a2.get("cameras") or ()))
        hw = f"{a2['heatmap_hw'][0]}x{a2['heatmap_hw'][1]}"
        aff2d_line = (
            f"- 2D affordance annotations (stored in raw HDF5 `affordance_2d/`): cameras {cams}, "
            f"normalized object/affordance BBoxes, top-{a2.get('top_k', 16)} 2D grasp keypoints, "
            f"and compact `uint8` heatmaps ({hw}).\n"
        )
    return f"""# {name}

{robot_label} `{q["task"]}` demonstrations generated in simulation by OpenFlywheel's scripted expert
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

- {state_desc}
- {cam_desc}: 640 x 480 RGB video.
{depth_line}{aff2d_line}
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
    parser.add_argument(
        "--filter-twitch",
        action="store_true",
        help="skip episodes exhibiting high-frequency arm twitching or physics-bounce spikes",
    )
    twitch.add_twitch_args(parser)
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
    chosen = select(
        [run / "raw" for run in args.run],
        args.episodes,
        filter_twitch=args.filter_twitch,
        window_s=args.window_s,
        rev_speed=args.rev_speed,
        max_rev_burst=args.max_rev_burst,
        max_state_speed=args.max_state_speed,
        max_track_err=args.max_track_err,
    )
    check_raw_episodes(chosen)
    task = chosen[0][1]["task"]
    cameras = chosen[0][1].get("cameras")
    state_keys = chosen[0][1].get("state_keys")
    wb19 = is_wb19_state_keys(state_keys)
    if args.shards == 1:
        stage(staging, task, chosen)
        convert(
            f"{args.name}.{task}.*",
            args.name,
            args.episodes,
            output,
            av1_preset=args.av1_preset,
            cameras=cameras,
            state_keys=state_keys,
            wb19=wb19,
        )
    else:
        convert_shards(
            args.name,
            task,
            chosen,
            output,
            args.shards,
            args.workers or args.shards,
            av1_preset=args.av1_preset,
            wb19=wb19,
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
