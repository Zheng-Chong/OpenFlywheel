"""Record expert demonstrations as XPolicyLab trajectories: one HDF5 per episode, in the layout XPolicyLab's
scripts/transform_lerobot_v30_format.py reads, plus a quality.json sidecar with the acceptance result.

Episodes run through src/eval_client/main.py like an eval, with this module as the in-process policy
(--policy_module), so the recorded state is the observation a policy gets at eval time (scripts/datagen/CONTRACT.md).
The task's expert, acceptance check and audit are scripts/datagen/plugins/<task>.py.

Usage (Isaac env, repo root; run_batch.py runs one per GPU):
    PYTHONPATH=.:XPolicyLab python scripts/datagen/collect.py --task g1d_pour_beans_measuring_cups \
        --env-cfg g1d_beans_collect --seed 9 --output output/datagen/probe [--layout-ids 0,1] [--stop-file PATH] \
        [--device cuda:1 --device_id 1]
Layouts already recorded in --output are skipped, so a rerun resumes. --no-trajectory writes only the sidecars (an
expert feasibility filter of benchmark layouts: no images, no HDF5). --plugin picks another plugin of the task (e.g.
g1d_pour_beans_plus_v2); a plugin with RECORD_TO_END records the whole script and its TAIL instead of stopping once
the acceptance has held for HOLD frames, and accepts when it holds over the last HOLD frames.
--expert vN runs version N of the task's expert (scripts/datagen/experts.json, its plugin) and refuses code that is
not vN's; without it the plugin's newest version is recorded, with the source files that differ from it.
An episode that can no longer be accepted ends there (scripts/datagen/early_abort.py; rejected, its sidecar's
"aborted" gives the reason and frame); --no-early-abort runs it to the end.
"""

import argparse
from collections.abc import Mapping
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.observation_manager.vision_utils import (  # noqa: E402
    DEFAULT_AFFORDANCE_HEATMAP_HW,
    DEFAULT_AFFORDANCE_TOP_K,
    as_numpy as _as_numpy,
    is_usable_intrinsics as _is_usable_intrinsics,
    parse_camera_selection,
    resolve_affordance_target,
)
from scripts.datagen import early_abort, experts, motion  # noqa: E402
from scripts.datagen.export import valid_jpeg  # noqa: E402
from scripts.datagen.layout_pool import LayoutPool  # noqa: E402

ENV_KEY = "OPENFLYWHEEL_DATAGEN"  # JSON config for the in-process policy; survives main.py's os.execv restarts
LEGACY_ENV_KEY = "ROBODOJO_DATAGEN"
CAMERAS = ("cam_head", "cam_head_right", "cam_left_wrist", "cam_right_wrist")
SECTIONS = ("Rigid", "Articulation", "Fluid", "Geometry", "Clutter", "Dynamic", "Garment")
BODY_RECORD_KEYS = ("waist_joint_state", "lift_joint_state")
AFFORDANCE_2D_DEFAULT_CAMERAS = ("cam_head",)
AFFORDANCE_2D_DEFAULT_HEATMAP_HW = DEFAULT_AFFORDANCE_HEATMAP_HW
AFFORDANCE_2D_DEFAULT_TOP_K = DEFAULT_AFFORDANCE_TOP_K


def plugin_name(task):
    return f"scripts.datagen.plugins.{task}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_revision():
    """{commit, dirty} of the checkout; None outside a git checkout (an uploaded run dir)."""
    quiet = dict(cwd=ROOT, stderr=subprocess.DEVNULL)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, **quiet).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], **quiet))
    except (OSError, subprocess.CalledProcessError):
        return None
    return dict(commit=commit, dirty=dirty)


def load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def layout_dir(env_cfg, seed):
    config_name = load_yaml(ROOT / "env_cfg" / f"{env_cfg}.yml")["config_name"]
    return ROOT / "Assets" / "Eval_Layout" / "RoboDojo" / config_name / str(seed)


def layout_files(env_cfg, seed, task):
    """The seed folder's layouts in the eval client's order (SeedManager): index = layout id."""
    files = layout_dir(env_cfg, seed).glob(f"{task}_*.json")
    return sorted(
        (p for p in files if p.stem.rpartition("_")[2].isdigit()), key=lambda p: int(p.stem.rpartition("_")[2])
    )


def planned_poses(layout):
    """{label: [position, wxyz]} of every labelled object as laid out (before physics settles it)."""
    poses = {}
    for section in SECTIONS:
        for entries in (layout.get(section) or {}).values():
            for entry in entries:
                if entry.get("label") and "default_pos" in entry:
                    poses[entry["label"]] = [list(entry["default_pos"]), list(entry.get("default_ori", [1, 0, 0, 0]))]
    return poses


BODY_RECORD_KEYS = ("waist_joint_state", "lift_joint_state")


def state_keys(env):
    """obs['state'] / action keys of the target arms, in the converters' order: arm, gripper per arm, left first,
    followed by fixed-base body keys (waist_joint_state, lift_joint_state) when observation.robot.body_states is on."""
    rm = env.robot_manager
    robots = sorted((r for r in rm.robot_list if r.type == "target"), key=lambda r: r.arm_name)
    keys = [rm.process_name(name) for r in robots for name in (r.arm_name, r.gripper_name)]
    om = getattr(env, "obs_manager", None)
    if getattr(om, "robot_cfg", None) and om.robot_cfg.get("body_states", False):
        body_dims = {k for r in robots for k in getattr(r, "body_dims", ())}
        keys.extend(k for k in BODY_RECORD_KEYS if k in body_dims)
    return keys


def validate_images(images, path="trajectory"):
    """Raise ValueError if any camera stream is empty or holds an invalid/truncated JPEG frame."""
    for camera, frames in images.items():
        if not frames:
            raise ValueError(f"{path} {camera}: empty image stream")
        for frame_idx, buf in enumerate(frames):
            if not valid_jpeg(buf):
                raise ValueError(f"{path} {camera} frame {frame_idx}: invalid or truncated JPEG")


def write_privileged(path, privileged):
    """Write lightweight privileged state as a compressed .npz (readable by early_abort.load_privileged)."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **{k: np.asarray(v) for k, v in privileged.items()})
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class DepthRecorder:
    """Record per-camera metric depth (m) and optional camera matrices aligned 1:1 with RGB frames."""

    def __init__(self, cameras, depth_cameras=None, *, strict: bool = False):
        self.all_cameras = tuple(cameras)
        self.explicit_cameras = bool(depth_cameras)
        if depth_cameras:
            wanted = [c for c in self.all_cameras if c in set(depth_cameras)]
            missing = [c for c in depth_cameras if c not in set(self.all_cameras)]
            if strict and missing:
                raise ValueError(f"Requested depth cameras {missing} not found in rendered cameras {self.all_cameras}")
        else:
            wanted = list(self.all_cameras)
        self.cameras = tuple(wanted)
        self.strict = bool(strict)
        self._depths: dict[str, list[np.ndarray]] = {}
        self._shapes: dict[str, tuple[int, int]] = {}
        self._intrinsics: dict[str, np.ndarray] = {}
        self._extrinsics: dict[str, list[np.ndarray]] = {}
        self._frames_seen = 0

    @staticmethod
    def _extract_depth_array(cam_obs: Mapping) -> np.ndarray | None:
        raw = cam_obs.get("depth")
        if raw is None and cam_obs.get("approximate_depth") is not None:
            raw = np.asarray(cam_obs["approximate_depth"], dtype=np.float32) * 1e-3
        if raw is None:
            return None
        arr = np.array(raw, dtype=np.float32, copy=True)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def record(self, obs: Mapping) -> None:
        self._frames_seen += 1
        vision = obs.get("vision") if isinstance(obs, Mapping) else None
        if not isinstance(vision, Mapping):
            if self.strict:
                raise ValueError("Depth recording requested, but obs['vision'] is missing")
            return
        recorded_this_frame = 0
        for cam in self.cameras:
            cam_obs = vision.get(cam)
            if not isinstance(cam_obs, Mapping):
                if self.strict and self.explicit_cameras:
                    raise ValueError(f"Depth recording requested for {cam!r}, but camera is missing in obs['vision']")
                if cam in self._depths:
                    raise ValueError(f"Depth stream for {cam!r} missing at frame {self._frames_seen - 1}")
                continue
            arr = self._extract_depth_array(cam_obs)
            if arr is None:
                if self.strict and self.explicit_cameras:
                    raise ValueError(f"Depth recording requested for {cam!r}, but obs['vision'][{cam!r}] lacks 'depth'")
                if cam in self._depths:
                    raise ValueError(f"Depth stream for {cam!r} dropped out at frame {self._frames_seen - 1}")
                continue
            if arr.ndim != 2:
                raise ValueError(f"{cam} depth must be 2D (H, W), got shape {arr.shape}")
            color = cam_obs.get("color")
            if color is not None:
                ch, cw = int(np.asarray(color).shape[0]), int(np.asarray(color).shape[1])
                if arr.shape != (ch, cw):
                    raise ValueError(f"{cam} depth shape {arr.shape} does not match color shape {(ch, cw)}")
            if cam in self._shapes:
                if arr.shape != self._shapes[cam]:
                    raise ValueError(f"{cam} depth shape changed mid-episode: {arr.shape} != {self._shapes[cam]}")
            else:
                if self._frames_seen > 1:
                    raise ValueError(f"{cam} depth appeared mid-episode at frame {self._frames_seen - 1}")
                self._shapes[cam] = (int(arr.shape[0]), int(arr.shape[1]))
            self._depths.setdefault(cam, []).append(arr)
            recorded_this_frame += 1

            if cam not in self._intrinsics and cam_obs.get("intrinsic_matrix") is not None:
                k_arr = _as_numpy(cam_obs["intrinsic_matrix"]).astype(np.float64)
                if k_arr.shape == (3, 3) and np.isfinite(k_arr).all():
                    self._intrinsics[cam] = k_arr
            if cam_obs.get("extrinsic_matrix") is not None:
                ext_arr = _as_numpy(cam_obs["extrinsic_matrix"]).astype(np.float64)
                if ext_arr.shape == (4, 4) and np.isfinite(ext_arr).all():
                    self._extrinsics.setdefault(cam, []).append(ext_arr)
        if self.strict and recorded_this_frame == 0:
            raise ValueError(
                f"Depth recording requested, but obs['vision'] lacks 'depth' for all cameras {self.cameras}"
            )

    def payload(self, expected_frames: int):
        if not self._depths:
            if self.strict:
                raise ValueError("Depth recording requested, but no depth frames were recorded")
            return {}, {}
        data, shapes, has_k, has_ext, datasets = {}, {}, [], [], []
        for cam, frames in self._depths.items():
            if len(frames) != expected_frames:
                raise ValueError(f"{cam} depth has {len(frames)} frames, expected {expected_frames}")
            stack = np.stack(frames, axis=0).astype(np.float32)
            entry = {"depths": stack}
            shapes[cam] = [int(stack.shape[1]), int(stack.shape[2])]
            datasets.append(f"vision/{cam}/depths")
            if cam in self._intrinsics:
                entry["intrinsic_matrix"] = self._intrinsics[cam]
                has_k.append(cam)
                datasets.append(f"vision/{cam}/intrinsic_matrix")
            ext_frames = self._extrinsics.get(cam)
            if ext_frames and len(ext_frames) == expected_frames:
                entry["extrinsic_matrices"] = np.stack(ext_frames, axis=0).astype(np.float64)
                has_ext.append(cam)
                datasets.append(f"vision/{cam}/extrinsic_matrices")
            data[cam] = entry
        meta = dict(
            cameras=list(data),
            unit="m",
            dtype="float32",
            shapes=shapes,
            has_intrinsics=has_k,
            has_extrinsics=has_ext,
            datasets=datasets,
        )
        return data, meta


def _write_depth_streams(h5file, depths: dict, expected_frames: int) -> None:
    for cam, entry in depths.items():
        arr = np.asarray(entry["depths"], dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != expected_frames:
            raise ValueError(f"{cam} depths must be ({expected_frames}, H, W), got {arr.shape}")
        ds = h5file.create_dataset(
            f"vision/{cam}/depths",
            data=arr,
            chunks=(1, arr.shape[1], arr.shape[2]),
            compression="gzip",
        )
        ds.attrs["unit"] = "m"
        ds.attrs["source"] = "distance_to_image_plane"
        if "intrinsic_matrix" in entry:
            k_arr = np.asarray(entry["intrinsic_matrix"], dtype=np.float64)
            if k_arr.shape != (3, 3):
                raise ValueError(f"{cam} intrinsic_matrix must be (3, 3), got {k_arr.shape}")
            h5file.create_dataset(f"vision/{cam}/intrinsic_matrix", data=k_arr)
        if "extrinsic_matrices" in entry:
            ext_arr = np.asarray(entry["extrinsic_matrices"], dtype=np.float64)
            if ext_arr.shape != (expected_frames, 4, 4):
                raise ValueError(f"{cam} extrinsic_matrices must be ({expected_frames}, 4, 4), got {ext_arr.shape}")
            ext_ds = h5file.create_dataset(
                f"vision/{cam}/extrinsic_matrices",
                data=ext_arr,
                compression="gzip",
            )
            ext_ds.attrs["convention"] = "usd"


class Affordance2DRecorder:
    """Opt-in per-frame 2D affordance recorder (reads obs['vision'][cam]['affordance'] or projects from 3D)."""

    def __init__(self, env, episode, cfg, layout_id, *, all_cameras=None):
        raw_cams = cfg.get("affordance_2d_cameras")
        cam_sel = parse_camera_selection(raw_cams)
        if raw_cams is None or cam_sel is True:
            self.cameras = tuple(all_cameras) if all_cameras else AFFORDANCE_2D_DEFAULT_CAMERAS
        elif isinstance(cam_sel, set):
            if isinstance(raw_cams, list | tuple):
                self.cameras = tuple(str(c).strip() for c in raw_cams if str(c).strip())
            else:
                self.cameras = tuple(sorted(cam_sel))
        else:
            self.cameras = AFFORDANCE_2D_DEFAULT_CAMERAS
        hm_hw = cfg.get("affordance_2d_heatmap_hw") or AFFORDANCE_2D_DEFAULT_HEATMAP_HW
        self.heatmap_hw = (int(hm_hw[0]), int(hm_hw[1]))
        if self.heatmap_hw[0] <= 0 or self.heatmap_hw[1] <= 0:
            raise ValueError(f"affordance_2d_heatmap_hw must be positive, got {self.heatmap_hw}")
        self.top_k = int(cfg.get("affordance_2d_top_k", AFFORDANCE_2D_DEFAULT_TOP_K))
        if self.top_k < 1:
            raise ValueError(f"affordance_2d_top_k must be >= 1, got {self.top_k}")
        self.enabled = False
        self.target_label = None
        self.affordance = None
        self.scale = 1.0
        self.gripper_types = None
        self.asset_id = ""
        self._streams = {}
        self._source_hw = {}
        self._frames_seen = 0
        self._has_selected_grasp = False
        self._fallback_pose = None
        self._init_target(env, episode, cfg, layout_id)

    def _init_target(self, env, episode, cfg, layout_id):
        preferred_label = (
            cfg.get("affordance_2d_target_label")
            or getattr(episode, "target_label", None)
            or getattr(episode, "target", None)
        )
        target = resolve_affordance_target(
            env,
            env_idx=0,
            layout_id=layout_id,
            preferred_label=preferred_label,
            affordance_path=cfg.get("affordance_2d_path"),
            affordance_data=(
                episode.affordance_data if isinstance(getattr(episode, "affordance_data", None), Mapping) else None
            ),
            robot_model=cfg.get("robot_model"),
        )
        if target is None:
            return
        self.target_label = target["target_label"]
        self.affordance = target["affordance"]
        self.asset_id = target["asset_id"]
        self.scale = target["scale"]
        self._fallback_pose = target["fallback_pose"]
        self.gripper_types = target["gripper_types"]
        self.enabled = True

    @staticmethod
    def _env_origin(env) -> np.ndarray:
        sm = getattr(env, "scene_manager", None)
        origins = getattr(sm, "env_origins", None)
        if origins is None:
            origins = getattr(env, "env_origins", None)
        if origins is not None and len(origins) > 0:
            return _as_numpy(origins[0]).astype(np.float64).reshape(3)
        return np.zeros(3, dtype=np.float64)

    def _get_object_pose(self, env, obs):
        sm = getattr(env, "scene_manager", None)
        lm = getattr(sm, "layout_manager", None)
        if lm is not None and hasattr(lm, "get_instance_pose"):
            try:
                pos, quat = lm.get_instance_pose(0, label=self.target_label)
                if pos is not None and quat is not None:
                    return _as_numpy(pos).astype(np.float64).reshape(3), _as_numpy(quat).astype(np.float64).reshape(4)
            except Exception:
                pass
        obj_poses = obs.get("object_poses") if isinstance(obs, Mapping) else None
        if isinstance(obj_poses, Mapping) and self.target_label in obj_poses:
            pos, quat = obj_poses[self.target_label]
            return _as_numpy(pos).astype(np.float64).reshape(3), _as_numpy(quat).astype(np.float64).reshape(4)
        return self._fallback_pose

    def _get_camera_params(self, env, obs, cam):
        cam_obs = obs.get("vision", {}).get(cam)
        if not isinstance(cam_obs, Mapping):
            return None
        K = cam_obs.get("intrinsic_matrix")
        ext = cam_obs.get("extrinsic_matrix")
        conv = cam_obs.get("camera_convention", "usd")
        if K is None or ext is None:
            cm = getattr(env, "camera_manager", None)
            if cm is not None and hasattr(cm, "camera_names"):
                try:
                    cam_id = list(cm.camera_names[0]).index(cam)
                    if K is None:
                        K = cm.get_camera_intrinsics(cam_id, 0)
                    if ext is None:
                        ext = cm.get_camera_extrinsics(cam_id, 0)
                except Exception:
                    return None
        if K is None or ext is None:
            return None
        k_arr = _as_numpy(K).astype(np.float64)
        ext_arr = _as_numpy(ext).astype(np.float64)
        if ext_arr.shape != (4, 4) or not np.isfinite(ext_arr).all() or not _is_usable_intrinsics(k_arr):
            return None
        ext_rel = ext_arr.copy()
        ext_rel[:3, 3] -= self._env_origin(env)
        color = cam_obs.get("color")
        if color is not None:
            h, w = int(np.asarray(color).shape[0]), int(np.asarray(color).shape[1])
        else:
            shape = cam_obs.get("shape")
            if shape is None:
                return None
            h, w = int(shape[0]), int(shape[1])
        depth = cam_obs.get("depth")
        if depth is None and cam_obs.get("approximate_depth") is not None:
            depth = np.asarray(cam_obs["approximate_depth"], dtype=np.float32) * 1e-3
        return k_arr, ext_rel, w, h, conv, depth

    def _append_empty_frame(self, stream):
        hm_h, hm_w = self.heatmap_hw
        stream["object_bbox_xyxy_norm"].append(np.full((4,), np.nan, dtype=np.float32))
        stream["affordance_bbox_xyxy_norm"].append(np.full((4,), np.nan, dtype=np.float32))
        stream["keypoints_xy_score"].append(np.full((self.top_k, 3), np.nan, dtype=np.float32))
        stream["selected_grasp_xy_norm"].append(np.full((2,), np.nan, dtype=np.float32))
        stream["heatmap_uint8"].append(np.zeros((hm_h, hm_w), dtype=np.uint8))

    def _ensure_stream(self, cam):
        if cam not in self._streams:
            st = dict(
                object_bbox_xyxy_norm=[],
                affordance_bbox_xyxy_norm=[],
                keypoints_xy_score=[],
                selected_grasp_xy_norm=[],
                heatmap_uint8=[],
            )
            for _ in range(self._frames_seen - 1):
                self._append_empty_frame(st)
            self._streams[cam] = st
        return self._streams[cam]

    @staticmethod
    def _resize_hm_uint8(hm: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(hm, dtype=np.uint8)
        if arr.shape == out_hw:
            return arr
        from PIL import Image

        return np.asarray(
            Image.fromarray(arr, mode="L").resize((int(out_hw[1]), int(out_hw[0])), resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    @staticmethod
    def _project_selected_grasp(sel_grasp, k_arr, c2w_rel, w: int, h: int, conv: str):
        from scripts.objects.affordance_2d import project_world_points

        sel_pt = np.asarray(sel_grasp, dtype=np.float64).reshape(1, 3)
        if not np.isfinite(sel_pt).all():
            return None
        su, sv, sz = project_world_points(sel_pt, k_arr, c2w_rel, camera_convention=conv)
        u0, v0, z0 = float(su[0]), float(sv[0]), float(sz[0])
        if np.isfinite(u0) and np.isfinite(v0) and z0 > 0.02 and 0.0 <= u0 < float(w) and 0.0 <= v0 < float(h):
            return {"u_norm": round(u0 / float(w), 6), "v_norm": round(v0 / float(h), 6), "visible": True}
        return None

    def _adopt_obs_metadata(self, aff_obs: Mapping) -> None:
        if not self.enabled:
            self.enabled = True
        if not self.target_label and aff_obs.get("target_label"):
            self.target_label = str(aff_obs["target_label"])
        if not self.asset_id and aff_obs.get("asset_id"):
            self.asset_id = str(aff_obs["asset_id"])
        if aff_obs.get("scale") is not None and self.scale == 1.0:
            self.scale = aff_obs["scale"]
        if not self.gripper_types and aff_obs.get("gripper_types"):
            self.gripper_types = tuple(str(g) for g in aff_obs["gripper_types"])

    def record(self, env, obs, episode):
        self._frames_seen += 1
        vision = obs.get("vision", {}) if isinstance(obs, Mapping) else {}
        has_obs_aff = any(
            isinstance(vision.get(c), Mapping) and isinstance(vision[c].get("affordance"), Mapping)
            for c in self.cameras
        )
        if not self.enabled and not has_obs_aff:
            return

        sel_grasp = getattr(episode, "selected_grasp_world", None)
        if sel_grasp is None:
            sel_grasp = getattr(getattr(episode, "robot", None), "selected_grasp_world", None)
        if sel_grasp is not None:
            self._has_selected_grasp = True

        hm_h, hm_w = self.heatmap_hw
        pose = None if has_obs_aff else self._get_object_pose(env, obs)

        for cam in self.cameras:
            cam_obs = vision.get(cam)
            aff_obs = cam_obs.get("affordance") if isinstance(cam_obs, Mapping) else None
            if isinstance(aff_obs, Mapping) and "heatmap_uint8" in aff_obs:
                self._adopt_obs_metadata(aff_obs)
                color = cam_obs.get("color")
                shape = cam_obs.get("shape")
                if color is not None:
                    h, w = int(np.asarray(color).shape[0]), int(np.asarray(color).shape[1])
                elif shape is not None:
                    h, w = int(shape[0]), int(shape[1])
                else:
                    h, w = hm_h, hm_w
                stream = self._ensure_stream(cam)
                self._source_hw[cam] = [h, w]
                ob = aff_obs.get("object_bbox_xyxy_norm")
                ab = aff_obs.get("affordance_bbox_xyxy_norm")
                stream["object_bbox_xyxy_norm"].append(
                    np.asarray(ob, dtype=np.float32) if ob is not None else np.full((4,), np.nan, dtype=np.float32)
                )
                stream["affordance_bbox_xyxy_norm"].append(
                    np.asarray(ab, dtype=np.float32) if ab is not None else np.full((4,), np.nan, dtype=np.float32)
                )
                kps = np.full((self.top_k, 3), np.nan, dtype=np.float32)
                for idx, kp in enumerate((aff_obs.get("keypoints") or [])[: self.top_k]):
                    kps[idx, 0] = float(kp["u_norm"])
                    kps[idx, 1] = float(kp["v_norm"])
                    kps[idx, 2] = float(kp["score"])
                stream["keypoints_xy_score"].append(kps)
                sel2d = aff_obs.get("selected_grasp_2d")
                if sel2d is None and sel_grasp is not None:
                    params = self._get_camera_params(env, obs, cam)
                    if params is not None:
                        K, c2w_rel, pw, ph, conv, _ = params
                        sel2d = self._project_selected_grasp(sel_grasp, K, c2w_rel, pw, ph, conv)
                if isinstance(sel2d, Mapping) and sel2d.get("visible"):
                    self._has_selected_grasp = True
                    stream["selected_grasp_xy_norm"].append(
                        np.asarray([float(sel2d["u_norm"]), float(sel2d["v_norm"])], dtype=np.float32)
                    )
                else:
                    stream["selected_grasp_xy_norm"].append(np.full((2,), np.nan, dtype=np.float32))
                stream["heatmap_uint8"].append(self._resize_hm_uint8(aff_obs["heatmap_uint8"], (hm_h, hm_w)))
                continue

            if not self.enabled:
                if cam in self._streams:
                    self._append_empty_frame(self._streams[cam])
                continue
            if pose is None:
                pose = self._get_object_pose(env, obs)
            if pose is None:
                if cam in self._streams:
                    self._append_empty_frame(self._streams[cam])
                continue
            params = self._get_camera_params(env, obs, cam)
            if params is None:
                if cam in self._streams:
                    self._append_empty_frame(self._streams[cam])
                continue
            from scripts.objects.affordance_2d import project_affordance_2d

            obj_pos, obj_quat = pose
            K, c2w_rel, w, h, conv, depth = params
            res = project_affordance_2d(
                self.affordance,
                obj_pos,
                obj_quat,
                K,
                c2w_rel,
                w,
                h,
                scale=self.scale,
                depth=depth,
                gripper_types=self.gripper_types,
                max_keypoints=self.top_k,
                selected_grasp_world=sel_grasp,
                camera_convention=conv,
                heatmap_hw=(hm_h, hm_w),
            )
            stream = self._ensure_stream(cam)
            self._source_hw[cam] = [h, w]
            stream["object_bbox_xyxy_norm"].append(
                np.asarray(res.object_bbox_xyxy_norm, dtype=np.float32)
                if res.object_bbox_xyxy_norm is not None
                else np.full((4,), np.nan, dtype=np.float32)
            )
            stream["affordance_bbox_xyxy_norm"].append(
                np.asarray(res.affordance_bbox_xyxy_norm, dtype=np.float32)
                if res.affordance_bbox_xyxy_norm is not None
                else np.full((4,), np.nan, dtype=np.float32)
            )
            kps = np.full((self.top_k, 3), np.nan, dtype=np.float32)
            for idx, kp in enumerate(res.keypoints[: self.top_k]):
                kps[idx, 0] = float(kp["u_norm"])
                kps[idx, 1] = float(kp["v_norm"])
                kps[idx, 2] = float(kp["score"])
            stream["keypoints_xy_score"].append(kps)
            if res.selected_grasp_2d and res.selected_grasp_2d.get("visible"):
                stream["selected_grasp_xy_norm"].append(
                    np.asarray(
                        [float(res.selected_grasp_2d["u_norm"]), float(res.selected_grasp_2d["v_norm"])],
                        dtype=np.float32,
                    )
                )
            else:
                stream["selected_grasp_xy_norm"].append(np.full((2,), np.nan, dtype=np.float32))
            stream["heatmap_uint8"].append(np.asarray(res.heatmap_uint8, dtype=np.uint8))

    def payload(self, expected_frames: int):
        if not self._streams:
            return {}, {}
        cams_out = {}
        datasets = []
        for cam, st in self._streams.items():
            for key, rows in st.items():
                if len(rows) != expected_frames:
                    raise ValueError(f"affordance_2d {cam}/{key} has {len(rows)} frames, expected {expected_frames}")
            cams_out[cam] = {
                "object_bbox_xyxy_norm": np.stack(st["object_bbox_xyxy_norm"], axis=0).astype(np.float32),
                "affordance_bbox_xyxy_norm": np.stack(st["affordance_bbox_xyxy_norm"], axis=0).astype(np.float32),
                "keypoints_xy_score": np.stack(st["keypoints_xy_score"], axis=0).astype(np.float32),
                "selected_grasp_xy_norm": np.stack(st["selected_grasp_xy_norm"], axis=0).astype(np.float32),
                "heatmap_uint8": np.stack(st["heatmap_uint8"], axis=0).astype(np.uint8),
                "source_hw": list(self._source_hw[cam]),
            }
            datasets.extend(
                [
                    f"affordance_2d/{cam}/object_bbox_xyxy_norm",
                    f"affordance_2d/{cam}/affordance_bbox_xyxy_norm",
                    f"affordance_2d/{cam}/keypoints_xy_score",
                    f"affordance_2d/{cam}/selected_grasp_xy_norm",
                    f"affordance_2d/{cam}/heatmap_uint8",
                ]
            )
        payload = {
            "target_label": self.target_label,
            "asset_id": self.asset_id,
            "scale": self.scale,
            "gripper_types": list(self.gripper_types or ()),
            "heatmap_hw": list(self.heatmap_hw),
            "top_k": int(self.top_k),
            "has_selected_grasp": bool(self._has_selected_grasp),
            "cameras": cams_out,
        }
        meta = {
            "target_label": self.target_label,
            "asset_id": self.asset_id,
            "gripper_types": list(self.gripper_types or ()),
            "cameras": list(cams_out),
            "heatmap_hw": list(self.heatmap_hw),
            "top_k": int(self.top_k),
            "source_hw": dict(self._source_hw),
            "has_selected_grasp": bool(self._has_selected_grasp),
            "datasets": datasets,
        }
        return payload, meta


def _write_affordance_2d_group(h5file, affordance_2d: Mapping, expected_frames: int) -> None:
    cams = affordance_2d.get("cameras") or {}
    if not cams:
        return
    hm_h, hm_w = (int(x) for x in affordance_2d["heatmap_hw"])
    top_k = int(affordance_2d["top_k"])
    grp = h5file.create_group("affordance_2d")
    grp.attrs["target_label"] = str(affordance_2d.get("target_label") or "")
    grp.attrs["asset_id"] = str(affordance_2d.get("asset_id") or "")
    grp.attrs["scale"] = json.dumps(affordance_2d.get("scale", 1.0))
    grp.attrs["gripper_types"] = json.dumps(list(affordance_2d.get("gripper_types") or ()))
    grp.attrs["top_k"] = top_k
    grp.attrs["heatmap_hw"] = json.dumps([hm_h, hm_w])
    grp.attrs["has_selected_grasp"] = bool(affordance_2d.get("has_selected_grasp", False))
    for cam, entry in cams.items():
        cg = grp.create_group(cam)
        cg.attrs["source_hw"] = json.dumps(list(entry["source_hw"]))
        cg.attrs["target_label"] = str(affordance_2d.get("target_label") or "")
        cg.attrs["asset_id"] = str(affordance_2d.get("asset_id") or "")
        cg.attrs["coordinate_system"] = "normalized_xyxy_image_0_1"
        obj_bb = np.asarray(entry["object_bbox_xyxy_norm"], dtype=np.float32)
        aff_bb = np.asarray(entry["affordance_bbox_xyxy_norm"], dtype=np.float32)
        kps = np.asarray(entry["keypoints_xy_score"], dtype=np.float32)
        sel = np.asarray(entry["selected_grasp_xy_norm"], dtype=np.float32)
        hm = np.asarray(entry["heatmap_uint8"], dtype=np.uint8)
        if obj_bb.shape != (expected_frames, 4):
            raise ValueError(f"object_bbox_xyxy_norm shape {obj_bb.shape} != ({expected_frames}, 4)")
        if aff_bb.shape != (expected_frames, 4):
            raise ValueError(f"affordance_bbox_xyxy_norm shape {aff_bb.shape} != ({expected_frames}, 4)")
        if kps.shape != (expected_frames, top_k, 3):
            raise ValueError(f"keypoints_xy_score shape {kps.shape} != ({expected_frames}, {top_k}, 3)")
        if sel.shape != (expected_frames, 2):
            raise ValueError(f"selected_grasp_xy_norm shape {sel.shape} != ({expected_frames}, 2)")
        if hm.shape != (expected_frames, hm_h, hm_w):
            raise ValueError(f"heatmap_uint8 shape {hm.shape} != ({expected_frames}, {hm_h}, {hm_w})")
        cg.create_dataset("object_bbox_xyxy_norm", data=obj_bb)
        cg.create_dataset("affordance_bbox_xyxy_norm", data=aff_bb)
        cg.create_dataset("keypoints_xy_score", data=kps)
        cg.create_dataset("selected_grasp_xy_norm", data=sel)
        cg.create_dataset("heatmap_uint8", data=hm, compression="gzip")


def write_trajectory(
    path,
    instruction,
    info,
    states,
    actions,
    images,
    privileged,
    *,
    depths=None,
    affordance_2d=None,
):
    """XPolicyLab trajectory layout (README "Trajectory Data Format"): state/ and action/ as <key>s (T, D) float32,
    vision/<camera>/colors as encode_image_bit JPEGs in a fixed-width byte column; privileged/ is ours (audit only);
    optional vision/<camera>/{depths,intrinsic_matrix,extrinsic_matrices} and affordance_2d/<camera>/."""
    import h5py

    validate_images(images, path)
    expected_frames = len(next(iter(states.values()))) if states else 0
    tmp = path.with_suffix(".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f["data_format_version"] = "v1.0"
            f["instruction"] = instruction
            for key, value in info.items():
                f[f"additional_info/{key}"] = json.dumps(value) if isinstance(value, dict | list) else value
            for key, rows in states.items():
                f[f"state/{key}s"] = np.asarray(rows, dtype=np.float32)
            for key, rows in actions.items():
                f[f"action/{key}s"] = np.asarray(rows, dtype=np.float32)
            for camera, frames in images.items():
                f[f"vision/{camera}/colors"] = np.asarray(frames, dtype=f"S{max(map(len, frames))}")
            if depths:
                _write_depth_streams(f, depths, expected_frames)
            for key, value in privileged.items():
                value = np.asarray(value)
                f.create_dataset(f"privileged/{key}", data=value, compression="gzip" if value.ndim else None)
            if affordance_2d:
                _write_affordance_2d_group(f, affordance_2d, expected_frames)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def record_episode(env, plugin, output, cfg):
    if env.num_envs != 1 or env.task_name != plugin.TASK:
        raise ValueError(f"Collector needs one {plugin.TASK} environment")
    layout_id = env.current_env_seed_map[0]
    path = output / f"episode_{layout_id:06d}.hdf5"
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    env.end_on_success = False  # acceptance is the plugin's, not the task's success check
    env.step_lim = plugin.STEP_LIM
    layout_file = Path(env.seed_manager.seed_info[layout_id]["scene_layout"])
    quality = dict(
        task=plugin.TASK,
        plugin=plugin.__name__,
        acceptance_version=plugin.ACCEPTANCE,
        hold_frames=plugin.HOLD,
        env_cfg=cfg["env_cfg"],
        config_name=env.config_name,
        physics_dt=env.dt,
        fps=cfg["fps"],
        layout_seed=int(env.eval_seed),
        layout_id=layout_id,
        layout_file=layout_file.name,
        layout_sha256=sha256(layout_file),
        planned_poses=planned_poses(env.seed_manager.get_seed_scene_info(layout_id)),
        revision=cfg.get("revision"),
        expert=cfg.get("expert"),
    )
    episode = plugin.Episode(env)
    obs = env.get_obs()
    quality["instruction"] = instruction = str(obs["instruction"])
    if episode.reject:
        quality.update(accepted=False, reason=episode.reject, **episode.summary())
        sidecar.write_text(json.dumps(quality, indent=2) + "\n")
        env.success[0], env.end_flag[0] = False, True
        return
    keys = state_keys(env)
    rm = env.robot_manager
    joints = {rm.process_name(r.arm_name): list(r.arm_joints_name) for r in rm.robot_list if r.type == "target"}
    hand_joints = {
        rm.process_name(r.gripper_name): list(r.gripper_joints_name)
        for r in rm.robot_list
        if r.type == "target" and len(getattr(r, "gripper_joints_name", ())) > 1
    }
    body_keys = [k for k in keys if k in BODY_RECORD_KEYS]
    init_body = {k: np.asarray(obs["state"][k], dtype=float).reshape(-1).tolist() for k in body_keys}
    states, actions = {k: [] for k in keys}, {k: [] for k in keys}
    trajectory = cfg.get("trajectory", True)
    if trajectory:
        from XPolicyLab.utils.process_data import encode_image_bit
    cameras = [c for c in CAMERAS if c in obs["vision"]]  # those the env cfg renders (observation.vision.cameras)
    images = {camera: [] for camera in cameras if trajectory}
    record_depth_cfg = cfg.get("record_depth", None)
    depth_cams_cfg = cfg.get("depth_cameras")
    if record_depth_cfg is None:
        has_obs_depth = any(
            isinstance(obs["vision"].get(c), Mapping)
            and (obs["vision"][c].get("depth") is not None or obs["vision"][c].get("approximate_depth") is not None)
            for c in cameras
        )
        depth_recorder = (
            DepthRecorder(cameras, depth_cams_cfg, strict=False) if (trajectory and has_obs_depth) else None
        )
    elif record_depth_cfg:
        depth_recorder = DepthRecorder(cameras, depth_cams_cfg, strict=True) if trajectory else None
    else:
        depth_recorder = None

    record_aff_cfg = cfg.get("record_affordance_2d", None)
    obs_aff_cams = [
        c
        for c in cameras
        if isinstance(obs["vision"].get(c), Mapping) and isinstance(obs["vision"][c].get("affordance"), Mapping)
    ]
    if record_aff_cfg is None:
        if trajectory and obs_aff_cams:
            aff_cfg = dict(cfg)
            aff_cfg.setdefault("affordance_2d_cameras", obs_aff_cams)
            affordance_2d_recorder = Affordance2DRecorder(env, episode, aff_cfg, layout_id, all_cameras=cameras)
        else:
            affordance_2d_recorder = None
    elif record_aff_cfg:
        affordance_2d_recorder = (
            Affordance2DRecorder(env, episode, cfg, layout_id, all_cameras=cameras) if trajectory else None
        )
    else:
        affordance_2d_recorder = None

    probe = motion.Probe(env)
    stable, last = 0, None
    to_end = getattr(plugin, "RECORD_TO_END", False)
    doomed = early_abort.for_episode(plugin, episode) if cfg.get("early_abort", True) else None
    for action in itertools.chain(episode.actions(), itertools.repeat(None, plugin.TAIL)):
        if action is None and last is None:
            raise RuntimeError("Expert produced no actions")
        action = last = action if action is not None else last
        for key in keys:
            if key not in action:
                raise KeyError(f"Expert action missing required state key {key!r} (wholebody env requires body keys)")
            states[key].append(np.asarray(obs["state"][key], dtype=np.float32).reshape(-1))
            actions[key].append(np.asarray(action[key], dtype=np.float32).reshape(-1))
        probe(obs)
        if depth_recorder is not None:
            depth_recorder.record(obs)
        if affordance_2d_recorder is not None:
            affordance_2d_recorder.record(env, obs, episode)
        for camera in images:
            images[camera].append(encode_image_bit(np.asarray(obs["vision"][camera]["color"], dtype=np.uint8)))
        env.take_action(action)
        obs = env.get_obs()
        stable = stable + 1 if episode.step() else 0
        frames = len(states[keys[0]])
        if (stable >= plugin.HOLD and not to_end) or not env.success[0] or frames >= env.step_lim:
            break
        if doomed and not stable and (reason := doomed(episode)):
            quality["aborted"] = dict(reason=reason, frame=frames)
            break
    if not all(np.isfinite(np.concatenate(rows)).all() for rows in (*states.values(), *actions.values())):
        raise ValueError("Non-finite state or action")
    accepted = stable >= plugin.HOLD and bool(env.success[0])
    probed, limits = probe.privileged(), probe.limits()
    keep_rejected_hdf5 = bool(cfg.get("keep_rejected_hdf5", False))
    save_rejected_privileged = bool(cfg.get("save_rejected_privileged", True))
    write_hdf5 = bool(trajectory and (accepted or keep_rejected_hdf5))
    depth_data, depth_meta = depth_recorder.payload(frames) if (write_hdf5 and depth_recorder is not None) else ({}, {})
    aff2d_data, aff2d_meta = (
        affordance_2d_recorder.payload(frames) if (write_hdf5 and affordance_2d_recorder is not None) else ({}, {})
    )
    body_meta = {}
    if body_keys:
        body_meta["init_body_state"] = init_body
        body_meta["body_joint_names"] = {
            k: (["torso_Joint", "Yaw_Joint"] if k == "waist_joint_state" else ["LZ_mt_Joint", "LZ_it_Joint"])
            for k in body_keys
        }
    if hand_joints:
        body_meta["hand_joint_names"] = hand_joints
    if depth_meta:
        body_meta["depth"] = depth_meta
    if aff2d_meta:
        body_meta["affordance_2d"] = aff2d_meta
    quality.update(
        accepted=accepted,
        frames=frames,
        stable_frames=stable,
        task_success=bool(env.task_success[0]),
        cameras=list(images),
        state_keys=keys,
        joint_names=joints,
        joint_limits=limits,
        motion=motion.summarize(states, actions, cfg["fps"], joints, limits, probed),
        trajectory_written=write_hdf5,
        **body_meta,
        **episode.summary(),
    )
    if write_hdf5:
        info = dict(frequency=cfg["fps"], env_cfg=cfg["env_cfg"], physics_dt=env.dt, layout=layout_file.name)
        write_trajectory(
            path,
            instruction,
            info,
            states,
            actions,
            images,
            episode.privileged() | probed,
            depths=depth_data or None,
            affordance_2d=aff2d_data or None,
        )
    elif trajectory and save_rejected_privileged:
        write_privileged(path.with_suffix(".npz"), episode.privileged() | probed)
    sidecar.write_text(json.dumps(quality, indent=2) + "\n")
    env.success[0], env.end_flag[0] = accepted, True
    aborted = f" aborted={quality['aborted']['reason']}" if "aborted" in quality else ""
    print(f"[datagen] {path.name}: accepted={accepted} frames={frames} stable={stable}{aborted}", flush=True)


def eval_one_episode(TASK_ENV, model_client):
    """Policy entry for main.py --policy_module scripts.datagen.collect."""
    cfg = json.loads(os.environ.get(ENV_KEY) or os.environ[LEGACY_ENV_KEY])
    record_episode(TASK_ENV, importlib.import_module(cfg["plugin"]), Path(cfg["output"]), cfg)
    if cfg.get("stop_file") and Path(cfg["stop_file"]).exists():
        TASK_ENV.seed_manager.idx = TASK_ENV.seed_manager.ed_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--seed", type=int, required=True, help="layout seed folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-ids", default="", help="comma-separated; default: every layout in the folder")
    parser.add_argument("--stop-file", type=Path, help="finish the current episode, then stop, once this exists")
    parser.add_argument("--no-trajectory", action="store_true", help="write only the sidecars")
    parser.add_argument(
        "--keep-rejected-hdf5",
        action="store_true",
        help="write full image HDF5 even for rejected episodes (default: sidecar + privileged .npz only)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="dynamically claim layout_ids from a flock-protected pool in --output instead of static assignment",
    )
    parser.add_argument(
        "--claim-pool",
        type=Path,
        help="explicit path to a flock-protected dynamic layout_id pool file (implies --dynamic)",
    )
    parser.add_argument("--plugin", help="plugin module under scripts/datagen/plugins (default: the task's)")
    parser.add_argument("--expert", help="expert version of the task, e.g. v2 (scripts/datagen/experts.json)")
    parser.add_argument("--no-early-abort", action="store_true", help="run every episode to its end (early_abort.py)")
    parser.add_argument(
        "--record-depth",
        action="store_true",
        help="enable depth capture in obs['vision'] and record vision/<camera>/depths (+ camera matrices) to HDF5",
    )
    parser.add_argument(
        "--depth-cameras",
        default="",
        help="comma-separated cameras for --record-depth (default: all rendered cameras)",
    )
    parser.add_argument(
        "--record-affordance-2d",
        "--record-affordance",
        dest="record_affordance_2d",
        action="store_true",
        help="enable 2D affordance in obs['vision'] and record affordance_2d/<camera>/ to HDF5",
    )
    parser.add_argument(
        "--affordance-2d-cameras",
        "--affordance-cameras",
        dest="affordance_2d_cameras",
        default=",".join(AFFORDANCE_2D_DEFAULT_CAMERAS),
        help="comma-separated cameras for --record-affordance-2d (default: cam_head)",
    )
    parser.add_argument(
        "--affordance-2d-heatmap-size",
        default=f"{AFFORDANCE_2D_DEFAULT_HEATMAP_HW[1]}x{AFFORDANCE_2D_DEFAULT_HEATMAP_HW[0]}",
        help="compact heatmap WxH resolution for --record-affordance-2d (default: 64x48)",
    )
    parser.add_argument(
        "--affordance-2d-top-k",
        type=int,
        default=AFFORDANCE_2D_DEFAULT_TOP_K,
        help=f"max 2D grasp keypoints stored per frame (default: {AFFORDANCE_2D_DEFAULT_TOP_K})",
    )
    args, remaining = parser.parse_known_args()
    try:
        if args.expert:
            args.plugin = experts.resolve(args.task, args.expert, args.plugin)["plugin"]
        expert = experts.record(args.task, args.expert, args.plugin or args.task)
    except experts.ExpertError as err:
        sys.exit(f"[datagen] {err}")
    if expert is None:
        print(f"[datagen] {args.task} has no registered expert version (scripts/datagen/experts.py add)", flush=True)
    elif expert["drift"]:
        print(f"[datagen] code differs from {args.task} {expert['version']}: {', '.join(expert['drift'])}", flush=True)
    files = layout_files(args.env_cfg, args.seed, args.task)
    wanted = [int(i) for i in args.layout_ids.split(",") if i.strip()] or list(range(len(files)))
    args.output.mkdir(parents=True, exist_ok=True)
    todo = [i for i in wanted if not (args.output / f"episode_{i:06d}.json").exists()]
    if not todo:
        print(f"[datagen] nothing left to record in {args.output}")
        return
    pool_path = args.claim_pool or (args.output / ".claim_pool.json" if args.dynamic else None)
    if pool_path is not None:
        pool_path = pool_path.resolve()
        pool = LayoutPool(pool_path, args.output, stop_file=args.stop_file)
        pool.ensure_initialized(todo)
        if not pool.has_pending():
            print(f"[datagen] no unclaimed layouts left in {pool_path}")
            return
    plugin = importlib.import_module(plugin_name(args.plugin or args.task))
    env_cfg = load_yaml(ROOT / "env_cfg" / f"{args.env_cfg}.yml")
    hm_parts = args.affordance_2d_heatmap_size.lower().split("x")
    if len(hm_parts) != 2 or not all(p.strip().isdigit() for p in hm_parts):
        parser.error(f"--affordance-2d-heatmap-size must be WxH (e.g. 64x48), got {args.affordance_2d_heatmap_size!r}")
    hm_w, hm_h = int(hm_parts[0]), int(hm_parts[1])
    if hm_w <= 0 or hm_h <= 0:
        parser.error(f"--affordance-2d-heatmap-size must be positive, got {args.affordance_2d_heatmap_size!r}")
    if args.affordance_2d_top_k < 1:
        parser.error(f"--affordance-2d-top-k must be >= 1, got {args.affordance_2d_top_k!r}")
    depth_sel = parse_camera_selection(args.depth_cameras)
    depth_cams = [c.strip() for c in args.depth_cameras.split(",") if c.strip()] if isinstance(depth_sel, set) else []
    env_depth_sel = parse_camera_selection(os.environ.get("OPENFLYWHEEL_COLLECT_DEPTH", ""))
    record_depth = True if (args.record_depth or env_depth_sel) else None
    if depth_sel is None and isinstance(env_depth_sel, set):
        depth_cams = sorted(env_depth_sel)
    aff_cli_sel = parse_camera_selection(args.affordance_2d_cameras)
    env_aff_sel = parse_camera_selection(os.environ.get("OPENFLYWHEEL_COLLECT_AFFORDANCE", ""))
    record_aff = True if (args.record_affordance_2d or env_aff_sel) else None
    if args.record_affordance_2d:
        if aff_cli_sel is True:
            aff2d_cams = None
        elif isinstance(aff_cli_sel, set):
            aff2d_cams = [c.strip() for c in args.affordance_2d_cameras.split(",") if c.strip()]
        else:
            aff2d_cams = list(AFFORDANCE_2D_DEFAULT_CAMERAS)
    elif env_aff_sel is True:
        aff2d_cams = None
    elif isinstance(env_aff_sel, set):
        aff2d_cams = sorted(env_aff_sel)
    else:
        aff2d_cams = list(AFFORDANCE_2D_DEFAULT_CAMERAS)
    cfg = dict(
        plugin=plugin.__name__,
        output=str(args.output.resolve()),
        env_cfg=args.env_cfg,
        fps=env_cfg["observation"]["collect_freq"],
        stop_file=str(args.stop_file.resolve()) if args.stop_file else None,
        trajectory=not args.no_trajectory,
        keep_rejected_hdf5=args.keep_rejected_hdf5,
        early_abort=not args.no_early_abort,
        record_depth=record_depth,
        depth_cameras=depth_cams or None,
        record_affordance_2d=record_aff,
        affordance_2d_cameras=aff2d_cams,
        affordance_2d_heatmap_hw=[hm_h, hm_w],
        affordance_2d_top_k=args.affordance_2d_top_k,
        revision=json.loads(os.environ.get("OPENFLYWHEEL_REVISION") or os.environ.get("ROBODOJO_REVISION", "null"))
        or code_revision(),
        expert=expert,
    )
    os.environ[ENV_KEY] = os.environ[LEGACY_ENV_KEY] = json.dumps(cfg)
    if not args.no_trajectory:
        if args.record_depth:
            os.environ["OPENFLYWHEEL_COLLECT_DEPTH"] = ",".join(depth_cams) if depth_cams else "1"
            os.environ.setdefault("OPENFLYWHEEL_COLLECT_CAMERA_MATRICES", "1")
        if args.record_affordance_2d:
            os.environ["OPENFLYWHEEL_COLLECT_AFFORDANCE"] = ",".join(aff2d_cams) if aff2d_cams else "1"
            os.environ["OPENFLYWHEEL_AFFORDANCE_HEATMAP_SIZE"] = f"{hm_w}x{hm_h}"
            os.environ["OPENFLYWHEEL_AFFORDANCE_TOP_K"] = str(args.affordance_2d_top_k)
            os.environ.setdefault("OPENFLYWHEEL_COLLECT_CAMERA_MATRICES", "1")
    os.environ.setdefault("OPENFLYWHEEL_NO_VIDEO", "1")
    os.environ["EVAL_NUM"] = str(len(todo))
    if pool_path is not None:
        os.environ["OPENFLYWHEEL_CLAIM_POOL"] = str(pool_path)
    from scripts.g1d import scripted_demo

    sys.argv = [sys.argv[0], "--env_cfg_type", args.env_cfg, "--task_name", args.task, "--seed", str(args.seed)]
    sys.argv += ["--layout_ids", ",".join(map(str, todo)), "--allow_train_layouts"]
    if pool_path is not None:
        sys.argv += ["--claim_pool", str(pool_path)]
    sys.argv += remaining
    scripted_demo.main(policy_module="scripts.datagen.collect")


if __name__ == "__main__":
    main()
