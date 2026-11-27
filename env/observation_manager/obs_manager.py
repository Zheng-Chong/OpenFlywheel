from collections.abc import Mapping
from copy import deepcopy
import os
from typing import List

import numpy as np

from env.description_manager.desc_manager import DescManager
from env.observation_manager import imaging
from env.observation_manager.vision_utils import (
    DEFAULT_AFFORDANCE_TOP_K,
    as_numpy as _as_numpy,
    env_truthy as _env_truthy,
    is_usable_intrinsics as _is_usable_intrinsics,
    merge_camera_selections,
    parse_camera_selection,
    parse_heatmap_hw as _parse_heatmap_hw,
    resolve_affordance_target,
    selection_matches as _selection_matches,
)


class ObsManager:
    ANNOTATORS_TO_COLLECT = {
        "rgb": "color",
        "depth": "depth",
        "distance_to_image_plane": "distance_to_image_plane",
    }

    def __init__(
        self,
        obs_config,
        num_envs,
        dt,
        task_name,
        description_cfg,
        seeds_per_env: List[int] = None,
    ):
        self.obs_config = obs_config
        self.num_envs = num_envs
        self.dt = dt
        self.task_name = task_name
        self.description_cfg = description_cfg
        self.desc_manager = DescManager(
            num_envs=self.num_envs,
            description_cfg=self.description_cfg,
            desc_type="seen",
            seeds_per_env=seeds_per_env,
        )
        self.robot_cfg = obs_config.get("robot", {})
        # Gripper opening in obs["state"]: the last command ("command", default) or the finger joints ("measured");
        # obs["action"] always holds the command.
        self.ee_state = self.robot_cfg.get("ee_state", "command")
        # body_states: true adds the robot's body (G1-D waist, lift, chassis) to obs["state"] / obs["action"].
        if self.ee_state not in ("command", "measured"):
            raise ValueError(f"observation.robot.ee_state must be 'command' or 'measured', got {self.ee_state!r}")
        self.collect_freq = self.obs_config.get("collect_freq", 0)
        # Frames the SDG dispatcher must deliver after the physics step before
        # images are read. One is exact with the zero-delay Kit settings that
        # the entrypoints enable; render_sync adds one more if they are absent.
        self.capture_render_passes = int(self.obs_config.get("capture_render_passes", 1))
        self.capture_render_max_passes = int(self.obs_config.get("capture_render_max_passes", 16))
        if self.collect_freq > 0:
            self.collect_interval = 1.0 / (self.dt * self.collect_freq)
        else:
            self.collect_interval = 0.0
        if not self.collect_interval.is_integer():
            raise ValueError("Collect Interval must be integer!")
        vision_cfg = self.obs_config.get("vision", {})
        self.collect_approximate_depth = vision_cfg.get("approximate_depth", True)
        self._cfg_depth_cameras = parse_camera_selection(vision_cfg.get("depth", False))
        self._env_depth_cameras = parse_camera_selection(os.environ.get("OPENFLYWHEEL_COLLECT_DEPTH", ""))
        self.collect_depth = bool(self._cfg_depth_cameras or self._env_depth_cameras)
        env_matrices = _env_truthy("OPENFLYWHEEL_COLLECT_CAMERA_MATRICES")
        self.collect_intrinsic_matrix = bool(vision_cfg.get("intrinsic_matrix", False) or env_matrices)
        self.collect_extrinsic_matrix = bool(vision_cfg.get("extrinsic_matrix", False) or env_matrices)
        self.collect_shape = vision_cfg.get("shape", True)
        aff_cfg = vision_cfg.get("affordance", False)
        aff_dict = aff_cfg if isinstance(aff_cfg, Mapping) else {}
        self._aff_cameras = merge_camera_selections(
            parse_camera_selection(aff_cfg),
            parse_camera_selection(os.environ.get("OPENFLYWHEEL_COLLECT_AFFORDANCE", "")),
        )
        self.collect_affordance = bool(self._aff_cameras)
        hm_env = os.environ.get("OPENFLYWHEEL_AFFORDANCE_HEATMAP_SIZE", "").strip()
        self._aff_heatmap_hw = _parse_heatmap_hw(hm_env or aff_dict.get("heatmap_hw"))
        top_k_env = os.environ.get("OPENFLYWHEEL_AFFORDANCE_TOP_K", "").strip()
        self._aff_top_k = int(top_k_env) if top_k_env else int(aff_dict.get("top_k", DEFAULT_AFFORDANCE_TOP_K))
        if self._aff_top_k < 1:
            raise ValueError(f"affordance top_k must be >= 1, got {self._aff_top_k}")
        lbl_env = os.environ.get("OPENFLYWHEEL_AFFORDANCE_TARGET_LABEL", "").strip()
        self._aff_target_label = lbl_env or aff_dict.get("target_label")
        self._aff_path = aff_dict.get("affordance_path")
        self._aff_verified_only = bool(aff_dict.get("verified_only", False))
        self._aff_per_part_heatmaps = bool(aff_dict.get("per_part_heatmaps", False))
        self._affordance_targets = {}
        self.frames = [0] * self.num_envs
        self.imaging = [None] * self.num_envs

    def _wants_depth_for(self, camera_name: str) -> bool:
        return _selection_matches(self._cfg_depth_cameras, camera_name) or _selection_matches(
            self._env_depth_cameras, camera_name
        )

    def _wants_affordance_for(self, camera_name: str) -> bool:
        return _selection_matches(self._aff_cameras, camera_name)

    def initialize(self, env):
        self.env = env
        self._affordance_targets.clear()
        if hasattr(env, "robot_manager"):
            self.robot_manager = env.robot_manager
        else:
            self.robot_manager = None
        if hasattr(env, "capture_manager"):
            self.capture_manager = env.capture_manager
        else:
            self.capture_manager = None
        if hasattr(env, "camera_manager"):
            self.camera_manager = env.camera_manager
        else:
            self.camera_manager = None

        self.initialize_instruction()

    def initialize_instruction(self):
        self.desc_manager.initialize(self.env)
        self.instruction = self.desc_manager.get_one_description()

    def reset(self):
        self.desc_manager.reset()
        self.instruction = self.desc_manager.get_one_description()
        self.frames = [0] * self.num_envs  # frames captured this episode, per env: the imaging noise draw
        self._affordance_targets.clear()

    def render_for_capture(self):
        """Render until the camera buffers hold the current physics state.

        Call this instead of a bare ``env.render()`` right before ``get_obs``.
        """
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras

        wait_for_latest_cameras(
            self.env,
            capture_manager=self.capture_manager,
            min_passes=self.capture_render_passes,
            max_passes=self.capture_render_max_passes,
        )

    def get_obs(self, env_idx_list=None):  # batch
        if env_idx_list is None:
            env_idx_list = range(self.num_envs)
        obs = dict()
        for env_idx in env_idx_list:
            obs[env_idx] = dict()
            obs[env_idx]["vision"] = dict()
            obs[env_idx]["state"] = dict()
            obs[env_idx]["action"] = dict()
            obs[env_idx]["data_format_version"] = "v1.0"
            obs[env_idx]["additional_info"] = {
                "frequency": self.collect_freq,
            }
            obs[env_idx]["instruction"] = self.instruction[env_idx]
        if self.camera_manager is not None and self.capture_manager is not None:
            layout_entries = getattr(self.env, "layout_entries", None)
            self.imaging = layout_entries("Imaging") if layout_entries else [None] * self.num_envs
            try:
                data = self.capture_manager.step(env_ids=env_idx_list)
                for ith in range(self.camera_manager.num_cams):
                    cam_data = data[ith]
                    for annotator_name, env_list in cam_data.items():
                        collect_name = self.ANNOTATORS_TO_COLLECT[annotator_name]
                        for idx, env_idx in enumerate(env_idx_list):
                            camera_name = self.camera_manager.camera_names[env_idx][ith]
                            if camera_name not in obs[env_idx]["vision"]:
                                obs[env_idx]["vision"][camera_name] = dict()
                            obs[env_idx]["vision"][camera_name][collect_name] = env_list[idx]["data"]

                            if annotator_name == "rgb":
                                obs[env_idx]["vision"][camera_name][collect_name] = imaging.perturb(
                                    obs[env_idx]["vision"][camera_name][collect_name][:, :, :3],
                                    self.imaging[env_idx],
                                    camera_name,
                                    self.frames[env_idx],
                                )
                                if self.collect_shape:
                                    obs[env_idx]["vision"][camera_name]["shape"] = obs[env_idx]["vision"][camera_name][
                                        collect_name
                                    ].shape

            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                print(stack_trace)
                print("[get_obs] Camera observation capture failed.")
                raise e

            for env_idx in env_idx_list:
                self.frames[env_idx] += 1
            for env_idx in env_idx_list:
                for camera_name, cam_data in obs[env_idx]["vision"].items():
                    raw = cam_data.pop("distance_to_image_plane", None)
                    if raw is None and "depth" in cam_data:
                        raw = cam_data.pop("depth")
                    if raw is not None:
                        raw_np = _as_numpy(raw)
                        if raw_np.ndim >= 1 and raw_np.shape[-1] == 1:
                            raw_np = raw_np.squeeze(-1)
                        if self.collect_approximate_depth:
                            safe = np.nan_to_num(raw_np, nan=0.0, posinf=65.535, neginf=0.0)
                            cam_data["approximate_depth"] = np.clip(safe * 1000, 0, 65535).astype(np.uint16)
                        if self._wants_depth_for(camera_name):
                            cam_data["depth"] = np.array(raw_np, dtype=np.float32, copy=True)

            if self.collect_intrinsic_matrix or self.collect_extrinsic_matrix:
                for ith in range(self.camera_manager.num_cams):
                    for idx, env_idx in enumerate(env_idx_list):
                        camera_name = self.camera_manager.camera_names[env_idx][ith]
                        if camera_name not in obs[env_idx]["vision"]:
                            continue
                        if self.collect_intrinsic_matrix:
                            obs[env_idx]["vision"][camera_name]["intrinsic_matrix"] = (
                                self.camera_manager.get_camera_intrinsics(ith, env_idx)
                            )
                        if self.collect_extrinsic_matrix:
                            obs[env_idx]["vision"][camera_name]["extrinsic_matrix"] = (
                                self.camera_manager.get_camera_extrinsics(ith, env_idx)
                            )

            if self.collect_affordance:
                self._populate_affordance(obs, env_idx_list)

        if self.robot_manager is not None:
            try:
                for robot in self.robot_manager.robot_list:
                    if robot.type != "target":
                        continue
                    if self.robot_cfg["joint_states"]:
                        if robot.robot_type == "arm":
                            joints = self.robot_manager.get_joint(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            if robot.robot_type == "arm":
                                obs[env_idx]["state"][f"{robot.arm_name}_joint_state"] = joints[env_idx]
                                obs[env_idx]["action"][f"{robot.arm_name}_joint_state"] = joints[env_idx]

                    force = self.robot_manager.get_gripper_force(robot, env_idx_list)
                    if force is not None:
                        for env_idx in env_idx_list:
                            obs[env_idx].setdefault("gripper_force", {})[robot.gripper_name] = force[env_idx]
                    contact_getter = getattr(self.robot_manager, "get_tactile_contact", None)
                    contact = contact_getter(robot, env_idx_list) if contact_getter is not None else None
                    if contact is not None:
                        for env_idx in env_idx_list:
                            obs[env_idx].setdefault("tactile_contact", {})[robot.gripper_name] = contact[env_idx]

                    if self.robot_cfg["world_ee_state"]:
                        endpose = self.robot_manager.get_real_endpose(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            if self.robot_manager.target_arm_nums == 1:
                                obs[env_idx]["state"]["ee_pose"] = endpose[env_idx]
                            else:
                                name = robot.arm_name.split("_")[0]
                                obs[env_idx]["state"][f"{name}_ee_pose"] = endpose[env_idx]

                    if self.robot_cfg.get("body_states", False) and hasattr(robot, "body_dims"):
                        body = self.robot_manager.get_body_state(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            obs[env_idx]["state"].update(body[env_idx])
                            prev = self.robot_manager.control_manager.prev_control[env_idx]
                            for name in robot.body_dims:
                                if name in prev:
                                    obs[env_idx]["action"][name] = list(prev[name]["position"])

                if hasattr(self.robot_manager, "control_manager"):
                    measured = (
                        self._measured_grippers(env_idx_list)
                        if self.ee_state == "measured" and self.robot_cfg["joint_states"]
                        else {}
                    )
                    for env_idx in env_idx_list:
                        for (
                            key,
                            data,
                        ) in self.robot_manager.control_manager.prev_control[env_idx].items():
                            position = data["position"]
                            if key.endswith("ee_joint_state"):
                                end_effector_name = self.robot_manager.restore_name(deepcopy(key))
                                robot = self.robot_manager.get_robot_by_gripper_name(end_effector_name)
                                if robot.ee_type == "gripper":
                                    val = self._to_gripper_scale(robot, position[0])
                                    obs[env_idx]["action"][key] = [val]
                                    if self.robot_cfg["joint_states"]:
                                        obs[env_idx]["state"][key] = [
                                            measured.get(end_effector_name, {}).get(env_idx, val)
                                        ]
                                else:
                                    obs[env_idx]["action"][key] = position
                                    if self.robot_cfg["joint_states"]:
                                        obs[env_idx]["state"][key] = position

            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                print(stack_trace)
                print("[get_obs] Robot observation capture failed.")
                raise e
        return obs

    @staticmethod
    def _to_gripper_scale(robot, q):
        """Finger joint position -> the 0..1 scale of gripper commands (inverse of EvalEnv.take_action_batch)."""
        lo, hi = robot.gripper_scale
        return (q - lo) / (hi - lo) if robot.gripper_move["sign"] == 1 else (hi - q) / (hi - lo)

    def _measured_grippers(self, env_idx_list):
        """{gripper_name: {env_idx: opening}} from the finger joints on the command scale: the mean over the fingers,
        the mimic finger mapped back onto the base one (Dex1 fingers are separate joints and stop on an object apart)."""
        result = {}
        for robot in self.robot_manager.robot_list:
            if robot.type != "target" or robot.ee_type != "gripper":
                continue
            joints = self.robot_manager.get_end_effector_real_val(robot=robot, env_idx_list=env_idx_list)
            mimic, gain, offset = robot.gripper_move["mimic"]
            result[robot.gripper_name] = {}
            for env_idx in env_idx_list:
                base = [
                    (q - offset) / gain if n == mimic else q for n, q in zip(robot.gripper_joints_name, joints[env_idx])
                ]
                result[robot.gripper_name][env_idx] = float(self._to_gripper_scale(robot, np.mean(base)))
        return result

    def _env_origin(self, env_idx: int) -> np.ndarray:
        env = getattr(self, "env", None)
        sm = getattr(env, "scene_manager", None)
        origins = getattr(sm, "env_origins", None)
        if origins is None:
            origins = getattr(env, "env_origins", None)
        if origins is not None and env_idx < len(origins):
            return _as_numpy(origins[env_idx]).astype(np.float64).reshape(3)
        return np.zeros(3, dtype=np.float64)

    def _get_affordance_target(self, env_idx: int):
        env = getattr(self, "env", None)
        if env is None:
            return None
        seed_map = getattr(env, "current_env_seed_map", None)
        layout_id = seed_map.get(env_idx) if isinstance(seed_map, Mapping) else None
        cached = self._affordance_targets.get(env_idx)
        if cached is not None and cached[0] == layout_id:
            return cached[1]
        target = resolve_affordance_target(
            env,
            env_idx=env_idx,
            layout_id=layout_id,
            preferred_label=self._aff_target_label,
            affordance_path=self._aff_path,
        )
        self._affordance_targets[env_idx] = (layout_id, target)
        return target

    @staticmethod
    def _get_object_pose(env, env_idx: int, target: dict):
        sm = getattr(env, "scene_manager", None)
        lm = getattr(sm, "layout_manager", None)
        if lm is not None and hasattr(lm, "get_instance_pose"):
            try:
                pos, quat = lm.get_instance_pose(env_idx, label=target["target_label"])
                if pos is not None and quat is not None:
                    return _as_numpy(pos).astype(np.float64).reshape(3), _as_numpy(quat).astype(np.float64).reshape(4)
            except Exception:
                pass
        return target.get("fallback_pose")

    @staticmethod
    def _camera_hw(cam_obs: Mapping):
        color = cam_obs.get("color")
        if color is not None:
            arr = np.asarray(color)
            if arr.ndim >= 2:
                return int(arr.shape[0]), int(arr.shape[1])
        shape = cam_obs.get("shape")
        if shape is not None and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        depth = cam_obs.get("depth")
        if depth is not None:
            arr = np.asarray(depth)
            if arr.ndim >= 2:
                return int(arr.shape[0]), int(arr.shape[1])
        return None

    def _populate_affordance(self, obs: dict, env_idx_list) -> None:
        env = getattr(self, "env", None)
        if env is None or self.camera_manager is None:
            return
        from scripts.objects.affordance_2d import project_affordance_2d

        selected_grasp_world = getattr(env, "selected_grasp_world", None)
        for env_idx in env_idx_list:
            sel_grasp_env = (
                selected_grasp_world.get(env_idx) if isinstance(selected_grasp_world, Mapping) else selected_grasp_world
            )
            target = self._get_affordance_target(env_idx)
            if target is None:
                continue
            pose = self._get_object_pose(env, env_idx, target)
            if pose is None:
                continue
            obj_pos, obj_quat = pose
            origin = self._env_origin(env_idx)
            cam_names = self.camera_manager.camera_names[env_idx]
            for ith in range(self.camera_manager.num_cams):
                camera_name = cam_names[ith]
                if not self._wants_affordance_for(camera_name):
                    continue
                cam_obs = obs[env_idx]["vision"].get(camera_name)
                if not isinstance(cam_obs, dict):
                    continue
                hw = self._camera_hw(cam_obs)
                if hw is None:
                    continue
                h, w = hw
                K = cam_obs.get("intrinsic_matrix")
                ext = cam_obs.get("extrinsic_matrix")
                conv = cam_obs.get("camera_convention", "usd")
                if K is None:
                    try:
                        K = self.camera_manager.get_camera_intrinsics(ith, env_idx)
                    except Exception:
                        K = None
                if ext is None:
                    try:
                        ext = self.camera_manager.get_camera_extrinsics(ith, env_idx)
                    except Exception:
                        ext = None
                if K is None or ext is None:
                    continue
                k_arr = _as_numpy(K).astype(np.float64)
                ext_arr = _as_numpy(ext).astype(np.float64)
                if ext_arr.shape != (4, 4) or not np.isfinite(ext_arr).all() or not _is_usable_intrinsics(k_arr):
                    continue
                ext_rel = ext_arr.copy()
                ext_rel[:3, 3] -= origin
                depth_m = cam_obs.get("depth")
                if depth_m is None and cam_obs.get("approximate_depth") is not None:
                    depth_m = np.asarray(cam_obs["approximate_depth"], dtype=np.float32) * 1e-3
                res = project_affordance_2d(
                    target["affordance"],
                    obj_pos,
                    obj_quat,
                    k_arr,
                    ext_rel,
                    w,
                    h,
                    scale=target["scale"],
                    depth=depth_m,
                    gripper_types=target["gripper_types"],
                    verified_only=self._aff_verified_only,
                    per_part_heatmaps=self._aff_per_part_heatmaps,
                    max_keypoints=self._aff_top_k,
                    selected_grasp_world=sel_grasp_env,
                    camera_convention=conv,
                    heatmap_hw=self._aff_heatmap_hw,
                )
                aff_payload = {
                    "target_label": target["target_label"],
                    "asset_id": target["asset_id"],
                    "scale": target["scale"],
                    "gripper_types": list(res.gripper_types),
                    "heatmap_hw": [int(res.heatmap_uint8.shape[0]), int(res.heatmap_uint8.shape[1])],
                    "heatmap": res.heatmap,
                    "heatmap_uint8": res.heatmap_uint8,
                    "object_bbox_xyxy": res.object_bbox_xyxy,
                    "object_bbox_xyxy_norm": res.object_bbox_xyxy_norm,
                    "affordance_bbox_xyxy": res.affordance_bbox_xyxy,
                    "affordance_bbox_xyxy_norm": res.affordance_bbox_xyxy_norm,
                    "keypoints": res.keypoints,
                    "selected_grasp_2d": res.selected_grasp_2d,
                }
                if res.part_heatmaps:
                    aff_payload["part_heatmaps"] = res.part_heatmaps
                cam_obs["affordance"] = aff_payload
