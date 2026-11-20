"""Camera and capture managers of the MuJoCo backend: the cameras of the camera yml (and the robots' wrist cameras) as
MuJoCo cameras with the Isaac templates' field of view, rendered offscreen by mujoco.Renderer in the annotator layout
the observation manager reads (rgb: H x W x 4 uint8, distance_to_image_plane: H x W x 1 float32 metres)."""

from collections.abc import Sequence
import math
import os

import mujoco
import numpy as np
from omegaconf import OmegaConf

from env.camera_manager.camera_pose import (
    REAL_MAP,
    intrinsics,
    layout_offset,
    pose_matrix,
    sample_camera_pose,
    split_camera_config,
)
from env.seeding import seed_everywhere

DEPTH_ANNOTATORS = ("depth", "distance_to_image_plane")
RGB_ANNOTATORS = ("rgb", "rgba")


def _fovy(camera_type):
    """Vertical field of view (deg) matching Isaac's rendering: square pixels, fx from the horizontal aperture."""
    width, height = REAL_MAP[camera_type]["resolution"]
    fx = intrinsics(camera_type)[0, 0]
    return math.degrees(2 * math.atan(height / 2 / fx))


class MjCameraView:
    """`cameras[env][cam]` element: name, resolution and world pose (Isaac Camera / xform stand-in)."""

    def __init__(self, world, name, resolution, env_origin, clipping_range=(0.01, 10.0)):
        self.world = world
        self.name = self.prim_path = name
        self._resolution = resolution
        self.clipping_range = clipping_range  # the near plane hides the shell around a lens (g1d_head)
        self.env_origin = np.asarray(env_origin, dtype=np.float64)

    def get_local_pose(self):
        cam = self.world.cameras[self.name]
        return np.asarray(cam["pos"]), np.asarray(cam["quat"])

    def get_world_pose(self):
        model, data = self.world.model, self.world.data
        cam = model.camera(self.name).id
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.cam_xmat[cam])
        return data.cam_xpos[cam] + self.env_origin, quat


class MujocoCameraManager:
    def __init__(self, num_envs, camera_config, device, seeds_per_env: Sequence[int] | None = None):
        self.num_envs = num_envs
        self.num_cams = 1
        self.camera_config = camera_config
        self.init_config = camera_config
        self.device = device
        self.sim = None
        self.cameras = []
        self.cameras_xform = self.cameras
        self.camera_names = []
        self.camera_poses = []
        self.layout_offsets = None
        self._seeds_per_env = None
        self.update_env_seeds(seeds_per_env)

    def update_env_seeds(self, seeds):
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}.")
        self._seeds_per_env = seed_list

    def _set_env_seed(self, env_id):
        if self._seeds_per_env is None:
            return
        seed_everywhere(self._seeds_per_env[env_id])

    def initialize(self, sim):
        self.sim = sim
        self.create_cameras()
        if self._seeds_per_env:
            seed_everywhere(self._seeds_per_env[0])

    def create_cameras(self):
        self.camera_config, self.random_config = split_camera_config(self.init_config)
        self.cameras.clear()
        self.camera_names.clear()
        for env_id in range(self.num_envs):
            world = self.sim.worlds[env_id]
            origin = self.sim.scene.env_origins[env_id].cpu().numpy()
            self.cameras.append([])
            self.camera_names.append([])
            for camera_name, camera_config in self.camera_config.items():
                cam = camera_config.camera
                if cam.type not in REAL_MAP:
                    raise ValueError(f"Camera type {cam.type} not found in template map.")
                mount = cam.get("mount_link")
                world.cameras[camera_name] = {
                    "body": mount,
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                    "fovy": _fovy(cam.type),
                }
                template = REAL_MAP[cam.type]
                clipping = template.get("clipping_range", (0.01, 10.0))
                view = MjCameraView(world, camera_name, template["resolution"], origin, clipping)
                self.cameras[env_id].append(view)
                self.camera_names[env_id].append(camera_name)
            self.num_cams = len(self.camera_names[env_id])
        self.sim.compile()

    def post_init(self):
        pass

    def init_camera_pose(self, env_ids=None, use_randomization=True):
        env_id_iter = range(self.num_envs) if env_ids is None else [int(e) for e in env_ids]
        if len(self.camera_poses) != self.num_envs:
            self.camera_poses = [[] for _ in range(self.num_envs)]
        for env_id in env_id_iter:
            self._set_env_seed(env_id)
            self.camera_poses[env_id] = []
            world = self.sim.worlds[env_id]
            for camera_name, camera_config in self.camera_config.items():
                offset = layout_offset(self.layout_offsets, env_id, camera_name)
                pos, quat = sample_camera_pose(
                    camera_name, camera_config, self.random_config, offset, use_randomization
                )
                pos = np.asarray(pos, dtype=np.float64)
                quat = np.asarray(quat, dtype=np.float64)
                world.set_camera_pose(camera_name, pos, quat / np.linalg.norm(quat))
                self.camera_poses[env_id].append((pos, quat))

    def reset(self):
        self.init_camera_pose(env_ids=list(range(self.num_envs)))
        self.sim.sim_step()

    def get_camera_intrinsics(self, cam_id, env_id=0):
        return intrinsics(self.camera_config[self.camera_names[env_id][cam_id]].camera.type)

    def get_camera_extrinsics(self, cam_id, env_id):
        pos, quat = self.cameras[env_id][cam_id].get_world_pose()
        return pose_matrix(pos, quat)

    def destroy(self):
        self.cameras.clear()
        self.camera_names.clear()
        self.camera_poses = []


class MujocoCaptureManager:
    """Offscreen rendering of every camera; one renderer per env and resolution, rebuilt when the env's model is."""

    def __init__(self, num_envs, config, camera_manager, device):
        self.sim = None
        self.config = config
        self.camera_manager = camera_manager
        self.num_envs = num_envs
        self.device = device
        self.annotator_type = []
        self._renderers = {}

    def initialize(self, sim):
        self.sim = sim
        self.num_cams = self.camera_manager.num_cams

    def init_cameras(self):
        annotators = OmegaConf.to_container(self.config.annotator, resolve=True) if "annotator" in self.config else None
        self.annotator_type = []
        names = self.camera_manager.camera_names[0]
        for camera_name in names:
            capture = (annotators or {}).get(camera_name, (annotators or {}).get("common"))
            types = []
            if capture and capture.get("enabled", False):
                for key, setting in capture.items():
                    if key == "enabled":
                        continue
                    if setting["type"] not in RGB_ANNOTATORS + DEPTH_ANNOTATORS:
                        raise NotImplementedError(f"MuJoCo backend: annotator '{setting['type']}' is not supported")
                    types.append(setting["type"])
            self.annotator_type.append(types)

    def reset(self):
        self.init_cameras()

    def _renderer(self, env_id, width, height):
        world = self.sim.worlds[env_id]
        key = (env_id, width, height)
        entry = self._renderers.get(key)
        if entry is None or entry[0] != world.version:
            if entry is not None:
                entry[1].close()
            entry = (world.version, mujoco.Renderer(world.model, height=height, width=width))
            self._renderers[key] = entry
        return entry[1]

    def step(self, env_ids=None, cam_ids=None):
        """data[cam][annotator] = [{"data": array, "info": {}} per env in env_ids] (TiledCaptureManager.step)."""
        env_ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        cam_ids = list(range(self.camera_manager.num_cams)) if cam_ids is None else list(cam_ids)
        data = []
        for cam_id in cam_ids:
            cam_data = {name: [] for name in self.annotator_type[cam_id]}
            for env_id in env_ids:
                view = self.camera_manager.cameras[env_id][cam_id]
                width, height = view._resolution
                renderer = self._renderer(env_id, width, height)
                world = self.sim.worlds[env_id]
                near, far = view.clipping_range
                world.model.vis.map.znear = near / world.model.stat.extent
                world.model.vis.map.zfar = far / world.model.stat.extent
                for name in self.annotator_type[cam_id]:
                    renderer.update_scene(world.data, camera=view.name)
                    if name in DEPTH_ANNOTATORS:
                        renderer.enable_depth_rendering()
                        image = renderer.render().astype(np.float32)[..., None]
                        renderer.disable_depth_rendering()
                    else:
                        rgb = renderer.render()
                        image = np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
                    cam_data[name].append({"data": image, "info": {}})
            data.append(cam_data)
        return data

    def destroy(self):
        for _, renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self.annotator_type = []
        self.sim = None


def default_gl_backend():
    """MUJOCO_GL for offscreen rendering: EGL on headless Linux, the platform default elsewhere."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if os.name == "posix" and not os.environ.get("DISPLAY") and os.uname().sysname == "Linux":
        os.environ["MUJOCO_GL"] = "egl"
    return os.environ.get("MUJOCO_GL")
