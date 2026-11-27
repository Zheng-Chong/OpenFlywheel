"""
Tiled capture manager: initializes tiled render products and annotators for eval.
"""

from copy import deepcopy
from typing import List

from isaacsim.sensors.camera import Camera
from omegaconf import DictConfig, OmegaConf
from omni.replicator.core.scripts.annotators import Annotator
import torch

from env.camera_manager.camera_manager import CameraManager
from env.camera_manager.capture.camera_view import CameraView
from env.environment.isaac.isaac_rl_env import IsaacRLEnv
from env.observation_manager.vision_utils import requested_depth_cameras

try:  # IsaacLab 3: SimulationContext.render() no longer runs Kit's app loop; cameras pump it when they read
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
except ImportError:
    ensure_isaac_rtx_render_update = None


class TiledCaptureManager:
    """
    Manages tiled camera capture: initializes cameras, render products, and annotators, and handles video recording.
    """

    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        camera_manager: CameraManager,
        device: torch.device,
    ):
        self.sim: IsaacRLEnv = None
        self.config = config
        self.camera_manager = camera_manager
        self.num_envs = num_envs
        self.device = device
        self.tiled_render_products: List[str] = []  # all the render products
        self.annotator: List[
            List[List[Annotator]]
        ] = []  # Defines the types and devices of annotator. See yaml for camera for more detail
        self.annotator_type: List[List[List[str]]] = []
        self.annotator_device: List[List[List[str]]] = []
        self.cameras: List[List[Camera]] = camera_manager.cameras
        self.camera_names: List[List[str]] = camera_manager.camera_names
        self.tiled_cameras: List[CameraView] = []
        self.camera_prim_paths: List[List[str]] = []
        # Pre-allocated output buffers for each camera and annotator to reduce memory allocation
        # Format: {cam_id: {annotator_name: wp.array}}
        self._output_buffers: dict = {}

    def initialize(self, sim: IsaacRLEnv):
        """
        Initialize the capture manager.
        This method should be called before the simulation context is created.
        """
        if len(self.cameras) == 0:
            self.cameras = self.camera_manager.cameras
            self.camera_names = self.camera_manager.camera_names
        self.num_cams = len(self.cameras[0])
        self.sim = sim

    def init_cameras(self):
        """
        Initialize cameras based on the configuration.
        1. Initialize the cameras
        2. Get all render_product control
        3. Attach Annotator
        """
        self.camera_prim_paths.clear()
        for env_id in range(self.num_envs):
            self.camera_prim_paths.append([])
            for camera in self.cameras[env_id]:
                self.camera_prim_paths[env_id].append(camera.prim_path)

        for cam_id in range(self.num_cams):
            self.annotator.append([])
            self.annotator_type.append([])
            self.annotator_device.append([])

        config = OmegaConf.to_container(self.config, resolve=True)
        if "colorize_depth" in config:
            self.colorize_depth = config["colorize_depth"]
            config.pop("colorize_depth")
        if "default_frequency" in config:
            config.pop("default_frequency")

        self.config = OmegaConf.create(config)
        depth_sel = requested_depth_cameras(getattr(self, "_requested_depth_cameras", None))
        annotator_config = (
            OmegaConf.to_container(self.config.annotator, resolve=True) if "annotator" in self.config else None
        )
        if annotator_config is None and depth_sel is None:
            print("[TiledCaptureManager] No annotator enabled in config. Please check your config file.")
            return
        annotator_config = annotator_config or {}
        for cam_id, camera_name in enumerate(self.camera_names[0]):
            if camera_name in annotator_config:
                capture_config = annotator_config[camera_name]
            else:
                capture_config = annotator_config.get("common", None)  # check if there is common annotator config

            if capture_config and capture_config.get("enabled", False):
                annotators = deepcopy(capture_config)
                annotators.pop("enabled")
                for annotator_name, annotator_setting in annotators.items():
                    type_annotator = annotator_setting["type"]
                    if type_annotator not in self.annotator_type[cam_id]:
                        self.annotator_type[cam_id].append(type_annotator)
            wants_depth = depth_sel is True or (isinstance(depth_sel, set) and camera_name in depth_sel)
            if wants_depth and not any(t in ("depth", "distance_to_image_plane") for t in self.annotator_type[cam_id]):
                self.annotator_type[cam_id].append("distance_to_image_plane")
            prim_paths_by_cam_id = [row[cam_id] for row in self.camera_prim_paths]

            camera_resolution = self.cameras[0][cam_id]._resolution
            height, width = camera_resolution[1], camera_resolution[0]
            tiled_camera = CameraView(
                prim_paths_by_cam_id,
                camera_resolution=[width, height],
                output_annotators=self.annotator_type[cam_id],
            )
            self.tiled_cameras.append(tiled_camera)
            self.tiled_render_products.append(tiled_camera._render_product)

            # Pre-allocate output buffers for this camera's annotators
            self._output_buffers[cam_id] = {}

            for annotator_name in self.annotator_type[cam_id]:
                from env.camera_manager.capture.camera_view import ANNOTATOR_SPEC

                spec = ANNOTATOR_SPEC.get(annotator_name)
                if spec is None:
                    continue

                channels = spec["channels"]
                shape = (self.num_envs, height, width, channels)

                # Pre-allocate warp array on CUDA to reuse memory
                import warp as wp

                self._output_buffers[cam_id][annotator_name] = wp.zeros(
                    shape, dtype=spec["dtype"], device=self._render_cuda_device()
                )

    @staticmethod
    def _render_cuda_device() -> str:
        # The annotator writes into this buffer, so it must sit on the render GPU; a buffer on another GPU costs a
        # cross-device copy per frame and a CUDA context there.
        import carb

        gpu = carb.settings.get_settings().get("/physics/cudaDevice")
        return f"cuda:{gpu}" if isinstance(gpu, int) and gpu >= 0 else "cuda:0"

    def step(self, env_ids: List[int] = None, cam_ids: List[int] = None) -> List[List[List[any]]]:
        """
        Step the annotator. When env_id and cam_id is given, use the given. Otherwise apply to all cameras.
        Args:
            env_ids: List[int] - list of environment IDs to process
            cam_ids: List[int] - list of camera IDs to process
        Returns:
            List[List[List[Any]]] : returns the required data from annotators for required env_ids and cam_ids
            Format: data[cam_id][annotator_name] = [env_0_data, env_1_data, ...]
            where each env_i_data is {data: numpy_array, info: dict}
        """

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if cam_ids is None:
            cam_ids = list(range(len(self.cameras[0])))
        if ensure_isaac_rtx_render_update is not None:
            ensure_isaac_rtx_render_update(force=True)

        data = []
        for cam_id in cam_ids:
            cam_data = {}
            annotator_names = self.annotator_type[cam_id]
            for annotator_name in annotator_names:
                pre_allocated_out = None
                if cam_id in self._output_buffers and annotator_name in self._output_buffers[cam_id]:
                    pre_allocated_out = self._output_buffers[cam_id][annotator_name]

                out, info = self.tiled_cameras[cam_id].get_data(annotator_name, out=pre_allocated_out)

                # Convert out to numpy if it's a warp array (only convert once, reuse buffer)
                if hasattr(out, "numpy"):
                    out_np = out.numpy()
                elif hasattr(out, "cpu"):
                    out_np = out.cpu().numpy()
                else:
                    out_np = out

                env_list = []
                for env_id in env_ids:
                    env_list.append({"data": out_np[env_id], "info": info})

                cam_data[annotator_name] = env_list
            data.append(cam_data)
        return data

    def reset(
        self,
    ):
        """
        Soft Reset do not need to reset the replicator writer and camera.
        Only Hard Reset need which means if we reset simulation backend we need to initialize camera again
        Since Render product change, we also need to attch a new writer maybe
        """
        self.init_cameras()

    def destroy(self):
        """
        Destroy the capture manager.
        This function will be called when we close the environment.
        """

        self.annotator.clear()
        self.annotator_type.clear()
        self.annotator_device.clear()
        self.tiled_cameras.clear()
        self.cameras.clear()
        self.camera_names.clear()
        self.sim = None
        for rp in self.tiled_render_products:
            rp.destroy()
        self.camera_prim_paths.clear()
