from typing import Any, List

from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from env.robot_manager.control_manager import ControlSeq


class TaskEnvCommon:
    """Task env flow of every simulation backend. A backend's TaskEnv inherits this, then its BaseEnv, and returns its
    RobotManager, SceneManager, CameraManager and capture manager classes from manager_classes()."""

    @staticmethod
    def manager_classes():
        raise NotImplementedError

    def __init__(self, config, app, **kwargs):
        robot_manager_cls, scene_manager_cls, camera_manager_cls, capture_manager_cls = self.manager_classes()
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.num_envs = self.sim_config.scene.num_envs
        self.env_spacing = self.sim_config.scene.env_spacing
        self.device = self.sim_config.get("device", "cpu")
        self.dt = self.sim_config.dt

        # RobotManager must be created before BaseEnv.__init__ because it
        # provides the InteractiveSceneCfg used to launch the simulation.
        self.robot_config = config.robot
        self.robot_manager = robot_manager_cls(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            config=self.robot_config,
            dt=self.dt,
            device=self.device,
        )
        self._interactiveSceneCfg = self.robot_manager._get_SceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing
        )
        super().__init__(self.sim_config, app=app, interactiveSceneCfg=self._interactiveSceneCfg, **kwargs)
        self.env_seed_list = self.env_seeds

        self.scene_manager = scene_manager_cls(
            self.num_envs,
            self.device,
            self.sim_config.scene.env_spacing,
            use_fabric=self.use_fabric,
            seeds_per_env=self.env_seed_list,
            scene_config=self.scene_config,
            task_config=config.task_env,
        )

        self.robot_manager._load_camera_cfg(config.camera)
        self._keep_only_cameras(config)
        self.capture_config = config.camera
        self.camera_config = config.camera
        self.camera_manager = camera_manager_cls(
            self.num_envs,
            self.camera_config,
            self.device,
            seeds_per_env=self.env_seed_list,
        )
        self.capture_manager = capture_manager_cls(
            self.num_envs,
            self.capture_config,
            self.camera_manager,
            self.device,
        )

    @staticmethod
    def _keep_only_cameras(config):
        """Drop cameras not listed in eval_cfg.observation.vision.cameras (all are kept when unset)."""
        keep = OmegaConf.select(config, "eval_cfg.observation.vision.cameras")
        if not keep:
            return
        cameras = [k for k, v in config.camera.items() if isinstance(v, DictConfig) and "camera" in v]
        missing = set(keep) - set(cameras)
        if missing:
            raise ValueError(f"observation.vision.cameras lists unknown cameras {sorted(missing)}; have {cameras}")
        with open_dict(config.camera):
            for name in cameras:
                if name not in keep:
                    config.camera.pop(name)

    def _setup_scene(self, sim):
        self.robot_manager.initialize(sim)
        super()._setup_scene(sim)

    def _post_setup_scene(self, sim):
        self.scene_manager.initialize(sim)
        super()._post_setup_scene(sim)
        self.camera_manager.initialize(sim)
        self.capture_manager.initialize(sim)
        self.camera_manager.post_init()

    def update_seed(self, seed: Any | None = None):
        super().update_seed(seed)
        if hasattr(self, "scene_manager") and self.scene_manager is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(self.env_seeds)

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self.scene_manager.reload_scene()
        self.robot_manager.reset(init_joints=self.layout_init_joints())
        for _ in range(300):
            self.sim_step(render=False)
        self.camera_manager.layout_offsets = self.layout_entries("Camera")
        self.camera_manager.reset()
        self.capture_manager.reset()

    def sync_granular_visuals(self):
        """Have the renderer draw the granular grains (GranularObject.sync_fabric_visuals); call once the reset is
        done, after any step that reloads or re-poses the scene."""
        for env_objs in getattr(self.scene_manager, "_fluid_objects", []):
            for obj in env_objs.values():
                if hasattr(obj, "sync_fabric_visuals"):
                    obj.sync_fabric_visuals()

    def layout_entries(self, key):
        """Per env, its layout's top-level `key` entry, or None."""
        saved = getattr(self.scene_manager.layout_manager, "saved_layouts", None) or [None] * self.num_envs
        return [(lay or {}).get(key) for lay in saved]

    def layout_init_joints(self):
        """Per env, the arm joints its layout starts from (Robot.init_joint_pos), or None."""
        return [(robot or {}).get("init_joint_pos") for robot in self.layout_entries("Robot")]

    def step(self, meta_control_list: List[ControlSeq]):
        self.robot_manager.control_robot(meta_control_list=meta_control_list)

    def close(self):
        self.capture_manager.destroy()
        self.camera_manager.destroy()
        self.robot_manager.close()
        self.scene_manager.close()
        super().close()
