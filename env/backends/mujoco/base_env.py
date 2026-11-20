from typing import Any

import gymnasium as gym
from omegaconf import DictConfig

from env.backends.mujoco.sim_env import MjSimEnv
from env.environment.seed_config import SeedConfigMixin


class MujocoBaseEnv(SeedConfigMixin, gym.Env):
    """BaseEnv of the MuJoCo backend: launches an MjSimEnv (no Kit app) on the first reset, as BaseEnv launches
    IsaacRLEnv; the sim config's physx / render sections are Isaac settings and are ignored here."""

    def __init__(self, config: DictConfig, app=None, **kwargs):
        self.app = app
        self.config = config
        self.device = self.config.get("device", "cpu")
        self.use_fabric = False
        self._seed_list: list[int] | None = None
        self._global_seed: int | None = None
        self.env_seeds: list[int] | None = None
        self._configure_seed_config(self.config.seed)
        self.sim = None
        self.interactiveSceneCfg = kwargs.get("interactiveSceneCfg", None)

    def launch_sim(self, config):
        self.sim = MjSimEnv(
            num_envs=config.scene.num_envs,
            env_spacing=config.scene.env_spacing,
            dt=config.get("dt", 1 / 60),
            decimation=config.decimation,
            render_interval=config.get("render_interval", 1),
            device=self.device,
            robots=self.interactiveSceneCfg or [],
        )
        self.env_spacing = config.scene.env_spacing
        self.sim.env_spacing = self.env_spacing
        self.env_origins = self.sim.scene.env_origins
        self.sim.setup(
            {
                "_setup_scene": self._setup_scene,
                "_reset_idx": self._reset_idx,
                "_post_setup_scene": self._post_setup_scene,
            }
        )

    def render(self):
        pass

    def sim_step(self, render: bool = True):
        self.sim.sim_step(render=render)

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        self.update_seed(seed)
        if self.sim is None:
            self.launch_sim(self.config)
        self.sim.reset(options=options)

    def _setup_scene(self, sim):
        pass

    def _reset_idx(self, env, env_ids=None):
        pass

    def _post_setup_scene(self, sim):
        pass
