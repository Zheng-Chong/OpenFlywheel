from collections.abc import Sequence

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

try:
    from isaaclab.sim import PhysxCfg

    _PHYSX_FIELD = "physx"
except ImportError:  # IsaacLab 3
    from isaaclab_physx.physics import PhysxCfg

    _PHYSX_FIELD = "physics"

from env import sim_backend
from env.environment.isaac.direct_rl_env import CustomDirectRLEnv


def _replicate_env_0(scene):
    """IsaacLab 3 version of clone_environments(copy_from_source=True) for a scene the task authored in USD.

    PhysX parses the stage itself; Newton simulates only what is imported into its model builder, which IsaacLab 3
    fills from the scene's asset cfgs, and ours has none (env/backends/newton/physics.py).
    """
    from isaaclab import cloner
    import numpy as np

    num_envs, source = scene.num_envs, scene.env_prim_paths[0]
    template = source.rsplit("_", 1)[0] + "_{}"
    positions = np.asarray(scene.env_origins.cpu(), dtype=np.float32)
    if num_envs > 1:
        cloner.usd_replicate(scene.stage, [source], [template], np.arange(1, num_envs), positions=positions[1:])
    if sim_backend.current() == sim_backend.ISAACLAB_NEWTON:
        from env.backends.newton import physics

        physics.replicate(scene)


@configclass
class IsaacRLEnvCfg(DirectRLEnvCfg):
    decimation: int = 3
    episode_length_s: float = 2000 * (1 / 60) * decimation
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        **{_PHYSX_FIELD: PhysxCfg()},
    )


class IsaacRLEnv(CustomDirectRLEnv):
    cfg: IsaacRLEnvCfg
    env_spacing: float = 4.0

    def __init__(self, cfg: IsaacRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.func = kwargs.get("func", None)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.func["_setup_scene"](self)
        if hasattr(self.scene, "clone_environments"):
            self.scene.clone_environments(copy_from_source=True)
        else:
            _replicate_env_0(self.scene)
        self.func["_post_setup_scene"](self)

    def _get_observations(self):
        pass

    def _get_dones(self):
        pass

    def _get_states(self):
        pass

    def _get_rewards(self):
        pass

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        self.func["_reset_idx"](self, env_ids)

    def _apply_action(self):
        pass

    def _pre_physics_step(self, actions, env_ids):
        pass
