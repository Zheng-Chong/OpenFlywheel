"""IsaacRLEnv stand-in for the MuJoCo backend: what the managers read from `sim` (scene, cfg.sim.dt, cfg.decimation,
sim.device, sim.current_time) and the step / reset entry points of env/environment/base_env.py."""

import math
from types import SimpleNamespace

from env.backends.mujoco.articulation import MjArticulation, MjContactSensor, MjScene, grid_origins
from env.backends.mujoco.world import EnvWorld

# Longest MuJoCo step: at the G1-D's 1/300 s an IK-servoed hand pinching a light block on the table oscillates (soft
# contacts close a loop the IK feedback excites) and squeezes it out; 1/600 s holds it. Longer config dts are split.
MAX_TIMESTEP = 1 / 600


class MjSimContext:
    """`sim.sim` (IsaacLab SimulationContext): device, time, render."""

    def __init__(self, env, device):
        self._env = env
        self.device = device
        self.is_rendering = True

    @property
    def current_time(self):
        return self._env.worlds[0].data.time

    def render(self):
        pass

    def has_gui(self):
        return False

    def has_rtx_sensors(self):
        return False


class MjSimEnv:
    def __init__(self, num_envs, env_spacing, dt, decimation, render_interval, device, robots, gravity=(0, 0, -9.81)):
        """`robots`: [(scene key "robot<i>", RobotSpec, [x, y, z, qw, qx, qy, qz] env-local root pose)]."""
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.physics_dt = dt
        self.step_dt = dt * decimation
        self.cfg = SimpleNamespace(
            sim=SimpleNamespace(dt=dt, render_interval=render_interval, device=device), decimation=decimation
        )
        origins = grid_origins(num_envs, env_spacing)
        self.substeps = math.ceil(dt / MAX_TIMESTEP - 1e-9)
        self.worlds = [EnvWorld(e, dt / self.substeps, gravity) for e in range(num_envs)]
        for key, spec, pose in robots:
            for world in self.worlds:
                world.add_robot(f"{key}/", spec, pose[:3], pose[3:])
        for world in self.worlds:
            world.compile()
        self.scene = MjScene(origins)
        for key, spec, _ in robots:
            art = MjArticulation(self.worlds, origins, f"{key}/", spec.init_joint_pos)
            self.scene.articulations[key] = art
            if spec.finger_pads:
                self.scene.sensors[f"{key}_pads"] = MjContactSensor(art, spec.finger_pads.values())
        self.sim = MjSimContext(self, device)
        self._sim_step_counter = 0

    def setup(self, func):
        """IsaacRLEnv._setup_scene: the task env's scene hooks, in IsaacLab's order."""
        func["_setup_scene"](self)
        func["_post_setup_scene"](self)
        self._reset_idx = func["_reset_idx"]

    def compile(self, env_ids=None):
        """Rebuild the models of `env_ids` (all) after the scene manager changed their specs; state is kept."""
        for e in range(self.num_envs) if env_ids is None else env_ids:
            self.worlds[e].compile()

    def sim_step(self, render: bool = True):
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            for world in self.worlds:
                for _ in range(self.substeps):
                    world.step()

    def reset(self, seed=None, options=None):
        for world in self.worlds:
            world.forward()
        self._reset_idx(self, None)
        return None, {}

    def close(self):
        self.worlds = []
