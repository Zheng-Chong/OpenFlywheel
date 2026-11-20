from omegaconf import open_dict

from env.backends.mujoco.base_env import MujocoBaseEnv
from env.environment.task_common import TaskEnvCommon


class MujocoTaskEnv(TaskEnvCommon, MujocoBaseEnv):
    def __init__(self, config, app=None, **kwargs):
        with open_dict(config.sim):
            config.sim.device = "cpu"  # MuJoCo steps on the CPU; the managers' tensors live there too
            config.sim.use_fabric = False
        super().__init__(config, app, **kwargs)

    @staticmethod
    def manager_classes():
        from env.backends.mujoco.cameras import MujocoCameraManager, MujocoCaptureManager
        from env.backends.mujoco.robot_manager import MujocoRobotManager
        from env.backends.mujoco.scene_manager import MujocoSceneManager

        return MujocoRobotManager, MujocoSceneManager, MujocoCameraManager, MujocoCaptureManager
