from env.environment.base_env import BaseEnv
from env.environment.task_common import TaskEnvCommon


class TaskEnv(TaskEnvCommon, BaseEnv):
    """TaskEnv of the Isaac Sim backends (isaacsim, isaaclab_newton)."""

    @staticmethod
    def manager_classes():
        from env.camera_manager.camera_manager import CameraManager
        from env.camera_manager.capture.tiled_capture_manager import TiledCaptureManager
        from env.robot_manager.robot_manager import RobotManager
        from env.scene_manager.scene_manager import SceneManager

        return RobotManager, SceneManager, CameraManager, TiledCaptureManager
