"""TaskEnv of the process's simulation backend (env/sim_backend.py); every task inherits it from here."""

from env import sim_backend

if sim_backend.current() == sim_backend.MUJOCO:
    from env.backends.mujoco.task_env import MujocoTaskEnv as TaskEnv
else:
    from env.environment.isaac_task_env import TaskEnv

__all__ = ["TaskEnv"]
