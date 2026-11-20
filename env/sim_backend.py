"""Which simulator runs the environment.

    isaacsim         Isaac Sim + IsaacLab 2.x, PhysX, RTX cameras (the benchmark's reference backend, default)
    isaaclab_newton  IsaacLab 3.x with Newton physics (MuJoCo-Warp solver), RTX cameras through Isaac Sim 6
    mujoco           MuJoCo 4 on the CPU, no Kit app; cameras through mujoco.Renderer (env/backends/mujoco/)

The choice is made once per process, before any env module is imported: `--sim_backend` of the eval client, else
ROBODOJO_SIM_BACKEND, else `backend:` of the sim config (env_cfg/sim/*.yml), else isaacsim. Only stdlib here: the eval
client reads it before it decides whether to launch Kit.
"""

import os

ISAACSIM = "isaacsim"
ISAACLAB_NEWTON = "isaaclab_newton"
MUJOCO = "mujoco"
BACKENDS = (ISAACSIM, ISAACLAB_NEWTON, MUJOCO)
ENV_VAR = "ROBODOJO_SIM_BACKEND"


def validate(name):
    if name not in BACKENDS:
        raise ValueError(f"unknown simulation backend {name!r}; choose one of {', '.join(BACKENDS)}")
    return name


def select(name=None, sim_config=None):
    """Fix the process's backend: `name` if given, else the env var, else sim_config['backend'], else isaacsim."""
    chosen = name or os.environ.get(ENV_VAR) or (sim_config or {}).get("backend") or ISAACSIM
    os.environ[ENV_VAR] = validate(chosen)
    return chosen


def current():
    return validate(os.environ.get(ENV_VAR) or ISAACSIM)


def uses_kit(name=None):
    """Whether the backend runs inside an Isaac Sim (Kit) app: it needs AppLauncher before any isaaclab import."""
    return (name or current()) != MUJOCO
