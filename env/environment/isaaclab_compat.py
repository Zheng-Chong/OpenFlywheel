"""Differences between the IsaacLab versions the Isaac backends run on (env/sim_backend.py)."""

from functools import cache
import importlib.util
import os
import shutil

from env import sim_backend


@cache
def quats_are_xyzw():
    """IsaacLab 3 (the package split into isaaclab_physx / isaaclab_newton) returns asset quaternions in Warp's xyzw
    order; IsaacLab 2, and the MuJoCo backend that mirrors its API, in wxyz."""
    return sim_backend.current() != sim_backend.MUJOCO and importlib.util.find_spec("isaaclab_physx") is not None


def implicit_actuator_cfg(**kwargs):
    """ImplicitActuatorCfg from IsaacLab 2 / 3.0 field names; IsaacLab 3.1 renamed the solver limits."""
    from isaaclab.actuators import ImplicitActuatorCfg

    if "effort_limit_sim" not in ImplicitActuatorCfg.__dataclass_fields__:
        renamed = {"effort_limit_sim": "joint_effort_limit", "velocity_limit_sim": "joint_velocity_limit"}
        kwargs = {renamed.get(k, k): v for k, v in kwargs.items()}
    return ImplicitActuatorCfg(**kwargs)


def joint_velocity_limit(actuator_cfg):
    """The solver joint speed limit of an actuator cfg (`velocity_limit_sim` before IsaacLab 3.1)."""
    return getattr(actuator_cfg, "velocity_limit_sim", getattr(actuator_cfg, "joint_velocity_limit", None))


def wxyz(quat):
    """`quat` (..., 4) of an IsaacLab asset's data in wxyz, for numpy arrays and torch tensors."""
    quat = getattr(quat, "torch", quat)  # IsaacLab 3's ProxyArray
    if not quats_are_xyzw():
        return quat
    return quat[..., [3, 0, 1, 2]]


def isaaclab_quat(quat_wxyz):
    """A wxyz quaternion as IsaacLab's cfgs and asset writes take it (e.g. ArticulationCfg.InitialStateCfg.rot)."""
    if not quats_are_xyzw():
        return tuple(quat_wxyz)
    w, x, y, z = quat_wxyz
    return (x, y, z, w)


def pose_wxyz(pose):
    """`pose` (..., 7) [position, quaternion] of an IsaacLab asset's data with the quaternion in wxyz."""
    pose = getattr(pose, "torch", pose)
    if not quats_are_xyzw():
        return pose
    return pose[..., [0, 1, 2, 6, 3, 4, 5]]


def writable_usd_dir(usd_dir):
    """The robot USD dir to import into. IsaacLab 3 rewrites a robot's library USD (the URDF converter when its hash
    differs from IsaacLab 2.3's, the G1-D's layer repair); on a read-only library (the Slurm asset mount) it works on a
    node-local copy under $XDG_CACHE_HOME, one per .asset_hash."""
    usd_dir = os.path.normpath(usd_dir)
    exists = os.path.isdir(usd_dir)
    if importlib.util.find_spec("isaaclab_physx") is None or os.access(
        usd_dir if exists else os.path.dirname(usd_dir), os.W_OK
    ):
        return usd_dir
    stamp = "nohash"
    if os.path.isfile(os.path.join(usd_dir, ".asset_hash")):
        with open(os.path.join(usd_dir, ".asset_hash")) as f:
            stamp = f.read().strip()
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    robot = os.path.basename(os.path.dirname(usd_dir))
    copy = os.path.join(cache_root, "robodojo", "robots", robot, f"{os.path.basename(usd_dir)}-{stamp}")
    if not os.path.isdir(copy):
        tmp = f"{copy}.tmp{os.getpid()}"
        if exists:
            shutil.copytree(usd_dir, tmp)
        os.makedirs(tmp, exist_ok=True)
        for root, _, files in os.walk(tmp):
            for path in [root] + [os.path.join(root, name) for name in files]:
                os.chmod(path, os.stat(path).st_mode | 0o200)
        try:
            os.rename(tmp, copy)
        except OSError:  # another process on this node made it first
            shutil.rmtree(tmp)
    return copy
