"""Pose and clock boundaries shared by native recording and independent tests."""

import math


def native_pose_xyzw(pose_wxyz):
    """PhysX tensor targets use xyzw; RigidObject writes and our traces use wxyz."""
    if pose_wxyz.shape[-1] != 7:
        raise ValueError("Rigid pose must have seven components")
    return pose_wxyz[..., [0, 1, 2, 4, 5, 6, 3]]


def camera_clock_ready(sensor_times, physics_time_s, dt_s):
    return bool(sensor_times) and all(
        t is not None and math.isfinite(t) and abs(t - physics_time_s) < dt_s * 0.1 for t in sensor_times
    )
