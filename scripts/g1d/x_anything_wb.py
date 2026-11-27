"""19D fixed-base wholebody (`g1d_wb19`: waist pitch + lift + dual-arm gripper) X-Anything scripted experts.

Covers `g1d_pick_anything`, `g1d_place_anything`, `g1d_put_anything_in`, and `g1d_handover_anything` on top of the
v4.1 / v4.4 / v4.3 arm experts without modifying `scripted_demo.py` or any registered 16D expert module.

Each episode adapts the robot's lift column and waist pitch to the layout's table height and farthest target/receptacle
depth *before* constructing `GraspRobot` / `PlaceRobot` / `HandoverRobot` (so `Reach()` and home poses reflect the
post-adaptation shoulder frame), then runs the arm expert while holding the adapted body posture inside every action.
"""

from functools import partial

import numpy as np

from scripts.g1d import (
    grasp_v41,
    handover_v43,
    place_v42,
    place_v44,
    scripted_demo as demo,
    wholebody_demo as wb,
)
from scripts.objects import place_anything

WB_MAX_PITCH = float(np.deg2rad(22.0))  # rad: safe active-vision + table-clearance pitch for X-Anything
WB_COMFORT_REACH_Y = 0.32  # m from base_y=-0.73 before allocating waist pitch


class PickRobotWB(wb.WholebodyMixin, place_v44.StallBreakMixin, grasp_v41.GraspRobot41):
    pass


class PlaceRobotWB(wb.WholebodyMixin, place_v44.PlaceRobot44):
    pass


class PutRobotWB(wb.WholebodyMixin, place_v44.PutRobot44):
    pass


def _target_pose(env, label):
    lm = env.scene_manager.layout_manager
    pos, _ = lm.get_instance_pose(0, label=label)
    return demo.as_np(pos)


def pick_anything_wb_actions(env):
    """19D wholebody expert for g1d_pick_anything (body adaptation + v4.1 grasp + stall-break)."""
    target_pos = _target_pose(env, "target")
    table_top_z = wb.table_top_z_from_env(env, fallback_z=float(target_pos[2]) - 0.03)
    body = yield from wb.adapt_body_to_scene(
        env,
        target_y=float(target_pos[1]),
        target_z=float(target_pos[2]),
        table_top_z=table_top_z,
        max_pitch=WB_MAX_PITCH,
        comfort_reach_y=WB_COMFORT_REACH_Y,
    )
    yield from demo.pick_anything_v4_actions(env, partial(PickRobotWB, body=body))


def _place_or_put_wb_actions(env, robot_class):
    target_pos = _target_pose(env, "target")
    rec_pos = _target_pose(env, place_anything.LABEL)
    reach_y = max(float(target_pos[1]), float(rec_pos[1]))
    reach_z = min(float(target_pos[2]), float(rec_pos[2]))
    table_top_z = wb.table_top_z_from_env(env, fallback_z=reach_z - 0.03)
    body = yield from wb.adapt_body_to_scene(
        env,
        target_y=reach_y,
        target_z=reach_z,
        table_top_z=table_top_z,
        max_pitch=WB_MAX_PITCH,
        comfort_reach_y=WB_COMFORT_REACH_Y,
    )
    yield from place_v42.place_anything_actions(env, partial(robot_class, body=body))


def place_anything_wb_actions(env):
    """19D wholebody expert for g1d_place_anything (body adaptation + place v4.4)."""
    yield from _place_or_put_wb_actions(env, PlaceRobotWB)


def put_anything_in_wb_actions(env):
    """19D wholebody expert for g1d_put_anything_in (body adaptation + put v4.4)."""
    yield from _place_or_put_wb_actions(env, PutRobotWB)


def handover_anything_wb_actions(env):
    """19D wholebody expert for g1d_handover_anything (body adaptation + handover v4.3)."""
    target_pos = _target_pose(env, "target")
    rec_pos = _target_pose(env, place_anything.LABEL)
    reach_y = max(float(target_pos[1]), float(rec_pos[1]))
    reach_z = min(float(target_pos[2]), float(rec_pos[2]))
    table_top_z = wb.table_top_z_from_env(env, fallback_z=reach_z - 0.03)
    body = yield from wb.adapt_body_to_scene(
        env,
        target_y=reach_y,
        target_z=reach_z,
        table_top_z=table_top_z,
        max_pitch=WB_MAX_PITCH,
        comfort_reach_y=WB_COMFORT_REACH_Y,
    )
    for cmd in handover_v43.handover_anything_actions(env):
        yield (dict(cmd) | body) if body else cmd
