import types

import numpy as np
import transforms3d as t3d

from env.environment.task_env import TaskEnv
from scripts.objects import cup_profile
from task.RoboDojo.tasks.g1d_pour_beans_plus import G1dPourBeansPlusCommon

UPRIGHT_DEG, REST_M = 15, 0.01
# defaults when a layout carries no Generalization.acceptance: beans in the target cup / spilled, shares of all beans
TARGET_SHARE, SPILL_SHARE = 0.35, 0.05


def is_beans_poured_real520(self, args):
    """Func_Parser check: the target cup's cavity holds at least `target` beans and at most `spill` beans are in
    neither cup's cavity (counts from the layout's Generalization.acceptance, else shares of all beans)."""
    env_idx, lm = args["env_idx"], self.layout_manager
    fluid = lm.get_scene_object(inst_name=lm.get_instance_name(label="beans", env_idx=env_idx), env_idx=env_idx)
    points = np.asarray(fluid.get_particle_positions()[0], dtype=float)
    if not len(points):
        return 0.0
    points = points @ t3d.quaternions.quat2mat(fluid.init_ori).T + np.asarray(fluid.init_pos, dtype=float)
    inside = []
    for label in ("source_cup", "target_cup"):
        name = lm.get_instance_name(label=label, env_idx=env_idx)
        cup = lm.get_scene_object(inst_name=name, env_idx=env_idx)
        pos, quat = lm.get_instance_pose(inst_name=name, env_idx=env_idx)
        pos, quat = np.asarray(pos, dtype=float).reshape(-1)[:3], np.asarray(quat, dtype=float).reshape(-1)[:4]
        scale = float(np.asarray(getattr(cup, "scale", (1.0, 1.0, 1.0)), dtype=float).reshape(-1)[0])
        inside.append(cup_profile.in_cavity(points, pos, quat, cup.instance_config["cavity"], scale))
    acc = ((lm.saved_layouts[env_idx] or {}).get("Generalization") or {}).get("acceptance") or {}
    target = acc.get("target_beans", TARGET_SHARE * len(points))
    spill = acc.get("spill_beans", SPILL_SHARE * len(points))
    in_target = int(inside[1].sum())
    outside = int((~inside[0] & ~inside[1]).sum())
    return 1.0 if in_target >= target and outside <= spill else 0.0


class G1dPourBeansReal520Common(G1dPourBeansPlusCommon):
    """倒豆子 · 深圳 520 孪生: the real G1-D teleoperated pour of generalization_pour_beans_eps520_v3 (ShenZhen
    stages B-D, ep220-519) in its sim twin. The right hand picks the frosted tall measuring cup of red beans on the
    right, the left hand the empty cup on the left, and the beans are poured into the left cup in mid-air until it is
    about full (in the real episodes ~60% of the source's beans go over); the source cup goes back on the desk, the
    target may stay in the hand (stages B and C) as it does in the recordings.

    Success (acceptance real520_v1): the target cup's cavity holds at least the layout's
    Generalization.acceptance.target_beans, at most spill_beans lie in neither cup, both cups stand within
    UPRIGHT_DEG of upright and the source cup is back on the desk (within REST_M of its start height).

    Layouts (one per real episode: cups where the real hands closed, its first-frame arm joints, the real
    instruction) come from real2sim/tasks/g1d_pour_beans_real520/make_layouts.py.
    """

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.step_lim = 1200  # the real episodes take 465-940 frames at 30 Hz
        fp = self.reward_manager.func_parser
        fp.is_beans_poured_real520 = types.MethodType(is_beans_poured_real520, fp)

    def pour_check(self):
        return ("is_beans_poured_real520", {})

    def run_reward(self):
        rm = self.reward_manager
        rm.check(
            [
                self.pour_check(),
                *(
                    rm.is_axis_up(label=cup, axis=[0, 0, 1], threshold=UPRIGHT_DEG)
                    for cup in ("source_cup", "target_cup")
                ),
                rm.is_not_lift(label="source_cup", z_threshold=REST_M),
            ]
        )


class g1d_pour_beans_real520(G1dPourBeansReal520Common, TaskEnv):
    pass
