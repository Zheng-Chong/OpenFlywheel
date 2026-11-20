import random

import numpy as np
from omegaconf import DictConfig, OmegaConf

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from scripts.objects import open_anything


class OpenAnythingG1dCommon:
    """G1-D: open the named door, drawer or lid of the articulated object on the desk (label target) by at least its
    goal (the layout's `open`: joint, closed value, opening sign, goal in rad or m). Layouts and names:
    scripts/objects/open_anything.py (via build_asset.py)."""

    STEP_LIM = 600  # actions: 20 s

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = self.STEP_LIM

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()
        for env_idx in range(self.num_envs):
            self.close_parts(env_idx)

    def close_parts(self, env_idx):
        """Every openable part of the object at its closed value, with the drive targets its USD authors: some
        library USDs start a door half open (a joint state). How a part stays shut or where it is left is the
        asset's own joint friction and drives (scripts/objects/robocasa.py joint_axis_params), not the task's."""
        lm = self.scene_manager.layout_manager
        inst = lm.get_scene_object(env_idx, lm.get_instance_name(env_idx, "target"))
        if not hasattr(inst, "set_joint_positions"):
            return
        q = inst.get_joint_positions()
        q = np.array(q.cpu() if hasattr(q, "cpu") else q, dtype=np.float32).reshape(-1)
        for name, sib in self.target_part(env_idx)["siblings"].items():
            if name in inst.dof_names:
                q[inst.dof_names.index(name)] = sib["closed"]
        from isaacsim.core.utils.types import ArticulationAction
        import torch

        inst.set_current_joint_positions(q)
        inst.initial_joint_positions = q.copy()
        inst.apply_action(ArticulationAction(joint_positions=torch.tensor(inst.authored_drive_targets(q))))

    def target_entry(self, env_idx):
        """The layout entry of the object to open."""
        arts = self.scene_manager.config[f"env{env_idx}"]["Articulation"]
        return next(e for es in arts.values() for e in es if e["label"] == "target")

    def target_part(self, env_idx):
        """The layout's part to open (open_anything.LAYOUT_KEYS) as plain Python values."""
        part = self.target_entry(env_idx)["open"]
        return OmegaConf.to_container(part, resolve=True) if isinstance(part, DictConfig) else dict(part)

    def run_reward(self):
        rm = self.reward_manager
        for env_idx in range(self.num_envs):
            part = self.target_part(env_idx)
            check = rm.is_joint_opened("target", part["joint"], part["closed"], part["sign"], part["goal"])
            rm.check_single_env(env_idx, [check])

    def gen_instruction(self, env_idx):
        """One instruction per layout, drawn from an RNG keyed by the object's layout pose: DescManager seeds every
        episode of a run alike, so a list of templates would give every episode the same one."""
        part = self.target_part(env_idx)
        rng = random.Random(repr(list(self.target_entry(env_idx)["default_pos"])) + part["joint"])
        return [open_anything.instruction(rng, part)]


class g1d_open_anything(OpenAnythingG1dCommon, TaskEnv):
    pass
