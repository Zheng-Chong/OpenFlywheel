from env.environment.task_env import TaskEnv
from scripts.objects import place_anything, sort_anything
from task.RoboDojo.tasks.g1d_pick_anything import PickAnythingG1dCommon


class SortAnythingG1dCommon(PickAnythingG1dCommon):
    """G1-D: sort the objects on the desk (item_<i>) by class into the receptacles the instruction gives each class
    (receptacle_<j>; a spare one gets nothing) and let go; every receptacle stays upright where it stood. Each item's
    layout entry names its class and receptacle ("sort"). Layouts, classes and instructions:
    scripts/objects/sort_anything.py (scripts/datagen/sort_anything_layouts.py); geometry as g1d_place_anything's."""

    STEP_LIM = 2400  # actions: 80 s, 4 objects picked, carried and placed one by one

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.start_ori = [
            {label: e["default_ori"] for label, e in self.entries(env_idx).items()} for env_idx in range(self.num_envs)
        ]

    def items(self, env_idx):
        """[(item label, its receptacle's label)]."""
        entries = self.entries(env_idx)
        return [(k, entries[k]["sort"]["receptacle"]) for k in sorted(entries) if k.startswith(sort_anything.ITEM)]

    def receptacles(self, env_idx):
        return sorted(k for k in self.entries(env_idx) if k.startswith(sort_anything.RECEPTACLE))

    def distractors(self, env_idx):
        return []

    def run_reward(self):
        rm = self.reward_manager
        for env_idx in range(self.num_envs):
            start, ori = self.start[env_idx], self.start_ori[env_idx]
            checks = [rm.is_placed_in_receptacle(x, y, x_quat0=ori[x]) for x, y in self.items(env_idx)]
            checks.append(rm.is_all_gripper_open())
            checks += [rm.is_block_near(y, start[y], tol=place_anything.Y_STAY) for y in self.receptacles(env_idx)]
            rm.check_single_env(env_idx, checks)

    def gen_instruction(self, env_idx):
        """The layout's own instruction (scripts/datagen/sort_anything_layouts.py writes one per layout)."""
        layout = self.scene_manager.layout_manager.saved_layouts[env_idx] or {}
        if not layout.get("Instruction"):
            raise ValueError("g1d_sort_anything layouts carry their instruction")
        return [layout["Instruction"]]


class g1d_sort_anything(SortAnythingG1dCommon, TaskEnv):
    pass
