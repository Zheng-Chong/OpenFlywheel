from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from scripts.objects import block_sorting


class SortBlocksG1dCommon:
    """G1-D sorts the painted wood blocks into two trays by the card lying behind each (scripts/objects/
    block_sorting.py): a colour swatch or a block shape. The cards' category index says each tray's class; the blocks
    (block_<i>) belong in the tray (box_0 left, box_1 right) whose class matches their paint or shape. Success: every
    block inside its tray, below the rim, and both hands open. Layouts come from scripts/g1d/build_asset.py
    (PATTERN_TASKS)."""

    BY = None  # "color" / "shape"
    BOXES = ("box_0", "box_1")
    # grippers at least this open: holding a 30 mm block leaves them ~0.3 open
    OPEN = 0.8

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 1800  # 60 s: four blocks at ~10 s each, waiting for the arm at every key, plus slack
        self.groups = [None] * self.num_envs

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()
        self.groups = [self.groups_of(env_idx) for env_idx in range(self.num_envs)]

    def groups_of(self, env_idx):
        """{box label: labels of the blocks that belong in it} of this env's layout."""
        rigid = self.scene_manager.config[f"env{env_idx}"]["Rigid"]
        entries = [e for es in rigid.values() for e in es]
        keys = {
            f"box_{e['label'].removeprefix('card_')}": block_sorting.card(e["category_idx"])["key"]
            for e in entries
            if str(e.get("label", "")).startswith("card_")
        }
        groups = {box: [] for box in self.BOXES}
        for e in entries:
            if str(e.get("label", "")).startswith("block_"):
                key = block_sorting.block_key(self.BY, e["category"], e["category_idx"])
                groups[next(box for box, k in keys.items() if k == key)].append(e["label"])
        return groups

    def sorted_checks(self, env_idx):
        rm = self.reward_manager
        checks = []
        for box, labels in self.groups[env_idx].items():
            checks += [rm.is_all_A_in_B(labels, box), rm.is_all_A_z_lower_than_B_bbox_zmax(labels, box)]
        return checks

    def run_reward(self):
        for env_idx in range(self.num_envs):
            self.reward_manager.check_single_env(
                env_idx,
                [*self.sorted_checks(env_idx), self.reward_manager.is_all_gripper_open(open_threshold=self.OPEN)],
            )

    def gen_instruction(self, env_idx):
        return [
            "按卡片把积木分拣到托盘里",
            "照托盘后面的卡片，把积木放进对应的托盘",
            "把每块积木放进卡片和它一样的那个托盘",
            "Sort the blocks into the trays by the cards.",
            "Put each block into the tray whose card matches it.",
            "Use the cards behind the trays to sort the blocks.",
        ]


class g1d_sort_blocks_by_color(SortBlocksG1dCommon, TaskEnv):
    """G1-D: sort the painted blocks into the two trays by the colour on each tray's card."""

    BY = "color"
