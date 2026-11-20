from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from scripts.objects import block_patterns


class BuildBlockPatternG1dCommon:
    """G1-D builds the block pattern printed on the card lying on the table (scripts/objects/block_patterns.py) from the
    wood blocks scattered around it. The card's category index is the pattern; the blocks are labelled block_<i> in
    the pattern's piece order. Success: the blocks stand in the pattern anywhere on the table, turned any way, and both
    hands have let go; blocks of the same shape and paint may swap places. Other blocks (distractor_<j>) may lie
    among them. Layouts come from scripts/g1d/build_asset.py (PATTERN_TASKS)."""

    CARD = "card"
    # grippers at least this open: holding a 30 mm block leaves them ~0.3 open
    OPEN = 0.8

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 2700  # 90 s for up to five blocks; reset() sets block_patterns.step_lim of the card's pattern
        self.pattern = [None] * self.num_envs

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()
        self.pattern = [self.pattern_of(env_idx) for env_idx in range(self.num_envs)]
        self.step_lim = max(block_patterns.step_lim(p) for p in self.pattern)

    def pattern_of(self, env_idx):
        """The pattern printed on this env's card."""
        rigid = self.scene_manager.config[f"env{env_idx}"]["Rigid"]
        card = next(e for entries in rigid.values() for e in entries if e.get("label") == self.CARD)
        return block_patterns.PATTERNS[card["category_idx"]]

    def built_check(self, env_idx):
        pattern = self.pattern[env_idx]
        return self.reward_manager.is_pattern_built(
            block_patterns.labels(pattern),
            block_patterns.targets(pattern).tolist(),
            [block_patterns.YAW_PERIOD[cat] for cat, _, _ in pattern["pieces"]],
            anchor=pattern["anchor"],
            groups=block_patterns.groups(pattern),
        )

    def run_reward(self):
        for env_idx in range(self.num_envs):
            self.reward_manager.check_single_env(
                env_idx, [self.built_check(env_idx), self.reward_manager.is_all_gripper_open(open_threshold=self.OPEN)]
            )

    def gen_instruction(self, env_idx):
        return [
            "用积木搭出这个形状",
            "照着卡片上的图案，用积木搭出来",
            "用桌上的积木搭出卡片上的形状",
            "Build this shape with the blocks.",
            "Use the blocks to build the shape on the card.",
            "Copy the picture on the card with the wooden blocks.",
        ]


class g1d_build_block_pattern(BuildBlockPatternG1dCommon, TaskEnv):
    """G1-D: build the block pattern shown on the card from the scattered wood blocks."""
