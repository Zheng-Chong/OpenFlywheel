from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from scripts.objects import block_atomic, block_patterns


class BlockAtomicG1dCommon:
    """The atomic block tasks of G1-D (scripts/objects/block_atomic.py): one skill each with the painted wood blocks.
    A task names its objects by layout label (block, mark, top_block, base_block, target_block, tray, block_<i>, ...),
    checks them with env/reward_manager/block_checks.py (checks()), and phrases its instruction (instructions()) with
    names no other block on the table shares. Layouts come from scripts/g1d/build_asset.py (PATTERN_TASKS)."""

    STEP_LIM = 900  # actions: 30 s
    HANDS_OPEN = True  # success also needs both grippers open at least OPEN
    # grippers at least this open: holding a 30 mm block leaves them ~0.3 open
    OPEN = 0.8

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = self.STEP_LIM
        self.start = [{} for _ in range(self.num_envs)]

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()
        self.start = [
            {label: e["default_pos"][:2] for label, e in self.entries(env_idx).items()}
            for env_idx in range(self.num_envs)
        ]

    def entries(self, env_idx):
        """{label: layout entry} of this env's objects."""
        rigid = self.scene_manager.config[f"env{env_idx}"]["Rigid"]
        return {e["label"]: e for es in rigid.values() for e in es}

    def size(self, env_idx, label):
        return block_patterns.SIZES[self.entries(env_idx)[label]["category"]]

    def table_z(self, env_idx):
        """The tabletop height: a block's layout pose stands it on the table (build_asset.table_entry, 1 mm up)."""
        label, e = next((k, e) for k, e in self.entries(env_idx).items() if e["category"].startswith("wood_block_"))
        return e["default_pos"][2] - self.size(env_idx, label)[2] / 2 - 1e-3

    def blocks(self, env_idx):
        return [k for k, e in self.entries(env_idx).items() if e["category"].startswith("wood_block_")]

    def names(self, env_idx, label, lang):
        """`label`'s instruction names (simple to detailed) that no other block on the table has."""
        lm = self.scene_manager.layout_manager
        theirs = {
            n
            for other in self.blocks(env_idx)
            if other != label
            for n in lm.get_label_descriptions(env_idx, other, lang=lang)
        }
        own = [n for n in lm.get_label_descriptions(env_idx, label, lang=lang)[1:] if n not in theirs]  # [0]: category
        return own or [lm.get_label_descriptions(env_idx, label, lang=lang)[-1]]

    def checks(self, env_idx):
        raise NotImplementedError

    def instructions(self, env_idx):
        """The instructions: three in Chinese, then three in English."""
        raise NotImplementedError

    def run_reward(self):
        for env_idx in range(self.num_envs):
            checks = self.checks(env_idx)
            if self.HANDS_OPEN:
                checks.append(self.reward_manager.is_all_gripper_open(open_threshold=self.OPEN))
            self.reward_manager.check_single_env(env_idx, checks)

    def gen_instruction(self, env_idx):
        return self.instructions(env_idx)


class g1d_place_block_on_mark(BlockAtomicG1dCommon, TaskEnv):
    """G1-D: put the block on the round red mark."""

    def checks(self, env_idx):
        rm = self.reward_manager
        size, table = self.size(env_idx, "block"), self.table_z(env_idx)
        return [rm.is_block_placed_on("block", "mark", size, table, radius=block_atomic.MARK_RADIUS)]

    def instructions(self, env_idx):
        zh, en = self.names(env_idx, "block", "zh"), self.names(env_idx, "block", "en")
        return [
            f"把{zh[0]}放到红色圆形标记上",
            f"把{zh[-1]}放在桌上的红点上",
            f"将{zh[0]}放到圆形标记的中间",
            f"Put the {en[0]} on the red mark.",
            f"Place the {en[-1]} on the red dot.",
            f"Set the {en[0]} down in the middle of the round mark.",
        ]
