import random

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from scripts.objects import pick_anything


class PickAnythingG1dCommon:
    """G1-D: pick up the object the instruction names (target) from among 3-5 others (distractor_<j>) and hold it up;
    the others stay where they lie. Layouts and names: scripts/objects/pick_anything.py (via build_asset.py)."""

    STEP_LIM = 600  # actions: 20 s
    LIFT = 0.08  # m the target's centre rises
    STAY = 0.03  # m the others may move

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

    def distractors(self, env_idx):
        return sorted(k for k in self.entries(env_idx) if k != "target")

    def names(self, env_idx, lang):
        """The target's names in `lang` (category first) that tell it apart from everything else on the table."""
        lm = self.scene_manager.layout_manager
        own = lm.get_label_descriptions(env_idx, "target", lang=lang)
        others = [lm.get_label_descriptions(env_idx, k, lang=lang) for k in self.distractors(env_idx)]
        return pick_anything.usable_names(own, others)

    def run_reward(self):
        rm = self.reward_manager
        for env_idx in range(self.num_envs):
            checks = [rm.is_lift(label="target", z_threshold=self.LIFT)]
            checks += [rm.is_block_near(k, self.start[env_idx][k], tol=self.STAY) for k in self.distractors(env_idx)]
            rm.check_single_env(env_idx, checks)

    def gen_instruction(self, env_idx):
        """One instruction per layout, drawn from an RNG keyed by the target's layout pose: DescManager seeds every
        episode of a run alike, so a list of templates would give every episode the same one."""
        rng = random.Random(repr(self.entries(env_idx)["target"]["default_pos"]))
        return [pick_anything.instruction(rng, {lang: self.names(env_idx, lang) for lang in ("zh", "en")})]


class g1d_pick_anything(PickAnythingG1dCommon, TaskEnv):
    pass
