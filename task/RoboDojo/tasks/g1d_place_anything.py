import random

from env.environment.task_env import TaskEnv
from scripts.objects import pick_anything, place_anything
from task.RoboDojo.tasks.g1d_pick_anything import PickAnythingG1dCommon


class PlaceAnythingG1dCommon(PickAnythingG1dCommon):
    """G1-D: put the object the instruction names (target, X) on the support the instruction names (receptacle, Y),
    and let go; the others (distractor_<j>, maybe another receptacle) and Y stay where they lie. g1d_put_anything_in
    is the same with a container. Layouts, geometry and names: scripts/objects/place_anything.py
    (scripts/datagen/place_anything_layouts.py)."""

    KIND = "support"
    STEP_LIM = 900  # actions: 30 s

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.start_ori = [
            {label: e["default_ori"] for label, e in self.entries(env_idx).items()} for env_idx in range(self.num_envs)
        ]

    def others(self, env_idx):
        return sorted(k for k in self.entries(env_idx) if k not in ("target", place_anything.LABEL))

    def label_names(self, env_idx, label, lang):
        """`label`'s names in `lang` (category first) that tell it apart from everything else on the table."""
        lm = self.scene_manager.layout_manager
        own = lm.get_label_descriptions(env_idx, label, lang=lang)
        rest = [lm.get_label_descriptions(env_idx, k, lang=lang) for k in self.entries(env_idx) if k != label]
        return pick_anything.usable_names(own, rest)

    def names(self, env_idx, lang):
        return self.label_names(env_idx, "target", lang)

    def run_reward(self):
        rm = self.reward_manager
        for env_idx in range(self.num_envs):
            start, ori = self.start[env_idx], self.start_ori[env_idx]
            checks = [
                rm.is_placed_in_receptacle("target", place_anything.LABEL, x_quat0=ori["target"]),
                rm.is_all_gripper_open(),
                rm.is_block_near(place_anything.LABEL, start[place_anything.LABEL], tol=place_anything.Y_STAY),
            ]
            checks += [rm.is_block_near(k, start[k], tol=self.STAY) for k in self.others(env_idx)]
            rm.check_single_env(env_idx, checks)

    def gen_instruction(self, env_idx):
        """One instruction per layout, from an RNG keyed by X's layout pose (as g1d_pick_anything)."""
        rng = random.Random(repr(self.entries(env_idx)["target"]["default_pos"]))
        names = {
            lang: {
                "x": self.label_names(env_idx, "target", lang),
                "y": self.label_names(env_idx, place_anything.LABEL, lang),
            }
            for lang in ("zh", "en")
        }
        return [place_anything.instruction(rng, self.KIND, names)]


class g1d_place_anything(PlaceAnythingG1dCommon, TaskEnv):
    pass
