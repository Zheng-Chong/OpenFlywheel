import random

from env.environment.task_env import TaskEnv
from scripts.objects import handover_anything, place_anything
from task.RoboDojo.tasks.g1d_place_anything import PlaceAnythingG1dCommon


class HandoverAnythingG1dCommon(PlaceAnythingG1dCommon):
    """G1-D: pick up the object the instruction names (target, X) with the hand on its side, hand it over in the air
    to the other hand, which puts it on the support the instruction names (receptacle, Y) on its side of the desk; the
    others stay where they lie. Success comes in two stages: both hands hold X off the desk at once, then X lies on Y
    with both hands open and Y and the others where they were. Layouts, geometry and names:
    scripts/objects/handover_anything.py (scripts/datagen/handover_anything_layouts.py)."""

    KIND = handover_anything.KIND
    STEP_LIM = 1500  # actions: 50 s

    def giver(self, env_idx):
        return handover_anything.hand_of(self.entries(env_idx)["target"]["default_pos"][0])

    def run_reward(self):
        rm = self.reward_manager
        for env_idx in range(self.num_envs):
            start, ori = self.start[env_idx], self.start_ori[env_idx]
            handed = [rm.is_held_by_both("target", air=handover_anything.AIR)]
            placed = [
                rm.is_placed_in_receptacle("target", place_anything.LABEL, x_quat0=ori["target"]),
                rm.is_all_gripper_open(),
                rm.is_block_near(place_anything.LABEL, start[place_anything.LABEL], tol=place_anything.Y_STAY),
            ]
            placed += [rm.is_block_near(k, start[k], tol=self.STAY) for k in self.others(env_idx)]
            rm.check_single_env(env_idx, handed)
            rm.check_single_env(env_idx, placed)

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
        return [handover_anything.instruction(rng, names, self.giver(env_idx))]


class g1d_handover_anything(HandoverAnythingG1dCommon, TaskEnv):
    pass
