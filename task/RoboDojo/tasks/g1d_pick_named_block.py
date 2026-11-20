from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_place_block_on_mark import BlockAtomicG1dCommon


class g1d_pick_named_block(BlockAtomicG1dCommon, TaskEnv):
    """G1-D: pick up the named block (target_block) and hold it up; the others stay where they lie."""

    HANDS_OPEN = False  # it is held up

    def checks(self, env_idx):
        rm = self.reward_manager
        checks = [rm.is_block_lifted("target_block", self.size(env_idx, "target_block"), self.table_z(env_idx))]
        others = [b for b in self.blocks(env_idx) if b != "target_block"]
        return checks + [rm.is_block_near(b, self.start[env_idx][b], tol=0.03) for b in others]

    def instructions(self, env_idx):
        zh, en = self.names(env_idx, "target_block", "zh"), self.names(env_idx, "target_block", "en")
        return [
            f"拿起{zh[0]}",
            f"把{zh[-1]}拿起来",
            f"抓起桌上的{zh[0]}",
            f"Pick up the {en[0]}.",
            f"Lift the {en[-1]}.",
            f"Grab the {en[0]} and hold it up.",
        ]
