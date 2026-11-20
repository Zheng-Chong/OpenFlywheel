from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_place_block_on_mark import BlockAtomicG1dCommon


class g1d_put_block_in_tray(BlockAtomicG1dCommon, TaskEnv):
    """G1-D: put the named block (target_block) in the tray; the others stay out, the tray where it was."""

    STEP_LIM = 1200  # 40 s

    def checks(self, env_idx):
        rm = self.reward_manager
        others = [b for b in self.blocks(env_idx) if b != "target_block"]
        return [
            rm.is_all_A_in_B(["target_block"], "tray"),
            rm.is_all_A_z_lower_than_B_bbox_zmax(["target_block"], "tray"),
            *[rm.is_A_not_in_B(b, "tray") for b in others],
            rm.is_block_near("tray", self.start[env_idx]["tray"], tol=0.02),
        ]

    def instructions(self, env_idx):
        zh, en = self.names(env_idx, "target_block", "zh"), self.names(env_idx, "target_block", "en")
        return [
            f"把{zh[0]}放进托盘里",
            f"将{zh[-1]}放到木托盘中",
            f"把{zh[0]}装进托盘",
            f"Put the {en[0]} in the tray.",
            f"Place the {en[-1]} into the wooden tray.",
            f"Drop the {en[0]} into the tray.",
        ]
