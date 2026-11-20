from env.environment.task_env import TaskEnv
from scripts.objects import block_atomic
from task.RoboDojo.tasks.g1d_place_block_on_mark import BlockAtomicG1dCommon


class g1d_rotate_block_parallel(BlockAtomicG1dCommon, TaskEnv):
    """G1-D: turn the long block (block) parallel to the table edge (its long side along X), about where it lay."""

    def checks(self, env_idx):
        rm = self.reward_manager
        return [
            rm.is_block_yaw_aligned("block", axis_deg=0.0, period=180, tol=block_atomic.ROTATE_TOL),
            rm.is_block_near("block", self.start[env_idx]["block"], tol=block_atomic.ROTATE_STAY),
        ]

    def instructions(self, env_idx):
        zh, en = self.names(env_idx, "block", "zh"), self.names(env_idx, "block", "en")
        return [
            f"把{zh[0]}转到和桌边平行",
            f"将{zh[-1]}摆正，和桌子边缘平行",
            f"把{zh[0]}转正",
            f"Turn the {en[0]} parallel to the table edge.",
            f"Straighten the {en[-1]} so it lines up with the table edge.",
            f"Rotate the {en[0]} until it is parallel to the edge of the table.",
        ]
