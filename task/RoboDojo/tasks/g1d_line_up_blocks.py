from env.environment.task_env import TaskEnv
from scripts.objects import block_atomic
from task.RoboDojo.tasks.g1d_place_block_on_mark import BlockAtomicG1dCommon


class g1d_line_up_blocks(BlockAtomicG1dCommon, TaskEnv):
    """G1-D: line the three blocks up from left to right in the order of their colours named (block_0 leftmost)."""

    STEP_LIM = 2100  # 70 s: three blocks

    LABELS = ("block_0", "block_1", "block_2")

    def checks(self, env_idx):
        return [self.reward_manager.is_blocks_in_row(self.LABELS)]

    def instructions(self, env_idx):
        paints = [block_atomic.paint_name(self.entries(env_idx)[b]["category_idx"]) for b in self.LABELS]
        zh = "、".join(block_atomic.PAINT_ZH[p] for p in paints)
        en = ", ".join(paints)
        return [
            f"把积木从左到右按{zh}的顺序排成一排",
            f"按{zh}的顺序，把三块积木从左往右排成一排",
            f"从左到右依次摆放{zh}积木，排成一排",
            f"Line the blocks up from left to right: {en}.",
            f"Put the three blocks in a row, left to right, in the order {en}.",
            f"Arrange the blocks in a row from left to right as {en}.",
        ]
