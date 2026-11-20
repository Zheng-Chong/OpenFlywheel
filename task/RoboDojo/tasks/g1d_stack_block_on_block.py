from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_place_block_on_mark import BlockAtomicG1dCommon


class g1d_stack_block_on_block(BlockAtomicG1dCommon, TaskEnv):
    """G1-D: stack the named block (top_block) on the named base (base_block); the base stays where it was."""

    STEP_LIM = 1200  # 40 s

    def checks(self, env_idx):
        rm = self.reward_manager
        top, base = self.size(env_idx, "top_block"), self.size(env_idx, "base_block")
        return [
            rm.is_block_stacked_on("top_block", "base_block", top, base),
            rm.is_block_near("base_block", self.start[env_idx]["base_block"], tol=0.03),
        ]

    def instructions(self, env_idx):
        t, b = self.names(env_idx, "top_block", "zh"), self.names(env_idx, "base_block", "zh")
        te, be = self.names(env_idx, "top_block", "en"), self.names(env_idx, "base_block", "en")
        return [
            f"把{t[0]}叠到{b[0]}上",
            f"把{t[-1]}放到{b[-1]}上面",
            f"将{t[0]}摞在{b[0]}上",
            f"Stack the {te[0]} on the {be[0]}.",
            f"Put the {te[-1]} on top of the {be[-1]}.",
            f"Place the {te[0]} onto the {be[0]}.",
        ]
