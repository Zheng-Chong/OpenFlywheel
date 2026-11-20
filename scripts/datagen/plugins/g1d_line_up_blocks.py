"""scripts/datagen plugin for g1d_line_up_blocks (排成一排): see _block_atomic.py."""

from scripts.datagen.plugins._block_atomic import ACCEPTANCE, HOLD, TAIL, Episode, audit

TASK = "g1d_line_up_blocks"
STEP_LIM = 2100  # actions: the task's

__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
