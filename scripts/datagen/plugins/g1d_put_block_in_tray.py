"""scripts/datagen plugin for g1d_put_block_in_tray (放进托盘): see _block_atomic.py."""

from scripts.datagen.plugins._block_atomic import ACCEPTANCE, HOLD, TAIL, Episode, audit

TASK = "g1d_put_block_in_tray"
STEP_LIM = 1200  # actions: the task's

__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
