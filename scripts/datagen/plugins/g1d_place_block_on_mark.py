"""scripts/datagen plugin for g1d_place_block_on_mark (放到标记上): see _block_atomic.py."""

from scripts.datagen.plugins._block_atomic import ACCEPTANCE, HOLD, TAIL, Episode, audit

TASK = "g1d_place_block_on_mark"
STEP_LIM = 900  # actions: the task's

__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
