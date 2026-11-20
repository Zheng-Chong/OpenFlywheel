"""scripts/datagen plugin for g1d_pick_named_block (按指令拿起): see _block_atomic.py."""

from scripts.datagen.plugins._block_atomic import ACCEPTANCE, HOLD, TAIL, Episode, audit

TASK = "g1d_pick_named_block"
STEP_LIM = 900  # actions: the task's

__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
