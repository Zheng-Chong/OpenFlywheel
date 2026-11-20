"""scripts/datagen plugin for g1d_sort_blocks_by_shape (按卡片分拣积木，按形状): as g1d_sort_blocks_by_color."""

from scripts.datagen.plugins.g1d_sort_blocks_by_color import ACCEPTANCE, HOLD, STEP_LIM, TAIL, Episode, audit

TASK = "g1d_sort_blocks_by_shape"

__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
