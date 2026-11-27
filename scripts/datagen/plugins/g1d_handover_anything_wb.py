"""scripts/datagen plugin for 19D fixed-base wholebody g1d_handover_anything (`--plugin g1d_handover_anything_wb`):
scripts.g1d.x_anything_wb.handover_anything_wb_actions (pre-grasp lift/waist adaptation + handover v4.3), through
WholebodyLimiter. Acceptance, privileged data and audit are g1d_handover_anything_v43's."""

from scripts.datagen.plugins import g1d_handover_anything_v43 as base
from scripts.datagen.plugins.g1d_handover_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import wholebody_demo as wb, x_anything_wb

TASK, KIND = base.TASK, base.KIND
STEP_LIM = 1800  # 60 s: accommodates pre-grasp lift/waist adaptation + handover v4.3 + hold
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = wb.WholebodyLimiter(self.env)
        return (limiter(action) for action in x_anything_wb.handover_anything_wb_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
