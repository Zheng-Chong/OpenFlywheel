"""scripts/datagen plugin for 19D fixed-base wholebody g1d_put_anything_in (`--plugin g1d_put_anything_in_wb`):
scripts.g1d.x_anything_wb.put_anything_in_wb_actions (pre-grasp lift/waist adaptation + put v4.4), through
WholebodyLimiter. Acceptance, privileged data and audit are g1d_put_anything_in_v44's."""

from scripts.datagen.plugins import g1d_put_anything_in_v44 as base
from scripts.datagen.plugins.g1d_put_anything_in import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import wholebody_demo as wb, x_anything_wb

TASK, KIND = base.TASK, base.KIND
STEP_LIM = 1800  # 60 s: accommodates pre-grasp lift/waist adaptation + put v4.4 + regrasp + hold
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    expert = staticmethod(x_anything_wb.put_anything_in_wb_actions)

    def actions(self):
        limiter = wb.WholebodyLimiter(self.env)
        return (limiter(action) for action in self.expert(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
