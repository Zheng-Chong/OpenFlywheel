"""scripts/datagen plugin for 19D fixed-base wholebody g1d_pick_anything (`--plugin g1d_pick_anything_wb`):
scripts.g1d.x_anything_wb.pick_anything_wb_actions (pre-grasp lift/waist adaptation + v4.1 grasp + stall-break),
through WholebodyLimiter. Acceptance, privileged data and audit are g1d_pick_anything_v41's."""

from scripts.datagen.plugins import g1d_pick_anything_v41 as base
from scripts.datagen.plugins.g1d_pick_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import wholebody_demo as wb, x_anything_wb

TASK = base.TASK
STEP_LIM = 900  # 30 s: accommodates pre-grasp lift/waist adaptation + v4.1 grasp/regrasps + hold
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = wb.WholebodyLimiter(self.env)
        return (limiter(action) for action in x_anything_wb.pick_anything_wb_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
