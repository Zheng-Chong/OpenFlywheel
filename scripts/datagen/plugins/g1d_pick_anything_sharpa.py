"""scripts/datagen plugin for G1-D Sharpa Wave 61D wholebody g1d_pick_anything.

Uses `scripts.dexhands.x_anything_sharpa.sharpa_pick_anything_actions` (multi-primitive grasp: power_wrap /
tripod_pinch / lateral + thumb-first tactile-gated closure + adaptive waist/lift, rate-limited by WholeBodyLimiter).
Acceptance: identical target lift & distractor stability to `pick_anything_v1` (g1d_pick_anything.py).

Collect with:
    --task g1d_pick_anything --plugin g1d_pick_anything_sharpa --env-cfg g1d_office_sharpa_x_anything_wb
"""

from scripts.datagen.plugins import g1d_pick_anything as base
from scripts.dexhands import x_anything_sharpa as expert

TASK = base.TASK
ACCEPTANCE = "pick_anything_sharpa_v1"
STEP_LIM = 900  # 30 s at 30 Hz: wholebody adapt + side approach + tactile closure + column lift
TAIL = base.TAIL
HOLD = base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        return expert.sharpa_pick_anything_actions(self.env)


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
