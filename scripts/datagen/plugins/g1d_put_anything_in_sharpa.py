"""scripts/datagen plugin for G1-D Sharpa Wave 61D wholebody g1d_put_anything_in (把 X 放进 Y 里).

Collect with:
    --task g1d_put_anything_in --plugin g1d_put_anything_in_sharpa --env-cfg g1d_office_sharpa_x_anything_wb
"""

from scripts.datagen.plugins import g1d_place_anything_sharpa as base

TASK = "g1d_put_anything_in"
KIND = "container"
ACCEPTANCE = base.ACCEPTANCE
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    task, kind = TASK, KIND


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
