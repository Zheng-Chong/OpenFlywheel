"""scripts/datagen plugin for g1d_put_anything_in (把 X 放进 Y 里): g1d_place_anything's plugin with a container (its
docstring has the expert and the acceptance)."""

from scripts.datagen.plugins import g1d_place_anything as base

TASK = "g1d_put_anything_in"
KIND = "container"
ACCEPTANCE = base.ACCEPTANCE
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    task, kind = TASK, KIND


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
