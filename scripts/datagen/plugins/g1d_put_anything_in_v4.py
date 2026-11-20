"""scripts/datagen plugin for g1d_put_anything_in with the pick anything expert v4's grasp (`--plugin
g1d_put_anything_in_v4`): g1d_place_anything_v4's plugin with a container (g1d_put_anything_in's acceptance)."""

from scripts.datagen.plugins import g1d_place_anything_v4 as base, g1d_put_anything_in as put
from scripts.datagen.plugins.g1d_put_anything_in import ACCEPTANCE  # noqa: F401 (experts.py follows the import)

TASK, KIND = put.TASK, put.KIND
STEP_LIM = base.STEP_LIM
TAIL, HOLD = put.TAIL, put.HOLD
audit = put.audit


class Episode(base.Episode):
    task, kind = TASK, KIND


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
