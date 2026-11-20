"""scripts/datagen plugin for g1d_put_anything_in with place anything expert v4.3 (`--plugin
g1d_put_anything_in_v43`): g1d_place_anything_v43's plugin with a container (g1d_put_anything_in's acceptance) and a
slender X tried deep first (place_v43.put_anything_in_actions)."""

from scripts.datagen.plugins import g1d_place_anything_v43 as base, g1d_put_anything_in as put
from scripts.datagen.plugins.g1d_put_anything_in import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import place_v43

TASK, KIND = put.TASK, put.KIND
STEP_LIM = base.STEP_LIM
TAIL, HOLD = put.TAIL, put.HOLD
audit = put.audit


class Episode(base.Episode):
    task, kind = TASK, KIND
    expert = staticmethod(place_v43.put_anything_in_actions)


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
