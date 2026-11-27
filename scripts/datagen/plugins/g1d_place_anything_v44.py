"""scripts/datagen plugin for g1d_place_anything with place anything expert v4.4 (`--plugin g1d_place_anything_v44`):
place_v43 plus PATTERN_SETTLE stall-break (scripts.g1d.place_v44.place_anything_actions), through its Limiter.
Acceptance, privileged data, audit and STEP_LIM are g1d_place_anything_v43's."""

from scripts.datagen.plugins import g1d_place_anything_v43 as base
from scripts.datagen.plugins.g1d_place_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import place_v44

TASK, KIND = base.TASK, base.KIND
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    expert = staticmethod(place_v44.place_anything_actions)


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
