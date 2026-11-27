"""scripts/datagen plugin for g1d_sort_anything with sort anything expert v4.4 (`--plugin g1d_sort_anything_v44`):
sort_v43 plus PATTERN_SETTLE stall-break (scripts.g1d.sort_v44.sort_anything_actions), through its Limiter.
Acceptance, privileged data, audit and STEP_LIM are g1d_sort_anything's."""

from scripts.datagen.plugins import g1d_sort_anything as base
from scripts.datagen.plugins.g1d_sort_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import scripted_demo as demo, sort_v44

TASK = base.TASK
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in sort_v44.sort_anything_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
