"""scripts/datagen plugin for g1d_sort_anything with the v3 expert (`--plugin g1d_sort_anything_v3`, the 11-20 A/B's
baseline): scripts.g1d.sort_anything_demo.sort_anything_actions, each item picked as g1d_pick_anything v3 (a miss
grasped once more) and set down as g1d_place_anything v3, through scripted_demo's Limiter. Acceptance, privileged data
and audit are g1d_sort_anything's (sort_anything_v1)."""

from scripts.datagen.plugins import g1d_sort_anything as base
from scripts.datagen.plugins.g1d_sort_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import scripted_demo as demo, sort_anything_demo

TASK = base.TASK
STEP_LIM = 2400  # actions: the task's
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in sort_anything_demo.sort_anything_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
