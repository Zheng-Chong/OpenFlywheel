"""scripts/datagen plugin for g1d_pick_anything with the pick anything expert v4.1's grasp (`--plugin
g1d_pick_anything_v41`): scripts.g1d.grasp_v41.pick_anything_actions, through its Limiter; otherwise
g1d_pick_anything_v4's plugin (acceptance pick_anything_v1, STEP_LIM, `grasp_attempts`)."""

from scripts.datagen.plugins import g1d_pick_anything_v4 as base
from scripts.datagen.plugins.g1d_pick_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import grasp_v41, scripted_demo as demo

TASK = base.TASK
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in grasp_v41.pick_anything_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
