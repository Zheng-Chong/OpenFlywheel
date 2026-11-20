"""scripts/datagen plugin for g1d_sort_anything with the pick anything expert v4.1's grasp (`--plugin
g1d_sort_anything_v41`): scripts.g1d.sort_anything_demo.sort_anything_actions on grasp_v41.GraspRobot41, one pick per
item (v4.1 retries itself), place v3, through scripted_demo's Limiter. Acceptance, privileged data, audit and sidecar
are g1d_sort_anything's (sort_anything_v1)."""

from scripts.datagen.plugins import g1d_sort_anything as base
from scripts.datagen.plugins.g1d_sort_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import grasp_v41, scripted_demo as demo, sort_anything_demo

TASK = base.TASK
STEP_LIM = 3400  # actions: the task's 2400 and room for two more grasp tries of two items
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        script = sort_anything_demo.sort_anything_actions(self.env, grasp_v41.GraspRobot41, tries=1)
        return (limiter(action) for action in script)


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
