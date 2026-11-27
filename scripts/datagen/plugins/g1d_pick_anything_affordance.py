"""scripts/datagen plugin for g1d_pick_anything using AffordanceGraspRobot (`--plugin g1d_pick_anything_affordance`).

Consumes object `affordance.json` sidecars (from the asset directory or `OPENFLYWHEEL_AFFORDANCE_DIR`) and
falls back automatically to GraspRobot41's bounding-box heuristic when no affordance applies.
"""

from scripts.datagen.plugins import g1d_pick_anything_v4 as base
from scripts.datagen.plugins.g1d_pick_anything import ACCEPTANCE  # noqa: F401
from scripts.g1d import affordance_grasp, scripted_demo as demo

TASK = base.TASK
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in affordance_grasp.pick_anything_affordance_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
