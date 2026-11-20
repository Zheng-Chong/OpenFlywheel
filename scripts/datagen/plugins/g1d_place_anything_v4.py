"""scripts/datagen plugin for g1d_place_anything with the pick anything expert v4's grasp (`--plugin
g1d_place_anything_v4`): scripted_demo.place_anything_v4_actions (place_anything_actions on GraspRobot, see
docs/pick-expert-v4.md), through its Limiter. Acceptance, privileged data and audit are g1d_place_anything's
(place_anything_v1); the sidecar adds the grasp tries (`grasp_attempts`)."""

from scripts.datagen.plugins import g1d_place_anything as base
from scripts.datagen.plugins.g1d_place_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import scripted_demo as demo

TASK, KIND = base.TASK, base.KIND
STEP_LIM = base.STEP_LIM + 500  # actions: room for two more grasp tries (+200 cut third tries short)
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.place_anything_v4_actions(self.env))

    def summary(self):
        return dict(super().summary(), grasp_attempts=list(getattr(self.env, "grasp_attempts", [])))


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
