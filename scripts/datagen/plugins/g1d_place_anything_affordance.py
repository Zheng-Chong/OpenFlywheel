"""scripts/datagen plugin for g1d_place_anything using AffordancePlaceRobot (`--plugin g1d_place_anything_affordance`).

Consumes object `affordance.json` sidecars with the `place` semantic `part_tag` policy
(preferring `body_top`/`body`/`handle`/`neck`, excluding `base`/`blade`/`spout`) and
combines it with v4.2/v4.4 `ik_within` placing and `PATTERN_SETTLE` stall-break.
Automatically falls back to GraspRobot41's BBox heuristic when no affordance applies.
"""

from scripts.datagen.plugins import g1d_place_anything_v41 as base
from scripts.datagen.plugins.g1d_place_anything import ACCEPTANCE  # noqa: F401
from scripts.g1d import affordance_grasp, scripted_demo as demo

TASK, KIND = base.TASK, base.KIND
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    expert = staticmethod(affordance_grasp.place_anything_affordance_actions)

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in self.expert(self.env))

    def summary(self):
        env = self.env
        return dict(super().summary(), place_log=getattr(env, "place_log", None))


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
