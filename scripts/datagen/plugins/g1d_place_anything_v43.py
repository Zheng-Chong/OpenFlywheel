"""scripts/datagen plugin for g1d_place_anything with place anything expert v4.3 (`--plugin g1d_place_anything_v43`):
scripts.g1d.place_v43.place_anything_actions (the first grasp chosen by its whole route to the set-down, then place
v4.2), through its Limiter. Acceptance, privileged data, audit and STEP_LIM are g1d_place_anything_v41's; the sidecar
adds the grasp tries (`grasp_attempts`), the routes checked (`place_route`) and how far the hand ended off over the spot
(`place_log`)."""

from scripts.datagen.plugins import g1d_place_anything_v41 as base
from scripts.datagen.plugins.g1d_place_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import place_v43, scripted_demo as demo

TASK, KIND = base.TASK, base.KIND
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    expert = staticmethod(place_v43.place_anything_actions)

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in self.expert(self.env))

    def summary(self):
        env = self.env
        return dict(
            super().summary(), place_route=getattr(env, "place_route", None), place_log=getattr(env, "place_log", None)
        )


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
