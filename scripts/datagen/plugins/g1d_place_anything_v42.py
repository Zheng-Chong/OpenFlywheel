"""scripts/datagen plugin for g1d_place_anything with place anything expert v4.2 (`--plugin g1d_place_anything_v42`):
scripts.g1d.place_v42.place_anything_actions (pick anything expert v4.1's grasp, a place stage planned within the
arm's reach), through its Limiter. Acceptance, privileged data, audit and STEP_LIM are g1d_place_anything_v41's; the
sidecar adds the grasp tries (`grasp_attempts`)."""

from scripts.datagen.plugins import g1d_place_anything_v41 as base
from scripts.datagen.plugins.g1d_place_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import place_v42, scripted_demo as demo

TASK, KIND = base.TASK, base.KIND
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in place_v42.place_anything_actions(self.env))


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
