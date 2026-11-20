"""scripts/datagen plugin for g1d_handover_anything with handover expert v3 (`--plugin g1d_handover_anything_v43`):
scripts.g1d.handover_v43.handover_anything_actions (v2's handover, place v4.3's place stage for the taker), through its
Limiter. Acceptance, privileged data, audit and STEP_LIM are g1d_handover_anything's; the sidecar adds how far the
taker's hand ended off over the spot and X off in the hand (`place_log`)."""

from scripts.datagen.plugins import g1d_handover_anything as base
from scripts.datagen.plugins.g1d_handover_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import handover_v43, scripted_demo as demo

TASK, KIND = base.TASK, base.KIND
STEP_LIM, TAIL, HOLD = base.STEP_LIM, base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in handover_v43.handover_anything_actions(self.env))

    def summary(self):
        return dict(super().summary(), place_log=getattr(self.env, "place_log", None))


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
