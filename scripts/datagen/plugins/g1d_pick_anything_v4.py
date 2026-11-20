"""scripts/datagen plugin for g1d_pick_anything with the pick anything expert v4 (`--plugin g1d_pick_anything_v4` or
`--expert v4`): scripted_demo.pick_anything_v4_actions (GraspRobot: v3's grasp first, checked and retried, see
docs/pick-expert-v4.md), through its Limiter. Acceptance, privileged data and audit are g1d_pick_anything's
(pick_anything_v1); the sidecar adds the grasp tries (`grasp_attempts`: grasp, hand, result). STEP_LIM leaves room for
GRASP_TRIES tries."""

from scripts.datagen.plugins import g1d_pick_anything as base
from scripts.datagen.plugins.g1d_pick_anything import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import scripted_demo as demo

TASK = base.TASK
# actions: three tries of ~250 (a knocked target is chased), the lift and the hold; 900 cut about one in ten third
# tries short (A/B round 3)
STEP_LIM = 1200
TAIL, HOLD = base.TAIL, base.HOLD
audit = base.audit


class Episode(base.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.pick_anything_v4_actions(self.env))

    def summary(self):
        return dict(super().summary(), grasp_attempts=list(getattr(self.env, "grasp_attempts", [])))


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
