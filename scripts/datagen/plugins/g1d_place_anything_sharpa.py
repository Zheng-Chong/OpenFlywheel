"""scripts/datagen plugin for G1-D Sharpa Wave 61D wholebody g1d_place_anything (把 X 放到 Y 上).

Uses `scripts.dexhands.x_anything_sharpa.sharpa_place_anything_actions` (multi-primitive Sharpa grasp + tactile-gated
closure + adaptive waist/lift + smooth placement & fingers-first release). Acceptance matches `place_anything_v1`
with Sharpa `hand_openness` measured via `gripper_openness(env)`.

Collect with:
    --task g1d_place_anything --plugin g1d_place_anything_sharpa --env-cfg g1d_office_sharpa_x_anything_wb
"""

from scripts.datagen.plugins import g1d_place_anything as base
from scripts.dexhands import x_anything_sharpa as expert

TASK = base.TASK
KIND = base.KIND
ACCEPTANCE = "place_anything_sharpa_v1"
STEP_LIM = 1200  # 40 s at 30 Hz
TAIL = base.TAIL
HOLD = base.HOLD
audit = base.audit


class Episode(base.Episode):
    task, kind = TASK, KIND

    def actions(self):
        return expert.sharpa_place_anything_actions(self.env, kind=self.kind)


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
