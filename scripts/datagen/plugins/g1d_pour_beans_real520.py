"""scripts/datagen plugin for g1d_pour_beans_real520 (倒豆子·深圳 520 孪生, benchmarks pour_beans_real520_sz@v2 and
pour_beans_real520_sz_plus@v1): the scripted bimanual pour of g1d_pour_beans_half_cup on the real 520 stand, the
right hand's full measuring cup over the cup held by the left until about FILL of the target's capacity has gone
over, both cups put back on the desk.

Acceptance real520_v2 (the task's own check, task/RoboDojo/tasks/g1d_pour_beans_real520.py): in each of the last HOLD
frames the target cup's cavity holds at least the layout's Generalization.acceptance.target_beans, at most
spill_beans lie in neither cup, both cups stand within UPRIGHT_DEG of upright and the source cup is back within REST_M
of its start height. Used to check that benchmark layouts can be done, not to record training data.
"""

import numpy as np

from scripts.datagen.plugins import g1d_pour_beans_plus as plus
from scripts.g1d import scripted_demo as demo
from scripts.objects import cup_profile

TASK = "g1d_pour_beans_real520"
ACCEPTANCE = "real520_v2"
STEP_LIM = 1500
TAIL = 60
HOLD = 30
LABELS, COUNT_KEYS, UPRIGHT_DEG, REST_M = plus.LABELS, plus.COUNT_KEYS, plus.UPRIGHT_DEG, plus.REST_M
FILL = 0.6  # of the target cup's capacity poured over (at least 1.3 x the acceptance's target_beans)
TARGET_SHARE, SPILL_SHARE = 0.25, 0.10  # without a layout acceptance, as v2


def accepted(counts, poses, start_z, target, spill):
    return bool(
        counts["target"] >= target
        and counts["outside"] <= spill
        and all(plus.tilt_deg(p) <= UPRIGHT_DEG for p in poses)
        and abs(poses[0][2] - start_z[0]) <= REST_M
    )


class Episode(plus.Episode):
    def __init__(self, env):
        super().__init__(env)
        layout = self.lm.saved_layouts[0] or {}
        acc = (layout.get("Generalization") or {}).get("acceptance") or {}
        total = len(self.initial)
        self.target = acc.get("target_beans", TARGET_SHARE * total)
        self.spill = acc.get("spill_beans", SPILL_SHARE * total)

    def actions(self):
        cavity, scale = self.cups["target_cup"]
        cap = cup_profile.capacity(cavity, scale, getattr(self.fluid, "grain", None))
        share = min(0.95, max(1.3 * self.target, FILL * cap) / len(self.initial))
        return demo.cup_to_cup_actions(
            self.env,
            pour_share=share,
            place=True,
            meet=demo.HALF_MEET,
            lift=demo.HALF_LIFT,
            dry_rate=demo.HALF_DRY_RATE,
            profiles=self.cups,
        )

    def step(self):
        super().step()
        return accepted(self.final_counts, self.poses(), self.start_z, self.target, self.spill)

    def summary(self):
        return dict(super().summary(), target_beans=self.target, spill_beans=self.spill)


def audit(privileged, quality):
    """Recompute the acceptance over the last HOLD frames from the recorded beans and cups."""
    particles, poses = privileged["particles"], privileged["cup_poses"].astype(float)
    cups = {label: (quality["cups"][label]["cavity"], quality["cups"][label]["scale"]) for label in LABELS}
    start_z = np.asarray(quality["start_z"])
    for points, frame in zip(particles[-HOLD:], poses[-HOLD:]):
        result = plus.count(points.astype(float), frame, cups)
        assert accepted(result, frame, start_z, quality["target_beans"], quality["spill_beans"]), result
    return dict(beans=int(quality["initial_counts"]["total"]))
