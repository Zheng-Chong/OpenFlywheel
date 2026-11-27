"""scripts/datagen plugin for G1-D Sharpa Wave 61D wholebody pour beans (g1d_pour_beans_plus).

Acceptance sharpa_plus_v1:
- Same bean & cup criteria as plus_v3 (IN_B = 0.95 of beans in target cup over the last HOLD frames,
  both cups upright within 15 deg and within 1 cm of start_z, no cup-to-cup contact, distractors <= 2 cm).
- Both Sharpa Wave hands commanded open (hand_openness > OPEN_SHARE) over the last HOLD frames.
- Records to end of script plus TAIL still frames (RECORD_TO_END = True).

Collect with:
    --task g1d_pour_beans_plus --plugin g1d_pour_beans_plus_sharpa --env-cfg g1d_shenzhen_lab_sharpa_beans_wb
"""

import numpy as np

from scripts.datagen.plugins import g1d_pour_beans_plus as base
from scripts.dexhands import pour_beans_sharpa as expert

TASK = base.TASK
ACCEPTANCE = "sharpa_plus_v1"
STEP_LIM = 2100  # 70 s at 30 Hz: wholebody adapt + grasp + lift + pour + place + release
RECORD_TO_END = True
TAIL = 45  # 1.5 s still frames after script ends
HOLD = base.HOLD
LABELS = base.LABELS
IN_B = 0.95
OPEN_SHARE = 0.70  # mean openness of 22D hand joints (1.0 = all at 0 rad open pose)


def hand_keys(action):
    return sorted(k for k in action if "gripper" in k or "ee" in k)


def openness_from_limits(q, limits):
    """0 closed .. 1 open (mirrors DexHandTargets.hand_openness): 1 - mean(|q - rest| / span)."""
    lim = np.asarray(limits, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    rest = np.clip(0.0, lim[:, 0], lim[:, 1])
    span = np.maximum(lim[:, 1] - lim[:, 0], 1e-6)
    return float(1.0 - np.mean(np.abs(q - rest) / span))


class Episode(base.Episode):
    IN_B = IN_B

    def __init__(self, env):
        super().__init__(env)
        self.robots_by_key = {
            env.robot_manager.process_name(r.gripper_name): r
            for r in env.robot_manager.robot_list
            if r.type == "target"
        }
        self.hand_open = np.zeros(2, dtype=float)
        self.history["hand_openness"] = []

    def actions(self):
        for action in expert.sharpa_pour_actions(self.env):
            vals = []
            for key in hand_keys(action):
                rb = self.robots_by_key.get(key)
                if rb is not None and hasattr(rb, "hand_openness"):
                    vals.append(rb.hand_openness(action[key]))
                else:
                    vals.append(float(np.asarray(action[key], float).reshape(-1).mean()))
            self.hand_open = np.asarray(vals, dtype=float)
            yield action

    def step(self):
        self.history["hand_openness"].append(self.hand_open.astype(np.float32))
        return super().step() and bool((self.hand_open > OPEN_SHARE).all())


def audit(privileged, quality):
    """plus_v3 bean/cup/distractor audit plus both Sharpa hands commanded open in the last HOLD frames."""
    metrics = base.audit(privileged, quality, in_b=IN_B)
    openness = np.asarray(privileged["hand_openness"], dtype=float)
    assert openness.shape == (quality["frames"], 2), openness.shape
    assert (openness[-HOLD:] > OPEN_SHARE).all(), f"a Sharpa hand was not open in the last {HOLD} frames"
    return dict(metrics, tail_frames=TAIL, min_tail_hand_openness=round(float(openness[-HOLD:].min()), 4))
