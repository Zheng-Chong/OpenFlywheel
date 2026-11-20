"""scripts/datagen plugin for g1d_pour_beans_plus: the plus_v3 expert and acceptance (g1d_pour_beans_plus_v3.py) at the
real G1-D's motion limits (scripted_demo.Limiter: joint speed, joint-limit margin, gripper rate), as the block, pick
and open experts are collected; scripted_demo.LIMITED only limits the demo runs.

Each expert command is sent through the Limiter until the arms and grippers have reached it, and only then does the
script take its next step: the script waits for the slower real arm instead of running ahead of it. Only limiting
the commands let the script run ahead: its mid-air settle waits at most C2C_SETTLE actions, and the source cup met
the still-arriving target cup at the handover (2026-11-18 probe). The episode still has STEP_LIM (the task's 50 s).
grip_command, which the audit checks, records the gripper command sent. Collect with --plugin
g1d_pour_beans_plus_limited.
"""

import numpy as np

from scripts.datagen.plugins import g1d_pour_beans_plus_v2 as v2, g1d_pour_beans_plus_v3 as v3
from scripts.datagen.plugins.g1d_pour_beans_plus_v3 import ACCEPTANCE  # noqa: F401 (experts.py follows the import)
from scripts.g1d import scripted_demo as demo

TASK, HOLD, STEP_LIM, LABELS = v3.TASK, v3.HOLD, v3.STEP_LIM, v3.LABELS
RECORD_TO_END, TAIL, OPEN, IN_B = v3.RECORD_TO_END, v3.TAIL, v3.OPEN, v3.IN_B
audit = v3.audit


def reached(limiter, action, sent):
    """Whether the limited command `sent` is the expert's `action` (arms clipped to the Limiter's joint range)."""
    for key, value in action.items():
        want = np.asarray(value, float).reshape(-1)
        if key in limiter.lim:
            want = np.clip(want, *limiter.lim[key].T)
        if not np.allclose(np.asarray(sent[key], float).reshape(-1), want, atol=1e-9):
            return False
    return True


class Episode(v3.Episode):
    def actions(self):
        limiter = demo.Limiter(self.env)
        for action in super().actions():
            while True:
                sent = limiter(action)
                self.grip = np.array([float(np.asarray(sent[k]).reshape(-1)[0]) for k in v2.gripper_keys(sent)])
                yield sent
                if reached(limiter, action, sent):
                    break
