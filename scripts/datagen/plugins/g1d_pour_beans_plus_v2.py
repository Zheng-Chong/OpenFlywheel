"""scripts/datagen plugin for g1d_pour_beans_plus, acceptance plus_v2: the expert and checks of plus_v1
(g1d_pour_beans_plus.py), recorded to the end of the script (both cups set down, released, the hands backed off)
and then TAIL actions of the hands holding still, so the data shows what follows a finished pour. Under plus_v1 the
recording stopped once the cups had stood settled for HOLD frames, about 25 frames after the second gripper opened;
an evaluated policy that is not done by then keeps running to STEP_LIM in states the data never showed (re-grasps
and second pours, docs/pour-beans-plus-failures-20261109.md).

Accepted when in each of the last HOLD frames the plus_v1 settle holds and both grippers are commanded open
(command > OPEN, the 0 closed .. 1 open scale of scripts/datagen/CONTRACT.md), with the cups never touching and no
distractor moved over the whole episode. Evaluation keeps plus_v1 (scripts/g1d/remote_policy.py runs the task's
plugin), so benchmark results stay comparable. Collect with collect.py / run_batch.py --plugin g1d_pour_beans_plus_v2.
"""

import numpy as np

from scripts.datagen.plugins import g1d_pour_beans_plus as base

TASK, HOLD, STEP_LIM, LABELS = base.TASK, base.HOLD, base.STEP_LIM, base.LABELS
ACCEPTANCE = "plus_v2"
RECORD_TO_END = True
TAIL = 90  # actions (3 s) the backed-off hands hold still after the script
OPEN = 0.5


def gripper_keys(action):
    return sorted(k for k in action if "_ee_" in k)


class Episode(base.Episode):
    def __init__(self, env):
        super().__init__(env)
        self.grip = None
        self.history["grip_command"] = []

    def actions(self):
        for action in super().actions():
            self.grip = np.array([float(np.asarray(action[k]).reshape(-1)[0]) for k in gripper_keys(action)])
            yield action

    def step(self):
        settled = super().step()
        grip = self.grip if self.grip is not None else np.zeros(2)
        self.history["grip_command"].append(grip.astype(np.float32))
        return settled and bool((grip > OPEN).all())

    def summary(self):
        return dict(super().summary(), gripper_open_above=OPEN, record_to_end=True)


def audit(privileged, quality):
    """plus_v1's audit, and both grippers commanded open in each of the last HOLD frames."""
    metrics = base.audit(privileged, quality)
    grip = np.asarray(privileged["grip_command"], float)
    assert grip.shape[0] == quality["frames"], grip.shape
    assert (grip[-HOLD:] > OPEN).all(), f"a gripper closed in the last {HOLD} frames"
    return dict(metrics, tail_frames=TAIL)
