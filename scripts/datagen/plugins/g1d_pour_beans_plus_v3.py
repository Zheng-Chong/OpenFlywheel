"""scripts/datagen plugin for g1d_pour_beans_plus, acceptance plus_v3: records to the end of the script plus TAIL
still frames with both grippers commanded open (like plus_v2, g1d_pour_beans_plus_v2.py), and accepts >= 95% of the
beans in B over the last HOLD frames (matching the task's IN_CUP = 0.95 success threshold) while keeping no cup-to-cup
contact and distractor shift <= 2 cm throughout the episode.

Benchmark evaluation keeps plus_v1 (g1d_pour_beans_plus.py, IN_B = 1.0) so pour_beans_plus@v1 scores stay comparable.
Collect with collect.py / run_batch.py --plugin g1d_pour_beans_plus_v3 (or --expert v5).
"""

import numpy as np

from scripts.datagen.plugins import g1d_pour_beans_plus as base, g1d_pour_beans_plus_v2 as v2

TASK, HOLD, STEP_LIM, LABELS = v2.TASK, v2.HOLD, v2.STEP_LIM, v2.LABELS
ACCEPTANCE = "plus_v3"
RECORD_TO_END = v2.RECORD_TO_END
TAIL = v2.TAIL
OPEN = v2.OPEN
IN_B = 0.95


class Episode(v2.Episode):
    IN_B = IN_B


def audit(privileged, quality):
    """plus_v1's audit at IN_B = 0.95, and both grippers commanded open in each of the last HOLD frames."""
    metrics = base.audit(privileged, quality, in_b=IN_B)
    grip = np.asarray(privileged["grip_command"], float)
    assert grip.shape[0] == quality["frames"], grip.shape
    assert (grip[-HOLD:] > OPEN).all(), f"a gripper closed in the last {HOLD} frames"
    return dict(metrics, tail_frames=TAIL)
