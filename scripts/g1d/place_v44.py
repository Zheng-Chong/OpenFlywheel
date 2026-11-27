"""Place anything expert v4.4: place_v43 with PATTERN_SETTLE stall-break in Robot.move().

When an arm reaches a joint limit or physical obstacle during a settle phase, Robot.move() would otherwise burn all
60 frames of PATTERN_SETTLE waiting to get within 5 mm of an unreachable target. StallBreakMixin exits the settle
loop early once ||arm.grasp_point() - prev_pos|| < SETTLE_STALL_M (0.5 mm) for SETTLE_STALL_FRAMES (6) consecutive
settle frames. Kept out of scripted_demo.py and place_v43.py so registered earlier expert sources stay unchanged.
"""

import numpy as np

from scripts.g1d import place_v42, place_v43, scripted_demo as demo

SETTLE_STALL_M = 0.0005  # m: grasp point displacement below which a settle frame counts as stalled
SETTLE_STALL_FRAMES = 6  # consecutive stalled settle frames before breaking out of the settle loop


class StallBreakMixin:
    """Overrides Robot.move() to break out of the settle loop when the grasp point stalls."""

    settle_stall_m = SETTLE_STALL_M
    settle_stall_frames = SETTLE_STALL_FRAMES

    def move(self, arm, keys):
        for path in [keys] if self.settle is None else [[key] for key in keys]:
            for point, rot, grip in demo.waypoints(arm.grasp_point(), arm.pose()[1], path):
                yield self._command(arm, point, rot, grip)
            if self.settle is not None:
                point, rot, grip, _ = path[-1]
                tol, most = self.settle
                prev_pos = None
                stall = 0
                for _ in range(most):
                    pos = np.asarray(arm.grasp_point(), dtype=float)
                    if np.linalg.norm(pos - point) < tol:
                        break
                    if prev_pos is not None:
                        if np.linalg.norm(pos - prev_pos) < self.settle_stall_m:
                            stall += 1
                            if stall >= self.settle_stall_frames:
                                break
                        else:
                            stall = 0
                    prev_pos = pos.copy()
                    yield self._command(arm, point, rot, grip)


class PlaceRobot44(StallBreakMixin, place_v43.PlaceRobot43):
    pass


class PutRobot44(StallBreakMixin, place_v43.PutRobot43):
    pass


def place_anything_actions(env):
    """g1d_place_anything with expert v4.4 (v4.3 + PATTERN_SETTLE stall-break)."""
    return place_v42.place_anything_actions(env, PlaceRobot44)


def put_anything_in_actions(env):
    """g1d_put_anything_in with expert v4.4 (v4.3 + PATTERN_SETTLE stall-break)."""
    return place_v42.place_anything_actions(env, PutRobot44)
