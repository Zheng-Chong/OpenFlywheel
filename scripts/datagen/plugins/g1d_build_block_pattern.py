"""scripts/datagen plugin for g1d_build_block_pattern (照图搭积木): the scripted expert builds the pattern on the card
from the scattered wood blocks (scripted_demo.block_pattern_actions, through its Limiter).

Acceptance pattern_built_v2, the locked platform definition (real2sim/tasks/g1d_build_block_pattern/registration.json)
with tighter tolerances than the task's check, held for HOLD frames in a row: every block within XY_TOL / Z_TOL of its
place in the pattern (the pattern frame taken from its anchor block, anywhere on the table, turned any way; blocks of
the same shape and paint may swap places), upright within TILT_DEG and turned within YAW_DEG of the pattern (mod its
symmetry), and both grippers open at least OPEN. (v1, round 1: no swapping, the first three patterns only.) An episode
runs for its pattern's block_patterns.step_lim, as the task does.
"""

import numpy as np

from env.reward_manager.pattern_match import pattern_error
from scripts.g1d import scripted_demo as demo
from scripts.objects import block_patterns

TASK = "g1d_build_block_pattern"
ACCEPTANCE = "pattern_built_v2"
STEP_LIM = max(block_patterns.step_lim(p) for p in block_patterns.PATTERNS)  # each episode: its own
TAIL = 60  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
XY_TOL, Z_TOL, TILT_DEG, YAW_DEG = 0.010, 0.006, 6.0, 10.0  # the task's check: 15 mm, 10 mm, 10 deg, 15 deg
OPEN = 0.8  # gripper opening (0 closed .. 1 open), as the task's check


def gripper_openness(env):
    """Opening of each target arm's gripper, 0 closed .. 1 open (func_parser.is_all_gripper_open's measure)."""
    rm, out = env.robot_manager, []
    for robot in rm.robot_list:
        if robot.type != "target":
            continue
        val = rm.get_end_effector_real_val(robot=robot, env_idx_list=[0])[0]
        if hasattr(robot, "hand_openness"):
            out.append(robot.hand_openness(val))
            continue
        val = float(np.mean(val)) if isinstance(val, list | np.ndarray) else float(val)
        lo, hi = robot.gripper_scale
        out.append((val - lo) / (hi - lo) if robot.gripper_move["sign"] == 1 else (hi - val) / (hi - lo))
    return out


def errors(poses, pattern):
    """(xy m, z m, tilt deg, yaw deg) of block poses (n, 7: xyz, wxyz) from `pattern`."""
    poses = np.asarray(poses, dtype=float)
    return pattern_error(
        poses[:, :3],
        [demo.from_wxyz(p[3:]).as_matrix() for p in poses],
        block_patterns.targets(pattern),
        [block_patterns.YAW_PERIOD[cat] for cat, _, _ in pattern["pieces"]],
        pattern["anchor"],
        block_patterns.groups(pattern),
    )


def built(err, openness):
    xy, z, tilt, yaw = err
    return xy < XY_TOL and z < Z_TOL and tilt < TILT_DEG and yaw < YAW_DEG and min(openness) >= OPEN


class Episode:
    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        self.pattern = env.pattern[0]
        env.step_lim = block_patterns.step_lim(self.pattern)
        self.labels = block_patterns.labels(self.pattern)
        self.reject = None
        self.last = None
        self.history = dict(block_poses=[], gripper_openness=[])

    def poses(self):
        return np.array(
            [
                np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=label)])
                for label in self.labels
            ]
        )

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.block_pattern_actions(self.env))

    def step(self):
        """After each action: record the blocks and grippers; True while the pattern stands built, hands open."""
        poses, openness = self.poses(), gripper_openness(self.env)
        self.history["block_poses"].append(poses.astype(np.float32))
        self.history["gripper_openness"].append(np.asarray(openness, dtype=np.float32))
        self.last = errors(poses, self.pattern)
        return built(self.last, openness)

    def privileged(self):
        """Per frame after its action: block poses (T, n, 7) in label order, gripper openness (T, 2)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        index = block_patterns.PATTERNS.index(self.pattern)
        out = dict(pattern=self.pattern["name"], pattern_index=index, labels=self.labels)
        if self.last:
            out["final_error"] = dict(zip(("xy_m", "z_m", "tilt_deg", "yaw_deg"), map(float, self.last)))
        return out


def audit(privileged, quality):
    """Recompute the acceptance from the recorded blocks and grippers in each of the last HOLD frames. Raises
    AssertionError on a violation; returns the worst errors over those frames."""
    pattern = block_patterns.PATTERNS[quality["pattern_index"]]
    poses, openness = privileged["block_poses"], privileged["gripper_openness"]
    assert poses.shape == (quality["frames"], len(pattern["pieces"]), 7), poses.shape
    assert len(openness) == quality["frames"]
    worst = np.zeros(4)
    for frame, opened in zip(poses[-HOLD:], openness[-HOLD:]):
        err = errors(frame, pattern)
        assert built(err, opened), (err, opened)
        worst = np.maximum(worst, err)
    keys = ("max_xy_m", "max_z_m", "max_tilt_deg", "max_yaw_deg")
    return dict(pattern=pattern["name"], **{k: float(v) for k, v in zip(keys, worst)})
