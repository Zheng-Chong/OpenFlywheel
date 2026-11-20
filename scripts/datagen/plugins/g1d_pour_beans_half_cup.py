"""scripts/datagen plugin for g1d_pour_beans_half_cup (倒半杯): the scripted bimanual pour of about half a cup of red
beans from the tall measuring cup into the grey or blue cup, both cups put back on the desk.

Acceptance half_cup_v1, the locked platform definition (real2sim/tasks/g1d_pour_beans_half_cup/registration.json),
held for HOLD frames in a row:
- the target cup's cavity holds FILL of its capacity in beans (capacity: round_cup_shape.capacity);
- at most SPILL of the beans that left the tall cup are outside both cups;
- both cups stand within UPRIGHT_DEG of upright, back on the desk: within REST_M of their start height.
"""

import numpy as np

from scripts.datagen.plugins import g1d_pour_beans_measuring_cups as base
from scripts.g1d import scripted_demo as demo
from scripts.g1d.collect_office_beans import counts
from scripts.objects.round_cup_shape import capacity, fill_ok

TASK = "g1d_pour_beans_half_cup"
ACCEPTANCE = "half_cup_v1"
STEP_LIM = 1500  # actions: picking, pouring about half, putting both cups back
TAIL, HOLD = base.TAIL, base.HOLD
LABELS, COUNT_KEYS = base.LABELS, base.COUNT_KEYS
FILL = (0.30, 0.75)  # of the target's capacity: about half a cup
SPILL = 0.05  # of what left the tall cup
UPRIGHT_DEG = 15
REST_M = 0.01


def tilt_deg(pose):
    return float(np.degrees(np.arccos(np.clip(demo.from_wxyz(pose[3:]).as_matrix()[2, 2], -1, 1))))


def accepted(result, cup, poses, start_z, grain=None):
    return bool(
        fill_ok(result["target"], result["source"], result["total"] - result["overlap"], cup, FILL, SPILL, grain)
        and all(tilt_deg(p) <= UPRIGHT_DEG for p in poses)
        and np.all(np.abs(np.asarray(poses)[:, 2] - start_z) <= REST_M)
    )


class Episode(base.Episode):
    def __init__(self, env):
        super().__init__(env)
        if self.target is None:
            self.reject = "the target must be a round cup (plastic_cup 00008 / 00009)"
        else:
            self.capacity = capacity(self.target, self.grain)
        self.start_z = self.poses()[:, 2]

    @property
    def grain(self):
        """The beans' grain when they are rigid (GranularObject), else None: the PBD beans."""
        return getattr(self.fluid, "grain", None)

    def actions(self):
        return demo.half_cup_actions(self.env)

    def step(self):
        """After each action: record beans and cups; True while about half a cup is in the target, little is
        spilled, and both cups stand upright back on the desk."""
        points, poses = self.particles(), self.poses()
        self.final_counts = counts(points, base.pose_map(poses), self.target)
        self.history["particles"].append(points.astype(np.float32))
        self.history["cup_poses"].append(poses.astype(np.float32))
        self.history["counts"].append([self.final_counts[k] for k in COUNT_KEYS])
        return accepted(self.final_counts, self.target, poses, self.start_z, self.grain)

    def summary(self):
        cap = getattr(self, "capacity", None)
        return dict(
            super().summary(),
            capacity_beans=round(cap) if cap else None,
            fill=round(self.final_counts["target"] / cap, 3) if cap else None,
            fill_limits=FILL,
            spill_limit=SPILL,
            upright_limit_degrees=UPRIGHT_DEG,
            start_z=self.start_z.round(4).tolist(),
            **({"grain": self.grain} if self.grain else {}),
        )


def audit(privileged, quality):
    """Recompute the acceptance in each of the last HOLD frames from the recorded beans and cups."""
    total, target = quality["initial_counts"]["total"], quality["target_cup"]
    particles, poses = privileged["particles"], privileged["cup_poses"]
    assert particles.shape == (quality["frames"], total, 3) and poses.shape == (quality["frames"], 2, 7)
    start = counts(privileged["initial_particles"].astype(float), base.pose_map(poses[0]), target)
    assert start["source"] == total and start["outside"] == 0, start
    grain = quality.get("grain")
    cap, start_z = capacity(target, grain), np.asarray(quality["start_z"])
    fills = []
    for points, frame in zip(particles[-HOLD:], poses[-HOLD:]):
        result = counts(points.astype(float), base.pose_map(frame), target)
        assert accepted(result, target, frame, start_z, grain), result
        fills.append(result["target"] / cap)
    return dict(beans=total, capacity_beans=round(cap), fill=round(float(np.mean(fills)), 3))
