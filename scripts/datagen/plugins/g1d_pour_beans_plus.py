"""scripts/datagen plugin for g1d_pour_beans_plus (倒豆子 Plus): the scripted bimanual pour of every bean from cup A
into cup B, cups of any shape from real2sim/tasks/g1d_pour_beans_plus/cups.json, both cups put back on the desk.

Acceptance plus_v1 (real2sim/tasks/g1d_pour_beans_plus/definition.json), the cavities of the layout entries' "cavity"
profiles (scripts/objects/cup_profile.py):
- in each of the last HOLD frames at least IN_B (100% in plus_v1/plus_v2; 95% in plus_v3) of the beans are inside B's
  cavity, both cups stand within UPRIGHT_DEG of upright and within REST_M of their start height;
- in no frame do the two cups' bodies touch (cup_profile.touching);
- in no frame has a distractor (labels distractor_<i>) moved more than MOVED_M from where it started.
"""

import numpy as np

from scripts.g1d import scripted_demo as demo
from scripts.objects import cup_profile

TASK = "g1d_pour_beans_plus"
ACCEPTANCE = "plus_v1"
STEP_LIM = 1500  # actions: picking both cups, pouring every bean, putting both cups back
TAIL = 60  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
LABELS = ("source_cup", "target_cup")
COUNT_KEYS = ("total", "source", "target", "outside")
IN_B = 1.0  # plus_v1 / plus_v2 require every bean in B (benchmark comparability); plus_v3 sets 0.95
UPRIGHT_DEG = 15
REST_M = 0.01
MOVED_M = 0.02
EXITS = ("mouth", "wall", "floor")
MOUTH_M = 0.01  # a bean leaving a cup from this far under its spill level or higher left over the rim
OUT_M = 0.001  # past a cup's outer wall, under its underside or over its top by this, a bean has left it


def plain(value):
    """A layout entry's OmegaConf containers as JSON-ready dicts and lists."""
    if hasattr(value, "items"):
        return {k: plain(v) for k, v in value.items()}
    if value is None or isinstance(value, str | int | float):
        return value
    return [plain(v) for v in value]


def tilt_deg(pose):
    return float(np.degrees(np.arccos(np.clip(cup_profile.rotation(pose[3:])[2, 2], -1, 1))))


def count(points, poses, cups):
    """{total, source, target, outside} beans of `points` for cup poses (2, 7) [x, y, z, qw, qx, qy, qz]."""
    inside = [cup_profile.in_cavity(points, p[:3], p[3:], *cups[label]) for label, p in zip(LABELS, poses)]
    target = inside[1]
    source = inside[0] & ~target
    return dict(
        total=len(points),
        source=int(source.sum()),
        target=int(target.sum()),
        outside=int((~source & ~target).sum()),
    )


def homes(points, poses, cups, home):
    """Each bean's cup (0 A, 1 B, -1 neither) after a frame, from its cup `home` before: a bean in a cavity is that
    cup's; one out of the cavity stays its cup's until it is past the cup's outer wall, under its underside or over
    its top (a bean in a corner of a square cup, or squeezed into the wall, has not left it yet)."""
    out = np.where(home >= 0, home, -1)
    for k, (label, p) in enumerate(zip(LABELS, poses)):
        cup, scale = cups[label]
        local = cup_profile.local_points(points, p[:3], p[3:], scale)
        d = np.hypot(local[:, 0] - cup["axis"][0], local[:, 1] - cup["axis"][1])
        body = np.interp(local[:, 2], cup["body"]["z"], cup["body"]["r"])
        gone = (d > body + OUT_M) | (local[:, 2] < cup["bottom"] - OUT_M) | (local[:, 2] > cup["top"] + OUT_M)
        out[(home == k) & gone] = -1
    for k, (label, p) in enumerate(zip(LABELS, poses)):
        out[cup_profile.in_cavity(points, p[:3], p[3:], *cups[label])] = k
    return out


def exit_kind(before, after, pose_before, pose_after, cup, scale):
    """How beans left a cup between two frames (positions and the cup's pose before and after): 0 over the rim, from
    MOUTH_M under its spill level or higher; else 2 under its underside, 1 through its wall."""
    a = cup_profile.local_points(before, pose_before[:3], pose_before[3:], scale)
    b = cup_profile.local_points(after, pose_after[:3], pose_after[3:], scale)
    kind = np.where(b[:, 2] < cup["bottom"], 2, 1)
    kind[np.maximum(a[:, 2], b[:, 2]) >= cup["z"][-1] - MOUTH_M] = 0
    return kind


def touching(poses, cups):
    (a, b), (cup_a, cup_b) = poses, (cups[label] for label in LABELS)
    return cup_profile.touching((a[:3], a[3:]), cup_a[0], (b[:3], b[3:]), cup_b[0], cup_a[1], cup_b[1])


def moved(poses, start):
    """Largest horizontal shift (m) of the distractors (k, 7) from where they started."""
    return float(np.linalg.norm(poses[:, :2] - start[:, :2], axis=1).max()) if len(start) else 0.0


def settled(counts, poses, start_z, rest=REST_M, in_b=IN_B):
    return bool(
        counts["target"] >= in_b * counts["total"]
        and all(tilt_deg(p) <= UPRIGHT_DEG for p in poses)
        and np.all(np.abs(np.asarray(poses)[:, 2] - start_z) <= rest)
    )


class Episode:
    IN_B = IN_B

    def __init__(self, env):
        self.env = env
        lm = self.lm = env.scene_manager.layout_manager
        self.fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
        self.cups = {}
        for label in LABELS:
            cup = lm.get_scene_object(0, lm.get_instance_name(0, label))
            scale = float(np.asarray(getattr(cup, "scale", (1.0, 1.0, 1.0)), float).reshape(-1)[0])
            self.cups[label] = (plain(cup.instance_config["cavity"]), scale)
        self.distractors = sorted(lm.get_labels_by_prefix(0, "distractor_"))
        self.initial = self.particles()
        self.start = self.poses()
        self.start_z = self.start[:, 2]
        self.start_distractors = self.distractor_poses()
        self.initial_counts = self.final_counts = count(self.initial, self.start, self.cups)
        self.touched, self.moved = False, 0.0
        self.home, self.last = np.zeros(len(self.initial), int), (self.initial, self.start)
        self.escapes = {label: [0] * len(EXITS) for label in LABELS}
        self.leaks = []  # [frame, cup (0 A, 1 B), exit (EXITS index)] of each bean out through a wall or floor
        ok = self.initial_counts["source"] == len(self.initial)
        self.reject = None if ok else "beans not all inside the source cup at the start"
        self.history = dict(particles=[], cup_poses=[], counts=[], distractor_poses=[])

    def particles(self):
        points = np.asarray(self.fluid.get_particle_positions()[0], dtype=float)
        return points @ cup_profile.rotation(self.fluid.init_ori).T + np.asarray(self.fluid.init_pos, float)

    def poses(self):
        return np.array(
            [np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=label)]) for label in LABELS]
        )

    def distractor_poses(self):
        """(k, 7) poses of the distractors."""
        rows = [
            np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=d)]) for d in self.distractors
        ]
        return np.array(rows).reshape(-1, 7)

    def actions(self):
        return demo.plus_actions(self.env)

    def step(self):
        """After each action: record beans and cups; True while IN_B of the beans are in B and both cups stand
        upright back on the desk, the cups have never touched and no distractor has moved."""
        points, poses, others = self.particles(), self.poses(), self.distractor_poses()
        self.final_counts = count(points, poses, self.cups)
        self.touched |= touching(poses, self.cups)
        self.moved = max(self.moved, moved(others, self.start_distractors))
        self.history["particles"].append(points.astype(np.float32))
        self.history["cup_poses"].append(poses.astype(np.float32))
        self.history["counts"].append([self.final_counts[k] for k in COUNT_KEYS])
        self.history["distractor_poses"].append(others.astype(np.float32))
        self.track_exits(points, poses)
        return (
            settled(self.final_counts, poses, self.start_z, in_b=self.IN_B)
            and not self.touched
            and self.moved <= MOVED_M
        )

    def track_exits(self, points, poses):
        """Count the beans that left a cup since the last frame by how they left it (EXITS)."""
        home = homes(points, poses, self.cups, self.home)
        (before, poses_before), frame = self.last, len(self.history["particles"])
        for k, label in enumerate(LABELS):
            left = np.flatnonzero((self.home == k) & (home != k))
            kind = exit_kind(before[left], points[left], poses_before[k], poses[k], *self.cups[label])
            for e in range(len(EXITS)):
                self.escapes[label][e] += int((kind == e).sum())
            self.leaks += [[frame, k, int(e)] for e in kind[kind > 0]]
        self.home, self.last = home, (points, poses)

    def privileged(self):
        """Per frame after its action: bean positions (T, N, 3), cup poses (T, 2, 7), counts (T, 4) in COUNT_KEYS,
        distractor poses (T, k, 7); and the start of the beans and distractors."""
        return dict(
            initial_particles=self.initial.astype(np.float32),
            distractor_poses_start=self.start_distractors.astype(np.float32),
            **{k: np.asarray(v) for k, v in self.history.items()},
        )

    def summary(self):
        return dict(
            initial_counts=self.initial_counts,
            final_counts=self.final_counts,
            cups={label: {"cavity": cup, "scale": scale} for label, (cup, scale) in self.cups.items()},
            start_z=self.start_z.round(6).tolist(),
            touched=self.touched,
            escapes={label: dict(zip(EXITS, n)) for label, n in self.escapes.items()},
            leaks=self.leaks,
            distractors=self.distractors,
            distractor_moved_m=round(self.moved, 4),
            in_target_share=self.IN_B,
            upright_limit_degrees=UPRIGHT_DEG,
            rest_limit_m=REST_M,
            moved_limit_m=MOVED_M,
        )


def audit(privileged, quality, in_b=IN_B):
    """Recompute the acceptance from the recorded beans, cups and distractors: all beans in A at the start; `in_b` of
    the beans in B and both cups upright back on the desk in each of the last HOLD frames; in no frame the cups
    touching or a distractor moved. Raises AssertionError on a violation; returns metrics for the report."""
    total, frames = quality["initial_counts"]["total"], quality["frames"]
    particles, poses = privileged["particles"], privileged["cup_poses"].astype(float)
    assert particles.shape == (frames, total, 3) and poses.shape == (frames, 2, 7)
    cups = {label: (quality["cups"][label]["cavity"], quality["cups"][label]["scale"]) for label in LABELS}
    start = count(privileged["initial_particles"].astype(float), poses[0], cups)
    assert start["source"] == total, start
    # Sidecars before 2026-11-05 round start_z to 0.1 mm; the online check used the exact value.
    start_z, rest = np.asarray(quality["start_z"]), REST_M + 5e-5
    for points, frame in zip(particles[-HOLD:], poses[-HOLD:]):
        result = count(points.astype(float), frame, cups)
        assert settled(result, frame, start_z, rest, in_b=in_b), result
    assert not any(touching(frame, cups) for frame in poses), "the cups touched"
    others, first = privileged["distractor_poses"].astype(float), privileged["distractor_poses_start"].astype(float)
    shift = max((moved(frame, first) for frame in others), default=0.0)
    assert shift <= MOVED_M, f"a distractor moved {shift:.3f} m"
    tilt = max(tilt_deg(p) for p in poses[-1])
    landed = count(particles[-1].astype(float), poses[-1], cups)["target"] / total
    return dict(
        beans=total,
        in_target_share=round(landed, 4),
        max_final_tilt_degrees=round(tilt, 2),
        distractor_moved_m=round(shift, 4),
    )
