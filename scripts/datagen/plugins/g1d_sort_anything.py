"""scripts/datagen plugin for g1d_sort_anything (按种类分拣任意物体). The scripted expert is
scripts/g1d/sort_v43.py (place anything expert v4.3 item by item, docs/sort-anything.md): the items one by one in
sort_anything_demo's order, each by the hand on its receptacle's side, its first grasp chosen by its whole route to its
slot's set-down, carried and set down as place v4.2 (a dropped item grasped again); through scripted_demo's Limiter.
The v3 expert (pick v3, a miss grasped once more, place v3) is g1d_sort_anything_v3, pick v4.1's g1d_sort_anything_v41.

Acceptance sort_anything_v1, held for HOLD frames in a row (so it holds HOLD frames after the last release):
    let go          both grippers at least OPEN open
    nothing wrong   no item on / in a receptacle other than its class's
    all sorted      every item on / in its class's receptacle (scripts/objects/place_anything.placed with that
                    receptacle's geometry: a support's face with the item within X_TILT of how it lay, a container's
                    cavity)
    receptacles     every receptacle (the spare one too) within Y_STAY of its start horizontally, within Y_TILT of
                    upright
    settled         every item moving under SETTLED m/s
Sorting part of the items is not accepted; every frame records how many items sit in their receptacle and how many in
another (privileged "sorted", "wrong"), the sidecar the counts at the end ("sorted_final", "wrong_final") and at most
("sorted_max"), with each object's asset, class and names, and each receptacle's geometry and license; the grasp
tries (`grasp_attempts`), the routes checked per item (`place_route`) and how far the hand ended off over each spot
(`place_log`).
"""

import numpy as np

from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.datagen.plugins.g1d_place_anything import FPS, OPEN, SETTLED, plain
from scripts.g1d import scripted_demo as demo, sort_v43
from scripts.objects import place_anything as pa, sort_anything as sa

TASK = sa.TASK
ACCEPTANCE = "sort_anything_v1"
STEP_LIM = 3800  # actions: the task's 2400 and room for more grasp tries and a regrasp of two items
TAIL = 30  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold


def counts(poses, labels, items, receptacles, start, sizes):
    """(sorted, wrong): items on / in their own receptacle, and on / in another one. poses (n, 7: xyz, wxyz) and start
    in `labels` order; items [(item, its receptacle)]; receptacles {label: layout geometry}; sizes {item: extents}."""
    at = {label: i for i, label in enumerate(labels)}
    good = bad = 0
    for x, own in items:
        p = poses[at[x]]
        on = [
            y
            for y, geom in receptacles.items()
            if pa.placed(geom["kind"], geom, 1.0, p[:3], p[3:], sizes[x], poses[at[y]][:3], poses[at[y]][3:],
                         start[at[x]][3:])
        ]  # fmt: skip
        good += own in on
        bad += any(y != own for y in on)
    return good, bad


def failing(poses, labels, items, receptacles, start, sizes, opening, prev=None):
    """The first acceptance condition the poses miss (counts' arguments, the grippers' `opening`, the poses a frame
    before `prev` for the items' speed): "holding", "wrong_receptacle", "not_sorted", "receptacle_moved", "moving",
    or None when it holds."""
    start = np.asarray(start, dtype=float)
    at = {label: i for i, label in enumerate(labels)}
    if min(opening) < OPEN:
        return "holding"
    good, bad = counts(poses, labels, items, receptacles, start, sizes)
    if bad:
        return "wrong_receptacle"
    if good < len(items):
        return "not_sorted"
    if not all(pa.kept(poses[at[y]][:3], poses[at[y]][3:], start[at[y], :3]) for y in receptacles):
        return "receptacle_moved"
    if prev is not None and any(np.linalg.norm(poses[at[x], :3] - prev[at[x], :3]) * FPS >= SETTLED for x, _ in items):
        return "moving"
    return None


class Episode:
    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        entries = env.entries(0)
        self.labels = sorted(entries)
        items = env.items(0)
        ys = env.receptacles(0)
        layout = self.lm.saved_layouts[0] or {}
        self.meta = dict(
            labels=self.labels,
            items=[list(p) for p in items],
            classes={x: entries[x]["sort"]["class"] for x, _ in items},
            receptacle_classes={y: entries[y]["sort"]["class"] for y in ys},
            objects={k: [entries[k]["category"], int(entries[k]["category_idx"])] for k in self.labels},
            names={
                k: {lang: self.lm.get_label_descriptions(0, k, lang=lang) for lang in ("zh", "en")} for k in self.labels
            },
            receptacles={y: plain(entries[y]["receptacle"]) for y in ys},
            sizes={x: [float(v) for v in entries[x]["physics"]["size"]] for x, _ in items},
            sort=plain((layout.get("Generalization") or {}).get("sort") or {}),
            start=self.poses().round(5).tolist(),
        )
        kinds = {g.get("kind") for g in self.meta["receptacles"].values()}
        self.reject = None if kinds <= set(pa.KINDS) and items else f"items {items}, receptacle kinds {kinds}"
        self.last, self.prev, self.fail = None, None, None
        self.history = dict(object_poses=[], gripper_openness=[], sorted=[], wrong=[])

    def poses(self):
        return np.array(
            [np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=k)]) for k in self.labels]
        )

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in sort_v43.sort_anything_actions(self.env))

    def step(self):
        """After each action: record every object, the grippers and the counts; True while the acceptance holds."""
        poses = self.poses()
        opening = np.asarray(gripper_openness(self.env), dtype=np.float32)
        m = self.meta
        good, bad = counts(poses, self.labels, m["items"], m["receptacles"], m["start"], m["sizes"])
        self.history["object_poses"].append(poses.astype(np.float32))
        self.history["gripper_openness"].append(opening)
        self.history["sorted"].append(good)
        self.history["wrong"].append(bad)
        self.fail = failing(
            poses, self.labels, m["items"], m["receptacles"], m["start"], m["sizes"], opening, self.prev
        )
        self.last = self.fail is None
        self.prev = poses
        return self.last

    def privileged(self):
        """Per frame after its action: every object's pose (T, n, 7: xyz, wxyz) in meta label order, gripper openness
        (T, 2), items in their own receptacle and in another (T,)."""
        h = self.history
        return dict(
            object_poses=np.asarray(h["object_poses"]),
            gripper_openness=np.asarray(h["gripper_openness"]),
            sorted=np.asarray(h["sorted"], dtype=np.int8),
            wrong=np.asarray(h["wrong"], dtype=np.int8),
        )

    def summary(self):
        h = self.history
        return dict(
            self.meta,
            final=bool(self.last),
            final_fail=self.fail,
            n_items=len(self.meta["items"]),
            sorted_final=int(h["sorted"][-1]) if h["sorted"] else 0,
            sorted_max=int(max(h["sorted"], default=0)),
            wrong_final=int(h["wrong"][-1]) if h["wrong"] else 0,
            picks=list(getattr(self.env, "sort_picks", [])),
            grasp_attempts=list(getattr(self.env, "grasp_attempts", [])),
            place_route=list(getattr(self.env, "place_route", [])),
            place_log=list(getattr(self.env, "place_log", [])),
        )


def audit(privileged, quality):
    """Recompute the acceptance from the recorded poses in each of the last HOLD frames, and that the instruction
    names every class (its class name) and its receptacle (the name the layout gave it). Raises AssertionError."""
    poses, labels, opening = privileged["object_poses"], quality["labels"], privileged["gripper_openness"]
    n = quality["frames"]
    assert poses.shape == (n, len(labels), 7), poses.shape
    assert len(opening) == n
    args = (labels, quality["items"], quality["receptacles"], quality["start"], quality["sizes"])
    for frame in range(n - HOLD, n):
        assert failing(poses[frame], *args, opening[frame], poses[frame - 1]) is None, frame
    text = quality["instruction"]
    for c in set(quality["classes"].values()):
        assert any(name in text for names in sa.class_names(c).values() for name in names), (c, text)
    for y, name in (quality.get("sort") or {}).get("receptacle_names", {}).items():
        assert name in text, (y, name, text)
    moved = [
        float(np.linalg.norm(poses[-1, labels.index(x), :2] - np.asarray(quality["start"])[labels.index(x), :2]))
        for x, _ in quality["items"]
    ]
    return dict(
        task=quality["task"],
        frames_checked=HOLD,
        items=len(quality["items"]),
        receptacles=len(quality["receptacles"]),
        carried_m=round(sum(moved), 4),
    )


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
