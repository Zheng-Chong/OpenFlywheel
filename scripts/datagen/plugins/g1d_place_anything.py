"""scripts/datagen plugin for g1d_place_anything (把 X 放到 Y 上); g1d_put_anything_in (把 X 放进 Y 里) is the same with
a container (its plugin sets KIND). The scripted expert is scripted_demo.place_anything_actions: g1d_pick_anything's
grasp (expert v3), carried over the receptacle, set down on its face or let go just above a container's rim, the hand
opened and backed off upward; through its Limiter.

Acceptance place_anything_v1, held for HOLD frames in a row (so it holds HOLD frames after the release):
    X on Y         scripts/objects/place_anything.placed: a support: X's centre over its face, X's bottom FACE_BELOW
                   .. FACE_ABOVE about it and X within X_TILT of how it lay; a container: X's centre over its opening,
                   below its rim and above its floor
    let go         both grippers at least OPEN open (func_parser.is_all_gripper_open's measure)
    settled        X moving under SETTLED m/s
    Y kept         Y within Y_STAY of its start horizontally and within Y_TILT of upright
    others kept    every distractor's centre within STAY of its start horizontally and STAY_Z vertically
The sidecar keeps every object's asset, X's and Y's names in both languages with the ones the instruction could use,
and Y's geometry and license, so a dataset can be relabelled or split by license.
"""

import numpy as np

from scripts.datagen.plugins.g1d_build_block_pattern import gripper_openness
from scripts.g1d import scripted_demo as demo
from scripts.objects import place_anything as pa

TASK = "g1d_place_anything"
KIND = "support"
ACCEPTANCE = "place_anything_v1"
STEP_LIM = 900  # actions: the task's
TAIL = 30  # actions the last command is held after the script ends
HOLD = 30  # frames (1 s) the acceptance must hold
OPEN, SETTLED, FPS = 0.6, 0.02, 30  # gripper opening, m/s, frames per second
STAY, STAY_Z = 0.03, 0.02  # m


def plain(value):
    """A layout value (OmegaConf containers once loaded) as JSON-ready dicts, lists and numbers."""
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, str):
        return value
    if hasattr(value, "__iter__"):
        return [plain(v) for v in value]
    return value


def failing(poses, labels, start, receptacle, x_size, opening, prev=None, kind=KIND):
    """The first acceptance condition poses (n, 7: xyz, wxyz) in `labels` order miss, against the start poses (n, 7),
    with the receptacle's layout geometry (`receptacle`: kind, region, floor, rim, ...; scale 1), X's extents, the
    grippers' `opening` and the poses a frame before (`prev`, for X's speed): "holding", "not_placed",
    "receptacle_moved", "moving", "distractor_moved", or None when it holds."""
    start = np.asarray(start, dtype=float)
    t, r = labels.index("target"), labels.index(pa.LABEL)
    x, y = poses[t], poses[r]
    if min(opening) < OPEN:
        return "holding"
    if not pa.placed(kind, receptacle, 1.0, x[:3], x[3:], x_size, y[:3], y[3:], start[t, 3:]):
        return "not_placed"
    if not pa.kept(y[:3], y[3:], start[r, :3]):
        return "receptacle_moved"
    if prev is not None and np.linalg.norm(x[:3] - prev[t, :3]) * FPS >= SETTLED:
        return "moving"
    for i, label in enumerate(labels):
        if label in ("target", pa.LABEL):
            continue
        move = poses[i, :3] - start[i, :3]
        if np.linalg.norm(move[:2]) >= STAY or abs(move[2]) >= STAY_Z:
            return "distractor_moved"
    return None


def accepted(*args, **kwargs):
    """Does the acceptance hold (failing's arguments)?"""
    return failing(*args, **kwargs) is None


class Episode:
    task, kind = TASK, KIND

    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        entries = env.entries(0)
        self.labels = sorted(entries)
        rec = entries[pa.LABEL]
        names = {
            label: {lang: self.lm.get_label_descriptions(0, label, lang=lang) for lang in ("zh", "en")}
            for label in ("target", pa.LABEL)
        }
        self.meta = dict(
            labels=self.labels,
            kind=self.kind,
            objects={k: [entries[k]["category"], int(entries[k]["category_idx"])] for k in self.labels},
            target_names=names["target"],
            receptacle_names=names[pa.LABEL],
            usable_names={lang: env.label_names(0, "target", lang) for lang in ("zh", "en")},
            usable_receptacle_names={lang: env.label_names(0, pa.LABEL, lang) for lang in ("zh", "en")},
            receptacle=plain(rec["receptacle"]),
            target_size=[float(v) for v in entries["target"]["physics"]["size"]],
            start=self.poses().round(5).tolist(),
        )
        geom = rec["receptacle"]
        self.reject = None if geom.get("kind") == self.kind else f"receptacle kind {geom.get('kind')}, not {self.kind}"
        self.last, self.prev, self.fail = None, None, None
        self.history = dict(object_poses=[], gripper_openness=[])

    def poses(self):
        return np.array(
            [np.concatenate([demo.as_np(v) for v in self.lm.get_instance_pose(0, label=k)]) for k in self.labels]
        )

    def actions(self):
        limiter = demo.Limiter(self.env)
        return (limiter(action) for action in demo.DEMOS[self.task](self.env))

    def step(self):
        """After each action: record every object and the grippers; True while the acceptance holds."""
        poses = self.poses()
        opening = np.asarray(gripper_openness(self.env), dtype=np.float32)
        self.history["object_poses"].append(poses.astype(np.float32))
        self.history["gripper_openness"].append(opening)
        m = self.meta
        self.fail = failing(
            poses, self.labels, m["start"], m["receptacle"], m["target_size"], opening, self.prev, self.kind
        )
        self.last = self.fail is None
        self.prev = poses
        return self.last

    def privileged(self):
        """Per frame after its action: every object's pose (T, n, 7: xyz, wxyz) in meta label order, gripper openness
        (T, 2)."""
        return {k: np.asarray(v) for k, v in self.history.items()}

    def summary(self):
        return dict(self.meta, final=bool(self.last), final_fail=self.fail)


def audit(privileged, quality):
    """Recompute the acceptance from the recorded poses in each of the last HOLD frames, and that the instruction
    names X and Y by usable names. Raises AssertionError on a violation."""
    poses, labels, opening = privileged["object_poses"], quality["labels"], privileged["gripper_openness"]
    n = quality["frames"]
    assert poses.shape == (n, len(labels), 7), poses.shape
    assert len(opening) == n
    kind = quality["kind"]
    for frame in range(n - HOLD, n):
        ok = accepted(
            poses[frame],
            labels,
            quality["start"],
            quality["receptacle"],
            quality["target_size"],
            opening[frame],
            poses[frame - 1],
            kind,
        )
        assert ok, frame
    for key in ("usable_names", "usable_receptacle_names"):
        usable = [x for names in quality[key].values() for x in names]
        assert any(x in quality["instruction"] for x in usable), (key, quality["instruction"])
    t = labels.index("target")
    moved = float(np.linalg.norm(poses[-1, t, :2] - np.asarray(quality["start"])[t, :2]))
    return dict(task=quality["task"], frames_checked=HOLD, carried_m=round(moved, 4), objects=len(labels), kind=kind)


__all__ = ["ACCEPTANCE", "HOLD", "KIND", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
