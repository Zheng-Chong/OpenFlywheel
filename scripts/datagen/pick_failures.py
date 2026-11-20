"""Why the shared pick-anything grasp (scripted_demo.TopGripRobot.pick) misses: classify every recorded episode of
pick / place / put anything runs offline from its sidecar and HDF5 (privileged object poses and gripper openness, arm
joints, gripper commands), and group the misses by object family, shape, size, spot, grasp yaw and hand.

The close is found from the active hand's gripper command (the first frame it is fully shut, walked back over the
ramp down); the grasp point there comes from the G1-D URDF (real2sim/robot/g1d_v4) at the recorded joints, with the
model-to-sim offset taken as the median miss of the accepted episodes of each env config (`report` refits it).

Grasp outcome (`outcome`), first that applies:
    no_close      the gripper was never commanded shut (episode ended or crashed before)
    knocked       the target moved > KNOCK horizontally or tipped > TIP before the jaw began to close (the approach
                  pushed it)
    empty_reach   the jaw shut on nothing (openness < EMPTY) with the grasp point > REACH_MISS from the target: the
                  arm never got there
    empty_aim     shut on nothing, the grasp point within REACH_MISS, the target not moved by the close: aimed at the
                  wrong spot or height (e.g. a thin piece below the pads)
    squeezed      shut on nothing after the closing jaw moved the target > SQUEEZE (touched, slipped out)
    blocked       not lifted, the grasp point stopped > BLOCK above the planned grip height (the fingertips came down
                  on the object or a neighbour and never got round it)
    not_lifted    reached the grip height and closed on something, but the target never rose HELD (slid out of the
                  pads, or the jaw held something else)
    dropped       held after the close, rose >= HELD, then fell back by the end
    carry_slip    place / put: lifted, then sank > CARRY_DROP in the hand before the release (swung out on the way)
    lifted        pick: rose LIFTED and stayed; place / put: rose HELD within LIFT_WINDOW and was carried to the
                  release: the grasp worked (a reject is then a distractor or the place)

    python scripts/datagen/pick_failures.py scan --runs '/obs/Employee/filler/raw/pick-anything-r3-g1d/*' \
        --out /shared/tmp/$USER/pk4/r3.jsonl --sample 150 --sample-accepted 25
    python scripts/datagen/pick_failures.py report a.jsonl b.jsonl --out report.md [--json report.json]
    python scripts/datagen/pick_failures.py frames a.jsonl --out frames/ --per 3
"""

import argparse
from collections import Counter, defaultdict
import glob
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"
PAD_REACH = 0.12  # scripted_demo.PAD_REACH: dex1_base_link origin -> grip point on the pads
ROBOT_ROOT, ROBOT_YAW = np.array([0.0, -0.73, 0.1115]), np.pi / 2  # env_cfg/robot/g1d*.yml
LIFT_JOINTS = {"g1d": 0.0175, "g1d_shenzhen_lab": 0.035}  # LZ_mt_Joint / LZ_it_Joint init
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
EMPTY = 0.04  # measured openness below which the jaw holds nothing (pool objects are >= 15 mm: ~0.1)
KNOCK, TIP, SQUEEZE, REACH_MISS = 0.02, 25.0, 0.01, 0.025  # m, deg, m, m
HELD, LIFTED = 0.03, 0.08  # m the target centre rose: held (place's PLACE_LIFTED), lifted (pick's acceptance LIFT)
SHUT = 0.01  # gripper command counted as shut
# actions after the shut the lift is looked for; m the target moved against the grasp point while carried (from
# HAND_REF actions after the shut, when it is up in the jaw)
LIFT_WINDOW, CARRY_DROP, HAND_REF = 120, 0.03, 30
# m: scripted_demo.PICK_ANYTHING_DEPTH below the top, FINGER_TIPS + TABLE_CLEAR above the bottom; the descent counted
# as stopped short (the fingertips on the object or a neighbour) this far above the planned grip
PICK_DEPTH, PICK_FLOOR, BLOCK = 0.012, 0.029, 0.02
GRASP_FAIL = (
    "no_close", "knocked", "empty_reach", "empty_aim", "squeezed", "blocked", "not_lifted", "dropped", "carry_slip",
)  # fmt: skip
GRIP_YAW_FAR, GRIP_YAW_NEAR, NEAR_Y = 45.0, 15.0, -0.45  # scripts/objects/pick_anything.grip_yaw


class Chain:
    """URDF joints from AGV_link to <side>_dex1_base_link; fk(q7, lift) -> 4x4 in the env frame."""

    def __init__(self, side, urdf=URDF):
        joints = ET.parse(urdf).getroot().findall("joint")
        parent = {j.find("child").get("link"): j for j in joints}
        chain, link = [], f"{side}_dex1_base_link"
        while link in parent:
            chain.insert(0, parent[link])
            link = parent[link].find("parent").get("link")
        arm = [f"{side}_{n}_joint" for n in ARM_JOINTS]
        self.steps = []
        for j in chain:
            o, t = j.find("origin"), np.eye(4)
            if o is not None:
                t[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
                t[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            kind, name = j.get("type"), j.get("name")
            axis = None if kind == "fixed" else np.array([float(v) for v in j.find("axis").get("xyz").split()])
            src = arm.index(name) if name in arm else ("lift" if name.startswith("LZ_") else None)
            self.steps.append((t, axis, kind, src))
        self.root = np.eye(4)
        self.root[:3, :3] = Rotation.from_euler("z", ROBOT_YAW).as_matrix()
        self.root[:3, 3] = ROBOT_ROOT

    def fk(self, q, lift):
        t = self.root.copy()
        for origin, axis, kind, src in self.steps:
            t = t @ origin
            if axis is None:
                continue
            v = q[src] if isinstance(src, int) else (lift if src == "lift" else 0.0)
            m = np.eye(4)
            if kind == "prismatic":
                m[:3, 3] = axis * v
            else:
                m[:3, :3] = Rotation.from_rotvec(axis * v).as_matrix()
            t = t @ m
        return t

    def grasp_point(self, q, lift):
        t = self.fk(q, lift)
        return t[:3, 3] + t[:3, :3] @ [PAD_REACH, 0.0, 0.0]


_CHAINS = {}


def chain(side):
    if side not in _CHAINS:
        _CHAINS[side] = Chain(side)
    return _CHAINS[side]


def up_angle(q0, q1):
    """deg between the local z axes of two wxyz quaternions."""
    r0, r1 = (Rotation.from_quat([q[1], q[2], q[3], q[0]]) for q in (q0, q1))
    return float(np.degrees(np.arccos(np.clip(r0.apply([0, 0, 1]) @ r1.apply([0, 0, 1]), -1, 1))))


def shape_of(size):
    w, d, h = size
    narrow, wide = min(w, d), max(w, d)
    if h >= 2.0 * narrow and h >= 0.08:
        return "tall_thin"
    if h <= 0.03 or h <= 0.4 * narrow:
        return "flat"
    if wide <= 1.15 * narrow and 0.7 * wide <= h <= 1.4 * wide:
        return "round"
    return "box"


def size_of(size):
    m = max(size)
    return "S<5cm" if m < 0.05 else ("M5-10cm" if m < 0.10 else "L>10cm")


def region_of(x, y):
    ax = abs(x)
    xb = "|x|<4" if ax < 0.04 else ("|x|4-10" if ax < 0.10 else ("|x|10-19" if ax < 0.19 else "|x|>=19"))
    yb = "y<-50" if y < -0.50 else ("y-50..-47" if y < -0.47 else ("y-47..-36" if y < -0.36 else "y>=-36"))
    return f"{xb} {yb}"


def yaw_dev(quat, size, x, y):
    """deg the target's narrow horizontal axis is off the grip yaw of its spot (mod 180, -90..90)."""
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    axis = r.apply([1, 0, 0] if size[0] <= size[1] else [0, 1, 0])
    yaw = np.degrees(np.arctan2(axis[1], axis[0]))
    want = (GRIP_YAW_FAR if y > NEAR_Y else GRIP_YAW_NEAR) * (1 if x > 0 else -1)
    return float((yaw - want + 90) % 180 - 90)


def pool():
    """'<cat>/<idx:05d>' -> (family, unscaled size) of the pick anything pool (place / put draw X from it too)."""
    objects = json.loads((ROOT / "scripts/objects/pick_anything_pool.json").read_text())["objects"]
    return {f"{o['cat']}/{o['idx']:05d}": (o["family"], o["size"]) for o in objects}


_POOL = None


def side_info(sidecar):
    """What the sidecar alone tells: task, acceptance, target (family, pool size, shape), its start spot, and the hand
    pick() takes it with (Robot.arm_for: the one on its side of the midline)."""
    global _POOL
    if _POOL is None:
        _POOL = pool()
    side = Path(sidecar)
    meta = json.loads(side.read_text())
    rec = dict(
        sidecar=str(side),
        round=side.parents[3].name,  # <collection>/<unit>/raw/seed<N>/episode_<j>.json
        task=meta.get("task"),
        scene=meta.get("env_cfg"),
        expert=(meta.get("expert") or {}).get("version"),
        accepted=bool(meta.get("accepted")),
        frames=meta.get("frames"),
        aborted=(meta.get("aborted") or {}).get("reason"),
        reject=meta.get("reason"),
        layout_sha256=meta.get("layout_sha256"),
        grasp_attempts=meta.get("grasp_attempts"),
    )
    labels = meta.get("labels") or []
    if "target" not in labels or not meta.get("start"):
        return dict(rec, outcome="unknown", why="no target label")
    ti = labels.index("target")
    cat, idx = (meta.get("objects") or {}).get("target", [None, 0])
    rec["object"] = f"{cat}/{int(idx):05d}"
    rec["family"], size = _POOL.get(rec["object"], ("?", None))
    if size is None:
        return dict(rec, outcome="unknown", why="target not in the pool")
    # the layouts (drawn on the node, not kept) may scale it 0.85-1.15 (round 3+ object L2): the pool's size
    x, y = meta["start"][ti][:2]
    planned = (meta.get("planned_poses") or {}).get("target")
    return dict(
        rec,
        size=size,
        shape=shape_of(size),
        size_class=size_of(size),
        xy=[round(x, 4), round(y, 4)],
        region=region_of(x, y),
        arm="left" if x < 0 else "right",
        yaw_dev=round(yaw_dev(planned[1], size, x, y), 1) if planned else None,
    )


def episode(sidecar):
    """Features and grasp outcome of one recorded episode from its sidecar (`side_info`) and HDF5."""
    import h5py

    rec = side_info(sidecar)
    if rec.get("outcome") == "unknown":
        return rec
    meta = json.loads(Path(sidecar).read_text())
    ti, size = meta["labels"].index("target"), rec["size"]
    start = np.asarray(meta["start"][ti], dtype=float)
    h5 = Path(sidecar).with_suffix(".hdf5")
    if not h5.is_file():
        return dict(rec, outcome="unknown", why="no hdf5")
    try:
        with h5py.File(h5, "r") as f:
            poses = f["privileged/object_poses"][()]
            opening = f["privileged/gripper_openness"][()]
            cmd = {s: f[f"action/{s}_ee_joint_states"][()][:, 0] for s in ("left", "right")}
            q = {s: f[f"state/{s}_arm_joint_states"][()] for s in ("left", "right")}
    except Exception as e:  # noqa: BLE001 - a broken file is reported, not fatal
        return dict(rec, outcome="unknown", why=f"hdf5: {type(e).__name__}")
    labels = meta["labels"]
    lows = {s: float(cmd[s].min()) for s in cmd}
    arm = min(lows, key=lows.get)
    arm_i = 0 if arm == "left" else 1
    rec["arm"] = arm
    c = cmd[arm]
    shut = np.flatnonzero(c <= SHUT)
    if not len(shut):
        return dict(rec, outcome="no_close")
    t_shut = int(shut[0])
    t0 = t_shut
    while t0 > 0 and c[t0 - 1] > c[t0] + 1e-4:
        t0 -= 1
    n = len(poses)
    t_held = min(n - 1, t_shut + 12)
    lift = LIFT_JOINTS.get(rec["scene"], LIFT_JOINTS["g1d"])
    g0 = chain(arm).grasp_point(q[arm][t0], lift)
    p0 = poses[t0, ti, :3]
    rec.update(
        t_close=t0,
        t_shut=t_shut,
        open_cmd=round(float(c[max(t0 - 1, 0)]), 3),
        open_at_close=round(float(opening[t0, arm_i]), 3),
        grip_end=round(float(np.median(opening[min(n - 1, t_shut + 6) : t_held + 1, arm_i])), 3),
        approach_move=round(float(np.linalg.norm(p0[:2] - start[:2])), 4),
        approach_tip=round(up_angle(poses[0, ti, 3:], poses[t0, ti, 3:]), 1),
        close_move=round(float(np.linalg.norm(poses[t_held, ti, :2] - p0[:2])), 4),
        rise_max=round(float(poses[t_shut:, ti, 2].max() - start[2]), 4),
        rise_lift=round(float(poses[t_shut : t_shut + LIFT_WINDOW, ti, 2].max() - start[2]), 4),
        rise_end=round(float(poses[-1, ti, 2] - start[2]), 4),
        grasp_err=[round(float(v), 4) for v in g0 - p0],
        grasp_z_rel_top=round(float(g0[2] - (start[2] + size[2] / 2)), 4),
    )
    opened = np.flatnonzero(c[t_shut:] > 0.5)
    t_release = t_shut + int(opened[0]) if len(opened) else n
    hand = np.array([chain(arm).grasp_point(q[arm][t], lift) for t in range(t_shut, t_release)])
    if len(hand):
        rel = poses[t_shut:t_release, ti, :3] - hand
        ref = min(HAND_REF, len(rel) - 1)
        rec.update(
            hand_rise=round(float(hand[:LIFT_WINDOW, 2].max() - hand[0, 2]), 4),
            jaw_lift=round(float(opening[min(n - 1, t_shut + HAND_REF), arm_i]), 3),
            in_hand_move=round(float(np.linalg.norm(rel[ref:] - rel[ref], axis=1).max()), 4),
            frames_after_shut=n - t_shut,
        )
    others = [i for i in range(len(labels)) if i != ti]
    if others:
        moved = np.linalg.norm(poses[:, others, :2] - poses[:1, others, :2], axis=-1).max()
        rec["distractor_move"] = round(float(moved), 4)
    rec["outcome"] = outcome(rec)
    return rec


def outcome(r, offset=(0.0, 0.0)):
    """The grasp outcome of a scanned record; `offset`: the model-to-sim xy offset to take off grasp_err."""
    if "t_close" not in r:
        return r["outcome"]
    if r["approach_move"] > KNOCK or r["approach_tip"] > TIP:
        return "knocked"
    if r["grip_end"] < EMPTY:
        if r["close_move"] > SQUEEZE:
            return "squeezed"
        err = np.hypot(r["grasp_err"][0] - offset[0], r["grasp_err"][1] - offset[1])
        return "empty_reach" if err > REACH_MISS else "empty_aim"
    pick = r["task"] == "g1d_pick_anything"
    if (r["rise_end"] if pick else r["rise_lift"]) < (LIFTED if pick else HELD):
        if r["rise_max"] >= HELD:
            return "dropped"
        planned = max(-PICK_DEPTH, PICK_FLOOR - r["size"][2])  # TopGripRobot.grip_z, against the top
        return "blocked" if r["grasp_z_rel_top"] - planned > BLOCK else "not_lifted"
    if pick:
        return "lifted"
    return "carry_slip" if r.get("in_hand_move", 0.0) > CARRY_DROP else "lifted"


def scan(args):
    """Every sidecar into <out>.sidecars.jsonl; then the HDF5 of `--sample` rejected and `--sample-accepted` accepted
    episodes per collection (a fixed draw; all with --sample 0) into <out>. /obs is an rclone mount that reads 128 MB
    chunks: an HDF5 costs about its whole size, so sample and keep -j low."""
    files = sorted(f for pat in args.runs for d in glob.glob(pat) for f in glob.glob(f"{d}/raw/seed*/episode_*.json"))
    print(f"{len(files)} sidecars", flush=True)
    with Pool(args.jobs) as pool:
        sides = pool.map(side_info, files, chunksize=32)
    with open(args.out + ".sidecars.jsonl", "w") as out:
        out.writelines(json.dumps(r) + "\n" for r in sides)
    pick = files
    if args.sample:
        rng, pick = np.random.default_rng(0), []
        by = defaultdict(lambda: ([], []))
        for r in sides:
            if r.get("outcome") != "unknown":
                by[r["round"]][r["accepted"]].append(r["sidecar"])
        for rejected, accepted in by.values():
            for group, n in ((rejected, args.sample), (accepted, args.sample_accepted)):
                pick += [group[i] for i in sorted(rng.permutation(len(group))[:n])]
    print(f"{len(pick)} HDF5", flush=True)
    with Pool(args.h5_jobs) as pool, open(args.out, "w") as out:
        for i, rec in enumerate(pool.imap_unordered(episode, pick, chunksize=2)):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 200 == 0:
                print(i, flush=True)


def load(paths):
    return [json.loads(line) for p in paths for line in open(p)]


def offsets(recs):
    """Per env config: median grasp_err xy of accepted, lifted episodes (the URDF's offset from the sim)."""
    by = defaultdict(list)
    for r in recs:
        if r.get("accepted") and r.get("outcome") == "lifted":
            by[r["scene"]].append(r["grasp_err"][:2])
    return {k: tuple(np.median(v, axis=0).round(4)) for k, v in by.items() if len(v) >= 20}


def table(sides, sample, weight, key, min_n=30, top=None):
    """Rows (group, n, accepted share, estimated grasp-failure share, sampled rejects, their outcome shares) grouped by
    key(r): acceptance over every sidecar, outcomes over the sampled rejects, each weighted by its collection's
    rejects per sampled reject (`weight`)."""
    groups, rejects = defaultdict(list), defaultdict(list)
    for r in sides:
        if (k := key(r)) is not None:
            groups[k].append(r)
    for r in sample:
        if not r["accepted"] and (k := key(r)) is not None:
            rejects[k].append(r)
    rows = []
    for k, rs in groups.items():
        if len(rs) < min_n:
            continue
        acc = sum(r["accepted"] for r in rs) / len(rs)
        c, total = Counter(), 0.0
        for r in rejects[k]:
            c[r["outcome"]] += weight[r["round"]]
            total += weight[r["round"]]
        share = {o: v / total for o, v in c.items()} if total else {}
        fail = (1 - acc) * sum(share.get(o, 0.0) for o in GRASP_FAIL) if total else float("nan")
        rows.append((k, len(rs), acc, fail, len(rejects[k]), share))
    rows.sort(key=lambda row: -(row[3] if row[3] == row[3] else 1 - row[2]))
    return rows[:top] if top else rows


def yaw_bin(r):
    if "yaw_dev" not in r:
        return None
    lo = int(min(abs(r["yaw_dev"]), 89) // 15) * 15
    return f"{lo:02d}-{lo + 15}°"


def md(rows, title, head):
    cols = [o for o in GRASP_FAIL if o != "no_close"] + ["no_close", "lifted"]
    out = [f"### {title}", "", f"| {head} | n | 合格 | 估计抓空 | 抽样不合格 | " + " | ".join(cols) + " |"]
    out.append("|" + " --- |" * (5 + len(cols)))
    for k, n, acc, fail, ns, share in rows:
        cells = " | ".join(f"{share.get(o, 0):.0%}" if ns else "" for o in cols)
        out.append(f"| {k} | {n} | {acc:.0%} | {fail:.0%} | {ns} | {cells} |".replace("nan%", "—"))
    return out + [""]


def report(args):
    """Inputs are scan outputs; each one's <out>.sidecars.jsonl is read with it."""
    sample = [r for r in load(args.inputs) if r.get("outcome") != "unknown"]
    sides = [r for r in load([p + ".sidecars.jsonl" for p in args.inputs]) if r.get("outcome") != "unknown"]
    off = offsets(sample)
    for r in sample:
        if "t_close" in r:
            r["outcome"] = outcome(r, off.get(r["scene"], (0.0, 0.0)))
    rejected = Counter(r["round"] for r in sides if not r["accepted"])
    got = Counter(r["round"] for r in sample if not r["accepted"])
    weight = {k: rejected[k] / v for k, v in got.items()}
    lines = [
        f"# pick 抓取失败分析（{len(sides)} 回合，其中 {len(sample)} 条读了 HDF5）",
        "",
        "合格率按全部回合的 sidecar；「估计抓空」= 不合格率 × 抽样不合格里抓取失败的占比（按各集合的抽样比例加权）；"
        "右边各列是抽样不合格回合的抓取结果占比（lifted = 抓起来了，失败在别处：干扰物被碰、放置）。"
        f"URDF 到仿真的抓取点偏移（合格回合中位）：{off}",
        "",
    ]
    for task in sorted({r["task"] for r in sides}):
        rs, ss = [r for r in sides if r["task"] == task], [r for r in sample if r["task"] == task]
        lines += [f"## {task}", ""]
        acc_s = [r for r in ss if r["accepted"]]
        if acc_s:
            bad = Counter(r["outcome"] for r in acc_s if r["outcome"] != "lifted")
            lines += [f"标定：抽样合格 {len(acc_s)} 条里被判成抓取失败的 {sum(bad.values())} 条 {dict(bad)}", ""]
        args_t = (rs, ss, weight)
        lines += md(table(*args_t, lambda r: r["round"], 1), "按集合", "集合")
        lines += md(table(*args_t, lambda r: r.get("family"), 60, 30), "按物体大类（前 30）", "大类")
        lines += md(table(*args_t, lambda r: r.get("shape")), "按形状", "形状")
        lines += md(table(*args_t, lambda r: r.get("size_class")), "按尺寸（最长边）", "尺寸")
        lines += md(table(*args_t, lambda r: r.get("region"), 30), "按目标位置（env 坐标，cm）", "区域")
        lines += md(table(*args_t, lambda r: r.get("arm")), "按手", "手")
        lines += md(table(*args_t, lambda r: f"{r['arm']} {r['shape']}"), "按手 × 形状", "手 形状")
        lines += md(table(*args_t, yaw_bin), "按目标窄边与该处抓取朝向的夹角", "夹角")
        lines += md(table(*args_t, lambda r: r.get("scene")), "按场景", "场景")
        tall = [r for r in ss if r.get("shape") == "tall_thin" and "grasp_z_rel_top" in r]
        if tall:
            ratio = [(r["size"][2] - (-r["grasp_z_rel_top"])) / r["size"][2] for r in tall]
            lines += [f"高细物体 {len(tall)} 条：抓取点在物体高度的 {np.median(ratio):.0%}（中位，自底向上），"
                      f"抓空率 {sum(r['outcome'] in GRASP_FAIL for r in tall) / len(tall):.0%}", ""]  # fmt: skip
        reach = [np.hypot(*np.subtract(r["grasp_err"][:2], off.get(r["scene"], (0, 0))))
                 for r in ss if "grasp_err" in r]  # fmt: skip
        if reach:
            lines += [f"合爪时抓取点离目标（水平）：中位 {np.median(reach) * 1000:.0f} mm，"
                      f"90% {np.percentile(reach, 90) * 1000:.0f} mm", ""]  # fmt: skip
    Path(args.out).write_text("\n".join(lines) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(sample))
    print("\n".join(lines[:60]))


def missed(r):
    """The A/B's grasp failure: pick: the target not up LIFTED at the end; place / put: never up HELD (the
    retried expert's later tries count; a target carried and set down still counts as grasped)."""
    if r.get("outcome") == "no_close":
        return True
    return r["rise_end"] < LIFTED if r["task"] == "g1d_pick_anything" else r["rise_max"] < HELD


def coarse_region(r):
    x, y = abs(r["xy"][0]), r["xy"][1]
    near = "近身 y<-0.47" if y < -0.47 else ("远 y>=-0.36" if y >= -0.36 else "中 y-0.47..-0.36")
    return f"{'中线 |x|<0.10' if x < 0.10 else '两侧 |x|>=0.10'} · {near}"


def compare(args):
    """Paired A/B of two sets of scan outputs (`--a`, `--b`: scan.jsonl files of the same layouts): episodes paired by
    task, env config and layout sha256 (layouts that differ are not paired), grouped as the report."""

    def keyed(paths):
        return {(r["task"], r["scene"], r["layout_sha256"]): r for r in load(paths) if r.get("outcome") != "unknown"}

    a, b = keyed(args.a), keyed(args.b)
    pairs = [(a[k], b[k]) for k in sorted(set(a) & set(b), key=str)]
    lines = [f"# A/B：{args.name_a} vs {args.name_b}（同布局配对 {len(pairs)} 对；A {len(a)} 条，B {len(b)} 条）", ""]
    groups = [
        ("任务", lambda r: r["task"]),
        ("场景", lambda r: r["scene"]),
        ("手（A 的实际用手）", lambda r: r.get("arm")),
        ("形状", lambda r: r.get("shape")),
        ("尺寸", lambda r: r.get("size_class")),
        ("区域", coarse_region),
        ("物体大类", lambda r: r.get("family")),
    ]
    na, nb = args.name_a, args.name_b
    head = f"| {{}} | 对数 | 合格 {na} | 合格 {nb} | 抓空 {na} | 抓空 {nb} | 抓空降幅 | 合格翻转 +/- |"
    for task in sorted({p[0]["task"] for p in pairs}):
        tp = [p for p in pairs if p[0]["task"] == task]
        lines += [f"## {task}", ""]
        for title, key in groups:
            rows = defaultdict(list)
            for pa_, pb in tp:
                rows[key(pa_)].append((pa_, pb))
            lines += [f"### 按{title}", "", head.format(title), "|" + " --- |" * 8]
            for k, ps in sorted(rows.items(), key=lambda kv: -len(kv[1])):
                if k is None or len(ps) < args.min_n:
                    continue
                n = len(ps)
                acc = [sum(p[i]["accepted"] for p in ps) / n for i in (0, 1)]
                miss = [sum(missed(p[i]) for p in ps) / n for i in (0, 1)]
                drop = f"{1 - miss[1] / miss[0]:.0%}" if miss[0] else "—"
                gain = sum(not p[0]["accepted"] and p[1]["accepted"] for p in ps)
                loss = sum(p[0]["accepted"] and not p[1]["accepted"] for p in ps)
                cells = f"{acc[0]:.0%} | {acc[1]:.0%} | {miss[0]:.0%} | {miss[1]:.0%} | {drop} | +{gain}/-{loss}"
                lines.append(f"| {k} | {n} | {cells} |")
            lines.append("")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


def frames(args):
    """Example JPEG frames of `--per` episodes per grasp outcome: head and active wrist camera at the close's start,
    the shut + 12 and the last frame."""
    import h5py

    recs = [r for r in load(args.inputs) if "t_close" in r or r.get("outcome") == "no_close"]
    out = Path(args.out)
    rng = np.random.default_rng(0)
    index = []
    for oc in GRASP_FAIL + ("lifted",):
        pick = [r for r in recs if r["outcome"] == oc]
        for r in [pick[i] for i in rng.permutation(len(pick))[: args.per]]:
            h5 = Path(r["sidecar"]).with_suffix(".hdf5")
            tag = f"{oc}_{Path(r['sidecar']).parent.parent.parent.name}_{Path(r['sidecar']).stem}"
            with h5py.File(h5, "r") as f:
                n = len(f["vision/cam_head/colors"])
                ts = {"close": r.get("t_close", n // 2), "held": min(n - 1, r.get("t_shut", n // 2) + 12), "end": n - 1}
                for cam in ("cam_head", f"cam_{r.get('arm', 'left')}_wrist"):
                    for name, t in ts.items():
                        d = out / oc
                        d.mkdir(parents=True, exist_ok=True)
                        (d / f"{tag}_{cam}_{name}.jpg").write_bytes(bytes(f[f"vision/{cam}/colors"][t]))
            index.append(dict(r, tag=tag))
    (out / "index.json").write_text(json.dumps(index, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--runs", nargs="+", required=True, help="unit dirs (globs) holding raw/seed*/episode_*.json")
    s.add_argument("--out", required=True)
    s.add_argument("-j", "--jobs", type=int, default=16, help="sidecar readers")
    s.add_argument("--h5-jobs", type=int, default=8, help="HDF5 readers")
    s.add_argument("--sample", type=int, default=0, help="rejected episodes per collection (0: every episode)")
    s.add_argument("--sample-accepted", type=int, default=100, help="accepted episodes per collection")
    r = sub.add_parser("report")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--json")
    f = sub.add_parser("frames")
    f.add_argument("inputs", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--per", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--a", nargs="+", required=True, help="scan outputs of the baseline")
    c.add_argument("--b", nargs="+", required=True, help="scan outputs of the candidate, same layouts")
    c.add_argument("--name-a", default="v3")
    c.add_argument("--name-b", default="v4")
    c.add_argument("--min-n", type=int, default=10)
    c.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"scan": scan, "report": report, "frames": frames, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
