"""The real G1D-PourBeansEps380 episodes that the 倒半杯 task (g1d_pour_beans_half_cup) is a twin of.

Writes real2sim/tasks/g1d_pour_beans_half_cup/real_episodes.json: for every usable pour-half-a-cup episode at the
reference stand, its instruction and target cup, where the real right hand closes on the tall cup and the left on
the target (FK of the recorded joints from the fitted base, gripper pad points), and its first-frame arm joints.
The L0 benchmark layouts and the collection rounds' layouts (prepare_pour_half_layouts.py) are made from it.

Inputs (DSW: /mnt/data/datasets/G1D-自采数据/G1D-PourBeansEps380):
    states.npz   per frame: ep, fr, s = observation.state (23: arms 14, then grippers at 17 (left), 18 (right));
                 `python scripts/g1d/eps380_episodes.py extract <dataset> states.npz` (needs pyarrow)
    real2sim/tasks/g1d_pour_beans_half_cup/stand.json
                 per episode, the table's two far corners (px) and its front edge row (px) in the first head frame
    tasks_zh_backup.jsonl  the instructions (meta/)

    python scripts/g1d/eps380_episodes.py rows states.npz tasks_zh_backup.jsonl
    python scripts/g1d/eps380_episodes.py tilt states.npz      # real_pour.json: how far the real tall cup tilts
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "g1d")]
TASK_DIR = ROOT / "real2sim" / "tasks" / "g1d_pour_beans_half_cup"

GRAY = "把方口杯里的红豆，倒进灰色的粗口杯里，倒半杯"
BLUE = "把方口杯里的红豆，倒进蓝色杯子里，倒半杯"
TARGETS = {GRAY: "Rigid/plastic_cup/00009", BLUE: "Rigid/plastic_cup/00008"}
# The reference stand (env_cfg/robot/g1d_lab_eps380.yml): fitted to the head frames' table outline, 1.3 px rms
STAND = dict(dx=-0.0575, dy=-0.201, yaw_deg=-0.16)
REF_CORNERS, REF_EDGE, CORNER_TOL, EDGE_TOL = [[140, 118], [550, 119]], 308, 15, 12  # px
PAD_REACH = 0.12  # dex1 base link to the pads' centre (m)
CLOSE = 0.3  # a gripper closes when its state drops this far below the first frame
MIN_APART = 0.18  # m between the cups; closer means the hands met over one cup (a misread close)
ARMS = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]


def extract(dataset, out):
    import pyarrow.parquet as pq

    tables = [
        pq.read_table(p, columns=["episode_index", "frame_index", "observation.state"])
        for p in sorted(Path(dataset, "data").glob("*/*.parquet"))
    ]
    ep = np.concatenate([t["episode_index"].to_numpy() for t in tables])
    fr = np.concatenate([t["frame_index"].to_numpy() for t in tables])
    s = np.concatenate([np.stack(t["observation.state"].to_numpy(zero_copy_only=False)) for t in tables])
    np.savez_compressed(out, ep=ep, fr=fr, s=s.astype(np.float32))
    print(f"{len(np.unique(ep))} episodes, {len(ep)} frames -> {out}")


def rows(states, tasks_file):
    from real2sim.scene3_pour_pack.motion.kinematics import G1Kinematics
    from scripts.scenes.overlay import load_yaml, pose

    robot = load_yaml(ROOT / "env_cfg/robot/g1d_lab.yml")["robots"][0]
    base_pos = np.add(robot["default_root_pos"], [STAND["dx"], STAND["dy"], 0.0])
    r0 = robot["default_root_rot"]
    rot = Rotation.from_euler("z", np.deg2rad(STAND["yaw_deg"])) * Rotation.from_quat([*r0[1:], r0[0]])
    base = pose(base_pos, np.r_[rot.as_quat()[3], rot.as_quat()[:3]])
    kin = G1Kinematics(ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf")

    def grip(q14, side):
        links = kin.fk({"LZ_mt_Joint": 0.0, "LZ_it_Joint": 0.0, **dict(zip(ARMS, map(float, q14)))}, base=base)
        T = links[f"{side}_dex1_base_link"]
        return T[:3, 3] + T[:3, :3] @ np.array([PAD_REACH, 0.0, 0.0])

    d = np.load(states)
    ep, fr, s = d["ep"], d["fr"], d["s"]
    tasks = {
        json.loads(line)["episode_index"]: json.loads(line)["tasks"][0] for line in open(tasks_file, encoding="utf-8")
    }
    stand = json.loads((TASK_DIR / "stand.json").read_text())
    at = sorted(
        int(e)
        for e, v in stand.items()
        if v and np.abs(np.array(v[:2]) - REF_CORNERS).max() < CORNER_TOL and abs(v[2] - REF_EDGE) < EDGE_TOL
    )
    out, skipped = [], {}
    for e in at:
        if tasks.get(e) not in TARGETS:
            skipped[e] = "another instruction"
            continue
        S = s[ep == e][np.argsort(fr[ep == e])]
        closes = [np.flatnonzero(S[:, g] < S[0, g] - CLOSE) for g in (17, 18)]
        if not all(len(c) for c in closes):
            skipped[e] = "a gripper never closes"
            continue
        target, source = grip(S[closes[0][0], :14], "left"), grip(S[closes[1][0], :14], "right")
        if source[0] <= target[0] or np.linalg.norm(source[:2] - target[:2]) < MIN_APART:
            skipped[e] = "the closes do not read as two cups"
            continue
        out.append(
            dict(
                episode=e,
                instruction=tasks[e],
                target_object=TARGETS[tasks[e]],
                source=source.round(4).tolist(),
                target=target.round(4).tolist(),
                start=dict(zip(ARMS, S[0, :14].round(4).tolist())),
                frames=len(S),
            )
        )
    body = dict(
        dataset="G1D-PourBeansEps380",
        stand=STAND,
        note="source / target: real gripper pad points at the frame each hand closes (world, g1d_lab scene); start: "
        "first-frame arm joints (rad)",
        at_stand=len(at),
        skipped=skipped,
        episodes=out,
    )
    (TASK_DIR / "real_episodes.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    kinds = [r["target_object"] for r in out]
    grey, blue = kinds.count(TARGETS[GRAY]), kinds.count(TARGETS[BLUE])
    print(f"{len(at)} episodes at the stand, {len(out)} usable ({grey} grey, {blue} blue)")


# Where the real target cup first shows beans (left wrist camera, judged by eye at 60/70/80/85/90/95 deg and the peak
# of the tall cup's tilt; the grey cup is see-through, the blue one is not), 2026-11-01
ONSET_SEEN = {0: "88-90", 10: "90-93", 120: "~90", 134: "86-90", 148: "80-85", 158: "~88"}


def tilt(states):
    """real_pour.json: per episode, the tall cup's tilt (its axis off vertical: the right hand's rotation since it
    closed on the upright cup, FK) at its peak and how long it stays near it; the reference of bean_tilt_probe.py."""
    from real2sim.scene3_pour_pack.motion.kinematics import G1Kinematics

    kin = G1Kinematics(ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf")
    d = np.load(states)
    ep, fr, s = d["ep"], d["fr"], d["s"]
    out = {}
    for r in json.loads((TASK_DIR / "real_episodes.json").read_text(encoding="utf-8"))["episodes"]:
        S = s[ep == r["episode"]][np.argsort(fr[ep == r["episode"]])]
        close = int(np.flatnonzero(S[:, 18] < S[0, 18] - CLOSE)[0])
        rot = [
            Rotation.from_matrix(
                kin.fk({"LZ_mt_Joint": 0.0, "LZ_it_Joint": 0.0, **dict(zip(ARMS, map(float, q)))})[
                    "right_dex1_base_link"
                ][:3, :3]
            )
            for q in S[close:, :14]
        ]
        deg = np.degrees([np.arccos(np.clip((R * rot[0].inv()).apply([0, 0, 1])[2], -1, 1)) for R in rot])
        peak = int(np.argmax(deg))
        out[r["episode"]] = dict(
            target_object=r["target_object"],
            peak_deg=round(float(deg[peak]), 1),
            s_close_to_peak=round(peak / 30, 2),
            s_over_80_deg=round(float((deg > 80).sum()) / 30, 2),
            onset_seen_deg=ONSET_SEEN.get(r["episode"]),
        )
    peaks = [v["peak_deg"] for v in out.values()]
    body = dict(
        note="tall-cup tilt from FK of the recorded joints; onset_seen_deg: the tilt at which beans first reach the "
        "target cup in the left wrist video (by eye, grey-cup episodes)",
        peak_deg_median=round(float(np.median(peaks)), 1),
        peak_deg_p10_p90=np.percentile(peaks, [10, 90]).round(1).tolist(),
        episodes=out,
    )
    (TASK_DIR / "real_pour.json").write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(out)} episodes: peak tilt median {body['peak_deg_median']}, p10/p90 {body['peak_deg_p10_p90']}")


if __name__ == "__main__":
    {"extract": extract, "rows": rows, "tilt": tilt}[sys.argv[1]](*sys.argv[2:])
