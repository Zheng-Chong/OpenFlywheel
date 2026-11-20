"""Layouts of g1d_pour_beans_half_cup (倒半杯) from the real G1D-PourBeansEps380 episodes it is a twin of
(real2sim/tasks/g1d_pour_beans_half_cup/real_episodes.json, made by eps380_episodes.py).

    l0     one layout per real episode: both cups where the real hands closed on them, the episode's own target cup,
           first-frame arm joints (Robot.init_joint_pos) and instruction. The L0 twin set of the test benchmark.
    train  COUNT layouts for a collection round: each from a random real episode's cups and arm start, the cups moved
           by up to JITTER (m) and the target cup drawn from both. The round's spatial L1; marked split=train.

The desk is at 0.83 m (layout_template.json, the real stand's desk; env config g1d_lab_beans_eps380, robot
g1d_lab_eps380). Layouts go to <assets>/Eval_Layout/RoboDojo/g1d_lab/<seed>/g1d_pour_beans_half_cup_<i>.json.

    python scripts/g1d/prepare_pour_half_layouts.py l0 --assets <run dir>/Assets --seed 950
    python scripts/g1d/prepare_pour_half_layouts.py train --assets <run dir>/Assets --seed 600 --count 240

--rigid: layouts of the variant g1d_pour_beans_half_cup_rigid instead, the beans one rigid body each (GranularObject,
physics.type "granular") from a settled pack (rigid_bean_probe.py --pack) at <assets>/BEAN_PACK: red beans in l0 (as
the real ones), in train a random kind of BEANS and the pack turned about the cup's axis by a multiple of 90 deg.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d.build_asset import RED_BEAN_PHYSICS  # noqa: E402
from scripts.objects import measuring_cup_shape  # noqa: E402

TASK = "g1d_pour_beans_half_cup"
DIR = ROOT / "real2sim" / "tasks" / TASK
TOP = 0.83  # desk height of the layouts (m)
# target cups: platform object -> layout physics and cavity (scripts/objects/round_cup_shape.py)
CUPS = {
    "Rigid/plastic_cup/00009": dict(  # grey straight tumbler, from the real head frames
        idx=9, size=[0.066, 0.066, 0.089], mass=0.06, cavity=dict(height=0.089, radius=0.031, bottom=0.004)
    ),
    "Rigid/plastic_cup/00008": dict(  # blue tapered cup, measured 68.4 / 50.2 / 101.3 mm
        idx=8,
        size=[0.0684, 0.0684, 0.1013],
        mass=0.02,
        cavity=dict(height=0.1013, radius=0.0327, radius_bottom=0.0236, bottom=0.003),
    ),
}
JITTER = 0.03  # m, each cup, train layouts
# The tall cup's beans, filled as the real ones (131 mm in a 170 mm cup, ep0 head frame): ~3100 of them, the
# fluid mesh's bounding box (width 0.92) packed hexagonally to this height scale, settling to 130 mm
# (bean_tilt_probe.py); a cubic fill of the mesh itself holds half that. Physics: build_asset.RED_BEAN_PHYSICS.
BEANS_SZ, BEANS_HALF, BEANS_BOTTOM = 2.15, 0.035, 0.006  # height scale; mesh half height at scale 1; m off the desk
BEANS_SPAWN = {"region": "box", "packing": "hex"}
MIN_APART = 0.18  # m between the cups, as the real episodes
# --rigid: bean kinds, all ESTIMATES (not measured): shape as per-axis scale of the red bean (make_red_bean.py,
# 7 x 5 x 4.7 mm, 0.1 g; mass by volume), linear RGB and roughness of the coat. count: the lowest of the pack's beans
# that heap to the real 131 mm in the tall cup (95th percentile of the centres; packs of rigid_bean_probe.py --pack,
# settled at packing 0.55 red, 0.51 mung, 0.59 black).
RIGID_TASK = "g1d_pour_beans_half_cup_rigid"
BEAN_PACK = "Object/Granular/{kind}_bean/pack.npz"
BEANS = {
    "red": dict(axes=(1.0, 1.0, 1.0), count=2930, color=(0.075, 0.006, 0.008), roughness=0.22,
                name={"zh": "红豆", "en": "red beans"}),
    "mung": dict(axes=(0.78, 0.84, 0.84), count=5000, color=(0.115, 0.188, 0.021), roughness=0.45,
                 name={"zh": "绿豆", "en": "mung beans"}),
    "black": dict(axes=(1.45, 1.4, 1.15), count=1350, color=(0.012, 0.012, 0.015), roughness=0.18,
                  name={"zh": "黑豆", "en": "black beans"}),
}  # fmt: skip
# iterations: granular.ITERATIONS (with 4 the grasp pushed the full cup out and tilted it)
GRAIN_PHYSICS = dict(
    type="granular", friction=0.3, restitution=0.1, iterations=32, contact_offset=0.001, rest_offset=0.00035
)


def rigid_beans(beans, src, kind, yaw):
    """The Fluid entry as rigid beans of `kind`: the pack in the tall cup's frame (its outside bottom centre, the
    cup upright, measuring_cup_shape), turned `yaw` deg about the cup's axis."""
    bean = BEANS[kind]
    shift = measuring_cup_shape.origin_shift(measuring_cup_shape.MEASURING[1][0])
    half = np.deg2rad(yaw) / 2
    beans.update(
        default_pos=[round(float(v), 5) for v in np.add(src["default_pos"], shift)],
        default_ori=[round(float(np.cos(half)), 6), 0.0, 0.0, round(float(np.sin(half)), 6)],
        scale=[1.0, 1.0, 1.0],
        bean=dict(kind=kind, **bean["name"]),
        physics=dict(GRAIN_PHYSICS, pack=f"$Robodojo_ASSETS/{BEAN_PACK.format(kind=kind)}", count=bean["count"]),
        visual=dict(visible=True, color=list(bean["color"]), roughness=bean["roughness"]),
    )


def layout(template, source, target, cup, start, instruction=None, kind=None, yaw=0):
    lay = deepcopy(template)
    src = lay["Rigid"]["measuring_cup"][0]
    src["default_pos"][:2] = [round(float(v), 4) for v in source[:2]]
    src["default_ori"] = [1.0, 0.0, 0.0, 0.0]
    c = CUPS[cup]
    tgt = lay["Rigid"]["plastic_cup"][0]
    tgt.update(category_idx=c["idx"], cavity=dict(c["cavity"]), default_ori=[1.0, 0.0, 0.0, 0.0])
    tgt["default_pos"] = [round(float(target[0]), 4), round(float(target[1]), 4), round(TOP + c["size"][2] / 2, 4)]
    tgt["physics"] = dict(tgt["physics"], mass=c["mass"], size=list(c["size"]))
    beans = lay["Fluid"]["wuliangye"][0]
    beans["default_pos"] = [*src["default_pos"][:2], round(TOP + BEANS_BOTTOM + BEANS_HALF * BEANS_SZ, 4)]
    width = beans["visual"]["scale"][0]
    beans["visual"] = dict(beans["visual"], scale=[width, width, BEANS_SZ])
    beans["scale"] = [1 / width, 1 / width, 1 / BEANS_SZ]
    beans["physics"] = dict(beans["physics"], **deepcopy(RED_BEAN_PHYSICS), spawn=dict(BEANS_SPAWN))
    if kind:
        rigid_beans(beans, src, kind, yaw)
    lay["Robot"] = {"init_joint_pos": start}
    if instruction:
        lay["Instruction"] = instruction
    return lay


def write(folder, layouts, task, split=None):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{task}_*.json"):
        old.unlink()
    for i, lay in enumerate(layouts):
        (folder / f"{task}_{i}.json").write_text(json.dumps(lay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if split:
        (folder / "split.json").write_text(json.dumps({"split": split, "task": task}) + "\n")
    print(f"{len(layouts)} layouts -> {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["l0", "train"])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed folder of the layouts")
    parser.add_argument("--count", type=int, default=240, help="train: layouts")
    parser.add_argument("--rng", type=int, default=20261101, help="train: random seed")
    parser.add_argument("--rigid", action="store_true", help=f"layouts of {RIGID_TASK} (rigid beans)")
    args = parser.parse_args()
    task = RIGID_TASK if args.rigid else TASK
    template = json.loads((DIR / "layout_template.json").read_text())
    episodes = json.loads((DIR / "real_episodes.json").read_text(encoding="utf-8"))["episodes"]
    folder = args.assets / "Eval_Layout" / "RoboDojo" / "g1d_lab" / str(args.seed)
    if args.mode == "l0":
        kind = "red" if args.rigid else None
        rows = [
            layout(template, e["source"], e["target"], e["target_object"], e["start"], e["instruction"], kind)
            for e in episodes
        ]
        write(folder, rows, task)
        (folder / "episodes.json").write_text(json.dumps([e["episode"] for e in episodes]) + "\n")
        return
    rng = np.random.default_rng(args.rng)
    rows = []
    while len(rows) < args.count:
        e = episodes[rng.integers(len(episodes))]
        source = np.add(e["source"][:2], rng.uniform(-JITTER, JITTER, 2))
        target = np.add(e["target"][:2], rng.uniform(-JITTER, JITTER, 2))
        if source[0] <= target[0] or np.linalg.norm(source - target) < MIN_APART:
            continue
        cup = list(CUPS)[rng.integers(len(CUPS))]
        kind = list(BEANS)[rng.integers(len(BEANS))] if args.rigid else None
        rows.append(layout(template, source, target, cup, e["start"], kind=kind, yaw=90 * int(rng.integers(4))))
    write(folder, rows, task, split="train")


if __name__ == "__main__":
    main()
