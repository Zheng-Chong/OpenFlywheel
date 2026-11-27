"""Layouts of g1d_pick_anything round 3 (按名称拿起任意物体): every axis of the platform's five (spatial, scene,
object, camera, language; OpenFlywheel index.html AXES) at L2, the top of the training range (L3 is for held-out
benchmarks), on round 2's pool (scripts/objects/pick_anything_pool.json, 551 objects; no benchmark holds any out):

    spatial   target |x| 0.02-0.25 m, y -0.53 - -0.30 m (round 2's distractor area; round 2: |x| 0.04-0.19,
              y -0.47 - -0.36), turned within ±30 deg of the grip yaw (round 2: ±20); robot base ±3 cm sideways and
              to the desk, ±5 deg yaw, lift ±2 cm, desk ±3 cm (pour_beans_plus_layouts BASE at L2)
    scene     the office (g1d) or Shenzhen lab (g1d_shenzhen_lab) desk; 1-3 extra lights; floor, walls and tabletop
              each with SWAP chance a material of the scene profile's pool (pour_beans_plus_layouts visual L2); the
              3-5 pool distractors stand for its 2-4 training distractors
    object    the pool's instances (round 2's level) each scaled within SCALE where its names give no size and the
              jaw still takes it (pick_anything.POOL_RULES at that scale)
    camera    extrinsics head ±1 cm / ±2 deg, wrists ±0.5 cm / ±2 deg; image brightness, contrast, saturation ±25%
              and noise (pour_beans_plus_layouts camera L2)
    language  1-2 distractors of the target's family whenever it has another member (round 2: half the layouts), so
              the instruction the task draws (pick_anything.usable_names) names it by a detailed name

Objects stand in the desk's frame as round 2 places them; the robot base moves by moving the scene and every object
rigidly about the robot root the other way (pour_beans_plus_layouts.transform). Each layout records its draws in
"Generalization".

Unit <folder>-<k> of a config: layouts 50k .. 50k+49 of candidate folder <folder> (layout i targets the pool's fixed
shuffle at i mod 551, as round 2), written to <assets>/Eval_Layout/RoboDojo/<config>/<folder*100+50+k>/
g1d_pick_anything_<j>.json, the candidates filler-job collects into seed folder*100+k.

    python scripts/datagen/pick_anything_layouts.py --assets <run>/Assets --config g1d --folders 760 768 --units 11

A later round of the same recipe (round 4: real2sim/collections/g1d_pick_anything-r4) passes --round with its own
folders; the round name seeds the draws and is recorded in "Generalization". --commercial keeps the pool's
commercially usable objects (pick_anything.commercial: 295 of the 551, 37 families; round 6), whose fixed shuffle then
gives layout i's target. From round 7 on the pool also drops what pick_anything.EXCLUDED leaves out for the task (auto
parts: 293 commercially usable objects, 36 families); rounds 1-6 keep the pool they were drawn from.
"""

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen import pour_beans_plus_layouts as plus  # noqa: E402
from scripts.g1d import build_asset  # noqa: E402
from scripts.objects import object_names, pick_anything  # noqa: E402
from scripts.scenes import appearance  # noqa: E402

TASK = "g1d_pick_anything"
ROUND = "g1d_pick_anything-r3"
DRAWN_BEFORE_EXCLUDED = {f"{TASK}-r{n}" for n in range(1, 7)}  # no pick_anything.EXCLUDED (round 7 on: auto parts out)
DIMS = ("spatial", "scene", "object", "camera", "language")
LEVEL = 2
CONFIGS = ("g1d", "g1d_shenzhen_lab", "g1d_lab")
PER_UNIT = 50
SPEC = dict(pick_anything.SPEC, target_x=0.25, target_y=(-0.53, -0.30), mid_gap=0.02, grasp_yaw=30.0, same_family=1.0)
WHOLEBODY_DEFAULT = {
    "table_top_z": (0.64, 0.96),
    "init_lift": (0.0, 0.34),
    "init_waist_pitch_deg": (0.0, 12.0),
    "min_lift_delta": 0.04,
    "target_y": (-0.50, -0.22),
}
SCALE = (0.85, 1.15)
# a name that gives a size (large, short, 4 cm, 大号, 长款 ...) would be wrong once the object is scaled
SIZE_WORDS = re.compile(
    r"\d|centimet|millimet|inch|\b(large|small|big|little|tall|short|long|tiny|mini|medium|wide|narrow|thick|thin|huge"
    r"|giant)\b|大|小|高|矮|长|短|粗|细|宽|窄|厘米|毫米",
    re.I,
)


def sized(names=None):
    """'<cat>/<idx:05d>' of the Rigid objects whose names give a size."""
    names = names or object_names.load()
    return {
        oid.removeprefix("Rigid/")
        for oid, n in names.items()
        if oid.startswith("Rigid/") and any(SIZE_WORDS.search(x) for lang in ("en", "zh") for x in n.get(lang, []))
    }


def scale_range(obj):
    """(low, high) of SCALE at which the pool rules still hold for `obj`."""
    w, d, h = obj["size"]
    r = pick_anything.POOL_RULES
    narrow = min(w, d)
    low = max(SCALE[0], r["width"][0] / narrow, r["height"][0] / h)
    high = min(
        SCALE[1],
        r["width"][1] / narrow,
        r["height"][1] / h,
        r["long"] / max(w, d),
        (r["mass"] / obj["mass"]) ** (1 / 3),
    )
    return low, high


def scaler(fixed):
    def draw(rng, obj):
        if f"{obj['cat']}/{obj['idx']:05d}" in fixed:
            return 1.0
        low, high = scale_range(obj)
        return round(rng.uniform(low, high), 3) if low < high else 1.0

    return draw


def entry_maker(assets, top):
    def entry(cat, idx, label, xy, yaw, scale=1.0):
        e = build_asset.table_entry(assets, top, cat, idx, label, xy, yaw)
        size = e["physics"]["size"]
        e["scale"] = [scale] * 3
        e["default_pos"][2] = top + size[2] * scale / 2 + 1e-3
        e["physics"] = dict(e["physics"], size=[round(v * scale, 5) for v in size])
        if "mass" in e["physics"]:
            e["physics"]["mass"] = round(e["physics"]["mass"] * scale**3, 5)
        return e

    return entry


class Generator:
    """`levels` ({axis: level}, default every axis at LEVEL) lowers axes for a diagnostic: spatial, language below 2
    as round 2 (pick_anything.SPEC: its target area and yaw, a same-family distractor in half the layouts), object
    below 2 unscaled, scene and camera at pour_beans_plus_layouts' level."""

    task = TASK

    def __init__(self, assets, pool=None, names=None, levels=None, round_=ROUND, wholebody=None):
        self.assets, self.pool, self.round = Path(assets), pool or pick_anything.load_pool(), round_
        self.levels = dict(dict.fromkeys(DIMS, LEVEL), **(levels or {}))
        self.wholebody = (
            dict(WHOLEBODY_DEFAULT, **(wholebody if isinstance(wholebody, dict) else {})) if wholebody else None
        )
        lv = self.levels
        self.spec = dict(SPEC if lv["spatial"] >= 2 else pick_anything.SPEC)
        if self.wholebody and "target_y" in self.wholebody:
            self.spec["target_y"] = tuple(self.wholebody["target_y"])
        self.spec["same_family"] = SPEC["same_family"] if lv["language"] >= 2 else pick_anything.SAME_FAMILY
        self.spec["scale"] = scaler(sized(names)) if lv["object"] >= 2 else None
        self.family = {(o["cat"], o["idx"]): o["family"] for o in self.pool}
        self.scenes = {c: appearance.load(build_asset.SCENES_DIR / plus.SCENES[c]) for c in CONFIGS}

    def layout(self, config, folder, index):
        rng = random.Random(f"{self.round}/{config}/{folder}/{index}")
        lv = self.levels
        parts, _, library = self.scenes[config]
        mats = plus.draw_materials(rng, config, lv["scene"])
        base = plus.draw_base(rng, lv["spatial"])
        wb_info = None
        if self.wholebody:
            base = {**base, "lift": 0.0, "table": 0.0}
            z_lo, z_hi = self.wholebody["table_top_z"]
            if parts.get("lift"):
                r_lo, r_hi = parts["lift"]["range"]
                z_lo, z_hi = max(z_lo, r_lo), min(z_hi, r_hi)
            top = round(rng.uniform(z_lo, z_hi), 4)
        else:
            # the desk: its own height plus the change, less the lift column's (a taller robot sees a lower desk)
            top = parts["top_z"] + base["table"] - base["lift"]
        choice = {**mats, "table_top_z": round(top, 4) if abs(top - parts["top_z"]) > 1e-6 else None}
        room_over, top = appearance.overrides(parts, choice, library)
        _, room, room_pos, _ = build_asset.SCENES.get(config, build_asset.SCENES["g1d_school"])
        rigid = self.rigid(rng, index, top)
        out = {"Rigid": rigid, "Room": build_asset.room_entry(room, room_pos, top)}
        out["Room"].update({k: v for k, v in room_over.items() if v}, appearance_choice=choice)
        if self.wholebody:
            out["Room"]["table_height"] = round(top, 4)
            work_lift = float(min(max(top - 0.68, 0.0), 0.42))
            l_lo, l_hi = self.wholebody["init_lift"]
            min_dl = float(self.wholebody["min_lift_delta"])
            init_lift = rng.uniform(l_lo, l_hi)
            for _ in range(20):
                if abs(init_lift - work_lift) >= min_dl:
                    break
                init_lift = rng.uniform(l_lo, l_hi)
            p_lo, p_hi = self.wholebody["init_waist_pitch_deg"]
            init_pitch = math.radians(rng.uniform(p_lo, p_hi))
            out["Robot"] = {
                "init_joint_pos": {
                    "LZ_mt_Joint": round(init_lift / 2.0, 5),
                    "LZ_it_Joint": round(init_lift / 2.0, 5),
                    "Yaw_Joint": round(init_pitch, 5),
                    "torso_Joint": 0.0,
                }
            }
            wb_info = {
                "table_top_z": round(top, 4),
                "init_lift": round(init_lift, 4),
                "work_lift": round(work_lift, 4),
                "init_waist_pitch": round(init_pitch, 4),
            }
        lights, light_record = plus.draw_lights(rng, lv["scene"])
        camera, imaging = plus.draw_camera(rng, lv["camera"])
        for key, value in (("Light", lights), ("Camera", camera), ("Imaging", imaging)):
            if value:
                out[key] = value
        entries = {e["label"]: e for es in rigid.values() for e in es}
        plus.transform(out, base)
        out["Generalization"] = {
            "round": self.round,
            "split": "train",
            "levels": dict(lv),
            "scene": config,
            **self.record(entries),
            "base": {k: round(v, 4) for k, v in base.items()},
            "materials": mats,
            "lights": light_record,
        }
        if wb_info is not None:
            out["Generalization"]["wholebody"] = wb_info
        return out

    def record(self, entries):
        """The layout's objects in "Generalization" ({label: entry}, moved with the robot base; a subclass without a
        target records its own)."""
        family = {k: self.family.get((e["category"], e["category_idx"])) for k, e in entries.items()}
        target = entries["target"]
        return {
            "target": f"{target['category']}/{target['category_idx']:05d}",
            "target_xy": [round(v, 4) for v in target["default_pos"][:2]],
            "scale": {k: e["scale"][0] for k, e in sorted(entries.items())},
            "same_family": sum(f == family["target"] for k, f in family.items() if k != "target"),
            "objects": len(entries),
        }

    def rigid(self, rng, index, top):
        """The layout's Rigid section, its desk top at z `top` (a subclass draws other objects)."""
        return pick_anything.layout(rng, index, entry_maker(self.assets, top), self.pool, self.spec)

    def write_unit(self, config, folder, k, per=PER_UNIT):
        """The candidates of unit <folder>-<k> (the env config's name is its config_name): their folder."""
        dst = self.assets / "Eval_Layout" / "RoboDojo" / config / str(folder * 100 + 50 + k)
        dst.mkdir(parents=True, exist_ok=False)
        for j in range(per):
            layout = self.layout(config, folder, per * k + j)
            (dst / f"{self.task}_{j}.json").write_text(json.dumps(layout, indent=1) + "\n")
        return dst


def round_pool(round_, commercial=False, pool=None):
    """The objects a round draws from: the pool, its commercially usable part, less EXCLUDED's from round 7 on."""
    pool = pool or pick_anything.load_pool()
    if commercial:
        pool = pick_anything.commercial(pool)
    return pool if round_ in DRAWN_BEFORE_EXCLUDED else pick_anything.usable(pool, TASK)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True, help="Assets with the pool's object metadata")
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--folders", type=int, nargs=2, required=True, help="first and last candidate folder")
    parser.add_argument("--units", type=int, default=11, help="units of PER_UNIT layouts per folder")
    parser.add_argument("--levels", default="", help="diagnostics: axis=level,... (the rest at L2)")
    parser.add_argument("--round", default=ROUND, help=f"collection round (default {ROUND})")
    parser.add_argument("--commercial", action="store_true", help="only the pool's commercially usable objects")
    parser.add_argument(
        "--wholebody",
        action="store_true",
        help="enable 19D fixed-base wholebody recipe (expanded reach depth y, table height & initial lift/waist)",
    )
    args = parser.parse_args(argv)
    levels = {k: int(v) for k, v in (item.split("=") for item in args.levels.split(",") if item)}
    if set(levels) - set(DIMS) or not all(0 <= v <= LEVEL for v in levels.values()):
        parser.error(f"--levels: axes of {DIMS} at 0-{LEVEL}")
    gen = Generator(
        args.assets,
        round_pool(args.round, args.commercial),
        levels=levels,
        round_=args.round,
        wholebody=args.wholebody,
    )
    for folder in range(args.folders[0], args.folders[1] + 1):
        for k in range(args.units):
            print(gen.write_unit(args.config, folder, k))


if __name__ == "__main__":
    main()
