"""Layouts of g1d_place_anything (把 X 放到 Y 上) and g1d_put_anything_in (把 X 放进 Y 里), round 1: g1d_pick_anything
round 3's recipe (scripts/datagen/pick_anything_layouts.py: every axis of the platform's five at L2, the office or the
Shenzhen lab desk, lights, materials, camera, robot base, object scale) with a receptacle Y in the middle of the desk
(scripts/objects/place_anything.py):

    spatial   X as round 3 (|x| 0.02-0.25 m, y -0.53 - -0.30 m, ±30 deg of the grip yaw); Y's centre |x| <= 0.10 m,
              y -0.47 - -0.37 m, any yaw (L1: |x| <= 0.05, y -0.44 - -0.40, ±15 deg either way along X)
    object    X: the pool's 551 objects scaled as round 3; Y: the training receptacles of the task's kind, unscaled
              (their geometry is measured at scale 1)
    language  X's same-family distractors as round 3; with probability 0.5 a second receptacle (of Y's family when
              there is one) among the distractors, so the instruction names Y by a name it alone has (L1: neither)

X is from g1d_pick_anything's pool without the objects of noncommercial libraries (commercial: 295 of 551; the
receptacles follow the same rule): the object pick_anything.pool_order puts at index i of those (mod the ones some
training receptacle takes); Y a receptacle it fits at its yaw. Units, candidate folders and seeds as round 3: unit
<folder>-<k> = layouts 50k .. 50k+49 of <folder>, written to
<assets>/Eval_Layout/RoboDojo/<config>/<folder*100+50+k>/<task>_<j>.json.

    python scripts/datagen/place_anything_layouts.py --task g1d_place_anything --assets <run>/Assets --config g1d \
        --folders 1100 1109 --units 11

A later round passes --round (its name seeds the draws) and its own folders; round 2 of g1d_place_anything also
--receptacles scripts/objects/place_anything_receptacles_r2.json (round 1's receptacles and 12 more supports).
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen import pick_anything_layouts as pal  # noqa: E402
from scripts.g1d import build_asset  # noqa: E402
from scripts.objects import (
    pick_anything,  # noqa: E402
    place_anything as pa,  # noqa: E402
)

ROUNDS = {task: f"{task}-r1" for task in pa.TASKS.values()}
DRAWN_BEFORE_EXCLUDED = {f"{task}-r{n}" for task in pa.TASKS.values() for n in (1, 2)}  # no pick_anything.EXCLUDED
KIND = {task: kind for kind, task in pa.TASKS.items()}
SPEC_L2 = dict(pal.SPEC, receptacle_x=0.10, receptacle_y=(-0.47, -0.37), receptacle_yaw=180.0)
OTHER_RECEPTACLE = 0.5  # language L2
MASS_DENSITY, MASS_FILL, MASS_RANGE = 400.0, 0.5, (0.05, 0.5)  # kg/m3 of the bounding box, share filled: no mass given


def commercial(pool):
    """The pool objects of commercially usable libraries: not of those the platform's library registrations mark
    noncommercial (geniesim CC-BY-NC-SA-4.0, partnet_mobility), the rule the receptacle candidates follow
    (place_anything.NONCOMMERCIAL)."""
    return pick_anything.commercial(pool)


def estimated_mass(size):
    w, d, h = size
    return round(min(max(MASS_DENSITY * MASS_FILL * w * d * h, MASS_RANGE[0]), MASS_RANGE[1]), 4)


def receptacle_maker(assets, top, kind):
    """receptacle_entry(rec, label, xy, yaw) of receptacles standing on the desk top (z `top`): the root at its
    measured height above its underside, the pool's size and (estimated) mass, and for the task's Y its geometry."""

    def entry(rec, label, xy, yaw):
        meta = build_asset.object_meta(assets, f"Rigid/{rec['cat']}/{rec['idx']:05d}")
        physics = dict(meta.get("physics") or {}, type="rigid", size=rec["size"])
        physics["mass"] = rec["mass"] or estimated_mass(rec["size"])
        geom = rec["geometry"][kind] if kind in rec["geometry"] else next(iter(rec["geometry"].values()))
        e = {
            "category": rec["cat"],
            "category_idx": rec["idx"],
            "label": label,
            "default_pos": [*map(float, xy), top - geom["bottom"] + 1e-3],
            "default_ori": build_asset._yaw_quat(yaw),
            "scale": [1.0, 1.0, 1.0],
            "physics": physics,
            "visual": {},
            "relative_plane": "Table",
            "need_check_stable": True,
        }
        if label == pa.LABEL:
            e["receptacle"] = dict(geom, kind=kind, id=rec["id"])
        return e

    return entry


WHOLEBODY_DEFAULT = dict(pal.WHOLEBODY_DEFAULT, target_y=(-0.50, -0.24), receptacle_y=(-0.46, -0.28))


class Generator(pal.Generator):
    def __init__(self, task, assets, pool=None, receptacles=None, names=None, levels=None, round_=None, wholebody=None):
        round_ = round_ or ROUNDS[task]
        pool = commercial(pool or pick_anything.load_pool())
        if round_ not in DRAWN_BEFORE_EXCLUDED:
            pool = pick_anything.usable(pool, task)
        wb = dict(WHOLEBODY_DEFAULT, **(wholebody if isinstance(wholebody, dict) else {})) if wholebody else None
        super().__init__(assets, pool, names, levels, round_, wholebody=wb)
        self.task, self.kind = task, KIND[task]
        self.receptacles = receptacles or pa.load_receptacles()
        lv = self.levels
        spec = dict(SPEC_L2 if lv["spatial"] >= 2 else pa.SPEC)
        spec.update(same_family=self.spec["same_family"], scale=self.spec["scale"])
        spec["other_receptacle"] = OTHER_RECEPTACLE if lv["language"] >= 2 else 0.0
        if self.wholebody:
            spec["target_y"] = tuple(self.wholebody["target_y"])
            spec["receptacle_y"] = tuple(self.wholebody["receptacle_y"])
        self.spec = spec
        self.order = pa.x_order(self.pool, self.receptacles, self.kind)
        self.drawn = None

    def rigid(self, rng, index, top):
        rigid, self.drawn = pa.layout(
            rng,
            index,
            pal.entry_maker(self.assets, top),
            receptacle_maker(self.assets, top, self.kind),
            self.kind,
            self.pool,
            self.receptacles,
            self.spec,
            self.order,
        )
        return rigid

    def layout(self, config, folder, index):
        out = super().layout(config, folder, index)
        rec = next(e for es in out["Rigid"].values() for e in es if e["label"] == pa.LABEL)
        out["Generalization"].update(self.drawn, receptacle_xy=[round(v, 4) for v in rec["default_pos"][:2]])
        return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(KIND), required=True)
    parser.add_argument("--assets", type=Path, required=True, help="Assets with the pools' object metadata")
    parser.add_argument("--config", choices=pal.CONFIGS, required=True)
    parser.add_argument("--folders", type=int, nargs=2, required=True, help="first and last candidate folder")
    parser.add_argument("--units", type=int, default=11, help="units of PER_UNIT layouts per folder")
    parser.add_argument("--levels", default="", help="diagnostics: axis=level,... (the rest at L2)")
    parser.add_argument("--round", default=None, help="collection round (default <task>-r1)")
    parser.add_argument("--receptacles", type=Path, default=pa.RECEPTACLES, help="receptacle pool (default round 1's)")
    parser.add_argument(
        "--wholebody",
        action="store_true",
        help="enable 19D fixed-base wholebody recipe (expanded reach depth y, table height & initial lift/waist)",
    )
    args = parser.parse_args(argv)
    levels = {k: int(v) for k, v in (item.split("=") for item in args.levels.split(",") if item)}
    if set(levels) - set(pal.DIMS) or not all(0 <= v <= pal.LEVEL for v in levels.values()):
        parser.error(f"--levels: axes of {pal.DIMS} at 0-{pal.LEVEL}")
    receptacles = pa.load_receptacles(args.receptacles)
    gen = Generator(
        args.task,
        args.assets,
        receptacles=receptacles,
        levels=levels,
        round_=args.round,
        wholebody=args.wholebody,
    )
    print(f"{args.task}: {len(gen.order)} of {len(gen.pool)} pool objects fit a training receptacle")
    for folder in range(args.folders[0], args.folders[1] + 1):
        for k in range(args.units):
            print(gen.write_unit(args.config, folder, k))


if __name__ == "__main__":
    main()
