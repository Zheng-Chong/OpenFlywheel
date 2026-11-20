"""Layouts of g1d_sort_anything (按种类分拣任意物体), round 1: g1d_pick_anything round 3's recipe
(scripts/datagen/pick_anything_layouts.py: the office or Shenzhen lab desk, lights, materials, camera, robot base,
object scale, every axis at L2) with 2-4 objects of 2 or 3 sort classes and a receptacle per class
(scripts/objects/sort_anything.py):

    spatial   receptacles' centres |x| <= 0.20 m, y -0.47 - -0.37 m, at least 12 cm apart along X, any yaw; objects
              |x| 0.02-0.25 m, y -0.53 - -0.30 m within ±30 deg of the grip yaw (L1: receptacles in fixed slots
              along X ±2 cm, y -0.44 - -0.40, within 15 deg of either way along X; grip yaw ±20 deg)
    object    objects of 18 sort classes of the pool's commercially usable part, scaled as round 3 (unscaled when
              the class's receptacle would not take them scaled); receptacles from Place / Put Anything round 2's
              training pool, unscaled
    language  the instruction names each class by its class name and its receptacle by a name nothing else on the
              desk has; at L2 half the layouts with two classes add a spare receptacle and half draw the second
              receptacle of the first's family (L1: neither); left / middle / right in half the instructions

The instruction (one language per layout, zh or en) is the layout's "Instruction"; the draws are in
"Generalization" ("sort"). Units, candidate folders and seeds as pick_anything round 3: unit <folder>-<k> = layouts
50k .. 50k+49 of <folder>, written to <assets>/Eval_Layout/RoboDojo/<config>/<folder*100+50+k>/<task>_<j>.json.

    python scripts/datagen/sort_anything_layouts.py --assets <run>/Assets --config g1d --folders 1300 1309 --units 11
"""

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen import (
    pick_anything_layouts as pal,  # noqa: E402
    place_anything_layouts as pla,  # noqa: E402
)
from scripts.objects import (
    object_names,  # noqa: E402
    pick_anything,  # noqa: E402
    place_anything as pa,  # noqa: E402
    sort_anything as sa,  # noqa: E402
)

TASK = sa.TASK
ROUND = f"{TASK}-r1"
RECEPTACLES = ROOT / "scripts" / "objects" / "place_anything_receptacles_r2.json"
LANGS = ("en", "zh")


def receptacle_maker(assets, top):
    """receptacle_entry(rec, kind, label, xy, yaw): place_anything_layouts' entry with the geometry of `kind`."""

    def entry(rec, kind, label, xy, yaw):
        e = pla.receptacle_maker(assets, top, kind)(rec, label, xy, yaw)
        e["receptacle"] = dict(rec["geometry"][kind], kind=kind, id=rec["id"])
        return e

    return entry


class Generator(pal.Generator):
    task = TASK

    def __init__(self, assets, pool=None, receptacles=None, names=None, levels=None, round_=ROUND):
        names = names or object_names.load()
        super().__init__(assets, pla.commercial(pool or pick_anything.load_pool()), names, levels, round_)
        self.names = names
        self.settings = sa.ROUNDS[round_.rsplit("-", 1)[-1]]
        cap = self.settings["receptacle_max"]
        self.receptacles = [
            r for r in receptacles or pa.load_receptacles(RECEPTACLES) if cap is None or max(r["size"][:2]) <= cap
        ]
        self.classes = sa.class_pool(self.pool, names, self.receptacles)
        lv = self.levels
        scale = self.spec["scale"]
        self.spec = dict(sa.SPEC_L2 if lv["spatial"] >= 2 else sa.SPEC_L1, **self.settings["spec"], scale=scale)
        self.spare = sa.SPARE if lv["language"] >= 2 else 0.0
        self.same_family = sa.SAME_FAMILY if lv["language"] >= 2 else 0.0
        self.drawn = None

    def rigid(self, rng, index, top):
        rigid, self.drawn = sa.layout(
            rng,
            pal.entry_maker(self.assets, top),
            receptacle_maker(self.assets, top),
            self.classes,
            self.receptacles,
            self.spec,
            self.spec["scale"],
            self.spare,
            self.same_family,
            self.settings["three"],
            self.settings["items"],
            self.settings["receptacles"],
        )
        return rigid

    def record(self, entries):
        return {
            "scale": {k: e["scale"][0] for k, e in sorted(entries.items())},
            "objects": len(entries),
        }

    def names_of(self, entry, lang):
        n = self.names.get(f"Rigid/{entry['category']}/{entry['category_idx']:05d}") or {}
        category = (n.get("category") or {}).get(lang)
        return [category, *n.get(lang, [])] if category else list(n.get(lang, []))

    def layout(self, config, folder, index):
        out = super().layout(config, folder, index)
        entries = {e["label"]: e for es in out["Rigid"].values() for e in es}
        ys = sorted(k for k in entries if k.startswith(sa.RECEPTACLE))
        rng = random.Random(f"{self.round}/{config}/{folder}/{index}/instruction")
        lang = rng.choice(LANGS)
        side = sa.sides([entries[y]["default_pos"][0] for y in ys]) if rng.random() < sa.SIDE_WORDS else None
        parts, used = [], {}
        for j, (c, n) in enumerate(zip(self.drawn["classes"], self.drawn["items"])):
            label = ys[j]
            own = self.names_of(entries[label], lang)
            rest = [self.names_of(e, lang) for k, e in entries.items() if k != label]
            used[label] = rng.choice(pick_anything.usable_names(own, rest))
            parts.append((c, n, entries[label]["receptacle"]["kind"], used[label], j))
        rng.shuffle(parts)
        out["Instruction"] = sa.instruction(rng, parts, lang, side)
        out["Generalization"]["sort"] = dict(
            self.drawn,
            lang=lang,
            side_words=side is not None,
            receptacle_names=used,
            items_xy={k: [round(v, 4) for v in e["default_pos"][:2]] for k, e in sorted(entries.items())},
        )
        return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True, help="Assets with the pools' object metadata")
    parser.add_argument("--config", choices=pal.CONFIGS, required=True)
    parser.add_argument("--folders", type=int, nargs=2, required=True, help="first and last candidate folder")
    parser.add_argument("--units", type=int, default=11, help="units of PER_UNIT layouts per folder")
    parser.add_argument("--levels", default="", help="diagnostics: axis=level,... (the rest at L2)")
    parser.add_argument("--round", default=ROUND, help=f"collection round (default {ROUND})")
    args = parser.parse_args(argv)
    levels = {k: int(v) for k, v in (item.split("=") for item in args.levels.split(",") if item)}
    if set(levels) - set(pal.DIMS) or not all(0 <= v <= pal.LEVEL for v in levels.values()):
        parser.error(f"--levels: axes of {pal.DIMS} at 0-{pal.LEVEL}")
    gen = Generator(args.assets, levels=levels, round_=args.round)
    print(f"{TASK}: {', '.join(f'{c} {len(v)}' for c, v in sorted(gen.classes.items()))}")
    for folder in range(args.folders[0], args.folders[1] + 1):
        for k in range(args.units):
            print(gen.write_unit(args.config, folder, k))


if __name__ == "__main__":
    main()
