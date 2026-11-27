"""Layouts of g1d_handover_anything (把 X 交给另一只手再放到 Y 上), round 1: g1d_pick_anything round 3's recipe
(scripts/datagen/pick_anything_layouts.py: every axis of the platform's five at L2, the office or the Shenzhen lab
desk, lights, materials, camera, robot base, object scale) with X on one side of the desk and a support Y on the
other (scripts/objects/handover_anything.py):

    spatial   X |x| 0.08-0.24 m on the giver's side, y -0.50 - -0.32 m, ±30 deg of the grip yaw; Y |x| 0.10-0.22 m
              on the taker's side, y -0.50 - -0.36 m, any yaw X fits it at as set down (L1: X |x| 0.10-0.20,
              y -0.47 - -0.36, ±20 deg; Y |x| 0.12-0.18, y -0.46 - -0.40); either hand gives; robot base, lift and
              desk as round 3
    object    X: the handover pool (pick anything's commercial objects with a long side, handover_anything_pool.json)
              scaled as round 3 where no training support then takes it; Y: the training supports of
              place_anything_receptacles_r2.json, unscaled
    language  X's same-family distractors as round 3; with probability 0.5 a second support among the distractors;
              the instruction names the hands (L1: a same-family distractor in half the layouts, no second support)

Distractors are 1-3 commercial pick anything objects no taller than 12 cm, off the middle band the hands meet over.
Units, candidate folders and seeds as round 3: unit <folder>-<k> = layouts 50k .. 50k+49 of <folder>, written to
<assets>/Eval_Layout/RoboDojo/<config>/<folder*100+50+k>/g1d_handover_anything_<j>.json.

    python scripts/datagen/handover_anything_layouts.py --assets <run>/Assets --config g1d --folders 1300 1301 --units 2
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen import (
    pick_anything_layouts as pal,  # noqa: E402
    place_anything_layouts as plal,  # noqa: E402
)
from scripts.objects import (
    handover_anything as ho,  # noqa: E402
    pick_anything,  # noqa: E402
    place_anything as pa,  # noqa: E402
)

TASK = ho.TASK
ROUND = f"{TASK}-r1"
RECEPTACLES = ROOT / "scripts/objects/place_anything_receptacles_r2.json"


WHOLEBODY_DEFAULT = {
    **{k: v for k, v in pal.WHOLEBODY_DEFAULT.items() if k != "target_y"},
    "x_y": (-0.48, -0.25),
    "y_y": (-0.48, -0.28),
}


class Generator(pal.Generator):
    task = TASK

    def __init__(
        self,
        assets,
        pool=None,
        distractors=None,
        receptacles=None,
        names=None,
        levels=None,
        round_=ROUND,
        wholebody=None,
    ):
        distractors = plal.commercial(distractors or pick_anything.load_pool())
        wb = dict(WHOLEBODY_DEFAULT, **(wholebody if isinstance(wholebody, dict) else {})) if wholebody else None
        super().__init__(assets, distractors, names, levels, round_, wholebody=wb)
        self.x_pool = pool or ho.load_pool()
        self.receptacles = receptacles or pa.load_receptacles(RECEPTACLES)
        lv = self.levels
        spec = dict(ho.SPEC if lv["spatial"] >= 2 else ho.SPEC_L1)
        if lv["language"] < 2:
            spec.update(same_family=ho.SPEC_L1["same_family"], other_receptacle=0.0)
        spec["scale"] = self.spec["scale"]
        if self.wholebody:
            spec["x_y"] = tuple(self.wholebody["x_y"])
            spec["y_y"] = tuple(self.wholebody["y_y"])
        self.spec = spec
        self.order = ho.order(self.x_pool, self.receptacles)
        self.family.update({(o["cat"], o["idx"]): o["family"] for o in self.x_pool})
        self.drawn = None

    def rigid(self, rng, index, top):
        rigid, self.drawn = ho.layout(
            rng,
            index,
            pal.entry_maker(self.assets, top),
            plal.receptacle_maker(self.assets, top, ho.KIND),
            self.x_pool,
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
    parser.add_argument("--assets", type=Path, required=True, help="Assets with the pools' object metadata")
    parser.add_argument("--config", choices=pal.CONFIGS, required=True)
    parser.add_argument("--folders", type=int, nargs=2, required=True, help="first and last candidate folder")
    parser.add_argument("--units", type=int, default=11, help="units of PER_UNIT layouts per folder")
    parser.add_argument("--levels", default="", help="diagnostics: axis=level,... (the rest at L2)")
    parser.add_argument("--round", default=ROUND, help=f"collection round (default {ROUND})")
    parser.add_argument("--receptacles", type=Path, default=RECEPTACLES, help="support pool (default round 2's)")
    parser.add_argument(
        "--wholebody",
        action="store_true",
        help="enable 19D fixed-base wholebody recipe (expanded reach depth y, table height & initial lift/waist)",
    )
    args = parser.parse_args(argv)
    levels = {k: int(v) for k, v in (item.split("=") for item in args.levels.split(",") if item)}
    if set(levels) - set(pal.DIMS) or not all(0 <= v <= pal.LEVEL for v in levels.values()):
        parser.error(f"--levels: axes of {pal.DIMS} at 0-{pal.LEVEL}")
    gen = Generator(
        args.assets,
        receptacles=pa.load_receptacles(args.receptacles),
        levels=levels,
        round_=args.round,
        wholebody=args.wholebody,
    )
    print(f"{TASK}: {len(gen.order)} of {len(gen.x_pool)} pool objects fit a training support")
    for folder in range(args.folders[0], args.folders[1] + 1):
        for k in range(args.units):
            print(gen.write_unit(args.config, folder, k))


if __name__ == "__main__":
    main()
