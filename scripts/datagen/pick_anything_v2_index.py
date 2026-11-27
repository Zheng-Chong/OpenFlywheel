"""Per-episode round and generalization levels of the pick_anything@v2 parts, so training can pick by round or level.

A row per LeRobot episode (episode_index = its place in the part's collection_quality.json): the collection round,
the level of each of the platform's five axes, where it was collected (env config, layout seed / id / sha256), the
target asset and the instruction. Rounds 1 and 2 drew every layout at one set of levels (ROUNDS); rounds 3 on's
layouts carry their own (scripts/datagen/pick_anything_layouts.py "Generalization", read from the layouts the
collection ran, their sha256 checked against the sidecars'), and the row keeps what each drew. Every row also lists
each object on the table with its source library and license (`objects`, `licenses`), and `commercial` is true when
none of them is non-commercial (GenieSim is CC BY-NC-SA 4.0; real2sim/ASSETS.md).

    python scripts/datagen/pick_anything_v2_index.py part --dataset <part dir> --round g1d_pick_anything-r3 \
        [--layouts <code tar>/Assets/Eval_Layout/RoboDojo ...]
        writes <part dir>/meta/episodes_generalization.jsonl and adds it to the part's sha256.json
    python scripts/datagen/pick_anything_v2_index.py index --out <file> --part <name>=<dir>=<round> ... \
        [--layouts ...] [--commercial <file>]
        all parts in one file (rows also name their part), reading the parts only (without --layouts, rounds 3 on's
        levels and draws come from each part's meta/episodes_generalization.jsonl); --commercial: the rows whose
        objects are all commercially usable
"""

import argparse
import hashlib
import json
from pathlib import Path

DIMS = ("spatial", "scene", "object", "camera", "language")
R2 = dict(zip(DIMS, (1, 0, 2, 0, 1)))  # real2sim/collections/g1d_pick_anything-r2/round.json; round 1: same code
ROUNDS = {"g1d_pick_anything-r1": R2, "g1d_pick_anything-r2": R2} | {
    f"g1d_pick_anything-r{n}": None for n in (3, 4, 5, 6, 7)
}
META = "meta/episodes_generalization.jsonl"
NONCOMMERCIAL = {"CC-BY-NC-SA-4.0"}
SOURCES = (  # category prefix -> library, license (real2sim/ASSETS.md); first match wins
    ("geniesim_", "AgiBot GenieSim", "CC-BY-NC-SA-4.0"),
    ("sand3d_", "Sand3D", "CC-BY-4.0"),
    ("fmb_", "FMB", "CC-BY-4.0"),
    ("simready_", "NVIDIA SimReady", "CC-BY-4.0"),
    ("wood_block_", "OpenFlywheel Blocks", "in-house"),
)
REAL2SIM = {  # pool categories under Object/Real2Sim (some linked from Object/RoboDojo); the rest are RoboDojo's own
    "airpods", "aurora_flange", "ceramic_cup", "drink_bottle", "empty_bottle", "game_controller", "glass_bottle",
    "mango_prop", "marker_box", "paper_wad", "plastic_cup", "round_lid", "soda_can", "tapered_cup", "tumbler",
    "water_bottle", "wooden_arch_block",
}  # fmt: skip


def source(cat):
    """(library, license) of a pool category."""
    for prefix, lib, lic in SOURCES:
        if cat.startswith(prefix):
            return lib, lic
    if cat in REAL2SIM:
        return "OpenFlywheel Real2Sim", "in-house"
    return "RoboDojo (HF RoboDojo-Benchmark/RoboDojo)", "Apache-2.0"


def candidates(seed):
    """Candidate folder of shard seed folder*100+k (filler-job: candidates folder*100+50+k)."""
    return seed // 100 * 100 + 50 + seed % 100


def layout_path(layouts, q):
    rel = Path(q["config_name"]) / str(candidates(q["layout_seed"])) / q["layout_file"]
    for root in layouts or ():
        if (Path(root) / rel).is_file():
            return Path(root) / rel
    raise SystemExit(f"{rel} is in none of the layout trees {layouts}")


def rows(dataset, round_, layouts=None):
    qualities = json.loads((Path(dataset) / "collection_quality.json").read_text())
    meta = Path(dataset) / META
    drawn = None
    if ROUNDS[round_] is None and not layouts:  # the part's own rows, written from the layouts by `part`
        drawn = [json.loads(line) for line in meta.read_text().splitlines()]
        if len(drawn) != len(qualities):
            raise SystemExit(f"{meta} has {len(drawn)} rows for {len(qualities)} episodes")
    out = []
    for i, q in enumerate(qualities):
        objects = {label: [cat, idx, *source(cat)] for label, (cat, idx) in sorted(q["objects"].items())}
        licenses = sorted({o[3] for o in objects.values()})
        row = dict(
            episode_index=i,
            round=round_,
            levels=ROUNDS[round_],
            env_cfg=q["env_cfg"],
            layout_seed=q["layout_seed"],
            layout_id=q["layout_id"],
            layout_sha256=q["layout_sha256"],
            target=q["objects"]["target"],
            instruction=q["instruction"],
            objects=objects,
            licenses=licenses,
            commercial=not NONCOMMERCIAL & set(licenses),
        )
        if drawn:
            if drawn[i]["layout_sha256"] != q["layout_sha256"]:
                raise SystemExit(f"{meta} row {i} is not the layout episode {i} ran")
            row["levels"], row["generalization"] = drawn[i]["levels"], drawn[i]["generalization"]
        elif ROUNDS[round_] is None:
            path = layout_path(layouts, q)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != q["layout_sha256"]:
                raise SystemExit(f"{path} is not the layout episode {i} ran")
            g = json.loads(data)["Generalization"]
            row["levels"] = g["levels"]
            row["generalization"] = {k: g[k] for k in ("scene", "target_xy", "scale", "same_family", "objects", "base")}
            row["generalization"].update(materials=sorted(g["materials"]), lights=len(g["lights"]))
        out.append(row)
    return out


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("part")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--round", choices=sorted(ROUNDS), required=True)
    p.add_argument("--layouts", type=Path, nargs="+")
    p = sub.add_parser("index")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--part", action="append", required=True, help="<name>=<dataset dir>=<round>")
    p.add_argument("--layouts", type=Path, nargs="+")
    p.add_argument("--commercial", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "part":
        items = rows(args.dataset, args.round, args.layouts)
        out = args.dataset / META
        write_jsonl(out, items)
        sums_path = args.dataset / "sha256.json"
        sums = json.loads(sums_path.read_text())
        sums[META] = hashlib.sha256(out.read_bytes()).hexdigest()
        sums_path.write_text(json.dumps(dict(sorted(sums.items())), indent=2) + "\n")
        print(f"{len(items)} episodes -> {out}")
        return
    items = []
    for spec in args.part:
        name, dataset, round_ = spec.split("=")
        items += [dict(part=name, **r) for r in rows(dataset, round_, args.layouts)]
    write_jsonl(args.out, items)
    print(f"{len(items)} episodes of {len(args.part)} parts -> {args.out}")
    if args.commercial:
        subset = [r for r in items if r["commercial"]]
        write_jsonl(args.commercial, subset)
        print(f"{len(subset)} episodes with only commercially usable objects -> {args.commercial}")


if __name__ == "__main__":
    main()
