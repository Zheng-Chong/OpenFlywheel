"""Import ArtVIP articulated objects (X-Humanoid/ArtVIP on Hugging Face, Apache-2.0) as RoboDojo Articulation assets.

Scope: tabletop and table-side small appliances, small furniture and household items with a joint a robot can work
(SCOPE / EXCLUDED_KINDS below; every item's decision and reason goes into the selection manifest). The original files
stay as published under <library root>/sources/artvip/ (item folders, README.md, UPSTREAM.json with the revision and
licence); the adapter layer (articulated.py) goes to Assets/Object/RoboDojo/Articulation/artvip_<kind>/<index>/.

    fetch    --out ROOT/sources/artvip [--kinds ...]   files of the in-scope kinds at REVISION (urllib; the
             huggingface_hub client fails on some machines)
    select   --source ROOT/sources/artvip --manifest artvip_selection.json   per item: size, joints, decision, reason
    convert  --root ROOT --manifest artvip_selection.json   object.usd, metadata.json, description.json per item

Needs pxr (usd-core is enough for select and convert). Masses are estimates per kind (MASS_KG), sizes are the
source models' (ArtVIP models are metric, Z-up, origin at the bottom centre).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import articulated  # noqa: E402

REPO = "X-Humanoid/ArtVIP"
REVISION = "d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e"
URL = f"https://huggingface.co/datasets/{REPO}"
LICENSE = "Apache-2.0"
ATTRIBUTION = "ArtVIP, X-Humanoid (Beijing Humanoid Robot Innovation Center), huggingface.co/datasets/X-Humanoid/ArtVIP"
TOPS = ("small_appliances", "small_furniture", "household_items")
# kinds that are not tabletop or table-side objects, whatever their size
EXCLUDED_KINDS = {
    "small_furniture/chair": "seating, not a tabletop or table-side object",
    "small_furniture/stool": "seating, not a tabletop or table-side object",
    "small_furniture/door": "room door",
    "small_furniture/easel": "floor-standing easel",
    "small_furniture/faucet": "sink fixture, needs a basin",
    "small_furniture/floor_lamp": "floor-standing lamp",
    "small_furniture/table": "a table itself (desks, dressing tables, workbenches): large furniture",
    "small_furniture/bookcase": "bookcase: large furniture",
    "household_items/mop": "floor-cleaning tool",
    "household_items/showerhead": "bathroom wall fixture",
    "small_appliances/water dispenser": "floor-standing water dispenser",
}
# tabletop: fits on the G1-D table and within arm reach; table side: low furniture next to the table
TABLETOP_MAX_M, TABLESIDE_MAX_M = 0.65, 0.9
# estimated whole-object masses, kg: typical retail weights of the kind (not measured)
MASS_KG = {
    "airfryer": 4.0,
    "cash_register": 6.0,
    "coffee machine": 5.0,
    "desk_lamp": 1.5,
    "laptop": 1.8,
    "microwave": 12.0,
    "oven": 8.0,
    "printer": 6.0,
    "rice cooker": 3.5,
    "sound": 2.0,
    "table fan": 2.5,
    "television": 5.0,
    "filing_cabinet/double_door": 8.0,
    "filing_cabinet/drawers": 6.0,
    "filing_cabinet/sliding_door": 8.0,
    "shoe_cabinet": 15.0,
    "cabinet": 12.0,
    "tea_table": 15.0,
    "basket": 0.6,
    "cardboard_box": 0.3,
    "clock": 0.6,
    "holder": 0.3,
    "match": 0.05,
    "pump_bottle": 0.4,
    "stationery/folder": 0.15,
    "stationery/globe": 1.0,
    "stationery/knife": 0.08,
    "stationery/letter_tray": 0.8,
    "stationery/notebook": 0.3,
    "stationery/pencil": 0.05,
    "stationery/pencil_case": 0.15,
    "stationery/scissors": 0.08,
    "stationery/stapler": 0.25,
    "tongs": 0.15,
    "toolbox": 2.5,
    "trash_can": 1.5,
}
# link names that say nothing about the part: {source item folder name: {link: part}}
LINK_PARTS = {"match": {"E_ComponentM10_1": "drawer"}}
# parts that open something in these kinds only (articulated.role_of): a laptop's hinged screen closes, a letter
# tray's trays pull out like drawers, a notebook's cover opens; elsewhere a screen, tray or cover is role "other"
PART_ROLES = {
    "laptop": {"screen": "lid"},
    "stationery/letter_tray": {"tray": "drawer"},
    "stationery/notebook": {"cover": "lid"},
}
# portable kinds a task picks up or moves: layouts leave their base free (metadata physics.fixed_base false)
FREE_KINDS = (
    "basket",
    "cardboard_box",
    "clock",
    "holder",
    "match",
    "pump_bottle",
    "stationery/",
    "tongs",
    "toolbox",
    "laptop",
)
# real largest dimension of the kind, m, from the smallest to the largest common retail product (wooden pencils
# 19 cm, multi-tier letter trays, 80 cm kitchen bins, 70 cm tool chests): the size check (metadata size_check ->
# check.json "size"); beyond 1.5 x either end select drops the model as wrongly scaled
SIZE_M = {
    "airfryer": (0.2, 0.45),
    "cash_register": (0.3, 0.6),
    "coffee machine": (0.2, 0.55),
    "desk_lamp": (0.2, 0.8),
    "laptop": (0.25, 0.45),
    "microwave": (0.4, 0.65),
    "oven": (0.35, 0.7),
    "printer": (0.3, 0.65),
    "rice cooker": (0.2, 0.45),
    "sound": (0.08, 0.6),
    "table fan": (0.2, 0.6),
    "television": (0.3, 1.0),
    "filing_cabinet/double_door": (0.25, 0.9),
    "filing_cabinet/drawers": (0.25, 0.9),
    "filing_cabinet/sliding_door": (0.25, 0.9),
    "shoe_cabinet": (0.5, 1.3),
    "cabinet": (0.35, 0.9),
    "tea_table": (0.4, 1.4),
    "basket": (0.15, 0.75),
    "cardboard_box": (0.1, 0.7),
    "clock": (0.1, 0.5),
    "holder": (0.05, 0.4),
    "match": (0.04, 0.08),
    "pump_bottle": (0.12, 0.35),
    "stationery/folder": (0.2, 0.4),
    "stationery/globe": (0.2, 0.5),
    "stationery/knife": (0.1, 0.25),
    "stationery/letter_tray": (0.2, 0.5),
    "stationery/notebook": (0.15, 0.35),
    "stationery/pencil": (0.12, 0.22),
    "stationery/pencil_case": (0.15, 0.3),
    "stationery/scissors": (0.1, 0.3),
    "stationery/stapler": (0.1, 0.3),
    "tongs": (0.15, 0.4),
    "toolbox": (0.25, 0.75),
    "trash_can": (0.2, 0.85),
}


def kind_of(rel):
    """'small_furniture/filing_cabinet/drawers/chest_of_drawers_1' -> ('small_furniture/filing_cabinet/drawers',
    'filing_cabinet/drawers'): the folder path between the top group and the item, and its key in the tables."""
    parts = rel.split("/")
    top, middle = parts[1], parts[2:-1]
    full = "/".join([top, *middle])
    for n in range(len(middle), 0, -1):
        key = "/".join(middle[:n])
        if key in MASS_KG:
            return full, key
    return full, "/".join(middle)


def get(url, attempts=5):
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if k == attempts - 1:
                raise
            time.sleep(3 * (k + 1))


def listing():
    """[(path, size)] of the dataset at REVISION, from the Hub tree API."""
    out, url = [], f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=false"
    while url:
        with urllib.request.urlopen(url, timeout=120) as r:
            out += [(e["path"], e.get("size", 0)) for e in json.loads(r.read()) if e["type"] == "file"]
            nxt = [p for p in r.headers.get("Link", "").split(",") if 'rel="next"' in p]
            url = nxt[0].split(";")[0].strip(" <>") if nxt else None
    return out


def fetch(out, tops=TOPS):
    out = Path(out)
    files = [
        (p, s)
        for p, s in listing()
        if p in ("README.md", "Articulated_objects/README.md") or (p.split("/")[:2][-1] in tops and p.count("/") > 2)
    ]
    for k, (path, size) in enumerate(files):
        dest = out / path
        if dest.is_file() and dest.stat().st_size == size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = get(f"{URL}/resolve/{REVISION}/{urllib.parse.quote(path)}")
        if len(data) != size:
            raise OSError(f"{path}: {len(data)} bytes, listing says {size}")
        dest.write_bytes(data)
        if k % 200 == 0:
            print(f"{k}/{len(files)} {path}", flush=True)
    upstream = {"dataset": REPO, "url": URL, "revision": REVISION, "license": LICENSE, "attribution": ATTRIBUTION}
    articulated.write_json(out / "UPSTREAM.json", {**upstream, "files": len(files)})
    return files


def items(source):
    """Item folders: the ones holding a model_*.usd(a) (the 'origin' sub-folders only hold a base copy)."""
    found = []
    for top in TOPS:
        for model in sorted((Path(source) / "Articulated_objects" / top).rglob("model*.usd*")):
            if model.parent.name == "origin" or model.suffix not in (".usd", ".usda"):
                continue
            found.append(model)
    return found


def survey(model):
    """Size (m) and movable joints of a source model, without writing anything."""
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(model))
    facts = articulated.inspect(stage)
    for path in facts["lights"]:
        stage.GetPrimAtPath(path).SetActive(False)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    size = (np.asarray(rng.GetMax()) - np.asarray(rng.GetMin())) * facts["meters_per_unit"]
    return [round(float(v), 4) for v in size], facts


def decide(kind, key, size, facts):
    """(chosen, reason)."""
    if kind in EXCLUDED_KINDS:
        return False, EXCLUDED_KINDS[kind]
    if key not in MASS_KG:
        return False, f"kind {key} has no mass estimate (not reviewed)"
    if facts["world_movable"]:
        return False, f"{facts['world_movable'][0]['name']} hinges a link to the world, not to the body"
    if not facts["movable"]:
        return False, "no revolute or prismatic joint between two links"
    names = [j["name"] for j in facts["movable"]]
    if len(set(names)) < len(names):
        return False, "two joints share a prim name, so their DOFs cannot be told apart"
    parts = sorted({articulated.part_of(j["body1"].rsplit("/", 1)[-1], j["type"]) for j in facts["movable"]})
    if parts == ["pointer"]:
        return False, "only clock hands move (unlimited, display only): no joint to work"
    if facts["up_axis"] != "Z":
        return False, f"up axis {facts['up_axis']}"
    if len(facts["bases"]) != 1:
        return False, f"base link ambiguous: {facts['bases']}"
    if not facts["colliders"]:
        return False, "no collision geometry"
    big = max(size)
    lo, hi = SIZE_M[key]
    if big > 1.5 * hi or big < lo / 1.5:
        return False, f"model {big:.2f} m, far outside a real {key.split('/')[-1]} ({lo}-{hi} m): wrong scale"
    if big > TABLESIDE_MAX_M or size[2] > TABLESIDE_MAX_M:
        return False, f"{max(size):.2f} m: large furniture, out of reach next to the table"
    if min(size) < 0.003:
        return False, "degenerate size"
    where = "tabletop" if big <= TABLETOP_MAX_M else "table side"
    return True, f"{where}, {len(facts['movable'])} joint(s): {', '.join(parts)}"


def select(source):
    source = Path(source)
    chosen, dropped, counters = [], [], {}
    for model in items(source):
        rel = model.parent.relative_to(source).as_posix()
        kind, key = kind_of(rel)
        try:
            size, facts = survey(model)
            ok, reason = decide(kind, key, size, facts)
        except Exception as exc:  # unreadable source: record and go on
            dropped.append({"src": rel, "kind": kind, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        entry = {"src": rel, "model": model.name, "kind": kind, "size_m": size, "reason": reason}
        if not ok:
            dropped.append(entry)
            continue
        category = "artvip_" + articulated.slug(kind.split("/", 1)[1])
        idx = counters.get(category, 0)
        counters[category] = idx + 1
        chosen.append({**entry, "id": f"Articulation/{category}/{idx:05d}", "key": key})
    return {"source": URL, "revision": REVISION, "license": LICENSE, "chosen": chosen, "dropped": dropped}


def size_check(key, size):
    lo, hi = SIZE_M[key]
    big = max(size)
    ok = lo <= big <= hi
    return {
        "result": "pass" if ok else "fail",
        "note": f"largest side {big:.3f} m vs {lo}-{hi} m for a real {key} (estimated range)",
    }


def convert(root, manifest):
    root = Path(root)
    done, failed = [], []
    for entry in manifest["chosen"]:
        target = root / "Assets/Object/RoboDojo" / entry["id"]
        try:
            facts = articulated.write_layer(
                root / "sources/artvip" / entry["src"] / entry["model"],
                target,
                MASS_KG[entry["key"]],
                root,
                link_roles=LINK_PARTS.get(Path(entry["src"]).name),
                part_roles=PART_ROLES.get(entry["key"]),
                still_parts=True,  # ArtVIP drives its doors, drawers and lids toward their open limits
            )
            source = {
                "dataset": REPO,
                "url": URL,
                "revision": REVISION,
                "object": entry["src"],
                "license": LICENSE,
                "attribution": ATTRIBUTION,
                "rsi_pools": "allowed (Apache-2.0)",
            }
            meta = articulated.metadata(
                entry["id"],
                facts,
                MASS_KG[entry["key"]],
                f"estimated: typical {entry['key']} weight",
                source,
                fixed_base=not entry["key"].startswith(FREE_KINDS),
            )
            meta["size_check"] = size_check(entry["key"], facts["size"])
            articulated.write_json(target / "metadata.json", meta)
            name = Path(entry["src"]).name.replace("_", " ")
            articulated.write_json(
                target / "description.json",
                {
                    "uuid": meta["uuid"],
                    "caption": {"name": [name]},
                    "description": [f"An articulated {entry['key'].split('/')[-1]} from ArtVIP ({entry['src']})."],
                },
            )
            sha = hashlib.sha256((target / "object.usd").read_bytes()).hexdigest()
            done.append({"id": entry["id"], "src": entry["src"], "size_m": meta["physics"]["size"], "sha256": sha})
        except Exception as exc:
            failed.append({"id": entry["id"], "src": entry["src"], "error": f"{type(exc).__name__}: {exc}"})
    return {"revision": REVISION, "converted": done, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "select", "convert"))
    parser.add_argument("--out", type=Path, help="fetch: sources/artvip folder")
    parser.add_argument("--source", type=Path, help="select: sources/artvip folder")
    parser.add_argument("--root", type=Path, help="convert: library root holding Assets/ and sources/")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        print(f"{len(fetch(args.out))} files in {args.out}")
    elif args.command == "select":
        result = select(args.source)
        args.manifest.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        print(f"chosen={len(result['chosen'])}, dropped={len(result['dropped'])}")
    else:
        result = convert(args.root, json.loads(args.manifest.read_text()))
        out = args.root / "artvip_conversion.json"
        out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"converted={len(result['converted'])}, failed={len(result['failed'])} -> {out}")


if __name__ == "__main__":
    main()
