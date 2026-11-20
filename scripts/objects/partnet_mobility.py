"""Import PartNet-Mobility articulated objects from the community USD conversion AlbertS666/PartNet-Mobility-USD
(Hugging Face) as RoboDojo Articulation assets, in their own namespace (partnet_<category>).

LICENCE: the upstream SAPIEN PartNet-Mobility terms (built on ShapeNet / PartNet) allow NON-COMMERCIAL RESEARCH AND
EDUCATIONAL USE ONLY, and sharing only with colleagues who first agree to the same terms. The assets never go into
the RSI training or exam pools (rsi/AGENTS.md); every metadata.json says so (source.license, source.rsi_pools).

The community USDs (Isaac Sim URDF importer) keep PartNet's normalised geometry: /Object/<Category> carries a 0.5
scale and the model's diagonal is about 1 m, not its real size; the prismatic limits are the URDF's, without the
0.5; every link has a 0.3 kg placeholder mass. The adapter (articulated.py) references /Object/<Category>, scales
each model to the real size of its kind (REAL_SIZE: an estimate per category, PartNet has no real dimensions), applies
the same scale to the prismatic travel, splits an estimated total mass over the links, and names joint roles from the
URDF visual names (door_frame, drawer_front, handle, knob ...).

    extract  --tars data_<Category>.tar ... --out ROOT/sources/partnet_mobility
    select   --source ROOT/sources/partnet_mobility --manifest partnet_selection.json
    convert  --root ROOT --manifest partnet_selection.json
    names    --root ROOT --manifest partnet_selection.json --previews <check_articulation --image-dir>

STATUS (2026-11-17): not validated. In check_articulation.py most of the first 25 (Microwave, countertop Oven)
diverge in the joint sweeps or render blank, see real2sim/articulated/partnet_mobility.json; do not submit them to
the asset library until that is solved.

The tars come from https://huggingface.co/datasets/AlbertS666/PartNet-Mobility-USD/resolve/<REVISION>/data/<C>.tar.
Needs pxr (usd-core is enough).
"""

import argparse
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import articulated  # noqa: E402

REPO = "AlbertS666/PartNet-Mobility-USD"
REVISION = "faf11263510b3d41264b0d7644ea9e7282226890"
URL = f"https://huggingface.co/datasets/{REPO}"
CATEGORIES = ("Microwave", "Oven", "StorageFurniture")
LICENSE = "SAPIEN PartNet-Mobility terms of use: non-commercial research and educational use only"
TERMS = "https://sapien.ucsd.edu/about#term (also ShapeNet https://www.shapenet.org/terms)"
ATTRIBUTION = (
    "PartNet-Mobility / SAPIEN (Xiang et al., CVPR 2020), PartNet (Mo et al., CVPR 2019), ShapeNet (Chang et al., "
    "2015); USD conversion by Mutian Shen (huggingface.co/datasets/AlbertS666/PartNet-Mobility-USD)"
)
RSI = "excluded: non-commercial research licence (rsi/AGENTS.md)"
TABLESIDE_MAX_M = 0.9  # as artvip.py: larger is furniture out of reach next to the table
# real size per kind, estimated (PartNet keeps no real dimensions); models face -X, so Y is their width
REAL_SIZE = {
    "Microwave": ("width", 0.50, "countertop microwaves are 0.45-0.55 m wide"),
    "Oven/countertop": ("width", 0.50, "countertop and toaster ovens are 0.40-0.55 m wide"),
    "Oven/range": ("height", 0.90, "freestanding ranges are 0.90 m to the cooktop"),
    "StorageFurniture/low": ("height", 0.70, "nightstands and low cabinets are 0.55-0.80 m high"),
    "StorageFurniture/tall": ("height", 1.80, "wardrobes and tall cabinets are 1.6-2.0 m high"),
}
MASS_KG = {"Microwave": 12.0, "Oven/countertop": 8.0, "Oven/range": 50.0, "StorageFurniture": 15.0}
TALL_RATIO = 1.5  # height / width at or above which a cabinet is a wardrobe
MIN_OVEN_HEIGHT_M = 0.15  # lower than this at countertop width, an "oven" is a hob
# freestanding ranges with a cooktop whose proportions pass for a countertop oven (squat): told apart by looking at
# the renders (2026-11-17; their URDFs are base_body + door + knob, the countertop ovens have a side control panel)
RANGES_BY_REVIEW = {"101908", "101946", "101947", "102001", "102018", "102044"}
FRONT_YAW_DEG = 90.0  # PartNet models face -X; turned to face -Y like ArtVIP and the official assets
FIXTURES = ("handle", "hinge", "other_leaf")  # URDF visual pieces fixed to the moving part, not the part itself
PIECE_PARTS = {"glass": "door", "mirror": "door"}  # a moving pane is a door's glass or a mirror cabinet's door


def extract(tars, out):
    """Unpack the category tars (data/<Category>/<id>/...) to out/<Category>/<id>/."""
    out = Path(out)
    n = 0
    for tar_path in tars:
        with tarfile.open(tar_path) as tar:
            for member in tar:
                name = Path(member.name)
                parts = name.parts[1:] if name.parts and name.parts[0] == "data" else name.parts
                if not parts or ".." in parts or member.issym() or member.islnk():
                    continue
                member.name = str(Path(*parts))
                tar.extract(member, out, filter="data")
                n += 1
    articulated.write_json(
        out / "UPSTREAM.json",
        {
            "dataset": REPO,
            "url": URL,
            "revision": REVISION,
            "license": LICENSE,
            "terms": TERMS,
            "attribution": ATTRIBUTION,
            "rsi_pools": RSI,
            "contents": "README.md and the object folders of the imported objects, byte-identical to the revision",
            "imported_by": "scripts/objects/partnet_mobility.py",
        },
    )
    return n


def link_roles(urdf):
    """{link: part} from the URDF's visual names (door_frame-6, drawer_front-3, glass-17, handle-2 ...): the pieces of
    the link. A handle, hinge or trim (FIXTURES) is fixed to the door or drawer it moves with, so it names the link
    only when nothing else does: a microwave door modelled as glass + handle is a door, not a handle."""
    roles = {}
    if not Path(urdf).is_file():
        return roles
    for link in ET.parse(urdf).getroot().iter("link"):
        pieces = {v.get("name", "").rsplit("-", 1)[0] for v in link.iter("visual")} - {""}
        main = sorted(p for p in pieces if not p.startswith(FIXTURES)) or sorted(pieces)
        part = articulated.part_of(" ".join(main), "revolute")
        if part == "other":
            part = next((PIECE_PARTS[w] for p in main for w in PIECE_PARTS if p.startswith(w)), "other")
        if part != "other":
            roles[link.get("name")] = part
    return roles


def survey(model, category):
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(model))
    facts = articulated.inspect(stage, f"/Object/{category}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(f"/Object/{category}")).ComputeAlignedRange()
    size = (np.asarray(rng.GetMax()) - np.asarray(rng.GetMin())) * facts["meters_per_unit"]
    return [float(v) for v in size], facts


def kind_of(category, size, object_id=""):
    """The REAL_SIZE / MASS_KG key of a model from its proportions (x depth, y width, z height)."""
    if category == "Oven":
        return "Oven/range" if size[2] >= size[1] or object_id in RANGES_BY_REVIEW else "Oven/countertop"
    if category == "StorageFurniture":
        return "StorageFurniture/tall" if size[2] >= TALL_RATIO * size[1] else "StorageFurniture/low"
    return category


def real_scale(kind, size):
    axis, metres, _ = REAL_SIZE[kind]
    return metres / (size[1] if axis == "width" else size[2])


def decide(kind, scaled, facts):
    if kind == "Oven/range":
        return False, "freestanding range: a floor appliance whose oven door sits below the table, out of reach"
    if kind == "Oven/countertop" and scaled[2] < MIN_OVEN_HEIGHT_M:
        return False, f"{scaled[2]:.2f} m high at {REAL_SIZE[kind][1]} m wide: a flat cooktop, not an oven"
    if facts["world_movable"]:
        return False, f"{facts['world_movable'][0]['name']} hinges a link to the world"
    if not facts["movable"]:
        return False, "no revolute or prismatic joint between two links"
    names = [j["name"] for j in facts["movable"]]
    if len(set(names)) < len(names):
        return False, "two joints share a prim name"
    if len(facts["bases"]) != 1:
        return False, f"base link ambiguous: {facts['bases']}"
    if not facts["colliders"]:
        return False, "no collision geometry"
    if max(scaled) > TABLESIDE_MAX_M:
        return False, f"{max(scaled):.2f} m at real size ({kind}): large, out of reach next to the table"
    return True, f"{kind}, {len(facts['movable'])} joint(s)"


def select(source):
    source = Path(source)
    chosen, dropped = [], []
    for category in CATEGORIES:
        for model in sorted((source / category).glob("*/Object.usd"), key=lambda p: int(p.parent.name)):
            rel = model.parent.relative_to(source).as_posix()
            try:
                size, facts = survey(model, category)
                kind = kind_of(category, size, model.parent.name)
                scale = real_scale(kind, size)
                scaled = [round(v * scale, 4) for v in size]
                ok, reason = decide(kind, scaled, facts)
            except Exception as exc:
                dropped.append({"src": rel, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {"src": rel, "category": category, "kind": kind, "scale": round(scale, 5), "size_m": scaled}
            (chosen if ok else dropped).append({**entry, "reason": reason})
    counters = {}
    for entry in chosen:
        cat = "partnet_" + articulated.slug(entry["category"])
        entry["id"] = f"Articulation/{cat}/{counters.setdefault(cat, 0):05d}"
        counters[cat] += 1
    return {"source": URL, "revision": REVISION, "license": LICENSE, "chosen": chosen, "dropped": dropped}


def convert(root, manifest):
    root = Path(root)
    done, failed = [], []
    for entry in manifest["chosen"]:
        target = root / "Assets/Object/RoboDojo" / entry["id"]
        src = root / "sources/partnet_mobility" / entry["src"]
        mass_key = "StorageFurniture" if entry["category"] == "StorageFurniture" else entry["kind"]
        try:
            facts = articulated.write_layer(
                src / "Object.usd",
                target,
                MASS_KG[mass_key],
                root,
                scale=entry["scale"],
                prim_path=f"/Object/{entry['category']}",
                limits_in_root_units=True,
                replace_masses=True,
                link_roles=link_roles(src / "mobility.urdf"),
                yaw_deg=FRONT_YAW_DEG,
            )
            axis, metres, basis = REAL_SIZE[entry["kind"]]
            source = {
                "dataset": REPO,
                "url": URL,
                "revision": REVISION,
                "object": entry["src"],
                "license": LICENSE,
                "terms": TERMS,
                "attribution": ATTRIBUTION,
                "rsi_pools": RSI,
                "scale": {
                    "factor_on_converted_usd": entry["scale"],
                    "rule": f"{axis} set to {metres} m",
                    "basis": f"estimated: {basis}; PartNet models are normalised, not real size",
                },
            }
            meta = articulated.metadata(
                entry["id"], facts, MASS_KG[mass_key], f"estimated: typical {mass_key} weight", source
            )
            meta["dimension_policy"] = "estimated: scaled to a real size of its kind"
            meta["size_check"] = {
                "result": "pass",
                "note": f"{axis} {metres} m by construction ({basis}); other sides keep the model's proportions",
            }
            articulated.write_json(target / "metadata.json", meta)
            articulated.write_json(
                target / "description.json",
                {
                    "uuid": meta["uuid"],
                    "caption": {"name": [f"{entry['category']} {Path(entry['src']).name}"]},
                    "description": [f"A PartNet-Mobility {entry['category'].lower()} ({entry['src']})."],
                },
            )
            done.append({"id": entry["id"], "src": entry["src"], "size_m": meta["physics"]["size"]})
        except Exception as exc:
            failed.append({"id": entry["id"], "src": entry["src"], "error": f"{type(exc).__name__}: {exc}"})
    return {"revision": REVISION, "converted": done, "failed": failed}


# names: colour from the render (median of the pixels away from the floor), nearest of these
COLOURS = {
    "white": ((235, 235, 235), "白色"),
    "light grey": ((185, 185, 185), "浅灰色"),
    "grey": ((125, 125, 125), "灰色"),
    "black": ((35, 35, 35), "黑色"),
    "brown": ((110, 70, 40), "棕色"),
    "wooden": ((175, 125, 75), "木色"),
    "beige": ((215, 195, 160), "米色"),
}
KIND_NAMES = {  # (category en, zh), (first name en, zh)
    "Microwave": (("appliance", "电器"), ("microwave", "微波炉")),
    "Oven/countertop": (("appliance", "电器"), ("oven", "烤箱")),
    "StorageFurniture/low": (("cabinet", "柜子"), ("storage cabinet", "储物柜")),
}
COUNT_EN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
COUNT_ZH = {1: "单", 2: "双", 3: "三", 4: "四", 5: "五", 6: "六"}


def colour_of(png):
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(png).convert("RGB")).reshape(-1, 3).astype(float)
    floor = np.median(rgb, axis=0)
    obj = rgb[np.abs(rgb - floor).sum(1) > 60]
    if len(obj) < 100:
        obj = rgb
    c = np.median(obj, axis=0)
    return min(COLOURS, key=lambda k: float(((np.array(COLOURS[k][0]) - c) ** 2).sum()))


def openings(joints):
    """The doors and drawers a name counts, from metadata body_joints: joints with role door / drawer, one each (a
    door's release button is role button and no door). (" with two doors and three drawers", "双门、3个抽屉的")."""
    roles = [j["role"] for j in joints]
    bits_en, bits_zh = [], []
    for role, en, zh in (("door", "door", "门"), ("drawer", "drawer", "抽屉")):
        n = roles.count(role)
        if n:
            bits_en.append(f"{COUNT_EN.get(n, n)} {en}{'s' if n > 1 else ''}")
            bits_zh.append(f"{COUNT_ZH.get(n, n)}{zh}" if role == "door" else f"{n}个{zh}")
    return (f" with {' and '.join(bits_en)}" if bits_en else "", "、".join(bits_zh) + "的" if bits_zh else "")


def details(kind, colour, size_m, model, joints):
    """Candidate most detailed names (en, zh), shortest first: size, colour, kind and openings; PartNet has
    near-duplicates, so a clash takes the next (height, depth, last of all the model number)."""
    kind_en, kind_zh = KIND_NAMES[kind][1]
    c_en, c_zh = colour, COLOURS[colour][1]
    tail_en, head_zh = openings(joints)
    w, d, h = (round(v * 100) for v in size_m)
    sizes = (
        (f"{w} cm wide", f"{w}厘米宽"),
        (f"{w} by {h} cm", f"宽{w}高{h}厘米"),
        (f"{w} by {d} by {h} cm", f"宽{w}深{d}高{h}厘米"),
        (f"model {model} {w} by {d} by {h} cm", f"型号{model}的宽{w}深{d}高{h}厘米"),
    )
    return [(f"{s_en} {c_en} {kind_en}{tail_en}", f"{head_zh}{s_zh}{c_zh}{kind_zh}") for s_en, s_zh in sizes]


def names(root, manifest, previews, keep=None):
    """object_names.json entries: category, the kind, colour + the kind, and colour, doors / drawers and size in cm,
    which tells every model apart. keep: the current table; an entry whose detailed name is still one of the object's
    candidates stays (so fixing one object does not rename the next one it used to clash with)."""
    keep, rows = keep or {}, []
    for entry in manifest["chosen"]:
        meta = json.loads((Path(root) / "Assets/Object/RoboDojo" / entry["id"] / "metadata.json").read_text())
        png = Path(previews) / entry["id"] / "preview" / "sim.png"
        if png.is_file():
            colour = colour_of(png)
            size, model = meta["physics"]["size"], Path(entry["src"]).name
            rows.append((entry, colour, details(entry["kind"], colour, size, model, meta["body_joints"])))
    chosen, seen = {}, set()
    for entry, _, cands in rows:
        old = keep.get(entry["id"])
        pair = old and (old["en"][-1], old["zh"][-1])
        if pair in cands and not seen & set(pair):
            chosen[entry["id"]] = pair
            seen |= set(pair)
    out = {}
    for entry, colour, cands in rows:
        if entry["id"] not in chosen:
            chosen[entry["id"]] = next((c for c in cands if not seen & set(c)), cands[-1])
            seen |= set(chosen[entry["id"]])
        (cat_en, cat_zh), (kind_en, kind_zh) = KIND_NAMES[entry["kind"]]
        out[entry["id"]] = {
            "category": {"en": cat_en, "zh": cat_zh},
            "en": [kind_en, f"{colour} {kind_en}", chosen[entry["id"]][0]],
            "zh": [kind_zh, f"{COLOURS[colour][1]}{kind_zh}", chosen[entry["id"]][1]],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("extract", "select", "convert", "names"))
    parser.add_argument("--previews", type=Path, help="names: <dir>/<id>/preview/sim.png from check_articulation")
    parser.add_argument("--tars", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--keep", type=Path, help="names: the current object_names.json, its still-right names stay")
    args = parser.parse_args()
    if args.command == "extract":
        print(f"{extract(args.tars, args.out)} files -> {args.out}")
    elif args.command == "select":
        result = select(args.source)
        args.manifest.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        print(f"chosen={len(result['chosen'])}, dropped={len(result['dropped'])}")
    elif args.command == "names":
        keep = json.loads(args.keep.read_text()) if args.keep else None
        result = names(args.root, json.loads(args.manifest.read_text()), args.previews, keep)
        print(json.dumps(result, ensure_ascii=False))
    else:
        result = convert(args.root, json.loads(args.manifest.read_text()))
        out = args.root / "partnet_conversion.json"
        out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"converted={len(result['converted'])}, failed={len(result['failed'])} -> {out}")


if __name__ == "__main__":
    main()
