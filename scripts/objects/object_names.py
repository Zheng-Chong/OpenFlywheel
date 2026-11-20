"""Instruction names of the Real2Sim objects: object_names.json -> description.json "names".

Every object and every generalization variant has a category name ("cup" / "杯子", what it generalizes to) and three
names in English and Chinese, brief to detailed: the object's own kind ("metal cup"), that plus what tells it apart in
its family ("tapered aluminium cup"), and a phrase that singles it out among all objects ("tapered aluminium tumbler
with a rolled rim"). An instruction template fills `<label>` with the category or one of the names
(env/description_manager/desc_manager.py), in the template's language. object_names.json maps an asset id to
{"category": {"en": ..., "zh": ...}, "en": [3], "zh": [3]}, or to another id with the same names (the twist-cap
Articulation bottle of a Clutter bottle, the deformable twin of a rigid cup).

Every object in the asset library has names, the official RoboDojo, NVIDIA SimReady and AgiBot GenieSim ones included;
`missing` lists the object dirs the table does not cover. Generators rewrite description.json without "names": run
`apply` after every generation.

Usage:
    python scripts/objects/object_names.py check
    python scripts/objects/object_names.py apply OBJECT_ROOT...    # e.g. Assets/Object/Real2Sim Assets/Object/RoboDojo
    python scripts/objects/object_names.py missing OBJECT_ROOT...  # exit 1 when an object has no names
"""

import argparse
import json
from pathlib import Path
import re

TABLE = Path(__file__).with_name("object_names.json")
LANGS = ("en", "zh")
LEVELS = 3
# a name drops into "pick up the <x>" / "拿起<x>": no article, measure word, final punctuation or place in the scene
BAD = {
    "en": re.compile(r"^(a|an|the)\b|[.!?]$|\b(left|right|front|back|bigger|smaller|larger)\b", re.I),
    "zh": re.compile(r"^(一|这|那)|[。！？]$|左|右|更"),
}


def load(path=TABLE):
    """{asset id: {"category": {...}, "en": [...], "zh": [...]}} with aliases resolved."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {oid: raw[v] if isinstance(v, str) else v for oid, v in raw.items()}


def length(name, lang):
    return len(name.split()) if lang == "en" else len(name.replace(" ", ""))


def bare(name, lang):
    return isinstance(name, str) and name.strip() == name and bool(name) and not BAD[lang].search(name)


def problems(table):
    """What breaks the conventions above; empty when the table is fine."""
    found, last, seen = [], {lang: {} for lang in LANGS}, set()
    for oid, names in table.items():
        if id(names) in seen:  # an alias
            continue
        seen.add(id(names))
        if set(names) != {"category", *LANGS}:
            found.append(f"{oid}: keys {sorted(names)}, want category, {', '.join(LANGS)}")
            continue
        for lang in LANGS:
            category, items = (names["category"] or {}).get(lang), names[lang]
            if len(items) != LEVELS or len(set(items)) != LEVELS:
                found.append(f"{oid} {lang}: want {LEVELS} distinct names, got {items}")
            for name in [category, *items]:
                if not bare(name, lang):
                    found.append(f"{oid} {lang}: {name!r} is not a bare noun phrase")
            if category in items:
                found.append(f"{oid} {lang}: category {category!r} repeats a name")
            sizes = [length(n, lang) for n in items if isinstance(n, str)]
            if sizes != sorted(sizes) or (bare(category, lang) and sizes and length(category, lang) > sizes[0]):
                found.append(f"{oid} {lang}: not ordered brief to detailed {[category, *items]}")
            if items:
                last[lang].setdefault(items[-1], []).append(oid)
    for lang in LANGS:
        shared = [(name, ids) for name, ids in last[lang].items() if len(ids) > 1]
        found += [f"{lang} {name!r} is the most detailed name of {ids}" for name, ids in shared]
    return found


def apply(root, table):
    """Write `names` into root/<id>/description.json of every object in the table found under root (an object
    without a description.json gets one holding only its names)."""
    done = []
    for oid, names in table.items():
        path = Path(root) / oid / "description.json"
        if path.parent.is_dir():
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if data.get("names") == names:
                continue
            data["names"] = names
            path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
            done.append(oid)
    return done


def missing(root, table):
    """Object dirs root/<type>/<category>/<index>/ the table has no names for."""
    dirs = (p for p in Path(root).glob("*/*/*") if p.is_dir() and p.name.isdigit())
    return sorted(oid for oid in (p.relative_to(root).as_posix() for p in dirs) if oid not in table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "apply", "missing"))
    parser.add_argument("roots", nargs="*", help="object roots holding <type>/<category>/<index>/ (apply, missing)")
    args = parser.parse_args()
    table = load()
    found = problems(table)
    if found:
        raise SystemExit("\n".join(found))
    if args.command != "check" and not args.roots:
        parser.error(f"{args.command} needs OBJECT_ROOT")
    if args.command == "apply":
        for root in args.roots:
            done = apply(root, table)
            print(f"names written for {len(done)} objects under {root}")
    elif args.command == "missing":
        gaps = [f"{root}/{oid}" for root in args.roots for oid in missing(root, table)]
        print("\n".join(gaps) if gaps else f"every object under {' '.join(args.roots)} has names")
        raise SystemExit(1 if gaps else 0)
    else:
        print(f"{len(table)} objects ok")


if __name__ == "__main__":
    main()
