"""Release notes of a g1d_place_anything / g1d_put_anything_in dataset: a README section listing where the objects come
from and under which license, the objects to pick (X, targets), the receptacles (Y) and the distractors apart, read
from the dataset's collection_quality.json (each episode's objects); X and distractor libraries as in
pick_anything_v2_index.source, Y's from the receptacle pool's own library / license fields. The section replaces an
earlier one and README.md's sha256 in sha256.json is updated.

    python scripts/datagen/place_anything_release.py readme --dataset <dir> \
        --receptacles scripts/objects/place_anything_receptacles_r2.json [--new-in r2]
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.pick_anything_v2_index import NONCOMMERCIAL, source  # noqa: E402

HEAD = "## Object sources and licenses"


def receptacle_sources(path):
    """{(cat, idx): (library, license, added_in)} of a receptacle pool file."""
    pool = json.loads(Path(path).read_text())["receptacles"]
    return {(r["cat"], r["idx"]): (r["library"], r["license"], r.get("added_in", "r1")) for r in pool}


def tally(qualities, recs):
    """{role: {(library, license): [objects, episodes]}} and Y episodes per round the receptacle joined the pool."""
    seen = defaultdict(lambda: defaultdict(set))
    eps = defaultdict(lambda: defaultdict(int))
    added = defaultdict(int)
    for q in qualities:
        for label, (cat, idx) in q["objects"].items():
            if label == "receptacle":
                lib, lic, joined = recs[(cat, idx)]
                role, added[joined] = "receptacle", added[joined] + 1
            else:
                role = "target" if label == "target" else "distractor"
                lib, lic = source(cat)
            seen[role][(lib, lic)].add((cat, idx))
            eps[role][(lib, lic)] += 1
    return {r: {k: [len(v), eps[r][k]] for k, v in seen[r].items()} for r in seen}, dict(added)


def section(counts, added, episodes, new_in=None):
    lines = [HEAD, ""]
    titles = dict(target="Objects to pick (X, the target)", receptacle="Receptacles (Y)", distractor="Distractors")
    for role in ("target", "receptacle", "distractor"):
        lines += [f"{titles[role]}:", "", "| Library | License | Objects | Episodes |", "| :-- | :-- | --: | --: |"]
        for (lib, lic), (n, e) in sorted(counts.get(role, {}).items(), key=lambda kv: -kv[1][1]):
            lines.append(f"| {lib} | {lic} | {n} | {e} |")
        lines.append("")
    if new_in:
        lines += [f"Receptacles added in {new_in}: {added.get(new_in, 0)} of the {episodes} episodes.", ""]
    nc = sorted({lic for role in counts.values() for _, lic in role if lic in NONCOMMERCIAL})
    lines.append(
        f"Non-commercial licenses on the table: {', '.join(nc)}."
        if nc
        else "No object on the table has a non-commercial license; licenses as each source records them "
        "(CC-BY-4.0 asks for attribution)."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("readme",))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--receptacles", type=Path, required=True)
    parser.add_argument("--new-in", help="round whose new receptacles to count (r2)")
    args = parser.parse_args()
    qualities = json.loads((args.dataset / "collection_quality.json").read_text())
    counts, added = tally(qualities, receptacle_sources(args.receptacles))
    readme = args.dataset / "README.md"
    text = readme.read_text()
    text = text.split(f"\n{HEAD}\n")[0].rstrip("\n") + "\n\n" + section(counts, added, len(qualities), args.new_in)
    readme.write_text(text)
    sums_path = args.dataset / "sha256.json"
    sums = json.loads(sums_path.read_text())
    sums["README.md"] = hashlib.sha256(readme.read_bytes()).hexdigest()
    sums_path.write_text(json.dumps(dict(sorted(sums.items())), indent=2) + "\n")
    print(section(counts, added, len(qualities), args.new_in))


if __name__ == "__main__":
    main()
