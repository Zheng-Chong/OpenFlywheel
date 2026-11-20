"""Feasibility of g1d_sort_anything runs: accepted / recorded and items not picked up (sidecar `picks`, one entry per
pick try: an item whose last try did not lift it, of the items the expert went for) by scene, by objects x
receptacles, by sort class, X category and receptacle; the last frame's failing condition (final_fail) and how many
objects were sorted when not accepted. Reads the episode sidecars (episode_*.json) under the run_batch outputs given.
With --vs, an A/B: both sides cut to the layouts recorded in both (layout_sha256) and shown side by side.

    python3 scripts/datagen/sort_anything_stats.py <run_batch output> [...] [--vs <run_batch output> [...]]
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load(outputs):
    eps = {}
    for out in outputs:
        for p in sorted(out.rglob("episode_*.json")):
            e = json.loads(p.read_text())
            eps[e.get("layout_sha256") or str(p)] = e
    return eps


def tables(eps):
    """{table: {value: [accepted, episodes, items not picked up, items gone for]}}, final_fail and partial Counters."""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    fails, partial = Counter(), Counter()
    for e in eps:
        ok = bool(e.get("accepted"))
        rec = e.get("receptacles") or {}
        objects = e.get("objects") or {}
        classes = e.get("classes") or {}
        last = {}
        for p in e.get("picks") or []:  # an item's last try decides
            last[p["item"]] = p
        keys = {
            "all": ["all"],
            "scene": [e.get("config_name") or e.get("env_cfg")],
            "objects x receptacles": [f"{e['n_items']}x{len(rec)}"],
            "class": sorted(set(classes.values())),
            "X category": sorted({objects.get(x, ["?"])[0] for x in classes}),
            "receptacle": sorted({f"{objects.get(y, ['?'])[0]} ({g.get('kind')})" for y, g in rec.items()}),
        }
        for table, values in keys.items():
            for v in values:
                row = out[table][v]
                row[0] += ok
                row[1] += 1
                mine = [
                    p
                    for x, p in last.items()
                    if (table != "class" or classes.get(x) == v)
                    and (table != "X category" or objects.get(x, ["?"])[0] == v)
                ]
                row[2] += sum(not p["lifted"] for p in mine)
                row[3] += len(mine)
        if not ok:
            fails[e.get("final_fail")] += 1
            partial[f"{e.get('sorted_max', 0)}/{e['n_items']}"] += 1
    return out, fails, partial


def cell(row):
    ok, n, miss, picks = row  # picks: items gone for
    text = f"{ok}/{n} ({ok / max(n, 1):.0%})"
    return text + (f", not picked up {miss}/{picks} ({miss / picks:.0%})" if picks else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--vs", nargs="+", type=Path, default=[], help="B side of an A/B on the same layouts")
    args = parser.parse_args(argv)
    a = load(args.outputs)
    sides = [a]
    if args.vs:
        b = load(args.vs)
        common = sorted(set(a) & set(b))
        sides = [{k: a[k] for k in common}, {k: b[k] for k in common}]
        print(f"A/B on {len(common)} layouts recorded on both sides (A {len(a)}, B {len(b)})")
    stats = [tables(list(s.values())) for s in sides]
    for table in stats[0][0]:
        print(f"\n{table}:")
        rows = stats[0][0][table]
        for v in sorted(rows, key=lambda v: -rows[v][1]):
            print(f"  {v}: " + "  |  ".join(cell(st[0][table][v]) for st in stats))
    for name, (_, fails, partial) in zip("AB", stats):
        print(f"\n{name} final_fail of the rejected:", dict(fails.most_common()))
        print(f"{name} most objects sorted at once in the rejected:", dict(sorted(partial.items())))


if __name__ == "__main__":
    main()
