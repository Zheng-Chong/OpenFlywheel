"""Sharpa Wave 61D wholebody X-Anything (pick / place / put_in / handover): commercial object pool screening,
grasp-primitive classification, and layout hooks.

Screens 80-120 commercially usable rigid objects from `scripts/objects/pick_anything_pool.json` by geometry and mass
(protecting G1-D's 5 N*m wrist torque limit with the 1.31 kg Sharpa Wave hand) and tags each object with its default
grasp primitive:
    power_wrap    tall cylindrical/prismatic objects (h >= 72 mm, narrow >= 30 mm, long <= 140 mm)
    tripod_pinch  compact objects (h 36..72 mm, narrow 24..68 mm, long <= 120 mm)
    lateral       flat/elongated objects (h 24..60 mm, long >= 60 mm, narrow <= 60 mm)

Stdlib only (safe to import from platform / layout generators).

    python scripts/objects/sharpa_x_anything.py pool    # rewrite sharpa_x_anything_pool.json
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from scripts.objects import handover_anything, pick_anything, place_anything

POOL = Path(__file__).with_name("sharpa_x_anything_pool.json")
PRIMITIVES = ("power_wrap", "tripod_pinch", "lateral")

# Geometry & mass bounds for 1.31 kg Sharpa Wave on G1-D (5 N*m wrist torque limit, 5-finger side/angled grasp)
SHARPA_RULES = dict(
    mass=(0.015, 0.40),
    narrow=(0.022, 0.068),
    long_max=0.19,
    height=(0.024, 0.195),
    power_wrap=dict(height_min=0.072, narrow_min=0.030, long_max=0.14),
    tripod_pinch=dict(height_min=0.036, narrow_min=0.024, long_max=0.12),
    lateral=dict(height_min=0.024, height_max=0.060, long_min=0.060, narrow_max=0.060),
)
DEFAULT_FAMILY_CAP = 6
DEFAULT_PRIM_CAPS = {"power_wrap": 52, "tripod_pinch": 36, "lateral": 24}

# Sharpa-friendly layout overrides (slightly wider clearance for 5-finger approach)
SHARPA_PICK_SPEC = dict(
    pick_anything.SPEC,
    target_x=0.19,
    target_y=(-0.47, -0.35),
    mid_gap=0.05,
    grasp_yaw=20.0,
)
SHARPA_PLACE_SPEC = dict(
    place_anything.SPEC,
    target_x=0.20,
    target_y=(-0.47, -0.35),
    mid_gap=0.05,
    receptacle_x=0.05,
    receptacle_y=(-0.44, -0.39),
)
SHARPA_HANDOVER_SPEC = dict(
    handover_anything.SPEC_L1,
    x_x=(0.10, 0.20),
    x_y=(-0.47, -0.35),
    y_x=(0.11, 0.19),
    y_y=(-0.46, -0.39),
)


def select_primitive(size, mass=None):
    """Return 'power_wrap', 'tripod_pinch', 'lateral', or None if outside Sharpa safe bounds."""
    if size is None or len(size) != 3:
        return None
    w, d, h = (float(size[0]), float(size[1]), float(size[2]))
    narrow, long_s = min(w, d), max(w, d)
    r = SHARPA_RULES
    if mass is not None:
        m = float(mass)
        if not (r["mass"][0] <= m <= r["mass"][1]):
            return None
    if not (
        r["narrow"][0] <= narrow <= r["narrow"][1] and long_s <= r["long_max"] and r["height"][0] <= h <= r["height"][1]
    ):
        return None
    pw = r["power_wrap"]
    if h >= pw["height_min"] and narrow >= pw["narrow_min"] and long_s <= pw["long_max"]:
        return "power_wrap"
    tp = r["tripod_pinch"]
    if h >= tp["height_min"] and narrow >= tp["narrow_min"] and long_s <= tp["long_max"]:
        return "tripod_pinch"
    lat = r["lateral"]
    if lat["height_min"] <= h < lat["height_max"] and long_s >= lat["long_min"] and narrow <= lat["narrow_max"]:
        return "lateral"
    return None


def screen_pool(pick_pool=None, family_cap=DEFAULT_FAMILY_CAP, prim_caps=None):
    """Screen a balanced, commercially usable 80-120 object subset from the X-Anything commercial pool."""
    base = pick_anything.load_pool() if pick_pool is None else list(pick_pool)
    comm = pick_anything.commercial(base)
    caps = dict(DEFAULT_PRIM_CAPS if prim_caps is None else prim_caps)

    by_pf = {}
    for obj in comm:
        prim = select_primitive(obj.get("size"), obj.get("mass"))
        if prim is None:
            continue
        tagged = dict(obj, primitive=prim)
        by_pf.setdefault((prim, obj["family"]), []).append(tagged)

    per_prim = {p: [] for p in PRIMITIVES}
    for (prim, _family), members in sorted(by_pf.items()):
        members = sorted(members, key=lambda x: (x["cat"], x["idx"]))
        step = max(1.0, len(members) / family_cap)
        per_prim[prim].extend(members[int(i * step)] for i in range(min(family_cap, len(members))))

    selected = []
    for prim in PRIMITIVES:
        members = sorted(per_prim[prim], key=lambda x: (x["family"], x["cat"], x["idx"]))
        cap = int(caps.get(prim, len(members)))
        if cap <= 0:
            continue
        step = max(1.0, len(members) / cap)
        selected.extend(members[int(i * step)] for i in range(min(cap, len(members))))

    return sorted(selected, key=lambda x: (x["cat"], x["idx"]))


def load_pool(path=POOL):
    """Load the committed Sharpa commercial X-Anything pool."""
    return json.loads(Path(path).read_text())["objects"]


def handover_pool(pool=None):
    """Subset of the Sharpa commercial pool suitable for bimanual handover (long >= 0.09 m or height >= 0.095 m)."""
    items = load_pool() if pool is None else list(pool)
    out = []
    for obj in items:
        w, d, h = obj["size"]
        if max(w, d) >= 0.09 or h >= 0.095:
            out.append(dict(obj, shift=round(handover_anything.shift(obj["size"]), 4)))
    return out


def pool_order(pool=None):
    """Deterministic target shuffle for Sharpa X-Anything layouts."""
    items = load_pool() if pool is None else list(pool)
    order = list(range(len(items)))
    random.Random("g1d_sharpa_x_anything").shuffle(order)
    return order


def _default_entry(cat, idx, label, xy, yaw, scale=1.0):
    return {
        "category": cat,
        "category_idx": int(idx),
        "label": label,
        "default_pos": [round(float(xy[0]), 5), round(float(xy[1]), 5), 0.0],
        "yaw_deg": round(float(yaw), 3),
        "scale": float(scale),
    }


def _default_receptacle_entry(rec, label, xy, yaw, kind=None):
    k = kind if kind is not None else next(iter(rec["geometry"]))
    geom = dict(rec["geometry"][k], kind=k, id=rec["id"])
    return {
        "category": rec["cat"],
        "category_idx": int(rec["idx"]),
        "label": label,
        "default_pos": [round(float(xy[0]), 5), round(float(xy[1]), 5), 0.0],
        "yaw_deg": round(float(yaw), 3),
        "receptacle": geom,
    }


def pick_layout(rng, index, entry=_default_entry, pool=None, spec=SHARPA_PICK_SPEC):
    """Generate a g1d_pick_anything Rigid layout using the Sharpa commercial pool."""
    items = load_pool() if pool is None else list(pool)
    return pick_anything.layout(rng, index, entry, pool=items, spec=spec)


def place_layout(
    rng,
    index,
    entry=_default_entry,
    receptacle_entry=None,
    kind="support",
    pool=None,
    receptacles=None,
    spec=SHARPA_PLACE_SPEC,
):
    """Generate a g1d_place_anything (support) or g1d_put_anything_in (container) layout with the Sharpa pool."""
    items = load_pool() if pool is None else list(pool)
    recs = place_anything.load_receptacles() if receptacles is None else list(receptacles)
    rec_fn = (
        (lambda rec, label, xy, yaw: _default_receptacle_entry(rec, label, xy, yaw, kind=kind))
        if receptacle_entry is None
        else receptacle_entry
    )
    return place_anything.layout(rng, index, entry, rec_fn, kind, items, recs, spec=spec)


def handover_layout(
    rng,
    index,
    entry=_default_entry,
    receptacle_entry=None,
    pool=None,
    receptacles=None,
    spec=SHARPA_HANDOVER_SPEC,
):
    """Generate a g1d_handover_anything layout with the Sharpa handover-compatible commercial pool."""
    all_items = load_pool() if pool is None else list(pool)
    ho_items = handover_pool(all_items)
    recs = place_anything.load_receptacles() if receptacles is None else list(receptacles)
    rec_fn = (
        (lambda rec, label, xy, yaw: _default_receptacle_entry(rec, label, xy, yaw, kind=handover_anything.KIND))
        if receptacle_entry is None
        else receptacle_entry
    )
    return handover_anything.layout(rng, index, entry, rec_fn, ho_items, all_items, recs, spec=spec)


def write_pool(path=POOL):
    objects = screen_pool()
    counts = dict(Counter(o["primitive"] for o in objects))
    payload = {
        "about": "Commercially usable Sharpa Wave X-Anything object pool screened from pick_anything_pool.json",
        "rules": SHARPA_RULES,
        "count": len(objects),
        "primitive_counts": counts,
        "objects": objects,
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Sharpa Wave X-Anything commercial object pool tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool", help="Rewrite sharpa_x_anything_pool.json from pick_anything_pool.json")
    args = parser.parse_args()
    if args.cmd == "pool":
        payload = write_pool()
        print(f"Wrote {payload['count']} objects to {POOL}: {payload['primitive_counts']}")


if __name__ == "__main__":
    main()
