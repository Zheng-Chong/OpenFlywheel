"""g1d_pick_anything (按名称拿起任意物体): the object pool, the layouts and the instruction names.

The pool is every named rigid object of the shared asset library the Dex1-1 jaw can take from the top (POOL_RULES),
at most GROUP_CAP of one asset category with one simplest name (so 48 colours of each FMB peg do not crowd out
everything else), written to POOL (pick_anything_pool.json, committed: the layouts must not change with the library).
An object's family is its description.json names.category (cup, bottle, toy, ...).
A layout puts one pool object (label target) and DISTRACTORS other pool objects (distractor_<j>) on the table; layout
i of every seed folder targets POOL_ORDER[i mod len(pool)], so a folder of len(pool) layouts covers each object once.
With probability SAME_FAMILY one or two distractors are of the target's family (two cups, two bottles), so only the
target's more detailed names (colour, material, shape) tell it apart. Instructions use the names of the target that no
other object on the table has and that are not part of another's name (usable_names).

    python scripts/objects/pick_anything.py pool --assets Assets    # rewrite POOL from the library
"""

import argparse
import json
import math
from pathlib import Path
import random

POOL = Path(__file__).with_name("pick_anything_pool.json")
LIBRARIES = ("RoboDojo", "Real2Sim", "FMB", "Blocks")
# categories of the libraries the platform's registrations mark noncommercial (geniesim CC-BY-NC-SA-4.0,
# partnet_mobility)
NONCOMMERCIAL = ("geniesim_", "partnet_")
# metres: the narrow horizontal side within the jaw (95 mm open, opened GRASP_CLEAR 25 mm wider than the side,
# scripted_demo.pad_open) and above the pads' 15 mm; height between a flat piece the fingertips catch and what tips
# over; mass below RigidObject's 0.5 kg cap
POOL_RULES = dict(width=(0.015, 0.068), height=(0.02, 0.20), long=0.22, mass=0.5, slender=6.0)
GROUP_CAP = 4
DISTRACTORS = (3, 5)  # objects besides the target
SAME_FAMILY = 0.5
# m, env frame of the g1d office: the target where the hands grip well (g1d_general_pickup's layouts, y -0.48 ..
# -0.33, gave 100% expert success; blocks further out than |x| 0.19 or y -0.37 were gripped at the edge of reach,
# build_asset.SCATTER_X/Y), the distractors around it. |x| at least TARGET_MID_GAP and y above -0.47: within 7 cm
# of the midline and nearer than -0.46 the expert gripped 44% and 47% against 72-80% elsewhere (<gpu-server>, 11-11, 67)
TARGET_X, TARGET_Y, TARGET_MID_GAP = (-0.19, 0.19), (-0.47, -0.36), 0.04
SPREAD_X, SPREAD_Y = (-0.28, 0.28), (-0.53, -0.30)
TARGET_GAP, GAP = 0.04, 0.01  # m between footprint circles: round the target (the open fingers), between the others
# hand yaw (deg) the hand on the target's side grips best with at its spot, scripted_demo's top grasp (as
# block_patterns.grip_yaw, which needs numpy; the platform imports this): the target is turned so its narrow side
# lies across that yaw, within GRASP_YAW
GRIP_YAW_FAR, GRIP_YAW_NEAR, NEAR_Y, GRASP_YAW = 45.0, 15.0, -0.45, 20.0
# what layout() draws from; another round passes its own (scripts/datagen/pick_anything_layouts.py). scale: None, or
# f(rng, obj) -> the object's uniform scale
SPEC = dict(target_x=TARGET_X[1], target_y=TARGET_Y, mid_gap=TARGET_MID_GAP, grasp_yaw=GRASP_YAW,
            same_family=SAME_FAMILY, scale=None)  # fmt: skip
TEMPLATES = {
    "zh": ("拿起{}", "把{}拿起来", "请帮我拿起桌上的{}", "抓起{}", "找到{}并把它拿起来"),
    "en": ("Pick up the {}.", "Grab the {}.", "Lift the {} off the table.", "Find the {} and pick it up."),
}


def scan(assets):
    """Pool objects of the library at `assets`: [{cat, idx, family, size, mass}], one per (cat, idx) (the first
    library that has it, as env.global_configs.object_dir), sorted."""
    seen, out = set(), []
    for library in LIBRARIES:
        for meta_path in sorted((Path(assets) / "Object" / library / "Rigid").glob("*/[0-9]*/metadata.json")):
            folder = meta_path.parent
            key = (folder.parent.name, int(folder.name))
            if key in seen:
                continue
            seen.add(key)
            desc = folder / "description.json"
            if not desc.is_file() or not any((folder / f).is_file() for f in ("object.usd", "object.usdz")):
                continue
            names = json.loads(desc.read_text()).get("names") or {}
            family = (names.get("category") or {}).get("en")
            physics = json.loads(meta_path.read_text()).get("physics") or {}
            size, mass = physics.get("size"), physics.get("mass")
            if not (family and len(names.get("zh") or []) == 3 and len(names.get("en") or []) == 3 and size):
                continue
            if graspable(size, mass):
                out.append(dict(cat=key[0], idx=key[1], family=family, name=names["en"][0],
                                size=[round(float(s), 4) for s in size], mass=round(float(mass), 4)))  # fmt: skip
    return out


def graspable(size, mass):
    w, d, h = size
    r = POOL_RULES
    narrow = min(w, d)
    return (
        r["width"][0] <= narrow <= r["width"][1]
        and r["height"][0] <= h <= r["height"][1]
        and max(w, d) <= r["long"]
        and h <= r["slender"] * narrow
        and mass is not None
        and 0 < mass <= r["mass"]
    )


def cap_groups(objects, cap=GROUP_CAP):
    """At most `cap` objects of each group (asset category and simplest name: 48 colours of one FMB peg are one
    group, 139 different Sand3D packages are not), spread evenly over its sorted members."""
    by = {}
    for o in objects:
        by.setdefault((o["cat"], o["name"]), []).append(o)
    out = []
    for members in by.values():
        step = max(1, len(members) / cap)
        out += [members[int(i * step)] for i in range(min(cap, len(members)))]
    return sorted(out, key=lambda o: (o["cat"], o["idx"]))


def load_pool():
    return json.loads(POOL.read_text())["objects"]


def commercial(pool):
    """The pool objects of commercially usable libraries: not of a NONCOMMERCIAL one."""
    return [o for o in pool if not o["cat"].startswith(NONCOMMERCIAL)]


def pool_order(pool):
    """The target order of layout i (mod len(pool)): one fixed shuffle, so any run of layouts mixes the families."""
    order = list(range(len(pool)))
    random.Random("g1d_pick_anything").shuffle(order)
    return order


def radius(obj):
    return math.hypot(*obj["size"][:2]) / 2


def layout(rng, index, entry, pool=None, spec=SPEC):
    """Rigid section of layout `index`: {cat: [entry(cat, idx, label, xy, yaw)]}, the target and its distractors
    (entry(..., yaw, scale) when `spec` scales them)."""
    pool = pool or load_pool()
    target = pool[pool_order(pool)[index % len(pool)]]
    same = [o for o in pool if o["family"] == target["family"] and o is not target]
    count = rng.randint(*DISTRACTORS)
    picked = []
    if same and rng.random() < spec["same_family"]:
        picked += rng.sample(same, min(len(same), rng.choice((1, 2))))
    families = {target["family"]}
    for o in rng.sample(pool, len(pool)):
        if len(picked) >= count:
            break
        if o["family"] not in families:
            picked.append(o)
            families.add(o["family"])
    scales = [spec["scale"](rng, o) for o in [target, *picked]] if spec["scale"] else None
    for attempt in range(200):
        spots = place(rng, target, picked, spec, scales)
        if spots:
            break
        if attempt % 20 == 19 and len(picked) > DISTRACTORS[0]:
            picked.pop()  # large objects: one fewer
            if scales:
                scales.pop()
    else:
        raise RuntimeError(f"no room for pick_anything layout {index}")
    rigid = {}
    for j, (obj, xy) in enumerate(zip([target, *picked], spots)):
        label = "target" if j == 0 else f"distractor_{j - 1}"
        yaw = target_yaw(rng, obj, xy, spec["grasp_yaw"]) if j == 0 else rng.uniform(0.0, 360.0)
        args = (obj["cat"], obj["idx"], label, xy, yaw) + ((scales[j],) if scales else ())
        rigid.setdefault(obj["cat"], []).append(entry(*args))
    return rigid


def grip_yaw(x, y):
    return (GRIP_YAW_FAR if y > NEAR_Y else GRIP_YAW_NEAR) * (1 if x > 0 else -1)


def target_yaw(rng, obj, xy, within=GRASP_YAW):
    """deg: the object turned so its narrow horizontal side (local X or Y) lies along grip_yaw at `xy`, within
    `within`, either way round."""
    across = 0.0 if obj["size"][0] <= obj["size"][1] else -90.0
    return grip_yaw(*xy) + across + rng.uniform(-within, within) + 180.0 * rng.randrange(2)


def place(rng, target, others, spec=SPEC, scales=None):
    """Footprint centres [target, *others] within their areas (the target's from `spec`), apart by TARGET_GAP round
    the target and GAP between the others; None if one finds no room in 300 tries."""
    scales = scales or [1.0] * (1 + len(others))
    x = rng.uniform(spec["mid_gap"], spec["target_x"]) * rng.choice((-1, 1))
    spots = [(x, rng.uniform(*spec["target_y"]))]
    placed = [(spots[0], radius(target) * scales[0])]
    for obj, s in zip(others, scales[1:]):
        r = radius(obj) * s
        for _ in range(300):
            xy = (rng.uniform(*SPREAD_X), rng.uniform(*SPREAD_Y))
            gaps = [math.dist(xy, q) - r - rq for q, rq in placed]
            if gaps[0] >= TARGET_GAP and all(g >= GAP for g in gaps[1:]):
                break
        else:
            return None
        spots.append(xy)
        placed.append((xy, r))
    return spots


def usable_names(own, others):
    """Of `own` names (category first, then simple to detailed), those no other object's name equals or contains;
    else the most detailed one (unique across the library, scripts/objects/object_names.py check)."""
    theirs = [n for names in others for n in names]
    keep = [n for n in own if not any(n == t or n in t for t in theirs)]
    return keep or [own[-1]]


def instruction(rng, names):
    """One instruction: a language, one of its usable names (category first .. most detailed) and a template.
    `names`: {lang: usable names}."""
    lang = rng.choice(sorted(names))
    return rng.choice(TEMPLATES[lang]).format(rng.choice(names[lang]))


# Categories (or "<cat>/<idx:05d>" ids) later rounds leave out of the target and distractor draws, and the tasks they
# leave them out of ("*": every task drawing from this pool: pick, place, put, sort, handover; user 11-19). Accepted /
# recorded in round 1 of place | put anything (docs/place-anything-r2-supports.md「r1 的失败分布」): auto parts
# sand3d_vehicle 0/87 | 1/89, out everywhere (the car parts filed under other categories, two fuses, a relay, a shaped
# hose, an idler pulley and a chain tensioner, stay: 130/235 | 142/243); game controllers 0/34 | 12/32 and hammers
# 1/41 | 21/43, out of place only (they put fine). Rounds drawn before keep them.
EXCLUDED = {
    "sand3d_vehicle": "*",
    "game_controller": ("g1d_place_anything",),
    "hammer": ("g1d_place_anything",),
}


def excluded(task=None):
    """What EXCLUDED leaves out for `task` (None: only what every task leaves out)."""
    return {k for k, tasks in EXCLUDED.items() if tasks == "*" or task in tasks}


def usable(pool, task=None):
    """The pool without the objects EXCLUDED leaves out for `task`, by category or by id."""
    out = excluded(task)
    return [o for o in pool if o["cat"] not in out and f"{o['cat']}/{o['idx']:05d}" not in out]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="rewrite the pool from an asset library")
    p.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    found = scan(args.assets)
    pool = cap_groups(found)
    POOL.write_text(json.dumps(dict(rules=POOL_RULES, group_cap=GROUP_CAP, scanned=len(found), objects=pool),
                               indent=1, ensure_ascii=False) + "\n")  # fmt: skip
    families = len({o["family"] for o in pool})
    print(f"{len(found)} graspable named objects, {len(pool)} in the pool ({families} families)")


if __name__ == "__main__":
    main()
