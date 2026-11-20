"""g1d_handover_anything (把 X 交给另一只手再放到 Y 上): the objects X that two hands can hold at once, the layouts'
Rigid section, the instructions and the acceptance geometry. X comes from g1d_pick_anything's pool (its names, grip
yaw and the commercial rule of g1d_place_anything); Y is a support of g1d_place_anything's pool
(scripts/objects/place_anything.py). Stdlib only (the platform imports it).

The handover is lengthwise: X lies with a long horizontal side, the hand on its side (the giver) takes it from the
top EXTENT_SHIFT along that side off its centre, the way its fingers do not point, carries it level in front of the
chest with its fingers pointing at the other hand, which takes the other end from the top with its fingers pointing
back, so the two grips are 2 * shift apart and the hands lean away from each other. The taker sets it on Y on its own
side of the desk. A top grasp closes across X's narrow side, so the fingers always point along its long side.

    python scripts/objects/handover_anything.py pool    # rewrite POOL from pick_anything_pool.json
    python scripts/objects/handover_anything.py screen --sidecars <run>/*/raw/seed*/episode_*.json [--mark]
    python scripts/objects/handover_anything.py screen --stats r1.json --mark --least 50 --floor 0.1 --label "round 1"
"""

import argparse
import json
import math
from pathlib import Path
import random

from scripts.objects import pick_anything, place_anything as pa

POOL = Path(__file__).with_name("handover_anything_pool.json")
TASK = "g1d_handover_anything"
KIND = "support"
HANDS = ("left", "right")  # env -x, +x: the robot faces +y
# X: the long horizontal side at least LONG (two grips SHIFT_MIN off the centre, each END in from its end, at least
# GRIP_GAP apart) and at most MASS (the off-centre grip twists it in the pads)
RULES = dict(long=0.10, mass=0.45)
SHIFT_MIN, END, GRIP_GAP = 0.035, 0.015, 0.07
# not handed over from round 2 on (round 1's layouts keep them): what pick_anything.EXCLUDED leaves out for this task
# (auto parts sand3d_vehicle; game controllers and hammers are out of place only). POOL's screened_out marks (the
# screening's, and earlier exclusions) stay on top of it: load_pool drops every marked object
# m, deg, env frame of the g1d desk: X on the giver's side, Y on the taker's (both away from the midline the hands
# meet over); distractors off the middle band and no taller than DISTRACTOR_TOP (X is carried over them)
SPEC = dict(
    x_x=(0.08, 0.24), x_y=(-0.50, -0.32), grasp_yaw=30.0, y_x=(0.10, 0.22), y_y=(-0.50, -0.36),
    same_family=1.0, other_receptacle=0.5, scale=None,
)  # fmt: skip
SPEC_L1 = dict(SPEC, x_x=(0.10, 0.20), x_y=(-0.47, -0.36), grasp_yaw=20.0, y_x=(0.12, 0.18), y_y=(-0.46, -0.40))
SPEC_L1.update(same_family=pick_anything.SAME_FAMILY, other_receptacle=0.0)
DISTRACTORS = (1, 3)
MID_BAND, DISTRACTOR_TOP = 0.07, 0.12
TARGET_GAP, GAP = pick_anything.TARGET_GAP, pick_anything.GAP
NAMES = {"zh": {"left": "左手", "right": "右手"}, "en": {"left": "left hand", "right": "right hand"}}
TEMPLATES = {
    "zh": (
        "把{x}交给{to}，然后放到{y}上",
        "用{giver}拿起{x}，递给{to}，再放到{y}上",
        "把{x}从{giver}换到{to}，然后放在{y}上面",
        "先把{x}递到{to}，再用{to}把它放到{y}上",
    ),
    "en": (
        "Hand the {x} over to your {to} and put it on the {y}.",
        "Pick up the {x} with your {giver}, pass it to your {to}, then place it on the {y}.",
        "Pass the {x} from your {giver} to your {to} and set it on the {y}.",
        "Transfer the {x} to your {to}, then put it down on the {y}.",
    ),
}
# acceptance (scripts/datagen/plugins/g1d_handover_anything.py, the task's run_reward): a hand holds X while a pad of
# it presses HELD_FORCE (N) and it is open at least HELD_OPEN (shut on nothing: ~0). The fingers close together, so a
# jaw stopped open with one pad pressing has X between them: the right hand's top grasps from the desk often press
# X with one pad's sensor (the fingertip link) and the other finger lower down (pilot 11-19: 60 N and 0 N, X carried
# all the way); a thin handle pinched at the fingertips or a part caught by the finger roots presses neither. Both
# hold X for HANDED frames
# with X's bottom AIR above where it lay; from the lift (X LIFTED up) until it is set down X is held by a hand or
# AIR up at most LOST frames in a row; the hands' links stay CLASH apart
HELD_FORCE, HELD_OPEN, HANDED, AIR, LIFTED, LOST, CLASH = 2.0, 0.03, 5, 0.05, 0.03, 3, 0.03


def long_side(size):
    return max(size[0], size[1])


def shift(size):
    """m each grip sits off X's centre along its long side: a quarter of it, at least SHIFT_MIN, END in from the end."""
    return min(max(long_side(size) / 4, SHIFT_MIN), long_side(size) / 2 - END)


def suits(obj):
    """Can two hands hold `obj` (a pick_anything pool entry) at once, a grip near each end?"""
    return (
        long_side(obj["size"]) >= RULES["long"]
        and obj["mass"] <= RULES["mass"]
        and 2 * shift(obj["size"]) >= GRIP_GAP
        and not obj["cat"].startswith(pa.NONCOMMERCIAL)
    )


def excluded(obj):
    return not pick_anything.usable([obj], TASK)


def scan(pick_pool):
    return [dict(o, shift=round(shift(o["size"]), 4)) for o in pick_pool if suits(o)]


def load_pool(path=POOL):
    """The X pool without the objects the screening took out."""
    return [o for o in json.loads(Path(path).read_text())["objects"] if not o.get("screened_out")]


def hand_of(x):
    return HANDS[x > 0]


def other(hand):
    return HANDS[hand == "left"]


def place_yaw(obj, y_xy):
    """deg X lies at on Y: the taker sets it down with its hand at pick_anything.grip_yaw of Y's spot, X's long side
    along the fingers (scripted_demo.handover_anything_actions)."""
    long_local_x = obj["size"][0] >= obj["size"][1]
    return pick_anything.grip_yaw(*y_xy) + (90.0 if long_local_x else 0.0)


def order(pool, receptacles):
    """The X order of layout i (mod its length): a fixed shuffle of the pool objects some training support takes."""
    train = [r["geometry"][KIND] for r in pa.of_kind(receptacles, KIND)]
    keep = [i for i, o in enumerate(pool) if any(pa.fits_some_yaw(g, o["size"], KIND) for g in train)]
    random.Random(TASK).shuffle(keep)
    return keep


def layout(rng, index, entry, receptacle_entry, pool, distractor_pool, receptacles, spec=SPEC, order_=None):
    """Rigid section of layout `index` and what was drawn: X (label target, pool[order[index mod len]]) on a random
    side, a training support it fits (label receptacle) on the other, DISTRACTORS distractor_pool objects
    (distractor_<j>), one more training support among them with probability spec["other_receptacle"].
    entry(cat, idx, label, xy, yaw[, scale]) and receptacle_entry(rec, label, xy, yaw) make the entries."""
    order_ = order_ or order(pool, receptacles)
    target = pool[order_[index % len(order_)]]
    s_x = spec["scale"](rng, target) if spec["scale"] else 1.0
    size = [v * s_x for v in target["size"]]
    choices = [r for r in pa.of_kind(receptacles, KIND) if pa.fits_some_yaw(r["geometry"][KIND], size, KIND)]
    if not choices:
        s_x, size = 1.0, list(target["size"])
        choices = [r for r in pa.of_kind(receptacles, KIND) if pa.fits_some_yaw(r["geometry"][KIND], size, KIND)]
    rec = rng.choice(choices)
    giver = rng.choice(HANDS)
    family = target["family"]
    key = (target["cat"], target["idx"])
    low = [o for o in distractor_pool if o["size"][2] <= DISTRACTOR_TOP and (o["cat"], o["idx"]) != key]
    same = [o for o in low if o["family"] == family]
    count = rng.randint(*DISTRACTORS)
    picked = rng.sample(same, min(len(same), rng.choice((1, 2)))) if same and rng.random() < spec["same_family"] else []
    families = {family}
    for o in rng.sample(low, len(low)):
        if len(picked) >= count:
            break
        if o["family"] not in families:
            picked.append(o)
            families.add(o["family"])
    scales = [spec["scale"](rng, o) if spec["scale"] else 1.0 for o in picked]
    second = None
    if rng.random() < spec["other_receptacle"]:
        pool_y = [r for r in pa.of_kind(receptacles, KIND) if r["id"] != rec["id"]]
        same_y = [r for r in pool_y if r["family"] == rec["family"]]
        second = rng.choice(same_y or pool_y) if pool_y else None
    for attempt in range(400):
        sizes = [[v * s for v in o["size"]] for o, s in zip(picked, scales)]
        placed = place(rng, target, size, rec, giver, sizes, second, spec)
        if placed:
            break
        if attempt % 40 == 39:  # crowded: no second support, fewer distractors, then another support
            if second is not None:
                second = None
            elif len(picked) > DISTRACTORS[0]:
                picked.pop()
                scales.pop()
            else:
                rec = rng.choice(choices)
    else:
        raise RuntimeError(f"no room for {TASK} layout {index}")
    (x_xy, x_yaw), (y_xy, y_yaw), spots, second_spot = placed
    rigid = {}

    def add(cat, e):
        rigid.setdefault(cat, []).append(e)

    add(target["cat"], entry(target["cat"], target["idx"], "target", x_xy, x_yaw, *([s_x] if spec["scale"] else [])))
    add(rec["cat"], receptacle_entry(rec, pa.LABEL, y_xy, y_yaw))
    for j, (obj, s, (xy, yaw)) in enumerate(zip(picked, scales, spots)):
        add(obj["cat"], entry(obj["cat"], obj["idx"], f"distractor_{j}", xy, yaw, *([s] if spec["scale"] else [])))
    if second is not None:
        add(second["cat"], receptacle_entry(second, f"distractor_{len(picked)}", *second_spot))
    drawn = dict(
        receptacle=rec["id"], kind=KIND, other_receptacle=second["id"] if second else None, x_scale=s_x,
        giver=giver, taker=other(giver), shift=round(shift(size), 4),
    )  # fmt: skip
    return rigid, drawn


def place(rng, target, size, rec, giver, sizes, second, spec):
    """((X xy, yaw deg), (Y xy, yaw deg), [(xy, yaw) of the others, `sizes`], (xy, yaw) of the second support) or None:
    X on the giver's side, Y on the taker's at a yaw X fits it at as the taker sets it down."""
    side = -1.0 if giver == "left" else 1.0
    xy = (side * rng.uniform(*spec["x_x"]), rng.uniform(*spec["x_y"]))
    x_yaw = pick_anything.target_yaw(rng, target, xy, spec["grasp_yaw"])
    x_rect = (*xy, size[0] / 2, size[1] / 2, math.radians(x_yaw))
    y_xy = (-side * rng.uniform(*spec["y_x"]), rng.uniform(*spec["y_y"]))
    on_y = place_yaw(target, y_xy)
    geom = rec["geometry"][KIND]
    for _ in range(60):
        y_yaw = rng.uniform(0.0, 360.0)
        if pa.fits(geom, size, KIND, math.radians(on_y - y_yaw)):
            break
    else:
        return None
    y_rect = (*y_xy, rec["size"][0] / 2, rec["size"][1] / 2, math.radians(y_yaw))
    if not pa.apart(x_rect, y_rect, TARGET_GAP):
        return None
    placed = [(x_rect, TARGET_GAP), (y_rect, GAP)]
    out = []
    for osize in sizes + ([second["size"]] if second is not None else []):
        for _ in range(300):
            oxy = (rng.uniform(*pick_anything.SPREAD_X), rng.uniform(*pick_anything.SPREAD_Y))
            yaw = rng.uniform(0.0, 360.0)
            rect = (*oxy, osize[0] / 2, osize[1] / 2, math.radians(yaw))
            if abs(oxy[0]) - math.hypot(*osize[:2]) / 2 >= MID_BAND and all(pa.apart(rect, q, g) for q, g in placed):
                break
        else:
            return None
        placed.append((rect, GAP))
        out.append((oxy, yaw))
    second_spot = out.pop() if second is not None else None
    return (xy, x_yaw), (y_xy, y_yaw), out, second_spot


def instruction(rng, names, giver):
    """One instruction: a language, usable names of X and Y, a template naming the hands. `names`: {lang: {"x": [...],
    "y": [...]}}."""
    lang = rng.choice(sorted(names))
    hands = NAMES[lang]
    template = rng.choice(TEMPLATES[lang])
    return template.format(
        x=rng.choice(names[lang]["x"]), y=rng.choice(names[lang]["y"]), giver=hands[giver], to=hands[other(giver)]
    )


# --- acceptance (per frame) ---


def held(opening, forces):
    """Does a hand hold something: a pad pressing and the jaw not shut on nothing?"""
    return opening >= HELD_OPEN and max(forces) >= HELD_FORCE


def outcome(q):
    """An episode sidecar's outcome: "accepted" or the first condition it missed."""
    return "accepted" if q.get("accepted") else q.get("reason") or q.get("final_fail") or "?"


def screen_stats(sidecars):
    """{"object" | "giver" | "env": {key: {outcome: episodes}}} of screening episode sidecars."""
    out = {k: {} for k in ("object", "giver", "env")}
    for path in sidecars:
        q = json.loads(Path(path).read_text())
        cat, idx = q["objects"]["target"]
        for axis, key in (("object", f"{cat}/{int(idx)}"), ("giver", q.get("giver")), ("env", q.get("env_cfg"))):
            cell = out[axis].setdefault(key, {})
            cell[outcome(q)] = cell.get(outcome(q), 0) + 1
    return out


GRASP_MISSES = ("not_lifted", "wrong_hand")  # the giver's grasp lifted nothing, or lifted it but not by the pads


def screen_out(objects, stats, low, least, floor=0.0, label="screening"):
    """Mark the objects the giver's grasp does not hold: in at least `least` `label` episodes, GRASP_MISSES in half
    or more and accepted in less than `low` of them. wrong_hand there is a pinch the pads do not press on (a handle a
    few mm thin at the fingertips) or a part caught between the finger roots by its shape (screen 11-19); the other
    failures are the expert's to fix, unless the object is accepted in less than `floor` of them whatever the failure
    (round 1: a pen set down tipping over, 0/127; a fuse slipping out of the hand, 10/126)."""
    marked = []
    for o in objects:
        cell = stats["object"].get(f"{o['cat']}/{o['idx']}", {})
        n, acc, miss = sum(cell.values()), cell.get("accepted", 0), sum(cell.get(k, 0) for k in GRASP_MISSES)
        if n >= least and 2 * miss >= n and acc < low * n:
            o["screened_out"] = f"giver's grasp missed ({', '.join(GRASP_MISSES)}) in {miss}/{n} {label} episodes"
        elif n >= least and acc < floor * n:
            o["screened_out"] = f"accepted in {acc}/{n} {label} episodes (under {floor:.0%})"
        else:
            continue
        marked.append(f"{o['cat']}/{o['idx']}")
    return marked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool", help="rewrite POOL from pick_anything_pool.json (keeps the screened_out marks)")
    q = sub.add_parser("screen", help="outcomes of screening runs; --mark screens out the objects not lifted")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--sidecars", nargs="+", help="raw/seed*/episode_*.json")
    src.add_argument("--stats", type=Path, help="screen_stats of them, as JSON (made where the sidecars are)")
    q.add_argument("--mark", action="store_true", help="write the screened_out marks into POOL")
    q.add_argument("--low", type=float, default=0.25, help="... accepted in less than this share")
    q.add_argument("--least", type=int, default=2, help="... of at least this many episodes")
    q.add_argument("--floor", type=float, default=0.0, help="also out: accepted in less than this share, any failure")
    q.add_argument("--label", default="screening", help="the episodes' name in the marks (e.g. 'round 1')")
    args = parser.parse_args()
    if args.cmd == "pool":
        pick = pick_anything.load_pool()
        found = scan(pick)
        old = {(o["cat"], o["idx"]): o for o in json.loads(POOL.read_text())["objects"]} if POOL.exists() else {}
        for o in found:
            # two layers: every mark POOL has (this task's screening and round 1's: game controllers 0/95) is kept,
            # then the shared rule as it is now
            if old.get((o["cat"], o["idx"]), {}).get("screened_out"):
                o["screened_out"] = old[(o["cat"], o["idx"])]["screened_out"]
            elif excluded(o):
                o["screened_out"] = "excluded: pick_anything.EXCLUDED for handover (auto parts)"
        out = dict(rules=RULES, shift_min=SHIFT_MIN, end=END, grip_gap=GRIP_GAP, scanned=len(pick), objects=found)
        POOL.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        print(f"{len(found)} of {len(pick)} pick anything objects suit a handover")
    if args.cmd == "screen":
        stats = json.loads(args.stats.read_text()) if args.stats else screen_stats(args.sidecars)
        for axis in ("env", "giver", "object"):
            rows = sorted(stats[axis].items(), key=lambda kv: kv[1].get("accepted", 0) / sum(kv[1].values()))
            print(f"{axis}:")
            for key, cell in rows:
                n = sum(cell.values())
                fails = ", ".join(f"{k} {v}" for k, v in sorted(cell.items(), key=lambda kv: -kv[1]) if k != "accepted")
                print(f"  {key}: {cell.get('accepted', 0)}/{n}  {fails}")
        if args.mark:
            pool = json.loads(POOL.read_text())
            marked = screen_out(pool["objects"], stats, args.low, args.least, args.floor, args.label)
            POOL.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
            print(f"screened out {len(marked)}: {', '.join(marked)}")


if __name__ == "__main__":
    main()
