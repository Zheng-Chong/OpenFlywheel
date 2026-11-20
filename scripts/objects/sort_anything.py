"""g1d_sort_anything (按种类分拣任意物体): 2-4 objects X of 2 or 3 sort classes and one receptacle Y per class (a
tray, plate or cutting board to put on, a basket, box, bowl or deep tray to put in); every X goes to its class's Y.
The multi-step g1d_place_anything / g1d_put_anything_in: X from g1d_pick_anything's commercially usable pool, Y from
Place / Put Anything's receptacle pool (round 2, place_anything_receptacles_r2.json) and their acceptance geometry
(scripts/objects/place_anything.py). Stdlib only (the platform imports it).

    sort class    a name a person sorts by (水果 fruit, 文具 stationery ...) over pick_anything pool families (CLASSES);
                  classes one could mistake for each other (积木 / 玩具, 罐子 / 饮料, 杯子 / 餐具 ...) never share a
                  layout (GROUPS); objects whose names blur their class (字母O among toys, 喷漆罐 among bottles) are out
    receptacle    a training receptacle of SUPPORT_FAMILIES / CONTAINER_FAMILIES (cups and mugs are too small for
                  several objects and would read as objects to sort); each class's objects all fit it at once: its
                  region split along its long side into one slot per object (slots), each object inside its slot at
                  the yaw it lies (and is carried) at
    labels        item_<i> (X, layout entry "sort": {"class", "receptacle"}), receptacle_<j> (Y, "sort": {"class"},
                  class None for a spare one no instruction names) with the Y's geometry under "receptacle" as
                  place_anything's
    instruction   the classes by their class names and each Y by a name nothing else on the desk has
                  (pick_anything.usable_names), with left / middle / right in half the layouts (robot's view)
"""

import itertools
import math

from scripts.objects import pick_anything, place_anything as pa

TASK = "g1d_sort_anything"
ITEM, RECEPTACLE = "item", "receptacle"
# sort class: (zh, en singular, en plural), pool families, zh name parts that put an object out of it
CLASSES = {
    "fruit": (("水果", "fruit", "fruit"), ("fruit",), ()),
    "vegetable": (("蔬菜", "vegetable", "vegetables"), ("vegetable",), ()),
    "stationery": (("文具", "stationery", "stationery"), ("pen", "stationery"), ()),
    "block": (("积木", "block", "blocks"), ("block",), ()),
    "peg": (("插块", "peg", "pegs"), ("peg",), ()),
    "toy": (("玩具", "toy", "toys"), ("toy",), ("字母",)),
    "bottle": (("瓶子", "bottle", "bottles"), ("bottle",), ("罐",)),
    "can": (("罐子", "can", "cans"), ("can",), ()),
    "drink": (("饮料", "drink", "drinks"), ("drink",), ()),
    "condiment": (("调味品", "condiment", "condiments"), ("condiment",), ()),
    "cup": (("杯子", "cup", "cups"), ("cup", "mug"), ()),
    "electronics": (("电子产品", "electronic device", "electronics"), ("electronics",), ()),
    "tool": (("工具", "tool", "tools"), ("tool", "hammer"), ("枪架",)),
    "cleaning": (("清洁用品", "cleaning supply", "cleaning supplies"), ("cleaning supply", "cleaning"), ()),
    "trash": (("垃圾", "piece of trash", "trash"), ("trash",), ()),
    "dessert": (("甜点", "dessert", "desserts"), ("dessert",), ()),
    "utensil": (("餐具", "utensil", "utensils"), ("utensil",), ()),
    "medical": (("医用品", "medical supply", "medical supplies"), ("medical supply",), ()),
}
GROUPS = (
    {"block", "peg", "toy"},  # building blocks and FMB pegs are toys, and look alike
    {"bottle", "can", "drink", "condiment", "cleaning"},  # drinks come in cans and bottles; a powder canister
    {"cup", "utensil"},  # tableware either way
    {"stationery", "medical"},  # a tape roll and a bandage roll
)
SUPPORT_FAMILIES = {"tray", "plate", "cutting board"}
CONTAINER_FAMILIES = {"basket", "box", "bowl", "tray"}
MAX_PER_CLASS = 3
# a layout: 2 classes (3 with THREE_CLASSES), ITEMS objects (at least one per class), with 2 classes and language L2
# a spare receptacle in SPARE of the layouts; at language L2 the second receptacle is of the first's family in
# SAME_FAMILY of the layouts (two baskets: named by what tells them apart)
THREE_CLASSES, ITEMS, SPARE, SAME_FAMILY = 0.3, (2, 4), 0.5, 0.5
# a spare receptacle is one of the smaller ones (m, footprint side): with a large one the objects, in front of the
# receptacles, rarely found room (17% of 2-object draws, 66% with spares up to 20 cm)
SPARE_MAX = 0.20
# per collection round: objects per layout, share of layouts with 3 classes, receptacles per layout (3: a spare one
# with 2 classes; None: as SPARE), largest receptacle footprint side (m; None: any), SPEC_L2 changes. r1 (2 objects
# of 2 classes): a per-grasp miss rate of ~50% (Pick v3) leaves layouts of 3-4 objects few accepted; r2: 3-4 objects
# and 3 receptacles, small receptacles closer along X (3 of the pool's larger ones rarely leave the objects room) in
# the far half of the band, the objects in front of them (3 objects fit 35% of the draws, 16% with the whole band)
ROUNDS = {
    "r1": dict(items=(2, 2), three=0.0, receptacles=None, receptacle_max=None, spec={}),
    "r2": dict(
        items=(3, 4),
        three=0.5,
        receptacles=3,
        receptacle_max=0.16,
        spec=dict(side_gap=0.10, receptacle_y=(-0.42, -0.37)),
    ),
}
# a layout redraws what it holds DRAWS times, each placed at most PLACES times, before it holds less (fewer); STARTS
# times from a new draw at the smallest
DRAWS, PLACES, STARTS = 8, 40, 30
# m, deg (env frame of the g1d desk, the robot's front edge toward -Y): receptacles' centres in receptacle_y (Place
# Anything's band: the probe's receptacles centred at y -0.41 – -0.24 put the spots out of the carrying hand's reach,
# 4-22 cm off), their far edges nearer than far_edge, centres within receptacle_x of the middle, at least side_gap
# apart along X (so left / middle / right says which) and their footprints Y_GAP apart; objects in
# pick_anything_layouts' L2 area, TARGET_GAP clear of everything (the open fingers), and on their receptacle's side
# when it stands more than side_free off the middle: the hand on the receptacle's side carries them (sort_arm; one
# arm does not reach a spot across the desk, and screening 8821 missed 13 of 16 grasps of objects across the middle);
# a share `cross` of them within side_free across the middle; and no farther from the robot than their receptacle's
# centre (screens 8933 / 8994: carried back toward the robot, the hand stopped 7-17 cm short and pressed the
# receptacle away as it lowered onto its far rim)
SPEC_L2 = dict(receptacle_x=0.24, receptacle_y=(-0.47, -0.37), far_edge=-0.16, receptacle_yaw=180.0, slots=None,
               side_gap=0.12, side_free=0.08, cross=0.0, target_x=0.25, target_y=(-0.53, -0.30), mid_gap=0.02,
               grasp_yaw=30.0)  # fmt: skip
# L1: the receptacles in fixed slots along X (jittered 2 cm), within 15 deg of either way along X; grip yaw ±20;
# objects across the middle too (the receptacles along X leave the sides little room)
SPEC_L1 = dict(SPEC_L2, receptacle_x=0.02, receptacle_y=(-0.44, -0.40), receptacle_yaw=15.0, grasp_yaw=20.0,
               slots={2: (-0.13, 0.13), 3: (-0.20, 0.0, 0.20)}, cross=0.3)  # fmt: skip
Y_GAP, TARGET_GAP = 0.03, pick_anything.TARGET_GAP
SIDES = {
    2: (("左边的", "left"), ("右边的", "right")),
    3: (("左边的", "left"), ("中间的", "middle"), ("右边的", "right")),
}
SIDE_WORDS = 0.5  # share of instructions naming the receptacles with their side
TEMPLATES = {
    "zh": {
        "each": ("{clauses}。", "按类别整理桌面：{clauses}。"),
        "sort": ("把桌上的东西按种类分开放：{short}。", "请分拣桌上的物品，{short}。"),
        "clause": {"support": "把{c}放到{y}上", "container": "把{c}放进{y}里"},
        "short": {"support": "{c}放到{y}上", "container": "{c}放进{y}里"},
        "join": "，",
    },
    "en": {
        "each": ("{Clauses}.", "Tidy up the table by type: {clauses}."),
        "sort": ("Sort the things on the table by type: {short}.", "Please sort the items: {short}."),
        "clause": {"support": "put the {c} on the {y}", "container": "put the {c} in the {y}"},
        "short": {"support": "the {c} on the {y}", "container": "the {c} in the {y}"},
        "join": ", ",
    },
}


# --- objects and receptacles ---


def class_of(obj, names):
    """The sort class of a pool object (`names`: its object_names entry), or None."""
    zh = (names or {}).get("zh") or []
    for key, (_, families, out) in CLASSES.items():
        if obj["family"] in families:
            return None if any(part in n for part in out for n in zh) else key
    return None


def name_of(names, oid):
    return names.get(oid) or {}


def oid(obj):
    return f"Rigid/{obj['cat']}/{obj['idx']:05d}"


def sort_receptacles(receptacles, split="train"):
    """[(receptacle, kind)] Y may be: of `split` (and not screened out of training), of a sorting family."""
    out = []
    for r in receptacles:
        for kind, families in (("container", CONTAINER_FAMILIES), ("support", SUPPORT_FAMILIES)):
            if kind in r["kinds"] and r["family"] in families and r in pa.of_kind(receptacles, kind, split):
                out.append((r, kind))
                break
    return out


def class_pool(pool, names, receptacles):
    """{class: [pool objects]}: each of a sort class, not left out for Sort (pick_anything.usable), that fits some sort
    receptacle alone."""
    ys = sort_receptacles(receptacles)
    out = {}
    for obj in pick_anything.usable(pool, TASK):
        key = class_of(obj, name_of(names, oid(obj)))
        if key and any(pa.fits_some_yaw(r["geometry"][kind], obj["size"], kind) for r, kind in ys):
            out.setdefault(key, []).append(obj)
    return out


def compatible(classes):
    return all(len(set(classes) & g) <= 1 for g in GROUPS)


# --- slots: several objects in one receptacle ---


def split_region(geom, k):
    """k sub-geometries of a receptacle's region: its rectangle cut along its long side into k equal slots, from the
    negative end; k = 1: the geometry itself (its circle too)."""
    if k == 1:
        return [geom]
    reg = geom["region"]
    axis = 0 if reg["half"][0] >= reg["half"][1] else 1
    out = []
    for i in range(k):
        centre, half = list(reg["center"]), list(reg["half"])
        half[axis] = reg["half"][axis] / k
        centre[axis] = reg["center"][axis] - reg["half"][axis] + (2 * i + 1) * half[axis]
        out.append(
            dict(geom, region=dict(center=[round(v, 5) for v in centre], half=[round(v, 5) for v in half], radius=0.0))
        )
    return out


def long_axis(geom):
    reg = geom["region"]
    return 0 if reg["half"][0] >= reg["half"][1] else 1


def assign_slots(geom, kind, items):
    """Sub-geometries of `geom` for `items` [(size, yaw rad off the receptacle's, where it comes from in the
    receptacle's frame (x, y))], in their order: the slots by where the items come from along the long side first,
    else any order every item fits; None when no order does."""
    subs = split_region(geom, len(items))
    axis = long_axis(geom)
    first = sorted(range(len(items)), key=lambda i: items[i][2][axis])
    orders = [first] + [list(p) for p in itertools.permutations(range(len(items))) if list(p) != first]
    for order in orders:
        out = [None] * len(items)
        for slot, i in enumerate(order):
            size, rel, _ = items[i]
            if not pa.fits(subs[slot], size, kind, rel):
                break
            out[i] = subs[slot]
        else:
            return out
    return None


def holds(geom, kind, sizes):
    """Could `geom` take all of `sizes` at once at some yaws (each in its own slot)?"""
    return all(pa.fits_some_yaw(sub, size, kind) for sub, size in zip(split_region(geom, len(sizes)), sizes))


# --- layouts ---


def draw_counts(rng, k, items=ITEMS):
    """Objects per class: `items` (least, most) in all, at least one each, at most MAX_PER_CLASS."""
    n = rng.randint(max(items[0], k), items[1])
    while True:
        counts = [1] * k
        for _ in range(n - k):
            counts[rng.randrange(k)] += 1
        if max(counts) <= MAX_PER_CLASS:
            return counts


def draw_combo(rng, three=THREE_CLASSES, spare=0.0, items=ITEMS, receptacles=None):
    """How many of what a layout holds: {"counts": objects per class, "spare": a spare receptacle (2 classes only;
    always with `receptacles` 3)}."""
    k = 3 if rng.random() < three else 2
    return dict(counts=draw_counts(rng, k, items), spare=k == 2 and (receptacles == 3 or rng.random() < spare))


def draw(rng, combo, classes_pool, receptacles, scale=None, same_family=0.0):
    """What a layout of `combo` holds: [(class, [(object, scale)], (receptacle, kind))] and the spare (receptacle,
    kind) or None. scale: None or f(rng, obj) -> uniform scale (dropped when no receptacle takes the class's objects
    scaled); a class no receptacle takes that many of gets one fewer; None when the receptacles left take none of a
    class's objects."""
    ys = sort_receptacles(receptacles)
    keys = sorted(classes_pool)
    k = len(combo["counts"])
    while True:
        classes = rng.sample(keys, k)
        if compatible(classes):
            break
    groups, used = [], set()
    for c, n in zip(classes, combo["counts"]):
        objs = rng.sample(classes_pool[c], min(n, len(classes_pool[c])))
        scales = [scale(rng, o) if scale else 1.0 for o in objs]
        while True:
            fit = []
            for s in (scales, [1.0] * len(objs)):
                sizes = [[v * f for v in o["size"]] for o, f in zip(objs, s)]
                fit = [(r, kd) for r, kd in ys if r["id"] not in used and holds(r["geometry"][kd], kd, sizes)]
                if fit:
                    scales = s
                    break
            if fit or len(objs) == 1:
                break
            objs, scales = objs[:-1], scales[:-1]
        if not fit:  # the receptacles left take none of the class's objects
            return None
        if groups and rng.random() < same_family:
            family = groups[0][2][0]["family"]
            fit = [y for y in fit if y[0]["family"] == family] or fit
        y = rng.choice(fit)
        used.add(y[0]["id"])
        groups.append((c, list(zip(objs, scales)), y))
    extra = None
    if combo["spare"]:
        rest = [y for y in ys if y[0]["id"] not in used and max(y[0]["size"][:2]) <= SPARE_MAX]
        if not rest:
            return None
        same = [y for y in rest if y[0]["family"] in {g[2][0]["family"] for g in groups}]
        extra = rng.choice(same if same and rng.random() < 0.5 else rest)
    return groups, extra


def fewer(combo, items=ITEMS, receptacles=None):
    """The next smaller combo when a layout finds no room: no spare, then one object fewer from the fullest class,
    then two classes; None at the smallest (no fewer than `items[0]` objects, with `receptacles` 3 keeping 3)."""
    counts, spare = list(combo["counts"]), combo["spare"]
    if spare and receptacles != 3:
        return dict(counts=counts, spare=False)
    if sum(counts) > max(len(counts), items[0]):
        counts[counts.index(max(counts))] -= 1
        return dict(counts=counts, spare=spare)
    if len(counts) > 2 and receptacles != 3 and items[0] <= 2:
        return dict(counts=[1, 1], spare=False)
    return None


def side_of(x, free=SPEC_L2["side_free"]):
    """-1 / 1: the side (env X) of a receptacle at `x` whose hand carries its objects; 0 within `free` of the
    middle (the hand on the object's side)."""
    return 0 if abs(x) <= free else (1 if x > 0 else -1)


def footprint(size, xy, yaw_deg):
    return (xy[0], xy[1], size[0] / 2, size[1] / 2, math.radians(yaw_deg))


def place(rng, groups, extra, spec):
    """Poses of a draw, or None: {"receptacles": [(xy, yaw deg)] (groups' then the spare's), "items": [[(xy, yaw deg)]
    per group]}: receptacles along X side_gap apart (or in spec["slots"]), Y_GAP between their footprints; objects
    TARGET_GAP clear of everything, each class's objects in slots of their receptacle at their yaws."""
    ys = [g[2] for g in groups] + ([extra] if extra else [])
    n = len(ys)
    order = list(range(n))
    rng.shuffle(order)  # which receptacle stands where along X, from the left
    yaws = [rng.uniform(-spec["receptacle_yaw"], spec["receptacle_yaw"]) + 180.0 * rng.randrange(2) for _ in ys]
    if spec["slots"]:
        xs = [spec["slots"][n][order.index(i)] + rng.uniform(-spec["receptacle_x"], spec["receptacle_x"])
              for i in range(n)]  # fmt: skip
    else:
        # neighbours along X: centres side_gap apart and their footprints' extents along X Y_GAP apart; the slack
        # spread at random
        hx = [pa.turned_half(*ys[i][0]["size"][:2], math.radians(yaws[i]))[0] for i in order]
        steps = [max(spec["side_gap"], a + b + Y_GAP) for a, b in zip(hx, hx[1:])]
        slack = 2 * spec["receptacle_x"] - sum(steps)
        if slack < 0:
            return None
        cuts = sorted(rng.uniform(0.0, slack) for _ in range(n))
        line = [-spec["receptacle_x"] + cuts[0]]
        for step, (a, b) in zip(steps, zip(cuts, cuts[1:])):
            line.append(line[-1] + step + b - a)
        xs = [line[order.index(i)] for i in range(n)]
    rects, poses = [], []
    for (r, _), x, yaw in zip(ys, xs, yaws):
        hy = pa.turned_half(*r["size"][:2], math.radians(yaw))[1]
        xy = (x, rng.uniform(*spec["receptacle_y"]))
        if xy[1] + hy > spec["far_edge"]:
            return None
        rect = footprint(r["size"], xy, yaw)
        if not all(pa.apart(rect, q, Y_GAP) for q in rects):
            return None
        rects.append(rect)
        poses.append((xy, yaw))
    placed = list(rects)
    items = []
    for (_, objs, (r, kind)), (y_xy, y_yaw) in zip(groups, poses):
        mine = []
        side = side_of(y_xy[0], spec["side_free"])
        for obj, s in objs:
            size = [v * s for v in obj["size"]]
            for _ in range(200):
                x = rng.uniform(spec["mid_gap"], spec["target_x"]) * (side or rng.choice((-1, 1)))
                if side and rng.random() < spec["cross"]:
                    x = -side * rng.uniform(spec["mid_gap"], spec["side_free"])
                xy = (x, rng.uniform(spec["target_y"][0], min(spec["target_y"][1], y_xy[1])))
                yaw = pick_anything.target_yaw(rng, obj, xy, spec["grasp_yaw"])
                rect = footprint(size, xy, yaw)
                if all(pa.apart(rect, q, TARGET_GAP) for q in placed):
                    break
            else:
                return None
            placed.append(rect)
            mine.append((xy, yaw, size))
        local = [
            (size, math.radians(yaw - y_yaw), to_local(xy, y_xy, y_yaw)) for xy, yaw, size in mine
        ]  # fmt: skip
        if assign_slots(r["geometry"][kind], kind, local) is None:
            return None
        items.append([(xy, yaw) for xy, yaw, _ in mine])
    return dict(receptacles=poses, items=items)


def to_local(xy, origin, yaw_deg):
    a = math.radians(yaw_deg)
    dx, dy = xy[0] - origin[0], xy[1] - origin[1]
    return (math.cos(a) * dx + math.sin(a) * dy, -math.sin(a) * dx + math.cos(a) * dy)


def layout(rng, entry, receptacle_entry, classes_pool, receptacles, spec=SPEC_L2, scale=None, spare=0.0,
           same_family=0.0, three=THREE_CLASSES, items=ITEMS, n_receptacles=None):  # fmt: skip
    """Rigid section of a layout and what was drawn. entry(cat, idx, label, xy, yaw, scale) makes an object's entry,
    receptacle_entry(rec, kind, label, xy, yaw) a receptacle's (with its geometry under "receptacle"); items,
    n_receptacles: a round's (ROUNDS)."""
    combo = draw_combo(rng, three, spare, items, n_receptacles)
    planned, poses, starts = dict(combo), None, 1
    while poses is None:
        for _ in range(DRAWS):
            drawn = draw(rng, combo, classes_pool, receptacles, scale, same_family)
            if drawn is None or sum(len(g[1]) for g in drawn[0]) < items[0]:
                continue
            groups, extra = drawn
            for _ in range(PLACES):
                poses = place(rng, groups, extra, spec)
                if poses:
                    break
            if poses:
                break
        else:
            combo = fewer(combo, items, n_receptacles)
            if combo is None:  # large receptacles even at the smallest: start over
                if starts == STARTS:
                    raise RuntimeError("no room for a sort_anything layout")
                combo, starts = draw_combo(rng, three, spare, items, n_receptacles), starts + 1
                planned = dict(combo)
    rigid = {}
    ys = [g[2] for g in groups] + ([extra] if extra else [])
    y_labels = [f"{RECEPTACLE}_{j}" for j in range(len(ys))]
    for j, ((r, kind), (xy, yaw)) in enumerate(zip(ys, poses["receptacles"])):
        e = receptacle_entry(r, kind, y_labels[j], xy, yaw)
        e["sort"] = {"class": groups[j][0] if j < len(groups) else None}
        rigid.setdefault(r["cat"], []).append(e)
    i = 0
    for j, ((c, objs, _), spots) in enumerate(zip(groups, poses["items"])):
        for (obj, s), (xy, yaw) in zip(objs, spots):
            e = entry(obj["cat"], obj["idx"], f"{ITEM}_{i}", xy, yaw, s)
            e["sort"] = {"class": c, "receptacle": y_labels[j]}
            rigid.setdefault(obj["cat"], []).append(e)
            i += 1
    drawn = dict(
        classes=[g[0] for g in groups],
        items=[len(g[1]) for g in groups],
        receptacles={y_labels[j]: dict(id=r["id"], kind=kind) for j, (r, kind) in enumerate(ys)},
        spare=y_labels[-1] if extra else None,
        planned=planned,
    )
    return rigid, drawn


# --- instructions ---


def sides(xs):
    """Side words (zh, en) of receptacles at env X `xs`, the robot's left is -X."""
    ranks = sorted(range(len(xs)), key=lambda i: xs[i])
    words = SIDES[len(xs)]
    return {i: words[ranks.index(i)] for i in range(len(xs))}


def instruction(rng, parts, lang, side=None):
    """One instruction in `lang`: parts [(class, its object count, receptacle kind, receptacle name, receptacle
    index)] in the order they are said; `side`: {receptacle index: (zh, en)} to name each with its side, or None."""
    t = TEMPLATES[lang]
    style = rng.choice(("each", "sort"))
    pieces = []
    for c, count, kind, yname, at in parts:
        zh, one, many = CLASSES[c][0]
        cname = zh if lang == "zh" else (one if count == 1 else many)
        if side is not None:
            word = side[at][0 if lang == "zh" else 1]
            yname = word + yname if lang == "zh" else f"{word} {yname}"
        pieces.append(t["clause" if style == "each" else "short"][kind].format(c=cname, y=yname))
    if lang == "zh":
        body = t["join"].join(pieces)
    elif len(pieces) == 2:
        body = " and ".join(pieces)
    else:
        body = ", ".join(pieces[:-1]) + " and " + pieces[-1]
    template = rng.choice(t[style])
    key = "clauses" if style == "each" else "short"
    return template.format(**{key: body, "Clauses": body[:1].upper() + body[1:]})


def class_names(c):
    zh, one, many = CLASSES[c][0]
    return {"zh": [zh], "en": sorted({one, many})}
