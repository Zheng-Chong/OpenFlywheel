"""Atomic block tasks of G1-D in the office (g1d): one skill each with the painted wood blocks (make_wood_blocks.py),
stdlib only (block_patterns is loaded where sizes or grip yaws are needed). The layout builders (LAYOUTS, registered
in scripts/g1d/build_asset.py PATTERN_TASKS), the tasks, their checks (env/reward_manager/block_checks.py), the
scripted experts (scripts/g1d/scripted_demo.py) and the collection plugins read it from here.

    g1d_place_block_on_mark   放到标记上：one block, a round mark (block_mark) to set it on
    g1d_stack_block_on_block  叠一块：three blocks; stack the named one on the named base
    g1d_pick_named_block      按指令拿起：three or four blocks; pick up the named one
    g1d_put_block_in_tray     放进托盘：a tray (block_tray) and three blocks; put the named one in
    g1d_rotate_block_parallel 转向对齐：one long block lying askew; turn it parallel to the table edge
    g1d_line_up_blocks        排成一排：three blocks of three colours; line them up left to right in the named order

On one table no two blocks share shape and paint, so a block's colour and shape name it. Blocks lie out to both sides
as in the pattern and sorting tasks (|x| SIDE_X, y SCATTER_Y, flat ones in FLAT_Y), each turned so the hand on its
side grips it best; a block is only picked by the hand on its side and set down at most a few cm across the midline.
Env frame: robot at the table's near edge (y -0.60), facing +Y; its left is -X.
"""

import math

TASKS = (
    "g1d_place_block_on_mark",
    "g1d_stack_block_on_block",
    "g1d_pick_named_block",
    "g1d_put_block_in_tray",
    "g1d_rotate_block_parallel",
    "g1d_line_up_blocks",
)
SHAPES = tuple(f"wood_block_{s}" for s in ("cube", "short_cylinder", "half_cylinder", "small_prism", "tile", "cuboid"))
PAINTS = {"blue": 1, "red": 2, "green": 4, "yellow": 5, "pink": 7}  # make_wood_blocks.PALETTE index
PAINT_ZH = {"blue": "蓝色", "red": "红色", "green": "绿色", "yellow": "黄色", "pink": "粉色"}
BASE_SHAPES = tuple(f"wood_block_{s}" for s in ("cube", "short_cylinder", "cuboid"))  # flat tops to stack on
TOP_SHAPES = tuple(f"wood_block_{s}" for s in ("cube", "short_cylinder", "tile", "half_cylinder", "small_prism"))
LONG_SHAPES = tuple(f"wood_block_{s}" for s in ("plank", "short_plank", "cuboid"))  # to turn parallel
ROW_SHAPES = tuple(f"wood_block_{s}" for s in ("cube", "short_cylinder", "tile", "half_cylinder"))  # 30 mm wide

SIDE_X, SCATTER_Y, FLAT_Y = (0.125, 0.18), (-0.53, -0.37), (-0.45, -0.40)  # as block_sorting: reachable picks
GAP, GRASP_YAW = 0.015, 20.0  # m between footprints; deg about the hand's best grip yaw
MARK = ("block_mark", 0)  # make_block_mark.py: 70 mm square, a 50 mm red disc printed on it
MARK_X, MARK_Y, MARK_RADIUS = (-0.06, 0.06), (-0.47, -0.39), 0.02  # where the mark lies; block centre within radius
MARK_SIDE = 0.02  # m: a mark further off the midline has its block on its side
TRAY = ("block_tray", 0)
TRAY_XY = (0.0, -0.42)  # the tray in the middle, long side along Y
ROW_XY, ROW_STEP = (0.0, -0.42), 0.045  # line-up slots at ROW_XY + (-1, 0, 1) * ROW_STEP
ROTATE_TOL, ROTATE_STAY = 10.0, 0.06  # deg off parallel; m it may move


def paint_name(paint):
    return next(k for k, v in PAINTS.items() if v == paint)


def _sizes():
    from scripts.objects.block_patterns import SIZES  # numpy: not at import (the platform imports build_asset)

    return SIZES


def scatter(rng, blocks, taken=()):
    """xy per block of `blocks` [(category, side)] (side -1 left, 1 right, 0 either), clear of each other and of
    `taken` [(xy, radius)]; None if there is no room."""
    sizes, placed, out = _sizes(), list(taken), []
    for cat, side in blocks:
        w, d, h = sizes[cat]
        radius = math.hypot(w, d) / 2
        for _ in range(500):
            s = side or rng.choice((-1, 1))
            xy = (s * rng.uniform(*SIDE_X), rng.uniform(*(FLAT_Y if h < 0.02 else SCATTER_Y)))
            if all(math.dist(xy, q) >= radius + r + GAP for q, r in placed):
                break
        else:
            return None
        placed.append((xy, radius))
        out.append(xy)
    return out


def grip_turn(rng, cat, xy):
    """A yaw (deg) for a block at `xy` that the hand on its side grips across its local Y at its best grip yaw."""
    from scripts.objects.block_patterns import YAW_PERIOD, grip_yaw

    period = YAW_PERIOD[cat]
    if not period:
        return rng.uniform(0.0, 360.0)
    return grip_yaw(*xy) - 90.0 + rng.uniform(-GRASP_YAW, GRASP_YAW) + period * rng.randrange(360 // period)


def distinct_blocks(rng, n, shapes=SHAPES):
    """`n` (category, paint) pairs, no two alike."""
    pool = [(c, p) for c in shapes for p in PAINTS.values()]
    return rng.sample(pool, n)


def place_blocks(rng, entry, blocks, taken=()):
    """Layout entries of `blocks` [(category, paint, label, side)] scattered by scatter(): {category: [entry]}."""
    for _ in range(100):
        spots = scatter(rng, [(c, s) for c, _, _, s in blocks], taken)
        if spots:
            break
    else:
        raise RuntimeError(f"no room for {[label for _, _, label, _ in blocks]}")
    rigid = {}
    for (cat, paint, label, _), xy in zip(blocks, spots):
        rigid.setdefault(cat, []).append(entry(cat, paint, label, xy, grip_turn(rng, cat, xy)))
    return rigid


def merge(*parts):
    out = {}
    for part in parts:
        for cat, entries in part.items():
            out.setdefault(cat, []).extend(entries)
    return out


def place_on_mark(rng, entry):
    """A block and the mark (label "mark"); the block (label "block") lies on the mark's side when the mark is off
    the midline."""
    mark = (rng.uniform(*MARK_X), rng.uniform(*MARK_Y))
    ((cat, paint),) = distinct_blocks(rng, 1)
    side = (1 if mark[0] > 0 else -1) if abs(mark[0]) > MARK_SIDE else 0
    marks = {MARK[0]: [entry(MARK[0], MARK[1], "mark", mark, 0.0)]}
    return merge(marks, place_blocks(rng, entry, [(cat, paint, "block", side)], [(mark, 0.05)]))


def stack_on_block(rng, entry):
    """Three blocks: "top_block" to stack on "base_block" (a flat-topped one, on the same side) and "other_block"."""
    base = rng.choice([(c, p) for c in BASE_SHAPES for p in PAINTS.values()])
    top = rng.choice([(c, p) for c in TOP_SHAPES for p in PAINTS.values() if (c, p) != base])
    other = rng.choice([(c, p) for c in SHAPES for p in PAINTS.values() if (c, p) not in (base, top)])
    side = rng.choice((-1, 1))
    blocks = [(*base, "base_block", side), (*top, "top_block", side), (*other, "other_block", 0)]
    return place_blocks(rng, entry, blocks)


def pick_named(rng, entry):
    """Three or four blocks; "target_block" is the one to pick up, the others "other_<j>"."""
    picks = distinct_blocks(rng, rng.choice((3, 4)))
    blocks = [(*picks[0], "target_block", 0)] + [(*b, f"other_{j}", 0) for j, b in enumerate(picks[1:])]
    return place_blocks(rng, entry, blocks)


def put_in_tray(rng, entry):
    """The tray in the middle ("tray") and three blocks; "target_block" goes in, the others "other_<j>" stay out."""
    picks = distinct_blocks(rng, 3, TOP_SHAPES)  # small enough for the tray (cuboid 60 mm: tight)
    tray = {TRAY[0]: [entry(TRAY[0], TRAY[1], "tray", TRAY_XY, 0.0)]}
    blocks = [(*picks[0], "target_block", 0)] + [(*b, f"other_{j}", 0) for j, b in enumerate(picks[1:])]
    return merge(tray, place_blocks(rng, entry, blocks, [(TRAY_XY, 0.075)]))


def rotate_parallel(rng, entry):
    """One long block ("block"), turned for its hand's grip: its long side 25-95 deg off parallel to the edge."""
    cat = rng.choice(LONG_SHAPES)
    return place_blocks(rng, entry, [(cat, rng.choice(list(PAINTS.values())), "block", 0)])


def line_up(rng, entry):
    """Three blocks of three paints, block_<i> the i-th from the left in the row asked for: block_0 lies on the left,
    block_2 on the right, block_1 on either side (each hand sets its block down at most ROW_STEP across)."""
    paints = rng.sample(list(PAINTS.values()), 3)
    blocks = [(rng.choice(ROW_SHAPES), p, f"block_{i}", (-1, 0, 1)[i]) for i, p in enumerate(paints)]
    return place_blocks(rng, entry, blocks)


def row_slots():
    return [(ROW_XY[0] + k * ROW_STEP, ROW_XY[1]) for k in (-1, 0, 1)]


LAYOUTS = dict(
    zip(TASKS, (place_on_mark, stack_on_block, pick_named, put_in_tray, rotate_parallel, line_up), strict=True)
)
