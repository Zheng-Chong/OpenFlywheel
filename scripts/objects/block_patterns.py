"""Block patterns for g1d_build_block_pattern: what to build from the wood blocks (make_wood_blocks.py), numpy only.

A pattern is a front-facing build like a picture-book toy (house, gate, bridge): pieces standing in their resting pose
(make_wood_blocks.py frame, only turned about Z), centred on one plane, listed bottom-up in the order they are placed.
A pattern may give each piece a paint ("paints": make_wood_blocks.PALETTE index, 0 = natural wood, the default); pieces
of the same shape and paint are interchangeable (groups()).
The layout builder (scripts/g1d/build_asset.py), the pattern cards (make_pattern_cards.py), the task's success check
and the scripted expert (scripts/g1d/scripted_demo.py) all read it from here.

Pattern frame: X along the facade (to the robot's right when the card faces it), Y its depth, Z up from the table;
a piece's pose is its bbox centre (the asset origin) and its yaw in degrees.
"""

import numpy as np

# model extents (m, X Y Z) of the wood_block_* assets, as make_wood_blocks.py builds them (test_block_patterns.py)
SIZES = {
    "wood_block_plank": (0.090, 0.030, 0.015),
    "wood_block_short_plank": (0.060, 0.030, 0.015),
    "wood_block_tile": (0.030, 0.030, 0.015),
    "wood_block_cuboid": (0.060, 0.030, 0.030),
    "wood_block_cube": (0.030, 0.030, 0.030),
    "wood_block_cylinder": (0.030, 0.030, 0.060),
    "wood_block_short_cylinder": (0.030, 0.030, 0.030),
    "wood_block_arch": (0.0601, 0.0297, 0.03004),
    "wood_block_prism": (0.0557, 0.0293, 0.0293),
    "wood_block_small_prism": (0.040, 0.030, 0.0215),
    "wood_block_half_cylinder": (0.0298, 0.030, 0.015),
}
ARCH_OPENING = 0.03004 - 0.00958  # radius of the arch's half-round opening
# env frame (robot at the table's near edge, facing +Y): the pattern is built about BUILD_XY, the card lies at CARD_XY
BUILD_XY, CARD_XY = (0.0, -0.42), (0.0, -0.26)  # both hands reach the build space across, 3 cm up
# hand yaw (deg) the hand on a block's side grips it best with (the top grasp of scripts/g1d/scripted_demo.py):
# fingers turned toward the midline, less so near the robot; build_asset.py turns each scattered block to suit
GRIP_YAW_FAR, GRIP_YAW_NEAR, NEAR_Y = 45.0, 15.0, -0.45


def grip_yaw(x, y):
    return (GRIP_YAW_FAR if y > NEAR_Y else GRIP_YAW_NEAR) * (1 if x > 0 else -1)


# yaw (deg) after which a block looks the same: 0 = any (round), 90 = square, 180 = the rest
YAW_PERIOD = {cat: 180 for cat in SIZES} | {
    "wood_block_tile": 90,
    "wood_block_cube": 90,
    "wood_block_cylinder": 0,
    "wood_block_short_cylinder": 0,
}
# front view outline: rectangle, triangle (prism on its hypotenuse), arch, half disc (half-cylinder on its flat side)
OUTLINE = {cat: "rect" for cat in SIZES} | {
    "wood_block_prism": "triangle",
    "wood_block_small_prism": "triangle",
    "wood_block_arch": "arch",
    "wood_block_half_cylinder": "dome",
}

# pieces: (category, x, bottom z), m, yaw 0 (long side along X); anchor: the piece whose yaw fixes the pattern frame
# (an elongated one). Neighbours stand 10 mm apart so a piece can be set down beside another.
PATTERNS = [
    dict(
        name="house",
        zh="小房子",
        anchor=2,
        pieces=[
            ("wood_block_cube", -0.020, 0.0),
            ("wood_block_cube", 0.020, 0.0),
            ("wood_block_short_plank", 0.0, 0.030),
            ("wood_block_prism", 0.0, 0.045),
        ],
    ),
    dict(
        name="gate",
        zh="牌坊",
        anchor=2,
        pieces=[
            ("wood_block_cylinder", -0.030, 0.0),
            ("wood_block_cylinder", 0.030, 0.0),
            ("wood_block_plank", 0.0, 0.060),
            ("wood_block_half_cylinder", 0.0, 0.075),
        ],
    ),
    dict(
        name="bridge",
        zh="拱桥",
        anchor=0,
        pieces=[
            ("wood_block_arch", 0.0, 0.0),
            ("wood_block_cube", -0.070, 0.0),  # 10 mm clear of the plank's ends: nearer, a cube landing 5 mm in
            ("wood_block_cube", 0.070, 0.0),  # bore one end of the plank and the top collapsed
            ("wood_block_plank", 0.0, 0.03004),
            ("wood_block_small_prism", 0.0, 0.04504),
        ],
    ),
    # from here on painted (blue 1, red 2, green 4, yellow 5) and larger; the first three keep their indices (cards
    # 00000-00002, the layouts of benchmark block_pattern@v1)
    dict(
        name="tower",
        zh="彩色塔",
        anchor=0,
        pieces=[("wood_block_cube", 0.0, 0.0), ("wood_block_cube", 0.0, 0.030), ("wood_block_cube", 0.0, 0.060)],
        paints=[2, 5, 1],
    ),
    dict(
        name="ell",
        zh="L 形",
        anchor=1,
        pieces=[
            ("wood_block_cube", -0.040, 0.0),
            ("wood_block_cube", 0.0, 0.0),
            ("wood_block_cube", 0.040, 0.0),
            ("wood_block_cube", -0.040, 0.030),
        ],
        paints=[1, 4, 4, 1],
    ),
    dict(
        name="car",
        zh="小汽车",
        anchor=2,
        pieces=[
            ("wood_block_cube", -0.030, 0.0),
            ("wood_block_cube", 0.030, 0.0),
            ("wood_block_plank", 0.0, 0.030),
            ("wood_block_cuboid", -0.020, 0.045),
            ("wood_block_tile", 0.035, 0.045),
        ],
        paints=[1, 1, 2, 5, 4],
    ),
    dict(
        name="steps",
        zh="台阶",
        anchor=0,
        pieces=[
            ("wood_block_cube", -0.040, 0.0),
            ("wood_block_cube", 0.0, 0.0),
            ("wood_block_cube", 0.040, 0.0),
            ("wood_block_cube", 0.0, 0.030),
            ("wood_block_cube", 0.040, 0.030),
            ("wood_block_cube", 0.040, 0.060),
        ],
        paints=[2, 5, 4, 5, 4, 4],
    ),
    dict(
        name="pyramid",
        zh="金字塔",
        anchor=0,
        pieces=[  # each upper cube spans the 10 mm gap between two below, 10 mm on each
            ("wood_block_cube", -0.040, 0.0),
            ("wood_block_cube", 0.0, 0.0),
            ("wood_block_cube", 0.040, 0.0),
            ("wood_block_cube", -0.020, 0.030),
            ("wood_block_cube", 0.020, 0.030),
            ("wood_block_cube", 0.0, 0.060),
        ],
        paints=[2, 2, 2, 5, 5, 1],
    ),
    dict(
        name="big_house",
        zh="大房子",
        anchor=2,
        pieces=[
            ("wood_block_cube", -0.035, 0.0),
            ("wood_block_cube", 0.035, 0.0),
            ("wood_block_plank", 0.0, 0.030),
            ("wood_block_cube", -0.025, 0.045),
            ("wood_block_cube", 0.025, 0.045),
            ("wood_block_short_plank", 0.0, 0.075),
            ("wood_block_prism", 0.0, 0.090),
        ],
    ),
    dict(
        name="wall",
        zh="城墙",
        anchor=7,
        pieces=[
            ("wood_block_cube", -0.060, 0.0),
            ("wood_block_cube", -0.020, 0.0),
            ("wood_block_cube", 0.020, 0.0),
            ("wood_block_cube", 0.060, 0.0),
            ("wood_block_cube", -0.040, 0.030),
            ("wood_block_cube", 0.0, 0.030),
            ("wood_block_cube", 0.040, 0.030),
            ("wood_block_short_plank", 0.0, 0.060),
        ],
        paints=[2, 1, 2, 1, 5, 5, 5, 0],
    ),
]
MAX_PIECES = max(len(p["pieces"]) for p in PATTERNS)


def paints(pattern):
    """Per piece: its paint (make_wood_blocks.PALETTE index, the asset index of its wood_block_* category)."""
    return pattern.get("paints") or [0] * len(pattern["pieces"])


def groups(pattern):
    """Per piece: an id shared by the pieces that look alike (same shape and paint), so may swap places."""
    keys = [(cat, paint) for (cat, _, _), paint in zip(pattern["pieces"], paints(pattern))]
    return [keys.index(k) for k in keys]


def step_lim(pattern):
    """Actions the task allows: 90 s up to five pieces (as benchmark block_pattern@v1), 20 s more per extra piece."""
    return 2700 + 600 * max(0, len(pattern["pieces"]) - 5)


def labels(pattern):
    return [f"block_{i}" for i in range(len(pattern["pieces"]))]


def targets(pattern):
    """Per piece: bbox centre (x, y, z) in the pattern frame, z from the table, and yaw (deg)."""
    return np.array([(x, 0.0, z + SIZES[cat][2] / 2, 0.0) for cat, x, z in pattern["pieces"]])


def outline(cat, x, z, samples=24):
    """Front-view outline polygon (x, z) of a piece standing at x with its bottom at z."""
    w, _, h = SIZES[cat]
    kind = OUTLINE[cat]
    if kind == "triangle":
        return np.array([(x - w / 2, z), (x + w / 2, z), (x, z + h)])
    if kind == "dome":
        a = np.linspace(0, np.pi, samples)
        return np.c_[x + w / 2 * np.cos(a), z + h * np.sin(a)]
    if kind == "arch":
        a = np.linspace(0, np.pi, samples)
        hole = np.c_[x + ARCH_OPENING * np.cos(a), z + ARCH_OPENING * np.sin(a)]
        return np.r_[[(x - w / 2, z), (x - ARCH_OPENING, z)], hole[::-1], [(x + w / 2, z), (x + w / 2, z + h)],
                     [(x - w / 2, z + h)]]  # fmt: skip
    return np.array([(x - w / 2, z), (x + w / 2, z), (x + w / 2, z + h), (x - w / 2, z + h)])


def overlaps(pattern, gap=0.001):
    """Pairs of pieces whose bboxes in the pattern frame overlap by more than `gap` (a valid pattern has none)."""
    boxes = []
    for (cat, x, z), (_, _, zc, _) in zip(pattern["pieces"], targets(pattern)):
        w, d, h = SIZES[cat]
        boxes.append((np.array([x - w / 2, -d / 2, zc - h / 2]), np.array([x + w / 2, d / 2, zc + h / 2])))
    return [
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if np.all(np.minimum(boxes[i][1], boxes[j][1]) - np.maximum(boxes[i][0], boxes[j][0]) > gap)
    ]


def supported(pattern, min_share=0.3):
    """Pieces resting on the table or on pieces below over at least `min_share` of their footprint width."""
    out = []
    for i, (cat, x, z) in enumerate(pattern["pieces"]):
        if z < 1e-6:
            out.append(True)
            continue
        w = SIZES[cat][0]
        below = 0.0
        for c, xb, zb in pattern["pieces"][:i]:
            if abs(zb + SIZES[c][2] - z) < 0.0015:
                wb = SIZES[c][0]
                below += max(0.0, min(x + w / 2, xb + wb / 2) - max(x - w / 2, xb - wb / 2))
        out.append(below >= min_share * w)
    return out
