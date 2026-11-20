"""Block sorting for g1d_sort_blocks_by_color / g1d_sort_blocks_by_shape (按卡片分拣积木): what goes where, stdlib only.

Two narrow wooden trays (make_block_tray.py) stand side by side in the middle of the table, where both hands reach,
each with a card lying flat behind it (further from the robot) printed with one colour (a swatch) or one block shape
(a grey 3D icon). The painted wood blocks (make_wood_blocks.py, variants 00001-00007) lie out to both sides; each
belongs in the tray whose card matches it: its paint, or its shape. The layout builder (scripts/g1d/build_asset.py),
the trays, the cards (make_sort_cards.py), the tasks' success check and the scripted expert
(scripts/g1d/scripted_demo.py) all read it from here.

Why this way round: across the midline the G1-D's hands neither pick (the wrist cannot turn to the grasp: 7 of 7
missed) nor reach far (they stopped 3 cm past it, 6.5 cm short of a box beyond), and blocks crowded between two big
boxes were knocked into each other and the walls. Here a block is picked by the hand on its side and set into
either tray, reaching at most BOX_X (4.5 cm) across.

Env frame: robot at the table's near edge (y -0.60), facing +Y; tray 0 on its left (-X), tray 1 on its right.
"""

# make_wood_blocks.PALETTE: asset index -> paint
PAINTS = ("natural", "blue", "red", "orange-red", "green", "yellow", "orange", "pink")
PAINT_ZH = {"blue": "蓝色", "red": "红色", "green": "绿色", "yellow": "黄色", "pink": "粉色"}
# sorted by: colours that read apart at a glance (no orange next to red / orange-red), shapes that differ from above
# or from the front, all small enough to drop into a box with the top grasp
COLORS = ("blue", "red", "green", "yellow", "pink")
SHAPES = (
    "wood_block_cube",
    "wood_block_short_cylinder",
    "wood_block_half_cylinder",
    "wood_block_small_prism",
    "wood_block_tile",
)
SHAPE_ZH = {
    "wood_block_cube": "正方体",
    "wood_block_short_cylinder": "圆柱",
    "wood_block_half_cylinder": "半圆柱",
    "wood_block_small_prism": "三棱柱",
    "wood_block_tile": "方片",
}
# card index -> (by, key): the colour swatches, then the shape icons (Rigid/block_sort_card/<index>)
CARDS = [("color", c) for c in COLORS] + [("shape", s) for s in SHAPES]
PER_CLASS = 2  # blocks of each of the two classes

BOX = ("block_tray", 0)  # Blocks library, make_block_tray.py
# m: narrow, so two stand side by side. Blocks are dropped in from just above the walls (open fingers inside the tray
# caught its walls and dragged it), so the walls are low: 2.5 cm, a short drop
BOX_SIZE = (0.07, 0.13, 0.025)
BOX_X, BOX_Y = 0.045, -0.42  # tray centres at (-+BOX_X, BOX_Y): block_patterns.BUILD_XY, both hands reach it
SLOT_DY = 0.03  # m: a tray's two slots, this far beyond (slot 0) and short of (slot 1) its centre
TRAY_WALL, TRAY_FLOOR = 0.005, 0.004  # m: wall and floor thickness
CARD_SIZE = (0.074, 0.105)  # A7 portrait: two fit side by side behind the trays
CARD_GAP = 0.005  # between a tray and its card
CARD_Y = BOX_Y + BOX_SIZE[1] / 2 + CARD_GAP + CARD_SIZE[1] / 2  # beyond the tray, out of the hands' way
# the blocks lie out to both sides, as the pattern task's (build_asset.py SCATTER_X / SCATTER_Y, FLAT_Y): centres |x|
# in SCATTER_X, y in SCATTER_Y (blocks under 2 cm high in FLAT_Y), footprints SCATTER_GAP apart; ACROSS blocks of
# each class or none on the side away from its tray, so a block's side does not tell its tray. Tighter than the
# pattern task's: |x| to 0.18 and flat blocks to y -0.40 (the grasp probe's range), half-cylinders at y -0.374
# (x -0.188, -0.178) slid out of the fingers
SCATTER_X, SCATTER_Y, FLAT_Y, SCATTER_GAP, ACROSS = (0.125, 0.18), (-0.53, -0.37), (-0.45, -0.40), 0.015, 1


def card(index):
    """{"by": "color" | "shape", "key": colour or block category} of card `index`."""
    by, key = CARDS[index]
    return {"by": by, "key": key}


def card_index(by, key):
    return CARDS.index((by, key))


def block_key(by, category, index):
    """What a block is sorted by: its paint (asset index) or its category."""
    return PAINTS[index] if by == "color" else category


def side(box):
    """-1 for the left tray, 1 for the right one."""
    return 2 * box - 1


def box_xy(box):
    return (side(box) * BOX_X, BOX_Y)


def card_xy(box):
    return (side(box) * BOX_X, CARD_Y)


def slot_xy(box, slot):
    """Slot 0 (far) or 1 (near) of tray `box`."""
    x, y = box_xy(box)
    return (x, y + (1 - 2 * slot) * SLOT_DY)


def label_zh(by, key):
    return PAINT_ZH[key] if by == "color" else SHAPE_ZH[key]
