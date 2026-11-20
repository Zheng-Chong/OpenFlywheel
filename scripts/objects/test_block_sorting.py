import math

import block_patterns as bp
import block_sorting as bs


def radius(cat):
    w, d, _ = bp.SIZES[cat]
    return math.hypot(w, d) / 2


def test_cards_round_trip():
    for index, (by, key) in enumerate(bs.CARDS):
        assert bs.card(index) == {"by": by, "key": key}
        assert bs.card_index(by, key) == index
        assert bs.label_zh(by, key)
    assert set(bs.COLORS) <= set(bs.PAINTS) - {"natural"}
    assert set(bs.SHAPES) <= set(bp.SIZES)


def test_block_keys():
    blue = bs.PAINTS.index("blue")
    assert bs.block_key("color", "wood_block_cube", blue) == "blue"
    assert bs.block_key("shape", "wood_block_cube", blue) == "wood_block_cube"


def test_slots_fit_the_tray():
    """Every sorting block fits each slot clear of the walls, a tray's blocks clear of each other, and both trays
    within a short reach across the midline."""
    biggest = max(radius(cat) for cat in bs.SHAPES)
    wall = bs.TRAY_WALL
    for box in (0, 1):
        bx, by_ = bs.box_xy(box)
        far, near = (bs.slot_xy(box, slot) for slot in (0, 1))
        for x, y in (far, near):
            assert x == bx and abs(x) <= 0.05
            assert abs(y - by_) + biggest < bs.BOX_SIZE[1] / 2 - wall
        assert biggest < bs.BOX_SIZE[0] / 2 - wall
        assert far[1] > near[1] and math.dist(far, near) > 2 * biggest
    assert bs.PER_CLASS == 2


def test_scatter_and_cards_clear_the_trays():
    biggest = max(radius(cat) for cat in bs.SHAPES)
    assert bs.SCATTER_X[0] - biggest - bs.SCATTER_GAP > bs.BOX_X + bs.BOX_SIZE[0] / 2  # out beside the trays
    assert bs.ACROSS <= 1  # one slot of each tray for a block from across
    assert bs.box_xy(0)[0] + bs.BOX_SIZE[0] / 2 < bs.box_xy(1)[0] - bs.BOX_SIZE[0] / 2  # the trays apart
    assert bs.card_xy(0)[0] + bs.CARD_SIZE[0] / 2 < bs.card_xy(1)[0] - bs.CARD_SIZE[0] / 2  # the cards apart
    for box in (0, 1):
        cx, cy = bs.card_xy(box)
        assert cx == bs.box_xy(box)[0]
        assert cy - bs.CARD_SIZE[1] / 2 > bs.BOX_Y + bs.BOX_SIZE[1] / 2  # beyond the tray
        assert cy + bs.CARD_SIZE[1] / 2 < -0.05  # on the table (its far edge at y ~ -0.01)
