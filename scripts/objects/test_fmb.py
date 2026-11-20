"""FMB import: collider sections, convex pieces, names. No STEP, no OCP, no trimesh, no Isaac."""

import importlib.util
import json
from pathlib import Path

import shapely

spec = importlib.util.spec_from_file_location("fmb", Path(__file__).with_name("fmb.py"))
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def tiles(poly, pieces):
    union = shapely.union_all([shapely.Polygon(p) for p in pieces])
    return (
        poly.symmetric_difference(union).area < 1e-6
        and abs(sum(shapely.Polygon(p).area for p in pieces) - poly.area) < 1e-6
    )


def test_convex_pieces_tile_board_with_holes():
    board = shapely.box(0, 0, 230, 230)
    holes = [shapely.Point(60, 60).buffer(16), shapely.box(140, 40, 190, 80), shapely.Point(115, 170).buffer(20)]
    poly = board.difference(shapely.union_all(holes))
    pieces = F.convex_pieces(poly)
    assert tiles(poly, pieces)
    assert all(F.convex(p) and len(p) <= F.FACE_MAX for p in pieces)


def test_convex_polygon_becomes_wedges():
    disc = shapely.Point(0, 0).buffer(20, quad_segs=32)
    pieces = F.convex_pieces(disc)
    assert tiles(disc, pieces)
    assert len(pieces) <= 5 and all(len(p) <= F.FACE_MAX for p in pieces)


def test_collider_holes_never_tighter():
    hole = shapely.Point(50, 50).buffer(10, quad_segs=64)
    s = F.collider_section(shapely.box(0, 0, 100, 100).difference(hole))
    collider_hole = shapely.Polygon(s.interiors[0])
    assert collider_hole.contains(hole.buffer(-1e-6))


def test_every_asset_named_once():
    words = F.all_words()
    assert len(words) == 9 * 3 * 2 * 8 + 3 + 2 + 3 * 5
    table = json.loads(F.TABLE.read_text(encoding="utf-8"))
    for oid, w in words.items():
        assert table[oid]["en"] == w["en"] and table[oid]["zh"] == w["zh"]
        assert len(w["en"]) == len(w["zh"]) == 3


def test_table_text_round_trips():
    text = F.TABLE.read_text(encoding="utf-8")
    assert F.table_text(json.loads(text)) == text
