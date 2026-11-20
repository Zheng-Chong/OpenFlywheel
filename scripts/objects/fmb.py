#!/usr/bin/env python3
"""FMB (Functional Manipulation Benchmark) objects as RoboDojo `Rigid` assets, in the FMB object library (Object/FMB/).

Source: the benchmark's CAD files (STEP, mm) from functional-manipulation-benchmark.github.io/files/, licence
CC-BY-4.0 (LICENSE, ATTRIBUTION; written into every metadata.json "source"). Not twins: the geometry is the CAD, the
colours are the benchmark's colour sheet (static/doc/FMB Shape and Color Number Reference Sheet), the mass is the CAD
volume x DENSITY (estimated: printed PLA with partial infill), friction the runtime defaults.

    fmb_peg_<shape>/<index>  the 54 pegs of peg.step: 9 cross-sections (SHAPES, the sheet's order) x 3 sizes x 2
                             lengths (100 / 150 mm), each in the sheet's 8 colours: index = (size * 2 + length) * 8 +
                             colour; colour 0 (dark red) is the base, 1-7 its variants ("variant_of")
    fmb_peg_board/0000s      the three 230 x 230 x 50 mm boards of peg_board.step, through holes for size s; metadata
                             "fmb.holes": where each peg of that size goes in and its clearance
    fmb_peg_fixture/00000, fmb_board_fixture/00000   the fixtures pegs are reoriented on
    fmb_assembly_board/0000a, fmb_assembly_part/<a * 4 + k>   the multi-object assemblies (Board <a + 1>.step): the base
                             board and its four interlocking parts; the base's metadata "fmb.assembly" gives every
                             part's centre in the assembled pose
Multi-lobe pegs (square_circle, double_square, three_prong) stand on their solid grip, lobes up: they go into a board
turned half a turn about X ("inserted_end": "top"); the others go in as they stand.

Usage (no Isaac; needs cadquery-ocp, trimesh, shapely >= 2.1, scipy, networkx, usd-core):
    python scripts/objects/fmb.py fetch DIR                 # the STEP files into DIR
    python scripts/objects/fmb.py convert DIR [RIGID_DIR]   # default Assets/Object/FMB/Rigid
    python scripts/objects/fmb.py names                     # their instruction names into object_names.json
    python scripts/objects/object_names.py apply Assets/Object/FMB
    python scripts/objects/check_object.py Assets/Object/FMB/Rigid/*/*      # on a GPU server

Frame: Z up, metres, origin at bbox centre; every solid keeps its CAD orientation (pegs and boards extruded along Z,
peg cross-sections lined up with their holes).
USD: /root (RigidBodyAPI) -> visual/model (the CAD tessellated, crisp edges between CAD faces) -> collision/hull_NN.
Collision: each solid is cut along one axis (the one its faces are most normal or parallel to) into bands of constant
cross-section; slanted bits (the holes' chamfers, the fixtures' fillets) go into 1 mm bands keeping the narrower end,
and bands that end up alike merge, so chamfers are filled in. Every band's section is split exactly into convex
polygons of <= FACE_MAX points (constrained Delaunay, then Hertel-Mehlhorn merging), extruded into hulls. Outlines
are simplified by at most TOL; hole outlines are pushed out by HOLE_TOL so a collider never makes a hole tighter.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning, J. Luo, C. Xu, F. Liu, L. Tan, "
    "Z. Lin, J. Wu, P. Abbeel, S. Levine (IJRR 2024), functional-manipulation-benchmark.github.io"
)
URL = "https://functional-manipulation-benchmark.github.io/static/files/"
FILES = [
    "peg.step",
    "peg_board.step",
    "peg fixture.step",
    "board fixture.step",
    *(f"Board {a}.step" for a in (1, 2, 3)),
]
RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/FMB/Rigid"
TABLE = Path(__file__).with_name("object_names.json")

DENSITY = 600.0  # kg/m3, estimated: PLA is 1240 solid, FMB's prints are partly infilled
MM = 0.001
TOL = 0.05  # mm, outline simplification
HOLE_TOL = 0.3  # mm, hole outlines: simplified by this, then pushed out by it
STEP_MM = 1.0  # band sampling where a section changes along the axis
FACE_MAX = 32  # PhysX GPU convex limits: points per hull face (a band's polygon), so 64 per hull
FLAT_BAND = 0.0015  # collider points stay on the bottom face or at least this far above it (PhysX plane tolerance)

# the colour sheet: (en, zh, sRGB)
COLORS = [
    ("dark red", "深红色", "#660000"),  # "brown" on the sheet; it renders maroon, beside red and light red
    ("red", "红色", "#FF0000"),
    ("light red", "浅红色", "#E06666"),
    ("yellow", "黄色", "#FFFF00"),
    ("green", "绿色", "#00FF00"),
    ("light blue", "浅蓝色", "#A2C4C9"),
    ("dark blue", "深蓝色", "#0000FF"),
    ("purple", "紫色", "#9900FF"),
]
GREY = "#C8C8C8"  # boards and fixtures, estimated (light grey PLA)
# the sheet's cross-sections: (category suffix, en, zh, lobes at the inserted end)
SHAPES = [
    ("rectangle", "rectangle", "长方形", 1),
    ("round", "round", "圆形", 1),
    ("oval", "oval", "椭圆形", 1),
    ("hexagon", "hexagon", "六边形", 1),
    ("arch", "arch", "拱形", 1),
    ("square_circle", "square and circle", "方圆双柱", 2),
    ("double_square", "double square", "双方柱", 2),
    ("three_prong", "three-prong", "三叉", 3),
    ("star", "star", "五角星形", 1),
]
SIZES = [("large", "大号"), ("medium", "中号"), ("small", "小号")]
LENGTHS = [("short", "短款", 0.100), ("long", "长款", 0.150)]
# the assemblies' parts in Board <a>.step order (solid 0 is the base): kind, colour index
PART_WORDS = {"plate": ("plate", "板件"), "bar": ("bar", "长条"), "pin": ("pin", "销钉"), "block": ("block", "方块")}
PART_COLORS = (1, 3, 4, 6)
ORDINAL = ["first", "second", "third"]


def srgb(hex_rgb):
    c = np.array([int(hex_rgb[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(float(round(v, 4)) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


def peg_id(shape, size, length, color):
    return f"Rigid/fmb_peg_{SHAPES[shape][0]}/{(size * 2 + length) * 8 + color:05d}"


# ---------------------------------------------------------------- STEP
def fetch(step_dir):
    Path(step_dir).mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = Path(step_dir) / name
        if not path.is_file():
            urllib.request.urlretrieve(URL + urllib.parse.quote(name), path)
        print(path, path.stat().st_size)


def solids(path):
    """The STEP file's solids, in file order."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1:  # IFSelect_RetDone
        raise RuntimeError(f"cannot read {path}")
    reader.TransferRoots()
    ex, out = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID), []
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def tessellate(solid, linear=0.05, angular=0.2):
    """(points (V, 3) in m, triangles (F, 3)): one vertex set per CAD face, so normals stay crisp at the edges."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(solid, linear, False, angular, True)
    pts, tris, off = [], [], 0
    ex = TopExp_Explorer(solid, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tf)
                pts.append((p.X(), p.Y(), p.Z()))
            flip = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = (off + k - 1 for k in tri.Triangle(i).Get())
                tris.append((a, c, b) if flip else (a, b, c))
            off += tri.NbNodes()
        ex.Next()
    return np.array(pts) * MM, np.array(tris, dtype=np.int64)


def watertight(pts, tris):
    import trimesh

    m = trimesh.Trimesh(pts / MM, tris)  # mm: sections and areas below are in mm
    m.merge_vertices(digits_vertex=6)
    assert m.is_watertight and m.volume > 0, "CAD tessellation is not a closed outward mesh"
    return m


# ---------------------------------------------------------------- cross-sections and convex pieces
AXES = {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2)}  # cut along axis a: (u, v) in the section, a the band axis


def section(m, axis, h):
    """Shapely (Multi)Polygon of the mesh cut at axis = h, in the AXES (u, v) plane, mm."""
    import shapely

    u, v, w = AXES[axis]
    tf = np.zeros((4, 4))
    tf[0, u] = tf[1, v] = tf[2, w] = tf[3, 3] = 1
    tf[2, 3] = -h
    cut = m.section(plane_origin=np.eye(3)[axis] * h, plane_normal=np.eye(3)[axis])
    if cut is None:
        return shapely.Polygon()
    path, _ = cut.to_2D(to_2D=tf)
    return shapely.union_all(path.polygons_full).buffer(0)


def prism_share(m, axis):
    n = np.abs(m.face_normals[:, axis])
    return float(m.area_faces[(n > 0.999) | (n < 1e-3)].sum() / m.area)


def bands(m, axis):
    """[(lo, hi, section)] along axis (mm): constant sections between the CAD faces normal to the axis; where the
    section changes, STEP_MM bands with the narrower end; alike neighbours merged."""
    lo, hi = m.bounds[:, axis]
    n = np.abs(m.face_normals[:, axis])
    flat = np.round(m.triangles_center[n > 0.999, axis], 4)
    levels = np.unique(np.r_[lo, hi, flat[(flat > lo) & (flat < hi)]])
    levels = levels[np.r_[True, np.diff(levels) > 1e-3]]
    out = []
    for a, b in zip(levels[:-1], levels[1:]):
        eps = min(1e-3, (b - a) / 10)
        ends = section(m, axis, a + eps), section(m, axis, b - eps)
        if alike(*ends):
            out.append([a, b, ends[0]])
            continue
        ts = np.linspace(a, b, max(2, int(np.ceil((b - a) / STEP_MM)) + 1))
        cuts = [section(m, axis, float(np.clip(t, a + eps, b - eps))) for t in ts]
        out += [[t0, t1, s0.intersection(s1)] for t0, t1, s0, s1 in zip(ts[:-1], ts[1:], cuts[:-1], cuts[1:])]
    merged = [out[0]]
    for band in out[1:]:
        if alike(merged[-1][2], band[2]):
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    return [tuple(b) for b in merged if not b[2].is_empty]


def alike(s0, s1):
    return s0.symmetric_difference(s1).area < 0.5 + 1e-3 * max(s0.area, s1.area)  # mm2: drafts of a few hundredths


def collider_section(s):
    """The section as colliders see it: outlines simplified by TOL, holes by HOLE_TOL and then pushed out by it."""
    import shapely

    parts = []
    for poly in getattr(s, "geoms", [s]):
        if poly.area < 1e-3:
            continue
        outer = shapely.Polygon(poly.exterior).simplify(TOL)
        holes = [
            shapely.Polygon(r).simplify(HOLE_TOL).buffer(HOLE_TOL, join_style="mitre", mitre_limit=10)
            for r in poly.interiors
        ]
        parts.append(outer.difference(shapely.union_all(holes)) if holes else outer)
    return shapely.union_all(parts).buffer(0)


def convex_pieces(s, cap=FACE_MAX):
    """Convex CCW polygons (lists of (u, v) mm, <= cap points each) tiling the (Multi)Polygon s exactly: its
    constrained Delaunay triangles merged across their shared edges, longest first, while the union stays convex."""
    import shapely
    from shapely.geometry.polygon import orient

    polys = list(getattr(s, "geoms", [s]))
    if len(polys) > 1:
        return [p for poly in polys for p in convex_pieces(poly, cap)]
    ring = simple([tuple(c) for c in orient(s, 1.0).exterior.coords[:-1]])
    if not s.interiors and convex(ring):  # wedges sharing ring[0]
        step = cap - 2
        return [[ring[0], *ring[i : i + step + 1]] for i in range(1, len(ring) - 1, step)]
    tris = [orient(t, 1.0) for t in shapely.constrained_delaunay_triangles(s).geoms if t.area > 1e-6]
    pieces = {i: [tuple(c) for c in t.exterior.coords[:-1]] for i, t in enumerate(tris)}
    owner = {e: i for i, p in pieces.items() for e in ring_edges(p)}
    shared = sorted({tuple(sorted(e)) for e in owner if e[::-1] in owner}, key=lambda e: -np.hypot(*np.subtract(*e)))
    changed = True
    while changed:
        changed = False
        for a, b in shared:
            i, j = owner.get((a, b)), owner.get((b, a))
            if i is None or j is None or i == j:
                continue
            merged = splice(pieces[i], pieces[j], a, b)
            if not convex(merged) or len(simple(merged)) > cap:
                continue
            for e in ring_edges(pieces[i]) + ring_edges(pieces[j]):
                owner.pop(e, None)
            pieces[i] = merged
            del pieces[j]
            owner.update({e: i for e in ring_edges(merged)})
            changed = True
    out = [simple(p) for p in pieces.values()]
    assert all(len(p) <= cap for p in out), "a triangle has more points than a hull face may"
    return out


def ring_edges(p):
    return list(zip(p, p[1:] + p[:1]))


def splice(p, q, a, b):
    """p (with edge a -> b) and q (with b -> a) joined across that edge."""
    i, j = p.index(b), q.index(a)
    p, q = p[i:] + p[:i], q[j:] + q[:j]  # p: b ... a, q: a ... b
    return p + q[1:-1]


def turns(p):
    q = np.asarray(p)
    d0, d1 = q - np.roll(q, 1, 0), np.roll(q, -1, 0) - q
    return d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]


def convex(p):
    return bool(np.all(turns(p) > -1e-9))


def simple(p):
    """p without its collinear points."""
    t = turns(p)
    return [x for x, c in zip(p, t) if abs(c) > 1e-9]


def hulls_of(m, axis):
    """(hulls in m, collider volume mm3, band count) of the watertight mm mesh m cut along axis."""
    from shapely.geometry import Polygon

    u, v, w = AXES[axis]
    out, volume, cache = [], 0.0, {}
    found = bands(m, axis)
    for a, b, s in found:
        key = s.wkb
        if key not in cache:
            cache[key] = convex_pieces(collider_section(s))
        for poly in cache[key]:
            volume += Polygon(poly).area * (b - a)
            q = np.asarray(poly)
            h = np.zeros((2 * len(q), 3))
            h[:, u], h[:, v] = np.tile(q[:, 0], 2), np.tile(q[:, 1], 2)
            h[:, w] = np.r_[np.full(len(q), a), np.full(len(q), b)]
            out.append(h * MM)
    return out, volume, len(found)


def snap_bottom(hulls, z0):
    """Collider points just above the bottom face (fillets cut along a horizontal axis) moved onto it."""
    out = []
    for h in hulls:
        h = h.copy()
        low = (h[:, 2] > z0) & (h[:, 2] < z0 + FLAT_BAND)
        h[low, 2] = z0
        out.append(np.unique(np.round(h, 9), axis=0))
    return out


# ---------------------------------------------------------------- the catalogue
def pegs_of(step_dir):
    """{(shape, size, length): watertight mm mesh, points, tris} of peg.step, classified by cross-section."""
    found = []
    for solid in solids(Path(step_dir) / "peg.step"):
        pts, tris = tessellate(solid)
        m = watertight(pts, tris)
        found.append((m, pts, tris, shape_of(m)))
    out = {}
    for shape in range(len(SHAPES)):
        for length, (_, _, h) in enumerate(LENGTHS):
            group = [f for f in found if f[3] == shape and abs(np.ptp(f[0].bounds[:, 2]) * MM - h) < 1e-4]
            assert len(group) == len(SIZES), f"{SHAPES[shape][0]} {h} m: {len(group)} pegs, want {len(SIZES)}"
            group.sort(key=lambda f: -f[0].volume)
            for size, f in enumerate(group):
                out[shape, size, length] = f[:3]
    return out


def inserted_end(m):
    """(section, "top" | "bottom") of a peg at the end that goes into the board: the lobed one (multi-lobe pegs have
    a solid grip at the bottom, so they go in turned half a turn about X: the section is mirrored in y about the
    peg's centre to where it lands); single-lobe pegs go in as they stand."""
    import shapely.affinity

    lo, hi = m.bounds[:, 2]
    top, bottom = section(m, 2, hi - 1.0), section(m, 2, lo + 1.0)
    if len(getattr(top, "geoms", [top])) > len(getattr(bottom, "geoms", [bottom])):
        cy = m.bounds[:, 1].mean()
        return shapely.affinity.scale(top, 1, -1, origin=(0, cy)), "top"
    return bottom, "bottom"


def shape_of(m):
    """SHAPES index of a peg from its inserted end's cross-section."""
    import shapely

    s = inserted_end(m)[0]
    lobes = sorted(getattr(s, "geoms", [s]), key=lambda p: -p.area)
    if len(lobes) == 3:
        return 7
    circ = [4 * np.pi * p.area / p.length**2 for p in lobes]
    if len(lobes) == 2:
        return 5 if max(circ) > 0.97 else 6
    p, c = lobes[0], circ[0]
    solidity = p.area / p.convex_hull.area
    rect = np.asarray(shapely.oriented_envelope(p).exterior.coords)
    sides = sorted(np.hypot(*np.diff(rect[:3], axis=0).T))
    if solidity < 0.7:
        return 8  # star 0.68
    if solidity < 0.97:
        return 4  # arch 0.73
    if c > 0.97:
        return 1
    if sides[1] / sides[0] < 1.3:
        return 3
    return 2 if c > 0.88 else 0


def peg_holes(boards, pegs):
    """Per board size: [{peg, center (m, board frame), inserted_end, clearance_mm, collider_clearance_mm}], every hole
    used once."""
    import shapely.affinity

    out = []
    for size, (bm, _, _) in enumerate(boards):
        mid = bm.bounds[:, 2].mean()
        cad = section(bm, 2, mid)
        col = collider_section(cad)
        holes = [shapely.Polygon(r) for poly in getattr(cad, "geoms", [cad]) for r in poly.interiors]
        centre = (bm.bounds[0, :2] + bm.bounds[1, :2]) / 2
        used, rows = set(), []
        for shape in range(len(SHAPES)):
            pm = pegs[shape, size, 0][0]
            end, which = inserted_end(pm)
            lobes = sorted(getattr(end, "geoms", [end]), key=lambda p: -p.area)
            best = None
            for hole in holes:  # the largest lobe centred in some hole
                t = np.subtract(hole.centroid.coords[0], lobes[0].centroid.coords[0])
                moved = [shapely.affinity.translate(p, *t) for p in lobes]
                inside = [next((j for j, h in enumerate(holes) if h.contains(p)), None) for p in moved]
                if None in inside or len(set(inside)) != len(lobes):
                    continue
                spare = sum(holes[j].area for j in inside) - sum(p.area for p in moved)
                if best is None or spare < best[0]:
                    best = (spare, t, inside, moved)
            if best is None:
                raise RuntimeError(f"no hole of board {size} takes the {SHAPES[shape][0]} peg")
            _, t, inside, moved = best
            assert not used & set(inside), f"board {size}: two pegs in one hole"
            used |= set(inside)
            gap = min(holes[j].exterior.distance(p) for j, p in zip(inside, moved))
            peg_col = collider_section(end)
            col_gap = col.boundary.distance(shapely.affinity.translate(peg_col, *t))
            assert not col.intersects(shapely.affinity.translate(peg_col, *t).buffer(-1e-3)), "colliders overlap"
            pc_xy = (pm.bounds[0, :2] + pm.bounds[1, :2]) / 2 + t - centre
            rows.append(
                {
                    "peg": f"fmb_peg_{SHAPES[shape][0]}",
                    "center": [round(float(x) * MM, 5) for x in pc_xy],
                    "inserted_end": which,
                    "clearance_mm": round(float(gap), 2),
                    "collider_clearance_mm": round(float(col_gap), 2),
                }
            )
        assert len(used) == len(holes), f"board {size}: {len(holes) - len(used)} holes take no peg"
        out.append(rows)
    return out


def part_kind(ext):
    d0, d1, d2 = sorted(ext)
    if d2 >= 3 * d1:
        return "pin" if d1 < 0.025 else "bar"
    return "plate" if d1 >= 2.5 * d0 else "block"


def catalogue(step_dir):
    """[item]: id, mesh (watertight mm), pts / tris (visual, m), colour, words, extra metadata."""
    items = []
    pegs = pegs_of(step_dir)
    for (shape, size, length), (m, pts, tris) in sorted(pegs.items()):
        for color in range(len(COLORS)):
            oid = peg_id(shape, size, length, color)
            items.append(
                dict(id=oid, mesh=m, pts=pts, tris=tris, hex=COLORS[color][2], file="peg.step",
                     base=None if color == 0 else peg_id(shape, size, length, 0),
                     words=peg_words(shape, size, length, color))
            )  # fmt: skip
    boards = []
    for solid in solids(Path(step_dir) / "peg_board.step"):
        pts, tris = tessellate(solid)
        boards.append((watertight(pts, tris), pts, tris))
    boards.sort(key=lambda b: b[0].volume)  # the large holes leave the least material
    holes = peg_holes(boards, pegs)
    for size, (m, pts, tris) in enumerate(boards):
        items.append(
            dict(id=f"Rigid/fmb_peg_board/{size:05d}", mesh=m, pts=pts, tris=tris, hex=GREY, file="peg_board.step",
                 base=None, words=board_words(size), extra={"holes": holes[size]})
        )  # fmt: skip
    for cat, name in (("fmb_peg_fixture", "peg fixture.step"), ("fmb_board_fixture", "board fixture.step")):
        (solid,) = solids(Path(step_dir) / name)
        pts, tris = tessellate(solid)
        items.append(
            dict(id=f"Rigid/{cat}/00000", mesh=watertight(pts, tris), pts=pts, tris=tris, hex=GREY, file=name,
                 base=None, words=fixture_words(cat))
        )  # fmt: skip
    for a in range(3):
        name = f"Board {a + 1}.step"
        found = []
        for solid in solids(Path(step_dir) / name):
            pts, tris = tessellate(solid)
            found.append((watertight(pts, tris), pts, tris))
        assert len(found) == 5, f"{name}: {len(found)} solids, want a base and 4 parts"
        centre = (found[0][1].min(0) + found[0][1].max(0)) / 2
        assembly = []
        for k, (m, pts, tris) in enumerate(found[1:]):
            oid = f"Rigid/fmb_assembly_part/{a * 4 + k:05d}"
            kind = part_kind(np.ptp(pts, axis=0))
            c = (pts.min(0) + pts.max(0)) / 2 - centre
            assembly.append({"part": oid, "center": [round(float(x), 5) for x in c]})
            items.append(
                dict(id=oid, mesh=m, pts=pts, tris=tris, hex=COLORS[PART_COLORS[k]][2], file=name, base=None,
                     words=part_words(a, kind, PART_COLORS[k]), kind=kind)
            )  # fmt: skip
        m, pts, tris = found[0]
        items.append(
            dict(id=f"Rigid/fmb_assembly_board/{a:05d}", mesh=m, pts=pts, tris=tris, hex=GREY, file=name, base=None,
                 words=assembly_words(a), extra={"assembly": assembly})
        )  # fmt: skip
    return items


# ---------------------------------------------------------------- words
def peg_words(shape, size, length, color):
    _, shape_en, shape_zh, _ = SHAPES[shape]
    size_en, size_zh = SIZES[size]
    len_en, len_zh, h = LENGTHS[length]
    color_en, color_zh, _ = COLORS[color]
    return dict(
        category=("peg", "插块"),
        en=[f"{shape_en} peg", f"{size_en} {color_en} {shape_en} peg",
            f"{size_en} {len_en} {color_en} {shape_en} peg of the fmb set"],
        zh=[f"{shape_zh}插块", f"{color_zh}{size_zh}{shape_zh}插块",
            f"FMB套件的{color_zh}{size_zh}{len_zh}{shape_zh}插块"],
        caption={
            "color": [color_en, f"{color_en} plastic", f"printed in {color_en}"],
            "material": ["plastic", "3D-printed PLA", "matte printed plastic"],
            "size": [f"{h * 100:.0f} cm long", f"{size_en} cross-section", "fits in one hand"],
            "name": [f"{shape_en} peg", "assembly peg", "FMB peg"],
            "shape": [f"{shape_en} cross-section", "straight prism", "flat ends"],
            "function": ["inserting into a matching hole", "peg-in-hole assembly", "being grasped and reoriented"],
        },
        description=[
            f"{color_en} {shape_en} peg",
            f"3D-printed plastic peg with a {size_en} {shape_en} cross-section",
            f"{len_en} peg {h * 100:.0f} cm long",
            f"straight prism that goes into the {shape_en} hole of a peg board",
            "peg from the Functional Manipulation Benchmark set",
        ],
    )  # fmt: skip


def board_words(size):
    size_en, size_zh = SIZES[size]
    return dict(
        category=("board", "板子"),
        en=["peg board", f"{size_en} hole peg board", f"fmb peg board with nine {size_en} shaped holes"],
        zh=["插孔板", f"{size_zh}孔插孔板", f"FMB套件里带九种{size_zh}异形孔的插孔板"],
        caption={
            "color": ["light grey", "grey plastic", "plain grey"],
            "material": ["plastic", "3D-printed PLA", "matte printed plastic"],
            "size": ["23 cm square", "5 cm thick", "about the size of a plate"],
            "name": ["peg board", "assembly board", "FMB board"],
            "shape": ["square slab", "shaped through holes", "flat top"],
            "function": ["holding inserted pegs", "peg-in-hole assembly", "matching pegs to holes"],
        },
        description=[
            f"grey peg board with {size_en} holes",
            "square 3D-printed plastic slab 23 cm wide and 5 cm thick",
            "nine through holes shaped like rectangles, circles, ovals, hexagons, arches and stars",
            f"each hole takes the {size_en} peg of the same shape",
            "board from the Functional Manipulation Benchmark set",
        ],
    )  # fmt: skip


def fixture_words(cat):
    peg = cat == "fmb_peg_fixture"
    what, zh = ("peg", "插块") if peg else ("board", "板子")
    return dict(
        category=("fixture", "工装"),
        en=["reorienting fixture", f"{what} reorienting fixture", f"fmb {'v-shaped' if peg else 'l-shaped'} fixture "
            f"for reorienting {what}s"],
        zh=["翻转工装", f"{zh}翻转工装", f"FMB套件里给{zh}换向用的{'V形' if peg else 'L形'}工装"],
        caption={
            "color": ["light grey", "grey plastic", "plain grey"],
            "material": ["plastic", "3D-printed PLA", "matte printed plastic"],
            "size": ["about 11 cm wide", "12 cm deep", "palm-sized"],
            "name": ["fixture", f"{what} fixture", "FMB fixture"],
            "shape": ["V-shaped cradle" if peg else "L-shaped rest", "open trough", "flat base"],
            "function": [f"reorienting a {what}", "regrasping", "propping a part at an angle"],
        },
        description=[
            f"grey {what} reorienting fixture",
            f"3D-printed plastic {'V-shaped cradle' if peg else 'L-shaped rest'} with a flat base",
            f"a {what} laid in it can be regrasped in another orientation",
            "it stays on the table while the robot works against it",
            "fixture from the Functional Manipulation Benchmark set",
        ],
    )  # fmt: skip


def part_words(a, kind, color):
    kind_en, kind_zh = PART_WORDS[kind]
    color_en, color_zh, _ = COLORS[color]
    return dict(
        category=("part", "零件"),
        en=[f"{kind_en} part", f"{color_en} {kind_en} part",
            f"{color_en} {kind_en} of the {ORDINAL[a]} fmb interlocking assembly"],
        zh=[f"{kind_zh}零件", f"{color_zh}{kind_zh}零件", f"FMB第{a + 1}套互锁装配里的{color_zh}{kind_zh}"],
        caption={
            "color": [color_en, f"{color_en} plastic", f"printed in {color_en}"],
            "material": ["plastic", "3D-printed PLA", "matte printed plastic"],
            "size": ["about 10 cm" if kind != "bar" else "about 20 cm long", "hand-sized", "a few centimetres thick"],
            "name": [f"{kind_en} part", "interlocking part", "FMB assembly part"],
            "shape": [kind_en, "notched to interlock", "flat faces"],
            "function": ["assembling in order", "slotting into the base board", "interlocking with the other parts"],
        },
        description=[
            f"{color_en} interlocking {kind_en}",
            f"3D-printed plastic {kind_en} with notches that lock into the other parts",
            f"one of the four parts of the {ORDINAL[a]} multi-object assembly",
            "the parts go onto the base board in a fixed order",
            "part from the Functional Manipulation Benchmark set",
        ],
    )  # fmt: skip


def assembly_words(a):
    return dict(
        category=("board", "板子"),
        en=["assembly board", f"{ORDINAL[a]} assembly board", f"base board of the {ORDINAL[a]} fmb interlocking "
            "assembly"],
        zh=["装配底板", f"{a + 1}号装配底板", f"FMB第{a + 1}套互锁装配的底板"],
        caption={
            "color": ["light grey", "grey plastic", "plain grey"],
            "material": ["plastic", "3D-printed PLA", "matte printed plastic"],
            "size": ["23 cm square", "5 cm thick", "about the size of a plate"],
            "name": ["assembly board", "base board", "FMB assembly board"],
            "shape": ["square slab", "slots on top", "flat bottom"],
            "function": ["holding the assembled parts", "multi-step assembly", "fixing the parts' places"],
        },
        description=[
            f"grey base board of the {ORDINAL[a]} assembly",
            "square 3D-printed plastic slab 23 cm wide with slots on top",
            "four interlocking parts slot into it one after another",
            "the slots fix where every part goes",
            "board from the Functional Manipulation Benchmark set",
        ],
    )  # fmt: skip


def all_words():
    """{id: words} of every asset, from the catalogue's structure alone (no STEP needed)."""
    out = {}
    for shape in range(len(SHAPES)):
        for size in range(len(SIZES)):
            for length in range(len(LENGTHS)):
                for color in range(len(COLORS)):
                    out[peg_id(shape, size, length, color)] = peg_words(shape, size, length, color)
    for size in range(len(SIZES)):
        out[f"Rigid/fmb_peg_board/{size:05d}"] = board_words(size)
    for cat in ("fmb_peg_fixture", "fmb_board_fixture"):
        out[f"Rigid/{cat}/00000"] = fixture_words(cat)
    for a, kinds in enumerate(ASSEMBLY_KINDS):
        out[f"Rigid/fmb_assembly_board/{a:05d}"] = assembly_words(a)
        for k, kind in enumerate(kinds):
            out[f"Rigid/fmb_assembly_part/{a * 4 + k:05d}"] = part_words(a, kind, PART_COLORS[k])
    return out


# part_kind() of every assembly's parts in file order, checked against the CAD by convert
ASSEMBLY_KINDS = [
    ("plate", "bar", "plate", "plate"),
    ("plate", "plate", "pin", "pin"),
    ("bar", "bar", "plate", "plate"),
]


def names():
    """Write every asset's instruction names into object_names.json."""
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    for oid, w in all_words().items():
        table[oid] = {"category": dict(zip(("en", "zh"), w["category"])), "en": w["en"], "zh": w["zh"]}
    TABLE.write_text(table_text(table), encoding="utf-8")
    print(f"{TABLE}: {len(all_words())} FMB entries")


def table_text(table):
    """object_names.json as it is kept: one line per category / language, aliases on one line."""

    def one(v):
        return json.dumps(v, ensure_ascii=False)

    rows = []
    for oid, v in table.items():
        if isinstance(v, str):
            rows.append(f"    {one(oid)}: {one(v)}")
        else:
            inner = ",\n".join(f"        {one(k)}: {one(x)}" for k, x in v.items())
            rows.append(f"    {one(oid)}: {{\n{inner}\n    }}")
    return "{\n" + ",\n".join(rows) + "\n}\n"


# ---------------------------------------------------------------- USD and metadata
def convert(step_dir, rigid_dir):
    import make_pitcher as mp
    import make_plastic_cups as pc

    items = catalogue(step_dir)
    for it in items:
        if "kind" in it:
            a, k = divmod(int(it["id"][-5:]), 4)
            assert it["kind"] == ASSEMBLY_KINDS[a][k], f"{it['id']}: {it['kind']}, ASSEMBLY_KINDS says otherwise"
    cache = {}
    for it in items:
        key = id(it["mesh"])
        if key not in cache:
            cache[key] = colliders(it)
        hulls, stats = cache[key]
        d, meta = save(mp, pc, rigid_dir, it, hulls)
        print(f"{it['id']}: {meta['physics']['size']} m, {meta['physics']['mass'] * 1000:.0f} g (est.), {stats}")
    sys.stdout.flush()  # APP.close() exits without flushing a redirected stdout


def colliders(it):
    from scipy.spatial import ConvexHull

    m = it["mesh"]
    axis = max(range(3), key=lambda a: (prism_share(m, a), a == 2))
    hulls, volume, n_bands = hulls_of(m, axis)
    hulls = snap_bottom(hulls, it["pts"][:, 2].min())
    ratio = volume / m.volume
    assert 0.95 < ratio < 1.01, f"{it['id']}: colliders hold {ratio:.3f} of the CAD volume"  # > 1: chords of concave
    z0 = it["pts"][:, 2].min()
    for h in hulls:
        eq = ConvexHull(h).equations
        assert len(h) <= 2 * FACE_MAX, f"{it['id']}: hull of {len(h)} points"
        on_face = (np.abs(h @ eq[:, :3].T + eq[:, 3]) < 1e-7).sum(0)
        assert on_face.max() <= FACE_MAX, f"{it['id']}: hull face of {on_face.max()} points"
        z = h[:, 2] - z0
        assert np.all((z < 1e-7) | (z > FLAT_BAND - 1e-7)), f"{it['id']}: collider point just above the bottom face"
    return hulls, f"{len(hulls)} hulls in {n_bands} bands along {'xyz'[axis]}, {ratio:.3f} of the CAD volume"


def write_usdz(mp, pc, out, pts, tris, hex_rgb, hulls):
    from make_pitcher import Usd, UsdGeom, UsdPhysics, UsdShade

    tmp = Path(tempfile.mkdtemp()) / "object.usdc"
    st = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = mp.add_xform(st, "/root")
    st.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mp.add_xform(st, "/root/visual")
    mesh = mp.add_mesh(st, "/root/visual/model", pts, tris.tolist(), mp.vertex_normals(pts, tris))
    rgb = srgb(hex_rgb)
    look = mp.add_material(
        st,
        "/root/_materials/Plastic",
        "OmniPBR",
        {"diffuse_color_constant": rgb, "reflection_roughness_constant": 0.55, "metallic_constant": 0.0},
        rgb,
        1.0,
        0.55,
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)
    pc.add_collision(st, hulls)
    st.Save()
    zip_writer = getattr(mp.Sdf, "ZipFileWriter", None) or Usd.ZipFileWriter
    zw = zip_writer.CreateNew(str(out))
    zw.AddFile(str(tmp), "object.usdc")
    zw.Save()


def save(mp, pc, rigid_dir, it, hulls):
    pts = it["pts"]
    lo, hi = pts.min(0), pts.max(0)
    shift = -(lo + hi) / 2
    pts, hulls = pts + shift, [h + shift for h in hulls]
    assert mp.signed_volume(pts, it["tris"]) > 0, f"{it['id']}: visual mesh faces inward"
    _, cat, idx = it["id"].split("/")
    d = Path(rigid_dir) / cat / idx
    d.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{cat}/{int(idx)}"))
    write_usdz(mp, pc, d / "object.usdz", pts, it["tris"], it["hex"], hulls)
    ext = hi - lo
    mass = round(DENSITY * it["mesh"].volume * MM**3, 4)
    meta = pc.metadata(uid, dict(pts=pts, faces=it["tris"], contact_r=float(min(ext[:2]) / 2)), mass, 0 * shift)
    meta["measurements_mm"] = {k: round(float(v) / MM, 1) for k, v in zip(("长", "宽", "高"), ext)}
    meta["source"] = {"dataset": "FMB", "file": it["file"], "license": LICENSE, "attribution": ATTRIBUTION}
    if it.get("extra"):
        meta["fmb"] = it["extra"]
    if it["base"]:
        meta["variant_of"] = it["base"]
    (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False))
    w = it["words"]
    (d / "description.json").write_text(
        json.dumps({"uuid": uid, "caption": w["caption"], "description": w["description"]}, indent=4)
    )
    return d, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch").add_argument("step_dir")
    c = sub.add_parser("convert")
    c.add_argument("step_dir")
    c.add_argument("rigid_dir", nargs="?", default=RIGID_DIR)
    sub.add_parser("names")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.step_dir)
    elif args.command == "names":
        names()
    else:
        convert(args.step_dir, args.rigid_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import make_pitcher as mp

        if mp.APP is None:
            raise
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if "make_pitcher" in sys.modules and sys.modules["make_pitcher"].APP is not None:
        sys.modules["make_pitcher"].APP.close()
