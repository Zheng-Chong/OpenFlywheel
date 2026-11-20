#!/usr/bin/env python3
"""Solid-wood toy blocks (彩色实木积木) as RoboDojo `Rigid` assets: every shape of the set in every colour.

Sources: the product photo of the set (user, 2026-10-31, real2sim/objects/wood_blocks_product_20261031.jpg; nominal
sizes printed on it, colours taken from it) and the Real2Sim platform registrations of three lab blocks of that set,
whose measured values replace the nominal ones:
- Rigid/cube_building_blocks/00000 (数采组, 2026-10-30 11:01:00): 30 x 30 x 30 mm.
- Rigid/triangular_prism_building_block/00000 (杨开元, 2026-10-29 22:59:06): legs 40.24 mm, thickness 29.27 mm, 20.1 g.
- Rigid/wooden_arch_block/00000 (杨开元, 2026-10-30 09:56:02): 60.10 x 29.72 x 30.04 mm, 9.58 mm over the opening,
  24.6 g.
Those registrations keep their own ids and assets; this is a separate family (wood_block_*), user decision 2026-10-31.
These blocks are not twins, so they live in the Blocks object library (Object/Blocks/), not Object/Real2Sim/.
Shape and colour are separate: per shape (one category each), index 00000 is unpainted wood with a beech grain texture
and 00001-00007 are its generalization variants (metadata "variant_of") painted in the photo's seven colours, same
geometry and mass. Tasks may also recolour any of them per layout (scripts/g1d/build_asset.py RANDOMIZE "color").
Unmeasured masses are mesh volume x the mean density of the two weighed blocks. Edge radii, the small prism's height
and the half-cylinder's height are estimated from the photo. Friction is unmeasured: runtime defaults.

Usage:
    python scripts/objects/make_wood_blocks.py [RIGID_DIR]    # default Assets/Object/Blocks/Rigid
    python scripts/objects/object_names.py apply Assets/Object/Blocks
    python scripts/objects/check_object.py Assets/Object/Blocks/Rigid/wood_block_*/0000*
Writes RIGID_DIR/wood_block_<shape>/<index>/{object.usdz, metadata.json, description.json}. Needs what
make_pitcher.py needs, and PIL.

Frame: Z up, metres, origin at bbox centre. Every block is a 2D profile in the X-Z plane extruded along Y (cylinders:
a circle extruded along Z), with rounded profile corners (FILLET) and rounded edges round both end faces (BEVEL).
Long sides along X; prisms and the half-cylinder lie on their widest flat face, the arch stands on its two feet.
USD:   /root (RigidBodyAPI) -> visual/model -> collision/hull_NN (one convex hull; the arch: wedges round its opening);
       the unpainted ones pack textures/wood.png, mapped by face-varying `st`.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pitcher as mp  # noqa: E402  (shared USD helpers; starts Kit when pxr needs it)
from make_pitcher import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
import make_plastic_cups as pc  # noqa: E402  (add_collision, metadata)

RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/Blocks/Rigid"
FILLET = 0.0015  # radius of the edges along the extrusion, estimated from the photos
BEVEL = 0.0012  # radius of the edges round the end faces, estimated; < FILLET keeps the end faces simple polygons
ARC_STEP = np.radians(15)  # visual sampling of the fillets
N_ROUND = 96  # visual points on a circle; a multiple of 4 puts points on the extremes, so the extents are exact
N_BEVEL = 5  # rings per bevel quarter circle
N_OPENING = 39  # visual points on the arch opening; odd puts one on the crown
N_HALF = 25  # inner points of the half-cylinder's arc
ARCH_WEDGES = 12  # collision pieces round the arch opening; 15 deg each cuts at most 0.18 mm into the opening
HULL_MAX = 64  # PhysX GPU convex limits: vertices per hull, vertices per hull face
FACE_MAX = 32
FLAT_BAND = 0.0015  # collider points stay on the bottom face or at least this far above it (PhysX plane tolerance)
U45 = np.array([1.0, 1.0]) / np.sqrt(2)  # along a prism leg (right isosceles, lying on the hypotenuse)


def srgb(hex_rgb):
    c = np.array([int(hex_rgb[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(float(round(v, 4)) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


# asset index -> colour: 00000 unpainted wood, then the product photo's paints (sRGB, darkened so check_object's bright
# preview lighting renders them like the photo)
PALETTE = [
    ("natural", None),
    ("blue", "#1673D0"),
    ("red", "#E01E24"),
    ("orange-red", "#E8401A"),
    ("green", "#62B81E"),
    ("yellow", "#F5D010"),
    ("orange", "#EE6A12"),
    ("pink", "#E43C98"),
]
WOOD = ("#8E6652", "#76523F", "#553526")  # sRGB beech, from the unpainted lab arch block: base, grain lines, ray flecks
WOOD_TEX = "textures/wood.png"
TEX_M = 0.06  # wood covered by one texture tile
TEX_PX = 1024  # 17 px/mm
# kind and dims (m): box (x, y, z); cylinder (diameter, height); half_cylinder (diameter, length);
# prism (fit, target, thickness): right isosceles lying on its hypotenuse, sized so a leg or the hypotenuse side
# measures target; arch (length, width, height, thickness over the opening).
# target: measurements_mm the model must reproduce (within TOL_MM); mass: measured kg, else None (density estimate).
SHAPES = [
    dict(cat="wood_block_plank", kind="box", dims=(0.090, 0.030, 0.015), mass=None,
         target={"长": 90, "宽": 30, "高": 15}, src="标称",
         words=dict(name="plank block", shape="long flat bar", size="9 cm long, 3 cm wide, 1.5 cm thick")),
    dict(cat="wood_block_short_plank", kind="box", dims=(0.060, 0.030, 0.015), mass=None,
         target={"长": 60, "宽": 30, "高": 15}, src="标称",
         words=dict(name="plank block", shape="short flat bar", size="6 cm long, 3 cm wide, 1.5 cm thick")),
    dict(cat="wood_block_tile", kind="box", dims=(0.030, 0.030, 0.015), mass=None,
         target={"长": 30, "宽": 30, "高": 15}, src="标称",
         words=dict(name="square tile block", shape="thick square tile", size="3 cm square, 1.5 cm thick")),
    dict(cat="wood_block_cuboid", kind="box", dims=(0.060, 0.030, 0.030), mass=None,
         target={"长": 60, "宽": 30, "高": 30}, src="标称",
         words=dict(name="cuboid block", shape="rectangular box", size="6 cm long, 3 cm wide, 3 cm high")),
    dict(cat="wood_block_cube", kind="box", dims=(0.030, 0.030, 0.030), mass=None,
         target={"长": 30, "宽": 30, "高": 30}, src="实测（cube_building_blocks/00000）",
         words=dict(name="cube block", shape="cube", size="3 cm on each side")),
    dict(cat="wood_block_cylinder", kind="cylinder", dims=(0.030, 0.060), mass=None,
         target={"直径": 30, "高": 60}, src="标称",
         words=dict(name="cylinder block", shape="tall upright cylinder", size="3 cm across, 6 cm high")),
    dict(cat="wood_block_short_cylinder", kind="cylinder", dims=(0.030, 0.030), mass=None,
         target={"直径": 30, "高": 30}, src="标称",
         words=dict(name="cylinder block", shape="short upright cylinder", size="3 cm across, 3 cm high")),
    dict(cat="wood_block_arch", kind="arch", dims=(0.06010, 0.02972, 0.03004, 0.00958),
         mass=0.0246, target={"长": 60.10, "宽": 29.72, "高": 30.04, "最薄处": 9.58},
         src="实测（wooden_arch_block/00000）",
         words=dict(name="arch block", shape="bridge with a half-round opening underneath",
                    size="6 cm long, 3 cm wide, 3 cm high")),
    dict(cat="wood_block_prism", kind="prism", dims=("leg", 0.04024, 0.02927), mass=0.0201,
         target={"直角边": 40.24, "厚": 29.27}, src="实测（triangular_prism_building_block/00000）",
         words=dict(name="triangular prism block", shape="right-angled roof ridge",
                    size="about 6 cm wide, 3 cm high, 3 cm thick")),
    dict(cat="wood_block_small_prism", kind="prism", dims=("base", 0.040, 0.030), mass=None,
         target={"底边": 40, "厚": 30}, src="标称；高按直角等腰估计",
         words=dict(name="triangular prism block", shape="small right-angled roof ridge",
                    size="4 cm wide, 2 cm high, 3 cm thick")),
    dict(cat="wood_block_half_cylinder", kind="half_cylinder", dims=(0.030, 0.030), mass=None,
         target={"直径": 30, "长": 30}, src="标称；高按半圆估计",
         words=dict(name="half-cylinder block", shape="half-round dome lying on its flat side",
                    size="3 cm across, 3 cm long, 1.5 cm high")),
]  # fmt: skip
TOL_MM = 0.5


# ---------------------------------------------------------------- profiles
# A profile is a list of arcs (centre, angles, radius, sign) traced CCW (solid on the left); consecutive arcs are
# joined by straight edges. ring(arcs, d) is the profile shrunk by d: convex corners (sign +1) lose d of radius,
# the arch opening (sign -1) gains it. The point count does not depend on d, so rings stack into a mesh.
def ring(arcs, d):
    return np.concatenate([c + (r - s * d) * np.c_[np.cos(a), np.sin(a)] for c, a, r, s in arcs])


def arc(centre, a0, a1, r, n=None):
    n = n or max(2, int(np.ceil(abs(a1 - a0) / ARC_STEP)) + 1)
    return (np.asarray(centre, float), np.linspace(a0, a1, n), r, 1)


def inset(poly, f):
    """Convex CCW polygon with every edge moved in by f."""
    e = np.roll(poly, -1, 0) - poly
    n = np.c_[e[:, 1], -e[:, 0]] / np.linalg.norm(e, axis=1, keepdims=True)  # outward edge normals
    c = np.einsum("ij,ij->i", n, poly) - f
    return np.array([np.linalg.solve(n[[i - 1, i]], c[[i - 1, i]]) for i in range(len(poly))])


def rounded(q, rho):
    """Convex CCW polygon q grown by rho: one corner arc per vertex, from the incoming to the outgoing edge normal."""
    e = np.roll(q, -1, 0) - q
    ang = np.arctan2(-e[:, 0], e[:, 1])
    return [arc(q[i], ang[i - 1], ang[i - 1] + (ang[i] - ang[i - 1]) % (2 * np.pi), rho) for i in range(len(q))]


def convex_hull2d(arcs):
    """<= FACE_MAX collider points of a convex profile: both ends of every real corner (they lie on the flat faces,
    so a face never gets points just off it) and the middle of the small arcs that sample a curve."""
    sharp, smooth = [], []
    for c, a, r, _ in arcs:
        if a[-1] - a[0] > np.radians(20):
            sharp += [c + r * np.r_[np.cos(t), np.sin(t)] for t in (a[0], a[-1])]
        else:
            t = a.mean()
            smooth.append(c + r * np.r_[np.cos(t), np.sin(t)])
    step = int(np.ceil(len(smooth) / max(1, FACE_MAX - len(sharp)))) if smooth else 1
    return [np.array(sharp + smooth[::step])]


def box_profile(x, z):
    rect = np.array([(-x / 2, 0.0), (x / 2, 0.0), (x / 2, z), (-x / 2, z)])
    return rounded(inset(rect, FILLET), FILLET)


def circle_profile(radius):
    a = np.linspace(0, 2 * np.pi, N_ROUND, endpoint=False)
    hull = radius * np.c_[np.cos(a[:: N_ROUND // FACE_MAX]), np.sin(a[:: N_ROUND // FACE_MAX])]
    return [(np.zeros(2), a, radius, 1)], [hull]


def half_cylinder_profile(radius):
    """Half disc on its flat side; the corners where the flat meets the curve are rounded (FILLET)."""
    t = np.arcsin(FILLET / (radius - FILLET))
    a = np.linspace(t, np.pi - t, N_HALF)
    return rounded((radius - FILLET) * np.c_[np.cos(a), np.sin(a)], FILLET)


def prism_profile(h):
    """Right isosceles triangle on its hypotenuse (half-width h, apex height h), corners rounded."""
    return rounded(inset(np.array([(-h, 0.0), (h, 0.0), (0.0, h)]), FILLET), FILLET)


def fit_prism(fit, target):
    """Half-width whose rounded profile measures target along a leg (fit 'leg') or across the base ('base')."""
    k = np.sqrt(2) if fit == "leg" else 2.0
    h = target / k
    for _ in range(10):
        r = ring(prism_profile(h), 0.0)
        h += (target - np.ptp(r @ U45 if fit == "leg" else r[:, 0])) / k
    return h


def arch_profile(length, height, thick):
    """[-A, A] x [0, H] minus the half disc of radius R = H - thick on the bottom centre (the opening); outer corners
    rounded with FILLET, the feet (where the bottom meets the opening) with a fillet tangent to both."""
    a, h, r, f, g = length / 2, height, height - thick, FILLET, FILLET
    s = np.sqrt((r + g) ** 2 - g**2)
    t = np.arctan2(g, s)  # angle of the opening where the foot fillets touch it
    return [
        arc((-a + f, f), np.pi, 1.5 * np.pi, f),
        arc((-s, g), -0.5 * np.pi, -t, g),
        (np.zeros(2), np.linspace(np.pi - t, t, N_OPENING + 2)[1:-1], r, -1),  # ends are the fillets' ends
        arc((s, g), -np.pi + t, -0.5 * np.pi, g),
        arc((a - f, f), -0.5 * np.pi, 0.0, f),
        arc((a - f, h - f), 0.0, 0.5 * np.pi, f),
        arc((-a + f, h - f), 0.5 * np.pi, np.pi, f),
    ]


def arch_hulls2d(length, height, thick):
    """Wedges of the (sharp-cornered) solid between rays from the opening's centre: each is the two opening points
    and where the rays leave the block, plus a top corner between them. The opening stays open."""
    a, h, r = length / 2, height, height - thick

    def leave(th):
        c, s = np.cos(th), np.sin(th)
        return min(a / abs(c) if abs(c) > 1e-12 else np.inf, h / s if s > 1e-12 else np.inf) * np.r_[c, s]

    hulls = []
    th = np.linspace(0, np.pi, ARCH_WEDGES + 1)
    for t0, t1 in zip(th, th[1:]):
        pts = [r * np.r_[np.cos(t0), np.sin(t0)], r * np.r_[np.cos(t1), np.sin(t1)], leave(t0), leave(t1)]
        pts += [p for p in (np.r_[a, h], np.r_[-a, h]) if t0 + 1e-9 < np.arctan2(p[1], p[0]) < t1 - 1e-9]
        hulls.append(np.array(pts))
    return hulls


# ---------------------------------------------------------------- mesh
def ear_clip(p):
    """Triangles of a simple CCW polygon."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    idx, tris = list(range(len(p))), []
    while len(idx) > 3:
        n = len(idx)
        for k in range(n):
            i, j, m = idx[k - 1], idx[k], idx[(k + 1) % n]
            if cross(p[i], p[j], p[m]) <= 0:
                continue
            inside = (
                cross(p[i], p[j], p[o]) > 0 and cross(p[j], p[m], p[o]) > 0 and cross(p[m], p[i], p[o]) > 0
                for o in idx
                if o not in (i, j, m)
            )
            if not any(inside):
                tris.append([i, j, m])
                idx.pop(k)
                break
        else:
            raise AssertionError("end face is not a simple polygon")
    return tris + [idx]


def extrude(arcs, length):
    """Closed mesh of the profile from w = -length/2 to +length/2, both end edges rounded with BEVEL. (u, v, w)."""
    half, phi = length / 2, np.linspace(0, np.pi / 2, N_BEVEL)
    levels = [(-(half - BEVEL + BEVEL * np.sin(p)), BEVEL * (1 - np.cos(p))) for p in phi[::-1]]
    levels += [(half - BEVEL + BEVEL * np.sin(p), BEVEL * (1 - np.cos(p))) for p in phi]
    rings = [ring(arcs, d) for _, d in levels]
    n = len(rings[0])
    pts = np.concatenate([np.c_[r, np.full(n, w)] for r, (w, _) in zip(rings, levels)])
    faces = [
        [k * n + j, k * n + (j + 1) % n, (k + 1) * n + (j + 1) % n, (k + 1) * n + j]
        for k in range(len(levels) - 1)
        for j in range(n)
    ]
    cap, last = ear_clip(rings[0]), (len(levels) - 1) * n
    faces += [t[::-1] for t in cap] + [[last + i for i in t] for t in cap]
    return pts, faces


def to_xyz(p, axis):
    """(u, v, w) -> (X, Y, Z): profile in X-Z extruded along -Y (a proper rotation), or in X-Y along Z."""
    return np.c_[p[:, 0], -p[:, 2], p[:, 1]] if axis == "y" else np.asarray(p)


def build(b):
    kind, dims = b["kind"], b["dims"]
    axis = "y"
    if kind == "box":
        length, arcs = dims[1], box_profile(dims[0], dims[2])
    elif kind == "cylinder":
        axis, length = "z", dims[1]
        arcs, hulls2d = circle_profile(dims[0] / 2)
    elif kind == "half_cylinder":
        length, arcs = dims[1], half_cylinder_profile(dims[0] / 2)
    elif kind == "prism":
        fit, target, length = dims
        arcs = prism_profile(fit_prism(fit, target))
    else:
        length, arcs = dims[1], arch_profile(dims[0], dims[2], dims[3])
        hulls2d = arch_hulls2d(dims[0], dims[2], dims[3])
    if kind in ("box", "half_cylinder", "prism"):
        hulls2d = convex_hull2d(arcs)
    pts, faces = extrude(arcs, length)
    pts = to_xyz(pts, axis)
    assert mp.signed_volume(pts, faces) > 0, f"{b['cat']}: inward-facing winding"
    hulls = [to_xyz(np.concatenate([np.c_[h, np.full(len(h), w)] for w in (-length / 2, length / 2)]), axis)
             for h in hulls2d]  # fmt: skip
    return dict(pts=pts, faces=faces, hulls=hulls, kind=kind, dims=dims)


def measure(c):
    """measurements_mm from the mesh, by the definitions of the names in SHAPES."""
    p = c["pts"]
    ext = np.ptp(p, axis=0) * 1000
    kind = c["kind"]
    if kind == "box":
        m = {"长": ext[0], "宽": ext[1], "高": ext[2]}
    elif kind == "cylinder":
        m = {"直径": ext[0], "高": ext[2]}
    elif kind == "half_cylinder":
        m = {"直径": ext[0], "长": ext[1], "高": ext[2]}
    elif kind == "prism":
        m = {"底边": ext[0], "直角边": np.ptp(p[:, [0, 2]] @ U45) * 1000, "高": ext[2], "厚": ext[1]}
    else:  # the crown of the opening is the lowest mesh point on the mid plane x = 0
        crown = p[np.abs(p[:, 0] - (p[:, 0].min() + p[:, 0].max()) / 2) < 1e-6, 2].min()
        m = {"长": ext[0], "宽": ext[1], "高": ext[2], "最薄处": (p[:, 2].max() - crown) * 1000}
    return {k: round(float(v), 1) for k, v in m.items()}


def self_check(b, c):
    pts, hulls = c["pts"], c["hulls"]
    eqs = [ConvexHull(h).equations for h in hulls]
    for h, eq in zip(hulls, eqs):
        assert len(h) <= HULL_MAX, f"{b['cat']}: hull of {len(h)} points"
        on_face = (np.abs(h @ eq[:, :3].T + eq[:, 3]) < 1e-7).sum(0)
        assert on_face.max() <= FACE_MAX, f"{b['cat']}: hull face of {on_face.max()} points"
        z = h[:, 2] - pts[:, 2].min()
        assert np.all((z < 1e-7) | (z > FLAT_BAND - 1e-7)), f"{b['cat']}: collider point just above the bottom face"
    # every visual point within 1.5 mm of some collider; colliders at most 1 mm past the visual's convex hull
    gap = np.min([(pts @ eq[:, :3].T + eq[:, 3]).max(1) for eq in eqs], axis=0).max()
    vis = ConvexHull(pts).equations
    over = max((h @ vis[:, :3].T + vis[:, 3]).max() for h in hulls)
    assert gap < 0.0015 and over < 0.001, f"{b['cat']}: collider off the mesh ({gap * 1e3:.2f}, {over * 1e3:.2f} mm)"
    if c["kind"] == "arch":  # wedge chords cut into the opening by at most the sagitta; the opening stays open
        r = c["dims"][2] - c["dims"][3]
        assert r * (1 - np.cos(np.pi / ARCH_WEDGES / 2)) < 0.0003
        assert min(np.linalg.norm(h[:, [0, 2]], axis=1).min() for h in hulls) > r - 1e-9
    got = measure(c)
    bad = {k: (got[k], v) for k, v in b["target"].items() if abs(got[k] - v) > TOL_MM}
    assert not bad, f"{b['cat']}: measurements off target {bad}"
    return got, float(gap)


# ---------------------------------------------------------------- wood texture
def wood_texture(path, seed=0):
    """Tileable beech grain, TEX_PX square for TEX_M of wood, grain along the image x axis: broad streaks, fine dark
    lines and the short dark ray flecks of beech. Periodic by construction (filtered FFT noise, wrapped flecks)."""
    from PIL import Image  # only here: the geometry needs no PIL

    rng = np.random.default_rng(seed)
    n = TEX_PX
    f = np.fft.fftfreq(n) * n  # cycles per tile
    fu, fv = np.meshgrid(f, f)  # along x (grain), along y (across)

    def noise(cu, cv):
        x = np.real(np.fft.ifft2(np.exp(-((fu / cu) ** 2 + (fv / cv) ** 2)) * np.fft.fft2(rng.standard_normal((n, n)))))
        return (x - x.mean()) / x.std()

    tone = 0.6 * noise(1.5, 8) + 0.4 * noise(4, 40)
    lines = np.clip(noise(6, 220), 0, None) ** 2
    lines /= lines.max()
    flecks = np.zeros((n, n))
    px = TEX_PX / (TEX_M * 1000)  # px per mm
    for _ in range(550):
        length, height = rng.uniform(1.0, 3.5) * px, rng.uniform(0.08, 0.2) * px
        x0, y0, depth = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0.4, 1.0)
        xs, ys = (
            np.arange(int(x0 - length / 2), int(x0 + length / 2) + 1),
            np.arange(int(y0 - height), int(y0 + height) + 1),
        )
        shape = np.clip(1 - ((xs[None] - x0) / (length / 2)) ** 2 - ((ys[:, None] - y0) / height) ** 2, 0, 1)
        flecks[np.ix_(ys % n, xs % n)] = np.maximum(flecks[np.ix_(ys % n, xs % n)], depth * np.sqrt(shape))
    base, line, fleck = (np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)]) / 255 for h in WOOD)
    img = base * (1 + 0.05 * tone[..., None])
    img += (line - img) * 0.6 * lines[..., None]
    img += (fleck - img) * 0.95 * flecks[..., None]
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path)
    return path


def wood_uv(pts, faces, kind, offset):
    """Face-varying texture coordinates (in tiles): each face projected along its main normal axis, so the grain runs
    along X on the sides and top; a cylinder's side is unrolled round Z with the grain along its axis, a whole number
    of tiles round so the seam matches."""
    turns = max(1, round(2 * np.pi * np.linalg.norm(pts[:, :2], axis=1).max() / TEX_M))  # tiles round a cylinder
    uv = []
    for f in faces:
        p = pts[f]
        normal = np.sum(np.cross(p, np.roll(p, -1, 0)), axis=0)  # Newell: fine for any planar polygon
        axis = int(np.argmax(np.abs(normal)))
        if kind == "cylinder" and axis != 2:
            q = np.c_[p[:, 2] / TEX_M, np.unwrap(np.arctan2(p[:, 1], p[:, 0])) / (2 * np.pi) * turns]
        else:
            q = p[:, [(1, 2), (0, 2), (0, 1)][axis]] / TEX_M
        uv.append(q)
    return np.concatenate(uv) + offset


# ---------------------------------------------------------------- USD and metadata
def paint(st, rgb):
    return mp.add_material(
        st,
        "/root/_materials/Paint",
        "OmniPBR",
        {"diffuse_color_constant": rgb, "reflection_roughness_constant": 0.5, "metallic_constant": 0.0},
        rgb,
        1.0,
        0.5,
    )


def wood(st):
    mat = mp.add_material(
        st,
        "/root/_materials/Wood",
        "OmniPBR",
        {"reflection_roughness_constant": 0.6, "metallic_constant": 0.0},
        srgb(WOOD[0]),  # UsdPreviewSurface fallback colour
        1.0,
        0.6,
    )
    UsdShade.Shader.Get(st, "/root/_materials/Wood/Shader").CreateInput(
        "diffuse_texture", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(WOOD_TEX))
    return mat


def write_usdz(out, c, hex_rgb, texture, offset):
    """hex_rgb None: unpainted wood with the grain texture."""
    tmp = Path(tempfile.mkdtemp()) / "object.usdc"
    st = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = mp.add_xform(st, "/root")
    st.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mp.add_xform(st, "/root/visual")
    mesh = mp.add_mesh(st, "/root/visual/model", c["pts"], c["faces"], mp.vertex_normals(c["pts"], c["faces"]))
    if hex_rgb is None:
        uv = wood_uv(c["pts"], c["faces"], c["kind"], offset)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        look = wood(st)
    else:
        look = paint(st, srgb(hex_rgb))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)
    pc.add_collision(st, c["hulls"])
    st.Save()
    zip_writer = getattr(Sdf, "ZipFileWriter", None) or Usd.ZipFileWriter
    zw = zip_writer.CreateNew(str(out))
    zw.AddFile(str(tmp), "object.usdc")
    if hex_rgb is None:
        zw.AddFile(str(texture), WOOD_TEX)
    zw.Save()


def description(uid, b, color):
    w = b["words"]
    natural = color == "natural"
    look = "unpainted beech" if natural else color
    caption = {
        "color": ["natural wood", "light beech", "unpainted"] if natural
                 else [color, f"painted {color}", f"bright {color}"],
        "material": ["wood", "unpainted solid beech" if natural else "painted solid wood",
                     "visible wood grain" if natural else "water-based paint on wood"],
        "size": [w["size"], "palm-sized", "a few centimetres across"],
        "name": [w["name"], f"{look} {w['name']}", "wooden building block"],
        "shape": [w["shape"], "flat faces with slightly rounded edges", "solid, no holes" if b["kind"] != "arch"
                  else "open underneath"],
        "function": ["stacking", "building towers and bridges", "being picked up and placed"],
    }  # fmt: skip
    desc = [
        f"{look} wooden {w['name']}",
        f"solid wood toy block shaped like a {w['shape']}",
        f"{w['name']} {w['size']}",
        f"smooth {'light brown block with beech grain' if natural else color + ' block'} and slightly rounded edges",
        f"light wooden {w['name']} that stacks with the other blocks of the set",
    ]
    return {"uuid": uid, "caption": caption, "description": desc}


def save(rigid_dir, b, c, mass, idx, texture):
    color, hex_rgb = PALETTE[idx]
    lo, hi = c["pts"].min(0), c["pts"].max(0)
    shift = -(lo + hi) / 2
    c = dict(c, pts=c["pts"] + shift, hulls=[h + shift for h in c["hulls"]])
    d = Path(rigid_dir) / b["cat"] / f"{idx:05d}"
    d.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{b['cat']}/{idx}"))
    offset = np.random.default_rng(uuid.UUID(uid).int % 2**32).uniform(0, 1, 2)  # shapes do not share one grain patch
    write_usdz(d / "object.usdz", c, hex_rgb, texture, offset)
    ext = hi - lo
    meta = pc.metadata(uid, dict(pts=c["pts"], faces=c["faces"], contact_r=float(min(ext[:2]) / 2)), mass, 0 * shift)
    meta["measurements_mm"] = measure(c)
    if idx:
        meta["variant_of"] = f"Rigid/{b['cat']}/00000"
    (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False))
    (d / "description.json").write_text(json.dumps(description(uid, b, color), indent=4))
    return d, meta


def main(rigid_dir):
    built = [(b, build(b)) for b in SHAPES]
    checks = [self_check(b, c) for b, c in built]
    volume = {b["cat"]: mp.signed_volume(c["pts"], c["faces"]) for b, c in built}
    weighed = [b["mass"] / volume[b["cat"]] for b in SHAPES if b["mass"]]
    density = float(np.mean(weighed))
    print(f"density of the weighed blocks {[round(r) for r in weighed]} kg/m3, unweighed blocks use {density:.0f}")
    texture = wood_texture(Path(tempfile.mkdtemp()) / "wood.png")
    for (b, c), (got, gap) in zip(built, checks):
        mass = b["mass"] or round(density * volume[b["cat"]], 4)
        for idx in range(len(PALETTE)):
            d, meta = save(rigid_dir, b, c, mass, idx, texture)
        print(
            f"{d.parent}/0000[0-{len(PALETTE) - 1}]: {meta['physics']['size']} m, {mass * 1000:.1f} g"
            f"{'' if b['mass'] else ' (est.)'}, measurements_mm {got}, {len(c['hulls'])} hull(s), "
            f"max gap {gap * 1000:.2f} mm"
        )
    sys.stdout.flush()  # APP.close() exits without flushing a redirected stdout


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else RIGID_DIR)
    except Exception:
        if mp.APP is None:
            raise
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if mp.APP is not None:
        mp.APP.close()
