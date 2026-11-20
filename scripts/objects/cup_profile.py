"""Any cup as a bean container, numpy only: a profile measured from the cup's collider, the cavity test, the capacity
in PBD red beans, and the body the gripper closes on. Shared by the g1d_pour_beans_plus task check, its datagen plugin,
the scripted expert and the layout generator, so they all count the same beans.

A profile (a layout entry's "cavity"; cup frame at scale 1: origin at the cup root, Z up) is
    axis    [x, y] of the cavity axis: the centre of its largest inscribed circle;
    z, r    the cavity's inner radius r[i] (m) at height z[i], from its floor z[0] up to its spill level z[-1]: the
            median over sectors around the axis; r_in the narrowest sector's (where a bean can be put); optional r_max
            the 95th-percentile sector's (corners of a square cup or belly of an inward-lip cup, used by in_cavity);
    bottom  z of the cup's underside; top: of its highest point;
    body    {"z": [...], "r": [...]} outer radius of the cup without its handle, bottom to top;
    handle  yaw (deg, cup frame) of a handle sticking out of the body, or None.
profile() makes one from the height map of scripts/objects/check_cup_holds_beans.py (cavity/<n>.npz); refine_from_mesh()
replaces top-down-occluded inner/outer radii and recessed inner floors from the 3D triangle mesh.
"""

import numpy as np
from scipy import ndimage

BEAN_D, PACKING = 0.0056, 0.6  # the PBD red beans (m) and how densely they settle in a cup (round_cup_shape)
SECTORS = 36  # around the axis; a radius is the median of the sectors' largest
DZ = 0.0025  # m between profile heights
SLACK = 0.003  # m: a bean this far over the spill level or outside the profile still counts as inside (a heap settles
# a little above the spill level; beans tuck under an inward lip the vertical rays do not see)
PACK_FLOOR = 0.002  # m: inner floor of the tall measuring cup the rigid grain packs settle in (measuring_cup_shape)
HANDLE = 1.25  # a sector reaching this much past the body's median radius (and 5 mm) is part of a handle


def rotation(quaternion):
    """3x3 rotation of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quaternion, float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _sector_max(dx, dy, cell):
    """Per sector around the origin, the largest distance of the cells (dx, dy); NaN where none."""
    d = np.hypot(dx, dy) + cell / 2
    k = ((np.arctan2(dy, dx) % (2 * np.pi)) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    out = np.full(SECTORS, np.nan)
    for s in range(SECTORS):
        if (k == s).any():
            out[s] = d[k == s].max()
    return out


def profile(xs, ys, height, level, cavity, bottom):
    """Profile of a cup from its height map (cells xs x ys, the collider's top surface, each cell's spill level, the
    cavity cells; cup frame) and the z of its underside."""
    cell = float(xs[1] - xs[0])
    labels, n = ndimage.label(cavity)
    if not n:
        raise ValueError("No cavity")
    largest = labels == 1 + int(np.argmax(ndimage.sum(cavity, labels, range(1, n + 1))))
    dist = ndimage.distance_transform_edt(np.pad(largest, 1))[1:-1, 1:-1]
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    axis = np.array([xs[i], ys[j]])
    gx, gy = np.meshgrid(xs - axis[0], ys - axis[1], indexing="ij")
    floor, spill = float(np.min(height[largest])), float(np.median(level[largest]))
    z = np.arange(floor, spill, DZ).tolist() + [spill]
    r, r_in = [], []
    for zi in z:
        open_ = largest & (height <= zi + 1e-9)
        sectors = _sector_max(gx[open_], gy[open_], cell)
        r.append(float(np.nanmedian(sectors)) if np.isfinite(sectors).any() else 0.0)
        r_in.append(float(np.nan_to_num(sectors).min()))
    solid = np.isfinite(height)
    top = float(np.nanmax(height))
    body_z = np.arange(bottom, top, 4 * DZ).tolist() + [top]
    body_r = []
    for zi in body_z:
        above = solid & (np.nan_to_num(height, nan=-np.inf) >= zi)
        sectors = _sector_max(gx[above], gy[above], cell) if above.any() else np.full(SECTORS, np.nan)
        body_r.append(float(np.nanmedian(sectors)) if np.isfinite(sectors).any() else body_r[-1] if body_r else 0.0)
    # rays from above do not see a flared wall under its rim: where the body reads as wide as the rim, it is the
    # cavity's radius there plus the wall's thickness at the rim
    wall = max(body_r[-1] - r[-1], 0.001)
    body_r = [min(b, np.interp(zi, z, r) + wall) if b > body_r[-1] - 0.001 else b for zi, b in zip(body_z, body_r)]
    mid = solid & (np.nan_to_num(height, nan=-np.inf) >= (bottom + top) / 2)
    sectors = _sector_max(gx[mid], gy[mid], cell)
    med = np.nanmedian(sectors)
    out = np.nan_to_num(sectors) > max(HANDLE * med, med + 0.005)
    handle = None
    if out.any():
        a = (np.nonzero(out)[0] + 0.5) * 2 * np.pi / SECTORS
        handle = round(float(np.degrees(np.arctan2(np.sin(a).sum(), np.cos(a).sum()))), 1) % 360
    return {
        "axis": [round(float(v), 4) for v in axis],
        "z": [round(v, 4) for v in z],
        "r": [round(v, 4) for v in r],
        "r_in": [round(v, 4) for v in r_in],
        "bottom": round(float(bottom), 4),
        "top": round(top, 4),
        "body": {"z": [round(v, 4) for v in body_z], "r": [round(v, 4) for v in body_r]},
        "handle": handle,
    }


def _ray_floor_z(points, triangles, xy, z_max):
    """Highest mesh surface z <= `z_max` struck by vertical rays at `xy` (K, 2)."""
    t0, t1, t2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    keep = np.minimum(np.minimum(t0[:, 2], t1[:, 2]), t2[:, 2]) <= z_max
    t0, t1, t2 = t0[keep], t1[keep], t2[keep]
    lo, hi = xy.min(0) - 1e-6, xy.max(0) + 1e-6
    in_box = (
        (np.maximum(np.maximum(t0[:, 0], t1[:, 0]), t2[:, 0]) >= lo[0])
        & (np.minimum(np.minimum(t0[:, 0], t1[:, 0]), t2[:, 0]) <= hi[0])
        & (np.maximum(np.maximum(t0[:, 1], t1[:, 1]), t2[:, 1]) >= lo[1])
        & (np.minimum(np.minimum(t0[:, 1], t1[:, 1]), t2[:, 1]) <= hi[1])
    )
    t0, t1, t2 = t0[in_box], t1[in_box], t2[in_box]
    if not len(t0):
        return np.array([])
    v0, v1 = t2[:, :2] - t0[:, :2], t1[:, :2] - t0[:, :2]
    denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(denom) > 1e-14
    t0, t1, t2, v0, v1, denom = t0[ok], t1[ok], t2[ok], v0[ok], v1[ok], denom[ok]
    hits = []
    for px, py in xy:
        v2x, v2y = px - t0[:, 0], py - t0[:, 1]
        u = (v2x * v1[:, 1] - v1[:, 0] * v2y) / denom
        v = (v0[:, 0] * v2y - v2x * v0[:, 1]) / denom
        m = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
        if m.any():
            zh = t0[m, 2] + u[m] * (t2[m, 2] - t0[m, 2]) + v[m] * (t1[m, 2] - t0[m, 2])
            zb = zh[zh <= z_max]
            if len(zb):
                hits.append(float(zb.max()))
    return np.asarray(hits, float)


def refine_from_mesh(cup, points, triangles):
    """Refine a top-down height-map profile `cup` using horizontal slices of its 3D triangle mesh (`points` (V, 3),
    `triangles` (F, 3) in the cup root frame at scale 1). Recovers inward-curving cavity bellies hidden under a
    narrower rim, outward-bulging bodies clipped by top-down rays, non-circular corner radii (`r_max`), and recessed
    inner floors where a 2.5 mm height-map ray slipped through a convex-decomposition seam."""
    pts = np.asarray(points, float)
    tris = np.asarray(triangles, int).reshape(-1, 3)
    if not len(pts) or not len(tris):
        return dict(cup)
    ax = np.asarray(cup["axis"], float)
    spill = float(cup["z"][-1])
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    probe = np.vstack(
        [ax, ax + 0.002 * np.c_[np.cos(ang[:8]), np.sin(ang[:8])], ax + 0.004 * np.c_[np.cos(ang), np.sin(ang)]]
    )
    hz = _ray_floor_z(pts, tris, probe, spill - 0.008)
    floor = float(cup["z"][0])
    if len(hz) >= 8:
        cand = float(np.median(hz))
        if floor - 0.002 <= cand <= spill - 0.010:
            floor = max(floor, cand)
    z_cav = np.arange(floor, spill, DZ).tolist() + [spill]

    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    p0, p1 = pts[edges[:, 0]], pts[edges[:, 1]]
    z0, z1 = p0[:, 2], p1[:, 2]
    z_lo, z_hi = np.minimum(z0, z1), np.maximum(z0, z1)
    dz = z1 - z0

    def slice_sectors(zi):
        m = (z_lo <= zi) & (z_hi >= zi)
        if not m.any():
            return np.full(SECTORS, np.nan), np.full(SECTORS, np.nan)
        d = dz[m]
        denom = np.where(np.abs(d) > 1e-12, d, 1.0)
        t = np.where(np.abs(d) > 1e-12, (zi - z0[m]) / denom, 0.5)
        xy = p0[m, :2] + t[:, None] * (p1[m, :2] - p0[m, :2]) - ax
        rad = np.hypot(xy[:, 0], xy[:, 1])
        sec = ((np.arctan2(xy[:, 1], xy[:, 0]) % (2.0 * np.pi)) / (2.0 * np.pi) * SECTORS).astype(int) % SECTORS
        s_min, s_max = np.full(SECTORS, np.inf), np.full(SECTORS, -np.inf)
        np.minimum.at(s_min, sec, rad)
        np.maximum.at(s_max, sec, rad)
        s_min[np.isinf(s_min)] = np.nan
        s_max[np.isinf(s_max)] = np.nan
        return s_min, s_max

    r_med, r_in, r_max = [], [], []
    zc_lo, zc_hi = min(floor + 0.0015, spill - 0.0005), spill - 0.0005
    for idx, zi in enumerate(z_cav):
        s_min, _ = slice_sectors(float(np.clip(zi, zc_lo, zc_hi)))
        valid = np.isfinite(s_min)
        r_med.append(float(np.nanmedian(s_min)) if valid.any() else 0.0)
        if idx == 0:
            r_in.append(0.0)
        elif valid.all():
            r_in.append(float(np.min(s_min)))
        else:
            r_in.append(float(np.nanpercentile(s_min, 5)) if valid.any() else 0.0)
        r_max.append(float(np.nanpercentile(s_min, 95)) if valid.any() else 0.0)

    body_z = list(cup["body"]["z"])
    body_r = []
    zb_lo, zb_hi = body_z[0] + 0.0005, body_z[-1] - 0.0005
    for zi in body_z:
        _, s_max = slice_sectors(float(np.clip(zi, zb_lo, zb_hi)))
        valid = np.isfinite(s_max)
        body_r.append(float(np.nanmedian(s_max)) if valid.any() else (body_r[-1] if body_r else 0.0))

    out = dict(cup)
    out.update(
        z=[round(v, 4) for v in z_cav],
        r=[round(v, 4) for v in r_med],
        r_in=[round(v, 4) for v in r_in],
        r_max=[round(v, 4) for v in r_max],
        body={"z": [round(float(v), 4) for v in body_z], "r": [round(v, 4) for v in body_r]},
    )
    return out


def local_points(points, position, quaternion, scale=1.0):
    """Points in the cup frame at scale 1."""
    return (np.asarray(points, float) - np.asarray(position, float)) @ rotation(quaternion) / scale


def in_cavity(points, position, quaternion, cup, scale=1.0):
    """Bean centres inside the cup at `position` / `quaternion` (w, x, y, z), scaled uniformly by `scale`: above
    its cavity floor, below its spill level, within the inner wall (`r_max` when available so square-cup corners and
    spout lips count as inside, else `r`)."""
    local = local_points(points, position, quaternion, scale)
    z, r = np.asarray(cup["z"]), np.asarray(cup.get("r_max", cup["r"]))
    radius = np.interp(local[:, 2], z, r)
    d = np.hypot(local[:, 0] - cup["axis"][0], local[:, 1] - cup["axis"][1])
    return (local[:, 2] > z[0] - SLACK) & (local[:, 2] < z[-1] + SLACK) & (d < radius + SLACK)


def volume(cup, scale=1.0):
    """Cavity volume (m^3) up to the spill level."""
    z, area = np.asarray(cup["z"]), np.pi * np.asarray(cup["r"]) ** 2  # numpy 1 (the sim env) has no trapezoid
    return float(np.sum((area[1:] + area[:-1]) / 2 * np.diff(z))) * scale**3


PBD_GRAIN = dict(volume=np.pi / 6 * BEAN_D**3, packing=PACKING)


def capacity(cup, scale=1.0, grain=None):
    """Beans the cup holds filled to its spill level; `grain` dict(volume (m^3), packing), default the PBD beans."""
    grain = grain or PBD_GRAIN
    return volume(cup, scale) * grain["packing"] / grain["volume"]


def fill_need(fill, capacity_, beans):
    """Beans the target cup must hold for a pour to `fill`: that share of its capacity, or of the episode's beans when
    they could not fill it that far (a small source cup into a large target). The task check
    (func_parser.is_fluid_in_cup_profile) judges a layout with Generalization.pour.target_fill by this number."""
    return fill * min(capacity_, beans)


def body_radius(cup, z, scale=1.0):
    """Outer radius of the body (no handle) at height z of the cup frame, scaled."""
    return float(np.interp(z / scale, cup["body"]["z"], cup["body"]["r"])) * scale


def body_points(cup, scale=1.0, around=24):
    """Points on the outer wall of the body (no handle; cup frame, scaled): `around` per profile height."""
    a = np.arange(around) * 2 * np.pi / around
    axis = np.asarray(cup["axis"]) * scale
    return np.array(
        [
            (axis[0] + r * scale * np.cos(t), axis[1] + r * scale * np.sin(t), z * scale)
            for z, r in zip(cup["body"]["z"], cup["body"]["r"])
            for t in a
        ]
    )


def touching(pose_a, cup_a, pose_b, cup_b, scale_a=1.0, scale_b=1.0, gap=0.001):
    """Whether two cups' bodies (poses (position, (w, x, y, z))) come within `gap` of each other: a point of either
    wall inside the other's wall grown by `gap`."""

    def into(pose_p, cup_p, scale_p, pose_q, cup_q, scale_q):
        world = body_points(cup_p, scale_p) @ rotation(pose_p[1]).T + np.asarray(pose_p[0], float)
        local = local_points(world, *pose_q)
        z = local[:, 2]
        d = np.hypot(local[:, 0] - cup_q["axis"][0] * scale_q, local[:, 1] - cup_q["axis"][1] * scale_q)
        inside = (z > cup_q["bottom"] * scale_q - gap) & (z < cup_q["top"] * scale_q + gap)
        wall = np.interp(z / scale_q, cup_q["body"]["z"], cup_q["body"]["r"]) * scale_q
        return bool((inside & (d < wall + gap)).any())

    return into(pose_a, cup_a, scale_a, pose_b, cup_b, scale_b) or into(pose_b, cup_b, scale_b, pose_a, cup_a, scale_a)


def fill_height(cup, beans, scale=1.0, grain=None):
    """z (cup frame, scaled) the top of `beans` settled in the cup reaches (`grain` as capacity())."""
    grain = grain or PBD_GRAIN
    z, r = np.asarray(cup["z"]) * scale, np.asarray(cup["r"]) * scale
    want = beans * grain["volume"] / grain["packing"]
    filled = np.r_[0.0, np.cumsum(np.pi * ((r[1:] + r[:-1]) / 2) ** 2 * np.diff(z))]
    return float(np.interp(want, filled, z)) if want < filled[-1] else float(z[-1])


def column(cup, beans, scale=1.0, spacing=1.05 * BEAN_D):
    """`beans` bean centres on a square lattice inside the cavity from its floor up (cup frame, scaled): each layer
    fills the cavity's narrowest radius over the beans' height less a bean (a bean pinched by a tapering floor is
    pushed out through the wall); spawned there they settle without a drop. Stops at the profile's top."""
    z, r = np.asarray(cup["z"]) * scale, np.asarray(cup.get("r_in", cup["r"])) * scale
    axis = np.asarray(cup["axis"]) * scale
    n = int(r.max() // spacing) + 1
    grid = np.array([(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)], float) * spacing
    out, zi = [], z[0] + BEAN_D / 2 + 0.001
    while len(out) < beans and zi < z[-1]:
        span = np.linspace(zi - BEAN_D / 2, zi + BEAN_D / 2, 5)
        room = np.interp(span, z, r).min() - BEAN_D / 2 - 0.0015
        out.extend((axis[0] + x, axis[1] + y, zi) for x, y in grid[np.hypot(grid[:, 0], grid[:, 1]) <= room])
        zi += spacing
    return np.array(out[:beans]).reshape(-1, 3)


def grain_half(pack):
    """Each grain's largest half extent (m) in a pack (an np.load of pack.npz)."""
    return (np.abs(pack["hull"]).max(axis=0) * pack["scale"]).max(axis=1)


def pack_grain(pack):
    """A pack's grain for capacity() / fill_height(): its mean volume and packing."""
    return dict(volume=float(pack["volume"] * np.prod(pack["scale"], axis=1).mean()), packing=float(pack["packing"]))


def pack_fill(cup, scale, centres, half, floor=PACK_FLOOR, yaw=0.0, count=None):
    """Rigid grains of a settled pack (scripts/g1d/rigid_bean_probe.py --pack) that fit in the cup: the pack's
    `centres` (its container's frame: axis at x = y = 0, inner floor at z = `floor`), turned `yaw` rad about the axis
    and moved onto the cup's axis and cavity floor (cup frame, scaled), kept where the cavity's narrowest radius over
    each grain's height, less its largest half extent `half` (m, per grain) and 1.5 mm, holds it, below the spill
    level. Returns (indices into the pack, lowest first, at most `count`; their centres in the cup frame), the same
    for the layout generator and GranularObject. A random heap, unlike a lattice of aligned grains that jams."""
    z, r = np.asarray(cup["z"]) * scale, np.asarray(cup.get("r_in", cup["r"])) * scale
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.asarray(centres, float) @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    p = p + [*(np.asarray(cup["axis"]) * scale), z[0] - floor]
    room = np.min([np.interp(p[:, 2] + d * half, z, r) for d in (-1.0, 0.0, 1.0)], axis=0) - half - 0.0015
    fit = np.flatnonzero((np.hypot(p[:, 0] - cup["axis"][0] * scale, p[:, 1] - cup["axis"][1] * scale) <= room)
                         & (p[:, 2] + half <= z[-1]))  # fmt: skip
    fit = fit[np.argsort(p[fit, 2], kind="stable")][:count]
    return fit, p[fit]
