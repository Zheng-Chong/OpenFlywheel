"""Black parts of the G1-D that the official URDF meshes do not separate (numpy only; applied by g1d.add_black_trim).

Arms (BLACK_FACES): black faces of the metal arm links, chosen by face centroid in the link frame. SCREWS: socket-head
screws in their holes on the wrists, visual only.

Chassis (AGV_link): the official mesh is one plain silver shell. The real chassis has a black panel over its front face
with sensor windows (PANEL, WINDOWS), a rubber bumper round its foot (BUMPER) and a lidar puck on its top face (LIDAR);
they are added as visual-only meshes. AGV_link frame: x forward, z up; the shell spans z -0.078..0.086, its top face
lies at z 0.0855 and its front face at x 0.2725 on the axis, bowing back towards the corners.
"""

import numpy as np

# Link name suffix (after left_/right_) -> selector of face centroids c (n, 3). The forearm ring at the end of the
# elbow link (x 7.5..20 cm, just below the curved wrist link; the elbow bracket and ribbed motor section stay metal),
# and the elbow motor's end cap: the boss of radius 3.2 cm round the elbow axis (x 0.0158, z -0.0805) standing out of
# the upper arm's inner face (|y| > 2.85 cm; the rest of the link ends at 2.8 cm).
BLACK_FACES = {
    "elbow_link": lambda c: (c[:, 0] > 0.075) & (c[:, 0] < 0.2),
    "shoulder_yaw_link": lambda c: (np.abs(c[:, 1]) > 0.0285) & (np.hypot(c[:, 0] - 0.0158, c[:, 2] + 0.0805) < 0.034),
}


def _arc(cn, cz, r, t0, t1, k=7):
    return [(cn + r * np.cos(t), cz + r * np.sin(t)) for t in np.linspace(t0, t1, k)]


def _ring(cx, cy, r, n):
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in np.linspace(0.0, 2 * np.pi, n, endpoint=False)]


# Screws in holes on the wrists' +z faces (link frames), the side the head camera sees; the official meshes have
# none. Placed from real head-camera frames (G1D-PourBeansEps380 ep134, 640x480, so to about 1 mm); faces no frame
# shows stay plain. (link suffix, sides, centres (x, y) dropped onto the link along -z, radius.) Both arms: ten screws
# round the wrist pitch link's flange about the yaw axis (x 0.046; the yaw link hides the ones under it). The yaw
# motor's oval cap: four holes round the axis in a diamond on the left arm (one towards the gripper, one behind the
# axis, two beside it); two screws at its gripper end and one at the axis on the right arm.
SCREWS = [
    ("wrist_pitch_link", ("left", "right"), _ring(0.046, 0.0, 0.0195, 10), 0.0017),
    ("wrist_yaw_link", ("left",), _ring(0.0, 0.0, 0.0075, 4), 0.0022),
    ("wrist_yaw_link", ("right",), [(0.028, 0.0063), (0.028, -0.0063), (-0.003, 0.0)], 0.0016),
]


# Profiles are strips of (n, z): n is the outward offset from a path, z the height; normals are smoothed within a strip
# and break between strips. Estimated from product photos, not measured.
# Panel: a bent black sheet over the front face, 0.32 m of the 0.525 m width, from 2 cm above the bumper up to the
# top edge (front view), wrapping over the 5 mm rounded edge and running 2 cm onto the top face (front-left view).
# Swept along the front face's bow.
PANEL = dict(
    y=(-0.16, 0.16),
    strip=[(0.0015, -0.036), *_arc(-0.005, 0.08, 0.0065, 0, np.pi / 2), (-0.02, 0.0865)],
)
# Sensor windows in the panel (front view), glossy and darker than it: two small rounded rectangles near its top and
# two long slots below. (y centre, z centre, width, height, corner radius), mirrored in y.
WINDOWS = [(0.1025, 0.052, 0.031, 0.019, 0.007), (0.095, 0.0085, 0.094, 0.0255, 0.01275)]
# Glossy black puck on the top face, on the axis between the lift column (x +-0.079) and the front edge (the
# front-left product view puts it ~2/3 of the way out); the official 215 mm chassis height leaves it out. A wider foot
# ring, a slightly bulging body about 9.5 cm across and a rounded top edge, 4 cm tall; turned about its axis.
LIDAR = dict(
    base=(0.17, 0.0, 0.0845),  # seated 1 mm into the top face
    strips=[
        [(0.054, 0.0), (0.054, 0.008)],  # foot ring
        [(0.054, 0.008), (0.047, 0.0095)],
        [(0.047, 0.0095), (0.0478, 0.018), (0.048, 0.025), (0.0477, 0.031), *_arc(0.041, 0.034, 0.006, 0, np.pi / 2)]
        + [(0.03, 0.0405), (0.0, 0.041)],
    ],
)
# Black rubber bumper round the foot of the shell, standing 12 mm out of it with rounded edges (the official 557 mm
# length is the 538 mm shell plus this on both ends; at the sides it comes out to the wheel arches, y +-0.262). Swept
# along the shell's outline at its height (ray-cast: x -0.264..0.267, y +-0.246, corners r ~0.1); its inner edge sits
# 12 mm inside the shell, past where the rounded-rectangle fit departs from the real corners.
BUMPER = dict(
    outline=dict(x=(-0.264, 0.267), y=(-0.246, 0.246), r=0.1),
    strip=[(-0.012, -0.083), *_arc(0.006, -0.077, 0.006, -np.pi / 2, 0), *_arc(0.006, -0.063, 0.006, 0, np.pi / 2)]
    + [(-0.012, -0.057)],
)


def _sweep(path, path_normals, strips, z0=0.0, closed=True):
    """Points, face vertex counts, indices and vertex normals of the profile `strips` swept along the counter-clockwise
    `path` (m, 2) with outward unit `path_normals` (m, 2), lifted by `z0`."""
    m = len(path)
    i = np.arange(m if closed else m - 1)
    j = (i + 1) % m
    points, normals, faces = [], [], []
    for strip in strips:
        p = np.array(strip)
        d = np.diff(p, axis=0)
        seg = np.stack([d[:, 1], -d[:, 0]], axis=1)  # outward normal of each profile segment
        seg /= np.linalg.norm(seg, axis=1, keepdims=True)
        n = np.concatenate([seg[:1], seg[:-1] + seg[1:], seg[-1:]])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        start = len(points) * m
        for k, ((off, z), (nn, nz)) in enumerate(zip(p, n)):
            points.append(np.concatenate([path + off * path_normals, np.full((m, 1), z + z0)], axis=1))
            normals.append(np.concatenate([nn * path_normals, np.full((m, 1), nz)], axis=1))
            if k:
                lo, hi = start + (k - 1) * m, start + k * m
                faces.append(np.stack([lo + i, lo + j, hi + j, hi + i], axis=1))
    faces = np.concatenate(faces)
    return np.concatenate(points), [4] * len(faces), faces.ravel(), np.concatenate(normals)


def _rounded_rect(x0, x1, y0, y1, r, per_corner=16):
    """Counter-clockwise outline (m, 2) and outward normals (m, 2) of a rounded rectangle."""
    path, radial = [], []
    for k, (cx, cy) in enumerate([(x1 - r, y1 - r), (x0 + r, y1 - r), (x0 + r, y0 + r), (x1 - r, y0 + r)]):
        t = np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, per_corner)
        radial.append(np.stack([np.cos(t), np.sin(t)], axis=1))
        path.append(np.array([cx, cy]) + r * radial[-1])
    return np.concatenate(path), np.concatenate(radial)


def _front_x(triangles, ys, z):
    """x where rays from the chassis axis along +x at height z leave the shell `triangles` (n, 3, 3), for each y."""
    t = np.asarray(triangles, dtype=float)
    v0, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    p = np.cross([1.0, 0.0, 0.0], e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    v0, e1, e2, p, det = v0[ok], e1[ok], e2[ok], p[ok], det[ok]
    xs = []
    for y in ys:
        s = np.array([0.0, y, z]) - v0
        q = np.cross(s, e1)
        u, v, dist = np.einsum("ij,ij->i", s, p) / det, q[:, 0] / det, np.einsum("ij,ij->i", e2, q) / det
        xs.append(dist[(u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 0)].max())
    return np.array(xs)


def panel_meshes(triangles, samples=65):
    """(panel, windows) as from _sweep on the chassis shell `triangles`: the PANEL swept along the front face's bow,
    and the WINDOWS as flat fans 1.5 mm proud of it."""
    ys = np.linspace(*PANEL["y"], samples)
    xs = _front_x(triangles, ys, 0.0)  # the bow does not change with height (ray-cast)
    normal = np.stack([np.ones(samples), -np.gradient(xs, ys)], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    panel = _sweep(np.stack([xs, ys], axis=1), normal, [PANEL["strip"]], closed=False)
    points, faces = [], []
    for yc, zc, w, h, r in WINDOWS:
        for side in (1.0, -1.0):
            ring, _ = _rounded_rect(side * yc - w / 2, side * yc + w / 2, zc - h / 2, zc + h / 2, r, per_corner=8)
            ring = np.concatenate([[[side * yc, zc]], ring])  # fan centre first
            start = sum(len(q) for q in points)
            points.append(np.stack([np.interp(ring[:, 0], ys, xs) + 0.003, ring[:, 0], ring[:, 1]], axis=1))
            k = np.arange(1, len(ring))
            faces.append(np.stack([np.full(len(k), start), start + k, start + k % (len(ring) - 1) + 1], axis=1))
    points, faces = np.concatenate(points), np.concatenate(faces)
    windows = points, [3] * len(faces), faces.ravel(), np.tile([1.0, 0.0, 0.0], (len(points), 1))
    return panel, windows


# Socket-head screws sunk in their holes, as stacked fans (fraction of the hole radius, height above the surface in m,
# corners, how far the normals lean out at the rim): the hole's dark gap, the screw's domed metal head (the lean
# catches highlights), the dark hex socket in it.
SCREW_PARTS = {"hole": (1.0, 0.0002, 16, 0.0), "head": (0.65, 0.0004, 16, 0.7), "socket": (0.3, 0.0005, 6, 0.0)}


def screw_mesh(triangles, centres, radius):
    """{part: (points, counts, indices, normals)} of the SCREW_PARTS of screws of hole `radius` on the link `triangles`
    (n, 3, 3), where rays down -z through `centres` first hit them, tilted to the surface there. Centres that miss the
    link are skipped."""
    t = np.asarray(triangles, dtype=float)
    v0, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    face_n = np.cross(e1, e2)
    face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
    p = np.cross([0.0, 0.0, -1.0], e2)
    det = np.einsum("ij,ij->i", e1, p)
    frames = []
    for x, y in centres:
        s = np.array([x, y, 1.0]) - v0
        q = np.cross(s, e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v, dist = np.einsum("ij,ij->i", s, p) / det, q[:, 2] / -det, np.einsum("ij,ij->i", e2, q) / det
            hit = np.nonzero((np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 0))[0]
        if not len(hit):
            continue
        k = hit[np.argmin(dist[hit])]
        n = face_n[k] if face_n[k, 2] > 0 else -face_n[k]
        e = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        e /= np.linalg.norm(e)
        frames.append((np.array([x, y, 1.0 - dist[k]]), n, e, np.cross(n, e)))
    parts = {}
    for name, (scale, lift, corners, lean) in SCREW_PARTS.items():
        a = np.linspace(0.0, 2 * np.pi, corners, endpoint=False)
        points, normals, faces = [], [], []
        for centre, n, e, f in frames:
            radial = np.outer(np.cos(a), e) + np.outer(np.sin(a), f)
            rim = n + lean * radial
            start = len(points) * (corners + 1)
            points.append(np.concatenate([[centre + lift * n], centre + lift * n + scale * radius * radial]))
            normals.append(np.concatenate([[n], rim / np.linalg.norm(rim, axis=1, keepdims=True)]))
            i = np.arange(1, corners + 1)
            faces.append(np.stack([np.full(corners, start), start + i, start + i % corners + 1], axis=1))
        if points:
            faces = np.concatenate(faces)
            parts[name] = np.concatenate(points), [3] * len(faces), faces.ravel(), np.concatenate(normals)
    return parts


def lidar_mesh(segments=96):
    """_sweep of the LIDAR puck. (UsdGeom.Cylinder renders as a 12-sided prism in RTX.)"""
    a = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    radial = np.stack([np.cos(a), np.sin(a)], axis=1)
    centre = np.tile(LIDAR["base"][:2], (segments, 1))
    return _sweep(centre, radial, LIDAR["strips"], LIDAR["base"][2])


def bumper_mesh():
    """_sweep of the BUMPER along its rounded-rectangle outline."""
    o = BUMPER["outline"]
    return _sweep(*_rounded_rect(*o["x"], *o["y"], o["r"]), [BUMPER["strip"]])
