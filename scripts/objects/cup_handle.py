"""A cup's handle loop as the G1-D gripper can take it (scripts/g1d/scripted_demo.handle_grasp): the fingers level and
square to the handle's plane, the inner finger through the hole between the handle and the body, the pads closing
across the handle's outer bar. Measured from the cup's collider, numpy only (pxr for loading the USD).

handle(triangles, profile) -> {yaw, bar: [inner, outer] (m from the cup axis), hole: [z low, z high] (the inner
finger's height, centred in the free height against the bar), z (its centre), body (body radius there), width (the
bar along the fingers), room: [across, tall] (m free for the inner finger against the bar)} in the cup frame at scale
1 (scripts/objects/cup_profile.py), or None: the hole is the largest empty region of the collider's silhouette seen
along the fingers (projected onto the handle's plane) closed off by the body and the handle, and holds the inner
finger (FINGER, plus CLEAR all round) against the bar.

    python scripts/objects/cup_handle.py --assets Assets [cup ids]   (default: the pool's cups with a handle)
prints each cup's handle as a grasp entry for real2sim/tasks/g1d_pour_beans_plus/grasps.json.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.objects import cup_profile  # noqa: E402

CELL = 0.0005  # m, silhouette resolution
SIDE = 0.04  # m: collider within this of the handle's plane (tangentially) is in its silhouette
FINGER = (0.0155, 0.030)  # m, a dex1 finger with its pad: across (closing direction) and tall (Link1_2_col, Link1_3)
CLEAR = 0.002  # m all round the inner finger in the hole


def triangles(path):
    """(n, 3, 3) collider triangles of the USD cup at `path` in its root frame (m)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    inv = cache.GetLocalToWorldTransform(root).GetInverse()
    out = []
    for prim in stage.Traverse():
        if not (prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh)):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        mat = np.asarray(cache.GetLocalToWorldTransform(prim) * inv, float)
        points = (np.c_[points, np.ones(len(points))] @ mat)[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
        index = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        for k in range(1, counts.max() - 1):  # fan triangulation of the polygons
            sel = counts > k + 1
            out.append(points[index[np.c_[start[sel], start[sel] + k, start[sel] + k + 1]]])
    return np.concatenate(out) * UsdGeom.GetStageMetersPerUnit(stage)


def silhouette(tris, axis, yaw):
    """The collider within SIDE of the handle plane seen along the fingers: occupancy over (u: m out from the axis
    toward the handle, z), the u and z of the cells."""
    d = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
    t = np.array([-d[1], d[0]])
    rel = tris[..., :2] - axis
    u, v, z = rel @ d, rel @ t, tris[..., 2]
    keep = (np.abs(v) < SIDE).any(1) & (u.max(1) > 0)
    u, z = u[keep], z[keep]
    us = np.arange(0.0, u.max() + 2 * CELL, CELL)
    zs = np.arange(z.min() - 2 * CELL, z.max() + 2 * CELL, CELL)
    grid = np.zeros((len(us), len(zs)), bool)
    # each triangle filled by points closer than a cell
    span = np.maximum(np.ptp(u, 1), np.ptp(z, 1))
    for n in np.unique(np.clip(np.ceil(span / CELL * 1.5).astype(int), 1, 400)):
        sel = np.clip(np.ceil(span / CELL * 1.5).astype(int), 1, 400) == n
        a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        ok = a + b <= n
        wa, wb = a[ok] / n, b[ok] / n
        wc = 1 - wa - wb
        pu = u[sel, 0, None] * wa + u[sel, 1, None] * wb + u[sel, 2, None] * wc
        pz = z[sel, 0, None] * wa + z[sel, 1, None] * wb + z[sel, 2, None] * wc
        i = np.clip(np.round(pu / CELL).astype(int), 0, len(us) - 1)
        j = np.clip(np.round((pz - zs[0]) / CELL).astype(int), 0, len(zs) - 1)
        grid[i.ravel(), j.ravel()] = True
    return ndimage.binary_closing(grid, iterations=2), us, zs


def handle(tris, profile):
    """The handle's grasp geometry (module docstring), None without a handle or a hole the inner finger fits."""
    yaw = profile.get("handle")
    if yaw is None:
        return None
    solid, us, zs = silhouette(tris, np.asarray(profile["axis"]), yaw)
    holes, n = ndimage.label(~solid)
    edge = set(holes[0]) | set(holes[-1]) | set(holes[:, 0]) | set(holes[:, -1])
    inner = [k for k in range(1, n + 1) if k not in edge]
    if not inner:
        return None
    hole = holes == max(inner, key=lambda k: (holes == k).sum())
    across, tall = (round((FINGER[0] + 2 * CLEAR) / CELL), round((FINGER[1] + 2 * CLEAR) / CELL))

    def widest(rows):
        """The longest run of columns (u) of the hole free over all `rows` (z), None if none."""
        free = np.flatnonzero(hole[:, rows].all(1))
        return max(np.split(free, np.flatnonzero(np.diff(free) > 1) + 1), key=len) if len(free) else None

    # the finger's height where the hole is widest, then centred in the free height of its columns against the bar
    runs = [(run, j0) for j0 in range(hole.shape[1] - tall + 1) if (run := widest(slice(j0, j0 + tall))) is not None]
    run, j0 = max(runs, key=lambda r: len(r[0]), default=(np.array([]), 0))
    if len(run) < across:
        return None
    free_z = np.flatnonzero(hole[run[-across:]].all(0))
    run_z = next(r for r in np.split(free_z, np.flatnonzero(np.diff(free_z) > 1) + 1) if j0 in r)
    zm = (zs[run_z[0]] + zs[run_z[-1]]) / 2
    band = np.abs(zs - zm) <= FINGER[1] / 2 + CLEAR
    run = widest(band)
    bar_in = us[run[-1]] + CELL / 2
    beyond = np.flatnonzero(~solid[:, band].any(1) & (us > bar_in))
    bar_out = us[beyond[0]] - CELL / 2 if len(beyond) else us[-1]
    z_lo, z_hi = zm - FINGER[1] / 2, zm + FINGER[1] / 2
    return dict(
        yaw=round(float(yaw), 1),
        bar=[round(float(bar_in), 4), round(float(bar_out), 4)],
        hole=[round(float(z_lo), 4), round(float(z_hi), 4)],
        z=round(float(zm), 4),
        body=round(cup_profile.body_radius(profile, zm), 4),
        width=round(bar_width(tris, profile, yaw, bar_in, bar_out, z_lo, z_hi), 4),
        room=[round(len(run) * CELL, 4), round(len(run_z) * CELL, 4)],
    )


def bar_width(tris, profile, yaw, bar_in, bar_out, z_lo, z_hi):
    """The bar's extent (m) along the fingers over the hole's height: of the triangles across it there."""
    d = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
    t = np.array([-d[1], d[0]])
    rel = tris[..., :2] - np.asarray(profile["axis"])
    u, v, z = rel @ d, rel @ t, tris[..., 2]
    sel = (u.max(1) >= bar_in) & (u.min(1) <= bar_out) & (z.max(1) >= z_lo) & (z.min(1) <= z_hi)
    sel &= (np.abs(v) < SIDE).all(1)
    return float(np.ptp(v[sel])) if sel.any() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=ROOT / "real2sim/tasks/g1d_pour_beans_plus/cups.json")
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text())["cups"]
    ids = args.ids or sorted(c for c, v in pool.items() if v["cavity"].get("handle") is not None)
    out = {}
    for c in ids:
        folder = args.assets / "Object" / c
        path = next(p for p in (folder / "object.usd", folder / "object.usdz") if os.path.exists(p))
        out[c] = handle(triangles(path), pool[c]["cavity"])
        print(c, json.dumps(out[c]), flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
