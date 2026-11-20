#!/usr/bin/env python3
"""Test object of the platform's registration -> modeling -> library flow as a RoboDojo `Rigid` asset.

- test_object/00000: solid 10 mm cube, sharp edges, milky translucent PP, no labels (platform registration by 徐鹤鸣,
  2026-11-05 11:58:39; no photos, modeled from the registered description).
  Edge 10 mm measured (registration); mass 0.9 g estimated from PP density 0.9 g/cm^3 (registered 100 g contradicts
  the size, changed with the registrant's consent, see spec.json notes).

Usage:
    python scripts/objects/make_test_object.py [RIGID_DIR]    # default Assets/Object/Real2Sim/Rigid
    python scripts/objects/check_object.py Assets/Object/Real2Sim/Rigid/test_object/00000
Writes RIGID_DIR/test_object/00000/{object.usdz, metadata.json, description.json}. Needs what make_pitcher.py needs.

Frame: Z up, metres, origin at bbox centre.
USD:   /root (RigidBodyAPI) -> visual/model (the cube, flat-shaded, one material) -> collision/hull_00 (the cube)
"""

import os
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pitcher as mp  # noqa: E402  (shared USD helpers; starts Kit when pxr needs it)
import make_plastic_cups as pc  # noqa: E402

RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/Real2Sim/Rigid"
EDGE = 0.010  # edge length, m (measured: registration "边长10mm")
DENSITY = 900.0  # PP, kg/m^3 (estimated)
MASS = round(DENSITY * EDGE**3, 4)  # 0.0009 kg (estimated from density)
MILKY_PP = (0.94, 0.94, 0.92)  # milky translucent PP (estimated, no photos)

WORDS = dict(
    color=["milky white", "translucent white", "semi-transparent"], material=["plastic", "polypropylene", "PP"],
    size=["10 mm edge", "tiny", "fingertip-sized"], name=["test cube", "plastic cube", "small cube"],
    shape=["cube", "sharp edges", "flat faces"], function=["testing the asset pipeline", "a placeholder object",
                                                           "picking up"],
)  # fmt: skip


def build_cube(a=EDGE):
    """Cube [0, a]^3 as 6 quads with their own corners (flat shading), outward winding; one convex hull."""
    corners = np.array([(x, y, z) for x in (0, a) for y in (0, a) for z in (0, a)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]  # -x +x -y +y -z +z
    pts, faces, normals = [], [], []
    for q in quads:
        p = corners[list(q)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        base = len(pts)
        pts += list(p)
        normals += [n] * 4
        faces.append([base, base + 1, base + 2, base + 3])
    pts, normals = np.array(pts), np.array(normals)
    assert mp.signed_volume(pts, faces) > 0, "cube has inward-facing winding"
    return dict(pts=pts, faces=faces, normals=normals, hulls=[corners], contact_r=a / 2, volume=a**3)


def measurements(c):
    """spec dimensions_mm ("边长") measured on the mesh: edge along X."""
    return {"measurements_mm": {"边长": round(float(np.ptp(c["pts"][:, 0])) * 1000, 1)}}


def description(uid):
    w = WORDS
    caption = {k: w[k] for k in ("color", "material", "size", "name", "shape", "function")}
    desc = [
        "small milky white translucent plastic cube with a 10 mm edge",
        "solid polypropylene cube with sharp edges and flat faces",
        "tiny semi-transparent test cube without labels or markings",
        "fingertip-sized plastic cube used to test the asset pipeline",
        "light milky PP cube that can be picked up between two fingers",
    ]
    return {"uuid": uid, "caption": caption, "description": desc}


def main(rigid_dir):
    c = build_cube()
    material = lambda st: pc.clear_pp(st, MILKY_PP, 0.5)  # noqa: E731  (half milky: whitish, still translucent)
    d, meta = pc.save(rigid_dir, "test_object", 0, c, material, MASS, description, measurements)
    print(f"{d}: mass {MASS} kg (PP-density estimate), size {meta['physics']['size']}, {meta['measurements_mm']}")


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
