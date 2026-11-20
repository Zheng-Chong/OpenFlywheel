#!/usr/bin/env python3
"""Unpainted wood tray for the block sorting tasks (block_sorting.py): a narrow open box the blocks are set into.

Low walls (TRAY height) so a block carried at scripted_demo.HOVER clears them, and narrow (TRAY width) so two trays
stand side by side in the middle of the table, where both hands reach. Not a twin: in the Blocks object library, in
the beech grain of make_wood_blocks.py.

Usage:
    python scripts/objects/make_block_tray.py [RIGID_DIR]    # default Assets/Object/Blocks/Rigid
    python scripts/objects/object_names.py apply Assets/Object/Blocks
Writes RIGID_DIR/block_tray/00000/{object.usdz, metadata.json, description.json}. Needs what make_wood_blocks.py
needs.

Frame: Z up, metres, origin at bbox centre, long side along Y.
USD:   /root (RigidBodyAPI) -> visual/model (floor and walls, wood texture) -> collision/hull_00 floor, hull_01-04 walls
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_sorting as bs  # noqa: E402
import make_pitcher as mp  # noqa: E402  (shared USD helpers; starts Kit when pxr needs it)
import make_plastic_cups as pc  # noqa: E402  (metadata)
import make_wood_blocks as wb  # noqa: E402  (wood texture, UVs, packing)

RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/Blocks/Rigid"
CATEGORY, INDEX = bs.BOX
WALL, FLOOR = bs.TRAY_WALL, bs.TRAY_FLOOR
DENSITY = 749.0  # kg/m3: the weighed wood blocks' (make_wood_blocks.py)


def boxes():
    """(lo, hi) corners of the floor slab and the four walls, tray frame with its underside at z = 0."""
    w, d, h = bs.BOX_SIZE
    x, y = w / 2, d / 2
    return [
        ((-x, -y, 0.0), (x, y, FLOOR)),
        ((-x, -y, FLOOR), (-x + WALL, y, h)),
        ((x - WALL, -y, FLOOR), (x, y, h)),
        ((-x + WALL, -y, FLOOR), (x - WALL, -y + WALL, h)),
        ((-x + WALL, y - WALL, FLOOR), (x - WALL, y, h)),
    ]


def build():
    pts, faces, hulls = [], [], []
    for lo, hi in boxes():
        p, f = mp.prism([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])], lo[2], hi[2])
        faces += [[i + len(pts) for i in face] for face in f]
        pts += list(p)
        hulls.append(np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
    pts = np.array(pts)
    assert mp.signed_volume(pts, faces) > 0, "tray has inward-facing winding"
    volume = sum(np.prod(np.subtract(hi, lo)) for lo, hi in boxes())
    return dict(pts=pts, faces=faces, hulls=hulls, kind="box"), float(volume)


def description(uid):
    caption = {
        "color": ["natural wood", "light beech", "unpainted"],
        "material": ["wood", "unpainted solid beech", "visible wood grain"],
        "size": ["about 7 cm by 13 cm", "3.5 cm high", "holds two blocks"],
        "name": ["tray", "wooden tray", "block tray"],
        "shape": ["narrow open box", "low walls on four sides", "flat floor"],
        "function": ["holding sorted blocks", "a place to put blocks", "sorting into"],
    }
    desc = [
        "narrow unpainted wooden tray with low walls",
        "small open wooden box for sorting toy blocks into",
        "light beech tray about 7 cm wide, 13 cm long and 3.5 cm high",
        "wooden tray with a flat floor and four thin walls",
        "tray that holds two wooden blocks side by side",
    ]
    return {"uuid": uid, "caption": caption, "description": desc}


def main(rigid_dir):
    c, volume = build()
    shift = -(c["pts"].min(0) + c["pts"].max(0)) / 2
    c = dict(c, pts=c["pts"] + shift, hulls=[h + shift for h in c["hulls"]])
    d = Path(rigid_dir) / CATEGORY / f"{INDEX:05d}"
    d.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{CATEGORY}/{INDEX}"))
    texture = wb.wood_texture(Path(tempfile.mkdtemp()) / "wood.png")
    wb.write_usdz(d / "object.usdz", c, None, texture, np.zeros(2))
    ext = np.ptp(c["pts"], axis=0)
    mass = round(DENSITY * volume, 4)
    meta = pc.metadata(uid, dict(pts=c["pts"], faces=c["faces"], contact_r=float(min(ext[:2]) / 2)), mass, 0 * shift)
    (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False))
    (d / "description.json").write_text(json.dumps(description(uid), indent=4))
    print(f"{d}: {ext.round(4)} m, {mass * 1000:.1f} g, {len(c['hulls'])} hulls")
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
