#!/usr/bin/env python3
"""The target mark of g1d_place_block_on_mark (block_atomic.py): a 70 mm square of thin card lying on the table with a
red disc printed on it, where the block is to be set down. On the pattern cards' off-white board
(make_pattern_cards.card_meshes / write_usdz). Not a twin: in the Blocks object library.

Usage:
    python scripts/objects/make_block_mark.py [RIGID_DIR]    # default Assets/Object/Blocks/Rigid
    python scripts/objects/object_names.py apply Assets/Object/Blocks
Writes RIGID_DIR/block_mark/00000/{object.usdz, metadata.json, description.json, mark.png}. Needs what
make_wood_blocks.py needs.
"""

import json
import os
from pathlib import Path
import sys
import traceback
import uuid

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_atomic as ba  # noqa: E402
import make_pattern_cards as mpc  # noqa: E402  (card board, packing; starts Kit when pxr needs it)
import make_plastic_cups as pc  # noqa: E402  (metadata)
import make_wood_blocks as wb  # noqa: E402  (paints)

RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/Blocks/Rigid"
CATEGORY, INDEX = ba.MARK
CARD = (0.070, 0.070, 0.001)  # 1 mm card
MASS = 0.005
DISC_D = 2 * (ba.MARK_RADIUS + 0.005)  # 50 mm: a block centred within MARK_RADIUS stands on it
COLOUR = wb.PALETTE[2][1]  # the blocks' red


def draw():
    w_px, h_px = (round(v * mpc.PX_PER_M) for v in CARD[:2])
    img = Image.new("RGB", (w_px, h_px), mpc.PAPER)
    d = ImageDraw.Draw(img)
    inset = round(0.003 * mpc.PX_PER_M)
    d.rectangle([inset, inset, w_px - inset, h_px - inset], outline=mpc.BORDER, width=3)
    r = DISC_D / 2 * mpc.PX_PER_M
    d.ellipse([w_px / 2 - r, h_px / 2 - r, w_px / 2 + r, h_px / 2 + r], fill=COLOUR)
    return img


def description(uid):
    caption = {
        "color": ["off-white", "white card with a red disc", "red"],
        "material": ["paper", "card", "printed card"],
        "size": ["small", "7 cm square", "1 mm thick"],
        "name": ["mark", "round red mark", "target mark"],
        "shape": ["flat square", "thin card", "lies flat on the table"],
        "function": ["where to put a block", "a target to place on", "visual reference"],
    }
    desc = [
        "small square card printed with a red disc",
        "round red target mark lying flat on the table",
        "thin off-white card with a 5 cm red circle to set a block on",
        "place mark: a red dot on a square of card",
        "flat target mark that shows where to put the block",
    ]
    return {"uuid": uid, "caption": caption, "description": desc}


def main(rigid_dir):
    (box_pts, _), _, _ = mpc.card_meshes(CARD)
    d = Path(rigid_dir) / CATEGORY / f"{INDEX:05d}"
    d.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{CATEGORY}/{INDEX}"))
    picture = draw()
    picture.save(d / "mark.png")
    mpc.write_usdz(d / "object.usdz", picture, CARD, f"textures/{CATEGORY}_{INDEX:05d}.png")
    meta = pc.metadata(uid, dict(pts=box_pts, faces=[], contact_r=float(CARD[0] / 2)), MASS, np.zeros(3))
    meta["geometry"]["faces"] = 6
    (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False))
    (d / "description.json").write_text(json.dumps(description(uid), indent=4))
    print(f"{d}: {CARD} m, red disc {DISC_D * 1000:.0f} mm")
    sys.stdout.flush()  # APP.close() exits without flushing a redirected stdout


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else RIGID_DIR)
    except Exception:
        if mpc.mp.APP is None:
            raise
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if mpc.mp.APP is not None:
        mpc.mp.APP.close()
