#!/usr/bin/env python3
"""Sort cards for g1d_sort_blocks_by_color / g1d_sort_blocks_by_shape: an A7 card lying behind each tray, showing
what goes in it.

One card per block_sorting.CARDS entry (index = card index): a colour card is a rounded swatch of that paint (the hex
make_wood_blocks.PALETTE paints the blocks with), a shape card the block's grey 3D icon (its make_wood_blocks mesh seen
from the front-right and above, back faces culled, far faces drawn first, shaded by its normals). On the pattern cards'
off-white board (make_pattern_cards.card_meshes / write_usdz) cut to block_sorting.CARD_SIZE. Not twins: in the
Blocks object library; 00000 is the card, 00001+ its variants (metadata "variant_of").

Usage:
    python scripts/objects/make_sort_cards.py [RIGID_DIR]    # default Assets/Object/Blocks/Rigid
    python scripts/objects/object_names.py apply Assets/Object/Blocks
Writes RIGID_DIR/block_sort_card/<index>/{object.usdz, metadata.json, description.json, card.png}. Needs what
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
import block_sorting as bs  # noqa: E402
import make_pattern_cards as mpc  # noqa: E402  (card board, packing; starts Kit when pxr needs it)
import make_plastic_cups as pc  # noqa: E402  (metadata)
import make_wood_blocks as wb  # noqa: E402  (block meshes, paints)

RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/Blocks/Rigid"
CATEGORY = "block_sort_card"
CARD = (*bs.CARD_SIZE, mpc.CARD[2])  # the pattern cards' 2 mm board
VIEW = np.radians([30.0, 28.0])  # icon view: yaw from the front toward the right, pitch above the table
LIGHT = np.array([-0.35, -0.55, 0.76])  # toward the light, from the front-left and above
ICON_GREY = np.array([150.0, 150.0, 148.0])
SWATCH_INSET = 0.016  # m from the card edge to the swatch


def view_axes():
    yaw, pitch = VIEW
    toward = np.array([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)])  # to the eye
    right = np.cross(-toward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return toward, right, np.cross(right, -toward)


def icon_faces(cat):
    """[(2D polygon (k, 2) m, brightness 0-1)] of block `cat`, far to near, visible faces only."""
    shape = next(s for s in wb.SHAPES if s["cat"] == cat)
    c = wb.build(shape)
    pts = c["pts"] - (c["pts"].min(0) + c["pts"].max(0)) / 2
    toward, right, up = view_axes()
    light = LIGHT / np.linalg.norm(LIGHT)
    faces = []
    for f in c["faces"]:
        p = pts[f]
        normal = np.sum(np.cross(p, np.roll(p, -1, 0)), axis=0)
        area = np.linalg.norm(normal)
        if area < 1e-12 or normal @ toward <= 0:
            continue
        normal /= area
        faces.append((float(p.mean(0) @ toward), np.c_[p @ right, p @ up], 0.35 + 0.65 * max(0.0, normal @ light)))
    return [(poly, shade) for _, poly, shade in sorted(faces, key=lambda t: t[0])]


def draw(index):
    """Card picture (PIL RGB) for card `index`."""
    w, h = bs.CARD_SIZE
    w_px, h_px = round(w * mpc.PX_PER_M), round(h * mpc.PX_PER_M)
    img = Image.new("RGB", (w_px, h_px), mpc.PAPER)
    d = ImageDraw.Draw(img)
    inset = round(0.004 * mpc.PX_PER_M)
    d.rectangle([inset, inset, w_px - inset, h_px - inset], outline=mpc.BORDER, width=4)
    by, key = bs.CARDS[index]
    if by == "color":
        m = round(SWATCH_INSET * mpc.PX_PER_M)
        d.rounded_rectangle([m, m, w_px - m, h_px - m], radius=round(0.008 * mpc.PX_PER_M), fill=dict(wb.PALETTE)[key])
        return img
    faces = icon_faces(key)
    allpts = np.concatenate([poly for poly, _ in faces])
    lo, hi = allpts.min(0), allpts.max(0)
    scale = min((w - 2 * mpc.MARGIN) / np.ptp(allpts[:, 0]), (h - 2 * mpc.MARGIN) / np.ptp(allpts[:, 1])) * 0.85
    centre = (lo + hi) / 2
    for poly, shade in faces:
        q = (poly - centre) * scale * mpc.PX_PER_M
        rgb = tuple(int(v) for v in np.clip(ICON_GREY * shade, 0, 255))
        d.polygon([(w_px / 2 + x, h_px / 2 - y) for x, y in q], fill=rgb, outline=rgb)
    return img


def description(uid, by, key):
    name = key if by == "color" else key.removeprefix("wood_block_").replace("_", " ")
    what = f"a {name} colour swatch" if by == "color" else f"a grey {name} block"
    caption = {
        "color": ["off-white", f"white card with {what}", "light"],
        "material": ["paper", "paperboard", "printed card"],
        "size": ["A7 card", "about 7 cm by 10 cm", "2 mm thick"],
        "name": ["sorting card", f"{name} {'colour' if by == 'color' else 'shape'} card", "box label card"],
        "shape": ["flat rectangle", "thin card", "lies flat on the table"],
        "function": ["showing what goes in the tray in front of it", "a sorting label", "visual reference"],
    }
    desc = [
        f"flat card printed with {what}",
        "A7 paper card lying on the table behind a tray",
        f"sorting card: blocks matching its {'colour' if by == 'color' else 'shape'} go in the tray in front of it",
        f"off-white card showing {what}",
        "thin printed card that labels a tray for sorting blocks",
    ]
    return {"uuid": uid, "caption": caption, "description": desc}


def main(rigid_dir):
    (box_pts, _), _, _ = mpc.card_meshes(CARD)
    for idx, (by, key) in enumerate(bs.CARDS):
        d = Path(rigid_dir) / CATEGORY / f"{idx:05d}"
        d.mkdir(parents=True, exist_ok=True)
        uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{CATEGORY}/{idx}"))
        picture = draw(idx)
        picture.save(d / "card.png")  # the card face, for the platform and for checking by eye
        # one texture name per card: two cards sharing "textures/pattern.png" in one scene both rendered card 0's face
        mpc.write_usdz(d / "object.usdz", picture, CARD, f"textures/{CATEGORY}_{idx:05d}.png")
        meta = pc.metadata(uid, dict(pts=box_pts, faces=[], contact_r=float(CARD[0] / 2)), mpc.MASS, np.zeros(3))
        meta["geometry"]["faces"] = 6
        meta["sort"] = {"index": idx, "by": by, "key": key, "zh": bs.label_zh(by, key)}
        if idx:
            meta["variant_of"] = "Rigid/block_sort_card/00000"  # one card, a colour or shape per variant
        (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False))
        (d / "description.json").write_text(json.dumps(description(uid, by, key), indent=4))
        print(f"{d}: {by} {key}")
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
