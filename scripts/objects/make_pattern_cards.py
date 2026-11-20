#!/usr/bin/env python3
"""Pattern cards for g1d_build_block_pattern: an A6 card lying on the table with the target build printed on it.

One card per pattern of block_patterns.PATTERNS (index = pattern index): the build seen from the front, every block in
the unpainted wood of make_wood_blocks.py or in its paint (block_patterns.paints, make_wood_blocks.PALETTE) with a dark
outline, standing on a ground line, on off-white card. The card is what tells the policy which shape (and colours) to
build. Not twins: they live in the Blocks object library with the blocks; 00000 is the card, 00001+ its variants
(metadata "variant_of"), one picture each.

Usage:
    python scripts/objects/make_pattern_cards.py [RIGID_DIR]    # default Assets/Object/Blocks/Rigid
    python scripts/objects/object_names.py apply Assets/Object/Blocks
Writes RIGID_DIR/block_pattern_card/<index>/{object.usdz, metadata.json, description.json}. Needs what
make_wood_blocks.py needs.

Frame: Z up, metres, origin at bbox centre, card in the X-Y plane; the picture's top edge on +Y, so a card lying in
front of the robot (which looks along +Y) reads upright to it.
USD:   /root (RigidBodyAPI) -> visual/model (card), visual/print (picture, textures/pattern.png) -> collision/hull_00
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_patterns as bp  # noqa: E402
import make_pitcher as mp  # noqa: E402  (shared USD helpers; starts Kit when pxr needs it)
from make_pitcher import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
import make_plastic_cups as pc  # noqa: E402  (add_collision, metadata)
import make_wood_blocks as wb  # noqa: E402  (wood texture, sRGB helper)

RIGID_DIR = Path(__file__).resolve().parents[2] / "Assets/Object/Blocks/Rigid"
CATEGORY = "block_pattern_card"
CARD = (0.148, 0.105, 0.002)  # A6 landscape, 2 mm board
MASS = 0.02  # 2 mm paperboard at ~0.7 g/cm3
PX_PER_M = 10000  # 10 px/mm
MARGIN = 0.012  # free card round the picture
PAPER, BORDER = "#F4F1EA", "#C9C2B5"
PRINT_WOOD = 1.45  # the wood texture is dark for RTX lighting; printed on the card it is brightened to look like wood
INK, GROUND = (90, 58, 38), (156, 148, 136)
PRINT_UP = 0.0002  # picture above the card's top face
PRINT_TEX = "textures/pattern.png"


def draw(pattern, wood):
    """Card picture (PIL RGB): the pattern's front view scaled to fill the card inside MARGIN."""
    w_px, h_px = round(CARD[0] * PX_PER_M), round(CARD[1] * PX_PER_M)
    img = Image.new("RGB", (w_px, h_px), PAPER)
    d = ImageDraw.Draw(img)
    inset = round(0.004 * PX_PER_M)
    d.rectangle([inset, inset, w_px - inset, h_px - inset], outline=BORDER, width=4)
    polys = [bp.outline(cat, x, z) for cat, x, z in pattern["pieces"]]
    pts = np.concatenate(polys)
    lo, hi = pts.min(0), pts.max(0)
    ground_gap = 0.006 / 2  # picture units below the pieces for the ground line
    scale = min((CARD[0] - 2 * MARGIN) / (hi[0] - lo[0]), (CARD[1] - 2 * MARGIN) / (hi[1] - lo[1] + ground_gap))
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1] - ground_gap) / 2])

    def px(p):
        q = (np.asarray(p) - centre) * scale
        return [(w_px / 2 + x * PX_PER_M, h_px / 2 - z * PX_PER_M) for x, z in q]

    base = px([(lo[0] - 0.01, lo[1] - ground_gap), (hi[0] + 0.01, lo[1] - ground_gap)])
    d.line(base, fill=GROUND, width=6)
    rng = np.random.default_rng(len(pattern["pieces"]))
    for poly, paint in zip(polys, bp.paints(pattern)):
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(px(poly), fill=255)
        tone = rng.uniform(0.92, 1.05)  # neighbours read apart
        if paint:
            hex_rgb = wb.PALETTE[paint][1]
            rgb = np.array([int(hex_rgb[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
            img.paste(tuple(int(v) for v in np.clip(rgb * tone, 0, 255)), (0, 0), mask)
        else:
            shade = np.clip(wood * PRINT_WOOD * tone, 0, 255).astype(np.uint8)
            img.paste(Image.fromarray(shade).resize(img.size), (0, 0), mask)
        d.line(px(np.r_[poly, poly[:1]]), fill=INK, width=5, joint="curve")
    return img


def card_meshes(card=CARD):
    """Card box and the picture quad on its top face (with UVs), card frame (centre at the origin)."""
    w, h, t = card
    box = mp.prism([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)], -t / 2, t / 2)
    z = t / 2 + PRINT_UP
    quad = np.array([(-w / 2, -h / 2, z), (w / 2, -h / 2, z), (w / 2, h / 2, z), (-w / 2, h / 2, z)])
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)
    return box, (quad, [[0, 1, 2, 3]]), uv


def write_usdz(out, picture, card=CARD, texture=PRINT_TEX):
    tmp = Path(tempfile.mkdtemp())
    picture.save(tmp / "pattern.png")
    st = Usd.Stage.CreateNew(str(tmp / "object.usdc"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = mp.add_xform(st, "/root")
    st.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mp.add_xform(st, "/root/visual")
    (box_pts, box_faces), (quad, quad_faces), uv = card_meshes(card)
    assert mp.signed_volume(box_pts, box_faces) > 0, "card has inward-facing winding"
    card = mp.add_mesh(st, "/root/visual/model", box_pts, box_faces, mp.vertex_normals(box_pts, box_faces))
    rgb = wb.srgb(PAPER)
    paper = mp.add_material(
        st, "/root/_materials/Paper", "OmniPBR", {"diffuse_color_constant": rgb, "reflection_roughness_constant": 0.85},
        rgb, 1.0, 0.85,
    )  # fmt: skip
    UsdShade.MaterialBindingAPI.Apply(card.GetPrim()).Bind(paper)
    face = mp.add_mesh(st, "/root/visual/print", quad, quad_faces, np.tile([0.0, 0.0, 1.0], (4, 1)))
    UsdGeom.PrimvarsAPI(face).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        Vt.Vec2fArray.FromNumpy(uv[quad_faces[0]].astype(np.float32))
    )
    ink = mp.add_material(
        st, "/root/_materials/Print", "OmniPBR", {"reflection_roughness_constant": 0.8, "metallic_constant": 0.0},
        rgb, 1.0, 0.8,
    )  # fmt: skip
    UsdShade.Shader.Get(st, "/root/_materials/Print/Shader").CreateInput(
        "diffuse_texture", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(texture))
    UsdShade.MaterialBindingAPI.Apply(face.GetPrim()).Bind(ink)
    pc.add_collision(st, [box_pts[np.unique(np.concatenate(box_faces))]])
    st.Save()
    zip_writer = getattr(Sdf, "ZipFileWriter", None) or Usd.ZipFileWriter
    zw = zip_writer.CreateNew(str(out))
    zw.AddFile(str(tmp / "object.usdc"), "object.usdc")
    zw.AddFile(str(tmp / "pattern.png"), texture)
    zw.Save()


def description(uid, pattern):
    name = pattern["name"].replace("_", " ")
    colours = "coloured" if any(bp.paints(pattern)) else "wood-coloured"
    caption = {
        "color": ["off-white", f"white card with {colours} drawing", "light"],
        "material": ["paper", "paperboard", "printed card"],
        "size": ["A6 card", "about 15 cm by 10 cm", "2 mm thick"],
        "name": ["pattern card", f"card showing a block {name}", "building instruction card"],
        "shape": ["flat rectangle", "thin card", "lies flat on the table"],
        "function": ["showing what to build", "a picture to copy with blocks", "visual reference"],
    }
    desc = [
        f"flat card with a drawing of a {name} built from wooden blocks",
        "A6 paper card lying on the table, printed with a block pattern",
        "off-white card showing the front view of a toy block build",
        f"picture card to copy: wooden blocks stacked into a {name}",
        "thin printed card that shows which shape to build",
    ]
    return {"uuid": uid, "caption": caption, "description": desc}


def main(rigid_dir):
    wood = np.asarray(Image.open(wb.wood_texture(Path(tempfile.mkdtemp()) / "wood.png")).convert("RGB"), dtype=float)
    (box_pts, _), _, _ = card_meshes()
    for idx, pattern in enumerate(bp.PATTERNS):
        assert not bp.overlaps(pattern) and all(bp.supported(pattern)), f"pattern {pattern['name']} cannot stand"
        d = Path(rigid_dir) / CATEGORY / f"{idx:05d}"
        d.mkdir(parents=True, exist_ok=True)
        uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/rigid/{CATEGORY}/{idx}"))
        picture = draw(pattern, wood)
        picture.save(d / "pattern.png")  # the card face, for the platform and for checking by eye
        write_usdz(d / "object.usdz", picture)
        meta = pc.metadata(uid, dict(pts=box_pts, faces=[], contact_r=float(CARD[1] / 2)), MASS, np.zeros(3))
        meta["geometry"]["faces"] = 6
        meta["pattern"] = {"index": idx, "name": pattern["name"], "zh": pattern["zh"]}
        if idx:
            meta["variant_of"] = "Rigid/block_pattern_card/00000"  # one card, a picture per variant
        (d / "metadata.json").write_text(json.dumps(meta, indent=4, ensure_ascii=False))
        (d / "description.json").write_text(json.dumps(description(uid, pattern), indent=4))
        print(f"{d}: {pattern['name']} ({pattern['zh']}), {len(pattern['pieces'])} pieces, {np.ptp(box_pts, 0)} m")
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
