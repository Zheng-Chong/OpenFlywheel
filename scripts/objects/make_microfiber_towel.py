"""Blue microfiber cleaning towel (Garment/microfiber_towel/00000, 30x30 cm) and its size variant (/00001, 30x70 cm)
as PhysX particle-cloth assets.

Source: platform registration Garment/microfiber_towel/00000 by 种政 (agent), updated 2026-11-02 16:3x: three product
images (not shots of this towel), the nominal 30x30 cm product size (not measured with a tape), 25 g weighed by the
user, "almost no stretch" (user). The 30x70 cm variant keeps the base's areal density, colour and hem. Stretch and
contact parameters are tuned candidates (check_cloth.py runs), not fitted to measured fabric data.

Usage:
    python scripts/objects/make_microfiber_towel.py [TYPE_DIR]    # default Assets/Object/Real2Sim/Garment
Writes TYPE_DIR/microfiber_towel/<index>/{object.usd, object.usdz, textures/, mesh.npz, metadata.json,
runtime_config.json, description.json}; then run object_names.py apply on the object root.

Coordinates: metres, Z up, X along the length, origin at the centre of the flat towel (z = 0 is its midsurface).
USD: /root (defaultPrim, no physics schema: GarmentObject applies the particle cloth at load)
     -> /root/visual/model (the single cloth mesh: simulation and render mesh, st UVs) -> /root/Looks/towel
The sewn corner tag is not modelled: a particle cloth moves only its own mesh, so a tag needs its own particles.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "real2sim/objects/Garment/microfiber_towel"
BASE = 0
THICKNESS = 0.002  # m, estimated from the photos; the cloth is simulated as a midsurface
SPACING = 0.01  # m, target node spacing (the LeHome cloth's)
HEM = 0.005  # m, hemmed edge band width, estimated from the photos
PX_PER_M = 2000  # texture resolution, 2 px/mm
# index: (length, width) in m; the base's are the registration's nominal "长", "宽", the variant's its product size
TOWELS = {0: (0.30, 0.30), 1: (0.70, 0.30)}
# product photo 3 (flat towel on white): a flat terry patch with even light, (left, top, right, bottom) px
TEXTURE_SOURCE = ("IMG_5005.jpg", (380, 430, 620, 570))
SOURCE_PX_PER_M = 1300  # the towel's ~0.70 m spans ~900 px in that photo
RUNTIME = {  # started from the LeHome wiping cloth's candidates; tuned with check_cloth.py on 2026-11-02:
    # stretch 1e8 (the towel barely stretches; 1e4-1e8 and 64 vs 128 iterations change little, the physics rate does:
    # thread strain while hanging ~5 % at the pinch at 300 Hz, < 1 % at 480 Hz), particle friction 1.0 (microfiber
    # grips; at 0.5 friction pads lose the corners while folding)
    "garment_config": {
        "self_collision": True,
        "self_collision_filter": True,
        "stretch_stiffness": 1e8,
        "bend_stiffness": 0.00005,
        "shear_stiffness": 1000,
        "spring_damping": 0.2,
    },
    "particle_material": {"friction": 1.0, "adhesion": 0.0, "particle_adhesion_scale": 0.0},
    "particle_system": {
        "solver_position_iteration_count": 64,
        "contact_offset": 0.001,
        "rest_offset": 0.0003,
        "particle_contact_offset": 0.001,
        "solid_rest_offset": 0.0003,
        "fluid_rest_offset": 0.0003,
    },
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def registered_size(spec):
    """(length, width) in m from the registration's "长 <mm>，宽 <mm>"."""
    values = dict(re.findall(r"(长|宽)\s*([\d.]+)", spec["dimensions_mm"]))
    return float(values["长"]) / 1000, float(values["宽"]) / 1000


def grid(length, width):
    """Flat rectangle, ~SPACING apart, diagonals alternating per cell so the mesh has no shear bias."""
    nx, ny = round(length / SPACING) + 1, round(width / SPACING) + 1
    x, y = np.meshgrid(np.linspace(-length / 2, length / 2, nx), np.linspace(-width / 2, width / 2, ny))
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            faces += [(a, b, d), (a, d, c)] if (i + j) % 2 else [(a, b, c), (b, d, c)]
    faces = np.asarray(faces, dtype=np.int32)
    uv = np.c_[(points[:, 0] + length / 2) / length, (points[:, 1] + width / 2) / width]
    return points, faces, uv


def signed_areas(points, faces):
    e1, e2 = points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
    return 0.5 * np.cross(e1, e2)[:, 2]


def towel_texture(reference, length, width, seed=0):
    """Full-towel diffuse map: the photo's mean colour plus its terry grain, at true scale, and a darker hem band.

    Only the grain (high-pass) is taken from the photo, so its folds and shading are not printed onto a flat towel;
    feathered random patches (rotated, flipped) of that grain cover the towel without a visible repeat.
    """
    from PIL import Image, ImageFilter

    name, box = TEXTURE_SOURCE
    patch = Image.open(reference / name).convert("RGB").crop(box)
    patch = patch.resize(
        (round(patch.width * PX_PER_M / SOURCE_PX_PER_M), round(patch.height * PX_PER_M / SOURCE_PX_PER_M)),
        Image.LANCZOS,
    )
    a = np.asarray(patch, dtype=np.float32)
    grain = a - np.asarray(patch.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
    h, w = round(width * PX_PER_M), round(length * PX_PER_M)
    size = min(96, *grain.shape[:2])
    window = np.outer(np.hanning(size), np.hanning(size))[..., None]
    rng = np.random.default_rng(seed)
    acc, weight = np.zeros((h + size, w + size, 3), np.float32), np.zeros((h + size, w + size, 1), np.float32)
    for y in range(0, h + 1, size // 3):
        for x in range(0, w + 1, size // 3):
            oy, ox = rng.integers(0, grain.shape[0] - size + 1), rng.integers(0, grain.shape[1] - size + 1)
            piece = np.rot90(grain[oy : oy + size, ox : ox + size], rng.integers(4))
            piece = piece[:, ::-1] if rng.integers(2) else piece
            acc[y : y + size, x : x + size] += piece * window
            weight[y : y + size, x : x + size] += window
    detail = acc[size // 2 : size // 2 + h, size // 2 : size // 2 + w] / np.maximum(
        weight[size // 2 : size // 2 + h, size // 2 : size // 2 + w], 1e-6
    )
    detail *= grain.std() / max(detail.std(), 1e-6)  # overlap averaging lowers the grain's contrast; restore it
    image = a.reshape(-1, 3).mean(0) + detail
    hem = round(HEM * PX_PER_M)
    yy, xx = np.mgrid[:h, :w]
    edge = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    image *= np.where(edge < hem, 0.82, 1.0)[..., None]  # the hem's denser, darker overlock
    image *= np.where(np.abs(edge - hem) < 1.5, 0.75, 1.0)[..., None]  # its inner stitch line
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)), patch.size


def write_usd(dest, points, faces, uv, texture):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    stage = Usd.Stage.CreateNew(str(dest / "object.usd"))
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/root/visual")
    mesh = UsdGeom.Mesh.Define(stage, "/root/visual/model")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    material = UsdShade.Material.Define(stage, "/root/Looks/towel")
    shader = UsdShade.Shader.Define(stage, "/root/Looks/towel/pbr")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    tex = UsdShade.Shader.Define(stage, "/root/Looks/towel/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{texture}")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, "/root/Looks/towel/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    lo, hi = points.min(0), points.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str((dest / "object.usd").resolve())), str(dest / "object.usdz"))


def build(index, type_dir):
    spec = json.loads((SPECS / f"{BASE:05d}/spec.json").read_text(encoding="utf-8"))
    assert registered_size(spec) == TOWELS[BASE], (spec["dimensions_mm"], TOWELS[BASE])
    areal = spec["mass_g"] / 1000 / (TOWELS[BASE][0] * TOWELS[BASE][1])  # weighed base mass over its nominal area
    length, width = TOWELS[index]
    points, faces, uv = grid(length, width)
    assert (signed_areas(points, faces) > 0).all(), "faces must wind counter-clockwise seen from +Z"
    area = float(signed_areas(points, faces).sum())
    assert abs(area - length * width) < 1e-9
    mass = areal * length * width
    dest = type_dir / "microfiber_towel" / f"{index:05d}"
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "textures").mkdir()
    texture = f"microfiber_towel_{index:05d}.png"
    image, patch_px = towel_texture(SPECS / f"{BASE:05d}/reference", length, width, seed=index)
    image.save(dest / "textures" / texture)
    write_usd(dest, points, faces, uv, texture)
    np.savez_compressed(dest / "mesh.npz", points=points, faces=faces)

    lo, hi = points.min(0), points.max(0)
    extents = [float(hi[0] - lo[0]), float(hi[1] - lo[1]), THICKNESS]
    box = {
        "vertices": [
            [x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (-THICKNESS / 2, THICKNESS / 2)
        ],
        "extents": extents,
    }
    mesh_sha = hashlib.sha256(points.astype("<f4").tobytes() + faces.astype("<i4").tobytes()).hexdigest()
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/garment/microfiber_towel/{index}"))
    reference = SPECS / f"{BASE:05d}/reference" / TEXTURE_SOURCE[0]
    variant = {"variant_of": f"Garment/microfiber_towel/{BASE:05d}"} if index != BASE else {}
    dump(
        dest / "metadata.json",
        {
            "uuid": uid,
            "asset_id": f"Garment/microfiber_towel/{index:05d}",
            **variant,
            "physics": {"mass": round(mass, 6), "size": extents},
            # measured on the generated mesh, per the registration's names: 长 = X extent, 宽 = Y extent
            "measurements_mm": {"长": round(extents[0] * 1000, 1), "宽": round(extents[1] * 1000, 1)},
            "geometry": {
                "aligned_bbox": box,
                "oriented_bbox": box,
                "bbox": extents,
                "center": [0, 0, 0],
                "vertices": len(points),
                "faces": len(faces),
            },
            "active": {
                "place": {
                    "up": {
                        "orientation": [1, 0, 0, 0],
                        "projection_circle": {
                            "center": [0, 0, 0, 1, 0, 0, 0],
                            "radius": float(np.linalg.norm(hi[:2])),
                        },
                        "contact_circle": {"center": [0, 0, 0, 1, 0, 0, 0], "radius": float(min(extents[:2]) / 2)},
                    }
                },
                "functional": {},
            },
            "passive": {},
            "backend": "isaac_physx_particle_cloth",
            "deformable": {"family": "cloth", "mesh_sha256": mesh_sha},
            "generator": "scripts/objects/make_microfiber_towel.py",
            "registration_updated_at": spec["updated_at"],
            "dimensions_source": "nominal product size given by the user, not measured",
            "mass_source": (
                "weighed by the user (registration mass_g)"
                if index == BASE
                else f"base areal density {areal * 1000:.1f} g/m2 x {length:.2f} m x {width:.2f} m (size variant)"
            ),
            "areal_density_kg_m2": round(areal, 6),
            "candidate_thickness_m": THICKNESS,
            "material_source": "tuned candidates for an almost inextensible towel (check_cloth.py), not fitted",
            "texture": {
                "file": f"textures/{texture}",
                "source": reference.name,
                "source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "source_box_px": list(TEXTURE_SOURCE[1]),
                "tile_px": list(patch_px),
                "px_per_mm": PX_PER_M / 1000,
                "hem_mm": HEM * 1000,
            },
            "not_modelled": ["sewn corner tag (needs its own particles)", "terry pile thickness (midsurface cloth)"],
            "real_material_calibrated": False,
            "robot_grasp_validated": False,
        },
    )
    dump(
        dest / "runtime_config.json",
        {
            "physics": {
                "backend": "isaac_physx_particle_cloth",
                **RUNTIME,
                "garment_config": {**RUNTIME["garment_config"], "particle_mass": mass / len(points)},
            }
        },
    )
    size = (
        ["300 mm square", "about 2 mm thick", "hand-sized"]
        if length == width
        else ["700 mm long", "300 mm wide", "about 2 mm thick"]
    )
    shape = (
        ["flat square", "hemmed edges", "thin sheet"]
        if length == width
        else ["long rectangle", "hemmed edges", "thin sheet"]
    )
    dump(
        dest / "description.json",
        {
            "uuid": uid,
            "caption": {
                "name": ["towel", "cleaning towel", "microfiber cleaning towel"],
                "color": ["royal blue", "blue", "same-colour hemmed edges"],
                "material": ["microfiber", "terry loop fabric", "soft textile"],
                "size": size,
                "shape": shape,
                "function": ["wiping tables", "cleaning surfaces", "drying spills"],
            },
            "description": [
                f"A blue microfiber cleaning towel, {width * 100:.0f} by {length * 100:.0f} centimetres.",
                "It is a single layer of soft terry loop fabric with a matte surface.",
                "All four edges have a narrow same-colour overlocked hem.",
                "It drapes, folds and crumples like ordinary household cloth.",
                "It is used to wipe tables and clean surfaces.",
            ],
        },
    )
    print("GENERATED", dest, f"{len(points)} nodes, {len(faces)} triangles, {mass * 1000:.1f} g", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("type_dir", type=Path, nargs="?", default=ROOT / "Assets/Object/Real2Sim/Garment")
    parser.add_argument("--texture-only", type=Path, help="write the textures to this directory and stop (no pxr)")
    args = parser.parse_args()
    if args.texture_only:
        args.texture_only.mkdir(parents=True, exist_ok=True)
        for index, (length, width) in TOWELS.items():
            image, _ = towel_texture(SPECS / f"{BASE:05d}/reference", length, width, seed=index)
            image.save(args.texture_only / f"microfiber_towel_{index:05d}.png")
        return
    try:
        import pxr  # noqa: F401

        app = None
    except ImportError:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "fast_shutdown": True})
    try:
        for index in TOWELS:
            build(index, args.type_dir)
    except Exception:
        if app is None:
            raise
        import traceback

        traceback.print_exc()
        os._exit(1)
    if app is not None:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
