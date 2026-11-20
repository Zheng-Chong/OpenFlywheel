"""Measured wooden arch, 杨开元 conversation confirmation 2026-10-29.

Usage: python scripts/objects/make_wooden_arch_block.py [RIGID_DIR]
Metres, Z up, whole visible bounding-box centre. X length, Y tunnel depth.
The minimum crown thickness is measured at the arch apex in the centre plane.
An earlier untracked multi-object generator remains unchanged; this entry
isolates this registration from the other objects and their unpublished work.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid

import cv2
import make_pitcher as mp
from make_pitcher import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
import make_plastic_cups as cups
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
import trimesh

ROOT = Path(__file__).resolve().parents[2]
ID = "Rigid/wooden_arch_block/00000"
REVISION = "ff33fb65e47df98cac53a62e1c55494a3c78ab537595ae5c8748367a95bc6e75"
PHOTOS = {
    "original_20261024.jpg": "925069375e2dd4ac594b6d01f6cb5c5e2d8921dbb08b35de6350adf11556be79",
    "measurement_20261029_03.jpg": "2a76597522ed89298ad1846122d47137c454e80295a40f0d3b6fec8bacc33757",
}
# User-confirmed measured envelope and crown; SI throughout geometry.
L, D, H, CROWN, MASS = 0.06010, 0.02972, 0.03004, 0.00958, 0.0246
R = H - CROWN  # semicircular aperture, centre at outside bottom, inferred circularity from photo
# Source photo face polygons, clockwise from upper left. Original 632x494 pixels.
PANELS = {
    "front": ([[161, 292], [629, 143], [578, 348], [144, 491]], (601, 301)),
    "top": ([[14, 120], [473, 1], [629, 143], [161, 292]], (601, 298)),
    "left": ([[14, 120], [161, 292], [144, 491], [4, 330]], (298, 301)),
    "inner_observed": ([[455, 274], [475, 286], [520, 339], [512, 351]], (298, 161)),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    folder = ROOT / "real2sim/objects" / ID
    spec = json.loads((folder / "spec.json").read_text())
    text = (folder / "modeling_prompt.md").read_text()
    data = json.JSONDecoder().raw_decode(text[text.index("{", text.index("## 已确认输入")) :])[0]
    digest = hashlib.sha256(
        json.dumps({"version": "real-object-v3", "inputs": data}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == spec["modeling_confirmation"]["revision"] == REVISION
    mapping = {"dimensions": "dimensions_mm", "real_mass": "mass_g", "real_friction": "static_friction"}
    for key in data:
        if key in ("gen", "photos", "existing_model"):
            continue
        assert data[key] == spec.get(mapping.get(key, key)), key
    assert spec["dimensions_mm"] == "长度60.10，宽度29.72，高度30.04，最薄处9.58"
    assert spec["mass_g"] == 24.6
    assert not data["gen"].get("count")
    assert {p.name for p in (folder / "reference").glob("*.jpg")} == set(PHOTOS)
    for name, digest in PHOTOS.items():
        assert sha(folder / "reference" / name) == digest
    return folder


def build():
    arc = np.linspace(np.pi, 0, 129)
    polygon = [(-L / 2, 0)] + [(R * np.cos(t), R * np.sin(t)) for t in arc]
    polygon += [(L / 2, 0), (L / 2, H), (-L / 2, H)]
    mesh = trimesh.creation.extrude_polygon(Polygon(polygon), D, engine="earcut")
    # Original extrusion axes are x / cross-section height / depth. Swap Y,Z
    # and reverse winding because this coordinate transform reflects handedness.
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    mesh.vertices -= [0, D / 2, H / 2]
    mesh.faces = mesh.faces[:, ::-1]
    assert mesh.is_watertight and mesh.volume > 0
    assert np.allclose(mesh.extents, [L, D, H], atol=1e-9)
    hulls = []
    for xa, xb in [(-L / 2, -R), (R, L / 2)]:
        hulls.append(np.array([(x, y, z) for x in [xa, xb] for y in [-D / 2, D / 2] for z in [-H / 2, H / 2]]))
    angles = np.linspace(0, np.pi, 33)
    for a, b in zip(angles, angles[1:]):
        x0, z0 = R * np.cos(a), R * np.sin(a) - H / 2
        x1, z1 = R * np.cos(b), R * np.sin(b) - H / 2
        hulls.append(
            np.array([(x, y, z) for x, z in [(x0, z0), (x1, z1), (x1, H / 2), (x0, H / 2)] for y in [-D / 2, D / 2]])
        )
    # A 32-segment arch collider differs from the 128-segment visual by <25um.
    from scipy.spatial import ConvexHull

    probes = np.array(
        [
            (x, 0, z - H / 2)
            for x in np.linspace(-R + 0.001, R - 0.001, 21)
            for z in np.linspace(0.001, R - 0.001, 21)
            if x * x + z * z < (R - 0.0001) ** 2
        ]
    )
    for hull in hulls:
        convex = ConvexHull(hull)
        assert len(convex.vertices) <= 64
        assert not np.any(np.all(probes @ convex.equations[:, :3].T + convex.equations[:, 3] < -1e-7, axis=1))
    return mesh, hulls


def textures(folder, output):
    original = np.asarray(Image.open(folder / "reference/original_20261024.jpg").convert("RGB"))
    annotation = Image.fromarray(original)
    draw = ImageDraw.Draw(annotation)
    for name, (quad, size) in PANELS.items():
        draw.polygon([tuple(p) for p in quad], outline="red", width=2)
        draw.text(tuple(quad[0]), name, fill="red")
        width, height = size
        transform = cv2.getPerspectiveTransform(
            np.float32(quad), np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        )
        image = cv2.warpPerspective(original, transform, size).astype(float) / 255
        background = ((image.min(2) > 0.85) | ((image[:, :, 0] - image[:, :, 2]) < 0.065)).astype("uint8") * 255
        image = cv2.inpaint((image * 255).astype("uint8"), background, 5, cv2.INPAINT_TELEA).astype(float) / 255
        # Remove only broad photographic shading; no generated grain or text.
        luminance = image.mean(2)
        valid = (luminance < 0.94).astype(float)
        smooth = gaussian_filter(luminance * valid, 30) / np.maximum(gaussian_filter(valid, 30), 0.001)
        image *= 0.62 / np.maximum(smooth[:, :, None], 0.12)
        image = np.clip(image, 0, 1)
        Image.fromarray((image * 255).astype("uint8")).save(output / f"{name}.png")
    annotation.save(output / "photo_regions.png")
    info = dict(
        source_sha256=PHOTOS["original_20261024.jpg"],
        source_size_px=[632, 494],
        panels=PANELS,
        upsampled=True,
        native_grain_detail_not_increased=True,
        missing_faces=["rear", "right end", "underside", "unobserved inner arch sectors"],
    )
    (output / "texture_provenance.json").write_text(json.dumps(info, indent=2))


def textured_material(stage, name, file):
    mat = UsdShade.Material.Define(stage, "/root/_materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.64)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    return mat


def write_usdz(path, mesh, hulls, texture_folder):
    folder = Path(tempfile.mkdtemp(prefix="wooden-arch-"))
    stage = Usd.Stage.CreateNew(str(folder / "object.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = mp.add_xform(stage, "/root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    physical = UsdPhysics.MassAPI.Apply(root)
    physical.CreateMassAttr(MASS)
    physical.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, mesh.center_mass)))
    inertia = mesh.moment_inertia * (MASS / mesh.volume)
    assert np.max(np.abs(inertia - np.diag(np.diag(inertia)))) < 1e-10
    physical.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, np.diag(inertia))))
    mp.add_xform(stage, "/root/visual")
    model = mp.add_mesh(stage, "/root/visual/model", mesh.vertices, mesh.faces.tolist())
    normals = np.repeat(mesh.face_normals, 3, axis=0)
    model.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype("float32")))
    model.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    fallback = mp.add_material(
        stage,
        "/root/_materials/UnobservedWood",
        "OmniPBR",
        {"diffuse_color_constant": (0.53, 0.34, 0.20), "reflection_roughness_constant": 0.64, "metallic_constant": 0.0},
        (0.53, 0.34, 0.20),
        1.0,
        0.64,
    )
    UsdShade.MaterialBindingAPI.Apply(model.GetPrim()).Bind(fallback)
    uv = []
    groups = {key: [] for key in PANELS}
    for idx, (face, normal) in enumerate(zip(mesh.faces, mesh.face_normals)):
        points = mesh.vertices[face]
        center = points.mean(0)
        if normal[1] < -0.9:
            key = "front"
            coords = np.c_[(points[:, 0] + L / 2) / L, (points[:, 2] + H / 2) / H]
        elif normal[2] > 0.9:
            key = "top"
            coords = np.c_[(points[:, 0] + L / 2) / L, (points[:, 1] + D / 2) / D]
        elif normal[0] < -0.9 and center[0] < -L / 2 + 0.0001:
            key = "left"
            coords = np.c_[(D / 2 - points[:, 1]) / D, (points[:, 2] + H / 2) / H]
        elif center[0] > 0 and normal[0] < -0.7 and normal[2] < 0:
            key = "inner_observed"
            theta = np.arctan2(points[:, 2] + H / 2, points[:, 0])
            coords = np.c_[(points[:, 1] + D / 2) / D, theta / (np.pi / 4)]
        else:
            key = None
            coords = np.zeros((3, 2))
        uv.extend(np.clip(coords, 0, 1))
        if key:
            groups[key].append(idx)
    primvar = UsdGeom.PrimvarsAPI(model).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype="float32")))
    (folder / "textures").mkdir()
    for name, indices in groups.items():
        import shutil

        shutil.copy2(texture_folder / f"{name}.png", folder / f"textures/{name}.png")
        material = textured_material(stage, name, f"textures/{name}.png")
        subset = UsdGeom.Subset.CreateGeomSubset(model, name, UsdGeom.Tokens.face, Vt.IntArray(indices), "materialBind")
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    UsdGeom.Subset.SetFamilyType(model, "materialBind", UsdGeom.Tokens.nonOverlapping)
    cups.add_collision(stage, hulls)
    stage.Save()
    writer = (getattr(Sdf, "ZipFileWriter", None) or Usd.ZipFileWriter).CreateNew(str(path))
    writer.AddFile(str(folder / "object.usdc"), "object.usdc")
    for name in groups:
        writer.AddFile(str(folder / "textures" / f"{name}.png"), f"textures/{name}.png")
    writer.Save()
    return inertia


def main(output):
    source = verify()
    mesh, hulls = build()
    destination = Path(output) / "wooden_arch_block/00000"
    destination.mkdir(parents=True, exist_ok=True)
    tex = destination / "textures"
    tex.mkdir(exist_ok=True)
    textures(source, tex)
    inertia = write_usdz(destination / "object.usdz", mesh, hulls, tex)
    c = dict(pts=mesh.vertices, faces=mesh.faces, contact_r=D / 2)
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, "RoboDojo/" + ID))  # retain existing asset UUID
    meta = cups.metadata(uid, c, MASS, np.zeros(3))
    lo, hi = mesh.bounds
    arch_apex = mesh.vertices[(abs(mesh.vertices[:, 0]) < 1e-8) & (mesh.vertices[:, 2] < hi[2] - 1e-8)]
    assert len(arch_apex), "arch apex missing"
    crown = float(hi[2] - arch_apex[:, 2].max()) * 1000
    measurements = dict(zip(["长度", "宽度", "高度"], np.round((hi - lo) * 1000, 2)))
    measurements["最薄处"] = round(crown, 2)
    assert np.allclose(list(measurements.values()), [60.1, 29.72, 30.04, 9.58], atol=1e-6)
    meta.update(
        asset_id=ID,
        measurements_mm=measurements,
        registration_revision=REVISION,
        dimension_policy="measured",
        mass_policy="measured",
        author="杨开元",
        generator="scripts/objects/make_wooden_arch_block.py",
        source_photo_sha256=PHOTOS,
        inertia_kg_m2=inertia.tolist(),
        center_of_mass_m=mesh.center_mass.tolist(),
        inertia_basis="uniform-density solid arch geometry scaled to measured total mass; not inertia measurement",
        geometry_assumptions=dict(aperture="semicircle through full Y depth", radius_m=R, edge_radius_measured=None),
        collision=dict(
            method="two piers and 32 arch staves",
            count=len(hulls),
            maximum_chord_error_mm=R * (1 - np.cos(np.pi / 64)) * 1000,
        ),
        friction_source="unmeasured; production defaults",
        real_material_calibrated=False,
        missing_photos=["rear", "right end", "underside", "unobserved arch interior sectors"],
        texture_source="perspective-corrected original photograph; broad shading removed; no procedural grain",
    )
    # The support consists of two piers; never describe the hollow centre as a contact disk.
    meta["active"]["place"]["up"].pop("contact_circle", None)
    meta["active"]["place"]["up"]["contact_regions"] = [
        dict(x_mm=[-L * 500, -R * 1000], y_mm=[-D * 500, D * 500]),
        dict(x_mm=[R * 1000, L * 500], y_mm=[-D * 500, D * 500]),
    ]
    desc = dict(
        uuid=uid,
        caption=dict(
            name=["block", "wooden block", "wooden arch block"],
            color=["natural wood", "light brown", "tan wood grain"],
            material=["wood", "wooden solid", "wood grain surface"],
            size=["60 mm long", "30 mm deep", "30 mm tall"],
            shape=["rectangular outer body", "semicircular bottom opening", "two flat piers"],
            function=["stacking blocks", "bridging small objects", "grasping and placement"],
        ),
        description=[
            "Light brown wooden block with a semicircular tunnel through its base.",
            "Rectangular wooden arch with photographed grain on the visible faces.",
            "Small wooden bridge block with two flat supporting piers.",
            "Wooden arch block measuring 60.10 by 29.72 by 30.04 mm.",
            "Natural wood block with a 9.58 mm thick crown above its opening.",
        ],
    )
    for name, value in [
        ("metadata.json", meta),
        ("description.json", desc),
        ("runtime_config.json", {"physics": {"mass": MASS}}),
    ]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(destination),
                measurements_mm=measurements,
                mass_kg=MASS,
                sha256=sha(destination / "object.usdz"),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Assets/Object/Real2Sim/Rigid")
    except Exception:
        if mp.APP is None:
            raise
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if mp.APP is not None:
        mp.APP.close()
