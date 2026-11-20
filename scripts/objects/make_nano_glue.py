"""Nano Glue physical twin, measured by 杨开元, spec 2026-10-29 23:32:21.

Usage: python scripts/objects/make_nano_glue.py [RIGID_DIR]
SI metres/kg, Z up, origin at complete visible bounding-box centre.
Measured outer/inner diameters are maximum outside and clear through-hole
diameters, height is axial envelope. No extra real dimensions are estimated.
Rigid approximation requested by registration; no adhesive/deformation model.
Photo-derived appearance is incomplete: unobserved inner printing stays plain.
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
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
ID = "Rigid/nano_glue/00000"
REVISION = "5f644898fc244b11b83668e12489e60aaff67d35e0745d9b178c9f8f4ba6b939"
# All physical envelope values measured. Offsets below are render tolerances only.
RO, RI, H, MASS = 0.07250 / 2, 0.03877 / 2, 0.02985, 0.0918
N = 256
PHOTOS = {
    "2c1d762bf053a64f4138d29991f6bda6.jpg": "b7a5a4adce9fd734463aa1903e17206b93d8432a628ea6ab03890371849e2d40",
    "4f864028563e7be05ac3abdbf265d9f8.jpg": "f5b4de73f020f89bed2ece13e397fbe1a22ee8cc326cb8bae07df1542d11aabe",
    "ef4cafd542500cd01e9cd0fd0ce5cb33.jpg": "c7b2532f194d88d4cea71079881efcabdcbf8bc5a0b612902eaadbacb54c004e",
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
        if key not in ("gen", "photos", "existing_model"):
            assert data[key] == spec.get(mapping.get(key, key)), key
    assert not any(data["gen"].values())
    assert spec["mass_g"] == 91.8
    assert spec["dimensions_mm"] == "外径72.50，内径38.77，高度29.85"
    assert {p.name for p in (folder / "reference").glob("*.jpg")} == set(PHOTOS)
    for p in data["photos"]:
        file = folder / "reference" / Path(p["path"]).name
        assert file.stat().st_size == p["bytes"] and sha(file) == p["sha256"] == PHOTOS[file.name]
    return folder


def build():
    angles = np.arange(N) * 2 * np.pi / N
    points = np.array(
        [
            (r * np.cos(t), r * np.sin(t), z)
            for r, z in [(RO, -H / 2), (RO, H / 2), (RI, H / 2), (RI, -H / 2)]
            for t in angles
        ]
    )
    faces = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        for j in range(N):
            k = (j + 1) % N
            faces.append([a * N + j, a * N + k, b * N + k, b * N + j])
    assert mp.signed_volume(points, faces) > 0
    hulls = []
    # 64 independent convex wedges, eight corners each. Inscribed inner faces
    # intrude at most 0.024 mm; preserve a >38.7 mm traversable channel.
    for a in np.arange(64) * 2 * np.pi / 64:
        b = a + 2 * np.pi / 64
        hull = np.array([(r * np.cos(t), r * np.sin(t), z) for z in [-H / 2, H / 2] for r in [RI, RO] for t in [a, b]])
        convex = ConvexHull(hull)
        assert len(convex.vertices) <= 64
        probes = np.array([(x, y, z) for x in [-0.01, 0, 0.01] for y in [-0.01, 0, 0.01] for z in [-H, 0, H]])
        assert not np.any(np.all(probes @ convex.equations[:, :3].T + convex.equations[:, 3] <= 0, axis=1))
        hulls.append(hull)
    return points, faces, hulls


def material(stage, name, color, texture=None, roughness=0.4):
    mat = UsdShade.Material.Define(stage, "/root/_materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if texture:
        reader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    return mat


def mesh(stage, name, points, faces, mat, uv=None):
    item = mp.add_mesh(stage, "/root/visual/" + name, points, faces)
    UsdShade.MaterialBindingAPI.Apply(item.GetPrim()).Bind(mat)
    item.CreateDoubleSidedAttr(False)
    if uv is not None:
        UsdGeom.PrimvarsAPI(item).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            Vt.Vec2fArray.FromNumpy(np.float32(uv))
        )
    return item


def photos(folder, target):
    # Control points are in the displayed 1824x1368 coordinate system, mapped
    # back to the original 4032x3024 photograph. Only the observed far inner
    # wall is used. Cylindrical projection, not generic replacement lettering.
    original = np.asarray(Image.open(folder / "reference" / next(iter(PHOTOS))).convert("RGB"))
    sx, sy = original.shape[1] / 1824, original.shape[0] / 1368
    theta = np.linspace(np.radians(35), np.radians(145), 1024)
    v = np.linspace(0, 1, 512)
    xx = np.broadcast_to((787 + 221 * np.cos(theta))[None, :], (512, 1024))
    yy = 275 - 127 * np.sin(theta)[None, :] + v[:, None] * 220
    rect = cv2.remap(original, np.float32(xx * sx), np.float32(yy * sy), cv2.INTER_CUBIC)
    # Remove broad lighting only, retaining the photographed text and colours.
    image = rect.astype(float) / 255
    lum = image.mean(2)
    smooth = gaussian_filter(lum, 60)
    image *= np.median(smooth) / np.maximum(smooth[..., None], 0.06)
    Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).save(target / "inner_print.png")
    # Top winding texture is also sampled from the physical photograph.
    # A radial ellipse mapping rectifies the annulus; no procedural rings.
    grid = np.linspace(-RO, RO, 1024)
    gx, gy = np.meshgrid(grid, grid[::-1])
    radius = np.hypot(gx, gy)
    theta_top = np.arctan2(gy, gx)
    blend = np.clip((radius - RI) / (RO - RI), 0, 1)
    tx = (1 - blend) * (787 + 221 * np.cos(theta_top)) + blend * (778 + 390 * np.cos(theta_top))
    ty = (1 - blend) * (275 - 127 * np.sin(theta_top)) + blend * (298 - 238 * np.sin(theta_top))
    top_image = cv2.remap(original, np.float32(tx * sx), np.float32(ty * sy), cv2.INTER_CUBIC).astype(float) / 255
    smooth_top = gaussian_filter(top_image.mean(2), 70)
    top_image *= 0.68 / np.maximum(smooth_top[..., None], 0.1)
    Image.fromarray(np.uint8(np.clip(top_image, 0, 1) * 255)).save(target / "wound_top.png")
    annotation = Image.fromarray(original)
    draw = ImageDraw.Draw(annotation)
    top = np.c_[787 + 221 * np.cos(theta), 275 - 127 * np.sin(theta)]
    bottom = np.c_[top[:, 0], np.minimum(top[:, 1] + 220, 275 + 127 * np.sin(theta) - 10)]
    polygon = np.concatenate([top, bottom[::-1]]) * [sx, sy]
    draw.line([tuple(p) for p in np.vstack([polygon, polygon[:1]])], fill="red", width=8)
    draw.text((int(top[512, 0] * sx), int(top[512, 1] * sy)), "inner_print: observed pixels only", fill="red")
    annotation.save(target / "photo_label_regions.png")
    info = dict(
        source=next(iter(PHOTOS)),
        sha256=next(iter(PHOTOS.values())),
        control_system_px=[1824, 1368],
        inner_top_ellipse=[787, 275, 221, 127],
        axial_projection_px=220,
        angular_coverage_deg=[35, 145],
        texture_px=[1024, 512],
        output_px_per_mm=[1024 / (RI * np.radians(110) * 1000), 512 / (H * 1000)],
        native_axial_px_per_mm=220 * sy / (H * 1000),
        unknown="remaining circumferential sectors, occluded lower side portions, underside",
        top_mapping="radial interpolation of inner/outer photograph ellipses; full observed annulus",
        top_texture_px_per_mm=1024 / (2 * RO * 1000),
        label_axial_bounds_mm=[-H * 500, H * 500],
        photo_use="real physical reference; colour/printing evidence, not simulation validation",
        limitations=(
            "manual cylindrical fit; unknown correspondence between the three rotated views; "
            "unobserved printing is plain, not reconstructed"
        ),
    )
    (target / "texture_provenance.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def write(output):
    folder = verify()
    points, faces, hulls = build()
    output = Path(output) / "nano_glue/00000"
    output.mkdir(parents=True, exist_ok=True)
    tex = output / "textures"
    tex.mkdir(exist_ok=True)
    photos(folder, tex)
    temp = Path(tempfile.mkdtemp(prefix="nano-glue-"))
    (temp / "textures").mkdir()
    import shutil

    for name in ["inner_print.png", "wound_top.png"]:
        shutil.copy2(tex / name, temp / "textures" / name)
    stage = Usd.Stage.CreateNew(str(temp / "object.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = mp.add_xform(stage, "/root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    # Uniform annulus inertia is a stated approximation, not a measurement.
    iz = MASS * (RO**2 + RI**2) / 2
    ix = MASS * (3 * (RO**2 + RI**2) + H**2) / 12
    inertia = np.diag([ix, ix, iz])
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(ix, ix, iz))
    mp.add_xform(stage, "/root/visual")
    clear = mp.add_material(
        stage,
        "/root/_materials/WoundClearPlastic",
        "OmniSurface",
        {
            "diffuse_reflection_weight": 1.0,
            "diffuse_reflection_color": (0.87, 0.80, 0.60),
            "enable_specular_transmission": True,
            "specular_transmission_color": (0.97, 0.94, 0.83),
            "specular_transmission_weight": 0.42,
            "specular_reflection_weight": 0.7,
            "specular_reflection_roughness": 0.30,
            "specular_reflection_ior": 1.49,
            "thin_walled": False,
        },
        (0.87, 0.80, 0.60),
        0.65,
        0.30,
    )
    mesh(stage, "model", points, faces, clear, np.c_[points[:, 0] / (2 * RO) + 0.5, points[:, 1] / (2 * RO) + 0.5])
    top_material = material(stage, "PhotographedWinding", (1, 1, 1), "textures/wound_top.png", 0.32)
    top_subset = UsdGeom.Subset.Define(stage, "/root/visual/model/top_winding")
    top_subset.CreateElementTypeAttr("face")
    top_subset.CreateFamilyNameAttr("materialBind")
    top_subset.CreateIndicesAttr(list(range(N, 2 * N)))
    UsdShade.MaterialBindingAPI.Apply(top_subset.GetPrim()).Bind(top_material)
    paper = material(stage, "PaperBacking", (0.79, 0.77, 0.69))
    printmat = material(stage, "PhotographedPrint", (1, 1, 1), "textures/inner_print.png")
    # Paper is a surface shell; thickness not measured and not invented as
    # a physical parameter. Offsets 0.01–0.02 mm is a numerical anti-z-fighting gap.
    angle = np.arange(N + 1) * 2 * np.pi / N
    paper_points = np.array(
        [((RI - 0.00001) * np.cos(t), (RI - 0.00001) * np.sin(t), z) for z in [H / 2, -H / 2] for t in angle]
    )
    pf = [[j, j + 1, N + 2 + j, N + 1 + j] for j in range(N)]
    shell = mesh(stage, "paper_core_missing_photo", paper_points, pf, paper)
    shell.CreateDoubleSidedAttr(True)  # both sides plain backing, no mirrored print
    # Printed surface normal points inward. Its unseen reverse is supplied by
    # the plain shell, so transparency never shows mirrored front artwork.
    ts = np.linspace(np.radians(35), np.radians(145), 129)
    pp, uv = [], []
    for layer in [0, 1]:
        for j, t in enumerate(ts):
            depth = 0 if layer == 0 else min(1, (254 * np.sin(t) - 10) / 220)
            z = H / 2 - H * depth
            pp.append(((RI - 0.00002) * np.cos(t), (RI - 0.00002) * np.sin(t), z))
            uv.append((j / 128, 1 - depth))
    mesh(stage, "label_inner_observed", np.array(pp), [[j, j + 1, 130 + j, 129 + j] for j in range(128)], printmat, uv)
    cups.add_collision(stage, hulls)
    stage.GetRootLayer().Save()
    writer = (getattr(Sdf, "ZipFileWriter", None) or Usd.ZipFileWriter).CreateNew(str(output / "object.usdz"))
    writer.AddFile(str(temp / "object.usdc"), "object.usdc")
    for name in ["inner_print.png", "wound_top.png"]:
        writer.AddFile(str(temp / "textures" / name), "textures/" + name)
    writer.Save()
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, "robodojo/rigid/nano_glue/0"))
    meta = cups.metadata(uid, dict(pts=points, faces=faces, contact_r=RO), MASS, np.zeros(3))
    outer = points[: 2 * N]
    inner = np.vstack([paper_points, np.asarray(pp)])
    measurements = {
        "外径": round(float(2 * np.linalg.norm(outer[:, :2], axis=1).max() * 1000), 2),
        "内径": round(float(2 * np.linalg.norm(inner[:, :2], axis=1).min() * 1000), 2),
        "高度": round(float(np.ptp(points[:, 2]) * 1000), 2),
    }
    assert np.allclose(list(measurements.values()), [72.5, 38.73, 29.85], atol=1e-6)
    meta.update(
        asset_id=ID,
        measurements_mm=measurements,
        registration_revision=REVISION,
        generator="scripts/objects/make_nano_glue.py",
        author="Codex（用户委托；实物登记杨开元）",
        asset_origin="physical_object",
        dimension_policy="measured",
        mass_policy="measured",
        estimated_dimensions_mm=None,
        estimated_mass_g=None,
        source_photo_sha256=PHOTOS,
        inertia_kg_m2=inertia.tolist(),
        inertia_basis="homogeneous annulus approximation scaled to measured mass, not measured",
        origin_definition="complete visible bounding-box centre; through-hole axis Z",
        collision=dict(
            method="64 explicit convex annular wedges",
            max_vertices=8,
            minimum_clear_diameter_mm=2 * RI * np.cos(np.pi / 64) * 1000,
        ),
        missing_photos=["complete inner printed circumference including near wall", "underside"],
        friction_source="unmeasured; project defaults",
        appearance_parameters_source="visual fit to supplied photographs; roughness/IOR/transmission not calibrated",
        real_material_calibrated=False,
    )
    meta["active"]["place"]["up"].pop("contact_circle", None)
    meta["active"]["place"]["up"]["contact_annulus"] = dict(inner_radius=RI, outer_radius=RO, z=-H / 2)
    desc = dict(
        uuid=uid,
        caption=dict(
            name=["roll", "tape roll", "nano glue roll"],
            color=["clear", "warm translucent", "blue and white inner printing"],
            material=["plastic", "transparent polymer", "paper-lined plastic roll"],
            size=["72.50 mm outside diameter", "38.77 mm inside diameter", "29.85 mm high"],
            shape=["annular cylinder", "open through-hole", "straight circular sides"],
            function=["grasping", "placement", "adhesive tape storage"],
        ),
        description=[
            "Transparent nano glue roll with a circular through-hole.",
            "Short annular plastic roll with blue and white printing inside its paper core.",
            "Clear tape roll measuring 72.50 mm outside and 38.77 mm inside diameter.",
            "Hollow roll with an axial height of 29.85 mm and measured mass of 91.8 g.",
            "Rigid simulation approximation of a clear adhesive roll with photographed inner printing.",
        ],
    )
    for name, data in [
        ("metadata.json", meta),
        ("description.json", desc),
        ("runtime_config.json", {"physics": {"mass": MASS}}),
    ]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(path=str(output), measurements_mm=measurements, mass_kg=MASS, sha256=sha(output / "object.usdz")),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        write(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Assets/Object/Real2Sim/Rigid")
    except Exception:
        if mp.APP is None:
            raise
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if mp.APP is not None:
        mp.APP.close()
