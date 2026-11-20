"""Scene assets for the MuJoCo backend: object USDs through MuJoCo's USD importer (MuJoCo 4), wrapped as item templates
(world.py: a body `root` holding the asset) with the layout's scale, mass, friction and colour.

MuJoCo's importer reads geometry, colliders (PhysicsCollisionAPI) and UsdPreviewSurface looks; the base colour of the
glTF-style shaders our generated assets carry (base_color_texture inside the usdz) is added here. Meshes keep
MuJoCo's convex-hull collision, so a concave collider that is not already split into parts collides as its hull.
"""

from functools import lru_cache
import hashlib
import io
import os
from pathlib import Path
import re
import zipfile

import mujoco
import numpy as np

from env.backends.mujoco.robots import COLLISION_GROUP, VISUAL_GROUP
from env.backends.mujoco.world import ROOT

TEXTURE_CACHE = Path(os.environ.get("ROBODOJO_MUJOCO_CACHE", Path.home() / ".cache" / "robodojo-mujoco")) / "textures"
_BASE_COLOR_INPUTS = ("base_color_texture", "diffuse_texture", "diffuseColor", "diffuse_color_texture", "albedo")
_BASE_FACTOR_INPUTS = ("base_color_factor", "diffuse_color_constant", "diffuseColor", "diffuse_tint")


def _is_collider(geom):
    return bool(geom.contype or geom.conaffinity)


def _texture_png(resolved):
    """A PNG MuJoCo can load for a USD texture path (`a.usdz[tex/b.jpg]` is extracted from the package)."""
    from PIL import Image

    match = re.fullmatch(r"(.+\.usdz)\[(.+)\]", resolved)
    key = hashlib.sha1(resolved.encode()).hexdigest()[:16]
    out = TEXTURE_CACHE / f"{key}.png"
    if out.exists():
        return str(out)
    if match:
        with zipfile.ZipFile(match.group(1)) as package:
            raw = package.read(match.group(2))
    else:
        raw = Path(resolved).read_bytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp.png")
    Image.open(io.BytesIO(raw)).convert("RGBA").save(tmp)
    os.replace(tmp, out)
    return str(out)


def _base_color(material):
    """(texture path or None, rgba) of a UsdShade material's surface shader."""
    from pxr import Sdf, UsdShade

    texture, rgba = None, [1.0, 1.0, 1.0, 1.0]
    for output in material.GetOutputs():
        for source in output.GetConnectedSources()[0]:
            shader = UsdShade.Shader(source.source.GetPrim())
            for name in _BASE_COLOR_INPUTS:
                inp = shader.GetInput(name)
                if inp and inp.HasConnectedSource():
                    tex = UsdShade.Shader(inp.GetConnectedSources()[0][0].source.GetPrim())
                    for file_input in ("texture", "file"):
                        value = tex.GetInput(file_input).Get() if tex.GetInput(file_input) else None
                        if isinstance(value, Sdf.AssetPath) and value.resolvedPath:
                            texture = value.resolvedPath
                            break
                if texture:
                    break
            for name in _BASE_FACTOR_INPUTS:
                inp = shader.GetInput(name)
                value = inp.Get() if inp else None
                if value is not None and not inp.HasConnectedSource() and hasattr(value, "__len__") and len(value) >= 3:
                    rgba[:3] = [float(v) for v in value[:3]]
                    break
    return texture, rgba


def _add_glTF_looks(spec, usd_path):
    """Give visual meshes without a MuJoCo material the base colour of their bound USD material."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(usd_path)
    looks = {}
    for geom in spec.geoms:
        if geom.material or _is_collider(geom) or not geom.name:
            continue
        prim = stage.GetPrimAtPath(geom.name)
        if not prim or not prim.IsValid():
            continue
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not material:
            color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get() if prim.IsA(UsdGeom.Gprim) else None
            if color:
                geom.rgba = [*map(float, color[0]), 1.0]
            continue
        key = str(material.GetPath())
        if key not in looks:
            texture, rgba = _base_color(material)
            name = f"look{len(looks)}"
            mat = spec.add_material(name=name, rgba=rgba)
            if texture:
                tex = spec.add_texture(
                    name=f"{name}_rgb", type=mujoco.mjtTexture.mjTEXTURE_2D, file=_texture_png(texture)
                )
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            looks[key] = name
        geom.material = looks[key]


@lru_cache(maxsize=256)
def _imported(usd_path):
    spec = mujoco.MjSpec.from_file(usd_path)
    spec.compiler.discardvisual = False
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # thin visual meshes have no volume
    _add_glTF_looks(spec, usd_path)
    return spec


def _wrap(asset, scale, dynamic, keep_joints=False):
    """Item template: body `root` (free if dynamic) holding the asset's bodies and geoms; its free joints are
    removed, and its other joints too unless `keep_joints`."""
    asset = asset.copy()
    for joint in list(asset.joints):
        if not keep_joints or joint.type == mujoco.mjtJoint.mjJNT_FREE:
            asset.delete(joint)
    scale = np.asarray(scale, dtype=float)
    if not np.allclose(scale, 1.0):
        for mesh in asset.meshes:
            mesh.scale = list(np.asarray(mesh.scale) * scale)
        for geom in asset.geoms:
            geom.pos = list(np.asarray(geom.pos) * scale)
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                geom.size = list(np.asarray(geom.size) * scale)
        for body in asset.bodies[1:]:
            body.pos = list(np.asarray(body.pos) * scale)
    template = mujoco.MjSpec()
    template.compiler.discardvisual = False
    root = template.worldbody.add_body(name=ROOT)
    if dynamic:
        root.add_freejoint()
    template.attach(asset, frame=root.add_frame(), prefix="asset")
    return template


def _split_groups(spec, collide):
    """Visual geoms in group 1, colliders in group 3 (not drawn); `collide` False makes colliders inert."""
    for geom in spec.geoms:
        if _is_collider(geom):
            geom.group = COLLISION_GROUP
            if not collide:
                geom.contype = geom.conaffinity = 0
        else:
            geom.group = VISUAL_GROUP


def _recolor(spec, color, parts=None):
    for geom in spec.geoms:
        if _is_collider(geom):
            continue
        if parts and geom.name.rsplit("/", 1)[-1] not in parts:
            continue
        geom.material = ""
        geom.rgba = [*color[:3], 1.0]


def _set_mass(template, mass):
    """Scale the colliders' density so the body weighs `mass`; visual geoms weigh nothing."""
    colliders = [g for g in template.geoms if _is_collider(g)]
    for geom in template.geoms:
        geom.density = 1000.0 if _is_collider(geom) else 0.0
    if not colliders:
        root = template.body(ROOT)
        root.mass, root.inertia, root.explicitinertial = mass, [mass * 1e-4] * 3, True
        return
    probe = template.copy().compile()
    total = float(probe.body_subtreemass[probe.body(ROOT).id])
    for geom in colliders:
        geom.density = 1000.0 * mass / max(total, 1e-9)


def rigid_template(usd_path, scale=(1, 1, 1), mass=0.5, friction=(0.6, 1.5), color=None, color_parts=None):
    """A free body: the asset's colliders weigh `mass`; sliding friction is the static coefficient (MuJoCo has one)."""
    template = _wrap(_imported(usd_path), scale, dynamic=True)
    _split_groups(template, collide=True)
    for geom in template.geoms:
        if _is_collider(geom):
            geom.friction = [float(friction[0]), 0.005, 0.0001]
            geom.condim = 4  # torsional friction: a pinched object does not spin about the pad normal
    if color is not None:
        _recolor(template, color, color_parts)
    _set_mass(template, mass)
    return template


def articulation_template(usd_path, scale=(1, 1, 1), fixed_base=False, color=None):
    """An articulated asset with its joints (free root unless `fixed_base`), at the importer's masses."""
    template = _wrap(_imported(usd_path), scale, dynamic=not fixed_base, keep_joints=True)
    _split_groups(template, collide=True)
    if color is not None:
        _recolor(template, color)
    return template


def static_template(usd_path, scale=(1, 1, 1), collide=True, color=None, friction=None):
    """Geometry, rooms: the asset fixed in place (movable only by the scene manager)."""
    template = _wrap(_imported(usd_path), scale, dynamic=False)
    _split_groups(template, collide)
    if friction is not None:
        for geom in template.geoms:
            if _is_collider(geom):
                geom.friction = [float(friction), 0.005, 0.0001]
    if color is not None:
        _recolor(template, color)
    return template


def box_template(size, rgba=(0.6, 0.6, 0.6, 1.0), texture=None, friction=0.8):
    """Tables and grounds: a static box of full extents `size`, optionally with a tiled base-colour image."""
    template = mujoco.MjSpec()
    root = template.worldbody.add_body(name=ROOT)
    kwargs = {}
    if texture:
        tex = template.add_texture(name="rgb", type=mujoco.mjtTexture.mjTEXTURE_CUBE, file=_texture_png(texture))
        mat = template.add_material(name="look", rgba=list(rgba), texuniform=True, texrepeat=[2, 2])
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        kwargs["material"] = "look"
    root.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[s / 2 for s in size],
        rgba=list(rgba),
        friction=[friction, 0.005, 0.0001],
        group=VISUAL_GROUP,
        **kwargs,
    )
    return template


def material_base_color(mdl_path_or_dir):
    """The base-colour image next to an MDL material (Assets/Material/<name>/*_BaseColor.png), if there is one."""
    if not mdl_path_or_dir:
        return None
    folder = Path(mdl_path_or_dir)
    folder = folder if folder.is_dir() else folder.parent
    if not folder.is_dir():
        return None
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for pattern in ("basecolor", "base_color", "albedo", "diffuse", "color"):
        for image in images:
            if pattern in image.stem.lower() and not re.search(r"(norm|_n$|rough|orm|id_)", image.stem.lower()):
                return str(image)
    return None
