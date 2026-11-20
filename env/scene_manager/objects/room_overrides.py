"""Per-layout overrides of a reconstructed room (real2sim/scenes/): rebound surface materials and the desk geometry.

A layout's Room entry may carry, as written by scripts/scenes/appearance.py from the scene's usd/parts.json:
    "appearance": [{"prims": ["Background/Floor", ...], "material": {...}, "uv_scale": 0.6}, ...]
        material {"kind": "preview", "color": [r, g, b] | null, "texture": "<repo-relative path>" | null,
                  "roughness": float, "metallic": float}
              or {"kind": "mdl", "mdl": "$Robodojo_ASSETS/Material/<dir>/<name>.mdl", "tile_m": float}
        uv_scale multiplies the mesh's texture coordinates (preview textures keep their real repeat size); MDL
        materials are projected in world space, one texture repeat per tile_m metres (default 1).
    "table_lift": {"dz": 0.12, "move": [paths], "stretch": [[path, base_z, top_z], ...]}
        the desk top and everything above its telescoping joint rise by dz; legs stretch from their base.
    "table_size": [width_x, depth_y]  (or "table_xy_scale": [sx, sy], plus optional "table_xy_offset": [dx, dy])
        horizontal tabletop expansion/shift in the room's XY plane (preserving tabletop top z and (0, 0, 1) normal);
        may also be placed under Room["room_overrides"], Room["table_lift"], or env_cfg's "room_overrides" section.
Paths are below the Room prim; z values are in the room's frame. Applied right after the Room is referenced, before
PhysX parses its colliders.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from env.global_configs import ROOT_DIR


def _caller_room_overrides() -> Mapping | None:
    """Fallback `room_overrides` mapping from an active TaskEnv/EvalEnv `eval_cfg` or `scene_config` on the stack."""
    for depth in range(1, 12):
        try:
            frame = sys._getframe(depth)
        except ValueError:
            break
        loc = frame.f_locals
        self_obj = loc.get("self")
        for candidate in (
            loc.get("eval_cfg"),
            getattr(self_obj, "eval_cfg", None),
            loc.get("scene_config"),
            getattr(self_obj, "scene_config", None),
        ):
            if not isinstance(candidate, Mapping):
                continue
            ov = candidate.get("room_overrides")
            if isinstance(ov, Mapping):
                return ov
            cfg_sub = candidate.get("config")
            if isinstance(cfg_sub, Mapping) and isinstance(cfg_sub.get("room_overrides"), Mapping):
                return cfg_sub["room_overrides"]
            room_sub = candidate.get("Room")
            if isinstance(room_sub, Mapping) and isinstance(room_sub.get("room_overrides"), Mapping):
                return room_sub["room_overrides"]
    return None


def _resolve_table_resize(
    config: Mapping,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, tuple[float, float]]:
    sources: list[Mapping] = [config]
    for sub_key in ("room_overrides", "table_lift"):
        sub = config.get(sub_key)
        if isinstance(sub, Mapping):
            sources.append(sub)
    caller_ov = _caller_room_overrides()
    if isinstance(caller_ov, Mapping):
        sources.append(caller_ov)

    table_size = None
    table_xy_scale = None
    table_xy_offset = None
    for src in sources:
        if table_size is None and src.get("table_size") is not None:
            raw = src["table_size"]
            table_size = (float(raw[0]), float(raw[1]))
        if table_xy_scale is None:
            raw_scale = src.get("table_xy_scale") if src.get("table_xy_scale") is not None else src.get("xy_scale")
            if raw_scale is not None:
                table_xy_scale = (float(raw_scale[0]), float(raw_scale[1]))
        if table_xy_offset is None:
            raw_off = src.get("table_xy_offset") if src.get("table_xy_offset") is not None else src.get("xy_offset")
            if raw_off is not None:
                table_xy_offset = (float(raw_off[0]), float(raw_off[1]))

    return table_size, table_xy_scale, table_xy_offset or (0.0, 0.0)


def apply_room_overrides(stage, room_path: str, config) -> None:
    room = stage.GetPrimAtPath(room_path)
    for i, group in enumerate(config.get("appearance") or []):
        material = _material(stage, f"{room_path}/OverrideLooks/m{i}", group["material"], group.get("uv_scale"))
        for rel in group["prims"]:
            prim = stage.GetPrimAtPath(f"{room_path}/{rel}")
            if prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

    table_size, table_xy_scale, table_xy_offset = _resolve_table_resize(config)
    if table_size is not None or table_xy_scale is not None or any(abs(v) > 1e-6 for v in table_xy_offset):
        _apply_table_xy_resize(stage, room, room_path, table_size, table_xy_scale, table_xy_offset)

    lift = config.get("table_lift")
    if lift and abs(lift.get("dz", 0.0)) > 1e-6:
        dz = float(lift["dz"])
        for rel in lift.get("move", []):
            _outer_op(stage, room, f"{room_path}/{rel}", Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, dz)))
        for rel, base, top in lift.get("stretch", []):
            s = (top - base + dz) / (top - base)
            m = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -base))
            m *= Gf.Matrix4d().SetScale(Gf.Vec3d(1, 1, s))
            m *= Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, base))
            _outer_op(stage, room, f"{room_path}/{rel}", m)


def _apply_table_xy_resize(
    stage,
    room,
    room_path: str,
    table_size: tuple[float, float] | None,
    table_xy_scale: tuple[float, float] | None,
    table_xy_offset: tuple[float, float],
) -> None:
    """Scale the tabletop horizontally in the Room XY plane (sz=1, tz=0) and shift side legs cleanly."""
    ref_prim = None
    for rel in ("Table/Colliders/top", "Table/Top/slab", "Table/Top/surface", "Table/Top"):
        cand = stage.GetPrimAtPath(f"{room_path}/{rel}")
        if cand.IsValid():
            ref_prim = cand
            break
    if ref_prim is None:
        return

    # Reset any prior tableSize ops to identity before measuring base bounds so repeated calls stay exact.
    table_root = stage.GetPrimAtPath(f"{room_path}/Table")
    if table_root.IsValid():
        identity = Gf.Matrix4d(1.0)
        for p in Usd.PrimRange(table_root):
            if p.IsA(UsdGeom.Xformable):
                for op in UsdGeom.Xformable(p).GetOrderedXformOps():
                    if op.GetOpName() == "xformOp:transform:tableSize":
                        op.Set(identity)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    rng = bbox_cache.ComputeRelativeBound(ref_prim, room).ComputeAlignedRange()
    if rng.IsEmpty():
        return
    lo, hi = rng.GetMin(), rng.GetMax()
    cur_w, cur_d = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    cx, cy = 0.5 * float(lo[0] + hi[0]), 0.5 * float(lo[1] + hi[1])

    if table_xy_scale is not None:
        sx, sy = table_xy_scale
    elif table_size is not None:
        if cur_w <= 1e-6 or cur_d <= 1e-6:
            return
        sx, sy = table_size[0] / cur_w, table_size[1] / cur_d
    else:
        sx, sy = 1.0, 1.0
    dx, dy = table_xy_offset
    if abs(sx - 1.0) <= 1e-6 and abs(sy - 1.0) <= 1e-6 and abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return

    m_top = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-cx, -cy, 0.0))
    m_top *= Gf.Matrix4d().SetScale(Gf.Vec3d(sx, sy, 1.0))
    m_top *= Gf.Matrix4d().SetTranslate(Gf.Vec3d(cx + dx, cy + dy, 0.0))

    top_targets: list[str] = []
    if stage.GetPrimAtPath(f"{room_path}/Table/Colliders/top").IsValid():
        top_targets.append("Table/Colliders/top")
    leaf_visuals = [
        r for r in ("Table/Top/slab", "Table/Top/surface") if stage.GetPrimAtPath(f"{room_path}/{r}").IsValid()
    ]
    if leaf_visuals:
        top_targets.extend(leaf_visuals)
    elif stage.GetPrimAtPath(f"{room_path}/Table/Top").IsValid():
        top_targets.append("Table/Top")
    for rel in top_targets:
        _outer_op(stage, room, f"{room_path}/{rel}", m_top, op_suffix="tableSize")

    m_beam = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-cx, -cy, 0.0))
    m_beam *= Gf.Matrix4d().SetScale(Gf.Vec3d(sx, 1.0, 1.0))
    m_beam *= Gf.Matrix4d().SetTranslate(Gf.Vec3d(cx + dx, cy + dy, 0.0))
    for rel in (
        "Table/Frame/beam",
        "Table/Colliders/Frame_beam",
        "Table/Colliders/beam",
        "Table/Panel/board",
        "Table/Colliders/panel",
    ):
        _outer_op(stage, room, f"{room_path}/{rel}", m_beam, op_suffix="tableSize")

    colliders_root = stage.GetPrimAtPath(f"{room_path}/Table/Colliders")
    for side, col_prefixes in (
        ("pos", ("Legs_pos", "bracket_pos")),
        ("neg", ("Legs_neg", "bracket_neg")),
    ):
        leg_prims = []
        leg_group = stage.GetPrimAtPath(f"{room_path}/Table/Legs/{side}")
        if leg_group.IsValid():
            leg_prims.extend(p for p in Usd.PrimRange(leg_group) if p.IsA(UsdGeom.Gprim))
        if colliders_root.IsValid():
            for child in colliders_root.GetChildren():
                if child.IsA(UsdGeom.Gprim) and child.GetName().startswith(col_prefixes):
                    leg_prims.append(child)
        crossbar = stage.GetPrimAtPath(f"{room_path}/Table/Frame/crossbar_{side[0]}")
        if crossbar.IsValid() and crossbar.IsA(UsdGeom.Gprim):
            leg_prims.append(crossbar)
        if not leg_prims:
            continue
        xs = []
        for lp in leg_prims:
            lrng = bbox_cache.ComputeRelativeBound(lp, room).ComputeAlignedRange()
            if not lrng.IsEmpty():
                xs.extend((float(lrng.GetMin()[0]), float(lrng.GetMax()[0])))
        if not xs:
            continue
        leg_cx = 0.5 * (min(xs) + max(xs))
        tx = (leg_cx - cx) * (sx - 1.0) + dx
        m_leg = Gf.Matrix4d().SetTranslate(Gf.Vec3d(tx, dy, 0.0))
        for lp in leg_prims:
            _outer_op(stage, room, str(lp.GetPath()), m_leg, op_suffix="tableSize")


def _outer_op(stage, room, path, room_matrix, op_suffix: str = "tableLift"):
    """Apply `room_matrix` (row-vector, in the room's frame) to a prim as its outermost transform op."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    time = Usd.TimeCode.Default()
    room_world = UsdGeom.Xformable(room).ComputeLocalToWorldTransform(time)
    parent_world = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(time)
    parent_in_room = parent_world * room_world.GetInverse()
    outer = parent_in_room * room_matrix * parent_in_room.GetInverse()
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    op_name = f"xformOp:transform:{op_suffix}"
    existing = next((o for o in ops if o.GetOpName() == op_name), None)
    if existing is not None:
        existing.Set(outer)
        xform.SetXformOpOrder([existing, *[o for o in ops if o.GetOpName() != op_name]])
        return
    op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, op_suffix)
    op.Set(outer)
    xform.SetXformOpOrder([op, *ops])


def _material(stage, path, spec, uv_scale):
    if spec["kind"] == "mdl":
        from env.scene_manager.objects.mdl_material import create_mdl_material
        from utils.path import resolve_path

        mdl = resolve_path(spec["mdl"])
        create_mdl_material(mdl, os.path.splitext(os.path.basename(mdl))[0], path)
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        shader = material.ComputeSurfaceSource("mdl")[0]
        if shader:  # OmniPBR world-space projection: independent of the mesh's UVs
            shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(*(2 * [1.0 / float(spec.get("tile_m") or 1.0)]))
            )
        return material
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(spec.get("roughness") or 0.8))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(spec.get("metallic") or 0.0))
    if spec.get("texture"):
        reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        if uv_scale and abs(uv_scale - 1.0) > 1e-6:
            scale = UsdShade.Shader.Define(stage, f"{path}/UvScale")
            scale.CreateIdAttr("UsdTransform2d")
            scale.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
            scale.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(uv_scale, uv_scale))
            st = scale.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, f"{path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.join(ROOT_DIR, spec["texture"]))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        for wrap in ("wrapS", "wrapT"):
            texture.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set("repeat")
        rgb = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    else:
        color = spec.get("color") or [0.7, 0.7, 0.7]
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material
