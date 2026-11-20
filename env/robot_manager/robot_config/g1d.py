from collections.abc import Mapping
import os
import re
from typing import Any
from weakref import WeakKeyDictionary

from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import from_files
import numpy as np
from pxr import Pcp, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

try:
    from pxr import PhysxSchema as _PhysxSchema
except ImportError:  # standalone CPU pxr without omni.usd.schema.physx loaded
    _PhysxSchema = None

from env.environment.isaaclab_compat import implicit_actuator_cfg
from env.global_configs import ROBOTS_PATH
from env.robot_manager.robot_config import g1d_spec
from env.robot_manager.robot_config._g1d_trim import (
    BLACK_FACES,
    SCREWS,
    bumper_mesh,
    lidar_mesh,
    panel_meshes,
    screw_mesh,
)

FINGER_PADS = g1d_spec.FINGER_PADS

# Keyed outside UrdfFileCfg so AssetConverterBase._config_to_hash stays byte-identical on read-only shared assets.
_SPAWN_FINGER_CONTACT: WeakKeyDictionary[object, dict[str, Any]] = WeakKeyDictionary()
_ROBOT_IDX_FINGER_CONTACT: dict[int, dict[str, Any]] = {}

MATERIALS = {name: sim_utils.PreviewSurfaceCfg(**finish) for name, finish in g1d_spec.FINISHES.items()}
GLOSS_BLACK = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.008, 0.008, 0.01), roughness=0.12)  # lidar, front panel
SCREW_STEEL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.56, 0.58), metallic=1.0, roughness=0.2)
LINK_MATERIALS = g1d_spec.LINK_FINISHES


def _paint_faces(link, material, select):
    """Bind `material` to the faces of the link's meshes whose link-frame centroids pass `select`; the other faces
    keep the material bound to the link's visuals."""
    visuals = link.GetChild("visuals")
    visuals.SetInstanceable(False)
    base = UsdShade.MaterialBindingAPI(visuals).ComputeBoundMaterial()[0]
    UsdShade.MaterialBindingAPI(visuals).UnbindDirectBinding()  # a stronger-than-descendants binding hides subsets
    xform_cache = UsdGeom.XformCache()
    for p in Usd.PrimRange(visuals):
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        to_link = np.array(xform_cache.ComputeRelativeTransform(p, link)[0])
        pts = np.array(mesh.GetPointsAttr().Get()) @ to_link[:3, :3] + to_link[3, :3]  # Gf acts on row vectors
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        centroids = np.add.reduceat(
            pts[np.array(mesh.GetFaceVertexIndicesAttr().Get())], np.r_[0, np.cumsum(counts)[:-1]]
        )
        faces = np.nonzero(select(centroids / counts[:, None]))[0].tolist()
        binding = UsdShade.MaterialBindingAPI.Apply(p)
        if base:
            binding.Bind(base)
        subset = binding.CreateMaterialBindSubset("black", Vt.IntArray(faces), UsdGeom.Tokens.face)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)


def _link_triangles(link):
    """Triangles (n, 3, 3) of the link's visual meshes, in the link frame."""
    link.GetChild("visuals").SetInstanceable(False)
    xform_cache = UsdGeom.XformCache()
    tris = []
    for p in Usd.PrimRange(link.GetChild("visuals")):
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        to_link = np.array(xform_cache.ComputeRelativeTransform(p, link)[0])
        pts = np.array(mesh.GetPointsAttr().Get()) @ to_link[:3, :3] + to_link[3, :3]
        counts, indices = mesh.GetFaceVertexCountsAttr().Get(), np.array(mesh.GetFaceVertexIndicesAttr().Get())
        start = 0
        for k in counts:
            face = indices[start : start + k]
            tris += [pts[[face[0], face[i], face[i + 1]]] for i in range(1, k - 1)]
            start += k
    return np.array(tris)


def _add_mesh(stage, path, mesh_data, material):
    points, counts, indices, normals = mesh_data
    mesh = UsdGeom.Mesh.Define(stage, path)  # visual only: no collision API
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices.tolist())
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(normals.tolist())
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def add_black_trim(prim):
    """Parts of the G1-D articulation `prim` that its meshes do not separate (_g1d_trim.py): the arm faces in
    BLACK_FACES, the wrist SCREWS, and the chassis front panel with its windows, bumper and lidar. Call after binding
    the link materials."""
    stage = prim.GetStage()
    path = prim.GetPath().AppendPath("Looks/trim_black")
    MATERIALS["black"].func(str(path), MATERIALS["black"])
    black = UsdShade.Material(stage.GetPrimAtPath(path))
    path = prim.GetPath().AppendPath("Looks/trim_rubber")
    MATERIALS["rubber"].func(str(path), MATERIALS["rubber"])
    rubber = UsdShade.Material(stage.GetPrimAtPath(path))
    path = prim.GetPath().AppendPath("Looks/screw_steel")
    SCREW_STEEL.func(str(path), SCREW_STEEL)
    steel = UsdShade.Material(stage.GetPrimAtPath(path))
    for side in ("left", "right"):
        for suffix, select in BLACK_FACES.items():
            link = prim.GetChild(f"{side}_{suffix}")
            if link and link.GetChild("visuals"):
                _paint_faces(link, black, select)
        for suffix, sides, centres, radius in SCREWS:
            link = prim.GetChild(f"{side}_{suffix}")
            if side in sides and link and link.GetChild("visuals"):
                for part, mesh_data in screw_mesh(_link_triangles(link), centres, radius).items():
                    material = steel if part == "head" else rubber
                    _add_mesh(stage, link.GetPath().AppendChild(f"screw_{part}"), mesh_data, material)
    path = prim.GetPath().AppendPath("Looks/gloss_black")
    GLOSS_BLACK.func(str(path), GLOSS_BLACK)
    gloss = UsdShade.Material(stage.GetPrimAtPath(path))
    link = prim.GetChild("AGV_link")
    panel, windows = panel_meshes(_link_triangles(link))
    for name, mesh_data, material in (
        ("front_panel", panel, gloss),
        ("panel_windows", windows, rubber),
        ("bumper", bumper_mesh(), rubber),
        ("lidar", lidar_mesh(), gloss),
    ):
        _add_mesh(stage, link.GetPath().AppendChild(name), mesh_data, material)


def _repair_imported_usd(usd):
    """Two defects of IsaacLab 3's URDF importer output, fixed in the generated layers:

    - it references a visual for every link, also links without one; Newton refuses the stage for those composition
      errors, so the dangling references are removed;
    - it puts a mesh collider's PhysicsCollisionAPI / PhysicsMeshCollisionAPI on the Xform above the mesh; Newton only
      collides geometry prims, so the fingers passed through what they grasped. The APIs move to the meshes (in the
      root layer, the collision prims made non-instanced)."""
    stage = Usd.Stage.Open(usd)
    _drop_dangling_references(stage)
    stage = Usd.Stage.Open(usd)
    proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    holders = [p for p in stage.Traverse(proxies) if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Gprim)]
    if not holders:
        return
    for prim in stage.Traverse(proxies):
        if prim.GetName() == "collisions" and prim.IsInstance():
            stage.OverridePrim(prim.GetPath()).SetInstanceable(False)
    for path in [p.GetPath() for p in holders]:
        holder = stage.GetPrimAtPath(path)
        approximation = UsdPhysics.MeshCollisionAPI(holder).GetApproximationAttr().Get() or UsdPhysics.Tokens.convexHull
        for mesh in Usd.PrimRange(holder):
            if mesh.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(mesh)
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
        holder.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        holder.RemoveAPI(UsdPhysics.CollisionAPI)
    stage.GetRootLayer().Save()


def _drop_dangling_references(stage):
    for error in stage.GetCompositionErrors():
        if error.errorType != Pcp.ErrorType_UnresolvedPrimPath:
            continue
        for spec in stage.GetPrimAtPath(error.rootSite.path).GetPrimStack():
            items = list(spec.referenceList.prependedItems)
            layers = [
                Sdf.Layer.FindOrOpen(spec.layer.ComputeAbsolutePath(r.assetPath)) if r.assetPath else spec.layer
                for r in items
            ]
            kept = [r for r, layer in zip(items, layers) if layer and layer.GetPrimAtPath(r.primPath)]
            if len(kept) != len(items):
                spec.referenceList.prependedItems = kept
                spec.layer.Save()


def _prepend_api_schema(prim, schema_token: str) -> None:
    stage = prim.GetStage()
    layer = stage.GetEditTarget().GetLayer()
    spec = Sdf.CreatePrimInLayer(layer, prim.GetPath())
    if spec.specifier == Sdf.SpecifierOver and not prim.GetTypeName():
        spec.specifier = Sdf.SpecifierDef
    if schema_token not in list(prim.GetAppliedSchemas()):
        spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[schema_token]))


def apply_finger_contact(prim, finger_contact: Mapping[str, Any] | None) -> None:
    """Bind finger physics material, torsional friction patches, and link depenetration caps on Dex1-1 colliders."""
    fc = g1d_spec.resolve_finger_contact(finger_contact)
    if fc is None or not prim or not prim.IsValid():
        return

    stage = prim.GetStage()
    prim_path = str(prim.GetPath())
    patterns = fc["links"]
    target_links = [
        child for child in prim.GetChildren() if any(re.fullmatch(pat, child.GetName()) for pat in patterns)
    ]
    if not target_links:
        return

    mat = None
    if fc["has_material"]:
        mat_path = f"{prim_path}/Looks/finger_physics_material"
        mat = UsdShade.Material.Define(stage, mat_path)
        mprim = mat.GetPrim()
        umat = UsdPhysics.MaterialAPI.Apply(mprim)
        umat.CreateStaticFrictionAttr().Set(fc["static_friction"])
        umat.CreateDynamicFrictionAttr().Set(fc["dynamic_friction"])
        umat.CreateRestitutionAttr().Set(fc["restitution"])
        f_comb = fc.get("friction_combine")
        r_comb = fc.get("restitution_combine")
        c_stiff = fc.get("compliant_stiffness")
        c_damp = fc.get("compliant_damping")
        if any(v is not None for v in (f_comb, r_comb, c_stiff, c_damp)):
            if _PhysxSchema is not None:
                pmat = _PhysxSchema.PhysxMaterialAPI.Apply(mprim)
                if f_comb is not None:
                    pmat.CreateFrictionCombineModeAttr().Set(str(f_comb))
                if r_comb is not None:
                    pmat.CreateRestitutionCombineModeAttr().Set(str(r_comb))
                if c_stiff is not None:
                    pmat.CreateCompliantContactStiffnessAttr().Set(float(c_stiff))
                if c_damp is not None:
                    pmat.CreateCompliantContactDampingAttr().Set(float(c_damp))
            else:
                _prepend_api_schema(mprim, "PhysxMaterialAPI")
                if f_comb is not None:
                    mprim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, False).Set(
                        str(f_comb)
                    )
                if r_comb is not None:
                    mprim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, False).Set(
                        str(r_comb)
                    )
                if c_stiff is not None:
                    mprim.CreateAttribute(
                        "physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float, False
                    ).Set(float(c_stiff))
                if c_damp is not None:
                    mprim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float, False).Set(
                        float(c_damp)
                    )

    max_depen = fc.get("max_depenetration_velocity")
    lin_damp = fc.get("linear_damping")
    ang_damp = fc.get("angular_damping")
    torsional = fc.get("torsional_patch_radius")
    min_torsional = fc.get("min_torsional_patch_radius")
    c_offset = fc.get("contact_offset")
    r_offset = fc.get("rest_offset")

    for link in target_links:
        if any(v is not None for v in (max_depen, lin_damp, ang_damp)):
            if _PhysxSchema is not None:
                rb_api = _PhysxSchema.PhysxRigidBodyAPI.Apply(link)
                if max_depen is not None:
                    rb_api.CreateMaxDepenetrationVelocityAttr().Set(float(max_depen))
                if lin_damp is not None:
                    rb_api.CreateLinearDampingAttr().Set(float(lin_damp))
                if ang_damp is not None:
                    rb_api.CreateAngularDampingAttr().Set(float(ang_damp))
            else:
                _prepend_api_schema(link, "PhysxRigidBodyAPI")
                if max_depen is not None:
                    link.CreateAttribute(
                        "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, False
                    ).Set(float(max_depen))
                if lin_damp is not None:
                    link.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, False).Set(
                        float(lin_damp)
                    )
                if ang_damp is not None:
                    link.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, False).Set(
                        float(ang_damp)
                    )

        collisions = link.GetChild("collisions")
        if not collisions or not collisions.IsValid():
            continue
        if collisions.IsInstance() or collisions.IsInstanceable():
            collisions.SetInstanceable(False)  # material bindings and collision attrs are skipped on instanced prims
        proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for p in Usd.PrimRange(collisions, proxies):
            if p.IsInstance():
                p.SetInstanceable(False)

        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(collisions).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")

        if any(v is not None for v in (torsional, min_torsional, c_offset, r_offset)):
            col_prims = [p for p in Usd.PrimRange(collisions) if p.HasAPI(UsdPhysics.CollisionAPI)]
            if not col_prims:
                col_prims = [p for p in Usd.PrimRange(collisions) if p.IsA(UsdGeom.Gprim)] or [collisions]
            for cprim in col_prims:
                if _PhysxSchema is not None:
                    col_api = _PhysxSchema.PhysxCollisionAPI.Apply(cprim)
                    if torsional is not None:
                        col_api.CreateTorsionalPatchRadiusAttr().Set(float(torsional))
                    if min_torsional is not None:
                        col_api.CreateMinTorsionalPatchRadiusAttr().Set(float(min_torsional))
                    if c_offset is not None:
                        col_api.CreateContactOffsetAttr().Set(float(c_offset))
                    if r_offset is not None:
                        col_api.CreateRestOffsetAttr().Set(float(r_offset))
                else:
                    _prepend_api_schema(cprim, "PhysxCollisionAPI")
                    if torsional is not None:
                        cprim.CreateAttribute(
                            "physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float, False
                        ).Set(float(torsional))
                    if min_torsional is not None:
                        cprim.CreateAttribute(
                            "physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float, False
                        ).Set(float(min_torsional))
                    if c_offset is not None:
                        cprim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, False).Set(
                            float(c_offset)
                        )
                    if r_offset is not None:
                        cprim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, False).Set(
                            float(r_offset)
                        )


def _lookup_spawn_finger_contact(prim_path: str, cfg: object) -> dict[str, Any] | None:
    try:
        fc = _SPAWN_FINGER_CONTACT.get(cfg)
        if fc is not None:
            return fc
    except TypeError:
        pass
    m = re.search(r"/robot(\d+)$", str(prim_path))
    idx = int(m.group(1)) if m else 0
    return _ROBOT_IDX_FINGER_CONTACT.get(idx)


@sim_utils.clone
def _spawn_g1d(prim_path, cfg, *args, **kwargs):
    if hasattr(from_files, "spawn_from_usd_file"):  # IsaacLab 3: its URDF importer nests links under Geometry/
        usd = os.path.join(cfg.usd_dir, os.path.splitext(os.path.basename(cfg.asset_path))[0] + ".usd")
        _repair_imported_usd(usd)
        prim = from_files.spawn_from_usd_file(prim_path, usd, cfg, *args, **kwargs)
    else:
        prim = from_files.spawn_from_urdf.__wrapped__(prim_path, cfg, *args, **kwargs)
    materials = {}
    for name, material_cfg in MATERIALS.items():
        material_cfg.func(f"{prim_path}/Looks/{name}", material_cfg)  # returns the shader, not the material
        materials[name] = UsdShade.Material(prim.GetStage().GetPrimAtPath(f"{prim_path}/Looks/{name}"))
    for link in prim.GetChildren():
        visuals = link.GetChild("visuals")
        name = next((m for pattern, m in LINK_MATERIALS.items() if re.fullmatch(pattern, link.GetName())), None)
        if not visuals or name is None:
            continue
        visuals.SetInstanceable(False)  # material bindings are skipped on instanced prims
        sim_utils.bind_visual_material(str(visuals.GetPath()), str(materials[name].GetPath()))
    add_black_trim(prim)
    for a, b in g1d_spec.filtered_pairs():
        rel = UsdPhysics.FilteredPairsAPI.Apply(prim.GetChild(a)).CreateFilteredPairsRel()
        rel.AddTarget(prim.GetChild(b).GetPath())
    # The wrist camera's lens bezel is a solid shell 10-12 mm in front of the lens (the real one has the aperture),
    # so it blacked out the view unless the near plane sat at 3 cm, and a 3 cm near plane cut away a held cup's wall.
    # It stays a collider; only its visual is hidden, letting the camera's near plane come in to 1 cm (DEX1_RGB).
    for side in ("left", "right"):
        for p in Usd.PrimRange(prim.GetChild(f"{side}_dex1_model_base_link")):
            if "housing_visual" in p.GetName() and p.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(p).MakeInvisible()
    fc = _lookup_spawn_finger_contact(prim_path, cfg)
    if fc is not None:
        apply_finger_contact(prim, fc)
    return prim


def get_robot_config(mobile=False, cfg: Mapping[str, Any] | None = None, **overrides):
    caller_cfg, robot_idx = g1d_spec.caller_robot_yml_cfg()
    eff_cfg = cfg if cfg is not None else caller_cfg
    finger_contact_raw = overrides.pop("finger_contact", None)
    if finger_contact_raw is None and isinstance(eff_cfg, Mapping):
        finger_contact_raw = eff_cfg.get("finger_contact")
    fc = g1d_spec.resolve_finger_contact(finger_contact_raw)

    spec = g1d_spec.get_robot_spec(mobile, cfg=eff_cfg, **overrides)
    art_cfg = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            func=_spawn_g1d,
            asset_path=spec.urdf_path,
            usd_dir=f"{ROBOTS_PATH}/g1d/{'usd_mobile' if mobile else 'usd'}",  # reused while the URDF/config hash matches
            fix_base=spec.fix_base,
            activate_contact_sensors=True,  # FINGER_PADS
            merge_fixed_joints=False,  # keep d435_link / *_dex1_base_link for cameras and the EE frame
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=spec.disable_gravity,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=spec.self_collisions,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=spec.fix_base,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=spec.init_joint_pos),
        actuators={name: implicit_actuator_cfg(**group.as_kwargs()) for name, group in spec.actuators.items()},
    )
    if fc is not None:
        try:
            _SPAWN_FINGER_CONTACT[art_cfg.spawn] = fc
        except TypeError:
            pass
        _ROBOT_IDX_FINGER_CONTACT[robot_idx] = fc
    else:
        _ROBOT_IDX_FINGER_CONTACT.pop(robot_idx, None)
    return art_cfg
