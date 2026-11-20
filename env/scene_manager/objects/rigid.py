from collections.abc import Mapping
import random
from typing import Any

from isaacsim.core.api.materials import PreviewSurface
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
import numpy as np
from omegaconf import DictConfig
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
import torch

from env.scene_manager.layout_manager import LayoutManager

CONTAINER_MIN_WALL = 0.0028  # m, minimum unscaled collision stave thickness for rigid-bean containers

try:
    from pxr import PhysxSchema as _PhysxSchema
except ImportError:  # standalone CPU pxr without omni.usd.schema.physx loaded
    _PhysxSchema = None


def _prepend_api_schema(prim, schema_token: str) -> None:
    """Ensure `schema_token` is recorded in `prim`'s `apiSchemas` metadata when PhysxSchema is unavailable."""
    stage = prim.GetStage()
    layer = stage.GetEditTarget().GetLayer()
    spec = Sdf.CreatePrimInLayer(layer, prim.GetPath())
    if spec.specifier == Sdf.SpecifierOver and not prim.GetTypeName():
        spec.specifier = Sdf.SpecifierDef
    existing = list(prim.GetAppliedSchemas())
    if schema_token not in existing:
        spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[schema_token]))


def _first_not_none(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def resolve_rigid_material_config(physics_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve RigidObject friction/restitution and optional PhysX material attributes with backward compatibility."""
    cfg = physics_config or {}
    fallback_friction = cfg.get("friction")
    static_raw = cfg.get("static_friction")
    dynamic_raw = cfg.get("dynamic_friction")
    static_friction = (
        float(static_raw)
        if static_raw is not None
        else (float(fallback_friction) if fallback_friction is not None else 0.6)
    )
    dynamic_friction = (
        float(dynamic_raw)
        if dynamic_raw is not None
        else (float(fallback_friction) if fallback_friction is not None else 1.5)
    )
    restitution_raw = cfg.get("restitution")
    restitution = float(restitution_raw) if restitution_raw is not None else 0.0
    return {
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "restitution": restitution,
        "friction_combine": _first_not_none(cfg, "friction_combine", "friction_combine_mode"),
        "restitution_combine": _first_not_none(cfg, "restitution_combine", "restitution_combine_mode"),
        "compliant_stiffness": _first_not_none(cfg, "compliant_stiffness", "compliant_contact_stiffness"),
        "compliant_damping": _first_not_none(cfg, "compliant_damping", "compliant_contact_damping"),
    }


def apply_extended_material_props(stage, material_prim_path: str, mat_cfg: Mapping[str, Any]) -> None:
    """Author optional PhysxMaterialAPI attributes (combine modes, compliant contact) when explicitly configured."""
    f_comb = mat_cfg.get("friction_combine")
    r_comb = mat_cfg.get("restitution_combine")
    c_stiff = mat_cfg.get("compliant_stiffness")
    c_damp = mat_cfg.get("compliant_damping")
    if f_comb is None and r_comb is None and c_stiff is None and c_damp is None:
        return
    mprim = stage.GetPrimAtPath(material_prim_path)
    if not mprim or not mprim.IsValid():
        return
    if _PhysxSchema is not None:
        api = _PhysxSchema.PhysxMaterialAPI.Apply(mprim)
        if f_comb is not None:
            api.CreateFrictionCombineModeAttr().Set(str(f_comb))
        if r_comb is not None:
            api.CreateRestitutionCombineModeAttr().Set(str(r_comb))
        if c_stiff is not None:
            api.CreateCompliantContactStiffnessAttr().Set(float(c_stiff))
        if c_damp is not None:
            api.CreateCompliantContactDampingAttr().Set(float(c_damp))
        return
    _prepend_api_schema(mprim, "PhysxMaterialAPI")
    if f_comb is not None:
        mprim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, False).Set(str(f_comb))
    if r_comb is not None:
        mprim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, False).Set(str(r_comb))
    if c_stiff is not None:
        mprim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float, False).Set(
            float(c_stiff)
        )
    if c_damp is not None:
        mprim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float, False).Set(
            float(c_damp)
        )


def apply_rigid_physics_overrides(stage, prim_path: str, physics_config: Mapping[str, Any] | None) -> None:
    """Author optional center_of_mass, PhysxRigidBodyAPI, and PhysxCollisionAPI overrides onto a rigid object prim."""
    cfg = physics_config or {}
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return

    com = cfg.get("center_of_mass")
    if com is not None:
        cx, cy, cz = (float(com[0]), float(com[1]), float(com[2]))
        UsdPhysics.MassAPI.Apply(prim).CreateCenterOfMassAttr().Set(Gf.Vec3f(cx, cy, cz))

    max_depen = cfg.get("max_depenetration_velocity")
    lin_damp = cfg.get("linear_damping")
    ang_damp = cfg.get("angular_damping")
    sleep_thr = cfg.get("sleep_threshold")
    slop = _first_not_none(cfg, "contact_slop", "contact_slop_coefficient")
    pos_iters = _first_not_none(cfg, "position_iterations", "solver_position_iteration_count")
    vel_iters = _first_not_none(cfg, "velocity_iterations", "solver_velocity_iteration_count")
    if any(v is not None for v in (max_depen, lin_damp, ang_damp, sleep_thr, slop, pos_iters, vel_iters)):
        if _PhysxSchema is not None:
            rb_api = _PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            if max_depen is not None:
                rb_api.CreateMaxDepenetrationVelocityAttr().Set(float(max_depen))
            if lin_damp is not None:
                rb_api.CreateLinearDampingAttr().Set(float(lin_damp))
            if ang_damp is not None:
                rb_api.CreateAngularDampingAttr().Set(float(ang_damp))
            if sleep_thr is not None:
                rb_api.CreateSleepThresholdAttr().Set(float(sleep_thr))
            if slop is not None:
                rb_api.CreateContactSlopCoefficientAttr().Set(float(slop))
            if pos_iters is not None:
                rb_api.CreateSolverPositionIterationCountAttr().Set(int(pos_iters))
            if vel_iters is not None:
                rb_api.CreateSolverVelocityIterationCountAttr().Set(int(vel_iters))
        else:
            _prepend_api_schema(prim, "PhysxRigidBodyAPI")
            if max_depen is not None:
                prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float, False).Set(
                    float(max_depen)
                )
            if lin_damp is not None:
                prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, False).Set(
                    float(lin_damp)
                )
            if ang_damp is not None:
                prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float, False).Set(
                    float(ang_damp)
                )
            if sleep_thr is not None:
                prim.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float, False).Set(
                    float(sleep_thr)
                )
            if slop is not None:
                prim.CreateAttribute("physxRigidBody:contactSlopCoefficient", Sdf.ValueTypeNames.Float, False).Set(
                    float(slop)
                )
            if pos_iters is not None:
                prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int, False).Set(
                    int(pos_iters)
                )
            if vel_iters is not None:
                prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, False).Set(
                    int(vel_iters)
                )

    torsional = cfg.get("torsional_patch_radius")
    min_torsional = cfg.get("min_torsional_patch_radius")
    c_offset = cfg.get("contact_offset")
    r_offset = cfg.get("rest_offset")
    if any(v is not None for v in (torsional, min_torsional, c_offset, r_offset)):
        proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for p in Usd.PrimRange(prim, proxies):
            if p.IsInstance():
                p.SetInstanceable(False)
        col_prims = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)] or [prim]
        layer = stage.GetEditTarget().GetLayer()
        existing_schemas = {p.GetPath(): list(p.GetAppliedSchemas()) for p in col_prims}
        with Sdf.ChangeBlock():
            for cprim in col_prims:
                cpath = cprim.GetPath()
                spec = Sdf.CreatePrimInLayer(layer, cpath)
                if spec.specifier == Sdf.SpecifierOver and not cprim.GetTypeName():
                    spec.specifier = Sdf.SpecifierDef
                if "PhysxCollisionAPI" not in existing_schemas[cpath]:
                    spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysxCollisionAPI"]))

                def _set_col_float(attr_name: str, val: float) -> None:
                    aspec = spec.attributes.get(attr_name)
                    if aspec is None:
                        aspec = Sdf.AttributeSpec(spec, attr_name, Sdf.ValueTypeNames.Float, Sdf.VariabilityVarying)
                    aspec.default = float(val)

                if c_offset is not None:
                    _set_col_float("physxCollision:contactOffset", c_offset)
                if r_offset is not None:
                    _set_col_float("physxCollision:restOffset", r_offset)
                if torsional is not None:
                    _set_col_float("physxCollision:torsionalPatchRadius", torsional)
                if min_torsional is not None:
                    _set_col_float("physxCollision:minTorsionalPatchRadius", min_torsional)


class RigidObject(SingleRigidPrim, SingleGeometryPrim):
    """Rigid body object with physical properties and collision detection.
    Combines geometry (visual/collision) and rigid body dynamics.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        inst_config: DictConfig,
        env_origin: torch.Tensor,
        default_pos: tuple,
        default_ori: tuple,
        scale: tuple,
    ):
        if usd_path:
            prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        else:
            prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            error_message = (
                f"Failed to load USD from {usd_path} to {prim_path}"
                if usd_path
                else f"Failed to find an existing prim at path {prim_path}"
            )
            raise RuntimeError(error_message)

        """Initialize a rigid body object with geometry and physics properties."""
        self.stage = prim.GetStage()
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.model_name, self.model_id = LayoutManager.decode_inst_name(self.instance_name)

        self.instance_config = inst_config
        self.visual_config = self.instance_config.get("visual", {})
        self.physics_config = self.instance_config.get("physics", {})

        self.env_origin = env_origin
        if isinstance(self.env_origin, torch.Tensor):
            self.env_origin = self.env_origin.cpu().numpy().tolist()
        self.usd_path = usd_path
        self.usd_prim_path = prim_path
        self.default_pos = default_pos
        self.default_ori = default_ori
        self.scale = scale
        # tabletop objects are capped at 0.5 kg; furniture the robot pushes (lounge chairs) raises it in its layout
        self.mass = min(self.physics_config.get("mass", 0.5), self.physics_config.get("mass_cap", 0.5))
        self.visible = self.visual_config.get("visible", True)

        mat_cfg = resolve_rigid_material_config(self.physics_config)
        self._resolved_material_config = mat_cfg
        self.physics_material_path = find_unique_string_name(
            prim_path + "/physics_material",
            lambda x: not is_prim_path_valid(x),
        )
        self.physics_material = PhysicsMaterial(
            prim_path=self.physics_material_path,
            static_friction=mat_cfg["static_friction"],
            dynamic_friction=mat_cfg["dynamic_friction"],
            restitution=mat_cfg["restitution"],
        )
        apply_extended_material_props(self.stage, self.physics_material_path, mat_cfg)

        self._prepare_container_colliders()

        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            scale=self.scale,
            visible=self.visible,
            collision=False,
            track_contact_forces=False,
        )

        SingleRigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            translation=self.default_pos,
            orientation=self.default_ori,
            scale=self.scale,
        )
        self._default_linear_velocity = [0.0, 0.0, 0.0]
        self._default_angular_velocity = [0.0, 0.0, 0.0]

        self._setup_physics()
        if self.visual_config.get("color") is not None:
            self._apply_color(self.visual_config["color"], self.visual_config.get("color_parts"))

    def _apply_color(self, color, parts=None):
        """Bind a plain colour (linear RGB, 0-1) over the asset's own looks, set per layout (scripts/g1d/build_asset.py
        RANDOMIZE): on the meshes named in `parts` (e.g. a cup body, keeping its label and see-through parts) or on the
        whole object."""
        path = find_unique_string_name(self._prim_path + "/Looks/layout_color", lambda x: not is_prim_path_valid(x))
        look = PreviewSurface(path, color=np.asarray(color[:3], dtype=np.float32), roughness=0.5, metallic=0.0)
        root = get_prim_at_path(self._prim_path)
        targets = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and p.GetName() in parts] if parts else [root]
        for prim in targets:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(look.material, UsdShade.Tokens.strongerThanDescendants)

    def _get_object_transform(self, device=None):
        """
        Get Pose and Orientation Matrix

        Returns:
            Tuple of (obj_pos, obj_quat, device)
        """
        obj_translation, obj_orientation = self.get_local_pose()

        if device is None:
            if isinstance(obj_translation, torch.Tensor):
                device = obj_translation.device
            elif isinstance(obj_orientation, torch.Tensor):
                device = obj_orientation.device
            else:
                device = "cpu"

        if isinstance(obj_translation, torch.Tensor):
            obj_pos = obj_translation.to(dtype=torch.float32, device=device).squeeze()[:3]
        else:
            obj_pos = torch.tensor(obj_translation, dtype=torch.float32, device=device)[:3]

        if isinstance(obj_orientation, torch.Tensor):
            obj_quat = obj_orientation.to(dtype=torch.float32, device=device).squeeze()[:4]
        else:
            obj_quat = torch.tensor(obj_orientation, dtype=torch.float32, device=device)[:4]

        return obj_pos, obj_quat, device

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def sample_mesh_vertices(self):
        """
        Get current mesh vertex positions for this rigid object.

        Returns (points_world, points_local, pos_world, ori_world).
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return np.array([]), np.array([]), None, None

        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        points_local = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        world_tf = omni.usd.get_world_transform_matrix(mesh_prim.GetPrim())

        points_world = np.array(
            [list(world_tf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))) for p in points_local],
            dtype=np.float32,
        )

        rot_quat = world_tf.ExtractRotationQuat()
        pos_world = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
        ori_world = np.array([rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32)

        return points_world, points_local, pos_world, ori_world

    def get_bbox(self, is_relative=True):
        points_world, points_local, pos_world, ori_world = self.sample_mesh_vertices()

        if is_relative:
            pos = np.subtract(pos_world, self.env_origin)
        else:
            pos = pos_world
        ori = ori_world
        points_local = points_local * np.asarray(self.scale, dtype=np.float32)  # the layout's scale, as rendered
        bbox = np.concatenate([points_local.min(axis=0), points_local.max(axis=0)])
        return pos, ori, bbox

    def hide_prim(self, prim_path: str):
        """
        Hide a prim by setting its visibility to invisible.
        This will make the prim and all its children invisible and ignored by physics.

        Args:
            prim_path: The prim path to hide
        """
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f"Warning: Invalid prim path {prim_path}")
                return

            # Set visibility to invisible
            visibility_attribute = prim.GetAttribute("visibility")
            if visibility_attribute is None:
                # Create the visibility attribute if it doesn't exist
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
            else:
                visibility_attribute.Set("invisible")

        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def _setup_physics(self):
        """Configure physics properties (rigid type, mass, COM, damping, collision patch, material) from config."""
        if self.mass <= 0:
            self.mass = 0.05
        self.set_mass(self.mass)
        apply_rigid_physics_overrides(self.stage, self._prim_path, self.physics_config)

        if self._default_linear_velocity is not None or self._default_angular_velocity is not None:
            self.set_default_state(
                linear_velocity=self._default_linear_velocity,
                angular_velocity=self._default_angular_velocity,
            )
        if self.physics_material is not None:
            self.apply_physics_material(physics_material=self.physics_material)

    def apply_saved_pose(self):
        """Apply the saved eval-layout pose and clear velocities."""
        self.set_local_pose(translation=self.default_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))
        self._apply_default_velocities()

    def relocate_offscreen(self):
        """Move this object far away so stale prims do not affect the next layout."""
        _FAR_CENTER = (100000.0, 100000.0, 100000)
        _FAR_JITTER = 1000.0
        far_pos = (
            _FAR_CENTER[0] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[1] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[2] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
        )
        self.set_local_pose(translation=far_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))
        self._apply_default_velocities()

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)

    def _apply_default_velocities(self):
        """Re-apply default linear/angular velocity if configured."""
        if self._default_linear_velocity is not None:
            self.set_linear_velocity(torch.tensor(self._default_linear_velocity))
        if self._default_angular_velocity is not None:
            self.set_angular_velocity(torch.tensor(self._default_angular_velocity))

    def _prepare_container_colliders(self):
        """Harden thin-walled rigid-bean containers in the in-memory stage before PhysX parses the rigid prim:
        clear stale child mass attributes on SimReady packages, match the robot articulation's 5 m/s depenetration cap
        (0.5 m/s let a 55 N Dex1-1 finger punch through a 22 g paper cup wall before the cup could center) with
        GranularObject's 32/2 solver iterations and speculative CCD, replace dynamic SDF colliders (which eject grain
        heaps under gripper squeeze) with tight convex decompositions, and pad ultra-thin (< CONTAINER_MIN_WALL)
        convex-hull wall staves outward without touching the inner cavity surface or disk USD files."""
        cavity = self.instance_config.get("cavity")
        if not cavity or not hasattr(cavity, "get") or "axis" not in cavity or _PhysxSchema is None:
            return
        root = get_prim_at_path(self._prim_path)
        rb = _PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rb.CreateMaxDepenetrationVelocityAttr().Set(5.0)
        rb.CreateSolverPositionIterationCountAttr().Set(32)
        rb.CreateSolverVelocityIterationCountAttr().Set(2)
        rb.CreateEnableSpeculativeCCDAttr().Set(True)
        axis = np.asarray(cavity["axis"], dtype=float)
        cache = UsdGeom.XformCache()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim == root or prim.IsInstanceProxy():
                continue
            for prop in ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"):
                if prim.HasAttribute(prop):
                    prim.GetAttribute(prop).Clear()
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            approx_attr = prim.GetAttribute("physics:approximation")
            approx = approx_attr.Get() if approx_attr else None
            if approx == "sdf":
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
                cd = _PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                cd.CreateMaxConvexHullsAttr().Set(128)
                cd.CreateHullVertexLimitAttr().Set(32)
                cd.CreateErrorPercentageAttr().Set(0.5)
                cd.CreateMinThicknessAttr().Set(0.003)
                cd.CreateVoxelResolutionAttr().Set(500000)
                cd.CreateShrinkWrapAttr().Set(True)
            elif approx == "convexHull" and prim.IsA(UsdGeom.Mesh):
                self._pad_thin_stave_outward(UsdGeom.Mesh(prim), root, axis, cache)

    @staticmethod
    def _pad_thin_stave_outward(mesh: UsdGeom.Mesh, root: Usd.Prim, axis: np.ndarray, cache: UsdGeom.XformCache):
        local = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
        if len(local) < 6:
            return
        tf = cache.ComputeRelativeTransform(mesh.GetPrim(), root)[0]
        mat = np.asarray(tf, dtype=float)
        pts = local @ mat[:3, :3] + mat[3, :3]
        rad = np.hypot(pts[:, 0] - axis[0], pts[:, 1] - axis[1])
        if pts[:, 2].max() - pts[:, 2].min() <= 0.015 or rad.min() <= 0.008:
            return
        top = np.abs(pts[:, 2] - pts[:, 2].max()) < 0.0015
        if top.sum() < 4:
            return
        thick = float(rad[top].max() - rad[top].min())
        if thick >= CONTAINER_MIN_WALL:
            return
        pad = CONTAINER_MIN_WALL - thick
        updated = pts.copy()
        extras = []
        for z0 in np.unique(pts[:, 2].round(3)):
            band = np.abs(pts[:, 2] - z0) < 0.0015
            if band.sum() < 2:
                continue
            span = float(rad[band].max() - rad[band].min())
            if span > 0.0002:
                outer = band & (rad > 0.5 * (rad[band].max() + rad[band].min()))
                dx, dy = pts[outer, 0] - axis[0], pts[outer, 1] - axis[1]
                dn = np.maximum(np.hypot(dx, dy), 1e-9)
                updated[outer, 0] += pad * dx / dn
                updated[outer, 1] += pad * dy / dn
            elif len(local) + len(extras) + int(band.sum()) <= 60:
                dx, dy = pts[band, 0] - axis[0], pts[band, 1] - axis[1]
                dn = np.maximum(np.hypot(dx, dy), 1e-9)
                ring = pts[band].copy()
                ring[:, 0] += CONTAINER_MIN_WALL * dx / dn
                ring[:, 1] += CONTAINER_MIN_WALL * dy / dn
                extras.append(ring)
        if extras:
            updated = np.vstack([updated, *extras])
        inv = np.asarray(tf.GetInverse(), dtype=float)
        new_local = updated @ inv[:3, :3] + inv[3, :3]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_local.astype(np.float32)))

    def destroy(self):
        self._rigid_prim_view.disable_gravities()
        self._geometry_prim_view.disable_collision()
        self._rigid_prim_view.set_visibilities([False])
        self._geometry_prim_view.set_visibilities([False])
        self.hide_prim(self._prim_path)
