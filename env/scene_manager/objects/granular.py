from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from env import sim_backend
from scripts.objects import cup_profile

if TYPE_CHECKING:
    from omegaconf import DictConfig
    import torch

try:
    from pxr import Gf, Sdf, Vt
except ImportError:  # pragma: no cover - CPU CI without usd-core
    Gf = Sdf = Vt = None

try:
    from isaacsim.core.utils.stage import get_current_stage
except ImportError:  # pragma: no cover - CPU tests outside Isaac Sim

    def get_current_stage():
        raise RuntimeError("Isaac Sim stage is unavailable; pass stage=... explicitly in CPU tests.")


BODY_SCHEMAS = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
HULL_SCHEMAS = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
CAPSULE_COLLIDER_SCHEMAS = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
CONTACT_OFFSET = 0.001
# PhysX's cooked hull lies ~0.35 mm inside the bean's points; a positive rest offset puts the surface back where
# the points are (rigid_bean_probe.py). Analytic capsules have no hull shrinkage and default to rest_offset = 0.0.
REST_OFFSET = 0.00035
# Solver position iterations; PhysX solves a contact island with the most of its bodies'. With 4, a gripper closing
# on a cup filled with ~3000 grains converged so poorly that the cup was pushed 2 cm out of the grasp and tilted 20 deg,
# and the heap held until 120 deg before sliding out at once; 32 grips and pours as the real cup (86-93 deg onset).
ITERATIONS = 32
VELOCITY_ITERATIONS = 2
MAX_DEPENETRATION_VELOCITY = 0.5
FRICTION_COMBINE = "min"

# Verified default physics preset for new pour-beans tasks (validated on GPU Slurm GPU in pour_beans_real520_sz@v2):
# analytic capsule beans stop within ~4 cm / 0.3 s on the tabletop while pouring cleanly without wall/floor leaks.
POUR_BEANS_DEFAULT_PHYSICS: dict[str, object] = {
    "type": "granular",
    "shape": "capsule",
    "friction": 0.38,
    "friction_combine": "average",
    "restitution": 0.02,
    "restitution_combine": "min",
    "contact_offset": 0.001,
    "rest_offset": 0.0,
    "torsional_patch_radius": 0.0008,
    "max_depenetration_velocity": 0.5,
    "iterations": 32,
    "position_iterations": 32,
    "velocity_iterations": 1,
    "linear_damping": 2.0,
    "angular_damping": 60.0,
    "sleep_threshold": 0.005,
}
POUR_BEANS_CUP_CONTACT: dict[str, object] = {
    "static_friction": 0.55,
    "dynamic_friction": 0.45,
    "restitution": 0.05,
    "max_depenetration_velocity": 5.0,
    "position_iterations": 32,
    "velocity_iterations": 1,
}
POUR_BEANS_SOURCE_CUP_OFFSETS: dict[str, float] = {
    "contact_offset": 0.003,
    "rest_offset": 0.0007,
}
POUR_BEANS_TARGET_CUP_OFFSETS: dict[str, float] = {
    "contact_offset": 0.004,
    "rest_offset": 0.001,
}


def _pxr():
    if Gf is not None and Sdf is not None and Vt is not None:
        return Gf, Sdf, Vt
    from pxr import Gf as _Gf, Sdf as _Sdf, Vt as _Vt

    return _Gf, _Sdf, _Vt


def _normalize_shape(physics: Mapping) -> str:
    raw = physics.get("shape", "convex_hull")
    shape = "convex_hull" if raw in (None, "convex_hull", "convexHull") else str(raw)
    if shape not in ("convex_hull", "capsule"):
        raise ValueError(f"Unsupported granular collider shape {raw!r}; expected 'convex_hull' or 'capsule'")
    return shape


def _pack_points(pack) -> np.ndarray:
    key = "hull" if "hull" in pack else "points"
    return np.asarray(pack[key], dtype=float)


def _capsule_base_dims(pack, physics: Mapping) -> tuple[float, float]:
    r_override = physics.get("capsule_radius")
    h_override = physics.get("capsule_height")
    if r_override is not None and h_override is not None:
        return float(r_override), max(0.0, float(h_override))
    pts = _pack_points(pack)
    a_pts = float(np.abs(pts[:, 0]).max())
    r_pts = float(np.hypot(pts[:, 1], pts[:, 2]).max())
    base_r = float(r_override) if r_override is not None else r_pts
    base_h = max(0.0, float(h_override)) if h_override is not None else max(0.0, 2.0 * (a_pts - base_r))
    return base_r, base_h


def _capsule_grain_dims(base_r: float, base_h: float, grain_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(grain_scale, dtype=float)
    base_a = base_r + 0.5 * base_h
    radii = base_r * (0.5 * (scale[:, 1] + scale[:, 2]))
    heights = np.maximum(0.0, 2.0 * (base_a * scale[:, 0] - radii))
    return radii, heights


class GranularObject:
    """Granular media (beans, grains) as one rigid body per grain, in place of FluidObject's PBD spheres.

    The grains come from a pack (scripts/g1d/rigid_bean_probe.py --pack): their settled poses in a container's frame,
    per-grain scale and mass, and the grain's hull. The layout entry (Fluid section, physics.type "granular") places
    that frame: default_pos / default_ori (w, x, y, z), env-local. physics supports:
    - pack, count (the lowest `count` grains of the pack, so a smaller count is a lower fill);
    - shape ("convex_hull" default, or analytic "capsule" along X with optional capsule_radius / capsule_height);
    - friction, static_friction, dynamic_friction, friction_combine (default "min"), restitution, restitution_combine;
    - contact_offset, rest_offset (default 0.00035 for convex_hull, 0.0 for capsule), contact_slop (convex_hull);
    - torsional_patch_radius, min_torsional_patch_radius;
    - position_iterations (or legacy iterations, default 32), velocity_iterations (default 2),
      max_depenetration_velocity (default 0.5), speculative_ccd (default on for convex_hull grains), linear_damping,
      angular_damping, sleep_threshold.
    visual: color (linear RGB), roughness. With physics.cavity (a cup profile, scripts/objects/cup_profile.py, at
    cavity_scale) the frame is that cup's instead, and the grains the pack's that fit in its cavity, the pack turned
    `yaw` rad about its axis (its inner floor at `floor`, cup_profile.PACK_FLOOR): cup_profile.pack_fill.

    Like FluidObject, get_particle_positions()[0] are the grain centres, mapped to env-local by init_ori / init_pos
    (here the identity), so the bean counters and fill checks read either kind. `grain` (mean volume, packing) gives a
    cup's capacity in grains (round_cup_shape.capacity).
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        inst_config: DictConfig | dict,
        env_origin: torch.Tensor | np.ndarray | tuple,
        default_pos: tuple,
        default_ori: tuple,
        scale: tuple = (1.0, 1.0, 1.0),
        stage=None,
    ):
        self.prim_path = self.usd_prim_path = prim_path
        self.instance_name = prim_path.split("/")[-1]
        self.instance_config = inst_config
        if isinstance(inst_config, Mapping):
            cfg = inst_config
        else:
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(inst_config, resolve=True)
        physics, visual = cfg.get("physics", {}), cfg.get("visual", {})
        origin = env_origin.detach().cpu().numpy() if hasattr(env_origin, "detach") else env_origin
        self.env_origin = np.asarray(origin, dtype=float)[:3]
        self.init_pos, self.init_ori = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        self.stage = stage if stage is not None else get_current_stage()

        pack_src = physics["pack"]
        ctx = nullcontext(pack_src) if isinstance(pack_src, Mapping) else np.load(pack_src)
        with ctx as pack:
            if physics.get("cavity"):
                order, pose = self._in_cavity(pack, physics)
            else:
                order = np.argsort(pack["pose"][:, 2], kind="stable")[: physics.get("count", len(pack["pose"]))]
                pose = pack["pose"][order]
            self.grain_scale = np.asarray(pack["scale"][order], dtype=float)
            mass = np.asarray(pack["mass"][order], dtype=float)
            frame = Rotation.from_quat(np.roll(np.asarray(default_ori, dtype=float), -1))
            self.local_pos = frame.apply(pose[:, :3]) + np.asarray(default_pos, dtype=float)
            self.local_quat = (frame * Rotation.from_quat(pose[:, 3:])).as_quat()  # x, y, z, w
            volume = float(pack["volume"]) * np.prod(self.grain_scale, axis=1)
            self.grain = dict(volume=float(volume.mean()), packing=float(pack["packing"]))
            self.count = len(order)
            self._author(pack, mass, physics, visual)
        self.view = None

    @staticmethod
    def _in_cavity(pack, physics):
        """(pack indices, poses in the cup frame) of the pack's grains that fit in a cup's cavity
        (cup_profile.pack_fill), the pack turned `yaw` rad about its axis."""
        shape = _normalize_shape(physics)
        if shape == "capsule" or "hull" not in pack:
            base_r, base_h = _capsule_base_dims(pack, physics)
            radii, heights = _capsule_grain_dims(base_r, base_h, pack["scale"])
            half = np.maximum(radii, radii + 0.5 * heights)
        else:
            half = cup_profile.grain_half(pack)
        yaw = float(physics.get("yaw", 0.0))
        order, centres = cup_profile.pack_fill(
            physics["cavity"],
            float(physics.get("cavity_scale", 1.0)),
            pack["pose"][:, :3],
            half,
            float(physics.get("floor", cup_profile.PACK_FLOOR)),
            yaw,
            physics.get("count"),
        )
        turn = Rotation.from_euler("z", yaw) * Rotation.from_quat(pack["pose"][order, 3:])
        return order, np.c_[centres, turn.as_quat()]

    def _author(self, pack, mass, physics, visual):
        """All grains in one Sdf change block: either an Xform rigid body with a convex-hull Mesh child or a single-prim
        analytic Capsule per grain; one physics material and one preview surface shared by all."""
        gf, sdf, vt = _pxr()
        t = sdf.ValueTypeNames
        xform_order = vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
        shape = _normalize_shape(physics)
        layer = self.stage.GetEditTarget().GetLayer()
        root = sdf.Path(self.prim_path)

        if shape == "capsule":
            base_r, base_h = _capsule_base_dims(pack, physics)
            cap_radii, cap_heights = _capsule_grain_dims(base_r, base_h, self.grain_scale)
            if "faces" in pack and ("hull" in pack or "points" in pack):
                hull = _pack_points(pack)
                faces = pack["faces"]
                semi = np.abs(hull).max(axis=0)
                hull_normals = hull / semi**2
                hull_normals /= np.linalg.norm(hull_normals, axis=1, keepdims=True)
                points = vt.Vec3fArray.FromNumpy(hull.astype(np.float32))
                normals = vt.Vec3fArray.FromNumpy(hull_normals.astype(np.float32))
                counts = vt.IntArray([3] * len(faces))
                indices = vt.IntArray(np.asarray(faces).reshape(-1).tolist())
            else:
                points = normals = counts = indices = None
        else:
            hull = _pack_points(pack)
            faces = pack["faces"]
            semi = np.abs(hull).max(axis=0)
            hull_normals = hull / semi**2
            hull_normals /= np.linalg.norm(hull_normals, axis=1, keepdims=True)
            points = vt.Vec3fArray.FromNumpy(hull.astype(np.float32))
            normals = vt.Vec3fArray.FromNumpy(hull_normals.astype(np.float32))
            counts = vt.IntArray([3] * len(faces))
            indices = vt.IntArray(np.asarray(faces).reshape(-1).tolist())
            cap_radii = cap_heights = None
        # the child of each bean that the renderer draws
        self._visual_child = "hull" if shape == "convex_hull" else "visual" if points is not None else "capsule"

        friction = float(physics.get("friction", 0.3))
        static_friction = float(physics["static_friction"]) if physics.get("static_friction") is not None else friction
        dynamic_friction = (
            float(physics["dynamic_friction"]) if physics.get("dynamic_friction") is not None else friction
        )
        friction_combine = physics.get("friction_combine", FRICTION_COMBINE)
        restitution_combine = physics.get("restitution_combine")
        restitution = float(physics.get("restitution", 0.1))

        offset = float(physics.get("contact_offset", CONTACT_OFFSET))
        default_rest = 0.0 if shape == "capsule" else REST_OFFSET
        rest = float(physics["rest_offset"]) if physics.get("rest_offset") is not None else default_rest
        torsional = (
            float(physics["torsional_patch_radius"]) if physics.get("torsional_patch_radius") is not None else None
        )
        min_torsional = (
            float(physics["min_torsional_patch_radius"])
            if physics.get("min_torsional_patch_radius") is not None
            else None
        )

        pos_iters_raw = physics.get("position_iterations")
        iters = int(pos_iters_raw) if pos_iters_raw is not None else int(physics.get("iterations", ITERATIONS))
        vel_iters = int(physics.get("velocity_iterations", VELOCITY_ITERATIONS))
        max_depen = float(physics.get("max_depenetration_velocity", MAX_DEPENETRATION_VELOCITY))
        ccd = bool(physics.get("speculative_ccd", shape == "convex_hull"))
        linear_damping = float(physics["linear_damping"]) if physics.get("linear_damping") is not None else None
        angular_damping = float(physics["angular_damping"]) if physics.get("angular_damping") is not None else None
        sleep_threshold = float(physics["sleep_threshold"]) if physics.get("sleep_threshold") is not None else None
        contact_slop = float(physics["contact_slop"]) if physics.get("contact_slop") is not None else None

        def prim(path, type_name, schemas=()):
            spec = sdf.CreatePrimInLayer(layer, path)
            spec.specifier, spec.typeName = sdf.SpecifierDef, type_name
            if schemas:
                spec.SetInfo("apiSchemas", sdf.TokenListOp.CreateExplicit(list(schemas)))
            return spec

        def attr(spec, name, type_name, value, uniform=False):
            a = sdf.AttributeSpec(spec, name, type_name, sdf.VariabilityUniform if uniform else sdf.VariabilityVarying)
            a.default = value
            return a

        def bind(spec, name, target):
            sdf.RelationshipSpec(spec, name, custom=False).targetPathList.Prepend(target)

        self.stage.DefinePrim(self.prim_path, "Xform")  # and its ancestors: PhysX skips prims under an undefined one
        with sdf.ChangeBlock():
            phys = prim(root.AppendChild("physics_material"), "Material", ["PhysicsMaterialAPI", "PhysxMaterialAPI"])
            attr(phys, "physics:staticFriction", t.Float, static_friction)
            attr(phys, "physics:dynamicFriction", t.Float, dynamic_friction)
            if friction_combine:
                attr(phys, "physxMaterial:frictionCombineMode", t.Token, str(friction_combine), uniform=True)
            if restitution_combine:
                attr(phys, "physxMaterial:restitutionCombineMode", t.Token, str(restitution_combine), uniform=True)
            attr(phys, "physics:restitution", t.Float, restitution)
            look = prim(root.AppendChild("look"), "Material")
            shader = prim(look.path.AppendChild("shader"), "Shader")
            attr(shader, "info:id", t.Token, "UsdPreviewSurface", uniform=True)
            color = gf.Vec3f(*map(float, visual.get("color", (0.08, 0.01, 0.01))))
            attr(shader, "inputs:diffuseColor", t.Color3f, color)
            attr(shader, "inputs:roughness", t.Float, float(visual.get("roughness", 0.3)))
            surface = attr(shader, "outputs:surface", t.Token, "")
            attr(look, "outputs:surface", t.Token, "").connectionPathList.Prepend(surface.path)
            unit_scale = gf.Vec3f(1.0, 1.0, 1.0)
            scale_order = vt.TokenArray(["xformOp:scale"])
            for i in range(self.count):
                bean_path = root.AppendChild(f"bean_{i}")
                if shape == "capsule":
                    body = prim(bean_path, "Xform", BODY_SCHEMAS)
                    scale_vec = unit_scale
                    collider = prim(body.path.AppendChild("capsule"), "Capsule", CAPSULE_COLLIDER_SCHEMAS)
                    r_i, h_i = float(cap_radii[i]), float(cap_heights[i])
                    hx = r_i + 0.5 * h_i
                    attr(collider, "axis", t.Token, "X", uniform=True)
                    attr(collider, "radius", t.Double, r_i)
                    attr(collider, "height", t.Double, h_i)
                    attr(
                        collider,
                        "extent",
                        t.Float3Array,
                        vt.Vec3fArray([gf.Vec3f(-hx, -r_i, -r_i), gf.Vec3f(hx, r_i, r_i)]),
                    )
                    if points is not None:
                        attr(collider, "purpose", t.Token, "guide", uniform=True)
                        vis = prim(body.path.AppendChild("visual"), "Mesh", ["MaterialBindingAPI"])
                        attr(vis, "points", t.Point3fArray, points)
                        attr(vis, "normals", t.Normal3fArray, normals).SetInfo("interpolation", "vertex")
                        attr(vis, "faceVertexCounts", t.IntArray, counts)
                        attr(vis, "faceVertexIndices", t.IntArray, indices)
                        attr(vis, "subdivisionScheme", t.Token, "none", uniform=True)
                        attr(vis, "xformOp:scale", t.Float3, gf.Vec3f(*map(float, self.grain_scale[i])))
                        attr(vis, "xformOpOrder", t.TokenArray, scale_order, uniform=True)
                        visual_target = vis
                    else:
                        visual_target = collider
                else:
                    body = prim(bean_path, "Xform", BODY_SCHEMAS)
                    scale_vec = gf.Vec3f(*map(float, self.grain_scale[i]))
                    mesh = prim(body.path.AppendChild("hull"), "Mesh", HULL_SCHEMAS)
                    attr(mesh, "points", t.Point3fArray, points)
                    attr(mesh, "normals", t.Normal3fArray, normals).SetInfo("interpolation", "vertex")
                    attr(mesh, "faceVertexCounts", t.IntArray, counts)
                    attr(mesh, "faceVertexIndices", t.IntArray, indices)
                    attr(mesh, "subdivisionScheme", t.Token, "none", uniform=True)
                    attr(mesh, "physics:approximation", t.Token, "convexHull", uniform=True)
                    collider = mesh
                    visual_target = mesh

                x, y, z, w = map(float, self.local_quat[i])
                attr(body, "xformOp:translate", t.Double3, gf.Vec3d(*map(float, self.local_pos[i])))
                attr(body, "xformOp:orient", t.Quatf, gf.Quatf(w, x, y, z))
                attr(body, "xformOp:scale", t.Float3, scale_vec)
                attr(body, "xformOpOrder", t.TokenArray, xform_order, uniform=True)
                attr(body, "physics:mass", t.Float, float(mass[i]))
                attr(body, "physxRigidBody:solverPositionIterationCount", t.Int, iters)
                attr(body, "physxRigidBody:solverVelocityIterationCount", t.Int, vel_iters)
                attr(body, "physxRigidBody:maxDepenetrationVelocity", t.Float, max_depen)
                attr(body, "physxRigidBody:enableSpeculativeCCD", t.Bool, ccd)
                if linear_damping is not None:
                    attr(body, "physxRigidBody:linearDamping", t.Float, linear_damping)
                if angular_damping is not None:
                    attr(body, "physxRigidBody:angularDamping", t.Float, angular_damping)
                if sleep_threshold is not None:
                    attr(body, "physxRigidBody:sleepThreshold", t.Float, sleep_threshold)
                if shape == "convex_hull" and contact_slop is not None:
                    attr(body, "physxRigidBody:contactSlopCoefficient", t.Float, contact_slop)

                attr(collider, "physxCollision:contactOffset", t.Float, offset)
                attr(collider, "physxCollision:restOffset", t.Float, rest)
                if torsional is not None:
                    attr(collider, "physxCollision:torsionalPatchRadius", t.Float, torsional)
                if min_torsional is not None:
                    attr(collider, "physxCollision:minTorsionalPatchRadius", t.Float, min_torsional)
                bind(visual_target, "material:binding", look.path)
                bind(collider, "material:binding:physics", phys.path)

    def initialize(self):
        if sim_backend.current() == sim_backend.ISAACLAB_NEWTON:  # the rebuilt Newton model already has the grains
            self._view()
            return
        from isaaclab.sim import SimulationContext
        from omni.physx import get_physx_simulation_interface
        import omni.usd

        get_physx_simulation_interface().flush_changes()
        sim = SimulationContext.instance()
        if sim is not None and sim.is_fabric_enabled():
            # PhysX writes poses to Fabric only for the bodies it found when it attached to the stage; grains spawned
            # later would render where they were spawned
            from omni.physxfabric import get_physx_fabric_interface

            fabric = get_physx_fabric_interface()
            fabric.detach_stage()
            fabric.attach_stage(omni.usd.get_context().get_stage_id())
        self._view()

    def sync_fabric_visuals(self) -> None:
        """Put the grains' visual prims into Fabric as authored, where the renderer reads them; call once the reset is
        done (TaskEnvCommon.sync_granular_visuals).

        Without it the grains are simulated but never drawn: Fabric holds those prims only as the shells PhysX made (no
        points, no material), and a copy populated during the reset is emptied again before it ends. usdrt populates a
        prim from USD on access only if it is not in Fabric yet, so the shell is removed first."""
        stage = self._fabric_stage()
        if stage is None:
            return
        from usdrt import Sdf as RtSdf

        for path in self._visual_paths():
            stage.RemovePrim(RtSdf.Path(path))
            stage.GetPrimAtPath(path)

    def _fabric_stage(self):
        """The Fabric stage the renderer reads, or None where the grains are not drawn from Fabric."""
        if sim_backend.current() == sim_backend.ISAACLAB_NEWTON:
            return None
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if sim is None or not sim.is_fabric_enabled():
            return None
        import omni.usd
        from usdrt import Usd as RtUsd

        return RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())

    def _visual_paths(self):
        return [f"{self.prim_path}/bean_{i}/{self._visual_child}" for i in range(self.count)]

    def _view(self):
        """The grains' rigid body view, made once PhysX has them (grains spawned while the simulation runs are only
        parsed at its next step)."""
        if self.view is None:
            try:
                from isaacsim.core.simulation_manager import SimulationManager
                import torch
            except ImportError:
                return None
            sim_view = SimulationManager.get_physics_sim_view()
            if sim_view is None:
                return None
            view = sim_view.create_rigid_body_view(f"{self.prim_path}/bean_*")
            if view is None or getattr(view, "_backend", True) is None or view.count != self.count:
                return None
            self.view = view
            world = np.c_[self.local_pos + self.env_origin, self.local_quat]
            self.init_transforms = torch.tensor(world, dtype=torch.float32, device=view.get_transforms().device)
            self._indices = torch.arange(self.count, dtype=torch.int32, device=self.init_transforms.device)
        return self.view

    def get_particle_positions(self):
        """Grain centres, env-local (init_pos / init_ori are the identity)."""
        if self._view() is None:
            return self.local_pos.copy(), None, None
        return self.view.get_transforms()[:, :3].cpu().numpy().astype(float) - self.env_origin, None, None

    def apply_saved_pose(self):
        """All grains back to the pack's settled poses, at rest."""
        if self._view() is None:
            return
        import torch

        self.view.set_transforms(self.init_transforms, self._indices)
        zero = torch.zeros((self.count, 6), dtype=torch.float32, device=self.init_transforms.device)
        self.view.set_velocities(zero, self._indices)

    def destroy(self):
        from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

        self.view = None
        stage = self._fabric_stage()
        if stage is not None:  # Fabric keeps the prims sync_fabric_visuals put there after USD deletes them
            from usdrt import Sdf as RtSdf

            for path in self._visual_paths():
                stage.RemovePrim(RtSdf.Path(path))
        if is_prim_path_valid(self.prim_path):
            delete_prim(self.prim_path)
