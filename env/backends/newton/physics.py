"""Newton (IsaacLab 3, MuJoCo-Warp solver) under the Isaac Sim scene code.

The tasks author their scene in USD and reload its objects every episode, after the simulation started. PhysX parses
the stage as it changes; a Newton model is built once from an import of the stage. So the model is re-imported after
every scene reload (`rebuild`), and Isaac Sim core's rigid prims, which read and write bodies through a PhysX tensor
view, get a stand-in for that view on Newton's state (`install_core_sim_view`).
"""

import fnmatch
import re

import numpy as np
import torch
import warp as wp

_scene = None
_model_current = False


def replicate(scene):
    """Import env 0 of `scene` into Newton's model builder for every env, plus the prims outside the envs."""
    global _scene
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.cloner import newton_physics_replicate
    from isaaclab_newton.physics import NewtonBuilderCfg, NewtonManager

    _scene = scene
    sim = SimulationContext.instance()
    builder_cfg = NewtonBuilderCfg(physics_cfg=sim.cfg.physics)
    later = not hasattr(NewtonManager, "get_model")  # IsaacLab after 3.0: the manager keeps its builder
    if later:  # and the scene's own (empty) clone already opened a world in it
        sim.close_backend(sim.get_or_create_backend(builder_cfg))
    num_envs, source = scene.num_envs, scene.env_prim_paths[0]
    template = source.rsplit("_", 1)[0] + "_{}"
    envs_root = source.rsplit("/", 1)[0]
    world = scene.stage.GetPrimAtPath(envs_root).GetParent()
    global_paths = tuple(str(p.GetPath()) for p in world.GetChildren() if str(p.GetPath()) != envs_root)
    positions = np.asarray(scene.env_origins.cpu(), dtype=np.float32)
    mapping = np.ones((1, num_envs), dtype=bool)
    _author_contacts(scene.stage, (source, *global_paths))
    newton_physics_replicate(
        scene.stage, [source], [template], np.arange(num_envs), mapping, positions=positions, global_paths=global_paths
    )
    if later and getattr(sim.backend_manager, "builder", None) is not None:
        sim.backend_manager.builder = sim.get_or_create_backend(builder_cfg)


def _author_contacts(stage, roots):
    """The mujoco backend's contact (env/backends/mujoco/world.py) as MuJoCo-Warp shape attributes: stiff `solimp` on
    every collider and torsional friction (condim 4) on the scene's, without which the pads let a pinched 8 g block
    spin and squeeze out."""
    from pxr import Sdf, Usd, UsdPhysics

    from env.backends.mujoco.world import CONTACT_SOLIMP

    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.FloatArray).Set(CONTACT_SOLIMP)
            if "/robot" not in str(prim.GetPath()):
                prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(4)


def rebuild():
    """Re-import the stage and re-finalize the model; IsaacLab's assets rebind to it on PHYSICS_READY."""
    global _model_current
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics import NewtonBuilderCfg

    sim = SimulationContext.instance()
    sim.close_backend(sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics)))
    replicate(_scene)
    sim.physics_manager.reset(soft=False)
    _model_current = True


def _manager():
    """IsaacLab 3.0's NewtonManager is a class with class methods; later IsaacLab's is the simulation's instance."""
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics import NewtonManager

    return NewtonManager if hasattr(NewtonManager, "get_model") else SimulationContext.instance().backend_manager


def _model_state():
    manager = _manager()
    if hasattr(manager, "get_model"):
        return manager, manager.get_model(), manager.get_state_0()
    return manager, manager.backend.model, manager.backend.state_0


def _rotate(q, v):
    """Rotate v by the xyzw quaternions q (broadcast over leading dims)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def link_jacobians(view):
    """PhysX's get_jacobians() for a fixed-base Newton articulation view: (envs, links - 1, 6, dofs), rows
    [linear; angular] of each link's origin in world. Newton's eval_jacobian refers the linear rows to the centres
    of mass."""
    _, model, state = _model_state()
    ids = wp.to_torch(view.articulation_ids).reshape(-1).long()
    n, links, dofs = len(ids), view.link_count, view.joint_dof_count
    jac = wp.to_torch(view.eval_jacobian(state))[ids, : links * 6, :dofs].reshape(n, links, 6, dofs)
    pose = wp.to_torch(view.get_link_transforms(state)).reshape(n, links, 7)
    com = wp.to_torch(view.get_attribute("body_com", model)).reshape(n, links, 3)
    origin_from_com = -_rotate(pose[..., 3:], com)
    ang = jac[:, :, 3:]
    lin = jac[:, :, :3] + torch.cross(ang, origin_from_com[..., None].expand_as(ang), dim=2)
    return torch.cat([lin, ang], dim=2)[:, 1:]


class RigidBodyView:
    """The part of omni.physics.tensors' RigidBodyView that isaacsim.core's RigidPrim uses, for free Newton bodies.

    Newton imports a free rigid body as a one-body articulation with a free joint; its pose and twist are that
    joint's coordinates. The view re-resolves its bodies whenever the model is rebuilt.
    """

    def __init__(self, patterns):
        self._regex = re.compile("|".join(f"(?:{fnmatch.translate(p)})" for p in patterns))
        self._model = None
        self._gravity_off = None

    def _articulations(self):
        from newton.selection import ArticulationView

        _, model, _ = _model_state()
        if self._model is not model:
            self._view = ArticulationView(model, self._regex, verbose=False)
            self._model = model
        return self._view

    @property
    def count(self):
        return self._articulations().count

    @property
    def max_shapes(self):
        return max(1, int(getattr(self._articulations(), "shape_count", 1)))

    def check(self):
        return True

    def _root(self, getter, width):
        return wp.to_torch(getter(_model_state()[2])).reshape(-1, width).cpu()

    def _write(self, getter, setter, dtype, width, data, indices):
        manager, _, state = _model_state()
        current = getter(state)
        values = wp.to_torch(current).reshape(-1, width).clone()
        index = torch.as_tensor(indices, device=values.device).long().reshape(-1)
        values[index] = torch.as_tensor(data, device=values.device, dtype=values.dtype).reshape(-1, width)[index]
        setter(state, wp.from_torch(values.reshape(*current.shape, width), dtype=dtype))
        manager.invalidate_fk()

    def get_transforms(self):
        return self._root(self._articulations().get_root_transforms, 7)

    def set_transforms(self, data, indices):
        view = self._articulations()
        self._write(view.get_root_transforms, view.set_root_transforms, wp.transform, 7, data, indices)

    def get_velocities(self):
        return self._root(self._articulations().get_root_velocities, 6)

    def set_velocities(self, data, indices):
        view = self._articulations()
        self._write(view.get_root_velocities, view.set_root_velocities, wp.spatial_vector, 6, data, indices)

    def get_masses(self):
        view = self._articulations()
        return wp.to_torch(view.get_attribute("body_mass", self._model)).reshape(view.count, -1)[:, :1].cpu()

    def set_masses(self, data, indices):
        from newton import ModelFlags

        manager, model, _ = _model_state()
        view = self._articulations()
        mass = wp.to_torch(view.get_attribute("body_mass", model))
        flat = mass.reshape(view.count, -1)
        index = torch.as_tensor(indices).long().reshape(-1)
        new = torch.as_tensor(data, dtype=flat.dtype).reshape(-1)[index].to(flat.device)
        scale = new / flat[index, 0].clamp_min(1e-9)
        flat[index, 0] = new
        view.set_attribute("body_mass", model, wp.from_torch(flat.reshape(mass.shape)))
        inv_mass = wp.to_torch(view.get_attribute("body_inv_mass", model)).reshape(view.count, -1)
        inv_mass[index, 0] = 1.0 / new
        view.set_attribute("body_inv_mass", model, wp.from_torch(inv_mass.reshape(mass.shape)))
        for name, power in (("body_inertia", 1.0), ("body_inv_inertia", -1.0)):
            inertia = wp.to_torch(view.get_attribute(name, model))
            rows = inertia.reshape(view.count, -1, 3, 3)
            rows[index, 0] *= (scale**power).reshape(-1, 1, 1)
            view.set_attribute(name, model, wp.from_torch(rows.reshape(inertia.shape), dtype=wp.mat33))
        manager.add_model_change(ModelFlags.BODY_INERTIAL_PROPERTIES)

    def get_disable_gravities(self):
        if self._gravity_off is None:
            self._gravity_off = torch.zeros(self.count, dtype=torch.uint8)
        return self._gravity_off

    def set_disable_gravities(self, data, indices):
        # Newton has no per-body gravity switch; objects being retired are moved off-scene afterwards anyway.
        self._gravity_off = torch.as_tensor(data, dtype=torch.uint8).clone()

    def get_disable_simulations(self):
        return torch.zeros(self.count, dtype=torch.uint8)

    def set_disable_simulations(self, data, indices):
        pass


class SimulationView:
    """Stand-in for omni.physics.tensors' SimulationView: only what isaacsim.core's rigid prims ask for."""

    is_valid = True

    def create_rigid_body_view(self, patterns):
        return RigidBodyView([patterns] if isinstance(patterns, str) else list(patterns))


def install_core_sim_view(device):
    """Between begin_reload and rebuild there is no view: the prims then author their poses in USD, which the
    rebuild imports, instead of looking for bodies the model does not have yet. The core prims stay on the CPU
    whatever Newton's `device`: the scene and reward code reads their poses as numpy arrays."""
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.set_backend("torch")
    SimulationManager.set_device("cpu")
    view = SimulationView()
    SimulationManager.get_physics_sim_view = staticmethod(lambda: view if _model_current else None)


def limit_joint_speeds():
    """MuJoCo has no joint speed limit, so Newton drops `velocity_limit_sim` (PhysX caps the joint speed with it). As
    in the mujoco backend (env/backends/mujoco/robots.py), the position target an implicit drive receives moves
    toward the commanded one at most `velocity_limit_sim` per second: an unloaded joint moves as under PhysX, and the
    36 kN/m gripper closes at its 6.4 cm/s instead of slamming a light object out from between the fingers."""
    from isaaclab.actuators.actuator_collection import ActuatorCollection
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.assets import Articulation

    from env.environment.isaaclab_compat import joint_velocity_limit

    submit, reset = ActuatorCollection.submit_commands, ActuatorCollection.reset

    def limits(collection, like):
        articulation = collection._control._articulation
        limit = torch.full_like(like, float("inf"))
        for name, cfg in articulation.cfg.actuators.items():
            speed = joint_velocity_limit(cfg)
            if isinstance(speed, dict):
                for expr, value in speed.items():
                    limit[:, articulation.find_joints(expr)[0]] = float(value)
            elif speed is not None:
                limit[:, collection._group_joint_indices[name]] = float(speed)
        return limit

    def submit_commands(self):
        native = self._control.native_actuator_path_active  # Newton runs the drives on the raw targets
        target = wp.to_torch(self._joint_pos_target if native else self._joint_pos_target_sim)
        ref = getattr(self, "_speed_limited_ref", None)
        if ref is None:
            ref = self._control.joint_pos.torch.clone()
            self._speed_limits = limits(self, target)
        step = self._speed_limits * SimulationContext.instance().get_physics_dt()
        ref = torch.minimum(torch.maximum(target, ref - step), ref + step)
        commanded = target.clone()
        target.copy_(ref)
        submit(self)
        target.copy_(commanded)
        self._speed_limited_ref = ref

    def reset_ref(self, *args, **kwargs):
        self._speed_limited_ref = None
        return reset(self, *args, **kwargs)

    ActuatorCollection.submit_commands = submit_commands
    ActuatorCollection.reset = reset_ref
    for name in [
        n for n in dir(Articulation) if n.startswith(("write_joint_state_to_sim", "write_joint_position_to_sim"))
    ]:
        write = getattr(Articulation, name)

        def write_and_reset(self, *args, _write=write, **kwargs):
            if isinstance(getattr(self, "actuators", None), ActuatorCollection):
                self.actuators._speed_limited_ref = None
            return _write(self, *args, **kwargs)

        setattr(Articulation, name, write_and_reset)


def begin_reload():
    global _model_current
    _model_current = False
