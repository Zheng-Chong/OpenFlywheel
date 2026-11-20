"""Scene objects of the MuJoCo backend with the interface the layout, reward and eval code use on Isaac's objects
(env/scene_manager/objects/): env-local poses, saved-pose restore, offscreen relocation, bbox and mesh samples."""

import random

import mujoco
import numpy as np
import torch

from env.backends.mujoco.robots import COLLISION_GROUP
from env.scene_manager.layout_manager import LayoutManager

_FAR_CENTER = (100000.0, 100000.0, 100000.0)
_FAR_JITTER = 1000.0


def _far_pos():
    return [c + random.uniform(-_FAR_JITTER, _FAR_JITTER) for c in _FAR_CENTER]


def _origin_list(env_origin):
    if isinstance(env_origin, torch.Tensor):
        return env_origin.detach().cpu().numpy().tolist()
    return list(env_origin)


class MjSceneObject:
    """An item of one env's world (world.py) at `prim_path` (/World/envs/env_<i>/<type>/<category>/<instance>)."""

    def __init__(self, world, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale, template):
        self.world = world
        self._prim_path = self.usd_prim_path = self.prim_path = prim_path
        parts = prim_path.split("/")
        self.category_name, self.instance_name = parts[-2], parts[-1]
        self.name = self.instance_name
        self.usd_path = usd_path
        self.instance_config = inst_config
        self.visual_config = inst_config.get("visual", {}) or {}
        self.physics_config = inst_config.get("physics", {}) or {}
        self.env_origin = _origin_list(env_origin)
        self.default_pos = list(default_pos)
        self.default_ori = list(default_ori)
        self.scale = list(scale)
        self.item = f"{self.category_name}.{self.instance_name}"
        world.add(self.item, template, self.default_pos, self.default_ori)

    def initialize(self):
        pass

    def _get_object_transform(self, device=None):
        pos, quat = self.world.item_pose(self.item)
        device = device or "cpu"
        return (
            torch.tensor(pos, dtype=torch.float32, device=device),
            torch.tensor(quat, dtype=torch.float32, device=device),
            device,
        )

    def get_local_pose(self):
        return self.world.item_pose(self.item)

    def get_world_pose(self):
        pos, quat = self.world.item_pose(self.item)
        return pos + np.asarray(self.env_origin), quat

    def get_state(self, is_relative=False):
        pos, quat = self.get_local_pose() if is_relative else self.get_world_pose()
        return {"root_pose": torch.tensor(np.concatenate([pos, quat]), dtype=torch.float32)}

    def set_local_pose(self, translation=None, orientation=None):
        pos, quat = self.world.item_pose(self.item)
        self.world.set_item_pose(
            self.item,
            pos if translation is None else list(translation),
            quat if orientation is None else list(orientation),
        )

    def apply_saved_pose(self):
        self.set_local_pose(self.default_pos, self.default_ori)

    def relocate_offscreen(self):
        self.set_local_pose(_far_pos(), self.default_ori)

    def destroy(self):
        self.world.remove(self.item)

    def _visual_geoms(self):
        model = self.world.model
        root = self.world.root_body(self.item).id
        geoms = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root]
        meshes = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        visual = [g for g in meshes if model.geom_group[g] != COLLISION_GROUP]
        return visual or meshes or geoms

    def sample_mesh_vertices(self):
        """(points_world, points_local, pos_world, ori_world) of the first mesh, as Isaac's objects: local points are
        unscaled and in the object frame, world ones include the env origin."""
        model, data = self.world.model, self.world.data
        geoms = [g for g in self._visual_geoms() if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        if not geoms:
            return np.array([]), np.array([]), None, None
        g = geoms[0]
        mesh = model.geom_dataid[g]
        verts = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        points_world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        root = self.world.root_body(self.item).id
        rot = data.xmat[root].reshape(3, 3)
        points_body = (points_world - data.xpos[root]) @ rot
        points_local = points_body / np.asarray(self.scale, dtype=np.float64)
        origin = np.asarray(self.env_origin, dtype=np.float64)
        return (
            (points_world + origin).astype(np.float32),
            points_local.astype(np.float32),
            (data.xpos[root] + origin).astype(np.float32),
            data.xquat[root].astype(np.float32),
        )

    def get_bbox(self, is_relative=True):
        points_world, points_local, pos_world, ori_world = self.sample_mesh_vertices()
        if pos_world is None:
            return None, None, None
        pos = np.subtract(pos_world, self.env_origin) if is_relative else pos_world
        points_local = points_local * np.asarray(self.scale, dtype=np.float32)
        return pos, ori_world, np.concatenate([points_local.min(axis=0), points_local.max(axis=0)])


class MjRigidObject(MjSceneObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name, self.model_id = LayoutManager.decode_inst_name(self.instance_name)
        self.mass = min(self.physics_config.get("mass", 0.5), self.physics_config.get("mass_cap", 0.5))
        if self.mass <= 0:
            self.mass = 0.05


class MjDynamicObject(MjSceneObject):
    pass


class MjGeometryObject(MjSceneObject):
    pass


class MjArticulationObject(MjSceneObject):
    """An articulated asset; joints are named `<item>/<asset joint>` and read back by their asset name."""

    def initialize(self):
        if getattr(self, "initial_joint_positions", None) is None:
            self.dof_names = self._joints()[1]
            self.initial_joint_positions = self.get_joint_positions()
            lims = self._limits()
            self.lower_joint_positions, self.upper_joint_positions = lims[:, 0], lims[:, 1]

    def _joints(self):
        model = self.world.model
        prefix = f"{self.item}/"
        ids, names = [], []
        for j in range(model.njnt):
            name = model.joint(j).name
            if name.startswith(prefix) and model.jnt_type[j] in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                ids.append(j)
                names.append(name[len(prefix) :].rsplit("/", 1)[-1])
        return ids, names

    def _limits(self):
        model = self.world.model
        ids, _ = self._joints()
        lims = model.jnt_range[ids].copy() if ids else np.zeros((0, 2))
        for k, j in enumerate(ids):
            if not model.jnt_limited[j]:
                lims[k] = (-np.inf, np.inf)
        return lims

    def get_joint_positions(self):
        model, data = self.world.model, self.world.data
        ids, _ = self._joints()
        return np.array([data.qpos[model.jnt_qposadr[j]] for j in ids], dtype=np.float32)

    def set_joint_positions(self, positions):
        model, data = self.world.model, self.world.data
        positions = positions.cpu().numpy() if isinstance(positions, torch.Tensor) else np.asarray(positions)
        for j, q in zip(self._joints()[0], positions):
            data.qpos[model.jnt_qposadr[j]] = q
            data.qvel[model.jnt_dofadr[j]] = 0.0
        mujoco.mj_forward(model, data)

    get_current_joint_positions = get_joint_positions
    set_current_joint_positions = set_joint_positions

    def apply_saved_pose(self):
        self.initialize()
        self.set_joint_positions(self.initial_joint_positions)
        super().apply_saved_pose()

    def relocate_offscreen(self):
        self.initialize()
        self.set_joint_positions(self.initial_joint_positions)
        super().relocate_offscreen()

    def get_link_pose(self, link_name):
        model, data = self.world.model, self.world.data
        prefix = f"{self.item}/"
        for b in range(model.nbody):
            name = model.body(b).name
            if name.startswith(prefix) and name.rsplit("/", 1)[-1] == link_name:
                origin = np.asarray(self.env_origin)
                return np.concatenate([data.xpos[b] + origin, data.xquat[b]]).astype(np.float32)
        raise ValueError(f"Cannot find link '{link_name}' under articulation {self._prim_path}")

    def get_joint_info(self, joint_name="button_joint"):
        info = self.get_all_joints_info()
        if joint_name not in info:
            raise ValueError(f"Joint '{joint_name}' not found")
        return info[joint_name]

    def get_all_joints_info(self):
        self.initialize()
        positions = self.get_joint_positions()
        return {
            name: {
                "index": idx,
                "position": float(positions[idx]),
                "lower": float(self.lower_joint_positions[idx]),
                "upper": float(self.upper_joint_positions[idx]),
            }
            for idx, name in enumerate(self.dof_names)
        }


class MjTable:
    """Layout `Table`: a static textured box, moved offscreen on reload and back by apply_saved_poses (as Isaac)."""

    def __init__(self, world, item, template, instance_config):
        self.world, self.item = world, item
        self.position = list(instance_config.get("default_pos", [0.0, 0.0, 0.0]))
        self.orientation = list(instance_config.get("default_ori", [1.0, 0.0, 0.0, 0.0]))
        self.scale = list(instance_config.get("scale", [1.0, 1.0, 1.0]))
        world.add(item, template, self.position, self.orientation)

    def apply_saved_pose(self):
        self.world.set_item_pose(self.item, self.position, self.orientation)

    def relocate_offscreen(self):
        self.world.set_item_pose(self.item, _far_pos(), self.orientation)


def light_kwargs(light_type, config):
    """MuJoCo light for a layout Light, drawing the same random samples, in the same order, as Isaac's Light.reset."""
    from utils.rotations import euler_to_quat

    common = config["common"]
    pr, orr = common["initial_pos_range"], common["initial_ori_range"]
    pos = [random.uniform(pr[i], pr[i + 3]) for i in range(3)]
    euler = [random.uniform(orr[i], orr[i + 3]) for i in range(3)]
    color = np.ones(3)
    if "color_range" in common:
        lo, hi = common["color_range"]
        color = np.array([random.uniform(lo[i], hi[i]) for i in range(3)])
    if "color_temperature_range" in common:
        random.uniform(*common["color_temperature_range"])
    specific = config.get(light_type, {}) or {}
    intensity = random.uniform(*specific["intensity_range"]) if "intensity_range" in specific else 1000.0
    for key in {
        "Rect": ("width_range", "height_range"),
        "Distant": ("angle_range",),
        "Sphere": ("radius_range",),
        "Cylinder": ("radius_range", "length_range"),
        "Disk": ("radius_range",),
    }.get(light_type, ()):
        random.uniform(*specific[key])
    quat = np.asarray(euler_to_quat(euler), dtype=np.float64)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    direction = mat.reshape(3, 3) @ np.array([0.0, 0.0, -1.0])  # USD lights emit along their -Z
    level = float(np.clip(intensity / 5000.0, 0.1, 1.0))
    if light_type == "Dome":
        return {"ambient": list(0.4 * level * color)}
    kind = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if light_type == "Distant" else mujoco.mjtLightType.mjLIGHT_SPOT
    return {
        "type": kind,
        "pos": pos,
        "dir": list(direction),
        "diffuse": list(0.6 * level * color),
        "specular": [0.1, 0.1, 0.1],
        "cutoff": 80.0,
        "castshadow": False,
    }
