"""The part of IsaacLab's Articulation / ContactSensor / InteractiveScene API that RobotManager and the task code use,
over the per-env MuJoCo models of world.py. Tensors are torch float32 on the CPU, (num_envs, ...) like IsaacLab's;
poses are [x, y, z, qw, qx, qy, qz] in the world frame (env origin added).
"""

import re

import mujoco
import numpy as np
import torch


def resolve_matching_names(keys, names, preserve_order=False):
    """isaaclab.utils.string.resolve_matching_names: (indices, names) of `names` fully matching the regexes in `keys`;
    in `names` order, or in `keys` order with preserve_order. Every key must match."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    matches = {k: [i for i, n in enumerate(names) if re.fullmatch(k, n)] for k in keys}
    missing = [k for k, ids in matches.items() if not ids]
    if missing:
        raise ValueError(f"no names match {missing}; available: {names}")
    if preserve_order:
        ids = [i for k in keys for i in matches[k]]
    else:
        ids = sorted({i for v in matches.values() for i in v})
    return ids, [names[i] for i in ids]


def _ids(ids, n):
    if ids is None or (isinstance(ids, slice) and ids == slice(None)):
        return list(range(n))
    if isinstance(ids, torch.Tensor):
        return ids.long().tolist()
    return [int(i) for i in ids]


def _np(x):
    return x.detach().cpu().numpy().astype(np.float64) if isinstance(x, torch.Tensor) else np.asarray(x, np.float64)


class MjArticulation:
    def __init__(self, worlds, origins, prefix, init_joint_pos):
        self.worlds = worlds
        self.origins = origins  # (num_envs, 3) numpy
        self.prefix = prefix
        self.num_instances = len(worlds)
        self._init_joint_pos = dict(init_joint_pos)
        self._version = None
        self.data = _ArticulationData(self)
        self.root_physx_view = _RootView(self)

    # ---- names and ids ------------------------------------------------------------------------------------------
    def _refresh(self):
        version = tuple(w.version for w in self.worlds)
        if version == self._version:
            return
        self._version = version
        m = self.worlds[0].model
        p = self.prefix
        self.joint_names = self.worlds[0].drives[p].joint_names
        bodies = [m.body(i).name for i in range(m.nbody)]
        self.body_names = [b[len(p) :] for b in bodies if b.startswith(p)]
        self._body_ids = [[w.model.body(p + b).id for b in self.body_names] for w in self.worlds]
        self._limits = np.array([m.joint(p + j).range for j in self.joint_names])

    @property
    def num_joints(self):
        self._refresh()
        return len(self.joint_names)

    @property
    def num_bodies(self):
        self._refresh()
        return len(self.body_names)

    def find_joints(self, name_keys, joint_subset=None, preserve_order=False):
        self._refresh()
        names = self.joint_names if joint_subset is None else joint_subset
        return resolve_matching_names(name_keys, names, preserve_order)

    def find_bodies(self, name_keys, preserve_order=False):
        self._refresh()
        return resolve_matching_names(name_keys, self.body_names, preserve_order)

    def default_joint_pos(self):
        self._refresh()
        q = np.zeros(len(self.joint_names))
        for pattern, value in self._init_joint_pos.items():
            for i, name in enumerate(self.joint_names):
                if re.fullmatch(pattern, name):
                    q[i] = value
        return np.clip(q, self._limits[:, 0], self._limits[:, 1])

    # ---- commands -----------------------------------------------------------------------------------------------
    def _write(self, attr, values, joint_ids, env_ids):
        self._refresh()
        joints = _ids(joint_ids, len(self.joint_names))
        values = _np(values).reshape(-1, len(joints))
        for row, env in enumerate(_ids(env_ids, self.num_instances)):
            drive = self.worlds[env].drives[self.prefix]
            getattr(drive, attr)[joints] = values[row if len(values) > 1 else 0]

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self._write("q_target", target, joint_ids, env_ids)

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        self._write("qd_target", target, joint_ids, env_ids)

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        self._refresh()
        joints = _ids(joint_ids, len(self.joint_names))
        position = _np(position).reshape(-1, len(joints))
        velocity = _np(velocity).reshape(-1, len(joints))
        for row, env in enumerate(_ids(env_ids, self.num_instances)):
            world = self.worlds[env]
            drive = world.drives[self.prefix]
            world.data.qpos[drive.qadr[joints]] = position[row]
            world.data.qvel[drive.dadr[joints]] = velocity[row]
            drive.teleport(joints, position[row])
            world.forward()

    def reset(self, env_ids=None):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


class _ArticulationData:
    def __init__(self, art: MjArticulation):
        self._art = art

    def _stack(self, fn):
        art = self._art
        art._refresh()
        return torch.tensor(
            np.stack([fn(w, w.drives[art.prefix], e) for e, w in enumerate(art.worlds)]), dtype=torch.float32
        )

    @property
    def joint_names(self):
        return self._art.joint_names

    @property
    def joint_pos(self):
        return self._stack(lambda w, d, e: w.data.qpos[d.qadr])

    @property
    def joint_vel(self):
        return self._stack(lambda w, d, e: w.data.qvel[d.dadr])

    @property
    def joint_pos_target(self):
        return self._stack(lambda w, d, e: d.q_target)

    @property
    def joint_vel_target(self):
        return self._stack(lambda w, d, e: d.qd_target)

    @property
    def default_joint_pos(self):
        q = self._art.default_joint_pos()
        return torch.tensor(np.tile(q, (self._art.num_instances, 1)), dtype=torch.float32)

    @property
    def default_joint_vel(self):
        return torch.zeros_like(self.default_joint_pos)

    @property
    def soft_joint_pos_limits(self):
        self._art._refresh()
        lim = np.tile(self._art._limits, (self._art.num_instances, 1, 1))
        return torch.tensor(lim, dtype=torch.float32)

    @property
    def body_link_pose_w(self):
        art = self._art

        def pose(w, d, e):
            ids = art._body_ids[e]
            return np.concatenate([w.data.xpos[ids] + art.origins[e], w.data.xquat[ids]], axis=-1)

        return self._stack(pose)

    @property
    def body_pos_w(self):
        return self.body_link_pose_w[..., :3]

    @property
    def body_quat_w(self):
        return self.body_link_pose_w[..., 3:]

    @property
    def body_state_w(self):
        pose = self.body_link_pose_w
        return torch.cat([pose, torch.zeros(*pose.shape[:-1], 6)], dim=-1)

    @property
    def root_pose_w(self):
        return self.body_link_pose_w[:, 0]

    @property
    def root_pos_w(self):
        return self.root_pose_w[:, :3]

    @property
    def root_quat_w(self):
        return self.root_pose_w[:, 3:]


class _RootView:
    """root_physx_view of a fixed-base articulation: get_jacobians() (num_envs, bodies - 1, 6, joints), the world-frame
    [linear; angular] Jacobian of each non-root link's origin over the joints, as PhysX reports it."""

    def __init__(self, art: MjArticulation):
        self._art = art

    def get_jacobians(self):
        art = self._art
        art._refresh()
        out = np.zeros((art.num_instances, len(art.body_names) - 1, 6, len(art.joint_names)))
        jacp, jacr = np.zeros((3, 0)), np.zeros((3, 0))
        for e, world in enumerate(art.worlds):
            m, d = world.model, world.data
            dadr = world.drives[art.prefix].dadr
            if jacp.shape[1] != m.nv:
                jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            for k, body in enumerate(art._body_ids[e][1:]):
                mujoco.mj_jac(m, d, jacp, jacr, d.xpos[body], body)
                out[e, k, :3] = jacp[:, dadr]
                out[e, k, 3:] = jacr[:, dadr]
        return torch.tensor(out, dtype=torch.float32)


class MjContactSensor:
    """ContactSensor over `body_names` of an articulation: data.net_forces_w (num_envs, bodies, 3), the summed contact
    force each body feels (normal and friction), N, world frame."""

    def __init__(self, articulation: MjArticulation, body_names):
        self.articulation = articulation
        self.body_names = list(body_names)
        self.data = self

    @property
    def net_forces_w(self):
        art = self.articulation
        out = np.zeros((art.num_instances, len(self.body_names), 3))
        force = np.zeros(6)
        for e, world in enumerate(art.worlds):
            m, d = world.model, world.data
            ids = {m.body(art.prefix + b).id: k for k, b in enumerate(self.body_names)}
            for c in range(d.ncon):
                contact = d.contact[c]
                b1, b2 = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
                if b1 not in ids and b2 not in ids:
                    continue
                mujoco.mj_contactForce(m, d, c, force)
                world_force = contact.frame.reshape(3, 3).T @ force[:3]  # on geom2, from geom1
                if b2 in ids:
                    out[e, ids[b2]] += world_force
                if b1 in ids:
                    out[e, ids[b1]] -= world_force
        return torch.tensor(out, dtype=torch.float32)


class MjScene:
    """InteractiveScene stand-in: scene["robot0"], scene.sensors, env_origins (num_envs, 3) and num_envs."""

    def __init__(self, env_origins):
        self.env_origins = torch.tensor(env_origins, dtype=torch.float32)
        self.num_envs = len(env_origins)
        self.articulations = {}
        self.sensors = {}

    def __getitem__(self, name):
        if name in self.articulations:
            return self.articulations[name]
        return self.sensors[name]

    def keys(self):
        return [*self.articulations, *self.sensors]

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


def grid_origins(num_envs, spacing):
    """IsaacLab's env grid (InteractiveScene, cloner): rows of ceil(sqrt(n)) envs, centred on the world origin."""
    rows = int(np.ceil(np.sqrt(num_envs)))
    cols = int(np.ceil(num_envs / rows))
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    x = -(ii.flatten()[:num_envs] - (rows - 1) / 2) * spacing
    y = (jj.flatten()[:num_envs] - (cols - 1) / 2) * spacing
    return np.stack([x, y, np.zeros(num_envs)], axis=1)
