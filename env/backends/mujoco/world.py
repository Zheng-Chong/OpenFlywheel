"""One MuJoCo model per env: its robots, then the scene manager's fixtures and objects (scene_manager.py).

Envs do not share a model because each loads its own layout. Positions inside a model are env-local; the IsaacLab-like
views (articulation.py) add the env origin, as IsaacLab reports world poses of cloned envs. The model is rebuilt from
templates whenever the scene changes (a reload spawns other objects); joints present before and after keep their state.
"""

import logging

import mujoco
import numpy as np

from env.backends.mujoco import robots

logger = logging.getLogger(__name__)
ROOT = "root"
CONTACT_SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]


def _log_warning(message):
    # Every attached template carries default options (gravity, ...) that equal the env's: MuJoCo reports them.
    level = logging.DEBUG if message.startswith("Attach conflict") else logging.WARNING
    logger.log(level, "MuJoCo: %s", message.strip())


mujoco.set_mju_user_warning(_log_warning)  # MuJoCo's default writes MUJOCO_LOG.TXT into the working directory


class EnvWorld:
    def __init__(self, env_id, timestep, gravity=(0.0, 0.0, -9.81), offscreen_size=(1280, 960)):
        self.env_id = env_id
        self.timestep = timestep
        self.gravity = list(gravity)
        self.offscreen_size = offscreen_size
        self.robots = {}  # prefix -> (template spec, pos, quat)
        self.drives = {}  # robot prefix -> RobotDrives
        self.items = {}  # name -> (template spec, pos, quat): scene objects and fixtures
        self.lights = []  # kwargs of spec.worldbody.add_light
        self.ambient = None  # headlight ambient (a layout Dome light)
        self.cameras = {}  # name -> {"body": mount body or None (env frame), "pos", "quat", "fovy"}
        self.spec = None
        self.model = None
        self.data = None
        self.version = 0  # bumped by every compile: views re-resolve their ids

    def add_robot(self, prefix, robot_spec, pos, quat):
        """Attach the robot at the env-local pose (pos, wxyz quat); its elements are named `<prefix><name>`."""
        self.robots[prefix] = (robots.build_robot(robot_spec), list(pos), list(quat))
        self.drives[prefix] = RobotDrives(prefix, robot_spec)

    def add(self, name, template, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        """A scene item: `template`'s body `root` placed at the env-local pose; elements are named `<name>/<element>`."""
        self.items[name] = (template, list(pos), list(quat))

    def remove(self, name):
        self.items.pop(name, None)

    def _new_spec(self):
        spec = mujoco.MjSpec()
        spec.modelname = f"env_{self.env_id}"
        spec.option.timestep = self.timestep
        spec.option.gravity = self.gravity
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # stiff PD drives (kp up to 1e6)
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.noslip_iterations = 3  # grasped objects do not creep out of the fingers
        spec.visual.global_.offwidth, spec.visual.global_.offheight = self.offscreen_size
        spec.visual.headlight.ambient = self.ambient or [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
        spec.visual.headlight.specular = [0.05, 0.05, 0.05]
        spec.visual.quality.shadowsize = 4096
        spec.visual.map.znear = 0.005  # wrist cameras see objects a few cm away
        spec.stat.extent = 2.0
        for light in self.lights:
            spec.worldbody.add_light(**light)
        return spec

    def compile(self):
        spec = self._new_spec()
        for prefix, (template, pos, quat) in self.robots.items():
            spec.attach(template.copy(), frame=spec.worldbody.add_frame(pos=pos, quat=quat), prefix=prefix)
        for name, (template, pos, quat) in self.items.items():
            spec.attach(template.copy(), frame=spec.worldbody.add_frame(pos=pos, quat=quat), prefix=f"{name}/")
        for name, cam in self.cameras.items():
            parent = spec.body(cam["body"]) if cam["body"] else spec.worldbody
            if parent is None:
                raise ValueError(f"camera {name}: mount body {cam['body']} is not in the model")
            parent.add_camera(name=name, pos=list(cam["pos"]), quat=list(cam["quat"]), fovy=cam["fovy"])
        model = spec.compile()
        # MuJoCo contacts are soft in proportion to the bodies' mass: with the defaults (0.02 s, d 0.95) a 55 N
        # gripper squeezes through an 8 g block. The stiffest stable contact (time constant 2 dt, d -> 0.999) stands
        # in for PhysX's rigid ones.
        model.geom_solref[:] = [2 * self.timestep, 1.0]
        model.geom_solimp[:] = CONTACT_SOLIMP
        data = mujoco.MjData(model)
        if self.model is not None:
            _transfer_state(self.model, self.data, model, data)
        self.spec, self.model, self.data = spec, model, data
        self.version += 1
        for drive in self.drives.values():
            drive.bind(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def root_body(self, name):
        return self.model.body(f"{name}/{ROOT}")

    def item_pose(self, name):
        body = self.root_body(name)
        return self.data.xpos[body.id].copy(), self.data.xquat[body.id].copy()

    def set_item_pose(self, name, pos, quat):
        """Move an item (a free root sets its joint, a static one its body pose) and zero its velocity."""
        body = self.root_body(name)
        if body.jntnum[0] and self.model.jnt_type[body.jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE:
            qadr, dadr = self.model.jnt_qposadr[body.jntadr[0]], self.model.jnt_dofadr[body.jntadr[0]]
            self.data.qpos[qadr : qadr + 7] = [*pos, *quat]
            self.data.qvel[dadr : dadr + 6] = 0.0
        else:
            self.model.body_pos[body.id] = pos
            self.model.body_quat[body.id] = quat
        mujoco.mj_forward(self.model, self.data)

    def set_camera_pose(self, name, pos, quat):
        """Camera pose in its mount frame (MuJoCo and USD cameras both look along -Z with +Y up)."""
        self.cameras[name].update(pos=list(pos), quat=list(quat))
        if self.model is not None:
            cam = self.model.camera(name).id
            self.model.cam_pos[cam] = pos
            self.model.cam_quat[cam] = quat
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_camlight(self.model, self.data)

    def step(self):
        for drive in self.drives.values():
            drive.apply(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)


def _transfer_state(old_model, old_data, model, data):
    data.time = old_data.time
    for j in range(model.njnt):
        name = model.joint(j).name
        try:
            old = old_model.joint(name)
        except KeyError:
            continue
        if old.type[0] != model.jnt_type[j]:
            continue
        nq = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(model.jnt_type[j], 1)
        nv = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(model.jnt_type[j], 1)
        qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
        data.qpos[qa : qa + nq] = old_data.qpos[old.qposadr[0] : old.qposadr[0] + nq]
        data.qvel[da : da + nv] = old_data.qvel[old.dofadr[0] : old.dofadr[0] + nv]


class RobotDrives:
    """PD targets of one robot's driven joints (robots.py); q_ref approaches q_target at most velocity_limit_sim fast."""

    def __init__(self, prefix, robot_spec):
        self.prefix = prefix
        self.robot_spec = robot_spec
        self.joint_names = None

    def bind(self, model, data):
        """Resolve qpos/dof/actuator addresses after a compile; targets survive recompiles."""
        first = self.joint_names is None
        if first:
            names = [model.joint(i).name for i in range(model.njnt)]
            self.joint_names = [n[len(self.prefix) :] for n in names if n.startswith(self.prefix)]
            params = robots.drives(self.robot_spec, self.joint_names)
            n = len(self.joint_names)
            self.kp, self.kd, self.vmax = np.zeros(n), np.zeros(n), np.full(n, np.inf)
            self.driven = np.zeros(n, dtype=bool)
            for i, name in enumerate(self.joint_names):
                if name in params:
                    kp, kd, _, vmax = params[name]
                    self.kp[i], self.kd[i], self.driven[i] = kp, kd, True
                    if vmax is not None:
                        self.vmax[i] = vmax
        joints = [model.joint(self.prefix + n) for n in self.joint_names]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.actuator = np.array(
            [model.actuator(self.prefix + n).id for n, d in zip(self.joint_names, self.driven) if d], dtype=int
        )
        # PhysX's implicit drive takes the spring at the next step's position, q + dt * qd': kp * dt more damping on
        # qd', which implicitfast integrates implicitly. An explicit spring diverges once sqrt(kp / inertia) * dt > 2
        # (the 36 kN/m fingers: 3.1).
        d = self.driven
        model.actuator_biasprm[self.actuator, 2] = -(self.kd[d] + self.kp[d] * model.opt.timestep)
        if first:
            self.q_target = data.qpos[self.qadr].copy()
            self.qd_target = np.zeros(len(self.joint_names))
            self.q_ref = self.q_target.copy()

    def teleport(self, idx, q):
        """write_joint_state_to_sim: the joints are at q, so their ramped references restart there."""
        self.q_ref[idx] = q

    def apply(self, model, data):
        dt = model.opt.timestep
        step = np.clip(self.q_target - self.q_ref, -self.vmax * dt, self.vmax * dt)
        self.q_ref = self.q_ref + step
        d = self.driven
        data.ctrl[self.actuator] = self.kp[d] * self.q_ref[d] + self.kd[d] * self.qd_target[d]
