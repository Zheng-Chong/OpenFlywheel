"""Robot models for the MuJoCo backend: the robot's URDF and RobotSpec (env/robot_manager/robot_spec.py) as an MjSpec.

Drives follow IsaacLab's implicit actuators: per joint a PD force kp * (q_target - q) + kd * (qd_target - qd), clipped
to effort_limit_sim, as a MuJoCo affine actuator whose control is kp * q_ref + kd * qd_target (world.py sets it every
physics step). PhysX caps the joint speed at velocity_limit_sim; here q_ref approaches the target at most that fast,
which moves an unloaded joint the same way and still squeezes with full force once blocked.
"""

from functools import partial
import os
import re
import xml.etree.ElementTree as ET

import mujoco

from env.robot_manager.robot_config import g1d_spec

# robot_name in env_cfg/robot/*.yml -> RobotSpec factory. Robots missing here only run on the Isaac Sim backends.
ROBOT_SPECS = {
    "g1d": partial(g1d_spec.get_robot_spec, mobile=False),
    "g1d_mobile": partial(g1d_spec.get_robot_spec, mobile=True),
}

VISUAL_GROUP = 1  # rendered
COLLISION_GROUP = 3  # not rendered (mujoco.Renderer draws groups 0-2)


def get_robot_spec(robot_name):
    if robot_name not in ROBOT_SPECS:
        raise NotImplementedError(
            f"robot {robot_name!r} has no MuJoCo model; supported: {sorted(ROBOT_SPECS)} (env/backends/mujoco/robots.py)"
        )
    return ROBOT_SPECS[robot_name]()


def _urdf_xml(robot_spec):
    """The URDF with absolute mesh paths, without the hidden visuals and geom names (duplicated across links), and
    MuJoCo compiler options that keep fixed links (camera and EE frames) and visual meshes."""
    root = ET.parse(robot_spec.urdf_path).getroot()
    base = os.path.dirname(os.path.abspath(robot_spec.urdf_path))
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith("package://"):
            filename = filename.split("/", 3)[-1]
        mesh.set("filename", filename if os.path.isabs(filename) else os.path.join(base, filename))
    for link in root.iter("link"):
        for visual in list(link.findall("visual")):
            if any(re.fullmatch(p, visual.get("name", "")) for p in robot_spec.hidden_visuals):
                link.remove(visual)
        for element in (*link.findall("visual"), *link.findall("collision")):
            element.attrib.pop("name", None)
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = {"fusestatic": "false", "discardvisual": "false", "strippath": "false", "balanceinertia": "true"}
    ET.SubElement(extension, "compiler", compiler)
    return ET.tostring(root, encoding="unicode")


def _matches(patterns, name):
    return any(re.fullmatch(p, name) for p in patterns)


def drives(robot_spec, joint_names):
    """{joint: (kp, kd, effort limit or None, velocity limit or None)} of the joints an actuator group covers; groups
    later in the spec win, as IsaacLab applies them in order."""
    result = {}
    for group in robot_spec.actuators.values():
        for joint in joint_names:
            if _matches(group.joint_names_expr, joint):
                result[joint] = (group.stiffness, group.damping, group.effort_limit_sim, group.velocity_limit_sim)
    return result


def build_robot(robot_spec):
    """MjSpec of the robot: geoms split into visual and collision groups, one affine actuator per driven joint (named
    after its joint), armature, gravity compensation when the spec disables gravity, filtered pairs and finishes."""
    spec = mujoco.MjSpec.from_string(_urdf_xml(robot_spec))
    joint_names = [j.name for j in spec.joints]
    for group in robot_spec.actuators.values():
        if group.armature:
            for joint in spec.joints:
                if _matches(group.joint_names_expr, joint.name):
                    joint.armature = group.armature
    for joint, (kp, kd, effort, _) in drives(robot_spec, joint_names).items():
        actuator = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.gainprm[0] = 1.0
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[:3] = [0.0, -kp, -kd]
        if effort is not None:
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = [-effort, effort]
    for body in spec.bodies:
        if robot_spec.disable_gravity and body.name != "world":
            body.gravcomp = 1.0
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = VISUAL_GROUP
            finish = next((f for p, f in robot_spec.link_finishes.items() if re.fullmatch(p, geom.parent.name)), None)
            if finish is not None:
                geom.rgba[:3] = finish["diffuse_color"]
        else:
            geom.group = COLLISION_GROUP
            if not robot_spec.self_collisions:
                geom.contype, geom.conaffinity = 2, 1  # robot geoms collide with the scene (1), not each other (2)
    for a, b in robot_spec.filtered_pairs:
        spec.add_exclude(bodyname1=a, bodyname2=b)
    return spec
