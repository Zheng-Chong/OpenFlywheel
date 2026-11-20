"""Manufacturer hand dynamics on the existing complete G1-D articulation."""

import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import from_files
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

from env.global_configs import ROBOTS_PATH
from env.robot_manager.robot_config.g1d import (
    LINK_MATERIALS,
    MATERIALS,
    add_black_trim,
    get_robot_config as base_config,
)


@sim_utils.clone
def spawn(prim_path, cfg, *args, **kwargs):
    prim = from_files.spawn_from_urdf.__wrapped__(prim_path, cfg, *args, **kwargs)
    directory = Path(cfg.asset_path).parent
    contract = json.loads((directory / "model_contract.json").read_text())
    dynamics = json.loads((directory / "vendor_dynamics.json").read_text())
    stage = prim.GetStage()
    # STL carries no appearance; URDF import ignores some inline colours.
    # Restore the existing G1-D finish and manufacturer hand RGB explicitly.
    source = ET.parse(cfg.asset_path).getroot()
    hand_links = {name for data in dynamics.values() for name in data["rigid_bodies"]}
    materials = {}
    for link in source.findall("link"):
        name = link.get("name")
        visuals = prim.GetChild(name).GetChild("visuals")
        if not visuals:
            continue
        if name in hand_links:
            color = link.find("visual/material/color")
            if color is None:
                continue
            rgb = tuple(float(v) for v in color.get("rgba").split()[:3])
            key = "hand_" + str(tuple(round(v, 3) for v in rgb)).replace(" ", "").replace(".", "_").replace(
                ",", "_"
            ).replace("(", "").replace(")", "")
            material_cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=rgb, metallic=0.65 if min(rgb) > 0.5 else 0.0, roughness=0.55
            )
        else:
            key = next((m for pattern, m in LINK_MATERIALS.items() if re.fullmatch(pattern, name)), None)
            if key is None:
                continue
            material_cfg = MATERIALS[key]
        path = prim_path + "/Looks/" + key
        if key not in materials:
            material_cfg.func(path, material_cfg)
            materials[key] = path
        visuals.SetInstanceable(False)
        sim_utils.bind_visual_material(str(visuals.GetPath()), materials[key])
    add_black_trim(prim)
    for first, second in contract["collision_exclusions"]:
        a, b = prim.GetChild(first), prim.GetChild(second)
        if not a or not b:
            raise ValueError(f"Unknown collision pair {first}, {second}")
        UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(b.GetPath())
    for data in dynamics.values():
        # Preserve vendor USD numerical masses for geometry-free query frames.
        # Otherwise URDF import can assign its 1 kg fallback to a massless tip.
        for name, attrs in data["rigid_bodies"].items():
            if not name.endswith(("_tip", "_fingertip", "_wrist_imu_frame")):
                continue
            body = prim.GetChild(name)
            if not body:
                raise ValueError(f"Missing query frame: {name}")
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateMassAttr(float(attrs["physics:mass"]))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*ast.literal_eval(attrs["physics:diagonalInertia"])))
        for name, attrs in data["joints"].items():
            if not attrs:
                continue  # Source standalone world fixation is absent on the mounted hand.
            joint = stage.GetPrimAtPath(prim_path + "/joints/" + name)
            if not joint:
                raise ValueError(f"Missing manufacturer joint: {name}")
            PhysxSchema.PhysxJointAPI.Apply(joint)
            for key, value in attrs.items():
                # Limits come from the original URDF; default drive targets
                # come from the explicit startup pose, not a saved GUI pose.
                if not key.startswith(("drive:", "physxJoint:")) or key.endswith(("targetPosition", "targetVelocity")):
                    continue
                attr = joint.GetAttribute(key)
                if not attr:
                    attr = joint.CreateAttribute(
                        key, Sdf.ValueTypeNames.Token if isinstance(value, str) else Sdf.ValueTypeNames.Float
                    )
                attr.Set(value)
    return prim


def get_config(model):
    directory = Path(ROBOTS_PATH) / model
    contract = json.loads((directory / "model_contract.json").read_text())
    config = base_config()
    config.spawn.func = spawn
    config.spawn.asset_path = str(directory / "robot.urdf")
    config.spawn.usd_dir = str(directory / "usd")
    config.spawn.collider_type = "convex_hull"
    config.spawn.rigid_props.disable_gravity = False
    source = ET.parse(directory / "robot.urdf").getroot()
    # Preserve G1-D control gains, while respecting each URDF arm motor's
    # individual torque/velocity limit (in particular the wrist limits).
    arm_joints = [
        j
        for j in source.findall("joint")
        if j.get("type") != "fixed"
        and any(re.fullmatch(pattern, j.get("name")) for pattern in config.actuators["arm"].joint_names_expr)
    ]
    config.actuators["arm"].effort_limit_sim = {j.get("name"): float(j.find("limit").get("effort")) for j in arm_joints}
    config.actuators["arm"].velocity_limit_sim = {
        j.get("name"): float(j.find("limit").get("velocity")) for j in arm_joints
    }
    config.init_state.joint_pos = {k: v for k, v in config.init_state.joint_pos.items() if "dex1" not in k}
    for hand in contract["hands"].values():
        config.init_state.joint_pos.update({j: 0.0 for j in hand["joints"]})
    config.actuators.pop("gripper")
    actuator = IdealPDActuatorCfg if "sharpa" in model else ImplicitActuatorCfg
    hand_limits = {j: values for hand in contract["hands"].values() for j, values in hand["limits"].items()}
    config.actuators["dexhand"] = actuator(
        joint_names_expr=[j for h in contract["hands"].values() for j in h["joints"]],
        stiffness=None,
        damping=None,
        effort_limit=None,
        # Manufacturer USD is a display/hold preset (Wuji: 2 Nm throughout).
        # URDF contains the per-motor nominal limits; preserve its safer caps.
        effort_limit_sim={j: v["effort"] for j, v in hand_limits.items()},
        velocity_limit=None,
        velocity_limit_sim={j: v["velocity"] for j, v in hand_limits.items()},
    )
    return config
