"""Complete G1-D with source RH56DFTP mimic constraints; no hardware drivers."""

import inspect
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files import from_files
from pxr import UsdPhysics

from env.global_configs import ROBOTS_PATH
from env.robot_manager.robot_config.g1d import (
    LINK_MATERIALS,
    MATERIALS,
    add_black_trim,
    get_robot_config as base_config,
)

NATIVE_CONTACT_LINKS = {
    side: [side + "_" + finger + "_2" for finger in ("little", "ring", "middle", "index")]
    + [side + ("_thumb_3" if side == "left" else "_thumb_4"), side + "_inspire_base_link"]
    for side in ("left", "right")
}
# Followers are held by rigid mimic constraints; their source limits equal the mimic image of the active range
# (1.5238 x 1.05 = 1.6), so at an active limit the follower limit, the mimic and the active limit all bind at once
# and a small kick makes the solve singular: positions reach 1e9 in one action (g1d_build_block_pattern tower, the
# right little finger pushed onto its limit while the arm decelerated, 11-06). Keep follower limits this far
# outside the image so they never bind.
FOLLOWER_LIMIT_MARGIN = 0.05


@sim_utils.clone
def spawn(prim_path, cfg, *args, **kwargs):
    prim = from_files.spawn_from_urdf.__wrapped__(prim_path, cfg, *args, **kwargs)
    source = ET.parse(cfg.asset_path).getroot()
    contract = json.loads((Path(cfg.asset_path).parent / "model_contract.json").read_text())
    hand_links = {n for h in contract["hands"].values() for n in h["links"]}
    for link in prim.GetChildren():
        collisions = link.GetChild("collisions")
        if collisions:
            collisions.SetInstanceable(False)
    # URDF conversion instances collision meshes. Apply the small-hand contact
    # offset only after making this private instance editable; otherwise the
    # generic spawn modifier reports no matching editable collision prims.
    sim_utils.modify_collision_properties(prim_path, cfg.collision_props)
    for link in source.findall("link"):
        name = link.get("name")
        visuals = prim.GetChild(name).GetChild("visuals")
        if not visuals:
            continue
        if name in hand_links:
            color = link.find("visual/material/color")
            rgb = tuple(map(float, color.get("rgba").split()[:3])) if color is not None else (0.6, 0.6, 0.6)
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.65)
        else:
            key = next((v for p, v in LINK_MATERIALS.items() if re.fullmatch(p, name)), None)
            if key is None:
                continue
            material = MATERIALS[key]
        path = prim_path + "/Looks/" + name
        material.func(path, material)
        visuals.SetInstanceable(False)
        sim_utils.bind_visual_material(str(visuals.GetPath()), path)
    add_black_trim(prim)
    for a, b in contract["collision_exclusions"]:
        UsdPhysics.FilteredPairsAPI.Apply(prim.GetChild(a)).CreateFilteredPairsRel().AddTarget(
            prim.GetChild(b).GetPath()
        )
    # Do not silently accept an importer which drops the six passive couplings.
    native = [
        p
        for p in prim.GetStage().Traverse()
        if str(p.GetPath()).startswith(prim_path + "/") and any("MimicJoint" in s for s in p.GetAppliedSchemas())
    ]
    if len(native) != 12:
        raise RuntimeError(f"RH56DFTP requires 12 native mimic constraints, found {len(native)}")
    for h in contract["hands"].values():
        for name in h["all_joints"]:
            joint = prim.GetStage().GetPrimAtPath(prim_path + "/joints/" + name)
            limits = h["effective_active_limits"].get(name, h["limits"][name])
            # The importer widens follower limits by 20%; use the recorded
            # source limits, with the explicit feasible active intersection.
            if name in h["mimic"]:
                m, src = h["mimic"][name], h["effective_active_limits"][h["mimic"][name]["joint"]]
                lo, hi = sorted(m["multiplier"] * src[k] + m["offset"] for k in ("lower", "upper"))
                limits = {"lower": lo - FOLLOWER_LIMIT_MARGIN, "upper": hi + FOLLOWER_LIMIT_MARGIN}
            joint.GetAttribute("physics:lowerLimit").Set(limits["lower"] * 180 / 3.141592653589793)
            joint.GetAttribute("physics:upperLimit").Set(limits["upper"] * 180 / 3.141592653589793)
            if name in h["mimic"]:
                axis = "rot" + joint.GetAttribute("physics:axis").Get()
                # Imported layers carry a weak 25/.005 compliance preset.
                # Author a strong zero override: removing the local property
                # would merely reveal that weak preset again. This runtime's
                # Python schema bindings lack convenience getters, so use the
                # authored attributes, verified with physical follower motion.
                joint.GetAttribute(f"physxMimicJoint:{axis}:naturalFrequency").Set(0.0)
                joint.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Set(0.0)
                joint.GetAttribute(f"physxMimicJoint:{axis}:referenceJointAxis").Set("rotZ")
    return prim


def get_robot_config():
    directory = Path(ROBOTS_PATH) / "g1d_inspire_rh56dftp"
    contract = json.loads((directory / "model_contract.json").read_text())
    config = base_config()
    config.spawn.func = spawn
    config.spawn.asset_path = str(directory / "robot.urdf")
    config.spawn.usd_dir = str(directory / "usd")
    # Isaac Lab 2.3.x passes this inverted-name flag straight to parse_mimic.
    # The installed importer tests require parse_mimic=True to create MimicAPI.
    # Accommodate both that version and a corrected converter, then assert the
    # resulting USD schemas in spawn instead of trusting a misleading flag.
    converter = inspect.getsource(UrdfConverter._get_urdf_import_config)
    config.spawn.convert_mimic_joints_to_normal_joints = (
        "set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)" in converter
    )
    config.spawn.collider_type = "convex_hull"
    config.spawn.rigid_props.disable_gravity = False
    config.spawn.rigid_props.max_depenetration_velocity = 0.2
    config.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)
    config.spawn.articulation_props.solver_position_iteration_count = 64
    config.spawn.articulation_props.solver_velocity_iteration_count = 8
    root = ET.parse(directory / "robot.urdf").getroot()
    arm = [
        j
        for j in root.findall("joint")
        if j.get("type") != "fixed"
        and any(re.fullmatch(p, j.get("name")) for p in config.actuators["arm"].joint_names_expr)
    ]
    config.actuators["arm"].effort_limit_sim = {j.get("name"): float(j.find("limit").get("effort")) for j in arm}
    config.actuators["arm"].velocity_limit_sim = {j.get("name"): float(j.find("limit").get("velocity")) for j in arm}
    config.init_state.joint_pos = {k: v for k, v in config.init_state.joint_pos.items() if "dex1" not in k}
    config.actuators.pop("gripper")
    for h in contract["hands"].values():
        config.init_state.joint_pos.update(h["initial_rad"])
    active = [j for h in contract["hands"].values() for j in h["joints"]]
    config.actuators["inspire_active"] = ImplicitActuatorCfg(
        joint_names_expr=active, stiffness=3.0, damping=0.15, effort_limit_sim=1.0, velocity_limit_sim=2.0
    )
    # Passive joints retain native PhysX mimic constraints; zero drive avoids
    # fighting those constraints with independent target positions.
    config.actuators["inspire_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[j for h in contract["hands"].values() for j in h["mimic"]],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=1.0,
        velocity_limit_sim=None,
    )
    return config
