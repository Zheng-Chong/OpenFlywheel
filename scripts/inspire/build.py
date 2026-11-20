"""Assemble the attached RH56DFTP pair on G1-D; simulation only, no SDK."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from scripts.dexhands.kinematics import fk, rotation

ROOT = Path(__file__).resolve().parents[2]
MODEL = "g1d_inspire_rh56dftp"
FINGERS = ("little", "ring", "middle", "index")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"
    robot = ET.parse(base).getroot()
    robot.set("name", MODEL)
    inputs = {str(base.relative_to(ROOT)): digest(base)}
    for e in list(robot):
        if "dex1" in e.get("name", "").lower():
            robot.remove(e)
    for j in robot.findall("joint"):
        if j.get("name").endswith("Wheel_Joint"):
            j.set("type", "fixed")

    def meshes(tree, urdf, group):
        for node in tree.iter("mesh"):
            name = node.get("filename")
            original = (
                urdf.parent.parent / "meshes" / Path(name).name if name.startswith("package://") else urdf.parent / name
            )
            sha = digest(original)
            relative = Path("meshes") / group / ("m_" + sha[:12] + "_" + original.name)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            node.set("filename", relative.as_posix())
            inputs[
                ("attachment/" + str(original.relative_to(source)))
                if original.is_relative_to(source)
                else str(original.relative_to(ROOT))
            ] = sha

    meshes(robot, base, "g1d")
    contract = {
        "model": MODEL,
        "joint_unit": "rad",
        "length_unit": "m",
        "hands": {},
        "source_archive_sha256": "fd4eb08c419f5754df87e28d160044670a07bb0338f4dc6f18415fdec5a671ab",
        "source_selection": (
            "matching left.xacro and urdf_right.xacro with their own meshes; display world mounts removed"
        ),
        "physical_calibration": "pending_real_measurements",
        "mount_status": "CAD_flange_center_aligned_simulation_candidate_G1D_bolt_pattern_and_TCP_unmeasured",
        "mount_orientation_contract": "source_+Z_to_wrist_+X_thumb_side_up_positive_flexion_inward",
        "collision_exclusions": [],
        "tactile": {},
        "forbidden_runtime_interfaces": ["DDS", "real_camera", "hardware_SDK"],
        "simulation_dynamics": {
            "stiffness_Nm_per_rad": 3.0,
            "damping_Nms_per_rad": 0.15,
            "effort_cap_Nm": 1.0,
            "active_velocity_cap_rad_per_s": 2.0,
            "active_target_lead_rad": 0.05,
            "physics_dt_s": 1 / 480,
            "solver_position_velocity_iterations": [64, 8],
            "max_depenetration_velocity_m_s": 0.2,
            "collision_contact_offset_m": 0.0005,
            "mimic_constraint": "native_rigid_zero_compliance_strong_layer_override",
            "status": "conservative_simulation_preset_not_measured_motor_response",
        },
        "optional_body_commands": {"waist_joint_state": 2, "lift_joint_state": 1},
    }
    sources = {"left": next(source.rglob("left.xacro")), "right": next(source.rglob("urdf_right.xacro"))}
    for side, path in sources.items():
        hand = ET.parse(path).getroot()
        inputs["attachment/" + str(path.relative_to(source))] = digest(path)
        for e in list(hand):
            if (
                e.tag == "link"
                and e.get("name") == "world"
                or e.tag == "joint"
                and e.find("parent").get("link") == "world"
            ):
                hand.remove(e)
        renamed = side + "_inspire_base_link"
        hand.find("link[@name='base_link']").set("name", renamed)
        for node in hand.iter():
            if node.get("link") == "base_link":
                node.set("link", renamed)
        meshes(hand, path, side)
        joints = [j for j in hand.findall("joint") if j.get("type") != "fixed"]
        limits = {j.get("name"): {k: float(v) for k, v in j.find("limit").attrib.items()} for j in joints}
        mimics = {
            j.get("name"): {
                "joint": j.find("mimic").get("joint"),
                "multiplier": float(j.find("mimic").get("multiplier", "1")),
                "offset": float(j.find("mimic").get("offset", "0")),
            }
            for j in joints
            if j.find("mimic") is not None
        }
        # Public active order matches ANGLE_SET register order; slave DOFs are never actions.
        active = [side + "_" + f + "_1_joint" for f in FINGERS] + (
            ["left_thumb_1_joint", "left_thumb_swing_joint"]
            if side == "left"
            else ["right_thumb_2_joint", "right_thumb_1_joint"]
        )
        effective = copy.deepcopy({n: limits[n] for n in active})
        for child, m in mimics.items():
            n, gain, offset = m["joint"], m["multiplier"], m["offset"]
            assert n in active and gain > 0
            effective[n]["lower"] = max(effective[n]["lower"], (limits[child]["lower"] - offset) / gain)
            effective[n]["upper"] = min(effective[n]["upper"], (limits[child]["upper"] - offset) / gain)
        initial = {n: effective[n]["lower"] for n in active}
        initial.update({n: initial[m["joint"]] * m["multiplier"] + m["offset"] for n, m in mimics.items()})
        for j in joints:
            n = j.get("name")
            if n in effective:
                j.find("limit").set("lower", str(effective[n]["lower"]))
                j.find("limit").set("upper", str(effective[n]["upper"]))
        # Mating face derived from the bottom planar vertices of the supplied base STL.
        import trimesh

        b = hand.find(f"link[@name='{renamed}']/visual/geometry/mesh")
        mesh = trimesh.load(output / b.get("filename"), force="mesh", process=False)
        bottom = mesh.vertices[mesh.vertices[:, 2] < mesh.bounds[0, 2] + 1e-5]
        anchor = (bottom.min(axis=0) + bottom.max(axis=0)) / 2
        anchor[2] = mesh.bounds[0, 2]
        r = rotation([0, 1, 0], np.pi / 2)
        target = np.array([0.0415, 0.003 if side == "left" else -0.003, 0.0])
        xyz = target - r @ anchor
        for e in hand:
            if e.tag in ("link", "joint", "material"):
                robot.append(copy.deepcopy(e))
        j = ET.SubElement(robot, "joint", name=side + "_inspire_mount", type="fixed")
        ET.SubElement(j, "parent", link=side + "_wrist_yaw_link")
        ET.SubElement(j, "child", link=renamed)
        ET.SubElement(j, "origin", xyz=" ".join(map(str, xyz)), rpy=f"0 {np.pi / 2} 0")
        contract["hands"][side] = {
            "root": renamed,
            "joints": active,
            "all_joints": [j.get("name") for j in joints],
            "links": [link.get("name") for link in hand.findall("link")],
            "limits": limits,
            "effective_active_limits": effective,
            "mimic": mimics,
            "initial_rad": initial,
            "cad_flange_anchor_m": anchor.tolist(),
            "mount_xyz_m": xyz.tolist(),
            "mount_rpy_rad": [0, float(np.pi / 2), 0],
            "configured_mass_kg": sum(float(link.find("inertial/mass").get("value")) for link in hand.findall("link")),
            "source": "attachment/" + str(path.relative_to(source)),
            "limit_correction": (
                "four active finger upper limits intersected with 1.05 slave limit: "
                "1.6 / 1.05 rad; source limits retained in this contract"
            ),
        }
        contract["tactile"][side] = {
            "piezoresistive": {
                "patches": 17,
                "taxels": 1062,
                "bytes": 2124,
                "raw_unit": "ADC_count_0_to_4096_not_Newtons",
                "addresses": [3000, 5123],
            },
            "capacitive": {"fingertip_modules": 5, "bytes_per_module": 58, "physical_float_units": "not_confirmed"},
            "simulation": "native_filtered_link_contact_force_N_not_hardware_taxels",
            "taxel_geometry_and_pressure_calibration": "not_supplied_not_invented",
        }
        # Adjacent-link collision suppression is PhysX's standard joint topology behavior only.
        camera = yaml.safe_load((ROOT / "scripts/g1d/left_robot_config.yml").read_text())["camera"]
        cfg = {
            "urdf_path": "./robot.urdf",
            "ee_joints": side + "_wrist_yaw_joint",
            "ee_link": renamed,
            "ee_type": "hand",
            "base_link": "torso_link",
            "arm_joints_name": [
                side + "_" + n
                for n in (
                    "shoulder_pitch_joint",
                    "shoulder_roll_joint",
                    "shoulder_yaw_joint",
                    "elbow_joint",
                    "wrist_roll_joint",
                    "wrist_pitch_joint",
                    "wrist_yaw_joint",
                )
            ],
            "gripper_joints_name": active,
            "save_gripper_joints_name": active,
            "gripper_move": {},
            "gripper_bias": 0.0,
            "gripper_scale": [0.0, 1.0],
            "camera": [c for c in camera if c["link"] == "head_link"] if side == "left" else [],
        }
        (output / (side + "_robot_config.yml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert len(fk(robot)) == len(robot.findall("link"))
    contract["action_dimension"] = 26
    contract["collision_exclusions"] = [
        ["left_inspire_base_link", "left_thumb_1"],
        ["right_inspire_base_link", "right_thumb_2"],
    ]
    contract["collision_exclusion_reason"] = (
        "Only the palm housing versus thumb proximal grandchild: supplied native convex hulls overlap "
        "internally by 4.107/2.893 mm insphere radius at the feasible open pose. "
        "External finger/palm and inter-finger collisions retained."
    )
    contract["action_layout"] = ["left_arm_7_rad", "left_hand_6_rad", "right_arm_7_rad", "right_hand_6_rad"]
    ET.indent(robot)
    ET.ElementTree(robot).write(output / "robot.urdf", encoding="utf-8", xml_declaration=True)
    save(output / "model_contract.json", contract)
    save(output / "input_sha256.json", inputs)
    save(
        output / "sha256.json",
        {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            {
                "model": MODEL,
                "links": len(fk(robot)),
                "action_dimension": 26,
                "urdf_sha256": digest(output / "robot.urdf"),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.source.resolve(), a.output.resolve())
