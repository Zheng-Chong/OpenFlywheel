"""Build three complete G1-D variants from pinned, unmodified manufacturer hand URDFs.

No real hardware connection. Wrist transforms are explicit simulation designs,
not claims about bolt compatibility or measured TCP calibration.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from kinematics import fk
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("g1d_wuji_hand2_1", "g1d_wuji_hand2_2", "g1d_sharpa_wave")
PINS = {
    "wuji-description": "5c4dfa16917355d073e2e1d7e3d03897b6cf9508",
    "sharpa-models": "0d19cac602f46456b819e4b6a2c09a74982c9a3e",
    "sharpa-tacmap": "e4f8827407e15625368ef0b9cf5a72c44304e16b",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def hand_path(vendor, model, side):
    if model.endswith("sharpa_wave"):
        return vendor / "sharpa-models/wave_01" / (side + "_sharpa_wave") / (side + "_sharpa_wave_with_flange.urdf")
    return vendor / "wuji-description/hand2" / model.removeprefix("g1d_wuji_") / "body/urdf" / (side + ".urdf")


def mesh_path(urdf, filename):
    return urdf.parent / (filename.split("/", 3)[3] if filename.startswith("package://") else filename)


def copy_meshes(root, source, destination, group, inputs):
    for mesh in root.iter("mesh"):
        original = mesh_path(source, mesh.get("filename")).resolve()
        sha = digest(original)
        # Digest prefix avoids silently overwriting same-named left/right meshes.
        relative = Path("meshes") / group / ("m_" + sha[:12] + "_" + original.name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(original, target)
        mesh.set("filename", relative.as_posix())
        inputs[str(original)] = sha


def build(vendor, out, model, usd_audit):
    destination = out / model
    destination.mkdir(parents=True, exist_ok=False)
    base = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"
    robot = ET.parse(base).getroot()
    robot.set("name", model)
    inputs = {str(base): digest(base)}
    for element in list(robot):
        if "dex1" in element.get("name", "").lower():
            robot.remove(element)
    for joint in robot.findall("joint"):
        if joint.get("name").endswith("Wheel_Joint"):
            joint.set("type", "fixed")
    copy_meshes(robot, base, destination, "g1d", inputs)
    contract = {
        "model": model,
        "source_commits": PINS,
        "joint_unit": "rad",
        "length_unit": "m",
        "mount_status": "simulation_candidate_not_measured",
        "physical_calibration": "pending",
        "hand_source_preservation": "links, joint axes/origins/limits, mass/inertia and mesh bytes preserved",
        "hands": {},
        "collision_exclusions": [],
        "tactile": {},
        "forbidden_runtime_interfaces": ["DDS", "real_camera", "hardware_SDK"],
    }
    sharpa = model.endswith("sharpa_wave")
    for side in ("left", "right"):
        source = hand_path(vendor, model, side)
        hand = ET.parse(source).getroot()
        inputs[str(source)] = digest(source)
        children = {j.find("child").get("link") for j in hand.findall("joint")}
        mount = (set(link.get("name") for link in hand.findall("link")) - children).pop()
        joint_names = [j.get("name") for j in hand.findall("joint") if j.get("type") != "fixed"]
        limits = {
            j.get("name"): {k: float(v) for k, v in j.find("limit").attrib.items()}
            for j in hand.findall("joint")
            if j.get("type") != "fixed"
        }
        copy_meshes(hand, source, destination, side, inputs)
        for item in hand:
            if item.tag in ("link", "joint", "material"):
                robot.append(copy.deepcopy(item))
        # Official Unitree palm reference includes the ±3 mm lateral offsets.
        xyz = [0.0415, 0.003 if side == "left" else -0.003, 0.0]
        if sharpa:
            # Hand +Z -> wrist +X; the two palm normals point toward the other arm.
            rpy = [-np.pi / 2, 0.0, -np.pi / 2] if side == "left" else [np.pi / 2, 0.0, np.pi / 2]
        else:
            # Wuji fingers extend along hand -Z. Align hand +X (index /
            # thumb side) with wrist +Z, and positive finger flexion toward
            # the other hand. The previous [pi, pi/2, 0] aligned extension
            # but inverted both palm normal and thumb side by 180 degrees.
            rpy = [0.0, -np.pi / 2, 0.0]
        joint = ET.SubElement(robot, "joint", name=side + "_dexhand_mount", type="fixed")
        ET.SubElement(joint, "parent", link=side + "_wrist_yaw_link")
        ET.SubElement(joint, "child", link=mount)
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
        mjcf = source.with_suffix(".xml") if sharpa else source.parent.parent / "mjcf" / (side + ".xml")
        xml = ET.parse(mjcf).getroot()
        inputs[str(mjcf)] = digest(mjcf)
        exclusions = [list(p.attrib.values()) for p in xml.findall("contact/exclude")]
        # Sharpa's fixed-link elastomer belongs to its distal rigid body after
        # fixed-joint merge; only that same-body pair is filtered when unmerged.
        if sharpa:
            exclusions += [
                [side + "_" + finger + "_DP", side + "_" + finger + "_elastomer"]
                for finger in ("thumb", "index", "middle", "ring", "pinky")
            ]
        contract["collision_exclusions"] += exclusions
        dynamics = {
            p.get("joint"): {k: float(v) for k, v in p.attrib.items() if k in ("kp", "kv")}
            for p in xml.findall("actuator/position")
        }
        contract["hands"][side] = {
            "root": mount,
            "joints": joint_names,
            "limits": limits,
            "mount_xyz_m": xyz,
            "mount_rpy_rad": rpy,
            "configured_mass_kg": sum(
                float(link.find("inertial/mass").get("value"))
                for link in hand.findall("link")
                if link.find("inertial/mass") is not None
            ),
            "mjcf_position_gains": dynamics,
            "vendor_urdf": str(source.relative_to(vendor)),
        }
        contract["presentation_pose"] = "mechanical_zero_not_natural_open"
        contract["mount_orientation_contract"] = "thumb_side_up_and_positive_index_flexion_inward"
        tactile_frames = [
            link.get("name")
            for link in hand.findall("link")
            if link.get("name").endswith("_elastomer" if sharpa else "_tip_sensor_frame")
        ]
        contract["tactile"][side] = {
            "frames": tactile_frames,
            "mode": "official_tacmap_depth"
            if sharpa
            else ("simulation_contact_probe_only" if model.endswith("_2") else "no_tip_sensor_in_vendor_model"),
            "real_sensor_calibration": "not_available",
            "force_source": (
                "PhysX contact force, separate from geometric depth; never convert depth to N without a calibrated law"
            ),
        }
    poses = fk(robot)
    names = [link.get("name") for link in robot.findall("link")]
    assert len(names) == len(set(names))
    assert all(a in poses and b in poses for a, b in contract["collision_exclusions"])
    contract["action_layout"] = [
        s + "_" + kind
        for s in ("left", "right")
        for kind in ("arm_7_rad", "hand_" + str(len(contract["hands"][s]["joints"])) + "_rad")
    ]
    contract["action_dimension"] = 14 + sum(len(h["joints"]) for h in contract["hands"].values())
    contract["optional_body_commands"] = {"waist_joint_state": 2, "lift_joint_state": 1}
    ET.indent(robot)
    ET.ElementTree(robot).write(destination / "robot.urdf", encoding="utf-8", xml_declaration=True)
    for side, data in usd_audit[model].items():
        for first, targets in data["filtered_pairs"].items():
            contract["collision_exclusions"] += [[first, Path(t).name] for t in targets]
    contract["collision_exclusions"] = sorted(set(tuple(sorted(pair)) for pair in contract["collision_exclusions"]))
    assert all(a in poses and b in poses for a, b in contract["collision_exclusions"])
    save(destination / "model_contract.json", contract)
    save(destination / "vendor_dynamics.json", usd_audit[model])
    for side in ("left", "right"):
        arm = [
            side + "_" + suffix
            for suffix in (
                "shoulder_pitch_joint",
                "shoulder_roll_joint",
                "shoulder_yaw_joint",
                "elbow_joint",
                "wrist_roll_joint",
                "wrist_pitch_joint",
                "wrist_yaw_joint",
            )
        ]
        cfg = {
            "urdf_path": "./robot.urdf",
            "ee_joints": side + "_wrist_yaw_joint",
            "ee_link": contract["hands"][side]["root"],
            "ee_type": "hand",
            "base_link": "torso_link",
            "arm_joints_name": arm,
            "gripper_joints_name": contract["hands"][side]["joints"],
            "save_gripper_joints_name": contract["hands"][side]["joints"],
            "gripper_move": {},
            "gripper_bias": 0.0,
            "gripper_scale": [0.0, 1.0],
            "camera": [
                c
                for c in yaml.safe_load((ROOT / "scripts/g1d/left_robot_config.yml").read_text())["camera"]
                if c["link"] == "head_link"
            ]
            if side == "left"
            else [],
        }
        save(destination / (side + "_robot_config.yml"), cfg)
    save(destination / "input_sha256.json", inputs)
    save(
        destination / "sha256.json",
        {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()},
    )
    return {
        "model": model,
        "urdf_sha256": digest(destination / "robot.urdf"),
        "links": len(poses),
        "hand_dof": [len(h["joints"]) for h in contract["hands"].values()],
        "action_dimension": contract["action_dimension"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usd-audit", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.usd_audit.read_text())
    manifest = json.loads(args.source_manifest.read_text())
    for repository, commit in PINS.items():
        if manifest[repository]["commit"] != commit:
            raise ValueError(f"Unexpected source commit for {repository}")
        for filename, expected in manifest[repository]["files"].items():
            source = args.vendor / repository / filename
            if not source.is_file() or digest(source) != expected:
                raise ValueError(f"Missing or modified manufacturer input: {repository}/{filename}")
    save(
        args.output / "build_results.json",
        [build(args.vendor.resolve(), args.output.resolve(), model, audit) for model in MODELS],
    )


if __name__ == "__main__":
    main()
