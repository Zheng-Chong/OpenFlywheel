"""Audit frozen vendor URDFs without importing simulators or hardware SDKs."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from kinematics import fk
import numpy as np


def audit(path):
    root = ET.parse(path).getroot()
    poses = fk(root)
    children = {j.find("child").get("link") for j in root.findall("joint")}
    joints = []
    inertias = []
    meshes = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            a = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
            m = np.array(
                [[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]]
            )
            eig = np.linalg.eigvalsh(m)
            inertias.append(
                {
                    "link": link.get("name"),
                    "mass_kg": float(inertial.find("mass").get("value")),
                    "principal_inertia_kg_m2": eig.tolist(),
                    "positive": bool(np.all(eig > 0)),
                    "triangle": bool(eig[2] <= eig[0] + eig[1] + 1e-12),
                }
            )
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename.startswith("package://"):
                resolved = path.parent / filename.split("/", 3)[3]
            else:
                resolved = path.parent / filename
            meshes.append({"file": filename, "exists": resolved.exists(), "path": str(resolved.resolve())})
    for j in root.findall("joint"):
        if j.get("type") != "fixed":
            joints.append(
                {
                    "name": j.get("name"),
                    "type": j.get("type"),
                    "axis": j.find("axis").get("xyz"),
                    "limit": {k: float(v) for k, v in j.find("limit").attrib.items()},
                    "mimic": None if j.find("mimic") is None else j.find("mimic").attrib,
                }
            )
    frames = {
        n: t.tolist() for n, t in poses.items() if n.endswith(("_tip", "_elastomer", "_fingertip", "_sensor_frame"))
    }
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "root": sorted(set(poses) - children),
        "links": len(poses),
        "joints": joints,
        "configured_mass_kg": sum(x["mass_kg"] for x in inertias),
        "inertias": inertias,
        "meshes": meshes,
        "frames_zero_m": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for model in ("wuji_2_1", "wuji_2_2", "sharpa_wave"):
        result[model] = {}
        for side in ("left", "right"):
            if model.startswith("wuji"):
                path = args.vendor / "wuji-description/hand2" / ("hand" + model[5:]) / "body/urdf" / (side + ".urdf")
            else:
                path = (
                    args.vendor
                    / "sharpa-models/wave_01"
                    / (side + "_sharpa_wave")
                    / (side + "_sharpa_wave_with_flange.urdf")
                )
            result[model][side] = audit(path)
    args.output.write_text(json.dumps(result, indent=2))
    for model, hands in result.items():
        for side, d in hands.items():
            print(
                model,
                side,
                "links",
                d["links"],
                "dof",
                len(d["joints"]),
                "mass",
                d["configured_mass_kg"],
                "missing meshes",
                sum(not x["exists"] for x in d["meshes"]),
                "inertia invalid",
                [i["link"] for i in d["inertias"] if not (i["positive"] and i["triangle"])],
            )
            print("tips", {n: np.round(np.array(t)[:3, 3], 4).tolist() for n, t in d["frames_zero_m"].items()})


if __name__ == "__main__":
    main()
