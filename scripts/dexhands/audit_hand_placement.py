"""Separate manufacturer hand fidelity from complete-robot wrist orientation.

The orientation convention is a simulation design: at the review arm pose,
the index/thumb side points up and positive index flexion points inward.
It does not establish a measured physical flange or TCP.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from build_models import hand_path, mesh_path
from kinematics import fk
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anatomy(model, side):
    if "sharpa" in model:
        return (
            side + "_index_MCP_VL",
            side + "_pinky_MCP_VL",
            side + "_index_fingertip",
            side + "_index_MCP_FE",
        )
    prefix = side[0] + "_"
    return (
        prefix + "index_finger_proximal",
        prefix + "pinky_proximal",
        prefix + "index_finger_tip",
        prefix + "index_finger_mcp_flex",
    )


def orientation(root, model, side):
    index, pinky, tip, flex = anatomy(model, side)
    q = {"left_elbow_joint": -0.6, "right_elbow_joint": -0.6}
    base = fk(root, q)
    bent = fk(root, {**q, flex: 0.001})
    lateral = base[index][:3, 3] - base[pinky][:3, 3]
    lateral /= np.linalg.norm(lateral)
    curl = bent[tip][:3, 3] - base[tip][:3, 3]
    curl /= np.linalg.norm(curl)
    inward = np.array([0, -1 if side == "left" else 1, 0])
    return {
        "pinky_to_index_unit_world": lateral.tolist(),
        "positive_index_flexion_unit_world": curl.tolist(),
        "thumb_side_up_cosine": float(lateral[2]),
        "inward_flexion_cosine": float(curl @ inward),
        "passed": bool(lateral[2] > 0.6 and curl @ inward > 0.95),
    }


def audit(directory, vendor):
    contract = json.loads((directory / "model_contract.json").read_text())
    assembled = ET.parse(directory / "robot.urdf").getroot()
    result = {"urdf_sha256": sha(directory / "robot.urdf"), "hands": {}}
    for side, data in contract["hands"].items():
        path = hand_path(vendor, directory.name, side)
        source = ET.parse(path).getroot()

        # Whitespace differs between serializers. Compare semantic trees.
        def semantic(e):
            return e.tag, sorted(e.attrib.items()), [semantic(c) for c in e]

        unchanged = all(
            semantic(j) == semantic(assembled.find(f"joint[@name='{j.get('name')}']")) for j in source.findall("joint")
        )
        meshes = []
        links_unchanged = True
        for link in source.findall("link"):
            copied = assembled.find(f"link[@name='{link.get('name')}']")
            original_meshes, copied_meshes = link.findall(".//mesh"), copied.findall(".//mesh")
            for a, b in zip(original_meshes, copied_meshes, strict=True):
                meshes.append(sha(mesh_path(path, a.get("filename"))) == sha(directory / b.get("filename")))
                a.set("filename", b.get("filename"))
            links_unchanged &= semantic(link) == semantic(copied)
        cases = [{}, {j: 0.1 for j in data["joints"]}]
        cases += [{j: 0.1} for j in data["joints"] if "thumb" in j]
        max_error = 0.0
        for q in cases:
            src, dst = fk(source, q), fk(assembled, q)
            inverse = np.linalg.inv(dst[data["root"]])
            max_error = max(max_error, max(float(np.abs(src[n] - inverse @ dst[n]).max()) for n in src))
        direction = orientation(assembled, directory.name, side)
        result["hands"][side] = {
            "vendor_urdf_sha256": sha(path),
            "manufacturer_joint_definitions_preserved": unchanged,
            "manufacturer_link_properties_preserved": bool(links_unchanged),
            "mesh_references_checked": len(meshes),
            "mesh_bytes_preserved": all(meshes),
            "mount_relative_fk_max_matrix_error": max_error,
            "fk_cases": len(cases),
            "orientation": direction,
            "passed": bool(unchanged and links_unchanged and all(meshes) and max_error < 1e-10 and direction["passed"]),
        }
    result["passed"] = all(h["passed"] for h in result["hands"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "source fidelity and simulation mounting convention; no measured physical calibration",
        "models": {p.name: audit(p, args.vendor) for p in sorted((args.assets / "Robots").glob("g1d_*"))},
    }
    result["passed"] = all(m["passed"] for m in result["models"].values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({m: r["passed"] for m, r in result["models"].items()}))


if __name__ == "__main__":
    main()
