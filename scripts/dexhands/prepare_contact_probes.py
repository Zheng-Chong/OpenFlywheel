"""Derive reproducible exposed mesh points for simulation-only contact tests."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from kinematics import fk, origin_transform
import numpy as np
from scipy.spatial import ConvexHull
import trimesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {}
    for directory in sorted((args.assets / "Robots").glob("g1d_*")):
        contract = json.loads((directory / "model_contract.json").read_text())
        root = ET.parse(directory / "robot.urdf").getroot()
        poses = fk(root)
        probes = []
        for side, h in contract["hands"].items():
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                if "sharpa" in directory.name:
                    name = side + "_" + finger + "_elastomer"
                    kind = "TH" if finger == "thumb" else "4F"
                    points = np.load(args.maps / f"{kind}_points_mm.npy") * 0.001
                    normals = np.load(args.maps / f"{kind}_normals.npy")
                    valid = np.load(args.maps / f"{kind}_valid.npy")
                    score = np.linalg.norm(points - np.array([0, -0.019, -0.006]), axis=-1)
                    score[~valid | (normals[..., 2] > -0.8)] = np.inf
                    idx = np.unravel_index(np.argmin(score), score.shape)
                    point = points[idx]
                    normal = normals[idx]
                else:
                    prefix = "l_" if side == "left" else "r_"
                    stem = prefix + (
                        {"index": "index_finger", "middle": "middle_finger", "ring": "ring_finger"}.get(finger, finger)
                    )
                    name = stem + ("_tip_sensor_frame" if directory.name.endswith("_2") else "_distal")
                    link = root.find(f"link[@name='{name}']")
                    mesh_node = link.find("collision/geometry/mesh")
                    mesh = trimesh.load(directory / mesh_node.get("filename"), force="mesh")
                    mesh.apply_transform(origin_transform(link.find("collision/origin")))
                    mesh = mesh.convex_hull
                    parent_joint = next(j for j in root.findall("joint") if j.find("child").get("link") == name)
                    parent = parent_joint.find("parent").get("link")
                    pl = root.find(f"link[@name='{parent}']")
                    pm = pl.find("collision/geometry/mesh")
                    other = trimesh.load(directory / pm.get("filename"), force="mesh")
                    other.apply_transform(
                        np.linalg.inv(poses[name]) @ poses[parent] @ origin_transform(pl.find("collision/origin"))
                    )
                    planes = ConvexHull(other.vertices).equations
                    distance = (mesh.triangles_center @ planes[:, :3].T + planes[:, 3]).max(axis=1)
                    outward = (mesh.face_normals * (mesh.triangles_center - other.centroid)).sum(-1) > 0
                    score = np.maximum(0, distance) * np.sqrt(mesh.area_faces)
                    score[~outward] = 0
                    idx = int(score.argmax())
                    point = mesh.triangles_center[idx]
                    normal = mesh.face_normals[idx]
                probes.append(
                    {
                        "side": side,
                        "finger": finger,
                        "link": name,
                        "point_m": point.tolist(),
                        "normal": normal.tolist(),
                        "virtual": directory.name.endswith("_1"),
                        "source": "exact_CAD_surface_simulation_fixture",
                    }
                )
        result[directory.name] = probes
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
