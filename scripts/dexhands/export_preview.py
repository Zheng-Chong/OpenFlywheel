"""Full robot GLB at one recorded native pose. Visual geometry only, not a physics asset."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from kinematics import visuals
import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0, help="Recorded frame index; never invent a display pose")
    a = p.parse_args()
    report = json.loads((a.run / "report.json").read_text())
    if hashlib.sha256((a.asset / "robot.urdf").read_bytes()).hexdigest() != report["urdf_sha256"]:
        raise ValueError("Preview robot differs from the recording")
    rows = (a.run / "frames.jsonl").read_text().splitlines()
    if not 0 <= a.frame < len(rows):
        raise ValueError("Preview frame is outside the recording")
    row = json.loads(rows[a.frame])
    joints = dict(zip(report["joint_names"], row["joint_position_rad"]))
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    transform[1, 3] = 0.1115
    for name, mesh, color in visuals(a.asset / "robot.urdf", joints, measured_mimics=True):
        if re.fullmatch(r"torso_link|AGV_link|LZ_.*_Link|.*_(shoulder_.*|elbow|wrist_.*)_link", name):
            color = np.array([0.62, 0.63, 0.65, 1])
        elif name in ["Yaw_Link", "Pitching_Link", "head_link", "logo_link"] or name.endswith("_Wheel_Link"):
            color = np.array([0.02, 0.02, 0.02, 1])
        mesh.visual.face_colors = np.tile(np.rint(color * 255).astype(np.uint8), (len(mesh.faces), 1))
        mesh.apply_transform(transform)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(scene.export(file_type="glb"))
    check = trimesh.load(a.output, force="scene")
    assert len(check.geometry) == len(scene.geometry)
    a.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "source_run": a.run.name,
                "frame": row["frame"],
                "time_s": row["time_s"],
                "phase": row["phase"],
                "root_z_m": 0.1115,
                "urdf_sha256": report["urdf_sha256"],
                "trace_sha256": hashlib.sha256((a.run / "frames.jsonl").read_bytes()).hexdigest(),
                "glb_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "geometry_count": len(scene.geometry),
                "scope": "recorded pose, visual only; URDF is physics authority",
                "joint_replay": "all recorded joint values including measured passive followers",
                "recorded_run_passed": report["passed"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
