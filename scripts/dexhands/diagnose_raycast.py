"""Minimal Warp sphere-ray diagnostic, independent of the robot and sensor code."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import warp as wp


@wp.kernel
def query(
    mesh: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    pose: wp.transform,
    distance: wp.array(dtype=float),
    limit: float,
):
    i = wp.tid()
    inv = wp.transform_inverse(pose)
    start = wp.transform_point(inv, starts[i])
    direction = wp.transform_vector(inv, directions[i])
    result = wp.mesh_query_ray(mesh, start, direction, limit)
    if result.result:
        distance[i] = result.t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.diagnostics.read_text())
    worst = max(rows, key=lambda r: r["error_m"])
    tests = [rows[0], worst]
    reports = []
    wp.init()
    for scale in (1.0, 1000.0):
        mesh = trimesh.creation.uv_sphere(radius=0.002 * scale)
        points = wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        triangles = wp.array(np.asarray(mesh.faces, dtype=np.int32).ravel(), dtype=wp.int32, device="cuda:0")
        obj = wp.Mesh(points=points, indices=triangles)
        for row in tests:
            start = np.asarray(row["start_w"]) * scale
            direction = np.asarray(row["direction"])
            center = np.asarray(row["probe_center_w"]) * scale
            q = np.asarray(row["tracked_quat_wxyz"][0])
            q = q[[1, 2, 3, 0]]
            output = wp.array(np.array([np.inf], dtype=np.float32), dtype=float, device="cuda:0")
            wp.launch(
                query,
                dim=1,
                inputs=[
                    obj.id,
                    wp.array(start.reshape(1, 3), dtype=wp.vec3, device="cuda:0"),
                    wp.array(direction.reshape(1, 3), dtype=wp.vec3, device="cuda:0"),
                    wp.transform(wp.vec3(*center), wp.quat(*q)),
                    output,
                    0.015 * scale,
                ],
                device="cuda:0",
            )
            first = float(output.numpy()[0])
            second = wp.array(np.array([np.inf], dtype=np.float32), dtype=float, device="cuda:0")
            if np.isfinite(first):
                advanced = start + direction * (first - 0.0001 * scale)
                wp.launch(
                    query,
                    dim=1,
                    inputs=[
                        obj.id,
                        wp.array(advanced.reshape(1, 3), dtype=wp.vec3, device="cuda:0"),
                        wp.array((-direction).reshape(1, 3), dtype=wp.vec3, device="cuda:0"),
                        wp.transform(wp.vec3(*center), wp.quat(*q)),
                        second,
                        0.5 * scale,
                    ],
                    device="cuda:0",
                )
            reports.append(
                {
                    "scale": scale,
                    "channel": row["channel"],
                    "phase": row["phase"],
                    "distance_m": float(output.numpy()[0]) / scale,
                    "second_m": float(second.numpy()[0]) / scale,
                    "cpd_depth_m": first / scale if np.isfinite(first) and np.isfinite(second.numpy()[0]) else 0.0,
                    "original_sensor_raw_m": row["raw_m"],
                    "analytic_m": row["analytic_m"],
                }
            )
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
