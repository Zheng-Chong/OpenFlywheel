"""Project vendor sampling grids to the exact shipped CAD surface for simulation.

This is a simulation geometry adaptation. It does not calibrate a real sensor,
infer taxel force sensitivities, or modify the manufacturer source files.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(vendor, output, kind):
    directory = vendor / "sharpa-tacmap/assets/tactilesensor_map"
    suffix = "_origin" if kind == "4F" else ""
    points_file = directory / f"tactileSensor_map_{kind}_point{suffix}.npy"
    normals_file = directory / f"tactileSensor_map_{kind}_normal{suffix}.npy"
    mesh_file = vendor / f"sharpa-models/wave_01/right_sharpa_wave/meshes/elastomer_HB1_{kind}.STL"
    points = np.load(points_file).astype(float) * 0.001
    normals = np.load(normals_file).astype(float)
    valid = np.linalg.norm(normals, axis=-1) > 0.9
    mesh = trimesh.load(mesh_file, force="mesh").convex_hull
    ids = np.flatnonzero(valid)
    p = points.reshape(-1, 3)[ids]
    aligned = np.zeros_like(points).reshape(-1, 3)
    aligned_normals = np.zeros_like(normals).reshape(-1, 3)
    distances = []
    alignments = []
    for start in range(0, len(ids), 96):
        near, dist, face = trimesh.proximity.closest_point_naive(mesh, p[start : start + 96])
        normal = mesh.face_normals[face]
        dot = (normal * normals.reshape(-1, 3)[ids[start : start + 96]]).sum(-1)
        aligned[ids[start : start + 96]] = near
        aligned_normals[ids[start : start + 96]] = normal
        distances.extend(dist.tolist())
        alignments.extend(dot.tolist())
    distances = np.array(distances)
    alignments = np.array(alignments)
    # Reject back/side-face projections and edge regions outside a 3 mm fit
    # envelope. These limits are explicit simulation mask choices.
    keep = (alignments > 0.5) & (distances < 0.003)
    mask = np.zeros(valid.size, dtype=bool)
    mask[ids[keep]] = True
    aligned[~mask] = 0.0
    aligned_normals[~mask] = 0.0
    # Keep the vendor millimetre format so the official correction_scale=1e-3
    # remains correct. Record the mask separately: a zero map pixel is not a taxel.
    np.save(output / f"{kind}_points_mm.npy", (aligned * 1000).reshape(points.shape).astype(np.float32))
    np.save(output / f"{kind}_normals.npy", aligned_normals.reshape(normals.shape).astype(np.float32))
    np.save(output / f"{kind}_valid.npy", mask.reshape(valid.shape))
    report = {
        "type": kind,
        "resolution": [240, 240],
        "vendor_valid_samples": int(valid.sum()),
        "cad_valid_samples": int(mask.sum()),
        "original_distance_mm_p50_p95_max": (np.quantile(distances, [0.5, 0.95, 1]) * 1000).tolist(),
        "outward_normal_dot_p05_p50_p95": np.quantile(alignments, [0.05, 0.5, 0.95]).tolist(),
        "projection_tolerance_m": 0.003,
        "normal_cosine_threshold": 0.5,
        "projection_target": "convex_hull_collision_surface_used_by_URDF_import",
        "scope": "CAD collision projection only; no real unit calibration; no inferred force law",
        "input_sha256": {str(p.relative_to(vendor)): sha(p) for p in (points_file, normals_file, mesh_file)},
    }
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = [fit(args.vendor, args.output, k) for k in ("4F", "TH")]
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    manifest = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
