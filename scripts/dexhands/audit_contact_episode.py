"""Read-only checks of recorded native contacts, sensor identity and camera clocks."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def audit(run, asset, episode):
    report = json.loads((run / "report.json").read_text())
    contract = json.loads((asset / "model_contract.json").read_text())
    rows = [json.loads(s) for s in (run / "frames.jsonl").read_text().splitlines()]
    channels = report["channels"]
    expected = [n for side in ("left", "right") for n in contract["tactile"][side].get("frames", [])]
    if expected:
        assert set(channels) == set(expected), "The tested links differ from the registered sensor pads"
    force = np.asarray([r["force_world_N"] for r in rows])
    normal = np.asarray([r["normal_force_world_N"] for r in rows])
    friction = np.asarray([r["friction_force_world_N"] for r in rows])
    counter = np.asarray([r["opposite_force_world_N"] for r in rows])
    poses = np.asarray([r["link_pose_wxyz"] for r in rows])
    local = np.asarray([r["force_link_N"] for r in rows])
    matrices = Rotation.from_quat(poses[..., 3:][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    matrices = matrices.reshape(*poses.shape[:2], 3, 3)
    transform_error = float(np.max(abs(np.einsum("...ji,...j->...i", matrices, force) - local)))
    composition_error = float(np.max(abs(force - normal - friction)))
    opposition_error = float(np.max(np.linalg.norm(force + counter, axis=-1)))
    assert transform_error < 1e-7 and composition_error < 1e-6 and opposition_error < 0.01
    peaks = {
        shape: np.linalg.norm([r["force_world_N"] for r in rows if r["shape"] == shape], axis=-1).max(axis=0).tolist()
        for shape in ("sphere", "flat", "cylinder")
    }
    recovered = float(max(np.linalg.norm(r["force_world_N"], axis=-1).max() for r in rows if r["phase"] == "recovered"))
    assert min(min(v) for v in peaks.values()) > 0.001 and recovered < 0.01
    assert report["urdf_sha256"] == hashlib.sha256((asset / "robot.urdf").read_bytes()).hexdigest()
    assert all(
        hashlib.sha256((run / "execution_inputs" / n).read_bytes()).hexdigest() == digest
        for n, digest in report["execution_input_sha256"].items()
    )
    camera_errors = {}
    with h5py.File(episode) as f:
        assert np.array_equal(f["tactile/force_world_N"][:], force)
        assert np.array_equal(f["tactile/friction_force_world_N"][:], friction)
        assert np.array_equal(f["tactile/opposite_force_world_N"][:], counter)
        for name in rows[0]["camera_sensor_time_s"]:
            g = f["camera/" + name]
            camera_errors[name] = float(np.max(abs(g["time_s"][:] - f["time_s"][:])))
            assert camera_errors[name] < report["dt_s"] * 0.1
            assert np.array_equal(g["capture_step"][:], f["step"][:])
            for encoded in g["jpeg"]:
                im = np.asarray(Image.open(io.BytesIO(encoded.tobytes())).convert("RGB"))
                assert im.std() > 1
        if "tactile/maps" in f:
            depth = f["tactile/maps/depth_m"][:]
            assert depth.shape[0] == len(rows) and depth.shape[1] == len(channels)
            assert np.isfinite(depth).all() and depth.min() >= 0 and depth.max() > 0.0001
            assert report["vendor_tacmap_executed"]
            assert report["metric_sphere_max_error_m"] < 0.0001
            assert report["official_quantizer_max_uint8_error"] <= 1
    return {
        "model": report["model"],
        "run": run.name,
        "passed": True,
        "frames": len(rows),
        "channels": channels,
        "shape_channel_cases": len(channels) * len(peaks),
        "per_shape_peak_N": peaks,
        "recovered_force_max_N": recovered,
        "force_frame_error_N": transform_error,
        "normal_plus_friction_error_N": composition_error,
        "action_reaction_error_N": opposition_error,
        "camera_clock_error_s": camera_errors,
        "recorded_jpeg_decoding": True,
        "sensor_identity_matches_contract": bool(expected),
        "scope": "Simulation fixture only. Aggregate link forces are not calibrated hardware taxels.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run, a.asset, a.episode)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_shape_peak_N", "channels")}))
