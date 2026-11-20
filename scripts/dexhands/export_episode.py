"""Export one native recording, preserving frame identity, SI units and canonical action order."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export(run, asset, output, maps=None):
    report = json.loads((run / "report.json").read_text())
    if not report["passed"]:
        raise ValueError("Only an accepted individual run can be exported")
    contract = json.loads((asset / "model_contract.json").read_text())
    if digest(asset / "robot.urdf") != report["urdf_sha256"]:
        raise ValueError("Robot differs from recording")
    rows = [json.loads(v) for v in (run / "frames.jsonl").read_text().splitlines()]
    names = report["joint_names"]
    arm = [
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
    ]
    canonical = [
        j for side in ("left", "right") for j in ([side + "_" + v for v in arm] + contract["hands"][side]["joints"])
    ]
    ids = [names.index(j) for j in canonical]
    joints = {j.get("name"): j.get("type") for j in ET.parse(asset / "robot.urdf").getroot().findall("joint")}
    units = ["m" if joints[n] == "prismatic" else "rad" for n in names]
    q = np.array([r["joint_position_rad"] for r in rows])
    target_key = "target_rad" if "target_rad" in rows[0] else "target_joint_rad"
    target = np.array([r[target_key] for r in rows])
    requested = np.array([r.get("requested_target_rad", r[target_key]) for r in rows])
    time = np.array([r["time_s"] for r in rows])
    assert np.all(np.diff(time) > 0) and len(rows) == report["frames"]
    sources = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl", ".npz")}
    validity = None
    if (run / "tactile.npz").exists():
        if maps is None:
            raise ValueError("Sharpa export requires the recorded CAD map directory (--maps)")
        if digest(maps / "report.json") != report["execution_input_sha256"]["report.json"]:
            raise ValueError("CAD sampling map differs from recording")
        validity = np.stack(
            [
                np.load(maps / ("TH_valid.npy" if "thumb" in channel else "4F_valid.npy"))
                for channel in report["channels"]
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as f:
        f.attrs.update(
            schema="robodojodexhands.episode.v1",
            model=report["model"],
            source_run=run.name,
            urdf_sha256=report["urdf_sha256"],
            hardware_connected=False,
            source_sha256=json.dumps(sources),
            provenance="One recording only. Native and contact episodes are separate.",
            raw_trace_unit_note=(
                "Legacy JSON row key joint_position_rad includes lift metres; joint_units is authoritative."
            ),
        )
        string = h5py.string_dtype("utf-8")
        for key, val in {
            "frame": np.arange(len(rows)),
            "time_s": time,
            "robot/joint_position_si": q,
            "robot/target_si": target,
            "robot/requested_target_si": requested,
            "observation/state": q[:, ids],
            "action/position_rad": requested[:, ids],
        }.items():
            f.create_dataset(key, data=val, compression="gzip")
        for key, val in {
            "robot/joint_names": names,
            "robot/joint_units": units,
            "action/joint_names": canonical,
            "phase": [r["phase"] for r in rows],
        }.items():
            f.create_dataset(key, data=val, dtype=string)
        if "camera_sensor_time_s" in rows[0]:
            # Keep the actual sensor clock, never a frame-index-derived time.
            f.create_dataset("step", data=[r["step"] for r in rows])
            for camera in rows[0]["camera_sensor_time_s"]:
                stamps = np.array([r["camera_sensor_time_s"][camera] for r in rows])
                assert np.max(abs(stamps - time)) < report["dt_s"] * 0.1
                g = f.require_group("camera/" + camera)
                g.create_dataset("time_s", data=stamps)
                g.create_dataset("capture_step", data=[r["step"] for r in rows])
                jpeg = g.create_dataset("jpeg", (len(rows),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for i, row in enumerate(rows):
                    encoded = (run / camera / f"{row['frame']:05d}.jpg").read_bytes()
                    jpeg[i] = np.frombuffer(encoded, dtype="uint8")
                g.attrs.update(encoding="JPEG; original recorded bytes", color_space="RGB")
        if "force_world_N" in rows[0]:
            channels = report["channels"]
            f.create_dataset("tactile/channel_names", data=channels, dtype=string)
            for field in ["force_world_N", "force_link_N", "link_pose_wxyz", "probe_position_w_m", "probe_quat_wxyz"]:
                f.create_dataset("tactile/" + field, data=np.array([r[field] for r in rows]), compression="gzip")
            for field in ("normal_force_world_N", "friction_force_world_N", "opposite_force_world_N"):
                if field in rows[0]:
                    f.create_dataset("tactile/" + field, data=np.array([r[field] for r in rows]), compression="gzip")
            f.create_dataset("shape", data=[r["shape"] for r in rows], dtype=string)
            if (run / "tactile.npz").exists():
                f.create_dataset("tactile/maps/valid_mask", data=validity, compression="gzip")
                f.attrs["cad_map_sha256"] = json.dumps({p.name: digest(p) for p in maps.iterdir() if p.is_file()})
                with np.load(run / "tactile.npz") as tactile:
                    for k in tactile.files:
                        f.create_dataset("tactile/maps/" + k, data=tactile[k], compression="gzip")
            f["tactile"].attrs.update(
                force_unit="N",
                depth_unit="m",
                calibration="simulation_only",
                response="geometric_depth_not_FEM"
                if "sharpa" in report["model"]
                else "aggregate_native_contact_not_taxels",
            )
    # Independent read-back compares the stored action and transformed forces
    # with the original same-frame trace; never trusts only export metadata.
    with h5py.File(output) as f:
        assert np.array_equal(f["observation/state"][:], q[:, ids])
        assert np.array_equal(f["action/position_rad"][:], requested[:, ids])
        assert np.array_equal(f["robot/target_si"][:], target)
        assert f["action/joint_names"].asstr()[:].tolist() == canonical
        if "camera" in f:
            for camera in f["camera"]:
                g = f["camera/" + camera]
                assert np.array_equal(g["time_s"][:], [r["camera_sensor_time_s"][camera] for r in rows])
                for i, row in enumerate(rows):
                    assert g["jpeg"][i].tobytes() == (run / camera / f"{row['frame']:05d}.jpg").read_bytes()
        error = 0.0
        if "force_world_N" in rows[0]:
            world = f["tactile/force_world_N"][:]
            local = f["tactile/force_link_N"][:]
            pose = f["tactile/link_pose_wxyz"][:]
            rotations = (
                Rotation.from_quat(pose[..., 3:][..., [1, 2, 3, 0]].reshape(-1, 4))
                .as_matrix()
                .reshape(*pose.shape[:2], 3, 3)
            )
            error = float(np.max(np.abs(np.einsum("...ji,...j->...i", rotations, world) - local)))
            assert error < 1e-7
    result = {
        "run": run.name,
        "model": report["model"],
        "frames": len(rows),
        "action_dimension": len(ids),
        "action_joint_order": canonical,
        "hdf5_sha256": digest(output),
        "raw_source_sha256": sources,
        "frame_time_monotonic": True,
        "action_exact_readback": True,
        "action_semantics": "requested_active_joint_targets_rad; robot/target_si stores applied servo targets in SI",
        "force_frame_error_N": error,
        "passed": True,
    }
    output.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maps", type=Path)
    a = p.parse_args()
    print(json.dumps(export(a.run, a.asset, a.output, a.maps)))


if __name__ == "__main__":
    main()
