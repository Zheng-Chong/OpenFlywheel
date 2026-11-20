"""Independent read-only acceptance of one four-hand native recording and HDF5."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def audit(run, plan):
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    rows = [json.loads(s) for s in (run / "frames.jsonl").read_text().splitlines()]
    criteria = json.loads(Path(plan).read_text())["acceptance"]
    checks = {}
    N = len(rows)
    with h5py.File(run / "modalities.h5") as f:
        assert len(f["frame"]) == N == report["frames"]
        assert f["joint_names"].asstr()[:].tolist() == report["joint_names"]
        for key, field in [
            ("state/position_si", "joint_position_rad"),
            ("action/target_si", "target_rad"),
            ("action/requested_si", "requested_target_rad"),
            ("contact/force_world_N", "force_world_N"),
            ("contact/normal_world_N", "normal_force_world_N"),
            ("contact/friction_world_N", "friction_force_world_N"),
            ("contact/force_link_N", "force_link_N"),
            ("contact/link_pose_wxyz", "link_pose_wxyz"),
        ]:
            assert np.array_equal(f[key][:], np.asarray([r[field] for r in rows])), key
        assert np.all(np.diff(f["time_s"][:]) > 0)
        assert np.all(np.diff(f["step"][:]) > 0)
        force = f["contact/force_world_N"][:]
        normal = f["contact/normal_world_N"][:]
        friction = f["contact/friction_world_N"][:]
        pose = f["contact/link_pose_wxyz"][:]
        local = f["contact/force_link_N"][:]
        mats = Rotation.from_quat(pose[..., 3:][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(N, -1, 3, 3)
        force_frame = float(np.max(np.abs(np.einsum("...ji,...j->...i", mats, force) - local)))
        sum_error = float(np.max(abs(force - normal - friction)))
        assert force_frame < 1e-5 and sum_error < 1e-5
        cams = {}
        for name in report["camera_metadata"]:
            g = f["camera/" + name]
            assert len(g["rgb"]) == len(g["depth_m"]) == N
            assert np.array_equal(g["capture_step"][:], f["step"][:])
            valid = []
            std = []
            for k in [0, N // 2, N - 1]:
                rgb = g["rgb"][k]
                depth = g["depth_m"][k]
                std.append(float(rgb.std()))
                valid.append(float(np.mean(np.isfinite(depth) & (depth > 0) & (depth < 10))))
                assert rgb.dtype == np.uint8 and depth.dtype == np.float32
                assert rgb.shape[:2] == depth.shape
            assert min(std) > 1 and min(valid) > 0.05
            stamps = [r["camera_sensor_time_s"][name] for r in rows]
            camera_error = (
                max(abs(float(t) - r["time_s"]) for t, r in zip(stamps, rows) if t is not None)
                if any(t is not None for t in stamps)
                else None
            )
            cams[name] = {
                "frames": N,
                "sample_rgb_std": std,
                "sample_depth_finite_fraction": valid,
                "sensor_clock_max_error_s": camera_error,
                "identical_capture_step": True,
            }
        checks.update(
            camera_clock_synchronized=all(
                v["sensor_clock_max_error_s"] is not None and v["sensor_clock_max_error_s"] < report["dt_s"] * 0.1
                for v in cams.values()
            ),
            exact_hdf5_readback=True,
            frame_count=N,
            monotonic_time=True,
            force_frame_error_N=force_frame,
            normal_plus_friction_error_N=sum_error,
            cameras=cams,
        )
        if "contact/object_reaction_N" in f:
            reaction = f["contact/object_reaction_N"][:]
            assert np.array_equal(reaction, [r["object_reaction_force_world_N"] for r in rows])
            pair_error = float(np.max(abs(force.sum(axis=1) + reaction.sum(axis=1))))
            assert pair_error < 1e-5
            checks["action_reaction_error_N"] = pair_error
        frozen_contract = list((run / "execution_inputs").glob("*_model_contract.json"))
        if frozen_contract:
            contract = json.loads(frozen_contract[0].read_text())
            names = f["joint_names"].asstr()[:].tolist()
            positions = f["state/position_si"][:]
            checks["mimic_error_rad"] = {
                name: float(
                    np.max(
                        abs(
                            positions[:, names.index(name)]
                            - coupling["multiplier"] * positions[:, names.index(coupling["joint"])]
                            - coupling["offset"]
                        )
                    )
                )
                for hand in contract["hands"].values()
                for name, coupling in hand.get("mimic", {}).items()
            }
    checks["finite_state"] = all(np.isfinite(r["joint_position_rad"]).all() for r in rows)
    checks["fk_max_error_m"] = max(r["fk_max_error_m"] for r in rows)
    checks["source_inputs_unchanged"] = all(
        hashlib.sha256((run / "execution_inputs" / n).read_bytes()).hexdigest() == v
        for n, v in report["input_sha256"].items()
    )
    assert (
        checks["source_inputs_unchanged"]
        and checks["finite_state"]
        and checks["fk_max_error_m"] < criteria["fk_error_m"]
    )
    motion = []
    diagnostics = []
    trials = []
    if report["mode"] in ("motion", "smoke"):
        phases = report["phases"]
        baseline = np.array([r["joint_position_rad"] for r in rows if r["phase"] == "baseline"])[-1]
        names = report["joint_names"]
        for phase in phases:
            name = phase["phase"]
            start, end = phase["frames"]
            rs = rows[start:end]
            if name.startswith("axis_"):
                goal = np.array(rs[-1]["requested_target_rad"])
                changed = np.where(abs(goal - baseline) > 0.025)[0]
                actual = np.array([r["joint_position_rad"] for r in rs])
                per = []
                for j in changed:
                    if names[j] not in report["action_joint_names"] or "shoulder" in names[j]:
                        continue
                    signed = (actual[:, j] - baseline[j]) * np.sign(goal[j] - baseline[j])
                    exc = float(signed.max())
                    err = float(phase["per_joint_error_rad"][names[j]])
                    per.append(
                        {
                            "joint": names[j],
                            "excursion_rad": exc,
                            "steady_error_rad": err,
                            "passed": exc >= criteria["independent_motion_rad"]
                            and err < criteria["steady_tracking_rad"],
                        }
                    )
                motion += per
            if "range_" in name and not name.startswith("range_return"):
                # Passive mimic joints have no independent position drive.
                # Their nominal target buffer is not a tracking command.
                active_error = max(phase["per_joint_error_rad"][n] for n in report["action_joint_names"])
                diagnostics.append(
                    {
                        "phase": name,
                        "tracking_error_rad": active_error,
                        "free_tracking_passed": active_error < criteria["steady_tracking_rad"],
                        "scope": "active commanded joints; passive mimic constraints checked separately",
                    }
                )
        final = phases[-1]
        checks["final_return_error_rad"] = final["max_steady_error_rad"]
        checks["independent_axis_cases"] = motion
        checks["independent_axes_passed"] = bool(motion) and all(x["passed"] for x in motion)
        checks["range_diagnostics"] = diagnostics
        passed = checks["independent_axes_passed"] and final["max_steady_error_rad"] < criteria["return_error_rad"]
    else:
        for side in ("left", "right"):
            for shape in ("sphere", "cube", "cylinder"):
                index = next(i for i, x in enumerate(report["objects"]) if x == {"side": side, "shape": shape})
                hold = [r for r in rows if r["phase"] == "hold_" + shape]
                settle = [r for r in rows if r["phase"] == "settle_" + shape]
                released = [r for r in rows if r["phase"] == "recovered_" + shape]
                initial = np.asarray(settle[-1]["object_states_si"][index][:3])
                pos = np.asarray([r["object_states_si"][index][:3] for r in hold])
                rel = []
                for r in hold:
                    w = np.asarray(r["wrist_pose_wxyz"][side])
                    R = Rotation.from_quat(w[3:][[1, 2, 3, 0]]).as_matrix()
                    rel.append(R.T @ (np.asarray(r["object_states_si"][index][:3]) - w[:3]))
                height = float(np.min(pos[:, 2] - initial[2]))
                drift = float(np.max(np.linalg.norm(np.asarray(rel) - rel[0], axis=1)))
                speed = float(np.max([np.linalg.norm(r["object_states_si"][index][7:10]) for r in hold]))
                clear = float(np.linalg.norm(np.asarray(released[-1]["object_states_si"][index][:3]) - initial))
                channel_ids = [
                    k for k, n in enumerate(report["channels"]) if n.startswith(("left_", "l_")) == (side == "left")
                ]
                hold_contacts = [
                    int(
                        np.count_nonzero(
                            np.linalg.norm(np.asarray(r["force_world_N"])[channel_ids], axis=1)
                            > criteria["contact_force_N"]
                        )
                    )
                    for r in hold
                ]
                release_force = float(
                    max(np.linalg.norm(np.asarray(r["force_world_N"])[channel_ids], axis=1).max() for r in released)
                )
                hold_duration = hold[-1]["time_s"] - hold[0]["time_s"]
                entry = {
                    "side": side,
                    "shape": shape,
                    "minimum_hold_lift_m": height,
                    "relative_drift_m": drift,
                    "hold_peak_speed_m_s": speed,
                    "release_offset_m": clear,
                    "release_force_max_N": release_force,
                    "minimum_hold_contact_channels": min(hold_contacts),
                    "recorded_hold_duration_s": hold_duration,
                    "passed": bool(
                        height >= criteria["object_lift_m"]
                        and drift < criteria["object_relative_drift_m"]
                        and min(hold_contacts) >= 1
                        and release_force < criteria["release_force_N"]
                        and hold_duration >= criteria["hold_time_s"]
                    ),
                }
                trials.append(entry)
        checks["dynamic_trials"] = trials
        passed = all(x["passed"] for x in trials)
    result = {
        "model": report["model"],
        "mode": report["mode"],
        "passed": bool(passed and checks["camera_clock_synchronized"]),
        "checks": checks,
        "scope": "current simulation tests only; no hardware calibration or all-pose safety guarantee",
    }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = audit(a.run, a.plan)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": r["model"],
                "mode": r["mode"],
                "passed": r["passed"],
                "failed_motion": [x for x in r["checks"].get("independent_axis_cases", []) if not x["passed"]],
                "dynamic_trials": r["checks"].get("dynamic_trials", []),
            }
        )
    )
