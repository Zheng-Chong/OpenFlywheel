"""Read-only source/data/media audit; writes only the requested audit result."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

MODEL = "g1d_inspire_rh56dftp"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths += [v for k, v in attrs if k in ("src", "href") and v]


def audit(root):
    checks = []
    sources = json.loads((root / "vendor/source_manifest.json").read_text())
    for name, value in sources["files"].items():
        assert sha(root / "vendor/inspire" / name) == value
    checks.append({"check": "attachment_bytes", "files": len(sources["files"])})
    asset = root / "repo_overlay/real2sim/robot" / MODEL
    asset_manifest = json.loads((asset / "sha256.json").read_text())
    for name, value in asset_manifest.items():
        assert sha(asset / name) == value
    contract = json.loads((asset / "model_contract.json").read_text())
    assert contract["action_dimension"] == 26
    assert json.loads((root / "validation/model_audit.json").read_text())["passed"]
    reg = json.loads((root / "repo_overlay/real2sim/embodiments" / MODEL / "registration.json").read_text())
    assert reg["wrist_mount_verified"] is False and reg["full_task_simulation_verified"] is False
    assert reg["tactile_response_validated"] is False and reg["tactile_simulation_verified"] is True
    checks.append({"check": "complete_robot_and_claim_boundaries", "asset_files": len(asset_manifest)})
    for name in ("native_motion", "native_contacts"):
        path = root / "datasets" / (name + ".h5")
        exported = json.loads(path.with_suffix(".audit.json").read_text())
        run = root / "evidence" / exported["run"]
        report = json.loads((run / "report.json").read_text())
        rows = [json.loads(v) for v in (run / "frames.jsonl").read_text().splitlines()]
        assert report["passed"] and report["urdf_sha256"] == sha(asset / "robot.urdf")
        assert sha(path) == exported["hdf5_sha256"]
        for file, digest in report["execution_input_sha256"].items():
            assert sha(run / "execution_inputs" / file) == digest
        for file, digest in exported["raw_source_sha256"].items():
            assert sha(run / file) == digest
        q = np.array([r["joint_position_rad"] for r in rows])
        assert np.isfinite(q).all()
        with h5py.File(path) as f:
            assert len(rows) == report["frames"] == len(f["time_s"])
            assert np.all(np.diff(f["time_s"][:]) > 0)
            assert f["action/position_rad"].shape == (len(rows), 26)
            names = f["robot/joint_names"].asstr()[:].tolist()
            action = f["action/joint_names"].asstr()[:].tolist()
            ids = [names.index(n) for n in action]
            np.testing.assert_array_equal(f["robot/joint_position_si"][:], q)
            target = np.array([r.get("target_rad", r.get("target_joint_rad")) for r in rows])
            requested = np.array(
                [r.get("requested_target_rad", r.get("target_rad", r.get("target_joint_rad"))) for r in rows]
            )
            np.testing.assert_array_equal(f["robot/target_si"][:], target)
            np.testing.assert_array_equal(f["action/position_rad"][:], requested[:, ids])
            if name == "native_motion":
                coupling = max(
                    np.abs(q[:, names.index(n)] - q[:, names.index(m["joint"])] * m["multiplier"] - m["offset"]).max()
                    for h in contract["hands"].values()
                    for n, m in h["mimic"].items()
                )
                assert coupling < 0.005 and abs(coupling - report["mimic_max_error_rad"]) < 1e-9
                assert report["bounded_axis_error_rad"] < 0.04 and report["fk_max_error_m"] < 0.0005
                assert report["endpoint_tracking_accepted"] is False
                assert np.abs(q[-1, ids] - requested[-1, ids]).max() < 0.04
            else:
                world = np.array([r["force_world_N"] for r in rows])
                normal = np.array([r["normal_force_world_N"] for r in rows])
                friction = np.array([r["friction_force_world_N"] for r in rows])
                np.testing.assert_allclose(world, normal + friction, atol=1e-9)
                np.testing.assert_array_equal(f["tactile/force_world_N"][:], world)
                np.testing.assert_array_equal(f["tactile/friction_force_world_N"][:], friction)
                pose = np.array([r["link_pose_wxyz"] for r in rows])
                rot = (
                    Rotation.from_quat(pose[..., 3:][..., [1, 2, 3, 0]].reshape(-1, 4))
                    .as_matrix()
                    .reshape(*pose.shape[:2], 3, 3)
                )
                local = np.einsum("...ji,...j->...i", rot, world)
                np.testing.assert_allclose(local, f["tactile/force_link_N"][:], atol=1e-7)
                for shape in ("sphere", "flat"):
                    samples = [i for i, r in enumerate(rows) if r["shape"] == shape]
                    peak = np.linalg.norm(world[samples], axis=-1).max(axis=0)
                    assert peak.shape == (12,) and np.all(peak > 0.001)
                    np.testing.assert_allclose(peak, report["per_shape_peak_N"][shape], atol=1e-9)
                recovery = [i for i, r in enumerate(rows) if r["phase"] == "recovered"]
                assert np.linalg.norm(world[recovery], axis=-1).max() < 0.001
                assert max(max(r["pair_opposition_error_N"]) for r in rows) < 0.001
        for view in ("whole", "left", "right"):
            video = run / (view + ".mp4")
            info = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-show_entries",
                        "stream=nb_read_frames,width,height",
                        "-of",
                        "json",
                        str(video),
                    ]
                )
            )["streams"][0]
            assert int(info["nb_read_frames"]) == len(rows)
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(video), "-f", "null", "-"], check=True)
        checks.append(
            {"check": "independent_recording_and_media", "run": run.name, "frames": len(rows), "action_dimension": 26}
        )
    framework = json.loads((root / "validation/framework05.json").read_text())
    assert framework["passed"] and framework["same_articulation"] and framework["rejected_invalid_commands"]
    for data in framework["native_contact_observations"].values():
        assert len(data["link_names"]) == 6 and np.isfinite(data["normal_force_world_N"]).all()
        assert data["timestamp_s"] > 0 and data["hardware_connected"] is False
    checks.append({"check": "registered_controller_and_auxiliary_observation"})
    glb = root / "repo_overlay/real2sim/embodiments" / MODEL / "preview/model.glb"
    preview = json.loads(glb.with_suffix(".provenance.json").read_text())
    assert preview["source_run"] == "native07" and preview["frame"] == 0 and preview["recorded_run_passed"]
    assert preview["urdf_sha256"] == sha(asset / "robot.urdf") and preview["glb_sha256"] == sha(glb)
    assert preview["trace_sha256"] == sha(root / "evidence/native07/frames.jsonl")
    links = Links()
    links.feed((root / "index.html").read_text())
    for url in links.paths:
        parsed = urlparse(url)
        if not parsed.scheme and parsed.path:
            if parsed.path == "validation/delivery_audit.json" and not (root / parsed.path).exists():
                continue  # This audit's result is written after all input checks.
            assert (root / unquote(parsed.path)).is_file(), url
    checks.append({"check": "recorded_preview_and_local_links"})
    manifest_path = root / "delivery_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["files"].items():
            assert sha(root / name) == digest, name
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        assert actual - set(manifest["excluded"]) == set(manifest["files"])
        checks.append({"check": "sealed_file_manifest", "files": len(manifest["files"])})
    return {
        "passed": True,
        "scope": "local robot/data/media/source integrity; not hardware calibration or grasp success",
        "checks": checks,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("delivery", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.delivery)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
