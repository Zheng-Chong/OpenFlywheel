"""Read-only audit of an unpacked delivery: immutable sources, physical assets, episodes and media."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    checks = []
    sources = json.loads((root / "source_manifest.json").read_text())
    for name, record in sources.items():
        for file, digest in record["files"].items():
            assert sha(root / "vendor" / name / file) == digest, (name, file)
    checks.append({"check": "manufacturer_sources", "files": sum(len(r["files"]) for r in sources.values())})
    for asset in (root / "repo_overlay/real2sim/robot").glob("g1d_*"):
        manifest = json.loads((asset / "sha256.json").read_text())
        for file, digest in manifest.items():
            assert sha(asset / file) == digest, (asset.name, file)
        checks.append({"check": "full_robot_asset", "model": asset.name, "files": len(manifest)})
    assert len(list((root / "datasets").glob("*.h5"))) == 6
    assert len(list((root / "validation").glob("*_framework03.json"))) == 3
    episodes = list((root / "datasets").glob("*.h5")) + list((root / "diagnostics").glob("*.h5"))
    for path in sorted(episodes):
        evidence = json.loads(path.with_suffix(".audit.json").read_text())
        run = root / "evidence" / evidence["run"]
        if path.parent.name == "diagnostics":
            run = root / "diagnostics" / evidence["run"]
        report = json.loads((run / "report.json").read_text())
        assert report["passed"]
        asset = root / "repo_overlay/real2sim/robot" / report["model"]
        assert report["urdf_sha256"] == sha(asset / "robot.urdf"), (run.name, "recorded asset is outdated")
        for name, digest in evidence["raw_source_sha256"].items():
            assert sha(run / name) == digest, (run.name, name)
        for name, digest in report["execution_input_sha256"].items():
            assert sha(run / "execution_inputs" / name) == digest
        trace = [json.loads(r) for r in (run / "frames.jsonl").read_text().splitlines()]
        with h5py.File(path) as f:
            assert sha(path) == evidence["hdf5_sha256"]
            assert len(f["time_s"]) == report["frames"] == len(trace)
            assert np.all(np.diff(f["time_s"][:]) > 0)
            assert f["action/position_rad"].shape[1] in (54, 58)
            assert not f.attrs["hardware_connected"]
            np.testing.assert_array_equal(f["robot/joint_position_si"][:], [r["joint_position_rad"] for r in trace])
            if "force_world_N" in trace[0]:
                np.testing.assert_array_equal(f["tactile/force_world_N"][:], [r["force_world_N"] for r in trace])
                np.testing.assert_array_equal(f["tactile/force_link_N"][:], [r["force_link_N"] for r in trace])
                force = np.linalg.norm([r["force_world_N"] for r in trace], axis=-1)
                for shape in ["sphere", "flat", "cylinder"]:
                    rows = [i for i, r in enumerate(trace) if r["shape"] == shape]
                    peak = force[rows].max(0)
                    assert np.all(peak > 0.001)
                    np.testing.assert_allclose(peak, report["per_shape_peak_N"][shape], atol=1e-10)
                assert max(max(r["pair_opposition_error_N"]) for r in trace) < 0.01
                release = force[[i for i, r in enumerate(trace) if r["phase"] in ["release", "recovered"]]]
                assert release.max() < 0.01
            else:
                assert max(r["fk_max_error_m"] for r in trace) < 0.0005
                assert report["max_steady_error_rad"] < 0.035
                assert report["reset_error_rad"] < 1e-6
        for video in run.glob("*.mp4"):
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
            assert int(info["nb_read_frames"]) == len(trace), (video, info)
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(video), "-f", "null", "-"], check=True)
        checks.append({"check": "single_recording_data_media", "run": run.name, "frames": len(trace)})
    for path in (root / "validation").glob("*_framework03.json"):
        r = json.loads(path.read_text())
        assert r["passed"] and r["same_articulation"] and r["no_planner_loaded"]
        assert len(r["verified_hand_effort_caps_Nm"]) in (40, 44)
        checks.append({"check": "platform_order_and_motor_caps", "model": r["model"]})
    return {
        "passed": True,
        "checks": checks,
        "hardware_connected": False,
        "scope": "File integrity, raw trace acceptance and data/video read-back; no physical hardware validation.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("delivery", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = audit(a.delivery)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
