"""Create a local review from explicitly selected independent simulation recordings."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def encode_rgb(run, name, target, fps):
    if not target.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(run / name / "%05d.jpg"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                "-movflags",
                "+faststart",
                str(target),
            ],
            check=True,
        )


def encode_depth(dataset, target, fps):
    if target.exists():
        return
    height, width = dataset.shape[1:]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(target),
        ],
        stdin=subprocess.PIPE,
    )
    for depth in dataset:
        valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
        t = np.nan_to_num(np.clip((depth - 0.1) / 2, 0, 1), nan=1, posinf=1)
        rgb = np.stack([np.clip(1.5 - abs(4 * t - j), 0, 1) * 255 for j in (3, 2, 1)], axis=-1).astype("uint8")
        rgb[~valid] = 0
        process.stdin.write(rgb.tobytes())
    process.stdin.close()
    assert process.wait() == 0


def export(root, selection, output):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for model in selection["models"]:
        entry = {"model": model["model"], "title": model["title"], "runs": {}}
        framework = root / (model["model"] + "_framework01.json")
        entry["interface"] = json.loads(framework.read_text())
        for kind in ("motion", "grasp", "contact"):
            name = model[kind]
            run = root / "runs" / name
            report = json.loads((run / "report.json").read_text())
            audit = json.loads((run / "audit.json").read_text())
            assert report["model"] == model["model"] == audit["model"]
            rows = [json.loads(s) for s in (run / "frames.jsonl").read_text().splitlines()]
            destination = output / name
            destination.mkdir(exist_ok=True)
            for filename in ("report.json", "audit.json"):
                shutil.copyfile(run / filename, destination / filename)
            source = root / "episodes" / (name + ".h5") if kind == "contact" else run / "modalities.h5"
            link = destination / "modalities.h5"
            if link.is_symlink():
                assert link.resolve() == source.resolve(), "Refusing to replace another recording"
            elif not link.exists():
                link.symlink_to(source)
            lite = [
                {
                    **{
                        k: r[k]
                        for k in (
                            "frame",
                            "time_s",
                            "phase",
                            "force_world_N",
                            "normal_force_world_N",
                            "friction_force_world_N",
                            "joint_position_rad",
                            "camera_sensor_time_s",
                        )
                    },
                    "requested_target_rad": r.get(
                        "requested_target_rad", r.get("target_rad", r.get("target_joint_rad"))
                    ),
                    "object_states_si": r.get("object_states_si"),
                    "shape": r.get("shape"),
                    "fixture_depth_m": r.get("fixture_depth_m"),
                }
                for r in rows
            ]
            (destination / "frames.json").write_text(json.dumps(lite, separators=(",", ":")))
            streams = {}
            fps = report["fps"]
            views = list(rows[0]["camera_sensor_time_s"])
            for view in views:
                encode_rgb(run, view, destination / (view + ".mp4"), fps)
                streams[view] = {"rgb": name + "/" + view + ".mp4", "depth": None}
            if kind != "contact":
                with h5py.File(source) as f:
                    for view in views:
                        target = destination / (view + "_depth.mp4")
                        encode_depth(f["camera/" + view + "/depth_m"], target, fps)
                        streams[view]["depth"] = name + "/" + target.name
            elif (run / "tactile.npz").exists():
                tactile = destination / "tactile"
                if not (tactile / "depth.mp4").exists():
                    subprocess.run(
                        [
                            __import__("sys").executable,
                            str(Path(__file__).with_name("render_tactile.py")),
                            "--run",
                            str(run),
                            "--output",
                            str(tactile),
                        ],
                        check=True,
                    )
                streams["tacmap"] = {"rgb": name + "/tactile/depth.mp4", "depth": None}
            entry["runs"][kind] = {
                "name": name,
                "frames": len(rows),
                "fps": fps,
                "playback_duration_s": len(rows) / fps,
                "physical_duration_s": rows[-1]["time_s"] - rows[0]["time_s"],
                "streams": streams,
                "channels": report["channels"],
                "joint_names": report["joint_names"],
                "audit": audit,
                "report": name + "/report.json",
                "audit_file": name + "/audit.json",
                "data": name + "/modalities.h5",
                "hdf5_sha256": digest(source),
                "trace": name + "/frames.json",
                "force_composition": report.get("force_composition", "normal_plus_friction"),
                "joint_units": ["m" if n.startswith("LZ_") else "rad" for n in report["joint_names"]],
            }
            print(name, len(rows), "frames exported", flush=True)
        entries.append(entry)
    summary = {"models": entries, "scope": selection["scope"], "schema": "four_hand_review.v1"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    shutil.copyfile(Path(__file__).with_name("four_hand_review.html"), output / "index.html")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, json.loads(args.selection.read_text()), args.output)
