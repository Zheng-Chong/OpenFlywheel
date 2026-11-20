"""Native Isaac Lab joint/FK validation and three-view recording of one complete robot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--thumb-review", action="store_true", help="Review each thumb axis, opposition and return")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import os  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from kinematics import fk  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402

from env.robot_manager.robot_config._g1d_dexhands import get_config  # noqa: E402

DT = 1 / 240
ROOT = Path(__file__).resolve().parents[2]


def look(eye, target):
    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    q = Rotation.from_matrix(np.stack([forward, left, up], axis=1)).as_quat()
    return q[[3, 0, 1, 2]]


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    model_dir = ROOT / "Assets/Robots" / args.model
    contract = json.loads((model_dir / "model_contract.json").read_text())
    frozen = args.output / "execution_inputs"
    frozen.mkdir()
    for source in (
        Path(__file__),
        ROOT / "env/robot_manager/robot_config/_g1d_dexhands.py",
        model_dir / "model_contract.json",
        model_dir / "vendor_dynamics.json",
    ):
        shutil.copyfile(source, frozen / source.name)
    tree = ET.parse(model_dir / "robot.urdf").getroot()
    cfg = get_config(args.model).replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0.0, 0.0, 0.1115)
    cfg.init_state.joint_pos[".*_elbow_joint"] = -0.6
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    floor = sim_utils.CuboidCfg(
        size=(8, 8, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.57, 0.61)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.01))
    dome = sim_utils.DomeLightCfg(intensity=800)
    dome.func("/World/Dome", dome)
    sun = sim_utils.DistantLightCfg(intensity=2500, angle=5)
    sun.func("/World/Sun", sun, orientation=(0.92388, 0.38268, 0, 0))
    robot = Articulation(cfg)
    cameras = []
    for name, resolution in [("whole", (960, 720)), ("left", (640, 640)), ("right", (640, 640))]:
        camera = Camera(prim_path="/World/Camera_" + name, resolution=resolution)
        cameras.append((name, camera))
    sim.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    q0 = robot.data.default_joint_pos.clone()
    for _ in range(240):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
    for name, camera in cameras:
        camera.initialize()
        camera.prim.GetAttribute("focalLength").Set(16.0)
        camera.set_clipping_range(0.001, 20.0)
        if name == "whole":
            target = np.array([0.02, 0.0, 0.72])
            eye = target + np.array([1.45, -1.3, 0.72])
        else:
            mount = contract["hands"][name]["root"]
            ids, _ = robot.find_bodies(mount)
            wrist = robot.data.body_pos_w[0, ids[0]].cpu().numpy()
            tip_names = [
                n
                for n in robot.body_names
                if (n.startswith("l_" if name == "left" else "r_") and n.endswith("_tip"))
                or (n.startswith(name + "_") and n.endswith("_fingertip"))
            ]
            tip_ids = [robot.body_names.index(n) for n in tip_names]
            tip_mean = robot.data.body_pos_w[0, tip_ids].cpu().numpy().mean(axis=0)
            target = (wrist + tip_mean) / 2
            eye = target + np.array([0.32, -0.26 if name == "left" else 0.26, 0.17])
        camera.set_world_pose(eye, look(eye, target), camera_axes="world")
        (args.output / name).mkdir()
    for _ in range(10):
        sim.render()
    joint_names = robot.joint_names
    hand_ids = {s: [joint_names.index(n) for n in h["joints"]] for s, h in contract["hands"].items()}
    # Each hand axis is independently commanded and returned before the next axis.
    # ±0.15 rad is a bounded motion check, not a full workspace claim.
    phases = [("zero", q0.clone())]
    if args.thumb_review:
        for axis in range(5 if "sharpa" in args.model else 4):
            goal = q0.clone()
            for side, ids in hand_ids.items():
                name = contract["hands"][side]["joints"][axis]
                assert "thumb" in name
                goal[0, ids[axis]] = 0.15
            phases += [("thumb_axis_" + str(axis), goal), ("thumb_return_" + str(axis), q0.clone())]
        opposed = q0.clone()
        for side, ids in hand_ids.items():
            # Explicit demonstration targets in manufacturer coordinates.
            # This is not a calibrated natural-open or hardware home pose.
            for idx, name in zip(ids, contract["hands"][side]["joints"]):
                if "sharpa" in args.model:
                    if name.endswith("thumb_CMC_FE"):
                        opposed[0, idx] = 1.0
                elif "thumb" in name and not name.endswith("_abd"):
                    opposed[0, idx] = 0.4
        phases += [("thumb_opposition_demo", opposed), ("thumb_unload", q0.clone())]
    elif not args.smoke:
        count = max(len(x) for x in hand_ids.values())
        for axis in range(count):
            goal = q0.clone()
            for side, ids in hand_ids.items():
                name = contract["hands"][side]["joints"][axis]
                limit = contract["hands"][side]["limits"][name]
                amplitude = 0.04 if "sharpa" in args.model and name.endswith("_AA") else 0.15
                value = (
                    min(amplitude, limit["upper"] * 0.4)
                    if limit["upper"] > 0
                    else max(-amplitude, limit["lower"] * 0.4)
                )
                goal[0, ids[axis]] = value
            phases += [("joint_" + str(axis), goal), ("return_" + str(axis), q0.clone())]
        goal = q0.clone()
        for side, ids in hand_ids.items():
            for index, name in zip(ids, contract["hands"][side]["joints"]):
                if any(w in name.lower() for w in ("flex", "pip", "dip", "_fe", "_ip")):
                    goal[0, index] = min(0.4, contract["hands"][side]["limits"][name]["upper"] * 0.5)
        phases += [("curl", goal), ("unload", q0.clone())]
    records = []
    frame = 0
    phase_reports = []
    fk_max = 0.0
    finite = True
    torque_peak = np.zeros(robot.num_joints)
    prev = q0.clone()
    steps = 480 if args.smoke else 240
    trace = open(args.output / "frames.jsonl", "w")
    for phase_index, (phase, goal) in enumerate(phases):
        errors = []
        for step in range(steps):
            alpha = min(1.0, (step + 1) / (steps * 0.65))
            alpha = alpha * alpha * (3 - 2 * alpha)
            target = prev + (goal - prev) * alpha
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
            torque_peak = np.maximum(torque_peak, np.abs(robot.data.applied_torque[0].cpu().numpy()))
            if step >= steps * 0.8:
                errors.append(torch.abs(robot.data.joint_pos - goal).cpu().numpy()[0])
            if step % 24:
                continue
            q = robot.data.joint_pos.cpu().numpy()[0]
            finite &= bool(np.isfinite(q).all())
            positions = robot.data.body_pos_w.cpu().numpy()[0]
            predicted = fk(tree, dict(zip(joint_names, q)))
            fk_error = max(
                float(np.linalg.norm(positions[k] - predicted[n][:3, 3] - np.array(cfg.init_state.pos)))
                for k, n in enumerate(robot.body_names)
                if n in predicted
            )
            fk_max = max(fk_max, fk_error)
            sim.render()
            for name, camera in cameras:
                rgb = camera.get_rgba()[..., :3]
                if rgb.size == 0:
                    raise ValueError("Empty render frame")
                Image.fromarray(rgb).save(args.output / name / f"{frame:05d}.jpg", quality=92)
            row = {
                "frame": frame,
                "time_s": (phase_index * steps + step + 1) * DT,
                "phase": phase,
                "joint_position_rad": q.tolist(),
                "target_rad": target[0].cpu().numpy().tolist(),
                "fk_max_error_m": fk_error,
            }
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            records.append(row)
            frame += 1
        errors = np.asarray(errors)
        phase_reports.append(
            {
                "phase": phase,
                "max_steady_error_rad": float(errors.max()),
                "mean_steady_error_rad": float(errors.mean()),
                "per_joint_max_error_rad": dict(zip(joint_names, errors.max(axis=0).tolist())),
            }
        )
        prev = goal
        print(
            json.dumps({"phase": phase, "error_rad": phase_reports[-1]["max_steady_error_rad"], "frames": frame}),
            flush=True,
        )
    trace.close()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    robot.reset()
    robot.update(0.0)
    reset_error = float(torch.abs(robot.data.joint_pos - q0).max().item())
    np.savez_compressed(
        args.output / "trajectory.npz",
        joint_names=np.array(joint_names),
        q=np.array([r["joint_position_rad"] for r in records]),
        target=np.array([r["target_rad"] for r in records]),
        time_s=np.array([r["time_s"] for r in records]),
    )
    worst = max(p["max_steady_error_rad"] for p in phase_reports)
    report = {
        "model": args.model,
        "smoke": args.smoke,
        "thumb_review": args.thumb_review,
        "urdf_sha256": hashlib.sha256((model_dir / "robot.urdf").read_bytes()).hexdigest(),
        "frames": frame,
        "fps": 10,
        "simulation_dt_s": DT,
        "gravity_enabled": True,
        "self_collision_enabled": True,
        "joint_names": joint_names,
        "body_names": robot.body_names,
        "phases": phase_reports,
        "fk_max_error_m": fk_max,
        "reset_error_rad": reset_error,
        "finite": finite,
        "max_steady_error_rad": worst,
        "peak_actuator_torque_estimate_Nm": dict(zip(joint_names, torque_peak.tolist())),
        "torque_scope": "IsaacLab applied_torque buffer; implicit drives estimated, not a measured load",
        "arm_effort_limits_Nm": cfg.actuators["arm"].effort_limit_sim,
        "execution_input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()},
        "thresholds": {"fk_m": 0.0005, "reset_rad": 1e-6, "steady_error_rad": 0.035},
        "passed": finite and fk_max < 0.0005 and reset_error < 1e-6 and worst < 0.035,
        "scope": (
            "bounded unloaded thumb review with full robot; mechanical zero and explicit opposition demo, "
            "not natural-open calibration"
            if args.thumb_review
            else "bounded unloaded joint motions with full robot; "
            "no hardware calibration or manipulation success claimed"
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("phases", "joint_names", "body_names")}), flush=True)


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
