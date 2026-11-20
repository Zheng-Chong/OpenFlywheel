"""Native complete-robot RH56DFTP axis, mimic, limit and return checks with three views."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app

import os  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402

from env.robot_manager.inspire_contract import HAND_TARGET_LEAD_RAD  # noqa: E402
from env.robot_manager.robot_config.g1d_inspire_rh56dftp import get_robot_config  # noqa: E402
from scripts.dexhands.kinematics import fk  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "g1d_inspire_rh56dftp"
DT = 1 / 480


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aim(eye, target):
    f = np.asarray(target) - eye
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f)
    left /= np.linalg.norm(left)
    q = Rotation.from_matrix(np.stack([f, left, np.cross(f, left)], axis=1)).as_quat()
    return q[[3, 0, 1, 2]]


def main():
    a.output.mkdir(parents=True, exist_ok=False)
    asset = ROOT / "Assets/Robots" / MODEL
    contract = json.loads((asset / "model_contract.json").read_text())
    frozen = a.output / "execution_inputs"
    frozen.mkdir()
    for source in (
        Path(__file__),
        asset / "model_contract.json",
        ROOT / "env/robot_manager/robot_config/g1d_inspire_rh56dftp.py",
    ):
        shutil.copyfile(source, frozen / source.name)
    measured_tree = ET.parse(asset / "robot.urdf").getroot()
    for j in measured_tree.findall("joint"):
        if j.find("mimic") is not None:
            j.remove(j.find("mimic"))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=a.device))
    cfg = get_robot_config().replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0, 0, 0.1115)
    cfg.init_state.joint_pos[".*_elbow_joint"] = -0.6
    robot = Articulation(cfg)
    floor = sim_utils.CuboidCfg(
        size=(8, 8, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.48, 0.53)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.01))
    light = sim_utils.DomeLightCfg(intensity=1100)
    light.func("/World/Light", light)
    cameras = [
        (name, Camera(prim_path="/World/Camera_" + name, resolution=(960, 720) if name == "whole" else (640, 640)))
        for name in ("whole", "left", "right")
    ]
    sim.reset()
    q0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    names = robot.joint_names
    ids = {s: [names.index(n) for n in h["joints"]] for s, h in contract["hands"].items()}
    active = [i for v in ids.values() for i in v]
    for _ in range(480):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
    for name, cam in cameras:
        cam.initialize()
        cam.prim.GetAttribute("focalLength").Set(16.0)
        cam.set_clipping_range(0.001, 20)
        if name == "whole":
            target = np.array([0.02, 0, 0.72])
            eye = target + [1.45, -1.3, 0.72]
        else:
            bid = [robot.body_names.index(n) for n in contract["hands"][name]["links"]]
            target = robot.data.body_pos_w[0, bid].cpu().numpy().mean(axis=0)
            eye = target + np.array([0.26, -0.28 if name == "left" else 0.28, 0.19])
        cam.set_world_pose(eye, aim(eye, target), camera_axes="world")
        (a.output / name).mkdir()
    for _ in range(8):
        sim.render()
    phases = [("open", q0.clone())]
    if not a.smoke:
        for k in range(6):
            goal = q0.clone()
            for side in ids:
                goal[0, ids[side][k]] += 0.15
            phases.extend([("axis_" + str(k), goal), ("return_" + str(k), q0.clone())])
        # Full manufacturer-feasible actuator endpoints; report collisions or stalls honestly.
        for label, fraction in [("half_range", 0.5), ("closed_range", 1.0), ("return_open", 0.0)]:
            goal = q0.clone()
            for side, h in contract["hands"].items():
                for n in h["joints"]:
                    lim = h["effective_active_limits"][n]
                    goal[0, names.index(n)] = lim["lower"] + fraction * (lim["upper"] - lim["lower"])
            phases.append((label, goal))
    rows, reports, prev = [], [], q0.clone()
    fk_peak = coupling_peak = 0.0
    motion_steps = 480 if a.smoke else 960
    steps = motion_steps + 240  # 0.5 s hold after the original motion window.
    trace = (a.output / "frames.jsonl").open("w")
    for label, goal in phases:
        errors = []
        for step in range(steps):
            alpha = min(1, (step + 1) / (motion_steps * 0.65))
            alpha = alpha * alpha * (3 - 2 * alpha)
            target = prev + (goal - prev) * alpha
            for h in contract["hands"].values():
                for n, m in h["mimic"].items():
                    target[0, names.index(n)] = target[0, names.index(m["joint"])] * m["multiplier"] + m["offset"]
            requested = target.clone()
            target[:, active] = torch.clamp(
                target[:, active],
                robot.data.joint_pos[:, active] - HAND_TARGET_LEAD_RAD,
                robot.data.joint_pos[:, active] + HAND_TARGET_LEAD_RAD,
            )
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
            if step >= steps * 0.8:
                errors.append(torch.abs(robot.data.joint_pos[0, active] - goal[0, active]).cpu().numpy())
            if step % 48:
                continue
            q = robot.data.joint_pos[0].cpu().numpy()
            predicted = fk(measured_tree, dict(zip(names, q)))
            positions = robot.data.body_pos_w[0].cpu().numpy()
            err = max(
                np.linalg.norm(positions[k] - predicted[n][:3, 3] - cfg.init_state.pos)
                for k, n in enumerate(robot.body_names)
                if n in predicted
            )
            coupling = max(
                abs(q[names.index(n)] - q[names.index(m["joint"])] * m["multiplier"] - m["offset"])
                for h in contract["hands"].values()
                for n, m in h["mimic"].items()
            )
            fk_peak, coupling_peak = max(fk_peak, float(err)), max(coupling_peak, float(coupling))
            sim.render()
            frame = len(rows)
            for name, cam in cameras:
                Image.fromarray(cam.get_rgba()[..., :3]).save(a.output / name / f"{frame:05d}.jpg", quality=92)
            row = {
                "frame": frame,
                "time_s": (len(reports) * steps + step + 1) * DT,
                "phase": label,
                "joint_position_rad": q.tolist(),
                "target_rad": target[0].cpu().tolist(),
                "requested_target_rad": requested[0].cpu().tolist(),
                "fk_max_error_m": float(err),
                "mimic_max_error_rad": float(coupling),
            }
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            trace.flush()
        reports.append(
            {
                "phase": label,
                "max_active_error_rad": float(np.max(errors)),
                "per_active_joint_error_rad": dict(zip([names[i] for i in active], np.max(errors, axis=0).tolist())),
            }
        )
        prev = goal.clone()
        print(json.dumps(reports[-1]), flush=True)
    trace.close()
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    native_mimics = [
        str(p.GetPath()) for p in stage.Traverse() if any("MimicJoint" in s for s in p.GetAppliedSchemas())
    ]
    limit_violation = max(
        max(
            float(v["lower"]) - r["joint_position_rad"][names.index(n)],
            r["joint_position_rad"][names.index(n)] - float(v["upper"]),
            0,
        )
        for r in rows
        for h in contract["hands"].values()
        for n, v in h["limits"].items()
    )
    # Exclude full endpoint poses from the bounded-axis acceptance; retain their own measured status.
    bounded = max(r["max_active_error_rad"] for r in reports if r["phase"] not in ("half_range", "closed_range"))
    endpoint = {r["phase"]: r["max_active_error_rad"] for r in reports if r["phase"] in ("half_range", "closed_range")}
    finite = all(np.isfinite(r["joint_position_rad"]).all() for r in rows)
    result = {
        "model": MODEL,
        "smoke": a.smoke,
        "frames": len(rows),
        "fps": 10,
        "simulation_dt_s": DT,
        "urdf_sha256": digest(asset / "robot.urdf"),
        "joint_names": names,
        "body_names": robot.body_names,
        "gravity_enabled": True,
        "self_collision_enabled": True,
        "native_mimic_constraints": native_mimics,
        "mimic_max_error_rad": coupling_peak,
        "fk_max_error_m": fk_peak,
        "limit_violation_rad": float(limit_violation),
        "bounded_axis_error_rad": bounded,
        "endpoint_errors_rad": endpoint,
        "endpoint_tracking_accepted": all(v < 0.04 for v in endpoint.values()),
        "endpoint_note": (
            "Simultaneous mechanical endpoint commands encounter self-contact; "
            "do not treat stalls as free-space tracking or hardware angle calibration."
        ),
        "phases": reports,
        "hardware_connected": False,
        "hand_target_lead_rad": HAND_TARGET_LEAD_RAD,
        "execution_input_sha256": {p.name: digest(p) for p in frozen.iterdir()},
        "thresholds": {"fk_m": 0.0005, "mimic_rad": 0.005, "bounded_tracking_rad": 0.04, "limit_rad": 0.01},
        "scope": (
            "unloaded complete-robot kinematics and native mimic validation; "
            "endpoint stalls reported separately; no real calibration or grasp claim"
        ),
        "passed": bool(
            finite
            and len(native_mimics) == 12
            and coupling_peak < 0.005
            and fk_peak < 0.0005
            and bounded < 0.04
            and limit_violation < 0.01
        ),
    }
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        a.output / "trajectory.npz",
        joint_names=np.array(names),
        q=np.array([r["joint_position_rad"] for r in rows]),
        target=np.array([r["target_rad"] for r in rows]),
        time_s=[r["time_s"] for r in rows],
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("phases", "joint_names", "body_names", "native_mimic_constraints")
            }
        ),
        flush=True,
    )


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
