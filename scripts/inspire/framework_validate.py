"""Verify registered six-axis RobotManager control, bounds, caps and inherited cameras."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

from isaaclab.scene import InteractiveScene  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from env.robot_manager.inspire_contract import HAND_TARGET_LEAD_RAD  # noqa: E402
from env.robot_manager.robot_manager import RobotManager  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[2]
    model = "g1d_inspire_rh56dftp"
    c = json.loads((root / "Assets/Robots" / model / "model_contract.json").read_text())
    cfg = OmegaConf.load(root / "env_cfg/robot" / (model + ".yml"))
    manager = RobotManager(1, 2, cfg, 1 / 480, a.device)
    sc = manager._get_SceneCfg(1, 2)
    sc.robot0.spawn.usd_dir = str(a.output.parent / (a.output.stem + "_usd_cache"))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 480, device=a.device))
    scene = InteractiveScene(sc)
    sim.reset()
    manager.initialize(
        SimpleNamespace(scene=scene, sim=sim, cfg=SimpleNamespace(sim=SimpleNamespace(dt=1 / 480), decimation=48))
    )
    manager.reset()
    manager.set_robot_init_state()
    command, ordered, rejected = {}, [], []
    for robot in manager.robot_list:
        h = c["hands"][robot.config_key]
        values = [h["effective_active_limits"][n]["lower"] + 0.1 for n in h["joints"]]
        for bad in ([float("nan")] * 6, [1000] * 6, [0] * 5):
            try:
                robot.validate_hand_position(bad)
            except ValueError:
                rejected.append(True)
            else:
                rejected.append(False)
        command[manager.process_name(robot.arm_name)] = {
            "position": manager.get_joint(robot)[0].tolist(),
            "velocity": [0] * 7,
        }
        command[manager.process_name(robot.gripper_name)] = {"position": values, "velocity": [0] * 6}
        ordered += robot.arm_joints_name + robot.gripper_joints_name
    manager.control_manager.push([0], [[command]])
    meta = manager.control_manager.pop([0])
    for _ in range(480):
        manager.control_robot(meta)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 480)
    targets, caps, errors = [], {}, []
    for i, robot in enumerate(manager.robot_list):
        key = manager.robot_key[i]
        targets += key.data.joint_pos_target[0, robot.arm_joint_indices].cpu().tolist()
        targets += key.data.joint_pos_target[0, robot.gripper_joint_indices].cpu().tolist()
        h = c["hands"][robot.config_key]
        for name, idx in zip(robot.gripper_joints_name, robot.gripper_joint_indices):
            caps[name] = float(key.data.joint_effort_limits[0, idx])
        q = dict(zip(key.joint_names, key.data.joint_pos[0].cpu().tolist()))
        errors += [abs(q[n] - q[m["joint"]] * m["multiplier"] - m["offset"]) for n, m in h["mimic"].items()]
    expected = [v for entry in command.values() for v in entry["position"]]
    error = float(np.max(abs(np.array(targets) - expected)))
    cameras = OmegaConf.load(root / "env_cfg/camera/g1d_dexhands_camera.yml")
    manager._load_camera_cfg(cameras)
    mounted = {k: OmegaConf.to_container(v.camera) for k, v in cameras.items() if k != "annotator"}
    result = {
        "model": model,
        "action_dimension": len(ordered),
        "joint_order": ordered,
        "target_readback_error_rad": error,
        "mimic_error_rad": max(errors),
        "same_articulation": manager.robot_key[0] is manager.robot_key[1],
        "camera_configuration": mounted,
        "rejected_invalid_commands": all(rejected),
        "hand_target_lead_rad": HAND_TARGET_LEAD_RAD,
        "hand_effort_caps_Nm": caps,
        "body_observation_keys": list(manager.robot_list[0].body_dims),
        "hardware_connected": False,
        "native_contact_observations": {
            robot.config_key: manager.get_tactile_contact(robot)[0] for robot in manager.robot_list
        },
        "passed": bool(
            len(ordered) == 26
            and error < 1e-7
            and max(errors) < 0.005
            and all(rejected)
            and all(abs(v - 1) < 1e-7 for v in caps.values())
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
