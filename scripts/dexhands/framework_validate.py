"""Exercise registered RobotManager actions and inherited head-camera descriptors, simulation only."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

from isaaclab.scene import InteractiveScene  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from env.robot_manager.robot_manager import RobotManager  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "env_cfg/robot" / (a.model + ".yml"))
    manager = RobotManager(1, 2, cfg, 1 / 240, a.device)
    scene_cfg = manager._get_SceneCfg(1, 2)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240, device=a.device))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    manager.initialize(
        SimpleNamespace(scene=scene, sim=sim, cfg=SimpleNamespace(sim=SimpleNamespace(dt=1 / 240), decimation=24))
    )
    manager.reset()
    manager.set_robot_init_state()
    command = {}
    ordered = []
    for robot in manager.robot_list:
        command[manager.process_name(robot.arm_name)] = {
            "position": manager.get_joint(robot)[0].tolist(),
            "velocity": [0.0] * 7,
        }
        values = np.linspace(0.005, 0.025, len(robot.gripper_joints_name)).tolist()
        command[manager.process_name(robot.gripper_name)] = {"position": values, "velocity": [0.0] * len(values)}
        ordered += robot.arm_joints_name + robot.gripper_joints_name
    # Public queue fills omitted body commands with their previous targets.
    manager.control_manager.push([0], [[command]])
    manager.control_robot(manager.control_manager.pop([0]))
    for _ in range(240):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 240)
    actual_targets = []
    for robot in manager.robot_list:
        key = manager.robot_key[manager.robot_list.index(robot)]
        actual_targets += key.data.joint_pos_target[0, robot.arm_joint_indices].cpu().tolist()
        actual_targets += key.data.joint_pos_target[0, robot.gripper_joint_indices].cpu().tolist()
    expected = [v for entry in command.values() for v in entry["position"]]
    error = float(np.max(np.abs(np.array(actual_targets) - expected)))
    cameras = OmegaConf.load(root / "env_cfg/camera/g1d_dexhands_camera.yml")
    manager._load_camera_cfg(cameras)
    mounted = {k: OmegaConf.to_container(v.camera) for k, v in cameras.items() if k != "annotator"}
    assert set(mounted) == {"cam_head", "cam_head_right", "cam_third"}
    assert all("head_link" in mounted[k]["mount_link"] for k in ["cam_head", "cam_head_right"])
    contract = json.loads((root / "Assets/Robots" / a.model / "model_contract.json").read_text())
    motor_caps = {}
    for robot in manager.robot_list:
        key = manager.robot_key[manager.robot_list.index(robot)]
        for name, index in zip(robot.gripper_joints_name, robot.gripper_joint_indices):
            actual = float(key.data.joint_effort_limits[0, index])
            expected_cap = next(h["limits"][name]["effort"] for h in contract["hands"].values() if name in h["limits"])
            assert abs(actual - expected_cap) < 1e-6
            motor_caps[name] = actual
    result = {
        "model": a.model,
        "action_dimension": len(ordered),
        "joint_order": ordered,
        "target_readback_error_rad": error,
        "same_articulation": manager.robot_key[0] is manager.robot_key[1],
        "camera_configuration": mounted,
        "verified_hand_effort_caps_Nm": motor_caps,
        "body_observation_keys": list(manager.robot_list[0].body_dims),
        "no_planner_loaded": "env.planner_manager.curobo_planner" not in sys.modules,
        "hardware_connected": False,
        "native_contact_observations": {
            robot.config_key: manager.get_tactile_contact(robot)[0] for robot in manager.robot_list
        },
        "passed": error < 1e-7 and len(ordered) == contract["action_dimension"],
    }
    for robot in manager.robot_list:
        obs = result["native_contact_observations"][robot.config_key]
        assert len(obs["link_names"]) == 5
        assert np.isfinite(obs["normal_force_world_N"]).all()
        assert abs(obs["timestamp_s"] - sim.current_time) < 1e-8
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
