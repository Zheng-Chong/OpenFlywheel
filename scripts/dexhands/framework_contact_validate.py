"""Load registered contact links and read real solver responses through RobotManager."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--fixtures", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args()
a.headless = True
app = AppLauncher(a).app

from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402

from env.robot_manager.robot_manager import RobotManager  # noqa: E402

DT = 1 / 480


def main():
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(a.fixtures.read_text())
    fixture = fixture[a.model] if isinstance(fixture, dict) else fixture
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=a.device, enable_scene_query_support=True))
    manager = RobotManager(1, 2, OmegaConf.load(root / "env_cfg/robot" / (a.model + ".yml")), DT, a.device)
    scene = InteractiveScene(manager._get_SceneCfg(1, 2))
    links = [n for robot in manager.robot_list for n in robot.tactile_contact_links]
    probes = []
    independent = []
    for i, link in enumerate(links):
        cfg = sim_utils.SphereCfg(
            radius=0.002,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        )
        probes.append(RigidObject(RigidObjectCfg(prim_path=f"/World/Probe_{i}", spawn=cfg)))
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.5, compliant_contact_stiffness=280, compliant_contact_damping=20
        )
        path = "/World/envs/env_0/robot0/" + link
        independent.append(
            ContactSensor(
                ContactSensorCfg(prim_path=path, filter_prim_paths_expr=[f"/World/Probe_{i}"], update_period=0)
            )
        )
        collisions = sim.stage.GetPrimAtPath(path + "/collisions")
        if collisions:
            collisions.SetInstanceable(False)
        material.func(path + "/ValidationMaterial", material)
        sim_utils.bind_physics_material(path, path + "/ValidationMaterial")
    sim.reset()
    manager.initialize(
        SimpleNamespace(scene=scene, sim=sim, cfg=SimpleNamespace(sim=SimpleNamespace(dt=DT), decimation=1))
    )
    manager.reset()
    manager.set_robot_init_state()
    # Use the registered public queue to hold its own valid initial targets.
    command = {}
    for robot, key in zip(manager.robot_list, manager.robot_key):
        for label, ids in (
            (robot.arm_name, robot.arm_joint_indices),
            (robot.gripper_name, robot.gripper_joint_indices),
        ):
            values = key.data.joint_pos_target[0, ids].cpu().tolist()
            command[manager.process_name(label)] = {"position": values, "velocity": [0.0] * len(values)}
    manager.control_manager.push([0], [[command]])
    manager.control_robot(manager.control_manager.pop([0]))
    for i, probe in enumerate(probes):
        probe.write_root_pose_to_sim(torch.tensor([[3, i * 0.03, 3, 1, 0, 0, 0]], device=sim.device))
    for _ in range(480):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    references = []
    surface_queries = []
    for robot, key in zip(manager.robot_list, manager.robot_key):
        for link in robot.tactile_contact_links:
            item = next(p for p in fixture if p["link"] == link)
            i = key.body_names.index(link)
            rotation = Rotation.from_quat(key.data.body_quat_w[0, i].cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
            position = key.data.body_pos_w[0, i].cpu().numpy() + rotation @ item["point_m"]
            normal = rotation @ np.asarray(item["normal"])
            normal /= np.linalg.norm(normal)
            hit = query.raycast_closest(tuple(position + normal * 0.015), tuple(-normal), 0.03)
            surface_queries.append({"link": link, "hit": dict(hit)})
            if (
                not hit["hit"]
                or str(hit["rigidBody"]).split("/")[-1] != link
                or hit.get("material") != "/World/envs/env_0/robot0/" + link + "/ValidationMaterial"
            ):
                raise ValueError("Contact fixture did not see intended native link: " + link)
            references.append((np.asarray(hit["position"]), np.asarray(hit["normal"])))
    local_references = []
    for robot, key in zip(manager.robot_list, manager.robot_key):
        for link in robot.tactile_contact_links:
            i = key.body_names.index(link)
            rotation = Rotation.from_quat(key.data.body_quat_w[0, i].cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
            point, normal = references[len(local_references)]
            local_references.append(
                (key, i, rotation.T @ (point - key.data.body_pos_w[0, i].cpu().numpy()), rotation.T @ normal)
            )
    records = []

    def observe(phase):
        observations = {robot.config_key: manager.get_tactile_contact(robot)[0] for robot in manager.robot_list}
        for robot in manager.robot_list:
            obs = observations[robot.config_key]
            assert obs["schema"] == "robodojo.native_normal_contact.v1"
            assert abs(obs["timestamp_s"] - sim.current_time) < 1e-8
            sensor = robot.tactile_contact_sensor
            indices = [sensor.body_names.index(n) for n in obs["link_names"]]
            assert np.array_equal(
                np.asarray(obs["normal_force_world_N"]), sensor.data.net_forces_w[0, indices].cpu().numpy()
            )
        independent_forces = []
        for sensor in independent:
            sensor.update(DT, force_recompute=True)
            independent_forces.append(sensor.data.force_matrix_w[0, 0, 0].cpu().tolist())
        records.append(
            {
                "phase": phase,
                "time_s": sim.current_time,
                "observations": observations,
                "independent_filtered_normal_force_world_N": independent_forces,
                "actual_probe_pose_xyzw": [p.root_physx_view.get_transforms()[0].cpu().tolist() for p in probes],
            }
        )

    for phase, gap in (("baseline", 0.002), ("loaded", -0.0008), ("recovered", 0.002)):
        for probe, (key, index, local_point, local_normal) in zip(probes, local_references):
            rotation = Rotation.from_quat(key.data.body_quat_w[0, index].cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
            point = key.data.body_pos_w[0, index].cpu().numpy() + rotation @ local_point
            normal = rotation @ local_normal
            pose = torch.tensor(
                np.r_[point + normal * (0.002 + gap), [1, 0, 0, 0]][None], dtype=torch.float32, device=sim.device
            )
            probe.write_root_pose_to_sim(pose)
        for tick in range(240):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            if tick % 24 == 0 or tick == 239:
                observe(phase)

    def vectors(record):
        return np.concatenate([v["normal_force_world_N"] for v in record["observations"].values()])

    baseline = vectors([r for r in records if r["phase"] == "baseline"][-1])
    loaded = np.array([vectors(r) for r in records if r["phase"] == "loaded"])
    released = vectors(records[-1])
    peak = np.linalg.norm(loaded, axis=-1).max(axis=0)
    recovery = float(np.linalg.norm(released, axis=-1).max())
    delta = np.linalg.norm(loaded - baseline, axis=-1).max(axis=0)
    recovery_delta = float(np.linalg.norm(released - baseline, axis=-1).max())
    result = {
        "model": a.model,
        "passed": bool(delta.min() > 0.001 and recovery_delta < 0.01),
        "action_dimension": sum(len(v["position"]) for v in command.values()),
        "registered_link_names": links,
        "loaded_normal_force_N": peak.tolist(),
        "baseline_normal_force_N": np.linalg.norm(baseline, axis=-1).tolist(),
        "loaded_force_delta_N": delta.tolist(),
        "recovery_delta_from_baseline_N": recovery_delta,
        "recovered_force_max_N": recovery,
        "independent_filtered_peak_N": np.linalg.norm(
            [r["independent_filtered_normal_force_world_N"] for r in records if r["phase"] == "loaded"], axis=-1
        )
        .max(axis=0)
        .tolist(),
        "public_getter_equals_native_sensor": True,
        "records": records,
        "hardware_connected": False,
        "api_fixture_tracks_pad_deflection": False,
        "fixture_referenced_at_phase_boundaries": True,
        "native_surface_queries": surface_queries,
        "fixture_sha256": hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),
        "scope": "Public aggregate normal-force API, not a calibrated hardware tactile stream",
    }
    a.output.write_text(json.dumps(result, indent=2, default=list) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "native_surface_queries")}), flush=True)
    assert result["passed"], "Public channel loading or recovery was not verified"


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
