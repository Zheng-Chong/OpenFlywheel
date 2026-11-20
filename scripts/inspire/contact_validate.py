"""Native RH56DFTP finger/palm contact fixtures; aggregate solver forces, not invented taxels."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--fixtures", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--shapes", nargs="+", choices=["sphere", "flat", "cylinder"], default=["sphere", "flat"])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app

import os  # noqa: E402
import traceback  # noqa: E402

from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402

from env.robot_manager.robot_config.g1d_inspire_rh56dftp import get_robot_config  # noqa: E402
from scripts.dexhands.probe_geometry import make_probe  # noqa: E402
from scripts.dexhands.review_contract import camera_clock_ready  # noqa: E402

DT = 1 / 480
ROOT = Path(__file__).resolve().parents[2]
MODEL = "g1d_inspire_rh56dftp"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def aim(eye, target):
    f = np.asarray(target) - eye
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f)
    left /= np.linalg.norm(left)
    q = Rotation.from_matrix(np.stack([f, left, np.cross(f, left)], axis=1)).as_quat()
    return q[[3, 0, 1, 2]]


@sim_utils.clone
def spawn_probe(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    import omni.usd

    kind = (
        "sphere"
        if isinstance(cfg, sim_utils.MeshSphereCfg)
        else ("cylinder" if isinstance(cfg, sim_utils.MeshCylinderCfg) else "flat")
    )
    mesh = make_probe(kind, radius=0.002, size=(0.004, 0.004, 0.002))
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)


def main():
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = a.output / "execution_inputs"
    frozen.mkdir()
    asset = ROOT / "Assets/Robots" / MODEL
    for source in (
        Path(__file__),
        a.fixtures,
        asset / "model_contract.json",
        asset / "robot.urdf",
        ROOT / "scripts/dexhands/review_contract.py",
        ROOT / "scripts/dexhands/probe_geometry.py",
        ROOT / "env/robot_manager/robot_config/g1d_inspire_rh56dftp.py",
    ):
        shutil.copyfile(source, frozen / source.name)
    fixtures = json.loads(a.fixtures.read_text())
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=a.device, enable_scene_query_support=True))
    cfg = get_robot_config().replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0, 0, 0.1115)
    cfg.init_state.joint_pos[".*_elbow_joint"] = -0.6
    robot = Articulation(cfg)
    light = sim_utils.DomeLightCfg(intensity=1100)
    light.func("/World/Light", light)
    floor = sim_utils.CuboidCfg(
        size=(8, 8, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.48, 0.53)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.01))
    objects, sensors, opposites = [], [], []
    shapes = tuple(a.shapes)
    for k, f in enumerate(fixtures):
        path = "/World/Robot/" + f["link"]
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0,
            compliant_contact_stiffness=280,
            compliant_contact_damping=20,
        )
        material.func(path + "/ContactMaterial", material)
        sim_utils.bind_physics_material(path, path + "/ContactMaterial")
        group, counters, paths = [], [], []
        for shape in shapes:
            probe_path = f"/World/Probe_{k}_{shape}"
            paths.append(probe_path)
            common = dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.22, 0.1)),
            )
            spawn = (
                sim_utils.MeshSphereCfg(radius=0.002, **common)
                if shape == "sphere"
                else (
                    sim_utils.MeshCuboidCfg(size=(0.004, 0.004, 0.002), **common)
                    if shape == "flat"
                    else sim_utils.MeshCylinderCfg(radius=0.002, height=0.002, **common)
                )
            )
            spawn.func = spawn_probe
            group.append(
                RigidObject(
                    RigidObjectCfg(
                        prim_path=probe_path,
                        spawn=spawn,
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(3, k * 0.03, 3)),
                    )
                )
            )
            counters.append(
                ContactSensor(
                    ContactSensorCfg(
                        prim_path=probe_path,
                        filter_prim_paths_expr=[path],
                        update_period=0,
                        track_friction_forces=True,
                        max_contact_data_count_per_prim=64,
                    )
                )
            )
        objects.append(group)
        opposites.append(counters)
        sensors.append(
            ContactSensor(
                ContactSensorCfg(
                    prim_path=path,
                    filter_prim_paths_expr=paths,
                    update_period=0,
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=64,
                )
            )
        )
    cameras = [
        (name, Camera(prim_path="/World/Camera_" + name, resolution=(960, 720) if name == "whole" else (640, 640)))
        for name in ("whole", "left", "right")
    ]
    sim.reset()
    q0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    for _ in range(480):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
    for name, cam in cameras:
        cam.initialize()
        cam.prim.GetAttribute("focalLength").Set(16)
        cam.set_clipping_range(0.001, 20)
        if name == "whole":
            target = np.array([0.02, 0, 0.72])
            eye = target + [1.45, -1.3, 0.72]
        else:
            links = [robot.body_names.index(f["link"]) for f in fixtures if f["side"] == name]
            target = robot.data.body_pos_w[0, links].cpu().numpy().mean(axis=0)
            eye = target + np.array([0.26, -0.28 if name == "left" else 0.28, 0.19])
        cam.set_world_pose(eye, aim(eye, target), camera_axes="world")
        (a.output / name).mkdir()
    for _ in range(12):
        sim.render()
    reference = []
    collision_queries = []
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    for f in fixtures:
        i = robot.body_names.index(f["link"])
        r = rotation(robot.data.body_quat_w[0, i].cpu().numpy())
        pos = robot.data.body_pos_w[0, i].cpu().numpy() + r @ f["point_m"]
        normal = r @ f["normal"]
        normal /= np.linalg.norm(normal)
        hit = query.raycast_closest(tuple(pos + normal * 0.01), tuple(-normal), 0.025)
        collision_queries.append({"link": f["link"], "hit": dict(hit), "cad_point_w_m": pos.tolist()})
        (a.output / "native_surface_queries.json").write_text(
            json.dumps(collision_queries, indent=2, default=list) + "\n"
        )
        if not hit["hit"] or str(hit["rigidBody"]).split("/")[-1] != f["link"]:
            raise ValueError("CAD fixture does not see the intended native collider: " + f["link"])
        pos = np.asarray(hit["position"])
        normal = np.asarray(hit["normal"])
        tangent = np.cross(normal, [0, 0, 1])
        if np.linalg.norm(tangent) < 0.1:
            tangent = np.cross(normal, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        q = Rotation.from_matrix(np.stack([tangent, np.cross(normal, tangent), normal], axis=1)).as_quat()[[3, 0, 1, 2]]
        reference.append((pos, normal, tangent, q))
    (a.output / "native_surface_queries.json").write_text(json.dumps(collision_queries, indent=2, default=list) + "\n")
    rows = []
    trace = (a.output / "frames.jsonl").open("w")
    waypoints = [
        ("baseline", -0.002, 0),
        ("touch", 0, 0),
        ("press_02", 0.0002, 0),
        ("press_05", 0.0005, 0),
        ("press_08", 0.0008, 0),
        ("shear_plus", 0.0005, 0.001),
        ("shear_minus", 0.0005, -0.001),
        ("release", -0.002, 0),
        ("recovered", -0.002, 0),
    ]
    for si, shape in enumerate(shapes):
        for group in objects:
            for obj in group:
                pose = obj.data.default_root_state[:, :7].clone()
                obj.write_root_pose_to_sim(pose)
        previous_depth, previous_shear = -0.002, 0.0
        for k, (pos, normal, tangent, quat) in enumerate(reference):
            extent = 0.002 if shape == "sphere" else 0.001
            pose = torch.tensor(
                np.r_[pos + normal * (extent + 0.002), quat][None], dtype=torch.float32, device=sim.device
            )
            objects[k][si].write_root_pose_to_sim(pose)
        for phase, depth, shear in waypoints:
            for step in range(96):
                alpha = (step + 1) / 96
                actual_depth = previous_depth + alpha * (depth - previous_depth)
                actual_shear = previous_shear + alpha * (shear - previous_shear)
                for k, (pos, normal, tangent, quat) in enumerate(reference):
                    extent = 0.002 if shape == "sphere" else 0.001
                    pose = torch.tensor(
                        np.r_[pos + normal * (extent - actual_depth) + tangent * actual_shear, quat[[1, 2, 3, 0]]][
                            None
                        ],
                        dtype=torch.float32,
                        device=sim.device,
                    )
                    objects[k][si].root_physx_view.set_kinematic_targets(
                        pose, torch.tensor([0], dtype=torch.int32, device=sim.device)
                    )
                robot.set_joint_position_target(q0)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(DT)
                for group in objects:
                    for obj in group:
                        obj.update(DT)
                if step % 48 != 47:
                    continue
                forces, local, poses, errors, normals, frictions, opposite_vectors = [], [], [], [], [], [], []
                for k, f in enumerate(fixtures):
                    sensors[k].update(DT, force_recompute=True)
                    opposites[k][si].update(DT, force_recompute=True)
                    normal_force = sensors[k].data.force_matrix_w[0, 0].sum(dim=0).cpu().numpy()
                    friction = sensors[k].data.friction_forces_w[0, 0].sum(dim=0).cpu().numpy()
                    force = normal_force + friction
                    opposite = (
                        (opposites[k][si].data.force_matrix_w[0, 0] + opposites[k][si].data.friction_forces_w[0, 0])
                        .sum(dim=0)
                        .cpu()
                        .numpy()
                    )
                    i = robot.body_names.index(f["link"])
                    quat = robot.data.body_quat_w[0, i].cpu().numpy()
                    forces.append(force.tolist())
                    local.append((rotation(quat).T @ force).tolist())
                    poses.append(np.r_[robot.data.body_pos_w[0, i].cpu().numpy(), quat].tolist())
                    errors.append(float(np.linalg.norm(force + opposite)))
                    opposite_vectors.append(opposite.tolist())
                    normals.append(normal_force.tolist())
                    frictions.append(friction.tolist())
                sim.render()
                for _ in range(3):
                    if camera_clock_ready(
                        [c.get_current_frame().get("rendering_time") for _, c in cameras], sim.current_time, DT
                    ):
                        break
                    sim.render()
                camera_stamps = {n: c.get_current_frame().get("rendering_time") for n, c in cameras}
                if not camera_clock_ready(camera_stamps.values(), sim.current_time, DT):
                    raise ValueError("Contact camera clock differs from solver state")
                frame = len(rows)
                for name, cam in cameras:
                    Image.fromarray(cam.get_rgba()[..., :3]).save(a.output / name / f"{frame:05d}.jpg", quality=92)
                row = {
                    "frame": frame,
                    "time_s": sim.current_time,
                    "step": round(sim.current_time / DT),
                    "camera_sensor_time_s": camera_stamps,
                    "phase": phase,
                    "shape": shape,
                    "fixture_depth_m": actual_depth,
                    "shear_m": actual_shear,
                    "joint_position_rad": robot.data.joint_pos[0].cpu().tolist(),
                    "target_rad": q0[0].cpu().tolist(),
                    "force_world_N": forces,
                    "force_link_N": local,
                    "opposite_force_world_N": opposite_vectors,
                    "normal_force_world_N": normals,
                    "friction_force_world_N": frictions,
                    "link_pose_wxyz": poses,
                    "pair_opposition_error_N": errors,
                    "probe_position_w_m": [g[si].data.root_pos_w[0].cpu().tolist() for g in objects],
                    "probe_quat_wxyz": [g[si].data.root_quat_w[0].cpu().tolist() for g in objects],
                }
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                rows.append(row)
            print(
                json.dumps(
                    {"shape": shape, "phase": phase, "max_contact_N": float(np.linalg.norm(forces, axis=-1).max())}
                ),
                flush=True,
            )
            previous_depth, previous_shear = depth, shear
    trace.close()
    forces = np.array([r["force_world_N"] for r in rows])
    recovered = max(np.linalg.norm(r["force_world_N"], axis=-1).max() for r in rows if r["phase"] == "recovered")
    peaks = {
        s: np.linalg.norm([r["force_world_N"] for r in rows if r["shape"] == s], axis=-1).max(axis=0).tolist()
        for s in shapes
    }
    opposition = max(max(r["pair_opposition_error_N"]) for r in rows)
    passed = all(min(v) > 0.001 for v in peaks.values()) and recovered < 0.001 and opposition < 0.001
    result = {
        "model": MODEL,
        "urdf_sha256": digest(asset / "robot.urdf"),
        "joint_names": robot.joint_names,
        "frames": len(rows),
        "fps": 10,
        "dt_s": DT,
        "camera_clock_synchronized": True,
        "channels": [f["link"] for f in fixtures],
        "native_contact_peak_N": np.linalg.norm(forces, axis=-1).max(axis=0).tolist(),
        "per_shape_peak_N": peaks,
        "recovered_force_max_N": float(recovered),
        "force_opposition_max_N": float(opposition),
        "probe_motion": "native_kinematic_targets_with_per_step_ramps_not_teleported_shear",
        "friction_peak_N": np.linalg.norm([r["friction_force_world_N"] for r in rows], axis=-1).max(axis=0).tolist(),
        "force_composition": "normal_contact_plus_friction_solver_force",
        "gravity_enabled": True,
        "self_collision_enabled": True,
        "hardware_connected": False,
        "physics_device": a.device,
        "scope": (
            "12 filtered rigid-body contact channels (five distal fingers and palm per side); "
            "these are not 17 hardware tactile patches or 1062 calibrated taxels"
        ),
        "material": {
            "stiffness_N_m": 280,
            "damping_Ns_m": 20,
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "status": "simulation_fixture_not_real_elastomer_calibration",
        },
        "execution_input_sha256": {p.name: digest(p) for p in frozen.iterdir()},
        "passed": bool(passed),
    }
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        a.output / "forces.npz",
        world_N=forces,
        link_N=[r["force_link_N"] for r in rows],
        time_s=[r["time_s"] for r in rows],
    )
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
