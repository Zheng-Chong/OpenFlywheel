"""Press all ten mapped links on the complete robot with native mesh probes.

Only simulation is used. Wuji exports aggregate solver forces, not invented
taxels. Sharpa also runs the official TacMap implementation on CAD-fitted grids.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--vendor", type=Path, required=True)
p.add_argument("--maps", type=Path, required=True)
p.add_argument("--probes", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--reference-each-phase",
    action="store_true",
    help="Reposition quasistatic fixture from measured pad pose at phase boundaries",
)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import os  # noqa: E402
import traceback  # noqa: E402

from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sensors.ray_caster import patterns  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from probe_geometry import make_probe  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from tacmap_adapter import load_vendor  # noqa: E402
import torch  # noqa: E402

from env.robot_manager.robot_config._g1d_dexhands import get_config  # noqa: E402
from scripts.dexhands.review_contract import camera_clock_ready  # noqa: E402

DT = 1 / 240
ROOT = Path(__file__).resolve().parents[2]


@sim_utils.clone
def spawn_probe(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Generate at unit scale before shrinking: trimesh's metre-scale UV
    sphere generation removes polar triangles at this small radius."""
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh

    kind = (
        "sphere"
        if isinstance(cfg, sim_utils.MeshSphereCfg)
        else ("flat" if isinstance(cfg, sim_utils.MeshCuboidCfg) else "cylinder")
    )
    mesh = make_probe(
        kind,
        radius=getattr(cfg, "radius", 0.002),
        height=getattr(cfg, "height", 0.002),
        size=getattr(cfg, "size", (0.004, 0.004, 0.002)),
    )
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    import omni.usd

    return omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)


def quat_matrix(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def aim(eye, target):
    f = np.asarray(target) - eye
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f)
    left /= np.linalg.norm(left)
    q = Rotation.from_matrix(np.stack([f, left, np.cross(f, left)], axis=1)).as_quat()
    return q[[3, 0, 1, 2]]


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "execution_inputs"
    frozen.mkdir()
    for source in (
        Path(__file__),
        Path(__file__).parent / "tacmap_adapter.py",
        Path(__file__).parent / "probe_geometry.py",
        Path(__file__).parent / "review_contract.py",
        ROOT / "Assets/Robots" / args.model / "robot.urdf",
        ROOT / "Assets/Robots" / args.model / "model_contract.json",
        ROOT / "env/robot_manager/robot_config/_g1d_dexhands.py",
        args.probes,
        args.maps / "report.json",
        args.maps / "sha256.json",
    ):
        shutil.copyfile(source, frozen / source.name)
    probes = json.loads(args.probes.read_text())[args.model]
    sharpa = "sharpa" in args.model
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device=args.device, enable_scene_query_support=True)
    )
    cfg = get_config(args.model).replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0, 0, 0.1115)
    cfg.init_state.joint_pos[".*_elbow_joint"] = -0.6
    robot = Articulation(cfg)
    light = sim_utils.DomeLightCfg(intensity=1100)
    light.func("/World/Dome", light)
    floor = sim_utils.CuboidCfg(
        size=(6, 6, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.49, 0.55)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.01))
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.6,
        dynamic_friction=0.5,
        restitution=0.0,
        compliant_contact_stiffness=280.0,
        compliant_contact_damping=20.0,
    )
    shapes = ("sphere", "flat", "cylinder")
    objects = []
    pad_contacts = []
    object_contacts = []
    tactiles = []
    tacmap_class, tacmap_cfg = load_vendor(args.vendor / "sharpa-tacmap") if sharpa else (None, None)
    for k, probe in enumerate(probes):
        pad = "/World/Robot/" + probe["link"]
        matpath = pad + "/probe_material"
        collisions = sim.stage.GetPrimAtPath(pad + "/collisions")
        if collisions:
            collisions.SetInstanceable(False)
        material.func(matpath, material)
        sim_utils.bind_physics_material(pad, matpath)
        paths = []
        group = []
        counter = []
        for shape in shapes:
            path = f"/World/Probe_{k}_{shape}"
            paths.append(path)
            common = {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
                "mass_props": sim_utils.MassPropertiesCfg(mass=0.01),
                "activate_contact_sensors": True,
                "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.22, 0.1)),
            }
            if shape == "sphere":
                spawn = sim_utils.MeshSphereCfg(radius=0.002, **common)
            elif shape == "flat":
                spawn = sim_utils.MeshCuboidCfg(size=(0.004, 0.004, 0.002), **common)
            else:
                spawn = sim_utils.MeshCylinderCfg(radius=0.002, height=0.002, **common)
            spawn.func = spawn_probe
            group.append(
                RigidObject(
                    RigidObjectCfg(
                        prim_path=path, spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=(3, k * 0.03, 3))
                    )
                )
            )
            counter.append(
                ContactSensor(
                    ContactSensorCfg(
                        prim_path=path,
                        filter_prim_paths_expr=[pad],
                        update_period=0.0,
                        track_friction_forces=True,
                        max_contact_data_count_per_prim=128,
                    )
                )
            )
        objects.append(group)
        object_contacts.append(counter)
        pad_contacts.append(
            ContactSensor(
                ContactSensorCfg(
                    prim_path=pad,
                    filter_prim_paths_expr=paths,
                    update_period=0.0,
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=128,
                )
            )
        )
        if sharpa:
            kind = "TH" if probe["finger"] == "thumb" else "4F"
            sensor_cfg = tacmap_cfg(
                prim_path=pad,
                mesh_prim_paths=[tacmap_cfg.RaycastTargetCfg(prim_expr=q + "/geometry/mesh") for q in paths],
                update_period=0.0,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
                offset=tacmap_cfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                data_types=["distance_along_normal"],
                points_npy=str(args.maps / f"{kind}_points_mm.npy"),
                normals_npy=str(args.maps / f"{kind}_normals.npy"),
                resolution_step=1,
                correction_scale=1e-3,
                max_distance=0.015,
            )
            tactiles.append(tacmap_class(sensor_cfg))
    camera = Camera(prim_path="/World/Camera", resolution=(960, 720))
    sim.reset()
    q0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    for _ in range(240):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
    camera.initialize()
    camera.prim.GetAttribute("focalLength").Set(16.0)
    camera.set_clipping_range(0.001, 20.0)
    target = np.array([0.05, 0, 0.8])
    eye = target + np.array([1.2, -1.2, 0.65])
    camera.set_world_pose(eye, aim(eye, target), camera_axes="world")
    for _ in range(8):
        sim.render()
    # Freeze fixture reference poses after settling, then use measured link poses
    # for every tactile/force transform. The plunger does not chase deflection.
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    surface_queries = []
    fixture = []
    for probe in probes:
        index = robot.body_names.index(probe["link"])
        r = quat_matrix(robot.data.body_quat_w[0, index].cpu().numpy())
        origin = robot.data.body_pos_w[0, index].cpu().numpy()
        point = origin + r @ probe["point_m"]
        normal = r @ probe["normal"]
        normal /= np.linalg.norm(normal)
        tangent = np.cross(normal, [0, 0, 1])
        if np.linalg.norm(tangent) < 0.1:
            tangent = np.cross(normal, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        rotation = np.stack([tangent, np.cross(normal, tangent), normal], axis=1)
        q = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        hit = query.raycast_closest(tuple(point + normal * 0.01), tuple(-normal), 0.025)
        surface_queries.append({"link": probe["link"], "hit": dict(hit)})
        (args.output / "native_surface_queries.json").write_text(json.dumps(surface_queries, indent=2, default=list))
        expected_material = "/World/Robot/" + probe["link"] + "/probe_material"
        if (
            not hit["hit"]
            or str(hit["rigidBody"]).split("/")[-1] != probe["link"]
            or hit.get("material") != expected_material
        ):
            raise ValueError("Native contact material was not applied to mapped pad: " + probe["link"])
        if not sharpa:
            point = np.asarray(hit["position"])
            normal = np.asarray(hit["normal"])
            tangent = np.cross(normal, [0, 0, 1])
            if np.linalg.norm(tangent) < 0.1:
                tangent = np.cross(normal, [0, 1, 0])
            tangent /= np.linalg.norm(tangent)
            q = Rotation.from_matrix(np.stack([tangent, np.cross(normal, tangent), normal], axis=1)).as_quat()[
                [3, 0, 1, 2]
            ]
        fixture.append((point, normal, tangent, q))
    (args.output / "native_surface_queries.json").write_text(json.dumps(surface_queries, indent=2, default=list))
    local_fixture = []
    for probe, (point, normal, tangent, _) in zip(probes, fixture):
        index = robot.body_names.index(probe["link"])
        rotation = quat_matrix(robot.data.body_quat_w[0, index].cpu().numpy())
        origin = robot.data.body_pos_w[0, index].cpu().numpy()
        local_fixture.append((rotation.T @ (point - origin), rotation.T @ normal, rotation.T @ tangent))
    # Positive depth is penetration, negative depth is a physical gap.
    waypoints = [
        ("baseline", -0.002, 0.0),
        ("approach", -0.0002, 0.0),
        ("touch", 0.0, 0.0),
        ("press_02", 0.0002, 0.0),
        ("press_05", 0.0005, 0.0),
        ("press_08", 0.0008, 0.0),
        ("shear_plus", 0.0005, 0.0008),
        ("shear_minus", 0.0005, -0.0008),
        ("release", -0.002, 0.0),
        ("recovered", -0.002, 0.0),
    ]
    rows = []
    depths = []
    images = []
    metric_errors = []
    quant_errors = []
    metric_diagnostics = []
    diagnostics = {
        "fixture": [
            {**p, "world_point_m": f[0].tolist(), "world_normal": f[1].tolist()} for p, f in zip(probes, fixture)
        ],
        "sensor_body_names": [s.body_names for s in pad_contacts],
    }
    (args.output / "fixture.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (args.output / "whole").mkdir()
    trace = open(args.output / "frames.jsonl", "w")
    for shape_index, shape in enumerate(shapes):
        # Park all three probe sets before switching shape.
        for k, group in enumerate(objects):
            for obj in group:
                pose = torch.tensor([[3.0, k * 0.03, 3.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
                obj.write_root_pose_to_sim(pose)
        for phase, depth, shear in waypoints:
            if args.reference_each_phase:
                for k, probe in enumerate(probes):
                    index = robot.body_names.index(probe["link"])
                    rotation = quat_matrix(robot.data.body_quat_w[0, index].cpu().numpy())
                    origin = robot.data.body_pos_w[0, index].cpu().numpy()
                    point, normal, tangent = local_fixture[k]
                    normal, tangent = rotation @ normal, rotation @ tangent
                    quat = Rotation.from_matrix(
                        np.stack([tangent, np.cross(normal, tangent), normal], axis=1)
                    ).as_quat()[[3, 0, 1, 2]]
                    fixture[k] = (origin + rotation @ point, normal, tangent, quat)
            for k, group in enumerate(objects):
                point, normal, tangent, q = fixture[k]
                radius = 0.002 if shape == "sphere" else 0.001
                center = point + normal * (radius - depth) + tangent * shear
                pose = torch.tensor([np.r_[center, q]], device=sim.device, dtype=torch.float32)
                group[shape_index].write_root_pose_to_sim(pose)
            for _ in range(24):
                robot.set_joint_position_target(q0)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(DT)
                for group in objects:
                    for obj in group:
                        obj.update(DT)
            forces = []
            forces_local = []
            normals = []
            frictions = []
            opposites = []
            opposition = []
            peak = []
            area = []
            maps = []
            poses = []
            sensors_uint8 = []
            for k, probe in enumerate(probes):
                pad_contacts[k].update(0.1, force_recompute=True)
                object_contacts[k][shape_index].update(0.1, force_recompute=True)
                normal_force = pad_contacts[k].data.force_matrix_w[0, 0].sum(dim=0).cpu().numpy()
                friction = pad_contacts[k].data.friction_forces_w[0, 0].sum(dim=0).cpu().numpy()
                force = normal_force + friction
                counter = object_contacts[k][shape_index].data
                opposite = (counter.force_matrix_w[0, 0] + counter.friction_forces_w[0, 0]).sum(dim=0).cpu().numpy()
                normals.append(normal_force.tolist())
                frictions.append(friction.tolist())
                opposites.append(opposite.tolist())
                index = robot.body_names.index(probe["link"])
                q = robot.data.body_quat_w[0, index].cpu().numpy()
                r = quat_matrix(q)
                forces.append(force.tolist())
                forces_local.append((r.T @ force).tolist())
                opposition.append(float(np.linalg.norm(force + opposite)))
                poses.append(np.r_[robot.data.body_pos_w[0, index].cpu().numpy(), q].tolist())
                if sharpa:
                    sensor = tactiles[k]
                    sensor.update(0.1, force_recompute=True)
                    raw = sensor.metric_depth()[0].reshape(240, 240)
                    stock = sensor.data.output["distance_along_normal"][0].reshape(240, 240)
                    from sharpa_vendor.torch_jit_utils import deform_quantize

                    expected = sensor.blur(deform_quantize(raw.clone()).reshape(1, 240, 240))[0]
                    quant_errors.append(int(torch.abs(stock.to(torch.int16) - expected.to(torch.int16)).max().item()))
                    kind = "TH" if probe["finger"] == "thumb" else "4F"
                    mask = torch.as_tensor(np.load(args.maps / f"{kind}_valid.npy"), device=sim.device)
                    raw = torch.where(mask, raw, torch.zeros_like(raw))
                    # Analytic sphere is independent of triangle ray casting.
                    if shape == "sphere":
                        center = objects[k][shape_index].data.root_pos_w[0]
                        starts = sensor._ray_starts_w[0]
                        directions = sensor._ray_directions_w[0]
                        offset = starts - center
                        dot = (offset * directions).sum(-1)
                        c = (offset * offset).sum(-1) - 0.002**2
                        analytic = torch.where(
                            c < 0, -dot + torch.sqrt(torch.clamp(dot * dot - c, min=0)), torch.zeros_like(c)
                        ).reshape(240, 240)
                        analytic = torch.where(mask, analytic, torch.zeros_like(analytic))
                        error = torch.abs(raw - analytic)
                        metric_errors.append(float(error.max().item()))
                        if error.max() > 0.0001:
                            worst = int(error.argmax().item())
                            diagnostic = {
                                "frame": len(rows),
                                "channel": probe["link"],
                                "phase": phase,
                                "error_m": float(error.max().item()),
                                "raw_m": float(raw.reshape(-1)[worst].item()),
                                "analytic_m": float(analytic.reshape(-1)[worst].item()),
                                "start_w": starts[worst].cpu().numpy().tolist(),
                                "direction": directions[worst].cpu().numpy().tolist(),
                                "hit_w": sensor.ray_hits_w[0, worst].cpu().numpy().tolist(),
                                "probe_center_w": center.cpu().numpy().tolist(),
                                "tracked_centers_w": sensor._mesh_positions_w[0].cpu().numpy().tolist(),
                                "tracked_quat_wxyz": sensor._mesh_orientations_w[0].cpu().numpy().tolist(),
                            }
                            metric_diagnostics.append(diagnostic)
                            print("METRIC_DIAGNOSTIC " + json.dumps(diagnostic), flush=True)
                    maps.append(raw.cpu().numpy())
                    sensors_uint8.append(stock.cpu().numpy())
                    peak.append(float(raw.max().item()))
                    area.append(int((raw > 1e-5).sum().item()))
            if sharpa:
                depths.append(np.stack(maps))
                images.append(np.stack(sensors_uint8))
            sim.render()
            for _ in range(3):
                if camera_clock_ready([camera.get_current_frame().get("rendering_time")], sim.current_time, DT):
                    break
                sim.render()
            camera_stamp = camera.get_current_frame().get("rendering_time")
            if not camera_clock_ready([camera_stamp], sim.current_time, DT):
                raise ValueError("Contact camera clock differs from solver state")
            Image.fromarray(camera.get_rgba()[..., :3]).save(args.output / "whole" / f"{len(rows):05d}.jpg", quality=92)
            row = {
                "frame": len(rows),
                "time_s": sim.current_time,
                "step": round(sim.current_time / DT),
                "camera_sensor_time_s": {"whole": camera_stamp},
                "shape": shape,
                "phase": phase,
                "fixture_depth_m": depth,
                "shear_m": shear,
                "force_world_N": forces,
                "force_link_N": forces_local,
                "normal_force_world_N": normals,
                "friction_force_world_N": frictions,
                "opposite_force_world_N": opposites,
                "pair_opposition_error_N": opposition,
                "link_pose_wxyz": poses,
                "peak_depth_m": peak,
                "contact_pixels": area,
                "joint_position_rad": robot.data.joint_pos[0].cpu().numpy().tolist(),
                "probe_position_w_m": [
                    group[shape_index].data.root_pos_w[0].cpu().numpy().tolist() for group in objects
                ],
                "probe_quat_wxyz": [group[shape_index].data.root_quat_w[0].cpu().numpy().tolist() for group in objects],
                "target_joint_rad": q0[0].cpu().numpy().tolist(),
            }
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            print(
                json.dumps(
                    {
                        "shape": shape,
                        "phase": phase,
                        "max_force_N": float(np.linalg.norm(forces, axis=1).max()),
                        "peak_depth_m": max(peak, default=0.0),
                    }
                ),
                flush=True,
            )
    trace.close()
    (args.output / "metric_diagnostics.json").write_text(json.dumps(metric_diagnostics, indent=2) + "\n")
    sim.stage.Export(str(args.output / "scene.usdc"))
    if sharpa:
        np.savez_compressed(args.output / "tactile.npz", depth_m=np.stack(depths), vendor_uint8=np.stack(images))
    np.savez_compressed(
        args.output / "forces.npz",
        world_N=np.array([r["force_world_N"] for r in rows]),
        link_N=np.array([r["force_link_N"] for r in rows]),
        time_s=np.array([r["time_s"] for r in rows]),
    )
    force = np.array([r["force_world_N"] for r in rows])
    opposition = np.array([r["pair_opposition_error_N"] for r in rows])
    contact_by_channel = np.linalg.norm(force, axis=2).max(axis=0)
    released = np.array([np.linalg.norm(r["force_world_N"], axis=1) for r in rows if r["phase"] == "recovered"])
    per_shape = {
        shape: np.linalg.norm([row["force_world_N"] for row in rows if row["shape"] == shape], axis=2)
        .max(axis=0)
        .tolist()
        for shape in shapes
    }
    all_shape_contact = all(min(values) > 0.001 for values in per_shape.values())
    report = {
        "model": args.model,
        "urdf_sha256": hashlib.sha256((ROOT / "Assets/Robots" / args.model / "robot.urdf").read_bytes()).hexdigest(),
        "joint_names": robot.joint_names,
        "frames": len(rows),
        "fps": 10,
        "dt_s": DT,
        "camera_clock_synchronized": True,
        "quasistatic_fixture_referenced_each_phase": args.reference_each_phase,
        "channels": [p["link"] for p in probes],
        "native_contact_peak_N": contact_by_channel.tolist(),
        "force_composition": "normal_contact_plus_friction_solver_force",
        "friction_peak_N": np.linalg.norm([r["friction_force_world_N"] for r in rows], axis=-1).max(axis=0).tolist(),
        "all_channels_contact": bool((contact_by_channel > 0.001).all()),
        "per_shape_peak_N": per_shape,
        "all_shape_channels_contact": all_shape_contact,
        "force_opposition_max_N": float(opposition.max()),
        "recovered_force_max_N": float(released.max()),
        "metric_sphere_max_error_m": max(metric_errors, default=None),
        "official_quantizer_max_uint8_error": max(quant_errors, default=None),
        "applied_material_verified_by_native_raycast": True,
        "contact_material": {
            "stiffness_N_m": 280.0,
            "damping_N_s_m": 20.0,
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "status": "simulation_fixture_parameters",
        },
        "vendor_tacmap_executed": sharpa,
        "raycast_numeric_units": "millimetres_internal_metres_external" if sharpa else None,
        "execution_input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()},
        "wuji_per_taxel_calibration": "not_available_requires_per_sensor_info_metadata",
        "virtual_contact_only": args.model.endswith("_1"),
        "hardware_connected": False,
        "scope": (
            "Full robot, native contact pairs and same-frame link transforms. "
            "Geometry maps are not real SN calibration. Depth is not force."
        ),
        "thresholds": {
            "contact_N": 0.001,
            "opposition_N": 0.01,
            "recovered_N": 0.01,
            "sphere_m": 0.0001,
            "quantizer_uint8": 1,
        },
        "passed": all_shape_contact
        and float(opposition.max()) < 0.01
        and float(released.max()) < 0.01
        and (not sharpa or (max(metric_errors) < 0.0001 and max(quant_errors) <= 1)),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
