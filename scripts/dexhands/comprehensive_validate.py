"""Four-hand native simulation acceptance: motion, free-body grasp and synchronized RGB-D/contact.

Objects are initialized on supports, then evolve only under gravity and contact.
No attachment, gravity disabling, object-following pose writes or hardware APIs.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--plan", type=Path, required=True)
p.add_argument("--grasps", type=Path)
p.add_argument("--mode", choices=["motion", "grasp", "smoke"], default="motion")
p.add_argument("--fps", type=int, default=5)
p.add_argument("--no-head", action="store_true")
p.add_argument(
    "--contact-servo", action="store_true", help="Force-limited closure using native object contact feedback"
)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app

import h5py  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from scripts.dexhands.kinematics import fk  # noqa: E402
from scripts.dexhands.review_contract import camera_clock_ready, native_pose_xyzw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DT = 1 / 480


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def look(eye, target):
    f = np.asarray(target) - eye
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f)
    left /= np.linalg.norm(left)
    q = Rotation.from_matrix(np.stack([f, left, np.cross(f, left)], axis=1)).as_quat()
    return q[[3, 0, 1, 2]]


def rot(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def main():
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = a.output / "execution_inputs"
    inputs.mkdir()
    asset = ROOT / "Assets/Robots" / a.model
    module = importlib.import_module("env.robot_manager.robot_config." + a.model)
    contract = json.loads((asset / "model_contract.json").read_text())
    snapshot = [
        ROOT / "scripts/dexhands/review_contract.py",
        Path(__file__),
        a.plan,
        asset / "model_contract.json",
        asset / "robot.urdf",
        Path(module.__file__),
    ]
    snapshot += [
        ROOT / "env/robot_manager/robot_config/_g1d_dexhands.py",
        ROOT / "env/robot_manager/robot_config/g1d.py",
        asset / "left_robot_config.yml",
        asset / "right_robot_config.yml",
        ROOT / "env/robot_manager/robot_manager.py",
    ]
    if a.grasps:
        snapshot.append(a.grasps)
    snapshot.append(ROOT / "scripts/dexhands/review_contract.py")
    hashes = {}
    for i, source in enumerate(snapshot):
        dest = inputs / (str(i) + "_" + source.name)
        shutil.copyfile(source, dest)
        hashes[dest.name] = sha(dest)
    (inputs / "command.json").write_text(json.dumps(sys.argv, indent=2))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=a.device, enable_scene_query_support=True))
    cfg = module.get_robot_config().replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0, 0, 0.1115)
    cfg.init_state.joint_pos[".*_elbow_joint"] = -0.6
    grasp_profiles = json.loads(a.grasps.read_text())[a.model] if a.mode == "grasp" else None
    if grasp_profiles:
        wrist_default = cfg.init_state.joint_pos.pop(".*_wrist_.*")
        cfg.init_state.joint_pos[".*_wrist_(pitch|yaw)_joint"] = wrist_default
        for side, profile in grasp_profiles.items():
            cfg.init_state.joint_pos[side + "_wrist_roll_joint"] = profile.get("wrist_roll_rad", 0.0)
    robot = Articulation(cfg)
    floor = sim_utils.CuboidCfg(
        size=(6, 6, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.36, 0.41)),
    )
    floor.func("/World/Floor", floor, translation=(0, 0, -0.01))
    dome = sim_utils.DomeLightCfg(intensity=1300)
    dome.func("/World/Light", dome)
    urdf = ET.parse(asset / "robot.urdf").getroot()
    channels = [n for group in module.NATIVE_CONTACT_LINKS.values() for n in group]
    # Per-pad sensors can filter many objects; one many-body sensor cannot support
    # many-to-many filtering in this pinned PhysX tensor interface.
    objects = []
    pads = []
    opposites = []
    supports = []
    support_objects = {}
    placements = {}
    if a.mode == "grasp":
        grasps = grasp_profiles
        initial_q = {j: 0 for h in contract["hands"].values() for j in h["joints"]}
        for h in contract["hands"].values():
            initial_q.update(h.get("initial_rad", {}))
        initial_q.update({side + "_wrist_roll_joint": p.get("wrist_roll_rad", 0.0) for side, p in grasps.items()})
        f = fk(urdf, {**initial_q, "left_elbow_joint": -0.6, "right_elbow_joint": -0.6})
        for side in ("left", "right"):
            center = np.asarray(grasps[side]["center_wrist_m"])
            T = f[side + "_wrist_yaw_link"]
            world = T[:3, :3] @ center + T[:3, 3] + cfg.init_state.pos
            placements[side] = world.tolist()
            common = dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.025),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9, dynamic_friction=0.7, restitution=0
                ),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.31, 0.10)),
            )
            for shape in ("sphere", "cube", "cylinder"):
                path = "/World/Objects/" + side + "_" + shape
                if shape == "sphere":
                    spawn = sim_utils.SphereCfg(radius=0.025, **common)
                elif shape == "cube":
                    spawn = sim_utils.CuboidCfg(size=(0.044, 0.044, 0.044), **common)
                else:
                    spawn = sim_utils.CylinderCfg(radius=0.023, height=0.048, **common)
                objects.append(
                    (
                        side,
                        shape,
                        RigidObject(
                            RigidObjectCfg(
                                prim_path=path, spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=(2, 0, 0.1))
                            )
                        ),
                    )
                )
            # Three independent convex posts form a real, retractable ball nest.
            # The object is never attached; the nest is absent during free hold.
            for post_index in range(3):
                angle = post_index * 2 * np.pi / 3
                delta = np.array([np.cos(angle), np.sin(angle)]) * 0.010
                rho = np.linalg.norm(np.maximum(np.abs(delta) - 0.003, 0))
                pos = world.copy()
                pos[:2] += delta
                pos[2] -= np.sqrt(0.025**2 - rho**2) + 0.006
                support = sim_utils.CuboidCfg(
                    size=(0.006, 0.006, 0.012),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.42, 0.55)),
                )
                support_objects[side + str(post_index)] = RigidObject(
                    RigidObjectCfg(
                        prim_path="/World/Support_" + side + str(post_index),
                        spawn=support,
                        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos + np.array([0, 0, -0.3]))),
                    )
                )
                supports.append(pos.tolist())
    for n in channels:
        paths = ["/World/Objects/" + side + "_" + shape for side, shape, _ in objects] or []
        pads.append(
            ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/Robot/" + n,
                    filter_prim_paths_expr=paths,
                    update_period=0,
                    track_friction_forces=bool(paths),
                    max_contact_data_count_per_prim=128,
                )
            )
        )
    for side, shape, obj in objects:
        paths = ["/World/Robot/" + n for n in channels]
        opposites.append(
            ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/Objects/" + side + "_" + shape,
                    filter_prim_paths_expr=paths,
                    update_period=0,
                    track_friction_forces=True,
                    max_contact_data_count_per_prim=128,
                )
            )
        )
    cameras = {n: Camera(prim_path="/World/Camera_" + n, resolution=(640, 480)) for n in ("whole", "left", "right")}
    if not a.no_head:
        for name in ("head_left", "head_right"):
            cameras[name] = Camera(prim_path="/World/Robot/head_link/Validation_" + name, resolution=(320, 240))
    sim.reset()
    names = robot.joint_names
    ids = {s: [names.index(n) for n in h["joints"]] for s, h in contract["hands"].items()}
    active = [i for group in ids.values() for i in group]
    q0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))

    hand_limits = {
        n: v for h in contract["hands"].values() for n, v in h.get("effective_active_limits", h["limits"]).items()
    }

    def step(goal):
        applied = goal.clone()
        if "inspire" in a.model:
            applied[:, active] = torch.clamp(
                applied[:, active], robot.data.joint_pos[:, active] - 0.05, robot.data.joint_pos[:, active] + 0.05
            )
        robot.set_joint_position_target(applied)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
        if not torch.isfinite(robot.data.joint_pos).all():
            raise FloatingPointError("Nonfinite articulation state")
        for _, _, obj in objects:
            obj.update(DT)
        for obj in support_objects.values():
            obj.update(DT)
        return applied

    for _ in range(480):
        step(q0)
    metadata = {}
    descriptors = yaml.safe_load((asset / "left_robot_config.yml").read_text())["camera"]
    for name, cam in cameras.items():
        cam.initialize()
        cam.add_distance_to_image_plane_to_frame()
        cam.prim.GetAttribute("focalLength").Set(16)
        cam.set_clipping_range(0.005, 10)
        if name == "whole":
            target = np.array([0.05, 0, 0.75])
            eye = target + [1.45, -1.3, 0.65]
            cam.set_world_pose(eye, look(eye, target), camera_axes="world")
        elif name in ("left", "right"):
            k = robot.body_names.index(contract["hands"][name]["root"])
            target = robot.data.body_pos_w[0, k].cpu().numpy() + np.array([0.11, 0, 0.055])
            eye = target + [0.30, -0.28 if name == "left" else 0.28, 0.22]
            cam.set_world_pose(eye, look(eye, target), camera_axes="world")
        else:
            desc = descriptors[0 if name == "head_left" else 1]
            k = robot.body_names.index(desc["link"])
            R = rot(robot.data.body_quat_w[0, k].cpu().numpy())
            pos = robot.data.body_pos_w[0, k].cpu().numpy() + R @ desc["pos"]
            qc = Rotation.from_matrix(R) * Rotation.from_quat(np.asarray(desc["ori"])[[1, 2, 3, 0]])
            cam.set_world_pose(pos, qc.as_quat()[[3, 0, 1, 2]], camera_axes="usd")
            cam.set_clipping_range(0.1, 10)
            cam.prim.GetAttribute("focalLength").Set(10)
            cam.prim.GetAttribute("horizontalAperture").Set(22.898)
            cam.prim.GetAttribute("verticalAperture").Set(17.174)
        (a.output / name).mkdir()
        metadata[name] = {
            "intrinsic_K": np.asarray(cam.get_intrinsics_matrix()).tolist(),
            "resolution": cam.get_resolution(),
            "reference_extrinsics_not_hardware_calibrated": name.startswith("head"),
        }
    for _ in range(10):
        sim.render()
    trace = (a.output / "frames.jsonl").open("w")
    data = h5py.File(a.output / "modalities.h5", "x")
    datasets = {}
    data.attrs.update(
        model=a.model,
        urdf_sha256=sha(asset / "robot.urdf"),
        mode=a.mode,
        hardware_connected=False,
        dt_s=DT,
        source_input_sha256=json.dumps(hashes),
        contact_scope="native normal and friction aggregates; not hardware taxels",
    )
    data.create_dataset("joint_names", data=names, dtype=h5py.string_dtype())
    types = {j.get("name"): j.get("type") for j in urdf.findall("joint")}
    data.create_dataset(
        "joint_units", data=["m" if types[n] == "prismatic" else "rad" for n in names], dtype=h5py.string_dtype()
    )
    data.attrs["camera_axes"] = "USD: -Z forward, +Y up; depth is distance to image plane in metres"
    data.create_dataset("channel_names", data=channels, dtype=h5py.string_dtype())
    reports = []
    rows = []
    frame = 0
    step_count = 480

    def append(key, val):
        arr = np.asarray(val)
        if key not in datasets:
            datasets[key] = data.create_dataset(
                key,
                shape=(0, *arr.shape),
                maxshape=(None, *arr.shape),
                dtype=arr.dtype,
                chunks=(1, *arr.shape),
                compression="gzip",
                compression_opts=1,
            )
        ds = datasets[key]
        ds.resize((len(ds) + 1, *arr.shape))
        ds[-1] = arr

    def record(phase, requested, applied, shape="none"):
        nonlocal frame
        q = robot.data.joint_pos[0].cpu().numpy()
        poses = robot.data.body_pos_w[0].cpu().numpy()
        pred = fk(urdf, dict(zip(names, q)))
        errors = [
            np.linalg.norm(poses[k] - pred[n][:3, 3] - cfg.init_state.pos)
            for k, n in enumerate(robot.body_names)
            if n in pred
        ]
        forces = []
        normal = []
        friction = []
        locals_ = []
        linkposes = []
        for n, s in zip(channels, pads):
            s.update(1 / a.fps, force_recompute=True)
            index = robot.body_names.index(n)
            R = rot(robot.data.body_quat_w[0, index].cpu().numpy())
            if objects:
                fn = s.data.force_matrix_w[0, 0].sum(0).cpu().numpy()
                ff = s.data.friction_forces_w[0, 0].sum(0).cpu().numpy()
            else:
                fn = s.data.net_forces_w[0, 0].cpu().numpy()
                ff = np.zeros(3)
            normal.append(fn)
            friction.append(ff)
            forces.append(fn + ff)
            locals_.append(R.T @ (fn + ff))
            linkposes.append(np.r_[poses[index], robot.data.body_quat_w[0, index].cpu().numpy()])
        states = [obj.data.root_state_w[0].cpu().numpy() for _, _, obj in objects]
        reaction = []
        for s in opposites:
            s.update(1 / a.fps, force_recompute=True)
            reaction.append((s.data.force_matrix_w[0, 0].sum(0) + s.data.friction_forces_w[0, 0].sum(0)).cpu().numpy())
        sim.render()
        for _ in range(3):
            if all(
                abs(float(cam.get_current_frame().get("rendering_time", -1)) - sim.current_time) < DT * 0.1
                for cam in cameras.values()
            ):
                break
            sim.render()
        if not camera_clock_ready(
            [cam.get_current_frame().get("rendering_time") for cam in cameras.values()], sim.current_time, DT
        ):
            raise ValueError("Camera rendering clock is stale relative to physical state")
        camera_stamps = {}
        for name, cam in cameras.items():
            rgb = cam.get_rgba()[..., :3].copy()
            depth = cam.get_depth().copy()
            stamp = cam.get_current_frame().get("rendering_time")
            if not rgb.size or not depth.size:
                raise ValueError("Empty camera " + name)
            if np.std(rgb) < 1:
                raise ValueError("Constant image " + name)
            Image.fromarray(rgb).save(a.output / name / f"{frame:05d}.jpg", quality=88)
            append("camera/" + name + "/rgb", rgb)
            append("camera/" + name + "/depth_m", depth)
            append("camera/" + name + "/capture_step", np.int64(step_count))
            camera_stamps[name] = stamp
            pos, quat = cam.get_world_pose(camera_axes="usd")
            append("camera/" + name + "/pose_wxyz", np.r_[pos, quat])
        row = {
            "frame": frame,
            "step": step_count,
            "time_s": sim.current_time,
            "phase": phase,
            "shape": shape,
            "joint_position_rad": q.tolist(),
            "joint_velocity_si": robot.data.joint_vel[0].cpu().tolist(),
            "target_rad": applied[0].cpu().tolist(),
            "requested_target_rad": requested[0].cpu().tolist(),
            "joint_effort_Nm": robot.data.applied_torque[0].cpu().tolist(),
            "force_world_N": np.asarray(forces).tolist(),
            "normal_force_world_N": np.asarray(normal).tolist(),
            "friction_force_world_N": np.asarray(friction).tolist(),
            "force_link_N": np.asarray(locals_).tolist(),
            "link_pose_wxyz": np.asarray(linkposes).tolist(),
            "object_states_si": np.asarray(states).tolist(),
            "object_reaction_force_world_N": np.asarray(reaction).tolist(),
            "wrist_pose_wxyz": {
                side: np.r_[
                    robot.data.body_pos_w[0, robot.body_names.index(side + "_wrist_yaw_link")].cpu().numpy(),
                    robot.data.body_quat_w[0, robot.body_names.index(side + "_wrist_yaw_link")].cpu().numpy(),
                ].tolist()
                for side in ("left", "right")
            },
            "fk_max_error_m": float(max(errors)),
            "camera_sensor_time_s": camera_stamps,
            "support_positions_w_m": {
                side: obj.data.root_pos_w[0].cpu().tolist() for side, obj in support_objects.items()
            },
        }
        trace.write(json.dumps(row) + "\n")
        trace.flush()
        rows.append(row)
        for key, val in [
            ("frame", np.int64(frame)),
            ("step", np.int64(step_count)),
            ("time_s", np.float64(sim.current_time)),
            ("state/position_si", q),
            ("state/velocity_si", robot.data.joint_vel[0].cpu().numpy()),
            ("action/target_si", applied[0].cpu().numpy()),
            ("action/requested_si", requested[0].cpu().numpy()),
            ("contact/force_world_N", forces),
            ("contact/normal_world_N", normal),
            ("contact/friction_world_N", friction),
            ("contact/force_link_N", locals_),
            ("contact/link_pose_wxyz", linkposes),
        ]:
            append(key, val)
        if objects:
            append("object/state_si", states)
            append("contact/object_reaction_N", reaction)
        frame += 1

    prev = q0.clone()
    support_offset = -0.3

    def phase(name, goal, seconds=1.0, shape="none", ramp=0.6, support_shift=None):
        nonlocal prev, step_count, support_offset
        start = prev.clone()
        steady = []
        begin = len(rows)
        steps = round(seconds / DT)
        for k in range(steps):
            alpha = min(1.0, (k + 1) / (steps * ramp))
            alpha = alpha * alpha * (3 - 2 * alpha)
            target = start + (goal - start) * alpha
            if support_shift is not None:
                shift = support_offset + alpha * (support_shift - support_offset)
                for si, obj in enumerate(support_objects.values()):
                    pos = np.asarray(supports[si]).copy()
                    pos[2] += shift
                    pose = torch.tensor([[*pos, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=sim.device)
                    obj.root_physx_view.set_kinematic_targets(
                        native_pose_xyzw(pose), indices=torch.tensor([0], dtype=torch.int32, device=sim.device)
                    )
            if name == "close_" + shape and k > steps * 0.7:
                for side, profile in (grasp_profiles or {}).items():
                    for finger, joint in profile.get("contact_seek_joints", {}).get(shape, {}).items():
                        channel = next(
                            i
                            for i, n in enumerate(channels)
                            if n.startswith("l_" if side == "left" else "r_") and finger in n
                        )
                        sensor = pads[channel]
                        sensor.update(DT, force_recompute=True)
                        force = sensor.data.force_matrix_w[0, 0].sum(0) + sensor.data.friction_forces_w[0, 0].sum(0)
                        if float(torch.linalg.vector_norm(force)) < 0.15:
                            j = names.index(joint)
                            cap = profile["goal_rad"][joint] + 0.22
                            goal[0, j] = min(cap, hand_limits[joint]["upper"], float(goal[0, j]) + 0.0002)
            applied = step(target)
            step_count += 1
            if k >= int(steps * 0.8):
                steady.append(torch.abs(robot.data.joint_pos - goal)[0].cpu().numpy())
            if k % round(1 / (a.fps * DT)) == 0:
                record(name, target, applied, shape)
        report = {
            "phase": name,
            "frames": [begin, len(rows)],
            "max_steady_error_rad": float(np.max(steady)),
            "per_joint_error_rad": dict(zip(names, np.max(steady, axis=0).tolist())),
        }
        reports.append(report)
        if support_shift is not None:
            support_offset = support_shift
        prev = goal.clone()
        (a.output / "progress.json").write_text(
            json.dumps(
                {"phase": name, "frames": frame, "simulation_time_s": sim.current_time, "phase_result": report},
                indent=2,
            )
        )
        print(json.dumps({"phase": name, "frames": frame, "max_error": report["max_steady_error_rad"]}), flush=True)

    if a.mode in ("motion", "smoke"):
        phase("baseline", q0, 1)
        count = 1 if a.mode == "smoke" else max(len(v) for v in ids.values())
        for axis in range(count):
            for sign in (1, -1):
                goal = q0.clone()
                changed = False
                for side, h in contract["hands"].items():
                    n = h["joints"][axis]
                    lim = h.get("effective_active_limits", h["limits"])[n]
                    base = float(q0[0, names.index(n)])
                    bound = lim["upper"] if sign > 0 else lim["lower"]
                    amp = min(0.15, abs(bound - base) * 0.4)
                    if "sharpa" in a.model and n.endswith("_AA"):
                        amp = min(amp, 0.04)
                    if amp < 0.03:
                        continue
                    goal[0, names.index(n)] = base + sign * amp
                    changed = True
                if changed:
                    phase(f"axis_{axis}_{sign:+d}", goal, 1.2)
                    phase(f"return_{axis}_{sign:+d}", q0, 1)
        if a.mode != "smoke":
            for fraction in (0.25, 0.5, 0.8):
                goal = q0.clone()
                for side, h in contract["hands"].items():
                    for n in h["joints"]:
                        if any(x in n for x in ("flex", "pip", "dip", "_FE", "_IP", "_1_joint")):
                            lim = h.get("effective_active_limits", h["limits"])[n]
                            goal[0, names.index(n)] = float(q0[0, names.index(n)]) + fraction * (
                                lim["upper"] - float(q0[0, names.index(n)])
                            )
                phase(f"coordinated_range_{fraction}", goal, 2)
                phase(f"range_return_{fraction}", q0, 1.5)
        phase("final_return", q0, 2)
    else:
        for shape_index, shape in enumerate(("sphere", "cube", "cylinder")):
            goal = q0.clone()
            for side in ("left", "right"):
                for n, v in grasps[side]["goal_rad"].items():
                    closing = any(
                        t in n
                        for t in (
                            "flex",
                            "mcp",
                            "pip",
                            "dip",
                            "_FE",
                            "_IP",
                            "_PIP",
                            "_DIP",
                            "index_1_joint",
                            "thumb_1_joint",
                            "thumb_2_joint",
                        )
                    ) and not any(t in n for t in ("_abd", "_AA", "swing"))
                    if "inspire" in a.model:
                        closing = n == (
                            "left_thumb_1_joint" if side == "left" else "right_thumb_2_joint"
                        ) or n.endswith("index_1_joint")
                    lim = contract["hands"][side].get("effective_active_limits", contract["hands"][side]["limits"])[n]
                    goal[0, names.index(n)] = min(
                        lim["upper"],
                        max(
                            lim["lower"],
                            v + (grasps[side].get("shape_closing_margin_rad", {}).get(shape, 0.10) if closing else 0),
                        ),
                    )
            pre = goal.clone()
            for side in ("left", "right"):
                for n, v in grasps[side]["goal_rad"].items():
                    closing = any(
                        t in n for t in ("flex", "mcp", "pip", "dip", "_FE", "_IP", "_PIP", "_DIP")
                    ) and not any(t in n for t in ("_abd", "_AA", "swing"))
                    if "inspire" in a.model:
                        closing = n == (
                            "left_thumb_1_joint" if side == "left" else "right_thumb_2_joint"
                        ) or n.endswith("index_1_joint")
                    lim = contract["hands"][side].get("effective_active_limits", contract["hands"][side]["limits"])[n]
                    pre[0, names.index(n)] = max(lim["lower"], v - (0.20 if closing else 0))
            # Complete thumb opposition before the object is introduced; only
            # distal flexion closes around the object afterwards.
            for side, h in contract["hands"].items():
                for n in h["joints"]:
                    if "thumb" in n and ("CMC_" in n or "cmc_" in n or "swing" in n):
                        if shape in grasps[side].get("exact_opposition_shapes", []) and n in grasps[side]["goal_rad"]:
                            goal[0, names.index(n)] = grasps[side]["goal_rad"][n]
                        pre[0, names.index(n)] = goal[0, names.index(n)]
            closing_directions = np.sign((goal - pre)[0].cpu().numpy())
            # Each trial starts with a prepared opening around a supported object.
            # Object writes occur only outside its continuous interaction window.
            for side, s, obj in objects:
                pose = torch.tensor(
                    [[2.0, shape_index * 0.1, 0.1, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=sim.device
                )
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
            phase("support_park_" + shape, prev, 1, shape, support_shift=-0.3)
            phase("prepare_" + shape, pre, 2, shape)
            phase("support_ready_" + shape, pre, 1, shape, support_shift=0.0)
            for side, s, obj in objects:
                xyz = placements[side] if s == shape else [2, shape_index * 0.1, 0.1]
                pose = torch.tensor([[*xyz, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=sim.device)
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
                obj.update(DT)
            phase("settle_" + shape, pre, 1, shape)
            phase("close_" + shape, goal, 4, shape)
            phase("support_retract_" + shape, goal, 1, shape, support_shift=-0.12)
            lift = goal.clone()
            for side in ("left", "right"):
                lift[0, names.index(side + "_shoulder_pitch_joint")] -= grasps[side].get("arm_lift_rad", 0.35)
            phase("lift_" + shape, lift, 2, shape)
            phase("hold_" + shape, lift, 1.5, shape)
            phase("support_return_" + shape, lift, 1, shape, support_shift=0.0)
            phase("place_" + shape, goal, 2, shape)
            release = goal.clone()
            for idx in active:
                if closing_directions[idx] != 0:
                    lim = hand_limits[names[idx]]
                    release[0, idx] = np.clip(
                        float(robot.data.joint_pos[0, idx])
                        - closing_directions[idx]
                        * grasps["left" if idx in ids["left"] else "right"].get("release_margin_rad", 0.20),
                        lim["lower"],
                        lim["upper"],
                    )
            phase("release_" + shape, release, 2, shape)
            clear = release.clone()
            for index in active:
                clear[0, index] = q0[0, index]
            phase("release_clear_" + shape, clear, 2, shape)
            phase("recovered_" + shape, clear, 2, shape)
    trace.close()
    data.close()
    canonical = [
        n
        for side in ("left", "right")
        for n in [
            side + "_" + s
            for s in (
                "shoulder_pitch_joint",
                "shoulder_roll_joint",
                "shoulder_yaw_joint",
                "elbow_joint",
                "wrist_roll_joint",
                "wrist_pitch_joint",
                "wrist_yaw_joint",
            )
        ]
        + contract["hands"][side]["joints"]
    ]
    result = {
        "model": a.model,
        "mode": a.mode,
        "frames": frame,
        "joint_names": names,
        "channels": channels,
        "action_joint_names": canonical,
        "urdf_sha256": sha(asset / "robot.urdf"),
        "input_sha256": hashes,
        "phases": reports,
        "camera_metadata": metadata,
        "objects": [{"side": s, "shape": sh} for s, sh, _ in objects],
        "placements": placements,
        "supports": supports,
        "dt_s": DT,
        "fps": a.fps,
        "hardware_connected": False,
        "object_attachment": False,
        "gravity_disabled": False,
        "controller": (
            "position targets; thumb opposition completed before object initialization; "
            "measured release retreat; per-hand grasp profile frozen with run"
        ),
        "support_fixture": {
            "post_size_m": [0.006, 0.006, 0.012],
            "posts_per_hand": 3,
            "post_retract_m": -0.12,
            "receiving_tray_size_m": [0.16, 0.16, 0.012],
            "tray_below_post_m": 0.05,
        },
        "object_pose_writes": (
            "parking/initialization only between independent trials; no writes inside settle-through-recovered window"
        ),
        "finite_state": bool(all(np.isfinite(r["joint_position_rad"]).all() for r in rows)),
        "fk_max_error_m": max(r["fk_max_error_m"] for r in rows),
        "execution_complete": True,
        "acceptance_pending_independent_audit": True,
    }
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "frames": frame, "output": str(a.output)}), flush=True)


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
