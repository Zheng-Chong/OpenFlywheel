"""Project-factory nano glue checks: axial hole, annulus blocking, clamp proxy, reset and reload.

No robot-grasp success, measured adhesive/friction or material calibration claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np


def arr(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    args = parser.parse_args()
    device = args.device
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    report = dict(passed=False, real_material_calibrated=False, robot_grasp_validated=False)
    try:
        import carb

        carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
        carb.settings.get_settings().set_bool("/rtx/translucency/enabled", True)
        carb.settings.get_settings().set_bool("/rtx/reflections/enabled", True)
        carb.settings.get_settings().set_float("/rtx/translucency/worldEps", 0.0)
        from isaaclab.assets import RigidObject as NativeRigid, RigidObjectCfg
        import isaaclab.sim as sim_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from omegaconf import OmegaConf
        from PIL import Image
        from pxr import UsdGeom, UsdLux, UsdPhysics
        import torch

        from env.scene_manager.scene_manager import SceneManager

        meta = json.loads((args.asset / "metadata.json").read_text())
        L = meta["measurements_mm"]["外径"] / 1000
        H = meta["measurements_mm"]["高度"] / 1000
        RI = meta["measurements_mm"]["内径"] / 2000
        world = World(physics_dt=1 / 240, rendering_dt=1 / 30, backend="torch", device=device)
        world.get_physics_context().enable_gpu_dynamics(device != "cpu")
        world.get_physics_context().set_broadphase_type("MBP" if device == "cpu" else "GPU")
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Floor",
                name="floor",
                position=np.array([0, 0, -0.01]),
                scale=np.array([2, 2, 0.02]),
                size=1,
                color=np.array([0.45, 0.45, 0.45]),
            )
        )
        factory = SceneManager.__new__(SceneManager)
        factory.sim = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros((1, 3), device=device)))
        base = "/World/env_0/Object/Rigid/nano_glue_00000_"

        def load(index, pos):
            return factory.instantiate_object_wrapper(
                obj_type="rigid",
                primitive_type=None,
                env_id=0,
                prim_path=base + str(index),
                usd_path=str((args.asset / "object.usdz").resolve()),
                inst_cfg=OmegaConf.create({"physics": {}}),
                default_pos=pos,
                default_ori=(1, 0, 0, 0),
                scale=(1, 1, 1),
            )

        obj = load(1, (0, 0, H / 2 + 0.06))
        other = load(2, (0.35, 0, 0.1))
        stage = get_current_stage()
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(base + "2")).CreateKinematicEnabledAttr(True)

        def spawn(path, shape, pos, kinematic=False):
            cfg = shape(
                mass_props=sim_utils.MassPropertiesCfg(mass=0.002 if not kinematic else 1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=kinematic),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.65, 0.25)),
            )
            return NativeRigid(
                RigidObjectCfg(prim_path=path, spawn=cfg, init_state=RigidObjectCfg.InitialStateCfg(pos=pos))
            )

        spheres = [
            spawn(f"/World/Sphere{i}", lambda **kw: sim_utils.SphereCfg(radius=0.002, **kw), (1, 1 + i * 0.1, 1))
            for i in range(3)
        ]
        pusher = spawn(
            "/World/Pusher", lambda **kw: sim_utils.CuboidCfg(size=(0.012, 0.05, 0.045), **kw), (1.5, 1.5, 1.5), True
        )
        jaws = [
            spawn(
                f"/World/Jaw{i}",
                lambda **kw: sim_utils.CuboidCfg(size=(0.012, 0.035, 0.016), **kw),
                (-0.055 if i == 0 else 0.055, 0, 0.018),
                True,
            )
            for i in range(2)
        ]
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(650)
        sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        sun.CreateIntensityAttr(900)
        camera = Camera("/World/Camera", resolution=(960, 720))
        camera.set_clipping_range(0.001, 10)
        camera.prim.GetAttribute("focalLength").Set(42)
        set_camera_view((-0.13, -0.20, 0.16), (0, 0, H / 2), camera_prim_path=camera.prim_path)
        world.reset()
        obj.initialize()
        other.initialize()
        camera.initialize()

        def step(n):
            for _ in range(n):
                world.step(render=False)

        def place(probe, pos, vel=(0, 0, 0)):
            state = probe.data.default_root_state.clone()
            state[:, :3] = torch.tensor(pos, device=state.device)
            state[:, 7:10] = torch.tensor(vel, device=state.device)
            state[:, 10:] = 0
            if probe.cfg.spawn.rigid_props.kinematic_enabled:
                probe.write_root_pose_to_sim(state[:, :7])
            else:
                probe.write_root_state_to_sim(state)

        def target(probe, pos):
            transforms = probe.root_physx_view.get_transforms().clone()
            transforms[:, :3] = torch.tensor(pos, device=transforms.device)
            probe.root_physx_view.set_kinematic_targets(
                transforms, torch.tensor([0], device=transforms.device, dtype=torch.int32)
            )

        def pp(probe):
            return arr(probe.root_physx_view.get_transforms())[0, :3]

        def capture(name, az=-0.55, elev=0.65, x=0, z=None):
            z = H / 2 if z is None else z
            dist = 0.19
            set_camera_view(
                (x + dist * np.sin(az), -dist * np.cos(az), z + dist * elev),
                (x, 0, z),
                camera_prim_path=camera.prim_path,
            )
            for _ in range(12):
                world.render()
            rgba = camera.get_rgba()
            assert rgba is not None and rgba.shape == (720, 960, 4)
            Image.fromarray(rgba).save(args.output / name)

        def clean_pose():
            world.pause()
            obj.apply_saved_pose()
            obj.set_world_pose(
                position=torch.tensor([0, 0, H / 2 + 0.0002], device=device),
                orientation=torch.tensor([1.0, 0, 0, 0], device=device),
            )
            obj._rigid_prim_view.set_velocities(torch.zeros((1, 6), device=device))
            world.play()
            step(180)

        report["initial_clearance"] = dict(
            factory_initial_pose_m=arr(obj.get_local_pose()[0]).tolist(),
            floor_gap_m=float(arr(obj.get_local_pose()[0])[2] - H / 2),
            second_floor_gap_m=0.1 - H / 2,
            sphere_radius_m=0.002,
            passed=bool(float(arr(obj.get_local_pose()[0])[2] - H / 2) > 0.05 and 0.1 - H / 2 > 0),
            scope="factory initial pose readback and conservative collider bounds; separated external bodies",
        )
        step(720)
        pos, q = map(arr, obj.get_local_pose())
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
        speed = float(np.linalg.norm(arr(obj.get_linear_velocity())))
        report["drop"] = dict(
            position_m=pos.tolist(),
            tilt_deg=tilt,
            speed_m_s=speed,
            height_error_m=float(abs(pos[2] - H / 2)),
            passed=bool(tilt < 5 and abs(pos[2] - H / 2) < 0.005 and speed < 0.01),
        )
        actual_mass = float(arr(obj.get_mass()))
        inertia = arr(obj._rigid_prim_view.get_inertias()).reshape(3, 3)
        expected = np.array(meta["inertia_kg_m2"])
        report["mass_inertia"] = dict(
            actual_mass_kg=actual_mass,
            inertia_kg_m2=inertia.tolist(),
            expected_inertia_kg_m2=expected.tolist(),
            passed=bool(
                abs(actual_mass - 0.0918) < 1e-6
                and np.linalg.eigvalsh(inertia).min() > 0
                and np.max(abs(inertia - expected)) < 1e-8
            ),
            basis="mass measured; inertia from homogeneous reconstructed solid, not measured",
        )
        report["second_factory_load"] = dict(
            mass_kg=float(arr(other.get_mass())),
            wrapper=type(other).__name__,
            passed=bool(abs(float(arr(other.get_mass())) - 0.0918) < 1e-6),
            scope="independent held wrapper for aperture probes",
        )
        for index in range(2):
            UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Jaw{index}")).MakeInvisible()
        for name, az, elev in [
            ("sim.png", -0.55, 0.65),
            ("reference_1.png", 0, 0.65),
            ("reference_2.png", 0, 0.78),
            ("reference_3.png", 0, 1.35),
            ("front.png", 0, 0.05),
            ("top.png", 0, 2.8),
            ("back_missing_photo.png", 3.14, 0.35),
            ("right_missing_photo.png", 1.57, 0.35),
        ]:
            capture(name, az, elev)
        for sphere, x in zip(spheres, [0.35, 0.35 + (RI + L / 2) / 2, 0.35 + L / 2 + 0.008]):
            place(sphere, [x, 0, 0.17], [0, 0, -0.2])
        step(240)
        positions = [pp(s).tolist() for s in spheres]
        report["aperture"] = dict(
            positions_m=positions,
            radius_m=0.002,
            opening_passed=positions[0][2] < 0.06,
            annulus_blocked=positions[1][2] > 0.1 + H / 2,
            outside_passed=positions[2][2] < 0.06,
        )
        report["aperture"]["passed"] = all(
            report["aperture"][key] for key in ["opening_passed", "annulus_blocked", "outside_passed"]
        )
        capture("aperture_probes.png", x=0.35, z=0.1)
        for i, s in enumerate(spheres):
            place(s, [1, 1 + i * 0.1, 1])
        before = arr(obj.get_local_pose()[0]).copy()
        for i in range(241):
            place(pusher, [-0.07 + 0.07 * i / 240, 0, H / 2])
            step(1)
        after = arr(obj.get_local_pose()[0])
        report["solid_blocking"] = dict(
            before_m=before.tolist(), after_m=after.tolist(), passed=bool(after[0] - before[0] > 0.008)
        )
        place(pusher, [1.5, 1.5, 1.5])
        clean_pose()
        negative_start = arr(obj.get_local_pose()[0]).copy()
        for i in range(241):
            for jaw, sign in zip(jaws, [-1, 1]):
                target(jaw, [sign * 0.07, 0, 0.018 + 0.05 * i / 240])
            step(1)
        negative_end = arr(obj.get_local_pose()[0]).copy()
        report["open_jaw_control"] = dict(
            before_m=negative_start.tolist(),
            after_m=negative_end.tolist(),
            passed=bool(abs(negative_end[2] - negative_start[2]) < 0.002),
            scope="open jaws lifted 50 mm without object contact; object must remain on floor",
        )
        for jaw, sign in zip(jaws, [-1, 1]):
            target(jaw, [sign * 0.055, 0, 0.018])
        step(240)
        # Opposed kinematic jaws provide an explicit contact/grasp proxy. No
        # forces or measured actuator model are asserted by this diagnostic.
        for index in range(2):
            UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Jaw{index}")).MakeVisible()
        jaw_z = 0.018
        for i in range(181):
            x = 0.055 - (0.055 - (L / 2 + 0.006 - 0.00025)) * i / 180
            for jaw, sign in zip(jaws, [-1, 1]):
                target(jaw, [sign * x, 0, jaw_z])
            step(1)
        step(120)
        before_lift = arr(obj.get_local_pose()[0]).copy()
        samples = []
        for i in range(241):
            lift = 0.05 * i / 240
            for jaw, sign in zip(jaws, [-1, 1]):
                target(jaw, [sign * (L / 2 + 0.006 - 0.00025), 0, jaw_z + lift])
            step(1)
            if i % 12 == 0:
                samples.append(dict(jaw_lift_m=lift, object_position_m=arr(obj.get_local_pose()[0]).tolist()))
        step(480)
        held = arr(obj.get_local_pose()[0])
        lift = float(held[2] - before_lift[2])
        slip = abs(0.05 - lift)
        report["clamp_lift"] = dict(
            before_m=before_lift.tolist(),
            held_m=held.tolist(),
            lift_m=lift,
            slip_m=slip,
            hold_seconds=2.0,
            jaw_overlap_m=0.00025,
            samples=samples,
            passed=bool(lift > 0.035 and slip < 0.005),
            scope="two prescribed kinematic cuboid jaws; no robot, contact force or real adhesive slip calibration",
        )
        capture("clamp_lift.png", z=float(held[2]))
        for jaw, sign in zip(jaws, [-1, 1]):
            place(jaw, [sign * 0.07, 0, jaw_z + 0.05])
        step(720)
        released = arr(obj.get_local_pose()[0])
        report["clamp_lift"]["release_height_m"] = float(released[2])
        report["clamp_lift"]["passed"] &= bool(
            released[2] < 0.03 and np.linalg.norm(arr(obj.get_linear_velocity())) < 0.01
        )
        for i, jaw in enumerate(jaws):
            place(jaw, [1.5, 1.5 + i * 0.1, 1.5])
        clean_pose()
        slide_start = arr(obj.get_local_pose()[0]).copy()
        obj._rigid_prim_view.set_velocities(torch.tensor([[0.25, 0, 0, 0, 0, 0]], device=device))
        slide_samples = []
        for i in range(120):
            step(1)
            if i % 10 == 0:
                slide_samples.append(
                    dict(
                        t_s=(i + 1) / 240,
                        position_m=arr(obj.get_local_pose()[0]).tolist(),
                        velocity_m_s=arr(obj.get_linear_velocity()).tolist(),
                    )
                )
        slide_end = arr(obj.get_local_pose()[0])
        end_speed = float(np.linalg.norm(arr(obj.get_linear_velocity())))
        report["sliding"] = dict(
            start_m=slide_start.tolist(),
            end_m=slide_end.tolist(),
            initial_velocity_m_s=0.25,
            displacement_m=float(slide_end[0] - slide_start[0]),
            end_speed_m_s=end_speed,
            samples=slide_samples,
            passed=bool(slide_end[0] - slide_start[0] > 0.001 and end_speed < 0.01),
            scope="flat-floor free sliding at runtime defaults; no friction coefficient identified",
        )
        world.pause()
        obj.set_world_pose(
            position=torch.tensor([0.1, -0.1, 0.3], device=device),
            orientation=torch.tensor([0.9238795, 0, 0, 0.3826834], device=device),
        )
        obj._rigid_prim_view.set_velocities(torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.1, 0.3]], device=device))
        injected = arr(obj._rigid_prim_view.get_velocities())
        obj.apply_saved_pose()
        rp, rq = map(arr, obj.get_local_pose())
        rv = arr(obj._rigid_prim_view.get_velocities())
        ep = float(np.max(abs(rp - np.array(obj.default_pos))))
        eq = float(min(np.linalg.norm(rq - [1, 0, 0, 0]), np.linalg.norm(rq + [1, 0, 0, 0])))
        report["reset"] = dict(
            position_error_m=ep,
            quaternion_error=eq,
            velocity=rv.tolist(),
            injected_velocity=injected.tolist(),
            passed=bool(ep < 1e-5 and eq < 1e-5 and np.max(abs(rv)) < 1e-5 and np.linalg.norm(injected) > 0.2),
        )
        world.play()
        step(720)
        report["reset"]["resettled_z_m"] = float(arr(obj.get_local_pose()[0])[2])
        report["reset"]["passed"] &= abs(report["reset"]["resettled_z_m"] - H / 2) < 0.005
        report.update(
            asset_id="Rigid/nano_glue/00000",
            backend=f"project SceneManager / PhysX {device}",
            physics_dt_s=1 / 240,
            asset_file_sha256={
                name: hashlib.sha256((args.asset / name).read_bytes()).hexdigest()
                for name in ["object.usdz", "metadata.json", "runtime_config.json", "description.json"]
            },
        )
        report["passed"] = all(
            report[key]["passed"]
            for key in [
                "initial_clearance",
                "drop",
                "mass_inertia",
                "second_factory_load",
                "aperture",
                "solid_blocking",
                "open_jaw_control",
                "clamp_lift",
                "sliding",
                "reset",
            ]
        )
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (args.output / "report.json").write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist(),
            )
            + "\n"
        )
        print(
            json.dumps(report, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()),
            flush=True,
        )
        if not report["passed"]:
            import os
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        app.close(skip_cleanup=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
