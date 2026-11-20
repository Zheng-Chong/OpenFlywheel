"""Native project factory validation; run once per cup variant, preserve all reports."""

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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    report = dict(passed=False, real_material_calibrated=False, robot_grasp_validated=False)
    try:
        import carb

        carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
        from isaaclab.assets import RigidObject as NativeRigid, RigidObjectCfg
        import isaaclab.sim as sim_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from omegaconf import OmegaConf
        from PIL import Image
        from pxr import UsdLux, UsdPhysics
        import torch

        from env.scene_manager.scene_manager import SceneManager

        meta = json.loads((args.asset / "metadata.json").read_text())
        h = 0.09301
        world = World(physics_dt=1 / 240, rendering_dt=1 / 30, backend="torch", device="cuda:0")
        world.get_physics_context().enable_gpu_dynamics(True)
        world.get_physics_context().set_broadphase_type("GPU")
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
        factory.sim = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros((1, 3), device="cuda:0")))
        path = f"/World/env_0/Object/Rigid/wave_curve_cup_{args.asset.name}_1"

        def load(prim, pos):
            return factory.instantiate_object_wrapper(
                obj_type="rigid",
                primitive_type=None,
                env_id=0,
                prim_path=prim,
                usd_path=str((args.asset / "object.usdz").resolve()),
                inst_cfg=OmegaConf.create({"physics": {}}),
                default_pos=pos,
                default_ori=(1, 0, 0, 0),
                scale=(1, 1, 1),
            )

        obj = load(path, (0, 0, h / 2 + 0.06))
        other_path = f"/World/env_0/Object/Rigid/wave_curve_cup_{args.asset.name}_2"
        other = load(other_path, (0.35, 0, 0.2))
        stage = get_current_stage()
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(other_path)).CreateKinematicEnabledAttr(True)
        probes = []
        for i in range(4):
            probes.append(
                NativeRigid(
                    RigidObjectCfg(
                        prim_path=f"/World/Probe{i}",
                        spawn=sim_utils.SphereCfg(
                            radius=0.004,
                            mass_props=sim_utils.MassPropertiesCfg(mass=0.002),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=i not in (0, 3)),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.75, 0.25)),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(1, 1 + i * 0.1, 1)),
                    )
                )
            )
        pusher = NativeRigid(
            RigidObjectCfg(
                prim_path="/World/Pusher",
                spawn=sim_utils.CuboidCfg(
                    size=(0.02, 0.1, 0.15),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.5, 1.5)),
            )
        )
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(650)
        # Neutral fixed lighting for all variants and all cameras.
        sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        sun.CreateIntensityAttr(900)
        camera = Camera("/World/Camera", resolution=(960, 960))
        camera.set_clipping_range(0.001, 10)
        camera.prim.GetAttribute("focalLength").Set(42)
        set_camera_view((0, -0.35, 0.25), (0, 0, h / 2), camera_prim_path=camera.prim_path)
        world.reset()
        obj.initialize()
        other.initialize()
        camera.initialize()

        def step(n):
            for _ in range(n):
                world.step(render=False)

        def place(probe, pos, velocity=(0, 0, 0)):
            state = probe.data.default_root_state.clone()
            state[:, :3] = torch.tensor(pos, device=state.device)
            state[:, 7:10] = torch.tensor(velocity, device=state.device)
            state[:, 10:] = 0
            probe.write_root_state_to_sim(state)

        def pp(probe):
            return arr(probe.root_physx_view.get_transforms())[0, :3]

        def capture(name, azimuth=0, elevation=0.55, x=0, z=h / 2):
            dist = 0.31
            set_camera_view(
                (x + dist * np.sin(azimuth), -dist * np.cos(azimuth), z + dist * elevation),
                (x, 0, z),
                camera_prim_path=camera.prim_path,
            )
            for _ in range(12):
                world.render()
            rgba = camera.get_rgba()
            assert rgba is not None and rgba.shape == (960, 960, 4)
            Image.fromarray(rgba).save(args.output / name)

        step(720)
        pos, q = map(arr, obj.get_local_pose())
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
        speed = float(np.linalg.norm(arr(obj.get_linear_velocity())))
        report["drop"] = dict(
            position_m=pos.tolist(),
            tilt_deg=tilt,
            speed_m_s=speed,
            half_height_error_m=float(abs(pos[2] - h / 2)),
            passed=bool(tilt < 5 and abs(pos[2] - h / 2) < 0.005 and speed < 0.01),
        )
        report["mass"] = dict(
            expected_kg=meta["physics"]["mass"],
            actual_kg=float(arr(obj.get_mass())),
            passed=bool(abs(float(arr(obj.get_mass())) - meta["physics"]["mass"]) < 1e-5),
        )
        report["second_factory_load"] = dict(
            wrapper=type(other).__name__,
            mass_kg=float(arr(other.get_mass())),
            passed=bool(abs(float(arr(other.get_mass())) - meta["physics"]["mass"]) < 1e-5),
            scope="second native wrapper; held kinematic only for independent cavity probes",
        )
        for name, az, elev in [
            ("sim.png", 0, 0.55),
            ("reference_view.png", 0, 0.65),
            ("side.png", 1.57, 0.1),
            ("back_missing_photo.png", 3.14, 0.4),
            ("top.png", 0, 1.5),
        ]:
            capture(name, az, elev)
        # Open-mouth drop and identical-radius probes aimed at side and bottom.
        place(probes[0], [0.35, 0, 0.2 + h / 2 + 0.04])
        place(probes[1], [0.27, 0, 0.2], [0.2, 0, 0])
        place(probes[2], [0.35, 0, 0.10], [0, 0, 0.2])
        step(360)
        positions = [pp(probe).tolist() for probe in probes[:3]]
        expected_z = 0.2 - h / 2 + 0.003 + 0.004
        report["container"] = dict(
            positions_m=positions,
            sphere_radius_m=0.004,
            opening_and_retention=bool(
                np.linalg.norm(np.array(positions[0])[:2] - [0.35, 0]) < 0.015
                and abs(positions[0][2] - expected_z) < 0.004
            ),
            wall_blocks=bool(positions[1][0] < 0.325),
            bottom_blocks=bool(positions[2][2] < 0.2 - h / 2 + 0.001),
            scope="factory-loaded floating kinematic cup, dynamic spheres; no fluid claim",
        )
        report["container"]["passed"] = all(
            report["container"][key] for key in ["opening_and_retention", "wall_blocks", "bottom_blocks"]
        )
        capture("container_probes.png", x=0.35, z=0.2)
        for i, probe in enumerate(probes):
            place(probe, [1, 1 + i * 0.1, 1])
        place(probes[3], [0, 0, h + 0.04])
        step(480)
        free_ball = pp(probes[3])
        free_cup = arr(obj.get_local_pose()[0])
        report["dynamic_retention"] = dict(
            sphere_position_m=free_ball.tolist(),
            cup_position_m=free_cup.tolist(),
            passed=bool(np.linalg.norm(free_ball[:2] - free_cup[:2]) < 0.025 and 0.005 < free_ball[2] < 0.020),
        )
        capture("dynamic_retention.png")
        place(probes[3], [1, 1.3, 1])
        before = arr(obj.get_local_pose()[0]).copy()
        for j in range(241):
            place(pusher, [-0.08 + 0.08 * j / 240, 0, h / 2])
            step(1)
        after = arr(obj.get_local_pose()[0])
        report["solid_blocking"] = dict(
            before_m=before.tolist(), after_m=after.tolist(), passed=bool(after[0] - before[0] > 0.008)
        )
        place(pusher, [1.5, 1.5, 1.5])
        world.pause()
        obj.set_world_pose(
            position=torch.tensor([0.1, -0.1, 0.3], device="cuda:0"),
            orientation=torch.tensor([0.9238795, 0, 0, 0.3826834], device="cuda:0"),
        )
        obj._rigid_prim_view.set_velocities(torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.1, 0.3]], device="cuda:0"))
        pre_velocity = arr(obj._rigid_prim_view.get_velocities())
        obj.apply_saved_pose()
        reset_pos, reset_q = map(arr, obj.get_local_pose())
        vel = arr(obj._rigid_prim_view.get_velocities())
        epos = float(np.max(abs(reset_pos - np.array(obj.default_pos))))
        eq = float(min(np.linalg.norm(reset_q - [1, 0, 0, 0]), np.linalg.norm(reset_q + [1, 0, 0, 0])))
        report["reset"] = dict(
            position_error_m=epos,
            quaternion_error=eq,
            velocity=vel.tolist(),
            injected_velocity=pre_velocity.tolist(),
            passed=bool(epos < 1e-5 and eq < 1e-5 and np.max(abs(vel)) < 1e-5 and np.linalg.norm(pre_velocity) > 0.2),
        )
        world.play()
        step(720)
        report["reset"]["resettled_z_m"] = float(arr(obj.get_local_pose()[0])[2])
        report["reset"]["passed"] &= abs(report["reset"]["resettled_z_m"] - h / 2) < 0.005
        report.update(
            asset_id=meta["asset_id"],
            wrapper=type(obj).__name__,
            physics_dt_s=1 / 240,
            backend="project SceneManager / PhysX GPU rigid body",
            asset_file_sha256={
                name: hashlib.sha256((args.asset / name).read_bytes()).hexdigest()
                for name in ["object.usdz", "metadata.json", "description.json", "runtime_config.json"]
            },
        )
        report["passed"] = all(
            report[key]["passed"]
            for key in [
                "drop",
                "mass",
                "second_factory_load",
                "container",
                "dynamic_retention",
                "solid_blocking",
                "reset",
            ]
        )
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        app.close(skip_cleanup=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
