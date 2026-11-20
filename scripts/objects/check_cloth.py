"""Physical checks of a particle-cloth Garment asset through GarmentObject, the loader tasks use (GPU, one case a run).

Cases (each in a fresh process; recorded nodes go to <output>/nodes.npz for rendering and replay comparison):
  drop   the cloth falls 5 cm onto a table, settles 2 s; then GarmentObject.reset() and the same drop again
  pinch  two velocity-steered friction pads (mu 1.0, 60 x 60 mm) close on the middle of one edge, no node pinning:
         the lower jaw supports it, the upper one closes in 0.25 s; they hold to 1 s, lift 0.15 m in 1.5 s, hold 1.5 s
  fold   the cloth lies on a table with its -X edge 30 mm over the table edge; two friction grippers (lower + upper
         pad each) pinch the two -X corners, turn over like a wrist along a half ellipse over the cloth's middle and
         set them on the +X corners; then the jaws vanish (idealized release, see the fold case); checks corner
         alignment, layer order and spring-back
Writes <output>/report.json. Exit 0 also when a check fails: read "passed" and the per-check values.

    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. python scripts/objects/check_cloth.py ASSET_DIR --case drop --output OUT
Run on physical GPU 0 without CUDA_VISIBLE_DEVICES: on <gpu-server> the particle-cloth tensor view fails on other devices
(illegal memory access, or no view) in Isaac Sim 5.1. --runtime replaces the asset's runtime_config.json (tuning).
"""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FRAME_HZ = 60
PINCH_PAD = (0.06, 0.06, 0.012)  # friction pad box (m)
FOLD_PAD = (0.03, 0.04, 0.006)  # thin enough to sit between the two folded layers
FOLD_OVERHANG = 0.03
FOLD_SLACK = 0.04  # m, the carry arc's top below taut (see the fold case); 50 mm let the 30 cm towel tuck once in 2
PAD_MU = 1.0
PAD_MASS = 50.0  # kg: velocity-steered jaws the cloth cannot push aside (a light jaw lets particles into it)
TABLE_MU = 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", type=Path)
    p.add_argument("--case", choices=["drop", "pinch", "fold"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physics-hz", type=int, default=240)
    p.add_argument("--runtime", type=Path, help="runtime_config.json to use instead of the asset's")
    p.add_argument("--max-strain", type=float, default=0.30, help="largest allowed edge strain in pinch and fold")
    p.add_argument("--gap", type=float, default=0.0003, help="closed jaw face gap (m), below 2 x rest offset = squeeze")
    p.add_argument("--lower-mu", type=float, default=PAD_MU, help="friction of the lower jaws (fold pulls them out)")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    report = {"case": a.case, "asset": str(a.asset.resolve()), "passed": False, "material_calibrated": False}
    try:
        run(a, report)
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("REPORT", json.dumps({k: v for k, v in report.items() if not isinstance(v, list | dict)}), flush=True)
        app.close()


def run(a, report):
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    import numpy as np
    from omegaconf import OmegaConf
    from pxr import PhysxSchema, UsdPhysics
    import torch

    from env.scene_manager.objects.garment import GarmentObject

    meta = json.loads((a.asset / "metadata.json").read_text())
    runtime = json.loads((a.runtime or a.asset / "runtime_config.json").read_text())
    mesh = np.load(a.asset / "mesh.npz")
    faces = mesh["faces"]
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    rest = np.linalg.norm(mesh["points"][edges[:, 0]] - mesh["points"][edges[:, 1]], axis=1)
    thread = rest < 1.05 * rest.min()  # grid edges along the threads; the rest are cell diagonals (shear)
    length, width = meta["physics"]["size"][:2]

    world = World(physics_dt=1 / a.physics_hz, rendering_dt=1 / FRAME_HZ, backend="torch", device="cuda:0")
    context = world.get_physics_context()
    context.enable_gpu_dynamics(True)
    context.set_broadphase_type("GPU")
    table_mat = PhysicsMaterial("/World/Materials/table", static_friction=TABLE_MU, dynamic_friction=TABLE_MU)
    table_x = (-length / 2 + FOLD_OVERHANG + 0.65) if a.case == "fold" else 0.0  # fold: the -X edge overhangs
    FixedCuboid(
        "/World/Table",
        position=np.array([table_x, 0, -0.025]),
        scale=np.array([1.3, 1.0, 0.05]),
        physics_material=table_mat,
    )
    z0 = {"drop": 0.05, "pinch": length + 0.10, "fold": 0.0006}[a.case]  # pinch: the hanging cloth clears the table
    cloth = GarmentObject(
        "/World/env_0/Object/Garment/cloth_00000",
        str((a.asset / "object.usd").resolve()),
        OmegaConf.create(runtime),
        torch.zeros(3, device="cuda:0"),
        [0.0, 0.0, z0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )
    cloth.particle_system.set_simulation_owner(context.prim_path)
    pad_size = {"pinch": PINCH_PAD, "fold": FOLD_PAD}.get(a.case)
    names = {"pinch": ["PadLower", "PadUpper"], "fold": ["LowerA", "UpperA", "LowerB", "UpperB"]}.get(a.case, [])
    pads = []
    if names:
        upper_mat = PhysicsMaterial("/World/Materials/pad", static_friction=PAD_MU, dynamic_friction=PAD_MU)
        lower_mat = PhysicsMaterial("/World/Materials/lower", static_friction=a.lower_mu, dynamic_friction=a.lower_mu)
        for k, name in enumerate(names):
            pad = DynamicCuboid(
                f"/World/{name}",
                name=name,
                position=np.array([0, k * 0.2, 1.0]),
                scale=np.array(pad_size),
                mass=PAD_MASS,
                physics_material=lower_mat if "Lower" in name else upper_mat,
            )
            PhysxSchema.PhysxRigidBodyAPI.Apply(pad.prim).CreateDisableGravityAttr(True)
            pads.append(world.scene.add(pad))  # in the scene: reset() gives it a physics view, else poses stay USD-only
        for pad in pads:  # jaws do not push each other apart
            others = [p.prim_path for p in pads if p is not pad]
            UsdPhysics.FilteredPairsAPI.Apply(pad.prim).CreateFilteredPairsRel().SetTargets(others)
    world.reset()
    cloth.initialize()

    def nodes():
        return cloth._cloth_prim_view.get_world_positions().detach().cpu().numpy().reshape(-1, 3)

    initial = nodes().copy()
    masses = cloth._cloth_prim_view.get_particle_masses().detach().cpu().numpy()
    report.update(
        nodes=len(initial),
        mass_expected_kg=meta["physics"]["mass"],
        mass_readback_kg=float(masses.sum()),
        # float32 per-particle masses: ~1e-5 relative after summing a few thousand of them
        mass_matches=bool(abs(masses.sum() / meta["physics"]["mass"] - 1) < 1e-4),
        physics_hz=a.physics_hz,
        frame_hz=FRAME_HZ,
        runtime_config=runtime,
        mesh_sha256=meta["deformable"]["mesh_sha256"],
    )
    sub = a.physics_hz // FRAME_HZ
    trajectory, pad_track, pad_actual = [], [], []  # pad_actual rows: (x, y, z, pitch) per pad
    finite, max_strain, p99_strain, shear = True, 0.0, 0.0, 0.0
    stretch_track = []

    placed = False

    def vec(v):
        return torch.tensor(v, dtype=torch.float32, device="cuda:0")

    def pitch_quat(t):  # rotation by t about +Y, wxyz
        return np.array([np.cos(t / 2), 0.0, np.sin(t / 2), 0.0])

    def pose(pad):
        p, q = (v.detach().cpu().numpy() for v in pad.get_world_pose())
        return p, 2 * np.arctan2(q[2], q[0])

    def frame(centers=(), pitch=0.0):
        """One 1/60 s frame; centers: one (x, y, z) per pad, pitch: their rotation about +Y (rad), both reached
        linearly over the frame's physics steps.

        Pads are gravity-free dynamic bodies steered by velocity each step, so they carry the velocity friction needs
        (a teleported or kinematic-by-pose body has none and drags nothing sideways), like robot finger links.
        Only the first placement teleports, before the pads touch anything.
        Strain is tensile strain of the thread edges (compression is buckling, not stretch); diagonals are shear."""
        nonlocal finite, max_strain, p99_strain, shear, placed
        targets = [np.asarray(c, dtype=float) for c in centers]
        if targets and not placed:
            for pad, c in zip(pads, targets):
                pad.set_world_pose(position=c, orientation=pitch_quat(pitch))
                pad.set_linear_velocity(vec(np.zeros(3)))
            starts, pitch0, placed = targets, pitch, True
        elif targets:
            starts = [np.asarray(row, dtype=float) for row in pad_track[-1][0]]
            pitch0 = pad_track[-1][1]
        else:
            starts = []
        for k in range(sub):
            f = (k + 1) / sub
            for pad, a0, c in zip(pads, starts, targets):
                here, turned = pose(pad)
                pad.set_linear_velocity(vec((a0 + (c - a0) * f - here) * a.physics_hz))
                error = (pitch0 + (pitch - pitch0) * f - turned + np.pi) % (2 * np.pi) - np.pi  # pi and -pi are one
                pad.set_angular_velocity(vec([0.0, error * a.physics_hz, 0.0]))
            world.step(render=False)
        x = nodes()
        finite = finite and bool(np.isfinite(x).all())
        strain = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1) / rest - 1
        tensile = np.maximum(strain[thread], 0)
        max_strain = max(max_strain, float(tensile.max()))
        p99_strain = max(p99_strain, float(np.percentile(tensile, 99)))
        shear = max(shear, float(np.abs(strain[~thread]).max()))
        stretch_track.append(float(tensile.max()))
        trajectory.append(x.astype(np.float32))
        pad_track.append(([list(map(float, c)) for c in centers], pitch))
        pad_actual.append([[*pose(pad)[0].tolist(), float(pose(pad)[1])] for pad in pads] if placed else [])
        return x

    def pad_penetration():
        traj, deepest = np.asarray(trajectory), 0.0
        half = np.array(pad_size) / 2
        for k, rows in enumerate(pad_actual):
            for *c, t in rows:
                local = (traj[k] - np.array(c)) @ np.array(
                    [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
                )  # rows @ R_y(t): each node in the pad's frame
                d = half - np.abs(local)
                hit = (d > 0).all(axis=1)
                if hit.any():
                    deepest = max(deepest, float(d[hit].min(axis=1).max()))
        return deepest

    start = time.perf_counter()
    if a.case == "drop":
        for _ in range(2 * FRAME_HZ):
            x = frame()
        settled = x.copy()
        first = np.asarray(trajectory)
        cloth.reset()
        world.step(render=False)  # let the restored positions reach the readback buffers
        reset_error = float(np.linalg.norm(nodes() - initial, axis=1).max())
        trajectory.clear()
        pad_track.clear()
        pad_actual.clear()
        for _ in range(2 * FRAME_HZ):
            frame()
        second = np.asarray(trajectory)
        rerun = np.linalg.norm(second - first, axis=-1)
        extent = np.ptp(settled[:, :2], axis=0)
        report.update(
            min_z_m=float(min(first[..., 2].min(), second[..., 2].min())),
            settled_mean_z_m=float(settled[:, 2].mean()),
            settled_z_span_m=float(np.ptp(settled[:, 2])),
            settled_extent_ratio=[float(extent[0] / length), float(extent[1] / width)],
            reset_node_max_error_m=reset_error,
            redrop_max_node_error_m=float(rerun.max()),
            redrop_final_max_node_error_m=float(rerun[-1].max()),
        )
        trajectory[:] = list(first) + list(second)
        pad_track[:] = [[] for _ in trajectory]
        pad_actual[:] = [[] for _ in trajectory]
        checks = {
            "mass": report["mass_matches"],
            "finite": finite,
            "no_table_penetration": report["min_z_m"] > -0.001,
            "lies_flat": report["settled_z_span_m"] < 0.005 and min(report["settled_extent_ratio"]) > 0.97,
            "edge_strain_below_10pct": max_strain < 0.10,
            "reset_restores_nodes": reset_error < 0.001,  # the platform's 1 mm replay threshold
        }
    elif a.case == "pinch":
        # pads pinch a 60 x 60 mm patch at the middle of the -X edge, like a gripper; the rest hangs below.
        # The lower jaw supports the cloth 0.3 mm (its rest offset) below it; the upper jaw comes down from 12 mm
        # in 0.25 s to a 0.3 mm face gap, below the cloth's 2 x 0.3 mm rest offsets: a small normal load.
        pad_x = -length / 2 + PINCH_PAD[0] / 2 - 0.005
        lower_top = z0 - 0.0003
        lower = lower_top - PINCH_PAD[2] / 2
        upper_open, upper = lower_top + 0.012 + PINCH_PAD[2] / 2, lower_top + a.gap + PINCH_PAD[2] / 2
        held = np.flatnonzero(
            (np.abs(initial[:, 0] - pad_x) < PINCH_PAD[0] / 2 - 0.002)
            & (np.abs(initial[:, 1]) < PINCH_PAD[1] / 2 - 0.002)
        )
        close_frames = FRAME_HZ // 4

        def jaws(dz, up):
            return [(pad_x, 0.0, lower + dz), (pad_x, 0.0, up + dz)]

        for i in range(FRAME_HZ):
            frame(jaws(0.0, upper_open + (upper - upper_open) * min(1.0, (i + 1) / close_frames)))
        before = trajectory[-1][held].mean(0)
        lift_frames = int(1.5 * FRAME_HZ)
        for i in range(lift_frames):
            frame(jaws(0.15 * (i + 1) / lift_frames, upper))
        for _ in range(int(1.5 * FRAME_HZ)):
            x = frame(jaws(0.15, upper))
        after = x[held].mean(0)
        hanging = x[:, 2].min()
        report.update(
            held_nodes=len(held),
            held_rise_m=float(after[2] - before[2]),
            pad_rise_m=0.15,
            slip_m=float(0.15 - (after[2] - before[2])),
            max_pad_penetration_m=pad_penetration(),
            final_lowest_node_above_table_m=float(hanging),
            hanging_drop_m=float(after[2] - hanging),
            min_z_m=float(np.asarray(trajectory)[..., 2].min()),
        )
        checks = {
            "mass": report["mass_matches"],
            "finite": finite,
            "held_by_friction": report["slip_m"] < 0.01,
            "pad_penetration_below_1mm": report["max_pad_penetration_m"] < 0.001,
            "edge_strain_within_limit": max_strain < a.max_strain,
            "no_table_penetration": report["min_z_m"] > -0.001,
        }
    else:
        # grip points 12 mm in from the -X edge, 17 mm in from the sides; the fold line is x = 0
        gx, gy = -length / 2 + 0.012, width / 2 - 0.017
        # jaw centres are offsets from the grip point on the cloth's midplane along the jaws' normal; the grippers turn
        # with the arc (pitch pi s about +Y), like a wrist, so the gripped edge flips over with the flap (jaws that
        # only translate curl the cloth into a U at the grip, which unwinds at random once they let go)
        lower_top = z0 - 0.0003  # the overhanging corners rest on the lower jaws, flush with the table top
        below = z0 - (lower_top - FOLD_PAD[2] / 2)
        closed = lower_top + a.gap + FOLD_PAD[2] / 2 - z0
        opened = closed + 0.012
        # set down with the (now lower) upper jaw between the layers, its underside 0.6 mm (rest offset + margin)
        # above the bottom layer: the carried corner lies one jaw thickness up until the jaws let go
        place_dz = FOLD_PAD[2] + 0.0006
        radius = -gx

        def grippers(x, dz, low, up, t=0.0):
            n = np.array([np.sin(t), 0.0, np.cos(t)])  # the jaws' normal, +Z before the turn
            return [g + s * n for g in (np.array([x, -gy, z0 + dz]), np.array([x, gy, z0 + dz])) for s in (-low, up)]

        settle0 = 15
        for i in range(settle0 + FRAME_HZ // 4):  # settle on the jaws, then close in 0.25 s
            s = max(0, i - settle0 + 1) / (FRAME_HZ // 4)
            frame(grippers(gx, 0.0, below, opened + (closed - opened) * s))
        for _ in range(FRAME_HZ // 4):
            frame(grippers(gx, 0.0, below, closed))
        carry = int(2 * FRAME_HZ * max(1.0, radius / 0.138))  # 2 s for a 30 cm cloth; same peak speed when longer
        # Half ellipse over the fold line, its top FOLD_SLACK below the taut half circle. Taut, the gripped cloth drags
        # the fold line 15-30 mm along and the top layer gets the excess, which shoves the corners off on release;
        # with more slack (135 mm on the 70 cm cloth) the hanging flap drapes onto the bottom layer first and tucks.
        for i in range(carry):  # smooth start and stop
            s = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / carry)
            dz = (radius - FOLD_SLACK) * np.sin(np.pi * s) + (place_dz + 0.01) * s
            frame(grippers(-radius * np.cos(np.pi * s), dz, below, closed, np.pi * s), np.pi * s)
        for i in range(FRAME_HZ // 4):  # set it down slowly over the last 10 mm
            frame(grippers(radius, place_dz + 0.01 * (1 - (i + 1) / (FRAME_HZ // 4)), below, closed, np.pi), np.pi)
        # Idealized release: the jaws vanish (a teleport carries no velocity, so it drags nothing). Moving a jaw off
        # the cloth instead takes cloth with it: sliding the one between the layers out carries the whole flap
        # 40-55 mm, lifting the top one off can pull a node up to 66 % strain. Particle friction is position based
        # and sticks to a jaw that moves less per step than friction x contact correction, whatever the load on it.
        # That is a gripper-release problem for robot tasks, not a property of the cloth.
        out = trajectory[-1].copy()
        for k, pad in enumerate(pads):
            pad.set_world_pose(position=np.array([0.0, k * 0.2, 1.0]))
            pad.set_linear_velocity(vec(np.zeros(3)))
        for _ in range(int(1.5 * FRAME_HZ)):
            x = frame()
        # corners that were folded over, and where they should land
        corners = [
            int(np.argmin(np.linalg.norm(initial[:, :2] - c, axis=1)))
            for c in ([-length / 2, -width / 2], [-length / 2, width / 2])
        ]
        targets = np.array([[length / 2, -width / 2], [length / 2, width / 2]])
        corner_error = np.linalg.norm(x[corners, :2] - targets, axis=1)
        top = np.flatnonzero(initial[:, 0] < -0.02)  # the carried half, away from the fold line
        bottom = np.flatnonzero(initial[:, 0] > 0.02)
        near = np.argmin(np.linalg.norm(x[top, None, :2] - x[None, bottom, :2], axis=-1), axis=1)
        gap = x[top, 2] - x[bottom[near], 2]  # > 0: top layer above the bottom layer under it
        x_extent = float(np.ptp(x[:, 0]))
        traj_x, traj_z = np.asarray(trajectory)[..., 0], np.asarray(trajectory)[..., 2]
        report.update(
            corner_error_m=[float(e) for e in corner_error],
            folded_x_extent_ratio=x_extent / (length / 2),
            layer_gap_min_m=float(gap.min()),
            layer_gap_median_m=float(np.median(gap)),
            top_nodes_below_bottom_layer=int((gap < -0.0003).sum()),  # beyond one rest offset: crossed through
            top_layer_nodes=len(top),
            spring_back_m=float(np.abs(x[top, 0].mean() - out[top, 0].mean())),  # after the jaws are gone
            settle_motion_m=float(np.linalg.norm(x - out, axis=1).max()),
            max_pad_penetration_m=pad_penetration(),
            # nodes over the table only: the overhanging edge may hang below its top before it is picked up
            min_z_m=float(np.where(traj_x > -length / 2 + FOLD_OVERHANG, traj_z, np.inf).min()),
            final_max_z_m=float(x[:, 2].max()),
        )
        checks = {
            "mass": report["mass_matches"],
            "finite": finite,
            "corners_aligned_within_40mm": float(corner_error.max()) < 0.04,
            "folded_to_half_length": 0.9 < report["folded_x_extent_ratio"] < 1.15,
            "no_layer_crossing": report["top_nodes_below_bottom_layer"] <= 0.01 * len(top),
            "stays_folded": report["spring_back_m"] < 0.01,
            # the lower jaw wedges between the layers on set-down, inside the particles' 1 mm contact shell
            "pad_penetration_below_2mm": report["max_pad_penetration_m"] < 0.002,
            "edge_strain_within_limit": max_strain < a.max_strain,
            "no_table_penetration": report["min_z_m"] > -0.001,
        }
    report.update(
        max_edge_strain=max_strain,  # tensile, thread edges, any frame
        p99_edge_strain=p99_strain,
        final_max_edge_strain=stretch_track[-1],
        max_diagonal_strain=shear,
        finite=finite,
        checks=checks,
        passed=all(checks.values()),
        wall_s=time.perf_counter() - start,
        frames=len(trajectory),
    )
    poses = np.zeros((len(pad_actual), len(pads), 4), dtype=np.float32)
    for k, row in enumerate(pad_actual):
        if row:
            poses[k] = row
    np.savez_compressed(
        a.output / "nodes.npz",
        trajectory=np.asarray(trajectory),
        faces=faces,
        pads=poses[..., :3],  # (frames, pads, 3) box centres
        pad_pitch=poses[..., 3],  # (frames, pads) rotation about +Y (rad)
        pad_size=np.asarray(pad_size or (0, 0, 0), dtype=np.float32),
    )


if __name__ == "__main__":
    main()
