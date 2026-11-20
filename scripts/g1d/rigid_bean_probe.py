"""Feasibility probe for red beans as rigid bodies (one convex superellipsoid per bean) instead of PBD sphere particles.

No robot and no task: the tall measuring cup (measuring_cup/00001) is driven on a path over a floor, COUNT beans fall
into it through a temporary chute and settle, the chute goes, then the cup is lifted LIFT and rolled about its lip as
bean_tilt_probe.py does (35 deg/s to 70 deg, then RATE deg/s to 120 deg), so the flow compares with the PBD beans
calibrated against the real G1D-PourBeansEps380 pours (PBD, friction 0.3: 5% out at 88 deg, 25% at 98 deg).

Each bean is make_red_bean.py's superellipsoid (7 x 5 x 5 mm, 0.1 g) as a convex hull, scaled per bean by a random
size (uniform in 1 +- SIZE_JITTER, plus +-AXIS_JITTER per axis), mass by volume. Reports the wall time per physics step
(the cost against PBD), the settled heap height, beans lost through the floor or moving after settling (stability),
and the share poured against the tilt.

Usage (sim env, repo root, on a GPU):
    PYTHONPATH=. python scripts/g1d/rigid_bean_probe.py --count 3000 --out output/rigid_bean_probe/n3000
        [--rate 4] [--friction 0.3] [--iters 4] [--dt 0.0016667] [--axes 1 1 1] [--pack PACK.npz]
Writes OUT/metrics.json and OUT/poses.npz (per-bean scale, hull points, bean and cup poses settled and at SNAP_TILT).

--pack stops after settling and writes the beans settled in the cup as a granular pack (GranularObject): pose (x y z
qx qy qz qw, the tall cup's frame of measuring_cup_shape), scale, mass, the unit hull (hull, faces, volume) and the
packing (grain volume per cavity volume below the heap top), for the task's beans.

--shape capsule settles each bean as the capsule GranularObject builds from the pack for layouts with physics.shape
"capsule" (radius and length from the hull points and the bean's scale), so the settled grains do not overlap there;
--friction-combine, --torsional-patch, --max-depenetration, --velocity-iters, --linear-damping, --angular-damping and
--sleep-threshold set the beans' contact and body as the layout's physics does.
"""

import argparse

# PhysX's cooked hull of the bean lies ~0.35 mm inside its points: at rest offset 0 a bean sank 0.35 mm into the floor
# and flat beans stacked 4.0 mm apart, not 4.65; at 0.35 mm, 2.32 mm and 4.62 mm (12 beans in a tube, 4 or 32
# iterations alike)
REST_OFFSET = 0.00035

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--assets", default="Assets")
parser.add_argument("--count", type=int, default=3000)
parser.add_argument("--out", default="output/rigid_bean_probe")
parser.add_argument("--rate", type=float, default=4.0, help="slow tilt rate after 70 deg (deg/s)")
parser.add_argument("--friction", type=float, default=0.3)
parser.add_argument("--restitution", type=float, default=0.1)
parser.add_argument("--iters", type=int, default=4, help="solver position iterations per bean")
parser.add_argument("--rest-offset", type=float, default=REST_OFFSET, help="bean rest offset (m)")
parser.add_argument("--dt", type=float, default=1 / 600, help="physics dt; the g1d bean collection runs at 600 Hz")
parser.add_argument("--partitions", type=int, default=8, help="PhysX GPU constraint partitions (g1d_beans_rigid_sim)")
parser.add_argument("--size", type=float, default=1.0, help="bean size relative to the red bean (mass by volume)")
parser.add_argument("--axes", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="bean shape: per-axis scale")
parser.add_argument("--pack", help="write the settled beans as a granular pack here and stop")
parser.add_argument("--size-jitter", type=float, default=0.12)
parser.add_argument("--axis-jitter", type=float, default=0.04)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--shape", choices=("convex_hull", "capsule"), default="convex_hull", help="bean collider")
parser.add_argument("--friction-combine", help="PhysX friction combine mode of the beans (e.g. average)")
parser.add_argument("--torsional-patch", type=float, help="bean torsional patch radius (m)")
parser.add_argument("--max-depenetration", type=float, default=0.5, help="bean max depenetration velocity (m/s)")
parser.add_argument("--velocity-iters", type=int, default=1, help="solver velocity iterations per bean")
parser.add_argument("--linear-damping", type=float)
parser.add_argument("--angular-damping", type=float)
parser.add_argument("--sleep-threshold", type=float)
parser.add_argument("--settle-s", type=float, default=2.5, help="settling time with the chute (s)")
parser.add_argument("--clear-s", type=float, default=0.8, help="settling time after the chute goes (s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402

from env.scene_manager.objects.granular import _capsule_base_dims, _capsule_grain_dims  # noqa: E402
from scripts.objects import (
    make_red_bean as rb,  # noqa: E402
    measuring_cup_shape as cup_shape,  # noqa: E402
)

H = cup_shape.MEASURING[1][0]  # tall cup, 170 mm
SHIFT = cup_shape.origin_shift(H)  # asset point = cup-frame point + SHIFT
PIVOT = np.array([-cup_shape.CUP["A_LIP"], 0.0, H])  # lip on -X, cup frame; the mouth tips toward -X
AXIS = np.array([0.0, -1.0, 0.0])
LIFT, LIFT_S = 0.12, 1.5
FAST_TO, FAST_RATE, MAX_TILT, AFTER_S = 70.0, 35.0, 120.0, 1.5
SNAP_TILT = 100.0  # deg; poses.npz keeps the beans at settle and here
BEAN_MASS = rb.MASS
CONTACT_OFFSET = 0.001


def bean_hull():
    """Convex hull points of the bean (rb.surface, coarse; PhysX keeps <= 64 hull vertices)."""
    u = np.linspace(0, np.pi, 8)[1:-1]
    v = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    pts = np.vstack([rb.surface(u[:, None], v[None, :]).reshape(-1, 3), [[rb.A, 0, 0], [-rb.A, 0, 0]]])
    hull = ConvexHull(pts)
    return pts, hull.simplices, hull.volume


def cup_pose(theta, lift):
    """World pose (pos, quat xyzw) of the cup asset rolled theta deg about its lip and raised lift; the cup frame's
    origin (outside bottom centre) is at the world origin when upright on the floor."""
    rot = Rotation.from_rotvec(np.deg2rad(theta) * AXIS)
    origin = PIVOT + [0, 0, lift] - rot.apply(PIVOT)
    return origin - rot.apply(SHIFT), rot


def in_cup(p, pos, rot):
    return cup_shape.in_cavity(rot.inv().apply(p - pos) - SHIFT, H)


def spawn_positions(n, rng):
    """Beans lying along X in a grid inside the cup's narrowest (bottom) section, stacked as high as needed."""
    a = cup_shape.CUP["A0"] - cup_shape.CUP["T_WALL"]
    s = args.size * (1 + args.size_jitter + args.axis_jitter)
    dx, dy = 2 * rb.A * s * args.axes[0] + 0.0006, 2 * rb.B * s * args.axes[1] + 0.0006
    xs = np.arange(-a + dx / 2, a - dx / 2 + 1e-9, dx)
    ys = np.arange(-a + dy / 2, a - dy / 2 + 1e-9, dy)
    layer = np.array([(x, y) for x in xs for y in ys])
    margin = np.array([[sx * dx / 2, sy * dy / 2] for sx in (-1, 1) for sy in (-1, 1)])
    zb = cup_shape.CUP["T_BOTTOM"] + 0.004
    ok = np.array([cup_shape.in_cavity(np.c_[p + margin, np.full(4, zb)], H).all() for p in layer])
    layer = layer[ok]
    k = np.arange(n)
    z = zb + dy / 2 + (k // len(layer)) * dy
    xy = layer[k % len(layer)] + rng.uniform(-0.0003, 0.0003, (n, 2))
    return np.c_[xy, z], len(layer)


def physics_material(stage, path, friction, restitution, combine=None):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(restitution)
    if combine:
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr(combine)
    return mat


def bind_physics(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def build(stage, rng):
    floor = sim_utils.CuboidCfg(
        size=(2, 2, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38)),
    )
    floor.func("/World/floor", floor, translation=(0, 0, -0.01))
    light = sim_utils.DomeLightCfg(intensity=800)
    light.func("/World/light", light)

    bean_mat = physics_material(stage, "/World/Physics/bean", args.friction, args.restitution, args.friction_combine)
    cup_dir = Path(args.assets) / "Object/Real2Sim/Rigid/measuring_cup/00001"
    cup = stage.DefinePrim("/World/cup", "Xform")
    cup.GetReferences().AddReference(str((cup_dir / "object.usdz").resolve()))
    UsdGeom.XformCommonAPI(cup).SetTranslate(Gf.Vec3d(*cup_pose(0.0, 0.0)[0]))
    # dynamic without gravity, put on its path every step (GPU rigid body views cannot set kinematic targets)
    UsdPhysics.RigidBodyAPI.Apply(cup).CreateKinematicEnabledAttr(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(cup).CreateDisableGravityAttr(True)
    for p in Usd.PrimRange(cup):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            c = PhysxSchema.PhysxCollisionAPI.Apply(p)
            c.CreateContactOffsetAttr(0.002)
            c.CreateRestOffsetAttr(0.0)
    bind_physics(cup, bean_mat)

    # chute: four walls around the lip, up to 0.5 m, so a column of beans taller than the cup falls into it
    chute = UsdGeom.Xform.Define(stage, "/World/chute")
    chute.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(chute.GetPrim()).CreateKinematicEnabledAttr(True)
    half, t, z0, z1 = cup_shape.CUP["A_LIP"] + cup_shape.TAB_OUT + 0.002, 0.004, H - 0.02, 0.5
    for i, (cx, cy, sx, sy) in enumerate(
        [(half + t / 2, 0, t, 2 * half + 2 * t), (-half - t / 2, 0, t, 2 * half + 2 * t),
         (0, half + t / 2, 2 * half, t), (0, -half - t / 2, 2 * half, t)]
    ):  # fmt: skip
        wall = UsdGeom.Cube.Define(stage, f"/World/chute/wall_{i}")
        wall.CreateSizeAttr(1.0)
        wall.AddTranslateOp().Set(Gf.Vec3d(cx, cy, (z0 + z1) / 2))
        wall.AddScaleOp().Set(Gf.Vec3f(sx, sy, z1 - z0))
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        UsdGeom.Imageable(wall).MakeInvisible()

    pts, faces, volume = bean_hull()
    spawn, per_layer = spawn_positions(args.count, rng)
    scale = (
        args.size
        * np.asarray(args.axes)
        * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter, (args.count, 1))
        * rng.uniform(1 - args.axis_jitter, 1 + args.axis_jitter, (args.count, 3))
    )
    yaw = Rotation.from_euler("z", rng.uniform(-8, 8, args.count), degrees=True)
    roll = Rotation.from_euler("x", rng.uniform(0, 360, args.count), degrees=True)
    quat = (yaw * roll).as_quat()  # xyzw
    color = np.array(rb.COAT) * 3.0
    UsdGeom.Scope.Define(stage, "/World/beans")
    vertices = Vt.Vec3fArray.FromNumpy(pts.astype(np.float32))
    counts, indices = Vt.IntArray([3] * len(faces)), Vt.IntArray(faces.reshape(-1).tolist())
    capsule = args.shape == "capsule"
    if capsule:  # the capsule GranularObject builds from this pack: dims from the hull points and the bean's scale
        radii, heights = _capsule_grain_dims(*_capsule_base_dims({"hull": pts}, {}), scale)
    for i in range(args.count):
        path = f"/World/beans/bean_{i}"
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*spawn[i]))
        x.AddOrientOp().Set(Gf.Quatf(float(quat[i, 3]), *map(float, quat[i, :3])))
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, np.ones(3) if capsule else scale[i])))
        prim = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(BEAN_MASS * np.prod(scale[i])))
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSolverPositionIterationCountAttr(args.iters)
        body.CreateSolverVelocityIterationCountAttr(args.velocity_iters)
        body.CreateMaxDepenetrationVelocityAttr(args.max_depenetration)
        for value, create in ((args.linear_damping, body.CreateLinearDampingAttr),
                              (args.angular_damping, body.CreateAngularDampingAttr),
                              (args.sleep_threshold, body.CreateSleepThresholdAttr)):  # fmt: skip
            if value is not None:
                create(value)
        if capsule:
            geom = UsdGeom.Capsule.Define(stage, f"{path}/capsule")
            geom.CreateAxisAttr("X")
            geom.CreateRadiusAttr(float(radii[i]))
            geom.CreateHeightAttr(float(heights[i]))
        else:
            geom = UsdGeom.Mesh.Define(stage, f"{path}/hull")
            geom.CreatePointsAttr(vertices)
            geom.CreateFaceVertexCountsAttr(counts)
            geom.CreateFaceVertexIndicesAttr(indices)
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
        geom.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, color))]))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        c = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        c.CreateContactOffsetAttr(CONTACT_OFFSET)
        c.CreateRestOffsetAttr(args.rest_offset)
        if args.torsional_patch is not None:
            c.CreateTorsionalPatchRadiusAttr(args.torsional_patch)
        bind_physics(geom.GetPrim(), bean_mat)
    info = dict(per_layer=per_layer, column_top_m=round(float(spawn[:, 2].max()), 3), hull_vertices=len(pts),
                bean_volume_mm3=round(volume * 1e9, 1))  # fmt: skip
    return info, scale


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=args.dt,
            device=args.device,
            physx=sim_utils.PhysxCfg(
                gpu_max_rigid_contact_count=2**24,
                gpu_max_rigid_patch_count=2**22,
                gpu_found_lost_pairs_capacity=2**23,
                gpu_total_aggregate_pairs_capacity=2**23,
                gpu_max_num_partitions=args.partitions,
            ),
        )
    )
    stage = get_current_stage()
    t0 = time.time()
    info, scale = build(stage, rng)
    sim.reset()
    build_s = time.time() - t0
    view = sim.physics_sim_view
    beans = view.create_rigid_body_view("/World/beans/bean_*")
    cup = view.create_rigid_body_view("/World/cup")
    chute = view.create_rigid_body_view("/World/chute")
    assert beans.count == args.count, f"bean view has {beans.count} bodies"
    idx = torch.tensor([0], dtype=torch.int32, device=args.device)

    def tensor(values):
        return torch.tensor([list(values)], dtype=torch.float32, device=args.device)

    def drive(now, nxt):
        """Cup at pose `now`, moving so that it reaches `nxt` one step later."""
        (p0, r0), (p1, r1) = now, nxt
        cup.set_transforms(tensor([*p0, *r0.as_quat()]), idx)
        cup.set_velocities(tensor([*(p1 - p0) / args.dt, *(r1 * r0.inv()).as_rotvec() / args.dt]), idx)

    def positions():
        return beans.get_transforms()[:, :3].cpu().numpy()

    def speeds():
        return np.linalg.norm(beans.get_velocities()[:, :3].cpu().numpy(), axis=1)

    poses = {}

    def snapshot(name, cup_at):
        """Bean poses (x y z qx qy qz qw) and the cup pose."""
        poses[name] = beans.get_transforms().cpu().numpy()
        poses[f"{name}_cup"] = np.r_[cup_at[0], cup_at[1].as_quat()]

    timing = {}

    def run(phase, steps, target):
        t = time.time()
        for k in range(steps):
            target(k)
            sim.step(render=False)
        torch.cuda.synchronize()
        timing[phase] = round((time.time() - t) / max(steps, 1) * 1000, 3)

    upright = cup_pose(0.0, 0.0)
    run("settle_ms_per_step", round(args.settle_s / args.dt), lambda k: drive(upright, upright))
    chute.set_transforms(tensor([0, 0, 2.0, 0, 0, 0, 1]), idx)
    run("clear_ms_per_step", round(args.clear_s / args.dt), lambda k: drive(upright, upright))
    p = positions()
    v = speeds()
    inside = in_cup(p, *upright)
    local = upright[1].inv().apply(p - upright[0]) - SHIFT
    settled = dict(
        in_cup=int(inside.sum()),
        lost_below_floor=int((p[:, 2] < -0.02).sum()),
        nan=int((~np.isfinite(p)).any(1).sum()),
        heap_top_m=round(float(np.percentile(local[inside, 2], 95)), 4) if inside.any() else 0.0,
        moving_over_5cm_s=int((v > 0.05).sum()),
        moving_in_cup_over_5cm_s=int((v[inside] > 0.05).sum()),
        max_speed_m_s=round(float(v.max()), 3),
    )
    snapshot("settle", upright)
    n0 = max(int(inside.sum()), 1)
    if args.pack:
        return write_pack(poses["settle"][inside], scale[inside], settled)

    def poured(theta, lift):
        return 1.0 - float(in_cup(positions(), *cup_pose(theta, lift)).sum()) / n0

    n = round(LIFT_S / args.dt)

    def lifted(k):
        return cup_pose(0.0, LIFT * (1 - np.cos(np.pi * min(k, n) / n)) / 2)

    run("lift_ms_per_step", n, lambda k: drive(lifted(k), lifted(k + 1)))
    curve, theta, steps, t = [], 0.0, 0, time.time()
    sub = max(1, round(1 / 30 / args.dt))
    while theta < MAX_TILT:
        nxt = min(theta + (FAST_RATE if theta < FAST_TO else args.rate) * args.dt, MAX_TILT)
        drive(cup_pose(theta, LIFT), cup_pose(nxt, LIFT))
        theta = nxt
        sim.step(render=False)
        steps += 1
        if steps % sub == 0:
            curve.append([round(steps * args.dt, 3), round(theta, 2), round(poured(theta, LIFT), 4)])
        if "pour" not in poses and theta >= SNAP_TILT:
            snapshot("pour", cup_pose(theta, LIFT))
    torch.cuda.synchronize()
    timing["pour_ms_per_step"] = round((time.time() - t) / steps * 1000, 3)
    tipped = cup_pose(MAX_TILT, LIFT)
    run("after_ms_per_step", round(AFTER_S / args.dt), lambda k: drive(tipped, tipped))
    rec = np.array(curve)

    def angle_at(share):
        hit = np.flatnonzero(rec[:, 2] >= share)
        return round(float(rec[hit[0], 1]), 1) if len(hit) else None

    sim_s = args.settle_s + args.clear_s + LIFT_S + steps * args.dt + AFTER_S
    total_ms = sum(timing[k] * s for k, s in [
        ("settle_ms_per_step", args.settle_s), ("clear_ms_per_step", args.clear_s), ("lift_ms_per_step", LIFT_S),
        ("pour_ms_per_step", steps * args.dt), ("after_ms_per_step", AFTER_S)]) / args.dt  # fmt: skip
    result = dict(
        count=args.count,
        dt=args.dt,
        iters=args.iters,
        partitions=args.partitions,
        rest_offset=args.rest_offset,
        shape=args.shape,
        friction=args.friction,
        rate_deg_s=args.rate,
        size=args.size,
        size_jitter=args.size_jitter,
        axis_jitter=args.axis_jitter,
        build_s=round(build_s, 1),
        **info,
        timing=timing,
        realtime_factor=round(sim_s * 1000 / total_ms, 3),
        settled=settled,
        onset_deg={str(s): angle_at(s) for s in (0.02, 0.05, 0.25, 0.5, 0.75, 0.9)},
        final=round(poured(MAX_TILT, LIFT), 4),
        lost_below_floor_end=int((positions()[:, 2] < -0.02).sum()),
        curve=curve,
    )
    print("[rigid_bean_probe]", json.dumps({k: v for k, v in result.items() if k != "curve"}), flush=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out / "poses.npz", scale=scale, hull=bean_hull()[0], **poses)


def write_pack(pose, scale, settled):
    pts, faces, volume = bean_hull()
    top = settled["heap_top_m"]
    box = np.random.default_rng(0).uniform([-0.05, -0.05, 0], [0.05, 0.05, top], (1_000_000, 3))
    cavity = cup_shape.in_cavity(box, H).mean() * 0.1 * 0.1 * top
    below = pose[:, 2] < top
    packing = float(volume * np.prod(scale[below], axis=1).sum() / cavity)
    mass = BEAN_MASS * np.prod(scale, axis=1)
    Path(args.pack).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.pack, pose=pose, scale=scale, mass=mass, hull=pts, faces=faces, volume=volume,
                        packing=packing, axes=np.asarray(args.axes), size=args.size, shape=args.shape)  # fmt: skip
    print("[rigid_bean_probe] pack", json.dumps(dict(pack=args.pack, beans=len(pose), packing=round(packing, 3),
                                                     **settled)), flush=True)  # fmt: skip


try:
    main()
    failed = False
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
