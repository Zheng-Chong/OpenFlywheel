"""Drop the g1d tasks' red beans into cups and count how many stay inside: can the cup hold beans in simulation?

Each cup stands on the floor as a dynamic rigid body (as in the pour tasks), alone for STAND s first: its tilt then
says whether it stands upright. It is then held still where it stood (kinematic, as the target cup in the hand).
Its cavity is found from its collider, so no per-cup shape is needed: vertical rays over the cup give a height map,
and a priority flood from outside the cup gives each cell the level at which water poured there would spill; cells
lying CAVITY_MIN below their spill level are the cavity. A collider that closes the mouth (one convex hull, a filled
decomposition, a modelled drink) has none. A column of beans (FluidObject PBD solid particles with RED_BEAN_PHYSICS,
as in g1d_pour_beans_cup_to_cup) is then dropped onto the centre of the cavity's opening (or the cup's centre); a
bean is held when it settles over a cavity cell below the spill level, or under an inward lip next to it.

Cups are object directories under Assets/Object (`Real2Sim/Rigid/plastic_cup/00001`), loaded from object.usdz /
object.usd or, for the raw NVIDIA SimReady packages, the single top-level .usd(a); their rigid body moves from the
geometry child to the cup root, as RigidObject expects. Writes under --out:
    metrics.json  per cup: tilt standing alone, cavity depth, opening and volume, beans dropped, share held, share
                  escaped to the floor, pass (holds the beans), stands (upright on its own)
    cavity/<n>.npz  the height map, spill levels and cavity cells, relative to the cup root as it stood, and the
                  root's height above the floor (root_z)
    <n>.png       each cup from above after the beans settled; sheet.png all of them

PYTHONPATH=. python scripts/objects/check_cup_holds_beans.py --out output/cup_beans Real2Sim/Rigid/plastic_cup/00001 ...
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("cups", nargs="+", help="object directories relative to Assets/Object")
parser.add_argument("--assets", default="Assets")
parser.add_argument("--out", default="output/cup_beans")
parser.add_argument("--beans", type=int, default=300, help="beans dropped into each cup")
parser.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.device != "cpu":
    parser.error("Use --device cpu for particle USD readback; PBD physics still runs on the GPU.")
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import heapq  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402

import carb  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from omni.physx import get_physx_interface, get_physx_scene_query_interface  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402

from env.scene_manager.objects.asset_defaults import with_asset_defaults  # noqa: E402
from env.scene_manager.objects.fluid import FluidObject  # noqa: E402
from env.scene_manager.objects.rigid import RigidObject  # noqa: E402
from scripts.g1d.build_asset import RED_BEAN_MDL, RED_BEAN_PHYSICS  # noqa: E402

DT = 0.004
GPU_OVERRIDE = get_physx_interface().get_overwrite_gpu_setting()
PITCH = 0.4  # m between cups on the floor grid
COLS = 12
BEAN = 2 * RED_BEAN_PHYSICS["particle_system"]["solid_rest_offset"]  # particle diameter (hex packed)
CELL = 0.0025  # m, height map resolution
CAVITY_MIN = 0.01  # m a cell lies below its spill level to count as cavity
DROP = 0.01  # column bottom above the spill level
STAND = 0.5  # s the cups stand alone first, the beans in a heap ASIDE (m) of each
ASIDE = np.array([0.15, 0.15])
LIP = 0.01  # m an inward lip may overhang the cavity seen from above
SLACK = 0.003  # m: beans this far over the spill level still count as held (a heap settles a little above it)
HELD_OK, TILT_OK = 0.95, 5.0  # pass: share of beans held; stands: deg from upright standing alone
FALLBACK_MASS = 0.2  # kg when the asset metadata has none (SimReady)


def object_usd(d):
    for name in ("object.usdz", "object.usd"):
        if (d / name).is_file():
            return d / name
    raw = sorted(p for p in d.iterdir() if p.suffix in (".usd", ".usda"))
    assert len(raw) == 1, f"{d}: no object.usd(z) and {len(raw)} top-level USD files"
    return raw[0]


def mesh_bounds(prim):
    cache = UsdGeom.XformCache()
    pts = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputePurpose() != UsdGeom.Tokens.guide:
            local = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], dtype=float)
            if len(local):
                tf = np.asarray(cache.GetLocalToWorldTransform(p))
                pts.append(local @ tf[:3, :3] + tf[3, :3])
    pts = np.concatenate(pts)
    return pts.min(0), pts.max(0)


def move_rigid_body_to_root(prim):
    """SimReady packages put the rigid body on the geometry child; RigidObject applies it on the root."""
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p != prim and p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.IsInstanceProxy():
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            p.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)


def spill_levels(height):
    """Priority flood: the lowest level at which water in each cell runs off the map (NaN cells are outside)."""
    h = np.where(np.isnan(height), -np.inf, height)
    level = np.full(h.shape, np.inf)
    border = np.isnan(height)
    border[[0, -1], :] = border[:, [0, -1]] = True
    heap = [(max(h[i, j], -1.0), i, j) for i, j in zip(*np.nonzero(border))]
    for lv, i, j in heap:
        level[i, j] = lv
    heapq.heapify(heap)
    while heap:
        lv, i, j = heapq.heappop(heap)
        if lv > level[i, j]:
            continue
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < h.shape[0] and 0 <= b < h.shape[1]:
                nl = max(lv, h[a, b])
                if nl < level[a, b]:
                    level[a, b] = nl
                    heapq.heappush(heap, (nl, a, b))
    return level


def cavity_map(cup_path, lo, hi):
    """Height map of the cup's colliders on a CELL grid over its footprint (world frame), and the cavity in it."""
    xs = np.arange(lo[0] - 2 * CELL, hi[0] + 2 * CELL, CELL)
    ys = np.arange(lo[1] - 2 * CELL, hi[1] + 2 * CELL, CELL)
    query = get_physx_scene_query_interface()
    top = hi[2] + 0.05
    height = np.full((len(xs), len(ys)), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hit = query.raycast_closest((float(x), float(y), float(top)), (0.0, 0.0, -1.0), 1.0)
            if hit["hit"] and hit["rigidBody"] == cup_path:
                height[i, j] = hit["position"][2]
    # rays slipping through the seams between convex pieces: outside only when connected to the map's edge
    missed, n = ndimage.label(np.isnan(height))
    edge = set(np.unique(np.r_[missed[[0, -1], :].ravel(), missed[:, [0, -1]].ravel()])) - {0}
    seams = (missed > 0) & ~np.isin(missed, list(edge))
    if seams.any() and not np.isnan(height).all():
        nearest = ndimage.distance_transform_edt(np.isnan(height), return_distances=False, return_indices=True)
        height[seams] = height[tuple(nearest)][seams]
    level = spill_levels(height)
    cavity = ~np.isnan(height) & (level - height >= CAVITY_MIN)
    return xs, ys, height, level, cavity


def column(count, radius):
    """`count` bean centres on a square lattice (1.05 bean apart) inside a vertical cylinder from z=0 up."""
    step = 1.05 * BEAN
    n = int(radius // step)
    layer = np.array([(x, y) for x in range(-n, n + 1) for y in range(-n, n + 1) if np.hypot(x, y) * step <= radius])
    layers = -(-count // len(layer))
    pts = np.array([(x * step, y * step, z * step) for z in range(layers) for x, y in layer])
    return pts[:count]


def column_usd(radius, height):
    """Convex point cloud of the bean column, a 16-gon prism from z=0 up, centred on the origin."""
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.c_[radius * np.cos(a), radius * np.sin(a)]
    points = np.array([(x, y, z) for z in (0.0, height) for x, y in ring], dtype=np.float32)
    path = Path(tempfile.mkdtemp()) / "column.usda"
    st = Usd.Stage.CreateNew(str(path))
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/root").GetPrim())
    UsdGeom.Mesh.Define(st, "/root/model").CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    st.Save()
    return str(path)


def load_cup(stage, k, rel, assets):
    d = assets / "Object" / rel
    usd = object_usd(d)
    meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").is_file() else {}
    cat = d.parent.name
    path = f"/World/envs/env_{k}/Rigid/{cat}/{cat}_{int(d.name)}_0"
    prim = add_reference_to_stage(usd_path=str(usd), prim_path=path)
    move_rigid_body_to_root(prim)
    lo, hi = mesh_bounds(prim)  # the reference sits at the origin until RigidObject places it
    grid = np.array([PITCH * (k % COLS), PITCH * (k // COLS)])
    pos = np.r_[grid - (lo[:2] + hi[:2]) / 2, -lo[2] + 0.001]
    physics = dict(meta.get("physics", {}))
    physics.setdefault("mass", FALLBACK_MASS)
    inst = with_asset_defaults(str(usd), {"physics": {k_: v for k_, v in physics.items() if k_ != "size"}})
    obj = RigidObject(
        prim_path=path,
        usd_path=None,
        inst_config=inst,
        env_origin=[0, 0, 0],
        default_pos=tuple(pos),
        default_ori=(1, 0, 0, 0),
        scale=(1, 1, 1),
    )
    return dict(name=rel, usd=usd.name, path=path, obj=obj, pos=pos, lo=lo + pos, hi=hi + pos)


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets).resolve()
    fluid_dir = assets / "Object/RoboDojo/Fluid/wuliangye/00000"
    fluid_meta = json.loads((fluid_dir / "metadata.json").read_text())
    bean_cfg = {
        "physics": dict(fluid_meta["physics"], **RED_BEAN_PHYSICS),
        "visual": {"visual_mdl_path": RED_BEAN_MDL.replace("$Robodojo_ASSETS", str(assets)), "scale": [1, 1, 1]},
    }

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            device=args.device,
            use_fabric=False,
            enable_scene_query_support=True,  # the cavity raycasts
            render=sim_utils.RenderCfg(),
        )
    )
    stage = get_current_stage()
    PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(sim.cfg.physics_prim_path)).CreateEnableGPUDynamicsAttr().Set(True)
    get_physx_interface().overwrite_gpu_setting(1)
    rows = len(args.cups) // COLS + 1
    floor = sim_utils.CuboidCfg(
        size=(PITCH * COLS + 1, PITCH * rows + 1, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38)),
    )
    floor.func("/World/floor", floor, translation=(PITCH * (COLS - 1) / 2, PITCH * (rows - 1) / 2, -0.01))
    sim_utils.DomeLightCfg(intensity=300).func("/World/light", sim_utils.DomeLightCfg(intensity=300))
    sun = sim_utils.DistantLightCfg(intensity=2500, angle=5)
    sun.func("/World/sun", sun, orientation=(0.9397, 0.2418, 0.2418, 0))

    cups, failed_load = [], {}
    for k, rel in enumerate(args.cups):
        try:
            c = load_cup(stage, k, rel, assets)
        except Exception as e:  # report and go on: one broken package must not hide the others
            failed_load[rel] = f"{type(e).__name__}: {e}"
            continue
        # sized for the narrowest mouths; moved onto the cavity once the colliders can be queried
        radius = max(BEAN, 0.4 * (c["hi"] - c["lo"])[:2].min() / 2)
        height = args.beans * BEAN**3 / np.sqrt(2) / (np.pi * radius**2)
        c["fluid"] = FluidObject(
            prim_path=f"/World/envs/env_{k}/Fluid/red_bean/red_bean_0_0",
            usd_path=column_usd(radius, height),
            inst_config=json.loads(json.dumps(bean_cfg)),
            env_origin=torch.zeros(3),
            default_pos=(*((c["lo"][:2] + c["hi"][:2]) / 2 + ASIDE), 0.005),
            default_ori=(1, 0, 0, 0),
        )
        c["fluid"].particle_system.CreateSimulationOwnerRel().SetTargets([sim.cfg.physics_prim_path])
        c["count"] = len(c["fluid"].init_particle_positions)
        cups.append(c)

    cam = Camera(prim_path="/World/cam", resolution=(480, 360))
    UsdGeom.Camera(stage.GetPrimAtPath("/World/cam")).GetClippingRangeAttr().Set((0.01, 100.0))
    sim.reset()
    carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
    assert sim.get_physics_context().is_gpu_dynamics_enabled(), "PBD requires GPU dynamics"
    for c in cups:
        c["obj"].initialize()
        c["fluid"].initialize()
    cam.initialize()

    def pose(c):
        pos, quat = (np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float) for v in c["obj"].get_world_pose())
        return pos, Rotation.from_quat(quat[[1, 2, 3, 0]])

    def tilt_deg(rot):
        return float(np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1))))

    for _ in range(round(STAND / DT)):  # the cups stand on their own; the beans lie beside them
        sim.step(render=False)
    (out / "cavity").mkdir(exist_ok=True)
    for n_cup, c in enumerate(cups):
        c["stand_pos"], c["stand_rot"] = pose(c)
        c["stand_tilt"] = tilt_deg(c["stand_rot"])
        # held still as it stands, as the target cup in the hand: a 20 g cup tips over under 30 g of falling beans
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(c["path"])).CreateKinematicEnabledAttr().Set(True)
        lo, hi = c["lo"] + c["stand_pos"] - c["pos"], c["hi"] + c["stand_pos"] - c["pos"]
        xs, ys, height, level, cavity = cavity_map(c["path"], lo, hi)
        labels, n = ndimage.label(cavity)
        if n:  # the largest basin is the cup's; drop onto the centre of the largest circle inside its opening
            cavity = labels == 1 + np.argmax(ndimage.sum(cavity, labels, range(1, n + 1)))
            inside = ndimage.distance_transform_edt(np.pad(cavity, 1))[1:-1, 1:-1]
            i, j = np.unravel_index(np.argmax(inside), inside.shape)
            c["inscribed"] = float(inside[i, j] * CELL)
            c["depth"] = float(np.max(np.where(cavity, level - height, 0)))
            c["rim"] = float(level[i, j])
            c["volume_ml"] = float(np.where(cavity, level - height, 0).sum() * CELL**2 * 1e6)
            target = np.r_[xs[i], ys[j], c["rim"] + DROP]
        else:
            c["inscribed"] = c["depth"] = c["volume_ml"] = 0.0
            c["rim"] = float(np.nanmax(height))
            target = np.r_[(lo[:2] + hi[:2]) / 2, c["rim"] + DROP]
        c.update(xs=xs, ys=ys, height=height, level=level, cavity=cavity, floor_z=lo[2])
        np.savez_compressed(
            out / "cavity" / f"{n_cup:03d}.npz",  # relative to the cup root as it stood (upright unless stand_tilt)
            xs=xs - c["stand_pos"][0],
            ys=ys - c["stand_pos"][1],
            height=height - c["stand_pos"][2],
            level=level - c["stand_pos"][2],
            cavity=cavity,
            root_z=c["stand_pos"][2],  # the root above the floor it stands on
        )
        origin = np.r_[(c["lo"][:2] + c["hi"][:2]) / 2 + ASIDE, 0.005]  # the fluid prim, set at creation
        radius = c["inscribed"] / 2 if c["inscribed"] else 0.2 * (c["hi"] - c["lo"])[:2].min()
        c["fluid"].set_particle_positions(column(c["count"], radius) + target - origin)

    for _ in range(round(args.seconds / DT)):
        sim.step(render=False)
    get_physx_interface().update_transformations(False, True)

    report = {}
    xf = UsdGeom.XformCache()
    for n, c in enumerate(cups):
        pos, rot = pose(c)
        local = c["fluid"].get_particle_positions()[0]
        tf = np.asarray(xf.GetLocalToWorldTransform(c["fluid"].point_instancer.GetPrim()))
        world = local @ tf[:3, :3] + tf[3, :3]
        # where the bean would be had the cup stayed as it stood when its cavity was mapped
        p = c["stand_rot"].apply(rot.inv().apply(world - pos)) + c["stand_pos"]
        i = np.clip(np.rint((p[:, 0] - c["xs"][0]) / CELL).astype(int), 0, len(c["xs"]) - 1)
        j = np.clip(np.rint((p[:, 1] - c["ys"][0]) / CELL).astype(int), 0, len(c["ys"]) - 1)
        over_cavity = c["cavity"][i, j] & (p[:, 2] <= c["level"][i, j] + SLACK)
        # under an inward lip the rays saw the lip: the cup above the bean, the bean below the rim, next to the cavity
        # and no lower than its floor (a bean on the ground under the handle is not)
        lip_zone = ndimage.binary_dilation(c["cavity"], iterations=round(LIP / CELL))[i, j]
        floor = np.nanmin(np.where(c["cavity"], c["height"], np.nan)) if c["cavity"].any() else np.inf
        under_lip = lip_zone & (c["height"][i, j] > p[:, 2] + BEAN / 2) & (p[:, 2] <= c["rim"])
        held = over_cavity | under_lip & (p[:, 2] >= floor + BEAN / 2 - SLACK)
        tilt = tilt_deg(rot)
        r = dict(
            usd=c["usd"],
            size_m=(c["hi"] - c["lo"]).round(4).tolist(),
            stand_tilt_deg=round(c["stand_tilt"], 1),
            cavity_depth_mm=round(c["depth"] * 1000, 1),
            rim_mm=round((c["rim"] - c["floor_z"]) * 1000, 1),
            opening_radius_mm=round(c["inscribed"] * 1000, 1),  # largest circle inside the opening
            volume_ml=round(c["volume_ml"]),
            beans=c["count"],
            held=round(float(held.mean()), 4),
            escaped=round(float((~held & (world[:, 2] < 0.012)).mean()), 4),
            tilt_deg=round(tilt, 1),
        )
        r["stands"] = bool(c["stand_tilt"] <= TILT_OK)
        r["pass"] = bool(r["cavity_depth_mm"] > 0 and r["held"] >= HELD_OK)
        report[c["name"]] = r
        print(f"[cup_beans] {c['name']}: {json.dumps(r)}", flush=True)

        mouth = np.r_[(c["lo"][:2] + c["hi"][:2]) / 2, c["hi"][2]]
        eye = mouth + [0.0, -0.22, 0.32]
        set_camera_view(tuple(eye), tuple(mouth - [0, 0, (c["hi"] - c["lo"])[2] / 2]), camera_prim_path="/World/cam")
        for _ in range(20):
            sim.render()
        Image.fromarray(cam.get_rgba()[:, :, :3]).save(out / f"{n:03d}.png")

    names = [c["name"] for c in cups]
    sheet = Image.new("RGB", (480 * 8, 400 * ((len(names) + 7) // 8)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, name in enumerate(names):
        x, y = 480 * (n % 8), 400 * (n // 8)
        sheet.paste(Image.open(out / f"{n:03d}.png"), (x, y))
        r = report[name]
        draw.text((x + 6, y + 364), f"{'OK ' if r['pass'] else 'BAD'} {name.split('/', 2)[-1]}", fill="black")
        line = f"depth {r['cavity_depth_mm']} mm held {r['held']:.0%} escaped {r['escaped']:.0%} tilt {r['tilt_deg']}"
        draw.text((x + 6, y + 380), line, fill="black")
    sheet.save(out / "sheet.png")

    summary = dict(
        cups=report,
        failed_load=failed_load,
        passed=sum(r["pass"] for r in report.values()),
        stand=sum(r["stands"] for r in report.values()),
        seconds=args.seconds,
        bean_physics=RED_BEAN_PHYSICS,
        criteria=dict(cavity_min_mm=CAVITY_MIN * 1000, held=HELD_OK, stand_tilt_deg=TILT_OK),
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[cup_beans] passed {summary['passed']}/{len(report)}, failed to load {len(failed_load)}", flush=True)
    return bool(failed_load) or not all(r["pass"] for r in report.values())


try:
    failed = main()
except Exception:
    traceback.print_exc()
    failed = True
finally:
    get_physx_interface().overwrite_gpu_setting(GPU_OVERRIDE)
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
