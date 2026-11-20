"""Load one scene of an open-source scene library (scripts/scenes/open_scenes.py) in Isaac Sim, measure it and render
its platform previews with RTX.

The scene USD is referenced alone (no robot) under /World/Scene. Ceilings (prims whose name matches --hide) are hidden
and a dome and a sun light are added, so every view sees in from above: overview.jpg (oblique, whole scene), top.jpg
(plan view) and room_<n>.jpg (eye height, from a corner of the largest floors). Writes <out>/stats.json: what
loaded, unresolved asset paths, bounding box, floors, mesh / rigid body / joint counts.

Usage (GPU host with the RoboDojo Isaac env; one scene per process, the Kit app does not close cleanly):
    OMNI_KIT_ACCEPT_EULA=YES python scripts/scenes/render_library_scene.py <scene.usd> <out dir>
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure and render one open-source library scene.")
parser.add_argument("usd", help="the scene's entry USD")
parser.add_argument("out", help="output directory: the renders and stats.json")
parser.add_argument("--hide", default=r"(?i)ceil", help="regex on prim names hidden for the renders")
parser.add_argument("--floor", default=r"(?i)floor|/ground/", help="regex on mesh prim paths that are floors")
parser.add_argument("--rooms", type=int, default=2, help="eye-height views, one per largest floor")
parser.add_argument("--gpu", type=int, help="physical GPU index; default: the one card Slurm (or the env) gives")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
# one GPU only: with multi-GPU rendering, or a CUDA context on every card, Kit allocates on the shared hosts'
# busy cards and runs out of memory. Vulkan ignores CUDA_VISIBLE_DEVICES, so activeGpu stays the physical index.
# Under Slurm the job's cgroup shows Vulkan only its own card: index 0.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.device = "cuda:0"
active = args.gpu or 0
args.kit_args = " ".join(
    [getattr(args, "kit_args", "") or "", f"--/renderer/activeGpu={active}", "--/renderer/multiGpu/enabled=false"]
).strip()
app = AppLauncher(args).app

from collections import Counter  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import carb  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils  # noqa: E402

OVERVIEW = np.array([0.55, -0.75, 0.9])  # from the south-east, above
EYE_Z = 1.5  # m above a floor, eye-height views
INSET = 0.35  # m from the floor's corner
MAX_MESH = 100.0  # m: larger meshes (ground planes, sky shells) are left out of the framing
CEILING_Z = 2.0  # m: horizontal slabs whose underside is this far above the lowest floor are hidden
STATS = {}
SLABS = []  # (box low corner, mesh prim) of every large thin horizontal mesh


def look_at(eye, target):
    """(w, x, y, z) of a camera at `eye` looking at `target`, Isaac "world" camera axes (+X forward, +Z up)."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    m = np.stack([forward, left, np.cross(forward, left)], axis=1)
    w = np.sqrt(max(1e-12, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])


def world_box(cache, prim):
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return (None, None) if r.IsEmpty() else (np.array(r.GetMin()), np.array(r.GetMax()))


def measure(stage, root):
    """Counts and boxes of the scene under `root`; floors are (lo, hi, path) of floor meshes, largest first."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    types, apis, los, his, floors = Counter(), Counter(), [], [], []
    for prim in Usd.PrimRange(root):
        types[prim.GetTypeName()] += 1
        for api in prim.GetAppliedSchemas():
            apis[api] += 1
        if not prim.IsA(UsdGeom.Mesh):
            continue
        lo, hi = world_box(cache, prim)
        if lo is None or np.linalg.norm(hi - lo) > MAX_MESH:
            continue
        los.append(lo)
        his.append(hi)
        size = hi - lo
        if size[2] < 0.3 and size[0] * size[1] > 1.0:  # a horizontal slab: floor, ceiling, roof
            SLABS.append((lo, prim))
            if re.search(args.floor, str(prim.GetPath())):
                floors.append((lo, hi, str(prim.GetPath())))
    floors.sort(key=lambda f: -(f[1][0] - f[0][0]) * (f[1][1] - f[0][1]))
    joints = {t: n for t, n in types.items() if t.endswith("Joint")}
    lo, hi = (np.min(los, axis=0), np.max(his, axis=0)) if los else (np.zeros(3), np.ones(3))
    STATS.update(
        meshes=types["Mesh"],
        prims=sum(types.values()),
        rigid_bodies=apis["PhysicsRigidBodyAPI"],
        articulation_roots=apis["PhysicsArticulationRootAPI"],
        joints=joints,
        movable_joints=sum(n for t, n in joints.items() if "Revolute" in t or "Prismatic" in t),
        bbox=[lo.round(3).tolist(), hi.round(3).tolist()],
        floors=[{"path": p, "lo": a.round(3).tolist(), "hi": b.round(3).tolist()} for a, b, p in floors],
        floor_area_m2=round(sum((b[0] - a[0]) * (b[1] - a[1]) for a, b, _ in floors), 1),
    )
    return lo, hi, floors


def hide(stage, root, floor_z):
    """Hide the ceilings: prims named like one, and slabs CEILING_Z above the floor (GRScenes also files ceiling
    panels and roofs under Base/other)."""
    hidden = []
    for prim in Usd.PrimRange(root):
        if re.search(args.hide, prim.GetName()) and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
    for lo, prim in SLABS:
        if lo[2] > floor_z + CEILING_Z:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
    STATS["hidden"] = len(hidden)


def lights(stage, has_own):
    dome = UsdLux.DomeLight.Define(stage, "/World/PreviewDome")
    dome.CreateIntensityAttr(300.0 if has_own else 500.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/PreviewSun")
    sun.CreateIntensityAttr(1200.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 20.0, 0.0))


def wait_loaded(limit=900):
    """Kit keeps streaming textures and compiling MDL after the stage opens: update until nothing is pending."""
    ctx = omni.usd.get_context()
    for i in range(limit):
        app.update()
        _, loaded, total = ctx.get_stage_loading_status()
        if i > 30 and loaded >= total:
            return i
    return limit


def camera(sim, name, hfov=66.0):
    path = f"/World/{name}"
    cam = Camera(prim_path=path, resolution=(1280, 720))
    usd = UsdGeom.Camera(sim.stage.GetPrimAtPath(path))
    usd.GetFocalLengthAttr().Set(16.0)
    usd.GetHorizontalApertureAttr().Set(float(2 * 16.0 * np.tan(np.radians(hfov) / 2)))
    usd.GetVerticalApertureAttr().Set(float(2 * 16.0 * np.tan(np.radians(hfov) / 2) * 720 / 1280))
    usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 5000.0))
    return cam


def views(lo, hi, floors):
    """[(name, eye, target)]: the whole scene from above, a plan view, eye height in the largest floors."""
    centre, size = (lo + hi) / 2, hi - lo
    reach = 0.5 * float(np.linalg.norm(size)) / np.tan(np.radians(30)) * 1.05
    d = OVERVIEW / np.linalg.norm(OVERVIEW)
    top_z = 0.5 * max(size[0], size[1] * 16 / 9) / np.tan(np.radians(33)) * 1.05 + hi[2]
    out = [
        ("overview", centre + d * reach, centre),
        ("top", np.array([centre[0], centre[1] - 1e-3, top_z]), np.array([centre[0], centre[1], lo[2]])),
    ]
    picked = []
    for f_lo, f_hi, _ in floors:
        c = (f_lo + f_hi) / 2
        if any(np.linalg.norm((c - p)[:2]) < 1.0 for p in picked):
            continue  # a second mesh of the same floor
        picked.append(c)
        z = f_hi[2]
        eye = np.array([f_lo[0] + INSET, f_lo[1] + INSET, z + EYE_Z])
        target = np.array([f_hi[0] - INSET, f_hi[1] - INSET, z + 0.6])
        out.append((f"room_{len(picked)}", eye, target))
        if len(picked) == args.rooms:
            break
    return out


def main():
    usd, out = Path(args.usd).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(usd)))
    STATS.update(
        usd=str(usd),
        layers=len(layers),
        assets=len(assets),
        unresolved=sorted({str(p) for p in unresolved})[:50],
        unresolved_count=len(unresolved),
    )
    render = sim_utils.RenderCfg(enable_reflections=True, enable_global_illumination=True, rendering_mode="quality")
    carb.settings.get_settings().set_float("/rtx/translucency/worldEps", 0.0)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device, render=render))
    stage = sim.stage
    root = stage.DefinePrim("/World/Scene", "Xform")
    src = Usd.Stage.Open(str(usd), Usd.Stage.LoadNone)
    if src.GetDefaultPrim():
        root.GetReferences().AddReference(str(usd))
    else:  # no default prim to reference: bring its root prims in under /World/Scene one by one
        for prim in src.GetPseudoRoot().GetChildren():
            stage.DefinePrim(f"/World/Scene/{prim.GetName()}").GetReferences().AddReference(str(usd), prim.GetPath())
    up = UsdGeom.GetStageUpAxis(src)
    STATS.update(up_axis=up, meters_per_unit=UsdGeom.GetStageMetersPerUnit(src))
    if up == UsdGeom.Tokens.y:
        UsdGeom.Xformable(root).AddRotateXOp().Set(90.0)
    mpu = UsdGeom.GetStageMetersPerUnit(src)
    if abs(mpu - 1.0) > 1e-6:
        UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3f(mpu, mpu, mpu))
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)  # the SimulationContext brings its own
    own_lights = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdLux.LightAPI)]
    STATS["lights"] = len(own_lights)
    STATS["open_s"] = round(time.time() - t0, 1)
    lo, hi, floors = measure(stage, root)
    hide(stage, root, min((f[0][2] for f in floors), default=lo[2]))
    lights(stage, bool(own_lights))
    shots = [(name, camera(sim, f"cam_{name}"), eye, target) for name, eye, target in views(lo, hi, floors)]
    # no sim.reset(): the timeline stays stopped, so PhysX never parses the scene; cooking the collision shapes of
    # a GRScenes scene's ~1000 objects takes minutes and a preview needs none of it
    for _, cam, eye, target in shots:
        cam.initialize()
        cam.set_world_pose(eye, look_at(eye, target), camera_axes="world")
    STATS["load_updates"] = wait_loaded()
    for _ in range(90):  # let RTX accumulate
        sim.render()
    extra = 0  # a large scene's shaders can still be compiling: its cameras have no frame yet
    while extra < 1200 and any(cam.get_rgba().ndim != 3 for _, cam, _, _ in shots):
        sim.render()
        extra += 1
    STATS["extra_frames"] = extra
    STATS["renders"] = []
    for name, cam, eye, target in shots:
        rgb = cam.get_rgba()[:, :, :3]
        path = out / f"{name}.jpg"
        Image.fromarray(rgb).save(path, quality=88)
        STATS["renders"].append(
            {"name": name, "file": path.name, "eye": eye.round(2).tolist(), "mean": float(rgb.mean())}
        )
    STATS["total_s"] = round(time.time() - t0, 1)


try:
    main()
    STATS["ok"] = True
except Exception:
    STATS["ok"] = False
    STATS["error"] = traceback.format_exc()[-4000:]
    traceback.print_exc()
Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / "stats.json").write_text(json.dumps(STATS, indent=1) + "\n")
print(json.dumps({k: v for k, v in STATS.items() if k not in ("floors", "unresolved", "error")}))
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if STATS["ok"] else 1)  # app.close() hangs once a camera render product exists, and Kit eats exit codes
