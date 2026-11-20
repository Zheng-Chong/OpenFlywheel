"""Check articulated objects (object.usd(z) + metadata.json) in Isaac Sim through env ArticulationObject, the loader
tasks use, the way check_object.py checks rigid ones.

For each object, placed off the origin and turned about Z (--pose), with its base fixed (metadata physics.fixed_base):
    fixed_base  the base link sits where the layout put it (not pulled back to where the source model was authored)
    settle      over --seconds it neither drifts nor jitters
    rest        meanwhile no door, drawer or lid moves off its start (closed) by more than REST_TOL under its own
                weight and drives: nothing opens by itself
    hold        every door, drawer and lid set half open and left with its own drives for HOLD_S stays within
                REST_TOL (it stays where the hand leaves it); reported, not part of the pass: library drives that
                close a door on purpose (ArtVIP's return springs) fail it
    joints      each revolute / prismatic joint, driven past its upper limit, then past its lower limit, then back:
                the range it reaches matches the nominal limits, it gets there (no jam), nothing explodes and the base
                stays put
Pass/fail thresholds are the constants below. With --image-dir it renders each object alone, closed (sim.png) and
with every joint at the limit farthest from its start (open.png), and writes check.json beside them in
<image-dir>/<type>/<category>/<index>/preview/ (the layout the Real2Sim platform reads). Checks it did not run are
not written. metadata.json "size_check" (written by the importers: the model against the real size range of its
kind) is copied into check.json.

Usage:
    python scripts/objects/check_articulation.py DIR [DIR ...] [--image-dir real2sim/objects] [--report out.json]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check articulated objects: fixed base, settling, joint ranges.")
parser.add_argument("dirs", nargs="+", help="object dirs holding object.usd(z) and metadata.json")
parser.add_argument("--seconds", type=float, default=3.0, help="settling time")
parser.add_argument("--phase", type=float, default=2.0, help="seconds per joint phase (to upper, to lower, back)")
parser.add_argument("--pose", type=float, nargs=3, default=(0.37, -0.21, 33.0), metavar=("X", "Y", "YAW_DEG"))
parser.add_argument("--spacing", type=float, default=3.0, help="m between objects along Y")
parser.add_argument("--lift", type=float, default=0.05, help="m between the fixed base and the floor")
parser.add_argument(
    "--free", action="store_true", help="base not fixed: dropped from 1 cm, only the free_settle check (no sweeps)"
)
parser.add_argument(
    "--image-dir", help="render each object and write check.json under <dir>/<type>/<cat>/<idx>/preview/"
)
parser.add_argument("--report", help="also write all results to this JSON file")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # as TaskEnv (sim_config device); ArticulationObject reads DOF limits as numpy
args = parser.parse_args()
args.headless = True
args.enable_cameras = bool(args.image_dir) and not args.free
app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import carb  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
import torch  # noqa: E402

from env.scene_manager.objects.articulation import ArticulationObject  # noqa: E402

DT = 0.004  # env_cfg/sim/sim_config.yml
MIN_SHOWN = 0.01  # of a preview's pixels that must differ from the floor, or the object did not render
FREE_SPEED_TOL, FREE_MOVE_TOL = 0.01, 0.05  # --free: base link speed over the last second, sideways travel
POS_TOL = 0.002  # m: base link vs where the layout put it, and its drift while settling
ROT_TOL = 0.5  # deg
JITTER_TOL = 0.005  # m/s: base link speed over the last second of settling
REV_TOL, PRI_TOL = 2.0, 0.002  # deg / m: a reached limit vs the nominal one (or 3% of the range when larger)
JAM_FRACTION = 0.97  # of the nominal range, swept end to end
SPEED_LIMIT = 50.0  # rad/s or m/s of any joint: an explosion
OVERSHOOT = 0.05  # drive target past the limit, as a fraction of the range (at least 3 deg / 3 mm)
GAIN = 2000.0  # 1/s^2: drive stiffness per unit inertia while sweeping; damping critical
OPENERS = ("door", "lid")  # roles opened when another joint of the object is blocked with them as they start
RAMP = 0.6  # of each phase: the drive target ramps to its end value, then holds
GRAVITY_MARGIN = 4.0  # drive force at the overshoot vs the link's weight (at half the object size for revolute)
VIEW = np.array([0.82, -0.55, 0.45]) / np.linalg.norm([0.82, -0.55, 0.45])
OPENABLE = ("door", "drawer", "lid")  # roles whose start must hold (rest) and that are tried half open (hold)
REST_TOL = (2.0, 0.005)  # deg / m an openable joint may move while left alone
HOLD_S = 2.0


def as_np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=float).reshape(-1)


def yaw_quat(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def quat_angle(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def world_pose(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).RemoveScaleShear()
    q = m.ExtractRotationQuat()  # links may carry a scale, which PhysX body frames do not
    return np.asarray(m.ExtractTranslation(), float), np.array([q.GetReal(), *q.GetImaginary()], float)


def gains(obj, meta, device):
    """Sweep gains per DOF: GAIN x the child link's inertia about the joint (rough: mass x (half the object size)^2
    for revolute, mass for prismatic), and at least enough that the overshoot alone lifts the link GRAVITY_MARGIN
    times against gravity and at 1/8 of it beats the joint's friction (metadata `friction`); acceleration drives are
    mass-independent, so their stiffness is in 1/s^2."""
    joints = {j["name"]: j for j in meta.get("body_joints", [])}
    masses = meta.get("mass_partition", {})
    arm = 0.5 * max(meta["physics"]["size"])
    kps, kds = [], []
    for i, name in enumerate(obj.dof_names):
        j = joints.get(name, {})
        m = masses.get(j.get("child"), 0.5)
        revolute = j.get("type") == "revolute"
        inertia = m * arm * arm if revolute else m
        over = overshoot(obj.dof_properties["lower"][i], obj.dof_properties["upper"][i], revolute)
        friction = (j.get("friction") or {}).get("static", 0.0)
        stiff = max(GAIN * inertia, (GRAVITY_MARGIN * m * 9.81 * (arm if revolute else 1.0) + 8 * friction) / over)
        if (j.get("drive") or {}).get("type") == "acceleration":
            stiff, inertia = stiff / inertia, 1.0
        kps.append(stiff)
        kds.append(2 * math.sqrt(stiff * inertia))
    t = lambda v: torch.tensor([v], dtype=torch.float32, device=device)  # noqa: E731
    return t(kps), t(kds)


def overshoot(lo, hi, revolute):
    """How far past a limit the sweep drives: a fraction of the range, at least 3 deg / 3 mm."""
    rng = hi - lo if math.isfinite(hi - lo) else 0.0
    return max(OVERSHOOT * rng, math.radians(3) if revolute else 0.003)


def continuous(item, i):
    lo, hi = item["lower"][i], item["upper"][i]
    return not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo > 2 * math.pi + 0.01


def sweep_range(item, i):
    """The joint's limits, or a quarter turn either side of its start for an unlimited joint."""
    if continuous(item, i):
        return item["q0"][i] - math.pi / 2, item["q0"][i] + math.pi / 2
    return item["lower"][i], item["upper"][i]


def set_state(obj, q, device):
    t = torch.tensor(np.asarray(q, dtype=np.float32), device=device)
    obj.set_joint_positions(t)
    obj.set_joint_velocities(torch.zeros_like(t))
    obj.apply_action(ArticulationAction(joint_positions=torch.tensor(obj.authored_drive_targets(q), device=device)))


def moved_joints(item, moved):
    """{pass, moved: {joint: deg or m}} over the item's openable joints, against REST_TOL."""
    out, ok = {}, True
    for k in item["openable"]:
        rev = item["revolute"][k]
        value = math.degrees(moved[k]) if rev else float(moved[k])
        out[item["obj"].dof_names[k]] = round(value, 4)
        ok &= bool(np.isfinite(value)) and value <= REST_TOL[0 if rev else 1]
    return {"pass": bool(ok), "moved": out, "unit": "deg (revolute) / m (prismatic)"}


def step(sim, seconds, probe=None):
    for _ in range(max(1, round(seconds / DT))):
        sim.step(render=False)
        if probe:
            probe()


def main():
    render = sim_utils.RenderCfg(enable_translucency=True, enable_reflections=True, rendering_mode="quality")
    carb.settings.get_settings().set_float("/rtx/translucency/worldEps", 0.0)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device, render=render))
    x0, y0, yaw = args.pose
    row = (len(args.dirs) - 1) * args.spacing
    floor = sim_utils.CuboidCfg(
        size=(4.0, row + 4.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38), emissive_color=(0.0, 0.0, 0.0)),
    )
    floor.func("/World/floor", floor, translation=(x0, y0 + row / 2, -0.01))
    sim_utils.DomeLightCfg(intensity=300.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=300.0))
    sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=5.0)
    sun.func("/World/sun", sun, orientation=(0.9397, 0.2418, 0.2418, 0.0))

    stage = sim_utils.get_current_stage()
    items = []
    for k, d in enumerate(map(Path, args.dirs)):
        meta = json.loads((d / "metadata.json").read_text())
        size = meta["physics"]["size"]
        usd = d / "object.usdz" if (d / "object.usdz").is_file() else d / "object.usd"
        pos = (x0, y0 + k * args.spacing, size[2] / 2 + (0.01 if args.free else args.lift))
        item = {"dir": d, "meta": meta, "pos": pos, "result": {"asset_dir": str(d.resolve())}}
        items.append(item)
        try:
            cat, idx = d.parent.name, int(d.name)
            obj = ArticulationObject(
                prim_path=f"/World/envs/env_{k}/articulation/{cat}/{cat}_{idx}",
                usd_path=str(usd),
                inst_config={"physics": {"fixed_base": not args.free}},
                env_origin=np.zeros(3),
                default_pos=pos,
                default_ori=yaw_quat(yaw),
                scale=(1.0, 1.0, 1.0),
            )
            item["obj"] = obj
            base = stage.GetPrimAtPath(f"{obj.prim_path}/{meta['base_link']}")
            item["expect"] = world_pose(base)
        except Exception as exc:  # one broken asset must not stop the batch
            item["result"]["load_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        if args.image_dir:
            item["preview"] = Path(args.image_dir) / d.parent.parent.name / d.parent.name / d.name / "preview"
        if args.image_dir and not args.free:
            cam = Camera(prim_path=f"/World/solo_camera_{k}", resolution=(640, 480))
            centre = np.array(pos)
            eye = centre + VIEW * max(3.2 * float(np.linalg.norm(size)), 0.3)
            set_camera_view(tuple(eye), tuple(centre), camera_prim_path=cam.prim_path)
            item["camera"] = cam

    sim.reset()
    live = []
    for item in items:
        if "obj" not in item:
            continue
        try:
            item["obj"].initialize()
            live.append(item)
        except Exception as exc:
            item["result"]["load_error"] = f"initialize: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    # settle
    for item in live:
        p, q = map(as_np, item["obj"].get_world_pose())
        item["start"] = (p, q)
        item["speed"] = 0.0
        specs = {j["name"]: j for j in item["meta"].get("body_joints", [])}
        names = item["obj"].dof_names
        item["openable"] = [k for k, n in enumerate(names) if specs.get(n, {}).get("role") in OPENABLE]
        item["revolute"] = [specs.get(n, {}).get("type") == "revolute" for n in names]
        item["q_rest"] = as_np(item["obj"].get_joint_positions())
        item["rest_move"] = np.zeros(len(names))
    last = round(1.0 / DT)

    def jitter():
        jitter.n += 1
        for item in live:
            q = as_np(item["obj"].get_joint_positions())
            item["rest_move"] = np.maximum(item["rest_move"], np.abs(q - item["q_rest"]))
        if jitter.n > round(args.seconds / DT) - last:
            for item in live:
                v = np.linalg.norm(as_np(item["obj"].get_linear_velocity()))
                item["speed"] = max(item["speed"], float(v))

    jitter.n = 0
    step(sim, args.seconds, probe=jitter)
    if args.free:
        for item in live:
            p, q = map(as_np, item["obj"].get_world_pose())
            moved = float(np.linalg.norm((p - item["start"][0])[:2]))
            finite = bool(np.all(np.isfinite(p)) and np.all(np.isfinite(as_np(item["obj"].get_joint_positions()))))
            item["result"]["free_settle"] = {
                "pass": bool(finite and item["speed"] < FREE_SPEED_TOL and moved < FREE_MOVE_TOL),
                "moved_m": round(moved, 4),
                "max_speed_m_s": round(item["speed"], 4),
                "z_m": round(float(p[2]), 4),
            }
        live = []  # no sweeps on a free base: the object would move instead of the joint
    for item in live:
        obj, res = item["obj"], item["result"]
        p, q = map(as_np, obj.get_world_pose())
        ep, eq = item["expect"]
        off, turn = float(np.linalg.norm(p - ep)), quat_angle(q, eq)
        drift, drift_turn = float(np.linalg.norm(p - item["start"][0])), quat_angle(q, item["start"][1])
        res["fixed_base"] = {
            "pass": bool(off < POS_TOL and turn < ROT_TOL),
            "offset_m": round(off, 5),
            "offset_deg": round(turn, 3),
            "pose": [round(v, 4) for v in [*args.pose]],
            "distance_from_origin_m": round(float(np.linalg.norm(p[:2])), 4),
        }
        res["settle"] = {
            "pass": bool(drift < POS_TOL and drift_turn < ROT_TOL and item["speed"] < JITTER_TOL),
            "drift_m": round(drift, 5),
            "drift_deg": round(drift_turn, 3),
            "max_speed_m_s": round(item["speed"], 5),
        }
        item["home"] = p
        item["q0"] = as_np(obj.get_joint_positions())
        res["rest"] = moved_joints(item, item["rest_move"])

    # hold: the openable joints half open, the object's own drives aimed where the USD aims them
    for item in live:
        obj, q = item["obj"], item["q0"].copy()
        for k in item["openable"]:
            lo, hi = (
                sweep_range(item, k)
                if "lower" in item
                else (obj.dof_properties["lower"][k], obj.dof_properties["upper"][k])
            )
            if not (math.isfinite(lo) and math.isfinite(hi)):
                continue
            far = hi if abs(hi - q[k]) >= abs(lo - q[k]) else lo
            q[k] = q[k] + 0.5 * (far - q[k])
        item["q_half"] = q
        set_state(obj, q, sim.device)
    step(sim, HOLD_S)
    for item in live:
        moved = np.abs(as_np(item["obj"].get_joint_positions()) - item["q_half"])
        item["result"]["hold"] = moved_joints(item, moved)
        set_state(item["obj"], item["q0"], sim.device)
    step(sim, 0.2)

    # joint sweeps, all objects at once, one DOF index at a time
    for item in live:
        obj = item["obj"]
        props = obj.dof_properties
        item["lower"], item["upper"] = np.asarray(props["lower"], float), np.asarray(props["upper"], float)
        item["joints"] = []
        item["peak"] = 0.0
        item["max_off"] = 0.0
        try:
            kp, kd = gains(obj, item["meta"], sim.device)
            obj._articulation_view.set_gains(kps=kp, kds=kd)
            obj._articulation_view.set_max_efforts(torch.full((1, obj.num_dof), 1e6, device=sim.device))
        except Exception as exc:
            item["result"]["gain_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def watch():
        for item in live:
            obj = item["obj"]
            v = as_np(obj.get_joint_velocities())
            item["peak"] = max(item["peak"], float(np.nanmax(np.abs(v))) if np.all(np.isfinite(v)) else math.inf)
            p = as_np(obj.get_world_pose()[0])
            item["max_off"] = max(item["max_off"], float(np.linalg.norm(p - item["home"])))

    def drive(item, i=None, value=None):
        obj = item["obj"]
        target = item["base"].copy()
        if i is not None:
            target[i] = value
        obj.apply_action(
            ArticulationAction(joint_positions=torch.tensor(target, dtype=torch.float32, device=sim.device))
        )

    for item in live:  # doors and lids, opened when a joint behind them (rack, tray) is blocked with them closed
        specs = {j["name"]: j for j in item["meta"].get("body_joints", [])}
        item["base"], item["open"] = item["q0"].copy(), item["q0"].copy()
        item["openers"] = [k for k, n in enumerate(item["obj"].dof_names) if specs.get(n, {}).get("role") in OPENERS]
        for k in item["openers"]:
            lo, hi = sweep_range(item, k)
            item["open"][k] = hi if abs(hi - item["q0"][k]) >= abs(lo - item["q0"][k]) else lo

    def move_openers(items, opening):
        plans = [
            (item, k, item["base"][k], (item["open"] if opening else item["q0"])[k])
            for item in items
            for k in item["openers"]
        ]
        if not plans:
            return
        n = max(1, round(RAMP * args.phase / DT))
        for s in range(n):
            for item, k, start, end in plans:
                item["base"][k] = start + (end - start) * (s + 1) / n
            for item in items:
                drive(item)
            sim.step(render=False)
            watch()
        step(sim, (1 - RAMP) * args.phase, probe=watch)

    def sweep(group, i):
        """Drive DOF i of every item past its upper, then lower limit and back; {id(item): joint record}."""
        reached = {}
        for phase in ("upper", "lower", "back"):
            ramps = []
            for item in group:
                lo, hi = sweep_range(item, i)
                revolute = item["obj"].dof_names[i] in {
                    j["name"] for j in item["meta"].get("body_joints", []) if j["type"] == "revolute"
                }
                over = 0.0 if continuous(item, i) else overshoot(lo, hi, revolute)
                value = {"upper": hi + over, "lower": lo - over, "back": item["q0"][i]}[phase]
                ramps.append((item, float(as_np(item["obj"].get_joint_positions())[i]), value))
            n = max(1, round(RAMP * args.phase / DT))
            for k in range(n):  # the target moves at a bounded speed, then holds
                for item, start, value in ramps:
                    drive(item, i, start + (value - start) * (k + 1) / n)
                sim.step(render=False)
                watch()
            step(sim, (1 - RAMP) * args.phase, probe=watch)
            for item in group:
                reached.setdefault(id(item), {})[phase] = float(as_np(item["obj"].get_joint_positions())[i])
        records = {}
        for item in group:
            name = item["obj"].dof_names[i]
            spec = next((j for j in item["meta"].get("body_joints", []) if j["name"] == name), {})
            revolute = spec.get("type") == "revolute"
            lo, hi = sweep_range(item, i)
            free = continuous(item, i)
            r = reached[id(item)]
            unit = math.degrees if revolute else (lambda v: v)
            tol = max(REV_TOL if revolute else PRI_TOL, 0.03 * unit(hi - lo))
            nominal = spec.get("limits_deg" if revolute else "limits_m") or [unit(lo), unit(hi)]
            if free or None in nominal:  # an unlimited joint is swept a quarter turn either way
                nominal = [unit(lo), unit(hi)]
            err_hi, err_lo = abs(unit(r["upper"]) - nominal[1]), abs(unit(r["lower"]) - nominal[0])
            swept = (r["upper"] - r["lower"]) / (hi - lo) if hi > lo else 0.0
            back = abs(unit(r["back"] - item["q0"][i]))
            records[id(item)] = {
                "name": name,
                "role": spec.get("role"),
                "part": spec.get("part"),
                "unit": "deg" if revolute else "m",
                "nominal": [round(v, 4) for v in nominal],
                "physx": [round(unit(lo), 4), round(unit(hi), 4)],
                "reached": [round(unit(r["lower"]), 4), round(unit(r["upper"]), 4)],
                "back_error": round(back, 4),
                "continuous": bool(free),
                "pass": bool(err_hi <= tol and err_lo <= tol and swept >= JAM_FRACTION and back <= tol),
            }
        # leave the joint where it started before the next one moves
        for item in group:
            drive(item, i, item["q0"][i])
        return records

    n_max = max((item["obj"].num_dof for item in live), default=0)
    for i in range(n_max):
        group = [item for item in live if i < item["obj"].num_dof]
        records = sweep(group, i)
        # blocked with the doors as they start (an oven rack behind its door): again with them open; the far limit
        # is not always the open one (ArtVIP's oven_2 door rests open at 0), so the first sweep comes first
        retry = [
            item for item in group if not records[id(item)]["pass"] and item["openers"] and i not in item["openers"]
        ]
        if retry:
            move_openers(retry, True)
            again = sweep(retry, i)
            move_openers(retry, False)
            for item in retry:
                if again[id(item)]["pass"]:
                    records[id(item)] = {**again[id(item)], "doors_opened": True}
        for item in group:
            item["joints"].append(records[id(item)])
    for item in live:
        res = item["result"]
        ok = all(j["pass"] for j in item["joints"]) and item["peak"] < SPEED_LIMIT and item["max_off"] < POS_TOL
        res["joints"] = {
            "pass": bool(item["joints"]) and bool(ok),
            "dof": item["joints"],
            "peak_joint_speed": round(item["peak"], 3),
            "base_max_offset_m": round(item["max_off"], 5),
        }

    cameras = [item for item in items if "camera" in item]
    if cameras:
        for item in cameras:
            item["camera"].initialize()
            item["camera"].set_clipping_range(0.01, 100.0)
        snapshots = [("sim.png", None), ("open.png", "open")]
        for name, state in snapshots:
            if state == "open":
                for item in live:
                    q = item["q0"].copy()
                    lo = np.array([sweep_range(item, i)[0] for i in range(len(q))])
                    hi = np.array([sweep_range(item, i)[1] for i in range(len(q))])
                    far = np.where(np.abs(hi - q) >= np.abs(lo - q), hi, lo)
                    t = torch.tensor(far, dtype=torch.float32, device=sim.device)
                    item["obj"].set_joint_positions(t)
                    item["obj"].apply_action(ArticulationAction(joint_positions=t))
                step(sim, 0.5)
            for _ in range(30):
                sim.render()
            for item in cameras:
                rgba = item["camera"].get_rgba()
                for _ in range(20):  # the RTX pipeline sometimes hands back empty frames first
                    if getattr(rgba, "ndim", 0) == 3:
                        break
                    for _ in range(10):
                        sim.render()
                    rgba = item["camera"].get_rgba()
                if getattr(rgba, "ndim", 0) != 3:
                    item["result"]["render_error"] = f"{name}: empty frame"
                    continue
                item["preview"].mkdir(parents=True, exist_ok=True)
                rgb = np.asarray(rgba[:, :, :3])
                Image.fromarray(rgb).save(item["preview"] / name)
                item["result"].setdefault("images", []).append(name)
                # the object must show: pixels away from the floor's colour (the image median)
                shown = float((np.abs(rgb.astype(int) - np.median(rgb, axis=(0, 1))).sum(-1) > 60).mean())
                if shown < MIN_SHOWN:
                    item["result"]["render_error"] = f"{name}: object covers {shown:.1%} of the frame"

    failed = False
    for item in items:
        res, meta = item["result"], item["meta"]
        checks, notes = {}, {}
        for key in ("fixed_base", "settle", "rest", "joints", "free_settle", "hold"):
            if key in res:
                checks[key] = "pass" if res[key]["pass"] else "fail"
        if "size_check" in meta:
            checks["size"] = meta["size_check"]["result"]
            notes["size"] = meta["size_check"]["note"]
        if res.get("render_error"):
            checks["render"] = "fail"
            notes["render"] = res["render_error"]
        elif res.get("images"):
            checks["render"] = "pass"
            notes["render"] = (
                f"start pose sim.png and every joint at its far limit open.png: the object covers at least "
                f"{MIN_SHOWN:.0%} of each frame; appearance judged by eye"
            )
        if "joints" in res:
            notes["joints"] = (
                f"each DOF driven {OVERSHOOT:.0%} past its upper then lower limit and back, {args.phase} s each "
                f"(target ramped over {RAMP:.0%}); a joint that fails with the doors and lids as they start is "
                "swept again with them open; "
                "unlimited joints a quarter turn either way; "
                f"reached within {REV_TOL} deg / {PRI_TOL * 1000:.0f} mm or 3% of the range; penetration not checked"
            )
        if "fixed_base" in res:
            notes["fixed_base"] = (
                f"placed at x={args.pose[0]}, y={args.pose[1]} turned {args.pose[2]} deg, {args.lift * 100:.0f} cm "
                "above the floor (a bottom-hinged door may swing below the base) with fixed_base; base link "
                f"within {POS_TOL * 1000:.0f} mm / {ROT_TOL} deg of that pose after {args.seconds} s"
            )
        res.update(
            {
                "checks": checks,
                "notes": notes,
                "mass_kg": meta["physics"].get("mass"),
                "size_m": meta["physics"]["size"],
                "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        if "free_settle" in res:
            notes["free_settle"] = (
                f"base not fixed, dropped from 1 cm onto the floor: after {args.seconds} s slower than "
                f"{FREE_SPEED_TOL} m/s and within {FREE_MOVE_TOL * 100:.0f} cm sideways"
            )
        if "rest" in res:
            notes["rest"] = (
                f"during the {args.seconds} s settle no door, drawer or lid moved off its start by more than "
                f"{REST_TOL[0]} deg / {REST_TOL[1] * 1000:.0f} mm (nothing opens by itself)"
            )
        if "hold" in res:
            notes["hold"] = (
                f"every door, drawer and lid set half open, left {HOLD_S} s with its own drives: within "
                f"{REST_TOL[0]} deg / {REST_TOL[1] * 1000:.0f} mm (stays where left); not part of the pass"
            )
        ran = "free_settle" if args.free else "joints"
        ok = "load_error" not in res and all(v != "fail" for k, v in checks.items() if k != "hold") and ran in checks
        res["pass"] = ok
        failed |= not ok
        for j in res.get("joints", {}).get("dof", []):
            print(
                f"    {'ok  ' if j['pass'] else 'bad '}{j['name']} ({j['role']}): nominal {j['nominal']} "
                f"reached {j['reached']} back {j['back_error']} {j['unit']}"
            )
        print(
            f"{'OK  ' if ok else 'FAIL'} {item['dir']}: "
            + json.dumps({k: res.get(k) for k in ("checks", "load_error") if res.get(k)})
        )
        if "preview" in item:
            item["preview"].mkdir(parents=True, exist_ok=True)
            name = "check_free.json" if args.free else "check.json"
            (item["preview"] / name).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    if args.report:
        Path(args.report).write_text(json.dumps([i["result"] for i in items], ensure_ascii=False, indent=2) + "\n")
    return failed


try:
    failed = main()
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)  # app.close() hangs once a camera render product exists, and Kit eats exit codes
