"""Drop every FMB peg into its hole of a peg board in Isaac Sim and check that it goes through to the floor.

Reads the holes from the board's metadata.json (fmb.holes, scripts/objects/fmb.py): each peg starts 5 mm above the
board, centred on its hole and turned the way it goes in. The holes go through the board, so a peg that fits ends on
the floor; one the colliders stop rests on the board.

Usage:
    python scripts/objects/check_fmb_insertion.py RIGID_DIR [--size 0 1 2] [--length short|long] [--offset-mm MM]
        [--image-dir DIR]
    e.g. python scripts/objects/check_fmb_insertion.py Assets/Object/FMB/Rigid --image-dir .cache/fmb_insertion
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drop FMB pegs into their peg-board holes.")
parser.add_argument("rigid_dir", help="the FMB Rigid dir (fmb_peg_board/, fmb_peg_<shape>/)")
parser.add_argument("--size", type=int, nargs="+", default=[0, 1, 2], help="peg boards / peg sizes to test")
parser.add_argument("--length", choices=["short", "long"], default="short")
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument(
    "--offset-mm",
    type=float,
    default=0.0,
    help="start pegs this far off their holes along X (a control: past the clearance every peg must fail)",
)
parser.add_argument("--image-dir", help="render each board with its pegs to <dir>/board_<size>.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = bool(args.image_dir)
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from env.scene_manager.objects.rigid import RigidObject  # noqa: E402

DT = 0.004  # env_cfg/sim/sim_config.yml
SPACING = 0.6  # m between boards along Y
FLIP = (0.0, 1.0, 0.0, 0.0)  # half a turn about X: multi-lobe pegs go in lobes first


def as_np(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=float).reshape(-1)


def load(d, name, pos, ori):
    meta = json.loads((d / "metadata.json").read_text())
    obj = RigidObject(
        prim_path=f"/World/rigid/{name}",
        usd_path=str(d / "object.usdz"),
        inst_config={"physics": meta["physics"]},
        env_origin=[0.0, 0.0, 0.0],
        default_pos=pos,
        default_ori=ori,
        scale=(1.0, 1.0, 1.0),
    )
    return obj, meta


def main():
    """Returns True if any peg failed."""
    rigid = Path(args.rigid_dir)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    row = (len(args.size) - 1) * SPACING
    floor = sim_utils.CuboidCfg(
        size=(2.0, row + 2.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.38)),
    )
    floor.func("/World/floor", floor, translation=(0.0, row / 2, -0.01))
    dome = sim_utils.DomeLightCfg(intensity=300.0)
    dome.func("/World/dome", dome)
    sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=5.0)
    sun.func("/World/sun", sun, orientation=(0.9397, 0.2418, 0.2418, 0.0))

    boards, pegs, cams = [], [], []
    for k, size in enumerate(args.size):
        y0 = k * SPACING
        board, meta = load(rigid / "fmb_peg_board" / f"{size:05d}", f"board_{size}_{k}", (0.0, y0, 0.026), (1, 0, 0, 0))
        boards.append(board)
        top = meta["physics"]["size"][2]
        for hole in meta["fmb"]["holes"]:
            shape = hole["peg"].removeprefix("fmb_peg_")
            index = (size * 2 + (args.length == "long")) * 8
            d = rigid / hole["peg"] / f"{index:05d}"
            half = json.loads((d / "metadata.json").read_text())["physics"]["size"][2] / 2
            x, y = hole["center"]
            x += args.offset_mm / 1000
            ori = FLIP if hole["inserted_end"] == "top" else (1.0, 0.0, 0.0, 0.0)
            peg, _ = load(d, f"{hole['peg']}_{size}_{k}", (x, y0 + y, top + half + 0.005), ori)
            pegs.append((size, shape, peg, half, (x, y0 + y), hole["collider_clearance_mm"]))
        if args.image_dir:
            cam = Camera(prim_path=f"/World/camera_{size}", resolution=(960, 720))
            set_camera_view((0.45, y0 - 0.25, 0.45), (0.0, y0, 0.03), camera_prim_path=cam.prim_path)
            cams.append((cam, Path(args.image_dir) / f"board_{size}.png"))

    sim.reset()
    for obj in boards + [p[2] for p in pegs]:
        obj.initialize()
    for _ in range(round(args.seconds / DT)):
        sim.step(render=False)

    failed = False
    for size, shape, peg, half, xy, gap in pegs:
        pos, quat = map(as_np, peg.get_world_pose())
        _, qx, qy, _ = quat
        tilt = np.degrees(np.arccos(min(abs(1 - 2 * (qx * qx + qy * qy)), 1.0)))  # flipped pegs point down
        off = float(np.hypot(*(pos[:2] - xy)))
        ok = abs(pos[2] - half) < 0.003 and off < 0.003 and tilt < 3
        failed |= not ok
        print(
            f"{'OK  ' if ok else 'FAIL'} board {size} {shape}: z={pos[2]:.4f} (floor {half:.4f}) "
            f"xy off={off * 1000:.1f} mm tilt={tilt:.2f} deg (collider clearance {gap} mm)"
        )

    for cam, path in cams:
        cam.initialize()
        cam.set_clipping_range(0.001, 100.0)
    for _ in range(30 if cams else 0):
        sim.render()
    for cam, path in cams:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cam.get_rgba()[:, :, :3]).save(path)
        print(f"image: {path}")
    return failed


try:
    failed = main()
except Exception:
    traceback.print_exc()
    failed = True
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)  # app.close() hangs once a camera render product exists, and Kit eats exit codes
