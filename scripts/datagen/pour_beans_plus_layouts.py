"""Layouts of g1d_pour_beans_plus (倒豆子 Plus): training layouts and the graded benchmark, generalized along the five
dimensions of the README (§三 分级泛化评测集), each at a level L0 (twin) to L3 (out of distribution):

    spatial   L0 the twin cups' places (±1 cm; nearer the robot where a cup's grasp is out of the arm's reach,
              REACH_MAX); L1 ±5 cm, handle ±15 deg off the fingers; L2 anywhere in reach,
              ±30 deg, robot base ±3 cm sideways / to the desk, ±5 deg yaw, lift ±2 cm, desk ±3 cm; L3 the reach's
              edge, ±45 deg, base and desk past the training range (±6 cm, ±10 deg, desk ±5 cm)
    visual    L0 the scene as built; L1 one extra light, brightness and colour temperature near the scene's; L2 1-3
              lights moved, floor / wall / tabletop from the scene profile's material pool, 2-4 training distractors;
              L3 the held-out scene (HELD_SCENE) with materials from outside its profile, 6-10 distractors with
              held-out ones
    object    L0 the twin pair (two plastic_cup 00000); L1 the twin and one other training cup unscaled (after the
              feasibility exclusions the twin's family has no second holdable cup); L2 any training cups,
              scaled ±15%; L3 evaluation cups only (cups.json split), half of them scaled past the training range;
              the bean count a share of the smaller cup's capacity, wider with the level; red beans at L0, red,
              mung or black beans above
    camera    L0 as configured; L1 image brightness / contrast / saturation ±10%; L2 extrinsics (head ±1 cm / ±2 deg,
              wrists ±0.5 cm / ±2 deg), image ±25% and noise; L3 extrinsics past that, blur, over- or underexposure
    language  L0 the canonical instruction with the short names (side names when the cups look alike); L1 any
              training template, a short or middle name or the side; L2 any name level, a same-kind distractor cup
              the names must tell apart; L3 held-out templates, a distractor differing in one attribute only

Every layout carries what it was drawn from in "Generalization", its instruction in "Instruction" (--lang; both
languages in the record), each cup's cavity profile (scripts/objects/cup_profile.py) in its entry, and the beans as
rigid grains of the kind's pack (prepare_pour_half_layouts.BEAN_PACK, in the assets) that fit in the source cup
(Fluid entry in the cup's frame, physics.type granular, physics.cavity / yaw / count: cup_profile.pack_fill). The
robot base moves by moving the scene and every object rigidly about the robot root the other way (layouts have no
robot pose); the desk height through the scene's table lift (scripts/scenes/appearance.py).

    train  --count layouts, each dimension at L0 / L1 / L2 with TRAIN_MIX, training cups only, TRAIN_SCENES
    bench  the graded sets G0 (25, all L0) to G3 (50 each) and the single-dimension diagnostics (10 per dimension
           and level L1-L3, the rest L0): 325 layouts; --reps N writes them N times over (layout i is set i mod 325,
           the first 325 the same as with --reps 1), spares for scripts/datagen/pour_beans_plus_bench.py
    probe  each loadable cup once the source and once the target, all L0
    narrow --count layouts in one --scene that vary only like a real collection: the cups' places (spatial L2, hand
           either side) and the 2-4 training distractors (visual L2), with the robot base, desk, lights and
           materials as built (NARROW_FIXED) and the other dimensions at L0 (twin cups, red beans, camera, canonical
           instruction): the narrow control group of the train layouts
    recipe the strata of --recipe (a JSON, scripts/datagen/recipes/), stratum k in seed folder --seed + k of each of
           its scenes, its sets.json with the stratum's share of accepted episodes ("target", for run_batch.py
           --target): per stratum the candidates' level combos or per-dimension level weights, cup pools, scale
           ranges and distractor count, training content at L0-L2 only (check_stratum)
    exam   the grasp exam (scripts/datagen/grasp_exam.py): --count cycles of the --cups, spatial L1
    gen    the expert's generalization exam: --count random pairs of the --cups, either one the source, each at its
           own scale in GEN_SCALE, GEN_FILL of the smaller capacity in beans of any kind, either hand, spatial L1
    capsule each scene's folder --src rewritten into --seed with the beans as capsules (CAPSULE_PHYSICS, packs
           CAPSULE_PACK): everything else as laid out, the bean count recomputed for the same fill share
           (pour_beans_plus@v2's candidates from v1's)

Writes <assets>/Eval_Layout/RoboDojo/<scene>/<seed>/g1d_pour_beans_plus_<i>.json and sets.json (file -> set,
levels) per scene. Train and bench need different --seed folders; their random streams are keyed by mode.

    python scripts/datagen/pour_beans_plus_layouts.py train --assets <run dir>/Assets --seed 970 --count 1500
    python scripts/datagen/pour_beans_plus_layouts.py bench --assets <run dir>/Assets --seed 971
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d import scripted_demo as demo  # noqa: E402
from scripts.g1d.prepare_bean_pour_layouts import TABLE, ZONES, clutter_library, place_pose  # noqa: E402
from scripts.g1d.prepare_pour_half_layouts import BEAN_PACK, BEANS, GRAIN_PHYSICS  # noqa: E402
from scripts.objects import cup_profile  # noqa: E402
from scripts.objects.cup_pool import POOL, TWIN  # noqa: E402
from scripts.scenes import appearance  # noqa: E402

TASK = "g1d_pour_beans_plus"
DIMS = ("spatial", "visual", "object", "camera", "language")
INSTRUCTIONS = ROOT / "real2sim" / "tasks" / TASK / "instructions.json"
# per-cup grasp entries {cups: {id: {rim_down, pitch, width_max, source: {...}, target: {...}}}} from the grasp exam
# (scripts/datagen/grasp_exam.py): copied into the layout entries ("grasp"), read by scripted_demo.plus_actions;
# the cups under "drop" ({id: reason}) are never drawn
GRASPS = ROOT / "real2sim" / "tasks" / TASK / "grasps.json"
SCENES = {"g1d": "real_scene1_office", "g1d_shenzhen_lab": "real_scene5_shenzhen_lab", "g1d_lab": "real_scene3_lab"}
TRAIN_SCENES, HELD_SCENE, TWIN_SCENE = ("g1d", "g1d_shenzhen_lab"), "g1d_lab", "g1d"
TEMPLATE = "Eval_Layout/RoboDojo/{scene}/0/g1d_pour_beans_cup_to_cup_0.json"  # the twin cups on each scene's desk
TRAIN_MIX = (0.1, 0.3, 0.6)  # chance of L0 / L1 / L2 per dimension in a training layout
BENCH = {"G0": 25, "G1": 50, "G2": 50, "G3": 50}
NARROW = {"spatial": 2, "visual": 2, "object": 0, "camera": 0, "language": 0}
NARROW_FIXED = ("base", "lights", "materials")  # parts of spatial / visual L2 the narrow mode keeps at L0
DIAG = 10  # per dimension and level
RECIPE_KEYS = {"name", "why", "target", "candidates", "combos", "levels", "scenes", "cups", "same", "scale", "clutter"}

# spatial (m, deg). Cups: |x| and y of the twin cups, the probed reach (prepare_office_beans_layouts), its edge
ROBOT_ROOT = np.array([0.0, -0.73])  # env_cfg/robot/g1d*.yml default_root_pos
TWIN_XY, TWIN_JITTER, NEAR = (0.126, -0.33), 0.01, 0.05
REACH_X, REACH_Y = (0.05, 0.22), (-0.46, -0.28)
EDGE_X, EDGE_Y = (0.22, 0.26), ((-0.50, -0.46), (-0.28, -0.25))
# The arm is straight (elbow ~1.43 rad, the shoulder 0.292 m from the wrist) when a grasp point is ~0.48 m from the
# shoulder; holding a cup there the IK Jacobian is singular and the lift flings it (probe3: 41% of the grasps with the
# elbow >= 1.4 lost the cup, 13% below). Every cup's grasp point (its axis, the pads' height) is kept within REACH_MAX
# of the holding arm's shoulder; the twin places (L0, L1) move toward the robot until it is.
SHOULDER = (0.1, -0.673, 1.093)  # m, world, |x| and z at lift 0 (g1_d_full_dex1_v4.urdf, robot root at ROBOT_ROOT)
REACH_MAX = 0.46
HANDLE_DEV = (0, 15, 30, 45)  # a handle this far off the holding hand's fingers (prepare_bean_pour_layouts)
# A cup held by its handle (scripted_demo.handle_grasp): the inner finger has 1-2 mm to spare in the hole, and the IK
# puts the nearly level hand within 0.5 mm of the bar only with the bar in this zone (|x|, y m, the robot frame; probed
# with the URDF and the arm's IK at both training scenes' lifts) and the handle straight out: no deviation, any level
HANDLE_ZONE = ((0.09, 0.18), (-0.50, -0.42))
CUP_GAP = 0.04  # m between the cups' bodies
# robot base and desk: train range (L2) and the out-of-distribution bound (L3; sampled past the train range)
BASE = {
    "lateral": (0.03, 0.06),
    "distance": (0.03, 0.05),
    "yaw": (5.0, 10.0),
    "lift": (0.02, 0.03),
    "table": (0.03, 0.05),
}

# visual
LIGHT_BOX = ((-0.6, 0.6), (-0.8, 0.2), (1.6, 2.4))  # m, an extra light above the desk (x5_layouts, the G1-D desk)
LIGHT_TILT = 20
LIGHT_REF = 3e4  # rsi_profile.LIGHT_REF of the area lights
LIGHT_SHAPE = {
    "Sphere": {"radius_range": (0.05, 0.2)},
    "Disk": {"radius_range": (0.05, 0.2)},
    "Rect": {"width_range": (0.3, 1.0), "height_range": (0.3, 1.0)},
}
LIGHTS = {  # level -> (count, intensity factor, colour temperature K)
    1: ((1, 1), (0.1, 0.4), (4500, 6500)),
    2: ((1, 3), (0.2, 1.0), (3500, 7500)),
    3: ((2, 4), (0.2, 1.5), (2700, 9000)),
}
SWAP = 0.7  # chance a surface gets a new material (L2, L3)
CLUTTER = {2: (2, 4), 3: (6, 10)}
HELD_CLUTTER = 0.3  # share of the clutter categories kept for L3
DISTRACTOR_CLEAR, DISTRACTOR_GAP = 0.16, 0.02  # m: from a task cup's centre / between footprints

# object
SCALE = {2: (0.85, 1.15), 3: ((0.75, 0.85), (1.15, 1.25))}
PROBE_SCALES = (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15)  # the training range (L2)
RIM_FREE = 0.025  # m the beans' heap stays below each rim: beans settled within ~20 mm of it splash out on the lift
FILL = {0: (0.35, 0.35), 1: (0.3, 0.4), 2: (0.2, 0.5), 3: (0.2, 0.6)}  # beans: share of the smaller capacity
BEANS_MAX = 1200
GEN_SCALE, GEN_FILL = (0.85, 1.15), (0.1, 0.6)
GRIP_WIDTH = (0.03, 0.08)  # m, widest body the pads slide past: >= 7.5 mm a side in the 95 mm opening
CUP_HEIGHT = (0.05, 0.20)
MOUTH = 0.02  # m, smallest inner rim radius of a target
DEPTH = 0.06  # m, shallowest cavity: RIM_FREE below the rim leaves a shallower cup a few dozen beans
# beans: one rigid body per grain (GranularObject) of a kind of BEANS, red at object L0, any kind above, taken from
# the kind's pack (grains settled at random in the tall measuring cup, rigid_bean_probe.py --pack): grains spawned on
# a lattice lie aligned like bricks and jam in the tilted cup (half the probe's pours left most beans in it). A grain
# on a cup wall holds by its own friction (dry beans on glazed ceramic or plastic, ~0.3): averaged with the cups'
# (0.6 static, 1.5 dynamic, RigidObject) they stayed on the lower wall until it sloped ~25 deg past vertical
GRAIN_COMBINE = "min"
# capsule beans (pour_beans_plus@v2, mode capsule): the grain physics of the real520 twin
# (real2sim/tasks/g1d_pour_beans_real520_v2/make_layouts.py), from packs settled as these capsules
# (rigid_bean_probe.py --shape capsule, real2sim/tasks/g1d_pour_beans_plus/packs/)
CAPSULE_PACK = "real2sim/tasks/g1d_pour_beans_plus/packs/{kind}_capsule.npz"
CAPSULE_PHYSICS = dict(
    type="granular",
    shape="capsule",
    friction=0.42,
    friction_combine="average",
    restitution=0.1,
    contact_offset=0.001,
    rest_offset=0.0,
    torsional_patch_radius=0.0008,
    max_depenetration_velocity=1.0,
    iterations=32,
    position_iterations=32,
    velocity_iterations=1,
    linear_damping=1.5,
    angular_damping=6.0,
    sleep_threshold=0.005,
)

# camera
WRISTS = ("cam_left_wrist", "cam_right_wrist")
EXTRINSIC = {  # level -> {camera group: ((pos low, high) m, (ori low, high) deg)} magnitudes
    2: {"head": ((0.0, 0.01), (0.0, 2.0)), "wrist": ((0.0, 0.005), (0.0, 2.0))},
    3: {"head": ((0.01, 0.02), (2.0, 5.0)), "wrist": ((0.005, 0.01), (2.0, 4.0))},
}
IMAGE = {1: 0.1, 2: 0.25, 3: 0.25}  # brightness / contrast / saturation factor 1 ± this
NOISE = {2: (0.0, 0.02), 3: (0.02, 0.04)}

# language
SIDE = {"zh": ("左边的{}", "右边的{}"), "en": ("the {} on the left", "the {} on the right")}
NOUN = {"zh": "杯子", "en": "cup"}
SIDE_SHARE = 0.2  # of the L1 instructions naming the cups by side


def signed(rng, low, high):
    return rng.choice((-1, 1)) * rng.uniform(low, high)


def level_of(set_name, dim):
    if set_name.startswith("G"):
        return int(set_name[1])
    _, d, lv = set_name.split("_")
    return int(lv[1]) if d == dim else 0


class Cups:
    def __init__(self, pool, grasps=None, depth=DEPTH):
        self.all = pool["cups"]
        self.grasps = (grasps or {}).get("cups", {})
        self.depth = depth
        split = pool.get("split") or {}
        drop = (grasps or {}).get("drop", {})  # {cup id: why the expert cannot hold it}
        usable = {c for c, v in self.all.items() if v["loads"] and c not in drop}
        self.train = sorted(usable & set(split.get("train") or self.all))
        self.eval = sorted(usable & set(split.get("eval") or ()))
        if not self.eval:
            raise SystemExit(f"No evaluation cups in {POOL}: run scripts/objects/cup_pool.py split first")

    def grip(self, cup_id, target):
        """The cup's grasp entry (GRASPS) for its role."""
        return demo.role_grip(self.grasps.get(cup_id), "target" if target else "source")

    def grip_up(self, cup_id, scale, target):
        """Pads above the cup's underside (m) as the expert grasps it in its role."""
        g, prof = self.grip(cup_id, target), self.all[cup_id]["cavity"]
        if g.get("mode") == "handle":
            return float(demo.handle_point(prof, scale, g["handle"])[2] - prof["bottom"] * scale)
        return demo.plus_grip(prof, scale, g.get("rim_down"), g.get("pitch"))[0]

    def grip_out(self, cup_id, scale, target):
        """The grasp point's offset (m) from the cup axis out toward the holding arm's side: the handle's bar held by
        it (handle_yaw), else none."""
        g = self.grip(cup_id, target)
        return (g["handle"]["bar"][0] + g["handle"]["bar"][1]) / 2 * scale if g.get("mode") == "handle" else 0.0

    def fits(self, cup_id, scale, target):
        """The expert can hold the cup at `scale` in its role, and as a target pour into it."""
        cup = self.all[cup_id]
        prof = cup["cavity"]
        height = (prof["top"] - prof["bottom"]) * scale
        g = self.grip(cup_id, target)
        if g.get("mode") == "handle":
            held = demo.handle_fits(g["handle"], scale)
        else:
            up, pitch = demo.plus_grip(prof, scale, g.get("rim_down"), g.get("pitch"))
            width = demo.plus_pad_width(prof, scale, up, pitch)
            held = GRIP_WIDTH[0] <= width <= g.get("width_max", GRIP_WIDTH[1])
        ok = held and CUP_HEIGHT[0] <= height <= CUP_HEIGHT[1]
        ok = ok and (prof["z"][-1] - prof["z"][0]) * scale >= self.depth
        return ok and (not target or prof["r"][-1] * scale >= MOUTH)


def scale_of(rng, level):
    if level < 2:
        return 1.0
    if level == 2 or rng.random() < 0.5:
        return rng.uniform(*SCALE[2])
    return rng.uniform(*rng.choice(SCALE[3]))


def draw_cups(rng, cups, level, split):
    """(source id, target id, source scale, target scale) for an object level."""
    if level == 0:
        return TWIN, TWIN, 1.0, 1.0
    if level == 1:
        other = rng.choice(
            [c for c in cups.train if c != TWIN and cups.fits(c, 1.0, False) and cups.fits(c, 1.0, True)]
        )
        return (TWIN, other, 1.0, 1.0) if rng.random() < 0.5 else (other, TWIN, 1.0, 1.0)
    pool = cups.eval if level == 3 else cups.train
    for _ in range(500):
        a, b = rng.sample(pool, 2)
        sa, sb = scale_of(rng, level), scale_of(rng, level)
        if cups.fits(a, sa, False) and cups.fits(b, sb, True):
            return a, b, sa, sb
    raise RuntimeError(f"no holdable cup pair at object L{level} ({split})")


def radius(cup, scale):
    return max(cup["cavity"]["body"]["r"]) * scale


def handle_yaw(rng, cup, out, level, by_handle=False, base_yaw=0.0):
    """Cup yaw (deg): a handle along the holding hand's fingers (scripted_demo.cup_grasp turns the hand CUP_YAW in from
    fingers-forward) within the level's deviation, or, held `by_handle` (scripted_demo.handle_grasp), straight out to
    the side of the robot turned `base_yaw` (transform()); any yaw without one."""
    handle = cup["cavity"].get("handle")
    if handle is None:
        return rng.uniform(0, 360) if level else 0.0
    if by_handle:
        return ((0.0 if out > 0 else 180.0) + base_yaw - handle) % 360
    return (90 + out * demo.CUP_YAW + rng.uniform(-HANDLE_DEV[level], HANDLE_DEV[level]) - handle) % 360


def lift_of(scene):
    """The robot's lift (m, the sum of its two stages) in the scene's robot config."""
    text = (ROOT / "env_cfg" / "robot" / f"{scene}.yml").read_text()
    return sum(float(v) for v in re.findall(r"LZ_\w+_Joint:\s*([-\d.eE]+)", text))


def reach(xy, up, out, top, base, lift):
    """Distance (m) from the shoulder on side `out` to a grasp point `up` above the desk top `top` on the cup axis at
    `xy` (layout frame, before the robot base moves by `base`, transform())."""
    pivot = np.r_[ROBOT_ROOT, 0.0]
    point = np.r_[xy, top + up]
    point = pivot + Rotation.from_euler("z", -base["yaw"], degrees=True).apply(point - pivot)
    point -= [base["lateral"], base["distance"], 0.0]
    return float(np.linalg.norm(point - (out * SHOULDER[0], SHOULDER[1], SHOULDER[2] + lift)))


def to_layout(xy, base):
    """A desk point (x, y) in the frame of the robot moved by `base` back in the layout frame (reach() undone)."""
    rel = np.r_[np.add(xy, [base["lateral"], base["distance"]]) - ROBOT_ROOT, 0.0]
    return ROBOT_ROOT + Rotation.from_euler("z", base["yaw"], degrees=True).apply(rel)[:2]


def draw_places(rng, level, ra, rb, ups, top, base, lift, outs=(0.0, 0.0), handles=(False, False), side=None):
    """(side, source xy, target xy), None if none found: side -1 puts the source cup on the robot's left (the twin's),
    given or drawn. `ups`: the source and target cups' grasp heights above the desk, `outs` their grasp points'
    offsets out to the holding arm's side (Cups.grip_out); both grasp points within REACH_MAX of their shoulders. A
    cup held by its handle (`handles`) has its grasp point anywhere in HANDLE_ZONE at every level, the handle out
    along the robot's x (handle_yaw)."""
    side = side or (-1 if level == 0 else rng.choice((-1, 1)))

    def held(xy, up, out, s):
        return reach(np.add(xy, [s * out, 0.0]), up, s, top, base, lift)

    def twin(s, up, out, margin):
        """The twin place on side s, moved toward the robot until a grasp `up` high is `margin` inside the reach."""
        y = TWIN_XY[1]
        while held((s * TWIN_XY[0], y), up, out, s) > REACH_MAX - margin and y > REACH_Y[0]:
            y -= 0.005
        return np.array([s * TWIN_XY[0], y])

    for _ in range(200):
        if level < 2:
            jitter = TWIN_JITTER if level == 0 else NEAR
            xy = [
                twin(s, up, out, jitter) + [rng.uniform(-jitter, jitter) for _ in "xy"]
                for s, up, out in ((side, ups[0], outs[0]), (-side, ups[1], outs[1]))
            ]
        else:
            xy = [np.array([s * rng.uniform(*REACH_X), rng.uniform(*REACH_Y)]) for s in (side, -side)]
            if level == 3:
                edge = rng.choice([[0], [1], [0, 1]])
                for k in edge:
                    s = side if k == 0 else -side
                    if rng.random() < 0.5:
                        xy[k][0] = s * rng.uniform(*EDGE_X)
                    else:
                        xy[k][1] = rng.uniform(*rng.choice(EDGE_Y))
        for k, s in enumerate((side, -side)):
            if handles[k]:
                grip = (s * rng.uniform(*HANDLE_ZONE[0]), rng.uniform(*HANDLE_ZONE[1]))
                xy[k] = to_layout(np.subtract(grip, [s * outs[k], 0.0]), base)
        if (
            xy[0][0] * side > 0 > xy[1][0] * side
            and np.linalg.norm(xy[0] - xy[1]) >= ra + rb + CUP_GAP
            and all(held(p, up, out, s) <= REACH_MAX for p, up, out, s in zip(xy, ups, outs, (side, -side)))
        ):
            return side, xy[0], xy[1]
    return None


def draw_base(rng, level):
    """Robot base offset (lateral, distance, yaw deg), lift and desk change (m)."""
    if level < 2:
        return dict.fromkeys(BASE, 0.0)
    if level == 2:
        return {k: rng.uniform(-low, low) for k, (low, _) in BASE.items()}
    return {k: signed(rng, low, high) for k, (low, high) in BASE.items()}


def bean_count(pack, source, s_scale, target, t_scale, fill, yaw):
    """Beans in the source cup: `fill` of the smaller capacity, the lowest of the pack (turned `yaw`) that fit in the
    source cavity with their heap RIM_FREE below its rim, fewer while they would settle within RIM_FREE of the
    target's rim."""
    g = cup_profile.pack_grain(pack)
    cap = min(cup_profile.capacity(c, s, g) for c, s in ((source, s_scale), (target, t_scale)))
    want = int(min(BEANS_MAX, fill * cap))
    half = cup_profile.grain_half(pack)
    idx, cen = cup_profile.pack_fill(source, s_scale, pack["pose"][:, :3], half, yaw=yaw, count=want)
    below = cen[:, 2] + half[idx] <= source["z"][-1] * s_scale - RIM_FREE
    beans = int(np.argmin(below)) if not below.all() else len(idx)
    rim_b = target["z"][-1] * t_scale - RIM_FREE
    while beans > 50 and cup_profile.fill_height(target, beans, t_scale, g) > rim_b:
        beans = int(beans * 0.95)
    return beans


def capsule_beans(layout, pack, pack_ref):
    """`layout` with its beans as CAPSULE_PHYSICS grains of `pack`: the same kind, yaw and fill share of the smaller
    capacity (the cups' cavities as laid out), the count of them from the pack (bean_count)."""
    out = deepcopy(layout)
    fluid = out["Fluid"]["wuliangye"][0]
    old = fluid["physics"]
    cups = {e["label"]: e for entries in out["Rigid"].values() for e in entries}
    src, tgt = cups["source_cup"], cups["target_cup"]
    record = out["Generalization"]["beans"]
    count = bean_count(pack, src["cavity"], src["scale"][0], tgt["cavity"], tgt["scale"][0], record["fill"], old["yaw"])
    fluid["physics"] = dict(
        CAPSULE_PHYSICS,
        pack=pack_ref,
        cavity=old["cavity"],
        cavity_scale=old["cavity_scale"],
        yaw=old["yaw"],
        count=count,
    )
    record["count"] = count
    return out


def convert_capsule(assets, src, seed):
    """Each scene's layout folder `src` as `seed`: the same layouts (capsule_beans), sets.json and split.json."""
    packs, written = {}, {}
    for folder in sorted((Path(assets) / "Eval_Layout" / "RoboDojo").glob(f"*/{src}")):
        if not (folder / "sets.json").is_file():
            continue
        dst = folder.parent / str(seed)
        dst.mkdir(exist_ok=False)
        files = sorted(folder.glob(f"{TASK}_*.json"))
        for f in files:
            layout = json.loads(f.read_text(encoding="utf-8"))
            kind = layout["Generalization"]["beans"]["kind"]
            if kind not in packs:
                packs[kind] = dict(np.load(ROOT / CAPSULE_PACK.format(kind=kind)))
            out = capsule_beans(layout, packs[kind], "$RoboDojo_HOME/" + CAPSULE_PACK.format(kind=kind))
            (dst / f.name).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        meta = json.loads((folder / "sets.json").read_text(encoding="utf-8")) | {"capsule_of": src}
        (dst / "sets.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        if (folder / "split.json").is_file():
            (dst / "split.json").write_text((folder / "split.json").read_text())
        written[folder.parent.name] = len(files)
    return written


def cup_entry(template, cup_id, cup, label, xy, yaw, scale, top):
    _, section, category, index = cup_id.split("/")
    prof = cup["cavity"]
    rot = Rotation.from_euler("z", yaw, degrees=True)
    pos = np.r_[xy, top + 1e-4] - rot.apply([prof["axis"][0] * scale, prof["axis"][1] * scale, prof["bottom"] * scale])
    x, y, z, w = rot.as_quat()
    entry = deepcopy(template)
    entry.update(
        category=category,
        category_idx=int(index),
        label=label,
        default_pos=[round(float(v), 5) for v in pos],
        default_ori=[float(w), float(x), float(y), float(z)],
        scale=[scale] * 3,
        physics=dict(
            entry["physics"],
            mass=round(cup["mass"] * scale**3, 4),
            size=[round(v * scale, 4) for v in cup["size"]],
            type="rigid",
        ),
        visual={},
        cavity=prof,
    )
    entry.pop("type", None)
    if section == "Clutter":
        entry["type"] = "cluttered"
    return entry


def free_spot(rng, r, taken, cups_xy):
    for _ in range(500):
        x_range, y_range = rng.choice(ZONES)
        x, y = rng.uniform(*x_range), rng.uniform(*y_range)
        inside = TABLE[0][0] + r <= x <= TABLE[0][1] - r and TABLE[1][0] + r <= y <= TABLE[1][1] - r
        clear = all(math.dist((x, y), c) >= r + DISTRACTOR_CLEAR for c in cups_xy)
        apart = all(math.dist((x, y), p) >= r + q + DISTRACTOR_GAP for p, q in taken)
        if inside and clear and apart:
            return x, y
    return None


def name_levels(cup, lang):
    return cup["names"][lang] or [cup["family"].replace("_", " ")]


def pick_name(rng, cup_id, others, cups, lang, start):
    """The least detailed name of the cup from level `start` that none of `others` (ids) is also called."""
    names = name_levels(cups.all[cup_id], lang)
    taken = {n for o in others for n in name_levels(cups.all[o], lang)}
    for name in names[min(start, len(names) - 1) :]:
        if name not in taken:
            return name
    return None


def distractor_cup(rng, cups, pool, a, b, level):
    """A same-kind distractor cup: L2 the family of A or B; L3 a cup sharing A's or B's short name (one attribute
    apart), else the family."""
    ref = rng.choice((a, b))
    fam = [c for c in pool if c not in (a, b) and cups.all[c]["family"] == cups.all[ref]["family"]]
    if level >= 3:
        short = name_levels(cups.all[ref], "zh")[0]
        twins = [c for c in pool if c not in (a, b) and name_levels(cups.all[c], "zh")[0] == short]
        fam = twins or fam
    return rng.choice(fam or [c for c in pool if c not in (a, b)])


def instruction(rng, level, templates, names, beans):
    """({lang: text}, (template set, index)): the same template in both languages."""
    out = {}
    pick = rng.random()
    for lang in ("zh", "en"):
        pool = templates["heldout"][lang] if level == 3 else templates[lang]
        index = 0 if level == 0 else int(pick * len(pool))
        text = pool[index]
        for key, value in (*names[lang].items(), ("<beans>", BEANS[beans]["name"][lang])):
            text = text.replace(key, value)
        out[lang] = text[0].upper() + text[1:] if lang == "en" else text
    return out, ("heldout" if level == 3 else "train", index)


def draw_materials(rng, scene, level):
    """{role: library id} new surface materials: the scene profile's pool (L2) or the groups outside it (L3)."""
    if level < 2:
        return {}
    profile = appearance.SURFACE_POOLS["scenes"][SCENES[scene]]
    allowed = appearance.SURFACE_POOLS["profiles"][profile]
    out = {}
    for role in appearance.ROLES:
        groups = appearance.SURFACE_POOLS["pools"][role]
        names = allowed[role] if level == 2 else [g for g in groups if g not in allowed[role]] or allowed[role]
        if rng.random() < SWAP:
            out[role] = "mdl/" + rng.choice(groups[rng.choice(names)])
    return out


def draw_lights(rng, level):
    """(Light section or None, record) of `level`'s extra lights above the desk."""
    if not level:
        return None, []
    (low, high), power, temp = LIGHTS[level]
    out, record = {}, []
    for n in range(rng.randint(low, high)):
        kind = "Sphere" if level == 1 else rng.choice(sorted(LIGHT_SHAPE))
        pos = [round(rng.uniform(*r), 3) for r in LIGHT_BOX]
        ori = [round(rng.uniform(-LIGHT_TILT, LIGHT_TILT), 1) for _ in range(2)] + [0.0]
        t = round(rng.uniform(*temp))
        intensity = round(LIGHT_REF * rng.uniform(*power), 1)
        shape = {k: [v := round(rng.uniform(*r), 3), v] for k, r in LIGHT_SHAPE[kind].items()}
        out[f"light_{n}"] = {
            "types": kind,
            "common": {
                "initial_pos_range": pos + pos,
                "initial_ori_range": ori + ori,
                "color_temperature_range": [t, t],
            },
            kind: {"intensity_range": [intensity, intensity], **shape},
        }
        record.append({"type": kind, "pos": pos, "intensity": intensity, "temperature": t})
    return out, record


def draw_camera(rng, level):
    """(Camera entry, Imaging entry)."""
    cam = {}
    for name in ("cam_head", *WRISTS) if level >= 2 else ():
        (pos, ori) = EXTRINSIC[level]["head" if name == "cam_head" else "wrist"]
        d = np.array([rng.gauss(0, 1) for _ in range(3)])
        a = np.array([rng.gauss(0, 1) for _ in range(3)])
        cam[name] = {
            "pos": (d / np.linalg.norm(d) * rng.uniform(*pos)).round(4).tolist(),
            "ori": (a / np.linalg.norm(a) * rng.uniform(*ori)).round(2).tolist(),
        }
    if not level:
        return cam, None
    k = IMAGE[level]
    img = {key: round(rng.uniform(1 - k, 1 + k), 3) for key in ("brightness", "contrast", "saturation")}
    if level >= 2:
        img["noise"] = round(rng.uniform(*NOISE[level]), 4)
    if level == 3:
        effect = rng.choice(("blur", "over", "under"))
        if effect == "blur":
            img["blur"] = round(rng.uniform(1.0, 2.5), 2)
        else:
            img["brightness"] = round(rng.uniform(1.35, 1.6) if effect == "over" else rng.uniform(0.45, 0.65), 3)
    img["seed"] = rng.randrange(2**31)
    return cam, img


def transform(layout, base):
    """Move the robot base by `base` (lateral, distance, yaw): the scene and its objects the other way about the
    robot root."""
    yaw = np.deg2rad(-base["yaw"])
    rot = Rotation.from_euler("z", yaw)
    shift = -np.array([base["lateral"], base["distance"], 0.0])
    pivot = np.r_[ROBOT_ROOT, 0.0]

    def move(entry):
        entry["default_pos"] = [
            round(float(v), 5) for v in pivot + rot.apply(np.asarray(entry["default_pos"]) - pivot) + shift
        ]
        w, x, y, z = entry["default_ori"]
        qx, qy, qz, qw = (rot * Rotation.from_quat([x, y, z, w])).as_quat()
        entry["default_ori"] = [float(qw), float(qx), float(qy), float(qz)]

    move(layout["Room"])
    for section in ("Rigid", "Fluid"):
        for group in (layout.get(section) or {}).values():
            for entry in group:
                move(entry)


class Generator:
    def __init__(self, assets, cups, templates):
        self.assets, self.cups, self.templates = assets, cups, templates
        self.scenes = {}
        self.lifts = {scene: lift_of(scene) for scene in SCENES}
        self.packs = {}
        for scene, folder in SCENES.items():
            path = Path(assets) / TEMPLATE.format(scene=scene)
            parts, _, library = appearance.load(ROOT / "real2sim" / "scenes" / folder)
            self.scenes[scene] = (json.loads(path.read_text()), parts, library)
        cup_categories = {c.split("/")[2] for c in cups.all}
        self.clutter = [c for c in (clutter_library(assets) if assets else []) if c[0] not in cup_categories]
        categories = sorted({c for c, _, _ in self.clutter})
        held = set(random.Random(TASK).sample(categories, round(HELD_CLUTTER * len(categories))))
        self.clutter_train = [c for c in self.clutter if c[0] not in held]
        self.clutter_held = [c for c in self.clutter if c[0] in held]

    def materials(self, rng, scene, level):
        return draw_materials(rng, scene, level)

    def pack(self, kind):
        """The kind's grain pack (BEAN_PACK under the assets)."""
        if kind not in self.packs:
            self.packs[kind] = dict(np.load(Path(self.assets) / BEAN_PACK.format(kind=kind)))
        return self.packs[kind]

    def lights(self, rng, level):
        return draw_lights(rng, level)

    def camera(self, rng, level):
        return draw_camera(rng, level)

    def layout(self, key, levels, split, pair=None, scene=None, fixed=(), clutter=None):
        """One layout of `levels`; `pair` (source id, target id, their scales) fixes the cups, `scene` the scene;
        `fixed` (NARROW_FIXED names) keeps those parts at L0 whatever their dimension's level; `clutter` (low, high)
        the number of training distractors whatever the visual level."""
        rng = random.Random(key)
        lv = levels
        scene = scene or (
            HELD_SCENE
            if lv["visual"] == 3
            else TWIN_SCENE
            if all(v == 0 for v in lv.values())
            else rng.choice(TRAIN_SCENES)
        )
        template, parts, library = self.scenes[scene]
        cups = self.cups
        a, b, sa, sb, *extra = pair if pair else draw_cups(rng, cups, lv["object"], split)
        kind, hand, fill_range = (extra + [None, None, None])[:3]
        ca, cb = cups.all[a], cups.all[b]
        mats = {} if "materials" in fixed else self.materials(rng, scene, lv["visual"])
        loose = dict(parts, scene=None) if lv["visual"] == 3 else parts  # held-out materials are off the profile
        ups = [cups.grip_up(c, s, role) for c, s, role in ((a, sa, False), (b, sb, True))]
        outs = [cups.grip_out(c, s, role) for c, s, role in ((a, sa, False), (b, sb, True))]
        handles = [cups.grip(c, role).get("mode") == "handle" for c, role in ((a, False), (b, True))]
        for _ in range(50):  # a base whose cup places are in reach
            base = draw_base(rng, 0 if "base" in fixed else lv["spatial"])
            # the desk: its own height plus the change, less the lift column's (a taller robot sees a lower desk)
            top = parts["top_z"] + base["table"] - base["lift"]
            choice = {**mats, "table_top_z": round(top, 4) if abs(top - parts["top_z"]) > 1e-6 else None}
            room, top = appearance.overrides(loose, choice, library)
            places = draw_places(
                rng,
                lv["spatial"],
                radius(ca, sa),
                radius(cb, sb),
                ups,
                top,
                base,
                self.lifts[scene],
                outs,
                handles,
                hand,
            )
            if places:
                break
        else:
            raise RuntimeError(f"no cup places in reach at spatial L{lv['spatial']}")
        side, xa, xb = places

        layout = {k: deepcopy(v) for k, v in template.items() if k not in ("Rigid", "Fluid")}
        layout["Room"].update(room, table_height=top, appearance_choice=choice)
        cup_template = template["Rigid"]["plastic_cup"][0]
        rigid = layout["Rigid"] = {}
        entries = {}
        pairs = (("source_cup", a, ca, xa, sa, side, handles[0]), ("target_cup", b, cb, xb, sb, -side, handles[1]))
        for label, cid, cup, xy, s, out, by_handle in pairs:
            yaw = handle_yaw(rng, cup, out, lv["spatial"], by_handle, base["yaw"])
            e = cup_entry(cup_template, cid, cup, label, xy, yaw, s, top)
            if cid in cups.grasps:
                e["grasp"] = deepcopy(cups.grasps[cid])
            rigid.setdefault(e["category"], []).append(e)
            entries[label] = e

        # distractors: a same-kind cup for the language, clutter for the visual dimension
        taken, k, record_cups, record_clutter = [], 0, [], []
        others = []
        if lv["language"] >= 2:
            pool = cups.eval if split == "eval" and lv["object"] == 3 else cups.train
            for _ in range(50):
                d = distractor_cup(rng, cups, pool, a, b, lv["language"])
                sd = scale_of(rng, min(lv["object"], 2))
                spot = free_spot(rng, radius(cups.all[d], sd), taken, [xa, xb])
                # the twin pair (A and B the same cup) goes by side: only the distractor must be told apart
                if spot and all(pick_name(rng, c, [o for o in (a, b, d) if o != c], cups, "zh", 0) for c in (a, b)):
                    break
            else:
                raise RuntimeError("no distractor cup the names can tell apart")
            e = cup_entry(cup_template, d, cups.all[d], f"distractor_{k}", spot, rng.uniform(0, 360), sd, top)
            rigid.setdefault(e["category"], []).append(e)
            taken.append((spot, radius(cups.all[d], sd)))
            others.append(d)
            record_cups.append(d)
            k += 1
        if (lv["visual"] >= 2 or clutter is not None) and self.clutter:
            low, high = clutter if clutter is not None else CLUTTER[lv["visual"]]
            library_ = self.clutter_train if lv["visual"] < 3 else self.clutter
            by_category = {}
            for category, index, meta in library_:
                by_category.setdefault(category, []).append((index, meta))
            wanted = rng.randint(low, high)
            chosen = rng.sample(sorted(by_category), min(wanted, len(by_category)))
            if lv["visual"] == 3 and self.clutter_held:
                held = rng.choice(sorted({c for c, _, _ in self.clutter_held}))
                chosen = [held] + [c for c in chosen if c != held][: wanted - 1]
            for category in chosen:
                index, meta = rng.choice(by_category[category])
                circle = meta["active"]["place"]["up"]["projection_circle"]
                spot = free_spot(rng, circle["radius"], taken, [xa, xb])
                if spot is None:
                    continue
                pos, ori = place_pose(circle["center"], spot, top, rng.uniform(0, 360))
                physics = dict(meta.get("physics") or {}, type="rigid")
                e = dict(deepcopy(cup_template), category=category, category_idx=index, type="cluttered",
                         label=f"distractor_{k}", default_pos=pos, default_ori=ori, physics=physics, visual={},
                         scale=[1.0, 1.0, 1.0])  # fmt: skip
                rigid.setdefault(category, []).append(e)
                taken.append((spot, circle["radius"]))
                record_clutter.append(f"{category}/{index}")
                k += 1

        # beans: a share of the smaller capacity, the lowest of the kind's settled pack that fit in the source cup
        # (turned a random yaw), their heap RIM_FREE below its rim and settled RIM_FREE below the target's
        kind = kind or ("red" if lv["object"] == 0 else rng.choice(sorted(BEANS)))
        pack, yaw = self.pack(kind), rng.uniform(0.0, 2 * math.pi)
        fill = rng.uniform(*(fill_range or FILL[lv["object"]]))
        beans = bean_count(pack, ca["cavity"], sa, cb["cavity"], sb, fill, yaw)

        lights, light_record = self.lights(rng, 0 if "lights" in fixed else lv["visual"])
        if lights:
            layout["Light"] = lights
        camera, imaging = self.camera(rng, lv["camera"])
        if camera:
            layout["Camera"] = camera
        if imaging:
            layout["Imaging"] = imaging

        transform(layout, base)
        src = entries["source_cup"]
        fluid = deepcopy(template["Fluid"]["wuliangye"][0])
        bean = BEANS[kind]
        fluid.update(
            default_pos=list(src["default_pos"]),
            default_ori=list(src["default_ori"]),  # GranularObject places the grains in the cup's frame
            scale=[1.0, 1.0, 1.0],
            bean=dict(kind=kind, **bean["name"]),
            physics=dict(
                GRAIN_PHYSICS,
                friction_combine=GRAIN_COMBINE,
                pack=f"$Robodojo_ASSETS/{BEAN_PACK.format(kind=kind)}",
                cavity=ca["cavity"],
                cavity_scale=sa,
                yaw=round(yaw, 4),
                count=beans,
            ),
            visual=dict(visible=True, color=list(bean["color"]), roughness=bean["roughness"]),
        )
        layout["Fluid"] = {"wuliangye": [fluid]}

        # language: names (or sides) of A and B, a template of the level
        lang_level = lv["language"]
        names = {}
        same = a == b or all(name_levels(ca, lang) == name_levels(cb, lang) for lang in ("zh", "en"))
        by_side = same or (lang_level == 1 and not others and rng.random() < SIDE_SHARE)
        start = 0 if lang_level == 0 else rng.randint(0, 1 if lang_level == 1 else 2)
        for lang in ("zh", "en"):
            if by_side:
                noun = pick_name(rng, a, others, cups, lang, start) if others else NOUN[lang]
                left, right = (s.format(noun) for s in SIDE[lang])
                pair = (left, right) if side < 0 else (right, left)
            else:
                pair = tuple(
                    pick_name(rng, cid, [o for o in (a, b, *others) if o != cid], cups, lang, start)
                    or name_levels(cups.all[cid], lang)[-1]
                    for cid in (a, b)
                )
                if lang == "en":
                    pair = tuple(f"the {n}" for n in pair)
            names[lang] = {"<source_cup>": pair[0], "<target_cup>": pair[1]}
        text, template_id = instruction(rng, lang_level, self.templates, names, kind)

        layout["Generalization"] = {
            "split": split,
            "levels": dict(lv),
            "fixed": list(fixed),
            "scene": scene,
            "cups": {"source_cup": a, "target_cup": b, "distractors": record_cups},
            "scale": {"source_cup": round(sa, 3), "target_cup": round(sb, 3)},
            "beans": {"kind": kind, "count": beans, "fill": round(fill, 3)},
            "hand": "left" if side < 0 else "right",
            "base": {k_: round(v, 4) for k_, v in base.items()},
            "materials": mats,
            "lights": light_record,
            "clutter": record_clutter,
            "names": {lang: {k_.strip("<>"): v for k_, v in n.items()} for lang, n in names.items()},
            "by_side": bool(by_side),
            "template": list(template_id),
            "instruction": text,
        }
        return scene, layout, text


def held_scale(cups, cup_id):
    """The cup's scale nearest 1 of PROBE_SCALES that the expert can hold it at in both roles and pour into it; None
    if none."""
    fit = [s for s in PROBE_SCALES if cups.fits(cup_id, s, True) and cups.fits(cup_id, s, False)]
    return min(fit, key=lambda s: abs(s - 1)) if fit else None


def ring(scaled, levels, name, key, offset=0, side=None):
    """A shuffled cycle of (cup, scale): each cup once the source and once the target, the bean kinds in turn; the
    source held in the `side` hand (draw_places) or either."""
    scaled = list(scaled)
    random.Random(key).shuffle(scaled)
    kinds = sorted(BEANS)
    pairs = zip(scaled, scaled[1:] + scaled[:1])
    return [
        (name, levels, (a, b, sa, sb, kinds[(i + offset) % len(kinds)], side))
        for i, ((a, sa), (b, sb)) in enumerate(pairs)
    ]


def mouth_ratio(cups, a, sa, b, sb):
    """Source over target inner rim radius."""
    return cups.all[a]["cavity"]["r"][-1] * sa / (cups.all[b]["cavity"]["r"][-1] * sb)


def sets(mode, count, cups=None, chosen=None, min_ratio=0.0, reps=1):
    """[(set name, levels, cup pair)]: training draws, the benchmark's graded sets and diagnostics (`reps` times over,
    the first time the 325 of the benchmark, then spares for the expert feasibility filter), the probe: a
    cycle of the loadable cups at their held_scale (cups held at none left out), the rest L0 (a failed pair is retried
    against the twin cup to tell which cup failed); or the grasp exam (scripts/datagen/grasp_exam.py): `count` cycles
    of the `chosen` cups, each reshuffled, the source in the left hand in even cycles and the right in odd ones (each
    cup in each role and hand once in two cycles), the places at spatial L1, the rest L0; or the generalization exam:
    `count` random pairs of the `chosen` cups (mode docstring), the source mouth at least `min_ratio` of the target's
    (mouth_ratio)."""
    if mode == "probe":
        scaled = [(c, s) for c in sorted(c for c, v in cups.all.items() if v["loads"]) if (s := held_scale(cups, c))]
        return ring(scaled, dict.fromkeys(DIMS, 0), "probe", f"{TASK}/probe")
    if mode == "exam":
        scaled = [(c, s) for c in chosen if (s := held_scale(cups, c))]
        skipped = sorted(set(chosen) - {c for c, _ in scaled})
        if skipped:
            print(f"no scale the expert holds, left out: {', '.join(skipped)}")
        levels = dict(dict.fromkeys(DIMS, 0), spatial=1)
        return [
            row
            for k in range(count)
            for row in ring(scaled, levels, f"exam_{k}", f"{TASK}/exam/{k}", k, side=(-1, 1)[k % 2])
        ]
    if mode == "gen":
        rng = random.Random(f"{TASK}/gen")
        levels, kinds, out = dict(dict.fromkeys(DIMS, 0), spatial=1), sorted(BEANS), []
        while len(out) < count:
            a, b = rng.sample(sorted(chosen), 2)
            sa, sb = rng.uniform(*GEN_SCALE), rng.uniform(*GEN_SCALE)
            fit = cups.fits(a, sa, False) and cups.fits(b, sb, True)
            if fit and mouth_ratio(cups, a, sa, b, sb) >= min_ratio:
                out.append(("gen", levels, (a, b, sa, sb, rng.choice(kinds), rng.choice((-1, 1)), GEN_FILL)))
        return out
    if mode == "train":
        rng = random.Random(f"{TASK}/train-levels")
        return [("train", {d: rng.choices(range(3), TRAIN_MIX)[0] for d in DIMS}, None) for _ in range(count)]
    if mode == "narrow":
        return [("narrow", dict(NARROW), None) for _ in range(count)]
    out = [(g, {d: int(g[1]) for d in DIMS}, None) for g, n in BENCH.items() for _ in range(n)]
    for d in DIMS:
        for level in (1, 2, 3):
            out += [(f"diag_{d}_L{level}", {e: level if e == d else 0 for e in DIMS}, None)] * DIAG
    return out * reps


def check_stratum(st, cups):
    """A recipe stratum draws training content only (rsi/AGENTS.md): no held-out scene, cup, template or clutter, so
    no level above L2, scales within the training range and fewer distractors than visual L3 puts out."""
    problems = [f"unknown keys {sorted(set(st) - RECIPE_KEYS)}"] if set(st) - RECIPE_KEYS else []
    if not 0 < st["target"] <= st["candidates"]:
        problems.append("0 < target <= candidates")
    if any(len(c) != len(DIMS) or not set(c) <= {0, 1, 2} for c in st.get("combos", ())):
        problems.append(f"combos: one level of 0-2 per dimension {DIMS}")
    weights = st.get("levels", {})
    if set(weights) - set(DIMS) or any(len(w) != 3 for w in weights.values()):
        problems.append(f"levels: weights of L0-L2 per dimension of {DIMS}")
    if set(st.get("scenes", ())) - set(TRAIN_SCENES):
        problems.append(f"scenes: {TRAIN_SCENES} only")
    pools = st.get("cups", {})
    if set(pools) - {"source", "target"} or any(set(ids) - set(cups.train) for ids in pools.values()):
        problems.append("cups: source / target pools of training cups")
    ranges = st.get("scale", {})
    if set(ranges) - {"source", "target"} or not all(
        SCALE[2][0] <= lo <= hi <= SCALE[2][1] for lo, hi in ranges.values()
    ):
        problems.append(f"scale: source / target ranges within {SCALE[2]}")
    if st.get("same", False) not in (True, False) or (st.get("same") and "target" in ranges):
        problems.append("same: true or false, without a target scale range (the target is the source)")
    low, high = st.get("clutter", (0, 0))
    if not 0 <= low <= high < CLUTTER[3][0]:
        problems.append(f"clutter: 0 <= low <= high < {CLUTTER[3][0]}")
    if problems:
        raise SystemExit(f"recipe stratum {st.get('name')}: " + "; ".join(problems))


def recipe_pair(rng, cups, st):
    """(source id, target id, source scale, target scale) from the stratum's pools and ranges (object L2 otherwise);
    `same`: one cup at one scale for both, so only their places tell them apart."""
    pools, ranges, same = st.get("cups", {}), st.get("scale", {}), st.get("same", False)
    for _ in range(500):
        a, b = (rng.choice(pools.get(role, cups.train)) for role in ("source", "target"))
        sa, sb = (rng.uniform(*ranges[role]) if role in ranges else scale_of(rng, 2) for role in ("source", "target"))
        if same:
            b, sb = a, sa
        if (a == b) == same and cups.fits(a, sa, False) and cups.fits(b, sb, True):
            return a, b, sa, sb
    raise RuntimeError(f"no holdable cup pair in recipe stratum {st['name']}")


def recipe_sets(recipe, cups, seed):
    """[(stratum index, name, levels, cup pair, scene, clutter)]: each stratum's `candidates` layouts, the levels its
    `combos` in turn or drawn with its per-dimension `levels` weights (TRAIN_MIX otherwise), the cups from its `cups`
    pools or `scale` ranges at object L2, the scene one of its `scenes` (TRAIN_SCENES)."""
    out = []
    for k, st in enumerate(recipe["strata"]):
        check_stratum(st, cups)
        rng = random.Random(f"{TASK}/recipe/{seed}/{st['name']}")
        for i in range(st["candidates"]):
            if st.get("combos"):
                levels = dict(zip(DIMS, st["combos"][i % len(st["combos"])]))
            else:
                levels = {d: rng.choices(range(3), st.get("levels", {}).get(d, TRAIN_MIX))[0] for d in DIMS}
            pair = None
            if {"cups", "same", "scale"} & set(st):
                levels["object"] = 2
                pair = recipe_pair(rng, cups, st)
            scene = rng.choice(st.get("scenes", TRAIN_SCENES))
            out.append((k, st["name"], levels, pair, scene, st.get("clutter")))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "bench", "probe", "narrow", "recipe", "exam", "gen", "capsule"])
    parser.add_argument("--assets", type=Path, required=True, help="Assets with the scene templates and Clutter")
    parser.add_argument("--seed", type=int, required=True, help="seed folder of the layouts")
    parser.add_argument("--count", type=int, default=1500, help="train, narrow, gen: layouts; exam: cycles")
    parser.add_argument("--reps", type=int, default=1, help="bench: times over the 325 layouts")
    parser.add_argument("--cups", type=Path, help="exam, gen: JSON list of cup ids, or a feasibility file")
    parser.add_argument("--grasps", type=Path, default=GRASPS, help="per-cup grasp entries (default: the task's)")
    parser.add_argument("--depth", type=float, default=DEPTH, help="exam, gen: shallowest cavity (m)")
    parser.add_argument("--min-ratio", type=float, default=0.0, help="gen: least source / target mouth radius")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="language of the layouts' Instruction")
    parser.add_argument("--random-seed", type=int, default=20261103)
    parser.add_argument("--scene", choices=sorted(SCENES), help="narrow: the one scene (required)")
    parser.add_argument("--recipe", type=Path, help="recipe: strata JSON (scripts/datagen/recipes/)")
    parser.add_argument("--src", type=int, help="capsule: the seed folder converted into --seed")
    args = parser.parse_args(argv)
    if (args.mode == "capsule") != (args.src is not None):
        parser.error("--src goes with capsule, and capsule needs it")
    if args.mode == "capsule":
        written = convert_capsule(args.assets, args.src, args.seed)
        print(", ".join(f"{scene}/{args.seed}: {n}" for scene, n in written.items()))
        return
    if (args.mode == "narrow") != bool(args.scene):
        parser.error("--scene goes with narrow, and narrow needs it")
    if (args.mode == "recipe") != bool(args.recipe):
        parser.error("--recipe goes with recipe, and recipe needs it")
    if args.depth != DEPTH and args.mode not in ("exam", "gen"):
        parser.error("--depth is for the exams: the task's cups keep DEPTH")
    grasps = json.loads(args.grasps.read_text(encoding="utf-8"))
    cups = Cups(json.loads(POOL.read_text(encoding="utf-8")), grasps, args.depth)
    chosen = None
    if args.mode in ("exam", "gen"):
        if args.cups is None:
            parser.error(f"{args.mode} needs --cups")
        listed = json.loads(args.cups.read_text(encoding="utf-8"))
        chosen = listed["feasible"] if isinstance(listed, dict) else listed
        unknown = [c for c in chosen if c not in cups.all or not cups.all[c]["loads"]]
        if unknown:
            parser.error(f"not loadable cups of {POOL}: {', '.join(unknown)}")
    gen = Generator(args.assets, cups, json.loads(INSTRUCTIONS.read_text(encoding="utf-8")))
    split = "eval" if args.mode == "bench" else "train"
    fixed = NARROW_FIXED if args.mode == "narrow" else ()
    if args.mode == "recipe":
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        rows = recipe_sets(recipe, cups, args.random_seed)
    else:
        rows = [
            (0, n, levels, pair, args.scene, None)
            for n, levels, pair in sets(
                args.mode, args.count, cups, reps=args.reps, chosen=chosen, min_ratio=args.min_ratio
            )
        ]
    written, index = {}, {}
    for i, (k, set_name, levels, pair, scene, clutter) in enumerate(rows):
        key = f"{args.random_seed}/{args.mode}/{i}"
        scene, layout, text = gen.layout(key, levels, split, pair, scene=scene, fixed=fixed, clutter=clutter)
        layout["Generalization"]["set"] = set_name
        layout["Instruction"] = text[args.lang]
        where = (scene, args.seed + k)  # recipe: stratum k in seed folder --seed + k
        folder = args.assets / "Eval_Layout" / "RoboDojo" / scene / str(where[1])
        if where not in written:
            folder.mkdir(parents=True, exist_ok=False)
            written[where], index[where] = 0, []
        name = f"{TASK}_{written[where]}.json"
        (folder / name).write_text(json.dumps(layout, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        index[where].append({"file": name, "index": i, "set": set_name, "levels": levels})
        written[where] += 1
    plan = []
    for (scene, seed), rows_ in index.items():
        folder = args.assets / "Eval_Layout" / "RoboDojo" / scene / str(seed)
        meta = {"task": TASK, "split": split, "mode": args.mode, "random_seed": args.random_seed, "layouts": rows_}
        if args.mode == "recipe":  # the stratum's target shared over its scenes' folders by their candidates
            st = recipe["strata"][seed - args.seed]
            target = math.ceil(st["target"] * len(rows_) / st["candidates"])
            meta |= {"recipe": recipe.get("name"), "plugin": recipe.get("plugin"), "stratum": st["name"]}
            meta["target"] = target
            plan.append(
                {"scene": scene, "seed": seed, "stratum": st["name"], "candidates": len(rows_), "target": target}
            )
        (folder / "sets.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        if split == "train":  # the evaluation client refuses a folder marked train
            (folder / "split.json").write_text(json.dumps({"split": "train", "task": TASK}) + "\n")
    print(json.dumps(plan, indent=1) if plan else ", ".join(f"{s}/{seed}: {n}" for (s, seed), n in written.items()))


if __name__ == "__main__":
    main()
