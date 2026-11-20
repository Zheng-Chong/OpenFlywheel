"""g1d_place_anything (把 X 放到 Y 上) and g1d_put_anything_in (把 X 放进 Y 里): the receptacles Y, which X fits
which Y, the layouts' Rigid section, the instructions and the acceptance geometry. X is g1d_pick_anything's: its
pool, target order, names and grip yaw (scripts/objects/pick_anything.py). Stdlib only (the platform imports it).

A receptacle (label "receptacle") is a support (KINDS[0]: plate, tray, book ...) or a container (KINDS[1]: bowl, basket,
box, cup ...) of RECEPTACLES (place_anything_receptacles.json, written by `pool` from the height maps and spill levels
scripts/objects/check_cup_holds_beans.py measures of its colliders). Its geometry, root frame, scale 1, standing:
    region   {"center": [x, y], "half": [hx, hy], "radius": r}: a support's face (a tray's or rimmed plate's floor) or a
             container's opening, as a rectangle along the object's axes and the largest circle inside it
    floor    z of the support face, or of the cavity's floor
    rim      z of a container's spill level (a support: its floor)
    top      z of its highest point; bottom: of its underside
X fits Y when its footprint, turned as it is carried (the jaw holds it at its lying yaw), lies inside the region MARGIN
in from its edge, and for a container its centre stays below the rim standing on the floor.

    python scripts/objects/place_anything.py pool --checks <check_cup_holds_beans --out dir> ...   # rewrite RECEPTACLES
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.objects import pick_anything

RECEPTACLES = Path(__file__).with_name("place_anything_receptacles.json")
KINDS = ("support", "container")
TASKS = {"support": "g1d_place_anything", "container": "g1d_put_anything_in"}
LABEL = "receptacle"
MARGIN = 0.01  # m between X's footprint and the region's edge
CENTRE_SLACK = 0.005  # m below the rim X's centre stands at least, on a container's floor
EVAL_SHARE = 0.15  # receptacles of each kind (at least one, the lowest id hashes) only the benchmarks see
DISTRACTORS = (2, 4)  # others besides X and Y (pick_anything's 3-5 crowd the desk once Y takes its middle)
TARGET_GAP, GAP = pick_anything.TARGET_GAP, pick_anything.GAP
# m, deg: where Y stands (env frame of the g1d desk, as pick_anything's areas): in the middle, in both hands' reach
# (block_atomic's tray at (0, -0.42)); turned within `yaw` of either way along X
SPEC = dict(pick_anything.SPEC, receptacle_x=0.05, receptacle_y=(-0.44, -0.40), receptacle_yaw=15.0)
SPEC["other_receptacle"] = 0.0
# acceptance: X's bottom this far from the face (place); X tilted at most this off how it lay (place); Y moved, tilted
FACE_BELOW, FACE_ABOVE, X_TILT, Y_STAY, Y_TILT = 0.01, 0.015, 20.0, 0.03, 10.0
# the pool: a tray or rimmed plate shallower than SUPPORT_DEPTH is also a support (X stands on its floor); a region is
# usable when half its narrow side or its circle is MIN_REGION (the pool's narrowest X is 1.5 cm); a receptacle stands
# within STAND_TILT on its own
SUPPORT_DEPTH, MIN_REGION, STAND_TILT = 0.035, 0.03, 5.0
SUPPORT_FAMILIES = {"plate", "tray", "book", "board", "mark", "stationery", "box", "dish"}
CONTAINER_FAMILIES = {"bowl", "basket", "box", "bin", "container", "cup", "mug", "pan", "pot", "bucket", "jar", "tray"}
TEMPLATES = {
    "support": {
        "zh": ("把{x}放到{y}上", "将{x}放在{y}上面", "把{x}拿起来放到{y}上", "请把桌上的{x}放到{y}上"),
        "en": (
            "Put the {x} on the {y}.",
            "Place the {x} on top of the {y}.",
            "Pick up the {x} and set it on the {y}.",
            "Move the {x} onto the {y}.",
        ),  # fmt: skip
    },
    "container": {
        "zh": ("把{x}放进{y}里", "将{x}放到{y}里面", "把{x}装进{y}", "拿起{x}放进{y}里"),
        "en": (
            "Put the {x} into the {y}.",
            "Place the {x} in the {y}.",
            "Drop the {x} into the {y}.",
            "Pick up the {x} and put it in the {y}.",
        ),  # fmt: skip
    },
}


def load_receptacles(path=RECEPTACLES):
    return json.loads(Path(path).read_text())["receptacles"]


def id_hash(oid):
    return int(hashlib.sha256(oid.encode()).hexdigest()[:8], 16)


def splits(receptacles):
    """{kind: {id: "train" | "eval"}}: per kind the EVAL_SHARE (at least one) with the lowest id hashes are eval."""
    out = {}
    for kind in KINDS:
        ids = sorted((r["id"] for r in receptacles if kind in r["kinds"]), key=id_hash)
        n = max(1, round(EVAL_SHARE * len(ids))) if ids else 0
        out[kind] = {oid: "eval" if i < n else "train" for i, oid in enumerate(ids)}
    return out


def of_kind(receptacles, kind, split="train"):
    """The receptacles of `kind` in `split`, but those the screening took out of training."""
    return [
        r
        for r in receptacles
        if kind in r["kinds"]
        and r["split"][kind] == split
        and (split != "train" or kind not in r.get("screened_out", {}))
    ]


# --- planar geometry: a footprint is (centre x, centre y, half x, half y, yaw rad) ---


def corners(rect):
    x, y, hx, hy, a = rect
    c, s = math.cos(a), math.sin(a)
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def apart(r1, r2, gap):
    """Are footprints r1 and r2 at least `gap` apart (separating axis test on both grown by gap / 2)?"""
    grown = [(x, y, hx + gap / 2, hy + gap / 2, a) for x, y, hx, hy, a in (r1, r2)]
    pts = [corners(r) for r in grown]
    for r in grown:
        for ax in ((math.cos(r[4]), math.sin(r[4])), (-math.sin(r[4]), math.cos(r[4]))):
            p1 = [ax[0] * x + ax[1] * y for x, y in pts[0]]
            p2 = [ax[0] * x + ax[1] * y for x, y in pts[1]]
            if max(p1) < min(p2) or max(p2) < min(p1):
                return True
    return False


def turned_half(w, d, yaw):
    """Half extents along the axes of a w x d rectangle turned by `yaw` (rad)."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (c * w + s * d) / 2, (s * w + c * d) / 2


def spots(geom, size, rel_yaw, scale=1.0):
    """Where X's centre may go in the receptacle's frame (at its `scale`): ((x0, x1), (y0, y1)) or None. X's footprint
    `size` (w, d) turned `rel_yaw` (rad) off the receptacle's, MARGIN inside the region's rectangle, else the region's
    centre when X's circumcircle fits its circle."""
    reg = geom["region"]
    cx, cy = (v * scale for v in reg["center"])
    hx, hy = turned_half(size[0], size[1], rel_yaw)
    ax, ay = (v * scale - h - MARGIN for v, h in zip(reg["half"], (hx, hy)))
    if ax >= 0 and ay >= 0:
        return (cx - ax, cx + ax), (cy - ay, cy + ay)
    if math.hypot(*size[:2]) / 2 + MARGIN <= reg["radius"] * scale:
        return (cx, cx), (cy, cy)
    return None


def deep_enough(geom, height, kind, scale=1.0):
    """A container: X (`height` tall) standing on its floor has its centre CENTRE_SLACK below its rim."""
    return kind != "container" or height / 2 + CENTRE_SLACK <= (geom["rim"] - geom["floor"]) * scale


def fits(geom, size, kind, rel_yaw, scale=1.0):
    return deep_enough(geom, size[2], kind, scale) and spots(geom, size, rel_yaw, scale) is not None


def fits_some_yaw(geom, size, kind, scale=1.0):
    return any(fits(geom, size, kind, math.radians(a), scale) for a in range(0, 180, 15))


def spot_toward(geom, size, rel_yaw, toward, scale=1.0):
    """X's centre in the receptacle's frame: the allowed spot nearest `toward` (where X comes from, in that frame),
    so the hand on X's side reaches less far across; the region's centre when nothing is allowed."""
    allowed = spots(geom, size, rel_yaw, scale)
    if allowed is None:
        return [v * scale for v in geom["region"]["center"]]
    return [min(max(t, lo), hi) for t, (lo, hi) in zip(toward, allowed)]


# --- the layouts ---


def x_order(pool, receptacles, kind):
    """pick_anything.pool_order without the pool objects no training receptacle of `kind` takes."""
    train = of_kind(receptacles, kind)
    geoms = [r["geometry"][kind] for r in train]
    return [i for i in pick_anything.pool_order(pool) if any(fits_some_yaw(g, pool[i]["size"], kind) for g in geoms)]


def footprint(obj, xy, yaw_deg, scale=1.0):
    w, d = obj["size"][:2]
    return (xy[0], xy[1], w * scale / 2, d * scale / 2, math.radians(yaw_deg))


def layout(rng, index, entry, receptacle_entry, kind, pool, receptacles, spec=SPEC, order=None):
    """Rigid section of layout `index` and what was drawn: X (label target, pool object order[index mod len]), a
    training receptacle of `kind` it fits (label receptacle), DISTRACTORS pool objects (distractor_<j>) and with
    probability spec["other_receptacle"] one more training receptacle (same family first) among them.
    entry(cat, idx, label, xy, yaw[, scale]) and receptacle_entry(rec, label, xy, yaw) make the entries."""
    order = order or x_order(pool, receptacles, kind)
    target = pool[order[index % len(order)]]
    s_x = spec["scale"](rng, target) if spec["scale"] else 1.0

    def taking(scale):
        size = [v * scale for v in target["size"]]
        return size, [r for r in of_kind(receptacles, kind) if fits_some_yaw(r["geometry"][kind], size, kind)]

    size, choices = taking(s_x)
    if not choices:  # scaled up past every receptacle: unscaled, some take it (x_order)
        s_x = 1.0
        size, choices = taking(s_x)
    rec = rng.choice(choices)
    others = [o for o in pool if o["family"] == target["family"] and o is not target]
    count = rng.randint(*DISTRACTORS)
    picked = []
    if others and rng.random() < spec["same_family"]:
        picked = rng.sample(others, min(len(others), rng.choice((1, 2))))
    families = {target["family"]}
    for o in rng.sample(pool, len(pool)):
        if len(picked) >= count:
            break
        if o["family"] not in families:
            picked.append(o)
            families.add(o["family"])
    scales = [spec["scale"](rng, o) if spec["scale"] else 1.0 for o in picked]
    second = None
    if rng.random() < spec["other_receptacle"]:
        pool_y = [r for r in of_kind(receptacles, kind) if r["id"] != rec["id"]]
        same = [r for r in pool_y if r["family"] == rec["family"]]
        second = rng.choice(same or pool_y) if pool_y else None
    for attempt in range(400):
        placed = place(rng, target, size, rec, kind, picked, scales, second, spec)
        if placed:
            break
        if attempt % 40 == 39:  # crowded: no second receptacle, fewer distractors, then another receptacle
            if second is not None:
                second = None
            elif len(picked) > DISTRACTORS[0]:
                picked.pop()
                scales.pop()
            else:
                rec = rng.choice(choices)
    else:
        raise RuntimeError(f"no room for {TASKS[kind]} layout {index}")
    (x_xy, x_yaw), (y_xy, y_yaw), spots_, second_spot = placed
    rigid = {}

    def add(cat, e):
        rigid.setdefault(cat, []).append(e)

    add(target["cat"], entry(target["cat"], target["idx"], "target", x_xy, x_yaw, *([s_x] if spec["scale"] else [])))
    add(rec["cat"], receptacle_entry(rec, LABEL, y_xy, y_yaw))
    j = 0
    for obj, s, (xy, yaw) in zip(picked, scales, spots_):
        add(obj["cat"], entry(obj["cat"], obj["idx"], f"distractor_{j}", xy, yaw, *([s] if spec["scale"] else [])))
        j += 1
    if second is not None:
        add(second["cat"], receptacle_entry(second, f"distractor_{j}", *second_spot))
    drawn = dict(receptacle=rec["id"], kind=kind, other_receptacle=second["id"] if second else None, x_scale=s_x)
    return rigid, drawn


def place(rng, target, size, rec, kind, others, scales, second, spec):
    """((X xy, yaw deg), (Y xy, yaw deg), [(xy, yaw) of others], (xy, yaw) of the second receptacle) or None."""
    rw, rd = rec["size"][:2]
    y_xy = (rng.uniform(-spec["receptacle_x"], spec["receptacle_x"]), rng.uniform(*spec["receptacle_y"]))
    y_yaw = rng.uniform(-spec["receptacle_yaw"], spec["receptacle_yaw"]) + 180.0 * rng.randrange(2)
    y_rect = (*y_xy, rw / 2, rd / 2, math.radians(y_yaw))
    x_xy = None
    for _ in range(100):
        xy = (rng.uniform(spec["mid_gap"], spec["target_x"]) * rng.choice((-1, 1)), rng.uniform(*spec["target_y"]))
        yaw = pick_anything.target_yaw(rng, target, xy, spec["grasp_yaw"])
        rect = (*xy, size[0] / 2, size[1] / 2, math.radians(yaw))
        if apart(rect, y_rect, TARGET_GAP) and fits(rec["geometry"][kind], size, kind, math.radians(yaw - y_yaw)):
            x_xy, x_yaw, x_rect = xy, yaw, rect
            break
    if x_xy is None:
        return None
    placed = [(x_rect, TARGET_GAP), (y_rect, GAP)]
    out = []
    items = list(zip(others, scales)) + ([(second, 1.0)] if second is not None else [])
    for obj, s in items:
        for _ in range(300):
            xy = (rng.uniform(*pick_anything.SPREAD_X), rng.uniform(*pick_anything.SPREAD_Y))
            yaw = rng.uniform(0.0, 360.0)
            rect = footprint(obj, xy, yaw, s)
            if all(apart(rect, q, g) for q, g in placed):
                break
        else:
            return None
        placed.append((rect, GAP))
        out.append((xy, yaw))
    second_spot = out.pop() if second is not None else None
    return (x_xy, x_yaw), (y_xy, y_yaw), out, second_spot


# --- instructions ---


def instruction(rng, kind, names):
    """One instruction: a language, usable names of X and Y, a template. `names`: {lang: {"x": [...], "y": [...]}}."""
    lang = rng.choice(sorted(names))
    template = rng.choice(TEMPLATES[kind][lang])
    return template.format(x=rng.choice(names[lang]["x"]), y=rng.choice(names[lang]["y"]))


# --- acceptance geometry (poses: xyz and wxyz quaternions, env frame) ---


def rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx)


def inverse(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    return (w / n, -x / n, -y / n, -z / n)


def local(pos, quat, point):
    """`point` in the frame of the body at pos, quat."""
    return rotate(inverse(quat), [p - c for p, c in zip(point, pos)])


def tilt_deg(quat, ref=(1.0, 0.0, 0.0, 0.0)):
    """deg between the body's Z axis now and at orientation `ref`."""
    a, b = rotate(quat, (0, 0, 1)), rotate(ref, (0, 0, 1))
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(u * v for u, v in zip(a, b))))))


def bottom_z(pos, quat, size):
    return min(
        pos[2] + rotate(quat, (sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2))[2]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    )


def in_region(geom, xy, scale=1.0, slack=0.0):
    reg = geom["region"]
    dx, dy = (v - c * scale for v, c in zip(xy, reg["center"]))
    inside = abs(dx) <= reg["half"][0] * scale + slack and abs(dy) <= reg["half"][1] * scale + slack
    return inside or math.hypot(dx, dy) <= reg["radius"] * scale + slack


def placed(kind, geom, scale, x_pos, x_quat, x_size, y_pos, y_quat, x_quat0=None):
    """Is X (pos, quat, extents at its scale) on the support / in the container Y (pos, quat, scale, geometry)? A
    support: X's centre over its face and X's bottom FACE_BELOW .. FACE_ABOVE about it, X within X_TILT of how it lay
    (x_quat0); a container: X's centre over its opening, below its rim and above its floor."""
    p = local(y_pos, y_quat, x_pos)
    if not in_region(geom, p[:2], scale):
        return False
    if kind == "container":
        return geom["floor"] * scale - FACE_BELOW <= p[2] <= geom["rim"] * scale
    face = y_pos[2] + rotate(y_quat, (0.0, 0.0, geom["floor"] * scale))[2]
    gap = bottom_z(x_pos, x_quat, x_size) - face
    return -FACE_BELOW <= gap <= FACE_ABOVE and (x_quat0 is None or tilt_deg(x_quat, x_quat0) <= X_TILT)


def kept(y_pos, y_quat, y_start):
    """Y not pushed off (Y_STAY) or tipped (Y_TILT)."""
    return math.dist(y_pos[:2], y_start[:2]) < Y_STAY and tilt_deg(y_quat) <= Y_TILT


# --- the receptacle pool ---


def _region(mask, xs, ys):
    """{"center", "half", "radius"} of a boolean cell mask (numpy) over cells xs x ys: the largest axis-aligned
    rectangle about the mask's centroid with the mask's aspect inside it, and the largest circle inside the mask."""
    import numpy as np
    from scipy import ndimage

    cell = float(xs[1] - xs[0])
    ii, jj = np.nonzero(mask)
    ci, cj = ii.mean(), jj.mean()
    ext = np.array([ii.max() - ii.min() + 1, jj.max() - jj.min() + 1]) / 2
    lo, hi = 0.0, 1.0
    for _ in range(20):
        a = (lo + hi) / 2
        i0, i1 = int(math.floor(ci - a * ext[0])), int(math.ceil(ci + a * ext[0]))
        j0, j1 = int(math.floor(cj - a * ext[1])), int(math.ceil(cj + a * ext[1]))
        ok = 0 <= i0 and 0 <= j0 and i1 < mask.shape[0] and j1 < mask.shape[1] and mask[i0 : i1 + 1, j0 : j1 + 1].all()
        lo, hi = (a, hi) if ok else (lo, a)
    dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    k = np.unravel_index(np.argmax(dist), dist.shape)
    centre = [float(np.interp(ci, np.arange(len(xs)), xs)), float(np.interp(cj, np.arange(len(ys)), ys))]
    half = [max(0.0, float(lo * ext[0] * cell - cell / 2)), max(0.0, float(lo * ext[1] * cell - cell / 2))]
    circle = dict(at=[float(xs[k[0]]), float(ys[k[1]])], radius=float(dist[k] * cell - cell / 2))
    if math.dist(circle["at"], centre) > 0.01:  # the circle elsewhere than the rectangle: keep the one about the centre
        circle["radius"] = min(circle["radius"], min(half) if min(half) > 0 else 0.0)
    return dict(
        center=[round(v, 4) for v in centre], half=[round(v, 4) for v in half], radius=round(circle["radius"], 4)
    )


def geometry(npz, flat_tol=0.004, min_depth=0.015):
    """Receptacle geometry of a cavity/<n>.npz (check_cup_holds_beans.py): as a container (its cavity below the spill
    level, min_depth deep) and as a support (the cavity's floor when shallower than SUPPORT_DEPTH, else the top face:
    cells within flat_tol of the highest), each None if there is none."""
    import numpy as np
    from scipy import ndimage

    d = np.load(npz)
    xs, ys, height, level, cavity = d["xs"], d["ys"], d["height"], d["level"], d["cavity"]
    bottom, top = -float(d["root_z"]), float(np.nanmax(height))
    out = {}
    if cavity.any():
        labels, n = ndimage.label(cavity)
        basin = labels == 1 + int(np.argmax(ndimage.sum(cavity, labels, range(1, n + 1))))
        rim = float(np.median(level[basin]))
        floor = float(np.nanpercentile(height[basin], 10))
        if rim - floor >= min_depth:
            out["container"] = dict(region=_region(basin, xs, ys), floor=round(floor, 4), rim=round(rim, 4))
        if rim - floor <= SUPPORT_DEPTH:
            # the floor a tray or rimmed plate holds X on: its cells within flat_tol of its median
            face = basin & (np.abs(height - np.median(height[basin])) <= flat_tol)
            if face.any():
                z = float(np.median(height[face]))
                out["support"] = dict(region=_region(face, xs, ys), floor=round(z, 4), rim=round(z, 4))
    if "support" not in out and "container" not in out:
        face = ~np.isnan(height) & (height >= top - flat_tol)
        labels, n = ndimage.label(face)
        if n:
            face = labels == 1 + int(np.argmax(ndimage.sum(face, labels, range(1, n + 1))))
            z = float(np.median(height[face]))
            out["support"] = dict(region=_region(face, xs, ys), floor=round(z, 4), rim=round(z, 4))
    for g in out.values():
        g.update(top=round(top, 4), bottom=round(bottom, 4))
    return out


# candidates: Rigid objects of the commercially usable libraries (not GenieSim, CC-BY-NC-SA-4.0, or PartNet-Mobility)
# with a size, a family name of the kind and a size the desk and the hands take (m: footprint narrow side, long side
# and height; a support is flat). Sand3D's collider is one convex hull, closing any cavity: supports only. Only paper
# stacks of the stationery and marker boxes of the boxes are supports
NONCOMMERCIAL = pick_anything.NONCOMMERCIAL
CANDIDATE_RULES = dict(
    support=dict(narrow=(0.065, 0.24), long=0.32, height=(0.0, 0.05), families=SUPPORT_FAMILIES),
    container=dict(narrow=(0.07, 0.24), long=0.32, height=(0.025, 0.20), families=CONTAINER_FAMILIES),
)


def license_of(lib, cat):
    if cat.startswith("simready_"):
        return "CC-BY-4.0 (NVIDIA SimReady)"
    if cat.startswith("sand3d_"):
        return "CC-BY-4.0 (Sand3D)"
    if lib == "FMB":
        return "CC-BY-4.0 (FMB)"
    if lib in ("Real2Sim", "Blocks"):
        return "OpenFlywheel (own models)"
    return "MIT (RoboDojo repository)"


def candidates(assets):
    """Receptacle candidates of the library at `assets` ([{rel, fam, sup, con, size, mass, names, license}])."""
    out = []
    for meta_path in sorted(Path(assets).glob("Object/*/Rigid/*/[0-9]*/metadata.json")):
        folder = meta_path.parent
        lib, cat = folder.parents[2].name, folder.parent.name
        if cat.startswith(NONCOMMERCIAL) or not (folder / "description.json").is_file():
            continue
        names = json.loads((folder / "description.json").read_text()).get("names") or {}
        fam = (names.get("category") or {}).get("en")
        meta = json.loads(meta_path.read_text())
        physics = meta.get("physics") or {}
        # RoboDojo's own objects give only the mesh's box (the check measures the standing size)
        size = physics.get("size") or ((meta.get("geometry") or {}).get("aligned_bbox") or {}).get("extents")
        if not fam or not size or min(size) <= 0:
            continue
        mass = physics.get("mass")
        en = " ".join(names.get("en") or [])
        ok = {}
        for kind, r in CANDIDATE_RULES.items():
            narrow, h = min(size[:2]), size[2]
            ok[kind] = (
                fam in r["families"]
                and r["narrow"][0] <= narrow <= r["narrow"][1]
                and max(size[:2]) <= r["long"]
                and r["height"][0] <= h <= r["height"][1]
            )
        ok["support"] &= not (fam == "stationery" and "paper" not in en) and not (
            fam == "box" and "marker box" not in en
        )
        ok["support"] &= not (fam == "board" and lib == "FMB")  # peg and assembly boards: holes and slots
        ok["container"] &= not cat.startswith("sand3d_")
        if ok["support"] or ok["container"]:
            rel = f"{lib}/Rigid/{cat}/{folder.name}"
            out.append(dict(rel=rel, fam=fam, sup=ok["support"], con=ok["container"], size=size, mass=mass,
                            names=names, license=license_of(lib, cat)))  # fmt: skip
    return out


def build_pool(checks, candidates):
    """The receptacle pool from check_cup_holds_beans.py output dirs and the candidates
    [{rel, fam, sup, con, size, mass, name, names, library, license}]: each candidate that loaded, stands
    (STAND_TILT) and has a region of a kind its family is, with its geometry and split; the rest under "rejected"."""
    by = {c["rel"]: c for c in candidates}
    found = {}
    for d in map(Path, checks):
        metrics = json.loads((d / "metrics.json").read_text())
        for n, (rel, r) in enumerate(metrics["cups"].items()):
            found[rel] = (r, d / "cavity" / f"{n:03d}.npz")
        for rel, err in metrics.get("failed_load", {}).items():
            found.setdefault(rel, ({"error": err}, None))
    out, rejected = [], {}
    for rel, c in sorted(by.items()):
        if rel not in found:
            rejected[rel] = "not checked"
            continue
        r, npz = found[rel]
        if npz is None:
            rejected[rel] = r["error"][:200]
            continue
        if r["stand_tilt_deg"] > STAND_TILT:
            rejected[rel] = f"leans {r['stand_tilt_deg']} deg standing alone"
            continue
        geo = geometry(npz)
        kinds = {}
        for kind, ok in (("support", c["sup"]), ("container", c["con"])):
            g = geo.get(kind)
            if ok and g and max(min(g["region"]["half"]), g["region"]["radius"]) >= MIN_REGION:
                kinds[kind] = g
        if not kinds:
            rejected[rel] = f"no region ({sorted(geo)}; check {json.dumps(r)[:160]})"
            continue
        lib, _, cat, idx = rel.split("/")
        out.append(
            dict(
                id=rel,
                library=lib,
                cat=cat,
                idx=int(idx),
                family=c["fam"],
                names=c["names"],
                license=c["license"],
                size=[round(v, 4) for v in r["size_m"]],
                mass=c["mass"] if c.get("mass") else None,
                mass_estimated=not c.get("mass"),
                kinds=sorted(kinds),
                geometry=kinds,
                stand_tilt_deg=r["stand_tilt_deg"],
                cavity_depth_mm=r["cavity_depth_mm"],
            )
        )
    by_kind = splits(out)
    for r in out:
        r["split"] = {kind: by_kind[kind][r["id"]] for kind in r["kinds"]}
    return out, rejected


def screen_stats(sidecars, pool):
    """{"receptacle" | "receptacle_family" | "x_family" | "kind": {key: [accepted, recorded]}} of episode sidecars."""
    family = {(o["cat"], o["idx"]): o["family"] for o in pool}
    out = {k: {} for k in ("kind", "receptacle", "receptacle_family", "x_family")}
    for path in sidecars:
        q = json.loads(Path(path).read_text())
        rec, (cat, idx) = q.get("receptacle") or {}, q["objects"]["target"]
        rid = rec.get("id")
        keys = dict(kind=q.get("kind"), receptacle=rid, receptacle_family=rid and rid.split("/")[2],
                    x_family=family.get((cat, int(idx))))  # fmt: skip
        for axis, key in keys.items():
            cell = out[axis].setdefault(key, [0, 0])
            cell[0] += bool(q.get("accepted"))
            cell[1] += 1
    return out


def screen_out(receptacles, stats, kind, low, least, family_least=None):
    """Mark the receptacles of `kind` accepted less than `low` of at least `least` screening episodes, and every one
    of an asset category accepted less than `low` of at least `family_least` (each member drawn only a few times)."""
    marked = []
    for r in receptacles:
        if kind not in r["kinds"]:
            continue
        acc, n = stats["receptacle"].get(r["id"], (0, 0))
        f_acc, f_n = stats["receptacle_family"].get(r["cat"], (0, 0))
        if n >= least and acc / n < low:
            why = f"{acc}/{n} accepted in the screening"
        elif family_least and f_n >= family_least and f_acc / f_n < low:
            why = f"its category {r['cat']} {f_acc}/{f_n} accepted in the screening"
        else:
            continue
        r.setdefault("screened_out", {})[kind] = why
        marked.append(r["id"])
    return marked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="receptacle candidates of an asset library (JSON to stdout)")
    c.add_argument("--assets", type=Path, required=True)
    p = sub.add_parser("pool", help="rewrite the receptacle pool from check_cup_holds_beans.py output")
    p.add_argument("--checks", nargs="+", required=True, help="its --out dirs")
    p.add_argument("--candidates", type=Path, required=True, help="the candidates' JSON list")
    p.add_argument("--release", type=int, required=True, help="the asset library revision the check ran on")
    q = sub.add_parser("screen", help="accept rates of screening runs; --low marks receptacles screened_out")
    q.add_argument("--kind", choices=KINDS, required=True)
    q.add_argument("--sidecars", nargs="+", required=True, help="raw/seed*/episode_*.json")
    q.add_argument("--low", type=float, default=0.0, help="screen out receptacles accepted less than this")
    q.add_argument("--least", type=int, default=3, help="... of at least this many episodes")
    q.add_argument("--family-least", type=int, default=6, help="... or its category of at least this many")
    q.add_argument("--json", type=Path, help="write the rates here")
    args = parser.parse_args()
    if args.cmd == "screen":
        stats = screen_stats(args.sidecars, pick_anything.load_pool())
        for axis in ("kind", "receptacle_family", "x_family"):
            rows = sorted(stats[axis].items(), key=lambda kv: kv[1][0] / kv[1][1])
            print(axis + ": " + ", ".join(f"{k} {a}/{n} ({a / n:.0%})" for k, (a, n) in rows))
        if args.json:
            args.json.write_text(json.dumps(stats, indent=1) + "\n")
        if args.low:
            data = json.loads(RECEPTACLES.read_text())
            marked = screen_out(data["receptacles"], stats, args.kind, args.low, args.least, args.family_least)
            RECEPTACLES.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
            print(f"screened out {len(marked)}: {marked}")
        return
    if args.cmd == "candidates":
        print(json.dumps(candidates(args.assets), indent=1, ensure_ascii=False))
        return
    pool, rejected = build_pool(args.checks, json.loads(args.candidates.read_text()))
    RECEPTACLES.write_text(
        json.dumps(
            dict(release=args.release, margin=MARGIN, receptacles=pool, rejected=rejected), indent=1, ensure_ascii=False
        )
        + "\n"  # fmt: skip
    )
    for kind in KINDS:
        n = {s: len(of_kind(pool, kind, s)) for s in ("train", "eval")}
        print(f"{kind}: {n['train']} train, {n['eval']} eval")
    print(f"rejected {len(rejected)}")


if __name__ == "__main__":
    main()
