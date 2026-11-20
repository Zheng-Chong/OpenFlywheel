"""g1d_open_anything (打开任意铰接物体): the part pool, the layouts and the instruction names.

A pool entry is one openable part (a door, drawer or lid joint, scripts/objects/SKILL.md「铰接物体的关节作用」) of an
Articulation object of the asset library, pinned to one library release (POOL: open_anything_pool.json, committed, so
the layouts do not change with the library). `pool` reads each object's USD (pxr, numpy) at its authored, closed pose
and writes, in the object's own frame (centred on its bounding box, as the loader places it):

    pivot, axis      the joint's axis through its anchor (prismatic: the slide direction), from the parent's side
    sign, closed     the joint value it is closed at and which way opens it (toward the limit further away)
    front            horizontal unit vector the part opens toward (a drawer's pull, a door's first swing), the side
                     the robot stands on; layouts turn the object so it faces the robot
    grasp            the grip point on the part's handle: the protrusion of the part's collision geometry beyond its
                     front face (handles, knobs, a lid's overhanging lip); close: the axis the jaw closes along (the
                     handle's thin side), width: the handle along it, depth: how far it stands out
    names            the object's names (description.json) and the part's, with left / right / top / bottom when the
                     object has more than one part of that role

Parts are left out (EXCLUDED, with the reason) when they have no handle the Dex1-1 jaw can take (narrower than
JAW_MAX, standing out at least MIN_DEPTH), when the object does not fit the desk or the grip point is out of reach
(REACH), or when the feasibility screen (`screen`) found the scripted expert could not open them. Every entry has a
`split`: train, or eval for the held-out objects (EVAL_SHARE of each family, by a hash of the object id; `screen`
redoes it among the parts it keeps, per asset category), so training layouts never show an eval object.

Layout i of a seed folder opens TRAIN_ORDER[i mod n] (one fixed shuffle of the train entries); the object stands on the
desk with its front toward the robot, turned within YAW, its grip point within GRIP_X / GRIP_Y.

    python scripts/objects/open_anything.py pool --assets <library copy> --objects <find --motion openable output> \
        --release 27
    python scripts/objects/open_anything.py screen-layouts --folder Assets/Eval_Layout/RoboDojo/g1d/<seed> --trials 3
    python scripts/objects/open_anything.py screen --layouts <that folder> --raw <run_batch output>/raw
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

POOL = Path(__file__).with_name("open_anything_pool.json")
ROLES = ("door", "drawer", "lid")
# metres: what the Dex1-1 jaw (95 mm open, scripted_demo.pad_open adds GRASP_CLEAR 25 mm) closes on, and how far a
# handle must stand out of its part's face for the pads to hold it: the fingertips reach 23 mm past the pads' middle,
# so on a shallower handle they stop on the face before the pads close on it (screen 11-18: of 46 handles standing
# out 10-20 mm the jaw held none)
JAW_MAX, MIN_DEPTH = 0.065, 0.022
# the three ways the jaw takes a part (METHODS): a handle standing out at least MIN_DEPTH is pinched from the front; a
# bar (HOOK_BAR x longer than wide) standing out at least MIN_KNOB with HOOK_GAP of room behind it is hooked; else the
# part's free edge, its front panel's last EDGE_SPAN, is pinched across the panel EDGE_IN from the edge where
# EDGE_GAP behind the panel is free (a lid's front lip the same way, top and bottom)
METHODS = ("pinch", "side", "hook", "edge")
MIN_KNOB, HOOK_GAP, HOOK_BAR = 0.008, 0.014, 2.5
# a lying bar standing out SIDE_DEPTH but not MIN_DEPTH is pinched from the side: the open jaw comes in level along
# the bar from its end, the pads closing top and bottom across it next to the face (a standing bar would need the
# fingers pointing down onto it from above, which the arm cannot reach at a cabinet's handle height: screen8 reached
# 4 of 55)
SIDE_DEPTH = 0.012
EDGE_PANEL, EDGE_SPAN, EDGE_IN, EDGE_GAP, EDGE_PATH = 0.03, 0.01, 0.012, 0.015, 0.06
# a flap's lip: its front-most LIP, LIP_SPAN either side of its middle, with no base within LIP_GAP under it (the
# lower finger goes there)
LIP, LIP_SPAN, LIP_GAP = 0.015, 0.04, 0.02
SAMPLE, MAX_SAMPLES = 0.004, 400_000  # m between surface samples of a mesh; most samples a mesh gets
# the desk of the g1d office (real_scene1, build_params.json "table"): x half-length, y of its near and far edges in the
# env frame (robot root at y -0.73, facing +y); objects may stand OVERHANG past the far edge
DESK_X, DESK_NEAR, DESK_FAR, OVERHANG = 0.56, -0.60, -0.015, 0.06
# grip point in the env frame, where layouts put it; and (REACH_Y, REACH_Z: height above the desk) where the pool
# looks for parts at all, wider: the feasibility screen keeps what the expert opens
GRIP_X, GRIP_Y = (-0.20, 0.20), (-0.42, -0.30)
REACH_Y, REACH_Z = (-0.44, -0.24), (0.02, 0.45)
# env y the object (closed) and the part at OPEN_OVER x its goal open keep in front of: the hands rest about 0.5 m
# ahead of the robot root and its chest is at about -0.62; an object in them threw the robot over at the start, a
# door swinging into it pushed it (screen 11-18)
CLEAR_Y, SWEEP_Y, OPEN_OVER = -0.46, -0.60, 1.3
YAW = 15.0  # deg the object's front turns off facing the robot
# m the object's lowest point is put above the desk: a bottom drawer or a drop-down door flush with the base's
# underside, set into the desk by the loader's 1 mm, was thrown open at the start (partnet_storagefurniture/00083)
DESK_GAP = 0.006
# opening the acceptance asks for: revolute parts (door, lid) by angle, drawers by travel, at most this share of the
# joint's range
OPEN_DEG, OPEN_M, OPEN_SHARE = 40.0, 0.08, 0.6
EVAL_SHARE = 0.15
# what a layout carries of its pool entry ("open"): the task, the expert and the plugin need nothing else
LAYOUT_KEYS = (
    *("id", "joint", "role", "part", "type", "child", "pivot", "axis", "closed", "sign", "travel", "goal", "front"),
    *("flap", "grasp", "close", "width", "depth", "handle", "names", "part_name", "sole", "siblings", "split"),
    *("sources", "method", "approach", "gap"),
)
LIBRARY_LICENCES = {
    "artvip": "Apache-2.0 (ArtVIP, X-Humanoid)",
    "partnet": "SAPIEN PartNet-Mobility terms of use: non-commercial research and educational use only",
    "robocasa": "CC-BY-4.0 (NVIDIA PhysicalAI Manipulation Objects Kitchen; RoboCasa365; fixtures by Lightwheel)",
    "geniesim": "CC-BY-NC-SA-4.0 (GenieSim)",
}


def noncommercial(licence):
    """Does the licence forbid commercial use (CC-*-NC-*, the PartNet-Mobility terms)? Such objects stay out of the
    pool: the dataset has to stay usable commercially (user, 11-18)."""
    text = licence.lower().replace("_", "-")
    return "-nc" in text or "non-commercial" in text or "noncommercial" in text


PART_NAMES = {  # part -> (zh, en) for the instruction
    "door": ("门", "door"),
    "drawer": ("抽屉", "drawer"),
    "basket": ("炸篮", "basket"),
    "tray": ("抽屉", "drawer"),
    "lid": ("盖子", "lid"),
    "flap": ("翻盖", "flap"),
    "cover": ("盖子", "cover"),
    "screen": ("屏幕", "screen"),
    "flipframe": ("翻门", "flap door"),
}
TEMPLATES = {
    "door": {
        "zh": ("打开{obj}的{part}", "把{obj}的{part}打开", "请打开{obj}的{part}"),
        "en": ("Open the {part} of the {obj}.", "Open the {obj}'s {part}.", "Swing open the {part} of the {obj}."),
    },
    "drawer": {
        "zh": ("拉开{obj}的{part}", "把{obj}的{part}拉出来", "打开{obj}的{part}"),
        "en": ("Pull out the {part} of the {obj}.", "Open the {obj}'s {part}.", "Pull the {obj}'s {part} open."),
    },
    "lid": {
        "zh": ("打开{obj}的{part}", "掀开{obj}的{part}", "把{obj}的{part}掀起来"),
        "en": ("Open the {part} of the {obj}.", "Lift the {obj}'s {part}.", "Flip open the {part} of the {obj}."),
    },
}
# the object alone, when it has one openable part
SOLE = {"zh": ("打开{obj}", "把{obj}打开"), "en": ("Open the {obj}.",)}
ROW_WORDS = {  # rows top to bottom, by how many there are
    2: (("上面的", "top"), ("下面的", "bottom")),
    3: (("最上面的", "top"), ("中间的", "middle"), ("最下面的", "bottom")),
}
COL_WORDS = {
    2: (("左边的", "left"), ("右边的", "right")),
    3: (("左边的", "left"), ("中间的", "middle"), ("右边的", "right")),
}
ORDINAL_EN = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


# --------------------------------------------------------------------------------------------- reading the USD


def analyze(obj_dir):
    """The openable joints of the Articulation object at `obj_dir` (metadata.json body_joints with an opening role),
    each with its geometry in the object frame (module docstring); joints without a usable handle carry `reject`."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    obj_dir = Path(obj_dir)
    meta = json.loads((obj_dir / "metadata.json").read_text())
    usd = next(obj_dir / f for f in ("object.usdz", "object.usd", "object.usda") if (obj_dir / f).exists())
    stage = Usd.Stage.Open(str(usd))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    world = lambda prim: np.array(cache.GetLocalToWorldTransform(prim), dtype=float)  # noqa: E731
    walk = lambda prim: Usd.PrimRange(prim, Usd.TraverseInstanceProxies())  # noqa: E731

    def points(prim, colliders):
        """World points on the surfaces of the meshes of the link `prim` (not of links nested in it): its colliders,
        or every mesh. Surfaces, not vertices: a low-poly bar handle has vertices only at its ends."""
        out = []
        for q in walk(prim):
            if q != prim and q.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if not q.IsA(UsdGeom.Mesh) or (colliders and not q.HasAPI(UsdPhysics.CollisionAPI)):
                continue
            mesh = UsdGeom.Mesh(q)
            v = np.array(mesh.GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
            if not len(v):
                continue
            v = (np.c_[v, np.ones(len(v))] @ world(q))[:, :3]
            out += [v, _surface_samples(v, mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())]
        return np.concatenate(out) if out else np.zeros((0, 3))

    def link_points(prim):
        pts = points(prim, True)
        return pts if len(pts) >= 8 else points(prim, False)

    links = [p for p in walk(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    every = np.concatenate([link_points(p) for p in links] or [np.zeros((0, 3))])
    lo, hi = every.min(0), every.max(0)  # the object frame: the loader puts its origin at the layout position
    roles = {j["name"]: j for j in meta.get("body_joints") or []}
    joints = {p.GetName(): p for p in walk(root) if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)}
    out = []
    for name, info in roles.items():
        if info.get("role") not in ROLES or name not in joints:
            continue
        prim = joints[name]
        joint = UsdPhysics.Joint(prim)
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        typed = UsdPhysics.RevoluteJoint(prim) if revolute else UsdPhysics.PrismaticJoint(prim)
        body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        m0, m1 = world(body0), world(body1)
        # rows: the joint frames' axes in the object frame (USD's row-vector convention), without the links' scale
        frame0 = _unit_rows(_rot(joint.GetLocalRot0Attr().Get()) @ m0[:3, :3])
        frame1 = _unit_rows(_rot(joint.GetLocalRot1Attr().Get()) @ m1[:3, :3])
        axis = frame0[["X", "Y", "Z"].index(typed.GetAxisAttr().Get())]
        axis = axis / np.linalg.norm(axis)
        pivot = (np.r_[joint.GetLocalPos0Attr().Get(), 1.0] @ m0)[:3]
        anchor1 = (np.r_[joint.GetLocalPos1Attr().Get(), 1.0] @ m1)[:3]
        lower, upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
        scale = math.radians(1.0) if revolute else 1.0
        # the joint value at the authored pose: frame1 against frame0 about (along) the axis
        if revolute:
            q_auth = _angle_about(frame0, frame1, axis)
        else:
            q_auth = float((anchor1 - pivot) @ axis)
        if lower is None or upper is None or not (math.isfinite(lower) and math.isfinite(upper)):
            kind = "revolute" if revolute else "prismatic"
            out.append(
                dict(joint=name, role=info["role"], type=kind, closed=round(q_auth, 5), reject="no joint limits")
            )
            continue
        lower, upper = lower * scale, upper * scale
        sign = 1.0 if abs(upper - q_auth) >= abs(lower - q_auth) else -1.0
        travel = (upper - q_auth) if sign > 0 else (q_auth - lower)
        entry = dict(
            joint=name,
            role=info["role"],
            part=info.get("part", info["role"]),
            type="revolute" if revolute else "prismatic",
            child=body1.GetName(),
            pivot=_r(pivot),
            axis=_r(axis),
            closed=round(q_auth, 5),
            sign=sign,
            travel=round(travel, 5),
            level=bool(revolute and abs(axis[2]) <= 0.7),  # hinged about a level axis: gravity opens or shuts it
            drive=(info.get("drive") or {}).get("type"),
        )
        child = link_points(body1)
        base = np.concatenate([link_points(p) for p in links if p != body1] or [np.zeros((0, 3))])
        if len(child) < 4:
            out.append(dict(entry, reject="no geometry on the part"))
            continue
        moved = _move(child, entry, pivot, axis, sign * (math.radians(5) if revolute else 0.02))
        shift = (moved - child).mean(0)
        # a lid lying flat first rises: it faces the side away from its hinge; a door (an oven's drop-down one too)
        # and a drawer face where they first move to
        flap = revolute and abs(axis[2]) <= 0.7 and abs(shift[2]) > np.linalg.norm(shift[:2])
        front = child.mean(0) - pivot if flap else shift
        front[2] = 0.0
        if np.linalg.norm(front) < 1e-6:
            out.append(dict(entry, reject="part does not open sideways"))
            continue
        front = front / np.linalg.norm(front)
        grip = _handle(child, base, front, flap, pivot, axis, revolute and not flap)
        centre = (child.min(0) + child.max(0)) / 2
        entry.update(front=_r(front), flap=bool(flap), centre=_r(centre))
        entry.update(part_lo=_r(child.min(0)), part_hi=_r(child.max(0)))
        entry.update({k: v for k, v in grip.items() if k != "grasp"})
        if "grasp" in grip:
            entry["grasp"] = _r(grip["grasp"])
        out.append(entry)
    digest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in (obj_dir / "metadata.json", usd)}
    return dict(lo=_r(lo), hi=_r(hi), joints=out, sources=_sources(usd), sha256=digest)


def _sources(usd):
    """Folders of the files the object's USD references outside its own folder, relative to the library root (the
    folder holding Assets/): asset_client.py pull does not follow a reference with a space in it."""
    import re

    text = usd.read_bytes().decode("utf-8", "ignore") if usd.suffix != ".usdz" else ""
    root = usd.parents[6]  # <root>/Assets/Object/<library>/Articulation/<cat>/<idx>/object.usd
    out = set()
    for ref in re.findall(r"@([^@]+)@", text):
        path = (usd.parent / ref).resolve() if not ref.startswith("/") else Path(ref)
        try:
            rel = path.relative_to(root.resolve())
        except ValueError:
            continue
        if rel.parts[0] == "sources":
            out.add(str(rel.parent))
    return sorted(out)


def _surface_samples(v, counts, indices, spacing=SAMPLE):
    """Points spread over a mesh's faces (fan-triangulated), about one per `spacing` squared, the same every run."""
    import numpy as np

    if counts is None or indices is None or not len(counts):
        return np.zeros((0, 3))
    counts, indices = np.asarray(counts), np.asarray(indices)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    tri = []
    for c in np.unique(counts[counts >= 3]):
        first = starts[counts == c][:, None]  # fan: (v0, v_i, v_i+1) for i = 1 .. c-2
        tri.append(np.c_[indices[first.repeat(c - 2, 1)].ravel(), indices[(first + np.arange(1, c - 1)).ravel()],
                         indices[(first + np.arange(2, c)).ravel()]])  # fmt: skip
    if not tri:
        return np.zeros((0, 3))
    tri = np.concatenate(tri)
    tri = tri[(tri < len(v)).all(1)]
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.minimum(np.ceil(area / spacing**2), 400).astype(int)
    if n.sum() > MAX_SAMPLES:
        n = np.ceil(n * MAX_SAMPLES / n.sum()).astype(int)
    rng = np.random.default_rng(0)
    which = np.repeat(np.arange(len(tri)), n)
    r1, r2 = rng.random(len(which)), rng.random(len(which))
    s1 = np.sqrt(r1)
    return (1 - s1)[:, None] * a[which] + (s1 * (1 - r2))[:, None] * b[which] + (s1 * r2)[:, None] * c[which]


def _rot(q):
    """3x3 of a USD quaternion in USD's row-vector convention (p' = p @ M)."""
    import numpy as np
    from pxr import Gf

    m = Gf.Matrix4d().SetRotate(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))
    return np.array(m, dtype=float)[:3, :3]


def _unit_rows(m):
    import numpy as np

    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _angle_about(frame0, frame1, axis):
    """Signed angle (rad) about `axis` from joint frame 0 to joint frame 1 (rows: their axes)."""
    import numpy as np

    rel = frame1.T @ frame0  # column convention: maps a vector in frame 0's place to frame 1's
    perp = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    perp /= np.linalg.norm(perp)
    turned = rel @ perp
    return float(math.atan2(np.cross(perp, turned) @ axis, perp @ turned))


def _move(pts, entry, pivot, axis, dq):
    """Points of the part moved by dq (rad or m) about / along the axis."""
    import numpy as np

    if entry["type"] == "prismatic":
        return pts + dq * axis
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + math.sin(dq) * k + (1 - math.cos(dq)) * k @ k
    return (pts - pivot) @ r.T + pivot


def _handle(child, base, front, flap, pivot, axis, door):
    """How the jaw takes the part (module docstring, METHODS) and where: {method, grasp, approach, close, width,
    depth, gap, handle} or {reject}. A lid lying flat: its front lip pinched top and bottom (edge). Else the blob of
    the part's geometry standing out of its front face (a handle, a knob), the largest or, on a door, the one
    furthest from the hinge: pinched from the front when it stands out MIN_DEPTH (pinch); else, if it is a bar with
    room behind it for the closed fingers, hooked from behind (hook); else the part's free edge (a door's edge away
    from the hinge, a drawer front's top edge) pinched across the panel where nothing is behind it (edge)."""
    import numpy as np

    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(-front, up)  # the robot's right, facing the part
    f = child @ front
    fmax = f.max()
    if flap:
        lip = child[f >= fmax - LIP]
        mid = (lip @ side).mean()
        lip = lip[np.abs(lip @ side - mid) <= LIP_SPAN]
        under = lip[:, 2].min()
        near = (base @ front >= fmax - 2 * LIP) & (np.abs(base @ side - mid) <= LIP_SPAN)
        if np.any(near & (base[:, 2] < under) & (base[:, 2] > under - LIP_GAP)):
            return dict(reject="lid has no free lip (the base is right under it)")
        thick = float(np.ptp(lip[:, 2]))
        if thick > JAW_MAX:
            return dict(reject=f"lip {thick * 1000:.0f} mm thick (jaw {JAW_MAX * 1000:.0f})")
        point = np.r_[0.0, 0.0, (lip[:, 2].min() + lip[:, 2].max()) / 2]
        point += (fmax - LIP / 2) * front + mid * side
        return dict(method="edge", grasp=point, approach=_r(-front), close=_r(up), width=round(thick, 4),
                    depth=LIP, gap=0.0, handle="lip")  # fmt: skip
    # occupancy of 1 cm cells across the face, of the slab within d of the front-most point
    cell = np.floor(np.c_[child @ side, child[:, 2]] / 0.01).astype(int)
    key = cell[:, 0] * 100003 + cell[:, 1]
    full = len(np.unique(key))
    depth = 0.0
    for d in np.arange(0.004, 0.08, 0.002):
        if len(np.unique(key[f >= fmax - d])) > 0.35 * full:
            break
        depth = d
    reasons = []
    if depth >= MIN_KNOB:
        sel = f >= fmax - depth
        blobs = _blobs({tuple(c) for c in cell[sel]})
        # handle-like blobs first: a raised panel or moulding also stands out, but the jaw spans neither of its sides
        narrow = [b for b in blobs if min(np.ptp([c[0] for c in b]), np.ptp([c[1] for c in b])) * 0.01 < JAW_MAX]
        blobs = narrow or blobs
        best = max(blobs, key=len)
        if door:
            big = [b for b in blobs if len(b) >= 0.3 * len(best)]
            best = max(big, key=lambda b: _axis_distance(np.mean(list(b), 0) * 0.01, pivot, axis, side))
        keep = sel & np.isin(key, [c[0] * 100003 + c[1] for c in best])
        hu, hv = child[keep] @ side, child[keep, 2]
        width_u, width_v = np.ptp(hu) + 0.01, np.ptp(hv) + 0.01
        upright = width_u <= width_v  # a bar standing up: the jaw closes across it sideways
        close = side if upright else up
        width, length = float(min(width_u, width_v)), float(max(width_u, width_v))
        face = fmax - depth
        centre = (hu.min() + hu.max()) / 2 * side + np.r_[0.0, 0.0, (hv.min() + hv.max()) / 2]
        # room behind the bar: per cell of the blob, how far its nearest point stands off the face (posts reach it)
        stand = [f[keep & (key == c[0] * 100003 + c[1])].min() - face for c in best]
        gap = float(np.percentile(stand, 60)) if stand else 0.0
        if width > JAW_MAX:
            reasons.append(f"handle {width * 1000:.0f} mm wide")
        elif depth >= MIN_DEPTH:
            return dict(method="pinch", grasp=(fmax - depth / 2) * front + centre, approach=_r(-front),
                        close=_r(close), width=round(width, 4), depth=round(float(depth), 4), gap=round(gap, 4),
                        handle="protrusion")  # fmt: skip
        else:
            reasons.append(f"handle stands out {depth * 1000:.0f} mm")
        if gap >= HOOK_GAP and length >= HOOK_BAR * width:
            # the closed fingers slide into the room behind the bar across it, along the face, then pull outward:
            # down into a lying bar from above, sideways into a standing one from the side away from the hinge
            into = -up if not upright else (side if door and (centre - np.asarray(pivot)) @ side < 0 else -side)
            return dict(method="hook", grasp=(face + gap / 2) * front + centre, approach=_r(into),
                        close=_r(up if upright else side), width=round(width, 4), depth=round(float(depth), 4),
                        gap=round(gap, 4), handle="bar")  # fmt: skip
        if SIDE_DEPTH <= depth and width <= JAW_MAX and length >= HOOK_BAR * width and not upright:
            # too shallow to pinch from the front (the fingertips would hit the face first): the open jaw comes in
            # level from the bar's end, and closes across it next to the face
            into = side if door and (centre - np.asarray(pivot)) @ side < 0 else -side
            grasp = (face + depth / 2) * front + centre
            return dict(method="side", grasp=grasp, approach=_r(into), close=_r(close), width=round(width, 4),
                        depth=round(float(depth), 4), gap=round(gap, 4), handle="bar")  # fmt: skip
        reasons.append(f"{gap * 1000:.0f} mm behind it" + (" (standing bar)" if upright else ""))
    else:
        reasons.append(f"no handle standing out {MIN_KNOB * 1000:.0f} mm (found {depth * 1000:.0f})")
    edge = _free_edge(child, base, front, side, up, fmax, door, pivot)
    if edge is not None:
        return edge
    return dict(reject="; ".join(reasons) + "; no free edge")


def _free_edge(child, base, front, side, up, fmax, door, pivot):
    """The part's free edge with EDGE_GAP of nothing behind the panel there (a door's edge away from the hinge, a
    drawer front's top edge): pinched across the panel, the fingers coming along the face toward the edge. Or None."""
    import numpy as np

    f = child @ front
    panel = child[f >= fmax - EDGE_PANEL]  # the front panel
    t = float(np.ptp(panel @ front)) + 0.002
    if t > JAW_MAX - 0.01:
        return None
    if door:
        out = side if (panel.mean(0) - np.asarray(pivot)) @ side > 0 else -side  # from the hinge to the free edge
        u = panel @ out
        rim = panel[u >= u.max() - EDGE_SPAN]
        into = -out
        mid_v = (rim[:, 2].min() + rim[:, 2].max()) / 2
        point = (rim @ out).max() * out + mid_v * up - EDGE_IN * out
    else:
        rim = panel[panel[:, 2] >= panel[:, 2].max() - EDGE_SPAN]
        into = -up
        mid_u = (rim @ side).mean()
        point = mid_u * side + (rim[:, 2].max() - EDGE_IN) * up
    point = point + ((panel @ front).max() - t / 2) * front
    # nothing of the base or the part's own body behind the panel, around the pinch: room for the far finger
    behind = np.concatenate([base, child[f < fmax - t - 0.002]])
    rel = behind - point
    near = (np.abs(rel @ front + t / 2 + EDGE_GAP / 2) <= EDGE_GAP / 2) & (np.abs(rel @ np.cross(front, into)) <= 0.02)
    near &= (rel @ into >= -EDGE_PATH) & (rel @ into <= 0.025)  # and the way in for the far finger, over the edge
    if np.any(near):
        return None
    return dict(method="edge", grasp=point, approach=_r(into), close=_r(front), width=round(t, 4), depth=EDGE_IN,
                gap=EDGE_GAP, handle="edge")  # fmt: skip


def _axis_distance(uv, pivot, axis, side):
    """Distance of a face point (side, z) from the hinge axis, across the face."""
    import numpy as np

    p = uv[0] * side + np.array([0.0, 0.0, uv[1]])
    q = np.asarray(pivot) * [1, 1, 0] @ side * side + np.array([0.0, 0.0, pivot[2]])
    d = p - q
    return float(np.linalg.norm(d - (d @ axis) * axis))


def _blobs(cells):
    """4-connected groups of 2D cells."""
    left, out = set(cells), []
    while left:
        stack, blob = [left.pop()], set()
        while stack:
            c = stack.pop()
            blob.add(c)
            for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
                if n in left:
                    left.remove(n)
                    stack.append(n)
        out.append(blob)
    return out


def _r(v, n=4):
    return [round(float(x), n) for x in v]


# ------------------------------------------------------------------------------------------------- the pool


def open_goal(entry):
    """Opening (rad or m past closed) the acceptance asks for."""
    want = math.radians(OPEN_DEG) if entry["type"] == "revolute" else OPEN_M
    return round(min(want, OPEN_SHARE * entry["travel"]), 5)


def split_of(oid, family):
    return "eval" if _rank(oid) < EVAL_SHARE else "train"


def _rank(oid):
    return int(hashlib.sha256(f"g1d_open_anything/{oid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def resplit(entries):
    """Hold out whole objects among those with a part the screen kept: per family (asset category) the
    ceil(EVAL_SHARE x n) objects first by hash, when the family has at least two; every other entry train."""
    families = {}
    for e in entries:
        if "screen" in e and not e.get("screened_out"):
            families.setdefault(e["cat"], set()).add(e["id"])
    held = set()
    for ids in families.values():
        if len(ids) >= 2:
            held |= set(sorted(ids, key=_rank)[: math.ceil(EVAL_SHARE * len(ids))])
    for e in entries:
        e["split"] = "eval" if e["id"] in held else "train"


def part_names(entries):
    """For the openable parts of one object, {joint: (zh, en)}: the part's word, with its column (left / middle /
    right) and row (top / middle / bottom, or counted from the top) among the parts of its role, by their centres."""
    out = {}
    for role in ROLES:
        group = [e for e in entries if e["role"] == role]
        for e in group:
            out[e["joint"]] = PART_NAMES.get(e["part"], PART_NAMES[role])
        if len(group) < 2:
            continue
        front = group[0]["front"]
        side = (-front[1], front[0])  # the robot's right, facing the parts: (-front) x up
        cols = _ranks([e["centre"][0] * side[0] + e["centre"][1] * side[1] for e in group], 0.05)
        rows = _ranks([-e["centre"][2] for e in group], 0.04)
        ncol, nrow = max(cols) + 1, max(rows) + 1
        for e, c, r in zip(group, cols, rows):
            zh, en = out[e["joint"]]
            col = COL_WORDS[ncol][c] if ncol in COL_WORDS else (f"从左数第{c + 1}个", f"{ORDINAL_EN[c]}-from-left")
            if nrow in ROW_WORDS:
                row = ROW_WORDS[nrow][r]
            elif r in (0, nrow - 1):
                row = ROW_WORDS[3][0 if r == 0 else 2]
            else:
                row = (f"从上数第{r + 1}个", f"{ORDINAL_EN[r]}")
            words = [w for w in (col if ncol > 1 else None, row if nrow > 1 else None) if w]
            zh_words = "".join(w[0].removesuffix("的") for w in words)
            en_words = " ".join(w[1] for w in reversed(words))
            out[e["joint"]] = (zh_words + ("" if zh_words.endswith("个") else "的") + zh, f"{en_words} {en}")
    return out


def _ranks(values, tol):
    """Rank of each value among distinct levels (values within `tol` share one)."""
    levels = []
    for x in sorted(values):
        if not levels or x - levels[-1][-1] > tol:
            levels.append([x])
        else:
            levels[-1].append(x)
    return [next(i for i, lv in enumerate(levels) if lv[0] <= x <= lv[-1]) for x in values]


def object_names(desc):
    """{lang: [category, simple .. detailed]} from description.json, without the names that call the object open."""
    names = desc.get("names") or {}
    out = {}
    for lang, word in (("zh", "打开"), ("en", "open")):
        levels = [n for n in names.get(lang) or [] if word not in n.lower().split() and word not in n]
        cat = (names.get("category") or {}).get(lang)
        out[lang] = [n for n in [cat, *levels] if n]
    return out


def reachable(obj, entry):
    """Why a part is out of the desk's reach, else None: turned to face the robot (within YAW), the object must fit
    the desk with its grip point in REACH_Y and GRIP_X, at a height in REACH_Z."""
    face = _face_yaw(entry["front"])
    for yaw in (face - YAW, face, face + YAW):
        spots = _spots(dict(obj, **entry), yaw, REACH_Y)
        if spots is None:
            deep = obj["hi"][1] - obj["lo"][1]
            return f"no spot on the desk with the grip in reach and the robot clear ({deep:.2f} m deep)"
    height = entry["grasp"][2] - obj["lo"][2]
    if not REACH_Z[0] <= height <= REACH_Z[1]:
        return f"grip {height:.2f} m above the desk (reach {REACH_Z[0]}-{REACH_Z[1]})"
    return None


def _spots(part, yaw, grip_y):
    """(x range, y range) of the object's origin, turned `yaw`, that keep it on the desk (OVERHANG past the far
    edge) with its grip point in GRIP_X and `grip_y`, the object beyond CLEAR_Y and the part, opened OPEN_OVER x its
    goal, beyond SWEEP_Y; None if there is none."""
    lo, hi = _footprint(part["lo"], part["hi"], yaw)
    gx, gy = _turn(part["grasp"][:2], yaw)
    near = min(_turn(c, yaw)[1] for c in _sweep(part, part.get("goal", open_goal(part)) * OPEN_OVER))
    x = (max(GRIP_X[0] - gx, -DESK_X - lo[0]), min(GRIP_X[1] - gx, DESK_X - hi[0]))
    y_lo = max(grip_y[0] - gy, DESK_NEAR - lo[1], CLEAR_Y - lo[1], SWEEP_Y - near)
    y = (y_lo, min(grip_y[1] - gy, DESK_FAR + OVERHANG - hi[1]))
    return (x, y) if x[0] <= x[1] and y[0] <= y[1] else None


def _sweep(part, opening):
    """(x, y) corners, in the object frame, of the part's box opened `opening` (rad or m) past closed."""
    lo, hi = part["part_lo"], part["part_hi"]
    k, pivot, dq = part["axis"], part["pivot"], part["sign"] * opening
    out = []
    for c in [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]:
        if part["type"] == "prismatic":
            out.append((c[0] + k[0] * dq, c[1] + k[1] * dq))
            continue
        v = [c[i] - pivot[i] for i in range(3)]
        dot = sum(k[i] * v[i] for i in range(3))
        cross = (k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0])
        cos, sin = math.cos(dq), math.sin(dq)
        out.append(tuple(v[i] * cos + cross[i] * sin + k[i] * dot * (1 - cos) + pivot[i] for i in range(2)))
    return out


def _face_yaw(front):
    """deg the object turns about z so its local `front` points at the robot (env -y)."""
    return math.degrees(math.atan2(-1.0, 0.0) - math.atan2(front[1], front[0]))


def _turn(xy, yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]


def _footprint(lo, hi, yaw):
    """Bounds (x, y) of the object's turned footprint about its origin."""
    corners = [_turn((x, y), yaw) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    return [min(c[i] for c in corners) for i in (0, 1)], [max(c[i] for c in corners) for i in (0, 1)]


def _analyze_safe(folder):
    try:
        return analyze(folder)
    except Exception as e:  # a broken asset is an exclusion, not a crash of the whole pool
        return dict(error=f"cannot read: {type(e).__name__}: {e}")


def build_pool(assets, objects, release, jobs=1, checks=None):
    """Pool and exclusions from the objects (paths under `assets`) of `release`; `checks` ({id: check_articulation
    checks}) leaves out the objects whose parts open by themselves (its `rest` check failed or did not run)."""
    from multiprocessing import Pool

    objects = [rel.strip() for rel in objects if rel.strip()]
    folders = [Path(assets) / rel for rel in objects]
    with Pool(jobs) as workers:
        geos = workers.map(_analyze_safe, folders, chunksize=1)
    entries, excluded = [], []
    for rel, folder, geo in zip(objects, folders, geos):
        parts = Path(rel).parts  # Assets/Object/<library>/Articulation/<cat>/<idx>
        lib, cat, idx = parts[2], parts[4], int(parts[5])
        oid = f"Articulation/{cat}/{idx:05d}"
        if "error" in geo:
            excluded.append(dict(id=oid, library=lib, reason=geo["error"]))
            continue
        if checks is not None and (checks.get(oid) or {}).get("rest") != "pass":
            why = "opens by itself (check_articulation rest)" if oid in checks else "not checked at rest"
            excluded.append(dict(id=oid, library=lib, reason=why))
            continue
        desc = json.loads((folder / "description.json").read_text())
        names = object_names(desc)
        source = next((k for k in LIBRARY_LICENCES if cat.startswith(f"{k}_")), lib.lower())
        meta_source = json.loads((folder / "metadata.json").read_text()).get("source") or {}
        family = desc.get("names", {}).get("en", [cat])[0]
        obj = dict(
            id=oid, cat=cat, idx=idx, library=lib, source=source, lo=geo["lo"], hi=geo["hi"], sources=geo["sources"]
        )
        obj["sha256"] = geo["sha256"]  # metadata.json and the USD the pool read
        obj["license"] = meta_source.get("license") or LIBRARY_LICENCES.get(source, "RoboDojo / Real2Sim asset")
        if noncommercial(obj["license"]):
            excluded.append(dict(id=oid, library=lib, reason=f"non-commercial licence ({obj['license']})"))
            continue
        ok = [j for j in geo["joints"] if "reject" not in j]
        for j in geo["joints"]:
            if "reject" in j:
                excluded.append(dict(id=oid, library=lib, joint=j["joint"], role=j["role"], reason=j["reject"]))
        reach = {j["joint"]: reachable(obj, j) for j in ok}
        for j in ok:
            if reach[j["joint"]]:
                excluded.append(dict(id=oid, library=lib, joint=j["joint"], role=j["role"], reason=reach[j["joint"]]))
        named = part_names([j for j in geo["joints"] if "centre" in j]) if ok else {}
        siblings = {
            j["joint"]: {k: j[k] for k in ("type", "closed", "sign", "level", "drive") if k in j}
            for j in geo["joints"]
            if "closed" in j
        }
        for j in ok:
            if reach[j["joint"]]:
                continue
            entries.append(
                dict(
                    obj,
                    **{k: v for k, v in j.items()},
                    goal=open_goal(j),
                    family=family,
                    split=split_of(oid, family),
                    names=names,
                    part_name={"zh": named[j["joint"]][0], "en": named[j["joint"]][1]},
                    siblings=siblings,
                    sole=len(siblings) == 1,
                )
            )
    return dict(release=release, entries=entries, excluded=excluded)


# ---------------------------------------------------------------------------------------- layouts and words


def load_pool():
    return json.loads(POOL.read_text())


def entries(split="train", pool=None):
    """The split's parts layouts draw from: once the pool is screened, only the parts the screen kept."""
    pool = pool or load_pool()
    screened = "screened" in pool
    return [
        e
        for e in pool["entries"]
        if e["split"] == split and not e.get("screened_out") and ("screen" in e or not screened)
    ]


def order(items, key="g1d_open_anything"):
    """The part order of layout i (mod len): one fixed shuffle, so any run of layouts mixes objects and roles."""
    idx = list(range(len(items)))
    random.Random(key).shuffle(idx)
    return idx


def layout(rng, index, entry, split="train", pool=None, part=None):
    """Articulation section of layout `index`, {cat: [entry(cat, idx, "target", xy, yaw, bottom)]} (bottom: the
    object's lowest point below its origin), and the part to open: `part`, else the split's index-th in order."""
    if part is None:
        items = entries(split, pool)
        part = items[order(items, f"g1d_open_anything/{split}")[index % len(items)]]
    yaw = _face_yaw(part["front"]) + rng.uniform(-YAW, YAW)
    spots = _spots(part, yaw, GRIP_Y) or _spots(part, yaw, REACH_Y)  # deep objects: the grip nearer than GRIP_Y
    if spots is None:
        raise RuntimeError(f"no room for open_anything layout {index}: {part['id']} {part['joint']}")
    x, y = rng.uniform(*spots[0]), rng.uniform(*spots[1])
    target = {k: part[k] for k in LAYOUT_KEYS}
    return {part["cat"]: [entry(part["cat"], part["idx"], "target", (x, y), yaw, part["lo"][2] - DESK_GAP)]}, target


def instruction(rng, target):
    """One instruction: a language, one of the object's names and a template of the part's role (or, for an object
    with one openable part, sometimes the object alone)."""
    lang = rng.choice(("zh", "en"))
    obj = rng.choice(target["names"][lang])
    if target["sole"] and rng.random() < 0.3:
        return rng.choice(SOLE[lang]).format(obj=obj)
    return rng.choice(TEMPLATES[target["role"]][lang]).format(obj=obj, part=target["part_name"][lang])


def screen_layouts(assets, folder, trials, pool, only=()):
    """`trials` layouts of each pool part (train and eval) into `folder` as g1d_open_anything_<i>.json, and
    screen.json mapping i to the part."""
    from scripts.g1d import build_asset as b

    _, room, room_pos, _ = b.SCENES["g1d"]
    params = json.loads((assets / "Room" / room / "usd" / "build_params.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    plan = []
    for e in pool["entries"]:
        key = f"{e['id']}|{e['joint']}"
        if only and key not in only:
            continue
        for t in range(trials):
            rng = random.Random(f"g1d_open_anything/screen/{key}/{t}")
            layout = b.open_anything_layout(assets, room, room_pos, params, rng, len(plan), part=e)
            (folder / f"g1d_open_anything_{len(plan)}.json").write_text(json.dumps(layout, indent=4))
            plan.append(dict(part=key, trial=t))
    (folder / "screen.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{len(plan)} layouts in {folder}")


def screen_results(layouts, raw):
    """{"<id>|<joint>": {tried, opened, why}} from the episodes of a screen-layouts folder; why: the most common
    reason a try was not accepted (its sidecar's reason, else the part not opened far enough)."""
    plan = json.loads((Path(layouts) / "screen.json").read_text())
    out = {}
    for side in sorted(Path(raw).glob("seed*/episode_*.json")):
        try:
            q = json.loads(side.read_text())
        except ValueError:  # cut short
            continue
        key = plan[int(side.stem.split("_")[1])]["part"]
        r = out.setdefault(key, dict(tried=0, opened=0, why=[]))
        r["tried"] += 1
        if q.get("accepted"):
            r["opened"] += 1
        else:
            r["why"].append(q.get("reason") or f"opened {q.get('opening_max', 0):.3f} of {q.get('goal', 0):.3f}")
    for r in out.values():
        r["why"] = max(set(r["why"]), key=r["why"].count) if r["why"] else ""
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="rewrite the pool from a library copy")
    p.add_argument("--assets", type=Path, required=True, help="the copy's root (holding Assets/ and sources/)")
    p.add_argument("--objects", type=Path, required=True, help="asset_client.py find --motion openable output")
    p.add_argument("--release", type=int, required=True)
    p.add_argument("--out", type=Path, default=POOL)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--checks", type=Path, help="{id: check_articulation checks} (rest must pass)")
    w = sub.add_parser("screen-layouts", help="layouts of every pool part, --trials each (run in the sim env)")
    w.add_argument("--assets", type=Path, default=Path("Assets"))
    w.add_argument("--folder", type=Path, required=True, help="Assets/Eval_Layout/RoboDojo/g1d/<seed>")
    w.add_argument("--trials", type=int, default=3)
    w.add_argument("--pool", type=Path, default=POOL)
    w.add_argument("--only", nargs="*", default=[], help="<id>|<joint> of the parts to lay out (default: all)")
    s = sub.add_parser("screen", help="mark the parts the feasibility screen could not open")
    s.add_argument("--layouts", type=Path, required=True, help="the screen-layouts folder (its screen.json)")
    s.add_argument("--raw", type=Path, required=True, help="run_batch output's raw/ (episode sidecars)")
    s.add_argument("--min-rate", type=float, default=0.5)
    args = parser.parse_args()
    if args.cmd == "pool":
        paths = [line.split("\t")[0] for line in args.objects.read_text().splitlines() if line.strip()]
        checks = json.loads(args.checks.read_text()) if args.checks else None
        pool = build_pool(args.assets, paths, args.release, args.jobs, checks)
        rules = dict(jaw_max=JAW_MAX, min_depth=MIN_DEPTH, grip_x=GRIP_X, grip_y=GRIP_Y, reach_y=REACH_Y)
        rules.update(reach_z=REACH_Z, yaw=YAW, open_deg=OPEN_DEG, open_m=OPEN_M, open_share=OPEN_SHARE)
        rules.update(clear_y=CLEAR_Y, sweep_y=SWEEP_Y, open_over=OPEN_OVER)
        pool["rules"] = dict(rules, eval_share=EVAL_SHARE, licences="commercial use allowed (noncommercial())")
        pool["licences"] = LIBRARY_LICENCES
        args.out.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
        n = len(pool["entries"])
        objs = len({e["id"] for e in pool["entries"]})
        print(f"{n} parts of {objs} objects in the pool, {len(pool['excluded'])} parts left out")
    elif args.cmd == "screen-layouts":
        screen_layouts(args.assets, args.folder, args.trials, json.loads(args.pool.read_text()), args.only)
    else:
        pool = load_pool()
        results = screen_results(args.layouts, args.raw)
        for e in pool["entries"]:
            r = results.get(f"{e['id']}|{e['joint']}")
            if r is None:
                continue
            e["screen"] = dict(tried=r["tried"], opened=r["opened"])
            e.pop("screened_out", None)
            if r["opened"] < args.min_rate * r["tried"]:
                e["screened_out"] = f"expert opened {r['opened']}/{r['tried']}" + (f": {r['why']}" if r["why"] else "")
        resplit(pool["entries"])
        pool["screened"] = dict(layouts=str(args.layouts), min_rate=args.min_rate, release=pool["release"])
        kept = [e for e in pool["entries"] if "screen" in e and not e.get("screened_out")]
        POOL.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n")
        opened = sum(r["opened"] for r in results.values())
        print(f"{len(kept)} of {len(results)} screened parts kept ({opened} episodes opened)")


if __name__ == "__main__":
    main()
