"""Adapter layer for public articulated-object libraries (artvip.py, robocasa.py, partnet_mobility.py).

The original package stays untouched under <library root>/sources/<library>/; each object gets a text layer
Assets/Object/RoboDojo/Articulation/<prefix>_<category>/<index>/object.usd that references it by a relative path (the
same in the asset library and in a run dir holding Assets/ and sources/ side by side), plus metadata.json and
description.json. What the layer changes, as overs on the referenced prims only:

    /root            the referenced default prim; gets ArticulationRootAPI + PhysxArticulationAPI (moved off the link
                     that had them), as ArticulationObject.fix_root_link expects (env/scene_manager/objects/)
    base link        put first among /root's children (fix_root_link fixes the first Xform child with RigidBodyAPI);
                     keeps PhysxArticulationAPI, which fix_root_link copies to /root and then removes
    world joints     fixed joints with one empty body are deactivated: they pin a link at its authored world position,
                     wherever a layout puts the object; fixed_base (metadata physics) adds a fresh one at load time
    lights, cameras  deactivated; render settings outside the default prim are not referenced at all
    scripts          omni:scripting:scripts blocked (BehaviorScript controllers)
    origin           /root's own transform (which the loader overwrites with the layout pose) and the unit scale
                     folded into every child of /root, then all shifted by -centre of the bounding box (the loader
                     places object centres)
    mass             MassAPI on links that have none: the estimated total split by link bounding-box volume
    limits           a joint whose lower limit exceeds its upper (PhysX locks it) gets the pair swapped

Needs pxr (usd-core is enough). CPU tests: scripts/objects/test_articulated.py.
"""

import json
import math
import os
from pathlib import Path
import re
import uuid

LIGHT_TYPES = ("DistantLight", "DomeLight", "RectLight", "SphereLight", "DiskLight", "CylinderLight", "Camera")
SCRIPT_ATTR = "omni:scripting:scripts"
PHYSX_ARTICULATION = "physxArticulation"
CENTER_OP = "xformOp:translate:ofw_center"
MIN_LINK_MASS = 0.02  # kg: a knob's box volume alone would give grams, too light next to a stiff drive
# what a joint does, from words in its child link's name (or a library's part labels); the first match wins.
# A word matches at the start of a name part ("E_drawer_2", "displayer"); "cap" only as a whole word (not "capsule").
# `part` is the detail; `role` is the closed set the platform classifies by. Only door, drawer and lid "open" something:
# a hinged door, a pull-out container, a flap that really closes. Racks and trays in ovens, pump heads, screw caps,
# tilting display panels, fan covers are role "other" with their part; a library maps a part to a role where its
# kind makes it one (artvip.PART_ROLES: a laptop's screen is a lid, a letter tray's tray a drawer).
PARTS = (
    ("pointer", "pointer"),
    ("drawer", "drawer"),
    ("basket", "drawer"),
    ("tray", "tray"),
    ("rack", "rack"),
    ("grill", "rack"),
    ("door", "door"),
    ("flipframe", "door"),
    ("lid", "lid"),
    ("flap", "lid"),
    ("leaves", "lid"),
    ("envelope", "lid"),
    ("cover", "cover"),
    ("screen", "screen"),
    ("display", "screen"),
    ("plate", "turntable"),
    ("turntable", "turntable"),
    ("knob", "knob"),
    ("dial", "knob"),
    ("button", "button"),
    ("switch", "switch"),
    ("pedal", "pedal"),
    ("handle", "handle"),
    ("pump", "pump"),
    ("cap(?![a-z])", "cap"),
    ("fan_blade", "rotor"),
    ("fan", "rotor"),
)
# scripts/objects/SKILL.md「铰接物体的关节作用」: door / drawer / lid open something, the rest only move
ROLE_VALUES = ("door", "drawer", "lid", "knob", "button", "turntable", "handle", "pedal", "pointer", "rotor", "other")
OPENING_PARTS = ("door", "drawer", "lid")
MIN_OPENING_M = 0.015  # a prismatic door / drawer / lid travelling less (and < 5% of the object) is a push button
# a hinged link named a knob, a button or nothing that is a plate covering the object (at least PANEL_SPAN of it along
# both sides, thinner than PANEL_THIN of its shorter side) and swings PANEL_MIN_DEG or more is a door (standing) or a
# lid (lying): ArtVIP's folder cover named E_knob_1, PartNet's lift-up cabinet top named countertop
PANEL_MIN_DEG = 60.0
PANEL_SPAN = 0.5
PANEL_THIN = 0.25
TURNTABLE_MIN_DEG = 180.0  # a tray that turns this far, or without limits, is a microwave's turntable


def panel_part(extent, size):
    """The part ("door" standing, "lid" lying) if a link of this extent (x, y, z, m) is a plate covering an object of
    this size, else None."""
    thin, short, long_ = sorted(range(3), key=lambda i: extent[i])
    if extent[thin] > PANEL_THIN * extent[short]:
        return None
    if min(extent[short] / size[short], extent[long_] / size[long_]) < PANEL_SPAN:
        return None
    return "lid" if thin == 2 else "door"


def joint_part(part, joint_type, lower, upper, extent=None, size=None):
    """The part a joint moves, from the part its name gives (part_of) and its motion where the two disagree.
    lower / upper: degrees or metres; extent: the child link's bounding box, size: the object's (m)."""
    stroke = abs(upper - lower) if None not in (lower, upper) else None
    if joint_type == "revolute":
        if part == "button":  # a "button" that turns is a knob
            part = "knob"
        elif part == "tray" and (stroke is None or stroke >= TURNTABLE_MIN_DEG):
            return "turntable"
        if part in ("knob", "other") and stroke is not None and stroke >= PANEL_MIN_DEG and extent and size:
            return panel_part(extent, size) or part
        return part
    if part == "knob":  # a "knob" that slides is pressed
        return "button"
    if part in OPENING_PARTS and stroke is not None and size and stroke < min(MIN_OPENING_M, 0.05 * max(size)):
        return "button"  # millimetres of travel open nothing: a push button labelled with the door it sits on
    return part


def part_of(link_name, joint_type):
    words = link_name.lower()
    for word, part in PARTS:
        if re.search(rf"(?<![a-z]){word}", words):
            if part == "switch":  # a switch that slides is pressed, one that turns is a knob
                return "button" if joint_type == "prismatic" else "knob"
            return part
    return "other"


def role_of(part, part_roles=None):
    """The role of a part: a part named like a role is that role (ROLE_VALUES), the rest are "other"; a library's
    part_roles map a part to a role where its kind makes it one."""
    role = (part_roles or {}).get(part, part)
    return role if role in ROLE_VALUES else "other"


def rel_path(target, start):
    return Path(os.path.relpath(os.path.abspath(target), os.path.abspath(start))).as_posix()


def _joint_info(prim):
    from pxr import UsdPhysics

    joint = UsdPhysics.Joint(prim)
    kind = "fixed"
    axis = lower = upper = None
    for cls, name in ((UsdPhysics.RevoluteJoint, "revolute"), (UsdPhysics.PrismaticJoint, "prismatic")):
        if prim.IsA(cls):
            typed = cls(prim)
            kind, axis = name, typed.GetAxisAttr().Get()
            lower, upper = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
    if kind == "fixed" and not prim.IsA(UsdPhysics.FixedJoint):
        kind = prim.GetTypeName()
    bodies = [[str(t) for t in rel.GetTargets()] for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]
    drive = {}
    for token in ("angular", "linear"):
        if prim.HasAPI(UsdPhysics.DriveAPI, token):
            api = UsdPhysics.DriveAPI(prim, token)
            drive = {
                "type": api.GetTypeAttr().Get(),
                "stiffness": api.GetStiffnessAttr().Get(),
                "damping": api.GetDampingAttr().Get(),
                "target": api.GetTargetPositionAttr().Get(),
            }
    # per-axis joint friction and armature (PhysxJointAxisAPI, robocasa.joint_axis_params; read as attributes so
    # usd-core without the PhysX schemas sees them)
    friction = {}
    for token in ("angular", "linear"):
        for key, attr in (
            ("static", "staticFrictionEffort"),
            ("dynamic", "dynamicFrictionEffort"),
            ("armature", "armature"),
        ):
            a = prim.GetAttribute(f"physxJointAxis:{token}:{attr}")
            if a and a.HasAuthoredValue():
                friction[key] = float(a.Get())
    return {
        "path": str(prim.GetPath()),
        "name": prim.GetName(),
        "type": kind,
        "axis": axis,
        "lower": None if lower is None or not math.isfinite(lower) else float(lower),
        "upper": None if upper is None or not math.isfinite(upper) else float(upper),
        "body0": bodies[0][0] if bodies[0] else None,
        "body1": bodies[1][0] if bodies[1] else None,
        "enabled": joint.GetJointEnabledAttr().Get() is not False,
        "excluded": bool(joint.GetExcludeFromArticulationAttr().Get()),
        "drive": drive,
        "friction": friction,
    }


def inspect(stage, prim_path=None):
    """What an adapter needs to know about a composed source stage; the object is its default prim, or prim_path."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"no prim {prim_path or '(default)'}")
    links, joints, roots, lights, scripts, colliders, masses = [], [], [], [], [], 0, {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.MassAPI):
                m = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                if m and m > 0:
                    masses[str(prim.GetPath())] = float(m)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            joints.append(_joint_info(prim))
        if prim.GetTypeName() in LIGHT_TYPES:
            lights.append(str(prim.GetPath()))
        if prim.HasAttribute(SCRIPT_ATTR) and prim.GetAttribute(SCRIPT_ATTR).Get():
            scripts.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            colliders += 1
    anchored = [j for j in joints if (j["body0"] is None) != (j["body1"] is None)]
    world = [j for j in anchored if j["type"] == "fixed"]
    world_movable = [j for j in anchored if j["type"] != "fixed"]
    movable = [j for j in joints if j["type"] in ("revolute", "prismatic") and j["body0"] and j["body1"]]
    children = {j["body1"] for j in joints if j["body0"] and j["body1"] and not j["excluded"]}
    bases = [link for link in links if link not in children]
    world_bodies = {j["body0"] or j["body1"] for j in world}
    if len(bases) > 1 and world_bodies & set(bases):
        bases = [b for b in bases if b in world_bodies]
    return {
        "root": str(root.GetPath()),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "links": links,
        "bases": bases,
        "joints": joints,
        "movable": movable,
        "world_joints": world,
        "world_movable": world_movable,
        "articulation_roots": roots,
        "lights": lights,
        "scripts": scripts,
        "colliders": colliders,
        "masses": masses,
    }


def _bound(cache, prim):
    import numpy as np

    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return np.asarray(rng.GetMin(), float), np.asarray(rng.GetMax(), float)


STILL_FRICTION = {"level": 1.5, "upright": 0.3, "prismatic": 2.0}  # see still_joints


def still_joints(layer_path, joints, facts, masses, spans, here):
    """Overs on the adapter layer so a door, drawer or lid neither opens by itself nor springs back: a drive whose
    target is not the closed pose (0, where the libraries author their parts) loses its stiffness (its damping
    stays), and each such joint gets Coulomb friction (PhysxJointAxisAPI, as robocasa.joint_axis_params): on a level
    hinge STILL_FRICTION["level"] x the part's weight torque at its box centre held level (m g r), so it holds at any
    angle; on an upright hinge or a slide a fixed STILL_FRICTION (N*m, N), small next to the hand. Estimates, written
    into the entries' `friction` and `drive` (metadata.json)."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(layer_path))
    by_name = {j["name"]: j for j in facts["movable"]}
    for entry in joints:
        if entry["role"] not in ("door", "drawer", "lid"):
            continue
        joint = by_name[entry["name"]]
        prim = stage.GetPrimAtPath(here(joint["path"]))
        token = "angular" if entry["type"] == "revolute" else "linear"
        drive = entry.get("drive") or {}
        if drive.get("stiffness") and abs(drive.get("target") or 0.0) > 1e-6:
            UsdPhysics.DriveAPI(prim, token).GetStiffnessAttr().Set(0.0)
            entry["drive"] = dict(drive, stiffness=0.0, stiffness_source=drive["stiffness"])
        if entry["type"] == "prismatic":
            friction = STILL_FRICTION["prismatic"]
        elif _axis_vertical(stage, prim):
            friction = STILL_FRICTION["upright"]
        else:
            span = spans.get(joint["body1"]) or [0.1, 0.1, 0.1]
            friction = round(STILL_FRICTION["level"] * masses.get(joint["body1"], 0.1) * 9.81 * max(span) / 2, 3)
        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{token}")
        for name in ("staticFrictionEffort", "dynamicFrictionEffort"):
            prim.CreateAttribute(f"physxJointAxis:{token}:{name}", Sdf.ValueTypeNames.Float).Set(float(friction))
        entry["friction"] = {"static": friction, "dynamic": friction, "source": "estimated (still_joints)"}
    stage.GetRootLayer().Save()


def _axis_vertical(stage, prim):
    """Is the joint's axis upright in the object frame (its body0 frame times localRot0)?"""
    from pxr import Gf, UsdGeom, UsdPhysics

    joint = UsdPhysics.Joint(prim)
    body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    world = UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(0)
    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())))
    axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[
        prim.GetAttribute("physics:axis").Get()
    ]
    v = world.TransformDir(rot.TransformDir(axis))
    return abs(v.GetNormalized()[2]) > 0.7


def write_layer(
    source,
    target_dir,
    total_mass,
    library_root,
    scale=1.0,
    limit_overrides=None,
    prim_path=None,
    limits_in_root_units=False,
    replace_masses=False,
    link_roles=None,
    part_roles=None,
    yaw_deg=0.0,
    still_parts=False,
):
    """object.usd in target_dir referencing `source` (a USD file under library_root/sources/); returns the facts the
    metadata needs. prim_path: the object's prim when it is not the default prim. scale: uniform, for libraries whose
    models are not at real size (PartNet); it scales every child of /root about the origin, and the prismatic limits,
    which are lengths. limits_in_root_units: the source's prismatic limits ignore a scale on the object's own prim
    (PartNet's URDF-importer USDs scale the links by 0.5 there but keep the URDF travel), so they get it too.
    replace_masses: authored link masses are placeholders; split total_mass anew, with box inertias of the links.
    link_roles: {link name: part} where link names do not tell (PartNet's link_0, link_1). part_roles: {part: role}
    where the object's kind makes a part open something (a laptop's screen is its lid). yaw_deg: turn about Z so the
    front faces -Y like the other libraries (PartNet models face -X). still_parts: doors, drawers and lids stay where
    they are left (still_joints): a drive spring aimed away from the closed pose loses its stiffness, and the joint
    gets friction (ArtVIP drives its parts toward their open limits)."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    source, target_dir = Path(os.path.abspath(source)), Path(target_dir)
    if not source.is_relative_to(Path(os.path.abspath(library_root)) / "sources"):
        raise ValueError(f"{source} is not under {library_root}/sources")
    target_dir.mkdir(parents=True, exist_ok=True)
    original = Usd.Stage.Open(str(source))
    facts = inspect(original, prim_path)
    if facts["up_axis"] != "Z":
        raise ValueError(f"up axis {facts['up_axis']} (only Z-up sources are adapted)")
    if not facts["movable"]:
        raise ValueError("no revolute or prismatic joint between two links")
    if facts["world_movable"]:
        raise ValueError(f"joint anchored to the world instead of a link: {facts['world_movable'][0]['path']}")
    if len(facts["bases"]) != 1:
        raise ValueError(f"cannot tell the base link: {facts['bases']}")
    if not facts["colliders"]:
        raise ValueError("no enabled collision geometry")
    src_root = facts["root"]

    layer_path = target_dir / "object.usd"
    layer = Sdf.Layer.CreateNew(str(layer_path), args={"format": "usda"})
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(rel_path(source, target_dir), Sdf.Path(src_root))

    def here(path):  # source path -> path under the adapter's /root
        return Sdf.Path("/root" + path[len(src_root) :])

    unit = facts["meters_per_unit"] * scale
    if unit <= 0:
        raise ValueError("bad metersPerUnit")
    art_attrs = {}
    # an articulation root above the referenced prim (PartNet's /Object over /Object/<Category>) is left out of the
    # reference, and with it settings such as enabledSelfCollisions = false: carry them over
    for prim in reversed(list(Sdf.Path(src_root).GetAncestorsRange())[1:]):
        above = original.GetPrimAtPath(prim)
        for attr in above.GetAttributes() if above else ():
            if attr.GetName().startswith(PHYSX_ARTICULATION + ":") and attr.HasAuthoredValue():
                art_attrs[attr.GetName()] = (attr.GetTypeName(), attr.Get())
    for path in facts["articulation_roots"]:
        prim = stage.GetPrimAtPath(here(path))
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(PHYSX_ARTICULATION + ":") and attr.HasAuthoredValue():
                art_attrs[attr.GetName()] = (attr.GetTypeName(), attr.Get())
        if prim.GetPath() != root.GetPath():
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAppliedSchema("PhysxArticulationAPI")
    base = here(facts["bases"][0])
    if base.GetParentPath() != root.GetPath():
        raise ValueError(f"base link {base} is not a child of the default prim")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # fix_root_link copies every PhysxArticulationAPI attribute from the base link to /root and fails on a link
    # without the API; by name, usd-core has no PhysxSchema
    for prim in (root, stage.GetPrimAtPath(base)):
        prim.AddAppliedSchema("PhysxArticulationAPI")
        for name, (kind, value) in art_attrs.items():
            prim.CreateAttribute(name, kind).Set(value)
    for joint in facts["world_joints"]:
        stage.GetPrimAtPath(here(joint["path"])).SetActive(False)
    for path in facts["lights"]:
        stage.GetPrimAtPath(here(path)).SetActive(False)
    for path in facts["scripts"]:
        stage.GetPrimAtPath(here(path)).GetAttribute(SCRIPT_ATTR).Block()

    names = [c.GetName() for c in root.GetChildren()]
    root.SetChildrenReorder([base.name] + [n for n in names if n != base.name])

    # the loader overwrites /root's own transform with the layout pose: fold it (and the unit scale) into the children
    root_xf = UsdGeom.Xformable(root)
    root_matrix = root_xf.GetLocalTransformation()
    root_xf.ClearXformOpOrder()
    travel = unit * (
        abs(root_matrix.ExtractRotationMatrix().GetDeterminant()) ** (1 / 3) if limits_in_root_units else 1
    )
    movable_children = [c for c in root.GetChildren() if UsdGeom.Xformable(c) and not c.IsA(UsdPhysics.Joint)]
    for child in movable_children:
        xf = UsdGeom.Xformable(child)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        front = []
        if unit != 1.0:
            front.append(xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "ofw_unit"))
            front[-1].Set(Gf.Vec3d(unit, unit, unit))
        if yaw_deg:
            front.append(xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, "ofw_yaw"))
            front[-1].Set(float(yaw_deg))
        if root_matrix != Gf.Matrix4d(1):
            front.append(xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ofw_root"))
            front[-1].Set(root_matrix)
        xf.SetXformOpOrder(front + list(ops), reset)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = _bound(cache, root)
    if bound is None:
        raise ValueError("empty bounding box")
    centre = (bound[0] + bound[1]) / 2
    for child in movable_children:
        xf = UsdGeom.Xformable(child)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        shift = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "ofw_center")
        shift.Set(Gf.Vec3d(*map(float, -centre)))
        xf.SetXformOpOrder([shift, *ops], reset)
    if travel != 1.0:
        for joint in facts["movable"]:
            if joint["type"] == "prismatic":
                prim = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(here(joint["path"])))
                for attr, value in (
                    (prim.GetLowerLimitAttr(), joint["lower"]),
                    (prim.GetUpperLimitAttr(), joint["upper"]),
                ):
                    if value is not None:
                        attr.Set(float(value * travel))
    limit_overrides = dict(limit_overrides or {})
    for joint in facts["movable"]:  # an inverted pair (lower > upper) locks the joint in PhysX: an authoring slip
        if joint["lower"] is not None and joint["upper"] is not None and joint["lower"] > joint["upper"]:
            limit_overrides.setdefault(joint["path"], (joint["upper"], joint["lower"]))
    for path, (lower, upper) in limit_overrides.items():
        prim = stage.GetPrimAtPath(here(path))
        typed = (
            UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(prim)
        )
        typed.GetLowerLimitAttr().Set(float(lower))
        typed.GetUpperLimitAttr().Set(float(upper))

    cache.Clear()
    lo, hi = _bound(cache, root)
    size = hi - lo
    if np.linalg.norm((lo + hi) / 2) > 1e-4:
        raise ValueError(f"origin not at the bounding-box centre after the shift: {(lo + hi) / 2}")
    volumes, extents, centres, spans = {}, {}, {}, {}
    for path in facts["links"]:
        b = _bound(cache, stage.GetPrimAtPath(here(path)))
        centres[path] = (b[0] + b[1]) / 2 if b else None
        spans[path] = [float(v) for v in b[1] - b[0]] if b else None
        volumes[path] = float(np.prod(np.maximum(b[1] - b[0], 0.005))) if b else 0.005**3
        extents[path] = np.maximum(b[1] - b[0], 0.05) if b else np.full(3, 0.05)  # inertia floor: no needle links
    authored = {} if replace_masses else dict(facts["masses"])
    free = [p for p in facts["links"] if p not in authored]
    left = total_mass - sum(authored.values())
    masses = dict(authored)
    if free:
        if left <= MIN_LINK_MASS * len(free):
            raise ValueError(f"authored link masses {sum(authored.values()):.3f} kg leave nothing of {total_mass} kg")
        weights = {p: volumes[p] for p in free}
        total_w = sum(weights.values())
        for p in free:
            masses[p] = max(MIN_LINK_MASS, left * weights[p] / total_w)
        scale_down = left / sum(masses[p] for p in free)
        for p in free:
            masses[p] = max(MIN_LINK_MASS, masses[p] * scale_down)
            api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(here(p)))
            api.CreateMassAttr(float(masses[p]))
            if p in facts["masses"]:
                # a placeholder inertia would not fit the new mass, and PhysX computes negative inertia from some
                # libraries' colliders (PartNet's instanced hulls): a solid box of the link's extent, an estimate
                # (and its centre of mass from the same colliders): a solid box of the link's extent at its centre,
                # both in the link's own (possibly scaled) frame, an estimate
                world = UsdGeom.Xformable(stage.GetPrimAtPath(here(p))).ComputeLocalToWorldTransform(0)
                link_scale = abs(world.ExtractRotationMatrix().GetDeterminant()) ** (1 / 3) or 1.0
                x, y, z = extents[p] ** 2
                box = [masses[p] / 12 * v / link_scale**2 for v in (y + z, x + z, x + y)]
                api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, box)))
                api.GetPrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))
                if centres[p] is not None:
                    local = world.GetInverse().Transform(Gf.Vec3d(*map(float, centres[p])))
                    api.GetCenterOfMassAttr().Set(Gf.Vec3f(local))
                else:
                    api.GetCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
    layer.Save()

    joints = []
    for joint in facts["movable"]:
        lower, upper = joint["lower"], joint["upper"]
        if limit_overrides.get(joint["path"]):
            lower, upper = limit_overrides[joint["path"]]
        elif joint["type"] == "prismatic" and travel != 1.0:
            lower, upper = (None if v is None else v * travel for v in (lower, upper))
        child = joint["body1"].rsplit("/", 1)[-1]
        part = (link_roles or {}).get(child) or part_of(child, joint["type"])
        part = joint_part(part, joint["type"], lower, upper, spans.get(joint["body1"]), [float(v) for v in size])
        entry = {
            "name": joint["name"],
            "type": joint["type"],
            "role": role_of(part, part_roles),
            "part": part,
            "parent": joint["body0"].rsplit("/", 1)[-1],
            "child": child,
            "axis": joint["axis"],
        }
        if joint["type"] == "revolute":
            entry["limits_deg"] = [lower, upper]
        else:
            entry["limits_m"] = [lower, upper]
        if joint["drive"]:
            entry["drive"] = {k: v for k, v in joint["drive"].items() if v is not None}
        if joint.get("friction"):
            entry["friction"] = joint["friction"]
        joints.append(entry)
    if still_parts:
        still_joints(layer_path, joints, facts, masses, spans, here)
    return {
        "size": [float(v) for v in size],
        "centre_shift": [float(v) for v in -centre],
        "unit": unit,
        "yaw_deg": yaw_deg,
        "joints": joints,
        "link_masses": {p.rsplit("/", 1)[-1]: round(m, 4) for p, m in masses.items()},
        "mass_authored": {p.rsplit("/", 1)[-1]: m for p, m in authored.items()},
        "limit_fixes": {p.rsplit("/", 1)[-1]: list(v) for p, v in limit_overrides.items()},
        "removed": {
            "world_joints": [j["path"] for j in facts["world_joints"]],
            "lights": facts["lights"],
            "scripts": facts["scripts"],
            "articulation_root_moved_from": facts["articulation_roots"],
        },
        "base_link": base.name,
        "links": [p.rsplit("/", 1)[-1] for p in facts["links"]],
    }


def box_geometry(size):
    corners = [
        [x, y, z]
        for x in (-size[0] / 2, size[0] / 2)
        for y in (-size[1] / 2, size[1] / 2)
        for z in (-size[2] / 2, size[2] / 2)
    ]
    radius = math.hypot(size[0], size[1]) / 2
    box = {"extents": list(size), "vertices": corners}
    return {"aligned_bbox": box, "oriented_bbox": box, "radius": radius}


def metadata(oid, facts, mass, mass_basis, source, fixed_base=True):
    """metadata.json: the loader keys (physics, geometry, active, passive) plus the joint list as the Real2Sim
    microwave writes it (body_joints) and where everything came from. ArticulationObject reads masses from the USD,
    not from here; physics.fixed_base (appliances and furniture yes, handheld things no) is what layouts copy into an
    instance to fix the base link."""
    size = facts["size"]
    geometry = box_geometry(size)
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source['url']}/{source['object']}")),
        "physics": {"mass": round(mass, 4), "size": [round(v, 4) for v in size], "fixed_base": fixed_base},
        "geometry": geometry,
        "active": {
            "place": {
                "up": {
                    "orientation": [1, 0, 0, 0],
                    "contact_circle": {"center": [0, 0, -size[2] / 2, 1, 0, 0, 0], "radius": geometry["radius"]},
                }
            }
        },
        "passive": {},
        "body_joints": facts["joints"],
        "base_link": facts["base_link"],
        "mass_source": mass_basis,
        "mass_partition": facts["link_masses"],
        "real_material_calibrated": False,
        "dimension_policy": "source model",
        "source": {
            **source,
            "adapter": {
                k: facts[k] for k in ("removed", "limit_fixes", "centre_shift", "unit", "yaw_deg") if facts[k] != 0.0
            },
        },
        "asset_id": oid,
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]", "_", text)).strip("_").lower()


def failure_reasons(result):
    """Why a check_articulation.py result failed, as short codes."""
    why = []
    if "load_error" in result:
        why.append("load")
    checks = result.get("checks", {})
    for key in ("fixed_base", "settle", "free_settle", "size", "render"):
        if checks.get(key) == "fail":
            why.append(key)
    joints = result.get("joints", {})
    if checks.get("joints") == "fail":
        if not math.isfinite(joints.get("peak_joint_speed", 0)) or joints.get("peak_joint_speed", 0) >= 50:
            why.append("exploded")
        for dof in joints.get("dof", []):
            if dof["pass"]:
                continue
            lo, hi = dof["nominal"]
            r_lo, r_hi = dof["reached"]
            short = any(not math.isfinite(v) for v in (r_lo, r_hi)) or (r_hi - r_lo) < 0.97 * (hi - lo)
            why.append("joint_blocked" if short else "joint_return")
    return sorted(set(why))


def manifest(selection, results):
    """One record of an import: every candidate with its decision, and every converted object with its checks."""
    by_id = {}
    for result in results:
        oid = "/".join(Path(result["asset_dir"]).parts[-3:])
        by_id[oid] = result
    passed, failed = [], []
    for entry in selection["chosen"]:
        result = by_id.get(entry["id"])
        record = {k: entry[k] for k in ("id", "src", "size_m", "reason") if k in entry}
        if result is None:
            failed.append({**record, "why": ["not checked"]})
        elif result.get("pass"):
            passed.append({**record, "checks": result["checks"]})
        else:
            failed.append({**record, "checks": result.get("checks", {}), "why": failure_reasons(result)})
    counts = {}
    for item in failed:
        for code in item["why"]:
            counts[code] = counts.get(code, 0) + 1
    return {
        **{k: selection[k] for k in ("source", "revision", "license")},
        "summary": {
            "candidates": len(selection["chosen"]) + len(selection["dropped"]),
            "dropped": len(selection["dropped"]),
            "converted": len(selection["chosen"]),
            "passed": len(passed),
            "failed": len(failed),
            "failure_reasons": counts,
        },
        "passed": passed,
        "failed": failed,
        "dropped": selection["dropped"],
    }


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="manifest: an import's selection + check_articulation.py reports")
    parser.add_argument("command", choices=("manifest",))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", required=True, help="report JSON files; later ones win")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [
        r for pattern in args.reports for f in sorted(glob.glob(pattern)) for r in json.loads(Path(f).read_text())
    ]
    data = manifest(json.loads(args.selection.read_text()), results)
    write_json(args.out, data)
    print(json.dumps(data["summary"], ensure_ascii=False))
