"""Adapt NVIDIA SimReady rigid packages in the asset library into ordinary RoboDojo rigid objects.

A SimReady package dir (`RoboDojo/Rigid/simready_<pkg>/00000/`: one top-level `sm_<pkg>_01.usd`, `payloads/`,
`Textures/`, ... linked from `sources/nvidia/simready-assets`) gets an `object.usd` wrapper and a full `metadata.json`:
    /root                    defaultPrim, Z-up, metres; RigidBodyAPI + MassAPI (RigidObject expects the body here)
      normalize              translate: the meshes' bounding-box centre to the origin (+ unit / up-axis transform)
        asset                references ./sm_<pkg>_01.usd, the package next to it
Overrides in the wrapper layer only (the package files stay untouched): instancing off below `asset` so its prims can
be edited, child rigid bodies and masses removed, child physics-material bindings cleared (the loader binds its own on
the root), cameras and lights deactivated. The package's colliders stay as authored (SimReady cups use SDF meshes,
which keep the cavity open). Mass is the mesh's material volume times the package material's density, clamped to a
plausible range for the material (MASS_RANGE). The reference is relative, so the wrapper stays valid in the overlay
and in the library, wherever the dir sits next to its package files.

    convert  write <out>/<id>/ for each id (dirs under Assets/Object): object.usd, metadata.json (runtime_ready
             false); the package's other entries mirrored (links as the same links, files copied unchanged):
             python scripts/objects/simready.py convert --lib Assets/Object --out ovl/Object RoboDojo/Rigid/<d>/00000
    ready    after the floor check (check_object.py --image-dir) and the bean check (check_cup_holds_beans.py):
             record both in metadata.json and set runtime_ready where both pass:
             python scripts/objects/simready.py ready --out ovl/Object --check-dir CHECK_DIR --beans metrics.json IDS
--ids-file FILE reads ids from a file (one per line) instead of the command line.
"""

import argparse
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

VERSION = 1
SCRIPT = "scripts/objects/simready.py"
# kg: thin-walled tableware of the material, from a paper cup / espresso cup up to a large stoneware mug
MASS_RANGE = {"paper": (0.006, 0.025), "plastic": (0.015, 0.08), "glass": (0.1, 0.3), "ceramic": (0.15, 0.45)}
# fallback when the package names no physics material: keywords of the package name
KIND_WORDS = (("paper", "paper"), ("plastic", "plastic"), ("glass", "glass"), ("wine", "glass"), ("mug", "ceramic"))
FRICTION = 0.5  # as the other cups' metadata; the loader reads only static/dynamic_friction (defaults 0.6 / 1.5)
CONTACT_BAND = 0.002  # m above the lowest point counted as the footprint
PLACE_KEYS = ("metadata.json", "object.usd")  # written here; every other entry of the package dir is linked


def r4(values):
    return [round(float(v), 4) + 0.0 for v in values]  # + 0.0: no -0.0 in the JSON


def material_kind(material_names, package):
    """paper / plastic / glass / ceramic from the package's physics material names (`m_opaque__ceramic__mug_a01`)."""
    for name in material_names:
        tokens = name.lower().split("__")
        for kind in MASS_RANGE:
            if kind in tokens:
                return kind
    for word, kind in KIND_WORDS:
        if word in package.lower():
            return kind
    raise ValueError(f"{package}: no material among {list(material_names)}")


def triangles(counts, indices):
    """Fan-triangulated vertex index triples of polygon faces (faceVertexCounts, faceVertexIndices)."""
    counts, indices = np.asarray(counts, int), np.asarray(indices, int)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    tris = [(indices[s], indices[s + k], indices[s + k + 1]) for s, c in zip(starts, counts) for k in range(1, c - 1)]
    return np.asarray(tris, int).reshape(-1, 3)


def solid_volume(points, tris):
    """Volume enclosed by a closed triangle mesh (m^3, by the divergence theorem); raises when it has open edges."""
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    if (uses == 1).any():
        raise ValueError(f"mesh is not closed: {int((uses == 1).sum())} open edges")
    t = np.asarray(points, float)[tris]
    return abs(float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6)


def choose_mass(volume, density, kind):
    """Mass (kg) of the material volume at the package's density, clamped to MASS_RANGE[kind]; and how it was chosen."""
    lo, hi = MASS_RANGE[kind]
    raw = volume * density
    if not math.isfinite(raw) or raw <= 0:
        raise ValueError(f"invalid material volume {volume} m^3 or density {density} kg/m^3")
    mass = min(max(raw, lo), hi)
    return round(mass, 4), {
        "rule": "material volume of the closed mesh x package material density, clamped to the material's range",
        "material": kind,
        "volume_cm3": round(volume * 1e6, 1),
        "density_kg_m3": density,
        "raw_kg": round(raw, 4),
        "range_kg": [lo, hi],
        "clamped": mass != raw,
        "measured": False,
    }


def geometry_metadata(points, faces):
    """geometry and active.place.up of an object whose points (root frame, m) are centred on their bounding box."""
    pts = np.asarray(points, float)
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    corners = [r4((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    box = {"vertices": corners, "extents": r4(ext)}
    foot = pts[pts[:, 2] <= lo[2] + CONTACT_BAND, :2]
    foot_c = (foot.min(0) + foot.max(0)) / 2
    geometry = {
        "faces": int(faces),
        "vertices": len(pts),
        "aligned_bbox": box,
        "oriented_bbox": box,
        "radius": round(float(np.linalg.norm(pts, axis=1).max()), 4),
        "bbox": r4(ext),
        "center": r4((lo + hi) / 2),
    }
    up = {
        "orientation": [1.0, 0.0, 0.0, 0.0],
        "projection_circle": {
            "center": r4((0, 0, lo[2], 1, 0, 0, 0)),
            "radius": round(float(np.linalg.norm(pts[:, :2], axis=1).max()), 4),
        },
        "contact_circle": {
            "center": r4((*foot_c, lo[2], 1, 0, 0, 0)),
            "radius": round(float(np.linalg.norm(foot - foot_c, axis=1).max()), 4),
        },
    }
    return geometry, {"place": {"up": up}, "functional": {}}


def object_uuid(object_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/{object_id}"))


def link_package(src, target):
    """Mirror the package dir except PLACE_KEYS into target: links as the same links, files (description.json, the
    thumbnail) copied unchanged, so the dir differs from the library's only in PLACE_KEYS."""
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        dest = target / entry.name
        if entry.name in PLACE_KEYS:
            continue
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        if entry.is_symlink():
            if not os.path.isabs(os.readlink(entry)):
                raise ValueError(f"{entry}: relative link would not resolve from {target}")
            dest.symlink_to(os.readlink(entry))
        elif entry.is_file():
            dest.write_bytes(entry.read_bytes())
        else:
            raise ValueError(f"{entry}: a plain directory in the package dir")


def pxr_modules():
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        globals()["APP"] = SimulationApp({"headless": True})
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    return Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


def convert_one(lib, object_id, out):
    """Write <out>/<object_id>/ from the package at <lib>/<object_id>/; returns a summary."""
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = pxr_modules()
    src, target = Path(lib) / object_id, Path(out) / object_id
    meta_in = json.loads((src / "metadata.json").read_text())
    raw = sorted(p for p in src.iterdir() if p.suffix in (".usd", ".usda") and p.name != "object.usd")
    if len(raw) != 1:
        raise ValueError(f"{src}: {len(raw)} top-level USD files")
    package = raw[0].name
    original = Usd.Stage.Open(str(src / package))
    if not original or not original.GetDefaultPrim():
        raise ValueError("package has no default prim")
    for prim in original.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError(f"articulated package: {prim.GetPath()}")
    axis, unit = UsdGeom.GetStageUpAxis(original), UsdGeom.GetStageMetersPerUnit(original)

    link_package(src, target)
    wrapper = target / "object.usd"
    if wrapper.exists():
        wrapper.unlink()
    stage = Usd.Stage.Open(Sdf.Layer.CreateNew(str(wrapper), args={"format": "usda"}))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    normalize = UsdGeom.Xform.Define(stage, "/root/normalize")
    asset = UsdGeom.Xform.Define(stage, "/root/normalize/asset").GetPrim()
    asset.GetReferences().AddReference(f"./{package}")
    shift = normalize.AddTranslateOp()
    if axis != UsdGeom.Tokens.z or unit != 1:
        transform = Gf.Matrix4d(1)
        if axis != UsdGeom.Tokens.z:
            transform.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
        transform = Gf.Matrix4d().SetScale(unit) * transform
        normalize.AddTransformOp(opSuffix="metric_z_up").Set(transform)
    while instances := [p for p in Usd.PrimRange(asset) if p.IsInstance()]:
        for p in instances:
            p.SetInstanceable(False)

    materials, colliders, approximations = set(), 0, set()
    for prim in Usd.PrimRange(asset):
        for api in (UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxRigidBodyAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        if prim.IsA(UsdGeom.Camera) or "Light" in prim.GetTypeName():
            prim.SetActive(False)
        binding = UsdShade.MaterialBindingAPI(prim)
        bound = binding.GetDirectBinding(materialPurpose="physics").GetMaterialPath()
        if not bound.isEmpty:
            materials.add(bound.name)
            binding.UnbindDirectBinding(materialPurpose="physics")
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            colliders += 1
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximations.add(str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()))
    if not colliders:
        raise ValueError("no enabled collision geometry")

    def mesh_points():
        cache = UsdGeom.XformCache()
        pts, tris, offset = [], [], 0
        for p in Usd.PrimRange(root.GetPrim()):
            if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputePurpose() != UsdGeom.Tokens.guide:
                mesh = UsdGeom.Mesh(p)
                local = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
                tf = np.asarray(cache.GetLocalToWorldTransform(p))
                faces = triangles(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
                tris.append(faces + offset)
                pts.append(local @ tf[:3, :3] + tf[3, :3])
                offset += len(local)
        if not pts:
            raise ValueError("no visible mesh")
        return np.concatenate(pts), np.concatenate(tris)

    pts, _ = mesh_points()
    lo, hi = pts.min(0), pts.max(0)
    center, size = (lo + hi) / 2, hi - lo
    if not np.isfinite(size).all() or not 0.03 <= max(size) <= 0.5 or min(size) < 0.003:
        raise ValueError(f"mesh bounds outside tabletop range: {size.tolist()}")
    shift.Set(Gf.Vec3d(*map(float, -center)))
    pts, tris = mesh_points()
    if np.abs((pts.min(0) + pts.max(0)) / 2).max() > 1e-5:
        raise ValueError("origin is not the bounding-box centre")

    physics_materials = [p for p in original.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]
    if not materials:  # bound by a collection or not at all: the package's only physics material
        materials = {p.GetName() for p in physics_materials}
    bound = [p for p in physics_materials if p.GetName() in materials]
    densities = sorted({float(UsdPhysics.MaterialAPI(p).GetDensityAttr().Get() or 0) for p in bound} - {0.0})
    if len(densities) != 1:
        raise ValueError(f"expected one physics material density, got {densities} from {sorted(materials)}")
    kind = material_kind(materials, src.parent.name)
    mass, mass_source = choose_mass(solid_volume(pts, tris), densities[0], kind)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(wrapper))
    bodies = [p.GetPath().pathString for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies != ["/root"]:
        raise ValueError(f"rigid bodies other than /root remain: {bodies}")
    # the loader binds its material on the root, weaker than descendants: every collider must end up with it
    reopened.SetEditTarget(reopened.GetSessionLayer())
    probe = UsdShade.Material.Define(reopened, "/root/probe_material")
    UsdShade.MaterialBindingAPI.Apply(reopened.GetPrimAtPath("/root")).Bind(
        probe, UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )
    for p in reopened.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            got = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose="physics")[0]
            if got.GetPath() != probe.GetPath():
                raise ValueError(f"{p.GetPath()} keeps its own physics material")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(wrapper)))
    builtins = {"OmniPBR.mdl", "OmniPBR_Opacity.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "OmniSurfaceBase.mdl"}
    if missing := [str(p) for p in unresolved if Path(str(p)).name not in builtins]:
        raise ValueError(f"unresolved dependencies: {missing}")

    geometry, active = geometry_metadata(pts, len(tris))
    source = dict(meta_in.get("source") or {})
    metadata = {
        "uuid": object_uuid(object_id),
        "physics": {"mass": mass, "friction": FRICTION, "size": r4(size)},
        "geometry": geometry,
        "active": active,
        "passive": {"functional": {}, "support": {}},
        "source": source,
        "adapter": {
            "script": SCRIPT,
            "version": VERSION,
            "wrapper": "object.usd",
            "reference": f"./{package}",
            "package_axis": {"up": str(axis), "meters_per_unit": unit},
            "offset_m": r4(-center),
            "collider": {"count": colliders, "approximation": sorted(approximations), "source": "package, unchanged"},
            "removed": "child rigid bodies, masses and physics-material bindings (wrapper layer only)",
            "mass": mass_source,
            "friction": "runtime defaults (static 0.6, dynamic 1.5); package materials "
            + ", ".join(sorted(materials))
            + " unbound",
            "validation": None,
        },
        "runtime_ready": False,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return {"id": object_id, "dir": str(target), "mass_kg": mass, "size_m": r4(size), "material": kind}


def ready(out, object_id, check_dir, beans, note=None):
    """Record the floor check and the bean check (and how they were run, `note`) in metadata.json; runtime_ready where
    both pass."""
    target = Path(out) / object_id
    meta = json.loads((target / "metadata.json").read_text())
    kind, category, index = object_id.split("/")[-3:]
    check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
    check = json.loads(check_file.read_text()) if check_file.is_file() else None
    bean = beans.get(object_id)
    passed = bool(check and check.get("settle") == "pass" and bean and bean.get("pass") and bean.get("stands"))
    meta["adapter"]["validation"] = {
        "check_object": check and {k: check.get(k) for k in ("settle", "tilt_deg", "z_m", "expect_z_m", "checked_at")},
        "check_cup_holds_beans": bean
        and {k: bean.get(k) for k in ("pass", "stands", "held", "stand_tilt_deg", "cavity_depth_mm", "volume_ml")},
        "passed": passed,
        "note": note,
    }
    meta["runtime_ready"] = passed
    meta.setdefault("source", {})["runtime_validated"] = passed
    (target / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("convert", "ready"))
    parser.add_argument("ids", nargs="*", help="object dirs relative to Assets/Object")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--lib", type=Path, default=Path("Assets/Object"), help="library holding the packages")
    parser.add_argument("--out", type=Path, required=True, help="Object dir of the overlay to write")
    parser.add_argument("--check-dir", type=Path, help="ready: the --image-dir given to check_object.py")
    parser.add_argument("--beans", type=Path, help="ready: metrics.json of check_cup_holds_beans.py")
    parser.add_argument("--note", help="ready: how the checks were run (run dir, settle time, ...)")
    args = parser.parse_intermixed_args()
    ids = list(args.ids) + (args.ids_file.read_text().split() if args.ids_file else [])
    if not ids:
        parser.error("no ids")
    if args.command == "ready":
        if not (args.check_dir and args.beans):
            parser.error("ready needs --check-dir and --beans")
        beans = json.loads(args.beans.read_text())["cups"]
        results = {i: ready(args.out, i, args.check_dir, beans, args.note) for i in ids}
        print(f"ready {sum(results.values())}/{len(results)}; not ready: {[i for i, ok in results.items() if not ok]}")
        return 0 if all(results.values()) else 1
    done, failed = [], {}
    for object_id in ids:
        try:
            done.append(convert_one(args.lib, object_id, args.out))
            print(f"[simready] {json.dumps(done[-1])}", flush=True)
        except (OSError, ValueError, RuntimeError) as exc:
            failed[object_id] = f"{type(exc).__name__}: {exc}"
            print(f"[simready] FAIL {object_id}: {failed[object_id]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"script": SCRIPT, "version": VERSION, "converted": done, "failed": failed}
    (args.out / "simready_conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[simready] converted {len(done)}, failed {len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
