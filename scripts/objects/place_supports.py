"""Supports Y for g1d_place_anything (把 X 放到 Y 上) from commercially usable library objects the task loader cannot
take as they are: NVIDIA SimReady packages (no object.usd, CC-BY-4.0) and RoboDojo Clutter objects (layouts place
Rigid objects; MIT, the clutter library comes from ManiTwin-100K, Apache-2.0). Each gets a Rigid object the loader
reads, lying flat on its widest face, the source files untouched:

    SimReady  RoboDojo/Rigid/simready_<pkg>/00000/ (the package dir): object.usd, metadata.json as
              scripts/objects/simready.py writes them, the package's other entries mirrored
    Clutter   RoboDojo/Rigid/clutter_<cat>/<idx>/ (a new dir): object.usd referencing
              ../../../Clutter/<cat>/<idx>/object.usdz (no mesh copied), metadata.json, description.json

object.usd:
    /root                defaultPrim, Z-up, metres; RigidBodyAPI + MassAPI
      normalize          translate (bounding-box centre to the origin) * orient (lay flat) * unit / up-axis transform
        asset            references the source's default prim; its rigid bodies and masses removed in this layer
Lay flat: a Clutter object turned by its own active.place.up orientation (how the clutter generator stands it), a
SimReady package with its thinnest box axis turned to +Z (cutting boards stand on edge in the package). Mass: SimReady
the closed mesh's volume times the package material's density, Clutter by hand (MASS), clamped to MASS_RANGE; all
estimates, runtime_ready false until `ready`.

    convert  python scripts/objects/place_supports.py convert --lib Assets/Object --out ovl/Object [IDS]
             (--out equal to --lib converts copied packages in place)
    candidates  place_anything.py pool --candidates input over the converted supports (JSON to stdout):
             python scripts/objects/place_supports.py candidates --out ovl/Object > cands.json
    ready    after check_object.py --image-dir CHECK and place_anything.py pool (geometry of each support):
             python scripts/objects/place_supports.py ready --out ovl/Object --check-dir CHECK --pool POOL.json \
                 [--screen S.json] [IDS]
    pool     the receptacle pool of round 2: round 1's unchanged plus the new supports that passed, split apart:
             python scripts/objects/place_supports.py pool --r1 R1.json --new POOL.json [--screen S.json] --out R2.json
IDS default to every id of SUPPORTS (written as Rigid ids under RoboDojo/).
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from scripts.objects import simready
except ImportError:  # run as a script from scripts/objects
    import simready

VERSION = 1
SCRIPT = "scripts/objects/place_supports.py"
SIMREADY_LICENSE = "CC-BY-4.0 (NVIDIA SimReady)"
CLUTTER_LICENSE = "MIT (RoboDojo repository; clutter library from ManiTwin-100K, Apache-2.0)"
# kg; the loader clamps a rigid body to 0.5 kg (env/scene_manager/objects/rigid.py), so wooden boards stay at that
MASS_RANGE = {"wood": (0.15, 0.5), "paper": (0.05, 0.5), "plastic": (0.05, 0.4), "ceramic": (0.2, 0.5)}
BOX_FILL = 0.8  # share of the bounding box a notebook or board fills, for the open-mesh mass estimate
KIND_WORDS = (("cuttingboard", "wood"), ("notebook", "paper"), ("plastic", "plastic"), ("plate", "ceramic"))
# Clutter id: (kg, material, what the estimate is from); every value an estimate
MASS = {
    "book/00000": (0.5, "paper", "21.8 x 16.4 x 4 cm hardcover, about 0.8 kg real, at the loader's 0.5 kg cap"),
    "book/00001": (0.35, "paper", "18 x 23.6 x 1.9 cm paperback"),
    "book/00002": (0.3, "paper", "13.5 x 21 x 2.5 cm paperback"),
    "book/00003": (0.3, "paper", "17.8 x 24.8 x 1.5 cm leather notebook"),
    "book/00004": (0.35, "paper", "the clutter metadata's 0.35 kg"),
    "plate/00000": (0.45, "ceramic", "22 cm ceramic dinner plate"),
    "plate/00001": (0.35, "ceramic", "19 cm ceramic plate"),
    "box/00001": (0.3, "paper", "18 x 17 x 4.6 cm board game box with its contents"),
}
SIMREADY = [f"kitchenware_cuttingBoard_{c}01" for c in "abcdefghijkl"] + [f"misc_notebook_a0{i}" for i in range(1, 5)]
SUPPORTS = {f"RoboDojo/Rigid/simready_{p}/00000": None for p in SIMREADY} | {
    f"RoboDojo/Rigid/clutter_{c}": f"RoboDojo/Clutter/{c}" for c in MASS
}
FLAT = {0: (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0), 1: (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0), 2: (1, 0, 0, 0)}
# the round-2 pool takes the supports the desk and the hands take, as place_anything.CANDIDATE_RULES["support"]
DESK = dict(narrow=(0.065, 0.24), long=0.32, height=0.05)
EVAL_SHARE = 0.15  # as place_anything.EVAL_SHARE, drawn among the new supports only: round 1's splits stay
LOW, LEAST = 0.2, 3  # screening: out when accepted less than LOW of at least LEAST episodes (place_anything screen)


def source_of(object_id):
    return SUPPORTS.get(object_id)


def kind_of(names, package):
    """wood / paper / plastic / ceramic from the package's physics material names, else words of its name."""
    for name in names:
        for kind in MASS_RANGE:
            if kind in name.lower().split("__"):
                return kind
    for word, kind in KIND_WORDS:
        if word in package.lower():
            return kind
    raise ValueError(f"{package}: no material among {sorted(names)}")


def lay_flat(extents):
    """(w, x, y, z) quaternion turning the thinnest axis of a box with `extents` to +Z (a quarter turn or none)."""
    return list(FLAT[int(np.argmin(extents))])


def clamp_mass(raw, kind):
    lo, hi = MASS_RANGE[kind]
    if not math.isfinite(raw) or raw <= 0:
        raise ValueError(f"invalid mass {raw}")
    return round(min(max(raw, lo), hi), 4), [lo, hi]


def wrap(lib, object_id, out):
    """Write <out>/<object_id>/ (object.usd, metadata.json; a Clutter one also description.json); a summary."""
    try:
        Gf, _, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = simready.pxr_modules()
    except ImportError:  # usd-core has no PhysxSchema; the schemas are removed by name
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    clutter = source_of(object_id)
    target = Path(out) / object_id
    if clutter:
        src = Path(lib) / clutter
        reference = os.path.relpath(src / "object.usdz", Path(lib) / object_id)
        meta_in = json.loads((src / "metadata.json").read_text())
        up = (meta_in.get("active") or {}).get("place", {}).get("up", {}).get("orientation", [1, 0, 0, 0])
        target.mkdir(parents=True, exist_ok=True)
        package = str(src / "object.usdz")
    else:
        src = Path(lib) / object_id
        meta_in = json.loads((src / "metadata.json").read_text())
        raw = sorted(p for p in src.iterdir() if p.suffix in (".usd", ".usda") and p.name != "object.usd")
        if len(raw) != 1:
            raise ValueError(f"{src}: {len(raw)} top-level USD files")
        reference, package, up = f"./{raw[0].name}", str(src / raw[0].name), None
        if target.resolve() != src.resolve():  # else converting a copy of the package in place
            simready.link_package(src, target)
    original = Usd.Stage.Open(package)
    if not original or not original.GetDefaultPrim():
        raise ValueError("source has no default prim")
    for prim in original.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError(f"articulated source: {prim.GetPath()}")
    axis, unit = UsdGeom.GetStageUpAxis(original), UsdGeom.GetStageMetersPerUnit(original)

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
    asset.GetReferences().AddReference(reference)
    shift = normalize.AddTranslateOp()
    orient = normalize.AddOrientOp()
    orient.Set(Gf.Quatf(1, 0, 0, 0))
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
        for api in ("PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"):
            if api in prim.GetAppliedSchemas():
                prim.RemoveAppliedSchema(api)
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
        shown = [
            p
            for p in Usd.PrimRange(root.GetPrim())
            if p.IsA(UsdGeom.Mesh)
            and UsdGeom.Imageable(p).ComputePurpose() != UsdGeom.Tokens.guide
            and UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible
        ]
        # sources with a separate collision mesh (Clutter) are measured on their visual meshes only
        visual = [p for p in shown if not p.HasAPI(UsdPhysics.CollisionAPI)] or shown
        for p in visual:
            mesh = UsdGeom.Mesh(p)
            local = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
            tf = np.asarray(cache.GetLocalToWorldTransform(p))
            faces = simready.triangles(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
            tris.append(faces + offset)
            pts.append(local @ tf[:3, :3] + tf[3, :3])
            offset += len(local)
        if not pts:
            raise ValueError("no visible mesh")
        return np.concatenate(pts), np.concatenate(tris)

    q = [float(v) for v in up] if up else lay_flat(np.ptp(mesh_points()[0], axis=0))
    q = [v / math.sqrt(sum(c * c for c in q)) for v in q]
    orient.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    pts, _ = mesh_points()
    lo, hi = pts.min(0), pts.max(0)
    center, size = (lo + hi) / 2, hi - lo
    if not np.isfinite(size).all() or not 0.05 <= max(size) <= 0.5 or min(size) < 0.003:
        raise ValueError(f"mesh bounds outside tabletop range: {size.tolist()}")
    if size[2] > min(size[:2]):
        raise ValueError(f"not lying flat: extents {simready.r4(size)}")
    shift.Set(Gf.Vec3d(*map(float, -center)))
    pts, tris = mesh_points()
    if np.abs((pts.min(0) + pts.max(0)) / 2).max() > 1e-5:
        raise ValueError("origin is not the bounding-box centre")

    physics_materials = [p for p in original.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]
    if clutter:
        mass_kg, kind, why = MASS[clutter.split("/", 2)[2]]
        mass, span = clamp_mass(mass_kg, kind)
        mass_source = dict(rule="estimated by hand", material=kind, basis=why, range_kg=span, measured=False)
    else:
        if not materials:
            materials = {p.GetName() for p in physics_materials}
        bound = [p for p in physics_materials if p.GetName() in materials]
        densities = sorted({float(UsdPhysics.MaterialAPI(p).GetDensityAttr().Get() or 0) for p in bound} - {0.0})
        if not densities:
            raise ValueError(f"no physics material density among {sorted(materials)}")
        kind = kind_of(materials, src.parent.name)
        try:
            if len(densities) != 1:
                raise ValueError("several densities")
            volume, density = simready.solid_volume(pts, tris), densities[0]
            rule = "material volume of the closed mesh x package material density"
        except ValueError:
            volume, density = float(np.prod(size)) * BOX_FILL, float(np.mean(densities))
            rule = f"bounding box volume x {BOX_FILL} x mean package material density (open mesh or several materials)"
        mass, span = clamp_mass(volume * density, kind)
        mass_source = {
            "rule": rule + ", clamped to the material's range",
            "material": kind,
            "volume_cm3": round(volume * 1e6, 1),
            "density_kg_m3": density,
            "raw_kg": round(volume * density, 4),
            "range_kg": span,
            "measured": False,
        }
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(wrapper))
    bodies = [p.GetPath().pathString for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies != ["/root"]:
        raise ValueError(f"rigid bodies other than /root remain: {bodies}")
    reopened.SetEditTarget(reopened.GetSessionLayer())
    probe = UsdShade.Material.Define(reopened, "/root/probe_material")
    UsdShade.MaterialBindingAPI.Apply(reopened.GetPrimAtPath("/root")).Bind(
        probe, UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )
    for p in reopened.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            got = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose="physics")[0]
            if got.GetPath() != probe.GetPath() and got.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise ValueError(f"{p.GetPath()} keeps its own physics material")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(wrapper)))
    builtins = {"OmniPBR.mdl", "OmniPBR_Opacity.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "OmniSurfaceBase.mdl"}
    if missing := [str(p) for p in unresolved if Path(str(p)).name not in builtins and str(p) != "gltf/pbr.mdl"]:
        raise ValueError(f"unresolved dependencies: {missing}")

    geometry, active = simready.geometry_metadata(pts, len(tris))
    source = dict(meta_in.get("source") or {})
    if clutter:
        source.update(library="RoboDojo Clutter", id=clutter.split("/", 1)[1], license=CLUTTER_LICENSE)
        desc = json.loads((src / "description.json").read_text()) if (src / "description.json").is_file() else {}
        desc["uuid"] = simready.object_uuid(object_id)
        (target / "description.json").write_text(json.dumps(desc, indent=2, ensure_ascii=False) + "\n")
    else:
        source.setdefault("license", SIMREADY_LICENSE)
    metadata = {
        "uuid": simready.object_uuid(object_id),
        "physics": {"mass": mass, "size": simready.r4(size)},
        "geometry": geometry,
        "active": active,
        "passive": {"functional": {}, "support": {}},
        "source": source,
        "adapter": {
            "script": SCRIPT,
            "version": VERSION,
            "wrapper": "object.usd",
            "reference": reference,
            "package_axis": {"up": str(axis), "meters_per_unit": unit},
            "lay_flat": simready.r4(q),
            "offset_m": simready.r4(-center),
            "collider": {"count": colliders, "approximation": sorted(approximations), "source": "unchanged"},
            "removed": "rigid bodies and masses below /root, physics-material bindings (wrapper layer only)",
            "mass": mass_source,
            "friction": "runtime defaults (static 0.6, dynamic 1.5)",
            "validation": None,
        },
        "runtime_ready": False,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return {"id": object_id, "mass_kg": mass, "size_m": simready.r4(size), "material": kind, "lay_flat": q}


def support_entry(geom):
    """passive.support from a pool support geometry (place_anything.geometry, root frame): the face's largest inside
    circle in the existing {"<name>/0": {"center": [[x, y, z, qw, qx, qy, qz]], "radius": [r]}} form, plus the
    largest inside rectangle's half extents along the object's axes."""
    reg = geom["region"]
    return {
        "top/0": {
            "center": [[*reg["center"], geom["floor"], 1.0, 0.0, 0.0, 0.0]],
            "radius": [reg["radius"]],
            "half": [reg["half"]],
            "source": "check_cup_holds_beans.py height map, place_anything.geometry",
        }
    }


def ready(out, object_id, check_dir, pool, note=None, stats=None):
    """Record the floor check, the support face and (`stats`: place_anything.py screen --json) the place screening in
    metadata.json; runtime_ready when it settles and has a face."""
    target = Path(out) / object_id
    meta = json.loads((target / "metadata.json").read_text())
    kind, category, index = object_id.split("/")[-3:]
    check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
    check = json.loads(check_file.read_text()) if check_file.is_file() else None
    geom = (pool.get(object_id) or {}).get("support")
    settled = bool(check and check.get("settle") == "pass")
    if geom:
        meta["passive"]["support"] = support_entry(geom)
    meta["adapter"]["validation"] = {
        "check_object": check and {k: check.get(k) for k in ("settle", "tilt_deg", "z_m", "expect_z_m", "checked_at")},
        "support_face": geom and {k: geom[k] for k in ("region", "floor", "top", "bottom")},
        "place_screening": None,
        "passed": settled and bool(geom),
        "note": note,
    }
    acc, n = (stats or {}).get("receptacle", {}).get(object_id, (0, 0))
    if n:
        meta["adapter"]["validation"]["place_screening"] = {
            "accepted": acc,
            "recorded": n,
            "passed": not screened(object_id, stats),
            "rule": f"g1d_place_anything expert v2 on the office desk; out below {LOW:.0%} of at least {LEAST}",
        }
    meta["runtime_ready"] = settled and bool(geom)
    meta.setdefault("source", {})["runtime_validated"] = meta["runtime_ready"]
    (target / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta["runtime_ready"]


def candidates(out, ids, names):
    """place_anything.py pool --candidates entries of the converted supports (as place_anything.candidates)."""
    rows = []
    for object_id in ids:
        meta_path = Path(out) / object_id / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        n = names[object_id.split("/", 1)[1]]
        rows.append(dict(rel=object_id, fam=n["category"]["en"], sup=True, con=False, size=meta["physics"]["size"],
                         mass=meta["physics"]["mass"], names=n, license=meta["source"]["license"]))  # fmt: skip
    return rows


def on_desk(size):
    narrow, long_ = sorted(size[:2])
    return DESK["narrow"][0] <= narrow <= DESK["narrow"][1] and long_ <= DESK["long"] and size[2] <= DESK["height"]


def id_hash(oid):
    return int(hashlib.sha256(oid.encode()).hexdigest()[:8], 16)


def screened(rid, stats):
    acc, n = (stats or {}).get("receptacle", {}).get(rid, (0, 0))
    return f"{acc}/{n} accepted in the screening" if n >= LEAST and acc / n < LOW else None


def round2(r1, new, stats=None):
    """Round 2's receptacles: round 1's unchanged, then the new supports (`new`: place_anything pool entries) that
    fit the desk and passed the screening, EVAL_SHARE of them (at least one, lowest id hashes) held out for eval;
    and why the others were left out."""
    old = {r["id"] for r in r1}
    keep, left = [], {}
    for r in new:
        if r["id"] in old or "support" not in r["kinds"]:
            continue
        if not on_desk(r["size"]):
            left[r["id"]] = f"size {r['size']} off the desk rule {DESK}"
        elif why := screened(r["id"], stats):
            left[r["id"]] = why
        else:
            geometry = {"support": r["geometry"]["support"]}
            keep.append(dict(r, kinds=["support"], geometry=geometry, mass_estimated=True, added_in="r2"))
    order = sorted(keep, key=lambda r: id_hash(r["id"]))
    n = max(1, round(EVAL_SHARE * len(order))) if order else 0
    for i, r in enumerate(order):
        r["split"] = {"support": "eval" if i < n else "train"}
    return [*r1, *sorted(keep, key=lambda r: r["id"])], left


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("convert", "candidates", "ready", "pool"))
    parser.add_argument("ids", nargs="*", help="Rigid object dirs relative to Assets/Object (default: SUPPORTS)")
    parser.add_argument("--lib", type=Path, default=Path("Assets/Object"), help="library holding the sources")
    parser.add_argument("--out", type=Path, help="convert / ready: Object dir of the overlay; pool: the R2 file")
    parser.add_argument("--check-dir", type=Path, help="ready: the --image-dir given to check_object.py")
    parser.add_argument("--pool", type=Path, help="ready: place_anything.py pool output over these supports")
    parser.add_argument("--note", help="ready: how the checks were run")
    parser.add_argument("--r1", type=Path, help="pool: round 1's place_anything_receptacles.json")
    parser.add_argument("--new", type=Path, help="pool: place_anything.py pool output over the new supports")
    parser.add_argument("--screen", type=Path, help="ready / pool: place_anything.py screen --json output")
    args = parser.parse_intermixed_args()
    if not args.out:
        parser.error("--out is required")
    if args.command == "pool":
        r1 = json.loads(args.r1.read_text())
        new = json.loads(args.new.read_text())
        stats = json.loads(args.screen.read_text()) if args.screen else None
        receptacles, left = round2(r1["receptacles"], new["receptacles"], stats)
        added = [r for r in receptacles if r.get("added_in") == "r2"]
        args.out.write_text(
            json.dumps(
                dict(
                    r1,
                    round="r2",
                    release=new["release"],
                    release_r1=r1["release"],
                    receptacles=receptacles,
                    left_out_r2=left,
                    rejected_r2=new.get("rejected", {}),
                ),
                indent=1,
                ensure_ascii=False,
            )
            + "\n"
        )
        split = [r["split"]["support"] for r in added]
        print(f"r2 adds {len(added)} supports: {split.count('train')} train, {split.count('eval')} eval; left {left}")
        return 0
    ids = list(args.ids) or list(SUPPORTS)
    if args.command == "candidates":
        names = json.loads(Path(__file__).with_name("object_names.json").read_text())
        names = {k: names[v] if isinstance(v, str) else v for k, v in names.items()}
        print(json.dumps(candidates(args.out, ids, names), indent=1, ensure_ascii=False))
        return 0
    if args.command == "ready":
        if not (args.check_dir and args.pool):
            parser.error("ready needs --check-dir and --pool")
        pool = {r["id"]: r["geometry"] for r in json.loads(args.pool.read_text())["receptacles"]}
        stats = json.loads(args.screen.read_text()) if args.screen else None
        results = {i: ready(args.out, i, args.check_dir, pool, args.note, stats) for i in ids}
        print(f"ready {sum(results.values())}/{len(results)}; not ready: {[i for i, ok in results.items() if not ok]}")
        return 0
    done, failed = [], {}
    for object_id in ids:
        try:
            done.append(wrap(args.lib, object_id, args.out))
            print(f"[supports] {json.dumps(done[-1])}", flush=True)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            failed[object_id] = f"{type(exc).__name__}: {exc}"
            print(f"[supports] FAIL {object_id}: {failed[object_id]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"script": SCRIPT, "version": VERSION, "converted": done, "failed": failed}
    (args.out / "place_supports_conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[supports] converted {len(done)}, failed {len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    if "APP" in vars(simready):
        if code:
            os._exit(code)
        simready.APP.close()
    raise SystemExit(code)
