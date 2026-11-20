"""Import Sand3D rigid objects as RoboDojo public assets. Programs are never executed.

Sand3D publishes each object as a parquet row (a Python program, a render, and an OpenUSD
file) on Hugging Face under CC-BY-4.0. The USD exporter writes joints and rigid bodies, and
writes no mass, inertia, friction, or collider. This tool keeps the per-object
source_license and attribution, and turns a rigid, tabletop-sized USDA into a self-contained
object.usdz the rigid loader can drop:

    python scripts/objects/sand3d.py select --rows metadata.jsonl --manifest selection.json --limit 20
    python scripts/objects/sand3d.py select --rows data/*/*.parquet --manifest selection.json
    python scripts/objects/sand3d.py convert --source <dataset checkout> --manifest selection.json --out extra

select reads id, path, joint count, score, family, and licence columns. It does not read the
program or the render. convert reads the USDA only. Articulated rows are listed and skipped.
Output is Assets-style Object/RoboDojo/Rigid/sand3d_<family>/<index>/. Nothing is written
under Object/Blocks/, and the 88 wood blocks are not replaced. Passing check_object.py is a
later GPU step; runtime_ready stays false. Do not submit the result to the shared library
until that check has been run on the objects you keep.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

CODE_URL = "https://github.com/Sandflow-ai/sand3d"
DATASET_LICENSE = "CC-BY-4.0"
CODE_LICENSE = "Apache-2.0"
DATASETS = {
    "simready": {
        "repo": "Sandflow/sand3d-simready",
        "url": "https://huggingface.co/datasets/Sandflow/sand3d-simready",
        "f_min": 0.70,
    },
    "articraft": {
        "repo": "Sandflow/sand3d-articraft",
        "url": "https://huggingface.co/datasets/Sandflow/sand3d-articraft",
        "f_min": 0.90,
    },
}
ALLOWED_SOURCE_LICENSES = {"CC-BY-4.0", "CC0-1.0", "CC0", "Apache-2.0"}
REQUIRED_COLUMNS = (
    "id",
    "usda_path",
    "n_joints",
    "f_score",
    "family",
    "license",
    "source_license",
    "attribution",
    "license_status",
    "source",
    "asset_id",
)
OPTIONAL_COLUMNS = ("source_group", "sand3d_version", "usda_sha256", "program_sha256", "split")
SIZE_M = (0.03, 0.50)  # longest side of a tabletop object, metres
THIN_M = 0.003
DENSITY = 250.0  # kg/m^3, a hollow household object; the USDA has no mass
MASS_RANGE = (0.005, 0.5)
HULL_VERTICES = 64  # PhysX GPU particles need a convex hull at or under this
ID_RE = re.compile(r"Rigid/sand3d_[a-z][a-z0-9_]{0,63}/\d{5}")


def scrub(row):
    """A row with the program and the render removed. Those columns are never interpreted."""
    kept = {key: row[key] for key in row if key not in ("program", "render")}
    return kept


def normalize_license(value):
    text = str(value or "").strip()
    folded = text.upper().replace(" ", "")
    aliases = {"CCBY4.0": "CC-BY-4.0", "CC-BY4.0": "CC-BY-4.0", "APACHE2.0": "Apache-2.0"}
    return aliases.get(folded, text)


def category_name(family):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(family).strip()).strip("_").lower()
    if not slug or slug[0].isdigit():
        slug = f"x_{slug}" if slug else "x"
    return f"sand3d_{slug[:48]}"


def reject(row):
    """Why a scrubbed row is not a rigid public candidate, or None."""
    if row.get("license_status") != "ok":
        return "license status is not ok"
    if row.get("license") != DATASET_LICENSE:
        return "dataset license is not CC-BY-4.0"
    if normalize_license(row.get("source_license")) not in ALLOWED_SOURCE_LICENSES:
        return "source license is not CC-BY-4.0, CC0, or Apache-2.0"
    if not str(row.get("attribution") or "").strip():
        return "missing attribution"
    source = row.get("source")
    if source not in DATASETS:
        return "source is not simready or articraft"
    try:
        joints = int(row.get("n_joints"))
    except (TypeError, ValueError):
        return "missing joint count"
    if joints != 0:
        return "articulated"
    score = row.get("f_score")
    if not isinstance(score, int | float) or not math.isfinite(score) or score < DATASETS[source]["f_min"]:
        return "f_score below the dataset floor"
    usda = str(row.get("usda_path") or "")
    path = Path(usda)
    if not usda or path.is_absolute() or ".." in path.parts or path.suffix != ".usda":
        return "usda path"
    if not str(row.get("id") or "").strip() or not str(row.get("family") or "").strip():
        return "missing id or family"
    if not str(row.get("asset_id") or "").strip():
        return "missing asset id"
    return None


def public_id(family, index):
    oid = f"Rigid/{category_name(family)}/{index:05d}"
    if not ID_RE.fullmatch(oid) or "/wood_block_" in f"/{oid}":
        raise ValueError(f"refusing id {oid}")
    return oid


def select(rows, limit=None):
    """Manifest of rigid candidates. `rows` must already be scrubbed of program text."""
    chosen, dropped, pending, seen = [], [], [], set()
    for raw in rows:
        row = scrub(raw)
        oid = row.get("id")
        if oid in seen:
            dropped.append({"id": oid, "reason": "duplicate id"})
            continue
        seen.add(oid)
        why = reject(row)
        if why:
            dropped.append({"id": oid, "reason": why})
        else:
            pending.append(row)
    pending.sort(key=lambda row: str(row["id"]))
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        dropped.extend({"id": row["id"], "reason": "over limit"} for row in pending[limit:])
        pending = pending[:limit]
    counts = {}
    for row in pending:
        family = str(row["family"])
        cat = category_name(family)
        index = counts.get(cat, 0)
        if index > 99999:
            dropped.append({"id": row["id"], "reason": "category full"})
            continue
        counts[cat] = index + 1
        chosen.append(
            {
                "id": public_id(family, index),
                "src": row["usda_path"],
                "dataset": row["source"],
                "asset_id": row["asset_id"],
                "family": family,
                "f_score": float(row["f_score"]),
                "license": DATASET_LICENSE,
                "source_license": normalize_license(row["source_license"]),
                "attribution": str(row["attribution"]).strip(),
                **{key: row[key] for key in OPTIONAL_COLUMNS if row.get(key) not in (None, "")},
            }
        )
    return {
        "dataset_license": DATASET_LICENSE,
        "code_license": CODE_LICENSE,
        "code_url": CODE_URL,
        "sources": {name: spec["url"] for name, spec in DATASETS.items()},
        "chosen": chosen,
        "dropped": dropped,
        "note": "Rigid tabletop candidates. Programs are not executed. Articulated objects are omitted.",
    }


def read_parquet(path):
    """Metadata columns of a Sand3D parquet file. The program and the render stay on disk."""
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    missing = [column for column in REQUIRED_COLUMNS if column not in names]
    if missing:
        raise ValueError(f"parquet missing {missing}")
    columns = [column for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if column in names]
    if "program" in columns or "render" in columns:
        raise RuntimeError("refusing to load the program or the render")
    return pq.read_table(path, columns=columns).to_pylist()


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return read_parquet(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
        if not isinstance(rows, list):
            raise ValueError("JSON rows must be a list")
        return rows
    raise ValueError(f"unsupported row file {path.suffix}; use .parquet, .jsonl, or .json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r4(values):
    return [round(float(value), 4) + 0.0 for value in values]


def placement_metadata(size):
    corners = [
        [x, y, z]
        for x in (-size[0] / 2, size[0] / 2)
        for y in (-size[1] / 2, size[1] / 2)
        for z in (-size[2] / 2, size[2] / 2)
    ]
    radius = math.hypot(size[0], size[1]) / 2
    box = {"extents": r4(size), "vertices": [r4(corner) for corner in corners]}
    return {"aligned_bbox": box, "oriented_bbox": box, "radius": round(radius, 4) + 0.0}, {
        "place": {
            "up": {
                "orientation": [1, 0, 0, 0],
                "projection_circle": {"center": [0, 0, round(-size[2] / 2, 4) + 0.0, 1, 0, 0, 0], "radius": radius},
            }
        }
    }


def mesh_world_points(stage):
    """(prim, local point count, face count, world points) for every Mesh."""
    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    found = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            raise ValueError(f"{prim.GetPath()} has no polygon mesh")
        transform = cache.GetLocalToWorldTransform(prim)
        world = [transform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in points]
        found.append((prim, len(points), len(counts), world, counts, indices))
    if not found:
        raise ValueError("USDA has no mesh")
    return found


def _hull(points):
    """(vertices (N, 3), triangle indices (F, 3), volume m^3) of the point set."""
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(np.asarray(points, float))
    except QhullError as exc:
        raise ValueError(f"mesh has no volume for a convex hull: {exc}") from exc
    vertices = np.asarray(points, float)[hull.vertices]
    remap = {int(old): new for new, old in enumerate(hull.vertices)}
    triangles = np.array([[remap[int(index)] for index in face] for face in hull.simplices], int)
    faces = vertices[triangles]
    signed = float(np.einsum("ij,ij->i", faces[:, 0], np.cross(faces[:, 1], faces[:, 2])).sum()) / 6
    if signed < 0:
        triangles = triangles[:, ::-1]
    return vertices, triangles, abs(float(hull.volume))


def support_points(points, limit):
    """At most `limit` points on the convex hull, one per Fibonacci direction."""
    import numpy as np

    pts = np.asarray(points, float)
    index = np.arange(limit)
    z = 1 - 2 * (index + 0.5) / limit
    radius = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = np.pi * (1 + 5**0.5) * index
    directions = np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=1)
    chosen = pts[np.argmax(pts @ directions.T, axis=0)]
    return np.unique(np.round(chosen, 5), axis=0)


def convex_hull(points):
    """Hull of `points`, reduced to HULL_VERTICES when the full hull is finer than PhysX particles allow.

    Returns vertices, triangles, volume, and the vertex count before that reduction.
    """
    vertices, triangles, volume = _hull(points)
    source_vertices = len(vertices)
    if source_vertices > HULL_VERTICES:
        vertices, triangles, volume = _hull(support_points(vertices, HULL_VERTICES))
    return vertices, triangles, volume, source_vertices


def choose_mass(volume):
    raw = float(volume) * DENSITY
    mass = min(max(raw, MASS_RANGE[0]), MASS_RANGE[1])
    return round(mass, 4) + 0.0, {
        "measured": False,
        "method": "convex hull volume times 250 kg/m^3",
        "density_kg_m3": DENSITY,
        "volume_m3": round(float(volume), 6) + 0.0,
        "unclamped_kg": round(raw, 4) + 0.0,
        "clamped": abs(raw - mass) > 5e-5,
    }


def diffuse_color(prim):
    from pxr import Usd, UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    root = material.GetPrim() if material and material.GetPrim().IsValid() else None
    if root is None:
        return (0.7, 0.7, 0.7)
    for child in Usd.PrimRange(root):
        attr = child.GetAttribute("inputs:diffuseColor")
        if attr and attr.HasValue():
            got = attr.Get()
            if got is not None and len(got) >= 3:
                return tuple(float(channel) for channel in got[:3])
    return (0.7, 0.7, 0.7)


def add_preview(stage, path, color):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def add_mesh(stage, path, points, counts, indices, purpose=None):
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, point)) for point in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(count) for count in counts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(index) for index in indices]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    lo = [min(point[axis] for point in points) for axis in range(3)]
    hi = [max(point[axis] for point in points) for axis in range(3)]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    if purpose:
        mesh.CreatePurposeAttr(purpose)
    return mesh


def convert_one(source, entry, out):
    """Write one self-contained rigid object.usdz plus metadata. Raises if the USDA cannot be a tabletop rigid."""
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    if not ID_RE.fullmatch(entry["id"]) or "/wood_block" in entry["id"]:
        raise ValueError(f"refusing id {entry['id']}")
    source = Path(source).resolve()
    usda = (source / entry["src"]).resolve()
    if not usda.is_relative_to(source) or not usda.is_file():
        raise ValueError("usda path escapes the checkout or is missing")
    digest = file_sha256(usda)
    expected = entry.get("usda_sha256")
    if expected and expected != digest:
        raise ValueError("usda sha256 does not match the manifest")
    stage_in = Usd.Stage.Open(str(usda))
    if not stage_in or not stage_in.GetDefaultPrim():
        raise ValueError("USDA has no default prim")
    if UsdGeom.GetStageUpAxis(stage_in) != UsdGeom.Tokens.z or UsdGeom.GetStageMetersPerUnit(stage_in) != 1:
        raise ValueError("USDA is not Z-up metres")
    for prim in stage_in.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("USDA contains a joint or an articulation root")
    meshes = mesh_world_points(stage_in)
    stacked = np.concatenate([np.asarray(item[3], float) for item in meshes], axis=0)
    lo, hi = stacked.min(0), stacked.max(0)
    size = hi - lo
    if not np.isfinite(size).all() or not SIZE_M[0] <= float(size.max()) <= SIZE_M[1] or float(size.min()) < THIN_M:
        raise ValueError(f"bounds outside tabletop range: {size.tolist()}")
    center = (lo + hi) / 2
    centered = stacked - center
    hull_points, hull_faces, volume, source_vertices = convex_hull(centered)
    if len(hull_points) < 4 or volume <= 0:
        raise ValueError("convex hull is empty")
    mass, mass_source = choose_mass(volume)
    target = Path(out) / "Object" / "RoboDojo" / entry["id"]
    if "Blocks" in target.parts:
        raise ValueError("refusing to write into the block library")
    target.mkdir(parents=True, exist_ok=True)
    wrapper, package = target / "object.usda", target / "object.usdz"
    wrapper.unlink(missing_ok=True)
    package.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(wrapper))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/visual")
    for number, (prim, _, _, world_points, counts, indices) in enumerate(meshes):
        local = np.asarray(world_points, float) - center
        name = "model" if number == 0 else f"part_{number:02d}"
        visual = add_mesh(stage, f"/World/visual/{name}", local, counts, indices)
        color = diffuse_color(prim)
        material = add_preview(stage, f"/World/Looks/look_{number:02d}", color)
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    UsdGeom.Scope.Define(stage, "/World/collision")
    collider = add_mesh(
        stage,
        "/World/collision/hull_00",
        hull_points,
        [3] * len(hull_faces),
        hull_faces.reshape(-1),
        purpose=UsdGeom.Tokens.guide,
    )
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
    UsdPhysics.RigidBodyAPI.Apply(world.GetPrim())
    UsdPhysics.MassAPI.Apply(world.GetPrim()).CreateMassAttr(mass)
    stage.GetRootLayer().Save()
    package = target / "object.usdz"
    with Sdf.ZipFileWriter.CreateNew(str(package)) as archive:
        if not archive.AddFile(str(wrapper), "object.usda"):
            raise ValueError("USDZ packaging failed")
    wrapper.unlink()
    packaged = Usd.Stage.Open(str(package))
    bodies = [prim.GetPath().pathString for prim in packaged.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies != ["/World"]:
        raise ValueError(f"rigid bodies other than /World remain: {bodies}")
    joints = [prim.GetPath().pathString for prim in packaged.Traverse() if prim.IsA(UsdPhysics.Joint)]
    if joints:
        raise ValueError(f"joints remain: {joints}")
    colliders = [prim for prim in packaged.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1:
        raise ValueError(f"expected one collider, found {len(colliders)}")
    geometry, active = placement_metadata(size.tolist())
    geometry["faces"] = int(sum(item[2] for item in meshes))
    geometry["vertices"] = int(sum(item[1] for item in meshes))
    dataset = DATASETS[entry["dataset"]]
    source_info = {
        "dataset": dataset["repo"],
        "url": dataset["url"],
        "object": entry["src"],
        "asset_id": entry["asset_id"],
        "family": entry["family"],
        "f_score": entry.get("f_score"),
        "license": DATASET_LICENSE,
        "source_license": entry["source_license"],
        "attribution": entry["attribution"],
        "license_status": "ok",
        "code_url": CODE_URL,
        "code_license": CODE_LICENSE,
        "sand3d_version": entry.get("sand3d_version"),
        "usda_sha256": digest,
        "program_sha256": entry.get("program_sha256"),
        "program": "not executed",
        "measured": False,
        "candidate": "generalization asset, not a calibrated twin",
        "dynamics": "source USDA has no mass, inertia, friction, or collider",
        "mass": mass_source,
        "collision": {
            "approximation": "convexHull",
            "vertices": int(len(hull_points)),
            "source_vertices": int(source_vertices),
            "particle_ready": int(len(hull_points)) <= HULL_VERTICES,
            "source": "hull of the visual meshes; the USDA collider was absent",
        },
        "friction": "runtime defaults (static 0.6, dynamic 1.5); the USDA has no friction",
    }
    metadata = {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset['url']}/{entry['asset_id']}")),
        "physics": {"mass": mass, "size": r4(size)},
        "geometry": geometry,
        "active": active,
        "passive": {"functional": {}, "support": {}},
        "source": source_info,
        "runtime_ready": False,
    }
    description = {
        "caption": {"name": [entry["family"]], "source": [entry["dataset"]]},
        "description": [
            f"{entry['family']} imported from {dataset['repo']} as a generalization candidate. {entry['attribution']}"
        ],
    }
    for name, value in (("metadata.json", metadata), ("description.json", description)):
        (target / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "id": entry["id"],
        "src": entry["src"],
        "dir": str(target),
        "size_m": r4(size),
        "mass_kg": mass,
        "source_license": entry["source_license"],
        "attribution": entry["attribution"],
        "sha256": file_sha256(package),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "convert"))
    parser.add_argument(
        "--rows", type=Path, nargs="+", help="select: parquet, jsonl, or json metadata files; programs are not read"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="convert: dataset checkout that contains the usda/ files")
    parser.add_argument("--out", type=Path, help="convert: directory that receives Object/RoboDojo/...")
    parser.add_argument("--limit", type=int, help="select: keep at most this many candidates, after sorting by id")
    args = parser.parse_args()
    if args.command == "select":
        if args.rows is None:
            parser.error("select requires --rows")
        result = select([row for path in args.rows for row in read_rows(path)], args.limit)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"selected={len(result['chosen'])}, excluded={len(result['dropped'])}")
        return
    if args.source is None or args.out is None:
        parser.error("convert requires --source and --out")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {
        "dataset_license": manifest.get("dataset_license", DATASET_LICENSE),
        "code_license": CODE_LICENSE,
        "converted": [],
        "failed": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in manifest["chosen"]:
        try:
            result["converted"].append(convert_one(args.source, entry, args.out))
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            result["failed"].append({**entry, "error": str(exc)})
        text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        (args.out / "conversion.json").write_text(text, encoding="utf-8")
        print(f"converted={len(result['converted'])}, failed={len(result['failed'])}", flush=True)


if __name__ == "__main__":
    main()
