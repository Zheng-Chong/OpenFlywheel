"""Adapt NVIDIA SimReady rigid packages in the asset library into ordinary RoboDojo rigid objects.

A SimReady package dir (`RoboDojo/Rigid/simready_<pkg>/00000/`: one top-level `sm_<pkg>_01.usd`, `payloads/`,
`Textures/`, ... linked from `sources/nvidia/simready-assets`) gets an `object.usd` wrapper and a full `metadata.json`:
    /root                    defaultPrim, Z-up, metres; RigidBodyAPI + MassAPI (RigidObject expects the body here)
      normalize              translate (bbox centre to origin) * orient (optional stable rest) * unit / up-axis
        asset                references ./sm_<pkg>_01.usd, the package next to it
Overrides in the wrapper layer only (the package files stay untouched): instancing off below `asset` so its prims can
be edited, child rigid bodies and masses removed, child physics-material bindings cleared (the loader binds its own on
the root), cameras and lights deactivated. The package's colliders stay as authored (SimReady cups use SDF meshes,
which keep the cavity open). Mass is the closed mesh's volume times the package material's density (falling back to
bounding-box fill volume and/or category default density for open meshes, multi-material packages, or packages without
authored density), clamped to a plausible range for the material (MASS_RANGE). The reference is relative, so the
wrapper stays valid in the overlay and in the library, wherever the dir sits next to its package files.

    survey   scan Assets/Object for SimReady packages and report readiness / broken links:
             python scripts/objects/simready.py survey --lib Assets/Object [--out survey.json] [--write-ids raw.txt]
    convert  write <out>/<id>/ for each id (dirs under Assets/Object): object.usd, metadata.json (runtime_ready
             false); the package's other entries mirrored (links as the same links, files copied unchanged):
             python scripts/objects/simready.py convert --lib Assets/Object --out ovl/Object [--all] [--skip-existing]
                 [--orient auto|keep|lay-flat] [IDS...]
    ready    after the floor check (check_object.py --image-dir) and optional bean check (check_cup_holds_beans.py):
             record check results in metadata.json and set runtime_ready where checks pass:
             python scripts/objects/simready.py ready --out ovl/Object --check-dir CHECK_DIR [--beans metrics.json]
                 [--all] [IDS...]
--ids-file FILE reads ids from a file (one per line) instead of the command line.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np

VERSION = 2
SCRIPT = "scripts/objects/simready.py"
SIMREADY_LICENSE = "CC-BY-4.0 (NVIDIA SimReady)"

# kg: plausible tabletop / rigid-object mass ranges per material kind.
# Keep paper / ceramic bounds compatible with earlier cup calibrations; upper bounds respect the loader's 0.5 kg cap.
MASS_RANGE = {
    "paper": (0.006, 0.025),
    "cardboard": (0.02, 0.45),
    "plastic": (0.01, 0.45),
    "glass": (0.05, 0.45),
    "ceramic": (0.15, 0.45),
    "wood": (0.03, 0.5),
    "metal": (0.01, 0.5),
    "food": (0.005, 0.5),
    "rubber": (0.01, 0.35),
    "foam": (0.003, 0.08),
    "fabric": (0.01, 0.3),
    "stone": (0.01, 0.5),
    "soap": (0.02, 0.25),
    "default": (0.005, 0.5),
}

# kg/m^3: fallback material density when a SimReady package authors no non-zero physics material density.
DEFAULT_DENSITY = {
    "paper": 700.0,
    "cardboard": 650.0,
    "plastic": 950.0,
    "glass": 2500.0,
    "ceramic": 2400.0,
    "wood": 650.0,
    "metal": 7800.0,
    "food": 850.0,
    "rubber": 1100.0,
    "foam": 80.0,
    "fabric": 300.0,
    "stone": 2600.0,
    "soap": 1000.0,
    "default": 800.0,
}

# Map physics-material name tokens (from `m_opaque__<token>__...`) to canonical MASS_RANGE keys.
MATERIAL_ALIASES = {
    "steel": "metal",
    "iron": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "brass": "metal",
    "copper": "metal",
    "chrome": "metal",
    "tin": "metal",
    "stainless": "metal",
    "bronze": "metal",
    "titanium": "metal",
    "alloy": "metal",
    "porcelain": "ceramic",
    "stoneware": "ceramic",
    "clay": "ceramic",
    "terracotta": "ceramic",
    "earthenware": "ceramic",
    "silicone": "rubber",
    "latex": "rubber",
    "elastomer": "rubber",
    "cloth": "fabric",
    "textile": "fabric",
    "cotton": "fabric",
    "leather": "fabric",
    "sponge": "foam",
    "styrofoam": "foam",
    "marble": "stone",
    "granite": "stone",
    "concrete": "stone",
    "rock": "stone",
    "chalk": "stone",
    "carton": "cardboard",
    "organic": "food",
    "fruit": "food",
    "vegetable": "food",
    "produce": "food",
    "meat": "food",
    "bread": "food",
    "wax": "soap",
    "acrylic": "plastic",
    "pvc": "plastic",
    "polypropylene": "plastic",
    "polyethylene": "plastic",
    "nylon": "plastic",
    "resin": "plastic",
    "vinyl": "plastic",
    "bamboo": "wood",
    "cork": "wood",
    "plywood": "wood",
    "oak": "wood",
    "pine": "wood",
}

# Fallback when the package names no recognized physics material: token/bigram keywords of the package name
# (ordered most-specific first; matched via `_package_terms` to avoid false substring hits like "pot" in "potato").
KIND_WORDS = (
    ("cuttingboard", "wood"),
    ("rollingpin", "wood"),
    ("notebook", "cardboard"),
    ("packaged", "cardboard"),
    ("carton", "cardboard"),
    ("food_box", "cardboard"),
    ("paper", "paper"),
    ("wipes", "paper"),
    ("barsoap", "soap"),
    ("soap", "soap"),
    ("sponge", "foam"),
    ("foam", "foam"),
    ("plastic", "plastic"),
    ("dishsoap", "plastic"),
    ("windowspray", "plastic"),
    ("dispenser", "plastic"),
    ("eyeglasses", "plastic"),
    ("mouse", "plastic"),
    ("marker", "plastic"),
    ("tape", "plastic"),
    ("ruler", "plastic"),
    ("toy", "plastic"),
    ("glass", "glass"),
    ("wine", "glass"),
    ("jar", "glass"),
    ("vial", "glass"),
    ("marble", "glass"),
    ("bottle_laboratory", "glass"),
    ("mug", "ceramic"),
    ("ceramic", "ceramic"),
    ("bowl", "ceramic"),
    ("plate", "ceramic"),
    ("cup", "ceramic"),
    ("bread", "food"),
    ("canned", "metal"),
    ("can", "metal"),
    ("thermos", "metal"),
    ("kettle", "metal"),
    ("pan", "metal"),
    ("pot", "metal"),
    ("knife", "metal"),
    ("hobbyknife", "metal"),
    ("fork", "metal"),
    ("spoon", "metal"),
    ("ladle", "metal"),
    ("spatula", "metal"),
    ("scissors", "metal"),
    ("wrench", "metal"),
    ("pliers", "metal"),
    ("hammer", "metal"),
    ("screwdriver", "metal"),
    ("socket", "metal"),
    ("ratchet", "metal"),
    ("torque", "metal"),
    ("pry", "metal"),
    ("breakerbar", "metal"),
    ("extension", "metal"),
    ("hexkey", "metal"),
    ("hexdriver", "metal"),
    ("clamp", "metal"),
    ("nut", "metal"),
    ("bolt", "metal"),
    ("screw", "metal"),
    ("tacks", "metal"),
    ("handtool", "metal"),
    ("tool", "metal"),
    ("hardware", "metal"),
    ("factory", "metal"),
    ("pencil", "wood"),
    ("wood", "wood"),
    ("chalk", "stone"),
    ("pastel", "soap"),
    ("golfball", "rubber"),
    ("rubber", "rubber"),
    ("food", "food"),
    ("produce", "food"),
    ("prepared", "food"),
    ("processed", "food"),
    ("beverage", "plastic"),
    ("bottle", "plastic"),
    ("jug", "plastic"),
    ("shaker", "glass"),
    ("tray", "plastic"),
    ("pen", "plastic"),
    ("cube", "plastic"),
    ("box", "cardboard"),
)

# Fraction of bounding-box volume used to estimate mass when a mesh is open (non-watertight).
BOX_FILL = 0.5
FRICTION = 0.5  # as the other cups' metadata; the loader reads only static/dynamic_friction (defaults 0.6 / 1.5)
CONTACT_BAND = 0.002  # m above the lowest point counted as the footprint
PLACE_KEYS = ("metadata.json", "object.usd")  # written here; every other entry of the package dir is linked

# Relaxed default size bounds (m) so small items (nuts, marbles, berries, thin blades) and larger tools/pans convert.
DEFAULT_MIN_SIZE = 0.008
DEFAULT_MAX_SIZE = 1.5
DEFAULT_MIN_THICKNESS = 0.0005
TABLETOP_BOUNDS = (0.03, 0.5, 0.003)  # (min_max_dim, max_max_dim, min_min_dim) for tabletop pool tagging

# Isaac Sim built-in MDL shader modules that resolve at runtime rather than inside each object package.
BUILTIN_MDLS = {
    "OmniPBR.mdl",
    "OmniPBR_Opacity.mdl",
    "OmniPBR_ClearCoat.mdl",
    "OmniPBR_ClearCoat_Opacity.mdl",
    "OmniGlass.mdl",
    "OmniSurface.mdl",
    "OmniSurfaceBase.mdl",
    "OmniSurfaceLite.mdl",
    "OmniSurfaceLiteBase.mdl",
    "OmniHair.mdl",
    "OmniHairBase.mdl",
    "OmniEmissive.mdl",
    "AperturePBR_Opacity.mdl",
    "AperturePBR_Translucent.mdl",
}
BUILTIN_MDL_PATHS = {"gltf/pbr.mdl"}

# Quarter-turn (w, x, y, z) quaternions rotating axis 0 (+X), 1 (+Y), or 2 (+Z) to +Z.
FLAT = {
    0: (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
    1: (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
    2: (1.0, 0.0, 0.0, 0.0),
}

# Packages that are authored upright and must stay upright in `--orient auto` even when tall and narrow.
UPRIGHT_WORDS = (
    "cup",
    "mug",
    "glass",
    "bottle",
    "jar",
    "can",
    "canned",
    "jug",
    "kettle",
    "thermos",
    "bowl",
    "pot",
    "shaker",
    "vial",
    "beverage",
    "dairy",
    "condiment",
    "dishsoap",
    "windowspray",
    "wipes",
    "dispenser",
    "packaged",
    "food_box",
)

# Packages that are frequently authored standing on edge / upright on a thin tip and settle best laid flat.
LAY_FLAT_WORDS = (
    "cuttingboard",
    "notebook",
    "tray",
    "knife",
    "hobbyknife",
    "fork",
    "spoon",
    "spatula",
    "ladle",
    "wrench",
    "pliers",
    "hammer",
    "screwdriver",
    "scissors",
    "clamp",
    "pry",
    "breakerbar",
    "extension",
    "hexkey",
    "hexdriver",
    "ratchet",
    "torque",
    "ruler",
    "pencil",
    "marker",
    "pen",
    "chalk",
    "pastel",
    "eyeglasses",
    "barsoap",
    "sponge",
    "rollingpin",
)


def r4(values):
    return [round(float(v), 4) + 0.0 for v in values]  # + 0.0: no -0.0 in the JSON


def _package_terms(package):
    """Lowercase underscore/camelCase tokens and adjacent bigrams for exact keyword matching."""
    toks = [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+", package)]
    bigrams = [f"{toks[i]}_{toks[i + 1]}" for i in range(len(toks) - 1)]
    return set(toks) | set(bigrams) | {seg.lower() for seg in package.split("_") if seg}


def material_kind(material_names, package, default=None):
    """Canonical material kind from physics material names (`m_opaque__ceramic__mug_a01`), else package keywords."""
    for name in sorted(material_names):
        tokens = [t for seg in name.lower().split("__") for t in seg.split("_") if t]
        for tok in tokens:
            if tok in MASS_RANGE and tok != "default":
                return tok
            if tok in MATERIAL_ALIASES:
                return MATERIAL_ALIASES[tok]
    terms = _package_terms(package)
    for word, kind in KIND_WORDS:
        if word in terms:
            return kind
    if default is not None:
        return default
    raise ValueError(f"{package}: no material among {sorted(material_names)}")


def lay_flat(extents):
    """(w, x, y, z) quaternion turning the thinnest axis of a box with `extents` to +Z (a quarter turn or none)."""
    return list(FLAT[int(np.argmin(extents))])


def stable_orientation(extents, package, mode="auto"):
    """Choose a (w, x, y, z) resting quaternion and reason string for `extents` under `mode` (auto|keep|lay-flat)."""
    ext = np.asarray(extents, float)
    if mode == "keep":
        return [1.0, 0.0, 0.0, 0.0], "keep"
    if mode == "lay-flat":
        return lay_flat(ext), "lay_flat"
    if mode != "auto":
        raise ValueError(f"unknown orient mode: {mode}")
    # Already resting on its widest face (within 5% tolerance)
    if ext[2] <= min(ext[0], ext[1]) * 1.05:
        return [1.0, 0.0, 0.0, 0.0], "already_flat"
    terms = _package_terms(package)
    if any(w in terms for w in UPRIGHT_WORDS):
        return [1.0, 0.0, 0.0, 0.0], "upright_category"
    narrow, wide = sorted((float(ext[0]), float(ext[1])))
    thin_on_edge = ext[2] > 0 and narrow / ext[2] < 0.25 and (wide / max(narrow, 1e-9)) >= 2.5
    if any(w in terms for w in LAY_FLAT_WORDS) or thin_on_edge:
        return lay_flat(ext), "lay_flat_auto"
    return [1.0, 0.0, 0.0, 0.0], "default_upright"


def check_bounds(size, min_size=DEFAULT_MIN_SIZE, max_size=DEFAULT_MAX_SIZE, min_thickness=DEFAULT_MIN_THICKNESS):
    """Validate finite, positive mesh extents against configurable cutoffs; return whether they fit tabletop bounds."""
    ext = np.asarray(size, float)
    if ext.shape != (3,) or not np.isfinite(ext).all() or (ext <= 0).any():
        raise ValueError(f"invalid mesh bounds: {ext.tolist()}")
    hi, lo = float(ext.max()), float(ext.min())
    if not (min_size <= hi <= max_size) or lo < min_thickness:
        raise ValueError(
            f"mesh bounds outside allowed range [{min_size}, {max_size}] / min {min_thickness}: {ext.tolist()}"
        )
    tt_min, tt_max, tt_thick = TABLETOP_BOUNDS
    return bool(tt_min <= hi <= tt_max and lo >= tt_thick)


def unresolved_dependencies(unresolved):
    """Filter UsdUtils.ComputeAllDependencies unresolved list against Isaac Sim built-in MDL modules."""
    missing = []
    for item in unresolved:
        raw = str(item)
        norm = raw.replace("\\", "/")
        if Path(norm).name in BUILTIN_MDLS or norm in BUILTIN_MDL_PATHS:
            continue
        missing.append(raw)
    return missing


def triangles(counts, indices):
    """Fan-triangulated vertex index triples of polygon faces (faceVertexCounts, faceVertexIndices)."""
    counts, indices = np.asarray(counts, int), np.asarray(indices, int)
    if len(counts) == 0:
        return np.zeros((0, 3), int)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    tris = [(indices[s], indices[s + k], indices[s + k + 1]) for s, c in zip(starts, counts) for k in range(1, c - 1)]
    return np.asarray(tris, int).reshape(-1, 3)


def solid_volume(points, tris):
    """Volume enclosed by a closed triangle mesh (m^3, by the divergence theorem); raises when it has open edges."""
    if len(tris) == 0:
        raise ValueError("mesh has no triangles")
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


def estimate_mass(pts, tris, size, densities, kind, box_fill=BOX_FILL):
    """Estimate mass (kg) and provenance dict across closed/open meshes, multi-material densities, and fallbacks."""
    valid_densities = [float(d) for d in (densities or []) if math.isfinite(float(d)) and float(d) > 0]
    if len(valid_densities) == 1:
        try:
            vol = solid_volume(pts, tris)
            if math.isfinite(vol) and vol > 0:
                return choose_mass(vol, valid_densities[0], kind)
        except ValueError:
            pass
    # Determine density (authored single, mean of multiple authored, or category default)
    if valid_densities:
        density = float(np.mean(valid_densities))
        density_source = "package_material" if len(valid_densities) == 1 else "mean_package_materials"
    else:
        density = DEFAULT_DENSITY[kind]
        density_source = "category_default"
    # Determine volume (closed mesh if valid, else bounding box * box_fill)
    volume_source = "closed_mesh"
    try:
        volume = solid_volume(pts, tris)
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("non-positive closed volume")
    except ValueError:
        ext = np.asarray(size, float)
        if ext.shape != (3,) or not np.isfinite(ext).all() or (ext <= 0).any():
            raise ValueError(f"invalid bounds for fallback mass: {ext.tolist()}")
        volume = float(np.prod(ext)) * box_fill
        volume_source = f"bbox_fill_{box_fill}"
    mass, info = choose_mass(volume, density, kind)
    info["rule"] = f"{volume_source} x {density_source} density, clamped to the material's range"
    info["volume_source"] = volume_source
    info["density_source"] = density_source
    return mass, info


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
            link_target = os.readlink(entry)
            if not os.path.isabs(link_target):
                link_target = os.path.abspath(os.path.join(str(src), link_target))
            dest.symlink_to(link_target)
        elif entry.is_file():
            dest.write_bytes(entry.read_bytes())
        else:
            raise ValueError(f"{entry}: a plain directory in the package dir")


def discover_simready_ids(root):
    """Find all `RoboDojo/Rigid/simready_*/00000` object IDs under `root` (e.g. Assets/Object)."""
    base = Path(root) / "RoboDojo" / "Rigid"
    if not base.is_dir():
        return []
    ids = []
    for pkg_dir in sorted(base.iterdir()):
        if not pkg_dir.name.startswith("simready_") or not pkg_dir.is_dir():
            continue
        inst = pkg_dir / "00000"
        if inst.is_dir():
            ids.append(f"RoboDojo/Rigid/{pkg_dir.name}/00000")
    return ids


def survey_one(lib, object_id):
    """Inspect a single SimReady package dir without opening USD; classify readiness and link integrity."""
    src = Path(lib) / object_id
    if not src.is_dir():
        return {"id": object_id, "status": "missing"}
    broken = [p.name for p in sorted(src.iterdir()) if p.is_symlink() and not p.exists()]
    raw_usds = sorted(p.name for p in src.iterdir() if p.suffix in (".usd", ".usda") and p.name != "object.usd")
    has_wrapper = (src / "object.usd").is_file() or (src / "object.usdz").is_file()
    meta_path = src / "metadata.json"
    runtime_ready = False
    adapter_script = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            runtime_ready = bool(meta.get("runtime_ready"))
            adapter_script = (meta.get("adapter") or {}).get("script")
        except (OSError, json.JSONDecodeError):
            pass
    if broken:
        status = "broken_links"
    elif has_wrapper and runtime_ready:
        status = "already_ready"
    elif has_wrapper:
        status = "converted_unvalidated"
    else:
        status = "raw_only"
    return {
        "id": object_id,
        "status": status,
        "has_wrapper": has_wrapper,
        "runtime_ready": runtime_ready,
        "adapter_script": adapter_script,
        "raw_usds": raw_usds,
        "broken_links": broken,
    }


def survey(lib, ids=None):
    """Survey all (or specified) SimReady packages under `lib` and return structured counts + per-item status."""
    object_ids = list(ids) if ids is not None else discover_simready_ids(lib)
    items = [survey_one(lib, oid) for oid in object_ids]
    counts = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"lib": str(lib), "total": len(items), "counts": counts, "items": items}


def pxr_modules():
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        globals()["APP"] = SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    return Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


def convert_one(
    lib,
    object_id,
    out,
    *,
    orient="auto",
    min_size=DEFAULT_MIN_SIZE,
    max_size=DEFAULT_MAX_SIZE,
    min_thickness=DEFAULT_MIN_THICKNESS,
    freeze_joints=False,
):
    """Write <out>/<object_id>/ from the package at <lib>/<object_id>/; returns a summary."""
    Gf, _, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = pxr_modules()
    src, target = Path(lib) / object_id, Path(out) / object_id
    meta_path = src / "metadata.json"
    meta_in = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    raw = sorted(p for p in src.iterdir() if p.suffix in (".usd", ".usda") and p.name != "object.usd")
    if len(raw) != 1:
        raise ValueError(f"{src}: {len(raw)} top-level USD files")
    package = raw[0].name
    original = Usd.Stage.Open(str(src / package))
    if not original or not original.GetDefaultPrim():
        raise ValueError("package has no default prim")
    frozen_joints = []
    for prim in original.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if not freeze_joints:
                raise ValueError(f"articulated package: {prim.GetPath()}")
            frozen_joints.append(str(prim.GetPath()))
    axis, unit = UsdGeom.GetStageUpAxis(original), UsdGeom.GetStageMetersPerUnit(original)

    if target.resolve() != src.resolve():
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
    orient_op = normalize.AddOrientOp()
    orient_op.Set(Gf.Quatf(1, 0, 0, 0))
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
    strip_apis = ("PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI")
    if freeze_joints:
        strip_apis = (*strip_apis, "PhysicsArticulationRootAPI", "PhysxArticulationAPI")
    for prim in Usd.PrimRange(asset):
        for api in strip_apis:
            if api in prim.GetAppliedSchemas():
                prim.RemoveAppliedSchema(api)
        if prim.IsA(UsdGeom.Camera) or "Light" in prim.GetTypeName():
            prim.SetActive(False)
        elif freeze_joints and prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        binding = UsdShade.MaterialBindingAPI(prim)
        bound = binding.GetDirectBinding(materialPurpose="physics").GetMaterialPath()
        if not bound.isEmpty:
            materials.add(bound.name)
            binding.UnbindDirectBinding(materialPurpose="physics")
        for col_binding in binding.GetCollectionBindings(materialPurpose="physics"):
            col_mat = col_binding.GetMaterialPath()
            if not col_mat.isEmpty:
                materials.add(col_mat.name)
            bind_name = (
                col_binding.GetBindingName()
                if hasattr(col_binding, "GetBindingName")
                else col_binding.GetCollectionName()
            )
            if bind_name:
                binding.UnbindCollectionBinding(bind_name, materialPurpose="physics")
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
        # Packages with separate collision meshes are measured on their visual meshes only.
        visual = [p for p in shown if not p.HasAPI(UsdPhysics.CollisionAPI)] or shown
        for p in visual:
            mesh = UsdGeom.Mesh(p)
            raw_pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if (
                raw_pts is None
                or len(raw_pts) == 0
                or counts is None
                or len(counts) == 0
                or indices is None
                or len(indices) == 0
            ):
                continue
            local = np.asarray(raw_pts, dtype=float)
            tf = np.asarray(cache.GetLocalToWorldTransform(p))
            faces = triangles(counts, indices)
            if len(faces) == 0:
                continue
            tris.append(faces + offset)
            pts.append(local @ tf[:3, :3] + tf[3, :3])
            offset += len(local)
        if not pts:
            raise ValueError("no visible mesh")
        return np.concatenate(pts), np.concatenate(tris)

    pre_pts, _ = mesh_points()
    q, orient_reason = stable_orientation(np.ptp(pre_pts, axis=0), src.parent.name, mode=orient)
    q_norm = math.sqrt(sum(c * c for c in q))
    q = [c / q_norm for c in q]
    orient_op.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))

    pts, _ = mesh_points()
    lo, hi = pts.min(0), pts.max(0)
    center, size = (lo + hi) / 2, hi - lo
    tabletop_ok = check_bounds(size, min_size=min_size, max_size=max_size, min_thickness=min_thickness)
    shift.Set(Gf.Vec3d(*map(float, -center)))
    pts, tris = mesh_points()
    if np.abs((pts.min(0) + pts.max(0)) / 2).max() > 1e-5:
        raise ValueError("origin is not the bounding-box centre")

    physics_materials = [p for p in original.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]
    if not materials:  # bound by a collection or not at all: the package's physics materials
        materials = {p.GetName() for p in physics_materials}
    bound = [p for p in physics_materials if p.GetName() in materials]
    densities = sorted({float(UsdPhysics.MaterialAPI(p).GetDensityAttr().Get() or 0) for p in bound} - {0.0})
    kind = material_kind(materials, src.parent.name, default="default")
    mass, mass_source = estimate_mass(pts, tris, size, densities, kind)
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
            if got.GetPath() != probe.GetPath() and got.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise ValueError(f"{p.GetPath()} keeps its own physics material")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(wrapper)))
    if missing := unresolved_dependencies(unresolved):
        raise ValueError(f"unresolved dependencies: {missing}")

    geometry, active = geometry_metadata(pts, len(tris))
    source = dict(meta_in.get("source") or {})
    source.setdefault("license", SIMREADY_LICENSE)
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
            "orient": r4(q),
            "orient_reason": orient_reason,
            "offset_m": r4(-center),
            "tabletop_range": tabletop_ok,
            "collider": {"count": colliders, "approximation": sorted(approximations), "source": "package, unchanged"},
            "removed": (
                "child rigid bodies, masses, joints, articulation roots and physics-material bindings "
                "(wrapper layer only)"
                if frozen_joints
                else "child rigid bodies, masses and physics-material bindings (wrapper layer only)"
            ),
            "frozen_joints": len(frozen_joints),
            "mass": mass_source,
            "friction": "runtime defaults (static 0.6, dynamic 1.5); package materials "
            + (", ".join(sorted(materials)) if materials else "(none)")
            + " unbound",
            "validation": None,
        },
        "runtime_ready": False,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return {
        "id": object_id,
        "dir": str(target),
        "mass_kg": mass,
        "size_m": r4(size),
        "material": kind,
        "orient_reason": orient_reason,
        "tabletop_range": tabletop_ok,
    }


def ready(out, object_id, check_dir, beans=None, note=None):
    """Record the floor check (and optional bean check) in metadata.json; runtime_ready where required checks pass."""
    target = Path(out) / object_id
    meta_path = target / "metadata.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    kind, category, index = object_id.split("/")[-3:]
    check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
    check = json.loads(check_file.read_text()) if check_file.is_file() else None
    settled = bool(check and check.get("settle") == "pass")
    if beans is None:
        bean = None
        passed = settled
    else:
        bean = beans.get(object_id)
        passed = bool(settled and bean and bean.get("pass") and bean.get("stands"))
    meta.setdefault("adapter", {})["validation"] = {
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
    parser.add_argument("command", choices=("survey", "convert", "ready"))
    parser.add_argument("ids", nargs="*", help="object dirs relative to Assets/Object")
    parser.add_argument("--ids-file", type=Path, help="file listing object ids, one per line")
    parser.add_argument("--all", action="store_true", help="discover all RoboDojo/Rigid/simready_*/00000 ids")
    parser.add_argument("--lib", type=Path, default=Path("Assets/Object"), help="library holding the packages")
    parser.add_argument("--out", type=Path, help="convert/ready: overlay Object dir; survey: optional JSON report path")
    parser.add_argument("--write-ids", type=Path, help="survey: write raw_only object ids to this file (one per line)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="convert: skip ids that already have object.usd + runtime_ready=true in --lib",
    )
    parser.add_argument("--orient", choices=("auto", "keep", "lay-flat"), default="auto", help="convert: orientation")
    parser.add_argument("--min-size", type=float, default=DEFAULT_MIN_SIZE, help="convert: min longest bbox dim (m)")
    parser.add_argument("--max-size", type=float, default=DEFAULT_MAX_SIZE, help="convert: max longest bbox dim (m)")
    parser.add_argument(
        "--min-thickness", type=float, default=DEFAULT_MIN_THICKNESS, help="convert: min shortest bbox dim (m)"
    )
    parser.add_argument(
        "--freeze-joints",
        action="store_true",
        help="convert: deactivate joints and articulation roots in wrapper layer so articulated packs run as rigid",
    )
    parser.add_argument("--check-dir", type=Path, help="ready: the --image-dir given to check_object.py")
    parser.add_argument("--beans", type=Path, help="ready: optional metrics.json of check_cup_holds_beans.py")
    parser.add_argument("--summary", type=Path, help="convert/ready: optional JSON path for summary report")
    parser.add_argument("--note", help="ready: how the checks were run (run dir, settle time, ...)")
    args = parser.parse_intermixed_args()

    ids = list(args.ids) + (
        [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
        if args.ids_file
        else []
    )

    if args.command == "survey":
        report = survey(args.lib, ids if ids else None)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if args.write_ids:
            raw_ids = [item["id"] for item in report["items"] if item["status"] == "raw_only"]
            args.write_ids.parent.mkdir(parents=True, exist_ok=True)
            args.write_ids.write_text("\n".join(raw_ids) + ("\n" if raw_ids else ""))
        print(f"[simready] survey total={report['total']} counts={json.dumps(report['counts'], sort_keys=True)}")
        return 0

    if not args.out:
        parser.error("--out is required for convert and ready")

    if args.all:
        root = args.out if args.command == "ready" else args.lib
        ids = sorted(set(ids) | set(discover_simready_ids(root)))
    if not ids:
        parser.error("no ids (pass ids, --ids-file, or --all)")

    if args.command == "ready":
        if not args.check_dir:
            parser.error("ready needs --check-dir")
        beans = json.loads(args.beans.read_text())["cups"] if args.beans else None
        results = {i: ready(args.out, i, args.check_dir, beans, args.note) for i in ids}
        passed_ids = [i for i, ok in results.items() if ok]
        failed_ids = [i for i, ok in results.items() if not ok]
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(
                json.dumps(
                    {
                        "total": len(results),
                        "passed": len(passed_ids),
                        "failed": len(failed_ids),
                        "not_ready": failed_ids,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"ready {len(passed_ids)}/{len(results)}; not ready: {failed_ids}")
        return 0 if not failed_ids else 1

    done, skipped, failed = [], [], {}
    for object_id in ids:
        if args.skip_existing and (
            survey_one(args.lib, object_id)["status"] == "already_ready"
            or survey_one(args.out, object_id)["status"] == "already_ready"
        ):
            skipped.append(object_id)
            continue
        try:
            done.append(
                convert_one(
                    args.lib,
                    object_id,
                    args.out,
                    orient=args.orient,
                    min_size=args.min_size,
                    max_size=args.max_size,
                    min_thickness=args.min_thickness,
                    freeze_joints=args.freeze_joints,
                )
            )
            print(f"[simready] {json.dumps(done[-1])}", flush=True)
        except (OSError, ValueError, RuntimeError) as exc:
            failed[object_id] = f"{type(exc).__name__}: {exc}"
            print(f"[simready] FAIL {object_id}: {failed[object_id]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "script": SCRIPT,
        "version": VERSION,
        "orient": args.orient,
        "converted": done,
        "skipped": skipped,
        "failed": failed,
    }
    summary_path = args.summary or (args.out / "simready_conversion.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[simready] converted {len(done)}, skipped {len(skipped)}, failed {len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
