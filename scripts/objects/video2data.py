"""Import NVIDIA video-to-data-object-assets (CC-BY-4.0, Einstar 3D scans) as RoboDojo rigid objects.

Upstream `nvidia/video-to-data-object-assets` on Hugging Face holds 69 real-world objects across three scan/recon
variants (`einstar`: 59 structured-light 3D scans, `bundlesdf`: 43, `sam3d`: 66). Each `einstar/` package contains
`output.usd` (`defaultPrim=/Asset`, Z-up, metres, `RigidBodyAPI`, `MassAPI`, `convexHull` collider, `@gltf/pbr.mdl@`),
`visual_asset.usd`, `textures/*.png`, and an optional `output.glb` (not needed by USD).

Pipeline:
    fetch    download variant files (excluding *.glb by default) from Hugging Face / HF mirror with retry/backoff:
             python scripts/objects/video2data.py fetch --out <source_dir> [--variant einstar] [--objects ...]
    wire     symlink (or copy) fetched packages into an Assets/Object tree (`RoboDojo/Rigid/video2data_<obj>/00000/`):
             python scripts/objects/video2data.py wire --source <source_dir> --lib <lib_object_dir> \
                 [--write-ids wired_ids.txt] [--out wire_manifest.json]
    convert  write `object.usd` wrapper (/root defaultPrim, Z-up, metres, bbox-centred, RigidBodyAPI + MassAPI,
             child bodies/masses/physics-material bindings stripped in wrapper layer) + `metadata.json`:
             isaac-sim python -u scripts/objects/video2data.py convert --lib <lib_object_dir> --out <ovl_object_dir> \
                 [--ids-file wired_ids.txt | --all] [--orient keep|auto|lay-flat]
    ready    after GPU floor-drop validation (`check_object.py --image-dir`), stamp `metadata.json` readiness:
             python scripts/objects/video2data.py ready --out <ovl_object_dir> --check-dir <check_dir> \
                 [--ids-file converted_ids.txt | --all] [--summary ready_summary.json]
"""

import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

try:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import simready  # noqa: E402
except ImportError:  # bare host python3 on Slurm nodes (fetch/wire/ready need stdlib only)
    np = None
    simready = None

VERSION = 1
SCRIPT = "scripts/objects/video2data.py"
REPO = "nvidia/video-to-data-object-assets"
REVISION = "835d343a573e3a872a37f6fe71773dbb2999ec6a"
URL = f"https://huggingface.co/datasets/{REPO}"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "NVIDIA video-to-data-object-assets (Einstar 3D scans), "
    "https://huggingface.co/datasets/nvidia/video-to-data-object-assets"
)
DEFAULT_VARIANT = "einstar"
VALID_VARIANTS = ("einstar", "bundlesdf", "sam3d")
ENTRY_USD = "output.usd"
USER_AGENT = "curl/8.0"
RETRY_CODES = {429, 500, 502, 503, 504}
DEFAULT_MIN_SIZE = 0.008
DEFAULT_MAX_SIZE = 2.5
DEFAULT_MIN_THICKNESS = 0.0005
MASS_BOUNDS_KG = (0.005, 0.5)
PLACE_KEYS = ("metadata.json", "object.usd")
SKIP_MIRROR_SUFFIXES = (".glb",)
TRANSIENT_NET_ERRORS = (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError)


def _object_uuid(object_id):
    if simready is not None:
        return simready.object_uuid(object_id)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/{object_id}"))


def _require_simready():
    if simready is None or np is None:
        raise RuntimeError(
            "numpy and simready are required for convert/mass estimation (run via `isaac-sim python -u`)"
        )
    return simready


class _CrossHostAuthStripRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Strip Authorization when redirecting to another host (e.g. HF/mirror -> S3/OSS presigned CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            old_host = urllib.parse.urlparse(req.full_url).netloc.lower()
            new_host = urllib.parse.urlparse(new_req.full_url).netloc.lower()
            if old_host != new_host:
                new_req.remove_header("Authorization")
        return new_req


_ORIG_URLOPEN = urllib.request.urlopen
_URL_OPENER = urllib.request.build_opener(_CrossHostAuthStripRedirectHandler())


def _open_url(req, *, timeout):
    if urllib.request.urlopen is not _ORIG_URLOPEN:
        return urllib.request.urlopen(req, timeout=timeout)
    return _URL_OPENER.open(req, timeout=timeout)


def sanitize_token(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()


def object_slug(obj_name, variant=DEFAULT_VARIANT):
    """Canonical RoboDojo category slug for a video-to-data object (`video2data_<obj>` for einstar)."""
    clean = sanitize_token(obj_name)
    if not clean:
        raise ValueError(f"invalid object name: {obj_name!r}")
    if variant == DEFAULT_VARIANT:
        return f"video2data_{clean}"
    return f"video2data_{sanitize_token(variant)}_{clean}"


def object_id_for(obj_name, variant=DEFAULT_VARIANT):
    return f"RoboDojo/Rigid/{object_slug(obj_name, variant)}/00000"


def discover_video2data_ids(root):
    """Find all `RoboDojo/Rigid/video2data_*/00000` object IDs under `root` (e.g. Assets/Object)."""
    base = Path(root) / "RoboDojo" / "Rigid"
    if not base.is_dir():
        return []
    ids = []
    for pkg_dir in sorted(base.iterdir()):
        if not pkg_dir.name.startswith("video2data_") or not pkg_dir.is_dir():
            continue
        inst = pkg_dir / "00000"
        if inst.is_dir():
            ids.append(f"RoboDojo/Rigid/{pkg_dir.name}/00000")
    return ids


def _next_link(header_val):
    if not header_val:
        return None
    for part in header_val.split(","):
        if 'rel="next"' in part:
            m = re.search(r"<([^>]+)>", part)
            if m:
                return m.group(1)
    return None


def _auth_headers(token=None):
    headers = {"User-Agent": USER_AGENT}
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _retry_wait(exc, attempt):
    retry_after = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if retry_after and str(retry_after).strip().isdigit():
        return min(60.0, float(retry_after))
    return min(60.0, 2.0**attempt)


def fetch_hf_tree(
    repo_id=REPO,
    revision=REVISION,
    *,
    endpoint=None,
    token=None,
    max_attempts=8,
    sleep_fn=time.sleep,
):
    """Fetch recursive file tree metadata from Hugging Face / mirror with retry/backoff on HTTP 429/5xx."""
    base = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").strip().rstrip("/")
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    url = f"{base}/api/datasets/{quoted_repo}/tree/{quoted_rev}?recursive=true"
    headers = _auth_headers(token)
    attempts = max(1, int(max_attempts))
    entries = []
    while url:
        req = urllib.request.Request(url, headers=headers)
        page = None
        nxt = None
        for attempt in range(attempts):
            try:
                with _open_url(req, timeout=60) as resp:
                    page = json.loads(resp.read().decode("utf-8"))
                    nxt = _next_link(resp.headers.get("Link"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in RETRY_CODES and attempt < attempts - 1:
                    sleep_fn(_retry_wait(exc, attempt))
                    continue
                raise
            except TRANSIENT_NET_ERRORS:
                if attempt < attempts - 1:
                    sleep_fn(min(60.0, 2.0**attempt))
                    continue
                raise
        if not isinstance(page, list):
            raise ValueError(f"unexpected HF tree payload type: {type(page).__name__}")
        for item in page:
            entries.append(
                {
                    "path": item.get("path", ""),
                    "size": int(item.get("size") or 0),
                    "type": item.get("type", "file"),
                }
            )
        url = urllib.parse.urljoin(url, nxt) if nxt else None
        if url:
            sleep_fn(0.25)
    return entries


def parse_variant_files(entries, *, variant=DEFAULT_VARIANT, include_glb=False, objects=None, limit=None):
    """Group HF tree entries into per-object file lists for `variant` (requiring `output.usd`)."""
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unsupported variant {variant!r}; expected one of {VALID_VARIANTS}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    obj_filter = {o.strip() for o in objects if o.strip()} if objects else None
    by_obj = {}
    root_docs = {}
    for entry in entries:
        if entry.get("type", "file") == "directory":
            continue
        path = str(entry.get("path") or "").lstrip("/")
        if not path:
            continue
        size = int(entry.get("size") or 0)
        parts = path.split("/")
        if len(parts) == 1 and parts[0].lower() in ("readme.md", "license", "license.md", "license.txt"):
            root_docs[path] = {"path": path, "size": size}
            continue
        if len(parts) < 3 or parts[1] != variant:
            continue
        obj_name = parts[0]
        if obj_filter is not None and obj_name not in obj_filter:
            continue
        if not include_glb and path.lower().endswith(".glb"):
            continue
        by_obj.setdefault(obj_name, {})[path] = {"path": path, "size": size}

    packages = []
    for obj_name in sorted(by_obj):
        files = [by_obj[obj_name][p] for p in sorted(by_obj[obj_name])]
        rel_names = {item["path"].split("/", 2)[2] for item in files}
        if ENTRY_USD not in rel_names:
            continue
        packages.append(
            {
                "object": obj_name,
                "variant": variant,
                "slug": object_slug(obj_name, variant),
                "id": object_id_for(obj_name, variant),
                "files": files,
                "size_bytes": sum(item["size"] for item in files),
            }
        )
    if limit is not None:
        packages = packages[:limit]
    return packages, [root_docs[p] for p in sorted(root_docs)]


def download_file(
    url,
    dest,
    *,
    expected_size=None,
    token=None,
    force=False,
    max_attempts=8,
    sleep_fn=time.sleep,
):
    """Download a single URL atomically via `.part` temp file with retry/backoff on HTTP 429/5xx."""
    dest = Path(dest)
    if not force and dest.is_file():
        if expected_size is None or expected_size <= 0 or dest.stat().st_size == expected_size:
            return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers = _auth_headers(token)
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers=headers)
        try:
            with _open_url(req, timeout=120) as resp, part.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1024 * 1024)
            if expected_size and expected_size > 0 and part.stat().st_size != expected_size:
                raise OSError(f"size mismatch: got {part.stat().st_size}, expected {expected_size}")
            os.replace(part, dest)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code in RETRY_CODES and attempt < attempts - 1:
                sleep_fn(_retry_wait(exc, attempt))
                continue
            raise
        except TRANSIENT_NET_ERRORS:
            part.unlink(missing_ok=True)
            if attempt < attempts - 1:
                sleep_fn(min(60.0, 2.0**attempt))
                continue
            raise
    raise RuntimeError(f"failed to download {url}")


def fetch_dataset(
    out_dir,
    *,
    repo_id=REPO,
    revision=REVISION,
    variant=DEFAULT_VARIANT,
    include_glb=False,
    objects=None,
    limit=None,
    endpoint=None,
    token=None,
    tree_entries=None,
    force=False,
    sleep_fn=time.sleep,
):
    """Fetch variant packages from HF / mirror into `out_dir` and write `UPSTREAM.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = (
        tree_entries
        if tree_entries is not None
        else fetch_hf_tree(repo_id, revision, endpoint=endpoint, token=token, sleep_fn=sleep_fn)
    )
    packages, root_docs = parse_variant_files(
        entries,
        variant=variant,
        include_glb=include_glb,
        objects=objects,
        limit=limit,
    )
    base = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")

    downloaded, skipped, failed = 0, 0, {}
    for doc in root_docs:
        rel = doc["path"]
        url = f"{base}/datasets/{quoted_repo}/resolve/{quoted_rev}/{urllib.parse.quote(rel, safe='/')}"
        try:
            download_file(url, out_dir / rel, expected_size=doc["size"], token=token, force=force, sleep_fn=sleep_fn)
        except Exception:
            pass

    for pkg in packages:
        for item in pkg["files"]:
            rel = item["path"]
            url = f"{base}/datasets/{quoted_repo}/resolve/{quoted_rev}/{urllib.parse.quote(rel, safe='/')}"
            try:
                status = download_file(
                    url,
                    out_dir / rel,
                    expected_size=item["size"],
                    token=token,
                    force=force,
                    sleep_fn=sleep_fn,
                )
                if status == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed[rel] = f"{type(exc).__name__}: {exc}"

    manifest = {
        "script": SCRIPT,
        "version": VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "variant": variant,
        "include_glb": include_glb,
        "object_count": len(packages),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "failed_files": failed,
        "total_bytes": sum(p["size_bytes"] for p in packages),
        "objects": [p["object"] for p in packages],
    }
    (out_dir / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _remove_dest(dest):
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)


def mirror_package(src, target, *, copy=False, include_glb=False):
    """Mirror a video-to-data package dir (excluding object.usd, metadata.json, *.part, and optionally *.glb)."""
    src, target = Path(src), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if src.resolve() == target.resolve():
        return
    for entry in sorted(src.iterdir()):
        if entry.name in PLACE_KEYS or entry.suffix.lower() == ".part":
            continue
        if not include_glb and entry.suffix.lower() in SKIP_MIRROR_SUFFIXES:
            continue
        dest = target / entry.name
        _remove_dest(dest)
        if copy:
            if entry.is_dir():
                shutil.copytree(entry, dest)
            else:
                shutil.copy2(entry, dest)
        elif entry.is_symlink():
            link_target = os.readlink(entry)
            if not os.path.isabs(link_target):
                link_target = os.path.abspath(os.path.join(str(src), link_target))
            dest.symlink_to(link_target)
        else:
            dest.symlink_to(entry.resolve())


def wire_packages(source_dir, lib_dir, *, variant=DEFAULT_VARIANT, objects=None, copy=False, include_glb=False):
    """Wire fetched `<source>/<obj>/<variant>/` packages into `<lib>/RoboDojo/Rigid/video2data_<obj>/00000/`."""
    source_dir, lib_dir = Path(source_dir), Path(lib_dir)
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unsupported variant {variant!r}; expected one of {VALID_VARIANTS}")
    upstream_meta = {}
    upstream_path = source_dir / "UPSTREAM.json"
    if upstream_path.is_file():
        try:
            upstream_meta = json.loads(upstream_path.read_text())
        except (OSError, json.JSONDecodeError):
            upstream_meta = {}
    obj_filter = {o.strip() for o in objects if o.strip()} if objects else None
    wired, skipped = [], []
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        obj_name = child.name
        if obj_filter is not None and obj_name not in obj_filter:
            continue
        pkg_src = child / variant
        if not pkg_src.is_dir():
            continue
        if not (pkg_src / ENTRY_USD).is_file():
            skipped.append({"object": obj_name, "reason": f"missing {ENTRY_USD}"})
            continue
        oid = object_id_for(obj_name, variant)
        target = lib_dir / oid
        mirror_package(pkg_src, target, copy=copy, include_glb=include_glb)
        meta_path = target / "metadata.json"
        if not meta_path.is_file():
            seed_meta = {
                "uuid": _object_uuid(oid),
                "source": {
                    "dataset": "video-to-data-object-assets",
                    "repo_id": upstream_meta.get("repo_id", REPO),
                    "revision": upstream_meta.get("revision", REVISION),
                    "url": upstream_meta.get("url", URL),
                    "license": LICENSE,
                    "attribution": ATTRIBUTION,
                    "object_name": obj_name,
                    "variant": variant,
                    "source_path": f"{obj_name}/{variant}/{ENTRY_USD}",
                },
                "runtime_ready": False,
            }
            meta_path.write_text(json.dumps(seed_meta, indent=2, ensure_ascii=False) + "\n")
        wired.append(
            {
                "id": oid,
                "object": obj_name,
                "variant": variant,
                "source": str(pkg_src),
                "target": str(target),
            }
        )
    return {
        "script": SCRIPT,
        "version": VERSION,
        "source": str(source_dir),
        "lib": str(lib_dir),
        "variant": variant,
        "wired_count": len(wired),
        "skipped_count": len(skipped),
        "wired": wired,
        "skipped": skipped,
    }


def resolve_mass(authored_masses, pts, tris, size, densities, kind, mass_bounds=MASS_BOUNDS_KG):
    """Prefer positive finite authored `MassAPI` mass (clamped to `mass_bounds`), else `simready.estimate_mass`."""
    valid = [float(m) for m in (authored_masses or []) if m is not None and math.isfinite(float(m)) and float(m) > 0]
    if valid:
        raw = sum(valid)
        lo, hi = mass_bounds
        mass = min(max(raw, lo), hi)
        return round(mass, 4), {
            "rule": "authored UsdPhysics.MassAPI mass, clamped to loader rigid-object bounds",
            "source": "authored_mass_api",
            "material": kind,
            "raw_kg": round(raw, 4),
            "range_kg": [lo, hi],
            "clamped": mass != raw,
            "measured": False,
        }
    return _require_simready().estimate_mass(pts, tris, size, densities, kind)


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


def _is_already_ready(root, object_id):
    target = Path(root) / object_id
    meta_path = target / "metadata.json"
    if not ((target / "object.usd").is_file() or (target / "object.usdz").is_file()) or not meta_path.is_file():
        return False
    try:
        return bool(json.loads(meta_path.read_text()).get("runtime_ready"))
    except (OSError, json.JSONDecodeError):
        return False


def convert_one(
    lib,
    object_id,
    out,
    *,
    orient="keep",
    min_size=DEFAULT_MIN_SIZE,
    max_size=DEFAULT_MAX_SIZE,
    min_thickness=DEFAULT_MIN_THICKNESS,
    freeze_joints=False,
):
    """Write `<out>/<object_id>/object.usd` and `metadata.json` from `<lib>/<object_id>/output.usd`."""
    _require_simready()
    Gf, _, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = pxr_modules()
    src, target = Path(lib) / object_id, Path(out) / object_id
    entry_usd = src / ENTRY_USD
    if not entry_usd.is_file():
        raise ValueError(f"{src}: missing {ENTRY_USD}")
    meta_path = src / "metadata.json"
    meta_in = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    original = Usd.Stage.Open(str(entry_usd))
    if not original or not original.GetDefaultPrim():
        raise ValueError("package has no default prim")

    frozen_joints = []
    authored_masses = []
    for prim in original.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if not freeze_joints:
                raise ValueError(f"articulated package: {prim.GetPath()}")
            frozen_joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.MassAPI):
            mval = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mval is not None:
                authored_masses.append(float(mval))
    axis, unit = UsdGeom.GetStageUpAxis(original), UsdGeom.GetStageMetersPerUnit(original)

    if target.resolve() != src.resolve():
        mirror_package(src, target, copy=False, include_glb=False)
    wrapper = target / "object.usd"
    _remove_dest(wrapper)
    try:
        return _build_wrapper(
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            UsdUtils=UsdUtils,
            src=src,
            target=target,
            wrapper=wrapper,
            object_id=object_id,
            original=original,
            meta_in=meta_in,
            axis=axis,
            unit=unit,
            frozen_joints=frozen_joints,
            authored_masses=authored_masses,
            orient=orient,
            min_size=min_size,
            max_size=max_size,
            min_thickness=min_thickness,
            freeze_joints=freeze_joints,
        )
    except Exception:
        wrapper.unlink(missing_ok=True)
        raise


def _build_wrapper(
    *,
    Gf,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    UsdUtils,
    src,
    target,
    wrapper,
    object_id,
    original,
    meta_in,
    axis,
    unit,
    frozen_joints,
    authored_masses,
    orient,
    min_size,
    max_size,
    min_thickness,
    freeze_joints,
):
    stage = Usd.Stage.Open(Sdf.Layer.CreateNew(str(wrapper), args={"format": "usda"}))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    normalize = UsdGeom.Xform.Define(stage, "/root/normalize")
    asset = UsdGeom.Xform.Define(stage, "/root/normalize/asset").GetPrim()
    asset.GetReferences().AddReference(f"./{ENTRY_USD}")
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
            faces = simready.triangles(counts, indices)
            if len(faces) == 0:
                continue
            tris.append(faces + offset)
            pts.append(local @ tf[:3, :3] + tf[3, :3])
            offset += len(local)
        if not pts:
            raise ValueError("no visible mesh")
        return np.concatenate(pts), np.concatenate(tris)

    pre_pts, _ = mesh_points()
    q, orient_reason = simready.stable_orientation(np.ptp(pre_pts, axis=0), src.parent.name, mode=orient)
    q_norm = math.sqrt(sum(c * c for c in q))
    q = [c / q_norm for c in q]
    orient_op.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))

    pts, _ = mesh_points()
    lo, hi = pts.min(0), pts.max(0)
    center, size = (lo + hi) / 2, hi - lo
    tabletop_ok = simready.check_bounds(size, min_size=min_size, max_size=max_size, min_thickness=min_thickness)
    shift.Set(Gf.Vec3d(*map(float, -center)))
    pts, tris = mesh_points()
    if np.abs((pts.min(0) + pts.max(0)) / 2).max() > 1e-5:
        raise ValueError("origin is not the bounding-box centre")

    physics_materials = [p for p in original.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]
    if not materials:
        materials = {p.GetName() for p in physics_materials}
    bound = [p for p in physics_materials if p.GetName() in materials]
    densities = sorted({float(UsdPhysics.MaterialAPI(p).GetDensityAttr().Get() or 0) for p in bound} - {0.0})
    kind = simready.material_kind(materials, src.parent.name, default="default")
    mass, mass_source = resolve_mass(authored_masses, pts, tris, size, densities, kind)
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
    if missing := simready.unresolved_dependencies(unresolved):
        raise ValueError(f"unresolved dependencies: {missing}")

    geometry, active = simready.geometry_metadata(pts, len(tris))
    source = dict(meta_in.get("source") or {})
    source.setdefault("dataset", "video-to-data-object-assets")
    source.setdefault("repo_id", REPO)
    source.setdefault("revision", REVISION)
    source.setdefault("url", URL)
    source.setdefault("license", LICENSE)
    source.setdefault("attribution", ATTRIBUTION)
    metadata = {
        "uuid": simready.object_uuid(object_id),
        "physics": {"mass": mass, "friction": simready.FRICTION, "size": simready.r4(size)},
        "geometry": geometry,
        "active": active,
        "passive": {"functional": {}, "support": {}},
        "source": source,
        "adapter": {
            "script": SCRIPT,
            "version": VERSION,
            "wrapper": "object.usd",
            "reference": f"./{ENTRY_USD}",
            "package_axis": {"up": str(axis), "meters_per_unit": unit},
            "orient": simready.r4(q),
            "orient_reason": orient_reason,
            "offset_m": simready.r4(-center),
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
        "mass_source": mass_source.get("source", "estimated"),
        "size_m": simready.r4(size),
        "material": kind,
        "orient_reason": orient_reason,
        "tabletop_range": tabletop_ok,
    }


def ready(out, object_id, check_dir, note=None):
    """Record check_object.py floor-drop results into metadata.json and set runtime_ready."""
    if simready is not None:
        return simready.ready(out, object_id, check_dir, beans=None, note=note)
    target = Path(out) / object_id
    meta_path = target / "metadata.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    kind, category, index = object_id.split("/")[-3:]
    check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
    check = json.loads(check_file.read_text()) if check_file.is_file() else None
    passed = bool(check and check.get("settle") == "pass")
    meta.setdefault("adapter", {})["validation"] = {
        "check_object": check and {k: check.get(k) for k in ("settle", "tilt_deg", "z_m", "expect_z_m", "checked_at")},
        "check_cup_holds_beans": None,
        "passed": passed,
        "note": note,
    }
    meta["runtime_ready"] = passed
    meta.setdefault("source", {})["runtime_validated"] = passed
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return passed


def _read_ids(cli_ids, ids_file):
    ids = list(cli_ids or [])
    if ids_file:
        ids.extend(
            line.strip()
            for line in Path(ids_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "wire", "convert", "ready"))
    parser.add_argument("ids", nargs="*", help="convert/ready: object ids relative to Assets/Object")
    parser.add_argument("--ids-file", type=Path, help="convert/ready: file listing object ids, one per line")
    parser.add_argument("--all", action="store_true", help="convert/ready: discover all video2data_* ids")
    parser.add_argument("--repo", default=REPO, help="fetch: HF dataset repo id")
    parser.add_argument("--revision", default=REVISION, help="fetch: HF dataset revision SHA/branch")
    parser.add_argument("--endpoint", help="fetch: HF endpoint override (default: $HF_ENDPOINT or huggingface.co)")
    parser.add_argument("--tree-json", type=Path, help="fetch: read pre-saved HF tree JSON instead of querying network")
    parser.add_argument("--variant", choices=VALID_VARIANTS, default=DEFAULT_VARIANT, help="fetch/wire: scan variant")
    parser.add_argument("--objects", help="fetch/wire: comma-separated subset of upstream object names")
    parser.add_argument("--limit", type=int, help="fetch: max number of objects to download")
    parser.add_argument("--include-glb", action="store_true", help="fetch/wire: also include *.glb meshes")
    parser.add_argument("--force", action="store_true", help="fetch: re-download existing files")
    parser.add_argument("--source", type=Path, help="wire: fetched dataset source dir")
    parser.add_argument("--lib", type=Path, default=Path("Assets/Object"), help="wire/convert: Assets/Object dir")
    parser.add_argument("--copy", action="store_true", help="wire: copy package files instead of symlinking")
    parser.add_argument("--write-ids", type=Path, help="wire: write wired object ids (one per line)")
    parser.add_argument("--out", type=Path, help="fetch: source dir; wire: manifest JSON; convert/ready: ovl dir")
    parser.add_argument("--skip-existing", action="store_true", help="convert: skip ids already runtime_ready=true")
    parser.add_argument("--orient", choices=("keep", "auto", "lay-flat"), default="keep", help="convert: orientation")
    parser.add_argument("--min-size", type=float, default=DEFAULT_MIN_SIZE, help="convert: min longest bbox dim (m)")
    parser.add_argument("--max-size", type=float, default=DEFAULT_MAX_SIZE, help="convert: max longest bbox dim (m)")
    parser.add_argument(
        "--min-thickness", type=float, default=DEFAULT_MIN_THICKNESS, help="convert: min shortest bbox dim (m)"
    )
    parser.add_argument(
        "--freeze-joints",
        action="store_true",
        help="convert: deactivate joints and articulation roots in wrapper layer",
    )
    parser.add_argument("--check-dir", type=Path, help="ready: --image-dir passed to check_object.py")
    parser.add_argument("--summary", type=Path, help="convert/ready: optional JSON summary output path")
    parser.add_argument("--note", help="ready: validation note recorded in metadata.json")
    args = parser.parse_intermixed_args(argv)

    obj_list = [o.strip() for o in args.objects.split(",") if o.strip()] if args.objects else None

    if args.command == "fetch":
        if not args.out:
            parser.error("fetch requires --out")
        tree_entries = json.loads(args.tree_json.read_text()) if args.tree_json else None
        manifest = fetch_dataset(
            args.out,
            repo_id=args.repo,
            revision=args.revision,
            variant=args.variant,
            include_glb=args.include_glb,
            objects=obj_list,
            limit=args.limit,
            endpoint=args.endpoint,
            tree_entries=tree_entries,
            force=args.force,
        )
        mb = manifest["total_bytes"] / (1024**2)
        print(
            f"[video2data] fetched variant={args.variant} objects={manifest['object_count']} "
            f"downloaded={manifest['downloaded_files']} skipped={manifest['skipped_files']} "
            f"failed={len(manifest['failed_files'])} ({mb:.1f} MiB) -> {args.out}",
            flush=True,
        )
        return 1 if manifest["failed_files"] else 0

    if args.command == "wire":
        if not args.source:
            parser.error("wire requires --source")
        report = wire_packages(
            args.source,
            args.lib,
            variant=args.variant,
            objects=obj_list,
            copy=args.copy,
            include_glb=args.include_glb,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if args.write_ids:
            wired_ids = [item["id"] for item in report["wired"]]
            args.write_ids.parent.mkdir(parents=True, exist_ok=True)
            args.write_ids.write_text("\n".join(wired_ids) + ("\n" if wired_ids else ""))
        print(
            f"[video2data] wired={report['wired_count']} skipped={report['skipped_count']} -> {args.lib}",
            flush=True,
        )
        return 0

    if not args.out:
        parser.error("--out is required for convert and ready")

    ids = _read_ids(args.ids, args.ids_file)
    if args.all:
        root = args.out if args.command == "ready" else args.lib
        ids = sorted(set(ids) | set(discover_video2data_ids(root)))
    if not ids:
        parser.error("no ids (pass ids, --ids-file, or --all)")

    if args.command == "ready":
        if not args.check_dir:
            parser.error("ready requires --check-dir")
        results = {oid: ready(args.out, oid, args.check_dir, note=args.note) for oid in ids}
        passed_ids = [oid for oid, ok in results.items() if ok]
        failed_ids = [oid for oid, ok in results.items() if not ok]
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(
                json.dumps(
                    {
                        "total": len(results),
                        "passed": len(passed_ids),
                        "failed": len(failed_ids),
                        "passed_ids": passed_ids,
                        "not_ready": failed_ids,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"[video2data] ready {len(passed_ids)}/{len(results)}; not ready: {failed_ids}", flush=True)
        return 0 if not failed_ids else 1

    done, skipped, failed = [], [], {}
    for oid in ids:
        if args.skip_existing and (_is_already_ready(args.lib, oid) or _is_already_ready(args.out, oid)):
            skipped.append(oid)
            continue
        try:
            done.append(
                convert_one(
                    args.lib,
                    oid,
                    args.out,
                    orient=args.orient,
                    min_size=args.min_size,
                    max_size=args.max_size,
                    min_thickness=args.min_thickness,
                    freeze_joints=args.freeze_joints,
                )
            )
            print(f"[video2data] {json.dumps(done[-1])}", flush=True)
        except (OSError, ValueError, RuntimeError) as exc:
            failed[oid] = f"{type(exc).__name__}: {exc}"
            print(f"[video2data] FAIL {oid}: {failed[oid]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "script": SCRIPT,
        "version": VERSION,
        "orient": args.orient,
        "converted": done,
        "skipped": skipped,
        "failed": failed,
    }
    summary_path = args.summary or (args.out / "video2data_conversion.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[video2data] converted {len(done)}, skipped {len(skipped)}, failed {len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
