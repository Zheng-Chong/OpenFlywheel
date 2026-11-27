"""Import MolmoSpaces THOR objects (`allenai/molmospaces`, `isaac/objects/thor/20260128`, CC BY 4.0) into RoboDojo.

MolmoSpaces THOR ships 1,667 native Z-up, meters Isaac OpenUSD `.usda` assets (330 articulated + 1,337 rigid)
packed as `.tar.zst` archives inside `isaac/objects/thor/20260128/shards/00000.tar` (1.10 GB total), indexed by:
  - `arrow_table.json` (`path`, `shard_id`, `offset`, `size`, `inflated_size` for HTTP Range fetching)
  - `mjthor_resources_combined_meta.json.gz` (maps each `.tar.zst` archive to its contained `<asset_id>/` trees)
  - `thor_usd_assets_metadata.json.tar.zst` -> `usd_assets_metadata.json` (`asset_id`, `hash_id`, `articulated`)

Why a dedicated adapter instead of `articulated.write_layer` directly:
  1. THOR nests all links under `/<Name>/Geometry/*` and places `ArticulationRootAPI` on `/<Name>/Geometry/<Name>`
     with a dummy `physics:mass = 1e-8` kg (and child links often have `1e-8` kg or no authored `physics:mass`),
     whereas `ArticulationObject.fix_root_link()` requires `ArticulationRootAPI` on `/root` and the base `Xform`
     with `RigidBodyAPI` as the first direct child of `/root`.
  2. THOR's MJCF-USD converter authors joints in child->parent order (`body0 = child`, `body1 = parent`), which
     inverts `articulated.inspect()` base detection and part-role classification. We canonicalize joints along the
     spanning tree from the true root (`body0 = parent`, `body1 = child`, swapping `localPos`/`localRot` and negating
     + swapping joint limits) so physical motion is preserved identically while matching USD/RoboDojo conventions.
  3. After flattening and reparenting `/<Name>/Geometry/* -> /root/*` and renaming `/<Name> -> /root`, USD 24.x
     (Isaac Sim 5.1) `Sdf.Layer.Apply(BatchNamespaceEdit)` does not reliably retarget `RelationshipSpec.targetPathList`
     or `AttributeSpec.connectionPathList` (e.g. `UsdPreviewSurface` / `UsdUVTexture` `.connect` paths); we rewrite
     both explicitly.
  4. THOR articulated joints have `drives={}` (zero drives and zero friction). We replace dummy `<= 1e-5 kg` masses
     with bbox-volume-proportional per-link masses (>= `0.02 kg`) and author `UsdPhysics.DriveAPI` +
     `PhysxJointAxisAPI` (`staticFrictionEffort`, `dynamicFrictionEffort`, `armature`).
  5. THOR `.usda` files also include a sibling top-level `PhysicsScene` prim which is stripped on export.

Pipeline subcommands:
  filter   Fetch/load the THOR index and filter by physics_type, categories, asset_ids, limit (pure stdlib).
  fetch    Range-fetch (or slice from local `shards/00000.tar`) only selected `<asset_id>/` directories (pure stdlib).
  survey   Inspect downloaded `.usda` assets with `pxr` and emit a conversion manifest with deterministic IDs.
  convert  Flatten, reparent, rewrite rels/connections, canonicalize joints, assign masses/drives -> `object.usd`.
  ready    Stamp `check_articulation.py` / `check_object.py` results into `metadata.json` (pure stdlib).

Needs `pxr` (`usd-core` or `isaac-sim python` with `SimulationApp` fallback) only for `survey` and `convert`.
CPU unit tests: `scripts/objects/test_molmospaces_thor.py`.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import articulated  # noqa: E402

SCRIPT = "scripts/objects/molmospaces_thor.py"
VERSION = 1

REPO = "allenai/molmospaces"
SUBPATH = "isaac/objects/thor/20260128"
DEFAULT_REVISION = "main"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "curl/8.0"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "MolmoSpaces THOR (AI2-THOR), Allen Institute for AI, huggingface.co/datasets/allenai/molmospaces"

METADATA_ARCHIVE = "thor_usd_assets_metadata.json.tar.zst"
ARROW_TABLE_FILE = "arrow_table.json"
COMBINED_META_FILE = "mjthor_resources_combined_meta.json.gz"

# Size, mass, drive, and friction bounds (metres, kg, N*m / N).
TABLETOP_MAX_M = 0.65
TABLESIDE_MAX_M = 0.90
MIN_THICKNESS_M = 0.003
DUMMY_MASS_THRESHOLD_KG = 1e-5  # THOR authors 1e-8 kg placeholder masses on root/lever/handle links.
ARTICULATED_SHELL_DENSITY = 120.0  # kg/m^3 effective bulk density for hollow appliances / furniture.
RIGID_ESTIMATED_DENSITY = 500.0  # kg/m^3 effective density for tabletop rigid objects.
DEFAULT_MIN_ARTICULATED_MASS_KG = 0.05
DEFAULT_MAX_TABLETOP_MASS_KG = 15.0
DEFAULT_MAX_TABLESIDE_MASS_KG = 25.0
DEFAULT_MAX_PORTABLE_MASS_KG = 3.0
DEFAULT_MIN_RIGID_MASS_KG = 0.005
DEFAULT_MAX_RIGID_MASS_KG = 0.5
MIN_LINK_MASS_KG = articulated.MIN_LINK_MASS
RIGID_FRICTION = 0.6

DEFAULT_ANGULAR_DAMPING = 0.05  # N*m/(deg/s) viscous damping for revolute drives (stiffness=0).
DEFAULT_LINEAR_DAMPING = 2.0  # N/(m/s) viscous damping for prismatic drives (stiffness=0).
DEFAULT_ANGULAR_ARMATURE = 0.01  # kg*m^2 (matches robocasa UNIT_ARMATURE).
DEFAULT_LINEAR_ARMATURE = 0.01  # kg.
MIN_LEVEL_FRICTION = 0.05  # N*m floor for horizontal hinges.

PORTABLE_TOKENS = (
    "alarm",
    "bag",
    "basket",
    "book",
    "bottle",
    "bowl",
    "box",
    "bread",
    "candle",
    "cd",
    "cellphone",
    "clock",
    "container",
    "creditcard",
    "cup",
    "dish",
    "folder",
    "fork",
    "glass",
    "jar",
    "kettle",
    "keychain",
    "knife",
    "ladle",
    "laptop",
    "mug",
    "newspaper",
    "notebook",
    "pan",
    "pen",
    "pencil",
    "pillow",
    "plate",
    "plunger",
    "pot",
    "remote",
    "saltshaker",
    "soap",
    "spatula",
    "spoon",
    "spray",
    "statue",
    "tissue",
    "towel",
    "tray",
    "vase",
    "watch",
    "wateringcan",
    "wine",
)

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def hf_endpoint(endpoint: str | None = None) -> str:
    raw = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    return raw.rstrip("/")


def dataset_url(endpoint: str | None = None) -> str:
    return f"{hf_endpoint(endpoint)}/datasets/{REPO}"


def canonical_url() -> str:
    return f"{DEFAULT_HF_ENDPOINT}/datasets/{REPO}"


def resolve_url(rel_path: str, *, revision: str = DEFAULT_REVISION, endpoint: str | None = None) -> str:
    quoted = urllib.parse.quote(f"{SUBPATH}/{rel_path.lstrip('/')}", safe="/")
    return f"{dataset_url(endpoint)}/resolve/{revision}/{quoted}"


def pxr_modules():
    """Import pxr modules, falling back to headless Isaac SimulationApp when standalone usd-core is absent."""
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        globals()["APP"] = SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


def http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    attempts: int = 6,
    timeout: float = 120.0,
    backoff: float = 2.0,
    user_agent: str = USER_AGENT,
    sleep_fn=time.sleep,
) -> tuple[bytes, dict[str, str]]:
    """GET `url` with `User-Agent: curl/8.0` and exponential backoff on HTTP 429/5xx and transient network errors."""
    req_headers = {"User-Agent": user_agent}
    if headers:
        req_headers.update(headers)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.read(), resp_headers
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in RETRYABLE_HTTP_CODES or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = backoff * (2**attempt)
            if retry_after:
                try:
                    delay = max(delay, min(float(retry_after), 60.0))
                except ValueError:
                    pass
            sleep_fn(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            sleep_fn(backoff * (2**attempt))
    assert last_exc is not None
    raise last_exc


def decompress_zst(data: bytes) -> bytes:
    """Decompress a `.zst` payload using `zstandard` if available, else the `zstd` CLI."""
    try:
        import zstandard

        return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    except ImportError:
        pass
    proc = subprocess.run(["zstd", "-dc"], input=data, capture_output=True, check=True)
    return proc.stdout


def category_from_asset_id(asset_id: str) -> str:
    """Derive THOR category from asset_id (e.g. `Side_Table_320_1` -> `Side_Table`, `Microwave_4` -> `Microwave`)."""
    stripped = re.sub(r"(?:_\d+)+$", "", asset_id.strip())
    return stripped or asset_id.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically via a unique sibling temp file + `os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _has_textures_in_tree(node: object) -> bool:
    if not isinstance(node, dict):
        return False
    for k, v in node.items():
        kl = k.lower()
        if kl.endswith((".png", ".jpg", ".jpeg", ".tga", ".exr", ".webp")):
            return True
        if isinstance(v, dict) and _has_textures_in_tree(v):
            return True
    return False


def build_catalog(
    usd_metadata: dict[str, dict],
    arrow_table: list[dict],
    combined_meta: dict[str, dict],
) -> list[dict]:
    """Combine THOR's 3 index files into a normalized, deterministically sorted list of 1,667 asset entries."""
    by_archive = {e["path"]: e for e in arrow_table if isinstance(e, dict) and "path" in e}
    asset_to_archive: dict[str, str] = {}
    asset_has_textures: dict[str, bool] = {}

    for archive_name, tree in combined_meta.items():
        if archive_name == METADATA_ARCHIVE or not isinstance(tree, dict):
            continue
        for top_key, subtree in tree.items():
            if top_key in usd_metadata and top_key not in asset_to_archive:
                asset_to_archive[top_key] = archive_name
                asset_has_textures[top_key] = _has_textures_in_tree(subtree)

    rows: list[dict] = []
    for asset_id, meta in usd_metadata.items():
        category = category_from_asset_id(asset_id)
        is_art = bool(meta.get("articulated"))
        archive = asset_to_archive.get(asset_id, "")
        arrow = by_archive.get(archive, {})
        rows.append(
            {
                "asset_id": asset_id,
                "hash_id": str(meta.get("hash_id") or ""),
                "category": category,
                "category_slug": articulated.slug(category) or "object",
                "articulated": is_art,
                "physics_type": "articulated" if is_art else "rigid",
                "archive": archive,
                "shard_id": int(arrow.get("shard_id", 0)) if arrow else 0,
                "offset": int(arrow.get("offset", -1)) if arrow else -1,
                "size": int(arrow.get("size", 0)) if arrow else 0,
                "has_textures": asset_has_textures.get(asset_id, False),
            }
        )
    rows.sort(key=lambda r: (r["physics_type"], r["category_slug"], r["asset_id"]))
    return rows


def fetch_index(
    sources_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    force: bool = False,
) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    """Fetch or reuse cached `arrow_table.json`, `mjthor_resources_combined_meta.json.gz`, and metadata JSON."""
    idx_dir = Path(sources_dir) / "_index"
    idx_dir.mkdir(parents=True, exist_ok=True)

    arrow_path = idx_dir / ARROW_TABLE_FILE
    if force or not arrow_path.is_file():
        data, _ = http_get(resolve_url(ARROW_TABLE_FILE, revision=revision, endpoint=endpoint))
        _atomic_write_bytes(arrow_path, data)
    arrow_table = json.loads(arrow_path.read_text(encoding="utf-8"))

    cmeta_path = idx_dir / COMBINED_META_FILE
    if force or not cmeta_path.is_file():
        data, _ = http_get(resolve_url(COMBINED_META_FILE, revision=revision, endpoint=endpoint))
        _atomic_write_bytes(cmeta_path, data)
    combined_meta = json.loads(gzip.decompress(cmeta_path.read_bytes()).decode("utf-8"))

    umeta_path = idx_dir / "usd_assets_metadata.json"
    if force or not umeta_path.is_file():
        entry = next((e for e in arrow_table if e.get("path") == METADATA_ARCHIVE), None)
        if entry is None:
            raise ValueError(f"{METADATA_ARCHIVE} not found in {ARROW_TABLE_FILE}")
        raw_zst = _read_archive_bytes(Path(sources_dir), entry, revision=revision, endpoint=endpoint)
        tar_bytes = decompress_zst(raw_zst)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith("usd_assets_metadata.json")), None)
            if member is None:
                raise ValueError(f"usd_assets_metadata.json missing inside {METADATA_ARCHIVE}")
            extracted = tf.extractfile(member)
            assert extracted is not None
            _atomic_write_bytes(umeta_path, extracted.read())
    usd_metadata = json.loads(umeta_path.read_text(encoding="utf-8"))
    return usd_metadata, arrow_table, combined_meta


def _read_archive_bytes(
    sources_dir: Path,
    arrow_entry: dict,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
) -> bytes:
    """Read a `.tar.zst` archive slice from local `shards/00000.tar` if present, else via HTTP Range GET."""
    shard_id = int(arrow_entry.get("shard_id", 0))
    offset = int(arrow_entry["offset"])
    size = int(arrow_entry["size"])
    if offset < 0 or size <= 0:
        raise ValueError(f"invalid byte range for {arrow_entry.get('path')}: offset={offset}, size={size}")

    local_shard = Path(sources_dir) / "shards" / f"{shard_id:05d}.tar"
    if local_shard.is_file() and local_shard.stat().st_size >= offset + size:
        with local_shard.open("rb") as fh:
            fh.seek(offset)
            return fh.read(size)

    shard_rel = f"shards/{shard_id:05d}.tar"
    url = resolve_url(shard_rel, revision=revision, endpoint=endpoint)
    data, _ = http_get(url, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
    if len(data) != size:
        # Some mirrors ignore Range and return the entire shard; slice defensively if so.
        if len(data) >= offset + size:
            return data[offset : offset + size]
        raise ValueError(f"short Range response for {arrow_entry.get('path')}: got {len(data)} bytes, expected {size}")
    return data


def _matches_any(text: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    low = text.lower()
    for pat in patterns:
        if pat.lower() in low or re.search(pat, text, re.IGNORECASE):
            return True
    return False


def filter_catalog(
    rows: list[dict],
    *,
    physics_type: str = "articulated",
    categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
    textured_only: bool = False,
    max_per_category: int | None = None,
    limit: int | None = None,
    asset_ids: set[str] | None = None,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Filter THOR catalog rows before downloading and assign deterministic global RoboDojo IDs."""
    ordered = sorted(rows, key=lambda r: (r["physics_type"], r["category_slug"], r["asset_id"]))
    chosen: list[dict] = []
    dropped: list[dict] = []
    per_cat: dict[tuple[str, str], int] = {}
    id_counters: dict[str, int] = {}

    for row in ordered:
        aid = row["asset_id"]
        ptype = row["physics_type"]
        cat = row["category"]
        cat_slug = row["category_slug"]

        if asset_ids is not None and aid not in asset_ids:
            continue
        if physics_type != "all" and ptype != physics_type:
            dropped.append({"asset_id": aid, "category": cat, "reason": f"physics_type={ptype}"})
            continue
        if not row.get("archive") or int(row.get("offset", -1)) < 0 or int(row.get("size", 0)) <= 0:
            dropped.append({"asset_id": aid, "category": cat, "reason": "missing archive byte range"})
            continue
        if textured_only and not row.get("has_textures"):
            dropped.append({"asset_id": aid, "category": cat, "reason": "untextured"})
            continue
        if categories and not (_matches_any(cat, categories) or _matches_any(aid, categories)):
            dropped.append({"asset_id": aid, "category": cat, "reason": "category not in filter"})
            continue
        if exclude_categories and (_matches_any(cat, exclude_categories) or _matches_any(aid, exclude_categories)):
            dropped.append({"asset_id": aid, "category": cat, "reason": "category excluded"})
            continue

        cat_key = (ptype, cat_slug)
        if max_per_category is not None and per_cat.get(cat_key, 0) >= max_per_category:
            dropped.append({"asset_id": aid, "category": cat, "reason": f"max_per_category={max_per_category}"})
            continue
        if limit is not None and len(chosen) >= limit:
            dropped.append({"asset_id": aid, "category": cat, "reason": f"limit={limit}"})
            continue

        per_cat[cat_key] = per_cat.get(cat_key, 0) + 1
        kind_dir = "Articulation" if ptype == "articulated" else "Rigid"
        rd_cat = f"thor_{cat_slug}"
        key = f"{kind_dir}/{rd_cat}"
        idx = id_counters.get(key, 0)
        id_counters[key] = idx + 1
        oid = f"{key}/{idx:05d}"

        chosen.append({**row, "id": oid, "src": f"assets/{aid}/{aid}.usda"})

    by_cat: dict[str, int] = {}
    for c in chosen:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1

    return {
        "source": canonical_url(),
        "dataset": REPO,
        "subpath": SUBPATH,
        "revision": revision,
        "license": LICENSE,
        "summary": {
            "total_catalog": len(rows),
            "chosen": len(chosen),
            "dropped": len(dropped),
            "articulated": sum(1 for c in chosen if c["physics_type"] == "articulated"),
            "rigid": sum(1 for c in chosen if c["physics_type"] == "rigid"),
            "categories": len(by_cat),
        },
        "chosen": chosen,
        "dropped": dropped,
    }


def _safe_extract_assets(tar_bytes: bytes, dest_assets_dir: Path, wanted_asset_ids: set[str]) -> list[str]:
    """Extract only `<asset_id>/*` members for `wanted_asset_ids` (skipping `_mesh/` and `_prim/` variants).

    Writes auxiliary files (textures, etc.) before atomically writing `<asset_id>/<asset_id>.usda` last so an
    interrupted extraction cannot leave a `.usda` without its textures and poison subsequent `fetch_assets` runs.
    """
    dest_assets_dir = Path(dest_assets_dir).resolve()
    dest_assets_dir.mkdir(parents=True, exist_ok=True)
    extracted_usda_ids: set[str] = set()
    deferred_usdas: list[tuple[str, Path, bytes]] = []

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                continue
            top = rel.parts[0]
            if top not in wanted_asset_ids:
                continue
            out_path = (dest_assets_dir / rel).resolve()
            try:
                out_path.relative_to(dest_assets_dir)
            except ValueError:
                continue
            src_fh = tf.extractfile(member)
            if src_fh is None:
                continue
            payload = src_fh.read()
            if len(rel.parts) == 2 and rel.parts[1] == f"{top}.usda":
                deferred_usdas.append((top, out_path, payload))
            else:
                _atomic_write_bytes(out_path, payload)

    for top, out_path, payload in deferred_usdas:
        _atomic_write_bytes(out_path, payload)
        extracted_usda_ids.add(top)
    return sorted(extracted_usda_ids)


def fetch_assets(
    entries: list[dict],
    sources_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    workers: int = 4,
) -> dict:
    """Fetch and extract selected THOR `<asset_id>/` directories grouped by `.tar.zst` archive."""
    sources_dir = Path(sources_dir)
    assets_dir = sources_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Skip assets already extracted on disk.
    already: list[str] = []
    by_archive: dict[str, list[dict]] = {}
    for entry in entries:
        aid = entry["asset_id"]
        usda = assets_dir / aid / f"{aid}.usda"
        if usda.is_file():
            already.append(aid)
            continue
        by_archive.setdefault(entry["archive"], []).append(entry)

    fetched: list[str] = list(already)
    failed: list[dict] = []

    def _one_archive(archive_name: str, group: list[dict]) -> tuple[list[str], list[dict]]:
        wanted = {e["asset_id"] for e in group}
        sample = group[0]
        arrow_entry = {
            "path": archive_name,
            "shard_id": sample.get("shard_id", 0),
            "offset": sample["offset"],
            "size": sample["size"],
        }
        try:
            raw_zst = _read_archive_bytes(sources_dir, arrow_entry, revision=revision, endpoint=endpoint)
            tar_bytes = decompress_zst(raw_zst)
            got = _safe_extract_assets(tar_bytes, assets_dir, wanted)
            missing = sorted(wanted - set(got))
            errs = [{"asset_id": aid, "archive": archive_name, "error": "missing from archive"} for aid in missing]
            return got, errs
        except Exception as exc:
            return [], [
                {"asset_id": aid, "archive": archive_name, "error": f"{type(exc).__name__}: {exc}"}
                for aid in sorted(wanted)
            ]

    if by_archive:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(_one_archive, arch, grp) for arch, grp in sorted(by_archive.items())]
            for fut in as_completed(futs):
                got, errs = fut.result()
                fetched.extend(got)
                failed.extend(errs)

    return {
        "fetched": sorted(set(fetched)),
        "already_present": sorted(set(already)),
        "failed": sorted(failed, key=lambda x: x["asset_id"]),
    }


def is_fixed_base(category: str, size: list[float]) -> bool:
    """Portable tabletop items stay free-base; heavier appliances & furniture use `fixed_base=True`."""
    low = category.lower().replace("_", " ")
    if any(tok in low for tok in PORTABLE_TOKENS) and max(size) <= TABLETOP_MAX_M:
        return False
    return True


def _scale_with_floor(weights: dict[str, float], target_total: float, floor: float) -> dict[str, float]:
    """Distribute `target_total` proportionally to `weights` while keeping every value >= `floor`."""
    keys = list(weights)
    if not keys:
        return {}
    n = len(keys)
    if target_total <= floor * n + 1e-12:
        each = target_total / n
        return {k: each for k in keys}

    pos = {k: max(float(weights[k]), 1e-12) for k in keys}
    fixed: dict[str, float] = {}
    free = list(keys)
    while free:
        rem = target_total - sum(fixed.values())
        w_sum = sum(pos[k] for k in free)
        if w_sum <= 0:
            each = rem / len(free)
            for k in free:
                fixed[k] = each
            break
        below = [k for k in free if rem * pos[k] / w_sum < floor - 1e-12]
        if not below:
            for k in free:
                fixed[k] = rem * pos[k] / w_sum
            break
        for k in below:
            fixed[k] = floor
        free = [k for k in free if k not in fixed]
    return {k: fixed[k] for k in keys}


def estimate_link_masses(
    links: list[str],
    authored_masses: dict[str, float],
    volumes: dict[str, float],
    total_bbox_size: list[float],
    *,
    max_total_mass: float,
    min_total_mass: float = DEFAULT_MIN_ARTICULATED_MASS_KG,
    min_link_mass: float = MIN_LINK_MASS_KG,
    dummy_threshold: float = DUMMY_MASS_THRESHOLD_KG,
) -> tuple[dict[str, float], dict]:
    """Compute per-link masses for THOR articulated objects, replacing `1e-8 kg` dummy masses and missing links.

    Every returned link mass satisfies `m >= min(min_link_mass, max_total_mass / len(links))`, and
    `min_total_mass <= sum(masses.values()) <= max_total_mass`.
    """
    if not links:
        raise ValueError("no links to assign mass to")
    if max_total_mass <= 0 or min_total_mass <= 0 or min_total_mass > max_total_mass:
        raise ValueError(f"invalid mass bounds: [{min_total_mass}, {max_total_mass}]")

    eff_min_link = min(min_link_mass, max_total_mass / len(links))
    valid_authored = {
        p: float(m)
        for p, m in authored_masses.items()
        if p in links and m is not None and math.isfinite(m) and m > dummy_threshold
    }
    dummy_links = [
        p
        for p, m in authored_masses.items()
        if p in links and m is not None and math.isfinite(m) and 0 < m <= dummy_threshold
    ]
    raw_total = sum(valid_authored.values())
    free = [p for p in links if p not in valid_authored]

    bbox_vol = max(float(total_bbox_size[0]) * float(total_bbox_size[1]) * float(total_bbox_size[2]), 1e-6)
    est_total = min(max(bbox_vol * ARTICULATED_SHELL_DENSITY, min_total_mass), max_total_mass)

    masses: dict[str, float] = {}
    clamped = False
    if not free:
        target_total = min(max(raw_total, min_total_mass), max_total_mass)
        clamped = abs(target_total - raw_total) > 1e-9
        masses = _scale_with_floor(valid_authored, target_total, eff_min_link)
    elif not valid_authored:
        weights = {p: max(float(volumes.get(p, 0.0)), 0.005**3) for p in links}
        target_total = max(est_total, eff_min_link * len(links))
        target_total = min(target_total, max_total_mass)
        masses = _scale_with_floor(weights, target_total, eff_min_link)
    else:
        min_free_budget = eff_min_link * len(free)
        max_authored_budget = max(eff_min_link * len(valid_authored), max_total_mass - min_free_budget)
        floored_authored = {p: max(eff_min_link, m) for p, m in valid_authored.items()}
        if sum(floored_authored.values()) > max_authored_budget + 1e-9:
            masses.update(_scale_with_floor(valid_authored, max_authored_budget, eff_min_link))
            clamped = True
        else:
            masses.update(floored_authored)
        authored_sum = sum(masses.values())
        target_total = min(max(authored_sum + min_free_budget, est_total, min_total_mass), max_total_mass)
        left = max(min_free_budget, target_total - authored_sum)
        weights = {p: max(float(volumes.get(p, 0.0)), 0.005**3) for p in free}
        masses.update(_scale_with_floor(weights, left, eff_min_link))

    final_total = sum(masses.values())
    return masses, {
        "raw_authored_kg": round(raw_total, 4),
        "final_kg": round(final_total, 4),
        "dummy_links_replaced": len(dummy_links),
        "filled_missing_links": bool(free),
        "clamped": clamped,
        "bounds_kg": [round(min_total_mass, 4), round(max_total_mass, 4)],
    }


def find_thor_base_and_tree(facts: dict) -> tuple[str, dict[str, tuple[str, str, bool]], list[str]]:
    """Identify the true base link and parent->child joint orientations for a THOR articulated stage.

    THOR's MJCF-USD converter sets `body0 = child`, `body1 = parent`, pointing toward `/<Name>/Geometry/<Name>`
    (the prim carrying `ArticulationRootAPI`). This helper finds the true base link and traverses all non-excluded
    link-to-link joints via BFS to return:
      - `true_base`: prim path of the root link
      - `joint_dirs`: `{joint_path: (parent_path, child_path, is_reversed)}`
      - `unreachable`: link paths disconnected from `true_base` (floating parts)
    """
    links = list(facts["links"])
    link_set = set(links)
    if not links:
        raise ValueError("no RigidBody links found")

    active_joints = [
        j for j in facts["joints"] if j["body0"] in link_set and j["body1"] in link_set and not j["excluded"]
    ]

    # 1. Prefer the unique RigidBody link carrying ArticulationRootAPI (standard in THOR).
    art_links = [p for p in facts.get("articulation_roots", []) if p in link_set]
    if len(art_links) == 1:
        true_base = art_links[0]
    else:
        # 2. Check standard USD orientation (links never appearing as body1) or THOR reversed orientation (never body0).
        body0s = {j["body0"] for j in active_joints}
        body1s = {j["body1"] for j in active_joints}
        std_bases = [p for p in links if p not in body1s]
        rev_bases = [p for p in links if p not in body0s]
        if len(rev_bases) == 1 and len(std_bases) != 1:
            true_base = rev_bases[0]
        elif len(std_bases) == 1:
            true_base = std_bases[0]
        elif len(rev_bases) == 1:
            true_base = rev_bases[0]
        else:
            raise ValueError(f"base link ambiguous: art_roots={art_links}, std={std_bases}, rev={rev_bases}")

    adj: dict[str, list[dict]] = {p: [] for p in links}
    for j in active_joints:
        adj[j["body0"]].append(j)
        adj[j["body1"]].append(j)

    visited = {true_base}
    queue = deque([true_base])
    joint_dirs: dict[str, tuple[str, str, bool]] = {}

    while queue:
        parent = queue.popleft()
        for j in adj[parent]:
            jp = j["path"]
            if jp in joint_dirs:
                continue
            other = j["body1"] if j["body0"] == parent else j["body0"]
            if other in visited:
                # Cycle edge: keep existing direction
                joint_dirs[jp] = (j["body0"], j["body1"], False)
                continue
            is_reversed = j["body0"] == other and j["body1"] == parent
            joint_dirs[jp] = (parent, other, is_reversed)
            visited.add(other)
            queue.append(other)

    unreachable = [p for p in links if p not in visited]
    return true_base, joint_dirs, unreachable


def resolve_collision_free_name(desired: str, used: set[str]) -> str:
    """Pick a collision-free prim name under `/root` when reparenting links."""
    if desired not in used:
        return desired
    candidate = f"link_{desired}"
    idx = 1
    while candidate in used:
        candidate = f"link_{desired}_{idx}"
        idx += 1
    return candidate


def decide_articulated(
    size: list[float],
    facts: dict,
    *,
    max_size: float = TABLESIDE_MAX_M,
    allow_floating: bool = False,
) -> tuple[bool, str]:
    """Return `(chosen, reason)` for a MolmoSpaces THOR articulated candidate."""
    if facts["up_axis"] != "Z":
        return False, f"up axis {facts['up_axis']}"
    if facts["world_movable"]:
        return False, f"{facts['world_movable'][0]['name']} hinges a link to the world, not to the body"
    if not facts["movable"]:
        return False, "no revolute or prismatic joint between two links"
    names = [j["name"] for j in facts["movable"]]
    if len(set(names)) < len(names):
        return False, "two joints share a prim name, so their DOFs cannot be told apart"
    try:
        _, joint_dirs, unreachable = find_thor_base_and_tree(facts)
    except ValueError as exc:
        return False, str(exc)
    if unreachable and not allow_floating:
        return False, f"{len(unreachable)} disconnected link(s): {unreachable[:3]}"
    if not facts["colliders"]:
        return False, "no collision geometry"
    big = max(size)
    if min(size) < MIN_THICKNESS_M:
        return False, f"degenerate size ({min(size):.4f} m)"
    if big > max_size or size[2] > max_size:
        return False, f"{big:.2f} m exceeds max_size {max_size:.2f} m"

    parts = sorted(
        {
            articulated.part_of(
                (joint_dirs[j["path"]][1] if j["path"] in joint_dirs else j["body1"]).rsplit("/", 1)[-1],
                j["type"],
            )
            for j in facts["movable"]
        }
    )
    where = "tabletop" if big <= TABLETOP_MAX_M else "table side"
    return True, f"{where}, {len(facts['movable'])} joint(s): {', '.join(parts)}"


def decide_rigid(
    size: list[float],
    facts: dict,
    *,
    max_size: float = TABLETOP_MAX_M,
) -> tuple[bool, str]:
    """Return `(chosen, reason)` for a MolmoSpaces THOR rigid candidate."""
    if facts["up_axis"] != "Z":
        return False, f"up axis {facts['up_axis']}"
    if not facts["colliders"]:
        return False, "no collision geometry"
    big = max(size)
    if min(size) < MIN_THICKNESS_M:
        return False, f"degenerate size ({min(size):.4f} m)"
    if big > max_size:
        return False, f"{big:.2f} m exceeds max_size {max_size:.2f} m"
    return True, f"rigid tabletop ({big:.3f} m)"


def discover_assets(sources_dir: Path) -> list[Path]:
    """Find all extracted THOR `<asset_id>/<asset_id>.usda` files under `sources_dir/assets/`."""
    assets_dir = Path(sources_dir) / "assets"
    if not assets_dir.is_dir():
        return []
    out: list[Path] = []
    for d in sorted(assets_dir.iterdir()):
        if not d.is_dir():
            continue
        cand = d / f"{d.name}.usda"
        if cand.is_file():
            out.append(cand)
    return out


def survey_asset(usda_path: Path) -> tuple[list[float], dict]:
    """Inspect a THOR `.usda` without modifying any files."""
    import numpy as np

    _, _, Usd, UsdGeom, _, _, _ = pxr_modules()
    stage = Usd.Stage.Open(str(usda_path))
    facts = articulated.inspect(stage)
    for path in facts["lights"]:
        stage.GetPrimAtPath(path).SetActive(False)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if rng.IsEmpty():
        raise ValueError("empty bounding box")
    size = (np.asarray(rng.GetMax(), float) - np.asarray(rng.GetMin(), float)) * facts["meters_per_unit"]
    return [round(float(v), 4) for v in size], facts


def survey(
    sources_dir: Path,
    *,
    selection: dict | None = None,
    max_size: float = TABLESIDE_MAX_M,
    max_rigid_size: float = TABLETOP_MAX_M,
    allow_floating: bool = False,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Survey extracted THOR assets with `pxr` and produce a conversion manifest with deterministic RoboDojo IDs."""
    sources_dir = Path(sources_dir)
    chosen: list[dict] = []
    dropped: list[dict] = []
    counters: dict[str, int] = {}

    if selection:
        work: list[tuple[Path, dict]] = []
        for entry in selection.get("chosen", []):
            aid = entry["asset_id"]
            usda = sources_dir / "assets" / aid / f"{aid}.usda"
            if usda.is_file():
                work.append((usda, entry))
            else:
                dropped.append({"asset_id": aid, "src": entry.get("src", ""), "reason": "missing usda (fetch failed)"})
    else:
        work = [
            (p, {"asset_id": p.parent.name, "category": category_from_asset_id(p.parent.name)})
            for p in discover_assets(sources_dir)
        ]

    for usda_path, sel_row in work:
        aid = sel_row.get("asset_id") or usda_path.parent.name
        rel = usda_path.relative_to(sources_dir).as_posix()
        try:
            size, facts = survey_asset(usda_path)
            ptype = sel_row.get("physics_type") or ("articulated" if facts["movable"] else "rigid")
            if ptype == "articulated":
                ok, reason = decide_articulated(size, facts, max_size=max_size, allow_floating=allow_floating)
            else:
                ok, reason = decide_rigid(size, facts, max_size=max_rigid_size)
        except Exception as exc:
            dropped.append({"asset_id": aid, "src": rel, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        category = sel_row.get("category") or category_from_asset_id(aid)
        cat_slug = articulated.slug(category) or "object"
        entry = {
            "asset_id": aid,
            "hash_id": sel_row.get("hash_id", ""),
            "src": rel,
            "category": category,
            "category_slug": cat_slug,
            "physics_type": ptype,
            "size_m": size,
            "reason": reason,
        }
        if not ok:
            dropped.append(entry)
            continue

        if sel_row.get("id"):
            oid = sel_row["id"]
        else:
            kind_dir = "Articulation" if ptype == "articulated" else "Rigid"
            key = f"{kind_dir}/thor_{cat_slug}"
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            oid = f"{key}/{idx:05d}"
        chosen.append({**entry, "id": oid})

    return {
        "source": canonical_url(),
        "dataset": REPO,
        "subpath": SUBPATH,
        "revision": selection.get("revision", revision) if selection else revision,
        "license": LICENSE,
        "chosen": chosen,
        "dropped": dropped,
    }


def rewrite_layer_paths(layer, remap_pairs: list[tuple], *, Sdf_mod=None) -> tuple[int, int]:
    """Explicitly rewrite BOTH `RelationshipSpec.targetPathList` AND `AttributeSpec.connectionPathList` across `layer`.

    Required for USD 24.x (Isaac Sim 5.1) where `Sdf.Layer.Apply(BatchNamespaceEdit)` does not reliably retarget
    relationship target lists or shader attribute connection lists after reparenting/renaming prims.
    Returns `(rels_rewritten, conns_rewritten)`.
    """
    if Sdf_mod is None:
        _, Sdf_mod, _, _, _, _, _ = pxr_modules()

    ordered = sorted(remap_pairs, key=lambda pair: len(str(pair[0])), reverse=True)

    def _remap(p):
        for src_p, dst_p in ordered:
            if p == src_p or p.HasPrefix(src_p):
                return p.ReplacePrefix(src_p, dst_p)
        return p

    rels_n = 0
    conns_n = 0

    def _visit(prim_spec):
        nonlocal rels_n, conns_n
        for prop in prim_spec.properties:
            if isinstance(prop, Sdf_mod.RelationshipSpec):
                targets = list(prop.targetPathList.GetAddedOrExplicitItems())
                if targets:
                    updated = [_remap(t) for t in targets]
                    if updated != targets:
                        prop.targetPathList.ClearEdits()
                        prop.targetPathList.explicitItems = updated
                        rels_n += 1
            elif isinstance(prop, Sdf_mod.AttributeSpec):
                conns = list(prop.connectionPathList.GetAddedOrExplicitItems())
                if conns:
                    updated = [_remap(c) for c in conns]
                    if updated != conns:
                        prop.connectionPathList.ClearEdits()
                        prop.connectionPathList.explicitItems = updated
                        conns_n += 1
        for child in prim_spec.nameChildren:
            _visit(child)

    for top in layer.rootPrims:
        _visit(top)
    return rels_n, conns_n


def _strip_non_root_top_prims(flat, keep_root_path) -> list[str]:
    """Remove top-level sibling prims outside `keep_root_path` (e.g. THOR's `/PhysicsScene`)."""
    _, Sdf, _, _, _, _, _ = pxr_modules()
    removed: list[str] = []
    edit = Sdf.BatchNamespaceEdit()
    for top in list(flat.rootPrims):
        if top.path != keep_root_path:
            edit.Add(top.path, Sdf.Path.emptyPath)
            removed.append(str(top.path))
    if removed and flat.CanApply(edit):
        flat.Apply(edit)
    return removed


def _reparent_geometry_to_root(flat, src_root: str) -> tuple[list[tuple[str, str]], tuple[int, int]]:
    """Promote `/<Name>/Geometry/*` (and any nested RigidBody links) directly under `/root` and rewrite all paths."""
    _, Sdf, Usd, UsdGeom, UsdPhysics, _, _ = pxr_modules()

    root_path = Sdf.Path(src_root)
    _strip_non_root_top_prims(flat, root_path)

    tmp_stage = Usd.Stage.Open(flat)
    root_prim = tmp_stage.GetPrimAtPath(root_path)
    if not root_prim:
        raise ValueError(f"missing root prim {src_root}")

    # Collect nested RigidBody links + any direct children of `/<Name>/Geometry` so everything under Geometry moves up.
    to_move_map: dict[str, object] = {}
    geom_scope = tmp_stage.GetPrimAtPath(root_path.AppendChild("Geometry"))
    if geom_scope:
        for child in geom_scope.GetChildren():
            to_move_map[str(child.GetPath())] = child
    for prim in Usd.PrimRange(root_prim):
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and prim.GetPath().GetParentPath() != root_path
            and prim.GetPath() != root_path
        ):
            to_move_map[str(prim.GetPath())] = prim
    nested = list(to_move_map.values())

    # Fold non-identity ancestor transforms between `src_root` and each moved prim before reparenting.
    anc_transforms: list[tuple[object, object]] = []
    for prim in nested:
        if not UsdGeom.Xformable(prim) or UsdGeom.Xformable(prim).GetResetXformStack():
            continue
        anc_xf = None
        cur = prim.GetParent()
        chain = []
        while cur and cur.GetPath() != root_path:
            if UsdGeom.Xformable(cur):
                cur_xf = UsdGeom.Xformable(cur)
                mat = cur_xf.GetLocalTransformation()
                if not mat.IsIdentity():
                    chain.append(mat)
                if cur_xf.GetResetXformStack():
                    break
            cur = cur.GetParent()
        for mat in chain:
            anc_xf = mat if anc_xf is None else (anc_xf * mat)
        if anc_xf is not None and not anc_xf.IsIdentity():
            anc_transforms.append((prim, anc_xf))

    for prim, anc_xf in anc_transforms:
        xf = UsdGeom.Xformable(prim)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ofw_reparent")
        op.Set(anc_xf)
        xf.SetXformOpOrder([op, *ops], reset)

    moved: list[tuple[str, str]] = []
    if nested:
        parent_containers = {p.GetPath().GetParentPath() for p in nested}
        used_names = {c.GetName() for c in root_prim.GetChildren()}
        ordered = sorted(nested, key=lambda p: len(p.GetPath().pathString), reverse=True)
        edit = Sdf.BatchNamespaceEdit()
        for prim in ordered:
            old_path = prim.GetPath()
            dest_name = resolve_collision_free_name(prim.GetName(), used_names)
            used_names.add(dest_name)
            new_path = root_path.AppendChild(dest_name)
            edit.Add(old_path, new_path)
            moved.append((str(old_path), str(new_path)))
        if not flat.CanApply(edit):
            raise ValueError(f"cannot reparent nested prims: {moved}")
        flat.Apply(edit)

        cleanup = Sdf.BatchNamespaceEdit()
        removed_any = False
        for container in sorted(parent_containers, key=lambda p: len(str(p)), reverse=True):
            spec = flat.GetPrimAtPath(container)
            if spec and not spec.nameChildren:
                cleanup.Add(container, Sdf.Path.emptyPath)
                removed_any = True
        if removed_any and flat.CanApply(cleanup):
            flat.Apply(cleanup)

    if root_path != Sdf.Path("/root"):
        rename = Sdf.BatchNamespaceEdit()
        rename.Add(root_path, Sdf.Path("/root"))
        if not flat.CanApply(rename):
            raise ValueError(f"cannot rename {root_path} to /root")
        flat.Apply(rename)

    remap_pairs: list[tuple] = []
    for old_s, new_s in moved:
        old_p = Sdf.Path(old_s)
        new_p = Sdf.Path(new_s)
        if root_path != Sdf.Path("/root"):
            new_p = new_p.ReplacePrefix(root_path, Sdf.Path("/root"))
        remap_pairs.append((old_p, new_p))
        if root_path != Sdf.Path("/root"):
            old_under_root = old_p.ReplacePrefix(root_path, Sdf.Path("/root"))
            if old_under_root != new_p:
                remap_pairs.append((old_under_root, new_p))
    if root_path != Sdf.Path("/root"):
        remap_pairs.append((root_path, Sdf.Path("/root")))

    rewritten = rewrite_layer_paths(flat, remap_pairs, Sdf_mod=Sdf)
    flat.defaultPrim = "root"
    return moved, rewritten


def _copy_and_rewrite_textures(flat, source_usda: Path, target_dir: Path) -> list[str]:
    """Copy referenced external textures (`./Textures/*.png`) into `target_dir/textures/` and rewrite AssetPaths."""
    _, Sdf, _, _, _, _, _ = pxr_modules()

    src_dir = source_usda.parent.resolve()
    copied: list[str] = []
    seen_map: dict[Path, str] = {}
    used_rels: set[str] = set()

    def _rewrite_asset_path(ap):
        raw = ap.path
        resolved_str = ap.resolvedPath or ""
        if not raw and not resolved_str:
            return ap
        cand = Path(resolved_str) if resolved_str else (src_dir / raw)
        if not cand.is_file():
            alt = src_dir / raw
            if alt.is_file():
                cand = alt
            else:
                return ap
        cand = cand.resolve()
        if cand in seen_map:
            return Sdf.AssetPath(seen_map[cand])
        rel = Path("textures") / cand.name
        stem, suffix = rel.stem, rel.suffix
        idx = 1
        while rel.as_posix() in used_rels:
            rel = rel.with_name(f"{stem}_{idx}{suffix}")
            idx += 1
        used_rels.add(rel.as_posix())
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            try:
                same = dest.samefile(cand)
            except OSError:
                same = False
            if not same:
                dest.unlink()
        if not dest.exists():
            try:
                os.link(cand, dest)
            except OSError:
                shutil.copy2(cand, dest)
        rel_str = "./" + rel.as_posix()
        seen_map[cand] = rel_str
        copied.append(rel.as_posix())
        return Sdf.AssetPath(rel_str)

    def _visit(prim_spec):
        for prop in prim_spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec):
                continue
            val = prop.default
            if isinstance(val, Sdf.AssetPath):
                new_val = _rewrite_asset_path(val)
                if new_val != val:
                    prop.default = new_val
            elif isinstance(val, Sdf.AssetPathArray):
                new_arr = Sdf.AssetPathArray([_rewrite_asset_path(x) for x in val])
                if list(new_arr) != list(val):
                    prop.default = new_arr
        for child in prim_spec.nameChildren:
            _visit(child)

    for top in flat.rootPrims:
        _visit(top)
    return sorted(set(copied))


def _canonicalize_thor_joints(
    stage,
    joint_dirs: dict[str, tuple[str, str, bool]],
    *,
    Sdf_mod=None,
    UsdPhysics_mod=None,
) -> int:
    """Flip reversed THOR joints (`body0 = child, body1 = parent`) into canonical `body0 = parent, body1 = child`.

    Swaps `body0 <-> body1`, `localPos0 <-> localPos1`, `localRot0 <-> localRot1`, and negates+swaps
    `(lowerLimit, upperLimit)` so the physical constraint and world-space range of motion remain identical.
    """
    if Sdf_mod is None or UsdPhysics_mod is None:
        _, Sdf_mod, _, _, UsdPhysics_mod, _, _ = pxr_modules()
    flipped = 0
    for jpath, (parent_path, child_path, is_reversed) in joint_dirs.items():
        if not is_reversed:
            continue
        prim = stage.GetPrimAtPath(jpath)
        if not prim:
            continue
        joint = UsdPhysics_mod.Joint(prim)
        joint.GetBody0Rel().SetTargets([Sdf_mod.Path(parent_path)])
        joint.GetBody1Rel().SetTargets([Sdf_mod.Path(child_path)])

        p0_attr, p1_attr = joint.GetLocalPos0Attr(), joint.GetLocalPos1Attr()
        p0_has, p1_has = p0_attr.HasAuthoredValue(), p1_attr.HasAuthoredValue()
        p0_val, p1_val = p0_attr.Get(), p1_attr.Get()
        if p0_has or p1_has:
            if p1_has:
                p0_attr.Set(p1_val)
            else:
                p0_attr.Clear()
            if p0_has:
                p1_attr.Set(p0_val)
            else:
                p1_attr.Clear()

        r0_attr, r1_attr = joint.GetLocalRot0Attr(), joint.GetLocalRot1Attr()
        r0_has, r1_has = r0_attr.HasAuthoredValue(), r1_attr.HasAuthoredValue()
        r0_val, r1_val = r0_attr.Get(), r1_attr.Get()
        if r0_has or r1_has:
            if r1_has:
                r0_attr.Set(r1_val)
            else:
                r0_attr.Clear()
            if r0_has:
                r1_attr.Set(r0_val)
            else:
                r1_attr.Clear()

        typed = None
        if prim.IsA(UsdPhysics_mod.RevoluteJoint):
            typed = UsdPhysics_mod.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics_mod.PrismaticJoint):
            typed = UsdPhysics_mod.PrismaticJoint(prim)
        if typed is not None:
            lo_attr, hi_attr = typed.GetLowerLimitAttr(), typed.GetUpperLimitAttr()
            lo = lo_attr.Get() if lo_attr.HasAuthoredValue() else None
            hi = hi_attr.Get() if hi_attr.HasAuthoredValue() else None
            new_lo = None if (hi is None or not math.isfinite(hi)) else float(-hi)
            new_hi = None if (lo is None or not math.isfinite(lo)) else float(-lo)
            if new_lo is not None:
                lo_attr.Set(new_lo)
            else:
                lo_attr.Clear()
            if new_hi is not None:
                hi_attr.Set(new_hi)
            else:
                hi_attr.Clear()
        flipped += 1
    return flipped


def _author_drives_and_friction(stage, joints: list[dict], facts: dict, masses: dict[str, float], spans: dict) -> None:
    """Author `UsdPhysics.DriveAPI` (stiffness=0, small damping) and `PhysxJointAxisAPI` friction + armature."""
    _, Sdf, _, _, UsdPhysics, _, _ = pxr_modules()
    by_name = {j["name"]: j for j in facts["movable"]}

    for entry in joints:
        joint = by_name[entry["name"]]
        prim = stage.GetPrimAtPath(joint["path"])
        is_rev = entry["type"] == "revolute"
        token = "angular" if is_rev else "linear"

        # 1. DriveAPI (if not already authored): zero stiffness so parts hold where placed, gentle damping.
        if not entry.get("drive"):
            damping = DEFAULT_ANGULAR_DAMPING if is_rev else DEFAULT_LINEAR_DAMPING
            drive = UsdPhysics.DriveAPI.Apply(prim, token)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(float(damping))
            drive.CreateTargetPositionAttr(0.0)
            entry["drive"] = {
                "type": "force",
                "stiffness": 0.0,
                "damping": float(damping),
                "target": 0.0,
                "source": "estimated (thor_default_drive)",
            }

        # 2. PhysxJointAxisAPI friction + armature.
        existing_fric = dict(entry.get("friction") or {})
        if "static" not in existing_fric or "dynamic" not in existing_fric:
            if not is_rev:
                friction = articulated.STILL_FRICTION["prismatic"]
            elif articulated._axis_vertical(stage, prim):
                friction = articulated.STILL_FRICTION["upright"]
            else:
                span = spans.get(joint["body1"]) or [0.1, 0.1, 0.1]
                friction = max(
                    MIN_LEVEL_FRICTION,
                    round(
                        articulated.STILL_FRICTION["level"] * masses.get(joint["body1"], 0.1) * 9.81 * max(span) / 2, 3
                    ),
                )
            existing_fric["static"] = float(friction)
            existing_fric["dynamic"] = float(friction)
            existing_fric["source"] = "estimated (thor_default_friction)"
        if "armature" not in existing_fric:
            existing_fric["armature"] = DEFAULT_ANGULAR_ARMATURE if is_rev else DEFAULT_LINEAR_ARMATURE

        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{token}")
        for attr_name, key in (
            ("staticFrictionEffort", "static"),
            ("dynamicFrictionEffort", "dynamic"),
            ("armature", "armature"),
        ):
            prim.CreateAttribute(f"physxJointAxis:{token}:{attr_name}", Sdf.ValueTypeNames.Float).Set(
                float(existing_fric[key])
            )
        entry["friction"] = existing_fric


def adapt_articulated(
    source_usda: Path,
    target_dir: Path,
    *,
    category: str,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float | None = None,
    min_mass: float = DEFAULT_MIN_ARTICULATED_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Flatten, reparent `/<Name>/Geometry/* -> /root/*`, rewrite rels/connections, canonicalize joints, and export."""
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, _, UsdUtils = pxr_modules()

    source_usda = Path(source_usda).resolve()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(source_usda))
    src_facts = articulated.inspect(src_stage)
    if src_facts["up_axis"] != "Z":
        raise ValueError(f"up axis {src_facts['up_axis']} (only Z-up sources are adapted)")
    if not src_facts["movable"]:
        raise ValueError("no revolute or prismatic joint between two links")
    if src_facts["world_movable"]:
        raise ValueError(f"joint anchored to the world instead of a link: {src_facts['world_movable'][0]['path']}")
    if not src_facts["colliders"]:
        raise ValueError("no enabled collision geometry")

    art_attrs: dict[str, tuple] = {}
    for anc in reversed(list(Sdf.Path(src_facts["root"]).GetAncestorsRange())[1:]):
        above = src_stage.GetPrimAtPath(anc)
        for attr in above.GetAttributes() if above else ():
            if attr.GetName().startswith(articulated.PHYSX_ARTICULATION + ":") and attr.HasAuthoredValue():
                art_attrs[attr.GetName()] = (attr.GetTypeName(), attr.Get())

    flat = src_stage.Flatten()
    moved_prims, (rels_rewritten, conns_rewritten) = _reparent_geometry_to_root(flat, src_facts["root"])
    copied_textures = _copy_and_rewrite_textures(flat, source_usda, target_dir)

    stage = Usd.Stage.Open(flat)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetPrimAtPath("/root")
    if root.GetTypeName() != "Xform":
        root.SetTypeName("Xform")
    stage.SetDefaultPrim(root)

    # Identify true base & canonicalize THOR's child->parent joints into parent->child before articulated.inspect().
    pre_facts = articulated.inspect(stage)
    true_base_str, joint_dirs, unreachable = find_thor_base_and_tree(pre_facts)
    if unreachable:
        raise ValueError(f"disconnected links after reparenting: {unreachable}")
    joints_flipped = _canonicalize_thor_joints(stage, joint_dirs)

    facts = articulated.inspect(stage)
    if len(facts["bases"]) != 1:
        raise ValueError(f"base link ambiguous after joint canonicalization: {facts['bases']}")
    base_path = Sdf.Path(facts["bases"][0])
    if str(base_path) != true_base_str:
        raise ValueError(f"base link mismatch: expected {true_base_str}, got {base_path}")
    if base_path.GetParentPath() != root.GetPath():
        raise ValueError(f"base link {base_path} is not a direct child of /root")

    # Move ArticulationRootAPI onto /root; apply PhysxArticulationAPI to BOTH /root and base_prim.
    for path in facts["articulation_roots"]:
        prim = stage.GetPrimAtPath(path)
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(articulated.PHYSX_ARTICULATION + ":") and attr.HasAuthoredValue():
                art_attrs[attr.GetName()] = (attr.GetTypeName(), attr.Get())
        if prim.GetPath() != root.GetPath():
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAppliedSchema("PhysxArticulationAPI")

    UsdPhysics.ArticulationRootAPI.Apply(root)
    base_prim = stage.GetPrimAtPath(base_path)
    for prim in (root, base_prim):
        prim.AddAppliedSchema("PhysxArticulationAPI")
        for attr_name, (kind, value) in art_attrs.items():
            prim.CreateAttribute(attr_name, kind).Set(value)

    for joint in facts["world_joints"]:
        stage.GetPrimAtPath(joint["path"]).SetActive(False)
    for path in facts["lights"]:
        stage.GetPrimAtPath(path).SetActive(False)
    for path in facts["scripts"]:
        stage.GetPrimAtPath(path).GetAttribute(articulated.SCRIPT_ATTR).Block()

    # Put true base link FIRST among /root's children (for ArticulationObject.fix_root_link()).
    names = [c.GetName() for c in root.GetChildren()]
    root.SetChildrenReorder([base_path.name] + [n for n in names if n != base_path.name])

    unit = src_facts["meters_per_unit"] * scale
    if unit <= 0:
        raise ValueError("bad metersPerUnit")
    root_xf = UsdGeom.Xformable(root)
    root_matrix = root_xf.GetLocalTransformation()
    root_xf.ClearXformOpOrder()

    movable_children = [c for c in root.GetChildren() if UsdGeom.Xformable(c) and not c.IsA(UsdPhysics.Joint)]
    if not movable_children:
        raise ValueError("no Xformable direct children under /root after reparenting")

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
        if front:
            xf.SetXformOpOrder(front + list(ops), reset)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = articulated._bound(cache, root)
    if bound is None:
        raise ValueError("empty bounding box")
    centre = (bound[0] + bound[1]) / 2
    for child in movable_children:
        xf = UsdGeom.Xformable(child)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        shift = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "ofw_center")
        shift.Set(Gf.Vec3d(*map(float, -centre)))
        xf.SetXformOpOrder([shift, *ops], reset)

    limit_fixes: dict[str, tuple[float, float]] = {}
    for joint in facts["movable"]:
        lo_val, hi_val = joint["lower"], joint["upper"]
        if joint["type"] == "prismatic" and unit != 1.0:
            lo_val = None if lo_val is None else lo_val * unit
            hi_val = None if hi_val is None else hi_val * unit
            prim = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint["path"]))
            if lo_val is not None:
                prim.GetLowerLimitAttr().Set(float(lo_val))
            if hi_val is not None:
                prim.GetUpperLimitAttr().Set(float(hi_val))
        if lo_val is not None and hi_val is not None and lo_val > hi_val:
            limit_fixes[joint["path"]] = (hi_val, lo_val)
            prim = stage.GetPrimAtPath(joint["path"])
            typed = (
                UsdPhysics.RevoluteJoint(prim)
                if prim.IsA(UsdPhysics.RevoluteJoint)
                else UsdPhysics.PrismaticJoint(prim)
            )
            typed.GetLowerLimitAttr().Set(float(hi_val))
            typed.GetUpperLimitAttr().Set(float(lo_val))

    cache.Clear()
    if root.HasAttribute("extentsHint"):
        root.RemoveProperty("extentsHint")
    lo, hi = articulated._bound(cache, root)
    size = hi - lo
    if np.linalg.norm((lo + hi) / 2) > 1e-4:
        raise ValueError(f"origin not at the bounding-box centre after shift: {(lo + hi) / 2}")

    # Strip MassAPI (density) from non-RigidBody descendant collider meshes so link-level physics:mass is unambiguous.
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)

    volumes, spans = {}, {}
    for path in facts["links"]:
        b = articulated._bound(cache, stage.GetPrimAtPath(path))
        spans[path] = [float(v) for v in b[1] - b[0]] if b else None
        volumes[path] = float(np.prod(np.maximum(b[1] - b[0], 0.005))) if b else 0.005**3

    fixed_base = is_fixed_base(category, [float(v) for v in size])
    if max_mass is None:
        if not fixed_base:
            eff_max_mass = DEFAULT_MAX_PORTABLE_MASS_KG
        elif float(np.max(size)) <= TABLETOP_MAX_M:
            eff_max_mass = DEFAULT_MAX_TABLETOP_MASS_KG
        else:
            eff_max_mass = DEFAULT_MAX_TABLESIDE_MASS_KG
    else:
        eff_max_mass = max_mass

    masses, mass_info = estimate_link_masses(
        facts["links"],
        facts["masses"],
        volumes,
        [float(v) for v in size],
        max_total_mass=eff_max_mass,
        min_total_mass=min_mass,
    )
    for path, new_m in masses.items():
        prim = stage.GetPrimAtPath(path)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(new_m))

    joints = []
    for joint in facts["movable"]:
        lower, upper = joint["lower"], joint["upper"]
        if joint["path"] in limit_fixes:
            lower, upper = limit_fixes[joint["path"]]
        elif joint["type"] == "prismatic" and unit != 1.0:
            lower, upper = (None if v is None else v * unit for v in (lower, upper))
        child = joint["body1"].rsplit("/", 1)[-1]
        part = articulated.part_of(child, joint["type"])
        part = articulated.joint_part(
            part, joint["type"], lower, upper, spans.get(joint["body1"]), [float(v) for v in size]
        )
        entry = {
            "name": joint["name"],
            "type": joint["type"],
            "role": articulated.role_of(part),
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

    _author_drives_and_friction(stage, joints, facts, masses, spans)

    out_usd = target_dir / "object.usd"
    flat.Export(str(out_usd), args={"format": usd_format})

    # Post-write sanity check on reopened USD.
    reopened = Usd.Stage.Open(str(out_usd))
    re_root = reopened.GetDefaultPrim()
    if not re_root or re_root.GetPath() != Sdf.Path("/root") or not re_root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("exported object.usd missing /root with ArticulationRootAPI")
    first_child = re_root.GetChildren()[0] if re_root.GetChildren() else None
    if not first_child or first_child.GetTypeName() != "Xform" or not first_child.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"first child of /root is not an Xform with RigidBodyAPI: {first_child}")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(out_usd)))
    if unresolved:
        raise ValueError(f"unresolved asset dependencies: {list(unresolved)}")

    return {
        "size": [float(v) for v in size],
        "centre_shift": [float(v) for v in -centre],
        "unit": unit,
        "yaw_deg": yaw_deg,
        "joints": joints,
        "link_masses": {p.rsplit("/", 1)[-1]: round(m, 4) for p, m in masses.items()},
        "mass_info": mass_info,
        "fixed_base": fixed_base,
        "limit_fixes": {p.rsplit("/", 1)[-1]: list(v) for p, v in limit_fixes.items()},
        "removed": {
            "world_joints": [j["path"] for j in facts["world_joints"]],
            "lights": facts["lights"],
            "scripts": facts["scripts"],
            "articulation_root_moved_from": facts["articulation_roots"],
        },
        "reparented_prims": moved_prims,
        "rewritten_paths": {"relationships": rels_rewritten, "connections": conns_rewritten},
        "joints_canonicalized": joints_flipped,
        "textures": copied_textures,
        "base_link": base_path.name,
        "links": [p.rsplit("/", 1)[-1] for p in facts["links"]],
    }


def adapt_rigid(
    source_usda: Path,
    target_dir: Path,
    *,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float = DEFAULT_MAX_RIGID_MASS_KG,
    min_mass: float = DEFAULT_MIN_RIGID_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Adapt a rigid MolmoSpaces THOR `.usda` into a RoboDojo `RigidObject` (`object.usd`)."""
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = pxr_modules()

    source_usda = Path(source_usda).resolve()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(source_usda))
    src_facts = articulated.inspect(src_stage)
    if src_facts["up_axis"] != "Z":
        raise ValueError(f"up axis {src_facts['up_axis']} (only Z-up sources are adapted)")
    if not src_facts["colliders"]:
        raise ValueError("no enabled collision geometry")

    raw_mass = sum(m for m in src_facts["masses"].values() if m > DUMMY_MASS_THRESHOLD_KG)
    flat = src_stage.Flatten()
    _, (rels_rewritten, conns_rewritten) = _reparent_geometry_to_root(flat, src_facts["root"])
    copied_textures = _copy_and_rewrite_textures(flat, source_usda, target_dir)

    stage = Usd.Stage.Open(flat)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetPrimAtPath("/root")
    if root.GetTypeName() != "Xform":
        root.SetTypeName("Xform")
    stage.SetDefaultPrim(root)

    if root.HasAPI(UsdPhysics.ArticulationRootAPI):
        root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root.RemoveAppliedSchema("PhysxArticulationAPI")
    if root.HasAttribute(articulated.SCRIPT_ATTR) and root.GetAttribute(articulated.SCRIPT_ATTR).Get():
        root.GetAttribute(articulated.SCRIPT_ATTR).Block()
    for prim in list(Usd.PrimRange(root)):
        if prim.GetPath() == root.GetPath():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        prim.RemoveAppliedSchema("PhysxArticulationAPI")
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() in articulated.LIGHT_TYPES:
            prim.SetActive(False)
        if prim.HasAttribute(articulated.SCRIPT_ATTR) and prim.GetAttribute(articulated.SCRIPT_ATTR).Get():
            prim.GetAttribute(articulated.SCRIPT_ATTR).Block()
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding(materialPurpose="physics")

    unit = src_facts["meters_per_unit"] * scale
    root_xf = UsdGeom.Xformable(root)
    root_matrix = root_xf.GetLocalTransformation()
    root_xf.ClearXformOpOrder()

    movable_children = [c for c in root.GetChildren() if UsdGeom.Xformable(c) and not c.IsA(UsdPhysics.Joint)]
    if not movable_children:
        raise ValueError("no Xformable children under /root")
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
        if front:
            xf.SetXformOpOrder(front + list(ops), reset)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = articulated._bound(cache, root)
    if bound is None:
        raise ValueError("empty bounding box")
    centre = (bound[0] + bound[1]) / 2
    for child in movable_children:
        xf = UsdGeom.Xformable(child)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        shift = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "ofw_center")
        shift.Set(Gf.Vec3d(*map(float, -centre)))
        xf.SetXformOpOrder([shift, *ops], reset)

    cache.Clear()
    if root.HasAttribute("extentsHint"):
        root.RemoveProperty("extentsHint")
    lo, hi = articulated._bound(cache, root)
    size = hi - lo
    if np.linalg.norm((lo + hi) / 2) > 1e-4:
        raise ValueError(f"origin not at bounding-box centre: {(lo + hi) / 2}")

    mass = raw_mass if raw_mass > 0 else float(np.prod(np.maximum(size, 0.01)) * RIGID_ESTIMATED_DENSITY)
    clamped_mass = round(min(max(mass, min_mass), max_mass), 4)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(clamped_mass))

    out_usd = target_dir / "object.usd"
    flat.Export(str(out_usd), args={"format": usd_format})

    reopened = Usd.Stage.Open(str(out_usd))
    bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies != ["/root"]:
        raise ValueError(f"rigid bodies other than /root remain: {bodies}")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(out_usd)))
    if unresolved:
        raise ValueError(f"unresolved asset dependencies: {list(unresolved)}")

    return {
        "size": [float(v) for v in size],
        "centre_shift": [float(v) for v in -centre],
        "unit": unit,
        "yaw_deg": yaw_deg,
        "mass_kg": clamped_mass,
        "mass_info": {
            "raw_authored_kg": round(raw_mass, 4),
            "final_kg": clamped_mass,
            "clamped": abs(clamped_mass - mass) > 1e-6,
            "bounds_kg": [min_mass, max_mass],
        },
        "rewritten_paths": {"relationships": rels_rewritten, "connections": conns_rewritten},
        "textures": copied_textures,
    }


def convert_one(
    sources_dir: Path,
    out_root: Path,
    entry: dict,
    *,
    revision: str = DEFAULT_REVISION,
    max_mass: float | None = None,
    usd_format: str = "usdc",
) -> dict:
    """Convert one selected MolmoSpaces THOR entry into `out_root/Assets/Object/RoboDojo/<id>/`."""
    sources_dir = Path(sources_dir)
    out_root = Path(out_root)
    aid = entry["asset_id"]
    src_rel = entry.get("src") or f"assets/{aid}/{aid}.usda"
    usda_path = sources_dir / src_rel
    oid = entry["id"]
    target = out_root / "Assets/Object/RoboDojo" / oid

    source_provenance = {
        "dataset": REPO,
        "subpath": SUBPATH,
        "url": canonical_url(),
        "object": f"{SUBPATH}/{aid}",
        "revision": revision,
        "asset_id": aid,
        "hash_id": entry.get("hash_id", ""),
        "category": entry.get("category", category_from_asset_id(aid)),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "rsi_pools": "allowed (CC-BY-4.0)",
        "runtime_validated": False,
    }

    ptype = entry.get("physics_type", "articulated")
    if ptype == "articulated":
        facts = adapt_articulated(
            usda_path,
            target,
            category=source_provenance["category"],
            max_mass=max_mass,
            usd_format=usd_format,
        )
        mass_info = facts["mass_info"]
        mass_basis = (
            f"estimated from bbox volume ({mass_info['final_kg']} kg, "
            f"{mass_info['dummy_links_replaced']} dummy link(s) replaced)"
        )
        meta = articulated.metadata(
            oid,
            facts,
            mass_info["final_kg"],
            mass_basis,
            source_provenance,
            fixed_base=facts["fixed_base"],
        )
        meta["adapter"] = {
            "script": SCRIPT,
            "version": VERSION,
            "reparented_prims": len(facts["reparented_prims"]),
            "rewritten_paths": facts["rewritten_paths"],
            "joints_canonicalized": facts["joints_canonicalized"],
            "textures": facts["textures"],
            "mass": mass_info,
            "validation": None,
        }
        meta["runtime_ready"] = False
    else:
        facts = adapt_rigid(
            usda_path,
            target,
            max_mass=max_mass if max_mass is not None else DEFAULT_MAX_RIGID_MASS_KG,
            usd_format=usd_format,
        )
        size = facts["size"]
        geom = articulated.box_geometry(size)
        meta = {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{canonical_url()}/{SUBPATH}/{aid}")),
            "physics": {
                "mass": facts["mass_kg"],
                "friction": RIGID_FRICTION,
                "size": [round(v, 4) for v in size],
            },
            "geometry": geom,
            "active": {
                "place": {
                    "up": {
                        "orientation": [1, 0, 0, 0],
                        "contact_circle": {"center": [0, 0, -size[2] / 2, 1, 0, 0, 0], "radius": geom["radius"]},
                    }
                }
            },
            "passive": {"functional": {}, "support": {}},
            "source": source_provenance,
            "adapter": {
                "script": SCRIPT,
                "version": VERSION,
                "rewritten_paths": facts["rewritten_paths"],
                "textures": facts["textures"],
                "mass": facts["mass_info"],
                "validation": None,
            },
            "asset_id": oid,
            "runtime_ready": False,
        }

    articulated.write_json(target / "metadata.json", meta)
    title = source_provenance["category"].replace("_", " ")
    desc_text = f"A {ptype} {title} ({aid}) from MolmoSpaces THOR."
    articulated.write_json(
        target / "description.json",
        {
            "uuid": meta["uuid"],
            "caption": {"name": [title]},
            "description": [desc_text],
        },
    )
    sha = _sha256_file(target / "object.usd")
    return {
        "id": oid,
        "asset_id": aid,
        "src": src_rel,
        "physics_type": ptype,
        "size_m": meta["physics"]["size"],
        "mass_kg": meta["physics"]["mass"],
        "sha256": sha,
    }


def convert(
    sources_dir: Path,
    out_root: Path,
    manifest: dict,
    *,
    skip_existing: bool = False,
    max_mass: float | None = None,
    usd_format: str = "usdc",
) -> dict:
    """Convert every chosen entry in `manifest`."""
    out_root = Path(out_root)
    revision = manifest.get("revision", DEFAULT_REVISION)
    done, skipped, failed = [], [], []
    for entry in manifest.get("chosen", []):
        target = out_root / "Assets/Object/RoboDojo" / entry["id"]
        if skip_existing and (target / "object.usd").is_file() and (target / "metadata.json").is_file():
            skipped.append(entry["id"])
            continue
        try:
            done.append(
                convert_one(
                    sources_dir,
                    out_root,
                    entry,
                    revision=revision,
                    max_mass=max_mass,
                    usd_format=usd_format,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "id": entry["id"],
                    "asset_id": entry.get("asset_id", ""),
                    "src": entry.get("src", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "script": SCRIPT,
        "version": VERSION,
        "revision": revision,
        "converted": done,
        "skipped": skipped,
        "failed": failed,
    }


def stamp_ready(
    out_root: Path,
    object_id: str,
    check_dir: Path,
    *,
    note: str | None = None,
) -> tuple[bool, list[str]]:
    """Record `check_articulation.py` or `check_object.py` result in `metadata.json` and set `runtime_ready`."""
    target = Path(out_root) / "Assets/Object/RoboDojo" / object_id
    if not (target / "metadata.json").is_file():
        target = Path(out_root) / object_id
    meta_path = target / "metadata.json"
    if not meta_path.is_file():
        return False, ["missing_metadata"]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parts = object_id.strip("/").split("/")
    kind, category, index = parts[-3], parts[-2], parts[-1]
    check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
    if not check_file.is_file():
        meta.setdefault("adapter", {})["validation"] = {"passed": False, "why": ["not_checked"], "note": note}
        meta["runtime_ready"] = False
        meta.setdefault("source", {})["runtime_validated"] = False
        articulated.write_json(meta_path, meta)
        return False, ["not_checked"]

    check = json.loads(check_file.read_text(encoding="utf-8"))
    if kind == "Articulation":
        passed = bool(check.get("pass"))
        if passed:
            why = []
        else:
            why = list(articulated.failure_reasons(check))
            if check.get("checks", {}).get("rest") == "fail" and "rest" not in why:
                why = sorted(set(why + ["rest"]))
            if not why:
                why = ["unknown"]
        validation = {
            "check_articulation": {
                "pass": passed,
                "checks": check.get("checks", {}),
                "checked_at": check.get("checked_at"),
            },
            "passed": passed,
            "why": why,
            "note": note,
        }
    else:
        passed = check.get("settle") == "pass"
        why = [] if passed else ["settle"]
        validation = {
            "check_object": {
                k: check.get(k) for k in ("settle", "tilt_deg", "z_m", "expect_z_m", "checked_at") if k in check
            },
            "passed": passed,
            "why": why,
            "note": note,
        }

    meta.setdefault("adapter", {})["validation"] = validation
    meta["runtime_ready"] = passed
    meta.setdefault("source", {})["runtime_validated"] = passed
    articulated.write_json(meta_path, meta)
    return passed, why


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_filter = sub.add_parser("filter", help="Fetch/load THOR index and filter candidates before downloading.")
    p_filter.add_argument("--sources-dir", type=Path, required=True, help="Cache dir holding _index/ and assets/")
    p_filter.add_argument("--revision", default=DEFAULT_REVISION)
    p_filter.add_argument("--endpoint", default=None, help="Override HF_ENDPOINT")
    p_filter.add_argument("--refresh-index", action="store_true", help="Re-download THOR index files")
    p_filter.add_argument("--physics-type", choices=("articulated", "rigid", "all"), default="articulated")
    p_filter.add_argument("--categories", nargs="*", default=None, help="Regex/substring category patterns to include")
    p_filter.add_argument("--exclude-categories", nargs="*", default=None)
    p_filter.add_argument("--textured-only", action="store_true")
    p_filter.add_argument("--max-per-category", type=int, default=None)
    p_filter.add_argument("--limit", type=int, default=None)
    p_filter.add_argument("--asset-ids", nargs="*", default=None)
    p_filter.add_argument("--asset-ids-file", type=Path, default=None)
    p_filter.add_argument("--out", type=Path, required=True, help="Output selection JSON path")

    p_fetch = sub.add_parser("fetch", help="Selectively Range-fetch & extract THOR assets from selection JSON.")
    p_fetch.add_argument("--selection", type=Path, required=True)
    p_fetch.add_argument("--out", type=Path, required=True, help="Destination sources/molmospaces_thor directory")
    p_fetch.add_argument("--revision", default=None)
    p_fetch.add_argument("--endpoint", default=None)
    p_fetch.add_argument("--workers", type=int, default=4)
    p_fetch.add_argument("--shard-index", type=int, default=0)
    p_fetch.add_argument("--shard-total", type=int, default=1)

    p_survey = sub.add_parser("survey", help="Inspect extracted THOR .usda assets with pxr and emit manifest.")
    p_survey.add_argument("--source", type=Path, required=True, help="sources/molmospaces_thor directory")
    p_survey.add_argument("--selection", type=Path, default=None, help="Optional filter selection JSON")
    p_survey.add_argument("--max-size", type=float, default=TABLESIDE_MAX_M)
    p_survey.add_argument("--max-rigid-size", type=float, default=TABLETOP_MAX_M)
    p_survey.add_argument("--allow-floating", action="store_true")
    p_survey.add_argument("--out", type=Path, required=True, help="Output conversion manifest JSON")

    p_conv = sub.add_parser("convert", help="Adapt THOR .usda assets into object.usd + metadata.json.")
    p_conv.add_argument("--source", type=Path, required=True, help="sources/molmospaces_thor directory")
    p_conv.add_argument("--root", type=Path, required=True, help="Output root holding Assets/Object/RoboDojo/")
    p_conv.add_argument("--manifest", type=Path, required=True, help="Manifest JSON from `survey`")
    p_conv.add_argument("--skip-existing", action="store_true")
    p_conv.add_argument("--max-mass", type=float, default=None)
    p_conv.add_argument("--usd-format", choices=("usdc", "usda"), default="usdc")
    p_conv.add_argument("--summary", type=Path, default=None)

    p_ready = sub.add_parser("ready", help="Stamp check_articulation/check_object results into metadata.json.")
    p_ready.add_argument("--root", type=Path, required=True, help="Output root holding Assets/Object/RoboDojo/")
    p_ready.add_argument(
        "--check-dir", type=Path, required=True, help="--image-dir passed to check_articulation/object"
    )
    p_ready.add_argument("--manifest", type=Path, default=None, help="Manifest JSON listing converted ids")
    p_ready.add_argument("--ids-file", type=Path, default=None, help="Optional newline-separated object ids")
    p_ready.add_argument("ids", nargs="*", help="Explicit object ids (e.g. Articulation/thor_microwave/00000)")
    p_ready.add_argument("--summary", type=Path, default=None)
    p_ready.add_argument("--note", default=None)

    args = parser.parse_args(argv)

    if args.command == "filter":
        umeta, arrow_table, cmeta = fetch_index(
            args.sources_dir, revision=args.revision, endpoint=args.endpoint, force=args.refresh_index
        )
        rows = build_catalog(umeta, arrow_table, cmeta)
        aids = None
        if args.asset_ids or args.asset_ids_file:
            aids = set(args.asset_ids or [])
            if args.asset_ids_file:
                aids |= {
                    ln.strip()
                    for ln in args.asset_ids_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")
                }
        result = filter_catalog(
            rows,
            physics_type=args.physics_type,
            categories=args.categories,
            exclude_categories=args.exclude_categories,
            textured_only=args.textured_only,
            max_per_category=args.max_per_category,
            limit=args.limit,
            asset_ids=aids,
            revision=args.revision,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, result)
        print(f"[thor] filter {json.dumps(result['summary'], ensure_ascii=False)}", flush=True)
        return 0

    if args.command == "fetch":
        sel = json.loads(args.selection.read_text(encoding="utf-8"))
        entries = sel.get("chosen", [])
        if args.shard_total > 1:
            entries = [e for i, e in enumerate(entries) if i % args.shard_total == args.shard_index]
        revision = args.revision or sel.get("revision", DEFAULT_REVISION)
        res = fetch_assets(entries, args.out, revision=revision, endpoint=args.endpoint, workers=args.workers)
        print(
            f"[thor] fetch shard {args.shard_index}/{args.shard_total}: "
            f"fetched={len(res['fetched'])}, already={len(res['already_present'])}, failed={len(res['failed'])}",
            flush=True,
        )
        return 1 if res["failed"] else 0

    if args.command == "survey":
        sel = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
        res = survey(
            args.source,
            selection=sel,
            max_size=args.max_size,
            max_rigid_size=args.max_rigid_size,
            allow_floating=args.allow_floating,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, res)
        print(f"[thor] survey chosen={len(res['chosen'])}, dropped={len(res['dropped'])}", flush=True)
        return 0

    if args.command == "convert":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        res = convert(
            args.source,
            args.root,
            manifest,
            skip_existing=args.skip_existing,
            max_mass=args.max_mass,
            usd_format=args.usd_format,
        )
        summary_path = args.summary or (args.root / "molmospaces_thor_conversion.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(summary_path, res)
        print(
            f"[thor] convert converted={len(res['converted'])}, skipped={len(res['skipped'])}, "
            f"failed={len(res['failed'])} -> {summary_path}",
            flush=True,
        )
        return 1 if res["failed"] else 0

    # ready
    ids = list(args.ids)
    if args.manifest:
        m = json.loads(args.manifest.read_text(encoding="utf-8"))
        ids.extend(e["id"] for e in m.get("chosen", []))
    if args.ids_file:
        ids.extend(
            ln.strip()
            for ln in args.ids_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        )
    ids = list(dict.fromkeys(ids))
    if not ids:
        parser.error("ready needs ids, --manifest, or --ids-file")

    passed_ids, failed_ids, reason_counts = [], [], {}
    for oid in ids:
        ok, why = stamp_ready(args.root, oid, args.check_dir, note=args.note)
        if ok:
            passed_ids.append(oid)
        else:
            failed_ids.append({"id": oid, "why": why})
            for code in why:
                reason_counts[code] = reason_counts.get(code, 0) + 1
    summary = {
        "total": len(ids),
        "passed": len(passed_ids),
        "failed": len(failed_ids),
        "failure_reasons": reason_counts,
        "passed_ids": passed_ids,
        "not_ready": failed_ids,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.summary, summary)
    print(f"[thor] ready {len(passed_ids)}/{len(ids)}; reasons={json.dumps(reason_counts)}", flush=True)
    return 0 if not failed_ids else 1


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
