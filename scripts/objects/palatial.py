"""Import Palatial SimReady Asset Library (PalatialSim/palatial-nv-asset-library, CC-BY-4.0) into RoboDojo format.

Palatial SimReady provides 1,013 Z-up, metre-scaled UsdPreviewSurface assets (814 single_body, 109 multi_body,
90 articulated; 955 stock-passing, ~98% tabletop <= 0.65 m).

Key library-specific adaptations handled here:
  1. Space-free paths: `catalog.jsonl` lists `batch*/` paths containing spaces (736/1,013 entries), which break
     `asset_client.py pull`. This adapter ALWAYS fetches USD + textures from `<logical_asset>/simready_usd/` (100%
     space-free) and stores `physics_*.json` at `<logical_asset>/metadata/physics.json`. `object.usd` never
     references any path with spaces.
  2. `multi_body` flattening: Some `multi_body` items (e.g. `D00191` Cookie Jar) carry `PhysicsArticulationRootAPI`
     despite `moving_joints: false` and have loose lids/parts. When adapted as `Rigid/*`, all
     `PhysicsArticulationRootAPI`/`PhysxArticulationAPI` and descendant `RigidBodyAPI`/`MassAPI`/joints are stripped
     so they run cleanly as single rigid bodies under `/root`.
  3. `articulated` base-link & friction repair: `part_0`/`part_00` is NOT always the base link (e.g. `D00519` has
     `part_1` as `body0` base; `D00854` has `part_01` as base and `part_00` = 0.01 kg < `MIN_LINK_MASS` = 0.02 kg).
     This adapter reparents nested links to direct children of `/root`, rewrites BOTH `RelationshipSpec.targetPathList`
     and `AttributeSpec.connectionPathList`, places the true `body0` base link FIRST under `/root` (for
     `ArticulationObject.fix_root_link()`), clamps every link mass >= 0.02 kg (scaling `diagonalInertia` in lockstep),
     and authors `PhysxJointAxisAPI` (`staticFrictionEffort` / `dynamicFrictionEffort`) since Palatial only writes
     legacy `physxJoint:jointFriction`.

Pipeline subcommands (pure-stdlib for `filter`, `fetch`, `survey`, `ready`; lazy `pxr`/`numpy` only in `convert`):
  filter   Filter `data/catalog.jsonl` before downloading (physics_kind, runtime_status, class_label, size, limit).
  fetch    Download `<logical_asset>/simready_usd/*` and `metadata/physics.json` with HF_ENDPOINT + retry/backoff.
  survey   Pure-stdlib inspection of downloaded `simready_usd/` + `metadata/physics.json`, emitting `manifest.json`.
  convert  Flatten, normalize, repair rigid/articulated physics, and write `object.usd` + `metadata.json`.
  ready    Stamp `check_object.py` / `check_articulation.py` results into `metadata.json` (`runtime_ready`).

CPU unit tests: `scripts/objects/test_palatial.py`.
Slurm pipeline: `scripts/objects/slurm_palatial.sbatch`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import articulated  # noqa: E402

SCRIPT = "scripts/objects/palatial.py"
VERSION = 1

REPO = "PalatialSim/palatial-nv-asset-library"
CATALOG_RELPATH = "data/catalog.jsonl"
DEFAULT_REVISION = "main"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "curl/8.0"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "Palatial SimReady Asset Library, PalatialSim, huggingface.co/datasets/PalatialSim/palatial-nv-asset-library"
)

PHYSICS_KINDS = ("single_body", "multi_body", "articulated")
PHYSICS_FILTER_CHOICES = ("all", "rigid", "single_body", "multi_body", "articulated")
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# Default size / mass bounds (metres, kg).
MIN_SIZE_M = 0.015
TABLETOP_MAX_M = 0.65
TABLESIDE_MAX_M = 0.90
MIN_THICKNESS_M = 0.002
DEFAULT_MIN_RIGID_MASS_KG = 0.005
DEFAULT_MAX_RIGID_MASS_KG = 0.5
DEFAULT_MIN_ARTICULATED_MASS_KG = 0.05
DEFAULT_MAX_TABLETOP_MASS_KG = 15.0
DEFAULT_MAX_TABLESIDE_MASS_KG = 25.0
DEFAULT_MAX_PORTABLE_MASS_KG = 3.0
MIN_LINK_MASS_KG = articulated.MIN_LINK_MASS
RIGID_FRICTION = 0.6
DEFAULT_NON_OPENING_FRICTION = {"revolute": 0.05, "prismatic": 0.2}

# Class / name tokens indicating portable handheld or tabletop articulated items whose base should stay free.
PORTABLE_TOKENS = (
    "bag",
    "basket",
    "book",
    "bottle",
    "bowl",
    "box",
    "briefcase",
    "calculator",
    "camera",
    "can",
    "case",
    "clip",
    "clock",
    "compass",
    "container",
    "cup",
    "cutter",
    "flask",
    "flashlight",
    "folder",
    "glasses",
    "headphones",
    "jar",
    "jug",
    "kettle",
    "knife",
    "laptop",
    "lighter",
    "lunchbox",
    "mug",
    "notebook",
    "padlock",
    "pan",
    "pen",
    "pencil",
    "phone",
    "pitcher",
    "pliers",
    "pot",
    "pouch",
    "purse",
    "remote",
    "ruler",
    "scissors",
    "shaker",
    "soap",
    "spoon",
    "stapler",
    "suitcase",
    "teapot",
    "thermos",
    "tongs",
    "toolbox",
    "toothbrush",
    "tray",
    "turntable",
    "umbrella",
    "wallet",
    "watch",
    "wrench",
)


def hf_endpoint(endpoint: str | None = None) -> str:
    raw = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    return raw.rstrip("/")


def canonical_url() -> str:
    return f"{DEFAULT_HF_ENDPOINT}/datasets/{REPO}"


def hf_resolve_url(rel_path: str, *, revision: str = DEFAULT_REVISION, endpoint: str | None = None) -> str:
    base = hf_endpoint(endpoint)
    quoted = urllib.parse.quote(rel_path.lstrip("/"), safe="/")
    return f"{base}/datasets/{REPO}/resolve/{revision}/{quoted}"


def rewrite_hf_url(url: str, endpoint: str | None = None) -> str:
    """Rewrite a `https://huggingface.co/...` URL to the active `HF_ENDPOINT` mirror."""
    if not url:
        return url
    target = hf_endpoint(endpoint)
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc.lower() in {"huggingface.co", "www.huggingface.co"}:
        t_parsed = urllib.parse.urlsplit(target)
        base_path = t_parsed.path.rstrip("/")
        return urllib.parse.urlunsplit(
            (t_parsed.scheme, t_parsed.netloc, base_path + parsed.path, parsed.query, parsed.fragment)
        )
    return url


def pxr_modules():
    """Import pxr modules lazily, falling back to headless Isaac SimulationApp when standalone usd-core is absent."""
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
    attempts: int = 6,
    timeout: float = 120.0,
    backoff: float = 2.0,
    user_agent: str = USER_AGENT,
    sleep_fn=time.sleep,
) -> tuple[bytes, dict[str, str]]:
    """GET `url` with `User-Agent: curl/8.0` and exponential backoff on HTTP 429/5xx and transient network errors."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.read(), headers
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


def list_hf_tree(
    subpath: str = "",
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    attempts: int = 6,
) -> list[tuple[str, int]]:
    """List `(path, size)` file entries under `subpath` from the Hugging Face tree API with pagination and retries."""
    base = hf_endpoint(endpoint)
    quoted = f"/{urllib.parse.quote(subpath.strip('/'), safe='/')}" if subpath.strip("/") else ""
    url: str | None = f"{base}/api/datasets/{REPO}/tree/{revision}{quoted}?recursive=true&expand=false"
    out: list[tuple[str, int]] = []
    while url:
        body, headers = http_get(url, attempts=attempts)
        for entry in json.loads(body.decode("utf-8")):
            if entry.get("type") == "file":
                out.append((entry["path"], int(entry.get("size") or 0)))
        link = headers.get("link", "")
        nxt = [part for part in link.split(",") if 'rel="next"' in part]
        if nxt:
            raw_next = nxt[0].split(";")[0].strip(" <>")
            url = urllib.parse.urljoin(base + "/", raw_next)
        else:
            url = None
    return out


def _has_whitespace(text: str) -> bool:
    return bool(re.search(r"\s", text or ""))


def _parse_float(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _parse_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def fetch_catalog_jsonl(
    sources_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    attempts: int = 6,
) -> Path:
    """Download `data/catalog.jsonl` into `sources_dir/catalog.jsonl` if not already cached."""
    dest = Path(sources_dir) / "catalog.jsonl"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = hf_resolve_url(CATALOG_RELPATH, revision=revision, endpoint=endpoint)
    data, _ = http_get(url, attempts=attempts)
    _atomic_write_bytes(dest, data)
    return dest


def load_catalog_jsonl(path: Path) -> list[dict]:
    """Parse `data/catalog.jsonl` into normalized row dicts."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        logical_asset = str(item.get("logical_asset") or item.get("asset_id") or "").strip()
        if not logical_asset:
            continue
        bx = _parse_float(item.get("bounds_x_m"))
        by = _parse_float(item.get("bounds_y_m"))
        bz = _parse_float(item.get("bounds_z_m"))
        bounds = [bx, by, bz] if None not in (bx, by, bz) else None
        physics_kind = str(item.get("physics_kind") or "").strip().lower()
        if physics_kind not in PHYSICS_KINDS:
            physics_kind = "articulated" if _parse_bool(item.get("moving_joints")) else "single_body"
        rows.append(
            {
                "logical_asset": logical_asset,
                "asset_id": str(item.get("asset_id") or logical_asset).strip(),
                "catalog_id": str(item.get("catalog_id") or logical_asset).strip(),
                "name": str(item.get("name") or logical_asset).strip(),
                "display_name": str(item.get("display_name") or item.get("name") or logical_asset).strip(),
                "class_label": str(item.get("class_label") or "object").strip(),
                "wikidata_qcode": str(item.get("wikidata_qcode") or "").strip(),
                "batch": str(item.get("batch") or "").strip(),
                "physics_kind": physics_kind,
                "moving_joints": _parse_bool(item.get("moving_joints")),
                "collider_prims": _parse_int(item.get("collider_prims"), 0),
                "bounds_m": bounds,
                "total_mass_kg": _parse_float(item.get("total_mass_kg")),
                "up_axis": str(item.get("up_axis") or "Z").strip().upper(),
                "runtime_status": str(item.get("runtime_status") or "").strip().lower(),
                "static_validation_status": str(item.get("static_validation_status") or "").strip().lower(),
                "package_status": str(item.get("package_status") or "").strip().lower(),
                "source_revision": str(item.get("source_revision") or "").strip(),
                "asset_version": str(item.get("asset_version") or "").strip(),
                "physics_path": str(item.get("physics_path") or "").strip(),
                "physics_download_url": str(item.get("physics_download_url") or "").strip(),
                "physics_sha256": str(item.get("physics_sha256") or "").strip().lower(),
                "package_manifest_url": str(item.get("package_manifest_url") or "").strip(),
            }
        )
    return rows


def category_slug(class_label: str) -> str:
    return "palatial_" + (articulated.slug(class_label or "") or "object")


def robodojo_kind(physics_kind: str) -> str:
    return "Articulation" if physics_kind == "articulated" else "Rigid"


def _matches_any(text: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    low = text.lower()
    for pat in patterns:
        if re.search(pat, low, re.IGNORECASE):
            return True
    return False


def filter_catalog(
    rows: list[dict],
    *,
    physics_kind: str = "all",
    stock_passing_only: bool = True,
    class_labels: list[str] | None = None,
    exclude_class_labels: list[str] | None = None,
    max_rigid_size: float = TABLETOP_MAX_M,
    max_articulated_size: float = TABLESIDE_MAX_M,
    min_size: float = MIN_SIZE_M,
    max_per_category: int | None = None,
    limit: int | None = None,
    asset_ids: set[str] | None = None,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Filter `catalog.jsonl` rows before downloading and assign deterministic RoboDojo IDs."""
    if physics_kind not in PHYSICS_FILTER_CHOICES:
        raise ValueError(f"unsupported physics_kind: {physics_kind}")

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            robodojo_kind(r.get("physics_kind", "single_body")),
            category_slug(r.get("class_label", "")),
            r.get("logical_asset", ""),
        ),
    )

    chosen: list[dict] = []
    dropped: list[dict] = []
    reason_counts: dict[str, int] = {}
    per_cat_counts: dict[str, int] = {}
    id_counters: dict[str, int] = {}

    def _drop(row: dict, reason: str) -> None:
        dropped.append(
            {
                "logical_asset": row.get("logical_asset", ""),
                "asset_id": row.get("asset_id", ""),
                "name": row.get("name", ""),
                "class_label": row.get("class_label", ""),
                "physics_kind": row.get("physics_kind", ""),
                "reason": reason,
            }
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for row in sorted_rows:
        la = row.get("logical_asset", "")
        aid = row.get("asset_id", "")
        pk = row.get("physics_kind", "single_body")
        cls_label = row.get("class_label", "object")

        if not la or _has_whitespace(la):
            _drop(row, "invalid_logical_asset")
            continue

        if asset_ids is not None and la not in asset_ids and aid not in asset_ids:
            _drop(row, "asset_id_filter")
            continue

        if physics_kind == "rigid":
            if pk not in ("single_body", "multi_body"):
                _drop(row, f"physics_kind={pk}")
                continue
        elif physics_kind != "all" and pk != physics_kind:
            _drop(row, f"physics_kind={pk}")
            continue

        if stock_passing_only and row.get("runtime_status") != "passed":
            _drop(row, f"runtime_status={row.get('runtime_status') or 'unknown'}")
            continue

        if row.get("static_validation_status") and row["static_validation_status"] != "passed":
            _drop(row, f"static_validation_status={row['static_validation_status']}")
            continue

        if row.get("package_status") and row["package_status"] != "completed":
            _drop(row, f"package_status={row['package_status']}")
            continue

        if row.get("up_axis", "Z") != "Z":
            _drop(row, f"up_axis={row.get('up_axis')}")
            continue

        if _parse_int(row.get("collider_prims"), 0) <= 0:
            _drop(row, "no_colliders")
            continue

        match_text = f"{cls_label} {row.get('name', '')}"
        if class_labels and not _matches_any(match_text, class_labels):
            _drop(row, "class_label_filter")
            continue
        if exclude_class_labels and _matches_any(match_text, exclude_class_labels):
            _drop(row, "excluded_class_label")
            continue

        bounds = row.get("bounds_m")
        if bounds:
            max_dim = max(bounds)
            min_dim = min(bounds)
            size_cap = max_articulated_size if pk == "articulated" else max_rigid_size
            if max_dim > size_cap:
                _drop(row, f"too_large>{size_cap:.2f}m")
                continue
            if max_dim < min_size:
                _drop(row, f"too_small<{min_size:.3f}m")
                continue
            if min_dim < MIN_THICKNESS_M:
                _drop(row, f"too_thin<{MIN_THICKNESS_M:.3f}m")
                continue

        kind_dir = robodojo_kind(pk)
        cat = category_slug(cls_label)
        cat_key = f"{kind_dir}/{cat}"
        if max_per_category is not None and per_cat_counts.get(cat_key, 0) >= max_per_category:
            _drop(row, "max_per_category")
            continue
        if limit is not None and len(chosen) >= limit:
            _drop(row, "limit")
            continue

        idx = id_counters.get(cat_key, 0)
        id_counters[cat_key] = idx + 1
        per_cat_counts[cat_key] = per_cat_counts.get(cat_key, 0) + 1
        oid = f"{cat_key}/{idx:05d}"

        chosen.append({**row, "id": oid})

    kind_counts: dict[str, int] = {}
    for e in chosen:
        pk = e["physics_kind"]
        kind_counts[pk] = kind_counts.get(pk, 0) + 1

    return {
        "source": canonical_url(),
        "dataset": REPO,
        "revision": revision,
        "license": LICENSE,
        "chosen": chosen,
        "dropped": dropped,
        "summary": {
            "total_rows": len(rows),
            "chosen": len(chosen),
            "dropped": len(dropped),
            "by_physics_kind": kind_counts,
            "drop_reasons": reason_counts,
        },
    }


def _physics_json_url(entry: dict, *, revision: str = DEFAULT_REVISION, endpoint: str | None = None) -> str:
    raw_url = str(entry.get("physics_download_url") or "").strip()
    if raw_url:
        return rewrite_hf_url(raw_url, endpoint=endpoint)
    phys_path = str(entry.get("physics_path") or "").strip()
    if not phys_path:
        return ""
    rev = str(entry.get("source_revision") or revision).strip() or revision
    return hf_resolve_url(phys_path, revision=rev, endpoint=endpoint)


def _list_simready_files_from_bom(
    logical_asset: str,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    attempts: int = 6,
) -> list[tuple[str, str]]:
    """Return `(repo_rel_path, expected_sha256)` for `simready_usd/*` (excluding `.thumbs/`) from BOM JSON."""
    bom_rel = f"{logical_asset}/.metadata/com.nvidia.simready.packaging.bom.json"
    url = hf_resolve_url(bom_rel, revision=revision, endpoint=endpoint)
    try:
        data, _ = http_get(url, attempts=attempts)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    bom = json.loads(data.decode("utf-8"))
    files: list[tuple[str, str]] = []
    for item in bom.get("items", []):
        rel = str(item.get("relative_path") or "").lstrip("/")
        if not rel.startswith("simready_usd/"):
            continue
        if "/.thumbs/" in rel or rel.startswith("simready_usd/.thumbs/"):
            continue
        sha = str((item.get("hash") or {}).get("sha256") or "").strip().lower()
        files.append((f"{logical_asset}/{rel}", sha))
    return files


def fetch_one_package(
    entry: dict,
    out_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    attempts: int = 6,
) -> dict:
    """Download space-free `<logical_asset>/simready_usd/*` and `metadata/physics.json` for one Palatial asset."""
    logical_asset = str(entry.get("logical_asset") or "").strip()
    if not logical_asset or _has_whitespace(logical_asset):
        raise ValueError(f"invalid or whitespace-containing logical_asset: {logical_asset!r}")

    out_dir = Path(out_dir)
    pkg_dir = out_dir / logical_asset
    downloaded: list[str] = []

    def _download_to(url: str, dest: Path, *, expected_sha256: str = "", label: str = "") -> bool:
        if dest.is_file() and dest.stat().st_size > 0:
            if not expected_sha256 or _sha256_file(dest) == expected_sha256:
                return False
        data, _ = http_get(url, attempts=attempts)
        if expected_sha256:
            got = hashlib.sha256(data).hexdigest()
            if got != expected_sha256:
                raise OSError(f"{label or dest.name}: sha256 {got} != expected {expected_sha256}")
        _atomic_write_bytes(dest, data)
        downloaded.append(label or str(dest.relative_to(out_dir)))
        return True

    # 1. Discover `simready_usd/*` files via BOM (includes SHA-256), falling back to HF tree API.
    sim_files = _list_simready_files_from_bom(logical_asset, revision=revision, endpoint=endpoint, attempts=attempts)
    if not any(rel.lower().endswith((".usd", ".usdc", ".usda")) for rel, _ in sim_files):
        tree_entries = list_hf_tree(
            f"{logical_asset}/simready_usd", revision=revision, endpoint=endpoint, attempts=attempts
        )
        sim_files = [
            (rel, "") for rel, _size in tree_entries if "/.thumbs/" not in rel and not rel.endswith("/.thumbs")
        ]

    if not any(rel.lower().endswith((".usd", ".usdc", ".usda")) for rel, _ in sim_files):
        raise FileNotFoundError(f"no USD file found under {logical_asset}/simready_usd/")

    for repo_rel, expected_sha in sim_files:
        if _has_whitespace(repo_rel):
            raise ValueError(f"refusing to fetch path containing whitespace: {repo_rel!r}")
        dest = out_dir / repo_rel
        url = hf_resolve_url(repo_rel, revision=revision, endpoint=endpoint)
        _download_to(url, dest, expected_sha256=expected_sha, label=repo_rel)

    # 2. Separately fetch `physics_*.json` into a guaranteed space-free metadata path.
    phys_url = _physics_json_url(entry, revision=revision, endpoint=endpoint)
    phys_fetched = False
    if phys_url:
        phys_dest = pkg_dir / "metadata" / "physics.json"
        _download_to(
            phys_url,
            phys_dest,
            expected_sha256=str(entry.get("physics_sha256") or "").strip().lower(),
            label=f"{logical_asset}/metadata/physics.json",
        )
        phys_fetched = True

    # 3. Persist catalog entry metadata for standalone `survey` / `convert`.
    meta_dest = pkg_dir / "metadata" / "catalog_entry.json"
    _atomic_write_bytes(
        meta_dest,
        (json.dumps(entry, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    return {
        "logical_asset": logical_asset,
        "id": entry.get("id", ""),
        "simready_files": len(sim_files),
        "downloaded_files": len(downloaded),
        "physics_json": phys_fetched,
    }


def fetch_packages(
    entries: list[dict],
    out_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    workers: int = 4,
    attempts: int = 6,
) -> dict:
    """Download selected Palatial packages into `out_dir` and write `UPSTREAM.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok: list[dict] = []
    failed: list[dict] = []

    if workers <= 1 or len(entries) <= 1:
        for entry in entries:
            try:
                ok.append(fetch_one_package(entry, out_dir, revision=revision, endpoint=endpoint, attempts=attempts))
            except Exception as exc:
                failed.append(
                    {
                        "logical_asset": entry.get("logical_asset", ""),
                        "id": entry.get("id", ""),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    fetch_one_package, entry, out_dir, revision=revision, endpoint=endpoint, attempts=attempts
                ): entry
                for entry in entries
            }
            for fut in as_completed(futs):
                entry = futs[fut]
                try:
                    ok.append(fut.result())
                except Exception as exc:
                    failed.append(
                        {
                            "logical_asset": entry.get("logical_asset", ""),
                            "id": entry.get("id", ""),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    upstream = {
        "dataset": REPO,
        "url": canonical_url(),
        "endpoint": hf_endpoint(endpoint),
        "revision": revision,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "packages_fetched": len(ok),
        "packages_failed": len(failed),
    }
    _atomic_write_bytes(
        out_dir / "UPSTREAM.json",
        (json.dumps(upstream, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return {"fetched": ok, "failed": failed}


def find_package_usd(pkg_dir: Path) -> Path | None:
    """Locate the primary space-free USD file inside `<pkg_dir>/simready_usd/`."""
    sim_dir = Path(pkg_dir) / "simready_usd"
    if not sim_dir.is_dir():
        return None
    preferred = sim_dir / f"{pkg_dir.name}.usd"
    if preferred.is_file():
        return preferred
    cands = sorted(p for p in sim_dir.iterdir() if p.is_file() and p.suffix.lower() in {".usd", ".usdc", ".usda"})
    return cands[0] if cands else None


def discover_packages(source_dir: Path) -> list[Path]:
    """Find downloaded `<logical_asset>/` directories containing `simready_usd/*.usd`."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(source_dir.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and find_package_usd(child) is not None:
            out.append(child)
    return out


def _read_json_optional(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def parse_physics_json(phys: dict) -> dict:
    """Pure-stdlib parser for Palatial `physics_*.json` metadata."""
    asset = phys.get("asset") or {}
    bounds_obj = asset.get("bounds") or {}
    raw_size = bounds_obj.get("size")
    size_m: list[float] | None = None
    if isinstance(raw_size, list) and len(raw_size) == 3:
        parsed = [_parse_float(v) for v in raw_size]
        if None not in parsed:
            size_m = [round(float(v), 4) for v in parsed]  # type: ignore[arg-type]

    art = phys.get("articulation") or {}
    joints_raw = art.get("joints") or []
    movable_joints: list[dict] = []
    all_joints: list[dict] = []
    child_bodies: set[str] = set()
    parent_bodies: set[str] = set()

    for j in joints_raw:
        usd_type = str(j.get("usd_type") or (j.get("authored_usd") or {}).get("joint_type") or "").lower()
        jtype = str(j.get("joint_type") or "").lower()
        if "revolute" in usd_type or jtype == "revolute":
            kind = "revolute"
        elif "prismatic" in usd_type or jtype == "prismatic":
            kind = "prismatic"
        elif "fixed" in usd_type or jtype == "fixed":
            kind = "fixed"
        else:
            kind = jtype or usd_type or "unknown"

        b0_list = j.get("body0_targets") or (j.get("authored_usd") or {}).get("body0") or []
        b1_list = j.get("body1_targets") or (j.get("authored_usd") or {}).get("body1") or []
        b0 = str(b0_list[0]).rsplit("/", 1)[-1] if b0_list else str(j.get("parent") or "")
        b1 = str(b1_list[0]).rsplit("/", 1)[-1] if b1_list else str(j.get("child") or "")
        enabled = j.get("enabled", True) is not False

        j_entry = {
            "id": str(j.get("id") or j.get("name") or ""),
            "type": kind,
            "body0": b0,
            "body1": b1,
            "enabled": enabled,
        }
        all_joints.append(j_entry)
        if b0 and b1:
            parent_bodies.add(b0)
            child_bodies.add(b1)
        if kind in ("revolute", "prismatic") and b0 and b1 and enabled:
            movable_joints.append(j_entry)

    parts_raw = phys.get("parts") or []
    part_masses: dict[str, float] = {}
    part_labels: dict[str, str] = {}
    low_mass_parts: list[str] = []
    for idx, part in enumerate(parts_raw):
        src_prim = str(part.get("source_prim_path") or "").strip()
        prim_leaf = src_prim.rsplit("/", 1)[-1] if src_prim else ""
        pid = str(part.get("id") or prim_leaf or part.get("name") or f"part_{idx}").strip()
        pname = str(part.get("name") or "").strip()
        if pname and pname != pid:
            part_labels[pid] = pname
        pm = _parse_float(part.get("mass"))
        if pm is not None:
            part_masses[pid] = pm
            if pm < MIN_LINK_MASS_KG:
                low_mass_parts.append(pid)

    base_candidates = sorted(parent_bodies - child_bodies)
    return {
        "description": str(asset.get("description") or "").strip(),
        "total_mass_kg": _parse_float(asset.get("total_mass")),
        "size_m": size_m,
        "object_type": str(asset.get("object_type") or "").strip(),
        "rigid_body_count": _parse_int(art.get("rigid_body_count"), len(parts_raw) or 1),
        "has_joints": bool(art.get("has_joints") or all_joints),
        "joint_count": len(all_joints),
        "movable_joint_count": len(movable_joints),
        "movable_joints": movable_joints,
        "base_candidates": base_candidates,
        "part_masses": part_masses,
        "part_labels": part_labels,
        "low_mass_parts": low_mass_parts,
    }


def survey_package(pkg_dir: Path, catalog_row: dict | None = None) -> tuple[bool, str, dict]:
    """Pure-stdlib inspection of a downloaded Palatial package (no `pxr`, no `numpy`)."""
    pkg_dir = Path(pkg_dir)
    la = pkg_dir.name
    cat = _read_json_optional(pkg_dir / "metadata" / "catalog_entry.json")
    if catalog_row:
        cat = {**cat, **catalog_row}
    phys_raw = _read_json_optional(pkg_dir / "metadata" / "physics.json")
    phys = parse_physics_json(phys_raw) if phys_raw else {}

    usd_path = find_package_usd(pkg_dir)
    if usd_path is None:
        return False, "missing simready_usd/*.usd", {"logical_asset": la}

    rel_usd = usd_path.relative_to(pkg_dir).as_posix()
    if _has_whitespace(la) or _has_whitespace(rel_usd):
        return False, f"whitespace in USD path: {la}/{rel_usd}", {"logical_asset": la}

    tex_dir = pkg_dir / "simready_usd" / "textures"
    tex_files: list[str] = []
    if tex_dir.is_dir():
        for p in sorted(tex_dir.rglob("*")):
            if p.is_file():
                rel_tex = p.relative_to(pkg_dir).as_posix()
                if _has_whitespace(rel_tex):
                    return False, f"whitespace in texture path: {rel_tex}", {"logical_asset": la}
                tex_files.append(rel_tex)

    physics_kind = str(cat.get("physics_kind") or "").strip().lower()
    if physics_kind not in PHYSICS_KINDS:
        if phys.get("movable_joint_count", 0) > 0:
            physics_kind = "articulated"
        elif phys.get("rigid_body_count", 1) > 1:
            physics_kind = "multi_body"
        else:
            physics_kind = "single_body"

    size_m = phys.get("size_m") or cat.get("bounds_m")
    if not size_m or len(size_m) != 3 or any(v is None or v <= 0 for v in size_m):
        return False, "missing or invalid bounds", {"logical_asset": la, "physics_kind": physics_kind}

    up_axis = str(cat.get("up_axis") or "Z").strip().upper()
    if up_axis != "Z":
        return False, f"up_axis={up_axis}", {"logical_asset": la, "physics_kind": physics_kind}

    info = {
        "logical_asset": la,
        "asset_id": str(cat.get("asset_id") or la),
        "catalog_id": str(cat.get("catalog_id") or la),
        "name": str(cat.get("name") or la),
        "display_name": str(cat.get("display_name") or cat.get("name") or la),
        "class_label": str(cat.get("class_label") or "object"),
        "wikidata_qcode": str(cat.get("wikidata_qcode") or ""),
        "batch": str(cat.get("batch") or ""),
        "physics_kind": physics_kind,
        "moving_joints": physics_kind == "articulated",
        "size_m": [round(float(v), 4) for v in size_m],
        "total_mass_kg": (
            phys.get("total_mass_kg") if phys.get("total_mass_kg") is not None else cat.get("total_mass_kg")
        ),
        "description": phys.get("description") or "",
        "usd_rel": f"{la}/{rel_usd}",
        "texture_count": len(tex_files),
        "multi_body_rigid_flatten": physics_kind == "multi_body",
        "base_candidates": phys.get("base_candidates", []),
        "part_labels": phys.get("part_labels", {}),
        "low_mass_parts": phys.get("low_mass_parts", []),
        "movable_joint_count": phys.get("movable_joint_count", 0),
    }

    if physics_kind == "articulated" and phys_raw and phys.get("movable_joint_count", 0) <= 0:
        return False, "articulated asset has no enabled revolute/prismatic joints in physics.json", info

    return True, "ok", info


def survey(
    source_dir: Path,
    *,
    selection: dict | None = None,
    max_rigid_size: float = TABLETOP_MAX_M,
    max_articulated_size: float = TABLESIDE_MAX_M,
    min_size: float = MIN_SIZE_M,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Pure-stdlib survey of downloaded Palatial packages, emitting a deterministic conversion manifest."""
    source_dir = Path(source_dir)
    by_la: dict[str, dict] = {}
    chosen: list[dict] = []
    dropped: list[dict] = []

    if selection:
        for entry in selection.get("chosen", []):
            by_la[entry["logical_asset"]] = entry
        pkg_dirs = []
        for la in sorted(by_la):
            pkg = source_dir / la
            if pkg.is_dir():
                pkg_dirs.append(pkg)
            else:
                row = by_la[la]
                dropped.append(
                    {
                        "logical_asset": la,
                        "id": row.get("id", ""),
                        "physics_kind": row.get("physics_kind", ""),
                        "reason": "package directory missing (fetch failed or not downloaded)",
                    }
                )
    else:
        pkg_dirs = discover_packages(source_dir)

    counters: dict[str, int] = {}

    for pkg_dir in pkg_dirs:
        la = pkg_dir.name
        sel_row = by_la.get(la, {})
        ok, reason, info = survey_package(pkg_dir, catalog_row=sel_row)
        if ok:
            size_m = info["size_m"]
            size_cap = max_articulated_size if info["physics_kind"] == "articulated" else max_rigid_size
            if max(size_m) > size_cap:
                ok, reason = False, f"too_large>{size_cap:.2f}m"
            elif max(size_m) < min_size:
                ok, reason = False, f"too_small<{min_size:.3f}m"
            elif min(size_m) < MIN_THICKNESS_M:
                ok, reason = False, f"too_thin<{MIN_THICKNESS_M:.3f}m"

        if not ok:
            dropped.append({**info, "id": sel_row.get("id", ""), "reason": reason})
            continue

        if sel_row.get("id"):
            oid = sel_row["id"]
        else:
            key = f"{robodojo_kind(info['physics_kind'])}/{category_slug(info['class_label'])}"
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            oid = f"{key}/{idx:05d}"

        chosen.append({**info, "id": oid})

    kind_counts: dict[str, int] = {}
    for e in chosen:
        pk = e["physics_kind"]
        kind_counts[pk] = kind_counts.get(pk, 0) + 1

    return {
        "source": canonical_url(),
        "dataset": REPO,
        "revision": selection.get("revision", revision) if selection else revision,
        "license": LICENSE,
        "chosen": chosen,
        "dropped": dropped,
        "summary": {
            "chosen": len(chosen),
            "dropped": len(dropped),
            "by_physics_kind": kind_counts,
        },
    }


def is_fixed_base(class_label: str, name: str, size: list[float] | tuple[float, ...]) -> bool:
    """Heuristic for metadata `physics.fixed_base`: portable articulated items stay free; appliances stay fixed."""
    tokens = set(re.findall(r"[a-z]+", f"{class_label} {name}".lower()))
    if tokens & set(PORTABLE_TOKENS) and max(size) <= 0.45:
        return False
    if max(size) <= 0.25:
        return False
    return True


def resolve_collision_free_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    idx = 1
    while f"{name}_{idx}" in used:
        idx += 1
    return f"{name}_{idx}"


def _scale_with_floor(raw: dict[str, float], target: float, floor: float) -> dict[str, float]:
    """Scale positive weights in `raw` to sum to `target` while keeping every entry >= `floor`."""
    n = len(raw)
    if n == 0:
        return {}
    if target <= floor * n + 1e-12:
        return {k: floor for k in raw}
    total_raw = sum(raw.values())
    if total_raw <= 0:
        return {k: target / n for k in raw}
    vals = {k: v * (target / total_raw) for k, v in raw.items()}
    pinned: set[str] = set()
    while True:
        newly_pinned = [k for k, v in vals.items() if k not in pinned and v < floor]
        if not newly_pinned:
            break
        for k in newly_pinned:
            vals[k] = floor
            pinned.add(k)
        remaining_keys = [k for k in raw if k not in pinned]
        if not remaining_keys:
            break
        rem_target = target - floor * len(pinned)
        rem_raw = sum(raw[k] for k in remaining_keys)
        if rem_target <= 0 or rem_raw <= 0:
            for k in remaining_keys:
                vals[k] = floor
            break
        for k in remaining_keys:
            vals[k] = raw[k] * (rem_target / rem_raw)
    return vals


def clamp_link_masses(
    links: list[str],
    authored_masses: dict[str, float],
    volumes: dict[str, float],
    *,
    max_total_mass: float,
    min_total_mass: float = DEFAULT_MIN_ARTICULATED_MASS_KG,
    min_link_mass: float = MIN_LINK_MASS_KG,
) -> tuple[dict[str, float], dict]:
    """Compute final per-link masses, clamping tiny links >= `min_link_mass` (0.02 kg) and total within bounds."""
    if not links:
        raise ValueError("no links to assign mass to")
    if max_total_mass <= 0 or min_total_mass <= 0 or min_total_mass > max_total_mass:
        raise ValueError(f"invalid mass bounds: [{min_total_mass}, {max_total_mass}]")

    eff_min_link = min(min_link_mass, max_total_mass / len(links))
    valid_authored = {
        p: float(m) for p, m in authored_masses.items() if p in links and m is not None and math.isfinite(m) and m > 0
    }
    raw_total = sum(valid_authored.values())
    free = [p for p in links if p not in valid_authored]

    masses: dict[str, float] = {}
    clamped = False
    filled = bool(free)

    if not free:
        target_total = raw_total
        if raw_total > max_total_mass:
            target_total = max_total_mass
            clamped = True
        elif raw_total < min_total_mass:
            target_total = min_total_mass
            clamped = True

        if clamped:
            masses = _scale_with_floor(valid_authored, target_total, eff_min_link)
        else:
            masses = {p: max(eff_min_link, valid_authored[p]) for p in links}
            if any(masses[p] > valid_authored[p] + 1e-9 for p in links):
                clamped = True
            if sum(masses.values()) > max_total_mass + 1e-9:
                masses = _scale_with_floor(valid_authored, max_total_mass, eff_min_link)
                clamped = True
    else:
        min_free_budget = eff_min_link * len(free)
        max_authored_budget = max(eff_min_link * len(valid_authored), max_total_mass - min_free_budget)
        floored_authored = {p: max(eff_min_link, m) for p, m in valid_authored.items()}
        if valid_authored and any(floored_authored[p] > valid_authored[p] + 1e-9 for p in valid_authored):
            clamped = True
        if valid_authored and sum(floored_authored.values()) > max_authored_budget + 1e-9:
            masses.update(_scale_with_floor(valid_authored, max_authored_budget, eff_min_link))
            clamped = True
        else:
            masses.update(floored_authored)

        authored_sum = sum(masses.values())
        free_vols = {p: max(float(volumes.get(p, 0.005**3)), 1e-9) for p in free}
        default_free_total = max(min_free_budget, sum(free_vols.values()) * 400.0)
        rem_max = max(min_free_budget, max_total_mass - authored_sum)
        rem_min = max(min_free_budget, min_total_mass - authored_sum)
        target_free = min(max(default_free_total, rem_min), rem_max)
        masses.update(_scale_with_floor(free_vols, target_free, eff_min_link))

    final_total = sum(masses.values())
    return masses, {
        "raw_authored_kg": round(raw_total, 4),
        "final_kg": round(final_total, 4),
        "clamped": clamped,
        "filled": filled,
        "min_link_kg": round(eff_min_link, 4),
        "bounds_kg": [min_total_mass, max_total_mass],
    }


def _reparent_links_to_root(flat, src_root: str, *, promote_links: bool = True) -> list[tuple[str, str]]:
    """Optionally promote nested `RigidBodyAPI` links to direct children of `/root` and rename `src_root` -> `/root`.

    Preserves inherited visual `MaterialBindingAPI` bindings from ancestor containers before reparenting,
    and explicitly rewrites BOTH `RelationshipSpec.targetPathList` (e.g. `physics:body0`/`body1`, `material:binding`)
    AND `AttributeSpec.connectionPathList` (e.g. shader/material attribute connections) because
    `Sdf.Layer.Apply(BatchNamespaceEdit)` in USD 24.x does not always retarget them.
    """
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, _ = pxr_modules()

    tmp_stage = Usd.Stage.Open(flat)
    root_path = Sdf.Path(src_root)
    root_prim = tmp_stage.GetPrimAtPath(root_path)
    if not root_prim:
        raise ValueError(f"missing root prim {src_root}")

    nested_links = (
        [
            p
            for p in Usd.PrimRange(root_prim)
            if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and p.GetPath().GetParentPath() != root_path
            and p.GetPath() != root_path
        ]
        if promote_links
        else []
    )

    # Fold non-identity ancestor transforms and inherited visual material bindings BEFORE reparenting.
    anc_transforms: list[tuple[object, object]] = []
    inherited_mats: list[tuple[object, object]] = []
    for link in nested_links:
        if UsdGeom.Xformable(link) and not UsdGeom.Xformable(link).GetResetXformStack():
            anc_xf = None
            cur = link.GetParent()
            chain = []
            while cur and cur.GetPath() != root_path:
                if UsdGeom.Xformable(cur):
                    cur_xf = UsdGeom.Xformable(cur)
                    mat = cur_xf.GetLocalTransformation()
                    if mat != Gf.Matrix4d(1.0):
                        chain.append(mat)
                    if cur_xf.GetResetXformStack():
                        break
                cur = cur.GetParent()
            for mat in chain:
                anc_xf = mat if anc_xf is None else (anc_xf * mat)
            if anc_xf is not None and anc_xf != Gf.Matrix4d(1.0):
                anc_transforms.append((link, anc_xf))

        # Preserve inherited visual material binding from any ancestor container between `src_root` and `link`.
        link_binding = UsdShade.MaterialBindingAPI(link).GetDirectBinding()
        if not link_binding.GetMaterialPath():
            cur = link.GetParent()
            while cur and cur.GetPath() != root_path:
                anc_binding = UsdShade.MaterialBindingAPI(cur).GetDirectBinding()
                mat_path = anc_binding.GetMaterialPath()
                if mat_path:
                    mat_prim = tmp_stage.GetPrimAtPath(mat_path)
                    if mat_prim:
                        inherited_mats.append((link, UsdShade.Material(mat_prim)))
                    break
                cur = cur.GetParent()

    for link, anc_xf in anc_transforms:
        xf = UsdGeom.Xformable(link)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ofw_reparent")
        op.Set(anc_xf)
        xf.SetXformOpOrder([op, *ops], reset)

    for link, mat in inherited_mats:
        UsdShade.MaterialBindingAPI.Apply(link).Bind(mat)

    moved: list[tuple[str, str]] = []
    if nested_links:
        parent_containers = {p.GetPath().GetParentPath() for p in nested_links}
        used_names = {c.GetName() for c in root_prim.GetChildren()}
        ordered_links = sorted(nested_links, key=lambda p: len(p.GetPath().pathString), reverse=True)
        edit = Sdf.BatchNamespaceEdit()
        for link in ordered_links:
            old_path = link.GetPath()
            dest_name = resolve_collision_free_name(link.GetName(), used_names)
            used_names.add(dest_name)
            new_path = root_path.AppendChild(dest_name)
            edit.Add(old_path, new_path)
            moved.append((str(old_path), str(new_path)))
        if not flat.CanApply(edit):
            raise ValueError(f"cannot reparent nested links: {moved}")
        flat.Apply(edit)

        # Remove now-empty container prims (e.g. `/World/textured_mesh_...` if emptied).
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

    remap_pairs: list[tuple[Sdf.Path, Sdf.Path]] = []
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
    remap_pairs.sort(key=lambda pair: len(pair[0].pathString), reverse=True)

    def _remap_path(p):
        for src_p, dst_p in remap_pairs:
            if p == src_p or p.HasPrefix(src_p):
                return p.ReplacePrefix(src_p, dst_p)
        return p

    def _rewrite_specs(prim_spec):
        for rel in prim_spec.relationships:
            targets = list(rel.targetPathList.GetAddedOrExplicitItems())
            if targets:
                updated = [_remap_path(t) for t in targets]
                if updated != targets:
                    rel.targetPathList.ClearEdits()
                    rel.targetPathList.explicitItems = updated
        for attr in prim_spec.attributes:
            conns = list(attr.connectionPathList.GetAddedOrExplicitItems())
            if conns:
                updated = [_remap_path(c) for c in conns]
                if updated != conns:
                    attr.connectionPathList.ClearEdits()
                    attr.connectionPathList.explicitItems = updated
        for child in prim_spec.nameChildren:
            _rewrite_specs(child)

    for top in flat.rootPrims:
        _rewrite_specs(top)

    flat.defaultPrim = "root"
    return moved


def _copy_and_rewrite_textures(flat, source_usd: Path, target_dir: Path) -> list[str]:
    """Copy referenced external textures into `target_dir/textures/` with space-free `./textures/...` paths."""
    _, Sdf, _, _, _, _, _ = pxr_modules()

    src_dir = source_usd.parent.resolve()
    copied: list[str] = []
    seen_map: dict[Path, str] = {}
    used_rels: set[str] = set()

    def _sanitize_rel(rel: Path) -> Path:
        parts = [re.sub(r"\s+", "_", part) for part in rel.parts]
        return Path(*parts)

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
        try:
            rel = cand.relative_to(src_dir)
        except ValueError:
            rel = Path("textures") / cand.name
        if rel.parts[0] != "textures":
            rel = Path("textures") / rel.name
        rel = _sanitize_rel(rel)
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


def _assert_no_spaces_in_asset_paths(layer) -> None:
    """Verify no authored `Sdf.AssetPath` in the exported USD layer contains whitespace."""
    _, Sdf, _, _, _, _, _ = pxr_modules()

    bad: list[str] = []

    def _visit(prim_spec):
        for prop in prim_spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec):
                continue
            val = prop.default
            if isinstance(val, Sdf.AssetPath) and _has_whitespace(val.path):
                bad.append(f"{prim_spec.path}.{prop.name}={val.path!r}")
            elif isinstance(val, Sdf.AssetPathArray):
                for item in val:
                    if _has_whitespace(item.path):
                        bad.append(f"{prim_spec.path}.{prop.name}={item.path!r}")
        for child in prim_spec.nameChildren:
            _visit(child)

    for top in layer.rootPrims:
        _visit(top)
    if bad:
        raise ValueError(f"exported object.usd references paths with whitespace: {bad}")


def _joint_axis_vertical(stage, prim) -> bool:
    """Safe wrapper around `articulated._axis_vertical` handling missing `localRot0`."""
    Gf, _, _, UsdGeom, UsdPhysics, _, _ = pxr_modules()
    joint = UsdPhysics.Joint(prim)
    targets = joint.GetBody0Rel().GetTargets()
    if not targets:
        return False
    body0 = stage.GetPrimAtPath(targets[0])
    world = UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(0)
    raw_rot = joint.GetLocalRot0Attr().Get()
    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(raw_rot))) if raw_rot is not None else Gf.Matrix4d(1)
    axis_tok = prim.GetAttribute("physics:axis").Get() or "Z"
    axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}.get(axis_tok, Gf.Vec3d(0, 0, 1))
    v = world.TransformDir(rot.TransformDir(axis))
    return abs(v.GetNormalized()[2]) > 0.7


def _strip_physics_scenes(stage) -> list[str]:
    """Remove any embedded UsdPhysics.Scene prims (and top-level non-/root PhysicsScenes) from a flattened stage.

    Leaving a PhysicsScene inside an object USD causes Isaac Sim / PhysX to crash with
    `Physics scenes stepping is not the same` when spawning into Isaac Lab's world.
    """
    _, _, _, _, UsdPhysics, _, _ = pxr_modules()
    removed: list[str] = []
    # Remove any PhysicsScene anywhere in the stage (deepest first).
    scenes = [p.GetPath() for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    for path in sorted(scenes, key=lambda p: len(p.pathString), reverse=True):
        if stage.RemovePrim(path):
            removed.append(str(path))
    return removed


def _author_joint_axis_friction(stage, joints: list[dict], facts: dict, masses: dict[str, float], spans: dict) -> None:
    """Author `PhysxJointAxisAPI` (`staticFrictionEffort` / `dynamicFrictionEffort`) on movable joints.

    Palatial only authors legacy `physxJoint:jointFriction`. For openable roles (`door`, `drawer`, `lid`),
    we ensure at least `articulated.STILL_FRICTION` so parts hold still under gravity; for other movable joints
    we carry over positive legacy `physxJoint:jointFriction` or apply a light default.
    """
    _, Sdf, _, _, _, _, _ = pxr_modules()
    by_name = {j["name"]: j for j in facts["movable"]}
    for entry in joints:
        existing = entry.get("friction") or {}
        if "static" in existing and "dynamic" in existing:
            continue
        joint = by_name[entry["name"]]
        prim = stage.GetPrimAtPath(joint["path"])
        token = "angular" if entry["type"] == "revolute" else "linear"

        legacy_attr = prim.GetAttribute("physxJoint:jointFriction")
        legacy_val = 0.0
        if legacy_attr and legacy_attr.HasAuthoredValue():
            parsed = _parse_float(legacy_attr.Get(), 0.0)
            if parsed is not None and parsed > 0:
                legacy_val = parsed

        if entry["role"] in ("door", "drawer", "lid"):
            if entry["type"] == "prismatic":
                floor_fric = articulated.STILL_FRICTION["prismatic"]
            elif _joint_axis_vertical(stage, prim):
                floor_fric = articulated.STILL_FRICTION["upright"]
            else:
                span = spans.get(joint["body1"]) or [0.1, 0.1, 0.1]
                floor_fric = round(
                    articulated.STILL_FRICTION["level"] * masses.get(joint["body1"], 0.1) * 9.81 * max(span) / 2, 3
                )
            friction = round(max(legacy_val, floor_fric), 4)
            source = "legacy_or_still_floor" if legacy_val > floor_fric else "estimated (still_joints)"
        else:
            if legacy_val > 0:
                friction = round(legacy_val, 4)
                source = "migrated (physxJoint:jointFriction)"
            else:
                friction = DEFAULT_NON_OPENING_FRICTION[entry["type"]]
                source = "default (non_opening)"

        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{token}")
        for attr_name in ("staticFrictionEffort", "dynamicFrictionEffort"):
            prim.CreateAttribute(f"physxJointAxis:{token}:{attr_name}", Sdf.ValueTypeNames.Float).Set(float(friction))

        fric_meta = {**existing, "static": friction, "dynamic": friction, "source": source}
        if legacy_val > 0:
            fric_meta["legacy_joint_friction"] = round(legacy_val, 4)
        entry["friction"] = fric_meta


def adapt_articulated(
    source_usd: Path,
    target_dir: Path,
    *,
    class_label: str,
    name: str = "",
    part_labels: dict[str, str] | None = None,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float | None = None,
    min_mass: float = DEFAULT_MIN_ARTICULATED_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Flatten, reparent, reorder true `body0` base link first, clamp masses >= 0.02 kg, and author joint friction."""
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, _, UsdUtils = pxr_modules()

    source_usd = Path(source_usd).resolve()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(source_usd))
    src_facts = articulated.inspect(src_stage)
    if src_facts["up_axis"] != "Z":
        raise ValueError(f"up axis {src_facts['up_axis']} (only Z-up sources are adapted)")
    if not src_facts["movable"]:
        raise ValueError("no revolute or prismatic joint between two links")
    if src_facts["world_movable"]:
        raise ValueError(f"joint anchored to the world instead of a link: {src_facts['world_movable'][0]['path']}")
    if len(src_facts["bases"]) != 1:
        raise ValueError(f"cannot tell the base link: {src_facts['bases']}")
    if not src_facts["colliders"]:
        raise ValueError("no enabled collision geometry")

    art_attrs: dict[str, tuple] = {}
    for anc in reversed(list(Sdf.Path(src_facts["root"]).GetAncestorsRange())[1:]):
        above = src_stage.GetPrimAtPath(anc)
        for attr in above.GetAttributes() if above else ():
            if attr.GetName().startswith(articulated.PHYSX_ARTICULATION + ":") and attr.HasAuthoredValue():
                art_attrs[attr.GetName()] = (attr.GetTypeName(), attr.Get())

    flat = src_stage.Flatten()
    moved_links = _reparent_links_to_root(flat, src_facts["root"])
    copied_textures = _copy_and_rewrite_textures(flat, source_usd, target_dir)

    stage = Usd.Stage.Open(flat)
    _strip_physics_scenes(stage)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetPrimAtPath("/root")
    if root.GetTypeName() != "Xform":
        root.SetTypeName("Xform")
    stage.SetDefaultPrim(root)

    facts = articulated.inspect(stage)
    if len(facts["bases"]) != 1:
        raise ValueError(f"base link ambiguous after reparenting: {facts['bases']}")
    base_path = Sdf.Path(facts["bases"][0])
    if base_path.GetParentPath() != root.GetPath():
        raise ValueError(f"base link {base_path} is not a direct child of /root")

    # Move ArticulationRootAPI onto /root; apply PhysxArticulationAPI to BOTH /root and true base link.
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

    # Put true `body0` base link FIRST among /root's children (`fix_root_link` picks the first Xform+RigidBodyAPI).
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
    lo, hi = articulated._bound(cache, root)
    size = hi - lo
    if np.linalg.norm((lo + hi) / 2) > 1e-4:
        raise ValueError(f"origin not at the bounding-box centre after shift: {(lo + hi) / 2}")

    volumes, spans = {}, {}
    for path in facts["links"]:
        b = articulated._bound(cache, stage.GetPrimAtPath(path))
        spans[path] = [float(v) for v in b[1] - b[0]] if b else None
        volumes[path] = float(np.prod(np.maximum(b[1] - b[0], 0.005))) if b else 0.005**3

    fixed_base = is_fixed_base(class_label, name, [float(v) for v in size])
    if max_mass is None:
        if not fixed_base:
            eff_max_mass = DEFAULT_MAX_PORTABLE_MASS_KG
        elif float(np.max(size)) <= TABLETOP_MAX_M:
            eff_max_mass = DEFAULT_MAX_TABLETOP_MASS_KG
        else:
            eff_max_mass = DEFAULT_MAX_TABLESIDE_MASS_KG
    else:
        eff_max_mass = max_mass

    masses, mass_info = clamp_link_masses(
        facts["links"],
        facts["masses"],
        volumes,
        max_total_mass=eff_max_mass,
        min_total_mass=min_mass,
        min_link_mass=MIN_LINK_MASS_KG,
    )
    for path, new_m in masses.items():
        prim = stage.GetPrimAtPath(path)
        old_m = facts["masses"].get(path)
        if old_m is None or abs(new_m - old_m) > 1e-6:
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(float(new_m))
            if old_m and old_m > 0:
                inertia_attr = api.GetDiagonalInertiaAttr()
                if inertia_attr and inertia_attr.HasAuthoredValue():
                    diag = inertia_attr.Get()
                    if diag is not None:
                        ratio = float(new_m / old_m)
                        inertia_attr.Set(Gf.Vec3f(diag[0] * ratio, diag[1] * ratio, diag[2] * ratio))

    joints = []
    for joint in facts["movable"]:
        lower, upper = joint["lower"], joint["upper"]
        if joint["path"] in limit_fixes:
            lower, upper = limit_fixes[joint["path"]]
        elif joint["type"] == "prismatic" and unit != 1.0:
            lower, upper = (None if v is None else v * unit for v in (lower, upper))
        child = joint["body1"].rsplit("/", 1)[-1]
        semantic_label = (part_labels or {}).get(child, "")
        part = articulated.part_of(semantic_label, joint["type"]) if semantic_label else "other"
        if part == "other":
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

    _author_joint_axis_friction(stage, joints, facts, masses, spans)

    out_usd = target_dir / "object.usd"
    flat.Export(str(out_usd), args={"format": usd_format})

    reopened = Usd.Stage.Open(str(out_usd))
    _assert_no_spaces_in_asset_paths(reopened.GetRootLayer())
    re_root = reopened.GetDefaultPrim()
    if not re_root or re_root.GetPath() != Sdf.Path("/root") or not re_root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("exported object.usd missing /root with ArticulationRootAPI")
    first_child = re_root.GetChildren()[0] if re_root.GetChildren() else None
    if not first_child or first_child.GetTypeName() != "Xform" or not first_child.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"first child of /root is not an Xform with RigidBodyAPI: {first_child}")
    if first_child.GetName() != base_path.name:
        raise ValueError(f"first child {first_child.GetName()} != base link {base_path.name}")
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
        "reparented_links": moved_links,
        "textures": copied_textures,
        "base_link": base_path.name,
        "links": [p.rsplit("/", 1)[-1] for p in facts["links"]],
    }


def adapt_rigid(
    source_usd: Path,
    target_dir: Path,
    *,
    catalog_mass_kg: float | None = None,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float = DEFAULT_MAX_RIGID_MASS_KG,
    min_mass: float = DEFAULT_MIN_RIGID_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Adapt a `single_body` or `multi_body` Palatial USD into a single-body RoboDojo `RigidObject` (`object.usd`).

    Strips `PhysicsArticulationRootAPI` / `PhysxArticulationAPI` everywhere and strips descendant
    `RigidBodyAPI` / `PhysxRigidBodyAPI` / `MassAPI` / physics material bindings so `multi_body` items
    with loose lids or stray articulation roots simulate cleanly as single rigid bodies.
    """
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = pxr_modules()

    source_usd = Path(source_usd).resolve()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(source_usd))
    src_facts = articulated.inspect(src_stage)
    if src_facts["up_axis"] != "Z":
        raise ValueError(f"up axis {src_facts['up_axis']} (only Z-up sources are adapted)")
    if not src_facts["colliders"]:
        raise ValueError("no enabled collision geometry")

    raw_mass = sum(src_facts["masses"].values())
    if raw_mass <= 0 and catalog_mass_kg and catalog_mass_kg > 0:
        raw_mass = float(catalog_mass_kg)

    flat = src_stage.Flatten()
    _reparent_links_to_root(flat, src_facts["root"], promote_links=False)
    copied_textures = _copy_and_rewrite_textures(flat, source_usd, target_dir)

    stage = Usd.Stage.Open(flat)
    _strip_physics_scenes(stage)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetPrimAtPath("/root")
    if root.GetTypeName() != "Xform":
        root.SetTypeName("Xform")
    stage.SetDefaultPrim(root)

    stripped_art_roots: list[str] = []
    stripped_child_bodies: list[str] = []

    if root.HasAPI(UsdPhysics.ArticulationRootAPI):
        root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        stripped_art_roots.append("/root")
    root.RemoveAppliedSchema("PhysxArticulationAPI")
    if root.HasAttribute(articulated.SCRIPT_ATTR) and root.GetAttribute(articulated.SCRIPT_ATTR).Get():
        root.GetAttribute(articulated.SCRIPT_ATTR).Block()

    for prim in list(Usd.PrimRange(root)):
        if prim.GetPath() == root.GetPath():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            stripped_child_bodies.append(str(prim.GetPath()))
        prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            stripped_art_roots.append(str(prim.GetPath()))
        prim.RemoveAppliedSchema("PhysxArticulationAPI")
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() in articulated.LIGHT_TYPES:
            prim.SetActive(False)
        if prim.HasAttribute(articulated.SCRIPT_ATTR) and prim.GetAttribute(articulated.SCRIPT_ATTR).Get():
            prim.GetAttribute(articulated.SCRIPT_ATTR).Block()
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding(materialPurpose="physics")

    unit = src_facts["meters_per_unit"] * scale
    if unit <= 0:
        raise ValueError("bad metersPerUnit")
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
    lo, hi = articulated._bound(cache, root)
    size = hi - lo
    if np.linalg.norm((lo + hi) / 2) > 1e-4:
        raise ValueError(f"origin not at bounding-box centre: {(lo + hi) / 2}")

    mass = raw_mass if raw_mass > 0 else float(np.prod(np.maximum(size, 0.01)) * 500.0)
    clamped_mass = round(min(max(mass, min_mass), max_mass), 4)
    if root.HasAPI(UsdPhysics.MassAPI):
        root.RemoveAPI(UsdPhysics.MassAPI)
    for stale_attr in (
        "physics:mass",
        "physics:density",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ):
        if root.HasAttribute(stale_attr):
            root.RemoveProperty(stale_attr)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(clamped_mass))

    out_usd = target_dir / "object.usd"
    flat.Export(str(out_usd), args={"format": usd_format})

    reopened = Usd.Stage.Open(str(out_usd))
    _assert_no_spaces_in_asset_paths(reopened.GetRootLayer())
    bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies != ["/root"]:
        raise ValueError(f"rigid bodies other than /root remain: {bodies}")
    art_roots = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if art_roots:
        raise ValueError(f"ArticulationRootAPI remains on rigid object: {art_roots}")
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
        "stripped_articulation_roots": stripped_art_roots,
        "stripped_child_bodies": stripped_child_bodies,
        "textures": copied_textures,
    }


def convert_one(
    source_dir: Path,
    out_root: Path,
    entry: dict,
    *,
    revision: str = DEFAULT_REVISION,
    max_mass: float | None = None,
    usd_format: str = "usdc",
) -> dict:
    """Convert one surveyed Palatial entry into `out_root/Assets/Object/RoboDojo/<id>/`."""
    source_dir = Path(source_dir)
    out_root = Path(out_root)
    la = entry["logical_asset"]
    pkg_dir = source_dir / la
    source_usd = (source_dir / entry["usd_rel"]) if entry.get("usd_rel") else find_package_usd(pkg_dir)
    if source_usd is None or not source_usd.is_file():
        raise FileNotFoundError(f"USD file not found for {la} under {pkg_dir}")

    oid = entry["id"]
    target = out_root / "Assets/Object/RoboDojo" / oid
    pk = entry.get("physics_kind", "single_body")

    source_provenance = {
        "dataset": REPO,
        "url": canonical_url(),
        "object": la,
        "revision": revision,
        "logical_asset": la,
        "asset_id": entry.get("asset_id", la),
        "catalog_id": entry.get("catalog_id", la),
        "class_label": entry.get("class_label", ""),
        "wikidata_qcode": entry.get("wikidata_qcode", ""),
        "physics_kind": pk,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "rsi_pools": "allowed (CC-BY-4.0)",
        "runtime_validated": False,
    }

    if pk == "articulated":
        facts = adapt_articulated(
            source_usd,
            target,
            class_label=entry.get("class_label", ""),
            name=entry.get("name", ""),
            part_labels=entry.get("part_labels"),
            max_mass=max_mass,
            usd_format=usd_format,
        )
        mass_info = facts["mass_info"]
        mass_basis = (
            f"authored ({mass_info['raw_authored_kg']} kg), clamped to {mass_info['final_kg']} kg"
            if mass_info["clamped"]
            else f"authored ({mass_info['final_kg']} kg)"
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
            "reparented_links": len(facts["reparented_links"]),
            "base_link": facts["base_link"],
            "textures": facts["textures"],
            "mass": mass_info,
            "validation": None,
        }
        meta["runtime_ready"] = False
    else:
        facts = adapt_rigid(
            source_usd,
            target,
            catalog_mass_kg=entry.get("total_mass_kg"),
            max_mass=max_mass if max_mass is not None else DEFAULT_MAX_RIGID_MASS_KG,
            usd_format=usd_format,
        )
        size = facts["size"]
        geom = articulated.box_geometry(size)
        meta = {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{canonical_url()}/{la}")),
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
                "multi_body_flattened": pk == "multi_body",
                "stripped_articulation_roots": len(facts["stripped_articulation_roots"]),
                "stripped_child_bodies": len(facts["stripped_child_bodies"]),
                "textures": facts["textures"],
                "mass": facts["mass_info"],
                "validation": None,
            },
            "asset_id": oid,
            "runtime_ready": False,
        }

    articulated.write_json(target / "metadata.json", meta)
    title = (entry.get("name") or entry.get("class_label") or la).replace("_", " ").strip()
    desc_text = str(entry.get("description") or "").strip()
    if not desc_text:
        desc_text = f"A {title} ({entry.get('class_label') or pk}) from the Palatial SimReady Asset Library."
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
        "logical_asset": la,
        "physics_kind": pk,
        "size_m": meta["physics"]["size"],
        "mass_kg": meta["physics"]["mass"],
        "sha256": sha,
    }


def convert(
    source_dir: Path,
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
                    source_dir,
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
                    "logical_asset": entry.get("logical_asset", ""),
                    "physics_kind": entry.get("physics_kind", ""),
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
            why: list[str] = []
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

    p_filter = sub.add_parser("filter", help="Filter data/catalog.jsonl before downloading packages.")
    p_filter.add_argument("--catalog-jsonl", type=Path, help="Local catalog.jsonl (auto-fetched if --sources-dir set)")
    p_filter.add_argument("--sources-dir", type=Path, help="Cache dir to auto-fetch catalog.jsonl into if missing")
    p_filter.add_argument("--revision", default=DEFAULT_REVISION)
    p_filter.add_argument("--endpoint", default=None, help="Override HF_ENDPOINT")
    p_filter.add_argument("--physics-kind", choices=PHYSICS_FILTER_CHOICES, default="all")
    p_filter.add_argument("--include-failed-stock", action="store_true", help="Include runtime_status!=passed entries")
    p_filter.add_argument("--class-labels", nargs="*", default=None, help="Regex/substring patterns to include")
    p_filter.add_argument("--exclude-class-labels", nargs="*", default=None, help="Regex/substring patterns to exclude")
    p_filter.add_argument("--max-rigid-size", type=float, default=TABLETOP_MAX_M)
    p_filter.add_argument("--max-articulated-size", type=float, default=TABLESIDE_MAX_M)
    p_filter.add_argument("--min-size", type=float, default=MIN_SIZE_M)
    p_filter.add_argument("--max-per-category", type=int, default=None)
    p_filter.add_argument("--limit", type=int, default=None)
    p_filter.add_argument("--asset-ids", nargs="*", default=None, help="Specific logical_asset/asset_id values")
    p_filter.add_argument("--asset-ids-file", type=Path, default=None)
    p_filter.add_argument("--out", type=Path, required=True, help="Output selection JSON path")

    p_fetch = sub.add_parser("fetch", help="Download space-free simready_usd/ + metadata/physics.json packages.")
    p_fetch.add_argument("--selection", type=Path, required=True)
    p_fetch.add_argument("--out", type=Path, required=True, help="Destination sources/palatial directory")
    p_fetch.add_argument("--revision", default=None)
    p_fetch.add_argument("--endpoint", default=None)
    p_fetch.add_argument("--workers", type=int, default=4)
    p_fetch.add_argument("--shard-index", type=int, default=0)
    p_fetch.add_argument("--shard-total", type=int, default=1)
    p_fetch.add_argument("--report", type=Path, default=None)

    p_survey = sub.add_parser("survey", help="Pure-stdlib survey of downloaded packages, emitting manifest.json.")
    p_survey.add_argument("--source", type=Path, required=True, help="sources/palatial directory")
    p_survey.add_argument("--selection", type=Path, default=None, help="Optional filter selection JSON")
    p_survey.add_argument("--max-rigid-size", type=float, default=TABLETOP_MAX_M)
    p_survey.add_argument("--max-articulated-size", type=float, default=TABLESIDE_MAX_M)
    p_survey.add_argument("--min-size", type=float, default=MIN_SIZE_M)
    p_survey.add_argument("--out", type=Path, required=True, help="Output conversion manifest JSON")

    p_conv = sub.add_parser("convert", help="Flatten/normalize Palatial USDs into object.usd + metadata.json.")
    p_conv.add_argument("--source", type=Path, required=True, help="sources/palatial directory")
    p_conv.add_argument("--root", type=Path, required=True, help="Output root holding Assets/Object/RoboDojo/")
    p_conv.add_argument("--manifest", type=Path, required=True, help="Manifest JSON from `survey`")
    p_conv.add_argument("--skip-existing", action="store_true")
    p_conv.add_argument("--max-mass", type=float, default=None)
    p_conv.add_argument("--usd-format", choices=("usdc", "usda"), default="usdc")
    p_conv.add_argument("--summary", type=Path, default=None)

    p_ready = sub.add_parser("ready", help="Stamp check_object/check_articulation results into metadata.json.")
    p_ready.add_argument("--root", type=Path, required=True, help="Output root holding Assets/Object/RoboDojo/")
    p_ready.add_argument(
        "--check-dir", type=Path, required=True, help="--image-dir passed to check_object/check_articulation"
    )
    p_ready.add_argument("--manifest", type=Path, default=None, help="Manifest JSON listing converted ids")
    p_ready.add_argument("--ids-file", type=Path, default=None, help="Optional newline-separated object ids")
    p_ready.add_argument("ids", nargs="*", help="Explicit object ids (e.g. Rigid/palatial_basket/00000)")
    p_ready.add_argument("--summary", type=Path, default=None)
    p_ready.add_argument("--note", default=None)

    args = parser.parse_args(argv)

    if args.command == "filter":
        cat_path = args.catalog_jsonl
        if cat_path is None:
            if not args.sources_dir:
                parser.error("filter requires --catalog-jsonl or --sources-dir")
            cat_path = fetch_catalog_jsonl(args.sources_dir, revision=args.revision, endpoint=args.endpoint)
        elif not cat_path.is_file():
            if args.sources_dir:
                cat_path = fetch_catalog_jsonl(args.sources_dir, revision=args.revision, endpoint=args.endpoint)
            else:
                raise FileNotFoundError(f"catalog.jsonl not found: {cat_path}")

        asset_ids = None
        if args.asset_ids or args.asset_ids_file:
            asset_ids = set(args.asset_ids or [])
            if args.asset_ids_file:
                asset_ids |= {
                    ln.strip()
                    for ln in args.asset_ids_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")
                }

        rows = load_catalog_jsonl(cat_path)
        result = filter_catalog(
            rows,
            physics_kind=args.physics_kind,
            stock_passing_only=not args.include_failed_stock,
            class_labels=args.class_labels,
            exclude_class_labels=args.exclude_class_labels,
            max_rigid_size=args.max_rigid_size,
            max_articulated_size=args.max_articulated_size,
            min_size=args.min_size,
            max_per_category=args.max_per_category,
            limit=args.limit,
            asset_ids=asset_ids,
            revision=args.revision,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, result)
        print(f"[palatial] filter {json.dumps(result['summary'], ensure_ascii=False)}", flush=True)
        return 0

    if args.command == "fetch":
        sel = json.loads(args.selection.read_text(encoding="utf-8"))
        entries = sel.get("chosen", [])
        if args.shard_total > 1:
            entries = [e for i, e in enumerate(entries) if i % args.shard_total == args.shard_index]
        revision = args.revision or sel.get("revision", DEFAULT_REVISION)
        res = fetch_packages(entries, args.out, revision=revision, endpoint=args.endpoint, workers=args.workers)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            articulated.write_json(args.report, res)
        print(
            f"[palatial] fetch shard {args.shard_index}/{args.shard_total}: "
            f"fetched={len(res['fetched'])}, failed={len(res['failed'])}",
            flush=True,
        )
        return 1 if res["failed"] else 0

    if args.command == "survey":
        sel = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
        res = survey(
            args.source,
            selection=sel,
            max_rigid_size=args.max_rigid_size,
            max_articulated_size=args.max_articulated_size,
            min_size=args.min_size,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, res)
        print(f"[palatial] survey {json.dumps(res['summary'], ensure_ascii=False)}", flush=True)
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
        summary_path = args.summary or (args.root / "palatial_conversion.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(summary_path, res)
        print(
            f"[palatial] convert converted={len(res['converted'])}, skipped={len(res['skipped'])}, "
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
    print(f"[palatial] ready {len(passed_ids)}/{len(ids)}; reasons={json.dumps(reason_counts)}", flush=True)
    return 0 if not failed_ids else 1


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
