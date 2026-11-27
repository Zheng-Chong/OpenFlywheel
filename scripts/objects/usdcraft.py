"""Import NVIDIA USDCraft-10K objects (nvidia/USDCraft-10K on Hugging Face, Apache-2.0) into RoboDojo format.

USDCraft-10K packages live under `rigid/<group>/<category_id>_<category>/<NN>_<title>/` with:
  - `meta.json` (floating_part_count, tips_over, category/category_zh, prompt, ...)
  - `physx/model.usdc` (defaultPrim=/Asset, Z-up, meters, kg; top-level Scopes: Links, Joints, Looks, PhysicsMaterials)
  - `physx/textures/*` (optional, when has_textures is true)

Why a dedicated adapter instead of `articulated.write_layer` directly:
  1. Both `articulated.py` and `ArticulationObject.fix_root_link()` require the base `Xform` with `RigidBodyAPI` to be
     the first direct child of `/root`, whereas USDCraft-10K nests all links under `/Asset/Links/<name>`.
  2. `articulated.py` collects `Xformable` direct children of `/root` to centre and scale the object, which is empty
     when all top-level children are `Scope` prims.
  3. Multi-base objects (`floating_part_count > 0`) fail single-base articulation assumptions, and some solid-volume
     authored masses are unrealistically heavy and must be clamped proportionally (scaling diagonalInertia in lockstep).
  4. USDCraft-10K already authors `UsdPhysics.DriveAPI` and `PhysxJointAxisAPI` (`staticFrictionEffort`,
     `dynamicFrictionEffort`, `armature`); this adapter preserves them instead of overwriting with `still_parts=True`.

Pipeline subcommands:
  filter   Filter `manifest.csv` before downloading (by physics_type, group, category, textured, joint_types, limit).
  fetch    Selectively download `meta.json`, `physx/model.usdc`, and `physx/textures/*` with retry/backoff & SHA-256.
  survey   Inspect downloaded packages (`meta.json` + `pxr`) and emit a selection manifest with deterministic IDs.
  convert  Flatten, reparent `/Asset/Links/* -> /root/*`, normalize, clamp mass, and write `object.usd` + metadata.
  ready    Stamp `check_articulation.py` / `check_object.py` results into `metadata.json` (`runtime_ready`).

Needs `pxr` (`usd-core` or `isaac-sim python` with `SimulationApp` fallback) only for `survey` and `convert`.
CPU unit tests: `scripts/objects/test_usdcraft.py`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
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

SCRIPT = "scripts/objects/usdcraft.py"
VERSION = 1

REPO = "nvidia/USDCraft-10K"
DEFAULT_REVISION = "main"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "curl/8.0"
LICENSE = "Apache-2.0"
ATTRIBUTION = "USDCraft-10K, NVIDIA, huggingface.co/datasets/nvidia/USDCraft-10K"

GROUPS = (
    "kitchen_dining",
    "living_bedroom_electronics_toys",
    "office_commercial_hospitality",
    "bath_laundry_hardware",
    "industrial_warehouse_construction_energy",
    "medical_outdoor_garden_transport_sports",
)

# Default size / mass bounds (metres, kg).
TABLETOP_MAX_M = 0.65
TABLESIDE_MAX_M = 0.90
MIN_THICKNESS_M = 0.003
DEFAULT_MIN_ARTICULATED_MASS_KG = 0.05
DEFAULT_MAX_TABLETOP_MASS_KG = 15.0
DEFAULT_MAX_TABLESIDE_MASS_KG = 25.0
DEFAULT_MAX_PORTABLE_MASS_KG = 3.0
DEFAULT_MIN_RIGID_MASS_KG = 0.005
DEFAULT_MAX_RIGID_MASS_KG = 0.5
MIN_LINK_MASS_KG = articulated.MIN_LINK_MASS
RIGID_FRICTION = 0.6

# Category tokens that typically indicate portable handheld / tabletop items whose base should stay free.
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
    "umbrella",
    "wallet",
    "watch",
    "wrench",
)

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
FETCH_MARKER = ".fetched"


def hf_endpoint(endpoint: str | None = None) -> str:
    raw = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    return raw.rstrip("/")


def dataset_url(endpoint: str | None = None) -> str:
    return f"{hf_endpoint(endpoint)}/datasets/{REPO}"


def canonical_url() -> str:
    return f"{DEFAULT_HF_ENDPOINT}/datasets/{REPO}"


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
    """List `(path, size)` entries under `subpath` from the Hugging Face tree API with pagination and retries."""
    base = hf_endpoint(endpoint)
    quoted = f"/{urllib.parse.quote(subpath.strip('/'))}" if subpath.strip("/") else ""
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


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _parse_joint_types(raw: object) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list | tuple):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    parts = re.split(r"[,;|\s]+", text)
    return [p.strip().lower() for p in parts if p.strip()]


def load_manifest_csv(csv_path: Path) -> list[dict]:
    """Load and normalize rows from USDCraft-10K `manifest.csv`."""
    rows: list[dict] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            path = (row.get("path") or "").strip().strip("/")
            if not path:
                continue
            rows.append(
                {
                    "object_id": (row.get("object_id") or "").strip(),
                    "path": path,
                    "group": (row.get("group") or "").strip(),
                    "category_id": (row.get("category_id") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "physics_type": (row.get("physics_type") or "").strip().lower(),
                    "part_count": _parse_int(row.get("part_count")),
                    "joint_count": _parse_int(row.get("joint_count")),
                    "joint_types": _parse_joint_types(row.get("joint_types")),
                    "has_textures": _parse_bool(row.get("has_textures")),
                    "physx_sha256": (row.get("physx_sha256") or "").strip().lower(),
                    "prompt": (row.get("prompt") or "").strip(),
                }
            )
    return rows


def filter_manifest(
    rows: list[dict],
    *,
    physics_type: str = "articulated",
    groups: list[str] | None = None,
    categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
    textured_only: bool = False,
    joint_types: list[str] | None = None,
    max_joints: int | None = None,
    max_per_category: int | None = None,
    limit: int | None = None,
    object_ids: set[str] | None = None,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Filter `manifest.csv` rows before downloading any `.usdc` files."""
    group_set = {g.strip() for g in (groups or []) if g.strip()}
    cat_pats = [re.compile(p, re.IGNORECASE) for p in (categories or []) if p.strip()]
    excl_pats = [re.compile(p, re.IGNORECASE) for p in (exclude_categories or []) if p.strip()]
    req_joints = {j.strip().lower() for j in (joint_types or []) if j.strip()}
    if "all" in req_joints:
        req_joints = set()
    elif not req_joints and physics_type == "articulated":
        req_joints = {"revolute", "prismatic"}
    allow_empty_joints = bool(req_joints & {"fixed", "none"})
    req_named_joints = req_joints - {"none"}

    # Sort deterministically by (category, object_id, path) so limits and per-category caps are reproducible.
    ordered = sorted(rows, key=lambda r: (r["category"], r["object_id"], r["path"]))
    chosen: list[dict] = []
    dropped: list[dict] = []
    cat_counts: dict[str, int] = {}
    id_counters: dict[str, int] = {}

    for row in ordered:
        reason = None
        if object_ids is not None and row["object_id"] not in object_ids:
            reason = "not in requested object_ids"
        elif physics_type != "all" and row["physics_type"] != physics_type:
            reason = f"physics_type={row['physics_type']} != {physics_type}"
        elif group_set and row["group"] not in group_set:
            reason = f"group={row['group']} excluded"
        elif cat_pats and not any(p.search(row["category"]) or p.search(row["category_id"]) for p in cat_pats):
            reason = f"category={row['category']} does not match filter"
        elif excl_pats and any(p.search(row["category"]) or p.search(row["category_id"]) for p in excl_pats):
            reason = f"category={row['category']} excluded by pattern"
        elif textured_only and not row["has_textures"]:
            reason = "untextured"
        elif (
            row["physics_type"] == "articulated"
            and req_joints
            and not (
                (not row["joint_types"] and allow_empty_joints) or bool(set(row["joint_types"]) & req_named_joints)
            )
        ):
            reason = f"joint_types={row['joint_types']} lacks any of {sorted(req_joints)}"
        elif max_joints is not None and row["joint_count"] > max_joints:
            reason = f"joint_count={row['joint_count']} > {max_joints}"
        elif max_per_category is not None and cat_counts.get(row["category"], 0) >= max_per_category:
            reason = f"category cap ({max_per_category}) reached"
        elif limit is not None and len(chosen) >= limit:
            reason = f"limit ({limit}) reached"

        if reason is not None:
            dropped.append({"object_id": row["object_id"], "path": row["path"], "reason": reason})
            continue
        cat_counts[row["category"]] = cat_counts.get(row["category"], 0) + 1
        cat_slug = "usdcraft_" + (articulated.slug(row["category"]) or "object")
        kind_dir = "Articulation" if row["physics_type"] == "articulated" else "Rigid"
        id_key = f"{kind_dir}/{cat_slug}"
        idx = id_counters.get(id_key, 0)
        id_counters[id_key] = idx + 1
        chosen.append({**row, "id": f"{id_key}/{idx:05d}"})

    return {
        "source": canonical_url(),
        "dataset": REPO,
        "revision": revision,
        "license": LICENSE,
        "filters": {
            "physics_type": physics_type,
            "groups": sorted(group_set),
            "categories": categories or [],
            "exclude_categories": exclude_categories or [],
            "textured_only": textured_only,
            "joint_types": sorted(req_joints),
            "max_joints": max_joints,
            "max_per_category": max_per_category,
            "limit": limit,
        },
        "summary": {
            "total": len(rows),
            "chosen": len(chosen),
            "dropped": len(dropped),
            "categories": len(cat_counts),
        },
        "chosen": chosen,
        "dropped": dropped,
    }


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_manifest_csv(
    out_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    force: bool = False,
) -> Path:
    """Download `manifest.csv` into `out_dir/manifest.csv` if missing (or `force=True`)."""
    dest = Path(out_dir) / "manifest.csv"
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    url = f"{dataset_url(endpoint)}/resolve/{revision}/manifest.csv"
    data, _ = http_get(url)
    _atomic_write_bytes(dest, data)
    return dest


def _read_fetch_marker(pkg_dir: Path) -> dict | None:
    marker_path = pkg_dir / FETCH_MARKER
    if not marker_path.is_file():
        return None
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def validate_package_complete(
    pkg_dir: Path,
    *,
    entry: dict | None = None,
    require_fetched_marker: bool = False,
) -> tuple[bool, str]:
    """Verify a downloaded package has `physx/model.usdc`, `meta.json`, valid `.fetched` files, and matching SHA-256."""
    pkg_dir = Path(pkg_dir)
    usdc = pkg_dir / "physx" / "model.usdc"
    if not usdc.is_file() or usdc.stat().st_size == 0:
        return False, "missing physx/model.usdc (fetch failed or not downloaded)"
    meta_file = pkg_dir / "meta.json"
    if not meta_file.is_file() or meta_file.stat().st_size == 0:
        return False, "incomplete package: missing meta.json"

    marker_path = pkg_dir / FETCH_MARKER
    if marker_path.is_file():
        marker = _read_fetch_marker(pkg_dir)
        if not marker:
            return False, f"incomplete package: invalid {FETCH_MARKER} marker"
        if entry and entry.get("has_textures") and not marker.get("has_textures"):
            return False, f"incomplete package: {FETCH_MARKER} missing textures (has_textures=false)"
        files = marker.get("files")
        if not isinstance(files, list) or "meta.json" not in files or "physx/model.usdc" not in files:
            return False, f"incomplete package: malformed {FETCH_MARKER} file list"
        for rel_f in files:
            if not isinstance(rel_f, str) or not rel_f:
                return False, f"incomplete package: invalid entry in {FETCH_MARKER}"
            fp = pkg_dir / rel_f
            if not fp.is_file() or fp.stat().st_size == 0:
                return False, f"incomplete package: missing {rel_f} listed in {FETCH_MARKER}"
    elif require_fetched_marker:
        return False, f"incomplete package: missing {FETCH_MARKER} marker (fetch incomplete)"

    expected_sha = (entry.get("physx_sha256") or "").strip().lower() if entry else ""
    if expected_sha and _sha256_file(usdc) != expected_sha:
        return False, "incomplete package: physx/model.usdc sha256 mismatch"
    return True, ""


def fetch_one_package(
    entry: dict,
    out_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    attempts: int = 6,
) -> dict:
    """Download one USDCraft-10K package (`meta.json`, `physx/model.usdc`, optional `physx/textures/*`).

    Writes `.fetched` atomically ONLY after every required file (including all `physx/textures/*`) is on disk.
    """
    out_dir = Path(out_dir)
    pkg_rel = entry["path"].strip("/")
    pkg_dir = out_dir / pkg_rel
    marker_path = pkg_dir / FETCH_MARKER
    expected_sha = (entry.get("physx_sha256") or "").strip().lower()

    # Fast path: skip HTTP/tree calls when a valid `.fetched` marker matches revision, sha256, and all files exist.
    marker = _read_fetch_marker(pkg_dir)
    if marker is not None:
        marker_ok = (
            marker.get("revision") == revision
            and bool(marker.get("has_textures")) == bool(entry.get("has_textures"))
            and (not expected_sha or marker.get("physx_sha256") == expected_sha)
        )
        if marker_ok:
            complete, _ = validate_package_complete(pkg_dir, entry=entry, require_fetched_marker=True)
            if complete:
                return {
                    "object_id": entry.get("object_id", ""),
                    "path": pkg_rel,
                    "downloaded_files": 0,
                    "cached": True,
                }
    marker_path.unlink(missing_ok=True)

    base_url = f"{dataset_url(endpoint)}/resolve/{revision}"
    downloaded: list[str] = []
    expected_files: list[str] = []

    def _download(
        rel_path: str,
        *,
        expected_sha256: str = "",
        expected_size: int = 0,
        optional: bool = False,
    ) -> bool:
        dest = out_dir / rel_path
        if dest.is_file() and dest.stat().st_size > 0:
            size_ok = not expected_size or dest.stat().st_size == expected_size
            sha_ok = not expected_sha256 or _sha256_file(dest) == expected_sha256
            if size_ok and sha_ok:
                return True
        url = f"{base_url}/{urllib.parse.quote(rel_path)}"
        try:
            data, _ = http_get(url, attempts=attempts)
        except urllib.error.HTTPError as exc:
            if optional and exc.code == 404:
                return False
            raise
        if not data:
            raise OSError(f"{rel_path}: empty response")
        if expected_size and len(data) != expected_size:
            raise OSError(f"{rel_path}: size {len(data)} != expected {expected_size}")
        if expected_sha256:
            got = hashlib.sha256(data).hexdigest()
            if got != expected_sha256:
                raise OSError(f"{rel_path}: sha256 {got} != expected {expected_sha256}")
        _atomic_write_bytes(dest, data)
        downloaded.append(rel_path)
        return True

    _download(f"{pkg_rel}/meta.json")
    expected_files.append("meta.json")
    _download(f"{pkg_rel}/physx/model.usdc", expected_sha256=expected_sha)
    expected_files.append("physx/model.usdc")
    if _download(f"{pkg_rel}/physx/model.json", optional=True):
        expected_files.append("physx/model.json")

    if entry.get("has_textures"):
        tex_subpath = f"{pkg_rel}/physx/textures"
        try:
            tex_files = list_hf_tree(tex_subpath, revision=revision, endpoint=endpoint, attempts=attempts)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                tex_files = []
            else:
                raise
        for rel_path, size in tex_files:
            _download(rel_path, expected_size=size)
            expected_files.append(Path(rel_path).relative_to(pkg_rel).as_posix())

    marker_payload = {
        "revision": revision,
        "object_id": entry.get("object_id", ""),
        "has_textures": bool(entry.get("has_textures")),
        "physx_sha256": expected_sha,
        "files": sorted(set(expected_files)),
    }
    _atomic_write_bytes(marker_path, (json.dumps(marker_payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    return {
        "object_id": entry.get("object_id", ""),
        "path": pkg_rel,
        "downloaded_files": len(downloaded),
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
    """Download selected USDCraft-10K packages into `out_dir` and write `UPSTREAM.json`."""
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
                        "object_id": entry.get("object_id", ""),
                        "path": entry.get("path", ""),
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
                            "object_id": entry.get("object_id", ""),
                            "path": entry.get("path", ""),
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


def is_fixed_base(category: str, size: list[float] | tuple[float, ...], group: str = "") -> bool:
    """Heuristic for metadata `physics.fixed_base`: appliances/furniture stay fixed; small portable items stay free."""
    tokens = set(re.findall(r"[a-z]+", category.lower()))
    if tokens & set(PORTABLE_TOKENS) and max(size) <= 0.45:
        return False
    if max(size) <= 0.25 and group in ("kitchen_dining", "office_commercial_hospitality", "bath_laundry_hardware"):
        return False
    return True


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
    # Iteratively pin entries below `floor` and re-scale the remaining headroom.
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
    """Compute final per-link masses, clamping overly heavy solid-volume masses and filling missing links.

    Guarantees `all(m >= eff_min_link)` and `min_total_mass <= sum(masses.values()) <= max_total_mass`.
    Returns `(masses_by_path, info_dict)` where `info_dict` records `raw_authored_kg`, `final_kg`, `clamped`, `filled`.
    """
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
            if sum(masses.values()) > max_total_mass + 1e-9:
                masses = _scale_with_floor(valid_authored, max_total_mass, eff_min_link)
                clamped = True
    else:
        # Floor authored links first, then clamp authored subset if it leaves less than `eff_min_link` per free link.
        min_free_budget = eff_min_link * len(free)
        max_authored_budget = max(eff_min_link * len(valid_authored), max_total_mass - min_free_budget)
        floored_authored = {p: max(eff_min_link, m) for p, m in valid_authored.items()}
        if valid_authored and sum(floored_authored.values()) > max_authored_budget + 1e-9:
            masses.update(_scale_with_floor(valid_authored, max_authored_budget, eff_min_link))
            clamped = True
        else:
            masses.update(floored_authored)

        authored_sum = sum(masses.values())
        target_total = min(max(authored_sum + min_free_budget, min_total_mass), max_total_mass)
        left = max(min_free_budget, target_total - authored_sum)
        weights = {p: max(float(volumes.get(p, 0.0)), 0.005**3) for p in free}
        masses.update(_scale_with_floor(weights, left, eff_min_link))

    final_total = sum(masses.values())
    return masses, {
        "raw_authored_kg": round(raw_total, 4),
        "final_kg": round(final_total, 4),
        "clamped": clamped,
        "filled_missing_links": filled,
        "bounds_kg": [round(min_total_mass, 4), round(max_total_mass, 4)],
    }


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
    meta: dict,
    size: list[float],
    facts: dict,
    *,
    max_size: float = TABLESIDE_MAX_M,
    allow_floating: bool = False,
) -> tuple[bool, str]:
    """Return `(chosen, reason)` for a USDCraft-10K articulated candidate."""
    floating = _parse_int(meta.get("floating_part_count"), 0)
    if floating > 0 and not allow_floating:
        return False, f"floating_part_count={floating} (multi-base)"
    if facts["up_axis"] != "Z":
        return False, f"up axis {facts['up_axis']}"
    if facts["world_movable"]:
        return False, f"{facts['world_movable'][0]['name']} hinges a link to the world, not to the body"
    if not facts["movable"]:
        return False, "no revolute or prismatic joint between two links"
    names = [j["name"] for j in facts["movable"]]
    if len(set(names)) < len(names):
        return False, "two joints share a prim name, so their DOFs cannot be told apart"
    if len(facts["bases"]) != 1:
        return False, f"base link ambiguous: {facts['bases']}"
    if not facts["colliders"]:
        return False, "no collision geometry"
    big = max(size)
    if min(size) < MIN_THICKNESS_M:
        return False, f"degenerate size ({min(size):.4f} m)"
    if big > max_size or size[2] > max_size:
        return False, f"{big:.2f} m exceeds max_size {max_size:.2f} m"
    parts = sorted({articulated.part_of(j["body1"].rsplit("/", 1)[-1], j["type"]) for j in facts["movable"]})
    where = "tabletop" if big <= TABLETOP_MAX_M else "table side"
    return True, f"{where}, {len(facts['movable'])} joint(s): {', '.join(parts)}"


def decide_rigid(
    meta: dict,
    size: list[float],
    facts: dict,
    *,
    max_size: float = TABLETOP_MAX_M,
    drop_tips_over: bool = True,
    allow_floating: bool = False,
) -> tuple[bool, str]:
    """Return `(chosen, reason)` for a USDCraft-10K static_rigid candidate."""
    floating = _parse_int(meta.get("floating_part_count"), 0)
    if floating > 0 and not allow_floating:
        return False, f"floating_part_count={floating} (multi-part rigid)"
    if drop_tips_over and _parse_bool(meta.get("tips_over")):
        return False, "tips_over=true in meta.json"
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


def discover_packages(source_dir: Path) -> list[Path]:
    """Find all downloaded USDCraft-10K package dirs containing `physx/model.usdc`."""
    source_dir = Path(source_dir)
    return sorted(p.parent.parent for p in source_dir.rglob("physx/model.usdc"))


def _read_pkg_meta(pkg_dir: Path) -> dict:
    meta_path = pkg_dir / "meta.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def survey_package(pkg_dir: Path) -> tuple[list[float], dict, dict]:
    """Inspect a downloaded USDCraft-10K package without modifying any files."""
    import numpy as np

    _, _, Usd, UsdGeom, _, _, _ = pxr_modules()
    meta = _read_pkg_meta(pkg_dir)
    model = pkg_dir / "physx" / "model.usdc"
    stage = Usd.Stage.Open(str(model))
    facts = articulated.inspect(stage)
    for path in facts["lights"]:
        stage.GetPrimAtPath(path).SetActive(False)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if rng.IsEmpty():
        raise ValueError("empty bounding box")
    size = (np.asarray(rng.GetMax(), float) - np.asarray(rng.GetMin(), float)) * facts["meters_per_unit"]
    return [round(float(v), 4) for v in size], facts, meta


def survey(
    source_dir: Path,
    *,
    selection: dict | None = None,
    max_size: float = TABLESIDE_MAX_M,
    max_rigid_size: float = TABLETOP_MAX_M,
    allow_floating: bool = False,
    drop_tips_over: bool = True,
    fixed_as_rigid: bool = False,
    existing_sources: dict[str, str] | None = None,
    existing_max_index: dict[str, int] | None = None,
    revision: str = DEFAULT_REVISION,
) -> dict:
    """Survey downloaded packages and produce a conversion selection manifest with deterministic RoboDojo IDs."""
    source_dir = Path(source_dir)
    by_rel: dict[str, dict] = {}
    chosen: list[dict] = []
    dropped: list[dict] = []
    if selection:
        for entry in selection.get("chosen", []):
            by_rel[entry["path"].strip("/")] = entry
        pkg_dirs = []
        for rel in sorted(by_rel):
            pkg = source_dir / rel
            csv_row = by_rel[rel]
            ok_pkg, pkg_reason = validate_package_complete(pkg, entry=csv_row, require_fetched_marker=True)
            if ok_pkg:
                pkg_dirs.append(pkg)
            else:
                dropped.append(
                    {
                        "src": rel,
                        "object_id": csv_row.get("object_id", ""),
                        "reason": pkg_reason,
                    }
                )
    else:
        pkg_dirs = []
        for pkg in discover_packages(source_dir):
            ok_pkg, pkg_reason = validate_package_complete(pkg, require_fetched_marker=False)
            if ok_pkg:
                pkg_dirs.append(pkg)
            else:
                dropped.append(
                    {
                        "src": pkg.relative_to(source_dir).as_posix(),
                        "object_id": pkg.name,
                        "reason": pkg_reason,
                    }
                )

    counters: dict[str, int] = {}
    existing_src_map = dict(existing_sources or {})
    for oid in existing_src_map.values():
        parts = oid.strip("/").split("/")
        if len(parts) == 3 and parts[2].isdigit():
            k = f"{parts[0]}/{parts[1]}"
            counters[k] = max(counters.get(k, 0), int(parts[2]) + 1)
    if existing_max_index:
        for k, max_idx in existing_max_index.items():
            counters[k] = max(counters.get(k, 0), int(max_idx) + 1)
    has_existing = bool(existing_src_map or existing_max_index)

    for pkg_dir in pkg_dirs:
        rel = pkg_dir.relative_to(source_dir).as_posix()
        csv_row = by_rel.get(rel, {})
        obj_key = csv_row.get("object_id") or ""
        if (obj_key and obj_key in existing_src_map) or rel in existing_src_map:
            existing_oid = existing_src_map.get(obj_key) or existing_src_map.get(rel)
            dropped.append(
                {
                    "src": rel,
                    "object_id": obj_key,
                    "reason": f"already imported as {existing_oid}",
                }
            )
            continue
        reclassified_rigid = False
        try:
            size, facts, meta = survey_package(pkg_dir)
            meta_obj_id = str(meta.get("object_id") or "")
            if meta_obj_id and meta_obj_id in existing_src_map:
                dropped.append(
                    {
                        "src": rel,
                        "object_id": meta_obj_id,
                        "reason": f"already imported as {existing_src_map[meta_obj_id]}",
                    }
                )
                continue
            physics_type = (
                csv_row.get("physics_type")
                or str(meta.get("physics_type") or ("articulated" if facts["movable"] else "static_rigid")).lower()
            )
            if physics_type == "articulated" and not facts["movable"] and fixed_as_rigid:
                physics_type = "static_rigid"
                reclassified_rigid = True
            if physics_type == "articulated":
                ok, reason = decide_articulated(meta, size, facts, max_size=max_size, allow_floating=allow_floating)
            else:
                ok, reason = decide_rigid(
                    meta,
                    size,
                    facts,
                    max_size=max_rigid_size,
                    drop_tips_over=drop_tips_over,
                    allow_floating=allow_floating,
                )
        except Exception as exc:
            dropped.append(
                {
                    "src": rel,
                    "object_id": csv_row.get("object_id", ""),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        category = csv_row.get("category") or str(meta.get("category") or pkg_dir.parent.name)
        group = csv_row.get("group") or str(meta.get("group") or (rel.split("/")[1] if "/" in rel else ""))
        entry = {
            "src": rel,
            "object_id": csv_row.get("object_id") or str(meta.get("object_id") or pkg_dir.name),
            "group": group,
            "category_id": csv_row.get("category_id") or str(meta.get("category_id") or ""),
            "category": category,
            "category_zh": str(meta.get("category_zh") or ""),
            "title": csv_row.get("title") or str(meta.get("title") or pkg_dir.name),
            "physics_type": physics_type,
            "has_textures": bool(csv_row.get("has_textures", (pkg_dir / "physx" / "textures").is_dir())),
            "floating_part_count": _parse_int(meta.get("floating_part_count"), 0),
            "tips_over": _parse_bool(meta.get("tips_over")),
            "size_m": size,
            "reason": reason,
        }
        if not ok:
            dropped.append(entry)
            continue

        cat_slug = "usdcraft_" + (articulated.slug(category) or "object")
        kind_dir = "Articulation" if physics_type == "articulated" else "Rigid"
        key = f"{kind_dir}/{cat_slug}"
        if csv_row.get("id") and not has_existing and not reclassified_rigid:
            oid = csv_row["id"]
        else:
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            oid = f"{key}/{idx:05d}"
        chosen.append({**entry, "id": oid})

    return {
        "source": canonical_url(),
        "dataset": REPO,
        "revision": selection.get("revision", revision) if selection else revision,
        "license": LICENSE,
        "chosen": chosen,
        "dropped": dropped,
    }


def _reparent_links_to_root(flat, src_root: str) -> list[tuple[str, str]]:
    """Promote nested RigidBody links (e.g. `/Asset/Links/<name>`) to direct children of `/root`.

    Uses `Sdf.BatchNamespaceEdit` so USD automatically rewrites all internal joint `body0`/`body1`,
    material binding, and shader connection paths.
    """
    _, Sdf, Usd, UsdGeom, UsdPhysics, _, _ = pxr_modules()

    tmp_stage = Usd.Stage.Open(flat)
    root_path = Sdf.Path(src_root)
    root_prim = tmp_stage.GetPrimAtPath(root_path)
    if not root_prim:
        raise ValueError(f"missing root prim {src_root}")

    # Collect rigid-body prims that are nested deeper than direct children of `src_root`.
    nested_links = [
        p
        for p in Usd.PrimRange(root_prim)
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetPath().GetParentPath() != root_path and p.GetPath() != root_path
    ]

    # Compute ancestor transforms between `src_root` and each nested link BEFORE mutating any xform ops,
    # respecting `resetXformStack` so link world poses stay identical after reparenting.
    anc_transforms: list[tuple[object, object]] = []
    for link in nested_links:
        if not UsdGeom.Xformable(link) or UsdGeom.Xformable(link).GetResetXformStack():
            continue
        anc_xf = None
        cur = link.GetParent()
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
            anc_transforms.append((link, anc_xf))

    for link, anc_xf in anc_transforms:
        xf = UsdGeom.Xformable(link)
        ops, reset = xf.GetOrderedXformOps(), xf.GetResetXformStack()
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ofw_reparent")
        op.Set(anc_xf)
        xf.SetXformOpOrder([op, *ops], reset)

    moved: list[tuple[str, str]] = []
    if nested_links:
        parent_containers = {p.GetPath().GetParentPath() for p in nested_links}
        # Include ALL current children of `root_prim` (including containers like `Links`) because the
        # container prim still exists while `BatchNamespaceEdit` executes.
        used_names = {c.GetName() for c in root_prim.GetChildren()}
        # Move deepest links first so nested-under-nested rigid bodies stay at valid paths during batch edit.
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

        # Remove now-empty container prims (typically `/Asset/Links`).
        cleanup = Sdf.BatchNamespaceEdit()
        removed_any = False
        for container in sorted(parent_containers, key=lambda p: len(str(p)), reverse=True):
            spec = flat.GetPrimAtPath(container)
            if spec and not spec.nameChildren:
                cleanup.Add(container, Sdf.Path.emptyPath)
                removed_any = True
        if removed_any and flat.CanApply(cleanup):
            flat.Apply(cleanup)

    # Rename `/Asset` (or whatever `src_root` is) -> `/root`.
    if root_path != Sdf.Path("/root"):
        rename = Sdf.BatchNamespaceEdit()
        rename.Add(root_path, Sdf.Path("/root"))
        if not flat.CanApply(rename):
            raise ValueError(f"cannot rename {root_path} to /root")
        flat.Apply(rename)

    # Explicitly rewrite RelationshipSpec target paths (e.g. physics:body0/body1, material:binding)
    # AND AttributeSpec connection paths (e.g. UsdShade material outputs:surface & shader inputs:* connections)
    # because Sdf.Layer.Apply(BatchNamespaceEdit) in USD 24.x does not always retarget them.
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

    def _rewrite_paths(prim_spec):
        for rel in prim_spec.relationships:
            targets = list(rel.targetPathList.GetAddedOrExplicitItems())
            if not targets:
                continue
            updated = [_remap_path(t) for t in targets]
            if updated != targets:
                rel.targetPathList.ClearEdits()
                rel.targetPathList.explicitItems = updated
        for attr in prim_spec.attributes:
            conns = list(attr.connectionPathList.GetAddedOrExplicitItems())
            if not conns:
                continue
            updated = [_remap_path(c) for c in conns]
            if updated != conns:
                attr.connectionPathList.ClearEdits()
                attr.connectionPathList.explicitItems = updated
        for child in prim_spec.nameChildren:
            _rewrite_paths(child)

    for top in flat.rootPrims:
        _rewrite_paths(top)

    flat.defaultPrim = "root"
    return moved


def _copy_and_rewrite_textures(flat, source_usdc: Path, target_dir: Path) -> list[str]:
    """Copy referenced external textures into `target_dir/textures/` and rewrite `Sdf.AssetPath` to `./textures/...`."""
    _, Sdf, _, _, _, _, _ = pxr_modules()

    src_dir = source_usdc.parent.resolve()
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
        try:
            rel = cand.relative_to(src_dir)
        except ValueError:
            rel = Path("textures") / cand.name
        if rel.parts[0] != "textures":
            rel = Path("textures") / rel.name
        # Disambiguate colliding basenames from different source directories.
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


def _still_missing_joints(stage, joints: list[dict], facts: dict, masses: dict[str, float], spans: dict) -> None:
    """Apply fallback Coulomb friction ONLY to openable joints that have no authored PhysxJointAxisAPI friction."""
    _, Sdf, _, _, UsdPhysics, _, _ = pxr_modules()
    by_name = {j["name"]: j for j in facts["movable"]}
    for entry in joints:
        if entry["role"] not in ("door", "drawer", "lid"):
            continue
        existing_fric = entry.get("friction") or {}
        if "static" in existing_fric or "dynamic" in existing_fric:
            continue  # preserve authored PhysxJointAxisAPI friction
        joint = by_name[entry["name"]]
        prim = stage.GetPrimAtPath(joint["path"])
        token = "angular" if entry["type"] == "revolute" else "linear"
        if entry["type"] == "prismatic":
            friction = articulated.STILL_FRICTION["prismatic"]
        elif articulated._axis_vertical(stage, prim):
            friction = articulated.STILL_FRICTION["upright"]
        else:
            span = spans.get(joint["body1"]) or [0.1, 0.1, 0.1]
            friction = round(
                articulated.STILL_FRICTION["level"] * masses.get(joint["body1"], 0.1) * 9.81 * max(span) / 2, 3
            )
        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{token}")
        for attr_name in ("staticFrictionEffort", "dynamicFrictionEffort"):
            prim.CreateAttribute(f"physxJointAxis:{token}:{attr_name}", Sdf.ValueTypeNames.Float).Set(float(friction))
        entry["friction"] = {
            **existing_fric,
            "static": friction,
            "dynamic": friction,
            "source": "estimated (missing_friction_fallback)",
        }


def _promote_staged_dir(stage_dir: Path, target_dir: Path) -> None:
    """Atomically promote staged `object.usd` (+ optional `textures/`) into `target_dir`."""
    if not target_dir.exists():
        os.rename(stage_dir, target_dir)
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    staged_tex = stage_dir / "textures"
    dst_tex = target_dir / "textures"
    if staged_tex.is_dir():
        old_tex = target_dir / f".textures.old.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        if dst_tex.exists():
            os.rename(dst_tex, old_tex)
        os.rename(staged_tex, dst_tex)
        if old_tex.exists():
            shutil.rmtree(old_tex, ignore_errors=True)
    elif dst_tex.is_dir():
        shutil.rmtree(dst_tex, ignore_errors=True)
    os.replace(stage_dir / "object.usd", target_dir / "object.usd")
    shutil.rmtree(stage_dir, ignore_errors=True)


def adapt_articulated(
    source_usdc: Path,
    target_dir: Path,
    *,
    category: str,
    group: str = "",
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float | None = None,
    min_mass: float = DEFAULT_MIN_ARTICULATED_MASS_KG,
    fill_missing_friction: bool = False,
    usd_format: str = "usdc",
) -> dict:
    """Flatten, reparent `/Asset/Links/* -> /root/*`, normalize, clamp mass, and write `target_dir/object.usd`."""
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, _, UsdUtils = pxr_modules()

    source_usdc = Path(source_usdc).resolve()
    target_dir = Path(target_dir)

    src_stage = Usd.Stage.Open(str(source_usdc))
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

    # Collect any PhysxArticulationAPI attributes from ancestors above defaultPrim before flattening.
    art_attrs: dict[str, tuple] = {}
    for anc in reversed(list(Sdf.Path(src_facts["root"]).GetAncestorsRange())[1:]):
        above = src_stage.GetPrimAtPath(anc)
        for attr in above.GetAttributes() if above else ():
            if attr.GetName().startswith(articulated.PHYSX_ARTICULATION + ":") and attr.HasAuthoredValue():
                art_attrs[attr.GetName()] = (attr.GetTypeName(), attr.Get())

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = target_dir.parent / f".{target_dir.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        flat = src_stage.Flatten()
        moved_links = _reparent_links_to_root(flat, src_facts["root"])
        copied_textures = _copy_and_rewrite_textures(flat, source_usdc, stage_dir)

        stage = Usd.Stage.Open(flat)
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

        # Move ArticulationRootAPI onto /root; apply PhysxArticulationAPI to BOTH /root and base_link
        # so `ArticulationObject.fix_root_link()` can copy attributes from base_link to /root without error.
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

        # Put base link FIRST among /root's children (`fix_root_link` picks the first Xform child with RigidBodyAPI).
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

        fixed_base = is_fixed_base(category, [float(v) for v in size], group)
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

        if fill_missing_friction:
            _still_missing_joints(stage, joints, facts, masses, spans)

        staged_usd = stage_dir / "object.usd"
        flat.Export(str(staged_usd), args={"format": usd_format})

        # Post-write sanity check on the staged USD BEFORE promoting to `target_dir/object.usd`.
        reopened = Usd.Stage.Open(str(staged_usd))
        re_root = reopened.GetDefaultPrim()
        if not re_root or re_root.GetPath() != Sdf.Path("/root") or not re_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("exported object.usd missing /root with ArticulationRootAPI")
        first_child = re_root.GetChildren()[0] if re_root.GetChildren() else None
        if not first_child or first_child.GetTypeName() != "Xform" or not first_child.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"first child of /root is not an Xform with RigidBodyAPI: {first_child}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(staged_usd)))
        if unresolved:
            raise ValueError(f"unresolved asset dependencies: {list(unresolved)}")
        del reopened

        _promote_staged_dir(stage_dir, target_dir)
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

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
    source_usdc: Path,
    target_dir: Path,
    *,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    max_mass: float = DEFAULT_MAX_RIGID_MASS_KG,
    min_mass: float = DEFAULT_MIN_RIGID_MASS_KG,
    usd_format: str = "usdc",
) -> dict:
    """Adapt a `static_rigid` USDCraft-10K `.usdc` into a RoboDojo `RigidObject` (`object.usd`)."""
    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils = pxr_modules()

    source_usdc = Path(source_usdc).resolve()
    target_dir = Path(target_dir)

    src_stage = Usd.Stage.Open(str(source_usdc))
    src_facts = articulated.inspect(src_stage)
    if src_facts["up_axis"] != "Z":
        raise ValueError(f"up axis {src_facts['up_axis']} (only Z-up sources are adapted)")
    if not src_facts["colliders"]:
        raise ValueError("no enabled collision geometry")

    raw_mass = sum(src_facts["masses"].values())
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = target_dir.parent / f".{target_dir.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        flat = src_stage.Flatten()
        _reparent_links_to_root(flat, src_facts["root"])
        copied_textures = _copy_and_rewrite_textures(flat, source_usdc, stage_dir)

        stage = Usd.Stage.Open(flat)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        root = stage.GetPrimAtPath("/root")
        if root.GetTypeName() != "Xform":
            root.SetTypeName("Xform")
        stage.SetDefaultPrim(root)

        # Strip ArticulationRootAPI / PhysxArticulationAPI everywhere; strip child RigidBodyAPI / MassAPI / physics
        # material bindings; deactivate lights/joints; block BehaviorScripts.
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
        lo, hi = articulated._bound(cache, root)
        size = hi - lo
        if np.linalg.norm((lo + hi) / 2) > 1e-4:
            raise ValueError(f"origin not at bounding-box centre: {(lo + hi) / 2}")

        mass = raw_mass if raw_mass > 0 else float(np.prod(np.maximum(size, 0.01)) * 500.0)
        clamped_mass = round(min(max(mass, min_mass), max_mass), 4)
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(clamped_mass))

        staged_usd = stage_dir / "object.usd"
        flat.Export(str(staged_usd), args={"format": usd_format})

        reopened = Usd.Stage.Open(str(staged_usd))
        bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if bodies != ["/root"]:
            raise ValueError(f"rigid bodies other than /root remain: {bodies}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(staged_usd)))
        if unresolved:
            raise ValueError(f"unresolved asset dependencies: {list(unresolved)}")
        del reopened

        _promote_staged_dir(stage_dir, target_dir)
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

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
        "textures": copied_textures,
    }


def convert_one(
    source_dir: Path,
    out_root: Path,
    entry: dict,
    *,
    revision: str = DEFAULT_REVISION,
    max_mass: float | None = None,
    fill_missing_friction: bool = False,
    usd_format: str = "usdc",
) -> dict:
    """Convert one selected USDCraft-10K entry into `out_root/Assets/Object/RoboDojo/<id>/`."""
    source_dir = Path(source_dir)
    out_root = Path(out_root)
    pkg_dir = source_dir / entry["src"]
    model = pkg_dir / "physx" / "model.usdc"
    pkg_meta = _read_pkg_meta(pkg_dir)
    oid = entry["id"]
    target = out_root / "Assets/Object/RoboDojo" / oid

    source_provenance = {
        "dataset": REPO,
        "url": canonical_url(),
        "revision": revision,
        "object": entry["src"],
        "object_id": entry.get("object_id", ""),
        "group": entry.get("group", ""),
        "category_id": entry.get("category_id", ""),
        "category": entry.get("category", ""),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "rsi_pools": "allowed (Apache-2.0)",
        "runtime_validated": False,
    }

    if entry.get("physics_type", "articulated") == "articulated":
        facts = adapt_articulated(
            model,
            target,
            category=entry.get("category", ""),
            group=entry.get("group", ""),
            max_mass=max_mass,
            fill_missing_friction=fill_missing_friction,
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
            "textures": facts["textures"],
            "mass": mass_info,
            "validation": None,
        }
        meta["runtime_ready"] = False
    else:
        facts = adapt_rigid(
            model,
            target,
            max_mass=max_mass if max_mass is not None else DEFAULT_MAX_RIGID_MASS_KG,
            usd_format=usd_format,
        )
        size = facts["size"]
        geom = articulated.box_geometry(size)
        meta = {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{canonical_url()}/{entry['src']}")),
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
                "textures": facts["textures"],
                "mass": facts["mass_info"],
                "validation": None,
            },
            "asset_id": oid,
            "runtime_ready": False,
        }

    articulated.write_json(target / "metadata.json", meta)
    title = (entry.get("title") or entry.get("category") or oid.split("/")[-2]).replace("_", " ")
    prompt = str(pkg_meta.get("prompt") or "")
    desc_text = prompt if prompt else f"A {entry.get('physics_type', 'articulated')} {title} from USDCraft-10K."
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
        "object_id": entry.get("object_id", ""),
        "src": entry["src"],
        "physics_type": entry.get("physics_type", "articulated"),
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
    fill_missing_friction: bool = False,
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
                    fill_missing_friction=fill_missing_friction,
                    usd_format=usd_format,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "id": entry["id"],
                    "object_id": entry.get("object_id", ""),
                    "src": entry["src"],
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
        # Also allow `out_root` to point directly at `Assets/Object/RoboDojo`.
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

    p_filter = sub.add_parser("filter", help="Filter manifest.csv before downloading packages.")
    p_filter.add_argument("--manifest-csv", type=Path, help="Local manifest.csv (auto-fetched if --sources-dir set)")
    p_filter.add_argument("--sources-dir", type=Path, help="Cache dir to auto-fetch manifest.csv into if missing")
    p_filter.add_argument("--revision", default=DEFAULT_REVISION)
    p_filter.add_argument("--endpoint", default=None, help="Override HF_ENDPOINT")
    p_filter.add_argument("--physics-type", choices=("articulated", "static_rigid", "all"), default="articulated")
    p_filter.add_argument("--groups", nargs="*", choices=GROUPS, default=None)
    p_filter.add_argument("--categories", nargs="*", default=None, help="Regex/substring patterns to include")
    p_filter.add_argument("--exclude-categories", nargs="*", default=None, help="Regex/substring patterns to exclude")
    p_filter.add_argument("--textured-only", action="store_true")
    p_filter.add_argument("--joint-types", nargs="*", default=None)
    p_filter.add_argument("--max-joints", type=int, default=None)
    p_filter.add_argument("--max-per-category", type=int, default=None)
    p_filter.add_argument("--limit", type=int, default=None)
    p_filter.add_argument("--object-ids", nargs="*", default=None)
    p_filter.add_argument("--object-ids-file", type=Path, default=None)
    p_filter.add_argument("--out", type=Path, required=True, help="Output selection JSON path")

    p_fetch = sub.add_parser("fetch", help="Selectively download packages listed in a filter selection JSON.")
    p_fetch.add_argument("--selection", type=Path, required=True)
    p_fetch.add_argument("--out", type=Path, required=True, help="Destination sources/usdcraft directory")
    p_fetch.add_argument("--revision", default=None)
    p_fetch.add_argument("--endpoint", default=None)
    p_fetch.add_argument("--workers", type=int, default=4)
    p_fetch.add_argument("--shard-index", type=int, default=0)
    p_fetch.add_argument("--shard-total", type=int, default=1)

    p_survey = sub.add_parser("survey", help="Inspect downloaded packages with pxr and emit conversion manifest.")
    p_survey.add_argument("--source", type=Path, required=True, help="sources/usdcraft directory")
    p_survey.add_argument("--selection", type=Path, default=None, help="Optional filter selection JSON")
    p_survey.add_argument("--max-size", type=float, default=TABLESIDE_MAX_M)
    p_survey.add_argument("--max-rigid-size", type=float, default=TABLETOP_MAX_M)
    p_survey.add_argument("--allow-floating", action="store_true")
    p_survey.add_argument("--keep-tips-over", action="store_true")
    p_survey.add_argument(
        "--fixed-as-rigid",
        action="store_true",
        help="Reclassify articulated packages with no movable joints (fixed-only) as static_rigid",
    )
    p_survey.add_argument(
        "--existing-state",
        type=Path,
        default=None,
        help="Optional JSON with uc_sources / uc_max_idx to skip imported objects and avoid ID collisions",
    )
    p_survey.add_argument("--out", type=Path, required=True, help="Output conversion manifest JSON")

    p_conv = sub.add_parser("convert", help="Flatten/reparent/normalize packages into object.usd + metadata.json.")
    p_conv.add_argument("--source", type=Path, required=True, help="sources/usdcraft directory")
    p_conv.add_argument("--root", type=Path, required=True, help="Output root holding Assets/Object/RoboDojo/")
    p_conv.add_argument("--manifest", type=Path, required=True, help="Manifest JSON from `survey`")
    p_conv.add_argument("--skip-existing", action="store_true")
    p_conv.add_argument("--max-mass", type=float, default=None)
    p_conv.add_argument("--fill-missing-friction", action="store_true")
    p_conv.add_argument("--usd-format", choices=("usdc", "usda"), default="usdc")
    p_conv.add_argument("--summary", type=Path, default=None)

    p_ready = sub.add_parser("ready", help="Stamp check_articulation/check_object results into metadata.json.")
    p_ready.add_argument("--root", type=Path, required=True, help="Output root holding Assets/Object/RoboDojo/")
    p_ready.add_argument(
        "--check-dir", type=Path, required=True, help="--image-dir passed to check_articulation/object"
    )
    p_ready.add_argument("--manifest", type=Path, default=None, help="Manifest JSON listing converted ids")
    p_ready.add_argument("--ids-file", type=Path, default=None, help="Optional newline-separated object ids")
    p_ready.add_argument("ids", nargs="*", help="Explicit object ids (e.g. Articulation/usdcraft_microwave/00000)")
    p_ready.add_argument("--summary", type=Path, default=None)
    p_ready.add_argument("--note", default=None)

    args = parser.parse_args(argv)

    if args.command == "filter":
        csv_path = args.manifest_csv
        if csv_path is None:
            if not args.sources_dir:
                parser.error("filter requires --manifest-csv or --sources-dir")
            csv_path = fetch_manifest_csv(args.sources_dir, revision=args.revision, endpoint=args.endpoint)
        elif not csv_path.is_file():
            if args.sources_dir:
                csv_path = fetch_manifest_csv(args.sources_dir, revision=args.revision, endpoint=args.endpoint)
            else:
                raise FileNotFoundError(f"manifest CSV not found: {csv_path}")
        obj_ids = None
        if args.object_ids or args.object_ids_file:
            obj_ids = set(args.object_ids or [])
            if args.object_ids_file:
                obj_ids |= {
                    ln.strip()
                    for ln in args.object_ids_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")
                }
        rows = load_manifest_csv(csv_path)
        result = filter_manifest(
            rows,
            physics_type=args.physics_type,
            groups=args.groups,
            categories=args.categories,
            exclude_categories=args.exclude_categories,
            textured_only=args.textured_only,
            joint_types=args.joint_types,
            max_joints=args.max_joints,
            max_per_category=args.max_per_category,
            limit=args.limit,
            object_ids=obj_ids,
            revision=args.revision,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, result)
        print(f"[usdcraft] filter {json.dumps(result['summary'], ensure_ascii=False)}", flush=True)
        return 0

    if args.command == "fetch":
        sel = json.loads(args.selection.read_text(encoding="utf-8"))
        entries = sel.get("chosen", [])
        if args.shard_total > 1:
            entries = [e for i, e in enumerate(entries) if i % args.shard_total == args.shard_index]
        revision = args.revision or sel.get("revision", DEFAULT_REVISION)
        res = fetch_packages(entries, args.out, revision=revision, endpoint=args.endpoint, workers=args.workers)
        print(
            f"[usdcraft] fetch shard {args.shard_index}/{args.shard_total}: "
            f"fetched={len(res['fetched'])}, failed={len(res['failed'])}",
            flush=True,
        )
        return 1 if res["failed"] else 0

    if args.command == "survey":
        sel = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
        existing_state = json.loads(args.existing_state.read_text(encoding="utf-8")) if args.existing_state else {}
        res = survey(
            args.source,
            selection=sel,
            max_size=args.max_size,
            max_rigid_size=args.max_rigid_size,
            allow_floating=args.allow_floating,
            drop_tips_over=not args.keep_tips_over,
            fixed_as_rigid=args.fixed_as_rigid,
            existing_sources=existing_state.get("uc_sources"),
            existing_max_index=existing_state.get("uc_max_idx"),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(args.out, res)
        print(f"[usdcraft] survey chosen={len(res['chosen'])}, dropped={len(res['dropped'])}", flush=True)
        return 0

    if args.command == "convert":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        res = convert(
            args.source,
            args.root,
            manifest,
            skip_existing=args.skip_existing,
            max_mass=args.max_mass,
            fill_missing_friction=args.fill_missing_friction,
            usd_format=args.usd_format,
        )
        summary_path = args.summary or (args.root / "usdcraft_conversion.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        articulated.write_json(summary_path, res)
        print(
            f"[usdcraft] convert converted={len(res['converted'])}, skipped={len(res['skipped'])}, "
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
    print(f"[usdcraft] ready {len(passed_ids)}/{len(ids)}; reasons={json.dumps(reason_counts)}", flush=True)
    return 0 if not failed_ids else 1


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
