"""Import CC0 3D models from Poly Haven (api.polyhaven.com) as RoboDojo rigid objects.

Poly Haven publishes ~522 CC0-1.0 3D models as native Z-up, metric `.usdc` files (with MaterialX +
UsdPreviewSurface shaders and zero physics/collision/joints authored). Filtering to tabletop scale
(`<= 0.65 m`), excluding outdoor/nature/architecture/plant categories and multi-piece sets, and
skipping models already imported from scene sources leaves ~180-190 clean tabletop props.

Output layout per object:
    <out>/Assets/Object/<namespace>/Rigid/<category>/<index>/
      object.usd         USDA wrapper (/root defaultPrim, Z-up, metres, RigidBodyAPI, MassAPI, colliders)
      <model>_1k.usdc    copied payload (texture paths rewritten .exr -> .png unless --keep-exr)
      textures/          copied textures (.exr converted to .png by default unless --keep-exr)
      metadata.json      geometry, physics, provenance, adapter record, runtime_ready flag
      description.json   English description + bilingual instruction names (object_names.py schema)

Pipeline subcommands (pure-stdlib except `convert`, which lazy-imports `numpy`/`pxr`/`PIL`):
    catalog  Query api.polyhaven.com/assets?t=models, filter to tabletop rigid props, assign IDs.
    fetch    Download 1k USD packages (.usdc + included textures) with retry/backoff and MD5 check.
    convert  Convert `.exr` -> `.png` (optional), normalize orientation/origin, author colliders & mass.
    ready    Stamp `check_object.py --image-dir` settle results into `metadata.json` (`runtime_ready`).

Usage:
    python3 scripts/objects/polyhaven_models.py catalog --out catalog.json
    python3 scripts/objects/polyhaven_models.py fetch --catalog catalog.json --out sources --workers 4
    isaac-sim python scripts/objects/polyhaven_models.py convert --catalog catalog.json --sources sources --out out
    python3 scripts/objects/polyhaven_models.py ready --out out --check-dir check --ids-file ids.txt
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
import time
from typing import TYPE_CHECKING
import urllib.error
import urllib.parse
import urllib.request
import uuid

if TYPE_CHECKING:
    import numpy as np

SCRIPT = "scripts/objects/polyhaven_models.py"
VERSION = 1

API = "https://api.polyhaven.com"
LICENSE = "CC0-1.0"
USER_AGENT = "curl/8.0"
RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

DEFAULT_SCENE_SOURCES: Path | None = None
SCENE_FALLBACK_IDS = frozenset(
    {
        "ceramic_vase_01",
        "ceramic_vase_03",
        "modern_ceiling_lamp_01",
        "potted_plant_01",
        "potted_plant_02",
        "potted_plant_04",
        "tea_set_01",
    }
)

DEFAULT_NAMESPACE = "RoboDojo"
DEFAULT_RES = "1k"
RES_FALLBACK_ORDER = ("1k", "2k", "4k", "8k")

# Tabletop rigid-object geometry / physics defaults (metres, kg).
MIN_SIZE_M = 0.02
MAX_SIZE_M = 0.65
MIN_THICKNESS_M = 0.003
CONTACT_BAND_M = 0.004
DEFAULT_FRICTION = 0.6

# Collider selection defaults for `--collider auto`.
DECOMP_MAX_MESHES = 16
DECOMP_MAX_FACES = 30_000
PROXY_TARGET_FACES = 4_000
HULL_MAX_MESHES = 32

# Built-in Isaac Sim MDL shaders that cannot be resolved outside Kit's MDL search path.
BUILTIN_MDLS = frozenset({"OmniGlass.mdl", "OmniPBR.mdl", "OmniSurface.mdl", "OmniSurfaceLite.mdl"})

# Poly Haven categories excluded from tabletop rigid-prop ingestion.
EXCLUDE_CATEGORIES = frozenset(
    {
        "buildings",
        "creature",
        "flowers",
        "grass",
        "ground cover",
        "nature",
        "plants",
        "potted plants",
        "rigged",
        "rocks",
        "ships",
        "structures",
        "succulent",
        "trees",
        "wall decoration",
        "walls",
    }
)

# Multi-piece sets / scatter packs whose individual parts should not be fused into one rigid body.
_MULTI_PIECE_ID_RE = re.compile(
    r"(?:^|_)(?:set|collection|pack|kit|pile|stack|group|assortment|variety|bunch|cluster|clutter|debris)(?:_|\d|$)",
    re.I,
)
_MULTI_PIECE_NAME_RE = re.compile(
    r"\b(?:set|collection|kit|pack|assortment|pile|clutter|debris|cluster)\b",
    re.I,
)

# Ordered Poly Haven category -> canonical USD-prim-safe `ph_<slug>` mapping + bilingual category names.
CATEGORY_PRIORITY: tuple[tuple[str, str, str, str], ...] = (
    # (polyhaven_category, slug, category_en, category_zh)
    ("vases", "vase", "vase", "花瓶"),
    ("dishes", "dish", "dish", "餐具"),
    ("food", "food", "food", "食物"),
    ("tools", "tool", "tool", "工具"),
    ("electronics", "electronics", "device", "电子设备"),
    ("appliances", "appliance", "appliance", "电器"),
    ("books", "book", "book", "书籍"),
    ("lighting", "lighting", "lamp", "灯具"),
    ("containers", "container", "container", "容器"),
    ("office", "office", "stationery", "办公用品"),
    ("instrument", "instrument", "instrument", "乐器"),
    ("weapons", "weapon", "prop", "道具"),
    ("firearms", "weapon", "prop", "道具"),
    ("industrial", "industrial", "hardware", "工业件"),
    ("furniture", "furniture", "furniture", "家具"),
    ("decorative", "decor", "ornament", "摆件"),
    ("props", "prop", "prop", "道具"),
)

CATEGORY_BILINGUAL: dict[str, tuple[str, str]] = {f"ph_{slug}": (en, zh) for _, slug, en, zh in CATEGORY_PRIORITY}

# Material-class effective density (kg/m^3), bbox fill factor, and plausible tabletop mass range (kg).
# Upper bounds respect RigidObject's 0.5 kg tabletop cap.
MATERIAL_MASS_SPEC: dict[str, tuple[float, float, tuple[float, float]]] = {
    "paper": (400.0, 0.45, (0.01, 0.30)),
    "cardboard": (220.0, 0.35, (0.02, 0.40)),
    "plastic": (450.0, 0.35, (0.015, 0.45)),
    "glass": (900.0, 0.30, (0.05, 0.45)),
    "ceramic": (1000.0, 0.30, (0.08, 0.45)),
    "stone": (1600.0, 0.45, (0.10, 0.50)),
    "wood": (600.0, 0.50, (0.02, 0.50)),
    "metal": (1800.0, 0.35, (0.03, 0.50)),
    "food": (750.0, 0.55, (0.015, 0.45)),
    "rubber": (850.0, 0.45, (0.015, 0.35)),
    "fabric": (220.0, 0.40, (0.015, 0.35)),
    "leather": (500.0, 0.40, (0.02, 0.45)),
    "electronic": (750.0, 0.50, (0.04, 0.50)),
    "default": (450.0, 0.40, (0.02, 0.50)),
}

# Direct mapping from Poly Haven `attributes.material` tags to MATERIAL_MASS_SPEC keys.
POLYHAVEN_MATERIAL_MAP: dict[str, str] = {
    "ceramic": "ceramic",
    "porcelain": "ceramic",
    "clay": "ceramic",
    "terracotta": "ceramic",
    "glass": "glass",
    "crystal": "glass",
    "metal": "metal",
    "steel": "metal",
    "iron": "metal",
    "brass": "metal",
    "copper": "metal",
    "bronze": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "tin": "metal",
    "gold": "metal",
    "silver": "metal",
    "stone": "stone",
    "marble": "stone",
    "concrete": "stone",
    "plaster": "stone",
    "wood": "wood",
    "bamboo": "wood",
    "cork": "wood",
    "plastic": "plastic",
    "acrylic": "plastic",
    "resin": "plastic",
    "rubber": "rubber",
    "silicone": "rubber",
    "fabric": "fabric",
    "cloth": "fabric",
    "textile": "fabric",
    "wool": "fabric",
    "cotton": "fabric",
    "leather": "leather",
    "paper": "paper",
    "cardboard": "cardboard",
    "food": "food",
    "organic": "food",
}

MATERIAL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ceramic", ("ceramic", "porcelain", "terracotta", "teapot", "teacup", "mug", "bowl", "plate", "dish", "vase")),
    ("glass", ("glass", "jar", "wine", "carafe", "pitcher", "bottle")),
    (
        "metal",
        (
            "metal",
            "steel",
            "iron",
            "brass",
            "copper",
            "bronze",
            "aluminum",
            "aluminium",
            "kettle",
            "pot",
            "pan",
            "goblet",
            "lantern",
            "wrench",
            "hammer",
            "pliers",
            "screwdriver",
            "blowtorch",
            "can",
            "tin",
            "ammo",
            "tool",
        ),
    ),
    ("stone", ("stone", "marble", "granite", "concrete", "bust", "statue", "sculpture")),
    ("wood", ("wood", "wooden", "bamboo", "ukulele", "crate", "chest", "cutting_board")),
    (
        "electronic",
        (
            "electronic",
            "camera",
            "radio",
            "television",
            "projector",
            "console",
            "drill",
            "megaphone",
            "clock",
            "phone",
            "speaker",
            "register",
            "binoculars",
            "appliance",
        ),
    ),
    ("cardboard", ("cardboard", "carton", "box", "packaging")),
    ("paper", ("book", "notebook", "binder", "magazine", "paper", "card")),
    ("leather", ("leather", "football", "baseball", "boot", "shoe", "bag", "satchel")),
    ("fabric", ("fabric", "cloth", "pillow", "cushion", "towel", "hat", "cap")),
    ("food", ("food", "fruit", "apple", "bread", "cheese", "cake", "pastry", "vegetable")),
    ("rubber", ("rubber", "ball", "tire", "hose", "silicone")),
    ("plastic", ("plastic", "cleaner", "bleach", "spray", "bucket", "tray", "sign", "container")),
)

# Texture filename stem tokens that indicate linear (non-color) data maps.
_LINEAR_TEX_RE = re.compile(
    r"(?:^|[_\-.])(nor|nor_gl|nor_dx|normal|rough|roughness|metal|metallic|ao|arm|orm|disp|displacement|bump|height|spec|specular|opacity|alpha|gloss)(?:$|[_\-.]|\d)",
    re.I,
)
_COLOR_TEX_RE = re.compile(
    r"(?:^|[_\-.])(diff|diffuse|albedo|col|color|basecolor|base_color|emit|emissive)(?:$|[_\-.]|\d)",
    re.I,
)


def r4(values) -> list[float]:
    return [round(float(x), 4) for x in values]


def object_uuid(object_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo/{object_id}"))


def http_get(
    url: str,
    *,
    attempts: int = 6,
    timeout: float = 120.0,
    backoff: float = 2.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Fetch `url` with `User-Agent: curl/8.0` and exponential backoff on 429/5xx/transient errors."""
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in RETRYABLE_HTTP_CODES or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = backoff * (2**attempt)
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            time.sleep(min(delay, 60.0))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            time.sleep(min(backoff * (2**attempt), 60.0))
    assert last_exc is not None
    raise last_exc


def http_get_json(url: str, **kwargs) -> dict:
    return json.loads(http_get(url, **kwargs).decode("utf-8"))


def _format_authors(raw_authors) -> list[str]:
    if isinstance(raw_authors, dict):
        return [str(k) for k in raw_authors if k]
    if isinstance(raw_authors, list):
        return [str(a) for a in raw_authors if a]
    if isinstance(raw_authors, str) and raw_authors.strip():
        return [raw_authors.strip()]
    return []


def load_skip_ids(
    sources_paths: list[Path] | None = None,
    assets_dir: Path | None = None,
) -> dict[str, str]:
    """Collect `{polyhaven_model_id: reason}` for models already imported in scenes or `Assets/`."""
    found: dict[str, str] = {}
    paths = (
        list(sources_paths) if sources_paths is not None else ([DEFAULT_SCENE_SOURCES] if DEFAULT_SCENE_SOURCES else [])
    )
    loaded_any_sources = False
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        loaded_any_sources = True
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in data.get("models") or {}:
            found[str(key)] = f"already in {p.as_posix()}"

    if sources_paths is None and not loaded_any_sources:
        for key in sorted(SCENE_FALLBACK_IDS):
            found[key] = "already imported in default scene fallback set"

    if assets_dir is not None and Path(assets_dir).exists():
        for meta_path in Path(assets_dir).glob("Object/*/*/*/*/metadata.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            src = meta.get("source") or {}
            upstream = str(src.get("upstream_dataset") or src.get("dataset") or "").lower()
            uid = src.get("upstream_id") or src.get("model_id") or src.get("id")
            if "poly haven" in upstream and uid:
                oid = "/".join(meta_path.parts[-4:-1])
                found[str(uid)] = f"already imported as {oid}"
    return found


def polyhaven_category(info: dict) -> str:
    """Map a Poly Haven model's `categories` list to a clean USD-prim-safe `ph_<slug>` category."""
    cats = {str(c).strip().lower() for c in (info.get("categories") or []) if c}
    for ph_cat, slug, _, _ in CATEGORY_PRIORITY:
        if ph_cat in cats:
            return f"ph_{slug}"
    # Fallback to non-collection category token if present.
    for raw in sorted(cats):
        if raw.startswith("collection:"):
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if slug:
            return f"ph_{slug}"
    return "ph_prop"


def is_multi_piece_model(model_id: str, info: dict) -> bool:
    """True when a Poly Haven model ID or title denotes a multi-piece set or scatter pack."""
    if _MULTI_PIECE_ID_RE.search(model_id):
        return True
    name = str(info.get("name") or "")
    if _MULTI_PIECE_NAME_RE.search(name):
        return True
    return False


def filter_models(
    assets: dict[str, dict],
    *,
    skip_ids: dict[str, str] | None = None,
    min_size: float = MIN_SIZE_M,
    max_size: float = MAX_SIZE_M,
    min_thickness: float = MIN_THICKNESS_M,
    exclude_categories: frozenset[str] | set[str] | None = None,
    include_categories: set[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Filter Poly Haven `assets?t=models` dict into deterministic tabletop rigid-prop catalog entries."""
    excl = {c.lower() for c in (EXCLUDE_CATEGORIES if exclude_categories is None else exclude_categories)}
    incl = {c.lower() for c in include_categories} if include_categories else None
    skip_map = skip_ids or {}

    selected_raw: list[tuple[str, dict, str, list[float]]] = []
    skipped: list[dict] = []

    for model_id in sorted(assets):
        info = assets[model_id]
        if not isinstance(info, dict):
            skipped.append({"model_id": model_id, "reason": "invalid metadata"})
            continue
        if model_id in skip_map:
            skipped.append({"model_id": model_id, "reason": skip_map[model_id]})
            continue

        cats = {str(c).strip().lower() for c in (info.get("categories") or []) if c}
        hit_excl = sorted(cats & excl)
        if hit_excl:
            skipped.append({"model_id": model_id, "reason": f"excluded category: {', '.join(hit_excl)}"})
            continue
        if incl is not None and not (cats & incl):
            skipped.append({"model_id": model_id, "reason": "not in included categories"})
            continue

        if is_multi_piece_model(model_id, info):
            skipped.append({"model_id": model_id, "reason": "multi-piece set"})
            continue

        dims_mm = info.get("dimensions")
        if not isinstance(dims_mm, list | tuple) or len(dims_mm) != 3:
            skipped.append({"model_id": model_id, "reason": "missing 3D dimensions"})
            continue
        try:
            dims_m = [float(x) / 1000.0 for x in dims_mm]
        except (TypeError, ValueError):
            skipped.append({"model_id": model_id, "reason": "non-numeric dimensions"})
            continue
        if any(not math.isfinite(d) or d <= 0 for d in dims_m):
            skipped.append({"model_id": model_id, "reason": "degenerate dimensions"})
            continue
        longest = max(dims_m)
        shortest = min(dims_m)
        if longest > max_size:
            skipped.append({"model_id": model_id, "reason": f"oversize ({longest:.3f} m > {max_size:.3f} m)"})
            continue
        if longest < min_size:
            skipped.append({"model_id": model_id, "reason": f"undersize ({longest:.3f} m < {min_size:.3f} m)"})
            continue
        if shortest < min_thickness:
            skipped.append({"model_id": model_id, "reason": f"too thin ({shortest:.4f} m < {min_thickness:.4f} m)"})
            continue

        cat = polyhaven_category(info)
        selected_raw.append((model_id, info, cat, dims_m))

    if limit is not None and limit >= 0:
        selected_raw = selected_raw[:limit]

    per_cat_index: dict[str, int] = {}
    models: list[dict] = []
    for global_idx, (model_id, info, cat, dims_m) in enumerate(selected_raw):
        idx = per_cat_index.get(cat, 0)
        per_cat_index[cat] = idx + 1
        index_str = f"{idx:05d}"
        object_id = f"Rigid/{cat}/{index_str}"
        models.append(
            {
                "global_index": global_idx,
                "id": object_id,
                "category": cat,
                "index": index_str,
                "model_id": model_id,
                "name": str(info.get("name") or model_id),
                "authors": _format_authors(info.get("authors")),
                "categories": list(info.get("categories") or []),
                "category_path": info.get("category") or "",
                "tags": list(info.get("tags") or []),
                "attributes": info.get("attributes") or {},
                "polycount": int(info["polycount"]) if info.get("polycount") is not None else None,
                "dimensions_mm": [round(float(x), 3) for x in info["dimensions"]],
                "dimensions_m": r4(dims_m),
                "description": str(info.get("description") or ""),
                "download_count": int(info.get("download_count") or 0),
                "license": LICENSE,
                "url": f"https://polyhaven.com/a/{model_id}",
            }
        )

    return {
        "script": SCRIPT,
        "version": VERSION,
        "source": "Poly Haven (polyhaven.com) 3D models",
        "license": LICENSE,
        "total_upstream": len(assets),
        "selected_count": len(models),
        "skipped_count": len(skipped),
        "filters": {
            "min_size_m": min_size,
            "max_size_m": max_size,
            "min_thickness_m": min_thickness,
            "exclude_categories": sorted(excl),
            "include_categories": sorted(incl) if incl is not None else None,
        },
        "models": models,
        "skipped": skipped,
    }


def resolve_usd_package(files_data: dict, res: str = DEFAULT_RES) -> tuple[str, dict]:
    """Extract `(chosen_res, usd_pkg_dict)` from `api.polyhaven.com/files/<id>` response."""
    usd_section = files_data.get("usd")
    if not isinstance(usd_section, dict) or not usd_section:
        raise KeyError("no usd section in Poly Haven files API")

    candidates = [res] + [r for r in RES_FALLBACK_ORDER if r != res]
    for cand in candidates:
        entry = usd_section.get(cand)
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("usd") if isinstance(entry.get("usd"), dict) else entry
        if isinstance(pkg, dict) and pkg.get("url"):
            return cand, pkg

    # Last resort: pick first available resolution key with a url.
    for cand in sorted(usd_section):
        entry = usd_section[cand]
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("usd") if isinstance(entry.get("usd"), dict) else entry
        if isinstance(pkg, dict) and pkg.get("url"):
            return cand, pkg

    raise KeyError(f"no downloadable USD package found (requested res={res!r})")


def _safe_relpath(rel: str) -> Path:
    """Validate an include relative path from the API so it cannot escape the target directory."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe relative path in Poly Haven include: {rel!r}")
    return p


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _download_verified(url: str, dst: Path, *, expected_md5: str | None, getter) -> int:
    data = getter(url)
    if expected_md5:
        actual = _md5_bytes(data)
        if actual.lower() != expected_md5.lower():
            raise ValueError(f"md5 mismatch for {url}: got {actual}, expected {expected_md5}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return len(data)


def fetch_one(
    model_entry: dict,
    out_dir: Path,
    *,
    res: str = DEFAULT_RES,
    skip_existing: bool = True,
    getter=http_get,
) -> dict:
    """Download one Poly Haven USD model package (`.usdc` + all `include` textures) into `<out_dir>/<model_id>/`."""
    model_id = model_entry["model_id"]
    model_dir = Path(out_dir) / model_id
    meta_file = model_dir / "polyhaven_meta.json"

    if skip_existing and meta_file.is_file():
        try:
            saved = json.loads(meta_file.read_text(encoding="utf-8"))
            usd_rel = saved.get("usd_file")
            includes = saved.get("includes") or []
            if usd_rel and (model_dir / usd_rel).is_file() and all((model_dir / rel).is_file() for rel in includes):
                return {
                    "model_id": model_id,
                    "status": "skipped_existing",
                    "res": saved.get("res"),
                    "usd_file": str(model_dir / usd_rel),
                    "file_count": 1 + len(includes),
                }
        except Exception:
            pass

    files_data = json.loads(getter(f"{API}/files/{model_id}").decode("utf-8"))
    chosen_res, pkg = resolve_usd_package(files_data, res=res)
    usd_url = pkg["url"]
    usd_filename = Path(urllib.parse.urlparse(usd_url).path).name or f"{model_id}_{chosen_res}.usdc"

    tmp_dir = Path(out_dir) / f".tmp_{model_id}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_bytes = _download_verified(
            usd_url,
            tmp_dir / usd_filename,
            expected_md5=pkg.get("md5"),
            getter=getter,
        )
        includes_dict = pkg.get("include") or {}
        included_rels: list[str] = []
        exr_count = 0
        for rel_str, inc in sorted(includes_dict.items()):
            rel_path = _safe_relpath(rel_str)
            total_bytes += _download_verified(
                inc["url"],
                tmp_dir / rel_path,
                expected_md5=inc.get("md5"),
                getter=getter,
            )
            rel_posix = rel_path.as_posix()
            included_rels.append(rel_posix)
            if rel_path.suffix.lower() == ".exr":
                exr_count += 1

        meta_record = {
            **model_entry,
            "res": chosen_res,
            "usd_file": usd_filename,
            "usd_url": usd_url,
            "usd_md5": pkg.get("md5"),
            "includes": included_rels,
            "exr_count": exr_count,
            "total_bytes": total_bytes,
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (tmp_dir / "polyhaven_meta.json").write_text(
            json.dumps(meta_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if model_dir.exists():
            shutil.rmtree(model_dir)
        tmp_dir.rename(model_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return {
        "model_id": model_id,
        "status": "downloaded",
        "res": chosen_res,
        "usd_file": str(model_dir / usd_filename),
        "file_count": 1 + len(included_rels),
        "exr_count": exr_count,
        "total_bytes": total_bytes,
    }


def fetch_catalog(
    catalog: dict,
    out_dir: Path,
    *,
    res: str = DEFAULT_RES,
    workers: int = 4,
    shard_index: int = 0,
    shard_total: int = 1,
    skip_existing: bool = True,
    getter=http_get,
) -> dict:
    """Download a sharded subset of a Poly Haven catalog concurrently."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_models = catalog.get("models") or []
    models = [m for i, m in enumerate(all_models) if i % max(1, shard_total) == shard_index]

    downloaded, skipped, failed = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(fetch_one, item, out_dir, res=res, skip_existing=skip_existing, getter=getter): item["model_id"]
            for item in models
        }
        for fut in as_completed(futs):
            mid = futs[fut]
            try:
                res_item = fut.result()
                if res_item["status"] == "downloaded":
                    downloaded.append(res_item)
                else:
                    skipped.append(res_item)
            except Exception as exc:
                failed.append({"model_id": mid, "error": f"{type(exc).__name__}: {exc}"})

    downloaded.sort(key=lambda x: x["model_id"])
    skipped.sort(key=lambda x: x["model_id"])
    failed.sort(key=lambda x: x["model_id"])
    return {
        "script": SCRIPT,
        "version": VERSION,
        "res": res,
        "shard": [shard_index, shard_total],
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# EXR -> PNG conversion (lazy numpy / PIL / OpenEXR / cv2 / imageio)
# ---------------------------------------------------------------------------


def is_linear_texture(path: Path) -> bool:
    """True when a texture filename denotes linear data (normal/rough/metal/ao/disp), False for color maps."""
    stem = path.stem.lower()
    if _COLOR_TEX_RE.search(stem):
        return False
    if _LINEAR_TEX_RE.search(stem):
        return True
    return True


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert linear float RGB(A) array to sRGB [0, 1], applying Reinhard tone-mapping if HDR (> 1.0)."""
    import numpy as np

    out = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    out = np.maximum(out, 0.0)
    if out.ndim == 3 and out.shape[2] == 4:
        rgb = out[:, :, :3]
        alpha = np.clip(out[:, :, 3:4], 0.0, 1.0)
    else:
        rgb = out
        alpha = None

    if float(rgb.max()) > 1.0:
        rgb = rgb / (1.0 + rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)
    if alpha is not None:
        return np.concatenate([srgb, alpha], axis=2)
    return srgb


def read_exr(path: Path) -> np.ndarray:
    """Read an `.exr` image as float32 `(H, W)` or `(H, W, C)` in RGB(A) channel order."""
    import numpy as np

    # 1. OpenEXR + Imath (when installed)
    try:
        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        try:
            header = exr.header()
            dw = header["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = set(header["channels"].keys())
            if {"R", "G", "B"}.issubset(channels):
                names = ["R", "G", "B"] + (["A"] if "A" in channels else [])
                planes = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w) for c in names]
                return np.stack(planes, axis=-1)
            first = sorted(channels)[0]
            return np.frombuffer(exr.channel(first, pt), dtype=np.float32).reshape(h, w)
        finally:
            exr.close()
    except Exception:
        pass

    # 2. OpenCV with OpenEXR enabled (BGR -> RGB swap)
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, [2, 1, 0, 3]].copy()
            return arr
    except Exception:
        pass

    # 3. imageio fallback
    try:
        import imageio.v3 as iio

        return np.asarray(iio.imread(path), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"unable to decode EXR {path} (install OpenEXR, opencv-python, or imageio): {exc}") from exc


def convert_exr_to_png(
    exr_path: Path,
    png_path: Path | None = None,
    *,
    is_linear: bool | None = None,
) -> Path:
    """Convert an `.exr` texture into an 8-bit `.png`, preserving linear data vs sRGB color semantics."""
    import numpy as np
    from PIL import Image

    exr_path = Path(exr_path)
    dst = Path(png_path) if png_path is not None else exr_path.with_suffix(".png")
    linear = is_linear_texture(exr_path) if is_linear is None else is_linear

    arr = read_exr(exr_path)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if linear:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = linear_to_srgb(arr)

    u8 = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8).save(dst, format="PNG")
    return dst


def convert_directory_exrs(target_dir: Path, *, strict: bool = False) -> tuple[dict[str, str], list[str]]:
    """Convert all `.exr` files under `target_dir` to `.png`.

    Returns `(replacements, warnings)` where `replacements` maps lowercase basename (e.g.
    `foo_nor_gl_1k.exr`) -> new basename (`foo_nor_gl_1k.png`). Does not delete `.exr` files yet
    so USD asset-path rewriting can run safely before cleanup.
    """
    replacements: dict[str, str] = {}
    warnings: list[str] = []
    for exr in sorted(Path(target_dir).rglob("*.exr")):
        if not exr.is_file():
            continue
        png = exr.with_suffix(".png")
        try:
            convert_exr_to_png(exr, png)
            replacements[exr.name.lower()] = png.name
        except Exception as exc:
            if strict:
                raise
            warnings.append(f"{exr.name}: {exc}")
    return replacements, warnings


def build_texture_remap(target_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Index actual texture files under `target_dir/textures/` by lowercase filename and stem.

    Returns `(by_name, by_stem)` mapping lowercase basename/stem to `./textures/<relpath>`.
    Used to repair Poly Haven USDs that contain absolute farm paths (`/mnt/prod/.../textures/...`)
    or extension mismatches (e.g. `.exr` referenced when 1k package ships `.jpg`, or `.png` after EXR->PNG).
    """
    tex_dir = Path(target_dir) / "textures"
    by_name: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    if not tex_dir.is_dir():
        return by_name, by_stem
    # Prefer .png > .jpg/.jpeg > other when multiple extensions share a stem.
    ext_rank = {".png": 0, ".jpg": 1, ".jpeg": 1, ".webp": 2, ".tif": 3, ".tiff": 3, ".exr": 4}
    files = sorted(
        (p for p in tex_dir.rglob("*") if p.is_file()),
        key=lambda p: (ext_rank.get(p.suffix.lower(), 9), p.as_posix()),
    )
    for p in files:
        rel = "./" + p.relative_to(target_dir).as_posix()
        by_name.setdefault(p.name.lower(), rel)
        by_stem.setdefault(p.stem.lower(), rel)
    return by_name, by_stem


def rewrite_asset_path_str(
    asset_path: str,
    replacements: dict[str, str],
    stem_replacements: dict[str, str] | None = None,
) -> str:
    """Rewrite a USD asset path using exact filename or stem lookup (case-insensitive).

    If a replacement value already contains `/` (e.g. `./textures/foo.png`), it replaces the entire
    asset path (stripping stale absolute prefixes like `/mnt/prod/.../staging/textures/`).
    Otherwise it preserves the original directory prefix.
    """
    if not asset_path or (not replacements and not stem_replacements):
        return asset_path
    prefix, sep, name = asset_path.rpartition("/")
    if not name:
        return asset_path
    new_val = replacements.get(name.lower()) if replacements else None
    if new_val is None and stem_replacements:
        stem = Path(name).stem.lower()
        new_val = stem_replacements.get(stem)
    if new_val is None:
        return asset_path
    if "/" in new_val:
        return new_val
    return f"{prefix}{sep}{new_val}" if sep else new_val


def rewrite_usd_texture_paths(
    usd_path: Path,
    replacements: dict[str, str],
    stem_replacements: dict[str, str] | None = None,
) -> int:
    """Rewrite texture asset path references in `usd_path` (`.usdc` or `.usda`)."""
    if not replacements and not stem_replacements:
        return 0
    usd_path = Path(usd_path)
    try:
        _, Sdf, _, _, _, _, UsdUtils, _ = pxr_modules()
        layer = Sdf.Layer.Find(str(usd_path))
        if layer is not None:
            layer.Reload(force=True)
        else:
            layer = Sdf.Layer.FindOrOpen(str(usd_path))
        if layer is None:
            raise RuntimeError(f"failed to open USD layer: {usd_path}")
        changed = 0

        def _remap(path: str) -> str:
            nonlocal changed
            updated = rewrite_asset_path_str(path, replacements, stem_replacements)
            if updated != path:
                changed += 1
            return updated

        UsdUtils.ModifyAssetPaths(layer, _remap)
        if changed:
            layer.Save()
        return changed
    except ImportError:
        if usd_path.suffix.lower() != ".usda":
            raise
        text = usd_path.read_text(encoding="utf-8")
        changed = 0

        def _sub(match: re.Match[str]) -> str:
            nonlocal changed
            inner = match.group(1)
            updated = rewrite_asset_path_str(inner, replacements, stem_replacements)
            if updated != inner:
                changed += 1
            return f"@{updated}@"

        new_text = re.sub(r"@([^@]+)@", _sub, text)
        if changed:
            usd_path.write_text(new_text, encoding="utf-8")
        return changed


# ---------------------------------------------------------------------------
# Geometry, orientation, mass, collider, and description helpers (lazy numpy)
# ---------------------------------------------------------------------------


def triangles(counts, indices) -> np.ndarray:
    """Fan-triangulate USD polygon face arrays into `(T, 3)` int64 indices."""
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if len(counts) == 0:
        return np.empty((0, 3), dtype=np.int64)
    if (counts == 3).all():
        return indices.reshape(-1, 3)
    if (counts == 4).all():
        q = indices.reshape(-1, 4)
        return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=0)
    tris: list[tuple[int, int, int]] = []
    offset = 0
    for n in counts:
        face = indices[offset : offset + n]
        for k in range(1, n - 1):
            tris.append((int(face[0]), int(face[k]), int(face[k + 1])))
        offset += n
    return np.asarray(tris, dtype=np.int64) if tris else np.empty((0, 3), dtype=np.int64)


def mesh_signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    """Absolute signed volume (m^3) of a triangle mesh via divergence theorem."""
    import numpy as np

    if len(faces) == 0 or len(points) == 0:
        return 0.0
    centered = points - points.mean(0)
    tris = centered[faces]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vol6 = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()
    return float(abs(vol6) / 6.0)


def infer_material_class(model_entry: dict) -> str:
    """Infer a `MATERIAL_MASS_SPEC` key from Poly Haven `attributes.material`, categories, tags, and name."""
    attrs = model_entry.get("attributes") or {}
    raw_mats = attrs.get("material") or []
    if isinstance(raw_mats, str):
        raw_mats = [raw_mats]
    for m in raw_mats:
        key = str(m).strip().lower()
        if key in POLYHAVEN_MATERIAL_MAP:
            return POLYHAVEN_MATERIAL_MAP[key]

    haystack = " ".join(
        [
            str(model_entry.get("model_id") or ""),
            str(model_entry.get("name") or ""),
            " ".join(str(c) for c in (model_entry.get("categories") or [])),
            " ".join(str(t) for t in (model_entry.get("tags") or [])),
        ]
    ).lower()
    tokens = re.sub(r"[^a-z0-9]+", " ", haystack)
    for mat_class, keywords in MATERIAL_KEYWORDS:
        for kw in keywords:
            pat = kw.replace("_", " ")
            if re.search(rf"\b{re.escape(pat)}s?\b", tokens):
                return mat_class
    return "default"


def estimate_mass(points: np.ndarray, faces: np.ndarray, model_entry: dict) -> tuple[float, dict]:
    """Estimate realistic rigid-object mass (kg) from mesh volume and inferred material class."""
    import numpy as np

    lo, hi = points.min(0), points.max(0)
    ext = hi - lo
    bbox_vol = float(np.prod(np.maximum(ext, 1e-4)))
    mat_class = infer_material_class(model_entry)
    density, fill_factor, (lo_kg, hi_kg) = MATERIAL_MASS_SPEC[mat_class]

    signed_vol = mesh_signed_volume(points, faces)
    if 0.05 * bbox_vol <= signed_vol <= 1.05 * bbox_vol:
        volume = signed_vol
        vol_source = "mesh_signed_volume"
    else:
        volume = bbox_vol * fill_factor
        vol_source = f"bbox_fill_{fill_factor}"

    raw_kg = volume * density
    clamped_kg = float(np.clip(raw_kg, lo_kg, hi_kg))
    info = {
        "material_class": mat_class,
        "density_kg_m3": density,
        "volume_m3": round(volume, 8),
        "volume_source": vol_source,
        "raw_kg": round(raw_kg, 4),
        "final_kg": round(clamped_kg, 4),
        "clamped": abs(clamped_kg - raw_kg) > 1e-6,
        "bounds_kg": [lo_kg, hi_kg],
    }
    return round(clamped_kg, 4), info


def stability_metrics(points: np.ndarray) -> dict:
    """Estimate static upright stability of a Z-up point cloud from its bottom contact band."""
    lo, hi = points.min(0), points.max(0)
    ext = hi - lo
    h = float(ext[2])
    band = max(CONTACT_BAND_M, 0.03 * h)
    foot = points[points[:, 2] <= lo[2] + band, :2]
    if len(foot) < 3:
        return {
            "stable": False,
            "base_span": 0.0,
            "com_margin": -1.0,
            "aspect": float("inf"),
            "height": h,
            "foot_area": 0.0,
        }
    foot_lo, foot_hi = foot.min(0), foot.max(0)
    foot_ext = foot_hi - foot_lo
    base_span = float(min(foot_ext[0], foot_ext[1]))
    com_xy = points[:, :2].mean(0)
    margin_x = min(float(com_xy[0] - foot_lo[0]), float(foot_hi[0] - com_xy[0]))
    margin_y = min(float(com_xy[1] - foot_lo[1]), float(foot_hi[1] - com_xy[1]))
    com_margin = min(margin_x, margin_y)
    aspect = h / max(base_span, 1e-6)
    stable = (com_margin >= -0.002) and (base_span >= 0.015 or aspect <= 2.5) and (aspect <= 4.0)
    return {
        "stable": stable,
        "base_span": base_span,
        "com_margin": com_margin,
        "aspect": aspect,
        "height": h,
        "foot_area": float(foot_ext[0] * foot_ext[1]),
    }


def compute_pose(points: np.ndarray, *, orient: str = "auto") -> tuple[np.ndarray, dict]:
    """Compute a 3x3 cardinal rotation aligning the model to a stable upright Z-up pose.

    - `keep`: preserve Poly Haven's native Z-up orientation (identity).
    - `lay-flat`: pick the cardinal rotation with smallest Z height and positive CoM margin.
    - `auto`: preserve native Z-up when statically stable; otherwise pick the most stable cardinal face.
    """
    import numpy as np

    if orient not in {"auto", "keep", "lay-flat"}:
        raise ValueError(f"unsupported orient mode: {orient}")

    cardinal: tuple[tuple[str, np.ndarray], ...] = (
        ("+z", np.eye(3)),
        ("-z", np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])),
        ("+x", np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])),
        ("-x", np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])),
        ("+y", np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
        ("-y", np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])),
    )

    base_rot = np.eye(3)
    chosen_face = "+z"
    metrics = stability_metrics(points)

    if orient == "lay-flat":
        best_score = None
        for face, rot in cardinal:
            cand = points @ rot.T
            m = stability_metrics(cand)
            score = (1 if m["com_margin"] >= 0 else 0, -m["height"], m["foot_area"])
            if best_score is None or score > best_score:
                best_score = score
                base_rot = rot
                chosen_face = face
                metrics = m
    elif orient == "auto" and not metrics["stable"]:
        best_score = None
        for face, rot in cardinal:
            cand = points @ rot.T
            m = stability_metrics(cand)
            score = (1 if m["stable"] else 0, 1 if m["com_margin"] >= 0 else 0, m["foot_area"], -m["aspect"])
            if best_score is None or score > best_score:
                best_score = score
                base_rot = rot
                chosen_face = face
                metrics = m

    info = {
        "orient_mode": orient,
        "up_face": chosen_face,
        "stable_estimate": bool(metrics["stable"]),
        "aspect": round(float(metrics["aspect"]), 3) if math.isfinite(metrics["aspect"]) else None,
    }
    return base_rot, info


def simplify_collision_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    target_faces: int = PROXY_TARGET_FACES,
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a high-poly mesh to `<= target_faces` using `fast_simplification` with pure-numpy voxel fallback."""
    import numpy as np

    if len(faces) <= target_faces:
        return points, faces
    try:
        import fast_simplification

        col_pts, col_faces = fast_simplification.simplify(
            points.astype(np.float32), faces, target_reduction=1.0 - target_faces / len(faces)
        )
        if len(col_faces) > 0:
            return np.asarray(col_pts, dtype=float), np.asarray(col_faces, dtype=np.int64)
    except Exception:
        pass

    lo, hi = points.min(0), points.max(0)
    span = np.maximum(hi - lo, 1e-6)
    max_span = float(span.max())
    res = max(6, int(math.sqrt(max(target_faces, 100) / 2.0)))
    best: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(16):
        cell_size = max_span / res
        quant = np.floor((points - lo) / cell_size).astype(np.int64)
        uniq, inv = np.unique(quant, axis=0, return_inverse=True)
        sums = np.zeros((len(uniq), 3), dtype=float)
        counts = np.bincount(inv, minlength=len(uniq))
        np.add.at(sums, inv, points)
        new_pts = sums / counts[:, None]
        new_faces = inv[faces]
        valid = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2])
        )
        new_faces = new_faces[valid]
        if len(new_faces) > 0:
            new_faces = np.unique(new_faces, axis=0)
            if best is None or len(new_faces) < len(best[1]):
                best = (new_pts, new_faces.astype(np.int64))
        if len(new_faces) <= target_faces or res <= 6:
            if len(new_faces) > 0:
                return new_pts, new_faces.astype(np.int64)
            break
        res = max(6, int(res * 0.75))
    if best is not None:
        return best
    return points, faces


def choose_collider_mode(
    mesh_count: int,
    total_faces: int,
    *,
    requested: str = "auto",
    decomp_max_meshes: int = DECOMP_MAX_MESHES,
    decomp_max_faces: int = DECOMP_MAX_FACES,
) -> str:
    """Pick PhysX collider strategy preserving concave geometry on high-poly photogrammetry models.

    In `auto` mode:
    - Low/medium-poly models (`mesh_count <= decomp_max_meshes` and `total_faces <= decomp_max_faces`):
      `convexDecomposition` directly on visual meshes.
    - High-poly / multi-mesh models: `proxy` (decimated combined mesh + `convexDecomposition`),
      avoiding `convexHull` which would seal concave bowls, cups, and vases.
    """
    if requested != "auto":
        return requested
    if mesh_count <= decomp_max_meshes and total_faces <= decomp_max_faces:
        return "convexDecomposition"
    return "proxy"


def geometry_metadata(points: np.ndarray, faces: int) -> tuple[dict, dict]:
    """Compute `geometry` and `active.place.up` for root-frame points centered on their bounding box."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    corners = [r4((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    box = {"vertices": corners, "extents": r4(ext)}
    foot = pts[pts[:, 2] <= lo[2] + CONTACT_BAND_M, :2]
    if len(foot) == 0:
        foot = pts[:, :2]
    foot_c = (foot.min(0) + foot.max(0)) / 2
    geometry = {
        "faces": int(faces),
        "vertices": int(len(pts)),
        "aligned_bbox": box,
        "oriented_bbox": box,
        "radius": round(float(np.linalg.norm(pts, axis=1).max()), 4),
        "bbox": r4(ext),
        "center": r4((lo + hi) / 2),
    }
    up = {
        "orientation": [1.0, 0.0, 0.0, 0.0],
        "projection_circle": {
            "center": r4((0.0, 0.0, lo[2], 1.0, 0.0, 0.0, 0.0)),
            "radius": round(float(np.linalg.norm(pts[:, :2], axis=1).max()), 4),
        },
        "contact_circle": {
            "center": r4((foot_c[0], foot_c[1], lo[2], 1.0, 0.0, 0.0, 0.0)),
            "radius": round(float(np.linalg.norm(foot - foot_c, axis=1).max()), 4),
        },
    }
    return geometry, {"place": {"up": up}, "functional": {}}


def unresolved_dependencies(paths: list[str]) -> list[str]:
    """Filter out known built-in Isaac Sim MDL shaders from unresolved USD asset dependencies."""
    return sorted(p for p in paths if Path(p).name not in BUILTIN_MDLS)


_BAD_EN_WORDS = re.compile(r"\b(left|right|front|back|bigger|smaller|larger)\b", re.I)
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\b\s*", re.I)
_BAD_ZH_CHARS = re.compile(r"^(?:一|这|那)|[。！？]$|[左右更]")


def _clean_en_phrase(raw: str) -> str:
    text = re.sub(r"[_-]+", " ", raw)
    text = _BAD_EN_WORDS.sub(" ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    while _LEADING_ARTICLE.match(text):
        text = _LEADING_ARTICLE.sub("", text).strip()
    return text or "object"


def build_description(model_entry: dict, object_id: str) -> dict:
    """Build `description.json` satisfying `scripts/objects/object_names.py` conventions."""
    uid = object_uuid(object_id)
    model_id = str(model_entry.get("model_id") or "model")
    category = str(model_entry.get("category") or "ph_prop")
    cat_en, cat_zh = CATEGORY_BILINGUAL.get(category, ("prop", "道具"))

    raw_name = str(model_entry.get("name") or model_id)
    # Strip trailing numeric variant suffix for the brief name, keep full identifier in detailed name.
    base_clean = _clean_en_phrase(re.sub(r"[_\s]+\d+$", "", raw_name))
    full_clean = _clean_en_phrase(model_id)
    if base_clean == cat_en:
        en_brief = f"tabletop {base_clean}"
    elif len(base_clean.split()) < len(cat_en.split()):
        en_brief = f"tabletop {base_clean}"
    else:
        en_brief = base_clean

    mat = infer_material_class(model_entry)
    mat_adj = mat if mat != "default" else "textured"
    if mat_adj in en_brief.split():
        en_mid = f"scanned {en_brief}"
    else:
        en_mid = f"{mat_adj} {en_brief}"
    if en_mid == en_brief:
        en_mid = f"tabletop {en_brief}"

    # Detailed name uniquely identifies the model_id so no two Poly Haven objects collide.
    en_detail = f"polyhaven {en_mid} {full_clean}"
    if len(en_detail.split()) <= len(en_mid.split()) or en_detail == en_mid:
        en_detail = f"polyhaven scanned {en_mid} {full_clean}"

    zh_token = _BAD_ZH_CHARS.sub("", full_clean.replace(" ", "")) or "item"
    zh_brief = f"{cat_zh}{zh_token}"
    zh_mid = f"桌面{zh_brief}"
    zh_detail = f"扫描模型{zh_mid}"

    desc_text = str(model_entry.get("description") or "").strip()
    if not desc_text:
        desc_text = f"A CC0 3D {en_mid} model from Poly Haven ({model_id})."

    return {
        "uuid": uid,
        "description": [desc_text],
        "names": {
            "category": {"en": cat_en, "zh": cat_zh},
            "en": [en_brief, en_mid, en_detail],
            "zh": [zh_brief, zh_mid, zh_detail],
        },
    }


# ---------------------------------------------------------------------------
# USD conversion (lazy pxr + SimulationApp fallback)
# ---------------------------------------------------------------------------


def pxr_modules():
    """Import pxr modules, falling back to headless Isaac SimulationApp when standalone usd-core is absent."""
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        globals()["APP"] = SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt


def _find_source_usd(model_dir: Path, meta: dict | None = None) -> Path:
    if meta and meta.get("usd_file"):
        cand = model_dir / meta["usd_file"]
        if cand.is_file():
            return cand
    for ext in (".usdc", ".usd", ".usda"):
        hits = sorted(model_dir.glob(f"*{ext}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no .usdc/.usd/.usda file found in {model_dir}")


def _copy_payload(src_usd: Path, target_dir: Path, includes: list[str] | None = None) -> str:
    """Copy `<model>.usdc`, `textures/` (if present), and any other fetched `includes` into `target_dir`."""
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_usd = target_dir / src_usd.name
    shutil.copy2(src_usd, dst_usd)
    src_tex = src_usd.parent / "textures"
    dst_tex = target_dir / "textures"
    if dst_tex.exists():
        shutil.rmtree(dst_tex)
    if src_tex.is_dir():
        shutil.copytree(src_tex, dst_tex)
    for rel_str in includes or []:
        rel = _safe_relpath(rel_str)
        src_file = src_usd.parent / rel
        dst_file = target_dir / rel
        if src_file.is_file() and not dst_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
    return dst_usd.name


def convert_one(
    model_entry: dict,
    sources_dir: Path,
    out_root: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    collider: str = "auto",
    orient: str = "auto",
    decomp_max_meshes: int = DECOMP_MAX_MESHES,
    decomp_max_faces: int = DECOMP_MAX_FACES,
    proxy_target_faces: int = PROXY_TARGET_FACES,
    keep_exr: bool = False,
    strict_exr: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Convert one fetched Poly Haven model into `<out_root>/Assets/Object/<namespace>/<object_id>/`."""
    object_id = model_entry["id"]
    model_id = model_entry["model_id"]
    target = Path(out_root) / "Assets" / "Object" / namespace / object_id
    meta_path = target / "metadata.json"

    if skip_existing and (target / "object.usd").is_file() and meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("runtime_ready"):
            adapter = existing.get("adapter") or {}
            col_info = adapter.get("collider") or {}
            approx = col_info.get("approximation") or []
            return {
                "id": object_id,
                "model_id": model_id,
                "status": "skipped_existing",
                "dir": str(target),
                "mass_kg": (existing.get("physics") or {}).get("mass"),
                "size_m": (existing.get("physics") or {}).get("size"),
                "collider": approx[0] if approx else None,
                "faces": (existing.get("geometry") or {}).get("faces"),
            }

    model_src_dir = Path(sources_dir) / model_id
    fetched_meta_path = model_src_dir / "polyhaven_meta.json"
    fetched_meta = json.loads(fetched_meta_path.read_text(encoding="utf-8")) if fetched_meta_path.is_file() else {}
    merged_entry = {**fetched_meta, **model_entry}
    src_usd = _find_source_usd(model_src_dir, fetched_meta)

    import numpy as np

    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt = pxr_modules()

    original = Usd.Stage.Open(str(src_usd))
    if not original:
        raise ValueError(f"{src_usd}: failed to open USD stage")
    ref_prim_path = None
    if not original.GetDefaultPrim():
        active_roots = [p for p in original.GetPseudoRoot().GetChildren() if p.IsActive()]
        if not active_roots:
            raise ValueError(f"{src_usd}: USD stage has no defaultPrim or active root prim")
        ref_prim_path = active_roots[0].GetPath()
    axis = UsdGeom.GetStageUpAxis(original)
    unit = float(UsdGeom.GetStageMetersPerUnit(original) or 1.0)

    payload_name = _copy_payload(src_usd, target, includes=fetched_meta.get("includes"))
    dst_payload = target / payload_name

    # Handle .exr -> .png conversion and asset path rewriting on the copied payload.
    exr_converted = 0
    exr_warnings: list[str] = []
    if not keep_exr:
        replacements, exr_warnings = convert_directory_exrs(target, strict=strict_exr)
        if replacements:
            for exr_file in list(target.rglob("*.exr")):
                if exr_file.name.lower() in replacements:
                    exr_file.unlink()
            exr_converted = len(replacements)
    # Always remap texture paths against actual files under `target/textures/` to fix both
    # `.exr -> .png`/`.jpg` extension mismatches and stale `/mnt/prod/...` absolute farm paths.
    tex_by_name, tex_by_stem = build_texture_remap(target)
    if tex_by_name or tex_by_stem:
        rewrite_usd_texture_paths(dst_payload, tex_by_name, tex_by_stem)

    wrapper = target / "object.usd"
    existing_layer = Sdf.Layer.Find(str(wrapper))
    if existing_layer is not None:
        existing_layer.Clear()
        stage = Usd.Stage.Open(existing_layer)
    else:
        if wrapper.exists():
            wrapper.unlink()
        stage = Usd.Stage.Open(Sdf.Layer.CreateNew(str(wrapper), args={"format": "usda"}))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    normalize = UsdGeom.Xform.Define(stage, "/root/normalize")
    asset = UsdGeom.Xform.Define(stage, "/root/normalize/asset").GetPrim()
    if ref_prim_path is not None:
        asset.GetReferences().AddReference(f"./{payload_name}", ref_prim_path)
    else:
        asset.GetReferences().AddReference(f"./{payload_name}")

    shift_op = normalize.AddTranslateOp()
    orient_op = normalize.AddTransformOp(opSuffix="pose")
    if axis != UsdGeom.Tokens.z or not math.isclose(unit, 1.0, rel_tol=1e-6):
        tf = Gf.Matrix4d(1.0)
        if axis != UsdGeom.Tokens.z:
            tf.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))
        tf = Gf.Matrix4d().SetScale(unit) * tf
        normalize.AddTransformOp(opSuffix="metric_z_up").Set(tf)

    while instances := [p for p in Usd.PrimRange(asset) if p.IsInstance()]:
        for inst in instances:
            inst.SetInstanceable(False)

    strip_apis = (
        "PhysicsRigidBodyAPI",
        "PhysxRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsArticulationRootAPI",
        "PhysxArticulationAPI",
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysxCollisionAPI",
        "PhysxConvexHullCollisionAPI",
        "PhysxConvexDecompositionCollisionAPI",
    )
    visual_meshes = []
    for prim in Usd.PrimRange(asset):
        for api in strip_apis:
            if api in prim.GetAppliedSchemas():
                prim.RemoveAppliedSchema(api)
        if prim.IsA(UsdGeom.Camera) or "Light" in prim.GetTypeName() or prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
            continue
        binding = UsdShade.MaterialBindingAPI(prim)
        if not binding.GetDirectBinding(materialPurpose="physics").GetMaterialPath().isEmpty:
            binding.UnbindDirectBinding(materialPurpose="physics")
        if (
            prim.IsA(UsdGeom.Mesh)
            and UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.guide
            and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        ):
            visual_meshes.append(prim)

    if not visual_meshes:
        raise ValueError(f"{src_usd}: no visible meshes found")

    def sample_mesh_geometry() -> tuple[np.ndarray, np.ndarray]:
        cache = UsdGeom.XformCache()
        all_pts: list[np.ndarray] = []
        all_tris: list[np.ndarray] = []
        vert_offset = 0
        for prim in visual_meshes:
            mesh = UsdGeom.Mesh(prim)
            raw_pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if raw_pts is None or len(raw_pts) == 0 or counts is None or len(counts) == 0 or indices is None:
                continue
            local = np.asarray(raw_pts, dtype=float)
            tf_mat = np.asarray(cache.GetLocalToWorldTransform(prim))
            world_pts = local @ tf_mat[:3, :3] + tf_mat[3, :3]
            tris = triangles(counts, indices)
            all_pts.append(world_pts)
            if len(tris) > 0:
                all_tris.append(tris + vert_offset)
            vert_offset += len(world_pts)
        if not all_pts:
            raise ValueError(f"{src_usd}: visible meshes have no vertices")
        pts_cat = np.concatenate(all_pts, axis=0)
        tris_cat = np.concatenate(all_tris, axis=0) if all_tris else np.empty((0, 3), dtype=np.int64)
        return pts_cat, tris_cat

    # 1. Sample initial Z-up metric geometry and compute stable upright rotation.
    pre_pts, _ = sample_mesh_geometry()
    rot_mat, orient_info = compute_pose(pre_pts, orient=orient)
    # USD uses row-vector multiplication (p @ M), so store rot_mat.T via the 16-arg constructor
    # (in pxr, Gf.Matrix4d[r] returns a Vec4d copy so mat4[r][c] = ... is a silent no-op).
    mat4 = Gf.Matrix4d(
        float(rot_mat[0, 0]),
        float(rot_mat[1, 0]),
        float(rot_mat[2, 0]),
        0.0,
        float(rot_mat[0, 1]),
        float(rot_mat[1, 1]),
        float(rot_mat[2, 1]),
        0.0,
        float(rot_mat[0, 2]),
        float(rot_mat[1, 2]),
        float(rot_mat[2, 2]),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    orient_op.Set(mat4)

    # 2. Re-sample after rotation and center bounding box at origin.
    rot_pts, _ = sample_mesh_geometry()
    lo, hi = rot_pts.min(0), rot_pts.max(0)
    center, size = (lo + hi) / 2.0, hi - lo
    shift_op.Set(Gf.Vec3d(*map(float, -center)))

    pts, tris = sample_mesh_geometry()
    if np.abs((pts.min(0) + pts.max(0)) / 2.0).max() > 1e-4:
        raise ValueError(f"{model_id}: origin is not centered on bounding box")
    total_tris = int(len(tris))

    # 3. Author collision geometry.
    col_mode = choose_collider_mode(
        len(visual_meshes),
        total_tris,
        requested=collider,
        decomp_max_meshes=decomp_max_meshes,
        decomp_max_faces=decomp_max_faces,
    )
    collider_count = 0
    collider_faces = total_tris
    collider_approx = col_mode

    if col_mode == "boundingCube":
        col_scope = UsdGeom.Scope.Define(stage, "/root/collision")
        UsdGeom.Imageable(col_scope).CreatePurposeAttr(UsdGeom.Tokens.guide)
        cube = UsdGeom.Cube.Define(stage, "/root/collision/bbox_proxy")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        UsdGeom.Imageable(cube).CreatePurposeAttr(UsdGeom.Tokens.guide)
        cube.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collider_count = 1
        collider_faces = 12
    elif col_mode == "proxy":
        col_pts, col_tris = simplify_collision_mesh(pts, tris, target_faces=proxy_target_faces)
        col_scope = UsdGeom.Scope.Define(stage, "/root/collision")
        UsdGeom.Imageable(col_scope).CreatePurposeAttr(UsdGeom.Tokens.guide)
        proxy = UsdGeom.Mesh.Define(stage, "/root/collision/proxy")
        proxy.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(col_pts, dtype=np.float32)))
        proxy.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(col_tris)))
        proxy.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(col_tris, dtype=np.int32).ravel()))
        proxy.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdGeom.Imageable(proxy).CreatePurposeAttr(UsdGeom.Tokens.guide)
        proxy.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).CreateApproximationAttr("convexDecomposition")
        collider_count = 1
        collider_faces = int(len(col_tris))
        collider_approx = "convexDecomposition"
    else:
        for prim in visual_meshes:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(col_mode)
            collider_count += 1

    # 4. Author rigid body + mass on /root.
    mass_kg, mass_info = estimate_mass(pts, tris, merged_entry)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass_kg))
    stage.GetRootLayer().Save()

    # 5. Post-save integrity checks (remove partial object.usd on failure so batch validators skip it).
    try:
        reopened = Usd.Stage.Open(str(wrapper))
        bodies = [p.GetPath().pathString for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if bodies != ["/root"]:
            raise ValueError(f"{model_id}: rigid bodies other than /root remain: {bodies}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(wrapper)))
        if missing := unresolved_dependencies(unresolved):
            raise ValueError(f"{model_id}: unresolved dependencies: {missing}")
    except Exception:
        if wrapper.exists():
            wrapper.unlink()
        raise

    geometry, active = geometry_metadata(pts, total_tris)
    authors = merged_entry.get("authors") or []
    author_str = ", ".join(authors) if authors else "Poly Haven contributors"
    url = merged_entry.get("url") or f"https://polyhaven.com/a/{model_id}"
    source_meta = {
        "upstream_dataset": "Poly Haven",
        "upstream_id": model_id,
        "upstream_name": merged_entry.get("name") or model_id,
        "authors": authors,
        "categories": merged_entry.get("categories") or [],
        "tags": merged_entry.get("tags") or [],
        "res": fetched_meta.get("res") or DEFAULT_RES,
        "license": LICENSE,
        "url": url,
        "attribution": f"{merged_entry.get('name') or model_id} by {author_str} ({LICENSE}), via Poly Haven ({url})",
        "orientation": orient_info,
        "runtime_validated": False,
    }
    metadata = {
        "uuid": object_uuid(object_id),
        "physics": {
            "mass": mass_kg,
            "friction": DEFAULT_FRICTION,
            "size": r4(size),
            "mass_estimation": mass_info,
        },
        "geometry": geometry,
        "active": active,
        "passive": {"functional": {}, "support": {}},
        "source": source_meta,
        "adapter": {
            "script": SCRIPT,
            "version": VERSION,
            "wrapper": "object.usd",
            "reference": f"./{payload_name}",
            "package_axis": {"up": str(axis), "meters_per_unit": unit},
            "offset_m": r4(-center),
            "exr_converted": exr_converted,
            "exr_kept": bool(keep_exr),
            "exr_warnings": exr_warnings,
            "collider": {
                "mode": col_mode,
                "count": collider_count,
                "approximation": [collider_approx],
                "faces": collider_faces,
                "source": "proxy_mesh" if col_mode == "proxy" else "wrapper",
            },
            "validation": None,
        },
        "runtime_ready": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    desc = build_description(merged_entry, object_id)
    (target / "description.json").write_text(json.dumps(desc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "id": object_id,
        "model_id": model_id,
        "status": "converted",
        "dir": str(target),
        "mass_kg": mass_kg,
        "size_m": r4(size),
        "collider": col_mode,
        "collider_approx": collider_approx,
        "collider_faces": collider_faces,
        "faces": total_tris,
        "exr_converted": exr_converted,
        "exr_warnings": exr_warnings,
    }


# ---------------------------------------------------------------------------
# Readiness stamping (pure stdlib)
# ---------------------------------------------------------------------------


def stamp_ready(
    out_root: Path,
    object_ids: list[str],
    check_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    note: str | None = None,
) -> dict:
    """Record `check_object.py --image-dir` settle results into `metadata.json` (`runtime_ready`)."""
    results: list[dict] = []
    passed_count = 0
    for oid in object_ids:
        target = Path(out_root) / "Assets" / "Object" / namespace / oid
        meta_path = target / "metadata.json"
        if not meta_path.is_file():
            results.append({"id": oid, "passed": False, "reason": "missing metadata.json"})
            continue
        parts = oid.strip("/").split("/")
        if len(parts) < 3:
            results.append({"id": oid, "passed": False, "reason": f"invalid object id: {oid!r}"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kind, category, index = parts[-3:]
        check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
        check = json.loads(check_file.read_text(encoding="utf-8")) if check_file.is_file() else None
        passed = bool(check and check.get("settle") == "pass")
        meta.setdefault("adapter", {})["validation"] = {
            "check_object": (
                {k: check.get(k) for k in ("settle", "tilt_deg", "z_m", "expect_z_m", "checked_at")} if check else None
            ),
            "passed": passed,
            "note": note,
        }
        meta["runtime_ready"] = passed
        meta.setdefault("source", {})["runtime_validated"] = passed
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if passed:
            passed_count += 1
        results.append(
            {
                "id": oid,
                "passed": passed,
                "settle": check.get("settle") if check else "missing",
                "tilt_deg": check.get("tilt_deg") if check else None,
            }
        )
    return {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "items": results,
    }


def _discover_converted_ids(out_root: Path, namespace: str) -> list[str]:
    base = Path(out_root) / "Assets" / "Object" / namespace / "Rigid"
    if not base.is_dir():
        return []
    return sorted(
        f"Rigid/{p.parent.parent.name}/{p.parent.name}" for p in base.glob("*/*/object.usd") if p.parent.name.isdigit()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cat = sub.add_parser("catalog", help="Query Poly Haven API and build a filtered tabletop model catalog.")
    p_cat.add_argument("--out", type=Path, required=True, help="Output catalog JSON path")
    p_cat.add_argument("--assets-json", type=Path, help="Optional cached api.polyhaven.com/assets?t=models JSON")
    p_cat.add_argument("--assets", type=Path, help="Optional Assets/ dir to skip already-imported Poly Haven models")
    p_cat.add_argument(
        "--skip-sources",
        type=Path,
        action="append",
        help="SOURCES.json file(s) whose models should be skipped",
    )
    p_cat.add_argument("--min-size", type=float, default=MIN_SIZE_M)
    p_cat.add_argument("--max-size", type=float, default=MAX_SIZE_M)
    p_cat.add_argument("--min-thickness", type=float, default=MIN_THICKNESS_M)
    p_cat.add_argument("--include-categories", nargs="*", help="Only keep models with at least one of these categories")
    p_cat.add_argument("--limit", type=int, help="Cap selected models")
    p_cat.add_argument("--write-ids", type=Path, help="Optional output file listing selected object IDs")

    p_fetch = sub.add_parser("fetch", help="Download Poly Haven 1k USD packages (.usdc + textures) with retry/MD5.")
    p_fetch.add_argument("--catalog", type=Path, required=True, help="Catalog JSON from `catalog`")
    p_fetch.add_argument("--out", type=Path, required=True, help="Directory to cache downloaded model folders")
    p_fetch.add_argument("--res", default=DEFAULT_RES, choices=RES_FALLBACK_ORDER)
    p_fetch.add_argument("--workers", type=int, default=4)
    p_fetch.add_argument("--shard-index", type=int, default=0)
    p_fetch.add_argument("--shard-total", type=int, default=1)
    p_fetch.add_argument("--no-skip-existing", action="store_true")
    p_fetch.add_argument("--report", type=Path, help="Write fetch report JSON")

    p_conv = sub.add_parser("convert", help="Convert fetched Poly Haven USD packages into RoboDojo rigid objects.")
    p_conv.add_argument("--catalog", type=Path, required=True, help="Catalog JSON from `catalog`")
    p_conv.add_argument("--sources", type=Path, required=True, help="Fetched model directory from `fetch`")
    p_conv.add_argument("--out", type=Path, required=True, help="Output root holding Assets/Object/<namespace>/")
    p_conv.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_conv.add_argument(
        "--collider",
        choices=("auto", "proxy", "convexDecomposition", "convexHull", "boundingCube"),
        default="auto",
    )
    p_conv.add_argument("--orient", choices=("auto", "keep", "lay-flat"), default="auto")
    p_conv.add_argument("--decomp-max-meshes", type=int, default=DECOMP_MAX_MESHES)
    p_conv.add_argument("--decomp-max-faces", type=int, default=DECOMP_MAX_FACES)
    p_conv.add_argument("--proxy-target-faces", type=int, default=PROXY_TARGET_FACES)
    p_conv.add_argument("--keep-exr", action="store_true", help="Pass through .exr textures without PNG conversion")
    p_conv.add_argument("--strict-exr", action="store_true", help="Fail if any .exr texture cannot be converted")
    p_conv.add_argument("--skip-existing", action="store_true")
    p_conv.add_argument("--models", nargs="*", help="Optional subset of Poly Haven model IDs to convert")
    p_conv.add_argument("--report", type=Path, help="Write conversion report JSON")

    p_ready = sub.add_parser("ready", help="Stamp check_object.py results into metadata.json (`runtime_ready`).")
    p_ready.add_argument("--out", type=Path, required=True, help="Output root holding Assets/Object/<namespace>/")
    p_ready.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_ready.add_argument("--check-dir", type=Path, required=True, help="--image-dir passed to check_object.py")
    p_ready.add_argument("--ids-file", type=Path, help="File listing object IDs (one per line)")
    p_ready.add_argument("ids", nargs="*", help="Object IDs (e.g. Rigid/ph_vase/00000)")
    p_ready.add_argument("--note", help="Validation note recorded in metadata.json")
    p_ready.add_argument("--summary", type=Path, help="Write readiness summary JSON")

    args = parser.parse_args(argv)

    if args.cmd == "catalog":
        if args.assets_json:
            assets = json.loads(args.assets_json.read_text(encoding="utf-8"))
        else:
            assets = http_get_json(f"{API}/assets?t=models")
        skip_ids = load_skip_ids(sources_paths=args.skip_sources, assets_dir=args.assets)
        cat = filter_models(
            assets,
            skip_ids=skip_ids,
            min_size=args.min_size,
            max_size=args.max_size,
            min_thickness=args.min_thickness,
            include_categories=set(args.include_categories) if args.include_categories else None,
            limit=args.limit,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.write_ids:
            args.write_ids.parent.mkdir(parents=True, exist_ok=True)
            txt = "\n".join(m["id"] for m in cat["models"])
            args.write_ids.write_text(txt + ("\n" if txt else ""), encoding="utf-8")
        print(
            f"[polyhaven_models] catalog: upstream={cat['total_upstream']} "
            f"selected={cat['selected_count']} skipped={cat['skipped_count']}",
            flush=True,
        )
        return 0

    if args.cmd == "fetch":
        cat = json.loads(args.catalog.read_text(encoding="utf-8"))
        report = fetch_catalog(
            cat,
            args.out,
            res=args.res,
            workers=args.workers,
            shard_index=args.shard_index,
            shard_total=args.shard_total,
            skip_existing=not args.no_skip_existing,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[polyhaven_models] fetch: downloaded={report['downloaded_count']} "
            f"skipped={report['skipped_count']} failed={report['failed_count']}",
            flush=True,
        )
        return 1 if report["failed_count"] else 0

    if args.cmd == "convert":
        cat = json.loads(args.catalog.read_text(encoding="utf-8"))
        wanted = set(args.models) if args.models else None
        converted, failed = [], []
        try:
            for entry in cat.get("models", []):
                mid = entry["model_id"]
                if wanted is not None and mid not in wanted:
                    continue
                if not (args.sources / mid).is_dir():
                    failed.append({"id": entry["id"], "model_id": mid, "error": "source directory missing"})
                    continue
                try:
                    res_item = convert_one(
                        entry,
                        args.sources,
                        args.out,
                        namespace=args.namespace,
                        collider=args.collider,
                        orient=args.orient,
                        decomp_max_meshes=args.decomp_max_meshes,
                        decomp_max_faces=args.decomp_max_faces,
                        proxy_target_faces=args.proxy_target_faces,
                        keep_exr=args.keep_exr,
                        strict_exr=args.strict_exr,
                        skip_existing=args.skip_existing,
                    )
                    converted.append(res_item)
                    print(
                        f"[polyhaven_models] {res_item['status']} {res_item['id']} ({mid}) "
                        f"size={res_item.get('size_m')} collider={res_item.get('collider')} "
                        f"exr_converted={res_item.get('exr_converted', 0)}",
                        flush=True,
                    )
                except Exception as exc:
                    failed.append({"id": entry["id"], "model_id": mid, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[polyhaven_models] FAILED {entry['id']} ({mid}): {exc!r}", flush=True)
        finally:
            if "APP" in globals():
                globals()["APP"].close()

        summary = {
            "script": SCRIPT,
            "version": VERSION,
            "out": str(args.out),
            "converted_count": len(converted),
            "failed_count": len(failed),
            "converted": converted,
            "failed": failed,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 1 if failed else 0

    if args.cmd == "ready":
        ids = list(args.ids)
        if args.ids_file:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            )
        if not ids:
            ids = _discover_converted_ids(args.out, args.namespace)
        summary = stamp_ready(args.out, ids, args.check_dir, namespace=args.namespace, note=args.note)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[polyhaven_models] ready: {summary['passed']}/{summary['total']} passed ({summary['failed']} failed)",
            flush=True,
        )
        return 0 if summary["total"] > 0 and summary["failed"] == 0 else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
