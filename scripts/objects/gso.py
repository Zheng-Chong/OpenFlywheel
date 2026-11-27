"""Import Google Scanned Objects (CC-BY 4.0, fuel.gazebosim.org/GoogleResearch) as RoboDojo rigid objects.

Supports two modes:
  1. Task-specific `shoe` mode (legacy RSI pool `pack_objects_into_box`, preserved for full compatibility):
     long axis on x with the heel at +x, length into SHOE_LENGTH_M, standing on z = 0, dropped when height is
     outside SHOE_HEIGHT_M (flip-flops, boots).
  2. General rigid-object mode (`--category general` / `--all`) for the remaining ~950 everyday GSO objects:
     bbox-centered origin (for `check_object.py` / RigidObject), orientation normalization (`auto|keep|lay-flat`),
     category + mesh-volume mass estimation clamped to RoboDojo's tabletop range, decimated convex-decomposition
     collision mesh, deduplication against existing RoboDojo/GSO assets, and `runtime_ready` stamping.

Pipeline subcommands (pure-Python except `convert` which needs `pxr` / `isaac-sim python`):
    catalog  List available GSO model names from Hugging Face (`HF_ENDPOINT`) or Gazebo Fuel API.
    fetch    Download GSO model zips with retry/backoff, `HF_ENDPOINT` mirror support, and sharding.
    survey   Inspect local GSO zips/dirs (no pxr), deduplicate, filter by size, and emit a deterministic manifest.
    convert  Convert `shoe` or `general` GSO models into `object.usdz` + `metadata.json` + `description.json`.
    ready    Stamp `check_object.py --image-dir` results into `metadata.json` (`runtime_ready`).

Legacy shoe usage (unchanged):
    python scripts/objects/gso.py convert --category shoe --zips gso/zips/*.zip --assets Assets --out extra_gso

General rigid-object usage:
    python scripts/objects/gso.py catalog --assets Assets --out catalog.json
    python scripts/objects/gso.py fetch --catalog catalog.json --out gso_sources --workers 4
    python scripts/objects/gso.py survey --zips-dir gso_sources --assets Assets --out manifest.json
    python scripts/objects/gso.py convert --category general --manifest manifest.json --zips-dir gso_sources \\
        --assets Assets --out gso_out
    python scripts/objects/gso.py ready --root gso_out --check-dir gso_check --manifest manifest.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manitwin  # noqa: E402

SCRIPT = "scripts/objects/gso.py"
VERSION = 2

SHOE_LENGTH_M = (0.15, 0.175)  # official Rigid/shoe 0.151-0.189 long; at 0.19 a scan jams in the box and kicks it
# off the table, the official ones are smoother
SHOE_HEIGHT_M = (0.045, 0.095)  # official 0.070-0.092: no flip-flops, no boots
FIRST_INDEX = 900
TASKS = {"shoe": ["pack_objects_into_box"]}
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "Google Scanned Objects, Google LLC (Downs et al., 2022), via fuel.gazebosim.org/GoogleResearch"

# Remote source defaults (Fuel API + Hugging Face mirror).
FUEL_API_BASE = "https://fuel.gazebosim.org/1.0/GoogleResearch"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_REPO = "jasongzy/GoogleScannedObjects"
DEFAULT_HF_REVISION = "main"
USER_AGENT = "curl/8.0"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# General rigid-object geometry / physics defaults (metres, kg).
MIN_SIZE_M = 0.02
MAX_SIZE_M = 0.65
MIN_THICKNESS_M = 0.003
CONTACT_BAND_M = 0.005
DEFAULT_FRICTION = 0.6

# Material-class effective density (kg/m^3), hollow fill factor, and plausible tabletop mass range (kg).
# Upper bounds respect RigidObject's 0.5 kg tabletop cap.
MATERIAL_MASS_SPEC: dict[str, tuple[float, float, tuple[float, float]]] = {
    # class: (effective_density_kg_m3, bbox_fill_factor, (min_kg, max_kg))
    "paper": (400.0, 0.45, (0.006, 0.25)),
    "cardboard": (220.0, 0.35, (0.02, 0.45)),
    "plastic": (450.0, 0.35, (0.01, 0.45)),
    "glass": (900.0, 0.30, (0.05, 0.45)),
    "ceramic": (1000.0, 0.30, (0.08, 0.45)),
    "wood": (600.0, 0.50, (0.02, 0.50)),
    "metal": (1800.0, 0.35, (0.02, 0.50)),
    "food": (750.0, 0.55, (0.01, 0.50)),
    "rubber": (850.0, 0.45, (0.01, 0.35)),
    "fabric": (220.0, 0.40, (0.01, 0.35)),
    "electronic": (750.0, 0.50, (0.03, 0.50)),
    "toy": (350.0, 0.40, (0.01, 0.40)),
    "default": (400.0, 0.40, (0.01, 0.50)),
}

# Ordered keyword rules mapping GSO category / object name / description tokens to MATERIAL_MASS_SPEC keys.
MATERIAL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ceramic", ("ceramic", "porcelain", "stoneware", "teapot", "teacup", "mug", "bowl", "plate", "dish", "vase")),
    ("glass", ("glass", "jar", "wine", "carafe", "pitcher")),
    (
        "metal",
        (
            "metal",
            "steel",
            "iron",
            "aluminum",
            "aluminium",
            "copper",
            "brass",
            "skillet",
            "frypan",
            "frying_pan",
            "saucepan",
            "pot",
            "pan",
            "kettle",
            "can",
            "tin",
            "hammer",
            "wrench",
            "pliers",
            "screwdriver",
            "scissors",
            "knife",
            "fork",
            "spoon",
            "spatula",
            "thermos",
            "lock",
        ),
    ),
    ("wood", ("wood", "wooden", "bamboo", "cutting_board", "block")),
    (
        "electronic",
        (
            "electronic",
            "camera",
            "mouse",
            "keyboard",
            "headphone",
            "speaker",
            "router",
            "phone",
            "tablet",
            "laptop",
            "charger",
            "adapter",
            "controller",
            "remote",
            "clock",
            "calculator",
            "toaster",
            "blender",
            "coffee_maker",
            "printer",
            "scanner",
            "hard_drive",
            "drive",
            "usb",
            "cable",
            "monitor",
        ),
    ),
    (
        "cardboard",
        (
            "box",
            "carton",
            "cereal",
            "board_game",
            "board games",
            "puzzle",
            "packaging",
            "pack",
            "tissue",
            "cracker",
            "pasta",
            "cake_mix",
            "oatmeal",
            "tea_box",
        ),
    ),
    ("paper", ("book", "notebook", "magazine", "paper", "card", "envelope", "folder", "binder")),
    (
        "fabric",
        (
            "shoe",
            "sneaker",
            "boot",
            "sandal",
            "slipper",
            "hat",
            "cap",
            "beanie",
            "bag",
            "backpack",
            "purse",
            "tote",
            "pouch",
            "pillow",
            "cushion",
            "towel",
            "cloth",
            "plush",
            "stuffed",
            "glove",
            "sock",
            "wallet",
            "case",
        ),
    ),
    ("toy", ("toy", "action_figure", "action figure", "lego", "doll", "figurine", "dinosaur", "teddy", "playset")),
    ("food", ("food", "fruit", "apple", "banana", "orange", "bread", "cookie", "snack", "chocolate", "candy")),
    ("rubber", ("rubber", "ball", "tire", "eraser", "mat", "hose", "silicone")),
    (
        "plastic",
        (
            "plastic",
            "bottle",
            "cup",
            "tumbler",
            "container",
            "bin",
            "basket",
            "tray",
            "bucket",
            "brush",
            "comb",
            "soap",
            "shampoo",
            "lotion",
            "detergent",
            "cleaner",
            "spray",
            "tape",
            "stapler",
            "pen",
            "marker",
            "crayon",
            "highlighter",
            "media_case",
            "media case",
            "dvd",
            "cd",
            "medicine",
            "vitamin",
            "supplement",
            "car_seat",
            "car seat",
        ),
    ),
)

# Canonical GSO category slug overrides (clean, short, USD-prim-safe).
GSO_CATEGORY_SLUGS: dict[str, str] = {
    "action figures": "action_figure",
    "bag": "bag",
    "board games": "board_game",
    "bottles and cans and cups": "bottle_can_cup",
    "camera": "camera",
    "car seat": "car_seat",
    "consumer goods": "consumer_goods",
    "hat": "hat",
    "headphones": "headphones",
    "keyboard": "keyboard",
    "legos": "lego",
    "media cases": "media_case",
    "mouse": "mouse",
    "shoe": "shoe",
    "stuffed toys": "stuffed_toy",
    "toys": "toy",
}


def pxr_modules():
    """Import pxr modules, falling back to headless Isaac SimulationApp when standalone usd-core is absent."""
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        globals()["APP"] = SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt


def hf_endpoint(endpoint: str | None = None) -> str:
    raw = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    return raw.rstrip("/")


def http_get(
    url: str,
    *,
    attempts: int = 6,
    timeout: float = 120.0,
    backoff: float = 2.0,
    user_agent: str = USER_AGENT,
    sleep_fn=time.sleep,
) -> tuple[bytes, dict[str, str]]:
    """GET `url` with exponential backoff on HTTP 429/5xx and transient network errors."""
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


def http_download(
    url: str,
    dest: Path,
    *,
    attempts: int = 6,
    timeout: float = 180.0,
    backoff: float = 2.0,
    user_agent: str = USER_AGENT,
    verify_zip: bool = False,
    sleep_fn=time.sleep,
) -> int:
    """Stream `url` atomically to `dest` with retry/backoff and optional zip header validation."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            written = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 18)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
            if written == 0:
                raise OSError(f"empty response from {url}")
            if verify_zip and not zipfile.is_zipfile(part):
                raise OSError(f"downloaded file is not a valid zip: {url}")
            os.replace(part, dest)
            return written
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
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
            part.unlink(missing_ok=True)
            last_exc = exc
            if attempt == attempts - 1:
                raise
            sleep_fn(backoff * (2**attempt))
    assert last_exc is not None
    raise last_exc


def list_fuel_models(*, per_page: int = 100, attempts: int = 6, max_pages: int = 200) -> list[dict]:
    """List GSO models from the Gazebo Fuel API (`fuel.gazebosim.org/1.0/GoogleResearch/models`)."""
    models: list[dict] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        url = f"{FUEL_API_BASE}/models?page={page}&per_page={per_page}"
        try:
            body, _ = http_get(url, attempts=attempts)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and page > 1:
                break
            raise
        items = json.loads(body.decode("utf-8"))
        if not items:
            break
        added = 0
        for entry in items:
            name = (entry.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            added += 1
            quoted = urllib.parse.quote(name, safe="")
            models.append(
                {
                    "name": name,
                    "backend": "fuel",
                    "url": f"{FUEL_API_BASE}/models/{quoted}/tip/{quoted}.zip",
                    "categories": [str(c) for c in (entry.get("categories") or []) if c],
                }
            )
        if added == 0:
            break
    models.sort(key=lambda m: m["name"])
    return models


def list_hf_models(
    *,
    repo: str = DEFAULT_HF_REPO,
    revision: str = DEFAULT_HF_REVISION,
    endpoint: str | None = None,
    attempts: int = 6,
) -> list[dict]:
    """List GSO model zips (or unpacked model dirs) from a Hugging Face dataset repo."""
    base = hf_endpoint(endpoint)
    url: str | None = f"{base}/api/datasets/{repo}/tree/{revision}?recursive=true&expand=false"
    files: list[tuple[str, int]] = []
    while url:
        body, headers = http_get(url, attempts=attempts)
        for entry in json.loads(body.decode("utf-8")):
            if entry.get("type") == "file":
                files.append((entry["path"], int(entry.get("size") or 0)))
        link = headers.get("link", "")
        nxt = [part for part in link.split(",") if 'rel="next"' in part]
        if nxt:
            raw_next = nxt[0].split(";")[0].strip(" <>")
            url = urllib.parse.urljoin(base + "/", raw_next)
        else:
            url = None

    models: dict[str, dict] = {}
    # Prefer .zip packages if present; otherwise group unpacked model directories by `<model>/meshes/model.obj`.
    for path, size in files:
        if path.lower().endswith(".zip"):
            name = Path(path).stem
            quoted = urllib.parse.quote(path, safe="/")
            models[name] = {
                "name": name,
                "backend": "hf",
                "hf_repo": repo,
                "hf_path": path,
                "size_bytes": size,
                "url": f"{base}/datasets/{repo}/resolve/{revision}/{quoted}",
            }
    if not models:
        obj_dirs: dict[str, list[str]] = {}
        for path, _ in files:
            parts = path.split("/")
            if len(parts) >= 3 and parts[-2] == "meshes" and parts[-1] == "model.obj":
                prefix = "/".join(parts[:-2])
                obj_dirs[prefix] = []
        for path, _ in files:
            for prefix in obj_dirs:
                if path.startswith(prefix + "/"):
                    obj_dirs[prefix].append(path)
        for prefix, rel_files in sorted(obj_dirs.items()):
            name = prefix.split("/")[-1]
            models[name] = {
                "name": name,
                "backend": "hf_dir",
                "hf_repo": repo,
                "hf_prefix": prefix,
                "files": sorted(rel_files),
            }

    return [models[k] for k in sorted(models)]


def build_catalog(
    *,
    backend: str = "auto",
    hf_repo: str = DEFAULT_HF_REPO,
    revision: str = DEFAULT_HF_REVISION,
    endpoint: str | None = None,
    assets: Path | None = None,
    exclude_imported: bool = True,
    limit: int | None = None,
) -> dict:
    """List remote GSO models and assign deterministic `global_index` values for sharded pipelines."""
    chosen_backend = backend
    models: list[dict]
    if backend == "fuel":
        models = list_fuel_models()
    elif backend == "hf":
        models = list_hf_models(repo=hf_repo, revision=revision, endpoint=endpoint)
    elif backend == "auto":
        if endpoint or os.environ.get("HF_ENDPOINT"):
            try:
                models = list_hf_models(repo=hf_repo, revision=revision, endpoint=endpoint)
                chosen_backend = "hf"
            except Exception:
                models = list_fuel_models()
                chosen_backend = "fuel"
        else:
            try:
                models = list_fuel_models()
                chosen_backend = "fuel"
            except Exception:
                models = list_hf_models(repo=hf_repo, revision=revision, endpoint=endpoint)
                chosen_backend = "hf"
    else:
        raise ValueError(f"unsupported backend: {backend}")

    imported = existing_gso_sources(assets) if (assets and exclude_imported) else {}
    entries: list[dict] = []
    skipped_imported: list[dict] = []
    for global_idx, item in enumerate(models):
        name = item["name"]
        if name in imported:
            skipped_imported.append({"name": name, "existing_id": imported[name]})
            continue
        entries.append({**item, "global_index": global_idx})
    if limit is not None:
        entries = entries[:limit]
    return {
        "script": SCRIPT,
        "version": VERSION,
        "dataset": "Google Scanned Objects",
        "license": LICENSE,
        "backend": chosen_backend,
        "total_remote": len(models),
        "skipped_imported": len(skipped_imported),
        "models": entries,
    }


def fetch_catalog(
    catalog: dict,
    out_dir: Path,
    *,
    workers: int = 4,
    shard_index: int = 0,
    shard_total: int = 1,
    skip_existing: bool = True,
    endpoint: str | None = None,
    revision: str = DEFAULT_HF_REVISION,
) -> dict:
    """Download GSO model zips (or unpacked model dirs) listed in `catalog`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for i, m in enumerate(catalog.get("models", [])) if i % shard_total == shard_index]
    base = hf_endpoint(endpoint)

    def _fetch_one(item: dict) -> tuple[str, str, str]:
        name = item["name"]
        backend = item.get("backend", "fuel")
        if backend == "hf_dir":
            model_dir = out_dir / name
            repo = item.get("hf_repo", DEFAULT_HF_REPO)
            prefix = item["hf_prefix"].rstrip("/") + "/"
            needed: list[tuple[str, str]] = []
            for rel_path in item.get("files", []):
                sub = rel_path[len(prefix) :] if rel_path.startswith(prefix) else Path(rel_path).name
                if sub == "metadata.pbtxt" or sub == "meshes/model.obj" or sub.startswith("materials/textures/"):
                    needed.append((sub, rel_path))
            if skip_existing and needed and all((model_dir / sub).is_file() for sub, _ in needed):
                return ("skipped", name, str(model_dir))
            for sub, rel_path in needed:
                target_file = model_dir / sub
                if skip_existing and target_file.is_file():
                    continue
                quoted = urllib.parse.quote(rel_path, safe="/")
                url = f"{base}/datasets/{repo}/resolve/{revision}/{quoted}"
                http_download(url, target_file, verify_zip=False)
            return ("downloaded", name, str(model_dir))

        dest = out_dir / f"{name}.zip"
        if skip_existing and dest.is_file() and zipfile.is_zipfile(dest):
            return ("skipped", name, str(dest))
        if backend == "hf" and item.get("hf_path"):
            repo = item.get("hf_repo", DEFAULT_HF_REPO)
            quoted = urllib.parse.quote(item["hf_path"], safe="/")
            url = f"{base}/datasets/{repo}/resolve/{revision}/{quoted}"
        else:
            url = item.get("url")
            if not url:
                quoted = urllib.parse.quote(name, safe="")
                url = f"{FUEL_API_BASE}/models/{quoted}/tip/{quoted}.zip"
        http_download(url, dest, verify_zip=True)
        return ("downloaded", name, str(dest))

    downloaded, skipped, failed = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_fetch_one, item): item["name"] for item in models}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                status, _, path = fut.result()
                if status == "downloaded":
                    downloaded.append({"name": name, "path": path})
                else:
                    skipped.append({"name": name, "path": path})
            except Exception as exc:
                failed.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

    downloaded.sort(key=lambda x: x["name"])
    skipped.sort(key=lambda x: x["name"])
    failed.sort(key=lambda x: x["name"])
    return {
        "script": SCRIPT,
        "version": VERSION,
        "shard": [shard_index, shard_total],
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


def _parse_obj_pure(obj_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python Wavefront OBJ parser (fallback when trimesh is unavailable)."""
    raw_v: list[tuple[float, float, float]] = []
    raw_vt: list[tuple[float, float]] = []
    vert_map: dict[tuple[int, int], int] = {}
    out_pts: list[tuple[float, float, float]] = []
    out_uv: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with obj_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "#" in line:
                line = line.split("#", 1)[0]
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                raw_v.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt" and len(parts) >= 3:
                raw_vt.append((float(parts[1]), float(parts[2])))
            elif tag == "f" and len(parts) >= 4:
                poly: list[int] = []
                nv = len(raw_v)
                nvt = len(raw_vt)
                for tok in parts[1:]:
                    comps = tok.split("/")
                    vi = int(comps[0])
                    vi = vi - 1 if vi > 0 else nv + vi
                    vti = -1
                    if len(comps) >= 2 and comps[1]:
                        raw_ti = int(comps[1])
                        vti = raw_ti - 1 if raw_ti > 0 else nvt + raw_ti
                    key = (vi, vti)
                    idx = vert_map.get(key)
                    if idx is None:
                        idx = len(out_pts)
                        vert_map[key] = idx
                        out_pts.append(raw_v[vi])
                        out_uv.append(raw_vt[vti] if 0 <= vti < nvt else (0.0, 0.0))
                    poly.append(idx)
                for k in range(1, len(poly) - 1):
                    faces.append((poly[0], poly[k], poly[k + 1]))

    pts_arr = np.asarray(out_pts, dtype=float).reshape(-1, 3)
    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    uv_arr = np.asarray(out_uv, dtype=float).reshape(-1, 2)
    return pts_arr, faces_arr, uv_arr


def _find_model_root(root: Path) -> Path:
    """Locate the directory containing `meshes/model.obj` inside an extracted zip or unpacked folder."""
    if (root / "meshes" / "model.obj").is_file():
        return root
    for candidate in sorted(root.glob("*/meshes/model.obj")):
        return candidate.parent.parent
    raise FileNotFoundError(f"meshes/model.obj not found under {root}")


def _load_from_dir(model_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes | None, str]:
    obj_path = model_root / "meshes" / "model.obj"
    try:
        import trimesh

        mesh = trimesh.load(obj_path, force="mesh", process=False)
        uv = getattr(mesh.visual, "uv", None)
        points = np.asarray(mesh.vertices, float)
        faces = np.asarray(mesh.faces, np.int64)
        uv_arr = np.zeros((len(points), 2)) if uv is None else np.asarray(uv, float)
    except ImportError:
        points, faces, uv_arr = _parse_obj_pure(obj_path)

    tex = model_root / "materials" / "textures" / "texture.png"
    if not tex.is_file():
        tex_dir = model_root / "materials" / "textures"
        if tex_dir.is_dir():
            imgs = sorted(p for p in tex_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if imgs:
                tex = imgs[0]
    meta = model_root / "metadata.pbtxt"
    return (
        points,
        faces,
        uv_arr,
        tex.read_bytes() if tex.is_file() else None,
        meta.read_text(encoding="utf-8", errors="replace") if meta.is_file() else "",
    )


def load(zip_path):
    """(points (V, 3), faces (F, 3), uv (V, 2), texture png bytes or None, metadata.pbtxt text) of a GSO zip or dir."""
    path = Path(zip_path)
    if path.is_dir():
        return _load_from_dir(_find_model_root(path))
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        return _load_from_dir(_find_model_root(Path(tmp)))


_PBTXT_TOKEN_RE = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*("([^"\\]|\\.)*"|[^\s{}]+)'
    r"|([A-Za-z_][A-Za-z0-9_]*)\s*:?\s*\{"
    r"|(\})"
)


def parse_pbtxt(pbtxt: str) -> dict:
    """Parse key fields (`name`, `description`, `categories`, `annotations`) from GSO `metadata.pbtxt`."""
    name = ""
    desc_lines = description(pbtxt)
    categories: list[str] = []
    annotations: dict[str, str] = {}

    # Strip full-line comments; token regex handles both multi-line and single-line protobuf text blocks.
    cleaned = "\n".join(line for line in pbtxt.splitlines() if not line.lstrip().startswith("#"))
    depth = 0
    block_stack: list[str] = []
    cur_key: str | None = None

    for match in _PBTXT_TOKEN_RE.finditer(cleaned):
        k, raw_v, _, block_open, block_close = match.groups()
        if block_open is not None:
            block_stack.append(block_open)
            depth += 1
            cur_key = None
            continue
        if block_close is not None:
            if block_stack:
                block_stack.pop()
            depth = max(0, depth - 1)
            cur_key = None
            continue
        assert k is not None and raw_v is not None
        v = raw_v
        if v.startswith('"') and v.endswith('"') and len(v) >= 2:
            v = v[1:-1].replace('\\"', '"')
        if depth == 0 and k == "name" and not name:
            name = v
        elif block_stack and block_stack[-1] == "categories" and k in {"first", "second"} and v:
            if v not in categories:
                categories.append(v)
        elif depth == 0 and k == "category" and v:
            if v not in categories:
                categories.append(v)
        elif block_stack and block_stack[-1] == "annotations":
            if k == "key":
                cur_key = v
            elif k == "value" and cur_key:
                annotations[cur_key] = v
                cur_key = None

    return {
        "name": name,
        "description": desc_lines,
        "categories": categories,
        "annotations": annotations,
    }


def description(pbtxt):
    """The model's description lines from metadata.pbtxt."""
    for line in pbtxt.splitlines():
        if line.startswith("description:"):
            text = line.split(":", 1)[1].strip().strip('"')
            return [t.strip() for t in text.split("\\n") if t.strip()]
    return []


def is_shoe(src: str, pbtxt: str) -> bool:
    """True when a GSO model belongs to the GSO Shoe category."""
    parsed = parse_pbtxt(pbtxt)
    if parsed["categories"]:
        return any(c.strip().lower() == "shoe" for c in parsed["categories"])
    return False


def gso_category_name(src: str, pbtxt: str) -> str:
    """Map a GSO model's metadata.pbtxt categories to a clean USD-prim-safe `gso_<slug>` category name."""
    parsed = parse_pbtxt(pbtxt)
    for cat in parsed["categories"]:
        cleaned = cat.strip().lower()
        if not cleaned or cleaned == "none":
            continue
        slug = GSO_CATEGORY_SLUGS.get(cleaned) or manitwin.prim_name(cleaned)
        slug = re.sub(r"_+", "_", slug).strip("_")
        if slug:
            return f"gso_{slug}"
    # Fallback when `categories` is empty or "None": stay deterministic under `gso_object`.
    return "gso_object"


def existing_gso_sources(assets: Path) -> dict[str, str]:
    """{gso_source_name: object_id} of every GSO object already present under `assets`."""
    found: dict[str, str] = {}
    root = Path(assets)
    if not root.exists():
        return found
    for meta_path in root.glob("Object/*/*/*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        src_info = meta.get("source") or {}
        if src_info.get("dataset") == "Google Scanned Objects" and src_info.get("object"):
            oid = "/".join(meta_path.parts[-4:-1])
            found[str(src_info["object"])] = oid
    return found


def shoe_pose(points):
    """(3x3 rotation about z, uniform scale) that puts a shoe's long axis on x with the heel (the higher end) at +x
    and its length into SHOE_LENGTH_M; None when its height is then outside SHOE_HEIGHT_M."""
    xy = points[:, :2] - points[:, :2].mean(0)
    axis = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
    rot = np.array([[axis[0], axis[1], 0.0], [-axis[1], axis[0], 0.0], [0.0, 0.0, 1.0]])  # long axis -> x
    p = points @ rot.T
    lo, hi = p.min(0), p.max(0)
    length = hi[0] - lo[0]
    ends = [p[p[:, 0] > hi[0] - 0.25 * length, 2].max(), p[p[:, 0] < lo[0] + 0.25 * length, 2].max()]
    if ends[1] > ends[0]:  # the heel is at -x: half a turn
        rot = np.diag([-1.0, -1.0, 1.0]) @ rot
    s = float(np.clip(1.0, SHOE_LENGTH_M[0] / length, SHOE_LENGTH_M[1] / length))
    return (rot, s) if SHOE_HEIGHT_M[0] <= (hi[2] - lo[2]) * s <= SHOE_HEIGHT_M[1] else None


def _yaw_align_xy(points: np.ndarray) -> np.ndarray:
    """3x3 Z-rotation aligning the principal horizontal axis to +X when elongated (no-op for isotropic XY)."""
    uniq = np.unique(np.round(points, 6), axis=0)
    if len(uniq) < 3:
        return np.eye(3)
    xy = uniq[:, :2] - uniq[:, :2].mean(0)
    cov = np.cov(xy.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return np.eye(3)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[1] <= 1e-12 or eigvals[1] <= 1.05 * max(float(eigvals[0]), 0.0):
        return np.eye(3)
    axis = eigvecs[:, -1]
    rot = np.array([[axis[0], axis[1], 0.0], [-axis[1], axis[0], 0.0], [0.0, 0.0, 1.0]])
    p = uniq @ rot.T
    # Break 180-degree sign ambiguity deterministically using X-skewness (heavier tail toward +X).
    x_centered = p[:, 0] - p[:, 0].mean()
    if float(np.mean(x_centered**3)) < 0.0:
        rot = np.diag([-1.0, -1.0, 1.0]) @ rot
    return rot


def _stability_metrics(points: np.ndarray) -> dict:
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
    # Distance of XY center of mass inside the foot bounding box (positive = inside).
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


# 6 cardinal 90-degree rotations mapping each axis (+Z, -Z, +X, -X, +Y, -Y) to +Z.
_CARDINAL_ROTATIONS: tuple[tuple[str, np.ndarray], ...] = (
    ("+z", np.eye(3)),
    ("-z", np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])),
    ("+x", np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])),
    ("-x", np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])),
    ("+y", np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
    ("-y", np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])),
)


def general_pose(points: np.ndarray, *, orient: str = "auto") -> tuple[np.ndarray, float, dict]:
    """Compute `(3x3 rotation, uniform scale=1.0, info)` for a general GSO rigid object.

    - `keep`: preserve scanned Z-up orientation; align principal XY axis to +X.
    - `lay-flat`: pick the cardinal rotation with smallest Z height (largest footprint) and align XY to +X.
    - `auto`: preserve scanned Z-up when statically stable; otherwise pick the most stable cardinal rotation.
    """
    if orient not in {"auto", "keep", "lay-flat"}:
        raise ValueError(f"unsupported orient mode: {orient}")

    base_rot = np.eye(3)
    chosen_face = "+z"
    metrics = _stability_metrics(points)

    if orient == "lay-flat":
        best_score = None
        for face, r in _CARDINAL_ROTATIONS:
            cand = points @ r.T
            m = _stability_metrics(cand)
            # Prefer shortest height, then largest foot area and positive CoM margin.
            score = (1 if m["com_margin"] >= 0 else 0, -m["height"], m["foot_area"])
            if best_score is None or score > best_score:
                best_score = score
                base_rot = r
                chosen_face = face
                metrics = m
    elif orient == "auto" and not metrics["stable"]:
        best_score = None
        for face, r in _CARDINAL_ROTATIONS:
            cand = points @ r.T
            m = _stability_metrics(cand)
            score = (1 if m["stable"] else 0, 1 if m["com_margin"] >= 0 else 0, m["foot_area"], -m["aspect"])
            if best_score is None or score > best_score:
                best_score = score
                base_rot = r
                chosen_face = face
                metrics = m

    rotated = points @ base_rot.T
    yaw_rot = _yaw_align_xy(rotated)
    total_rot = yaw_rot @ base_rot
    info = {
        "orient_mode": orient,
        "up_face": chosen_face,
        "stable_estimate": bool(metrics["stable"]),
        "aspect": round(float(metrics["aspect"]), 3) if np.isfinite(metrics["aspect"]) else None,
    }
    return total_rot, 1.0, info


def mesh_signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    """Absolute signed volume (m^3) of a triangle mesh via divergence theorem."""
    if len(faces) == 0 or len(points) == 0:
        return 0.0
    centered = points - points.mean(0)
    tris = centered[faces]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vol6 = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()
    return float(abs(vol6) / 6.0)


def infer_material_class(src: str, pbtxt: str) -> str:
    """Infer a MATERIAL_MASS_SPEC key from GSO categories, object name, and description."""
    parsed = parse_pbtxt(pbtxt)
    haystack = " ".join([src, parsed["name"], *parsed["categories"], *parsed["description"]]).lower()
    tokens = re.sub(r"[^a-z0-9]+", " ", haystack)
    for mat_class, keywords in MATERIAL_KEYWORDS:
        for kw in keywords:
            pat = kw.replace("_", " ")
            if re.search(rf"\b{re.escape(pat)}s?\b", tokens):
                return mat_class
    return "default"


def estimate_mass(points: np.ndarray, faces: np.ndarray, src: str, pbtxt: str) -> tuple[float, dict]:
    """Estimate realistic rigid-object mass (kg) from mesh volume and inferred category/material."""
    lo, hi = points.min(0), points.max(0)
    ext = hi - lo
    bbox_vol = float(np.prod(np.maximum(ext, 1e-4)))
    mat_class = infer_material_class(src, pbtxt)
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


def _simplify_collision_mesh(
    points: np.ndarray, faces: np.ndarray, target_faces: int = manitwin.COLLISION_FACES
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate collision mesh to `<= target_faces` using fast-simplification with pure-numpy voxel fallback."""
    if len(faces) <= target_faces:
        return points, faces
    try:
        import fast_simplification

        col_pts, col_faces = fast_simplification.simplify(
            points.astype(np.float32), faces, target_reduction=1.0 - target_faces / len(faces)
        )
        if len(col_faces) > 0:
            return np.asarray(col_pts, dtype=float), np.asarray(col_faces, dtype=np.int64)
    except ImportError:
        pass

    # Pure-numpy vertex clustering fallback: bin vertices on a uniform grid and drop degenerate triangles.
    lo, hi = points.min(0), points.max(0)
    span = np.maximum(hi - lo, 1e-6)
    max_span = float(span.max())
    # Estimate grid resolution from target face count (~2 faces per occupied surface cell).
    res = max(6, int(np.sqrt(max(target_faces, 100) / 2.0)))
    for _ in range(5):
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
            # Deduplicate identical triangles.
            new_faces = np.unique(new_faces, axis=0)
        if len(new_faces) <= target_faces or res <= 6:
            if len(new_faces) > 0:
                return new_pts, new_faces.astype(np.int64)
            break
        res = max(6, int(res * 0.75))
    return points, faces


def _r4(values) -> list[float]:
    return [round(float(v), 4) for v in values]


def general_metadata(
    points: np.ndarray,
    faces_total: int,
    vertices_total: int,
    src: str,
    mass_kg: float,
    mass_info: dict,
    *,
    pbtxt: str = "",
    scale: float = 1.0,
    orient_info: dict | None = None,
) -> dict:
    """Build RoboDojo `metadata.json` for a bbox-centered general GSO rigid object."""
    lo, hi = points.min(0), points.max(0)
    ext = hi - lo
    corners = [_r4((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    bbox = {"vertices": corners, "extents": _r4(ext)}
    foot = points[points[:, 2] <= lo[2] + CONTACT_BAND_M, :2]
    if len(foot):
        foot_c = (foot.min(0) + foot.max(0)) / 2
        contact_r = float(np.linalg.norm(foot - foot_c, axis=1).max())
    else:
        foot_c = np.zeros(2)
        contact_r = float(np.linalg.norm(points[:, :2], axis=1).max()) / 2
    parsed = parse_pbtxt(pbtxt)
    source: dict = {
        "dataset": "Google Scanned Objects",
        "object": src,
        "gso_categories": parsed["categories"],
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "scale": round(float(scale), 4),
        "runtime_validated": False,
    }
    if parsed["annotations"]:
        source["annotations"] = parsed["annotations"]
    if orient_info:
        source["orientation"] = orient_info
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"gso:{src}")),
        "geometry": {
            "faces": int(faces_total),
            "vertices": int(vertices_total),
            "aligned_bbox": bbox,
            "oriented_bbox": bbox,
            "radius": round(float(np.linalg.norm(points, axis=1).max()), 4),
            "bbox": _r4(ext),
            "center": _r4((lo + hi) / 2),
        },
        "active": {
            "place": {
                "up": {
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "projection_circle": {
                        "center": _r4((0.0, 0.0, lo[2], 1.0, 0.0, 0.0, 0.0)),
                        "radius": round(float(np.linalg.norm(points[:, :2], axis=1).max()), 4),
                    },
                    "contact_circle": {
                        "center": _r4((foot_c[0], foot_c[1], lo[2], 1.0, 0.0, 0.0, 0.0)),
                        "radius": round(contact_r, 4),
                    },
                }
            },
            "functional": {},
        },
        "physics": {
            "mass": round(float(mass_kg), 4),
            "size": _r4(ext),
            "friction": DEFAULT_FRICTION,
            "mass_estimation": mass_info,
        },
        "source": source,
        "adapter": {
            "script": SCRIPT,
            "version": VERSION,
            "mode": "general",
        },
        "runtime_ready": False,
    }


def write_object(target, points, faces, uv, texture, src, *, mass_kg: float | None = None):
    """Write `object.usdz` of a mesh already posed; returns legacy `manitwin.metadata(...)` dict."""
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt = pxr_modules()

    col_pts, col_faces = _simplify_collision_mesh(points, faces, manitwin.COLLISION_FACES)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Usd.Stage.CreateNew(str(Path(tmp) / "object.usda"))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(world.GetPrim())
        if mass_kg is not None:
            UsdPhysics.MassAPI.Apply(world.GetPrim()).CreateMassAttr(float(mass_kg))
        mesh = UsdGeom.Mesh.Define(stage, "/World/visual")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        material = UsdShade.Material.Define(stage, "/World/Looks/scan")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/scan/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        if texture:
            tex_name = "texture.jpg" if texture[:3] == b"\xff\xd8\xff" else "texture.png"
            (Path(tmp) / tex_name).write_bytes(texture)
            reader = UsdShade.Shader.Define(stage, "/World/Looks/scan/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            image = UsdShade.Shader.Define(stage, "/World/Looks/scan/texture")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{tex_name}")
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                image.ConnectableAPI(), "rgb"
            )
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        col = UsdGeom.Mesh.Define(stage, "/World/collision")
        col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(col_pts, np.float32)))
        col.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(col_faces)))
        col.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(col_faces).ravel().astype(np.int32)))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexDecomposition")
        col.MakeInvisible()
        stage.GetRootLayer().Save()
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        if not UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(str(Path(tmp) / "object.usda")), str(target / "object.usdz")
        ):
            raise RuntimeError(f"packaging {src} failed")
    meta = manitwin.metadata(points, int(len(faces)), int(len(points)), src)
    meta["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gso:{src}"))
    return meta


def source_name(path: Path) -> str:
    """Return the GSO model name from a `.zip` file or unpacked model directory (preserving dots in dir names)."""
    p = Path(path)
    return p.stem if p.suffix.lower() == ".zip" else p.name


def discover_sources(zips: list[Path] | None = None, zips_dir: Path | None = None) -> list[Path]:
    """Collect GSO `.zip` files and unpacked model directories from explicit paths and/or a directory."""
    found: list[Path] = []
    seen: set[str] = set()
    for p in zips or []:
        path = Path(p)
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            found.append(path)
    if zips_dir is not None:
        root = Path(zips_dir)
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_file() and child.suffix.lower() == ".zip":
                    key = str(child.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append(child)
                elif child.is_dir() and (child / "meshes" / "model.obj").is_file():
                    key = str(child.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append(child)
    found.sort(key=source_name)
    return found


def survey(
    sources: list[Path],
    assets: Path | None = None,
    *,
    catalog: dict | None = None,
    exclude_shoes: bool = True,
    min_size: float = MIN_SIZE_M,
    max_size: float = MAX_SIZE_M,
    min_thickness: float = MIN_THICKNESS_M,
    orient: str = "auto",
    dedup_extents: bool = False,
    limit: int | None = None,
    existing_sources: dict[str, str] | None = None,
    existing_max_index: dict[str, int] | None = None,
) -> dict:
    """Inspect local GSO models (pure Python, no pxr), deduplicate, and emit a deterministic conversion manifest."""
    official_faces = manitwin.robodojo_faces(assets) if (assets and Path(assets).exists()) else {}
    existing_gso = existing_gso_sources(assets) if (assets and Path(assets).exists()) else {}
    if existing_sources:
        existing_gso = {**existing_gso, **existing_sources}
    known_extents = manitwin.robodojo_objects(assets) if (dedup_extents and assets and Path(assets).exists()) else []

    global_idx_map: dict[str, int] = {}
    if catalog:
        for entry in catalog.get("models", []):
            if "name" in entry and "global_index" in entry:
                global_idx_map[str(entry["name"])] = int(entry["global_index"])

    chosen: list[dict] = []
    skipped: list[dict] = []
    seen_src: set[str] = set()
    per_cat_index: dict[str, int] = {}
    occupied_oids: set[str] = set()
    for oid in existing_gso.values():
        norm_oid = oid.strip("/")
        occupied_oids.add(norm_oid)
        parts = norm_oid.split("/")
        if len(parts) == 3 and parts[0] == "Rigid" and parts[1].startswith("gso_") and parts[2].isdigit():
            per_cat_index[parts[1]] = max(per_cat_index.get(parts[1], 0), int(parts[2]) + 1)
    existing_max_norm: dict[str, int] = {}
    if existing_max_index:
        for cat, max_idx in existing_max_index.items():
            cat_key = cat.strip("/").split("/")[-1]
            existing_max_norm[cat_key] = max(existing_max_norm.get(cat_key, -1), int(max_idx))
            per_cat_index[cat_key] = max(per_cat_index.get(cat_key, 0), int(max_idx) + 1)

    for src_path in sorted(sources, key=source_name):
        src = source_name(src_path)
        if src in seen_src:
            skipped.append({"src": src, "reason": "duplicate input"})
            continue
        seen_src.add(src)
        if src in existing_gso:
            skipped.append({"src": src, "reason": f"already imported as {existing_gso[src]}"})
            continue
        try:
            points, faces, _, _, pbtxt = load(src_path)
        except Exception as exc:
            skipped.append({"src": src, "reason": f"unreadable: {repr(exc)[:80]}"})
            continue
        if len(points) == 0 or len(faces) == 0:
            skipped.append({"src": src, "reason": "empty mesh"})
            continue
        if not np.isfinite(points).all():
            skipped.append({"src": src, "reason": "non-finite vertices"})
            continue
        if exclude_shoes and is_shoe(src, pbtxt):
            skipped.append({"src": src, "reason": "excluded shoe (use --category shoe)"})
            continue
        if (same := official_faces.get(len(faces))) is not None:
            skipped.append({"src": src, "reason": f"mesh of {same}"})
            continue

        rot, s, orient_info = general_pose(points, orient=orient)
        posed = (points @ rot.T) * s
        lo, hi = posed.min(0), posed.max(0)
        ext = hi - lo
        dims = sorted(float(v) for v in ext)
        if dims[0] < min_thickness or not (min_size <= dims[-1] <= max_size):
            skipped.append({"src": src, "reason": f"size outside [{min_size}, {max_size}] m", "extents": _r4(ext)})
            continue

        category = gso_category_name(src, pbtxt)
        if dedup_extents and (dup := manitwin.duplicate_of(category.removeprefix("gso_"), dims, known_extents)):
            skipped.append({"src": src, "reason": f"like {dup}", "extents": _r4(ext)})
            continue

        mass_kg, mass_info = estimate_mass(posed, faces, src, pbtxt)
        if src in global_idx_map:
            cand_idx = global_idx_map[src]
            cand_oid = f"Rigid/{category}/{cand_idx:05d}"
            if cand_oid in occupied_oids or (category in existing_max_norm and cand_idx <= existing_max_norm[category]):
                idx = per_cat_index.get(category, 0)
                per_cat_index[category] = idx + 1
            else:
                idx = cand_idx
                per_cat_index[category] = max(per_cat_index.get(category, 0), idx + 1)
        else:
            idx = per_cat_index.get(category, 0)
            per_cat_index[category] = idx + 1

        oid = f"Rigid/{category}/{idx:05d}"
        occupied_oids.add(oid)
        chosen.append(
            {
                "id": oid,
                "src": src,
                "path": str(src_path),
                "category": category,
                "index": idx,
                "faces": int(len(faces)),
                "vertices": int(len(points)),
                "extents_m": _r4(ext),
                "mass_kg": mass_kg,
                "material_class": mass_info["material_class"],
                "orient": orient_info,
            }
        )
        if limit is not None and len(chosen) >= limit:
            break

    return {
        "script": SCRIPT,
        "version": VERSION,
        "dataset": "Google Scanned Objects",
        "license": LICENSE,
        "orient": orient,
        "exclude_shoes": exclude_shoes,
        "chosen": chosen,
        "skipped": skipped,
    }


def _resolve_general_target(out: Path, oid: str) -> Path:
    """Resolve `<out>/Assets/Object/RoboDojo/<oid>` (avoiding double `Assets/` if `out` already ends in `Assets`)."""
    out = Path(out)
    if out.name == "Assets":
        return out / "Object" / "RoboDojo" / oid
    if out.name == "Object":
        return out / "RoboDojo" / oid
    return out / "Assets" / "Object" / "RoboDojo" / oid


def convert_general(
    sources: list[Path],
    assets: Path | None,
    out: Path,
    *,
    manifest: dict | None = None,
    zips_dir: Path | None = None,
    exclude_shoes: bool = True,
    min_size: float = MIN_SIZE_M,
    max_size: float = MAX_SIZE_M,
    min_thickness: float = MIN_THICKNESS_M,
    orient: str | None = None,
    dedup_extents: bool = False,
    skip_existing: bool = False,
    limit: int | None = None,
) -> dict:
    """Convert general GSO rigid objects into bbox-centered `object.usdz` + `metadata.json` + `description.json`."""
    effective_orient = orient or (manifest.get("orient") if manifest else None) or "auto"
    if manifest is None:
        manifest = survey(
            sources,
            assets,
            exclude_shoes=exclude_shoes,
            min_size=min_size,
            max_size=max_size,
            min_thickness=min_thickness,
            orient=effective_orient,
            dedup_extents=dedup_extents,
            limit=limit,
        )

    entries = manifest.get("chosen", [])
    if limit is not None:
        entries = entries[:limit]

    done: list[dict] = []
    skipped: list[dict] = list(manifest.get("skipped", []))
    failed: list[dict] = []
    out_path = Path(out)

    for entry in entries:
        oid = entry["id"]
        src = entry["src"]
        target = _resolve_general_target(out_path, oid)
        if skip_existing and (target / "object.usdz").is_file() and (target / "metadata.json").is_file():
            skipped.append({"src": src, "id": oid, "reason": "exists"})
            continue

        src_path = Path(entry["path"]) if entry.get("path") else None
        if (src_path is None or not src_path.exists()) and zips_dir is not None:
            cand_zip = Path(zips_dir) / f"{src}.zip"
            cand_dir = Path(zips_dir) / src
            src_path = cand_zip if cand_zip.is_file() else cand_dir
        if src_path is None or not src_path.exists():
            failed.append({"id": oid, "src": src, "error": f"source not found: {src_path}"})
            continue

        try:
            points, faces, uv, texture, pbtxt = load(src_path)
            rot, s, orient_info = general_pose(points, orient=effective_orient)
            p = (points @ rot.T) * s
            lo, hi = p.min(0), p.max(0)
            p -= (lo + hi) / 2.0  # bbox center at origin (matches check_object.py / simready.py / usdcraft.py)
            mass_kg, mass_info = estimate_mass(p, faces, src, pbtxt)
            write_object(target, p, faces, uv, texture, src, mass_kg=mass_kg)
            meta = general_metadata(
                p,
                len(faces),
                len(p),
                src,
                mass_kg,
                mass_info,
                pbtxt=pbtxt,
                scale=s,
                orient_info=orient_info,
            )
            (target / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            parsed = parse_pbtxt(pbtxt)
            desc = {
                "description": parsed["description"] or [parsed["name"] or src.replace("_", " ")],
                "name": parsed["name"] or src,
                "categories": parsed["categories"],
            }
            (target / "description.json").write_text(
                json.dumps(desc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            done.append(
                {
                    "id": oid,
                    "source": src,
                    "scale": round(s, 4),
                    "mass_kg": mass_kg,
                    "size_m": meta["physics"]["size"],
                }
            )
        except Exception as exc:
            failed.append({"id": oid, "src": src, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "script": SCRIPT,
        "version": VERSION,
        "mode": "general",
        "converted": done,
        "skipped": skipped,
        "failed": failed,
    }


def convert(category, zips, assets, out):
    """Legacy `shoe` conversion (preserved verbatim for backward compatibility) + `general` dispatch."""
    if category == "general":
        return convert_general(list(zips), assets, out)

    official = manitwin.robodojo_faces(assets)
    done, skipped = [], []
    index = FIRST_INDEX
    for zip_path in sorted(zips):
        src = Path(zip_path).stem
        try:
            points, faces, uv, texture, pbtxt = load(zip_path)
        except Exception as e:  # a broken download
            skipped.append({"src": src, "reason": f"unreadable: {repr(e)[:80]}"})
            continue
        if "Shoe" not in pbtxt and category == "shoe":
            skipped.append({"src": src, "reason": "not in GSO category Shoe"})
            continue
        if (same := official.get(len(faces))) is not None:
            skipped.append({"src": src, "reason": f"mesh of {same}"})
            continue
        pose = shoe_pose(points)
        if pose is None:
            skipped.append({"src": src, "reason": f"height outside {SHOE_HEIGHT_M} m at the official length"})
            continue
        rot, s = pose
        p = (points @ rot.T) * s
        lo, hi = p.min(0), p.max(0)
        p -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        target = Path(out) / "Object" / "RoboDojo" / "Rigid" / category / f"{index:05d}"
        meta = write_object(target, p, faces, uv, texture, src)
        meta["source"] = {"dataset": "Google Scanned Objects", "object": src, "license": LICENSE}
        meta["source"] |= {"attribution": ATTRIBUTION, "scale": round(s, 4), "tasks": TASKS[category]}
        (target / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        desc = {"description": description(pbtxt)}
        (target / "description.json").write_text(json.dumps(desc, ensure_ascii=False), encoding="utf-8")
        done.append({"id": f"Rigid/{category}/{index:05d}", "source": src, "scale": round(s, 4)})
        index += 1
    return {"converted": done, "skipped": skipped}


def stamp_ready(
    out_root: Path,
    object_id: str,
    check_dir: Path,
    *,
    note: str | None = None,
) -> tuple[bool, list[str]]:
    """Record `check_object.py` result in `metadata.json` and set `runtime_ready`."""
    root = Path(out_root)
    candidates = [
        root / "Assets/Object/RoboDojo" / object_id,
        root / "Object/RoboDojo" / object_id,
        root / object_id,
    ]
    target = next((c for c in candidates if (c / "metadata.json").is_file()), None)
    if target is None:
        return False, ["missing_metadata"]

    meta_path = target / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parts = object_id.strip("/").split("/")
    kind, category, index = parts[-3], parts[-2], parts[-1]
    check_file = Path(check_dir) / kind / category / index / "preview" / "check.json"
    if not check_file.is_file():
        meta.setdefault("adapter", {})["validation"] = {"passed": False, "why": ["not_checked"], "note": note}
        meta["runtime_ready"] = False
        meta.setdefault("source", {})["runtime_validated"] = False
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return False, ["not_checked"]

    check = json.loads(check_file.read_text(encoding="utf-8"))
    passed = check.get("settle") == "pass"
    why = [] if passed else ["settle"]
    validation = {
        "check_object": {
            k: check.get(k)
            for k in ("settle", "tilt_deg", "z_m", "expect_z_m", "speed_m_s", "checked_at")
            if k in check
        },
        "passed": passed,
        "why": why,
        "note": note,
    }
    meta.setdefault("adapter", {})["validation"] = validation
    meta["runtime_ready"] = passed
    meta.setdefault("source", {})["runtime_validated"] = passed
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return passed, why


def stamp_ready_batch(
    out_root: Path,
    object_ids: list[str],
    check_dir: Path,
    *,
    note: str | None = None,
) -> dict:
    """Stamp `runtime_ready` for a list of object IDs and return a summary report."""
    ready_ids: list[str] = []
    failed_items: list[dict] = []
    not_checked_ids: list[str] = []
    for oid in object_ids:
        passed, why = stamp_ready(out_root, oid, check_dir, note=note)
        if passed:
            ready_ids.append(oid)
        elif why == ["not_checked"]:
            not_checked_ids.append(oid)
        else:
            failed_items.append({"id": oid, "why": why})
    return {
        "script": SCRIPT,
        "version": VERSION,
        "total": len(object_ids),
        "ready_count": len(ready_ids),
        "failed_count": len(failed_items),
        "not_checked_count": len(not_checked_ids),
        "ready": ready_ids,
        "failed": failed_items,
        "not_checked": not_checked_ids,
    }


def _collect_ids(manifest: Path | None, ids_file: Path | None, explicit_ids: list[str]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    if manifest and manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in data.get("chosen", []) or data.get("converted", []):
            oid = entry.get("id")
            if oid and oid not in seen:
                seen.add(oid)
                ids.append(oid)
    if ids_file and ids_file.is_file():
        for line in ids_file.read_text(encoding="utf-8").splitlines():
            oid = line.strip()
            if oid and not oid.startswith("#") and oid not in seen:
                seen.add(oid)
                ids.append(oid)
    for oid in explicit_ids:
        if oid and oid not in seen:
            seen.add(oid)
            ids.append(oid)
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cat = sub.add_parser("catalog", help="List available GSO models from HF mirror or Gazebo Fuel API.")
    p_cat.add_argument("--backend", choices=("auto", "hf", "fuel"), default="auto")
    p_cat.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p_cat.add_argument("--revision", default=DEFAULT_HF_REVISION)
    p_cat.add_argument("--endpoint", default=None, help="Override HF_ENDPOINT")
    p_cat.add_argument("--assets", type=Path, default=None, help="Optional Assets dir to skip already-imported GSO")
    p_cat.add_argument("--include-imported", action="store_true", help="Do not skip already-imported GSO objects")
    p_cat.add_argument("--limit", type=int, default=None)
    p_cat.add_argument("--out", type=Path, required=True)

    p_fetch = sub.add_parser("fetch", help="Download GSO models listed in a catalog JSON.")
    p_fetch.add_argument("--catalog", type=Path, required=True)
    p_fetch.add_argument("--out", type=Path, required=True)
    p_fetch.add_argument("--workers", type=int, default=4)
    p_fetch.add_argument("--shard-index", type=int, default=0)
    p_fetch.add_argument("--shard-total", type=int, default=1)
    p_fetch.add_argument("--endpoint", default=None)
    p_fetch.add_argument("--revision", default=DEFAULT_HF_REVISION)
    p_fetch.add_argument("--no-skip-existing", action="store_true")
    p_fetch.add_argument("--report", type=Path, default=None)

    p_surv = sub.add_parser("survey", help="Inspect local GSO zips/dirs (no pxr) and emit a conversion manifest.")
    p_surv.add_argument("--zips", type=Path, nargs="*", default=None)
    p_surv.add_argument("--zips-dir", type=Path, default=None)
    p_surv.add_argument("--catalog", type=Path, default=None, help="Optional catalog JSON for global_index IDs")
    p_surv.add_argument("--assets", type=Path, default=None)
    p_surv.add_argument(
        "--existing-state",
        type=Path,
        default=None,
        help="Optional JSON with gso_sources / gso_max_idx to skip imported models and avoid ID collisions",
    )
    p_surv.add_argument("--include-shoes", action="store_true", help="Include GSO Shoe category in general survey")
    p_surv.add_argument("--min-size", type=float, default=MIN_SIZE_M)
    p_surv.add_argument("--max-size", type=float, default=MAX_SIZE_M)
    p_surv.add_argument("--min-thickness", type=float, default=MIN_THICKNESS_M)
    p_surv.add_argument("--orient", choices=("auto", "keep", "lay-flat"), default="auto")
    p_surv.add_argument("--dedup-extents", action="store_true")
    p_surv.add_argument("--limit", type=int, default=None)
    p_surv.add_argument("--out", type=Path, required=True)

    c = sub.add_parser("convert", help="Convert GSO models into RoboDojo rigid objects.")
    c.add_argument("--category", choices=sorted([*TASKS, "general"]), default=None)
    c.add_argument(
        "--all", dest="all_general", action="store_true", help="Alias for --category general --include-shoes"
    )
    c.add_argument("--zips", type=Path, nargs="*", default=None)
    c.add_argument("--zips-dir", type=Path, default=None)
    c.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON from `survey`")
    c.add_argument("--assets", type=Path, default=None)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--include-shoes", action="store_true", help="Include GSO Shoe category in general mode")
    c.add_argument("--min-size", type=float, default=MIN_SIZE_M)
    c.add_argument("--max-size", type=float, default=MAX_SIZE_M)
    c.add_argument("--min-thickness", type=float, default=MIN_THICKNESS_M)
    c.add_argument("--orient", choices=("auto", "keep", "lay-flat"), default=None)
    c.add_argument("--dedup-extents", action="store_true")
    c.add_argument("--skip-existing", action="store_true")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--report", type=Path, default=None)

    p_ready = sub.add_parser("ready", help="Stamp check_object.py results into metadata.json.")
    p_ready.add_argument("--root", type=Path, required=True)
    p_ready.add_argument("--check-dir", type=Path, required=True)
    p_ready.add_argument("--manifest", type=Path, default=None)
    p_ready.add_argument("--ids-file", type=Path, default=None)
    p_ready.add_argument("ids", nargs="*")
    p_ready.add_argument("--summary", type=Path, default=None)
    p_ready.add_argument("--note", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "catalog":
        result = build_catalog(
            backend=args.backend,
            hf_repo=args.hf_repo,
            revision=args.revision,
            endpoint=args.endpoint,
            assets=args.assets,
            exclude_imported=not args.include_imported,
            limit=args.limit,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"catalog ({result['backend']}): {len(result['models'])} selected "
            f"(remote={result['total_remote']}, skipped_imported={result['skipped_imported']})"
        )
        return 0

    if args.cmd == "fetch":
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        result = fetch_catalog(
            catalog,
            args.out,
            workers=args.workers,
            shard_index=args.shard_index,
            shard_total=args.shard_total,
            skip_existing=not args.no_skip_existing,
            endpoint=args.endpoint,
            revision=args.revision,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"fetched {len(result['downloaded'])}; skipped {len(result['skipped'])}; failed {len(result['failed'])}")
        return 1 if result["failed"] else 0

    if args.cmd == "survey":
        sources = discover_sources(args.zips, args.zips_dir)
        catalog = json.loads(args.catalog.read_text(encoding="utf-8")) if args.catalog else None
        if catalog and not sources and args.zips_dir:
            sources = discover_sources(zips_dir=args.zips_dir)
        if catalog:
            allowed = {m["name"] for m in catalog.get("models", [])}
            sources = [p for p in sources if source_name(p) in allowed]
        existing_state = json.loads(args.existing_state.read_text(encoding="utf-8")) if args.existing_state else {}
        result = survey(
            sources,
            args.assets,
            catalog=catalog,
            exclude_shoes=not args.include_shoes,
            min_size=args.min_size,
            max_size=args.max_size,
            min_thickness=args.min_thickness,
            orient=args.orient,
            dedup_extents=args.dedup_extents,
            limit=args.limit,
            existing_sources=existing_state.get("gso_sources"),
            existing_max_index=existing_state.get("gso_max_idx"),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"survey: {len(result['chosen'])} chosen; {len(result['skipped'])} skipped")
        return 0

    if args.cmd == "convert":
        category = "general" if args.all_general else args.category
        if not category:
            ap.error("convert requires --category or --all")
        if category == "shoe":
            if not args.zips:
                ap.error("convert --category shoe requires --zips")
            if not args.assets:
                ap.error("convert --category shoe requires --assets")
            result = convert(category, args.zips, args.assets, args.out)
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
            sources = discover_sources(args.zips, args.zips_dir)
            result = convert_general(
                sources,
                args.assets,
                args.out,
                manifest=manifest,
                zips_dir=args.zips_dir,
                exclude_shoes=not (args.include_shoes or args.all_general),
                min_size=args.min_size,
                max_size=args.max_size,
                min_thickness=args.min_thickness,
                orient=args.orient,
                dedup_extents=args.dedup_extents,
                skip_existing=args.skip_existing,
                limit=args.limit,
            )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=1), encoding="utf-8")
        reasons: dict[str, int] = {}
        for skip in result["skipped"]:
            key = skip["reason"].split(":")[0].split(" of ")[0]
            reasons[key] = reasons.get(key, 0) + 1
        extra = f"; failed {len(result.get('failed', []))}" if "failed" in result else ""
        print(f"converted {len(result['converted'])}; skipped {reasons}{extra}")
        return 1 if result.get("failed") else 0

    if args.cmd == "ready":
        ids = _collect_ids(args.manifest, args.ids_file, args.ids)
        result = stamp_ready_batch(args.root, ids, args.check_dir, note=args.note)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"ready: {result['ready_count']}/{result['total']} passed "
            f"(failed={result['failed_count']}, not_checked={result['not_checked_count']})"
        )
        return 0

    return 0


if __name__ == "__main__":
    code = main()
    if "APP" in globals():
        if code:
            os._exit(code)  # Kit eats exit codes on close
        APP.close()  # noqa: F821
    raise SystemExit(code)
