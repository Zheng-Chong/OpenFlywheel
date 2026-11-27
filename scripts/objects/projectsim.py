"""Import ProjectSim Articulated Storage & Appliances (projectsim/articulated-storage-and-appliances, CC-BY-4.0).

Dataset layout on Hugging Face (gated with automatic approval; pass HF_TOKEN via environment variable):
  assets/<category>/<variant_id>/
    <variant_id>.usda          OpenUSD Z-up, metres, kg, with per-link mass/inertia, colliders, joints
    textures/*.png             PBR texture maps referenced as @./textures/*.png@
    <variant_id>.glb           (not needed for Isaac Sim)
    mjcf/                      (not needed for Isaac Sim)

Categories (75 variants each, 225 total):
  - desk_with_drawers (1-6 sliding drawers)
  - oven (hinged drop-down door + turning knobs)
  - wardrobe (1-2 hinged doors)

Why a dedicated adapter on top of `articulated.write_layer`:
  1. Deduplication: some variants share identical geometry and PBR textures (differing only in variant index /
     generation timestamp); `survey` computes a normalized geometry+texture fingerprint and keeps the first variant.
  2. Joint drive & friction tuning: ProjectSim desk drawers author very light linear damping (0.002) and some joints
     need Coulomb friction / horizontal-hinge gravity-holding torque (`PhysxJointAxisAPI`) plus armature so doors and
     drawers neither open by themselves nor keep sliding after release.
  3. Tiny link mass floor: oven knobs can have authored masses ~0.006 kg (< `articulated.MIN_LINK_MASS`); clamping
     link mass to >= 0.02 kg and scaling `diagonalInertia` proportionally keeps PhysX articulation sweeps stable.
  4. Optional collision approximation override (e.g. `sdf` -> `convexDecomposition`) if needed for CPU PhysX.

Pipeline subcommands:
  fetch    Selectively download `README.md`, `AGENTS.md`, `*.usda`, and `textures/*` (skips `.glb` and `mjcf/`).
  survey   Inspect downloaded `.usda` + `textures/*`, deduplicate, and emit a deterministic selection manifest.
  convert  Write `object.usd` adapter layers, tune joints/masses, and write `metadata.json` + `description.json`.
  ready    Stamp `check_articulation.py` results into `metadata.json` (`runtime_ready`).

Security: `HF_TOKEN` is read solely from `os.environ.get("HF_TOKEN")` at runtime and is never written to files or logs.
CPU unit tests: `scripts/objects/test_projectsim.py`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import articulated  # noqa: E402

SCRIPT = "scripts/objects/projectsim.py"
VERSION = 1

REPO = "projectsim/articulated-storage-and-appliances"
DEFAULT_REVISION = "476ae4c120f36d7aefb16cd09d51c5223b67a4fa"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "curl/8.0"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "projectsim by Kaedim, huggingface.co/datasets/projectsim/articulated-storage-and-appliances"
CATEGORIES = ("desk_with_drawers", "oven", "wardrobe")

# Real-world largest dimension ranges (metres) for size_check metadata.
SIZE_M = {
    "desk_with_drawers": (0.6, 2.4),
    "oven": (0.45, 1.4),
    "wardrobe": (0.6, 2.8),
}

# Fallback total mass (kg) if a source model lacks authored link masses.
FALLBACK_MASS_KG = {
    "desk_with_drawers": 40.0,
    "oven": 55.0,
    "wardrobe": 60.0,
}

# Joint tuning floors (SI: N, N*m, N/(m/s), N*m/(deg/s), kg, kg*m^2).
MIN_PRISMATIC_DAMPING = 2.0
MIN_PRISMATIC_FRICTION = 2.0
MIN_PRISMATIC_ARMATURE = 0.01
MIN_REVOLUTE_DAMPING = 0.5
MIN_UPRIGHT_FRICTION = 0.3
MIN_REVOLUTE_ARMATURE = 0.01
MIN_KNOB_DAMPING = 0.02
MIN_KNOB_FRICTION = 0.05
MIN_KNOB_ARMATURE = 0.001
MIN_LINK_MASS_KG = articulated.MIN_LINK_MASS  # 0.02 kg

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
FETCH_MARKER = ".fetched"

# Metadata lines in .usda that vary across duplicate exports without changing geometry/physics/appearance.
_DEDUP_STRIP_RE = re.compile(
    r"^\s*(?:string\s+usd_date_generated\s*=.*|doc\s*=.*)$",
    re.MULTILINE,
)


def hf_endpoint(endpoint: str | None = None) -> str:
    raw = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    return raw.rstrip("/")


def canonical_url() -> str:
    return f"{DEFAULT_HF_ENDPOINT}/datasets/{REPO}"


def _auth_headers(user_agent: str = USER_AGENT) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def pxr_modules():
    """Import pxr modules, falling back to headless Isaac SimulationApp when standalone usd-core is absent."""
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        globals()["APP"] = SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    return Gf, Sdf, Usd, UsdGeom, UsdPhysics


def http_get(
    url: str,
    *,
    attempts: int = 6,
    timeout: float = 120.0,
    backoff: float = 2.0,
    sleep_fn=time.sleep,
) -> tuple[bytes, dict[str, str]]:
    """GET `url` with optional `HF_TOKEN` auth and exponential backoff on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers=_auth_headers())
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
    """List `(path, size)` files under `subpath` from the Hugging Face tree API with pagination."""
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


def is_needed_file(rel_path: str, categories: tuple[str, ...] = CATEGORIES) -> bool:
    """Keep top-level docs, `.usda` files, and `textures/*` under selected categories; skip `.glb` and `mjcf/`."""
    if rel_path in ("README.md", "AGENTS.md"):
        return True
    parts = rel_path.split("/")
    if len(parts) < 4 or parts[0] != "assets" or parts[1] not in categories:
        return False
    if "mjcf" in parts[3:]:
        return False
    if rel_path.endswith(".glb"):
        return False
    return rel_path.endswith(".usda") or (len(parts) >= 5 and parts[3] == "textures")


def _download_one(
    rel_path: str,
    expected_size: int,
    out_dir: Path,
    *,
    revision: str,
    endpoint: str | None,
) -> tuple[str, bool, int]:
    dest = out_dir / rel_path
    if dest.is_file() and (expected_size <= 0 or dest.stat().st_size == expected_size):
        return rel_path, False, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = hf_endpoint(endpoint)
    url = f"{base}/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(rel_path)}"
    data, _ = http_get(url)
    if expected_size > 0 and len(data) != expected_size:
        raise OSError(f"{rel_path}: got {len(data)} bytes, expected {expected_size}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return rel_path, True, len(data)


def fetch(
    out_dir: Path,
    *,
    categories: tuple[str, ...] = CATEGORIES,
    revision: str = DEFAULT_REVISION,
    endpoint: str | None = None,
    workers: int = 8,
) -> dict:
    """Selectively download `.usda` + `textures/*` + docs into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, int]] = []
    for doc in ("README.md", "AGENTS.md"):
        files.append((doc, 0))
    for cat in categories:
        for path, size in list_hf_tree(f"assets/{cat}", revision=revision, endpoint=endpoint):
            if is_needed_file(path, categories=categories):
                files.append((path, size))

    downloaded = 0
    skipped = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_download_one, p, sz, out_dir, revision=revision, endpoint=endpoint): (p, sz) for p, sz in files
        }
        for i, fut in enumerate(as_completed(futs), 1):
            rel, did_dl, nbytes = fut.result()
            total_bytes += nbytes
            if did_dl:
                downloaded += 1
            else:
                skipped += 1
            if i % 100 == 0 or i == len(files):
                print(
                    f"[projectsim fetch] {i}/{len(files)} (dl={downloaded}, skip={skipped}, "
                    f"total={total_bytes / 1e6:.1f} MB)",
                    flush=True,
                )

    upstream = {
        "dataset": REPO,
        "url": canonical_url(),
        "revision": revision,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "categories": list(categories),
        "files": len(files),
        "total_bytes": total_bytes,
    }
    articulated.write_json(out_dir / "UPSTREAM.json", upstream)
    (out_dir / FETCH_MARKER).write_text("ok\n", encoding="utf-8")
    return upstream


def variant_suffix(variant_dir_name: str) -> str:
    """Extract the 6-hex geometry hash suffix from `v01_oven_b77670` -> `b77670`."""
    parts = variant_dir_name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else variant_dir_name


def content_fingerprint(usda_path: Path) -> tuple[str, str]:
    """Return `(usda_norm_sha256, textures_sha256)` for deterministic deduplication.

    Strips generation timestamp lines (`usd_date_generated`) from `.usda` so identical exports match,
    and hashes all files under sibling `textures/` in sorted order.
    """
    raw = usda_path.read_text(encoding="utf-8", errors="replace")
    norm = _DEDUP_STRIP_RE.sub("", raw)
    usda_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()

    tex_dir = usda_path.parent / "textures"
    h = hashlib.sha256()
    if tex_dir.is_dir():
        for p in sorted(tex_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(tex_dir).as_posix()
                h.update(rel.encode("utf-8"))
                h.update(b"\0")
                h.update(hashlib.sha256(p.read_bytes()).digest())
    tex_hash = h.hexdigest()
    return usda_hash, tex_hash


def discover_models(sources_dir: Path, categories: tuple[str, ...] = CATEGORIES) -> list[tuple[str, Path]]:
    """Return sorted `(category, usda_path)` pairs under `<sources_dir>/assets/<category>/<vid>/<vid>.usda`."""
    found: list[tuple[str, Path]] = []
    assets_dir = Path(sources_dir) / "assets"
    for cat in categories:
        cat_dir = assets_dir / cat
        if not cat_dir.is_dir():
            continue
        for vdir in sorted(cat_dir.iterdir()):
            if not vdir.is_dir():
                continue
            cand = vdir / f"{vdir.name}.usda"
            if not cand.is_file():
                usdas = sorted(vdir.glob("*.usda"))
                if not usdas:
                    continue
                cand = usdas[0]
            found.append((cat, cand))
    return found


def inspect_model(usda_path: Path) -> tuple[list[float], dict]:
    """Compute bounding box size (m) and `articulated.inspect` facts for a source `.usda`."""
    import numpy as np

    _, _, Usd, UsdGeom, _ = pxr_modules()
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


def decide(category: str, size: list[float], facts: dict) -> tuple[bool, str]:
    """Decide whether a non-duplicate ProjectSim asset is valid for conversion."""
    if facts["up_axis"] != "Z":
        return False, f"up axis {facts['up_axis']}"
    if facts["world_movable"]:
        return False, f"{facts['world_movable'][0]['name']} anchored to world"
    if not facts["movable"]:
        return False, "no revolute or prismatic joint between two links"
    names = [j["name"] for j in facts["movable"]]
    if len(set(names)) < len(names):
        return False, "duplicate joint names"
    if len(facts["bases"]) != 1:
        return False, f"ambiguous base link: {facts['bases']}"
    if not facts["colliders"]:
        return False, "no enabled collision geometry"
    if min(size) < 0.005 or max(size) > 4.0:
        return False, f"degenerate size {size}"
    parts = sorted({articulated.part_of(j["body1"].rsplit("/", 1)[-1], j["type"]) for j in facts["movable"]})
    return True, f"{category}, {len(facts['movable'])} joint(s): {', '.join(parts)}"


def survey(
    sources_dir: Path,
    *,
    categories: tuple[str, ...] = CATEGORIES,
    revision: str = DEFAULT_REVISION,
    dedup_by_suffix_and_tex: bool = True,
) -> dict:
    """Survey all downloaded variants, deduplicate, and emit a deterministic selection manifest."""
    sources_dir = Path(sources_dir)
    chosen: list[dict] = []
    dropped: list[dict] = []
    counters: dict[str, int] = {}
    seen_exact: dict[tuple[str, str], str] = {}
    seen_suf_tex: dict[tuple[str, str, str], str] = {}

    for cat, usda_path in discover_models(sources_dir, categories=categories):
        rel = usda_path.parent.relative_to(sources_dir).as_posix()
        vid = usda_path.parent.name
        suf = variant_suffix(vid)
        usda_hash, tex_hash = content_fingerprint(usda_path)

        # 1. Exact (normalized usda + textures) duplicate check.
        exact_key = (usda_hash, tex_hash)
        if exact_key in seen_exact:
            dropped.append(
                {
                    "src": rel,
                    "model": usda_path.name,
                    "category": cat,
                    "variant_suffix": suf,
                    "reason": f"exact duplicate of {seen_exact[exact_key]}",
                }
            )
            continue

        # 2. Same (category, geometry suffix, texture set) duplicate check.
        suf_key = (cat, suf, tex_hash)
        if dedup_by_suffix_and_tex and suf_key in seen_suf_tex:
            dropped.append(
                {
                    "src": rel,
                    "model": usda_path.name,
                    "category": cat,
                    "variant_suffix": suf,
                    "reason": f"duplicate geometry+texture ({suf}) of {seen_suf_tex[suf_key]}",
                }
            )
            continue

        try:
            size, facts = inspect_model(usda_path)
            ok, reason = decide(cat, size, facts)
        except Exception as exc:
            dropped.append(
                {
                    "src": rel,
                    "model": usda_path.name,
                    "category": cat,
                    "variant_suffix": suf,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        entry = {
            "src": rel,
            "model": usda_path.name,
            "category": cat,
            "variant_suffix": suf,
            "usda_sha256": usda_hash[:16],
            "textures_sha256": tex_hash[:16],
            "size_m": size,
            "joint_count": len(facts["movable"]),
            "reason": reason,
        }
        if not ok:
            dropped.append(entry)
            continue

        seen_exact[exact_key] = rel
        seen_suf_tex[suf_key] = rel

        cat_slug = f"projectsim_{articulated.slug(cat)}"
        idx = counters.get(cat_slug, 0)
        counters[cat_slug] = idx + 1
        chosen.append({**entry, "id": f"Articulation/{cat_slug}/{idx:05d}"})

    return {
        "source": canonical_url(),
        "revision": revision,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "summary": {
            "total": len(chosen) + len(dropped),
            "chosen": len(chosen),
            "dropped": len(dropped),
            "by_category": counters,
        },
        "chosen": chosen,
        "dropped": dropped,
    }


def _set_drive_and_friction(
    stage,
    prim,
    token: str,
    *,
    min_damping: float,
    min_friction: float,
    min_armature: float,
) -> tuple[dict, dict]:
    """Ensure stiffness=0, target=0, and damping/friction/armature >= floors on `prim`."""
    _, Sdf, _, _, UsdPhysics = pxr_modules()

    api = UsdPhysics.DriveAPI.Apply(prim, token)
    stiff_attr = api.GetStiffnessAttr()
    damp_attr = api.GetDampingAttr()
    tgt_attr = api.GetTargetPositionAttr()
    vel_attr = api.GetTargetVelocityAttr()
    type_attr = api.GetTypeAttr()

    cur_damp = float(damp_attr.Get() or 0.0)
    new_damp = max(cur_damp, float(min_damping))
    stiff_attr.Set(0.0)
    damp_attr.Set(new_damp)
    tgt_attr.Set(0.0)
    vel_attr.Set(0.0)
    if not type_attr.HasAuthoredValue():
        type_attr.Set("force")

    prim.AddAppliedSchema(f"PhysxJointAxisAPI:{token}")
    cur_static = 0.0
    cur_dynamic = 0.0
    cur_arm = 0.0
    a_s = prim.GetAttribute(f"physxJointAxis:{token}:staticFrictionEffort")
    if a_s and a_s.HasAuthoredValue():
        cur_static = float(a_s.Get() or 0.0)
    a_d = prim.GetAttribute(f"physxJointAxis:{token}:dynamicFrictionEffort")
    if a_d and a_d.HasAuthoredValue():
        cur_dynamic = float(a_d.Get() or 0.0)
    a_a = prim.GetAttribute(f"physxJointAxis:{token}:armature")
    if a_a and a_a.HasAuthoredValue():
        cur_arm = float(a_a.Get() or 0.0)

    new_static = round(max(cur_static, float(min_friction)), 4)
    new_dynamic = round(max(cur_dynamic, float(min_friction)), 4)
    new_arm = round(max(cur_arm, float(min_armature)), 5)

    prim.CreateAttribute(f"physxJointAxis:{token}:staticFrictionEffort", Sdf.ValueTypeNames.Float).Set(new_static)
    prim.CreateAttribute(f"physxJointAxis:{token}:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(new_dynamic)
    prim.CreateAttribute(f"physxJointAxis:{token}:armature", Sdf.ValueTypeNames.Float).Set(new_arm)

    drive_meta = {
        "type": str(type_attr.Get() or "force"),
        "stiffness": 0.0,
        "damping": round(new_damp, 4),
        "target": 0.0,
    }
    fric_meta = {
        "static": new_static,
        "dynamic": new_dynamic,
        "armature": new_arm,
        "source": "projectsim_tuned",
    }
    return drive_meta, fric_meta


def tune_layer(
    layer_path: Path,
    facts: dict,
    src_facts: dict,
    *,
    replace_sdf: bool = False,
) -> dict:
    """Apply ProjectSim joint drive/friction tuning, link mass floor, and optional SDF->convexDecomposition."""
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = pxr_modules()

    stage = Usd.Stage.Open(str(layer_path))
    src_root = src_facts["root"]

    def here(path: str) -> Sdf.Path:
        return Sdf.Path("/root" + path[len(src_root) :])

    # 1. Clamp tiny link masses (e.g. 6 g oven knobs) to MIN_LINK_MASS_KG and scale diagonalInertia proportionally.
    mass_clamped: dict[str, list[float]] = {}
    link_masses: dict[str, float] = {}
    for path in src_facts["links"]:
        prim = stage.GetPrimAtPath(here(path))
        name = path.rsplit("/", 1)[-1]
        if prim and prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            m = api.GetMassAttr().Get()
            if m is not None and 0 < float(m) < MIN_LINK_MASS_KG:
                ratio = MIN_LINK_MASS_KG / float(m)
                api.GetMassAttr().Set(float(MIN_LINK_MASS_KG))
                inertia = api.GetDiagonalInertiaAttr().Get()
                if inertia is not None:
                    api.GetDiagonalInertiaAttr().Set(
                        Gf.Vec3f(
                            float(inertia[0]) * ratio,
                            float(inertia[1]) * ratio,
                            float(inertia[2]) * ratio,
                        )
                    )
                mass_clamped[name] = [round(float(m), 5), MIN_LINK_MASS_KG]
                facts["link_masses"][name] = round(MIN_LINK_MASS_KG, 4)
                link_masses[path] = MIN_LINK_MASS_KG
                continue
        link_masses[path] = float(facts["link_masses"].get(name, MIN_LINK_MASS_KG))

    # 2. Optional SDF -> convexDecomposition override (if CPU PhysX dislikes SDF colliders).
    sdf_replaced = 0
    if replace_sdf:
        root_prim = stage.GetDefaultPrim()
        for prim in Usd.PrimRange(root_prim):
            approx = prim.GetAttribute("physics:approximation")
            if approx and approx.Get() == "sdf":
                approx.Set("convexDecomposition")
                prim.RemoveAppliedSchema("PhysxSDFMeshCollisionAPI")
                prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
                sdf_replaced += 1

    # 3. Tune joints: drawers (prismatic), doors/lids (revolute), and knobs.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    by_name = {j["name"]: j for j in src_facts["movable"]}
    for entry in facts["joints"]:
        joint = by_name[entry["name"]]
        prim = stage.GetPrimAtPath(here(joint["path"]))
        if not prim:
            continue
        token = "angular" if entry["type"] == "revolute" else "linear"
        if entry["role"] == "drawer" or entry["type"] == "prismatic":
            drive_m, fric_m = _set_drive_and_friction(
                stage,
                prim,
                token,
                min_damping=MIN_PRISMATIC_DAMPING,
                min_friction=MIN_PRISMATIC_FRICTION,
                min_armature=MIN_PRISMATIC_ARMATURE,
            )
        elif entry["role"] in ("door", "lid"):
            if articulated._axis_vertical(stage, prim):
                fric = MIN_UPRIGHT_FRICTION
            else:
                b = articulated._bound(cache, stage.GetPrimAtPath(here(joint["body1"])))
                span = [float(v) for v in (b[1] - b[0])] if b else [0.2, 0.2, 0.2]
                m = link_masses.get(joint["body1"], 1.0)
                fric = round(articulated.STILL_FRICTION["level"] * m * 9.81 * max(span) / 2.0, 3)
            drive_m, fric_m = _set_drive_and_friction(
                stage,
                prim,
                token,
                min_damping=MIN_REVOLUTE_DAMPING,
                min_friction=fric,
                min_armature=MIN_REVOLUTE_ARMATURE,
            )
        else:
            # Knobs and other auxiliary joints.
            drive_m, fric_m = _set_drive_and_friction(
                stage,
                prim,
                token,
                min_damping=MIN_KNOB_DAMPING,
                min_friction=MIN_KNOB_FRICTION,
                min_armature=MIN_KNOB_ARMATURE,
            )
        entry["drive"] = drive_m
        entry["friction"] = fric_m

    stage.GetRootLayer().Save()
    return {"mass_clamped": mass_clamped, "sdf_replaced": sdf_replaced}


def size_check(category: str, size: list[float]) -> dict:
    lo, hi = SIZE_M.get(category, (0.3, 3.0))
    big = max(size)
    ok = lo <= big <= hi
    return {
        "result": "pass" if ok else "fail",
        "note": f"largest side {big:.3f} m vs {lo}-{hi} m for a real {category.replace('_', ' ')}",
    }


def convert(
    root: Path,
    manifest: dict,
    *,
    sources_subdir: str = "sources/projectsim",
    replace_sdf: bool = False,
) -> dict:
    """Convert all chosen variants in `manifest` to RoboDojo `Articulation/projectsim_<cat>/<idx>/`."""
    root = Path(root)
    _, _, Usd, _, _ = pxr_modules()
    done: list[dict] = []
    failed: list[dict] = []
    revision = manifest.get("revision", DEFAULT_REVISION)

    for entry in manifest["chosen"]:
        target = root / "Assets/Object/RoboDojo" / entry["id"]
        src_usda = root / sources_subdir / entry["src"] / entry["model"]
        cat = entry["category"]
        try:
            src_stage = Usd.Stage.Open(str(src_usda))
            src_facts = articulated.inspect(src_stage)
            authored_total = sum(src_facts["masses"].values())
            total_mass = authored_total if authored_total > 0 else FALLBACK_MASS_KG.get(cat, 40.0)

            facts = articulated.write_layer(
                src_usda,
                target,
                total_mass,
                root,
                still_parts=False,
            )
            tune_info = tune_layer(target / "object.usd", facts, src_facts, replace_sdf=replace_sdf)
            final_mass = round(sum(facts["link_masses"].values()), 4)

            source_meta = {
                "dataset": REPO,
                "url": canonical_url(),
                "revision": revision,
                "object": entry["src"],
                "license": LICENSE,
                "attribution": ATTRIBUTION,
                "rsi_pools": "allowed (CC-BY-4.0)",
                "script": SCRIPT,
                "version": VERSION,
            }
            meta = articulated.metadata(
                entry["id"],
                facts,
                final_mass,
                "authored in source USD (per-link mass/inertia, tiny links floored to 0.02 kg)",
                source_meta,
                fixed_base=True,
            )
            meta["size_check"] = size_check(cat, facts["size"])
            meta["source"]["adapter"]["projectsim_tuning"] = tune_info
            articulated.write_json(target / "metadata.json", meta)

            pretty_cat = cat.replace("_", " ")
            vid = Path(entry["src"]).name
            articulated.write_json(
                target / "description.json",
                {
                    "uuid": meta["uuid"],
                    "caption": {"name": [f"{pretty_cat} ({vid})"]},
                    "description": [
                        f"An articulated {pretty_cat} from ProjectSim Articulated Storage & Appliances ({vid})."
                    ],
                },
            )
            sha = hashlib.sha256((target / "object.usd").read_bytes()).hexdigest()
            done.append(
                {
                    "id": entry["id"],
                    "src": entry["src"],
                    "category": cat,
                    "size_m": meta["physics"]["size"],
                    "mass_kg": final_mass,
                    "joints": len(facts["joints"]),
                    "sha256": sha,
                }
            )
        except Exception as exc:
            failed.append({"id": entry["id"], "src": entry["src"], "error": f"{type(exc).__name__}: {exc}"})

    return {
        "source": canonical_url(),
        "revision": revision,
        "converted": done,
        "failed": failed,
    }


def ready(out_root: Path, report_paths: list[Path]) -> dict:
    """Stamp `check_articulation.py` report outcomes into `metadata.json` (`runtime_ready`, `asset_validation`)."""
    out_root = Path(out_root)
    results_by_id: dict[str, dict] = {}
    for rp in report_paths:
        raw = json.loads(Path(rp).read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            oid = "/".join(Path(item["asset_dir"]).parts[-3:])
            results_by_id[oid] = item

    passed = 0
    failed = 0
    missing = 0
    for oid, res in sorted(results_by_id.items()):
        meta_path = out_root / "Assets/Object/RoboDojo" / oid / "metadata.json"
        if not meta_path.is_file():
            missing += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ok = bool(res.get("pass"))
        meta["runtime_ready"] = ok
        meta["asset_validation"] = "pass" if ok else "fail"
        meta["check_summary"] = {
            "pass": ok,
            "checks": res.get("checks", {}),
            "why": [] if ok else articulated.failure_reasons(res),
            "checked_at": res.get("checked_at"),
        }
        articulated.write_json(meta_path, meta)
        if ok:
            passed += 1
        else:
            failed += 1
    return {"total": len(results_by_id), "passed": passed, "failed": failed, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Download .usda + textures/* + docs from Hugging Face")
    p_fetch.add_argument("--out", type=Path, required=True, help="Output dir (e.g. <root>/sources/projectsim)")
    p_fetch.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    p_fetch.add_argument("--revision", default=DEFAULT_REVISION)
    p_fetch.add_argument("--endpoint", default=None, help="Override HF_ENDPOINT (e.g. https://hf-mirror.com)")
    p_fetch.add_argument("--workers", type=int, default=8)

    p_survey = sub.add_parser("survey", help="Inspect downloaded .usda files, deduplicate, emit selection manifest")
    p_survey.add_argument("--sources-dir", type=Path, required=True)
    p_survey.add_argument("--out", type=Path, required=True)
    p_survey.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    p_survey.add_argument("--revision", default=DEFAULT_REVISION)
    p_survey.add_argument("--no-suffix-dedup", action="store_true", help="Only deduplicate exact usda+texture matches")

    p_convert = sub.add_parser("convert", help="Write object.usd adapter layers + metadata.json + description.json")
    p_convert.add_argument("--root", type=Path, required=True, help="Root holding Assets/ and sources/projectsim/")
    p_convert.add_argument("--manifest", type=Path, required=True)
    p_convert.add_argument("--out", type=Path, default=None, help="Optional conversion summary JSON path")
    p_convert.add_argument("--replace-sdf", action="store_true", help="Override SDF colliders with convexDecomposition")

    p_ready = sub.add_parser("ready", help="Stamp check_articulation.py reports into metadata.json")
    p_ready.add_argument("--root", type=Path, required=True)
    p_ready.add_argument("--reports", type=Path, nargs="+", required=True)

    args = parser.parse_args(argv)
    if args.command == "fetch":
        info = fetch(
            args.out,
            categories=tuple(args.categories),
            revision=args.revision,
            endpoint=args.endpoint,
            workers=args.workers,
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))
    elif args.command == "survey":
        res = survey(
            args.sources_dir,
            categories=tuple(args.categories),
            revision=args.revision,
            dedup_by_suffix_and_tex=not args.no_suffix_dedup,
        )
        articulated.write_json(args.out, res)
        print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
    elif args.command == "convert":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        res = convert(args.root, manifest, replace_sdf=args.replace_sdf)
        out = args.out or (args.root / "projectsim_conversion.json")
        articulated.write_json(out, res)
        print(f"converted={len(res['converted'])}, failed={len(res['failed'])} -> {out}")
    elif args.command == "ready":
        res = ready(args.root, args.reports)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
