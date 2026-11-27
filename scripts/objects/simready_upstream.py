"""Index upstream NVIDIA SimReady (`nvidia/simready-assets`), diff against local library, and build manifests.

Upstream `nvidia/simready-assets` on Hugging Face holds ~2,919 packages (~518 GB). This tool queries the HF Hub tree
metadata (without downloading package blobs) or reads a saved tree JSON, groups files into SimReady packages, compares
them against the local asset store (`Assets/Object` and/or `scripts/objects/object_names.json`), and generates a
size/category-filtered manifest for selective import.

    index     fetch or parse the upstream repo tree and write an index JSON:
              python scripts/objects/simready_upstream.py index --out upstream_index.json [--tree-json tree.json]
    diff      compare upstream_index.json against Assets/Object and/or object_names.json:
              python scripts/objects/simready_upstream.py diff --index upstream_index.json \
                  --lib Assets/Object --names scripts/objects/object_names.json --out gap_report.json
    manifest  build a prioritized download manifest from the missing packages in gap_report.json:
              python scripts/objects/simready_upstream.py manifest --diff gap_report.json \
                  --categories food,kitchenware,dishware --max-size-mb 150 --limit 200 --out manifest.json
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request


def _discover_simready_ids(root):
    base = Path(root) / "RoboDojo" / "Rigid"
    if not base.is_dir():
        return []
    return [
        f"RoboDojo/Rigid/{p.name}/00000"
        for p in sorted(base.iterdir())
        if p.name.startswith("simready_") and p.is_dir() and (p / "00000").is_dir()
    ]


UPSTREAM_REPO = "nvidia/simready-assets"
UPSTREAM_URL = f"https://huggingface.co/datasets/{UPSTREAM_REPO}"
LICENSE = "CC-BY-4.0"
AUX_DIRS = {"payloads", "textures", "materials", "thumbs", "thumbnails", "sublayers", "runnables", "source_assets"}
GENERIC_PARENTS = {"", ".", "assets", "common_assets", "props", "simready", "simready_assets", "simready_usd", "usd"}


def sanitize_token(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def package_slug(pkg_dir):
    """Derive canonical `simready_<pkg>` slug from a repo-relative package directory path."""
    parts = [p for p in Path(pkg_dir).as_posix().split("/") if p]
    while len(parts) > 1 and parts[-1].lower() in GENERIC_PARENTS:
        parts.pop()
    if not parts:
        raise ValueError("empty package directory")
    leaf = sanitize_token(parts[-1])
    if "_" not in leaf and len(parts) >= 2 and parts[-2].lower() not in GENERIC_PARENTS:
        leaf = f"{sanitize_token(parts[-2])}_{leaf}"
    if not leaf.startswith("simready_"):
        leaf = f"simready_{leaf}"
    return leaf


def category_of(slug):
    """Top-level category prefix from a `simready_<pkg>` slug (e.g. `simready_kitchenware_mug_a01` -> `kitchenware`)."""
    body = slug[len("simready_") :] if slug.startswith("simready_") else slug
    return body.split("_", 1)[0] if body else "unknown"


def parse_upstream_tree(entries):
    """Group HF repo tree entries into SimReady rigid packages with file counts and byte sizes.

    Each entry is a dict with `path` (POSIX relative path), optional `size` (int bytes), and optional `type`
    (`"file"` or `"directory"`). Returns `(packages, duplicates)`.
    """
    files = []
    for entry in entries:
        etype = entry.get("type", "file")
        if etype == "directory":
            continue
        path = str(entry.get("path") or "").lstrip("/")
        if not path:
            continue
        size = int(entry.get("size") or 0)
        files.append((path, size))

    # Identify package directories: directories holding a top-level .usd/.usda/.usdz not inside AUX_DIRS.
    pkg_usds = defaultdict(list)
    for path, _ in files:
        p = Path(path)
        if p.suffix.lower() not in (".usd", ".usda", ".usdz"):
            continue
        if p.name == "object.usd":
            continue
        parts = p.parts[:-1]
        if not parts:
            continue
        if any(seg.lower() in AUX_DIRS for seg in parts):
            continue
        pkg_dir = "/".join(parts)
        pkg_usds[pkg_dir].append(p.name)

    # Sort package dirs deepest-first so nested packages (if any) claim their own files first.
    pkg_dirs = sorted(pkg_usds.keys(), key=lambda d: (-d.count("/"), d))
    pkg_sizes = Counter()
    pkg_counts = Counter()
    for path, size in files:
        # Walk up parent directories to find the owning package dir.
        parts = path.split("/")
        for depth in range(len(parts) - 1, 0, -1):
            cand = "/".join(parts[:depth])
            if cand in pkg_usds:
                pkg_sizes[cand] += size
                pkg_counts[cand] += 1
                break

    packages = []
    seen_ids = {}
    duplicates = []
    for pkg_dir in sorted(pkg_dirs):
        usds = sorted(pkg_usds[pkg_dir])
        slug = package_slug(pkg_dir)
        oid = f"RoboDojo/Rigid/{slug}/00000"
        row = {
            "id": oid,
            "slug": slug,
            "category": category_of(slug),
            "path": pkg_dir,
            "usd": usds[0],
            "usd_count": len(usds),
            "files": pkg_counts[pkg_dir],
            "size_bytes": pkg_sizes[pkg_dir],
        }
        if oid in seen_ids:
            duplicates.append({"id": oid, "first_path": seen_ids[oid], "duplicate_path": pkg_dir})
            continue
        seen_ids[oid] = pkg_dir
        packages.append(row)
    return packages, duplicates


def _next_link(header_val):
    if not header_val:
        return None
    for part in header_val.split(","):
        if 'rel="next"' in part:
            m = re.search(r"<([^>]+)>", part)
            if m:
                return m.group(1)
    return None


def fetch_hf_tree(repo_id=UPSTREAM_REPO, revision="main", token=None, endpoint=None):
    """Fetch recursive file tree metadata from Hugging Face Hub without downloading any file blobs."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import HfApi

        api = HfApi(endpoint=endpoint or os.environ.get("HF_ENDPOINT"), token=token)
        entries = []
        for item in api.list_repo_tree(repo_id=repo_id, revision=revision, repo_type="dataset", recursive=True):
            size = getattr(item, "size", 0) or 0
            etype = "file" if hasattr(item, "size") and item.size is not None else "directory"
            entries.append({"path": item.path, "size": int(size), "type": etype})
        return entries
    except ImportError:
        pass

    base = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    url = f"{base}/api/datasets/{quoted_repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
    entries = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenFlywheel-simready-upstream/1.0"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        page = None
        nxt = None
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    page = json.loads(resp.read().decode("utf-8"))
                    nxt = _next_link(resp.headers.get("Link"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 7:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0**attempt)
                    time.sleep(wait)
                    continue
                raise
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
            time.sleep(0.25)
    return entries


def normalize_object_id(oid):
    """Normalize `Rigid/simready_<pkg>/00000` or `RoboDojo/Rigid/simready_<pkg>/00000` to full `RoboDojo/...` form."""
    oid = oid.strip().strip("/")
    if oid.startswith("Rigid/simready_"):
        return f"RoboDojo/{oid}"
    return oid


def load_local_simready_ids(lib=None, names_json=None):
    """Collect locally imported SimReady object IDs from `Assets/Object` and/or `object_names.json`."""
    ids = set()
    if lib and Path(lib).is_dir():
        ids.update(_discover_simready_ids(lib))
    if names_json and Path(names_json).is_file():
        names = json.loads(Path(names_json).read_text())
        for key in names:
            if "simready_" in key:
                ids.add(normalize_object_id(key))
    return sorted(ids)


def diff_upstream(packages, local_ids):
    """Compare upstream package list against locally imported SimReady IDs."""
    local_set = {normalize_object_id(i) for i in local_ids}
    upstream_ids = {p["id"] for p in packages}
    imported = [p for p in packages if p["id"] in local_set]
    missing = [p for p in packages if p["id"] not in local_set]
    local_only = sorted(local_set - upstream_ids)

    missing_by_category = {}
    for p in missing:
        cat = p["category"]
        slot = missing_by_category.setdefault(cat, {"count": 0, "size_bytes": 0})
        slot["count"] += 1
        slot["size_bytes"] += p["size_bytes"]
    missing_by_category = dict(sorted(missing_by_category.items(), key=lambda kv: (-kv[1]["count"], kv[0])))

    return {
        "repo_id": UPSTREAM_REPO,
        "url": UPSTREAM_URL,
        "license": LICENSE,
        "upstream_total": len(packages),
        "local_total": len(local_set),
        "imported_count": len(imported),
        "missing_count": len(missing),
        "local_only_count": len(local_only),
        "upstream_total_bytes": sum(p["size_bytes"] for p in packages),
        "imported_bytes": sum(p["size_bytes"] for p in imported),
        "missing_bytes": sum(p["size_bytes"] for p in missing),
        "missing_by_category": missing_by_category,
        "local_only": local_only,
        "missing": missing,
    }


def build_download_manifest(missing, *, categories=None, max_size_mb=None, limit=None):
    """Filter and prioritize missing upstream packages into a selective HF download manifest."""
    cat_set = {c.strip().lower() for c in categories if c.strip()} if categories else None
    max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb is not None else None
    selected = []
    for p in missing:
        if cat_set is not None and p["category"].lower() not in cat_set:
            continue
        if max_bytes is not None and p["size_bytes"] > max_bytes:
            continue
        selected.append(p)
    selected.sort(key=lambda p: (p["size_bytes"], p["id"]))
    if limit is not None:
        selected = selected[:limit]
    return {
        "repo_id": UPSTREAM_REPO,
        "url": UPSTREAM_URL,
        "license": LICENSE,
        "count": len(selected),
        "total_bytes": sum(p["size_bytes"] for p in selected),
        "allow_patterns": [f"{p['path']}/**" for p in selected],
        "packages": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("index", "diff", "manifest"))
    parser.add_argument("--repo", default=UPSTREAM_REPO, help="HF dataset repo id")
    parser.add_argument("--revision", default="main", help="HF dataset revision")
    parser.add_argument("--tree-json", type=Path, help="index: read pre-saved HF tree JSON instead of querying network")
    parser.add_argument("--index", type=Path, help="diff: upstream_index.json written by `index`")
    parser.add_argument("--lib", type=Path, help="diff: local Assets/Object directory")
    parser.add_argument(
        "--names",
        type=Path,
        default=Path("scripts/objects/object_names.json"),
        help="diff: object_names.json",
    )
    parser.add_argument("--diff", type=Path, help="manifest: gap_report.json written by `diff`")
    parser.add_argument("--categories", help="manifest: comma-separated category prefixes to include")
    parser.add_argument("--max-size-mb", type=float, help="manifest: exclude packages larger than this size (MiB)")
    parser.add_argument("--limit", type=int, help="manifest: maximum number of packages to include")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "index":
        entries = json.loads(args.tree_json.read_text()) if args.tree_json else fetch_hf_tree(args.repo, args.revision)
        packages, duplicates = parse_upstream_tree(entries)
        payload = {
            "repo_id": args.repo,
            "revision": args.revision,
            "license": LICENSE,
            "package_count": len(packages),
            "duplicate_count": len(duplicates),
            "total_bytes": sum(p["size_bytes"] for p in packages),
            "duplicates": duplicates,
            "packages": packages,
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        gb = payload["total_bytes"] / (1024**3)
        print(
            f"[simready_upstream] indexed {len(packages)} packages ({gb:.2f} GiB), "
            f"{len(duplicates)} duplicates -> {args.out}"
        )
        return 0

    if args.command == "diff":
        if not args.index:
            parser.error("diff requires --index")
        idx = json.loads(args.index.read_text())
        local_ids = load_local_simready_ids(lib=args.lib, names_json=args.names)
        report = diff_upstream(idx["packages"], local_ids)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            f"[simready_upstream] upstream={report['upstream_total']} imported={report['imported_count']} "
            f"missing={report['missing_count']} ({report['missing_bytes'] / (1024**3):.2f} GiB) -> {args.out}"
        )
        return 0

    if not args.diff:
        parser.error("manifest requires --diff")
    gap = json.loads(args.diff.read_text())
    cats = args.categories.split(",") if args.categories else None
    manifest = build_download_manifest(gap["missing"], categories=cats, max_size_mb=args.max_size_mb, limit=args.limit)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[simready_upstream] manifest: {manifest['count']} packages "
        f"({manifest['total_bytes'] / (1024**3):.2f} GiB) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
