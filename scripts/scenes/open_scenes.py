"""Open-source indoor scene libraries (GRScenes, SceneSmith) for the Real2Sim platform's scene page.

The raw packages are kept off the platform machine: extracted on <asset-server>'s NAS (<scene-sources-root>/<lib>/)
with the original archives, as downloaded, in the long-term asset library (sources/scenes/<lib>/, archived to OSS); a
GPU host downloads its own copy for loading and rendering. Both copies are pinned to one Hugging Face revision and
checked against its LFS SHA-256, so they hold the same bytes.

    python scripts/scenes/open_scenes.py download grscenes /path/to/sources/grscenes   # resumable, verified
    python scripts/scenes/open_scenes.py verify /path/to/sources/grscenes
    python scripts/scenes/open_scenes.py extract grscenes /path/to/sources/grscenes   # into <src>/extracted/
    python scripts/scenes/open_scenes.py render grscenes /path/to/sources/grscenes \\
        --slot root@<cluster-host>:34142:4 --run /data/robodojo-runs/<run>                # into <src>/renders/<scene>/
    python scripts/scenes/open_scenes.py catalog grscenes /path/to/sources/grscenes <out dir>

`render` runs where the sources are (<asset-server>, no GPU): for each scene it copies only the files that scene uses
to a GPU host (`--slot user@host:port:gpu`, repeatable; one scene at a time per slot), runs render_library_scene.py
there, brings stats.json and the jpgs back and deletes the copy. `catalog` turns the renders into the platform's
scenes.json and
preview/<scene>/ (publish preview/ with `r2s publish`, commit scenes.json under real2sim/scene_libraries/<lib>/).

Downloads go through HF_ENDPOINT (default https://hf-mirror.com; huggingface.co is not reachable from our servers).
Stdlib only: runs on <asset-server>, the 5090s and the cluster login node as they are.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
# id -> the dataset repo and the files the platform uses (robots, benchmarks and the ablation subsets are left out)
LIBRARIES = {
    "grscenes": {
        "repo": "InternRobotics/GRScenes",
        "include": ["README.md", "scenes/GRScenes-100/"],
    },
    "scenesmith": {
        "repo": "nepfaff/scenesmith-example-scenes",
        "include": ["README.md", "text_prompt_annotations.csv", "House/", "Room/"],
    },
}
MANIFEST = "manifest.json"
PART = 256 << 20  # ranges fetched in parallel within one large file
UA = {"User-Agent": "robodojo-open-scenes/1"}


def get(url, headers=None, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout)


def revision(repo):
    with get(f"{ENDPOINT}/api/datasets/{repo}") as r:
        return json.load(r)["sha"]


def tree(repo, rev, path=""):
    """Every file under `path` at `rev`, following the API's pagination (a long listing is cut at ~1000 entries)."""
    url = f"{ENDPOINT}/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}?recursive=true"
    files = []
    while url:
        with get(url) as r:
            files += [x for x in json.load(r) if x["type"] == "file"]
            nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = nxt and urllib.parse.urljoin(ENDPOINT, nxt.group(1))
    return files


def plan(lib):
    spec = LIBRARIES[lib]
    rev = revision(spec["repo"])
    files = []
    for inc in spec["include"]:
        if inc.endswith("/"):
            files += tree(spec["repo"], rev, inc.rstrip("/"))
        else:
            files += [x for x in tree(spec["repo"], rev) if x["path"] == inc]
    out = [
        {"path": x["path"], "size": x["size"], "sha256": (x.get("lfs") or {}).get("oid"), "git_oid": x.get("oid")}
        for x in files
    ]
    return {"library": lib, "repo": spec["repo"], "revision": rev, "endpoint": ENDPOINT, "files": out}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_blob_oid(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def parts_file(p):
    return p.with_name(p.name + ".parts")  # ranges already written, while a download is incomplete


def ok(dest, entry):
    p = dest / entry["path"]
    if not p.is_file() or p.stat().st_size != entry["size"] or parts_file(p).exists():
        return False
    return sha256(p) == entry["sha256"] if entry["sha256"] else git_blob_oid(p) == entry["git_oid"]


class Progress:
    def __init__(self, total):
        self.total, self.done, self.t0, self.lock, self.last = total, 0, time.time(), threading.Lock(), 0

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last > 30:
                self.last = now
                rate = self.done / max(now - self.t0, 1e-3)
                print(
                    f"[{time.strftime('%H:%M:%S')}] {self.done / 2**30:.1f}/{self.total / 2**30:.1f} GiB "
                    f"{rate / 1e6:.1f} MB/s eta {(self.total - self.done) / max(rate, 1) / 60:.0f} min",
                    flush=True,
                )


class Parts:
    """The finished ranges of one file, kept in <file>.parts so a restarted download skips them."""

    def __init__(self, path, count):
        self.path, self.count, self.lock = parts_file(path), count, threading.Lock()
        self.done = set(json.loads(self.path.read_text())) if self.path.exists() else set()
        if not self.path.exists():
            self.path.write_text("[]")

    def finish(self, start):
        with self.lock:
            self.done.add(start)
            if len(self.done) == self.count:
                self.path.unlink()
            else:
                self.path.write_text(json.dumps(sorted(self.done)))


def fetch_range(url, path, start, end, prog, parts, tries=8):
    """Write bytes [start, end] of `url` into `path` at their offset; retried from where a broken read stopped
    (a whole small file is fetched again from its start)."""
    whole = start == 0 and end + 1 == path.stat().st_size
    pos = start
    for attempt in range(tries):
        try:
            pos = start if whole else pos
            with get(url, {} if whole else {"Range": f"bytes={pos}-{end}"}, timeout=120) as r, open(path, "r+b") as f:
                if r.status != (200 if whole else 206):
                    raise OSError(f"unexpected HTTP {r.status}")
                f.seek(pos)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    pos += len(chunk)
                    prog.add(len(chunk))
            if pos == end + 1:
                return parts.finish(start)
        except OSError as e:
            print(f"retry {path.name} @{pos}: {e}", flush=True)
        time.sleep(min(2**attempt, 60))
    raise OSError(f"{path}: range {start}-{end} stopped at {pos}")


def download(lib, dest, workers):
    dest.mkdir(parents=True, exist_ok=True)
    mpath = dest / MANIFEST
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else plan(lib)
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    rev = manifest["revision"]
    todo = [e for e in manifest["files"] if not ok(dest, e)]
    print(f"{lib} @ {rev[:12]}: {len(todo)}/{len(manifest['files'])} files to fetch", flush=True)
    work = []
    for e in todo:
        p = dest / e["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        # the mirror serves plain git files (README, csv) only whole; LFS files come from a CDN that takes ranges
        starts = list(range(0, e["size"], PART)) if e["sha256"] else [0] * bool(e["size"])
        if not parts_file(p).exists() and (not p.exists() or p.stat().st_size != e["size"]):
            p.unlink(missing_ok=True)
        with open(p, "ab") as f:
            f.truncate(e["size"])  # preallocate; ranges land at their offsets
        if not starts:
            continue
        parts = Parts(p, len(starts))
        url = f"{ENDPOINT}/datasets/{manifest['repo']}/resolve/{rev}/{urllib.parse.quote(e['path'])}"
        end = (lambda s: min(s + PART, e["size"]) - 1) if e["sha256"] else (lambda s: e["size"] - 1)
        work += [(url, p, s, end(s), parts) for s in starts if s not in parts.done]
    total = sum(end - start + 1 for _, _, start, end, _ in work)
    print(f"{len(work)} ranges, {total / 2**30:.1f} GiB", flush=True)
    prog = Progress(total)
    jobs = []
    with ThreadPoolExecutor(workers) as pool:
        for url, p, start, end, parts in work:
            jobs.append(pool.submit(fetch_range, url, p, start, end, prog, parts))
        for j in as_completed(jobs):
            j.result()
    return verify(dest)


def verify(dest):
    manifest = json.loads((dest / MANIFEST).read_text())
    bad = []
    with ThreadPoolExecutor(4) as pool:
        for e, good in zip(manifest["files"], pool.map(lambda e: ok(dest, e), manifest["files"])):
            if not good:
                bad.append(e["path"])
    report = {
        "checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": len(manifest["files"]),
        "bytes": sum(e["size"] for e in manifest["files"]),
        "bad": bad,
    }
    (dest / "verify.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)
    return not bad


GR_GROUPS = {"commercial_scenes": "commercial", "home_scenes": "home"}
SS_GROUPS = {"House": "house", "Room": "room"}


def run(cmd, tries=1, **kw):
    print("+", " ".join(map(str, cmd))[:300], flush=True)
    for attempt in range(tries):
        if subprocess.run(cmd, **kw).returncode == 0:
            return
        time.sleep(10 * (attempt + 1))
    raise OSError(f"failed {tries}x: {' '.join(map(str, cmd))[:300]}")


def ssh_cmd(port):
    # one shared connection per host: the 5090s' public port resets bursts of new ssh connections
    return [
        "ssh",
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPath=/tmp/open-scenes-%r@%h:%p",
        "-o",
        "ControlPersist=30m",
    ]


def extract(lib, src):
    """Unpack the verified archives into <src>/extracted/: GRScenes as <group>/ (scenes/, models/, Materials/),
    SceneSmith as <House|Room>/<scene>/. Archives already unpacked are skipped."""
    out = src / "extracted"
    out.mkdir(exist_ok=True)
    zips = src / "scenes" / "GRScenes-100"
    if lib == "grscenes":
        if not (out / "commercial_scenes" / "scenes").is_dir():
            run(["unzip", "-q", "-o", zips / "commercial_scenes.zip"], cwd=out)
        if not (out / "home_scenes" / "scenes").is_dir():
            run(["zip", "-q", "-s", "0", zips / "home_scenes.zip", "--out", out / "home_full.zip"])  # split zip
            run(["unzip", "-q", "-o", "home_full.zip"], cwd=out)
            (out / "home_full.zip").unlink()
        return True
    for group in SS_GROUPS:
        for tar in sorted((src / group).glob("scene_*.tar")):
            dest = out / group / tar.stem
            if dest.is_dir():
                continue
            tmp = dest.with_name(dest.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            run(["tar", "-xf", tar, "-C", tmp])
            tmp.rename(dest)
    return True


def scenes(lib, src):
    """[{id, category, base, files, entry}]: per scene, the paths (relative to base, directories copied whole) a GPU
    host needs to load it, and its entry USD."""
    out = []
    if lib == "grscenes":
        for group, cat in GR_GROUPS.items():
            root = src / "extracted" / group
            for d in sorted((root / "scenes").glob("*_usd")):
                meta = json.loads((d / "metadata.json").read_text())
                mats = {m.removeprefix("./") for m in meta["materials"]}
                # the .mdl files and the default grid textures every model's fallback material uses are not listed
                mdl = {f"Materials/{p.name}" for p in (root / "Materials").glob("*.mdl")}
                mdl |= {f"Materials/Textures/T_Default_Material_Grid_{c}.png" for c in "MN"}
                models = {str(Path(m).parent) for m in meta["models"]}
                files = [f"scenes/{d.name}", *sorted(models), *sorted(mats | mdl)]
                entry = f"scenes/{d.name}/start_result_interaction.usd"
                sid = d.name.removesuffix("_usd")
                out.append({"id": sid, "category": cat, "base": root, "files": files, "entry": entry})
    else:  # listed from the manifest, so scenes still being downloaded or unpacked show up (as not rendered)
        for f in json.loads((src / MANIFEST).read_text())["files"]:
            group, _, name = f["path"].partition("/")
            if group in SS_GROUPS and name.endswith(".tar"):
                sid = name.removesuffix(".tar")
                entry = f"mujoco/usd/scene_{sid}.usda"
                base, cat = src / "extracted" / group / sid, SS_GROUPS[group]
                out.append({"id": sid, "category": cat, "base": base, "files": ["mujoco/usd"], "entry": entry})
    return out


def tree_bytes(base, files):
    total = 0
    for f in files:
        p = base / f
        if p.is_dir() and not p.is_symlink():
            total += sum(q.stat().st_size for q in p.rglob("*") if q.is_file() and not q.is_symlink())
        elif p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def render_one(lib, src, sc, slot, args):
    """Copy one scene's files to the slot's host, render it there, bring stats.json and the jpgs back."""
    out = src / "renders" / sc["id"]
    host, port, gpu = slot
    ssh, rsh = [*ssh_cmd(port), host], " ".join(ssh_cmd(port))
    stage, rout = f"{args.run}/stage/{lib}/{sc['id']}", f"{args.run}/out/{lib}/{sc['id']}"
    out.mkdir(parents=True, exist_ok=True)
    info = {"entry": sc["entry"], "files": len(sc["files"]), "bytes": tree_bytes(sc["base"], sc["files"])}
    t0 = time.time()
    run([*ssh, f"mkdir -p {stage} {rout}"], tries=5)
    # -r: --files-from turns off the recursion -a implies; symlinks (Materials -> ../../Materials) stay links
    run(
        ["rsync", "-a", "-r", "--files-from=-", "-e", rsh, f"{sc['base']}/", f"{host}:{stage}/"],
        tries=5,
        input="\n".join(sc["files"]).encode(),
    )
    info["stage_s"] = round(time.time() - t0, 1)
    cmd = (
        f"cd {args.run} && OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y timeout -s KILL {args.timeout} {args.python} "
        f"render_library_scene.py {stage}/{sc['entry']} {rout} --gpu {gpu} > {rout}.log 2>&1; rm -rf {stage}"
    )
    subprocess.run([*ssh, cmd])
    run(["rsync", "-a", "-e", rsh, f"{host}:{rout}/", f"{host}:{rout}.log", f"{out}/"], tries=5)
    run([*ssh, f"rm -rf {rout} {rout}.log"], tries=5)
    stats = out / "stats.json"
    if not stats.is_file():  # Kit crashed or was killed before writing its stats
        log = (out / f"{sc['id']}.log").read_text(errors="replace") if (out / f"{sc['id']}.log").is_file() else ""
        err = [ln for ln in log.splitlines() if "[Error]" in ln or "[Fatal]" in ln or "Error" in ln[:40]][-3:]
        stats.write_text(json.dumps({"ok": False, "error": " | ".join(err) or "no stats.json (timeout?)"}) + "\n")
    info["total_s"] = round(time.time() - t0, 1)
    (out / "stage.json").write_text(json.dumps(info) + "\n")
    print(f"{sc['id']} on {host}:{gpu}: ok={json.loads(stats.read_text()).get('ok')} {info}", flush=True)


def render(lib, src, args):
    slots = [(s.rsplit(":", 2)[0], *s.rsplit(":", 2)[1:]) for s in args.slot]
    script = Path(__file__).with_name("render_library_scene.py")
    for host, port, _ in set(slots):
        run([*ssh_cmd(port), host, f"mkdir -p {args.run}"], tries=5)
        run(["rsync", "-a", "-e", " ".join(ssh_cmd(port)), script, f"{host}:{args.run}/"], tries=5)

    def done(sc):
        st = src / "renders" / sc["id"] / "stats.json"
        return st.is_file() and (json.loads(st.read_text()).get("ok") or not args.retry_failed)

    todo = [sc for sc in scenes(lib, src) if (not args.only or sc["id"] in args.only) and sc["base"].is_dir()]
    todo = [sc for sc in todo if not done(sc)]
    print(f"{lib}: {len(todo)} scenes to render on {len(slots)} slots", flush=True)
    free = list(slots)
    lock = threading.Lock()

    def job(sc):
        with lock:
            slot = free.pop()
        try:
            render_one(lib, src, sc, slot, args)
        except Exception as e:  # one scene's copy failing should not stop the batch
            print(f"{sc['id']}: {e}", flush=True)
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(len(slots)) as pool:
        list(pool.map(job, todo))
    return True


def floor_area(stats):
    return round(sum((f["hi"][0] - f["lo"][0]) * (f["hi"][1] - f["lo"][1]) for f in stats.get("floors", [])), 1)


def catalog(lib, src, out):
    """<out>/scenes.json (the scene page's list) and <out>/preview/<scene>/*.jpg from <src>/renders/."""
    report = src / "verify.json"  # written once the whole download is checked
    verify_ok = report.is_file() and not json.loads(report.read_text())["bad"]
    prompts = {}
    if lib == "scenesmith":
        with open(src / "text_prompt_annotations.csv", newline="") as f:
            prompts = {f"scene_{int(r['ID']):03d}": r["Description"] for r in csv.DictReader(f)}
    docs = []
    for sc in scenes(lib, src):
        rdir = src / "renders" / sc["id"]
        stats = json.loads((rdir / "stats.json").read_text()) if (rdir / "stats.json").is_file() else {}
        stage = json.loads((rdir / "stage.json").read_text()) if (rdir / "stage.json").is_file() else {}
        jpgs = sorted(rdir.glob("*.jpg"))
        if jpgs:
            pdir = out / "preview" / sc["id"]
            pdir.mkdir(parents=True, exist_ok=True)
            for j in jpgs:
                shutil.copy2(j, pdir / j.name)
        doc = {"id": sc["id"], "category": sc["category"], "entry": sc["entry"]}
        if stats.get("bbox"):
            doc["size_m"] = [round(h - x, 2) for x, h in zip(*stats["bbox"])]
        if lib == "grscenes":
            base = sc["base"] / "scenes" / f"{sc['id']}_usd"
            meta = json.loads((base / "metadata.json").read_text())
            art = json.loads((base / "interactive_obj_list.json").read_text()).get("articulated", [])
            kind = "商业空间" if sc["category"] == "commercial" else "住宅"
            doc |= {
                "name": f"{kind} {sc['id'][:8]}",
                "dir": f"extracted/{sc['base'].name}/scenes/{sc['id']}_usd",
                "objects": len(meta["models"]),
                "articulated": len(art),
            }
        else:
            pattern = r"room_geometry_([a-z0-9_]+?)/"
            found = [re.search(pattern, f["path"]) for f in stats.get("floors", [])]
            desc = prompts.get(sc["id"], "")
            doc |= {
                "name": desc.split(".")[0][:60] or sc["id"],
                "description": desc,
                "rooms": sorted({m.group(1).replace("_", " ") for m in found if m}),
                "dir": f"extracted/{sc['base'].parent.name}/{sc['id']}",
                "objects": stats.get("rigid_bodies"),
                "articulated": stats.get("articulation_roots"),
            }
        if stats.get("floors"):
            doc["area_m2"] = floor_area(stats)
        if stats and not stage and sc["base"].is_dir():  # rendered on the cluster, not through `render`
            stage = {"bytes": tree_bytes(sc["base"], sc["files"])}
            (rdir / "stage.json").write_text(json.dumps(stage) + "\n")
        if stage.get("bytes"):
            doc["bytes"] = stage["bytes"]
        unresolved = stats.get("unresolved_count", 0)
        if stats.get("error"):
            note = stats["error"]
        elif unresolved:
            note = f"{unresolved} 个资源路径没解析到"
        elif stats:
            note = f"{stats.get('meshes', 0)} 个网格，{stats.get('movable_joints', 0)} 个可动关节"
        else:
            note = ""
        doc["validation"] = {
            "download": "SHA-256 与 Hugging Face 版本一致" if verify_ok else "",
            "load": bool(stats.get("ok")) if stats else None,
            "load_note": note[:200],
            "render": bool(jpgs) if stats else None,
            "render_note": "" if jpgs else (stats.get("error") or "")[:200],
        }
        docs.append(doc)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenes.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1) + "\n")
    rendered = sum(bool(d["validation"]["render"]) for d in docs)
    print(f"{lib}: {len(docs)} scenes, {rendered} rendered -> {out}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("library", choices=sorted(LIBRARIES))
    d.add_argument("dest", type=Path)
    d.add_argument("--workers", type=int, default=12)
    v = sub.add_parser("verify")
    v.add_argument("dest", type=Path)
    e = sub.add_parser("extract")
    e.add_argument("library", choices=sorted(LIBRARIES))
    e.add_argument("src", type=Path)
    r = sub.add_parser("render")
    r.add_argument("library", choices=sorted(LIBRARIES))
    r.add_argument("src", type=Path)
    r.add_argument("--slot", action="append", required=True, help="user@host:port:gpu, one scene at a time each")
    r.add_argument("--run", required=True, help="scratch run dir on the GPU hosts")
    r.add_argument("--python", default="/root/miniconda3/envs/RoboDojo/bin/python")
    r.add_argument("--timeout", type=int, default=1200, help="s per scene")
    r.add_argument("--only", nargs="*", help="scene ids")
    r.add_argument("--retry-failed", action="store_true")
    c = sub.add_parser("catalog")
    c.add_argument("library", choices=sorted(LIBRARIES))
    c.add_argument("src", type=Path)
    c.add_argument("out", type=Path, help="real2sim/scene_libraries/<lib>/ (scenes.json, preview/)")
    a = ap.parse_args()
    if a.cmd == "download":
        good = download(a.library, a.dest, a.workers)
    elif a.cmd == "verify":
        good = verify(a.dest)
    elif a.cmd == "extract":
        good = extract(a.library, a.src)
    elif a.cmd == "render":
        good = render(a.library, a.src, a)
    else:
        good = catalog(a.library, a.src, a.out)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
