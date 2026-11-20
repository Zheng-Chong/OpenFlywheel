"""Versions of the scripted experts: scripts/datagen/experts.json.

A version of a task's expert is its collection plugin (scripts/datagen/plugins/<plugin>.py) together with every
scripts/ module the plugin imports, directly or through others (scripts/g1d/scripted_demo.py, scripts/objects/...),
as they were at one commit: `sources` holds their sha256. Versions are only appended; an expert improved in code is
a new version, and the earlier ones stay collectable.

collect.py --expert vN checks that the code it runs is vN's (the sources' hashes) and refuses otherwise; without
--expert it takes the task's newest version and records which source files differ from it (`drift`). To collect
with an older version, write its sources into the run dir first (`materialize`); the rest of the pipeline stays
the run dir's.

Usage (repo root):
    python scripts/datagen/experts.py list [--task TASK]
    python scripts/datagen/experts.py status            # which tasks' code differs from their newest version
    python scripts/datagen/experts.py add --task TASK [--plugin PLUGIN] [--commit SHA] --summary "what changed"
    python scripts/datagen/experts.py materialize --task TASK --version vN --root RUN_DIR
"""

import argparse
import ast
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts" / "datagen" / "experts.json"
PLUGINS = "scripts.datagen.plugins"


class ExpertError(Exception):
    pass


def module_path(module):
    """scripts.g1d.scripted_demo -> scripts/g1d/scripted_demo.py"""
    return "/".join(module.split(".")) + ".py"


def imported_modules(source, module):
    """The scripts.* modules a module's source imports (absolute and relative imports, `from pkg import mod`)."""
    package = module.rpartition(".")[0]
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = ".".join(filter(None, [parent, base]))
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__" and node.args:
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.add(node.args[0].value)  # scripted_demo.DEMOS["pick_bucket"]
    return {m for m in out if m.startswith("scripts.")}


def reader(commit=None, root=ROOT):
    """path -> source text or None, from the working tree or from a commit."""
    if commit is None:
        return lambda path: (root / path).read_text(encoding="utf-8") if (root / path).is_file() else None

    def read(path):
        try:
            return subprocess.check_output(
                ["git", "show", f"{commit}:{path}"], cwd=root, text=True, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return None

    return read


def source_files(plugin, read):
    """{path: text} of the plugin module and every scripts.* module it reaches through imports."""
    todo, files = [f"{PLUGINS}.{plugin}"], {}
    while todo:
        module = todo.pop()
        path = module_path(module)
        if path in files:
            continue
        text = read(path)
        if text is None:
            continue  # a package or a name imported from a module, not a module file
        files[path] = text
        todo += sorted(imported_modules(text, module))
    if module_path(f"{PLUGINS}.{plugin}") not in files:
        raise ExpertError(f"no plugin {plugin}")
    return files


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(plugin, read):
    return {path: digest(text) for path, text in sorted(source_files(plugin, read).items())}


def constant(plugin, name, read):
    """A module-level string constant of the plugin (ACCEPTANCE), following `from module import NAME`."""
    module = f"{PLUGINS}.{plugin}"
    for _ in range(5):
        tree = ast.parse(read(module_path(module)) or "")
        nxt = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                if isinstance(node.value, ast.Constant):
                    return node.value.value
            if isinstance(node, ast.ImportFrom) and any(a.name == name for a in node.names):
                nxt = node.module
        if not nxt:
            return None
        module = nxt
    return None


def load(path=MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {"experts": {}}


def versions(task, manifest=None):
    return (manifest or load()).get("experts", {}).get(task, [])


def resolve(task, version=None, plugin=None, manifest=None):
    """The version entry collect.py runs: `version` of the task, or its newest version (of `plugin` when given).
    None when the task has no registered version and none was asked for."""
    entries = versions(task, manifest)
    if version:
        found = [e for e in entries if e["version"] == version]
        if not found:
            known = ", ".join(e["version"] for e in entries) or "none"
            raise ExpertError(f"{task} has no expert {version} (registered: {known})")
        entry = found[0]
        if plugin and entry["plugin"] != plugin:
            raise ExpertError(f"{task} {version} is plugin {entry['plugin']}, not {plugin}")
        return entry
    entries = [e for e in entries if e["plugin"] == (plugin or e["plugin"]) and e.get("sources")]
    return entries[-1] if entries else None


def drift(entry, read=None):
    """Source files whose code differs from the version's: changed, added (imported now) or gone."""
    now = fingerprint(entry["plugin"], read or reader())
    then = entry.get("sources") or {}
    return sorted(p for p in set(now) | set(then) if now.get(p) != then.get(p))


def record(task, version=None, plugin=None, manifest=None, read=None):
    """What collect.py writes into each episode's quality sidecar as `expert`, after checking the code."""
    entry = resolve(task, version, plugin, manifest)
    if entry is None:
        return None
    if not entry.get("sources"):
        raise ExpertError(f"{task} {entry['version']} has no recorded sources (its commit is not available)")
    changed = drift(entry, read)
    if version and changed:
        raise ExpertError(
            f"the code is not {task} {version}: {', '.join(changed)} differ; write its sources into the run dir "
            f"(experts.py materialize --task {task} --version {version} --root <run dir>) or register a new version"
        )
    return dict(task=task, version=entry["version"], plugin=entry["plugin"], commit=entry["commit"], drift=changed)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def add(manifest, task, plugin, commit, summary, today=None, read=None):
    """Append the next version of the task: the plugin's sources at `commit` (a full sha)."""
    entries = manifest.setdefault("experts", {}).setdefault(task, [])
    read = read or reader(commit)
    sources = fingerprint(plugin, read)
    if entries and entries[-1]["plugin"] == plugin and entries[-1].get("sources") == sources:
        raise ExpertError(f"{task} {entries[-1]['version']} already has this code")
    entry = dict(
        version=f"v{len(entries) + 1}",
        plugin=plugin,
        acceptance=constant(plugin, "ACCEPTANCE", read),
        commit=commit,
        created=today or date.today().isoformat(),
        summary=summary,
        sources=sources,
    )
    entries.append(entry)
    return entry


def materialize(entry, root):
    """Write the version's source files, as at its commit, under `root` (a run dir); returns the paths written."""
    read, root = reader(entry["commit"]), Path(root)
    files = source_files(entry["plugin"], read)
    if {p: digest(t) for p, t in files.items()} != entry["sources"]:
        raise ExpertError(f"{entry['commit']} does not hold the recorded sources of {entry['version']}")
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return sorted(files)


def save(manifest, path=MANIFEST):
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--task")
    sub.add_parser("status")
    p = sub.add_parser("add")
    p.add_argument("--task", required=True)
    p.add_argument("--plugin", help="default: the task's plugin")
    p.add_argument("--commit", default="HEAD", help="a commit whose code this version is (default HEAD, clean)")
    p.add_argument("--summary", required=True, help="what this version changes, for people choosing one")
    p = sub.add_parser("materialize")
    p.add_argument("--task", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--root", type=Path, required=True, help="the run dir (never this checkout)")
    args = parser.parse_args()
    manifest = load()
    try:
        if args.cmd == "list":
            for task, entries in sorted(manifest["experts"].items()):
                if args.task in (None, task):
                    for e in entries:
                        print(f"{task} {e['version']} {e['plugin']} {e['acceptance']} {e['commit'][:9]} {e['summary']}")
        elif args.cmd == "status":
            for task in sorted(manifest["experts"]):
                entry = resolve(task, manifest=manifest)
                changed = drift(entry) if entry else ["(no version with sources)"]
                print(
                    f"{task} {entry['version'] if entry else '-'}: {'same code' if not changed else ', '.join(changed)}"
                )
        elif args.cmd == "add":
            if args.commit == "HEAD" and git("status", "--porcelain", "--untracked-files=no"):
                raise ExpertError("commit the expert first: a version is code at a commit")
            entry = add(manifest, args.task, args.plugin or args.task, git("rev-parse", args.commit), args.summary)
            save(manifest)
            print(f"{args.task} {entry['version']}: {len(entry['sources'])} source files at {entry['commit'][:9]}")
        else:
            if Path(args.root).resolve() == ROOT:
                raise ExpertError("materialize into a run dir, not this checkout")
            entry = resolve(args.task, args.version, manifest=manifest)
            for path in materialize(entry, args.root):
                print(path)
    except ExpertError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
