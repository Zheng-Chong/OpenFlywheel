"""Stage run dirs for export.py that keep, per collection folder of a recipe round, only its first `folder_target`
accepted episodes with distinct layouts, so each recipe stratum enters the dataset at its target (run_batch.py
overshoots a target by whatever its workers were still recording).

The round's units (`units_all.json`: one entry per collected unit `{scene, env_cfg, unit, parent, folder_target}`)
were collected as batches `<scene>_<unit>`, possibly on several hosts (`<root>/<host>/batch/<scene>_<unit>/`, the OBS
layout of raw/bp-r1-20261109). Writes <out>/<scene>_<unit>_<host>/ with the batch's collection_config.json and
raw/seed*/episode_* symlinks, in folder, unit, host, seed and layout order:
    python scripts/datagen/cap_runs.py --units units_all.json --root <object-store>/<user>/raw/bp-r1-20261109 \
        --out <run dir>/runs
    python scripts/datagen/export.py $(printf -- '--run %s ' <run dir>/runs/*) --episodes <printed total> --name ...
Exits non-zero when a folder is short of its target, the batches carry different code revisions or a sidecar's
HDF5 is missing.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.export import layout_key  # noqa: E402


def stage(units, root, out):
    """{(scene, parent): accepted staged} and the code revisions seen."""
    folders, revisions, seen = {}, set(), set()
    for u in sorted(units, key=lambda u: (u["scene"], u["parent"], u["unit"])):
        folder = (u["scene"], u["parent"])
        folders.setdefault(folder, 0)
        for batch in sorted(root.glob(f"*/batch/{u['scene']}_{u['unit']}")):
            config = json.loads((batch / "collection_config.json").read_text())
            revisions.add(json.dumps(config.get("revision"), sort_keys=True))
            run = out / f"{u['scene']}_{u['unit']}_{batch.parts[-3]}"  # a unit retried on another host repeats seeds
            for sidecar in sorted(batch.glob("raw/seed*/episode_*.json")):
                if folders[folder] >= u["folder_target"]:
                    break
                quality = json.loads(sidecar.read_text())
                key = (quality["env_cfg"], layout_key(quality))
                if not quality["accepted"] or key in seen:
                    continue
                if not sidecar.with_suffix(".hdf5").is_file():
                    sys.exit(f"{sidecar}: accepted but its HDF5 is missing")
                seen.add(key)
                folders[folder] += 1
                dst = run / "raw" / sidecar.parent.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in (sidecar, sidecar.with_suffix(".hdf5")):
                    (dst / f.name).symlink_to(f.resolve())
                if not (run / "collection_config.json").exists():
                    shutil.copyfile(batch / "collection_config.json", run / "collection_config.json")
    return folders, revisions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="dir of <host>/batch/<scene>_<unit>/")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error(f"{a.out} exists")
    units = json.loads(a.units.read_text())
    folders, revisions = stage(units, a.root, a.out)
    targets = {(u["scene"], u["parent"]): u["folder_target"] for u in units}
    for (scene, parent), n in sorted(folders.items()):
        print(f"{scene} {parent}: {n}/{targets[scene, parent]}")
    print(f"total {sum(folders.values())}/{sum(targets.values())} in {len(list(a.out.glob('*')))} runs: {a.out}")
    if len(revisions) > 1:
        sys.exit(f"batches carry {len(revisions)} code revisions: {sorted(revisions)}")
    if short := [f for f, n in folders.items() if n < targets[f]]:
        sys.exit(f"folders short of their target: {short}")


if __name__ == "__main__":
    main()
