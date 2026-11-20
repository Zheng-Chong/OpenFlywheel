"""Copy a seed folder of bean layouts to a new seed with the current red-bean physics (build_asset.RED_BEAN_PHYSICS):
cups, beans' fill and distractors stay as they were, so a benchmark's new version differs from the old one only in
the beans. Never writes into an existing folder.

    python scripts/g1d/rebean_layouts.py --assets <Assets> --env-cfg g1d_shenzhen_lab --src 900 --dst 901
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.g1d.build_asset import RED_BEAN_PHYSICS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--env-cfg", required=True, help="layout namespace (an env config's config_name)")
    parser.add_argument("--src", type=int, required=True)
    parser.add_argument("--dst", type=int, required=True)
    args = parser.parse_args()
    root = args.assets / "Eval_Layout" / "RoboDojo" / args.env_cfg
    src, dst = root / str(args.src), root / str(args.dst)
    dst.mkdir(parents=False, exist_ok=False)
    n = 0
    for path in sorted(src.iterdir()):
        if path.suffix != ".json":
            continue
        layout = json.loads(path.read_text(encoding="utf-8"))
        for entry in (layout.get("Fluid") or {}).get("wuliangye", []):
            entry["physics"] = dict(entry["physics"], **deepcopy(RED_BEAN_PHYSICS))
            n += 1
        (dst / path.name).write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in sorted(src.iterdir()):  # sidecars (split.json, sampling.json, ...) as they are
        if path.suffix != ".json" and path.is_file():
            shutil.copy2(path, dst / path.name)
    print(f"{n} bean entries -> {dst}")


if __name__ == "__main__":
    main()
