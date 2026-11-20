"""Verify and install only RH56DFTP into an explicitly selected private asset directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

MODEL = "g1d_inspire_rh56dftp"


def verify(path):
    files = json.loads((path / "sha256.json").read_text())
    for name, sha in files.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Asset content mismatch: " + name)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    source, target = a.repo / "real2sim/robot" / MODEL, a.assets / "Robots" / MODEL
    files = verify(source)
    if target.exists():
        if verify(target) != files:
            raise ValueError("Refusing to replace a different existing robot: " + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    verify(target)
    print(target)


if __name__ == "__main__":
    main()
