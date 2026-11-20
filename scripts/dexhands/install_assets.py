"""Install only these three assets into an explicitly chosen private library; never overwrite different data."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def verify(directory):
    manifest = json.loads((directory / "sha256.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Asset content mismatch: {directory / name}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    pending = []
    for name in ["g1d_wuji_hand2_1", "g1d_wuji_hand2_2", "g1d_sharpa_wave"]:
        src = a.repo / "real2sim/robot" / name
        expected = verify(src)
        dst = a.assets / "Robots" / name
        if dst.exists():
            if not (dst / "sha256.json").is_file() or verify(dst) != expected:
                raise ValueError(f"Refusing to overwrite different existing asset: {dst}")
        else:
            pending.append((src, dst))
    for src, dst in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        verify(dst)
        print(dst)


if __name__ == "__main__":
    main()
