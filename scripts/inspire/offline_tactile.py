"""Decode a saved RH56DFTP payload without opening a device or network."""

import argparse
import hashlib
import json
from pathlib import Path

from env.robot_manager.inspire_contract import decode_capacitive, decode_piezoresistive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=("piezoresistive", "capacitive"), required=True)
    p.add_argument("--side", choices=("left", "right"), required=True)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--timestamp-s", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = a.payload.read_bytes()
    decoder = decode_piezoresistive if a.format == "piezoresistive" else decode_capacitive
    result = decoder(raw, timestamp_s=a.timestamp_s)
    result.update(
        side=a.side, input_sha256=hashlib.sha256(raw).hexdigest(), source_kind="saved_payload", hardware_connected=False
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2, default=lambda v: v.tolist()) + "\n")


if __name__ == "__main__":
    main()
