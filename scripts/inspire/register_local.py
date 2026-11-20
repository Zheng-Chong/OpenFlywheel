"""Import only the new RH56DFTP embodiment into a chosen local database, without accounts or networking."""

import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo / "src/real2sim_platform"))
    import server

    model = "g1d_inspire_rh56dftp"
    record = json.loads((a.repo / "real2sim/embodiments" / model / "registration.json").read_text())
    reg = {k: v for k, v in record.items() if k not in ("embodiment", "owner", "updated_by", "updated_at")}
    store = server.Store(a.db)
    old = store.get("embodiment", model) or {}
    if old.get("reg") and old["reg"] != reg:
        raise ValueError("Refusing to overwrite conflicting registration")
    store.update(
        "embodiment",
        model,
        {"reg": reg, "registered": True, "owner": "liuzhonghan"},
        by="Codex (authorized local model import)",
    )
    for key, value in server.scan_embodiments()[model].items():
        store.set_auto("embodiment", model, key, value)
    print(model)


if __name__ == "__main__":
    main()
