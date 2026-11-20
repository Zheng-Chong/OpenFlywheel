"""Import these three embodiment records into an explicitly chosen local platform database.

Never creates accounts, contacts a remote platform, or overwrites conflicting existing registration data.
"""

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

    store = server.Store(a.db)
    pending = []
    for model in ["g1d_wuji_hand2_1", "g1d_wuji_hand2_2", "g1d_sharpa_wave"]:
        record = json.loads((a.repo / "real2sim/embodiments" / model / "registration.json").read_text())
        reg = {k: v for k, v in record.items() if k not in ["embodiment", "owner", "updated_by", "updated_at"]}
        existing = store.get("embodiment", model) or {}
        if existing.get("reg") and existing["reg"] != reg:
            raise ValueError(f"Refusing to overwrite a different existing registration: {model}")
        pending.append((model, reg))
    for model, reg in pending:
        store.update(
            "embodiment",
            model,
            {"reg": reg, "registered": True, "owner": "liuzhonghan"},
            by="Codex (authorized local model import)",
        )
        print(model)


if __name__ == "__main__":
    main()
