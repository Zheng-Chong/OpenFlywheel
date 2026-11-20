"""CPU checks for the expert versions (scripts/datagen/experts.py, experts.json): python -m pytest scripts/datagen"""

import json
from pathlib import Path
import unittest

from scripts.datagen import experts

ROOT = Path(__file__).resolve().parents[2]
TREE = {
    "scripts/datagen/plugins/t.py": 'from scripts.g1d import demo as d\nACCEPTANCE = "t_v1"\n',
    "scripts/datagen/plugins/t_v2.py": "from scripts.datagen.plugins.t import ACCEPTANCE\nfrom . import _shared\n",
    "scripts/datagen/plugins/_shared.py": "import numpy as np\n",
    "scripts/g1d/demo.py": 'from scripts.objects import shape\nB = lambda: __import__("scripts.g1d.extra")\n',
    "scripts/g1d/extra.py": "",
    "scripts/objects/shape.py": "from env.reward_manager import checks\n",
}


def tree_reader(tree):
    return tree.get


def manifest(tree):
    sources = experts.fingerprint("t", tree_reader(tree))
    entry = dict(version="v1", plugin="t", acceptance="t_v1", commit="c" * 40, created="2026-11-09", summary="s")
    return {"experts": {"t": [dict(entry, sources=sources)]}}


class ExpertVersionsTest(unittest.TestCase):
    def test_sources_follow_scripts_imports(self):
        files = experts.source_files("t", tree_reader(TREE))
        self.assertEqual(
            sorted(files),
            ["scripts/datagen/plugins/t.py", "scripts/g1d/demo.py", "scripts/g1d/extra.py", "scripts/objects/shape.py"],
        )
        v2 = experts.source_files("t_v2", tree_reader(TREE))
        self.assertIn("scripts/datagen/plugins/_shared.py", v2)  # relative import
        self.assertIn("scripts/datagen/plugins/t.py", v2)
        self.assertEqual(experts.constant("t_v2", "ACCEPTANCE", tree_reader(TREE)), "t_v1")
        with self.assertRaises(experts.ExpertError):
            experts.source_files("missing", tree_reader(TREE))

    def test_record_checks_the_code(self):
        m = manifest(TREE)
        same = experts.record("t", manifest=m, read=tree_reader(TREE))
        self.assertEqual((same["version"], same["drift"]), ("v1", []))
        changed = dict(TREE, **{"scripts/g1d/demo.py": TREE["scripts/g1d/demo.py"] + "X = 1\n"})
        self.assertEqual(experts.record("t", manifest=m, read=tree_reader(changed))["drift"], ["scripts/g1d/demo.py"])
        with self.assertRaisesRegex(experts.ExpertError, "materialize"):
            experts.record("t", "v1", manifest=m, read=tree_reader(changed))
        with self.assertRaisesRegex(experts.ExpertError, "no expert v9"):
            experts.record("t", "v9", manifest=m, read=tree_reader(TREE))
        with self.assertRaisesRegex(experts.ExpertError, "not t_v2"):
            experts.resolve("t", "v1", "t_v2", manifest=m)
        self.assertIsNone(experts.record("other", manifest=m, read=tree_reader(TREE)))

    def test_add_appends_only_new_code(self):
        m = {"experts": {}}
        read = tree_reader(TREE)
        self.assertEqual(experts.add(m, "t", "t", "a" * 40, "first", read=read)["version"], "v1")
        with self.assertRaisesRegex(experts.ExpertError, "already has this code"):
            experts.add(m, "t", "t", "b" * 40, "same", read=read)
        self.assertEqual(experts.add(m, "t", "t_v2", "b" * 40, "other plugin", read=read)["acceptance"], "t_v1")
        self.assertEqual([e["version"] for e in m["experts"]["t"]], ["v1", "v2"])

    def test_manifest_and_registry_agree(self):
        m = experts.load()
        for task, entries in m["experts"].items():
            self.assertEqual([e["version"] for e in entries], [f"v{i + 1}" for i in range(len(entries))], task)
            for e in entries:
                self.assertEqual(len(e["commit"]), 40, (task, e["version"]))
                self.assertTrue(e["summary"] and e["sources"], (task, e["version"]))
                self.assertIn(f"scripts/datagen/plugins/{e['plugin']}.py", e["sources"], (task, e["version"]))
        known = {f"{t}@{e['version']}" for t, es in m["experts"].items() for e in es}
        registry = json.loads((ROOT / "scripts/datagen/registry.json").read_text())["datasets"]
        for d in registry:
            if "expert" in d:
                self.assertIn(d["expert"], known, d["name"])
                self.assertEqual(d["expert"].partition("@")[0], d["task"], d["name"])


if __name__ == "__main__":
    unittest.main()
