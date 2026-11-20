"""CPU checks of the g1d_pick_anything round 3 layouts: pytest scripts/datagen/test_pick_anything_layouts.py"""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import (
    pick_anything_layouts as pal,  # noqa: E402
    pour_beans_plus_layouts as plus,  # noqa: E402
)
from scripts.objects import pick_anything  # noqa: E402


def fake_assets(root, pool):
    """Object/RoboDojo/Rigid/<cat>/<idx>/metadata.json of every pool object, with its size and mass."""
    for o in pool:
        d = Path(root) / "Object" / "RoboDojo" / "Rigid" / o["cat"] / f"{o['idx']:05d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({"physics": {"size": o["size"], "mass": o["mass"]}}))


class PickAnythingR3Layouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pool = pick_anything.load_pool()
        fake_assets(cls.tmp.name, cls.pool)
        cls.gen = pal.Generator(cls.tmp.name, cls.pool)
        cls.layouts = [(c, i, cls.gen.layout(c, 760, i)) for c in pal.CONFIGS for i in range(40)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reproducible(self):
        self.assertEqual(self.gen.layout("g1d", 760, 3), self.layouts[3][2])
        self.assertNotEqual(self.gen.layout("g1d", 761, 3), self.layouts[3][2])

    def test_every_axis_at_l2(self):
        by = {o["family"]: 0 for o in self.pool}
        for o in self.pool:
            by[o["family"]] += 1
        for config, i, lay in self.layouts:
            g = lay["Generalization"]
            self.assertEqual(g["levels"], dict.fromkeys(pal.DIMS, 2))
            self.assertEqual(g["scene"], config)
            self.assertTrue(1 <= len(lay["Light"]) <= 3)
            self.assertEqual(set(lay["Camera"]), {"cam_head", *plus.WRISTS})
            self.assertIn("noise", lay["Imaging"])
            for k, (low, high) in plus.BASE.items():
                self.assertLessEqual(abs(g["base"][k]), low)
            target = self.pool[pick_anything.pool_order(self.pool)[i % len(self.pool)]]
            self.assertEqual(g["target"], f"{target['cat']}/{target['idx']:05d}")
            if by[target["family"]] > 1:
                self.assertGreaterEqual(g["same_family"], 1)

    def test_target_area_and_scale(self):
        fixed = pal.sized()
        pos, scaled = [], 0
        for config, i, lay in self.layouts:
            base = lay["Generalization"]["base"]
            top = pal.appearance.load(pal.build_asset.SCENES_DIR / plus.SCENES[config])[0]["top_z"]
            self.assertAlmostEqual(lay["Room"]["table_height"], top + base["table"] - base["lift"], places=3)
            for es in lay["Rigid"].values():
                for e in es:
                    s = e["scale"][0]
                    key = f"{e['category']}/{e['category_idx']:05d}"
                    obj = next(o for o in self.pool if (o["cat"], o["idx"]) == (e["category"], e["category_idx"]))
                    self.assertTrue(pick_anything.graspable([v * s for v in obj["size"]], obj["mass"] * s**3), key)
                    if key in fixed:
                        self.assertEqual(s, 1.0)
                    scaled += s != 1.0
                    if e["label"] == "target":
                        pos.append(e["default_pos"][:2])
        self.assertGreater(scaled, 0)
        # the desk frame's |x| <= 0.25, y -0.53 .. -0.30, moved with the base (<= 3 cm, 5 deg about the robot root)
        for x, y in pos:
            self.assertLessEqual(abs(x), 0.25 + 0.03 + 0.73 * math.radians(5) * 0.3)
            self.assertTrue(-0.53 - 0.05 <= y <= -0.30 + 0.05, (x, y))
        self.assertGreater(max(abs(x) for x, _ in pos), 0.19)

    def test_lower_levels_for_diagnostics(self):
        r2 = dict(spatial=1, scene=0, object=1, camera=0, language=1)
        gen = pal.Generator(self.tmp.name, self.pool, levels=r2)
        for i in range(20):
            lay = gen.layout("g1d_shenzhen_lab", 790, i)
            self.assertEqual(lay["Generalization"]["levels"], r2)
            self.assertFalse({"Light", "Camera", "Imaging"} & set(lay))
            self.assertEqual(set(lay["Generalization"]["base"].values()), {0.0})
            self.assertEqual(set(lay["Generalization"]["scale"].values()), {1.0})
            x, y = lay["Generalization"]["target_xy"]
            self.assertTrue(0.04 <= abs(x) <= 0.19 and -0.47 <= y <= -0.36, (x, y))

    def test_later_round(self):
        gen = pal.Generator(self.tmp.name, self.pool, round_="g1d_pick_anything-r4")
        lay = gen.layout("g1d", 760, 3)
        self.assertEqual(lay["Generalization"]["round"], "g1d_pick_anything-r4")
        self.assertEqual(lay["Generalization"]["levels"], dict.fromkeys(pal.DIMS, 2))
        self.assertEqual(lay["Generalization"]["target"], self.layouts[3][2]["Generalization"]["target"])
        self.assertNotEqual(lay["Generalization"]["target_xy"], self.layouts[3][2]["Generalization"]["target_xy"])
        self.assertEqual(self.layouts[3][2]["Generalization"]["round"], pal.ROUND)

    def test_commercial_pool(self):
        pool = pick_anything.commercial(self.pool)
        self.assertEqual(len(pool), 295)
        self.assertFalse(any(o["cat"].startswith(pick_anything.NONCOMMERCIAL) for o in pool))
        gen = pal.Generator(self.tmp.name, pool, round_="g1d_pick_anything-r6")
        keys = {f"{o['cat']}/{o['idx']:05d}" for o in pool}
        for i in range(30):
            lay = gen.layout("g1d", 1400, i)
            self.assertEqual(lay["Generalization"]["round"], "g1d_pick_anything-r6")
            self.assertTrue(set(lay["Generalization"]["scale"]) and lay["Generalization"]["target"] in keys)
            labels = {f"{e['category']}/{e['category_idx']:05d}" for es in lay["Rigid"].values() for e in es}
            self.assertLessEqual(labels, keys)

    def test_write_unit(self):
        dst = self.gen.write_unit("g1d_shenzhen_lab", 780, 2, per=3)
        self.assertEqual(dst.relative_to(self.tmp.name).as_posix(), "Eval_Layout/RoboDojo/g1d_shenzhen_lab/78052")
        self.assertEqual(sorted(p.name for p in dst.iterdir()), [f"g1d_pick_anything_{j}.json" for j in range(3)])
        written = json.loads((dst / "g1d_pick_anything_1.json").read_text())
        self.assertEqual(written, self.gen.layout("g1d_shenzhen_lab", 780, 3 * 2 + 1))

    def test_v2_index_rows(self):
        import hashlib

        from scripts.datagen import pick_anything_v2_index as idx

        dst = self.gen.write_unit("g1d", 765, 3, per=2)  # candidates 76553, shard seed 76503
        layouts = Path(self.tmp.name) / "Eval_Layout" / "RoboDojo"

        def quality(j):
            f = f"g1d_pick_anything_{j}.json"
            sha = hashlib.sha256((dst / f).read_bytes()).hexdigest()
            return dict(env_cfg="g1d", config_name="g1d", layout_seed=76503, layout_id=j, layout_file=f,
                        layout_sha256=sha, objects={"target": ["mug", j], "distractor_0": [noncommercial[j], 0]},
                        instruction=f"拿起杯子{j}")  # fmt: skip

        noncommercial = {0: "geniesim_benchmark_plug_benchmark_plug_008", 1: "sand3d_packaging"}

        with tempfile.TemporaryDirectory() as d:
            part = Path(d)
            (part / "collection_quality.json").write_text(json.dumps([quality(1), quality(0)]))
            (part / "sha256.json").write_text("{}")
            idx.main(["part", "--dataset", str(part), "--round", "g1d_pick_anything-r3", "--layouts", str(layouts)])
            got = [json.loads(line) for line in (part / idx.META).read_text().splitlines()]
            self.assertEqual([r["episode_index"] for r in got], [0, 1])
            self.assertEqual([r["layout_id"] for r in got], [1, 0])
            self.assertEqual(got[0]["levels"], dict.fromkeys(pal.DIMS, 2))
            self.assertEqual(got[0]["generalization"]["scene"], "g1d")
            self.assertEqual(got[0]["licenses"], ["Apache-2.0", "CC-BY-4.0"])
            self.assertTrue(got[0]["commercial"])
            self.assertEqual(got[1]["objects"]["distractor_0"][2:], ["AgiBot GenieSim", "CC-BY-NC-SA-4.0"])
            self.assertFalse(got[1]["commercial"])
            self.assertIn(idx.META, json.loads((part / "sha256.json").read_text()))
            r2 = idx.rows(part, "g1d_pick_anything-r2")
            self.assertEqual(r2[0]["levels"], dict(spatial=1, scene=0, object=2, camera=0, language=1))
            sub = part / "commercial.jsonl"
            idx.main(["index", "--out", str(part / "all.jsonl"), "--part", f"p={part}=g1d_pick_anything-r2",
                      "--commercial", str(sub)])  # fmt: skip
            self.assertEqual([json.loads(line)["layout_id"] for line in sub.read_text().splitlines()], [1])
            self.assertEqual(idx.rows(part, "g1d_pick_anything-r3"), got)  # from the part's own rows
            meta = (part / idx.META).read_text().splitlines()
            (part / idx.META).write_text("\n".join([meta[0].replace(got[0]["layout_sha256"], "0" * 64), meta[1]]))
            with self.assertRaises(SystemExit):
                idx.rows(part, "g1d_pick_anything-r3")
            (dst / "g1d_pick_anything_1.json").write_text("{}")
            with self.assertRaises(SystemExit):  # not the layout the episode ran
                idx.rows(part, "g1d_pick_anything-r3", [layouts])

    def test_v2_index_sources_of_the_pool(self):
        from collections import Counter

        from scripts.datagen import pick_anything_v2_index as idx

        pool = json.loads((ROOT / "scripts/objects/pick_anything_pool.json").read_text())["objects"]
        libs = Counter(idx.source(o["cat"])[0] for o in pool)
        self.assertEqual(libs["AgiBot GenieSim"], 256)
        self.assertEqual(libs["OpenFlywheel Real2Sim"], 39)
        self.assertEqual(libs["RoboDojo (HF RoboDojo-Benchmark/RoboDojo)"], 33)
        self.assertEqual(sum(libs.values()), 551)


if __name__ == "__main__":
    unittest.main()
