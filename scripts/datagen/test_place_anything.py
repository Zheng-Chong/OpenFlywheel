"""CPU checks of g1d_place_anything / g1d_put_anything_in: the receptacle geometry, the layouts, the plugins' acceptance
and the receptacle pool from height maps. pytest scripts/datagen/test_place_anything.py"""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import place_anything_layouts as pal  # noqa: E402
from scripts.datagen.test_pick_anything_layouts import fake_assets  # noqa: E402
from scripts.objects import (
    pick_anything,  # noqa: E402
    place_anything as pa,  # noqa: E402
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def yaw_quat(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, 0.0, math.sin(a))


def receptacle(rid, kind, half, radius, floor, rim, top, family, size, split="train"):
    geom = dict(region=dict(center=[0.0, 0.0], half=list(half), radius=radius), floor=floor, rim=rim, top=top)
    geom["bottom"] = -size[2] / 2
    lib, _, cat, idx = rid.split("/")
    return dict(id=rid, library=lib, cat=cat, idx=int(idx), family=family, names={}, license="CC-BY-4.0",
                size=list(size), mass=0.2, mass_estimated=False, kinds=[kind], geometry={kind: geom},
                split={kind: split})  # fmt: skip


TRAY = receptacle("RoboDojo/Rigid/fake_tray/00000", "support", (0.09, 0.06), 0.06, -0.008, -0.008, 0.012, "tray",
                  (0.20, 0.14, 0.024))  # fmt: skip
PLATE = receptacle("RoboDojo/Rigid/fake_plate/00000", "support", (0.07, 0.07), 0.09, -0.005, -0.005, 0.01, "plate",
                   (0.22, 0.22, 0.02))  # fmt: skip
BOWL = receptacle("RoboDojo/Rigid/fake_bowl/00000", "container", (0.05, 0.05), 0.065, -0.025, 0.03, 0.03, "bowl",
                  (0.16, 0.16, 0.06))  # fmt: skip
BOX = receptacle("Real2Sim/Rigid/fake_box/00000", "container", (0.08, 0.08), 0.08, -0.04, 0.045, 0.045, "box",
                 (0.18, 0.18, 0.09))  # fmt: skip
POOL_Y = [TRAY, PLATE, BOWL, BOX]


def geo(rec):
    (kind,) = rec["kinds"]
    return dict(rec["geometry"][kind], kind=kind)


def fake_receptacle_assets(root, receptacles):
    for r in receptacles:
        d = Path(root) / "Object" / r["library"] / "Rigid" / r["cat"] / f"{r['idx']:05d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({"physics": {"size": r["size"], "mass": r["mass"]}}))


class Geometry(unittest.TestCase):
    def test_apart(self):
        a = (0.0, 0.0, 0.05, 0.02, 0.0)
        self.assertTrue(pa.apart(a, (0.08, 0.0, 0.02, 0.02, 0.0), 0.005))
        self.assertFalse(pa.apart(a, (0.08, 0.0, 0.02, 0.02, 0.0), 0.02))
        self.assertFalse(pa.apart(a, (0.0, 0.0, 0.01, 0.01, 1.0), 0.0))  # inside
        # turned 45 deg: the corner reaches 0.02 * sqrt(2) out
        self.assertFalse(pa.apart((0.0, 0.0, 0.02, 0.02, math.pi / 4), (0.05, 0.0, 0.02, 0.02, 0.0), 0.005))

    def test_fits_at_yaw(self):
        long_x = [0.15, 0.03, 0.05]
        self.assertTrue(pa.fits(geo(TRAY), long_x, "support", 0.0))  # along the tray's long side
        self.assertFalse(pa.fits(geo(TRAY), long_x, "support", math.pi / 2))  # across: 15 cm > 12 cm
        self.assertTrue(pa.fits_some_yaw(geo(TRAY), long_x, "support"))
        self.assertFalse(pa.fits_some_yaw(geo(TRAY), [0.25, 0.03, 0.05], "support"))
        self.assertTrue(pa.fits(geo(PLATE), [0.15, 0.03, 0.05], "support", 1.0))  # inside its circle at any yaw

    def test_container_depth(self):
        self.assertTrue(pa.fits(geo(BOWL), [0.05, 0.03, 0.09], "container", 0.0))  # centre 4.5 cm < 5.5 cm deep - slack
        self.assertFalse(pa.fits(geo(BOWL), [0.05, 0.03, 0.11], "container", 0.0))

    def test_spot_toward(self):
        size = [0.04, 0.03, 0.05]
        spot = pa.spot_toward(geo(TRAY), size, 0.0, (1.0, 1.0))
        self.assertAlmostEqual(spot[0], 0.09 - 0.02 - pa.MARGIN)
        self.assertAlmostEqual(spot[1], 0.06 - 0.015 - pa.MARGIN)
        self.assertEqual(pa.spot_toward(geo(TRAY), [0.3, 0.3, 0.1], 0.0, (1, 1)), [0.0, 0.0])

    def test_placed_support(self):
        y_pos, y_quat = (0.0, -0.42, 0.79), yaw_quat(30)
        geom = geo(TRAY)
        face = y_pos[2] + geom["floor"]
        size = [0.05, 0.03, 0.06]
        on = (0.02, -0.42, face + size[2] / 2 + 0.002)
        self.assertTrue(pa.placed("support", geom, 1.0, on, IDENTITY, size, y_pos, y_quat, IDENTITY))
        self.assertFalse(pa.placed("support", geom, 1.0, (0.2, -0.42, on[2]), IDENTITY, size, y_pos, y_quat))
        high = (on[0], on[1], on[2] + 0.03)
        self.assertFalse(pa.placed("support", geom, 1.0, high, IDENTITY, size, y_pos, y_quat))
        lying = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # tipped 90 deg about X
        self.assertFalse(pa.placed("support", geom, 1.0, on, lying, size, y_pos, y_quat, IDENTITY))

    def test_placed_container(self):
        y_pos = (0.0, -0.42, 0.80)
        geom = geo(BOWL)
        inside = (0.01, -0.41, y_pos[2] - 0.0)
        self.assertTrue(pa.placed("container", geom, 1.0, inside, IDENTITY, [0.03] * 3, y_pos, IDENTITY))
        self.assertFalse(pa.placed("container", geom, 1.0, (0.01, -0.41, 0.85), IDENTITY, [0.03] * 3, y_pos, IDENTITY))
        self.assertFalse(pa.placed("container", geom, 1.0, (0.12, -0.42, 0.79), IDENTITY, [0.03] * 3, y_pos, IDENTITY))

    def test_kept_and_tilt(self):
        self.assertTrue(pa.kept((0.01, -0.42, 0.8), IDENTITY, (0.0, -0.42, 0.8)))
        self.assertFalse(pa.kept((0.04, -0.42, 0.8), IDENTITY, (0.0, -0.42, 0.8)))
        tipped = (math.cos(math.radians(7.5)), math.sin(math.radians(7.5)), 0.0, 0.0)  # 15 deg
        self.assertFalse(pa.kept((0.0, -0.42, 0.8), tipped, (0.0, -0.42, 0.8)))
        self.assertAlmostEqual(pa.tilt_deg(yaw_quat(70)), 0.0, places=5)

    def test_splits(self):
        recs = [dict(id=f"RoboDojo/Rigid/x/{i:05d}", kinds=["container"]) for i in range(40)]
        recs += [dict(id=f"RoboDojo/Rigid/y/{i:05d}", kinds=["support", "container"]) for i in range(5)]
        by = pa.splits(recs)
        self.assertEqual(sum(v == "eval" for v in by["container"].values()), round(0.15 * 45))
        self.assertEqual(sum(v == "eval" for v in by["support"].values()), 1)
        self.assertEqual(by, pa.splits(list(reversed(recs))))


class Layouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pool = pick_anything.load_pool()
        fake_assets(cls.tmp.name, cls.pool)
        fake_receptacle_assets(cls.tmp.name, POOL_Y)
        cls.gens = {t: pal.Generator(t, cls.tmp.name, cls.pool, POOL_Y) for t in pal.KIND}
        cls.layouts = {t: [(c, i, g.layout(c, 1100, i)) for c in pal.pal.CONFIGS for i in range(25)]
                       for t, g in cls.gens.items()}  # fmt: skip

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reproducible(self):
        for task, gen in self.gens.items():
            self.assertEqual(gen.layout("g1d", 1100, 3), self.layouts[task][3][2])
            self.assertNotEqual(gen.layout("g1d", 1101, 3), self.layouts[task][3][2])

    def test_x_fits_y(self):
        for task, gen in self.gens.items():
            kind = pal.KIND[task]
            for config, i, lay in self.layouts[task]:
                entries = {e["label"]: e for es in lay["Rigid"].values() for e in es}
                x, y = entries["target"], entries[pa.LABEL]
                g = lay["Generalization"]
                self.assertEqual(g["kind"], kind)
                self.assertEqual(y["receptacle"]["kind"], kind)
                self.assertEqual(g["round"], f"{task}-r1")
                self.assertEqual(g["levels"], dict.fromkeys(pal.pal.DIMS, 2))
                rel = (
                    math.atan2(2 * x["default_ori"][0] * x["default_ori"][3], 1 - 2 * x["default_ori"][3] ** 2)
                    - math.atan2(2 * y["default_ori"][0] * y["default_ori"][3], 1 - 2 * y["default_ori"][3] ** 2)
                )  # fmt: skip
                self.assertTrue(pa.fits(y["receptacle"], x["physics"]["size"], kind, rel), (task, i))
                self.assertIn(g["receptacle"], {r["id"] for r in POOL_Y if kind in r["kinds"]})
                self.assertEqual(y["physics"]["size"], next(r for r in POOL_Y if r["id"] == g["receptacle"])["size"])
                self.assertTrue(2 <= len(entries) - 2 <= pa.DISTRACTORS[1] + 1)

    def test_footprints_apart(self):
        for task in self.gens:
            for config, i, lay in self.layouts[task]:
                rects = {}
                for es in lay["Rigid"].values():
                    for e in es:
                        w, d = e["physics"]["size"][:2]
                        q = e["default_ori"]
                        rects[e["label"]] = (*e["default_pos"][:2], w / 2, d / 2, 2 * math.atan2(q[3], q[0]))
                labels = sorted(rects)
                for a in labels:
                    for b in labels:
                        if a < b:
                            gap = pa.TARGET_GAP if "target" in (a, b) else pa.GAP
                            self.assertTrue(pa.apart(rects[a], rects[b], gap - 1e-6), (task, i, a, b))

    def test_x_order_covers_fitting_objects(self):
        for task, gen in self.gens.items():
            kind = pal.KIND[task]
            self.assertGreater(len(gen.order), 0)
            for i in gen.order[:50]:
                fit = [pa.fits_some_yaw(geo(r), self.pool[i]["size"], kind) for r in pa.of_kind(POOL_Y, kind)]
                self.assertTrue(any(fit))

    def test_instruction(self):
        import random

        names = {"zh": {"x": ["苹果"], "y": ["盘子"]}, "en": {"x": ["apple"], "y": ["plate"]}}
        seen = {pa.instruction(random.Random(i), "support", names) for i in range(40)}
        self.assertTrue(any("苹果" in s and "盘子" in s for s in seen))
        self.assertTrue(any("apple" in s and "plate" in s for s in seen))
        put = {pa.instruction(random.Random(i), "container", names) for i in range(80)}
        self.assertIn("Put the apple into the plate.", put)


class Licenses(unittest.TestCase):
    def test_commercial_only(self):
        pool = pick_anything.load_pool()
        kept = pal.commercial(pool)
        self.assertTrue(any(o["cat"].startswith("geniesim_") for o in pool))
        self.assertFalse(any(o["cat"].startswith(pa.NONCOMMERCIAL) for o in kept))
        self.assertEqual(len(kept), sum(not o["cat"].startswith(pa.NONCOMMERCIAL) for o in pool))
        self.assertFalse(any(r["cat"].startswith(pa.NONCOMMERCIAL) for r in pa.load_receptacles()))

    def test_later_rounds_leave_out_excluded(self):
        pool = pick_anything.load_pool()
        cats = lambda p: {o["cat"] for o in p}  # noqa: E731
        car_parts = {(o["cat"], o["idx"]) for o in pool if o["family"] == "car part"}
        self.assertEqual(len(car_parts), 6)  # filed under other categories, kept (130/235 | 142/243 in round 1)
        for task in (None, "g1d_pick_anything", "g1d_put_anything_in", "g1d_handover_anything"):
            kept = pick_anything.usable(pool, task)
            self.assertEqual(len(pool) - len(kept), 2, task)  # sand3d_vehicle only
            self.assertNotIn("sand3d_vehicle", cats(kept), task)
            self.assertTrue(car_parts <= {(o["cat"], o["idx"]) for o in kept}, task)
            self.assertTrue({"game_controller", "hammer"} <= cats(kept), task)
        place = pick_anything.usable(pool, "g1d_place_anything")
        self.assertEqual(len(pool) - len(place), 7)
        self.assertFalse(cats(place) & {"sand3d_vehicle", "game_controller", "hammer"})
        sizes = {"g1d_place_anything": 288, "g1d_put_anything_in": 293}
        for task in pa.TASKS.values():
            self.assertEqual(len(pal.Generator(task, "Assets", round_=f"{task}-r2").pool), 295, task)
            self.assertEqual(len(pal.Generator(task, "Assets", round_=f"{task}-r3").pool), sizes[task], task)

    def test_layouts_draw_commercial_x(self):
        for task, gen in Layouts.gens.items():
            self.assertFalse(any(o["cat"].startswith(pa.NONCOMMERCIAL) for o in gen.pool), task)


class Release(unittest.TestCase):
    def test_readme_lists_sources_and_licenses(self):
        from scripts.datagen import place_anything_release as rel

        recs = Path(pa.__file__).with_name("place_anything_receptacles_r2.json")
        quality = [
            dict(objects=dict(target=["sand3d_packaging", 269], receptacle=["clutter_book", 0],
                              distractor_0=["wood_block_cube", 2])),
            dict(objects=dict(target=["game_controller", 0], receptacle=["marker_box", 2],
                              distractor_0=["fmb_peg_round", 24])),
        ]  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "collection_quality.json").write_text(json.dumps(quality))
            (d / "README.md").write_text("# x\n")
            (d / "sha256.json").write_text("{}")
            for _ in range(2):  # a second run replaces the section
                sys.argv = ["x", "readme", "--dataset", str(d), "--receptacles", str(recs), "--new-in", "r2"]
                rel.main()
            text = (d / "README.md").read_text()
            self.assertEqual(text.count(rel.HEAD), 1)
            self.assertIn("| Sand3D | CC-BY-4.0 | 1 | 1 |", text)
            self.assertIn("Receptacles added in r2: 1 of the 2 episodes.", text)
            self.assertIn("No object on the table has a non-commercial license", text)
            self.assertEqual(len(json.loads((d / "sha256.json").read_text())["README.md"]), 64)


class Plugin(unittest.TestCase):
    def episode(self, kind, frames=40, release=True, pushed=False):
        from scripts.datagen.plugins import g1d_place_anything as plugin

        labels = ["distractor_0", pa.LABEL, "target"]
        rec = dict((TRAY if kind == "support" else BOWL)["geometry"][kind], kind=kind)
        y = np.array([0.0, -0.42, 0.79, 1, 0, 0, 0])
        start = np.array([[0.2, -0.5, 0.78, 1, 0, 0, 0], y, [-0.15, -0.4, 0.80, 1, 0, 0, 0]])
        size = [0.04, 0.03, 0.04]
        if kind == "support":
            end_z = y[2] + rec["floor"] + size[2] / 2 + 0.001
        else:
            end_z = y[2] + rec["floor"] + 0.02
        poses = np.repeat(start[None], frames, axis=0).astype(np.float32)
        poses[:, 2, :3] = [0.01, -0.42, end_z]
        if pushed:
            poses[-5:, 1, 0] += 0.05
        opening = np.full((frames, 2), 1.0 if release else 0.2, dtype=np.float32)
        quality = dict(task=plugin.TASK, frames=frames, labels=labels, start=start.tolist(), receptacle=rec,
                       target_size=size, kind=kind, instruction="把苹果放到托盘上",
                       usable_names={"zh": ["苹果"]}, usable_receptacle_names={"zh": ["托盘"]})  # fmt: skip
        return plugin, dict(object_poses=poses, gripper_openness=opening), quality

    def test_audit_accepts(self):
        for kind in pa.KINDS:
            plugin, priv, quality = self.episode(kind)
            self.assertEqual(plugin.audit(priv, quality)["kind"], kind)

    def test_audit_rejects(self):
        for kwargs in (dict(release=False), dict(pushed=True)):
            plugin, priv, quality = self.episode("support", **kwargs)
            with self.assertRaises(AssertionError):
                plugin.audit(priv, quality)
        plugin, priv, quality = self.episode("support")
        quality["instruction"] = "把苹果放到桌上"
        with self.assertRaises(AssertionError):
            plugin.audit(priv, quality)

    def test_put_in_plugin(self):
        from scripts.datagen.plugins import g1d_put_anything_in as put

        self.assertEqual((put.TASK, put.KIND, put.Episode.kind), ("g1d_put_anything_in", "container", "container"))


class Screening(unittest.TestCase):
    def test_rates_and_screen_out(self):
        pool = pick_anything.load_pool()
        x = pool[0]
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, (rid, ok) in enumerate([(TRAY["id"], True), (TRAY["id"], False)] + [(PLATE["id"], False)] * 4):
                p = Path(tmp) / f"episode_{i:06d}.json"
                q = dict(
                    kind="support", accepted=ok, receptacle=dict(id=rid), objects=dict(target=[x["cat"], x["idx"]])
                )
                p.write_text(json.dumps(q))
                paths.append(p)
            stats = pa.screen_stats(paths, pool)
        self.assertEqual(stats["receptacle"][TRAY["id"]], [1, 2])
        self.assertEqual(stats["x_family"][x["family"]], [1, 6])
        recs = json.loads(json.dumps(POOL_Y))
        self.assertEqual(pa.screen_out(recs, stats, "support", 0.3, 4), [PLATE["id"]])
        self.assertNotIn(PLATE["id"], {r["id"] for r in pa.of_kind(recs, "support")})
        self.assertIn(TRAY["id"], {r["id"] for r in pa.of_kind(recs, "support")})
        recs = json.loads(json.dumps(POOL_Y))
        stats["receptacle"] = {}  # each drawn too few times alone: their category decides
        self.assertEqual(pa.screen_out(recs, stats, "support", 0.3, 4, family_least=4), [PLATE["id"]])
        self.assertEqual(pa.screen_out(json.loads(json.dumps(POOL_Y)), stats, "support", 0.3, 4), [])


class Pool(unittest.TestCase):
    def check_dir(self, root, objects):
        """A check_cup_holds_beans.py --out dir with height maps of `objects` [(rel, height, level, root_z)]."""
        (root / "cavity").mkdir(parents=True)
        cups = {}
        xs = ys = np.arange(-0.1, 0.1, 0.0025)
        for n, (rel, height, level, root_z) in enumerate(objects):
            cavity = ~np.isnan(height) & (level - height >= 0.01)
            np.savez(root / "cavity" / f"{n:03d}.npz", xs=xs, ys=ys, height=height, level=level, cavity=cavity,
                     root_z=root_z)  # fmt: skip
            cups[rel] = dict(stand_tilt_deg=0.4, size_m=[0.2, 0.2, 0.05], cavity_depth_mm=10.0)
        (root / "metrics.json").write_text(json.dumps(dict(cups=cups, failed_load={})))

    def test_pool_from_height_maps(self):
        xs = np.arange(-0.1, 0.1, 0.0025)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        r = np.hypot(gx, gy)
        # a bowl: 7 cm opening radius, 5 cm deep, rim 2 cm above the root
        bowl = np.where(r <= 0.08, np.where(r <= 0.07, -0.03 + 0.2 * r**2, 0.02), np.nan)
        bowl_level = np.where(np.isnan(bowl), np.nan, np.maximum(bowl, 0.02))
        # a flat board 16 x 10 cm, top 1 cm above the root
        board = np.where((abs(gx) <= 0.08) & (abs(gy) <= 0.05), 0.01, np.nan)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "check"
            self.check_dir(d, [("RoboDojo/Rigid/bowl/00001", bowl, bowl_level, 0.03),
                               ("RoboDojo/Rigid/board/00002", board, board.copy(), 0.012)])  # fmt: skip
            cands = [dict(rel="RoboDojo/Rigid/bowl/00001", fam="bowl", sup=False, con=True, mass=0.3, names={},
                          license="MIT"),
                     dict(rel="RoboDojo/Rigid/board/00002", fam="board", sup=True, con=False, mass=None, names={},
                          license="MIT"),
                     dict(rel="RoboDojo/Rigid/lost/00000", fam="bowl", sup=False, con=True, mass=1, names={},
                          license="MIT")]  # fmt: skip
            pool, rejected = pa.build_pool([d], cands)
        by = {r["id"]: r for r in pool}
        self.assertEqual(set(rejected), {"RoboDojo/Rigid/lost/00000"})
        bowl_g = by["RoboDojo/Rigid/bowl/00001"]["geometry"]["container"]
        self.assertAlmostEqual(bowl_g["rim"], 0.02, places=3)
        self.assertAlmostEqual(bowl_g["bottom"], -0.03, places=4)
        self.assertGreater(bowl_g["region"]["radius"], 0.055)
        self.assertLess(bowl_g["floor"], -0.02)
        board_g = by["RoboDojo/Rigid/board/00002"]["geometry"]["support"]
        self.assertAlmostEqual(board_g["floor"], 0.01, places=4)
        self.assertAlmostEqual(board_g["region"]["half"][0], 0.08, delta=0.005)
        self.assertAlmostEqual(board_g["region"]["half"][1], 0.05, delta=0.005)
        self.assertTrue(by["RoboDojo/Rigid/board/00002"]["mass_estimated"])


if __name__ == "__main__":
    unittest.main()
