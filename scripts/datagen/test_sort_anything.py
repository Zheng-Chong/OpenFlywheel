"""CPU checks of g1d_sort_anything: sort classes, slots, layouts, instructions and the plugin's acceptance.
pytest scripts/datagen/test_sort_anything.py"""

import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import sort_anything_layouts as sal  # noqa: E402
from scripts.datagen.test_pick_anything_layouts import fake_assets  # noqa: E402
from scripts.datagen.test_place_anything import (  # noqa: E402
    BOWL,
    BOX,
    IDENTITY,
    TRAY,
    fake_receptacle_assets,
    geo,
)
from scripts.objects import (
    object_names,  # noqa: E402
    pick_anything,  # noqa: E402
    place_anything as pa,  # noqa: E402
    sort_anything as sa,  # noqa: E402
)


def yaw_of_quat(q):
    return 2 * math.atan2(q[3], q[0])


class Classes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = object_names.load()
        cls.pool = [o for o in pick_anything.load_pool() if not o["cat"].startswith(pa.NONCOMMERCIAL)]
        cls.classes = sa.class_pool(cls.pool, cls.names, pa.load_receptacles(sal.RECEPTACLES))

    def test_classes_commercial_and_named(self):
        self.assertGreaterEqual(len(self.classes), 15)
        for c, objs in self.classes.items():
            self.assertGreaterEqual(len(objs), 3, c)
            for o in objs:
                self.assertFalse(o["cat"].startswith(pa.NONCOMMERCIAL))
                self.assertNotIn(o["cat"], pick_anything.excluded(sa.TASK))
                self.assertIn(o["family"], sa.CLASSES[c][1])

    def test_blurred_names_out(self):
        zh = {c: {self.names[sa.oid(o)]["zh"][0] for o in objs} for c, objs in self.classes.items()}
        self.assertNotIn("字母O", zh["toy"])
        self.assertFalse(any("罐" in n for n in zh["bottle"]))
        self.assertIn("苹果", zh["fruit"])
        self.assertTrue({"钢笔", "胶带卷"} <= zh["stationery"])

    def test_groups(self):
        self.assertTrue(sa.compatible(["fruit", "vegetable"]))
        self.assertFalse(sa.compatible(["block", "toy"]))
        self.assertFalse(sa.compatible(["can", "drink", "fruit"]))
        for group in sa.GROUPS:
            self.assertTrue(group <= set(sa.CLASSES))

    def test_receptacles_train_and_families(self):
        recs = pa.load_receptacles(sal.RECEPTACLES)
        train = sa.sort_receptacles(recs)
        families = {r["family"] for r, _ in train}
        self.assertFalse(families & {"cup", "mug", "book"})
        self.assertTrue({"basket", "bowl", "tray"} <= families)
        for r, kind in train:
            self.assertEqual(r["split"][kind], "train")
        evals = {r["id"] for r, _ in sa.sort_receptacles(recs, "eval")}
        self.assertFalse(evals & {r["id"] for r, _ in train})


class Slots(unittest.TestCase):
    def test_split_region(self):
        subs = sa.split_region(geo(TRAY), 2)  # tray region half (0.09, 0.06): cut along X
        self.assertEqual([s["region"]["center"][0] for s in subs], [-0.045, 0.045])
        self.assertEqual(subs[0]["region"]["half"], [0.045, 0.06])
        self.assertEqual(sa.split_region(geo(TRAY), 1), [geo(TRAY)])

    def test_assign_by_where_they_come_from(self):
        small = [0.04, 0.03, 0.04]
        slots = sa.assign_slots(geo(TRAY), "support", [(small, 0.0, (0.2, 0.0)), (small, 0.0, (-0.2, 0.0))])
        self.assertGreater(slots[0]["region"]["center"][0], 0)
        self.assertLess(slots[1]["region"]["center"][0], 0)

    def test_assign_needs_room(self):
        long_ = [0.08, 0.03, 0.04]  # 8 cm along X does not fit a 9 cm slot with the margin both sides at 0 deg
        self.assertIsNone(sa.assign_slots(geo(TRAY), "support", [(long_, 0.0, (0, 0))] * 2))
        self.assertIsNotNone(sa.assign_slots(geo(TRAY), "support", [(long_, math.pi / 2, (0, 0))] * 2))
        self.assertTrue(sa.holds(geo(TRAY), "support", [long_, long_]))
        self.assertFalse(sa.holds(geo(BOWL), "container", [[0.04, 0.04, 0.04]] * 3))

    def test_counts(self):
        rng = random.Random(0)
        for _ in range(50):
            combo = sa.draw_combo(rng, spare=0.5)
            counts = combo["counts"]
            self.assertTrue(sa.ITEMS[0] <= sum(counts) <= sa.ITEMS[1])
            self.assertTrue(min(counts) >= 1 and max(counts) <= sa.MAX_PER_CLASS)
            self.assertFalse(combo["spare"] and len(counts) == 3)
        self.assertEqual(sa.fewer(dict(counts=[2, 1], spare=True)), dict(counts=[2, 1], spare=False))
        self.assertEqual(sa.fewer(dict(counts=[2, 1], spare=False)), dict(counts=[1, 1], spare=False))
        self.assertEqual(sa.fewer(dict(counts=[1, 1, 1], spare=False)), dict(counts=[1, 1], spare=False))
        self.assertIsNone(sa.fewer(dict(counts=[1, 1], spare=False)))
        r2 = sa.ROUNDS["r2"]
        for _ in range(50):
            combo = sa.draw_combo(rng, r2["three"], 0.0, r2["items"], r2["receptacles"])
            self.assertTrue(3 <= sum(combo["counts"]) <= 4)
            self.assertEqual(len(combo["counts"]) + combo["spare"], 3)
        keep = dict(items=r2["items"], receptacles=3)
        self.assertEqual(sa.fewer(dict(counts=[2, 2], spare=True), **keep), dict(counts=[1, 2], spare=True))
        self.assertEqual(sa.fewer(dict(counts=[2, 1, 1], spare=False), **keep), dict(counts=[1, 1, 1], spare=False))
        self.assertIsNone(sa.fewer(dict(counts=[2, 1], spare=True), **keep))
        self.assertIsNone(sa.fewer(dict(counts=[1, 1, 1], spare=False), **keep))


class Layouts(unittest.TestCase):
    round_ = sal.ROUND

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pool = pick_anything.load_pool()
        fake_assets(cls.tmp.name, cls.pool)
        cls.recs = pa.load_receptacles(sal.RECEPTACLES)
        fake_receptacle_assets(cls.tmp.name, cls.recs)
        cls.gen = sal.Generator(cls.tmp.name, cls.pool, cls.recs, round_=cls.round_)
        cls.layouts = [(c, i, cls.gen.layout(c, 1300, i)) for c in sal.pal.CONFIGS for i in range(30)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def entries(self, lay):
        return {e["label"]: e for es in lay["Rigid"].values() for e in es}

    def test_reproducible(self):
        self.assertEqual(self.gen.layout("g1d", 1300, 3), self.layouts[3][2])
        self.assertNotEqual(self.gen.layout("g1d", 1301, 3), self.layouts[3][2])

    def test_items_fit_their_receptacles(self):
        for config, i, lay in self.layouts:
            e = self.entries(lay)
            s = lay["Generalization"]["sort"]
            self.assertEqual(lay["Generalization"]["levels"], dict.fromkeys(sal.pal.DIMS, 2))
            self.assertEqual(lay["Generalization"]["round"], self.round_)
            ys = sorted(k for k in e if k.startswith(sa.RECEPTACLE))
            items = sorted(k for k in e if k.startswith(sa.ITEM))
            settings = self.gen.settings
            self.assertTrue(settings["items"][0] <= len(items) <= settings["items"][1])
            self.assertEqual(len(ys), len(s["classes"]) + bool(s["spare"]))
            if settings["receptacles"]:
                self.assertEqual(len(ys), settings["receptacles"])
            if settings["receptacle_max"]:
                self.assertTrue(all(max(e[y]["physics"]["size"][:2]) <= settings["receptacle_max"] for y in ys))
            self.assertTrue(sa.compatible(s["classes"]))
            by = {}
            for x in items:
                self.assertIn(e[x]["sort"]["receptacle"], ys)
                self.assertEqual(e[x]["sort"]["class"], e[e[x]["sort"]["receptacle"]]["sort"]["class"])
                by.setdefault(e[x]["sort"]["receptacle"], []).append(x)
            for y, xs in by.items():
                geom = e[y]["receptacle"]
                y_yaw = yaw_of_quat(e[y]["default_ori"])
                parts = [
                    (e[x]["physics"]["size"], yaw_of_quat(e[x]["default_ori"]) - y_yaw,
                     sa.to_local(e[x]["default_pos"][:2], e[y]["default_pos"][:2], math.degrees(y_yaw)))
                    for x in xs
                ]  # fmt: skip
                self.assertIsNotNone(sa.assign_slots(geom, geom["kind"], parts), (config, i, y))
            if s["spare"]:
                self.assertIsNone(e[s["spare"]]["sort"]["class"])
            classes = {e[x]["sort"]["class"] for x in items}
            self.assertEqual(classes, set(s["classes"]))

    def test_footprints_apart(self):
        for config, i, lay in self.layouts:
            rects = {}
            for k, e in self.entries(lay).items():
                w, d = e["physics"]["size"][:2]
                rects[k] = (*e["default_pos"][:2], w / 2, d / 2, yaw_of_quat(e["default_ori"]))
            labels = sorted(rects)
            for a in labels:
                for b in labels:
                    if a < b:
                        gap = sa.TARGET_GAP if sa.ITEM in a + b else sa.Y_GAP
                        self.assertTrue(pa.apart(rects[a], rects[b], gap - 2e-3), (config, i, a, b))

    def test_receptacles_along_x(self):
        for config, i, lay in self.layouts:
            e = self.entries(lay)
            xs = sorted(e[k]["default_pos"][0] for k in e if k.startswith(sa.RECEPTACLE))
            gap = self.gen.spec["side_gap"]
            self.assertTrue(all(b - a >= gap - 0.02 for a, b in zip(xs, xs[1:])), (config, i))

    def test_reach(self):
        spec = self.gen.spec
        for config, i, lay in self.layouts:
            base = lay["Generalization"]["base"]  # undo the robot base move (pour_beans_plus_layouts.transform)
            pivot = np.r_[sal.pal.plus.ROBOT_ROOT, 0.0]
            back = Rotation.from_euler("z", np.deg2rad(base["yaw"]))
            shift = np.array([base["lateral"], base["distance"], 0.0])
            xy = {k: (back.apply(np.asarray(e["default_pos"]) + shift - pivot) + pivot)[:2]
                  for k, e in self.entries(lay).items()}  # fmt: skip
            for k, e in self.entries(lay).items():
                if k.startswith(sa.RECEPTACLE):
                    lo, hi = spec["receptacle_y"]
                    self.assertTrue(lo - 1e-4 <= xy[k][1] <= hi + 1e-4, (config, i, k, xy[k]))
                if k.startswith(sa.ITEM):
                    own = xy[e["sort"]["receptacle"]]
                    self.assertLessEqual(xy[k][1], own[1] + 1e-4, (config, i, k))  # not behind its receptacle
                    side = sa.side_of(own[0])
                    if side:
                        self.assertGreaterEqual(side * xy[k][0], spec["mid_gap"] - 1e-4, (config, i, k))

    def test_instruction_names_classes_and_receptacles(self):
        langs = set()
        sides = 0
        for config, i, lay in self.layouts:
            s = lay["Generalization"]["sort"]
            text = lay["Instruction"]
            langs.add(s["lang"])
            sides += s["side_words"]
            for c in s["classes"]:
                self.assertTrue(any(n in text for n in sa.class_names(c)[s["lang"]]), (text, c))
            for y, name in s["receptacle_names"].items():
                self.assertIn(name, text)
            self.assertEqual(set(s["receptacle_names"]), set(s["receptacles"]) - {s["spare"]})
        self.assertEqual(langs, {"zh", "en"})
        self.assertTrue(0 < sides < len(self.layouts))

    def test_names_tell_receptacles_apart(self):
        for config, i, lay in self.layouts:
            e = self.entries(lay)
            s = lay["Generalization"]["sort"]
            for y, name in s["receptacle_names"].items():
                others = [self.gen.names_of(e[k], s["lang"]) for k in e if k != y]
                self.assertFalse(any(name == n or name in n for ns in others for n in ns), (y, name))

    def test_levels(self):
        gen = sal.Generator(self.tmp.name, self.pool, self.recs, levels=dict(spatial=1, language=1))
        for i in range(10):
            lay = gen.layout("g1d", 1300, i)
            s = lay["Generalization"]["sort"]
            self.assertIsNone(s["spare"])
            e = self.entries(lay)
            for k in e:
                if k.startswith(sa.RECEPTACLE):
                    yaw = math.degrees(yaw_of_quat(e[k]["default_ori"])) % 180
                    self.assertTrue(min(yaw, 180 - yaw) <= 15 + 6, yaw)  # robot base turns everything ±5 deg

    def test_instruction_text(self):
        rng = random.Random(1)
        side = sa.sides([0.1, -0.1])
        self.assertEqual(side[1][0], "左边的")
        parts = [("fruit", 2, "container", "篮子", 0), ("stationery", 1, "support", "托盘", 1)]
        texts = {sa.instruction(rng, parts, "zh", side) for _ in range(30)}
        self.assertTrue(any("把水果放进右边的篮子里" in t and "把文具放到左边的托盘上" in t for t in texts), texts)
        en = [("fruit", 2, "container", "basket", 0), ("toy", 1, "support", "tray", 1)]
        texts = {sa.instruction(rng, en, "en") for _ in range(30)}
        self.assertIn("Put the fruit in the basket and put the toy on the tray.", texts)


class LayoutsR2(Layouts):
    round_ = f"{sa.TASK}-r2"


class Plugin(unittest.TestCase):
    def episode(self, frames=40, wrong=False, partial=False, pushed=False, holding=False):
        from scripts.datagen.plugins import g1d_sort_anything as plugin

        labels = ["item_0", "item_1", "item_2", "receptacle_0", "receptacle_1"]
        items = [["item_0", "receptacle_0"], ["item_1", "receptacle_0"], ["item_2", "receptacle_1"]]
        receptacles = {"receptacle_0": dict(geo(BOX), kind="container"), "receptacle_1": dict(geo(TRAY))}
        y0 = [-0.12, -0.40, 0.80, *IDENTITY]
        y1 = [0.12, -0.40, 0.78, *IDENTITY]
        start = np.array(
            [[-0.2, -0.5, 0.78, *IDENTITY], [0.0, -0.5, 0.78, *IDENTITY], [0.2, -0.5, 0.78, *IDENTITY], y0, y1]
        )
        size = [0.04, 0.03, 0.04]
        sizes = {x: size for x, _ in items}
        poses = np.repeat(start[None], frames, axis=0).astype(np.float32)
        in_box = y0[2] + BOX["geometry"]["container"]["floor"] + 0.02
        on_tray = y1[2] + TRAY["geometry"]["support"]["floor"] + size[2] / 2 + 0.001
        poses[:, 0, :3] = [-0.15, -0.40, in_box]
        poses[:, 1, :3] = [-0.09, -0.40, in_box]
        poses[:, 2, :3] = [0.12, -0.40, on_tray]
        if wrong:
            poses[:, 2, :3] = [-0.12, -0.42, in_box]
        if partial:
            poses[:, 2, :3] = start[2, :3]
        if pushed:
            poses[-5:, 4, 0] += 0.05
        opening = np.full((frames, 2), 0.2 if holding else 1.0, dtype=np.float32)
        quality = dict(task=plugin.TASK, frames=frames, labels=labels, items=items, receptacles=receptacles,
                       start=start.tolist(), sizes=sizes, classes={"item_0": "fruit", "item_1": "fruit",
                       "item_2": "toy"}, instruction="把水果放进收纳盒里，把玩具放到托盘上。",
                       sort=dict(receptacle_names={"receptacle_0": "收纳盒", "receptacle_1": "托盘"}))  # fmt: skip
        return plugin, dict(object_poses=poses, gripper_openness=opening), quality

    def test_audit_accepts(self):
        plugin, priv, quality = self.episode()
        self.assertEqual(plugin.audit(priv, quality)["items"], 3)

    def test_audit_rejects(self):
        for kwargs in (dict(wrong=True), dict(partial=True), dict(pushed=True), dict(holding=True)):
            plugin, priv, quality = self.episode(**kwargs)
            with self.assertRaises(AssertionError, msg=kwargs):
                plugin.audit(priv, quality)
        plugin, priv, quality = self.episode()
        quality["instruction"] = "把水果放进收纳盒里。"
        with self.assertRaises(AssertionError):
            plugin.audit(priv, quality)

    def test_failing_reasons_and_counts(self):
        cases = {(): (None, 3, 0), ("wrong",): ("wrong_receptacle", 2, 1), ("partial",): ("not_sorted", 2, 0),
                 ("holding",): ("holding", 3, 0)}  # fmt: skip
        for flags, (reason, good, bad) in cases.items():
            plugin, priv, q = self.episode(**dict.fromkeys(flags, True))
            poses = priv["object_poses"][-1]
            args = (q["labels"], q["items"], q["receptacles"], q["start"], q["sizes"])
            self.assertEqual(plugin.failing(poses, *args, priv["gripper_openness"][-1], poses), reason, flags)
            self.assertEqual(plugin.counts(poses, *args), (good, bad), flags)


if __name__ == "__main__":
    unittest.main()
