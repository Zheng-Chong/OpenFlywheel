"""CPU checks of g1d_handover_anything: the X pool, the layouts, the instructions and the plugin's acceptance on
synthetic episodes. pytest scripts/datagen/test_handover_anything.py"""

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import handover_anything_layouts as hal  # noqa: E402
from scripts.datagen.plugins import g1d_handover_anything as plugin  # noqa: E402
from scripts.datagen.test_pick_anything_layouts import fake_assets  # noqa: E402
from scripts.objects import (
    handover_anything as ho,  # noqa: E402
    pick_anything,  # noqa: E402
    place_anything as pa,  # noqa: E402
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


class Pool(unittest.TestCase):
    def test_screening_marks_stay_on_top_of_the_shared_rule(self):
        marked = {(o["cat"], o["idx"]) for o in json.loads(ho.POOL.read_text())["objects"] if o.get("screened_out")}
        kept = {(o["cat"], o["idx"]) for o in ho.load_pool()}
        self.assertFalse(kept & marked)
        # the shared rule no longer leaves game controllers or hammers out of handover; the marks still do
        self.assertTrue(pick_anything.usable([{"cat": "game_controller", "idx": 0}], ho.TASK))
        self.assertFalse({c for c, _ in kept} & {"game_controller", "hammer", "mallet"})
        self.assertNotIn("sand3d_vehicle", {c for c, _ in kept})
        # car parts filed elsewhere are back unless the screening or round 1 took them out (the chain tensioner, 4/4
        # missed; a fuse, 10/126 accepted in round 1)
        self.assertIn(("sand3d_machine_or_industrial", 15), kept)
        self.assertFalse({("sand3d_machine_or_industrial", 35), ("sand3d_fastener_or_hardware", 2)} & kept)

    def test_pool_is_the_rule(self):
        pool = ho.load_pool()
        pick = pick_anything.load_pool()
        everything = json.loads(ho.POOL.read_text())["objects"]
        self.assertEqual(len(everything), len(ho.scan(pick)))
        self.assertEqual(len(pool), sum(not o.get("screened_out") for o in everything))
        self.assertGreater(len(pool), 35)
        for o in pool:
            self.assertFalse(o["cat"].startswith(pa.NONCOMMERCIAL), o)
            self.assertFalse(ho.excluded(o), o)
            self.assertGreaterEqual(ho.long_side(o["size"]), ho.RULES["long"])
            self.assertGreaterEqual(2 * o["shift"], ho.GRIP_GAP)
            self.assertLessEqual(o["shift"] + ho.END, ho.long_side(o["size"]) / 2 + 1e-9)

    def test_screen_out_only_the_unliftable(self):
        objects = [dict(cat=c, idx=0) for c in ("a", "b", "c", "d", "e")]
        stats = {"object": {
            "a/0": {"not_lifted": 3},  # never lifted
            "b/0": {"not_lifted": 2, "accepted": 2},  # lifted often enough
            "c/0": {"wrong_hand": 1, "not_lifted": 1, "not_placed": 1},  # lifted not by the pads
            "d/0": {"not_lifted": 1},  # too few episodes
            "e/0": {"not_placed": 3},  # the expert's placing to fix
        }}  # fmt: skip
        self.assertEqual(ho.screen_out(objects, stats, 0.25, 2), ["a/0", "c/0"])
        self.assertEqual([("screened_out" in o) for o in objects], [True, False, True, False, False])

    def test_screen_out_floor(self):
        objects = [dict(cat=c, idx=0) for c in ("a", "b", "c")]
        stats = {"object": {
            "a/0": {"not_placed": 120, "dropped": 7},  # never accepted, not by the grasp
            "b/0": {"dropped": 90, "accepted": 30},  # 25%: above the floor
            "c/0": {"not_placed": 9},  # too few episodes
        }}  # fmt: skip
        self.assertEqual(ho.screen_out(objects, stats, 0.25, 50, 0.10, "round 1"), ["a/0"])
        self.assertEqual(objects[0]["screened_out"], "accepted in 0/127 round 1 episodes (under 10%)")

    def test_held(self):
        self.assertTrue(ho.held(0.3, [60.0, 60.0]))
        self.assertTrue(ho.held(0.57, [59.9, 0.0]))  # one pad's sensor pressed, the other finger lower (pilot)
        self.assertFalse(ho.held(0.05, [0.0, 0.0]))  # a thin handle pinched at the fingertips
        self.assertFalse(ho.held(0.0, [30.0, 30.0]))  # shut on nothing (on its own fingers)

    def test_shift(self):
        self.assertAlmostEqual(ho.shift([0.03, 0.12, 0.02]), 0.035)
        self.assertAlmostEqual(ho.shift([0.22, 0.03, 0.02]), 0.055)
        self.assertAlmostEqual(ho.shift([0.10, 0.03, 0.02]), 0.035)

    def test_instruction_names_the_hands(self):
        names = {lang: {"x": ["笔" if lang == "zh" else "pen"], "y": ["托盘" if lang == "zh" else "tray"]}
                 for lang in ("zh", "en")}  # fmt: skip
        seen = set()
        for i in range(200):
            text = ho.instruction(random.Random(i), names, "left")
            seen.add(text)
            self.assertTrue("右手" in text or "right hand" in text, text)
            self.assertTrue(("笔" in text and "托盘" in text) or ("pen" in text and "tray" in text), text)
        self.assertGreaterEqual(len(seen), 8)


class Layouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fake_assets(cls.tmp.name, pick_anything.load_pool())
        recs = pa.load_receptacles(hal.RECEPTACLES)
        fake_assets(
            cls.tmp.name, [dict(cat=r["cat"], idx=r["idx"], size=r["size"], mass=r["mass"] or 0.2) for r in recs]
        )
        cls.gen = hal.Generator(cls.tmp.name)
        cls.layouts = [(c, i, cls.gen.layout(c, 1300, i)) for c in hal.pal.CONFIGS for i in range(60)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reproducible(self):
        self.assertEqual(self.gen.layout("g1d", 1300, 3), self.layouts[3][2])
        self.assertNotEqual(self.gen.layout("g1d", 1301, 3), self.layouts[3][2])

    def test_sides_and_fit(self):
        givers = set()
        recs = {r["id"]: r for r in self.gen.receptacles}
        for _, _, lay in self.layouts:
            g = lay["Generalization"]
            entries = {e["label"]: e for es in lay["Rigid"].values() for e in es}
            x, y = entries["target"], entries[pa.LABEL]
            givers.add(g["giver"])
            self.assertEqual(ho.hand_of(x["default_pos"][0]), g["giver"])
            self.assertEqual(ho.hand_of(y["default_pos"][0]), g["taker"])
            self.assertEqual(y["receptacle"]["kind"], ho.KIND)
            rec = recs[g["receptacle"]]
            self.assertEqual(rec["split"][ho.KIND], "train")
            self.assertTrue(pa.fits_some_yaw(rec["geometry"][ho.KIND], x["physics"]["size"], ho.KIND))
            for label, e in entries.items():
                if label.startswith("distractor") and "receptacle" not in e:
                    self.assertLessEqual(e["physics"]["size"][2], ho.DISTRACTOR_TOP * 1.15 + 1e-6)
        self.assertEqual(givers, set(ho.HANDS))

    def test_every_x_comes_up(self):
        targets = {lay["Generalization"]["target"] for _, _, lay in self.layouts}
        self.assertGreaterEqual(len(targets), min(len(self.gen.order), 50))


def episode(giver="left", drop=False, clash=False, one_hand=False, wrong=False):
    """Synthetic frames: X (pen, 18 x 2 cm) lifted by the giver, held by both over the midline, carried by the
    taker onto Y, let go and resting there: (privileged, quality)."""
    side = -1.0 if giver == "left" else 1.0
    x0, y_xy, top = np.array([0.15 * side, -0.42, 0.725]), np.array([-0.15 * side, -0.42]), 0.715
    geom = dict(region=dict(center=[0.0, 0.0], half=[0.12, 0.08], radius=0.08), floor=0.005, rim=0.005, top=0.012)
    geom.update(bottom=-0.006, kind=ho.KIND, id="RoboDojo/Rigid/tray/00000")
    y0 = np.r_[y_xy, top + 0.006]
    labels = sorted(["target", pa.LABEL, "distractor_0"])
    start = {"target": x0, pa.LABEL: y0, "distractor_0": np.array([0.0, -0.31, 0.74])}
    poses, opening, forces, gaps = [], [], [], []
    g, t = (0, 1) if giver == "left" else (1, 0)

    def frame(x, hold_g, hold_t, open_g=0.3, open_t=0.3, gap=0.12):
        p = np.array([np.r_[x if k == "target" else start[k], IDENTITY] for k in labels])
        f = np.zeros((2, 2))
        f[g], f[t] = (10.0 if hold_g else 0.0), (10.0 if hold_t else 0.0)
        o = np.zeros(2)
        o[g], o[t] = open_g, open_t
        poses.append(p), forces.append(f), opening.append(o), gaps.append(gap)

    for _ in range(20):  # reach, grasp
        frame(x0, False, False, 1.0, 1.0)
    for k in range(30):  # lift and carry to the midline
        s = (k + 1) / 30
        x = x0 + s * (np.array([0.0, -0.42, 0.90]) - x0)
        lift_g, lift_t = (not wrong, wrong)
        frame(x, lift_g, lift_t, 0.3, 0.3 if wrong else 1.0)
    mid = np.array([0.0, -0.42, 0.90])
    for _ in range(12):  # both hold
        frame(mid, True, not one_hand, 0.3, 0.3 if not one_hand else 1.0, 0.02 if clash else 0.12)
    target = np.r_[y_xy, top + 0.012 + 0.01]
    for k in range(30):  # the taker carries; the giver has let go
        s = (k + 1) / 30
        x = mid + s * (target - mid)
        if drop and k > 5:
            x = np.r_[mid[:2] + [0.05 * side * -1, 0.0], top + 0.01]
        frame(x, False, not drop, 1.0, 0.3)
    rest = target if not drop else np.r_[mid[:2] + [0.05 * side * -1, 0.0], top + 0.01]
    for _ in range(40):  # let go, resting
        frame(rest, False, False, 1.0, 1.0)
    priv = dict(
        object_poses=np.asarray(poses, dtype=np.float32),
        gripper_openness=np.asarray(opening, dtype=np.float32),
        pad_forces=np.asarray(forces, dtype=np.float32),
        hand_gap=np.asarray(gaps, dtype=np.float32),
    )
    quality = dict(
        task=ho.TASK, frames=len(poses), labels=labels, receptacle=geom, target_size=[0.02, 0.18, 0.02],
        start=[np.r_[start[k], IDENTITY].tolist() for k in labels], hands=list(ho.HANDS), giver=giver,
        taker=ho.other(giver), usable_names={"zh": ["笔"], "en": ["pen"]}, usable_receptacle_names={"zh": ["托盘"]},
        instruction=f"把笔交给{ho.NAMES['zh'][ho.other(giver)]}，然后放到托盘上",
    )  # fmt: skip
    return priv, quality


class Acceptance(unittest.TestCase):
    def test_accepts_a_handover(self):
        for giver in ho.HANDS:
            priv, quality = episode(giver)
            report = plugin.audit(priv, quality)
            self.assertIsNotNone(report["stage_frames"]["handed"])
            self.assertLess(report["stage_frames"]["handed"], report["stage_frames"]["placed"])

    def test_rejects(self):
        for kwargs, why in (
            (dict(drop=True), "dropped"),
            (dict(clash=True), "clash"),
            (dict(one_hand=True), "not_handed"),
            (dict(wrong=True), "wrong_hand"),
        ):
            priv, quality = episode(**kwargs)
            with self.assertRaises(AssertionError, msg=why):
                plugin.audit(priv, quality)
            tracker = plugin.Tracker(quality["labels"], quality["start"], quality["receptacle"],
                                     quality["target_size"], quality["hands"], quality["giver"])  # fmt: skip
            fails = [
                tracker.update(i, priv["object_poses"][i], priv["gripper_openness"][i], priv["pad_forces"][i],
                               priv["hand_gap"][i])
                for i in range(quality["frames"])
            ]  # fmt: skip
            self.assertEqual(fails[-1], why, kwargs)

    def test_audit_needs_the_taker_named(self):
        priv, quality = episode("left")
        quality["instruction"] = "把笔放到托盘上"
        with self.assertRaises(AssertionError):
            plugin.audit(priv, quality)


if __name__ == "__main__":
    unittest.main()
