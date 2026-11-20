"""CPU checks of the handle measurement and grasp (no Isaac): python -m pytest scripts/objects/test_cup_handle.py"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1d import scripted_demo as demo
from scripts.objects import cup_handle
from scripts.objects.test_cup_profile import HEIGHT, R_OUT, straight_cup


def box(lo, hi):
    """(12, 3, 3) triangles of the box between corners `lo` and `hi`."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([c[list(t)] for f in faces for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))])


def mug(hole=0.03, tall=0.04, bar=0.01, side=0.006):
    """Triangles of a straight cup (R_OUT, HEIGHT) with a square handle loop out along +x: a hole `hole` across and
    `tall` high between the body and a `bar` thick outer bar, `side` either side of the handle's plane."""
    t = np.linspace(0, 2 * np.pi, 49)
    ring = np.c_[R_OUT * np.cos(t), R_OUT * np.sin(t)]
    lo, hi = -HEIGHT / 2, HEIGHT / 2
    wall = [
        tri
        for a, b in zip(ring[:-1], ring[1:])
        for tri in (((*a, lo), (*b, lo), (*b, hi)), ((*a, lo), (*b, hi), (*a, hi)))
    ]
    caps = [((0, 0, z), (*a, z), (*b, z)) for a, b in zip(ring[:-1], ring[1:]) for z in (lo, hi)]
    out, arm = R_OUT + hole + bar, 0.008
    handle = [
        box((R_OUT - 0.002, -side, tall / 2), (out, side, tall / 2 + arm)),
        box((R_OUT - 0.002, -side, -tall / 2 - arm), (out, side, -tall / 2)),
        box((R_OUT + hole, -side, -tall / 2 - arm), (out, side, tall / 2 + arm)),
    ]
    return np.concatenate([np.array(wall), np.array(caps), *handle])


class Lm:
    """The layout manager calls handle_grasp makes, for one cup at `pos` turned `yaw` deg."""

    def __init__(self, pos, yaw):
        self.pos, self.quat = np.asarray(pos, float), np.roll(Rotation.from_euler("z", yaw, degrees=True).as_quat(), 1)

    def get_instance_pose(self, env, label):
        return self.pos, self.quat


class CupHandleTest(unittest.TestCase):
    def test_a_finger_sized_hole_is_measured(self):
        profile = straight_cup(handle=True)
        h = cup_handle.handle(mug(), profile)
        self.assertIsNotNone(h)
        self.assertLess(min(h["yaw"], 360 - h["yaw"]), 15)
        self.assertAlmostEqual(h["bar"][0], R_OUT + 0.03, delta=0.0015)
        self.assertAlmostEqual(h["bar"][1], R_OUT + 0.04, delta=0.0015)
        self.assertAlmostEqual(h["z"], 0.0, delta=0.0015)
        self.assertAlmostEqual(h["hole"][1] - h["hole"][0], cup_handle.FINGER[1], places=4)
        self.assertAlmostEqual(h["room"][0], 0.03, delta=0.002)
        self.assertAlmostEqual(h["room"][1], 0.04 - 0.0015, delta=0.0015)  # the raster closes ~1 mm off each edge
        self.assertAlmostEqual(h["width"], 0.012, delta=0.0015)
        self.assertTrue(demo.handle_fits(h, 1.0))
        self.assertFalse(demo.handle_fits(h, 0.8))  # 30 mm high: too low for the tilted 30 mm finger

    def test_a_hole_the_finger_misses_is_none(self):
        profile = straight_cup(handle=True)
        self.assertIsNone(cup_handle.handle(mug(hole=0.014), profile))  # narrower than the 15.5 mm finger
        self.assertIsNone(cup_handle.handle(mug(tall=0.028), profile))  # lower than the 30 mm finger
        self.assertIsNone(cup_handle.handle(mug(), straight_cup()))  # no handle in the profile

    def test_handle_grasp_takes_the_bar_with_the_fingers_through_the_hole(self):
        profile = straight_cup(handle=True)
        h = cup_handle.handle(mug(), profile)
        for out, yaw in ((1, 0.0 - h["yaw"] + 10), (-1, 180.0 - h["yaw"] - 10)):
            pos = np.array([0.15 * out, -0.35, 0.8])
            grip, hand, rim, _, opening = demo.handle_grasp(Lm(pos, yaw), out, "source_cup", profile, 1.0, h)
            along = Rotation.from_euler("z", yaw + h["yaw"], degrees=True).apply([1.0, 0.0, 0.0])
            bar = pos + along * (h["bar"][0] + h["bar"][1]) / 2 + [0, 0, h["z"]]
            np.testing.assert_allclose(grip, bar, atol=1e-3)
            fingers, closing = hand.apply([1.0, 0.0, 0.0]), hand.apply([0.0, 1.0, 0.0])
            self.assertAlmostEqual(abs(fingers @ along), 0.0, places=6)  # through the hole
            self.assertAlmostEqual(fingers[2], -np.sin(np.radians(demo.HANDLE_PITCH)), places=6)
            self.assertAlmostEqual(abs(closing @ along), 1.0, places=6)
            self.assertAlmostEqual(closing[2], 0.0, places=6)
            self.assertGreater(fingers[1], 0.9)  # forward, as the jug's
            gap = demo.PAD_CLOSED + opening * demo.PAD_TRAVEL
            spare = (gap - (h["bar"][1] - h["bar"][0])) / 2
            self.assertGreater(spare, 0.0)
            inner_back = (h["bar"][0] + h["bar"][1]) / 2 - gap / 2 - cup_handle.FINGER[0]
            self.assertGreater(inner_back, R_OUT)  # the inner finger clears the body
            self.assertAlmostEqual(rim, pos[2] + profile["top"], places=6)


if __name__ == "__main__":
    unittest.main()
