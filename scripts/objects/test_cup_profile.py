"""CPU checks of the cup profiles and pool (no Isaac): python -m pytest scripts/objects/test_cup_profile.py"""

import sys
import types
import unittest
from unittest import mock

import numpy as np

from scripts.objects import cup_pool, cup_profile

R_OUT, R_IN, HEIGHT, FLOOR = 0.035, 0.032, 0.09, 0.004  # a straight cup, root at its centre


def straight_cup(handle=False):
    """Height map of a straight cup as check_cup_holds_beans.py measures it (1 mm cells)."""
    xs = ys = np.arange(-0.06, 0.0605, 0.001)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    d = np.hypot(gx, gy)
    bottom, top = -HEIGHT / 2, HEIGHT / 2
    height = np.full(d.shape, np.nan)
    height[d <= R_OUT] = top
    cavity = d < R_IN
    height[cavity] = bottom + FLOOR
    if handle:
        height[(gx > R_OUT) & (gx < 0.055) & (np.abs(gy) < 0.006)] = top - 0.02
    level = np.full(d.shape, top)
    return cup_profile.profile(xs, ys, height, level, cavity, bottom)


def flared_cup(r_floor=0.02, r_rim=0.04, wall=0.003):
    """Height map of a cup flaring from r_floor to r_rim: from above only its rim and the inside are seen."""
    xs = ys = np.arange(-0.06, 0.0605, 0.001)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    d = np.hypot(gx, gy)
    bottom, top = -HEIGHT / 2, HEIGHT / 2
    height = np.full(d.shape, np.nan)
    height[d <= r_rim + wall] = top
    cavity = d < r_rim
    rise = np.clip((d - r_floor) / (r_rim - r_floor), 0, 1) * (HEIGHT - FLOOR)
    height[cavity] = (bottom + FLOOR + rise)[cavity]
    level = np.full(d.shape, top)
    return cup_profile.profile(xs, ys, height, level, cavity, bottom), (r_floor, r_rim, wall)


class CupProfileTest(unittest.TestCase):
    def test_a_flared_body_follows_its_wall_under_the_rim(self):
        cup, (r_floor, r_rim, wall) = flared_cup()
        mid = (r_floor + r_rim) / 2 + wall  # the outer wall halfway up, which the rays from above do not see
        self.assertAlmostEqual(cup_profile.body_radius(cup, 0.0), mid, delta=0.002)
        self.assertAlmostEqual(cup_profile.body_radius(cup, cup["top"]), r_rim + wall, delta=0.0015)
        straight = straight_cup()
        self.assertAlmostEqual(cup_profile.body_radius(straight, -0.03), R_OUT, delta=0.0015)
        spawned = cup_profile.column(cup, 200)
        self.assertTrue(cup_profile.in_cavity(spawned, [0, 0, 0], [1, 0, 0, 0], cup).all())

    def test_profile_of_a_straight_cup(self):
        cup = straight_cup()
        self.assertLess(np.abs(cup["axis"]).max(), 0.0011)
        self.assertAlmostEqual(cup["z"][0], -HEIGHT / 2 + FLOOR, places=4)
        self.assertAlmostEqual(cup["z"][-1], HEIGHT / 2, places=4)
        self.assertAlmostEqual(np.median(cup["r"]), R_IN, delta=0.0015)
        self.assertAlmostEqual(cup_profile.body_radius(cup, 0.0), R_OUT, delta=0.0015)
        self.assertIsNone(cup["handle"])
        volume = np.pi * R_IN**2 * (HEIGHT - FLOOR)
        self.assertAlmostEqual(cup_profile.volume(cup) / volume, 1.0, delta=0.08)
        self.assertAlmostEqual(cup_profile.volume(cup, 0.5) / cup_profile.volume(cup), 0.125, places=6)

    def test_handle_is_found_and_left_out_of_the_body(self):
        cup = straight_cup(handle=True)
        self.assertIsNotNone(cup["handle"])
        self.assertLess(min(cup["handle"], 360 - cup["handle"]), 15)  # along +x
        self.assertAlmostEqual(cup_profile.body_radius(cup, 0.0), R_OUT, delta=0.0015)

    def test_cavity_and_column(self):
        cup = straight_cup()
        position, upright = np.array([0.1, -0.3, 0.8]), [1.0, 0.0, 0.0, 0.0]
        beans = cup_profile.column(cup, 400)
        self.assertEqual(beans.shape, (400, 3))
        self.assertTrue(cup_profile.in_cavity(beans + position, position, upright, cup).all())
        self.assertLess(beans[:, 2].max(), cup["z"][-1])
        outside = position + [[0.05, 0, 0], [0, 0, 0.06], [0, 0, -0.05]]
        self.assertFalse(cup_profile.in_cavity(outside, position, upright, cup).any())
        self.assertFalse(cup_profile.in_cavity(beans + position, position + [0.08, 0, 0], upright, cup).any())
        bigger = cup_profile.column(cup, 400, scale=1.2)
        self.assertTrue(cup_profile.in_cavity(bigger + position, position, upright, cup, 1.2).all())

    def test_touching(self):
        cup = straight_cup()
        upright = [1.0, 0.0, 0.0, 0.0]
        a = (np.array([0.0, 0.0, 0.0]), upright)
        self.assertTrue(cup_profile.touching(a, cup, (np.array([0.069, 0, 0]), upright), cup))
        self.assertFalse(cup_profile.touching(a, cup, (np.array([0.073, 0, 0]), upright), cup))
        self.assertFalse(cup_profile.touching(a, cup, (np.array([0.0, 0, 0.1]), upright), cup))
        # Scaled cups: 0.85x top is +0.03825, bottom is -0.03825; 10 mm vertical gap (dz = 0.0865) must not touch,
        # while 1.15x horizontal overlap (dx = 0.078 < 2 * 0.035 * 1.15 = 0.0805) must touch.
        self.assertFalse(cup_profile.touching(a, cup, (np.array([0.0, 0, 0.0865]), upright), cup, 0.85, 0.85))
        self.assertTrue(cup_profile.touching(a, cup, (np.array([0.078, 0, 0]), upright), cup, 1.15, 1.15))

    def test_refine_from_mesh_recovers_inward_belly_and_outward_bulge(self):
        # A wine-bowl / barrel mug: rim inner radius 28 mm, mid-belly inner radius 40 mm, wall thickness 3 mm.
        # Top-down rays only see the 28 mm opening; refine_from_mesh slices the 3D mesh and recovers both.
        xs = ys = np.arange(-0.06, 0.0605, 0.001)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        d = np.hypot(gx, gy)
        bottom, top, floor = -0.045, 0.045, -0.040
        height = np.full(d.shape, np.nan)
        height[d <= 0.043] = 0.0  # outer belly seen around the rim
        height[d <= 0.031] = top
        cavity = d < 0.028
        height[cavity] = floor
        raw = cup_profile.profile(xs, ys, height, np.full(d.shape, top), cavity, bottom)
        self.assertLess(max(raw["r"]), 0.030)

        n_th = 48
        th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
        zs = np.linspace(floor, top, 19)
        r_in_z = 0.028 + 0.012 * np.sin(np.pi * (zs - floor) / (top - floor))
        r_out_z = r_in_z + 0.003
        pts, tris = [], []

        def add_tube(rz, z_vals):
            base = len(pts)
            for zi, ri in zip(z_vals, rz):
                pts.extend((ri * np.cos(t), ri * np.sin(t), zi) for t in th)
            for k in range(len(z_vals) - 1):
                for j in range(n_th):
                    jn = (j + 1) % n_th
                    r0, r1 = base + k * n_th, base + (k + 1) * n_th
                    a, b, c, e = r0 + j, r0 + jn, r1 + jn, r1 + j
                    tris.extend(([a, b, c], [a, c, e]))

        add_tube(r_in_z, zs)
        add_tube(np.r_[r_out_z[0], r_out_z], np.r_[bottom, zs])
        c_idx = len(pts)
        pts.append((0.0, 0.0, floor))
        for j in range(n_th):
            tris.append([c_idx, j, (j + 1) % n_th])

        refined = cup_profile.refine_from_mesh(raw, np.asarray(pts), np.asarray(tris))
        mid = len(refined["z"]) // 2
        self.assertAlmostEqual(refined["r"][mid], 0.040, delta=0.0015)
        self.assertAlmostEqual(cup_profile.body_radius(refined, 0.0025), 0.043, delta=0.0015)
        belly_bean = np.array([[0.036, 0.0, 0.0025]])
        self.assertFalse(cup_profile.in_cavity(belly_bean, [0, 0, 0], [1, 0, 0, 0], raw).any())
        self.assertTrue(cup_profile.in_cavity(belly_bean, [0, 0, 0], [1, 0, 0, 0], refined).all())

    def test_fill_need(self):
        self.assertAlmostEqual(cup_profile.fill_need(0.8, 500.0, 900), 400.0)  # of the target's capacity
        self.assertAlmostEqual(cup_profile.fill_need(0.8, 500.0, 300), 240.0)  # of the beans, too few to fill it
        self.assertEqual(cup_profile.fill_need(0.8, 500.0, 0), 0.0)


class PourCheckTest(unittest.TestCase):
    """func_parser.is_fluid_in_cup_profile: the share of every bean, or a layout's target fill of the cup."""

    def check(self, inside, total, layout, share=0.95):
        try:
            import torch  # noqa: F401

            stub = {}
        except ImportError:  # CPU CI has no torch; func_parser only imports it
            stub = {"torch": types.ModuleType("torch")}
        with mock.patch.dict(sys.modules, stub):
            from env.reward_manager.func_parser import Func_Parser

            cup, position = straight_cup(), np.array([0.1, -0.3, 0.8])
            rng = np.random.default_rng(0)  # bean centres well inside the cavity, denser than a column packs
            angle, radius = rng.uniform(0, 2 * np.pi, inside), 0.8 * R_IN * np.sqrt(rng.uniform(0, 1, inside))
            depth = rng.uniform(cup["z"][0] + 0.002, cup["z"][-1] - 0.002, inside)
            cavity = np.stack([radius * np.cos(angle), radius * np.sin(angle), depth], axis=1) + [*cup["axis"][:2], 0]
            points = np.vstack([cavity + position, np.tile(position + [0.3, 0, 0], (total - inside, 1))])
            fluid = types.SimpleNamespace(
                get_particle_positions=lambda: [points], init_ori=[1.0, 0, 0, 0], init_pos=[0.0, 0, 0]
            )
            target = types.SimpleNamespace(instance_config={"cavity": cup}, scale=(1.0, 1.0, 1.0))
            lm = types.SimpleNamespace(
                saved_layouts=[layout],
                get_instance_name=lambda label, env_idx: label,
                get_scene_object=lambda inst_name, env_idx: fluid if inst_name == "beans" else target,
                get_instance_pose=lambda inst_name, env_idx: (position, [1.0, 0, 0, 0]),
            )
            parser = Func_Parser.__new__(Func_Parser)
            parser.layout_manager = lm
            args = dict(env_idx=0, label_A="beans", label_B="target_cup", share=share)
            return parser.is_fluid_in_cup_profile(args)

    def test_without_target_fill_the_share_of_every_bean(self):
        for layout in (None, {}, {"Generalization": {"grasp_lag": 3}}, {"Generalization": {"pour": {}}}):
            self.assertEqual(self.check(96, 100, layout), 1.0)
            self.assertEqual(self.check(94, 100, layout), 0.0)

    def test_target_fill_of_the_cup_or_of_the_beans(self):
        layout = {"Generalization": {"pour": {"source_fill": 0.9, "target_fill": 0.8}}}
        cap = cup_profile.capacity(straight_cup())
        need = int(np.ceil(0.8 * cap))
        total = int(cap * 1.2)  # more beans than the cup holds: 80% of the cup, beans left behind are fine
        self.assertEqual(self.check(need, total, layout), 1.0)
        self.assertEqual(self.check(need - 1, total, layout), 0.0)
        self.assertEqual(self.check(need, total, {}), 0.0)  # the old rule wants 95% of them
        few = int(cap / 2)  # fewer beans than the cup holds: 80% of the beans
        self.assertEqual(self.check(int(np.ceil(0.8 * few)), few, layout), 1.0)
        self.assertEqual(self.check(int(np.ceil(0.8 * few)) - 1, few, layout), 0.0)


class CupPoolTest(unittest.TestCase):
    def test_eval_split_takes_whole_families_and_a_share_of_the_rest(self):
        cups = {}
        for family, n in (("mug", 20), ("plastic_cup", 10), ("wave_curve_cup", 4), ("simready_glass", 4), ("x", 2)):
            for i in range(n):
                cups[f"Real2Sim/Rigid/{family}/{i:05d}"] = {"family": family}
        cups[cup_pool.TWIN] = {"family": "plastic_cup"}
        held = cup_pool.choose_eval(cups)
        families = [cups[c]["family"] for c in held]
        self.assertEqual(families.count("wave_curve_cup"), 4)
        self.assertEqual(families.count("simready_glass"), 4)
        self.assertEqual(families.count("x"), 0)  # too few to give any away
        self.assertNotIn(cup_pool.TWIN, held)
        self.assertEqual(len(held), round(cup_pool.SHARE * len(cups)))
        self.assertEqual(held, cup_pool.choose_eval(cups))


if __name__ == "__main__":
    unittest.main()
