import unittest

from env.reward_manager.pick_bucket_checks import Progress


class TestProgress(unittest.TestCase):
    def advance(self, s, food, basket, inside, opening, n):
        for _ in range(n):
            s.update(food, basket, inside, opening, [0, 0, basket], food_near=True, handle_near=True)

    def test_full_sequence(self):
        s = Progress()
        for args in [
            (0.03, 0, False, 0.4, 3),
            (0.03, 0, True, 1, 3),
            (0.03, 0, True, 0.2, 3),
            (0.09, 0.06, True, 0.2, 15),
        ]:
            self.advance(s, *args)
        self.assertEqual(s.stage, 4)
        self.advance(s, 0.03, 0, True, 1, 16)
        self.assertEqual(s.stage, 5)

    def test_static_and_out_of_order(self):
        for args in [(0, 0, True, 1), (0.08, 0.06, True, 0.3)]:
            s = Progress()
            self.advance(s, *args, 100)
            self.assertLess(s.stage, 5)

    def test_lost_food_resets_hold(self):
        s = Progress()
        s.stage = 3
        self.advance(s, 0.09, 0.06, True, 0.2, 14)
        self.advance(s, 0.09, 0.06, False, 0.2, 1)
        self.advance(s, 0.09, 0.06, True, 0.2, 14)
        self.assertEqual(s.stage, 3)

    def test_moving_final_pose(self):
        s = Progress()
        s.stage = 4
        for i in range(30):
            s.update(0.03, 0, True, 1, [i * 0.01, 0, 0])
        self.assertEqual(s.stage, 4)

    def test_closed_hand_away_is_not_grasp(self):
        s = Progress()
        for _ in range(30):
            s.update(0.03, 0, False, 0.2, [0, 0, 0])
        self.assertEqual(s.stage, 0)
        s.stage = 2
        for _ in range(30):
            s.update(0.03, 0, True, 0.2, [0, 0, 0])
        self.assertEqual(s.stage, 2)
        for _ in range(30):
            s.update(0.03, 0, True, 0, [0, 0, 0], handle_near=True)
        self.assertEqual(s.stage, 2)

    def test_nonfinite_state_rejected(self):
        s = Progress()
        s.stage = 3
        for _ in range(30):
            s.update(0.03, float("inf"), True, 0.2, [0, 0, 0], handle_near=True)
        self.assertEqual(s.stage, 3)


class TestRecordedAlignment(unittest.TestCase):
    def test_basket_right_and_near(self):
        from scripts.g1d.pick_bucket_alignment import source, xy

        items = source()["object_initialization"]
        food, basket = xy(items["mango"]), xy(items["basket"])
        self.assertGreater(basket[0] - food[0], 0.20)
        self.assertLess(basket[1] - food[1], -0.08)

    def test_torso_origin_not_chassis_origin(self):
        from scripts.g1d.pick_bucket_alignment import AGV_XY, ROOT_XY, init_joints, source

        self.assertAlmostEqual(AGV_XY[1] + 0.0533714301, ROOT_XY[1])
        self.assertAlmostEqual(AGV_XY[0] - 0.000167718, ROOT_XY[0])
        joints = init_joints()
        self.assertAlmostEqual(joints["right_elbow_joint"], source()["initial_state"]["right_arm"]["qpos"][3])
        self.assertNotAlmostEqual(
            joints["LZ_mt_Joint"] + joints["LZ_it_Joint"], source()["initial_state"]["torso"]["height"]
        )

    def test_camera_remains_conditional(self):
        from scripts.g1d.pick_bucket_alignment import source

        camera = source()["head_camera"]
        self.assertFalse(camera["qualified_real_calibration"])
        self.assertIsNone(camera["distortion"])
        self.assertAlmostEqual(camera["K_pixels_640x480"][0][0], 266.43189856215844)
