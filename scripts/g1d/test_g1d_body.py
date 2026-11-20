"""CPU checks for the G1-D body (waist, lift, chassis): action keys <-> joints and the mobile URDF (no Isaac)."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from env.robot_manager.robot_class.g1d import BASE_LEAD, G1DLeft
from env.robot_manager.robot_config._g1d_trim import (
    BLACK_FACES,
    LIDAR,
    SCREW_PARTS,
    SCREWS,
    WINDOWS,
    bumper_mesh,
    lidar_mesh,
    panel_meshes,
    screw_mesh,
)
from scripts.g1d.build_asset import V4_URDF, mobile_urdf

ROOT = Path(__file__).resolve().parents[2]
DT = 1 / 300


def robot(mobile):
    r = G1DLeft.__new__(G1DLeft)  # body_state / body_targets only read `mobile`
    r.mobile = mobile
    return r


class BodyTests(unittest.TestCase):
    def test_tabletop_waist_and_lift(self):
        command = {"waist_joint_state": [0.3, 0.5], "lift_joint_state": [0.2]}
        pos, vel = robot(False).body_targets(command, np.zeros(4), np.zeros(4), DT)
        np.testing.assert_allclose(pos, [0.3, 0.5, 0.1, 0.1])  # torso_Joint (yaw), Yaw_Joint (pitch), two stages
        np.testing.assert_allclose(vel, 0)
        state = robot(False).body_state(pos, vel)
        self.assertEqual(state, {"waist_joint_state": [0.3, 0.5], "lift_joint_state": [0.2]})

    def test_chassis_drives_along_its_heading(self):
        target = np.array([0, 0, 0.1, 0.1, 0.0, 0.0, np.pi / 2])  # facing +y of the start frame
        command = {"waist_joint_state": [0, 0], "lift_joint_state": [0.2], "base_velocity": [1.0, 0.3]}
        pos, vel = robot(True).body_targets(command, target, target, DT)
        np.testing.assert_allclose(vel[4:], [0, 1.0, 0.3], atol=1e-12)
        np.testing.assert_allclose(pos[4:], target[4:] + vel[4:] * DT)
        state = robot(True).body_state(pos, vel)
        np.testing.assert_allclose(state["base_velocity"], [1.0, 0.3], atol=1e-5)  # chassis frame, yaw one step on
        np.testing.assert_allclose(state["base_pose"], pos[4:])

    def test_chassis_clips_to_official_limits_and_stalls_when_blocked(self):
        target = np.zeros(7)
        command = {"waist_joint_state": [0, 0], "lift_joint_state": [0], "base_velocity": [5.0, -2.0]}
        _, vel = robot(True).body_targets(command, target, target, DT)
        np.testing.assert_allclose(vel[4:], [1.5, 0, -0.6])
        blocked = robot(True).body_targets(command, target + [0, 0, 0, 0, 0.5, 0, 0], target, 1.0)[0]
        np.testing.assert_allclose(blocked[4:], target[4:] + [BASE_LEAD[0], 0, -BASE_LEAD[2]])

    def test_mobile_urdf_puts_the_chassis_on_planar_joints(self):
        urdf = mobile_urdf(ET.parse(ROOT / "real2sim/robot/g1d_v4" / V4_URDF).getroot())
        joints = {j.get("name"): j for j in urdf.findall("joint")}
        chain = [
            (joints[n].find("parent").get("link"), joints[n].find("child").get("link"))
            for n in ("base_x_joint", "base_y_joint", "base_yaw_joint")
        ]
        self.assertEqual(
            chain, [("base_world", "base_x_link"), ("base_x_link", "base_y_link"), ("base_y_link", "AGV_link")]
        )
        children = {j.find("child").get("link") for j in urdf.findall("joint")}
        self.assertEqual({link.get("name") for link in urdf.findall("link")} - children, {"base_world"})
        for name in ("Left_Wheel_Link", "RIght_Wheel_Link"):
            self.assertIsNone(urdf.find(f"link[@name='{name}']/collision"))
        self.assertIsNotNone(urdf.find("link[@name='AGV_link']/collision"))

    def triangles(self, link):
        stl = (ROOT / f"real2sim/robot/g1d_v4/meshes/{link}.STL").read_bytes()
        if stl.startswith(b"version https://git-lfs"):
            self.skipTest(f"{link}.STL is an LFS pointer")
        n = int.from_bytes(stl[80:84], "little")
        rec = np.frombuffer(stl[84 : 84 + 50 * n], np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        return rec["v"].astype(float)

    def test_black_faces_are_the_elbow_cap_and_forearm_ring(self):
        for side, inner in (("left", -1), ("right", 1)):
            c = self.triangles(f"{side}_shoulder_yaw_link").mean(1)
            cap = c[BLACK_FACES["shoulder_yaw_link"](c)]
            self.assertGreater(len(cap), 1000)
            self.assertTrue(np.all(inner * cap[:, 1] > 0))  # only the boss facing the torso
            self.assertGreater(inner * cap[:, 1].mean(), 0.035)  # it stands out of the 2.8 cm side face
            c = self.triangles(f"{side}_elbow_link").mean(1)
            ring = c[BLACK_FACES["elbow_link"](c)]
            self.assertGreater(len(ring), 100)
            self.assertLess(len(ring), len(c) / 2)

    def test_wrist_screws_sit_on_the_faces_the_head_camera_sees(self):
        for suffix, sides, centres, radius in SCREWS:
            for side in sides:
                parts = screw_mesh(self.triangles(f"{side}_{suffix}"), centres, radius)
                self.assertEqual(set(parts), set(SCREW_PARTS))
                heights = {}
                for name, (points, counts, indices, normals) in parts.items():
                    scale, lift, corners, lean = SCREW_PARTS[name]
                    ring = corners + 1
                    self.assertEqual(len(points), len(centres) * ring)  # every centre lands on the link
                    self.assertEqual(sum(counts), len(indices))
                    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
                    self.assertTrue(np.all(normals[::ring, 2] > 0.9))  # on the +z face, not a side wall
                    np.testing.assert_allclose(np.linalg.norm(points[1:ring] - points[0], axis=1), scale * radius)
                    top = 0.0285 if suffix == "wrist_pitch_link" else 0.0305
                    self.assertTrue(np.all((points[::ring, 2] > 0.02) & (points[::ring, 2] < top + 0.001)))
                    heights[name] = points[::ring, 2]
                self.assertTrue(
                    np.all(heights["hole"] < heights["head"]) and np.all(heights["head"] < heights["socket"])
                )

    def test_lidar_sits_on_the_chassis_in_front_of_the_column(self):
        points, counts, indices, normals = lidar_mesh()
        self.assertEqual(sum(counts), len(indices))
        self.assertEqual(set(indices.tolist()), set(range(len(points))))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        outward = np.einsum("ij,ij->i", normals[:, :2], points[:, :2] - LIDAR["base"][:2])
        self.assertTrue(np.all((outward > -1e-9) & (normals[:, 2] > -0.2)))  # out and up (the bulge's foot dips)
        lo, hi = points.min(0), points.max(0)
        top = self.triangles("AGV_link")[..., 2].max()
        self.assertAlmostEqual(lo[2], top - 0.0015, delta=1e-3)  # seated in the top face
        self.assertAlmostEqual(hi[2] - lo[2], 0.041)
        column = self.triangles("LZ_ot_Link")[..., 0].max()  # LZ_ot_Link sits on the chassis axis
        self.assertGreater(lo[0], column)
        self.assertLess(hi[0], 0.24)  # on the flat of the top face, short of its rounded front edge
        self.assertAlmostEqual(lo[1] + hi[1], 0.0)

    def test_bumper_stands_out_of_the_shell_to_the_official_length(self):
        points, counts, indices, normals = bumper_mesh()
        self.assertEqual(sum(counts), len(indices))
        self.assertEqual(set(indices.tolist()), set(range(len(points))))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        lo, hi = points.min(0), points.max(0)
        self.assertAlmostEqual(hi[0] - lo[0], 0.557, delta=0.005)  # official chassis length
        self.assertLess(hi[1] - lo[1], 0.525)  # within the official width (the wheel arches)
        shell = self.triangles("AGV_link")
        self.assertGreater(hi[0], shell[..., 0].max() + 0.005)
        self.assertLess(lo[2], shell[..., 2].min())  # down over the shell's lower edge
        self.assertLess(hi[2], -0.038 - 0.01)  # silver band below the front panel

    def test_front_panel_wraps_over_the_top_edge_and_carries_the_windows(self):
        shell = self.triangles("AGV_link")
        (points, counts, indices, normals), windows = panel_meshes(shell)
        self.assertEqual(sum(counts), len(indices))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        np.testing.assert_allclose([points[:, 1].min(), points[:, 1].max()], [-0.16, 0.16], atol=1e-3)
        centre = points[np.abs(points[:, 1]) < 1e-9]
        front, lip = centre[centre[:, 2] < 0.08], centre[centre[:, 2] > 0.0855]
        np.testing.assert_allclose(front[:, 0], 0.2725 + 0.0015, atol=5e-4)  # 1.5 mm off the front face
        self.assertAlmostEqual(front[:, 2].min(), -0.036)  # above the bumper (top at -0.057)
        self.assertLess(lip[:, 0].min(), 0.255)  # runs onto the top face
        self.assertLess(lip[:, 2].max(), 0.0855 + 0.002)
        wp, wc, wi, wn = windows
        self.assertEqual(len(wc), len(wi) // 3)
        self.assertEqual(len(wc), 2 * len(WINDOWS) * 4 * 8)  # mirrored, 8 points per rounded corner, fanned
        self.assertTrue(np.all(np.abs(wp[:, 1]) < 0.16) and np.all((wp[:, 2] > -0.036) & (wp[:, 2] < 0.08)))
        face = points[:65]  # the panel's lowest row, along its width
        self.assertTrue(np.all(wp[:, 0] > np.interp(wp[:, 1], face[:, 1], face[:, 0])))  # in front of the panel


if __name__ == "__main__":
    unittest.main()
