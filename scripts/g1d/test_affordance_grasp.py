"""Unit & integration tests for scripts/g1d/affordance_grasp.py."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1d.affordance_grasp import AffordanceGraspRobot, AffordancePlanner
from scripts.objects import affordance as aff
from scripts.objects.test_affordance import make_box_mesh, make_cylinder_mesh


def _quat_wxyz(rot: Rotation) -> list[float]:
    xyzw = rot.as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _make_sample_affordance() -> dict:
    """Annotate a 4x6x8 cm box end-to-end via Stage 1."""
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    return aff.annotate_mesh(verts, faces, asset_id="box_4x6x8", n_surface_samples=200, max_grasps=16, seed=11)


def test_end_to_end_annotate_and_plan_world_transform(tmp_path: Path):
    data = _make_sample_affordance()
    aff_path = tmp_path / "affordance.json"
    aff_path.write_text(__import__("json").dumps(data), encoding="utf-8")

    planner = AffordancePlanner(gripper_type="dex1_1")
    obj_pos = np.array([0.15, -0.40, 0.80])
    obj_rot = Rotation.from_euler("z", 45, degrees=True)
    planned = planner.plan_grasps(aff_path, obj_pos, _quat_wxyz(obj_rot), table_z=0.76)

    assert len(planned) >= 2
    for pg in planned:
        # Grasp must be near the object in world coordinates
        assert np.linalg.norm(pg.position - obj_pos) < 0.08
        # Approach must not point upward from under the table
        assert pg.approach_dir[2] <= planner.max_approach_z + 1e-6
        # Fingertips must stay above table_z
        p_tip = pg.position + planner.spec.finger_tips * pg.approach_dir
        assert p_tip[2] >= 0.76
        # Orthonormal right-handed rotation
        mat = pg.rotation.as_matrix()
        assert np.allclose(mat.T @ mat, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-5)


def test_dr_scaling_filters_overwide_and_rescales_width():
    data = _make_sample_affordance()
    planner = AffordancePlanner(gripper_type="dex1_1")
    obj_pos = np.array([0.0, -0.40, 0.85])
    identity_q = [1.0, 0.0, 0.0, 0.0]

    # At 1.0x, box is 4x6x8 cm -> valid grasps
    p_1x = planner.plan_grasps(data, obj_pos, identity_q, scale=1.0, table_z=0.76)
    assert len(p_1x) >= 2

    # At 0.5x, box is 2x3x4 cm -> widths shrink by half
    p_half = planner.plan_grasps(data, obj_pos, identity_q, scale=0.5, table_z=0.76)
    assert len(p_half) >= 2
    assert max(pg.width for pg in p_half) <= 0.035

    # At 2.5x, box is 10x15x20 cm -> exceeds Dex1-1 max_width (0.070 m) -> 0 grasps!
    p_huge = planner.plan_grasps(data, obj_pos, identity_q, scale=2.5, table_z=0.76)
    assert p_huge == []

    # Non-uniform scale: stretch Y by 2.0x (12 cm, ungraspable) while keeping X at 1.0x (4 cm, graspable)
    p_nonunif = planner.plan_grasps(data, obj_pos, identity_q, scale=(1.0, 2.0, 1.0), table_z=0.76)
    assert len(p_nonunif) >= 1
    assert all(pg.width <= planner.spec.max_width for pg in p_nonunif)


def test_continuous_symmetry_expands_multiple_yaws():
    verts, faces = make_cylinder_mesh(radius=0.025, height=0.10, n_seg=36, n_rings=8)
    data = aff.annotate_mesh(verts, faces, asset_id="cyl", n_surface_samples=180, max_grasps=4, seed=2)
    assert data["symmetry"]["z_axis"] == "continuous"

    # Keep only 1 raw grasp to verify symmetry expansion fans it out across yaw angles
    data["grasps"]["dex1_1"] = data["grasps"]["dex1_1"][:1]
    planner = AffordancePlanner(gripper_type="dex1_1", continuous_sym_samples=8)
    planned = planner.plan_grasps(data, [0.0, -0.4, 0.85], [1.0, 0.0, 0.0, 0.0], table_z=0.76, top_k=16)
    assert len(planned) >= 4


def test_part_tag_and_verified_filtering():
    data = {
        "schema_version": 1,
        "asset_id": "mug",
        "scale_ref": 1.0,
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_handle",
                    "position": [0.05, 0.0, 0.02],
                    "quaternion_wxyz": _quat_wxyz(Rotation.from_euler("y", 60, degrees=True)),
                    "approach_dir": [0.5, 0.0, -0.866],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.018,
                    "part_tag": "handle",
                    "score": 0.80,
                    "verified": True,
                    "sim_score": 0.92,
                },
                {
                    "id": "g_rim",
                    "position": [0.0, 0.0, 0.06],
                    "quaternion_wxyz": _quat_wxyz(Rotation.from_euler("y", 90, degrees=True)),
                    "approach_dir": [0.0, 0.0, -1.0],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.012,
                    "part_tag": "rim",
                    "score": 0.90,
                    "verified": False,
                    "sim_score": None,
                },
            ]
        },
    }
    planner = AffordancePlanner()
    # Allow only handle
    only_handle = planner.plan_grasps(data, [0, -0.4, 0.82], [1, 0, 0, 0], part_tags=["handle"])
    assert len(only_handle) >= 1 and all(pg.part_tag == "handle" for pg in only_handle)

    # Exclude rim
    no_rim = planner.plan_grasps(data, [0, -0.4, 0.82], [1, 0, 0, 0], exclude_tags=["rim"])
    assert len(no_rim) >= 1 and all(pg.part_tag != "rim" for pg in no_rim)

    # Verified only
    ver_only = planner.plan_grasps(data, [0, -0.4, 0.82], [1, 0, 0, 0], verified_only=True)
    assert len(ver_only) >= 1 and all(pg.verified for pg in ver_only)


def test_table_and_obstacle_and_ik_filters():
    # Single top-down grasp at local z=0
    rot_down = Rotation.from_euler("y", 90, degrees=True)  # +X_ee -> -Z_world
    data = {
        "schema_version": 1,
        "asset_id": "block",
        "scale_ref": 1.0,
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g0",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_down),
                    "approach_dir": [0.0, 0.0, -1.0],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.03,
                    "part_tag": "body_top",
                    "score": 0.9,
                    "verified": True,
                    "sim_score": 0.9,
                }
            ]
        },
    }
    planner = AffordancePlanner()

    # 1. Table collision: object at z=0.80, table_z=0.795 -> fingertip at 0.80 - 0.023 = 0.777 < 0.795 -> rejected!
    assert planner.plan_grasps(data, [0, -0.4, 0.80], [1, 0, 0, 0], table_z=0.795) == []
    # Raise object to z=0.84 -> fingertip at 0.817 > 0.795 -> passes!
    ok = planner.plan_grasps(data, [0, -0.4, 0.84], [1, 0, 0, 0], table_z=0.795)
    assert len(ok) >= 1

    # 2. Obstacle collision: sphere right at [0, -0.4, 0.84] blocks grasp
    blocked = planner.plan_grasps(
        data,
        [0, -0.4, 0.84],
        [1, 0, 0, 0],
        table_z=0.795,
        obstacles=[{"position": [0.0, -0.4, 0.84], "radius": 0.03}],
    )
    assert blocked == []

    # 3. IK filter callback: reject all
    rejected_ik = planner.plan_grasps(
        data, [0, -0.4, 0.84], [1, 0, 0, 0], table_z=0.795, ik_filter=lambda p, r, w: (False, 1.0)
    )
    assert rejected_ik == []


def test_topk_stochastic_sampling_diversity():
    data = _make_sample_affordance()
    planner = AffordancePlanner()
    top_ids = set()
    for seed in range(10):
        res = planner.plan_grasps(
            data,
            [0.0, -0.4, 0.85],
            [1.0, 0.0, 0.0, 0.0],
            table_z=0.76,
            top_k=8,
            sample_temperature=0.5,
            rng=np.random.default_rng(seed),
        )
        assert len(res) >= 2
        top_ids.add(res[0].grasp_id)
    assert len(top_ids) >= 2


def test_affordance_grasp_robot_candidates_and_fallback():
    """Verify AffordanceGraspRobot emits affordance candidates first and falls back cleanly without Isaac Sim."""
    data = _make_sample_affordance()

    # Build a lightweight mock instance bypassing Isaac Sim Robot.__init__
    robot = object.__new__(AffordanceGraspRobot)
    left_arm = SimpleNamespace(arm_key="left_arm", grip_key="left_gripper")
    right_arm = SimpleNamespace(arm_key="right_arm", grip_key="right_gripper")
    robot.arms = [left_arm, right_arm]
    robot.arm_for = lambda x: left_arm if x < 0 else right_arm
    robot.object = lambda label: (
        np.array([-0.10, -0.40, 0.84]),
        Rotation.identity(),
        np.array([0.04, 0.06, 0.08]),
        0.80,
        0.88,
    )
    robot.lm = SimpleNamespace(
        get_instance_pose=lambda idx, label: (np.array([-0.10, -0.40, 0.84]), np.array([1.0, 0.0, 0.0, 0.0]))
    )
    robot.env = SimpleNamespace(cfg=SimpleNamespace(layout={"objects": []}))
    robot.grip_z = lambda bottom, top: max(top - 0.03, bottom + 0.029)
    robot.affordance_map = {"target": data}
    robot.assets_root = None
    robot.planner = AffordancePlanner(gripper_type="dex1_1")
    robot.miss = lambda cand, lift: (0.002, 1.0)  # pretend all candidates are URDF-reachable

    cands = robot.candidates("target", across=None, prefer=None, period=math.pi, arm=None)
    assert cands[0][4].startswith("aff:")
    # Fallback bbox candidates must still be appended at the tail
    assert any(c[4] == "v3" for c in cands)
    chosen, miss = robot.choose(cands, lift=0.15)
    assert chosen[4].startswith("aff:")
    assert miss == (0.002, 1.0)

    # Now remove affordance -> must fall back cleanly to v3 bbox candidates
    robot.affordance_map = {}
    cands_fb = robot.candidates("target", across=None, prefer=None, period=math.pi, arm=None)
    assert not any(c[4].startswith("aff:") for c in cands_fb)
    assert cands_fb[0][4] == "v3"


def test_offset_origin_cylinder_symmetry_stays_on_object():
    vc, fc = make_cylinder_mesh(0.025, 0.10, n_seg=36, n_rings=8)
    vc_off = vc + np.array([0.05, -0.02, 0.0])
    data = aff.annotate_mesh(vc_off, fc, asset_id="offset_cyl", n_surface_samples=200, max_grasps=4, seed=0)
    assert data["symmetry"]["z_axis"] == "continuous"
    assert np.allclose(data["symmetry"]["axis_xy"], [0.05, -0.02], atol=1e-3)

    planner = AffordancePlanner(gripper_type="dex1_1", continuous_sym_samples=8)
    planned = planner.plan_grasps(data, [0.0, 0.0, 0.85], [1.0, 0.0, 0.0, 0.0], table_z=0.75)
    assert len(planned) >= 4
    for pg in planned:
        # Every rotated grasp must remain centered near (0.05, -0.02), never swung to (-0.05, +0.02)
        assert np.linalg.norm(pg.position[:2] - np.array([0.05, -0.02])) < 0.015


def test_nonuniform_xy_scale_rejects_broken_symmetry_directions():
    rot_x = Rotation.from_matrix(np.column_stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))
    data = {
        "schema_version": 1,
        "asset_id": "cyl",
        "scale_ref": 1.0,
        "symmetry": {"z_axis": "continuous", "fold": 0, "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g0",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_x),
                    "approach_dir": [0.0, 0.0, -1.0],
                    "closing_axis": [1.0, 0.0, 0.0],
                    "width": 0.05,
                    "part_tag": "body_top",
                    "score": 0.9,
                    "verified": True,
                    "sim_score": 0.9,
                }
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1", continuous_sym_samples=8)
    # Scale Y by 2.0x -> Y diameter becomes 0.10 m > JAW_MAX (0.070 m)
    planned = planner.plan_grasps(data, [0.0, 0.0, 0.85], [1.0, 0.0, 0.0, 0.0], scale=(1.0, 2.0, 1.0), table_z=0.75)
    assert len(planned) >= 1
    for pg in planned:
        assert pg.width <= planner.spec.max_width
        # No grasp may close along the stretched 10 cm Y axis
        assert abs(pg.closing_axis[1]) < 0.5


def test_vertical_closing_open_finger_and_palm_table_clearance():
    # Horizontal approach (+X) with vertical closing (+Z)
    rot_side_vert = Rotation.from_matrix(np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    data = {
        "schema_version": 1,
        "asset_id": "vert_jaw",
        "scale_ref": 1.0,
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g0",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_side_vert),
                    "approach_dir": [1.0, 0.0, 0.0],
                    "closing_axis": [0.0, 0.0, 1.0],
                    "width": 0.02,
                    "part_tag": "body_side",
                    "score": 0.9,
                    "verified": True,
                    "sim_score": 0.9,
                }
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1")
    # At z=0.795 above table=0.76 (3.5 cm), open finger (0.0385 m) and rolled palm (0.0425 m) penetrate table -> reject!
    assert planner.plan_grasps(data, [0.0, 0.0, 0.795], [1.0, 0.0, 0.0, 0.0], table_z=0.76) == []
    # Raise to z=0.82 (6.0 cm above table) -> clears both open finger and rolled palm -> accept!
    ok = planner.plan_grasps(data, [0.0, 0.0, 0.82], [1.0, 0.0, 0.0, 0.0], table_z=0.76)
    assert len(ok) >= 1


def test_affordance_grasp_robot_real_scene_manager_resolution_and_obstacles(tmp_path: Path):
    """Verify real OpenFlywheel env.scene_manager.config['env0']['Rigid'] lookup & distractor obstacle extraction."""
    import json

    data = _make_sample_affordance()
    asset_dir = tmp_path / "Object" / "Real2Sim" / "Rigid" / "plastic_cup" / "00001"
    asset_dir.mkdir(parents=True)
    (asset_dir / "affordance.json").write_text(json.dumps(data), encoding="utf-8")

    robot = object.__new__(AffordanceGraspRobot)
    left_arm = SimpleNamespace(arm_key="left_arm", grip_key="left_gripper")
    right_arm = SimpleNamespace(arm_key="right_arm", grip_key="right_gripper")
    robot.arms = [left_arm, right_arm]
    robot.arm_for = lambda x: left_arm if x < 0 else right_arm
    robot.object = lambda label: (
        np.array([-0.10, -0.40, 0.84]),
        Rotation.identity(),
        np.array([0.04, 0.06, 0.08]),
        0.80,
        0.88,
    )
    poses = {
        "target": (np.array([-0.10, -0.40, 0.84]), np.array([1.0, 0.0, 0.0, 0.0])),
        "distractor_0": (np.array([-0.10, -0.40, 0.84]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    robot.lm = SimpleNamespace(get_instance_pose=lambda idx, label: poses[label])
    robot.env = SimpleNamespace(
        scene_manager=SimpleNamespace(
            config={
                "env0": {
                    "Rigid": {
                        "plastic_cup": [
                            {
                                "category": "plastic_cup",
                                "category_idx": 1,
                                "label": "target",
                                "scale": [1.0, 1.0, 1.0],
                                "physics": {"size": [0.04, 0.06, 0.08]},
                            }
                        ],
                        "bottle": [
                            {
                                "category": "bottle",
                                "category_idx": 2,
                                "label": "distractor_0",
                                "scale": [1.0, 1.0, 1.0],
                                "physics": {"size": [0.06, 0.06, 0.15]},
                            }
                        ],
                    }
                }
            }
        )
    )
    robot.grip_z = lambda bottom, top: max(top - 0.03, bottom + 0.029)
    robot.affordance_map = {}
    robot.assets_root = tmp_path
    robot.planner = AffordancePlanner(gripper_type="dex1_1")
    robot.miss = lambda cand, lift: (0.002, 1.0)

    # 1. Resolves affordance.json via category/category_idx + assets_root
    resolved = robot._resolve_affordance("target")
    assert resolved is not None and resolved["asset_id"] == "box_4x6x8"

    # 2. Extracts distractor_0 from scene_manager.config["env0"]["Rigid"]
    obs = robot._scene_obstacles(exclude_label="target")
    assert len(obs) == 1
    assert np.allclose(obs[0]["position"], [-0.10, -0.40, 0.84])

    # 3. With distractor sitting right on top of target, all affordance grasps are blocked -> falls back to v3
    cands_blocked = robot.candidates("target", across=None, prefer=None, period=math.pi, arm=None)
    assert not any(c[4].startswith("aff:") for c in cands_blocked)
    assert cands_blocked[0][4] == "v3"

    # 4. Move distractor away -> affordance grasps succeed and appear first
    poses["distractor_0"] = (np.array([0.30, -0.40, 0.84]), np.array([1.0, 0.0, 0.0, 0.0]))
    cands_clear = robot.candidates("target", across=None, prefer=None, period=math.pi, arm=None)
    assert cands_clear[0][4].startswith("aff:")


def test_side_grasp_pre_grasp_waypoints_and_affordance_dir_override(tmp_path: Path):
    from scripts.g1d.affordance_grasp import compute_pre_grasp_waypoints

    # Top-down grasp (-Z approach) -> is_side=False, hover == pre_grasp == lift_pos
    rot_top = Rotation.from_matrix(np.column_stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))
    hover, pre, lift_pos, is_side = compute_pre_grasp_waypoints(
        np.array([0.0, -0.4, 0.85]), rot_top, lift=0.13, table_z=0.76
    )
    assert not is_side
    assert np.allclose(hover, [0.0, -0.4, 0.98])
    assert np.allclose(pre, lift_pos)

    # Horizontal side grasp (+Y approach) -> is_side=True, pre_grasp retracted along -Y
    rot_side = Rotation.from_matrix(np.column_stack([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    hover_s, pre_s, lift_s, is_side_s = compute_pre_grasp_waypoints(
        np.array([0.0, -0.4, 0.82]), rot_side, lift=0.13, table_z=0.76, approach_back=0.07
    )
    assert is_side_s
    assert np.allclose(pre_s, [0.0, -0.47, 0.82], atol=1e-6)
    assert hover_s[0] == pre_s[0] and hover_s[1] == pre_s[1] and hover_s[2] >= lift_s[2]

    # Verify affordance_dir overlay resolves without touching read-only assets_root
    import json

    data = _make_sample_affordance()
    override_dir = tmp_path / "aff_cache" / "plastic_cup" / "00001"
    override_dir.mkdir(parents=True)
    (override_dir / "affordance.json").write_text(json.dumps(data), encoding="utf-8")

    robot = object.__new__(AffordanceGraspRobot)
    robot.affordance_map = {}
    robot.affordance_dir = tmp_path / "aff_cache"
    robot.assets_root = tmp_path / "empty_assets"
    robot.env = SimpleNamespace(
        scene_manager=SimpleNamespace(
            config={
                "env0": {"Rigid": {"plastic_cup": [{"category": "plastic_cup", "category_idx": 1, "label": "target"}]}}
            }
        )
    )
    resolved = robot._resolve_affordance("target")
    assert resolved is not None and resolved["asset_id"] == "box_4x6x8"


def test_affordance_grasp_robot_handles_mapping_and_sequence_configs(tmp_path: Path):
    """Ensure non-dict Mapping / non-list Sequence configs (like omegaconf DictConfig/ListConfig) resolve cleanly."""
    from collections.abc import Mapping, Sequence as AbcSequence
    import json

    class FakeDictConfig(Mapping):
        def __init__(self, d):
            self._d = d

        def __getitem__(self, k):
            return self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    class FakeListConfig(AbcSequence):
        def __init__(self, items):
            self._items = list(items)

        def __getitem__(self, i):
            return self._items[i]

        def __len__(self):
            return len(self._items)

    data = _make_sample_affordance()
    override_dir = tmp_path / "aff_cache" / "hammer" / "00008"
    override_dir.mkdir(parents=True)
    (override_dir / "affordance.json").write_text(json.dumps(data), encoding="utf-8")

    entry = FakeDictConfig(
        {
            "category": "hammer",
            "category_idx": 8,
            "label": "target",
            "scale": FakeListConfig([1.05, 1.05, 1.05]),
            "physics": FakeDictConfig({"size": FakeListConfig([0.04, 0.06, 0.08])}),
        }
    )
    cfg = FakeDictConfig({"env0": FakeDictConfig({"Rigid": FakeDictConfig({"hammer": FakeListConfig([entry])})})})

    robot = object.__new__(AffordanceGraspRobot)
    robot.affordance_map = {}
    robot.affordance_dir = tmp_path / "aff_cache"
    robot.assets_root = None
    robot.env = SimpleNamespace(scene_manager=SimpleNamespace(config=cfg))

    resolved = robot._resolve_affordance("target")
    assert resolved is not None and resolved["asset_id"] == "box_4x6x8"
    assert robot._object_scale("target") == [1.05, 1.05, 1.05]
    assert len(robot._iter_rigid_entries()) == 1


def test_top_down_beats_45deg_and_45deg_triggers_side_pre_grasp():
    """Regression for lid=15: dorsal_z must not bias 45-deg over top-down, and 45-deg uses side pre-grasp."""
    from scripts.g1d.affordance_grasp import compute_pre_grasp_waypoints

    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    # 45-deg angled grasp with dorsal_z = +0.7071
    s2 = math.sqrt(0.5)
    rot_45 = Rotation.from_matrix(np.column_stack([[0.0, -s2, -s2], [1.0, 0.0, 0.0], [0.0, -s2, s2]]))

    def _wxyz(r: Rotation) -> list[float]:
        q = r.as_quat()
        return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    aff = {
        "schema_version": 1,
        "asset_id": "Rigid/fmb_assembly_part/00009",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.0185, -0.1, -0.0185], "max": [0.0185, 0.1, 0.0185], "extents": [0.037, 0.2, 0.037]},
        "symmetry": {"z_axis": "none", "fold": 1, "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g000_top",
                    "position": [0.0, 0.0, 0.008],
                    "quaternion_wxyz": _wxyz(rot_top),
                    "width": 0.037,
                    "part_tag": "body_top",
                    "score": 0.94,
                    "verified": True,
                    "sim_score": 0.931,
                },
                {
                    "id": "g008_45deg",
                    "position": [0.0, -0.01, 0.012],
                    "quaternion_wxyz": _wxyz(rot_45),
                    "width": 0.037,
                    "part_tag": "body_top",
                    "score": 0.915,
                    "verified": True,
                    "sim_score": 0.905,
                },
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1")
    planned = planner.plan_grasps(aff, [0.0, -0.4, 0.74], [1.0, 0.0, 0.0, 0.0], table_z=0.71)
    assert len(planned) >= 2
    assert planned[0].grasp_id.startswith("g000_top"), f"Expected pure top-down first, got {planned[0].grasp_id}"

    # Shallow-angle side grasp (approach_z = -0.5 > -0.70) triggers approach-axis pre-grasp
    rot_shallow = Rotation.from_matrix(
        np.column_stack([[0.0, -math.sqrt(0.75), -0.5], [1.0, 0.0, 0.0], [0.0, -0.5, math.sqrt(0.75)]])
    )
    _, _, _, is_side = compute_pre_grasp_waypoints(np.array([0.0, -0.4, 0.75]), rot_shallow, lift=0.13, table_z=0.71)
    assert is_side


def test_tall_object_palm_top_clearance_prefers_upper_grasp():
    """Regression for lid=0: deep top-down grasp with <8 mm palm-to-top clearance must be filtered out."""
    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    q = rot_top.as_quat()
    q_wxyz = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    aff = {
        "schema_version": 1,
        "asset_id": "Rigid/sand3d_packaging/00158",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.03, -0.03, -0.06], "max": [0.03, 0.03, 0.06], "extents": [0.06, 0.06, 0.12]},
        "symmetry": {"z_axis": "none", "fold": 1, "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g000_deep",
                    "position": [0.0, 0.0, 0.0189],  # 4.1 cm below top -> only 4 mm palm clearance!
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.059,
                    "part_tag": "body_top",
                    "score": 0.975,
                    "verified": True,
                    "sim_score": 0.998,
                },
                {
                    "id": "g002_upper",
                    "position": [0.0, 0.0, 0.0280],  # 3.2 cm below top -> 13 mm palm clearance
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.059,
                    "part_tag": "body_top",
                    "score": 0.865,
                    "verified": True,
                    "sim_score": 0.848,
                },
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1")
    planned = planner.plan_grasps(aff, [0.23, -0.43, 0.77], [1.0, 0.0, 0.0, 0.0], table_z=0.71, prefer_yaw=0.0)
    assert len(planned) >= 2
    assert planned[0].grasp_id.startswith("g002_upper"), f"Expected upper grasp first, got {planned[0].grasp_id}"


def test_vertical_approach_corridor_obstacle_rejection():
    """An obstacle blocking the open-finger vertical approach corridor must reject the candidate."""
    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    q = rot_top.as_quat()
    q_wxyz = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    aff = {
        "schema_version": 1,
        "asset_id": "box",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.02, -0.02, -0.02], "max": [0.02, 0.02, 0.02], "extents": [0.04, 0.04, 0.04]},
        "symmetry": {"z_axis": "none", "fold": 1, "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g0",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.04,
                    "part_tag": "body_top",
                    "score": 0.9,
                    "verified": True,
                    "sim_score": 0.9,
                }
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1")
    # Obstacle offset along closing axis (+X) and +0.05 m above grip: clear of center/palm, hits open finger corridor
    obs = [{"position": [0.055, -0.40, 0.85], "radius": 0.015}]
    planned_blocked = planner.plan_grasps(aff, [0.0, -0.40, 0.80], [1.0, 0.0, 0.0, 0.0], table_z=0.75, obstacles=obs)
    assert planned_blocked == []
    planned_clear = planner.plan_grasps(aff, [0.0, -0.40, 0.80], [1.0, 0.0, 0.0, 0.0], table_z=0.75, obstacles=[])
    assert len(planned_clear) >= 1


def test_skip_sim_failed_grasps_and_scoring():
    """Stage-2-failed grasps must be skipped by default and never outscore verified grasps."""
    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    q_wxyz = _quat_wxyz(rot_top)

    # 1. All-failed asset (Stage 2 ran and rejected all grasps) -> returns [] so robot falls back to v4.1
    all_failed = {
        "schema_version": 1,
        "asset_id": "Rigid/paper_wad/00001",
        "scale_ref": 1.0,
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_fail",
                    "position": [0.0, 0.0, 0.02],
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.03,
                    "part_tag": "body_top",
                    "score": 0.97,
                    "verified": False,
                    "sim_score": 0.0,
                    "sim_metrics": {"lift_height_m": 0.0, "reason": "insufficient_lift"},
                }
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1")
    assert planner.plan_grasps(all_failed, [0.0, -0.40, 0.80], [1.0, 0.0, 0.0, 0.0], table_z=0.75) == []

    # 2. Mixed asset: high-geom Stage-2-failed grasp must be skipped and never outscore verified grasp
    mixed = {
        "schema_version": 1,
        "asset_id": "Rigid/sand3d_packaging/00001",
        "scale_ref": 1.0,
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_fail_high_geom",
                    "position": [0.0, 0.0, 0.02],
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.03,
                    "part_tag": "body_top",
                    "score": 0.97,
                    "verified": False,
                    "sim_score": 0.0,
                    "sim_metrics": {"lift_height_m": 0.0, "reason": "insufficient_lift"},
                },
                {
                    "id": "g_verified",
                    "position": [0.0, 0.0, 0.03],
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.03,
                    "part_tag": "body_top",
                    "score": 0.82,
                    "verified": True,
                    "sim_score": 0.80,
                    "sim_metrics": {"lift_height_m": 0.12, "reason": "ok"},
                },
                {
                    "id": "g_stage1_untested",
                    "position": [0.02, 0.0, 0.02],
                    "quaternion_wxyz": q_wxyz,
                    "width": 0.03,
                    "part_tag": "body_top",
                    "score": 0.70,
                    "verified": False,
                    "sim_score": None,
                },
            ]
        },
    }
    planned = planner.plan_grasps(mixed, [0.0, -0.40, 0.80], [1.0, 0.0, 0.0, 0.0], table_z=0.75)
    ids = [pg.grasp_id for pg in planned]
    assert not any(gid.startswith("g_fail_high_geom") for gid in ids)
    assert planned[0].grasp_id.startswith("g_verified")
    # Untested Stage-1 grasp is still preserved when verified_only=False
    assert any(gid.startswith("g_stage1_untested") for gid in ids)

    # 3. Even with skip_sim_failed=False, sim_score=0.0 is used so g_verified still wins over g_fail_high_geom
    planned_incl = planner.plan_grasps(
        mixed, [0.0, -0.40, 0.80], [1.0, 0.0, 0.0, 0.0], table_z=0.75, skip_sim_failed=False
    )
    assert planned_incl[0].grasp_id.startswith("g_verified")
    failed_cands = [pg for pg in planned_incl if pg.grasp_id.startswith("g_fail_high_geom")]
    assert failed_cands and failed_cands[0].base_score == 0.0


def test_45deg_triggers_side_pre_grasp_and_60deg_stays_vertical():
    from scripts.g1d.affordance_grasp import compute_pre_grasp_waypoints

    s2 = math.sqrt(0.5)
    rot_45 = Rotation.from_matrix(np.column_stack([[0.0, s2, -s2], [-1.0, 0.0, 0.0], [0.0, s2, s2]]))
    _, pre_45, _, is_side_45 = compute_pre_grasp_waypoints(np.array([0.0, -0.4, 0.82]), rot_45, lift=0.13, table_z=0.72)
    assert is_side_45
    assert pre_45[1] < -0.42 and pre_45[2] > 0.85

    # 60-deg TOP_ROT (approach_z = -0.866) stays vertical top-down
    rot_60 = Rotation.from_matrix(
        np.column_stack([[0.0, 0.5, -math.sqrt(0.75)], [-1.0, 0.0, 0.0], [0.0, math.sqrt(0.75), 0.5]])
    )
    _, _, _, is_side_60 = compute_pre_grasp_waypoints(np.array([0.0, -0.4, 0.82]), rot_60, lift=0.13, table_z=0.72)
    assert not is_side_60


def test_colliding_palm_top_clear_rejected_and_top_down_beats_side_with_prefer_yaw():
    """Regressions for drink_bottle/00002 (palm_top_clear < 0) and tumbler/00001 (top-down beats side)."""
    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    rot_side = Rotation.from_matrix(np.column_stack([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    planner = AffordancePlanner(gripper_type="dex1_1")

    # 1. Tall bottle (18 cm): grasp 6.4 cm below top has palm_top_clear = 0.044 - 0.064 = -0.020 m -> hard-rejected
    bottle = {
        "schema_version": 1,
        "asset_id": "Rigid/drink_bottle/00002",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.027, -0.023, -0.089], "max": [0.027, 0.023, 0.089], "extents": [0.054, 0.046, 0.178]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_deep_collide",
                    "position": [0.0, 0.0, 0.025],
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.055,
                    "part_tag": "body_top",
                    "score": 0.90,
                    "verified": True,
                    "sim_score": 0.90,
                }
            ]
        },
    }
    assert planner.plan_grasps(bottle, [0.05, -0.46, 0.81], [1, 0, 0, 0], table_z=0.72) == []

    # 2. Tumbler: safe top-down grasp beats horizontal side grasp even when prefer_yaw=0 aligns with +Y side approach
    tumbler = {
        "schema_version": 1,
        "asset_id": "Rigid/tumbler/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.03, -0.03, -0.087], "max": [0.03, 0.03, 0.087], "extents": [0.06, 0.06, 0.174]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_side",
                    "position": [0.0, 0.0, 0.004],
                    "quaternion_wxyz": _quat_wxyz(rot_side),
                    "width": 0.055,
                    "part_tag": "body_side",
                    "score": 0.96,
                    "verified": True,
                    "sim_score": 0.96,
                },
                {
                    "id": "g_top",
                    "position": [0.0, 0.0, 0.056],
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.058,
                    "part_tag": "body_top",
                    "score": 0.94,
                    "verified": True,
                    "sim_score": 0.94,
                },
            ]
        },
    }
    planned = planner.plan_grasps(tumbler, [0.23, -0.35, 0.81], [1, 0, 0, 0], table_z=0.72, prefer_yaw=0.0)
    assert len(planned) >= 2
    assert planned[0].grasp_id.startswith("g_top"), f"Expected safe top-down first, got {planned[0].grasp_id}"


def test_rollout_step_ik_detects_joint_limit_stall():
    """Verify _rollout_step_ik passes reachable paths and flags joint-limit stalls."""
    from scripts.g1d import scripted_demo as demo
    from scripts.g1d.affordance_grasp import _rollout_step_ik

    class _PlanarArmReach:
        """Simple 3-DOF planar RRR arm (x, y, yaw) for deterministic step-IK unit testing."""

        def __init__(self, lim):
            self.lim = np.asarray(lim, dtype=np.float64)
            self.base = np.eye(4)
            self.reach = np.array([0.0, 0.0, 0.0])

        def fk(self, q, axes=None):
            th1, th2, th3 = float(q[0]), float(q[1]), float(q[2])
            a1, a2, a3 = th1, th1 + th2, th1 + th2 + th3
            p1 = np.array([0.25 * math.cos(a1), 0.25 * math.sin(a1), 0.0])
            p2 = p1 + np.array([0.25 * math.cos(a2), 0.25 * math.sin(a2), 0.0])
            p3 = p2 + np.array([0.10 * math.cos(a3), 0.10 * math.sin(a3), 0.0])
            if axes is not None:
                axes.append((np.zeros(3), np.array([0.0, 0.0, 1.0])))
                axes.append((p1.copy(), np.array([0.0, 0.0, 1.0])))
                axes.append((p2.copy(), np.array([0.0, 0.0, 1.0])))
            t = np.eye(4)
            t[:3, :3] = Rotation.from_euler("z", a3).as_matrix()
            t[:3, 3] = p3
            return t

    reach = _PlanarArmReach(lim=[[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    q0 = np.array([0.2, 0.6, -0.3])
    t0 = reach.fk(q0)
    r0 = Rotation.from_matrix(t0[:3, :3])
    p0 = t0[:3, 3]

    # Small reachable move within joint limits -> near-zero error
    pos_ok, rot_ok = _rollout_step_ik(reach, q0, [(p0 + np.array([-0.04, 0.04, 0.0]), r0, 20, True)], demo.waypoints)
    assert pos_ok < 0.008 and rot_ok < 12.0

    # Target far outside arm reach (x=1.2 m, arm max=0.6 m) -> large error detected
    pos_bad, _ = _rollout_step_ik(reach, q0, [(np.array([1.2, 0.0, 0.0]), r0, 20, True)], demo.waypoints)
    assert pos_bad > 0.40

    # Rate-limited mid-segment tracking spike (Limiter dq_max): fast 0.25 m move over 8 steps
    # succeeds without dq_max, but exceeds max_mid_err when dq_max=0.0267 rad/step is enforced.
    target_fast = p0 + np.array([-0.22, 0.16, 0.0])
    pos_unlim, _ = _rollout_step_ik(reach, q0, [(target_fast, r0, 8, True)], demo.waypoints, settle_steps=60)
    pos_lim, _ = _rollout_step_ik(
        reach, q0, [(target_fast, r0, 8, True)], demo.waypoints, settle_steps=60, dq_max=0.8 / 30.0
    )
    assert pos_unlim < 0.05
    assert pos_lim > 0.10


def test_vertical_palm_top_clear_margin_rejects_grazing_bottle_but_keeps_angled_and_safe():
    """Vertical top-down grasps need >= 6 mm palm clearance (empty_bottle/00001 +3.3 mm rejected);
    45-deg angled grasps keep >= 0 mm threshold."""
    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    s2 = math.sqrt(0.5)
    rot_45 = Rotation.from_matrix(np.column_stack([[0.0, s2, -s2], [-1.0, 0.0, 0.0], [0.0, s2, s2]]))
    planner = AffordancePlanner(gripper_type="dex1_1", max_approach_z=-0.35)

    # Tall 19.8 cm empty bottle: g_graze has +3.3 mm clear (rejected), g_safe has +7.0 mm clear (kept),
    # g_45_tight has +1.0 mm clear at 45 deg (kept).
    bottle = {
        "schema_version": 1,
        "asset_id": "Rigid/empty_bottle/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.022, -0.022, -0.099], "max": [0.022, 0.022, 0.099], "extents": [0.044, 0.044, 0.198]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_graze",
                    "position": [0.0, 0.0, 0.0583],  # clear = 0.044 - (0.099 - 0.0583) = +3.3 mm
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.044,
                    "part_tag": "body_top",
                    "score": 0.98,
                    "verified": True,
                    "sim_score": 0.98,
                },
                {
                    "id": "g_safe",
                    "position": [0.0, 0.0, 0.062],  # clear = +7.0 mm
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.044,
                    "part_tag": "body_top",
                    "score": 0.90,
                    "verified": True,
                    "sim_score": 0.90,
                },
                {
                    "id": "g_45_tight",
                    "position": [0.0, 0.0, 0.056],  # clear = +1.0 mm, 45 deg
                    "quaternion_wxyz": _quat_wxyz(rot_45),
                    "width": 0.044,
                    "part_tag": "body_top",
                    "score": 0.85,
                    "verified": True,
                    "sim_score": 0.85,
                },
            ]
        },
    }
    planned = planner.plan_grasps(bottle, [0.19, -0.35, 0.82], [1, 0, 0, 0], table_z=0.72)
    ids = [pg.grasp_id for pg in planned]
    assert not any(gid.startswith("g_graze") for gid in ids)
    assert any(gid.startswith("g_safe") for gid in ids)
    assert any(gid.startswith("g_45_tight") for gid in ids)


def test_tabletop_max_approach_z_excludes_horizontal_side_but_keeps_45deg():
    """Tabletop planner (max_approach_z=-0.35) excludes horizontal body_side while keeping 45-deg angled grasps."""
    s2 = math.sqrt(0.5)
    rot_45 = Rotation.from_matrix(np.column_stack([[0.0, s2, -s2], [-1.0, 0.0, 0.0], [0.0, s2, s2]]))
    rot_side = Rotation.from_matrix(np.column_stack([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    data = {
        "schema_version": 1,
        "asset_id": "Rigid/tumbler/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.03, -0.03, -0.087], "max": [0.03, 0.03, 0.087], "extents": [0.06, 0.06, 0.174]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_horiz",
                    "position": [0.0, 0.0, 0.004],
                    "quaternion_wxyz": _quat_wxyz(rot_side),
                    "width": 0.055,
                    "part_tag": "body_side",
                    "score": 0.99,
                    "verified": True,
                    "sim_score": 0.99,
                },
                {
                    "id": "g_45deg",
                    "position": [0.0, 0.0, 0.056],
                    "quaternion_wxyz": _quat_wxyz(rot_45),
                    "width": 0.055,
                    "part_tag": "body_top",
                    "score": 0.91,
                    "verified": True,
                    "sim_score": 0.91,
                },
            ]
        },
    }
    planner_table = AffordancePlanner(gripper_type="dex1_1", max_approach_z=-0.35)
    planned = planner_table.plan_grasps(data, [0.23, -0.35, 0.81], [1, 0, 0, 0], table_z=0.72)
    assert len(planned) >= 1
    assert all(pg.grasp_id.startswith("g_45deg") for pg in planned)


def test_prefer_tags_ranking_and_mapping_weights():
    """prefer_tags boosts ranked or explicitly weighted semantic parts without dropping fallback tags."""
    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    rot_side = Rotation.from_matrix(np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]))
    data = {
        "schema_version": 1,
        "asset_id": "Rigid/mug/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.05], "max": [0.06, 0.04, 0.05], "extents": [0.10, 0.08, 0.10]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_body",
                    "position": [0.0, 0.0, 0.02],
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.055,
                    "part_tag": "body",
                    "tag_source": "vlm",
                    "score": 0.92,
                    "verified": True,
                    "sim_score": 0.92,
                },
                {
                    "id": "g_handle",
                    "position": [0.05, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_side),
                    "width": 0.018,
                    "part_tag": "handle",
                    "tag_source": "vlm",
                    "score": 0.85,
                    "verified": True,
                    "sim_score": 0.85,
                },
            ]
        },
    }
    planner = AffordancePlanner(gripper_type="dex1_1", max_approach_z=0.15)
    # Without prefer_tags: higher base_score + top-down bonus makes g_body win
    base_res = planner.plan_grasps(data, [0.0, -0.35, 0.80], [1, 0, 0, 0], table_z=0.72)
    assert base_res[0].part_tag == "body"

    # With prefer_tags sequence ("handle", "body"): handle wins while body remains as fallback
    pref_res = planner.plan_grasps(data, [0.0, -0.35, 0.80], [1, 0, 0, 0], table_z=0.72, prefer_tags=("handle", "body"))
    assert pref_res[0].part_tag == "handle"
    assert any(pg.part_tag == "body" for pg in pref_res)

    # With explicit weight mapping: custom boost works
    map_res = planner.plan_grasps(
        data, [0.0, -0.35, 0.80], [1, 0, 0, 0], table_z=0.72, prefer_tags={"handle": 0.40, "body": 0.0}
    )
    assert map_res[0].part_tag == "handle"


def test_select_pour_grasp_prefers_handle_and_excludes_rim_spout_lid():
    """Pouring helper picks handle first, excludes rim/spout/lid, and falls back to body on handleless cups."""
    from scripts.g1d.affordance_grasp import TASK_PART_POLICIES, make_task_planner, select_pour_grasp

    assert set(TASK_PART_POLICIES.keys()) >= {"pick", "place", "put", "pour"}
    assert make_task_planner("pour").max_approach_z == 0.15

    rot_top = Rotation.from_matrix(np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    rot_side = Rotation.from_matrix(np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]))

    pitcher = {
        "schema_version": 1,
        "asset_id": "Rigid/pitcher/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.06], "max": [0.06, 0.04, 0.06], "extents": [0.10, 0.08, 0.12]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_rim",
                    "position": [0.0, 0.0, 0.05],
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.060,
                    "part_tag": "rim",
                    "tag_source": "vlm",
                    "score": 0.99,
                    "verified": True,
                    "sim_score": 0.99,
                },
                {
                    "id": "g_spout",
                    "position": [-0.04, 0.0, 0.04],
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.025,
                    "part_tag": "spout",
                    "tag_source": "vlm",
                    "score": 0.98,
                    "verified": True,
                    "sim_score": 0.98,
                },
                {
                    "id": "g_lid",
                    "position": [0.0, 0.0, 0.055],
                    "quaternion_wxyz": _quat_wxyz(rot_top),
                    "width": 0.050,
                    "part_tag": "lid",
                    "tag_source": "vlm",
                    "score": 0.97,
                    "verified": True,
                    "sim_score": 0.97,
                },
                {
                    "id": "g_body",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_side),
                    "width": 0.058,
                    "part_tag": "body",
                    "tag_source": "vlm",
                    "score": 0.90,
                    "verified": True,
                    "sim_score": 0.90,
                },
                {
                    "id": "g_handle",
                    "position": [0.052, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_side),
                    "width": 0.018,
                    "part_tag": "handle",
                    "tag_source": "vlm",
                    "score": 0.86,
                    "verified": True,
                    "sim_score": 0.86,
                },
            ]
        },
    }
    res = select_pour_grasp(pitcher, [0.15, -0.35, 0.80], [1, 0, 0, 0], table_z=0.72)
    assert len(res) >= 2
    assert res[0].part_tag == "handle"
    assert all(pg.part_tag not in {"rim", "spout", "lid"} for pg in res)

    # Handleless cup: remove handle grasp, should fall back cleanly to body
    cup = {**pitcher, "grasps": {"dex1_1": [g for g in pitcher["grasps"]["dex1_1"] if g["part_tag"] != "handle"]}}
    res_cup = select_pour_grasp(cup, [0.15, -0.35, 0.80], [1, 0, 0, 0], table_z=0.72)
    assert len(res_cup) >= 1
    assert res_cup[0].part_tag == "body"


def test_affordance_place_robot_and_plugin_part_tag_filtering():
    """AffordancePlaceRobot applies place part_tag policy and plugin exports datagen contract."""
    from scripts.datagen.plugins import g1d_place_anything_affordance as plugin
    from scripts.g1d.affordance_grasp import make_affordance_place_robot_class

    PlaceRobotCls = make_affordance_place_robot_class("place")
    assert "base" in PlaceRobotCls.affordance_exclude_tags
    assert "spout" in PlaceRobotCls.affordance_exclude_tags
    assert "body_top" in PlaceRobotCls.affordance_prefer_tags

    PutRobotCls = make_affordance_place_robot_class("put")
    assert "neck" in PutRobotCls.affordance_prefer_tags

    PourRobotCls = make_affordance_place_robot_class("pour")
    assert PourRobotCls.affordance_max_approach_z == 0.15

    assert plugin.TASK == "g1d_place_anything"
    assert hasattr(plugin, "Episode")
    assert callable(plugin.audit)


def test_affordance_pour_robot_allows_horizontal_handle_and_place_robot_stall_break(monkeypatch):
    """PourRobot propagates max_approach_z=0.15; AffordancePlaceRobot runs StallBreakMixin.move + ik_within."""
    from types import SimpleNamespace

    from scripts.g1d import place_v42
    from scripts.g1d.affordance_grasp import AffordancePlanner, make_affordance_place_robot_class

    PourRobotCls = make_affordance_place_robot_class("pour")
    rot_side = Rotation.from_matrix(np.column_stack([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    pitcher = {
        "schema_version": 1,
        "asset_id": "Rigid/pitcher/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.06], "max": [0.06, 0.04, 0.06], "extents": [0.10, 0.08, 0.12]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_handle",
                    "position": [0.05, 0.0, 0.0],
                    "quaternion_wxyz": _quat_wxyz(rot_side),
                    "width": 0.018,
                    "part_tag": "handle",
                    "tag_source": "vlm",
                    "score": 0.90,
                    "verified": True,
                    "sim_score": 0.90,
                }
            ]
        },
    }
    # Verify default planner created from PourRobotCls.affordance_max_approach_z accepts horizontal handle grasp
    planner = AffordancePlanner(gripper_type="dex1_1", max_approach_z=PourRobotCls.affordance_max_approach_z)
    planned = planner.plan_grasps(
        pitcher,
        [0.15, -0.35, 0.80],
        [1, 0, 0, 0],
        table_z=0.72,
        exclude_tags=PourRobotCls.affordance_exclude_tags,
        prefer_tags=PourRobotCls.affordance_prefer_tags,
    )
    assert len(planned) >= 1
    assert all(pg.part_tag == "handle" for pg in planned)

    # Exercise AffordancePlaceRobot MRO: StallBreakMixin.move() + _command() with placing=True
    PlaceRobotCls = make_affordance_place_robot_class("place")
    ik_calls = []
    monkeypatch.setattr(place_v42, "ik_within", lambda arm, pt, rot: (ik_calls.append(pt.copy()), np.zeros(7))[1])

    robot = PlaceRobotCls.__new__(PlaceRobotCls)
    robot.placing = True
    robot.settle = (0.005, 40)
    robot.hold = {"right_arm": np.zeros(7)}
    robot.grip = {"right_grip": [0.0]}
    robot.step = 0
    robot._advance = lambda key: None

    # Arm stays stuck at origin (joint-limit stall) -> StallBreakMixin breaks well before 40 settle frames
    arm = SimpleNamespace(
        arm_key="right_arm",
        grip_key="right_grip",
        grasp_point=lambda: np.array([0.0, 0.0, 0.0]),
        pose=lambda: (np.array([0.0, 0.0, 0.0]), Rotation.identity()),
    )
    target_pt = np.array([0.10, 0.0, 0.0])
    cmds = list(robot.move(arm, [(target_pt, Rotation.identity(), 0.5, 4)]))
    # 4 waypoint steps + (1 initial prev_pos + 6 stalled frames) = 11 total, far below 4 + 40 = 44
    assert len(cmds) < 20
    assert len(ik_calls) == len(cmds)
    assert all("right_arm" in c and "right_grip" in c for c in cmds)
