"""CPU unit tests for Sharpa Wave online affordance planner (`scripts/g1d/affordance_dexhand.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.g1d import affordance_dexhand as dex_plan
from scripts.objects import affordance as aff, affordance_dexhand as dex_aff
from scripts.objects.test_affordance_dexhand import _make_box_mesh, _make_cylinder_mesh


def _make_cylinder_affordance(radius: float = 0.036, height: float = 0.11) -> dict:
    verts, faces = _make_cylinder_mesh(radius=radius, height=height)
    payload = dex_aff.annotate_mesh_sharpa_wave(verts, faces, asset_id="Rigid/test_cup/00000", max_grasps=16, seed=0)
    # Mark first half verified to test verified/unverified ranking
    for i, g in enumerate(payload["grasps"]["sharpa_wave"]):
        if i % 2 == 0:
            g["verified"] = True
            g["sim_score"] = 0.90 - 0.01 * i
    return payload


def _make_small_box_affordance() -> dict:
    verts, faces = _make_box_mesh(0.032, 0.032, 0.035)
    payload = dex_aff.annotate_mesh_sharpa_wave(verts, faces, asset_id="Rigid/test_cube/00000", max_grasps=12, seed=0)
    for g in payload["grasps"]["sharpa_wave"]:
        g["verified"] = True
        g["sim_score"] = 0.88
    return payload


def test_left_and_right_hand_mirroring_and_fk_consistency():
    payload = _make_cylinder_affordance()
    planner = dex_plan.DexHandAffordancePlanner(continuous_sym_samples=4)
    obj_pos = np.array([0.15, -0.35, 0.80])
    obj_quat = [1.0, 0.0, 0.0, 0.0]

    left_cands = planner.plan_grasps(payload, obj_pos, obj_quat, hand_side="left", table_z=0.74, top_k=8)
    right_cands = planner.plan_grasps(payload, obj_pos, obj_quat, hand_side="right", table_z=0.74, top_k=8)
    assert len(left_cands) > 0
    assert len(right_cands) > 0

    kin = dex_aff.get_sharpa_kinematics()
    for c in left_cands:
        assert c.hand_side == "left"
        assert c.ee_frame == "left_hand_flange"
        assert np.isclose(np.linalg.det(c.wrist_rotation.as_matrix()), 1.0, atol=1e-5)
        assert np.all(c.preshape_qpos >= kin.lower["left"] - 1e-6)
        assert np.all(c.preshape_qpos <= kin.upper["left"] + 1e-6)
        assert np.all(c.target_qpos >= kin.lower["left"] - 1e-6)
        assert np.all(c.target_qpos <= kin.upper["left"] + 1e-6)

    for c in right_cands:
        assert c.hand_side == "right"
        assert c.ee_frame == "right_hand_flange"
        assert np.isclose(np.linalg.det(c.wrist_rotation.as_matrix()), 1.0, atol=1e-5)
        assert np.all(c.preshape_qpos >= kin.lower["right"] - 1e-6)
        assert np.all(c.preshape_qpos <= kin.upper["right"] + 1e-6)
        assert np.all(c.target_qpos >= kin.lower["right"] - 1e-6)
        assert np.all(c.target_qpos <= kin.upper["right"] + 1e-6)
        for aa_idx in dex_plan.MCP_AA_INDICES:
            assert abs(float(c.preshape_qpos[aa_idx])) <= dex_plan.MCP_AA_SAFE_LIMIT + 1e-6
            assert abs(float(c.target_qpos[aa_idx])) <= dex_plan.MCP_AA_SAFE_LIMIT + 1e-6

    # Invalid hand_side raises ValueError
    with pytest.raises(ValueError):
        planner.plan_grasps(payload, obj_pos, obj_quat, hand_side="center")


def test_dr_scale_and_closure_adaptation_preserves_hand_geometry():
    payload = _make_cylinder_affordance(radius=0.036, height=0.11)
    kin = dex_aff.get_sharpa_kinematics()
    sw = payload["grasps"]["sharpa_wave"]
    power_g = next(g for g in sw if g["grasp_type"] == "power_cylindrical")

    # 1. Scale=1.0 is identity
    p1, _, wp1, wr1, w1 = dex_plan.scale_dex_grasp_local(power_g, np.ones(3))
    np.testing.assert_allclose(p1, power_g["position"], atol=1e-6)
    np.testing.assert_allclose(wp1, power_g["wrist_position"], atol=1e-6)
    assert np.isclose(w1, power_g["width"], atol=1e-6)

    # 2. Scale=1.25 shifts wrist outward by 0.5*(w_sc - w_raw) rather than stretching the 13 cm hand by 1.25x
    p2, r2, wp2, _, w2 = dex_plan.scale_dex_grasp_local(power_g, np.full(3, 1.25))
    assert np.isclose(w2, 1.25 * power_g["width"], atol=1e-5)
    orig_dist = float(np.linalg.norm(wp1 - p1))
    scaled_dist = float(np.linalg.norm(wp2 - p2))
    # Delta distance should be ~0.5 * 0.25 * width (~9 mm), NOT 0.25 * 0.14 (~35 mm)
    assert abs((scaled_dist - orig_dist) - 0.5 * (w2 - w1)) < 0.005

    # 3. Closure adaptation: larger object relaxes closure, smaller tightens, both within contract limits
    q_pre_big, q_tgt_big = dex_plan.adapt_qpos_for_scale(
        power_g["preshape_qpos"], power_g["target_qpos"], w1, w1 * 1.25, side="left", kin=kin
    )
    q_pre_sml, q_tgt_sml = dex_plan.adapt_qpos_for_scale(
        power_g["preshape_qpos"], power_g["target_qpos"], w1, w1 * 0.80, side="left", kin=kin
    )
    # Finger flexion (index MCP_FE = idx 5) should be smaller for big object than for small object
    assert q_tgt_big[5] < q_tgt_sml[5]
    for q in (q_pre_big, q_tgt_big, q_pre_sml, q_tgt_sml):
        assert np.all(q >= kin.lower["left"] - 1e-6)
        assert np.all(q <= kin.upper["left"] + 1e-6)

    # 4. Extreme scale that violates width bounds is filtered out
    planner = dex_plan.DexHandAffordancePlanner()
    out_of_range = planner.plan_grasps(
        payload, [0.0, -0.35, 0.80], [1, 0, 0, 0], scale=4.0, grasp_types=["power_cylindrical"]
    )
    assert out_of_range == []

    # 5. Non-positive scale raises ValueError
    with pytest.raises(ValueError):
        planner.plan_grasps(payload, [0.0, -0.35, 0.80], [1, 0, 0, 0], scale=0.0)


def test_symmetry_expansion_and_part_tag_filtering():
    payload = _make_cylinder_affordance()
    # Keep a single power_cylindrical grasp with continuous Z symmetry
    one_g = next(g for g in payload["grasps"]["sharpa_wave"] if g["grasp_type"] == "power_cylindrical")
    one_g = dict(one_g, part_tag="body", verified=True, sim_score=0.9)
    handle_g = dict(one_g, id="sw_h", part_tag="handle", score=0.75, sim_score=0.75)
    slim = dict(
        payload, symmetry={"z_axis": "continuous", "axis_xy": [0.0, 0.0]}, grasps={"sharpa_wave": [one_g, handle_g]}
    )

    planner = dex_plan.DexHandAffordancePlanner(continuous_sym_samples=6)
    cands = planner.plan_grasps(slim, [0.0, -0.35, 0.82], [1, 0, 0, 0], part_tags=["body"], top_k=16)
    assert len(cands) >= 4  # expanded across azimuths
    assert all(c.part_tag == "body" for c in cands)

    # exclude_tags removes body, leaves handle
    only_handle = planner.plan_grasps(slim, [0.0, -0.35, 0.82], [1, 0, 0, 0], exclude_tags=["body"], top_k=16)
    assert len(only_handle) > 0
    assert all(c.part_tag == "handle" for c in only_handle)

    # prefer_tags boosts handle above body even though handle had lower sim_score
    pref = planner.plan_grasps(slim, [0.0, -0.35, 0.82], [1, 0, 0, 0], prefer_tags=["handle"], top_k=8)
    assert pref[0].part_tag == "handle"


def test_tabletop_and_obstacle_collision_filtering():
    payload = _make_cylinder_affordance(radius=0.036, height=0.11)
    planner = dex_plan.DexHandAffordancePlanner(continuous_sym_samples=4)
    obj_pos = np.array([0.0, -0.35, 0.80])
    obj_quat = [1.0, 0.0, 0.0, 0.0]

    # Normal tabletop below object (0.745): candidates pass
    ok_cands = planner.plan_grasps(payload, obj_pos, obj_quat, table_z=0.745, top_k=8)
    assert len(ok_cands) > 0

    # Raised tabletop cutting through object (0.95): all candidates rejected
    blocked_table = planner.plan_grasps(payload, obj_pos, obj_quat, table_z=0.95, top_k=8)
    assert blocked_table == []

    # Large distractor sphere engulfing the target blocks all grasps
    big_obs = [{"position": obj_pos, "radius": 0.35}]
    blocked_sphere = planner.plan_grasps(payload, obj_pos, obj_quat, table_z=0.745, obstacles=big_obs, top_k=8)
    assert blocked_sphere == []

    # Large AABB engulfing the target blocks all grasps
    big_aabb = [{"min": obj_pos - 0.35, "max": obj_pos + 0.35}]
    blocked_aabb = planner.plan_grasps(payload, obj_pos, obj_quat, table_z=0.745, obstacles=big_aabb, top_k=8)
    assert blocked_aabb == []

    # Far-away distractor does not block
    far_obs = [{"position": obj_pos + np.array([0.6, 0.0, 0.0]), "radius": 0.05}]
    unblocked = planner.plan_grasps(payload, obj_pos, obj_quat, table_z=0.745, obstacles=far_obs, top_k=8)
    assert len(unblocked) > 0


def test_wholebody_posture_ik_filter_and_dual_mode_routing():
    # Whole-body posture: power_* uses column lift and forward waist pitch; tripod_pinch uses gentler pitch
    p_power = dex_plan.compute_wholebody_posture("power_cylindrical", [-0.15, -0.35], 0.75, [0.0, 0.0], 0.07)
    p_pinch = dex_plan.compute_wholebody_posture("tripod_pinch", [-0.15, -0.35], 0.75, [0.0, 0.0], 0.07)
    assert p_power.use_column_lift is True
    assert p_pinch.use_column_lift is False
    assert p_power.waist_grasp[1] > p_pinch.waist_grasp[1]
    assert p_power.lift_up[0] > p_power.lift_grasp[0]

    # Shoulder base prediction shifts Z by dlift and preserves orthonormal rotation
    base0 = np.eye(4)
    base0[:3, 3] = [0.2, -0.6, 1.1]
    base1 = dex_plan.predict_shoulder_base_transform(base0, 0.07, [0.0, 0.0], 0.0, [0.0, 0.25])
    assert np.isclose(np.linalg.det(base1[:3, :3]), 1.0, atol=1e-6)
    assert base1[1, 3] > base0[1, 3]  # pitching forward (+0.25 rad) moves shoulder forward in +Y
    assert base1[2, 3] < base0[2, 3]  # lowering lift + pitching forward lowers shoulder Z

    # Dual-mode execution routing + PinchBridgeSpec
    cube_payload = _make_small_box_affordance()
    planner = dex_plan.DexHandAffordancePlanner()
    pinch_cands = planner.plan_grasps(
        cube_payload, [0.0, -0.35, 0.77], [1, 0, 0, 0], grasp_types=["tripod_pinch"], top_k=4
    )
    assert len(pinch_cands) > 0
    assert pinch_cands[0].execution_mode == "pinch_bridge"
    bridge = dex_plan.build_pinch_bridge_spec(pinch_cands[0])
    np.testing.assert_allclose(bridge.openness_to_qpos(1.0), bridge.preshape_qpos)
    np.testing.assert_allclose(bridge.openness_to_qpos(0.0), bridge.target_qpos)

    # IK filter callback rejects candidates when returning (False, ...) and penalizes score otherwise
    seen_calls = []

    def _ik_cb(wp, wr, gt, w):
        seen_calls.append((gt, w))
        return (wp[2] > 0.75, 0.12)

    cyl_payload = _make_cylinder_affordance()
    ik_cands = planner.plan_grasps(cyl_payload, [0.0, -0.35, 0.82], [1, 0, 0, 0], ik_filter=_ik_cb, top_k=4)
    assert len(seen_calls) > 0
    assert len(ik_cands) > 0
    assert all(c.wrist_position[2] > 0.75 for c in ik_cands)

    reject_all = planner.plan_grasps(
        cyl_payload, [0.0, -0.35, 0.82], [1, 0, 0, 0], ik_filter=lambda *_: (False, 1.0), top_k=4
    )
    assert reject_all == []


def test_sidecar_resolution_and_failure_attribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = _make_small_box_affordance()
    sidecar = tmp_path / "cube" / "00000" / "affordance.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    class _FakeEnv:
        def entries(self, env_idx: int = 0):
            return {"target": {"category": "cube", "category_idx": 0}}

    monkeypatch.setenv("OPENFLYWHEEL_AFFORDANCE_DIR", str(tmp_path))
    loaded = dex_plan.resolve_affordance_for_env(_FakeEnv(), "target")
    assert loaded is not None
    assert loaded["asset_id"] == "Rigid/test_cube/00000"

    # Failure attribution cases
    labels = ["distractor_0", "target"]
    start = np.array([[0.2, -0.35, 0.77], [-0.1, -0.35, 0.77]])
    # 1. ok
    poses_ok = np.zeros((40, 2, 7))
    poses_ok[:, 0, :3] = start[0]
    poses_ok[:, 1, :3] = start[1] + [0.0, 0.0, 0.12]
    assert dex_plan.classify_pick_failure(poses_ok, labels, start, hold_frames=30) == "ok"

    # 2. distractor_moved
    poses_dist = poses_ok.copy()
    poses_dist[-1, 0, 0] += 0.05
    assert dex_plan.classify_pick_failure(poses_dist, labels, start, hold_frames=30) == "distractor_moved"

    # 3. dropped_during_lift
    poses_drop = poses_ok.copy()
    poses_drop[-5:, 1, 2] = start[1, 2] + 0.01
    assert dex_plan.classify_pick_failure(poses_drop, labels, start, hold_frames=30) == "dropped_during_lift"

    # 4. no_lift
    poses_none = np.zeros((40, 2, 7))
    poses_none[:, 0, :3] = start[0]
    poses_none[:, 1, :3] = start[1]
    assert dex_plan.classify_pick_failure(poses_none, labels, start, hold_frames=30) == "no_lift"

    # 5. empty_trajectory
    assert dex_plan.classify_pick_failure(np.zeros((0, 2, 7)), labels, start) == "empty_trajectory"


def test_plugin_module_exports_and_temperature_sampling():
    from scripts.datagen.plugins import g1d_pick_anything_sharpa_wave as plugin

    assert plugin.TASK == "g1d_pick_anything"
    assert plugin.ACCEPTANCE == "pick_anything_sharpa_wave_v1"
    assert plugin.STEP_LIM == 900
    assert hasattr(plugin, "Episode")
    assert hasattr(plugin, "audit")

    # Deterministic softmax sampling with rng
    payload = _make_cylinder_affordance()
    planner = dex_plan.DexHandAffordancePlanner()
    c1 = planner.plan_grasps(
        payload, [0.0, -0.35, 0.82], [1, 0, 0, 0], sample_temperature=0.2, rng=np.random.default_rng(42), top_k=6
    )
    c2 = planner.plan_grasps(
        payload, [0.0, -0.35, 0.82], [1, 0, 0, 0], sample_temperature=0.2, rng=np.random.default_rng(42), top_k=6
    )
    assert [c.grasp_id for c in c1] == [c.grasp_id for c in c2]
    assert aff.SHARPA_WAVE_DOF == 22


def test_tripod_antipodal_flip_and_topdown_wrist_clearance_preference():
    # Non-symmetric small box with a single tripod_pinch grasp -> planner expands 180-deg antipodal flip
    payload = _make_small_box_affordance()
    pinch_g = next(g for g in payload["grasps"]["sharpa_wave"] if g["grasp_type"] == "tripod_pinch")
    slim = dict(payload, symmetry={"z_axis": "none"}, grasps={"sharpa_wave": [pinch_g]})
    planner = dex_plan.DexHandAffordancePlanner(table_clear=0.015)
    cands = planner.plan_grasps(slim, [0.0, -0.35, 0.77], [1, 0, 0, 0], table_z=0.75, top_k=8)
    ids = [c.grasp_id for c in cands]
    assert any(gid.endswith("_f") for gid in ids), f"expected 180-deg antipodal flip in {ids}"

    # Top-down high-wrist tripod_pinch outranks 45-deg tilted low-wrist tripod_pinch on a flat 2.5 cm object
    g_top = dict(
        pinch_g,
        id="sw_top",
        verified=True,
        sim_score=0.90,
        position=[0.0, 0.0, 0.005],
        wrist_position=[0.0, 0.0, 0.16],
        approach_dir=[0.0, 0.0, -1.0],
        quaternion_wxyz=[0.70710678, 0.0, 0.70710678, 0.0],
    )
    # Build a valid SO(3) semantic frame for 45-deg tilted approach [0, 0.707, -0.707]
    r_tilt = np.array([[0.0, -1.0, 0.0], [0.70710678, 0.0, 0.70710678], [-0.70710678, 0.0, 0.70710678]])
    from scipy.spatial.transform import Rotation

    q_xyzw = Rotation.from_matrix(r_tilt).as_quat()
    g_tilt = dict(
        pinch_g,
        id="sw_tilt",
        verified=True,
        sim_score=0.98,
        position=[0.0, 0.0, 0.005],
        wrist_position=[0.0, -0.08, 0.085],
        approach_dir=[0.0, 0.70710678, -0.70710678],
        quaternion_wxyz=[float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])],
    )
    flat_payload = dict(payload, symmetry={"z_axis": "none"}, grasps={"sharpa_wave": [g_tilt, g_top]})
    ranked = planner.plan_grasps(flat_payload, [0.0, -0.35, 0.728], [1, 0, 0, 0], table_z=0.715, top_k=8)
    assert len(ranked) >= 1
    assert ranked[0].grasp_id.startswith("sw_top")
