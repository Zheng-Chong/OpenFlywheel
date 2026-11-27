"""CPU unit tests for Sharpa Wave Stage 2 physics & tactile verification."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.objects import (
    affordance as aff,
    affordance_dexhand as dex_aff,
    verify_affordance_dexhand_sim as vdex,
)
from scripts.objects.test_affordance_dexhand import _make_cylinder_mesh


def test_build_left_hand_urdf_structure_and_meshes(tmp_path: Path):
    out = vdex.build_left_hand_urdf(vdex.SHARPA_URDF_PATH, tmp_path / "left_sharpa.urdf")
    assert out.is_file()

    root = ET.parse(out).getroot()
    link_names = {el.get("name") for el in root.findall("link")}
    joint_names = {el.get("name") for el in root.findall("joint")}

    # 35 left Sharpa Wave links + 6 virtual wrist links
    assert len(link_names) == 41
    assert len(joint_names) == 40

    # All 6 virtual wrist joints + all 22 active Sharpa Wave left joints + 5 elastomer contact links
    for j in vdex.VIRTUAL_WRIST_JOINTS:
        assert j in joint_names
    contract = json.loads(vdex.SHARPA_CONTRACT_PATH.read_text(encoding="utf-8"))
    for j in contract["hands"]["left"]["joints"]:
        assert j in joint_names
    for link in vdex.LEFT_CONTACT_LINKS:
        assert link in link_names
    assert "left_hand_flange" in link_names
    # No right-hand links leaked
    assert not any(n.startswith("right_") for n in link_names if n)

    # All mesh filenames resolved to existing absolute files
    meshes = [m.get("filename", "") for m in root.findall(".//mesh")]
    assert len(meshes) > 20
    for mpath in meshes:
        assert Path(mpath).is_absolute()
        assert Path(mpath).is_file(), f"Missing resolved mesh: {mpath}"


def test_interpolate_close_qpos_thumb_first_and_simultaneous():
    q0 = np.zeros(22, dtype=np.float64)
    q1 = np.ones(22, dtype=np.float64)

    # At p=0 -> preshape
    np.testing.assert_allclose(
        vdex.interpolate_close_qpos(q0, q1, 0.0, thumb_first_joints=5, thumb_lead_ratio=0.35), q0
    )

    # At p=0.35 -> thumb finished (1.0), fingers still at preshape (0.0)
    q_mid = vdex.interpolate_close_qpos(q0, q1, 0.35, thumb_first_joints=5, thumb_lead_ratio=0.35)
    np.testing.assert_allclose(q_mid[:5], 1.0, atol=1e-9)
    np.testing.assert_allclose(q_mid[5:], 0.0, atol=1e-9)

    # At p=0.675 -> thumb held at 1.0, fingers halfway at 0.5
    q_wrap = vdex.interpolate_close_qpos(q0, q1, 0.675, thumb_first_joints=5, thumb_lead_ratio=0.35)
    np.testing.assert_allclose(q_wrap[:5], 1.0, atol=1e-9)
    np.testing.assert_allclose(q_wrap[5:], 0.5, atol=1e-9)

    # At p=1.0 -> target
    np.testing.assert_allclose(
        vdex.interpolate_close_qpos(q0, q1, 1.0, thumb_first_joints=5, thumb_lead_ratio=0.35), q1
    )

    # Simultaneous mode (tripod_pinch: thumb_first_joints=0)
    q_sim = vdex.interpolate_close_qpos(q0, q1, 0.4, thumb_first_joints=0, thumb_lead_ratio=0.0)
    np.testing.assert_allclose(q_sim, 0.4, atol=1e-9)

    # Joint limit clamping
    lo = np.full(22, 0.2)
    hi = np.full(22, 0.8)
    q_clamped = vdex.interpolate_close_qpos(q0, q1, 1.0, lower_limits=lo, upper_limits=hi)
    np.testing.assert_allclose(q_clamped, 0.8)

    with pytest.raises(ValueError):
        vdex.interpolate_close_qpos(np.zeros(22), np.zeros(10), 0.5)


def test_dual_gate_power_and_pinch_scoring():
    power_forces = {"thumb": 2.5, "index": 1.4, "middle": 1.2, "ring": 0.9, "pinky": 0.5}
    res = vdex.score_dexhand_trial(
        "sw000", "power_cylindrical", lift_height_m=0.148, slip_m=0.005, slip_deg=2.5, finger_forces_n=power_forces
    )
    assert res.passed is True
    assert 0.5 <= res.sim_score <= 1.0
    assert res.contact_fingers_n == 5
    assert res.reason == "ok"
    m = res.to_metrics()
    assert m["thumb_force_n"] == 2.5
    assert m["opposing_force_n"] == 4.0

    # Power grasp fails if thumb force <= 0.30 N
    no_thumb = dict(power_forces, thumb=0.25)
    r_nt = vdex.score_dexhand_trial("sw001", "power_cylindrical", 0.14, 0.005, 2.0, no_thumb)
    assert not r_nt.passed and r_nt.reason == "insufficient_thumb_contact"

    # Power grasp fails if only 1 opposing finger > 0.20 N
    one_opp = {"thumb": 2.0, "index": 1.5, "middle": 0.15, "ring": 0.0, "pinky": 0.0}
    r_oo = vdex.score_dexhand_trial("sw002", "power_spherical", 0.14, 0.005, 2.0, one_opp)
    assert not r_oo.passed and r_oo.reason == "insufficient_opposing_contact"

    # Tripod pinch passes with thumb > 0.25 N and index > 0.25 N (even with ring/pinky = 0)
    pinch_forces = {"thumb": 0.9, "index": 0.8, "middle": 0.1, "ring": 0.0, "pinky": 0.0}
    r_pinch = vdex.score_dexhand_trial("sw003", "tripod_pinch", 0.12, 0.01, 5.0, pinch_forces)
    assert r_pinch.passed is True
    assert r_pinch.contact_fingers_n == 2

    # Tripod pinch rejects if neither index nor middle > 0.25 N (even if ring is high)
    bad_pinch = {"thumb": 0.9, "index": 0.1, "middle": 0.1, "ring": 1.5, "pinky": 0.0}
    r_bp = vdex.score_dexhand_trial("sw004", "tripod_pinch", 0.12, 0.01, 5.0, bad_pinch)
    assert not r_bp.passed and r_bp.reason == "insufficient_opposing_contact"

    # Handle hook passes with thumb brace (>0.20 N) + >=2 fingers (>0.20 N)
    hook_brace = {"thumb": 0.25, "index": 0.8, "middle": 0.6, "ring": 0.0, "pinky": 0.0}
    assert vdex.score_dexhand_trial("sw005", "handle_hook", 0.13, 0.01, 4.0, hook_brace).passed is True
    # Handle hook also passes pure multi-finger gravity hook (>=2 fingers >0.30 N, sum >1.0 N)
    hook_grav = {"thumb": 0.05, "index": 0.9, "middle": 0.7, "ring": 0.0, "pinky": 0.0}
    assert vdex.score_dexhand_trial("sw006", "handle_hook", 0.13, 0.01, 4.0, hook_grav).passed is True
    # Handle hook rejects weak single-finger or under-loaded contact
    hook_weak = {"thumb": 0.05, "index": 0.4, "middle": 0.35, "ring": 0.0, "pinky": 0.0}
    assert vdex.score_dexhand_trial("sw007", "handle_hook", 0.13, 0.01, 4.0, hook_weak).passed is False

    # Kinematics & boundary gates
    assert vdex.score_dexhand_trial("b1", "power_cylindrical", 0.10, 0.025, 20.0, power_forces).passed is True
    assert (
        vdex.score_dexhand_trial("b2", "power_cylindrical", 0.099, 0.01, 5.0, power_forces).reason
        == "insufficient_lift"
    )
    assert (
        vdex.score_dexhand_trial("b3", "power_cylindrical", 0.14, 0.026, 5.0, power_forces).reason
        == "excessive_slip_pos"
    )
    assert (
        vdex.score_dexhand_trial("b4", "power_cylindrical", 0.14, 0.01, 20.5, power_forces).reason
        == "excessive_slip_rot"
    )
    assert (
        vdex.score_dexhand_trial("b5", "power_cylindrical", 0.14, 0.01, 5.0, power_forces, approach_disp_m=0.035).reason
        == "approach_collision"
    )
    assert (
        vdex.score_dexhand_trial("b6", "power_cylindrical", float("nan"), 0.0, 0.0, power_forces).reason
        == "unstable_nan"
    )


def test_compute_world_wrist_waypoints():
    rot_obj = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    obj_pos = np.array([1.0, 2.0, 0.05])
    grasp = {
        "wrist_position": [0.1, 0.0, 0.0],
        "wrist_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "approach_dir": [1.0, 0.0, 0.0],
    }
    ok, reason, p_pre, p_grasp, r_grasp, p_lift = vdex.compute_world_wrist_waypoints(
        obj_pos, rot_obj, grasp, approach_dist_m=0.07, lift_height_m=0.15
    )
    assert ok and reason == "ok"
    # +X local rotated 90 deg about Z -> +Y world
    np.testing.assert_allclose(p_grasp, [1.0, 2.1, 0.05], atol=1e-9)
    np.testing.assert_allclose(p_pre, [1.0, 2.03, 0.05], atol=1e-9)
    np.testing.assert_allclose(p_lift, [1.0, 2.1, 0.20], atol=1e-9)
    np.testing.assert_allclose(r_grasp, rot_obj, atol=1e-9)

    # Upward approach rejected
    grasp_up = dict(grasp, approach_dir=[0.0, 0.0, 1.0])
    ok_u, reason_u, *_ = vdex.compute_world_wrist_waypoints(obj_pos, np.eye(3), grasp_up)
    assert not ok_u and reason_u == "upward_approach"


def test_idempotent_writeback_preserves_dex1_and_validates(tmp_path: Path):
    verts, faces = _make_cylinder_mesh(radius=0.036, height=0.11)
    payload = dex_aff.annotate_mesh_sharpa_wave(verts, faces, asset_id="Rigid/test_cup/00000", max_grasps=8, seed=0)
    # Attach a dummy dex1_1 entry to verify it is untouched
    payload["grasps"]["dex1_1"] = [{"id": "g000", "score": 0.9}]

    verifier = vdex.DryRunDexHandVerifier(score_threshold=0.50)
    once = vdex.apply_dexhand_verification(payload, verifier)
    twice = vdex.apply_dexhand_verification(once, verifier)

    assert once == twice, "Stage 2 writeback must be idempotent across repeated runs"
    assert once["grasps"]["dex1_1"] == [{"id": "g000", "score": 0.9}]

    sw = once["grasps"]["sharpa_wave"]
    assert len(sw) > 0
    assert any(g["verified"] for g in sw)
    for g in sw:
        assert "geom_score" in g and "sim_score" in g and "sim_metrics" in g
        assert "thumb_force_n" in g["sim_metrics"] and "opposing_force_n" in g["sim_metrics"]

    # Validate against schema contract (ignore dummy dex1_1 by checking sharpa_wave payload)
    check_payload = json.loads(json.dumps(once))
    check_payload["grasps"].pop("dex1_1")
    errs, _ = aff.validate_affordance(check_payload)
    assert errs == [], f"Schema validation errors: {errs}"


def test_cli_dry_run_single_and_manifest_sharding(tmp_path: Path):
    verts, faces = _make_cylinder_mesh(radius=0.038, height=0.11)
    manifest = []
    for i in range(3):
        adir = tmp_path / f"asset_{i}"
        adir.mkdir()
        aff_path = adir / "affordance.json"
        payload = dex_aff.annotate_mesh_sharpa_wave(verts, faces, asset_id=f"Rigid/cyl/{i:05d}", max_grasps=4, seed=i)
        aff_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "asset_id": f"Rigid/cyl/{i:05d}",
                "asset_dir": str(adir),
                "affordance_path": str(aff_path),
                "status": "ok",
            }
        )

    # Single-file CLI --dry-run
    out_single = tmp_path / "single_out.json"
    rc = vdex.main([str(tmp_path / "asset_0" / "affordance.json"), "-o", str(out_single), "--dry-run"])
    assert rc == 0
    d0 = json.loads(out_single.read_text(encoding="utf-8"))
    assert all("sim_metrics" in g for g in d0["grasps"]["sharpa_wave"])

    # Manifest shard CLI --dry-run (--offset 1 --limit 2)
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rc2 = vdex.main(["--manifest", str(mpath), "--offset", "1", "--limit", "2", "--dry-run"])
    assert rc2 == 0
    m_after = json.loads(mpath.read_text(encoding="utf-8"))
    assert "sim_verified_sharpa" not in m_after[0]
    assert m_after[1]["sim_verified_sharpa"] >= 1
    assert m_after[2]["sim_verified_sharpa"] >= 1

    # --skip-verified skips already-verified entries (resume support)
    aff1_path = tmp_path / "asset_1" / "affordance.json"
    aff1_before = aff1_path.read_text(encoding="utf-8")
    rc3 = vdex.main(["--manifest", str(mpath), "--skip-verified", "--dry-run"])
    assert rc3 == 0
    m_after3 = json.loads(mpath.read_text(encoding="utf-8"))
    assert m_after3[0]["sim_verified_sharpa"] >= 1
    assert aff1_path.read_text(encoding="utf-8") == aff1_before

    # --drop-failed removes failed grasps
    out_drop = tmp_path / "drop_out.json"
    rc4 = vdex.main([str(tmp_path / "asset_0" / "affordance.json"), "-o", str(out_drop), "--drop-failed", "--dry-run"])
    assert rc4 == 0
    d_drop = json.loads(out_drop.read_text(encoding="utf-8"))
    assert all(g["verified"] for g in d_drop["grasps"]["sharpa_wave"])


def test_quat_guard_velocity_gate_and_multibody_contact_force():
    # Zero-norm and non-finite quaternions rejected before SciPy Rotation
    assert vdex._is_valid_quat(np.array([1.0, 0.0, 0.0, 0.0])) is True
    assert vdex._is_valid_quat(np.array([0.0, 0.0, 0.0, 0.0])) is False
    assert vdex._is_valid_quat(np.array([1.0, float("nan"), 0.0, 0.0])) is False
    assert vdex._is_valid_quat(np.array([1.0, 0.0, 0.0])) is False

    # Multi-body contact force matrix: shape (1, 1, M, 3) sums vector across all filtered bodies
    assert vdex._extract_contact_force_mag(None) == 0.0
    assert vdex._extract_contact_force_mag(np.zeros((1, 1, 0, 3))) == 0.0
    single = np.array([[[[3.0, 4.0, 0.0]]]])
    assert vdex._extract_contact_force_mag(single) == pytest.approx(5.0)
    multi = np.array([[[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]]])
    assert vdex._extract_contact_force_mag(multi) == pytest.approx(5.0)
    nan_mat = np.array([[[[float("nan"), 0.0, 0.0]]]])
    assert np.isnan(vdex._extract_contact_force_mag(nan_mat))

    # Velocity explosion gate
    power_forces = {"thumb": 2.5, "index": 1.4, "middle": 1.2, "ring": 0.9, "pinky": 0.5}
    r_vlin = vdex.score_dexhand_trial("v1", "power_cylindrical", 0.14, 0.005, 2.0, power_forces, obj_lin_vel_mps=2.5)
    assert not r_vlin.passed and r_vlin.reason == "unstable_velocity"
    r_vang = vdex.score_dexhand_trial("v2", "power_cylindrical", 0.14, 0.005, 2.0, power_forces, obj_ang_vel_rps=18.0)
    assert not r_vang.passed and r_vang.reason == "unstable_velocity"
    r_vnan = vdex.score_dexhand_trial(
        "v3", "power_cylindrical", 0.14, 0.005, 2.0, power_forces, obj_lin_vel_mps=float("nan")
    )
    assert not r_vnan.passed and r_vnan.reason == "unstable_nan"


def test_vis_snapshot_ranking_never_downgrades_pass_to_fail():
    passed_low = vdex.DexHandVerificationResult("sw001", passed=True, sim_score=0.72, lift_height_m=0.12)
    passed_high = vdex.DexHandVerificationResult("sw002", passed=True, sim_score=0.91, lift_height_m=0.15)
    failed_high_lift = vdex.DexHandVerificationResult("sw003", passed=False, sim_score=0.0, lift_height_m=0.16)

    assert vdex._vis_snapshot_key(passed_low) > vdex._vis_snapshot_key(failed_high_lift)
    assert vdex._vis_snapshot_key(passed_high) > vdex._vis_snapshot_key(passed_low)


def test_affordance_dexhand_batch_cli_commercial_skip_and_corrupt_recovery(tmp_path: Path):
    assets_root = tmp_path / "Assets"
    out_dir = tmp_path / "out"
    verts, faces = _make_cylinder_mesh(radius=0.036, height=0.11)
    pts_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in verts)
    counts_str = ", ".join("3" for _ in faces)
    idx_str = ", ".join(f"{int(f[0])}, {int(f[1])}, {int(f[2])}" for f in faces)
    usda_text = (
        "#usda 1.0\n"
        'def Xform "World" {\n'
        '    def Mesh "body" {\n'
        f"        int[] faceVertexCounts = [{counts_str}]\n"
        f"        int[] faceVertexIndices = [{idx_str}]\n"
        f"        point3f[] points = [{pts_str}]\n"
        "    }\n"
        "}\n"
    )

    for cat, idx in [("cup", 0), ("geniesim_cup", 1)]:
        adir = assets_root / "Object" / "RoboDojo" / "Rigid" / cat / f"{idx:05d}"
        adir.mkdir(parents=True)
        (adir / "object.usda").write_text(usda_text, encoding="utf-8")

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        json.dumps({"objects": [{"cat": "cup", "idx": 0}, {"cat": "geniesim_cup", "idx": 1}]}),
        encoding="utf-8",
    )

    # First run with --commercial-only should annotate only Rigid/cup/00000
    rc = dex_aff.main(
        [
            "batch",
            "--pool",
            str(pool_path),
            "--assets-root",
            str(assets_root),
            "--output-dir",
            str(out_dir),
            "--commercial-only",
            "--max-grasps",
            "4",
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["asset_id"] == "Rigid/cup/00000"
    assert manifest[0]["status"] == "ok"
    assert manifest[0]["grasps_sharpa"] > 0

    # Second run with --skip-existing should mark it skipped
    rc2 = dex_aff.main(
        [
            "batch",
            "--pool",
            str(pool_path),
            "--assets-root",
            str(assets_root),
            "--output-dir",
            str(out_dir),
            "--commercial-only",
            "--skip-existing",
            "--max-grasps",
            "4",
        ]
    )
    assert rc2 == 0
    manifest2 = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2[0]["status"] == "skipped"

    # Corrupt existing affordance.json should NOT crash --skip-existing; it should re-annotate cleanly
    aff_file = out_dir / "cup" / "00000" / "affordance.json"
    aff_file.write_text("{corrupt json", encoding="utf-8")
    rc3 = dex_aff.main(
        [
            "batch",
            "--pool",
            str(pool_path),
            "--assets-root",
            str(assets_root),
            "--output-dir",
            str(out_dir),
            "--commercial-only",
            "--skip-existing",
            "--max-grasps",
            "4",
        ]
    )
    assert rc3 == 0
    manifest3 = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest3[0]["status"] == "ok"
    repaired = json.loads(aff_file.read_text(encoding="utf-8"))
    assert len(repaired["grasps"]["sharpa_wave"]) > 0
