"""CPU unit tests for Sharpa Wave Stage 1 affordance annotation & contract validation."""

from __future__ import annotations

import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts.dexhands import kinematics as dex_kin
from scripts.objects import affordance as aff, affordance_dexhand as dex_aff

# ---------------------------------------------------------------------------
# Synthetic mesh builders (pure NumPy, zero external assets)
# ---------------------------------------------------------------------------


def _make_box_mesh(lx: float, ly: float, lz: float, center: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0
    cx, cy, cz = center
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _make_cylinder_mesh(radius: float, height: float, n_seg: int = 24, z_offset: float = 0.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    z0, z1 = z_offset - height / 2.0, z_offset + height / 2.0
    bot = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_seg, z0)])
    top = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_seg, z1)])
    centers = np.array([[0.0, 0.0, z0], [0.0, 0.0, z1]], dtype=np.float64)
    verts = np.vstack([bot, top, centers])
    c_bot, c_top = 2 * n_seg, 2 * n_seg + 1
    faces = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([i, j, n_seg + j])
        faces.append([i, n_seg + j, n_seg + i])
        faces.append([c_bot, j, i])
        faces.append([c_top, n_seg + i, n_seg + j])
    return verts, np.asarray(faces, dtype=np.int64)


def _make_sphere_mesh(radius: float, n_lat: int = 12, n_lon: int = 20):
    verts = [[0.0, 0.0, -radius]]
    for i in range(1, n_lat):
        phi = -0.5 * np.pi + np.pi * i / n_lat
        for j in range(n_lon):
            th = 2.0 * np.pi * j / n_lon
            verts.append([radius * np.cos(phi) * np.cos(th), radius * np.cos(phi) * np.sin(th), radius * np.sin(phi)])
    verts.append([0.0, 0.0, radius])
    v = np.asarray(verts, dtype=np.float64)
    faces = []
    top_idx = len(v) - 1
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([0, 1 + jn, 1 + j])
    for i in range(n_lat - 2):
        row0 = 1 + i * n_lon
        row1 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([row0 + j, row0 + jn, row1 + jn])
            faces.append([row0 + j, row1 + jn, row1 + j])
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([last_row + j, last_row + jn, top_idx])
    return v, np.asarray(faces, dtype=np.int64)


def _merge_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    v_list, f_list = [], []
    off = 0
    for v, f in meshes:
        v_list.append(v)
        f_list.append(f + off)
        off += len(v)
    return np.vstack(v_list), np.vstack(f_list)


def _make_mug_with_handle() -> tuple[np.ndarray, np.ndarray]:
    body = _make_cylinder_mesh(radius=0.040, height=0.105)
    # Side vertical handle bar protruding along +X at x=0.068 (thickness 16 mm)
    bar = _make_box_mesh(0.016, 0.016, 0.065, center=(0.068, 0.0, 0.005))
    top_strut = _make_box_mesh(0.030, 0.014, 0.012, center=(0.053, 0.0, 0.032))
    bot_strut = _make_box_mesh(0.030, 0.014, 0.012, center=(0.053, 0.0, -0.022))
    return _merge_meshes([body, bar, top_strut, bot_strut])


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts]
    lines.extend(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}" for f in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fast_fk_matches_reference_kinematics_both_sides():
    kin = dex_aff.get_sharpa_kinematics()
    urdf_root = ET.parse(dex_aff.SHARPA_URDF_PATH).getroot()
    rng = np.random.default_rng(7)

    for side in ("left", "right"):
        q = rng.uniform(kin.lower[side], kin.upper[side])
        fast_tf = kin.fk(q, side=side)
        ref_tf = dex_kin.fk(urdf_root, dict(zip(kin.joints[side], q)))
        T_flange_inv = np.linalg.inv(ref_tf[f"{side}_hand_flange"])
        for link, T_fast in fast_tf.items():
            T_ref = T_flange_inv @ ref_tf[link]
            assert np.allclose(T_fast, T_ref, atol=1e-10), f"FK mismatch on {link}"


def test_refitted_tripod_pinch_clearance_and_monotonicity():
    kin = dex_aff.get_sharpa_kinematics()
    geom = kin.tripod_pinch_geometry()

    assert geom.open_pad_clearance_m >= 0.075, f"Expected >= 75 mm, got {geom.open_pad_clearance_m * 1000:.1f} mm"
    assert geom.mp_corridor_clearance_m >= 0.050
    assert geom.closed_pad_gap_m <= 0.010

    # Verify contract limits and |MCP_AA| <= 0.04 rad across full width range
    prev_pre_span = 0.0
    for w in (0.010, 0.025, 0.040, 0.055, 0.070):
        q_pre, q_tgt = dex_aff.tripod_pinch_qpos(w, kin)
        for q in (q_pre, q_tgt):
            assert np.all(q >= kin.lower["left"] - 1e-6)
            assert np.all(q <= kin.upper["left"] + 1e-6)
            for aa_idx in dex_aff.MCP_AA_INDICES:
                assert abs(float(q[aa_idx])) <= dex_aff.MCP_AA_SAFE_LIMIT + 1e-6
        pads = kin.elastomer_pads(q_pre, side="left")
        span = float(np.linalg.norm(pads["thumb"][0] - 0.5 * (pads["index"][0] + pads["middle"][0])))
        assert span >= prev_pre_span - 1e-4
        prev_pre_span = span

    # Re-running the optimizer reproduces >= 75 mm clearance
    q_c, q_o, clear_m = dex_aff.refit_tripod_pinch_template(kin)
    assert clear_m >= 0.075
    assert q_c.shape == (22,) and q_o.shape == (22,)


def test_left_to_right_mirror_fk_error_under_1mm_all_primitives():
    kin = dex_aff.get_sharpa_kinematics()
    urdf_root = ET.parse(dex_aff.SHARPA_URDF_PATH).getroot()

    v_mug, f_mug = _make_mug_with_handle()
    v_blk, f_blk = _make_box_mesh(0.032, 0.032, 0.032)
    v_sph, f_sph = _make_sphere_mesh(0.038)

    samples = []
    for verts, faces in ((v_mug, f_mug), (v_blk, f_blk), (v_sph, f_sph)):
        samples.extend(dex_aff.generate_sharpa_wave_grasps(verts, faces, max_grasps=28, seed=0, kin=kin))

    by_type: dict[str, dict] = {}
    for g in samples:
        by_type.setdefault(g["grasp_type"], g)
    assert set(by_type) == {"power_cylindrical", "power_spherical", "tripod_pinch", "handle_hook"}

    for gtype, g_left in by_type.items():
        default_key = "approach_dir" if gtype in ("power_cylindrical", "handle_hook") else "closing_axis"
        u = np.asarray(g_left[default_key], dtype=np.float64)
        u /= np.linalg.norm(u)
        c = np.asarray(g_left["position"], dtype=np.float64)
        M_u = np.eye(3) - 2.0 * np.outer(u, u)

        g_right = dex_aff.mirror_grasp_left_to_right(g_left)
        assert g_right["hand_side"] == "right"
        assert g_right["ee_frame"] == "right_hand_flange"
        # Semantic basis [approach_dir, closing_axis, thumb_dir] must remain right-handed and match quaternion_wxyz
        R_pos_r = dex_aff._quat_wxyz_to_rot(g_right["quaternion_wxyz"])
        R_sem_r = np.column_stack([g_right["approach_dir"], g_right["closing_axis"], g_right["thumb_dir"]])
        assert np.linalg.det(R_sem_r) > 0.99
        assert np.allclose(R_pos_r, R_sem_r, atol=2e-4), f"{gtype} mirrored semantic frame mismatch"
        # Reject mirroring an already-right grasp
        with pytest.raises(ValueError):
            dex_aff.mirror_grasp_left_to_right(g_right)

        # Validate mirrored grasp against schema
        payload = {
            "schema_version": 1,
            "asset_id": "Rigid/test/00000",
            "mesh_hash": "sha256:0123456789abcdef",
            "scale_ref": 1.0,
            "bounds": {"min": [-0.05, -0.05, -0.05], "max": [0.05, 0.05, 0.05], "extents": [0.1, 0.1, 0.1]},
            "symmetry": {"z_axis": "none", "fold": 1},
            "grasps": {"sharpa_wave": [g_right]},
        }
        errs, _ = aff.validate_affordance(payload)
        assert errs == [], f"Mirrored {gtype} failed validation: {errs}"

        # Compare every link position in object frame using full reference URDF kinematics.fk()
        R_wl = dex_aff._quat_wxyz_to_rot(g_left["wrist_quaternion_wxyz"])
        p_wl = np.asarray(g_left["wrist_position"], dtype=np.float64)
        R_wr = dex_aff._quat_wxyz_to_rot(g_right["wrist_quaternion_wxyz"])
        p_wr = np.asarray(g_right["wrist_position"], dtype=np.float64)

        for q_key in ("preshape_qpos", "target_qpos"):
            tf_l = dex_kin.fk(urdf_root, dict(zip(kin.joints["left"], g_left[q_key])))
            tf_r = dex_kin.fk(urdf_root, dict(zip(kin.joints["right"], g_right[q_key])))
            T_lf_inv = np.linalg.inv(tf_l["left_hand_flange"])
            T_rf_inv = np.linalg.inv(tf_r["right_hand_flange"])
            max_err = 0.0
            for lname in tf_l:
                if not lname.startswith("left_"):
                    continue
                rname = "right_" + lname[5:]
                if rname not in tf_r:
                    continue
                if not any(k in lname for k in ("hand", "thumb", "index", "middle", "ring", "pinky")):
                    continue
                pl_obj = p_wl + R_wl @ (T_lf_inv @ tf_l[lname])[:3, 3]
                pr_obj = p_wr + R_wr @ (T_rf_inv @ tf_r[rname])[:3, 3]
                expected_r = c + M_u @ (pl_obj - c)
                max_err = max(max_err, float(np.linalg.norm(pr_obj - expected_r)))
            assert max_err < 1e-3, f"{gtype}/{q_key} left->right FK mirror error {max_err * 1000:.3f} mm >= 1 mm"


def test_validate_affordance_sharpa_wave_contract_and_error_cases():
    v_cyl, f_cyl = _make_cylinder_mesh(radius=0.036, height=0.12)
    payload = dex_aff.annotate_mesh_sharpa_wave(v_cyl, f_cyl, asset_id="Rigid/cup/00000", max_grasps=12, seed=0)
    errs, warns = aff.validate_affordance(payload)
    assert errs == []
    assert warns == []
    assert len(payload["grasps"]["sharpa_wave"]) > 0

    def _corrupt(**mut):
        bad = json.loads(json.dumps(payload))
        g0 = bad["grasps"]["sharpa_wave"][0]
        g0.update(mut)
        return aff.validate_affordance(bad)[0]

    # Invalid grasp_type
    assert any("grasp_type" in e for e in _corrupt(grasp_type="unknown_grip"))
    # Mismatched ee_frame
    assert any("ee_frame" in e for e in _corrupt(hand_side="left", ee_frame="right_hand_flange"))
    # Wrong qpos length
    assert any("preshape_qpos" in e for e in _corrupt(preshape_qpos=[0.0] * 21))
    # Out of contract joint limits (index_DIP upper is 1.3963)
    bad_q = list(payload["grasps"]["sharpa_wave"][0]["target_qpos"])
    bad_q[8] = 2.5
    assert any("outside contract limits" in e for e in _corrupt(target_qpos=bad_q))
    # Non-unit wrist quaternion
    assert any("wrist_quaternion_wxyz" in e for e in _corrupt(wrist_quaternion_wxyz=[2.0, 0.0, 0.0, 0.0]))
    # Invalid close_schedule
    assert any(
        "thumb_first_joints" in e for e in _corrupt(close_schedule={"thumb_first_joints": -1, "thumb_lead_ratio": 0.3})
    )
    assert any(
        "thumb_lead_ratio" in e for e in _corrupt(close_schedule={"thumb_first_joints": 5, "thumb_lead_ratio": 1.5})
    )
    # Invalid contact_fingers (empty, unknown, duplicates)
    assert any("contact_fingers" in e for e in _corrupt(contact_fingers=[]))
    assert any("contact_fingers" in e for e in _corrupt(contact_fingers=["thumb", "claw"]))
    assert any("contact_fingers" in e for e in _corrupt(contact_fingers=["thumb", "thumb"]))


def test_ten_representative_objects_coverage_limits_and_speed():
    kin = dex_aff.get_sharpa_kinematics()
    objects = [
        ("cup", *_make_cylinder_mesh(0.038, 0.11)),
        ("mug_handle", *_make_mug_with_handle()),
        ("bottle", *_make_cylinder_mesh(0.032, 0.18)),
        ("apple", *_make_sphere_mesh(0.038)),
        ("bowl", *_make_cylinder_mesh(0.048, 0.060)),
        ("cube_30mm", *_make_box_mesh(0.030, 0.030, 0.030)),
        ("small_box", *_make_box_mesh(0.055, 0.045, 0.075)),
        ("tall_box", *_make_box_mesh(0.065, 0.065, 0.140)),
        ("cap_disc", *_make_cylinder_mesh(0.024, 0.022)),
        ("slim_bar", *_make_box_mesh(0.026, 0.085, 0.030)),
    ]

    t0 = time.perf_counter()
    results = {}
    for name, verts, faces in objects:
        res = dex_aff.annotate_mesh_sharpa_wave(
            verts, faces, asset_id=f"Rigid/{name}/00000", max_grasps=28, seed=0, kin=kin
        )
        errs, _ = aff.validate_affordance(res)
        assert errs == [], f"{name} validation errors: {errs}"
        grasps = res["grasps"]["sharpa_wave"]
        assert len(grasps) > 0, f"{name} produced 0 grasps"
        for g in grasps:
            for q_key in ("preshape_qpos", "target_qpos"):
                q = np.asarray(g[q_key], dtype=np.float64)
                assert np.all(q >= kin.lower["left"] - 1e-4)
                assert np.all(q <= kin.upper["left"] + 1e-4)
                for aa_idx in dex_aff.MCP_AA_INDICES:
                    assert abs(float(q[aa_idx])) <= dex_aff.MCP_AA_SAFE_LIMIT + 1e-4
        results[name] = {g["grasp_type"] for g in grasps}
    elapsed_per_obj = (time.perf_counter() - t0) / len(objects)
    assert elapsed_per_obj < 3.0, f"Too slow: {elapsed_per_obj:.2f} s/object (threshold < 3.0 s)"

    assert "power_cylindrical" in results["cup"]
    assert "handle_hook" in results["mug_handle"]
    assert "power_cylindrical" in results["bottle"]
    assert "power_spherical" in results["apple"]
    assert "tripod_pinch" in results["cube_30mm"]
    assert "tripod_pinch" in results["cap_disc"]


def test_cli_merge_and_edge_cases(tmp_path: Path):
    v_box, f_box = _make_box_mesh(0.055, 0.055, 0.10)
    obj_path = tmp_path / "box.obj"
    out_path = tmp_path / "affordance.json"
    _write_obj(obj_path, v_box, f_box)

    # First annotate dex1_1 with affordance.py, then merge sharpa_wave via affordance_dexhand.py CLI
    aff.annotate_asset(obj_path, output=out_path, gripper_types=("dex1_1",), seed=0)
    assert dex_aff.main(["annotate", str(obj_path), "-o", str(out_path), "--merge", "--seed", "0"]) == 0
    assert dex_aff.main(["inspect", str(out_path), "--validate"]) == 0
    assert dex_aff.main(["verify-pinch"]) == 0

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert "dex1_1" in merged["grasps"] and len(merged["grasps"]["dex1_1"]) > 0
    assert "sharpa_wave" in merged["grasps"] and len(merged["grasps"]["sharpa_wave"]) > 0

    # Edge cases: empty mesh, oversized mesh, bad mirror_normal, bad qpos shape
    assert dex_aff.generate_sharpa_wave_grasps(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)) == []
    v_huge, f_huge = _make_box_mesh(0.35, 0.35, 0.35)
    assert dex_aff.generate_sharpa_wave_grasps(v_huge, f_huge) == []

    with pytest.raises(ValueError):
        dex_aff.mirror_grasp_left_to_right(merged["grasps"]["sharpa_wave"][0], mirror_normal=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        dex_aff.get_sharpa_kinematics().clip_qpos(np.zeros(10))


def test_right_hand_capsules_mirror_mass_routing_and_empty_pts():
    kin = dex_aff.get_sharpa_kinematics()
    M = np.diag([1.0, -1.0, 1.0])

    # 1. Right-hand link_capsules must match diag(1, -1, 1) reflection of left-hand capsules (< 0.01 mm)
    for q in (dex_aff.TRIPOD_PINCH_OPEN_Q, dex_aff.TRIPOD_PINCH_CLOSED_Q, np.zeros(22)):
        p0_l, p1_l, r_l = kin.link_capsules(q, side="left")
        p0_r, p1_r, r_r = kin.link_capsules(q, side="right")
        assert np.allclose(r_l, r_r)
        assert np.max(np.linalg.norm(p0_r - (p0_l @ M), axis=1)) < 1e-5
        assert np.max(np.linalg.norm(p1_r - (p1_l @ M), axis=1)) < 1e-5

    # 2. Empty surface_pts must not raise ValueError in check_hand_capsules_collision
    ok, palm_c, table_c = dex_aff.check_hand_capsules_collision(
        np.zeros((0, 3)), p0_l, p1_l, r_l, np.zeros(3), np.eye(3), support_z=-1.0
    )
    assert ok and np.isinf(palm_c) and table_c > 0.0

    # 3. Heavy object (mass > 0.3 kg) suppresses body_top tripod_pinch when power grasps exist
    v_cup, f_cup = _make_cylinder_mesh(0.038, 0.11)
    heavy = dex_aff.annotate_mesh_sharpa_wave(
        v_cup, f_cup, asset_id="Rigid/heavy_cup/00000", metadata={"mass": 0.8}, max_grasps=28, seed=0, kin=kin
    )
    sw_heavy = heavy["grasps"]["sharpa_wave"]
    assert len(sw_heavy) > 0
    assert sw_heavy[0]["grasp_type"] in ("power_cylindrical", "power_spherical")
    assert all(g["part_tag"] in ("rim", "lid", "cap") for g in sw_heavy if g["grasp_type"] == "tripod_pinch")

    # 4. Pure top-down tripod_pinch approach is explored on a 30 mm cube
    v_cube, f_cube = _make_box_mesh(0.030, 0.030, 0.030)
    cube_grasps = dex_aff.generate_sharpa_wave_grasps(v_cube, f_cube, max_grasps=28, seed=0, kin=kin)
    assert any(g["grasp_type"] == "tripod_pinch" and g["approach_dir"][2] <= -0.95 for g in cube_grasps)
