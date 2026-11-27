"""Unit & integration tests for scripts/objects/affordance.py and verify_affordance_sim.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.objects import affordance as aff, verify_affordance_sim as verify_sim

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Synthetic mesh builders (guaranteed outward-pointing normals)
# ---------------------------------------------------------------------------


def make_box_mesh(sx: float, sy: float, sz: float, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box centered at `center` with outward CCW triangles."""
    cx, cy, cz = center
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],  # 0
            [cx + hx, cy - hy, cz - hz],  # 1
            [cx + hx, cy + hy, cz - hz],  # 2
            [cx - hx, cy + hy, cz - hz],  # 3
            [cx - hx, cy - hy, cz + hz],  # 4
            [cx + hx, cy - hy, cz + hz],  # 5
            [cx + hx, cy + hy, cz + hz],  # 6
            [cx - hx, cy + hy, cz + hz],  # 7
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # -Z
            [4, 5, 6],
            [4, 6, 7],  # +Z
            [0, 1, 5],
            [0, 5, 4],  # -Y
            [2, 3, 7],
            [2, 7, 6],  # +Y
            [0, 4, 7],
            [0, 7, 3],  # -X
            [1, 2, 6],
            [1, 6, 5],  # +X
        ],
        dtype=np.int64,
    )
    return verts, faces


def make_cylinder_mesh(
    radius: float, height: float, n_seg: int = 32, n_rings: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Closed Z-aligned cylinder centered at origin with outward normals."""
    zs = np.linspace(-height / 2.0, height / 2.0, n_rings)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    verts = []
    for z in zs:
        for th in thetas:
            verts.append([radius * math.cos(th), radius * math.sin(th), z])
    bottom_center = len(verts)
    verts.append([0.0, 0.0, -height / 2.0])
    top_center = len(verts)
    verts.append([0.0, 0.0, height / 2.0])

    faces = []
    for r in range(n_rings - 1):
        for i in range(n_seg):
            j = (i + 1) % n_seg
            a = r * n_seg + i
            b = r * n_seg + j
            c = (r + 1) * n_seg + j
            d = (r + 1) * n_seg + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    # Bottom cap (-Z)
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([bottom_center, j, i])
    # Top cap (+Z)
    top_ring = (n_rings - 1) * n_seg
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([top_center, top_ring + i, top_ring + j])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def merge_meshes(*meshes: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vs, fs = [], []
    offset = 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + offset)
        offset += len(v)
    return np.concatenate(vs, axis=0), np.concatenate(fs, axis=0)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_usda(path: Path, verts: np.ndarray, faces: np.ndarray, meters_per_unit: float = 1.0) -> None:
    pts_str = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in verts)
    cnt_str = ", ".join("3" for _ in faces)
    idx_str = ", ".join(f"{a}, {b}, {c}" for a, b, c in faces)
    content = f"""#usda 1.0
(
    defaultPrim = "Object"
    metersPerUnit = {meters_per_unit}
)
def Xform "Object"
{{
    def Mesh "Collider"
    {{
        int[] faceVertexCounts = [{cnt_str}]
        int[] faceVertexIndices = [{idx_str}]
        point3f[] points = [{pts_str}]
    }}
}}
"""
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: Stage 1 geometry & antipodal annotation
# ---------------------------------------------------------------------------


def test_box_antipodal_grasps_invariants():
    verts, faces = make_box_mesh(0.04, 0.08, 0.06)
    res = aff.annotate_mesh(verts, faces, asset_id="test_box", n_surface_samples=200, max_grasps=16, seed=42)

    assert res["schema_version"] == aff.SCHEMA_VERSION
    assert res["asset_id"] == "test_box"
    assert res["mesh_hash"].startswith("sha256:")
    assert res["symmetry"] == {"z_axis": "discrete", "fold": 2, "axis_xy": [0.0, 0.0]}
    assert np.allclose(res["bounds"]["extents"], [0.04, 0.08, 0.06], atol=1e-4)

    grasps = res["grasps"]["dex1_1"]
    assert len(grasps) >= 4
    spec = aff.GRIPPER_SPECS["dex1_1"]
    for g in grasps:
        assert spec.min_width <= g["width"] <= spec.max_width
        # Box is 0.04 m wide along X and 0.06 m tall along Z, 0.08 m along Y (> max_width 0.07)
        assert g["width"] <= 0.065
        q = g["quaternion_wxyz"]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-4)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-4)
        assert np.allclose(rot[:, 0], g["approach_dir"], atol=1e-3)
        assert np.allclose(rot[:, 1], g["closing_axis"], atol=1e-3)
        # Verify zero gripper collision against mesh vertices
        assert not aff._gripper_collides(verts, np.asarray(g["position"]), rot, g["width"], spec)


def test_cylinder_symmetry_and_grasps():
    verts, faces = make_cylinder_mesh(radius=0.025, height=0.10, n_seg=36, n_rings=8)
    res = aff.annotate_mesh(verts, faces, asset_id="test_cyl", n_surface_samples=220, max_grasps=16, seed=7)
    assert res["symmetry"] == {"z_axis": "continuous", "fold": 0, "axis_xy": [0.0, 0.0]}
    grasps = res["grasps"]["dex1_1"]
    assert len(grasps) >= 4
    # Diameter is 0.05 m -> all lateral grasps should be close to 0.05 m
    widths = [g["width"] for g in grasps]
    assert min(widths) >= 0.045
    assert max(widths) <= 0.055


def test_mug_with_handle_breaks_symmetry_and_finds_handle_protrusion():
    # Wide body (diameter 0.09 m > max_width 0.07 m!) + narrow handle bar (0.015 m thick) sticking out along +X
    body = make_cylinder_mesh(radius=0.045, height=0.10, n_seg=32, n_rings=8)
    handle_bar = make_box_mesh(0.015, 0.018, 0.06, center=(0.068, 0.0, 0.0))
    verts, faces = merge_meshes(body, handle_bar)
    res = aff.annotate_mesh(verts, faces, asset_id="wide_mug", n_surface_samples=350, max_grasps=16, seed=3)

    assert res["symmetry"]["z_axis"] == "none"
    grasps = res["grasps"]["dex1_1"]
    assert len(grasps) >= 1
    # Because body diameter 0.09 > 0.07 max_width, valid grasps must lie on the handle protrusion!
    assert any(g["position"][0] > 0.05 and g["width"] < 0.03 for g in grasps)
    assert any(g["part_tag"] == "protrusion" for g in grasps)


def test_too_wide_block_returns_zero_grasps():
    # 0.15 m cube exceeds Dex1-1 max_width (0.070 m) along every axis
    verts, faces = make_box_mesh(0.15, 0.15, 0.15)
    res = aff.annotate_mesh(verts, faces, asset_id="giant_cube", n_surface_samples=150, seed=0)
    assert res["grasps"]["dex1_1"] == []


def test_empty_and_degenerate_mesh_graceful():
    empty_v = np.zeros((0, 3), dtype=np.float64)
    empty_f = np.zeros((0, 3), dtype=np.int64)
    res = aff.annotate_mesh(empty_v, empty_f, asset_id="empty")
    assert res["grasps"]["dex1_1"] == []
    assert res["mesh_stats"]["valid_triangles"] == 0


def test_determinism_with_fixed_seed():
    verts, faces = make_box_mesh(0.04, 0.04, 0.08)
    r1 = aff.annotate_mesh(verts, faces, asset_id="det", n_surface_samples=120, seed=99)
    r2 = aff.annotate_mesh(verts, faces, asset_id="det", n_surface_samples=120, seed=99)
    assert r1 == r2


def test_obj_and_usda_loaders_and_cli(tmp_path: Path):
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    obj_path = tmp_path / "box.obj"
    usda_path = tmp_path / "box.usda"
    write_obj(obj_path, verts, faces)
    write_usda(usda_path, verts * 100.0, faces, meters_per_unit=0.01)  # cm -> m scaling test

    v_obj, f_obj = aff.load_mesh(obj_path)
    v_usd, f_usd = aff.load_mesh(usda_path)
    assert v_obj.shape == (8, 3) and f_obj.shape == (12, 3)
    assert np.allclose(v_usd, verts, atol=1e-5)
    assert f_usd.shape == (12, 3)

    out_json = tmp_path / "affordance.json"
    rc = aff.main(["annotate", str(obj_path), "--output", str(out_json), "--samples", "120", "--seed", "1"])
    assert rc == 0
    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(loaded["grasps"]["dex1_1"]) >= 2
    assert aff.main(["inspect", str(out_json)]) == 0


def test_real_repo_obj_mesh_annotates():
    finger_obj = ROOT / "third_party/curobo/curobo/content/assets/robot/franka_description/meshes/collision/finger.obj"
    if not finger_obj.is_file():
        pytest.skip("Sample curobo OBJ not present")
    res = aff.annotate_asset(finger_obj, n_surface_samples=150, max_grasps=12, seed=0)
    assert res["mesh_stats"]["valid_triangles"] > 0
    assert len(res["grasps"]["dex1_1"]) >= 1


# ---------------------------------------------------------------------------
# Tests: Stage 2 physics verification workflow
# ---------------------------------------------------------------------------


def test_stage2_verification_updates_scores_and_filters(tmp_path: Path):
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    data = aff.annotate_mesh(verts, faces, asset_id="verify_box", n_surface_samples=160, max_grasps=8, seed=5)
    aff_path = tmp_path / "affordance.json"
    aff_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Custom mock verifier that passes only even indices
    class MockVerifier:
        def verify_batch(self, asset_path, gripper_type, grasps):
            out = []
            for idx, g in enumerate(grasps):
                ok = idx % 2 == 0
                out.append(
                    verify_sim.GraspVerificationResult(
                        grasp_id=g["id"],
                        passed=ok,
                        sim_score=0.95 if ok else 0.1,
                        lift_height_m=0.14 if ok else 0.01,
                        slip_m=0.003 if ok else 0.06,
                        reason="ok" if ok else "slipped",
                    )
                )
            return out

    updated = verify_sim.verify_affordance_file(aff_path, MockVerifier(), drop_failed=False)
    glist = updated["grasps"]["dex1_1"]
    assert len(glist) == len(data["grasps"]["dex1_1"])
    # Verified grasps must be sorted ahead of unverified ones
    verified_flags = [g["verified"] for g in glist]
    assert verified_flags == sorted(verified_flags, reverse=True)
    assert glist[0]["verified"] is True
    assert glist[0]["sim_score"] == 0.95
    assert glist[0]["sim_metrics"]["reason"] == "ok"

    # Now drop failed
    dropped = verify_sim.verify_affordance_file(aff_path, MockVerifier(), drop_failed=True)
    assert all(g["verified"] for g in dropped["grasps"]["dex1_1"])


def make_hollow_cup_mesh(
    r_out: float, r_in: float, height: float, bottom_thick: float = 0.005, n_seg: int = 32, n_rings: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Watertight hollow open-top cup with outward normals."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    zs_out = np.linspace(0.0, height, n_rings)
    zs_in = np.linspace(height, bottom_thick, n_rings)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    for z in zs_out:
        for th in thetas:
            verts.append([r_out * math.cos(th), r_out * math.sin(th), float(z)])
    for z in zs_in:
        for th in thetas:
            verts.append([r_in * math.cos(th), r_in * math.sin(th), float(z)])
    c_bot_out = len(verts)
    verts.append([0.0, 0.0, 0.0])
    c_bot_in = len(verts)
    verts.append([0.0, 0.0, bottom_thick])

    for r in range(n_rings - 1):
        for i in range(n_seg):
            j = (i + 1) % n_seg
            a, b, c, d = r * n_seg + i, r * n_seg + j, (r + 1) * n_seg + j, (r + 1) * n_seg + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    r_top_out = (n_rings - 1) * n_seg
    r_top_in = n_rings * n_seg
    for i in range(n_seg):
        j = (i + 1) % n_seg
        a, b, c, d = r_top_out + i, r_top_out + j, r_top_in + j, r_top_in + i
        faces.append([a, b, c])
        faces.append([a, c, d])
    for r in range(n_rings, 2 * n_rings - 1):
        for i in range(n_seg):
            j = (i + 1) % n_seg
            a, b, c, d = r * n_seg + i, r * n_seg + j, (r + 1) * n_seg + j, (r + 1) * n_seg + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([c_bot_out, j, i])
    r_bot_in = (2 * n_rings - 1) * n_seg
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([c_bot_in, r_bot_in + i, r_bot_in + j])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_flipped_normal_mesh_recovers_antipodal_grasps():
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    flipped_faces = faces[:, [0, 2, 1]]
    res = aff.annotate_mesh(verts, flipped_faces, asset_id="flipped_box", n_surface_samples=200, max_grasps=16, seed=42)
    grasps = res["grasps"]["dex1_1"]
    assert len(grasps) >= 4
    spec = aff.GRIPPER_SPECS["dex1_1"]
    for g in grasps:
        assert spec.min_width <= g["width"] <= spec.max_width


def test_hollow_thin_and_thick_rim_cups():
    # Narrow thin-walled cup (d=6cm, wall=3mm): grasped across full outer diameter
    v_narrow, f_narrow = make_hollow_cup_mesh(0.030, 0.027, 0.10)
    r_narrow = aff.annotate_mesh(v_narrow, f_narrow, asset_id="thin_cup", n_surface_samples=250, max_grasps=16, seed=0)
    assert len(r_narrow["grasps"]["dex1_1"]) >= 4
    assert all(0.055 <= g["width"] <= 0.065 for g in r_narrow["grasps"]["dex1_1"])

    # Wide thick-rimmed cup (d=10cm > max_width 7cm, wall=10mm): top-down grasps on rim must be tagged "rim"
    v_wide, f_wide = make_hollow_cup_mesh(0.050, 0.040, 0.10)
    r_wide = aff.annotate_mesh(v_wide, f_wide, asset_id="wide_cup", n_surface_samples=350, max_grasps=16, seed=0)
    grasps = r_wide["grasps"]["dex1_1"]
    assert len(grasps) >= 2
    assert any(g["part_tag"] == "rim" for g in grasps)
    assert all(0.008 <= g["width"] <= 0.015 for g in grasps)


def test_binary_and_ascii_stl_loaders(tmp_path: Path):
    import struct

    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    # Write binary STL
    bin_stl = tmp_path / "box_bin.stl"
    buf = bytearray(80) + struct.pack("<I", len(faces))
    for f in faces:
        p0, p1, p2 = verts[f[0]], verts[f[1]], verts[f[2]]
        buf += struct.pack("<12fH", 0.0, 0.0, 0.0, *p0, *p1, *p2, 0)
    bin_stl.write_bytes(bytes(buf))
    vb, fb = aff.load_mesh(bin_stl)
    assert len(fb) == len(faces)
    assert np.allclose(vb.min(axis=0), verts.min(axis=0), atol=1e-5)
    assert np.allclose(vb.max(axis=0), verts.max(axis=0), atol=1e-5)

    # Write ASCII STL
    asc_stl = tmp_path / "box_asc.stl"
    lines = ["solid box"]
    for f in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for vi in f:
            p = verts[vi]
            lines.append(f"      vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid box")
    asc_stl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    va, fa = aff.load_mesh(asc_stl)
    assert len(fa) == len(faces)
    assert np.allclose(va.min(axis=0), verts.min(axis=0), atol=1e-5)


def test_stage2_dry_run_cli(tmp_path: Path):
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    data = aff.annotate_mesh(verts, faces, asset_id="cli_box", n_surface_samples=160, max_grasps=6, seed=1)
    aff_path = tmp_path / "affordance.json"
    out_path = tmp_path / "affordance_verified.json"
    aff_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    rc = verify_sim.main([str(aff_path), "--output", str(out_path), "--dry-run"])
    assert rc == 0
    assert out_path.is_file()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert any(g["verified"] for g in loaded["grasps"]["dex1_1"])


def test_stage2_verification_math_and_floor_clearance():
    spec = aff.GRIPPER_SPECS["dex1_1"]
    rot_down = np.column_stack([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    ok, reason = verify_sim.check_floor_clearance(np.array([0.0, 0.0, 0.05]), rot_down, 0.04, spec, floor_z=0.0)
    assert ok and reason == "ok"

    # Too close to floor -> fingertips penetrate
    ok_low, reason_low = verify_sim.check_floor_clearance(np.array([0.0, 0.0, 0.01]), rot_down, 0.04, spec, floor_z=0.0)
    assert not ok_low and reason_low == "floor_tip_collision"

    # Relative slip: rigidly translated object + gripper -> 0 slip
    slip_m, slip_deg = verify_sim.compute_relative_slip(
        np.array([0.0, 0.0, 0.1]),
        rot_down,
        np.array([0.0, 0.0, 0.08]),
        np.eye(3),
        np.array([0.0, 0.0, 0.25]),
        rot_down,
        np.array([0.0, 0.0, 0.23]),
        np.eye(3),
    )
    assert slip_m < 1e-9 and slip_deg < 1e-6

    # Score verification trial pass & fail branches
    r_ok = verify_sim.score_verification_trial("g0", lift_height_m=0.15, slip_m=0.002, slip_deg=1.0)
    assert r_ok.passed and 0.9 < r_ok.sim_score <= 1.0
    r_drop = verify_sim.score_verification_trial("g1", lift_height_m=0.02, slip_m=0.001, slip_deg=0.0)
    assert not r_drop.passed and r_drop.reason == "insufficient_lift"
    r_slip = verify_sim.score_verification_trial("g2", lift_height_m=0.14, slip_m=0.05, slip_deg=1.0)
    assert not r_slip.passed and r_slip.reason == "excessive_slip_pos"
    r_rot = verify_sim.score_verification_trial("g3", lift_height_m=0.14, slip_m=0.005, slip_deg=35.0)
    assert not r_rot.passed and r_rot.reason == "excessive_slip_rot"


def test_batch_cli_and_manifest_dry_run_verify(tmp_path: Path):
    # Create a tiny fake Assets/Object/Real2Sim/Rigid/box/00000 with ASCII .usda
    obj_dir = tmp_path / "Assets" / "Object" / "Real2Sim" / "Rigid" / "box" / "00000"
    obj_dir.mkdir(parents=True)
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    pts_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in verts)
    cnt_str = ", ".join("3" for _ in faces)
    idx_str = ", ".join(str(int(i)) for f in faces for i in f)
    (obj_dir / "object.usda").write_text(
        f'#usda 1.0\n(\n    metersPerUnit = 1\n)\ndef Xform "root" {{\n  def Mesh "m" {{\n'
        f"    point3f[] points = [{pts_str}]\n"
        f"    int[] faceVertexCounts = [{cnt_str}]\n"
        f"    int[] faceVertexIndices = [{idx_str}]\n  }}\n}}\n",
        encoding="utf-8",
    )
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps({"objects": [{"cat": "box", "idx": 0}]}), encoding="utf-8")
    out_dir = tmp_path / "aff_out"

    rc = aff.main(
        [
            "batch",
            "--pool",
            str(pool_path),
            "--assets-root",
            str(tmp_path / "Assets"),
            "--output-dir",
            str(out_dir),
            "--samples",
            "120",
            "--max-grasps",
            "6",
        ]
    )
    assert rc == 0
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(entries) == 1 and entries[0]["status"] == "ok" and entries[0]["grasps"] >= 1

    rc2 = verify_sim.main(["--manifest", str(manifest_path), "--dry-run"])
    assert rc2 == 0
    aff_data = json.loads((out_dir / "box" / "00000" / "affordance.json").read_text(encoding="utf-8"))
    assert any(g.get("verified") for g in aff_data["grasps"]["dex1_1"])


def test_verify_idempotent_and_isaac_mode_safe_without_app(tmp_path: Path):
    verts, faces = make_box_mesh(0.04, 0.06, 0.08)
    data = aff.annotate_mesh(verts, faces, asset_id="box_idem", n_surface_samples=120, max_grasps=4, seed=0)
    v = verify_sim.DryRunVerifier(score_threshold=0.0)
    once = verify_sim.apply_verification(data, v)
    twice = verify_sim.apply_verification(once, v)
    assert [g["score"] for g in once["grasps"]["dex1_1"]] == [g["score"] for g in twice["grasps"]["dex1_1"]]

    # On CPU without IsaacLab, --isaac must return exit code 1 rather than calling os._exit(1)
    aff_path = tmp_path / "aff.json"
    aff_path.write_text(json.dumps(data), encoding="utf-8")
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        rc = verify_sim.main([str(aff_path), "--asset", str(tmp_path), "--isaac"])
        assert rc == 1
