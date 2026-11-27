"""Unit tests for scripts/objects/gso.py (pure CPU: no usd-core, no trimesh, no Isaac)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import urllib.error
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gso  # noqa: E402


def shoe(length=0.28, heel_up=0.11, toe_up=0.06, yaw_deg=37.0):
    """Points of a shoe-like block: sole 0.1 wide, the heel end (+x before the turn) taller than the toe."""
    x = np.linspace(-length / 2, length / 2, 40)
    pts = [
        [xi, y, z] for xi in x for y in (-0.05, 0.05) for z in (0.0, toe_up + (heel_up - toe_up) * (xi > length / 4))
    ]
    t = np.radians(yaw_deg)
    rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return np.asarray(pts) @ rz.T


def test_shoe_pose():
    for yaw in (37.0, 200.0, -95.0):
        pts = shoe(yaw_deg=yaw)
        rot, s = gso.shoe_pose(pts)
        p = (pts @ rot.T) * s
        lo, hi = p.min(0), p.max(0)
        assert np.isclose(hi[0] - lo[0], gso.SHOE_LENGTH_M[1]) and hi[1] - lo[1] < 0.1  # long axis on x, scaled
        heel = p[p[:, 0] > hi[0] - 0.25 * (hi[0] - lo[0]), 2].max()
        toe = p[p[:, 0] < lo[0] + 0.25 * (hi[0] - lo[0]), 2].max()
        assert heel > toe  # the heel at +x, as every official Rigid/shoe
    assert gso.shoe_pose(shoe(heel_up=0.2)) is None  # a boot: too tall at the official length
    assert gso.shoe_pose(shoe(heel_up=0.04, toe_up=0.03)) is None  # a flip-flop: too flat


def _box_obj(sx: float, sy: float, sz: float, *, offset=(0.0, 0.0, 0.0), use_quads: bool = False) -> str:
    """Create a closed watertight Wavefront OBJ box with outward normals."""
    ox, oy, oz = offset
    verts = [
        (ox, oy, oz),
        (ox + sx, oy, oz),
        (ox + sx, oy + sy, oz),
        (ox, oy + sy, oz),
        (ox, oy, oz + sz),
        (ox + sx, oy, oz + sz),
        (ox + sx, oy + sy, oz + sz),
        (ox, oy + sy, oz + sz),
    ]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["vt 0.0 0.0", "vt 1.0 0.0", "vt 1.0 1.0", "vt 0.0 1.0"]
    quads = [
        (1, 4, 3, 2),  # bottom (-z)
        (5, 6, 7, 8),  # top (+z)
        (1, 2, 6, 5),  # front (-y)
        (3, 4, 8, 7),  # back (+y)
        (1, 5, 8, 4),  # left (-x)
        (2, 3, 7, 6),  # right (+x)
    ]
    for a, b, c, d in quads:
        if use_quads:
            lines.append(f"f {a}/1 {b}/2 {c}/3 {d}/4")
        else:
            lines.append(f"f {a}/1 {b}/2 {c}/3")
            lines.append(f"f {a}/1 {c}/3 {d}/4")
    return "\n".join(lines) + "\n"


def _make_gso_zip(
    zip_path: Path,
    *,
    name: str,
    category: str = "Consumer Goods",
    desc: str = "A test object",
    size: tuple[float, float, float] = (0.10, 0.08, 0.12),
    nested: bool = False,
    with_texture: bool = True,
    obj_text: str | None = None,
) -> Path:
    prefix = f"{name}/" if nested else ""
    pbtxt = (
        f'name: "{name}"\n'
        f'description: "{desc}"\n'
        f'authors {{\n  name: "Google"\n}}\n'
        f'annotations {{\n  key: "brand"\n  value: "TestBrand"\n}}\n'
        f'categories {{\n  first: "{category}"\n}}\n'
    )
    obj = obj_text if obj_text is not None else _box_obj(*size)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{prefix}meshes/model.obj", obj)
        zf.writestr(f"{prefix}metadata.pbtxt", pbtxt)
        if with_texture:
            zf.writestr(f"{prefix}materials/textures/texture.png", b"\x89PNG\r\n\x1a\nFAKEPNG")
    return zip_path


def test_parse_pbtxt_and_categories():
    pbtxt = """
model {
  file: "model.sdf"
}
name: "Ceramic_Coffee_Mug"
description: "Line one\\nLine two"
authors {
  name: "Google"
}
annotations {
  key: "brand"
  value: "Acme"
}
categories {
  first: "Bottles and Cans and Cups"
  second: "Kitchen"
}
"""
    parsed = gso.parse_pbtxt(pbtxt)
    assert parsed["name"] == "Ceramic_Coffee_Mug"
    assert parsed["description"] == ["Line one", "Line two"]
    assert gso.description(pbtxt) == ["Line one", "Line two"]
    assert parsed["categories"] == ["Bottles and Cans and Cups", "Kitchen"]
    assert parsed["annotations"] == {"brand": "Acme"}
    assert gso.gso_category_name("Ceramic_Coffee_Mug", pbtxt) == "gso_bottle_can_cup"
    assert not gso.is_shoe("Ceramic_Coffee_Mug", pbtxt)

    shoe_pb = 'name: "Running_Shoe"\ncategories {\n  first: "Shoe"\n}\n'
    assert gso.is_shoe("Running_Shoe", shoe_pb)
    assert gso.gso_category_name("Running_Shoe", shoe_pb) == "gso_shoe"

    none_pb = 'name: "Mystery"\ncategories {\n  first: "None"\n}\n'
    assert gso.gso_category_name("Mystery", none_pb) == "gso_object"


def test_load_zip_nested_and_dir(tmp_path: Path):
    z1 = _make_gso_zip(tmp_path / "Box_Root.zip", name="Box_Root", size=(0.1, 0.08, 0.06))
    pts, faces, uv, tex, pbtxt = gso.load(z1)
    assert pts.shape[1] == 3 and len(faces) == 12
    assert uv.shape == (len(pts), 2)
    assert tex is not None and tex.startswith(b"\x89PNG")
    assert "Box_Root" in pbtxt

    # Nested folder layout + quad faces fan-triangulated into 12 tris + no texture.
    z2 = _make_gso_zip(
        tmp_path / "Box_Nested.zip",
        name="Box_Nested",
        nested=True,
        with_texture=False,
        obj_text=_box_obj(0.1, 0.08, 0.06, use_quads=True),
    )
    pts2, faces2, uv2, tex2, _ = gso.load(z2)
    assert len(faces2) == 12
    assert uv2.shape == (len(pts2), 2)
    assert tex2 is None

    # Extracted directory directly.
    unpacked = tmp_path / "Box_Dir"
    with zipfile.ZipFile(z1) as zf:
        zf.extractall(unpacked)
    pts3, faces3, _, _, _ = gso.load(unpacked)
    assert len(faces3) == 12 and len(pts3) == len(pts)


def test_general_pose_orientations():
    # Stable upright box (0.12 x 0.08 x 0.10): `auto` keeps +z up and aligns long axis to x.
    xs = np.linspace(0, 0.12, 8)
    ys = np.linspace(0, 0.08, 6)
    zs = np.linspace(0, 0.10, 6)
    upright = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    rot, s, info = gso.general_pose(upright, orient="auto")
    assert s == 1.0
    assert info["up_face"] == "+z"
    assert info["stable_estimate"] is True
    p = upright @ rot.T
    ext = p.max(0) - p.min(0)
    assert np.isclose(ext[2], 0.10, atol=1e-6)
    assert ext[0] >= ext[1] - 1e-6

    # Tall thin unstable needle (0.01 x 0.01 x 0.25): `auto` tips it onto its side so height becomes 0.01.
    needle = np.array(
        [[x, y, z] for x in np.linspace(0, 0.01, 4) for y in np.linspace(0, 0.01, 4) for z in np.linspace(0, 0.25, 20)]
    )
    rot_n, _, info_n = gso.general_pose(needle, orient="auto")
    ext_n = (needle @ rot_n.T).max(0) - (needle @ rot_n.T).min(0)
    assert np.isclose(ext_n[2], 0.01, atol=1e-6)
    assert info_n["up_face"] != "+z"

    # `keep` keeps the needle upright.
    rot_k, _, info_k = gso.general_pose(needle, orient="keep")
    ext_k = (needle @ rot_k.T).max(0) - (needle @ rot_k.T).min(0)
    assert np.isclose(ext_k[2], 0.25, atol=1e-6)
    assert info_k["up_face"] == "+z"

    # `lay-flat` minimizes Z height of upright box -> 0.08.
    rot_f, _, _ = gso.general_pose(upright, orient="lay-flat")
    ext_f = (upright @ rot_f.T).max(0) - (upright @ rot_f.T).min(0)
    assert np.isclose(ext_f[2], 0.08, atol=1e-6)

    with pytest.raises(ValueError):
        gso.general_pose(upright, orient="bogus")


def test_mass_estimation_and_signed_volume(tmp_path: Path):
    z = _make_gso_zip(tmp_path / "mug.zip", name="Ceramic_Mug", size=(0.10, 0.10, 0.10))
    pts, faces, _, _, _ = gso.load(z)
    vol = gso.mesh_signed_volume(pts, faces)
    assert np.isclose(vol, 0.001, rtol=1e-4)

    m_ceramic, info_c = gso.estimate_mass(
        pts, faces, "Ceramic_Mug", 'categories { first: "Bottles and Cans and Cups" }'
    )
    assert info_c["material_class"] == "ceramic"
    assert info_c["volume_source"] == "mesh_signed_volume"
    assert 0.08 <= m_ceramic <= 0.45

    m_cardboard, info_cb = gso.estimate_mass(pts, faces, "Cereal_Box", 'categories { first: "Consumer Goods" }')
    assert info_cb["material_class"] == "cardboard"
    assert 0.02 <= m_cardboard <= 0.45
    assert m_cardboard < m_ceramic

    # Open/single-triangle mesh falls back to bbox_fill_*.
    open_pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.05]])
    open_faces = np.array([[0, 1, 2]])
    m_open, info_o = gso.estimate_mass(open_pts, open_faces, "Plastic_Toy", 'categories { first: "Toys" }')
    assert info_o["volume_source"].startswith("bbox_fill_")
    assert 0.01 <= m_open <= 0.40


def test_simplify_collision_mesh():
    # Build a triangulated grid with > 2000 faces (35x35x2 = 2450 triangles).
    n = 36
    xs, ys = np.meshgrid(np.linspace(0, 0.2, n), np.linspace(0, 0.2, n))
    pts = np.column_stack([xs.ravel(), ys.ravel(), 0.02 * np.sin(xs.ravel() * 20.0)])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces.append([i, i + 1, i + n])
            faces.append([i + 1, i + n + 1, i + n])
    faces_arr = np.asarray(faces, dtype=np.int64)
    assert len(faces_arr) == 2450

    c_pts, c_faces = gso._simplify_collision_mesh(pts, faces_arr, target_faces=500)
    assert 0 < len(c_faces) <= 500
    assert c_faces.min() >= 0 and c_faces.max() < len(c_pts)


def test_survey_dedup_and_convert_general(tmp_path: Path, monkeypatch):
    zips_dir = tmp_path / "zips"
    assets_dir = tmp_path / "Assets"

    # Existing RoboDojo asset with 12 faces (would match a standard 12-face box) -> use custom face count to test dedup.
    existing_meta = assets_dir / "Object/RoboDojo/Rigid/shoe/00900/metadata.json"
    existing_meta.parent.mkdir(parents=True)
    existing_meta.write_text(
        json.dumps(
            {
                "geometry": {"faces": 99999, "aligned_bbox": {"extents": [0.16, 0.07, 0.08]}},
                "source": {"dataset": "Google Scanned Objects", "object": "Already_Imported_Shoe"},
            }
        ),
        encoding="utf-8",
    )
    existing_face_dup = assets_dir / "Object/RoboDojo/Rigid/box/00001/metadata.json"
    existing_face_dup.parent.mkdir(parents=True)
    existing_face_dup.write_text(
        json.dumps({"geometry": {"faces": 14, "aligned_bbox": {"extents": [0.2, 0.2, 0.2]}}}),
        encoding="utf-8",
    )

    # 1. Valid mug
    _make_gso_zip(
        zips_dir / "Blue_Mug.zip", name="Blue_Mug", category="Bottles and Cans and Cups", size=(0.1, 0.1, 0.1)
    )
    # 2. Already imported GSO source
    _make_gso_zip(zips_dir / "Already_Imported_Shoe.zip", name="Already_Imported_Shoe", category="Shoe")
    # 3. New shoe (excluded by default in general mode)
    _make_gso_zip(zips_dir / "New_Shoe.zip", name="New_Shoe", category="Shoe")
    # 4. Too large (> 0.65m)
    _make_gso_zip(zips_dir / "Giant_Chair.zip", name="Giant_Chair", category="Consumer Goods", size=(0.8, 0.5, 0.9))
    # 5. Face-count duplicate of existing RoboDojo box (14 faces)
    extra_faces_obj = _box_obj(0.1, 0.1, 0.1) + "f 1/1 2/2 3/3\nf 2/2 3/3 4/4\n"
    _make_gso_zip(zips_dir / "Dup_Box.zip", name="Dup_Box", category="Consumer Goods", obj_text=extra_faces_obj)
    # 6. Corrupt zip
    (zips_dir / "Broken.zip").write_bytes(b"not-a-zip")

    catalog = {
        "models": [
            {"name": "Blue_Mug", "global_index": 42},
            {"name": "New_Shoe", "global_index": 43},
        ]
    }
    sources = gso.discover_sources(zips_dir=zips_dir)
    manifest = gso.survey(sources, assets_dir, catalog=catalog, exclude_shoes=True)
    assert [e["src"] for e in manifest["chosen"]] == ["Blue_Mug"]
    assert manifest["chosen"][0]["id"] == "Rigid/gso_bottle_can_cup/00042"
    skip_map = {s["src"]: s["reason"] for s in manifest["skipped"]}
    assert "already imported" in skip_map["Already_Imported_Shoe"]
    assert "excluded shoe" in skip_map["New_Shoe"]
    assert "size outside" in skip_map["Giant_Chair"]
    assert "mesh of" in skip_map["Dup_Box"]
    assert "unreadable" in skip_map["Broken"]

    # Mock write_object (no pxr on CPU) and verify general conversion produces bbox-centered metadata + usdz.
    captured_pts: list[np.ndarray] = []

    def fake_write_object(target, points, faces, uv, texture, src, *, mass_kg=None):
        captured_pts.append(points.copy())
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "object.usdz").write_bytes(b"FAKEUSDZ")
        return {"uuid": "fake"}

    monkeypatch.setattr(gso, "write_object", fake_write_object)
    out_root = tmp_path / "out"
    res = gso.convert_general([], assets_dir, out_root, manifest=manifest, zips_dir=zips_dir)
    assert len(res["converted"]) == 1
    assert res["failed"] == []
    assert len(captured_pts) == 1
    lo, hi = captured_pts[0].min(0), captured_pts[0].max(0)
    assert np.allclose((lo + hi) / 2.0, [0.0, 0.0, 0.0], atol=1e-6)  # bbox-centered for check_object.py!

    obj_dir = out_root / "Assets/Object/RoboDojo/Rigid/gso_bottle_can_cup/00042"
    meta = json.loads((obj_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["runtime_ready"] is False
    assert meta["physics"]["size"] == [0.1, 0.1, 0.1]
    assert meta["source"]["dataset"] == "Google Scanned Objects"
    assert meta["source"]["object"] == "Blue_Mug"
    assert meta["adapter"]["mode"] == "general"
    desc = json.loads((obj_dir / "description.json").read_text(encoding="utf-8"))
    assert desc["name"] == "Blue_Mug"

    # Re-run with skip_existing=True -> skipped cleanly.
    res2 = gso.convert_general([], assets_dir, out_root, manifest=manifest, zips_dir=zips_dir, skip_existing=True)
    assert res2["converted"] == []
    assert any(s.get("reason") == "exists" for s in res2["skipped"])


def test_stamp_ready_and_cli(tmp_path: Path):
    out_root = tmp_path / "out"
    check_dir = tmp_path / "check"
    oid_pass = "Rigid/gso_toy/00001"
    oid_fail = "Rigid/gso_toy/00002"
    oid_unchecked = "Rigid/gso_toy/00003"

    for oid in (oid_pass, oid_fail, oid_unchecked):
        d = out_root / "Assets/Object/RoboDojo" / oid
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(
            json.dumps({"runtime_ready": False, "source": {"runtime_validated": False}}), encoding="utf-8"
        )

    p1 = check_dir / oid_pass / "preview/check.json"
    p1.parent.mkdir(parents=True)
    p1.write_text(json.dumps({"settle": "pass", "tilt_deg": 0.4, "z_m": 0.05, "expect_z_m": 0.05}), encoding="utf-8")

    p2 = check_dir / oid_fail / "preview/check.json"
    p2.parent.mkdir(parents=True)
    p2.write_text(json.dumps({"settle": "fail", "tilt_deg": 42.1, "z_m": 0.02, "expect_z_m": 0.05}), encoding="utf-8")

    summary_path = tmp_path / "ready.json"
    rc = gso.main(
        [
            "ready",
            "--root",
            str(out_root),
            "--check-dir",
            str(check_dir),
            "--summary",
            str(summary_path),
            "--note",
            "unit-test",
            oid_pass,
            oid_fail,
            oid_unchecked,
        ]
    )
    assert rc == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["ready"] == [oid_pass]
    assert summary["failed"] == [{"id": oid_fail, "why": ["settle"]}]
    assert summary["not_checked"] == [oid_unchecked]

    m1 = json.loads((out_root / "Assets/Object/RoboDojo" / oid_pass / "metadata.json").read_text(encoding="utf-8"))
    assert m1["runtime_ready"] is True
    assert m1["source"]["runtime_validated"] is True
    assert m1["adapter"]["validation"]["note"] == "unit-test"

    m2 = json.loads((out_root / "Assets/Object/RoboDojo" / oid_fail / "metadata.json").read_text(encoding="utf-8"))
    assert m2["runtime_ready"] is False
    assert m2["adapter"]["validation"]["why"] == ["settle"]


def test_http_retry_and_hf_endpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert gso.hf_endpoint() == "https://hf-mirror.com"

    calls = {"n": 0}
    sleeps: list[float] = []

    class FakeResp:
        def __init__(self, payload: bytes, headers: dict[str, str] | None = None):
            self._buf = io.BytesIO(payload)
            self.headers = headers or {}

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, None)
        if calls["n"] == 2:
            raise urllib.error.URLError("transient")
        return FakeResp(b'[{"type": "file", "path": "models/Cup_A.zip", "size": 123}]')

    monkeypatch.setattr(gso.urllib.request, "urlopen", fake_urlopen)
    body, _ = gso.http_get("https://hf-mirror.com/test", attempts=4, backoff=1.0, sleep_fn=sleeps.append)
    assert b"Cup_A.zip" in body
    assert calls["n"] == 3
    assert sleeps == [3.0, 2.0]

    # Non-retryable 404 fails immediately without sleeping.
    def fake_404(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(gso.urllib.request, "urlopen", fake_404)
    sleeps.clear()
    with pytest.raises(urllib.error.HTTPError):
        gso.http_get("https://hf-mirror.com/missing", attempts=4, sleep_fn=sleeps.append)
    assert sleeps == []


def test_fetch_catalog_sharding_and_skip_existing(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "sources"
    # Pre-create Model_0.zip as valid zip so skip_existing skips it.
    _make_gso_zip(out_dir / "Model_0.zip", name="Model_0")

    downloaded_urls: list[str] = []

    def fake_download(url, dest, **kwargs):
        downloaded_urls.append(url)
        _make_gso_zip(Path(dest), name=Path(dest).stem)
        return 100

    monkeypatch.setattr(gso, "http_download", fake_download)
    catalog = {
        "models": [
            {"name": f"Model_{i}", "backend": "fuel", "url": f"https://example.test/Model_{i}.zip"} for i in range(4)
        ]
    }
    res0 = gso.fetch_catalog(catalog, out_dir, workers=2, shard_index=0, shard_total=2)
    assert [x["name"] for x in res0["skipped"]] == ["Model_0"]
    assert [x["name"] for x in res0["downloaded"]] == ["Model_2"]
    assert res0["failed"] == []
    assert downloaded_urls == ["https://example.test/Model_2.zip"]


def test_legacy_shoe_convert_backward_compat(tmp_path: Path, monkeypatch):
    # Create a shoe zip with heel higher than toe and length 0.25 m.
    pts_shoe = shoe(length=0.25, heel_up=0.09, toe_up=0.05, yaw_deg=25.0)
    # Build minimal OBJ containing those points and a dummy triangle.
    lines = [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in pts_shoe]
    lines.append("f 1 2 3")
    z = _make_gso_zip(
        tmp_path / "ASICS_Test.zip",
        name="ASICS_Test",
        category="Shoe",
        desc="Test running shoe",
        obj_text="\n".join(lines) + "\n",
    )
    assets_dir = tmp_path / "Assets"
    assets_dir.mkdir()
    out_dir = tmp_path / "extra_gso"

    captured: list[np.ndarray] = []

    def fake_write(target, points, faces, uv, texture, src, *, mass_kg=None):
        captured.append(points.copy())
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "object.usdz").write_bytes(b"FAKEUSDZ")
        return {"uuid": "fake-shoe", "geometry": {"faces": len(faces)}}

    monkeypatch.setattr(gso, "write_object", fake_write)
    rc = gso.main(
        ["convert", "--category", "shoe", "--zips", str(z), "--assets", str(assets_dir), "--out", str(out_dir)]
    )
    assert rc == 0
    assert len(captured) == 1
    # Legacy shoe stands on z = 0 (not bbox-centered).
    assert np.isclose(captured[0][:, 2].min(), 0.0, atol=1e-6)
    meta_path = out_dir / "Object/RoboDojo/Rigid/shoe/00900/metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["source"]["tasks"] == ["pack_objects_into_box"]
    assert meta["source"]["license"] == "CC-BY-4.0"


def test_list_fuel_and_hf_models(monkeypatch):
    # Fuel pagination: server caps page size (< per_page=100) across multiple pages, then returns 404 on page 3.
    fuel_pages = {
        "page=1": json.dumps([{"name": "Zebra_Toy", "categories": ["Toys"]}, {"name": "Apple_Mug"}]).encode("utf-8"),
        "page=2": json.dumps([{"name": "Bowl_B", "categories": ["Kitchen"]}]).encode("utf-8"),
    }

    def fake_fuel_get(url, **kwargs):
        for key, payload in fuel_pages.items():
            if f"?{key}&" in url:
                return payload, {}
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(gso, "http_get", fake_fuel_get)
    fuel_models = gso.list_fuel_models()
    assert [m["name"] for m in fuel_models] == ["Apple_Mug", "Bowl_B", "Zebra_Toy"]
    assert fuel_models[0]["url"].endswith("/Apple_Mug/tip/Apple_Mug.zip")

    # HF tree API pagination via Link header + unpacked hf_dir layout.
    pages = {
        0: (
            json.dumps(
                [
                    {"type": "file", "path": "models/Cup_A/meshes/model.obj", "size": 500},
                    {"type": "file", "path": "models/Cup_A/metadata.pbtxt", "size": 50},
                ]
            ).encode("utf-8"),
            {"link": '<https://hf-mirror.com/api/datasets/test/tree/main?cursor=2>; rel="next"'},
        ),
        1: (
            json.dumps(
                [
                    {"type": "file", "path": "models/Cup_A/materials/textures/texture.png", "size": 200},
                ]
            ).encode("utf-8"),
            {},
        ),
    }
    idx = {"i": 0}

    def fake_hf_get(url, **kwargs):
        res = pages[idx["i"]]
        idx["i"] += 1
        return res

    monkeypatch.setattr(gso, "http_get", fake_hf_get)
    hf_models = gso.list_hf_models(repo="test/gso", endpoint="https://hf-mirror.com")
    assert len(hf_models) == 1
    assert hf_models[0]["name"] == "Cup_A"
    assert hf_models[0]["backend"] == "hf_dir"
    assert len(hf_models[0]["files"]) == 3


def test_general_pose_pointed_mesh_and_degenerate():
    # Pyramid standing on a single tip vertex (len(foot) == 1 < 3) must not crash with KeyError.
    pyramid = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [-0.1, 0.0, 0.2],
            [0.0, 0.1, 0.2],
            [0.0, -0.1, 0.2],
        ]
    )
    rot_auto, s_auto, info_auto = gso.general_pose(pyramid, orient="auto")
    assert s_auto == 1.0
    assert info_auto["up_face"] == "-z"  # flips onto its wide 4-vertex base
    assert np.isclose(np.linalg.det(rot_auto), 1.0, atol=1e-6)

    rot_flat, _, _ = gso.general_pose(pyramid, orient="lay-flat")
    assert np.isclose(np.linalg.det(rot_flat), 1.0, atol=1e-6)

    # Degenerate 2-point cloud should not crash or emit RuntimeWarning.
    rot_deg, _, _ = gso.general_pose(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), orient="keep")
    assert np.allclose(rot_deg, np.eye(3))


def test_pbtxt_single_line_and_is_shoe_no_false_positive():
    single = (
        'name: "Kiwi_Shoe_Polish" categories { first: "Consumer Goods" } annotations { key: "brand" value: "Kiwi" }'
    )
    parsed = gso.parse_pbtxt(single)
    assert parsed["name"] == "Kiwi_Shoe_Polish"
    assert parsed["categories"] == ["Consumer Goods"]
    assert parsed["annotations"] == {"brand": "Kiwi"}
    # Mentions "Shoe" in name, but category is Consumer Goods -> NOT a shoe.
    assert not gso.is_shoe("Kiwi_Shoe_Polish", single)

    shoe_single = 'name: "Sneaker" categories { first: "Shoe" }'
    assert gso.is_shoe("Sneaker", shoe_single)


def test_dotted_hf_dir_name_and_dedup_gso_prefix(tmp_path: Path, monkeypatch):
    # Real GSO model names can contain dots (e.g. J.A._Henckels_Set).
    zips_dir = tmp_path / "sources"
    model_dir = zips_dir / "J.A._Henckels_Set"
    (model_dir / "meshes").mkdir(parents=True)
    # Include an inline OBJ comment to verify _parse_obj_pure strips it cleanly.
    obj_with_comments = _box_obj(0.12, 0.08, 0.06).replace("f 1/1 4/2 3/3", "f 1/1 4/2 3/3 # inline comment", 1)
    (model_dir / "meshes" / "model.obj").write_text(obj_with_comments, encoding="utf-8")
    (model_dir / "metadata.pbtxt").write_text(
        'name: "J.A._Henckels_Set"\ncategories { first: "Consumer Goods" }\n', encoding="utf-8"
    )

    # Existing Assets has a different gso_* category with the same aspect ratio (2:1.33:1 scaled 2x).
    assets_dir = tmp_path / "Assets"
    prev_meta = assets_dir / "Object/RoboDojo/Rigid/gso_toy/00001/metadata.json"
    prev_meta.parent.mkdir(parents=True)
    prev_meta.write_text(
        json.dumps({"geometry": {"faces": 999, "aligned_bbox": {"extents": [0.24, 0.16, 0.12]}}}),
        encoding="utf-8",
    )

    catalog = {"models": [{"name": "J.A._Henckels_Set", "global_index": 77}]}
    cat_path = tmp_path / "cat.json"
    cat_path.write_text(json.dumps(catalog), encoding="utf-8")
    man_path = tmp_path / "manifest.json"
    rc = gso.main(
        [
            "survey",
            "--zips-dir",
            str(zips_dir),
            "--catalog",
            str(cat_path),
            "--assets",
            str(assets_dir),
            "--dedup-extents",
            "--orient",
            "lay-flat",
            "--out",
            str(man_path),
        ]
    )
    assert rc == 0
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    assert [e["src"] for e in manifest["chosen"]] == ["J.A._Henckels_Set"]
    assert manifest["chosen"][0]["id"] == "Rigid/gso_consumer_goods/00077"

    # convert_general inherits orient="lay-flat" from manifest when orient is not overridden.
    seen_orient: list[str] = []

    def fake_write(target, points, faces, uv, texture, src, *, mass_kg=None):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "object.usdz").write_bytes(b"USDZ")
        return {"uuid": "u"}

    monkeypatch.setattr(gso, "write_object", fake_write)
    out_root = tmp_path / "out"
    res = gso.convert_general([], assets_dir, out_root, manifest=manifest, zips_dir=zips_dir)
    assert len(res["converted"]) == 1
    meta = json.loads(
        (out_root / "Assets/Object/RoboDojo/Rigid/gso_consumer_goods/00077/metadata.json").read_text(encoding="utf-8")
    )
    seen_orient.append(meta["source"]["orientation"]["orient_mode"])
    assert seen_orient == ["lay-flat"]


def test_fetch_hf_endpoint_override_and_partial_hf_dir_retry(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "sources"
    # Pre-create only model.obj for an hf_dir model (texture.png missing -> must NOT be skipped!).
    partial_dir = out_dir / "Cup_Partial"
    (partial_dir / "meshes").mkdir(parents=True)
    (partial_dir / "meshes" / "model.obj").write_text(_box_obj(0.1, 0.1, 0.1), encoding="utf-8")

    urls: list[str] = []

    def fake_dl(url, dest, **kwargs):
        urls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if str(dest).endswith(".zip"):
            _make_gso_zip(Path(dest), name=Path(dest).stem)
        else:
            Path(dest).write_bytes(b"OK")
        return 10

    monkeypatch.setattr(gso, "http_download", fake_dl)
    catalog = {
        "models": [
            {
                "name": "Cup_Partial",
                "backend": "hf_dir",
                "hf_repo": "org/gso",
                "hf_prefix": "models/Cup_Partial",
                "files": [
                    "models/Cup_Partial/meshes/model.obj",
                    "models/Cup_Partial/materials/textures/texture.png",
                ],
            },
            {
                "name": "Zip_Item",
                "backend": "hf",
                "hf_repo": "org/gso",
                "hf_path": "zips/Zip_Item.zip",
                "url": "https://huggingface.co/datasets/org/gso/resolve/main/zips/Zip_Item.zip",
            },
        ]
    }
    res = gso.fetch_catalog(catalog, out_dir, workers=1, endpoint="https://hf-mirror.com", revision="v2")
    assert res["failed"] == []
    assert sorted(x["name"] for x in res["downloaded"]) == ["Cup_Partial", "Zip_Item"]
    # Partial hf_dir only re-fetched missing texture.png, and hf zip honored the mirror endpoint + revision!
    assert any(
        u == "https://hf-mirror.com/datasets/org/gso/resolve/v2/models/Cup_Partial/materials/textures/texture.png"
        for u in urls
    )
    assert any(u == "https://hf-mirror.com/datasets/org/gso/resolve/v2/zips/Zip_Item.zip" for u in urls)
    assert not any(u.endswith("model.obj") for u in urls)


def test_open_mesh_far_from_origin_uses_bbox_fill():
    # Single open triangle translated far from origin should NOT get a bogus origin-offset signed volume.
    far_pts = np.array([[10.0, 10.0, 10.0], [10.1, 10.0, 10.0], [10.0, 10.1, 10.05]])
    far_faces = np.array([[0, 1, 2]])
    assert gso.mesh_signed_volume(far_pts, far_faces) < 1e-9
    _, info = gso.estimate_mass(far_pts, far_faces, "Open_Lid", 'categories { first: "Consumer Goods" }')
    assert info["volume_source"].startswith("bbox_fill_")


def test_survey_respects_existing_sources_and_max_index(tmp_path: Path):
    z1 = tmp_path / "Alpha_Box.zip"
    z2 = tmp_path / "Beta_Box.zip"
    z3 = tmp_path / "Gamma_Box.zip"
    for zp in (z1, z2, z3):
        _make_gso_zip(zp, name=zp.stem, category="Consumer Goods")

    res = gso.survey(
        [z1, z2, z3],
        existing_sources={"Alpha_Box": "Rigid/gso_consumer_goods/00000"},
        existing_max_index={"Rigid/gso_consumer_goods": 182},
    )
    assert [c["src"] for c in res["chosen"]] == ["Beta_Box", "Gamma_Box"]
    assert [c["id"] for c in res["chosen"]] == [
        "Rigid/gso_consumer_goods/00183",
        "Rigid/gso_consumer_goods/00184",
    ]
    assert any(s["src"] == "Alpha_Box" and "already imported" in s["reason"] for s in res["skipped"])

    # Catalog global_index must not collide with existing_max_index or occupied OIDs
    res_cat = gso.survey(
        [z1, z2, z3],
        catalog={"models": [{"name": "Beta_Box", "global_index": 0}, {"name": "Gamma_Box", "global_index": 200}]},
        existing_sources={"Alpha_Box": "Rigid/gso_consumer_goods/00000"},
        existing_max_index={"gso_consumer_goods": 182},
    )
    assert [c["id"] for c in res_cat["chosen"]] == [
        "Rigid/gso_consumer_goods/00183",
        "Rigid/gso_consumer_goods/00200",
    ]
