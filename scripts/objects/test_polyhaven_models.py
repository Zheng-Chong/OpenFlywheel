"""Tests for scripts/objects/polyhaven_models.py (CPU logic, real EXR->PNG, CLI, and optional pxr conversion)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import object_names  # noqa: E402
import polyhaven_models as ph  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HAS_PXR = importlib.util.find_spec("pxr") is not None


def _write_sample_sources(tmp_path: Path) -> Path:
    src = tmp_path / "SOURCES.json"
    src.write_text(json.dumps({"models": {k: {} for k in sorted(ph.SCENE_FALLBACK_IDS)}}), encoding="utf-8")
    return src


def _has_cv2_exr() -> bool:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:
        import tempfile

        import cv2

        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "probe.exr")
            return bool(cv2.imwrite(p, np.zeros((4, 4, 3), dtype=np.float32)))
    except Exception:
        return False


HAS_CV2_EXR = _has_cv2_exr()


def _sample_assets() -> dict[str, dict]:
    return {
        "antique_ceramic_vase_01": {
            "name": "Antique Ceramic Vase 01",
            "categories": ["decorative", "containers", "vases"],
            "tags": ["ceramic", "vase", "decor"],
            "authors": {"James Ray Cock": "All"},
            "attributes": {"material": ["ceramic"], "condition": ["worn"]},
            "polycount": 9408,
            "dimensions": [220.0, 220.0, 440.0],
            "description": "An antique glazed ceramic vase.",
            "download_count": 12000,
        },
        "brass_pot_01": {
            "name": "Brass Pot 01",
            "categories": ["props", "containers", "dishes"],
            "tags": ["brass", "pot"],
            "authors": {"Rico Cilliers": "All"},
            "attributes": {"material": ["brass"]},
            "polycount": 3760,
            "dimensions": [302.0, 260.0, 180.0],
            "description": "A hammered brass cooking pot.",
            "download_count": 8000,
        },
        "Camera_01": {
            "name": "Camera 01",
            "categories": ["props", "electronics"],
            "tags": ["camera", "vintage"],
            "authors": {"Alex": "All"},
            "attributes": {"material": ["metal", "plastic"]},
            "polycount": 26987,
            "dimensions": [180.0, 140.0, 120.0],
            "description": "A vintage film camera.",
            "download_count": 15000,
        },
        # Already in default scene fallback set -> skipped.
        "ceramic_vase_01": {
            "name": "Ceramic Vase 01",
            "categories": ["vases", "containers"],
            "authors": {"James Ray Cock": "All"},
            "dimensions": [205.0, 206.0, 401.0],
        },
        # Excluded category -> skipped.
        "mossy_rock_01": {
            "name": "Mossy Rock 01",
            "categories": ["nature", "rocks"],
            "dimensions": [300.0, 280.0, 210.0],
        },
        "potted_fern_01": {
            "name": "Potted Fern 01",
            "categories": ["plants", "potted plants"],
            "dimensions": [240.0, 240.0, 350.0],
        },
        # Oversize (> 0.65 m) -> skipped.
        "ArmChair_01": {
            "name": "Arm Chair 01",
            "categories": ["furniture", "seating"],
            "dimensions": [848.0, 765.0, 1065.0],
        },
        # Undersize (< 0.02 m) -> skipped.
        "tiny_screw_01": {
            "name": "Tiny Screw 01",
            "categories": ["tools"],
            "dimensions": [8.0, 8.0, 15.0],
        },
        # Too thin (< 0.003 m) -> skipped.
        "paper_sheet_01": {
            "name": "Paper Sheet 01",
            "categories": ["office"],
            "dimensions": [210.0, 297.0, 1.0],
        },
        # Multi-piece set -> skipped.
        "cutlery_set_01": {
            "name": "Cutlery Set 01",
            "categories": ["dishes"],
            "dimensions": [250.0, 180.0, 30.0],
        },
        "book_pile_02": {
            "name": "Book Stack",
            "categories": ["books"],
            "dimensions": [240.0, 180.0, 140.0],
        },
        # Missing dimensions -> skipped.
        "no_dims_prop": {
            "name": "No Dims Prop",
            "categories": ["props"],
        },
    }


def test_pure_stdlib_import():
    """Importing polyhaven_models must not eagerly pull in numpy, pxr, PIL, or cv2."""
    code = (
        "import sys; sys.path.insert(0, 'scripts/objects'); "
        "import polyhaven_models; "
        "bad = [m for m in ('numpy', 'pxr', 'PIL', 'cv2', 'OpenEXR') if m in sys.modules]; "
        "assert not bad, f'eager imports found: {bad}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_load_skip_ids_scene_sources_and_assets(tmp_path):
    scene_sources = _write_sample_sources(tmp_path)
    assert scene_sources.is_file()
    skip = ph.load_skip_ids(sources_paths=[scene_sources])
    assert ph.SCENE_FALLBACK_IDS.issubset(set(skip))

    # Fallback when SOURCES.json does not exist.
    fallback = ph.load_skip_ids(sources_paths=None)
    assert ph.SCENE_FALLBACK_IDS.issubset(set(fallback))

    # Dedup against already-imported Assets/Object/.../metadata.json.
    meta_dir = tmp_path / "Assets" / "Object" / "RoboDojo" / "Rigid" / "ph_vase" / "00000"
    meta_dir.mkdir(parents=True)
    (meta_dir / "metadata.json").write_text(
        json.dumps({"source": {"upstream_dataset": "Poly Haven", "upstream_id": "antique_ceramic_vase_01"}}),
        encoding="utf-8",
    )
    skip_with_assets = ph.load_skip_ids(sources_paths=[scene_sources], assets_dir=tmp_path / "Assets")
    assert "antique_ceramic_vase_01" in skip_with_assets
    assert "Rigid/ph_vase/00000" in skip_with_assets["antique_ceramic_vase_01"]


def test_filter_models_tabletop_and_exclusions(tmp_path):
    skip_ids = ph.load_skip_ids(sources_paths=[_write_sample_sources(tmp_path)])
    cat = ph.filter_models(_sample_assets(), skip_ids=skip_ids)

    assert cat["total_upstream"] == 12
    assert cat["selected_count"] == 3
    assert cat["skipped_count"] == 9

    selected = {m["model_id"]: m for m in cat["models"]}
    assert set(selected) == {"Camera_01", "antique_ceramic_vase_01", "brass_pot_01"}
    assert selected["antique_ceramic_vase_01"]["id"] == "Rigid/ph_vase/00000"
    assert selected["brass_pot_01"]["id"] == "Rigid/ph_dish/00000"
    assert selected["Camera_01"]["id"] == "Rigid/ph_electronics/00000"
    assert selected["antique_ceramic_vase_01"]["license"] == "CC0-1.0"
    assert selected["antique_ceramic_vase_01"]["authors"] == ["James Ray Cock"]

    skipped = {s["model_id"]: s["reason"] for s in cat["skipped"]}
    assert "already in" in skipped["ceramic_vase_01"]
    assert "excluded category" in skipped["mossy_rock_01"]
    assert "excluded category" in skipped["potted_fern_01"]
    assert "oversize" in skipped["ArmChair_01"]
    assert "undersize" in skipped["tiny_screw_01"]
    assert "too thin" in skipped["paper_sheet_01"]
    assert "multi-piece" in skipped["cutlery_set_01"]
    assert "multi-piece" in skipped["book_pile_02"]
    assert "missing 3D dimensions" in skipped["no_dims_prop"]

    # Limit and include_categories.
    limited = ph.filter_models(_sample_assets(), skip_ids=skip_ids, limit=2)
    assert limited["selected_count"] == 2
    only_elec = ph.filter_models(_sample_assets(), skip_ids=skip_ids, include_categories={"electronics"})
    assert [m["model_id"] for m in only_elec["models"]] == ["Camera_01"]


def test_http_get_retries_429_and_503_but_not_404(monkeypatch):
    calls: list[str] = []

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=120.0):
        assert req.get_header("User-agent") == "curl/8.0"
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, io.BytesIO())
        if len(calls) == 2:
            raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, io.BytesIO())
        return _Resp(b'{"ok": true}')

    monkeypatch.setattr(ph.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(ph.time, "sleep", lambda _: None)

    out = ph.http_get("https://api.polyhaven.com/assets?t=models", attempts=4, backoff=0.01)
    assert out == b'{"ok": true}'
    assert len(calls) == 3

    def _fake_404(req, timeout=120.0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr(ph.urllib.request, "urlopen", _fake_404)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        ph.http_get("https://api.polyhaven.com/files/nonexistent", attempts=4, backoff=0.01)
    assert exc_info.value.code == 404


def test_resolve_usd_package_and_fetch_one(tmp_path):
    usdc_bytes = b"PXR-USDC-DUMMY"
    exr_bytes = b"EXR-DUMMY-BYTES"
    jpg_bytes = b"\xff\xd8\xffJPG-DUMMY"

    files_api = {
        "usd": {
            "2k": {
                "usd": {
                    "url": "https://dl.polyhaven.org/file/ph-assets/Models/usd/2k/demo_pot/demo_pot_2k.usdc",
                    "md5": hashlib.md5(usdc_bytes).hexdigest(),
                    "size": len(usdc_bytes),
                    "include": {
                        "textures/demo_pot_nor_gl_2k.exr": {
                            "url": "https://dl.polyhaven.org/file/ph-assets/Models/exr/2k/demo_pot/demo_pot_nor_gl_2k.exr",
                            "md5": hashlib.md5(exr_bytes).hexdigest(),
                            "size": len(exr_bytes),
                        },
                        "textures/demo_pot_diff_2k.jpg": {
                            "url": "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/demo_pot/demo_pot_diff_2k.jpg",
                            "md5": hashlib.md5(jpg_bytes).hexdigest(),
                            "size": len(jpg_bytes),
                        },
                    },
                }
            }
        }
    }

    # 1k requested -> gracefully falls back to 2k.
    chosen_res, pkg = ph.resolve_usd_package(files_api, res="1k")
    assert chosen_res == "2k"
    assert pkg["url"].endswith("demo_pot_2k.usdc")

    url_map = {
        f"{ph.API}/files/demo_pot": json.dumps(files_api).encode("utf-8"),
        pkg["url"]: usdc_bytes,
        pkg["include"]["textures/demo_pot_nor_gl_2k.exr"]["url"]: exr_bytes,
        pkg["include"]["textures/demo_pot_diff_2k.jpg"]["url"]: jpg_bytes,
    }
    entry = {"id": "Rigid/ph_dish/00000", "model_id": "demo_pot", "name": "Demo Pot", "category": "ph_dish"}

    res1 = ph.fetch_one(entry, tmp_path, res="1k", getter=lambda u: url_map[u])
    assert res1["status"] == "downloaded"
    assert res1["res"] == "2k"
    assert res1["exr_count"] == 1
    assert (tmp_path / "demo_pot" / "demo_pot_2k.usdc").read_bytes() == usdc_bytes
    assert (tmp_path / "demo_pot" / "textures" / "demo_pot_nor_gl_2k.exr").read_bytes() == exr_bytes
    assert (tmp_path / "demo_pot" / "polyhaven_meta.json").is_file()

    # Second run skips existing without touching network.
    def _fail_getter(_u):
        raise AssertionError("network should not be called when skipping existing")

    res2 = ph.fetch_one(entry, tmp_path, res="1k", skip_existing=True, getter=_fail_getter)
    assert res2["status"] == "skipped_existing"


def test_fetch_rejects_md5_mismatch_and_path_traversal(tmp_path):
    bad_md5_api = {
        "usd": {
            "1k": {
                "usd": {
                    "url": "https://dl.polyhaven.org/demo_1k.usdc",
                    "md5": "0" * 32,
                    "include": {},
                }
            }
        }
    }
    entry = {"id": "Rigid/ph_prop/00000", "model_id": "bad_md5", "name": "Bad MD5"}
    with pytest.raises(ValueError, match="md5 mismatch"):
        ph.fetch_one(
            entry,
            tmp_path,
            getter=lambda u: json.dumps(bad_md5_api).encode() if "/files/" in u else b"wrong-bytes",
        )
    assert not list(tmp_path.glob(".tmp_*"))

    traversal_api = {
        "usd": {
            "1k": {
                "usd": {
                    "url": "https://dl.polyhaven.org/demo_1k.usdc",
                    "include": {"../escape.exr": {"url": "https://dl.polyhaven.org/escape.exr"}},
                }
            }
        }
    }
    with pytest.raises(ValueError, match="unsafe relative path"):
        ph.fetch_one(
            {"id": "Rigid/ph_prop/00001", "model_id": "escape_test", "name": "Escape"},
            tmp_path,
            getter=lambda u: json.dumps(traversal_api).encode() if "/files/" in u else b"bytes",
        )


def test_exr_texture_classification_and_srgb():
    assert ph.is_linear_texture(Path("ceramic_vase_01_nor_gl_1k.exr")) is True
    assert ph.is_linear_texture(Path("ceramic_vase_01_metal_1k.exr")) is True
    assert ph.is_linear_texture(Path("ceramic_vase_01_rough_1k.exr")) is True
    assert ph.is_linear_texture(Path("ceramic_vase_01_disp_1k.exr")) is True
    assert ph.is_linear_texture(Path("ceramic_vase_01_diff_1k.exr")) is False
    assert ph.is_linear_texture(Path("ceramic_vase_01_albedo_1k.exr")) is False

    # Linear 0.5 -> sRGB ~0.735; HDR values > 1.0 tone-mapped into [0, 1] without NaN.
    mid = ph.linear_to_srgb(np.array([[[0.5, 0.0, 1.0]]], dtype=np.float32))
    assert abs(float(mid[0, 0, 0]) - 0.7354) < 0.01
    hdr = ph.linear_to_srgb(np.array([[[4.0, np.nan, -0.5]]], dtype=np.float32))
    assert np.isfinite(hdr).all()
    assert 0.0 <= float(hdr.min()) <= float(hdr.max()) <= 1.0


@pytest.mark.skipif(not HAS_CV2_EXR, reason="OpenCV OpenEXR write/read not available")
def test_exr_to_png_conversion_and_usda_rewrite(tmp_path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()

    # 1. Linear normal map: RGB = (0.2, 0.5, 1.0) -> OpenCV expects BGR = (1.0, 0.5, 0.2).
    nor_bgr = np.zeros((8, 8, 3), dtype=np.float32)
    nor_bgr[:, :, 0] = 1.0  # B
    nor_bgr[:, :, 1] = 0.5  # G
    nor_bgr[:, :, 2] = 0.2  # R
    nor_exr = tex_dir / "pot_nor_gl_1k.exr"
    assert cv2.imwrite(str(nor_exr), nor_bgr)

    # 2. Color diffuse map: RGB = (0.5, 0.5, 0.5) linear -> sRGB ~188.
    diff_bgr = np.full((8, 8, 3), 0.5, dtype=np.float32)
    diff_exr = tex_dir / "pot_diff_1k.exr"
    assert cv2.imwrite(str(diff_exr), diff_bgr)

    # 3. Grayscale metallic map.
    metal_exr = tex_dir / "pot_metal_1k.exr"
    assert cv2.imwrite(str(metal_exr), np.full((8, 8), 0.25, dtype=np.float32))

    # 4. Corrupt EXR file (should warn when strict=False, raise when strict=True).
    bad_exr = tex_dir / "broken_rough_1k.exr"
    bad_exr.write_bytes(b"not-an-exr")

    with pytest.raises(RuntimeError):
        ph.convert_directory_exrs(tmp_path, strict=True)

    replacements, warnings = ph.convert_directory_exrs(tmp_path, strict=False)
    assert len(warnings) == 1
    assert "broken_rough_1k.exr" in warnings[0]
    assert replacements == {
        "pot_diff_1k.exr": "pot_diff_1k.png",
        "pot_metal_1k.exr": "pot_metal_1k.png",
        "pot_nor_gl_1k.exr": "pot_nor_gl_1k.png",
    }

    # Verify BGR->RGB swap and linear (no gamma) on normal map.
    nor_png = np.asarray(Image.open(tex_dir / "pot_nor_gl_1k.png"))
    assert nor_png.shape == (8, 8, 3)
    assert abs(int(nor_png[0, 0, 0]) - 51) <= 1  # R = 0.2 * 255
    assert abs(int(nor_png[0, 0, 1]) - 128) <= 1  # G = 0.5 * 255
    assert abs(int(nor_png[0, 0, 2]) - 255) <= 1  # B = 1.0 * 255

    # Verify sRGB curve applied on diffuse map.
    diff_png = np.asarray(Image.open(tex_dir / "pot_diff_1k.png"))
    assert abs(int(diff_png[0, 0, 0]) - 188) <= 2

    # Verify grayscale map.
    metal_png = np.asarray(Image.open(tex_dir / "pot_metal_1k.png"))
    assert metal_png.ndim == 2
    assert abs(int(metal_png[0, 0]) - 64) <= 1

    # Verify USDA asset path rewriting.
    usda = tmp_path / "pot_1k.usda"
    usda.write_text(
        '#usda 1.0\ndef Shader "S" {\n'
        "    asset inputs:file = @./textures/pot_nor_gl_1k.exr@\n"
        "    asset inputs:diff = @textures/pot_diff_1k.exr@\n"
        "    asset inputs:keep = @./textures/pot_ao_1k.jpg@\n"
        "}\n",
        encoding="utf-8",
    )
    changed = ph.rewrite_usd_texture_paths(usda, replacements)
    assert changed == 2
    text = usda.read_text(encoding="utf-8")
    assert "@./textures/pot_nor_gl_1k.png@" in text
    assert "@textures/pot_diff_1k.png@" in text
    assert "@./textures/pot_ao_1k.jpg@" in text


def _unit_cube_mesh(sx: float, sy: float, sz: float) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    pts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
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
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return pts, faces


def test_geometry_mass_pose_and_colliders():
    # Signed volume of 0.1 x 0.2 x 0.3 m box = 0.006 m^3.
    pts, faces = _unit_cube_mesh(0.1, 0.2, 0.3)
    vol = ph.mesh_signed_volume(pts, faces)
    assert abs(vol - 0.006) < 1e-6

    # Material inference prioritizes attributes.material, then keywords.
    assert ph.infer_material_class({"attributes": {"material": ["brass"]}, "name": "Pot"}) == "metal"
    assert ph.infer_material_class({"attributes": {}, "name": "Ceramic Vase 01"}) == "ceramic"
    assert ph.infer_material_class({"attributes": {}, "name": "Mystery Widget"}) == "default"

    # Mass estimation clamps to tabletop cap (0.5 kg).
    mass_kg, info = ph.estimate_mass(pts, faces, {"attributes": {"material": ["metal"]}, "name": "Block"})
    assert mass_kg == 0.5
    assert info["material_class"] == "metal"
    assert info["clamped"] is True

    small_pts, small_faces = _unit_cube_mesh(0.08, 0.08, 0.08)
    wood_kg, wood_info = ph.estimate_mass(small_pts, small_faces, {"attributes": {"material": ["wood"]}})
    assert 0.25 < wood_kg < 0.35
    assert wood_info["volume_source"] == "mesh_signed_volume"

    # Orientation: tall unstable stick (0.02 x 0.02 x 0.25) gets laid flat by auto, kept by keep.
    rod_pts, _ = _unit_cube_mesh(0.02, 0.02, 0.25)
    rot_keep, info_keep = ph.compute_pose(rod_pts, orient="keep")
    assert np.allclose(rot_keep, np.eye(3))
    assert info_keep["up_face"] == "+z"

    rot_auto, info_auto = ph.compute_pose(rod_pts, orient="auto")
    rot_rod = rod_pts @ rot_auto.T
    assert (rot_rod.max(0) - rot_rod.min(0))[2] < 0.05
    assert info_auto["stable_estimate"] is True

    # Collider selection: high-poly uses decimated `proxy` (NOT `convexHull`).
    assert ph.choose_collider_mode(2, 10_000, requested="auto") == "convexDecomposition"
    assert ph.choose_collider_mode(2, 45_000, requested="auto") == "proxy"
    assert ph.choose_collider_mode(20, 10_000, requested="auto") == "proxy"
    assert ph.choose_collider_mode(2, 45_000, requested="convexHull") == "convexHull"
    assert ph.choose_collider_mode(2, 45_000, requested="boundingCube") == "boundingCube"

    # High-poly mesh decimation reduces > 30k tris to <= 800 tris cleanly (even if fast_simplification errors).
    n = 125
    xs = np.linspace(-0.1, 0.1, n)
    ys = np.linspace(-0.1, 0.1, n)
    gx, gy = np.meshgrid(xs, ys)
    gz = 0.02 * np.sin(gx * 20.0) * np.cos(gy * 20.0)
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel()
    d = idx[1:, 1:].ravel()
    grid_faces_arr = np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)], axis=0)
    assert len(grid_faces_arr) > 30_000
    dec_pts, dec_faces = ph.simplify_collision_mesh(grid_pts, grid_faces_arr, target_faces=800)
    assert 0 < len(dec_faces) <= 800
    assert len(dec_pts) < len(grid_pts)

    # Quad triangulation fast path.
    quad_tris = ph.triangles([4, 4], [0, 1, 2, 3, 4, 5, 6, 7])
    assert quad_tris.shape == (4, 3)
    assert quad_tris.tolist() == [[0, 1, 2], [4, 5, 6], [0, 2, 3], [4, 6, 7]]


def test_descriptions_pass_object_names_rules():
    entries = [
        {
            "model_id": "antique_ceramic_vase_01",
            "name": "Antique Ceramic Vase 01",
            "category": "ph_vase",
            "attributes": {"material": ["ceramic"]},
        },
        {
            "model_id": "brass_pot_01",
            "name": "Brass Pot 01",
            "category": "ph_dish",
            "attributes": {"material": ["brass"]},
        },
        {
            "model_id": "brass_pot_02",
            "name": "Brass Pot 02",
            "category": "ph_dish",
            "attributes": {"material": ["brass"]},
        },
        {"model_id": "Camera_01", "name": "Camera 01", "category": "ph_electronics", "attributes": {}},
        {
            "model_id": "CheeseBox_01",
            "name": "CheeseBox_01",
            "category": "ph_container",
            "attributes": {"material": ["wood"]},
        },
        {"model_id": "left_cup_01", "name": "The Left Cup 01!", "category": "ph_dish", "attributes": {}},
        {"model_id": "vase", "name": "Vase", "category": "ph_vase", "attributes": {}},
    ]
    table = {}
    for idx, e in enumerate(entries):
        oid = f"Rigid/{e['category']}/{idx:05d}"
        desc = ph.build_description(e, oid)
        assert desc["uuid"] == ph.object_uuid(oid)
        assert desc["description"]
        table[oid] = desc["names"]
    assert object_names.problems(table) == []


def test_cli_catalog_and_ready(tmp_path):
    assets_json = tmp_path / "assets.json"
    assets_json.write_text(json.dumps(_sample_assets()), encoding="utf-8")
    catalog_out = tmp_path / "catalog.json"
    ids_out = tmp_path / "ids.txt"

    scene_sources = _write_sample_sources(tmp_path)
    rc = ph.main(
        [
            "catalog",
            "--assets-json",
            str(assets_json),
            "--skip-sources",
            str(scene_sources),
            "--out",
            str(catalog_out),
            "--write-ids",
            str(ids_out),
        ]
    )
    assert rc == 0
    cat = json.loads(catalog_out.read_text(encoding="utf-8"))
    assert cat["selected_count"] == 3
    ids = [line.strip() for line in ids_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ids) == 3

    # Stamp ready for 2 objects (1 pass, 1 fail) -> exit code 1.
    out_root = tmp_path / "out"
    check_dir = tmp_path / "check"
    pass_id, fail_id = ids[0], ids[1]
    for oid, settle in ((pass_id, "pass"), (fail_id, "fail")):
        obj_dir = out_root / "Assets" / "Object" / "RoboDojo" / oid
        obj_dir.mkdir(parents=True)
        (obj_dir / "metadata.json").write_text(
            json.dumps({"uuid": ph.object_uuid(oid), "runtime_ready": False, "source": {}, "adapter": {}}),
            encoding="utf-8",
        )
        cfile = check_dir / oid / "preview" / "check.json"
        cfile.parent.mkdir(parents=True)
        cfile.write_text(json.dumps({"settle": settle, "tilt_deg": 0.5, "z_m": 0.1}), encoding="utf-8")

    summary_path = tmp_path / "ready.json"
    rc_ready = ph.main(
        [
            "ready",
            "--out",
            str(out_root),
            "--check-dir",
            str(check_dir),
            "--summary",
            str(summary_path),
            "--note",
            "pytest",
            pass_id,
            fail_id,
        ]
    )
    assert rc_ready == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] == 1
    assert summary["failed"] == 1

    meta_pass = json.loads((out_root / "Assets/Object/RoboDojo" / pass_id / "metadata.json").read_text())
    meta_fail = json.loads((out_root / "Assets/Object/RoboDojo" / fail_id / "metadata.json").read_text())
    assert meta_pass["runtime_ready"] is True
    assert meta_pass["source"]["runtime_validated"] is True
    assert meta_fail["runtime_ready"] is False
    assert meta_fail["source"]["runtime_validated"] is False


def test_sbatch_bash_syntax():
    sbatch = REPO_ROOT / "scripts" / "objects" / "slurm_polyhaven_models.sbatch"
    if not sbatch.is_file():
        pytest.skip("cluster sbatch template not included in OpenFlywheel")
    proc = subprocess.run(["bash", "-n", str(sbatch)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_mock_pxr_convert_one_rotation_proxy_includes_and_reconvert(tmp_path, monkeypatch):
    """Exercise convert_one end-to-end on CPU with a strict mock pxr matching real OpenUSD semantics.

    Crucially, MockMatrix4d.__getitem__ returns a COPY of the row (like Boost.Python Gf.Matrix4d),
    and MockLayer enforces the Sdf.Layer registry so CreateNew fails if a layer is already registered.
    """

    class _Vec3d:
        def __init__(self, x, y, z):
            self.v = np.array([float(x), float(y), float(z)], dtype=float)

    class _Matrix4d:
        def __init__(self, *args):
            if len(args) == 1 and isinstance(args[0], int | float):
                self._m = np.eye(4, dtype=float) * float(args[0])
            elif len(args) == 16:
                self._m = np.array(args, dtype=float).reshape(4, 4)
            elif len(args) == 0:
                self._m = np.eye(4, dtype=float)
            else:
                raise TypeError(f"unsupported Matrix4d args: {len(args)}")

        def __getitem__(self, idx):
            # Return a COPY like real pxr.Gf.Matrix4d[r] -> Gf.Vec4d value!
            return self._m[idx].copy()

        def __array__(self, dtype=None):
            return np.asarray(self._m, dtype=dtype)

        def SetScale(self, s):
            self._m = np.diag([float(s), float(s), float(s), 1.0])
            return self

        def __mul__(self, other):
            out = _Matrix4d()
            out._m = self._m @ other._m
            return out

    class _Gf:
        Matrix4d = _Matrix4d
        Vec3d = _Vec3d
        Vec3f = _Vec3d

    class _Vt:
        class Vec3fArray:
            @staticmethod
            def FromNumpy(arr):
                return np.asarray(arr, dtype=np.float32)

        class IntArray:
            def __init__(self, vals):
                self.vals = list(vals)

            @staticmethod
            def FromNumpy(arr):
                return np.asarray(arr, dtype=np.int32)

    class _Path:
        def __init__(self, s: str):
            self.pathString = s
            self.isEmpty = not bool(s)

    class _Attr:
        def __init__(self, val=None):
            self._val = val

        def Set(self, v):
            self._val = v
            return self

        def Get(self):
            return self._val

    class _Prim:
        def __init__(self, path: str, type_name: str = "Xform"):
            self._path = _Path(path)
            self._type = type_name
            self._apis: set[str] = set()
            self._attrs: dict[str, _Attr] = {}
            self._ops: list[tuple[str, _Attr]] = []
            self._refs: list[tuple[str, object]] = []
            self._active = True
            self._children: list[_Prim] = []
            self._purpose = "default"
            self._vis = "inherited"

        def GetPrim(self):
            return self

        def GetPath(self):
            return self._path

        def GetTypeName(self):
            return self._type

        def GetAppliedSchemas(self):
            return list(self._apis)

        def RemoveAppliedSchema(self, api):
            self._apis.discard(api)

        def HasAPI(self, api_cls):
            return api_cls.__name__ in self._apis

        def IsA(self, cls):
            return self._type == cls.__name__

        def IsInstance(self):
            return False

        def IsActive(self):
            return self._active

        def SetActive(self, v):
            self._active = bool(v)

        def GetChildren(self):
            return list(self._children)

        def GetReferences(self):
            prim = self

            class _Refs:
                def AddReference(self, asset_path, prim_path=None):
                    prim._refs.append((asset_path, prim_path))

            return _Refs()

        def AddTranslateOp(self):
            a = _Attr(_Vec3d(0, 0, 0))
            self._ops.append(("translate", a))
            return a

        def AddTransformOp(self, opSuffix=""):
            a = _Attr(_Matrix4d(1.0))
            self._ops.append((f"transform:{opSuffix}", a))
            return a

        def AddScaleOp(self):
            a = _Attr(_Vec3d(1, 1, 1))
            self._ops.append(("scale", a))
            return a

        def CreatePointsAttr(self, v=None):
            a = self._attrs.setdefault("points", _Attr())
            if v is not None:
                a.Set(v)
            return a

        def GetPointsAttr(self):
            return self._attrs.setdefault("points", _Attr())

        def CreateFaceVertexCountsAttr(self, v=None):
            a = self._attrs.setdefault("counts", _Attr())
            if v is not None:
                a.Set(v)
            return a

        def GetFaceVertexCountsAttr(self):
            return self._attrs.setdefault("counts", _Attr())

        def CreateFaceVertexIndicesAttr(self, v=None):
            a = self._attrs.setdefault("indices", _Attr())
            if v is not None:
                a.Set(v)
            return a

        def GetFaceVertexIndicesAttr(self):
            return self._attrs.setdefault("indices", _Attr())

        def CreateSubdivisionSchemeAttr(self, v=None):
            return self._attrs.setdefault("subdiv", _Attr(v))

        def CreateSizeAttr(self, v=None):
            return self._attrs.setdefault("size", _Attr(v))

        def CreatePurposeAttr(self, v):
            self._purpose = v

        def MakeInvisible(self):
            self._vis = "invisible"

    layer_registry: dict[str, _MockLayer] = {}
    rod_pts, rod_faces = _unit_cube_mesh(0.02, 0.02, 0.25)
    rod_pts = rod_pts + np.array([0.1, -0.05, 0.2])

    class _MockLayer:
        def __init__(self, ident: str):
            self.identifier = ident
            self.text = Path(ident).read_text(encoding="utf-8") if Path(ident).is_file() else ""
            self.reloaded = 0
            self.cleared = 0

        @classmethod
        def Find(cls, ident: str):
            return layer_registry.get(ident)

        @classmethod
        def FindOrOpen(cls, ident: str):
            if ident in layer_registry:
                return layer_registry[ident]
            lyr = cls(ident)
            layer_registry[ident] = lyr
            return lyr

        @classmethod
        def CreateNew(cls, ident: str, args=None):
            if ident in layer_registry:
                raise RuntimeError(f"A layer already exists with identifier {ident}")
            lyr = cls(ident)
            layer_registry[ident] = lyr
            return lyr

        def Reload(self, force=False):
            self.reloaded += 1
            if Path(self.identifier).is_file():
                self.text = Path(self.identifier).read_text(encoding="utf-8")

        def Clear(self):
            self.cleared += 1
            self.text = ""

        def Save(self):
            Path(self.identifier).parent.mkdir(parents=True, exist_ok=True)
            Path(self.identifier).write_text(self.text or "#usda 1.0\n", encoding="utf-8")

    class _Sdf:
        Layer = _MockLayer
        AssetPath = str

    class _Stage:
        def __init__(self, layer: _MockLayer, *, is_source: bool = False):
            self._layer = layer
            self._prims: dict[str, _Prim] = {}
            self._default_prim: _Prim | None = None
            if is_source:
                m = _Prim("/rod/mesh", "Mesh")
                m.CreatePointsAttr(rod_pts)
                m.CreateFaceVertexCountsAttr(np.full(len(rod_faces), 3, dtype=np.int64))
                m.CreateFaceVertexIndicesAttr(rod_faces.ravel())
                root = _Prim("/rod", "Xform")
                root._children.append(m)
                self._prims["/rod"] = root
                self._prims["/rod/mesh"] = m
                self._default_prim = root

        def GetRootLayer(self):
            return self._layer

        def SetDefaultPrim(self, p):
            self._default_prim = p

        def GetDefaultPrim(self):
            return self._default_prim

        def GetPseudoRoot(self):
            root = _Prim("/", "PseudoRoot")
            root._children = [p for k, p in self._prims.items() if k.count("/") == 1]
            return root

        def Traverse(self):
            return list(self._prims.values())

        def define(self, path: str, type_name: str):
            p = _Prim(path, type_name)
            self._prims[path] = p
            if path == "/root/normalize/asset":
                # Simulate referenced mesh under /root/normalize/asset/mesh.
                m = _Prim("/root/normalize/asset/mesh", "Mesh")
                m.CreatePointsAttr(rod_pts)
                m.CreateFaceVertexCountsAttr(np.full(len(rod_faces), 3, dtype=np.int64))
                m.CreateFaceVertexIndicesAttr(rod_faces.ravel())
                p._children.append(m)
                self._prims[m.GetPath().pathString] = m
            return p

    stages_by_path: dict[str, _Stage] = {}

    class _Usd:
        class Stage:
            @staticmethod
            def Open(arg):
                if isinstance(arg, _MockLayer):
                    st = _Stage(arg, is_source=False)
                    stages_by_path[arg.identifier] = st
                    return st
                if str(arg) in stages_by_path:
                    return stages_by_path[str(arg)]
                return _Stage(_MockLayer.FindOrOpen(str(arg)), is_source=True)

        @staticmethod
        def PrimRange(prim):
            out = [prim]
            for c in prim.GetChildren():
                out.extend(_Usd.PrimRange(c))
            return out

    class _Tokens:
        z = "Z"
        guide = "guide"
        invisible = "invisible"
        none = "none"

    class _XformCache:
        def GetLocalToWorldTransform(self, prim):
            # Compose /root/normalize ops in USD row-vector order: M_pose * M_translate.
            stage = next(s for s in stages_by_path.values() if prim in s._prims.values())
            norm = stage._prims.get("/root/normalize")
            m = np.eye(4, dtype=float)
            if norm:
                t_mat = np.eye(4, dtype=float)
                p_mat = np.eye(4, dtype=float)
                for kind, attr in norm._ops:
                    if kind == "translate":
                        t_mat[3, :3] = attr.Get().v
                    elif kind == "transform:pose":
                        p_mat = np.asarray(attr.Get(), dtype=float)
                m = p_mat @ t_mat
            return m

    class _UsdGeom:
        Tokens = _Tokens
        XformCache = _XformCache

        class Mesh:
            def __new__(cls, prim):
                return prim

            @staticmethod
            def Define(stage, path):
                return stage.define(path, "Mesh")

        class Xform:
            @staticmethod
            def Define(stage, path):
                return stage.define(path, "Xform")

        class Scope:
            @staticmethod
            def Define(stage, path):
                return stage.define(path, "Scope")

        class Cube:
            @staticmethod
            def Define(stage, path):
                return stage.define(path, "Cube")

        class Camera:
            pass

        class Imageable:
            def __init__(self, prim):
                self._p = prim

            def ComputePurpose(self):
                return self._p._purpose

            def ComputeVisibility(self):
                return self._p._vis

            def CreatePurposeAttr(self, v):
                self._p._purpose = v

        @staticmethod
        def GetStageUpAxis(stage):
            return "Z"

        @staticmethod
        def GetStageMetersPerUnit(stage):
            return 1.0

        @staticmethod
        def SetStageUpAxis(stage, axis):
            pass

        @staticmethod
        def SetStageMetersPerUnit(stage, u):
            pass

    class _UsdPhysics:
        class Joint:
            pass

        class RigidBodyAPI:
            @classmethod
            def Apply(cls, prim):
                prim._apis.add(cls.__name__)
                return prim

        class MassAPI:
            @classmethod
            def Apply(cls, prim):
                prim._apis.add(cls.__name__)

                class _M:
                    def CreateMassAttr(self, v):
                        prim._attrs["mass"] = _Attr(v)

                return _M()

        class CollisionAPI:
            @classmethod
            def Apply(cls, prim):
                prim._apis.add(cls.__name__)
                return prim

        class MeshCollisionAPI:
            @classmethod
            def Apply(cls, prim):
                prim._apis.add(cls.__name__)

                class _MC:
                    def CreateApproximationAttr(self, v):
                        prim._attrs["approx"] = _Attr(v)

                return _MC()

    class _UsdShade:
        class MaterialBindingAPI:
            def __init__(self, prim):
                self._p = prim

            def GetDirectBinding(self, materialPurpose=""):
                class _B:
                    def GetMaterialPath(self):
                        return _Path("")

                return _B()

    class _UsdUtils:
        @staticmethod
        def ModifyAssetPaths(layer, fn):
            import re

            layer.text = re.sub(r"@([^@]+)@", lambda m: f"@{fn(m.group(1))}@", layer.text)

        @staticmethod
        def ComputeAllDependencies(asset_path):
            return [], [], []

    monkeypatch.setattr(ph, "pxr_modules", lambda: (_Gf, _Sdf, _Usd, _UsdGeom, _UsdPhysics, _UsdShade, _UsdUtils, _Vt))

    # Build source model with a non-textures/ extra include file.
    src_dir = tmp_path / "sources" / "tall_rod"
    (src_dir / "extra").mkdir(parents=True)
    (src_dir / "extra" / "custom.mtlx").write_text("<materialx/>", encoding="utf-8")
    (src_dir / "tall_rod_1k.usdc").write_text("#usda 1.0\n", encoding="utf-8")
    (src_dir / "polyhaven_meta.json").write_text(
        json.dumps({"usd_file": "tall_rod_1k.usdc", "res": "1k", "includes": ["extra/custom.mtlx"]}),
        encoding="utf-8",
    )
    entry = {
        "id": "Rigid/ph_tool/00000",
        "model_id": "tall_rod",
        "name": "Tall Rod 01",
        "category": "ph_tool",
        "attributes": {"material": ["metal"]},
    }
    out_root = tmp_path / "out"

    # 1. orient="auto" MUST rotate the unstable 0.02x0.02x0.25 rod flat (size_m[2] < 0.05)!
    res1 = ph.convert_one(entry, tmp_path / "sources", out_root, collider="proxy", orient="auto", keep_exr=True)
    assert res1["status"] == "converted"
    assert res1["size_m"][2] < 0.05, f"orient_op did not rotate rod: size={res1['size_m']}"
    assert res1["collider"] == "proxy"
    obj_dir = out_root / "Assets" / "Object" / "RoboDojo" / "Rigid" / "ph_tool" / "00000"
    assert (obj_dir / "extra" / "custom.mtlx").is_file()

    # 2. Re-converting the SAME target in the same process must reuse/clear cached Sdf.Layer cleanly.
    res2 = ph.convert_one(entry, tmp_path / "sources", out_root, collider="boundingCube", orient="keep", keep_exr=True)
    assert res2["status"] == "converted"
    assert res2["collider"] == "boundingCube"
    assert res2["size_m"][2] > 0.20
    assert layer_registry[str(obj_dir / "object.usd")].cleared == 1


@pytest.mark.skipif(not HAS_PXR, reason="pxr (usd-core / Isaac Sim) not available in CPU test env")
def test_pxr_convert_one_low_poly_and_proxy_and_keep_exr(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    src_dir = tmp_path / "sources" / "demo_bowl"
    tex_dir = src_dir / "textures"
    tex_dir.mkdir(parents=True)

    # Create a real .exr if OpenCV EXR is available, else a minimal PNG + fake EXR with keep_exr.
    exr_path = tex_dir / "demo_bowl_nor_gl_1k.exr"
    if HAS_CV2_EXR:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        import cv2

        cv2.imwrite(str(exr_path), np.full((8, 8, 3), 0.5, dtype=np.float32))
    else:
        exr_path.write_bytes(b"dummy")

    usd_path = src_dir / "demo_bowl_1k.usdc"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/demo_bowl")
    stage.SetDefaultPrim(root.GetPrim())
    pts, faces = _unit_cube_mesh(0.04, 0.04, 0.24)
    pts_shifted = pts + np.array([0.05, -0.02, 0.12])
    mesh = UsdGeom.Mesh.Define(stage, "/demo_bowl/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts_shifted.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
    mat = UsdShade.Material.Define(stage, "/demo_bowl/mat")
    tex = UsdShade.Shader.Define(stage, "/demo_bowl/mat/nor")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./textures/demo_bowl_nor_gl_1k.exr")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()

    entry = {
        "id": "Rigid/ph_dish/00000",
        "model_id": "demo_bowl",
        "name": "Demo Bowl",
        "category": "ph_dish",
        "authors": ["Tester"],
        "attributes": {"material": ["ceramic"]},
    }

    # 1. Default convert (EXR->PNG when cv2 EXR available, convexDecomposition, auto lays tall prism flat).
    out1 = tmp_path / "out1"
    res1 = ph.convert_one(entry, tmp_path / "sources", out1, keep_exr=not HAS_CV2_EXR)
    assert res1["status"] == "converted"
    assert res1["collider"] == "convexDecomposition"
    assert res1["size_m"][2] < 0.06
    obj_dir1 = out1 / "Assets" / "Object" / "RoboDojo" / "Rigid" / "ph_dish" / "00000"
    assert (obj_dir1 / "object.usd").is_file()
    assert (obj_dir1 / "metadata.json").is_file()
    assert (obj_dir1 / "description.json").is_file()
    if HAS_CV2_EXR:
        assert res1["exr_converted"] == 1
        assert (obj_dir1 / "textures" / "demo_bowl_nor_gl_1k.png").is_file()
        assert not (obj_dir1 / "textures" / "demo_bowl_nor_gl_1k.exr").exists()

    # 2. Force proxy collider + keep_exr=True + orient="keep".
    out2 = tmp_path / "out2"
    res2 = ph.convert_one(entry, tmp_path / "sources", out2, collider="proxy", orient="keep", keep_exr=True)
    assert res2["collider"] == "proxy"
    assert res2["size_m"][2] > 0.20
    assert res2["exr_converted"] == 0
    obj_dir2 = out2 / "Assets" / "Object" / "RoboDojo" / "Rigid" / "ph_dish" / "00000"
    assert (obj_dir2 / "textures" / "demo_bowl_nor_gl_1k.exr").is_file()
    reopened = Usd.Stage.Open(str(obj_dir2 / "object.usd"))
    proxy_prim = reopened.GetPrimAtPath("/root/collision/proxy")
    assert proxy_prim.IsValid()
    assert proxy_prim.HasAPI(UsdPhysics.CollisionAPI)


def test_build_texture_remap_fixes_farm_paths_and_extension_mismatch(tmp_path):
    tex = tmp_path / "textures"
    tex.mkdir()
    (tex / "CheeseBox_01_roughness_1k.jpg").write_bytes(b"jpg")
    (tex / "adjustable_wrench_diff_1k.jpg").write_bytes(b"jpg")
    (tex / "adjustable_wrench_metal_1k.png").write_bytes(b"png")

    by_name, by_stem = ph.build_texture_remap(tmp_path)

    # Extension mismatch (.exr referenced in USD, .jpg shipped in 1k package)
    assert (
        ph.rewrite_asset_path_str("./textures/CheeseBox_01_roughness_1k.exr", by_name, by_stem)
        == "./textures/CheeseBox_01_roughness_1k.jpg"
    )
    # Absolute Poly Haven farm path (/mnt/prod/...) with exact & stem match
    assert (
        ph.rewrite_asset_path_str(
            "/mnt/prod/Assets/Models/adjustable_wrench/staging/textures/adjustable_wrench_diff_1k.jpg",
            by_name,
            by_stem,
        )
        == "./textures/adjustable_wrench_diff_1k.jpg"
    )
    assert (
        ph.rewrite_asset_path_str(
            "/mnt/prod/Assets/Models/adjustable_wrench/staging/textures/adjustable_wrench_metal_1k.exr",
            by_name,
            by_stem,
        )
        == "./textures/adjustable_wrench_metal_1k.png"
    )
    # Unknown built-in MDL path left untouched
    assert ph.rewrite_asset_path_str("OmniPBR.mdl", by_name, by_stem) == "OmniPBR.mdl"
