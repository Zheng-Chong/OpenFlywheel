"""SimReady adapter & upstream indexer tests: material, mass, mesh volume, bounds, orientation, MDL whitelist,
package mirroring, survey, readiness (floor-only & cup+beans), and upstream diff/manifest (no Isaac)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("simready", Path(__file__).with_name("simready.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

up_spec = importlib.util.spec_from_file_location("simready_upstream", Path(__file__).with_name("simready_upstream.py"))
U = importlib.util.module_from_spec(up_spec)
up_spec.loader.exec_module(U)


def box(lo, hi):
    """Closed, outward-wound quad mesh of an axis-aligned box: points, faceVertexCounts, faceVertexIndices."""
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return pts, [4] * 6, [i for q in quads for i in q]


def rotate(q, v):
    w, x, y, z = q
    u = np.array([x, y, z], float)
    v = np.asarray(v, float)
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


def test_material_from_physics_material_name_then_package_name():
    assert S.material_kind(["physics_mat_m_opaque__ceramic__mug_a01"], "simready_kitchenware_mug_a01") == "ceramic"
    assert S.material_kind(["physics_mat_m_trans__glass__clean_a01"], "x") == "glass"
    assert S.material_kind(["physics_mat_m_opaque__paper__cup_c01"], "simready_kitchenware_cup_c01") == "paper"
    assert S.material_kind([], "simready_dishware_glass_wineFlute_a01") == "glass"
    # Generalized materials and aliases
    assert S.material_kind(["physics_mat_m_opaque__metal__x"], "simready_kitchenware_pot_a01") == "metal"
    assert S.material_kind(["physics_mat_m_opaque__stainless_steel__x"], "x") == "metal"
    assert S.material_kind(["physics_mat_m_opaque__oak__board"], "x") == "wood"
    assert S.material_kind([], "simready_food_produce_apple_a01") == "food"
    assert S.material_kind([], "simready_handTool_wrench_a01") == "metal"
    assert S.material_kind([], "simready_cleaning_sponge_b01") == "foam"
    assert S.material_kind([], "simready_cleaning_barSoap_a01") == "soap"
    assert S.material_kind([], "simready_misc_notebook_a01") == "cardboard"
    # Unknown material + unknown package raises when default=None, falls back when default is set
    with pytest.raises(ValueError):
        S.material_kind(["physics_mat_m_opaque__unobtainium__x"], "simready_mystery_widget_a01")
    assert S.material_kind([], "simready_mystery_widget_a01", default="default") == "default"
    # Regression: short substrings ("pot", "nut", "pan", "cup") must not misclassify food packages
    for food_pkg in (
        "simready_food_produce_potato_a01",
        "simready_food_produce_sweetPotato_a01",
        "simready_food_prepared_donut_a01",
        "simready_food_fish_spottedSardine_a01",
        "simready_food_prepared_bread_panDiCasa_a01",
        "simready_food_prepared_cupcake_a01",
    ):
        assert S.material_kind([], food_pkg) == "food", food_pkg
    # Deterministic ordering when given a set of multiple physics materials
    mats = {"physics_mat_m_opaque__plastic__lid", "physics_mat_m_opaque__glass__body"}
    assert S.material_kind(mats, "x") == "glass"


def test_all_1001_cataloged_simready_packages_resolve_material_without_default():
    names_path = Path(__file__).with_name("object_names.json")
    names = json.loads(names_path.read_text())
    sr_pkgs = sorted({k.split("/")[1] for k in names if "simready_" in k})
    assert len(sr_pkgs) >= 1000
    resolved = 0
    for pkg in sr_pkgs:
        kind = S.material_kind([], pkg, default="default")
        if kind != "default":
            assert kind in S.MASS_RANGE and kind in S.DEFAULT_DENSITY
            resolved += 1
    assert resolved >= 1000


def test_volume_of_a_closed_mesh_and_open_meshes_are_refused():
    pts, counts, idx = box((0, 0, 0), (0.1, 0.2, 0.05))
    tris = S.triangles(counts, idx)
    assert tris.shape == (12, 3)
    assert S.solid_volume(pts, tris) == pytest.approx(0.1 * 0.2 * 0.05)
    # a hollow cup-like shell: outer box minus an inner box wound inwards
    inner, _, _ = box((0.01, 0.01, 0.01), (0.09, 0.19, 0.06))
    shell = np.concatenate([pts, inner])
    inner_tris = S.triangles(counts, idx)[:, ::-1] + 8
    assert S.solid_volume(shell, np.concatenate([tris, inner_tris])) == pytest.approx(0.001 - 0.08 * 0.18 * 0.05)
    with pytest.raises(ValueError, match="not closed"):
        S.solid_volume(pts, tris[:-1])
    with pytest.raises(ValueError, match="no triangles"):
        S.solid_volume(pts, np.zeros((0, 3), int))


def test_mass_is_volume_times_density_clamped_to_the_material_range():
    mass, why = S.choose_mass(120e-6, 2600, "ceramic")  # a 120 cm^3 stoneware mug
    assert mass == pytest.approx(0.312) and not why["clamped"] and not why["measured"]
    assert S.choose_mass(272e-6, 2600, "ceramic")[0] == 0.45  # chunky mug: capped below the loader's 0.5 kg
    mass, why = S.choose_mass(1e-6, 689, "paper")
    assert mass == S.MASS_RANGE["paper"][0] and why["clamped"] and why["raw_kg"] == pytest.approx(0.0007, abs=1e-4)
    for volume, density in [(0, 2600), (1e-4, float("nan"))]:
        with pytest.raises(ValueError):
            S.choose_mass(volume, density, "glass")


def test_estimate_mass_handles_open_multi_material_and_missing_density():
    pts, counts, idx = box((0, 0, 0), (0.05, 0.05, 0.04))  # 100 cm^3 = 1e-4 m^3
    tris = S.triangles(counts, idx)
    size = np.array([0.05, 0.05, 0.04])

    # 1. Closed mesh + single authored density -> exact choose_mass path
    m1, info1 = S.estimate_mass(pts, tris, size, [2400.0], "ceramic")
    assert m1 == pytest.approx(0.24) and info1["density_kg_m3"] == 2400.0

    # 2. Closed mesh + multiple authored densities -> mean density
    m2, info2 = S.estimate_mass(pts, tris, size, [1000.0, 3000.0], "ceramic")
    assert m2 == pytest.approx(0.2) and info2["density_source"] == "mean_package_materials"
    assert info2["volume_source"] == "closed_mesh"

    # 3. Open mesh + authored density -> bbox * BOX_FILL * density
    m3, info3 = S.estimate_mass(pts, tris[:-1], size, [1000.0], "food", box_fill=0.5)
    assert m3 == pytest.approx(0.05) and info3["volume_source"] == "bbox_fill_0.5"
    assert info3["density_source"] == "package_material"

    # 4. No authored density -> category default density (both closed and open)
    m4, info4 = S.estimate_mass(pts, tris, size, [], "wood")
    assert m4 == pytest.approx(1e-4 * S.DEFAULT_DENSITY["wood"])
    assert info4["density_source"] == "category_default" and info4["volume_source"] == "closed_mesh"

    m5, info5 = S.estimate_mass(pts, tris[:-1], size, [0.0], "metal", box_fill=0.5)
    assert m5 == pytest.approx(min(1e-4 * 0.5 * S.DEFAULT_DENSITY["metal"], S.MASS_RANGE["metal"][1]))
    assert info5["density_source"] == "category_default" and info5["volume_source"] == "bbox_fill_0.5"

    # 5. Invalid bounds on open mesh raises ValueError
    with pytest.raises(ValueError, match="invalid bounds"):
        S.estimate_mass(pts, tris[:-1], [0.0, 0.05, 0.05], [], "food")


def test_check_bounds_relaxed_defaults_and_tabletop_flag():
    assert S.check_bounds([0.1, 0.1, 0.12]) is True  # standard tabletop mug
    assert S.check_bounds([0.015, 0.015, 0.015]) is False  # small marble/nut: allowed, not tabletop
    assert S.check_bounds([0.25, 0.03, 0.001]) is False  # thin knife blade/ruler: allowed, not strict tabletop
    assert S.check_bounds([0.75, 0.12, 0.08]) is False  # long pry bar / pan: allowed, not tabletop
    for bad in ([0.0, 0.1, 0.1], [-0.1, 0.1, 0.1], [float("nan"), 0.1, 0.1], [0.005, 0.005, 0.005], [2.5, 0.5, 0.5]):
        with pytest.raises(ValueError):
            S.check_bounds(bad)


def test_stable_orientation_preserves_vessels_and_lays_flat_standing_slabs():
    # Cups / bottles / kettles stay upright in auto mode even when taller than wide
    q, why = S.stable_orientation([0.08, 0.08, 0.22], "simready_food_beverage_bottle_a01", mode="auto")
    assert q == [1.0, 0.0, 0.0, 0.0] and why == "upright_category"
    q, why = S.stable_orientation([0.09, 0.09, 0.12], "simready_kitchenware_mug_a01", mode="auto")
    assert q == [1.0, 0.0, 0.0, 0.0] and why == "upright_category"

    # Cutting board standing on edge lays flat
    ext = [0.015, 0.25, 0.35]
    q, why = S.stable_orientation(ext, "simready_kitchenware_cuttingBoard_a01", mode="auto")
    assert why == "lay_flat_auto" and math.isclose(sum(c * c for c in q), 1.0)
    assert np.allclose(rotate(q, [1, 0, 0]), [0, 0, 1])

    # Unknown thin slab standing on edge lays flat by aspect ratio
    q, why = S.stable_orientation([0.01, 0.20, 0.30], "simready_misc_slab_a01", mode="auto")
    assert why == "lay_flat_auto"

    # Regression: eyeglasses must lay flat (not match "glass" in UPRIGHT_WORDS), and potato/open must not falsely match
    q, why = S.stable_orientation([0.04, 0.14, 0.15], "simready_accessory_eyeglasses_a01", mode="auto")
    assert why == "lay_flat_auto"
    q, why = S.stable_orientation([0.06, 0.07, 0.10], "simready_food_produce_potato_a01", mode="auto")
    assert why == "default_upright"
    q, why = S.stable_orientation([0.08, 0.08, 0.12], "simready_container_can_tin_open_a01", mode="auto")
    assert why == "upright_category"

    # Already flat object stays identity
    q, why = S.stable_orientation([0.30, 0.20, 0.02], "simready_kitchenware_cuttingBoard_b01", mode="auto")
    assert q == [1.0, 0.0, 0.0, 0.0] and why == "already_flat"

    # Explicit modes
    assert S.stable_orientation(ext, "simready_kitchenware_cuttingBoard_a01", mode="keep") == (
        [1.0, 0.0, 0.0, 0.0],
        "keep",
    )
    q_flat, why_flat = S.stable_orientation([0.08, 0.08, 0.22], "simready_kitchenware_mug_a01", mode="lay-flat")
    assert why_flat == "lay_flat" and np.allclose(rotate(q_flat, [1, 0, 0]), [0, 0, 1])
    with pytest.raises(ValueError, match="unknown orient mode"):
        S.stable_orientation(ext, "x", mode="upside-down")


def test_unresolved_dependencies_whitelist_includes_gltf_pbr_and_builtins():
    unresolved = [
        "OmniPBR.mdl",
        "gltf/pbr.mdl",
        "gltf\\pbr.mdl",
        "OmniSurfaceLite.mdl",
        "AperturePBR_Opacity.mdl",
        "Textures/missing_albedo.png",
        "./payloads/missing.usd",
    ]
    assert S.unresolved_dependencies(unresolved) == ["Textures/missing_albedo.png", "./payloads/missing.usd"]
    assert S.unresolved_dependencies([]) == []


def test_placement_metadata_from_a_centred_point_set():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 400)
    body = np.c_[0.04 * np.cos(a), 0.04 * np.sin(a), rng.uniform(-0.05, 0.05, 400)]
    foot = np.c_[0.03 * np.cos(a[:40]), 0.03 * np.sin(a[:40]), np.full(40, -0.05)]
    handle = np.array([[0.0, -0.07, 0.0], [0.0, -0.07, 0.02]])
    pts = np.concatenate([body[body[:, 2] > -0.045], foot, handle])
    centre = (pts.min(0) + pts.max(0)) / 2
    pts -= centre
    geometry, active = S.geometry_metadata(pts, 99)
    ext = pts.max(0) - pts.min(0)
    assert geometry["faces"] == 99 and geometry["vertices"] == len(pts)
    assert geometry["aligned_bbox"]["extents"] == pytest.approx(ext, abs=1e-4)
    assert len(geometry["oriented_bbox"]["vertices"]) == 8 and geometry["center"] == [0, 0, 0]
    assert geometry["radius"] == pytest.approx(np.linalg.norm(pts, axis=1).max(), abs=1e-4)
    up = active["place"]["up"]
    bottom = pts[:, 2].min()
    assert up["orientation"] == [1, 0, 0, 0]
    assert up["projection_circle"]["center"] == pytest.approx([0, 0, bottom, 1, 0, 0, 0], abs=1e-4)
    assert up["projection_circle"]["radius"] == pytest.approx(np.linalg.norm(pts[:, :2], axis=1).max(), abs=1e-4)
    # the footprint is the foot ring, not the handle, offset from the bbox centre like the mug body
    contact = up["contact_circle"]
    assert contact["center"][:2] == pytest.approx(-centre[:2], abs=2e-3)
    assert contact["radius"] == pytest.approx(0.03, abs=2e-3)
    assert not any(str(v) == "-0.0" for c in ("projection_circle", "contact_circle") for v in up[c]["center"])


def test_package_dir_is_mirrored_except_the_written_files(tmp_path):
    source = tmp_path / "sources/pkg"
    (source / "payloads").mkdir(parents=True)
    (source / "sm_pkg_01.usd").write_text("#usda 1.0\n")
    lib = tmp_path / "lib/RoboDojo/Rigid/simready_pkg/00000"
    lib.mkdir(parents=True)
    for name in ("payloads", "sm_pkg_01.usd"):
        (lib / name).symlink_to(source / name)
    (lib / "description.json").write_text('{"names": {}}')
    (lib / "white_thumbnail.png").write_bytes(b"png")
    (lib / "metadata.json").write_text("{}")
    out = tmp_path / "ovl/RoboDojo/Rigid/simready_pkg/00000"
    S.link_package(lib, out)
    S.link_package(lib, out)  # rerunning replaces, never nests links
    names = ["description.json", "payloads", "sm_pkg_01.usd", "white_thumbnail.png"]
    assert sorted(p.name for p in out.iterdir()) == names
    assert (out / "payloads").readlink() == source / "payloads"
    assert not (out / "white_thumbnail.png").is_symlink() and (out / "white_thumbnail.png").read_bytes() == b"png"
    assert (out / "description.json").read_text() == '{"names": {}}'
    (lib / "rel").symlink_to("payloads")
    S.link_package(lib, out)
    assert (out / "rel").is_symlink() and (out / "rel").readlink() == lib / "payloads"


def test_ready_needs_both_checks(tmp_path):
    ids = ["RoboDojo/Rigid/simready_a/00000", "RoboDojo/Rigid/simready_b/00000"]
    for i in ids:
        (tmp_path / "ovl" / i).mkdir(parents=True)
        meta = {"physics": {}, "source": {"runtime_validated": False}, "adapter": {"validation": None}}
        (tmp_path / "ovl" / i / "metadata.json").write_text(json.dumps(meta))
        check = tmp_path / "img/Rigid" / i.split("/")[2] / "00000/preview/check.json"
        check.parent.mkdir(parents=True)
        check.write_text(json.dumps({"settle": "pass", "tilt_deg": 0.1}))
    beans = {ids[0]: {"pass": True, "stands": True, "held": 1.0}, ids[1]: {"pass": True, "stands": False}}
    assert S.ready(tmp_path / "ovl", ids[0], tmp_path / "img", beans)
    assert not S.ready(tmp_path / "ovl", ids[1], tmp_path / "img", beans)
    good = json.loads((tmp_path / "ovl" / ids[0] / "metadata.json").read_text())
    assert good["runtime_ready"] and good["source"]["runtime_validated"] and good["adapter"]["validation"]["passed"]
    bad = json.loads((tmp_path / "ovl" / ids[1] / "metadata.json").read_text())
    assert bad["runtime_ready"] is False and bad["adapter"]["validation"]["check_cup_holds_beans"]["stands"] is False


def test_ready_without_beans_uses_floor_check_only(tmp_path):
    ids = [
        "RoboDojo/Rigid/simready_pass/00000",
        "RoboDojo/Rigid/simready_fail/00000",
        "RoboDojo/Rigid/simready_none/00000",
    ]
    for i in ids:
        (tmp_path / "ovl" / i).mkdir(parents=True)
        meta = {"physics": {}, "source": {"runtime_validated": False}, "adapter": {"validation": None}}
        (tmp_path / "ovl" / i / "metadata.json").write_text(json.dumps(meta))
    for i, settle in [(ids[0], "pass"), (ids[1], "fail")]:
        check = tmp_path / "img/Rigid" / i.split("/")[2] / "00000/preview/check.json"
        check.parent.mkdir(parents=True)
        check.write_text(json.dumps({"settle": settle, "tilt_deg": 0.5}))
    assert S.ready(tmp_path / "ovl", ids[0], tmp_path / "img", note="general drop") is True
    assert S.ready(tmp_path / "ovl", ids[1], tmp_path / "img") is False
    assert S.ready(tmp_path / "ovl", ids[2], tmp_path / "img") is False
    assert S.ready(tmp_path / "ovl", "RoboDojo/Rigid/simready_missing_meta/00000", tmp_path / "img") is False
    ok_meta = json.loads((tmp_path / "ovl" / ids[0] / "metadata.json").read_text())
    assert ok_meta["runtime_ready"] is True
    assert ok_meta["adapter"]["validation"]["check_cup_holds_beans"] is None
    assert ok_meta["adapter"]["validation"]["note"] == "general drop"


def test_survey_classifies_ready_converted_raw_and_broken(tmp_path):
    lib = tmp_path / "Object"
    src = tmp_path / "sources"
    src.mkdir()
    (src / "real.usd").write_text("#usda 1.0\n")

    # 1. already_ready
    d1 = lib / "RoboDojo/Rigid/simready_ready_a01/00000"
    d1.mkdir(parents=True)
    (d1 / "sm_ready_a01.usd").symlink_to(src / "real.usd")
    (d1 / "object.usd").write_text("#usda 1.0\n")
    (d1 / "metadata.json").write_text(json.dumps({"runtime_ready": True, "adapter": {"script": S.SCRIPT}}))

    # 2. converted_unvalidated
    d2 = lib / "RoboDojo/Rigid/simready_conv_a01/00000"
    d2.mkdir(parents=True)
    (d2 / "sm_conv_a01.usd").symlink_to(src / "real.usd")
    (d2 / "object.usd").write_text("#usda 1.0\n")
    (d2 / "metadata.json").write_text(json.dumps({"runtime_ready": False}))

    # 3. raw_only
    d3 = lib / "RoboDojo/Rigid/simready_raw_a01/00000"
    d3.mkdir(parents=True)
    (d3 / "sm_raw_a01.usd").symlink_to(src / "real.usd")

    # 4. broken_links
    d4 = lib / "RoboDojo/Rigid/simready_broken_a01/00000"
    d4.mkdir(parents=True)
    (d4 / "sm_broken_a01.usd").symlink_to(src / "does_not_exist.usd")

    # Non-simready dir should be ignored by discover_simready_ids
    (lib / "RoboDojo/Rigid/robocasa_apple/00000").mkdir(parents=True)

    discovered = S.discover_simready_ids(lib)
    assert len(discovered) == 4
    report = S.survey(lib)
    assert report["total"] == 4
    assert report["counts"] == {"already_ready": 1, "converted_unvalidated": 1, "raw_only": 1, "broken_links": 1}
    assert S.survey_one(lib, "RoboDojo/Rigid/simready_absent/00000")["status"] == "missing"


def test_upstream_tree_parsing_diff_and_manifest():
    entries = [
        {"path": "common_assets/props/food_produce_apple_a01/sm_apple_01.usd", "size": 1000, "type": "file"},
        {"path": "common_assets/props/food_produce_apple_a01/payloads/geom.usd", "size": 50_000, "type": "file"},
        {"path": "common_assets/props/food_produce_apple_a01/Textures/albedo.png", "size": 200_000, "type": "file"},
        {"path": "common_assets/props/kitchenware_pan_z99/sm_pan_01.usd", "size": 2000, "type": "file"},
        {"path": "common_assets/props/kitchenware_pan_z99/Textures/metal.png", "size": 5_000_000, "type": "file"},
        {"path": "common_assets/props/furniture_chair_a01/sm_chair_01.usd", "size": 3000, "type": "file"},
        {"path": "common_assets/props/furniture_chair_a01/Textures/wood.png", "size": 300_000_000, "type": "file"},
        {"path": "README.md", "size": 500, "type": "file"},
        {"path": "common_assets/props/food_produce_apple_a01", "size": 0, "type": "directory"},
    ]
    packages, dups = U.parse_upstream_tree(entries)
    assert dups == [] and len(packages) == 3
    by_slug = {p["slug"]: p for p in packages}
    assert by_slug["simready_food_produce_apple_a01"]["size_bytes"] == 251_000
    assert by_slug["simready_food_produce_apple_a01"]["files"] == 3
    assert by_slug["simready_food_produce_apple_a01"]["category"] == "food"

    gap = U.diff_upstream(packages, ["Rigid/simready_food_produce_apple_a01/00000"])
    assert gap["upstream_total"] == 3 and gap["imported_count"] == 1 and gap["missing_count"] == 2
    assert gap["missing_by_category"]["kitchenware"]["count"] == 1
    assert gap["missing_by_category"]["furniture"]["count"] == 1

    # Manifest filtering by max_size_mb excludes 300 MB chair, keeps ~5 MB pan
    manifest = U.build_download_manifest(gap["missing"], max_size_mb=50)
    assert manifest["count"] == 1
    assert manifest["packages"][0]["slug"] == "simready_kitchenware_pan_z99"
    assert manifest["allow_patterns"] == ["common_assets/props/kitchenware_pan_z99/**"]

    # Manifest filtering by category + limit
    m_cat = U.build_download_manifest(gap["missing"], categories=["furniture"], limit=1)
    assert m_cat["count"] == 1 and m_cat["packages"][0]["slug"] == "simready_furniture_chair_a01"

    # Relative Link header pagination resolves cleanly against current URL
    import urllib.parse

    nxt = U._next_link('</api/datasets/nvidia/simready-assets/tree/main?recursive=true&cursor=abc>; rel="next"')
    assert nxt == "/api/datasets/nvidia/simready-assets/tree/main?recursive=true&cursor=abc"
    joined = urllib.parse.urljoin(
        "https://huggingface.co/api/datasets/nvidia/simready-assets/tree/main?recursive=true",
        nxt,
    )
    assert joined == "https://huggingface.co/api/datasets/nvidia/simready-assets/tree/main?recursive=true&cursor=abc"


def test_convert_one_end_to_end_with_fake_pxr(tmp_path, monkeypatch=None):
    """Exercise convert_one() USD traversal, schema removal, collection unbinding (GetBindingName), mesh filtering,
    stable_orientation, estimate_mass, and metadata.json output without requiring Isaac Sim."""
    import types

    # cm standing thin slab -> 0.02 x 0.20 x 0.30 m
    pts_local, counts, indices = box((0.0, 0.0, 0.0), (2.0, 20.0, 30.0))

    class FakePath:
        def __init__(self, path):
            self.pathString = path
            self.isEmpty = not path
            self.name = path.rsplit("/", 1)[-1] if path else ""

        def __eq__(self, other):
            return isinstance(other, FakePath) and self.pathString == other.pathString

        def __str__(self):
            return self.pathString

    class FakeAttr:
        def __init__(self, val):
            self._val = val

        def Get(self):
            return self._val

        def Set(self, val):
            self._val = val

    class FakeColBinding:
        def __init__(self, mat_path, col_name, binding_name):
            self._mat = FakePath(mat_path)
            self._col = col_name
            self._bind = binding_name

        def GetMaterialPath(self):
            return self._mat

        def GetCollectionName(self):
            return self._col

        def GetBindingName(self):
            return self._bind

    class FakePrim:
        def __init__(
            self,
            path,
            type_name="Xform",
            schemas=None,
            attrs=None,
            direct_phys=None,
            col_phys=None,
            purpose="default",
            vis="inherited",
        ):
            self._path = FakePath(path)
            self._type = type_name
            self._schemas = set(schemas or [])
            self._attrs = dict(attrs or {})
            self._direct_phys = FakePath(direct_phys or "")
            self._col_phys = list(col_phys or [])
            self._purpose = purpose
            self._vis = vis
            self._active = True
            self._instance = False

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._path.name

        def GetTypeName(self):
            return self._type

        def IsA(self, cls):
            return self._type == cls._type_name

        def HasAPI(self, api_cls):
            return api_cls._schema_name in self._schemas

        def GetAppliedSchemas(self):
            return list(self._schemas)

        def RemoveAppliedSchema(self, name):
            self._schemas.discard(name)

        def IsInstance(self):
            return self._instance

        def SetInstanceable(self, val):
            self._instance = val

        def SetActive(self, val):
            self._active = val

        def GetReferences(self):
            return types.SimpleNamespace(AddReference=lambda ref: None)

    class FakeStage:
        def __init__(self, prims, up_axis="Y", mpu=0.01, path=None):
            self._prims = {p.GetPath().pathString: p for p in prims}
            self._up = up_axis
            self._mpu = mpu
            self._path = path
            self._default = prims[0] if prims else None

        def GetDefaultPrim(self):
            return self._default

        def SetDefaultPrim(self, p):
            self._default = p

        def Traverse(self):
            return list(self._prims.values())

        def GetPrimAtPath(self, path):
            return self._prims[path]

        def SetEditTarget(self, _):
            pass

        def GetSessionLayer(self):
            return None

        def GetRootLayer(self):
            return types.SimpleNamespace(Save=lambda: Path(self._path).write_text("#usda 1.0\n"))

    # Build source prims: visual mesh, invisible mesh, collider mesh (with collection physics binding), physics material
    vis_mesh = FakePrim(
        "/root/normalize/asset/visual",
        type_name="Mesh",
        schemas={"PhysicsRigidBodyAPI", "PhysicsMassAPI"},
        attrs={"points": pts_local, "counts": np.asarray(counts), "indices": np.asarray(indices)},
    )
    invis_mesh = FakePrim(
        "/root/normalize/asset/hidden",
        type_name="Mesh",
        attrs={"points": np.array([[100.0, 100.0, 100.0]]), "counts": [3], "indices": [0, 0, 0]},
        vis="invisible",
    )
    col_mesh = FakePrim(
        "/root/normalize/asset/collider",
        type_name="Mesh",
        schemas={"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"},
        attrs={"collisionEnabled": True, "approximation": "convexDecomposition"},
        col_phys=[FakeColBinding("/Looks/m_opaque__wood__board", "phys_col", "custom_bind_name")],
    )
    phys_mat = FakePrim(
        "/Looks/m_opaque__wood__board",
        type_name="Material",
        schemas={"PhysicsMaterialAPI"},
        attrs={"density": 650.0},
    )
    src_stage = FakeStage([vis_mesh, invis_mesh, col_mesh, phys_mat], up_axis="Y", mpu=0.01)

    state = {"shift": np.zeros(3), "orient": [1.0, 0.0, 0.0, 0.0], "scale": 1.0, "rot_x90": False, "unbind_calls": []}

    class FakeOp:
        def __init__(self, kind):
            self.kind = kind

        def Set(self, val):
            if self.kind == "translate":
                state["shift"] = np.asarray(val, float)
            elif self.kind == "orient":
                state["orient"] = list(val)
            elif self.kind == "transform":
                state["scale"] = val["scale"]
                state["rot_x90"] = val["rot_x90"]

    class FakeXform:
        _type_name = "Xform"

        def __init__(self, prim):
            self._p = prim

        @classmethod
        def Define(cls, stage, path):
            p = FakePrim(path, "Xform")
            stage._prims[path] = p
            return cls(p)

        def GetPrim(self):
            return self._p

        def AddTranslateOp(self):
            return FakeOp("translate")

        def AddOrientOp(self):
            return FakeOp("orient")

        def AddTransformOp(self, opSuffix=""):
            return FakeOp("transform")

    class FakeXformCache:
        def GetLocalToWorldTransform(self, _prim):
            # Build 4x4 row-vector transform: scale -> rot_x90 -> orient -> translate
            M = np.eye(4)
            M[:3, :3] *= state["scale"]
            if state["rot_x90"]:
                Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # row-vector +90 deg about X
                M[:3, :3] = M[:3, :3] @ Rx
            q = state["orient"]
            R = np.stack([rotate(q, [1, 0, 0]), rotate(q, [0, 1, 0]), rotate(q, [0, 0, 1])], axis=0)
            M[:3, :3] = M[:3, :3] @ R
            M[3, :3] = state["shift"]
            return M

    class FakeBindingAPI:
        def __init__(self, prim):
            self._p = prim

        @classmethod
        def Apply(cls, prim):
            return cls(prim)

        def Bind(self, mat, _strength=None, materialPurpose=None):
            self._p._direct_phys = mat.GetPath()

        def GetDirectBinding(self, materialPurpose=None):
            return types.SimpleNamespace(GetMaterialPath=lambda: self._p._direct_phys)

        def UnbindDirectBinding(self, materialPurpose=None):
            self._p._direct_phys = FakePath("")

        def GetCollectionBindings(self, materialPurpose=None):
            return list(self._p._col_phys)

        def UnbindCollectionBinding(self, bind_name, materialPurpose=None):
            state["unbind_calls"].append(bind_name)
            self._p._col_phys = [b for b in self._p._col_phys if b.GetBindingName() != bind_name]

        def ComputeBoundMaterial(self, materialPurpose=None):
            if self._p._col_phys:
                bound = types.SimpleNamespace(
                    GetPath=lambda: self._p._col_phys[0].GetMaterialPath(),
                    GetPrim=lambda: phys_mat,
                )
                return (bound, None)
            probe_p = FakePrim("/root/probe_material")
            probe = types.SimpleNamespace(
                GetPath=lambda: FakePath("/root/probe_material"),
                GetPrim=lambda: probe_p,
            )
            return (probe, None)

    wrapper_stage = [None]

    class FakeLayer:
        def __init__(self, path):
            self.path = path

    def open_stage(arg):
        if isinstance(arg, FakeLayer):
            st = FakeStage([vis_mesh, invis_mesh, col_mesh], up_axis="Z", mpu=1.0, path=arg.path)
            wrapper_stage[0] = st
            return st
        if str(arg).endswith("object.usd"):
            return wrapper_stage[0]
        return src_stage

    class FakeGf:
        @staticmethod
        def Vec3d(x, y, z):
            return (x, y, z)

        @staticmethod
        def Vec3f(x, y, z):
            return (x, y, z)

        @staticmethod
        def Quatf(*args):
            if len(args) == 2:
                w, v = args
                return (w, v[0], v[1], v[2])
            return tuple(args)

        class Rotation:
            def __init__(self, axis, deg):
                self.deg = deg

        class Matrix4d:
            def __init__(self, _=1):
                self.scale = 1.0
                self.rot_x90 = False

            def SetRotate(self, rot):
                self.rot_x90 = rot.deg == 90

            def SetScale(self, s):
                self.scale = s
                return self

            def __mul__(self, other):
                return {"scale": self.scale * other.scale, "rot_x90": self.rot_x90 or other.rot_x90}

    FakeSdf = types.SimpleNamespace(
        Layer=types.SimpleNamespace(CreateNew=lambda path, args=None: FakeLayer(path)),
        AssetPath=lambda p: p,
    )
    FakeUsd = types.SimpleNamespace(
        Stage=types.SimpleNamespace(Open=open_stage),
        PrimRange=lambda root: list(wrapper_stage[0].Traverse()),
    )
    FakeUsdGeom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(z="Z", guide="guide", invisible="invisible"),
        GetStageUpAxis=lambda st: st._up,
        GetStageMetersPerUnit=lambda st: st._mpu,
        SetStageUpAxis=lambda st, ax: setattr(st, "_up", ax),
        SetStageMetersPerUnit=lambda st, u: setattr(st, "_mpu", u),
        Xform=FakeXform,
        XformCache=FakeXformCache,
        Camera=type("Camera", (), {"_type_name": "Camera"}),
        Mesh=type(
            "Mesh",
            (),
            {
                "_type_name": "Mesh",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetPointsAttr": lambda self: FakeAttr(self._p._attrs.get("points")),
                "GetFaceVertexCountsAttr": lambda self: FakeAttr(self._p._attrs.get("counts")),
                "GetFaceVertexIndicesAttr": lambda self: FakeAttr(self._p._attrs.get("indices")),
            },
        ),
        Imageable=lambda p: types.SimpleNamespace(ComputePurpose=lambda: p._purpose, ComputeVisibility=lambda: p._vis),
    )
    FakeUsdPhysics = types.SimpleNamespace(
        Joint=type("Joint", (), {"_type_name": "PhysicsJoint"}),
        ArticulationRootAPI=type("ArticulationRootAPI", (), {"_schema_name": "PhysicsArticulationRootAPI"}),
        RigidBodyAPI=type(
            "RigidBodyAPI",
            (),
            {
                "_schema_name": "PhysicsRigidBodyAPI",
                "Apply": classmethod(lambda cls, p: p._schemas.add(cls._schema_name)),
            },
        ),
        MassAPI=type(
            "MassAPI",
            (),
            {
                "_schema_name": "PhysicsMassAPI",
                "Apply": classmethod(
                    lambda cls, p: (
                        p._schemas.add(cls._schema_name),
                        types.SimpleNamespace(CreateMassAttr=lambda m: p._attrs.__setitem__("mass", m)),
                    )[1]
                ),
            },
        ),
        CollisionAPI=type(
            "CollisionAPI",
            (),
            {
                "_schema_name": "PhysicsCollisionAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetCollisionEnabledAttr": lambda self: FakeAttr(self._p._attrs.get("collisionEnabled", True)),
            },
        ),
        MeshCollisionAPI=type(
            "MeshCollisionAPI",
            (),
            {
                "_schema_name": "PhysicsMeshCollisionAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetApproximationAttr": lambda self: FakeAttr(self._p._attrs.get("approximation", "none")),
            },
        ),
        MaterialAPI=type(
            "MaterialAPI",
            (),
            {
                "_schema_name": "PhysicsMaterialAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetDensityAttr": lambda self: FakeAttr(self._p._attrs.get("density", 0.0)),
            },
        ),
    )
    FakeUsdShade = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(weakerThanDescendants="weakerThanDescendants"),
        MaterialBindingAPI=FakeBindingAPI,
        Material=types.SimpleNamespace(Define=lambda st, path: FakePrim(path, "Material")),
    )
    FakeUsdUtils = types.SimpleNamespace(
        ComputeAllDependencies=lambda _ap: ([], [], ["OmniPBR.mdl", "gltf/pbr.mdl"]),
    )

    # Set up fake package directory in lib
    oid = "RoboDojo/Rigid/simready_kitchenware_cuttingBoard_a01/00000"
    src_real = tmp_path / "sources/board"
    src_real.mkdir(parents=True)
    (src_real / "sm_board_01.usd").write_text("#usda 1.0\n")
    pkg_dir = tmp_path / "lib" / oid
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "sm_board_01.usd").symlink_to(src_real / "sm_board_01.usd")

    orig_pxr = S.pxr_modules

    def fake_pxr_modules():
        return (FakeGf, None, FakeSdf, FakeUsd, FakeUsdGeom, FakeUsdPhysics, FakeUsdShade, FakeUsdUtils)

    S.pxr_modules = fake_pxr_modules
    try:
        summary = S.convert_one(tmp_path / "lib", oid, tmp_path / "ovl", orient="auto")
    finally:
        S.pxr_modules = orig_pxr

    assert summary["material"] == "wood"
    assert summary["orient_reason"] == "lay_flat_auto"
    assert summary["tabletop_range"] is True
    # Laid flat: 0.02 m thinnest axis is now Z
    assert summary["size_m"][2] == pytest.approx(0.02, abs=1e-4)
    # Collection binding was unbound using GetBindingName() ("custom_bind_name"), NOT GetCollectionName() ("phys_col")
    assert state["unbind_calls"] == ["custom_bind_name"]
    meta = json.loads((tmp_path / "ovl" / oid / "metadata.json").read_text())
    assert meta["runtime_ready"] is False
    assert meta["adapter"]["version"] == 2
    assert meta["adapter"]["orient_reason"] == "lay_flat_auto"
    assert meta["physics"]["mass"] == pytest.approx(0.5)  # 0.02*0.2*0.3*650 = 0.78 kg clamped to 0.5 kg wood cap
