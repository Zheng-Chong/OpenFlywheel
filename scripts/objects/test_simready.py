"""SimReady adapter: material, mass, mesh volume, placement metadata, package mirroring and readiness (no Isaac)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("simready", Path(__file__).with_name("simready.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def box(lo, hi):
    """Closed, outward-wound quad mesh of an axis-aligned box: points, faceVertexCounts, faceVertexIndices."""
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return pts, [4] * 6, [i for q in quads for i in q]


def test_material_from_physics_material_name_then_package_name():
    assert S.material_kind(["physics_mat_m_opaque__ceramic__mug_a01"], "simready_kitchenware_mug_a01") == "ceramic"
    assert S.material_kind(["physics_mat_m_trans__glass__clean_a01"], "x") == "glass"
    assert S.material_kind(["physics_mat_m_opaque__paper__cup_c01"], "simready_kitchenware_cup_c01") == "paper"
    assert S.material_kind([], "simready_dishware_glass_wineFlute_a01") == "glass"
    with pytest.raises(ValueError):
        S.material_kind(["physics_mat_m_opaque__metal__x"], "simready_kitchenware_pot_a01")


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


def test_mass_is_volume_times_density_clamped_to_the_material_range():
    mass, why = S.choose_mass(120e-6, 2600, "ceramic")  # a 120 cm^3 stoneware mug
    assert mass == pytest.approx(0.312) and not why["clamped"] and not why["measured"]
    assert S.choose_mass(272e-6, 2600, "ceramic")[0] == 0.45  # chunky mug: capped below the loader's 0.5 kg
    mass, why = S.choose_mass(1e-6, 689, "paper")
    assert mass == S.MASS_RANGE["paper"][0] and why["clamped"] and why["raw_kg"] == pytest.approx(0.0007, abs=1e-4)
    for volume, density in [(0, 2600), (1e-4, float("nan"))]:
        with pytest.raises(ValueError):
            S.choose_mass(volume, density, "glass")


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
    with pytest.raises(ValueError, match="relative link"):
        S.link_package(lib, out)


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
