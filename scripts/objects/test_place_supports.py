"""Place-anything support adapters: laying flat, masses, passive.support, readiness and the round-2 pool (no Isaac)."""

import json
import math

import numpy as np
import pytest

from scripts.objects import place_supports as P


def rotate(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


@pytest.mark.parametrize("extents", [(0.01, 0.3, 0.2), (0.3, 0.012, 0.2), (0.3, 0.2, 0.02)])
def test_lay_flat_turns_the_thinnest_axis_up(extents):
    q = P.lay_flat(extents)
    assert math.isclose(sum(c * c for c in q), 1.0)
    axis = np.eye(3)[int(np.argmin(extents))]
    assert np.allclose(rotate(q, axis), (0, 0, 1))


def test_supports_cover_both_sources_and_clutter_ids_point_at_their_object():
    simready = [k for k, v in P.SUPPORTS.items() if v is None]
    clutter = {k: v for k, v in P.SUPPORTS.items() if v}
    assert len(simready) == 16 and all(k.startswith("RoboDojo/Rigid/simready_") for k in simready)
    assert clutter["RoboDojo/Rigid/clutter_book/00001"] == "RoboDojo/Clutter/book/00001"
    assert set(c.split("/", 2)[2] for c in clutter.values()) == set(P.MASS)


def test_masses_are_clamped_to_the_material_and_the_loader_cap():
    assert P.clamp_mass(2.0, "wood") == (0.5, [0.15, 0.5])
    assert P.clamp_mass(0.01, "paper")[0] == 0.05
    assert P.kind_of(["physics_mat_m_opaque__wood__board_a01"], "x") == "wood"
    assert P.kind_of([], "simready_kitchenware_cuttingBoard_a01") == "wood"
    assert P.kind_of([], "simready_misc_notebook_a01") == "paper"
    with pytest.raises(ValueError):
        P.clamp_mass(float("nan"), "wood")
    assert all(mass <= 0.5 for mass, _, _ in P.MASS.values())


GEOM = {"region": {"center": [0.001, 0.0], "half": [0.08, 0.1], "radius": 0.08}, "floor": 0.01, "rim": 0.01,
        "top": 0.0105, "bottom": -0.01}  # fmt: skip


def test_support_entry_keeps_the_existing_circle_form():
    entry = P.support_entry(GEOM)["top/0"]
    assert entry["center"] == [[0.001, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0]]
    assert entry["radius"] == [0.08] and entry["half"] == [[0.08, 0.1]]


def test_ready_needs_a_settled_check_and_a_face(tmp_path):
    oid = "RoboDojo/Rigid/clutter_book/00001"
    (tmp_path / "ovl" / oid).mkdir(parents=True)
    meta = {"passive": {"functional": {}, "support": {}}, "adapter": {}, "source": {}}
    (tmp_path / "ovl" / oid / "metadata.json").write_text(json.dumps(meta))
    check = tmp_path / "check" / "Rigid" / "clutter_book" / "00001" / "preview"
    check.mkdir(parents=True)
    (check / "check.json").write_text(json.dumps({"settle": "pass", "tilt_deg": 0.0}))
    assert not P.ready(tmp_path / "ovl", oid, tmp_path / "check", {})
    assert P.ready(tmp_path / "ovl", oid, tmp_path / "check", {oid: {"support": GEOM}}, note="job 1")
    out = json.loads((tmp_path / "ovl" / oid / "metadata.json").read_text())
    assert out["runtime_ready"] and out["passive"]["support"]["top/0"]["radius"] == [0.08]
    assert out["adapter"]["validation"]["note"] == "job 1"
    assert out["adapter"]["validation"]["place_screening"] is None
    P.ready(tmp_path / "ovl", oid, tmp_path / "check", {oid: {"support": GEOM}}, stats={"receptacle": {oid: [1, 7]}})
    screening = json.loads((tmp_path / "ovl" / oid / "metadata.json").read_text())["adapter"]["validation"]
    assert screening["place_screening"]["passed"] is False and screening["place_screening"]["recorded"] == 7


def entry(oid, size, kinds=("support",)):
    return {"id": oid, "size": size, "kinds": list(kinds), "geometry": {k: GEOM for k in kinds},
            "split": {k: "train" for k in kinds}}  # fmt: skip


def test_round2_keeps_round1_and_holds_out_new_supports_apart():
    r1 = [dict(entry("Real2Sim/Rigid/marker_box/00001", [0.12, 0.08, 0.02]), split={"support": "eval"})]
    new = [entry(f"RoboDojo/Rigid/clutter_book/0000{i}", [0.18, 0.23, 0.02]) for i in range(7)]
    new += [
        entry("RoboDojo/Rigid/simready_kitchenware_cuttingBoard_d01/00000", [0.27, 0.41, 0.01]),
        entry("RoboDojo/Rigid/clutter_box/00001", [0.18, 0.17, 0.05]),
        entry("Real2Sim/Rigid/marker_box/00001", [0.12, 0.08, 0.02]),
    ]
    stats = {"receptacle": {"RoboDojo/Rigid/clutter_box/00001": [0, 4]}}
    out, left = P.round2(r1, new, stats)
    assert out[0] == r1[0]
    added = out[1:]
    assert [r["id"] for r in added] == [f"RoboDojo/Rigid/clutter_book/0000{i}" for i in range(7)]
    assert sum(r["split"]["support"] == "eval" for r in added) == 1
    assert all(r["added_in"] == "r2" and r["mass_estimated"] for r in added)
    assert "off the desk" in left["RoboDojo/Rigid/simready_kitchenware_cuttingBoard_d01/00000"]
    assert left["RoboDojo/Rigid/clutter_box/00001"] == "0/4 accepted in the screening"


def test_the_round2_pool_file_adds_named_validated_supports():
    data = json.loads((P.Path(P.__file__).with_name("place_anything_receptacles_r2.json")).read_text())
    added = [r for r in data["receptacles"] if r.get("added_in") == "r2"]
    r1 = json.loads((P.Path(P.__file__).with_name("place_anything_receptacles.json")).read_text())["receptacles"]
    assert data["round"] == "r2" and data["release"] == 38 and data["receptacles"][: len(r1)] == r1
    assert len(data["receptacles"]) == len(r1) + 12 and sum(r["split"]["support"] == "eval" for r in added) == 2
    names = json.loads(P.Path(P.__file__).with_name("object_names.json").read_text())
    for r in added:
        assert r["id"] in P.SUPPORTS and r["id"].split("/", 1)[1] in names
        assert r["geometry"]["support"]["region"]["radius"] >= 0.03 and P.on_desk(r["size"])
