"""Sand3D selection and rigid conversion. No network, no program execution, no Isaac."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("sand3d", Path(__file__).with_name("sand3d.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

MARKER = "SAN3D_PROGRAM_MUST_NOT_RUN"


def row(**overrides):
    base = {
        "id": "simready-fork",
        "usda_path": "usda/0003/fork.usda",
        "n_joints": 0,
        "f_score": 0.95,
        "family": "tableware",
        "license": "CC-BY-4.0",
        "source_license": "CC-BY-4.0",
        "attribution": "NVIDIA SimReady assets: fork, CC BY 4.0; modified",
        "license_status": "ok",
        "source": "simready",
        "asset_id": "simready:fork",
        "program": f"import pathlib; pathlib.Path({MARKER!r})",
        "render": "pixels",
    }
    base.update(overrides)
    return base


def test_select_keeps_rigid_rows_and_never_keeps_the_program():
    result = S.select(
        [
            row(),
            row(id="lamp", n_joints=3, asset_id="articraft:lamp", source="articraft", f_score=1.0, family="task lamp"),
            row(id="nc", source_license="CC-BY-NC-4.0"),
            row(id="anon", attribution="  "),
            row(id="low", source="articraft", f_score=0.85, asset_id="articraft:low"),
            row(id="escape", usda_path="../secret.usda"),
            row(),
        ]
    )
    assert [item["id"] for item in result["chosen"]] == ["Rigid/sand3d_tableware/00000"]
    assert result["chosen"][0]["source_license"] == "CC-BY-4.0"
    assert result["chosen"][0]["attribution"].startswith("NVIDIA")
    reasons = {item["id"]: item["reason"] for item in result["dropped"]}
    assert reasons["lamp"] == "articulated"
    assert reasons["nc"].startswith("source license")
    assert reasons["anon"] == "missing attribution"
    assert reasons["low"].startswith("f_score")
    assert reasons["escape"] == "usda path"
    assert reasons["simready-fork"] == "duplicate id"
    text = json.dumps(result)
    assert MARKER not in text and "program" not in text
    assert "Blocks" not in text and "wood_block" not in text


def test_select_limit_is_stable_and_articraft_floor_is_higher():
    rows = [
        row(id="b", family="tool", asset_id="simready:b"),
        row(id="a", family="tool", asset_id="simready:a", f_score=0.75),
    ]
    result = S.select(rows, limit=1)
    assert result["chosen"][0]["asset_id"] == "simready:a"
    assert result["dropped"][0] == {"id": "b", "reason": "over limit"}
    kept = S.select([row(id="ok", source="articraft", f_score=0.9, asset_id="articraft:ok", family="cup")])
    assert kept["chosen"][0]["id"] == "Rigid/sand3d_cup/00000"


def test_parquet_reader_does_not_load_program_or_render(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "rows.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "id": ["simready-box"],
                "asset_id": ["simready:box"],
                "usda_path": ["usda/0003/box.usda"],
                "n_joints": [0],
                "f_score": [1.0],
                "family": ["packaging"],
                "license": ["CC-BY-4.0"],
                "source_license": ["CC-BY-4.0"],
                "attribution": ["NVIDIA, CC BY 4.0; modified"],
                "license_status": ["ok"],
                "source": ["simready"],
                "program": [MARKER],
                "render": ["png"],
            }
        ),
        path,
    )
    loaded = S.read_parquet(path)
    assert loaded[0]["id"] == "simready-box"
    assert "program" not in loaded[0] and "render" not in loaded[0]
    assert S.select(loaded)["chosen"][0]["id"] == "Rigid/sand3d_packaging/00000"


def cube(path, translate=(0.2, 0.0, 0.0), size=0.1, joint=False, up="Z"):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, up)
    root = UsdGeom.Xform.Define(stage, "/Box")
    stage.SetDefaultPrim(root.GetPrim())
    link = UsdGeom.Xform.Define(stage, "/Box/Geometry/base")
    link.AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Box/Geometry/base/body")
    half = size / 2
    points = [(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)]
    # 0 ---, 1 --+, 2 -+-, 3 -++, 4 +--, 5 +-+, 6 ++-, 7 +++
    quads = [
        (0, 2, 6, 4),
        (1, 5, 7, 3),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 1, 3, 2),
        (4, 6, 7, 5),
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([index for face in quads for index in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    if joint:
        hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Box/Joints/hinge")
        hinge.CreateBody0Rel().SetTargets([link.GetPath()])
        hinge.CreateBody1Rel().SetTargets([link.GetPath()])
        hinge.CreateAxisAttr("Z")
        hinge.CreateLowerLimitAttr(0)
        hinge.CreateUpperLimitAttr(45)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    stage.GetRootLayer().Save()
    return stage


def entry(src="box.usda", **overrides):
    base = {
        "id": "Rigid/sand3d_packaging/00000",
        "src": src,
        "dataset": "simready",
        "asset_id": "simready:box",
        "family": "packaging",
        "f_score": 1.0,
        "license": "CC-BY-4.0",
        "source_license": "CC-BY-4.0",
        "attribution": "NVIDIA SimReady assets: box, CC BY 4.0; modified",
    }
    base.update(overrides)
    return base


def test_fine_hulls_are_capped_for_physx_particles():
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(0)
    points = rng.normal(size=(400, 3))
    points /= numpy.linalg.norm(points, axis=1, keepdims=True) * 20  # 10 cm sphere
    vertices, _, volume, source_vertices = S.convex_hull(points)
    assert source_vertices > S.HULL_VERTICES
    assert len(vertices) <= S.HULL_VERTICES
    assert volume > 0


def test_conversion_centers_mesh_adds_hull_and_keeps_attribution(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    cube(tmp_path / "box.usda")
    done = S.convert_one(tmp_path, entry(), tmp_path / "out")
    assert done["size_m"] == pytest.approx([0.1, 0.1, 0.1])
    assert done["mass_kg"] == pytest.approx(0.25, abs=0.02)
    assert done["source_license"] == "CC-BY-4.0"
    target = Path(done["dir"])
    meta = json.loads((target / "metadata.json").read_text())
    assert meta["runtime_ready"] is False
    assert meta["source"]["program"] == "not executed"
    assert meta["source"]["attribution"].startswith("NVIDIA")
    assert meta["source"]["source_license"] == "CC-BY-4.0"
    assert meta["source"]["measured"] is False
    assert meta["source"]["collision"]["vertices"] <= S.HULL_VERTICES
    assert meta["source"]["collision"]["source_vertices"] == 8
    assert meta["physics"]["mass"] == done["mass_kg"]
    assert "static_friction" not in meta["physics"]
    description = json.loads((target / "description.json").read_text())
    assert "NVIDIA" in description["description"][0]
    stage = Usd.Stage.Open(str(target / "object.usdz"))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    bodies = [prim.GetPath().pathString for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert bodies == ["/World"]
    assert not any(prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse())
    assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    center = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange().GetMidpoint()
    assert abs(center[0]) < 1e-4 and abs(center[1]) < 1e-4 and abs(center[2]) < 1e-4
    assert "Blocks" not in str(target) and "wood_block" not in str(target)


def test_conversion_rejects_joints_oversized_meshes_and_bad_ids(tmp_path):
    pytest.importorskip("pxr")
    cube(tmp_path / "hinge.usda", joint=True)
    with pytest.raises(ValueError, match="joint"):
        S.convert_one(tmp_path, entry("hinge.usda"), tmp_path / "out")
    cube(tmp_path / "room.usda", size=2.0, translate=(0, 0, 0))
    with pytest.raises(ValueError, match="tabletop"):
        S.convert_one(tmp_path, entry("room.usda"), tmp_path / "out")
    cube(tmp_path / "yup.usda", up="Y")
    with pytest.raises(ValueError, match="Z-up"):
        S.convert_one(tmp_path, entry("yup.usda"), tmp_path / "out")
    cube(tmp_path / "box.usda")
    with pytest.raises(ValueError, match="sha256"):
        S.convert_one(tmp_path, entry(usda_sha256="0" * 64), tmp_path / "out")
    with pytest.raises(ValueError, match="refusing id"):
        S.convert_one(tmp_path, entry(id="Rigid/wood_block_cube/00000"), tmp_path / "out")
