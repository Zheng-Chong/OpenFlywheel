"""RoboCasa importer: MJCF parsing, joint axes, the generated cabinet units, rigid objects (no Isaac; pxr optional)."""

import importlib.util
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("robocasa", Path(__file__).with_name("robocasa.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

MICROWAVE = """<mujoco model="model"><compiler angle="radian"/>
<asset><mesh file="visuals/door.obj" name="door_vis"/><texture file="visuals/T.png" name="t" type="2d"/>
<material name="m" rgba="1 1 1 1" texture="t"/></asset>
<default><default class="visual"><geom contype="0" conaffinity="0" type="mesh"/></default>
<default class="collision"><geom group="0"/></default><default class="region"><geom contype="0"/></default></default>
<worldbody><body><body name="object">
  <geom class="collision" type="box" pos="0 0 -0.1" size="0.25 0.15 0.01"/>
  <geom class="region" name="reg_main" type="box" size="0.25 0.15 0.15"/>
  <body name="door">
    <joint axis="0 0 1" pos="-0.23 -0.11 0" limited="true" name="microjoint" range="-1.5708 0" type="hinge"
      damping="1" stiffness="0.05"/>
    <geom mesh="door_vis" class="visual" material="m"/>
    <geom class="collision" type="box" pos="-0.05 -0.11 0" size="0.16 0.01 0.13"/>
    <body name="door_Clear"><geom class="collision" type="box" pos="0 -0.11 0" size="0.14 0.002 0.1"/></body>
  </body>
  <body name="Disc001">
    <joint axis="0 0 1" pos="0 0 -0.13" limited="false" name="Disc001_joint" range="0 360" type="hinge"/>
    <geom class="collision" type="cylinder" pos="0 0 -0.13" size="0.1 0.002"/>
  </body>
</body></body></worldbody><actuator><motor joint="microjoint" gear="10"/></actuator></mujoco>"""


def rotate(q, v):
    return R._rotate(q, v)


def test_x_to_turns_x_onto_any_axis():
    for axis in ([0, 0, 1], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0.98, -0.194]):
        n = math.sqrt(sum(a * a for a in axis))
        assert rotate(R.x_to(axis), [1, 0, 0]) == pytest.approx([a / n for a in axis], abs=1e-6)


def test_euler_is_mujoco_xyz():
    q = R.euler_xyz([math.pi / 2, 0, 0])
    assert rotate(q, [0, 1, 0]) == pytest.approx([0, 0, 1], abs=1e-9)


def test_parse_mjcf_links_joints_and_helpers(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text(MICROWAVE)
    m = R.parse_mjcf(xml)
    assert set(m["links"]) == {"base", "door", "Disc001"}  # door_Clear merged into the door
    assert len(m["links"]["door"]) == 3 and [g["kind"] for g in m["links"]["base"]] == ["collision"]  # no region
    door, disc = m["joints"]
    assert door["type"] == "revolute" and door["range"] == pytest.approx([-1.5708, 0]) and door["stiffness"] == 0.05
    assert disc["range"] is None  # limited="false": a turntable
    assert R.model_size(m) == pytest.approx([0.5, 0.3, 0.13 + 0.01 + 0.13 - 0.002 + 0.0], abs=0.02)


def test_parse_mjcf_refuses_body_poses(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text(MICROWAVE.replace('<body name="door">', '<body name="door" pos="0 0 1">'))
    with pytest.raises(ValueError, match="pose"):
        R.parse_mjcf(xml)


def test_parse_mjcf_folds_jointless_body_poses_and_grouped_colliders(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text(
        MICROWAVE.replace('<body name="door_Clear">', '<body name="door_Clear" pos="0 0.5 0">').replace(
            '<geom class="collision" type="box" pos="0 0 -0.1"', '<geom class="collision_2" type="box" pos="0 0 -0.1"'
        )
    )
    m = R.parse_mjcf(xml)
    assert m["links"]["door"][-1]["pos"] == pytest.approx([0, -0.11 + 0.5, 0])  # the part's pose, in its geom
    assert [g["kind"] for g in m["links"]["base"]] == ["collision"]  # collision_2 is a collider, the region is not


OBJECT = """<mujoco model="model"><compiler angle="radian"/>
<asset><mesh file="visual/cup.obj" name="vis"/><mesh file="collision/c0.obj" name="c0"/>
<material name="m" rgba="0.8 0.2 0.2 1"/></asset>
<default><default class="visual"><geom contype="0" conaffinity="0" type="mesh"/></default>
<default class="collision"><geom group="0" density="1000"/></default></default>
<worldbody><body><body name="object">
  <geom mesh="vis" class="visual" material="m"/>
  <geom mesh="c0" type="mesh" class="collision"/>
  <body pos="0 0 0.05"><geom class="collision" type="box" size="0.02 0.02 0.01"/></body>
</body></body></worldbody></mujoco>"""


def test_write_rigid_object(tmp_path):
    pxr = pytest.importorskip("pxr")
    trimesh = pytest.importorskip("trimesh")
    for sub in ("visual", "collision"):
        (tmp_path / sub).mkdir()
    box = trimesh.creation.box((0.06, 0.06, 0.1))
    box.export(tmp_path / "visual/cup.obj")
    box.export(tmp_path / "collision/c0.obj")
    (tmp_path / "model.xml").write_text(OBJECT)
    size, volume, mass, colliders, _ = R.write_rigid(R.parse_mjcf(tmp_path / "model.xml"), tmp_path / "out")
    assert size == pytest.approx([0.06, 0.06, 0.1]) and colliders == 2
    assert mass == pytest.approx(min(volume * R.OBJECT_DENSITY, R.OBJECT_MASS_KG[1]), abs=1e-4)
    stage = pxr.Usd.Stage.Open(str(tmp_path / "out/object.usdz"))
    bodies = [p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(pxr.UsdPhysics.RigidBodyAPI)]
    hulls = [p for p in stage.Traverse() if p.HasAPI(pxr.UsdPhysics.MeshCollisionAPI)]
    assert bodies == ["/World"] and len(hulls) == 2
    assert all(pxr.UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull" for p in hulls)


def fake_parts(source, kind, n, size, extra=""):
    for i in range(1, n + 1):
        d = (
            source
            / "fixtures_lightwheel"
            / kind
            / f"{'CabinetDoorPanel' if 'panel' in kind else 'CabinetHandle'}{i:03d}"
        )
        d.mkdir(parents=True)
        (d / "model.xml").write_text(
            f"""<mujoco><compiler angle="radian"/><asset><mesh file="visuals/a.obj" name="v"/>
            <material name="mat" rgba="1 1 1 1"/></asset><worldbody><body><body name="object">
            <geom mesh="v" class="visual" material="mat"/>
            <geom type="box" pos="0 0 0" size="{size}" class="collision"/>{extra}
            </body></body></worldbody></mujoco>"""
        )


def test_cabinet_units_are_reproducible_and_work(tmp_path):
    fake_parts(tmp_path, "cabinets/cabinet_panels", 51, "0.2157 0.0095 0.3485")
    fake_parts(tmp_path, "handles", 51, "0.0085 0.0023 0.065")
    units = R.cabinet_units(tmp_path)
    assert len(units) == len(R.CABINET_UNITS) * R.UNITS_PER_KIND == len({u["name"] for u in units})
    assert units == R.cabinet_units(tmp_path)
    by_kind = {u["kind"]: u for u in units}
    double = R.build_unit(tmp_path, by_kind["double_door"])
    w, d, h = R.CABINET_UNITS["double_door"][0]
    left, right = double["joints"]
    assert left["axis"] == [0, 0, -1] and right["axis"] == [0, 0, 1]  # both swing out to -Y
    assert left["pos"][0] == pytest.approx(-w / 2 + R.GAP) and right["pos"][0] == pytest.approx(w / 2 - R.GAP)
    assert R.model_size(double) == pytest.approx([w, d + R.BOARD, h], abs=0.03)  # fronts stand ahead of the carcass
    drawers = R.build_unit(tmp_path, by_kind["two_drawer"])
    assert [j["type"] for j in drawers["joints"]] == ["prismatic", "prismatic"]
    assert drawers["joints"][0]["range"][1] == pytest.approx(R.DRAWER_TRAVEL * R.CABINET_UNITS["two_drawer"][0][1])
    assert 2 < R.unit_mass(drawers) < 20  # board volume x density, kg
    for unit_size, _, _ in R.CABINET_UNITS.values():
        assert max(unit_size) <= 0.65  # tabletop scale


def test_registry_scale_only_where_registered():
    assert R.registry_scale("ToasterOven001", [0.5, 0.66, 0.3]) == pytest.approx(0.5)
    assert R.registry_scale("Microwave079", [0.48, 0.29, 0.29]) == 1.0
    assert set(R.TOASTER_OVEN_DEPTH_M).isdisjoint(R.UPSTREAM_EXCLUDED["toaster_ovens"])
