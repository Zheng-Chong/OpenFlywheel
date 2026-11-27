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
    assert set(R.TOASTER_OVEN_DEPTH_M).isdisjoint(R.FIXTURE_TARGET_MAX_M)


def test_parse_mjcf_contact_excludes(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text(
        MICROWAVE.replace(
            "</mujoco>",
            '<contact><exclude body1="door_Clear" body2="Disc001"/>'
            '<exclude body1="door" body2="door_Clear"/>'
            '<exclude body1="Disc001" body2="door"/></contact></mujoco>',
        )
    )
    m = R.parse_mjcf(xml)
    # door_Clear merges into door -> self-exclusion dropped, duplicate pair deduplicated
    assert m["exclude_contacts"] == [("Disc001", "door")]


def test_open_door_rack_pairs_detects_blocked_dishwasher_rack(tmp_path):
    # Bottom-hinged door at y=-0.30, z=-0.35 swings 90 deg down (about +X -> opens to -Y, horizontal at z=-0.35).
    # Lower rack slides out along -Y at z=-0.345 (collides with open door); upper rack at z=0.10 (clear).
    dw_xml = """<mujoco model="dw"><compiler angle="radian"/>
    <worldbody><body><body name="object">
      <geom class="collision" type="box" pos="0 0 0" size="0.3 0.3 0.4"/>
      <body name="door">
        <joint name="door_hinge" type="hinge" pos="0 -0.30 -0.35" axis="-1 0 0" range="0 1.5708"/>
        <geom class="collision" type="box" pos="0 -0.30 0.0" size="0.28 0.02 0.35"/>
      </body>
      <body name="lower_rack">
        <joint name="lower_slide" type="slide" pos="0 0 -0.345" axis="0 -1 0" range="0 0.45"/>
        <geom class="collision" type="box" pos="0 0 -0.345" size="0.25 0.25 0.02"/>
      </body>
      <body name="upper_rack">
        <joint name="upper_slide" type="slide" pos="0 0 0.15" axis="0 -1 0" range="0 0.45"/>
        <geom class="collision" type="box" pos="0 0 0.15" size="0.25 0.25 0.02"/>
      </body>
    </body></body></worldbody></mujoco>"""
    xml = tmp_path / "model.xml"
    xml.write_text(dw_xml)
    m = R.parse_mjcf(xml)
    assert R.open_door_rack_pairs(m) == [("door", "lower_rack")]

    # Also matches when role/keywords appear in joint name or alternative rack names (shelf/grill).
    alt_xml = dw_xml.replace('name="door"', 'name="front_panel"').replace('name="lower_rack"', 'name="bottom_shelf"')
    xml.write_text(alt_xml)
    assert R.open_door_rack_pairs(R.parse_mjcf(xml)) == [("front_panel", "bottom_shelf")]


def test_size_recovery_covers_all_16_failed_fixtures():
    import json

    report = Path(__file__).resolve().parents[2] / "real2sim/articulated/robocasa.json"
    data = json.loads(report.read_text())
    size_failed = [it for it in data["failed"] if "size" in it.get("why", [])]
    assert len(size_failed) == 16
    for it in size_failed:
        kind = it["id"].split("/")[1].removeprefix("robocasa_")
        name = Path(it["src"]).name
        scale = R.registry_scale(name, it["size_m"])
        scaled = [round(v * scale, 4) for v in it["size_m"]]
        assert R.size_check(kind, scaled)["result"] == "pass", f"{name} still fails: {scaled}"
        if kind == "oven":
            assert max(scaled) <= R.TABLESIDE_MAX_M


def test_select_failed_preserves_ids_and_scales():
    report = Path(__file__).resolve().parents[2] / "real2sim/articulated/robocasa.json"
    manifest = R.select_failed(report)
    assert len(manifest["chosen"]) == 193
    art = [e for e in manifest["chosen"] if not e.get("rigid")]
    rigid = [e for e in manifest["chosen"] if e.get("rigid")]
    assert len(art) == 26 and len(rigid) == 167
    scaled_art = [e for e in art if e.get("scale", 1.0) != 1.0]
    assert len(scaled_art) == 12
    # TomatoSlice004 override brings [5.1848, 5.1848, 0.4899] into OBJECT_SIZE_M
    s = R.OBJECT_SCALE_OVERRIDES["TomatoSlice004"]
    scaled_tomato = [v * s for v in (5.1848, 5.1848, 0.4899)]
    assert R.OBJECT_SIZE_M[0] <= max(scaled_tomato) <= R.OBJECT_SIZE_M[1]
    assert min(scaled_tomato) >= R.OBJECT_THIN_M


def test_compute_stable_pose_keeps_upright_and_rotates_unstable():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from sand3d import convex_hull

    def box_hull(sx, sy, sz, offset=(0.0, 0.0, 0.0)):
        pts = np.array(np.meshgrid([-sx / 2, sx / 2], [-sy / 2, sy / 2], [-sz / 2, sz / 2])).T.reshape(-1, 3)
        pts = pts + np.asarray(offset, float)
        v, f, vol, _ = convex_hull(pts)
        return (v, f, vol)

    # 1. Upright mug/box (tipping angle ~28 deg >> 6 deg): stays identity
    rot, deg = R.compute_stable_pose([box_hull(0.06, 0.06, 0.10)])
    assert deg == 0.0 and np.allclose(rot, np.eye(3))

    # 2. Thin tall wall-mounted utensil rack (tipping angle atan(0.01/0.18) = 3.2 deg < 6 deg): rotates ~90 deg flat
    rack = box_hull(0.40, 0.02, 0.36)
    rot, deg = R.compute_stable_pose([rack])
    assert deg == pytest.approx(90.0, abs=1.0)
    rotated = rack[0] @ rot.T
    extents = rotated.max(0) - rotated.min(0)
    assert extents[2] == pytest.approx(0.02, abs=1e-4)  # now lying flat on its 0.40 x 0.36 face

    # 3. Rounded 64-vertex ball (like CookieDoughBall001, R=30 mm) balanced on a vertex: must not falsely stay at eye(3)
    sphere_pts = 0.03 * R._unit_sphere()
    # Add an explicit south-pole vertex so authored pose rests on a single point rather than a flat facet
    sphere_pts = np.vstack([sphere_pts, [[0.0, 0.0, -0.0305]]])
    sv, sf, svol, _ = convex_hull(sphere_pts)
    rot, deg = R.compute_stable_pose([(sv, sf, svol)])
    assert deg > 0.5 and not np.allclose(rot, np.eye(3))


def test_names_skips_missing_metadata(tmp_path):
    manifest = {
        "chosen": [
            {
                "id": "Rigid/robocasa_tomato_slice/00003",
                "src": "objects_lightwheel/tomato_slice/TomatoSlice004",
                "kind": "tomato_slice",
                "rigid": True,
            }
        ]
    }
    assert R.names(tmp_path, manifest, tmp_path) == {}


def test_align_hulls_to_visual_z_fixes_floating_and_sinking():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from sand3d import convex_hull

    def box_pts(sx, sy, sz, z_center=0.0):
        pts = np.array(np.meshgrid([-sx / 2, sx / 2], [-sy / 2, sy / 2], [-sz / 2, sz / 2])).T.reshape(-1, 3)
        pts[:, 2] += z_center
        return pts

    vis = box_pts(0.30, 0.20, 0.04, z_center=0.0)  # vis z in [-0.02, +0.02]

    # 1. Already aligned -> no-op
    v0, f0, vol0, _ = convex_hull(vis)
    hulls, fix = R.align_hulls_to_visual_z([(v0, f0, vol0)], vis, -0.02, 0.02)
    assert fix is None and len(hulls) == 1

    # 2. Floating (collider extends 10 mm below visual, like oven_tray) -> bottom clamped to -0.02
    col_float = box_pts(0.30, 0.20, 0.05, z_center=-0.005)  # z in [-0.03, +0.02]
    vf, ff, volf, _ = convex_hull(col_float)
    hulls, fix = R.align_hulls_to_visual_z([(vf, ff, volf)], vis, -0.02, 0.02)
    assert fix is not None and fix.get("clamped_bottom")
    assert min(h[0][:, 2].min() for h in hulls) == pytest.approx(-0.02, abs=1e-4)

    # 3. Sinking (collider stops 40 mm above visual bottom, like BlenderJug012) -> base hull added
    vis_jug = box_pts(0.16, 0.16, 0.24, z_center=0.0)  # z in [-0.12, +0.12]
    col_sink = box_pts(0.16, 0.16, 0.20, z_center=0.02)  # z in [-0.08, +0.12]
    vs, fs, vols, _ = convex_hull(col_sink)
    hulls, fix = R.align_hulls_to_visual_z([(vs, fs, vols)], vis_jug, -0.12, 0.12)
    assert fix is not None and fix.get("added_base_hull") and len(hulls) == 2
    assert min(h[0][:, 2].min() for h in hulls) == pytest.approx(-0.12, abs=1e-4)
