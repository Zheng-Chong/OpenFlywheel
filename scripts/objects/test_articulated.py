"""The articulated-object adapter layer and the importers' selection rules (no Isaac; usd-core for the layer)."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("articulated")
V = load("artvip")
P = load("partnet_mobility")


def test_roles_follow_link_names():
    assert A.part_of("E_door_1", "revolute") == "door"
    assert A.part_of("E_drawer_2_35", "prismatic") == "drawer"
    assert A.part_of("E_switch_11", "prismatic") == "button"
    assert A.part_of("E_switch_1_2", "revolute") == "knob"
    assert A.part_of("E_base_plate_7", "revolute") == "turntable"
    assert A.part_of("E_leaves_1_2", "revolute") == "lid"
    assert A.part_of("E_body_3", "revolute") == "other"
    assert A.part_of("E_pump_2", "prismatic") == "pump"
    assert A.part_of("E_cap_3", "revolute") == "cap"
    assert A.part_of("E_capsule_holder_2", "prismatic") == "other"
    assert A.part_of("E_grill_1", "prismatic") == "rack"
    assert A.part_of("E_tray001_3", "prismatic") == "tray"
    assert A.part_of("E_screen1_7", "revolute") == "screen"
    assert A.part_of("E_displayer_5", "revolute") == "screen"
    assert A.part_of("drawer_front-3 handle-2", "prismatic") == "drawer"
    assert [A.role_of(p) for p in ("pump", "cap", "rack", "tray", "screen", "cover")] == ["other"] * 6
    assert A.role_of("screen", V.PART_ROLES["laptop"]) == "lid"
    assert A.role_of("tray", V.PART_ROLES["stationery/letter_tray"]) == "drawer"
    assert [A.role_of(p) for p in ("door", "lid", "turntable", "handle", "pedal")] == [
        "door",
        "lid",
        "turntable",
        "handle",
        "pedal",
    ]


def test_motion_corrects_the_name():
    jp = A.joint_part
    # extents and sizes (x, y, z, m) of the 2026-11-18 audit cases
    assert jp("knob", "revolute", -80.0, 0.0, [0.32, 0.255, 0.004], [0.32, 0.26, 0.07]) == "lid"  # ArtVIP folder_25
    assert jp("other", "revolute", 0.0, 90.0, [0.75, 0.4, 0.021], [0.76, 0.43, 0.7]) == "lid"  # PartNet 46699 top
    assert jp("other", "revolute", 0.0, 90.0, [0.341, 0.0105, 0.17], [0.5, 0.34, 0.23]) == "door"
    assert jp("tray", "revolute", None, None) == "turntable"  # PartNet microwave glass plate, continuous
    assert jp("tray", "revolute", -90.0, 0.0) == "tray"  # a coffee machine's fold-down cup tray
    assert jp("tray", "prismatic", 0.0, 0.16, [0.3, 0.3, 0.01], [0.4, 0.4, 0.3]) == "tray"  # an oven's pull-out tray
    assert jp("knob", "prismatic", -0.002, 0.0, [0.014, 0.014, 0.004], [0.38, 0.14, 0.28]) == "button"
    assert jp("button", "revolute", 0.0, 90.0, [0.03, 0.01, 0.03], [0.5, 0.4, 0.3]) == "knob"
    assert jp("door", "prismatic", 0.0, 0.004, [0.05, 0.01, 0.02], [0.5, 0.32, 0.26]) == "button"
    assert jp("drawer", "prismatic", -0.2, 0.0, [0.4, 0.3, 0.1], [0.5, 0.4, 0.8]) == "drawer"
    assert jp("drawer", "revolute", 0.0, 80.0, [0.4, 0.3, 0.1], [0.58, 0.84, 0.73]) == "drawer"  # swivel drawers
    # not plates covering the object, swinging too little, or named for what they are: unchanged
    assert jp("knob", "revolute", 0.0, 270.0, [0.1, 0.03, 0.1], [0.38, 0.36, 0.46]) == "knob"
    assert jp("other", "revolute", -50.0, 0.0, [0.191, 0.054, 0.249], [0.19, 0.07, 0.26]) == "other"  # lamp pole
    assert jp("other", "revolute", 0.0, 30.0, [0.05, 0.184, 0.007], [0.08, 0.2, 0.01]) == "other"  # scissor blade
    assert jp("other", "revolute", 0.0, 120.0, [0.259, 0.449, 0.031], [0.69, 0.45, 0.27]) == "other"  # bail handle
    assert jp("handle", "revolute", 0.0, 90.0, [0.297, 0.2, 0.024], [0.3, 0.37, 0.24]) == "handle"
    assert jp("cover", "revolute", -90.0, 30.0, [0.42, 0.234, 0.416], [0.45, 0.23, 0.45]) == "cover"
    assert jp("other", "revolute", None, None, [0.43, 0.42, 0.42], [0.43, 0.42, 0.46]) == "other"  # a globe


def test_partnet_parts_ignore_fixtures(tmp_path):
    links = {
        "link_0": ["glass-17", "handle-18"],  # Microwave 7119: the door is glass + handle
        "link_1": ["tray-20", "other_leaf-22"],
        "link_2": ["handle-3"],
        "link_3": ["mirror_frame-4", "mirror_surface-5"],  # StorageFurniture 45642: sliding mirror doors
        "link_4": ["drawer_front-6", "handle-7", "other_leaf-8"],
        "link_5": ["cabinet_door_surface-9", "glass-10", "hinge-11"],
        "link_6": ["countertop-12"],  # left to the motion (joint_part): a hinged top is a lid
        "link_7": ["door_button-13"],
        "link_8": ["control_button-14"],
    }
    urdf = tmp_path / "mobility.urdf"
    urdf.write_text(
        "<robot>"
        + "".join(
            f'<link name="{link}">' + "".join(f'<visual name="{v}"/>' for v in vis) + "</link>"
            for link, vis in links.items()
        )
        + "</robot>"
    )
    assert P.link_roles(urdf) == {
        "link_0": "door",
        "link_1": "tray",
        "link_2": "handle",
        "link_3": "door",
        "link_4": "drawer",
        "link_5": "door",
        "link_7": "door",
        "link_8": "button",
    }


def test_partnet_names_count_door_joints(tmp_path, monkeypatch):
    door, drawer, button, knob = ({"role": r} for r in ("door", "drawer", "button", "knob"))
    assert P.openings([door, button, knob, {"role": "turntable"}]) == (" with one door", "单门的")  # release button
    assert P.openings([door] * 4 + [drawer] * 3) == (" with four doors and three drawers", "四门、3个抽屉的")
    assert P.openings([button, knob]) == ("", "")

    monkeypatch.setattr(P, "colour_of", lambda png: "white")
    chosen = []
    for k, joints in enumerate(([door, button], [door, {"role": "turntable"}])):
        oid = f"Articulation/partnet_microwave/{k:05d}"
        (tmp_path / "Assets/Object/RoboDojo" / oid).mkdir(parents=True)
        meta = {"physics": {"size": [0.5, 0.34, 0.23]}, "body_joints": joints}
        (tmp_path / "Assets/Object/RoboDojo" / oid / "metadata.json").write_text(json.dumps(meta))
        (tmp_path / "previews" / oid / "preview").mkdir(parents=True)
        (tmp_path / "previews" / oid / "preview/sim.png").write_bytes(b"")
        chosen.append({"id": oid, "kind": "Microwave", "src": f"Microwave/{7000 + k}"})
    manifest = {"chosen": chosen}
    fresh = P.names(tmp_path, manifest, tmp_path / "previews")
    short, longer = "50 cm wide white microwave with one door", "50 by 23 cm white microwave with one door"
    assert [fresh[c["id"]]["en"][2] for c in chosen] == [short, longer]
    assert fresh[chosen[0]["id"]]["zh"][2] == "单门的50厘米宽白色微波炉"
    # the second already has the short name and it is still right: it keeps it; the first one's stale "two doors"
    # name is replaced
    keep = {chosen[0]["id"]: {"en": ["", "", "50 cm wide white microwave with two doors"], "zh": ["", "", "双门的"]}}
    keep[chosen[1]["id"]] = fresh[chosen[0]["id"]]
    kept = P.names(tmp_path, manifest, tmp_path / "previews", keep)
    assert [kept[c["id"]]["en"][2] for c in chosen] == [longer, short]


def test_artvip_kinds_and_scope():
    assert V.kind_of("Articulated_objects/small_furniture/filing_cabinet/drawers/chest_of_drawers_1") == (
        "small_furniture/filing_cabinet/drawers",
        "filing_cabinet/drawers",
    )
    assert V.kind_of("Articulated_objects/household_items/stationery/scissors/scissors_16")[1] == "stationery/scissors"
    assert V.kind_of("Articulated_objects/small_appliances/microwave/microwave_1")[1] == "microwave"
    facts = {
        "movable": [{"name": "hinge", "body1": "/root/E_door_1", "type": "revolute"}],
        "up_axis": "Z",
        "bases": ["/root/E_body"],
        "colliders": 3,
        "world_movable": [],
    }
    assert V.decide("small_appliances/microwave", "microwave", [0.57, 0.43, 0.34], facts)[0]
    assert V.decide("small_furniture/filing_cabinet/drawers", "filing_cabinet/drawers", [0.5, 0.4, 0.8], facts) == (
        True,
        "table side, 1 joint(s): door",
    )
    assert not V.decide("small_furniture/chair", "chair", [0.5, 0.5, 0.9], facts)[0]
    assert "large furniture" in V.decide("small_furniture/shoe_cabinet", "shoe_cabinet", [1.0, 0.3, 1.2], facts)[1]
    assert not V.decide("small_appliances/oven", "oven", [0.5, 0.4, 0.3], {**facts, "movable": []})[0]
    assert (
        "wrong scale" in V.decide("household_items/stationery/stapler", "stationery/stapler", [1.7, 0.3, 0.4], facts)[1]
    )
    hands = {**facts, "movable": [{"name": "h", "body1": "/root/E_pointer_1", "type": "revolute"}]}
    assert "clock hands" in V.decide("household_items/clock", "clock", [0.28, 0.01, 0.28], hands)[1]
    hinged = {**facts, "world_movable": [{"name": "RevoluteJoint", "path": "/root/E_door/RevoluteJoint"}]}
    assert "world" in V.decide("small_appliances/oven", "oven", [0.36, 0.21, 0.2], hinged)[1]
    twins = {**facts, "movable": [{"name": "J", "body1": f"/root/E_lid_{k}", "type": "revolute"} for k in (1, 2)]}
    assert "share a prim name" in V.decide("household_items/toolbox", "toolbox", [0.4, 0.2, 0.2], twins)[1]
    twins = {**facts, "movable": [{"name": "J", "body1": f"/root/E_lid_{k}", "type": "revolute"} for k in (1, 2)]}
    assert "share a prim name" in V.decide("household_items/toolbox", "toolbox", [0.4, 0.2, 0.2], twins)[1]
    assert set(V.MASS_KG) == set(V.SIZE_M)


def source_stage(path):
    """An ArtVIP-like model: the door link comes first, the body link holds the articulation root, a world fixed joint
    and a BehaviorScript, plus a light; origin at the bottom centre."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    for name, centre, size in (
        ("E_door_1", (0.2, -0.21, 0.15), (0.02, 0.3, 0.3)),
        ("E_body_2", (0, 0, 0.15), (0.4, 0.4, 0.3)),
    ):
        link = UsdGeom.Xform.Define(stage, f"/root/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(*centre))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        cube = UsdGeom.Cube.Define(stage, f"/root/{name}/box")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    body = stage.GetPrimAtPath("/root/E_body_2")
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray).Set(Sdf.AssetPathArray(["./c.py"]))
    world = UsdPhysics.FixedJoint.Define(stage, "/root/E_body_2/FixedJoint")
    world.CreateBody1Rel().SetTargets(["/root/E_body_2"])
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/root/E_door_1/hinge")
    hinge.CreateBody0Rel().SetTargets(["/root/E_body_2"])
    hinge.CreateBody1Rel().SetTargets(["/root/E_door_1"])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLowerLimitAttr(-90.0)
    hinge.CreateUpperLimitAttr(0.0)
    UsdLux.RectLight.Define(stage, "/root/light/RectLight")
    stage.GetRootLayer().Save()


def test_layer_moves_the_root_frees_the_pose_and_centres(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    src = tmp_path / "sources/lib/item/model.usda"
    src.parent.mkdir(parents=True)
    source_stage(src)
    target = tmp_path / "Assets/Object/RoboDojo/Articulation/lib_x/00000"
    facts = A.write_layer(src, target, 10.0, tmp_path)
    text = (target / "object.usd").read_text()
    assert text.startswith("#usda") and "@../../../../../../sources/lib/item/model.usda@" in text

    stage = Usd.Stage.Open(str(target / "object.usd"))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert [c.GetName() for c in root.GetChildren()][0] == "E_body_2"
    body = stage.GetPrimAtPath("/root/E_body_2")
    assert not body.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "PhysxArticulationAPI" in body.GetPrimStack()[0].GetInfo("apiSchemas").prependedItems  # unregistered here
    assert not stage.GetPrimAtPath("/root/E_body_2/FixedJoint").IsActive()
    assert not stage.GetPrimAtPath("/root/light/RectLight").IsActive()
    assert not body.GetAttribute("omni:scripting:scripts").Get()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    assert max(abs(v) for v in rng.GetMidpoint()) < 1e-6
    assert facts["size"] == pytest.approx([0.41, 0.56, 0.3], abs=1e-6)
    masses = {p: UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/root/{p}")).GetMassAttr().Get() for p in facts["links"]}
    assert sum(masses.values()) == pytest.approx(10.0, rel=1e-4) and masses["E_body_2"] > masses["E_door_1"]
    assert facts["joints"] == [
        {
            "name": "hinge",
            "type": "revolute",
            "role": "door",
            "part": "door",
            "parent": "E_body_2",
            "child": "E_door_1",
            "axis": "Z",
            "limits_deg": [-90.0, 0.0],
        }
    ]
    meta = A.metadata("Articulation/lib_x/00000", facts, 10.0, "estimated", {"url": "u", "object": "item"})
    assert meta["physics"]["fixed_base"] and meta["base_link"] == "E_body_2"
    assert meta["active"]["place"]["up"]["contact_circle"]["center"][2] == pytest.approx(-0.15)


def test_layer_refuses_sources_outside_the_library(tmp_path):
    pytest.importorskip("pxr")
    src = tmp_path / "elsewhere/model.usda"
    src.parent.mkdir(parents=True)
    source_stage(src)
    with pytest.raises(ValueError, match="sources"):
        A.write_layer(src, tmp_path / "Assets/x/00000", 1.0, tmp_path)
