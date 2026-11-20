"""Meaningful packet, coordinate, asset and ordering regression checks (CPU only)."""

import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.robot_manager.wuji_tactile_contract import decode


def packet():
    fields = [{"name": n, "type": "i16", "offset": 2 * i, "scale": 0.01} for i, n in enumerate(("fx", "fy", "fz"))]
    aggregate = [{"name": n, "type": "f32", "offset": 4 * i} for i, n in enumerate(("fx", "fy", "fz", "temperature"))]
    fmt = {
        "v": 1,
        "encoding": "point_array",
        "point_count": 2,
        "point_stride": 6,
        "aggregate_stride": 16,
        "point_fields": fields,
        "aggregate_fields": aggregate,
        "unit": "normalized",
        "positions": [[0.01, 0, 0], [0, 0.01, 0]],
        "point_rpy": [[0, 0, 0], [0, 0, 0]],
        "base_xyz": [0.1, 0, 0],
        "base_rpy": [0, 0, np.pi / 2],
        "aggregate_xyz": [0.01, 0, 0],
        "aggregate_rpy": [0, 0, 0],
    }
    data = struct.pack("<6h4f", 10, 20, 30, 40, 50, 60, 1, 2, 3, 25)
    return fmt, data


def test_normalized_points_are_not_newtons_and_wrench_has_correct_frame():
    fmt, data = packet()
    result = decode(fmt, data)
    assert result["point_unit"] == "normalized"
    assert result["point_values"][0]["fz"] == pytest.approx(0.3)
    np.testing.assert_allclose(result["positions_link_m"][0], [0.1, 0.01, 0], atol=1e-12)
    np.testing.assert_allclose(result["aggregate_force_link_N"], [-2, 1, 3], atol=1e-12)
    np.testing.assert_allclose(result["aggregate_moment_link_Nm"], np.cross([0.1, 0.01, 0], [-2, 1, 3]), atol=1e-12)


def test_legacy_metadata_does_not_invent_positions_or_moment():
    fmt, data = packet()
    for key in ("point_rpy", "base_xyz", "base_rpy", "aggregate_xyz"):
        fmt.pop(key)
    result = decode(fmt, data)
    assert result["positions_link_m"] is None
    assert result["aggregate_moment_link_Nm"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "truncated",
        "partial_mount",
        "mixed_units",
        "unknown_type",
        "bad_offset",
        "bad_count",
        "nonfinite",
        "bad_version",
    ],
)
def test_corrupt_or_ambiguous_metadata_is_rejected(fault):
    fmt, data = packet()
    if fault == "truncated":
        data = data[:-1]
    elif fault == "partial_mount":
        fmt.pop("base_rpy")
    elif fault == "mixed_units":
        fmt["point_fields"][0]["unit"] = "N"
    elif fault == "unknown_type":
        fmt["point_fields"][0]["type"] = "i24"
    elif fault == "bad_offset":
        fmt["point_fields"][0]["offset"] = 6
    elif fault == "bad_count":
        fmt["point_count"] = 3
    elif fault == "nonfinite":
        data = data[:-4] + struct.pack("<f", np.nan)
    elif fault == "bad_version":
        fmt["v"] = 2
    with pytest.raises((ValueError, struct.error)):
        decode(fmt, data)


@pytest.mark.parametrize("model,dim", [("g1d_wuji_hand2_1", 20), ("g1d_wuji_hand2_2", 20), ("g1d_sharpa_wave", 22)])
def test_asset_controller_contract_preserves_each_hand_axis(model, dim):
    directory = ROOT / "Assets/Robots" / model
    contract = json.loads((directory / "model_contract.json").read_text())
    root = ET.parse(directory / "robot.urdf").getroot()
    names = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    assert contract["action_dimension"] == 14 + 2 * dim
    assert not any("dex1" in n for n in names)
    for side, hand in contract["hands"].items():
        config = json.loads((directory / (side + "_robot_config.yml")).read_text())
        assert config["ee_type"] == "hand"
        assert len(hand["joints"]) == dim
        assert config["gripper_joints_name"] == hand["joints"]
        assert set(hand["joints"]) <= names
        assert hand["mount_xyz_m"][1] == (0.003 if side == "left" else -0.003)
    for mesh in root.iter("mesh"):
        assert Path(mesh.get("filename")).stem[0].isalpha()
        assert (directory / mesh.get("filename")).is_file()


def test_revisions_keep_distinct_tactile_capabilities():
    a = json.loads((ROOT / "Assets/Robots/g1d_wuji_hand2_1/model_contract.json").read_text())
    b = json.loads((ROOT / "Assets/Robots/g1d_wuji_hand2_2/model_contract.json").read_text())
    assert all(not h["frames"] for h in a["tactile"].values())
    assert all(len(h["frames"]) == 5 for h in b["tactile"].values())
    for side in ("left", "right"):
        assert a["hands"][side]["joints"] == b["hands"][side]["joints"]
        assert a["hands"][side]["limits"] == b["hands"][side]["limits"]


@pytest.mark.parametrize("model", ["g1d_wuji_hand2_1", "g1d_wuji_hand2_2", "g1d_sharpa_wave"])
def test_complete_robot_thumb_side_is_up_and_flexion_is_inward(model):
    from audit_hand_placement import orientation

    root = ET.parse(ROOT / "Assets/Robots" / model / "robot.urdf").getroot()
    for side in ("left", "right"):
        assert orientation(root, model, side)["passed"]


def test_orientation_check_rejects_inverted_wuji_mount():
    from audit_hand_placement import orientation

    model = "g1d_wuji_hand2_2"
    root = ET.parse(ROOT / "Assets/Robots" / model / "robot.urdf").getroot()
    for side in ("left", "right"):
        root.find(f"joint[@name='{side}_dexhand_mount']/origin").set("rpy", f"{np.pi} {np.pi / 2} 0")
        assert not orientation(root, model, side)["passed"]


def test_probe_closed_poles_and_scale():
    from probe_geometry import make_probe

    sphere = make_probe("sphere")
    assert sphere.is_watertight
    np.testing.assert_allclose(sphere.bounds, [[-0.002] * 3, [0.002] * 3], atol=1e-9)
    origins = np.array([[0.0, 0.0, 0.0018], [0.0018, 0.0, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    hits, ray, _ = sphere.ray.intersects_location(origins, directions, multiple_hits=False)
    assert len(ray) == 2
    np.testing.assert_allclose(np.linalg.norm(hits - origins[ray], axis=1), 0.0002, atol=1e-6)
    for shape in ("flat", "cylinder"):
        assert make_probe(shape).is_watertight
