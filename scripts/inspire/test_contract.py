"""Independent boundary, packet orientation, coupled range and time checks."""

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from env.robot_manager.inspire_contract import (
    angle_set_to_rad,
    decode_capacitive,
    decode_piezoresistive,
    expand_mimic,
    piezo_layout,
    synchronize,
)

ROOT = Path(__file__).resolve().parents[2]
C = json.loads((ROOT / "Assets/Robots/g1d_inspire_rh56dftp/model_contract.json").read_text())


@pytest.mark.parametrize("side", ["left", "right"])
def test_coupling_endpoints_and_hold(side):
    h = C["hands"][side]
    assert len(h["joints"]) == 6 and len(h["mimic"]) == 6 and len(h["all_joints"]) == 12
    for values in ([1000] * 5 + [0], [500] * 6, [0] * 5 + [1000]):
        q = angle_set_to_rad(values, h)
        allq = expand_mimic(q, h)
        assert len(allq) == 12
        for n, v in allq.items():
            assert h["limits"][n]["lower"] - 1e-9 <= v <= h["limits"][n]["upper"] + 1e-9
        np.testing.assert_array_equal(angle_set_to_rad([-1] * 6, h, previous_rad=q), q)
    with pytest.raises(ValueError):
        angle_set_to_rad([-1] * 6, h)
    with pytest.raises(ValueError):
        angle_set_to_rad([2000] * 6, h)


def test_full_matrix_boundaries_and_palm_wire_orientation():
    raw = np.arange(1062, dtype="<u2")
    out = decode_piezoresistive(raw.tobytes(), timestamp_s=1.0)
    assert len(out["matrices"]) == 17 and out["force_N"] is None
    assert out["matrices"]["index_tip"].shape == (12, 8)
    palm = out["matrices"]["palm"]
    assert palm[7, 0] == 950 and palm[0, 0] == 957 and palm[7, 1] == 958
    assert palm[0, 13] == 1061
    assert sum(p["bytes"] for p in piezo_layout()) == 2124


@pytest.mark.parametrize("data", [b"", b"\0" * 2123, b"\0" * 2125, b"\xff\xff" * 1062])
def test_piezo_rejects_corrupt_block(data):
    with pytest.raises(ValueError):
        decode_piezoresistive(data, timestamp_s=1.0)


def test_capacitive_offsets_and_units():
    data = struct.pack("<8I4fHIHH", *range(8), 1.25, -0.25, 2.5, -0.5, 270, 42, 0x1234, 0)
    out = decode_capacitive(data, timestamp_s=1)
    assert out["normal"] == 1.25 and out["shear"] == 2.5 and out["proximity_change_raw"] == 42
    assert out["checksum_raw"] == 0x1234 and not out["checksum_verified"] and out["force_N"] is None
    for bad in (data[:-1], data[:32] + struct.pack("<f", float("nan")) + data[36:]):
        with pytest.raises(ValueError):
            decode_capacitive(bad, timestamp_s=1)


def test_sync_rejects_stale_and_reordered_frames():
    ids, skew = synchronize([0, 0.1, 0.2], [0.002, 0.101, 0.199], max_skew_s=0.01)
    np.testing.assert_array_equal(ids, [0, 1, 2])
    assert skew.max() < 0.003
    for touch in ([0, -0.1], [0.5], [0.1, 0.1]):
        with pytest.raises(ValueError):
            synchronize([0, 0.1, 0.2], touch, max_skew_s=0.01)
