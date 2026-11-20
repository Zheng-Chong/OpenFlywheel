"""Offline RH56DFTP action conversion and the two attached tactile protocols.

No socket, serial, DDS, camera or hardware SDK is imported. Raw tactile counts
and undocumented capacitive force units are never presented as Newtons.
"""

import math
import struct

import numpy as np

ORDER = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
HAND_TARGET_LEAD_RAD = 0.05  # Numerical jam protection, not measured servo behavior.


def angle_set_to_rad(values, hand, *, previous_rad=None, thumb_rotation_increases_q=True):
    """Simulation affine endpoint adapter, not a hardware zero calibration.

    ANGLE_SET 1000 opens the first five axes. Thumb rotation polarity must be
    explicitly confirmed for real hardware; the simulation convention is +q.
    POS_SET (0..2000) has opposite polarity and is deliberately not accepted.
    """
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (6,)
        or not np.isfinite(values).all()
        or np.any((values != -1) & ((values < 0) | (values > 1000)))
    ):
        raise ValueError("Six ANGLE_SET entries, each -1 or 0..1000, required")
    if np.any(values == -1) and previous_rad is None:
        raise ValueError("Hold requires explicit previous active state")
    limits = np.array([[hand["effective_active_limits"][n][v] for v in ("lower", "upper")] for n in hand["joints"]])
    u = values / 1000
    u[:5] = 1 - u[:5]
    if not thumb_rotation_increases_q:
        u[5] = 1 - u[5]
    result = limits[:, 0] + u * (limits[:, 1] - limits[:, 0])
    if previous_rad is not None:
        previous = np.asarray(previous_rad, dtype=float)
        if (
            previous.shape != (6,)
            or not np.isfinite(previous).all()
            or np.any(previous < limits[:, 0])
            or np.any(previous > limits[:, 1])
        ):
            raise ValueError("Previous state outside feasible active limits")
        result[values == -1] = previous[values == -1]
    return result


def expand_mimic(active_rad, hand):
    q = dict(zip(hand["joints"], map(float, active_rad), strict=True))
    for n, lim in hand["effective_active_limits"].items():
        if not math.isfinite(q[n]) or not lim["lower"] <= q[n] <= lim["upper"]:
            raise ValueError("Active joint outside feasible coupled range: " + n)
    q.update({n: q[m["joint"]] * m["multiplier"] + m["offset"] for n, m in hand["mimic"].items()})
    return q


def piezo_layout():
    result, address = [], 3000
    for finger in ORDER[:4] + ("thumb",):
        shapes = [("end", 3, 3), ("tip", 12, 8)]
        shapes += [("middle", 3, 3), ("pad", 12, 8)] if finger == "thumb" else [("pad", 10, 8)]
        for patch, rows, cols in shapes:
            size = rows * cols * 2
            result.append({"name": finger + "_" + patch, "rows": rows, "cols": cols, "address": address, "bytes": size})
            address += size
    result.append({"name": "palm", "rows": 8, "cols": 14, "address": address, "bytes": 224})
    assert address + 224 == 5124
    return result


def decode_piezoresistive(payload, *, timestamp_s):
    if len(payload) != 2124 or not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError("Require complete 3000..5123 register block and valid timestamp")
    raw = np.frombuffer(payload, dtype="<u2")
    if np.any(raw > 4096):
        raise ValueError("Piezoresistive raw count exceeds documented 4096")
    matrices = {}
    for p in piezo_layout():
        start = (p["address"] - 3000) // 2
        v = raw[start : start + p["bytes"] // 2]
        # Palm transmission is bottom-to-top within each column.
        matrix = v.reshape(p["cols"], p["rows"]).T[::-1] if p["name"] == "palm" else v.reshape(p["rows"], p["cols"])
        matrices[p["name"]] = matrix.copy()
    return {
        "schema": "inspire.piezoresistive.v1",
        "timestamp_s": timestamp_s,
        "unit": "ADC_count",
        "matrices": matrices,
        "taxels": 1062,
        "force_N": None,
        "hardware_taxel_geometry": None,
        "calibration": "raw_only",
    }


def decode_capacitive(payload, *, timestamp_s):
    if len(payload) != 58 or not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError("Require one complete 58-byte module and valid timestamp")
    raw = struct.unpack_from("<8I", payload, 0)
    values = struct.unpack_from("<4f", payload, 32)
    if not all(map(math.isfinite, values)):
        raise ValueError("Non-finite capacitive derived value")
    return {
        "schema": "inspire.capacitive.v1",
        "timestamp_s": timestamp_s,
        "raw_channels": raw,
        **dict(zip(("normal", "normal_change", "shear", "shear_change"), values)),
        "shear_direction_raw": struct.unpack_from("<H", payload, 48)[0],
        "proximity_change_raw": struct.unpack_from("<I", payload, 50)[0],
        "checksum_raw": struct.unpack_from("<H", payload, 54)[0],
        "reserved": payload[56:].hex(),
        "checksum_verified": False,
        "checksum_note": "inner checksum algorithm absent in attached manual; outer transport validation is caller responsibility",
        "derived_unit": "manufacturer_unspecified",
        "force_N": None,
    }


def synchronize(robot_time_s, tactile_time_s, *, max_skew_s=1 / 30):
    """Explicit timestamp association, rejecting stale/out-of-order records."""
    robot = np.asarray(robot_time_s, dtype=float)
    touch = np.asarray(tactile_time_s, dtype=float)
    if (
        robot.ndim != 1
        or touch.ndim != 1
        or not robot.size
        or not touch.size
        or not np.isfinite(robot).all()
        or not np.isfinite(touch).all()
        or np.any(np.diff(robot) <= 0)
        or np.any(np.diff(touch) <= 0)
        or not math.isfinite(max_skew_s)
        or max_skew_s < 0
    ):
        raise ValueError("Finite strictly increasing timestamps and a nonnegative skew bound required")
    right = np.searchsorted(robot, touch).clip(0, len(robot) - 1)
    left = (right - 1).clip(0)
    ids = np.where(abs(robot[left] - touch) <= abs(robot[right] - touch), left, right)
    skew = abs(robot[ids] - touch)
    if np.any(skew > max_skew_s):
        raise ValueError("Tactile record is stale relative to robot state")
    return ids, skew
