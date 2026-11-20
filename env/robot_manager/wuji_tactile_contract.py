"""Offline Wuji Hand2 fingertip metadata decoder. Does not import a hardware SDK.

The official typed example specifies self-describing layouts and mount poses.
Per-point normalized values must remain normalized; aggregate values are N.
No real device metadata or SN calibration is fabricated by this module.
"""

import struct

import numpy as np

TYPES = {"i8": "b", "u8": "B", "i16": "h", "u16": "H", "i32": "i", "u32": "I", "f32": "f"}
MOUNT_KEYS = ("point_rpy", "base_xyz", "base_rpy")


def rotation(rpy):
    a = np.asarray(rpy, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("Invalid RPY")
    r, p, y = a
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def decode(fmt, payload):
    if fmt.get("v") != 1 or fmt.get("encoding") != "point_array":
        raise ValueError("Unsupported metadata version/encoding")
    count = fmt["point_count"]
    stride = fmt["point_stride"]
    aggregate_stride = fmt["aggregate_stride"]
    if not isinstance(count, int) or not 0 < count < 10000 or stride <= 0 or aggregate_stride <= 0:
        raise ValueError("Invalid metadata dimensions")
    if len(payload) != count * stride + aggregate_stride:
        raise ValueError("Frame length does not match metadata")

    def read(fields, offset, size):
        result = {}
        for f in fields:
            code = TYPES.get(f["type"])
            if code is None:
                raise ValueError("Unknown field type")
            end = f["offset"] + struct.calcsize("<" + code)
            if f["offset"] < 0 or end > size or f["name"] in result:
                raise ValueError("Invalid field layout")
            value = struct.unpack_from("<" + code, payload, offset + f["offset"])[0] * f.get("scale", 1.0)
            if not np.isfinite(value):
                raise ValueError("Non-finite sensor value")
            result[f["name"]] = value
        return result

    points = [read(fmt["point_fields"], i * stride, stride) for i in range(count)]
    aggregate = read(fmt["aggregate_fields"], count * stride, aggregate_stride)
    if not {"fx", "fy", "fz", "temperature"} <= aggregate.keys():
        raise ValueError("Incomplete aggregate force/temperature fields")
    units = {f.get("unit", fmt.get("unit")) for f in fmt["point_fields"] if f["name"] in ("fx", "fy", "fz")}
    if len(units) != 1 or next(iter(units)) not in ("N", "normalized"):
        raise ValueError("Unknown or mixed point force units")
    present = [k for k in MOUNT_KEYS if k in fmt]
    result = {
        "point_values": points,
        "point_unit": next(iter(units)),
        "aggregate_N": np.array([aggregate[k] for k in ("fx", "fy", "fz")]),
        "temperature_C": aggregate["temperature"],
        "positions_link_m": None,
        "point_axes_to_link": None,
        "aggregate_force_link_N": None,
        "aggregate_moment_link_Nm": None,
        "calibration_status": "requires_sensor_metadata_and_unloaded_baseline",
    }
    if not present:
        return result
    if len(present) != len(MOUNT_KEYS):
        raise ValueError("Mount metadata is partially present")
    positions = np.asarray(fmt["positions"], dtype=float)
    rpy = np.asarray(fmt["point_rpy"], dtype=float)
    base = np.asarray(fmt["base_xyz"], dtype=float)
    if (
        positions.shape != (count, 3)
        or rpy.shape != (count, 3)
        or base.shape != (3,)
        or not np.isfinite(np.r_[positions.ravel(), rpy.ravel(), base]).all()
    ):
        raise ValueError("Invalid sensor placement arrays")
    R = rotation(fmt["base_rpy"])
    result["positions_link_m"] = positions @ R.T + base
    result["point_axes_to_link"] = np.stack([R @ rotation(v) for v in rpy])
    result["aggregate_force_link_N"] = R @ rotation(fmt.get("aggregate_rpy", [0, 0, 0])) @ result["aggregate_N"]
    if "aggregate_xyz" in fmt:
        at = np.asarray(fmt["aggregate_xyz"], dtype=float)
        if at.shape != (3,) or not np.isfinite(at).all():
            raise ValueError("Invalid aggregate application point")
        result["aggregate_moment_link_Nm"] = np.cross(R @ at + base, result["aggregate_force_link_N"])
    return result
