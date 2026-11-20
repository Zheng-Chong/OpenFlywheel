"""Round cups (straight or tapered) as bean containers, numpy only: the cavity test and how many PBD red beans (or
rigid grains of a kind) fill one. Shared by the bean counter of the datagen plugins
(scripts/g1d/collect_office_beans.py) and the task checks (env/reward_manager func_parser.is_fluid_fill_in_range), so
data acceptance and task success count the same beans.

A cup is dict(height, radius (inner, at the rim), bottom (thickness)[, radius_bottom (inner, at the bottom) for a
tapered cup]), in the cup's frame: origin at its bbox centre, Z up (layout entries carry it as "cavity").
"""

import numpy as np

BEAN_D, PACKING = 0.0056, 0.6  # the PBD beans (m) and how densely they settle in a cup
PBD_GRAIN = dict(volume=np.pi / 6 * BEAN_D**3, packing=PACKING)


def rotation(quaternion):
    """3x3 rotation of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quaternion, float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def in_cavity(points, position, quaternion, cup):
    """Bean centres inside the cup at `position` / `quaternion` (w, x, y, z): above the inner bottom, below the rim,
    within the inner wall."""
    local = (np.asarray(points, float) - np.asarray(position, float)) @ rotation(quaternion)
    z = local[:, 2] + cup["height"] / 2
    low = cup.get("radius_bottom", cup["radius"])
    radius = low + (cup["radius"] - low) * z / cup["height"]
    return (z > cup["bottom"]) & (z < cup["height"]) & (np.linalg.norm(local[:, :2], axis=1) < radius)


def capacity(cup, grain=None):
    """Beans the cup holds when full; `grain` dict(volume (m^3), packing), default the PBD beans (a granular
    object's .grain for rigid ones)."""
    h = cup["height"] - cup["bottom"]
    r1, r0 = cup["radius"], cup.get("radius_bottom", cup["radius"])
    volume = np.pi * h / 3 * (r1**2 + r1 * r0 + r0**2)
    grain = grain or PBD_GRAIN
    return volume * grain["packing"] / grain["volume"]


def fill_ok(target, source, total, cup, fill, spill_max, grain=None):
    """`target` of `total` beans in the cup fill it to a share of its capacity within `fill` (lo, hi), with at most
    `spill_max` of the beans out of the source (all but `source`) in neither."""
    cap, poured, outside = capacity(cup, grain), total - source, total - source - target
    return bool(fill[0] * cap <= target <= fill[1] * cap and outside <= spill_max * max(poured, 1))
