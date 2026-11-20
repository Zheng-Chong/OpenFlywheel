"""Checks of the atomic block tasks (scripts/objects/block_atomic.py), numpy only: func_parser's is_block_* read the
poses and call these, the collection plugins call them on recorded poses with tighter tolerances.

Poses are a centre (x, y, z) and a 3x3 rotation; sizes are a block's model extents (X, Y, Z: block_patterns.SIZES).
"""

import numpy as np

from env.reward_manager.pattern_match import wrap_deg, yaw_of


def tilt_deg(rot):
    """How far a block's local Z is off vertical."""
    return float(np.degrees(np.arccos(np.clip(np.asarray(rot, float)[2, 2], -1.0, 1.0))))


def placed_on(pos, rot, size, mark, table_z, radius, tilt=10.0):
    """A block (centre `pos`) resting upright on the table or a flat mark there, its centre within `radius` of `mark`
    (x, y) horizontally."""
    pos = np.asarray(pos, float)
    resting = abs(pos[2] - (table_z + size[2] / 2)) < 0.006  # a thin mark under it, at most a few mm
    return bool(np.linalg.norm(pos[:2] - np.asarray(mark, float)[:2]) < radius and resting and tilt_deg(rot) < tilt)


def stacked_on(top_pos, top_rot, top_size, base_pos, base_size, xy_tol, z_tol, tilt=10.0):
    """Block `top` resting upright on block `base`: centre over the base's within `xy_tol`, its bottom on the base's
    top within `z_tol`."""
    top_pos, base_pos = np.asarray(top_pos, float), np.asarray(base_pos, float)
    gap = (top_pos[2] - top_size[2] / 2) - (base_pos[2] + base_size[2] / 2)
    over = np.linalg.norm(top_pos[:2] - base_pos[:2])
    return bool(over < xy_tol and abs(gap) < z_tol and tilt_deg(top_rot) < tilt)


def lifted(pos, size, table_z, height):
    """A block's bottom at least `height` above the table."""
    return bool(np.asarray(pos, float)[2] - size[2] / 2 - table_z > height)


def yaw_aligned(rot, axis_deg, period, tol):
    """A block's local X (its long side) within `tol` deg of the world yaw `axis_deg`, mod `period`."""
    return bool(abs(wrap_deg(np.degrees(yaw_of(np.asarray(rot, float))) - axis_deg, period)) < tol)


def in_row(positions, rots, gap_range, y_tol, tilt=10.0):
    """Blocks standing in a row along world X in the given order (left to right, -X first): x increasing with each
    neighbour's centre `gap_range` (m, min / max) apart, all within `y_tol` of their mean y, upright."""
    p = np.asarray(positions, float)
    steps = np.diff(p[:, 0])
    return bool(
        np.all(steps > gap_range[0])
        and np.all(steps < gap_range[1])
        and np.all(np.abs(p[:, 1] - p[:, 1].mean()) < y_tol)
        and all(tilt_deg(r) < tilt for r in rots)
    )
