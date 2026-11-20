"""Does a set of objects form a given pattern? numpy only (func_parser.is_pattern_built, unit-tested on its own).

The pattern gives every object's centre and yaw in a pattern frame. The frame is taken from the anchor object: its
centre fixes the offset, its yaw the turn (an anchor that repeats every 180 deg leaves two turns, both tried). The
pattern may stand anywhere on the table, turned any way. Objects that look alike (`groups`: the same id) may take
each other's places: every assignment within a group is tried.
"""

import itertools

import numpy as np


def yaw_of(rot):
    """Yaw (rad) of a 3x3 rotation's local X axis about world Z."""
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def wrap_deg(a, period):
    """Angle `a` (deg) folded into [-period/2, period/2)."""
    return (a + period / 2) % period - period / 2


def assignments(groups):
    """Object orders (order[i]: the object put at target i) that only swap objects of the same group."""
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    slots = list(members.values())
    for perms in itertools.product(*(itertools.permutations(m) for m in slots)):
        order = list(range(len(groups)))
        for slot, perm in zip(slots, perms):
            for i, j in zip(slot, perm):
                order[i] = j
        yield order


def pattern_error(positions, rotations, target, yaw_period, anchor, groups=None):
    """Worst deviation of the objects from the pattern, over the anchor's possible turns.

    positions (n, 3) and rotations (n, 3, 3): world poses; target (n, 4): centre x, y, z and yaw (deg) per object in
    the pattern frame; yaw_period (n): deg after which each object looks the same (0 = any yaw); groups (n): objects
    with the same id are interchangeable (None: none are). Returns (xy m, z m, tilt deg, yaw deg), each the largest
    over the objects, for the turn and assignment with the smallest xy deviation.
    """
    positions, target = np.asarray(positions, float), np.asarray(target, float)
    rotations = np.asarray(rotations, float)
    if groups is not None:
        errors = (pattern_error(positions[o], rotations[o], target, yaw_period, anchor) for o in assignments(groups))
        return min(errors, key=lambda e: e[0])
    tilt = max(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))) for r in rotations)  # local Z off vertical
    yaws = np.degrees([yaw_of(r) for r in rotations])
    rel = target[:, :3] - target[anchor, :3]
    best = None
    turns = (0.0,) if yaw_period[anchor] == 0 else np.arange(0.0, 360.0, yaw_period[anchor])
    for turn in turns:
        phi = np.radians(yaws[anchor] - target[anchor, 3] + turn)
        c, s = np.cos(phi), np.sin(phi)
        want = positions[anchor] + np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]]
        xy = float(np.linalg.norm((positions - want)[:, :2], axis=1).max())
        z = float(np.abs((positions - want)[:, 2]).max())
        yaw = max(
            (abs(wrap_deg(yaws[i] - target[i, 3] - np.degrees(phi), p)) for i, p in enumerate(yaw_period) if p),
            default=0.0,
        )
        if best is None or xy < best[0]:
            best = (xy, z, float(tilt), float(yaw))
    return best


def pattern_built(positions, rotations, target, yaw_period, anchor, xy_tol, z_tol, tilt_tol, yaw_tol, groups=None):
    xy, z, tilt, yaw = pattern_error(positions, rotations, target, yaw_period, anchor, groups)
    return xy < xy_tol and z < z_tol and tilt < tilt_tol and yaw < yaw_tol
