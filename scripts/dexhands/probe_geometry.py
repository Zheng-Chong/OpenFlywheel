"""Closed millimetre probe geometry without size-dependent generation tolerances."""

import numpy as np
import trimesh


def make_probe(kind, radius=0.002, height=0.002, size=(0.004, 0.004, 0.002)):
    if kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.vertices *= radius
    elif kind == "flat":
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        mesh.vertices *= np.asarray(size)
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
        mesh.vertices *= np.array([radius, radius, height])
    else:
        raise ValueError(kind)
    if not mesh.is_watertight:
        raise ValueError("Probe mesh must be closed")
    return mesh
