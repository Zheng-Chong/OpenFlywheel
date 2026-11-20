"""A narrow compatibility wrapper around the pinned official Sharpa implementation.

The sensor's ray casting, CPD filter, quantizer and blur remain vendor code.
The duplicate base constructor is avoided. Rays are evaluated in millimetres
inside Warp and converted back to metres at the boundary, to avoid the observed
small-mesh intersection failure. The source CPD and quantizer stay unchanged.
"""

from contextvars import ContextVar
import importlib.util
from pathlib import Path
import sys


def load_vendor(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(
        "sharpa_vendor", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sharpa_vendor"] = module
    spec.loader.exec_module(module)
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData
    from isaaclab.utils.warp import raycast_dynamic_meshes as original_raycast
    import numpy as np
    from sharpa_vendor.tacmap_sensor import SharpaTacmap, SharpaTacmapCfg
    import sharpa_vendor.tacmap_sensor.sharpa_tacmap_vbts as vendor_module
    import torch
    import warp as wp

    scaled_meshes = {}
    scaled_ids = {}
    capture = ContextVar("tacmap_depth_capture", default=None)

    def raycast_dynamic_meshes(starts, directions, mesh_ids_wp, **kwargs):
        ids = mesh_ids_wp.numpy()
        key = tuple(int(v) for v in ids.ravel())
        if key not in scaled_ids:
            originals = {int(mesh.id): mesh for mesh in MultiMeshRayCaster.meshes.values()}
            new_ids = np.empty_like(ids)
            for index in np.ndindex(ids.shape):
                old = int(ids[index])
                if old not in scaled_meshes:
                    mesh = originals[old]
                    scaled_meshes[old] = wp.Mesh(
                        points=wp.array(mesh.points.numpy() * 1000.0, dtype=wp.vec3, device=mesh_ids_wp.device),
                        indices=wp.array(mesh.indices.numpy(), dtype=wp.int32, device=mesh_ids_wp.device),
                    )
                new_ids[index] = scaled_meshes[old].id
            scaled_ids[key] = wp.array2d(new_ids, dtype=wp.uint64, device=mesh_ids_wp.device)
        kwargs = dict(kwargs)
        kwargs["max_dist"] = kwargs.get("max_dist", 1e6) * 1000.0
        if kwargs.get("mesh_positions_w") is not None:
            kwargs["mesh_positions_w"] = kwargs["mesh_positions_w"] * 1000.0
        hits, depth, normals, faces, meshes = original_raycast(
            starts * 1000.0, directions, mesh_ids_wp=scaled_ids[key], **kwargs
        )
        # The upstream multi-mesh kernel atomically minimizes distance, but its
        # separate hit-position write can be overwritten by a farther mesh.
        # Reconstruct display hits from the winning distance. The pinned
        # vendor CPD itself already starts its reverse ray from distance.
        if depth is None:
            raise ValueError("TacMap requires ray distances")
        depth = depth / 1000.0
        hits = torch.where(
            torch.isfinite(depth).unsqueeze(-1),
            starts + directions * depth.unsqueeze(-1),
            torch.full_like(starts, float("inf")),
        )
        result = (hits, depth, normals, faces, meshes)
        records = capture.get()
        if records is not None:
            records.append(None if result[1] is None else result[1].clone())
        return result

    vendor_module.raycast_dynamic_meshes = raycast_dynamic_meshes

    class CompatibleTacmap(SharpaTacmap):
        def __init__(self, cfg):
            if cfg.resolution_step != 1:
                raise ValueError("Only the verified 240x240 vendor resolution is enabled")
            if cfg.data_types != ["distance_along_normal"]:
                raise ValueError("Unsupported tactile output")
            MultiMeshRayCaster.__init__(self, cfg)
            self._data = MultiMeshRayCasterData()

        def metric_depth(self):
            return self._metric_depth_m.clone()

        def _update_buffers_impl(self, env_ids):
            records = []
            token = capture.set(records)
            try:
                super()._update_buffers_impl(env_ids)
                if len(records) != 2:
                    raise RuntimeError("Pinned vendor CPD call sequence changed")
                first, second = records
                # Use the source ray-distance buffers, just as its quantizer
                # does. Multi-mesh hit-position writes have a documented GPU
                # race, so reconstructing distance from ray_hits is unreliable.
                self._metric_depth_m = torch.where(
                    torch.isfinite(first) & torch.isfinite(second), first, torch.zeros_like(first)
                )
            finally:
                capture.reset(token)

    return CompatibleTacmap, SharpaTacmapCfg
