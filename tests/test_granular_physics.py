from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import env.scene_manager.objects.granular as granular  # noqa: E402
from env.scene_manager.objects.granular import (  # noqa: E402
    BODY_SCHEMAS,
    CAPSULE_COLLIDER_SCHEMAS,
    HULL_SCHEMAS,
    ITERATIONS,
    MAX_DEPENETRATION_VELOCITY,
    REST_OFFSET,
    VELOCITY_ITERATIONS,
    GranularObject,
)

try:
    from pxr import Usd

    HAS_USD = True
except ImportError:  # pragma: no cover - fallback for bare CPU CI without usd-core
    Usd = None
    HAS_USD = False


class _StubPath(str):
    def AppendChild(self, name: str) -> _StubPath:
        return _StubPath(f"{self.rstrip('/')}/{name}")

    def AppendProperty(self, name: str) -> _StubPath:
        return _StubPath(f"{self}.{name}")


class _StubListProxy(list):
    def Prepend(self, item) -> None:
        self.insert(0, _StubPath(item))


class _StubAttrSpec:
    def __init__(self, prim_spec: _StubPrimSpec, name: str, type_name: str, variability: str):
        self.name = name
        self.typeName = type_name
        self.variability = variability
        self.default = None
        self.info: dict[str, object] = {}
        self.connectionPathList = _StubListProxy()
        self.path = prim_spec.path.AppendProperty(name)
        prim_spec.attributes[name] = self

    def SetInfo(self, key: str, value: object) -> None:
        self.info[key] = value


class _StubRelSpec:
    def __init__(self, prim_spec: _StubPrimSpec, name: str, custom: bool = False):
        self.name = name
        self.custom = custom
        self.targetPathList = _StubListProxy()
        prim_spec.relationships[name] = self


class _StubPrimSpec:
    def __init__(self, path: _StubPath):
        self.path = _StubPath(path)
        self.specifier = None
        self.typeName = ""
        self.info: dict[str, object] = {}
        self.attributes: dict[str, _StubAttrSpec] = {}
        self.relationships: dict[str, _StubRelSpec] = {}

    def SetInfo(self, key: str, value: object) -> None:
        self.info[key] = value


class _StubLayer:
    def __init__(self):
        self.prims: dict[str, _StubPrimSpec] = {}

    def GetPrimAtPath(self, path: str) -> _StubPrimSpec | None:
        return self.prims.get(str(path))


class _StubStage:
    def __init__(self):
        self._layer = _StubLayer()

    def GetEditTarget(self):
        return types.SimpleNamespace(GetLayer=lambda: self._layer)

    def DefinePrim(self, path: str, type_name: str):
        spec = self._layer.prims.setdefault(str(path), _StubPrimSpec(_StubPath(path)))
        spec.specifier = "def"
        spec.typeName = type_name
        return spec


def _build_stub_pxr():
    class _Vec3:
        def __init__(self, x: float, y: float, z: float):
            self._v = (float(x), float(y), float(z))

        def __iter__(self):
            return iter(self._v)

        def __getitem__(self, idx: int) -> float:
            return self._v[idx]

        def __len__(self) -> int:
            return 3

    class _Quatf:
        def __init__(self, w: float, x: float, y: float, z: float):
            self._w = float(w)
            self._im = _Vec3(x, y, z)

        def GetReal(self) -> float:
            return self._w

        def GetImaginary(self) -> _Vec3:
            return self._im

    gf = types.SimpleNamespace(Vec3f=_Vec3, Vec3d=_Vec3, Quatf=_Quatf)
    value_types = types.SimpleNamespace(
        Bool="bool",
        Float="float",
        Double="double",
        Double3="double3",
        Float3="float3",
        Float3Array="float3[]",
        Color3f="color3f",
        Quatf="quatf",
        Int="int",
        IntArray="int[]",
        Token="token",
        TokenArray="token[]",
        Point3fArray="point3f[]",
        Normal3fArray="normal3f[]",
    )

    def _create_prim(layer: _StubLayer, path: _StubPath) -> _StubPrimSpec:
        spec = layer.prims.get(str(path))
        if spec is None:
            spec = _StubPrimSpec(_StubPath(path))
            layer.prims[str(path)] = spec
        return spec

    sdf = types.SimpleNamespace(
        Path=_StubPath,
        ValueTypeNames=value_types,
        SpecifierDef="def",
        VariabilityUniform="uniform",
        VariabilityVarying="varying",
        TokenListOp=types.SimpleNamespace(CreateExplicit=lambda items: list(items)),
        CreatePrimInLayer=_create_prim,
        AttributeSpec=_StubAttrSpec,
        RelationshipSpec=_StubRelSpec,
        ChangeBlock=nullcontext,
    )

    class _Vec3fArray(list):
        @classmethod
        def FromNumpy(cls, arr: np.ndarray):
            return cls(_Vec3(*row) for row in np.asarray(arr, dtype=float))

    vt = types.SimpleNamespace(
        TokenArray=lambda items: list(items),
        IntArray=lambda items: [int(v) for v in items],
        Vec3fArray=_Vec3fArray,
    )
    return gf, sdf, vt


def _make_stage(monkeypatch: pytest.MonkeyPatch):
    if HAS_USD:
        return Usd.Stage.CreateInMemory()
    gf, sdf, vt = _build_stub_pxr()
    monkeypatch.setattr(granular, "Gf", gf)
    monkeypatch.setattr(granular, "Sdf", sdf)
    monkeypatch.setattr(granular, "Vt", vt)
    return _StubStage()


def _prim_spec(stage, path: str):
    return stage.GetEditTarget().GetLayer().GetPrimAtPath(path)


def _schemas(spec) -> list[str]:
    op = spec.GetInfo("apiSchemas") if hasattr(spec, "GetInfo") else spec.info.get("apiSchemas", [])
    if isinstance(op, list):
        return op
    return list(op.explicitItems)


def _attr_spec(spec, name: str):
    if hasattr(spec, "attributes"):
        return spec.attributes.get(name)
    return None


def _has_attr(spec, name: str) -> bool:
    return _attr_spec(spec, name) is not None


def _attr_val(spec, name: str):
    a = _attr_spec(spec, name)
    assert a is not None, f"Missing attribute {name!r} on {spec.path}"
    return a.default


def _attr_type(spec, name: str) -> str:
    a = _attr_spec(spec, name)
    assert a is not None, f"Missing attribute {name!r} on {spec.path}"
    return str(a.typeName)


def _attr_uniform(spec, name: str) -> bool:
    a = _attr_spec(spec, name)
    assert a is not None, f"Missing attribute {name!r} on {spec.path}"
    return "Uniform" in str(a.variability) or str(a.variability) == "uniform"


def _rel_targets(spec, name: str) -> list[str]:
    rel = spec.relationships.get(name)
    assert rel is not None, f"Missing relationship {name!r} on {spec.path}"
    items = getattr(rel.targetPathList, "prependedItems", rel.targetPathList)
    return [str(p) for p in items]


def _superellipsoid_red_bean_hull() -> tuple[np.ndarray, np.ndarray]:
    """62-vertex red bean superellipsoid matching scripts/g1d/rigid_bean_probe.py (7.0 x 5.0 x 5.0 mm, e=0.85)."""
    a, b, c, e_end = 0.0035, 0.0025, 0.0025, 0.85
    u = np.linspace(-np.pi / 2, np.pi / 2, 8)[1:-1]
    v = np.linspace(-np.pi, np.pi, 10, endpoint=False)
    pts = [[-a, 0.0, 0.0], [a, 0.0, 0.0]]
    for ui in u:
        cu = np.sign(np.cos(ui)) * (np.abs(np.cos(ui)) ** e_end)
        su = np.sign(np.sin(ui)) * (np.abs(np.sin(ui)) ** e_end)
        for vi in v:
            pts.append([a * su, b * cu * np.cos(vi), c * cu * np.sin(vi)])
    hull_pts = np.asarray(pts, dtype=np.float32)
    faces = ConvexHull(hull_pts).simplices.astype(np.int32)
    return hull_pts, faces


def _write_pack(
    tmp_path: Path,
    *,
    scales: np.ndarray | None = None,
    use_points_key: bool = False,
    include_faces: bool = True,
) -> Path:
    hull_pts, faces = _superellipsoid_red_bean_hull()
    n = 4 if scales is None else len(scales)
    if scales is None:
        scales = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.10, 1.10, 1.10],
                [1.06, 0.96, 1.04],
                [0.92, 0.92, 0.92],
            ],
            dtype=np.float32,
        )
    poses = np.zeros((n, 7), dtype=np.float32)
    poses[:, 0] = np.linspace(-0.006, 0.006, n)
    poses[:, 2] = np.linspace(0.010, 0.025, n)
    poses[:, 6] = 1.0  # (x, y, z, qx, qy, qz, qw)
    masses = np.full(n, 1.25e-4, dtype=np.float32)
    payload: dict[str, object] = {
        "pose": poses,
        "scale": np.asarray(scales, dtype=np.float32),
        "mass": masses,
        "volume": np.float64(9.0e-8),
        "packing": np.float64(0.62),
    }
    payload["points" if use_points_key else "hull"] = hull_pts
    if include_faces:
        payload["faces"] = faces
    pack_path = tmp_path / ("pack_points.npz" if use_points_key else "pack.npz")
    np.savez(pack_path, **payload)
    return pack_path


def test_default_backward_compatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_path = _write_pack(tmp_path)
    stage = _make_stage(monkeypatch)
    cfg = {
        "physics": {"type": "granular", "pack": str(pack_path), "count": 3},
        "visual": {"color": [0.075, 0.006, 0.008], "roughness": 0.22},
    }
    obj = GranularObject(
        prim_path="/World/envs/env_0/beans",
        usd_path="",
        inst_config=cfg,
        env_origin=(0.5, -0.2, 0.0),
        default_pos=(0.1, 0.2, 0.3),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        stage=stage,
    )
    assert obj.count == 3
    pos, _, _ = obj.get_particle_positions()
    assert pos.shape == (3, 3)

    phys = _prim_spec(stage, "/World/envs/env_0/beans/physics_material")
    assert phys is not None
    assert _schemas(phys) == ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    assert _attr_val(phys, "physics:staticFriction") == pytest.approx(0.3)
    assert _attr_val(phys, "physics:dynamicFriction") == pytest.approx(0.3)
    assert _attr_val(phys, "physxMaterial:frictionCombineMode") == "min"
    assert _attr_uniform(phys, "physxMaterial:frictionCombineMode")
    assert not _has_attr(phys, "physxMaterial:restitutionCombineMode")
    assert _attr_val(phys, "physics:restitution") == pytest.approx(0.1)

    body0 = _prim_spec(stage, "/World/envs/env_0/beans/bean_0")
    body1 = _prim_spec(stage, "/World/envs/env_0/beans/bean_1")
    hull0 = _prim_spec(stage, "/World/envs/env_0/beans/bean_0/hull")
    assert body0 is not None and hull0 is not None
    assert body0.typeName == "Xform"
    assert _schemas(body0) == BODY_SCHEMAS
    assert tuple(_attr_val(body1, "xformOp:scale")) == pytest.approx((1.10, 1.10, 1.10))
    assert _attr_val(body0, "physxRigidBody:solverPositionIterationCount") == ITERATIONS == 32
    assert _attr_val(body0, "physxRigidBody:solverVelocityIterationCount") == VELOCITY_ITERATIONS == 2
    assert _attr_val(body0, "physxRigidBody:maxDepenetrationVelocity") == pytest.approx(MAX_DEPENETRATION_VELOCITY)
    for opt_attr in (
        "physxRigidBody:linearDamping",
        "physxRigidBody:angularDamping",
        "physxRigidBody:sleepThreshold",
        "physxRigidBody:contactSlopCoefficient",
    ):
        assert not _has_attr(body0, opt_attr)

    assert hull0.typeName == "Mesh"
    assert _schemas(hull0) == HULL_SCHEMAS
    assert _attr_val(hull0, "physics:approximation") == "convexHull"
    assert _attr_val(hull0, "physxCollision:contactOffset") == pytest.approx(0.001)
    assert _attr_val(hull0, "physxCollision:restOffset") == pytest.approx(REST_OFFSET)
    assert not _has_attr(hull0, "physxCollision:torsionalPatchRadius")
    assert not _has_attr(hull0, "physxCollision:minTorsionalPatchRadius")
    assert _rel_targets(hull0, "material:binding") == ["/World/envs/env_0/beans/look"]
    assert _rel_targets(hull0, "material:binding:physics") == ["/World/envs/env_0/beans/physics_material"]


def test_capsule_geometry_scaling_and_zero_default_rest_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_path = _write_pack(tmp_path)
    hull_pts, _ = _superellipsoid_red_bean_hull()
    expected_r = float(np.hypot(hull_pts[:, 1], hull_pts[:, 2]).max())
    expected_a = float(np.abs(hull_pts[:, 0]).max())
    expected_h = 2.0 * (expected_a - expected_r)
    assert expected_r == pytest.approx(0.00244662, abs=1e-5)
    assert expected_h == pytest.approx(0.00210676, abs=1e-5)
    assert 2.0 * expected_r + expected_h == pytest.approx(0.0070, abs=1e-6)

    stage = _make_stage(monkeypatch)
    GranularObject(
        prim_path="/World/beans_cap",
        usd_path="",
        inst_config={"physics": {"type": "granular", "pack": str(pack_path), "shape": "capsule", "count": 3}},
        env_origin=(0.0, 0.0, 0.0),
        default_pos=(0.0, 0.0, 0.0),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        stage=stage,
    )

    body0 = _prim_spec(stage, "/World/beans_cap/bean_0")
    body1 = _prim_spec(stage, "/World/beans_cap/bean_1")
    body2 = _prim_spec(stage, "/World/beans_cap/bean_2")
    cap0 = _prim_spec(stage, "/World/beans_cap/bean_0/capsule")
    cap1 = _prim_spec(stage, "/World/beans_cap/bean_1/capsule")
    cap2 = _prim_spec(stage, "/World/beans_cap/bean_2/capsule")
    vis0 = _prim_spec(stage, "/World/beans_cap/bean_0/visual")
    assert body0 is not None and cap0 is not None and vis0 is not None
    assert _prim_spec(stage, "/World/beans_cap/bean_0/hull") is None
    assert body0.typeName == "Xform"
    assert _schemas(body0) == BODY_SCHEMAS
    assert cap0.typeName == "Capsule"
    assert _schemas(cap0) == CAPSULE_COLLIDER_SCHEMAS
    assert vis0.typeName == "Mesh"
    assert _attr_val(cap0, "axis") == "X"
    assert _attr_uniform(cap0, "axis")
    assert _attr_type(cap0, "radius") == "double"
    assert _attr_type(cap0, "height") == "double"

    # Grain 0 has unit scale (1, 1, 1)
    assert _attr_val(cap0, "radius") == pytest.approx(expected_r)
    assert _attr_val(cap0, "height") == pytest.approx(expected_h)
    assert 2.0 * _attr_val(cap0, "radius") + _attr_val(cap0, "height") == pytest.approx(0.0070, abs=1e-6)
    ext0 = [tuple(v) for v in _attr_val(cap0, "extent")]
    assert ext0[0] == pytest.approx((-expected_a, -expected_r, -expected_r), abs=1e-6)
    assert ext0[1] == pytest.approx((expected_a, expected_r, expected_r), abs=1e-6)
    assert tuple(_attr_val(body0, "xformOp:scale")) == pytest.approx((1.0, 1.0, 1.0))

    # Grain 1 has uniform scale (1.10, 1.10, 1.10)
    assert _attr_val(cap1, "radius") == pytest.approx(expected_r * 1.10, rel=1e-6)
    assert _attr_val(cap1, "height") == pytest.approx(expected_h * 1.10, rel=1e-6)
    assert tuple(_attr_val(body1, "xformOp:scale")) == pytest.approx((1.0, 1.0, 1.0))

    # Grain 2 has non-uniform jitter (1.06, 0.96, 1.04): baked into radius/height so xformOp:scale stays unit
    assert _attr_val(cap2, "radius") == pytest.approx(expected_r * 1.00, rel=1e-5)
    assert 2.0 * _attr_val(cap2, "radius") + _attr_val(cap2, "height") == pytest.approx(0.0070 * 1.06, rel=1e-5)
    assert tuple(_attr_val(body2, "xformOp:scale")) == pytest.approx((1.0, 1.0, 1.0))

    # Analytic capsule has no hull cooking shrinkage -> restOffset defaults to 0.0
    assert _attr_val(cap0, "physxCollision:restOffset") == pytest.approx(0.0)
    assert _attr_val(cap0, "physxCollision:contactOffset") == pytest.approx(0.001)
    assert _rel_targets(vis0, "material:binding") == ["/World/beans_cap/look"]
    assert _rel_targets(cap0, "material:binding:physics") == ["/World/beans_cap/physics_material"]


def test_capsule_explicit_overrides_points_pack_and_cavity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_path = _write_pack(tmp_path, use_points_key=True, include_faces=False)
    cavity = {
        "axis": [0.0, 0.0],
        "z": [0.002, 0.040, 0.080],
        "r": [0.030, 0.032, 0.035],
        "r_in": [0.029, 0.031, 0.034],
        "bottom": 0.0,
        "top": 0.085,
        "body": {"z": [0.0, 0.085], "r": [0.034, 0.038]},
        "handle": None,
    }
    stage = _make_stage(monkeypatch)
    obj = GranularObject(
        prim_path="/World/beans_override",
        usd_path="",
        inst_config={
            "physics": {
                "type": "granular",
                "pack": str(pack_path),
                "shape": "capsule",
                "capsule_radius": 0.00245,
                "capsule_height": 0.00210,
                "rest_offset": 0.00015,
                "cavity": cavity,
                "cavity_scale": 1.0,
                "yaw": 0.25,
            }
        },
        env_origin=(0.0, 0.0, 0.0),
        default_pos=(0.0, 0.0, 0.0),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        stage=stage,
    )
    assert obj.count == 4
    cap0 = _prim_spec(stage, "/World/beans_override/bean_0/capsule")
    assert _attr_val(cap0, "radius") == pytest.approx(0.00245)
    assert _attr_val(cap0, "height") == pytest.approx(0.00210)
    assert _attr_val(cap0, "physxCollision:restOffset") == pytest.approx(0.00015)


def test_friction_decoupling_and_combine_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_path = _write_pack(tmp_path)
    stage = _make_stage(monkeypatch)
    GranularObject(
        prim_path="/World/beans_mat",
        usd_path="",
        inst_config={
            "physics": {
                "type": "granular",
                "pack": str(pack_path),
                "count": 1,
                "friction": 0.30,
                "static_friction": 0.42,
                "dynamic_friction": 0.34,
                "friction_combine": "average",
                "restitution": 0.05,
                "restitution_combine": "min",
            }
        },
        env_origin=(0.0, 0.0, 0.0),
        default_pos=(0.0, 0.0, 0.0),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        stage=stage,
    )
    phys = _prim_spec(stage, "/World/beans_mat/physics_material")
    assert _attr_val(phys, "physics:staticFriction") == pytest.approx(0.42)
    assert _attr_val(phys, "physics:dynamicFriction") == pytest.approx(0.34)
    assert _attr_val(phys, "physxMaterial:frictionCombineMode") == "average"
    assert _attr_uniform(phys, "physxMaterial:frictionCombineMode")
    assert _attr_val(phys, "physics:restitution") == pytest.approx(0.05)
    assert _attr_val(phys, "physxMaterial:restitutionCombineMode") == "min"
    assert _attr_uniform(phys, "physxMaterial:restitutionCombineMode")


def test_tgs_depenetration_damping_torsional_and_contact_slop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_path = _write_pack(tmp_path)
    stage = _make_stage(monkeypatch)
    GranularObject(
        prim_path="/World/beans_real520",
        usd_path="",
        inst_config={
            "physics": {
                "type": "granular",
                "pack": str(pack_path),
                "count": 2,
                "shape": "capsule",
                "capsule_radius": 0.00245,
                "capsule_height": 0.0021,
                "friction": 0.35,
                "friction_combine": "average",
                "restitution": 0.05,
                "restitution_combine": "min",
                "contact_offset": 0.0008,
                "torsional_patch_radius": 0.0008,
                "min_torsional_patch_radius": 0.0002,
                "max_depenetration_velocity": 0.15,
                "position_iterations": 24,
                "velocity_iterations": 4,
                "linear_damping": 0.5,
                "angular_damping": 1.2,
                "sleep_threshold": 0.0001,
            }
        },
        env_origin=(0.0, 0.0, 0.0),
        default_pos=(0.0, 0.0, 0.0),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        stage=stage,
    )
    body0 = _prim_spec(stage, "/World/beans_real520/bean_0")
    cap0 = _prim_spec(stage, "/World/beans_real520/bean_0/capsule")
    assert _attr_val(body0, "physxRigidBody:maxDepenetrationVelocity") == pytest.approx(0.15)
    assert _attr_val(body0, "physxRigidBody:solverPositionIterationCount") == 24
    assert _attr_val(body0, "physxRigidBody:solverVelocityIterationCount") == 4
    assert _attr_val(body0, "physxRigidBody:linearDamping") == pytest.approx(0.5)
    assert _attr_val(body0, "physxRigidBody:angularDamping") == pytest.approx(1.2)
    assert _attr_val(body0, "physxRigidBody:sleepThreshold") == pytest.approx(0.0001)
    assert _attr_val(cap0, "physxCollision:contactOffset") == pytest.approx(0.0008)
    assert _attr_val(cap0, "physxCollision:restOffset") == pytest.approx(0.0)
    assert _attr_val(cap0, "physxCollision:torsionalPatchRadius") == pytest.approx(0.0008)
    assert _attr_val(cap0, "physxCollision:minTorsionalPatchRadius") == pytest.approx(0.0002)
    assert not _has_attr(body0, "physxRigidBody:contactSlopCoefficient")

    # Verify contact_slop and legacy `iterations` on convex_hull, plus invalid shape rejection
    stage_hull = _make_stage(monkeypatch)
    GranularObject(
        prim_path="/World/beans_hull_slop",
        usd_path="",
        inst_config={
            "physics": {
                "type": "granular",
                "pack": str(pack_path),
                "count": 1,
                "shape": "convex_hull",
                "iterations": 20,
                "contact_slop": 0.0009,
                "torsional_patch_radius": 0.0006,
            }
        },
        env_origin=(0.0, 0.0, 0.0),
        default_pos=(0.0, 0.0, 0.0),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        stage=stage_hull,
    )
    body0 = _prim_spec(stage_hull, "/World/beans_hull_slop/bean_0")
    hull0 = _prim_spec(stage_hull, "/World/beans_hull_slop/bean_0/hull")
    assert _attr_val(body0, "physxRigidBody:solverPositionIterationCount") == 20
    assert _attr_val(body0, "physxRigidBody:contactSlopCoefficient") == pytest.approx(0.0009)
    assert _attr_val(hull0, "physxCollision:torsionalPatchRadius") == pytest.approx(0.0006)

    with pytest.raises(ValueError, match="Unsupported granular collider shape"):
        GranularObject(
            prim_path="/World/beans_bad",
            usd_path="",
            inst_config={"physics": {"type": "granular", "pack": str(pack_path), "shape": "octahedron"}},
            env_origin=(0.0, 0.0, 0.0),
            default_pos=(0.0, 0.0, 0.0),
            default_ori=(1.0, 0.0, 0.0, 0.0),
            stage=stage_hull,
        )
