"""CPU unit tests for scripts/objects/video2data.py (no Isaac Sim or network required)."""

import http.client
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import types
import urllib.error
import urllib.request

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("video2data", Path(__file__).with_name("video2data.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def box_mesh(lo, hi):
    """Closed, outward-wound quad mesh of an axis-aligned box: points, faceVertexCounts, faceVertexIndices."""
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return pts, [4] * 6, [i for q in quads for i in q]


def test_slug_and_object_id_conventions():
    assert V.object_slug("contigo_coffee_mug") == "video2data_contigo_coffee_mug"
    assert V.object_id_for("contigo_coffee_mug") == "RoboDojo/Rigid/video2data_contigo_coffee_mug/00000"
    assert V.object_slug("Contigo-Coffee Mug!") == "video2data_contigo_coffee_mug"
    assert V.object_slug("corn_can", variant="bundlesdf") == "video2data_bundlesdf_corn_can"
    assert V.object_id_for("corn_can", variant="sam3d") == "RoboDojo/Rigid/video2data_sam3d_corn_can/00000"
    with pytest.raises(ValueError, match="invalid object name"):
        V.object_slug("!!!")


def test_parse_variant_files_filters_variant_glb_objects_limit_and_missing_entry():
    entries = [
        {"path": "README.md", "size": 500, "type": "file"},
        {"path": "contigo_coffee_mug/einstar", "size": 0, "type": "directory"},
        {"path": "contigo_coffee_mug/einstar/output.usd", "size": 22000, "type": "file"},
        {"path": "contigo_coffee_mug/einstar/visual_asset.usd", "size": 6000000, "type": "file"},
        {"path": "contigo_coffee_mug/einstar/output.glb", "size": 12000000, "type": "file"},
        {"path": "contigo_coffee_mug/einstar/textures/tex0.png", "size": 1500000, "type": "file"},
        {"path": "contigo_coffee_mug/bundlesdf/output.usd", "size": 10000, "type": "file"},
        {"path": "corn_can/einstar/output.usd", "size": 15000, "type": "file"},
        {"path": "corn_can/einstar/visual_asset.usd", "size": 1000000, "type": "file"},
        # Incomplete package missing output.usd must be ignored
        {"path": "broken_obj/einstar/visual_asset.usd", "size": 500000, "type": "file"},
    ]

    pkgs, docs = V.parse_variant_files(entries, variant="einstar", include_glb=False)
    assert [d["path"] for d in docs] == ["README.md"]
    assert [p["object"] for p in pkgs] == ["contigo_coffee_mug", "corn_can"]
    mug_paths = [f["path"] for f in pkgs[0]["files"]]
    assert "contigo_coffee_mug/einstar/output.glb" not in mug_paths
    assert len(mug_paths) == 3
    assert pkgs[0]["size_bytes"] == 22000 + 6000000 + 1500000

    pkgs_glb, _ = V.parse_variant_files(entries, variant="einstar", include_glb=True)
    assert any(f["path"].endswith(".glb") for f in pkgs_glb[0]["files"])

    pkgs_sub, _ = V.parse_variant_files(entries, variant="einstar", objects=["corn_can"])
    assert [p["object"] for p in pkgs_sub] == ["corn_can"]

    pkgs_lim, _ = V.parse_variant_files(entries, variant="einstar", limit=1)
    assert [p["object"] for p in pkgs_lim] == ["contigo_coffee_mug"]

    with pytest.raises(ValueError, match="unsupported variant"):
        V.parse_variant_files(entries, variant="unknown")


def test_fetch_hf_tree_pagination_headers_and_retry(monkeypatch):
    seen_reqs = []
    sleeps = []
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, body, headers=None):
            self._body = body
            self.headers = headers or {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=60):
        seen_reqs.append(req)
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, io.BytesIO(b""))
        if calls["n"] == 2:
            body = json.dumps([{"path": "a/einstar/output.usd", "size": 10, "type": "file"}]).encode()
            return FakeResp(body, {"Link": '<https://hf-mirror.com/api/page2>; rel="next"'})
        if calls["n"] == 3:
            raise urllib.error.URLError("transient reset")
        body = json.dumps([{"path": "b/einstar/output.usd", "size": 20, "type": "file"}]).encode()
        return FakeResp(body, {})

    monkeypatch.setattr(V.urllib.request, "urlopen", fake_urlopen)
    entries = V.fetch_hf_tree(
        endpoint="https://hf-mirror.com",
        token="hf_secret",
        sleep_fn=sleeps.append,
    )
    assert len(entries) == 2
    assert [e["path"] for e in entries] == ["a/einstar/output.usd", "b/einstar/output.usd"]
    assert seen_reqs[0].headers["User-agent"] == "curl/8.0"
    assert seen_reqs[0].headers["Authorization"] == "Bearer hf_secret"
    assert seen_reqs[0].full_url.startswith("https://hf-mirror.com/api/datasets/nvidia/video-to-data-object-assets/")
    assert sleeps == [3.0, 0.25, 1.0]


def test_download_file_atomic_retry_size_check_and_skip(monkeypatch, tmp_path):
    attempts = {"n": 0}
    sleeps = []

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=120):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResp(b"short")  # size mismatch triggers retry
        if attempts["n"] == 2:
            raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, io.BytesIO(b""))
        return FakeResp(b"0123456789")

    monkeypatch.setattr(V.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "obj/einstar/output.usd"
    status = V.download_file("https://hf-mirror.com/file", dest, expected_size=10, sleep_fn=sleeps.append)
    assert status == "downloaded"
    assert dest.read_bytes() == b"0123456789"
    assert not dest.with_name(dest.name + ".part").exists()
    assert len(sleeps) == 2

    # Second call with matching size skips without network
    assert V.download_file("https://hf-mirror.com/file", dest, expected_size=10, sleep_fn=sleeps.append) == "skipped"
    assert attempts["n"] == 3

    # Non-retryable 404 fails immediately
    def fail_404(req, timeout=120):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(V.urllib.request, "urlopen", fail_404)
    with pytest.raises(urllib.error.HTTPError):
        V.download_file("https://hf-mirror.com/missing", tmp_path / "missing.usd", sleep_fn=sleeps.append)


def test_fetch_dataset_writes_upstream_manifest(monkeypatch, tmp_path):
    blobs = {
        "README.md": b"# video-to-data\n",
        "corn_can/einstar/output.usd": b"USD1",
        "corn_can/einstar/visual_asset.usd": b"VIS123",
    }

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=120):
        for rel, data in blobs.items():
            if req.full_url.endswith(rel):
                return FakeResp(data)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(V.urllib.request, "urlopen", fake_urlopen)
    tree = [
        {"path": "README.md", "size": len(blobs["README.md"]), "type": "file"},
        {"path": "corn_can/einstar/output.usd", "size": len(blobs["corn_can/einstar/output.usd"]), "type": "file"},
        {
            "path": "corn_can/einstar/visual_asset.usd",
            "size": len(blobs["corn_can/einstar/visual_asset.usd"]),
            "type": "file",
        },
        {"path": "corn_can/einstar/output.glb", "size": 99999, "type": "file"},
    ]
    manifest = V.fetch_dataset(
        tmp_path / "src",
        tree_entries=tree,
        endpoint="https://hf-mirror.com",
        sleep_fn=lambda _: None,
    )
    assert manifest["object_count"] == 1
    assert manifest["downloaded_files"] == 2
    assert manifest["failed_files"] == {}
    assert manifest["license"] == "CC-BY-4.0"
    assert (tmp_path / "src/corn_can/einstar/output.usd").read_bytes() == b"USD1"
    assert not (tmp_path / "src/corn_can/einstar/output.glb").exists()
    upstream = json.loads((tmp_path / "src/UPSTREAM.json").read_text())
    assert upstream["revision"] == V.REVISION
    assert upstream["objects"] == ["corn_can"]


def test_wire_packages_symlinks_copies_and_skips_incomplete(tmp_path):
    src = tmp_path / "source"
    mug = src / "contigo_coffee_mug/einstar"
    (mug / "textures").mkdir(parents=True)
    (mug / "output.usd").write_text("#usda 1.0\n")
    (mug / "visual_asset.usd").write_text("#usda 1.0\n")
    (mug / "output.glb").write_bytes(b"GLB")
    (mug / "textures/tex0.png").write_bytes(b"PNG")
    # Incomplete object without output.usd
    bad = src / "incomplete_obj/einstar"
    bad.mkdir(parents=True)
    (bad / "visual_asset.usd").write_text("#usda 1.0\n")

    lib = tmp_path / "lib/Object"
    ids_file = tmp_path / "wired_ids.txt"
    manifest_file = tmp_path / "wire_manifest.json"
    rc = V.main(
        [
            "wire",
            "--source",
            str(src),
            "--lib",
            str(lib),
            "--write-ids",
            str(ids_file),
            "--out",
            str(manifest_file),
        ]
    )
    assert rc == 0
    oid = "RoboDojo/Rigid/video2data_contigo_coffee_mug/00000"
    assert ids_file.read_text().splitlines() == [oid]
    report = json.loads(manifest_file.read_text())
    assert report["wired_count"] == 1 and report["skipped_count"] == 1

    wired_dir = lib / oid
    assert (wired_dir / "output.usd").is_symlink()
    assert (wired_dir / "visual_asset.usd").is_symlink()
    assert (wired_dir / "textures").is_symlink()
    assert (wired_dir / "textures/tex0.png").read_bytes() == b"PNG"
    assert not (wired_dir / "output.glb").exists()
    meta = json.loads((wired_dir / "metadata.json").read_text())
    assert meta["runtime_ready"] is False
    assert meta["source"]["license"] == "CC-BY-4.0"
    assert meta["source"]["object_name"] == "contigo_coffee_mug"
    assert V.discover_video2data_ids(lib) == [oid]

    # Copy mode copies plain files and textures directory
    lib_copy = tmp_path / "lib_copy/Object"
    V.wire_packages(src, lib_copy, copy=True, include_glb=True)
    copy_dir = lib_copy / oid
    assert not (copy_dir / "output.usd").is_symlink() and (copy_dir / "output.usd").is_file()
    assert not (copy_dir / "textures").is_symlink() and (copy_dir / "textures/tex0.png").is_file()
    assert (copy_dir / "output.glb").is_file()


def test_resolve_mass_prefers_authored_clamps_and_falls_back():
    pts, counts, idx = box_mesh((0, 0, 0), (0.08, 0.08, 0.12))
    tris = V.simready.triangles(counts, idx)
    size = np.array([0.08, 0.08, 0.12])

    # Authored MassAPI within range (e.g. contigo_coffee_mug 0.30 kg)
    m, info = V.resolve_mass([0.30], pts, tris, size, [], "ceramic")
    assert m == pytest.approx(0.30)
    assert info["source"] == "authored_mass_api" and not info["clamped"]

    # Heavy authored mass (e.g. dumbbell_15lb = 6.8 kg) clamped to loader cap 0.5 kg
    m_heavy, info_heavy = V.resolve_mass([6.8], pts, tris, size, [], "metal")
    assert m_heavy == pytest.approx(0.5)
    assert info_heavy["clamped"] is True and info_heavy["raw_kg"] == pytest.approx(6.8)

    # Zero/None/NaN authored mass falls back to simready.estimate_mass
    m_fb, info_fb = V.resolve_mass([0.0, None, float("nan")], pts, tris, size, [], "ceramic")
    assert 0.15 <= m_fb <= 0.45
    assert info_fb["density_source"] == "category_default"


def _install_fake_pxr(
    monkeypatch,
    *,
    vis_bounds=((0.0, 0.0, 0.0), (0.08, 0.08, 0.16)),
    authored_mass=0.30,
    has_collider=True,
    unresolved=("gltf/pbr.mdl",),
):
    pts, counts, idx = box_mesh(*vis_bounds)

    class FakePath:
        def __init__(self, s):
            self.pathString = s
            self.isEmpty = not bool(s)
            self.name = s.rsplit("/", 1)[-1] if s else ""

        def __str__(self):
            return self.pathString

        def __eq__(self, other):
            return isinstance(other, FakePath) and self.pathString == other.pathString

    class FakeAttr:
        def __init__(self, val):
            self._val = val

        def Get(self):
            return self._val

    class FakePrim:
        def __init__(self, path, type_name="Xform", schemas=(), attrs=None, purpose="default", vis="inherited"):
            self._path = FakePath(path)
            self._type = type_name
            self._schemas = set(schemas)
            self._attrs = dict(attrs or {})
            self._purpose = purpose
            self._vis = vis
            self._active = True
            self._refs = []
            self._direct_phys = FakePath("")

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._path.name

        def GetTypeName(self):
            return self._type

        def GetAppliedSchemas(self):
            return list(self._schemas)

        def RemoveAppliedSchema(self, api):
            self._schemas.discard(api)

        def HasAPI(self, api_cls):
            return api_cls._schema_name in self._schemas

        def IsA(self, cls):
            return self._type == cls._type_name

        def IsInstance(self):
            return False

        def SetInstanceable(self, _flag):
            pass

        def SetActive(self, flag):
            self._active = flag

        def GetReferences(self):
            return types.SimpleNamespace(AddReference=lambda ref: self._refs.append(ref))

    asset_schemas = ["PhysicsRigidBodyAPI"]
    asset_attrs = {}
    if authored_mass is not None:
        asset_schemas.append("PhysicsMassAPI")
        asset_attrs["mass"] = authored_mass
    src_asset = FakePrim("/Asset", "Xform", schemas=asset_schemas, attrs=asset_attrs)
    col_schemas = ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI") if has_collider else ()

    class FakeStage:
        def __init__(self, prims, up_axis="Z", mpu=1.0, path=None):
            self._prims = {p.GetPath().pathString: p for p in prims}
            self._up = up_axis
            self._mpu = mpu
            self._default = prims[0] if prims else None
            self._path = path

        def GetDefaultPrim(self):
            return self._default

        def SetDefaultPrim(self, prim):
            self._default = prim

        def Traverse(self):
            return list(self._prims.values())

        def GetPrimAtPath(self, path):
            return self._prims[path]

        def SetEditTarget(self, _layer):
            pass

        def GetSessionLayer(self):
            return None

        def GetRootLayer(self):
            return types.SimpleNamespace(Save=lambda: Path(self._path).write_text("#usda 1.0\n"))

    src_stage = FakeStage([src_asset], up_axis="Z", mpu=1.0)
    wrapper_stage = [None]
    state = {"shift": (0.0, 0.0, 0.0), "orient": (1.0, 0.0, 0.0, 0.0)}

    class FakeXform:
        def __init__(self, prim):
            self._prim = prim

        def GetPrim(self):
            return self._prim

        @classmethod
        def Define(cls, stage, path):
            p = FakePrim(path, "Xform")
            stage._prims[path] = p
            return cls(p)

        def AddTranslateOp(self):
            return types.SimpleNamespace(Set=lambda v: state.__setitem__("shift", tuple(v)))

        def AddOrientOp(self):
            return types.SimpleNamespace(Set=lambda q: state.__setitem__("orient", tuple(q)))

        def AddTransformOp(self, opSuffix=None):
            return types.SimpleNamespace(Set=lambda _m: None)

    class FakeXformCache:
        def GetLocalToWorldTransform(self, _prim):
            w, x, y, z = state["orient"]
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ],
                float,
            )
            tf = np.eye(4)
            tf[:3, :3] = rot.T
            tf[3, :3] = state["shift"]
            return tf

    class FakeBindingAPI:
        def __init__(self, p):
            self._p = p

        @classmethod
        def Apply(cls, p):
            return cls(p)

        def Bind(self, mat, _strength=None, materialPurpose=None):
            self._p._direct_phys = mat.GetPath()

        def GetDirectBinding(self, materialPurpose=None):
            return types.SimpleNamespace(GetMaterialPath=lambda: self._p._direct_phys)

        def UnbindDirectBinding(self, materialPurpose=None):
            self._p._direct_phys = FakePath("")

        def GetCollectionBindings(self, materialPurpose=None):
            return []

        def ComputeBoundMaterial(self, materialPurpose=None):
            probe_p = FakePrim("/root/probe_material")
            probe = types.SimpleNamespace(GetPath=lambda: FakePath("/root/probe_material"), GetPrim=lambda: probe_p)
            return (probe, None)

    class FakeLayer:
        def __init__(self, path):
            self.path = path

    def open_stage(arg):
        if isinstance(arg, FakeLayer):
            state["shift"] = (0.0, 0.0, 0.0)
            state["orient"] = (1.0, 0.0, 0.0, 0.0)
            fresh_asset = FakePrim("/root/normalize/asset", "Xform", schemas=asset_schemas, attrs=asset_attrs)
            fresh_vis = FakePrim(
                "/root/normalize/asset/visual",
                "Mesh",
                attrs={"points": pts, "counts": counts, "indices": idx},
            )
            fresh_col = FakePrim(
                "/root/normalize/asset/collider",
                "Mesh",
                schemas=col_schemas,
                attrs={
                    "points": pts,
                    "counts": counts,
                    "indices": idx,
                    "collisionEnabled": True,
                    "approximation": "convexHull",
                },
            )
            st = FakeStage([fresh_asset, fresh_vis, fresh_col], up_axis="Z", mpu=1.0, path=arg.path)
            wrapper_stage[0] = st
            return st
        if str(arg).endswith("object.usd"):
            return wrapper_stage[0]
        return src_stage

    class FakeGf:
        @staticmethod
        def Vec3d(x, y, z):
            return (x, y, z)

        @staticmethod
        def Vec3f(x, y, z):
            return (x, y, z)

        @staticmethod
        def Quatf(*args):
            if len(args) == 2:
                w, v = args
                return (w, v[0], v[1], v[2])
            return tuple(args)

        class Rotation:
            def __init__(self, axis, deg):
                self.deg = deg

        class Matrix4d:
            def __init__(self, _=1):
                self.scale = 1.0

            def SetRotate(self, _rot):
                pass

            def SetScale(self, s):
                self.scale = s
                return self

            def __mul__(self, other):
                return self

    FakeSdf = types.SimpleNamespace(
        Layer=types.SimpleNamespace(CreateNew=lambda path, args=None: FakeLayer(path)),
        AssetPath=lambda p: p,
    )
    FakeUsd = types.SimpleNamespace(
        Stage=types.SimpleNamespace(Open=open_stage),
        PrimRange=lambda _root: list(wrapper_stage[0].Traverse()),
    )
    FakeUsdGeom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(z="Z", guide="guide", invisible="invisible"),
        GetStageUpAxis=lambda st: st._up,
        GetStageMetersPerUnit=lambda st: st._mpu,
        SetStageUpAxis=lambda st, ax: setattr(st, "_up", ax),
        SetStageMetersPerUnit=lambda st, u: setattr(st, "_mpu", u),
        Xform=FakeXform,
        XformCache=FakeXformCache,
        Camera=type("Camera", (), {"_type_name": "Camera"}),
        Mesh=type(
            "Mesh",
            (),
            {
                "_type_name": "Mesh",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetPointsAttr": lambda self: FakeAttr(self._p._attrs.get("points")),
                "GetFaceVertexCountsAttr": lambda self: FakeAttr(self._p._attrs.get("counts")),
                "GetFaceVertexIndicesAttr": lambda self: FakeAttr(self._p._attrs.get("indices")),
            },
        ),
        Imageable=lambda p: types.SimpleNamespace(ComputePurpose=lambda: p._purpose, ComputeVisibility=lambda: p._vis),
    )
    FakeUsdPhysics = types.SimpleNamespace(
        Joint=type("Joint", (), {"_type_name": "PhysicsJoint"}),
        ArticulationRootAPI=type("ArticulationRootAPI", (), {"_schema_name": "PhysicsArticulationRootAPI"}),
        RigidBodyAPI=type(
            "RigidBodyAPI",
            (),
            {
                "_schema_name": "PhysicsRigidBodyAPI",
                "Apply": classmethod(lambda cls, p: p._schemas.add(cls._schema_name)),
            },
        ),
        MassAPI=type(
            "MassAPI",
            (),
            {
                "_schema_name": "PhysicsMassAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetMassAttr": lambda self: FakeAttr(self._p._attrs.get("mass")),
                "Apply": classmethod(
                    lambda cls, p: (
                        p._schemas.add(cls._schema_name),
                        types.SimpleNamespace(CreateMassAttr=lambda m: p._attrs.__setitem__("mass", m)),
                    )[1]
                ),
            },
        ),
        CollisionAPI=type(
            "CollisionAPI",
            (),
            {
                "_schema_name": "PhysicsCollisionAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetCollisionEnabledAttr": lambda self: FakeAttr(self._p._attrs.get("collisionEnabled", True)),
            },
        ),
        MeshCollisionAPI=type(
            "MeshCollisionAPI",
            (),
            {
                "_schema_name": "PhysicsMeshCollisionAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetApproximationAttr": lambda self: FakeAttr(self._p._attrs.get("approximation", "none")),
            },
        ),
        MaterialAPI=type(
            "MaterialAPI",
            (),
            {
                "_schema_name": "PhysicsMaterialAPI",
                "__init__": lambda self, p: setattr(self, "_p", p),
                "GetDensityAttr": lambda self: FakeAttr(self._p._attrs.get("density", 0.0)),
            },
        ),
    )
    FakeUsdShade = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(weakerThanDescendants="weakerThanDescendants"),
        MaterialBindingAPI=FakeBindingAPI,
        Material=types.SimpleNamespace(Define=lambda st, path: FakePrim(path, "Material")),
    )
    FakeUsdUtils = types.SimpleNamespace(
        ComputeAllDependencies=lambda _ap: ([], [], list(unresolved)),
    )
    monkeypatch.setattr(
        V,
        "pxr_modules",
        lambda: (FakeGf, None, FakeSdf, FakeUsd, FakeUsdGeom, FakeUsdPhysics, FakeUsdShade, FakeUsdUtils),
    )


def _seed_wired_package(lib, obj_name="contigo_coffee_mug"):
    oid = V.object_id_for(obj_name)
    pkg = lib / oid
    (pkg / "textures").mkdir(parents=True)
    (pkg / "output.usd").write_text("#usda 1.0\n")
    (pkg / "visual_asset.usd").write_text("#usda 1.0\n")
    (pkg / "textures/output_aligned_texture0.png").write_bytes(b"PNG")
    (pkg / "metadata.json").write_text(
        json.dumps(
            {
                "source": {
                    "dataset": "video-to-data-object-assets",
                    "object_name": obj_name,
                    "variant": "einstar",
                    "license": "CC-BY-4.0",
                },
                "runtime_ready": False,
            }
        )
    )
    return oid


def test_convert_one_handles_dual_usds_textures_authored_mass_and_gltf_mdl(monkeypatch, tmp_path):
    lib = tmp_path / "lib/Object"
    ovl = tmp_path / "ovl/Object"
    oid = _seed_wired_package(lib, "contigo_coffee_mug")
    _install_fake_pxr(monkeypatch, authored_mass=0.30, unresolved=("gltf/pbr.mdl",))

    summary = V.convert_one(lib, oid, ovl, orient="keep")
    assert summary["mass_kg"] == pytest.approx(0.30)
    assert summary["mass_source"] == "authored_mass_api"
    assert summary["material"] == "ceramic"
    assert summary["orient_reason"] == "keep"
    assert summary["tabletop_range"] is True
    assert (ovl / oid / "object.usd").is_file()
    assert (ovl / oid / "visual_asset.usd").exists()
    assert (ovl / oid / "textures/output_aligned_texture0.png").exists()

    meta = json.loads((ovl / oid / "metadata.json").read_text())
    assert meta["runtime_ready"] is False
    assert meta["adapter"]["reference"] == "./output.usd"
    assert meta["adapter"]["collider"] == {"count": 1, "approximation": ["convexHull"], "source": "package, unchanged"}
    assert meta["source"]["license"] == "CC-BY-4.0"
    assert meta["source"]["object_name"] == "contigo_coffee_mug"
    assert meta["physics"]["mass"] == pytest.approx(0.30)


def test_convert_one_orient_lay_flat_and_error_paths(monkeypatch, tmp_path):
    lib = tmp_path / "lib/Object"
    ovl = tmp_path / "ovl/Object"
    oid = _seed_wired_package(lib, "cutting_board")
    # Thin board standing on edge: X=0.02, Y=0.20, Z=0.30
    _install_fake_pxr(monkeypatch, vis_bounds=((0, 0, 0), (0.02, 0.20, 0.30)), authored_mass=None)

    summary = V.convert_one(lib, oid, ovl, orient="lay-flat")
    assert summary["orient_reason"] == "lay_flat"
    assert summary["size_m"][2] == pytest.approx(0.02, abs=1e-4)
    assert math.isclose(summary["mass_kg"], 0.5)  # 0.02*0.2*0.3*800 = 0.96 kg, clamped to 0.5 kg cap

    # Missing collider must raise
    _install_fake_pxr(monkeypatch, has_collider=False)
    with pytest.raises(ValueError, match="no enabled collision geometry"):
        V.convert_one(lib, oid, ovl)

    # Unresolved non-builtin dependency must raise
    _install_fake_pxr(monkeypatch, unresolved=("missing/custom.png",))
    with pytest.raises(ValueError, match="unresolved dependencies"):
        V.convert_one(lib, oid, ovl)

    # Missing output.usd must raise
    bad_oid = V.object_id_for("no_entry")
    (lib / bad_oid).mkdir(parents=True)
    with pytest.raises(ValueError, match="missing output.usd"):
        V.convert_one(lib, bad_oid, ovl)


def test_cli_convert_skip_existing_and_ready(monkeypatch, tmp_path):
    lib = tmp_path / "lib/Object"
    ovl = tmp_path / "ovl/Object"
    check_dir = tmp_path / "check"
    oid_pass = _seed_wired_package(lib, "contigo_coffee_mug")
    oid_fail = _seed_wired_package(lib, "corn_can")
    _install_fake_pxr(monkeypatch, authored_mass=0.25)

    conv_summary = tmp_path / "conv.json"
    rc = V.main(["convert", "--lib", str(lib), "--out", str(ovl), "--all", "--summary", str(conv_summary)])
    assert rc == 0
    assert len(json.loads(conv_summary.read_text())["converted"]) == 2

    # Write check.json for pass and fail
    p_pass = check_dir / "Rigid/video2data_contigo_coffee_mug/00000/preview/check.json"
    p_pass.parent.mkdir(parents=True)
    p_pass.write_text(json.dumps({"settle": "pass", "tilt_deg": 0.4, "z_m": 0.08, "expect_z_m": 0.08}))
    p_fail = check_dir / "Rigid/video2data_corn_can/00000/preview/check.json"
    p_fail.parent.mkdir(parents=True)
    p_fail.write_text(json.dumps({"settle": "fail", "tilt_deg": 45.0, "z_m": 0.04, "expect_z_m": 0.08}))

    ready_summary = tmp_path / "ready.json"
    rc_ready = V.main(
        [
            "ready",
            "--out",
            str(ovl),
            "--check-dir",
            str(check_dir),
            "--all",
            "--summary",
            str(ready_summary),
            "--note",
            "unit test",
        ]
    )
    assert rc_ready == 1
    rs = json.loads(ready_summary.read_text())
    assert rs["passed_ids"] == [oid_pass]
    assert rs["not_ready"] == [oid_fail]
    assert json.loads((ovl / oid_pass / "metadata.json").read_text())["runtime_ready"] is True
    assert json.loads((ovl / oid_fail / "metadata.json").read_text())["runtime_ready"] is False

    # Re-running convert with --skip-existing skips the already-ready mug
    conv2 = tmp_path / "conv2.json"
    rc2 = V.main(["convert", "--lib", str(lib), "--out", str(ovl), "--all", "--skip-existing", "--summary", str(conv2)])
    assert rc2 == 0
    c2 = json.loads(conv2.read_text())
    assert c2["skipped"] == [oid_pass]
    assert [row["id"] for row in c2["converted"]] == [oid_fail]


def test_incomplete_read_retry_and_cross_host_auth_strip(monkeypatch, tmp_path):
    calls = {"n": 0}
    sleeps = []

    class TruncatedResp:
        def read(self, _n=-1):
            raise http.client.IncompleteRead(b"partial", expected=10)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class OkResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=120):
        calls["n"] += 1
        if calls["n"] == 1:
            return TruncatedResp()
        return OkResp(b"0123456789")

    monkeypatch.setattr(V.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "obj/einstar/visual_asset.usd"
    status = V.download_file("https://hf-mirror.com/file", dest, expected_size=10, sleep_fn=sleeps.append)
    assert status == "downloaded"
    assert dest.read_bytes() == b"0123456789"
    assert not dest.with_name(dest.name + ".part").exists()
    assert sleeps == [1.0]

    handler = V._CrossHostAuthStripRedirectHandler()
    orig_req = urllib.request.Request(
        "https://hf-mirror.com/datasets/nvidia/video-to-data-object-assets/resolve/rev/a.usd",
        headers={"Authorization": "Bearer hf_secret", "User-Agent": "curl/8.0"},
    )
    cross_req = handler.redirect_request(orig_req, None, 302, "Found", {}, "https://cdn-oss.example.com/presigned")
    assert "Authorization" not in cross_req.headers
    assert cross_req.headers["User-agent"] == "curl/8.0"

    same_req = handler.redirect_request(orig_req, None, 302, "Found", {}, "https://hf-mirror.com/other/path")
    assert same_req.headers["Authorization"] == "Bearer hf_secret"


def test_convert_one_removes_dangling_wrapper_on_post_save_failure(monkeypatch, tmp_path):
    lib = tmp_path / "lib/Object"
    ovl = tmp_path / "ovl/Object"
    oid = _seed_wired_package(lib, "bad_dep_mug")
    _install_fake_pxr(monkeypatch, unresolved=("missing/texture.png",))

    with pytest.raises(ValueError, match="unresolved dependencies"):
        V.convert_one(lib, oid, ovl)
    assert not (ovl / oid / "object.usd").exists()
    assert not (ovl / oid / "metadata.json").exists()


def test_mirror_package_same_dir_noop_and_skips_part_files(tmp_path):
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "output.usd").write_text("#usda 1.0\n")
    (src / "visual_asset.usd.part").write_bytes(b"partial")
    (src / "output.glb").write_bytes(b"GLB")

    # Mirroring into itself must not destroy existing files
    V.mirror_package(src, src, copy=False, include_glb=False)
    assert (src / "output.usd").is_file() and not (src / "output.usd").is_symlink()

    # Mirroring into another dir must skip *.part even when include_glb=True
    dst = tmp_path / "dst"
    V.mirror_package(src, dst, copy=False, include_glb=True)
    assert (dst / "output.usd").is_symlink()
    assert (dst / "output.glb").is_symlink()
    assert not (dst / "visual_asset.usd.part").exists()


def test_parse_variant_files_deduplicates_and_rejects_negative_limit():
    entries = [
        {"path": "README.md", "size": 100, "type": "file"},
        {"path": "README.md", "size": 100, "type": "file"},
        {"path": "corn_can/einstar/output.usd", "size": 500, "type": "file"},
        {"path": "corn_can/einstar/output.usd", "size": 500, "type": "file"},
    ]
    pkgs, docs = V.parse_variant_files(entries)
    assert len(docs) == 1
    assert len(pkgs) == 1 and len(pkgs[0]["files"]) == 1 and pkgs[0]["size_bytes"] == 500

    with pytest.raises(ValueError, match="limit must be >= 0"):
        V.parse_variant_files(entries, limit=-1)


def test_bare_stdlib_wire_and_ready_without_numpy(tmp_path):
    """Simulate bare host python3 on Slurm nodes (no numpy) running wire and ready."""
    src = tmp_path / "source"
    pkg = src / "corn_can/einstar"
    pkg.mkdir(parents=True)
    (pkg / "output.usd").write_text("#usda 1.0\n")
    lib = tmp_path / "lib/Object"
    check_dir = tmp_path / "check"
    chk = check_dir / "Rigid/video2data_corn_can/00000/preview/check.json"
    chk.parent.mkdir(parents=True)
    chk.write_text(json.dumps({"settle": "pass", "tilt_deg": 0.2, "z_m": 0.05, "expect_z_m": 0.05}))

    script = Path(__file__).with_name("video2data.py")
    code = f"""
import sys
sys.modules["numpy"] = None
import importlib.util
spec = importlib.util.spec_from_file_location("video2data", {str(script)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.np is None and mod.simready is None
rc_w = mod.main(["wire", "--source", {str(src)!r}, "--lib", {str(lib)!r}])
assert rc_w == 0
rc_r = mod.main(["ready", "--out", {str(lib)!r}, "--check-dir", {str(check_dir)!r}, "--all", "--note", "bare"])
assert rc_r == 0
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    meta = json.loads((lib / "RoboDojo/Rigid/video2data_corn_can/00000/metadata.json").read_text())
    assert meta["runtime_ready"] is True
    assert meta["source"]["runtime_validated"] is True
    assert meta["adapter"]["validation"]["passed"] is True
    assert meta["adapter"]["validation"]["note"] == "bare"
