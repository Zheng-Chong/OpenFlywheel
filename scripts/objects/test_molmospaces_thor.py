"""CPU unit tests for scripts/objects/molmospaces_thor.py (no Isaac; pxr tests guarded with importorskip)."""

from __future__ import annotations

import email.message
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


T = _load("molmospaces_thor")


def _make_tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _compress_zst(data: bytes) -> bytes:
    zstandard = pytest.importorskip("zstandard")
    return zstandard.ZstdCompressor().compress(data)


def test_pure_stdlib_import_does_not_require_pxr_or_numpy():
    # Importing molmospaces_thor should never eagerly pull in pxr.
    assert "pxr" not in sys.modules or hasattr(T, "filter_catalog")
    assert T.category_from_asset_id("Side_Table_320_1") == "Side_Table"
    assert T.category_from_asset_id("Microwave_4") == "Microwave"
    assert T.category_from_asset_id("CD_1") == "CD"
    assert T.category_from_asset_id("Bathroom_Faucet_12") == "Bathroom_Faucet"
    assert T.category_from_asset_id("PlainName") == "PlainName"


def test_build_catalog_and_filter_deterministic_ids():
    umeta = {
        "Microwave_4": {"asset_id": "Microwave_4", "hash_id": "h1", "articulated": True},
        "Microwave_1": {"asset_id": "Microwave_1", "hash_id": "h2", "articulated": True},
        "Box_1": {"asset_id": "Box_1", "hash_id": "h3", "articulated": True},
        "Alarm_Clock_1": {"asset_id": "Alarm_Clock_1", "hash_id": "h4", "articulated": False},
        "Orphan_1": {"asset_id": "Orphan_1", "hash_id": "h5", "articulated": True},
    }
    arrow = [
        {"path": "thor_Microwave_4.tar.zst", "shard_id": 0, "offset": 512, "size": 1000},
        {"path": "thor_Microwave_1.tar.zst", "shard_id": 0, "offset": 2048, "size": 1200},
        {"path": "thor_Box_1.tar.zst", "shard_id": 0, "offset": 4096, "size": 800},
        {"path": "thor_Alarm_Clock.tar.zst", "shard_id": 0, "offset": 6144, "size": 1500},
    ]
    cmeta = {
        "thor_Microwave_4.tar.zst": {"Microwave_4": {"Textures": {"a.png": {}}, "Microwave_4.usda": {}}},
        "thor_Microwave_1.tar.zst": {"Microwave_1": {"Microwave_1.usda": {}}},
        "thor_Box_1.tar.zst": {"Box_1": {"Textures": {"b.png": {}}, "Box_1.usda": {}}},
        "thor_Alarm_Clock.tar.zst": {"Alarm_Clock_1": {"Textures": {"c.png": {}}, "Alarm_Clock_1.usda": {}}},
    }

    rows = T.build_catalog(umeta, arrow, cmeta)
    assert len(rows) == 5

    # Articulated filter: sorted by category_slug then asset_id; Orphan_1 dropped for missing archive range.
    sel = T.filter_catalog(rows, physics_type="articulated")
    assert [c["asset_id"] for c in sel["chosen"]] == ["Box_1", "Microwave_1", "Microwave_4"]
    assert [c["id"] for c in sel["chosen"]] == [
        "Articulation/thor_box/00000",
        "Articulation/thor_microwave/00000",
        "Articulation/thor_microwave/00001",
    ]
    dropped = {d["asset_id"]: d["reason"] for d in sel["dropped"]}
    assert "missing archive" in dropped["Orphan_1"]
    assert dropped["Alarm_Clock_1"] == "physics_type=rigid"

    # textured_only drops Microwave_1; max_per_category=1 caps microwaves.
    sel_tex = T.filter_catalog(rows, physics_type="articulated", textured_only=True)
    assert [c["asset_id"] for c in sel_tex["chosen"]] == ["Box_1", "Microwave_4"]

    sel_cap = T.filter_catalog(rows, physics_type="articulated", max_per_category=1)
    assert [c["asset_id"] for c in sel_cap["chosen"]] == ["Box_1", "Microwave_1"]

    # physics_type="all" assigns Rigid IDs independently starting at 00000.
    sel_all = T.filter_catalog(rows, physics_type="all")
    by_aid = {c["asset_id"]: c["id"] for c in sel_all["chosen"]}
    assert by_aid["Alarm_Clock_1"] == "Rigid/thor_alarm_clock/00000"
    assert by_aid["Box_1"] == "Articulation/thor_box/00000"

    # Category include & exclude filters + limit.
    sel_cat = T.filter_catalog(rows, physics_type="all", categories=["microwave"], limit=1)
    assert [c["asset_id"] for c in sel_cat["chosen"]] == ["Microwave_1"]

    sel_ex = T.filter_catalog(rows, physics_type="articulated", exclude_categories=["box"])
    assert [c["asset_id"] for c in sel_ex["chosen"]] == ["Microwave_1", "Microwave_4"]

    # Empty catalog produces empty result.
    empty = T.filter_catalog([])
    assert empty["chosen"] == [] and empty["dropped"] == []


def test_hf_endpoint_and_http_get_retries_and_user_agent(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert T.hf_endpoint() == "https://hf-mirror.com"
    assert T.dataset_url() == "https://hf-mirror.com/datasets/allenai/molmospaces"
    assert T.resolve_url("arrow_table.json") == (
        "https://hf-mirror.com/datasets/allenai/molmospaces/resolve/main/isaac/objects/thor/20260128/arrow_table.json"
    )

    seen_uas: list[str] = []
    seen_ranges: list[str | None] = []
    sleeps: list[float] = []
    calls = {"n": 0}

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body
            self.headers = {"Content-Type": "application/octet-stream"}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=120):
        seen_uas.append(req.get_header("User-agent"))
        seen_ranges.append(req.get_header("Range"))
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "5"
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, io.BytesIO(b""))
        if calls["n"] == 2:
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", email.message.Message(), io.BytesIO()
            )
        return _FakeResp(b"ok-range")

    monkeypatch.setattr(T.urllib.request, "urlopen", fake_urlopen)
    data, _ = T.http_get(
        "https://hf-mirror.com/test",
        headers={"Range": "bytes=0-7"},
        attempts=4,
        backoff=1.0,
        sleep_fn=sleeps.append,
    )
    assert data == b"ok-range"
    assert seen_uas == ["curl/8.0", "curl/8.0", "curl/8.0"]
    assert seen_ranges == ["bytes=0-7", "bytes=0-7", "bytes=0-7"]
    assert sleeps == [5.0, 2.0]

    # 404 fails immediately without sleeping.
    sleeps_404: list[float] = []

    def fake_404(req, timeout=120):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", email.message.Message(), io.BytesIO())

    monkeypatch.setattr(T.urllib.request, "urlopen", fake_404)
    with pytest.raises(urllib.error.HTTPError):
        T.http_get("https://hf-mirror.com/missing", attempts=4, backoff=1.0, sleep_fn=sleeps_404.append)
    assert sleeps_404 == []


def test_safe_extract_and_fetch_assets_from_local_shard(tmp_path):
    inner_tar = _make_tar_bytes(
        {
            "Microwave_4/Microwave_4.usda": b"#usda 1.0\n",
            "Microwave_4/Textures/albedo.png": b"pngdata",
            "Microwave_4_mesh/Microwave_4_mesh.usda": b"skip-mesh-variant",
            "Microwave_4_prim/Microwave_4_prim.usda": b"skip-prim-variant",
            "../escape.txt": b"evil",
        }
    )
    zst_payload = _compress_zst(inner_tar)

    # Place zst_payload at offset 512 inside a fake local shards/00000.tar.
    shard = tmp_path / "shards" / "00000.tar"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"\x00" * 512 + zst_payload + b"\x00" * 128)

    entries = [
        {
            "asset_id": "Microwave_4",
            "archive": "thor_Microwave_4.tar.zst",
            "shard_id": 0,
            "offset": 512,
            "size": len(zst_payload),
        }
    ]
    res = T.fetch_assets(entries, tmp_path, workers=2)
    assert res["fetched"] == ["Microwave_4"]
    assert res["failed"] == []
    assert (tmp_path / "assets/Microwave_4/Microwave_4.usda").read_bytes() == b"#usda 1.0\n"
    assert (tmp_path / "assets/Microwave_4/Textures/albedo.png").read_bytes() == b"pngdata"
    assert not (tmp_path / "assets/Microwave_4_mesh").exists()
    assert not (tmp_path / "assets/Microwave_4_prim").exists()
    assert not (tmp_path / "escape.txt").exists()

    # Re-running fetch skips already-extracted asset without touching network or shard.
    res2 = T.fetch_assets(entries, tmp_path, workers=2)
    assert res2["already_present"] == ["Microwave_4"]
    assert res2["failed"] == []

    # Archive with only auxiliary textures and no <asset_id>.usda must NOT count as extracted.
    partial_tar = _make_tar_bytes({"Broken_1/Textures/only.png": b"png"})
    got_partial = T._safe_extract_assets(partial_tar, tmp_path / "assets_partial", {"Broken_1"})
    assert got_partial == []


def test_find_thor_base_and_tree_reversed_and_canonical():
    # Multi-hop THOR reversed tree (Fridge_1 <- FridgeBodyMesh <- doormesh <- handle).
    facts_thor = {
        "links": [
            "/root/Fridge_1",
            "/root/Fridge_1_FridgeBodyMesh",
            "/root/Fridge_1_doormesh",
            "/root/Fridge_1_handle",
        ],
        "articulation_roots": ["/root/Fridge_1"],
        "joints": [
            {
                "path": "/root/Fridge_1_FridgeBodyMesh/PhysicsFixedJoint",
                "name": "PhysicsFixedJoint",
                "type": "fixed",
                "body0": "/root/Fridge_1_FridgeBodyMesh",
                "body1": "/root/Fridge_1",
                "excluded": False,
            },
            {
                "path": "/root/Fridge_1_doormesh/Fridge_1_joint",
                "name": "Fridge_1_joint",
                "type": "revolute",
                "body0": "/root/Fridge_1_doormesh",
                "body1": "/root/Fridge_1_FridgeBodyMesh",
                "excluded": False,
            },
            {
                "path": "/root/Fridge_1_handle/PhysicsFixedJoint",
                "name": "PhysicsFixedJoint",
                "type": "fixed",
                "body0": "/root/Fridge_1_handle",
                "body1": "/root/Fridge_1_doormesh",
                "excluded": False,
            },
        ],
    }
    base, dirs, unreachable = T.find_thor_base_and_tree(facts_thor)
    assert base == "/root/Fridge_1"
    assert unreachable == []
    assert dirs["/root/Fridge_1_FridgeBodyMesh/PhysicsFixedJoint"] == (
        "/root/Fridge_1",
        "/root/Fridge_1_FridgeBodyMesh",
        True,
    )
    assert dirs["/root/Fridge_1_doormesh/Fridge_1_joint"] == (
        "/root/Fridge_1_FridgeBodyMesh",
        "/root/Fridge_1_doormesh",
        True,
    )
    assert dirs["/root/Fridge_1_handle/PhysicsFixedJoint"] == (
        "/root/Fridge_1_doormesh",
        "/root/Fridge_1_handle",
        True,
    )

    # Disconnected floating link is detected.
    facts_float = {**facts_thor, "links": [*facts_thor["links"], "/root/floating_piece"]}
    _, _, unreach = T.find_thor_base_and_tree(facts_float)
    assert unreach == ["/root/floating_piece"]

    # Already-canonical tree reports is_reversed=False.
    facts_canon = {
        "links": ["/root/base", "/root/lid"],
        "articulation_roots": ["/root/base"],
        "joints": [
            {
                "path": "/root/hinge",
                "name": "hinge",
                "type": "revolute",
                "body0": "/root/base",
                "body1": "/root/lid",
                "excluded": False,
            }
        ],
    }
    base_c, dirs_c, unreach_c = T.find_thor_base_and_tree(facts_canon)
    assert base_c == "/root/base" and unreach_c == []
    assert dirs_c["/root/hinge"] == ("/root/base", "/root/lid", False)


def test_estimate_link_masses_replaces_dummy_and_enforces_floor():
    links = ["/root/Microwave_4", "/root/bodymesh", "/root/doormesh"]
    authored = {"/root/Microwave_4": 1e-8}  # dummy mass; children have no authored mass
    volumes = {"/root/Microwave_4": 0.005**3, "/root/bodymesh": 0.04, "/root/doormesh": 0.005}
    masses, info = T.estimate_link_masses(
        links,
        authored,
        volumes,
        [0.5, 0.35, 0.3],
        max_total_mass=15.0,
    )
    assert info["dummy_links_replaced"] == 1
    assert info["filled_missing_links"] is True
    assert all(m >= T.MIN_LINK_MASS_KG - 1e-9 for m in masses.values())
    assert masses["/root/bodymesh"] > masses["/root/doormesh"] > masses["/root/Microwave_4"]
    assert T.DEFAULT_MIN_ARTICULATED_MASS_KG <= sum(masses.values()) <= 15.0

    # Empty links raises.
    with pytest.raises(ValueError):
        T.estimate_link_masses([], {}, {}, [0.2, 0.2, 0.2], max_total_mass=5.0)


def test_decide_articulated_and_rigid_edges():
    base_facts = {
        "up_axis": "Z",
        "world_movable": [],
        "links": ["/m/Geometry/m", "/m/Geometry/m_body", "/m/Geometry/m_doormesh"],
        "articulation_roots": ["/m/Geometry/m"],
        "joints": [
            {
                "path": "/m/j0",
                "name": "j0",
                "type": "fixed",
                "body0": "/m/Geometry/m_body",
                "body1": "/m/Geometry/m",
                "excluded": False,
            },
            {
                "path": "/m/j1",
                "name": "j1",
                "type": "revolute",
                "body0": "/m/Geometry/m_doormesh",
                "body1": "/m/Geometry/m_body",
                "excluded": False,
            },
        ],
        "movable": [
            {
                "path": "/m/j1",
                "name": "j1",
                "type": "revolute",
                "body0": "/m/Geometry/m_doormesh",
                "body1": "/m/Geometry/m_body",
                "excluded": False,
            },
        ],
        "colliders": 5,
    }
    ok, reason = T.decide_articulated([0.5, 0.35, 0.3], base_facts)
    assert ok and "door" in reason

    ok_y, reason_y = T.decide_articulated([0.5, 0.35, 0.3], {**base_facts, "up_axis": "Y"})
    assert not ok_y and "up axis" in reason_y

    ok_big, reason_big = T.decide_articulated([1.2, 0.5, 0.5], base_facts, max_size=0.9)
    assert not ok_big and "exceeds" in reason_big

    ok_rigid, _ = T.decide_rigid([0.15, 0.12, 0.1], {"up_axis": "Z", "colliders": 3})
    assert ok_rigid
    ok_rigid_nocoll, _ = T.decide_rigid([0.15, 0.12, 0.1], {"up_axis": "Z", "colliders": 0})
    assert not ok_rigid_nocoll


def test_rewrite_layer_paths_rewrites_relationships_and_connections():
    """Verify rewrite_layer_paths updates BOTH RelationshipSpec.targetPathList AND AttributeSpec.connectionPathList."""

    class FakePath:
        def __init__(self, s: str):
            self.s = s

        def HasPrefix(self, other: FakePath) -> bool:
            return self.s == other.s or self.s.startswith(other.s + "/") or self.s.startswith(other.s + ".")

        def ReplacePrefix(self, old: FakePath, new: FakePath) -> FakePath:
            assert self.HasPrefix(old)
            return FakePath(new.s + self.s[len(old.s) :])

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakePath) and self.s == other.s

        def __str__(self) -> str:
            return self.s

    class FakeListOp:
        def __init__(self, items: list[FakePath]):
            self.explicitItems = list(items)

        def GetAddedOrExplicitItems(self):
            return list(self.explicitItems)

        def ClearEdits(self):
            self.explicitItems = []

    class FakeRelSpec:
        def __init__(self, targets: list[FakePath]):
            self.targetPathList = FakeListOp(targets)

    class FakeAttrSpec:
        def __init__(self, conns: list[FakePath]):
            self.connectionPathList = FakeListOp(conns)

    class FakeSdf:
        RelationshipSpec = FakeRelSpec
        AttributeSpec = FakeAttrSpec

    class FakePrimSpec:
        def __init__(self, props=(), children=()):
            self.properties = list(props)
            self.nameChildren = list(children)

    class FakeLayer:
        def __init__(self, root_prims):
            self.rootPrims = list(root_prims)

    rel_body = FakeRelSpec([FakePath("/Microwave_4/Geometry/Microwave_4_doormesh")])
    rel_mat = FakeRelSpec([FakePath("/Microwave_4/Materials/mat_0")])
    attr_surf = FakeAttrSpec([FakePath("/Microwave_4/Materials/mat_0/PreviewSurface.outputs:surface")])
    attr_st = FakeAttrSpec([FakePath("/Microwave_4/Materials/mat_0/Primvar_st.outputs:result")])
    attr_plain = FakeAttrSpec([])

    layer = FakeLayer(
        [
            FakePrimSpec(
                props=[rel_body, rel_mat],
                children=[FakePrimSpec(props=[attr_surf, attr_st, attr_plain])],
            )
        ]
    )
    remap = [
        (FakePath("/Microwave_4/Geometry/Microwave_4_doormesh"), FakePath("/root/Microwave_4_doormesh")),
        (FakePath("/Microwave_4"), FakePath("/root")),
    ]
    rels_n, conns_n = T.rewrite_layer_paths(layer, remap, Sdf_mod=FakeSdf)
    assert rels_n == 2 and conns_n == 2
    assert rel_body.targetPathList.explicitItems == [FakePath("/root/Microwave_4_doormesh")]
    assert rel_mat.targetPathList.explicitItems == [FakePath("/root/Materials/mat_0")]
    assert attr_surf.connectionPathList.explicitItems == [
        FakePath("/root/Materials/mat_0/PreviewSurface.outputs:surface")
    ]
    assert attr_st.connectionPathList.explicitItems == [FakePath("/root/Materials/mat_0/Primvar_st.outputs:result")]


def test_stamp_ready_and_cli_filter_and_ready(tmp_path):
    # 1. CLI `filter` using pre-cached `_index/` files (pure stdlib, no network).
    sources = tmp_path / "sources"
    idx_dir = sources / "_index"
    idx_dir.mkdir(parents=True)
    (idx_dir / T.ARROW_TABLE_FILE).write_text(
        json.dumps([{"path": "thor_Box_1.tar.zst", "shard_id": 0, "offset": 512, "size": 900}]),
        encoding="utf-8",
    )
    (idx_dir / T.COMBINED_META_FILE).write_bytes(
        gzip.compress(json.dumps({"thor_Box_1.tar.zst": {"Box_1": {"Box_1.usda": {}}}}).encode("utf-8"))
    )
    (idx_dir / "usd_assets_metadata.json").write_text(
        json.dumps({"Box_1": {"asset_id": "Box_1", "hash_id": "abc", "articulated": True}}),
        encoding="utf-8",
    )
    sel_out = tmp_path / "sel.json"
    rc = T.main(["filter", "--sources-dir", str(sources), "--out", str(sel_out)])
    assert rc == 0
    sel = json.loads(sel_out.read_text(encoding="utf-8"))
    assert [c["id"] for c in sel["chosen"]] == ["Articulation/thor_box/00000"]

    # 2. `stamp_ready` for Articulation (pass & fail) + Rigid (pass & fail) + missing.
    out_root = tmp_path / "out"
    check_dir = tmp_path / "check"
    art_ok = "Articulation/thor_box/00000"
    art_bad = "Articulation/thor_microwave/00000"
    rig_ok = "Rigid/thor_alarm_clock/00000"
    rig_bad = "Rigid/thor_alarm_clock/00001"
    for oid in (art_ok, art_bad, rig_ok, rig_bad):
        d = out_root / "Assets/Object/RoboDojo" / oid
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({"asset_id": oid, "runtime_ready": False}), encoding="utf-8")

    (check_dir / art_ok / "preview").mkdir(parents=True)
    (check_dir / art_ok / "preview/check.json").write_text(json.dumps({"pass": True, "checks": {"rest": "pass"}}))
    (check_dir / art_bad / "preview").mkdir(parents=True)
    (check_dir / art_bad / "preview/check.json").write_text(json.dumps({"pass": False, "checks": {"rest": "fail"}}))
    (check_dir / rig_ok / "preview").mkdir(parents=True)
    (check_dir / rig_ok / "preview/check.json").write_text(json.dumps({"settle": "pass", "tilt_deg": 0.2}))
    (check_dir / rig_bad / "preview").mkdir(parents=True)
    (check_dir / rig_bad / "preview/check.json").write_text(json.dumps({"settle": "fail", "tilt_deg": 88.0}))

    summary_path = tmp_path / "ready_summary.json"
    rc_ready = T.main(
        [
            "ready",
            "--root",
            str(out_root),
            "--check-dir",
            str(check_dir),
            "--summary",
            str(summary_path),
            art_ok,
            art_bad,
            rig_ok,
            rig_bad,
        ]
    )
    assert rc_ready == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed_ids"] == [art_ok, rig_ok]
    assert {x["id"]: x["why"] for x in summary["not_ready"]} == {art_bad: ["rest"], rig_bad: ["settle"]}
    assert (
        json.loads((out_root / "Assets/Object/RoboDojo" / art_ok / "metadata.json").read_text())["runtime_ready"]
        is True
    )
    assert (
        json.loads((out_root / "Assets/Object/RoboDojo" / rig_bad / "metadata.json").read_text())["runtime_ready"]
        is False
    )

    ok_miss, why_miss = T.stamp_ready(out_root, "Rigid/thor_none/00000", check_dir)
    assert not ok_miss and why_miss == ["missing_metadata"]


def test_canonicalize_thor_joints_handles_asymmetric_authored_attrs():
    """Verify _canonicalize_thor_joints clears unauthored attrs when swapping asymmetric poses or unbounded limits."""

    class FakeAttr:
        def __init__(self, val=None, authored=False):
            self._val = val
            self._authored = authored

        def HasAuthoredValue(self):
            return self._authored

        def Get(self):
            return self._val

        def Set(self, v):
            self._val = v
            self._authored = True

        def Clear(self):
            self._val = None
            self._authored = False

    class FakeRel:
        def __init__(self, targets):
            self.targets = list(targets)

        def SetTargets(self, targets):
            self.targets = list(targets)

    class FakePrim:
        def __init__(self, kind, *, p0, p1, r0, r1, lo, hi):
            self.kind = kind
            self.body0 = FakeRel([])
            self.body1 = FakeRel([])
            self.p0, self.p1 = p0, p1
            self.r0, self.r1 = r0, r1
            self.lo, self.hi = lo, hi

        def IsA(self, cls):
            return self.kind == cls.__name__.lstrip("_")

    class FakeJoint:
        def __init__(self, prim):
            self._p = prim

        def GetBody0Rel(self):
            return self._p.body0

        def GetBody1Rel(self):
            return self._p.body1

        def GetLocalPos0Attr(self):
            return self._p.p0

        def GetLocalPos1Attr(self):
            return self._p.p1

        def GetLocalRot0Attr(self):
            return self._p.r0

        def GetLocalRot1Attr(self):
            return self._p.r1

    class _RevoluteJoint(FakeJoint):
        def GetLowerLimitAttr(self):
            return self._p.lo

        def GetUpperLimitAttr(self):
            return self._p.hi

    class _PrismaticJoint(_RevoluteJoint):
        pass

    class FakeUsdPhysics:
        Joint = FakeJoint
        RevoluteJoint = _RevoluteJoint
        PrismaticJoint = _PrismaticJoint

    class FakeSdf:
        @staticmethod
        def Path(s):
            return s

    class FakeStage:
        def __init__(self, prims):
            self._prims = prims

        def GetPrimAtPath(self, path):
            return self._prims.get(path)

    # Case 1: only localPos0 authored, only localRot1 authored, only lowerLimit authored (upper unbounded).
    prim_asym = FakePrim(
        "RevoluteJoint",
        p0=FakeAttr((0.1, 0.2, 0.3), authored=True),
        p1=FakeAttr((0.0, 0.0, 0.0), authored=False),
        r0=FakeAttr((1, 0, 0, 0), authored=False),
        r1=FakeAttr((0, 0, -1, 0), authored=True),
        lo=FakeAttr(10.0, authored=True),
        hi=FakeAttr(None, authored=False),
    )
    # Case 2: symmetric limits [0, 90] -> [-90, 0].
    prim_sym = FakePrim(
        "PrismaticJoint",
        p0=FakeAttr((0.0, 0.0, 0.0), authored=True),
        p1=FakeAttr((0.0, 0.16, 0.17), authored=True),
        r0=FakeAttr((1, 0, 0, 0), authored=True),
        r1=FakeAttr((1, 0, 0, 0), authored=True),
        lo=FakeAttr(0.0, authored=True),
        hi=FakeAttr(0.11, authored=True),
    )
    stage = FakeStage({"/root/j_asym": prim_asym, "/root/j_sym": prim_sym})
    flipped = T._canonicalize_thor_joints(
        stage,
        {
            "/root/j_asym": ("/root/parent", "/root/child", True),
            "/root/j_sym": ("/root/parent", "/root/child2", True),
        },
        Sdf_mod=FakeSdf,
        UsdPhysics_mod=FakeUsdPhysics,
    )
    assert flipped == 2
    assert prim_asym.body0.targets == ["/root/parent"] and prim_asym.body1.targets == ["/root/child"]
    assert not prim_asym.p0.HasAuthoredValue()
    assert prim_asym.p1.HasAuthoredValue() and prim_asym.p1.Get() == (0.1, 0.2, 0.3)
    assert prim_asym.r0.HasAuthoredValue() and prim_asym.r0.Get() == (0, 0, -1, 0)
    assert not prim_asym.r1.HasAuthoredValue()
    assert not prim_asym.lo.HasAuthoredValue()
    assert prim_asym.hi.HasAuthoredValue() and prim_asym.hi.Get() == pytest.approx(-10.0)

    assert prim_sym.p0.Get() == (0.0, 0.16, 0.17) and prim_sym.p1.Get() == (0.0, 0.0, 0.0)
    assert prim_sym.lo.Get() == pytest.approx(-0.11) and prim_sym.hi.Get() == pytest.approx(0.0)


def test_convert_one_writes_valid_articulated_and_rigid_metadata_on_cpu(tmp_path, monkeypatch):
    """Exercise convert_one() + real articulated.metadata() on CPU (guards against missing source keys)."""
    sources = tmp_path / "sources"
    out_root = tmp_path / "out"

    def fake_adapt_articulated(source_usda, target_dir, **kwargs):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "object.usd").write_bytes(b"fake-usdc-art")
        return {
            "size": [0.5, 0.35, 0.3],
            "centre_shift": [0.0, 0.0, -0.15],
            "unit": 1.0,
            "yaw_deg": 0.0,
            "joints": [
                {
                    "name": "Microwave_4_joint",
                    "type": "revolute",
                    "role": "door",
                    "part": "door",
                    "parent": "Microwave_4_bodymesh",
                    "child": "Microwave_4_doormesh",
                    "axis": "Y",
                    "limits_deg": [-90.0, 0.0],
                    "drive": {"type": "force", "stiffness": 0.0, "damping": 0.05, "target": 0.0},
                    "friction": {"static": 0.3, "dynamic": 0.3, "armature": 0.01},
                }
            ],
            "link_masses": {"Microwave_4": 0.02, "Microwave_4_bodymesh": 5.0, "Microwave_4_doormesh": 1.0},
            "mass_info": {"raw_authored_kg": 0.0, "final_kg": 6.02, "dummy_links_replaced": 1},
            "fixed_base": True,
            "limit_fixes": {},
            "removed": {"world_joints": [], "lights": [], "scripts": [], "articulation_root_moved_from": []},
            "reparented_prims": [("/Microwave_4/Geometry/Microwave_4", "/Microwave_4/Microwave_4")],
            "rewritten_paths": {"relationships": 4, "connections": 3},
            "joints_canonicalized": 2,
            "textures": ["textures/albedo.png"],
            "base_link": "Microwave_4",
            "links": ["Microwave_4", "Microwave_4_bodymesh", "Microwave_4_doormesh"],
        }

    def fake_adapt_rigid(source_usda, target_dir, **kwargs):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "object.usd").write_bytes(b"fake-usdc-rigid")
        return {
            "size": [0.16, 0.12, 0.09],
            "centre_shift": [0.0, 0.0, -0.045],
            "unit": 1.0,
            "yaw_deg": 0.0,
            "mass_kg": 0.42,
            "mass_info": {"raw_authored_kg": 0.0, "final_kg": 0.42, "clamped": False, "bounds_kg": [0.005, 0.5]},
            "rewritten_paths": {"relationships": 2, "connections": 2},
            "textures": ["textures/clock.png"],
        }

    monkeypatch.setattr(T, "adapt_articulated", fake_adapt_articulated)
    monkeypatch.setattr(T, "adapt_rigid", fake_adapt_rigid)

    art_rec = T.convert_one(
        sources,
        out_root,
        {
            "id": "Articulation/thor_microwave/00000",
            "asset_id": "Microwave_4",
            "category": "Microwave",
            "physics_type": "articulated",
        },
    )
    assert art_rec["id"] == "Articulation/thor_microwave/00000"
    art_meta = json.loads((out_root / "Assets/Object/RoboDojo" / art_rec["id"] / "metadata.json").read_text())
    assert art_meta["uuid"]
    assert art_meta["base_link"] == "Microwave_4"
    assert art_meta["source"]["object"] == f"{T.SUBPATH}/Microwave_4"
    assert art_meta["runtime_ready"] is False
    assert art_meta["adapter"]["joints_canonicalized"] == 2

    rig_rec = T.convert_one(
        sources,
        out_root,
        {
            "id": "Rigid/thor_alarm_clock/00000",
            "asset_id": "Alarm_Clock_1",
            "category": "Alarm_Clock",
            "physics_type": "rigid",
        },
    )
    assert rig_rec["id"] == "Rigid/thor_alarm_clock/00000"
    rig_meta = json.loads((out_root / "Assets/Object/RoboDojo" / rig_rec["id"] / "metadata.json").read_text())
    assert rig_meta["uuid"]
    assert rig_meta["physics"]["mass"] == 0.42
    assert rig_meta["runtime_ready"] is False


def _build_thor_articulated_stage(usda_path: Path, tex_path: Path):
    """Build a synthetic THOR articulated `.usda` matching live `Microwave_4/Microwave_4.usda` findings."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    usda_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-thor-texture")

    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    top = UsdGeom.Xform.Define(stage, "/Microwave_4")
    UsdGeom.ModelAPI.Apply(top.GetPrim()).CreateExtentsHintAttr().Set(
        [Gf.Vec3f(-0.3, -0.2, 0.0), Gf.Vec3f(0.3, 0.2, 0.3)]
    )
    stage.SetDefaultPrim(top.GetPrim())
    UsdGeom.Scope.Define(stage, "/Microwave_4/Geometry")
    UsdGeom.Scope.Define(stage, "/Microwave_4/Materials")
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")

    # Material with PreviewSurface + UsdUVTexture + PrimvarReader connected via AttributeSpec.connectionPathList
    mat = UsdShade.Material.Define(stage, "/Microwave_4/Materials/mat_body")
    shader = UsdShade.Shader.Define(stage, "/Microwave_4/Materials/mat_body/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    tex_shader = UsdShade.Shader.Define(stage, "/Microwave_4/Materials/mat_body/DiffuseTexture")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./Textures/albedo.png"))
    st_reader = UsdShade.Shader.Define(stage, "/Microwave_4/Materials/mat_body/Primvar_st")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Empty root link with ArticulationRootAPI and dummy mass=1e-8 kg
    root_link = UsdGeom.Xform.Define(stage, "/Microwave_4/Geometry/Microwave_4")
    UsdPhysics.RigidBodyAPI.Apply(root_link.GetPrim())
    UsdPhysics.MassAPI.Apply(root_link.GetPrim()).CreateMassAttr(1e-8)
    UsdPhysics.ArticulationRootAPI.Apply(root_link.GetPrim())

    # Body link + Door link under /Microwave_4/Geometry/ (NO link-level mass, collider density=220)
    for name, centre, size in (
        ("Microwave_4_bodymesh", (0.0, 0.0, 0.15), (0.45, 0.35, 0.30)),
        ("Microwave_4_doormesh", (0.24, 0.0, 0.15), (0.03, 0.34, 0.28)),
    ):
        link = UsdGeom.Xform.Define(stage, f"/Microwave_4/Geometry/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(*centre))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        cube = UsdGeom.Cube.Define(stage, f"/Microwave_4/Geometry/{name}/geom")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateDensityAttr(220.0)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    # FilteredPairs rel
    body_prim = stage.GetPrimAtPath("/Microwave_4/Geometry/Microwave_4_bodymesh")
    UsdPhysics.FilteredPairsAPI.Apply(body_prim).CreateFilteredPairsRel().SetTargets(
        ["/Microwave_4/Geometry/Microwave_4_doormesh"]
    )

    # THOR reversed joints: body0 = child, body1 = parent; zero drives, zero friction
    fj = UsdPhysics.FixedJoint.Define(stage, "/Microwave_4/Geometry/Microwave_4_bodymesh/PhysicsFixedJoint")
    fj.CreateBody0Rel().SetTargets(["/Microwave_4/Geometry/Microwave_4_bodymesh"])
    fj.CreateBody1Rel().SetTargets(["/Microwave_4/Geometry/Microwave_4"])

    rj = UsdPhysics.RevoluteJoint.Define(stage, "/Microwave_4/Geometry/Microwave_4_doormesh/Microwave_4_joint")
    rj.CreateBody0Rel().SetTargets(["/Microwave_4/Geometry/Microwave_4_doormesh"])
    rj.CreateBody1Rel().SetTargets(["/Microwave_4/Geometry/Microwave_4_bodymesh"])
    rj.CreateAxisAttr("Z")
    rj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    rj.CreateLocalPos1Attr(Gf.Vec3f(0.24, -0.17, 0.0))
    rj.CreateLowerLimitAttr(0.0)
    rj.CreateUpperLimitAttr(90.0)

    stage.GetRootLayer().Save()


def test_pxr_adapt_thor_articulated_full_pipeline(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    usda = tmp_path / "sources/assets/Microwave_4/Microwave_4.usda"
    tex = tmp_path / "sources/assets/Microwave_4/Textures/albedo.png"
    _build_thor_articulated_stage(usda, tex)

    manifest = T.survey(tmp_path / "sources")
    assert len(manifest["chosen"]) == 1 and manifest["dropped"] == []
    assert manifest["chosen"][0]["id"] == "Articulation/thor_microwave/00000"

    out_root = tmp_path / "out"
    conv = T.convert(tmp_path / "sources", out_root, manifest)
    assert len(conv["converted"]) == 1 and conv["failed"] == []

    target = out_root / "Assets/Object/RoboDojo/Articulation/thor_microwave/00000"
    assert (target / "object.usd").is_file()
    assert (target / "textures/albedo.png").is_file()

    stage = Usd.Stage.Open(str(target / "object.usd"))
    assert not stage.GetPrimAtPath("/PhysicsScene")
    root = stage.GetDefaultPrim()
    assert str(root.GetPath()) == "/root"
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not root.HasAttribute("extentsHint")

    # First direct child of /root MUST be the true base link Microwave_4 Xform with RigidBodyAPI.
    children = root.GetChildren()
    assert children[0].GetName() == "Microwave_4"
    assert children[0].GetTypeName() == "Xform"
    assert children[0].HasAPI(UsdPhysics.RigidBodyAPI)
    assert not children[0].HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/root/Geometry")

    # Joint canonicalized: body0 = bodymesh (parent), body1 = doormesh (child); limits negated & swapped [-90, 0].
    hinge = stage.GetPrimAtPath("/root/Microwave_4_doormesh/Microwave_4_joint")
    j = UsdPhysics.RevoluteJoint(hinge)
    assert [str(p) for p in j.GetBody0Rel().GetTargets()] == ["/root/Microwave_4_bodymesh"]
    assert [str(p) for p in j.GetBody1Rel().GetTargets()] == ["/root/Microwave_4_doormesh"]
    assert j.GetLowerLimitAttr().Get() == pytest.approx(-90.0)
    assert j.GetUpperLimitAttr().Get() == pytest.approx(0.0)

    # DriveAPI + PhysxJointAxisAPI authored.
    assert hinge.HasAPI(UsdPhysics.DriveAPI, "angular")
    assert hinge.GetAttribute("physxJointAxis:angular:staticFrictionEffort").Get() > 0
    assert hinge.GetAttribute("physxJointAxis:angular:dynamicFrictionEffort").Get() > 0
    assert hinge.GetAttribute("physxJointAxis:angular:armature").Get() == pytest.approx(T.DEFAULT_ANGULAR_ARMATURE)

    # FilteredPairs & Material shader connections rewritten under /root.
    fp = (
        UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath("/root/Microwave_4_bodymesh"))
        .GetFilteredPairsRel()
        .GetTargets()
    )
    assert [str(p) for p in fp] == ["/root/Microwave_4_doormesh"]
    mat = UsdShade.Material(stage.GetPrimAtPath("/root/Materials/mat_body"))
    surf_sources = mat.GetSurfaceOutput().GetAttr().GetConnections()
    assert [str(p) for p in surf_sources] == ["/root/Materials/mat_body/PreviewSurface.outputs:surface"]

    # Dummy 1e-8 mass replaced with >= 0.02 kg on all links.
    for link_name in ("Microwave_4", "Microwave_4_bodymesh", "Microwave_4_doormesh"):
        m = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/root/{link_name}")).GetMassAttr().Get()
        assert m >= T.MIN_LINK_MASS_KG - 1e-6

    # Centred at origin.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    assert max(abs(v) for v in rng.GetMidpoint()) < 1e-4

    meta = json.loads((target / "metadata.json").read_text())
    assert meta["base_link"] == "Microwave_4"
    assert meta["body_joints"][0]["role"] == "door"
    assert meta["body_joints"][0]["drive"]["stiffness"] == 0.0
    assert meta["body_joints"][0]["friction"]["armature"] == pytest.approx(T.DEFAULT_ANGULAR_ARMATURE)
    assert meta["adapter"]["joints_canonicalized"] == 2


def test_pxr_adapt_thor_rigid(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    usda = tmp_path / "sources/assets/Alarm_Clock_1/Alarm_Clock_1.usda"
    usda.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    top = UsdGeom.Xform.Define(stage, "/Alarm_Clock_1")
    stage.SetDefaultPrim(top.GetPrim())
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    UsdGeom.Scope.Define(stage, "/Alarm_Clock_1/Geometry")
    body = UsdGeom.Xform.Define(stage, "/Alarm_Clock_1/Geometry/Alarm_Clock_1")
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.06))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Alarm_Clock_1/Geometry/Alarm_Clock_1/geom")
    cube.CreateSizeAttr(0.12)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateDensityAttr(200.0)
    stage.GetRootLayer().Save()

    manifest = T.survey(tmp_path / "sources")
    assert len(manifest["chosen"]) == 1
    assert manifest["chosen"][0]["id"] == "Rigid/thor_alarm_clock/00000"

    out_root = tmp_path / "out"
    conv = T.convert(tmp_path / "sources", out_root, manifest)
    assert len(conv["converted"]) == 1 and conv["failed"] == []

    target = out_root / "Assets/Object/RoboDojo/Rigid/thor_alarm_clock/00000"
    reopened = Usd.Stage.Open(str(target / "object.usd"))
    assert not reopened.GetPrimAtPath("/PhysicsScene")
    bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert bodies == ["/root"]
    m = UsdPhysics.MassAPI(reopened.GetPrimAtPath("/root")).GetMassAttr().Get()
    assert T.DEFAULT_MIN_RIGID_MASS_KG <= m <= T.DEFAULT_MAX_RIGID_MASS_KG
