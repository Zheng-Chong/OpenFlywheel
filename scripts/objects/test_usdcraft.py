"""CPU unit tests for scripts/objects/usdcraft.py (no Isaac; pxr tests guarded with importorskip)."""

from __future__ import annotations

import email.message
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


U = _load("usdcraft")


def _write_manifest_csv(path: Path, rows: list[dict]) -> Path:
    cols = [
        "object_id",
        "path",
        "group",
        "category_id",
        "category",
        "title",
        "physics_type",
        "part_count",
        "joint_count",
        "joint_types",
        "has_textures",
        "physx_sha256",
        "prompt",
    ]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = str(r.get(c, ""))
            if "," in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_helpers_and_load_manifest_csv(tmp_path):
    assert U._parse_bool("True") and U._parse_bool("1") and U._parse_bool(True)
    assert not U._parse_bool("false") and not U._parse_bool("0") and not U._parse_bool("")
    assert U._parse_joint_types('["revolute", "Prismatic"]') == ["revolute", "prismatic"]
    assert U._parse_joint_types("revolute;prismatic") == ["revolute", "prismatic"]
    assert U._parse_joint_types("revolute, spherical") == ["revolute", "spherical"]
    assert U._parse_joint_types("") == []

    csv_path = _write_manifest_csv(
        tmp_path / "manifest.csv",
        [
            {
                "object_id": "R001_01",
                "path": "rigid/kitchen_dining/R001_microwave/01_silver_microwave",
                "group": "kitchen_dining",
                "category_id": "R001",
                "category": "microwave",
                "title": "silver_microwave",
                "physics_type": "articulated",
                "part_count": 3,
                "joint_count": 2,
                "joint_types": '["revolute", "prismatic"]',
                "has_textures": "True",
                "physx_sha256": "abc123",
                "prompt": "A silver countertop microwave",
            },
            {"object_id": "EMPTY", "path": ""},  # skipped: empty path
        ],
    )
    rows = U.load_manifest_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["object_id"] == "R001_01"
    assert rows[0]["has_textures"] is True
    assert rows[0]["joint_types"] == ["revolute", "prismatic"]
    assert rows[0]["part_count"] == 3 and rows[0]["joint_count"] == 2


def test_filter_manifest_criteria_and_deterministic_ids():
    rows = [
        {
            "object_id": "R001_02",
            "path": "rigid/kitchen_dining/R001_microwave/02_black",
            "group": "kitchen_dining",
            "category_id": "R001",
            "category": "microwave",
            "title": "black",
            "physics_type": "articulated",
            "part_count": 2,
            "joint_count": 1,
            "joint_types": ["revolute"],
            "has_textures": True,
            "physx_sha256": "",
            "prompt": "",
        },
        {
            "object_id": "R001_01",
            "path": "rigid/kitchen_dining/R001_microwave/01_white",
            "group": "kitchen_dining",
            "category_id": "R001",
            "category": "microwave",
            "title": "white",
            "physics_type": "articulated",
            "part_count": 2,
            "joint_count": 1,
            "joint_types": ["revolute"],
            "has_textures": True,
            "physx_sha256": "",
            "prompt": "",
        },
        {
            "object_id": "R002_01",
            "path": "rigid/kitchen_dining/R002_globe/01_globe",
            "group": "kitchen_dining",
            "category_id": "R002",
            "category": "globe",
            "title": "globe",
            "physics_type": "articulated",
            "part_count": 2,
            "joint_count": 1,
            "joint_types": ["spherical"],  # lacks revolute/prismatic -> dropped
            "has_textures": True,
            "physx_sha256": "",
            "prompt": "",
        },
        {
            "object_id": "R003_01",
            "path": "rigid/bath_laundry_hardware/R003_cabinet/01_cab",
            "group": "bath_laundry_hardware",
            "category_id": "R003",
            "category": "cabinet",
            "title": "cab",
            "physics_type": "articulated",
            "part_count": 3,
            "joint_count": 2,
            "joint_types": ["revolute"],
            "has_textures": False,
            "physx_sha256": "",
            "prompt": "",
        },
        {
            "object_id": "R004_01",
            "path": "rigid/kitchen_dining/R004_mug/01_mug",
            "group": "kitchen_dining",
            "category_id": "R004",
            "category": "mug",
            "title": "mug",
            "physics_type": "static_rigid",
            "part_count": 1,
            "joint_count": 0,
            "joint_types": [],
            "has_textures": True,
            "physx_sha256": "",
            "prompt": "",
        },
    ]

    # Textured articulated only: R001_01 and R001_02 chosen, sorted deterministically.
    sel = U.filter_manifest(rows, physics_type="articulated", textured_only=True)
    assert [c["object_id"] for c in sel["chosen"]] == ["R001_01", "R001_02"]
    assert [c["id"] for c in sel["chosen"]] == [
        "Articulation/usdcraft_microwave/00000",
        "Articulation/usdcraft_microwave/00001",
    ]
    dropped_reasons = {d["object_id"]: d["reason"] for d in sel["dropped"]}
    assert "spherical" in dropped_reasons["R002_01"]
    assert dropped_reasons["R003_01"] == "untextured"
    assert "physics_type" in dropped_reasons["R004_01"]

    # max_per_category=1 caps microwaves to 1.
    sel_cap = U.filter_manifest(rows, physics_type="articulated", textured_only=True, max_per_category=1)
    assert [c["object_id"] for c in sel_cap["chosen"]] == ["R001_01"]

    # physics_type="all" includes static_rigid mug and starts its Rigid counter at 00000.
    sel_all = U.filter_manifest(rows, physics_type="all", textured_only=True)
    by_oid = {c["object_id"]: c["id"] for c in sel_all["chosen"]}
    assert by_oid["R004_01"] == "Rigid/usdcraft_mug/00000"
    assert by_oid["R001_01"] == "Articulation/usdcraft_microwave/00000"

    # Empty input produces empty chosen/dropped.
    empty = U.filter_manifest([])
    assert empty["chosen"] == [] and empty["dropped"] == []


def test_hf_endpoint_and_http_get_retries_and_user_agent(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert U.hf_endpoint() == "https://hf-mirror.com"
    assert U.dataset_url() == "https://hf-mirror.com/datasets/nvidia/USDCraft-10K"

    seen_uas: list[str] = []
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
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "5"
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, io.BytesIO(b""))
        if calls["n"] == 2:
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", email.message.Message(), io.BytesIO()
            )
        return _FakeResp(b"ok-bytes")

    monkeypatch.setattr(U.urllib.request, "urlopen", fake_urlopen)
    data, _ = U.http_get("https://hf-mirror.com/test", attempts=4, backoff=1.0, sleep_fn=sleeps.append)
    assert data == b"ok-bytes"
    assert seen_uas == ["curl/8.0", "curl/8.0", "curl/8.0"]
    assert sleeps == [5.0, 2.0]  # honors Retry-After=5 on attempt 0, then 1.0 * 2^1 on attempt 1

    # 404 fails immediately without sleeping or retrying.
    calls_404 = {"n": 0}
    sleeps_404: list[float] = []

    def fake_404(req, timeout=120):
        calls_404["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", email.message.Message(), io.BytesIO())

    monkeypatch.setattr(U.urllib.request, "urlopen", fake_404)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        U.http_get("https://hf-mirror.com/missing", attempts=4, sleep_fn=sleeps_404.append)
    assert exc_info.value.code == 404
    assert calls_404["n"] == 1 and sleeps_404 == []


def test_fetch_packages_verifies_sha256_skips_cached_and_downloads_textures(tmp_path, monkeypatch):
    usdc_bytes = b"PXR-USDC-DUMMY-DATA"
    usdc_sha = hashlib.sha256(usdc_bytes).hexdigest()
    meta_bytes = json.dumps({"floating_part_count": 0, "category": "microwave"}).encode()
    tex_bytes = b"\x89PNG\r\n\x1a\nfake"
    pkg_rel = "rigid/kitchen_dining/R001_microwave/01_silver"

    served = {
        f"/resolve/main/{pkg_rel}/meta.json": meta_bytes,
        f"/resolve/main/{pkg_rel}/physx/model.usdc": usdc_bytes,
        f"/resolve/main/{pkg_rel}/physx/textures/albedo.png": tex_bytes,
    }
    tree_resp = json.dumps([{"type": "file", "path": f"{pkg_rel}/physx/textures/albedo.png", "size": len(tex_bytes)}])
    hits: list[str] = []

    def fake_http_get(url, **kwargs):
        hits.append(url)
        if "/api/datasets/" in url and "/tree/" in url:
            return tree_resp.encode(), {}
        for suffix, payload in served.items():
            if url.endswith(suffix):
                return payload, {}
        raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), io.BytesIO())

    monkeypatch.setattr(U, "http_get", fake_http_get)
    entry = {
        "object_id": "R001_01",
        "path": pkg_rel,
        "has_textures": True,
        "physx_sha256": usdc_sha,
    }
    res = U.fetch_packages([entry], tmp_path, revision="main", workers=1)
    assert len(res["fetched"]) == 1 and res["failed"] == []
    assert (tmp_path / pkg_rel / "meta.json").read_bytes() == meta_bytes
    assert (tmp_path / pkg_rel / "physx/model.usdc").read_bytes() == usdc_bytes
    assert (tmp_path / pkg_rel / "physx/textures/albedo.png").read_bytes() == tex_bytes
    marker = json.loads((tmp_path / pkg_rel / U.FETCH_MARKER).read_text())
    assert marker["revision"] == "main" and marker["has_textures"] is True
    assert set(marker["files"]) == {"meta.json", "physx/model.usdc", "physx/textures/albedo.png"}
    upstream = json.loads((tmp_path / "UPSTREAM.json").read_text())
    assert upstream["license"] == "Apache-2.0" and upstream["packages_fetched"] == 1

    # Second fetch fast-skips via `.fetched` without any HTTP calls.
    hits.clear()
    res2 = U.fetch_packages([entry], tmp_path, revision="main", workers=1)
    assert res2["fetched"][0]["downloaded_files"] == 0
    assert res2["fetched"][0].get("cached") is True
    assert hits == []

    # Partial download failure (texture fails after model.usdc succeeds) MUST NOT write `.fetched`
    # and `survey(..., selection=...)` MUST drop the half-fetched package.
    partial_dir = tmp_path / "partial"

    def fake_http_tex_fail(url, **kwargs):
        if "/api/datasets/" in url and "/tree/" in url:
            return tree_resp.encode(), {}
        if url.endswith("/textures/albedo.png"):
            raise TimeoutError("simulated texture timeout")
        for suffix, payload in served.items():
            if url.endswith(suffix):
                return payload, {}
        raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), io.BytesIO())

    monkeypatch.setattr(U, "http_get", fake_http_tex_fail)
    res_part = U.fetch_packages([entry], partial_dir, revision="main", workers=1)
    assert len(res_part["failed"]) == 1 and res_part["fetched"] == []
    assert (partial_dir / pkg_rel / "physx/model.usdc").is_file()
    assert not (partial_dir / pkg_rel / U.FETCH_MARKER).exists()
    ok_part, why_part = U.validate_package_complete(partial_dir / pkg_rel, entry=entry, require_fetched_marker=True)
    assert not ok_part and U.FETCH_MARKER in why_part
    survey_part = U.survey(partial_dir, selection={"revision": "main", "chosen": [entry]})
    assert survey_part["chosen"] == [] and len(survey_part["dropped"]) == 1
    assert U.FETCH_MARKER in survey_part["dropped"][0]["reason"]

    # Deleting a texture listed in `.fetched` is caught by `validate_package_complete` & `survey`.
    (tmp_path / pkg_rel / "physx/textures/albedo.png").unlink()
    ok_miss_tex, why_miss_tex = U.validate_package_complete(
        tmp_path / pkg_rel, entry=entry, require_fetched_marker=True
    )
    assert not ok_miss_tex and "albedo.png" in why_miss_tex

    # Bad sha256 fails cleanly and does not write corrupt file or `.fetched`.
    monkeypatch.setattr(U, "http_get", fake_http_get)
    bad_dir = tmp_path / "bad"
    bad_entry = {**entry, "physx_sha256": "0" * 64}
    res_bad = U.fetch_packages([bad_entry], bad_dir, revision="main", workers=1)
    assert len(res_bad["failed"]) == 1 and "sha256" in res_bad["failed"][0]["error"]
    assert not (bad_dir / pkg_rel / "physx/model.usdc").exists()
    assert not (bad_dir / pkg_rel / U.FETCH_MARKER).exists()

    # Truncated texture download (len(data) != expected_size) fails cleanly without writing corrupt file or `.fetched`.
    trunc_dir = tmp_path / "trunc"

    def fake_http_trunc(url, **kwargs):
        if "/api/datasets/" in url and "/tree/" in url:
            return tree_resp.encode(), {}
        if url.endswith("/textures/albedo.png"):
            return tex_bytes[:3], {}  # truncated!
        for suffix, payload in served.items():
            if url.endswith(suffix):
                return payload, {}
        raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), io.BytesIO())

    monkeypatch.setattr(U, "http_get", fake_http_trunc)
    # Seed a corrupt `.fetched` marker first to verify it is unlinked on re-fetch failure.
    (trunc_dir / pkg_rel).mkdir(parents=True, exist_ok=True)
    (trunc_dir / pkg_rel / U.FETCH_MARKER).write_text("{corrupt json", encoding="utf-8")
    res_trunc = U.fetch_packages([entry], trunc_dir, revision="main", workers=1)
    assert len(res_trunc["failed"]) == 1 and "size" in res_trunc["failed"][0]["error"]
    assert not (trunc_dir / pkg_rel / "physx/textures/albedo.png").exists()
    assert not (trunc_dir / pkg_rel / U.FETCH_MARKER).exists()

    # `.fetched` with has_textures=False fails `validate_package_complete` when entry requires has_textures=True.
    no_tex_dir = tmp_path / "no_tex_marker" / pkg_rel
    (no_tex_dir / "physx").mkdir(parents=True)
    (no_tex_dir / "meta.json").write_bytes(meta_bytes)
    (no_tex_dir / "physx/model.usdc").write_bytes(usdc_bytes)
    (no_tex_dir / U.FETCH_MARKER).write_text(
        json.dumps({"revision": "main", "has_textures": False, "files": ["meta.json", "physx/model.usdc"]}),
        encoding="utf-8",
    )
    ok_ht, why_ht = U.validate_package_complete(no_tex_dir, entry=entry, require_fetched_marker=True)
    assert not ok_ht and "has_textures" in why_ht


def test_promote_staged_dir_replaces_existing_and_removes_stale_textures(tmp_path):
    target = tmp_path / "Articulation/usdcraft_microwave/00000"
    (target / "textures").mkdir(parents=True)
    (target / "textures/stale.png").write_bytes(b"old-tex")
    (target / "object.usd").write_bytes(b"old-usd")

    stage1 = tmp_path / "Articulation/usdcraft_microwave/.00000.stage1.tmp"
    (stage1 / "textures").mkdir(parents=True)
    (stage1 / "textures/albedo.png").write_bytes(b"new-tex")
    (stage1 / "object.usd").write_bytes(b"new-usd")
    U._promote_staged_dir(stage1, target)
    assert not stage1.exists()
    assert (target / "object.usd").read_bytes() == b"new-usd"
    assert (target / "textures/albedo.png").read_bytes() == b"new-tex"
    assert not (target / "textures/stale.png").exists()
    assert not list(target.glob(".textures.old.*"))

    # Re-promoting an untextured stage removes the previous textures/ directory cleanly.
    stage2 = tmp_path / "Articulation/usdcraft_microwave/.00000.stage2.tmp"
    stage2.mkdir(parents=True)
    (stage2 / "object.usd").write_bytes(b"untextured-usd")
    U._promote_staged_dir(stage2, target)
    assert not stage2.exists()
    assert (target / "object.usd").read_bytes() == b"untextured-usd"
    assert not (target / "textures").exists()


def test_clamp_link_masses_overweight_underweight_missing_and_edge_cases():
    links = ["/root/base", "/root/door", "/root/knob"]
    vols = {"/root/base": 0.04, "/root/door": 0.005, "/root/knob": 0.0001}

    # 1. Overweight solid-volume mass (120 kg) clamped proportionally to 15 kg; knob stays >= MIN_LINK_MASS_KG.
    masses, info = U.clamp_link_masses(
        links,
        {"/root/base": 100.0, "/root/door": 19.95, "/root/knob": 0.05},
        vols,
        max_total_mass=15.0,
    )
    assert info["clamped"] is True and info["raw_authored_kg"] == pytest.approx(120.0)
    assert sum(masses.values()) == pytest.approx(15.0, rel=1e-4)
    assert masses["/root/knob"] >= U.MIN_LINK_MASS_KG
    assert masses["/root/base"] > masses["/root/door"] > masses["/root/knob"]

    # 2. In-range authored masses preserved untouched.
    masses_ok, info_ok = U.clamp_link_masses(
        links,
        {"/root/base": 4.0, "/root/door": 0.8, "/root/knob": 0.05},
        vols,
        max_total_mass=15.0,
    )
    assert info_ok["clamped"] is False
    assert masses_ok == {"/root/base": 4.0, "/root/door": 0.8, "/root/knob": 0.05}

    # 3. Underweight total bumped to min_total_mass.
    masses_low, info_low = U.clamp_link_masses(
        ["/root/base", "/root/lid"],
        {"/root/base": 0.005, "/root/lid": 0.005},
        {"/root/base": 0.01, "/root/lid": 0.002},
        max_total_mass=3.0,
        min_total_mass=0.05,
    )
    assert info_low["clamped"] is True and sum(masses_low.values()) == pytest.approx(0.05, rel=1e-4)

    # 4. Missing / NaN / zero authored masses filled by volume.
    masses_fill, info_fill = U.clamp_link_masses(
        links,
        {"/root/base": 2.0, "/root/door": float("nan"), "/root/knob": 0.0},
        vols,
        max_total_mass=15.0,
    )
    assert info_fill["filled_missing_links"] is True
    assert all(m >= U.MIN_LINK_MASS_KG for m in masses_fill.values())
    assert masses_fill["/root/door"] >= masses_fill["/root/knob"]

    # 5. Many-link edge case where min_link_mass * N > max_total_mass does not crash.
    many = [f"/root/l_{i}" for i in range(40)]
    masses_many, _ = U.clamp_link_masses(
        many,
        {p: 1.0 for p in many},
        {p: 0.001 for p in many},
        max_total_mass=0.5,
        min_total_mass=0.05,
    )
    assert sum(masses_many.values()) == pytest.approx(0.5, rel=1e-4)

    # 6. Regression: partial-authored & near-cap under-floor links never violate max_total_mass or min_link_mass.
    m_cap1, info_cap1 = U.clamp_link_masses(
        links, {"/root/base": 0.99, "/root/door": 0.001}, vols, max_total_mass=1.0, min_link_mass=0.02
    )
    assert info_cap1["clamped"] is True
    assert sum(m_cap1.values()) <= 1.0 + 1e-9
    assert all(m >= 0.02 - 1e-12 for m in m_cap1.values())

    m_cap2, _ = U.clamp_link_masses(
        links, {"/root/base": 0.5, "/root/door": 0.001}, vols, max_total_mass=15.0, min_link_mass=0.02
    )
    assert all(m >= 0.02 - 1e-12 for m in m_cap2.values())
    assert sum(m_cap2.values()) <= 15.0 + 1e-9

    m_cap3, info_cap3 = U.clamp_link_masses(
        ["/root/base", "/root/door"],
        {"/root/base": 14.99, "/root/door": 0.001},
        vols,
        max_total_mass=15.0,
        min_link_mass=0.02,
    )
    assert info_cap3["clamped"] is True
    assert sum(m_cap3.values()) <= 15.0 + 1e-9
    assert all(m >= 0.02 - 1e-12 for m in m_cap3.values())

    with pytest.raises(ValueError):
        U.clamp_link_masses([], {}, {}, max_total_mass=1.0)


def test_decide_articulated_and_rigid_and_heuristics():
    facts = {
        "up_axis": "Z",
        "movable": [{"name": "door_joint", "body1": "/Asset/Links/door_link", "type": "revolute"}],
        "world_movable": [],
        "bases": ["/Asset/Links/base_link"],
        "colliders": 2,
    }
    assert U.decide_articulated({"floating_part_count": 0}, [0.5, 0.4, 0.3], facts)[0]
    ok_float, why_float = U.decide_articulated({"floating_part_count": 1}, [0.5, 0.4, 0.3], facts)
    assert not ok_float and "floating_part_count=1" in why_float
    assert U.decide_articulated({"floating_part_count": 1}, [0.5, 0.4, 0.3], facts, allow_floating=True)[0]
    assert not U.decide_articulated({}, [0.5, 0.4, 0.3], {**facts, "bases": ["/a", "/b"]})[0]
    assert not U.decide_articulated({}, [0.5, 0.4, 0.3], {**facts, "movable": []})[0]
    assert not U.decide_articulated({}, [0.5, 0.4, 0.3], {**facts, "colliders": 0})[0]
    assert not U.decide_articulated({}, [1.5, 0.5, 0.5], facts)[0]
    assert not U.decide_articulated({}, [0.5, 0.4, 0.001], facts)[0]
    twins = {**facts, "movable": [{"name": "J", "body1": "/a", "type": "revolute"}] * 2}
    assert not U.decide_articulated({}, [0.5, 0.4, 0.3], twins)[0]

    # Rigid decisions
    rigid_facts = {"up_axis": "Z", "colliders": 1}
    assert U.decide_rigid({"floating_part_count": 0, "tips_over": False}, [0.1, 0.1, 0.12], rigid_facts)[0]
    assert not U.decide_rigid({"tips_over": True}, [0.1, 0.1, 0.12], rigid_facts)[0]
    assert U.decide_rigid({"tips_over": True}, [0.1, 0.1, 0.12], rigid_facts, drop_tips_over=False)[0]
    assert not U.decide_rigid({"floating_part_count": 2}, [0.1, 0.1, 0.12], rigid_facts)[0]

    # fixed_base heuristic & collision-free naming
    assert U.is_fixed_base("microwave", [0.5, 0.4, 0.3], "kitchen_dining") is True
    assert U.is_fixed_base("scissors", [0.2, 0.08, 0.01], "office_commercial_hospitality") is False
    assert U.is_fixed_base("laptop", [0.35, 0.25, 0.02], "living_bedroom_electronics_toys") is False
    assert U.resolve_collision_free_name("base_link", {"Joints", "Looks"}) == "base_link"
    assert U.resolve_collision_free_name("Looks", {"Joints", "Looks"}) == "link_Looks"
    assert U.resolve_collision_free_name("Looks", {"Joints", "Looks", "link_Looks"}) == "link_Looks_1"


def test_stamp_ready_articulated_and_rigid(tmp_path):
    art_id = "Articulation/usdcraft_microwave/00000"
    art_dir = tmp_path / "Assets/Object/RoboDojo" / art_id
    art_dir.mkdir(parents=True)
    (art_dir / "metadata.json").write_text(json.dumps({"runtime_ready": False, "source": {}, "adapter": {}}))

    check_dir = tmp_path / "checks"
    art_check = check_dir / art_id / "preview" / "check.json"
    art_check.parent.mkdir(parents=True)
    art_check.write_text(json.dumps({"pass": True, "checks": {"fixed_base": "pass", "joints": "pass"}}))

    ok, why = U.stamp_ready(tmp_path, art_id, check_dir, note="pilot")
    assert ok is True and why == []
    meta = json.loads((art_dir / "metadata.json").read_text())
    assert meta["runtime_ready"] is True
    assert meta["source"]["runtime_validated"] is True
    assert meta["adapter"]["validation"]["note"] == "pilot"

    # Failing articulated check records failure reasons (including `rest` and fallback `unknown`).
    art_check.write_text(json.dumps({"pass": False, "checks": {"fixed_base": "pass", "settle": "fail"}}))
    ok2, why2 = U.stamp_ready(tmp_path, art_id, check_dir)
    assert ok2 is False and why2 == ["settle"]
    assert json.loads((art_dir / "metadata.json").read_text())["runtime_ready"] is False

    art_check.write_text(json.dumps({"pass": False, "checks": {"fixed_base": "pass", "rest": "fail"}}))
    ok_rest, why_rest = U.stamp_ready(tmp_path, art_id, check_dir)
    assert ok_rest is False and why_rest == ["rest"]

    art_check.write_text(json.dumps({"pass": False, "checks": {}}))
    ok_unk, why_unk = U.stamp_ready(tmp_path, art_id, check_dir)
    assert ok_unk is False and why_unk == ["unknown"]

    # Rigid object pass / missing check
    rigid_id = "Rigid/usdcraft_mug/00000"
    rigid_dir = tmp_path / "Assets/Object/RoboDojo" / rigid_id
    rigid_dir.mkdir(parents=True)
    (rigid_dir / "metadata.json").write_text(json.dumps({"runtime_ready": False, "source": {}, "adapter": {}}))
    ok_missing, why_missing = U.stamp_ready(tmp_path, rigid_id, check_dir)
    assert ok_missing is False and why_missing == ["not_checked"]

    rigid_check = check_dir / rigid_id / "preview" / "check.json"
    rigid_check.parent.mkdir(parents=True)
    rigid_check.write_text(json.dumps({"settle": "pass", "tilt_deg": 0.1, "z_m": 0.05}))
    ok_r, why_r = U.stamp_ready(tmp_path, rigid_id, check_dir)
    assert ok_r is True and why_r == []


def test_cli_filter_and_ready_smoke(tmp_path):
    csv_path = _write_manifest_csv(
        tmp_path / "manifest.csv",
        [
            {
                "object_id": "R001_01",
                "path": "rigid/kitchen_dining/R001_microwave/01_silver",
                "group": "kitchen_dining",
                "category_id": "R001",
                "category": "microwave",
                "title": "silver",
                "physics_type": "articulated",
                "part_count": 2,
                "joint_count": 1,
                "joint_types": '["revolute"]',
                "has_textures": "True",
                "physx_sha256": "",
                "prompt": "microwave",
            }
        ],
    )
    sel_out = tmp_path / "sel.json"
    rc = U.main(["filter", "--manifest-csv", str(csv_path), "--textured-only", "--out", str(sel_out)])
    assert rc == 0
    sel = json.loads(sel_out.read_text())
    assert sel["summary"]["chosen"] == 1

    # Missing explicit --manifest-csv without --sources-dir fails fast.
    with pytest.raises(FileNotFoundError):
        U.main(["filter", "--manifest-csv", str(tmp_path / "does_not_exist.csv"), "--out", str(sel_out)])


def test_survey_records_missing_packages_and_list_hf_tree_relative_pagination(tmp_path, monkeypatch):
    # Missing physx/model.usdc from a selection is recorded in `dropped` rather than silently omitted.
    sel = {
        "revision": "main",
        "chosen": [{"object_id": "R001_01", "path": "rigid/kitchen_dining/R001_microwave/01_missing"}],
    }
    res = U.survey(tmp_path, selection=sel)
    assert res["chosen"] == []
    assert len(res["dropped"]) == 1
    assert "missing physx/model.usdc" in res["dropped"][0]["reason"]

    # Relative `Link` header pagination resolves cleanly against endpoint base.
    seen_urls: list[str] = []

    def fake_get(url, **kwargs):
        seen_urls.append(url)
        if len(seen_urls) == 1:
            body = json.dumps([{"type": "file", "path": "a/tex1.png", "size": 10}]).encode()
            return body, {"link": '</api/datasets/nvidia/USDCraft-10K/tree/main/a?cursor=p2>; rel="next"'}
        body = json.dumps([{"type": "file", "path": "a/tex2.png", "size": 20}]).encode()
        return body, {}

    monkeypatch.setattr(U, "http_get", fake_get)
    files = U.list_hf_tree("a", endpoint="https://hf-mirror.com")
    assert files == [("a/tex1.png", 10), ("a/tex2.png", 20)]
    assert seen_urls[1] == "https://hf-mirror.com/api/datasets/nvidia/USDCraft-10K/tree/main/a?cursor=p2"


def test_usdcraft_stdlib_only_import_without_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    spec = importlib.util.spec_from_file_location("usdcraft_no_numpy", Path(__file__).with_name("usdcraft.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.hf_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"


def _build_usdcraft_articulated_stage(usdc_path: Path, tex_path: Path):
    """Build a synthetic USDCraft-10K articulated stage matching live pxr inspection findings."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    usdc_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-texture")

    stage = Usd.Stage.CreateNew(str(usdc_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.Scope.Define(stage, "/Asset/Links")
    UsdGeom.Scope.Define(stage, "/Asset/Joints")
    UsdGeom.Scope.Define(stage, "/Asset/Looks")
    UsdGeom.Scope.Define(stage, "/Asset/PhysicsMaterials")

    # Material with UsdPreviewSurface + external relative texture
    mat = UsdShade.Material.Define(stage, "/Asset/Looks/mat_body")
    shader = UsdShade.Shader.Define(stage, "/Asset/Looks/mat_body/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    tex_shader = UsdShade.Shader.Define(stage, "/Asset/Looks/mat_body/AlbedoTex")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/albedo.png"))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Base link (intentionally placed AFTER door_link to test reordering) and door link under /Asset/Links/
    for name, centre, size, mass_kg, diag in (
        ("door_link", (0.21, -0.15, 0.15), (0.02, 0.30, 0.30), 20.0, (0.2, 0.2, 0.2)),
        ("base_link", (0.0, 0.0, 0.15), (0.40, 0.35, 0.30), 100.0, (2.0, 2.0, 2.0)),
    ):
        link = UsdGeom.Xform.Define(stage, f"/Asset/Links/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(*centre))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        m_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
        m_api.CreateMassAttr(mass_kg)
        m_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
        cube = UsdGeom.Cube.Define(stage, f"/Asset/Links/{name}/geom")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    base = stage.GetPrimAtPath("/Asset/Links/base_link")
    UsdPhysics.ArticulationRootAPI.Apply(base)

    # Joint under /Asset/Joints with authored DriveAPI + PhysxJointAxisAPI friction & armature
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/Joints/door_hinge")
    joint.CreateBody0Rel().SetTargets(["/Asset/Links/base_link"])
    joint.CreateBody1Rel().SetTargets(["/Asset/Links/door_link"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(110.0)
    drv = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(0.0)
    drv.CreateDampingAttr(0.5)
    drv.CreateTargetPositionAttr(0.0)
    joint.GetPrim().AddAppliedSchema("PhysxJointAxisAPI:angular")
    joint.GetPrim().CreateAttribute("physxJointAxis:angular:staticFrictionEffort", Sdf.ValueTypeNames.Float).Set(0.75)
    joint.GetPrim().CreateAttribute("physxJointAxis:angular:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(0.55)
    joint.GetPrim().CreateAttribute("physxJointAxis:angular:armature", Sdf.ValueTypeNames.Float).Set(0.01)

    stage.GetRootLayer().Save()


def test_pxr_adapt_articulated_reparents_clamps_mass_and_preserves_friction(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    pkg_rel = "rigid/kitchen_dining/R001_microwave/01_silver"
    pkg_dir = tmp_path / "sources" / pkg_rel
    usdc = pkg_dir / "physx" / "model.usdc"
    tex = pkg_dir / "physx" / "textures" / "albedo.png"
    _build_usdcraft_articulated_stage(usdc, tex)
    (pkg_dir / "meta.json").write_text(
        json.dumps({"floating_part_count": 0, "category": "microwave", "physics_type": "articulated", "prompt": "mw"})
    )

    manifest = U.survey(tmp_path / "sources")
    assert len(manifest["chosen"]) == 1 and manifest["dropped"] == []

    out_root = tmp_path / "out"
    conv = U.convert(tmp_path / "sources", out_root, manifest)
    assert len(conv["converted"]) == 1 and conv["failed"] == []

    oid = manifest["chosen"][0]["id"]
    target = out_root / "Assets/Object/RoboDojo" / oid
    assert (target / "object.usd").is_file()
    assert (target / "textures/albedo.png").is_file()

    stage = Usd.Stage.Open(str(target / "object.usd"))
    root = stage.GetDefaultPrim()
    assert str(root.GetPath()) == "/root"
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)

    # First direct child of /root MUST be base_link Xform with RigidBodyAPI + PhysxArticulationAPI.
    children = root.GetChildren()
    assert children[0].GetName() == "base_link"
    assert children[0].GetTypeName() == "Xform"
    assert children[0].HasAPI(UsdPhysics.RigidBodyAPI)
    assert not children[0].HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/root/Links")

    # Joint targets rewritten to /root/base_link and /root/door_link; authored PhysxJointAxisAPI friction preserved!
    hinge = stage.GetPrimAtPath("/root/Joints/door_hinge")
    j = UsdPhysics.Joint(hinge)
    assert [str(p) for p in j.GetBody0Rel().GetTargets()] == ["/root/base_link"]
    assert [str(p) for p in j.GetBody1Rel().GetTargets()] == ["/root/door_link"]
    assert hinge.GetAttribute("physxJointAxis:angular:staticFrictionEffort").Get() == pytest.approx(0.75)
    assert hinge.GetAttribute("physxJointAxis:angular:dynamicFrictionEffort").Get() == pytest.approx(0.55)
    assert hinge.GetAttribute("physxJointAxis:angular:armature").Get() == pytest.approx(0.01)

    # Regression: Material binding AND AttributeSpec.connectionPathList MUST be rewritten from /Asset -> /root
    # so UsdShade material outputs:surface and PreviewSurface inputs:diffuseColor stay connected (no gray model!).
    geom_prim = stage.GetPrimAtPath("/root/base_link/geom")
    bound_mat, _ = UsdShade.MaterialBindingAPI(geom_prim).ComputeBoundMaterial()
    assert str(bound_mat.GetPath()) == "/root/Looks/mat_body"
    surf_src = bound_mat.GetSurfaceOutput().GetConnectedSource()
    assert surf_src is not None
    assert str(surf_src[0].GetPrim().GetPath()) == "/root/Looks/mat_body/PreviewSurface"
    assert surf_src[1] == "surface"
    prev_shader = UsdShade.Shader(stage.GetPrimAtPath("/root/Looks/mat_body/PreviewSurface"))
    diff_src = prev_shader.GetInput("diffuseColor").GetConnectedSource()
    assert diff_src is not None
    assert str(diff_src[0].GetPrim().GetPath()) == "/root/Looks/mat_body/AlbedoTex"
    assert diff_src[1] == "rgb"
    tex_shader = UsdShade.Shader(stage.GetPrimAtPath("/root/Looks/mat_body/AlbedoTex"))
    assert tex_shader.GetInput("file").Get().path == "./textures/albedo.png"

    # Zero stale `/Asset` references anywhere in relationships or attribute connections.
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            assert all(not str(t).startswith("/Asset") for t in rel.GetTargets()), f"stale rel on {prim.GetPath()}"
        for attr in prim.GetAttributes():
            assert all(not str(c).startswith("/Asset") for c in attr.GetConnections()), (
                f"stale conn on {attr.GetPath()}"
            )

    # Heavy 120 kg mass clamped to <= 15 kg and diagonalInertia scaled in lockstep.
    base_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/root/base_link")).GetMassAttr().Get()
    door_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/root/door_link")).GetMassAttr().Get()
    assert base_mass + door_mass == pytest.approx(U.DEFAULT_MAX_TABLETOP_MASS_KG, rel=1e-4)
    base_diag = UsdPhysics.MassAPI(stage.GetPrimAtPath("/root/base_link")).GetDiagonalInertiaAttr().Get()
    assert base_diag[0] == pytest.approx(2.0 * (base_mass / 100.0), rel=1e-4)

    # Centred at origin.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    assert max(abs(v) for v in rng.GetMidpoint()) < 1e-4

    meta = json.loads((target / "metadata.json").read_text())
    assert meta["base_link"] == "base_link"
    assert meta["runtime_ready"] is False
    assert meta["body_joints"][0]["friction"]["static"] == pytest.approx(0.75)
    assert meta["adapter"]["mass"]["clamped"] is True


def test_pxr_adapt_rigid_strips_child_bodies_and_clamps_mass(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    pkg_rel = "rigid/kitchen_dining/R004_mug/01_ceramic"
    pkg_dir = tmp_path / "sources" / pkg_rel
    usdc = pkg_dir / "physx" / "model.usdc"
    tex = pkg_dir / "physx" / "textures" / "mug.png"
    usdc.parent.mkdir(parents=True, exist_ok=True)
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_bytes(b"\x89PNG\r\n\x1a\nmug")

    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.Scope.Define(stage, "/Asset/Links")
    UsdGeom.Scope.Define(stage, "/Asset/Looks")
    mat = UsdShade.Material.Define(stage, "/Asset/Looks/mat_mug")
    sh = UsdShade.Shader.Define(stage, "/Asset/Looks/mat_mug/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    tx = UsdShade.Shader.Define(stage, "/Asset/Looks/mat_mug/AlbedoTex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/mug.png"))
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    body = UsdGeom.Xform.Define(stage, "/Asset/Links/body")
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.05))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(5.0)
    cube = UsdGeom.Cube.Define(stage, "/Asset/Links/body/geom")
    cube.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()

    (pkg_dir / "meta.json").write_text(
        json.dumps({"floating_part_count": 0, "tips_over": False, "category": "mug", "physics_type": "static_rigid"})
    )

    manifest = U.survey(tmp_path / "sources")
    assert len(manifest["chosen"]) == 1
    out_root = tmp_path / "out"
    conv = U.convert(tmp_path / "sources", out_root, manifest)
    assert len(conv["converted"]) == 1 and conv["failed"] == []

    target = out_root / "Assets/Object/RoboDojo" / manifest["chosen"][0]["id"]
    reopened = Usd.Stage.Open(str(target / "object.usd"))
    bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert bodies == ["/root"]
    assert UsdPhysics.MassAPI(reopened.GetPrimAtPath("/root")).GetMassAttr().Get() == pytest.approx(
        U.DEFAULT_MAX_RIGID_MASS_KG
    )
    surf_src = UsdShade.Material(reopened.GetPrimAtPath("/root/Looks/mat_mug")).GetSurfaceOutput().GetConnectedSource()
    assert surf_src is not None and str(surf_src[0].GetPrim().GetPath()) == "/root/Looks/mat_mug/PreviewSurface"


def test_pxr_adapt_atomic_cleanup_on_unresolved_dependency(tmp_path):
    """If `ComputeAllDependencies` fails, neither `object.usd` nor staging temp dirs are left behind."""
    pytest.importorskip("pxr")

    pkg_rel = "rigid/kitchen_dining/R001_microwave/02_missing_tex"
    pkg_dir = tmp_path / "sources" / pkg_rel
    usdc = pkg_dir / "physx" / "model.usdc"
    tex = pkg_dir / "physx" / "textures" / "albedo.png"
    _build_usdcraft_articulated_stage(usdc, tex)
    # Remove texture so `./textures/albedo.png` becomes unresolved during export validation.
    tex.unlink()

    target_art = tmp_path / "out" / "Articulation/usdcraft_microwave/00000"
    with pytest.raises(ValueError, match="unresolved asset dependencies"):
        U.adapt_articulated(usdc, target_art, category="microwave")
    assert not target_art.exists()
    assert not list(target_art.parent.glob("*.tmp"))

    target_rigid = tmp_path / "out" / "Rigid/usdcraft_microwave/00000"
    with pytest.raises(ValueError, match="unresolved asset dependencies"):
        U.adapt_rigid(usdc, target_rigid)
    assert not target_rigid.exists()
    assert not list(target_rigid.parent.glob("*.tmp"))


def test_filter_manifest_fixed_and_continuous_joints():
    rows = [
        {
            "object_id": "R069_05",
            "path": "rigid/kitchen_dining/R069_chef_knife/05_gyuto",
            "group": "kitchen_dining",
            "category_id": "R069",
            "category": "Chef Knife",
            "title": "gyuto",
            "physics_type": "articulated",
            "part_count": 3,
            "joint_count": 2,
            "joint_types": [],
            "has_textures": True,
            "physx_sha256": "",
            "prompt": "",
        },
        {
            "object_id": "R055_02",
            "path": "rigid/kitchen_dining/R055_kitchen_timer/02_tomato",
            "group": "kitchen_dining",
            "category_id": "R055",
            "category": "Kitchen Timer",
            "title": "tomato",
            "physics_type": "articulated",
            "part_count": 3,
            "joint_count": 2,
            "joint_types": ["continuous"],
            "has_textures": True,
            "physx_sha256": "",
            "prompt": "",
        },
    ]
    sel_fixed = U.filter_manifest(rows, physics_type="articulated", textured_only=True, joint_types=["fixed"])
    assert [c["object_id"] for c in sel_fixed["chosen"]] == ["R069_05"]

    # Explicit "fixed" in joint_types list should also match joint_types=["fixed"]
    rows_explicit_fixed = [{**rows[0], "object_id": "R069_06", "joint_types": ["fixed"]}]
    sel_explicit = U.filter_manifest(
        rows_explicit_fixed, physics_type="articulated", textured_only=True, joint_types=["fixed"]
    )
    assert [c["object_id"] for c in sel_explicit["chosen"]] == ["R069_06"]

    sel_cont = U.filter_manifest(rows, physics_type="articulated", textured_only=True, joint_types=["continuous"])
    assert [c["object_id"] for c in sel_cont["chosen"]] == ["R055_02"]

    sel_all = U.filter_manifest(rows, physics_type="articulated", textured_only=True, joint_types=["all"])
    assert sorted(c["object_id"] for c in sel_all["chosen"]) == ["R055_02", "R069_05"]


def test_survey_fixed_as_rigid_and_existing_state_offset(tmp_path, monkeypatch):
    src = tmp_path / "sources"
    p_fixed = src / "rigid/kitchen_dining/R069_chef_knife/05_gyuto"
    p_cont = src / "rigid/kitchen_dining/R055_kitchen_timer/02_tomato"
    p_dup = src / "rigid/kitchen_dining/R055_kitchen_timer/01_existing"
    for p in (p_fixed, p_cont, p_dup):
        (p / "physx").mkdir(parents=True)
        (p / "physx" / "model.usdc").write_bytes(b"USD")
        (p / "meta.json").write_text(json.dumps({"floating_part_count": 0, "tips_over": False}))

    def fake_survey_package(pkg_dir: Path):
        if "chef_knife" in str(pkg_dir):
            facts = {
                "up_axis": "Z",
                "movable": [],
                "world_movable": [],
                "bases": ["/Asset/Links/base"],
                "colliders": ["/Asset/Links/base/col"],
            }
            return [0.25, 0.05, 0.02], facts, {"floating_part_count": 0, "tips_over": False, "category": "Chef Knife"}
        facts = {
            "up_axis": "Z",
            "movable": [{"name": "dial_joint", "type": "revolute", "body1": "/Asset/Links/dial"}],
            "world_movable": [],
            "bases": ["/Asset/Links/base"],
            "colliders": ["/Asset/Links/base/col"],
        }
        return (
            [0.08, 0.08, 0.08],
            facts,
            {
                "floating_part_count": 0,
                "tips_over": False,
                "category": "Kitchen Timer",
                "object_id": "R055_01" if "01_existing" in str(pkg_dir) else "R055_02",
            },
        )

    monkeypatch.setattr(U, "survey_package", fake_survey_package)
    res = U.survey(
        src,
        fixed_as_rigid=True,
        existing_sources={"R055_01": "Articulation/usdcraft_kitchen_timer/00003"},
        existing_max_index={"Articulation/usdcraft_kitchen_timer": 3},
    )
    by_src = {c["src"]: c for c in res["chosen"]}
    assert by_src["rigid/kitchen_dining/R069_chef_knife/05_gyuto"]["id"] == "Rigid/usdcraft_chef_knife/00000"
    assert by_src["rigid/kitchen_dining/R069_chef_knife/05_gyuto"]["physics_type"] == "static_rigid"
    timer_entry = by_src["rigid/kitchen_dining/R055_kitchen_timer/02_tomato"]
    assert timer_entry["id"] == "Articulation/usdcraft_kitchen_timer/00004"
    assert any("already imported" in d["reason"] for d in res["dropped"])
