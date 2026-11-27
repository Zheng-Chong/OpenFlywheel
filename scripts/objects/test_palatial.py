"""CPU unit tests for scripts/objects/palatial.py (no Isaac; pxr tests guarded with importorskip)."""

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


P = _load("palatial")


def _sample_catalog_rows() -> list[dict]:
    return [
        {
            "catalog_id": "batch10_D00006_Bamboo_Basket",
            "asset_id": "D00006",
            "logical_asset": "D00006",
            "name": "Bamboo Basket",
            "display_name": "Bamboo Basket (D00006)",
            "class_label": "basket",
            "wikidata_qcode": "Q201097",
            "batch": "batch10",
            "physics_kind": "single_body",
            "moving_joints": False,
            "collider_prims": 220,
            "bounds_x_m": 0.2838,
            "bounds_y_m": 0.2844,
            "bounds_z_m": 0.3844,
            "total_mass_kg": 0.8,
            "up_axis": "Z",
            "runtime_status": "passed",
            "static_validation_status": "passed",
            "package_status": "completed",
            "source_revision": "8d45a6f0",
            "usd_path": "batch10/D00006 Bamboo Basket/IsaacSim_assets_aaa/IsaacSim_asset_aaa.usd",
            "physics_path": "batch10/D00006 Bamboo Basket/IsaacSim_assets_aaa/physics_aaa.json",
            "physics_download_url": (
                "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/8d45a6f0/"
                "batch10/D00006%20Bamboo%20Basket/IsaacSim_assets_aaa/physics_aaa.json"
            ),
            "physics_sha256": "",
        },
        {
            "catalog_id": "batch11_D00191_Cookie_Jar",
            "asset_id": "D00191",
            "logical_asset": "D00191",
            "name": "Cookie Jar",
            "display_name": "Cookie Jar (D00191)",
            "class_label": "jar",
            "wikidata_qcode": "Q133932",
            "batch": "batch11",
            "physics_kind": "multi_body",
            "moving_joints": False,
            "collider_prims": 29,
            "bounds_x_m": 0.2004,
            "bounds_y_m": 0.2012,
            "bounds_z_m": 0.2491,
            "total_mass_kg": 0.65,
            "up_axis": "Z",
            "runtime_status": "passed",
            "static_validation_status": "passed",
            "package_status": "completed",
            "source_revision": "8d45a6f0",
            "usd_path": "batch11/D00191 Cookie Jar/IsaacSim_assets_bbb/IsaacSim_asset_bbb.usd",
            "physics_path": "batch11/D00191 Cookie Jar/IsaacSim_assets_bbb/physics_bbb.json",
            "physics_download_url": (
                "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/8d45a6f0/"
                "batch11/D00191%20Cookie%20Jar/IsaacSim_assets_bbb/physics_bbb.json"
            ),
            "physics_sha256": "",
        },
        {
            "catalog_id": "batch26_D00519_Cake_Turntable",
            "asset_id": "D00519",
            "logical_asset": "D00519",
            "name": "Cake Turntable",
            "display_name": "Cake Turntable (D00519)",
            "class_label": "kitchenware",
            "wikidata_qcode": "Q1521410",
            "batch": "batch26",
            "physics_kind": "articulated",
            "moving_joints": True,
            "collider_prims": 45,
            "bounds_x_m": 0.28,
            "bounds_y_m": 0.28,
            "bounds_z_m": 0.08,
            "total_mass_kg": 0.9,
            "up_axis": "Z",
            "runtime_status": "passed",
            "static_validation_status": "passed",
            "package_status": "completed",
            "source_revision": "8d45a6f0",
            "physics_download_url": (
                "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/8d45a6f0/"
                "batch26/D00519%20Cake%20Turntable/IsaacSim_assets_ccc/physics_ccc.json"
            ),
            "physics_sha256": "",
        },
        {
            "catalog_id": "batch10_D00854_Tea_Pot",
            "asset_id": "D00854",
            "logical_asset": "D00854",
            "name": "Tea Pot",
            "display_name": "Tea Pot (D00854)",
            "class_label": "kitchenware",
            "wikidata_qcode": "Q1521410",
            "batch": "batch10",
            "physics_kind": "articulated",
            "moving_joints": True,
            "collider_prims": 229,
            "bounds_x_m": 0.2176,
            "bounds_y_m": 0.1108,
            "bounds_z_m": 0.1703,
            "total_mass_kg": 2.2,
            "up_axis": "Z",
            "runtime_status": "passed",
            "static_validation_status": "passed",
            "package_status": "completed",
            "source_revision": "8d45a6f0",
            "physics_download_url": (
                "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/8d45a6f0/"
                "batch10/D00854%20Tea%20Pot/IsaacSim_assets_ddd/physics_ddd.json"
            ),
            "physics_sha256": "",
        },
        {
            "catalog_id": "batch99_D00999_Failed_Stock",
            "asset_id": "D00999",
            "logical_asset": "D00999",
            "name": "Broken Toaster",
            "class_label": "home appliance",
            "physics_kind": "single_body",
            "moving_joints": False,
            "collider_prims": 10,
            "bounds_x_m": 0.25,
            "bounds_y_m": 0.18,
            "bounds_z_m": 0.20,
            "total_mass_kg": 1.2,
            "up_axis": "Z",
            "runtime_status": "completed_with_failed_tests",
            "static_validation_status": "passed",
            "package_status": "completed",
        },
        {
            "catalog_id": "batch99_D01000_Giant_Fridge",
            "asset_id": "D01000",
            "logical_asset": "D01000",
            "name": "Giant Fridge",
            "class_label": "home appliance",
            "physics_kind": "single_body",
            "moving_joints": False,
            "collider_prims": 12,
            "bounds_x_m": 0.80,
            "bounds_y_m": 0.75,
            "bounds_z_m": 1.80,
            "total_mass_kg": 50.0,
            "up_axis": "Z",
            "runtime_status": "passed",
            "static_validation_status": "passed",
            "package_status": "completed",
        },
        {
            "catalog_id": "bad_space_asset",
            "asset_id": "D01001",
            "logical_asset": "D01001 Has Space",
            "name": "Bad Space",
            "class_label": "basket",
            "physics_kind": "single_body",
            "moving_joints": False,
            "collider_prims": 10,
            "bounds_x_m": 0.2,
            "bounds_y_m": 0.2,
            "bounds_z_m": 0.2,
            "up_axis": "Z",
            "runtime_status": "passed",
            "static_validation_status": "passed",
            "package_status": "completed",
        },
    ]


def test_pure_stdlib_import_does_not_load_pxr_or_numpy():
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'scripts/objects'); import palatial; "
        "assert 'pxr' not in sys.modules and 'numpy' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


def test_fetch_bom_fallback_and_sha_mismatch(tmp_path, monkeypatch):
    usd_bytes = b"#usda 1.0\n"

    def _fake_http_get(url: str, **kwargs):
        if url.endswith("com.nvidia.simready.packaging.bom.json"):
            raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), io.BytesIO(b""))
        if url.endswith("D00006.usd"):
            return usd_bytes, {}
        if "physics.json" in url:
            return b'{"asset": {}}', {}
        raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), io.BytesIO(b""))

    monkeypatch.setattr(P, "http_get", _fake_http_get)
    monkeypatch.setattr(
        P,
        "list_hf_tree",
        lambda subpath, **kw: [
            ("D00006/simready_usd/.thumbs/256x256/D00006.usd.png", 100),
            ("D00006/simready_usd/D00006.usd", len(usd_bytes)),
        ],
    )

    res = P.fetch_one_package(
        {"logical_asset": "D00006", "physics_download_url": "https://huggingface.co/x/physics.json"},
        tmp_path,
        endpoint="https://hf-mirror.com",
    )
    assert res["simready_files"] == 1
    assert (tmp_path / "D00006/simready_usd/D00006.usd").is_file()

    # SHA-256 mismatch on physics.json raises OSError.
    with pytest.raises(OSError, match="sha256"):
        P.fetch_one_package(
            {
                "logical_asset": "D00006",
                "physics_download_url": "https://huggingface.co/x/physics.json",
                "physics_sha256": "0" * 64,
            },
            tmp_path / "fresh",
        )


def test_hf_url_helpers_and_mirror_rewrite(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert P.hf_endpoint() == "https://hf-mirror.com"
    assert (
        P.hf_resolve_url("D00191/simready_usd/D00191.usd", revision="main")
        == "https://hf-mirror.com/datasets/PalatialSim/palatial-nv-asset-library/resolve/main/"
        "D00191/simready_usd/D00191.usd"
    )
    orig = (
        "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/8d45a6f0/"
        "batch10/D00006%20Bamboo%20Basket/physics.json?download=true"
    )
    rewritten = P.rewrite_hf_url(orig)
    assert rewritten.startswith("https://hf-mirror.com/datasets/PalatialSim/")
    assert "D00006%20Bamboo%20Basket/physics.json?download=true" in rewritten
    # Non-HF URL stays untouched.
    assert P.rewrite_hf_url("https://example.com/a/b.json") == "https://example.com/a/b.json"


def test_http_get_retries_429_and_sets_user_agent(monkeypatch):
    seen_ua: list[str] = []
    sleeps: list[float] = []
    calls = {"n": 0}

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body
            self.headers = {"Content-Type": "application/json"}

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=0):
        seen_ua.append(req.get_header("User-agent"))
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "5"
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, io.BytesIO(b""))
        if calls["n"] == 2:
            raise urllib.error.URLError("transient reset")
        return _FakeResp(b'{"ok": true}')

    monkeypatch.setattr(P.urllib.request, "urlopen", _fake_urlopen)
    body, _ = P.http_get("https://hf-mirror.com/test", attempts=4, backoff=1.0, sleep_fn=sleeps.append)
    assert body == b'{"ok": true}'
    assert seen_ua == ["curl/8.0", "curl/8.0", "curl/8.0"]
    assert sleeps == [5.0, 2.0]

    # Non-retryable 404 raises immediately without sleeping.
    def _fake_404(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", email.message.Message(), io.BytesIO(b""))

    monkeypatch.setattr(P.urllib.request, "urlopen", _fake_404)
    sleeps_404: list[float] = []
    with pytest.raises(urllib.error.HTTPError):
        P.http_get("https://hf-mirror.com/missing", attempts=3, sleep_fn=sleeps_404.append)
    assert sleeps_404 == []


def test_load_and_filter_catalog_deterministic(tmp_path):
    cat_path = tmp_path / "catalog.jsonl"
    cat_path.write_text("\n".join(json.dumps(r) for r in _sample_catalog_rows()) + "\n", encoding="utf-8")
    rows = P.load_catalog_jsonl(cat_path)
    assert len(rows) == 7

    sel = P.filter_catalog(rows, physics_kind="all", stock_passing_only=True)
    chosen_ids = {c["logical_asset"]: c["id"] for c in sel["chosen"]}
    assert chosen_ids == {
        "D00519": "Articulation/palatial_kitchenware/00000",
        "D00854": "Articulation/palatial_kitchenware/00001",
        "D00006": "Rigid/palatial_basket/00000",
        "D00191": "Rigid/palatial_jar/00000",
    }
    dropped = {d["logical_asset"]: d["reason"] for d in sel["dropped"]}
    assert dropped["D00999"] == "runtime_status=completed_with_failed_tests"
    assert dropped["D01000"].startswith("too_large>")
    assert dropped["D01001 Has Space"] == "invalid_logical_asset"

    # Rigid-only includes single_body + multi_body, excludes articulated.
    sel_rigid = P.filter_catalog(rows, physics_kind="rigid")
    assert {c["logical_asset"] for c in sel_rigid["chosen"]} == {"D00006", "D00191"}

    # Articulated-only with max_per_category=1 picks D00519 deterministically.
    sel_art = P.filter_catalog(rows, physics_kind="articulated", max_per_category=1)
    assert [c["logical_asset"] for c in sel_art["chosen"]] == ["D00519"]

    # include_failed_stock lets D00999 through.
    sel_inc = P.filter_catalog(rows, physics_kind="single_body", stock_passing_only=False)
    assert {c["logical_asset"] for c in sel_inc["chosen"]} == {"D00006", "D00999"}

    # Empty catalog returns empty chosen/dropped.
    empty = P.filter_catalog([])
    assert empty["chosen"] == [] and empty["dropped"] == []


def test_fetch_one_package_space_free_bom_and_physics_json(tmp_path, monkeypatch):
    usd_bytes = b"#usda 1.0\n"
    tex_bytes = b"\x89PNG\r\n\x1a\nfake"
    phys_bytes = json.dumps({"asset": {"description": "A cookie jar", "bounds": {"size": [0.2, 0.2, 0.25]}}}).encode()

    usd_sha = hashlib.sha256(usd_bytes).hexdigest()
    tex_sha = hashlib.sha256(tex_bytes).hexdigest()
    phys_sha = hashlib.sha256(phys_bytes).hexdigest()

    bom_payload = json.dumps(
        {
            "format_version": "1.0",
            "items": [
                {"relative_path": "simready_usd/.thumbs/256x256/D00191.usd.png", "hash": {"sha256": "deadbeef"}},
                {"relative_path": "simready_usd/D00191.usd", "hash": {"sha256": usd_sha}},
                {"relative_path": "simready_usd/textures/base.png", "hash": {"sha256": tex_sha}},
            ],
        }
    ).encode()

    requested_urls: list[str] = []

    def _fake_http_get(url: str, **kwargs):
        requested_urls.append(url)
        if url.endswith("com.nvidia.simready.packaging.bom.json"):
            return bom_payload, {}
        if url.endswith("simready_usd/D00191.usd"):
            return usd_bytes, {}
        if url.endswith("simready_usd/textures/base.png"):
            return tex_bytes, {}
        if "physics_bbb.json" in url:
            return phys_bytes, {}
        raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), io.BytesIO(b""))

    monkeypatch.setattr(P, "http_get", _fake_http_get)

    entry = {
        "logical_asset": "D00191",
        "id": "Rigid/palatial_jar/00000",
        "physics_download_url": (
            "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/8d45a6f0/"
            "batch11/D00191%20Cookie%20Jar/IsaacSim_assets_bbb/physics_bbb.json"
        ),
        "physics_sha256": phys_sha,
    }
    res = P.fetch_one_package(entry, tmp_path, endpoint="https://hf-mirror.com")
    assert res["simready_files"] == 2  # .thumbs excluded
    assert res["downloaded_files"] == 3  # usd + texture + physics.json
    assert res["physics_json"] is True
    assert all(u.startswith("https://hf-mirror.com/") for u in requested_urls)
    assert (tmp_path / "D00191/simready_usd/D00191.usd").read_bytes() == usd_bytes
    assert (tmp_path / "D00191/simready_usd/textures/base.png").read_bytes() == tex_bytes
    assert (tmp_path / "D00191/metadata/physics.json").read_bytes() == phys_bytes
    assert (tmp_path / "D00191/metadata/catalog_entry.json").is_file()
    assert not (tmp_path / "D00191/simready_usd/.thumbs").exists()

    # Re-fetch skips already-present valid files (only BOM is queried).
    requested_urls.clear()
    res2 = P.fetch_one_package(entry, tmp_path, endpoint="https://hf-mirror.com")
    assert res2["downloaded_files"] == 0

    # Whitespace in logical_asset is refused immediately.
    with pytest.raises(ValueError, match="whitespace"):
        P.fetch_one_package({"logical_asset": "D00191 Bad"}, tmp_path)


def test_parse_physics_json_and_pure_stdlib_survey(tmp_path):
    # D00519: articulated with part_1 as body0 base (not part_0).
    p519 = tmp_path / "D00519"
    (p519 / "simready_usd/textures").mkdir(parents=True)
    (p519 / "metadata").mkdir(parents=True)
    (p519 / "simready_usd/D00519.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (p519 / "simready_usd/textures/base.png").write_bytes(b"png")
    (p519 / "metadata/physics.json").write_text(
        json.dumps(
            {
                "asset": {
                    "description": "A rotating cake turntable.",
                    "bounds": {"size": [0.28, 0.28, 0.08]},
                    "total_mass": 0.9,
                },
                "articulation": {
                    "has_joints": True,
                    "rigid_body_count": 2,
                    "joints": [
                        {
                            "id": "joint_part_1__part_0",
                            "usd_type": "PhysicsRevoluteJoint",
                            "enabled": True,
                            "body0_targets": ["/World/mesh/part_1"],
                            "body1_targets": ["/World/mesh/part_0"],
                        }
                    ],
                },
                "parts": [
                    {"name": "part_0", "mass": 0.3},
                    {"name": "part_1", "mass": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )
    (p519 / "metadata/catalog_entry.json").write_text(
        json.dumps(
            {
                "logical_asset": "D00519",
                "name": "Cake Turntable",
                "class_label": "kitchenware",
                "physics_kind": "articulated",
            }
        ),
        encoding="utf-8",
    )

    # D00854: articulated with part_01 as base and part_00 = 0.01 kg (< MIN_LINK_MASS).
    p854 = tmp_path / "D00854"
    (p854 / "simready_usd").mkdir(parents=True)
    (p854 / "metadata").mkdir(parents=True)
    (p854 / "simready_usd/D00854.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (p854 / "metadata/physics.json").write_text(
        json.dumps(
            {
                "asset": {
                    "description": "A ceramic tea pot with lid.",
                    "bounds": {"size": [0.22, 0.11, 0.17]},
                    "total_mass": 2.2,
                },
                "articulation": {
                    "has_joints": True,
                    "rigid_body_count": 2,
                    "joints": [
                        {
                            "id": "joint_part_01__part_00",
                            "usd_type": "PhysicsRevoluteJoint",
                            "enabled": True,
                            "body0_targets": ["/World/mesh/part_01"],
                            "body1_targets": ["/World/mesh/part_00"],
                        }
                    ],
                },
                "parts": [
                    {"id": "part_00", "name": "hinge", "mass": 0.01},
                    {"id": "part_01", "name": "body", "mass": 1.60},
                    {"id": "part_02", "name": "lid", "mass": 0.30},
                    {"id": "part_03", "name": "handle", "mass": 0.29},
                ],
            }
        ),
        encoding="utf-8",
    )
    (p854 / "metadata/catalog_entry.json").write_text(
        json.dumps(
            {
                "logical_asset": "D00854",
                "name": "Tea Pot",
                "class_label": "kitchenware",
                "physics_kind": "articulated",
            }
        ),
        encoding="utf-8",
    )

    # D00191: multi_body flattened as rigid.
    p191 = tmp_path / "D00191"
    (p191 / "simready_usd").mkdir(parents=True)
    (p191 / "metadata").mkdir(parents=True)
    (p191 / "simready_usd/D00191.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (p191 / "metadata/physics.json").write_text(
        json.dumps(
            {
                "asset": {"bounds": {"size": [0.20, 0.20, 0.25]}, "total_mass": 1.80},
                "articulation": {"rigid_body_count": 3},
                "parts": [
                    {"id": "part_00", "name": "D00191 Cookie_Jar", "mass": 0.05},
                    {"id": "part_01", "name": "D00191 Cookie_Jar", "mass": 1.40},
                    {"id": "part_02", "name": "D00191 Cookie_Jar", "mass": 0.35},
                ],
            }
        ),
        encoding="utf-8",
    )
    (p191 / "metadata/catalog_entry.json").write_text(
        json.dumps(
            {
                "logical_asset": "D00191",
                "name": "Cookie Jar",
                "class_label": "jar",
                "physics_kind": "multi_body",
            }
        ),
        encoding="utf-8",
    )

    # D00998: texture with space -> must be dropped by survey.
    pbad = tmp_path / "D00998"
    (pbad / "simready_usd/textures").mkdir(parents=True)
    (pbad / "metadata").mkdir(parents=True)
    (pbad / "simready_usd/D00998.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (pbad / "simready_usd/textures/has space.png").write_bytes(b"png")
    (pbad / "metadata/catalog_entry.json").write_text(
        json.dumps({"logical_asset": "D00998", "bounds_m": [0.2, 0.2, 0.2], "physics_kind": "single_body"}),
        encoding="utf-8",
    )

    manifest = P.survey(tmp_path)
    by_la = {e["logical_asset"]: e for e in manifest["chosen"]}
    assert set(by_la) == {"D00191", "D00519", "D00854"}
    assert by_la["D00519"]["base_candidates"] == ["part_1"]
    assert by_la["D00854"]["base_candidates"] == ["part_01"]
    assert by_la["D00854"]["low_mass_parts"] == ["part_00"]
    assert by_la["D00854"]["part_labels"] == {
        "part_00": "hinge",
        "part_01": "body",
        "part_02": "lid",
        "part_03": "handle",
    }
    assert by_la["D00191"]["multi_body_rigid_flatten"] is True
    assert by_la["D00191"]["id"].startswith("Rigid/palatial_jar/")
    assert by_la["D00519"]["id"].startswith("Articulation/palatial_kitchenware/")

    # Colliding part `name` values (as in live D00191) must stay keyed by unique `id`.
    p191_parsed = P.parse_physics_json(json.loads((p191 / "metadata/physics.json").read_text(encoding="utf-8")))
    assert p191_parsed["part_masses"] == {"part_00": 0.05, "part_01": 1.40, "part_02": 0.35}

    dropped_by_la = {d["logical_asset"]: d["reason"] for d in manifest["dropped"]}
    assert "whitespace" in dropped_by_la["D00998"]


def test_clamp_link_masses_enforces_min_link_mass_and_bounds():
    # D00854 case: part_00 = 0.01 kg (< 0.02 kg), part_01 = 2.19 kg.
    masses, info = P.clamp_link_masses(
        ["/root/part_01", "/root/part_00"],
        {"/root/part_01": 2.19, "/root/part_00": 0.01},
        {"/root/part_01": 0.002, "/root/part_00": 0.0002},
        max_total_mass=3.0,
    )
    assert masses["/root/part_00"] == pytest.approx(0.02)
    assert masses["/root/part_01"] == pytest.approx(2.19)
    assert info["clamped"] is True

    # Over-heavy total mass scaled down while respecting 0.02 kg floor.
    masses2, info2 = P.clamp_link_masses(
        ["/root/base", "/root/knob"],
        {"/root/base": 100.0, "/root/knob": 0.05},
        {"/root/base": 0.1, "/root/knob": 0.0001},
        max_total_mass=3.0,
    )
    assert masses2["/root/knob"] >= 0.02 - 1e-9
    assert sum(masses2.values()) == pytest.approx(3.0, abs=1e-6)
    assert info2["clamped"] is True

    # Fixed base heuristic.
    assert P.is_fixed_base("kitchenware", "Tea Pot", [0.22, 0.11, 0.17]) is False
    assert P.is_fixed_base("kitchenware", "Cake Turntable", [0.28, 0.28, 0.08]) is False
    assert P.is_fixed_base("home appliance", "Microwave Oven", [0.55, 0.45, 0.35]) is True


def test_stamp_ready_rigid_and_articulated_and_cli(tmp_path):
    out_root = tmp_path / "out"
    check_dir = tmp_path / "check"

    # Rigid passing + failing
    r_pass = "Rigid/palatial_basket/00000"
    r_fail = "Rigid/palatial_jar/00000"
    a_pass = "Articulation/palatial_kitchenware/00000"
    a_uncheck = "Articulation/palatial_kitchenware/00001"

    for oid in (r_pass, r_fail, a_pass, a_uncheck):
        d = out_root / "Assets/Object/RoboDojo" / oid
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"asset_id": oid, "runtime_ready": False}), encoding="utf-8")

    def _write_check(oid: str, payload: dict) -> None:
        kind, cat, idx = oid.split("/")
        p = check_dir / kind / cat / idx / "preview/check.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")

    _write_check(r_pass, {"settle": "pass", "tilt_deg": 0.4, "z_m": 0.19, "expect_z_m": 0.19})
    _write_check(r_fail, {"settle": "fail", "tilt_deg": 42.0})
    _write_check(a_pass, {"pass": True, "checks": {"rest": "pass", "open": "pass"}})

    summary_path = tmp_path / "ready_summary.json"
    rc = P.main(
        [
            "ready",
            "--root",
            str(out_root),
            "--check-dir",
            str(check_dir),
            "--summary",
            str(summary_path),
            "--note",
            "unit-test",
            r_pass,
            r_fail,
            a_pass,
            a_uncheck,
        ]
    )
    assert rc == 1  # r_fail and a_uncheck did not pass
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed_ids"] == [r_pass, a_pass]
    assert {e["id"]: e["why"] for e in summary["not_ready"]} == {
        r_fail: ["settle"],
        a_uncheck: ["not_checked"],
    }
    m_pass = json.loads((out_root / "Assets/Object/RoboDojo" / r_pass / "metadata.json").read_text(encoding="utf-8"))
    assert m_pass["runtime_ready"] is True
    assert m_pass["source"]["runtime_validated"] is True


def test_convert_one_and_convert_emit_valid_metadata_for_articulated_and_rigid(tmp_path, monkeypatch):
    """CPU test exercising `convert_one` + `convert` + real `articulated.metadata` (no pxr required)."""
    src_dir = tmp_path / "sources"
    for la in ("D00191", "D00854"):
        d = src_dir / la / "simready_usd"
        d.mkdir(parents=True)
        (d / f"{la}.usd").write_text("#usda 1.0\n", encoding="utf-8")

    seen_part_labels: list[dict | None] = []

    def _fake_adapt_art(source_usd, target_dir, *, class_label, name="", part_labels=None, **kw):
        seen_part_labels.append(part_labels)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "object.usd").write_bytes(b"fake-art-usd")
        return {
            "size": [0.22, 0.11, 0.17],
            "centre_shift": [0.0, 0.0, -0.085],
            "unit": 1.0,
            "yaw_deg": 0.0,
            "joints": [
                {
                    "name": "joint_part_00__part_02",
                    "type": "revolute",
                    "role": "lid",
                    "part": "lid",
                    "parent": "part_00",
                    "child": "part_02",
                    "axis": "Y",
                    "limits_deg": [-110.0, 0.0],
                    "friction": {"static": 0.3, "dynamic": 0.3, "source": "estimated (still_joints)"},
                }
            ],
            "link_masses": {"part_00": 0.02, "part_01": 1.6, "part_02": 0.3, "part_03": 0.29},
            "mass_info": {"raw_authored_kg": 2.2, "final_kg": 2.21, "clamped": True, "filled": False},
            "fixed_base": False,
            "limit_fixes": {},
            "removed": {"world_joints": [], "lights": [], "scripts": [], "articulation_root_moved_from": []},
            "reparented_links": [("/World/mesh/part_01", "/World/part_01")],
            "textures": ["textures/base.png"],
            "base_link": "part_01",
            "links": ["part_01", "part_00", "part_02", "part_03"],
        }

    def _fake_adapt_rigid(source_usd, target_dir, **kw):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "object.usd").write_bytes(b"fake-rigid-usd")
        return {
            "size": [0.2, 0.2, 0.25],
            "centre_shift": [0.0, 0.0, -0.125],
            "unit": 1.0,
            "yaw_deg": 0.0,
            "mass_kg": 0.5,
            "mass_info": {"raw_authored_kg": 1.8, "final_kg": 0.5, "clamped": True, "bounds_kg": [0.005, 0.5]},
            "stripped_articulation_roots": ["/root"],
            "stripped_child_bodies": ["/root/mesh/part_00", "/root/mesh/part_01"],
            "textures": ["textures/base.jpg"],
        }

    monkeypatch.setattr(P, "adapt_articulated", _fake_adapt_art)
    monkeypatch.setattr(P, "adapt_rigid", _fake_adapt_rigid)

    manifest = {
        "revision": "main",
        "chosen": [
            {
                "id": "Articulation/palatial_kitchenware/00000",
                "logical_asset": "D00854",
                "asset_id": "D00854",
                "catalog_id": "batch10_D00854_Tea_Pot",
                "name": "Tea Pot",
                "class_label": "kitchenware",
                "physics_kind": "articulated",
                "usd_rel": "D00854/simready_usd/D00854.usd",
                "part_labels": {"part_00": "hinge", "part_01": "body", "part_02": "lid"},
                "description": "A cast iron teapot.",
            },
            {
                "id": "Rigid/palatial_jar/00000",
                "logical_asset": "D00191",
                "asset_id": "D00191",
                "catalog_id": "batch11_D00191_Cookie_Jar",
                "name": "Cookie Jar",
                "class_label": "jar",
                "physics_kind": "multi_body",
                "total_mass_kg": 1.8,
                "usd_rel": "D00191/simready_usd/D00191.usd",
            },
        ],
    }

    out_root = tmp_path / "out"
    res = P.convert(src_dir, out_root, manifest)
    assert res["failed"] == []
    assert len(res["converted"]) == 2
    assert seen_part_labels == [{"part_00": "hinge", "part_01": "body", "part_02": "lid"}]

    art_meta = json.loads(
        (out_root / "Assets/Object/RoboDojo/Articulation/palatial_kitchenware/00000/metadata.json").read_text()
    )
    assert art_meta["uuid"]
    assert art_meta["base_link"] == "part_01"
    assert art_meta["source"]["object"] == "D00854"
    assert art_meta["body_joints"][0]["role"] == "lid"
    assert art_meta["runtime_ready"] is False

    rigid_meta = json.loads((out_root / "Assets/Object/RoboDojo/Rigid/palatial_jar/00000/metadata.json").read_text())
    assert rigid_meta["physics"]["mass"] == pytest.approx(0.5)
    assert rigid_meta["adapter"]["multi_body_flattened"] is True
    assert rigid_meta["source"]["object"] == "D00191"

    # Second run with skip_existing skips both.
    res2 = P.convert(src_dir, out_root, manifest, skip_existing=True)
    assert res2["converted"] == []
    assert len(res2["skipped"]) == 2


def test_adapt_rigid_strips_multibody_articulation_and_spaces(tmp_path):
    pxr = pytest.importorskip("pxr")
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = (
        pxr.Gf,
        pxr.Sdf,
        pxr.Usd,
        pxr.UsdGeom,
        pxr.UsdPhysics,
        pxr.UsdShade,
    )

    pkg = tmp_path / "src/D00191/simready_usd"
    (pkg / "textures").mkdir(parents=True)
    (pkg / "textures/base color.png").write_bytes(b"fake-png")  # intentional space to verify sanitization
    src_usd = pkg / "D00191.usd"

    stage = Usd.Stage.CreateNew(str(src_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.ArticulationRootAPI.Apply(world)
    world.AddAppliedSchema("PhysxArticulationAPI")
    stale_mass = UsdPhysics.MassAPI.Apply(world)
    stale_mass.CreateCenterOfMassAttr(Gf.Vec3f(9.0, 9.0, 9.0))

    container = UsdGeom.Xform.Define(stage, "/World/textured_mesh_jar").GetPrim()
    for name, z0, m in (("part_0", 0.0, 0.5), ("part_1", 0.12, 0.15)):
        link = UsdGeom.Xform.Define(stage, f"/World/textured_mesh_jar/{name}").GetPrim()
        UsdGeom.XformCommonAPI(link).SetTranslate(Gf.Vec3d(0.1, 0.2, z0))
        UsdPhysics.RigidBodyAPI.Apply(link)
        UsdPhysics.MassAPI.Apply(link).CreateMassAttr(m)
        cube = UsdGeom.Cube.Define(stage, f"/World/textured_mesh_jar/{name}/mesh")
        cube.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)

    # Material with texture containing space
    mat = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    tex = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/base color.png"))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(container).Bind(mat)

    stage.GetRootLayer().Save()

    out_dir = tmp_path / "out_rigid"
    facts = P.adapt_rigid(src_usd, out_dir)
    assert facts["mass_kg"] == pytest.approx(0.5)  # 0.65 clamped to 0.5
    assert facts["stripped_articulation_roots"]
    assert len(facts["stripped_child_bodies"]) == 2
    assert all(" " not in t for t in facts["textures"])

    reopened = Usd.Stage.Open(str(out_dir / "object.usd"))
    bodies = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    art_roots = [str(p.GetPath()) for p in reopened.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert bodies == ["/root"]
    assert art_roots == []
    # Container material binding preserved (not deleted by reparenting) and stale centerOfMass cleared.
    re_container = reopened.GetPrimAtPath("/root/textured_mesh_jar")
    assert re_container.IsValid()
    assert str(UsdShade.MaterialBindingAPI(re_container).GetDirectBinding().GetMaterialPath()) == "/root/Looks/Mat"
    assert not reopened.GetPrimAtPath("/root").HasAttribute("physics:centerOfMass")


def test_adapt_articulated_base_reorder_mass_clamp_connections_and_friction(tmp_path):
    pxr = pytest.importorskip("pxr")
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = (
        pxr.Gf,
        pxr.Sdf,
        pxr.Usd,
        pxr.UsdGeom,
        pxr.UsdPhysics,
        pxr.UsdShade,
    )

    pkg = tmp_path / "src/D00854/simready_usd"
    pkg.mkdir(parents=True)
    src_usd = pkg / "D00854.usd"

    stage = Usd.Stage.CreateNew(str(src_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Xform.Define(stage, "/World/textured_mesh_pot")

    # Author part_00 FIRST (0.01 kg lid) and part_01 SECOND (2.19 kg body0 base).
    p00 = UsdGeom.Xform.Define(stage, "/World/textured_mesh_pot/part_00").GetPrim()
    UsdGeom.XformCommonAPI(p00).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.14))
    UsdPhysics.RigidBodyAPI.Apply(p00)
    m00 = UsdPhysics.MassAPI.Apply(p00)
    m00.CreateMassAttr(0.01)
    m00.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
    c00 = UsdGeom.Cube.Define(stage, "/World/textured_mesh_pot/part_00/mesh")
    c00.CreateSizeAttr(0.06)
    UsdPhysics.CollisionAPI.Apply(c00.GetPrim()).CreateCollisionEnabledAttr(True)

    p01 = UsdGeom.Xform.Define(stage, "/World/textured_mesh_pot/part_01").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p01)
    m01 = UsdPhysics.MassAPI.Apply(p01)
    m01.CreateMassAttr(2.19)
    m01.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-2, 1e-2, 1e-2))
    c01 = UsdGeom.Cube.Define(stage, "/World/textured_mesh_pot/part_01/mesh")
    c01.CreateSizeAttr(0.14)
    UsdPhysics.CollisionAPI.Apply(c01.GetPrim()).CreateCollisionEnabledAttr(True)

    # Material inside part_01 to exercise AttributeSpec.connectionPathList rewriting on reparent.
    mat = UsdShade.Material.Define(stage, "/World/textured_mesh_pot/part_01/Mat")
    sh = UsdShade.Shader.Define(stage, "/World/textured_mesh_pot/part_01/Mat/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(c01.GetPrim()).Bind(mat)

    # Revolute joint with body0=part_01 (base), body1=part_00 (lid), legacy physxJoint:jointFriction only.
    UsdGeom.Scope.Define(stage, "/World/textured_mesh_pot/Joints")
    j = UsdPhysics.RevoluteJoint.Define(stage, "/World/textured_mesh_pot/Joints/joint_part_01__part_00")
    j.CreateAxisAttr("Y")
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(90.0)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.07))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.07))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateBody0Rel().SetTargets([p01.GetPath()])
    j.CreateBody1Rel().SetTargets([p00.GetPath()])
    j.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.05)

    stage.GetRootLayer().Save()

    out_dir = tmp_path / "out_art"
    facts = P.adapt_articulated(
        src_usd,
        out_dir,
        class_label="kitchenware",
        name="Tea Pot",
        part_labels={"part_00": "lid", "part_01": "body"},
    )
    assert facts["base_link"] == "part_01"
    assert facts["link_masses"]["part_00"] >= 0.02
    assert facts["joints"][0]["role"] == "lid"

    reopened = Usd.Stage.Open(str(out_dir / "object.usd"))
    root = reopened.GetDefaultPrim()
    assert root.GetPath() == Sdf.Path("/root")
    assert root.GetChildren()[0].GetName() == "part_01"

    # Verify mass clamp & proportional diagonalInertia scaling on part_00 (0.01 -> 0.02 => 2x inertia).
    re_p00 = reopened.GetPrimAtPath("/root/part_00")
    assert UsdPhysics.MassAPI(re_p00).GetMassAttr().Get() == pytest.approx(0.02)
    diag = UsdPhysics.MassAPI(re_p00).GetDiagonalInertiaAttr().Get()
    assert diag[0] == pytest.approx(2e-4)

    # Verify RelationshipSpec (joint body0/body1) and AttributeSpec.connectionPathList (shader surface) rewritten.
    re_j = reopened.GetPrimAtPath("/root/textured_mesh_pot/Joints/joint_part_01__part_00")
    assert [str(t) for t in UsdPhysics.Joint(re_j).GetBody0Rel().GetTargets()] == ["/root/part_01"]
    assert [str(t) for t in UsdPhysics.Joint(re_j).GetBody1Rel().GetTargets()] == ["/root/part_00"]
    assert re_j.GetAttribute("physxJointAxis:angular:staticFrictionEffort").Get() > 0
    assert re_j.GetAttribute("physxJointAxis:angular:dynamicFrictionEffort").Get() > 0

    re_mat = UsdShade.Material(reopened.GetPrimAtPath("/root/part_01/Mat"))
    src_conns = re_mat.GetSurfaceOutput().GetRawConnectedSourcePaths()
    assert [str(p) for p in src_conns] == ["/root/part_01/Mat/Shader.outputs:surface"]
