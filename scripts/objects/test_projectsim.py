"""Unit tests for scripts/objects/projectsim.py (pure CPU; optional pxr test when usd-core is present)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _load("projectsim")


def test_is_needed_file():
    assert P.is_needed_file("README.md")
    assert P.is_needed_file("AGENTS.md")
    assert P.is_needed_file("assets/oven/v01_oven_b77670/v01_oven_b77670.usda")
    assert P.is_needed_file("assets/oven/v01_oven_b77670/textures/base_color.png")
    assert not P.is_needed_file("assets/oven/v01_oven_b77670/v01_oven_b77670.glb")
    assert not P.is_needed_file("assets/oven/v01_oven_b77670/mjcf/v01_oven_b77670.xml")
    assert not P.is_needed_file("assets/oven/v01_oven_b77670/mjcf/visuals/textures/base_color.png")
    assert not P.is_needed_file("assets/unknown_cat/v01/v01.usda")
    assert not P.is_needed_file(".gitattributes")


def test_variant_suffix():
    assert P.variant_suffix("v01_oven_b77670") == "b77670"
    assert P.variant_suffix("v02_desk_with_drawers_0d1a63") == "0d1a63"
    assert P.variant_suffix("single") == "single"


def test_auth_headers_from_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    h0 = P._auth_headers()
    assert "Authorization" not in h0
    assert "User-Agent" in h0

    monkeypatch.setenv("HF_TOKEN", "hf_dummy_test_token")
    h1 = P._auth_headers()
    assert h1["Authorization"] == "Bearer hf_dummy_test_token"


def test_content_fingerprint_ignores_date_and_checks_textures(tmp_path):
    d1 = tmp_path / "v01_oven_abc123"
    d2 = tmp_path / "v02_oven_abc123"
    d3 = tmp_path / "v03_oven_abc123"
    for d in (d1, d2, d3):
        (d / "textures").mkdir(parents=True)
    usda_a = '#usda 1.0\n(\n    string usd_date_generated = "2026-09-15"\n)\ndef Xform "oven" {}\n'
    usda_b = '#usda 1.0\n(\n    string usd_date_generated = "2026-09-17"\n)\ndef Xform "oven" {}\n'
    (d1 / "v01_oven_abc123.usda").write_text(usda_a, encoding="utf-8")
    (d2 / "v02_oven_abc123.usda").write_text(usda_b, encoding="utf-8")
    (d3 / "v03_oven_abc123.usda").write_text(usda_b, encoding="utf-8")
    (d1 / "textures/base_color.png").write_bytes(b"PNG_A")
    (d2 / "textures/base_color.png").write_bytes(b"PNG_A")
    (d3 / "textures/base_color.png").write_bytes(b"PNG_B")

    fp1 = P.content_fingerprint(d1 / "v01_oven_abc123.usda")
    fp2 = P.content_fingerprint(d2 / "v02_oven_abc123.usda")
    fp3 = P.content_fingerprint(d3 / "v03_oven_abc123.usda")
    assert fp1 == fp2
    assert fp1[0] == fp3[0]
    assert fp1[1] != fp3[1]


def _dummy_facts(**overrides):
    base = {
        "up_axis": "Z",
        "world_movable": [],
        "movable": [{"name": "door_j", "body1": "/oven/door", "type": "revolute"}],
        "bases": ["/oven/body"],
        "colliders": 2,
    }
    base.update(overrides)
    return base


def test_decide_rules():
    ok, reason = P.decide("oven", [0.6, 0.6, 0.9], _dummy_facts())
    assert ok and "door" in reason

    ok, _ = P.decide("oven", [0.6, 0.6, 0.9], _dummy_facts(up_axis="Y"))
    assert not ok
    ok, _ = P.decide("oven", [0.6, 0.6, 0.9], _dummy_facts(movable=[]))
    assert not ok
    ok, _ = P.decide("oven", [0.6, 0.6, 0.9], _dummy_facts(bases=["/a", "/b"]))
    assert not ok
    ok, _ = P.decide("oven", [0.6, 0.6, 0.9], _dummy_facts(colliders=0))
    assert not ok
    ok, _ = P.decide("oven", [0.001, 0.6, 0.9], _dummy_facts())
    assert not ok
    ok, _ = P.decide(
        "oven",
        [0.6, 0.6, 0.9],
        _dummy_facts(
            movable=[
                {"name": "dup", "body1": "/oven/door", "type": "revolute"},
                {"name": "dup", "body1": "/oven/knob", "type": "revolute"},
            ]
        ),
    )
    assert not ok


def test_survey_deduplication(tmp_path, monkeypatch):
    # Create 3 oven variants: v01 and v02 have same suffix+tex (dup); v03 has same suffix, different tex (kept).
    src = tmp_path / "sources"
    for vid, tex_bytes, extra in (
        ("v01_oven_b77670", b"TEX1", "a"),
        ("v02_oven_b77670", b"TEX1", "b"),
        ("v03_oven_b77670", b"TEX2", "c"),
    ):
        d = src / "assets/oven" / vid
        (d / "textures").mkdir(parents=True)
        (d / f"{vid}.usda").write_text(f'#usda 1.0\n# {extra}\ndef Xform "oven" {{}}\n', encoding="utf-8")
        (d / "textures/base_color.png").write_bytes(tex_bytes)

    monkeypatch.setattr(P, "inspect_model", lambda p: ([0.6, 0.6, 0.9], _dummy_facts()))
    res = P.survey(src, categories=("oven",))
    assert res["summary"]["total"] == 3
    assert res["summary"]["chosen"] == 2
    assert res["summary"]["dropped"] == 1
    assert [c["id"] for c in res["chosen"]] == [
        "Articulation/projectsim_oven/00000",
        "Articulation/projectsim_oven/00001",
    ]
    assert "duplicate" in res["dropped"][0]["reason"]


def test_ready_stamps_metadata(tmp_path):
    root = tmp_path / "out"
    d_pass = root / "Assets/Object/RoboDojo/Articulation/projectsim_oven/00000"
    d_fail = root / "Assets/Object/RoboDojo/Articulation/projectsim_oven/00001"
    d_pass.mkdir(parents=True)
    d_fail.mkdir(parents=True)
    (d_pass / "metadata.json").write_text(json.dumps({"asset_id": "Articulation/projectsim_oven/00000"}))
    (d_fail / "metadata.json").write_text(json.dumps({"asset_id": "Articulation/projectsim_oven/00001"}))

    rpt = tmp_path / "report.json"
    rpt.write_text(
        json.dumps(
            [
                {
                    "asset_dir": str(d_pass),
                    "pass": True,
                    "checks": {"fixed_base": "pass", "settle": "pass", "joints": "pass", "render": "pass"},
                    "checked_at": "2026-11-24 12:00:00",
                },
                {
                    "asset_dir": str(d_fail),
                    "pass": False,
                    "checks": {"fixed_base": "pass", "settle": "fail"},
                    "checked_at": "2026-11-24 12:00:01",
                },
            ]
        )
    )
    summary = P.ready(root, [rpt])
    assert summary == {"total": 2, "passed": 1, "failed": 1, "missing": 0}
    m0 = json.loads((d_pass / "metadata.json").read_text())
    m1 = json.loads((d_fail / "metadata.json").read_text())
    assert m0["runtime_ready"] is True
    assert m0["asset_validation"] == "pass"
    assert m1["runtime_ready"] is False
    assert m1["asset_validation"] == "fail"
    assert "settle" in m1["check_summary"]["why"]

    # Also accept a single-dict per-object check.json file.
    single_check = tmp_path / "check.json"
    single_check.write_text(
        json.dumps(
            {
                "asset_dir": str(d_fail),
                "pass": True,
                "checks": {"fixed_base": "pass", "settle": "pass", "hold": "pass"},
                "checked_at": "2026-11-24 12:05:00",
            }
        )
    )
    summary2 = P.ready(root, [single_check])
    assert summary2 == {"total": 1, "passed": 1, "failed": 0, "missing": 0}
    m1_after = json.loads((d_fail / "metadata.json").read_text())
    assert m1_after["runtime_ready"] is True
    assert m1_after["asset_validation"] == "pass"


def test_size_check():
    assert P.size_check("oven", [0.6, 0.65, 0.9])["result"] == "pass"
    assert P.size_check("oven", [0.1, 0.1, 0.1])["result"] == "fail"
