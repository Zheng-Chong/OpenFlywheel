"""CPU unit tests for `scripts/objects/molmospaces_objaverse.py`."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import molmospaces_objaverse as obja  # noqa: E402


def _make_sample_catalog():
    arrow_table = [
        {"path": "objaverse_obja_aaa111.tar.zst", "shard_id": 0, "offset": 512, "size": 1024},
        {"path": "objaverse_obja_bbb222.tar.zst", "shard_id": 1, "offset": 1024, "size": 2048},
        {"path": "objaverse_obja_ccc333.tar.zst", "shard_id": 2, "offset": 2048, "size": 2048},
        {"path": "objaverse_obja_ddd444.tar.zst", "shard_id": 3, "offset": 4096, "size": 2048},
        {"path": "objaverse_obja_eee555.tar.zst", "shard_id": 4, "offset": 4096, "size": 2048},
        {"path": "objaverse_obja_fff666.tar.zst", "shard_id": 5, "offset": 4096, "size": 2048},
        {"path": "objaverse_obja_ggg777.tar.zst", "shard_id": 6, "offset": 4096, "size": 2048},
    ]
    combined_meta = {e["path"]: {"obja": {"Textures": {"tex.png": {}}}} for e in arrow_table}
    combined_meta["objaverse_obja_ggg777.tar.zst"] = {"obja": {"mesh.usda": {}}}  # untextured

    objathor_meta = {
        "aaa111": {
            "category": "coffee mug",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 0.10, "y": 0.08, "z": 0.11},
            "mass": 0.35,
            "description_short": "ceramic coffee mug",
        },
        "bbb222": {
            "category": "mug",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 0.09, "y": 0.09, "z": 0.10},
            "mass": 0.30,
            "description_short": "white mug",
        },
        "ccc333": {
            "category": "hammer",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 0.30, "y": 0.12, "z": 0.04},
            "mass": 0.60,
            "description_short": "steel claw hammer",
        },
        "ddd444": {
            "category": "hammer",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 0.35, "y": 0.15, "z": 0.05},
            "mass": 0.80,
            "description_short": "fantasy warhammer weapon",
        },
        "eee555": {
            "category": "handgun",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 0.20, "y": 0.12, "z": 0.04},
            "mass": 0.70,
            "description_short": "black pistol",
        },
        "fff666": {
            "category": "bowl",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 1.20, "y": 1.20, "z": 0.50},  # too big
            "mass": 2.0,
            "description_short": "giant bowl",
        },
        "ggg777": {
            "category": "apple",
            "primaryProperty": "CanPickup",
            "onObject": True,
            "boundingBox": {"x": 0.08, "y": 0.08, "z": 0.08},
            "mass": 0.15,
            "description_short": "red apple",
        },
    }
    licenses = {
        "aaa111": {"license": "cc0", "name": "Mug 1", "uri": "https://sketchfab.com/1", "user": "alice"},
        "bbb222": {"license": "by-nc", "name": "Mug NC", "uri": "https://sketchfab.com/2", "user": "bob"},
        "ccc333": {"license": "by", "name": "Claw Hammer", "uri": "https://sketchfab.com/3", "user": "carol"},
        "ddd444": {"license": "by", "name": "Warhammer", "uri": "https://sketchfab.com/4", "user": "dave"},
        "eee555": {"license": "by", "name": "Pistol", "uri": "https://sketchfab.com/5", "user": "eve"},
        "fff666": {"license": "cc0", "name": "Giant Bowl", "uri": "https://sketchfab.com/6", "user": "frank"},
        "ggg777": {"license": "by", "name": "Apple", "uri": "https://sketchfab.com/7", "user": "grace"},
    }
    return obja.build_catalog(arrow_table, combined_meta, objathor_meta, licenses)


def test_license_normalization_strict_commercial():
    assert obja.normalize_sketchfab_license("cc0") == "CC0"
    assert obja.normalize_sketchfab_license("CC0") == "CC0"
    assert obja.normalize_sketchfab_license("by") == "CC-BY-4.0"
    assert obja.normalize_sketchfab_license("BY") == "CC-BY-4.0"
    for rejected in ("by-nc", "by-nc-sa", "by-nc-nd", "by-sa", "by-nd", "", None, "unknown", "proprietary"):
        assert obja.normalize_sketchfab_license(rejected) is None


def test_category_canonicalization_and_uid_extraction():
    assert obja.canonical_category_slug("coffee mug") == "mug"
    assert obja.canonical_category_slug("Water Bottle") == "bottle"
    assert obja.canonical_category_slug("baseball cap") == "cap"
    assert obja.canonical_category_slug("alarm clock") == "alarm_clock"
    assert obja.uid_from_archive_path("objaverse_obja_00002bcb84af4a4781174e62619f14e2.tar.zst") == (
        "00002bcb84af4a4781174e62619f14e2"
    )
    assert obja.uid_from_archive_path("other.tar.zst") is None


def test_filter_catalog_strict_commercial_quality_and_noise():
    rows = _make_sample_catalog()
    res = obja.filter_catalog(rows, commercial_only=True)
    chosen_uids = {e["uid"] for e in res["chosen"]}
    assert chosen_uids == {"aaa111", "ccc333"}
    by_uid = {e["uid"]: e for e in res["chosen"]}
    assert by_uid["aaa111"]["id"] == "Rigid/obja_mug/00000"
    assert by_uid["aaa111"]["license"] == "CC0"
    assert by_uid["ccc333"]["id"] == "Rigid/obja_hammer/00000"
    assert by_uid["ccc333"]["license"] == "CC-BY-4.0"

    reasons = res["summary"]["drop_reasons"]
    assert reasons.get("license_rejected:by-nc") == 1
    assert reasons.get("noise_description") == 1
    assert reasons.get("not_in_whitelist") == 1
    assert reasons.get("size_out_of_range") == 1
    assert reasons.get("untextured") == 1


def test_filter_catalog_existing_state_and_caps():
    rows = _make_sample_catalog()
    res = obja.filter_catalog(
        rows,
        commercial_only=True,
        existing_uids={"aaa111"},
        next_index={"hammer": 7},
    )
    assert [e["uid"] for e in res["chosen"]] == ["ccc333"]
    assert res["chosen"][0]["id"] == "Rigid/obja_hammer/00007"


def test_reverse_face_slices():
    assert obja._reverse_face_slices([0, 1, 2, 3, 4, 5], [3, 3]) == [2, 1, 0, 5, 4, 3]
    assert obja._reverse_face_slices([10, 11, 12, 13, 20, 21, 22], [4, 3]) == [13, 12, 11, 10, 22, 21, 20]
    assert obja._reverse_face_slices([], []) == []


def test_convert_one_refuses_non_commercial(tmp_path: Path):
    entry = {
        "id": "Rigid/obja_mug/00000",
        "uid": "bbb222",
        "asset_id": "obja_bbb222",
        "category": "mug",
        "category_slug": "mug",
        "sketchfab_license": "by-nc",
        "license": "",
    }
    with pytest.raises(ValueError, match="non-commercial"):
        obja.convert_one(tmp_path, tmp_path / "out", entry)


def test_extract_asset_tar_zst_and_traversal_guard(tmp_path: Path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        good = b"#usda 1.0\n"
        info = tarfile.TarInfo("obja_abc/obja_abc.usda")
        info.size = len(good)
        tf.addfile(info, io.BytesIO(good))

        bad = b"pwned"
        info_bad = tarfile.TarInfo("../escape.txt")
        info_bad.size = len(bad)
        tf.addfile(info_bad, io.BytesIO(bad))

    monkeypatch.setattr(obja.thor, "decompress_zst", lambda data: data)
    out = obja._extract_asset_tar_zst(buf.getvalue(), tmp_path, "obja_abc")
    assert (out / "obja_abc.usda").is_file()
    assert not (tmp_path / "escape.txt").exists()


def test_stamp_ready(tmp_path: Path):
    oid = "Rigid/obja_mug/00000"
    obj_dir = tmp_path / "out/Assets/Object/RoboDojo" / oid
    obj_dir.mkdir(parents=True)
    (obj_dir / "metadata.json").write_text(json.dumps({"asset_id": oid, "source": {}, "adapter": {}}))

    check_dir = tmp_path / "check"
    # Missing check.json -> fail
    ok, why = obja.stamp_ready(tmp_path / "out", oid, check_dir, note="test")
    assert not ok and why == ["not_checked"]

    # Passing check.json -> ready
    prev = check_dir / oid / "preview"
    prev.mkdir(parents=True)
    (prev / "check.json").write_text(json.dumps({"settle": "pass", "tilt_deg": 0.2, "z_m": 0.05}))
    ok, why = obja.stamp_ready(tmp_path / "out", oid, check_dir, note="test-pass")
    assert ok and why == []
    meta = json.loads((obj_dir / "metadata.json").read_text())
    assert meta["runtime_ready"] is True
    assert meta["source"]["runtime_validated"] is True


def test_bilingual_captions_cover_all_whitelist_and_pass_object_names_check():
    import object_names

    # Every canonical slug produced by CURATED_WHITELIST_CATEGORIES must have a hand-curated bilingual entry.
    missing_slugs = {
        obja.canonical_category_slug(cat)
        for cat in obja.CURATED_WHITELIST_CATEGORIES
        if obja.canonical_category_slug(cat) not in obja.CATEGORY_BILINGUAL_MAP
    }
    assert missing_slugs == set(), f"Missing bilingual mappings: {sorted(missing_slugs)}"

    table = {}
    used_en, used_zh = set(), set()
    for i, slug in enumerate(sorted(obja.CATEGORY_BILINGUAL_MAP)):
        oid = f"Rigid/obja_{slug}/00000"
        table[oid] = obja.build_bilingual_entry(
            oid,
            category_slug=slug,
            size_m=[0.08 + 0.001 * (i % 15), 0.09, 0.12],
            desc_short={"two_words": "Red object" if i % 2 == 0 else "Wooden item"},
            used_l3_en=used_en,
            used_l3_zh=used_zh,
        )
        # Also test collision disambiguation with identical size & modifier on index 00001
        oid2 = f"Rigid/obja_{slug}/00001"
        table[oid2] = obja.build_bilingual_entry(
            oid2,
            category_slug=slug,
            size_m=[0.08 + 0.001 * (i % 15), 0.09, 0.12],
            desc_short={"two_words": "Red object" if i % 2 == 0 else "Wooden item"},
            used_l3_en=used_en,
            used_l3_zh=used_zh,
        )
    assert object_names.problems(table) == []


def test_description_short_normalization_and_formatting():
    raw_dict = {
        "one_word": "Clock",
        "two_words": "Black clock",
        "three_words": "Black digital clock",
        "five_words": "Black digital alarm clock display",
    }
    assert obja.normalize_description_short(raw_dict) == raw_dict
    # Legacy stringified Python dict repr must be parsed back into a clean dict
    assert obja.normalize_description_short(repr(raw_dict)) == raw_dict
    assert obja.normalize_description_short("plain white mug") == "plain white mug"
    assert obja.normalize_description_short(None) == ""

    # format_description_text prefers desc_long, falls back to clean short text, never emits dict repr
    assert (
        obja.format_description_text(raw_dict, "A black digital alarm clock with a green LED display.")
        == "A black digital alarm clock with a green LED display."
    )
    assert obja.format_description_text(repr(raw_dict), "") == "Black digital alarm clock display"
    assert not obja.format_description_text(repr(raw_dict), "").startswith("{")


def test_modifier_extraction_prioritizes_primary_color_over_accents():
    # Real Wave 1 regression cases where unordered set + fixed MODIFIER_BILINGUAL_MAP order picked wrong colors:
    # 1. Black digital alarm clock with green LED display -> must be black, NOT green
    e1 = obja.build_bilingual_entry(
        "Rigid/obja_alarm_clock/00017",
        category_slug="alarm_clock",
        size_m=[0.30, 0.08, 0.18],
        desc_short="{'one_word': 'Clock', 'two_words': 'Black clock', 'three_words': 'Black digital clock'}",
        desc_long="A black digital alarm clock with a green LED display and red power indicator.",
    )
    assert e1["en"][1] == "black alarm clock"
    assert e1["zh"][1] == "黑色闹钟"

    # 2. Worn white baseball with red stitching -> must be white, NOT red
    e2 = obja.build_bilingual_entry(
        "Rigid/obja_baseball/00015",
        category_slug="baseball",
        size_m=[0.07, 0.07, 0.07],
        desc_short={"two_words": "Worn baseball", "three_words": "Worn white baseball"},
        desc_long="A worn white leather baseball with classic red stitching.",
    )
    assert e2["en"][1] == "white baseball"
    assert e2["zh"][1] == "白色棒球"

    # 3. Blue calculator with red buttons -> must be blue, NOT red
    e3 = obja.build_bilingual_entry(
        "Rigid/obja_calculator/00006",
        category_slug="calculator",
        size_m=[0.12, 0.18, 0.02],
        desc_short={"two_words": "Blue calculator", "three_words": "Blue pocket calculator"},
        desc_long="A blue plastic calculator with red and green buttons.",
    )
    assert e3["en"][1] == "blue calculator"
    assert e3["zh"][1] == "蓝色计算器"

    # 4. Yellow cheese block with blue mold -> must be yellow, NOT blue
    e4 = obja.build_bilingual_entry(
        "Rigid/obja_cheese/00010",
        category_slug="cheese",
        size_m=[0.10, 0.08, 0.06],
        desc_short={"two_words": "Yellow cheese", "three_words": "Yellow cheese block"},
        desc_long="A yellow cheese wedge with blue mold spots.",
    )
    assert e4["en"][1] == "yellow cheese wedge"
    assert e4["zh"][1] == "黄色奶酪"

    # 5. Primary clause of desc_long wins over secondary 'with ...' clause when desc_short has no color
    e5 = obja.build_bilingual_entry(
        "Rigid/obja_mug/00099",
        category_slug="mug",
        size_m=[0.10, 0.09, 0.11],
        desc_short={"two_words": "Coffee mug"},
        desc_long="A white ceramic mug with a red logo printed on the side.",
    )
    assert e5["en"][1] == "white coffee mug"
    assert e5["zh"][1] == "白色马克杯"
