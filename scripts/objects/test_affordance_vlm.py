"""Unit and integration tests for Stage 3 VLM / 3D part-level semantic annotation."""

from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pytest

from scripts.objects.affordance import annotate_asset, annotate_mesh, validate_affordance
from scripts.objects.affordance_vlm import (
    OpenAICompatibleVLMLabeler,
    annotate_affordance_with_vlm,
    assign_semantic_tags_to_grasps,
    backproject_2d_masks_to_points,
    build_multiview_cameras,
    load_rendered_views,
    look_at_cam_to_world,
    main as vlm_main,
    normalize_part_tag,
    rasterize_regions_to_label_map,
    render_grasp_markers_image,
    render_mesh_view,
    render_mesh_views,
    save_rendered_views,
    visible_grasps_in_view,
)
from scripts.objects.test_affordance import make_box_mesh, make_cylinder_mesh, merge_meshes, write_obj


def _make_mug_mesh():
    body = make_cylinder_mesh(radius=0.036, height=0.09, n_seg=24, n_rings=4)
    handle = make_box_mesh(0.024, 0.014, 0.050, center=(0.052, 0.0, 0.0))
    return merge_meshes(body, handle)


def test_normalize_part_tag_aliases():
    assert normalize_part_tag("Handle") == "handle"
    assert normalize_part_tag("ear handle") == "handle"
    assert normalize_part_tag("cup lip") == "rim"
    assert normalize_part_tag("pour_spout") == "spout"
    assert normalize_part_tag("nozzle") == "spout"
    assert normalize_part_tag("top cover") == "lid"
    assert normalize_part_tag("bottleneck") == "neck"
    assert normalize_part_tag("screw cap") == "cap"
    assert normalize_part_tag("body_top") == "body"
    assert normalize_part_tag("totally_unrelated_xyz") == "unknown"
    assert normalize_part_tag(None) == "unknown"


def test_camera_project_unproject_roundtrip_and_rig():
    c2w = look_at_cam_to_world(eye=[0.3, 0.2, 0.25], target=[0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1e-6)
    # Top-down singularity fallback
    c2w_top = look_at_cam_to_world(eye=[0.0, 0.0, 0.5], target=[0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.det(c2w_top[:3, :3]), 1.0, atol=1e-6)

    cams = build_multiview_cameras([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], n_azimuth=4, elevations_deg=(25.0,))
    assert len(cams) == 5  # 4 azimuth + 1 top
    pts = np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.03], [-0.03, 0.02, -0.02]])
    for cam in cams:
        u, v, z = cam.project(pts)
        assert np.all(z > 0.0)
        assert np.all((u >= 0) & (u < cam.width))
        assert np.all((v >= 0) & (v < cam.height))
        recovered = cam.unproject(u, v, z)
        assert np.allclose(recovered, pts, atol=1e-6)


def test_cpu_zbuffer_renderer_rgb_depth_and_occlusion():
    # Two boxes along +X: front box centered at x=+0.05, back box centered at x=-0.05
    vf, ff = make_box_mesh(0.04, 0.04, 0.04)
    vf = vf + np.array([0.05, 0.0, 0.0])
    vb, fb = make_box_mesh(0.04, 0.04, 0.04)
    vb = vb + np.array([-0.05, 0.0, 0.0])
    verts = np.vstack([vf, vb])
    faces = np.vstack([ff, fb + len(vf)])

    cam_c2w = look_at_cam_to_world(eye=[0.30, 0.0, 0.0], target=[0.0, 0.0, 0.0])
    from scripts.objects.affordance_vlm import CameraView, pinhole_intrinsics

    view = CameraView("front", 128, 128, pinhole_intrinsics(128, 128, 45.0), cam_c2w)
    rv = render_mesh_view(verts, faces, view)

    assert rv.rgb.shape == (128, 128, 3)
    assert rv.depth.shape == (128, 128)
    assert rv.mask.sum() > 200
    # Center pixel should hit the front face of the front box at x = 0.07 -> distance = 0.30 - 0.07 = 0.23 m
    center_depth = float(rv.depth[64, 64])
    assert abs(center_depth - 0.23) < 0.005

    # Back-projection occlusion check: a point on the front face is visible, a point on the occluded back box is not
    lmap = np.where(rv.mask, 1, 0).astype(np.int32)
    test_pts = np.array([[0.07, 0.0, 0.0], [-0.03, 0.0, 0.0]])
    tags, confs, cnts = backproject_2d_masks_to_points(test_pts, [rv], [lmap], id_to_tag={1: "body"}, depth_tol=0.01)
    assert tags[0] == "body" and cnts[0] == 1 and confs[0] == 1.0
    assert tags[1] == "unknown" and cnts[1] == 0


def test_multiview_2d_mask_backprojection_onto_mug_handle_body_rim():
    verts, faces = _make_mug_mesh()
    aff = annotate_mesh(verts, faces, asset_id="Rigid/mug/00001", n_surface_samples=260, max_grasps=20, seed=3)
    grasps = aff["grasps"]["dex1_1"]
    assert len(grasps) > 0

    # Add an explicit handle grasp and rim grasp to guarantee coverage of all semantic parts
    grasps.append(
        {
            "id": "g_handle",
            "position": [0.056, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "approach_dir": [-1.0, 0.0, 0.0],
            "closing_axis": [0.0, 1.0, 0.0],
            "width": 0.018,
            "part_tag": "handle_or_protrusion",
            "tag_source": "geometry",
            "score": 0.9,
            "source": "antipodal_geom",
            "verified": True,
            "sim_score": 0.92,
        }
    )
    grasps.append(
        {
            "id": "g_rim",
            "position": [0.0, 0.0, 0.044],
            "quaternion_wxyz": [0.7071, 0.0, 0.7071, 0.0],
            "approach_dir": [0.0, 0.0, -1.0],
            "closing_axis": [0.0, 1.0, 0.0],
            "width": 0.068,
            "part_tag": "body_top",
            "tag_source": "geometry",
            "score": 0.85,
            "source": "antipodal_geom",
            "verified": True,
            "sim_score": 0.88,
        }
    )

    cams = build_multiview_cameras(
        verts.min(axis=0), verts.max(axis=0), n_azimuth=6, elevations_deg=(20.0,), image_size=(128, 128)
    )
    rendered = render_mesh_views(verts, faces, cams)

    # Synthesize ground-truth 2D semantic masks by unprojecting each foreground pixel to 3D:
    #   x > 0.038 -> handle (1), z > 0.036 -> rim (2), else body (3)
    id_to_tag = {1: "handle", 2: "rim", 3: "body"}
    label_maps = []
    for rv in rendered:
        h, w = rv.depth.shape
        lmap = np.zeros((h, w), dtype=np.int32)
        ys, xs = np.nonzero(rv.mask)
        pts3d = rv.view.unproject(xs.astype(np.float64), ys.astype(np.float64), rv.depth[ys, xs].astype(np.float64))
        cls = np.full(len(pts3d), 3, dtype=np.int32)
        cls[pts3d[:, 2] > 0.036] = 2
        cls[pts3d[:, 0] > 0.038] = 1
        lmap[ys, xs] = cls
        label_maps.append(lmap)

    updated = assign_semantic_tags_to_grasps(
        aff,
        rendered,
        label_maps=label_maps,
        id_to_tag=id_to_tag,
        model_name="synthetic_seg",
    )
    by_id = {g["id"]: g for g in updated["grasps"]["dex1_1"]}
    assert by_id["g_handle"]["part_tag"] == "handle"
    assert by_id["g_handle"]["tag_source"] == "vlm"
    assert by_id["g_handle"]["part_tag_geom"] == "handle_or_protrusion"
    assert by_id["g_rim"]["part_tag"] == "rim"
    assert by_id["g_rim"]["tag_source"] == "vlm"
    assert by_id["g_rim"]["part_tag_geom"] == "body_top"
    assert updated["semantic_annotation"]["grasps_updated"] >= 2

    errs, _ = validate_affordance(updated)
    assert errs == []


def test_assign_semantic_tags_verified_only_and_no_unknown_overwrite():
    aff = {
        "schema_version": 1,
        "asset_id": "Rigid/cup/00001",
        "mesh_hash": "sha256:abc123",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.05], "max": [0.04, 0.04, 0.05], "extents": [0.08, 0.08, 0.10]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_ver",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.04,
                    "part_tag": "body_side",
                    "tag_source": "geometry",
                    "score": 0.8,
                    "verified": True,
                },
                {
                    "id": "g_unver",
                    "position": [0.0, 0.0, 0.02],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.04,
                    "part_tag": "body_top",
                    "tag_source": "geometry",
                    "score": 0.7,
                    "verified": False,
                },
                {
                    "id": "g_unseen",
                    "position": [0.0, 0.0, -0.02],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.04,
                    "part_tag": "body_low",
                    "tag_source": "geometry",
                    "score": 0.6,
                    "verified": True,
                },
            ]
        },
    }
    votes = {"g_ver": [("handle", 0.95)], "g_unver": [("rim", 0.90)], "g_unseen": [("unknown", 0.5)]}
    updated = assign_semantic_tags_to_grasps(aff, grasp_marker_votes=votes, verified_only=True)
    by_id = {g["id"]: g for g in updated["grasps"]["dex1_1"]}
    assert by_id["g_ver"]["part_tag"] == "handle"
    assert by_id["g_ver"]["tag_source"] == "vlm"
    # Unverified grasp skipped when verified_only=True
    assert by_id["g_unver"]["part_tag"] == "body_top"
    assert by_id["g_unver"]["tag_source"] == "geometry"
    # Unseen / unknown vote never overwrites existing geometry tag
    assert by_id["g_unseen"]["part_tag"] == "body_low"
    assert by_id["g_unseen"]["tag_source"] == "geometry"


def test_openai_vlm_labeler_markers_and_regions_mock_http(monkeypatch):
    verts, faces = make_box_mesh(0.05, 0.05, 0.08)
    aff = annotate_mesh(verts, faces, asset_id="Rigid/bottle/00001", n_surface_samples=120, max_grasps=6, seed=1)
    assert len(aff["grasps"]["dex1_1"]) > 0

    cams = build_multiview_cameras(
        verts.min(axis=0),
        verts.max(axis=0),
        n_azimuth=2,
        elevations_deg=(25.0,),
        include_top=False,
        image_size=(96, 96),
    )
    rendered = render_mesh_views(verts, faces, cams)
    vis = visible_grasps_in_view(aff["grasps"]["dex1_1"], rendered[0])
    assert len(vis) > 0
    overlay = render_grasp_markers_image(rendered[0], vis)
    assert overlay.size == (96, 96)

    # Mock HTTP response for marker classification
    class _FakeResp:
        def __init__(self, content: str):
            self._buf = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"))

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    target_idx = int(vis[0]["index"])
    marker_reply = f'```json\n{{"markers": {{"{target_idx}": {{"part": "bottleneck", "confidence": 0.92}}}}\n}}\n```'
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=90.0: _FakeResp(marker_reply))
    labeler = OpenAICompatibleVLMLabeler(api_url="http://mock.local/v1", api_key="test-key", model="mock-vlm")

    out_markers = annotate_affordance_with_vlm(aff, rendered_views=rendered, vlm_labeler=labeler, vlm_mode="markers")
    g_target = out_markers["grasps"]["dex1_1"][target_idx]
    assert g_target["part_tag"] == "neck"
    assert g_target["tag_source"] == "vlm"
    assert out_markers["semantic_annotation"]["model"] == "mock-vlm"

    # Mock HTTP response for region back-projection
    region_reply = json.dumps({"parts": [{"part": "body", "bbox_norm": [0.0, 0.0, 1.0, 1.0], "confidence": 0.85}]})
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=90.0: _FakeResp(region_reply))
    out_regions = annotate_affordance_with_vlm(aff, rendered_views=rendered, vlm_labeler=labeler, vlm_mode="regions")
    assert any(g["part_tag"] == "body" and g["tag_source"] == "vlm" for g in out_regions["grasps"]["dex1_1"])


def test_rasterize_regions_smaller_part_overrides_larger_body():
    mask = np.ones((100, 100), dtype=bool)
    regions = [
        {"part": "body", "bbox_norm": [0.1, 0.1, 0.9, 0.9], "confidence": 0.9},
        {"part": "handle", "bbox_norm": [0.7, 0.3, 0.9, 0.7], "confidence": 0.8},
    ]
    lmap, id_to_tag = rasterize_regions_to_label_map(regions, mask)
    tag_at_body = id_to_tag[int(lmap[50, 30])]
    tag_at_handle = id_to_tag[int(lmap[50, 80])]
    assert tag_at_body == "body"
    assert tag_at_handle == "handle"


def test_vlm_cli_render_save_load_and_annotate_with_masks(tmp_path: Path):
    verts, faces = make_box_mesh(0.05, 0.05, 0.08)
    obj_path = tmp_path / "object.obj"
    write_obj(obj_path, verts, faces)
    aff_path = tmp_path / "affordance.json"
    annotate_asset(tmp_path, output=aff_path, asset_id="Rigid/box/00001", n_surface_samples=120, max_grasps=6)

    views_dir = tmp_path / "views"
    rc = vlm_main(["render", str(tmp_path), "-o", str(views_dir), "--azimuths", "3", "--image-size", "64"])
    assert rc == 0
    assert (views_dir / "cameras.json").is_file()

    loaded = load_rendered_views(views_dir)
    assert len(loaded) == 7  # 3 azimuths * 2 elevations + 1 top

    # Write per-view region masks covering the object as "spout"
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    for rv in loaded:
        (masks_dir / f"{rv.view.view_id}_regions.json").write_text(
            json.dumps({"parts": [{"part": "spout", "bbox_norm": [0.0, 0.0, 1.0, 1.0], "confidence": 0.9}]}),
            encoding="utf-8",
        )

    out_aff = tmp_path / "affordance_vlm.json"
    rc2 = vlm_main(
        [
            "annotate",
            str(aff_path),
            "--mesh",
            str(tmp_path),
            "--views-dir",
            str(views_dir),
            "--masks-dir",
            str(masks_dir),
            "--backend",
            "masks",
            "-o",
            str(out_aff),
        ]
    )
    assert rc2 == 0
    doc = json.loads(out_aff.read_text(encoding="utf-8"))
    assert doc["semantic_annotation"]["grasps_updated"] > 0
    assert all(g["part_tag"] == "spout" and g["tag_source"] == "vlm" for g in doc["grasps"]["dex1_1"])

    # Also exercise save_rendered_views roundtrip explicitly
    views_dir2 = tmp_path / "views2"
    save_rendered_views(loaded[:2], views_dir2)
    assert len(load_rendered_views(views_dir2)) == 2


def test_openai_labeler_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENFLYWHEEL_VLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    labeler = OpenAICompatibleVLMLabeler(api_url="http://mock.local/v1", api_key="")
    with pytest.raises(RuntimeError, match="Missing VLM API key"):
        labeler._post_chat([{"role": "user", "content": "hi"}])


def test_normalize_part_tag_token_boundaries_no_substring_collision():
    assert normalize_part_tag("side_handle") == "handle"
    assert normalize_part_tag("flip_lid") == "lid"
    assert normalize_part_tag("bottom_handle") == "handle"
    assert normalize_part_tag("clip_handle") == "handle"
    assert normalize_part_tag("clear_plastic") == "unknown"
    assert normalize_part_tag("cup_lip") == "rim"
    assert normalize_part_tag("mug_handle_grip") == "handle"


def test_rasterize_regions_polygon_area_and_reversed_bbox():
    mask = np.ones((64, 64), dtype=bool)
    lmap, id_to_tag = rasterize_regions_to_label_map(
        [
            {"part": "body", "bbox_norm": [0.9, 0.9, 0.1, 0.1], "confidence": 0.9},
            {
                "part": "handle",
                "polygon_norm": [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]],
                "confidence": 0.8,
            },
        ],
        mask,
    )
    assert id_to_tag[int(lmap[20, 20])] == "handle"
    assert id_to_tag[int(lmap[48, 48])] == "body"


def test_multigripper_vlm_markers_no_grasp_id_collision():
    verts, faces = make_box_mesh(0.05, 0.05, 0.08)
    aff = annotate_mesh(verts, faces, asset_id="Rigid/box/00001", n_surface_samples=80, max_grasps=2)
    # Give two grippers identical grasp id "g000"
    g_base = aff["grasps"]["dex1_1"][0]
    aff["grasps"] = {
        "dex1_1": [dict(g_base, id="g000", part_tag="body_side")],
        "inspire": [dict(g_base, id="g000", part_tag="body_side")],
    }
    cams = build_multiview_cameras(
        aff["bounds"]["min"],
        aff["bounds"]["max"],
        n_azimuth=4,
        elevations_deg=(30.0,),
        include_top=True,
        image_size=(64, 64),
    )
    rendered = render_mesh_views(verts, faces, cams)

    class _StubLabeler:
        model = "stub-vlm"

        def classify_grasp_markers(self, rv, visible_grasps, *, object_hint=""):
            out = {}
            for item in visible_grasps:
                if item["id"].startswith("dex1_1:"):
                    out[item["id"]] = ("handle", 0.95)
                elif item["id"].startswith("inspire:"):
                    out[item["id"]] = ("rim", 0.95)
            return out

    updated = annotate_affordance_with_vlm(aff, rendered_views=rendered, vlm_labeler=_StubLabeler(), vlm_mode="markers")
    assert updated["grasps"]["dex1_1"][0]["part_tag"] == "handle"
    assert updated["grasps"]["inspire"][0]["part_tag"] == "rim"


def test_vlm_cli_batch_masks_root_and_error_guards_and_pxr_close(tmp_path: Path, monkeypatch):
    import scripts.objects.affordance as aff_mod

    asset_dir = tmp_path / "cup_00001"
    asset_dir.mkdir()
    verts, faces = make_box_mesh(0.05, 0.05, 0.08)
    write_obj(asset_dir / "object.obj", verts, faces)
    aff_path = asset_dir / "affordance.json"
    annotate_asset(asset_dir, output=aff_path, asset_id="Rigid/cup/00001", n_surface_samples=80, max_grasps=4)

    # annotate --backend masks without --masks-dir must fail fast
    with pytest.raises(SystemExit) as exc_ann:
        vlm_main(["annotate", str(aff_path), "--mesh", str(asset_dir), "--backend", "masks"])
    assert exc_ann.value.code != 0

    # batch --backend masks without --masks-root must fail fast
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "asset_id": "Rigid/cup/00001",
                    "status": "ok",
                    "asset_dir": str(asset_dir),
                    "affordance_path": str(aff_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_bat:
        vlm_main(["batch", "--manifest", str(manifest), "--backend", "masks"])
    assert exc_bat.value.code != 0

    # batch --backend masks --masks-root works and closes _PXR_APP
    cams = build_multiview_cameras(
        verts.min(axis=0), verts.max(axis=0), n_azimuth=6, elevations_deg=(20.0, 50.0), image_size=(256, 256)
    )
    masks_root = tmp_path / "masks_root"
    mdir = masks_root / "Rigid/cup/00001"
    mdir.mkdir(parents=True)
    for cam in cams:
        (mdir / f"{cam.view_id}_regions.json").write_text(
            json.dumps({"parts": [{"part": "lid", "bbox_norm": [0.0, 0.0, 1.0, 1.0], "confidence": 0.9}]}),
            encoding="utf-8",
        )

    closed = []

    class _FakeApp:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(aff_mod, "_PXR_APP", _FakeApp())
    rc = vlm_main(["batch", "--manifest", str(manifest), "--backend", "masks", "--masks-root", str(masks_root)])
    assert rc == 0
    assert closed == [True]
    assert aff_mod._PXR_APP is None
    doc = json.loads(aff_path.read_text(encoding="utf-8"))
    assert doc["semantic_annotation"]["grasps_updated"] > 0
    assert all(g["part_tag"] == "lid" and g["tag_source"] == "vlm" for g in doc["grasps"]["dex1_1"])


def test_render_isaac_views_freezes_rigid_bodies_pauses_timeline_and_handles_res_change(tmp_path: Path, monkeypatch):
    import sys
    import types

    from scripts.objects.affordance_vlm import render_isaac_views

    asset_dir = tmp_path / "mug_00001"
    asset_dir.mkdir()
    (asset_dir / "object.usdz").write_bytes(b"fake")

    events: dict[str, list] = {
        "timeline_paused": [],
        "rb_kinematic": [],
        "rb_enabled": [],
        "step_kwargs": [],
        "rp_resolutions": [],
        "rp_destroyed": [],
    }

    class _FakeAttr:
        def Set(self, val):
            self.val = val

    class _FakePrim:
        def __init__(self, has_rb: bool):
            self._has_rb = has_rb

        def IsValid(self):
            return True

        def HasAPI(self, _api):
            return self._has_rb

    class _FakeRigidBodyAPI:
        def __init__(self, _prim):
            pass

        def CreateKinematicEnabledAttr(self, val):
            events["rb_kinematic"].append(val)

        def CreateRigidBodyEnabledAttr(self, val):
            events["rb_enabled"].append(val)

    class _FakeStage:
        def __init__(self):
            self._prims = [_FakePrim(False), _FakePrim(True)]

        def GetPrimAtPath(self, _path):
            return self._prims[0]

    fake_stage = _FakeStage()

    class _FakeCamera:
        def __init__(self):
            self._v = _FakeAttr()
            self._f = _FakeAttr()

        @classmethod
        def Define(cls, _stage, _path):
            return cls()

        def CreateHorizontalApertureAttr(self):
            return _FakeAttr()

        def CreateVerticalApertureAttr(self):
            return self._v

        def GetVerticalApertureAttr(self):
            return self._v

        def CreateFocalLengthAttr(self):
            return self._f

        def GetFocalLengthAttr(self):
            return self._f

        def CreateClippingRangeAttr(self):
            return _FakeAttr()

        def GetPrim(self):
            return _FakePrim(False)

    class _FakeXformable:
        def __init__(self, _prim):
            pass

        def AddTransformOp(self):
            return _FakeAttr()

    class _FakeRP:
        def __init__(self, res):
            self.res = res

        def destroy(self):
            events["rp_destroyed"].append(self.res)

    current_rp: list[_FakeRP] = []

    class _FakeAnnotator:
        def __init__(self, kind: str):
            self.kind = kind

        def attach(self, rp):
            self._rp = rp

        def detach(self):
            pass

        def get_data(self):
            w, h = self._rp.res
            if self.kind == "rgb":
                return np.full((h, w, 4), 128, dtype=np.uint8)
            return np.full((h, w), 0.35, dtype=np.float32)

    # Wire fake Isaac Sim / USD / Replicator modules into sys.modules
    carb_mod = types.ModuleType("carb")
    carb_mod.settings = types.SimpleNamespace(
        get_settings=lambda: types.SimpleNamespace(set_float=lambda *a, **kw: None)
    )

    sim_utils_mod = types.ModuleType("isaaclab.sim")
    sim_utils_mod.SimulationContext = types.SimpleNamespace(instance=lambda: None, clear_instance=lambda: None)
    sim_utils_mod.DomeLightCfg = lambda **kw: types.SimpleNamespace(func=lambda *a, **k: None)
    sim_utils_mod.DistantLightCfg = lambda **kw: types.SimpleNamespace(func=lambda *a, **k: None)
    sim_utils_mod.UsdFileCfg = lambda **kw: types.SimpleNamespace(func=lambda *a, **k: None)
    isaaclab_mod = types.ModuleType("isaaclab")
    isaaclab_mod.sim = sim_utils_mod

    timeline_mod = types.ModuleType("omni.timeline")
    timeline_mod.get_timeline_interface = lambda: types.SimpleNamespace(
        pause=lambda: events["timeline_paused"].append(True)
    )

    usd_ctx_mod = types.ModuleType("omni.usd")
    usd_ctx_mod.get_context = lambda: types.SimpleNamespace(new_stage=lambda: None, get_stage=lambda: fake_stage)

    def _create_rp(_cam_path, res):
        events["rp_resolutions"].append(res)
        rp = _FakeRP(res)
        current_rp.append(rp)
        return rp

    rep_mod = types.ModuleType("omni.replicator.core")
    rep_mod.create = types.SimpleNamespace(render_product=_create_rp)
    rep_mod.AnnotatorRegistry = types.SimpleNamespace(get_annotator=lambda kind: _FakeAnnotator(kind))
    rep_mod.orchestrator = types.SimpleNamespace(step=lambda **kw: events["step_kwargs"].append(kw))

    omni_mod = types.ModuleType("omni")
    omni_rep_pkg = types.ModuleType("omni.replicator")
    omni_rep_pkg.core = rep_mod
    omni_mod.replicator = omni_rep_pkg
    omni_mod.timeline = timeline_mod
    omni_mod.usd = usd_ctx_mod

    pxr_mod = types.ModuleType("pxr")
    pxr_mod.Gf = types.SimpleNamespace(Vec2f=lambda a, b: (a, b), Matrix4d=lambda *vals: vals)
    pxr_mod.Usd = types.SimpleNamespace(PrimRange=lambda _root: fake_stage._prims)
    pxr_mod.UsdGeom = types.SimpleNamespace(Camera=_FakeCamera, Xformable=_FakeXformable)
    pxr_mod.UsdPhysics = types.SimpleNamespace(RigidBodyAPI=_FakeRigidBodyAPI)

    for name, mod in [
        ("carb", carb_mod),
        ("isaaclab", isaaclab_mod),
        ("isaaclab.sim", sim_utils_mod),
        ("omni", omni_mod),
        ("omni.replicator", omni_rep_pkg),
        ("omni.replicator.core", rep_mod),
        ("omni.timeline", timeline_mod),
        ("omni.usd", usd_ctx_mod),
        ("pxr", pxr_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    cams_a = build_multiview_cameras(
        [-0.05, -0.05, 0.0],
        [0.05, 0.05, 0.1],
        n_azimuth=1,
        elevations_deg=(20.0,),
        include_top=False,
        image_size=(64, 64),
    )
    cams_b = build_multiview_cameras(
        [-0.05, -0.05, 0.0],
        [0.05, 0.05, 0.1],
        n_azimuth=1,
        elevations_deg=(50.0,),
        include_top=False,
        image_size=(32, 48),
    )
    views = render_isaac_views(asset_dir, [cams_a[0], cams_b[0]], warmup_renders=6)

    assert len(views) == 2
    assert views[0].rgb.shape == (64, 64, 3) and views[0].depth.shape == (64, 64)
    assert views[1].rgb.shape == (48, 32, 3) and views[1].depth.shape == (48, 32)
    assert np.allclose(views[1].depth, 0.35)
    assert events["timeline_paused"] == [True]
    assert events["rb_kinematic"] == [True]
    assert events["rb_enabled"] == [False]
    assert all(kw == {"delta_time": 0.0, "pause_timeline": True} for kw in events["step_kwargs"])
    assert events["rp_resolutions"] == [(64, 64), (32, 48)]
    assert events["rp_destroyed"] == [(64, 64), (32, 48)]
