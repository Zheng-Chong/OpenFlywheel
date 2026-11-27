"""Unit and CLI tests for 2D affordance projection, heatmaps, BBoxes, overlays, and VLM export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.objects.affordance import annotate_mesh
from scripts.objects.affordance_2d import (
    colorize_heatmap,
    export_views_to_vlm_jsonl,
    main as aff2d_main,
    project_affordance_2d,
    project_object_bbox_2d,
    project_world_points,
    render_affordance_2d_overlay,
    usd_to_opencv_cam_to_world,
)
from scripts.objects.affordance_vlm import (
    CameraView,
    build_multiview_cameras,
    look_at_cam_to_world,
    pinhole_intrinsics,
    render_mesh_view,
    render_mesh_views,
    save_rendered_views,
)
from scripts.objects.test_affordance import make_box_mesh, make_cylinder_mesh, merge_meshes, write_obj


def _make_mug_mesh():
    body = make_cylinder_mesh(radius=0.038, height=0.095, n_seg=24, n_rings=4)
    handle = make_box_mesh(0.026, 0.014, 0.052, center=(0.054, 0.0, 0.0))
    return merge_meshes(body, handle)


def _sample_multi_hand_affordance() -> dict:
    return {
        "schema_version": 1,
        "asset_id": "Rigid/mug/00001",
        "mesh_hash": "sha256:deadbeef",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.05], "max": [0.068, 0.04, 0.05], "extents": [0.108, 0.08, 0.10]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_handle",
                    "position": [0.055, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.016,
                    "part_tag": "handle",
                    "score": 0.7,
                    "sim_score": 0.95,
                    "verified": True,
                },
                {
                    "id": "g_body_ok",
                    "position": [0.0, 0.036, 0.01],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.072,
                    "part_tag": "body",
                    "score": 0.65,
                    "sim_score": 0.80,
                    "verified": True,
                },
                {
                    "id": "g_rim_failed",
                    "position": [0.0, 0.0, 0.048],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.012,
                    "part_tag": "rim",
                    "score": 0.50,
                    "sim_score": 0.05,
                    "verified": False,
                },
            ],
            "sharpa_wave": [
                {
                    "id": "sw_power_0",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.076,
                    "part_tag": "body",
                    "grasp_type": "power_cylindrical",
                    "score": 0.88,
                    "sim_score": 0.92,
                    "verified": True,
                }
            ],
            "wuji_hand2_2": [
                {
                    "id": "wj_pinch_0",
                    "position": [0.055, 0.0, 0.015],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.018,
                    "part_tag": "handle",
                    "grasp_type": "pinch",
                    "score": 0.82,
                    "verified": False,
                }
            ],
        },
    }


def test_project_multi_gripper_and_filters():
    aff = _sample_multi_hand_affordance()
    w, h = 320, 240
    K = pinhole_intrinsics(w, h, fov_deg=50.0)
    c2w = look_at_cam_to_world(eye=[0.35, 0.25, 0.18], target=[0.0, 0.0, 0.0])

    # Default: all grippers, skips sim-failed g_rim_failed
    res = project_affordance_2d(
        aff,
        obj_pos=[0.0, 0.0, 0.0],
        obj_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        K=K,
        cam_to_world=c2w,
        width=w,
        height=h,
        per_part_heatmaps=True,
        selected_grasp_world=[0.055, 0.0, 0.0],
    )
    assert res.heatmap.shape == (h, w)
    assert res.heatmap.dtype == np.float32
    assert res.heatmap_uint8.dtype == np.uint8
    assert float(res.heatmap.max()) == pytest.approx(1.0, abs=1e-5)
    assert int(res.heatmap_uint8.max()) == 255
    assert res.object_bbox_xyxy is not None
    assert res.affordance_bbox_xyxy is not None
    assert res.object_bbox_xyxy_norm is not None
    assert res.affordance_bbox_xyxy_norm is not None
    for box in (res.object_bbox_xyxy_norm, res.affordance_bbox_xyxy_norm):
        assert 0.0 <= box[0] < box[2] <= 1.0
        assert 0.0 <= box[1] < box[3] <= 1.0

    ids = [kp["grasp_id"] for kp in res.keypoints]
    assert "g_handle" in ids
    assert "sw_power_0" in ids
    assert "wj_pinch_0" in ids  # never sim-tested, so kept when verified_only=False
    assert "g_rim_failed" not in ids  # sim-failed excluded by default
    # Highest verified sim_score first
    assert res.keypoints[0]["grasp_id"] == "g_handle"
    assert res.keypoints[0]["score"] == pytest.approx(0.95, abs=1e-4)
    assert "handle" in res.part_heatmaps and "body" in res.part_heatmaps
    assert res.selected_grasp_2d is not None and res.selected_grasp_2d["visible"]

    # Verified-only + gripper filter for sharpa_wave
    res_sw = project_affordance_2d(
        aff,
        obj_pos=[0.0, 0.0, 0.0],
        obj_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        K=K,
        cam_to_world=c2w,
        width=w,
        height=h,
        gripper_types=("sharpa_wave",),
        verified_only=True,
    )
    assert len(res_sw.keypoints) == 1
    assert res_sw.keypoints[0]["grasp_type"] == "power_cylindrical"
    assert res_sw.keypoints[0]["gripper_type"] == "sharpa_wave"

    # Part-tag filter
    res_handle = project_affordance_2d(
        aff,
        obj_pos=[0.0, 0.0, 0.0],
        obj_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        K=K,
        cam_to_world=c2w,
        width=w,
        height=h,
        part_tags=("handle",),
    )
    assert all(kp["part_tag"] == "handle" for kp in res_handle.keypoints)


def test_usd_and_opencv_camera_conventions_match():
    w, h = 320, 240
    K = pinhole_intrinsics(w, h, fov_deg=50.0)
    c2w_cv = look_at_cam_to_world(eye=[0.4, 0.1, 0.2], target=[0.0, 0.0, 0.0])
    # USD camera has +Y up, -Z forward => c2w_usd = c2w_cv @ diag(1, -1, -1, 1)
    c2w_usd = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(usd_to_opencv_cam_to_world(c2w_usd), c2w_cv, atol=1e-12)

    pts = np.array([[0.0, 0.0, 0.0], [0.05, -0.02, 0.03]])
    u1, v1, z1 = project_world_points(pts, K, c2w_cv, camera_convention="opencv")
    u2, v2, z2 = project_world_points(pts, K, c2w_usd, camera_convention="usd")
    np.testing.assert_allclose(u1, u2, atol=1e-9)
    np.testing.assert_allclose(v1, v2, atol=1e-9)
    np.testing.assert_allclose(z1, z2, atol=1e-9)


def test_object_bbox_edge_cases_behind_and_straddling_camera():
    w, h = 320, 240
    K = pinhole_intrinsics(w, h, fov_deg=60.0)
    # Camera at origin looking along +Z in OpenCV
    c2w = np.eye(4, dtype=np.float64)

    # Completely behind camera
    px, norm = project_object_bbox_2d(
        [-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], [0.0, 0.0, -0.5], [1, 0, 0, 0], K, c2w, w, h
    )
    assert px is None and norm is None

    # Completely off-screen to the right
    px, norm = project_object_bbox_2d(
        [-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], [5.0, 0.0, 0.5], [1, 0, 0, 0], K, c2w, w, h
    )
    assert px is None and norm is None

    # Straddling camera near plane (z in [-0.1, +0.2]) -> should clip cleanly and stay within [0, W]x[0, H]
    px, norm = project_object_bbox_2d(
        [-0.05, -0.05, -0.15], [0.05, 0.05, 0.15], [0.0, 0.0, 0.05], [1, 0, 0, 0], K, c2w, w, h
    )
    assert px is not None and norm is not None
    assert 0.0 <= px[0] < px[2] <= w
    assert 0.0 <= px[1] < px[3] <= h

    # Non-finite object position must return (None, None), never NaN coordinates
    px_nan, norm_nan = project_object_bbox_2d(
        [-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], [float("nan"), 0.0, 0.5], [1, 0, 0, 0], K, c2w, w, h
    )
    assert px_nan is None and norm_nan is None


def test_depth_occlusion_culls_back_grasp_keeps_front_grasp():
    # Two boxes along +X: front box at x=+0.06, back box at x=-0.06
    vf, ff = make_box_mesh(0.04, 0.06, 0.06)
    vf = vf + np.array([0.06, 0.0, 0.0])
    vb, fb = make_box_mesh(0.04, 0.06, 0.06)
    vb = vb + np.array([-0.06, 0.0, 0.0])
    verts = np.vstack([vf, vb])
    faces = np.vstack([ff, fb + len(vf)])

    w, h = 160, 160
    K = pinhole_intrinsics(w, h, fov_deg=45.0)
    c2w = look_at_cam_to_world(eye=[0.35, 0.0, 0.0], target=[0.0, 0.0, 0.0])
    rv = render_mesh_view(verts, faces, CameraView("front", w, h, K, c2w))

    aff = {
        "asset_id": "test/two_box",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.08, -0.03, -0.03], "max": [0.08, 0.03, 0.03]},
        "symmetry": {"z_axis": "none"},
        "grasps": {
            "dex1_1": [
                {
                    "id": "front_grasp",
                    "position": [0.06, 0.0, 0.0],  # inside front box (front face at 0.08, 2cm behind surface)
                    "width": 0.04,
                    "part_tag": "body",
                    "score": 0.9,
                    "verified": True,
                },
                {
                    "id": "back_grasp",
                    "position": [-0.06, 0.0, 0.0],  # occluded by front box (14cm behind front face!)
                    "width": 0.04,
                    "part_tag": "body",
                    "score": 0.95,
                    "verified": True,
                },
            ]
        },
    }
    res = project_affordance_2d(
        aff,
        obj_pos=[0.0, 0.0, 0.0],
        obj_quat_wxyz=[1, 0, 0, 0],
        K=K,
        cam_to_world=c2w,
        width=w,
        height=h,
        depth=rv.depth,
        depth_tol=0.02,
    )
    ids = [kp["grasp_id"] for kp in res.keypoints]
    assert ids == ["front_grasp"]


def test_symmetry_expansion_compact_heatmap_and_empty_grasps():
    w, h = 320, 240
    K = pinhole_intrinsics(w, h, fov_deg=50.0)
    c2w = look_at_cam_to_world(eye=[0.35, 0.0, 0.15], target=[0.0, 0.0, 0.0])

    aff_sym = {
        "asset_id": "Rigid/cup/00001",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.05], "max": [0.04, 0.04, 0.05]},
        "symmetry": {"z_axis": "continuous", "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [{"id": "g0", "position": [0.035, 0.0, 0.01], "width": 0.04, "part_tag": "body", "score": 0.8}]
        },
    }
    res_sym = project_affordance_2d(
        aff_sym,
        [0, 0, 0],
        [1, 0, 0, 0],
        K,
        c2w,
        w,
        h,
        expand_symmetry=True,
        continuous_sym_samples=8,
        heatmap_hw=(48, 64),
    )
    assert len(res_sym.keypoints) == 8
    assert res_sym.heatmap.shape == (48, 64)
    assert res_sym.heatmap_uint8.shape == (48, 64)
    resized = res_sym.resize_heatmap_uint8(120, 160)
    assert resized.shape == (120, 160) and resized.dtype == np.uint8

    # On-axis grasp on a continuous-symmetry object must NOT duplicate 8 identical keypoints
    aff_on_axis = {
        **aff_sym,
        "grasps": {
            "dex1_1": [
                {"id": "g_center_body", "position": [0.0, 0.0, 0.0], "width": 0.06, "part_tag": "body", "score": 0.9},
                {"id": "g_center_neck", "position": [0.0, 0.0, 0.04], "width": 0.03, "part_tag": "neck", "score": 0.8},
            ]
        },
    }
    res_on_axis = project_affordance_2d(
        aff_on_axis,
        [0, 0, 0],
        [1, 0, 0, 0],
        K,
        c2w,
        w,
        h,
        expand_symmetry=True,
        continuous_sym_samples=8,
    )
    assert [kp["grasp_id"] for kp in res_on_axis.keypoints] == ["g_center_body", "g_center_neck"]

    # Empty grasps -> zero heatmap, None affordance bbox, valid object bbox
    aff_empty = {**aff_sym, "grasps": {"dex1_1": []}}
    res_empty = project_affordance_2d(aff_empty, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h)
    assert res_empty.keypoints == []
    assert res_empty.affordance_bbox_xyxy is None
    assert res_empty.object_bbox_xyxy is not None
    assert float(res_empty.heatmap.max()) == 0.0


def test_render_overlay_and_colormaps():
    aff = _sample_multi_hand_affordance()
    w, h = 256, 192
    K = pinhole_intrinsics(w, h, fov_deg=50.0)
    c2w = look_at_cam_to_world(eye=[0.32, 0.20, 0.16], target=[0.0, 0.0, 0.0])
    # Compact heatmap overlaid onto full-res RGB
    res = project_affordance_2d(
        aff,
        [0, 0, 0],
        [1, 0, 0, 0],
        K,
        c2w,
        w,
        h,
        heatmap_hw=(48, 64),
        selected_grasp_world=[0.055, 0.0, 0.0],
    )
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)
    for cmap in ("turbo", "jet"):
        cm_img = colorize_heatmap(res.heatmap, colormap=cmap)
        assert cm_img.shape == (48, 64, 3) and cm_img.dtype == np.uint8
        ov = render_affordance_2d_overlay(rgb, res, colormap=cmap)
        assert ov.shape == (h, w, 3) and ov.dtype == np.uint8
        assert not np.array_equal(ov, rgb)


def test_cli_render_and_export_vlm_roundtrip(tmp_path: Path):
    verts, faces = _make_mug_mesh()
    obj_path = tmp_path / "mug.obj"
    write_obj(obj_path, verts, faces)
    aff = annotate_mesh(verts, faces, asset_id="Rigid/mug/00001", n_surface_samples=220, max_grasps=16, seed=7)
    aff_path = tmp_path / "affordance.json"
    aff_path.write_text(json.dumps(aff), encoding="utf-8")

    # 1. CLI render from --mesh
    out_mesh = tmp_path / "out_mesh"
    rc = aff2d_main(
        [
            "render",
            str(aff_path),
            "--mesh",
            str(obj_path),
            "-o",
            str(out_mesh),
            "--image-size",
            "160",
            "--n-azimuth",
            "3",
        ]
    )
    assert rc == 0
    views_meta = json.loads((out_mesh / "affordance_2d_views.json").read_text(encoding="utf-8"))
    assert len(views_meta) == 7  # 3 az * 2 elev + 1 top
    assert (out_mesh / f"{views_meta[0]['view_id']}_overlay.png").is_file()
    assert (out_mesh / f"{views_meta[0]['view_id']}_heatmap.png").is_file()

    # 2. Pre-rendered views + CLI export-vlm
    cams = build_multiview_cameras(
        verts.min(axis=0), verts.max(axis=0), n_azimuth=3, elevations_deg=(25.0,), image_size=(128, 128)
    )
    rendered = render_mesh_views(verts, faces, cams)
    views_dir = tmp_path / "views"
    save_rendered_views(rendered, views_dir)

    out_vlm = tmp_path / "out_vlm"
    rc2 = aff2d_main(
        [
            "export-vlm",
            "--affordance",
            str(aff_path),
            "--views-dir",
            str(views_dir),
            "-o",
            str(out_vlm),
            "--heatmap-size",
            "64x48",
            "--save-overlays",
        ]
    )
    assert rc2 == 0
    lines = (out_vlm / "annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(rendered)
    first = json.loads(lines[0])
    assert first["heatmap_hw"] == [48, 64]
    hm_img = Image.open(out_vlm / first["heatmap_path"])
    assert hm_img.size == (64, 48)
    assert (out_vlm / first["overlay_path"]).is_file()

    # Direct Python API check
    summary = export_views_to_vlm_jsonl(aff_path, views_dir, tmp_path / "out_vlm_api")
    assert summary["records"] == len(rendered)


def test_contact_point_projection_avoids_hollow_center_and_matches_quaternion():
    w, h = 256, 256
    K = pinhole_intrinsics(w, h, fov_deg=45.0)
    # Top-down camera looking straight down at a hollow mug (outer radius ~0.032m)
    c2w = look_at_cam_to_world(eye=[0.0, 0.0, 0.25], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    aff_mug = {
        "asset_id": "Rigid/mug/test_hollow",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.035, -0.035, -0.04], "max": [0.035, 0.035, 0.04]},
        "symmetry": {"z_axis": "none"},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_x",
                    "position": [0.0, 0.0, 0.01],
                    "closing_axis": [1.0, 0.0, 0.0],
                    "width": 0.064,
                    "part_tag": "body_top",
                    "score": 0.95,
                    "verified": True,
                },
                {
                    "id": "g_y",
                    "position": [0.0, 0.0, 0.01],
                    "closing_axis": [0.0, 1.0, 0.0],
                    "width": 0.064,
                    "part_tag": "body_top",
                    "score": 0.90,
                    "verified": True,
                },
            ]
        },
    }

    res_contact = project_affordance_2d(aff_mug, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, use_contact_points=True)
    # Center of image is the hollow mug interior (128, 128) -> below overlay visibility threshold (0.08)
    assert float(res_contact.heatmap[h // 2, w // 2]) < 0.08
    # Keypoints must be anchored on visible wall contacts, not at hollow center (128, 128)
    assert len(res_contact.keypoints) == 2
    for kp in res_contact.keypoints:
        assert "contact_points_2d" in kp and len(kp["contact_points_2d"]) == 2
        dist_center = float(np.hypot(kp["u"] - w / 2.0, kp["v"] - h / 2.0))
        assert dist_center > 20.0
        assert float(res_contact.heatmap[int(round(kp["v"])), int(round(kp["u"]))]) > 0.8

    # Legacy midpoint-only mode puts peak right at hollow center
    res_mid = project_affordance_2d(aff_mug, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, use_contact_points=False)
    assert float(res_mid.heatmap[h // 2, w // 2]) > 0.8

    # Fallback to quaternion_wxyz (+Y_ee) when closing_axis is absent must match explicit closing_axis
    # 90 deg rotation about +Z maps +Y_ee -> [-1, 0, 0]
    q_z90 = [float(np.cos(np.pi / 4)), 0.0, 0.0, float(np.sin(np.pi / 4))]
    aff_quat_only = {
        **aff_mug,
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_q",
                    "position": [0.0, 0.0, 0.01],
                    "quaternion_wxyz": q_z90,
                    "width": 0.064,
                    "part_tag": "body_top",
                    "score": 0.95,
                    "verified": True,
                }
            ]
        },
    }
    aff_ca_only = {
        **aff_mug,
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_q",
                    "position": [0.0, 0.0, 0.01],
                    "closing_axis": [-1.0, 0.0, 0.0],
                    "width": 0.064,
                    "part_tag": "body_top",
                    "score": 0.95,
                    "verified": True,
                }
            ]
        },
    }
    r_q = project_affordance_2d(aff_quat_only, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h)
    r_ca = project_affordance_2d(aff_ca_only, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h)
    np.testing.assert_allclose(r_q.heatmap, r_ca.heatmap, atol=1e-5)


def test_foreground_mask_and_depth_clipping_and_symmetry_contact_ring():
    w, h = 200, 200
    K = pinhole_intrinsics(w, h, fov_deg=45.0)
    c2w = look_at_cam_to_world(eye=[0.0, 0.0, 0.25], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    # Continuous Z-symmetry on-axis grasp with closing_axis -> 1 deduped keypoint, full rim contact ring
    aff_sym_contact = {
        "asset_id": "Rigid/cup/sym_ring",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.032, -0.032, -0.04], "max": [0.032, 0.032, 0.04]},
        "symmetry": {"z_axis": "continuous", "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_ring",
                    "position": [0.0, 0.0, 0.01],
                    "closing_axis": [1.0, 0.0, 0.0],
                    "width": 0.060,
                    "part_tag": "body",
                    "score": 0.9,
                    "verified": True,
                }
            ]
        },
    }

    # Explicit ring-like fg_mask (annulus centered at image center)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - w / 2.0, yy - h / 2.0)
    fg_mask = (rr >= 24.0) & (rr <= 38.0)

    res = project_affordance_2d(
        aff_sym_contact,
        [0, 0, 0],
        [1, 0, 0, 0],
        K,
        c2w,
        w,
        h,
        fg_mask=fg_mask,
        per_part_heatmaps=True,
        continuous_sym_samples=8,
    )
    assert len(res.keypoints) == 1
    assert res.keypoints[0]["grasp_id"] == "g_ring"
    # On-axis continuous symmetry with 8 samples must deduplicate p1(ang + pi) == p2(ang) to 8 unique contacts
    assert len(res.keypoints[0]["contact_points_2d"]) == 8
    assert len({tuple(pt) for pt in res.keypoints[0]["contact_points_2d"]}) == 8
    assert float(res.heatmap[~fg_mask].max()) == 0.0
    assert float(res.part_heatmaps["body"][~fg_mask].max()) == 0.0
    assert float(res.heatmap.max()) == pytest.approx(1.0, abs=1e-5)
    assert res.affordance_bbox_xyxy is not None
    # Affordance BBox must stay tightly inside the foreground mask bounds
    fg_ys, fg_xs = np.nonzero(fg_mask)
    ax1, ay1, ax2, ay2 = res.affordance_bbox_xyxy
    assert ax1 >= float(fg_xs.min()) and ax2 <= float(fg_xs.max() + 1)
    assert ay1 >= float(fg_ys.min()) and ay2 <= float(fg_ys.max() + 1)

    # Depth-derived foreground clipping (inf background) zeros background cleanly
    depth = np.full((h, w), np.inf, dtype=np.float32)
    depth[fg_mask] = 0.24
    res_depth = project_affordance_2d(aff_sym_contact, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, depth=depth)
    assert float(res_depth.heatmap[~fg_mask].max()) == 0.0
    assert float(res_depth.heatmap.max()) == pytest.approx(1.0, abs=1e-5)

    # Compact heatmap_hw with fg_mask works cleanly
    res_compact = project_affordance_2d(
        aff_sym_contact, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, fg_mask=fg_mask, heatmap_hw=(50, 50)
    )
    assert res_compact.heatmap.shape == (50, 50)
    assert float(res_compact.heatmap.max()) == pytest.approx(1.0, abs=1e-5)

    # Empty fg_mask -> zero heatmap, empty keypoints, and None affordance bbox
    empty_mask = np.zeros((h, w), dtype=bool)
    res_empty_mask = project_affordance_2d(aff_sym_contact, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, fg_mask=empty_mask)
    assert float(res_empty_mask.heatmap.max()) == 0.0
    assert res_empty_mask.keypoints == []
    assert res_empty_mask.affordance_bbox_xyxy is None

    # clip_to_foreground=False must ignore explicit fg_mask for clipping/culling
    res_unclipped = project_affordance_2d(
        aff_sym_contact, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, fg_mask=empty_mask, clip_to_foreground=False
    )
    assert float(res_unclipped.heatmap.max()) == pytest.approx(1.0, abs=1e-5)
    assert len(res_unclipped.keypoints) == 1

    # Mismatched fg_mask shape raises ValueError
    with pytest.raises(ValueError, match="fg_mask shape"):
        project_affordance_2d(
            aff_sym_contact, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, fg_mask=np.ones((10, 10), dtype=bool)
        )


def test_cluttered_depth_band_grazing_contact_and_bbox_fallback():
    w, h = 200, 200
    K = pinhole_intrinsics(w, h, fov_deg=45.0)
    c2w = look_at_cam_to_world(eye=[0.0, 0.0, 0.40], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    aff = {
        "asset_id": "test/cluttered_obj",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.03, -0.03, -0.04], "max": [0.03, 0.03, 0.04]},
        "symmetry": {"z_axis": "continuous", "axis_xy": [0.0, 0.0]},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g0",
                    "position": [0.0, 0.0, 0.0],
                    "closing_axis": [1.0, 0.0, 0.0],
                    "width": 0.06,
                    "part_tag": "body",
                    "score": 0.9,
                    "verified": True,
                }
            ]
        },
    }

    # Cluttered depth map: far table at z=0.65, target object at z=0.40, closer robot/distractor at z=0.15
    yy, xx = np.mgrid[0:h, 0:w]
    depth = np.full((h, w), 0.65, dtype=np.float32)
    obj_region = (xx - 100) ** 2 + (yy - 100) ** 2 <= 25**2
    depth[obj_region] = 0.40
    closer_occluder = xx >= 105
    depth[closer_occluder] = 0.15
    far_table = (~obj_region) & (~closer_occluder)

    res_clut = project_affordance_2d(aff, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, depth=depth)
    assert float(res_clut.heatmap.max()) == pytest.approx(1.0, abs=1e-5)
    assert float(res_clut.heatmap[closer_occluder].max()) == 0.0
    assert float(res_clut.heatmap[far_table].max()) == 0.0

    # Grazing contact point at a sharp depth step (center pixel is 3.2cm closer than contact,
    # while adjacent side-wall pixel 2px away matches contact depth) stays visible.
    aff_grazing = {
        "asset_id": "test/grazing",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.03, -0.03, -0.04], "max": [0.03, 0.03, 0.04]},
        "symmetry": {"z_axis": "none"},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_graze",
                    "position": [0.0, 0.0, 0.0],
                    "closing_axis": [1.0, 0.0, 0.0],
                    "width": 0.06,
                    "part_tag": "body",
                    "score": 0.9,
                    "verified": True,
                }
            ]
        },
    }
    depth_g = np.full((h, w), np.inf, dtype=np.float32)
    fg_g = np.zeros((h, w), dtype=bool)
    # Left contact at u~118, right contact at u~82 (camera at +Z, up=+Y -> +X is left or right)
    u_pts, v_pts, _ = project_world_points(np.array([[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]]), K, c2w)
    u_a, v_a = int(round(u_pts[0])), int(round(v_pts[0]))
    u_b, v_b = int(round(u_pts[1])), int(round(v_pts[1]))
    # Object depth is 0.365 (3.5cm closer than z=0.40) at exact pixel (u_a, v_a), but 0.40 two pixels away
    fg_g[v_a - 10 : v_a + 11, min(u_a, u_b) - 4 : max(u_a, u_b) + 5] = True
    depth_g[fg_g] = 0.365
    depth_g[v_a, u_a + 2] = 0.40
    depth_g[v_b, u_b] = 0.40
    res_graze = project_affordance_2d(
        aff_grazing, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, depth=depth_g, fg_mask=fg_g, depth_tol=0.03
    )
    assert len(res_graze.keypoints) == 1
    assert len(res_graze.keypoints[0]["contact_points_2d"]) == 2

    # Verified grasp outside fg_mask + unverified grasp inside fg_mask -> bbox falls back cleanly to heatmap
    aff_mix = {
        "asset_id": "test/mix",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.04], "max": [0.04, 0.04, 0.04]},
        "symmetry": {"z_axis": "none"},
        "grasps": {
            "dex1_1": [
                {
                    "id": "v_out",
                    "position": [-0.03, 0.0, 0.0],
                    "width": 0.01,
                    "part_tag": "a",
                    "score": 0.9,
                    "verified": True,
                },
                {
                    "id": "u_in",
                    "position": [0.03, 0.0, 0.0],
                    "width": 0.01,
                    "part_tag": "b",
                    "score": 0.8,
                    "verified": False,
                },
            ]
        },
    }
    mask_right = np.zeros((h, w), dtype=bool)
    mask_right[:, 110:] = True
    res_mix = project_affordance_2d(aff_mix, [0, 0, 0], [1, 0, 0, 0], K, c2w, w, h, fg_mask=mask_right)
    assert float(res_mix.heatmap.max()) == pytest.approx(1.0, abs=1e-5)
    assert res_mix.affordance_bbox_xyxy is not None

    # Thin 3px-wide object at compact heatmap_hw=(48, 64) retains non-zero heatmap and Affordance BBox
    w_lg, h_lg = 640, 480
    K_lg = pinhole_intrinsics(w_lg, h_lg, fov_deg=50.0)
    c2w_lg = look_at_cam_to_world(eye=[0.0, 0.0, 0.60], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    aff_thin = {
        "asset_id": "test/cutlery",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.003, -0.06, -0.003], "max": [0.003, 0.06, 0.003]},
        "symmetry": {"z_axis": "none"},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_thin",
                    "position": [0.0, 0.0, 0.0],
                    "closing_axis": [1.0, 0.0, 0.0],
                    "width": 0.006,
                    "part_tag": "handle",
                    "score": 0.9,
                    "verified": True,
                }
            ]
        },
    }
    fg_thin = np.zeros((h_lg, w_lg), dtype=bool)
    fg_thin[200:280, 319:322] = True
    res_thin = project_affordance_2d(
        aff_thin, [0, 0, 0], [1, 0, 0, 0], K_lg, c2w_lg, w_lg, h_lg, fg_mask=fg_thin, heatmap_hw=(48, 64)
    )
    assert float(res_thin.heatmap.max()) == pytest.approx(1.0, abs=1e-5)
    assert res_thin.affordance_bbox_xyxy is not None
