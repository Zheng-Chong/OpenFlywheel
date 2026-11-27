"""CPU round-trip tests for opt-in `--record-affordance-2d` in scripts/datagen/collect.py & export.py."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import collect, export  # noqa: E402
from scripts.objects.affordance_2d import main as aff2d_main  # noqa: E402
from scripts.objects.affordance_vlm import look_at_cam_to_world, pinhole_intrinsics  # noqa: E402


class _Tensor(np.ndarray):
    def cpu(self):
        return np.asarray(self)


def _real_encode_jpeg(rgb):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _sample_affordance(asset_id="Rigid/mug/00001") -> dict:
    return {
        "schema_version": 1,
        "asset_id": asset_id,
        "mesh_hash": "sha256:0123456789",
        "scale_ref": 1.0,
        "bounds": {"min": [-0.04, -0.04, -0.05], "max": [0.06, 0.04, 0.05], "extents": [0.10, 0.08, 0.10]},
        "symmetry": {"z_axis": "none", "fold": 1},
        "grasps": {
            "dex1_1": [
                {
                    "id": "g_handle",
                    "position": [0.05, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.018,
                    "part_tag": "handle",
                    "score": 0.85,
                    "sim_score": 0.94,
                    "verified": True,
                },
                {
                    "id": "g_body",
                    "position": [0.0, 0.03, 0.01],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.065,
                    "part_tag": "body",
                    "score": 0.72,
                    "sim_score": 0.82,
                    "verified": True,
                },
            ],
            "sharpa_wave": [
                {
                    "id": "sw_0",
                    "position": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "width": 0.070,
                    "part_tag": "body",
                    "grasp_type": "power_cylindrical",
                    "score": 0.88,
                    "sim_score": 0.91,
                    "verified": True,
                }
            ],
        },
    }


def _articulation(side):
    jac = np.zeros((1, 1, 6, 7))
    jac[0, 0, :, :6] = np.eye(6)
    return SimpleNamespace(
        body_names=["base", f"{side}_hand"],
        root_physx_view=SimpleNamespace(get_jacobians=lambda: jac.view(_Tensor)),
        data=SimpleNamespace(soft_joint_pos_limits=np.tile([-2.0, 2.0], (1, 7, 1)), joint_pos=np.zeros((1, 7))),
    )


def _dex1_arm(side):
    return SimpleNamespace(
        type="target",
        robot_type="arm",
        asset_name="g1d",
        robot_name="g1d",
        arm_name=f"{side}_arm",
        gripper_name=f"{side}_ee",
        ee_type="gripper",
        gripper_joints_name=[f"{side}_finger_1", f"{side}_finger_2"],
        arm_joints_name=[f"{side}_arm_j{i}" for i in range(7)],
        ee_link_name=f"{side}_hand",
        arm_joint_indices=range(7),
    )


class _FakeEnv:
    def __init__(self, layout_file, *, layout_id=0, aff_path=None, target_label="target", obj_x0=0.45):
        robots = [_dex1_arm("right"), _dex1_arm("left")]
        self.num_envs, self.task_name, self.config_name, self.dt, self.eval_seed = (
            1,
            "g1d_pick_anything",
            "g1d",
            1 / 240,
            101,
        )
        self.current_env_seed_map = {0: layout_id}
        self.success, self.end_flag, self.task_success = [True], [False], [True]
        self.end_on_success, self.step_lim, self.t = True, 100, 0
        self.target_label = target_label
        self._obj_x0 = obj_x0
        self.robot_manager = SimpleNamespace(
            robot_list=robots,
            robot_key=[_articulation(r.arm_name.split("_")[0]) for r in robots],
            process_name=lambda n: n + "_joint_state",
        )
        self.obs_manager = SimpleNamespace(robot_cfg={"body_states": False})

        entry = {
            "label": target_label,
            "category": "mug",
            "category_idx": 1,
            "default_pos": [obj_x0, 0.0, 0.85],
            "default_ori": [1.0, 0.0, 0.0, 0.0],
            "scale": 1.0,
        }
        if aff_path is not None:
            entry["affordance_path"] = str(aff_path)
        scene_info = {"Rigid": {"mug": [entry]}}
        self.seed_manager = SimpleNamespace(
            seed_info={layout_id: {"scene_layout": str(layout_file)}},
            get_seed_scene_info=lambda lid: scene_info,
        )
        self.scene_manager = SimpleNamespace(
            env_origins=np.zeros((1, 3)),
            layout_manager=SimpleNamespace(get_instance_pose=self._get_instance_pose),
            config={"env0": scene_info},
        )
        # Head camera at (0.0, 0.0, 1.35) looking at (0.45, 0.0, 0.85) in USD convention
        c2w_cv = look_at_cam_to_world(eye=[0.0, 0.0, 1.35], target=[0.45, 0.0, 0.85])
        c2w_usd = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])
        K = pinhole_intrinsics(160, 120, fov_deg=55.0)
        self.camera_manager = SimpleNamespace(
            camera_names=[["cam_head", "cam_left_wrist", "cam_right_wrist"]],
            get_camera_intrinsics=lambda cid, eid=0: K,
            get_camera_extrinsics=lambda cid, eid=0: c2w_usd,
        )

    def _get_instance_pose(self, env_idx, label):
        # Object moves slightly along +Y and +Z each step to verify per-frame tracking
        pos = np.array([self._obj_x0, 0.015 * self.t, 0.85 + 0.01 * self.t], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return pos, quat

    def get_obs(self):
        img = np.full((120, 160, 3), 30 + (self.t * 10) % 150, dtype=np.uint8)
        return {
            "instruction": "pick up the mug",
            "state": {
                "left_arm_joint_state": np.zeros(7, dtype=np.float32),
                "left_ee_joint_state": np.array([0.5], dtype=np.float32),
                "right_arm_joint_state": np.zeros(7, dtype=np.float32),
                "right_ee_joint_state": np.array([0.5], dtype=np.float32),
            },
            "vision": {
                "cam_head": {"color": img},
                "cam_left_wrist": {"color": img},
                "cam_right_wrist": {"color": img},
            },
        }

    def take_action(self, action):
        self.t += 1


def _fake_plugin(n_steps=6):
    class Episode:
        reject = None

        def __init__(self, env):
            self.env = env
            self.target_label = env.target_label
            self.selected_grasp_world = np.array([env._obj_x0 + 0.05, 0.0, 0.85], dtype=np.float64)

        def actions(self):
            for _ in range(n_steps):
                yield {
                    "left_arm_joint_state": np.zeros(7, dtype=np.float32),
                    "left_ee_joint_state": np.array([0.5], dtype=np.float32),
                    "right_arm_joint_state": np.zeros(7, dtype=np.float32),
                    "right_ee_joint_state": np.array([0.5], dtype=np.float32),
                }

        def step(self):
            return True

        def summary(self):
            return {}

        def privileged(self):
            return {}

    return SimpleNamespace(
        __name__="scripts.datagen.plugins.g1d_pick_anything_affordance",
        TASK="g1d_pick_anything",
        ACCEPTANCE="v1",
        HOLD=2,
        TAIL=0,
        STEP_LIM=100,
        RECORD_TO_END=True,
        Episode=Episode,
    )


def _patch_xpolicylab():
    pkg = ModuleType("XPolicyLab")
    utils_mod = ModuleType("XPolicyLab.utils")
    proc_mod = ModuleType("XPolicyLab.utils.process_data")
    proc_mod.encode_image_bit = _real_encode_jpeg
    return patch.dict(
        sys.modules,
        {"XPolicyLab": pkg, "XPolicyLab.utils": utils_mod, "XPolicyLab.utils.process_data": proc_mod},
    )


def test_record_affordance_2d_roundtrip_validation_and_vlm_export(tmp_path: Path):
    layout_file = tmp_path / "layout_0.json"
    layout_file.write_text("{}", encoding="utf-8")
    aff_path = tmp_path / "affordance.json"
    aff_path.write_text(json.dumps(_sample_affordance("Rigid/mug/00001")), encoding="utf-8")

    out_dir = tmp_path / "seed101"
    out_dir.mkdir()
    env = _FakeEnv(layout_file, layout_id=0, aff_path=aff_path)
    cfg = {
        "env_cfg": "g1d_pick",
        "fps": 30,
        "trajectory": True,
        "early_abort": False,
        "record_affordance_2d": True,
        "affordance_2d_cameras": ["cam_head"],
        "affordance_2d_heatmap_hw": [48, 64],
        "affordance_2d_top_k": 8,
    }
    with _patch_xpolicylab():
        collect.record_episode(env, _fake_plugin(n_steps=6), out_dir, cfg)

    h5_path = out_dir / "episode_000000.hdf5"
    sidecar_path = out_dir / "episode_000000.json"
    assert h5_path.is_file() and sidecar_path.is_file()

    quality = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert quality["accepted"] is True
    assert quality["frames"] == 6
    aff_meta = quality["affordance_2d"]
    assert aff_meta["cameras"] == ["cam_head"]
    assert aff_meta["heatmap_hw"] == [48, 64]
    assert aff_meta["top_k"] == 8
    assert aff_meta["asset_id"] == "Rigid/mug/00001"
    assert aff_meta["has_selected_grasp"] is True

    with h5py.File(h5_path, "r") as f:
        cg = f["affordance_2d/cam_head"]
        obj_bb = cg["object_bbox_xyxy_norm"][()]
        aff_bb = cg["affordance_bbox_xyxy_norm"][()]
        kps = cg["keypoints_xy_score"][()]
        sel = cg["selected_grasp_xy_norm"][()]
        hms = cg["heatmap_uint8"][()]
        assert obj_bb.shape == (6, 4) and obj_bb.dtype == np.float32
        assert aff_bb.shape == (6, 4) and aff_bb.dtype == np.float32
        assert kps.shape == (6, 8, 3) and kps.dtype == np.float32
        assert sel.shape == (6, 2) and sel.dtype == np.float32
        assert hms.shape == (6, 48, 64) and hms.dtype == np.uint8
        assert np.isfinite(obj_bb).all()
        assert np.isfinite(aff_bb).all()
        assert np.isfinite(sel).all()
        # Object moved between frame 0 and frame 5 -> BBox must track the motion!
        assert not np.allclose(obj_bb[0], obj_bb[-1])
        assert int(hms.max()) == 255

    # export.check_raw_episodes passes on valid HDF5
    chosen = [(h5_path, quality)]
    export.check_raw_episodes(chosen)

    # export.card documents affordance_2d
    card_md = export.card("test_ds", {"revision": {}}, chosen, {"episodes": 1, "frames": 6, "fps": 30})
    assert "2D affordance annotations" in card_md and "48x64" in card_md

    # export.export_affordance_2d_jsonl & CLI export-vlm --hdf5 both succeed
    vlm_out = tmp_path / "vlm_out"
    summary = export.export_affordance_2d_jsonl(chosen, vlm_out, frame_stride=2, save_overlays=True)
    assert summary["records"] == 3
    recs = [json.loads(line) for line in (vlm_out / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 3
    assert recs[0]["instruction"] == "pick up the mug"
    assert (vlm_out / recs[0]["heatmap_path"]).is_file()
    assert (vlm_out / recs[0]["overlay_path"]).is_file()

    cli_out = tmp_path / "vlm_cli"
    rc = aff2d_main(["export-vlm", "--hdf5", str(h5_path), "-o", str(cli_out), "--frame-stride", "3"])
    assert rc == 0
    assert len((cli_out / "annotations.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_select_allows_different_objects_but_rejects_mismatched_schema_or_corrupt_hdf5(tmp_path: Path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")
    aff1 = tmp_path / "aff1.json"
    aff2 = tmp_path / "aff2.json"
    aff1.write_text(json.dumps(_sample_affordance("Rigid/mug/00001")), encoding="utf-8")
    aff2.write_text(json.dumps(_sample_affordance("Rigid/bottle/00002")), encoding="utf-8")

    raw_dir = tmp_path / "raw"
    seed_dir = raw_dir / "seed101"
    seed_dir.mkdir(parents=True)

    base_cfg = {
        "env_cfg": "g1d_pick",
        "fps": 30,
        "trajectory": True,
        "early_abort": False,
        "record_affordance_2d": True,
        "affordance_2d_cameras": ["cam_head"],
        "affordance_2d_heatmap_hw": [48, 64],
        "affordance_2d_top_k": 8,
    }
    with _patch_xpolicylab():
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=0, aff_path=aff1, target_label="mug_0", obj_x0=0.42),
            _fake_plugin(4),
            seed_dir,
            base_cfg,
        )
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=1, aff_path=aff2, target_label="bottle_0", obj_x0=0.48),
            _fake_plugin(4),
            seed_dir,
            base_cfg,
        )

    # Two episodes with different asset_id/target_label but same affordance_2d schema select cleanly
    chosen = export.select([raw_dir], 2)
    assert len(chosen) == 2
    export.check_raw_episodes(chosen)

    # Tampering normalized bbox out of [0, 1] is caught by check_raw_episodes
    h5_0 = chosen[0][0]
    with h5py.File(h5_0, "r+") as f:
        f["affordance_2d/cam_head/object_bbox_xyxy_norm"][0, 2] = 1.5
    with pytest.raises(ValueError, match="normalized bbox outside"):
        export.check_raw_episodes(chosen)

    # Mixing an episode with affordance_2d and one without is caught by select()
    with _patch_xpolicylab():
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=2, aff_path=aff1, obj_x0=0.52),
            _fake_plugin(4),
            seed_dir,
            {**base_cfg, "record_affordance_2d": False},
        )
    with pytest.raises(ValueError, match="affordance_2d"):
        export.select([raw_dir], 3)


def test_default_disabled_and_missing_affordance_graceful_degradation(tmp_path: Path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")

    # 1. Default (record_affordance_2d=False) leaves HDF5 and quality.json untouched
    out1 = tmp_path / "off"
    out1.mkdir()
    with _patch_xpolicylab():
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=0, aff_path=None),
            _fake_plugin(4),
            out1,
            {"env_cfg": "g1d_pick", "fps": 30, "trajectory": True, "early_abort": False},
        )
    q1 = json.loads((out1 / "episode_000000.json").read_text(encoding="utf-8"))
    assert "affordance_2d" not in q1
    with h5py.File(out1 / "episode_000000.hdf5", "r") as f:
        assert "affordance_2d" not in f

    # 2. Enabled on an object without any affordance.json degrades gracefully without failing
    out2 = tmp_path / "missing"
    out2.mkdir()
    with _patch_xpolicylab():
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=0, aff_path=None),
            _fake_plugin(4),
            out2,
            {
                "env_cfg": "g1d_pick",
                "fps": 30,
                "trajectory": True,
                "early_abort": False,
                "record_affordance_2d": True,
            },
        )
    q2 = json.loads((out2 / "episode_000000.json").read_text(encoding="utf-8"))
    assert q2["accepted"] is True
    assert "affordance_2d" not in q2


def test_multiseed_vlm_export_no_filename_collision_and_partial_nan_validation(tmp_path: Path):
    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")
    aff_path = tmp_path / "aff.json"
    aff_path.write_text(json.dumps(_sample_affordance("Rigid/mug/00001")), encoding="utf-8")

    seed1 = tmp_path / "seed101"
    seed2 = tmp_path / "seed102"
    seed1.mkdir()
    seed2.mkdir()
    cfg = {
        "env_cfg": "g1d_pick",
        "fps": 30,
        "trajectory": True,
        "early_abort": False,
        "record_affordance_2d": True,
        "affordance_2d_cameras": ["cam_head"],
        "affordance_2d_heatmap_hw": [48, 64],
        "affordance_2d_top_k": 4,
    }
    with _patch_xpolicylab():
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=0, aff_path=aff_path, obj_x0=0.42), _fake_plugin(3), seed1, cfg
        )
        collect.record_episode(
            _FakeEnv(layout_file, layout_id=0, aff_path=aff_path, obj_x0=0.52), _fake_plugin(3), seed2, cfg
        )

    h5_1 = seed1 / "episode_000000.hdf5"
    h5_2 = seed2 / "episode_000000.hdf5"
    q1 = json.loads((seed1 / "episode_000000.json").read_text(encoding="utf-8"))
    q2 = json.loads((seed2 / "episode_000000.json").read_text(encoding="utf-8"))

    # Both episodes have stem 'episode_000000': exported heatmap & overlay paths must not collide
    vlm_out = tmp_path / "vlm_multi"
    summary = export.export_affordance_2d_jsonl([(h5_1, q1), (h5_2, q2)], vlm_out, save_overlays=True)
    assert summary["records"] == 6
    recs = [json.loads(line) for line in (vlm_out / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    hm_paths = [r["heatmap_path"] for r in recs]
    ov_paths = [r["overlay_path"] for r in recs]
    assert len(set(hm_paths)) == 6
    assert len(set(ov_paths)) == 6
    for p in hm_paths + ov_paths:
        assert (vlm_out / p).is_file()

    # Partial-NaN bbox row is rejected by check_raw_episodes
    with h5py.File(h5_1, "r+") as f:
        f["affordance_2d/cam_head/object_bbox_xyxy_norm"][0, 1] = np.nan
    with pytest.raises(ValueError, match="partially NaN"):
        export.check_raw_episodes([(h5_1, q1)])

    # Negative keypoint score is rejected by check_raw_episodes
    with h5py.File(h5_2, "r+") as f:
        f["affordance_2d/cam_head/keypoints_xy_score"][0, 0, 2] = -0.5
    with pytest.raises(ValueError, match="negative keypoint score"):
        export.check_raw_episodes([(h5_2, q2)])


def test_recorder_handles_missing_mid_episode_pose_identity_K_and_mapping(tmp_path: Path):
    from collections.abc import Mapping

    class _ReadOnlyMap(Mapping):
        def __init__(self, d):
            self._d = d

        def __getitem__(self, k):
            return self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    layout_file = tmp_path / "layout.json"
    layout_file.write_text("{}", encoding="utf-8")
    env = _FakeEnv(layout_file, layout_id=0, aff_path=None)
    for r in env.robot_manager.robot_list:
        r.asset_name = r.robot_name = "g1d_wuji"
    # Inject OmegaConf-like Mapping for inline affordance and simulate missing pose on step 2
    aff = _sample_affordance("Rigid/mug/00001")
    aff["grasps"]["wuji_hand2_2"] = [
        {
            "id": "wj_0",
            "position": [0.05, 0.0, 0.0],
            "width": 0.02,
            "part_tag": "handle",
            "score": 0.9,
            "verified": True,
        }
    ]
    env.seed_manager.get_seed_scene_info = lambda lid: {
        "Rigid": {"mug": [_ReadOnlyMap({"label": "target", "affordance": _ReadOnlyMap(aff), "scale": 1.0})]}
    }
    orig_pose = env._get_instance_pose

    def _flaky_pose(env_idx, label):
        if env.t == 2:
            raise RuntimeError("transient missing pose")
        return orig_pose(env_idx, label)

    env.scene_manager.layout_manager.get_instance_pose = _flaky_pose

    out_dir = tmp_path / "flaky"
    out_dir.mkdir()
    cfg = {
        "env_cfg": "g1d_pick",
        "fps": 30,
        "trajectory": True,
        "early_abort": False,
        "robot_model": "g1d_wuji",
        "record_affordance_2d": True,
        "affordance_2d_cameras": ["cam_head"],
        "affordance_2d_heatmap_hw": [48, 64],
        "affordance_2d_top_k": 4,
    }
    with _patch_xpolicylab():
        collect.record_episode(env, _fake_plugin(4), out_dir, cfg)

    q = json.loads((out_dir / "episode_000000.json").read_text(encoding="utf-8"))
    assert q["affordance_2d"]["gripper_types"] == ["wuji_hand2_2"]
    export.check_raw_episodes([(out_dir / "episode_000000.hdf5", q)])
    with h5py.File(out_dir / "episode_000000.hdf5", "r") as f:
        bb = f["affordance_2d/cam_head/object_bbox_xyxy_norm"][()]
        assert bb.shape == (4, 4)
        assert np.isfinite(bb[0]).all()
        assert np.isnan(bb[2]).all()  # transient missing pose -> clean all-NaN placeholder frame
        assert np.isfinite(bb[3]).all()

    # Identity K = np.eye(3) (uncalibrated camera template) must be ignored cleanly
    env_id = _FakeEnv(layout_file, layout_id=0, aff_path=None)
    env_id.seed_manager.get_seed_scene_info = lambda lid: {
        "Rigid": {"mug": [{"label": "target", "affordance": aff, "scale": 1.0}]}
    }
    env_id.camera_manager.get_camera_intrinsics = lambda cid, eid=0: np.eye(3, dtype=np.float64)
    out_id = tmp_path / "id_k"
    out_id.mkdir()
    with _patch_xpolicylab():
        collect.record_episode(env_id, _fake_plugin(3), out_id, cfg)
    q_id = json.loads((out_id / "episode_000000.json").read_text(encoding="utf-8"))
    assert "affordance_2d" not in q_id
