"""Unit tests for twitch/jitter detection and filtering across twitch, motion, export, audit, and qc_publish."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import audit, export, motion, qc_publish, twitch  # noqa: E402


def _smooth_v_turn(frames: int = 60, fps: float = 30.0, speed: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Smooth ramp up at `speed` then clean single V-turn ramp down at `-speed` (normal Limiter motion)."""
    dt = 1.0 / fps
    mid = frames // 2
    q0 = np.concatenate([np.arange(mid) * speed * dt, (mid - np.arange(frames - mid)) * speed * dt])
    state = np.zeros((frames, 7), dtype=np.float64)
    state[:, 0] = q0
    action = np.vstack([state[1:], state[-1:]])  # 1-step ahead exact tracking
    return state, action


def _bang_bang_chatter(
    frames: int = 60, fps: float = 30.0, start: int = 15, length: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """15 Hz +/-0.6 rad/s bang-bang oscillation on joint 1 (Mode A twitching)."""
    state, action = _smooth_v_turn(frames, fps, speed=0.4)
    amp = 0.6 / fps  # 0.02 rad/frame -> 0.6 rad/s every frame
    for i in range(length):
        t = start + i
        if t < frames:
            delta = amp if i % 2 == 0 else 0.0
            state[t, 1] = delta
            action[max(0, t - 1), 1] = delta
    return state, action


def _physics_bounce(frames: int = 60, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Commanded action holds 0 while simulated state bounces at 4.2 rad/s (Mode B1 twitching)."""
    state = np.zeros((frames, 7), dtype=np.float64)
    action = np.zeros((frames, 7), dtype=np.float64)
    # 4.2 rad/s = 0.14 rad/frame (speed > 1.8 rad/s, track_err 0.14 < 0.25 rad)
    state[20, 0] = 0.14
    state[21, 0] = 0.0
    return state, action


def _snag_track_err(frames: int = 60, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Smooth low-speed state that diverges > 0.25 rad from action (Mode B2 arm snag)."""
    state = np.zeros((frames, 7), dtype=np.float64)
    action = np.zeros((frames, 7), dtype=np.float64)
    action[:, 2] = np.linspace(0.0, 0.4, frames)
    return state, action


def test_smooth_v_turn_and_micro_noise_not_flagged():
    state, action = _smooth_v_turn(60, fps=30.0, speed=0.8)
    # Add tiny sub-threshold numerical noise (0.001 rad -> 0.03 rad/s << 0.30 rad/s)
    rng = np.random.default_rng(0)
    state[:, 3] = rng.normal(0.0, 0.0005, size=60)
    action[:, 3] = state[:, 3]

    m = twitch.arm_twitch_metrics(state, action, fps=30.0)
    assert m["rev_burst_max"] <= 1
    assert m["speed_max"] == pytest.approx(0.8, abs=0.05)
    assert m["track_err_max"] < 0.05
    assert twitch.twitch_reasons({"left_arm_joint_state": m}) == []


def test_bang_bang_chatter_flagged():
    state, action = _bang_bang_chatter(60, fps=30.0, start=10, length=16)
    m = twitch.arm_twitch_metrics(state, action, fps=30.0)
    assert m["rev_burst_max"] >= 10
    assert m["rev_speed_max"] == pytest.approx(0.6, abs=0.05)
    assert m["osc_wiggle_max"] > 0.2
    reasons = twitch.twitch_reasons({"left_arm_joint_state": m})
    assert len(reasons) == 1
    assert twitch.reason_kind(reasons[0]) == "rev_burst"


def test_physics_bounce_and_snag_flagged():
    s_b, a_b = _physics_bounce()
    mb = twitch.arm_twitch_metrics(s_b, a_b, fps=30.0)
    assert mb["speed_max"] > 3.5
    rb = twitch.twitch_reasons({"left_arm_joint_state": mb})
    assert any(twitch.reason_kind(r) == "state_speed" for r in rb)

    s_s, a_s = _snag_track_err()
    ms = twitch.arm_twitch_metrics(s_s, a_s, fps=30.0)
    assert ms["rev_burst_max"] == 0
    assert ms["speed_max"] == 0.0
    assert ms["track_err_max"] > 0.35
    rs = twitch.twitch_reasons({"right_arm_joint_state": ms})
    assert len(rs) == 1 and twitch.reason_kind(rs[0]) == "track_err"
    # Disabling track_err check clears it
    assert twitch.twitch_reasons({"right_arm_joint_state": ms}, max_track_err=None) == []


def test_boundary_and_edge_cases():
    # Empty, 1-frame, 2-frame, 3-frame, 0-dim, 1D
    for shape in [(0, 7), (1, 7), (2, 7), (3, 7), (10, 0)]:
        z = np.zeros(shape, dtype=np.float64)
        m = twitch.arm_twitch_metrics(z, z, fps=30.0)
        assert m["rev_burst_max"] == 0 and m["osc_wiggle_max"] == 0.0

    # Short episode (< window of 15 frames) with 4 reversals still caught
    short_s = np.array([0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 0.0], dtype=np.float64)  # 1D, 5 reversals at 0.6 rad/s
    m_short = twitch.arm_twitch_metrics(short_s, short_s, fps=30.0)
    assert m_short["rev_burst_max"] == 5
    assert m_short["osc_wiggle_max"] > 0.1

    # Exactly 3 reversals in window (clean) vs 4 reversals (flagged)
    s3 = np.array([0.0, 0.02, 0.0, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)  # 3 reversals
    m3 = twitch.arm_twitch_metrics(s3, s3, fps=30.0)
    assert m3["rev_burst_max"] == 3
    assert twitch.twitch_reasons({"a": m3}) == []

    # Invalid inputs
    with pytest.raises(ValueError, match="fps"):
        twitch.arm_twitch_metrics(np.zeros((5, 7)), np.zeros((5, 7)), fps=0.0)
    with pytest.raises(ValueError, match="window_s"):
        twitch.arm_twitch_metrics(np.zeros((5, 7)), np.zeros((5, 7)), fps=30.0, window_s=-1.0)
    with pytest.raises(ValueError, match="identical 2D shape"):
        twitch.arm_twitch_metrics(np.zeros((5, 7)), np.zeros((6, 7)), fps=30.0)
    bad = np.zeros((5, 7))
    bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        twitch.arm_twitch_metrics(bad, np.zeros((5, 7)), fps=30.0)


def test_motion_summarize_worst_and_report_include_twitch():
    s_clean, a_clean = _smooth_v_turn()
    s_bad, a_bad = _bang_bang_chatter()
    grip = np.zeros((60, 1))
    arms = ["left_arm_joint_state", "right_arm_joint_state"]

    m_clean = motion.summarize(
        {
            "left_arm_joint_state": s_clean,
            "right_arm_joint_state": s_clean,
            "left_ee_joint_state": grip,
            "right_ee_joint_state": grip,
        },
        {
            "left_arm_joint_state": a_clean,
            "right_arm_joint_state": a_clean,
            "left_ee_joint_state": grip,
            "right_ee_joint_state": grip,
        },
        fps=30.0,
        arms=arms,
    )
    assert m_clean["twitch"] == []
    assert motion.worst(m_clean)["motion_twitch"] == 0

    m_bad = motion.summarize(
        {
            "left_arm_joint_state": s_bad,
            "right_arm_joint_state": s_clean,
            "left_ee_joint_state": grip,
            "right_ee_joint_state": grip,
        },
        {
            "left_arm_joint_state": a_bad,
            "right_arm_joint_state": a_clean,
            "left_ee_joint_state": grip,
            "right_ee_joint_state": grip,
        },
        fps=30.0,
        arms=arms,
    )
    assert len(m_bad["twitch"]) == 1
    assert m_bad["violations"] == []  # real-arm command limits remain separate from twitch
    assert motion.worst(m_bad)["motion_twitch"] == 1

    rep = motion.report(
        [
            (
                0,
                {
                    "left_arm_joint_state": s_clean,
                    "right_arm_joint_state": s_clean,
                    "left_ee_joint_state": grip,
                    "right_ee_joint_state": grip,
                },
                {
                    "left_arm_joint_state": a_clean,
                    "right_arm_joint_state": a_clean,
                    "left_ee_joint_state": grip,
                    "right_ee_joint_state": grip,
                },
            ),
            (
                1,
                {
                    "left_arm_joint_state": s_bad,
                    "right_arm_joint_state": s_clean,
                    "left_ee_joint_state": grip,
                    "right_ee_joint_state": grip,
                },
                {
                    "left_arm_joint_state": a_bad,
                    "right_arm_joint_state": a_clean,
                    "left_ee_joint_state": grip,
                    "right_ee_joint_state": grip,
                },
            ),
        ],
        fps=30.0,
    )
    assert rep["twitch"] == {"episodes": 1, "rev_burst": 1}


def _write_raw_episode(hdf5_path: Path, state: np.ndarray, action: np.ndarray, quality: dict) -> None:
    grip = np.zeros((len(state), 1), dtype=np.float32)
    with h5py.File(hdf5_path, "w") as f:
        f["state/left_arm_joint_states"] = state.astype(np.float32)
        f["action/left_arm_joint_states"] = action.astype(np.float32)
        f["state/right_arm_joint_states"] = np.zeros_like(state, dtype=np.float32)
        f["action/right_arm_joint_states"] = np.zeros_like(action, dtype=np.float32)
        f["state/left_ee_joint_states"] = grip
        f["action/left_ee_joint_states"] = grip
        f["state/right_ee_joint_states"] = grip
        f["action/right_ee_joint_states"] = grip
        f["privileged/steps"] = np.arange(len(state))
    hdf5_path.with_suffix(".json").write_text(json.dumps(quality))


def test_export_select_filter_twitch_sidecar_and_hdf5(tmp_path: Path):
    raw = tmp_path / "seed100"
    raw.mkdir()
    base = dict(
        accepted=True,
        hold_frames=10,
        stable_frames=10,
        frames=60,
        task="t",
        plugin="p",
        acceptance_version="v",
        env_cfg="g1d",
        physics_dt=1 / 300,
        fps=30,
        cameras=[],
        state_keys=["left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"],
        joint_names={"left_arm_joint_state": [], "right_arm_joint_state": []},
    )
    s_clean, a_clean = _smooth_v_turn()
    s_bad, a_bad = _bang_bang_chatter()

    # ep 0: twitching, detected via sidecar motion dict (fast path, no HDF5 needed)
    q0 = dict(
        base,
        planned_poses={"obj": [[0.1, 0.0, 0.7], [1, 0, 0, 0]]},
        motion={"arms": {"left_arm_joint_state": {"rev_burst_max": 9, "speed_max": 0.6, "track_err_max": 0.01}}},
    )
    (raw / "episode_000000.json").write_text(json.dumps(q0))

    # ep 1: twitching, older sidecar without motion dict -> detected from HDF5
    q1 = dict(base, planned_poses={"obj": [[0.2, 0.0, 0.7], [1, 0, 0, 0]]})
    _write_raw_episode(raw / "episode_000001.hdf5", s_bad, a_bad, q1)

    # ep 2: clean
    q2 = dict(base, planned_poses={"obj": [[0.3, 0.0, 0.7], [1, 0, 0, 0]]})
    _write_raw_episode(raw / "episode_000002.hdf5", s_clean, a_clean, q2)

    # Without filter_twitch: takes ep 0 and ep 1
    unfiltered = export.select([tmp_path], 2, filter_twitch=False)
    assert [p.name for p, _ in unfiltered] == ["episode_000000.hdf5", "episode_000001.hdf5"]

    # With filter_twitch: skips ep 0 & ep 1, takes ep 2
    filtered = export.select([tmp_path], 1, filter_twitch=True)
    assert [p.name for p, _ in filtered] == ["episode_000002.hdf5"]

    with pytest.raises(ValueError, match="skipped 2 twitching"):
        export.select([tmp_path], 2, filter_twitch=True)


def test_audit_check_episode_reports_twitch(tmp_path: Path):
    s_bad, a_bad = _bang_bang_chatter()
    quality = dict(
        frames=60,
        fps=30,
        state_keys=["left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"],
        cameras=[],
        joint_names={"left_arm_joint_state": [], "right_arm_joint_state": []},
    )
    p = tmp_path / "episode_000000.hdf5"
    _write_raw_episode(p, s_bad, a_bad, quality)
    plugin = SimpleNamespace(audit=lambda privileged, q: {"ok": 1})
    metrics, violations = audit.check_episode(p, quality, plugin)
    assert violations == []
    assert metrics["motion_twitch"] == 1
    assert metrics["motion_rev_burst_max"] >= 10


def _make_lerobot_dataset(root: Path, episodes: list[tuple[np.ndarray, np.ndarray]], dim: int = 16) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "total_episodes": len(episodes)}))

    ep_col, fr_col, st_col, ac_col = [], [], [], []
    for ep_idx, (s_arm, a_arm) in enumerate(episodes):
        t_len = len(s_arm)
        st_full = np.zeros((t_len, dim), dtype=np.float32)
        ac_full = np.zeros((t_len, dim), dtype=np.float32)
        st_full[:, 0:7] = s_arm
        ac_full[:, 0:7] = a_arm
        for fr in range(t_len):
            ep_col.append(ep_idx)
            fr_col.append(fr)
            st_col.append(st_full[fr].tolist())
            ac_col.append(ac_full[fr].tolist())

    table = pa.table(
        {
            "episode_index": pa.array(ep_col, type=pa.int64()),
            "frame_index": pa.array(fr_col, type=pa.int64()),
            "observation.state": pa.array(st_col, type=pa.list_(pa.float32())),
            "action": pa.array(ac_col, type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_scan_lerobot_16d_and_19d_and_load_excluded(tmp_path: Path):
    s_clean, a_clean = _smooth_v_turn()
    s_chat, a_chat = _bang_bang_chatter()
    s_bounce, a_bounce = _physics_bounce()
    s_snag, a_snag = _snag_track_err()

    for dim in (16, 19):
        ds = tmp_path / f"ds_{dim}d"
        _make_lerobot_dataset(
            ds, [(s_clean, a_clean), (s_chat, a_chat), (s_bounce, a_bounce), (s_snag, a_snag)], dim=dim
        )
        rep = twitch.scan_lerobot(ds)
        assert rep["episodes"] == 4
        assert rep["twitch_episodes"] == 3
        assert rep["clean_episodes"] == 1
        assert rep["excluded_episode_indices"] == [1, 2, 3]
        assert rep["by_reason"] == {"rev_burst": 1, "state_speed": 1, "track_err": 1}

        out_json = tmp_path / f"excl_{dim}.json"
        out_json.write_text(json.dumps(rep))
        assert twitch.load_excluded_indices(out_json) == {1, 2, 3}

    list_json = tmp_path / "list.json"
    list_json.write_text(json.dumps([2, 5]))
    assert twitch.load_excluded_indices(list_json) == {2, 5}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"unrelated": 1}))
    with pytest.raises(ValueError, match="Unrecognized"):
        twitch.load_excluded_indices(bad_json)


def test_qc_publish_excludes_twitch_before_sampling(tmp_path: Path, monkeypatch):
    ds = tmp_path / "g1d_demo_ds"
    s_clean, a_clean = _smooth_v_turn()
    s_chat, a_chat = _bang_bang_chatter()
    # 4 episodes: 0 & 2 twitch, 1 & 3 clean
    _make_lerobot_dataset(ds, [(s_chat, a_chat), (s_clean, a_clean), (s_chat, a_chat), (s_clean, a_clean)])

    qualities = []
    for i in range(4):
        qualities.append(
            {
                "task": "g1d_pick_anything",
                "env_cfg": "g1d",
                "layout_seed": 100,
                "layout_id": i,
                "raw": f"/nonexistent/episode_{i:06d}.hdf5",
                "accepted": True,
                "instruction": f"pick {i}",
                "frames": 60,
            }
        )
    (ds / "collection_quality.json").write_text(json.dumps(qualities))

    # Mock video lookup & encoder so we test filtering + sampling cleanly without ffmpeg
    monkeypatch.setattr(
        qc_publish,
        "load_lerobot_episode_videos",
        lambda _ds, idxs: {i: {"head": (Path("/fake.mp4"), 0.0, 2.0)} for i in idxs},
    )
    encoded_layouts: list[int] = []

    def _fake_encode(job):
        encoded_layouts.append(job[4])
        return job[1]

    monkeypatch.setattr(qc_publish, "encode", _fake_encode)

    out1 = tmp_path / "out_filter"
    n_eps, n_runs = qc_publish.run(ds, out1, round_id="r1", workers=1, limit=2, filter_twitch=True)
    assert (n_eps, n_runs) == (2, 1)
    assert sorted(encoded_layouts) == [1, 3]

    # Also test --exclude JSON path
    excl_file = tmp_path / "excl.json"
    excl_file.write_text(json.dumps({"excluded_episode_indices": [0, 1, 2]}))
    encoded_layouts.clear()
    out2 = tmp_path / "out_excl"
    n_eps2, _ = qc_publish.run(ds, out2, round_id="r1", workers=1, limit=2, exclude=excl_file)
    assert n_eps2 == 1
    assert encoded_layouts == [3]


def test_scan_lerobot_empty_parquet_and_invalid_exclusion_field(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = tmp_path / "empty_ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "data" / "chunk-000").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"fps": 30, "total_episodes": 0}))
    table = pa.table(
        {
            "episode_index": pa.array([], type=pa.int64()),
            "frame_index": pa.array([], type=pa.int64()),
            "observation.state": pa.array([], type=pa.list_(pa.float32())),
            "action": pa.array([], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, ds / "data" / "chunk-000" / "file-000.parquet")

    rep = twitch.scan_lerobot(ds)
    assert rep["episodes"] == 0
    assert rep["twitch_episodes"] == 0
    assert rep["clean_episodes"] == 0
    assert rep["excluded_episode_indices"] == []

    bad = tmp_path / "bad_excl.json"
    bad.write_text(json.dumps({"excluded_episode_indices": None}))
    with pytest.raises(ValueError, match="Unrecognized"):
        twitch.load_excluded_indices(bad)


def test_motion_cli_propagates_custom_twitch_thresholds(tmp_path: Path, monkeypatch):
    # Episode with exactly 3 reversals in 0.5 s: clean at default max_rev_burst=4, flagged at max_rev_burst=3.
    s3 = np.zeros((30, 7), dtype=np.float64)
    s3[:7, 0] = [0.0, 0.02, 0.0, 0.02, 0.0, 0.0, 0.0]
    a3 = np.vstack([s3[1:], s3[-1:]])
    s_clean, a_clean = _smooth_v_turn(30, fps=30.0)

    ds = tmp_path / "ds_burst3"
    _make_lerobot_dataset(ds, [(s_clean, a_clean), (s3, a3)])
    out_rep = tmp_path / "motion_report.json"
    out_tw = tmp_path / "twitch_excl.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion.py",
            "--lerobot",
            str(ds),
            "--max-rev-burst",
            "3",
            "--out",
            str(out_rep),
            "--twitch-out",
            str(out_tw),
        ],
    )
    motion.main()
    rep = json.loads(out_rep.read_text())
    tw = json.loads(out_tw.read_text())
    assert rep["twitch"]["episodes"] == 1
    assert tw["twitch_episodes"] == 1
    assert tw["excluded_episode_indices"] == [1]


def test_audit_main_strict_twitch_and_multi_seed_keys(tmp_path: Path, monkeypatch):
    s_bad, a_bad = _bang_bang_chatter()
    ds = tmp_path / "exported_ds"
    ds.mkdir()
    (ds / "export_verification.json").write_text(json.dumps({"status": "verified", "episodes": 2, "frames": 120}))

    qualities = []
    for idx, (seed, x_pos) in enumerate([(100, 0.1), (101, 0.2)]):
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir()
        h5_path = seed_dir / "episode_000000.hdf5"  # same basename in two seed folders
        q = {
            "raw": str(h5_path),
            "frames": 60,
            "fps": 30,
            "state_keys": [
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            ],
            "cameras": [],
            "joint_names": {"left_arm_joint_state": [], "right_arm_joint_state": []},
            "plugin": "fake_audit_plugin",
            "acceptance_version": "v1",
            "layout_seed": seed,
            "layout_id": idx,
            "planned_poses": {"obj": [[x_pos, 0.0, 0.7], [1.0, 0.0, 0.0, 0.0]]},
        }
        _write_raw_episode(h5_path, s_bad, a_bad, q)
        qualities.append(q)
    (ds / "collection_quality.json").write_text(json.dumps(qualities))

    fake_plugin = SimpleNamespace(audit=lambda privileged, q: {"ok": 1})
    monkeypatch.setattr(audit.importlib, "import_module", lambda name: fake_plugin)
    monkeypatch.setattr(audit, "plot", lambda rows, path: False)

    # Without --strict-twitch: succeeds and records BOTH colliding-basename episodes
    monkeypatch.setattr(sys, "argv", ["audit.py", "--dataset", str(ds)])
    audit.main()
    val_raw = json.loads((ds / "validation_raw.json").read_text())
    final_val = json.loads((ds / "final_validation.json").read_text())
    assert len(val_raw["motion_twitch"]) == 2
    assert final_val["audit"]["motion_twitch"] == 2

    # With --strict-twitch: fails cleanly reporting both episodes
    monkeypatch.setattr(sys, "argv", ["audit.py", "--dataset", str(ds), "--strict-twitch"])
    with pytest.raises(SystemExit, match="2 episodes exhibit arm twitching"):
        audit.main()


def test_qc_publish_missing_h5py_falls_back_to_lerobot_and_checks_mismatch(tmp_path: Path, monkeypatch):
    ds = tmp_path / "g1d_h5py_fallback"
    s_clean, a_clean = _smooth_v_turn()
    _make_lerobot_dataset(ds, [(s_clean, a_clean), (s_clean, a_clean)])

    # Create a readable dummy raw HDF5 file on disk so _is_readable_file() is True
    raw_h5 = tmp_path / "episode_000000.hdf5"
    raw_h5.write_bytes(b"not-used-when-h5py-missing")
    qualities = [
        {
            "task": "g1d_pick_anything",
            "env_cfg": "g1d",
            "layout_seed": 100,
            "layout_id": i,
            "raw": str(raw_h5),
            "accepted": True,
            "instruction": f"pick {i}",
            "frames": 60,
        }
        for i in range(2)
    ]
    (ds / "collection_quality.json").write_text(json.dumps(qualities))

    monkeypatch.setattr(qc_publish, "_has_h5py", lambda: False)
    monkeypatch.setattr(
        qc_publish,
        "load_lerobot_episode_videos",
        lambda _ds, idxs: {i: {"head": (Path("/fake.mp4"), 0.0, 2.0)} for i in idxs},
    )
    job_kinds: list[str] = []

    def _fake_encode(job):
        job_kinds.append(job[0])
        return job[1]

    monkeypatch.setattr(qc_publish, "encode", _fake_encode)
    n_eps, _ = qc_publish.run(ds, tmp_path / "out_fallback", round_id="r1", workers=1, limit=2)
    assert n_eps == 2
    assert job_kinds == ["lerobot", "lerobot"]

    # Length mismatch between collection_quality.json (1) and parquet (2) raises ValueError
    (ds / "collection_quality.json").write_text(json.dumps(qualities[:1]))
    with pytest.raises(ValueError, match="Episode count mismatch"):
        qc_publish.run(ds, tmp_path / "out_mismatch", round_id="r1", workers=1, filter_twitch=True)
