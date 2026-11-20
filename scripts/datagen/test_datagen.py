"""CPU checks for the data pipeline and its contract (no Isaac): python -m pytest scripts/datagen"""

import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules.setdefault("env.seeding", ModuleType("env.seeding"))  # torch + warp; only DescManager seeds with it
sys.modules["env.seeding"].seed_everywhere = getattr(sys.modules["env.seeding"], "seed_everywhere", lambda *a: None)

from env.observation_manager.obs_manager import ObsManager  # noqa: E402
from env.seed_manager import seed_manager  # noqa: E402
from scripts.datagen import collect, export, run_batch  # noqa: E402

try:
    from XPolicyLab.utils.data_loader import load as load_trajectory
    from XPolicyLab.utils.process_data import encode_image_bit  # noqa: F401
except ImportError:
    load_trajectory = None
needs_xpolicylab = unittest.skipIf(load_trajectory is None, "XPolicyLab (and h5py, cv2) not importable")
KEYS = ["left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"]
CLOSED, OPEN = 0.0245, -0.02  # Dex1 finger joint travel, closed .. open


def dex1(side):
    return SimpleNamespace(
        type="target",
        robot_type="arm",
        arm_name=f"{side}_arm",
        gripper_name=f"{side}_ee",
        ee_type="gripper",
        gripper_scale=[OPEN, CLOSED],
        gripper_move={"sign": -1.0, "base": f"{side}_1", "mimic": [f"{side}_2", 1.0, 0.0]},
        gripper_joints_name=[f"{side}_1", f"{side}_2"],
        arm_joints_name=[f"{side}_j{i}" for i in range(7)],
        ee_link_name=f"{side}_hand",
        arm_joint_indices=range(7),
    )


class Tensor(np.ndarray):
    def cpu(self):
        return np.asarray(self)


def articulation(side):
    """A fixed-base arm whose hand Jacobian is [I6 | 0] (manipulability 1), joints within +-2 rad."""
    jac = np.zeros((1, 1, 6, 7))
    jac[0, 0, :, :6] = np.eye(6)
    return SimpleNamespace(
        body_names=["base", f"{side}_hand"],
        root_physx_view=SimpleNamespace(get_jacobians=lambda: jac.view(Tensor)),
        data=SimpleNamespace(soft_joint_pos_limits=np.tile([-2.0, 2.0], (1, 7, 1)), joint_pos=np.zeros((1, 7))),
    )


class FakeEnv:
    """Just enough of EvalEnv for collect.record_episode: obs(t) holds t in every state entry."""

    def __init__(self, layout_file):
        robots = [dex1("right"), dex1("left")]
        self.num_envs, self.task_name, self.config_name, self.dt, self.eval_seed = 1, "t", "cfg", 1 / 600, 9
        self.current_env_seed_map = {0: 3}
        self.success, self.end_flag, self.task_success = [True], [False], [True]
        self.end_on_success, self.step_lim, self.t, self.actions = True, 600, 0, []
        self.robot_manager = SimpleNamespace(
            robot_list=robots,
            robot_key=[articulation(r.arm_name.split("_")[0]) for r in robots],
            process_name=lambda n: n + "_joint_state",
        )
        self.scene_manager = SimpleNamespace(env_origins=np.zeros((1, 3)))
        layout = {"Rigid": {"cup": [{"label": "cup", "default_pos": [0.1, -0.3, 0.7], "default_ori": [1, 0, 0, 0]}]}}
        self.seed_manager = SimpleNamespace(
            seed_info={3: {"scene_layout": str(layout_file)}}, get_seed_scene_info=lambda _: layout
        )

    def get_obs(self):
        state = {k: np.full(7 if "arm" in k else 1, float(self.t)) for k in KEYS}
        vision = {c: {"color": np.full((8, 8, 3), 10 * self.t, dtype=np.uint8)} for c in collect.CAMERAS}
        force = {"left_ee": np.array([1.0, 2.0]) * self.t, "right_ee": np.zeros(2)}
        return dict(state=state, vision=vision, instruction="pour it", gripper_force=force)

    def take_action(self, action):
        self.actions.append(action)
        self.t += 1


class Plugin(ModuleType):
    """Accepts once three steps are in: the expert's two actions, then the held tail."""

    TASK, ACCEPTANCE, HOLD, STEP_LIM, TAIL = "t", "test_v1", 2, 50, 5

    def __init__(self):
        super().__init__("test_plugin")
        outer = self

        class Episode:
            def __init__(self, env):
                self.reject, self.steps = None, 0

            def actions(self):
                for i in (1, 2):
                    yield {k: np.full(7 if "arm" in k else 1, i * 0.1) for k in KEYS}

            def step(self):
                self.steps += 1
                return self.steps >= 3

            def privileged(self):
                return dict(steps=np.arange(self.steps))

            def summary(self):
                return dict(hold=outer.HOLD)

        self.Episode = Episode


class RecorderTests(unittest.TestCase):
    @needs_xpolicylab
    def test_record_episode_writes_aligned_standard_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "layout.json").write_text("{}")
            env = FakeEnv(root / "layout.json")
            collect.record_episode(env, Plugin(), root, dict(env_cfg="cfg_600hz", fps=30, revision=None))
            quality = json.loads((root / "episode_000003.json").read_text())
            data = load_trajectory(str(root / "episode_000003.hdf5"), "RoboDojo", "v1.0")
        self.assertFalse(env.end_on_success)  # the recorder, not the task's check, ends the episode
        self.assertEqual((env.success, env.end_flag), ([True], [True]))
        self.assertTrue(quality["accepted"])
        self.assertEqual((quality["frames"], quality["stable_frames"]), (4, 2))
        self.assertEqual(quality["state_keys"], KEYS)  # left arm first, gripper after its arm
        self.assertEqual(quality["joint_names"]["left_arm_joint_state"][0], "left_j0")
        self.assertEqual(quality["planned_poses"], {"cup": [[0.1, -0.3, 0.7], [1, 0, 0, 0]]})
        self.assertEqual(data["instruction"], "pour it")
        self.assertEqual(data["additional_info"]["env_cfg"], "cfg_600hz")
        # frame t: observation before action t; the two expert actions, then the last one held
        np.testing.assert_array_equal(data["state"]["left_arm_joint_states"][:, 0], [0, 1, 2, 3])
        np.testing.assert_allclose(data["action"]["right_ee_joint_states"][:, 0], [0.1, 0.2, 0.2, 0.2])
        self.assertEqual(data["vision"]["cam_head"]["colors"].shape, (4, 8, 8, 3))
        self.assertEqual(int(data["vision"]["cam_left_wrist"]["colors"][2].mean().round()), 20)
        np.testing.assert_array_equal(data["privileged"]["steps"], np.arange(4))
        np.testing.assert_array_equal(data["privileged"]["motion_manipulability_left_arm_joint_state"], np.ones(4))
        np.testing.assert_array_equal(data["privileged"]["motion_grip_force_left_ee"][:, 1], [0, 2, 4, 6])

    def test_record_episode_writes_motion_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "layout.json").write_text("{}")
            env = FakeEnv(root / "layout.json")
            collect.record_episode(env, Plugin(), root, dict(env_cfg="cfg", fps=30, revision=None, trajectory=False))
            quality = json.loads((root / "episode_000003.json").read_text())
        self.assertEqual(quality["joint_limits"]["left_arm_joint_state"], [[-2.0, 2.0]] * 7)
        moved = quality["motion"]
        # arm commands 0 (the first state), 0.1, 0.2, 0.2, 0.2 rad: 3 rad/s, over the real 0.8
        self.assertAlmostEqual(moved["arms"]["right_arm_joint_state"]["cmd_speed_max"], 3.0, places=4)
        self.assertAlmostEqual(moved["arms"]["right_arm_joint_state"]["cmd_limit_margin_min"], 1.8, places=6)
        self.assertEqual(
            [v.split(":")[0] for v in moved["violations"]], ["left_arm_joint_state", "right_arm_joint_state"]
        )
        self.assertAlmostEqual(moved["grippers"]["left_ee_joint_state"]["cmd_rate_max"], 0.1, places=6)
        self.assertEqual(moved["manipulability_min"], {"right_arm_joint_state": 1.0, "left_arm_joint_state": 1.0})
        self.assertEqual(moved["grip_force_max"], {"left_ee": 6.0, "right_ee": 0.0})

    @needs_xpolicylab
    def test_record_to_end_runs_the_whole_script_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "layout.json").write_text("{}")
            env, plugin = FakeEnv(root / "layout.json"), Plugin()
            plugin.RECORD_TO_END = True
            collect.record_episode(env, plugin, root, dict(env_cfg="cfg", fps=30, revision=None, trajectory=False))
            quality = json.loads((root / "episode_000003.json").read_text())
        # two expert actions and the five held: settled from the third step on, so the last five frames hold
        self.assertEqual((quality["accepted"], quality["frames"], quality["stable_frames"]), (True, 7, 5))
        self.assertEqual(len(env.actions), 7)

    def test_beans_plus_v2_audit_needs_open_grippers_at_the_end(self):
        from scripts.datagen.plugins import g1d_pour_beans_plus_v2 as v2

        quality = dict(frames=40)
        grip = np.full((40, 2), 0.9, np.float32)
        with patch.object(v2.base, "audit", return_value=dict(beans=10)):
            self.assertEqual(v2.audit(dict(grip_command=grip), quality), dict(beans=10, tail_frames=v2.TAIL))
            grip[-5, 1] = 0.0  # the right gripper closed inside the last HOLD frames
            with self.assertRaises(AssertionError):
                v2.audit(dict(grip_command=grip), quality)
        self.assertEqual(v2.gripper_keys(dict.fromkeys(KEYS)), ["left_ee_joint_state", "right_ee_joint_state"])

    def test_export_shards_cover_every_episode_once_in_order(self):
        from scripts.datagen import export

        for count, shards in ((5000, 32), (10, 3), (7, 7), (3, 8)):
            slices = export.shard_slices(count, shards)
            self.assertLessEqual(len(slices), shards)
            self.assertEqual([i for a, b in slices for i in range(a, b)], list(range(count)))

    def test_export_takes_accepted_episodes_with_distinct_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "seed100"
            raw.mkdir()
            base = dict(accepted=True, hold_frames=30, stable_frames=30, frames=90, task="t", plugin="p")
            base |= dict(acceptance_version="v", env_cfg="c", physics_dt=0.1, fps=30, cameras=[], state_keys=[])
            layouts = [[0.1, -0.3], [0.1, -0.3], [0.12, -0.3]]
            for i, (xy, accepted) in enumerate(zip(layouts, (True, True, False))):
                quality = dict(base, accepted=accepted, planned_poses={"cup": [[*xy, 0.7], [1, 0, 0, 0]]})
                (raw / f"episode_{i:06d}.json").write_text(json.dumps(quality))
            self.assertEqual([p.name for p, _ in export.select([raw.parent], 1)], ["episode_000000.hdf5"])
            with self.assertRaisesRegex(ValueError, "Only 1 accepted"):
                export.select([raw.parent], 2)
            other = Path(directory) / "b" / "seed200"
            other.mkdir(parents=True)
            quality = dict(base, env_cfg="a", planned_poses={"cup": [[0.1, -0.3, 0.7], [1, 0, 0, 0]]})
            # Same poses in another scene are distinct; duplicates within one scene remain rejected.
            (other / "episode_000000.json").write_text(json.dumps(quality))
            chosen = export.select([raw.parent, other.parent], 2)  # second run's scene first: converter order
            self.assertEqual([(p.parent.name, q["env_cfg"]) for p, q in chosen], [("seed200", "a"), ("seed100", "c")])
            (raw / "episode_000000.json").write_text(json.dumps(dict(base, stable_frames=12, planned_poses={})))
            with self.assertRaisesRegex(ValueError, "Inconsistent"):
                export.select([raw.parent], 1)

    def test_cap_runs_keeps_each_folder_at_its_target(self):
        from scripts.datagen import cap_runs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "obs"
            units = [
                dict(scene="s", unit=1, parent=10, folder_target=2),
                dict(scene="s", unit=2, parent=10, folder_target=2),
                dict(scene="s", unit=3, parent=11, folder_target=1),
            ]
            # unit 1 on two hosts (the second repeats a layout), unit 2 never needed, unit 3 has one accepted
            episodes = {
                ("h1", 1): [(0.10, True), (0.11, False)],
                ("h2", 1): [(0.10, True), (0.12, True), (0.13, True)],
                ("h1", 2): [(0.20, True)],
                ("h1", 3): [(0.30, True)],
            }
            for (host, unit), eps in episodes.items():
                batch = root / host / "batch" / f"s_{unit}"
                (batch / "raw" / "seed5").mkdir(parents=True)
                (batch / "collection_config.json").write_text(json.dumps(dict(revision=dict(commit="abc"))))
                for i, (x, accepted) in enumerate(eps):
                    poses = {"cup": [[x, -0.3, 0.7], [1, 0, 0, 0]]}
                    quality = dict(accepted=accepted, env_cfg="c", planned_poses=poses)
                    (batch / "raw" / "seed5" / f"episode_{i:06d}.json").write_text(json.dumps(quality))
                    (batch / "raw" / "seed5" / f"episode_{i:06d}.hdf5").write_bytes(b"")
            out = Path(directory) / "runs"
            folders, revisions = cap_runs.stage(units, root, out)
            self.assertEqual(folders, {("s", 10): 2, ("s", 11): 1})
            self.assertEqual(len(revisions), 1)
            staged = sorted(str(p.relative_to(out)) for p in out.rglob("episode_*.json"))
            ep = "raw/seed5/episode_00000"
            self.assertEqual(staged, [f"s_1_h1/{ep}0.json", f"s_1_h2/{ep}1.json", f"s_3_h1/{ep}0.json"])
            self.assertEqual((out / "s_1_h2/raw/seed5/episode_000001.json").resolve().parts[-6], "h2")

    def test_shards_are_marked_as_training_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "9").mkdir()
            files = [root / "9" / f"t_{i}.json" for i in range(5)]
            for i, path in enumerate(files):
                path.write_text(json.dumps({"i": i}))
            args = SimpleNamespace(env_cfg="cfg", candidates=9, task="t", shards="100-101", per_shard=2)
            args.output = root / "run"
            with (
                patch.object(run_batch, "layout_files", lambda cfg, seed, task: files),
                patch.object(run_batch, "layout_dir", lambda cfg, seed: root / str(seed)),
            ):
                self.assertEqual(run_batch.make_shards(args), files[:4])
            self.assertEqual(json.loads((root / "101/t_1.json").read_text()), {"i": 3})
            for seed in ("9", "100", "101"):
                self.assertEqual(json.loads((root / seed / "split.json").read_text())["split"], "train")
        self.assertEqual(run_batch.span("0-2,5"), [0, 1, 2, 5])

    def test_more_worker_slots_than_shards(self):
        slots = [(gpu, k) for k in range(3) for gpu in (0, 2)]
        queue = [100, 101, 102, 103]
        got = run_batch.assign(slots, {(0, 0): None}, queue)
        self.assertEqual(got, [((2, 0), 100), ((0, 1), 101)])
        active = {(0, 0): None, (2, 0): None, (0, 1): None}
        self.assertEqual(run_batch.assign(slots, active, queue, {0}), [((2, 1), 102)])
        self.assertEqual(run_batch.assign(slots, {}, [104, 105], {0, 2}), [])

    def test_startup_retry_does_not_skip_an_unrecorded_layout(self):
        skipped = []
        self.assertEqual(run_batch.retry_layouts(100, [0, 1], [0, 1], skipped, -9), [0, 1])
        self.assertEqual(skipped, [])
        # An exit-0 worker that crashed before resetting (unstable=False) neither skips nor loops forever.
        self.assertEqual(run_batch.retry_layouts(100, [0, 1], [0, 1], skipped, 0, unstable=False), [])
        self.assertEqual(skipped, [])
        self.assertEqual(run_batch.retry_layouts(100, [0, 1], [0, 1], skipped, 0), [1])
        self.assertEqual(skipped, ["100_0"])
        self.assertEqual(run_batch.retry_layouts(100, [0, 1], [1], skipped, 0), [1])
        self.assertEqual(skipped, ["100_0"])

    def test_resume_with_nothing_left_keeps_skipped(self):
        # a filler unit run again after it was done: layout 0 recorded, layout 1 skipped, nothing to run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            config = dict(task="t", env_cfg="cfg", candidates=9, shards="100", per_shard=2, target=2, revision=None)
            (out / "collection_config.json").write_text(json.dumps(config))
            (out / "status.json").write_text(json.dumps(dict(skipped=["100_1"], state="done")))
            (out / "raw" / "seed100").mkdir(parents=True)
            (out / "raw" / "seed100" / "episode_000000.json").write_text(json.dumps(dict(accepted=True)))
            with (
                patch.object(run_batch, "layout_files", lambda cfg, seed, task: [0, 1]),
                patch.object(run_batch, "expert_version", lambda args: None),
                patch.object(sys, "argv", ["run_batch.py", "--resume", "--output", str(out)]),
                self.assertRaises(SystemExit),
            ):
                run_batch.main()
            status = json.loads((out / "status.json").read_text())
        self.assertEqual(status["skipped"], ["100_1"])
        self.assertEqual(status["accepted"], 1)

    def test_physx_abandoned_layouts_are_retried_then_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            log.write_bytes(
                b"[PhysX] broken envs=[0] -> abandon seeds=[45]; refill from queue=[46]; new batch=[46]\n"
                b"[PhysX] broken envs=[0] -> abandon seeds=[46]; refill from queue=[]; new batch=[None]\n"
            )
            self.assertEqual(run_batch.abandoned(log, 0), [45, 46])
            self.assertEqual(run_batch.abandoned(log, log.stat().st_size), [])
        broken, skipped = {}, []
        # 46 was recorded after all (an episode file): only 45 counts; once more and it is skipped
        self.assertTrue(run_batch.skip_broken(91000, [45, 46], {46}, broken, skipped))
        self.assertEqual((broken, skipped), ({"91000_45": 1}, []))
        run_batch.skip_broken(91000, [45], set(), broken, skipped)
        self.assertEqual(skipped, ["91000_45"])
        self.assertFalse(run_batch.skip_broken(91000, [45], set(), broken, skipped))  # already skipped

    def test_worker_unstable_distinguishes_reset_instability_from_exit_zero_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            log.write_bytes(b"Success nums: 0, Fail nums: 0, Unstable nums: 2\n")
            self.assertTrue(run_batch.worker_unstable(log, 0))
            offset = log.stat().st_size
            log.write_bytes(
                log.read_bytes()
                + b"[Eval] unhandled exception during reset/run_eval: FileNotFoundError: missing\n"
                + b"Success nums: 0, Fail nums: 0, Unstable nums: 1\n"
            )
            self.assertFalse(run_batch.worker_unstable(log, offset))
            offset = log.stat().st_size
            log.write_bytes(log.read_bytes() + b"Success nums: 0, Fail nums: 0, Unstable nums: 0\n")
            self.assertFalse(run_batch.worker_unstable(log, offset))

    def test_export_jpeg_check_block_diff_and_av1_preset(self):
        import h5py

        good_jpeg = b"\xff\xd8\xff\xe0payload\xff\xd9\x00\x00\x00"
        bad_jpeg = b"\xff\xd8\xff\xe0truncated\x00\x00"
        self.assertTrue(export.valid_jpeg(good_jpeg))
        self.assertFalse(export.valid_jpeg(bad_jpeg))

        raw_img = np.full((64, 64, 3), 200, dtype=np.uint8)
        clean_video = np.full((64, 64, 3), 198, dtype=np.uint8)
        self.assertLess(export.max_block_diff(clean_video, raw_img), 5.0)
        corrupt_video = clean_video.copy()
        corrupt_video[16:32, 32:48] = 80  # lone 16x16 dark square (~120 grey diff)
        self.assertGreater(export.max_block_diff(corrupt_video, raw_img), export.BLOCK_TOLERANCE)

        class DummyEncoder:
            def __init__(self, vcodec="libsvtav1"):
                self.vcodec = vcodec
                self.preset = None
                self.queue_maxsize = 30

        export.patch_streaming_encoder(DummyEncoder, av1_preset=8)
        export.patch_streaming_encoder(DummyEncoder, av1_preset=6)  # re-patching wraps original __init__, not itself
        enc = DummyEncoder()
        self.assertEqual((enc.queue_maxsize, enc.preset), (0, 6))
        h264 = DummyEncoder(vcodec="h264_nvenc")
        self.assertEqual((h264.queue_maxsize, h264.preset), (0, None))

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "episode_000000.hdf5"
            with h5py.File(h5_path, "w") as f:
                f["vision/cam_head/colors"] = np.asarray([good_jpeg, bad_jpeg], dtype="S32")
            with self.assertRaisesRegex(ValueError, "frame 1: truncated or corrupt JPEG"):
                export.check_raw_episodes([(h5_path, {"frames": 2, "cameras": ["cam_head"]})])
            with self.assertRaises(FileNotFoundError):
                export.check_raw_episodes([(Path(tmp) / "missing.hdf5", {"frames": 1, "cameras": ["cam_head"]})])

    def test_collect_env_key_supports_openflywheel_and_legacy_robodojo(self):
        import os

        env_obj = SimpleNamespace(seed_manager=SimpleNamespace(idx=0, ed_idx=3))
        payload = json.dumps({"plugin": "json", "output": "/tmp/out"})
        with (
            patch.dict(os.environ, {"OPENFLYWHEEL_DATAGEN": payload}, clear=False),
            patch.object(collect, "record_episode") as rec,
        ):
            os.environ.pop("ROBODOJO_DATAGEN", None)
            collect.eval_one_episode(env_obj, None)
            self.assertEqual(rec.call_args[0][2], Path("/tmp/out"))
        with (
            patch.dict(os.environ, {"ROBODOJO_DATAGEN": payload}, clear=False),
            patch.object(collect, "record_episode") as rec,
        ):
            os.environ.pop("OPENFLYWHEEL_DATAGEN", None)
            collect.eval_one_episode(env_obj, None)
            self.assertEqual(rec.call_args[0][2], Path("/tmp/out"))

    def test_a_worker_counts_as_started_after_its_first_step_in_this_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            log.write_bytes(b"env0 step: \x1b[92m1 / 1200\x1b[0m\r")
            offset = log.stat().st_size
            self.assertTrue(run_batch.stepped(log, 0))
            self.assertFalse(run_batch.stepped(log, offset))

    def test_a_shard_splits_into_chunks_of_layouts(self):
        args = SimpleNamespace(chunk=4)
        got = run_batch.chunks(args, 100, [0, 1, 2, 5, 6, 7, 8, 9, 10])
        self.assertEqual(got, [(100, [0, 1, 2, 5]), (100, [6, 7, 8, 9]), (100, [10])])


class MotionTests(unittest.TestCase):
    ARM, GRIP = "left_arm_joint_state", "left_ee_joint_state"

    def episode(self, speed, grip_step, high):
        """61 frames: the first arm joint at `speed` rad/s from 0, the gripper command by `grip_step` per action up to
        1; arm commands lead the state by a frame."""
        t = np.arange(61)
        arm = np.zeros((61, 7), np.float32)
        arm[:, 0] = speed / 30 * t
        grip = np.minimum(t * grip_step, 1.0)[:, None].astype(np.float32)
        states, actions = {self.ARM: arm, self.GRIP: grip}, {self.ARM: np.vstack([arm[1:], arm[-1:]]), self.GRIP: grip}
        return states, actions, {self.ARM: [[-1.0, high]] * 7}

    def test_commands_at_the_real_limits_pass(self):
        from scripts.datagen import motion

        states, actions, limits = self.episode(motion.REAL_ARM_VMAX, motion.REAL_GRIP_RATE, 1.6 + 0.05)
        privileged = {
            "motion_manipulability_" + self.ARM: np.linspace(0.2, 0.05, 61),
            "motion_grip_force_left_ee": np.full((61, 2), 3.0),
            "steps": np.arange(3),
        }
        moved = motion.summarize(states, actions, 30, {self.ARM: []}, limits, privileged)
        arm = moved["arms"][self.ARM]
        self.assertEqual(moved["violations"], [])
        self.assertAlmostEqual(arm["cmd_speed_max"], 0.8, places=4)
        self.assertAlmostEqual(arm["cmd_acc_max"], 24.0, places=2)  # the stop at the end, in one action
        self.assertAlmostEqual(arm["cmd_limit_margin_min"], 0.05, places=5)
        self.assertAlmostEqual(arm["limit_margin_min"], 0.05, places=5)
        self.assertEqual((moved["manipulability_min"], moved["grip_force_max"]), ({self.ARM: 0.05}, {"left_ee": 3.0}))

    def test_commands_past_the_real_limits_are_flagged(self):
        from scripts.datagen import motion

        states, actions, limits = self.episode(1.0, 1.0, 2.0 + 0.02)
        moved = motion.summarize(states, actions, 30, {self.ARM: []}, limits)
        self.assertEqual(len(moved["violations"]), 3)  # arm speed, joint limit margin, gripper rate
        self.assertNotIn(
            "cmd_limit_margin_min", motion.summarize(states, actions, 30, {self.ARM: []})["arms"][self.ARM]
        )

    def test_worst_takes_the_least_margin_and_the_largest_rest(self):
        from scripts.datagen import motion

        arms = {
            "a": dict(cmd_speed_max=0.5, cmd_limit_margin_min=0.3),
            "b": dict(cmd_speed_max=0.7, cmd_limit_margin_min=0.1),
        }
        moved = dict(
            arms=arms, grippers={}, manipulability_min={"a": 0.2, "b": 0.1}, grip_force_max={}, violations=["x"]
        )
        self.assertEqual(
            motion.worst(moved),
            dict(
                motion_cmd_speed_max=0.7,
                motion_cmd_limit_margin_min=0.1,
                motion_manipulability_min=0.1,
                motion_violations=1,
            ),
        )

    def test_report_spreads_the_worst_metrics_and_counts_kinds(self):
        from scripts.datagen import motion

        within = self.episode(motion.REAL_ARM_VMAX, motion.REAL_GRIP_RATE, 2.02)[:2]  # first joint up to 1.6 rad
        over = self.episode(1.0, 1.0, 2.02)[:2]  # up to 2.0 rad, 0.02 from the limit
        result = motion.report([(0, *within), (1, *over)], 30, {self.ARM: [[-1.0, 2.02]] * 7})
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["over_real_limits"], dict(episodes=1, arm_speed=1, limit_margin=1, gripper_rate=1))
        speed = result["metrics"]["motion_cmd_speed_max"]
        self.assertAlmostEqual(speed["min"], 0.8, places=3)
        self.assertAlmostEqual(speed["max"], 1.0, places=3)

    def test_audit_recomputes_motion_from_the_hdf5(self):
        import h5py

        from scripts.datagen import audit

        states, actions, limits = self.episode(1.0, 0.1, 2.0)
        quality = dict(frames=61, fps=30, state_keys=[self.ARM, self.GRIP], cameras=[], joint_names={self.ARM: []})
        plugin = SimpleNamespace(audit=lambda privileged, q: dict(steps=len(privileged["steps"])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000000.hdf5"
            with h5py.File(path, "w") as f:
                for key in quality["state_keys"]:
                    f[f"state/{key}s"], f[f"action/{key}s"] = states[key], actions[key]
                f["privileged/steps"] = np.arange(3)
            metrics, violations = audit.check_episode(path, quality, plugin)  # an older sidecar: no joint_limits
            quality["joint_limits"] = limits
            limited, _ = audit.check_episode(path, quality, plugin)
        self.assertEqual(metrics["steps"], 3)
        self.assertAlmostEqual(metrics["motion_cmd_speed_max"], 1.0, places=4)
        self.assertNotIn("motion_cmd_limit_margin_min", metrics)
        self.assertAlmostEqual(limited["motion_cmd_limit_margin_min"], 0.0, places=5)
        self.assertEqual(len(violations), 1)


class ContractTests(unittest.TestCase):
    def obs(self, ee_state):
        manager = ObsManager(
            obs_config=dict(robot=dict(joint_states=True, world_ee_state=False, ee_state=ee_state)),
            num_envs=1,
            dt=1 / 300,
            task_name="t",
            description_cfg={},
        )
        robot = dex1("left")
        manager.camera_manager = manager.capture_manager = None
        manager.instruction = [""]
        manager.robot_manager = SimpleNamespace(
            robot_list=[robot],
            get_joint=lambda robot, env_idx_list: {0: np.zeros(7)},
            get_end_effector_real_val=lambda robot, env_idx_list: {0: np.array([0.0070, 0.0086])},
            get_gripper_force=lambda robot, env_idx_list: {0: np.array([18.0, 17.5])},
            restore_name=lambda key: key[: -len("_joint_state")],
            get_robot_by_gripper_name=lambda name: robot,
            control_manager=SimpleNamespace(prev_control=[{"left_ee_joint_state": {"position": [CLOSED, CLOSED]}}]),
        )
        return manager.get_obs(env_idx_list=[0])[0]

    def test_gripper_state_is_measured_when_configured(self):
        measured = self.obs("measured")
        # commanded closed (0), the fingers stopped on a ~4 cm object (mean joint 0.0078 m)
        np.testing.assert_allclose(measured["state"]["left_ee_joint_state"], [(CLOSED - 0.0078) / (CLOSED - OPEN)])
        self.assertEqual(measured["action"]["left_ee_joint_state"], [0.0])
        np.testing.assert_allclose(measured["gripper_force"]["left_ee"], [18.0, 17.5])  # outside obs["state"]
        self.assertEqual(self.obs("command")["state"]["left_ee_joint_state"], [0.0])
        with self.assertRaisesRegex(ValueError, "ee_state"):
            self.obs("both")

    def test_eval_refuses_training_layouts_and_runs_selected_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory, "Eval_Layout/RoboDojo/cfg/5")
            folder.mkdir(parents=True)
            for i in range(3):
                (folder / f"t_{i}.json").write_text("{}")
            (folder / "split.json").write_text('{"split": "train"}')
            config = dict(num_envs=1, task_name="t", config_name="cfg", seed=5)
            with patch.object(seed_manager, "ASSETS_PATH", directory):
                with self.assertRaisesRegex(ValueError, "training layouts"):
                    seed_manager.SeedManager(config).init_eval()
                manager = seed_manager.SeedManager(dict(config, allow_train_layouts=True, layout_ids=[2, 0]))
                manager.init_eval()
                self.assertEqual(manager.seed_list, [0, 2])
                with self.assertRaisesRegex(ValueError, r"\[7\]"):
                    seed_manager.SeedManager(dict(config, allow_train_layouts=True, layout_ids=[7])).init_eval()

    def test_measuring_cup_audit_recomputes_every_bean(self):
        from scripts.datagen.plugins import g1d_pour_beans_measuring_cups as plugin

        hold = plugin.HOLD
        cups = np.tile([[-0.15, 0, 0, 1, 0, 0, 0], [0.15, 0, 0, 1, 0, 0, 0]], (hold, 1, 1))
        beans = np.tile([[0.147, 0, 0], [0.147, 0.005, 0]], (hold, 1, 1))
        privileged = dict(
            initial_particles=np.array([[-0.153, 0, 0], [-0.153, 0.005, 0]]), particles=beans, cup_poses=cups
        )
        quality = dict(frames=hold, initial_counts=dict(total=2))
        self.assertEqual(plugin.audit(privileged, quality)["beans"], 2)
        beans[-1, 1] = [0.0, 0.0, 0.0]  # one bean on the table in the last frame
        with self.assertRaises(AssertionError):
            plugin.audit(privileged, quality)

    def test_pick_anything_audit_checks_lift_others_and_name(self):
        from scripts.datagen.plugins import g1d_pick_anything as plugin

        hold, up = plugin.HOLD, [1.0, 0, 0, 0]
        start = [[0.2, -0.4, 0.75], [0.0, -0.4, 0.76], [-0.2, -0.4, 0.74]]
        poses = np.array([[*start[0], *up], [0.0, -0.4, 0.76 + 0.12, *up], [*start[2], *up]])
        privileged = dict(object_poses=np.tile(poses, (hold, 1, 1)), gripper_openness=np.zeros((hold, 2)))
        quality = dict(
            task=plugin.TASK,
            frames=hold,
            labels=["distractor_0", "target", "distractor_1"],
            start=start,
            usable_names=dict(zh=["红色杯子"], en=["red cup"]),
            instruction="Pick up the red cup.",
        )
        self.assertAlmostEqual(plugin.audit(privileged, quality)["lift_m"], 0.12, places=4)
        low, pushed = privileged["object_poses"].copy(), privileged["object_poses"].copy()
        low[-1, 1, 2] = 0.76 + 0.05  # the target slipped down in the last frame
        pushed[-1, 2, 0] += 0.04  # a distractor dragged along
        for changed in (dict(privileged, object_poses=low), dict(privileged, object_poses=pushed)):
            with self.assertRaises(AssertionError):
                plugin.audit(changed, quality)
        with self.assertRaises(AssertionError):
            plugin.audit(privileged, dict(quality, instruction="Pick up the cup."))

    def test_pick_expert_v4_plugins_keep_v3_acceptance(self):
        from scripts.datagen.plugins import (
            g1d_pick_anything as pick,
            g1d_pick_anything_v4 as pick4,
            g1d_place_anything as place,
            g1d_place_anything_v4 as place4,
            g1d_put_anything_in as put,
            g1d_put_anything_in_v4 as put4,
        )
        from scripts.g1d import scripted_demo as demo

        for v4, v3 in ((pick4, pick), (place4, place), (put4, put)):
            self.assertEqual((v4.TASK, v4.ACCEPTANCE, v4.HOLD, v4.audit), (v3.TASK, v3.ACCEPTANCE, v3.HOLD, v3.audit))
            self.assertGreater(v4.STEP_LIM, v3.STEP_LIM)  # room for the retries
        self.assertEqual((put4.Episode.task, put4.Episode.kind), (put.TASK, put.KIND))
        self.assertIs(demo.DEMOS["g1d_pick_anything"], demo.pick_anything_actions)  # v3 stays the default
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv(Path(directory) / "layout.json")
        far = {k: np.full(7 if "arm" in k else 1, 1.0) for k in KEYS}
        for plugin, entry in ((pick4, "pick_anything_v4_actions"), (place4, "place_anything_v4_actions")):
            ep = plugin.Episode.__new__(plugin.Episode)
            ep.env = env
            with patch.object(demo, entry, return_value=iter([far])) as actions:
                sent = list(ep.actions())
            actions.assert_called_once_with(env)
            self.assertLessEqual(sent[0]["left_arm_joint_state"][0], demo.REAL_ARM_VMAX / 30 + 1e-9)  # Limiter

    def test_pick_expert_v41_plugins(self):
        from scripts.datagen.plugins import (
            g1d_pick_anything as pick,
            g1d_pick_anything_v41 as pick41,
            g1d_place_anything as place,
            g1d_place_anything_v41 as place41,
            g1d_put_anything_in as put,
            g1d_put_anything_in_v41 as put41,
        )
        from scripts.g1d import grasp_v41, scripted_demo as demo

        for v41, v3 in ((pick41, pick), (place41, place), (put41, put)):
            self.assertEqual(
                (v41.TASK, v41.ACCEPTANCE, v41.HOLD, v41.audit), (v3.TASK, v3.ACCEPTANCE, v3.HOLD, v3.audit)
            )
            self.assertGreater(v41.STEP_LIM, v3.STEP_LIM)
        robot = grasp_v41.GraspRobot41
        self.assertTrue(issubclass(robot, demo.GraspRobot))
        self.assertEqual((robot.first_try_grip_check, robot.deep_first), (False, False))
        self.assertTrue(demo.GraspRobot.first_try_grip_check)  # pick v4 as registered
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv(Path(directory) / "layout.json")
        far = {k: np.full(7 if "arm" in k else 1, 1.0) for k in KEYS}
        for plugin, entry in ((pick41, "pick_anything_actions"), (place41, "place_anything_actions")):
            ep = plugin.Episode.__new__(plugin.Episode)
            ep.env = env
            with patch.object(grasp_v41, entry, return_value=iter([far])) as actions:
                list(ep.actions())
            actions.assert_called_once_with(env)

    def test_place_expert_v42(self):
        from scripts.datagen.plugins import (
            g1d_place_anything_v41 as place41,
            g1d_place_anything_v42 as place42,
            g1d_put_anything_in_v41 as put41,
            g1d_put_anything_in_v42 as put42,
        )
        from scripts.g1d import place_v42

        for v42, v41 in ((place42, place41), (put42, put41)):
            self.assertEqual(
                (v42.TASK, v42.KIND, v42.ACCEPTANCE, v42.HOLD, v42.STEP_LIM, v42.audit),
                (v41.TASK, v41.KIND, v41.ACCEPTANCE, v41.HOLD, v41.STEP_LIM, v41.audit),
            )
        from scripts.g1d import grasp_v41, scripted_demo as demo

        self.assertTrue(issubclass(place_v42.PlaceRobot, grasp_v41.GraspRobot41))
        self.assertFalse(place_v42.PlaceRobot.placing)  # the grasp steps with Arm.ik as v4.1's
        self.assertGreater(place_v42.PLACE_JOINT_MARGIN, demo.REAL_JOINT_MARGIN)  # stops inside the Limiter's clamp
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv(Path(directory) / "layout.json")
        far = {k: np.full(7 if "arm" in k else 1, 1.0) for k in KEYS}
        for plugin in (place42, put42):
            ep = plugin.Episode.__new__(plugin.Episode)
            ep.env = env
            with patch.object(place_v42, "place_anything_actions", return_value=iter([far])) as actions:
                list(ep.actions())
            actions.assert_called_once_with(env)

    def test_place_expert_v43(self):
        from scripts.datagen.plugins import (
            g1d_place_anything_v41 as place41,
            g1d_place_anything_v43 as place43,
            g1d_put_anything_in_v41 as put41,
            g1d_put_anything_in_v43 as put43,
        )
        from scripts.g1d import place_v43, scripted_demo as demo

        for v43, v41 in ((place43, place41), (put43, put41)):
            self.assertEqual(
                (v43.TASK, v43.KIND, v43.ACCEPTANCE, v43.HOLD, v43.STEP_LIM, v43.audit),
                (v41.TASK, v41.KIND, v41.ACCEPTANCE, v41.HOLD, v41.STEP_LIM, v41.audit),
            )
        self.assertIs(place43.Episode.expert, place_v43.place_anything_actions)
        self.assertIs(put43.Episode.expert, place_v43.put_anything_in_actions)
        cands = [(None, None, None, 0.02, n) for n in ("v3", "deep", "wrist turned", "tilted, 45 deg")]
        place, put = (
            place_v43.PlaceRobot43.__new__(place_v43.PlaceRobot43),
            place_v43.PutRobot43.__new__(place_v43.PutRobot43),
        )
        self.assertEqual([c[4] for c in place.order(cands)], ["v3", "deep", "wrist turned", "tilted, 45 deg"])
        self.assertEqual([c[4] for c in put.order(cands)], ["deep", "v3", "wrist turned", "tilted, 45 deg"])
        # Route replays the sim's step IK on the URDF: it gets to a pose near where the arm is, inside the clamp
        urdf = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"
        arm = [f"left_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
                                           "wrist_pitch", "wrist_yaw")]  # fmt: skip
        reach = demo.Reach.__new__(demo.Reach)
        reach.chain = demo.urdf_chain(urdf, "left_dex1_base_link", arm, {})
        reach.lim = np.tile([-1.5, 1.5], (7, 1))
        reach.reach, reach.base = demo.Arm.REACH, np.eye(4)
        route = place_v43.Route(reach)
        q_goal = np.array([-0.4, 0.3, 0.1, 0.9, 0.2, -0.3, 0.1])
        rot, point, _ = route.pose(q_goal)
        q, off = route.move(q_goal + 0.15, [(point, rot, 0.0, 40)], place_v43.ARM_IK_MARGIN)
        self.assertLess(off, demo.PATTERN_SETTLE[0])
        self.assertTrue((np.abs(q) <= 1.5 + 1e-9).all())
        far = point + [2.0, 0.0, 0.0]  # out of reach: the replay ends short, joints held at the clamp
        q, off = route.move(q_goal, [(far, rot, 0.0, 40)], place_v43.ARM_IK_MARGIN)
        self.assertGreater(off, 0.5)
        self.assertTrue((np.abs(q) <= 1.5 + 1e-9).all())

    def test_reach_solves_what_the_urdf_reaches(self):
        from scipy.spatial.transform import Rotation

        from scripts.g1d import scripted_demo as demo

        urdf = ROOT / "real2sim/robot/g1d_v4/g1_d_full_dex1_v4.urdf"
        arm = [f"left_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
                                           "wrist_pitch", "wrist_yaw")]  # fmt: skip
        reach = demo.Reach.__new__(demo.Reach)
        reach.chain = demo.urdf_chain(urdf, "left_dex1_base_link", arm, {})
        reach.lim = np.tile([-1.5, 1.5], (7, 1))
        reach.reach, reach.base = demo.Arm.REACH, np.eye(4)
        q_goal = np.array([-0.4, 0.3, 0.1, 0.9, 0.2, -0.3, 0.1])
        t = reach.fk(q_goal)
        point, rot = t[:3, 3] + t[:3, :3] @ reach.reach, Rotation.from_matrix(t[:3, :3])
        q, miss_m, miss_deg = reach.solve(point, rot, q_goal + 0.15)
        self.assertLess(miss_m, 1e-3)
        self.assertLess(miss_deg, 0.5)
        _, far_m, _ = reach.solve(point + [2.0, 0, 0], rot, q)  # 2 m away: out of reach, and says so
        self.assertGreater(far_m, 1.0)

    def test_pick_anything_names_and_layouts(self):
        import random

        from scripts.objects import pick_anything as pa

        own = ["cup", "paper cup", "red paper cup", "red paper cup with a white rim"]
        self.assertEqual(pa.usable_names(own, [["cup", "mug"]]), own[1:])
        self.assertEqual(pa.usable_names(own, [["paper cup", "blue paper cup"]]), own[2:])  # "cup" is in "paper cup"
        self.assertEqual(pa.usable_names(["x", "y"], [["x", "y"]]), ["y"])
        pool = pa.load_pool()
        self.assertEqual(len({(o["cat"], o["idx"]) for o in pool}), len(pool))
        self.assertTrue(all(pa.graspable(o["size"], o["mass"]) for o in pool))
        order, targets = pa.pool_order(pool), set()

        def entry(cat, idx, label, xy, yaw):
            return dict(c=cat, j=idx, label=label, xy=xy, yaw=yaw)

        from scripts.objects import block_patterns

        for i in range(60):
            rigid = pa.layout(random.Random(f"t/{i}"), i, entry, pool)
            placed = [e for es in rigid.values() for e in es]
            target = next(e for e in placed if e["label"] == "target")
            obj = pool[order[i]]
            self.assertEqual((target["c"], target["j"]), (obj["cat"], obj["idx"]))
            # the narrow side lies across the grip yaw of its spot, within GRASP_YAW, either way round
            across = target["yaw"] + (0.0 if obj["size"][0] <= obj["size"][1] else 90.0)
            off = (across - block_patterns.grip_yaw(*target["xy"]) + 90.0) % 180.0 - 90.0
            self.assertLessEqual(abs(off), pa.GRASP_YAW + 1e-9)
            self.assertTrue(pa.DISTRACTORS[0] <= len(placed) - 1 <= pa.DISTRACTORS[1])
            self.assertEqual(len({(e["c"], e["j"]) for e in placed}), len(placed))
            self.assertTrue(pa.TARGET_MID_GAP <= abs(target["xy"][0]) <= pa.TARGET_X[1])
            self.assertTrue(pa.TARGET_Y[0] <= target["xy"][1] <= pa.TARGET_Y[1])
            targets.add((target["c"], target["j"]))
        self.assertEqual(len(targets), 60)

    def test_open_anything_audit_checks_opening_siblings_and_name(self):
        from scripts.datagen.plugins import g1d_open_anything as plugin

        hold, goal = plugin.HOLD, np.radians(40)
        # the door shut while the hand comes to the handle, swung to 55 deg open (sign -1) over HOLD frames and held
        # there HOLD frames; the drawer shut
        values = np.tile([-np.radians(55), 0.001], (3 * hold, 1))
        values[:hold, 0] = 0.0
        values[hold : 2 * hold, 0] = -np.radians(np.linspace(0, 55, hold))
        gaps = np.r_[np.linspace(0.2, 0.0, hold), np.zeros(2 * hold)].astype(np.float32)
        privileged = dict(joint_values=values, gripper_openness=np.ones((3 * hold, 2)), hand_gap=gaps)
        quality = dict(
            task=plugin.TASK,
            frames=3 * hold,
            joint_type="revolute",
            joints=["door_joint", "drawer_joint"],
            revolute=[True, False],
            start=[0.0, 0.0],
            closed=0.0,
            sign=-1.0,
            goal=goal,
            role="door",
            object_names=dict(zh=["储物柜", "白色储物柜"], en=["cabinet"]),
            instruction="打开白色储物柜的门",
        )
        self.assertAlmostEqual(plugin.audit(privileged, quality)["opening"], np.radians(55), places=4)
        shut, drawer = values.copy(), values.copy()
        shut[-1, 0] = -np.radians(30)  # swung back under the goal in the last frame
        drawer[-1, 1] = 0.05  # the other part pulled out too
        alone = values.copy()
        alone[hold // 2 : hold - 2, 0] = -np.radians(np.linspace(0, 20, hold // 2 - 2))  # opened before the hand
        jump = values.copy()
        jump[2 * hold + 3, 0] -= np.radians(15)  # thrown 15 deg in one frame and back
        for changed in (dict(privileged, joint_values=v) for v in (shut, drawer, alone, jump)):
            with self.assertRaises(AssertionError):
                plugin.audit(changed, quality)
        with self.assertRaises(AssertionError):
            plugin.audit(privileged, dict(quality, instruction="打开微波炉的门"))

    def test_open_anything_pool_layouts_and_names(self):
        import random

        from scripts.objects import open_anything as oa

        pool = oa.load_pool()
        keys = [(e["id"], e["joint"]) for e in pool["entries"]]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual({e["split"] for e in pool["entries"]}, {"train", "eval"})
        for split in ("train", "eval"):  # an object is in one split only
            other = {e["id"] for e in pool["entries"] if e["split"] != split}
            self.assertFalse({e["id"] for e in oa.entries(split, pool)} & other)
        train = oa.entries("train", pool)

        def entry(cat, idx, label, xy, yaw, bottom):
            return dict(c=cat, j=idx, label=label, xy=xy, yaw=yaw, bottom=bottom)

        seen = set()
        for i in range(len(train)):
            arts, part = oa.layout(random.Random(f"t/{i}"), i, entry, pool=pool)
            (placed,) = [e for es in arts.values() for e in es]
            self.assertEqual(part["split"], "train")
            seen.add((part["id"], part["joint"]))
            gx, gy = oa._turn(part["grasp"][:2], placed["yaw"])
            self.assertTrue(oa.GRIP_X[0] - 1e-9 <= placed["xy"][0] + gx <= oa.GRIP_X[1] + 1e-9)
            self.assertGreaterEqual(placed["xy"][1] + gy, oa.REACH_Y[0])
            text = oa.instruction(random.Random(i), part)
            self.assertTrue(any(n in text for lang in part["names"].values() for n in lang), text)
        self.assertEqual(len(seen), len(train))
        mk = lambda j, x, z: dict(joint=j, role="drawer", part="drawer", front=[0, -1, 0], centre=[x, 0, z])  # noqa: E731
        names = oa.part_names([mk("a", -0.2, 0.3), mk("b", 0.2, 0.3), mk("c", -0.2, 0.1), mk("d", 0.2, 0.1)])
        self.assertEqual(names["a"], ("左边上面的抽屉", "top left drawer"))  # the robot's left: the object's -x
        stack = oa.part_names([mk(k, 0, z) for k, z in zip("abcd", (0.3, 0.2, 0.1, 0.0))])
        want = ["top drawer", "second drawer", "third drawer", "bottom drawer"]
        self.assertEqual([stack[k][1] for k in "abcd"], want)

    def test_open_anything_grasp_methods_follow_the_part_geometry(self):
        from scripts.objects import open_anything as oa

        rng = np.random.default_rng(0)

        def box(lo, hi, n=4000):
            return rng.uniform(lo, hi, size=(n, 3))

        # a door 40 x 40 cm, 2 cm thick, front at y -0.20 (front = -y), hinge on its left at x -0.2
        front, pivot, axis = np.array([0.0, -1.0, 0.0]), np.array([-0.2, -0.19, 0.0]), np.array([0.0, 0.0, 1.0])
        panel = box([-0.2, -0.20, -0.2], [0.2, -0.18, 0.2], 20000)
        carcass = box([-0.22, -0.18, -0.22], [0.22, 0.10, 0.22], 20000)

        def bar(standoff, thick=0.012, x=0.15):  # a standing bar on two posts, `standoff` out of the face
            posts = [
                box([x - 0.006, -0.20 - standoff, z - 0.006], [x + 0.006, -0.20, z + 0.006], 400) for z in (-0.06, 0.06)
            ]
            rod = box([x - thick / 2, -0.20 - standoff, -0.07], [x + thick / 2, -0.20 - standoff + thick, 0.07], 2000)
            return np.concatenate([*posts, rod])

        deep = oa._handle(np.concatenate([panel, bar(0.035)]), carcass, front, False, pivot, axis, True)
        self.assertEqual(deep["method"], "pinch")
        shallow = oa._handle(np.concatenate([panel, bar(0.03, thick=0.008)]), carcass, front, False, pivot, axis, True)
        self.assertEqual(shallow["method"], "pinch")
        self.assertGreaterEqual(shallow["gap"], oa.HOOK_GAP)
        hooked = oa._handle(np.concatenate([panel, bar(0.021, thick=0.006)]), carcass, front, False, pivot, axis, True)
        self.assertEqual(hooked["method"], "hook")  # too shallow to pinch, room behind the bar for the fingers
        self.assertLess(hooked["approach"][0], 0)  # in from the free edge's side (+x), toward the hinge
        standing = oa._handle(np.concatenate([panel, bar(0.016)]), carcass, front, False, pivot, axis, True)
        self.assertIn("standing bar", standing["reject"])  # shallow, no room behind: the jaw would come from above
        posts = [box([x - 0.006, -0.216, 0.144], [x + 0.006, -0.20, 0.156], 400) for x in (-0.06, 0.06)]
        rod = box([-0.07, -0.216, 0.144], [0.07, -0.204, 0.156], 2000)  # a lying bar 16 mm out of the face
        sided = oa._handle(np.concatenate([panel, *posts, rod]), carcass, front, False, pivot, axis, True)
        self.assertEqual(sided["method"], "side")  # in level from the bar's end, the pads closing top and bottom
        self.assertGreater(abs(sided["approach"][0]), 0.9)
        self.assertGreater(abs(sided["close"][2]), 0.9)
        knob = box([0.14, -0.215, -0.01], [0.16, -0.20, 0.01], 500)  # 15 mm, no room behind
        none = oa._handle(np.concatenate([panel, knob]), carcass, front, False, pivot, axis, True)
        self.assertIn("reject", none)  # the carcass is right behind the door's free edge
        lone = oa._handle(panel, np.zeros((0, 3)), front, False, pivot, axis, True)
        self.assertEqual(lone["method"], "edge")  # nothing behind it: pinched across its free edge
        self.assertAlmostEqual(lone["grasp"][0], 0.2 - oa.EDGE_IN, delta=0.01)

    def test_block_pattern_audit_recomputes_the_pattern(self):
        from scipy.spatial.transform import Rotation

        from scripts.datagen.plugins import g1d_build_block_pattern as plugin
        from scripts.objects import block_patterns

        hold, index = plugin.HOLD, 2  # the bridge: five blocks
        pattern = block_patterns.PATTERNS[index]
        target = block_patterns.targets(pattern)
        turn = Rotation.from_euler("z", 30, degrees=True)  # built anywhere, turned any way
        xyz = turn.apply(target[:, :3]) + [0.1, -0.4, 0.73]
        wxyz = [np.roll(Rotation.from_euler("z", 30 + yaw, degrees=True).as_quat(), 1) for yaw in target[:, 3]]
        poses = np.tile(np.c_[xyz, wxyz], (hold, 1, 1))
        privileged = dict(block_poses=poses, gripper_openness=np.ones((hold, 2)))
        quality = dict(frames=hold, pattern_index=index)
        report = plugin.audit(privileged, quality)
        self.assertEqual(report["pattern"], "bridge")
        self.assertLess(report["max_xy_m"], 1e-6)
        shifted = poses.copy()
        shifted[-1, 3, 0] += 0.012  # the plank 12 mm off in the last frame: within the task's 15 mm, not here
        closed = np.ones((hold, 2))
        closed[0, 1] = 0.5  # a hand still holding a block
        for changed in (dict(block_poses=shifted), dict(gripper_openness=closed)):
            with self.assertRaises(AssertionError):
                plugin.audit({**privileged, **changed}, quality)

    def test_block_sort_audit_checks_trays_cards_and_cavities(self):
        from scipy.spatial.transform import Rotation

        from scripts.datagen.plugins import g1d_sort_blocks_by_color as plugin, g1d_sort_blocks_by_shape as shape_plugin
        from scripts.objects import block_sorting as bs

        self.assertIs(shape_plugin.audit, plugin.audit)
        self.assertEqual(shape_plugin.TASK, "g1d_sort_blocks_by_shape")
        hold, up = plugin.HOLD, [1.0, 0, 0, 0]
        z = 0.716 + bs.BOX_SIZE[2] / 2
        trays = np.array([[*bs.box_xy(b), z, *up] for b in (0, 1)])
        cards = np.array([[*bs.card_xy(b), 0.717, *up] for b in (0, 1)])
        blocks = np.array([[*bs.slot_xy(b, s), 0.735, *up] for b in (0, 1) for s in (0, 1)])
        start = dict(trays=trays.tolist(), cards=cards.tolist())
        tile = lambda a: np.tile(a, (hold, 1, 1))  # noqa: E731
        privileged = dict(
            block_poses=tile(blocks),
            tray_poses=tile(trays),
            card_poses=tile(cards),
            gripper_openness=np.ones((hold, 2)),
        )
        quality = dict(
            frames=hold, labels=[f"block_{i}" for i in range(4)], tray_of=[0, 0, 1, 1], by="color", start=start
        )
        self.assertLess(plugin.audit(privileged, quality)["max_tray_move_m"], 1e-9)
        outside = privileged["block_poses"].copy()
        outside[-1, 0, 1] += 0.033  # slot 0 block 3.3 cm further: in the far wall, still inside the tray's outer bbox
        moved = privileged["tray_poses"].copy()
        moved[-1, 1, 0] += 0.025  # the right tray dragged 2.5 cm
        tilted = privileged["tray_poses"].copy()
        tilted[-1, 0, 3:] = np.roll(Rotation.from_euler("x", 8, degrees=True).as_quat(), 1)
        for changed in (
            dict(block_poses=outside),
            dict(tray_poses=moved),
            dict(tray_poses=tilted),
            dict(gripper_openness=np.full((hold, 2), 0.5)),
        ):
            with self.assertRaises(AssertionError):
                plugin.audit({**privileged, **changed}, quality)
        with self.assertRaises(AssertionError):  # blocks of tray 0 counted for tray 1
            plugin.audit(privileged, {**quality, "tray_of": [1, 1, 0, 0]})

    def test_cup_or_mug_audit_uses_the_target_cavity(self):
        from scripts.datagen.plugins import g1d_pour_beans_cup_or_mug as plugin
        from scripts.objects import measuring_cup_shape, plastic_mug_shape

        hold, n = plugin.HOLD, 360
        jitter = np.random.default_rng(0).uniform(-0.01, 0.01, (n, 3))
        # cavity centres in each cup's bbox frame (cups at x = -+0.15, upright)
        tall = np.array([0, 0, 0.06]) + measuring_cup_shape.origin_shift(0.17) + [-0.15, 0, 0]
        mug = np.array([0, 0, 0.05]) + plastic_mug_shape.origin_shift() + [0.15, 0, 0]
        cups = np.tile([[-0.15, 0, 0, 1, 0, 0, 0], [0.15, 0, 0, 1, 0, 0, 0]], (hold, 1, 1))
        beans = np.tile(mug + jitter, (hold, 1, 1))
        distractors = np.tile([[0.4, -0.1, 0, 1, 0, 0, 0], [-0.4, -0.1, 0, 1, 0, 0, 0]], (hold, 1, 1))
        privileged = dict(
            initial_particles=tall + jitter,
            particles=beans,
            cup_poses=cups,
            distractor_poses=distractors,
            distractor_poses_start=distractors[0].copy(),
        )
        cup_info = dict(
            source_cup=dict(kind="measuring_cup", height=0.17), target_cup=dict(kind="plastic_mug", height=0.1)
        )
        quality = dict(frames=hold, initial_counts=dict(total=n), cups=cup_info)
        report = plugin.audit(privileged, quality)
        self.assertEqual((report["beans"], report["target"]), (n, "plastic_mug"))

        def fails(**changes):
            with self.assertRaises(AssertionError):
                plugin.audit({k: changes.get(k, v) for k, v in privileged.items()}, quality)

        moved = distractors.copy()
        moved[3, 0, 0] += 0.03  # a distractor pushed 3 cm in one frame
        fails(distractor_poses=moved)
        touching = cups.copy()
        touching[3, 0, 0] = 0.07  # the tall cup's side inside the mug
        fails(cup_poses=touching)
        tilted = cups.copy()
        tilted[3, 1, 3:] = [np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0]  # target 20 deg over
        fails(cup_poses=tilted)
        stray = beans.copy()
        stray[-1, 0] = mug + [0.045, 0, 0]  # in the mug wall in the last frame
        fails(particles=stray)
        few = dict(privileged, initial_particles=privileged["initial_particles"][:250], particles=beans[:, :250])
        with self.assertRaises(AssertionError):
            plugin.audit(few, dict(quality, initial_counts=dict(total=250)))

    def test_pour_beans_plus_audit_counts_against_the_profiles(self):
        from scripts.datagen.plugins import g1d_pour_beans_plus as plugin
        from scripts.objects.test_cup_profile import straight_cup

        cup, hold, n = straight_cup(), plugin.HOLD, 300
        a, b = np.array([-0.15, -0.33, 0.8]), np.array([0.15, -0.33, 0.8])
        upright = [1.0, 0.0, 0.0, 0.0]
        frames = hold + 5
        cups = np.tile([[*a, *upright], [*b, *upright]], (frames, 1, 1))
        column = plugin.cup_profile.column(cup, n)
        distractors = np.tile([[0.4, -0.1, 0.8, *upright]], (frames, 1, 1))
        privileged = dict(
            initial_particles=column + a,
            particles=np.tile(column + b, (frames, 1, 1)),
            cup_poses=cups,
            distractor_poses=distractors,
            distractor_poses_start=distractors[0].copy(),
        )
        quality = dict(
            frames=frames,
            initial_counts=dict(total=n),
            cups={label: {"cavity": cup, "scale": 1.0} for label in plugin.LABELS},
            start_z=[0.8, 0.8],
        )
        report = plugin.audit(privileged, quality)
        self.assertEqual((report["beans"], report["in_target_share"]), (n, 1.0))
        # layout entries arrive as OmegaConf containers: the summary must still serialize
        view = MappingProxyType({"z": (0.0, 1.0), "axis": range(2), "top": 0.1})
        self.assertEqual(json.loads(json.dumps(plugin.plain(view))), {"z": [0.0, 1.0], "axis": [0, 1], "top": 0.1})

        def fails(**changes):
            with self.assertRaises(AssertionError):
                plugin.audit({k: changes.get(k, v) for k, v in privileged.items()}, quality)

        spilled = int(n * (1 - 0.95))
        stray = privileged["particles"].copy()
        stray[-1, 0] = b + [0.05, 0, 0]  # plus_v1 requires every bean in B
        fails(particles=stray)
        from scripts.datagen.plugins import g1d_pour_beans_plus_v3 as v3

        stray = privileged["particles"].copy()
        stray[-hold:, :spilled] = b + [0.05, 0, 0]  # plus_v3 allows up to 5% outside B with open grippers
        v3_priv = {**privileged, "particles": stray, "grip_command": np.ones((frames, 2), dtype=np.float32)}
        self.assertEqual(v3.audit(v3_priv, quality)["in_target_share"], v3.IN_B)
        stray[-1, spilled] = b + [0.05, 0, 0]  # one more in the last frame
        with self.assertRaises(AssertionError):
            v3.audit({**v3_priv, "particles": stray}, quality)
        with self.assertRaises(AssertionError):
            v3.audit({**v3_priv, "grip_command": np.zeros((frames, 2), dtype=np.float32)}, quality)
        touching = cups.copy()
        touching[2, 0, :3] = b - [0.069, 0, 0]  # A's wall against B's in one frame
        fails(cup_poses=touching)
        lifted = cups.copy()
        lifted[-1, 1, 2] += 0.02  # B not back on the desk
        fails(cup_poses=lifted)
        pushed = distractors.copy()
        pushed[1, 0, 0] += 0.03
        fails(distractor_poses=pushed)
        fails(initial_particles=column + b)  # beans not in A at the start

    def test_pour_beans_plus_limited_waits_for_the_limited_arms(self):
        from scripts.datagen.plugins import g1d_pour_beans_plus_limited as limited
        from scripts.g1d import scripted_demo as demo

        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv(Path(directory) / "layout.json")
        # the arms jump 1 rad at once, then the left one past its range less the margin; grippers snap open
        closed, opened = ({k: np.full(7 if "arm" in k else 1, 1.0 if "arm" in k else g) for k in KEYS} for g in (0, 1))
        opened["left_arm_joint_state"] = np.full(7, 1.99)
        ep = limited.Episode.__new__(limited.Episode)
        ep.env, ep.grip, grips = env, None, []
        with patch.object(demo, "plus_actions", return_value=iter([closed, opened])):
            sent = [grips.append(ep.grip.copy()) or a for a in ep.actions()]
        step, rate = demo.REAL_ARM_VMAX / 30, demo.REAL_GRIP_RATE
        reach = int(np.ceil(1.0 / step))  # sends until the arms are at 1 rad; the script waits that long
        np.testing.assert_allclose(
            [a["right_arm_joint_state"][0] for a in sent[:reach]], np.minimum(step * np.arange(1, reach + 1), 1)
        )
        np.testing.assert_allclose(sent[reach]["right_ee_joint_state"], [rate])  # the next command starts after
        left = [a["left_arm_joint_state"][0] for a in sent]
        self.assertAlmostEqual(left[-1], 1.95)  # clipped REAL_JOINT_MARGIN inside the 2 rad limit, then done
        self.assertLessEqual(max(np.diff(left)), step + 1e-9)
        np.testing.assert_allclose(grips[-1], [1.0, 1.0])
        np.testing.assert_allclose(grips[reach], [rate, rate])  # grip_command holds what was sent
        self.assertEqual((limited.ACCEPTANCE, limited.audit), ("plus_v3", limited.v3.audit))

    def test_pour_beans_plus_counts_how_beans_leave_a_cup(self):
        from scripts.datagen.plugins import g1d_pour_beans_plus as plugin
        from scripts.objects.test_cup_profile import straight_cup

        cup = straight_cup()
        a, b = np.array([-0.15, -0.33, 0.8]), np.array([0.15, -0.33, 0.8])
        poses = np.array([[*a, 1.0, 0.0, 0.0, 0.0], [*b, 1.0, 0.0, 0.0, 0.0]])
        r = cup["r"][len(cup["r"]) // 2]
        wall = cup["body"]["r"][len(cup["body"]["r"]) // 2]
        top, bottom = cup["z"][-1], cup["bottom"]
        start = np.array([
            [0.0, 0.0, top - 0.005],  # over the rim into B
            [0.0, 0.0, top - 0.005],  # over the rim onto the desk
            [r - 0.003, 0.0, 0.0],  # through the wall
            [0.0, 0.0, cup["z"][0] + 0.003],  # through the floor
            [r - 0.003, 0.0, 0.0],  # into the wall, then through it a frame later
        ]) + a  # fmt: skip
        ep = object.__new__(plugin.Episode)
        ep.cups = {label: (cup, 1.0) for label in plugin.LABELS}
        ep.home, ep.last = np.zeros(len(start), int), (start, poses)
        ep.escapes = {label: [0] * len(plugin.EXITS) for label in plugin.LABELS}
        ep.leaks, ep.history = [], dict(particles=[start])
        after = start.copy()
        after[0] = b + [0.0, 0.0, 0.0]
        after[1] = a + [0.0, 0.0, cup["top"] + 0.01]
        after[2] = a + [wall + 0.002, 0.0, 0.0]
        after[3] = a + [0.0, 0.0, bottom - 0.002]
        after[4] = a + [(r + plugin.cup_profile.SLACK + wall + plugin.OUT_M) / 2, 0.0, 0.0]
        ep.track_exits(after, poses)
        self.assertEqual(ep.home.tolist(), [1, -1, -1, -1, 0])
        self.assertEqual(ep.escapes["source_cup"], [2, 1, 1])
        self.assertEqual(ep.leaks, [[1, 0, 1], [1, 0, 2]])
        ep.history["particles"].append(after)
        later = after.copy()
        later[4] = a + [wall + 0.005, 0.0, -0.01]
        ep.track_exits(later, poses)
        self.assertEqual(ep.escapes["source_cup"], [2, 2, 1])
        self.assertEqual(ep.leaks[-1], [2, 0, 1])
        self.assertEqual(ep.escapes["target_cup"], [0, 0, 0])

    def test_rigid_beans_configs_use_32_solver_partitions(self):
        """With the default 8 GPU solver partitions rigid beans push through a held cup's wall: every rigid-bean env
        config has to run on a sim profile that raises them."""
        configs = sorted((collect.ROOT / "env_cfg").glob("*beans*rigid*.yml"))
        self.assertGreaterEqual(len(configs), 6)
        for path in configs:
            sim = collect.load_yaml(path)["config"]["sim"]
            physx = collect.load_yaml(collect.ROOT / "env_cfg" / "sim" / f"{sim}.yml")["physx"]
            self.assertEqual(physx["gpu_max_num_partitions"], 32, path.name)
            self.assertTrue(physx.get("enable_ccd"), path.name)
            self.assertTrue(physx.get("enable_external_forces_every_iteration"), path.name)
            self.assertGreaterEqual(physx.get("min_velocity_iteration_count", 0), 2, path.name)

    def test_pour_beans_plus_grip_avoids_a_cone(self):
        from scripts.g1d import scripted_demo as demo

        z = np.linspace(-0.045, 0.045, 10)
        cylinder = {"bottom": -0.045, "top": 0.045, "body": {"z": z.tolist(), "r": [0.03] * 10}}
        up, pitch = demo.plus_grip(cylinder, 1.0)
        self.assertAlmostEqual(up, 0.6 * 0.09, delta=0.003)
        self.assertEqual(pitch, demo.PLUS_PITCH[0])
        flare = 0.03 + np.clip(z, 0, None) * 0.6  # a 31-deg cone over the upper half
        cone_top = {**cylinder, "body": {"z": z.tolist(), "r": flare.tolist()}}
        up, _ = demo.plus_grip(cone_top, 1.0)
        pads = -0.045 + up + np.r_[-1, 1] * demo.PLUS_PAD / 2
        slope = np.diff(np.interp(pads, z, flare))[0] / demo.PLUS_PAD
        self.assertLessEqual(slope, np.tan(np.radians(demo.PLUS_SLOPE)))
        self.assertLess(-0.045 + up, 0.0)
        self.assertGreater(up, 0.25 * 0.09)

    def test_pour_beans_plus_palm_clearance(self):
        from scripts.g1d import scripted_demo as demo

        class Lm:
            def get_instance_pose(self, env, label):
                return np.array([0.1, -0.4, 0.8]), np.array([1.0, 0.0, 0.0, 0.0])

        z = np.linspace(-0.06, 0.06, 12)
        narrow = {"axis": [0.0, 0.0], "bottom": -0.04, "top": 0.04, "body": {"z": [-0.04, 0.04], "r": [0.025, 0.025]}}
        self.assertEqual(demo.plus_palm_push(narrow, 1.0, 0.05, 25), 0.0)
        grip, *_ = demo.profile_grasp(Lm(), 1, "target_cup", narrow, 1.0, 0.05, 25)
        np.testing.assert_allclose(grip[:2], [0.1, -0.4])
        # A 12 cm wide-mouthed cup at rim_down=0.6 puts the rim 7.2 cm above PAD_REACH, into the wrist-camera shell:
        # plus_grip raises the grasp until the push is within PLUS_PAD_AHEAD, and profile_grasp backs off by the rest.
        wide = {"axis": [0.0, 0.0], "bottom": -0.06, "top": 0.06, "body": {"z": z.tolist(), "r": [0.04] * 12}}
        self.assertGreater(demo.plus_palm_push(wide, 1.0, 0.4 * 0.12, 25), demo.PLUS_PAD_AHEAD)
        up, pitch = demo.plus_grip(wide, 1.0, rim_down=0.6)
        self.assertLessEqual(demo.plus_palm_push(wide, 1.0, up, pitch), demo.PLUS_PAD_AHEAD)
        grip, hand, *_ = demo.profile_grasp(Lm(), 1, "target_cup", wide, 1.0, up, pitch)
        fwd = hand.apply([1.0, 0.0, 0.0])[:2]
        shift = ([0.1, -0.4] - grip[:2]) @ (fwd / np.linalg.norm(fwd))
        self.assertGreater(shift, demo.PLUS_PALM_CLEAR)
        self.assertLessEqual(shift, demo.PLUS_PAD_AHEAD + 1e-9)

    def test_pour_beans_plus_ik_step(self):
        from scripts.g1d import scripted_demo as demo

        rng = np.random.default_rng(0)
        u, _, vt = np.linalg.svd(rng.normal(size=(6, 7)))
        lim = np.tile([-2.0, 2.0], (7, 1))
        err = rng.normal(size=6) * 0.01

        def jac(sigma):
            return u @ np.diag(sigma) @ vt[:6]

        def plain(j):
            return j.T @ np.linalg.solve(j @ j.T + 0.02**2 * np.eye(6), err)

        good = jac([0.7, 0.6, 0.4, 0.2, 0.12, 0.08])
        np.testing.assert_allclose(demo.plus_dls(good, err, np.zeros(7), lim), plain(good))  # far from limits
        # through a joint's limit zone the step changes smoothly, and at the limit that joint hardly moves
        steps = []
        for q1 in np.linspace(lim[1, 1] - 2 * demo.PLUS_LIMIT_ZONE, lim[1, 1], 200):
            steps.append(demo.plus_dls(good, err, np.r_[0.0, q1, np.zeros(5)], lim))
        steps = np.array(steps)
        self.assertLess(np.abs(np.diff(steps, axis=0)).max(), 0.05 * np.abs(steps).max())
        self.assertLess(abs(steps[-1, 1]), 0.05 * abs(steps[0, 1]))
        # near a singular pose the damping keeps the step bounded
        near = jac([0.7, 0.6, 0.4, 0.2, 0.12, 0.005])
        err = 0.01 * u[:, -1]
        self.assertLess(np.linalg.norm(demo.plus_dls(near, err, np.zeros(7), lim)), 0.2 * np.linalg.norm(plain(near)))
        # the joint step is scaled as a whole: same direction, no joint past PLUS_JOINT_STEP
        last = np.zeros(7)
        ask = np.r_[0.3, -0.1, 0.05, 0.0, 0.0, 0.0, 0.02]
        got = demo.plus_joint_step(ask, last)
        self.assertAlmostEqual(np.abs(got).max(), demo.PLUS_JOINT_STEP)
        np.testing.assert_allclose(got / np.linalg.norm(got), ask / np.linalg.norm(ask))
        small = np.full(7, 0.5 * demo.PLUS_JOINT_STEP)
        np.testing.assert_array_equal(demo.plus_joint_step(small, last), small)

    def test_pour_beans_plus_motion(self):
        from scipy.spatial.transform import Rotation

        from scripts.g1d import scripted_demo as demo

        # quintic moves start and stop without an acceleration jump and end on the key
        key = [(np.r_[0.0, 0.0, 0.06], Rotation.identity(), 0, 30)]
        path = demo.waypoints(np.zeros(3), Rotation.identity(), key, quintic=True)
        z = np.array([p[2] for p, _, _ in path])
        self.assertAlmostEqual(z[-1], 0.06)
        acc = np.diff(np.r_[0.0, 0.0, z, 0.06, 0.06], 2)
        self.assertLess(abs(acc[0]), 0.1 * np.abs(acc).max())
        self.assertLess(abs(acc[-1]), 0.1 * np.abs(acc).max())
        # the roll rate changes by at most PLUS_ROLL_ACCEL an action, the roll settles exactly on a held target
        state, cmd = [0.0, 0.0], []
        targets = list(np.deg2rad(np.r_[np.arange(2, 62, 2), [60] * 10, [58] * 20, [0] * 40]))
        for theta in targets:
            cmd.append(demo.plus_ease(state, theta))
        rate = np.diff(np.r_[0.0, cmd])
        self.assertLessEqual(np.abs(np.diff(np.r_[0.0, rate])).max(), demo.PLUS_ROLL_ACCEL + 1e-12)
        self.assertLessEqual(np.abs(rate).max(), demo.RETURN_RATE + 1e-12)
        self.assertEqual(cmd[-1], 0.0)
        self.assertLess(max(cmd), np.deg2rad(60) + demo.PLUS_ROLL_ACCEL * 3)
        # the lip target's 5 mm steps one way and back: at most PLUS_LIP_ACCEL / PLUS_LIP_RATE, then on the target
        state, cmd = [0.0, 0.0], []
        for target in [-0.005] * 30 + [0.0] * 30:
            cmd.append(demo.plus_ease(state, target, demo.PLUS_LIP_ACCEL, demo.PLUS_LIP_RATE))
        rate = np.diff(np.r_[0.0, cmd])
        self.assertLessEqual(np.abs(np.diff(np.r_[0.0, rate])).max(), demo.PLUS_LIP_ACCEL + 1e-12)
        self.assertLessEqual(np.abs(rate).max(), demo.PLUS_LIP_RATE + 1e-12)
        self.assertAlmostEqual(cmd[29], -0.005, places=9)
        self.assertAlmostEqual(cmd[-1], 0.0, places=9)
        self.assertGreaterEqual(min(cmd), -0.005 - demo.PLUS_LIP_ACCEL)

    def test_pour_beans_plus_landing(self):
        from scripts.g1d import scripted_demo as demo

        mouth = np.array([0.1, -0.3, 0.9])
        pts = mouth + np.array([[0.0, 0.0, 0.05], [0.01, 0.0, 0.05], [0.0, 0.0, 0.05], [0.0, 0.0, -0.01]])
        vel = np.array([[0.5, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3], [0.4, 0.0, -1.0]])
        off = demo.plus_landing(pts, vel, mouth)
        fall = np.sqrt(2 * 0.05 / demo.GRAVITY)  # from rest at 5 cm
        np.testing.assert_allclose(off[0], [0.5 * fall, 0.0])
        np.testing.assert_allclose(off[1], [0.01, -0.2 * fall])
        np.testing.assert_allclose(off[2], [0.0, 0.0], atol=1e-12)  # thrown straight up: back down on the spot
        np.testing.assert_allclose(off[3], [0.0, 0.0], atol=1e-12)  # falling under the plane: where it is

    def test_pour_beans_plus_due(self):
        from scripts.g1d import scripted_demo as demo

        cavity = {"axis": [0.0, 0.0], "z": [0.0, 0.08], "r": [0.03, 0.03]}
        local = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.07], [0.01, 0.0, 0.04], [0.0, 0.0, 0.1], [0.029, 0.0, 0.01]])
        # two actions ago: at rest, sliding up 1 cm an action, drifting 0.5 mm an action sideways, already out,
        # jittering across the lower wall (must not count as pouring over the rim)
        last = local - np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.02], [0.001, 0.0, 0.0], [0.0, 0.0, 0.02], [0.003, 0.0, 0.0]]
        )
        due = demo.plus_due(local, last, 2, cavity, 0.9)
        self.assertEqual(due.tolist(), [False, True, False, False, False])

    def test_layout_imaging_perturbs_only_its_cameras(self):
        from env.observation_manager import imaging

        rgb = np.full((8, 8, 3), 100, np.uint8)
        self.assertIs(imaging.perturb(rgb, None, "cam_head", 0), rgb)
        self.assertIs(imaging.perturb(rgb, {"brightness": 2.0, "cameras": ["cam_head"]}, "cam_left_wrist", 0), rgb)
        self.assertEqual(int(imaging.perturb(rgb, {"brightness": 1.5}, "cam_head", 0)[0, 0, 0]), 150)
        noisy = {"noise": 0.05, "seed": 3}
        first, again = (imaging.perturb(rgb, noisy, "cam_head", 4) for _ in range(2))
        np.testing.assert_array_equal(first, again)  # the same frame of the same camera: the same noise
        self.assertFalse(np.array_equal(first, imaging.perturb(rgb, noisy, "cam_head", 5)))
        gray = imaging.perturb(np.array([[[200, 100, 0]]], np.uint8), {"saturation": 0.0}, "cam_head", 0)
        self.assertEqual(len(set(gray.reshape(-1).tolist())), 1)

    def test_pour_beans_plus_layouts(self):
        from scipy.spatial.transform import Rotation

        from scripts.datagen import pour_beans_plus_layouts as layouts

        cup = {"category": "plastic_cup", "category_idx": 0, "label": "source_cup", "default_pos": [0, 0, 0]}
        cup |= {"default_ori": [1, 0, 0, 0], "physics": {"mass": 0.03, "friction": 0.5, "type": "rigid"}}
        fluid = {"label": "beans", "default_pos": [0, 0, 0], "default_ori": [1, 0, 0, 0], "visual": {"scale": [2] * 3}}
        fluid |= {"physics": {"type": "fluid"}, "scale": [0.5] * 3}
        room = {"default": "real_scene1", "default_pos": [0.0, -0.3, 0.0], "default_ori": [1.0, 0.0, 0.0, 0.0]}
        template = {"Rigid": {"plastic_cup": [cup]}, "Fluid": {"wuliangye": [fluid]}, "Room": room}
        with tempfile.TemporaryDirectory() as tmp:
            assets = Path(tmp)
            for scene in layouts.SCENES:
                path = assets / layouts.TEMPLATE.format(scene=scene)
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(template))
            rng = np.random.default_rng(0)
            n = 4000
            radius, angle = 0.035 * np.sqrt(rng.random(n)), rng.uniform(0, 2 * np.pi, n)
            pose = np.c_[radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.004, 0.17, n)]
            pose = np.c_[pose, Rotation.random(n, random_state=0).as_quat()]
            corners = ((-3.5e-3, 3.5e-3), (-2.4e-3, 2.4e-3), (-2.3e-3, 2.3e-3))
            hull = np.array([[x, y, z] for x in corners[0] for y in corners[1] for z in corners[2]])
            for kind in layouts.BEANS:
                path = assets / layouts.BEAN_PACK.format(kind=kind)
                path.parent.mkdir(parents=True)
                np.savez(path, pose=pose, scale=np.ones((n, 3)), mass=np.full(n, 1e-4), hull=hull, volume=8.92e-8,
                         packing=0.55)  # fmt: skip
            pool = json.loads(layouts.POOL.read_text(encoding="utf-8"))
            if not pool.get("split"):
                pool["split"] = {"train": sorted(pool["cups"])[::2], "eval": sorted(pool["cups"])[1::2]}
            cups = layouts.Cups(pool)
            gen = layouts.Generator(assets, cups, json.loads(layouts.INSTRUCTIONS.read_text(encoding="utf-8")))
            seen = {"train": set(), "eval": set()}
            exam = layouts.sets("exam", 2, cups, sorted(cups.train)[:3])
            for i, (name, levels, pair) in enumerate(layouts.sets("bench", 0)[::7] + layouts.sets("train", 30) + exam):
                split = "train" if name in ("train", "exam_0", "exam_1") else "eval"
                scene, lay, text = gen.layout(f"test/{i}", levels, split, pair)
                g = lay["Generalization"]
                if name.startswith("exam_"):
                    self.assertEqual(g["hand"], ("left", "right")[int(name[-1])])
                    self.assertEqual(g["beans"]["kind"], pair[4])
                self.assertEqual(g["levels"], levels)
                self.assertEqual(scene == layouts.HELD_SCENE, levels["visual"] == 3)
                ids = {g["cups"]["source_cup"], g["cups"]["target_cup"], *g["cups"]["distractors"]}
                seen[split] |= ids
                if split == "train" or levels["object"] < 3:
                    self.assertTrue(ids <= set(cups.train), ids)
                self.assertNotIn("<", text["zh"] + text["en"])
                self.assertEqual(levels["language"] == 3, g["template"][0] == "heldout")
                self.assertEqual(levels["language"] >= 2, bool(g["cups"]["distractors"]))
                src = next(e for es in lay["Rigid"].values() for e in es if e["label"] == "source_cup")
                beans = lay["Fluid"]["wuliangye"][0]
                self.assertEqual((beans["physics"]["type"], beans["bean"]["kind"]), ("granular", g["beans"]["kind"]))
                self.assertTrue(levels["object"] or pair or g["beans"]["kind"] == "red")
                self.assertIn(beans["bean"]["zh"], text["zh"])
                physics = beans["physics"]
                pack = gen.pack(beans["bean"]["kind"])
                _, centres = layouts.cup_profile.pack_fill(
                    physics["cavity"], physics["cavity_scale"], pack["pose"][:, :3],
                    layouts.cup_profile.grain_half(pack), yaw=physics["yaw"], count=physics["count"],
                )  # fmt: skip
                w, x, y, z = beans["default_ori"]
                points = Rotation.from_quat([x, y, z, w]).apply(centres) + beans["default_pos"]
                inside = layouts.cup_profile.in_cavity(
                    points, src["default_pos"], src["default_ori"], src["cavity"], src["scale"][0]
                )
                self.assertTrue(inside.all() and len(points) == g["beans"]["count"])
                for cup in (src, next(e for es in lay["Rigid"].values() for e in es if e["label"] == "target_cup")):
                    out = 1 if cup["default_pos"][0] > 0 else -1
                    g = layouts.demo.role_grip(cup.get("grasp"), cup["label"].split("_")[0])
                    prof, s = cup["cavity"], cup["scale"][0]
                    if g.get("mode") == "handle":
                        local, up = layouts.demo.handle_point(prof, s, g["handle"]), 0.0
                    else:
                        local = np.multiply([*prof["axis"], prof["bottom"]], s)
                        up = layouts.demo.plus_grip(prof, s, g.get("rim_down"), g.get("pitch"))[0]
                    rot = Rotation.from_quat(np.roll(cup["default_ori"], -1))
                    grasp = np.add(cup["default_pos"], rot.apply(local)) + [0, 0, up]
                    shoulder = np.add(layouts.SHOULDER, [0, 0, gen.lifts[scene]]) * [out, 1, 1]
                    self.assertLessEqual(np.linalg.norm(grasp - shoulder), layouts.REACH_MAX + 1e-3)
            self.assertFalse(seen["train"] & set(cups.eval))

            # narrow: only the cups' places and the clutter vary; one scene, the rest as built / L0
            hands, places = set(), set()
            for i, (name, levels, pair) in enumerate(layouts.sets("narrow", 12)):
                scene, lay, text = gen.layout(
                    f"narrow/{i}", levels, "train", pair, scene="g1d_shenzhen_lab", fixed=layouts.NARROW_FIXED
                )
                g = lay["Generalization"]
                self.assertEqual((scene, g["scene"]), ("g1d_shenzhen_lab", scene))
                self.assertEqual(g["fixed"], list(layouts.NARROW_FIXED))
                self.assertEqual((g["cups"]["source_cup"], g["cups"]["target_cup"]), (layouts.TWIN, layouts.TWIN))
                self.assertEqual((g["scale"]["source_cup"], g["scale"]["target_cup"]), (1.0, 1.0))
                self.assertEqual(g["beans"]["kind"], "red")
                self.assertEqual(set(g["base"].values()), {0.0})
                self.assertEqual((g["materials"], g["lights"], g["cups"]["distractors"]), ({}, [], []))
                self.assertTrue(2 <= len(g["clutter"]) <= 4 or not gen.clutter_train)
                self.assertNotIn("Light", lay)
                self.assertNotIn("Camera", lay)
                self.assertNotIn("Imaging", lay)
                self.assertEqual(g["template"], ["train", 0])
                self.assertIsNone(lay["Room"]["appearance_choice"]["table_top_z"])
                hands.add(g["hand"])
                places.add(tuple(e["default_pos"][0] for e in lay["Rigid"]["plastic_cup"]))
            self.assertEqual(hands, {"left", "right"})
            self.assertEqual(len(places), 12)

            # recipe: per-stratum combos, cup pools, scale ranges and scenes; training content only
            small = [c for c in cups.train if cups.fits(c, 0.9, True)][:3]
            recipe = {"strata": [
                {"name": "sparse", "target": 2, "candidates": 4, "combos": [[1, 0, 0, 0, 0], [0, 0, 0, 2, 0]]},
                {"name": "precision", "target": 1, "candidates": 3, "cups": {"target": small},
                 "scale": {"target": [0.85, 0.95]}, "scenes": ["g1d_shenzhen_lab"], "clutter": [0, 1]},
            ]}  # fmt: skip
            rows = layouts.recipe_sets(recipe, cups, 7)
            self.assertEqual([r[:2] for r in rows], [(0, "sparse")] * 4 + [(1, "precision")] * 3)
            self.assertEqual([(r[2]["spatial"], r[2]["camera"]) for r in rows[:4]], [(1, 0), (0, 2)] * 2)
            for k, name, levels, pair, scene, clutter in rows[4:]:
                self.assertEqual((levels["object"], scene, pair[1] in small), (2, "g1d_shenzhen_lab", True))
                self.assertTrue(0.85 <= pair[3] <= 0.95)
                lay = gen.layout(f"recipe/{k}", levels, "train", pair, scene=scene, clutter=clutter)[1]
                g = lay["Generalization"]
                self.assertEqual((g["cups"]["target_cup"], g["scale"]["target_cup"]), (pair[1], round(pair[3], 3)))
                self.assertLessEqual(len(g["clutter"]), 1)
            same = {"strata": [{"name": "same", "target": 1, "candidates": 3, "same": True}]}
            for k, name, levels, pair, scene, clutter in layouts.recipe_sets(same, cups, 3):
                self.assertEqual((pair[0], pair[2]), (pair[1], pair[3]))
                g = gen.layout(f"same/{k}", levels, "train", pair, scene=scene)[1]["Generalization"]
                self.assertTrue(g["by_side"])  # look-alike cups are named by their sides
            for bad in ({"combos": [[3, 0, 0, 0, 0]]}, {"scenes": [layouts.HELD_SCENE]}, {"clutter": [5, 6]},
                        {"cups": {"source": cups.eval[:1]}}, {"scale": {"target": [0.75, 0.85]}},
                        {"same": True, "scale": {"target": [0.9, 0.95]}}):  # fmt: skip
                with self.assertRaises(SystemExit):
                    layouts.recipe_sets({"strata": [{"name": "bad", "target": 1, "candidates": 1, **bad}]}, cups, 0)
        from scripts.datagen import pour_beans_plus_bench as bench

        one, three = layouts.sets("bench", 0), layouts.sets("bench", 0, reps=3)
        self.assertEqual((len(one), three[: len(one)], three[len(one) :]), (bench.PER_REP, one, one + one))
        self.assertEqual({bench.group_of(s) for s, _, _ in one}, set(bench.GROUPS))
        self.assertEqual(sum(bench.target(s) for s in {s for s, _, _ in one}), bench.PER_REP)

    def test_pour_beans_plus_score(self):
        from scripts.datagen import pour_beans_plus_bench as bench, pour_beans_plus_failures as failures

        lay = lambda **lv: {"set": "bench", "levels": dict(dict.fromkeys(bench.DIMS, 0), **lv)}  # noqa: E731
        job = lambda cfg, seed, scene, lays: dict(env_cfg=cfg, scene=scene, seed=seed, layouts=lays)  # noqa: E731
        manifest = {"groups": {
            "G0": {"seed": 1320, "jobs": [job("g1d_beans_rigid_collect", 1320, "g1d", [lay(), lay()])]},
            "diag_spatial": {"seed": 1324, "jobs": [job("g1d_beans_rigid_collect", 1324, "g1d", [lay(spatial=1),
                                                                                              lay(spatial=3)])]},
        }}  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "owam-bp-plus-20261110"
            ok, failed = dict(accepted=True, task_success=True), dict(accepted=False, task_success=True)
            outcomes = {(1320, 0, 0): ok, (1320, 0, 1): dict(reason="unstable"), (1324, 0, 0): failed, (1324, 0, 1): ok}
            for (seed, rep, i), r in outcomes.items():
                f = root / "g1d_beans_rigid_collect" / str(seed) / f"rep{rep}" / f"episode_{i:06d}.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(r))
            res = bench.score(manifest, "owam-bp-plus-20261110", [root], repeats=1, expert=1.0)
            g0, diag = res["runs"]
            self.assertEqual((g0["subset"], g0["seed"], g0["episodes"], g0["accepted"]), ("G0", 1320, 1, 1))
            self.assertEqual((g0["rejected_at_start"], g0["level"]), (1, "G0 · 办公室"))
            self.assertEqual((diag["subset"], diag["success_rate"]), ("diag_spatial", 0.5))
            self.assertEqual(set(diag["by_level"]), {"L1", "L3"})
            self.assertIn("计分子集（G0、G1、G2、G3）严格成功 1/1", res["summary"])  # not the diagnostic
            earlier = Path(tmp) / "g0.json"
            earlier.write_text(json.dumps({"runs": [{**g0, "accepted": 0, "episodes": 2, "subset": None}]}))
            again = bench.score(manifest, "x", [root], [earlier], repeats=1)
            self.assertEqual((again["runs"][0]["episodes"], again["runs"][0]["subset"]), (2, "G0"))  # kept, labelled
            self.assertNotIn("layouts", again)
            full = bench.score(manifest, "x", [root], repeats=1, episodes=[root], expert_episodes=[root])
            self.assertEqual(len(full["layouts"]), 4)
            self.assertEqual(full["layouts"][3]["levels"]["spatial"], 3)
            self.assertEqual(
                (full["layouts"][2]["subset"], full["layouts"][2]["expert"]["accepted"]), ("diag_spatial", 0)
            )
            self.assertNotIn("expert", full["layouts"][1])  # rejected at start
            self.assertEqual(len(full["episodes"]), 3)
            self.assertEqual(full["episode_records"], dict(recorded=3, episodes=3, expert=3))
            for name in ("owam-bp-plus-20261110", "owam-bp-plus-g1-20261108", "owam-bp-narrow-d-scene-20261108"):
                self.assertTrue(failures.EVAL_ID.match(name), name)
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(failures.layouts(path)["g1d_beans_rigid_collect", 1324][0], "diag_spatial")

    def test_pour_beans_plus_recipe(self):
        from scripts.datagen import pour_beans_plus_layouts as layouts, pour_beans_plus_recipe as recipe

        pool = json.loads(layouts.POOL.read_text(encoding="utf-8"))
        if not pool.get("split"):
            pool["split"] = {"train": sorted(pool["cups"])[::2], "eval": sorted(pool["cups"])[1::2]}
        cups = layouts.Cups(pool)
        a, b = [c for c in cups.train if c != layouts.TWIN][:2]
        zero = dict.fromkeys(layouts.DIMS, 0)
        train = [{"levels": zero, "cups": {"source_cup": a, "target_cup": layouts.TWIN}}] * recipe.FEW
        train += [{"levels": dict(zero, camera=2), "cups": {"source_cup": b, "target_cup": layouts.TWIN}}] * recipe.FEW
        combos = recipe.unseen_combos(train)
        self.assertEqual((len(combos), (0,) * 5 in combos, (0, 0, 0, 2, 0) in combos), (3**5 - 2, False, False))
        factors = {"source_cup": {a: {"n": 10, "strict": 1}, b: {"n": 10, "strict": 9}}, "target_cup": {}}
        hard = recipe.hard_cups(factors, train, cups)
        self.assertIn(a, hard)  # passed 10% of the time
        self.assertNotIn(b, hard)  # passed 90%, and enough training episodes
        self.assertNotIn(layouts.TWIN, hard)
        self.assertTrue(set(hard) - {a})  # the training cups the data never had
        built, _ = recipe.build({"factors": factors}, train, cups, "r")
        self.assertEqual(sum(s["target"] for s in built["strata"]), 500)
        self.assertGreaterEqual(built["strata"][0]["target"] / 500, 0.3)  # rsi/AGENTS.md marginal exploration

    def test_pour_beans_plus_handle_places(self):
        import random

        from scipy.spatial.transform import Rotation

        from scripts.datagen import pour_beans_plus_layouts as layouts

        rng = random.Random(0)
        (x0, x1), (y0, y1) = layouts.HANDLE_ZONE
        for level in range(4):
            for _ in range(20):
                outs, places = (0.09, 0.0), None
                while places is None:  # a new base when the places are out of reach, as layout() does
                    base = layouts.draw_base(rng, level)
                    places = layouts.draw_places(
                        rng, level, 0.045, 0.04, (0.05, 0.06), 0.75, base, 0.035, outs, (True, False)
                    )
                side, xa, _ = places
                # the source's bar, its handle straight out, in the frame of the moved robot (reach())
                yaw = layouts.handle_yaw(rng, {"cavity": {"handle": 30.0}}, side, level, True, base["yaw"])
                self.assertAlmostEqual((yaw + 30.0 - base["yaw"]) % 360, 0.0 if side > 0 else 180.0)
                rot = Rotation.from_euler("z", -base["yaw"], degrees=True)
                axis = layouts.ROBOT_ROOT + rot.apply(np.r_[xa - layouts.ROBOT_ROOT, 0.0])[:2]
                bar = axis - [base["lateral"], base["distance"]] + [side * outs[0], 0.0]
                self.assertTrue(x0 - 1e-9 <= side * bar[0] <= x1 + 1e-9 and y0 - 1e-9 <= bar[1] <= y1 + 1e-9, bar)

    def test_registry_entries_are_complete(self):
        registry = json.loads(Path(__file__).with_name("registry.json").read_text())
        required = {
            "name",
            "status",
            "task",
            "env_cfg",
            "episodes",
            "format",
            "acceptance",
            "code",
            "layouts",
            "location",
        }
        names = [d["name"] for d in registry["datasets"]]
        self.assertEqual(len(names), len(set(names)))
        for dataset in registry["datasets"]:
            self.assertLessEqual(required, set(dataset), dataset["name"])
            self.assertIn(dataset["status"], ("released", "superseded", "incomplete"))


if __name__ == "__main__":
    unittest.main()
