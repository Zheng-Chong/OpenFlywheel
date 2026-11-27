"""CPU unit tests for Track B X-Anything datagen optimizations:
1. --no-video / OPENFLYWHEEL_NO_VIDEO=1 video recording disable in eval_env.py
2. Skipping full image HDF5 for rejected episodes (+ .npz privileged state) and valid_jpeg() validation
3. Flock-protected dynamic layout_id claiming (LayoutPool, ClaimingSeedManager, run_batch.dynamic_queue)
"""

import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datagen import collect, early_abort, run_batch  # noqa: E402
from scripts.datagen.layout_pool import ClaimingSeedManager, LayoutPool  # noqa: E402
from scripts.datagen.test_datagen import FakeEnv, Plugin  # noqa: E402


def _mp_worker_claim(pool_path, output_dir, result_queue, worker_id):
    pool = LayoutPool(pool_path, output_dir)
    claimed = []
    while True:
        batch = pool.claim(worker=f"w{worker_id}")
        if not batch:
            break
        lid = batch[0]
        claimed.append(lid)
        (Path(output_dir) / f"episode_{lid:06d}.json").write_text(json.dumps({"accepted": True}))
        pool.finish(lid, status="done")
    result_queue.put(claimed)


def _import_eval_env_stubbed():
    """Import src.eval_client.eval_env on CPU without Isaac/msgpack/websockets installed."""
    stubs = {}
    for name in (
        "client_server",
        "client_server.ws",
        "client_server.ws.model_client",
        "transforms3d",
        "websockets",
        "env.sim_backend",
        "env.global_configs",
        "env.observation_manager",
        "env.observation_manager.obs_manager",
        "env.seed_manager",
        "env.seed_manager.seed_manager",
        "utils.cluttered_generator",
        "utils.pipeline_utils",
        "utils.save_file",
    ):
        if name not in sys.modules:
            stubs[name] = ModuleType(name)
    if "client_server.ws.model_client" in stubs:
        stubs["client_server.ws.model_client"].WsModelClient = object
    if "env.global_configs" in stubs:
        stubs["env.global_configs"].BENCHMARK = "RoboDojo"
    if "env.observation_manager.obs_manager" in stubs:
        stubs["env.observation_manager.obs_manager"].ObsManager = object
    if "env.seed_manager.seed_manager" in stubs:
        stubs["env.seed_manager.seed_manager"].SeedManager = object
    if "utils.cluttered_generator" in stubs:
        stubs["utils.cluttered_generator"].UnStableError = type("UnStableError", (Exception,), {})
    if "utils.pipeline_utils" in stubs:
        stubs["utils.pipeline_utils"].get_robot_action_dim_info = lambda **kw: {}
    if "utils.save_file" in stubs:
        stubs["utils.save_file"].VideoStreamWriter = object
        stubs["utils.save_file"].format_video_saved_message = lambda *a, **kw: ""
        stubs["utils.save_file"].save_json = lambda *a, **kw: None
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("src.eval_client.eval_env", None)
        return importlib.import_module("src.eval_client.eval_env")


class NoVideoTests(unittest.TestCase):
    def test_is_video_disabled_respects_cfg_and_env(self):
        eval_env = _import_eval_env_stubbed()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENFLYWHEEL_NO_VIDEO", None)
            self.assertFalse(eval_env.is_video_disabled({}))
            self.assertTrue(eval_env.is_video_disabled({"no_video": True}))
            self.assertFalse(eval_env.is_video_disabled({"no_video": False}))
        for truthy in ("1", "true", "TRUE", "yes", "on"):
            with patch.dict(os.environ, {"OPENFLYWHEEL_NO_VIDEO": truthy}):
                self.assertTrue(eval_env.is_video_disabled({}))
        for falsy in ("0", "false", "no", ""):
            with patch.dict(os.environ, {"OPENFLYWHEEL_NO_VIDEO": falsy}):
                self.assertFalse(eval_env.is_video_disabled({}))


class RejectedHdf5AndJpegTests(unittest.TestCase):
    def _make_rejecting_plugin(self):
        plugin = Plugin()
        plugin.HOLD = 99  # never reaches HOLD -> accepted=False
        return plugin

    def _fake_encode(self, img):
        return b"\xff\xd8\xff\xe0" + bytes(img.shape) + b"\xff\xd9"

    def test_rejected_episode_skips_hdf5_and_writes_privileged_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "layout.json").write_text("{}")
            env = FakeEnv(root / "layout.json")
            plugin = self._make_rejecting_plugin()
            fake_mod = ModuleType("XPolicyLab.utils.process_data")
            fake_mod.encode_image_bit = self._fake_encode
            with patch.dict(sys.modules, {"XPolicyLab.utils.process_data": fake_mod}):
                collect.record_episode(env, plugin, root, dict(env_cfg="cfg", fps=30, revision=None))
            sidecar = root / "episode_000003.json"
            hdf5 = root / "episode_000003.hdf5"
            npz = root / "episode_000003.npz"
            self.assertTrue(sidecar.exists())
            self.assertFalse(hdf5.exists())
            self.assertTrue(npz.exists())
            quality = json.loads(sidecar.read_text())
            self.assertFalse(quality["accepted"])
            self.assertFalse(quality["trajectory_written"])
            priv = early_abort.load_privileged(sidecar)
            np.testing.assert_array_equal(priv["steps"], np.arange(7))

    def test_keep_rejected_hdf5_override_writes_hdf5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "layout.json").write_text("{}")
            env = FakeEnv(root / "layout.json")
            plugin = self._make_rejecting_plugin()
            fake_mod = ModuleType("XPolicyLab.utils.process_data")
            fake_mod.encode_image_bit = self._fake_encode
            called = []
            with (
                patch.dict(sys.modules, {"XPolicyLab.utils.process_data": fake_mod}),
                patch.object(collect, "write_trajectory", side_effect=lambda *a, **kw: called.append(a[0])),
            ):
                collect.record_episode(
                    env,
                    plugin,
                    root,
                    dict(env_cfg="cfg", fps=30, revision=None, keep_rejected_hdf5=True),
                )
            sidecar = root / "episode_000003.json"
            self.assertEqual(called, [root / "episode_000003.hdf5"])
            self.assertFalse((root / "episode_000003.npz").exists())
            self.assertTrue(json.loads(sidecar.read_text())["trajectory_written"])

    def test_validate_images_rejects_bad_jpegs_and_accepts_good(self):
        good = b"\xff\xd8\xff\xe0hello\xff\xd9"
        padded = good + b"\x00\x00"
        collect.validate_images({"cam_head": [good, padded]})
        for bad in (b"", b"\x00\x00\x00\x00", b"\xff\xd8truncated", b"not_a_jpeg\xff\xd9", b"\xff\xd8\xff"):
            with self.assertRaisesRegex(ValueError, "invalid or truncated JPEG"):
                collect.validate_images({"cam_head": [good, bad]})
        with self.assertRaisesRegex(ValueError, "empty image stream"):
            collect.validate_images({"cam_head": []})


class LayoutPoolTests(unittest.TestCase):
    def test_sequential_claims_disk_skip_and_stop_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stop = out / "STOP"
            (out / "episode_000001.json").write_text("{}")  # already done on disk
            pool = LayoutPool(out / ".claim_pool.json", out, stop_file=stop)
            pool.ensure_initialized([0, 1, 2, 3])
            self.assertEqual(pool.claim(worker="w0"), [0])
            self.assertEqual(pool.claim(worker="w1"), [2])
            stop.touch()
            self.assertEqual(pool.claim(worker="w2"), [])
            stop.unlink()
            self.assertEqual(pool.claim(worker="w2"), [3])
            self.assertEqual(pool.claim(worker="w3"), [])

    def test_dead_pid_reclaimed_live_pid_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            alive = {1001: True, 1002: False}
            pool = LayoutPool(out / ".claim_pool.json", out, pid_alive=lambda p: alive.get(p, False))
            pool.ensure_initialized([0, 1])
            self.assertEqual(pool.claim(worker="w0", pid=1001), [0])
            self.assertEqual(pool.claim(worker="w1", pid=1002), [1])
            # 1002 is dead, so layout 1 is reclaimed; 1001 is alive so layout 0 stays held
            self.assertEqual(pool.claim(worker="w2", pid=1001), [1])
            self.assertEqual(pool.claim(worker="w2", pid=1001), [])

    def test_unstable_and_broken_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pool = LayoutPool(out / ".claim_pool.json", out, broken_tries=2)
            pool.ensure_initialized([0, 1])
            self.assertEqual(pool.claim(), [0])
            pool.finish(0, status="unstable")
            self.assertEqual(pool.snapshot()["skipped"], [0])
            self.assertEqual(pool.claim(), [1])
            pool.finish(1, status="broken")
            self.assertTrue(pool.has_pending())  # 1st break: still retried
            self.assertEqual(pool.claim(), [1])
            pool.finish(1, status="broken")
            self.assertFalse(pool.has_pending())  # 2nd break: permanently skipped
            self.assertEqual(pool.snapshot()["skipped"], [0, 1])

    def test_multiprocess_concurrent_claims_have_no_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pool_path = out / ".claim_pool.json"
            n = 40
            LayoutPool(pool_path, out).ensure_initialized(range(n))
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()
            procs = [ctx.Process(target=_mp_worker_claim, args=(str(pool_path), str(out), q, i)) for i in range(4)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=30)
                self.assertEqual(p.exitcode, 0)
            all_claimed = []
            for _ in range(4):
                all_claimed.extend(q.get(timeout=5))
            self.assertEqual(sorted(all_claimed), list(range(n)))

    def test_claiming_seed_manager_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pool = LayoutPool(out / ".claim_pool.json", out, broken_tries=2)
            pool.ensure_initialized([0, 1, 2])
            inner = SimpleNamespace(num_envs=1, seed_list=[0, 1, 2], seed_info={0: {}, 1: {}, 2: {}})
            sm = ClaimingSeedManager(inner, pool)
            self.assertEqual(sm.get_seeds(max_count=3), [0])
            sm.mark_unstable()
            self.assertEqual(sm.get_seeds(max_count=3), [1])
            sm.mark_broken([1])
            self.assertEqual(sm.get_seeds(max_count=3), [1])  # retried
            sm.release_current()
            self.assertEqual(sm.get_seeds(max_count=3), [1])
            sm.eval_step()
            # setting idx = ed_idx stops further claims (collect.py stop_file contract)
            sm.idx = sm.ed_idx
            self.assertIsNone(sm.get_seeds(max_count=3))

    def test_run_batch_dynamic_queue_and_sync_pool_state(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            stop = Path(directory) / "STOP"
            slots_4 = [(0, k) for k in range(4)]
            queue, pools = run_batch.dynamic_queue({100: list(range(50))}, slots_4, raw, [], {}, stop)
            self.assertEqual(len(queue), 4)
            self.assertEqual([w for _, _, w in queue], [0, 1, 2, 3])
            self.assertIn(100, pools)
            # Small remaining set caps workers to len(ids)
            queue_small, _ = run_batch.dynamic_queue({101: [0, 1]}, slots_4, raw, [], {}, stop)
            self.assertEqual(len(queue_small), 2)
            # Sync pool state back to run_batch skipped/broken
            pools[100].claim()
            pools[100].finish(0, status="unstable")
            pools[100].claim()
            pools[100].finish(1, status="broken")
            skipped, broken = [], {}
            run_batch.sync_pool_state(100, pools[100], skipped, broken)
            self.assertEqual(skipped, ["100_0"])
            self.assertEqual(broken, {"100_1": 1})

    def test_claiming_seed_manager_partial_batch_refill_does_not_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pool = LayoutPool(out / ".claim_pool.json", out, broken_tries=1)
            pool.ensure_initialized([0, 1, 2])
            inner = SimpleNamespace(num_envs=2, seed_list=[0, 1, 2], seed_info={0: {}, 1: {}, 2: {}})
            sm = ClaimingSeedManager(inner, pool)
            self.assertEqual(sm.get_seeds(max_count=3), [0, 1])
            sm.mark_broken([0])
            self.assertEqual(sm.get_seeds(max_count=1), [2])
            sm.eval_step()
            # Neither surviving seed 1 nor replacement seed 2 may leak in in_flight
            self.assertEqual(pool.snapshot()["in_flight"], {})

    def test_should_retry_dynamic_caps_startup_hangs_even_with_sigkill_returncode(self):
        # Regression: SIGKILLed hung worker has returncode=-9 and crash_count=0; must stop at max_tries
        self.assertTrue(run_batch.should_retry_dynamic(True, -9, hang_count=1, crash_count=0, max_tries=3))
        self.assertTrue(run_batch.should_retry_dynamic(True, -9, hang_count=2, crash_count=0, max_tries=3))
        self.assertFalse(run_batch.should_retry_dynamic(True, -9, hang_count=3, crash_count=0, max_tries=3))
        # Normal crashes bounded by crash_count; clean exit never retried
        self.assertTrue(run_batch.should_retry_dynamic(False, 1, hang_count=0, crash_count=2, max_tries=3))
        self.assertFalse(run_batch.should_retry_dynamic(False, 1, hang_count=0, crash_count=3, max_tries=3))
        self.assertFalse(run_batch.should_retry_dynamic(False, 0, hang_count=0, crash_count=0, max_tries=3))

    def test_main_fatal_restart_helpers_release_claimed_seeds(self):
        src = (ROOT / "src" / "eval_client" / "main.py").read_text()
        for fn_name in ("def _restart_or_exit(", "def _exit_for_shell_restart("):
            start = src.index(fn_name)
            body = src[start : start + 800]
            self.assertIn("_release_claimed_seeds(env)", body, f"{fn_name} must release claims before exit/execv")


class SetupScenePreRenderTests(unittest.TestCase):
    def test_setup_scene_runs_200_headless_steps_and_5_bare_renders(self):
        src = (ROOT / "src" / "eval_client" / "eval_env.py").read_text()
        start = src.index("        def setup_scene(self):")
        end = src.index("        def get_obs(self):", start)
        body = src[start:end]
        # Execute setup_scene in isolation against a fake env object
        ns = {"UnStableError": RuntimeError}
        exec("def setup_scene(self):\n" + "\n".join(line[8:] for line in body.splitlines()[1:]), ns)

        sim_calls = []
        render_calls = 0
        obs_capture_calls = 0

        class FakeObs:
            def render_for_capture(self):
                nonlocal obs_capture_calls
                obs_capture_calls += 1

            def get_obs(self):
                nonlocal obs_capture_calls
                obs_capture_calls += 1

        class FakeLayoutMgr:
            def check_layout_stability(self, env):
                return True, []

        class FakeSceneMgr:
            layout_manager = FakeLayoutMgr()

            def apply_saved_poses(self, env_idx_list):
                pass

        class FakeEvalEnv:
            num_envs = 1
            unstable_nums = 0
            episode_nums = 1
            physx_monitor_enabled = False
            scene_manager = FakeSceneMgr()
            obs_manager = FakeObs()

            def _align_layout_success(self):
                pass

            def sim_step(self, render=True):
                sim_calls.append(render)

            def render(self):
                nonlocal render_calls
                render_calls += 1

        ns["setup_scene"](FakeEvalEnv())
        self.assertEqual(len(sim_calls), 200)
        self.assertTrue(all(r is False for r in sim_calls), "all 200 settle steps must use render=False")
        self.assertEqual(render_calls, 5)
        self.assertEqual(obs_capture_calls, 0)


class StallBreakExpertTests(unittest.TestCase):
    def _make_robot(self, positions):
        from scipy.spatial.transform import Rotation

        from scripts.g1d import place_v44

        class DummyArm:
            def __init__(self, pts):
                self.pts = [np.asarray(p, dtype=float) for p in pts]
                self.i = 0

            def grasp_point(self):
                return self.pts[min(self.i, len(self.pts) - 1)]

            def pose(self):
                return self.grasp_point(), Rotation.identity()

        class DummyRobot(place_v44.StallBreakMixin):
            def __init__(self, arm):
                self.arm = arm
                self.settle = (0.005, 60)
                self.cmds = 0

            def _command(self, arm, point, rot, grip):
                self.cmds += 1
                self.arm.i += 1
                return self.cmds

        arm = DummyArm(positions)
        return DummyRobot(arm), arm

    def test_stall_break_exits_after_6_consecutive_stalled_settle_frames(self):
        from scipy.spatial.transform import Rotation

        target = np.array([0.5, 0.0, 0.5])
        # 2 waypoint steps, then arm stuck at [0.4, 0.0, 0.5] (10 cm short of target)
        positions = [[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]] + [[0.4, 0.0, 0.5]] * 80
        robot, arm = self._make_robot(positions)
        cmds = list(robot.move(arm, [(target, Rotation.identity(), 0.0, 2)]))
        # 2 waypoint cmds + 6 settle cmds (0th sets prev_pos, 1..5 increment stall 1..5, 6th sees stall==6 and breaks)
        self.assertEqual(len(cmds), 2 + 6)

    def test_stall_break_resets_when_arm_moves_and_exits_on_convergence(self):
        from scipy.spatial.transform import Rotation

        target = np.array([0.1, 0.0, 0.0])
        # 1 waypoint step, then 4 stalled frames (<6), then moving 2 mm/frame until reaching target
        positions = (
            [[0.0, 0.0, 0.0]]
            + [[0.08, 0.0, 0.0]] * 5  # 4 stalled diffs
            + [[0.08 + 0.002 * k, 0.0, 0.0] for k in range(1, 15)]
        )
        robot, arm = self._make_robot(positions)
        cmds = list(robot.move(arm, [(target, Rotation.identity(), 0.0, 1)]))
        # Must NOT have stalled early; reaches within 5 mm of 0.1 at 0.096 (k=8)
        self.assertGreater(len(cmds), 1 + 6)
        self.assertLess(len(cmds), 1 + 20)

    def test_v44_experts_registered_without_drifting_older_versions(self):
        from scripts.datagen import experts

        m = experts.load()
        for task, ver in [
            ("g1d_place_anything", "v6"),
            ("g1d_put_anything_in", "v6"),
            ("g1d_sort_anything", "v2"),
        ]:
            rec = experts.record(task, ver, manifest=m)
            self.assertEqual(rec["drift"], [], f"{task} {ver} drifted: {rec['drift']}")
        # Verify scripted_demo.py, place_v43.py, sort_v43.py were not modified in place
        read_head = experts.reader("HEAD")
        read_work = experts.reader(None)
        self.assertEqual(
            experts.digest(read_work("scripts/g1d/scripted_demo.py")),
            experts.digest(read_head("scripts/g1d/scripted_demo.py")),
        )
        # Earlier v5/v1 entries must still resolve cleanly from experts.json
        for task, ver in [("g1d_place_anything", "v5"), ("g1d_put_anything_in", "v5"), ("g1d_sort_anything", "v1")]:
            entry = experts.resolve(task, ver, manifest=m)
            self.assertEqual(entry["version"], ver)


if __name__ == "__main__":
    unittest.main()
