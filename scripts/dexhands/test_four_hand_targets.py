"""Controller boundary checks independent of a simulator or device SDK."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.robot_manager.robot_class.g1d_dexhands import DexHandTargets

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model", ["g1d_sharpa_wave", "g1d_wuji_hand2_1", "g1d_wuji_hand2_2", "g1d_inspire_rh56dftp"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_active_command_boundaries_reject_invalid_data(model, side):
    directory = ROOT / "Assets/Robots" / model
    h = json.loads((directory / "model_contract.json").read_text())["hands"][side]
    limits = h.get("effective_active_limits", h["limits"])
    low = np.array([limits[n]["lower"] for n in h["joints"]])
    high = np.array([limits[n]["upper"] for n in h["joints"]])
    robot = SimpleNamespace(robot_file=str(directory), config_key=side, gripper_joints_name=h["joints"])
    for legal in [low, high, (low + high) / 2]:
        DexHandTargets.validate_hand_position(robot, legal)
    for bad in [low - 0.01, high + 0.01, [np.nan] * len(low), [np.inf] * len(low), low[:-1], np.tile(low, (2, 1))]:
        with pytest.raises(ValueError):
            DexHandTargets.validate_hand_position(robot, bad)


def test_native_pose_boundary_does_not_rotate_a_stationary_support():
    from scripts.dexhands.review_contract import native_pose_xyzw

    identity = np.array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(native_pose_xyzw(identity), [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    half = np.sqrt(0.5)
    quarter_turn = np.array([[1.0, 2.0, 3.0, half, 0.0, 0.0, half]])
    np.testing.assert_array_equal(native_pose_xyzw(quarter_turn), [[1.0, 2.0, 3.0, 0.0, 0.0, half, half]])
    with pytest.raises(ValueError):
        native_pose_xyzw(np.zeros((1, 6)))


def test_capture_clock_rejects_initial_stale_render_and_missing_camera():
    from scripts.dexhands.review_contract import camera_clock_ready

    assert camera_clock_ready([1.2, 1.2, 1.2], 1.2, 1 / 480)
    assert not camera_clock_ready([0.0, 1.2], 1.2, 1 / 480)
    assert not camera_clock_ready([None], 1.2, 1 / 480)
    assert not camera_clock_ready([float("nan")], 1.2, 1 / 480)
    assert not camera_clock_ready([], 1.2, 1 / 480)
