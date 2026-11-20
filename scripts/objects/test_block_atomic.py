import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.reward_manager import block_checks  # noqa: E402
from scripts.objects import block_atomic as ba, block_patterns as bp  # noqa: E402


def fake_entry(cat, idx, label, xy, yaw):
    return {"category": cat, "category_idx": idx, "label": label, "xy": xy, "yaw": yaw}


@pytest.mark.parametrize("task", ba.TASKS)
def test_layouts_keep_blocks_apart_and_distinct(task):
    rng = random.Random(task)
    for _ in range(200):
        rigid = ba.LAYOUTS[task](rng, fake_entry)
        entries = [e for es in rigid.values() for e in es]
        blocks = [e for e in entries if e["category"].startswith("wood_block_")]
        assert len({(e["category"], e["category_idx"]) for e in blocks}) == len(blocks)  # no two alike
        for a in entries:
            for b in entries:
                if a is b:
                    continue
                ra, rb = (math.hypot(*bp.SIZES[e["category"]][:2]) / 2 if e in blocks else 0.03 for e in (a, b))
                assert math.dist(a["xy"], b["xy"]) >= ra + rb, (task, a["label"], b["label"])
        for e in blocks:
            assert ba.SIDE_X[0] <= abs(e["xy"][0]) <= ba.SIDE_X[1]
            if bp.SIZES[e["category"]][2] < 0.02:
                assert ba.FLAT_Y[0] <= e["xy"][1] <= ba.FLAT_Y[1]
        labels = {e["label"]: e for e in entries}
        if task == "g1d_stack_block_on_block":
            assert np.sign(labels["top_block"]["xy"][0]) == np.sign(labels["base_block"]["xy"][0])
        if task == "g1d_line_up_blocks":
            assert labels["block_0"]["xy"][0] < 0 < labels["block_2"]["xy"][0]
        if task == "g1d_place_block_on_mark" and abs(labels["mark"]["xy"][0]) > ba.MARK_SIDE:
            assert np.sign(labels["mark"]["xy"][0]) == np.sign(labels["block"]["xy"][0])
        if task == "g1d_rotate_block_parallel":  # askew to begin with
            yaw = (labels["block"]["yaw"] + 90.0) % 180.0 - 90.0
            assert abs(yaw) > 20.0


def pose(xyz, yaw=0.0, tilt=0.0):
    return np.array(xyz, float), (
        Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("x", tilt, degrees=True)
    ).as_matrix()


def test_checks():
    cube, table = bp.SIZES["wood_block_cube"], 0.7
    on = pose([0.0, -0.42, table + 0.001 + 0.015])
    assert block_checks.placed_on(*on, cube, [0.01, -0.42], table, 0.02)
    assert not block_checks.placed_on(*on, cube, [0.03, -0.42], table, 0.02)
    assert not block_checks.placed_on(*pose([0.0, -0.42, table + 0.05]), cube, [0.0, -0.42], table, 0.02)  # held up
    base = np.array([0.15, -0.45, table + 0.015])
    assert block_checks.stacked_on(*pose(base + [0.005, 0, 0.03]), cube, base, cube, 0.015, 0.008)
    assert not block_checks.stacked_on(*pose(base + [0.02, 0, 0.03]), cube, base, cube, 0.015, 0.008)
    assert not block_checks.stacked_on(*pose(base + [0, 0, 0.03], tilt=30), cube, base, cube, 0.015, 0.008)
    assert block_checks.lifted([0, 0, table + 0.08], cube, table, 0.05)
    assert not block_checks.lifted([0, 0, table + 0.04], cube, table, 0.05)
    assert block_checks.yaw_aligned(pose([0, 0, 0], yaw=175)[1], 0.0, 180, 10)
    assert not block_checks.yaw_aligned(pose([0, 0, 0], yaw=30)[1], 0.0, 180, 10)
    row = [pose([x, -0.42, table + 0.015]) for x in (-0.045, 0.0, 0.045)]
    assert block_checks.in_row([p for p, _ in row], [r for _, r in row], (0.035, 0.07), 0.015)
    assert not block_checks.in_row([p for p, _ in row[::-1]], [r for _, r in row], (0.035, 0.07), 0.015)


def test_plugin_acceptance():
    from scripts.datagen.plugins import _block_atomic as plugin

    table, cube = 0.7, list(bp.SIZES["wood_block_cube"])
    meta = dict(sizes={"top_block": cube, "base_block": cube, "other_block": cube}, table_z=table,
                start={"top_block": [0.15, -0.40], "base_block": [0.15, -0.45], "other_block": [-0.15, -0.45]},
                blocks=["base_block", "other_block", "top_block"])  # fmt: skip
    base = np.array([0.15, -0.45, table + 0.015])
    poses = {
        "base_block": pose(base),
        "top_block": pose(base + [0.004, 0, 0.03]),
        "other_block": pose([-0.15, -0.45, table + 0.015]),
    }
    assert plugin.accepted("g1d_stack_block_on_block", poses, [1.0, 1.0], meta)
    assert not plugin.accepted("g1d_stack_block_on_block", poses, [1.0, 0.4], meta)  # a hand still closed
    pushed = dict(poses, base_block=pose(base + [0.03, 0, 0]), top_block=pose(base + [0.034, 0, 0.03]))
    assert not plugin.accepted("g1d_stack_block_on_block", pushed, [1.0, 1.0], meta)  # the base pushed away
