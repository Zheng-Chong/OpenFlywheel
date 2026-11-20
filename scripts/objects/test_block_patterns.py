import importlib
from pathlib import Path
import subprocess
import sys
import types

import block_patterns as bp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.reward_manager import pattern_match  # noqa: E402
from env.reward_manager.pattern_match import pattern_built, pattern_error  # noqa: E402


def built(pattern, offset=(0.3, -0.2), turn=0.0, jitter=None):
    """World poses of a pattern built exactly, moved by `offset` and turned `turn` deg; `jitter`: per-piece edits."""
    t = bp.targets(pattern)
    rot = Rotation.from_euler("z", turn, degrees=True)
    pos = rot.apply(t[:, :3]) + [*offset, 0.7]
    rots = [Rotation.from_euler("z", turn + yaw, degrees=True) for yaw in t[:, 3]]
    for i, (dpos, drot) in (jitter or {}).items():
        pos[i] += dpos
        rots[i] = drot * rots[i]
    return pos, np.array([r.as_matrix() for r in rots])


def check(pattern, pos, rots):
    periods = [bp.YAW_PERIOD[c] for c, _, _ in pattern["pieces"]]
    return pattern_built(pos, rots, bp.targets(pattern), periods, pattern["anchor"], 0.015, 0.01, 10, 15)


@pytest.mark.parametrize("pattern", bp.PATTERNS, ids=lambda p: p["name"])
def test_patterns_stand(pattern):
    assert bp.overlaps(pattern) == []
    assert all(bp.supported(pattern))
    assert 0 <= pattern["anchor"] < len(pattern["pieces"]) and bp.YAW_PERIOD[pattern["pieces"][pattern["anchor"]][0]]
    assert len(pattern["pieces"]) <= bp.MAX_PIECES


@pytest.mark.parametrize("pattern", bp.PATTERNS, ids=lambda p: p["name"])
def test_a_built_pattern_passes_anywhere_and_turned(pattern):
    for turn in (0.0, 37.0, 180.0, -95.0):
        pos, rots = built(pattern, turn=turn)
        assert np.allclose(pattern_error(pos, rots, bp.targets(pattern), [bp.YAW_PERIOD[c] for c, _, _ in
                           pattern["pieces"]], pattern["anchor"])[:2], 0, atol=1e-9)  # fmt: skip
        assert check(pattern, pos, rots)


def test_symmetric_blocks_may_turn():
    house = bp.PATTERNS[0]
    anchor, cube = house["anchor"], 0
    spin = {anchor: (np.zeros(3), Rotation.from_euler("z", 180, degrees=True)),  # the plank, end for end
            cube: (np.zeros(3), Rotation.from_euler("z", 90, degrees=True))}  # fmt: skip
    assert check(house, *built(house, jitter=spin))
    gate = next(p for p in bp.PATTERNS if p["name"] == "gate")
    assert check(gate, *built(gate, jitter={0: (np.zeros(3), Rotation.from_euler("z", 33, degrees=True))}))


def test_a_wrong_build_fails():
    house = bp.PATTERNS[0]
    still = Rotation.identity()
    assert not check(house, *built(house, jitter={3: (np.array([0.02, 0.0, 0.0]), still)}))  # roof off by 2 cm
    assert not check(house, *built(house, jitter={3: (np.array([0.0, 0.0, 0.02]), still)}))  # roof floating
    assert not check(house, *built(house, jitter={3: (np.zeros(3), Rotation.from_euler("z", 30, degrees=True))}))
    assert not check(house, *built(house, jitter={1: (np.zeros(3), Rotation.from_euler("x", 90, degrees=True))}))
    pos, rots = built(house)
    pos[[0, 2]] = pos[[2, 0]]  # cube where the plank goes and back: the pieces are not interchangeable
    assert not check(house, pos, rots)


def test_lookalike_pieces_may_swap_places():
    def check_groups(pattern, pos, rots):
        periods = [bp.YAW_PERIOD[c] for c, _, _ in pattern["pieces"]]
        args = (bp.targets(pattern), periods, pattern["anchor"], 0.015, 0.01, 10, 15)
        return pattern_built(pos, rots, *args, groups=bp.groups(pattern))

    ell = next(p for p in bp.PATTERNS if p["name"] == "ell")
    pos, rots = built(ell)
    pos[[1, 2]] = pos[[2, 1]]  # the two green cubes swapped: the same build (the L is not symmetric)
    assert check_groups(ell, pos, rots) and not check(ell, pos, rots)
    house = bp.PATTERNS[0]
    pos, rots = built(house)
    pos[[0, 2]] = pos[[2, 0]]
    assert not check_groups(house, pos, rots)  # a cube where the plank goes
    wall = next(p for p in bp.PATTERNS if p["name"] == "wall")
    assert bp.groups(wall)[:4] == [0, 1, 0, 1] and len(list(pattern_match.assignments(bp.groups(wall)))) == 24
    pos, rots = built(wall)
    pos[[0, 2]] = pos[[2, 0]]  # two red cubes
    assert check_groups(wall, pos, rots)
    pos[[0, 1]] = pos[[1, 0]]  # a red and a blue cube
    assert not check_groups(wall, pos, rots)


def test_sizes_match_the_built_blocks(monkeypatch):
    stub = types.ModuleType("pxr")
    for name in "Gf Sdf Usd UsdGeom UsdPhysics UsdShade Vt".split():
        setattr(stub, name, types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pxr", stub)
    for name in ("make_pitcher", "make_plastic_cups", "make_wood_blocks"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    wb = importlib.import_module("make_wood_blocks")
    for shape in wb.SHAPES:
        ext = np.ptp(wb.build(shape)["pts"], axis=0)
        assert np.allclose(ext, bp.SIZES[shape["cat"]], atol=5e-4), shape["cat"]
    assert set(bp.SIZES) == {s["cat"] for s in wb.SHAPES}


def test_build_asset_imports_without_numpy():
    """The Real2Sim platform runs on the stdlib only and imports build_asset for its scene table."""
    code = "import sys; sys.modules['numpy'] = None; sys.path.insert(0, 'scripts/g1d'); import build_asset"
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)
