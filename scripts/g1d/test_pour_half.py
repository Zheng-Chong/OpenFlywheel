"""Checks of the 倒半杯 task's data side without Isaac: the real-episode twin data, the layouts made from it, and the
datagen plugin's half-cup acceptance.

    python -m pytest -q scripts/g1d/test_pour_half.py
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datagen.plugins import g1d_pour_beans_half_cup as plugin  # noqa: E402
from scripts.objects import round_cup_shape  # noqa: E402

TASK_DIR = ROOT / "real2sim/tasks/g1d_pour_beans_half_cup"
GREY = dict(height=0.089, radius=0.031, bottom=0.004)
BLUE = dict(height=0.1013, radius=0.0327, radius_bottom=0.0236, bottom=0.003)


def test_real_episodes():
    body = json.loads((TASK_DIR / "real_episodes.json").read_text(encoding="utf-8"))
    rows = body["episodes"]
    assert body["at_stand"] == 37 and len(rows) == 34
    assert {r["target_object"] for r in rows} == {"Rigid/plastic_cup/00009", "Rigid/plastic_cup/00008"}
    for r in rows:
        assert r["source"][0] > r["target"][0] and len(r["start"]) == 14  # tall cup on the right; both arms
        assert "倒半杯" in r["instruction"]


def test_capacity():
    assert 1500 < plugin.capacity(GREY) < 1800 and 1500 < plugin.capacity(BLUE) < 1800
    assert plugin.capacity(BLUE) < plugin.capacity(dict(BLUE, radius_bottom=BLUE["radius"]))  # tapered holds less


def test_half_cup_acceptance():
    cap, upright, start_z = plugin.capacity(GREY), np.array([0.0, 0, 0.9, 1, 0, 0, 0]), np.array([0.9, 0.9])
    poses = [upright, upright]

    def ok(target, outside=0, poses=poses, total=3100):
        counts = dict(total=total, source=total - target - outside, target=target, outside=outside, overlap=0)
        return plugin.accepted(counts, GREY, poses, start_z)

    assert ok(round(0.5 * cap))
    assert not ok(round(0.2 * cap)) and not ok(round(0.8 * cap))  # too little, overfilled
    assert ok(round(0.5 * cap), outside=20) and not ok(round(0.5 * cap), outside=100)  # spill: 5% of what was poured
    tilted = np.r_[0, 0, 0.9, np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10)), 0, 0]  # 20 deg about x
    assert not ok(round(0.5 * cap), poses=[upright, tilted])
    assert not ok(round(0.5 * cap), poses=[upright, np.r_[0, 0, 0.95, 1, 0, 0, 0]])  # still held up, not put back


def test_fill_check():
    """The task check (func_parser.is_fluid_fill_in_range) and the plugin count the same beans with the same rule."""
    at, upright = np.array([0.0, 0.0, 0.9]), np.array([1.0, 0, 0, 0])
    inside = np.array([[0.0, 0.0, 0.9], [0.02, 0.0, 0.88], [0.0, 0.0, 0.9 + GREY["height"] / 2 + 0.001]])
    assert round_cup_shape.in_cavity(inside, at, upright, GREY).tolist() == [True, True, False]  # above the rim: out
    half = round(0.5 * round_cup_shape.capacity(GREY))
    assert round_cup_shape.fill_ok(half, 3100 - half - 40, 3100, GREY, (0.3, 0.75), 0.05)  # 40 of ~880 spilled
    assert not round_cup_shape.fill_ok(half, 3100 - half - 100, 3100, GREY, (0.3, 0.75), 0.05)  # 100 of ~940


def test_layouts():
    with tempfile.TemporaryDirectory() as tmp:
        run = [sys.executable, str(ROOT / "scripts/g1d/prepare_pour_half_layouts.py")]
        subprocess.run([*run, "l0", "--assets", tmp, "--seed", "950"], check=True, capture_output=True)
        subprocess.run(
            [*run, "train", "--assets", tmp, "--seed", "600", "--count", "40"], check=True, capture_output=True
        )
        folder = Path(tmp, "Eval_Layout/RoboDojo/g1d_lab")
        l0 = sorted((folder / "950").glob("g1d_pour_beans_half_cup_*.json"))
        rows = json.loads((TASK_DIR / "real_episodes.json").read_text(encoding="utf-8"))["episodes"]
        assert len(l0) == len(rows) and not (folder / "950" / "split.json").exists()  # an eval set
        first = json.loads((folder / "950" / "g1d_pour_beans_half_cup_0.json").read_text(encoding="utf-8"))
        assert first["Instruction"] == rows[0]["instruction"] and first["Robot"]["init_joint_pos"] == rows[0]["start"]
        assert first["Rigid"]["measuring_cup"][0]["default_pos"][:2] == rows[0]["source"][:2]
        assert first["Room"]["table_height"] == 0.83
        beans = first["Fluid"]["wuliangye"][0]["physics"]
        assert beans["spawn"] == {"region": "box", "packing": "hex"} and beans["particle_material"]["friction"] == 0.3
        train = sorted((folder / "600").glob("g1d_pour_beans_half_cup_*.json"))
        assert len(train) == 40 and json.loads((folder / "600" / "split.json").read_text())["split"] == "train"
        for path in train:
            lay = json.loads(path.read_text(encoding="utf-8"))
            src = np.array(lay["Rigid"]["measuring_cup"][0]["default_pos"][:2])
            tgt = np.array(lay["Rigid"]["plastic_cup"][0]["default_pos"][:2])
            assert np.linalg.norm(src - tgt) >= 0.18 and "Instruction" not in lay
            assert lay["Rigid"]["plastic_cup"][0]["category_idx"] in (8, 9)


def test_rigid_grain_capacity():
    """Rigid beans (GranularObject.grain) count the cup's capacity in their own grains; the default is the PBD beans."""
    pbd = round_cup_shape.PBD_GRAIN
    assert round_cup_shape.capacity(GREY, pbd) == round_cup_shape.capacity(GREY)
    black = dict(volume=pbd["volume"] * 2.5, packing=pbd["packing"])
    assert abs(round_cup_shape.capacity(GREY, black) * 2.5 - round_cup_shape.capacity(GREY)) < 1e-6
    cap = round_cup_shape.capacity(GREY, black)
    assert round_cup_shape.fill_ok(round(0.5 * cap), 1900 - round(0.5 * cap), 1900, GREY, (0.3, 0.75), 0.05, black)
    assert not round_cup_shape.fill_ok(round(0.5 * cap), 1900 - round(0.5 * cap), 1900, GREY, (0.3, 0.75), 0.05)


def test_rigid_layouts():
    with tempfile.TemporaryDirectory() as tmp:
        run = [sys.executable, str(ROOT / "scripts/g1d/prepare_pour_half_layouts.py")]
        subprocess.run([*run, "l0", "--assets", tmp, "--seed", "950", "--rigid"], check=True, capture_output=True)
        subprocess.run(
            [*run, "train", "--assets", tmp, "--seed", "600", "--count", "40", "--rigid"],
            check=True,
            capture_output=True,
        )
        folder = Path(tmp, "Eval_Layout/RoboDojo/g1d_lab")
        first = json.loads((folder / "950" / "g1d_pour_beans_half_cup_rigid_0.json").read_text(encoding="utf-8"))
        beans, src = first["Fluid"]["wuliangye"][0], first["Rigid"]["measuring_cup"][0]
        assert beans["physics"]["type"] == "granular" and beans["bean"]["kind"] == "red"
        assert beans["physics"]["pack"].endswith("Object/Granular/red_bean/pack.npz")
        # the pack's frame: the tall cup's outside bottom centre, on the desk
        assert (
            abs(beans["default_pos"][2] - 0.83) < 0.002 and abs(beans["default_pos"][1] - src["default_pos"][1]) < 1e-6
        )
        kinds, yaws = set(), set()
        for path in sorted((folder / "600").glob("g1d_pour_beans_half_cup_rigid_*.json")):
            beans = json.loads(path.read_text(encoding="utf-8"))["Fluid"]["wuliangye"][0]
            kinds.add(beans["bean"]["kind"])
            yaws.add(round(float(np.degrees(2 * np.arctan2(beans["default_ori"][3], beans["default_ori"][0])))) % 360)
        assert kinds == {"red", "mung", "black"} and yaws == {0, 90, 180, 270}
