import collections
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.datagen import grasp_exam

T, CLOSE, LIFT_FROM = 160, 20, 50


def pose(pos, rot):
    return np.r_[pos, np.roll(rot.as_quat(), 1)]


def held(slide=None, turn=None, drop_at=None, fixed=False, knock=False):
    """Hand and cup per frame: the hand pitched 30 deg down closes at CLOSE, rises 0.2 m and rolls 60 deg; the cup
    sits 0.1 m ahead of the hand (rigidly unless `slide` (m along the hand z per frame), `turn` (deg per frame) after
    the lift, `drop_at` (falls 0.2 m from that frame), `fixed` (never leaves the desk), `knock` (tips before CLOSE)."""
    base = Rotation.from_euler("y", 30, degrees=True)
    hand, cup, closed = [], [], []
    offset, extra = np.array([0.1, 0.0, 0.0]), Rotation.identity()
    start = None
    for t in range(T):
        s = np.clip((t - LIFT_FROM) / 40, 0, 1)
        h_pos = np.array([0.3, 0.1, 0.8 + 0.2 * s])
        h_rot = Rotation.from_euler("x", 60 * s, degrees=True) * base
        if t > LIFT_FROM + 40:
            offset = offset + [0, 0, slide or 0.0]
            extra = Rotation.from_euler("z", turn or 0.0, degrees=True) * extra
        c_pos, c_rot = h_pos + h_rot.apply(offset), h_rot * base.inv() * extra  # upright at the start
        if t <= CLOSE + grasp_exam.SETTLE or fixed:
            start = start if start is not None else (h_pos + h_rot.apply(offset), h_rot)
            c_pos, c_rot = start
        if knock and t < CLOSE:
            c_rot = start[1] * Rotation.from_euler("x", 2 * t, degrees=True)
        if drop_at is not None and t >= drop_at:
            c_pos = c_pos - [0, 0, 0.2]
        hand.append(pose(h_pos, h_rot))
        cup.append(pose(c_pos, c_rot))
        closed.append(t >= CLOSE)
    return np.array(cup), np.array(hand), np.array(closed)


def test_rigid_grasp_through_a_moving_turning_hand_is_ok():
    out = grasp_exam.classify(*held())
    assert out["outcome"] == "ok", out
    assert out["slip_m"] < 1e-6 and out["slip_deg"] < 1e-3 and out["rise_m"] > grasp_exam.LIFT_M / 2


@pytest.mark.parametrize(
    ("kw", "outcome"),
    [
        (dict(slide=0.0003), "slipped"),
        (dict(turn=0.5), "slipped"),
        (dict(slide=0.003), "dropped"),
        (dict(drop_at=120), "dropped"),
        (dict(fixed=True), "missed"),
        (dict(knock=True), "knocked"),
    ],
)
def test_failures(kw, outcome):
    out = grasp_exam.classify(*held(**kw))
    assert out["outcome"] == outcome, out
    assert out["frame"] is not None


def test_never_closed_or_released():
    cup, hand, closed = held()
    assert grasp_exam.classify(cup, hand, np.zeros(T, bool))["outcome"] == "missed"
    closed[100:] = False  # let go after the lift: what follows is left out
    cup[100:, 2] -= 0.5
    assert grasp_exam.classify(cup, hand, closed)["outcome"] == "ok"


def episode(tmp_path, i, source, target, flip=False, **extra):
    """An exam quality file: (cup, outcome) per role, the source on the robot's left (right if `flip`)."""
    grasp = {r: dict(cup=c, outcome=o, rise_m=0.04) for r, (c, o) in zip(grasp_exam.ROLES, (source, target))}
    x = 0.1 if flip else -0.1
    planned = {"source_cup": [[x, -0.4, 0.75], [1, 0, 0, 0]], "target_cup": [[-x, -0.4, 0.75], [1, 0, 0, 0]]}
    (tmp_path / f"episode_{i:06d}.json").write_text(json.dumps(dict(grasp=grasp, planned_poses=planned, **extra)))


def test_report(tmp_path):
    rows = {"a": ["ok"] * 9 + ["slipped"], "b": ["ok"] * 8 + ["dropped", "knocked"]}
    for i in range(10):
        a, b = ("Object/RoboDojo/cup/a", "Object/RoboDojo/cup/b")[:: 1 if i % 2 else -1]
        counts = dict(initial_counts={"total": 100}, final_counts={"target": 100 if i < 4 else 100 - i})
        episode(tmp_path, i, (a, rows["a"][i]), (b, rows["b"][i]), flip=i >= 5, **counts)
    (tmp_path / "episode_000099.json").write_text(json.dumps(dict(reason="rejected")))
    cups, trials = grasp_exam.tally(sorted(tmp_path.glob("episode_*.json")))
    assert len(trials) == 20
    a, b = cups["Object/RoboDojo/cup/a [default]"], cups["Object/RoboDojo/cup/b [default]"]
    # a was the source in episodes 1, 3 (left hand), 5, 7, 9 (right hand)
    assert sum(a["source_cup/left"].values()) == 2 and sum(a["source_cup/right"].values()) == 3
    assert a["source_cup/right"]["slipped"] == 1 and sum(b["target_cup/left"].values()) == 3
    # both held in 0-7: 100, 100, 100, 100, 96, 95, 94, 93 of the 100 beans in the target, short of POUR_OK in 6, 7
    assert a["pour"] == {"in": 3, "short": 1} and b["pour"] == {"in": 3, "short": 1}
    assert a["pour_in"] == b["pour"] and b["pour_in"] == a["pour"]
    assert not grasp_exam.passes(a)
    text = grasp_exam.report(cups, trials)
    assert "2 件杯子（按抓法分），20 次抓取，17 次" in text and "8 次倾倒里 6 次 ≥ 95% 的豆子进了目标杯" in text
    assert "| knocked | 0 | 1 |" in text
    # a: the source in 1, 3 (left), 5, 7, 9 (right, 9 slipped), the target in 6, 8 (left), 0, 2, 4 (right);
    # poured in 1, 3, 5, 7: 100, 100, 95, 93 landed
    assert "| Object/RoboDojo/cup/a [default] | 2/2 | 2/3 | 1/2 | 3/3 | 3/4 | 3/4 | 98% |" in text
    # another grasp entry is another row, the same one pools
    for i, grip in ((100, {"mode": "handle", "handle": {}}), (101, {})):
        episode(tmp_path, i, ("Object/RoboDojo/cup/a", "ok"), ("Object/RoboDojo/cup/b", "ok"))
        path = tmp_path / f"episode_{i:06d}.json"
        q = json.loads(path.read_text())
        q["grasp"]["source_cup"]["grip"] = grasp_exam.grip_name(grip)
        path.write_text(json.dumps(q))
    cups, trials = grasp_exam.tally(sorted(tmp_path.glob("episode_*.json")))
    assert sum(cups["Object/RoboDojo/cup/a [handle]"]["source_cup/left"].values()) == 1
    assert sum(cups["Object/RoboDojo/cup/a [default]"]["source_cup/left"].values()) == 3
    assert grasp_exam.grip_name({"rim_down": 0.55, "pitch": 20}) == "pitch=20,rim_down=0.55"


def test_cut_and_unrun_are_no_trials(tmp_path):
    # the source fell: the target's ok was cut short
    episode(tmp_path, 0, ("c", "dropped"), ("d", "ok"), stopped="source_cup left its hand")
    # turned down at the spawn: the plugin wrote both cups missed without measures
    grasp = {r: dict(cup="e", outcome="missed", frame=None) for r in grasp_exam.ROLES}
    planned = {r: [[x, -0.4, 0.75], [1, 0, 0, 0]] for r, x in zip(grasp_exam.ROLES, (-0.1, 0.1))}
    (tmp_path / "episode_000001.json").write_text(json.dumps(dict(grasp=grasp, planned_poses=planned)))
    cups, _ = grasp_exam.tally(sorted(tmp_path.glob("episode_*.json")))
    assert cups["c [default]"]["source_cup/left"]["dropped"] == 1
    assert cups["d [default]"]["target_cup/right"]["cut"] == 1
    assert not grasp_exam.trials(cups["d [default]"]["target_cup/right"]) and not cups["c [default]"]["pour"]
    assert cups["e [default]"]["source_cup/left"]["unrun"] == 1
    assert not grasp_exam.trials(cups["e [default]"]["source_cup/left"])


def counts(**cells):
    """A tally entry: cells and pours given as "ok/tries" (pours: "in/tries")."""
    out = {c: collections.Counter() for c in grasp_exam.CELLS + grasp_exam.POURS}
    for name, text in cells.items():
        good, tries = map(int, text.split("/"))
        key = {"sl": "source_cup/left", "sr": "source_cup/right", "tl": "target_cup/left", "tr": "target_cup/right"}
        cell = key.get(name, name)
        out[cell].update(
            {
                "in" if cell in grasp_exam.POURS else "ok": good,
                "short" if cell in grasp_exam.POURS else "slipped": tries - good,
            }
        )
    return out


def test_pick_per_role():
    cups = {
        "a [default]": counts(sl="1/3", sr="1/2", tl="2/2", tr="2/2", pour="1/1", pour_in="2/2"),
        "a [rim_down=0.6]": counts(sl="3/3", sr="2/2", tl="1/2", pour="2/2", pour_in="1/2"),
        "b [default]": counts(sl="2/2", tr="1/1"),
        "b [rim_down=0.6]": counts(sl="2/2", tr="1/1"),  # as good as the default: the default stays
        "c [rim_down=0.6,width_max=0.088]": counts(sl="1/1", tl="1/1"),
        "c [width_max=0.088]": counts(sl="0/1", tl="1/1"),
        "d [default]": counts(sl="1/1"),  # never a target, the other grasp failed as one: the default for it too
        "d [rim_down=0.6]": counts(tl="0/2", sl="2/2"),  # better as the source than the default
        "g [width_max=0.088]": counts(sl="1/2", tl="1/1"),  # never tried with the default (no scale held it): not it
        "g [rim_down=0.6,width_max=0.088]": counts(sl="0/2"),
        "e [default]": counts(sl="0/2"),
        "e [handle]": counts(sl="2/2"),
        "f [default]": counts(sl="2/2"),
    }
    entries, chosen = grasp_exam.pick(cups, drop={"f": "too wide"})
    assert entries == {
        "a": {"source": {"rim_down": 0.6}},
        "c": {"width_max": 0.088, "source": {"rim_down": 0.6}},
        "d": {"source": {"rim_down": 0.6}},
        "g": {"width_max": 0.088},
    }
    assert chosen["a"] == {"source_cup": "a [rim_down=0.6]", "target_cup": "a [default]"}
    assert chosen["b"]["source_cup"] == "b [default]" and chosen["d"]["target_cup"] == "d [default]"
    assert chosen["g"] == {"source_cup": "g [width_max=0.088]", "target_cup": "g [width_max=0.088]"}
    assert chosen["e"]["source_cup"] == "e [default]" and "f" not in chosen
    a = grasp_exam.composite(cups, chosen["a"])
    assert a["source_cup/left"]["ok"] == 3 and a["target_cup/left"]["ok"] == 2 and a["pour"]["in"] == 2
    assert grasp_exam.passes(a) and not grasp_exam.passes(grasp_exam.composite(cups, chosen["b"]))
    text = grasp_exam.pick_report(cups, [], chosen, {"f": "too wide"})
    assert "| a [rim_down=0.6 / default] | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |" in text
    assert "## 支持的杯子：1 件" in text and "- a [rim_down=0.6 / default]" in text
    assert "- f：too wide" in text


def test_drop_keeps_a_cup_out_of_the_layouts():
    from scripts.datagen import pour_beans_plus_layouts as layouts

    cup = {"loads": True}
    pool = {"cups": {"a": cup, "b": cup, "c": cup}, "split": {"train": ["a", "b"], "eval": ["c"]}}
    cups = layouts.Cups(pool, {"cups": {}, "drop": {"b": "wider than the open pads"}})
    assert cups.train == ["a"] and cups.eval == ["c"]
