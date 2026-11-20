"""Tilt probe for the PBD red beans: the tall measuring cup of a layout, filled as the layout says, is driven
kinematically (no robot): it settles, lifts LIFT, then rolls about its lip on the target's side, fast to FAST_TO and
then at RATE (deg/s), as the real G1D-PourBeansEps380 pours do (eps380 FK: ~70 deg in 2 s, then 70 -> ~98 deg over
~6 s, beans first reaching the target at 80-90 deg). Prints and writes the share of the beans out of the tall cup
against the tilt, to compare bean physics settings with the real flow.

Usage (sim env, repo root; any task whose layout has a measuring_cup labelled source_cup and Fluid beans):
    PROBE_OUT=probe.json [PROBE_RATE=4] [PROBE_HOLD=0.25] EVAL_NUM=1 PYTHONPATH=.:XPolicyLab \
        python scripts/g1d/bean_tilt_probe.py --env_cfg_type g1d_lab_beans_eps380 \
        --task_name g1d_pour_beans_half_cup --seed <seed> --layout_ids <i>
PROBE_HOLD: once that share is out, hold the tilt HOLD_S and record how much more leaves (a real pour stops when the
tilt stops).
"""

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.g1d import scripted_demo as demo
from scripts.objects import measuring_cup_shape as cup_shape

SETTLE_S, LIFT, LIFT_S = 1.5, 0.12, 1.5
PILE_S = 3.0  # after the pour, for the beans on the desk to come to rest
FAST_TO, FAST_RATE, MAX_TILT = 70.0, 35.0, 120.0  # deg, deg/s
HOLD_S, AFTER_S = float(os.environ.get("PROBE_HOLD_S", "3")), 1.0


def eval_one_episode(TASK_ENV, model_client):
    env, lm = TASK_ENV, TASK_ENV.scene_manager.layout_manager
    rate = float(os.environ.get("PROBE_RATE", "4"))
    hold_at = float(os.environ.get("PROBE_HOLD", "0"))
    sim = env.sim
    dt = sim.physics_dt
    cup = lm.get_scene_object(0, lm.get_instance_name(0, "source_cup"))
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
    height = float(np.ptp(lm.get_instance_bbox_vertices(lm.get_instance_name(0, "source_cup"), 0)[:, 2]))
    pos0, quat0 = (demo.as_np(v) for v in cup.get_local_pose())
    rot0 = demo.from_wxyz(quat0)
    side = -1.0 if pos0[0] > 0 else 1.0  # roll toward the robot's midline, where the target cup is
    lip = np.r_[side * cup_shape.CUP["A_LIP"], 0.0, height / 2]  # cup frame, origin at the bbox centre
    axis = np.array([0.0, -side, 0.0])  # positive angle tips the mouth toward `side`

    def beans():
        return np.asarray(fluid.get_particle_positions()[0]).reshape(-1, 3) + fluid.init_pos

    def heap_top():
        """Height (m) above the cup's outside bottom under which 95% of the beans in it lie."""
        pos, quat = (demo.as_np(v) for v in cup.get_local_pose())
        local = demo.from_wxyz(quat).inv().apply(beans() - pos) - cup_shape.origin_shift(height)
        inside = cup_shape.in_cavity(local, height)
        return float(np.percentile(local[inside, 2], 95)) if inside.any() else 0.0

    def poured():
        pos, quat = (demo.as_np(v) for v in cup.get_local_pose())
        local = demo.from_wxyz(quat).inv().apply(beans() - pos)
        return 1.0 - float(cup_shape.in_cavity(local - cup_shape.origin_shift(height), height).mean())

    def place(theta, lift, omega=0.0, vz=0.0):
        """Cup rolled theta (deg) about its lip, raised `lift`, moving at omega (deg/s) and vz (m/s)."""
        pivot = pos0 + rot0.apply(lip) + [0.0, 0.0, lift]
        roll = Rotation.from_rotvec(np.deg2rad(theta) * axis)
        rot = roll * rot0
        centre = pivot + roll.apply(rot0.apply(-lip))
        w = np.deg2rad(omega) * axis
        cup.set_local_pose(centre, np.r_[rot.as_quat()[3], rot.as_quat()[:3]])
        velocity = np.r_[np.cross(w, centre - pivot) + [0.0, 0.0, vz], w]  # the physics view takes rad/s
        cup._rigid_prim_view.set_velocities(torch.tensor(velocity[None], dtype=torch.float32))

    def step(theta, lift, omega=0.0, vz=0.0):
        place(theta, lift, omega, vz)
        sim.scene.write_data_to_sim()
        sim.sim.step(render=False)
        sim.scene.update(dt=dt)

    cup._rigid_prim_view.disable_gravities()  # driven every physics step; no sag in between
    record, t = [], 0.0
    start_out = poured()
    for _ in range(int(SETTLE_S / dt)):
        step(0.0, 0.0)
    settled_out, top = poured(), heap_top()
    n = int(LIFT_S / dt)
    for i in range(n):
        step(0.0, LIFT * (1 - np.cos(np.pi * (i + 1) / n)) / 2, vz=LIFT * np.pi / 2 / LIFT_S * np.sin(np.pi * i / n))
    theta, held = 0.0, False
    sub = max(1, round(1 / 30 / dt))  # record at 30 Hz
    k = 0
    while theta < MAX_TILT:
        omega = FAST_RATE if theta < FAST_TO else rate
        theta = min(theta + omega * dt, MAX_TILT)
        step(theta, LIFT, omega)
        t += dt
        k += 1
        if k % sub == 0:
            out = poured()
            record.append([round(t, 3), round(theta, 2), round(out, 4)])
            if hold_at and not held and out >= hold_at:
                held, before = True, out
                for _ in range(int(HOLD_S / dt)):
                    step(theta, LIFT)
                hold = dict(at_deg=round(theta, 1), before=round(before, 4), after=round(poured(), 4))
    for _ in range(int(AFTER_S / dt)):
        step(MAX_TILT, LIFT)
    for _ in range(int(PILE_S / dt)):
        step(MAX_TILT, LIFT)
    rec = np.array(record)

    def angle_at(share):
        hit = np.flatnonzero(rec[:, 2] >= share)
        return round(float(rec[hit[0], 1]), 1) if len(hit) else None

    def pile_angle():
        """Repose angle (deg) of the pile the poured beans made on the desk: its height over the radius holding 80%
        of them, both from the pile's own centre."""
        b = beans()
        top_z = pos0[2] - height / 2  # the desk: where the cup stood
        desk = b[(b[:, 2] > top_z - 0.01) & (b[:, 2] < top_z + 0.06)]  # not on the floor, not in the raised cup
        if len(desk) < 50:
            return None
        centre = np.median(desk[:, :2], axis=0)
        r = np.linalg.norm(desk[:, :2] - centre, axis=1)
        floor = np.percentile(desk[:, 2], 2)
        top = np.percentile(desk[r < 0.015, 2], 95) if (r < 0.015).sum() > 5 else floor
        return round(float(np.degrees(np.arctan2(top - floor, np.percentile(r, 80)))), 1)

    result = dict(
        beans=int(len(beans())),
        out_at_spawn=round(start_out, 4),
        out_after_settle=round(settled_out, 4),
        heap_top_m=round(top, 4),
        cup_height_m=round(height, 4),
        rate_deg_s=rate,
        onset_deg={str(s): angle_at(s) for s in (0.02, 0.05, 0.25, 0.5, 0.75, 0.9)},
        out_at={str(a): round(float(rec[np.flatnonzero(rec[:, 1] >= a)[0], 2]), 4) for a in (80, 90, 100, 110)},
        final=round(poured(), 4),
        pile_deg=pile_angle(),
        hold=hold if held else None,
        curve=record,
    )
    print("[bean_probe]", json.dumps({k: v for k, v in result.items() if k != "curve"}), flush=True)
    if os.environ.get("PROBE_OUT"):
        with open(os.environ["PROBE_OUT"], "w") as f:
            json.dump(result, f)
    env.success[0] = False
    env.end_flag[0] = True


if __name__ == "__main__":
    demo.main(policy_module="scripts.g1d.bean_tilt_probe")
