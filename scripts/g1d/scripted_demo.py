"""Scripted demos for the G1-D tasks: no policy server, privileged object poses.

Runs src/eval_client/main.py in-process with the policy swapped for a Cartesian script, tracked by damped
least-squares IK on the simulator's Jacobian. Videos and _result.json land in eval_result/ like a normal eval,
under policy name "scripted_demo".
    g1d_pour_water_into_cup   bimanual: one arm grasps the pitcher handle from the robot side, the other picks the cup
                              from the outside and holds it up, mouth leaning to the jug; the jug rolls about its lip
                              until a quarter of the beans have left it, the actual spout steered over the cup; back
                              upright
    g1d_pour_beans_cup_to_cup bimanual: each arm picks the plastic cup on its side from the outside and lifts it; the
                              target cup is held up in front of the chest, the source cup rolls about its rim lip over
                              the target's mouth until most beans are over; back upright
    g1d_pour_beans_measuring_cups  the same with the 500 ml (source) and 400 ml measuring cups
    g1d_general_pickup        top-down grasp across the target's narrow side, lift 15 cm
    g1d_pick_anything         the same for the named object among others, at the grip yaw of its spot, lift 13 cm
    g1d_place_anything        g1d_pick_anything's grasp, carried onto the named support's face and let go
    g1d_put_anything_in       the same into the named container, let go with the fingertips just above its top
    g1d_handover_anything     the hand on the named object's side takes it off-centre from the top and holds it level
                              in front of the chest, the other hand takes the other end from the top, the first lets
                              go and backs off, the second sets it on the named support on its side
    g1d_open_anything         the named door / drawer / lid: the handle taken from the front, the hand then following
                              the part's own arc or slide until it is open, held there
    g1d_stack_blocks          stack the two outer blocks on the middle one, each with the arm on its side; arms back
    g1d_build_block_pattern   the card's block pattern (scripts/objects/block_patterns.py) about its build space,
                              bottom-up: each block picked top-down by the arm on its side, turned to its place with the
                              fingers closing along the pattern's depth, set down, that arm home before the next
    g1d_deliver_cup_to_lounge (g1d_counter_mobile) pick the cup from the counter, back off, drive to a gap between the
                              chairs at the lounge's round table, face it and set the cup down
    g1d_pour_water_deliver_to_lounge (g1d_counter_mobile) the jug pour above with water, the jug set back down, then
                              the cup, left standing on the counter for the pour, carried to the lounge table as above

Usage (sim env, repo root):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:XPolicyLab EVAL_NUM=1 python scripts/g1d/scripted_demo.py \
        --env_cfg_type g1d --task_name g1d_stack_blocks [--seed 0]
"""

import itertools
import os
from pathlib import Path
import runpy
import sys
import zlib

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.environment.isaaclab_compat import pose_wxyz  # noqa: E402
from env.reward_manager.pattern_match import pattern_error, yaw_of  # noqa: E402
from scripts.objects import (
    block_atomic,  # noqa: E402
    block_patterns,  # noqa: E402
    block_sorting,  # noqa: E402
    cup_handle,  # noqa: E402
    cup_profile,  # noqa: E402
    handover_anything,  # noqa: E402
    measuring_cup_shape as cup_shape,  # noqa: E402
    open_anything,  # noqa: E402
    pick_anything,  # noqa: E402
    place_anything,  # noqa: E402
    plastic_mug_shape as mug_shape,  # noqa: E402
    round_cup_shape,  # noqa: E402
)

POLICY = "scripted_demo"

# Calibration knobs: metres, radians, actions (30 Hz)
PAD_REACH = 0.12  # dex1_base_link origin -> grip point on the pads (fingertips 2.3 cm further)
GRASP_IN_PITCHER = np.array([0.0617, 0.0, 0.059])  # handle bar centre (metadata functional.handle)
OPEN = 0.15  # 19 mm between the pads: the inner finger fits in the 26 mm slot between handle and body
BEHIND = np.array([0.0, -0.09, 0.0])  # fingertips clear the pitcher's back face; descend here, then slide in
SPOUT_IN_PITCHER = np.array([-0.0663, 0.0, 0.1145])  # spout lip, the tilt pivot
BBOX_HALF = np.array([0.0663, 0.0445, 0.1225])
# lip short of the cup centre, inside the near rim of the 7.3 cm mouth (the cup hand holds the body below it)
POUR_BACKOFF = 0.025
SPOUT_CLEAR, BODY_CLEAR = 0.03, 0.02  # above the cup rim
TILT = np.deg2rad(125)
TILT_RATE, RETURN_RATE = np.deg2rad(2.0), np.deg2rad(4.0)  # per action
TILT_DRY, DRY_RATE = np.deg2rad(80), np.deg2rad(4.0)  # nothing leaves the jug below this: tilt faster
JUG_SPEED = 0.0035  # m per action carrying the jug: faster swings it about the handle and it slips out
POUR_TARGET = 0.25  # of the half-full jug (~130 ml, 2/3 of a mug); the stream in flight tops it up
FLOW = 0.0033  # of the beans per action (~50 ml/s): hold the tilt above this
STEP_MARGIN = 12  # actions kept free after the jug is back upright, for the rising-edge success check
# Real G1-D motion limits (Limiter): joints this far inside the URDF limits (the real arm server's margin); the
# fastest arm joint speed in the real replays (elbow ~0.8 rad/s, calib toolkit docs/05); the Dex1-1's 0.2 m/s finger
# speed over its 44.5 mm stroke, per 30 Hz action (real2sim/robot/g1d_v4/assemble.json)
REAL_JOINT_MARGIN, REAL_ARM_VMAX, REAL_GRIP_RATE = 0.05, 0.8, 0.2 / 0.0445 / 30
# Water in the jug (g1d_pour_water_deliver_to_lounge): it runs from ~45 deg; a third of the ~300 ml fill
WATER_DRY, WATER_POUR = np.deg2rad(40), 0.3
# tilt speeds scaled to stay under REAL_ARM_VMAX: the dry tilt's 4 deg/action is ~2 rad/s at the wrist, and clipped
# to 0.8 rad/s the hand fell 12 cm behind the spout target
WATER_RATE = 0.4
# Second arm holding the cup
CUP_LIFT = 0.015  # cup bottom above the table while held under the spout; higher puts the jug out of reach
CUP_MEET = 0.0  # the cup is carried this much toward the jug (the cup arm is near the edge of its reach)
# the cup's mouth leans toward the jug to meet the stream, in step with the jug's tilt. With a top grasp on a mug the
# hand sat over the mouth and the tilted jug hit it; held 13-20 cm up, the jug hand cannot reach its pour pose.
CUP_TILT = np.deg2rad(0)
SPOUT_GAIN, SPOUT_SLACK = 0.2, 0.05  # per action, share of the spout error (actual vs over the actual cup) fed back
# side grasp on the cup body (probed at the cup): fingers horizontal miss it by 6-14 cm at any height and wrist yaw;
# 35 deg down with the wrist turned 30-60 deg inward reaches it within 1-3 cm, 8-10 cm above the bottom. At 9 cm the
# wrist camera module on top of the gripper stands above the rim and the tilted jug hits it; at 6.5 cm it stays below
CUP_GRIP_UP = 0.065  # pads this far above the cup bottom
CUP_PITCH, CUP_YAW = 35, 45  # deg, fingers below horizontal; wrist turned inward from fingers-forward
CUP_SLIDE = 0.08  # m the open pads slide along the fingers onto the cup body
# Cup-to-cup pour (two 9.5 cm plastic cups, as in the real G1D-PourBeansPlus clips, or the 14 / 17 cm measuring
# cups): pads this far below the rim. Measuring cups: 45 mm keeps the pads clear of the 75 mm flange 12.5 mm down;
# at 32 mm the 17 cm cup pivoted ~45 deg in the grip (16/18 strict pours at 45 mm, 2 failures one spilled bean each)
C2C_RIM_DOWN = {"plastic_cup": 0.02, "measuring_cup": 0.045, "plastic_mug": 0.03}  # mug: pads under its handle top
C2C_RAISE = 0.06  # straight up first
# target cup bottom lifted, by its kind; the mug's mouth then sits about as high as the short measuring cup's (at 0.12
# the tall cup's hand could not reach over it)
C2C_LIFT = {"plastic_cup": 0.12, "measuring_cup": 0.06, "plastic_mug": 0.08}
# fingers this far below horizontal. The cup tilts by rolling about the fingers, cos(tilt) = sin^2 p + cos^2 p cos(r):
# at 35 deg the ~100 deg the beans need in the tall measuring cup takes ~140 deg of roll; the arm reaches ~116
C2C_PITCH = {"plastic_cup": CUP_PITCH, "measuring_cup": 20, "plastic_mug": CUP_PITCH}  # per cup, by its kind
# the square measuring cups stand at any yaw: the wrist turns beyond CUP_YAW (deg, outward .. inward) so the pads close
# across a pair of flats (they repeat every 90 deg); past this window the pads sit off the flats by the rest. Probed at
# the table centre: 25 deg further inward knocks the cup over, 44 deg outward misses the pour
C2C_FLAT_TURN = (-35, 15)
C2C_MEET_X, C2C_MEET_Y = 0.03, -0.36  # target cup axis in mid-air: this far on its own side of the midline
CUP_SPEED = 0.005  # m per action carrying a cup
# source lip this far inside the target's near rim, this high above it. Lying flat, the source cup reaches a cup width
# past its lip: further in, its far rim sweeps into the wrist camera module of the hand holding the target
LIP_IN, LIP_CLEAR = 0.010, 0.035
C2C_TILT, C2C_DRY = np.deg2rad(150), np.deg2rad(60)  # most tilt; nothing leaves a sixth-full cup below C2C_DRY
C2C_DRY_RATE, C2C_TILT_RATE = np.deg2rad(2.0), np.deg2rad(1.0)  # per action; faster and the arm lags, the cup slips
C2C_SETTLE, C2C_SETTLE_ERR = 40, 0.01  # most actions held in mid-air until both grasp points are this close
C2C_POUR, C2C_FLOW = 0.85, 0.004  # of the beans out of the source cup; tilt further while less leaves per action
# g1d_pour_beans_half_cup: the tall cup is full as the real one (~3100 beans) and sheds beans from ~88 deg; stop
# rolling once HALF_FILL of the target cup's capacity has left it (more follows while it rolls back), then put both
# cups back where they stood and let go
HALF_FILL = 0.4
# the first, empty roll at the real ~35 deg/s (eps380 FK: 0 to 70 deg in ~2 s); at C2C_DRY_RATE (60 deg/s) a hand
# still catching up with the meet point jerks the full cup and throws beans over the rim
HALF_DRY_RATE = np.deg2rad(1.0)  # per action
# where the real hands meet (eps380_episodes.py FK at the frame the right wrist rolls most): the target cup 40.7 cm
# ahead of the robot base and 3.3 cm left of it. The real cup is lifted 6 cm; the real tall cup rolls to 87-99 deg
# (median 96, eps380 FK) as here. Rolled to ~100 deg over a cup that low, the tall cup's body knocked the target hand
# (0-80% in the cup), so lift it 12 cm as the other cup-to-cup pours and hold it 3.5 cm under the rim: the hand then
# stays under the tall cup's sweep (4/4 episodes, no bean spilled; 2 cm under: the hands met, 2026-11-01). 40.7 cm
# also keeps the tall cup in reach: at 33-37 cm its hand fell 6-9 cm short of the meet and the jerk threw out 10-22%
HALF_MEET, HALF_LIFT, HALF_RIM_DOWN = (0.033, 0.407), 0.12, {"plastic_cup": 0.035}
# g1d_pour_beans_plus: cups of any shape, by their profile (scripts/objects/cup_profile.py) instead of their kind.
# Pads PLUS_RIM_DOWN (share of the height + m, clipped) below the rim; fingers PLUS_PITCH deg down (the tall cups
# need the measuring cups' shallower pitch to roll far enough); the target's rim held PLUS_MOUTH above the desk
# (the twin plastic cup, 9.5 cm: C2C_LIFT). Pads PLUS_RIM_DOWN (share of the height) under the rim, at least
# PLUS_RIM_MIN, not in the lowest share PLUS_RIM_DOWN[1]: 2 cm under the rim, as C2C_RIM_DOWN, put the rolled source
# hand on the target hand at ~95 deg (2026-11-03). Pads PLUS_PAD tall hold a wall within PLUS_SLOPE deg of vertical
# (friction 0.5: a cone steeper than ~27 deg slides out). At most tilt, the pour stops after PLUS_STALL actions
# without a bean leaving (some always stick in the corners); it rolls back while more than PLUS_SURGE of the beans
# leave per action
PLUS_RIM_DOWN, PLUS_RIM_MIN, PLUS_PAD, PLUS_SLOPE = (0.4, 0.25), 0.02, 0.02, 10
PLUS_PITCH, PLUS_TALL = (25, 20), 0.14
# Front of dex1_model_base_link in dex1_base_link (m: x, |y|, z): the palm (body_collision_36) and wrist-camera shell
# (housing_collision_00). Pitched down, they sit ~4 cm behind PAD_REACH and hit a wide or deep-grasped cup's near rim
# before the pads close; keep the grip clear of them, backing off at most PLUS_PAD_AHEAD (pads end 24.8 mm ahead).
PLUS_PALM = np.vstack(
    [
        np.c_[np.full(15, 0.0738), np.full(15, 0.014), np.linspace(-0.0335, 0.0335, 15)],
        np.c_[np.full(9, 0.0760), np.full(9, 0.007), np.linspace(0.0440, 0.0767, 9)],
    ]
)
PLUS_PAD_AHEAD, PLUS_PALM_CLEAR = 0.010, 0.002
# A cup whose grasp entry has mode "handle" (and its handle, scripts/objects/cup_handle.py) is held by the handle as
# the jug is: fingers forward through the hole, HANDLE_PITCH deg down, the handle pointing out along the closing axis
# (the mouth tips away from it), the inner finger centred between the bar and the body, HANDLE_CLEAR at most either
# side of it. Level fingers at the cups' places ask the wrist pitch past its limit (URDF reach probe 2026-11-05: the
# grasp point 1.7-4 cm short, the cups pushed off; 0 from 10 deg down). The hole holds the tilted finger, its height
# plus the drop over the bar's width, with HANDLE_FIT to spare (m) each way at the cup's scale. Grasp exam 2026-11-05:
# 32% of the handle grasps held (the pads turn about the curved bar), so no grasps.json entry uses it
HANDLE_PITCH, HANDLE_CLEAR, HANDLE_FIT = 10, 0.0125, 0.001
PLUS_MOUTH, PLUS_LIFT, PLUS_STALL, PLUS_SURGE = 0.215, (0.05, 0.12), 60, 0.015
# The source cup's tilt from its cavity: a bean at any height leaves once the wall from there to the rim slopes down,
# tilt > 90 deg - the least wall flare over >= PLUS_CHORD of height + atan(grain friction 0.3, the "min" combine of
# pour_beans_plus_layouts.GRAIN_COMBINE) + PLUS_TILT_MARGIN (heaps in the corners), within PLUS_NEED. The fingers
# take the steepest pitch of PLUS_PITCHES (deg; the grasp's reach) that gets there within PLUS_ROLL_NEED of roll;
# the roll stops PLUS_ROLL_SPARE past it, at most PLUS_ROLL_MAX (B probe 2026-11-04: 170 deg rolled cleanly; at 30
# deg the loaded cups slid in the hand or fell out of it, C probe; the cups tilt 10-25 deg short of the planned roll
# under load, E probe: 147 deg of roll tipped a cup to 94 deg and the pour ran dry at the step limit)
PLUS_CHORD, PLUS_TILT_MARGIN, PLUS_NEED = 0.015, 8.0, (100.0, 135.0)
PLUS_PITCHES, PLUS_ROLL_NEED, PLUS_ROLL_SPARE, PLUS_ROLL_MAX = (25, 20), 160.0, 30.0, 170.0
# The stream the target mouth takes: a surge is PLUS_SURGE_MOUTH of its area in grain sections (volume-equivalent
# diameter) per action (the twin cup, red beans: ~7), a weak stream a PLUS_WEAK share of that; past PLUS_SLOW deg of
# tilt (the last heap slides all at once, B probe: 7/16 spills went over the far rim at 103-112 deg) the roll is
# PLUS_SLOW_RATE of C2C_TILT_RATE
PLUS_SURGE_MOUTH, PLUS_SURGE_GRAINS, PLUS_WEAK = 0.045, (1.5, 12.0), 0.25
PLUS_SLOW, PLUS_SLOW_RATE = np.deg2rad(95), 0.5
# Spills (privileged: a grain out of both cavities, PLUS_SPILL_DROP under the target rim, off its mouth): each new one
# rolls the cup back PLUS_SPILL_BACK and holds the tilt PLUS_SPILL_CALM actions, lowers the lip PLUS_SPILL_LOWER (to
# PLUS_CLEAR_MIN over the rim) and moves the lip target PLUS_AIM_GAIN of the spill's offset along the pour back toward
# the mouth centre (over the far rim: back; short of the near rim: further in), at most PLUS_AIM_STEP at a time,
# within PLUS_AIM. Under PLUS_SPILL_FROM of tilt no bean runs over the rim: beans out then fell in the grasp or the
# lift and neither steer the lip nor end the dry roll (C probe 2026-11-04: such a bean pushed the lip 1.5 cm in, and
# the full cup's slide at 85 deg then went over the far rim)
PLUS_SPILL_DROP, PLUS_SPILL_BACK, PLUS_SPILL_CALM, PLUS_SPILL_FROM = 0.01, np.deg2rad(2.0), 15, np.deg2rad(30)
# The dry roll also ends once the highest bean is PLUS_LIP_NEAR under the lowest rim point: a full cup's heap then
# slides out at once (C probe: 15% at 54 deg, over the far rim)
PLUS_LIP_NEAR = 0.01
PLUS_SPILL_LOWER, PLUS_CLEAR_MIN, PLUS_AIM_GAIN, PLUS_AIM_STEP, PLUS_AIM = 0.003, 0.015, 0.3, 0.005, (-0.02, 0.015)
# At most tilt the arm runs out of roll (cups stop at 105-108 deg) with 1-3% wedged in the corners: a stall first
# jolts the cup PLUS_SHAKE along the pour and back, PLUS_SHAKE_ACTS actions, up to PLUS_SHAKES times (office
# feasibility 2026-11-04: beans left in the source 1.6 -> 0.7 an episode)
PLUS_SHAKES, PLUS_SHAKE, PLUS_SHAKE_ACTS = 3, 0.008, 12
# The pour starts from how the source sits in the hand as measured then (crept a few degrees on the way, the arm
# ~4 mm under its command): its first command stepped the hand 11 mm in one action (p90 21 mm, in every grasp-exam
# episode; 10-20 m/s^2 just as the roll starts, where most source slips began; 2026-11-06). That step from the carry's
# last command is eased out (smoothstep) over PLUS_HANDOVER actions, the lip feedback held meanwhile
PLUS_HANDOVER = 20
# Arm.ik's one damped least-squares step from the measured joints moved a joint up to 80 deg in one action (the held
# source arm's steps are 5.9 deg at p99). Capped joint by joint at PLUS_JOINT_STEP (V3a exam, 2026-11-06), 80 of 344
# episodes still rode the cap and held the source in 57% (76% without): lifting from a grasp near a singular pose (a
# 6 cm lift swung the elbow 60 deg; the elbow alone clipped, the hand's path bent and tipped the cup), the carry
# across, and a joint at its limit, whose column dropped out of the solve and back every action (the arm chattered
# +-6 deg to the end, beans thrown out). The plus task scales the whole joint step down to PLUS_JOINT_STEP (rad),
# weighs a joint down to PLUS_LIMIT_FLOOR within PLUS_LIMIT_ZONE (rad) of a limit instead (plus_dls), and damps up to
# PLUS_DAMP_MAX once the weighted Jacobian's smallest singular value is under PLUS_SINGULAR (~2% of the frames)
PLUS_JOINT_STEP = np.deg2rad(6.0)
PLUS_LIMIT_ZONE, PLUS_LIMIT_FLOOR = np.deg2rad(10.0), 0.05
PLUS_SINGULAR, PLUS_DAMP_MAX = 0.04, 0.06
# Of the source cups V3b dropped or slipped (A/B, 2026-11-07), 20 of 34 had started turning in the hand before any tilt
# (12 within 26 actions of lifting off: the 6 cm lift took 12 actions, 2 m/s2 at its ends), the held ones turned 0.5
# deg median and the failed 6 deg by the pour; most others turned where the roll jumped: 2 deg an action from rest, the
# spill feedback's 2 deg back in one action, the end shake's +-8 mm square wave (14-28 m/s2). The plus task lifts the
# first C2C_RAISE over at least PLUS_LIFT_ACTS actions, eases its moves with zero acceleration at both ends (quintic),
# changes the roll rate by at most PLUS_ROLL_ACCEL (rad per action, per action) and shakes in a sine of
# PLUS_SHAKE_PERIOD actions under a half-sine envelope
PLUS_LIFT_ACTS, PLUS_ROLL_ACCEL, PLUS_SHAKE_PERIOD = 30, np.deg2rad(0.4), 6
# The beans thrown off the source mouth were the main loss of 16 of the 37 V3b pours under 95% (confirmation exam,
# 2026-11-07), the hand at 4-7 m/s2 as they left, and V3c's too: every spill moved the lip target PLUS_AIM_STEP along
# the pour and PLUS_SPILL_LOWER down in one action, and the spills, seen some actions late, fell on both sides of the
# mouth, so the lip swung +-2 cm a few actions apart. The plus task moves the lip target once a spell (a spill after
# PLUS_SPILL_CALM actions without one) and eases the lip to it, its rate changing by at most PLUS_LIP_ACCEL (m per
# action, per action), up to PLUS_LIP_RATE
PLUS_LIP_ACCEL, PLUS_LIP_RATE = 0.0003, 0.001
# The plus task's spill feedback acts on predicted spills: a bean out of both cups above the target rim plane, falling
# from where it is at its velocity over the last action (PLUS_ACTION_DT s, an action per recorded frame) under
# gravity, lands on that plane past the inner mouth radius plus PLUS_PREDICT_PAD (m; a grain radius in: it hits the
# rim). Recorded V3c / V3d pours (96, 2026-11-09): the beans lost over the source mouth crossed the rim plane beside
# the mouth (43%) or short of its near rim (36%) more than past its far rim (21%), the more the wider the source mouth
# than the target's (lost 0.5% of the beans at radius ratio < 0.8, 8.3% over 1.5); one new predicted bean fired in all
# 30 pours that lost >= 2%, 6 actions (median) before the first spilled bean reached the spill test, before all but
# 1 of their 947 lost beans had left the source, and in 18 of the 43 pours that lost none
PLUS_ACTION_DT, PLUS_PREDICT_PAD, GRAVITY = 1 / 30, -0.003, 9.81
# Only beans falling faster than PLUS_PREDICT_FALL (m/s; a free fall reaches 0.33 in an action, a bean the tilting cup
# carries ~0.05) count: V4 (gen exam 2801, 2026-11-09) took beans resting on the source rim, landing short of the near
# rim at no speed, for a spill every few actions and held a 321-bean pour under 90 deg until the step limit. Falling:
# fired in 12 of the 43 clean pours, 1 action each (10.5 unfalling), still in all 30 lossy ones, 4 actions ahead. A
# falling bean due within PLUS_PREDICT_WARN (m) of that radius holds the tilt: in 1.2% of the clean pours' actions,
# 16.6% of the lossy ones'
PLUS_PREDICT_FALL, PLUS_PREDICT_WARN = 0.1, 0.005
# The stream coming: the beans in the source that, moving on at their velocity in its frame over the last action for
# PLUS_FORECAST actions, are out of its cavity, per action. The pre-tilt ends, and the tilt only goes on, while it is
# weak too: a heap sliding to the lip shows in it before it leaves (V3d / V4b / V4c gen pours, 324, 2026-11-10: over
# the weak level a median 7 actions before a slug over the target mouth's surge set in, 3 or more in 92% of the 507
# slugs; the measured flow 2, 49%); the two hold the tilt in 36% / 40% of the actions of the pours losing < 1%
PLUS_FORECAST = 6
# The predictions above cost time: a wide source over a narrow target sends a bean near the rim every few actions, and
# each held the tilt (V4c, hard exam 2802, 2026-11-10: 5 of 53 pours at the step limit, 67 predicted spells in one
# that lost 5 beans). Once the pour falls behind its pace (under PLUS_PACE x the share of the actions left at its
# start, counted from PLUS_PACE_ONSET of the beans gone and PLUS_PACE_GRACE actions on), only seen spills hold the
# tilt, as before them. Replayed on the V4c logs: behind in none of 108 gen pours, in the 5 hard ones (at 9-43% poured).
# Beans out of both cups as the pour starts (a full cup leaks in the carry) neither count as poured nor hold the tilt,
# and neither do beans leaving the source through its wall, more than PLUS_WALL (m, cup at scale 1) under its spill
# level: holding the tilt for them only keeps the heap on the wall longer (V4e, gen 2801 layout 15: rolled back at
# 32-56 deg for 4 such beans, poured 1051 frames, 39 of the 48 lost beans went through the wall; V3d 656, 17 lost)
PLUS_PACE, PLUS_PACE_ONSET, PLUS_PACE_GRACE, PLUS_WALL = 1.2, 0.02, 30, 0.01
# A spell of spills (each within PLUS_SPILL_CALM of the last) rolls the cup back at most PLUS_SPILL_DEPTH below where it
# began: a bean a step, seen or predicted, otherwise walks the tilt down into a slow sheet over the whole lip that a
# narrow mouth misses beside. Gen / hard exams 2801 / 2802 (V3d, V4e, V4g together, 2026-11-10): pours whose longest
# run of rollbacks (each within 2 actions) was <= 2 put >= 95% in 372 of 376 times, 3-5 in 29 of 30, >= 6 in 10 of 17;
# hard 2802 layout 9 (plastic cup into a flute): V4e rolled back 2 at a time, poured at 105-112 deg, lost none; V4g
# rolled back 9 in a row (108 to 90 deg), dribbled at 80-92 deg and lost 30 beans beside the mouth
PLUS_SPILL_DEPTH = np.deg2rad(6.0)
PLACE_CLEAR = 0.006  # m above the cup's start grip height when it is let go
PLACE_OPEN, PLACE_BACK = 20, 0.06  # actions to open the gripper; m to back off (along the slide-in) and rise
# g1d_pour_beans_plus: the poured cup first rises clear of the target's mouth (E probe 2026-11-04: turned 15 deg in the
# hand, it swept the full target cup over on the way back), opens fully (PLUS_PLACE_OPEN actions: Dex1-1 at 0.064 m/s
# needs ~22 actions for the full 44.5 mm stroke so the pads clear a cup that rotated in the grip), and backs off
# straight along the approach before rising; when `place` is on, reserve actions to unwind, set both cups down and hold
# HOLD=30 frames before STEP_LIM
PLACE_RISE, PLUS_PLACE_OPEN, PLUS_PLACE_ACTS = 0.04, 38, 265
LIP_GAIN, LIP_SLACK = 0.2, 0.04  # per action, share of the lip error fed back; bound on the correction
DAMPING = 0.05
GRASP_ROT = Rotation.from_euler("z", 90, degrees=True)  # fingers forward (+y), wrist camera up: the G1-D zero pose
# Pick and place
# fingers 60 deg below horizontal, closing along x: reaches ~8 cm further at table height than straight down
TOP_ROT = Rotation.from_euler("x", -60, degrees=True) * GRASP_ROT
PAD_CLOSED, PAD_TRAVEL = 0.006, 0.089  # pad gap (m): 6 mm at grip=0 .. 95 mm at grip=1 (2 * 44.5 mm finger stroke)
GRASP_CLEAR = 0.025  # open the pads this much wider than the grasped side so neighbours are not swept in
FINGER_TIPS = 0.023  # grip point -> fingertips
TABLE_CLEAR = 0.006  # fingertips above the table
HOVER, LIFT = 0.06, 0.15  # 6 cm hover keeps 3-block stacks below the wrist-pitch limit; pickup lifts 15 cm
PLACE_GAP = 0.004  # held object's bottom above the surface it is released on
# Mobile (g1d_mobile): chassis cruise speeds, ramp, tolerances
DRIVE_V, DRIVE_W, DRIVE_ACC = 0.35, 0.5, 0.5 / 30  # m/s, rad/s; m/s per action (a held cup slips at hard starts)
DRIVE_TOL, HEADING_TOL, TURN_FIRST = 0.03, 0.03, 0.25  # m, rad; turn on the spot first when off course by more
SETTLE = 15  # actions standing still after a drive
# m straight back from the counter before turning: the held cup clears its front edge, and the chassis stays clear of
# the bar stools on the far side of the aisle (real_scene2; 0.6 m already touches them)
BACK_OFF = 0.5
# Lounge table: root this far outside its rim; the cup set down this far ahead of the root; approach from this far out
TABLE_STAND, PLACE_AHEAD, APPROACH = 0.20, 0.38, 0.5
CHAIR_GAP = np.deg2rad(70)  # narrowest gap between the chairs round the table to stand in
LOUNGE_LIFT = 0.07  # lift column at the 0.75 m table: 0.07 m above 0.68 m, as the counter's 0.153
CARRY_UP = 0.08  # m the side-grasped cup is lifted off the counter for the drive (clears the chair backs, 0.8 m)


def pad_open(width):
    return float(np.clip((width + GRASP_CLEAR - PAD_CLOSED) / PAD_TRAVEL, 0.0, 1.0))


def as_np(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=float).reshape(-1)


def link_jacobians(art):
    """(envs, links - 1, 6, dofs) at the link origins in world, PhysX's fixed-base layout, on every sim backend."""
    view = art.root_view if hasattr(type(art), "root_view") else art.root_physx_view  # IsaacLab 3 / 2 and MuJoCo
    if not hasattr(view, "get_jacobians"):
        from env.backends.newton import physics

        return physics.link_jacobians(view)
    jac = view.get_jacobians()
    if not hasattr(jac, "cpu"):
        import warp as wp

        jac = wp.to_torch(jac)
    return jac


def from_wxyz(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def waypoints(p, rot, keys, quintic=False):
    """Smoothstep-interpolated (grasp point, hand rotation, gripper) per action through `keys`; `quintic`: smootherstep,
    no acceleration jump where a move starts or stops."""
    for p1, rot1, grip, n in keys:
        slerp = Slerp([0, 1], Rotation.concatenate([rot, rot1]))
        for i in range(1, n + 1):
            t = i / n
            s = t**3 * (10 - 15 * t + 6 * t * t) if quintic else t * t * (3 - 2 * t)
            yield p + (p1 - p) * s, slerp(s), grip
        p, rot = p1, rot1


def plus_ease(state, target, accel=PLUS_ROLL_ACCEL, top=RETURN_RATE):
    """The command toward `target` with its rate changing by at most `accel` an action and braking in time, up to
    `top` an action (defaults: the roll, rad); `state` is [command, rate], updated. Braking from a rate v covers about
    v^2 / 2 accel + v / 2 in whole actions."""
    err = target - state[0]
    want = np.sign(err) * min(abs(err), np.sqrt(accel * accel / 4 + 2 * accel * abs(err)) - accel / 2, top)
    state[1] = float(np.clip(want, state[1] - accel, state[1] + accel))
    state[0] += state[1]
    return state[0]


class Arm:
    REACH = np.array([PAD_REACH, 0.0, 0.0])  # ee link -> grasp point, ee link frame
    # joint posture ik() pulls the arm's redundancy toward, REST_GAIN of the way per step, in the null space of the
    # hand's pose (None: no pull). g1d_open_anything sets the start posture: without it the redundancy drifted over
    # the long approach, and sliding onto the handle threw the arm in 11 of 26 tries and broke PhysX in 41 of 66
    # episodes (screen13; with it 6 of 50, screen11). With a posture set, one step also aims at most STEP_MAX (m, rad)
    # toward the target: a hand far behind its waypoint swung the arm 0.4 m back and forth (screen14)
    rest, REST_GAIN, STEP_MAX = None, 0.1, (0.03, 0.2)

    def __init__(self, env, robot):
        rm = env.robot_manager
        self.robot, self.key = robot, rm.robot_key[rm.robot_list.index(robot)]
        self.body = self.key.body_names.index(robot.ee_link_name)
        self.cols = list(robot.arm_joint_indices)
        self.origin = as_np(env.scene_manager.env_origins[0])
        self.arm_key, self.grip_key = rm.process_name(robot.arm_name), rm.process_name(robot.gripper_name)

    def q(self):
        return as_np(self.key.data.joint_pos[0, self.cols])

    def root(self):
        return as_np(self.key.data.root_pos_w[0])[:3] - self.origin

    def pose(self):
        link = as_np(pose_wxyz(self.key.data.body_link_pose_w[0, self.body]))
        return link[:3] - self.origin, from_wxyz(link[3:])

    def grasp_point(self):
        pos, rot = self.pose()
        return pos + rot.apply(self.REACH)

    def at_limits(self, margin=0.03):
        """Indices of the arm joints within `margin` rad of a soft limit."""
        lim = as_np(self.key.data.soft_joint_pos_limits[0, self.cols]).reshape(-1, 2)
        q = self.q()
        return [i for i in range(len(q)) if q[i] - lim[i, 0] < margin or lim[i, 1] - q[i] < margin]

    def ik(self, point, rot_t, smooth=False):
        """Joint targets one damped least-squares step from the measured joints toward the grasp point and hand
        rotation; `smooth`: by plus_dls (no column switched off at a limit)."""
        pos, rot = self.pose()
        r = rot.apply(self.REACH)
        err = np.r_[point - (pos + r), 0.3 * (rot_t * rot.inv()).as_rotvec()]
        if self.rest is not None:
            for part, most in zip((err[:3], err[3:]), (self.STEP_MAX[0], 0.3 * self.STEP_MAX[1])):
                part *= min(1.0, most / max(np.linalg.norm(part), 1e-9))
        jac = as_np(link_jacobians(self.key)[0, self.body - 1][:, self.cols]).reshape(6, -1)
        jac = np.r_[jac[:3] - np.cross(r, jac[3:].T).T, 0.3 * jac[3:]]
        q = self.q()
        lim = as_np(self.key.data.soft_joint_pos_limits[0, self.cols]).reshape(-1, 2) + [0.025, -0.025]
        if smooth:
            return np.clip(q + plus_dls(jac, err, q, lim), lim[:, 0], lim[:, 1])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), err)
        jac[:, (q + dq <= lim[:, 0]) & (dq < 0) | (q + dq >= lim[:, 1]) & (dq > 0)] = 0.0
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + 0.02**2 * np.eye(6))
        step = pinv @ err
        if self.rest is not None:
            step += (np.eye(len(q)) - pinv @ jac) @ (self.REST_GAIN * (self.rest - q))
        return np.clip(q + step, lim[:, 0], lim[:, 1])


def plus_landing(pts, vel, mouth):
    """Where beans at `pts` (m) moving at `vel` (m/s) cross the horizontal plane through `mouth` under gravity, as
    horizontal offsets from `mouth` (beans under the plane taken as on it)."""
    drop = np.maximum(pts[:, 2] - mouth[2], 0.0)
    fall = (vel[:, 2] + np.sqrt(vel[:, 2] ** 2 + 2 * GRAVITY * drop)) / GRAVITY
    return pts[:, :2] + vel[:, :2] * fall[:, None] - mouth[:2]


def plus_due(local, last, actions, cavity, scale):
    """Per bean (cup frame at scale 1: at `local` now, at `last` `actions` actions before): in the cavity now and out
    of it near or past the spill level (not jittering through the floor or lower wall) PLUS_FORECAST actions on."""
    ahead = local + (local - last) * PLUS_FORECAST / actions
    pose = np.zeros(3), (1.0, 0.0, 0.0, 0.0)
    return (
        cup_profile.in_cavity(local * scale, *pose, cavity, scale)
        & ~cup_profile.in_cavity(ahead * scale, *pose, cavity, scale)
        & (ahead[:, 2] >= cavity["z"][-1] - PLUS_WALL)
    )


def plus_dls(jac, err, q, lim):
    """Joint step of the weighted damped least squares: each joint's column weighed by its room to the nearer limit
    (1 beyond PLUS_LIMIT_ZONE, down to PLUS_LIMIT_FLOOR), a function of the joints alone so the solve does not switch
    between actions; the damping 0.02 grows to PLUS_DAMP_MAX as the weighted Jacobian nears a singular pose."""
    room = np.minimum(q - lim[:, 0], lim[:, 1] - q)
    w = np.clip(room / PLUS_LIMIT_ZONE, PLUS_LIMIT_FLOOR, 1.0)
    jw = jac * w
    low = np.linalg.svd(jw, compute_uv=False)[-1]
    damp = np.hypot(0.02, PLUS_DAMP_MAX * np.sqrt(max(0.0, 1.0 - (low / PLUS_SINGULAR) ** 2)))
    return w * (jw.T @ np.linalg.solve(jw @ jw.T + damp**2 * np.eye(len(err)), err))


def plus_joint_step(q, last):
    """Joint command `q` scaled back toward the `last` one so that no joint moves more than PLUS_JOINT_STEP, all
    joints by the same share (the hand keeps its direction, only slower)."""
    step = q - last
    most = float(np.abs(step).max())
    return q if most <= PLUS_JOINT_STEP else last + step * (PLUS_JOINT_STEP / most)


def flat_turn(rot, out):
    """Wrist yaw (deg, world z) beyond out * CUP_YAW that closes the pads across a pair of a square cup's flats, within
    C2C_FLAT_TURN: the pads close along world angle out * CUP_YAW (= 45 mod 90), the flats face the cup's yaw + k * 90
    deg, and out * turn > 0 turns the wrist further inward. Past the window: the edge nearest a pair of flats."""
    want = out * (np.degrees(rot.as_euler("zyx")[0]) % 90 - 45)
    edges = (np.clip(w, *C2C_FLAT_TURN) for w in (want - 90, want, want + 90))
    inward = min(edges, key=lambda c: abs((c - want + 45) % 90 - 45))
    return out * float(inward)


def cup_grasp(lm, out, label="cup", grip_up=CUP_GRIP_UP, pitch=CUP_PITCH, turn=0.0):
    """Side grasp around the body of the (handleless, tapered) cup by the arm on side `out` (+-1 in x): fingers
    `pitch` deg down, turned CUP_YAW (+ `turn`) inward, pads closing across the body below the rim; the mouth stays
    free.

    Returns the grasp point (on the body axis), hand rotation, rim height, rim radius and gripper opening.
    """
    pos, quat = lm.get_instance_pose(0, label=label)
    pos, rot = as_np(pos), from_wxyz(as_np(quat))
    meta = lm.get_instance_metadata(0, label=label) or {}
    bbox = lm.get_instance_bbox_vertices(lm.get_instance_name(0, label), 0)
    lo, hi = bbox.min(0), bbox.max(0)
    circle = meta.get("active", {}).get("place", {}).get("up", {}).get("contact_circle", {})
    axis = pos + rot.apply(np.r_[circle.get("center", (lo + hi) / 2)[:2], 0])
    rim, radius = pos[2] + hi[2], (hi[1] - lo[1]) / 2
    foot = circle.get("radius", radius)
    up = min(grip_up, hi[2] - lo[2])
    grip = np.r_[axis[:2], pos[2] + lo[2] + up]
    hand = Rotation.from_euler("xz", [-pitch, out * CUP_YAW + turn], degrees=True) * GRASP_ROT
    return grip, hand, rim, radius, pad_open(2 * (foot + (radius - foot) * up / (hi[2] - lo[2])))


def plus_palm_push(profile, scale, grip_up, pitch):
    """How far (m, >= 0) the Dex1-1 palm or wrist-camera shell reaches past a cup's near wall when PAD_REACH sits on
    its axis `grip_up` above its underside with the fingers `pitch` deg below horizontal."""
    p = np.radians(pitch)
    dx, y, dz = PLUS_PALM[:, 0] - PAD_REACH, PLUS_PALM[:, 1], PLUS_PALM[:, 2]
    fwd = dx * np.cos(p) + dz * np.sin(p)
    zw = profile["bottom"] * scale + grip_up - dx * np.sin(p) + dz * np.cos(p)
    in_z = (zw >= profile["bottom"] * scale) & (zw <= profile["top"] * scale)
    if not in_z.any():
        return 0.0
    rad = np.array([cup_profile.body_radius(profile, zi, scale) for zi in zw[in_z]])
    m = y[in_z] < rad
    if not m.any():
        return 0.0
    return float(max(0.0, (fwd[in_z][m] + np.sqrt(rad[m] ** 2 - y[in_z][m] ** 2)).max()))


def plus_grip(profile, scale, rim_down=None, pitch=None):
    """g1d_pour_beans_plus: (pads above the cup's underside, fingers' pitch in deg) for a cup by its profile: where
    its wall is within PLUS_SLOPE of vertical (the pads squeeze a cone out of the grip) and the palm / wrist camera
    stays within PLUS_PAD_AHEAD of clearing the rim, nearest PLUS_RIM_DOWN, at least PLUS_RIM_MIN under the rim; the
    least sloped height when it is a cone throughout. A cup's grasp entry (real2sim/tasks/g1d_pour_beans_plus/
    grasps.json) may replace the share under the rim (`rim_down`, down to the lowest share left free) and the pitch."""
    bottom, top = profile["bottom"] * scale, profile["top"] * scale
    height = top - bottom
    pitch = PLUS_PITCH[height > PLUS_TALL] if pitch is None else pitch
    share, low = PLUS_RIM_DOWN
    if rim_down is not None:
        share, low = rim_down, min(low, 1.0 - rim_down)
    z = np.linspace(bottom + low * height, top - PLUS_RIM_MIN, 40)
    span = np.r_[-1, 1] * PLUS_PAD / 2
    radius = np.array([[cup_profile.body_radius(profile, zi + d, scale) for d in span] for zi in z])
    slope = np.abs(radius[:, 1] - radius[:, 0]) / PLUS_PAD
    push = np.array([plus_palm_push(profile, scale, zi - bottom, pitch) for zi in z])
    want = top - share * height
    ok, clear = slope <= np.tan(np.radians(PLUS_SLOPE)), push <= PLUS_PAD_AHEAD
    both = ok & clear
    if both.any():
        pick = z[both][np.argmin(np.abs(z[both] - want))]
    elif ok.any():
        pick = z[ok][np.argmin(push[ok])]
    elif clear.any():
        pick = z[clear][np.argmin(slope[clear])]
    else:
        pick = z[np.argmin(slope)]
    return float(pick - bottom), pitch


def plus_tilt(profile, tall=False, pitch=None):
    """g1d_pour_beans_plus source cup: (tilt it needs, fingers' pitch, most roll), deg, from its cavity (PLUS_CHORD);
    `pitch` (a grasp entry's) instead of the steepest of PLUS_PITCHES that gets there."""
    z, r = np.asarray(profile["z"])[1:], np.asarray(profile["r"])[1:]  # the floor sample reads the fillet
    dz, dr = z[None, :] - z[:, None], r[None, :] - r[:, None]
    flare = np.degrees(np.arctan2(dr, dz))[dz >= PLUS_CHORD].min() if (dz >= PLUS_CHORD).any() else 0.0
    need = float(np.clip(90.0 - flare + np.degrees(np.arctan(0.3)) + PLUS_TILT_MARGIN, *PLUS_NEED))
    pitches = [p for p in PLUS_PITCHES if not tall or p <= PLUS_PITCH[1]] if pitch is None else [pitch]
    for pitch in pitches:
        p = np.radians(pitch)
        cos_roll = (np.cos(np.radians(need)) - np.sin(p) ** 2) / np.cos(p) ** 2
        if cos_roll >= -1 and (roll := np.degrees(np.arccos(cos_roll))) <= PLUS_ROLL_NEED:
            return need, pitch, min(roll + PLUS_ROLL_SPARE, PLUS_ROLL_MAX)
    return need, pitches[-1], PLUS_ROLL_MAX


def profile_grasp(lm, out, label, profile, scale, grip_up, pitch, turn=0.0):
    """cup_grasp of a cup by its profile (scripts/objects/cup_profile.py, at `scale`): pads `grip_up` above its
    underside on the body axis (backed off up to PLUS_PAD_AHEAD when the palm or wrist camera would touch the near
    wall), opened to the widest body (handle left out) they pass sliding in along the pitched fingers. Same returns."""
    pos, quat = lm.get_instance_pose(0, label=label)
    pos, rot = as_np(pos), from_wxyz(as_np(quat))
    axis = pos + rot.apply(np.r_[np.asarray(profile["axis"]) * scale, 0.0])
    bottom, top = profile["bottom"] * scale, profile["top"] * scale
    hand = Rotation.from_euler("xz", [-pitch, out * CUP_YAW + turn], degrees=True) * GRASP_ROT
    push = plus_palm_push(profile, scale, grip_up, pitch)
    backoff = min(push + PLUS_PALM_CLEAR, PLUS_PAD_AHEAD) if push > PLUS_PALM_CLEAR else 0.0
    fwd = hand.apply([1.0, 0.0, 0.0])[:2]
    grip = np.r_[axis[:2] - backoff * fwd / np.linalg.norm(fwd), pos[2] + bottom + grip_up]
    width = plus_pad_width(profile, scale, grip_up, pitch)
    return grip, hand, pos[2] + top, cup_profile.body_radius(profile, top, scale), pad_open(width)


def handle_point(profile, scale, handle):
    """The handle grasp point (the bar's centre at the hole's height) in the cup's root frame at `scale`."""
    yaw = np.radians(handle["yaw"])
    mid = (handle["bar"][0] + handle["bar"][1]) / 2
    return np.r_[(np.asarray(profile["axis"]) + mid * np.r_[np.cos(yaw), np.sin(yaw)]) * scale, handle["z"] * scale]


def handle_fits(handle, scale):
    """The inner finger, HANDLE_PITCH down, fits the hole at `scale` (HANDLE_FIT spare each way)."""
    p = np.radians(HANDLE_PITCH)
    tall = cup_handle.FINGER[1] / np.cos(p) + handle["width"] * scale * np.tan(p)
    across, high = np.asarray(handle["room"]) * scale
    return bool(across >= cup_handle.FINGER[0] + 2 * HANDLE_FIT and high >= tall + 2 * HANDLE_FIT)


def handle_grasp(lm, out, label, profile, scale, handle):
    """profile_grasp by the cup's handle (HANDLE_PITCH): the grasp point at handle_point, the hand turned with the
    handle so that it points out along the closing axis (to side `out`), the pads open across the bar with the inner
    finger centred in the room between the bar and the body. Same returns."""
    pos, quat = lm.get_instance_pose(0, label=label)
    pos, rot = as_np(pos), from_wxyz(as_np(quat))
    grip = pos + rot.apply(handle_point(profile, scale, handle))
    yaw = np.radians(handle["yaw"])
    along = rot.apply([np.cos(yaw), np.sin(yaw), 0.0])
    turn = np.degrees(np.arctan2(along[1], along[0])) - (0.0 if out > 0 else 180.0)
    hand = Rotation.from_euler("xz", [-HANDLE_PITCH, (turn + 180) % 360 - 180], degrees=True) * GRASP_ROT
    clear = float(np.clip((handle["room"][0] * scale - cup_handle.FINGER[0]) / 2, 0.0, HANDLE_CLEAR))
    opening = ((handle["bar"][1] - handle["bar"][0]) * scale + 2 * clear - PAD_CLOSED) / PAD_TRAVEL
    top = profile["top"] * scale
    return grip, hand, pos[2] + top, cup_profile.body_radius(profile, top, scale), float(np.clip(opening, 0.0, 1.0))


def plus_pad_width(profile, scale, grip_up, pitch):
    """Widest body (m, handle left out) the pads pass sliding in along the pitched fingers to `grip_up`."""
    bottom, top = profile["bottom"] * scale, profile["top"] * scale
    path = bottom + grip_up + np.linspace(-PLUS_PAD / 2, CUP_SLIDE * np.sin(np.radians(pitch)) + PLUS_PAD / 2, 8)
    return 2 * max(cup_profile.body_radius(profile, min(zi, top), scale) for zi in path)


def pour_actions(env, fluid_label="wine", dry=TILT_DRY, pour_target=POUR_TARGET, hold_cup=True, rate=1.0):
    """Jug and cup as the module docstring says; fast tilt up to `dry`, then pour until `pour_target` of the jug's
    fill has left it. Water runs from ~45 deg, the beans from TILT_DRY. Without `hold_cup` the cup stays on the table
    and the other arm stays where it is. `rate` scales the tilt speeds."""
    lm = env.scene_manager.layout_manager
    pos, quat = lm.get_instance_pose(0, label="bottle")
    pitcher_pos, pitcher_rot = as_np(pos), from_wxyz(as_np(quat))
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, fluid_label))

    arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
    arm = min(arms, key=lambda a: abs(a.grasp_point()[0] - pitcher_pos[0]))
    cup_arm = next(a for a in arms if a is not arm)
    cup0 = as_np(lm.get_instance_pose(0, label="cup")[0])
    side = np.sign(cup0[0] - pitcher_pos[0])  # the spout faces the cup (build_asset.py)
    cup_grip, cup_hand, rim0, cup_r, cup_open = cup_grasp(lm, side)  # the cup arm is beyond the cup
    held_grip = cup_grip + ([-side * CUP_MEET, 0, CUP_LIFT] if hold_cup else 0.0)
    meet = held_grip[:2]
    rim = rim0 + CUP_LIFT

    grasp = pitcher_pos + pitcher_rot.apply(GRASP_IN_PITCHER)
    corners = np.array(list(itertools.product(*zip(-BBOX_HALF, BBOX_HALF)))) - GRASP_IN_PITCHER

    def pour(theta, centre=meet, top=rim):
        """Grasp point and hand rotation tilting the pitcher about its lip, held over the cup; and the lip target."""
        tilt = Rotation.from_rotvec([0, side * theta, 0])
        lip = (tilt * pitcher_rot).apply(SPOUT_IN_PITCHER - GRASP_IN_PITCHER)
        low = (tilt * pitcher_rot).apply(corners)[:, 2].min() - lip[2]
        target = np.array([centre[0] - side * POUR_BACKOFF, centre[1], top + max(SPOUT_CLEAR, BODY_CLEAR - low)])
        return target - lip, tilt * GRASP_ROT, target

    start = pour(0)[0]
    mid = np.r_[grasp[:2], start[2]]  # straight up first: pulled sideways while standing, the jug tips over

    def carry(a, b):
        return max(int(np.ceil(np.linalg.norm(b - a) / JUG_SPEED)), 12)

    jug_keys = [
        (grasp + BEHIND + [0, 0, 0.1], GRASP_ROT, OPEN, 30),
        (grasp + BEHIND, GRASP_ROT, OPEN, 20),
        (grasp + BEHIND, GRASP_ROT, OPEN, 8),  # settle
        (grasp, GRASP_ROT, OPEN, 48),
        (grasp, GRASP_ROT, 0, 18),  # close
        (mid, GRASP_ROT, 0, carry(grasp, mid)),
        (start, GRASP_ROT, 0, carry(mid, start)),
    ]
    cup_side = cup_grip - cup_hand.apply([CUP_SLIDE, 0, 0])
    cup_keys = [
        (cup_side + [0, 0, HOVER], cup_hand, cup_open, 54),
        (cup_side, cup_hand, cup_open, 30),
        (cup_grip, cup_hand, cup_open, 40),  # slide in
        (cup_grip, cup_hand, 0, 15),  # close
        (cup_grip + [0, 0, CUP_LIFT], cup_hand, 0, 18),
        (held_grip, cup_hand, 0, 20),  # carry toward the jug
    ]
    check = env.pour_check()

    def beans():
        return np.asarray(fluid.get_particle_positions()[0]).reshape(-1, 3) + fluid.init_pos

    def jug():
        jug_pos, jug_quat = lm.get_instance_pose(0, label="bottle")
        return as_np(jug_pos), from_wxyz(as_np(jug_quat))

    def left_jug():
        """Fraction of the beans outside the jug's bounding box (in the cup, in flight or spilled)."""
        jug_pos, jug_rot = jug()
        return np.mean(np.any(np.abs(jug_rot.inv().apply(beans() - jug_pos)) > BBOX_HALF, axis=1))

    axis_off = from_wxyz(as_np(lm.get_instance_pose(0, label="cup")[1])).inv().apply(cup_grip - cup0) * [1, 1, 0]
    cup_box = lm.get_instance_bbox_vertices(lm.get_instance_name(0, "cup"), 0)
    cup_box = cup_box[cup_box[:, 2] > cup_box[:, 2].max() - 1e-6]  # rim corners: the highest rim point of a leaning cup

    def in_cup():
        """Fraction of the beans inside the cup's rim cylinder, for the log."""
        pos, quat = lm.get_instance_pose(0, label="cup")
        pos, rot = as_np(pos), from_wxyz(as_np(quat))
        local = rot.inv().apply(beans() - pos) - axis_off
        return np.mean((np.hypot(local[:, 0], local[:, 1]) < cup_r) & (local[:, 2] < rim0 - cup0[2]))

    def act(jug_target, cup_target):
        nonlocal step
        if step % 10 == 0:
            jug_pos, jug_rot = jug()
            tilt = np.degrees(np.arccos(jug_rot.apply([0, 0, 1])[2]))
            lean = np.degrees(np.arccos(from_wxyz(as_np(lm.get_instance_pose(0, label="cup")[1])).apply([0, 0, 1])[2]))
            print(
                f"[scripted_demo] {step:3d} err jug {np.linalg.norm(arm.grasp_point() - jug_target[0]):.3f}"
                f" cup {np.linalg.norm(cup_arm.grasp_point() - cup_target[0]):.3f} jug {jug_pos.round(3)}"
                f" tilt {tilt:.0f} cup lean {lean:.0f} left jug {left_jug():.0%} in cup {in_cup():.0%}"
                f" check {env.reward_manager.call_func_parser(check, 0):.0f}",
                flush=True,
            )
        step += 1
        return {
            arm.arm_key: arm.ik(*jug_target[:2]),
            arm.grip_key: [float(jug_target[2])],
            cup_arm.arm_key: cup_arm.ik(*cup_target[:2]),
            cup_arm.grip_key: [float(cup_target[2])],
        }

    print(
        f"[scripted_demo] jug {pitcher_pos.round(3)} cup {cup0.round(3)} rim {rim0:.3f} jug arm {arm.arm_key}"
        f" cup arm {cup_arm.arm_key} grip {cup_grip.round(3)} open {cup_open:.2f}"
    )
    step = 0
    jug_path = list(waypoints(arm.grasp_point(), arm.pose()[1], jug_keys))
    idle = (cup_arm.grasp_point(), cup_arm.pose()[1], 1.0)
    cup_path = list(waypoints(cup_arm.grasp_point(), cup_arm.pose()[1], cup_keys)) if hold_cup else [idle]
    for j, c in itertools.zip_longest(jug_path, cup_path):
        yield act(j or jug_path[-1], c or cup_path[-1])
    held = as_np(lm.get_instance_pose(0, label="cup")[0])
    print(f"[scripted_demo] cup held at {held.round(3)} (from {cup0.round(3)})")

    fix = np.zeros(3)  # the jug slips in the handle grip: steer the actual spout over where the cup actually is

    def steer(theta):
        nonlocal fix
        phi = CUP_TILT * min(theta / TILT_DRY, 1.0)
        lean = Rotation.from_rotvec([0, -side * phi, 0])  # mouth toward the jug
        # the hand sits above the cup and would swing under the tilted jug: move the cup away so the hand stays put
        palm = cup_hand.apply([-PAD_REACH, 0, 0])
        shift = palm - lean.apply(palm)
        held = (held_grip + [shift[0], shift[1], 0], lean * cup_hand, 0)
        pos, quat = lm.get_instance_pose(0, label="cup")
        pos, rot = as_np(pos), from_wxyz(as_np(quat))
        top = (pos + rot.apply(cup_box))[:, 2].max()
        mouth = pos + rot.apply(axis_off + [0, 0, rim0 - cup0[2]])  # a leaning mouth moves toward the jug
        hand, hand_rot, lip_target = pour(theta, mouth[:2], top)
        jug_pos, jug_rot = jug()
        fix = np.clip(
            fix + SPOUT_GAIN * (lip_target - jug_pos - jug_rot.apply(SPOUT_IN_PITCHER)), -SPOUT_SLACK, SPOUT_SLACK
        )
        return act((hand + fix, hand_rot, 0), held if hold_cup else idle)

    returning = int(np.ceil(TILT / (rate * RETURN_RATE))) + STEP_MARGIN
    theta, prev = 0.0, 0.0  # tilt until enough beans have left the jug, then back upright
    while theta < dry:
        theta = min(theta + rate * DRY_RATE, dry)
        yield steer(theta)
    while (now := left_jug()) < pour_target and step < env.step_lim - returning:
        if now - prev < FLOW:  # tilt further only while the stream is weak: a surge overshoots the cup
            theta = min(theta + rate * TILT_RATE, TILT)
        prev = now
        yield steer(theta)
    while theta > 0:
        theta = max(theta - rate * RETURN_RATE, 0.0)
        yield steer(theta)
    for _ in range(STEP_MARGIN):
        yield steer(0.0)


def cup_to_cup_actions(
    env,
    pour_share=None,
    place=False,
    meet=None,
    lift=None,
    rim_down=None,
    dry_rate=C2C_DRY_RATE,
    carry_speed=CUP_SPEED,
    arrive_above=0.0,
    profiles=None,
    grips=None,
):
    """Each arm side-grasps the cup on its side and lifts it; the target cup is held up in front of the chest and the
    source cup rolls about its rim lip over the target's mouth until `pour_share` (default C2C_POUR) of the beans have
    gone over; back upright. With `place`, both cups then go back down where they stood and the hands open and back
    off. `meet` (x to the target's side, y ahead; m) puts the target cup's mid-air axis relative to the robot base
    instead of at the world C2C_MEET point; `lift` (m) replaces the target kind's C2C_LIFT; `rim_down` overrides
    C2C_RIM_DOWN entries; `dry_rate` (rad per action) is the first roll's rate, up to C2C_DRY; `carry_speed` (m per
    action) the cups' carry speed; `arrive_above` (m): the source cup comes in this far above its pour start and then
    goes straight down to it. `profiles` ({label: (cup profile, scale)}, scripts/objects/cup_profile.py) takes each
    cup's grip, pitch, lift and cavity from its profile instead of its kind (cups of any shape); the cup frame is then
    the cup's root, its centre at axes[label]. `grips` ({label: {rim_down, pitch}}, with `profiles`) replaces a
    cup's grip height share and pitch (plus_grip, plus_tilt); {mode: "handle", handle} holds it by the handle
    (handle_grasp), a target with its axis at the meet point."""
    pour_share = C2C_POUR if pour_share is None else pour_share
    rim_down = {**C2C_RIM_DOWN, **(rim_down or {})}
    lm = env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
    src0 = as_np(lm.get_instance_pose(0, label="source_cup")[0])
    tgt0 = as_np(lm.get_instance_pose(0, label="target_cup")[0])
    start_pos = {"source_cup": src0, "target_cup": tgt0}
    side = np.sign(src0[0])  # the source cup's side; the target is on the other one
    arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
    src_arm = min(arms, key=lambda a: abs(a.grasp_point()[0] - src0[0]))
    tgt_arm = next(a for a in arms if a is not src_arm)
    heights, kinds, turns, axes, pitches = {}, {}, {}, {}, {}
    grips = {label: (grips or {}).get(label) or {} for label in ("source_cup", "target_cup")}
    handles = {label: g["handle"] for label, g in grips.items() if g.get("mode") == "handle"}
    if profiles:
        s_cav, s_sc = profiles["source_cup"]
        need, s_pitch, most = plus_tilt(
            s_cav,
            (s_cav["top"] - s_cav["bottom"]) * s_sc > PLUS_TALL,
            HANDLE_PITCH if "source_cup" in handles else grips["source_cup"].get("pitch"),
        )
        print(f"[scripted_demo] source needs {need:.0f} deg of tilt: pitch {s_pitch}, roll to {most:.0f}")
        most = np.radians(most)
    for label, out in (("source_cup", side), ("target_cup", -side)) if profiles else ():
        cavity, scale = profiles[label]
        heights[label] = (cavity["top"] - cavity["bottom"]) * scale
        if label in handles:
            heights[label + "_grip"] = handle_point(cavity, scale, handles[label])[2] - cavity["bottom"] * scale
            pitches[label] = HANDLE_PITCH
        else:
            want_pitch = s_pitch if label == "source_cup" else grips[label].get("pitch")
            heights[label + "_grip"], pitches[label] = plus_grip(
                cavity, scale, grips[label].get("rim_down"), want_pitch
            )
        rot = from_wxyz(as_np(lm.get_instance_pose(0, label=label)[1]))
        kinds[label] = None
        turns[label] = flat_turn(rot, out) if lm.get_instance_name(0, label).startswith("measuring_cup") else 0.0
        axes[label] = np.r_[np.asarray(cavity["axis"]) * scale, (cavity["bottom"] + cavity["top"]) / 2 * scale]
    for label, out in (("source_cup", side), ("target_cup", -side)) if not profiles else ():
        name = lm.get_instance_name(0, label)
        bbox = lm.get_instance_bbox_vertices(name, 0)
        heights[label] = bbox[:, 2].max() - bbox[:, 2].min()
        kinds[label] = kind = next(c for c in rim_down if name.startswith(c))
        heights[label + "_grip"] = heights[label] - rim_down[kind]
        rot = from_wxyz(as_np(lm.get_instance_pose(0, label=label)[1]))
        turns[label] = flat_turn(rot, out) if kind == "measuring_cup" else 0.0
        circle = (lm.get_instance_metadata(0, label=label) or {}).get("active", {}).get("place", {}).get("up", {})
        center = circle.get("contact_circle", {}).get("center", (0.0, 0.0))  # cup axis in the cup (bbox centre) frame
        axes[label] = np.r_[center[0], center[1], 0.0]
    if profiles:
        lift = float(np.clip(PLUS_MOUTH - heights["target_cup"], *PLUS_LIFT)) if lift is None else lift
        grasps = [
            handle_grasp(lm, out, label, *profiles[label], handles[label])
            if label in handles
            else profile_grasp(lm, out, label, *profiles[label], heights[label + "_grip"], pitches[label], turns[label])
            for label, out in (("source_cup", side), ("target_cup", -side))
        ]
    else:
        lift = C2C_LIFT[kinds["target_cup"]] if lift is None else lift
        grasps = [
            cup_grasp(lm, out, label, heights[label + "_grip"], C2C_PITCH[kinds[label]], turns[label])
            for label, out in (("source_cup", side), ("target_cup", -side))
        ]
    (s_grip, s_hand, s_rim, s_r, s_open), (t_grip, t_hand, t_rim, t_r, t_open) = grasps
    height, t_height = heights["source_cup"], heights["target_cup"]
    up = heights["source_cup_grip"]  # pads above the cup bottom
    centre = axes["source_cup"] if profiles else np.zeros(3)  # of the source cup in its frame

    # the cup rolls about the fingers (the wrist roll); about world y the 35-deg-down hand runs the arm into its joint
    # limits within 15 deg, and fingers forward leave too little roll to empty the cup
    roll_axis = s_hand.apply([1.0, 0.0, 0.0])
    pour_dir = -side * np.cross(roll_axis, [0.0, 0.0, 1.0])  # the way the mouth tips: toward the target, slanted
    pour_dir /= np.linalg.norm(pour_dir)
    if meet is None:
        t_air = np.array([-side * C2C_MEET_X, C2C_MEET_Y, t_grip[2] + lift])
    else:
        root = src_arm.root()
        t_air = np.array([root[0] - side * meet[0], root[1] + meet[1], t_grip[2] + lift])
    if profiles or "target_cup" in handles:  # the target's axis goes there, not a handle or backed-off grip
        t_pos, t_quat = lm.get_instance_pose(0, label="target_cup")
        t_axis = as_np(t_pos) + from_wxyz(as_np(t_quat)).apply(np.r_[axes["target_cup"][:2], 0.0])
        t_air[:2] += (t_grip - t_axis)[:2]
    # source cup frame (origin at the bbox centre): its rim point that tips lowest, its bbox corners; the grasp point
    # and hand rotation in it, as planned until measured in mid-air (the cup sits lower or turned in the soft grip)
    lip_c = centre + np.r_[s_r * pour_dir[:2], height / 2]
    b_r = max(profiles["source_cup"][0]["body"]["r"]) * profiles["source_cup"][1] if profiles else s_r
    corners_c = centre + np.array(list(itertools.product((-b_r, b_r), (-b_r, b_r), (-height / 2, height / 2))))
    cup_rot, grasp_c, hand_rot = Rotation.identity(), centre + np.array([0.0, 0.0, up - height / 2]), s_hand
    if profiles or "source_cup" in handles:
        grasp_c = s_grip - src0
    aim, clear = 0.0, LIP_CLEAR  # lip target moved along the pour, height over the rim (the spill feedback)
    eased = [[aim, 0.0], [clear, 0.0]]  # the plus task's aim and clear as commanded (plus_ease), [value, rate] each

    def pour(theta, mouth, rim):
        """Source grasp point and hand rotation rolling the cup about its lip over the target mouth; the lip target."""
        tilt = Rotation.from_rotvec(-side * theta * roll_axis)
        rot = tilt * cup_rot
        low = rot.apply(corners_c - lip_c)[:, 2].min()
        a, c = (eased[0][0], eased[1][0]) if profiles else (aim, clear)
        lip = np.r_[mouth - (t_r - LIP_IN - a) * pour_dir[:2], rim + max(c, BODY_CLEAR - low)]
        return lip + rot.apply(grasp_c - lip_c), tilt * hand_rot, lip

    def carry(a, b):
        return max(int(np.ceil(np.linalg.norm(b - a) / carry_speed)), 12)

    s_start = pour(0.0, t_air[:2], t_rim + lift)[0]
    keys = []
    for grip, hand, opening, air in ((s_grip, s_hand, s_open, s_start), (t_grip, t_hand, t_open, t_air)):
        approach = grip - hand.apply([CUP_SLIDE, 0, 0])
        raised = grip + [0, 0, C2C_RAISE]
        # the source first rises to its pour height over its own place, then moves across: its bottom is then above
        # the target's rim (pour() keeps BODY_CLEAR), where a straight carry brushed the target cup in ~40% of the
        # ShenZhen episodes (2026-10-21)
        over = air + [0, 0, arrive_above if grip is s_grip else 0.0]
        high = np.r_[raised[:2], max(raised[2], over[2])] if grip is s_grip else raised
        keys.append(
            [
                (approach + [0, 0, HOVER], hand, opening, 54),
                (approach, hand, opening, 30),
                (grip, hand, opening, 40),  # slide in
                (grip, hand, 0, 15),  # close
                (raised, hand, 0, max(carry(grip, raised), PLUS_LIFT_ACTS if profiles else 0)),  # straight up first
                (high, hand, 0, carry(raised, high)),
                (over, hand, 0, carry(high, over)),
                *([(air, hand, 0, carry(over, air))] if arrive_above else []),
            ]
        )

    def cup(label):
        pos, quat = lm.get_instance_pose(0, label=label)
        return as_np(pos), from_wxyz(as_np(quat))

    def beans():
        return np.asarray(fluid.get_particle_positions()[0]).reshape(-1, 3) + fluid.init_pos

    def inside(label):
        """Per bean: inside the cup (cup frame, origin at the bbox centre): the measuring cups' and the mug's actual
        cavity (measuring-cup corners reach past the rim cylinder: beans left there read as poured), else the rim
        cylinder."""
        pos, rot = cup(label)
        if profiles:
            return cup_profile.in_cavity(beans(), pos, np.roll(rot.as_quat(), 1), *profiles[label])
        local = rot.inv().apply(beans() - pos)
        r, h = (s_r, height) if label == "source_cup" else (t_r, t_height)
        if kinds[label] == "measuring_cup":
            return cup_shape.in_cavity(local - cup_shape.origin_shift(h), h)
        if kinds[label] == "plastic_mug":
            return mug_shape.in_cavity(local - mug_shape.origin_shift())
        off = local - axes[label]
        return (np.hypot(off[:, 0], off[:, 1]) < r) & (np.abs(local[:, 2]) < h / 2)

    check = env.pour_check()
    step = 0
    last = {}  # each arm's last joint command

    def joints(arm, target):
        q = arm.ik(*target[:2], smooth=bool(profiles))
        if profiles and arm.arm_key in last:
            q = plus_joint_step(q, last[arm.arm_key])
        last[arm.arm_key] = q
        return q

    def act(src_target, tgt_target):
        nonlocal step
        if step % 10 == 0:
            tilt = np.degrees(np.arccos(cup("source_cup")[1].apply([0, 0, 1])[2]))
            print(
                f"[scripted_demo] {step:3d} err src {np.linalg.norm(src_arm.grasp_point() - src_target[0]):.3f}"
                f" tgt {np.linalg.norm(tgt_arm.grasp_point() - tgt_target[0]):.3f}"
                f" src {cup('source_cup')[0].round(3)} tgt {cup('target_cup')[0].round(3)} tilt {tilt:.0f}"
                f" left src {1 - inside('source_cup').mean():.0%} in tgt {inside('target_cup').mean():.0%}"
                f" check {env.reward_manager.call_func_parser(check, 0):.0f}",
                flush=True,
            )
        step += 1
        return {
            src_arm.arm_key: joints(src_arm, src_target),
            src_arm.grip_key: [float(src_target[2])],
            tgt_arm.arm_key: joints(tgt_arm, tgt_target),
            tgt_arm.grip_key: [float(tgt_target[2])],
        }

    print(
        f"[scripted_demo] source {src0.round(3)} with {src_arm.arm_key} grip {s_grip.round(3)} open {s_open:.2f}"
        f" turn {turns['source_cup']:.0f};"
        f" target {as_np(lm.get_instance_pose(0, label='target_cup')[0]).round(3)} with {tgt_arm.arm_key}"
        f" grip {t_grip.round(3)} open {t_open:.2f} turn {turns['target_cup']:.0f}"
    )
    s_path = list(waypoints(src_arm.grasp_point(), src_arm.pose()[1], keys[0], bool(profiles)))
    t_path = list(waypoints(tgt_arm.grasp_point(), tgt_arm.pose()[1], keys[1], bool(profiles)))
    pickup_q = {}
    lift_done = sum(n for *_, n in keys[0][:5]), sum(n for *_, n in keys[1][:5])
    for idx_p, (s, t) in enumerate(itertools.zip_longest(s_path, t_path), 1):
        cmd = act(s or s_path[-1], t or t_path[-1])
        if idx_p == lift_done[0]:
            pickup_q[src_arm.arm_key] = src_arm.q()
        if idx_p == lift_done[1]:
            pickup_q[tgt_arm.arm_key] = tgt_arm.q()
        yield cmd

    hold = (t_air, t_hand, 0)
    fix = np.zeros(3)  # the cup keeps slipping a little: steer the actual lip over the actual target mouth
    handover = None  # the carry's last command less the pour's first at no roll (position, rotation vector), actions
    roll = [0.0, 0.0]  # the roll commanded and its rate (plus_ease)

    def steer(theta, jolt=0.0):
        nonlocal fix, handover
        if profiles:
            theta = plus_ease(roll, theta)
            for state, target in zip(eased, (aim, clear)):
                plus_ease(state, target, PLUS_LIP_ACCEL, PLUS_LIP_RATE)
        t_pos, t_rot = cup("target_cup")
        mouth = t_pos + t_rot.apply(axes["target_cup"] + [0.0, 0.0, t_height / 2])  # on the cup axis, not the bbox
        hand, hand_rot, lip_target = pour(theta, mouth[:2], mouth[2])
        if handover is None:
            handover = [np.zeros(3), np.zeros(3), PLUS_HANDOVER]
            if profiles:
                zero, zero_rot = pour(0.0, mouth[:2], mouth[2])[:2]
                handover = [s_air[0] - zero, (s_hand * zero_rot.inv()).as_rotvec(), 0]
        u = handover[2] / PLUS_HANDOVER
        ease = 1.0 - u * u * (3 - 2 * u)
        handover[2] = min(handover[2] + 1, PLUS_HANDOVER)
        if not ease:
            s_pos, s_rot = cup("source_cup")
            fix = np.clip(fix + LIP_GAIN * (lip_target - s_pos - s_rot.apply(lip_c)), -LIP_SLACK, LIP_SLACK)
        hand = hand + fix + ease * handover[0] + jolt * np.r_[pour_dir[:2], 0.0]
        return act((hand, Rotation.from_rotvec(ease * handover[1]) * hand_rot, 0), hold)

    s_air = (s_path[-1][0], s_hand, 0)
    for _ in range(C2C_SETTLE):
        if max(np.linalg.norm(a.grasp_point() - p[0]) for a, p in ((src_arm, s_air), (tgt_arm, hold))) < C2C_SETTLE_ERR:
            break
        yield act(s_air, hold)
    pos, cup_rot = cup("source_cup")
    grasp_c = cup_rot.inv().apply(src_arm.grasp_point() - pos)
    hand_rot = src_arm.pose()[1]
    toward = cup_rot.inv().apply(np.r_[pour_dir[:2], 0.0])[:2]  # the cup may have turned about its axis
    lip_c = centre + np.r_[s_r * toward / np.linalg.norm(toward), height / 2]
    print(
        f"[scripted_demo] source grasp point in the cup {grasp_c.round(3)}, planned z {centre[2] + up - height / 2:.3f}"
    )
    most = most if profiles else C2C_TILT
    returning = int(np.ceil(most / RETURN_RATE)) + STEP_MARGIN
    returning += int(2 * RETURN_RATE / PLUS_ROLL_ACCEL) + 1 if profiles else 0
    returning += PLUS_PLACE_ACTS if place and profiles else 0
    total = len(inside("source_cup"))
    if profiles:  # the stream the target mouth takes, in grains per action
        grain = getattr(fluid, "grain", None) or cup_profile.PBD_GRAIN
        t_cavity, t_scale = profiles["target_cup"]
        t_in = t_cavity["r"][-1] * t_scale
        sections = (2 * t_in / (6 * grain["volume"] / np.pi) ** (1 / 3)) ** 2
        surge_grains = float(np.clip(PLUS_SURGE_MOUTH * sections, *PLUS_SURGE_GRAINS))
        flow, surge = PLUS_WEAK * surge_grains / total, surge_grains / total
        print(f"[scripted_demo] target mouth takes {surge_grains:.1f} grains an action")
    else:  # a few large grains: a weak stream is no grain an action, a surge three
        flow, surge = max(C2C_FLOW, 0.5 / total), max(PLUS_SURGE, 2.5 / total)

    def fallen():
        """Per bean: spilled (out of both cups, under the target rim, off its mouth); its offset along the pour from
        the mouth centre."""
        pts = beans()
        t_pos, t_rot = cup("target_cup")
        mouth = t_pos + t_rot.apply(axes["target_cup"] + [0.0, 0.0, t_height / 2])
        off = pts[:, :2] - mouth[:2]
        out = ~inside("source_cup") & ~inside("target_cup") & (pts[:, 2] < mouth[2] - PLUS_SPILL_DROP)
        return out & (np.hypot(off[:, 0], off[:, 1]) > t_r), off @ pour_dir[:2]

    seen = [None, -1, None]  # bean positions at the last prediction, its step, its result

    def predicted():
        """Per falling bean: predicted to land off the target mouth (PLUS_PREDICT_PAD), or within PLUS_PREDICT_WARN of
        that; its landing's offset along the pour from the mouth centre. Once a step."""
        last, at, result = seen
        if step == at:
            return result
        pts = beans()
        vel = np.zeros_like(pts) if last is None else (pts - last) / ((step - at) * PLUS_ACTION_DT)
        t_pos, t_rot = cup("target_cup")
        mouth = t_pos + t_rot.apply(axes["target_cup"] + [0.0, 0.0, t_height / 2])
        free = ~inside("source_cup") & ~inside("target_cup") & (pts[:, 2] > mouth[2]) & (vel[:, 2] < -PLUS_PREDICT_FALL)
        off = plus_landing(pts, vel, mouth)
        reach = np.hypot(off[:, 0], off[:, 1]) - t_in - PLUS_PREDICT_PAD
        seen[:] = pts, step, (free & (reach > 0), free & (reach > -PLUS_PREDICT_WARN), off @ pour_dir[:2])
        return seen[2]

    went = [None, -1, 0.0]  # bean positions in the source frame (scale 1) at the last forecast, its step, its result

    def coming():
        """Share of the beans an action due to leave the source in the next PLUS_FORECAST actions. Once a step."""
        last, at, result = went
        if step == at:
            return result
        pos, rot = cup("source_cup")
        local = cup_profile.local_points(beans(), pos, np.roll(rot.as_quat(), 1), profiles["source_cup"][1])
        if last is not None:
            result = float(plus_due(local, last, step - at, *profiles["source_cup"]).sum()) / PLUS_FORECAST / total
        went[:] = local, step, result
        return result

    theta, prev, best, stall, shakes = 0.0, 0.0, 0.0, 0, 0
    leaked = gone = 1 - inside("source_cup").mean()  # gone: out of the source before the pour
    flagged = np.zeros(total, bool)  # beans predicted (or seen) off the target mouth (the plus task)
    if profiles:  # stray: out of both cups before the pour (a carry leak) or since through the source wall
        stray = ~inside("source_cup") & ~inside("target_cup")
        flagged |= stray
        was_in = [inside("source_cup")]

    def leak():
        """Mark the beans that left the source through its wall since the last call stray."""
        now_in = inside("source_cup")
        out = was_in[0] & ~now_in
        was_in[0] = now_in
        if out.any():
            pos, rot = cup("source_cup")
            s_cavity, s_scale = profiles["source_cup"]
            local = cup_profile.local_points(beans()[out], pos, np.roll(rot.as_quat(), 1), s_scale)
            stray[np.flatnonzero(out)[local[:, 2] < s_cavity["z"][-1] - PLUS_WALL]] = True

    edge, due, held = False, 0.0, [0, 0]  # a falling bean due near the target rim; coming(); actions held for each
    while theta < C2C_DRY:
        theta = min(theta + dry_rate, C2C_DRY)
        yield steer(theta)
        prev = 1 - inside("source_cup").mean()
        if not profiles:
            continue
        s_pos, s_rot = cup("source_cup")
        if beans()[:, 2].max() > (s_pos + s_rot.apply(lip_c))[2] - PLUS_LIP_NEAR:  # a full cup runs early
            break
        leak()
        miss, due = predicted()[0] & ~stray, coming()
        if theta < PLUS_SPILL_FROM:
            leaked, flagged = prev, flagged | miss
        elif max(prev - leaked, due) > flow or (miss & ~flagged).any():
            break
    lowest = theta
    spilled, calm = (fallen()[0] | stray if profiles else None), 0
    onset, late, end = None, False, env.step_lim - returning  # the step PLUS_PACE_ONSET had gone; behind the pace
    if profiles:
        flagged |= spilled
    while (now := 1 - inside("source_cup").mean()) < pour_share and step < end:
        if profiles:
            onset = step if onset is None and now - gone >= PLUS_PACE_ONSET else onset
            if not late and onset is not None:
                late = now - gone < PLUS_PACE * (pour_share - gone) * (step - onset - PLUS_PACE_GRACE) / (end - onset)
                if late:
                    print(f"[scripted_demo] {step:3d} poured {now - gone:.0%}, behind the pace: only seen spills count")
            leak()
            new = (fell := fallen())[0] & ~spilled & ~stray
            spilled |= new
            miss, near, along = predicted()
            miss, near = miss & ~stray, near & ~stray
            ahead = new if late else (miss | new) & ~flagged  # predicted, or spilled unpredicted
            flagged |= miss | new
            edge, due = (False, 0.0) if late else (bool(near.any()), coming())
        if profiles and ahead.any() and theta >= PLUS_SPILL_FROM:
            off = float(np.where(miss, along, fell[1])[ahead].mean())
            if not calm:  # a new spell
                clear = max(clear - PLUS_SPILL_LOWER, PLUS_CLEAR_MIN)
                aim = float(np.clip(aim - np.clip(PLUS_AIM_GAIN * off, -PLUS_AIM_STEP, PLUS_AIM_STEP), *PLUS_AIM))
                floor = theta - PLUS_SPILL_DEPTH
            theta, calm = max(theta - PLUS_SPILL_BACK, floor, lowest), PLUS_SPILL_CALM
            print(
                f"[scripted_demo] {step:3d} {ahead.sum()} off the mouth ({(ahead & miss).sum()} predicted)"
                f" {off:+.3f} along the pour: tilt back to {np.degrees(theta):.0f}, lip {aim:+.3f} in,"
                f" {clear:.3f} over the rim"
            )
        rate = C2C_TILT_RATE * (PLUS_SLOW_RATE if profiles and theta > PLUS_SLOW else 1.0)
        if calm:
            calm -= 1
        elif max(now - prev, due) < flow and not edge:  # only while the stream (and the one coming) is weak, clear
            theta = min(theta + rate, most)
        elif profiles and now - prev > surge:  # and back while it surges: a slug splashes off the target mouth
            theta = max(theta - rate, lowest)
        elif edge:
            held[1] += 1
        elif due >= flow > now - prev:
            held[0] += 1
        prev = now
        best, stall = (now, 0) if now > best or theta < most else (best, stall + 1)
        if profiles and stall > PLUS_STALL:
            if shakes == PLUS_SHAKES:
                break
            shakes, stall = shakes + 1, 0
            for k in range(1, PLUS_SHAKE_ACTS + 1):
                wave = np.sin(2 * np.pi * k / PLUS_SHAKE_PERIOD) * np.sin(np.pi * k / (PLUS_SHAKE_ACTS + 1))
                yield steer(theta, PLUS_SHAKE * wave)
        yield steer(theta)
    if profiles:
        print(
            f"[scripted_demo] {step:3d} poured {now:.0%}: tilt held {held[0]} actions for the stream coming,"
            f" {held[1]} for beans due near the rim"
        )
    while theta > 0:
        theta = max(theta - RETURN_RATE, 0.0)
        yield steer(theta)
    for _ in range(int(2 * RETURN_RATE / PLUS_ROLL_ACCEL) + 1 if profiles else 0):  # plus_ease catching up
        if not roll[0]:
            break
        yield steer(0.0)
    for _ in range(STEP_MARGIN):
        yield steer(0.0)
    if not place:
        return
    if profiles:
        # Rise clear of the target mouth first, then unwind each arm in joint space back to its healthy post-lift
        # configuration (where source_cup is already over its own desk spot above the target rim) before Cartesian
        # descent: otherwise after a 110-155 deg pour roll differential IK gets trapped in an elbow-up branch (g1d/1)
        s_now = src_arm.grasp_point()
        s_up = s_now + [0.0, 0.0, PLACE_RISE]
        for pt, rt, gp in waypoints(s_now, src_arm.pose()[1], [(s_up, s_hand, 0, carry(s_now, s_up))], True):
            yield act((pt, rt, gp), hold)
        q_starts = {src_arm.arm_key: src_arm.q(), tgt_arm.arm_key: tgt_arm.q()}
        unwind_steps = max(
            15,
            *(
                int(np.ceil(np.abs(pickup_q.get(a.arm_key, q_starts[a.arm_key]) - q_starts[a.arm_key]).max() / 0.04))
                for a in (src_arm, tgt_arm)
            ),
        )
        for i in range(1, unwind_steps + 1):
            u = i / unwind_steps
            s_u = u**3 * (10 - 15 * u + 6 * u * u)
            cmd = {}
            for a in (src_arm, tgt_arm):
                q0 = q_starts[a.arm_key]
                q1 = pickup_q.get(a.arm_key, q0)
                q = plus_joint_step(q0 + (q1 - q0) * s_u, last.get(a.arm_key, q0))
                last[a.arm_key] = q
                cmd[a.arm_key] = q
                cmd[a.grip_key] = [0.0]
            step += 1
            yield cmd
    paths = []
    for label, arm, grip, hand, opening in (
        ("source_cup", src_arm, s_grip, s_hand, s_open),
        ("target_cup", tgt_arm, t_grip, t_hand, t_open),
    ):
        now = arm.grasp_point()
        if profiles:
            c_pos, c_rot = cup(label)
            c_up = c_rot.apply([0.0, 0.0, 1.0])
            arm_rot = arm.pose()[1]
            # Level minor in-hand pitch/roll while keeping fingers pitched >= 15 deg below horizontal so table-height
            # descent stays well inside the elbow / wrist soft limits (URDF reach probe: < 12 deg stalls above the desk)
            place_hand = Rotation.from_rotvec(np.cross(c_up, [0.0, 0.0, 1.0])) * arm_rot
            fwd = place_hand.apply([1.0, 0.0, 0.0])
            cur_pitch = float(np.degrees(np.arcsin(np.clip(-fwd[2], -1.0, 1.0))))
            min_pitch = min(pitches[label], 15.0)
            if cur_pitch < min_pitch:
                horiz = np.cross([0.0, 0.0, 1.0], fwd)
                horiz /= max(np.linalg.norm(horiz), 1e-9)
                place_hand = Rotation.from_rotvec(np.radians(min_pitch - cur_pitch) * horiz) * place_hand
            grip_offset = (place_hand * arm_rot.inv()).apply(now - c_pos)
            cup_down = start_pos[label] + [0.0, 0.0, PLACE_CLEAR]
            down = cup_down + grip_offset
            over = np.r_[down[:2], max(now[2], down[2] + C2C_RAISE)]
            rel_open = 1.0
            retreat_dir = np.r_[place_hand.apply([1.0, 0.0, 0.0])[:2], 0.0]
            retreat_dir /= max(np.linalg.norm(retreat_dir), 1e-9)
            retreat = down - (PLACE_BACK + 0.03) * retreat_dir
            keys = [
                (over, place_hand, 0, carry(now, over)),
                (down, place_hand, 0, carry(over, down) + 15),
                (down, place_hand, rel_open, PLUS_PLACE_OPEN),
                (retreat, place_hand, rel_open, carry(down, retreat)),
                (retreat + [0.0, 0.0, PLACE_BACK], place_hand, rel_open, 15),
            ]
        else:
            up = now + [0.0, 0.0, 0.0]
            over = np.r_[
                grip[:2], max(up[2], grip[2] + C2C_RAISE)
            ]  # back over its place at the carry height, then down
            down = grip + [0.0, 0.0, PLACE_CLEAR]
            back = down - hand.apply([PLACE_BACK, 0.0, 0.0]) + [0.0, 0.0, PLACE_BACK]
            keys = [
                (up, hand, 0, carry(now, up)),
                (over, hand, 0, carry(up, over)),
                (down, hand, 0, carry(over, down) + 10),
                (down, hand, opening, PLACE_OPEN),
                (back, hand, opening, carry(down, back)),
            ]
        paths.append(list(waypoints(now, arm.pose()[1], keys, bool(profiles))))
    for s, t in itertools.zip_longest(*paths):
        yield act(s or paths[0][-1], t or paths[1][-1])


def role_grip(grasp, role):
    """A cup's grasp entry (real2sim/tasks/g1d_pour_beans_plus/grasps.json, a layout entry's "grasp") for `role`
    ("source" or "target"): its keys, with the role's own over them."""
    grasp = dict(grasp or {})
    merged = {**grasp, **dict(grasp.get(role) or {})}
    return {k: v for k, v in merged.items() if k not in ("source", "target", "note")}


def plus_actions(env, place=True):
    """g1d_pour_beans_plus: the cup-to-cup pour of all the beans between cups of any shape, by their layout "cavity"
    profiles and "grasp" entries; both cups put back on the desk (without `place`: the script ends with both held
    after the pour)."""
    lm = env.scene_manager.layout_manager
    profiles, grips = {}, {}
    for label in ("source_cup", "target_cup"):
        cup = lm.get_scene_object(0, lm.get_instance_name(0, label))
        scale = float(np.asarray(getattr(cup, "scale", (1.0, 1.0, 1.0)), float).reshape(-1)[0])
        profiles[label] = (cup.instance_config["cavity"], scale)
        grips[label] = role_grip(cup.instance_config.get("grasp"), label.split("_")[0])
    return cup_to_cup_actions(env, pour_share=1.0, place=place, profiles=profiles, grips=grips)


def half_cup_actions(env):
    """g1d_pour_beans_half_cup(_rigid): the cup-to-cup pour of about half a cup, both cups put back on the desk."""
    lm = env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "beans"))
    beans = len(fluid.get_particle_positions()[0])
    cavity = lm.get_scene_object(0, lm.get_instance_name(0, "target_cup")).instance_config["cavity"]
    return cup_to_cup_actions(
        env,
        pour_share=HALF_FILL * round_cup_shape.capacity(cavity, getattr(fluid, "grain", None)) / beans,
        place=True,
        meet=HALF_MEET,
        lift=HALF_LIFT,
        rim_down=HALF_RIM_DOWN,
        dry_rate=HALF_DRY_RATE,
    )


class Robot:
    """Both arms: one moves to Cartesian targets; the other holds its last command, or follows its own Cartesian path
    in the background (send_aside), one step per action of the first. The gripper hooks (make_arm, opening, grip_z,
    closing, holding) are the Dex1 jaw's; scripts/dexhands/block_pattern_demo.py overrides them for a pinch."""

    DESCEND = 30  # actions pick() comes down from its hover to the grip

    def __init__(self, env):
        self.env, self.lm = env, env.scene_manager.layout_manager
        self.arms = [self.make_arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        self.hold = {a.arm_key: a.q() for a in self.arms}
        self.grip = {a.grip_key: [1.0] for a in self.arms}
        self.home = {a.arm_key: (a.grasp_point(), a.pose()[1]) for a in self.arms}
        self.home_q = {a.arm_key: a.q() for a in self.arms}
        self.background = {}  # arm key -> iterator of joint targets, advanced while another arm moves
        self.settle = (
            None  # (m, actions): after each key hold its target until the arm is this close, at most this long
        )
        self.step = 0

    def make_arm(self, env, robot):
        return Arm(env, robot)

    def opening(self, width):
        """Gripper command that opens wide enough to come down over a side `width` wide."""
        return pad_open(width)

    def grip_z(self, bottom, top):
        """Grip point height for an object between `bottom` and `top`."""
        return max(top - 0.03, bottom + FINGER_TIPS + TABLE_CLEAR)

    def closing(self, width):
        """Gripper command closed on a side `width` wide: shut, both fingers stop against it."""
        return 0.0

    def holding(self, arm, width):
        """Gripper command to carry with once closed: where the fingers stopped, so the PD squeezes 1.5 mm."""
        fingers = as_np(arm.key.data.joint_pos[0, list(arm.robot.gripper_joint_indices)])
        return float(np.clip((0.023 - fingers.mean()) / 0.0445, 0.0, 1.0))

    def yaw_off(self, arm, label, period):
        """rad block_pattern_actions turns the hand back at the place for how far the held block `label` sits turned
        in `arm`'s grip (modulo its yaw `period`): 0, the jaw's flat pads square it."""
        return 0.0

    def arm_for(self, x):
        return min(self.arms, key=lambda a: abs(self.home[a.arm_key][0][0] - x))

    def move(self, arm, keys):
        """Follow `keys` [(grip point, hand rotation, gripper, actions)] with `arm`; with `settle`, wait at each."""
        for path in [keys] if self.settle is None else [[key] for key in keys]:
            for point, rot, grip in waypoints(arm.grasp_point(), arm.pose()[1], path):
                yield self._command(arm, point, rot, grip)
            if self.settle is not None:
                point, rot, grip, _ = path[-1]
                tol, most = self.settle
                for _ in range(most):
                    if np.linalg.norm(arm.grasp_point() - point) < tol:
                        break
                    yield self._command(arm, point, rot, grip)

    def _command(self, arm, point, rot, grip):
        self.hold[arm.arm_key] = arm.ik(point, rot)
        self.grip[arm.grip_key] = [float(grip)]
        self._advance(arm.arm_key)
        if self.step % 20 == 0:
            err = np.linalg.norm(arm.grasp_point() - point)
            print(f"[scripted_demo] {self.step:3d} {arm.arm_key} err {err:.3f}", flush=True)
        self.step += 1
        return dict(self.hold) | self.grip

    def object(self, label):
        """World centre, rotation, local bbox extents and world z bounds of an object."""
        pos, quat = self.lm.get_instance_pose(0, label=label)
        rot = from_wxyz(as_np(quat))
        bbox = self.lm.get_instance_bbox_vertices(self.lm.get_instance_name(0, label), 0)
        lo, hi = bbox.min(0), bbox.max(0)
        z = (as_np(pos) + rot.apply(bbox))[:, 2]
        return as_np(pos) + rot.apply((lo + hi) / 2), rot, hi - lo, z.min(), z.max()

    def pick(self, label, lift=LIFT, close=False, across=None, prefer=None, period=np.pi, arm=None):
        """Top-down grasp across the narrow side (or the local axis `across`), lifted; returns (arm, hand, offset,
        grip_z, open_cmd, close_cmd). With `prefer` (rad, per arm key): of the hand yaws that grasp the object alike
        (every `period`: pi for the symmetric jaw, less for a symmetric object, 0 for a round one), the one nearest
        it. `arm`: this arm, else the one on the object's side."""
        centre, rot, ext, bottom, top = self.object(label)
        axes = rot.as_matrix()  # columns are the object's local axes in world frame
        up = int(np.argmax(np.abs(axes[2])))
        narrow = across if across is not None else min((a for a in range(3) if a != up), key=lambda a: ext[a])
        yaw = np.arctan2(axes[1, narrow], axes[0, narrow])  # close along the narrow horizontal local axis
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # the jaw is symmetric: keep the wrist camera facing forward
        open_cmd = self.opening(ext[narrow])
        grip = np.r_[centre[:2], self.grip_z(bottom, top)]
        arm = arm or self.arm_for(centre[0])
        if prefer is None:
            if abs(yaw) > np.deg2rad(60):  # sideways grasp: point the wrist toward the shoulder so it stays in reach
                if (arm.arm_key.startswith("right") and yaw > 0) or (arm.arm_key.startswith("left") and yaw < 0):
                    yaw += -np.pi if yaw > 0 else np.pi
        else:
            want = prefer[arm.arm_key]
            yaw = want if not period else yaw + period * round((want - yaw) / period)
        hand = Rotation.from_euler("z", yaw) * TOP_ROT
        print(
            f"[scripted_demo] pick {label} at {centre.round(3)} size {ext.round(3)}"
            f" width {ext[narrow]:.3f} open {open_cmd:.2f} with {arm.arm_key}"
        )
        yield from self.move(
            arm,
            [
                (grip + [0, 0, lift], hand, open_cmd, 54),
                (grip, hand, open_cmd, self.DESCEND),
                (grip, hand, self.closing(ext[narrow]), 18),  # close on the table: both fingers stop against it
            ],
        )
        tilt_err = np.degrees((hand * arm.pose()[1].inv()).magnitude())
        close_cmd = self.closing(ext[narrow]) if close else self.holding(arm, ext[narrow])
        yield from self.move(arm, [(grip + [0, 0, lift], hand, close_cmd, 30)])
        centre, _, _, _, _ = self.object(label)
        fingers = as_np(arm.key.data.joint_pos[0, list(arm.robot.gripper_joint_indices)])
        print(
            f"[scripted_demo] lifted {label} to {centre.round(3)} (grip {grip.round(3)}), fingers {fingers.round(4)},"
            f" hand {tilt_err:.1f} deg off at the grasp"
        )
        return arm, hand, arm.pose()[1].inv().apply(centre - arm.grasp_point()), grip[2] - bottom, open_cmd, close_cmd

    def place(
        self, arm, hand, offset_local, grip_height, open_cmd, close_cmd, xy, surface, carry=48, label=None, hover=HOVER
    ):
        """Lower the held object onto `surface` (z) at `xy`, release to `open_cmd` and back off upward; `carry`
        actions to above the spot. With the held object's `label`, the hand is moved over the spot by where the
        object actually is once there (it may have slipped in the hand on the way)."""
        point = np.r_[np.asarray(xy) - hand.apply(offset_local)[:2], surface + grip_height + PLACE_GAP]
        print(f"[scripted_demo] place at {point.round(3)} on surface {surface:.3f}")
        keys = [
            (point + [0, 0, hover], hand, close_cmd, carry),
            (point, hand, close_cmd, 30),
            (point, hand, open_cmd, 12),  # release
            (point + [0, 0, hover], hand, open_cmd, 24),
        ]
        if label is None:
            yield from self.move(arm, keys)
            return
        yield from self.move(arm, keys[:1])
        fix = np.asarray(xy) - self.object(label)[0][:2]
        print(f"[scripted_demo] {label} {np.linalg.norm(fix) * 1000:.0f} mm off over the spot, corrected")
        yield from self.move(arm, [(p + [*fix, 0.0], r, g, n) for p, r, g, n in keys[1:]])

    def go_home(self):
        for arm in self.arms:
            point, rot = self.home[arm.arm_key]
            yield from self.move(arm, [(point, rot, 1, 54), (point, rot, 1, 12)])

    def send_aside(self, arm, point, n):
        """Move `arm` to `point`, hand turned as it is, in the background: one step per action while another arm moves;
        finish() runs it out."""
        path = waypoints(arm.grasp_point(), arm.pose()[1], [(point, arm.pose()[1], 1.0, n)])
        self.grip[arm.grip_key] = [1.0]
        self.background[arm.arm_key] = (arm.ik(p, rot) for p, rot, _ in path)

    def _advance(self, busy):
        """One step of every background path but `busy`'s."""
        for key, path in list(self.background.items()):
            if key != busy:
                q = next(path, None)
                if q is None:
                    del self.background[key]
                else:
                    self.hold[key] = q

    def finish(self):
        """Run the background paths out."""
        while self.background:
            self._advance(None)
            self.step += 1
            yield dict(self.hold) | self.grip

    def chassis(self):
        """Chassis (AGV_link) position (x, y) and heading of its +X axis, env frame."""
        rm = self.env.robot_manager
        body = next(r for r in rm.robot_list if hasattr(r, "body_dims"))
        pose = as_np(rm.get_link_pose(body, "AGV_link", env_idx_list=[0], is_relative=True)[0])
        fwd = from_wxyz(pose[3:7]).apply([1.0, 0.0, 0.0])
        return pose[:2], float(np.arctan2(fwd[1], fwd[0]))

    def drive(self, target, heading=None, reverse=False, **body):
        """Drive the chassis to `target` (x, y) with base_velocity, arms held: turn on the spot when off course, then
        straight on with a speed ramp; optionally turn to `heading` at the end. `body` keys (the lift) ride along."""

        def act(v, w):
            self.step += 1
            return dict(self.hold) | self.grip | body | {"base_velocity": [float(v), float(w)]}

        v = 0.0
        while True:
            xy, yaw = self.chassis()
            d = np.subtract(target, xy)
            dist = float(np.linalg.norm(d))
            if dist < DRIVE_TOL:
                break
            err = wrap(np.arctan2(d[1], d[0]) + (np.pi if reverse else 0.0) - yaw)
            want = 0.0 if abs(err) > TURN_FIRST else min(DRIVE_V, 1.2 * dist)
            v = float(np.clip(want, v - DRIVE_ACC, v + DRIVE_ACC))
            if self.step % 30 == 0:
                print(f"[scripted_demo] {self.step:4d} chassis {xy.round(3)} yaw {yaw:.2f} to {np.round(target, 3)}")
            yield act(-v if reverse else v, np.clip(2.0 * err, -DRIVE_W, DRIVE_W))
        while heading is not None and abs(err := wrap(heading - self.chassis()[1])) > HEADING_TOL:
            yield act(0.0, np.clip(2.0 * err, -DRIVE_W, DRIVE_W))
        for _ in range(SETTLE):
            yield act(0.0, 0.0)


def wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def table_approach(robot, table, start):
    """Unit vector from the lounge table centre to where the robot stands: the middle of a gap of at least CHAIR_GAP
    between the chairs round it, the one facing `start` most."""
    bearings = []
    for rec in robot.lm.get_layout_records(0, "Rigid"):
        if str(rec.get("label")).startswith("chair_"):
            xy = robot.object(rec["label"])[0][:2]
            if np.linalg.norm(xy - table) < 1.2:
                bearings.append(np.arctan2(*(xy - table)[::-1]))
    toward = np.arctan2(*(np.subtract(start, table))[::-1])
    bearings = sorted(bearings)
    gaps = [(b, (n - b) % (2 * np.pi) or 2 * np.pi) for b, n in zip(bearings, bearings[1:] + bearings[:1])]
    mids = [b + g / 2 for b, g in gaps if g >= CHAIR_GAP] or [toward]
    mid = min(mids, key=lambda m: abs(wrap(m - toward)))
    return np.array([np.cos(mid), np.sin(mid)])


def deliver_cup_actions(env):
    """Side-grasp the cup on the counter (as the cup pours: a top grasp slips off the 30 g plastic cup), lift it, back
    off, drive through the lounge to a gap between the chairs at the round table, lowering the lift column for it on
    the way, face the table, set the cup down, open and slide the hand back out."""
    robot = Robot(env)
    arm, open_cmd = yield from lift_cup(robot)
    yield from carry_to_lounge(env, robot, arm, "cup", open_cmd)


def lift_cup(robot, arm=None):
    """Side-grasp the cup on the counter from the outside of `arm` (default: the arm on its side), just below the rim,
    and lift it CARRY_UP; returns (arm, open command)."""
    lm = robot.lm
    cup0 = robot.object("cup")[0]
    arm = arm or robot.arm_for(cup0[0])
    out = -1.0 if arm.arm_key.startswith("left") else 1.0
    bbox = lm.get_instance_bbox_vertices(lm.get_instance_name(0, "cup"), 0)
    height = bbox[:, 2].max() - bbox[:, 2].min()
    grip, hand, _, _, open_cmd = cup_grasp(lm, out, "cup", height - C2C_RIM_DOWN["plastic_cup"])
    approach = grip - hand.apply([CUP_SLIDE, 0, 0])
    print(f"[scripted_demo] cup at {cup0.round(3)} with {arm.arm_key} grip {grip.round(3)} open {open_cmd:.2f}")
    yield from robot.move(
        arm,
        [
            (approach + [0, 0, HOVER], hand, open_cmd, 54),
            (approach, hand, open_cmd, 30),
            (grip, hand, open_cmd, 40),  # slide in
            (grip, hand, 0, 15),  # close
            (grip + [0, 0, CARRY_UP], hand, 0, 36),
        ],
    )
    print(f"[scripted_demo] lifted cup to {robot.object('cup')[0].round(3)} (from {cup0.round(3)})")
    return arm, open_cmd


def carry_to_lounge(env, robot, arm, label, open_cmd):
    """With the cup `label` lifted in `arm`: back off from the counter, drive through the lounge to a gap between the
    chairs at the round table, lowering the lift column for it on the way, face the table, set the cup down, open to
    `open_cmd` and slide the hand back out."""
    start, yaw = robot.chassis()
    yield from robot.drive(start - BACK_OFF * np.array([np.cos(yaw), np.sin(yaw)]), reverse=True)
    table, r = np.asarray(env.table_xy[0]), env.table_r[0]
    u = table_approach(robot, table, start)
    lift = {"lift_joint_state": [LOUNGE_LIFT]}
    bearing = np.degrees(np.arctan2(u[1], u[0]))
    print(f"[scripted_demo] lounge table {table.round(3)} r {r:.3f}, approach bearing {bearing:.0f}")
    yield from robot.drive(table + (r + TABLE_STAND + APPROACH) * u, **lift)
    yield from robot.drive(table + (r + TABLE_STAND) * u, heading=float(np.arctan2(-u[1], -u[0])), **lift)
    spot = table + (r + TABLE_STAND - PLACE_AHEAD) * u
    # the cup sits in the hand as it does now (it may have slid in the grip on the way): keep that offset
    point, rot = arm.grasp_point(), arm.pose()[1]
    centre, _, _, bottom, _ = robot.object(label)
    down = np.r_[spot + point[:2] - centre[:2], env.table_z[0] + PLACE_GAP + point[2] - bottom]
    back = down - rot.apply([CUP_SLIDE, 0, 0])
    print(f"[scripted_demo] place cup at {spot.round(3)}, hand to {down.round(3)}")
    yield from robot.move(
        arm,
        [
            (down + [0, 0, HOVER], rot, 0, 48),
            (down, rot, 0, 36),
            (down, rot, open_cmd, 15),  # release
            (back, rot, open_cmd, 30),  # slide out along the fingers
            (back + [0, 0, HOVER], rot, open_cmd, 24),
        ],
    )
    checks = {c[0]: env.reward_manager.call_func_parser(c, 0) for c in env.delivered_checks()}
    print(f"[scripted_demo] cup at {robot.object(label)[0].round(3)}, checks {checks}, env success {env.success[0]}")


def pour_deliver_actions(env):
    """Pour water from the jug into the cup standing on the counter as pour_actions does (the other hand waits), set
    the jug back down where it stood (over its spot, down, open, back out of the handle, home), then pick the cup up
    and carry it to the lounge table."""
    robot = Robot(env)  # home poses from the start
    jug0, _, _, counter_z, _ = robot.object("bottle")
    yield from pour_actions(env, "water", WATER_DRY, WATER_POUR, hold_cup=False, rate=WATER_RATE)
    robot.hold = {a.arm_key: a.q() for a in robot.arms}
    jug_arm = robot.arm_for(jug0[0])
    robot.grip[jug_arm.grip_key] = [0.0]
    point, rot = jug_arm.grasp_point(), jug_arm.pose()[1]
    centre, _, _, bottom, _ = robot.object("bottle")
    down = np.r_[jug0[:2] + point[:2] - centre[:2], counter_z + PLACE_GAP + point[2] - bottom]
    home, home_rot = robot.home[jug_arm.arm_key]
    print(f"[scripted_demo] set the jug down at {jug0[:2].round(3)}, hand {down.round(3)}, chassis {robot.chassis()}")
    yield from robot.move(
        jug_arm,
        [
            (np.r_[down[:2], point[2]], rot, 0, max(int(np.linalg.norm(down[:2] - point[:2]) / JUG_SPEED), 30)),
            (down, rot, 0, max(int((point[2] - down[2]) / JUG_SPEED), 30)),
            (down, rot, OPEN, 15),  # release
            (down + BEHIND, rot, OPEN, 30),  # out of the handle the way the fingers came in
            (down + BEHIND + [0, 0, 0.1], rot, OPEN, 24),
            (home, home_rot, 1.0, 54),
        ],
    )
    # the other hand, from the far side of the jug (the cup may stand on the jug arm's side of the midline)
    arm, open_cmd = yield from lift_cup(robot, next(a for a in robot.arms if a is not jug_arm))
    yield from carry_to_lounge(env, robot, arm, "cup", open_cmd)


def pickup_actions(env):
    robot = Robot(env)
    arm, _, _, _, _, close_cmd = yield from robot.pick("target", close=True)
    point, rot = arm.grasp_point(), arm.pose()[1]
    yield from robot.move(arm, [(point, rot, close_cmd, 36)])  # hold it up


def stack_actions(env):
    robot = Robot(env)
    labels = ["block_0", "block_1", "block_2"]
    base, *others = sorted(labels, key=lambda label: abs(robot.object(label)[0][0]))
    stack = [base]
    for label in others:
        arm, hand, offset_local, grip_height, open_cmd, close_cmd = yield from robot.pick(label, lift=HOVER)
        top = max(robot.object(b)[4] for b in stack)
        yield from robot.place(
            arm, hand, offset_local, grip_height, open_cmd, close_cmd, robot.object(stack[-1])[0][:2], top
        )
        stack.append(label)
    yield from robot.go_home()


# Block patterns: the hand yaw every block is set down with, fingers closing along the pattern's depth. Probed from the
# start pose: each hand reaches its side's scattered blocks within 75 deg of fingers forward, and the build space
# with the fingers pointing across at it (left -90, right +90 deg) from 3 cm up at y -0.40 .. -0.44 (2 cm short at
# y -0.36); turned the other way neither reaches the midline. Blocks are gripped at block_patterns.grip_yaw, the
# layouts turn them to within 20 deg of it (build_asset.GRASP_YAW): the hand turns at most ~95 deg on the way, at
# PATTERN_TURN_RATE per action under Limiter.
PATTERN_PLACE_YAW = {"left_arm_joint_state": -np.pi / 2, "right_arm_joint_state": np.pi / 2}
PATTERN_TURN_RATE = 0.4 / 30  # rad per action, half of REAL_ARM_VMAX
# Limiter slows the long first moves from the start pose: the arm fell 9 cm behind on the way to a block and swept it
# away low down. Every key is held until the arm is this close (m), for at most this many actions.
PATTERN_SETTLE = (0.005, 60)
# |x|, y, height above the table (m) a hand waits at while the other one works, and the actions to get there. Its way
# back to the start pose in joint space first swung it in front of the chest, into the other hand reaching for a block
PATTERN_ASIDE, PATTERN_ASIDE_ACTIONS = (0.16, -0.40, 0.12), 45
LANDED_TOL = 0.03  # m: a block set down further than this from plan does not steer the next places


def block_pattern_actions(env, robot=None):
    """Build the card's pattern about block_patterns.BUILD_XY, bottom-up in its piece order. Every block is gripped
    across its local Y (its 30 mm depth; a half-cylinder then holds its dome to the front) and set down with the hand
    at PATTERN_PLACE_YAW, the fingers closing along the pattern's depth: they pass in front of and behind it, not into
    the gap to a neighbour, and the block gets its pattern yaw. Each place is corrected by how far the blocks already
    set down landed from plan, set onto the measured top of the blocks under it. A hand that picks the next block too
    goes straight on to it; one that does not moves aside, above its own side of the table (PATTERN_ASIDE), while the
    other hand starts on the next block, and picks its next block from there. `robot`: a Robot (subclass) to build
    with, else the Dex1 one."""
    robot = robot or Robot(env)
    robot.settle = PATTERN_SETTLE
    pattern = env.pattern[0]
    labels = block_patterns.labels(pattern)
    table = min(robot.object(label)[3] for label in labels)
    site = np.asarray(block_patterns.BUILD_XY, dtype=float)
    planned, landed = {}, {}
    print(f"[scripted_demo] pattern {pattern['name']} ({pattern['zh']}), {len(labels)} blocks, table {table:.3f}")
    for i, (cat, x, z) in enumerate(pattern["pieces"]):
        label = labels[i]
        period = np.radians(min(block_patterns.YAW_PERIOD[cat], 180))  # the jaw repeats every 180 deg anyway
        prefer = np.radians(block_patterns.grip_yaw(*robot.object(label)[0][:2]))
        arm, hand, offset, grip_height, open_cmd, close_cmd = yield from robot.pick(
            label,
            lift=HOVER,
            across=1,
            prefer=dict.fromkeys(robot.home, prefer),
            period=period,
        )
        width = block_patterns.SIZES[cat][0]
        under = [
            labels[j]
            for j, (c, xb, zb) in enumerate(pattern["pieces"][:i])
            if abs(zb + block_patterns.SIZES[c][2] - z) < 0.0015
            and min(x + width / 2, xb + block_patterns.SIZES[c][0] / 2)
            > max(x - width / 2, xb - block_patterns.SIZES[c][0] / 2)
        ]
        surface = max((robot.object(b)[4] for b in under), default=table + z)
        good = [landed[b] - planned[b] for b in landed if np.linalg.norm(landed[b] - planned[b]) < LANDED_TOL]
        shift = np.mean(good, axis=0) if good else np.zeros(2)
        planned[label] = site + [x, 0.0]
        yaw = (hand * TOP_ROT.inv()).as_euler("zyx")[0]  # pick() turned TOP_ROT this much about z
        turn = wrap(PATTERN_PLACE_YAW[arm.arm_key] - robot.yaw_off(arm, label, period) - yaw)
        target = planned[label] + shift
        print(f"[scripted_demo] {label} {cat} to {target.round(3)} on {surface:.3f}, turn {np.degrees(turn):.0f}")
        carry = max(48, int(np.ceil(abs(turn) / PATTERN_TURN_RATE)))
        held = Rotation.from_euler("z", turn) * hand
        yield from robot.place(
            arm, held, offset, grip_height, open_cmd, close_cmd, target, surface, carry=carry, label=label
        )
        landed[label] = robot.object(label)[0][:2]
        print(f"[scripted_demo] {label} landed at {landed[label].round(3)} (planned {planned[label].round(3)})")
        following = labels[i + 1] if i + 1 < len(labels) else None
        if following is None or robot.arm_for(robot.object(following)[0][0]) is not arm:
            x, y, up = PATTERN_ASIDE
            side = np.sign(robot.home[arm.arm_key][0][0])
            robot.send_aside(arm, np.array([side * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()
    poses = [robot.lm.get_instance_pose(0, label=label) for label in labels]
    xy, z, tilt, yaw = pattern_error(
        [as_np(p) for p, _ in poses],
        [from_wxyz(as_np(q)).as_matrix() for _, q in poses],
        block_patterns.targets(pattern),
        [block_patterns.YAW_PERIOD[cat] for cat, _, _ in pattern["pieces"]],
        pattern["anchor"],
        block_patterns.groups(pattern),
    )
    built = env.reward_manager.call_func_parser(env.built_check(0), 0)
    print(
        f"[scripted_demo] pattern built {built:.0f}: worst xy {xy * 1000:.0f} mm, z {z * 1000:.0f} mm,"
        f" tilt {tilt:.0f} deg, yaw {yaw:.0f} deg"
    )


def layout_entry(env, label):
    """The layout entry (category, category_idx, ...) of the object `label` in env 0."""
    rigid = env.scene_manager.config["env0"]["Rigid"]
    return next(e for es in rigid.values() for e in es if e.get("label") == label)


# m above the drop point (itself above the tray walls): HOVER there, 2.5 cm higher than on the table, left the hands
# 12-60 cm off when reaching across the midline (2 of 13 short cylinders)
SORT_HOVER = 0.03


def block_sort_actions(env):
    """Sort the blocks into the trays by their cards (block_sorting.py). Each block is picked by the hand on its side
    (the hands neither pick nor reach far across the midline), gripped from above across its local Y, carried over its
    tray and dropped into the tray's next free slot from just above the walls (open fingers inside the 6 cm tray
    caught its walls); the trays stand in the middle, in both hands' reach.
    The left hand's blocks first, one after the other, then the right hand's, the left hand moving aside meanwhile
    (PATTERN_ASIDE: above its own side, where its blocks lay)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    groups = env.groups[0]
    order = sorted(robot.arms, key=lambda a: robot.home[a.arm_key][0][0])  # left hand first
    plan = [(robot.arm_for(robot.object(b)[0][0]), box, b) for box in env.BOXES for b in groups[box]]
    plan.sort(key=lambda p: order.index(p[0]))
    free = {box: [0, 1] for box in env.BOXES}
    table = min(robot.object(label)[3] for *_, label in plan)
    print(f"[scripted_demo] sort by {env.BY}: {groups}")
    for n, (arm, box, label) in enumerate(plan):
        cat = layout_entry(env, label)["category"]
        centre = robot.object(label)[0]
        period = np.radians(min(block_patterns.YAW_PERIOD[cat], 180))  # the jaw repeats every 180 deg anyway
        prefer = np.radians(block_patterns.grip_yaw(centre[0], centre[1]))
        arm, hand, offset, grip_height, open_cmd, close_cmd = yield from robot.pick(
            label, lift=HOVER, across=1, prefer=dict.fromkeys(robot.home, prefer), period=period, arm=arm
        )
        slot = free[box].pop(0)
        target = np.asarray(block_sorting.slot_xy(env.BOXES.index(box), slot))
        rim = robot.object(box)[4]
        print(f"[scripted_demo] {label} {cat} into {box} slot {slot} at {target.round(3)}")
        yield from robot.place(
            arm, hand, offset, grip_height, open_cmd, close_cmd, target, rim, label=label, hover=SORT_HOVER
        )
        print(f"[scripted_demo] {label} landed at {robot.object(label)[0].round(3)}")
        if n + 1 == len(plan) or plan[n + 1][0] is not arm:
            x, y, up = PATTERN_ASIDE
            out = np.sign(robot.home[arm.arm_key][0][0])
            robot.send_aside(arm, np.array([out * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()
    checks = [env.reward_manager.call_func_parser(c, 0) for c in env.sorted_checks(0)]
    print(f"[scripted_demo] sorted checks {checks}")


def atomic_pick(robot, label, lift=HOVER):
    """Top grasp of block `label` across its local Y by the hand on its side, at that hand's best grip yaw."""
    cat = layout_entry(robot.env, label)["category"]
    centre = robot.object(label)[0]
    period = np.radians(min(block_patterns.YAW_PERIOD[cat], 180))  # the jaw repeats every 180 deg anyway
    prefer = np.radians(block_patterns.grip_yaw(centre[0], centre[1]))
    return (yield from robot.pick(label, lift=lift, across=1, prefer=dict.fromkeys(robot.home, prefer), period=period))


def place_turned(robot, grasp, target, surface, label):
    """Set the held block down at `target` on `surface` with the hand at about PATTERN_PLACE_YAW: the block's long side
    ends along world X (its pattern yaw 0) and the fingers close along Y, not into a gap beside it. The hand's final
    yaw takes up how far the block sits turned in it as lifted (a grip a few degrees off, or slipped on the way up)."""
    arm, hand, offset, grip_height, open_cmd, close_cmd = grasp
    yaw = (hand * TOP_ROT.inv()).as_euler("zyx")[0]  # pick() turned TOP_ROT this much about z
    period = np.radians(min(block_patterns.YAW_PERIOD[layout_entry(robot.env, label)["category"]], 180))
    in_hand = yaw_of(robot.object(label)[1].as_matrix()) - (arm.pose()[1] * TOP_ROT.inv()).as_euler("zyx")[0]
    final, off = PATTERN_PLACE_YAW[arm.arm_key], 0.0
    if period:
        off = (final + in_hand + period / 2) % period - period / 2
        final -= off
    turn = wrap(final - yaw)
    print(f"[scripted_demo] {label} sits {np.degrees(off):.1f} deg off in the hand, hand to {np.degrees(final):.1f}")
    carry = max(48, int(np.ceil(abs(turn) / PATTERN_TURN_RATE)))
    held = Rotation.from_euler("z", turn) * hand
    yield from robot.place(
        arm, held, offset, grip_height, TURNED_RELEASE, close_cmd, target, surface, carry=carry, label=label
    )


# Gripper command a block set down by place_turned() is let go with: opened only to its grip width, the fingertips
# caught a 90 mm plank as the hand backed off and turned it 6-17 deg, dragging it 1 cm toward the robot.
TURNED_RELEASE = 1.0


def atomic_done(robot, arm, table):
    """The hand moves aside (PATTERN_ASIDE, above the table top `table`) and both arms settle."""
    x, y, up = PATTERN_ASIDE
    robot.send_aside(arm, np.array([np.sign(robot.home[arm.arm_key][0][0]) * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()


def place_on_mark_actions(env):
    """g1d_place_block_on_mark: the block onto the mark's centre."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    table = robot.object("block")[3]
    grasp = yield from atomic_pick(robot, "block")
    mark = robot.object("mark")
    yield from robot.place(*grasp, mark[0][:2], mark[4], label="block")
    yield from atomic_done(robot, grasp[0], table)


def stack_on_block_actions(env):
    """g1d_stack_block_on_block: the top block onto the base block's top (both on the same side)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    base = robot.object("base_block")
    grasp = yield from atomic_pick(robot, "top_block")
    yield from robot.place(*grasp, base[0][:2], base[4], label="top_block")
    yield from atomic_done(robot, grasp[0], base[3])


def pick_named_actions(env):
    """g1d_pick_named_block: the named block up PICK_LIFT and held there."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    arm, hand, _, _, _, close_cmd = yield from atomic_pick(robot, "target_block", lift=PICK_LIFT)
    yield from robot.move(arm, [(arm.grasp_point(), hand, close_cmd, PICK_HOLD)])


PICK_ANYTHING_LIFT = 0.13  # m above the grasp
# m of a tall object above the grip point: at Robot's 3 cm the knuckles behind the 60 deg pads pressed 10 cm pegs and
# bottles over on the way down (<gpu-server>, 11-11)
PICK_ANYTHING_DEPTH = 0.012


class TopGripRobot(Robot):
    # 13 cm down in 30 actions outran the 0.8 rad/s joint limit: the clipped joints bowed the path 4 cm sideways
    # into narrow targets
    DESCEND = 75

    def grip_z(self, bottom, top):
        return max(top - PICK_ANYTHING_DEPTH, bottom + FINGER_TIPS + TABLE_CLEAR)


def pick_anything_actions(env, robot_class=None):
    """g1d_pick_anything: top grasp across the target's narrow side, PICK_ANYTHING_DEPTH below its top, at the grip
    yaw of its spot (the layouts turn it so, scripts/objects/pick_anything.py), waiting at each key for the
    rate-limited arm; lifted and held there. `robot_class`: TopGripRobot (expert v3) or GraspRobot (v4)."""
    robot = (robot_class or TopGripRobot)(env)
    robot.settle = PATTERN_SETTLE
    centre = robot.object("target")[0]
    prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(centre[0], centre[1])))
    arm, hand, _, _, _, close_cmd = yield from robot.pick("target", lift=PICK_ANYTHING_LIFT, close=True, prefer=prefer)
    yield from robot.move(arm, [(arm.grasp_point(), hand, close_cmd, PICK_HOLD)])


def urdf_chain(urdf, ee_link, arm, values):
    """The URDF's joints from the first of `arm` (joint names) to `ee_link`: (origin 4x4, axis or None, index in `arm`
    or the joint's fixed value from `values`)."""
    import xml.etree.ElementTree as ET

    joints = ET.parse(urdf).getroot().findall("joint")
    parent = {j.find("child").get("link"): j for j in joints}
    chain, link = [], ee_link
    while link in parent:
        chain.insert(0, parent[link])
        link = parent[link].find("parent").get("link")
    chain = chain[next(i for i, j in enumerate(chain) if j.get("name") in arm) :]

    def vec(text):
        return np.array([float(v) for v in text.split()]) if text else np.zeros(3)

    out = []
    for j in chain:
        o, t = j.find("origin"), np.eye(4)
        if o is not None:
            t[:3, :3] = Rotation.from_euler("xyz", vec(o.get("rpy"))).as_matrix()
            t[:3, 3] = vec(o.get("xyz"))
        name = j.get("name")
        axis = None if j.get("type") == "fixed" else vec(j.find("axis").get("xyz"))
        out.append((t, axis, arm.index(name) if name in arm else float(values.get(name, 0.0))))
    return out


class Reach:
    """Offline IK on an arm's URDF chain (first arm joint to the ee link), its root put where the sim has it when made:
    whether a hand pose is reachable within the Limiter's joint range before the arm is sent there (the sim's step IK,
    Arm.ik, still drives it). Plans only, never commands: a full solve can land on another arm configuration, and
    commanding it whipped the arm (docs/open-anything.md)."""

    def __init__(self, arm):
        names = list(arm.key.joint_names)
        values = dict(zip(names, as_np(arm.key.data.joint_pos[0])))
        self.chain = urdf_chain(arm.robot.urdf_path, arm.robot.ee_link_name, [names[c] for c in arm.cols], values)
        self.lim = as_np(arm.key.data.soft_joint_pos_limits[0, arm.cols]).reshape(-1, 2) + [
            REAL_JOINT_MARGIN,
            -REAL_JOINT_MARGIN,
        ]
        self.reach = arm.REACH
        pos, rot = arm.pose()
        now = np.eye(4)
        now[:3, :3], now[:3, 3] = rot.as_matrix(), pos
        self.base = now @ np.linalg.inv(self.fk(arm.q()))

    def fk(self, q, axes=None):
        """ee link pose (4x4) in the chain's root frame at arm joints `q`; with a list `axes`, each arm joint's (point,
        axis) in that frame appended to it."""
        t = np.eye(4)
        for origin, axis, v in self.chain:
            t = t @ origin
            if axis is not None:
                angle = q[v] if isinstance(v, int) else v
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                m = np.eye(4)
                m[:3, :3] += np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
                if axes is not None and isinstance(v, int):
                    axes.append((t[:3, 3].copy(), t[:3, :3] @ axis))
                t = t @ m
        return t

    def solve(self, point, rot_t, seed, iters=200):
        """(joints, m, deg): the arm joints nearest `seed` putting the grasp point at `point` with the hand at `rot_t`
        (damped least squares within the joint range), and how far that misses."""
        target, q = rot_t.as_matrix(), np.clip(np.asarray(seed, dtype=float), *self.lim.T)
        for _ in range(iters):
            axes = []
            t = self.base @ self.fk(q, axes)
            grip = t[:3, 3] + t[:3, :3] @ self.reach
            rv = Rotation.from_matrix(target @ t[:3, :3].T).as_rotvec()
            e = np.r_[point - grip, 0.3 * rv]
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(rv) < 2e-3:
                break
            jac = np.zeros((6, len(q)))
            for i, (o, a) in enumerate(axes):
                o, a = self.base[:3, :3] @ o + self.base[:3, 3], self.base[:3, :3] @ a
                jac[:3, i], jac[3:, i] = np.cross(a, grip - o), 0.3 * a
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.01**2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), *self.lim.T)
        return q, float(np.linalg.norm(e[:3])), float(np.degrees(np.linalg.norm(rv)))


def jaw_opening(env, arm):
    """`arm`'s gripper opening, 0 shut .. 1 open (func_parser._gripper_opening's measure)."""
    robot = arm.robot
    val = float(np.mean(env.robot_manager.get_end_effector_real_val(robot=robot, env_idx_list=[0])[0]))
    lo, hi = robot.gripper_scale
    return (val - lo) / (hi - lo) if robot.gripper_move["sign"] == 1 else (hi - val) / (hi - lo)


# Pick anything expert v4 (GraspRobot.pick, docs/pick-expert-v4.md): the first try is v3's grasp, untouched (a single-
# seed URDF solve from the start pose missed grasps the sim's step IK reaches along its path, A/B interim 11-19); the
# others must reach their hover and grip within GRASP_TOL on the URDF: the wrist turned half round, the jaw across the
# other side of a squarish object, the fingers tilted GRASP_PITCHES, a slender object (GRASP_SLENDER) taken GRASP_DEEP
# of its height up with the fingers at GRASP_DEEP_PITCH, the other hand within GRASP_MIDLINE of the midline. Each try
# comes down past its hover (a waypoint, as v3's) and checks the grasp point ended beside the target as it lies now and
# not above the grip (GRIP_MISS), shuts, gives up a jaw shut on nothing (GRASP_EMPTY), and lifts as v3 does: a target
# that rose less than TEST_RISE is set down, let go, and the next try plans from where it lies. The first try comes down
# and lifts exactly as v3; later ones slow down for the last GRASP_SLOW. GRASP_TRIES tries at most, none once the target
# is off the desk (GRASP_GONE); a hand that misses goes back up to its hover first. The last try closes wherever the
# hand got to and lifts whatever it holds, as v3 does
GRASP_TRIES = 3
GRASP_TOL = (0.008, 12.0)  # m, deg
GRASP_PITCHES = (60.0, 45.0, 75.0)  # deg the fingers point below level; 60 is TOP_ROT's
GRASP_DEEP, GRASP_DEEP_PITCH, GRASP_DEEP_MIN = 0.45, 40.0, 0.07  # of the height from the bottom; deg; m tall at least
GRASP_SLENDER = 2.0  # height / narrow side from which an object is taken deep too (first with deep_first)
GRASP_GONE = 0.05  # m the target's centre fell below where it started: knocked off the desk
GRASP_SQUARE = 1.35  # longer / shorter horizontal side up to which the jaw may close across either
GRASP_SLOW = (0.04, 24)  # m above the grip the descent slows down; actions for that last stretch
GRASP_MIDLINE = 0.06  # m: within this of the midline the other hand is a candidate too
# m, about the 99th percentile of v3's accepted grasps (300 sampled, 11-19): GRIP_MISS the grasp point beside the target
# (2.4 cm) and above its grip (3.4 cm) at the close; tighter limits gave up grasps v3 makes (A/B interims, 11-19). The
# hover is only a waypoint: v3 went on down from hovers more than 5 cm off and grasped (a tall object's hover is high).
# GRASP_MOVED: a grasp is tried again once the target lies this far from where it was tried
GRIP_MISS, GRASP_MOVED = (0.025, 0.035), 0.03
GRASP_EMPTY = 0.04  # jaw opening below which it holds nothing (pool objects are >= 15 mm: ~0.1)
# m: the target risen less than TEST_RISE once lifted as v3 lifts was not held (place_anything_actions' PLACE_LIFTED);
# it is set down TEST_LIFT above the grip and let go. A 3 cm test lift first let go targets v3 lifts (they lag, A/B)
TEST_LIFT, TEST_RISE = 0.03, 0.03
JAW_MAX = PAD_CLOSED + PAD_TRAVEL - GRASP_CLEAR  # widest side the open jaw comes down over with GRASP_CLEAR to spare


class GraspRobot(TopGripRobot):
    """TopGripRobot whose pick() checks and retries (pick anything expert v4); same arguments and result.
    `deep_first`: a slender object (GRASP_SLENDER) is first taken deep, not by v3's top grasp (an object carried
    somewhere: held by its top it swings and slips, docs/pick-expert-v4.md). `first_try_grip_check`: the first try also
    gives up a grasp point ending GRIP_MISS off (later tries always do)."""

    deep_first = False
    first_try_grip_check = True

    def __init__(self, env):
        super().__init__(env)
        self.reach = {a.arm_key: Reach(a) for a in self.arms}
        self.log = env.grasp_attempts = []  # this episode's tries, for the plugin's sidecar

    def candidates(self, label, across, prefer, period, arm):
        """Grasps of `label` as it lies now, v3's first: [(arm, hand, grip point, width, name)]."""
        centre, rot, ext, bottom, top = self.object(label)
        axes = rot.as_matrix()
        up = int(np.argmax(np.abs(axes[2])))
        sides = [a for a in range(3) if a != up]
        narrow = across if across is not None else min(sides, key=lambda a: ext[a])
        other = next(a for a in sides if a != narrow)
        yaw0 = np.arctan2(axes[1, narrow], axes[0, narrow])
        first = arm or self.arm_for(centre[0])
        arms = [first] + [a for a in self.arms if a is not first and arm is None and abs(centre[0]) < GRASP_MIDLINE]
        grip = np.r_[centre[:2], self.grip_z(bottom, top)]
        deep = np.r_[centre[:2], max(bottom + GRASP_DEEP * (top - bottom), bottom + FINGER_TIPS + TABLE_CLEAR)]
        square = ext[other] <= JAW_MAX and ext[other] <= GRASP_SQUARE * ext[narrow] and period != 0
        out = []
        for a in arms:
            yaw = (yaw0 + np.pi / 2) % np.pi - np.pi / 2  # Robot.pick's choice, per arm
            if prefer is None:
                if abs(yaw) > np.deg2rad(60):
                    if (a.arm_key.startswith("right") and yaw > 0) or (a.arm_key.startswith("left") and yaw < 0):
                        yaw += -np.pi if yaw > 0 else np.pi
            else:
                want = prefer[a.arm_key]
                yaw = want if not period else yaw + period * round((want - yaw) / period)

            def hand(turn, pitch):
                return Rotation.from_euler("z", yaw + turn) * Rotation.from_euler("x", -pitch, degrees=True) * GRASP_ROT

            own = "" if a is first else "other hand, "
            top_pitch = GRASP_PITCHES[0]
            v3 = (a, hand(0.0, top_pitch), grip, ext[narrow], own + "v3" if not own else "other hand")
            if top - bottom >= max(GRASP_DEEP_MIN, GRASP_SLENDER * ext[narrow]):
                deep_grasp = (a, hand(0.0, GRASP_DEEP_PITCH), deep, ext[narrow], own + "deep")
                out += [deep_grasp, v3] if self.deep_first else [v3, deep_grasp]
            else:
                out.append(v3)
            for pitch in GRASP_PITCHES:
                tag = "" if pitch == top_pitch else f", {pitch:.0f} deg"
                if pitch != top_pitch:
                    out.append((a, hand(0.0, pitch), grip, ext[narrow], f"{own}tilted{tag}"))
                out.append((a, hand(np.pi, pitch), grip, ext[narrow], f"{own}wrist turned{tag}"))
                if square:
                    out += [(a, hand(t, pitch), grip, ext[other], f"{own}other side {s}{tag}") for t, s in
                            ((np.pi / 2, "+"), (-np.pi / 2, "-"))]  # fmt: skip
        return out

    def miss(self, cand, lift):
        """(m, deg) the URDF misses the candidate's hover or grip by at worst, from where its arm is now."""
        a, hand, grip, _, _ = cand
        reach = self.reach[a.arm_key]
        q, *hover_off = reach.solve(grip + [0, 0, lift], hand, a.q())
        _, *grip_off = reach.solve(grip, hand, q)
        return max(hover_off[0], grip_off[0]), max(hover_off[1], grip_off[1])

    def choose(self, cands, lift):
        """v3's grasp whenever it is left, else the first within GRASP_TOL, else the nearest miss."""
        misses = []
        for c in cands:
            if c[4] == "v3":
                return c, None
            m = self.miss(c, lift)
            if m[0] <= GRASP_TOL[0] and m[1] <= GRASP_TOL[1]:
                return c, m
            misses.append((m, c))
        m, c = min(misses, key=lambda mc: mc[0][0])
        return c, m

    def pick(self, label, lift=LIFT, close=False, across=None, prefer=None, period=np.pi, arm=None):
        tried = {}  # grasp name (the other hand's are prefixed) -> where the target was when it was tried
        first = self.object(label)[0]
        for attempt in range(GRASP_TRIES):
            centre, _, _, bottom, _ = self.object(label)
            if attempt and centre[2] < first[2] - GRASP_GONE:
                print(f"[scripted_demo] {label} knocked off the desk ({centre.round(3)}): no more tries")
                break
            cands = self.candidates(label, across, prefer, period, arm)
            # a grasp is tried again only once the target lies elsewhere (a miss pushed it)
            fresh = [c for c in cands if np.linalg.norm(tried.get(c[4], centre + 1.0) - centre) > GRASP_MOVED]
            cand, urdf_miss = self.choose(fresh or cands[:1], lift)
            a, hand, grip, width, name = cand
            tried[name] = centre
            open_cmd = self.opening(width)
            hover = grip + [0, 0, lift]
            entry = dict(label=label, attempt=attempt, grasp=name, arm=a.arm_key)
            if urdf_miss is not None:
                entry["urdf_miss"] = [round(urdf_miss[0], 3), round(urdf_miss[1], 1)]
            self.log.append(entry)
            print(
                f"[scripted_demo] grasp try {attempt}: {label} by {a.arm_key} ({name}) at {grip.round(3)}"
                f" width {width:.3f} open {open_cmd:.2f}, URDF misses it by {entry.get('urdf_miss', '-')}"
            )
            last = attempt == GRASP_TRIES - 1
            yield from self.move(a, [(hover, hand, open_cmd, 54 if attempt == 0 else 40)])
            slow, slow_n = GRASP_SLOW
            down = [(grip, hand, open_cmd, self.DESCEND)]  # the first try comes down as v3 does
            if attempt:
                down = [(grip + [0, 0, slow], hand, open_cmd, self.DESCEND - slow_n), (grip, hand, open_cmd, slow_n)]
            yield from self.move(a, down)
            point = a.grasp_point()
            aside = np.linalg.norm(point[:2] - self.object(label)[0][:2])  # the target as it lies now: pushed or not
            above = point[2] - grip[2]
            checked = attempt or self.first_try_grip_check
            if not last and checked and (aside > GRIP_MISS[0] or above > GRIP_MISS[1]):
                entry["result"] = f"grip missed ({aside * 1000:.0f} mm beside the target, {above * 1000:.0f} mm high)"
                yield from self.move(a, [(hover, hand, open_cmd, 30)])
                continue
            yield from self.move(a, [(grip, hand, self.closing(width), 18)])
            jaw = jaw_opening(self.env, a)
            if jaw < GRASP_EMPTY:
                entry["result"] = f"shut on nothing ({jaw:.2f})"
                yield from self.move(a, [(grip, hand, open_cmd, 10), (hover, hand, open_cmd, 30)])
                continue
            close_cmd = self.closing(width) if close else self.holding(a, width)
            before = self.object(label)[0]
            yield from self.move(a, [(hover, hand, close_cmd, 30)])  # v3's lift
            rose = self.object(label)[0][2] - before[2]
            if not last and rose < TEST_RISE:
                entry["result"] = f"did not rise ({rose * 1000:.0f} mm, jaw {jaw:.2f})"
                low = np.r_[a.grasp_point()[:2], grip[2] + TEST_LIFT]  # set down what it may hold, then let go
                yield from self.move(
                    a, [(low, hand, close_cmd, 20), (low, hand, open_cmd, 10), (hover, hand, open_cmd, 24)]
                )
                continue
            centre = self.object(label)[0]
            entry["result"] = "held" if rose >= TEST_RISE else f"last try lifted anyway ({rose * 1000:.0f} mm)"
            print(f"[scripted_demo] grasp try {attempt}: lifted {label} to {centre.round(3)}, jaw {jaw:.2f}")
            offset = a.pose()[1].inv().apply(centre - a.grasp_point())
            return a, hand, offset, grip[2] - bottom, open_cmd, close_cmd
        print(f"[scripted_demo] {label}: no grasp held in {GRASP_TRIES} tries")
        offset = a.pose()[1].inv().apply(self.object(label)[0] - a.grasp_point())
        return a, hand, offset, grip[2] - bottom, open_cmd, self.closing(width)


PICK_REGRASPS = 1  # times pick_anything_v4_actions grasps again a target that fell out during the hold


def pick_anything_v4_actions(env, robot_class=None):
    """g1d_pick_anything with expert v4: pick_anything_actions' grasp and hold on GraspRobot (or `robot_class`, a
    subclass of it). The episode ends once the target has been held up for the acceptance's frames; a script still
    running after the hold dropped it (or moved a distractor): a dropped target is grasped once more (PICK_REGRASPS)."""
    robot = (robot_class or GraspRobot)(env)
    robot.settle = PATTERN_SETTLE
    start = robot.object("target")[0]
    for _ in range(1 + PICK_REGRASPS):
        centre = robot.object("target")[0]
        prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(centre[0], centre[1])))
        arm, hand, _, _, _, close_cmd = yield from robot.pick(
            "target", lift=PICK_ANYTHING_LIFT, close=True, prefer=prefer
        )
        yield from robot.move(arm, [(arm.grasp_point(), hand, close_cmd, PICK_HOLD)])
        if robot.object("target")[0][2] - start[2] >= PLACE_LIFTED:
            return
        print("[scripted_demo] target fell during the hold: grasping again")
        yield from robot.move(arm, [(arm.grasp_point(), hand, 1.0, 12)])


class CarryGraspRobot(GraspRobot):
    deep_first = True


def place_anything_v4_actions(env):
    """g1d_place_anything / g1d_put_anything_in with pick anything expert v4's grasp, slender objects deep first
    (CarryGraspRobot)."""
    return place_anything_actions(env, CarryGraspRobot)


# g1d_place_anything / g1d_put_anything_in: X's bottom clears the receptacle's top by PLACE_CARRY on the way over,
# carried at PLACE_SPEED at most (held by pick anything's grasp PICK_ANYTHING_DEPTH below its top, a tall object
# swung and slipped when carried 30-40 cm in 48 actions); over the spot the hand waits up to PLACE_SETTLE actions for X
# to stop swinging, then gives up if X sits more than PLACE_SLIPPED off where it sat in the hand once lifted, else
# moves by where X is (at most PLACE_FIX). The open fingertips stay PLACE_DROP above a container's top as X is let go;
# actions held once the hand is back up: PLACE_HOLD (at 30, 4 of 12 accepted pilot episodes failed on replay, X still
# rocking within the acceptance's last HOLD frames). X risen less than PLACE_LIFTED by the grasp was missed: the script
# ends there (carrying nothing ran out the episode)
PLACE_CARRY, PLACE_SPEED, PLACE_SETTLE, PLACE_SLIPPED, PLACE_FIX = 0.04, 0.15, 30, 0.04, 0.03
PLACE_DROP, PLACE_HOLD, PLACE_LIFTED = 0.01, 60, 0.03


def place_anything_actions(env, robot_class=None):
    """g1d_place_anything, g1d_put_anything_in: pick_anything_actions' grasp, then X carried over the spot of the
    receptacle's region nearest where it lay (place_anything.spot_toward, at X's yaw in the hand: the hand on X's side
    reaches less far across), lowered onto a support's face, or until the open fingertips are PLACE_DROP above a
    container's top (X then drops in), let go with the jaw fully open and the hand backed off upward. `robot_class`:
    TopGripRobot (place v3's grasp) or GraspRobot (pick anything expert v4's)."""
    robot = (robot_class or TopGripRobot)(env)
    robot.settle = PATTERN_SETTLE
    start = robot.object("target")[0]
    prefer = dict.fromkeys(robot.home, np.radians(pick_anything.grip_yaw(start[0], start[1])))
    arm, hand, offset, grip_height, _, close_cmd = yield from robot.pick(
        "target", lift=PICK_ANYTHING_LIFT, close=True, prefer=prefer
    )
    if robot.object("target")[0][2] - start[2] < PLACE_LIFTED:
        print("[scripted_demo] target not lifted: missed, giving up")
        return
    geom = layout_entry(env, place_anything.LABEL)["receptacle"]
    size = layout_entry(env, "target")["physics"]["size"]
    y_pos, y_quat = (as_np(v) for v in robot.lm.get_instance_pose(0, label=place_anything.LABEL))
    y_rot = from_wxyz(y_quat)
    _, x_rot, _, x_bottom, _ = robot.object("target")
    rel = yaw_of(x_rot.as_matrix()) - yaw_of(y_rot.as_matrix())
    toward = y_rot.inv().apply(start - y_pos)[:2]
    spot = place_anything.spot_toward(geom, size, rel, toward)
    xy = (y_pos + y_rot.apply([spot[0], spot[1], 0.0]))[:2]
    face = y_pos[2] + y_rot.apply([0.0, 0.0, geom["floor"]])[2]
    top = y_pos[2] + y_rot.apply([0.0, 0.0, geom["top"]])[2]
    surface = face
    if geom["kind"] == "container":
        surface = max(face, top + PLACE_DROP + FINGER_TIPS - grip_height)
    hover = max(HOVER, top + PLACE_CARRY - surface, x_bottom - surface - PLACE_GAP)
    point = np.r_[xy - hand.apply(offset)[:2], surface + grip_height + PLACE_GAP]
    above = point + [0.0, 0.0, hover]
    carry = max(48, int(np.ceil(np.linalg.norm(above - arm.grasp_point()) / PLACE_SPEED * 30)))
    print(
        f"[scripted_demo] {geom['kind']} {geom['id']}: spot {np.round(spot, 3)} (yaw {np.degrees(rel):.0f} deg off),"
        f" X bottom to {surface:.3f} (face {face:.3f}, top {top:.3f}), hover {hover:.3f}, carry {carry}"
    )
    yield from robot.move(arm, [(above, hand, close_cmd, carry)])
    robot.settle, before = None, robot.object("target")[0]
    for _ in range(PLACE_SETTLE):
        yield from robot.move(arm, [(above, hand, close_cmd, 1)])
        now = robot.object("target")[0]
        if np.linalg.norm(now - before) * 30 < SETTLED_SPEED:
            break
        before = now
    robot.settle = PATTERN_SETTLE
    held = arm.grasp_point() + arm.pose()[1].apply(offset)
    slipped = float(np.linalg.norm(now - held))
    if slipped > PLACE_SLIPPED:
        print(f"[scripted_demo] target {slipped * 1000:.0f} mm off where it sat in the hand: slipped, giving up")
        return
    fix = xy - now[:2]
    fix *= min(1.0, PLACE_FIX / max(float(np.linalg.norm(fix)), 1e-9))
    print(
        f"[scripted_demo] target {np.linalg.norm(xy - now[:2]) * 1000:.0f} mm off over the spot, moved {fix.round(3)}"
    )
    keys = [
        (point, hand, close_cmd, 30),
        (point, hand, TURNED_RELEASE, 12),  # release
        (point + [0, 0, hover], hand, TURNED_RELEASE, 24),
    ]
    yield from robot.move(arm, [(p + [*fix, 0.0], r, g, n) for p, r, g, n in keys])
    yield from robot.move(arm, [(arm.grasp_point(), arm.pose()[1], TURNED_RELEASE, PLACE_HOLD)])
    end = robot.object("target")[0]
    print(f"[scripted_demo] target at {end.round(3)}, receptacle at {robot.object(place_anything.LABEL)[0].round(3)}")


SETTLED_SPEED = 0.01  # m/s X counts as no longer swinging in the hand


# g1d_handover_anything: the giver (the hand on X's side) takes X from the top handover_anything.shift off its centre,
# away from where its fingers point, lifts it HO_LIFT, then carries it level to the handover: X's centre within HO_X
# of the midline, HO_Y out, the grips HO_GRIP above the desk (drawn per layout; X's bottom at least HO_AIR up), its
# fingers turned to point at the taker. Once X has stopped swinging (PLACE_SETTLE, SETTLED_SPEED) the taker takes
# the other end from the top, its fingers pointing back, HO_DEPTH under X's top (at most half its height): from
# HO_HOVER above and HO_OUT further out, down onto it, shut. Holding (both pads press, handover_anything.held), it
# waits HO_WAIT, the giver opens fully, waits HO_WAIT, rises HO_RISE and parks HO_PARK (out, back, up) in the
# background while the taker carries X onto the support as g1d_place_anything does. A taker that closed on nothing
# opens, backs off and tries again up to HO_TRIES times. Both carries turn the wrist at most HO_TURN: an off-centre X
# swung out of the pads when turned ~90 deg in 2 s (screen 11-19). Held off its centre X sags about the jaw's closing
# axis; over the support the taker turns its hand that way to bring X's long side level (at most HO_LEVEL) before
# setting it down: X set down 10-30 deg off level toppled or slid off (screen 11-19)
HO_LIFT, HO_GRIP, HO_X, HO_Y, HO_AIR = 0.10, (0.17, 0.23), 0.02, (-0.46, -0.38), 0.06
HO_HOVER, HO_OUT, HO_DEPTH, HO_TRIES, HO_WAIT, HO_RISE = 0.06, 0.03, 0.025, 2, 10, 0.06
HO_PARK = np.array([0.10, -0.05, 0.12])  # m: x outward on the giver's side
HO_TURN, HO_LEVEL, HO_LEVEL_MIN = np.radians(30.0), np.radians(25.0), np.radians(3.0)  # rad/s, rad, rad


def turn_actions(a, b):
    """Actions for a hand turning from rotation `a` to `b` at HO_TURN (30 a second)."""
    return int(np.ceil((b * a.inv()).magnitude() / HO_TURN * 30))


def settle_in(robot, arm, at, hand):
    """Hold `arm` at (`at`, `hand`), closed, until the held target stops swinging (SETTLED_SPEED, at most
    PLACE_SETTLE actions); returns where the target is then."""
    robot.settle, before = None, robot.object("target")[0]
    now = before
    for _ in range(PLACE_SETTLE):
        yield from robot.move(arm, [(at, hand, 0.0, 1)])
        now = robot.object("target")[0]
        if np.linalg.norm(now - before) * 30 < SETTLED_SPEED:
            break
        before = now
    robot.settle = PATTERN_SETTLE
    return now


def leveled(hand, long_axis):
    """`hand` turned about its jaw's horizontal closing axis so that X's `long_axis` (world) is level, by at most
    HO_LEVEL (None when it is within HO_LEVEL_MIN already)."""
    fingers = hand.apply([1.0, 0.0, 0.0])
    closing = np.cross([fingers[0], fingers[1], 0.0], [0.0, 0.0, 1.0])
    closing /= np.linalg.norm(closing)
    off = np.arcsin(np.clip(long_axis[2] / np.linalg.norm(long_axis), -1.0, 1.0))
    if abs(off) < HO_LEVEL_MIN:
        return None
    turns = [Rotation.from_rotvec(closing * s * min(abs(off), HO_LEVEL)) for s in (1.0, -1.0)]
    return min(turns, key=lambda r: abs(r.apply(long_axis)[2])) * hand


def jaw_state(env, arm):
    """(opening 0 closed .. 1 open, finger pad forces N) of `arm`'s gripper (func_parser._gripper_opening)."""
    rm, robot = env.robot_manager, arm.robot
    val = float(np.mean(rm.get_end_effector_real_val(robot=robot, env_idx_list=[0])[0]))
    lo, hi = robot.gripper_scale
    opening = (val - lo) / (hi - lo) if robot.gripper_move["sign"] == 1 else (hi - val) / (hi - lo)
    forces = rm.get_gripper_force(robot, [0])
    return opening, np.zeros(2) if forces is None else np.asarray(forces[0], dtype=float)


def top_hand(fingers):
    """TOP_ROT turned so the fingers point along horizontal `fingers`."""
    return Rotation.from_euler("z", np.arctan2(-fingers[0], fingers[1])) * TOP_ROT


def handover_anything_actions(env):
    """g1d_handover_anything: the giver's off-centre top grasp, the level carry to the handover in front of the chest,
    the taker's top grasp on the other end, the giver letting go and backing off, the taker setting X on the support
    (place_anything_actions' placing)."""
    robot = TopGripRobot(env)
    robot.settle = PATTERN_SETTLE
    entry = layout_entry(env, "target")
    rng = np.random.default_rng(zlib.crc32(repr(entry["default_pos"]).encode()))
    start, rot0, ext, bottom0, top0 = robot.object("target")
    side = 1.0 if start[0] > 0 else -1.0
    giver = robot.arm_for(start[0])
    taker = next(a for a in robot.arms if a is not giver)
    toward = np.array([-side, 0.0, 0.0])  # giver -> taker
    shift = handover_anything.shift(entry["physics"]["size"])
    axes = rot0.as_matrix()
    up = int(np.argmax(np.abs(axes[2])))
    narrow = min((a for a in range(3) if a != up), key=lambda a: ext[a])
    long_ax = 3 - up - narrow
    # the giver: pick anything's grasp, shifted along the long side away from the fingertips
    yaw = np.arctan2(axes[1, narrow], axes[0, narrow])
    want = np.radians(pick_anything.grip_yaw(start[0], start[1]))
    hand = Rotation.from_euler("z", yaw + np.pi * round((want - yaw) / np.pi)) * TOP_ROT
    fingers = hand.apply([1.0, 0.0, 0.0])[:2]
    fingers /= np.linalg.norm(fingers)
    grip = np.r_[start[:2] - shift * fingers, robot.grip_z(bottom0, top0)]
    open_cmd = robot.opening(ext[narrow])
    print(
        f"[scripted_demo] handover: {giver.arm_key} takes target at {start.round(3)} size {ext.round(3)}"
        f" {shift * 1000:.0f} mm off centre, {taker.arm_key} takes over"
    )
    yield from robot.move(
        giver,
        [
            (grip + [0, 0, PICK_ANYTHING_LIFT], hand, open_cmd, 54),
            (grip, hand, open_cmd, robot.DESCEND),
            (grip, hand, 0.0, 18),
            (grip + [0, 0, HO_LIFT], hand, 0.0, 30),
        ],
    )
    centre = robot.object("target")[0]
    if centre[2] - start[2] < PLACE_LIFTED:
        print("[scripted_demo] target not lifted: missed, giving up")
        return
    offset = giver.pose()[1].inv().apply(centre - giver.grasp_point())
    # the handover
    g = max(rng.uniform(*HO_GRIP), grip[2] - bottom0 + HO_AIR)
    meet = np.array([rng.uniform(-HO_X, HO_X), rng.uniform(*HO_Y)])
    hand_h = top_hand(toward)
    grip_h = np.r_[meet - hand_h.apply(offset)[:2], bottom0 + g]
    carry = max(60, int(np.ceil(np.linalg.norm(grip_h - giver.grasp_point()) / PLACE_SPEED * 30)))
    carry = max(carry, turn_actions(hand, hand_h))
    print(f"[scripted_demo] handover at {meet.round(3)}, grips {g:.3f} above the desk, carry {carry}")
    yield from robot.move(
        giver, [(np.r_[giver.grasp_point()[:2], grip_h[2]], hand, 0.0, 30), (grip_h, hand_h, 0.0, carry)]
    )
    now = yield from settle_in(robot, giver, grip_h, hand_h)
    slipped = float(np.linalg.norm(now - giver.grasp_point() - giver.pose()[1].apply(offset)))
    if slipped > PLACE_SLIPPED:
        print(f"[scripted_demo] target {slipped * 1000:.0f} mm off where it sat in the giver: slipped, giving up")
        return
    # the taker: the other end from the top, fingers pointing back at the giver
    for attempt in range(HO_TRIES):
        centre, rot, ext_now, _, _ = robot.object("target")
        m = rot.as_matrix()
        u = m[:, long_ax] * np.sign(m[:, long_ax] @ toward)
        upv = m[:, up] * np.sign(m[2, up])
        point = centre + shift * u + upv * (ext_now[up] / 2 - min(ext_now[up] / 2, HO_DEPTH))
        out = np.r_[u[:2] / np.linalg.norm(u[:2]), 0.0]
        hand_t = top_hand(-out[:2])
        open_t = robot.opening(ext_now[narrow])
        print(f"[scripted_demo] taker try {attempt}: grip {point.round(3)}")
        yield from robot.move(
            taker,
            [
                (point + [0, 0, HO_HOVER] + HO_OUT * out, hand_t, open_t, 75),
                (point, hand_t, open_t, 45),
                (point, hand_t, 0.0, 18),
            ],
        )
        yield from robot.move(taker, [(point, hand_t, 0.0, 6)])
        opening, forces = jaw_state(env, taker)
        print(f"[scripted_demo] taker jaw {opening:.2f}, pads {forces.round(1)} N")
        if handover_anything.held(opening, forces):
            break
        yield from robot.move(
            taker, [(point, hand_t, open_t, 12), (point + [0, 0, HO_HOVER] + HO_OUT * out, hand_t, open_t, 30)]
        )
    else:
        print("[scripted_demo] taker never held the target: giving up")
        return
    # the giver lets go and backs off
    yield from robot.move(taker, [(point, hand_t, 0.0, HO_WAIT)])
    yield from robot.move(giver, [(grip_h, hand_h, TURNED_RELEASE, 15)])
    yield from robot.move(taker, [(point, hand_t, 0.0, HO_WAIT)])
    yield from robot.move(giver, [(grip_h + [0, 0, HO_RISE], hand_h, TURNED_RELEASE, 20)])
    robot.send_aside(giver, grip_h + HO_PARK * [side, 1.0, 1.0], 45)
    # the taker sets X on the support (place_anything_actions)
    geom = layout_entry(env, place_anything.LABEL)["receptacle"]
    size = entry["physics"]["size"]
    y_pos, y_quat = (as_np(v) for v in robot.lm.get_instance_pose(0, label=place_anything.LABEL))
    y_rot = from_wxyz(y_quat)
    centre, x_rot, _, x_bottom, _ = robot.object("target")
    hand_now = taker.pose()[1]
    hand_p = Rotation.from_euler("z", np.radians(pick_anything.grip_yaw(y_pos[0], y_pos[1]))) * TOP_ROT
    turn = Rotation.from_euler("z", yaw_of(hand_p.as_matrix()) - yaw_of(hand_now.as_matrix()))
    rel = yaw_of((turn * x_rot).as_matrix()) - yaw_of(y_rot.as_matrix())
    spot = place_anything.spot_toward(geom, size, rel, geom["region"]["center"])
    xy = (y_pos + y_rot.apply([spot[0], spot[1], 0.0]))[:2]
    face = y_pos[2] + y_rot.apply([0.0, 0.0, geom["floor"]])[2]
    top = y_pos[2] + y_rot.apply([0.0, 0.0, geom["top"]])[2]
    offset = hand_now.inv().apply(centre - taker.grasp_point())
    grip_height = taker.grasp_point()[2] - x_bottom
    hover = max(HOVER, top + PLACE_CARRY - face, x_bottom - face - PLACE_GAP)
    point = np.r_[xy - hand_p.apply(offset)[:2], face + grip_height + PLACE_GAP]
    above = point + [0.0, 0.0, hover]
    carry = max(48, int(np.ceil(np.linalg.norm(above - taker.grasp_point()) / PLACE_SPEED * 30)))
    carry = max(carry, turn_actions(hand_now, hand_p))
    print(
        f"[scripted_demo] support {geom['id']}: spot {np.round(spot, 3)} (yaw {np.degrees(rel):.0f} deg off),"
        f" carry {carry}"
    )
    yield from robot.move(taker, [(above, hand_p, 0.0, carry)])
    now = yield from settle_in(robot, taker, above, hand_p)
    # level X's long side (it sags about the closing axis), then measure it again for the set-down
    _, x_rot, _, _, _ = robot.object("target")
    level = leveled(hand_p, x_rot.as_matrix()[:, long_ax])
    if level is not None:
        print(f"[scripted_demo] leveling X: hand turned {np.degrees((level * hand_p.inv()).magnitude()):.0f} deg")
        yield from robot.move(taker, [(above, level, 0.0, max(20, turn_actions(hand_p, level)))])
        now = yield from settle_in(robot, taker, above, level)
        hand_p = level
        centre, _, _, x_bottom, _ = robot.object("target")
        offset = hand_p.inv().apply(centre - taker.grasp_point())
        grip_height = taker.grasp_point()[2] - x_bottom
        point = np.r_[xy - hand_p.apply(offset)[:2], face + grip_height + PLACE_GAP]
    slipped = float(np.linalg.norm(now - taker.grasp_point() - taker.pose()[1].apply(offset)))
    if slipped > PLACE_SLIPPED:
        print(f"[scripted_demo] target {slipped * 1000:.0f} mm off where it sat in the taker: slipped, giving up")
        yield from robot.finish()
        return
    fix = xy - now[:2]
    fix *= min(1.0, PLACE_FIX / max(float(np.linalg.norm(fix)), 1e-9))
    keys = [
        (point, hand_p, 0.0, 30),
        (point, hand_p, TURNED_RELEASE, 12),  # release
        (point + [0, 0, hover], hand_p, TURNED_RELEASE, 24),
    ]
    yield from robot.move(taker, [(p + [*fix, 0.0], r, gr, n) for p, r, gr, n in keys])
    yield from robot.finish()
    yield from robot.move(taker, [(taker.grasp_point(), taker.pose()[1], TURNED_RELEASE, PLACE_HOLD)])
    end = robot.object("target")[0]
    print(f"[scripted_demo] target at {end.round(3)}, support at {robot.object(place_anything.LABEL)[0].round(3)}")


# g1d_open_anything: the open pads wait OPEN_APPROACH in front of the handle, close on it, then the hand follows the
# part's own motion (an arc about the hinge, a line along the slide) OPEN_LEAD steps of OPEN_RATE ahead of where the
# joint is, until it is OPEN_OVER x the goal open, and holds it there OPEN_HOLD actions. It does not let go: the
# libraries' joint drives pull a released door or drawer back toward closed, and a lid falls back
OPEN_APPROACH, OPEN_HOLD = 0.10, 60
# the arm must hold the hand within OPEN_REACH m and OPEN_TURN deg of the pose in front of the handle once it settled
# there, and then slide onto the grip within OPEN_MISS (else it backs out, the jaw still open), else the next pose is
# tried (the fingers OPEN_STEEPER deg further down; the hand turned half about the fingers sent the IK 0.3 m astray,
# screen9, and the other arm reaching across while the first went home broke PhysX in 136 episodes, screen11), else
# it gives up: Jacobian IK that could not get there drove the arm into the cabinet and broke PhysX (screen8: 83 of 303)
OPEN_REACH, OPEN_TURN = 0.015, 8.0
# level fingers low in front of the chest need the wrist bent past its range (the hands start at z 1.23; screen11
# reached 50 of 195 approach poses)
OPEN_STEEPER = 25.0
# (m, actions) the hand waits at each pose of the approach: two tries must leave the 600 actions room to open
OPEN_SETTLE = (0.005, 25)
# m: an edge thinner than this leaves the pads about shut on it, so "closed on nothing" is not told by the gap
EDGE_THIN = 0.008
# the fingertips stop TIP_CLEAR short of the face a handle stands out of (the pads then hold only a short handle's
# front); a grip that closed more than OPEN_MISS off the handle, or with the pads less than OPEN_HELD of the handle's
# width apart (on nothing), is given up, not pulled: an arm pressing on a fixed cabinet threw the robot over
TIP_CLEAR, OPEN_MISS, OPEN_HELD = 0.002, 0.04, 0.5
# deg the fingers point down from level, about the horizontal across the part: level fingers at a handle 20-30 cm above
# the desk left the arm 2-5 cm short (screen 11-18)
OPEN_PITCH = 25.0
# deg the fingers lean in toward the face when they come along it (hook, edge): the palm, ~5 cm thick about the
# finger line, then clears the face while the fingertips reach behind the bar or the panel
ALONG_TILT = 25.0
OPEN_RATE = {"revolute": np.radians(0.8), "prismatic": 0.003}  # per action
OPEN_LEAD, OPEN_OVER = 4, open_anything.OPEN_OVER  # the layouts keep the robot clear of the part this far open
OPEN_STALL = 45  # actions without the joint gaining a fifth of OPEN_RATE each: the grip slipped
OPEN_MOST = 360  # actions of following


class OpenPart:
    """The part to open of the layout's object (scripts/objects/open_anything.py "open"): its geometry in the
    object's frame, the object's pose (fixed base) and the joint's value in the sim."""

    def __init__(self, env):
        self.lm = lm = env.scene_manager.layout_manager
        self.inst = lm.get_scene_object(0, lm.get_instance_name(0, "target"))
        self.part = env.target_part(0)
        self.refresh()
        p = self.part
        self.pivot, self.axis = np.asarray(p["pivot"]), np.asarray(p["axis"])

    def refresh(self):
        """Read the object's pose again (a push can move even a fixed base a little)."""
        pos, quat = self.lm.get_instance_pose(0, label="target")
        self.pos, self.rot = as_np(pos), from_wxyz(as_np(quat))

    def q(self):
        return float(self.inst.get_joint_info(self.part["joint"])["position"])

    def opening(self, q=None):
        return self.part["sign"] * ((self.q() if q is None else q) - self.part["closed"])

    def joint_rot(self, dq):
        """Object-frame rotation of the part at dq past its authored pose (identity for a drawer)."""
        return Rotation.from_rotvec(self.axis * dq) if self.part["type"] == "revolute" else Rotation.identity()

    def to_env(self, local, q):
        """A point of the part, given in the object frame at its authored (closed) pose, at joint value q."""
        dq = q - self.part["closed"]
        if self.part["type"] == "revolute":
            local = self.pivot + self.joint_rot(dq).apply(np.asarray(local) - self.pivot)
        else:
            local = np.asarray(local) + self.axis * dq
        return self.pos + self.rot.apply(local)

    def to_part(self, point, q):
        """Inverse of to_env: an env point on the part at q, in the object frame at the authored pose."""
        local = self.rot.inv().apply(np.asarray(point) - self.pos)
        dq = q - self.part["closed"]
        if self.part["type"] == "revolute":
            return self.pivot + self.joint_rot(dq).inv().apply(local - self.pivot)
        return local - self.axis * dq

    def turn(self, q_from, q_to):
        """Env-frame rotation the part makes from q_from to q_to."""
        return self.rot * self.joint_rot(q_to - q_from) * self.rot.inv()


def open_hand(part, arm, more=0.0):
    """Hand rotation for the part's method (open_anything.METHODS): fingers along its `approach` (into the face for a
    pinch, OPEN_PITCH down when the jaw closes sideways; along the face for a hook or an edge, leaning ALONG_TILT in
    toward it), the jaw closing along `close`, the wrist camera up (or, when that is level, turned out to the arm's
    side); `more` deg further down."""
    p = part.part
    front = part.rot.apply(np.asarray(p["front"]))
    x = part.rot.apply(np.asarray(p.get("approach") or -np.asarray(p["front"])))
    if p.get("method", "pinch") != "pinch":
        x = np.cos(np.radians(ALONG_TILT)) * x - np.sin(np.radians(ALONG_TILT)) * front
    x /= np.linalg.norm(x)
    y = part.rot.apply(np.asarray(p["close"]))
    y = y - (y @ x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    out = np.sign(arm.root()[0] - part.pos[0]) or 1.0
    if z[2] < -1e-3 or (abs(z[2]) <= 1e-3 and z[0] * out < 0):
        y, z = -y, -z
    hand = Rotation.from_matrix(np.c_[x, y, z])
    # a lying bar keeps level fingers unless asked: tilted, the pads would slide off it up or down
    pitch = more + (OPEN_PITCH if p.get("method", "pinch") == "pinch" and abs(y[2]) < 0.5 else 0.0)
    side = np.cross(x, [0.0, 0.0, 1.0])
    if pitch and np.linalg.norm(side) > 0.3:
        hand = Rotation.from_rotvec(-np.radians(pitch) * side / np.linalg.norm(side)) * hand
    return hand


def open_anything_actions(env):
    """g1d_open_anything: the arm on the handle's side (where it is half way to the goal) comes in front of the
    handle with the jaw open, slides onto it and closes; then each action aims the hand where the part takes the
    grip OPEN_LEAD steps further open (the grip's place on the part as it closed, so the arc or line is the part's
    own) until the part is OPEN_OVER x its goal open or stalls; held there."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    part = OpenPart(env)
    p = part.part
    q0 = part.q()
    goal_q = p["closed"] + p["sign"] * p["goal"]
    method = p.get("method", "pinch")
    back = max(FINGER_TIPS + TIP_CLEAR - p["depth"] / 2, 0.0) if method == "pinch" else 0.0
    grip_local = np.asarray(p["grasp"]) + back * np.asarray(p["front"])
    grasp = part.to_env(grip_local, q0)
    arm = robot.arm_for((grasp[0] + part.to_env(grip_local, goal_q)[0]) / 2)
    open_cmd = 0.0 if method == "hook" else pad_open(p["width"])  # a hook goes in with the fingers together
    print(
        f"[scripted_demo] open {p['id']} {p['joint']} ({p['role']}, {p['type']}) by {method}, goal {p['goal']:.3f},"
        f" at {part.opening(q0):.3f}, grip {grasp.round(3)} width {p['width']:.3f} with {arm.arm_key}"
    )
    start = robot.home[arm.arm_key][0]
    arm.rest = robot.home_q[arm.arm_key]
    robot.settle = OPEN_SETTLE
    for hand in (open_hand(part, arm), open_hand(part, arm, OPEN_STEEPER)):
        fingers_in = hand.apply([1.0, 0.0, 0.0])
        before = grasp - fingers_in * OPEN_APPROACH
        # the hand turns on the way, half way there, then goes in front of the grip
        yield from robot.move(arm, [((start + before) / 2, hand, open_cmd, 30), (before, hand, open_cmd, 30)])
        off = float(np.linalg.norm(arm.grasp_point() - before))
        turn = float(np.degrees((hand * arm.pose()[1].inv()).magnitude()))
        print(
            f"[scripted_demo] in front of the grip: {off * 1000:.0f} mm, {turn:.0f} deg off"
            f" (at {arm.grasp_point().round(3)}, from {start.round(3)}; joints at a limit {arm.at_limits()})"
        )
        if off > OPEN_REACH or turn > OPEN_TURN:
            continue
        yield from robot.move(arm, [(grasp, hand, open_cmd, 40)])
        slid = float(np.linalg.norm(arm.grasp_point() - grasp))
        print(f"[scripted_demo] slid onto the grip: {slid * 1000:.0f} mm off")
        if slid <= OPEN_MISS:
            break
        yield from robot.move(arm, [(before, hand, open_cmd, 30)])
    else:
        print("[scripted_demo] cannot reach the grip, giving up")
        for _ in range(10):  # stand still: an expert with no actions is an error, not a failed episode
            yield dict(robot.hold) | robot.grip
        return
    part.refresh()
    grasp = part.to_env(grip_local, part.q())
    if np.linalg.norm(arm.grasp_point() - grasp) > OPEN_MISS:  # the part or the object moved under the jaw
        yield from robot.move(
            arm, [(arm.grasp_point() - hand.apply([1.0, 0.0, 0.0]) * OPEN_APPROACH, hand, open_cmd, 30)]
        )
        print("[scripted_demo] the grip moved away, backed out")
        return
    robot.settle = PATTERN_SETTLE
    yield from robot.move(arm, [(grasp, hand, 0.0, 15)])
    q_grip = part.q()
    on_part = part.to_part(arm.grasp_point(), q_grip)
    held = arm.pose()[1]
    miss = float(np.linalg.norm(arm.grasp_point() - grasp))
    fingers = as_np(arm.key.data.joint_pos[0, list(arm.robot.gripper_joint_indices)])
    gap = PAD_CLOSED + PAD_TRAVEL * float(np.clip((0.023 - fingers.mean()) / 0.0445, 0.0, 1.0))
    print(
        f"[scripted_demo] closed {miss * 1000:.0f} mm off the grip, pads {gap * 1000:.0f} mm apart,"
        f" part at {part.opening(q_grip):.3f}"
    )
    empty = gap - PAD_CLOSED < max(OPEN_HELD * (p["width"] - 0.01 - PAD_CLOSED), 0.002)
    if miss > OPEN_MISS or (empty and method != "hook" and p["width"] >= EDGE_THIN):
        print("[scripted_demo] missed the handle, giving up")
        return
    rate = OPEN_RATE[p["type"]]
    end = p["closed"] + p["sign"] * min(p["goal"] * OPEN_OVER, p["travel"])
    best, since = part.opening(), 0
    for _ in range(OPEN_MOST):
        q = part.q()
        if part.opening(q) >= part.opening(end) - 1e-4 or since >= OPEN_STALL:
            break
        q_t = q + p["sign"] * rate * OPEN_LEAD
        q_t = min(q_t, end) if p["sign"] > 0 else max(q_t, end)
        yield robot._command(arm, part.to_env(on_part, q_t), part.turn(q_grip, q_t) * held, 0.0)
        if part.opening() > best + rate / 5:
            best, since = part.opening(), 0
        else:
            since += 1
    q = part.q()
    point, rot = part.to_env(on_part, q), part.turn(q_grip, q) * held
    print(f"[scripted_demo] opened to {part.opening(q):.3f} (goal {p['goal']:.3f}){' stalled' if since else ''}")
    robot.settle = None
    yield from robot.move(arm, [(point, rot, 0.0, OPEN_HOLD)])


def put_in_tray_actions(env):
    """g1d_put_block_in_tray: the named block dropped into the tray from just above its walls (block_sort_actions)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    table = robot.object("target_block")[3]
    grasp = yield from atomic_pick(robot, "target_block")
    tray = robot.object("tray")
    yield from robot.place(*grasp, tray[0][:2], tray[4], label="target_block", hover=SORT_HOVER)
    yield from atomic_done(robot, grasp[0], table)


def rotate_parallel_actions(env):
    """g1d_rotate_block_parallel: the block set down where it lay, long side along X (place_turned)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    start, _, _, bottom, _ = robot.object("block")
    grasp = yield from atomic_pick(robot, "block")
    yield from place_turned(robot, grasp, start[:2], bottom, "block")
    yield from atomic_done(robot, grasp[0], bottom)
    end, rot, _, _, _ = robot.object("block")
    off, moved = (np.degrees(yaw_of(rot.as_matrix())) + 90) % 180 - 90, np.linalg.norm(end[:2] - start[:2])
    print(f"[scripted_demo] block long side {off:.1f} deg off X, moved {moved * 100:.1f} cm")


def line_up_actions(env):
    """g1d_line_up_blocks: block_<i> to block_atomic.row_slots()[i], the left hand's blocks first (place_turned: the
    fingers pass in front of and behind a block, not into the 15 mm gap to its neighbour)."""
    robot = Robot(env)
    robot.settle = PATTERN_SETTLE
    labels = ["block_0", "block_1", "block_2"]
    table = min(robot.object(label)[3] for label in labels)
    order = sorted(robot.arms, key=lambda a: robot.home[a.arm_key][0][0])
    plan = sorted(enumerate(labels), key=lambda p: order.index(robot.arm_for(robot.object(p[1])[0][0])))
    slots = block_atomic.row_slots()
    for n, (i, label) in enumerate(plan):
        grasp = yield from atomic_pick(robot, label)
        yield from place_turned(robot, grasp, np.asarray(slots[i]), table, label)
        following = plan[n + 1][1] if n + 1 < len(plan) else None
        if following is None or robot.arm_for(robot.object(following)[0][0]) is not grasp[0]:
            x, y, up = PATTERN_ASIDE
            side = np.sign(robot.home[grasp[0].arm_key][0][0])
            robot.send_aside(grasp[0], np.array([side * x, y, table + up]), PATTERN_ASIDE_ACTIONS)
    yield from robot.finish()


PICK_LIFT, PICK_HOLD = 0.10, 60  # m above the grasp, actions held there


# experts whose commands go through Limiter (the others predate it; their datasets were made without it)
LIMITED = {
    "pick_bucket",
    "g1d_pour_water_deliver_to_lounge",
    "g1d_build_block_pattern",
    "g1d_sort_blocks_by_color",
    "g1d_sort_blocks_by_shape",
    "g1d_pour_beans_plus",
    "g1d_pick_anything",
    *place_anything.TASKS.values(),
    handover_anything.TASK,
    "g1d_open_anything",
    *block_atomic.TASKS,
}

DEMOS = {
    "g1d_pour_water_into_cup": pour_actions,
    "g1d_pour_beans_cup_to_cup": cup_to_cup_actions,
    "g1d_pour_beans_measuring_cups": cup_to_cup_actions,
    "g1d_pour_beans_half_cup": half_cup_actions,
    "g1d_pour_beans_plus": plus_actions,
    "g1d_pour_beans_half_cup_rigid": half_cup_actions,
    "g1d_general_pickup": pickup_actions,
    "g1d_pick_anything": pick_anything_actions,
    "g1d_place_anything": place_anything_actions,
    "g1d_put_anything_in": place_anything_actions,
    "g1d_handover_anything": handover_anything_actions,
    "g1d_open_anything": open_anything_actions,
    "g1d_stack_blocks": stack_actions,
    "g1d_deliver_cup_to_lounge": deliver_cup_actions,
    "g1d_pour_water_deliver_to_lounge": pour_deliver_actions,
    "g1d_build_block_pattern": block_pattern_actions,
    "g1d_sort_blocks_by_color": block_sort_actions,
    "g1d_sort_blocks_by_shape": block_sort_actions,
    "g1d_place_block_on_mark": place_on_mark_actions,
    "g1d_stack_block_on_block": stack_on_block_actions,
    "g1d_pick_named_block": pick_named_actions,
    "g1d_put_block_in_tray": put_in_tray_actions,
    "pick_bucket": lambda env: __import__("scripts.g1d.pick_bucket_demo", fromlist=["actions"]).actions(env),
    "g1d_rotate_block_parallel": rotate_parallel_actions,
    "g1d_line_up_blocks": line_up_actions,
}


class Limiter:
    """Real-robot motion limits on every command: arm joints REAL_JOINT_MARGIN inside their limits and at most
    REAL_ARM_VMAX, the gripper command at most REAL_GRIP_RATE per action. Logs the fastest requested move."""

    def __init__(self, env):
        self.arms = [Arm(env, r) for r in env.robot_manager.robot_list if r.type == "target"]
        self.q = {a.arm_key: a.q() for a in self.arms}
        self.grip = {}
        self.lim = {
            a.arm_key: as_np(a.key.data.soft_joint_pos_limits[0, a.cols]).reshape(-1, 2)
            + [REAL_JOINT_MARGIN, -REAL_JOINT_MARGIN]
            for a in self.arms
        }
        self.dq_max = REAL_ARM_VMAX / 30  # rad per 30 Hz action
        self.asked = self.asked_grip = 0.0

    def __call__(self, action):
        out = dict(action)
        for a in self.arms:
            if a.arm_key in out:
                q = np.clip(np.asarray(out[a.arm_key], dtype=float), *self.lim[a.arm_key].T)
                step = q - self.q[a.arm_key]
                self.asked = max(self.asked, float(np.abs(step).max()))
                self.q[a.arm_key] = self.q[a.arm_key] + np.clip(step, -self.dq_max, self.dq_max)
                out[a.arm_key] = self.q[a.arm_key]
            if a.grip_key in out:
                g = float(out[a.grip_key][0])
                prev = self.grip.get(a.grip_key, g)
                self.asked_grip = max(self.asked_grip, abs(g - prev))
                self.grip[a.grip_key] = prev + float(np.clip(g - prev, -REAL_GRIP_RATE, REAL_GRIP_RATE))
                out[a.grip_key] = [self.grip[a.grip_key]]
        return out


def eval_one_episode(TASK_ENV, model_client):
    TASK_ENV.get_obs()  # initial frame (also triggers the pre-action reward check so is_lift records table height)
    actions = DEMOS[TASK_ENV.task_name](TASK_ENV)
    limiter = Limiter(TASK_ENV) if TASK_ENV.task_name in LIMITED else None
    for action in actions:
        TASK_ENV.take_action(limiter(action) if limiter else action)
        if TASK_ENV.is_episode_end():
            break
        TASK_ENV.get_obs()  # streams the camera frames into the episode videos
    if TASK_ENV.task_name == "pick_bucket":
        import json
        import os

        trace_path = ROOT / "output" / (os.environ.get("ROBODOJO_RUN_ID", "pick_bucket") + "_trace.json")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps({"trace": TASK_ENV.trace[0], "history": TASK_ENV.progress[0].history}, indent=2)
        )
    if limiter:
        print(
            f"[scripted_demo] limiter: fastest requested arm step {limiter.asked:.3f} rad"
            f" ({limiter.asked * 30:.2f} rad/s, cap {REAL_ARM_VMAX}),"
            f" gripper {limiter.asked_grip:.3f} per action (cap {REAL_GRIP_RATE:.3f})"
        )
    if not TASK_ENV.end_flag[0]:  # EvalEnv starts each episode with success=True: running out of script is a fail
        TASK_ENV.success[0] = False
        TASK_ENV.end_flag[0] = True


def main(extra_kit_args="", policy_module="scripts.g1d.scripted_demo"):
    """Run src/eval_client/main.py in-process with `policy_module`'s eval_one_episode as the policy (no server).
    ROBODOJO_KIT_ARGS adds Kit arguments (the isaaclab_newton environment's extsDeprecated folder)."""
    extra_kit_args = f"{extra_kit_args} {os.environ.get('ROBODOJO_KIT_ARGS', '')}".strip()
    args = sys.argv[1:]
    defaults = {"--task_name": "g1d_pour_water_into_cup", "--seed": "0", "--device_id": "0"}
    args += [x for k, v in defaults.items() if k not in args for x in (k, v)]
    sys.argv = [str(ROOT / "src/eval_client/main.py"), *args, "--policy_name", POLICY, "--port", "0"]
    sys.argv += ["--policy_module", policy_module]
    sys.argv += ["--additional_info", "scripted", "--headless", "--enable_cameras"]
    sys.argv += [
        "--kit_args",
        f"--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera {extra_kit_args}".strip(),
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
