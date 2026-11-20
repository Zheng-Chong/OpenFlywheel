"""Pick anything expert v4.1 (docs/pick-expert-v4.md「v4.1」): scripted_demo.GraspRobot whose first try is v3's
grasp through the close wherever the grasp point ended, and that never takes a slender object deep first.

v4's A/B round 5: on Pick the first try gave up grasp points 3.1-3.7 cm beside the target that v3 closed on and lifted
(3 of its 4 losses); on Place / Put the deep first try of CarryGraspRobot and its retries lost tall objects and
right-hand ones. Later tries keep v4's candidates and checks. Kept out of scripted_demo so the registered Pick v4
sources stay as they are."""

from scripts.g1d import scripted_demo as demo


class GraspRobot41(demo.GraspRobot):
    first_try_grip_check = False
    deep_first = False


def pick_anything_actions(env):
    """g1d_pick_anything with v4.1's grasp, v4's hold and regrasp."""
    return demo.pick_anything_v4_actions(env, GraspRobot41)


def place_anything_actions(env):
    """g1d_place_anything / g1d_put_anything_in with v4.1's grasp."""
    return demo.place_anything_actions(env, GraspRobot41)
