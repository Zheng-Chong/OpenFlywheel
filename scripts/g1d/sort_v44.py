"""Sort anything expert v4.4: sort_v43 with PATTERN_SETTLE stall-break (place_v44.StallBreakMixin).

Kept out of sort_v43.py and scripted_demo.py so registered earlier expert sources stay unchanged.
"""

from scripts.g1d import place_v44, sort_anything_demo, sort_v43


class SortRobot44(place_v44.StallBreakMixin, sort_v43.SortRobot43):
    pass


def sort_anything_actions(env):
    """g1d_sort_anything with expert v4.4 (v4.3 + PATTERN_SETTLE stall-break)."""
    return sort_anything_demo.sort_anything_actions(env, SortRobot44, carry=sort_v43.carry)
