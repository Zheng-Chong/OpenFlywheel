"""scripts/datagen plugin for G1-D Sharpa Wave `g1d_pick_anything` using `affordance.json`.

Consumes Stage 1+2 verified Sharpa Wave `affordance.json["grasps"]["sharpa_wave"]` sidecars
(from the asset directory or `OPENFLYWHEEL_AFFORDANCE_DIR`) via `scripts.g1d.affordance_dexhand` and
automatically falls back to the geometric `x_anything_sharpa` heuristic when no affordance applies.

Records per-episode `grasp_type`, `grasp_source`, `grasp_id`, `execution_mode`, `hand_side`,
`part_tag`, `verified`, and `failure_attribution` in `summary()` / `quality.json`.

Collect / smoke-test with:
    OPENFLYWHEEL_AFFORDANCE_DIR=/path/to/affordances \\
    isaac-sim python scripts/datagen/collect.py \\
        --task g1d_pick_anything --plugin g1d_pick_anything_sharpa_wave \\
        --env-cfg g1d_office_sharpa_wave --seed <seed> --layout-ids 0..19 \\
        --output /shared/tmp/$USER/sharpa-s3-smoke --device cuda:0 --device_id 0
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from scripts.datagen.plugins import g1d_pick_anything as base
from scripts.g1d import affordance_dexhand as dex_plan

TASK = base.TASK
ACCEPTANCE = "pick_anything_sharpa_wave_v1"
STEP_LIM = 900  # 30 s at 30 Hz: wholebody adapt + approach + staged/pinch closure + lift
TAIL = base.TAIL
HOLD = base.HOLD
audit = base.audit


class Episode(base.Episode):
    """Pick-anything episode driven by Sharpa Wave affordance planner with per-episode diagnostics."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.grasp_info: dict[str, Any] = {}

    def actions(self):
        verified_only = os.environ.get("OPENFLYWHEEL_SHARPA_VERIFIED_ONLY", "").strip().lower() in ("1", "true", "yes")
        return dex_plan.pick_anything_sharpa_affordance_actions(
            self.env,
            verified_only=verified_only,
            info_out=self.grasp_info,
        )

    def summary(self):
        out = super().summary()
        poses = np.asarray(self.history["object_poses"], dtype=np.float64)
        attribution = dex_plan.classify_pick_failure(
            poses,
            self.labels,
            np.asarray(self.meta["start"], dtype=np.float64),
            lift_thresh=base.LIFT,
            stay_xy=base.STAY,
            stay_z=base.STAY_Z,
            hold_frames=HOLD,
        )
        out.update(
            {
                "grasp_source": self.grasp_info.get("grasp_source", "unknown"),
                "grasp_type": self.grasp_info.get("grasp_type", "unknown"),
                "grasp_id": self.grasp_info.get("grasp_id", ""),
                "execution_mode": self.grasp_info.get("execution_mode", "unknown"),
                "hand_side": self.grasp_info.get("hand_side", "unknown"),
                "part_tag": self.grasp_info.get("part_tag", "unknown"),
                "verified": bool(self.grasp_info.get("verified", False)),
                "candidates_raw": int(self.grasp_info.get("candidates_raw", 0)),
                "candidates_after_filter": int(self.grasp_info.get("candidates_after_filter", 0)),
                "tactile_trigger": self.grasp_info.get("tactile_trigger"),
                "peak_force_n": self.grasp_info.get("peak_force_n"),
                "failure_attribution": attribution,
            }
        )
        return out


__all__ = ["ACCEPTANCE", "HOLD", "STEP_LIM", "TAIL", "TASK", "Episode", "audit"]
