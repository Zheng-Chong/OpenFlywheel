"""Six active RH56DFTP axes per hand; followers are native constraints."""

import json
from pathlib import Path

import numpy as np

from env.robot_manager.inspire_contract import HAND_TARGET_LEAD_RAD
from env.robot_manager.robot_class.g1d_dexhands import DexHandLeft, DexHandRight


class InspireTargets:
    hand_target_lead_rad = HAND_TARGET_LEAD_RAD

    def validate_hand_position(self, target):
        if not hasattr(self, "_hand_limits"):
            contract = json.loads((Path(self.robot_file) / "model_contract.json").read_text())
            limits = contract["hands"][self.config_key]["effective_active_limits"]
            self._hand_limits = np.array(
                [[limits[n][key] for key in ("lower", "upper")] for n in self.gripper_joints_name]
            )
        values = np.asarray(target, dtype=float)
        if (
            values.shape != (6,)
            or not np.isfinite(values).all()
            or np.any(values < self._hand_limits[:, 0] - 1e-7)
            or np.any(values > self._hand_limits[:, 1] + 1e-7)
        ):
            raise ValueError("RH56DFTP requires six finite active radians within the coupled feasible limits")


class InspireLeft(InspireTargets, DexHandLeft):
    pass


class InspireRight(InspireTargets, DexHandRight):
    pass
