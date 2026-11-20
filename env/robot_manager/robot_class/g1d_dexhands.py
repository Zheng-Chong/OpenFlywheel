"""Independent radian hand commands; no Dex1 finger coupling or scalar aperture."""

import json
from pathlib import Path

import numpy as np

from env.robot_manager.robot_class.g1d import G1DLeft, G1DRight


class DexHandTargets:
    def hand_limits(self):
        """(lower, upper) rad per active hand joint, robot config order."""
        if not hasattr(self, "_hand_limits"):
            contract = json.loads((Path(self.robot_file) / "model_contract.json").read_text())
            hand = contract["hands"][self.config_key]
            limits = hand.get("effective_active_limits", hand["limits"])
            self._hand_limits = np.array(
                [[limits[name][key] for key in ("lower", "upper")] for name in self.gripper_joints_name]
            )
        return self._hand_limits

    def hand_openness(self, position):
        """0 closed .. 1 open, the Dex1 gripper's opening share for a hand: one minus the mean share of its range
        each active joint is away from its open pose (0 rad, or its nearest limit)."""
        lim = self.hand_limits()
        q = np.asarray(position, dtype=float).reshape(-1)
        rest = np.clip(0.0, lim[:, 0], lim[:, 1])
        return float(1.0 - np.mean(np.abs(q - rest) / (lim[:, 1] - lim[:, 0])))

    def validate_hand_position(self, target):
        DexHandTargets.hand_limits(self)
        values = np.asarray(target, dtype=float)
        if (
            values.shape != (len(self.gripper_joints_name),)
            or not np.isfinite(values).all()
            or np.any(values < self._hand_limits[:, 0] - 1e-7)
            or np.any(values > self._hand_limits[:, 1] + 1e-7)
        ):
            raise ValueError("Hand commands require one finite radian target per active joint within its limits")


class DexHandLeft(DexHandTargets, G1DLeft):
    finger_coupling = None
    preserve_joint_order = True

    def __init__(self, cfg):
        self.asset_name = cfg["robot_name"]
        super().__init__(cfg)


class DexHandRight(DexHandTargets, G1DRight):
    finger_coupling = None
    preserve_joint_order = True

    def __init__(self, cfg):
        self.asset_name = cfg["robot_name"]
        super().__init__(cfg)
