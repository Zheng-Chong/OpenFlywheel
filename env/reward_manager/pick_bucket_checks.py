"""Ordered task progression, independent of Isaac and reusable for recorded pose audit."""

import numpy as np


class Progress:
    def __init__(self):
        self.stage = 0
        self.hold = 0
        self.last_pos = None
        self.samples = 0
        self.history = []

    def update(
        self,
        food_height,
        basket_height,
        inside,
        opening,
        basket_pos,
        *,
        food_near=False,
        handle_near=False,
        upright=True,
    ):
        pos = np.asarray(basket_pos, float)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite([food_height, basket_height, opening])):
            self.hold = 0
            return False
        speed = np.inf if self.last_pos is None else np.linalg.norm(pos - self.last_pos) * 30
        self.last_pos = pos.copy()
        self.samples += 1
        if self.stage == 0:
            good = food_height >= 0.020 and opening < 0.8 and food_near
        elif self.stage == 1:
            good = inside and opening >= 0.8
        elif self.stage == 2:
            good = inside and 0.02 < opening < 0.7 and basket_height < 0.030 and handle_near
        elif self.stage == 3:
            good = inside and basket_height >= 0.030 and 0.02 < opening < 0.7 and handle_near
        elif self.stage == 4:
            good = inside and abs(basket_height) <= 0.010 and opening >= 0.8 and speed < 0.03 and upright
        else:
            return inside and abs(basket_height) <= 0.010 and opening >= 0.8 and speed < 0.03 and upright
        self.hold = self.hold + 1 if good else 0
        required = 15 if self.stage in (3, 4) else 3
        if self.hold >= required:
            self.history.append({"stage": self.stage + 1, "sample": self.samples})
            self.stage += 1
            self.hold = 0
        return self.stage == 5
