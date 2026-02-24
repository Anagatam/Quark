import numpy as np
from quark.base import BaseConstraint

class ConstraintManager(BaseConstraint):
    def __init__(self, max_assets: int, lower_bound: float, upper_bound: float):
        self.max_assets = max_assets
        self.lb = lower_bound
        self.ub = upper_bound

    def map_to_feasible_space(self, weights: np.ndarray) -> np.ndarray:
        w = np.copy(weights)
        w = np.clip(w, 0, None)
        active_idx = np.argsort(w)[-self.max_assets:]
        mask = np.zeros_like(w)
        mask[active_idx] = 1.0
        w = w * mask
        active_w = w[active_idx]
        if np.sum(active_w) > 0:
            active_w = active_w / np.sum(active_w)
            active_w = np.clip(active_w, self.lb, self.ub)
            active_w = active_w / np.sum(active_w)
            w[active_idx] = active_w
        else:
            w[active_idx] = 1.0 / self.max_assets
        return w
