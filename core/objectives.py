import numpy as np
from numba import njit
from quark.base import BaseObjective
from quark.core.luminescence import calculate_cvar, calculate_max_drawdown

@njit(fastmath=True)
def _composite_eval_jit(weights, mean_returns, cov_matrix, gamma):
    ret = np.dot(weights, mean_returns)
    var = np.dot(weights, np.dot(cov_matrix, weights))
    penalty = gamma * np.sum(np.abs(weights)**1.5)
    return -ret + 2.0 * var + penalty

class CompositeObjective(BaseObjective):
    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma
        self.mean_returns_ = None
        self.cov_matrix_ = None
        
    def illuminate(self, returns: np.ndarray, cov_matrix: np.ndarray):
        self.ret_historical = returns
        self.mean_returns_ = np.mean(returns, axis=0)
        self.cov_matrix_ = cov_matrix
        return self
        
    def measure_brightness(self, weights: np.ndarray) -> float:
        # Base JIT penalty evaluation
        base_bright = _composite_eval_jit(weights, self.mean_returns_, self.cov_matrix_, self.gamma)
        cvar = calculate_cvar(self.ret_historical, weights)
        mdd = -calculate_max_drawdown(self.ret_historical, weights)
        return base_bright + cvar * 2.5 + mdd * 1.5
