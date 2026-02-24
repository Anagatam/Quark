from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseObjective(ABC):
    @abstractmethod
    def illuminate(self, returns: np.ndarray, cov_matrix: Optional[np.ndarray] = None) -> 'BaseObjective':
        pass

    @abstractmethod
    def measure_brightness(self, weights: np.ndarray) -> float:
        pass

class BaseConstraint(ABC):
    @abstractmethod
    def map_to_feasible_space(self, weights: np.ndarray) -> np.ndarray:
        pass
