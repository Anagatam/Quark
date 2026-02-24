import numpy as np
import pandas as pd

class DataProcessor:
    @staticmethod
    def _mad_winsorize(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        mad = np.where(mad == 0, 1e-6, mad)
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.where(modified_z_scores > threshold, median + (threshold * mad / 0.6745),
               np.where(modified_z_scores < -threshold, median - (threshold * mad / 0.6745), data))

    def clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change().fillna(0).values
        winsorized = self._mad_winsorize(returns)
        return pd.DataFrame(winsorized, index=prices.index, columns=prices.columns)
