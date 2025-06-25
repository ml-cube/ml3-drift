from abc import ABC
import numpy as np

from ml3_drift.monitoring.base import MonitoringAlgorithm


class UnivariateMonitoringAlgorithm(MonitoringAlgorithm, ABC):
    """Base class for univariate monitoring algorithm."""

    def _is_valid(self, X: np.ndarray) -> tuple[bool, str]:
        if X.shape[1] == 1:
            return True, ""
        else:
            return False, f"X must be 1-dimensional vector. Got {X.shape}"
