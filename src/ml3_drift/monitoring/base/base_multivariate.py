from abc import ABC
import numpy as np

from ml3_drift.monitoring.base.base import MonitoringAlgorithm


class MultivariateMonitoringAlgorithm(MonitoringAlgorithm, ABC):
    """
    Base class for multivariate monitoring algorithm.

    It is currently used only to validate the input data dimension.
    """

    def _is_valid(self, X: np.ndarray) -> tuple[bool, str]:
        if X.shape[1] >= 1:
            return True, ""
        else:
            return False, f"X must be multi-dimensional vector. Got {X.shape}"
