from typing import Callable

from numpy import ndarray
from ml3_drift.models.monitoring import DriftInfo, MonitoringOutput
from ml3_drift.monitoring.base import MonitoringAlgorithm
from ml3_drift.monitoring.univariate.discrete.chi_square import ChiSquareAlgorithm


class BonferroniChiSquareAlgorithm(MonitoringAlgorithm):
    """Extension of Chi Square algorithm with Bonferroni correction
    to handle multivariate categorical data"""

    def __init__(
        self,
        comparison_size: int,
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size, callbacks)
        self.p_value = p_value

        # post fit attributes
        self.dims = 0
        self.algorithms: list[ChiSquareAlgorithm] = []

    def _reset_internal_parameters(self):
        self.algorithms = []

    def _fit(self, X: ndarray):
        self.dims = X.shape[1]
        for i in range(self.dims):
            algorithm = ChiSquareAlgorithm(
                self.comparison_size, self.p_value / self.dims
            )
            algorithm.fit(X[:, i : i + 1])
            self.algorithms.append(algorithm)

    def _detect(self) -> MonitoringOutput:
        drift_detected = False
        for i, algorithm in enumerate(self.algorithms):
            algorithm.comparison_data = self.comparison_data[:, i : i + 1]
            output = algorithm._detect()
            if output.drift_detected:
                drift_detected = True
        return MonitoringOutput(drift_detected=drift_detected, drift_info=None)
