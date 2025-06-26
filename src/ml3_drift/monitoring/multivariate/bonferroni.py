from typing import Callable, TypeVar

from numpy import ndarray
from ml3_drift.enums.monitoring import DataDimension, DataType, MonitoringType
from ml3_drift.models.monitoring import (
    DriftInfo,
    MonitoringAlgorithmSpecs,
    MonitoringOutput,
)
from ml3_drift.monitoring.base import MonitoringAlgorithm
from ml3_drift.monitoring.univariate.base import UnivariateMonitoringAlgorithm

T = TypeVar("T", bound=UnivariateMonitoringAlgorithm)


class BonferroniCorrectionAlgorithm(MonitoringAlgorithm):
    """Extension of p-value based univariate algorithms with Bonferroni correction
    to handle multivariate data"""

    _specs = MonitoringAlgorithmSpecs(
        data_dimension=DataDimension.MULTIVARIATE,
        data_type=DataType.MIX,
        monitoring_type=MonitoringType.OFFLINE,
    )

    def __init__(
        self,
        comparison_size: int,
        algorithm_builder: Callable[[float], T],
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size, callbacks)
        self.p_value = p_value
        self.algorithm_builder = algorithm_builder

        # post fit attributes
        self.dims = 0
        self.algorithms: list[T] = []

    def _reset_internal_parameters(self):
        self.algorithms = []

    def _fit(self, X: ndarray):
        self.dims = X.shape[1]
        for i in range(self.dims):
            algorithm = self.algorithm_builder(self.p_value / self.dims)
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
