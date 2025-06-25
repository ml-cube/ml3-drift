import numpy as np
from typing import Callable

from scipy import stats
from ml3_drift.enums.monitoring import DataDimension, DataType, MonitoringType
from ml3_drift.models.monitoring import (
    DriftInfo,
    MonitoringAlgorithmSpecs,
    MonitoringOutput,
)
from ml3_drift.monitoring.univariate.base import UnivariateMonitoringAlgorithm


class KSAlgorithm(UnivariateMonitoringAlgorithm):
    """Monitoring algorithm based on the Kolmogorov-Smirnov statistic test.

    Parameters
    ----------
    p_value: float
        p-value threshold for detecting drift. Default is 0.005.
    """

    _specs = MonitoringAlgorithmSpecs(
        data_dimension=DataDimension.UNIVARIATE,
        data_type=DataType.CONTINUOUS,
        monitoring_type=MonitoringType.OFFLINE,
    )

    def __init__(
        self,
        comparison_size: int,
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size, callbacks)
        self.p_value = p_value

        # post fit attributes
        self.X_ref_: np.ndarray = np.array([])

    def _reset_internal_parameters(self):
        self.X_ref_ = np.array([])

    def _fit(self, X: np.ndarray):
        """Saves input data without any additional computation"""
        self.X_ref_ = X

    def _detect(self) -> MonitoringOutput:
        """Compute the statistic and create the monitoring output object"""
        _, p_value = stats.ks_2samp(self.X_ref_, self.comparison_data)
        p_value = float(p_value)  # type: ignore

        drift_detected = p_value < self.p_value

        return MonitoringOutput(
            drift_detected=drift_detected,
            drift_info=DriftInfo(
                test_statistic=p_value, statistic_threshold=self.p_value
            ),
        )
