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
    callbacks: list[Callable[[DriftInfo | None], None]] | None, optional
        A list of callback functions that are called when a drift is detected.
        Each callback receives a DriftInfo object containing information about the detected drift.
        If not provided, no callbacks are used. The current type hint also includes
        the case where drift_info is None (which happens for only some algorithms). This
        will change in the future as it's not very useful to have a callback that
        receives None as input.
    """

    @classmethod
    def specs(cls) -> MonitoringAlgorithmSpecs:
        return MonitoringAlgorithmSpecs(
            data_dimension=DataDimension.UNIVARIATE,
            data_type=DataType.CONTINUOUS,
            monitoring_type=MonitoringType.OFFLINE,
        )

    def __init__(
        self,
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo | None], None]] | None = None,
    ) -> None:
        super().__init__(
            comparison_size=None,
            callbacks=callbacks,
        )
        self._p_value = p_value

        # post fit attributes
        self.X_ref_: np.ndarray = np.array([])

    @property
    def p_value(self) -> float:
        """Get the p-value threshold for detecting drift."""
        return self._p_value

    @p_value.setter
    def p_value(self, value: float):
        """Set the p-value threshold for detecting drift"""
        if value <= 0 or value >= 1:
            raise ValueError("p_value must be in the range (0, 1).")
        self._p_value = value

    def _reset_internal_parameters(self):
        self.X_ref_ = np.array([])

    def _fit(self, X: np.ndarray):
        """Saves input data without any additional computation"""
        self.X_ref_ = X

    def _detect(self) -> MonitoringOutput:
        """Compute the statistic and create the monitoring output object"""
        _, test_p_val = stats.ks_2samp(self.X_ref_, self.comparison_data)

        test_p_val_float = test_p_val.item()

        drift_detected = test_p_val_float < self.p_value

        return MonitoringOutput(
            drift_detected=drift_detected,
            drift_info=DriftInfo(
                test_statistic=test_p_val_float, statistic_threshold=self.p_value
            ),
        )
