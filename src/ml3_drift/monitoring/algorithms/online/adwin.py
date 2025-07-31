from typing import Callable

import numpy as np
from ml3_drift.enums.monitoring import DataDimension, DataType, MonitoringType
from ml3_drift.models.monitoring import (
    DriftInfo,
    MonitoringAlgorithmSpecs,
    MonitoringOutput,
)
from ml3_drift.monitoring.base.base_univariate import UnivariateMonitoringAlgorithm
from ml3_drift.monitoring.base.online_monitorning_algorithm import (
    OnlineMonitorningAlgorithm,
)

RIVER = True
try:
    from river.drift.adwin import ADWIN as RiverADWIN
except ModuleNotFoundError:
    RIVER = False


class ADWIN(OnlineMonitorningAlgorithm, UnivariateMonitoringAlgorithm):
    @classmethod
    def specs(cls) -> MonitoringAlgorithmSpecs:
        return MonitoringAlgorithmSpecs(
            data_dimension=DataDimension.MULTIVARIATE,
            data_type=DataType.MIX,
            monitoring_type=MonitoringType.ONLINE,
        )

    def __init__(
        self,
        callbacks: list[Callable[[DriftInfo | None], None]] | None = None,
        p_value: float = 0.002,
        clock: float = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        *args,
        **kwargs,
    ) -> None:
        if not RIVER:
            raise ModuleNotFoundError(
                "River library is required for ADWIN algorithm. Please install it using pip install/ uv add ml3-drift[river]"
            )
        self.p_value = p_value
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        super().__init__(comparison_size=1, callbacks=callbacks)

    def _reset_internal_parameters(self):
        self.drift_agent = RiverADWIN(
            delta=self.p_value,
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=self.min_window_length,
            grace_period=self.grace_period,
        )

    def _fit(self, X: np.ndarray):
        """Fit the KSWIN algorithm to the data."""
        self._validate(X)
        self.reset_internal_parameters()
        self.is_fitted = True

    def _detect(self):
        self.drift_agent.update(self.comparison_data)
        drift_detected = self.drift_agent.drift_detected
        return MonitoringOutput(drift_detected=drift_detected, drift_info=None)
