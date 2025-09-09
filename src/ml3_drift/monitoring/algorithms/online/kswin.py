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
    from river.drift.kswin import KSWIN as RiverKSWIN
except ModuleNotFoundError:
    RIVER = False


class KSWIN(OnlineMonitorningAlgorithm, UnivariateMonitoringAlgorithm):
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
        p_value: float = 0.00,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        if not RIVER:
            raise ModuleNotFoundError(
                "River library is required for KSWIN algorithm. Please install it using pip install/ uv add ml3-drift[river]"
            )
        self.p_value = p_value
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._args = args
        self._kwargs = kwargs
        # since we add only one sample per step and river handles building the window internally we set comparison_size to 1
        super().__init__(comparison_size=1, callbacks=callbacks)

    def _reset_internal_parameters(self):
        self.drift_agent = RiverKSWIN(
            alpha=self.p_value,
            window_size=self.window_size,
            stat_size=self.stat_size,
            seed=self.seed,
            *self._args,
            **self._kwargs,
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
