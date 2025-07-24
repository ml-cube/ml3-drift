from typing import Callable
from ml3_drift.enums.monitoring import DataDimension, DataType, MonitoringType
from ml3_drift.models.monitoring import DriftInfo, MonitoringAlgorithmSpecs, MonitoringOutput
from ml3_drift.monitoring.base.base_multivariate import MultivariateMonitoringAlgorithm
from ml3_drift.monitoring.base.online_monitorning_algorithm import OnlineMonitorningAlgorithm
from river.drift.adwin import ADWIN as RiverADWIN


class ADWIN(OnlineMonitorningAlgorithm, MultivariateMonitoringAlgorithm):
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
        delta: float = 0.002,
        clock: float = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ) -> None:
        super().__init__(comparison_size=1, callbacks=callbacks)
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        
        
    def _reset_internal_parameters(self):
        self.drift_agent = RiverADWIN(
            delta=self.delta,
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=self.min_window_length,
            grace_period=self.grace_period,
        )
    def _detect(self):
        self.drift_agent.update(self.comparison_data)
        drift_detected = self.drift_agent.drift_detected
        return MonitoringOutput(drift_detected=drift_detected, drift_info=None)
        


