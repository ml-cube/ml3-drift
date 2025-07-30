from abc import ABC
from typing import Callable

import numpy as np
from ml3_drift.exceptions.monitoring import NotFittedError
from ml3_drift.models.monitoring import DriftInfo, MonitoringOutput
from ml3_drift.monitoring.base.base import MonitoringAlgorithm


class BatchMonitoringAlgorithm(MonitoringAlgorithm, ABC):
    """
    Abstract base class for offline monitoring algorithms.
    It implements the detect method for offline drift detection.
    """

    def __init__(
        self,
        comparison_size: int | None = None,
        callbacks: list[Callable[[DriftInfo | None], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size, callbacks)

    def _offline_detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """
        In offline detection we compare reference data, coming from fit(X), with the
        provided batch of data.

        Returns a single MonitoringOutput object containing the drift detection result.
        """

        # Comparison data are exactly the sample provided here.
        self.comparison_data = X
        return [self._detect()]

    def detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """
        Analyze the provided data samples against the reference dataset
        (which needs to be set by calling fit(X) first).

        If present, callbacks are called for each drifted sample.
        """
        if not self.is_fitted:
            raise NotFittedError("Algorithm must be fitted first.")

        if self.data_shape == 1 and len(X.shape) == 1:
            X = X.reshape(-1, 1)

        elif X.shape[1] != self.data_shape:
            raise ValueError(
                f"Data must have the same shape as reference. Expected {self.data_shape}, got {X.shape[1]}"
            )

        self._validate(X)

        detection_output = self._offline_detect(X)

        if self.has_callbacks:
            for sample_output in detection_output:
                if sample_output.drift_detected:
                    for callback in self.callbacks:
                        callback(sample_output.drift_info)

        return detection_output
