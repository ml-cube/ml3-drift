

from abc import ABC
from typing import Callable

import numpy as np
from ml3_drift.exceptions.monitoring import NotFittedError
from ml3_drift.models.monitoring import DriftInfo, MonitoringOutput
from ml3_drift.monitoring.base import MonitoringAlgorithm


class OnlineMonitorningAlgorithm(MonitoringAlgorithm, ABC):
    """
    Base class for online monitoring algorithms.
    """

    def __init__(
        self,
        comparison_size: int | None = None,
        callbacks: list[Callable[[DriftInfo | None], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size, callbacks)

    def _online_detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """In online detection, a sliding window is used over the samples to detect drift at each step.

        Test statistic is computed only when there is enough data for comparison.
        Specifically the number of comparison data is defined by the attribute `comparison_size`.

        Therefore, the first `comparison_size` - 1 samples are not monitored and they produce a "no drift" output.
        After that, any new sample is added to `comparison_data` by removing the oldest one,
        the detect method is called and the output is returned.
        """
        # Detection loop

        detection_output = []

        # initialize comparison data with all the available data
        samples_to_fill_comparison_data = min(
            X.shape[0], self.comparison_size - self.comparison_data.shape[0]
        )

        if samples_to_fill_comparison_data > 0:
            initial_comparison_data_size = self.comparison_data.shape[0]
            data_to_add = X[:samples_to_fill_comparison_data]
            if initial_comparison_data_size == 0:
                self.comparison_data = data_to_add
            else:
                self.comparison_data = np.vstack([self.comparison_data, data_to_add])

            # Explained in the docstring
            detection_output = [
                MonitoringOutput(drift_detected=False, drift_info=None)
                for _ in range(data_to_add.shape[0] - 1)
            ]

            if self.comparison_data.shape[0] == self.comparison_size:
                detection_output.append(self._detect())
            else:
                detection_output.append(
                    MonitoringOutput(drift_detected=False, drift_info=None)
                )

        for i in range(
            samples_to_fill_comparison_data,
            X.shape[0] - samples_to_fill_comparison_data,
        ):
            self.comparison_data = np.vstack([self.comparison_data[1:], X[i : i + 1]])
            detection_output.append(self._detect())

        return detection_output
    
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

        if self.comparison_size is None:
            raise ValueError(
                "Comparison size must be defined for online monitoring algorithms."
            )
            
        detection_output = self._online_detect(X)
    
        if self.has_callbacks:
            for sample_output in detection_output:
                if sample_output.drift_detected:
                    for callback in self.callbacks:
                        callback(sample_output.drift_info)

        return detection_output