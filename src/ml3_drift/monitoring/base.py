from abc import ABC, abstractmethod
from typing import Callable, Self
import numpy as np

from ml3_drift.enums.monitoring import MonitoringType
from ml3_drift.exceptions.monitoring import NotFittedError
from ml3_drift.models.monitoring import (
    DriftInfo,
    MonitoringAlgorithmSpecs,
    MonitoringOutput,
)


class MonitoringAlgorithm(ABC):
    """A Monitoring Algorithm is a class that analyze sequentially
    data samples comparing them with reference data.
    According to the computed statistics, the compared data can be
    marked as different and a drift is signaled.

    This abstract class provides standard interface following the
    sklearn paradigm with the method fit(X) to initialize the algorithm.
    The method detect(X) is equivalent to the known predict(X) or transform(X).

    The class can have a list of callbacks called as soon as a drift is detected,
    a callback is a function that receives as input the drift info
    """

    _specs: MonitoringAlgorithmSpecs

    @classmethod
    def specs(cls) -> MonitoringAlgorithmSpecs:
        return cls._specs

    def __init__(
        self,
        comparison_size: int,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ) -> None:
        self.comparison_size = comparison_size

        self.callbacks = callbacks if callbacks is not None else []
        self.has_callbacks = len(self.callbacks) > 0

        self.is_fitted = False
        # post fit attributes
        self.reference_size = 0
        self.data_shape = 0
        # post detect attributes
        self.comparison_data: np.ndarray = np.array([])

    def _is_valid(self, X: np.ndarray) -> tuple[bool, str]:
        """Additional validation performed by subclasses.

        If provided data is not valid then False is returned.
        """
        return True, ""

    def _validate(self, X: np.ndarray):
        """Input data validation calling subclasses _is_valid method.

        If provided data is not valid then ValueError is raised."""
        is_valid, message = self._is_valid(X)
        if not is_valid:
            raise ValueError(message)

    @abstractmethod
    def _reset_internal_parameters(self):
        pass

    @abstractmethod
    def _fit(self, X: np.ndarray):
        pass

    def fit(self, X: np.ndarray) -> Self:
        """Initialize the monitoring algorithm using as reference the provided data.

        At the end of the fit procedure, it stores the number of reference samples and
        the dimension of input data."""

        self.reference_size = 0
        self.data_shape = 0
        self.comparison_data: np.ndarray = np.array([])

        self._reset_internal_parameters()

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self._validate(X)

        self._fit(X)

        self.is_fitted = True

        self.reference_size, self.data_shape = X.shape

        return self

    @abstractmethod
    def _detect(self) -> MonitoringOutput:
        pass

    def _online_detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """In online detection, a sliding window is used over the samples to detect drift at each step.

        Test statistic is computed only when there is enough data for comparison.
        Specifically the number of comparison data is defined by the attribute `comparison_size`.

        Therefore, the first `comparison_size` - 1 are not monitored and they produce a "no drift" output.
        After that, any new sample is added to `comparison_data` by removing the oldest one, the KS statistic
        is computed and according to the p-value, the drift is detected."""
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

    def _offline_detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """In offline detection we compare reference data, coming from fit(X), with the
        provided batch of data.

        Therefore, we substitute completely the comparison_data with the provided X
        """
        self.comparison_data = X
        return [self._detect()]

    def detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """Analyzes the provided data computing statistics and defining if they belong
        to the reference distribution or to another determining a drift.

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

        if self.specs().monitoring_type == MonitoringType.ONLINE:
            detection_output = self._online_detect(X)
        else:
            detection_output = self._offline_detect(X)

        if self.has_callbacks:
            for sample_output in detection_output:
                if (
                    sample_output.drift_detected
                    and sample_output.drift_info is not None
                ):
                    for callback in self.callbacks:
                        callback(sample_output.drift_info)

        return detection_output
