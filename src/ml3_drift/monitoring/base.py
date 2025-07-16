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
    """
    A Monitoring Algorithm is statistical method that analyzes sequential data
    samples by comparing them against a reference dataset.
    Based on the computed statistics, it can identify significant differences and signal a drift.

    This abstract class follows the standard scikit-learn interface, using the fit(X) method
    to initialize the algorithm with reference data, and the detect(X) method to analyze new data samples
    (serving a similar purpose to predict(X) in scikit-learn classifiers).

    According to the algorithm specifications, the monitoring can be performed in two modes:
    - Online: where the algorithm analyzes data samples sequentially, using a sliding window approach.
    - Offline: where the algorithm analyzes a batch of data samples at once, comparing them against
        the reference dataset.

    The class also support a list of callbacks that are automatically called when a drift is detected. Each callback
    receives a DriftInfo object containing information about the detected drift.

    Parameters
    ----------
    comparison_size: int | None, optional
        Only relevant in online monitoring algorithms.
        It defines the size of the sliding window used for comparison.
    callbacks: list[Callable[[DriftInfo | None], None]], optional
        A list of callback functions that are called when a drift is detected.
        Each callback receives a DriftInfo object containing information about the detected drift.
        If not provided, no callbacks are used.
    """

    @classmethod
    @abstractmethod
    def specs(cls) -> MonitoringAlgorithmSpecs:
        """
        Abstract property that returns the specifications of the monitoring algorithm.
        """

    def __init__(
        self,
        comparison_size: int | None = None,
        callbacks: list[Callable[[DriftInfo | None], None]] | None = None,
    ) -> None:
        self.comparison_size = comparison_size

        self.callbacks = callbacks if callbacks is not None else []
        self.has_callbacks = len(self.callbacks) > 0

        self.is_fitted = False
        self.reset_internal_parameters()

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

    def reset_internal_parameters(self):
        """
        Reset the internal parameters of the monitoring algorithm.
        """

        self.reference_size = 0
        self.data_shape = 0
        self.comparison_data = np.array([])

        self._reset_internal_parameters()

    @abstractmethod
    def _reset_internal_parameters(self):
        """
        Abstract method to reset internal parameters of the monitoring algorithm.
        This method should be implemented by subclasses to reset any internal state
        or parameters that are specific to the algorithm.
        """

    @abstractmethod
    def _fit(self, X: np.ndarray):
        pass

    def fit(self, X: np.ndarray) -> Self:
        """Initialize the monitoring algorithm using as reference the provided data.

        At the end of the fit procedure, it stores the number of reference samples and
        the dimension of input data."""

        self.reset_internal_parameters()

        n_dim = X.ndim

        if n_dim > 2:
            raise ValueError(f"Data must be 1D or 2D array. Got {n_dim} dimensions.")

        if n_dim == 1:
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

        if self.specs().monitoring_type == MonitoringType.ONLINE:
            if self.comparison_size is None:
                raise ValueError(
                    "Comparison size must be defined for online monitoring algorithms."
                )
            detection_output = self._online_detect(X)
        else:
            detection_output = self._offline_detect(X)

        if self.has_callbacks:
            for sample_output in detection_output:
                if sample_output.drift_detected:
                    for callback in self.callbacks:
                        callback(sample_output.drift_info)

        return detection_output
