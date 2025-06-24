from abc import ABC, abstractmethod
from typing import Callable, Self
import numpy as np

from ml3_drift.exceptions.monitoring import NotFittedError
from ml3_drift.models.monitoring import DriftInfo, MonitoringOutput


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
    def _fit(self, X: np.ndarray):
        pass

    def fit(self, X: np.ndarray) -> Self:
        """Initialize the monitoring algorithm using as reference the provided data.

        At the end of the fit procedure, it stores the number of reference samples and
        the dimension of input data."""
        column_data = len(X.shape) == 1
        if column_data:
            X = X.reshape(-1, 1)

        self._validate(X)

        self._fit(X)

        self.is_fitted = True

        if column_data:
            self.reference_size = X.shape
            self.data_shape = 1
        else:
            self.reference_size, self.data_shape = X.shape

        return self

    @abstractmethod
    def _detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        pass

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

        detection_output = self._detect(X)

        if self.has_callbacks:
            for sample_output in detection_output:
                if (
                    sample_output.drift_detected
                    and sample_output.drift_info is not None
                ):
                    for callback in self.callbacks:
                        callback(sample_output.drift_info)

        return detection_output
