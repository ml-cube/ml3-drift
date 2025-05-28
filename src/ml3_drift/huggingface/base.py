from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ml3_drift.callbacks.models import DriftInfo


class BaseDriftDetector(ABC):
    """
    Base class for drift detectors.
    """

    def __init__(self, callbacks: list[Callable[[DriftInfo], None]] | None = None):
        """
        Initialize the drift detector.
        This method can be overridden in child classes to set up any necessary parameters.
        """

        self.is_fitted = False
        if callbacks is not None:
            self.callbacks: list[Callable[[DriftInfo], None]] = callbacks
        else:
            self.callbacks = []

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the drift detector on the reference data.

        Args:
            X (np.ndarray): Reference data to fit the drift detector on.
        """
        self._fit(X)
        self.is_fitted = True

    def detect(self, X: np.ndarray) -> None:
        """
        Detect drift in the provided data.

        Args:
            X (np.ndarray): Data to check for drift.

        Returns:
            np.ndarray: The input data, unchanged.
        """
        if not self.is_fitted:
            raise RuntimeError("Drift detector must be fitted before detecting drift.")

        if drift_info_list := self._detect(X):
            for drift_info in drift_info_list:
                for callback in self.callbacks:
                    callback(drift_info)

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """
        Core fit method that should be implemented in child classes
        """

    @abstractmethod
    def _detect(self, X: np.ndarray) -> None:
        """
        Core method for detecting drift that should be implemented in child classes.
        """
