from copy import deepcopy
from typing import Callable, TypeVar

from numpy import ndarray
from ml3_drift.enums.monitoring import DataDimension, DataType, MonitoringType
from ml3_drift.models.monitoring import (
    DriftInfo,
    MonitoringAlgorithmSpecs,
    MonitoringOutput,
)
from ml3_drift.monitoring.base import MonitoringAlgorithm
from ml3_drift.monitoring.univariate.base import UnivariateMonitoringAlgorithm

T = TypeVar("T", bound=UnivariateMonitoringAlgorithm)


class BonferroniCorrectionAlgorithm(MonitoringAlgorithm):
    """
    Extension of p-value based univariate algorithms with Bonferroni correction
    to handle multivariate data

    Parameters
    ----------
    algorithm: T (UnivariateMonitoringAlgorithm)
        The univariate monitoring algorithm to be used for each dimension.
    p_value: float, default=0.005
        The p-value threshold for detecting drift, will be adjusted using Bonferroni correction.
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
            data_dimension=DataDimension.MULTIVARIATE,
            data_type=DataType.MIX,
            monitoring_type=MonitoringType.OFFLINE,
        )

    def __init__(
        self,
        algorithm: T,
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo | None], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size=None, callbacks=callbacks)
        self.p_value = p_value
        self.base_algorithm = algorithm

        # post fit attributes
        self.dims = 0
        self.algorithms: list[T] = []

    def _reset_internal_parameters(self):
        self.algorithms = []
        self.dims = 0

    def _fit(self, X: ndarray):
        self.dims = X.shape[1]
        for i in range(self.dims):
            # Deepcopy the base algorithm to ensure each instance is independent
            # and can be fitted to its own data slice.
            algorithm = deepcopy(self.base_algorithm)

            # Check that the algorithm has a setter for p_value
            # and set it accordingly
            if (
                not hasattr(algorithm, "p_value")
                or not self.base_algorithm.__class__.p_value.fset is not None  # type: ignore
            ):
                raise ValueError(
                    f"Algorithm {algorithm.__class__.__name__} does not have a 'p_value' attribute."
                )

            algorithm.p_value = self.p_value / self.dims  # Bonferroni correction

            algorithm.fit(X[:, i : i + 1])
            self.algorithms.append(algorithm)

    def _detect(self) -> MonitoringOutput:
        drift_detected = False
        for i, algorithm in enumerate(self.algorithms):
            # Update the comparison data for the algorithm and call inner _detect method
            algorithm.comparison_data = self.comparison_data[:, i : i + 1]
            output = algorithm._detect()
            if output.drift_detected:
                drift_detected = True

        # Currently we output a drift_detected = True if any of the algorithms detected drift.
        # However the drift information is not aggregated. This neds to be improved
        return MonitoringOutput(drift_detected=drift_detected, drift_info=None)
