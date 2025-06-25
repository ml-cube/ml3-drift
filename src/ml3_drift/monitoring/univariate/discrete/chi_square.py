import numpy as np
from typing import Callable

from scipy import stats
from ml3_drift.models.monitoring import DriftInfo, MonitoringOutput
from ml3_drift.monitoring.base import MonitoringAlgorithm


class ChiSquareAlgorithm(MonitoringAlgorithm):
    """Monitoring algorithm based on the Chi Square statistic test.

    Parameters
    ----------
    p_value: float
        p-value threshold for detecting drift. Default is 0.005.
    """

    def __init__(
        self,
        comparison_size: int,
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ) -> None:
        super().__init__(comparison_size, callbacks)
        self.p_value = p_value

        # post fit attributes
        self.reference_counts: dict[str | int, int] = {}
        self.categories: list[str | int] = []

    def _is_valid(self, X: np.ndarray) -> tuple[bool, str]:
        if X.shape[1] == 1:
            return True, ""
        else:
            return False, f"X must be 1-dimensional vector. Got {X.shape}"

    def _reset_internal_parameters(self):
        self.reference_counts = {}
        self.categories = []

    def _fit(self, X: np.ndarray):
        """Saves input data without any additional computation"""
        self.categories = list(np.unique(X[:, 0]))
        self.reference_counts = self._compute_counts(X)

    def _compute_counts(self, X: np.ndarray) -> dict[str | int, int]:
        """Compute the frequency for each category in the input data"""
        counts = {}
        for category in self.categories:
            counts[category] = int(np.sum(X[:, 0] == category))
        return counts

    def _detect(self) -> MonitoringOutput:
        """Compute the statistic and create the monitoring output object"""
        comparison_counts = self._compute_counts(self.comparison_data)

        _, p_value, _, _ = stats.chi2_contingency(
            np.column_stack(
                (
                    [self.reference_counts[category] for category in self.categories],
                    [comparison_counts[category] for category in self.categories],
                )
            )
        )
        p_value = float(p_value)  # type: ignore

        drift_detected = p_value < self.p_value

        return MonitoringOutput(
            drift_detected=drift_detected,
            drift_info=DriftInfo(
                test_statistic=p_value, statistic_threshold=self.p_value
            ),
        )
