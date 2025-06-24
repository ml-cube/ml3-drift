import numpy as np
from typing import Callable

from scipy import stats
from ml3_drift.models.monitoring import DriftInfo, MonitoringOutput
from ml3_drift.monitoring.base import MonitoringAlgorithm


class KSAlgorithm(MonitoringAlgorithm):
    """Monitoring algorithm based on the Kolmogorov-Smirnov statistic test.

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
        self.X_ref_: np.ndarray = np.array([])
        # post detect attributes
        self.comparison_data: np.ndarray = np.array([])

    def _is_valid(self, X: np.ndarray) -> tuple[bool, str]:
        if X.shape[1] == 1:
            return True, ""
        else:
            return False, f"X must be 1-dimensional vector. Got {X.shape}"

    def _fit(self, X: np.ndarray):
        """Saves input data without any additional computation"""
        self.X_ref_ = X

    def _compute_statistic(self) -> MonitoringOutput:
        """Compute the statistic and create the monitoring output object"""
        _, p_value = stats.ks_2samp(self.X_ref_, self.comparison_data)
        p_value = float(p_value)  # type: ignore

        drift_detected = p_value < self.p_value

        return MonitoringOutput(
            drift_detected=drift_detected,
            drift_info=DriftInfo(
                test_statistic=p_value, statistic_threshold=self.p_value
            ),
        )

    def _detect(self, X: np.ndarray) -> list[MonitoringOutput]:
        """Compute KS statistic between reference and every sample.

        Test statistic is computed only when there is enough data for comparison.
        Specifically the number of comparison data is defined by the attribute `comparison_size`.

        Therefore, the first `comparison_size` - 1 are not monitored and they produce a "no drift" output.
        After that, any new sample is added to `comparison_data` by removing the oldest one, the KS statistic
        is computed and according to the p-value, the drift is detected.
        """
        output = []

        # initialize comparison data with all the available data
        samples_to_fill_comparison_data = (
            self.comparison_size - self.comparison_data.shape[0]
        )

        if samples_to_fill_comparison_data > 0:
            self.comparison_data = X[:samples_to_fill_comparison_data]
            output = [
                MonitoringOutput(drift_detected=False, drift_info=None)
                for _ in range(samples_to_fill_comparison_data - 1)
            ]

            output.append(self._compute_statistic())

        for i in range(
            samples_to_fill_comparison_data,
            X.shape[0] - samples_to_fill_comparison_data,
        ):
            self.comparison_data = np.vstack([self.comparison_data[1:], X[i : i + 1]])
            output.append(self._compute_statistic())

        return output
