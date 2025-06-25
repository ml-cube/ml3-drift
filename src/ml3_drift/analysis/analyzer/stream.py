import numpy as np

from typing import Callable
from ml3_drift.analysis.analyzer.base import DataDriftAnalyzer
from ml3_drift.analysis.report import Report
from ml3_drift.monitoring.base import MonitoringAlgorithm


class StreamDataDriftAnalyzer(DataDriftAnalyzer):
    def __init__(
        self,
        continuous_ma_builder: Callable[[int], MonitoringAlgorithm],
        categorical_ma_builder: Callable[[int], MonitoringAlgorithm],
        reference_size: int = 100,
        comparison_window_size: int = 100,
    ):
        super().__init__(
            continuous_ma_builder,
            categorical_ma_builder,
        )

        self.reference_size = reference_size
        self.comparison_window_size = comparison_window_size

    def _scan_data(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        continuous_columns_ids: list[int],
        categorical_columns_ids: list[int],
        y_categorical: bool,
    ) -> Report:
        """Scan the data to identify different data partitions according to monitoring algorithm.

        First, we build categorical and continuous data combining input and target.
        Then, we initialize monitoring algorithms with the first data.
        After that, we iterate over the remaining data samples updating the monitoring algorithm.
        If a drift is detected and there are enough data to reset the algorithm, we reset them and
        continue the analysis, otherwise we terminate it.
        """
        concepts = []

        # Continuous and categorical data creation
        continuous_data = X[:, continuous_columns_ids]
        categorical_data = X[:, categorical_columns_ids]
        if y is not None:
            if y_categorical:
                categorical_data = np.hstack([categorical_data, y])
            else:
                continuous_data = np.hstack([continuous_data, y])

        if X.shape[0] < self.reference_size:
            raise ValueError(
                f"Data must have at least {self.reference_size}. Got {X.shape[0]}"
            )

        # Algorithm initialization
        continuous_ma = self.continuous_ma_builder(self.comparison_window_size).fit(
            continuous_data[: self.reference_size, :]
        )

        categorical_ma = self.categorical_ma_builder(self.comparison_window_size).fit(
            categorical_data[: self.reference_size, :]
        )

        # actual data scan
        concept_start = 0
        row_id = self.reference_size
        available_data = (X.shape[0] - 1) > row_id

        while available_data:
            continuous_output = continuous_ma.detect(
                continuous_data[row_id : row_id + 1, :]
            )[0]
            categorical_output = categorical_ma.detect(
                categorical_data[row_id : row_id + 1, :]
            )[0]

            remaining_data = X.shape[0] - 1 - row_id

            if continuous_output.drift_detected | categorical_output.drift_detected:
                concepts.append((concept_start, row_id))
                concept_start = row_id + 1
                # reset monitoring algorithm with past comparison_window_size data and newest one

                if remaining_data >= self.reference_size:
                    continuous_ma.fit(
                        continuous_data[
                            row_id : row_id + self.reference_size,
                            :,
                        ]
                    )
                    categorical_ma.fit(
                        categorical_data[
                            row_id : row_id + self.reference_size,
                            :,
                        ]
                    )
                    row_id = row_id + self.reference_size
                else:
                    # concepts.append((row_id + 1, row_id + 1 + remaining_data))
                    available_data = False
            else:
                row_id += 1
                available_data = remaining_data > 0

        # If no drift is detected we have only one concept
        if len(concepts) == 0:
            concepts = [(concept_start, X.shape[0])]
        else:
            concepts.append((concepts[-1][1], X.shape[0]))

        return Report(concepts)
