from collections import defaultdict
import numpy as np

from typing import Callable
from ml3_drift.analysis.analyzer.base import DataDriftAnalyzer
from ml3_drift.analysis.report import Report
from ml3_drift.monitoring.base import MonitoringAlgorithm

from ml3_drift.models.monitoring import (
    MonitoringOutput,
)


class BatchDataDriftAnalyzer(DataDriftAnalyzer):
    """Batch data drift analyzer splits data into mini batches of size `batch_size`
    and through drift detection merges them into macro batch representing data that
    belong to the same distribution.

    Data can belong to the same distribution even if they are not contiguous.

    Parameters
    ----------
    continuous_ma_builder: closure function that accepts int parameter as `comparison_window_size`
        and returns an instance of a MonitoringAlgorithm
    categorical_ma_builder: closure function that accepts int parameter as `comparison_window_size`
        and returns an instance of a MonitoringAlgorithm
    batch_size: initial batch dimensions and also used as comparison_window_size
    """

    def __init__(
        self,
        continuous_ma_builder: Callable[[int], MonitoringAlgorithm],
        categorical_ma_builder: Callable[[int], MonitoringAlgorithm],
        batch_size: int = 100,
    ):
        super().__init__(
            continuous_ma_builder,
            categorical_ma_builder,
        )

        self.batch_size = batch_size

    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        continuous_columns_ids: list[int],
        categorical_columns_ids: list[int],
        y_categorical: bool,
        batch_indexes: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        continuous_data = X[batch_indexes[0] : batch_indexes[1], continuous_columns_ids]
        categorical_data = X[
            batch_indexes[0] : batch_indexes[1], categorical_columns_ids
        ]
        if y is not None:
            if y_categorical:
                categorical_data = np.hstack(
                    [categorical_data, y[batch_indexes[0] : batch_indexes[1]]]
                )
            else:
                continuous_data = np.hstack(
                    [continuous_data, y[batch_indexes[0] : batch_indexes[1]]]
                )

        return continuous_data, categorical_data

    def _single_scan_data(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        continuous_columns_ids: list[int],
        categorical_columns_ids: list[int],
        y_categorical: bool,
        first_batch_indexes: tuple[int, int],
        second_batch_indexes: tuple[int, int],
    ) -> tuple[MonitoringOutput, MonitoringOutput]:
        """
        Inner helper method that performs a single scan of two batches
        """

        first_batch_cont, first_batch_cat = self._prepare_data(
            X,
            y,
            continuous_columns_ids,
            categorical_columns_ids,
            y_categorical,
            first_batch_indexes,
        )
        second_batch_cont, second_batch_cat = self._prepare_data(
            X,
            y,
            continuous_columns_ids,
            categorical_columns_ids,
            y_categorical,
            second_batch_indexes,
        )
        cont_algorithm = self.continuous_ma_builder(self.batch_size).fit(
            first_batch_cont
        )
        cat_algorithm = self.categorical_ma_builder(self.batch_size).fit(
            first_batch_cat
        )

        cont_output = cont_algorithm.detect(second_batch_cont)[0]
        cat_output = cat_algorithm.detect(second_batch_cat)[0]

        return cont_output, cat_output

    def _scan_data(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        continuous_columns_ids: list[int],
        categorical_columns_ids: list[int],
        y_categorical: bool,
    ) -> Report:
        """Scan the data to identify different data partitions according to monitoring algorithm.

        - Step 0: split data into separate batches of size `batch_size`
        - Step 1: For each adjacent batch perform drift detection
        - Step 2: Merge batches that belong to the same distribution
        - Step 3: For each non-adjacent merged group perform drift detection
        - Step 4: Assign label merging groups that belong to the same distribution

        Step 3 and 4 are important because they identify groups in different time periods that belong
        to the same distribution.
        """
        # Step 0: compute batch indexes
        batch_indexes = [
            (i, i + self.batch_size) for i in range(0, X.shape[0], self.batch_size)
        ]

        # Step 1 and 2: drift detection for adjacent batches and merge into batches
        merged_batches = []
        current_batch_start = 0
        for batch_id in range(len(batch_indexes) - 1):
            current_batch_indexes = batch_indexes[batch_id]
            next_batch_indexes = batch_indexes[batch_id + 1]

            cont_output, cat_output = self._single_scan_data(
                X,
                y,
                continuous_columns_ids,
                categorical_columns_ids,
                y_categorical,
                current_batch_indexes,
                next_batch_indexes,
            )

            if cont_output.drift_detected | cat_output.drift_detected:
                # if a drift is detected then, we close the current batch and open a new one
                merged_batches.append(
                    (current_batch_start, current_batch_indexes[1] - 1)
                )
                current_batch_start = next_batch_indexes[0]

        # analysis is terminated, we add the last batch
        merged_batches.append((current_batch_start, batch_indexes[-1][1] - 1))

        # Step 3: non adjacent drift detection
        non_adjacent_pairs = [
            (i, j)
            for i in range(len(merged_batches))
            for j in range(i + 2, len(merged_batches))
        ]
        same_distributions = defaultdict(list)

        # TODO: parallelize this loop by getting a N_JOBS parameter in the method / class constructor
        for pair in non_adjacent_pairs:
            cont_output, cat_output = self._single_scan_data(
                X,
                y,
                continuous_columns_ids,
                categorical_columns_ids,
                y_categorical,
                merged_batches[pair[0]],
                merged_batches[pair[1]],
            )

            # if no drift is detected the two batches are considered to belong to the same distribution
            # and are added to the same distribution list
            if not (cont_output.drift_detected | cat_output.drift_detected):
                same_distributions[pair[0]].append(pair[1])

        return Report(concepts=merged_batches, same_distributions=same_distributions)
