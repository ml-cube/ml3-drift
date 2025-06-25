import numpy as np
from typing import TYPE_CHECKING, Callable, TypeIs, Union

from ml3_drift.analysis.report import Report
from ml3_drift.monitoring.base import MonitoringAlgorithm

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

POLARS = True
try:
    import polars as pl
except ModuleNotFoundError:
    POLARS = False


PANDAS = True
try:
    import pandas as pd
except ModuleNotFoundError:
    PANDAS = False


class DataDriftAnalyzer:
    """
    Analyze a dataset identifying the sequence of distributions due to data drifts.
    """

    def __init__(
        self,
        continuous_ma_builder: Callable[[int], MonitoringAlgorithm],
        categorical_ma_builder: Callable[[int], MonitoringAlgorithm],
        reference_size: int = 100,
        comparison_window_size: int = 100,
    ):
        self.reference_size = reference_size
        self.comparison_window_size = comparison_window_size

        self.continuous_ma_builder = continuous_ma_builder
        self.categorical_ma_builder = categorical_ma_builder

    def _is_list_str(self, columns: list[str] | list[int]) -> TypeIs[list[str]]:
        """Verify if the input variable is a list of str in any element"""

        return all(isinstance(elem, str) for elem in columns)

    def _to_index(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"],
        columns: list[str] | list[int] | None,
    ) -> list[int]:
        """Translate the list of columns in list of indices.

        If columns is None then all the indexes are returned.
        If columns is list[int] then it is directly returned.
        If columns is list[str] then the indexes are retrieved from column names,
        in this case X must be a DataFrame."""

        if columns is None:
            return list(range(X.shape[0]))

        if self._is_list_str(columns):
            if POLARS and isinstance(X, pl.DataFrame):
                return [i for (i, c) in enumerate(X.columns) if c in columns]
            elif PANDAS and isinstance(X, pd.DataFrame):
                return [i for (i, c) in enumerate(X.columns) if c in columns]
            else:
                raise ValueError(
                    "Type not valid, expecting polars DataFrame or pandas DataFrame when columns has string values. Got {type(X)}"
                )
        return columns

    def _to_numpy(
        self, X: Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]
    ) -> np.ndarray:
        """Transform input data into numpy array"""

        if POLARS and isinstance(X, pl.DataFrame):
            return X.to_numpy()
        elif PANDAS and isinstance(X, pd.DataFrame):
            return X.to_numpy()
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(
                f"Type not valid, expecting numpy array, polars DataFrame or pandas DataFrame. Got {type(X)}"
            )

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

    def analyze(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"],
        y: Union[None, np.ndarray, "pd.DataFrame", "pl.DataFrame"] = None,
        continuous_columns: list[str] | list[int] | None = None,
        categorical_columns: list[str] | list[int] | None = None,
        y_categorical: bool = False,
    ) -> Report:
        # Shape check
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError(
                f"When target y is not None it must have the same rows of input X. Got X: {X.shape} and y: {y.shape}"
            )

        # Continuous and categorical columns to canonical form
        if continuous_columns is not None:
            continuous_columns_ids = self._to_index(X, continuous_columns)
        else:
            continuous_columns_ids = []

        if categorical_columns is not None:
            categorical_columns_ids = self._to_index(X, categorical_columns)
        else:
            categorical_columns_ids = []

        # Input and target in canonical form
        array_X = self._to_numpy(X)

        if y is not None:
            array_y = self._to_numpy(y)
        else:
            array_y = None

        # Data analysis
        report = self._scan_data(
            array_X,
            array_y,
            continuous_columns_ids,
            categorical_columns_ids,
            y_categorical,
        )

        return report
