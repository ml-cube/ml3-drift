from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Union
from typing_extensions import TypeIs

from ml3_drift.analysis.report import Report
from ml3_drift.monitoring.algorithms.batch.bonferroni import BonferroniCorrectionAlgorithm
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm
from ml3_drift.monitoring.algorithms.batch.chi_square import (
    ChiSquareAlgorithm,
)
from ml3_drift.monitoring.base.base import MonitoringAlgorithm

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


class DataDriftAnalyzer(ABC):
    """
    Analyze a dataset identifying the sequence of distributions due to data drifts.

    Parameters
    ----------
    continuous_monitoring_algorithm: MonitoringAlgorithm | None
        Algorithm used to monitor continuous data. If None, a default algorithm is used.
    categorical_monitoring_algorithm: MonitoringAlgorithm | None
        Algorithm used to monitor categorical data. If None, a default algorithm is used.
    """

    def __init__(
        self,
        continuous_monitoring_algorithm: MonitoringAlgorithm | None = None,
        categorical_monitoring_algorithm: MonitoringAlgorithm | None = None,
    ):
        # We use default algorithms if None is provided.
        if continuous_monitoring_algorithm is None:
            continuous_monitoring_algorithm = BonferroniCorrectionAlgorithm(
                algorithm=KSAlgorithm(),
            )
        if categorical_monitoring_algorithm is None:
            categorical_monitoring_algorithm = BonferroniCorrectionAlgorithm(
                algorithm=ChiSquareAlgorithm(),
            )

        self.continuous_monitoring_algorithm = continuous_monitoring_algorithm
        self.categorical_monitoring_algorithm = categorical_monitoring_algorithm

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
                    f"Type not valid, expecting polars DataFrame or pandas DataFrame when columns has string values. Got {type(X)}"
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

    @abstractmethod
    def _scan_data(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        continuous_columns_ids: list[int],
        categorical_columns_ids: list[int],
        y_categorical: bool,
    ) -> Report:
        """Scan the data to identify different data partitions according to monitoring algorithm."""

    def analyze(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"],
        y: Union[None, np.ndarray, "pd.DataFrame", "pl.DataFrame"],
        continuous_columns: list[str] | list[int] | None,
        categorical_columns: list[str] | list[int] | None,
        y_categorical: bool,
    ) -> Report:
        """Analyze the data to split them into different distribution according to drift detectors.

        If target is provided then concept drift is used as split criterion, otherwise, it uses input drift.

        Parameters
        ----------
        X: input data. Can be numpy array, pandas dataframe or polars dataframe
        y: target data. It is optional and can be numpy array, pandas dataframe or polars dataframe
        continuous_columns: if not None it is the indices or names of the columns that are continuous
        categorical_columns: if not None it is the indices or names of the columns that are categorical
        y_categorical: if True, then the target is categorical, otherwise it is considered as continuous

        Output
        ------
        Report object containing information about identified data groups
        """
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
