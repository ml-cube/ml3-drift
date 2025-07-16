import numpy as np

from collections.abc import Callable
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from abc import ABC, abstractmethod
from sklearn.utils.validation import validate_data
from ml3_drift.callbacks.models import DriftInfo
from ml3_drift.enums.monitoring import DataDimension, DataType
from ml3_drift.monitoring.base import MonitoringAlgorithm

from copy import deepcopy


class BaseDriftDetector(TransformerMixin, BaseEstimator, ABC):
    """
    Base class for drift detector.
    Base Drift Detectors are neither transformers nor predictors, they
    just observe the data and detects drift, executing specified
    actions when necessary.
    For this reason, they implement both the transform and predict methods.

    Parameters
    ----------
    callbacks: list[Callable[[DriftInfo], None]
        list of callbacks function used to act when drift are detected
    """

    def __init__(self, callbacks: list[Callable[[DriftInfo], None]] | None = None):
        super().__init__()
        self.callbacks = callbacks

    @abstractmethod
    def _fit(self, X, y=None):
        """
        Fit method that should be implemented in child classes
        """

    @abstractmethod
    def _detect(self, X) -> list[DriftInfo]:
        """
        Core method for detecting drift that is implemented in child classes.
        """

    def fit(self, X, y=None):
        """
        Fit method. Calls _inner_fit and returns self
        """

        X = self._validate_data(X, y, reset=True)
        self._fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform method. Calls _detect method and return X.
        This step does not change the data but only performs drift detection.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        if (drift_info_list := self._detect(X)) and (self.callbacks is not None):
            for drift_info in drift_info_list:
                for callback in self.callbacks:
                    callback(drift_info)
        return X

    def predict(self, X):
        """
        Predict method. Calls _detect method and return X.
        This step does not change the data but only performs drift detection.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        if (drift_info_list := self._detect(X)) and (self.callbacks is not None):
            for drift_info in drift_info_list:
                for callback in self._callbacks:
                    callback(drift_info)
        return X

    def _validate_data(self, X, y=None, reset=False):
        """
        Validate data method. This calls validate_data sklearn method with
        provided parameters and returns the validated X.
        Child classes can override with their own validation methods if needed
        or just call the base class method with the custom parameters.
        """

        # Workaround since validate data doesn't return y if it is None
        if y is None:
            X = validate_data(self, X, reset=reset, accept_sparse=False)
        else:
            X, _ = validate_data(self, X, y, reset=reset, accept_sparse=False)
        return X

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Currently empty, but can be used to add tags to the estimator
        return tags


class SklearnDriftDetector(TransformerMixin, BaseEstimator):
    """Adapter class for sklearn library.

    It is subclass of base sklearn classes and acts like a transformer but it do not
    transform data it receives. A drift detector just observes data and detect drifts
    executing provided callbacks.

    It implements both transform and predict methods.

    Parameters
    ----------
    monitoring_algorithm: MonitoringAlgorithm
        instance of the monitoring algorithm to use during drift detection. If it is
        for univariate data, then one clone for each column is used.
    """

    def __repr__(self):
        return f"SklearnDriftDetector({self.monitoring_algorithm})"

    def __str__(self):
        return f"SklearnDriftDetector({self.monitoring_algorithm})"

    def __init__(self, monitoring_algorithm: MonitoringAlgorithm):
        super().__init__()
        self.monitoring_algorithm = monitoring_algorithm
        self._monitoring_algorithm_list: list[MonitoringAlgorithm] = []

    def fit(self, X, y=None):
        """Fit method.

        Validates data, and call fit method of monitoring algorithm.
        If the monitoring algorithm is univariate then for each column a clone
        is used.
        """
        self._monitoring_algorithm_list = []
        X = self._validate_data(X, reset=True)

        if (
            self.monitoring_algorithm.specs().data_dimension == DataDimension.UNIVARIATE
        ) and (X.shape[1] > 1):
            for i in range(X.shape[1]):
                ma = deepcopy(self.monitoring_algorithm)
                ma.fit(X[:, i])
                self._monitoring_algorithm_list.append(ma)
        else:
            self.monitoring_algorithm.fit(X)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform method.

        Calls detect to the internal monitoring algorithm and return X as it is.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        if (
            self.monitoring_algorithm.specs().data_dimension == DataDimension.UNIVARIATE
        ) and (X.shape[1] > 1):
            for i, ma in enumerate(self._monitoring_algorithm_list):
                ma.detect(X[:, i])
        else:
            self.monitoring_algorithm.detect(X)
        return X

    def predict(self, X):
        """Predict method.

        Calls detect to the internal monitoring algorithm and return X as it is.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        if (
            self.monitoring_algorithm.specs().data_dimension == DataDimension.UNIVARIATE
        ) and (X.shape[1] > 1):
            for i, ma in enumerate(self._monitoring_algorithm_list):
                ma.detect(X[:, i])
        else:
            self.monitoring_algorithm.detect(X)

        return X

    def _validate_data(self, X, y=None, reset=False):
        """
        Validate data method. This calls validate_data sklearn method with
        provided parameters and returns the validated X.
        Child classes can override with their own validation methods if needed
        or just call the base class method with the custom parameters.
        """
        # Workaround since validate data doesn't return y if it is None
        if self.monitoring_algorithm.specs().data_type in [
            DataType.DISCRETE,
            DataType.MIX,
        ]:
            dtype = None
        else:
            dtype = "numeric"

        if y is None:
            X = validate_data(
                self,
                X,
                reset=reset,
                accept_sparse=False,
                ensure_all_finite=False,
                dtype=dtype,
            )
        else:
            X, _ = validate_data(
                self,
                X,
                y,
                reset=reset,
                accept_sparse=False,
                ensure_all_finite=False,
                dtype=dtype,
            )
        X = self._safe_nan_clean(X)
        return X

    def _safe_nan_clean(self, X):
        """
        Clean NaN values from the data. Different approach according to the type of data.
        """
        # Extend with other types
        if isinstance(X, np.ndarray):
            if np.issubdtype(X.dtype, np.number):
                if len(X.shape) > 1:
                    return X[~np.any(np.isnan(X), axis=1)]
                else:
                    return X[~np.any(np.isnan(X))]
            elif np.issubdtype(X.dtype, np.object_):
                # Since the dtype is object we cannot apply the function np.isnan
                # Therefore, we need to iterate over each row
                # According to the number of columns we can have an array or a scalar
                # if it is an array then we cast it to a list and check if it contains None
                # if it is a scalar then we use np.isnan if it is a float or not None otherwise
                return X[
                    np.array(
                        [
                            [
                                not (
                                    (isinstance(x, np.ndarray) and None in x.tolist())
                                    or (isinstance(x, float) and np.isnan(x))
                                    or (x is None)
                                )
                            ]
                            for x in X
                        ]
                    ).ravel()
                ]
            elif np.issubdtype(X.dtype, np.str_):
                return X[~np.any(np.isin(X, [None, np.nan]))]
            else:
                raise ValueError("Unsupported data type")
        return X

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True

        if self.monitoring_algorithm.specs().data_type in [
            DataType.DISCRETE,
            DataType.MIX,
        ]:
            tags.input_tags.categorical = True
            tags.input_tags.string = True

        return tags
