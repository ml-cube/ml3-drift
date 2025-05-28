from collections.abc import Callable
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from abc import ABC, abstractmethod
from sklearn.utils.validation import validate_data

from ml3_drift.callbacks.models import DriftInfo


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
        if drift_info_list := self._detect(X) and self.callbacks is not None:
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
