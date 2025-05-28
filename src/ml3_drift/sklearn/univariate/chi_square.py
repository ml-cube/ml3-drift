from collections import defaultdict
from collections.abc import Callable
from logging import getLogger
import numpy as np
from scipy import stats
from ml3_drift.callbacks.models import DriftInfo
from ml3_drift.sklearn.base import BaseDriftDetector
from sklearn.utils.validation import validate_data


logger = getLogger(__name__)


class ChiSquareDriftDetector(BaseDriftDetector):
    """
    Drift detector based on the Chi-Square test.
    """

    def __init__(
        self,
        p_value: float = 0.0005,
        categories: dict | None = None,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ):
        """
        Parameters
        ----------
        p_value: float
            p-value threshold for detecting drift. Default is 0.0005.
        categories: dict or None
            Dictionary with categories for each column. If None, the detector will try to infer categories
            from the data used for fitting.

        """
        super().__init__(callbacks=callbacks)
        self.p_value = p_value
        self.categories = categories

    def _fit(self, X, y=None):
        """
        Fit method. Saves reference data possible values
        for each feature and their frequencies.
        """

        # TODO: check all values exist in categories if provided

        # If categories were not provided, infer them from reference data
        if self.categories is None:
            # We collect unique values for each column
            # and convert them to a list
            # We also remove NaN values
            self.categories_ = {
                i: np.unique(self._safe_nan_clean(X[:, i])).tolist()
                for i in range(X.shape[1])
            }
        else:
            # Validate categories
            self._validate_categories(self.categories)
            self.categories_ = self.categories

        # Precompute reference frequencies
        self.freqs_ref_ = self._get_frequencies(X)

    def _validate_data(self, X, y=None, reset=False):
        """
        Validate data method. We override the base class method to
        also allow NaN values in the data.
        """

        # Inspired from https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/preprocessing/_encoders.py#L47

        # Workaround since validate data doesn't return y if it is None
        if y is None:
            X = validate_data(
                self,
                X,
                reset=reset,
                accept_sparse=False,
                ensure_all_finite=False,
                dtype=None,  # input dtype is preserved
            )
        else:
            X, _ = validate_data(
                self,
                X,
                y,
                reset=reset,
                accept_sparse=False,
                ensure_all_finite=False,
                dtype=None,  # input dtype is preserved
            )

        return X

    def _safe_nan_clean(self, X):
        """
        Clean NaN values from the data. Different approach according to the type of data.
        """

        # Extend with other types
        if isinstance(X, np.ndarray):
            if np.issubdtype(X.dtype, np.number):
                return X[~np.isnan(X)]
            elif np.issubdtype(X.dtype, np.object_):
                return X[
                    np.array(
                        [
                            x is not None and not (isinstance(x, float) and np.isnan(x))
                            for x in X
                        ]
                    )
                ]
            elif np.issubdtype(X.dtype, np.str_):
                return X[~np.isin(X, [None, np.nan])]
            else:
                raise ValueError("Unsupported data type")
        raise ValueError("Unsupported data type")

    def _detect(self, X):
        """
        Core method for detecting drift
        """

        # TODO: check all values exist in categories if provided

        # Compute frequencies for the new data
        freqs_prod = self._get_frequencies(X)
        # list containing drift info events
        drift_info_list = []

        for f in range(X.shape[1]):
            _, p_value, _, _ = stats.chi2_contingency(
                np.column_stack(
                    (
                        list(self.freqs_ref_[f].values()),
                        list(freqs_prod[f].values()),
                    )
                )
            )

            if p_value < self.p_value:
                drift_info_list.append(
                    DriftInfo(
                        feature_index=f,
                        drift_detector=self.__class__.__name__,
                        p_value=p_value,
                        threshold=self.p_value,
                    )
                )

        return drift_info_list

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Input needs to be categorical
        # We need to set also allow nan
        # otherwise tests fail
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        return tags

    def _validate_categories(self, categories):
        """
        Validate categories for each column.
        """
        if not isinstance(categories, dict):
            raise ValueError("Categories must be a dictionary")

        for i, cats in categories.items():
            if not isinstance(cats, list):
                raise ValueError(f"Categories for column {i} must be a list")
            if len(cats) == 0:
                raise ValueError(f"Categories for column {i} cannot be empty")
            # Check no NaN values in categories
            if any(np.isnan(cat) or cat is None for cat in cats):
                raise ValueError(f"Categories for column {i} cannot contain NaN values")

    def _get_frequencies(self, X) -> dict:
        """
        Get frequencies for each category in a column.

        Returns a dict like:
        {
            feature_index: {possible_value_1: count, possible_value_2: count, ...},
            ...
        }
        """

        # For each of the columns, get the frequencies of each category
        freqs: dict = defaultdict(dict)

        for i in range(X.shape[1]):
            # get counts for each category. Since self.categories_
            # doesn't contain NaN values, this is safe
            for cat in self.categories_[i]:
                freqs[i][cat] = int(np.sum(X[:, i] == cat))

        return freqs
