import pytest

from ml3_drift.monitoring.univariate.discrete.chi_square import ChiSquareAlgorithm
from ml3_drift.monitoring.univariate.continuous.ks import KSAlgorithm
from tests.conftest import is_module_available

if is_module_available("sklearn"):
    import numpy as np

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.utils.estimator_checks import parametrize_with_checks

    from ml3_drift.sklearn.base import SklearnDriftDetector

else:
    # Prevent tests from running if sklearn is not available
    pytest.skip(allow_module_level=True)


class TestSklearnDriftDetector:
    """
    Test suite for KSDriftDetector in the SKlearn module
    """

    @parametrize_with_checks(
        [
            SklearnDriftDetector(monitoring_algorithm=KSAlgorithm()),
            SklearnDriftDetector(monitoring_algorithm=ChiSquareAlgorithm()),
        ]
    )
    def test_sklearn_compatible_estimator(self, estimator, check):
        """
        Sklearn utility to check estimator compliance.
        See https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html
        """
        check(estimator)

    @pytest.mark.parametrize(
        "detector",
        [
            SklearnDriftDetector(monitoring_algorithm=KSAlgorithm()),
            SklearnDriftDetector(monitoring_algorithm=ChiSquareAlgorithm()),
        ],
    )
    def test_supports_none(self, detector):
        """
        Test that the estimator fails when fit with None.
        """
        detector.fit(np.array([[2, 2, None, 3]]).reshape(-1, 1))
        detector.fit(np.array([[2, 2, None, 3]]).reshape(-1, 1), None)

    @pytest.mark.parametrize(
        "detector",
        [
            SklearnDriftDetector(monitoring_algorithm=KSAlgorithm()),
            SklearnDriftDetector(monitoring_algorithm=ChiSquareAlgorithm()),
        ],
    )
    def test_fit(self, detector):
        """
        Test the fit method of KSDriftDetector.
        """
        # Create a sample dataset
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1, 0])

        # Fit the detector to the data
        detector.fit(X, y)

        # Check that the detector is fitted
        assert detector.is_fitted_ is True

    @pytest.mark.parametrize(
        "detector",
        [
            SklearnDriftDetector(monitoring_algorithm=KSAlgorithm()),
            SklearnDriftDetector(monitoring_algorithm=ChiSquareAlgorithm()),
        ],
    )
    def test_in_pipeline(self, detector):
        """
        Test KSDriftDetector in a pipeline.
        """
        # Create a sample dataset

        cat_pipe = Pipeline(
            steps=[
                ("detector", detector),
                (
                    "regr",
                    LinearRegression(),
                ),
            ]
        )

        train_cont_data = np.column_stack(
            (
                np.random.randint(5, size=(100,)),
                np.random.randint(5, size=(100,)),
            )
        )

        y = np.random.rand(100)

        # Fit the detector to the data
        cat_pipe.fit(train_cont_data, y)

        # Check that the detector is fitted
        assert cat_pipe.named_steps["detector"].is_fitted_ is True

        cat_pipe.predict(train_cont_data)
