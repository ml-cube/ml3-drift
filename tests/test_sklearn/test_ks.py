from ml3_drift.monitoring.univariate.continuous.ks import KSAlgorithm
import pytest

from tests.conftest import is_module_available

if is_module_available("sklearn"):
    import numpy as np

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.utils.estimator_checks import parametrize_with_checks

    from ml3_drift.sklearn.base import SklearnDriftDetector
    from ml3_drift.sklearn.univariate.ks import KSDriftDetector

else:
    # Prevent tests from running if sklearn is not available
    pytest.skip(allow_module_level=True)


class TestKSDriftDetector:
    """
    Test suite for KSDriftDetector in the SKlearn module
    """

    @parametrize_with_checks([SklearnDriftDetector(monitoring_algorithm=KSAlgorithm())])
    def test_sklearn_compatible_estimator(self, estimator, check):
        """
        Sklearn utility to check estimator compliance.
        See https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html
        """
        check(estimator)

    def test_fails_with_none(self):
        """
        Test that the estimator fails when fit with None.
        """
        detector = KSDriftDetector()
        with pytest.raises(ValueError):
            detector.fit(np.array([[2, 2, None, 3]]).reshape(-1, 1))

        with pytest.raises(ValueError):
            detector.fit(np.array([[2, 2, None, 3]]).reshape(-1, 1), None)

    def test_fit(self):
        """
        Test the fit method of KSDriftDetector.
        """
        # Create a sample dataset
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1, 0])

        # Create an instance of the detector
        detector = KSDriftDetector()

        # Fit the detector to the data
        detector.fit(X, y)

        # Check that the detector is fitted
        assert detector.is_fitted_ is True

    def test_in_pipeline(self):
        """
        Test KSDriftDetector in a pipeline.
        """
        # Create a sample dataset

        cat_pipe = Pipeline(
            steps=[
                ("ks", KSDriftDetector()),
                (
                    "regr",
                    LinearRegression(),
                ),
            ]
        )

        train_cont_data = np.column_stack(
            (
                np.random.randn(100),
                np.random.randn(100),
            )
        )

        y = np.random.randn(100)

        # Fit the detector to the data
        cat_pipe.fit(train_cont_data, y)

        # Check that the detector is fitted
        assert cat_pipe.named_steps["ks"].is_fitted_ is True

        cat_pipe.predict(train_cont_data)

    def test_in_pipeline_new(self):
        """
        Test KSDriftDetector in a pipeline.
        """
        # Create a sample dataset

        cat_pipe = Pipeline(
            steps=[
                ("ks", SklearnDriftDetector(KSAlgorithm())),
                (
                    "regr",
                    LinearRegression(),
                ),
            ]
        )

        train_cont_data = np.column_stack(
            (
                np.random.randn(100),
                np.random.randn(100),
            )
        )

        y = np.random.randn(100)

        # Fit the detector to the data
        cat_pipe.fit(train_cont_data, y)

        # Check that the detector is fitted
        assert cat_pipe.named_steps["ks"].is_fitted_ is True

        cat_pipe.predict(train_cont_data)
