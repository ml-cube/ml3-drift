import numpy as np

from tests.conftest import is_module_available

if is_module_available("sklearn"):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.utils.estimator_checks import parametrize_with_checks

    from ml3_drift.sklearn.univariate.chi_square import ChiSquareDriftDetector

else:
    import pytest

    pytest.skip(
        allow_module_level=True,
    )


class TestChiSquareDriftDetector:
    """
    Test suite for ChiSquareDriftDetector in the SKlearn module
    """

    @parametrize_with_checks([ChiSquareDriftDetector()])
    def test_sklearn_compatible_estimator(self, estimator, check):
        """
        Sklearn utility to check estimator compliance.
        See https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html
        """
        check(estimator)

    def test_fit(self):
        """
        Test the fit method of ChiSquareDriftDetector.
        """
        # Create a sample dataset
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1, 0])

        # Create an instance of the detector
        detector = ChiSquareDriftDetector()

        # Fit the detector to the data
        detector.fit(X, y)

        # Check that the detector is fitted
        assert detector.is_fitted_ is True

    def test_fit_with_none(self):
        """
        Test that the estimator can fit with None.
        """
        detector = ChiSquareDriftDetector()
        detector.fit(np.array([[2, 2, None, 3, np.nan]]).reshape(-1, 1))
        assert detector.is_fitted_ is True
        assert list(detector.categories_.keys()) == [0]
        assert list(detector.categories_.values()) == [[2, 3]]
        assert detector.freqs_ref_[0] == {2: 2, 3: 1}

        detector.predict(np.array([[2, 2, None, 3, np.nan]]).reshape(-1, 1))

    def test_in_pipeline(self):
        """
        Test ChiSquareDriftDetector in a pipeline.
        """
        # Create a sample dataset

        cat_pipe = Pipeline(
            steps=[
                ("chi2", ChiSquareDriftDetector()),
                ("onehot", OneHotEncoder()),
                (
                    "regr",
                    LinearRegression(),
                ),  # Not a proper model for categorical data, but just for demonstration
            ]
        )

        train_cat_data = np.column_stack(
            (
                np.random.choice(["A", "B", "C"], size=100),
                np.random.choice(["X", "Y", "Z"], size=100),
            )
        )

        y = np.random.randn(100)

        # Fit the detector to the data
        cat_pipe.fit(train_cat_data, y)

        # Check that the detector is fitted
        assert cat_pipe.named_steps["chi2"].is_fitted_ is True

        cat_pipe.predict(train_cat_data)
