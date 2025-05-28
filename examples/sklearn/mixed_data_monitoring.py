from functools import partial
import logging
from ml3_drift.sklearn.univariate.ks import KSDriftDetector
from ml3_drift.sklearn.univariate.chi_square import ChiSquareDriftDetector

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np
from ml3_drift.callbacks.base import logger_callback


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Generate synthetic data for training and testing.
    # The dataset is composed by two continuous features and two categorical features.
    X_train_cont = np.random.randn(100, 2)
    X_train_cat = np.random.choice(["A", "B", "C"], size=(100, 2), p=(0.3, 0.3, 0.4))

    X_test_cont = np.random.randn(100, 2)

    # Introduce drift in the first continuous feature
    # by shifting the mean.
    X_test_cont[:, 0] += 1

    # Introduce drift in the categorical features
    # by changing the distribution of categories.
    X_test_cat = np.random.choice(["A", "B", "C"], size=(100, 2), p=(0.1, 0.85, 0.05))

    X_train = np.column_stack((X_train_cont, X_train_cat))
    y_train = np.random.randn(100)

    X_test = np.column_stack((X_test_cont, X_test_cat))
    y_test = np.random.randn(100)

    # -----------------------------------------
    # Create a pipeline with both continuous and categorical features
    # We leverage two ColumnTransformer objects to handle both
    # preprocessing and monitoring of mixed data types.

    transf = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), [0, 1]),
            ("cat", OrdinalEncoder(), [2, 3]),
        ]
    )

    monitoring_transf = ColumnTransformer(
        transformers=[
            (
                "cont",
                KSDriftDetector(
                    callbacks=[
                        partial(
                            logger_callback,
                            logger=logger,
                            level=logging.CRITICAL,
                        ),
                    ]
                ),
                [0, 1],
            ),
            (
                "cat",
                ChiSquareDriftDetector(
                    callbacks=[
                        partial(
                            logger_callback,
                            logger=logger,
                            level=logging.CRITICAL,
                        ),
                    ]
                ),
                [2, 3],
            ),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", transf),
            ("monitoring", monitoring_transf),
            (
                "model",
                DecisionTreeRegressor(),
            ),
        ]
    )

    # Fit the pipeline on the training data
    pipe.fit(X_train, y_train)

    # Predict on the test data
    print(f"Performance on test data: {pipe.score(X_test, y_test)}")
