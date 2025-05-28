import logging

import numpy as np
from ml3_drift.sklearn.univariate.ks import KSDriftDetector
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml3_drift.callbacks.base import logger_callback
from functools import partial

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Define your pipeline as usual, but also add a drift detector
    drift_detector = KSDriftDetector(
        callbacks=[
            partial(
                logger_callback,
                logger=logger,
                level=logging.CRITICAL,
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("monitoring", drift_detector),
            (
                "model",
                DecisionTreeRegressor(),
            ),
        ]
    )

    X_train = np.random.randn(100, 2)  # Example training data
    y_train = np.random.randn(100)  # Example training labels
    X_test = np.random.randn(50, 2)  # Example test data
    # Add drift
    X_test[:, 0] += 1  # Introduce drift in the first feature

    # When fitting the pipeline, the drift detector will
    # save the training data as reference data.
    # No effect on the model training.
    pipeline = pipeline.fit(X_train, y_train)

    # When making predictions, the drift detector will
    # check if the incoming data is similar to the reference data
    # and execute the action you specified if a drift is detected.
    predictions = pipeline.predict(X_test)
