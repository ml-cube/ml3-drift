from importlib import import_module

import pytest

import numpy as np
from PIL import Image


def is_module_available(module_name):
    """
    Check if a module is available in the current environment.

    Args:
        module_name (str): The name of the module to check.

    Returns:
        bool: True if the module is available, False otherwise.
    """
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False


# ------------------------------------
# Fixtures


@pytest.fixture
def text_data():
    """
    Fixture to provide text data for testing.
    """

    return "I am a drift detection warrior"


@pytest.fixture
def image_data():
    """
    Fixture to provide image data for testing.
    """

    return Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))


# ------------------------------------
# Test factories


def build_data(
    input_type: str,
    y_type: str | None,
    n_drifts: int,
    n_samples: int,
    n_cont: int,
    n_cat: int,
    data_format: str,
    seed: int = 0,
):
    """
    Helper function to generate data with specified characteristics.

    Args:
        input_type (str): Type of input data ('cont', 'cat', 'mix').
        y_type (str | None): Type of target data ('cont', 'cat', or None).
        n_drifts (int): Number of drifts to introduce in the data.
        n_samples (int): Number of samples to generate for each concept - total
            samples will be (n_drifts + 1) * n_samples.
        n_cont (int): Number of continuous features.
        n_cat (int): Number of categorical features.
        data_format (str): Format of the data ('numpy', 'polars', 'pandas').
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: Generated data (X, y, continuous_columns, categorical_columns, y_categorical).
    """

    rng = np.random.default_rng(seed)

    # Helper to generate data with drifts
    def generate_with_drifts(
        base_generator, drift_generator, n_drifts, n_samples, n_cols
    ):
        if n_drifts == 0:
            return base_generator(size=(n_samples, n_cols))
        data = []
        for i in range(n_drifts + 1):
            if i % 2 == 0:
                d = base_generator(size=(n_samples, n_cols))
            else:
                d = drift_generator(i, size=(n_samples, n_cols))
            data.append(d)
        return np.vstack(data)

    # Continuous features
    def cont_base_generator(size):
        return rng.normal(loc=0, scale=0.1, size=size)

    def cont_drift_generator(i, size):
        return rng.normal(loc=i + 10, scale=0.1, size=size)

    X_cont = generate_with_drifts(
        cont_base_generator, cont_drift_generator, n_drifts, n_samples, n_cont
    )

    # Categorical features
    def cat_base_generator(size):
        return rng.binomial(1, 0.3, size=size)

    def cat_drift_generator(i, size):
        return rng.binomial(1, 0.3 + (0.9 - 0.3) * (i / n_drifts), size=size)

    X_cat = generate_with_drifts(
        cat_base_generator, cat_drift_generator, n_drifts, n_samples, n_cat
    )

    # Build input X
    if input_type == "cont":
        if data_format == "polars":
            import polars as pl

            continuous_columns = list(map(str, range(n_cont)))
            X = pl.from_numpy(X_cont, schema=continuous_columns)
        elif data_format == "pandas":
            import pandas as pd

            continuous_columns = list(map(str, range(n_cont)))
            X = pd.DataFrame(X_cont)
            X.columns = continuous_columns
        else:
            X = X_cont
            continuous_columns = list(range(n_cont))
        categorical_columns = []
    elif input_type == "cat":
        if data_format == "polars":
            import polars as pl

            categorical_columns = list(map(str, range(n_cat)))
            X = pl.from_numpy(X_cat, schema=categorical_columns)
        elif data_format == "pandas":
            import pandas as pd

            categorical_columns = list(map(str, range(n_cat)))
            X = pd.DataFrame(X_cat)
            X.columns = categorical_columns
        else:
            X = X_cat
            categorical_columns = list(range(n_cat))
        continuous_columns = []
    elif input_type == "mix":
        if data_format == "polars":
            import polars as pl

            continuous_columns = list(map(str, range(n_cont)))
            categorical_columns = list(map(str, range(n_cont, n_cat + n_cont)))
            X = pl.from_numpy(
                np.hstack([X_cont, X_cat]),
                schema=continuous_columns + categorical_columns,
            )
        elif data_format == "pandas":
            import pandas as pd

            continuous_columns = list(map(str, range(n_cont)))
            categorical_columns = list(map(str, range(n_cont, n_cat + n_cont)))
            X = pd.DataFrame(
                np.hstack([X_cont, X_cat]),
            )
            X.columns = continuous_columns + categorical_columns
        else:
            X = np.hstack([X_cont, X_cat])
            continuous_columns = list(range(n_cont))
            categorical_columns = list(range(n_cont, n_cat + n_cont))
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    # Generate target y if needed
    y = None
    y_categorical = False
    if y_type == "cont":
        y = generate_with_drifts(
            cont_base_generator, cont_drift_generator, n_drifts, n_samples, 1
        )
    elif y_type == "cat":
        y_categorical = True
        y = generate_with_drifts(
            cat_base_generator, cat_drift_generator, n_drifts, n_samples, 1
        )

    return X, y, continuous_columns, categorical_columns, y_categorical
