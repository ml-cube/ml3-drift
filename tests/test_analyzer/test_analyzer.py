import numpy as np
import pytest
from ml3_drift.analysis.analyzer.batch import BatchDataDriftAnalyzer
from ml3_drift.monitoring.multivariate.bonferroni import BonferroniCorrectionAlgorithm
from ml3_drift.monitoring.univariate.continuous.ks import KSAlgorithm
from ml3_drift.monitoring.univariate.discrete.chi_square import ChiSquareAlgorithm
from tests.conftest import is_module_available


input_types = ["cont", "cat", "mix"]
y_types = ["cont", "cat", None]
n_drifts = [0, 1, 2]
data_formats = ["numpy", "polars", "pandas"]

input_definition_test_params: list[tuple[str, str | None, int, str]] = [
    (input_type, y_type, n_drift, data_format)
    for input_type in input_types
    for y_type in y_types
    for n_drift in n_drifts
    for data_format in data_formats
]

if not is_module_available("polars"):
    input_definition_test_params = [
        (input_type, y_type, n_drifts, data_format)
        for input_type, y_type, n_drifts, data_format in input_definition_test_params
        if data_format != "polars"
    ]

if not is_module_available("pandas"):
    input_definition_test_params = [
        (input_type, y_type, n_drifts, data_format)
        for input_type, y_type, n_drifts, data_format in input_definition_test_params
        if data_format != "pandas"
    ]


@pytest.mark.parametrize(
    "input_type, y_type, n_drifts, data_format",
    input_definition_test_params,
)
def test_batch_analyzer(input_type, y_type, n_drifts, data_format):
    rng = np.random.default_rng(2)
    n_samples = 300
    n_cont = 2
    n_cat = 2

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

    analyzer = BatchDataDriftAnalyzer(
        continuous_ma_builder=lambda comparison_size: BonferroniCorrectionAlgorithm(
            comparison_size, lambda p_value: KSAlgorithm(comparison_size, p_value)
        ),
        categorical_ma_builder=lambda comparison_size: BonferroniCorrectionAlgorithm(
            comparison_size,
            lambda p_value: ChiSquareAlgorithm(comparison_size, p_value),
        ),
        batch_size=50,
    )

    # Run analyze
    report = analyzer.analyze(
        X,
        y=y,
        continuous_columns=continuous_columns if continuous_columns else None,
        categorical_columns=categorical_columns if categorical_columns else None,
        y_categorical=y_categorical,
    )

    assert hasattr(report, "concepts")
    assert isinstance(report.concepts, list)
    assert len(report.concepts) == n_drifts + 1
