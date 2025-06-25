import numpy as np
import pytest
from ml3_drift.analysis.analyzer.batch import BatchDataDriftAnalyzer
from ml3_drift.analysis.analyzer.stream import StreamDataDriftAnalyzer
from ml3_drift.monitoring.multivariate.bonferroni import BonferroniCorrectionAlgorithm
from ml3_drift.monitoring.univariate.continuous.ks import KSAlgorithm
from ml3_drift.monitoring.univariate.discrete.chi_square import ChiSquareAlgorithm


@pytest.mark.parametrize(
    "input_type, y_type, n_drifts",
    [
        # Only input continuous
        ("cont", None, 0),
        ("cont", None, 1),
        ("cont", None, 2),
        # Only input categorical
        ("cat", None, 0),
        ("cat", None, 1),
        ("cat", None, 2),
        # Input mixed
        ("mix", None, 0),
        ("mix", None, 1),
        ("mix", None, 2),
        # Input continuous + target continuous
        ("cont", "cont", 1),
        # Input categorical + target continuous
        ("cat", "cont", 1),
        # Input mixed + target continuous
        ("mix", "cont", 1),
        # Input continuous + target categorical
        ("cont", "cat", 1),
        # Input categorical + target categorical
        ("cat", "cat", 1),
        # Input mixed + target categorical
        ("mix", "cat", 1),
    ],
)
def test_stream_analyzer_numpy(input_type, y_type, n_drifts):
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
            if i == 0:
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
        X = X_cont
        continuous_columns = list(range(n_cont))
        categorical_columns = []
    elif input_type == "cat":
        X = X_cat
        continuous_columns = []
        categorical_columns = list(range(n_cat))
    elif input_type == "mix":
        X = np.hstack([X_cont, X_cat])
        continuous_columns = list(range(n_cont))
        categorical_columns = list(range(n_cont, n_cat))
    else:
        raise ValueError("Unknown input_type")

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

    analyzer = StreamDataDriftAnalyzer(
        continuous_ma_builder=lambda comparison_size: BonferroniCorrectionAlgorithm(
            comparison_size, lambda p_value: KSAlgorithm(comparison_size, p_value)
        ),
        categorical_ma_builder=lambda comparison_size: BonferroniCorrectionAlgorithm(
            comparison_size,
            lambda p_value: ChiSquareAlgorithm(comparison_size, p_value),
        ),
        reference_size=50,
        comparison_window_size=50,
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


@pytest.mark.parametrize(
    "input_type, y_type, n_drifts",
    [
        # Only input continuous
        ("cont", None, 0),
        ("cont", None, 1),
        ("cont", None, 2),
        # Only input categorical
        ("cat", None, 0),
        ("cat", None, 1),
        ("cat", None, 2),
        # Input mixed
        ("mix", None, 0),
        ("mix", None, 1),
        ("mix", None, 2),
        # Input continuous + target continuous
        ("cont", "cont", 1),
        # Input categorical + target continuous
        ("cat", "cont", 1),
        # Input mixed + target continuous
        ("mix", "cont", 1),
        # Input continuous + target categorical
        ("cont", "cat", 1),
        # Input categorical + target categorical
        ("cat", "cat", 1),
        # Input mixed + target categorical
        ("mix", "cat", 1),
    ],
)
def test_batch_analyzer_numpy(input_type, y_type, n_drifts):
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
        X = X_cont
        continuous_columns = list(range(n_cont))
        categorical_columns = []
    elif input_type == "cat":
        X = X_cat
        continuous_columns = []
        categorical_columns = list(range(n_cat))
    elif input_type == "mix":
        X = np.hstack([X_cont, X_cat])
        continuous_columns = list(range(n_cont))
        categorical_columns = list(range(n_cont, n_cat))
    else:
        raise ValueError("Unknown input_type")

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
