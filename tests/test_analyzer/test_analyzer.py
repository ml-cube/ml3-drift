import pytest
from ml3_drift.analysis.analyzer.batch import BatchDataDriftAnalyzer
from ml3_drift.monitoring.algorithms.batch.bonferroni import BonferroniCorrectionAlgorithm
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm
from ml3_drift.monitoring.algorithms.batch.chi_square import ChiSquareAlgorithm
from tests.conftest import is_module_available, build_data

# Generate all possible combinations, then remove those that are not available
# in the current environment according to the installed extras.
input_types = ["cont", "cat", "mix"]
y_types = ["cont", "cat", None]
n_drifts = [0, 1, 2]
data_formats = ["numpy", "polars", "pandas"]
with_default_params = [True, False]

input_definition_test_params: list[tuple[str, str | None, int, str, bool]] = [
    (input_type, y_type, n_drift, data_format, with_default)
    for input_type in input_types
    for y_type in y_types
    for n_drift in n_drifts
    for data_format in data_formats
    for with_default in with_default_params
]

if not is_module_available("polars"):
    input_definition_test_params = [
        (input_type, y_type, n_drifts, data_format, with_default_params)
        for input_type, y_type, n_drifts, data_format, with_default_params in input_definition_test_params
        if data_format != "polars"
    ]

if not is_module_available("pandas"):
    input_definition_test_params = [
        (input_type, y_type, n_drifts, data_format, with_default_params)
        for input_type, y_type, n_drifts, data_format, with_default_params in input_definition_test_params
        if data_format != "pandas"
    ]


@pytest.mark.parametrize(
    "input_type, y_type, n_drifts, data_format, with_default",
    input_definition_test_params,
)
def test_batch_analyzer(input_type, y_type, n_drifts, data_format, with_default):
    """
    Test the BatchDataDriftAnalyzer with various input types, y types, number of drifts, and data formats.
    """
    n_samples = 300
    n_cont = 2
    n_cat = 2

    X, y, continuous_columns, categorical_columns, y_categorical = build_data(
        input_type=input_type,
        y_type=y_type,
        n_drifts=n_drifts,
        n_samples=n_samples,
        n_cont=n_cont,
        n_cat=n_cat,
        data_format=data_format,
        seed=2,
    )

    if with_default:
        continuous_algo = None
        categorical_algo = None

    else:
        continuous_algo = BonferroniCorrectionAlgorithm(
            algorithm=KSAlgorithm(p_value=0.05)
        )
        categorical_algo = BonferroniCorrectionAlgorithm(
            algorithm=ChiSquareAlgorithm(p_value=0.05)
        )

    analyzer = BatchDataDriftAnalyzer(
        continuous_monitoring_algorithm=continuous_algo,
        categorical_monitoring_algorithm=categorical_algo,
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
