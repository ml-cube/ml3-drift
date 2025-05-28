import numpy as np
from tests.conftest import is_module_available

if is_module_available("transformers"):
    from ml3_drift.huggingface.univariate.ks import KSDriftDetector
else:
    import pytest

    pytest.skip(
        "HuggingFace transformers module is not available.",
        allow_module_level=True,
    )


class TestKSDriftDetector:
    """
    Test suite for KSDriftDetector in the HuggingFace module
    """

    def test_fit(self):
        """
        Test the fit method of KSDriftDetector.
        """
        # Create a sample dataset
        X = np.array([[1, 2], [2, 3], [3, 4]])

        # Create an instance of the detector
        detector = KSDriftDetector()

        # Fit the detector to the data
        detector.fit(X)

        # Check that the detector is fitted
        assert detector.is_fitted is True
