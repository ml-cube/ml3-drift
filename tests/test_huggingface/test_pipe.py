from ml3_drift.monitoring.algorithms.batch.bonferroni import BonferroniCorrectionAlgorithm
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm
from tests.conftest import is_module_available

import pytest

if is_module_available("transformers"):
    from ml3_drift.huggingface.drift_detection_pipeline import (
        HuggingFaceDriftDetectionPipeline,
    )
else:
    pytest.skip(
        "HuggingFace transformers module is not available.",
        allow_module_level=True,
    )


class TestHuggingFaceDriftDetectionPipeline:
    """
    Test suite for the HuggingFace drift detection pipeline.
    """

    @pytest.mark.parametrize("return_tensors", [None, "pt"])
    def test_text(self, text_data, return_tensors):
        """
        Test pipeline with text data for drift detection.
        """

        pipe = HuggingFaceDriftDetectionPipeline(
            drift_detector=KSAlgorithm(p_value=0.05),
            task="feature-extraction",
            model="hf-internal-testing/tiny-random-distilbert",
            framework="pt",
        )

        pipe.fit_detector(
            text_data,
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (32, 1), (
            "Reference data shape mismatch."
        )

        pipe(
            text_data,
            return_tensors=return_tensors,
        )

        pipe.fit_detector(
            [text_data],
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (32, 1), (
            "Reference data shape mismatch."
        )

        pipe(
            text_data,
            return_tensors=return_tensors,
        )

        pipe = HuggingFaceDriftDetectionPipeline(
            drift_detector=BonferroniCorrectionAlgorithm(
                p_value=0.05, algorithm=KSAlgorithm()
            ),
            task="feature-extraction",
            model="hf-internal-testing/tiny-random-distilbert",
            framework="pt",
        )

        pipe.fit_detector(
            [text_data, text_data],
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted

        pipe(
            [text_data, text_data],
            return_tensors=return_tensors,
        )

    @pytest.mark.parametrize("return_tensors", [None, "pt"])
    def test_image(self, image_data, return_tensors):
        """
        Test pipeline with image data for drift detection.
        """

        pipe = HuggingFaceDriftDetectionPipeline(
            drift_detector=KSAlgorithm(p_value=0.05),
            task="image-feature-extraction",
            model="hf-internal-testing/tiny-random-vit",
            framework="pt",
        )

        pipe.fit_detector(
            image_data,
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (32, 1), (
            "Reference data shape mismatch."
        )

        pipe(
            image_data,
            return_tensors=return_tensors,
        )

        pipe.fit_detector(
            [image_data],
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (32, 1), (
            "Reference data shape mismatch."
        )

        pipe(
            image_data,
            return_tensors=return_tensors,
        )

        pipe = HuggingFaceDriftDetectionPipeline(
            drift_detector=BonferroniCorrectionAlgorithm(
                p_value=0.05, algorithm=KSAlgorithm()
            ),
            task="image-feature-extraction",
            model="hf-internal-testing/tiny-random-vit",
            framework="pt",
        )

        pipe.fit_detector(
            [image_data, image_data],
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted

        pipe(
            [image_data, image_data],
            return_tensors=return_tensors,
        )
