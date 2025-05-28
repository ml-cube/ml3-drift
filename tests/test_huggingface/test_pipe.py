from tests.conftest import is_module_available

import pytest

if is_module_available("transformers"):
    from ml3_drift.huggingface.univariate.ks import (
        KSDriftDetector,
    )
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

        # Not optimal as we are loading a big model,
        # but it didn't work with a simple model taken
        # from here:
        # https://github.com/huggingface/transformers/blob/6e3063422c4b1c014aa60c32b9254fd2902f0f28/tests/pipelines/test_pipelines_feature_extraction.py#L46
        # We should do something.
        pipe = HuggingFaceDriftDetectionPipeline(
            drift_detector=KSDriftDetector(),
            task="feature-extraction",
            model="hf-internal-testing/tiny-random-distilbert",
            framework="pt",
        )

        pipe.fit_detector(
            text_data,
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (1, 32), (
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
        assert pipe._drift_detector.X_ref_.shape == (1, 32), (
            "Reference data shape mismatch."
        )

        pipe(
            text_data,
            return_tensors=return_tensors,
        )

        pipe.fit_detector(
            [text_data, text_data],
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (2, 32), (
            "Reference data shape mismatch."
        )

        pipe(
            text_data,
            return_tensors=return_tensors,
        )

    @pytest.mark.parametrize("return_tensors", [None, "pt"])
    def test_image(self, image_data, return_tensors):
        """
        Test pipeline with image data for drift detection.
        """

        # Not optimal as we are loading a big model,
        # We should do something.
        pipe = HuggingFaceDriftDetectionPipeline(
            drift_detector=KSDriftDetector(),
            task="image-feature-extraction",
            model="hf-internal-testing/tiny-random-vit",
            framework="pt",
        )

        pipe.fit_detector(
            image_data,
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (1, 32), (
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
        assert pipe._drift_detector.X_ref_.shape == (1, 32), (
            "Reference data shape mismatch."
        )

        pipe(
            image_data,
            return_tensors=return_tensors,
        )

        pipe.fit_detector(
            [image_data, image_data],
            return_tensors=return_tensors,
        )

        assert pipe._drift_detector.is_fitted
        assert pipe._drift_detector.X_ref_.shape == (2, 32), (
            "Reference data shape mismatch."
        )

        pipe(
            image_data,
            return_tensors=return_tensors,
        )
