from functools import partial
import logging
from ml3_drift.huggingface.drift_detection_pipeline import (
    HuggingFaceDriftDetectionPipeline,
)
from ml3_drift.monitoring.algorithms.batch.bonferroni import BonferroniCorrectionAlgorithm
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm
from ml3_drift.callbacks.base import logger_callback


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # reference text data
    reference_texts = [
        "Drift detection in ML models is crucial",
        "ml3-drift helps monitor drift in ML models",
        "ml3-drift provides easy integration with Hugging Face models",
        "Monitoring drift in text data is essential for maintaining model performance",
    ] * 10

    # induce drift by changing topic
    production_texts = [
        "Rosso di Sera, bel tempo si spera",
        "Una rondine non fa primavera",
        "Tanto va la gatta al lardo che ci lascia lo zampino",
        "Chi va piano va sano e va lontano",
    ] * 3

    # Create a feature extraction pipeline (basically, an embedder)
    # https://huggingface.co/tasks/feature-extraction
    # as you would do with HuggingFace

    # You can use our class as you it would do with a HuggingFace pipeline
    # (both when instantiating it and when calling it).

    # You should also pass the drift detector which will be used
    # to monitor the drift in the embeddings.

    hf_pipe = HuggingFaceDriftDetectionPipeline(
        drift_detector=BonferroniCorrectionAlgorithm(
            algorithm=KSAlgorithm(p_value=0.05),
            callbacks=[
                partial(
                    logger_callback,
                    logger=logger,
                    level=logging.CRITICAL,
                )
            ],
        ),
        task="feature-extraction",
        model="facebook/bart-base",
        framework="pt",
    )

    # We need to fit the detector on the reference data
    # before using it to monitor the drift.
    # Alternatively, you can pass an already fitted detector
    hf_pipe.fit_detector(
        reference_texts,
    )

    # Call the pipeline. No effect on the pipeline execution!
    # But the drift detector will monitor data and execute the callback
    # you specified if drift is detected
    hf_pipe(
        production_texts,
    )
