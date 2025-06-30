import numpy as np
import torch
from transformers import Pipeline, pipeline

from ml3_drift.monitoring.base import MonitoringAlgorithm


class HuggingFaceDriftDetectionPipeline:
    def __init__(self, drift_detector: MonitoringAlgorithm, **kwargs):  # noqa: F821
        """
        Init
        """
        # If task is provided, check that it is a feature-extraction task
        if "task" in kwargs and kwargs["task"] not in (
            "feature-extraction",
            "image-feature-extraction",
        ):
            raise ValueError(
                "DriftDetectionPipeline only supports 'feature-extraction' task."
            )

        # We also support being getting a pipeline directly

        if "pipeline" in kwargs:
            local_pipeline = kwargs.pop("pipeline")
            if not isinstance(local_pipeline, Pipeline):
                raise ValueError(
                    "If 'pipeline' is provided, it must be an instance of transformers.Pipeline."
                )
            self._pipeline = local_pipeline
        else:
            self._pipeline = pipeline(**kwargs)

        # Finally, instantiate the drift detector
        self._drift_detector = drift_detector

    def _to_numpy(self, data) -> np.ndarray:
        """
        Convert the data to numpy arrays, that can be processed by the drift detector.
        Notice that we currently don't support tensorflow tensors and will raise an error
        if they are provided.

        Args:
            data: The data to convert. It can be a list, a torch.Tensor, or a numpy array.
        Returns:
            np.ndarray: The data converted to a numpy array, where each row is a sample
            and each column is a feature. For instance, if the input is the result
            of a single sample, the output will be a 2D numpy array with shape (1, n_features).
        """

        # If data is a list, it can be due to return_tensors being false
        # or to batched data being passed. Either way, each element
        # will possibly have different shapes, so we can't
        # just convert it to a numpy array directly. The output of this
        # step is a list of numpy arrays.

        if isinstance(data, list):
            if isinstance(data[0], torch.Tensor):
                # Convert each torch.Tensor to a numpy array
                data = [d.detach().cpu().numpy() for d in data]

            elif isinstance(data[0], np.ndarray):
                # If the first element is a numpy array, we can convert the whole list
                data = np.asarray(data)

            elif not isinstance(data[0], np.ndarray | list):
                raise ValueError(
                    "Unsupported data type in the list. "
                    "Expecting either a list, numpy array or torch.Tensor."
                    f" Got {type(data[0])} instead."
                )

            # Once we have list, try to convert the list to a numpy array
            # If it fails, it means that multiple samples with a different number
            # of tokens were passed.
            # Since this is a special case, we do all computation here.
            try:
                data = np.asarray(data)

            except ValueError:
                # Special case: data has lists inside.
                # We don't put this check before since data[0]
                # can be a list also when a single sample is passed
                # which is automatically handled here, since
                # it would not raise an error when converting
                # to numpy array.

                if isinstance(data[0], list):
                    # Convert each list to a numpy array
                    data = [np.asarray(d) for d in data]

                # Return mean over tokens for each sample
                return np.asarray([d.mean(axis=1) for d in data]).reshape(
                    -1, data[0].shape[-1]
                )

        elif isinstance(data, torch.Tensor):
            # If data is a torch.Tensor, convert it to a numpy array.
            data = data.detach().cpu().numpy()

        elif not isinstance(data, np.ndarray):
            raise ValueError(
                "Unsupported data type. Expected list, torch.Tensor, or numpy.ndarray."
                f" Got {type(data)} instead."
            )

        match data.ndim:
            case 2:
                # Usually we end up here for images feature-extraction, since
                # they have no tokens. Just return data
                return data
            case 3:
                # Take mean over the second-to-last dimension
                return data.mean(axis=1).reshape(-1, 1)

            case 4:
                # Take mean over the second-to-last dimension and reshape
                # so that each sample is a column vector
                return np.mean(data, axis=2).reshape(-1, data.shape[0])
            case _:
                raise ValueError(
                    "Shape mismatch detected: expected data to have 3 or 4 dimensions, "
                    f"but got {data.ndim} dimensions."
                )

    def fit_detector(self, *args, **kwargs) -> None:
        """
        Fit the drift detector on some reference data.
        It extracts features using the pipeline and fits
        the drift detector on those features.
        Notice that this method can also not be called,
        for instance if the drift detector is already fitted.

        Args:
            Any argument is passed to the pipeline.
        """

        ref_features = self._pipeline(*args, **kwargs)

        self._drift_detector.fit(
            self._to_numpy(ref_features),
        )

    def __call__(self, *args, **kwargs):
        """
        Call the pipeline and detect drift.

        Returns:
            The features extracted by the pipeline.
        """
        # Extract features using the pipeline
        features = self._pipeline(*args, **kwargs)

        # Check for drift using the drift detector
        self._drift_detector.detect(
            self._to_numpy(features),
        )

        return features
