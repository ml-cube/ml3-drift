from collections.abc import Callable
from logging import getLogger
from scipy import stats
from ml3_drift.callbacks.models import DriftInfo
from ml3_drift.huggingface.base import BaseDriftDetector


logger = getLogger(__name__)


class KSDriftDetector(BaseDriftDetector):
    """
    Drift detector based on the Kolmogorov-Smirnov test.
    """

    def __init__(
        self,
        p_value: float = 0.005,
        callbacks: list[Callable[[DriftInfo], None]] | None = None,
    ):
        """
        Parameters
        ----------
        p_value: float
            p-value threshold for detecting drift. Default is 0.005.
        """
        super().__init__(callbacks=callbacks)
        self.p_value = p_value

    def _fit(self, X):
        """
        Fit method. Saves reference data.
        """
        self.X_ref_ = X

    def _detect(self, X):
        """
        Core method for detecting drift
        """

        # list containing drift info events
        drift_info_list = []

        for i in range(X.shape[1]):
            _, p_value = stats.ks_2samp(
                self.X_ref_[:, i],
                X[:, i],
            )

            if p_value < self.p_value:
                drift_info_list.append(
                    DriftInfo(
                        feature_index=i,
                        drift_detector=self.__class__.__name__,
                        p_value=p_value,
                        threshold=self.p_value,
                    )
                )

        return drift_info_list
