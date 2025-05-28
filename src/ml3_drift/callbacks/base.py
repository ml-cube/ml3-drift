from logging import Logger
from ml3_drift.callbacks.models import DriftInfo


def logger_callback(
    drift_info: DriftInfo,
    logger: Logger,
    level: int,
) -> None:
    """
    Logger callback emits a log message with specified level

    Example
    -------

    from functools import partial
    import logging

    callback = partial(logger_callback, logger=logging.getLogger("drift_callback"), level=logging.INFO)
    """
    msg = f"Drift detected on feature at index {drift_info.feature_index} by drift detector {drift_info.drift_detector}."

    if drift_info.p_value is not None:
        msg += f"\n p-value = {drift_info.p_value}"

    if drift_info.threshold is not None:
        msg += f"\n Threshold = {drift_info.threshold}"

    logger.log(level, msg)
