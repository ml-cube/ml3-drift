import dataclasses
from logging import Logger

from ml3_drift.models.monitoring import DriftInfo


def logger_callback(
    drift_info: DriftInfo | None,
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

    if drift_info is None:
        logger.log(level, "Drift Detected, no drift info provided!")
        return

    logger.log(level, f"Drift Detected, drift info: {dataclasses.asdict(drift_info)}")


def print_callback(drift_info: DriftInfo | None) -> None:
    """
    Print callback prints a message to the console when drift is detected.
    It should be used only for testing purposes.

    Example
    -------

    callback = print_callback
    """
    if drift_info is None:
        print("Drift Detected, no drift info provided!")
        return

    print(f"Drift Detected, drift info: {dataclasses.asdict(drift_info)}")
