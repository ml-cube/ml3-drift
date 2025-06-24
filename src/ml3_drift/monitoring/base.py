from dataclasses import dataclass
from typing import Self
import numpy as np


@dataclass
class MonitoringOutput:
    drift: bool


class MonitoringAlgorithm:
    pass

    def __init__(self, reference_size: int, comparison_size: int) -> None:
        pass

    def reset(self, X: np.ndarray | None) -> Self:
        return self

    def update(self, X: np.ndarray) -> MonitoringOutput:
        pass
