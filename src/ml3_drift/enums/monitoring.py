from enum import Enum


class DataDimension(Enum):
    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"


class DataType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIX = "mix"


class MonitoringType(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    ONLINE_ERROR_BASED = "online_error_based"