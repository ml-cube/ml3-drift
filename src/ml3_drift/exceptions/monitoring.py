class NotFittedError(Exception):
    """Exception raised when detect(X) is called before fit(X) resulting
    in using the monitoring algorithm before it is ready."""
