class MMEError(Exception):
    """Base exception for expected application errors."""


class RateLimitError(MMEError):
    """Raised when the embedding provider rejects a request due to quota."""