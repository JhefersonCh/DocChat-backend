"""Domain errors: one hierarchy, every error carries a stable `code` and a user-facing message."""


class DomainError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class NotFound(DomainError):
    """The requested entity does not exist."""


class Conflict(DomainError):
    """The request is valid but clashes with the current state."""


class InvalidInput(DomainError):
    """The request violates a business rule or has an invalid shape."""


class ServiceUnavailable(DomainError):
    """A required infrastructure dependency is down."""


class ProviderError(DomainError):
    """A provider call failed (network, HTTP error, invalid response)."""

    def __init__(self, message: str, code: str = "provider_unreachable"):
        super().__init__(code, message)


class EmptyCompletion(ProviderError):
    """The model answered with no `content` (a reasoning model spent its whole token budget
    thinking). Retryable: asking again, or with a larger budget, usually works."""

    def __init__(self, message: str):
        super().__init__(message, "provider_empty_response")


class StreamingUnsupported(ProviderError):
    """The provider cannot stream; caller should fall back to complete(). Internal."""


class ToolsUnsupported(ProviderError):
    """The provider rejected the tools / tool_choice parameters. Internal."""
