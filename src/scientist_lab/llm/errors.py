"""Unified LLM error hierarchy (v1.4.2+).

Exception messages must never include API keys, Authorization headers,
or full sensitive request headers. Prefer redact_secrets before raising.
"""

from __future__ import annotations


class LLMError(Exception):
    retryable: bool = False

    def __init__(self, message: str = "", *, retryable: bool | None = None) -> None:
        if retryable is not None:
            self.retryable = retryable
        super().__init__(message)


class LLMConfigurationError(LLMError):
    pass


class MissingAPIKeyError(LLMConfigurationError):
    pass


class MissingBaseURLError(LLMConfigurationError):
    pass


class MissingModelError(LLMConfigurationError):
    pass


class RealProviderNotEnabledError(LLMConfigurationError):
    pass


class UnsupportedProviderError(LLMConfigurationError):
    pass


class UnsupportedAPIModeError(LLMConfigurationError):
    pass


class InvalidLLMConfigError(LLMConfigurationError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMPermissionError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


class LLMServerError(LLMError):
    retryable = True


class LLMInvalidRequestError(LLMError):
    pass


class LLMEmptyResponseError(LLMError):
    pass


class StructuredOutputValidationError(LLMError):
    def __init__(
        self,
        message: str = "",
        *,
        issues: list[str] | None = None,
    ) -> None:
        self.issues = list(issues or [])
        super().__init__(message or "; ".join(self.issues) or "structured output invalid")


class LLMBudgetExceededError(LLMError):
    pass


class LLMConcurrencyLimitError(LLMError):
    retryable = True


class LLMReplayWriteError(LLMError):
    pass
