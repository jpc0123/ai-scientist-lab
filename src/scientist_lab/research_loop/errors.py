"""Errors for the real research loop (v2.1.1)."""

from __future__ import annotations


class RealLoopError(RuntimeError):
    """Base error for real research loop operations."""


class RealLoopValidationError(RealLoopError, ValueError):
    """Raised when create/check inputs are invalid (offline)."""


class RealLoopNotFoundError(RealLoopError, KeyError):
    """Raised when a session or round cannot be found."""


class InvalidRealLoopTransition(RealLoopError, ValueError):
    """Raised when a session status transition is not allowed."""


class RealLoopProfileNotQualifiedError(RealLoopError, ValueError):
    """Profile missing or not eligible for a real loop (later gates)."""


class RealLoopProviderMismatchError(RealLoopError, ValueError):
    """Requested provider does not match the actual provider."""


class RealLoopBudgetMissingError(RealLoopError, ValueError):
    """Real-loop budget is not configured."""


class RealLoopFallbackDetectedError(RealLoopError, ValueError):
    """Silent mock/replay fallback was detected under real_only mode."""
