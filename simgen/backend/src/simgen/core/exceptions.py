"""
Custom exception hierarchy for SimGen application.
Provides specific exception types for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class SimGenException(Exception):
    """Base exception for all SimGen-specific errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ValidationError(SimGenException):
    """Raised when input validation fails."""
    pass


class CVPipelineError(SimGenException):
    """Raised when computer vision pipeline fails."""
    pass


class SketchAnalysisError(SimGenException):
    """Raised when sketch analysis fails."""
    pass


class PhysicsSpecError(SimGenException):
    """Raised when PhysicsSpec generation or validation fails."""
    pass


class MJCFCompilationError(SimGenException):
    """Raised when MJCF compilation fails."""
    pass


class SimulationError(SimGenException):
    """Raised when simulation execution fails."""
    pass


class LLMError(SimGenException):
    """Raised when LLM API calls fail."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.provider = provider
        self.model = model
        if provider:
            self.details["provider"] = provider
        if model:
            self.details["model"] = model


class DatabaseError(SimGenException):
    """Raised when database operations fail."""
    pass


class ConnectionPoolError(DatabaseError):
    """Raised when connection pool operations fail."""
    pass


class ResourceExhaustedError(SimGenException):
    """Raised when system resources are exhausted."""
    pass


class RateLimitError(SimGenException):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class StreamingError(SimGenException):
    """Raised when streaming operations fail."""
    pass


class WebSocketError(StreamingError):
    """Raised when WebSocket operations fail."""
    pass


class TimeoutError(SimGenException):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        if operation:
            self.details["operation"] = operation
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds