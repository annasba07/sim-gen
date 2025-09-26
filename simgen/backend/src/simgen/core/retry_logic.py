"""
Robust Retry Logic with Exponential Backoff
Improves system reliability when dealing with external APIs and transient failures.

Addresses the "missing retry logic" identified in the systems architect review.
Provides intelligent retries for LLM calls, OCR operations, and database connections.
"""

import asyncio
import logging
import random
import time
from typing import Callable, TypeVar, Optional, Type, Union, List, Any
from dataclasses import dataclass
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Different retry strategies for different types of operations"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class RetryableError(Exception):
    """Base class for errors that should trigger retries"""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should NOT trigger retries"""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = (NonRetryableError,)


@dataclass
class RetryResult:
    """Result of retry operation including metadata"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts_made: int = 0
    total_time: float = 0.0
    attempt_details: List[dict] = None


class RetryHandler:
    """
    Intelligent retry handler with multiple strategies.

    Features:
    - Exponential backoff with jitter
    - Configurable retry strategies
    - Exception classification (retryable vs non-retryable)
    - Detailed retry metrics
    - Circuit breaker integration
    """

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()

    async def execute_with_retry(
        self,
        operation: Callable[..., T],
        *args,
        operation_name: str = "unknown",
        **kwargs
    ) -> RetryResult:
        """
        Execute operation with intelligent retry logic.

        Args:
            operation: Async function to execute
            operation_name: Name for logging/metrics
            *args, **kwargs: Arguments for the operation

        Returns:
            RetryResult with success status and metadata
        """
        start_time = time.time()
        attempt_details = []
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()

            try:
                logger.debug(f"Attempt {attempt}/{self.config.max_attempts} for {operation_name}")

                # Execute the operation
                result = await operation(*args, **kwargs)

                # Success!
                attempt_time = time.time() - attempt_start
                total_time = time.time() - start_time

                attempt_details.append({
                    "attempt": attempt,
                    "success": True,
                    "duration": attempt_time,
                    "error": None
                })

                logger.info(f"{operation_name} succeeded on attempt {attempt}")

                return RetryResult(
                    success=True,
                    result=result,
                    attempts_made=attempt,
                    total_time=total_time,
                    attempt_details=attempt_details
                )

            except Exception as e:
                last_exception = e
                attempt_time = time.time() - attempt_start

                attempt_details.append({
                    "attempt": attempt,
                    "success": False,
                    "duration": attempt_time,
                    "error": str(e),
                    "error_type": type(e).__name__
                })

                # Check if this is a non-retryable error
                if self._is_non_retryable(e):
                    logger.warning(f"{operation_name} failed with non-retryable error: {e}")
                    break

                # Check if this is retryable
                if not self._is_retryable(e):
                    logger.warning(f"{operation_name} failed with non-retryable error type: {type(e).__name__}")
                    break

                # Don't wait after the last attempt
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"{operation_name} attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {attempt} attempts: {e}")

        # All attempts failed
        total_time = time.time() - start_time

        return RetryResult(
            success=False,
            error=last_exception,
            attempts_made=self.config.max_attempts,
            total_time=total_time,
            attempt_details=attempt_details
        )

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception type is retryable"""
        return isinstance(exception, self.config.retryable_exceptions)

    def _is_non_retryable(self, exception: Exception) -> bool:
        """Check if exception type is explicitly non-retryable"""
        return isinstance(exception, self.config.non_retryable_exceptions)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next attempt based on strategy"""
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0

        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))

        else:
            delay = self.config.base_delay

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter and delay > 0:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)


# Predefined retry configurations for common use cases
class RetryConfigs:
    """Predefined retry configurations for different types of operations"""

    # For LLM API calls (network-dependent, can be slow)
    LLM_API = RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            # Add specific LLM provider exceptions here
        ),
        non_retryable_exceptions=(
            ValueError,  # Bad input
            KeyError,    # Missing API key
            PermissionError,  # Auth issues
        )
    )

    # For OCR operations (CPU-bound, can fail on bad images)
    OCR_PROCESSING = RetryConfig(
        max_attempts=2,
        base_delay=1.0,
        max_delay=10.0,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        retryable_exceptions=(
            RuntimeError,  # OCR processing errors
            MemoryError,   # Out of memory
            OSError,       # File/system errors
        ),
        non_retryable_exceptions=(
            ValueError,    # Bad image format
            TypeError,     # Wrong input type
        )
    )

    # For database operations (transient connection issues)
    DATABASE = RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=10.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            # Add database-specific exceptions
        ),
        non_retryable_exceptions=(
            ValueError,    # SQL syntax errors
            PermissionError,  # Auth errors
        )
    )

    # For external API calls (network-dependent)
    EXTERNAL_API = RetryConfig(
        max_attempts=4,
        base_delay=1.0,
        max_delay=20.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
        non_retryable_exceptions=(
            ValueError,
            KeyError,
            PermissionError,
        )
    )

    # For file operations (I/O errors)
    FILE_OPERATIONS = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=2.0,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        retryable_exceptions=(
            OSError,
            IOError,
            FileNotFoundError,
        ),
        non_retryable_exceptions=(
            PermissionError,
            IsADirectoryError,
            ValueError,
        )
    )


# Convenience decorators for common retry patterns
def retry_llm_call(operation_name: str = None):
    """Decorator for LLM API calls with appropriate retry logic"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            handler = RetryHandler(RetryConfigs.LLM_API)
            result = await handler.execute_with_retry(
                func, *args,
                operation_name=operation_name or func.__name__,
                **kwargs
            )
            if result.success:
                return result.result
            else:
                raise result.error
        return wrapper
    return decorator


def retry_ocr_operation(operation_name: str = None):
    """Decorator for OCR operations with appropriate retry logic"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            handler = RetryHandler(RetryConfigs.OCR_PROCESSING)
            result = await handler.execute_with_retry(
                func, *args,
                operation_name=operation_name or func.__name__,
                **kwargs
            )
            if result.success:
                return result.result
            else:
                raise result.error
        return wrapper
    return decorator


def retry_db_operation(operation_name: str = None):
    """Decorator for database operations with appropriate retry logic"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            handler = RetryHandler(RetryConfigs.DATABASE)
            result = await handler.execute_with_retry(
                func, *args,
                operation_name=operation_name or func.__name__,
                **kwargs
            )
            if result.success:
                return result.result
            else:
                raise result.error
        return wrapper
    return decorator


# Factory functions for dependency injection
def create_llm_retry_handler() -> RetryHandler:
    """Create retry handler optimized for LLM calls"""
    return RetryHandler(RetryConfigs.LLM_API)


def create_ocr_retry_handler() -> RetryHandler:
    """Create retry handler optimized for OCR operations"""
    return RetryHandler(RetryConfigs.OCR_PROCESSING)


def create_db_retry_handler() -> RetryHandler:
    """Create retry handler optimized for database operations"""
    return RetryHandler(RetryConfigs.DATABASE)


def create_external_api_retry_handler() -> RetryHandler:
    """Create retry handler optimized for external API calls"""
    return RetryHandler(RetryConfigs.EXTERNAL_API)