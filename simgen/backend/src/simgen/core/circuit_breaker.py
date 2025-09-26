"""
Simple Circuit Breaker implementation for external service calls.
No external dependencies, just clean Python code.
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Simple configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying again
    expected_exception: type = Exception  # What exceptions to catch


class CircuitBreaker(Generic[T]):
    """
    Simple circuit breaker to prevent cascading failures.

    Usage:
        breaker = CircuitBreaker(config)

        @breaker
        async def call_external_api():
            return await external_api.call()
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN (failures: {self.failure_count})")

        # Try to execute the function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try again."""
        if not self.last_failure_time:
            return False
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info("Circuit breaker recovered, entering CLOSED state")
        else:
            self.success_count += 1

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

    def reset(self):
        """Manually reset the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time
        }


# Pre-configured circuit breakers for different services
llm_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=Exception
    )
)

cv_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=Exception
    )
)

database_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=10,
        expected_exception=Exception
    )
)