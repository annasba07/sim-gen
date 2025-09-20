"""
Production-grade resilience and error handling for SimGen AI
Implements circuit breakers, retry logic, graceful degradation, and comprehensive error handling
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
from functools import wraps
import json

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close
    timeout: int = 30          # Request timeout in seconds


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ErrorMetrics:
    """Error tracking metrics."""
    total_requests: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)


class SimGenError(Exception):
    """Base exception for SimGen errors."""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SIMGEN_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now()


class AIServiceError(SimGenError):
    """AI service related errors."""
    pass


class RenderingError(SimGenError):
    """Rendering related errors."""
    pass


class ValidationError(SimGenError):
    """Input validation errors."""
    pass


class RateLimitError(SimGenError):
    """Rate limiting errors."""
    pass


class CircuitBreakerOpenError(SimGenError):
    """Circuit breaker is open."""
    pass


class CircuitBreaker:
    """Production-grade circuit breaker for external service calls."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = ErrorMetrics()
        self.last_failure_time = None
        self.state_change_time = datetime.now()
        
        logger.info(f"Circuit breaker '{name}' initialized: {self.config}")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit breaker '{self.name}' attempting reset (half-open)")
                self.state = CircuitState.HALF_OPEN
                self.state_change_time = datetime.now()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    "CIRCUIT_BREAKER_OPEN",
                    {"state": self.state.value, "failure_count": self.metrics.consecutive_failures}
                )
        
        start_time = time.time()
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
            
        except asyncio.TimeoutError as e:
            self._record_failure(f"Timeout after {self.config.timeout}s")
            raise AIServiceError(
                f"Service call timed out after {self.config.timeout}s",
                "SERVICE_TIMEOUT",
                {"service": self.name, "timeout": self.config.timeout}
            )
        except Exception as e:
            self._record_failure(str(e))
            
            # Re-raise as appropriate SimGen error
            if isinstance(e, SimGenError):
                raise
            else:
                raise AIServiceError(
                    f"Service call failed: {str(e)}",
                    "SERVICE_ERROR",
                    {"service": self.name, "original_error": str(e)}
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _record_success(self, response_time: float):
        """Record successful call."""
        self.metrics.total_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = datetime.now()
        
        # Track response times (keep last 100)
        self.metrics.response_times.append(response_time)
        if len(self.metrics.response_times) > 100:
            self.metrics.response_times.pop(0)
        
        self.metrics.avg_response_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
        self.metrics.failure_rate = self.metrics.total_failures / self.metrics.total_requests
        
        # State transitions
        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                logger.info(f"Circuit breaker '{self.name}' closing (service recovered)")
                self.state = CircuitState.CLOSED
                self.state_change_time = datetime.now()
    
    def _record_failure(self, error_message: str):
        """Record failed call."""
        self.metrics.total_requests += 1
        self.metrics.total_failures += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = datetime.now()
        self.last_failure_time = datetime.now()
        
        self.metrics.failure_rate = self.metrics.total_failures / self.metrics.total_requests
        
        # State transitions
        if self.state == CircuitState.CLOSED:
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                logger.warning(f"Circuit breaker '{self.name}' opening (threshold reached)")
                self.state = CircuitState.OPEN
                self.state_change_time = datetime.now()
        
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker '{self.name}' re-opening (half-open test failed)")
            self.state = CircuitState.OPEN
            self.state_change_time = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_requests": self.metrics.total_requests,
            "total_failures": self.metrics.total_failures,
            "consecutive_failures": self.metrics.consecutive_failures,
            "failure_rate": round(self.metrics.failure_rate * 100, 2),
            "avg_response_time": round(self.metrics.avg_response_time * 1000, 2),  # ms
            "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "state_duration": (datetime.now() - self.state_change_time).total_seconds()
        }


class RetryHandler:
    """Intelligent retry handler with exponential backoff."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_on: tuple = (Exception,),
        skip_retry_on: tuple = (ValidationError, RateLimitError),
        **kwargs
    ):
        """Execute function with intelligent retry logic."""
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
                
            except skip_retry_on as e:
                # Don't retry these errors
                logger.debug(f"Skipping retry for {type(e).__name__}: {e}")
                raise
                
            except retry_on as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed: {e}")
        
        # All attempts failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # ±50% jitter
        
        return delay


class ResilienceManager:
    """Central manager for all resilience patterns."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler()
        self.error_counts: Dict[str, int] = {}
        
        # Initialize circuit breakers for critical services
        self._initialize_circuit_breakers()
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all external services."""
        
        # LLM Services
        self.circuit_breakers["anthropic"] = CircuitBreaker(
            "anthropic",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
        )
        
        self.circuit_breakers["openai"] = CircuitBreaker(
            "openai", 
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
        )
        
        # Database
        self.circuit_breakers["database"] = CircuitBreaker(
            "database",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=10)
        )
        
        # Redis Cache
        self.circuit_breakers["redis"] = CircuitBreaker(
            "redis",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=15)
        )
        
        # MuJoCo Rendering
        self.circuit_breakers["mujoco"] = CircuitBreaker(
            "mujoco",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=20)
        )
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            # Create default circuit breaker
            self.circuit_breakers[service_name] = CircuitBreaker(service_name)
        
        return self.circuit_breakers[service_name]
    
    async def call_with_resilience(
        self,
        service_name: str,
        func: Callable,
        *args,
        use_circuit_breaker: bool = True,
        use_retry: bool = True,
        **kwargs
    ):
        """Call function with full resilience protection."""
        
        async def _execute():
            if use_circuit_breaker:
                circuit_breaker = self.get_circuit_breaker(service_name)
                return await circuit_breaker.call(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        if use_retry:
            return await self.retry_handler.execute_with_retry(_execute)
        else:
            return await _execute()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        circuit_metrics = {
            name: breaker.get_metrics()
            for name, breaker in self.circuit_breakers.items()
        }
        
        # Calculate overall health score
        total_circuits = len(circuit_metrics)
        healthy_circuits = sum(
            1 for metrics in circuit_metrics.values()
            if metrics["state"] == "closed"
        )
        
        health_score = (healthy_circuits / total_circuits) * 100 if total_circuits > 0 else 100
        
        return {
            "health_score": round(health_score, 1),
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": circuit_metrics,
            "error_counts": self.error_counts.copy(),
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "unhealthy"
        }
    
    def record_error(self, error_type: str):
        """Record error for monitoring."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1


# Global resilience manager
_resilience_manager = None

def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


# Decorators for easy resilience integration
def resilient_service(service_name: str, use_circuit_breaker: bool = True, use_retry: bool = True):
    """Decorator to add resilience to service calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            try:
                return await manager.call_with_resilience(
                    service_name, func, *args,
                    use_circuit_breaker=use_circuit_breaker,
                    use_retry=use_retry,
                    **kwargs
                )
            except Exception as e:
                # Record error for monitoring
                error_type = type(e).__name__
                manager.record_error(error_type)
                
                # Log error with context
                logger.error(f"Resilient service call failed: {service_name} - {error_type}: {e}")
                raise
        
        return wrapper
    return decorator


def handle_errors(error_mappings: Dict[type, str] = None):
    """Decorator to standardize error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except SimGenError:
                # Already a SimGen error, re-raise
                raise
            except Exception as e:
                # Map to appropriate SimGen error
                error_code = "UNKNOWN_ERROR"
                if error_mappings:
                    for error_type, code in error_mappings.items():
                        if isinstance(e, error_type):
                            error_code = code
                            break
                
                # Log error with full traceback
                logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
                
                raise SimGenError(
                    f"Unexpected error: {str(e)}",
                    error_code,
                    {"function": func.__name__, "traceback": traceback.format_exc()}
                )
        
        return wrapper
    return decorator