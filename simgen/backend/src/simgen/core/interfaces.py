"""
Interface definitions for dependency injection.
Uses Python's Protocol for structural typing without runtime overhead.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime

from ..models import PhysicsSpec, CVResult


@runtime_checkable
class IPhysicsCompiler(Protocol):
    """Interface for physics compilation services."""

    async def compile(self, spec: PhysicsSpec) -> str:
        """Compile physics spec to MJCF XML."""
        ...

    async def validate(self, mjcf_xml: str) -> bool:
        """Validate MJCF XML structure."""
        ...


@runtime_checkable
class IPhysicsRuntime(Protocol):
    """Interface for physics simulation runtime."""

    async def load(self, mjcf_xml: str) -> None:
        """Load MJCF model into simulator."""
        ...

    async def step(self, count: int = 1) -> None:
        """Step the simulation forward."""
        ...

    async def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class ISketchAnalyzer(Protocol):
    """Interface for sketch analysis services."""

    async def analyze(
        self,
        image_data: bytes,
        user_text: Optional[str] = None,
        timeout: float = 30.0
    ) -> CVResult:
        """Analyze sketch and extract physics information."""
        ...

    def get_confidence(self) -> float:
        """Get confidence score of last analysis."""
        ...


@runtime_checkable
class IComputerVisionPipeline(Protocol):
    """Interface for computer vision processing."""

    async def analyze_sketch(self, image_data: bytes) -> Any:
        """Analyze sketch and return simplified CV result."""
        ...

    async def initialize(self) -> None:
        """Initialize heavy models asynchronously."""
        ...

    def get_confidence(self) -> float:
        """Get confidence score of last analysis."""
        ...


@runtime_checkable
class ILLMClient(Protocol):
    """Interface for LLM services."""

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7
    ) -> str:
        """Generate text from prompt."""
        ...

    async def enhance_physics_spec(
        self,
        spec: PhysicsSpec,
        context: Optional[str] = None
    ) -> PhysicsSpec:
        """Enhance physics spec with LLM understanding."""
        ...


@runtime_checkable
class ICacheService(Protocol):
    """Interface for caching services."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        ...


@runtime_checkable
class IWebSocketManager(Protocol):
    """Interface for WebSocket management."""

    async def connect(self, websocket: Any, session_id: str) -> None:
        """Register WebSocket connection."""
        ...

    async def disconnect(self, session_id: str) -> None:
        """Unregister WebSocket connection."""
        ...

    async def send(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific session."""
        ...

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[List[str]] = None
    ) -> int:
        """Broadcast message to all connected sessions."""
        ...

    async def get_session_count(self) -> int:
        """Get count of active sessions."""
        ...


@runtime_checkable
class IStreamingProtocol(Protocol):
    """Interface for streaming protocol services."""

    def encode_frame(self, frame_data: Dict[str, Any]) -> bytes:
        """Encode frame data to binary format."""
        ...

    def decode_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode binary data to frame."""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """Get encoding/decoding statistics."""
        ...

    def cleanup_buffers(self) -> None:
        """Clean up internal buffers to prevent memory leaks."""
        ...


@runtime_checkable
class IDatabaseRepository(Protocol):
    """Base interface for database repositories."""

    async def create(self, entity: Any) -> Any:
        """Create entity in database."""
        ...

    async def read(self, entity_id: str) -> Optional[Any]:
        """Read entity by ID."""
        ...

    async def update(self, entity_id: str, data: Dict) -> bool:
        """Update entity."""
        ...

    async def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        ...

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict] = None
    ) -> List[Any]:
        """List entities with optional filters."""
        ...


@runtime_checkable
class IResourceManager(Protocol):
    """Interface for resource management."""

    async def acquire(self, resource_type: str, resource_id: str) -> Any:
        """Acquire a resource."""
        ...

    async def release(self, resource_id: str) -> bool:
        """Release a resource."""
        ...

    async def cleanup_expired(self) -> int:
        """Clean up expired resources."""
        ...

    def get_active_count(self) -> int:
        """Get count of active resources."""
        ...


@runtime_checkable
class ICircuitBreaker(Protocol):
    """Interface for circuit breaker pattern."""

    async def call(self, func: Any, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        ...

    def get_state(self) -> str:
        """Get circuit breaker state (CLOSED, OPEN, HALF_OPEN)."""
        ...

    def reset(self) -> None:
        """Reset circuit breaker state."""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        ...


@runtime_checkable
class IRateLimiter(Protocol):
    """Interface for rate limiting."""

    async def check(self, identifier: str) -> bool:
        """Check if request is allowed."""
        ...

    async def increment(self, identifier: str) -> int:
        """Increment request count."""
        ...

    async def reset(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        ...

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests allowed."""
        ...


@runtime_checkable
class IMonitoringService(Protocol):
    """Interface for monitoring and metrics."""

    def increment_counter(self, metric: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        ...

    def gauge(self, metric: str, value: float) -> None:
        """Set a gauge metric."""
        ...

    def histogram(self, metric: str, value: float) -> None:
        """Record a histogram value."""
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        ...


@runtime_checkable
class IHealthChecker(Protocol):
    """Interface for health checking."""

    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        ...

    async def check_dependencies(self) -> Dict[str, bool]:
        """Check health of dependencies."""
        ...

    async def get_readiness(self) -> bool:
        """Check if service is ready to serve traffic."""
        ...