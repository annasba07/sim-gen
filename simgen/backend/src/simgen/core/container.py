"""
Dependency injection container for service registration and resolution.
Simple, lightweight implementation without external dependencies.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime:
    """Service lifetime options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class DIContainer:
    """
    Simple dependency injection container with support for different lifetimes.

    Features:
    - Singleton services (shared instance)
    - Transient services (new instance per request)
    - Scoped services (per-request scope)
    - Lazy initialization
    - Circular dependency detection
    """

    def __init__(self):
        self._services: Dict[Type, tuple[Union[Type, Callable, Any], str]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._scoped_instances: WeakValueDictionary = WeakValueDictionary()
        self._initialization_stack: list = []

    def register(
        self,
        interface: Type[T],
        implementation: Union[Type[T], Callable[[], T], T],
        lifetime: str = ServiceLifetime.SINGLETON
    ) -> None:
        """
        Register a service in the container.

        Args:
            interface: The interface/protocol type
            implementation: The implementation (class, factory, or instance)
            lifetime: Service lifetime (singleton, transient, or scoped)
        """
        if interface in self._services:
            logger.warning(f"Service {interface.__name__} is being re-registered")

        self._services[interface] = (implementation, lifetime)
        logger.debug(f"Registered {interface.__name__} with lifetime {lifetime}")

    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[['DIContainer'], T],
        lifetime: str = ServiceLifetime.SINGLETON
    ) -> None:
        """
        Register a service using a factory function.

        Args:
            interface: The interface/protocol type
            factory: Factory function that receives the container
            lifetime: Service lifetime
        """
        self._factories[interface] = factory
        self._services[interface] = (factory, lifetime)
        logger.debug(f"Registered factory for {interface.__name__}")

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register an existing instance as a singleton.

        Args:
            interface: The interface/protocol type
            instance: The instance to register
        """
        self._singletons[interface] = instance
        self._services[interface] = (instance, ServiceLifetime.SINGLETON)
        logger.debug(f"Registered instance for {interface.__name__}")

    def get(self, interface: Type[T], scope_id: Optional[str] = None) -> T:
        """
        Resolve a service from the container.

        Args:
            interface: The interface type to resolve
            scope_id: Optional scope identifier for scoped services

        Returns:
            The resolved service instance

        Raises:
            ValueError: If service is not registered
            RuntimeError: If circular dependency is detected
        """
        if interface not in self._services:
            raise ValueError(f"Service {interface.__name__} is not registered")

        # Check for circular dependencies
        if interface in self._initialization_stack:
            cycle = " -> ".join([t.__name__ for t in self._initialization_stack])
            cycle += f" -> {interface.__name__}"
            raise RuntimeError(f"Circular dependency detected: {cycle}")

        implementation, lifetime = self._services[interface]

        try:
            self._initialization_stack.append(interface)

            if lifetime == ServiceLifetime.SINGLETON:
                return self._get_singleton(interface, implementation)
            elif lifetime == ServiceLifetime.TRANSIENT:
                return self._create_instance(implementation)
            elif lifetime == ServiceLifetime.SCOPED:
                return self._get_scoped(interface, implementation, scope_id)
            else:
                raise ValueError(f"Unknown lifetime: {lifetime}")
        finally:
            self._initialization_stack.pop()

    def _get_singleton(self, interface: Type[T], implementation: Any) -> T:
        """Get or create a singleton instance."""
        if interface not in self._singletons:
            if interface in self._factories:
                # Use factory to create instance
                instance = self._factories[interface](self)
            else:
                instance = self._create_instance(implementation)
            self._singletons[interface] = instance
            logger.debug(f"Created singleton for {interface.__name__}")
        return self._singletons[interface]

    def _get_scoped(
        self,
        interface: Type[T],
        implementation: Any,
        scope_id: Optional[str]
    ) -> T:
        """Get or create a scoped instance."""
        if scope_id is None:
            # No scope, treat as transient
            return self._create_instance(implementation)

        scope_key = f"{scope_id}:{interface.__name__}"
        if scope_key not in self._scoped_instances:
            instance = self._create_instance(implementation)
            self._scoped_instances[scope_key] = instance
            logger.debug(f"Created scoped instance for {interface.__name__} in scope {scope_id}")
        return self._scoped_instances[scope_key]

    def _create_instance(self, implementation: Any) -> Any:
        """Create a new instance of the implementation."""
        if callable(implementation) and not isinstance(implementation, type):
            # It's a factory function or already an instance
            if implementation in self._factories.values():
                # It's a registered factory
                return implementation(self)
            return implementation
        elif isinstance(implementation, type):
            # It's a class, instantiate it
            return implementation()
        else:
            # It's already an instance
            return implementation

    def resolve(self, interface: Type[T]) -> T:
        """
        Alias for get() method for compatibility.

        Args:
            interface: The interface type to resolve

        Returns:
            The resolved service instance
        """
        return self.get(interface)

    def has(self, interface: Type) -> bool:
        """
        Check if a service is registered.

        Args:
            interface: The interface type to check

        Returns:
            True if the service is registered
        """
        return interface in self._services

    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()
        self._scoped_instances.clear()
        self._initialization_stack.clear()
        logger.info("Container cleared")

    def clear_singletons(self) -> None:
        """Clear only singleton instances (useful for testing)."""
        self._singletons.clear()
        logger.debug("Singleton instances cleared")

    def clear_scoped(self, scope_id: Optional[str] = None) -> None:
        """
        Clear scoped instances.

        Args:
            scope_id: Optional scope to clear, clears all if not specified
        """
        if scope_id:
            keys_to_remove = [
                key for key in self._scoped_instances
                if key.startswith(f"{scope_id}:")
            ]
            for key in keys_to_remove:
                del self._scoped_instances[key]
            logger.debug(f"Cleared scope {scope_id}")
        else:
            self._scoped_instances.clear()
            logger.debug("All scoped instances cleared")

    async def initialize_async_services(self) -> None:
        """
        Initialize all async services that require startup.
        Call this during application startup.
        """
        for interface, (implementation, lifetime) in self._services.items():
            if lifetime == ServiceLifetime.SINGLETON:
                instance = self.get(interface)
                # Check if instance has an async initialize method
                if hasattr(instance, 'initialize') and asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                    logger.info(f"Initialized async service {interface.__name__}")

    async def cleanup_async_services(self) -> None:
        """
        Cleanup all async services that require shutdown.
        Call this during application shutdown.
        """
        for interface, instance in self._singletons.items():
            # Check if instance has an async cleanup method
            if hasattr(instance, 'cleanup') and asyncio.iscoroutinefunction(instance.cleanup):
                try:
                    await instance.cleanup()
                    logger.info(f"Cleaned up service {interface.__name__}")
                except Exception as e:
                    logger.error(f"Error cleaning up {interface.__name__}: {e}")

    def get_all_registered(self) -> Dict[str, str]:
        """Get all registered services for debugging."""
        return {
            interface.__name__: lifetime
            for interface, (_, lifetime) in self._services.items()
        }


# Global container instance
container = DIContainer()


# Dependency injection helpers for FastAPI
def Depends(interface: Type[T]) -> T:
    """
    FastAPI-style dependency injection.

    Usage:
        @router.post("/endpoint")
        async def endpoint(service: IService = Depends(IService)):
            return await service.do_something()
    """
    def dependency():
        return container.get(interface)
    return dependency


# Factory function helpers
def get_physics_compiler():
    """Get physics compiler from container."""
    from .interfaces import IPhysicsCompiler
    return container.get(IPhysicsCompiler)


def get_sketch_analyzer():
    """Get sketch analyzer from container."""
    from .interfaces import ISketchAnalyzer
    return container.get(ISketchAnalyzer)


def get_llm_client():
    """Get LLM client from container."""
    from .interfaces import ILLMClient
    return container.get(ILLMClient)


def get_cache_service():
    """Get cache service from container."""
    from .interfaces import ICacheService
    return container.get(ICacheService)


def get_websocket_manager():
    """Get WebSocket manager from container."""
    from .interfaces import IWebSocketManager
    return container.get(IWebSocketManager)


def get_streaming_protocol():
    """Get streaming protocol from container."""
    from .interfaces import IStreamingProtocol
    return container.get(IStreamingProtocol)


def get_resource_manager():
    """Get resource manager from container."""
    from .interfaces import IResourceManager
    return container.get(IResourceManager)


def get_circuit_breaker():
    """Get circuit breaker from container."""
    from .interfaces import ICircuitBreaker
    return container.get(ICircuitBreaker)


def get_rate_limiter():
    """Get rate limiter from container."""
    from .interfaces import IRateLimiter
    return container.get(IRateLimiter)


def get_monitoring_service():
    """Get monitoring service from container."""
    from .interfaces import IMonitoringService
    return container.get(IMonitoringService)