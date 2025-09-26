"""
Simple resource management for preventing leaks.
Ensures proper cleanup of connections, file handles, and runtime resources.
"""

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List, Any, Optional, Set
import gc

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Centralized resource management to prevent leaks.

    Tracks and cleans up:
    - WebSocket connections
    - Database sessions
    - MuJoCo runtime instances
    - File handles
    - Redis connections
    """

    def __init__(self):
        self._resources: Dict[str, List[Any]] = {
            "websockets": [],
            "db_sessions": [],
            "mujoco_runtimes": [],
            "file_handles": [],
            "redis_connections": [],
            "cv_pipelines": []
        }
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._weak_refs: Set[weakref.ref] = set()

    def register(self, resource_type: str, resource: Any, cleanup_func=None):
        """
        Register a resource for tracking.

        Args:
            resource_type: Type of resource (websocket, db_session, etc.)
            resource: The resource object to track
            cleanup_func: Optional cleanup function to call
        """
        if resource_type not in self._resources:
            self._resources[resource_type] = []

        # Store weak reference to detect when resource is garbage collected
        if cleanup_func:
            weak_ref = weakref.ref(resource, lambda ref: self._on_resource_deleted(resource_type, ref))
            self._weak_refs.add(weak_ref)

        self._resources[resource_type].append({
            "resource": resource,
            "cleanup": cleanup_func
        })

        logger.debug(f"Registered {resource_type} resource: {id(resource)}")

    def unregister(self, resource_type: str, resource: Any):
        """Unregister a resource after proper cleanup."""
        if resource_type in self._resources:
            self._resources[resource_type] = [
                r for r in self._resources[resource_type]
                if r["resource"] is not resource
            ]

    def _on_resource_deleted(self, resource_type: str, ref):
        """Called when a resource is garbage collected without cleanup."""
        logger.warning(f"Resource {resource_type} was garbage collected without proper cleanup!")

    async def cleanup_resource(self, resource_type: str, resource: Any):
        """Clean up a specific resource."""
        for res in self._resources.get(resource_type, []):
            if res["resource"] is resource:
                if res["cleanup"]:
                    try:
                        if asyncio.iscoroutinefunction(res["cleanup"]):
                            await res["cleanup"](resource)
                        else:
                            res["cleanup"](resource)
                        logger.debug(f"Cleaned up {resource_type} resource: {id(resource)}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {resource_type}: {e}")

                self.unregister(resource_type, resource)
                break

    async def cleanup_all(self, resource_type: Optional[str] = None):
        """
        Clean up all resources of a specific type or all resources.

        Args:
            resource_type: Optional type to clean up, or None for all
        """
        types_to_cleanup = [resource_type] if resource_type else list(self._resources.keys())

        for rtype in types_to_cleanup:
            resources = self._resources.get(rtype, []).copy()
            for res in resources:
                await self.cleanup_resource(rtype, res["resource"])

        # Force garbage collection
        gc.collect()

        logger.info(f"Cleaned up resources: {types_to_cleanup}")

    async def periodic_cleanup(self, interval: int = 300):
        """
        Periodically check for and clean up leaked resources.

        Args:
            interval: Seconds between cleanup checks (default: 5 minutes)
        """
        while True:
            try:
                await asyncio.sleep(interval)

                # Check for leaked resources
                for resource_type, resources in self._resources.items():
                    active_count = len(resources)
                    if active_count > 0:
                        logger.info(f"Active {resource_type}: {active_count}")

                        # Check for stale resources (e.g., old WebSocket connections)
                        for res in resources.copy():
                            if self._is_stale(resource_type, res["resource"]):
                                logger.warning(f"Found stale {resource_type}, cleaning up")
                                await self.cleanup_resource(resource_type, res["resource"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def _is_stale(self, resource_type: str, resource: Any) -> bool:
        """Check if a resource is stale and should be cleaned up."""
        # Implement specific staleness checks for different resource types
        if resource_type == "websockets":
            # Check if WebSocket is still connected
            return getattr(resource, "client_state", None) == "disconnected"
        elif resource_type == "db_sessions":
            # Check if database session is still valid
            return not getattr(resource, "is_active", True)
        elif resource_type == "mujoco_runtimes":
            # Check if runtime has been idle too long
            return getattr(resource, "idle_time", 0) > 600  # 10 minutes

        return False

    def get_stats(self) -> Dict[str, int]:
        """Get current resource statistics."""
        return {
            resource_type: len(resources)
            for resource_type, resources in self._resources.items()
        }


# Global resource manager instance
resource_manager = ResourceManager()


class ManagedResource:
    """
    Context manager for automatic resource registration and cleanup.

    Usage:
        async with ManagedResource(websocket, "websockets", cleanup_func):
            # Use websocket
            pass
        # Automatically cleaned up
    """

    def __init__(self, resource: Any, resource_type: str, cleanup_func=None):
        self.resource = resource
        self.resource_type = resource_type
        self.cleanup_func = cleanup_func

    async def __aenter__(self):
        resource_manager.register(
            self.resource_type,
            self.resource,
            self.cleanup_func
        )
        return self.resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await resource_manager.cleanup_resource(
            self.resource_type,
            self.resource
        )


@asynccontextmanager
async def managed_websocket(websocket):
    """Context manager for WebSocket connections."""
    async def cleanup(ws):
        try:
            await ws.close()
        except Exception:
            pass

    async with ManagedResource(websocket, "websockets", cleanup):
        yield websocket


@asynccontextmanager
async def managed_db_session(session):
    """Context manager for database sessions."""
    async def cleanup(sess):
        try:
            await sess.close()
        except Exception:
            pass

    async with ManagedResource(session, "db_sessions", cleanup):
        yield session


@asynccontextmanager
async def managed_mujoco_runtime(runtime):
    """Context manager for MuJoCo runtime instances."""
    async def cleanup(rt):
        try:
            if hasattr(rt, 'cleanup'):
                await rt.cleanup()
            if hasattr(rt, 'close'):
                rt.close()
        except Exception:
            pass

    async with ManagedResource(runtime, "mujoco_runtimes", cleanup):
        yield runtime


def track_resources(func):
    """
    Decorator to automatically track resources in a function.

    Usage:
        @track_resources
        async def process_sketch(image_data):
            # Resources automatically tracked
            pass
    """
    async def wrapper(*args, **kwargs):
        initial_stats = resource_manager.get_stats()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            final_stats = resource_manager.get_stats()

            # Check for resource leaks
            for rtype in initial_stats:
                if final_stats.get(rtype, 0) > initial_stats.get(rtype, 0):
                    logger.warning(
                        f"Potential resource leak in {func.__name__}: "
                        f"{rtype} increased from {initial_stats[rtype]} to {final_stats[rtype]}"
                    )

    return wrapper