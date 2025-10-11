"""
Simple health check endpoints for monitoring and load balancing.
No over-engineered observability platform, just practical health checks.
"""

import asyncio
import time
from typing import Dict, Any, List
from fastapi import APIRouter, Response
from datetime import datetime, timedelta
import psutil
import os

from ..core.config_clean import settings
from ..database.connection_pool import get_connection_pool
from ..services.cache_service import get_cache_service
from ..services.websocket_session_manager import get_websocket_manager
from ..core.resource_manager import resource_manager

router = APIRouter()


class HealthStatus:
    """Simple health status tracking."""

    def __init__(self):
        self.start_time = time.time()
        self.checks_passed = 0
        self.checks_failed = 0
        self.last_check = None
        self.issues: List[str] = []

    def record_check(self, passed: bool, issue: str = None):
        """Record a health check result."""
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
            if issue:
                self.issues.append(f"{datetime.utcnow().isoformat()}: {issue}")
                # Keep only last 10 issues
                self.issues = self.issues[-10:]
        self.last_check = datetime.utcnow()

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time

    def get_health_score(self) -> float:
        """Get overall health score (0-100)."""
        total_checks = self.checks_passed + self.checks_failed
        if total_checks == 0:
            return 100.0
        return (self.checks_passed / total_checks) * 100


health_status = HealthStatus()


@router.get("/health")
async def health_check(response: Response) -> Dict[str, Any]:
    """
    Basic health check endpoint for load balancers.

    Returns 200 if healthy, 503 if unhealthy.
    """
    checks = {}
    is_healthy = True

    # Check database connection
    try:
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            checks["database"] = "ok" if result == 1 else "failed"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        is_healthy = False

    # Check Redis connection
    try:
        cache = await get_cache_service()
        if cache.redis_client:
            await cache.redis_client.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "not configured"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
        is_healthy = False

    # Check system resources
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    checks["memory_percent"] = memory.percent
    checks["disk_percent"] = disk.percent

    # Consider unhealthy if resources are critically low
    if memory.percent > 90:
        checks["memory"] = "critical"
        is_healthy = False
    if disk.percent > 95:
        checks["disk"] = "critical"
        is_healthy = False

    # Record the check
    health_status.record_check(
        is_healthy,
        None if is_healthy else f"Failed checks: {[k for k, v in checks.items() if v != 'ok']}"
    )

    # Set appropriate status code
    if not is_healthy:
        response.status_code = 503  # Service Unavailable

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "uptime_seconds": health_status.get_uptime(),
        "health_score": health_status.get_health_score()
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with more metrics.
    For internal monitoring, not for load balancers.
    """
    # Basic health check first
    basic_health = await health_check(Response())

    # Add detailed metrics
    detailed = {
        **basic_health,
        "server_id": os.getenv("SERVER_ID", "unknown"),
        "version": settings.app_version if hasattr(settings, 'app_version') else "1.0.0",
        "environment": settings.environment if hasattr(settings, 'environment') else "development"
    }

    # Resource manager stats
    try:
        detailed["resources"] = resource_manager.get_stats()
    except:
        detailed["resources"] = {}

    # WebSocket session stats
    try:
        ws_manager = await get_websocket_manager()
        detailed["websocket_sessions"] = await ws_manager.get_server_stats()
    except:
        detailed["websocket_sessions"] = {}

    # Cache statistics
    try:
        cache = await get_cache_service()
        detailed["cache_stats"] = cache.get_stats()
    except:
        detailed["cache_stats"] = {}

    # CPU and memory details
    detailed["system"] = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        }
    }

    # Process information
    process = psutil.Process()
    detailed["process"] = {
        "pid": process.pid,
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "num_threads": process.num_threads(),
        "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
    }

    # Recent issues
    if health_status.issues:
        detailed["recent_issues"] = health_status.issues

    return detailed


@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for container orchestration.
    Returns whether the service is ready to accept traffic.
    """
    # Check if all critical services are initialized
    ready = True
    checks = {}

    # Database pool ready
    try:
        pool = await get_connection_pool()
        checks["database_pool"] = pool is not None
        ready = ready and checks["database_pool"]
    except:
        checks["database_pool"] = False
        ready = False

    # Cache service ready
    try:
        cache = await get_cache_service()
        checks["cache_service"] = cache is not None
        ready = ready and checks["cache_service"]
    except:
        checks["cache_service"] = False
        ready = False

    # WebSocket manager ready (if applicable)
    try:
        if hasattr(settings, 'enable_websockets') and settings.enable_websockets:
            ws_manager = await get_websocket_manager()
            checks["websocket_manager"] = ws_manager is not None
            ready = ready and checks["websocket_manager"]
    except:
        checks["websocket_manager"] = False
        ready = False

    return {
        "ready": ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for container orchestration.
    Simple check to see if the service is alive (not deadlocked).
    """
    # Simple check - if we can respond, we're alive
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": health_status.get_uptime()
    }


@router.post("/health/drain")
async def drain_connections() -> Dict[str, Any]:
    """
    Gracefully drain connections before shutdown.
    Used for rolling updates without dropping connections.
    """
    # This would be called before removing a server from the load balancer

    # Stop accepting new connections
    # (In production, you'd set a flag that the WebSocket handler checks)

    # Wait for existing connections to complete
    ws_manager = await get_websocket_manager()
    initial_sessions = await ws_manager.get_server_stats()

    # Give connections time to finish (max 30 seconds)
    max_wait = 30
    waited = 0

    while waited < max_wait:
        current_sessions = await ws_manager.get_server_stats()
        if current_sessions["local_sessions"] == 0:
            break
        await asyncio.sleep(1)
        waited += 1

    # Clean up remaining connections
    await ws_manager.stop()

    return {
        "drained": True,
        "initial_sessions": initial_sessions["local_sessions"],
        "wait_time": waited,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """
    Simple Prometheus-compatible metrics endpoint.
    No need for complex observability platform.
    """
    # Collect metrics in Prometheus format
    metrics = []

    # Health score
    metrics.append(f"# HELP health_score Overall health score (0-100)")
    metrics.append(f"# TYPE health_score gauge")
    metrics.append(f"health_score {health_status.get_health_score()}")

    # Uptime
    metrics.append(f"# HELP uptime_seconds Service uptime in seconds")
    metrics.append(f"# TYPE uptime_seconds counter")
    metrics.append(f"uptime_seconds {health_status.get_uptime()}")

    # Resource usage
    memory = psutil.virtual_memory()
    metrics.append(f"# HELP memory_usage_percent Memory usage percentage")
    metrics.append(f"# TYPE memory_usage_percent gauge")
    metrics.append(f"memory_usage_percent {memory.percent}")

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    metrics.append(f"# HELP cpu_usage_percent CPU usage percentage")
    metrics.append(f"# TYPE cpu_usage_percent gauge")
    metrics.append(f"cpu_usage_percent {cpu_percent}")

    # WebSocket sessions
    try:
        ws_manager = await get_websocket_manager()
        stats = await ws_manager.get_server_stats()
        metrics.append(f"# HELP websocket_sessions Active WebSocket sessions")
        metrics.append(f"# TYPE websocket_sessions gauge")
        metrics.append(f"websocket_sessions {stats['local_sessions']}")
    except:
        pass

    # Cache stats
    try:
        cache = await get_cache_service()
        cache_stats = cache.get_stats()
        metrics.append(f"# HELP cache_hits Cache hit count")
        metrics.append(f"# TYPE cache_hits counter")
        total_hits = (cache_stats.get('memory_hits', 0) +
                     cache_stats.get('redis_hits', 0) +
                     cache_stats.get('db_hits', 0))
        metrics.append(f"cache_hits {total_hits}")

        metrics.append(f"# HELP cache_misses Cache miss count")
        metrics.append(f"# TYPE cache_misses counter")
        metrics.append(f"cache_misses {cache_stats.get('misses', 0)}")
    except:
        pass

    # Return as plain text for Prometheus
    return Response(
        content="\n".join(metrics),
        media_type="text/plain"
    )