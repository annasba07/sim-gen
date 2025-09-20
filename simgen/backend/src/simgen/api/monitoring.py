"""
Production monitoring and health check API endpoints
Provides comprehensive system status, metrics, and health information
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..monitoring.observability import get_observability_manager, track_performance
from ..services.resilience import get_resilience_manager
from ..middleware.security import get_current_user, require_api_key
from ..services.llm_client import get_llm_client

try:
    from ..database.service import get_database_service
    from ..database.connection_pool import get_connection_pool
    from ..database.query_optimizer import get_query_optimizer
    DATABASE_MONITORING_AVAILABLE = True
except ImportError:
    DATABASE_MONITORING_AVAILABLE = False

router = APIRouter()


@router.get("/health")
async def basic_health_check():
    """Basic health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "simgen-ai",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
@track_performance("health_detailed")
async def detailed_health_check():
    """Comprehensive health check with all system components."""
    observability = get_observability_manager()
    resilience = get_resilience_manager()
    
    # Get health status from all systems
    health_status = observability.health_monitor.get_health_status()
    system_health = resilience.get_system_health()
    
    # Test critical services
    service_checks = await _test_critical_services()
    
    # Determine overall health
    overall_healthy = (
        health_status["healthy"] and 
        system_health["health_score"] >= 80 and
        all(check["healthy"] for check in service_checks.values())
    )
    
    status_code = 200 if overall_healthy else 503
    
    response_data = {
        "healthy": overall_healthy,
        "timestamp": datetime.now().isoformat(),
        "service": "simgen-ai",
        "version": "1.0.0",
        "uptime_seconds": time.time() - _get_start_time(),
        "health_checks": health_status["checks"],
        "circuit_breakers": system_health["circuit_breakers"],
        "system_health_score": system_health["health_score"],
        "service_checks": service_checks
    }
    
    return JSONResponse(content=response_data, status_code=status_code)


@router.get("/metrics")
@track_performance("metrics")
async def get_metrics(user_info: Dict[str, Any] = Depends(get_current_user)):
    """Get comprehensive system metrics."""
    
    # Require authentication for detailed metrics
    if user_info.get("user_id") == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required for metrics")
    
    observability = get_observability_manager()
    resilience = get_resilience_manager()
    
    metrics_summary = observability.metrics_collector.get_metrics_summary()
    performance_summary = observability.performance_tracker.get_performance_summary()
    system_health = resilience.get_system_health()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics_summary,
        "performance": performance_summary,
        "health_score": system_health["health_score"],
        "error_counts": system_health["error_counts"]
    }


@router.get("/metrics/{metric_name}/timeseries")
@track_performance("metrics_timeseries")
async def get_metric_timeseries(
    metric_name: str,
    limit: int = 100,
    user_info: Dict[str, Any] = Depends(require_api_key)
):
    """Get time series data for a specific metric."""
    
    observability = get_observability_manager()
    time_series_data = observability.metrics_collector.get_time_series(metric_name, limit)
    
    return {
        "metric_name": metric_name,
        "data_points": len(time_series_data),
        "time_series": time_series_data,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status/circuit-breakers")
@track_performance("circuit_breaker_status")
async def get_circuit_breaker_status(user_info: Dict[str, Any] = Depends(require_api_key)):
    """Get detailed circuit breaker status."""
    
    resilience = get_resilience_manager()
    
    circuit_status = {}
    for name, breaker in resilience.circuit_breakers.items():
        circuit_status[name] = breaker.get_metrics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "circuit_breakers": circuit_status,
        "total_circuits": len(circuit_status),
        "healthy_circuits": sum(1 for cb in circuit_status.values() if cb["state"] == "closed"),
        "degraded_circuits": sum(1 for cb in circuit_status.values() if cb["state"] == "half_open"),
        "failed_circuits": sum(1 for cb in circuit_status.values() if cb["state"] == "open")
    }


@router.get("/status/performance")
@track_performance("performance_status")
async def get_performance_status():
    """Get real-time performance metrics."""
    
    observability = get_observability_manager()
    
    # Get recent performance data
    performance_summary = observability.performance_tracker.get_performance_summary()
    metrics_summary = observability.metrics_collector.get_metrics_summary()
    
    # Extract key performance indicators
    kpis = {
        "active_requests": performance_summary["active_requests"],
        "avg_request_duration_ms": 0,
        "requests_per_minute": 0,
        "error_rate_percent": 0,
        "cache_hit_rate_percent": 0
    }
    
    # Calculate request duration from timers
    if "timers" in metrics_summary and metrics_summary["timers"]:
        avg_durations = [
            timer_data["mean_ms"] 
            for timer_data in metrics_summary["timers"].values()
        ]
        if avg_durations:
            kpis["avg_request_duration_ms"] = sum(avg_durations) / len(avg_durations)
    
    # Calculate requests per minute from counters
    if "counters" in metrics_summary:
        total_requests = sum(
            count for name, count in metrics_summary["counters"].items()
            if "requests." in name and ".started" in name
        )
        # Rough estimate based on uptime
        uptime_minutes = (time.time() - _get_start_time()) / 60
        if uptime_minutes > 0:
            kpis["requests_per_minute"] = total_requests / uptime_minutes
    
    return {
        "timestamp": datetime.now().isoformat(),
        "kpis": kpis,
        "detailed_metrics": metrics_summary,
        "active_requests": performance_summary
    }


@router.post("/monitoring/reset-metrics")
async def reset_metrics(user_info: Dict[str, Any] = Depends(require_api_key)):
    """Reset metrics (admin only)."""
    
    # Check if user has admin privileges
    if user_info.get("tier") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    observability = get_observability_manager()
    
    # Reset metrics
    observability.metrics_collector.counters.clear()
    observability.metrics_collector.gauges.clear()
    observability.metrics_collector.histograms.clear()
    observability.metrics_collector.timers.clear()
    
    return {
        "status": "success",
        "message": "Metrics reset successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/debug/system")
async def get_system_debug_info(user_info: Dict[str, Any] = Depends(require_api_key)):
    """Get detailed system debug information (requires API key)."""
    
    import psutil
    import sys
    import platform
    
    # System information
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "disk_total_gb": psutil.disk_usage('/').total / (1024**3)
    }
    
    # Process information
    process = psutil.Process()
    process_info = {
        "pid": process.pid,
        "memory_mb": process.memory_info().rss / (1024**2),
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "connections": len(process.connections())
    }
    
    # Environment information
    observability = get_observability_manager()
    env_info = {
        "active_monitoring": observability.system_monitor.monitoring,
        "health_checks_count": len(observability.health_monitor.health_checks),
        "metrics_count": len(observability.metrics_collector.metrics)
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": system_info,
        "process": process_info,
        "environment": env_info,
        "uptime_seconds": time.time() - _get_start_time()
    }


async def _test_critical_services() -> Dict[str, Dict[str, Any]]:
    """Test critical services for health check."""
    
    service_checks = {}
    
    # Test LLM client
    try:
        llm_client = get_llm_client()
        # Quick connection test (without actual API call)
        llm_healthy = llm_client.anthropic_client is not None or llm_client.openai_client is not None
        
        service_checks["llm"] = {
            "healthy": llm_healthy,
            "message": "LLM clients available" if llm_healthy else "No LLM clients configured"
        }
    except Exception as e:
        service_checks["llm"] = {
            "healthy": False,
            "message": f"LLM service error: {str(e)}"
        }
    
    # Test Redis (if available)
    try:
        from ..services.performance_optimizer import get_performance_pipeline
        pipeline = get_performance_pipeline()
        redis_healthy = pipeline.redis_client is not None
        
        if redis_healthy:
            # Quick ping test
            pipeline.redis_client.ping()
            
        service_checks["redis"] = {
            "healthy": redis_healthy,
            "message": "Redis connection healthy" if redis_healthy else "Redis not available"
        }
    except Exception as e:
        service_checks["redis"] = {
            "healthy": False,
            "message": f"Redis error: {str(e)}"
        }
    
    # Test MuJoCo (if available)
    try:
        from ..services.optimized_renderer import MUJOCO_AVAILABLE
        
        service_checks["mujoco"] = {
            "healthy": MUJOCO_AVAILABLE,
            "message": "MuJoCo available" if MUJOCO_AVAILABLE else "MuJoCo not installed"
        }
    except Exception as e:
        service_checks["mujoco"] = {
            "healthy": False,
            "message": f"MuJoCo error: {str(e)}"
        }
    
    return service_checks


# Application startup time tracking
_app_start_time = time.time()

def _get_start_time() -> float:
    """Get application start time."""
    return _app_start_time


# Health check for external monitoring
@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity tests."""
    return {"pong": datetime.now().isoformat()}


# Readiness probe for Kubernetes
@router.get("/ready")
async def readiness_probe():
    """Readiness probe endpoint for container orchestration."""
    
    # Check if critical services are ready
    try:
        service_checks = await _test_critical_services()
        
        # Consider ready if at least LLM service is healthy
        ready = service_checks.get("llm", {}).get("healthy", False)
        
        if ready:
            return {"ready": True, "timestamp": datetime.now().isoformat()}
        else:
            return JSONResponse(
                content={"ready": False, "timestamp": datetime.now().isoformat()},
                status_code=503
            )
            
    except Exception as e:
        return JSONResponse(
            content={
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )


# Liveness probe for Kubernetes
@router.get("/live")
async def liveness_probe():
    """Liveness probe endpoint for container orchestration."""
    
    # Basic liveness check - if we can respond, we're alive
    try:
        # Check if main event loop is responsive
        await asyncio.sleep(0.001)
        
        return {"alive": True, "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        return JSONResponse(
            content={
                "alive": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )


# Database monitoring endpoints
@router.get("/database/health")
@track_performance("database_health")
async def get_database_health(user_info: Dict[str, Any] = Depends(require_api_key)):
    """Get detailed database health and performance metrics."""
    
    if not DATABASE_MONITORING_AVAILABLE:
        return JSONResponse(
            content={"error": "Database monitoring not available"},
            status_code=503
        )
    
    try:
        db_service = await get_database_service()
        health_info = await db_service.get_database_health()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "database": health_info
        }
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Database health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/database/connection-pool")
@track_performance("database_connection_pool")
async def get_connection_pool_status(user_info: Dict[str, Any] = Depends(require_api_key)):
    """Get connection pool status and metrics."""
    
    if not DATABASE_MONITORING_AVAILABLE:
        return JSONResponse(
            content={"error": "Database monitoring not available"},
            status_code=503
        )
    
    try:
        connection_pool = await get_connection_pool()
        pool_status = await connection_pool.get_pool_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "connection_pool": pool_status
        }
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Connection pool status failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/database/query-performance")
@track_performance("database_query_performance")
async def get_query_performance(user_info: Dict[str, Any] = Depends(require_api_key)):
    """Get query performance metrics and optimization stats."""
    
    if not DATABASE_MONITORING_AVAILABLE:
        return JSONResponse(
            content={"error": "Database monitoring not available"},
            status_code=503
        )
    
    try:
        query_optimizer = await get_query_optimizer()
        performance_metrics = query_optimizer.get_query_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "query_performance": performance_metrics
        }
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Query performance check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/database/statistics")
@track_performance("database_statistics")
async def get_database_statistics(
    session_id: Optional[str] = None,
    days: int = 30,
    user_info: Dict[str, Any] = Depends(require_api_key)
):
    """Get simulation database statistics and analytics."""
    
    if not DATABASE_MONITORING_AVAILABLE:
        return JSONResponse(
            content={"error": "Database monitoring not available"},
            status_code=503
        )
    
    try:
        db_service = await get_database_service()
        statistics = await db_service.get_simulation_statistics(
            session_id=session_id,
            days=days
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": statistics
        }
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Database statistics failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.post("/database/cache/invalidate")
@track_performance("database_cache_invalidate")
async def invalidate_database_cache(
    cache_data: Dict[str, Any],
    user_info: Dict[str, Any] = Depends(require_api_key)
):
    """Invalidate database query cache by tags or keys."""
    
    # Check admin permissions
    if user_info.get("tier") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    if not DATABASE_MONITORING_AVAILABLE:
        return JSONResponse(
            content={"error": "Database monitoring not available"},
            status_code=503
        )
    
    try:
        query_optimizer = await get_query_optimizer()
        
        tags = cache_data.get("tags", [])
        keys = cache_data.get("keys", [])
        
        if not tags and not keys:
            raise HTTPException(
                status_code=400,
                detail="Must specify either 'tags' or 'keys' to invalidate"
            )
        
        await query_optimizer.invalidate_cache(tags=tags, keys=keys)
        
        return {
            "status": "success",
            "message": f"Cache invalidated for {len(tags)} tags and {len(keys)} keys",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Cache invalidation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )