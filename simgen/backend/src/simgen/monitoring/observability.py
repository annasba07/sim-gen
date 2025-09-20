"""
Production-grade monitoring and observability for SimGen AI
Comprehensive metrics, logging, tracing, and health monitoring
"""

import asyncio
import time
import logging
import sys
import json
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class HealthCheck:
    """Health check configuration and state."""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 10
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0


class MetricsCollector:
    """High-performance metrics collection system."""
    
    def __init__(self, max_points: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            self._add_metric_point(name, value, MetricType.COUNTER, tags or {})
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
            self._add_metric_point(name, value, MetricType.GAUGE, tags or {})
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._add_metric_point(name, value, MetricType.HISTOGRAM, tags or {})
    
    def record_timer(self, name: str, duration_seconds: float, tags: Dict[str, str] = None):
        """Record a timer measurement."""
        with self._lock:
            self.timers[name].append(duration_seconds)
            # Keep last 1000 values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            self._add_metric_point(name, duration_seconds, MetricType.TIMER, tags or {})
    
    def _add_metric_point(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str]):
        """Add metric point to time series."""
        point = MetricPoint(name, value, metric_type, tags=tags)
        self.metrics[name].append(point)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "timers": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate histogram statistics
            for name, values in self.histograms.items():
                if values:
                    summary["histograms"][name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            
            # Calculate timer statistics
            for name, values in self.timers.items():
                if values:
                    summary["timers"][name] = {
                        "count": len(values),
                        "mean_ms": (sum(values) / len(values)) * 1000,
                        "min_ms": min(values) * 1000,
                        "max_ms": max(values) * 1000,
                        "p50_ms": self._percentile(values, 50) * 1000,
                        "p95_ms": self._percentile(values, 95) * 1000,
                        "p99_ms": self._percentile(values, 99) * 1000
                    }
            
            return summary
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_time_series(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        with self._lock:
            if metric_name in self.metrics:
                points = list(self.metrics[metric_name])[-limit:]
                return [point.to_dict() for point in points]
            return []


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_task = None
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
        logger.info(f"System monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("system.cpu.usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system.memory.usage_percent", memory.percent)
            self.metrics.set_gauge("system.memory.available_gb", memory.available / (1024**3))
            self.metrics.set_gauge("system.memory.used_gb", memory.used / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
            self.metrics.set_gauge("system.disk.free_gb", disk.free / (1024**3))
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics.set_gauge("system.network.bytes_sent", network.bytes_sent)
            self.metrics.set_gauge("system.network.bytes_recv", network.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            self.metrics.set_gauge("process.memory.rss_mb", process.memory_info().rss / (1024**2))
            self.metrics.set_gauge("process.cpu.percent", process.cpu_percent())
            self.metrics.set_gauge("process.threads", process.num_threads())
            
            # Python metrics
            self.metrics.set_gauge("python.gc.objects", len(sys.modules))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class HealthMonitor:
    """Health check monitoring system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitoring = False
        self.monitor_task = None
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable,
        interval_seconds: int = 30,
        timeout_seconds: int = 10
    ):
        """Register a health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds
        )
        logger.info(f"Health check registered: {name}")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if not self.health_checks:
            logger.warning("No health checks registered")
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Health monitoring started ({len(self.health_checks)} checks)")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main health monitoring loop."""
        while self.monitoring:
            try:
                await self._run_health_checks()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _run_health_checks(self):
        """Run all due health checks."""
        current_time = datetime.now()
        
        for health_check in self.health_checks.values():
            # Check if it's time to run this health check
            if (health_check.last_check is None or 
                (current_time - health_check.last_check).total_seconds() >= health_check.interval_seconds):
                
                try:
                    # Run health check with timeout
                    result = await asyncio.wait_for(
                        self._run_single_check(health_check),
                        timeout=health_check.timeout_seconds
                    )
                    
                    health_check.last_result = result
                    health_check.last_error = None
                    health_check.last_check = current_time
                    
                    if result:
                        health_check.consecutive_failures = 0
                        self.metrics.set_gauge(f"health.{health_check.name}.status", 1)
                    else:
                        health_check.consecutive_failures += 1
                        self.metrics.set_gauge(f"health.{health_check.name}.status", 0)
                    
                    self.metrics.set_gauge(f"health.{health_check.name}.consecutive_failures", 
                                         health_check.consecutive_failures)
                    
                except asyncio.TimeoutError:
                    health_check.last_result = False
                    health_check.last_error = f"Timeout after {health_check.timeout_seconds}s"
                    health_check.last_check = current_time
                    health_check.consecutive_failures += 1
                    
                    self.metrics.set_gauge(f"health.{health_check.name}.status", 0)
                    self.metrics.increment(f"health.{health_check.name}.timeouts")
                    
                except Exception as e:
                    health_check.last_result = False
                    health_check.last_error = str(e)
                    health_check.last_check = current_time
                    health_check.consecutive_failures += 1
                    
                    self.metrics.set_gauge(f"health.{health_check.name}.status", 0)
                    self.metrics.increment(f"health.{health_check.name}.errors")
                    
                    logger.error(f"Health check '{health_check.name}' failed: {e}")
    
    async def _run_single_check(self, health_check: HealthCheck) -> bool:
        """Run a single health check."""
        if asyncio.iscoroutinefunction(health_check.check_function):
            return await health_check.check_function()
        else:
            return health_check.check_function()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        overall_healthy = True
        checks_status = {}
        
        for name, health_check in self.health_checks.items():
            is_healthy = health_check.last_result is True and health_check.consecutive_failures < 3
            
            checks_status[name] = {
                "healthy": is_healthy,
                "last_check": health_check.last_check.isoformat() if health_check.last_check else None,
                "last_result": health_check.last_result,
                "last_error": health_check.last_error,
                "consecutive_failures": health_check.consecutive_failures
            }
            
            if not is_healthy:
                overall_healthy = False
        
        return {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "checks": checks_status
        }


class PerformanceTracker:
    """Track application performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_requests = {}
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def track_request(self, endpoint: str, method: str = "POST"):
        """Context manager to track request performance."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        with self._lock:
            self.active_requests[request_id] = {
                "endpoint": endpoint,
                "method": method,
                "start_time": start_time
            }
        
        self.metrics.increment(f"requests.{endpoint}.started")
        self.metrics.set_gauge("requests.active", len(self.active_requests))
        
        try:
            yield request_id
            
            # Success metrics
            duration = time.time() - start_time
            self.metrics.record_timer(f"requests.{endpoint}.duration", duration)
            self.metrics.increment(f"requests.{endpoint}.success")
            
        except Exception as e:
            # Error metrics
            error_type = type(e).__name__
            self.metrics.increment(f"requests.{endpoint}.errors")
            self.metrics.increment(f"errors.{error_type}")
            raise
            
        finally:
            with self._lock:
                self.active_requests.pop(request_id, None)
            self.metrics.set_gauge("requests.active", len(self.active_requests))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            active_count = len(self.active_requests)
            
            # Calculate active request ages
            current_time = time.time()
            request_ages = [
                current_time - req["start_time"] 
                for req in self.active_requests.values()
            ]
        
        return {
            "active_requests": active_count,
            "oldest_request_age": max(request_ages) if request_ages else 0,
            "avg_request_age": sum(request_ages) / len(request_ages) if request_ages else 0
        }


class ObservabilityManager:
    """Central observability manager."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        async def check_memory():
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        
        async def check_disk_space():
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 90  # Alert if disk usage > 90%
        
        async def check_cpu():
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 80  # Alert if CPU usage > 80%
        
        self.health_monitor.register_health_check("memory", check_memory, interval_seconds=60)
        self.health_monitor.register_health_check("disk_space", check_disk_space, interval_seconds=300)
        self.health_monitor.register_health_check("cpu", check_cpu, interval_seconds=60)
    
    async def start_monitoring(self):
        """Start all monitoring systems."""
        await self.system_monitor.start_monitoring()
        await self.health_monitor.start_monitoring()
        logger.info("Observability monitoring started")
    
    async def stop_monitoring(self):
        """Stop all monitoring systems."""
        await self.system_monitor.stop_monitoring()
        await self.health_monitor.stop_monitoring()
        logger.info("Observability monitoring stopped")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "metrics": self.metrics_collector.get_metrics_summary(),
            "health": self.health_monitor.get_health_status(),
            "performance": self.performance_tracker.get_performance_summary(),
            "timestamp": datetime.now().isoformat()
        }


# Global observability manager
_observability_manager = None

def get_observability_manager() -> ObservabilityManager:
    """Get global observability manager."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager


# Decorators for easy metrics integration
def track_performance(endpoint_name: str = None):
    """Decorator to track function performance."""
    def decorator(func):
        name = endpoint_name or func.__name__
        
        async def wrapper(*args, **kwargs):
            manager = get_observability_manager()
            async with manager.performance_tracker.track_request(name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator