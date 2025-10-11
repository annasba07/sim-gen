"""
Advanced Database Connection Pool Manager
Provides optimized connection pooling, health monitoring, and performance tracking
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy import event, text
from dataclasses import dataclass, field
from threading import Lock

from ..core.config_clean import settings
from ..monitoring.observability import get_observability_manager


logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pool."""
    
    # Basic pool settings
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    
    # Performance settings
    echo: bool = False
    echo_pool: bool = False
    pool_reset_on_return: str = "commit"
    
    # Connection validation
    pool_health_check_interval: int = 300  # 5 minutes
    max_connection_age: int = 7200  # 2 hours
    
    # Retry settings
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0
    
    # Monitoring
    enable_metrics: bool = True
    slow_query_threshold: float = 1.0  # seconds


@dataclass
class ConnectionMetrics:
    """Metrics tracking for database connections."""
    
    total_connections_created: int = 0
    total_connections_closed: int = 0
    active_connections: int = 0
    pool_size_current: int = 0
    pool_checked_out: int = 0
    pool_overflow: int = 0
    pool_invalid: int = 0
    
    # Query metrics
    total_queries: int = 0
    slow_queries: int = 0
    failed_queries: int = 0
    average_query_time: float = 0.0
    
    # Health metrics
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    health_status: str = "unknown"
    
    # Timing metrics
    connection_times: list = field(default_factory=list)
    query_times: list = field(default_factory=list)


class AdvancedConnectionPool:
    """Advanced database connection pool with monitoring and optimization."""
    
    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        self.config = config or ConnectionPoolConfig()
        self.metrics = ConnectionMetrics()
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._lock = Lock()
        self._observability = get_observability_manager()
        
        # Setup metrics tracking
        if self.config.enable_metrics:
            self._setup_metrics_tracking()
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            database_url = self._get_database_url()
            
            # Create optimized async engine
            self.engine = create_async_engine(
                database_url,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                pool_reset_on_return=self.config.pool_reset_on_return,
                poolclass=QueuePool,
                # Advanced connection arguments
                connect_args={
                    "server_settings": {
                        "application_name": "simgen_ai",
                        "jit": "off",  # Disable JIT for consistent performance
                    },
                    "command_timeout": 30,
                    "statement_cache_size": 0,  # Disable prepared statement cache for async
                }
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )
            
            # Setup event listeners for monitoring
            self._setup_event_listeners()
            
            # Perform initial health check
            await self._perform_health_check()
            
            logger.info(f"Database connection pool initialized with {self.config.pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def _get_database_url(self) -> str:
        """Get optimized database URL."""
        database_url = settings.database_url
        
        # Convert postgres:// to postgresql:// for SQLAlchemy 2.0
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        # Use asyncpg for async operations
        if "postgresql://" in database_url and "+asyncpg" not in database_url:
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        return database_url
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for monitoring."""
        if not self.engine:
            return
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            with self._lock:
                self.metrics.total_connections_created += 1
                self.metrics.active_connections += 1
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.increment("db.connections.created")
                self._observability.metrics_collector.gauge("db.connections.active", self.metrics.active_connections)
        
        @event.listens_for(self.engine.sync_engine, "close")
        def on_close(dbapi_connection, connection_record):
            with self._lock:
                self.metrics.total_connections_closed += 1
                self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.increment("db.connections.closed")
                self._observability.metrics_collector.gauge("db.connections.active", self.metrics.active_connections)
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            with self._lock:
                self.metrics.pool_checked_out += 1
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.increment("db.pool.checkout")
        
        @event.listens_for(self.engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            with self._lock:
                self.metrics.pool_checked_out = max(0, self.metrics.pool_checked_out - 1)
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.increment("db.pool.checkin")
        
        @event.listens_for(self.engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            with self._lock:
                self.metrics.pool_invalid += 1
                self.metrics.consecutive_failures += 1
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.increment("db.connections.invalidated")
            
            logger.warning(f"Database connection invalidated: {exception}")
    
    def _setup_metrics_tracking(self) -> None:
        """Setup background metrics collection."""
        asyncio.create_task(self._metrics_collector())
    
    async def _metrics_collector(self) -> None:
        """Background task for collecting pool metrics."""
        while True:
            try:
                if self.engine:
                    pool = self.engine.pool
                    
                    with self._lock:
                        self.metrics.pool_size_current = pool.size()
                        self.metrics.pool_checked_out = pool.checkedout()
                        self.metrics.pool_overflow = pool.overflow()
                    
                    if self.config.enable_metrics:
                        self._observability.metrics_collector.gauge("db.pool.size", self.metrics.pool_size_current)
                        self._observability.metrics_collector.gauge("db.pool.checked_out", self.metrics.pool_checked_out)
                        self._observability.metrics_collector.gauge("db.pool.overflow", self.metrics.pool_overflow)
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting database metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup and monitoring."""
        if not self.session_factory:
            raise RuntimeError("Connection pool not initialized")
        
        session = None
        start_time = time.time()
        
        try:
            # Create session with retry logic
            for attempt in range(self.config.connection_retry_attempts):
                try:
                    session = self.session_factory()
                    break
                except Exception as e:
                    if attempt == self.config.connection_retry_attempts - 1:
                        raise
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.config.connection_retry_delay * (attempt + 1))
            
            # Track connection time
            connection_time = time.time() - start_time
            with self._lock:
                self.metrics.connection_times.append(connection_time)
                if len(self.metrics.connection_times) > 1000:
                    self.metrics.connection_times = self.metrics.connection_times[-500:]
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.timer("db.connection.time", connection_time * 1000)
            
            yield session
            
            # Commit if no exception occurred
            await session.commit()
            
        except Exception as e:
            if session:
                await session.rollback()
            
            with self._lock:
                self.metrics.failed_queries += 1
                self.metrics.consecutive_failures += 1
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.increment("db.queries.failed")
            
            logger.error(f"Database session error: {e}")
            raise
            
        finally:
            if session:
                await session.close()
            
            # Reset consecutive failures on success
            if 'e' not in locals():
                with self._lock:
                    self.metrics.consecutive_failures = 0
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query with performance monitoring."""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                
                # Track query performance
                query_time = time.time() - start_time
                
                with self._lock:
                    self.metrics.total_queries += 1
                    self.metrics.query_times.append(query_time)
                    if len(self.metrics.query_times) > 1000:
                        self.metrics.query_times = self.metrics.query_times[-500:]
                    
                    # Update average query time
                    if self.metrics.query_times:
                        self.metrics.average_query_time = sum(self.metrics.query_times) / len(self.metrics.query_times)
                    
                    # Track slow queries
                    if query_time > self.config.slow_query_threshold:
                        self.metrics.slow_queries += 1
                
                if self.config.enable_metrics:
                    self._observability.metrics_collector.increment("db.queries.total")
                    self._observability.metrics_collector.timer("db.query.time", query_time * 1000)
                    
                    if query_time > self.config.slow_query_threshold:
                        self._observability.metrics_collector.increment("db.queries.slow")
                        logger.warning(f"Slow query detected: {query_time:.3f}s - {query[:100]}...")
                
                return result
                
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"Query failed after {query_time:.3f}s: {e}")
            raise
    
    async def _perform_health_check(self) -> bool:
        """Perform database health check."""
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            
            health_check_time = time.time() - start_time
            
            with self._lock:
                self.metrics.last_health_check = time.time()
                self.metrics.health_status = "healthy"
                self.metrics.consecutive_failures = 0
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.timer("db.health_check.time", health_check_time * 1000)
                self._observability.metrics_collector.gauge("db.health_check.status", 1)
            
            return True
            
        except Exception as e:
            with self._lock:
                self.metrics.health_status = "unhealthy"
                self.metrics.consecutive_failures += 1
            
            if self.config.enable_metrics:
                self._observability.metrics_collector.gauge("db.health_check.status", 0)
            
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status and metrics."""
        pool_info = {}
        
        if self.engine:
            pool = self.engine.pool
            pool_info = {
                "size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
        
        return {
            "pool": pool_info,
            "metrics": {
                "total_connections_created": self.metrics.total_connections_created,
                "total_connections_closed": self.metrics.total_connections_closed,
                "active_connections": self.metrics.active_connections,
                "total_queries": self.metrics.total_queries,
                "slow_queries": self.metrics.slow_queries,
                "failed_queries": self.metrics.failed_queries,
                "average_query_time": self.metrics.average_query_time,
                "consecutive_failures": self.metrics.consecutive_failures,
                "health_status": self.metrics.health_status,
                "last_health_check": self.metrics.last_health_check
            },
            "config": {
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_timeout": self.config.pool_timeout,
                "pool_recycle": self.config.pool_recycle
            }
        }
    
    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection pool closed")


# Global connection pool instance
_connection_pool: Optional[AdvancedConnectionPool] = None


async def get_connection_pool() -> AdvancedConnectionPool:
    """Get the global connection pool instance."""
    global _connection_pool
    
    if _connection_pool is None:
        config = ConnectionPoolConfig(
            pool_size=getattr(settings, 'db_pool_size', 20),
            max_overflow=getattr(settings, 'db_max_overflow', 40),
            pool_timeout=getattr(settings, 'db_pool_timeout', 30),
            echo=getattr(settings, 'debug', False),
            enable_metrics=True
        )
        
        _connection_pool = AdvancedConnectionPool(config)
        await _connection_pool.initialize()
    
    return _connection_pool


async def get_optimized_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an optimized database session."""
    pool = await get_connection_pool()
    async with pool.get_session() as session:
        yield session