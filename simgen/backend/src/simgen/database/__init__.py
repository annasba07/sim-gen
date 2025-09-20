"""
Database optimization and connection management package.
Provides advanced connection pooling, query optimization, and performance monitoring.
"""

from .connection_pool import (
    AdvancedConnectionPool,
    ConnectionPoolConfig,
    get_connection_pool,
    get_optimized_session
)
from .query_optimizer import (
    QueryOptimizer,
    QueryHint,
    CacheStrategy,
    get_query_optimizer
)
from .service import (
    DatabaseService,
    get_database_service
)

__all__ = [
    "AdvancedConnectionPool",
    "ConnectionPoolConfig", 
    "get_connection_pool",
    "get_optimized_session",
    "QueryOptimizer",
    "QueryHint",
    "CacheStrategy",
    "get_query_optimizer",
    "DatabaseService",
    "get_database_service"
]