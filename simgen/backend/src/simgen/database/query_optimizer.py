"""
Database Query Optimization and Caching Layer
Provides intelligent query caching, optimization hints, and performance monitoring
"""

import asyncio
import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from sqlalchemy import text, select, update, delete, and_, or_
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config_clean import settings
from ..monitoring.observability import get_observability_manager


logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different query types."""
    NO_CACHE = "no_cache"
    SHORT_TERM = "short_term"  # 5 minutes
    MEDIUM_TERM = "medium_term"  # 1 hour
    LONG_TERM = "long_term"  # 24 hours
    PERSISTENT = "persistent"  # Until explicitly invalidated


@dataclass
class QueryHint:
    """Performance hints for query optimization."""
    use_index: Optional[List[str]] = None
    join_strategy: Optional[str] = None  # 'nested', 'hash', 'merge'
    prefetch_relations: Optional[List[str]] = None
    limit_optimization: bool = True
    use_cache: CacheStrategy = CacheStrategy.MEDIUM_TERM
    cache_tags: Optional[List[str]] = None


@dataclass
class QueryMetrics:
    """Metrics for query performance tracking."""
    query_hash: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_executed: float = 0.0
    slow_query_count: int = 0


class QueryOptimizer:
    """Advanced query optimizer with caching and performance monitoring."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.metrics: Dict[str, QueryMetrics] = {}
        self.observability = get_observability_manager()
        self._initialized = False
        
        # Cache TTL settings (in seconds)
        self.cache_ttl = {
            CacheStrategy.SHORT_TERM: 300,      # 5 minutes
            CacheStrategy.MEDIUM_TERM: 3600,    # 1 hour
            CacheStrategy.LONG_TERM: 86400,     # 24 hours
            CacheStrategy.PERSISTENT: -1        # No expiration
        }
        
        # Query optimization patterns
        self.optimization_patterns = {
            "simulations": {
                "index_hints": ["session_id", "status", "created_at"],
                "prefetch": ["quality_assessments"],
                "default_cache": CacheStrategy.MEDIUM_TERM
            },
            "templates": {
                "index_hints": ["category", "name", "is_active"],
                "prefetch": [],
                "default_cache": CacheStrategy.LONG_TERM
            },
            "quality_assessments": {
                "index_hints": ["simulation_id", "iteration"],
                "prefetch": [],
                "default_cache": CacheStrategy.SHORT_TERM
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the query optimizer."""
        try:
            # Setup Redis for caching
            if hasattr(settings, 'redis_url') and settings.redis_url:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    encoding='utf-8',
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("Redis connection established for query caching")
            else:
                logger.warning("Redis not configured, query caching disabled")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize query optimizer: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any], table: str = "unknown") -> str:
        """Generate a unique cache key for a query."""
        # Normalize query by removing extra whitespace
        normalized_query = " ".join(query.split())
        
        # Create hash of query + parameters
        query_data = {
            "query": normalized_query,
            "params": params
        }
        
        query_str = json.dumps(query_data, sort_keys=True)
        query_hash = hashlib.sha256(query_str.encode()).hexdigest()[:16]
        
        return f"simgen:query:{table}:{query_hash}"
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate a hash for query metrics tracking."""
        normalized_query = " ".join(query.split())
        return hashlib.sha256(normalized_query.encode()).hexdigest()[:12]
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                self.observability.metrics_collector.increment("db.cache.hits")
                return json.loads(cached_data)
            
            self.observability.metrics_collector.increment("db.cache.misses")
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _set_cache(self, cache_key: str, data: Any, ttl: int) -> None:
        """Store data in cache."""
        if not self.redis_client:
            return
        
        try:
            serialized_data = json.dumps(data, default=str)
            
            if ttl > 0:
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            else:
                await self.redis_client.set(cache_key, serialized_data)
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _invalidate_cache_by_tags(self, tags: List[str]) -> None:
        """Invalidate cache entries by tags."""
        if not self.redis_client or not tags:
            return
        
        try:
            for tag in tags:
                # Find all keys with this tag
                pattern = f"simgen:query:*{tag}*"
                keys = await self.redis_client.keys(pattern)
                
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.debug(f"Invalidated {len(keys)} cache entries for tag: {tag}")
                    
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    def _update_query_metrics(self, query_hash: str, execution_time: float, cache_hit: bool) -> None:
        """Update metrics for a query."""
        if query_hash not in self.metrics:
            self.metrics[query_hash] = QueryMetrics(query_hash=query_hash)
        
        metrics = self.metrics[query_hash]
        metrics.execution_count += 1
        metrics.last_executed = time.time()
        
        if cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1
            metrics.total_time += execution_time
            metrics.avg_time = metrics.total_time / (metrics.execution_count - metrics.cache_hits)
            metrics.min_time = min(metrics.min_time, execution_time)
            metrics.max_time = max(metrics.max_time, execution_time)
            
            # Track slow queries (>1 second)
            if execution_time > 1.0:
                metrics.slow_query_count += 1
                self.observability.metrics_collector.increment("db.queries.slow")
        
        # Update observability metrics
        self.observability.metrics_collector.timer("db.query.execution_time", execution_time * 1000)
        if cache_hit:
            self.observability.metrics_collector.timer("db.query.cache_time", execution_time * 1000)
    
    def _optimize_query(self, query: str, table: str, hints: Optional[QueryHint] = None) -> str:
        """Apply optimization hints to a query."""
        if not hints:
            # Auto-detect optimization based on table
            if table in self.optimization_patterns:
                pattern = self.optimization_patterns[table]
                hints = QueryHint(
                    use_index=pattern["index_hints"],
                    prefetch_relations=pattern["prefetch"],
                    use_cache=pattern["default_cache"]
                )
        
        optimized_query = query
        
        # Add index hints (PostgreSQL specific)
        if hints and hints.use_index:
            # This is a simplified example - real implementation would be more sophisticated
            for index_col in hints.use_index:
                if f"ORDER BY {index_col}" not in optimized_query and "ORDER BY" not in optimized_query:
                    if "WHERE" in optimized_query and index_col in optimized_query:
                        optimized_query += f" ORDER BY {index_col}"
        
        return optimized_query
    
    async def execute_optimized_query(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        table: str = "unknown",
        hints: Optional[QueryHint] = None,
        return_type: str = "fetchall"  # fetchall, fetchone, execute
    ) -> Any:
        """Execute an optimized query with caching."""
        
        if not self._initialized:
            await self.initialize()
        
        params = params or {}
        start_time = time.time()
        
        # Generate cache key and query hash
        cache_key = self._generate_cache_key(query, params, table)
        query_hash = self._generate_query_hash(query)
        
        # Determine cache strategy
        cache_strategy = hints.use_cache if hints else CacheStrategy.MEDIUM_TERM
        use_cache = cache_strategy != CacheStrategy.NO_CACHE
        
        # Try to get from cache first
        if use_cache:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result is not None:
                cache_time = time.time() - start_time
                self._update_query_metrics(query_hash, cache_time, cache_hit=True)
                logger.debug(f"Cache hit for query: {query_hash}")
                return cached_result
        
        # Optimize the query
        optimized_query = self._optimize_query(query, table, hints)
        
        try:
            # Execute the query
            result = await session.execute(text(optimized_query), params)
            
            # Process result based on return type
            if return_type == "fetchall":
                data = result.fetchall()
                # Convert to list of dicts for JSON serialization
                processed_data = [dict(row._mapping) for row in data]
            elif return_type == "fetchone":
                row = result.fetchone()
                processed_data = dict(row._mapping) if row else None
            else:  # execute
                processed_data = {"rowcount": result.rowcount}
            
            execution_time = time.time() - start_time
            
            # Cache the result if appropriate
            if use_cache and return_type in ["fetchall", "fetchone"]:
                ttl = self.cache_ttl.get(cache_strategy, 3600)
                await self._set_cache(cache_key, processed_data, ttl)
            
            # Update metrics
            self._update_query_metrics(query_hash, execution_time, cache_hit=False)
            
            logger.debug(f"Query executed in {execution_time:.3f}s: {query_hash}")
            
            return processed_data
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.3f}s: {e}")
            self.observability.metrics_collector.increment("db.queries.failed")
            raise
    
    async def bulk_insert_optimized(
        self,
        session: AsyncSession,
        table_name: str,
        records: List[Dict[str, Any]],
        batch_size: int = 1000,
        on_conflict: str = "ignore"  # ignore, update, error
    ) -> int:
        """Perform optimized bulk insert with batching."""
        
        if not records:
            return 0
        
        start_time = time.time()
        total_inserted = 0
        
        try:
            # Process in batches
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                if on_conflict == "ignore":
                    # Use PostgreSQL-specific INSERT ... ON CONFLICT DO NOTHING
                    columns = list(batch[0].keys())
                    values_placeholder = ", ".join([f":{col}" for col in columns])
                    
                    query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({values_placeholder})
                    ON CONFLICT DO NOTHING
                    """
                    
                    for record in batch:
                        await session.execute(text(query), record)
                        total_inserted += 1
                
                elif on_conflict == "update":
                    # Handle upsert logic (simplified example)
                    for record in batch:
                        # This would need to be customized based on the specific table
                        pass
                
                else:  # error
                    # Standard insert that will fail on conflict
                    columns = list(batch[0].keys())
                    values_placeholder = ", ".join([f":{col}" for col in columns])
                    
                    query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({values_placeholder})
                    """
                    
                    for record in batch:
                        await session.execute(text(query), record)
                        total_inserted += 1
            
            # Commit the transaction
            await session.commit()
            
            execution_time = time.time() - start_time
            logger.info(f"Bulk inserted {total_inserted} records in {execution_time:.3f}s")
            
            # Update metrics
            self.observability.metrics_collector.increment("db.bulk_inserts.completed")
            self.observability.metrics_collector.timer("db.bulk_insert.time", execution_time * 1000)
            self.observability.metrics_collector.gauge("db.bulk_insert.records", total_inserted)
            
            return total_inserted
            
        except Exception as e:
            await session.rollback()
            execution_time = time.time() - start_time
            logger.error(f"Bulk insert failed after {execution_time:.3f}s: {e}")
            self.observability.metrics_collector.increment("db.bulk_inserts.failed")
            raise
    
    async def invalidate_cache(self, tags: Optional[List[str]] = None, keys: Optional[List[str]] = None) -> None:
        """Invalidate cache entries by tags or specific keys."""
        if tags:
            await self._invalidate_cache_by_tags(tags)
        
        if keys and self.redis_client:
            try:
                await self.redis_client.delete(*keys)
                logger.debug(f"Invalidated {len(keys)} specific cache keys")
            except Exception as e:
                logger.warning(f"Failed to invalidate specific cache keys: {e}")
    
    def get_query_metrics(self) -> Dict[str, Any]:
        """Get comprehensive query performance metrics."""
        total_queries = sum(m.execution_count for m in self.metrics.values())
        total_cache_hits = sum(m.cache_hits for m in self.metrics.values())
        total_slow_queries = sum(m.slow_query_count for m in self.metrics.values())
        
        cache_hit_rate = (total_cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        # Find slowest queries
        slowest_queries = sorted(
            self.metrics.values(),
            key=lambda m: m.avg_time,
            reverse=True
        )[:10]
        
        return {
            "summary": {
                "total_queries": total_queries,
                "total_cache_hits": total_cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "total_slow_queries": total_slow_queries,
                "unique_queries": len(self.metrics)
            },
            "slowest_queries": [
                {
                    "query_hash": q.query_hash,
                    "avg_time": q.avg_time,
                    "execution_count": q.execution_count,
                    "slow_query_count": q.slow_query_count
                }
                for q in slowest_queries
            ]
        }


# Cache management decorators
def cached_query(
    table: str,
    cache_strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM,
    cache_tags: Optional[List[str]] = None
):
    """Decorator for automatic query caching."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = await get_query_optimizer()
            
            # This would need to be implemented based on the specific function signature
            # For now, just call the original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global query optimizer instance
_query_optimizer: Optional[QueryOptimizer] = None


async def get_query_optimizer() -> QueryOptimizer:
    """Get the global query optimizer instance."""
    global _query_optimizer
    
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
        await _query_optimizer.initialize()
    
    return _query_optimizer