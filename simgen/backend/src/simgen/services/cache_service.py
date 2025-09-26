"""
Unified caching service for SimGen
Provides multi-tier caching with Redis and database fallback
"""

import json
import hashlib
import asyncio
import time
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
import pickle

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from ..database.optimized_models import SketchCache, LLMResponseCache
from ..core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class CacheTier:
    """Cache tier enumeration."""
    MEMORY = "memory"  # In-process memory (fastest, limited size)
    REDIS = "redis"    # Redis cache (fast, shared across processes)
    DATABASE = "database"  # Database cache (slower, persistent)


class CacheService:
    """
    Multi-tier caching service with automatic fallback.
    Provides caching for expensive operations like CV analysis and LLM calls.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        memory_cache_size: int = 100,
        default_ttl: int = 3600
    ):
        self.redis_client = redis_client
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_cache_size = memory_cache_size
        self.default_ttl = default_ttl
        self._memory_access_times: Dict[str, float] = {}
        self._stats = {
            "memory_hits": 0,
            "redis_hits": 0,
            "db_hits": 0,
            "misses": 0,
            "errors": 0
        }

    async def initialize(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize cache service with Redis connection."""
        try:
            if not self.redis_client:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                await self.redis_client.ping()
                logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory cache only: {e}")
            self.redis_client = None

    async def close(self):
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()

    def _generate_cache_key(self, prefix: str, data: Union[str, bytes, Dict]) -> str:
        """Generate a unique cache key from data."""
        if isinstance(data, str):
            hash_input = data.encode()
        elif isinstance(data, bytes):
            hash_input = data
        else:
            hash_input = json.dumps(data, sort_keys=True).encode()

        hash_value = hashlib.sha256(hash_input).hexdigest()
        return f"{prefix}:{hash_value}"

    def _evict_lru_from_memory(self):
        """Evict least recently used item from memory cache."""
        if len(self.memory_cache) >= self.memory_cache_size:
            # Find LRU item
            lru_key = min(self._memory_access_times, key=self._memory_access_times.get)
            del self.memory_cache[lru_key]
            del self._memory_access_times[lru_key]
            logger.debug(f"Evicted LRU item from memory cache: {lru_key}")

    async def get_cv_analysis(
        self,
        image_data: bytes,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached CV analysis results.

        Args:
            image_data: Raw image bytes
            session: Optional database session

        Returns:
            Cached CV analysis or None if not found
        """
        cache_key = self._generate_cache_key("cv", image_data)

        # Check memory cache
        if cache_key in self.memory_cache:
            self._memory_access_times[cache_key] = time.time()
            self._stats["memory_hits"] += 1
            logger.debug(f"Memory cache hit for CV analysis: {cache_key}")
            return self.memory_cache[cache_key]["data"]

        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self._stats["redis_hits"] += 1
                    logger.debug(f"Redis cache hit for CV analysis: {cache_key}")

                    # Deserialize and update memory cache
                    result = pickle.loads(cached_data)
                    self._evict_lru_from_memory()
                    self.memory_cache[cache_key] = {
                        "data": result,
                        "timestamp": time.time()
                    }
                    self._memory_access_times[cache_key] = time.time()
                    return result
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                self._stats["errors"] += 1

        # Check database cache
        if session:
            try:
                image_hash = hashlib.sha256(image_data).hexdigest()
                stmt = select(SketchCache).where(SketchCache.image_hash == image_hash)
                result = await session.execute(stmt)
                cache_entry = result.scalar_one_or_none()

                if cache_entry:
                    self._stats["db_hits"] += 1
                    logger.debug(f"Database cache hit for CV analysis: {image_hash}")

                    # Update hit count and last accessed
                    await session.execute(
                        update(SketchCache)
                        .where(SketchCache.id == cache_entry.id)
                        .values(
                            hit_count=SketchCache.hit_count + 1,
                            last_accessed=datetime.utcnow()
                        )
                    )
                    await session.commit()

                    cv_data = {
                        "cv_analysis": cache_entry.cv_analysis,
                        "physics_spec": cache_entry.physics_spec,
                        "confidence_score": cache_entry.confidence_score
                    }

                    # Update higher-tier caches
                    await self._update_cache_tiers(cache_key, cv_data)
                    return cv_data

            except Exception as e:
                logger.error(f"Database cache error: {e}")
                self._stats["errors"] += 1

        self._stats["misses"] += 1
        return None

    async def set_cv_analysis(
        self,
        image_data: bytes,
        cv_analysis: Dict[str, Any],
        physics_spec: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0,
        session: Optional[AsyncSession] = None,
        ttl: Optional[int] = None
    ):
        """
        Cache CV analysis results.

        Args:
            image_data: Raw image bytes
            cv_analysis: CV analysis results
            physics_spec: Optional physics spec data
            confidence_score: Analysis confidence score
            session: Optional database session
            ttl: Time to live in seconds
        """
        cache_key = self._generate_cache_key("cv", image_data)
        ttl = ttl or self.default_ttl

        cache_data = {
            "cv_analysis": cv_analysis,
            "physics_spec": physics_spec,
            "confidence_score": confidence_score
        }

        # Update memory cache
        self._evict_lru_from_memory()
        self.memory_cache[cache_key] = {
            "data": cache_data,
            "timestamp": time.time()
        }
        self._memory_access_times[cache_key] = time.time()

        # Update Redis cache
        if self.redis_client:
            try:
                serialized = pickle.dumps(cache_data)
                await self.redis_client.setex(cache_key, ttl, serialized)
                logger.debug(f"Cached CV analysis in Redis: {cache_key}")
            except Exception as e:
                logger.error(f"Redis set error: {e}")

        # Update database cache
        if session:
            try:
                image_hash = hashlib.sha256(image_data).hexdigest()

                # Check if entry exists
                stmt = select(SketchCache).where(SketchCache.image_hash == image_hash)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing entry
                    await session.execute(
                        update(SketchCache)
                        .where(SketchCache.id == existing.id)
                        .values(
                            cv_analysis=cv_analysis,
                            physics_spec=physics_spec,
                            confidence_score=confidence_score,
                            hit_count=0,
                            last_accessed=datetime.utcnow()
                        )
                    )
                else:
                    # Create new entry
                    cache_entry = SketchCache(
                        image_hash=image_hash,
                        cv_analysis=cv_analysis,
                        physics_spec=physics_spec,
                        confidence_score=confidence_score
                    )
                    session.add(cache_entry)

                await session.commit()
                logger.debug(f"Cached CV analysis in database: {image_hash}")

            except Exception as e:
                logger.error(f"Database cache set error: {e}")
                await session.rollback()

    async def get_llm_response(
        self,
        prompt: str,
        model: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached LLM response."""
        cache_data = {
            "prompt": prompt,
            "model": model,
            "parameters": parameters or {}
        }
        cache_key = self._generate_cache_key("llm", cache_data)

        # Check memory cache
        if cache_key in self.memory_cache:
            self._memory_access_times[cache_key] = time.time()
            self._stats["memory_hits"] += 1
            return self.memory_cache[cache_key]["data"]

        # Check Redis
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self._stats["redis_hits"] += 1
                    result = json.loads(cached_data)

                    # Update memory cache
                    self._evict_lru_from_memory()
                    self.memory_cache[cache_key] = {
                        "data": result,
                        "timestamp": time.time()
                    }
                    self._memory_access_times[cache_key] = time.time()
                    return result
            except Exception as e:
                logger.error(f"Redis LLM cache error: {e}")

        # Check database
        if session:
            try:
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                stmt = select(LLMResponseCache).where(
                    LLMResponseCache.prompt_hash == prompt_hash,
                    LLMResponseCache.model == model,
                    LLMResponseCache.expires_at > datetime.utcnow()
                )
                result = await session.execute(stmt)
                cache_entry = result.scalar_one_or_none()

                if cache_entry:
                    self._stats["db_hits"] += 1

                    # Update caches
                    await self._update_cache_tiers(cache_key, cache_entry.response)
                    return cache_entry.response

            except Exception as e:
                logger.error(f"Database LLM cache error: {e}")

        self._stats["misses"] += 1
        return None

    async def set_llm_response(
        self,
        prompt: str,
        response: Dict[str, Any],
        model: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        session: Optional[AsyncSession] = None,
        ttl: Optional[int] = None
    ):
        """Cache LLM response."""
        cache_data = {
            "prompt": prompt,
            "model": model,
            "parameters": parameters or {}
        }
        cache_key = self._generate_cache_key("llm", cache_data)
        ttl = ttl or self.default_ttl

        # Update memory cache
        self._evict_lru_from_memory()
        self.memory_cache[cache_key] = {
            "data": response,
            "timestamp": time.time()
        }
        self._memory_access_times[cache_key] = time.time()

        # Update Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(response)
                )
            except Exception as e:
                logger.error(f"Redis LLM set error: {e}")

        # Update database
        if session:
            try:
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)

                cache_entry = LLMResponseCache(
                    prompt_hash=prompt_hash,
                    model=model,
                    prompt_text=prompt[:1000],  # Store first 1000 chars
                    parameters=parameters,
                    response=response,
                    token_count=token_count,
                    response_time_ms=response_time_ms,
                    expires_at=expires_at
                )
                session.add(cache_entry)
                await session.commit()

            except Exception as e:
                logger.error(f"Database LLM set error: {e}")
                await session.rollback()

    async def _update_cache_tiers(self, cache_key: str, data: Any, ttl: int = None):
        """Update higher-tier caches with data."""
        ttl = ttl or self.default_ttl

        # Update memory cache
        self._evict_lru_from_memory()
        self.memory_cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        self._memory_access_times[cache_key] = time.time()

        # Update Redis if available
        if self.redis_client:
            try:
                if isinstance(data, dict):
                    serialized = json.dumps(data)
                else:
                    serialized = pickle.dumps(data)
                await self.redis_client.setex(cache_key, ttl, serialized)
            except Exception as e:
                logger.error(f"Failed to update Redis cache: {e}")

    async def invalidate_cv_cache(self, image_data: bytes):
        """Invalidate CV analysis cache for specific image."""
        cache_key = self._generate_cache_key("cv", image_data)

        # Remove from memory
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            del self._memory_access_times[cache_key]

        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")

    async def cleanup_old_entries(self, session: AsyncSession, days: int = 7):
        """Clean up old cache entries from database."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Clean sketch cache
            await session.execute(
                delete(SketchCache).where(
                    SketchCache.last_accessed < cutoff_date,
                    SketchCache.hit_count < 5
                )
            )

            # Clean LLM cache
            await session.execute(
                delete(LLMResponseCache).where(
                    LLMResponseCache.expires_at < datetime.utcnow()
                )
            )

            await session.commit()
            logger.info("Cleaned up old cache entries")

        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            await session.rollback()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum(self._stats.values()) - self._stats["errors"]
        hit_rate = 0.0
        if total_requests > 0:
            hits = self._stats["memory_hits"] + self._stats["redis_hits"] + self._stats["db_hits"]
            hit_rate = (hits / total_requests) * 100

        return {
            **self._stats,
            "memory_cache_size": len(self.memory_cache),
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests
        }


# Global cache service instance
cache_service = CacheService()


async def get_cache_service() -> CacheService:
    """Get the cache service instance."""
    return cache_service