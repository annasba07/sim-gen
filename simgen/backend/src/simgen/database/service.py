"""
Enhanced Database Service Layer
Provides high-level database operations with optimization, caching, and monitoring
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Type, TypeVar
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from contextlib import asynccontextmanager

from .connection_pool import get_optimized_session, get_connection_pool
from .query_optimizer import get_query_optimizer, QueryHint, CacheStrategy
from ..models.simulation import Simulation, SimulationTemplate, QualityAssessment, SimulationStatus
from ..models.schemas import SimulationCreate, SimulationUpdate
from ..monitoring.observability import get_observability_manager


logger = logging.getLogger(__name__)
T = TypeVar('T')


class DatabaseService:
    """Enhanced database service with optimization and monitoring."""
    
    def __init__(self):
        self.observability = get_observability_manager()
        self._query_optimizer = None
        self._connection_pool = None
    
    async def initialize(self) -> None:
        """Initialize the database service."""
        self._query_optimizer = await get_query_optimizer()
        self._connection_pool = await get_connection_pool()
        logger.info("Database service initialized")
    
    @asynccontextmanager
    async def get_session(self):
        """Get an optimized database session."""
        async with get_optimized_session() as session:
            yield session
    
    # Simulation operations
    async def create_simulation(
        self,
        simulation_data: Union[SimulationCreate, Dict[str, Any]],
        session: Optional[AsyncSession] = None
    ) -> Simulation:
        """Create a new simulation with optimized insertion."""
        
        if isinstance(simulation_data, dict):
            simulation_dict = simulation_data
        else:
            simulation_dict = simulation_data.dict()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if session:
                simulation = Simulation(**simulation_dict)
                session.add(simulation)
                await session.flush()  # Get the ID without committing
                await session.refresh(simulation)
            else:
                async with self.get_session() as session:
                    simulation = Simulation(**simulation_dict)
                    session.add(simulation)
                    await session.flush()
                    await session.refresh(simulation)
            
            # Track metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self.observability.metrics_collector.timer("db.simulation.create", execution_time * 1000)
            self.observability.metrics_collector.increment("db.simulation.created")
            
            logger.debug(f"Created simulation {simulation.id} in {execution_time:.3f}s")
            return simulation
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.simulation.create_failed")
            logger.error(f"Failed to create simulation: {e}")
            raise
    
    async def get_simulation(
        self,
        simulation_id: int,
        include_quality_assessments: bool = False,
        session: Optional[AsyncSession] = None
    ) -> Optional[Simulation]:
        """Get a simulation by ID with optimized querying."""
        
        cache_tags = ["simulations", f"simulation_{simulation_id}"]
        hints = QueryHint(
            use_cache=CacheStrategy.MEDIUM_TERM,
            cache_tags=cache_tags,
            prefetch_relations=["quality_assessments"] if include_quality_assessments else None
        )
        
        try:
            if session:
                query = select(Simulation).where(Simulation.id == simulation_id)
                
                if include_quality_assessments:
                    query = query.options(selectinload(Simulation.quality_assessments))
                
                result = await session.execute(query)
                simulation = result.scalar_one_or_none()
            else:
                async with self.get_session() as session:
                    query = select(Simulation).where(Simulation.id == simulation_id)
                    
                    if include_quality_assessments:
                        query = query.options(selectinload(Simulation.quality_assessments))
                    
                    result = await session.execute(query)
                    simulation = result.scalar_one_or_none()
            
            self.observability.metrics_collector.increment("db.simulation.retrieved")
            return simulation
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.simulation.retrieve_failed")
            logger.error(f"Failed to get simulation {simulation_id}: {e}")
            raise
    
    async def get_simulations_by_session(
        self,
        session_id: str,
        status: Optional[SimulationStatus] = None,
        limit: int = 50,
        offset: int = 0,
        session: Optional[AsyncSession] = None
    ) -> List[Simulation]:
        """Get simulations by session ID with pagination and filtering."""
        
        cache_tags = ["simulations", f"session_{session_id}"]
        hints = QueryHint(
            use_cache=CacheStrategy.SHORT_TERM,
            cache_tags=cache_tags,
            use_index=["session_id", "status", "created_at"]
        )
        
        try:
            conditions = [Simulation.session_id == session_id]
            if status:
                conditions.append(Simulation.status == status)
            
            if session:
                query = (
                    select(Simulation)
                    .where(and_(*conditions))
                    .order_by(Simulation.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                result = await session.execute(query)
                simulations = result.scalars().all()
            else:
                async with self.get_session() as session:
                    query = (
                        select(Simulation)
                        .where(and_(*conditions))
                        .order_by(Simulation.created_at.desc())
                        .limit(limit)
                        .offset(offset)
                    )
                    result = await session.execute(query)
                    simulations = result.scalars().all()
            
            self.observability.metrics_collector.increment("db.simulation.list_retrieved")
            return list(simulations)
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.simulation.list_failed")
            logger.error(f"Failed to get simulations for session {session_id}: {e}")
            raise
    
    async def update_simulation(
        self,
        simulation_id: int,
        update_data: Union[SimulationUpdate, Dict[str, Any]],
        session: Optional[AsyncSession] = None
    ) -> Optional[Simulation]:
        """Update a simulation with optimized query."""
        
        if isinstance(update_data, dict):
            update_dict = update_data
        else:
            update_dict = update_data.dict(exclude_unset=True)
        
        try:
            if session:
                # Update and return the updated record
                query = (
                    update(Simulation)
                    .where(Simulation.id == simulation_id)
                    .values(**update_dict)
                    .returning(Simulation)
                )
                result = await session.execute(query)
                simulation = result.scalar_one_or_none()
            else:
                async with self.get_session() as session:
                    query = (
                        update(Simulation)
                        .where(Simulation.id == simulation_id)
                        .values(**update_dict)
                        .returning(Simulation)
                    )
                    result = await session.execute(query)
                    simulation = result.scalar_one_or_none()
            
            # Invalidate cache
            if self._query_optimizer:
                await self._query_optimizer.invalidate_cache(
                    tags=["simulations", f"simulation_{simulation_id}"]
                )
            
            self.observability.metrics_collector.increment("db.simulation.updated")
            return simulation
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.simulation.update_failed")
            logger.error(f"Failed to update simulation {simulation_id}: {e}")
            raise
    
    async def delete_simulation(
        self,
        simulation_id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Delete a simulation and related data."""
        
        try:
            if session:
                # Delete related quality assessments first
                await session.execute(
                    delete(QualityAssessment).where(QualityAssessment.simulation_id == simulation_id)
                )
                
                # Delete the simulation
                result = await session.execute(
                    delete(Simulation).where(Simulation.id == simulation_id)
                )
                deleted = result.rowcount > 0
            else:
                async with self.get_session() as session:
                    # Delete related quality assessments first
                    await session.execute(
                        delete(QualityAssessment).where(QualityAssessment.simulation_id == simulation_id)
                    )
                    
                    # Delete the simulation
                    result = await session.execute(
                        delete(Simulation).where(Simulation.id == simulation_id)
                    )
                    deleted = result.rowcount > 0
            
            # Invalidate cache
            if self._query_optimizer:
                await self._query_optimizer.invalidate_cache(
                    tags=["simulations", f"simulation_{simulation_id}"]
                )
            
            if deleted:
                self.observability.metrics_collector.increment("db.simulation.deleted")
            
            return deleted
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.simulation.delete_failed")
            logger.error(f"Failed to delete simulation {simulation_id}: {e}")
            raise
    
    # Template operations
    async def get_templates(
        self,
        category: Optional[str] = None,
        active_only: bool = True,
        session: Optional[AsyncSession] = None
    ) -> List[SimulationTemplate]:
        """Get simulation templates with caching."""
        
        cache_tags = ["templates"]
        if category:
            cache_tags.append(f"category_{category}")
        
        hints = QueryHint(
            use_cache=CacheStrategy.LONG_TERM,
            cache_tags=cache_tags,
            use_index=["category", "is_active"]
        )
        
        try:
            conditions = []
            if category:
                conditions.append(SimulationTemplate.category == category)
            if active_only:
                conditions.append(SimulationTemplate.is_active == True)
            
            if session:
                if conditions:
                    query = select(SimulationTemplate).where(and_(*conditions))
                else:
                    query = select(SimulationTemplate)
                
                result = await session.execute(query)
                templates = result.scalars().all()
            else:
                async with self.get_session() as session:
                    if conditions:
                        query = select(SimulationTemplate).where(and_(*conditions))
                    else:
                        query = select(SimulationTemplate)
                    
                    result = await session.execute(query)
                    templates = result.scalars().all()
            
            self.observability.metrics_collector.increment("db.templates.retrieved")
            return list(templates)
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.templates.retrieve_failed")
            logger.error(f"Failed to get templates: {e}")
            raise
    
    async def get_template_by_name(
        self,
        name: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[SimulationTemplate]:
        """Get a template by name with caching."""
        
        hints = QueryHint(
            use_cache=CacheStrategy.LONG_TERM,
            cache_tags=["templates", f"template_{name}"],
            use_index=["name"]
        )
        
        try:
            if session:
                query = select(SimulationTemplate).where(SimulationTemplate.name == name)
                result = await session.execute(query)
                template = result.scalar_one_or_none()
            else:
                async with self.get_session() as session:
                    query = select(SimulationTemplate).where(SimulationTemplate.name == name)
                    result = await session.execute(query)
                    template = result.scalar_one_or_none()
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to get template {name}: {e}")
            raise
    
    # Quality assessment operations
    async def create_quality_assessment(
        self,
        assessment_data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> QualityAssessment:
        """Create a quality assessment."""
        
        try:
            if session:
                assessment = QualityAssessment(**assessment_data)
                session.add(assessment)
                await session.flush()
                await session.refresh(assessment)
            else:
                async with self.get_session() as session:
                    assessment = QualityAssessment(**assessment_data)
                    session.add(assessment)
                    await session.flush()
                    await session.refresh(assessment)
            
            # Invalidate related cache
            if self._query_optimizer:
                simulation_id = assessment_data.get("simulation_id")
                if simulation_id:
                    await self._query_optimizer.invalidate_cache(
                        tags=[f"simulation_{simulation_id}"]
                    )
            
            self.observability.metrics_collector.increment("db.quality_assessment.created")
            return assessment
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.quality_assessment.create_failed")
            logger.error(f"Failed to create quality assessment: {e}")
            raise
    
    # Analytics and reporting
    async def get_simulation_statistics(
        self,
        session_id: Optional[str] = None,
        days: int = 30,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Get simulation statistics with optimized aggregation."""

        # Input validation to prevent SQL injection
        if not isinstance(days, int) or days < 1 or days > 365:
            raise ValueError("Days must be an integer between 1 and 365")

        cache_tags = ["analytics", "statistics"]
        if session_id:
            cache_tags.append(f"session_{session_id}")

        hints = QueryHint(
            use_cache=CacheStrategy.MEDIUM_TERM,
            cache_tags=cache_tags
        )

        try:
            # Base conditions using parameterized query for safety
            from sqlalchemy import text
            interval_expr = text("INTERVAL :days DAY").bindparam(days=days)
            conditions = [
                Simulation.created_at >= func.now() - interval_expr
            ]
            
            if session_id:
                conditions.append(Simulation.session_id == session_id)
            
            if session:
                # Count by status
                status_query = (
                    select(
                        Simulation.status,
                        func.count(Simulation.id).label('count')
                    )
                    .where(and_(*conditions))
                    .group_by(Simulation.status)
                )
                status_result = await session.execute(status_query)
                status_counts = {row.status: row.count for row in status_result}
                
                # Average quality scores
                quality_query = (
                    select(
                        func.avg(Simulation.quality_score_overall).label('avg_overall'),
                        func.avg(Simulation.quality_score_physics).label('avg_physics'),
                        func.avg(Simulation.quality_score_visual).label('avg_visual'),
                        func.avg(Simulation.generation_duration).label('avg_duration')
                    )
                    .where(and_(*conditions))
                )
                quality_result = await session.execute(quality_query)
                quality_stats = quality_result.first()
                
            else:
                async with self.get_session() as session:
                    # Count by status
                    status_query = (
                        select(
                            Simulation.status,
                            func.count(Simulation.id).label('count')
                        )
                        .where(and_(*conditions))
                        .group_by(Simulation.status)
                    )
                    status_result = await session.execute(status_query)
                    status_counts = {row.status: row.count for row in status_result}
                    
                    # Average quality scores
                    quality_query = (
                        select(
                            func.avg(Simulation.quality_score_overall).label('avg_overall'),
                            func.avg(Simulation.quality_score_physics).label('avg_physics'),
                            func.avg(Simulation.quality_score_visual).label('avg_visual'),
                            func.avg(Simulation.generation_duration).label('avg_duration')
                        )
                        .where(and_(*conditions))
                    )
                    quality_result = await session.execute(quality_query)
                    quality_stats = quality_result.first()
            
            statistics = {
                "period_days": days,
                "status_counts": status_counts,
                "total_simulations": sum(status_counts.values()),
                "quality_metrics": {
                    "average_overall_score": float(quality_stats.avg_overall or 0),
                    "average_physics_score": float(quality_stats.avg_physics or 0),
                    "average_visual_score": float(quality_stats.avg_visual or 0),
                    "average_generation_duration": float(quality_stats.avg_duration or 0)
                } if quality_stats else {}
            }
            
            self.observability.metrics_collector.increment("db.analytics.statistics_retrieved")
            return statistics
            
        except Exception as e:
            self.observability.metrics_collector.increment("db.analytics.statistics_failed")
            logger.error(f"Failed to get simulation statistics: {e}")
            raise
    
    async def get_database_health(self) -> Dict[str, Any]:
        """Get comprehensive database health information."""
        
        try:
            # Get connection pool status
            pool_status = await self._connection_pool.get_pool_status() if self._connection_pool else {}
            
            # Get query optimizer metrics
            query_metrics = self._query_optimizer.get_query_metrics() if self._query_optimizer else {}
            
            # Test database connectivity
            async with self.get_session() as session:
                await session.execute(select(1))
                connectivity = "healthy"
            
            return {
                "connectivity": connectivity,
                "connection_pool": pool_status,
                "query_performance": query_metrics,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "connectivity": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service() -> DatabaseService:
    """Get the global database service instance."""
    global _database_service
    
    if _database_service is None:
        _database_service = DatabaseService()
        await _database_service.initialize()
    
    return _database_service