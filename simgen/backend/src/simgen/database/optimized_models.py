"""
Optimized database models with proper indexing and relationships
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class OptimizedSimulation(Base):
    """Optimized simulation model with proper indexes."""
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="pending", index=True)

    # MJCF and physics data
    mjcf_xml = Column(Text)
    physics_spec = Column(JSON)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Performance metrics
    processing_time_ms = Column(Float)
    frame_count = Column(Integer, default=0)

    # Relationships with lazy loading strategies
    quality_assessments = relationship(
        "QualityAssessment",
        back_populates="simulation",
        lazy="select",  # Load on access
        cascade="all, delete-orphan"
    )

    templates = relationship(
        "SimulationTemplate",
        back_populates="simulation",
        lazy="select"
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_session_status', 'session_id', 'status'),
        Index('idx_session_created', 'session_id', 'created_at'),
        Index('idx_status_created', 'status', 'created_at'),
        Index('idx_session_status_created', 'session_id', 'status', 'created_at'),
    )


class QualityAssessment(Base):
    """Quality assessment with optimized queries."""
    __tablename__ = "quality_assessments"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True)

    # Assessment scores
    physics_accuracy = Column(Float)
    visual_quality = Column(Float)
    performance_score = Column(Float)
    overall_score = Column(Float, index=True)

    # Details
    assessment_details = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationship
    simulation = relationship("OptimizedSimulation", back_populates="quality_assessments")

    __table_args__ = (
        Index('idx_sim_overall_score', 'simulation_id', 'overall_score'),
        Index('idx_created_score', 'created_at', 'overall_score'),
    )


class SimulationTemplate(Base):
    """Reusable simulation templates."""
    __tablename__ = "simulation_templates"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=True, index=True)

    name = Column(String(255), nullable=False, unique=True, index=True)
    category = Column(String(100), index=True)
    tags = Column(JSON)

    # Template data
    physics_spec = Column(JSON, nullable=False)
    mjcf_template = Column(Text)

    # Metadata
    is_public = Column(Boolean, default=True, index=True)
    usage_count = Column(Integer, default=0, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationship
    simulation = relationship("OptimizedSimulation", back_populates="templates")

    __table_args__ = (
        Index('idx_category_public', 'category', 'is_public'),
        Index('idx_usage_public', 'usage_count', 'is_public'),
        UniqueConstraint('name', name='uq_template_name'),
    )


class SketchCache(Base):
    """Cache for computer vision analysis results."""
    __tablename__ = "sketch_cache"

    id = Column(Integer, primary_key=True, index=True)
    image_hash = Column(String(64), nullable=False, unique=True, index=True)

    # Cached results
    cv_analysis = Column(JSON)
    physics_spec = Column(JSON)
    confidence_score = Column(Float)

    # Metadata
    hit_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)

    __table_args__ = (
        Index('idx_hash_accessed', 'image_hash', 'last_accessed'),
        Index('idx_created_accessed', 'created_at', 'last_accessed'),
    )


class LLMResponseCache(Base):
    """Cache for LLM API responses."""
    __tablename__ = "llm_response_cache"

    id = Column(Integer, primary_key=True, index=True)
    prompt_hash = Column(String(64), nullable=False, index=True)
    model = Column(String(100), index=True)

    # Cache key components
    prompt_text = Column(Text)
    parameters = Column(JSON)

    # Cached response
    response = Column(JSON, nullable=False)

    # Metadata
    token_count = Column(Integer)
    response_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), index=True)

    __table_args__ = (
        Index('idx_prompt_model', 'prompt_hash', 'model'),
        Index('idx_expires', 'expires_at'),
        UniqueConstraint('prompt_hash', 'model', 'parameters', name='uq_llm_cache'),
    )


class PerformanceMetrics(Base):
    """Track performance metrics for optimization."""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Operation details
    operation_type = Column(String(100), nullable=False, index=True)
    operation_name = Column(String(255), nullable=False, index=True)

    # Performance data
    duration_ms = Column(Float, nullable=False)
    memory_mb = Column(Float)
    cpu_percent = Column(Float)

    # Context
    session_id = Column(String(255), index=True)
    request_id = Column(String(255), index=True)
    user_id = Column(String(255), index=True)

    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metadata = Column(JSON)

    __table_args__ = (
        Index('idx_op_type_timestamp', 'operation_type', 'timestamp'),
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_duration_timestamp', 'duration_ms', 'timestamp'),
    )