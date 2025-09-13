from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, Enum
from sqlalchemy.sql import func
import enum

from ..db.base import Base


class SimulationStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"  
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationGenerationMethod(str, enum.Enum):
    TEMPLATE_BASED = "template_based"
    LLM_GENERATION = "llm_generation"
    HYBRID = "hybrid"


class Simulation(Base):
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    
    # Input data
    user_prompt = Column(Text, nullable=False)
    extracted_entities = Column(JSON, nullable=True)  # Structured entity data
    
    # Generation metadata
    status = Column(Enum(SimulationStatus), default=SimulationStatus.PENDING, nullable=False)
    generation_method = Column(Enum(SimulationGenerationMethod), nullable=True)
    
    # Results
    mjcf_content = Column(Text, nullable=True)
    execution_data = Column(JSON, nullable=True)  # Screenshots, physics stats, etc.
    
    # Quality metrics
    quality_score_overall = Column(Float, nullable=True)
    quality_score_physics = Column(Float, nullable=True)
    quality_score_visual = Column(Float, nullable=True)
    quality_score_functional = Column(Float, nullable=True)
    
    # Processing metadata
    refinement_iterations = Column(Integer, default=0)
    generation_duration = Column(Float, nullable=True)  # Total time in seconds
    error_message = Column(Text, nullable=True)
    
    # User feedback
    user_rating = Column(Integer, nullable=True)  # 1-10 scale
    user_feedback = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class SimulationTemplate(Base):
    __tablename__ = "simulation_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Template metadata
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)  # robotics, physics, vehicles, etc.
    
    # Template content
    mjcf_template = Column(Text, nullable=False)  # Jinja2 template
    parameter_schema = Column(JSON, nullable=False)  # JSON schema for parameters
    
    # Matching criteria
    keywords = Column(JSON, nullable=False)  # List of keywords for matching
    entity_patterns = Column(JSON, nullable=False)  # Patterns for entity matching
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    average_quality_score = Column(Float, default=0.0)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class QualityAssessment(Base):
    __tablename__ = "quality_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, nullable=False, index=True)
    
    # Assessment iteration (for feedback loops)
    iteration = Column(Integer, default=0, nullable=False)
    
    # Detailed quality metrics
    physics_stability = Column(Float, nullable=True)
    energy_conservation = Column(Float, nullable=True)
    constraint_satisfaction = Column(Float, nullable=True)
    motion_realism = Column(Float, nullable=True)
    visual_quality = Column(Float, nullable=True)
    performance_score = Column(Float, nullable=True)
    
    # Issue detection
    detected_issues = Column(JSON, nullable=True)  # List of issues found
    suggested_fixes = Column(JSON, nullable=True)  # Suggested parameter adjustments
    
    # Analysis metadata
    analysis_duration = Column(Float, nullable=True)
    analysis_method = Column(String(100), nullable=True)  # visual, physics, hybrid
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)