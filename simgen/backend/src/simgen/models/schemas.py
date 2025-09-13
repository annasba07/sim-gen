from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from .simulation import SimulationStatus, SimulationGenerationMethod


# Entity extraction schemas
class GeometrySchema(BaseModel):
    shape: str = Field(..., description="Shape type: box, sphere, cylinder, mesh")
    dimensions: List[float] = Field(..., description="Shape dimensions")


class MaterialSchema(BaseModel):
    density: float = Field(1000.0, description="Material density (kg/m³)")
    friction: float = Field(0.5, description="Friction coefficient")
    restitution: float = Field(0.1, description="Restitution coefficient")


class ObjectSchema(BaseModel):
    name: str = Field(..., description="Object name")
    type: str = Field(..., description="Object type: rigid_body, soft_body, robot, vehicle")
    geometry: GeometrySchema
    material: MaterialSchema
    position: List[float] = Field([0, 0, 0], description="Initial position [x, y, z]")
    orientation: List[float] = Field([0, 0, 0], description="Initial orientation [roll, pitch, yaw]")


class ConstraintSchema(BaseModel):
    type: str = Field(..., description="Constraint type: joint, contact, force")
    bodies: List[str] = Field(..., description="Names of connected bodies")
    parameters: Dict[str, Any] = Field({}, description="Constraint-specific parameters")


class EnvironmentSchema(BaseModel):
    gravity: List[float] = Field([0, 0, -9.81], description="Gravity vector [x, y, z]")
    ground: Dict[str, Any] = Field({"type": "plane", "friction": 0.8}, description="Ground configuration")
    boundaries: Dict[str, Any] = Field({"type": "none"}, description="Boundary configuration")


class ExtractedEntities(BaseModel):
    objects: List[ObjectSchema] = []
    constraints: List[ConstraintSchema] = []
    environment: EnvironmentSchema = Field(default_factory=EnvironmentSchema)


# Quality assessment schemas
class QualityScore(BaseModel):
    overall: float = Field(..., ge=0, le=10, description="Overall quality score (0-10)")
    physics: float = Field(..., ge=0, le=10, description="Physics quality score (0-10)")
    visual: float = Field(..., ge=0, le=10, description="Visual quality score (0-10)")
    functional: float = Field(..., ge=0, le=10, description="Functional quality score (0-10)")
    performance: float = Field(..., ge=0, le=10, description="Performance score (0-10)")


class DetectedIssue(BaseModel):
    type: str = Field(..., description="Issue type")
    severity: str = Field(..., description="Issue severity: low, medium, high")
    description: str = Field(..., description="Issue description")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix")


class QualityAssessmentSchema(BaseModel):
    iteration: int = 0
    quality_score: QualityScore
    detected_issues: List[DetectedIssue] = []
    analysis_duration: Optional[float] = None
    analysis_method: Optional[str] = None


# Request schemas
class SimulationRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=2000, description="Natural language simulation description")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    generation_method: Optional[SimulationGenerationMethod] = Field(None, description="Preferred generation method")
    max_iterations: Optional[int] = Field(5, ge=1, le=10, description="Maximum refinement iterations")


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=10, description="User rating (1-10)")
    feedback: Optional[str] = Field(None, max_length=1000, description="User feedback text")


# Response schemas
class SimulationResponse(BaseModel):
    id: int
    session_id: str
    status: SimulationStatus
    user_prompt: str
    
    # Generation results
    mjcf_content: Optional[str] = None
    generation_method: Optional[SimulationGenerationMethod] = None
    
    # Quality metrics
    quality_score: Optional[QualityScore] = None
    
    # Processing metadata
    refinement_iterations: int = 0
    generation_duration: Optional[float] = None
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class SimulationListResponse(BaseModel):
    simulations: List[SimulationResponse]
    total: int
    page: int
    page_size: int


class ExecutionData(BaseModel):
    success: bool
    screenshots: List[str] = []  # Base64 encoded images or file paths
    physics_stats: List[Dict[str, float]] = []
    duration: float
    error_message: Optional[str] = None


class SimulationExecutionResponse(BaseModel):
    simulation_id: int
    execution_data: ExecutionData
    quality_assessment: Optional[QualityAssessmentSchema] = None


# Template schemas
class TemplateResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str
    keywords: List[str]
    usage_count: int
    success_rate: float
    is_active: bool
    
    class Config:
        from_attributes = True


# WebSocket message schemas
class WebSocketMessage(BaseModel):
    type: str
    session_id: str
    data: Dict[str, Any]


class ProgressUpdate(BaseModel):
    stage: str = Field(..., description="Current processing stage")
    progress: float = Field(..., ge=0, le=1, description="Progress percentage (0-1)")
    message: str = Field(..., description="Progress message")
    estimated_time_remaining: Optional[float] = Field(None, description="Estimated time remaining (seconds)")


# Health check schema
class HealthCheck(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime
    services: Dict[str, str] = {}  # Service status checks


# Sketch-to-physics schemas
class SketchGenerationRequest(BaseModel):
    sketch_data: str = Field(..., description="Base64 encoded sketch image data")
    prompt: Optional[str] = Field(None, max_length=1000, description="Optional text prompt to accompany sketch")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    style_preferences: Optional[Dict[str, Any]] = Field(None, description="Visual/physics style preferences")
    max_iterations: Optional[int] = Field(5, ge=1, le=10, description="Maximum refinement iterations")


class SketchAnalysisResponse(BaseModel):
    success: bool
    confidence_score: float = Field(..., ge=0, le=1, description="Analysis confidence (0-1)")
    physics_description: str = Field(..., description="Generated physics description from sketch")
    detected_objects: int = Field(..., description="Number of physics objects detected")
    detected_constraints: int = Field(..., description="Number of constraints detected")
    raw_vision_output: Optional[str] = Field(None, description="Raw vision model analysis")
    error_message: Optional[str] = None


class MultiModalResponse(BaseModel):
    sketch_analysis: Optional[SketchAnalysisResponse] = None
    enhanced_prompt: str = Field(..., description="Combined sketch + text enhanced prompt")
    sketch_contribution: float = Field(..., ge=0, le=1, description="Sketch contribution weight")
    text_contribution: float = Field(..., ge=0, le=1, description="Text contribution weight")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence")


class SketchGenerationResponse(BaseModel):
    id: int
    session_id: str
    status: SimulationStatus
    
    # Input data
    user_prompt: Optional[str] = None
    has_sketch: bool = Field(..., description="Whether sketch was provided")
    
    # Multi-modal analysis
    multimodal_analysis: Optional[MultiModalResponse] = None
    
    # Generation results  
    mjcf_content: Optional[str] = None
    generation_method: Optional[SimulationGenerationMethod] = None
    
    # Quality metrics
    quality_score: Optional[QualityScore] = None
    
    # Processing metadata
    refinement_iterations: int = 0
    generation_duration: Optional[float] = None
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Test/demo schemas for sketch functionality
class SketchTestResponse(BaseModel):
    status: str
    input_data: Dict[str, Any] = Field(..., description="Input data summary")
    sketch_analysis: Optional[SketchAnalysisResponse] = None
    multimodal_analysis: Optional[MultiModalResponse] = None
    generation_result: Optional[Dict[str, Any]] = None
    processing_time: float = Field(..., description="Total processing time in seconds")
    success: bool
    error: Optional[str] = None