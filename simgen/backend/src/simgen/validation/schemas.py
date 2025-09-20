"""
Comprehensive API Validation Schemas
Enhanced Pydantic models with validation, documentation, and examples
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum
import re
from pydantic import (
    BaseModel, 
    Field, 
    validator, 
    root_validator,
    EmailStr,
    HttpUrl,
    constr,
    conint,
    confloat,
    conlist
)


# Enums for constrained values
class RenderQuality(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    CINEMATIC = "cinematic"


class PhysicsAccuracy(str, Enum):
    BASIC = "basic"
    REALISTIC = "realistic"
    PRECISE = "precise"


class LightingStyle(str, Enum):
    AMBIENT = "ambient"
    DIRECTIONAL = "directional"
    DRAMATIC = "dramatic"
    STUDIO = "studio"


class SimulationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisMode(str, Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    PHYSICS_FOCUSED = "physics_focused"


# Base validation models
class BaseRequest(BaseModel):
    """Base request model with common validation."""
    
    class Config:
        # Enable alias generation from field names
        allow_population_by_field_name = True
        # Validate assignments
        validate_assignment = True
        # Use enum values in schema
        use_enum_values = True
        # Custom JSON encoders
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    request_id: Optional[str] = Field(
        None,
        description="Optional request identifier for tracking",
        example="req_abc123def456",
        regex=r"^req_[a-zA-Z0-9_-]+$"
    )


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    
    class Config:
        allow_population_by_field_name = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response generation timestamp"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier if provided"
    )


# Style and preference models
class StylePreferences(BaseModel):
    """Visual and physics style preferences for simulation generation."""
    
    render_quality: RenderQuality = Field(
        default=RenderQuality.STANDARD,
        description="Rendering quality level - higher quality takes longer",
        example="high"
    )
    
    physics_accuracy: PhysicsAccuracy = Field(
        default=PhysicsAccuracy.REALISTIC,
        description="Physics simulation accuracy - precise mode uses more computation",
        example="realistic"  
    )
    
    lighting: LightingStyle = Field(
        default=LightingStyle.DIRECTIONAL,
        description="Lighting style for visual rendering",
        example="studio"
    )
    
    background_color: Optional[constr(regex=r"^#[0-9A-Fa-f]{6}$")] = Field(
        None,
        description="Background color in hex format",
        example="#FFFFFF"
    )
    
    camera_angle: Optional[confloat(ge=-180, le=180)] = Field(
        None,
        description="Camera angle in degrees",
        example=45.0
    )
    
    enable_shadows: bool = Field(
        default=True,
        description="Enable realistic shadow rendering"
    )
    
    enable_reflections: bool = Field(
        default=False,
        description="Enable surface reflections (performance intensive)"
    )


class GenerationOptions(BaseModel):
    """Options controlling simulation generation behavior."""
    
    enable_optimization: bool = Field(
        default=True,
        description="Enable performance optimizations during generation"
    )
    
    max_objects: conint(ge=1, le=100) = Field(
        default=20,
        description="Maximum number of objects in the simulation",
        example=10
    )
    
    simulation_duration: confloat(ge=1.0, le=120.0) = Field(
        default=10.0,
        description="Simulation duration in seconds",
        example=15.0
    )
    
    frame_rate: conint(ge=10, le=120) = Field(
        default=60,
        description="Simulation frame rate (FPS)",
        example=60
    )
    
    enable_collision_detection: bool = Field(
        default=True,
        description="Enable collision detection between objects"
    )
    
    gravity: confloat(ge=-50.0, le=50.0) = Field(
        default=-9.81,
        description="Gravity acceleration (m/s²)",
        example=-9.81
    )
    
    time_step: confloat(ge=0.0001, le=0.1) = Field(
        default=0.002,
        description="Physics simulation time step (seconds)",
        example=0.002
    )
    
    solver_iterations: conint(ge=1, le=100) = Field(
        default=10,
        description="Physics solver iteration count (higher = more accurate)",
        example=10
    )


# Main request/response schemas
class SimulationRequest(BaseRequest):
    """Request to generate a physics simulation from text prompt."""
    
    prompt: constr(min_length=10, max_length=2000) = Field(
        ...,
        description="Natural language description of the desired simulation",
        example="A red bouncy ball dropping onto a wooden table and bouncing several times before coming to rest"
    )
    
    session_id: constr(regex=r"^session_[a-zA-Z0-9_-]+$") = Field(
        ...,
        description="Unique session identifier for tracking related requests",
        example="session_user123_20241201"
    )
    
    style_preferences: Optional[StylePreferences] = Field(
        None,
        description="Visual and physics style preferences"
    )
    
    generation_options: Optional[GenerationOptions] = Field(
        None,
        description="Advanced generation control options"
    )
    
    enable_quality_check: bool = Field(
        default=True,
        description="Enable automatic quality assessment of generated simulation"
    )
    
    tags: Optional[conlist(str, max_items=10)] = Field(
        None,
        description="Optional tags for categorizing the simulation",
        example=["physics", "bouncing", "demo"]
    )
    
    @validator('prompt')
    def validate_prompt_content(cls, v):
        """Validate prompt content for security and quality."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'eval\s*\(',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Prompt contains potentially unsafe content")
        
        # Ensure reasonable content
        if len(v.split()) < 3:
            raise ValueError("Prompt should contain at least 3 words for meaningful simulation generation")
        
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID format and security."""
        if not v.startswith('session_'):
            raise ValueError("Session ID must start with 'session_'")
        
        # Remove prefix and check remaining characters
        suffix = v[8:]  # Remove 'session_'
        if not suffix or len(suffix) < 3:
            raise ValueError("Session ID must have meaningful suffix after 'session_'")
        
        # Ensure only safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', suffix):
            raise ValueError("Session ID can only contain alphanumeric characters, underscores, and hyphens")
        
        return v


class QualityScores(BaseModel):
    """Quality assessment scores for a simulation."""
    
    overall: confloat(ge=0.0, le=10.0) = Field(
        description="Overall quality score (0-10)",
        example=8.5
    )
    
    physics: confloat(ge=0.0, le=10.0) = Field(
        description="Physics realism and accuracy score (0-10)",
        example=9.2
    )
    
    visual: confloat(ge=0.0, le=10.0) = Field(
        description="Visual quality and aesthetics score (0-10)",
        example=7.8
    )
    
    functional: confloat(ge=0.0, le=10.0) = Field(
        description="Functional correctness score (0-10)",
        example=8.0
    )
    
    performance: Optional[confloat(ge=0.0, le=10.0)] = Field(
        None,
        description="Simulation performance score (0-10)",
        example=6.5
    )


class ExecutionData(BaseModel):
    """Simulation execution results and media."""
    
    screenshots: Optional[List[str]] = Field(
        None,
        description="Base64 encoded screenshots from simulation",
        example=["iVBORw0KGgoAAAANSUhEUgA..."]
    )
    
    video_path: Optional[str] = Field(
        None,
        description="Path to generated video file",
        example="/storage/videos/sim_12345.mp4"
    )
    
    video_base64: Optional[str] = Field(
        None,
        description="Base64 encoded video data (for small videos only)"
    )
    
    physics_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Physics simulation statistics and metrics",
        example={
            "total_energy": 156.7,
            "max_velocity": 12.3,
            "collision_count": 8,
            "simulation_stable": True
        }
    )
    
    render_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Rendering statistics and performance metrics",
        example={
            "render_time_ms": 1250,
            "frames_rendered": 600,
            "average_fps": 48.2
        }
    )
    
    @validator('screenshots')
    def validate_screenshots(cls, v):
        """Validate screenshot data."""
        if v is not None:
            for i, screenshot in enumerate(v):
                if not screenshot:
                    raise ValueError(f"Screenshot {i} is empty")
                
                # Basic base64 validation
                try:
                    import base64
                    base64.b64decode(screenshot, validate=True)
                except Exception:
                    raise ValueError(f"Screenshot {i} is not valid base64 data")
        
        return v


class SimulationResponse(BaseResponse):
    """Response from simulation generation request."""
    
    id: int = Field(
        description="Unique simulation identifier",
        example=12345
    )
    
    session_id: str = Field(
        description="Session identifier from request",
        example="session_user123_20241201"
    )
    
    status: SimulationStatus = Field(
        description="Current simulation status",
        example="completed"
    )
    
    mjcf_content: Optional[str] = Field(
        None,
        description="Generated MuJoCo XML simulation definition",
        example="<mujoco model=\"bouncing_ball\">...</mujoco>"
    )
    
    execution_data: Optional[ExecutionData] = Field(
        None,
        description="Simulation execution results and media"
    )
    
    quality_scores: Optional[QualityScores] = Field(
        None,
        description="Quality assessment scores"
    )
    
    generation_duration: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Time taken to generate simulation (seconds)",
        example=8.7
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if simulation failed",
        example="Failed to parse physics constraints"
    )
    
    created_at: datetime = Field(
        description="Simulation creation timestamp"
    )
    
    updated_at: datetime = Field(
        description="Last update timestamp"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Simulation tags",
        example=["physics", "bouncing", "demo"]
    )


class SketchAnalysisRequest(BaseRequest):
    """Request to analyze a hand-drawn sketch for simulation generation."""
    
    image_data: str = Field(
        ...,
        description="Base64 encoded sketch image (PNG/JPEG)",
        example="iVBORw0KGgoAAAANSUhEUgA..."
    )
    
    user_text: Optional[constr(max_length=1000)] = Field(
        None,
        description="Optional text description to accompany sketch",
        example="A ball rolling down a ramp with increasing speed"
    )
    
    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.DETAILED,
        description="Depth of analysis to perform on the sketch"
    )
    
    expected_objects: Optional[conint(ge=1, le=50)] = Field(
        None,
        description="Expected number of objects in the sketch",
        example=3
    )
    
    style_preferences: Optional[StylePreferences] = Field(
        None,
        description="Style preferences for the generated simulation"
    )
    
    @validator('image_data')
    def validate_image_data(cls, v):
        """Validate base64 image data."""
        if not v:
            raise ValueError("Image data cannot be empty")
        
        try:
            import base64
            decoded = base64.b64decode(v, validate=True)
            
            # Check minimum size (should be at least a few KB for meaningful image)
            if len(decoded) < 1000:
                raise ValueError("Image data appears too small to be a valid image")
            
            # Check maximum size (10MB limit)
            if len(decoded) > 10 * 1024 * 1024:
                raise ValueError("Image data exceeds maximum size of 10MB")
            
            # Basic format validation by checking magic numbers
            if not (decoded.startswith(b'\xff\xd8\xff') or  # JPEG
                   decoded.startswith(b'\x89PNG\r\n\x1a\n')):  # PNG
                raise ValueError("Image must be in JPEG or PNG format")
        
        except base64.binascii.Error:
            raise ValueError("Invalid base64 image data")
        
        return v


class SketchAnalysisResponse(BaseResponse):
    """Response from sketch analysis request."""
    
    analysis_id: str = Field(
        description="Unique analysis identifier",
        example="analysis_abc123"
    )
    
    detected_objects: List[Dict[str, Any]] = Field(
        description="Objects detected in the sketch",
        example=[
            {
                "type": "ball",
                "confidence": 0.95,
                "properties": {"size": "medium", "material": "rubber"},
                "position": {"x": 120, "y": 200}
            }
        ]
    )
    
    scene_description: str = Field(
        description="Natural language description of the analyzed scene",
        example="A ball positioned at the top of an inclined ramp, ready to roll down"
    )
    
    physics_relationships: List[Dict[str, Any]] = Field(
        description="Detected physics relationships between objects",
        example=[
            {
                "type": "gravity_affected",
                "objects": ["ball"],
                "strength": "normal"
            }
        ]
    )
    
    suggested_simulation: Optional[Dict[str, Any]] = Field(
        None,
        description="Automatically suggested simulation parameters"
    )
    
    analysis_duration: confloat(ge=0.0) = Field(
        description="Time taken for analysis (seconds)",
        example=2.3
    )
    
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        description="Overall confidence in the analysis",
        example=0.87
    )


class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    error: str = Field(
        description="Error type identifier",
        example="validation_failed"
    )
    
    message: str = Field(
        description="Human-readable error description",
        example="The provided prompt is too short"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error occurrence timestamp"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error context and details",
        example={
            "field": "prompt",
            "provided_length": 5,
            "minimum_length": 10
        }
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier if available",
        example="req_abc123def456"
    )
    
    suggested_action: Optional[str] = Field(
        None,
        description="Suggested action to resolve the error",
        example="Please provide a prompt with at least 10 characters"
    )


class HealthResponse(BaseModel):
    """System health check response."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall system health status",
        example="healthy"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )
    
    service: str = Field(
        default="simgen-ai",
        description="Service identifier"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Service version"
    )
    
    uptime_seconds: Optional[confloat(ge=0.0)] = Field(
        None,
        description="Service uptime in seconds",
        example=3600.5
    )
    
    health_checks: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Individual component health status",
        example={
            "database": {
                "status": "healthy",
                "response_time_ms": 12.3,
                "details": {"connections": 15, "pool_size": 20}
            },
            "llm_service": {
                "status": "healthy", 
                "response_time_ms": 234.5
            }
        }
    )


class ValidationRequest(BaseModel):
    """Request for API validation testing."""
    
    test_type: Literal["security", "performance", "format"] = Field(
        description="Type of validation test to perform"
    )
    
    test_data: Dict[str, Any] = Field(
        description="Test data to validate"
    )
    
    strict_mode: bool = Field(
        default=False,
        description="Enable strict validation mode"
    )


class ValidationResponse(BaseModel):
    """Response from validation testing."""
    
    is_valid: bool = Field(
        description="Whether the test data passed validation"
    )
    
    errors: List[str] = Field(
        default=[],
        description="List of validation errors found"
    )
    
    warnings: List[str] = Field(
        default=[],
        description="List of validation warnings"
    )
    
    security_issues: List[Dict[str, Any]] = Field(
        default=[],
        description="Security issues detected"
    )
    
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics from validation"
    )


# Authentication schemas
class LoginRequest(BaseModel):
    """User login request."""
    
    email: EmailStr = Field(
        description="User email address",
        example="user@example.com"
    )
    
    password: constr(min_length=8) = Field(
        description="User password",
        example="secure_password123"
    )


class LoginResponse(BaseModel):
    """User login response."""
    
    access_token: str = Field(
        description="JWT access token",
        example="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    )
    
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )
    
    expires_in: int = Field(
        description="Token expiration time in seconds",
        example=3600
    )
    
    user_info: Dict[str, Any] = Field(
        description="Basic user information",
        example={
            "user_id": "user123",
            "email": "user@example.com",
            "tier": "standard"
        }
    )


# Export all schemas for easy importing
__all__ = [
    # Enums
    "RenderQuality",
    "PhysicsAccuracy", 
    "LightingStyle",
    "SimulationStatus",
    "AnalysisMode",
    
    # Base models
    "BaseRequest",
    "BaseResponse",
    
    # Component models
    "StylePreferences",
    "GenerationOptions",
    "QualityScores",
    "ExecutionData",
    
    # Main schemas
    "SimulationRequest",
    "SimulationResponse",
    "SketchAnalysisRequest",
    "SketchAnalysisResponse",
    "ErrorResponse",
    "HealthResponse",
    "ValidationRequest",
    "ValidationResponse",
    "LoginRequest",
    "LoginResponse"
]