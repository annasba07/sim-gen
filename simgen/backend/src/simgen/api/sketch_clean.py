"""
Clean Sketch API endpoints with proper separation of concerns.
Thin controllers that delegate to vision services.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Optional, Dict, Any

from ..core.interfaces import ISketchAnalyzer, ICacheService
from ..core.container import container
from ..core.validation import RequestValidator, ValidationError
from ..core.circuit_breaker import cv_circuit_breaker
from ..models.schemas import SketchAnalysisResponse

router = APIRouter(prefix="/api/v2/sketch", tags=["sketch"])


def get_sketch_analyzer() -> ISketchAnalyzer:
    """Dependency injection for sketch analyzer."""
    return container.get(ISketchAnalyzer)


def get_cache_service() -> ICacheService:
    """Dependency injection for cache service."""
    return container.get(ICacheService)


@router.post("/analyze", response_model=SketchAnalysisResponse)
async def analyze_sketch(
    file: UploadFile = File(...),
    user_text: Optional[str] = None,
    analyzer: ISketchAnalyzer = Depends(get_sketch_analyzer),
    cache: ICacheService = Depends(get_cache_service)
) -> SketchAnalysisResponse:
    """
    Analyze a sketch image and extract physics information.

    Thin controller that:
    1. Validates input
    2. Checks cache
    3. Delegates to analyzer service
    4. Handles circuit breaker
    5. Returns response
    """
    try:
        # Read and validate image data
        image_data = await file.read()
        validated_image = RequestValidator.validate_image_data(
            image_data.decode('latin-1') if isinstance(image_data, bytes) else image_data
        )

        # Validate text if provided
        if user_text:
            user_text = RequestValidator.validate_text_prompt(user_text)

        # Generate cache key
        import hashlib
        image_hash = hashlib.sha256(validated_image).hexdigest()
        cache_key = f"sketch:{image_hash}:{user_text or 'none'}"

        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            return SketchAnalysisResponse(
                success=True,
                analysis=cached_result,
                cache_hit=True
            )

        # Use circuit breaker for CV analysis
        analysis_result = await cv_circuit_breaker.call(
            analyzer.analyze,
            validated_image,
            user_text
        )

        # Cache successful result
        await cache.set(cache_key, analysis_result, ttl=3600)

        return SketchAnalysisResponse(
            success=True,
            analysis=analysis_result,
            cache_hit=False
        )

    except ValidationError as e:
        # Input validation error
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Service error (circuit breaker open, etc.)
        if "Circuit breaker is OPEN" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Sketch analysis service temporarily unavailable"
            )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/extract-shapes")
async def extract_shapes(
    file: UploadFile = File(...),
    analyzer: ISketchAnalyzer = Depends(get_sketch_analyzer)
) -> Dict[str, Any]:
    """
    Extract geometric shapes from sketch.

    Thin controller for shape extraction only.
    """
    try:
        # Read and validate image
        image_data = await file.read()
        validated_image = RequestValidator.validate_image_data(
            image_data.decode('latin-1') if isinstance(image_data, bytes) else image_data
        )

        # Delegate to analyzer (shape extraction only)
        # This would call a specific method on the analyzer
        shapes = await analyzer.extract_shapes(validated_image)

        return {
            "success": True,
            "shapes": shapes
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shape extraction failed: {str(e)}")


@router.post("/enhance")
async def enhance_sketch_description(
    sketch_data: Dict[str, Any],
    analyzer: ISketchAnalyzer = Depends(get_sketch_analyzer)
) -> Dict[str, Any]:
    """
    Enhance sketch analysis with AI.

    Thin controller for AI enhancement.
    """
    try:
        # Validate input
        if "image" not in sketch_data:
            raise ValidationError("Image data required")

        # Delegate to AI enhancement service
        # This would call IAIService for enhancement
        enhanced = await analyzer.enhance(sketch_data)

        return {
            "success": True,
            "enhanced_description": enhanced
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@router.get("/health")
async def sketch_health_check() -> Dict[str, str]:
    """
    Health check for sketch/vision service.

    Verifies CV pipeline is responsive.
    """
    # Check circuit breaker status
    breaker_status = "open" if cv_circuit_breaker.is_open else "closed"

    return {
        "status": "healthy" if breaker_status == "closed" else "degraded",
        "service": "sketch",
        "circuit_breaker": breaker_status
    }