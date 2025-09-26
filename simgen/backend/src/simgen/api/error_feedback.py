"""
Progressive Error Feedback API
Provides detailed, educational error analysis instead of generic "Generation failed" messages.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from dataclasses import asdict

from ..core.container import container
from ..core.interfaces import IComputerVisionPipeline
from ..services.error_guidance import ProgressiveErrorAnalyzer, create_error_analyzer
from ..services.cv_simplified import SimplifiedCVResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/error-feedback", tags=["error-feedback"])


@router.post("/analyze-sketch")
async def analyze_sketch_errors(
    image: UploadFile = File(...),
    cv_pipeline: IComputerVisionPipeline = Depends(lambda: container.get(IComputerVisionPipeline))
):
    """
    Analyze sketch and provide progressive error feedback.

    Replaces generic error messages with:
    - Visual overlays showing detection confidence
    - Specific suggestions for improvement
    - Educational tips for better sketches
    - Step-by-step guidance to success
    """
    try:
        # Read image data
        image_data = await image.read()

        # Analyze with CV pipeline
        cv_result: SimplifiedCVResult = await cv_pipeline.analyze_sketch(image_data)

        # Create error analyzer
        error_analyzer = create_error_analyzer()

        # Analyze errors and create guidance
        error_result = await error_analyzer.analyze_sketch_errors(cv_result)

        # Format response for frontend
        response = {
            "success": error_result.can_proceed,
            "overall_confidence": error_result.overall_confidence,
            "analysis": {
                "detected_objects": len(cv_result.objects),
                "physics_objects": len(cv_result.physics_objects),
                "processing_notes": cv_result.processing_notes
            },
            "feedback": {
                "primary_issue": asdict(error_result.primary_issue) if error_result.primary_issue else None,
                "secondary_issues": [asdict(issue) for issue in error_result.secondary_issues],
                "visual_overlays": error_result.visual_overlays,
                "success_path": error_result.success_path
            },
            "recommendations": _create_user_recommendations(error_result, cv_result)
        }

        return response

    except Exception as e:
        logger.error(f"Error analyzing sketch: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze sketch")


@router.post("/quick-feedback")
async def quick_feedback_check(
    sketch_data: Dict[str, Any],
    cv_pipeline: IComputerVisionPipeline = Depends(lambda: container.get(IComputerVisionPipeline))
):
    """
    Quick feedback for real-time sketch analysis.

    Used by the real-time feedback system for lightweight error checking.
    """
    try:
        # This would normally convert sketch_data to image
        # For now, simulate with mock data
        mock_image = b"mock_image_data"

        cv_result = await cv_pipeline.analyze_sketch(mock_image)
        error_analyzer = create_error_analyzer()
        error_result = await error_analyzer.analyze_sketch_errors(cv_result)

        # Return simplified feedback for real-time use
        return {
            "confidence": error_result.overall_confidence,
            "can_proceed": error_result.can_proceed,
            "quick_tip": error_result.success_path[0] if error_result.success_path else "Keep drawing!",
            "issue_count": len(error_result.secondary_issues) + (1 if error_result.primary_issue else 0)
        }

    except Exception as e:
        logger.error(f"Quick feedback failed: {e}")
        return {
            "confidence": 0.0,
            "can_proceed": False,
            "quick_tip": "Having trouble analyzing your sketch. Try making lines clearer.",
            "issue_count": 1
        }


@router.get("/examples/good-sketches")
async def get_good_sketch_examples():
    """Get examples of sketches that work well with the system."""
    return {
        "examples": [
            {
                "name": "Simple Pendulum",
                "description": "Clear circle connected to a line with fixed pivot",
                "why_it_works": [
                    "Distinct geometric shapes",
                    "Clear connections between objects",
                    "Obvious physics intent",
                    "Simple and unambiguous"
                ],
                "sketch_features": {
                    "shapes": ["circle", "line", "point"],
                    "connections": ["string attachment"],
                    "physics_concepts": ["gravity", "oscillation"]
                }
            },
            {
                "name": "Ball and Ramp",
                "description": "Circle positioned above an angled line",
                "why_it_works": [
                    "Clear height difference shows potential energy",
                    "Simple geometric shapes",
                    "Obvious motion path",
                    "Easy to understand intent"
                ],
                "sketch_features": {
                    "shapes": ["circle", "angled line", "ground line"],
                    "connections": ["contact surfaces"],
                    "physics_concepts": ["gravity", "rolling", "friction"]
                }
            }
        ],
        "general_principles": [
            "Use basic geometric shapes (circles, rectangles, lines)",
            "Make objects large enough to be clearly visible",
            "Show clear connections between interacting objects",
            "Keep sketches simple and focused on physics intent",
            "Use thick, dark lines for better detection"
        ]
    }


@router.get("/examples/problem-sketches")
async def get_problem_sketch_examples():
    """Get examples of common sketch problems and how to fix them."""
    return {
        "common_problems": [
            {
                "problem": "Too Artistic",
                "description": "Detailed drawings with shading, textures, or artistic elements",
                "why_it_fails": "System optimized for geometric shapes, not artistic details",
                "how_to_fix": [
                    "Focus on the physics objects only",
                    "Remove artistic details and shading",
                    "Use simple outlines instead of filled shapes",
                    "Think 'blueprint' not 'artwork'"
                ],
                "example_fix": "Instead of a detailed car, draw a rectangle with circles for wheels"
            },
            {
                "problem": "Lines Too Thin",
                "description": "Faint or very thin lines that are hard to detect",
                "why_it_fails": "Computer vision needs sufficient contrast to detect shapes",
                "how_to_fix": [
                    "Use thicker pen/brush settings",
                    "Draw with confident, bold strokes",
                    "Ensure good contrast with background",
                    "Avoid sketchy, light construction lines"
                ]
            },
            {
                "problem": "Incomplete Shapes",
                "description": "Shapes that aren't fully closed or connected",
                "why_it_fails": "System needs complete shapes to understand object boundaries",
                "how_to_fix": [
                    "Make sure circles are fully closed",
                    "Connect all corners of rectangles",
                    "Close gaps in lines and curves",
                    "Use continuous strokes when possible"
                ]
            },
            {
                "problem": "No Clear Physics Intent",
                "description": "Objects drawn without obvious interactions or motion",
                "why_it_fails": "Physics simulations need forces and interactions",
                "how_to_fix": [
                    "Show clear connections between objects",
                    "Create height differences for gravity effects",
                    "Add labels like 'spring', 'hinge', 'pivot'",
                    "Think about how objects will move and interact"
                ]
            }
        ],
        "quick_fixes": [
            "Make lines thicker and darker",
            "Use simple geometric shapes",
            "Show clear object connections",
            "Add text labels for unclear elements",
            "Remove unnecessary artistic details"
        ]
    }


def _create_user_recommendations(error_result, cv_result: SimplifiedCVResult) -> Dict[str, Any]:
    """Create user-friendly recommendations based on error analysis."""
    recommendations = {
        "next_steps": error_result.success_path,
        "confidence_breakdown": {
            "current": cv_result.confidence,
            "target": 0.7,
            "status": "good" if cv_result.confidence > 0.7 else "needs_improvement"
        },
        "object_analysis": {
            "detected": len(cv_result.objects),
            "minimum_recommended": 2,
            "ideal_range": "2-4 objects for interesting physics"
        }
    }

    # Add specific recommendations based on confidence
    if cv_result.confidence < 0.3:
        recommendations["priority"] = "Fix basic sketch quality first"
        recommendations["focus_areas"] = ["Line thickness", "Shape clarity", "Object completion"]
    elif cv_result.confidence < 0.7:
        recommendations["priority"] = "Improve object recognition"
        recommendations["focus_areas"] = ["Shape consistency", "Remove stray marks", "Larger objects"]
    else:
        recommendations["priority"] = "Enhance physics interactions"
        recommendations["focus_areas"] = ["Add connections", "Show forces", "Label elements"]

    return recommendations