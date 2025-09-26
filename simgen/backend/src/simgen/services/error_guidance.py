"""
Progressive Error Guidance System
Replaces generic "Generation failed" messages with helpful, educational feedback.

Addresses core UX problem: Users get frustrated by unhelpful error messages
and resort to trial-and-error instead of learning what makes good sketches.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .cv_simplified import SimplifiedCVResult, DetectedObject

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for progressive disclosure"""
    INFO = "info"           # Helpful tips, not blocking
    WARNING = "warning"     # Issues that reduce quality
    ERROR = "error"         # Blocking issues that prevent generation
    CRITICAL = "critical"   # System errors


class ErrorCategory(Enum):
    """Categories of errors for targeted guidance"""
    SKETCH_QUALITY = "sketch_quality"
    OBJECT_DETECTION = "object_detection"
    PHYSICS_INTERPRETATION = "physics_interpretation"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorGuidance:
    """Structured error guidance with visual and textual help"""
    severity: ErrorSeverity
    category: ErrorCategory
    title: str
    message: str
    visual_feedback: Optional[Dict[str, Any]]  # Overlay data for canvas
    suggestions: List[str]
    learning_tips: List[str]
    example_fixes: List[Dict[str, Any]]
    retry_recommended: bool
    confidence_impact: float  # How much this reduces overall confidence


@dataclass
class ProgressiveErrorResult:
    """Complete error analysis with progressive disclosure"""
    can_proceed: bool
    overall_confidence: float
    primary_issue: Optional[ErrorGuidance]
    secondary_issues: List[ErrorGuidance]
    visual_overlays: List[Dict[str, Any]]
    success_path: List[str]  # Steps to fix the issues


class ProgressiveErrorAnalyzer:
    """
    Analyzes sketches and provides progressive, educational error guidance.

    Instead of "Generation failed", provides:
    - Visual overlays showing what was/wasn't detected
    - Specific suggestions for improvement
    - Learning tips to help users succeed
    - Progressive disclosure from simple to detailed feedback
    """

    def __init__(self):
        self.confidence_thresholds = {
            "critical": 0.1,    # Below this: critical errors
            "error": 0.3,       # Below this: blocking errors
            "warning": 0.6,     # Below this: quality warnings
            "info": 0.8         # Below this: helpful tips
        }

    async def analyze_sketch_errors(
        self,
        cv_result: SimplifiedCVResult,
        original_image_size: Tuple[int, int] = (800, 600)
    ) -> ProgressiveErrorResult:
        """
        Analyze CV result and provide progressive error guidance.

        Args:
            cv_result: Result from simplified CV pipeline
            original_image_size: Canvas dimensions for visual overlays

        Returns:
            Comprehensive error guidance with visual feedback
        """
        errors = []
        visual_overlays = []

        # Check for critical system errors first
        if cv_result.confidence == 0.0:
            critical_error = self._create_system_error(cv_result)
            return ProgressiveErrorResult(
                can_proceed=False,
                overall_confidence=0.0,
                primary_issue=critical_error,
                secondary_issues=[],
                visual_overlays=[],
                success_path=["Check image quality", "Try again", "Contact support if issue persists"]
            )

        # Analyze object detection issues
        object_errors, object_overlays = self._analyze_object_detection(cv_result, original_image_size)
        errors.extend(object_errors)
        visual_overlays.extend(object_overlays)

        # Analyze sketch quality issues
        quality_errors, quality_overlays = self._analyze_sketch_quality(cv_result, original_image_size)
        errors.extend(quality_errors)
        visual_overlays.extend(quality_overlays)

        # Analyze physics interpretation issues
        physics_errors = self._analyze_physics_interpretation(cv_result)
        errors.extend(physics_errors)

        # Sort errors by severity
        errors.sort(key=lambda e: self._severity_priority(e.severity))

        # Determine if generation can proceed
        blocking_errors = [e for e in errors if e.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]]
        can_proceed = len(blocking_errors) == 0 and cv_result.confidence > self.confidence_thresholds["error"]

        # Create success path
        success_path = self._create_success_path(errors, cv_result)

        return ProgressiveErrorResult(
            can_proceed=can_proceed,
            overall_confidence=cv_result.confidence,
            primary_issue=errors[0] if errors else None,
            secondary_issues=errors[1:] if len(errors) > 1 else [],
            visual_overlays=visual_overlays,
            success_path=success_path
        )

    def _analyze_object_detection(
        self,
        cv_result: SimplifiedCVResult,
        image_size: Tuple[int, int]
    ) -> Tuple[List[ErrorGuidance], List[Dict[str, Any]]]:
        """Analyze object detection issues and create visual overlays"""
        errors = []
        overlays = []

        if len(cv_result.objects) == 0:
            # No objects detected - critical issue
            error = ErrorGuidance(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.OBJECT_DETECTION,
                title="No Objects Detected",
                message="I can't see any clear shapes in your sketch. Physics simulations need recognizable objects.",
                visual_feedback={
                    "type": "no_detection_overlay",
                    "color": "red",
                    "opacity": 0.3
                },
                suggestions=[
                    "Draw larger, clearer shapes",
                    "Use darker lines with more contrast",
                    "Make sure shapes are complete and closed",
                    "Try drawing basic shapes like circles or rectangles first"
                ],
                learning_tips=[
                    "The system works best with simple, clear geometric shapes",
                    "Think of objects that can move: balls, blocks, rods",
                    "Avoid complex artwork - focus on physics objects"
                ],
                example_fixes=[
                    {
                        "description": "Draw a circle for a ball",
                        "sketch_hint": "Large circle with thick outline"
                    },
                    {
                        "description": "Draw a rectangle for a block",
                        "sketch_hint": "Rectangle at least 50x50 pixels"
                    }
                ],
                retry_recommended=True,
                confidence_impact=-0.8
            )
            errors.append(error)

            # Visual overlay: highlight entire canvas with suggestion
            overlays.append({
                "type": "full_canvas_message",
                "message": "Try drawing a simple circle or rectangle here",
                "color": "red",
                "position": {"x": image_size[0] // 2, "y": image_size[1] // 2}
            })

        elif len(cv_result.objects) == 1:
            # Only one object - physics needs interactions
            obj = cv_result.objects[0]

            error = ErrorGuidance(
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.PHYSICS_INTERPRETATION,
                title="Single Object Detected",
                message=f"I see a {obj.object_type.value}, but physics simulations work best with multiple interacting objects.",
                visual_feedback={
                    "type": "single_object_highlight",
                    "bbox": obj.bbox,
                    "color": "orange"
                },
                suggestions=[
                    "Add another object for interaction",
                    "Draw a surface or ramp for the object to interact with",
                    "Add a pivot point or constraint",
                    "Consider drawing a simple system like ball-and-ramp"
                ],
                learning_tips=[
                    "Physics is about interactions between objects",
                    "Even simple systems need at least 2 elements",
                    "Ground/surfaces count as physics objects too"
                ],
                example_fixes=[
                    {
                        "description": "Add a ramp below the ball",
                        "sketch_hint": "Draw angled line from upper left to lower right"
                    },
                    {
                        "description": "Add another ball for collision",
                        "sketch_hint": "Draw second circle nearby"
                    }
                ],
                retry_recommended=False,
                confidence_impact=-0.2
            )
            errors.append(error)

            # Visual overlay: highlight the single object and suggest additions
            overlays.append({
                "type": "object_highlight",
                "bbox": obj.bbox,
                "color": "orange",
                "message": f"Good {obj.object_type.value}! Now add another object."
            })

        else:
            # Multiple objects detected - check quality
            low_confidence_objects = [obj for obj in cv_result.objects if obj.confidence < 0.5]

            if low_confidence_objects:
                error = ErrorGuidance(
                    severity=ErrorSeverity.WARNING,
                    category=ErrorCategory.OBJECT_DETECTION,
                    title="Some Objects Unclear",
                    message=f"I detected {len(cv_result.objects)} objects, but {len(low_confidence_objects)} are unclear.",
                    visual_feedback={
                        "type": "confidence_overlay",
                        "objects": [(obj.bbox, obj.confidence) for obj in cv_result.objects]
                    },
                    suggestions=[
                        "Redraw unclear objects with thicker lines",
                        "Make shapes more geometric and regular",
                        "Ensure shapes are completely closed",
                        "Remove any stray marks or scribbles"
                    ],
                    learning_tips=[
                        "Higher confidence objects work better in physics",
                        "Clean, simple shapes are easier for the system to understand",
                        "Confidence above 70% is ideal for physics simulation"
                    ],
                    example_fixes=[
                        {
                            "description": "Redraw circles to be more round",
                            "sketch_hint": "Use smooth, continuous strokes"
                        }
                    ],
                    retry_recommended=True,
                    confidence_impact=-0.1 * len(low_confidence_objects)
                )
                errors.append(error)

                # Visual overlay: show confidence levels
                for obj in cv_result.objects:
                    color = "green" if obj.confidence > 0.7 else "orange" if obj.confidence > 0.4 else "red"
                    overlays.append({
                        "type": "confidence_indicator",
                        "bbox": obj.bbox,
                        "confidence": obj.confidence,
                        "color": color,
                        "label": f"{obj.object_type.value} ({obj.confidence:.0%})"
                    })

        return errors, overlays

    def _analyze_sketch_quality(
        self,
        cv_result: SimplifiedCVResult,
        image_size: Tuple[int, int]
    ) -> Tuple[List[ErrorGuidance], List[Dict[str, Any]]]:
        """Analyze overall sketch quality issues"""
        errors = []
        overlays = []

        # Check for very low overall confidence
        if cv_result.confidence < self.confidence_thresholds["error"]:
            error = ErrorGuidance(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SKETCH_QUALITY,
                title="Sketch Quality Too Low",
                message=f"Overall confidence is {cv_result.confidence:.0%}. The sketch needs improvement before physics generation.",
                visual_feedback={
                    "type": "quality_overlay",
                    "confidence": cv_result.confidence
                },
                suggestions=[
                    "Use thicker, darker lines",
                    "Make shapes larger and clearer",
                    "Remove unnecessary details or scribbles",
                    "Draw basic geometric shapes (circles, rectangles, lines)"
                ],
                learning_tips=[
                    "Think of your sketch as a blueprint for physics",
                    "Simple, clear shapes work better than artistic details",
                    "The system needs to recognize objects to simulate them"
                ],
                example_fixes=[
                    {
                        "description": "Start with a simple ball and ramp",
                        "sketch_hint": "Circle at top, angled line below"
                    }
                ],
                retry_recommended=True,
                confidence_impact=0.0  # Already accounted for
            )
            errors.append(error)

        elif cv_result.confidence < self.confidence_thresholds["warning"]:
            error = ErrorGuidance(
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.SKETCH_QUALITY,
                title="Sketch Could Be Clearer",
                message=f"Confidence is {cv_result.confidence:.0%}. Your sketch will work, but improvements could make it better.",
                visual_feedback={
                    "type": "improvement_suggestions",
                    "confidence": cv_result.confidence
                },
                suggestions=[
                    "Make lines thicker and more consistent",
                    "Ensure shapes are properly closed",
                    "Remove any stray marks",
                    "Make important objects larger"
                ],
                learning_tips=[
                    "Higher confidence leads to more accurate physics",
                    "Clean sketches produce better simulations"
                ],
                example_fixes=[],
                retry_recommended=False,
                confidence_impact=0.0
            )
            errors.append(error)

        return errors, overlays

    def _analyze_physics_interpretation(self, cv_result: SimplifiedCVResult) -> List[ErrorGuidance]:
        """Analyze physics interpretation issues"""
        errors = []

        physics_objects = cv_result.physics_objects
        if len(physics_objects) <= 1:  # Only ground plane
            error = ErrorGuidance(
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.PHYSICS_INTERPRETATION,
                title="Limited Physics Interaction",
                message="I can see your objects, but there's limited potential for interesting physics interactions.",
                visual_feedback=None,
                suggestions=[
                    "Add connections between objects (draw lines)",
                    "Create height differences for gravity effects",
                    "Add springs (draw zigzag lines)",
                    "Label parts with physics terms (spring, pivot, etc.)"
                ],
                learning_tips=[
                    "Physics is about forces and interactions",
                    "Height differences create potential energy",
                    "Connections create constraints and joints"
                ],
                example_fixes=[
                    {
                        "description": "Draw a line connecting two objects",
                        "sketch_hint": "Straight line between object centers"
                    },
                    {
                        "description": "Add a 'spring' label to a zigzag line",
                        "sketch_hint": "Text annotation helps recognition"
                    }
                ],
                retry_recommended=False,
                confidence_impact=-0.1
            )
            errors.append(error)

        return errors

    def _create_system_error(self, cv_result: SimplifiedCVResult) -> ErrorGuidance:
        """Create guidance for system errors"""
        return ErrorGuidance(
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM_ERROR,
            title="Analysis Failed",
            message="The system encountered an error while analyzing your sketch.",
            visual_feedback=None,
            suggestions=[
                "Check that your image is valid",
                "Try uploading the image again",
                "Ensure the image isn't corrupted",
                "Contact support if the problem persists"
            ],
            learning_tips=[
                "System errors are rare and usually temporary",
                "Your sketch data might be corrupted"
            ],
            example_fixes=[],
            retry_recommended=True,
            confidence_impact=0.0
        )

    def _create_success_path(
        self,
        errors: List[ErrorGuidance],
        cv_result: SimplifiedCVResult
    ) -> List[str]:
        """Create a step-by-step path to success"""
        if not errors:
            return ["Your sketch looks great! Click 'Generate Physics' to continue."]

        success_steps = []

        # Group errors by severity
        critical_errors = [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        blocking_errors = [e for e in errors if e.severity == ErrorSeverity.ERROR]
        warnings = [e for e in errors if e.severity == ErrorSeverity.WARNING]

        if critical_errors:
            success_steps.extend([
                "Fix system errors first",
                "Check your image and try again",
                "Contact support if issues persist"
            ])
        elif blocking_errors:
            success_steps.append("Address the blocking issues:")
            for error in blocking_errors:
                success_steps.extend(error.suggestions[:2])  # Top 2 suggestions
        elif warnings:
            success_steps.append("Optional improvements for better results:")
            for warning in warnings:
                success_steps.extend(warning.suggestions[:1])  # Top suggestion
            success_steps.append("You can proceed with current sketch if desired")

        success_steps.append("Click 'Generate Physics' when ready")
        return success_steps

    def _severity_priority(self, severity: ErrorSeverity) -> int:
        """Get priority order for error severity"""
        return {
            ErrorSeverity.CRITICAL: 0,
            ErrorSeverity.ERROR: 1,
            ErrorSeverity.WARNING: 2,
            ErrorSeverity.INFO: 3
        }[severity]


# Factory function for dependency injection
def create_error_analyzer() -> ProgressiveErrorAnalyzer:
    """Factory function for creating error analyzer"""
    return ProgressiveErrorAnalyzer()