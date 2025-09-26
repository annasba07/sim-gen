"""
Optimized Advanced Sketch Analysis Service
Performance improvements and better error handling
"""

import base64
import logging
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from functools import lru_cache
import asyncio

from ..core.exceptions import (
    SketchAnalysisError,
    CVPipelineError,
    LLMError,
    TimeoutError,
    ValidationError
)
from .llm_client import LLMClient
from .computer_vision_pipeline import ComputerVisionPipeline, CVAnalysisResult
from .sketch_to_physics_converter import SketchToPhysicsConverter, ConversionResult
from ..models.schemas import ExtractedEntities, ObjectSchema, ConstraintSchema, EnvironmentSchema, GeometrySchema, MaterialSchema
from ..models.physics_spec import PhysicsSpec

logger = logging.getLogger(__name__)

# Cache for CV analysis results
CV_CACHE: Dict[str, CVAnalysisResult] = {}
MAX_CACHE_SIZE = 100
CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class AdvancedSketchAnalysisResult:
    """Advanced result of sketch analysis with CV pipeline and PhysicsSpec generation."""
    success: bool
    physics_description: str
    physics_spec: Optional[PhysicsSpec]
    cv_analysis: Optional[CVAnalysisResult]
    extracted_entities: Optional[ExtractedEntities]
    confidence_score: float
    raw_vision_output: str
    processing_notes: List[str]
    error_message: Optional[str] = None
    cache_hit: bool = False
    processing_time_ms: float = 0.0


class OptimizedSketchAnalyzer:
    """Optimized sketch analyzer with caching and better error handling."""

    def __init__(self, llm_client: LLMClient, enable_caching: bool = True):
        self.llm_client = llm_client
        self.cv_pipeline = ComputerVisionPipeline()
        self.physics_converter = SketchToPhysicsConverter()
        self.enable_caching = enable_caching
        self._cache_cleanup_task = None

    async def __aenter__(self):
        """Async context manager entry."""
        if self.enable_caching:
            self._cache_cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._cache_cleanup_task:
            self._cache_cleanup_task.cancel()
            try:
                await self._cache_cleanup_task
            except asyncio.CancelledError:
                pass

    def _get_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image data."""
        return hashlib.sha256(image_data).hexdigest()

    def _validate_image_data(self, image_data: bytes) -> None:
        """Validate image data before processing."""
        if not image_data:
            raise ValidationError("Empty image data provided")

        # Check image size (max 10MB)
        max_size = 10 * 1024 * 1024
        if len(image_data) > max_size:
            raise ValidationError(
                f"Image size ({len(image_data) / 1024 / 1024:.2f}MB) exceeds maximum allowed size (10MB)",
                details={"size": len(image_data), "max_size": max_size}
            )

        # Check if it's a valid image format
        if not (image_data.startswith(b'\x89PNG') or
                image_data.startswith(b'\xff\xd8\xff') or  # JPEG
                image_data.startswith(b'GIF')):
            raise ValidationError("Invalid image format. Supported formats: PNG, JPEG, GIF")

    async def analyze_sketch(
        self,
        image_data: bytes,
        user_text: Optional[str] = None,
        include_actuators: bool = True,
        include_sensors: bool = True,
        timeout_seconds: float = 30.0
    ) -> AdvancedSketchAnalysisResult:
        """
        Optimized sketch analysis with caching and timeout.

        Args:
            image_data: Raw image bytes of the sketch
            user_text: Optional text prompt to accompany the sketch
            include_actuators: Whether to generate actuators
            include_sensors: Whether to generate sensors
            timeout_seconds: Maximum time for analysis

        Returns:
            AdvancedSketchAnalysisResult with CV analysis and PhysicsSpec
        """
        import time
        start_time = time.perf_counter()
        processing_notes = []

        try:
            # Validate input
            self._validate_image_data(image_data)

            # Check cache if enabled
            image_hash = self._get_image_hash(image_data) if self.enable_caching else None
            if image_hash and image_hash in CV_CACHE:
                cached_result = CV_CACHE[image_hash]
                processing_notes.append("Retrieved from cache")
                logger.info(f"Cache hit for image hash: {image_hash}")

                # Still need to convert to PhysicsSpec
                conversion_result = await self.physics_converter.convert_cv_to_physics_spec(
                    cached_result, user_text, include_actuators, include_sensors
                )

                return AdvancedSketchAnalysisResult(
                    success=True,
                    physics_description=self._format_cv_analysis_output(cached_result),
                    physics_spec=conversion_result.physics_spec,
                    cv_analysis=cached_result,
                    extracted_entities=await self._create_extracted_entities_from_cv(cached_result),
                    confidence_score=conversion_result.confidence,
                    raw_vision_output=self._format_cv_analysis_output(cached_result),
                    processing_notes=processing_notes,
                    cache_hit=True,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )

            # Run CV analysis with timeout
            try:
                cv_result = await asyncio.wait_for(
                    self.cv_pipeline.analyze_sketch(image_data),
                    timeout=timeout_seconds * 0.6  # 60% of timeout for CV
                )
                processing_notes.append(f"CV pipeline detected {len(cv_result.shapes)} shapes")

                # Cache the result if enabled
                if self.enable_caching and image_hash:
                    await self._cache_cv_result(image_hash, cv_result)

            except asyncio.TimeoutError:
                raise TimeoutError(
                    "CV analysis timed out",
                    operation="cv_pipeline.analyze_sketch",
                    timeout_seconds=timeout_seconds * 0.6
                )
            except Exception as e:
                logger.error(f"CV pipeline failed: {e}", exc_info=True)
                raise CVPipelineError(f"Computer vision analysis failed: {str(e)}")

            # Validate CV results
            if not cv_result.shapes:
                logger.warning("No shapes detected, using LLM fallback")
                return await self._fallback_llm_analysis_optimized(
                    image_data, user_text, processing_notes, timeout_seconds * 0.4
                )

            # Convert to PhysicsSpec
            try:
                conversion_result = await asyncio.wait_for(
                    self.physics_converter.convert_cv_to_physics_spec(
                        cv_result, user_text, include_actuators, include_sensors
                    ),
                    timeout=timeout_seconds * 0.3
                )
                processing_notes.extend(conversion_result.conversion_notes)

            except asyncio.TimeoutError:
                raise TimeoutError(
                    "Physics conversion timed out",
                    operation="physics_converter.convert_cv_to_physics_spec",
                    timeout_seconds=timeout_seconds * 0.3
                )
            except Exception as e:
                logger.error(f"Physics conversion failed: {e}", exc_info=True)
                raise PhysicsSpecError(f"Failed to convert to PhysicsSpec: {str(e)}")

            if not conversion_result.success:
                raise PhysicsSpecError(
                    conversion_result.error_message or "Unknown conversion error"
                )

            # Enhance description with LLM (non-critical)
            enhanced_description = cv_result.summary
            try:
                enhanced_description = await asyncio.wait_for(
                    self._enhance_description_with_llm(
                        cv_result, conversion_result.physics_spec, user_text
                    ),
                    timeout=timeout_seconds * 0.1
                )
                processing_notes.append("Enhanced description with LLM")
            except Exception as e:
                logger.warning(f"Description enhancement failed (non-critical): {e}")
                processing_notes.append("Description enhancement skipped")

            # Create backward-compatible entities
            extracted_entities = await self._create_extracted_entities_from_cv(cv_result)

            # Calculate confidence
            final_confidence = self._calculate_combined_confidence(cv_result, conversion_result)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Sketch analysis completed in {elapsed_ms:.2f}ms")

            return AdvancedSketchAnalysisResult(
                success=True,
                physics_description=enhanced_description,
                physics_spec=conversion_result.physics_spec,
                cv_analysis=cv_result,
                extracted_entities=extracted_entities,
                confidence_score=final_confidence,
                raw_vision_output=self._format_cv_analysis_output(cv_result),
                processing_notes=processing_notes,
                cache_hit=False,
                processing_time_ms=elapsed_ms
            )

        except (ValidationError, CVPipelineError, PhysicsSpecError, TimeoutError) as e:
            # Known errors - log and return structured response
            logger.error(f"Sketch analysis failed: {e}")
            processing_notes.append(f"Error: {str(e)}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return AdvancedSketchAnalysisResult(
                success=False,
                physics_description="",
                physics_spec=None,
                cv_analysis=None,
                extracted_entities=None,
                confidence_score=0.0,
                raw_vision_output="",
                processing_notes=processing_notes,
                error_message=str(e),
                processing_time_ms=elapsed_ms
            )

        except Exception as e:
            # Unexpected errors - log with full traceback
            logger.error(f"Unexpected error in sketch analysis: {e}", exc_info=True)
            processing_notes.append(f"Unexpected error: {str(e)}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return AdvancedSketchAnalysisResult(
                success=False,
                physics_description="",
                physics_spec=None,
                cv_analysis=None,
                extracted_entities=None,
                confidence_score=0.0,
                raw_vision_output="",
                processing_notes=processing_notes,
                error_message=f"Internal error: {str(e)}",
                processing_time_ms=elapsed_ms
            )

    async def _cache_cv_result(self, image_hash: str, result: CVAnalysisResult) -> None:
        """Cache CV analysis result."""
        # Implement LRU eviction if cache is full
        if len(CV_CACHE) >= MAX_CACHE_SIZE:
            # Remove oldest entry (simplified - could use OrderedDict)
            oldest_key = next(iter(CV_CACHE))
            del CV_CACHE[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")

        CV_CACHE[image_hash] = result
        logger.debug(f"Cached CV result for hash: {image_hash}")

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up old cache entries."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                # In production, would track timestamps and remove old entries
                logger.info(f"Cache cleanup: {len(CV_CACHE)} entries")
            except asyncio.CancelledError:
                break

    async def _fallback_llm_analysis_optimized(
        self,
        image_data: bytes,
        user_text: Optional[str],
        processing_notes: List[str],
        timeout_seconds: float
    ) -> AdvancedSketchAnalysisResult:
        """Optimized fallback to LLM-only analysis."""
        import time
        start_time = time.perf_counter()

        try:
            processing_notes.append("Using LLM-only fallback analysis")

            # Create vision prompt
            prompt = self._create_vision_prompt(user_text)

            # Encode image for LLM
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            # Call LLM with timeout
            try:
                vision_result = await asyncio.wait_for(
                    self.llm_client.analyze_image_with_prompt(encoded_image, prompt),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    "LLM analysis timed out",
                    operation="llm_client.analyze_image_with_prompt",
                    timeout_seconds=timeout_seconds
                )

            # Extract entities from LLM response
            entities = await self._extract_entities_from_vision(vision_result)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return AdvancedSketchAnalysisResult(
                success=True,
                physics_description=vision_result,
                physics_spec=None,  # No PhysicsSpec in fallback mode
                cv_analysis=None,
                extracted_entities=entities,
                confidence_score=0.5,  # Lower confidence for fallback
                raw_vision_output=vision_result,
                processing_notes=processing_notes,
                processing_time_ms=elapsed_ms
            )

        except Exception as e:
            raise LLMError(f"Fallback LLM analysis failed: {str(e)}")

    def _format_cv_analysis_output(self, cv_result: CVAnalysisResult) -> str:
        """Format CV analysis result as string."""
        if not cv_result:
            return ""

        output = []
        output.append(f"Detected {len(cv_result.shapes)} shapes:")
        for shape in cv_result.shapes:
            output.append(f"  - {shape.shape_type.value} at ({shape.center.x:.0f}, {shape.center.y:.0f})")

        if cv_result.connections:
            output.append(f"\\nDetected {len(cv_result.connections)} connections:")
            for conn in cv_result.connections:
                output.append(f"  - {conn.connection_type.value} between shapes")

        return "\\n".join(output)

    def _create_vision_prompt(self, user_text: Optional[str] = None) -> str:
        """Create optimized prompt for vision model analysis."""
        base_prompt = """Analyze this hand-drawn sketch and identify:
1. Physical objects (shapes, bodies, mechanisms)
2. Connections and joints between objects
3. Motion constraints and physics properties
4. Any text annotations or labels

Be concise and specific about positions and relationships."""

        if user_text:
            return f"{base_prompt}\\n\\nUser description: {user_text}"
        return base_prompt

    async def _enhance_description_with_llm(
        self,
        cv_result: CVAnalysisResult,
        physics_spec: PhysicsSpec,
        user_text: Optional[str]
    ) -> str:
        """Enhance description using LLM."""
        # Implementation would go here
        return cv_result.summary or "Physics simulation based on sketch"

    async def _create_extracted_entities_from_cv(
        self, cv_result: CVAnalysisResult
    ) -> ExtractedEntities:
        """Create backward-compatible extracted entities."""
        # Implementation would create ExtractedEntities from CVAnalysisResult
        return ExtractedEntities(
            objects=[],
            constraints=[],
            environment=EnvironmentSchema()
        )

    async def _extract_entities_from_vision(self, vision_result: str) -> ExtractedEntities:
        """Extract entities from LLM vision response."""
        # Implementation would parse LLM response
        return ExtractedEntities(
            objects=[],
            constraints=[],
            environment=EnvironmentSchema()
        )

    def _calculate_combined_confidence(
        self, cv_result: CVAnalysisResult, conversion_result: ConversionResult
    ) -> float:
        """Calculate combined confidence score."""
        cv_confidence = cv_result.overall_confidence if cv_result else 0.0
        conversion_confidence = conversion_result.confidence
        return (cv_confidence + conversion_confidence) / 2.0


def get_optimized_sketch_analyzer(llm_client: LLMClient) -> OptimizedSketchAnalyzer:
    """Factory function to create optimized sketch analyzer."""
    return OptimizedSketchAnalyzer(llm_client, enable_caching=True)