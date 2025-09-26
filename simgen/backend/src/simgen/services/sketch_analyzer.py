"""
Advanced Sketch Analysis Service

This service uses computer vision and AI to interpret hand-drawn sketches
and convert them into structured PhysicsSpec objects for MuJoCo simulation.

Combines:
- Computer vision pipeline for shape/connection detection
- OCR for text annotations
- LLM enhancement for physics interpretation
- Direct PhysicsSpec generation
"""

import base64
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel
import asyncio

from .llm_client import LLMClient
from .computer_vision_pipeline import ComputerVisionPipeline, CVAnalysisResult
from .sketch_to_physics_converter import SketchToPhysicsConverter, ConversionResult
from ..models.schemas import ExtractedEntities, ObjectSchema, ConstraintSchema, EnvironmentSchema, GeometrySchema, MaterialSchema
from ..models.physics_spec import PhysicsSpec

logger = logging.getLogger(__name__)


@dataclass
class AdvancedSketchAnalysisResult:
    """Advanced result of sketch analysis with CV pipeline and PhysicsSpec generation."""
    success: bool
    physics_description: str
    physics_spec: Optional[PhysicsSpec]
    cv_analysis: Optional[CVAnalysisResult]
    extracted_entities: Optional[ExtractedEntities]  # For backward compatibility
    confidence_score: float
    raw_vision_output: str
    processing_notes: List[str]
    error_message: Optional[str] = None

# Backward compatibility alias
SketchAnalysisResult = AdvancedSketchAnalysisResult


class AdvancedSketchAnalyzer:
    """Advanced sketch analyzer using computer vision + AI pipeline."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.cv_pipeline = ComputerVisionPipeline()
        self.physics_converter = SketchToPhysicsConverter()
    
    async def analyze_sketch(
        self,
        image_data: bytes,
        user_text: Optional[str] = None,
        include_actuators: bool = True,
        include_sensors: bool = True
    ) -> AdvancedSketchAnalysisResult:
        """
        Advanced sketch analysis using computer vision + AI pipeline.

        Args:
            image_data: Raw image bytes of the sketch
            user_text: Optional text prompt to accompany the sketch
            include_actuators: Whether to generate actuators
            include_sensors: Whether to generate sensors

        Returns:
            AdvancedSketchAnalysisResult with CV analysis and PhysicsSpec
        """
        processing_notes = []

        try:
            # Step 1: Computer Vision Analysis
            logger.info("Starting computer vision analysis of sketch")
            processing_notes.append("Starting computer vision pipeline")

            cv_result = await self.cv_pipeline.analyze_sketch(image_data)

            if not cv_result.shapes:
                logger.warning("No shapes detected in sketch")
                # Fallback to LLM-only analysis
                return await self._fallback_llm_analysis(image_data, user_text, processing_notes)

            processing_notes.append(f"CV pipeline detected {len(cv_result.shapes)} shapes, {len(cv_result.connections)} connections")

            # Step 2: Convert to PhysicsSpec
            logger.info("Converting CV analysis to PhysicsSpec")
            conversion_result = await self.physics_converter.convert_cv_to_physics_spec(
                cv_result, user_text, include_actuators, include_sensors
            )

            processing_notes.extend(conversion_result.conversion_notes)

            if not conversion_result.success:
                logger.error(f"PhysicsSpec conversion failed: {conversion_result.error_message}")
                return await self._fallback_llm_analysis(image_data, user_text, processing_notes)

            # Step 3: Enhance with LLM analysis for better description
            logger.info("Enhancing description with LLM analysis")
            enhanced_description = await self._enhance_description_with_llm(
                cv_result, conversion_result.physics_spec, user_text
            )
            processing_notes.append("Enhanced physics description with LLM analysis")

            # Step 4: Create backward-compatible extracted entities
            extracted_entities = await self._create_extracted_entities_from_cv(cv_result)

            # Step 5: Calculate combined confidence
            final_confidence = self._calculate_combined_confidence(cv_result, conversion_result)

            return AdvancedSketchAnalysisResult(
                success=True,
                physics_description=enhanced_description,
                physics_spec=conversion_result.physics_spec,
                cv_analysis=cv_result,
                extracted_entities=extracted_entities,
                confidence_score=final_confidence,
                raw_vision_output=self._format_cv_analysis_output(cv_result),
                processing_notes=processing_notes
            )

        except Exception as e:
            logger.error(f"Advanced sketch analysis failed: {e}")
            processing_notes.append(f"Error in advanced analysis: {str(e)}")

            # Try fallback analysis
            try:
                return await self._fallback_llm_analysis(image_data, user_text, processing_notes)
            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")
                return AdvancedSketchAnalysisResult(
                    success=False,
                    physics_description="",
                    physics_spec=None,
                    cv_analysis=None,
                    extracted_entities=None,
                    confidence_score=0.0,
                    raw_vision_output="",
                    processing_notes=processing_notes,
                    error_message=f"Both advanced and fallback analysis failed: {str(e)}, {str(fallback_error)}"
                )
    
    def _create_vision_prompt(self, user_text: Optional[str] = None) -> str:
        """Create a detailed prompt for vision model analysis."""
        base_prompt = """
Analyze this hand-drawn sketch and identify all physics objects and their properties.

Focus on:
1. **Objects**: What physical objects do you see? (spheres, boxes, ramps, pendulums, springs, etc.)
2. **Positions**: Where are objects located relative to each other?
3. **Connections**: How are objects connected? (joints, constraints, contacts)
4. **Motion Intent**: What kind of motion or physics behavior is suggested?
5. **Materials**: Are different materials or properties implied by the drawing?
6. **Environment**: Is there a ground plane, walls, or other environmental elements?

Be very specific about:
- Object shapes and estimated sizes
- Spatial relationships and positions
- Any arrows or indicators showing motion/forces
- Connection points between objects
- Surface types and orientations

Describe what you see in detail, focusing on physics-relevant aspects.
"""
        
        if user_text:
            return f"{base_prompt}\n\nUser also provided this description: \"{user_text}\"\n\nCombine the visual analysis with the user's intent."
        
        return base_prompt
    
    async def _convert_to_physics_description(
        self, 
        vision_analysis: str, 
        user_text: Optional[str]
    ) -> str:
        """Convert raw vision analysis to structured physics description."""
        
        conversion_prompt = f"""
Based on this vision analysis of a hand-drawn sketch:

{vision_analysis}

{f'User intent: {user_text}' if user_text else ''}

Convert this into a detailed physics simulation description suitable for MuJoCo/physics engine generation.

Include:
1. **Scene Setup**: Environment, ground plane, lighting
2. **Objects**: Each physical body with properties (mass, material, size, position)
3. **Constraints**: Joints, springs, or connections between objects
4. **Initial Conditions**: Starting positions, velocities, or forces
5. **Physics Parameters**: Gravity, damping, friction values

Write this as a comprehensive physics scenario that captures the intent of the sketch.
Focus on realistic physics behavior and interesting dynamics.

Example format:
"A physics scene featuring [objects] positioned [where] with [relationships]. The [main object] should [behavior] while [constraints/interactions]. Environmental conditions include [gravity/friction/etc.]."

Make it detailed enough to generate a professional MuJoCo simulation.
"""
        
        try:
            return await self.llm_client.complete(
                prompt=conversion_prompt,
                temperature=0.2,
                max_tokens=2000
            )
        except Exception as e:
            logger.error(f"Physics description conversion failed: {e}")
            return f"Physics simulation based on sketch: {vision_analysis[:500]}..."
    
    async def _extract_physics_entities(
        self, 
        physics_description: str, 
        raw_analysis: str
    ) -> Optional[ExtractedEntities]:
        """Extract structured entities from physics description."""
        
        extraction_schema = {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["rigid_body", "soft_body", "robot", "vehicle"]},
                            "geometry": {
                                "type": "object",
                                "properties": {
                                    "shape": {"type": "string", "enum": ["box", "sphere", "cylinder", "mesh"]},
                                    "dimensions": {"type": "array", "items": {"type": "number"}}
                                },
                                "required": ["shape", "dimensions"]
                            },
                            "material": {
                                "type": "object", 
                                "properties": {
                                    "density": {"type": "number", "default": 1000.0},
                                    "friction": {"type": "number", "default": 0.5},
                                    "restitution": {"type": "number", "default": 0.1}
                                }
                            },
                            "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "orientation": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
                        },
                        "required": ["name", "type", "geometry", "material", "position"]
                    }
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["joint", "contact", "force"]},
                            "bodies": {"type": "array", "items": {"type": "string"}},
                            "parameters": {"type": "object", "additionalProperties": True}
                        },
                        "required": ["type", "bodies"]
                    }
                },
                "environment": {
                    "type": "object",
                    "properties": {
                        "gravity": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        "ground": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "friction": {"type": "number"},
                                "position": {"type": "array", "items": {"type": "number"}}
                            }
                        }
                    }
                }
            },
            "required": ["objects", "environment"]
        }
        
        extraction_prompt = f"""
Analyze this physics description from a hand-drawn sketch and extract structured entities:

PHYSICS DESCRIPTION:
{physics_description}

ORIGINAL VISION ANALYSIS:
{raw_analysis}

Extract all physics objects, constraints, and environment settings into the structured format.

For objects from sketches, make reasonable assumptions about:
- Size (relative to context, e.g., balls ~0.1-0.5m radius)  
- Position (based on sketch layout)
- Material (wood, metal, plastic based on visual cues)
- Mass (appropriate for size and material)

For environment:
- Always include gravity (typically [0, 0, -9.81])
- Add ground plane if objects rest on something
- Set appropriate friction values

Focus on creating a simulation that captures the sketch's intent.
"""
        
        try:
            result = await self.llm_client.complete_with_schema(
                prompt=extraction_prompt,
                schema=extraction_schema,
                temperature=0.1
            )
            
            # Convert to ExtractedEntities model
            objects = []
            for obj_data in result.get("objects", []):
                # Create nested geometry and material objects
                geometry = GeometrySchema(**obj_data.get("geometry", {}))
                material = MaterialSchema(**obj_data.get("material", {}))
                
                # Create the main object
                obj = ObjectSchema(
                    name=obj_data.get("name", "unnamed_object"),
                    type=obj_data.get("type", "rigid_body"),
                    geometry=geometry,
                    material=material,
                    position=obj_data.get("position", [0, 0, 0]),
                    orientation=obj_data.get("orientation", [0, 0, 0])
                )
                objects.append(obj)
            
            constraints = [
                ConstraintSchema(**const) for const in result.get("constraints", [])
            ]
            
            env_data = result.get("environment", {})
            environment = EnvironmentSchema(
                gravity=env_data.get("gravity", [0.0, 0.0, -9.81]),
                ground=env_data.get("ground", {}),
                boundaries=env_data.get("boundaries", {})
            )
            
            return ExtractedEntities(
                objects=objects,
                constraints=constraints,
                environment=environment
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return None
    
    def _calculate_confidence(
        self, 
        raw_analysis: str, 
        extracted_entities: Optional[ExtractedEntities]
    ) -> float:
        """Calculate confidence score based on analysis quality."""
        try:
            base_confidence = 0.5
            
            # Boost confidence based on detail level
            if len(raw_analysis) > 200:
                base_confidence += 0.2
            if "object" in raw_analysis.lower():
                base_confidence += 0.1
            if "position" in raw_analysis.lower() or "location" in raw_analysis.lower():
                base_confidence += 0.1
            
            # Boost based on extracted entities
            if extracted_entities:
                if len(extracted_entities.objects) > 0:
                    base_confidence += 0.1
                if len(extracted_entities.constraints) > 0:
                    base_confidence += 0.1
            
            return min(base_confidence, 1.0)
            
        except Exception:
            return 0.3


    async def _fallback_llm_analysis(
        self,
        image_data: bytes,
        user_text: Optional[str],
        processing_notes: List[str]
    ) -> AdvancedSketchAnalysisResult:
        """Fallback to LLM-only analysis when CV pipeline fails"""
        try:
            processing_notes.append("Falling back to LLM-only vision analysis")

            # Step 1: Analyze the sketch with vision model
            vision_prompt = self._create_vision_prompt(user_text)
            raw_analysis = await self.llm_client.analyze_image(
                image_data=image_data,
                prompt=vision_prompt,
                temperature=0.1
            )

            # Step 2: Convert vision analysis to structured physics description
            physics_description = await self._convert_to_physics_description(
                raw_analysis, user_text
            )

            # Step 3: Extract entities for simulation generation
            extracted_entities = await self._extract_physics_entities(
                physics_description, raw_analysis
            )

            # Step 4: Calculate confidence score
            confidence = self._calculate_confidence(raw_analysis, extracted_entities)

            processing_notes.append("Completed fallback LLM analysis")

            return AdvancedSketchAnalysisResult(
                success=True,
                physics_description=physics_description,
                physics_spec=None,  # No PhysicsSpec from LLM-only analysis
                cv_analysis=None,
                extracted_entities=extracted_entities,
                confidence_score=confidence * 0.7,  # Reduce confidence for fallback
                raw_vision_output=raw_analysis,
                processing_notes=processing_notes
            )

        except Exception as e:
            logger.error(f"Fallback LLM analysis failed: {e}")
            return AdvancedSketchAnalysisResult(
                success=False,
                physics_description="",
                physics_spec=None,
                cv_analysis=None,
                extracted_entities=None,
                confidence_score=0.0,
                raw_vision_output="",
                processing_notes=processing_notes,
                error_message=f"Fallback analysis failed: {str(e)}"
            )

    async def _enhance_description_with_llm(
        self,
        cv_result: CVAnalysisResult,
        physics_spec: PhysicsSpec,
        user_text: Optional[str]
    ) -> str:
        """Enhance CV analysis with LLM-generated physics description"""
        try:
            # Create summary of CV findings
            cv_summary = self._summarize_cv_analysis(cv_result)

            enhancement_prompt = f"""
Based on computer vision analysis of a hand-drawn sketch:

DETECTED SHAPES:
{cv_summary['shapes']}

DETECTED CONNECTIONS:
{cv_summary['connections']}

DETECTED TEXT:
{cv_summary['text']}

{f'USER INTENT: {user_text}' if user_text else ''}

Create a comprehensive physics simulation description that:
1. Explains what the sketch represents
2. Describes the expected physics behavior
3. Mentions key interactions between objects
4. Includes realistic physical properties

Write in a clear, engaging style suitable for physics simulation documentation.
Focus on the intended dynamics and realistic behavior.
"""

            enhanced_description = await self.llm_client.complete(
                prompt=enhancement_prompt,
                temperature=0.3,
                max_tokens=1500
            )

            return enhanced_description.strip()

        except Exception as e:
            logger.error(f"LLM description enhancement failed: {e}")
            # Fallback to CV-based description
            return self._create_basic_description_from_cv(cv_result, user_text)

    def _summarize_cv_analysis(self, cv_result: CVAnalysisResult) -> Dict[str, str]:
        """Create a text summary of CV analysis results"""
        shapes_summary = []
        for shape in cv_result.shapes:
            shape_desc = f"- {shape.shape_type.value}"
            if shape.shape_type.value == 'circle':
                radius = shape.parameters.get('radius', 0)
                shape_desc += f" (radius: {radius:.1f}px)"
            elif shape.shape_type.value == 'rectangle':
                width = shape.parameters.get('width', 0)
                height = shape.parameters.get('height', 0)
                shape_desc += f" (size: {width:.1f}x{height:.1f}px)"
            elif shape.shape_type.value == 'line':
                length = shape.parameters.get('length', 0)
                shape_desc += f" (length: {length:.1f}px)"

            shape_desc += f" at ({shape.center.x:.1f}, {shape.center.y:.1f})"
            shapes_summary.append(shape_desc)

        connections_summary = []
        for conn in cv_result.connections:
            conn_desc = f"- {conn.connection_type.value} between shapes"
            connections_summary.append(conn_desc)

        text_summary = []
        for text in cv_result.text_annotations:
            text_desc = f"- '{text.text}' at ({text.position.x:.1f}, {text.position.y:.1f})"
            text_summary.append(text_desc)

        return {
            'shapes': '\n'.join(shapes_summary) if shapes_summary else 'No shapes detected',
            'connections': '\n'.join(connections_summary) if connections_summary else 'No connections detected',
            'text': '\n'.join(text_summary) if text_summary else 'No text detected'
        }

    def _create_basic_description_from_cv(self, cv_result: CVAnalysisResult, user_text: Optional[str]) -> str:
        """Create a basic description from CV analysis when LLM enhancement fails"""
        description_parts = []

        if user_text:
            description_parts.append(f"Physics simulation based on sketch: {user_text}")
        else:
            description_parts.append("Physics simulation generated from hand-drawn sketch")

        if cv_result.shapes:
            shape_types = [shape.shape_type.value for shape in cv_result.shapes]
            unique_shapes = list(set(shape_types))
            description_parts.append(f"Contains {len(cv_result.shapes)} objects: {', '.join(unique_shapes)}")

        if cv_result.connections:
            conn_types = [conn.connection_type.value for conn in cv_result.connections]
            unique_conns = list(set(conn_types))
            description_parts.append(f"With {len(cv_result.connections)} connections: {', '.join(unique_conns)}")

        if cv_result.text_annotations:
            all_text = [text.text for text in cv_result.text_annotations]
            description_parts.append(f"Annotations: {', '.join(all_text)}")

        return '. '.join(description_parts) + '.'

    async def _create_extracted_entities_from_cv(self, cv_result: CVAnalysisResult) -> Optional[ExtractedEntities]:
        """Create backward-compatible ExtractedEntities from CV analysis"""
        try:
            objects = []
            constraints = []

            # Convert CV shapes to ObjectSchema
            for i, shape in enumerate(cv_result.shapes):
                # Determine geometry
                if shape.shape_type.value == 'circle':
                    radius = shape.parameters.get('radius', 25) / 100.0  # Convert to meters
                    geometry = GeometrySchema(shape='sphere', dimensions=[radius])
                elif shape.shape_type.value == 'rectangle':
                    width = shape.parameters.get('width', 50) / 100.0
                    height = shape.parameters.get('height', 50) / 100.0
                    geometry = GeometrySchema(shape='box', dimensions=[width, height, min(width, height) * 0.1])
                else:
                    geometry = GeometrySchema(shape='box', dimensions=[0.1, 0.1, 0.1])

                # Create object
                obj = ObjectSchema(
                    name=f"{shape.shape_type.value}_{i}",
                    type='rigid_body',
                    geometry=geometry,
                    material=MaterialSchema(),
                    position=[shape.center.x / 100.0, shape.center.y / 100.0, 0.0],
                    orientation=[0, 0, 0]
                )
                objects.append(obj)

            # Convert CV connections to ConstraintSchema
            for conn in cv_result.connections:
                constraint = ConstraintSchema(
                    type='joint',
                    bodies=[conn.shape1_id, conn.shape2_id],
                    parameters={'joint_type': conn.connection_type.value}
                )
                constraints.append(constraint)

            # Create environment
            environment = EnvironmentSchema()

            return ExtractedEntities(
                objects=objects,
                constraints=constraints,
                environment=environment
            )

        except Exception as e:
            logger.error(f"Failed to create extracted entities from CV: {e}")
            return None

    def _format_cv_analysis_output(self, cv_result: CVAnalysisResult) -> str:
        """Format CV analysis result as readable text output"""
        output_parts = []

        output_parts.append("=== COMPUTER VISION ANALYSIS ===")
        output_parts.append(f"Confidence: {cv_result.confidence_score:.2f}")
        output_parts.append("")

        if cv_result.shapes:
            output_parts.append("DETECTED SHAPES:")
            for i, shape in enumerate(cv_result.shapes):
                shape_info = f"{i+1}. {shape.shape_type.value.upper()}"
                shape_info += f" at ({shape.center.x:.0f}, {shape.center.y:.0f})"
                shape_info += f" confidence: {shape.confidence:.2f}"

                # Add shape-specific parameters
                if shape.shape_type.value == 'circle':
                    shape_info += f" radius: {shape.parameters.get('radius', 0):.1f}px"
                elif shape.shape_type.value == 'rectangle':
                    width = shape.parameters.get('width', 0)
                    height = shape.parameters.get('height', 0)
                    shape_info += f" size: {width:.1f}×{height:.1f}px"
                elif shape.shape_type.value == 'line':
                    length = shape.parameters.get('length', 0)
                    angle = shape.parameters.get('angle', 0)
                    shape_info += f" length: {length:.1f}px angle: {angle:.1f}°"

                output_parts.append(shape_info)
            output_parts.append("")

        if cv_result.connections:
            output_parts.append("DETECTED CONNECTIONS:")
            for i, conn in enumerate(cv_result.connections):
                conn_info = f"{i+1}. {conn.connection_type.value.upper()}"
                conn_info += f" at ({conn.connection_point.x:.0f}, {conn.connection_point.y:.0f})"
                conn_info += f" confidence: {conn.confidence:.2f}"
                output_parts.append(conn_info)
            output_parts.append("")

        if cv_result.text_annotations:
            output_parts.append("DETECTED TEXT:")
            for i, text in enumerate(cv_result.text_annotations):
                text_info = f"{i+1}. '{text.text}'"
                text_info += f" at ({text.position.x:.0f}, {text.position.y:.0f})"
                text_info += f" confidence: {text.confidence:.2f}"
                output_parts.append(text_info)
            output_parts.append("")

        if cv_result.physics_interpretation:
            output_parts.append("PHYSICS INTERPRETATION:")
            interpretation = cv_result.physics_interpretation
            if 'objects' in interpretation:
                output_parts.append(f"Objects detected: {len(interpretation['objects'])}")
            if 'constraints' in interpretation:
                output_parts.append(f"Constraints detected: {len(interpretation['constraints'])}")
            output_parts.append("")

        return '\n'.join(output_parts)

    def _calculate_combined_confidence(self, cv_result: CVAnalysisResult, conversion_result: ConversionResult) -> float:
        """Calculate combined confidence from CV analysis and PhysicsSpec conversion"""
        cv_confidence = cv_result.confidence_score
        conversion_confidence = conversion_result.confidence_score

        # Weighted average favoring conversion quality
        combined = cv_confidence * 0.6 + conversion_confidence * 0.4

        return min(1.0, combined)


# Backward compatibility
SketchAnalyzer = AdvancedSketchAnalyzer

def get_sketch_analyzer() -> AdvancedSketchAnalyzer:
    """Get advanced sketch analyzer instance with LLM client."""
    from .llm_client import get_llm_client
    return AdvancedSketchAnalyzer(get_llm_client())