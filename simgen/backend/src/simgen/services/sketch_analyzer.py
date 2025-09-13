"""
Sketch Analysis Service

This service interprets hand-drawn sketches and converts them into structured
physics simulation descriptions that can be used by the simulation generator.
"""

import base64
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel

from .llm_client import LLMClient
from ..models.schemas import ExtractedEntities, ObjectSchema, ConstraintSchema, EnvironmentSchema, GeometrySchema, MaterialSchema

logger = logging.getLogger(__name__)


@dataclass
class SketchAnalysisResult:
    """Result of sketch analysis with physics interpretation."""
    success: bool
    physics_description: str
    extracted_entities: Optional[ExtractedEntities]
    confidence_score: float
    raw_vision_output: str
    error_message: Optional[str] = None


class SketchAnalyzer:
    """Analyzes hand-drawn sketches and converts them to physics simulations."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    async def analyze_sketch(
        self, 
        image_data: bytes, 
        user_text: Optional[str] = None
    ) -> SketchAnalysisResult:
        """
        Analyze a sketch and convert it to physics simulation description.
        
        Args:
            image_data: Raw image bytes of the sketch
            user_text: Optional text prompt to accompany the sketch
            
        Returns:
            SketchAnalysisResult with physics interpretation
        """
        try:
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
            
            return SketchAnalysisResult(
                success=True,
                physics_description=physics_description,
                extracted_entities=extracted_entities,
                confidence_score=confidence,
                raw_vision_output=raw_analysis
            )
            
        except Exception as e:
            logger.error(f"Sketch analysis failed: {e}")
            return SketchAnalysisResult(
                success=False,
                physics_description="",
                extracted_entities=None,
                confidence_score=0.0,
                raw_vision_output="",
                error_message=str(e)
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


def get_sketch_analyzer() -> SketchAnalyzer:
    """Get sketch analyzer instance with LLM client."""
    from .llm_client import get_llm_client
    return SketchAnalyzer(get_llm_client())