import logging
from typing import List, Dict, Any
from pydantic import ValidationError

from ..models.schemas import ExtractedEntities, ObjectSchema, ConstraintSchema, EnvironmentSchema
from ..services.llm_client import LLMClient, LLMError


logger = logging.getLogger(__name__)


class PromptParsingError(Exception):
    """Exception raised when prompt parsing fails."""
    pass


class PromptParser:
    """Service for parsing natural language prompts into structured simulation entities."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.entity_extraction_schema = self._build_entity_extraction_schema()
    
    def _build_entity_extraction_schema(self) -> Dict[str, Any]:
        """Build JSON schema for entity extraction."""
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "object",
                    "properties": {
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["rigid_body", "soft_body", "robot", "vehicle", "static"]
                                    },
                                    "geometry": {
                                        "type": "object",
                                        "properties": {
                                            "shape": {
                                                "type": "string",
                                                "enum": ["box", "sphere", "cylinder", "capsule", "plane", "mesh"]
                                            },
                                            "dimensions": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "minItems": 1,
                                                "maxItems": 3
                                            }
                                        },
                                        "required": ["shape", "dimensions"]
                                    },
                                    "material": {
                                        "type": "object",
                                        "properties": {
                                            "density": {"type": "number", "minimum": 0},
                                            "friction": {"type": "number", "minimum": 0},
                                            "restitution": {"type": "number", "minimum": 0, "maximum": 1}
                                        }
                                    },
                                    "position": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 3,
                                        "maxItems": 3
                                    },
                                    "orientation": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 3,
                                        "maxItems": 3
                                    }
                                },
                                "required": ["name", "type", "geometry"]
                            }
                        },
                        "constraints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["joint", "contact", "force", "spring", "damper"]
                                    },
                                    "bodies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1,
                                        "maxItems": 2
                                    },
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "joint_type": {
                                                "type": "string",
                                                "enum": ["revolute", "prismatic", "fixed", "ball", "universal"]
                                            },
                                            "axis": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "minItems": 3,
                                                "maxItems": 3
                                            },
                                            "limits": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "minItems": 2,
                                                "maxItems": 2
                                            }
                                        }
                                    }
                                },
                                "required": ["type", "bodies"]
                            }
                        },
                        "environment": {
                            "type": "object",
                            "properties": {
                                "gravity": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3
                                },
                                "ground": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["plane", "heightfield", "mesh", "none"]},
                                        "friction": {"type": "number", "minimum": 0},
                                        "position": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 3,
                                            "maxItems": 3
                                        }
                                    }
                                },
                                "boundaries": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["none", "box", "sphere"]},
                                        "size": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 1,
                                            "maxItems": 3
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "required": ["objects", "environment"]
                }
            },
            "required": ["entities"]
        }
    
    def _build_extraction_prompt(self, user_input: str, examples: List[str] = None) -> str:
        """Build prompt for entity extraction."""
        
        base_prompt = f"""
Analyze this simulation request and extract structured entities:

USER REQUEST: "{user_input}"

Extract the following information:
1. OBJECTS: All physical objects mentioned (rigid bodies, robots, vehicles, etc.)
2. CONSTRAINTS: Joints, connections, forces between objects
3. ENVIRONMENT: Gravity, ground, boundaries, world settings

For each object, determine:
- Name and type (rigid_body, robot, vehicle, etc.)
- Geometry (shape and dimensions in meters)
- Material properties (density kg/m³, friction coefficient, restitution)
- Position and orientation in 3D space

For constraints, identify:
- Type (joint, contact, force, etc.)
- Connected bodies
- Joint parameters (type, axis, limits)

For environment, specify:
- Gravity vector (default [0, 0, -9.81])
- Ground properties
- Boundary conditions

IMPORTANT GUIDELINES:
- Use realistic physical parameters
- Default gravity is Earth gravity unless specified
- Use SI units (meters, kilograms, seconds)
- Infer reasonable dimensions if not specified
- Create physically plausible configurations
- If objects need to be connected, add appropriate constraints

EXAMPLES:
- "Ball" → sphere with radius ~0.1m, density ~1000 kg/m³
- "Robot arm" → multiple connected rigid bodies with revolute joints
- "Car" → vehicle with wheels, body, appropriate mass distribution
- "Pendulum" → mass connected to fixed point with revolute joint
"""
        
        if examples:
            base_prompt += f"\n\nSIMILAR SUCCESSFUL EXAMPLES:\n"
            for i, example in enumerate(examples, 1):
                base_prompt += f"\nExample {i}:\n{example}\n"
        
        return base_prompt
    
    async def parse_prompt(self, user_input: str) -> ExtractedEntities:
        """Parse natural language prompt into structured entities."""
        
        try:
            # Build extraction prompt
            extraction_prompt = self._build_extraction_prompt(user_input)
            
            # Get structured response from LLM
            response = await self.llm_client.complete_with_schema(
                prompt=extraction_prompt,
                schema=self.entity_extraction_schema,
                temperature=0.1
            )
            
            # Validate and parse response
            if "entities" not in response:
                raise PromptParsingError("Invalid response format: missing 'entities' key")
            
            entities_data = response["entities"]
            
            # Parse objects
            objects = []
            for obj_data in entities_data.get("objects", []):
                try:
                    # Set defaults for optional fields
                    if "material" not in obj_data:
                        obj_data["material"] = {
                            "density": 1000.0,
                            "friction": 0.5,
                            "restitution": 0.1
                        }
                    if "position" not in obj_data:
                        obj_data["position"] = [0.0, 0.0, 0.0]
                    if "orientation" not in obj_data:
                        obj_data["orientation"] = [0.0, 0.0, 0.0]
                    
                    obj = ObjectSchema(**obj_data)
                    objects.append(obj)
                except ValidationError as e:
                    logger.warning(f"Invalid object data, skipping: {e}")
                    continue
            
            # Parse constraints
            constraints = []
            for const_data in entities_data.get("constraints", []):
                try:
                    # Set defaults
                    if "parameters" not in const_data:
                        const_data["parameters"] = {}
                    
                    constraint = ConstraintSchema(**const_data)
                    constraints.append(constraint)
                except ValidationError as e:
                    logger.warning(f"Invalid constraint data, skipping: {e}")
                    continue
            
            # Parse environment
            env_data = entities_data.get("environment", {})
            try:
                environment = EnvironmentSchema(**env_data)
            except ValidationError as e:
                logger.warning(f"Invalid environment data, using defaults: {e}")
                environment = EnvironmentSchema()
            
            # Create final ExtractedEntities
            extracted = ExtractedEntities(
                objects=objects,
                constraints=constraints,
                environment=environment
            )
            
            # Validate that we extracted at least one object
            if not extracted.objects:
                raise PromptParsingError("No valid objects extracted from prompt")
            
            logger.info(f"Successfully extracted {len(extracted.objects)} objects, "
                       f"{len(extracted.constraints)} constraints from prompt")
            
            return extracted
            
        except LLMError as e:
            logger.error(f"LLM error during prompt parsing: {e}")
            raise PromptParsingError(f"Failed to parse prompt due to LLM error: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error during prompt parsing: {e}", exc_info=True)
            raise PromptParsingError(f"Unexpected error during parsing: {e}")
    
    def validate_entities(self, entities: ExtractedEntities) -> List[str]:
        """Validate extracted entities and return list of issues."""
        issues = []
        
        # Check for empty objects
        if not entities.objects:
            issues.append("No objects defined in simulation")
        
        # Check object names are unique
        names = [obj.name for obj in entities.objects]
        if len(names) != len(set(names)):
            issues.append("Duplicate object names found")
        
        # Check constraint references
        for constraint in entities.constraints:
            for body_name in constraint.bodies:
                if body_name not in names and body_name != "world":
                    issues.append(f"Constraint references unknown body: {body_name}")
        
        # Check for reasonable dimensions
        for obj in entities.objects:
            for dim in obj.geometry.dimensions:
                if dim <= 0:
                    issues.append(f"Invalid dimension for {obj.name}: {dim}")
                if dim > 100:  # Larger than 100m might be suspicious
                    issues.append(f"Very large dimension for {obj.name}: {dim}m")
        
        # Check material properties
        for obj in entities.objects:
            if obj.material.density <= 0:
                issues.append(f"Invalid density for {obj.name}: {obj.material.density}")
            if obj.material.friction < 0:
                issues.append(f"Invalid friction for {obj.name}: {obj.material.friction}")
            if not (0 <= obj.material.restitution <= 1):
                issues.append(f"Invalid restitution for {obj.name}: {obj.material.restitution}")
        
        return issues