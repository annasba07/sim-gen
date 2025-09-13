import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from jinja2 import Environment, BaseLoader, Template

from ..models.schemas import ExtractedEntities, SimulationGenerationMethod
from ..models.simulation import SimulationGenerationMethod as DBGenerationMethod
from ..services.llm_client import LLMClient, LLMError
from ..services.dynamic_scene_composer import DynamicSceneComposer


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of simulation generation."""
    mjcf_content: str
    method: DBGenerationMethod
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class SimulationGenerationError(Exception):
    """Exception raised when simulation generation fails."""
    pass


class SimulationGenerator:
    """Service for generating MJCF simulation files from extracted entities."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.jinja_env = Environment(loader=BaseLoader())
        self.base_mjcf_template = self._get_base_mjcf_template()
        
        # NEW: Initialize dynamic scene composer for generalized generation
        self.dynamic_composer = DynamicSceneComposer(llm_client)
    
    def _get_base_mjcf_template(self) -> str:
        """Get cinematic-quality MJCF template with professional visuals."""
        return """<mujoco model="ai_cinematic_simulation">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  
  <option timestep="0.002" iterations="50" solver="PGS" gravity="{{ environment.gravity|join(' ') }}"/>
  
  <asset>
    <!-- CINEMATIC TEXTURES -->
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.0 0.1 0.2" 
             width="512" height="512"/>
    <texture name="groundplane" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.25 0.25 0.25" 
             width="200" height="200" mark="edge" markrgb="0.8 0.8 0.8"/>
    
    <!-- PROFESSIONAL MATERIALS -->
    <material name="groundplane" texture="groundplane" texuniform="true" 
              reflectance="0.3" shininess="0.2" specular="0.5"/>
    <material name="metal" rgba="0.7 0.7 0.8 1" reflectance="0.7" shininess="0.8" specular="0.9"/>
    <material name="pendulum_bob" rgba="0.9 0.2 0.2 1" reflectance="0.5" shininess="0.7" specular="0.8"/>
    <material name="pendulum_arm" rgba="0.6 0.6 0.9 1" reflectance="0.3" shininess="0.5" specular="0.6"/>
    <material name="ball_material" rgba="0.2 0.8 0.3 1" reflectance="0.6" shininess="0.8" specular="0.9"/>
    <material name="cube_material" rgba="0.8 0.6 0.2 1" reflectance="0.4" shininess="0.6" specular="0.7"/>
  </asset>
  
  <worldbody>
    <!-- PROFESSIONAL LIGHTING SETUP -->
    <light name="sun" pos="0 0 4" dir="0 0 -1" directional="true" 
           diffuse="0.9 0.9 0.8" specular="0.4 0.4 0.3" castshadow="true"/>
    <light name="fill1" pos="3 2 2" dir="-0.5 -0.3 -1" directional="true"
           diffuse="0.3 0.3 0.4" specular="0.1 0.1 0.1"/>
    <light name="fill2" pos="-2 2 1" dir="0.3 -0.3 -0.5" directional="true"
           diffuse="0.2 0.2 0.3" specular="0.05 0.05 0.05"/>
    
    {% if environment.ground.type != "none" %}
    <geom name="ground" type="plane" size="12 12 0.1" material="groundplane" friction="0.8 0.1 0.001"/>
    {% endif %}
    
    {% for obj in objects %}
    {% if obj.name in ["FixedPoint", "Fixed Point"] %}
    <!-- Fixed anchor point -->
    <body name="{{ obj.name }}" pos="{{ obj.position|join(' ') }}" euler="{{ obj.orientation|join(' ') }}">
      <geom name="{{ obj.name }}_geom" type="box" size="0.05 0.05 0.05" rgba="0.8 0.2 0.2 1"/>
      
      {% for other_obj in objects %}
      {% if other_obj.name != obj.name and other_obj.name in ["PendulumArm", "Pendulum Rod"] %}
      <!-- Pendulum arm attached to anchor -->
      <body name="{{ other_obj.name }}" pos="0 0 -1" euler="{{ other_obj.orientation|join(' ') }}">
        <joint name="pendulum_hinge" type="hinge" axis="1 0 0" range="-180 180"/>
        {% if other_obj.geometry.shape == "cylinder" %}
        <geom name="{{ other_obj.name }}_geom" type="cylinder" size="{{ other_obj.geometry.dimensions[0] }} {{ other_obj.geometry.dimensions[1] }}" 
              density="{{ other_obj.material.density }}" rgba="0.2 0.2 0.8 1"/>
        {% endif %}
        
        {% for bob_obj in objects %}
        {% if bob_obj.name in ["PendulumBob", "Pendulum Bob", "Bob"] %}
        <!-- Pendulum bob at end of arm -->
        <body name="{{ bob_obj.name }}" pos="0 0 -{{ other_obj.geometry.dimensions[1] }}">
          {% if bob_obj.geometry.shape == "sphere" %}
          <geom name="{{ bob_obj.name }}_geom" type="sphere" size="{{ bob_obj.geometry.dimensions[0] }}" 
                density="{{ bob_obj.material.density }}" rgba="0.2 0.8 0.2 1"/>
          {% endif %}
        </body>
        {% endif %}
        {% endfor %}
      </body>
      {% endif %}
      {% endfor %}
    </body>
    
    {% else %}
    <!-- Regular rigid body (for balls, cubes, etc.) -->
    <body name="{{ obj.name }}" pos="{{ obj.position|join(' ') }}" euler="{{ obj.orientation|join(' ') }}">
      {% if obj.type == "rigid_body" %}
      <joint name="{{ obj.name }}_freejoint" type="free"/>
      {% endif %}
      
      {% if obj.geometry.shape == "box" %}
      <geom name="{{ obj.name }}_geom" type="box" size="{{ (obj.geometry.dimensions[0]/2)|round(4) }} {{ (obj.geometry.dimensions[1]/2)|round(4) }} {{ (obj.geometry.dimensions[2]/2)|round(4) }}" 
            density="{{ obj.material.density }}" friction="{{ obj.material.friction }} 0.1 0.005" solref="0.02 1" solimp="0.8 0.8 0.01" rgba="0.8 0.2 0.2 1"/>
      {% elif obj.geometry.shape == "sphere" %}
      <geom name="{{ obj.name }}_geom" type="sphere" size="{{ obj.geometry.dimensions[0] }}" 
            material="ball_material" density="{{ obj.material.density }}" 
            friction="{{ obj.material.friction }} 0.1 0.005" solref="0.02 1" solimp="0.95 0.95 0.001"/>
      <geom name="{{ obj.name }}_highlight" type="sphere" size="{{ obj.geometry.dimensions[0] * 1.05 }}" 
            pos="0.02 0.02 0.02" rgba="1.0 1.0 1.0 0.3" density="10"/>
      {% elif obj.geometry.shape == "cylinder" %}
      <geom name="{{ obj.name }}_geom" type="cylinder" size="{{ obj.geometry.dimensions[0] }} {{ obj.geometry.dimensions[1] }}" 
            density="{{ obj.material.density }}" friction="{{ obj.material.friction }} 0.1 0.005" solref="0.02 1" solimp="0.8 0.8 0.01" rgba="0.2 0.2 0.8 1"/>
      {% elif obj.geometry.shape == "capsule" %}
      <geom name="{{ obj.name }}_geom" type="capsule" size="{{ obj.geometry.dimensions[0] }} {{ obj.geometry.dimensions[1] }}" 
            density="{{ obj.material.density }}" friction="{{ obj.material.friction }} 0.1 0.005" solref="0.02 1" solimp="0.8 0.8 0.01" rgba="0.8 0.8 0.2 1"/>
      {% endif %}
    </body>
    {% endif %}
    {% endfor %}
  </worldbody>
  
  <!-- CINEMATIC VISUAL SETTINGS -->
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map force="0.1" zfar="30"/>
  </visual>
</mujoco>"""
    
    async def generate_simulation(self, entities: ExtractedEntities, prompt: Optional[str] = None) -> GenerationResult:
        """Generate MJCF simulation from extracted entities using dynamic composition."""
        
        try:
            # FIRST: Try dynamic scene composition (NEW GENERALIZED APPROACH)
            if prompt:
                logger.info("Attempting dynamic scene composition (generalized approach)")
                dynamic_result = await self._dynamic_composition_generation(entities, prompt)
                if dynamic_result.success:
                    return dynamic_result
                logger.info("Dynamic composition failed, falling back to template approach")
            
            # FALLBACK 1: Template-based generation (OLD APPROACH)
            template_result = await self._template_based_generation(entities)
            if template_result.success:
                return template_result
            
            # FALLBACK 2: LLM-based generation
            logger.info("Template generation failed, trying LLM-based generation")
            llm_result = await self._llm_based_generation(entities)
            if llm_result.success:
                return llm_result
            
            # If all methods fail, return error
            raise SimulationGenerationError("All generation methods failed")
            
        except Exception as e:
            logger.error(f"Simulation generation failed: {e}", exc_info=True)
            return GenerationResult(
                mjcf_content="",
                method=DBGenerationMethod.TEMPLATE_BASED,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    async def _dynamic_composition_generation(self, entities: ExtractedEntities, prompt: str) -> GenerationResult:
        """Generate simulation using dynamic scene composition (NEW GENERALIZED APPROACH)."""
        
        try:
            logger.info("Attempting dynamic scene composition")
            
            # Use the dynamic scene composer for truly generalized generation
            mjcf_content = await self.dynamic_composer.compose_scene_from_prompt(prompt, entities)
            
            # Validate the generated content
            if not self._validate_mjcf_structure(mjcf_content):
                raise SimulationGenerationError("Dynamic composition generated invalid MJCF")
            
            logger.info("Dynamic scene composition successful")
            
            return GenerationResult(
                mjcf_content=mjcf_content,
                method=DBGenerationMethod.HYBRID,  # Using hybrid since it combines LLM + professional models
                metadata={
                    "generation_method": "dynamic_composition",
                    "objects_count": len(entities.objects),
                    "constraints_count": len(entities.constraints),
                    "prompt_length": len(prompt),
                    "professional_models": True,
                    "generalized_approach": True
                }
            )
            
        except Exception as e:
            logger.warning(f"Dynamic scene composition failed: {e}")
            return GenerationResult(
                mjcf_content="",
                method=DBGenerationMethod.HYBRID,
                metadata={"dynamic_composition_error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    async def _template_based_generation(self, entities: ExtractedEntities) -> GenerationResult:
        """Generate simulation using Jinja2 templates."""
        
        try:
            logger.info("Attempting template-based generation")
            
            # Prepare template context
            context = {
                "objects": entities.objects,
                "constraints": entities.constraints,
                "environment": entities.environment
            }
            
            # Render template
            template = Template(self.base_mjcf_template)
            mjcf_content = template.render(**context)
            
            # Basic validation
            if not self._validate_mjcf_structure(mjcf_content):
                raise SimulationGenerationError("Generated MJCF failed validation")
            
            logger.info("Template-based generation successful")
            
            return GenerationResult(
                mjcf_content=mjcf_content,
                method=DBGenerationMethod.TEMPLATE_BASED,
                metadata={
                    "objects_count": len(entities.objects),
                    "constraints_count": len(entities.constraints),
                    "template_used": "base_mjcf"
                }
            )
            
        except Exception as e:
            logger.warning(f"Template-based generation failed: {e}")
            return GenerationResult(
                mjcf_content="",
                method=DBGenerationMethod.TEMPLATE_BASED,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    async def _llm_based_generation(self, entities: ExtractedEntities) -> GenerationResult:
        """Generate simulation using LLM."""
        
        try:
            logger.info("Attempting LLM-based generation")
            
            # Build generation prompt
            generation_prompt = self._build_generation_prompt(entities)
            
            # Get MJCF from LLM
            mjcf_content = await self.llm_client.complete(
                prompt=generation_prompt,
                temperature=0.2,
                max_tokens=4000
            )
            
            # Clean up the response (remove any markdown formatting)
            mjcf_content = self._clean_mjcf_response(mjcf_content)
            
            # Validate generated MJCF
            if not self._validate_mjcf_structure(mjcf_content):
                raise SimulationGenerationError("LLM-generated MJCF failed validation")
            
            logger.info("LLM-based generation successful")
            
            return GenerationResult(
                mjcf_content=mjcf_content,
                method=DBGenerationMethod.LLM_GENERATION,
                metadata={
                    "objects_count": len(entities.objects),
                    "constraints_count": len(entities.constraints),
                    "generation_method": "llm_direct"
                }
            )
            
        except LLMError as e:
            logger.error(f"LLM generation failed: {e}")
            return GenerationResult(
                mjcf_content="",
                method=DBGenerationMethod.LLM_GENERATION,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"LLM generation failed with unexpected error: {e}")
            return GenerationResult(
                mjcf_content="",
                method=DBGenerationMethod.LLM_GENERATION,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _build_generation_prompt(self, entities: ExtractedEntities) -> str:
        """Build prompt for LLM-based MJCF generation."""
        
        objects_desc = []
        for obj in entities.objects:
            obj_desc = f"- {obj.name}: {obj.type} with {obj.geometry.shape} shape"
            obj_desc += f", dimensions {obj.geometry.dimensions}"
            obj_desc += f", position {obj.position}, material density {obj.material.density}"
            objects_desc.append(obj_desc)
        
        constraints_desc = []
        for constraint in entities.constraints:
            const_desc = f"- {constraint.type} between {', '.join(constraint.bodies)}"
            if constraint.parameters:
                const_desc += f" with parameters {constraint.parameters}"
            constraints_desc.append(const_desc)
        
        prompt = f"""
Generate a complete MJCF (MuJoCo XML) simulation file based on these specifications:

OBJECTS:
{chr(10).join(objects_desc)}

CONSTRAINTS:
{chr(10).join(constraints_desc)}

ENVIRONMENT:
- Gravity: {entities.environment.gravity}
- Ground: {entities.environment.ground}
- Boundaries: {entities.environment.boundaries}

REQUIREMENTS:
1. Generate valid MJCF XML format
2. Include proper compiler settings and options
3. Set realistic physics parameters (timestep, solver, etc.)
4. Add appropriate materials and visual properties
5. Ensure stable simulation setup
6. Include proper joint definitions for constraints
7. Add actuators where appropriate

MJCF STRUCTURE REQUIREMENTS:
- Start with <mujoco model="simulation">
- Include <compiler> with angle="degree" coordinate="local"
- Set <option> with appropriate timestep (0.001-0.002)
- Define <asset> section with materials/textures
- Create <worldbody> with all objects and constraints
- Add <actuator> section if needed for control

IMPORTANT: Return only the complete MJCF XML content, no additional text or explanation.
"""
        
        return prompt
    
    def _clean_mjcf_response(self, response: str) -> str:
        """Clean up LLM response to extract pure MJCF content."""
        
        # Remove markdown code blocks if present
        if "```xml" in response:
            response = response.split("```xml")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # Strip whitespace
        response = response.strip()
        
        # Ensure it starts with <?xml or <mujoco
        if not response.startswith(('<?xml', '<mujoco')):
            # Try to find the start of XML content
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('<?xml', '<mujoco')):
                    response = '\n'.join(lines[i:])
                    break
        
        return response
    
    def _validate_mjcf_structure(self, mjcf_content: str) -> bool:
        """Basic validation of MJCF structure."""
        
        try:
            # Check for basic XML structure
            if not mjcf_content.strip():
                return False
            
            # Check for required MJCF elements
            required_elements = ['<mujoco', '<worldbody', '</worldbody>', '</mujoco>']
            for element in required_elements:
                if element not in mjcf_content:
                    logger.warning(f"Missing required element: {element}")
                    return False
            
            # Try basic XML parsing
            import xml.etree.ElementTree as ET
            try:
                ET.fromstring(mjcf_content)
            except ET.ParseError as e:
                logger.warning(f"XML parsing failed: {e}")
                return False
            
            # Check for reasonable length (not empty, not too short)
            if len(mjcf_content) < 100:
                logger.warning("MJCF content too short")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"MJCF validation failed: {e}")
            return False
    
    async def refine_simulation(
        self, 
        mjcf_content: str, 
        issues: List[str]
    ) -> GenerationResult:
        """Refine simulation based on detected issues."""
        
        try:
            refinement_prompt = f"""
The following MJCF simulation has issues that need to be fixed:

ORIGINAL MJCF:
{mjcf_content}

ISSUES DETECTED:
{chr(10).join(f"- {issue}" for issue in issues)}

Please fix these issues and return the corrected MJCF content.

REQUIREMENTS:
1. Fix all detected issues
2. Maintain the original simulation intent
3. Ensure physics stability
4. Keep all original objects and constraints
5. Return only the corrected MJCF XML, no explanation

CORRECTED MJCF:
"""
            
            refined_content = await self.llm_client.complete(
                prompt=refinement_prompt,
                temperature=0.1,
                max_tokens=4000
            )
            
            refined_content = self._clean_mjcf_response(refined_content)
            
            if self._validate_mjcf_structure(refined_content):
                return GenerationResult(
                    mjcf_content=refined_content,
                    method=DBGenerationMethod.HYBRID,
                    metadata={
                        "refinement": True,
                        "issues_fixed": len(issues),
                        "original_length": len(mjcf_content),
                        "refined_length": len(refined_content)
                    }
                )
            else:
                raise SimulationGenerationError("Refined MJCF failed validation")
                
        except Exception as e:
            logger.error(f"Simulation refinement failed: {e}")
            return GenerationResult(
                mjcf_content=mjcf_content,  # Return original if refinement fails
                method=DBGenerationMethod.HYBRID,
                metadata={"refinement_error": str(e)},
                success=False,
                error_message=f"Refinement failed: {e}"
            )