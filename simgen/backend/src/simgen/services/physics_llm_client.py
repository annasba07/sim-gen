"""
Physics-aware LLM Client
Generates structured PhysicsSpec from prompts instead of raw MJCF
"""

import json
import logging
from typing import Dict, Any, Optional
from anthropic import AsyncAnthropic
import openai

from ..models.physics_spec import PhysicsSpec
from ..core.config_clean import settings

logger = logging.getLogger(__name__)

class PhysicsLLMClient:
    """
    LLM client specialized for physics generation
    Produces PhysicsSpec JSON, not raw MJCF
    """

    SYSTEM_PROMPT = """You are a physics simulation expert that creates MuJoCo simulations.

Your task is to generate a valid PhysicsSpec JSON that represents physics scenarios.

CRITICAL RULES:
1. Output ONLY valid JSON matching the PhysicsSpec schema - no commentary
2. Use SI units exclusively (meters, kilograms, seconds, Newtons)
3. Prefer capsule geometries for links and connectors
4. Keep dimensions realistic (0.01m to 10m scale)
5. Include proper inertial properties (mass must be between 1e-6 and 1e6 kg)
6. Add actuators for controllable systems
7. Add sensors for observable quantities
8. Use hierarchical body structure for connected parts

PhysicsSpec Schema Structure:
{
  "meta": {
    "name": "string",
    "description": "string",
    "gravity": [0, 0, -9.81],
    "timestep": 0.002
  },
  "bodies": [
    {
      "id": "unique_name",
      "pos": [x, y, z],
      "joint": {
        "type": "hinge|slide|ball|free",
        "axis": [x, y, z],
        "limited": true/false,
        "range": [min, max]
      },
      "geoms": [
        {
          "type": "box|sphere|capsule|cylinder",
          "size": [...],
          "fromto": [x1,y1,z1, x2,y2,z2],  // for capsules
          "material": {"rgba": [r,g,b,a]},
          "friction": {"slide": 1.0, "spin": 0.005, "roll": 0.0001}
        }
      ],
      "inertial": {
        "mass": 1.0,
        "pos": [x, y, z]
      },
      "children": [...]  // nested bodies
    }
  ],
  "actuators": [
    {
      "id": "motor_name",
      "type": "motor",
      "target": "joint_name",
      "gear": 1.0,
      "ctrlrange": [-1, 1]
    }
  ],
  "sensors": [
    {
      "type": "jointpos|jointvel|actuatorfrc",
      "source": "joint_or_actuator_name"
    }
  ]
}

Common Physics Patterns:
- Pendulum: Single body with hinge joint, gravity-driven
- Robot arm: Chain of bodies with hinge joints and actuators
- Cart-pole: Slider joint (cart) + hinge joint (pole)
- Projectile: Free joint with initial velocity
- Stack: Multiple free joints with collision"""

    def __init__(self):
        self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
        self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def generate_physics_spec(self,
                                   user_prompt: str,
                                   sketch_analysis: Optional[str] = None,
                                   use_anthropic: bool = True) -> PhysicsSpec:
        """
        Generate PhysicsSpec from natural language description

        Args:
            user_prompt: Natural language description
            sketch_analysis: Optional sketch analysis results
            use_anthropic: Use Anthropic Claude (True) or OpenAI GPT-4 (False)

        Returns:
            Validated PhysicsSpec object
        """
        # Build the full prompt
        full_prompt = self._build_prompt(user_prompt, sketch_analysis)

        # Generate with LLM
        if use_anthropic and self.anthropic_client:
            spec_json = await self._generate_anthropic(full_prompt)
        elif self.openai_client:
            spec_json = await self._generate_openai(full_prompt)
        else:
            raise ValueError("No LLM API key configured")

        # Parse and validate
        try:
            spec_data = json.loads(spec_json)
            spec = PhysicsSpec(**spec_data)
            logger.info(f"Generated PhysicsSpec with {len(spec.bodies)} bodies")
            return spec
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM: {e}")
            logger.error(f"Response: {spec_json[:500]}...")
            raise ValueError(f"LLM generated invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Failed to parse PhysicsSpec: {e}")
            raise

    def _build_prompt(self, user_prompt: str, sketch_analysis: Optional[str]) -> str:
        """Build complete prompt with context"""
        prompt_parts = []

        # Add sketch context if available
        if sketch_analysis:
            prompt_parts.append(f"Sketch Analysis Results:\n{sketch_analysis}\n")

        # Add user prompt
        prompt_parts.append(f"User Request:\n{user_prompt}\n")

        # Add generation instruction
        prompt_parts.append("\nGenerate a PhysicsSpec JSON that accurately represents this physics scenario.")

        return "\n".join(prompt_parts)

    async def _generate_anthropic(self, prompt: str) -> str:
        """Generate with Anthropic Claude"""
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.1,  # Low temperature for deterministic output
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text

            # Extract JSON if wrapped in markdown
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.index("```") + 3
                end = content.index("```", start)
                content = content[start:end].strip()

            return content

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _generate_openai(self, prompt: str) -> str:
        """Generate with OpenAI GPT-4"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}  # Force JSON output
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def enhance_spec_from_feedback(self,
                                        current_spec: PhysicsSpec,
                                        feedback: str) -> PhysicsSpec:
        """
        Enhance existing PhysicsSpec based on user feedback

        Args:
            current_spec: Current PhysicsSpec
            feedback: User feedback for improvements

        Returns:
            Enhanced PhysicsSpec
        """
        prompt = f"""Current PhysicsSpec:
{json.dumps(current_spec.dict(), indent=2)}

User Feedback:
{feedback}

Generate an updated PhysicsSpec that incorporates the feedback while maintaining physical validity."""

        return await self.generate_physics_spec(prompt)

    async def generate_from_template(self,
                                    template_name: str,
                                    customizations: Dict[str, Any]) -> PhysicsSpec:
        """
        Generate PhysicsSpec from template with customizations

        Args:
            template_name: Name of template (pendulum, robot_arm, etc.)
            customizations: Custom parameters

        Returns:
            Customized PhysicsSpec
        """
        prompt = f"""Generate a {template_name} physics simulation with these customizations:
{json.dumps(customizations, indent=2)}

Use standard parameters for any unspecified values."""

        return await self.generate_physics_spec(prompt)

# Singleton instance
_physics_llm_client: Optional[PhysicsLLMClient] = None

def get_physics_llm_client() -> PhysicsLLMClient:
    """Get or create physics LLM client instance"""
    global _physics_llm_client
    if _physics_llm_client is None:
        _physics_llm_client = PhysicsLLMClient()
    return _physics_llm_client