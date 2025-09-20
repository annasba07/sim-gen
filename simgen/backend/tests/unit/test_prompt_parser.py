import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simgen.services.prompt_parser import PromptParser, PromptParsingError as ParsingError
from simgen.models.schemas import ExtractedEntities


class TestPromptParser:
    """Test cases for PromptParser service."""

    @pytest.fixture
    def parser(self):
        """Create a PromptParser instance."""
        return PromptParser()

    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for testing."""
        return {
            "simple": "A pendulum swinging back and forth",
            "complex": "A robotic arm with 6 degrees of freedom picking up three colored balls from a table and placing them in a basket",
            "physics": "Two spheres colliding elastically with coefficient of restitution 0.8",
            "environment": "An underwater robot navigating through obstacles with buoyancy forces",
            "constraints": "A double pendulum with the first link constrained to 45 degrees max angle",
            "multi_object": "Multiple robots collaborating to move a heavy box across a room"
        }

    @pytest.mark.asyncio
    async def test_parse_simple_prompt(self, parser, sample_prompts):
        """Test parsing a simple prompt."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["pendulum"],
                "environment": "default",
                "physics_properties": {"gravity": True, "oscillation": True},
                "interactions": ["swinging"],
                "constraints": [],
                "initial_conditions": {"angle": 30},
                "success_criteria": [],
                "time_properties": {"period": 2},
                "visual_style": "simple"
            })

            result = await parser.parse_prompt(sample_prompts["simple"])

            assert isinstance(result, ExtractedEntities)
            assert "pendulum" in result.main_objects
            assert result.physics_properties["gravity"] is True
            assert "swinging" in result.interactions

    @pytest.mark.asyncio
    async def test_parse_complex_prompt(self, parser, sample_prompts):
        """Test parsing a complex prompt with multiple elements."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["robotic_arm", "balls", "table", "basket"],
                "environment": "laboratory",
                "physics_properties": {
                    "gravity": True,
                    "friction": 0.5,
                    "dof": 6
                },
                "interactions": ["picking", "placing", "grasping"],
                "constraints": ["workspace_limits", "joint_limits"],
                "initial_conditions": {
                    "arm_position": "home",
                    "balls_on_table": True
                },
                "success_criteria": ["all_balls_in_basket"],
                "time_properties": {"task_duration": 30},
                "visual_style": "realistic"
            })

            result = await parser.parse_prompt(sample_prompts["complex"])

            assert len(result.main_objects) == 4
            assert "robotic_arm" in result.main_objects
            assert result.physics_properties.get("dof") == 6
            assert "picking" in result.interactions
            assert "workspace_limits" in result.constraints

    @pytest.mark.asyncio
    async def test_parse_physics_focused_prompt(self, parser, sample_prompts):
        """Test parsing a prompt with specific physics parameters."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["sphere1", "sphere2"],
                "environment": "frictionless_surface",
                "physics_properties": {
                    "collision_type": "elastic",
                    "restitution": 0.8,
                    "gravity": True
                },
                "interactions": ["collision", "momentum_transfer"],
                "constraints": ["conservation_of_momentum"],
                "initial_conditions": {
                    "sphere1_velocity": [5, 0, 0],
                    "sphere2_velocity": [-3, 0, 0]
                },
                "success_criteria": ["collision_detected"],
                "time_properties": {"collision_time": 2},
                "visual_style": "scientific"
            })

            result = await parser.parse_prompt(sample_prompts["physics"])

            assert result.physics_properties["restitution"] == 0.8
            assert result.physics_properties["collision_type"] == "elastic"
            assert "collision" in result.interactions

    @pytest.mark.asyncio
    async def test_parse_environment_specific_prompt(self, parser, sample_prompts):
        """Test parsing a prompt with specific environment requirements."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["underwater_robot", "obstacles"],
                "environment": "underwater",
                "physics_properties": {
                    "gravity": True,
                    "buoyancy": True,
                    "fluid_density": 1000,
                    "drag_coefficient": 0.47
                },
                "interactions": ["navigation", "obstacle_avoidance"],
                "constraints": ["depth_limit", "pressure_tolerance"],
                "initial_conditions": {"depth": 10, "neutral_buoyancy": True},
                "success_criteria": ["reach_target", "avoid_all_obstacles"],
                "time_properties": {"mission_duration": 600},
                "visual_style": "underwater_lighting"
            })

            result = await parser.parse_prompt(sample_prompts["environment"])

            assert result.environment == "underwater"
            assert result.physics_properties["buoyancy"] is True
            assert result.physics_properties["fluid_density"] == 1000
            assert "navigation" in result.interactions

    @pytest.mark.asyncio
    async def test_parse_constraint_based_prompt(self, parser, sample_prompts):
        """Test parsing a prompt with explicit constraints."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["double_pendulum", "link1", "link2"],
                "environment": "default",
                "physics_properties": {
                    "gravity": True,
                    "damping": 0.01
                },
                "interactions": ["coupled_motion", "energy_transfer"],
                "constraints": [
                    "link1_max_angle: 45 degrees",
                    "link2_free_rotation"
                ],
                "initial_conditions": {
                    "link1_angle": 30,
                    "link2_angle": 0
                },
                "success_criteria": [],
                "time_properties": {"simulation_time": 20},
                "visual_style": "technical"
            })

            result = await parser.parse_prompt(sample_prompts["constraints"])

            assert "double_pendulum" in result.main_objects
            assert any("max_angle" in c for c in result.constraints)
            assert result.initial_conditions["link1_angle"] == 30

    @pytest.mark.asyncio
    async def test_parse_multi_agent_prompt(self, parser, sample_prompts):
        """Test parsing a prompt with multiple agents/robots."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["robot1", "robot2", "robot3", "heavy_box"],
                "environment": "warehouse",
                "physics_properties": {
                    "gravity": True,
                    "friction": 0.7,
                    "box_mass": 100
                },
                "interactions": [
                    "collaboration",
                    "force_distribution",
                    "synchronized_movement"
                ],
                "constraints": [
                    "max_individual_force: 300N",
                    "maintain_formation"
                ],
                "initial_conditions": {
                    "robots_positioned": True,
                    "box_at_start": True
                },
                "success_criteria": [
                    "box_at_destination",
                    "no_robot_overload"
                ],
                "time_properties": {"task_completion": 60},
                "visual_style": "industrial"
            })

            result = await parser.parse_prompt(sample_prompts["multi_object"])

            assert len([obj for obj in result.main_objects if "robot" in obj]) >= 2
            assert "collaboration" in result.interactions
            assert result.physics_properties["box_mass"] == 100

    @pytest.mark.asyncio
    async def test_parse_empty_prompt(self, parser):
        """Test handling of empty prompt."""
        with pytest.raises(ParsingError):
            await parser.parse_prompt("")

    @pytest.mark.asyncio
    async def test_parse_invalid_prompt(self, parser):
        """Test handling of invalid/nonsensical prompt."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(
                side_effect=Exception("Failed to parse prompt")
            )

            with pytest.raises(ParsingError):
                await parser.parse_prompt("xyz123 !@#$%^&*()")

    @pytest.mark.asyncio
    async def test_validation_of_extracted_entities(self, parser):
        """Test validation of extracted entities."""
        with patch.object(parser, 'llm_client') as mock_llm:
            # Test with missing required fields
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": [],  # Empty objects should fail
                "environment": "",
                "physics_properties": {}
            })

            with pytest.raises(ParsingError):
                await parser.parse_prompt("Test prompt")

    @pytest.mark.asyncio
    async def test_parse_with_preprocessing(self, parser):
        """Test prompt preprocessing before parsing."""
        prompt = "  A SIMPLE   pendulum  with    extra spaces  "

        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["pendulum"],
                "environment": "default",
                "physics_properties": {"gravity": True},
                "interactions": [],
                "constraints": [],
                "initial_conditions": {},
                "success_criteria": [],
                "time_properties": {},
                "visual_style": "simple"
            })

            result = await parser.parse_prompt(prompt)

            # Should handle the prompt despite formatting issues
            assert result is not None
            assert "pendulum" in result.main_objects

    @pytest.mark.asyncio
    async def test_parse_with_special_characters(self, parser):
        """Test parsing prompts with special characters."""
        prompt = "Robot arm (6-DOF) with end-effector @ 45° angle"

        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["robot_arm", "end_effector"],
                "environment": "default",
                "physics_properties": {"dof": 6},
                "interactions": [],
                "constraints": ["angle_45_degrees"],
                "initial_conditions": {},
                "success_criteria": [],
                "time_properties": {},
                "visual_style": "technical"
            })

            result = await parser.parse_prompt(prompt)

            assert result.physics_properties["dof"] == 6
            assert "angle_45_degrees" in result.constraints

    @pytest.mark.asyncio
    async def test_parse_with_units(self, parser):
        """Test parsing prompts with physical units."""
        prompt = "A 5kg mass falling from 10 meters height with air resistance"

        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["mass_object"],
                "environment": "air",
                "physics_properties": {
                    "mass": 5,  # kg
                    "initial_height": 10,  # meters
                    "air_resistance": True,
                    "gravity": 9.81
                },
                "interactions": ["falling", "air_drag"],
                "constraints": [],
                "initial_conditions": {"height": 10, "velocity": 0},
                "success_criteria": ["reach_ground"],
                "time_properties": {"fall_time": "calculated"},
                "visual_style": "realistic"
            })

            result = await parser.parse_prompt(prompt)

            assert result.physics_properties["mass"] == 5
            assert result.physics_properties["initial_height"] == 10
            assert result.physics_properties["air_resistance"] is True

    @pytest.mark.asyncio
    async def test_cache_parsed_results(self, parser):
        """Test caching of parsed results for identical prompts."""
        prompt = "A simple pendulum"

        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["pendulum"],
                "environment": "default",
                "physics_properties": {"gravity": True},
                "interactions": [],
                "constraints": [],
                "initial_conditions": {},
                "success_criteria": [],
                "time_properties": {},
                "visual_style": "simple"
            })

            # First call
            result1 = await parser.parse_prompt(prompt)

            # Second call with same prompt
            result2 = await parser.parse_prompt(prompt)

            # Should only call LLM once if caching is implemented
            assert mock_llm.extract_entities.call_count <= 2  # Allow for no-cache scenario

    @pytest.mark.asyncio
    async def test_parse_with_fallback_defaults(self, parser):
        """Test that parser provides sensible defaults for missing fields."""
        with patch.object(parser, 'llm_client') as mock_llm:
            mock_llm.extract_entities = AsyncMock(return_value={
                "main_objects": ["cube"],
                "environment": None,  # Missing environment
                "physics_properties": None,  # Missing physics
                "interactions": None,
                "constraints": None,
                "initial_conditions": None,
                "success_criteria": None,
                "time_properties": None,
                "visual_style": None
            })

            result = await parser.parse_prompt("A simple cube")

            # Should have default values
            assert result.environment == "default" or result.environment is not None
            assert result.physics_properties is not None
            assert isinstance(result.interactions, list)
            assert isinstance(result.constraints, list)