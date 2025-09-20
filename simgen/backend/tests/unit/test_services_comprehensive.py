"""Comprehensive unit tests for backend services to increase coverage."""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
from typing import Dict, Any, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simgen.services.prompt_parser import PromptParser, PromptParsingError
from simgen.services.simulation_generator import SimulationGenerator, SimulationGenerationError
from simgen.services.llm_client import LLMClient, LLMError
from simgen.services.dynamic_scene_composer import DynamicSceneComposer, MenagerieModel, SceneComposition
from simgen.models.schemas import (
    SimulationRequest,
    ExtractedEntities,
    EnvironmentSchema,
    GeometrySchema,
    MaterialSchema,
    ObjectSchema,
    ConstraintSchema,
    SimulationResponse,
    ProgressUpdate
)


class TestLLMClient:
    """Comprehensive tests for LLM Client."""

    @pytest.fixture
    def mock_anthropic(self):
        with patch('simgen.services.llm_client.anthropic') as mock:
            yield mock

    @pytest.fixture
    def llm_client(self, mock_anthropic):
        return LLMClient()

    def test_llm_client_initialization(self):
        """Test LLM client initialization."""
        client = LLMClient()
        assert client is not None
        assert hasattr(client, 'anthropic_client')

    @pytest.mark.asyncio
    async def test_extract_entities_success(self, llm_client, mock_anthropic):
        """Test successful entity extraction."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text=json.dumps({
                "entities": {
                    "main_objects": ["pendulum", "support"],
                    "environment": {
                        "type": "laboratory",
                        "lighting": "bright",
                        "temperature": 20
                    },
                    "physics_properties": {
                        "gravity": -9.81,
                        "air_resistance": 0.01
                    },
                    "interactions": ["swinging", "oscillation"],
                    "constraints": ["fixed_pivot"]
                }
            }))
        ]

        mock_anthropic.Anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)

        result = await llm_client.extract_entities("A pendulum swinging")

        assert "entities" in result
        assert "main_objects" in result["entities"]
        assert "pendulum" in result["entities"]["main_objects"]

    @pytest.mark.asyncio
    async def test_extract_entities_error(self, llm_client, mock_anthropic):
        """Test entity extraction with API error."""
        mock_anthropic.Anthropic.return_value.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(LLMError):
            await llm_client.extract_entities("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_mjcf_success(self, llm_client, mock_anthropic):
        """Test MJCF generation."""
        entities = {
            "main_objects": ["box"],
            "environment": {"type": "ground"},
            "physics_properties": {"gravity": -9.81}
        }

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="<mujoco><worldbody><geom type='box'/></worldbody></mujoco>")
        ]

        mock_anthropic.Anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)

        result = await llm_client.generate_mjcf(entities)

        assert "<mujoco>" in result
        assert "box" in result

    @pytest.mark.asyncio
    async def test_test_connection(self, llm_client, mock_anthropic):
        """Test connection testing."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="pong")]

        mock_anthropic.Anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)

        result = await llm_client.test_connection()

        assert result is True


class TestPromptParser:
    """Comprehensive tests for PromptParser."""

    @pytest.fixture
    def mock_llm_client(self):
        return Mock(spec=LLMClient)

    @pytest.fixture
    def parser(self, mock_llm_client):
        return PromptParser(mock_llm_client)

    @pytest.mark.asyncio
    async def test_parse_simple_prompt(self, parser, mock_llm_client):
        """Test parsing a simple prompt."""
        mock_llm_client.extract_entities = AsyncMock(return_value={
            "entities": {
                "main_objects": ["ball"],
                "environment": {
                    "type": "ground",
                    "size": [10, 10, 0.1]
                },
                "physics_properties": {"gravity": -9.81},
                "interactions": ["bouncing"],
                "constraints": []
            }
        })

        # Use the parse method that exists
        result = await parser.parse(prompt="A bouncing ball")

        assert isinstance(result, ExtractedEntities)
        assert "ball" in result.main_objects

    @pytest.mark.asyncio
    async def test_parse_complex_prompt(self, parser, mock_llm_client):
        """Test parsing a complex prompt with multiple objects."""
        mock_llm_client.extract_entities = AsyncMock(return_value={
            "entities": {
                "main_objects": ["robot_arm", "table", "cubes"],
                "environment": {
                    "type": "laboratory",
                    "lighting": "bright"
                },
                "physics_properties": {
                    "gravity": -9.81,
                    "friction": 0.5
                },
                "interactions": ["grasping", "stacking"],
                "constraints": ["workspace_limits", "joint_limits"]
            }
        })

        result = await parser.parse(
            prompt="A robot arm picking up and stacking colored cubes on a table"
        )

        assert len(result.main_objects) >= 3
        assert "robot_arm" in result.main_objects
        assert len(result.interactions) >= 2

    @pytest.mark.asyncio
    async def test_parse_empty_prompt(self, parser, mock_llm_client):
        """Test parsing an empty prompt."""
        with pytest.raises(PromptParsingError):
            await parser.parse(prompt="")

    @pytest.mark.asyncio
    async def test_parse_with_sketch(self, parser, mock_llm_client):
        """Test parsing with sketch data."""
        mock_llm_client.extract_entities = AsyncMock(return_value={
            "entities": {
                "main_objects": ["pendulum"],
                "environment": {"type": "default"},
                "physics_properties": {"gravity": -9.81},
                "interactions": ["swinging"],
                "constraints": ["fixed_pivot"]
            }
        })

        result = await parser.parse(
            prompt="Make this swing",
            sketch_data="data:image/png;base64,..."
        )

        assert "pendulum" in result.main_objects

    @pytest.mark.asyncio
    async def test_parse_error_handling(self, parser, mock_llm_client):
        """Test error handling during parsing."""
        mock_llm_client.extract_entities = AsyncMock(
            side_effect=LLMError("API Error")
        )

        with pytest.raises(PromptParsingError) as exc_info:
            await parser.parse(prompt="Test prompt")

        assert "Failed to parse" in str(exc_info.value)


class TestSimulationGenerator:
    """Comprehensive tests for SimulationGenerator."""

    @pytest.fixture
    def mock_llm_client(self):
        return Mock(spec=LLMClient)

    @pytest.fixture
    def generator(self, mock_llm_client):
        return SimulationGenerator(mock_llm_client)

    @pytest.fixture
    def mock_request(self):
        return SimulationRequest(
            prompt="A bouncing ball",
            session_id="test_session",
            generation_method="dynamic"
        )

    @pytest.fixture
    def mock_entities(self):
        return ExtractedEntities(
            main_objects=["ball"],
            environment=EnvironmentSchema(
                type="ground",
                size=[10, 10, 0.1]
            ),
            physics_properties={"gravity": -9.81},
            interactions=["bouncing"],
            constraints=[]
        )

    @pytest.mark.asyncio
    async def test_generate_simple_simulation(self, generator, mock_llm_client, mock_request):
        """Test generating a simple simulation."""
        # Mock the parse method
        mock_entities = {
            "main_objects": ["ball"],
            "environment": {"type": "ground"},
            "physics_properties": {"gravity": -9.81},
            "interactions": ["bouncing"],
            "constraints": []
        }

        mock_llm_client.extract_entities = AsyncMock(return_value={
            "entities": mock_entities
        })

        mock_llm_client.generate_mjcf = AsyncMock(
            return_value="<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>"
        )

        # Mock parser
        with patch.object(generator, 'parser') as mock_parser:
            mock_parser.parse = AsyncMock(return_value=ExtractedEntities(
                main_objects=["ball"],
                environment=EnvironmentSchema(type="ground"),
                physics_properties={"gravity": -9.81},
                interactions=["bouncing"],
                constraints=[]
            ))

            result = await generator.generate(mock_request)

            assert result is not None
            assert isinstance(result, dict)
            assert "mjcf" in result or "mjcf_content" in result

    @pytest.mark.asyncio
    async def test_generate_with_progress_callback(self, generator, mock_llm_client, mock_request):
        """Test generation with progress callbacks."""
        progress_updates = []

        async def progress_callback(update: Dict[str, Any]):
            progress_updates.append(update)

        mock_llm_client.extract_entities = AsyncMock(return_value={
            "entities": {
                "main_objects": ["cube"],
                "environment": {"type": "ground"},
                "physics_properties": {},
                "interactions": [],
                "constraints": []
            }
        })

        mock_llm_client.generate_mjcf = AsyncMock(
            return_value="<mujoco><worldbody><geom type='box'/></worldbody></mujoco>"
        )

        with patch.object(generator, 'parser') as mock_parser:
            mock_parser.parse = AsyncMock(return_value=ExtractedEntities(
                main_objects=["cube"],
                environment=EnvironmentSchema(type="ground"),
                physics_properties={},
                interactions=[],
                constraints=[]
            ))

            result = await generator.generate(
                mock_request,
                progress_callback=progress_callback
            )

            assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, generator, mock_llm_client, mock_request):
        """Test error handling during generation."""
        mock_llm_client.extract_entities = AsyncMock(
            side_effect=LLMError("Generation failed")
        )

        with pytest.raises(SimulationGenerationError):
            await generator.generate(mock_request)

    def test_validate_request(self, generator):
        """Test request validation."""
        # Valid request
        valid_request = SimulationRequest(
            prompt="Test prompt",
            session_id="session123"
        )
        assert generator.validate_request(valid_request) is True

        # Invalid request (empty prompt)
        invalid_request = SimulationRequest(
            prompt="",
            session_id="session123"
        )
        assert generator.validate_request(invalid_request) is False


class TestDynamicSceneComposer:
    """Comprehensive tests for DynamicSceneComposer."""

    @pytest.fixture
    def composer(self):
        return DynamicSceneComposer()

    @pytest.fixture
    def mock_entities(self):
        return ExtractedEntities(
            main_objects=["robot_arm", "table", "balls"],
            environment=EnvironmentSchema(
                type="laboratory",
                lighting="bright"
            ),
            physics_properties={
                "gravity": -9.81,
                "friction": 0.5
            },
            interactions=["grasping", "placing"],
            constraints=["workspace_limits"],
            initial_conditions={"arm_position": "home"},
            success_criteria=["all_balls_sorted"],
            time_properties={"duration": 30},
            visual_style="realistic"
        )

    @pytest.mark.asyncio
    async def test_compose_basic_scene(self, composer, mock_entities):
        """Test basic scene composition."""
        with patch.object(composer, 'model_library') as mock_library:
            mock_library.search_models = AsyncMock(return_value=[
                MenagerieModel(
                    name="franka_panda",
                    category="arm",
                    xml_path="/models/panda.xml",
                    assets_path="/models/assets",
                    description="Franka Panda robot arm",
                    dof=7,
                    license="Apache 2.0"
                )
            ])

            mock_library.initialize = AsyncMock(return_value=True)

            result = await composer.compose_scene(
                "Robot arm picking up balls",
                mock_entities
            )

            assert isinstance(result, SceneComposition)
            assert len(result.main_models) > 0

    @pytest.mark.asyncio
    async def test_generate_lighting_setup(self, composer):
        """Test lighting setup generation."""
        lighting = await composer._generate_lighting_setup("cinematic")

        assert "lights" in lighting
        assert isinstance(lighting["lights"], list)
        assert len(lighting["lights"]) >= 3  # Multi-light setup

    @pytest.mark.asyncio
    async def test_generate_materials(self, composer):
        """Test material generation."""
        materials = await composer._generate_materials(["metal", "plastic", "glass"])

        assert "metal" in materials
        assert "plastic" in materials
        assert "glass" in materials

        # Check material properties
        for material_name, properties in materials.items():
            assert "rgba" in properties
            assert isinstance(properties["rgba"], list)
            assert len(properties["rgba"]) == 4

    @pytest.mark.asyncio
    async def test_generate_mjcf_from_composition(self, composer):
        """Test MJCF generation from scene composition."""
        composition = SceneComposition(
            main_models=[
                MenagerieModel(
                    name="test_robot",
                    category="arm",
                    xml_path="/test.xml",
                    assets_path="/assets",
                    description="Test robot",
                    dof=6,
                    license="MIT"
                )
            ],
            environment_settings={
                "gravity": [0, 0, -9.81],
                "timestep": 0.002
            },
            physics_constraints=[],
            lighting_setup={
                "lights": [
                    {"pos": [0, 0, 10], "dir": [0, 0, -1], "diffuse": [1, 1, 1]}
                ]
            },
            camera_angles=[
                {"name": "main", "pos": [2, 2, 2], "euler": [0, -30, 0]}
            ],
            materials={
                "default": {"rgba": [0.8, 0.8, 0.8, 1.0]}
            }
        )

        mjcf = await composer.generate_mjcf(composition)

        assert "<mujoco>" in mjcf
        assert "gravity" in mjcf
        assert "<light" in mjcf
        assert "<camera" in mjcf