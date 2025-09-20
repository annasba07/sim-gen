"""Deep coverage tests for all service implementations with real method calls."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
import json
import numpy as np
from datetime import datetime
import asyncio
import base64
import os
import tempfile
from typing import Optional, Dict, Any, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import all services for deep testing
from simgen.core.config import settings
from simgen.services.llm_client import LLMClient
from simgen.services.prompt_parser import PromptParser
from simgen.services.simulation_generator import SimulationGenerator
from simgen.services.dynamic_scene_composer import DynamicSceneComposer
from simgen.services.mjcf_compiler import MJCFCompiler
from simgen.services.mujoco_runtime import MuJoCoRuntime
from simgen.services.multimodal_enhancer import MultiModalEnhancer
from simgen.services.performance_optimizer import PerformanceOptimizer
from simgen.services.physics_llm_client import PhysicsLLMClient
from simgen.services.realtime_progress import RealtimeProgressManager
from simgen.services.sketch_analyzer import SketchAnalyzer
from simgen.services.streaming_protocol import StreamingProtocol


class TestLLMClientDeep:
    """Deep testing of LLM client with real method coverage."""

    @patch('openai.AsyncOpenAI')
    def test_llm_client_initialization_complete(self, mock_openai):
        """Test complete LLM client initialization."""
        client = LLMClient(api_key="test-key")

        # Test all attributes
        assert hasattr(client, 'client')
        assert hasattr(client, 'extract_entities')
        assert hasattr(client, 'generate_mjcf')
        assert hasattr(client, 'enhance_prompt')
        assert hasattr(client, 'validate_response')

    @patch('openai.AsyncOpenAI')
    async def test_extract_entities_comprehensive(self, mock_openai):
        """Test comprehensive entity extraction scenarios."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Test various response scenarios
        test_cases = [
            # Valid response
            {
                "input": "Create a pendulum with a ball",
                "response": json.dumps({
                    "objects": ["pendulum", "ball"],
                    "environment": "indoor",
                    "physics": {"gravity": -9.81, "damping": 0.1}
                }),
                "expected_objects": 2
            },
            # Complex response
            {
                "input": "Build a robot arm with 3 joints picking up a cube",
                "response": json.dumps({
                    "objects": ["robot_arm", "cube", "joints"],
                    "environment": "workshop",
                    "physics": {"gravity": -9.81, "friction": 0.5},
                    "actions": ["pick", "place"]
                }),
                "expected_objects": 3
            },
            # Minimal response
            {
                "input": "Simple ball",
                "response": json.dumps({
                    "objects": ["ball"]
                }),
                "expected_objects": 1
            }
        ]

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        for case in test_cases:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=case["response"]))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await client.extract_entities(case["input"])
            assert "objects" in result
            assert len(result["objects"]) == case["expected_objects"]

    @patch('openai.AsyncOpenAI')
    async def test_generate_mjcf_scenarios(self, mock_openai):
        """Test MJCF generation for various scenarios."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mjcf_templates = [
            # Simple object
            "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>",
            # Complex scene
            "<mujoco><worldbody><body><geom type='box'/><joint type='hinge'/></body></worldbody></mujoco>",
            # Multi-object scene
            "<mujoco><worldbody><geom type='sphere'/><geom type='box'/></worldbody></mujoco>"
        ]

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        for template in mjcf_templates:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=template))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            entities = {"objects": ["test_object"]}
            result = await client.generate_mjcf(entities)

            assert "<mujoco>" in result
            assert "</mujoco>" in result

    @patch('openai.AsyncOpenAI')
    async def test_enhance_prompt_functionality(self, mock_openai):
        """Test prompt enhancement functionality."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        enhanced_prompts = [
            "Create a realistic pendulum simulation with proper physics including gravity, air resistance, and initial velocity of 2 m/s",
            "Build a detailed robot arm simulation with 6 degrees of freedom, inverse kinematics, and object manipulation capabilities",
            "Generate a complex multi-body system with springs, dampers, and collision detection"
        ]

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        for enhanced in enhanced_prompts:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=enhanced))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            original = "simple prompt"
            result = await client.enhance_prompt(original)

            assert len(result) > len(original)
            assert "simulation" in result.lower()

    def test_validate_response_method(self):
        """Test response validation functionality."""
        client = LLMClient(api_key="test-key")

        # Test valid responses
        valid_responses = [
            '{"objects": ["ball"], "physics": {"gravity": -9.81}}',
            '{"objects": ["pendulum", "string"], "environment": "lab"}',
            '<mujoco><worldbody></worldbody></mujoco>'
        ]

        for response in valid_responses:
            assert client.validate_response(response) is True

        # Test invalid responses
        invalid_responses = [
            '',
            'invalid json',
            '{"incomplete":',
            None
        ]

        for response in invalid_responses:
            assert client.validate_response(response) is False


class TestPromptParserDeep:
    """Deep testing of prompt parser functionality."""

    def test_prompt_parser_complete_initialization(self):
        """Test complete prompt parser initialization."""
        mock_llm = Mock()
        parser = PromptParser(llm_client=mock_llm)

        # Test all attributes and methods
        assert hasattr(parser, 'llm_client')
        assert hasattr(parser, 'parse')
        assert hasattr(parser, 'validate_prompt')
        assert hasattr(parser, 'extract_keywords')
        assert hasattr(parser, 'identify_objects')
        assert hasattr(parser, 'parse_constraints')

    async def test_parse_comprehensive_scenarios(self):
        """Test parsing various prompt scenarios."""
        mock_llm = AsyncMock()
        parser = PromptParser(llm_client=mock_llm)

        test_prompts = [
            {
                "prompt": "Create a pendulum that swings back and forth",
                "expected_objects": ["pendulum"],
                "expected_actions": ["swing"]
            },
            {
                "prompt": "Build a robot arm with 3 joints that picks up a red cube",
                "expected_objects": ["robot_arm", "cube"],
                "expected_actions": ["pick_up"]
            },
            {
                "prompt": "Make two balls collide with each other in zero gravity",
                "expected_objects": ["ball", "ball"],
                "expected_constraints": ["zero_gravity", "collision"]
            }
        ]

        for test_case in test_prompts:
            mock_llm.extract_entities = AsyncMock(return_value={
                "objects": test_case["expected_objects"],
                "actions": test_case.get("expected_actions", []),
                "constraints": test_case.get("expected_constraints", [])
            })

            result = await parser.parse(test_case["prompt"])
            assert result is not None

    def test_extract_keywords_functionality(self):
        """Test keyword extraction from prompts."""
        mock_llm = Mock()
        parser = PromptParser(llm_client=mock_llm)

        test_cases = [
            {
                "prompt": "Create a bouncing ball with red color",
                "expected_keywords": ["bouncing", "ball", "red", "color"]
            },
            {
                "prompt": "Build a robotic arm with servo motors",
                "expected_keywords": ["robotic", "arm", "servo", "motors"]
            }
        ]

        for case in test_cases:
            keywords = parser.extract_keywords(case["prompt"])
            assert isinstance(keywords, list)
            assert len(keywords) > 0

    def test_identify_objects_method(self):
        """Test object identification in prompts."""
        mock_llm = Mock()
        parser = PromptParser(llm_client=mock_llm)

        test_cases = [
            {
                "prompt": "A ball and a cube colliding",
                "expected_count": 2
            },
            {
                "prompt": "Three spheres in a triangle formation",
                "expected_count": 3
            },
            {
                "prompt": "A complex robot with multiple joints",
                "expected_count": 1  # One main object (robot)
            }
        ]

        for case in test_cases:
            objects = parser.identify_objects(case["prompt"])
            assert isinstance(objects, list)

    def test_parse_constraints_functionality(self):
        """Test constraint parsing from prompts."""
        mock_llm = Mock()
        parser = PromptParser(llm_client=mock_llm)

        test_prompts = [
            "Create a ball with mass 5kg moving at 10 m/s",
            "Build a pendulum with length 2 meters and damping 0.1",
            "Make a robot arm with maximum speed 0.5 rad/s"
        ]

        for prompt in test_prompts:
            constraints = parser.parse_constraints(prompt)
            assert isinstance(constraints, dict)


class TestSimulationGeneratorDeep:
    """Deep testing of simulation generator."""

    def test_generator_complete_initialization(self):
        """Test complete generator initialization."""
        mock_llm = Mock()
        mock_parser = Mock()
        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        # Test all attributes
        assert hasattr(generator, 'llm_client')
        assert hasattr(generator, 'prompt_parser')
        assert hasattr(generator, 'generate')
        assert hasattr(generator, 'validate_mjcf')
        assert hasattr(generator, 'optimize_mjcf')
        assert hasattr(generator, 'add_physics_properties')

    @patch('simgen.services.simulation_generator.MJCFCompiler')
    async def test_generate_comprehensive_flow(self, mock_compiler_class):
        """Test complete generation flow with various scenarios."""
        # Setup comprehensive mocks
        mock_llm = AsyncMock()
        mock_parser = AsyncMock()
        mock_compiler = Mock()

        mock_compiler_class.return_value = mock_compiler

        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        test_scenarios = [
            {
                "prompt": "Simple ball",
                "entities": {"objects": ["ball"], "physics": {"gravity": -9.81}},
                "mjcf": "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"
            },
            {
                "prompt": "Complex robot arm",
                "entities": {"objects": ["robot", "arm"], "physics": {"gravity": -9.81, "friction": 0.5}},
                "mjcf": "<mujoco><worldbody><body><geom type='cylinder'/><joint type='hinge'/></body></worldbody></mujoco>"
            }
        ]

        for scenario in test_scenarios:
            mock_parser.parse = AsyncMock(return_value={
                "entities": scenario["entities"]
            })
            mock_llm.generate_mjcf = AsyncMock(return_value=scenario["mjcf"])
            mock_compiler.compile = Mock(return_value=scenario["mjcf"])
            mock_compiler.validate = Mock(return_value=True)

            result = await generator.generate(scenario["prompt"], user_id="test-user")
            assert result is not None

    def test_validate_mjcf_functionality(self):
        """Test MJCF validation functionality."""
        mock_llm = Mock()
        mock_parser = Mock()
        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        valid_mjcf_examples = [
            "<mujoco><worldbody></worldbody></mujoco>",
            "<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>",
            "<mujoco><worldbody><body><geom type='box'/></body></worldbody></mujoco>"
        ]

        for mjcf in valid_mjcf_examples:
            assert generator.validate_mjcf(mjcf) is True

        invalid_mjcf_examples = [
            "",
            "<invalid>xml</invalid>",
            "<mujoco><unclosed_tag></mujoco>"
        ]

        for mjcf in invalid_mjcf_examples:
            assert generator.validate_mjcf(mjcf) is False

    def test_optimize_mjcf_method(self):
        """Test MJCF optimization functionality."""
        mock_llm = Mock()
        mock_parser = Mock()
        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        input_mjcf = "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"
        optimized = generator.optimize_mjcf(input_mjcf)

        assert optimized is not None
        assert "<mujoco>" in optimized

    def test_add_physics_properties_method(self):
        """Test adding physics properties to MJCF."""
        mock_llm = Mock()
        mock_parser = Mock()
        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        base_mjcf = "<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>"
        physics_props = {
            "gravity": -9.81,
            "timestep": 0.001,
            "friction": 0.5
        }

        enhanced_mjcf = generator.add_physics_properties(base_mjcf, physics_props)
        assert enhanced_mjcf is not None
        assert len(enhanced_mjcf) >= len(base_mjcf)


class TestDynamicSceneComposerDeep:
    """Deep testing of dynamic scene composer."""

    def test_composer_complete_functionality(self):
        """Test complete scene composer functionality."""
        composer = DynamicSceneComposer()

        # Test all methods
        assert hasattr(composer, 'compose_scene')
        assert hasattr(composer, 'add_object')
        assert hasattr(composer, 'add_lighting')
        assert hasattr(composer, 'add_camera')
        assert hasattr(composer, 'optimize_scene')

    def test_compose_scene_comprehensive(self):
        """Test comprehensive scene composition."""
        composer = DynamicSceneComposer()

        complex_scene_specs = [
            {
                "name": "simple_pendulum",
                "objects": [
                    {"type": "sphere", "size": [0.1], "pos": [0, 0, 1]},
                    {"type": "cylinder", "size": [0.01, 1], "pos": [0, 0, 0.5]}
                ],
                "lights": [{"type": "directional", "intensity": 1.0, "direction": [0, 0, -1]}],
                "camera": {"pos": [2, 2, 2], "target": [0, 0, 0]}
            },
            {
                "name": "robot_workspace",
                "objects": [
                    {"type": "box", "size": [2, 2, 0.1], "pos": [0, 0, -0.05]},  # Table
                    {"type": "box", "size": [0.1, 0.1, 0.1], "pos": [0.5, 0.5, 0.05]}  # Cube
                ],
                "lights": [
                    {"type": "ambient", "intensity": 0.3},
                    {"type": "point", "intensity": 0.7, "pos": [1, 1, 2]}
                ]
            }
        ]

        for spec in complex_scene_specs:
            result = composer.compose_scene(spec)
            assert result is not None
            assert "<worldbody>" in result
            assert len(result) > 50  # Should be substantial

    def test_add_object_variations(self):
        """Test adding various object types."""
        composer = DynamicSceneComposer()

        base_mjcf = "<mujoco><worldbody></worldbody></mujoco>"

        object_specs = [
            {"type": "sphere", "radius": 0.1, "pos": [0, 0, 1]},
            {"type": "box", "size": [0.1, 0.2, 0.3], "pos": [1, 0, 0]},
            {"type": "cylinder", "radius": 0.05, "height": 0.5, "pos": [0, 1, 0]},
            {"type": "capsule", "radius": 0.03, "length": 0.2, "pos": [-1, 0, 0]}
        ]

        for obj_spec in object_specs:
            result = composer.add_object(base_mjcf, obj_spec)
            assert result != base_mjcf  # Should be modified
            assert len(result) > len(base_mjcf)

    def test_add_lighting_systems(self):
        """Test adding various lighting systems."""
        composer = DynamicSceneComposer()

        base_mjcf = "<mujoco><worldbody></worldbody></mujoco>"

        lighting_specs = [
            {"type": "directional", "intensity": 1.0, "direction": [0, 0, -1]},
            {"type": "point", "intensity": 0.8, "pos": [2, 2, 3]},
            {"type": "ambient", "intensity": 0.2},
            {"type": "spot", "intensity": 1.2, "pos": [1, 1, 2], "target": [0, 0, 0]}
        ]

        for light_spec in lighting_specs:
            result = composer.add_lighting(base_mjcf, light_spec)
            assert result is not None

    def test_add_camera_configurations(self):
        """Test adding various camera configurations."""
        composer = DynamicSceneComposer()

        base_mjcf = "<mujoco><worldbody></worldbody></mujoco>"

        camera_specs = [
            {"pos": [3, 3, 3], "target": [0, 0, 0], "fovy": 45},
            {"pos": [0, 5, 2], "target": [0, 0, 1], "fovy": 60},
            {"pos": [-2, 2, 4], "target": [0, 0, 0.5], "fovy": 30}
        ]

        for camera_spec in camera_specs:
            result = composer.add_camera(base_mjcf, camera_spec)
            assert result is not None

    def test_optimize_scene_performance(self):
        """Test scene optimization for performance."""
        composer = DynamicSceneComposer()

        # Create a complex scene that could benefit from optimization
        complex_mjcf = """
        <mujoco>
            <worldbody>
                <geom type='sphere' size='0.1'/>
                <geom type='sphere' size='0.1'/>
                <geom type='sphere' size='0.1'/>
                <geom type='box' size='0.1 0.1 0.1'/>
                <geom type='box' size='0.1 0.1 0.1'/>
            </worldbody>
        </mujoco>
        """

        optimized = composer.optimize_scene(complex_mjcf)
        assert optimized is not None
        assert "<mujoco>" in optimized


class TestMJCFCompilerDeep:
    """Deep testing of MJCF compiler functionality."""

    def test_compiler_complete_functionality(self):
        """Test complete compiler functionality."""
        compiler = MJCFCompiler()

        # Test all methods
        assert hasattr(compiler, 'compile')
        assert hasattr(compiler, 'validate')
        assert hasattr(compiler, 'optimize')
        assert hasattr(compiler, 'add_assets')
        assert hasattr(compiler, 'merge_mjcf')

    def test_validate_comprehensive_mjcf(self):
        """Test validation of various MJCF structures."""
        compiler = MJCFCompiler()

        valid_mjcf_examples = [
            # Minimal valid MJCF
            "<mujoco><worldbody></worldbody></mujoco>",

            # MJCF with geometry
            """<mujoco>
                <worldbody>
                    <geom type='sphere' size='0.1'/>
                </worldbody>
            </mujoco>""",

            # MJCF with bodies and joints
            """<mujoco>
                <worldbody>
                    <body>
                        <geom type='box' size='0.1 0.1 0.1'/>
                        <joint type='hinge' axis='1 0 0'/>
                    </body>
                </worldbody>
            </mujoco>""",

            # MJCF with assets
            """<mujoco>
                <asset>
                    <texture name='grid' type='2d' builtin='checker'/>
                    <material name='grid' texture='grid'/>
                </asset>
                <worldbody>
                    <geom type='plane' material='grid'/>
                </worldbody>
            </mujoco>"""
        ]

        for mjcf in valid_mjcf_examples:
            assert compiler.validate(mjcf) is True

        invalid_mjcf_examples = [
            "",
            "<invalid>not mjcf</invalid>",
            "<mujoco><unclosed_tag></mujoco>",
            "<mujoco><worldbody><geom type='invalid_type'/></worldbody></mujoco>"
        ]

        for mjcf in invalid_mjcf_examples:
            assert compiler.validate(mjcf) is False

    def test_compile_optimization_scenarios(self):
        """Test compilation with various optimization scenarios."""
        compiler = MJCFCompiler()

        test_mjcf_scenarios = [
            # Simple scene
            """<mujoco>
                <worldbody>
                    <geom type='sphere' size='0.1'/>
                </worldbody>
            </mujoco>""",

            # Multi-object scene
            """<mujoco>
                <worldbody>
                    <body name='obj1'>
                        <geom type='box' size='0.1 0.1 0.1'/>
                    </body>
                    <body name='obj2'>
                        <geom type='cylinder' size='0.05 0.2'/>
                    </body>
                </worldbody>
            </mujoco>""",

            # Scene with constraints
            """<mujoco>
                <worldbody>
                    <body name='pendulum'>
                        <geom type='sphere' size='0.1'/>
                        <joint type='hinge' axis='1 0 0'/>
                    </body>
                </worldbody>
            </mujoco>"""
        ]

        for mjcf in test_mjcf_scenarios:
            compiled = compiler.compile(mjcf)
            assert compiled is not None
            assert "<mujoco>" in compiled
            assert compiler.validate(compiled) is True

    def test_add_assets_functionality(self):
        """Test adding assets to MJCF."""
        compiler = MJCFCompiler()

        base_mjcf = "<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>"

        asset_specs = [
            {
                "textures": [
                    {"name": "grid", "type": "2d", "builtin": "checker"},
                    {"name": "skybox", "type": "skybox", "builtin": "gradient"}
                ],
                "materials": [
                    {"name": "grid_mat", "texture": "grid"},
                    {"name": "metal", "rgba": "0.7 0.7 0.8 1"}
                ]
            }
        ]

        for assets in asset_specs:
            result = compiler.add_assets(base_mjcf, assets)
            assert result is not None
            assert "asset" in result
            assert len(result) > len(base_mjcf)

    def test_merge_mjcf_files(self):
        """Test merging multiple MJCF files."""
        compiler = MJCFCompiler()

        mjcf_files = [
            "<mujoco><worldbody><geom type='sphere' name='ball'/></worldbody></mujoco>",
            "<mujoco><worldbody><geom type='box' name='cube'/></worldbody></mujoco>",
            "<mujoco><worldbody><geom type='cylinder' name='pole'/></worldbody></mujoco>"
        ]

        merged = compiler.merge_mjcf(mjcf_files)
        assert merged is not None
        assert "ball" in merged
        assert "cube" in merged
        assert "pole" in merged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])