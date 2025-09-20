import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import tempfile
import os
from pathlib import Path

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simgen.services.simulation_generator import (
    SimulationGenerator,
    SimulationGenerationError
)
from simgen.models.schemas import (
    SimulationRequest,
    ExtractedEntities,
    SimulationResponse as SimulationResult,
    ProgressUpdate as SimulationProgress
)


class TestSimulationGenerator:
    """Test cases for SimulationGenerator service."""

    @pytest.fixture
    def generator(self):
        """Create a SimulationGenerator instance."""
        return SimulationGenerator()

    @pytest.fixture
    def mock_request(self):
        """Create a mock simulation request."""
        return SimulationRequest(
            prompt="A robotic arm picking up balls",
            sketch_data=None,
            user_id="test_user_123",
            configuration={
                "quality": "high",
                "duration": 10,
                "fps": 30
            }
        )

    @pytest.fixture
    def mock_entities(self):
        """Create mock extracted entities."""
        return ExtractedEntities(
            main_objects=["robot_arm", "balls", "table"],
            environment="laboratory",
            physics_properties={
                "gravity": True,
                "friction": 0.5
            },
            interactions=["grasping", "lifting"],
            constraints=["workspace_limits"],
            initial_conditions={"arm_at_home": True},
            success_criteria=["all_balls_picked"],
            time_properties={"duration": 10},
            visual_style="realistic"
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_basic(self, generator, mock_request, mock_entities):
        """Test basic simulation generation."""
        with patch.object(generator, 'prompt_parser') as mock_parser:
            with patch.object(generator, 'scene_composer') as mock_composer:
                with patch.object(generator, 'mjcf_builder') as mock_builder:
                    # Setup mocks
                    mock_parser.parse_prompt = AsyncMock(return_value=mock_entities)

                    mock_scene = MagicMock()
                    mock_composer.compose_scene = AsyncMock(return_value=mock_scene)

                    mock_mjcf = "<mujoco>test</mujoco>"
                    mock_builder.build = AsyncMock(return_value=mock_mjcf)

                    # Generate simulation
                    result = await generator.generate(mock_request)

                    # Verify result
                    assert isinstance(result, SimulationResult)
                    assert result.mjcf_content == mock_mjcf
                    assert result.success is True
                    assert result.request_id is not None

                    # Verify method calls
                    mock_parser.parse_prompt.assert_called_once_with(mock_request.prompt)
                    mock_composer.compose_scene.assert_called_once()
                    mock_builder.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_sketch_data(self, generator, mock_request, mock_entities):
        """Test simulation generation with sketch data."""
        mock_request.sketch_data = "data:image/png;base64,iVBORw0KGgoAAAANS..."

        with patch.object(generator, 'vision_analyzer') as mock_vision:
            with patch.object(generator, 'prompt_parser') as mock_parser:
                with patch.object(generator, 'scene_composer') as mock_composer:
                    with patch.object(generator, 'mjcf_builder') as mock_builder:
                        # Setup mocks
                        mock_vision.analyze_sketch = AsyncMock(
                            return_value="A robotic arm with gripper"
                        )
                        mock_parser.parse_prompt = AsyncMock(return_value=mock_entities)
                        mock_composer.compose_scene = AsyncMock(return_value=MagicMock())
                        mock_builder.build = AsyncMock(return_value="<mujoco>test</mujoco>")

                        result = await generator.generate(mock_request)

                        # Verify vision analysis was called
                        mock_vision.analyze_sketch.assert_called_once_with(
                            mock_request.sketch_data
                        )
                        assert result.success is True

    @pytest.mark.asyncio
    async def test_generate_with_progress_callback(self, generator, mock_request, mock_entities):
        """Test simulation generation with progress callbacks."""
        progress_updates = []

        async def progress_callback(progress: SimulationProgress):
            progress_updates.append(progress)

        with patch.object(generator, 'prompt_parser') as mock_parser:
            with patch.object(generator, 'scene_composer') as mock_composer:
                with patch.object(generator, 'mjcf_builder') as mock_builder:
                    mock_parser.parse_prompt = AsyncMock(return_value=mock_entities)
                    mock_composer.compose_scene = AsyncMock(return_value=MagicMock())
                    mock_builder.build = AsyncMock(return_value="<mujoco>test</mujoco>")

                    result = await generator.generate(
                        mock_request,
                        progress_callback=progress_callback
                    )

                    # Should have progress updates
                    assert len(progress_updates) > 0

                    # Check progress stages
                    stages = [p.stage for p in progress_updates]
                    assert any("parsing" in s.lower() for s in stages)
                    assert any("composing" in s.lower() or "generating" in s.lower() for s in stages)

    @pytest.mark.asyncio
    async def test_generate_with_caching(self, generator, mock_request):
        """Test that identical prompts use cached results."""
        with patch.object(generator, 'cache') as mock_cache:
            with patch.object(generator, 'prompt_parser') as mock_parser:
                with patch.object(generator, 'scene_composer') as mock_composer:
                    with patch.object(generator, 'mjcf_builder') as mock_builder:
                        # Setup mocks
                        mock_cache.get = AsyncMock(return_value=None)
                        mock_cache.set = AsyncMock()

                        mock_parser.parse_prompt = AsyncMock(return_value=MagicMock())
                        mock_composer.compose_scene = AsyncMock(return_value=MagicMock())
                        mock_builder.build = AsyncMock(return_value="<mujoco>cached</mujoco>")

                        # First generation
                        result1 = await generator.generate(mock_request)

                        # Cache should be set
                        mock_cache.set.assert_called_once()

                        # Second generation with same request
                        mock_cache.get = AsyncMock(return_value=result1)
                        result2 = await generator.generate(mock_request)

                        # Should use cached result
                        assert result2 == result1

    @pytest.mark.asyncio
    async def test_generate_error_handling_parse_error(self, generator, mock_request):
        """Test error handling when parsing fails."""
        with patch.object(generator, 'prompt_parser') as mock_parser:
            mock_parser.parse_prompt = AsyncMock(
                side_effect=Exception("Failed to parse prompt")
            )

            with pytest.raises(SimulationGenerationError) as exc_info:
                await generator.generate(mock_request)

            assert "parse" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_error_handling_compose_error(self, generator, mock_request):
        """Test error handling when scene composition fails."""
        with patch.object(generator, 'prompt_parser') as mock_parser:
            with patch.object(generator, 'scene_composer') as mock_composer:
                mock_parser.parse_prompt = AsyncMock(return_value=MagicMock())
                mock_composer.compose_scene = AsyncMock(
                    side_effect=Exception("Failed to compose scene")
                )

                with pytest.raises(SimulationGenerationError) as exc_info:
                    await generator.generate(mock_request)

                assert "compose" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_with_timeout(self, generator, mock_request):
        """Test generation with timeout."""
        with patch.object(generator, 'prompt_parser') as mock_parser:
            # Simulate long-running operation
            async def slow_parse(*args):
                await asyncio.sleep(10)
                return MagicMock()

            mock_parser.parse_prompt = slow_parse

            generator.timeout = 1  # Set 1 second timeout

            with pytest.raises(asyncio.TimeoutError):
                await generator.generate(mock_request)

    @pytest.mark.asyncio
    async def test_save_simulation_files(self, generator):
        """Test saving simulation files to disk."""
        mjcf_content = """<mujoco>
            <worldbody>
                <light diffuse="1 1 1"/>
                <geom type="sphere" size="0.1"/>
            </worldbody>
        </mujoco>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_sim.xml"

            await generator.save_simulation(mjcf_content, output_path)

            # Verify file was created
            assert output_path.exists()

            # Verify content
            with open(output_path, 'r') as f:
                content = f.read()
                assert "<mujoco>" in content
                assert "sphere" in content

    @pytest.mark.asyncio
    async def test_validate_mjcf(self, generator):
        """Test MJCF validation."""
        valid_mjcf = """<mujoco>
            <worldbody>
                <geom type="box" size="1 1 1"/>
            </worldbody>
        </mujoco>"""

        invalid_mjcf = """<invalid>
            This is not valid MJCF
        </invalid>"""

        # Valid MJCF should pass
        assert await generator.validate_mjcf(valid_mjcf) is True

        # Invalid MJCF should fail
        assert await generator.validate_mjcf(invalid_mjcf) is False

    @pytest.mark.asyncio
    async def test_generate_with_quality_settings(self, generator, mock_request):
        """Test generation with different quality settings."""
        quality_settings = ["low", "medium", "high", "ultra"]

        for quality in quality_settings:
            mock_request.configuration["quality"] = quality

            with patch.object(generator, 'prompt_parser') as mock_parser:
                with patch.object(generator, 'scene_composer') as mock_composer:
                    with patch.object(generator, 'mjcf_builder') as mock_builder:
                        mock_parser.parse_prompt = AsyncMock(return_value=MagicMock())
                        mock_composer.compose_scene = AsyncMock(return_value=MagicMock())

                        # Quality should affect MJCF generation
                        expected_shadowsize = {
                            "low": 2048,
                            "medium": 4096,
                            "high": 8192,
                            "ultra": 16384
                        }

                        mjcf = f'<mujoco><visual><map shadowsize="{expected_shadowsize[quality]}"/></visual></mujoco>'
                        mock_builder.build = AsyncMock(return_value=mjcf)

                        result = await generator.generate(mock_request)

                        assert str(expected_shadowsize[quality]) in result.mjcf_content

    @pytest.mark.asyncio
    async def test_generate_with_physics_settings(self, generator, mock_request):
        """Test generation with custom physics settings."""
        mock_request.configuration["physics"] = {
            "timestep": 0.001,
            "gravity": [0, 0, -9.81],
            "solver": "Newton"
        }

        with patch.object(generator, 'prompt_parser') as mock_parser:
            with patch.object(generator, 'scene_composer') as mock_composer:
                with patch.object(generator, 'mjcf_builder') as mock_builder:
                    mock_parser.parse_prompt = AsyncMock(return_value=MagicMock())
                    mock_composer.compose_scene = AsyncMock(return_value=MagicMock())

                    # Should pass physics settings to builder
                    mock_builder.build = AsyncMock(return_value="<mujoco>test</mujoco>")

                    await generator.generate(mock_request)

                    # Verify physics settings were passed
                    call_args = mock_builder.build.call_args
                    assert call_args is not None

    @pytest.mark.asyncio
    async def test_generate_batch(self, generator):
        """Test batch generation of multiple simulations."""
        requests = [
            SimulationRequest(
                prompt=f"Simulation {i}",
                user_id=f"user_{i}"
            )
            for i in range(3)
        ]

        with patch.object(generator, 'generate') as mock_generate:
            mock_generate.return_value = SimulationResult(
                success=True,
                mjcf_content="<mujoco>test</mujoco>",
                request_id="test_id",
                processing_time=1.0
            )

            results = await generator.generate_batch(requests)

            assert len(results) == 3
            assert all(r.success for r in results)
            assert mock_generate.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_with_template_override(self, generator, mock_request):
        """Test generation with template override."""
        mock_request.template_id = "pendulum_template"

        with patch.object(generator, 'template_manager') as mock_template:
            with patch.object(generator, 'mjcf_builder') as mock_builder:
                mock_template.get_template = AsyncMock(
                    return_value="<mujoco>pendulum_template</mujoco>"
                )
                mock_builder.build_from_template = AsyncMock(
                    return_value="<mujoco>customized_pendulum</mujoco>"
                )

                result = await generator.generate(mock_request)

                # Should use template
                mock_template.get_template.assert_called_once_with("pendulum_template")
                assert "pendulum" in result.mjcf_content

    @pytest.mark.asyncio
    async def test_post_processing_pipeline(self, generator):
        """Test post-processing pipeline after generation."""
        raw_mjcf = "<mujoco><worldbody><geom/></worldbody></mujoco>"

        with patch.object(generator, 'post_processors') as mock_processors:
            processors = [
                AsyncMock(return_value=raw_mjcf + "<!-- processed1 -->"),
                AsyncMock(return_value=raw_mjcf + "<!-- processed2 -->")
            ]
            mock_processors.get_all = Mock(return_value=processors)

            processed = await generator.apply_post_processing(raw_mjcf)

            # All processors should be applied
            assert "processed1" in processed or "processed2" in processed

    @pytest.mark.asyncio
    async def test_metadata_generation(self, generator, mock_request):
        """Test that metadata is generated for simulations."""
        with patch.object(generator, 'prompt_parser') as mock_parser:
            with patch.object(generator, 'scene_composer') as mock_composer:
                with patch.object(generator, 'mjcf_builder') as mock_builder:
                    mock_parser.parse_prompt = AsyncMock(return_value=MagicMock())
                    mock_composer.compose_scene = AsyncMock(return_value=MagicMock())
                    mock_builder.build = AsyncMock(return_value="<mujoco>test</mujoco>")

                    result = await generator.generate(mock_request)

                    # Should have metadata
                    assert hasattr(result, 'metadata')
                    metadata = result.metadata

                    assert metadata.get('prompt') == mock_request.prompt
                    assert metadata.get('user_id') == mock_request.user_id
                    assert 'timestamp' in metadata
                    assert 'version' in metadata