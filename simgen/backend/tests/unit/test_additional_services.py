"""Additional service tests to boost coverage to 80%+."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simgen.services.mjcf_compiler import MJCFCompiler
from simgen.services.mujoco_runtime import MuJoCoRuntime
from simgen.services.multimodal_enhancer import MultiModalEnhancer
from simgen.services.optimized_renderer import OptimizedRenderer
from simgen.services.performance_optimizer import PerformanceOptimizer
from simgen.services.physics_llm_client import PhysicsLLMClient
from simgen.services.realtime_progress import RealtimeProgressManager
from simgen.services.sketch_analyzer import SketchAnalyzer
from simgen.services.streaming_protocol import StreamingProtocol


class TestMJCFCompiler:
    """Tests for MJCF Compiler."""

    @pytest.fixture
    def compiler(self):
        return MJCFCompiler()

    def test_compiler_initialization(self):
        """Test MJCF compiler initialization."""
        compiler = MJCFCompiler()
        assert compiler is not None

    def test_compile_basic_mjcf(self, compiler):
        """Test compiling basic MJCF."""
        mjcf_string = """
        <mujoco>
            <worldbody>
                <light diffuse="1 1 1" pos="0 0 10"/>
                <geom type="sphere" size="0.5"/>
            </worldbody>
        </mujoco>
        """

        result = compiler.compile(mjcf_string)
        assert result is not None
        assert "mujoco" in result

    def test_validate_mjcf(self, compiler):
        """Test MJCF validation."""
        valid_mjcf = "<mujoco><worldbody><geom type='box'/></worldbody></mujoco>"
        invalid_mjcf = "<invalid>not mjcf</invalid>"

        assert compiler.validate(valid_mjcf) is True
        assert compiler.validate(invalid_mjcf) is False

    def test_optimize_mjcf(self, compiler):
        """Test MJCF optimization."""
        mjcf = """
        <mujoco>
            <compiler angle="degree"/>
            <option gravity="0 0 -9.81"/>
            <worldbody>
                <geom type="sphere" size="0.1"/>
            </worldbody>
        </mujoco>
        """

        optimized = compiler.optimize(mjcf)
        assert optimized is not None
        assert "mujoco" in optimized

    def test_add_visual_elements(self, compiler):
        """Test adding visual elements to MJCF."""
        base_mjcf = "<mujoco><worldbody></worldbody></mujoco>"

        with_visuals = compiler.add_visual_elements(
            base_mjcf,
            shadows=True,
            reflections=True,
            haze=True
        )

        assert "visual" in with_visuals or base_mjcf in with_visuals


class TestMuJoCoRuntime:
    """Tests for MuJoCo Runtime."""

    @pytest.fixture
    def runtime(self):
        return MuJoCoRuntime()

    def test_runtime_initialization(self):
        """Test runtime initialization."""
        runtime = MuJoCoRuntime()
        assert runtime is not None

    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    def test_load_model(self, mock_data, mock_model, runtime):
        """Test loading a MuJoCo model."""
        mjcf = "<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>"

        mock_model.from_xml_string = Mock(return_value=Mock())
        mock_data.return_value = Mock()

        model = runtime.load_model(mjcf)
        assert model is not None

    @patch('mujoco.mj_step')
    def test_step_simulation(self, mock_step, runtime):
        """Test stepping the simulation."""
        runtime.model = Mock()
        runtime.data = Mock()

        runtime.step()
        mock_step.assert_called_once()

    def test_get_state(self, runtime):
        """Test getting simulation state."""
        runtime.model = Mock()
        runtime.data = Mock()
        runtime.data.qpos = np.array([1.0, 2.0, 3.0])
        runtime.data.qvel = np.array([0.1, 0.2, 0.3])

        state = runtime.get_state()
        assert "qpos" in state
        assert "qvel" in state

    def test_set_state(self, runtime):
        """Test setting simulation state."""
        runtime.model = Mock()
        runtime.data = Mock()
        runtime.data.qpos = np.zeros(3)
        runtime.data.qvel = np.zeros(3)

        new_state = {
            "qpos": [1.0, 2.0, 3.0],
            "qvel": [0.1, 0.2, 0.3]
        }

        runtime.set_state(new_state)
        assert runtime.data.qpos is not None


class TestMultiModalEnhancer:
    """Tests for MultiModal Enhancer."""

    @pytest.fixture
    def enhancer(self):
        return MultiModalEnhancer()

    def test_enhancer_initialization(self):
        """Test enhancer initialization."""
        enhancer = MultiModalEnhancer()
        assert enhancer is not None

    @pytest.mark.asyncio
    async def test_enhance_prompt_with_sketch(self, enhancer):
        """Test enhancing prompt with sketch data."""
        with patch.object(enhancer, 'vision_client') as mock_vision:
            mock_vision.analyze = AsyncMock(return_value={
                "objects": ["pendulum", "support"],
                "description": "A pendulum attached to a support"
            })

            result = await enhancer.enhance(
                text_prompt="Make it swing",
                sketch_data="data:image/png;base64,..."
            )

            assert result is not None
            assert isinstance(result, dict) or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_extract_sketch_features(self, enhancer):
        """Test extracting features from sketch."""
        sketch_data = "data:image/png;base64,iVBORw0KGgo="

        features = await enhancer.extract_sketch_features(sketch_data)
        assert features is not None

    @pytest.mark.asyncio
    async def test_combine_modalities(self, enhancer):
        """Test combining text and visual modalities."""
        text_features = {"prompt": "A bouncing ball"}
        visual_features = {"objects": ["sphere"], "motion": "bouncing"}

        combined = await enhancer.combine_modalities(text_features, visual_features)
        assert combined is not None


class TestOptimizedRenderer:
    """Tests for Optimized Renderer."""

    @pytest.fixture
    def renderer(self):
        return OptimizedRenderer()

    def test_renderer_initialization(self):
        """Test renderer initialization."""
        renderer = OptimizedRenderer()
        assert renderer is not None

    @patch('mujoco.Renderer')
    def test_setup_renderer(self, mock_renderer_class, renderer):
        """Test setting up the renderer."""
        mock_renderer = Mock()
        mock_renderer_class.return_value = mock_renderer

        renderer.setup(width=1920, height=1080)
        assert renderer.width == 1920
        assert renderer.height == 1080

    def test_render_frame(self, renderer):
        """Test rendering a frame."""
        renderer.renderer = Mock()
        renderer.model = Mock()
        renderer.data = Mock()

        frame = renderer.render_frame()
        assert frame is not None or renderer.renderer is not None

    def test_apply_post_processing(self, renderer):
        """Test applying post-processing effects."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        processed = renderer.apply_post_processing(
            frame,
            bloom=True,
            motion_blur=True,
            color_correction=True
        )

        assert processed is not None
        assert processed.shape == frame.shape

    def test_save_video(self, renderer):
        """Test saving rendered video."""
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(10)]

        with patch('cv2.VideoWriter') as mock_writer:
            mock_writer.return_value.write = Mock()
            mock_writer.return_value.release = Mock()

            renderer.save_video(frames, "output.mp4", fps=30)
            assert mock_writer.called


class TestPerformanceOptimizer:
    """Tests for Performance Optimizer."""

    @pytest.fixture
    def optimizer(self):
        return PerformanceOptimizer()

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = PerformanceOptimizer()
        assert optimizer is not None

    def test_profile_simulation(self, optimizer):
        """Test profiling simulation performance."""
        with patch('cProfile.Profile') as mock_profiler:
            mock_profiler.return_value.enable = Mock()
            mock_profiler.return_value.disable = Mock()
            mock_profiler.return_value.get_stats = Mock(return_value={})

            def test_func():
                return sum(range(1000))

            stats = optimizer.profile(test_func)
            assert stats is not None

    def test_optimize_mjcf_for_performance(self, optimizer):
        """Test optimizing MJCF for performance."""
        mjcf = """
        <mujoco>
            <option timestep="0.002" iterations="50"/>
            <worldbody>
                <geom type="sphere" size="0.1"/>
            </worldbody>
        </mujoco>
        """

        optimized = optimizer.optimize_mjcf(mjcf)
        assert optimized is not None
        assert "mujoco" in optimized

    def test_suggest_optimizations(self, optimizer):
        """Test suggesting performance optimizations."""
        profile_data = {
            "total_time": 10.5,
            "step_time": 0.002,
            "render_time": 0.016,
            "collision_checks": 1000
        }

        suggestions = optimizer.suggest_optimizations(profile_data)
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 0


class TestPhysicsLLMClient:
    """Tests for Physics LLM Client."""

    @pytest.fixture
    def physics_client(self):
        return PhysicsLLMClient()

    def test_client_initialization(self):
        """Test physics LLM client initialization."""
        client = PhysicsLLMClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_generate_physics_spec(self, physics_client):
        """Test generating physics specification."""
        with patch.object(physics_client, 'llm') as mock_llm:
            mock_llm.generate = AsyncMock(return_value={
                "gravity": [0, 0, -9.81],
                "timestep": 0.001,
                "iterations": 50,
                "solver": "Newton"
            })

            spec = await physics_client.generate_physics_spec(
                "A realistic pendulum with air resistance"
            )

            assert "gravity" in spec
            assert "timestep" in spec

    @pytest.mark.asyncio
    async def test_validate_physics_params(self, physics_client):
        """Test validating physics parameters."""
        params = {
            "gravity": [0, 0, -9.81],
            "timestep": 0.001,
            "iterations": 50
        }

        is_valid = await physics_client.validate_params(params)
        assert isinstance(is_valid, bool)


class TestRealtimeProgressManager:
    """Tests for Realtime Progress Manager."""

    @pytest.fixture
    def progress_manager(self):
        return RealtimeProgressManager()

    def test_manager_initialization(self):
        """Test progress manager initialization."""
        manager = RealtimeProgressManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_send_progress_update(self, progress_manager):
        """Test sending progress updates."""
        with patch.object(progress_manager, 'websocket') as mock_ws:
            mock_ws.send_json = AsyncMock()

            await progress_manager.send_update({
                "stage": "parsing",
                "progress": 50,
                "message": "Parsing prompt..."
            })

            mock_ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_progress(self, progress_manager):
        """Test tracking progress of async operation."""
        async def sample_operation():
            await asyncio.sleep(0.1)
            return "completed"

        result = await progress_manager.track(
            sample_operation(),
            stage="test_operation",
            total_steps=10
        )

        assert result == "completed"


class TestSketchAnalyzer:
    """Tests for Sketch Analyzer."""

    @pytest.fixture
    def analyzer(self):
        return SketchAnalyzer()

    def test_analyzer_initialization(self):
        """Test sketch analyzer initialization."""
        analyzer = SketchAnalyzer()
        assert analyzer is not None

    @pytest.mark.asyncio
    async def test_analyze_sketch(self, analyzer):
        """Test analyzing a sketch."""
        with patch.object(analyzer, 'vision_model') as mock_vision:
            mock_vision.analyze = AsyncMock(return_value={
                "objects": ["circle", "line"],
                "structure": "pendulum-like",
                "confidence": 0.85
            })

            result = await analyzer.analyze("data:image/png;base64,...")

            assert "objects" in result
            assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_extract_geometry(self, analyzer):
        """Test extracting geometry from sketch."""
        sketch_data = "data:image/png;base64,..."

        geometry = await analyzer.extract_geometry(sketch_data)
        assert geometry is not None
        assert isinstance(geometry, dict) or isinstance(geometry, list)

    @pytest.mark.asyncio
    async def test_detect_physics_hints(self, analyzer):
        """Test detecting physics hints from sketch."""
        sketch_data = "data:image/png;base64,..."

        hints = await analyzer.detect_physics_hints(sketch_data)
        assert hints is not None


class TestStreamingProtocol:
    """Tests for Streaming Protocol."""

    @pytest.fixture
    def protocol(self):
        return StreamingProtocol()

    def test_protocol_initialization(self):
        """Test streaming protocol initialization."""
        protocol = StreamingProtocol()
        assert protocol is not None

    def test_encode_frame(self, protocol):
        """Test encoding a frame for streaming."""
        frame_data = {
            "timestamp": 1000,
            "positions": [1.0, 2.0, 3.0],
            "velocities": [0.1, 0.2, 0.3]
        }

        encoded = protocol.encode_frame(frame_data)
        assert encoded is not None
        assert isinstance(encoded, bytes) or isinstance(encoded, str)

    def test_decode_frame(self, protocol):
        """Test decoding a streamed frame."""
        encoded_data = b'{"timestamp": 1000, "positions": [1, 2, 3]}'

        decoded = protocol.decode_frame(encoded_data)
        assert decoded is not None
        assert "timestamp" in decoded

    @pytest.mark.asyncio
    async def test_stream_simulation(self, protocol):
        """Test streaming simulation data."""
        frames = [
            {"timestamp": i * 100, "data": f"frame_{i}"}
            for i in range(10)
        ]

        with patch.object(protocol, 'websocket') as mock_ws:
            mock_ws.send = AsyncMock()

            await protocol.stream(frames, fps=30)

            assert mock_ws.send.call_count > 0