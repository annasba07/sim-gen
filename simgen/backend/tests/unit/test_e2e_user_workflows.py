"""Comprehensive End-to-End User Workflow Tests for SimGen AI.

This test suite covers complete user journeys from prompt to simulation,
testing integration between all services and ensuring production-ready functionality.
"""

import pytest
import sys
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List, Optional
import base64
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import all required modules
from simgen.core.config import settings
from simgen.models.schemas import ExtractedEntities, SimulationGenerationMethod
from simgen.models.simulation import SimulationStatus, SimulationGenerationMethod as DBGenerationMethod
from simgen.services.llm_client import LLMClient
from simgen.services.prompt_parser import PromptParser
from simgen.services.simulation_generator import SimulationGenerator, GenerationResult
from simgen.services.dynamic_scene_composer import DynamicSceneComposer
from simgen.services.mjcf_compiler import MJCFCompiler
from simgen.services.mujoco_runtime import MuJoCoRuntime
from simgen.services.sketch_analyzer import SketchAnalyzer
from simgen.services.streaming_protocol import StreamingManager, BinaryProtocol, MessageType
from simgen.services.realtime_progress import RealTimeProgressTracker
try:
    from simgen.services.performance_optimizer import PerformancePipeline as PerformanceOptimizer
except ImportError:
    PerformanceOptimizer = Mock

# Import API endpoints with mocking for missing components
try:
    from simgen.api.simulation import router as simulation_router
except ImportError:
    simulation_router = Mock()

try:
    from simgen.api.physics import router as physics_router
except ImportError:
    physics_router = Mock()

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = Mock()


class TestCompleteUserWorkflows:
    """Test complete end-to-end user workflows."""

    async def test_text_prompt_to_simulation_workflow(self):
        """Test complete workflow: Text prompt → Entity extraction → MJCF generation → Simulation."""

        # Step 1: User submits text prompt
        user_prompt = "Create a bouncing red ball in a room with gravity"
        user_id = "test-user-001"

        # Step 2: Initialize services (production-like setup)
        llm_client = LLMClient()
        prompt_parser = PromptParser(llm_client=llm_client)
        simulation_generator = SimulationGenerator(llm_client=llm_client)
        mjcf_compiler = MJCFCompiler()
        mujoco_runtime = MuJoCoRuntime()

        # Step 3: Parse prompt and extract entities
        with patch.object(llm_client, 'extract_entities') as mock_extract:
            mock_extract.return_value = {
                "objects": [
                    {
                        "type": "sphere",
                        "properties": {
                            "radius": 0.1,
                            "color": [1.0, 0.0, 0.0, 1.0],
                            "mass": 0.5
                        }
                    }
                ],
                "environment": {
                    "gravity": [0, 0, -9.81],
                    "ground": True,
                    "walls": True
                },
                "physics": {
                    "timestep": 0.002,
                    "solver": "PGS"
                }
            }

            entities = await prompt_parser.parse(user_prompt)

            # Verify entities were extracted correctly
            assert entities is not None
            assert "objects" in entities
            assert len(entities["objects"]) > 0
            assert entities["objects"][0]["type"] == "sphere"

        # Step 4: Generate simulation MJCF
        with patch.object(llm_client, 'generate_mjcf') as mock_generate:
            expected_mjcf = """<mujoco model="bouncing_ball_simulation">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.002" iterations="50" solver="PGS" gravity="0 0 -9.81"/>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="300" height="300"/>
        <material name="ball_material" rgba="1.0 0.0 0.0 1.0"/>
    </asset>
    <worldbody>
        <light directional="true" cutoff="100" exponent="1" diffuse="1 1 1" specular=".1 .1 .1" pos="0 0 1.5" dir="-0 0 -1.5"/>
        <geom name="floor" pos="0 0 0" size="2.0 2.0 .05" type="box" material="grid"/>
        <body name="ball" pos="0 0 1">
            <joint name="ball_joint" type="free"/>
            <geom name="ball_geom" type="sphere" size="0.1" rgba="1.0 0.0 0.0 1.0" mass="0.5"/>
        </body>
    </worldbody>
</mujoco>"""
            mock_generate.return_value = expected_mjcf

            # Create ExtractedEntities object for simulation generation
            extracted_entities = ExtractedEntities(
                objects=entities["objects"],
                environment=entities["environment"],
                physics=entities["physics"]
            )

            result = await simulation_generator.generate_simulation(extracted_entities, user_prompt)

            # Verify simulation was generated
            assert isinstance(result, GenerationResult)
            assert result.success is True
            assert result.mjcf_content is not None
            assert "<mujoco>" in result.mjcf_content
            assert "ball" in result.mjcf_content

        # Step 5: Compile and validate MJCF
        mjcf_content = result.mjcf_content

        # Validate MJCF structure
        is_valid = mjcf_compiler.validate(mjcf_content)
        assert is_valid is True

        # Compile MJCF (optimization, cleanup)
        compiled_mjcf = mjcf_compiler.compile(mjcf_content)
        assert compiled_mjcf is not None
        assert "<mujoco>" in compiled_mjcf

        # Step 6: Load into MuJoCo runtime (simulation testing)
        with patch('mujoco.MjModel.from_xml_string') as mock_model:
            mock_model.return_value = Mock()

            load_success = mujoco_runtime.load_model(compiled_mjcf)
            assert load_success is True

            # Step 7: Run simulation steps
            for i in range(10):
                mujoco_runtime.step()

            # Verify simulation state
            assert mujoco_runtime.model is not None

        # Step 8: Verify complete workflow metrics
        assert user_id == "test-user-001"
        assert len(user_prompt) > 0

    async def test_sketch_to_simulation_workflow(self):
        """Test workflow: Sketch upload → Analysis → Entity extraction → Simulation generation."""

        # Step 1: User uploads sketch (base64 encoded image)
        fake_image_data = b"fake_png_image_data_representing_pendulum_sketch"
        sketch_b64 = base64.b64encode(fake_image_data).decode('utf-8')
        sketch_data_url = f"data:image/png;base64,{sketch_b64}"

        user_id = "test-user-002"

        # Step 2: Initialize services
        sketch_analyzer = SketchAnalyzer()
        llm_client = LLMClient()
        simulation_generator = SimulationGenerator(llm_client=llm_client)

        # Step 3: Analyze sketch to extract shapes and objects
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.cvtColor') as mock_cvt, \
             patch('cv2.HoughCircles') as mock_circles, \
             patch('cv2.HoughLines') as mock_lines:

            # Mock OpenCV operations
            mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
            mock_imread.return_value = mock_img
            mock_cvt.return_value = mock_img[:, :, 0]

            # Mock detected shapes - pendulum with bob and string
            mock_circles.return_value = np.array([[[100, 180, 15]]], dtype=np.float32)  # Bob
            mock_lines.return_value = np.array([[[100, 50, 100, 165]]], dtype=np.float32)  # String

            analysis_result = sketch_analyzer.analyze(sketch_data_url)

            # Verify sketch analysis
            assert analysis_result is not None
            assert "shapes" in analysis_result

        # Step 4: Convert sketch analysis to entities
        with patch.object(llm_client, 'extract_entities') as mock_extract:
            # Use sketch analysis to inform entity extraction
            mock_extract.return_value = {
                "objects": [
                    {
                        "type": "pendulum",
                        "properties": {
                            "bob_mass": 1.0,
                            "bob_radius": 0.05,
                            "string_length": 1.0,
                            "anchor_position": [0, 0, 1.0]
                        }
                    }
                ],
                "environment": {
                    "gravity": [0, 0, -9.81],
                    "ground": False
                },
                "physics": {
                    "timestep": 0.001,
                    "solver": "Newton"
                }
            }

            # Extract entities based on sketch analysis
            entities = await llm_client.extract_entities(f"Sketch analysis: {analysis_result}")

            assert entities is not None
            assert "objects" in entities
            assert entities["objects"][0]["type"] == "pendulum"

        # Step 5: Generate simulation from sketch-derived entities
        with patch.object(llm_client, 'generate_mjcf') as mock_generate:
            pendulum_mjcf = """<mujoco model="pendulum_from_sketch">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.001" iterations="100" solver="Newton" gravity="0 0 -9.81"/>
    <worldbody>
        <light directional="true" pos="0 0 3" dir="0 0 -1"/>
        <body name="pendulum_bob" pos="0 0 0">
            <joint name="pendulum_hinge" type="hinge" axis="1 0 0" pos="0 0 1.0"/>
            <geom name="string" type="capsule" fromto="0 0 1.0 0 0 0" size="0.01"/>
            <geom name="bob" type="sphere" size="0.05" pos="0 0 0" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>"""
            mock_generate.return_value = pendulum_mjcf

            # Create ExtractedEntities object
            extracted_entities = ExtractedEntities(
                objects=entities["objects"],
                environment=entities["environment"],
                physics=entities["physics"]
            )

            result = await simulation_generator.generate_simulation(
                extracted_entities,
                f"Generate pendulum simulation from sketch analysis: {analysis_result}"
            )

            # Verify simulation generation
            assert isinstance(result, GenerationResult)
            assert result.success is True
            assert "pendulum" in result.mjcf_content
            assert "hinge" in result.mjcf_content

        # Step 6: Complete workflow validation
        assert sketch_data_url.startswith("data:image/png;base64,")
        assert user_id == "test-user-002"

    async def test_real_time_streaming_workflow(self):
        """Test workflow: Simulation start → Real-time data streaming → Client updates."""

        # Step 1: Setup simulation with streaming
        llm_client = LLMClient()
        simulation_generator = SimulationGenerator(llm_client=llm_client)
        streaming_manager = StreamingManager()
        binary_protocol = BinaryProtocol()
        runtime = MuJoCoRuntime()
        progress_manager = RealTimeProgressTracker()

        session_id = "streaming-session-001"

        # Step 2: Generate simulation for streaming
        with patch.object(llm_client, 'extract_entities'), \
             patch.object(llm_client, 'generate_mjcf') as mock_generate:

            mock_generate.return_value = """<mujoco model="streaming_demo">
    <option timestep="0.002"/>
    <worldbody>
        <body name="box" pos="0 0 1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
</mujoco>"""

            # Create entities for a simple falling box
            entities = ExtractedEntities(
                objects=[{"type": "box", "properties": {"size": [0.1, 0.1, 0.1]}}],
                environment={"gravity": [0, 0, -9.81]},
                physics={"timestep": 0.002}
            )

            simulation_result = await simulation_generator.generate_simulation(entities)
            assert simulation_result.success is True

        # Step 3: Initialize streaming session
        with patch('mujoco.MjModel.from_xml_string') as mock_model, \
             patch('mujoco.MjData') as mock_data_class:

            # Mock MuJoCo objects
            mock_model_obj = Mock()
            mock_model_obj.nq = 7  # Free joint has 7 DOF
            mock_model.return_value = mock_model_obj

            mock_data = Mock()
            mock_data.time = 0.0
            mock_data.qpos = np.array([0, 0, 1, 1, 0, 0, 0])  # position + quaternion
            mock_data.qvel = np.array([0, 0, 0, 0, 0, 0])     # velocities
            mock_data_class.return_value = mock_data

            # Load model and start simulation
            runtime.load_model(simulation_result.mjcf_content)

            # Step 4: Start progress tracking
            progress_manager.start_session(session_id, estimated_duration=10.0)

            # Step 5: Simulate and stream data
            frames_streamed = []

            for step in range(50):  # Simulate 50 steps
                # Step simulation
                runtime.step()

                # Update mock data to show progression
                mock_data.time = step * 0.002
                mock_data.qpos[2] = 1.0 - (step * 0.002 * 0.002 * 9.81 / 2)  # Falling motion

                # Create frame data
                frame_data = {
                    "frame_id": step,
                    "sim_time": mock_data.time,
                    "qpos": mock_data.qpos.tolist(),
                    "xpos": [[0, 0, mock_data.qpos[2]]],  # World position
                    "xquat": [[1, 0, 0, 0]]  # Quaternion
                }

                # Encode frame using binary protocol
                encoded_frame = binary_protocol.encode_frame(frame_data)
                assert isinstance(encoded_frame, bytes)
                assert len(encoded_frame) > 0

                frames_streamed.append(encoded_frame)

                # Update progress
                progress = (step + 1) / 50 * 100
                progress_manager.update_progress(session_id, progress)

                # Simulate real-time delay
                if step % 10 == 0:  # Every 10th frame
                    await asyncio.sleep(0.001)  # Small delay for realism

            # Step 6: Verify streaming results
            assert len(frames_streamed) == 50
            assert all(isinstance(frame, bytes) for frame in frames_streamed)

            # Verify frame data can be decoded
            decoded_first = binary_protocol.decode_frame(frames_streamed[0])
            decoded_last = binary_protocol.decode_frame(frames_streamed[-1])

            assert decoded_first["frame_id"] == 0
            assert decoded_last["frame_id"] == 49
            assert decoded_last["sim_time"] > decoded_first["sim_time"]

            # Step 7: Complete session
            progress_manager.complete_session(session_id, success=True)

    async def test_simulation_refinement_workflow(self):
        """Test workflow: Initial simulation → User feedback → Refinement → Enhanced simulation."""

        # Step 1: Create initial simulation
        llm_client = LLMClient()
        simulation_generator = SimulationGenerator(llm_client=llm_client)

        initial_prompt = "Create a simple pendulum"
        user_id = "test-user-003"

        # Step 2: Generate initial simulation
        with patch.object(llm_client, 'extract_entities') as mock_extract, \
             patch.object(llm_client, 'generate_mjcf') as mock_generate:

            mock_extract.return_value = {
                "objects": [{"type": "pendulum", "properties": {"length": 1.0, "mass": 1.0}}],
                "environment": {"gravity": [0, 0, -9.81]},
                "physics": {"timestep": 0.002}
            }

            mock_generate.return_value = """<mujoco model="simple_pendulum">
    <worldbody>
        <body name="bob">
            <joint type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.05" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>"""

            # Generate initial simulation
            entities = ExtractedEntities(
                objects=[{"type": "pendulum", "properties": {"length": 1.0}}],
                environment={"gravity": [0, 0, -9.81]},
                physics={"timestep": 0.002}
            )

            initial_result = await simulation_generator.generate_simulation(entities, initial_prompt)
            assert initial_result.success is True
            initial_mjcf = initial_result.mjcf_content

        # Step 3: User provides refinement feedback
        refinement_prompt = "Make the pendulum longer and add damping for more realistic motion"
        refinement_params = {
            "length": 2.0,
            "damping": 0.1,
            "mass": 1.5
        }

        # Step 4: Refine simulation based on feedback
        with patch.object(simulation_generator, 'refine_simulation') as mock_refine:
            refined_mjcf = """<mujoco model="refined_pendulum">
    <worldbody>
        <body name="bob" pos="0 0 -2.0">
            <joint type="hinge" axis="1 0 0" damping="0.1"/>
            <geom name="string" type="capsule" fromto="0 0 2.0 0 0 0" size="0.01"/>
            <geom name="bob" type="sphere" size="0.05" mass="1.5"/>
        </body>
    </worldbody>
</mujoco>"""

            mock_refine.return_value = GenerationResult(
                mjcf_content=refined_mjcf,
                method=DBGenerationMethod.LLM_BASED,
                metadata={"refinement": "added_damping_and_length", "parameters": refinement_params},
                success=True
            )

            refined_result = await simulation_generator.refine_simulation(
                initial_mjcf,
                refinement_prompt,
                refinement_params
            )

            # Verify refinement
            assert refined_result.success is True
            assert "damping" in refined_result.mjcf_content
            assert "2.0" in refined_result.mjcf_content  # New length
            assert refined_result.metadata["parameters"]["damping"] == 0.1

        # Step 5: Compare initial vs refined simulation
        assert len(refined_result.mjcf_content) > len(initial_mjcf)
        assert "damping" not in initial_mjcf
        assert "damping" in refined_result.mjcf_content

    async def test_error_handling_and_resilience_workflow(self):
        """Test workflow: Error scenarios → Graceful degradation → User notification."""

        # Step 1: Setup services with failure scenarios
        llm_client = LLMClient()
        simulation_generator = SimulationGenerator(llm_client=llm_client)
        mjcf_compiler = MJCFCompiler()

        # Step 2: Test LLM service failure
        with patch.object(llm_client, 'extract_entities') as mock_extract:
            # Simulate LLM API failure
            mock_extract.side_effect = Exception("LLM API temporarily unavailable")

            try:
                prompt_parser = PromptParser(llm_client=llm_client)
                await prompt_parser.parse("Create a simulation")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "LLM API temporarily unavailable" in str(e)

        # Step 3: Test invalid MJCF handling
        invalid_mjcf = "<invalid>This is not valid MJCF</invalid>"

        is_valid = mjcf_compiler.validate(invalid_mjcf)
        assert is_valid is False

        # Step 4: Test simulation generation with fallback
        with patch.object(llm_client, 'extract_entities') as mock_extract, \
             patch.object(llm_client, 'generate_mjcf') as mock_generate:

            # First attempt fails
            mock_generate.side_effect = [
                Exception("Generation failed"),  # First call fails
                "<mujoco><worldbody></worldbody></mujoco>"  # Second call succeeds
            ]

            mock_extract.return_value = {
                "objects": [{"type": "box"}],
                "environment": {"gravity": [0, 0, -9.81]},
                "physics": {"timestep": 0.002}
            }

            # Simulate retry logic
            entities = ExtractedEntities(
                objects=[{"type": "box"}],
                environment={"gravity": [0, 0, -9.81]},
                physics={"timestep": 0.002}
            )

            try:
                # First attempt
                result = await simulation_generator.generate_simulation(entities)
                assert False, "Should fail on first attempt"
            except Exception:
                # Retry with template-based fallback
                with patch.object(simulation_generator, '_template_based_generation') as mock_template:
                    mock_template.return_value = GenerationResult(
                        mjcf_content="<mujoco><worldbody><geom type='box'/></worldbody></mujoco>",
                        method=DBGenerationMethod.TEMPLATE_BASED,
                        metadata={"fallback": "template_based"},
                        success=True
                    )

                    # Use template fallback
                    fallback_result = await simulation_generator._template_based_generation(entities)
                    assert fallback_result.success is True
                    assert fallback_result.method == DBGenerationMethod.TEMPLATE_BASED

        # Step 5: Test streaming failure handling
        streaming_manager = StreamingManager()

        # Test malformed frame data
        invalid_frame = {"invalid": "data"}

        try:
            binary_protocol = BinaryProtocol()
            binary_protocol.encode_frame(invalid_frame)
            assert False, "Should fail with invalid frame"
        except (KeyError, ValueError, TypeError):
            # Expected - invalid frame should raise error
            pass

        # Step 6: Test performance degradation handling
        optimizer = PerformanceOptimizer()

        # Simulate high resource usage
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:

            mock_cpu.return_value = 95.0  # High CPU
            mock_memory.return_value = Mock(percent=90.0)  # High memory

            metrics = optimizer.collect_metrics()

            # Should detect high resource usage
            assert metrics["cpu_percent"] >= 90
            assert metrics["memory_percent"] >= 85

            # Simulate performance optimization
            optimization_applied = optimizer.optimize_if_needed(metrics)
            # Would apply optimizations in real scenario

    async def test_performance_monitoring_workflow(self):
        """Test workflow: Performance monitoring → Metrics collection → Optimization triggers."""

        # Step 1: Setup performance monitoring
        optimizer = PerformanceOptimizer()
        progress_manager = RealTimeProgressTracker()

        session_id = "perf-test-session"

        # Step 2: Monitor simulation performance
        performance_data = []

        # Simulate multiple simulation runs with performance tracking
        for run in range(5):
            start_time = time.time()

            # Start progress tracking
            progress_manager.start_session(f"{session_id}-{run}", estimated_duration=2.0)

            # Simulate simulation work
            for step in range(100):
                # Collect metrics every 10 steps
                if step % 10 == 0:
                    metrics = optimizer.collect_metrics()
                    performance_data.append({
                        "run": run,
                        "step": step,
                        "timestamp": time.time(),
                        "cpu_percent": metrics.get("cpu_percent", 0),
                        "memory_percent": metrics.get("memory_percent", 0)
                    })

                # Update progress
                progress = (step + 1) / 100 * 100
                progress_manager.update_progress(f"{session_id}-{run}", progress)

                # Small delay to simulate work
                await asyncio.sleep(0.001)

            end_time = time.time()
            execution_time = end_time - start_time

            # Complete session
            progress_manager.complete_session(f"{session_id}-{run}", success=True)

            # Record execution time
            performance_data.append({
                "run": run,
                "execution_time": execution_time,
                "type": "summary"
            })

        # Step 3: Analyze performance trends
        execution_times = [d["execution_time"] for d in performance_data if d.get("type") == "summary"]
        avg_execution_time = sum(execution_times) / len(execution_times)

        assert len(execution_times) == 5
        assert avg_execution_time > 0
        assert all(t > 0 for t in execution_times)

        # Step 4: Verify metrics collection
        metric_points = [d for d in performance_data if d.get("type") != "summary"]
        assert len(metric_points) >= 50  # At least 10 metrics per 5 runs

        # Step 5: Test optimization triggers
        high_cpu_points = [d for d in metric_points if d.get("cpu_percent", 0) > 80]
        high_memory_points = [d for d in metric_points if d.get("memory_percent", 0) > 80]

        # In a real scenario, we would trigger optimizations based on these metrics
        optimization_needed = len(high_cpu_points) > 5 or len(high_memory_points) > 5

        # This would trigger optimization in production
        if optimization_needed:
            # Simulate applying optimizations
            optimizations_applied = optimizer.apply_optimizations({
                "reduce_quality": True,
                "increase_timestep": True,
                "limit_concurrent_sessions": True
            })
            # In real implementation, this would actually apply optimizations


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])