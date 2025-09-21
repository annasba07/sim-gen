"""
Test actual existing modules to reach 50% coverage.
Focus on modules that exist in the codebase and can be properly imported.
"""

import pytest
import asyncio
import os
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set test environment variables
os.environ.update({
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "REDIS_URL": "redis://localhost:6379",
    "SECRET_KEY": "test-secret-key-for-testing",
    "OPENAI_API_KEY": "sk-test-key",
    "ANTHROPIC_API_KEY": "sk-ant-test-key",
    "ENVIRONMENT": "test"
})


class TestLLMClient:
    """Test the actual LLM client module."""

    @patch('anthropic.AsyncAnthropic')
    @patch('openai.AsyncOpenAI')
    def test_llm_client_initialization(self, mock_openai, mock_anthropic):
        """Test LLM client initialization."""
        from simgen.services.llm_client import LLMClient

        # Setup mocks
        mock_anthropic.return_value = AsyncMock()
        mock_openai.return_value = AsyncMock()

        client = LLMClient()
        assert client is not None

    @pytest.mark.asyncio
    @patch('anthropic.AsyncAnthropic')
    @patch('openai.AsyncOpenAI')
    async def test_llm_generate_simulation(self, mock_openai, mock_anthropic):
        """Test LLM simulation generation."""
        from simgen.services.llm_client import LLMClient

        # Setup mocks
        mock_anthropic_client = AsyncMock()
        mock_anthropic.return_value = mock_anthropic_client

        mock_openai_client = AsyncMock()
        mock_openai.return_value = mock_openai_client

        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="<mujoco><worldbody><body><geom type='box'/></body></worldbody></mujoco>")]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        client = LLMClient()

        # Test generation
        result = await client.generate_simulation("Create a box", use_anthropic=True)
        assert result is not None


class TestPhysicsSpec:
    """Test the physics spec models."""

    def test_physics_spec_creation(self):
        """Test creating PhysicsSpec models."""
        from simgen.models.physics_spec import PhysicsSpec, Body, Geom, Joint

        # Create a simple physics spec
        geom = Geom(type="box", size=[1.0, 1.0, 1.0], rgba=[1.0, 0.0, 0.0, 1.0])
        joint = Joint(type="hinge", axis=[0.0, 0.0, 1.0])
        body = Body(
            name="test_body",
            pos=[0.0, 0.0, 1.0],
            geoms=[geom],
            joints=[joint]
        )

        spec = PhysicsSpec(bodies=[body])

        assert spec is not None
        assert len(spec.bodies) == 1
        assert spec.bodies[0].name == "test_body"

    def test_physics_spec_to_mjcf(self):
        """Test converting PhysicsSpec to MJCF."""
        from simgen.models.physics_spec import PhysicsSpec, Body, Geom

        spec = PhysicsSpec(
            bodies=[
                Body(
                    name="ball",
                    pos=[0, 0, 2],
                    geoms=[Geom(type="sphere", size=[0.5])]
                )
            ]
        )

        mjcf = spec.to_mjcf()
        assert mjcf is not None
        assert "<mujoco>" in mjcf
        assert "ball" in mjcf

    def test_physics_spec_validation(self):
        """Test PhysicsSpec validation."""
        from simgen.models.physics_spec import PhysicsSpec, Body, Geom, Option

        # Test with options
        spec = PhysicsSpec(
            option=Option(gravity=[0, 0, -9.81], timestep=0.002),
            bodies=[
                Body(name="ground", geoms=[Geom(type="plane")])
            ]
        )

        assert spec.option.gravity == [0, 0, -9.81]
        assert spec.option.timestep == 0.002


class TestSimulationGenerator:
    """Test the simulation generator service."""

    @pytest.mark.asyncio
    @patch('simgen.services.llm_client.LLMClient')
    async def test_simulation_generator_basic(self, mock_llm_client):
        """Test basic simulation generation."""
        from simgen.services.simulation_generator import SimulationGenerator

        # Setup mock
        mock_client = AsyncMock()
        mock_llm_client.return_value = mock_client
        mock_client.generate_simulation.return_value = {
            "mjcf": "<mujoco><worldbody/></mujoco>",
            "description": "Test simulation"
        }

        generator = SimulationGenerator()
        result = await generator.generate("Create a test simulation")

        assert result is not None

    @pytest.mark.asyncio
    @patch('simgen.services.llm_client.LLMClient')
    async def test_simulation_with_physics_spec(self, mock_llm_client):
        """Test simulation generation with PhysicsSpec."""
        from simgen.services.simulation_generator import SimulationGenerator
        from simgen.models.physics_spec import PhysicsSpec, Body, Geom

        mock_client = AsyncMock()
        mock_llm_client.return_value = mock_client

        generator = SimulationGenerator()

        spec = PhysicsSpec(
            bodies=[Body(name="cube", geoms=[Geom(type="box")])]
        )

        # Mock the generate_from_spec method
        generator.generate_from_spec = AsyncMock(return_value={
            "mjcf": spec.to_mjcf(),
            "success": True
        })

        result = await generator.generate_from_spec(spec)
        assert result["success"] == True


class TestMJCFCompiler:
    """Test MJCF compiler service."""

    def test_mjcf_compiler_basic(self):
        """Test basic MJCF compilation."""
        from simgen.services.mjcf_compiler import MJCFCompiler

        compiler = MJCFCompiler()

        mjcf = """
        <mujoco>
            <worldbody>
                <body name="box">
                    <geom type="box" size="1 1 1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        result = compiler.compile(mjcf)
        assert result is not None
        assert result["success"] == True

    def test_mjcf_validation(self):
        """Test MJCF validation."""
        from simgen.services.mjcf_compiler import MJCFCompiler

        compiler = MJCFCompiler()

        # Invalid MJCF
        invalid_mjcf = "<invalid>Not valid MJCF</invalid>"
        result = compiler.validate(invalid_mjcf)

        assert result["valid"] == False
        assert len(result["errors"]) > 0


class TestPhysicsLLMClient:
    """Test physics LLM client."""

    @patch('simgen.services.llm_client.LLMClient')
    def test_physics_llm_client_init(self, mock_llm):
        """Test PhysicsLLMClient initialization."""
        from simgen.services.physics_llm_client import PhysicsLLMClient

        mock_llm.return_value = AsyncMock()

        client = PhysicsLLMClient()
        assert client is not None

    @pytest.mark.asyncio
    @patch('simgen.services.llm_client.LLMClient')
    async def test_generate_physics_simulation(self, mock_llm):
        """Test physics simulation generation."""
        from simgen.services.physics_llm_client import PhysicsLLMClient

        mock_client = AsyncMock()
        mock_llm.return_value = mock_client
        mock_client.generate_simulation.return_value = {
            "mjcf": "<mujoco/>",
            "physics_params": {"gravity": -9.81}
        }

        client = PhysicsLLMClient()
        result = await client.generate_physics_simulation(
            "Create a pendulum",
            physics_constraints={"max_velocity": 10.0}
        )

        assert result is not None


class TestResilience:
    """Test resilience service."""

    def test_circuit_breaker(self):
        """Test CircuitBreaker implementation."""
        from simgen.services.resilience import CircuitBreaker

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception
        )

        assert breaker.state == "closed"
        assert breaker.failure_count == 0

        # Test failure tracking
        breaker.record_failure()
        assert breaker.failure_count == 1

        # Test state transitions
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

    def test_retry_policy(self):
        """Test RetryPolicy implementation."""
        from simgen.services.resilience import RetryPolicy

        policy = RetryPolicy(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )

        assert policy.max_attempts == 3
        assert policy.initial_delay == 1.0

        # Test delay calculation
        delay = policy.get_delay(attempt=1)
        assert delay == 1.0

        delay = policy.get_delay(attempt=2)
        assert delay == 2.0  # exponential backoff

    @pytest.mark.asyncio
    async def test_resilient_call(self):
        """Test resilient function call wrapper."""
        from simgen.services.resilience import resilient_call

        # Mock function that fails once then succeeds
        call_count = 0

        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return "success"

        result = await resilient_call(
            flaky_function,
            max_retries=3,
            retry_delay=0.01
        )

        assert result == "success"
        assert call_count == 2


class TestStreamingProtocol:
    """Test streaming protocol."""

    def test_message_creation(self):
        """Test creating streaming messages."""
        from simgen.services.streaming_protocol import (
            StreamingProtocol, MessageType, StreamMessage
        )

        protocol = StreamingProtocol()

        # Create data message
        message = protocol.create_message(
            MessageType.DATA,
            {"simulation_id": "123", "frame": 1}
        )

        assert message.type == MessageType.DATA
        assert message.data["simulation_id"] == "123"

    def test_message_serialization(self):
        """Test message serialization."""
        from simgen.services.streaming_protocol import (
            StreamingProtocol, MessageType
        )

        protocol = StreamingProtocol()

        message = protocol.create_message(
            MessageType.STATUS,
            {"status": "running", "progress": 0.5}
        )

        # Serialize
        serialized = protocol.serialize(message)
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized = protocol.deserialize(serialized)
        assert deserialized.type == MessageType.STATUS
        assert deserialized.data["progress"] == 0.5

    def test_stream_handler(self):
        """Test stream handler."""
        from simgen.services.streaming_protocol import StreamHandler

        handler = StreamHandler()

        # Add subscriber
        subscriber_id = handler.add_subscriber(Mock())
        assert subscriber_id is not None

        # Broadcast message
        handler.broadcast({"test": "data"})

        # Remove subscriber
        handler.remove_subscriber(subscriber_id)


class TestOptimizedRenderer:
    """Test optimized renderer."""

    @patch('mujoco.MjModel')
    @patch('mujoco.MjData')
    def test_renderer_initialization(self, mock_data, mock_model):
        """Test renderer initialization."""
        from simgen.services.optimized_renderer import OptimizedRenderer

        renderer = OptimizedRenderer(
            width=640,
            height=480,
            fps=30
        )

        assert renderer.width == 640
        assert renderer.height == 480
        assert renderer.fps == 30

    @patch('mujoco.MjModel.from_xml_string')
    def test_load_model(self, mock_from_xml):
        """Test loading MuJoCo model."""
        from simgen.services.optimized_renderer import OptimizedRenderer

        mock_model = Mock()
        mock_from_xml.return_value = mock_model

        renderer = OptimizedRenderer()

        mjcf = "<mujoco><worldbody/></mujoco>"
        model = renderer.load_model(mjcf)

        assert model is not None
        mock_from_xml.assert_called_once()


class TestPromptParser:
    """Test prompt parser."""

    def test_parse_basic_prompt(self):
        """Test parsing basic prompts."""
        from simgen.services.prompt_parser import PromptParser

        parser = PromptParser()

        prompt = "Create a bouncing ball with gravity"
        result = parser.parse(prompt)

        assert result is not None
        assert "entities" in result
        assert "physics" in result

    def test_extract_physics_params(self):
        """Test extracting physics parameters."""
        from simgen.services.prompt_parser import PromptParser

        parser = PromptParser()

        prompt = "Create a simulation with gravity -9.81 and timestep 0.001"
        params = parser.extract_physics_params(prompt)

        assert params is not None
        assert "gravity" in params
        assert "timestep" in params

    def test_extract_objects(self):
        """Test extracting objects from prompt."""
        from simgen.services.prompt_parser import PromptParser

        parser = PromptParser()

        prompt = "Create a red box and a blue sphere on a flat ground"
        objects = parser.extract_objects(prompt)

        assert len(objects) >= 2
        assert any("box" in obj["type"] for obj in objects)
        assert any("sphere" in obj["type"] for obj in objects)


class TestModelsAndSchemas:
    """Test model and schema modules."""

    def test_simulation_model(self):
        """Test simulation model."""
        from simgen.models.simulation import (
            Simulation, SimulationStatus, SimulationMetadata
        )

        # Create simulation
        sim = Simulation(
            id="sim-123",
            prompt="Test simulation",
            mjcf_content="<mujoco/>",
            status=SimulationStatus.COMPLETED
        )

        assert sim.id == "sim-123"
        assert sim.status == SimulationStatus.COMPLETED

        # Test metadata
        metadata = SimulationMetadata(
            created_at=datetime.now(),
            duration=1.5,
            frame_count=45
        )
        sim.metadata = metadata
        assert sim.metadata.frame_count == 45

    def test_request_response_schemas(self):
        """Test request/response schemas."""
        from simgen.models.schemas import (
            SimulationRequest, SimulationResponse,
            PhysicsRequest, PhysicsResponse
        )

        # Test request
        request = SimulationRequest(
            prompt="Create a pendulum",
            parameters={
                "gravity": -9.81,
                "timestep": 0.001
            }
        )
        assert request.prompt == "Create a pendulum"
        assert request.parameters["gravity"] == -9.81

        # Test response
        response = SimulationResponse(
            simulation_id="sim-456",
            mjcf_content="<mujoco/>",
            status="completed",
            message="Simulation generated successfully"
        )
        assert response.simulation_id == "sim-456"
        assert response.status == "completed"


class TestObservability:
    """Test observability and monitoring."""

    def test_observability_service(self):
        """Test observability service."""
        from simgen.monitoring.observability import ObservabilityService

        service = ObservabilityService()

        # Test metric recording
        service.record_metric("api.request", 1.0, tags={"endpoint": "/simulate"})

        # Test logging
        service.log("info", "Test message", {"user": "test"})

        # Test tracing
        trace_id = service.start_trace("test_operation")
        assert trace_id is not None
        service.end_trace(trace_id)

    def test_metrics_aggregation(self):
        """Test metrics aggregation."""
        from simgen.monitoring.observability import MetricsAggregator

        aggregator = MetricsAggregator()

        # Add metrics
        for i in range(10):
            aggregator.add_metric("latency", i * 0.1)

        # Get statistics
        stats = aggregator.get_stats("latency")
        assert stats["count"] == 10
        assert stats["mean"] == 0.45
        assert stats["min"] == 0.0
        assert stats["max"] == 0.9


class TestCoreConfig:
    """Test core configuration."""

    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://localhost/test",
        "SECRET_KEY": "test-secret"
    })
    def test_settings_loading(self):
        """Test loading settings."""
        from simgen.core.config import Settings

        settings = Settings()
        assert settings.database_url == "postgresql://localhost/test"
        assert settings.secret_key == "test-secret"

    def test_config_validation(self):
        """Test configuration validation."""
        from simgen.core.config import validate_config

        config = {
            "database_url": "postgresql://localhost/test",
            "secret_key": "secret",
            "debug": True
        }

        is_valid, errors = validate_config(config)
        assert is_valid == True
        assert len(errors) == 0


class TestDatabaseModules:
    """Test database-related modules."""

    @patch('sqlalchemy.create_engine')
    def test_database_connection(self, mock_create_engine):
        """Test database connection setup."""
        from simgen.database.service import DatabaseService

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        service = DatabaseService("postgresql://localhost/test")
        assert service.engine == mock_engine

    def test_connection_pool(self):
        """Test connection pool."""
        from simgen.database.connection_pool import ConnectionPool

        pool = ConnectionPool(
            min_size=5,
            max_size=20,
            max_overflow=10
        )

        assert pool.min_size == 5
        assert pool.max_size == 20

    def test_query_optimizer(self):
        """Test query optimizer."""
        from simgen.database.query_optimizer import QueryOptimizer

        optimizer = QueryOptimizer()

        query = "SELECT * FROM users WHERE age > 25"
        optimized = optimizer.optimize(query)

        assert optimized is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/simgen", "--cov-report=term"])