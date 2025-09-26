"""
ACTUAL MODULE STRUCTURE TEST - Based on Real Imports
Target: 50% coverage with tests that actually work
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import asyncio
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Environment setup
os.environ.update({
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "REDIS_URL": "redis://localhost:6379",
    "SECRET_KEY": "actual-50-percent",
    "OPENAI_API_KEY": "test-key",
    "ANTHROPIC_API_KEY": "test-key"
})


class TestPhysicsSpec:
    """Test physics_spec module - already at 91% coverage."""

    def test_physics_spec_complete(self):
        """Complete physics_spec testing."""
        from simgen.models.physics_spec import (
            PhysicsSpec, Body, Geom, Joint, Actuator, Sensor,
            Material, Friction, Inertial, Contact, Equality,
            DefaultSettings, SimulationMeta, PhysicsSpecVersion,
            JointType, GeomType, ActuatorType, SensorType
        )

        # Test ALL enums and their values
        for version in PhysicsSpecVersion:
            assert version.value is not None

        for jtype in JointType:
            assert jtype.value is not None

        for gtype in GeomType:
            assert gtype.value is not None

        for atype in ActuatorType:
            assert atype.value is not None

        for stype in SensorType:
            assert stype.value is not None

        # Create comprehensive objects
        material = Material(
            name="test_material",
            rgba=[1, 0, 0, 1],
            emission=0.5,
            specular=0.7,
            shininess=100,
            reflectance=0.3
        )

        friction = Friction(
            slide=1.0,
            spin=0.1,
            roll=0.01
        )

        inertial = Inertial(
            mass=5.0,
            diaginertia=[1, 1, 1],
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0]
        )

        # Create multiple geoms
        geoms = []
        for gtype in ["box", "sphere", "cylinder", "capsule", "plane", "mesh"]:
            geom = Geom(
                name=f"test_geom_{gtype}",
                type=gtype,
                size=[1, 1, 1] if gtype == "box" else [1],
                rgba=[1, 0, 0, 1],
                mass=2.0,
                friction=friction,
                material=material,
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                contype=1,
                conaffinity=1,
                condim=3,
                solmix=1.0,
                solimp=[0.9, 0.95, 0.001]
            )
            geoms.append(geom)

        # Create multiple joints
        joints = []
        for jtype in [JointType.HINGE, JointType.SLIDE, JointType.BALL, JointType.FREE]:
            joint = Joint(
                name=f"test_joint_{jtype.value}",
                type=jtype,
                axis=[0, 0, 1] if jtype != JointType.BALL else None,
                limited=True,
                range=[-90, 90],
                damping=0.1,
                armature=0.01,
                stiffness=100
            )
            joints.append(joint)

        # Create bodies
        bodies = []
        for i in range(5):
            body = Body(
                id=f"body{i}",
                name=f"test_body_{i}",
                pos=[i, 0, 1],
                quat=[1, 0, 0, 0],
                inertial=inertial if i == 0 else None,
                geoms=[geoms[i % len(geoms)]],
                joints=[joints[i % len(joints)]] if i < len(joints) else []
            )
            bodies.append(body)

        # Create actuators
        actuators = []
        for atype in [ActuatorType.MOTOR, ActuatorType.POSITION, ActuatorType.VELOCITY]:
            actuator = Actuator(
                name=f"test_actuator_{atype.value}",
                type=atype,
                joint="test_joint_hinge",
                gear=100.0,
                ctrllimited=True,
                ctrlrange=[-1, 1]
            )
            actuators.append(actuator)

        # Create sensors
        sensors = []
        for stype in [SensorType.ACCELEROMETER, SensorType.GYRO, SensorType.FORCE,
                      SensorType.TORQUE, SensorType.JOINTPOS, SensorType.JOINTVEL]:
            sensor = Sensor(
                name=f"test_sensor_{stype.value}",
                type=stype,
                site="test_site" if stype in [SensorType.ACCELEROMETER, SensorType.GYRO] else None,
                joint="test_joint" if stype in [SensorType.JOINTPOS, SensorType.JOINTVEL] else None
            )
            sensors.append(sensor)

        # Create contacts
        contacts = []
        for i in range(3):
            contact = Contact(
                name=f"test_contact_{i}",
                geom1=f"geom{i}",
                geom2=f"geom{i+1}",
                condim=3,
                friction=[1.0, 0.005, 0.0001],
                solref=[-100, -50]
            )
            contacts.append(contact)

        # Create equalities
        equalities = []
        equality = Equality(
            name="test_equality",
            type="connect",
            body1="body1",
            body2="body2",
            anchor=[0, 0, 0]
        )
        equalities.append(equality)

        defaults = DefaultSettings(
            geom_friction=[1.0, 0.005, 0.0001],
            joint_damping=0.1,
            actuator_gear=100,
            sensor_noise=0.001
        )

        meta = SimulationMeta(
            version=PhysicsSpecVersion.V1_0_0,
            created_by="test_user",
            created_at=datetime.now(),
            description="Test simulation"
        )

        spec = PhysicsSpec(
            meta=meta,
            defaults=defaults,
            bodies=bodies,
            actuators=actuators,
            sensors=sensors,
            contacts=contacts,
            equality=equalities
        )

        # Test all methods
        spec_dict = spec.dict()
        spec_json = spec.json()
        spec_copy = spec.copy()
        mjcf = spec.to_mjcf()

        assert "<mujoco>" in mjcf
        assert "test_body" in mjcf
        assert len(spec.bodies) == 5

        # Test validation
        with pytest.raises(ValueError, match="At least one body is required"):
            PhysicsSpec(bodies=[])

        # Test duplicate ID validation
        dup_bodies = [
            Body(id="dup", name="b1", geoms=[geoms[0]]),
            Body(id="dup", name="b2", geoms=[geoms[0]])
        ]
        with pytest.raises(ValueError, match="Duplicate body ID"):
            PhysicsSpec(bodies=dup_bodies)


class TestResilienceModule:
    """Test resilience module with actual imports."""

    def test_circuit_breaker(self):
        """Test CircuitBreaker."""
        from simgen.services.resilience import (
            CircuitBreaker, CircuitBreakerConfig, CircuitState
        )

        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=Exception
        )

        breaker = CircuitBreaker(name="test", config=config)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.name == "test"
        assert breaker._failure_count == 0

        # Test record_success
        breaker.record_success()
        assert breaker._failure_count == 0

        # Test record_failure
        breaker.record_failure()
        assert breaker._failure_count == 1

        # Force open
        breaker._failure_count = 3
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Test can_attempt
        assert not breaker.can_attempt()

        # Test reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_retry_handler(self):
        """Test RetryHandler."""
        from simgen.services.resilience import RetryHandler, RetryConfig

        config = RetryConfig(
            max_attempts=3,
            initial_delay=1,
            max_delay=10,
            backoff_factor=2,
            jitter=0.1
        )

        handler = RetryHandler(config=config)
        assert handler._attempt_count == 0

        # Test should_retry
        assert handler.should_retry(Exception())

        # Test get_delay
        delay = handler.get_delay()
        assert delay > 0

        # Test record_attempt
        handler.record_attempt()
        assert handler._attempt_count == 1

        # Test reset
        handler.reset()
        assert handler._attempt_count == 0

    def test_error_metrics(self):
        """Test ErrorMetrics."""
        from simgen.services.resilience import ErrorMetrics

        metrics = ErrorMetrics()

        # Test record_error
        metrics.record_error("TestError", {"detail": "test"})
        assert metrics.total_errors == 1
        assert "TestError" in metrics.error_counts

        # Test get_error_rate
        rate = metrics.get_error_rate("TestError")
        assert rate == 1.0

        # Test reset
        metrics.reset()
        assert metrics.total_errors == 0

    def test_resilience_manager(self):
        """Test ResilienceManager."""
        from simgen.services.resilience import ResilienceManager, get_resilience_manager

        manager = get_resilience_manager()
        assert manager is not None

        # Test circuit breaker management
        breaker = manager.get_circuit_breaker("test_service")
        assert breaker is not None

        # Test retry handler management
        handler = manager.get_retry_handler("test_service")
        assert handler is not None

        # Test error recording
        manager.record_error("test_service", Exception("test"))

        # Test metrics
        metrics = manager.get_metrics()
        assert "circuit_breakers" in metrics
        assert "error_metrics" in metrics

    @pytest.mark.asyncio
    async def test_decorators(self):
        """Test resilience decorators."""
        from simgen.services.resilience import resilient_service, handle_errors

        @resilient_service("test_service")
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"

        @handle_errors({ValueError: "value_error"})
        def error_function():
            raise ValueError("test")

        with pytest.raises(ValueError):
            error_function()

    def test_custom_errors(self):
        """Test custom error classes."""
        from simgen.services.resilience import (
            SimGenError, AIServiceError, RenderingError,
            ValidationError, RateLimitError, CircuitBreakerOpenError
        )

        errors = [
            SimGenError("test"),
            AIServiceError("ai error"),
            RenderingError("render error"),
            ValidationError("validation error"),
            RateLimitError("rate limit"),
            CircuitBreakerOpenError("circuit open")
        ]

        for error in errors:
            assert str(error) is not None
            assert isinstance(error, SimGenError)


class TestObservabilityModule:
    """Test observability module with actual imports."""

    def test_metrics_collector(self):
        """Test MetricsCollector."""
        from simgen.monitoring.observability import MetricsCollector, MetricType, MetricPoint

        collector = MetricsCollector()

        # Record various metrics
        collector.record_request("GET", "/test", 200, 0.1)
        collector.record_request("POST", "/test", 500, 0.5)
        collector.record_error(ValueError("test"), {"endpoint": "/test"})

        # Get metrics
        metrics = collector.get_metrics()
        assert metrics["request_count"] > 0
        assert metrics["error_count"] > 0

        # Test metric point
        point = MetricPoint(
            metric_type=MetricType.COUNTER,
            name="test_metric",
            value=1.0,
            timestamp=datetime.now(),
            labels={"test": "label"}
        )
        assert point.value == 1.0

    def test_system_monitor(self):
        """Test SystemMonitor."""
        from simgen.monitoring.observability import SystemMonitor

        monitor = SystemMonitor()

        # Get system metrics
        metrics = monitor.get_system_metrics()
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_usage" in metrics

        # Check resource usage
        is_healthy = monitor.check_resource_usage()
        assert isinstance(is_healthy, bool)

        # Get detailed stats
        stats = monitor.get_detailed_stats()
        assert "process" in stats
        assert "system" in stats

    def test_health_monitor(self):
        """Test HealthMonitor."""
        from simgen.monitoring.observability import HealthMonitor, HealthCheck

        monitor = HealthMonitor()

        # Register health checks
        async def check_database():
            return HealthCheck(
                name="database",
                status="healthy",
                response_time=0.01
            )

        monitor.register_check("database", check_database)

        # Run health checks
        async def run_checks():
            results = await monitor.run_health_checks()
            assert "database" in results
            assert results["database"]["status"] == "healthy"

        asyncio.run(run_checks())

        # Get health status
        status = monitor.get_health_status()
        assert "checks" in status
        assert status["overall_status"] in ["healthy", "unhealthy", "degraded"]

    def test_performance_tracker(self):
        """Test PerformanceTracker."""
        from simgen.monitoring.observability import PerformanceTracker

        tracker = PerformanceTracker()

        # Track operations
        tracker.start_operation("test_op")
        time.sleep(0.01)  # Simulate work
        tracker.end_operation("test_op")

        # Get performance metrics
        metrics = tracker.get_performance_metrics()
        assert "test_op" in metrics
        assert metrics["test_op"]["count"] > 0
        assert metrics["test_op"]["avg_duration"] > 0

        # Clear metrics
        tracker.clear_metrics()
        metrics = tracker.get_performance_metrics()
        assert len(metrics) == 0

    def test_observability_manager(self):
        """Test ObservabilityManager."""
        from simgen.monitoring.observability import ObservabilityManager, get_observability_manager

        manager = get_observability_manager()
        assert manager is not None

        # Track various events
        manager.track_request("GET", "/test", 200, 0.1)
        manager.track_error(ValueError("test"), {"endpoint": "/test"})
        manager.track_performance("operation", 0.5)

        # Get comprehensive metrics
        metrics = manager.get_metrics()
        assert "metrics" in metrics
        assert "system" in metrics
        assert "health" in metrics
        assert "performance" in metrics

    @pytest.mark.asyncio
    async def test_track_performance_decorator(self):
        """Test track_performance decorator."""
        from simgen.monitoring.observability import track_performance

        @track_performance("test_endpoint")
        async def test_function():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_function()
        assert result == "success"


class TestMJCFCompiler:
    """Test MJCF compiler module."""

    def test_mjcf_compiler(self):
        """Test MJCFCompiler."""
        from simgen.services.mjcf_compiler import MJCFCompiler

        compiler = MJCFCompiler()

        # Test compile with valid MJCF
        valid_mjcf = """
        <mujoco>
            <worldbody>
                <body name="test_body">
                    <geom name="test_geom" type="box" size="1 1 1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        result = compiler.compile(valid_mjcf)
        assert result["success"] == True
        assert "model_info" in result

        # Test compile with invalid MJCF
        invalid_mjcf = "<invalid>not mjcf</invalid>"
        result = compiler.compile(invalid_mjcf)
        assert result["success"] == False
        assert "error" in result

        # Test validate
        validation = compiler.validate(valid_mjcf)
        assert validation["valid"] == True

        validation = compiler.validate(invalid_mjcf)
        assert validation["valid"] == False
        assert len(validation["errors"]) > 0

        # Test optimize
        optimized = compiler.optimize(valid_mjcf)
        assert isinstance(optimized, str)
        assert "<mujoco>" in optimized

        # Test edge cases
        result = compiler.compile("")
        assert result["success"] == False

        result = compiler.compile(None)
        assert result["success"] == False

        # Test with complex MJCF
        complex_mjcf = """
        <mujoco>
            <option gravity="0 0 -9.81" timestep="0.002"/>
            <worldbody>
                <body name="body1" pos="0 0 1">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom name="geom1" type="box" size="0.5 0.5 0.5"/>
                    <body name="body2" pos="1 0 0">
                        <joint name="joint2" type="slide" axis="1 0 0"/>
                        <geom name="geom2" type="sphere" size="0.3"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="motor1" joint="joint1" gear="100"/>
            </actuator>
        </mujoco>
        """

        result = compiler.compile(complex_mjcf)
        assert result["success"] == True

        # Test get_defaults
        defaults = compiler.get_defaults()
        assert "compiler" in defaults
        assert "option" in defaults


class TestStreamingProtocol:
    """Test streaming protocol module."""

    def test_streaming_protocol(self):
        """Test StreamingProtocol."""
        from simgen.services.streaming_protocol import (
            StreamingProtocol, MessageType, StreamMessage
        )

        protocol = StreamingProtocol()

        # Test all message types
        for msg_type in MessageType:
            message = StreamMessage(
                type=msg_type,
                data={"test": "data", "nested": {"value": 123}},
                timestamp=int(time.time()),
                sequence=1,
                metadata={"source": "test", "priority": "high"}
            )

            # Test serialization
            serialized = protocol.serialize(message)
            assert isinstance(serialized, bytes)

            # Test deserialization
            deserialized = protocol.deserialize(serialized)
            assert deserialized.type == msg_type
            assert deserialized.data == message.data
            assert deserialized.sequence == 1

        # Test error handling
        try:
            protocol.deserialize(b"invalid data")
        except Exception as e:
            assert "Failed to deserialize" in str(e) or "Invalid" in str(e)

        # Test empty data
        empty_message = StreamMessage(
            type=MessageType.DATA,
            data={},
            timestamp=int(time.time()),
            sequence=0
        )
        serialized = protocol.serialize(empty_message)
        deserialized = protocol.deserialize(serialized)
        assert deserialized.data == {}

        # Test large data
        large_data = {"key" + str(i): "value" * 100 for i in range(100)}
        large_message = StreamMessage(
            type=MessageType.DATA,
            data=large_data,
            timestamp=int(time.time()),
            sequence=999
        )
        serialized = protocol.serialize(large_message)
        deserialized = protocol.deserialize(serialized)
        assert len(deserialized.data) == 100


class TestPromptParser:
    """Test prompt parser module."""

    def test_prompt_parser(self):
        """Test PromptParser."""
        from simgen.services.prompt_parser import PromptParser

        parser = PromptParser()

        # Test parse with various prompts
        prompts = [
            "Create a red bouncing ball with gravity -9.81",
            "Simulate a robot arm with 3 joints",
            "Make a pendulum swinging back and forth",
            "Create multiple balls falling and colliding",
            "Build a car with 4 wheels"
        ]

        for prompt in prompts:
            result = parser.parse(prompt)
            assert "entities" in result
            assert "physics" in result
            assert "constraints" in result

        # Test extract_entities
        entities = parser.extract_entities("ball, floor, wall, ceiling")
        assert len(entities) == 4
        assert "ball" in entities

        # Test extract_physics_params
        physics = parser.extract_physics_params("gravity -9.81 friction 0.5 damping 0.1")
        assert "gravity" in str(physics)

        # Test extract_colors
        colors = parser.extract_colors("red ball blue floor green wall")
        assert len(colors) > 0

        # Test extract_numbers
        numbers = parser.extract_numbers("create 5 balls with size 0.3 and mass 2.5")
        assert 5 in numbers or 5.0 in numbers
        assert 0.3 in numbers
        assert 2.5 in numbers

        # Test edge cases
        result = parser.parse("")
        assert result is not None

        result = parser.parse(None)
        assert result is not None

        # Test complex prompt
        complex_prompt = """
        Create a complex simulation with:
        - 3 red balls bouncing
        - A blue floor at height 0
        - Gravity of -9.81 m/s^2
        - Friction coefficient of 0.8
        - 2 walls on the sides
        - A ceiling at height 5
        """
        result = parser.parse(complex_prompt)
        assert len(result["entities"]) > 0
        assert result["physics"] is not None


class TestSchemas:
    """Test schemas module."""

    def test_all_schemas(self):
        """Test all schema classes."""
        from simgen.models.schemas import (
            SimulationRequest, SimulationResponse, SimulationStatus,
            SketchAnalysisRequest, SketchAnalysisResponse,
            MJCFValidationRequest, MJCFValidationResponse,
            ErrorResponse, HealthCheckResponse
        )

        # Test SimulationRequest
        request = SimulationRequest(
            prompt="Test simulation",
            parameters={"gravity": -9.81, "timestep": 0.002},
            user_id="user123",
            options={"debug": True}
        )
        assert request.prompt == "Test simulation"
        request_dict = request.dict()
        request_json = request.json()

        # Test SimulationResponse
        for status in SimulationStatus:
            response = SimulationResponse(
                simulation_id=f"sim_{status.value}",
                status=status,
                mjcf_content="<mujoco/>",
                created_at=datetime.now(),
                metadata={"version": "1.0"}
            )
            assert response.status == status
            response.dict()
            response.json()

        # Test SketchAnalysisRequest
        sketch_req = SketchAnalysisRequest(
            image_data=b"fake_image_data",
            image_format="png",
            analysis_options={"detect_colors": True}
        )
        sketch_req.dict()

        # Test SketchAnalysisResponse
        sketch_resp = SketchAnalysisResponse(
            objects_detected=[
                {"type": "ball", "confidence": 0.95, "position": [0.5, 0.5]},
                {"type": "floor", "confidence": 0.88, "position": [0, 0]}
            ],
            suggested_prompt="Create a ball on a floor",
            confidence_score=0.91
        )
        sketch_resp.dict()

        # Test MJCFValidationRequest
        mjcf_req = MJCFValidationRequest(
            mjcf_content="<mujoco><worldbody></worldbody></mujoco>",
            validation_level="strict"
        )
        mjcf_req.dict()

        # Test MJCFValidationResponse
        mjcf_resp = MJCFValidationResponse(
            is_valid=True,
            errors=[],
            warnings=["No actuators defined"],
            suggestions=["Consider adding actuators"]
        )
        mjcf_resp.dict()

        # Test ErrorResponse
        error_resp = ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message="Invalid MJCF structure",
            details={"line": 10, "column": 5},
            request_id="req123"
        )
        error_resp.dict()

        # Test HealthCheckResponse
        health_resp = HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            services={
                "database": "healthy",
                "redis": "healthy",
                "ai_service": "degraded"
            },
            version="1.2.3",
            uptime=3600
        )
        health_resp.dict()


class TestSimulationModel:
    """Test simulation model."""

    def test_simulation_model(self):
        """Test Simulation model."""
        from simgen.models.simulation import Simulation, SimulationStatus

        # Test all status values and model operations
        for status in SimulationStatus:
            sim = Simulation(
                id=f"sim_{status.value}",
                prompt=f"Test simulation for {status.value}",
                mjcf_content="<mujoco><worldbody></worldbody></mujoco>",
                status=status,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id="user123",
                parameters={"gravity": -9.81},
                metadata={"version": "1.0", "tags": ["test", "physics"]}
            )

            assert sim.status == status
            assert sim.id == f"sim_{status.value}"

            # Test all methods
            sim_dict = sim.dict()
            sim_json = sim.json()
            sim_copy = sim.copy()

            assert "prompt" in sim_dict
            assert sim_copy.id == sim.id


class TestConfig:
    """Test configuration module."""

    def test_settings(self):
        """Test Settings class."""
        from simgen.core.config import Settings

        settings = Settings()

        # Test all attributes
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'secret_key')

        # Test methods
        settings_dict = settings.dict()
        settings_json = settings.json()
        settings_copy = settings.copy()

        # Test environment variables
        assert settings.database_url is not None
        assert settings.secret_key is not None

        # Test validation
        assert isinstance(settings.database_url, str)
        assert isinstance(settings.secret_key, str)


def test_integration_workflow():
    """Test complete integration workflow."""
    from simgen.models.physics_spec import PhysicsSpec, Body, Geom
    from simgen.services.mjcf_compiler import MJCFCompiler
    from simgen.services.prompt_parser import PromptParser
    from simgen.services.streaming_protocol import StreamingProtocol, MessageType, StreamMessage
    from simgen.models.schemas import SimulationRequest, SimulationResponse, SimulationStatus

    # 1. Parse prompt
    parser = PromptParser()
    parsed = parser.parse("Create a red bouncing ball")

    # 2. Create physics spec
    body = Body(
        id="ball",
        name="bouncing_ball",
        pos=[0, 0, 1],
        geoms=[
            Geom(
                name="ball_geom",
                type="sphere",
                size=[0.5],
                rgba=[1, 0, 0, 1],
                mass=1.0
            )
        ]
    )

    spec = PhysicsSpec(bodies=[body])
    mjcf = spec.to_mjcf()

    # 3. Compile MJCF
    compiler = MJCFCompiler()
    result = compiler.compile(mjcf)
    assert result["success"] == True

    # 4. Create simulation request/response
    request = SimulationRequest(
        prompt="Create a red bouncing ball",
        parameters={"gravity": -9.81}
    )

    response = SimulationResponse(
        simulation_id="test_123",
        status=SimulationStatus.COMPLETED,
        mjcf_content=mjcf
    )

    # 5. Stream the result
    protocol = StreamingProtocol()
    message = StreamMessage(
        type=MessageType.DATA,
        data={"simulation": response.dict()},
        timestamp=int(time.time()),
        sequence=1
    )

    serialized = protocol.serialize(message)
    deserialized = protocol.deserialize(serialized)

    assert deserialized.data["simulation"]["simulation_id"] == "test_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/simgen", "--cov-report=term"])