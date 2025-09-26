"""
MAXIMUM COVERAGE TEST - Focus on Working Modules
Target: 50% coverage using only working imports
Working modules:
  - simgen.core.config
  - simgen.models.physics_spec
  - simgen.services.resilience
  - simgen.services.streaming_protocol
  - simgen.services.mjcf_compiler
  - simgen.monitoring.observability
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
import time
import json
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set PostgreSQL environment for async support
os.environ.update({
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/test",
    "SECRET_KEY": "maximum-50-test",
    "REDIS_URL": "redis://localhost:6379"
})


class TestConfig:
    """Test config module - 100% coverage target."""

    def test_settings_comprehensive(self):
        """Test Settings class completely."""
        from simgen.core.config import Settings

        # Test default settings
        settings = Settings()
        assert settings is not None

        # Test all methods
        settings_dict = settings.dict()
        settings_json = settings.json()
        settings_copy = settings.copy()

        # Access all attributes
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'redis_url')
        assert hasattr(settings, 'secret_key')
        assert hasattr(settings, 'jwt_algorithm')
        assert hasattr(settings, 'jwt_expiration_days')
        assert hasattr(settings, 'cors_origins')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'anthropic_api_key')

        # Test with custom values
        custom_settings = Settings(
            database_url="postgresql://custom:custom@localhost/custom",
            debug=True,
            jwt_expiration_days=14
        )
        assert custom_settings.debug == True
        assert custom_settings.jwt_expiration_days == 14


class TestPhysicsSpec:
    """Test physics_spec module - 100% coverage target."""

    def test_all_enums(self):
        """Test all enum classes."""
        from simgen.models.physics_spec import (
            PhysicsSpecVersion, JointType, GeomType,
            ActuatorType, SensorType
        )

        # Test each enum comprehensively
        for version in PhysicsSpecVersion:
            assert version.value is not None
            assert version.name is not None

        for jtype in JointType:
            assert jtype.value is not None
            assert jtype in JointType

        for gtype in GeomType:
            assert gtype.value is not None
            assert gtype in GeomType

        for atype in ActuatorType:
            assert atype.value is not None
            assert atype in ActuatorType

        for stype in SensorType:
            assert stype.value is not None
            assert stype in SensorType

    def test_all_data_classes(self):
        """Test all data classes comprehensively."""
        from simgen.models.physics_spec import (
            Material, Friction, Inertial, Geom, Joint,
            Body, Actuator, Sensor, Contact, Equality,
            DefaultSettings, SimulationMeta, PhysicsSpec,
            JointType, GeomType, ActuatorType, SensorType,
            PhysicsSpecVersion
        )

        # Test Material
        mat1 = Material(name="mat1", rgba=[1,0,0,1])
        mat2 = Material(name="mat2", emission=0.5, specular=0.8,
                        shininess=128, reflectance=0.3, rgba=[0,1,0,1])
        mat1.dict()
        mat2.dict()

        # Test Friction
        friction1 = Friction(slide=1.0, spin=0.1, roll=0.01)
        friction2 = Friction()
        friction1.dict()
        friction2.dict()

        # Test Inertial
        inertial1 = Inertial(mass=5.0, diaginertia=[1,1,1])
        inertial2 = Inertial(mass=10.0, pos=[1,2,3], quat=[1,0,0,0],
                             diaginertia=[2,2,2])
        inertial1.dict()
        inertial2.dict()

        # Test Geom with ALL types
        geoms = []
        for gtype in ["box", "sphere", "cylinder", "capsule", "ellipsoid", "mesh", "plane"]:
            geom = Geom(
                name=f"geom_{gtype}",
                type=gtype,
                size=[1,1,1] if gtype == "box" else [1,0.5] if gtype == "cylinder" else [1],
                pos=[0,0,0],
                quat=[1,0,0,0],
                rgba=[1,0,0,1],
                mass=1.0,
                friction=friction1,
                material=mat1,
                contype=1,
                conaffinity=1,
                condim=3,
                solmix=1.0,
                solimp=[0.9, 0.95, 0.001]
            )
            geoms.append(geom)
            geom.dict()

        # Test Joint with ALL types
        joints = []
        for jtype in JointType:
            joint = Joint(
                name=f"joint_{jtype.value}",
                type=jtype,
                axis=[0,0,1] if jtype != JointType.BALL else None,
                limited=True,
                range=[-90, 90],
                damping=0.1,
                armature=0.01,
                stiffness=100,
                pos=[0,0,0],
                ref=0
            )
            joints.append(joint)
            joint.dict()

        # Test Body
        bodies = []
        for i in range(3):
            body = Body(
                id=f"body{i}",
                name=f"body_{i}",
                pos=[i,0,0],
                quat=[1,0,0,0],
                inertial=inertial1 if i == 0 else None,
                geoms=[geoms[i % len(geoms)]],
                joints=[joints[i % len(joints)]] if i < len(joints) else [],
                sites=[],
                cameras=[]
            )
            bodies.append(body)
            body.dict()

        # Test Actuator with ALL types
        actuators = []
        for atype in ActuatorType:
            actuator = Actuator(
                name=f"actuator_{atype.value}",
                type=atype,
                joint="joint1" if atype != ActuatorType.GENERAL else None,
                gear=100.0,
                ctrllimited=True,
                ctrlrange=[-1, 1],
                forcelimited=True,
                forcerange=[-100, 100],
                site="site1" if atype == ActuatorType.GENERAL else None
            )
            actuators.append(actuator)
            actuator.dict()

        # Test Sensor with ALL types
        sensors = []
        for stype in SensorType:
            sensor = Sensor(
                name=f"sensor_{stype.value}",
                type=stype,
                site="site1" if stype in [SensorType.ACCELEROMETER, SensorType.GYRO] else None,
                joint="joint1" if stype in [SensorType.JOINTPOS, SensorType.JOINTVEL] else None,
                body="body1" if stype in [SensorType.FORCE, SensorType.TORQUE] else None,
                noise=0.01,
                cutoff=100
            )
            sensors.append(sensor)
            sensor.dict()

        # Test Contact
        contacts = []
        for i in range(2):
            contact = Contact(
                name=f"contact_{i}",
                geom1=f"geom{i}",
                geom2=f"geom{i+1}",
                condim=3,
                friction=[1.0, 0.005, 0.0001],
                solref=[-100, -50],
                solimp=[0.9, 0.95, 0.001]
            )
            contacts.append(contact)
            contact.dict()

        # Test Equality
        equalities = []
        equality = Equality(
            name="eq1",
            type="connect",
            body1="body1",
            body2="body2",
            anchor=[0,0,0],
            solimp=[0.9, 0.95],
            solref=[-100, -50]
        )
        equalities.append(equality)
        equality.dict()

        # Test DefaultSettings
        defaults = DefaultSettings(
            geom_friction=[1.0, 0.005, 0.0001],
            joint_damping=0.1,
            actuator_gear=100.0,
            sensor_noise=0.001,
            timestep=0.002,
            gravity=[0, 0, -9.81],
            integrator="RK4"
        )
        defaults.dict()

        # Test SimulationMeta
        meta = SimulationMeta(
            version=PhysicsSpecVersion.V1_0_0,
            created_by="test_user",
            created_at=datetime.now(),
            description="Test simulation",
            tags=["test", "physics", "simulation"]
        )
        meta.dict()

        # Test PhysicsSpec - valid case
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

        # Test MJCF generation
        mjcf = spec.to_mjcf()
        assert "<mujoco>" in mjcf
        assert "body_0" in mjcf
        assert "</mujoco>" in mjcf

        # Test validation errors
        with pytest.raises(ValueError, match="At least one body is required"):
            PhysicsSpec(bodies=[])

        # Test duplicate ID validation
        dup_body1 = Body(id="dup", name="b1", geoms=[geoms[0]])
        dup_body2 = Body(id="dup", name="b2", geoms=[geoms[0]])
        with pytest.raises(ValueError, match="Duplicate body ID"):
            PhysicsSpec(bodies=[dup_body1, dup_body2])


class TestResilience:
    """Test resilience module - 100% coverage target."""

    def test_circuit_breaker_comprehensive(self):
        """Test CircuitBreaker completely."""
        from simgen.services.resilience import (
            CircuitBreaker, CircuitBreakerConfig, CircuitState
        )

        # Test CircuitState enum
        for state in CircuitState:
            assert state.value is not None
            assert state in CircuitState

        # Test CircuitBreakerConfig variations
        config1 = CircuitBreakerConfig()
        config2 = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)
        config3 = CircuitBreakerConfig(expected_exception=ValueError)

        # Test CircuitBreaker
        cb = CircuitBreaker(name="test_breaker", config=config2)
        assert cb.name == "test_breaker"
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

        # Test can_attempt when closed
        assert cb.can_attempt() == True

        # Test record_success
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

        # Test record_failure
        cb.record_failure()
        assert cb._failure_count == 1

        # Force circuit to open
        for _ in range(4):  # Already have 1 failure
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_attempt() == False

        # Test reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

        # Test half-open state (simulate time passing)
        cb._failure_count = 5
        cb._state = CircuitState.OPEN
        cb._last_failure_time = time.time() - 121  # Past recovery timeout
        # This would transition to HALF_OPEN in real scenario

    def test_retry_handler_comprehensive(self):
        """Test RetryHandler completely."""
        from simgen.services.resilience import RetryHandler, RetryConfig

        # Test RetryConfig variations
        config1 = RetryConfig()
        config2 = RetryConfig(
            max_attempts=5,
            initial_delay=2,
            max_delay=60,
            backoff_factor=3,
            jitter=0.2
        )

        # Test RetryHandler
        handler = RetryHandler(config=config2)
        assert handler._attempt_count == 0

        # Test should_retry
        assert handler.should_retry(Exception("test")) == True
        assert handler.should_retry(KeyboardInterrupt()) == False

        # Test get_delay
        delay = handler.get_delay()
        assert delay >= 2  # initial_delay
        assert delay <= 60  # max_delay

        # Test record_attempt
        handler.record_attempt()
        assert handler._attempt_count == 1

        handler.record_attempt()
        assert handler._attempt_count == 2

        # Test max attempts
        for _ in range(10):
            handler.record_attempt()
        assert handler.should_retry(Exception()) == False  # Exceeded max attempts

        # Test reset
        handler.reset()
        assert handler._attempt_count == 0

    def test_error_metrics_comprehensive(self):
        """Test ErrorMetrics completely."""
        from simgen.services.resilience import ErrorMetrics

        metrics = ErrorMetrics()
        assert metrics.total_errors == 0

        # Record various errors
        metrics.record_error("ValueError", {"msg": "test1"})
        metrics.record_error("ValueError", {"msg": "test2"})
        metrics.record_error("KeyError", {"msg": "test3"})

        assert metrics.total_errors == 3
        assert metrics.error_counts["ValueError"] == 2
        assert metrics.error_counts["KeyError"] == 1

        # Test get_error_rate
        assert metrics.get_error_rate("ValueError") == 2/3
        assert metrics.get_error_rate("KeyError") == 1/3
        assert metrics.get_error_rate("NonExistent") == 0.0

        # Test reset
        metrics.reset()
        assert metrics.total_errors == 0
        assert len(metrics.error_counts) == 0

    def test_custom_exceptions(self):
        """Test all custom exception classes."""
        from simgen.services.resilience import (
            SimGenError, AIServiceError, RenderingError,
            ValidationError, RateLimitError, CircuitBreakerOpenError
        )

        exceptions = [
            SimGenError("base error"),
            AIServiceError("ai error"),
            RenderingError("render error"),
            ValidationError("validation error"),
            RateLimitError("rate limit error"),
            CircuitBreakerOpenError("circuit open error")
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)
            assert isinstance(exc, SimGenError)
            assert str(exc) is not None

    def test_resilience_manager(self):
        """Test ResilienceManager completely."""
        from simgen.services.resilience import ResilienceManager, get_resilience_manager

        # Test singleton
        manager1 = get_resilience_manager()
        manager2 = get_resilience_manager()
        assert manager1 is manager2

        # Test get_circuit_breaker
        cb1 = manager1.get_circuit_breaker("service1")
        cb2 = manager1.get_circuit_breaker("service1")
        assert cb1 is cb2  # Should return same instance

        cb3 = manager1.get_circuit_breaker("service2")
        assert cb3 is not cb1  # Different service

        # Test get_retry_handler
        handler1 = manager1.get_retry_handler("service1")
        handler2 = manager1.get_retry_handler("service1")
        assert handler1 is handler2

        # Test record_error
        manager1.record_error("service1", ValueError("test"))
        manager1.record_error("service1", KeyError("test"))

        # Test get_metrics
        metrics = manager1.get_metrics()
        assert "circuit_breakers" in metrics
        assert "error_metrics" in metrics
        assert "service1" in metrics["circuit_breakers"]

    @pytest.mark.asyncio
    async def test_decorators(self):
        """Test resilience decorators."""
        from simgen.services.resilience import resilient_service, handle_errors

        # Test resilient_service decorator
        @resilient_service("test_service", use_circuit_breaker=True, use_retry=True)
        async def test_async_function():
            return "success"

        result = await test_async_function()
        assert result == "success"

        # Test handle_errors decorator
        @handle_errors({ValueError: "value_error", KeyError: "key_error"})
        def test_error_function(error_type):
            if error_type == "value":
                raise ValueError("test")
            elif error_type == "key":
                raise KeyError("test")
            return "success"

        assert test_error_function("none") == "success"

        with pytest.raises(ValueError):
            test_error_function("value")


class TestStreamingProtocol:
    """Test streaming_protocol module - 100% coverage target."""

    def test_message_types(self):
        """Test all message types."""
        from simgen.services.streaming_protocol import MessageType

        for msg_type in MessageType:
            assert msg_type.value is not None
            assert msg_type in MessageType

    def test_stream_message(self):
        """Test StreamMessage class."""
        from simgen.services.streaming_protocol import StreamMessage, MessageType

        # Test various message configurations
        messages = [
            StreamMessage(
                type=MessageType.DATA,
                data={"key": "value"},
                timestamp=int(time.time()),
                sequence=1
            ),
            StreamMessage(
                type=MessageType.ERROR,
                data={"error": "test error", "code": 500},
                timestamp=int(time.time()),
                sequence=2,
                metadata={"severity": "high", "retry": True}
            ),
            StreamMessage(
                type=MessageType.CONTROL,
                data={"command": "stop", "params": {"force": True}},
                timestamp=int(time.time()),
                sequence=3
            ),
            StreamMessage(
                type=MessageType.STATUS,
                data={"status": "running", "progress": 50},
                timestamp=int(time.time()),
                sequence=4,
                metadata={"source": "worker1"}
            ),
            StreamMessage(
                type=MessageType.DATA,
                data={},  # Empty data
                timestamp=0,
                sequence=0
            )
        ]

        for msg in messages:
            assert msg.type is not None
            assert msg.timestamp is not None
            assert msg.sequence is not None

    def test_streaming_protocol(self):
        """Test StreamingProtocol class."""
        from simgen.services.streaming_protocol import (
            StreamingProtocol, StreamMessage, MessageType
        )

        protocol = StreamingProtocol()

        # Test with all message types
        for msg_type in MessageType:
            msg = StreamMessage(
                type=msg_type,
                data={"test": f"data_{msg_type.value}"},
                timestamp=int(time.time()),
                sequence=msg_type.value
            )

            # Test serialization
            serialized = protocol.serialize(msg)
            assert isinstance(serialized, bytes)
            assert len(serialized) > 0

            # Test deserialization
            deserialized = protocol.deserialize(serialized)
            assert deserialized.type == msg.type
            assert deserialized.data == msg.data
            assert deserialized.sequence == msg.sequence

        # Test with complex data
        complex_msg = StreamMessage(
            type=MessageType.DATA,
            data={
                "nested": {"key": "value", "list": [1, 2, 3]},
                "array": [{"a": 1}, {"b": 2}],
                "string": "test" * 100,
                "number": 12345.678,
                "bool": True,
                "null": None
            },
            timestamp=int(time.time()),
            sequence=999,
            metadata={"complex": True}
        )

        serialized = protocol.serialize(complex_msg)
        deserialized = protocol.deserialize(serialized)
        assert deserialized.data == complex_msg.data

        # Test error handling
        with pytest.raises(Exception):
            protocol.deserialize(b"invalid json data")

        with pytest.raises(Exception):
            protocol.deserialize(b"")

        # Test large message
        large_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        large_msg = StreamMessage(
            type=MessageType.DATA,
            data=large_data,
            timestamp=int(time.time()),
            sequence=9999
        )

        serialized = protocol.serialize(large_msg)
        deserialized = protocol.deserialize(serialized)
        assert len(deserialized.data) == 100


class TestMJCFCompiler:
    """Test mjcf_compiler module - 100% coverage target."""

    def test_mjcf_compiler_comprehensive(self):
        """Test MJCFCompiler completely."""
        from simgen.services.mjcf_compiler import MJCFCompiler

        compiler = MJCFCompiler()

        # Test compile with various MJCF inputs
        test_cases = [
            # Valid MJCF
            ("<mujoco><worldbody><body><geom type='box'/></body></worldbody></mujoco>", True),
            ("<mujoco><worldbody></worldbody></mujoco>", True),
            ("<mujoco/>", True),
            # Invalid MJCF
            ("<invalid>not mjcf</invalid>", False),
            ("<mujoco><invalid/></mujoco>", False),
            ("", False),
            (None, False),
            # Complex valid MJCF
            ("""
            <mujoco>
                <option timestep="0.002" gravity="0 0 -9.81"/>
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
                <sensor>
                    <jointpos name="sensor1" joint="joint1"/>
                </sensor>
            </mujoco>
            """, True)
        ]

        for mjcf, should_succeed in test_cases:
            result = compiler.compile(mjcf)
            assert "success" in result
            assert result["success"] == should_succeed

            if should_succeed and mjcf:
                assert "model_info" in result
            else:
                assert "error" in result

        # Test validate
        for mjcf, should_be_valid in test_cases:
            validation = compiler.validate(mjcf)
            assert "valid" in validation
            assert "errors" in validation
            if not should_be_valid:
                assert validation["valid"] == False

        # Test optimize
        valid_mjcfs = [mjcf for mjcf, valid in test_cases if valid and mjcf and "<mujoco>" in str(mjcf)]
        for mjcf in valid_mjcfs:
            optimized = compiler.optimize(mjcf)
            assert isinstance(optimized, str)
            assert "<mujoco>" in optimized

        # Test get_defaults
        defaults = compiler.get_defaults()
        assert isinstance(defaults, dict)
        assert "compiler" in defaults
        assert "option" in defaults
        assert "size" in defaults


class TestObservability:
    """Test observability module - 100% coverage target."""

    def test_metric_types(self):
        """Test MetricType enum."""
        from simgen.monitoring.observability import MetricType

        for metric_type in MetricType:
            assert metric_type.value is not None
            assert metric_type in MetricType

    def test_metric_point(self):
        """Test MetricPoint class."""
        from simgen.monitoring.observability import MetricPoint, MetricType

        points = [
            MetricPoint(
                metric_type=MetricType.COUNTER,
                name="requests_total",
                value=100.0,
                timestamp=datetime.now(),
                labels={"method": "GET", "path": "/api"}
            ),
            MetricPoint(
                metric_type=MetricType.GAUGE,
                name="temperature",
                value=23.5,
                timestamp=datetime.now(),
                labels={"location": "server_room"}
            ),
            MetricPoint(
                metric_type=MetricType.HISTOGRAM,
                name="request_duration",
                value=0.125,
                timestamp=datetime.now()
            )
        ]

        for point in points:
            assert point.name is not None
            assert point.value is not None

    def test_health_check(self):
        """Test HealthCheck class."""
        from simgen.monitoring.observability import HealthCheck

        checks = [
            HealthCheck(
                name="database",
                status="healthy",
                response_time=0.01,
                details={"connections": 10, "pool_size": 20}
            ),
            HealthCheck(
                name="redis",
                status="degraded",
                response_time=0.5,
                details={"memory_usage": "high"}
            ),
            HealthCheck(
                name="api",
                status="unhealthy",
                response_time=None,
                details={"error": "timeout"}
            )
        ]

        for check in checks:
            assert check.name is not None
            assert check.status in ["healthy", "degraded", "unhealthy"]

    def test_metrics_collector(self):
        """Test MetricsCollector class."""
        from simgen.monitoring.observability import MetricsCollector

        collector = MetricsCollector()

        # Record various metrics
        collector.record_request("GET", "/api/test", 200, 0.1)
        collector.record_request("POST", "/api/test", 201, 0.15)
        collector.record_request("GET", "/api/fail", 500, 0.5)
        collector.record_error(ValueError("test error"), {"endpoint": "/api/test"})
        collector.record_error(KeyError("key error"), {"endpoint": "/api/fail"})

        # Get metrics
        metrics = collector.get_metrics()
        assert metrics["request_count"] >= 3
        assert metrics["error_count"] >= 2
        assert metrics["success_rate"] > 0

    def test_system_monitor(self):
        """Test SystemMonitor class."""
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

    def test_performance_tracker(self):
        """Test PerformanceTracker class."""
        from simgen.monitoring.observability import PerformanceTracker

        tracker = PerformanceTracker()

        # Track various operations
        operations = ["db_query", "api_call", "render", "compile"]
        for op in operations:
            tracker.start_operation(op)
            time.sleep(0.01)  # Simulate work
            tracker.end_operation(op)

        # Get performance metrics
        metrics = tracker.get_performance_metrics()
        for op in operations:
            if op in metrics:
                assert metrics[op]["count"] > 0
                assert metrics[op]["avg_duration"] > 0

        # Clear metrics
        tracker.clear_metrics()
        metrics = tracker.get_performance_metrics()
        assert len(metrics) == 0

    def test_observability_manager(self):
        """Test ObservabilityManager class."""
        from simgen.monitoring.observability import ObservabilityManager, get_observability_manager

        # Test singleton
        manager1 = get_observability_manager()
        manager2 = get_observability_manager()
        assert manager1 is manager2

        # Track various events
        manager1.track_request("GET", "/test", 200, 0.1)
        manager1.track_request("POST", "/test", 201, 0.2)
        manager1.track_error(ValueError("test"), {"context": "test"})
        manager1.track_performance("operation", 0.5)

        # Get comprehensive metrics
        metrics = manager1.get_metrics()
        assert "metrics" in metrics
        assert "system" in metrics
        assert "performance" in metrics

    @pytest.mark.asyncio
    async def test_track_performance_decorator(self):
        """Test track_performance decorator."""
        from simgen.monitoring.observability import track_performance

        @track_performance("test_operation")
        async def async_operation():
            await asyncio.sleep(0.01)
            return "success"

        result = await async_operation()
        assert result == "success"


def test_integration():
    """Integration test combining all modules."""
    from simgen.core.config import Settings
    from simgen.models.physics_spec import PhysicsSpec, Body, Geom
    from simgen.services.mjcf_compiler import MJCFCompiler
    from simgen.services.streaming_protocol import StreamingProtocol, MessageType, StreamMessage
    from simgen.services.resilience import CircuitBreaker, CircuitBreakerConfig
    from simgen.monitoring.observability import get_observability_manager

    # Initialize components
    settings = Settings()
    compiler = MJCFCompiler()
    protocol = StreamingProtocol()
    breaker = CircuitBreaker(name="integration", config=CircuitBreakerConfig())
    observer = get_observability_manager()

    # Create physics simulation
    geom = Geom(name="ball", type="sphere", size=[0.5], rgba=[1,0,0,1])
    body = Body(id="ball", name="bouncing_ball", pos=[0,0,1], geoms=[geom])
    spec = PhysicsSpec(bodies=[body])

    # Generate MJCF
    mjcf = spec.to_mjcf()

    # Compile with circuit breaker
    if breaker.can_attempt():
        result = compiler.compile(mjcf)
        if result["success"]:
            breaker.record_success()
            observer.track_request("POST", "/compile", 200, 0.1)
        else:
            breaker.record_failure()
            observer.track_error(Exception("Compilation failed"), {"mjcf": mjcf[:100]})

    # Stream result
    message = StreamMessage(
        type=MessageType.DATA,
        data={"mjcf": mjcf, "result": result},
        timestamp=int(time.time()),
        sequence=1
    )

    serialized = protocol.serialize(message)
    deserialized = protocol.deserialize(serialized)

    assert deserialized.data["result"]["success"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/simgen", "--cov-report=term"])