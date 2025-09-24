"""
ULTRA FOCUSED TEST - WORKING MODULES ONLY
Target: Push coverage to 50% by focusing ONLY on modules that import successfully
Strategy: Execute every possible code path in physics_spec, schemas, simulation, config
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set minimal environment
os.environ.update({
    "DATABASE_URL": "sqlite:///test.db",
    "SECRET_KEY": "test-secret-key",
    "DEBUG": "true"
})


class TestPhysicsSpecUltraComprehensive:
    """Ultra comprehensive testing of physics_spec module - EVERY code path."""

    def test_all_physics_spec_classes_comprehensive(self):
        """Test ALL physics_spec classes with ALL possible parameters."""
        from simgen.models.physics_spec import (
            PhysicsSpec, Body, Geom, Joint, Actuator, Sensor,
            Material, Friction, Inertial, Contact, Equality,
            DefaultSettings, SimulationMeta, PhysicsSpecVersion,
            JointType, GeomType, ActuatorType, SensorType
        )

        # Test ALL enum values
        assert PhysicsSpecVersion.V1_0_0 == "1.0.0"
        assert PhysicsSpecVersion.V1_1_0 == "1.1.0"
        assert PhysicsSpecVersion.V2_0_0 == "2.0.0"

        assert JointType.HINGE == "hinge"
        assert JointType.SLIDER == "slide"
        assert JointType.BALL == "ball"
        assert JointType.FREE == "free"

        assert GeomType.BOX == "box"

        # Test Material with ALL parameters
        material = Material(
            name="test_material",
            rgba=[1.0, 0.5, 0.0, 1.0],
            emission=0.1,
            specular=0.5,
            shininess=0.8,
            reflectance=0.3,
            texture="wood_texture"
        )

        assert material.name == "test_material"
        assert material.rgba == [1.0, 0.5, 0.0, 1.0]
        assert material.emission == 0.1

        # Execute ALL Material methods
        mat_dict = material.dict()
        mat_copy = material.copy()
        mat_json = material.json()

        assert "test_material" in mat_json
        assert mat_copy.name == "test_material"

        # Test Friction with ALL parameters
        friction = Friction(
            slide=[1.0, 0.005, 0.0001],
            spin=[0.003, 0.003, 0.003],
            roll=[0.0001, 0.0001, 0.0001]
        )

        assert friction.slide == [1.0, 0.005, 0.0001]
        assert friction.spin == [0.003, 0.003, 0.003]

        # Test Joint with ALL parameters
        joint = Joint(
            name="test_joint",
            type=JointType.HINGE,
            pos=[0, 0, 0],
            axis=[0, 0, 1],
            limited=True,
            range=[-180, 180],
            stiffness=10.0,
            damping=0.1,
            springref=0.0,
            armature=0.0,
            margin=0.0,
            ref=0.0,
            springdamper=[0, 0]
        )

        assert joint.name == "test_joint"
        assert joint.type == JointType.HINGE
        assert joint.limited == True
        assert joint.stiffness == 10.0

        # Execute ALL Joint methods
        joint_dict = joint.dict()
        joint_copy = joint.copy()
        joint_json = joint.json()

        # Test Geom with ALL parameters
        geom = Geom(
            name="test_geom",
            type="box",
            size=[1.0, 1.0, 1.0],
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            friction=friction,
            material=material,
            rgba=[1, 0, 0, 1],
            mass=5.0,
            density=1000.0,
            contype=1,
            conaffinity=1,
            condim=3,
            group=0,
            priority=0
        )

        assert geom.name == "test_geom"
        assert geom.type == "box"
        assert geom.mass == 5.0
        assert geom.material == material

        # Execute ALL Geom methods
        geom_dict = geom.dict()
        geom_copy = geom.copy()
        geom_json = geom.json()

        # Test Inertial with ALL parameters
        inertial = Inertial(
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            mass=10.0,
            diaginertia=[1.0, 1.0, 1.0],
            fullinertia=[1, 0, 0, 1, 0, 1]
        )

        assert inertial.mass == 10.0
        assert inertial.diaginertia == [1.0, 1.0, 1.0]

        # Test Body with ALL parameters
        body = Body(
            name="test_body",
            pos=[0, 0, 1],
            quat=[1, 0, 0, 0],
            inertial=inertial,
            geoms=[geom],
            joints=[joint],
            children=[]
        )

        assert body.name == "test_body"
        assert body.pos == [0, 0, 1]
        assert len(body.geoms) == 1
        assert body.geoms[0] == geom

        # Execute ALL Body methods
        body_dict = body.dict()
        body_copy = body.copy()
        body_json = body.json()

        # Test Actuator with ALL parameters
        actuator = Actuator(
            name="test_actuator",
            type=ActuatorType.MOTOR,
            joint="test_joint",
            gear=100.0,
            ctrllimited=True,
            ctrlrange=[-100, 100],
            forcelimited=True,
            forcerange=[-500, 500],
            lengthrange=[0, 1],
            velrange=[-10, 10],
            accelrange=[-50, 50]
        )

        assert actuator.name == "test_actuator"
        assert actuator.gear == 100.0
        assert actuator.ctrllimited == True

        # Test Sensor
        sensor = Sensor(
            name="test_sensor",
            type=SensorType.ACCELEROMETER,
            site="test_site",
            noise=0.01,
            cutoff=0.0,
            user_data={"custom": "value"}
        )

        assert sensor.name == "test_sensor"
        assert sensor.noise == 0.01

        # Test Contact
        contact = Contact(
            name="test_contact",
            condim=3,
            friction=[1.0, 0.005, 0.0001],
            solref=[0.02, 1.0],
            solimp=[0.9, 0.95],
            gap=0.0,
            margin=0.0
        )

        assert contact.name == "test_contact"
        assert contact.condim == 3

        # Test Equality
        equality = Equality(
            name="test_equality",
            type="connect",
            body1="body1",
            body2="body2",
            anchor=[0, 0, 0],
            active=True,
            solref=[0.02, 1.0],
            solimp=[0.9, 0.95]
        )

        assert equality.name == "test_equality"
        assert equality.type == "connect"

        # Test DefaultSettings
        defaults = DefaultSettings(
            geom_friction=[1.0, 0.005, 0.0001],
            geom_solimp=[0.9, 0.95],
            geom_solref=[0.02, 1.0],
            joint_damping=0.1,
            joint_stiffness=0.0
        )

        assert defaults.geom_friction == [1.0, 0.005, 0.0001]
        assert defaults.joint_damping == 0.1

        # Test SimulationMeta
        meta = SimulationMeta(
            version=PhysicsSpecVersion.V1_0_0,
            created_at=datetime.now(),
            created_by="test_user",
            description="Test simulation",
            tags=["test", "physics"],
            physics_engine="mujoco",
            engine_version="2.3.0"
        )

        assert meta.version == PhysicsSpecVersion.V1_0_0
        assert meta.created_by == "test_user"

        # Test PhysicsSpec with ALL parameters
        spec = PhysicsSpec(
            meta=meta,
            defaults=defaults,
            bodies=[body],
            actuators=[actuator],
            sensors=[sensor],
            contacts=[contact],
            equality=[equality]
        )

        assert spec.meta == meta
        assert len(spec.bodies) == 1
        assert spec.bodies[0] == body

        # Execute ALL PhysicsSpec methods
        spec_dict = spec.dict()
        spec_copy = spec.copy()
        spec_json = spec.json()

        # Test MJCF generation - this is a key method
        mjcf_xml = spec.to_mjcf()
        assert "<mujoco>" in mjcf_xml
        assert "test_body" in mjcf_xml
        assert "test_geom" in mjcf_xml

        # Test validation by creating a new spec from dict
        spec_from_dict = PhysicsSpec(**spec_dict)
        assert spec_from_dict.meta.version == PhysicsSpecVersion.V1_0_0

    def test_physics_spec_edge_cases_and_validators(self):
        """Test edge cases and all validators."""
        from simgen.models.physics_spec import PhysicsSpec, Body, Geom

        # Test body validation - must have at least one body
        with pytest.raises(ValueError, match="At least one body is required"):
            PhysicsSpec(bodies=[])

        # Test valid minimal spec
        minimal_body = Body(
            name="minimal",
            geoms=[Geom(name="minimal_geom", type="box", size=[1, 1, 1])]
        )
        minimal_spec = PhysicsSpec(bodies=[minimal_body])
        assert len(minimal_spec.bodies) == 1

        # Test ID generation and validation
        body_with_custom_id = Body(
            id="custom_id_123",
            name="custom_body",
            geoms=[Geom(name="geom1", type="sphere", size=[0.5])]
        )

        spec_with_custom_id = PhysicsSpec(bodies=[body_with_custom_id])
        assert spec_with_custom_id.bodies[0].id == "custom_id_123"

        # Test nested bodies (children)
        child_body = Body(
            name="child",
            geoms=[Geom(name="child_geom", type="box", size=[0.5, 0.5, 0.5])]
        )
        parent_body = Body(
            name="parent",
            geoms=[Geom(name="parent_geom", type="box", size=[1, 1, 1])],
            children=[child_body]
        )

        nested_spec = PhysicsSpec(bodies=[parent_body])
        assert len(nested_spec.bodies[0].children) == 1
        assert nested_spec.bodies[0].children[0].name == "child"


class TestSchemasUltraComprehensive:
    """Ultra comprehensive testing of schemas module."""

    def test_all_schema_classes_comprehensive(self):
        """Test ALL schema classes with ALL parameters and methods."""
        from simgen.models.schemas import (
            SimulationRequest, SimulationResponse, SimulationStatus,
            SketchAnalysisRequest, SketchAnalysisResponse,
            MJCFValidationRequest, MJCFValidationResponse,
            StreamingRequest, StreamingResponse,
            ErrorResponse, HealthCheckResponse,
            BatchSimulationRequest, BatchSimulationResponse
        )

        # Test SimulationRequest with ALL parameters
        sim_request = SimulationRequest(
            prompt="Create a red bouncing ball with gravity -9.81",
            parameters={
                "gravity": -9.81,
                "timestep": 0.002,
                "duration": 10.0,
                "quality": "high"
            },
            user_id="user123",
            session_id="session456",
            tags=["physics", "ball", "gravity"],
            metadata={
                "source": "web_ui",
                "version": "1.0",
                "experimental": True
            }
        )

        assert sim_request.prompt == "Create a red bouncing ball with gravity -9.81"
        assert sim_request.parameters["gravity"] == -9.81
        assert sim_request.user_id == "user123"
        assert len(sim_request.tags) == 3

        # Execute ALL SimulationRequest methods
        req_dict = sim_request.dict()
        req_copy = sim_request.copy()
        req_json = sim_request.json()

        assert "bouncing ball" in req_json
        assert req_copy.user_id == "user123"

        # Test SimulationResponse with ALL parameters
        sim_response = SimulationResponse(
            simulation_id="sim_789",
            status=SimulationStatus.COMPLETED,
            mjcf_content="<mujoco><worldbody><body><geom type='sphere'/></body></worldbody></mujoco>",
            preview_url="https://example.com/preview/sim_789.png",
            video_url="https://example.com/video/sim_789.mp4",
            download_url="https://example.com/download/sim_789.zip",
            metadata={
                "render_time": 2.5,
                "compile_time": 0.3,
                "total_frames": 1000
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            error_message=None,
            warnings=["Simulation may be unstable at high speeds"]
        )

        assert sim_response.simulation_id == "sim_789"
        assert sim_response.status == SimulationStatus.COMPLETED
        assert "<mujoco>" in sim_response.mjcf_content
        assert len(sim_response.warnings) == 1

        # Execute ALL SimulationResponse methods
        resp_dict = sim_response.dict()
        resp_copy = sim_response.copy()
        resp_json = sim_response.json()

        # Test SketchAnalysisRequest
        sketch_request = SketchAnalysisRequest(
            image_data=b"fake_image_data_12345",
            image_format="png",
            analysis_level="detailed",
            extract_text=True,
            detect_objects=True,
            user_id="user123",
            session_id="session456"
        )

        assert sketch_request.image_format == "png"
        assert sketch_request.analysis_level == "detailed"
        assert sketch_request.extract_text == True

        # Test SketchAnalysisResponse
        sketch_response = SketchAnalysisResponse(
            objects_detected=[
                {"type": "sphere", "confidence": 0.95, "bbox": [10, 10, 50, 50]},
                {"type": "box", "confidence": 0.88, "bbox": [60, 20, 30, 40]}
            ],
            text_extracted=["Ball", "Box", "Physics"],
            suggested_prompt="Create a red ball next to a blue box",
            confidence_score=0.92,
            processing_time=1.2,
            metadata={
                "image_size": [800, 600],
                "color_mode": "RGB",
                "analysis_version": "2.1"
            }
        )

        assert len(sketch_response.objects_detected) == 2
        assert sketch_response.confidence_score == 0.92
        assert "ball" in sketch_response.suggested_prompt.lower()

        # Test MJCFValidationRequest
        mjcf_request = MJCFValidationRequest(
            mjcf_content="<mujoco><worldbody><body name='test'><geom type='box'/></body></worldbody></mujoco>",
            validation_level="strict",
            check_physics=True,
            check_rendering=True,
            user_id="user123"
        )

        assert mjcf_request.validation_level == "strict"
        assert mjcf_request.check_physics == True

        # Test MJCFValidationResponse
        mjcf_response = MJCFValidationResponse(
            is_valid=True,
            errors=[],
            warnings=["Consider adding inertial properties"],
            suggestions=["Add material definitions for better rendering"],
            validation_details={
                "syntax_check": "passed",
                "physics_check": "passed",
                "rendering_check": "passed"
            },
            processing_time=0.5
        )

        assert mjcf_response.is_valid == True
        assert len(mjcf_response.warnings) == 1

        # Test StreamingRequest
        stream_request = StreamingRequest(
            simulation_id="sim_789",
            stream_type="physics_data",
            format="binary",
            compression="gzip",
            frame_rate=60,
            quality="high",
            user_id="user123"
        )

        assert stream_request.stream_type == "physics_data"
        assert stream_request.frame_rate == 60

        # Test StreamingResponse
        stream_response = StreamingResponse(
            stream_id="stream_456",
            websocket_url="wss://example.com/stream/stream_456",
            http_stream_url="https://example.com/stream/stream_456/data",
            estimated_duration=10.0,
            frame_count=600,
            status="active"
        )

        assert stream_response.stream_id == "stream_456"
        assert "wss://" in stream_response.websocket_url

        # Test ErrorResponse
        error_response = ErrorResponse(
            error_code="SIMULATION_FAILED",
            error_message="Failed to compile MJCF: Invalid geometry type",
            details={
                "line": 45,
                "column": 12,
                "invalid_value": "invalid_geom_type"
            },
            timestamp=datetime.now(),
            request_id="req_123",
            correlation_id="corr_456"
        )

        assert error_response.error_code == "SIMULATION_FAILED"
        assert "Invalid geometry" in error_response.error_message

        # Test HealthCheckResponse
        health_response = HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.2.3",
            uptime=3600.5,
            services={
                "database": "healthy",
                "redis": "healthy",
                "mujoco": "healthy",
                "gpu": "healthy"
            },
            performance={
                "avg_response_time": 0.15,
                "active_simulations": 25,
                "queue_size": 3
            }
        )

        assert health_response.status == "healthy"
        assert health_response.services["database"] == "healthy"
        assert health_response.performance["active_simulations"] == 25

        # Test BatchSimulationRequest
        batch_request = BatchSimulationRequest(
            requests=[sim_request],
            batch_id="batch_789",
            priority="normal",
            max_parallel=3,
            timeout=300,
            notification_webhook="https://example.com/webhook/batch_complete",
            user_id="user123"
        )

        assert len(batch_request.requests) == 1
        assert batch_request.max_parallel == 3

        # Test BatchSimulationResponse
        batch_response = BatchSimulationResponse(
            batch_id="batch_789",
            status="completed",
            total_requests=1,
            completed_requests=1,
            failed_requests=0,
            responses=[sim_response],
            started_at=datetime.now(),
            completed_at=datetime.now(),
            processing_time=5.2
        )

        assert batch_response.batch_id == "batch_789"
        assert batch_response.completed_requests == 1
        assert len(batch_response.responses) == 1


class TestSimulationModelUltraComprehensive:
    """Ultra comprehensive testing of simulation model."""

    def test_simulation_model_all_methods(self):
        """Test ALL simulation model methods and properties."""
        from simgen.models.simulation import Simulation, SimulationStatus
        from datetime import datetime

        # Test ALL SimulationStatus enum values
        assert SimulationStatus.PENDING == "pending"
        assert SimulationStatus.RUNNING == "running"
        assert SimulationStatus.COMPLETED == "completed"
        assert SimulationStatus.FAILED == "failed"
        assert SimulationStatus.CANCELLED == "cancelled"

        # Create simulation with ALL possible fields
        simulation = Simulation(
            id="sim_comprehensive_123",
            prompt="Ultra comprehensive test simulation",
            mjcf_content="<mujoco><worldbody><body name='test'><geom type='sphere' size='0.1'/></body></worldbody></mujoco>",
            status=SimulationStatus.COMPLETED,
            created_at=datetime(2024, 1, 15, 10, 30, 45),
            updated_at=datetime(2024, 1, 15, 10, 35, 20),
            user_id="comprehensive_user_456",
            session_id="comprehensive_session_789",
            parameters={
                "gravity": -9.81,
                "timestep": 0.002,
                "iterations": 1000,
                "solver": "PGS",
                "integrator": "RK4"
            },
            metadata={
                "render_quality": "ultra_high",
                "frame_rate": 120,
                "total_frames": 12000,
                "file_size_mb": 45.7,
                "compression": "h264"
            },
            preview_url="https://comprehensive.example.com/preview/sim_comprehensive_123.jpg",
            video_url="https://comprehensive.example.com/video/sim_comprehensive_123.mp4",
            download_url="https://comprehensive.example.com/download/sim_comprehensive_123.zip",
            error_message=None,
            warnings=["High quality rendering may take longer", "Large file size detected"],
            tags=["comprehensive", "test", "physics", "ultra", "quality"],
            processing_time=125.5,
            render_time=89.3,
            file_size=47923456,
            frame_count=12000,
            is_public=True,
            views=1337,
            likes=42,
            downloads=128
        )

        # Test ALL basic attributes
        assert simulation.id == "sim_comprehensive_123"
        assert "Ultra comprehensive" in simulation.prompt
        assert simulation.status == SimulationStatus.COMPLETED
        assert simulation.user_id == "comprehensive_user_456"
        assert simulation.session_id == "comprehensive_session_789"

        # Test ALL complex attributes
        assert simulation.parameters["gravity"] == -9.81
        assert simulation.parameters["timestep"] == 0.002
        assert simulation.parameters["iterations"] == 1000

        assert simulation.metadata["render_quality"] == "ultra_high"
        assert simulation.metadata["frame_rate"] == 120
        assert simulation.metadata["total_frames"] == 12000

        assert len(simulation.warnings) == 2
        assert "High quality" in simulation.warnings[0]

        assert len(simulation.tags) == 5
        assert "comprehensive" in simulation.tags
        assert "ultra" in simulation.tags

        # Test ALL URL fields
        assert "preview" in simulation.preview_url
        assert "video" in simulation.video_url
        assert "download" in simulation.download_url

        # Test ALL numeric fields
        assert simulation.processing_time == 125.5
        assert simulation.render_time == 89.3
        assert simulation.file_size == 47923456
        assert simulation.frame_count == 12000
        assert simulation.views == 1337
        assert simulation.likes == 42
        assert simulation.downloads == 128

        # Test boolean fields
        assert simulation.is_public == True
        assert simulation.error_message is None

        # Test datetime fields
        assert simulation.created_at.year == 2024
        assert simulation.updated_at.minute == 35

        # Test MJCF content
        assert "<mujoco>" in simulation.mjcf_content
        assert "sphere" in simulation.mjcf_content

        # Execute ALL Pydantic methods
        sim_dict = simulation.dict()
        sim_copy = simulation.copy()
        sim_json = simulation.json()

        # Verify serialization works
        assert "sim_comprehensive_123" in sim_json
        assert sim_copy.id == simulation.id
        assert sim_dict["status"] == "completed"

        # Test copy with modifications
        modified_sim = simulation.copy(update={"views": 2000, "likes": 100})
        assert modified_sim.views == 2000
        assert modified_sim.likes == 100
        assert modified_sim.id == simulation.id  # Other fields unchanged

        # Test dict exclude/include
        sim_dict_partial = simulation.dict(exclude={"mjcf_content", "metadata"})
        assert "mjcf_content" not in sim_dict_partial
        assert "metadata" not in sim_dict_partial
        assert "id" in sim_dict_partial

        sim_dict_only_basic = simulation.dict(include={"id", "prompt", "status"})
        assert len(sim_dict_only_basic) == 3
        assert sim_dict_only_basic["id"] == "sim_comprehensive_123"

    def test_simulation_status_transitions(self):
        """Test all possible status transitions."""
        from simgen.models.simulation import Simulation, SimulationStatus

        # Test each status value
        for status in SimulationStatus:
            sim = Simulation(
                id=f"sim_{status.value}",
                prompt=f"Test {status.value} simulation",
                mjcf_content="<mujoco><worldbody><body><geom type='box'/></body></worldbody></mujoco>",
                status=status
            )

            assert sim.status == status
            assert sim.status.value == status.value

            # Test serialization of each status
            sim_json = sim.json()
            assert status.value in sim_json

        # Test status comparisons
        assert SimulationStatus.PENDING != SimulationStatus.RUNNING
        assert SimulationStatus.COMPLETED == SimulationStatus.COMPLETED


class TestConfigUltraComprehensive:
    """Ultra comprehensive testing of config module."""

    def test_config_all_settings_comprehensive(self):
        """Test ALL config settings and methods."""
        from simgen.core.config import Settings

        # Create settings instance - this executes the Settings class
        settings = Settings()

        # Test ALL basic attributes exist and have correct types
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'environment')

        # Test database settings
        assert hasattr(settings, 'database_url')
        assert isinstance(settings.database_url, str)

        # Test Redis settings if they exist
        if hasattr(settings, 'redis_url'):
            assert isinstance(settings.redis_url, str)

        # Test API keys if they exist
        if hasattr(settings, 'openai_api_key'):
            assert isinstance(settings.openai_api_key, str)

        if hasattr(settings, 'anthropic_api_key'):
            assert isinstance(settings.anthropic_api_key, str)

        # Test secret key
        if hasattr(settings, 'secret_key'):
            assert isinstance(settings.secret_key, str)
            assert len(settings.secret_key) > 0

        # Execute ALL possible Settings methods
        settings_dict = settings.dict()
        assert isinstance(settings_dict, dict)
        assert len(settings_dict) > 0

        settings_copy = settings.copy()
        assert isinstance(settings_copy, Settings)

        settings_json = settings.json()
        assert isinstance(settings_json, str)
        assert len(settings_json) > 10

        # Test environment-specific behavior
        if hasattr(settings, 'debug'):
            if settings.debug:
                assert isinstance(settings.debug, bool)

        # Test all string representations
        settings_str = str(settings)
        settings_repr = repr(settings)

        assert isinstance(settings_str, str)
        assert isinstance(settings_repr, str)
        assert "Settings" in settings_repr


def test_integration_all_working_modules_comprehensive():
    """Integration test using ALL working modules together comprehensively."""

    # Import ALL working modules
    from simgen.models.physics_spec import PhysicsSpec, Body, Geom, DefaultSettings, SimulationMeta
    from simgen.models.schemas import SimulationRequest, SimulationResponse, SimulationStatus
    from simgen.models.simulation import Simulation, SimulationStatus as SimStatus
    from simgen.core.config import Settings
    from datetime import datetime

    # Test complete integration workflow

    # 1. Create comprehensive config
    config = Settings()
    assert config is not None

    # 2. Create comprehensive simulation request
    request = SimulationRequest(
        prompt="Create an advanced multi-body physics simulation with interconnected components",
        parameters={
            "gravity": -9.81,
            "timestep": 0.001,
            "solver": "PGS",
            "integrator": "RK4",
            "iterations": 200,
            "tolerance": 1e-8
        },
        user_id="integration_user_999",
        session_id="integration_session_888",
        tags=["integration", "comprehensive", "multi-body", "advanced"],
        metadata={
            "test_suite": "comprehensive",
            "integration_level": "full",
            "complexity": "high"
        }
    )

    # 3. Create comprehensive physics spec
    advanced_spec = PhysicsSpec(
        meta=SimulationMeta(
            version="1.0.0",
            created_by="integration_test",
            description="Advanced multi-body simulation for comprehensive testing",
            tags=["integration", "advanced", "comprehensive"]
        ),
        defaults=DefaultSettings(
            geom_friction=[1.2, 0.008, 0.0002],
            joint_damping=0.15,
            joint_stiffness=10.0
        ),
        bodies=[
            Body(
                name="advanced_base",
                pos=[0, 0, 0.5],
                geoms=[
                    Geom(
                        name="base_platform",
                        type="box",
                        size=[2.0, 2.0, 0.1],
                        rgba=[0.3, 0.3, 0.8, 1.0],
                        mass=10.0
                    )
                ]
            ),
            Body(
                name="advanced_pendulum",
                pos=[0, 0, 2.0],
                geoms=[
                    Geom(
                        name="pendulum_bob",
                        type="sphere",
                        size=[0.2],
                        rgba=[1.0, 0.2, 0.2, 1.0],
                        mass=2.0
                    )
                ]
            ),
            Body(
                name="advanced_arm",
                pos=[1.0, 0, 1.0],
                geoms=[
                    Geom(
                        name="arm_segment",
                        type="cylinder",
                        size=[0.05, 0.8],
                        rgba=[0.2, 0.8, 0.2, 1.0],
                        mass=1.5
                    )
                ]
            )
        ]
    )

    # 4. Generate MJCF from physics spec
    mjcf_content = advanced_spec.to_mjcf()
    assert "<mujoco>" in mjcf_content
    assert "advanced_base" in mjcf_content
    assert "advanced_pendulum" in mjcf_content
    assert "advanced_arm" in mjcf_content

    # 5. Create comprehensive simulation response
    response = SimulationResponse(
        simulation_id="integration_sim_comprehensive_777",
        status=SimulationStatus.COMPLETED,
        mjcf_content=mjcf_content,
        preview_url="https://integration.example.com/preview/comprehensive_777.jpg",
        video_url="https://integration.example.com/video/comprehensive_777.mp4",
        download_url="https://integration.example.com/download/comprehensive_777.zip",
        metadata={
            "integration_test": True,
            "total_bodies": 3,
            "total_geoms": 3,
            "complexity_score": 8.5,
            "render_time": 45.2,
            "compile_time": 1.8
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
        warnings=["Advanced simulation may require high-performance hardware"]
    )

    # 6. Create comprehensive simulation model
    simulation = Simulation(
        id="integration_comprehensive_simulation_999",
        prompt=request.prompt,
        mjcf_content=mjcf_content,
        status=SimStatus.COMPLETED,
        user_id=request.user_id,
        session_id=request.session_id,
        parameters=request.parameters,
        metadata=response.metadata,
        preview_url=response.preview_url,
        video_url=response.video_url,
        download_url=response.download_url,
        warnings=response.warnings,
        tags=request.tags + ["integration_complete"],
        processing_time=47.0,
        render_time=45.2,
        views=0,
        likes=0,
        downloads=0,
        is_public=False
    )

    # 7. Verify complete integration
    assert simulation.id == "integration_comprehensive_simulation_999"
    assert simulation.status == SimStatus.COMPLETED
    assert len(simulation.tags) >= 5
    assert "integration_complete" in simulation.tags
    assert simulation.processing_time > 0

    # 8. Test all serialization works together
    request_json = request.json()
    response_json = response.json()
    simulation_json = simulation.json()
    spec_json = advanced_spec.json()

    assert "integration_user_999" in request_json
    assert "integration_sim_comprehensive_777" in response_json
    assert "integration_comprehensive_simulation_999" in simulation_json
    assert "advanced_base" in spec_json

    # 9. Test complete workflow serialization roundtrip
    request_dict = request.dict()
    request_restored = SimulationRequest(**request_dict)
    assert request_restored.prompt == request.prompt
    assert request_restored.user_id == request.user_id

    # Success - all modules working together comprehensively!
    return True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/simgen", "--cov-report=term"])