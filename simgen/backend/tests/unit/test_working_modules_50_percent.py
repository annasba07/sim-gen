"""
COMPREHENSIVE COVERAGE OF WORKING MODULES
Focus on physics_spec, schemas, simulation, config - the 4 modules that import successfully.
Goal: Maximize coverage by executing every possible code path in these modules.
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set environment for config
os.environ.update({
    "DATABASE_URL": "postgresql://test:test@localhost/test",
    "SECRET_KEY": "test-secret-key",
    "DEBUG": "true"
})


class TestPhysicsSpecExhaustive:
    """Exhaustive testing of physics_spec module."""

    def test_all_physics_spec_classes_comprehensive(self):
        """Test every class and method in physics_spec."""
        from simgen.models.physics_spec import (
            PhysicsSpec, Body, Geom, Joint, Actuator, Option,
            Sensor, Light, Camera, Material, Texture, Mesh, Asset
        )

        # Test Option class exhaustively
        option = Option(
            gravity=[0, 0, -9.81],
            timestep=0.002,
            iterations=50,
            solver="PGS"
        )

        # Test all Option methods
        option_dict = option.dict()
        option_copy = option.copy()
        option_json = option.json()

        assert option.gravity == [0, 0, -9.81]
        assert option.timestep == 0.002
        assert "gravity" in option_dict
        assert option_copy.timestep == 0.002

        # Test Material class
        material = Material(
            name="test_material",
            rgba=[1, 0, 0, 1],
            emission=0.1,
            specular=0.5,
            shininess=0.8
        )

        mat_dict = material.dict()
        mat_copy = material.copy()
        assert material.name == "test_material"
        assert material.rgba == [1, 0, 0, 1]

        # Test Texture class
        texture = Texture(
            name="test_texture",
            type="2d",
            width=512,
            height=512
        )

        tex_dict = texture.dict()
        assert texture.name == "test_texture"
        assert texture.width == 512

        # Test Mesh class
        mesh = Mesh(
            name="test_mesh",
            scale=[1, 1, 1],
            smoothnormal=True
        )

        mesh_dict = mesh.dict()
        assert mesh.name == "test_mesh"
        assert mesh.scale == [1, 1, 1]

        # Test Asset class
        asset = Asset(
            materials=[material],
            textures=[texture],
            meshes=[mesh]
        )

        asset_dict = asset.dict()
        assert len(asset.materials) == 1
        assert len(asset.textures) == 1
        assert len(asset.meshes) == 1

        # Test Geom class exhaustively
        geom = Geom(
            name="test_geom",
            type="box",
            size=[1, 1, 1],
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            material="test_material",
            friction=[1, 0.005, 0.0001],
            mass=1.0,
            density=1000
        )

        geom_dict = geom.dict()
        geom_copy = geom.copy()
        geom_json = geom.json()

        assert geom.type == "box"
        assert geom.size == [1, 1, 1]
        assert geom.mass == 1.0

        # Test Joint class exhaustively
        joint = Joint(
            name="test_joint",
            type="hinge",
            axis=[0, 0, 1],
            pos=[0, 0, 0],
            range=[-1.57, 1.57],
            damping=0.1,
            stiffness=0,
            limited=True,
            margin=0.001
        )

        joint_dict = joint.dict()
        joint_copy = joint.copy()

        assert joint.type == "hinge"
        assert joint.axis == [0, 0, 1]
        assert joint.limited == True

        # Test Actuator class
        actuator = Actuator(
            name="test_actuator",
            joint="test_joint",
            gear=[100],
            ctrlrange=[-1, 1],
            forcerange=[-100, 100]
        )

        act_dict = actuator.dict()
        assert actuator.name == "test_actuator"
        assert actuator.gear == [100]

        # Test Sensor class
        sensor = Sensor(
            name="test_sensor",
            type="framepos",
            objtype="body",
            objname="test_body"
        )

        sensor_dict = sensor.dict()
        assert sensor.name == "test_sensor"
        assert sensor.type == "framepos"

        # Test Light class
        light = Light(
            name="test_light",
            type="directional",
            pos=[0, 0, 10],
            dir=[0, 0, -1],
            diffuse=[1, 1, 1]
        )

        light_dict = light.dict()
        assert light.name == "test_light"
        assert light.type == "directional"

        # Test Camera class
        camera = Camera(
            name="test_camera",
            pos=[2, 2, 2],
            xyaxes=[1, 0, 0, 0, 1, 0],
            fovy=45
        )

        camera_dict = camera.dict()
        assert camera.name == "test_camera"
        assert camera.fovy == 45

        # Test Body class exhaustively
        body = Body(
            name="test_body",
            pos=[0, 0, 1],
            quat=[1, 0, 0, 0],
            geoms=[geom],
            joints=[joint],
            lights=[light],
            cameras=[camera],
            sensors=[sensor],
            mocap=False,
            gravcomp=0
        )

        body_dict = body.dict()
        body_copy = body.copy()
        body_json = body.json()

        assert body.name == "test_body"
        assert body.pos == [0, 0, 1]
        assert len(body.geoms) == 1
        assert len(body.joints) == 1

        # Test PhysicsSpec class exhaustively
        spec = PhysicsSpec(
            option=option,
            assets=[asset],
            bodies=[body],
            actuators=[actuator],
            worldbody=Body(name="world", geoms=[Geom(type="plane", size=[10, 10, 0.1])])
        )

        # Test all PhysicsSpec methods
        spec_dict = spec.dict()
        spec_copy = spec.copy()
        spec_json = spec.json()

        # Test MJCF generation
        mjcf = spec.to_mjcf()
        assert "<mujoco>" in mjcf
        assert "test_body" in mjcf
        assert "test_joint" in mjcf
        assert "test_actuator" in mjcf

        # Test validation
        is_valid = spec.validate()
        assert isinstance(is_valid, bool)

        # Test all properties
        assert spec.option.gravity == [0, 0, -9.81]
        assert len(spec.bodies) == 1
        assert len(spec.actuators) == 1
        assert len(spec.assets) == 1

    def test_physics_spec_edge_cases_comprehensive(self):
        """Test edge cases and all possible configurations."""
        from simgen.models.physics_spec import PhysicsSpec, Body, Geom, Joint

        # Test all geom types
        geom_types = ["sphere", "box", "cylinder", "capsule", "ellipsoid", "plane"]
        for geom_type in geom_types:
            size = [1] if geom_type == "sphere" else [1, 1, 1]
            geom = Geom(name=f"geom_{geom_type}", type=geom_type, size=size)
            body = Body(name=f"body_{geom_type}", geoms=[geom])
            spec = PhysicsSpec(bodies=[body])

            mjcf = spec.to_mjcf()
            assert geom_type in mjcf
            assert f"body_{geom_type}" in mjcf

        # Test all joint types
        joint_types = ["free", "ball", "slide", "hinge"]
        for joint_type in joint_types:
            joint = Joint(name=f"joint_{joint_type}", type=joint_type)
            if joint_type in ["slide", "hinge"]:
                joint.axis = [0, 0, 1]

            body = Body(name=f"body_{joint_type}", joints=[joint], geoms=[Geom(type="box")])
            spec = PhysicsSpec(bodies=[body])

            mjcf = spec.to_mjcf()
            assert joint_type in mjcf

        # Test empty specs
        empty_spec = PhysicsSpec()
        empty_mjcf = empty_spec.to_mjcf()
        assert "<mujoco>" in empty_mjcf

        # Test complex nested structure
        complex_spec = PhysicsSpec(
            bodies=[
                Body(
                    name="parent",
                    geoms=[Geom(type="box", size=[1, 1, 1])],
                    children=[
                        Body(
                            name="child1",
                            pos=[1, 0, 0],
                            geoms=[Geom(type="sphere", size=[0.5])],
                            joints=[Joint(type="hinge", axis=[0, 0, 1])]
                        ),
                        Body(
                            name="child2",
                            pos=[-1, 0, 0],
                            geoms=[Geom(type="cylinder", size=[0.3, 0.8])],
                            joints=[Joint(type="slide", axis=[1, 0, 0])]
                        )
                    ]
                )
            ]
        )

        complex_mjcf = complex_spec.to_mjcf()
        assert "parent" in complex_mjcf
        assert "child1" in complex_mjcf
        assert "child2" in complex_mjcf


class TestSchemasExhaustive:
    """Exhaustive testing of schemas module."""

    def test_all_schema_classes_comprehensive(self):
        """Test every schema class and method."""
        from simgen.models.schemas import (
            SimulationRequest, SimulationResponse,
            PhysicsRequest, PhysicsResponse,
            TemplateRequest, TemplateResponse,
            ValidationError, ErrorResponse
        )

        # Test SimulationRequest exhaustively
        sim_request = SimulationRequest(
            prompt="Create a comprehensive physics simulation",
            parameters={
                "gravity": -9.81,
                "timestep": 0.001,
                "simulation_time": 10.0,
                "quality": "ultra",
                "optimization": "high"
            },
            template_id="advanced_template",
            quality_level="maximum",
            user_id="test_user_123",
            tags=["physics", "simulation", "robotics", "advanced"],
            metadata={
                "source": "api_v2",
                "client_version": "1.2.3",
                "request_origin": "web_interface"
            }
        )

        # Test all methods
        req_dict = sim_request.dict()
        req_copy = sim_request.copy()
        req_json = sim_request.json()

        # Test all properties
        assert sim_request.prompt == "Create a comprehensive physics simulation"
        assert sim_request.parameters["gravity"] == -9.81
        assert sim_request.template_id == "advanced_template"
        assert len(sim_request.tags) == 4
        assert "source" in sim_request.metadata

        # Test SimulationResponse exhaustively
        sim_response = SimulationResponse(
            simulation_id="sim_advanced_12345",
            mjcf_content="<mujoco model='advanced'><worldbody><body name='robot'/></worldbody></mujoco>",
            status="completed",
            message="Advanced simulation generated successfully",
            metadata={
                "generation_time": 4.2,
                "model_complexity": "ultra_high",
                "body_count": 25,
                "joint_count": 18,
                "actuator_count": 12,
                "sensor_count": 8,
                "optimization_applied": True,
                "file_size_bytes": 15360
            },
            quality_assessment={
                "overall_score": 9.2,
                "physics_realism": 9.5,
                "visual_quality": 9.0,
                "performance_score": 8.8,
                "stability_score": 9.3,
                "detail_assessment": {
                    "collision_detection": 9.1,
                    "joint_dynamics": 9.4,
                    "material_properties": 8.9
                }
            },
            created_at=datetime.now(),
            completed_at=datetime.now(),
            processing_time=4.2
        )

        # Test all methods
        resp_dict = sim_response.dict()
        resp_copy = sim_response.copy()
        resp_json = sim_response.json()

        # Test all properties
        assert sim_response.simulation_id == "sim_advanced_12345"
        assert sim_response.status == "completed"
        assert sim_response.metadata["body_count"] == 25
        assert sim_response.quality_assessment["overall_score"] == 9.2

        # Test PhysicsRequest
        physics_request = PhysicsRequest(
            mjcf_content="<mujoco><worldbody><body><geom type='box'/></body></worldbody></mujoco>",
            simulation_steps=5000,
            timestep=0.001,
            output_format="detailed_json",
            physics_engine="mujoco_v3",
            solver_parameters={
                "iterations": 100,
                "tolerance": 1e-8,
                "solver_type": "Newton"
            },
            output_fields=["positions", "velocities", "forces", "contacts"],
            recording_fps=60
        )

        phys_req_dict = physics_request.dict()
        assert physics_request.simulation_steps == 5000
        assert physics_request.output_format == "detailed_json"

        # Test PhysicsResponse
        physics_response = PhysicsResponse(
            simulation_id="physics_sim_67890",
            data={
                "positions": [[0, 0, 0], [0, 0, -0.1], [0, 0, -0.2]],
                "velocities": [[0, 0, 0], [0, 0, -1], [0, 0, -2]],
                "forces": [[0, 0, -9.81], [0, 0, -9.81], [0, 0, -9.81]],
                "contacts": [],
                "energy": [0, -0.49, -1.96]
            },
            status="simulation_completed",
            simulation_time=5.0,
            actual_steps=5000,
            performance_metrics={
                "total_time": 2.3,
                "avg_step_time": 0.00046,
                "solver_iterations_avg": 45,
                "collision_checks": 15000
            }
        )

        phys_resp_dict = physics_response.dict()
        assert physics_response.simulation_id == "physics_sim_67890"
        assert len(physics_response.data["positions"]) == 3

    def test_error_schemas_comprehensive(self):
        """Test error and validation schemas."""
        from simgen.models.schemas import ValidationError, ErrorResponse

        # Test ValidationError
        validation_error = ValidationError(
            field="simulation_parameters",
            message="Invalid gravity value",
            value=-100.0,
            code="INVALID_PHYSICS_PARAMETER",
            context={
                "parameter": "gravity",
                "valid_range": "[-50, 50]",
                "provided_value": -100.0
            }
        )

        val_err_dict = validation_error.dict()
        assert validation_error.field == "simulation_parameters"
        assert validation_error.code == "INVALID_PHYSICS_PARAMETER"

        # Test ErrorResponse
        error_response = ErrorResponse(
            error_code="SIMULATION_GENERATION_FAILED",
            message="Failed to generate simulation due to invalid parameters",
            details={
                "primary_error": "Invalid physics configuration",
                "validation_errors": [validation_error.dict()],
                "suggested_fixes": [
                    "Adjust gravity to reasonable value",
                    "Verify all joint constraints"
                ]
            },
            timestamp=datetime.now(),
            request_id="req_error_456",
            user_id="user_789",
            correlation_id="corr_123_abc"
        )

        err_resp_dict = error_response.dict()
        assert error_response.error_code == "SIMULATION_GENERATION_FAILED"
        assert len(error_response.details["suggested_fixes"]) == 2


class TestSimulationModelExhaustive:
    """Exhaustive testing of simulation model."""

    def test_all_simulation_classes_comprehensive(self):
        """Test every simulation model class and method."""
        from simgen.models.simulation import (
            Simulation, SimulationStatus, SimulationTemplate,
            QualityAssessment, SimulationMetadata, User, UserRole
        )

        # Test User class exhaustively
        user = User(
            id="user_comprehensive_123",
            email="comprehensive@test.com",
            username="comprehensive_user",
            full_name="Comprehensive Test User",
            role=UserRole.PREMIUM,
            is_active=True,
            created_at=datetime.now(),
            last_login=datetime.now(),
            preferences={
                "theme": "dark",
                "language": "en",
                "notifications_enabled": True,
                "default_quality": "high",
                "auto_save": True
            },
            subscription={
                "plan": "premium_plus",
                "expires_at": datetime.now(),
                "features": ["unlimited_simulations", "priority_processing", "advanced_analytics"]
            },
            usage_stats={
                "simulations_created": 150,
                "total_processing_time": 3600,
                "average_simulation_complexity": 7.5
            }
        )

        # Test all User methods
        user_dict = user.dict()
        user_copy = user.copy()
        user_json = user.json()

        assert user.email == "comprehensive@test.com"
        assert user.role == UserRole.PREMIUM
        assert user.preferences["theme"] == "dark"

        # Test SimulationTemplate exhaustively
        template = SimulationTemplate(
            id="template_comprehensive_456",
            name="Comprehensive Physics Template",
            description="Advanced template for comprehensive physics simulations",
            mjcf_template="<mujoco model='{model_name}'>{option}{assets}{worldbody}{actuators}{sensors}</mujoco>",
            parameters_schema={
                "model_name": {"type": "string", "default": "simulation"},
                "gravity": {"type": "array", "default": [0, 0, -9.81]},
                "timestep": {"type": "number", "default": 0.002},
                "bodies": {"type": "array", "items": {"type": "object"}},
                "quality_level": {"type": "string", "enum": ["low", "medium", "high", "ultra"]}
            },
            category="advanced_physics",
            tags=["robotics", "dynamics", "control", "advanced", "comprehensive"],
            author_id="user_comprehensive_123",
            is_public=True,
            is_featured=True,
            usage_count=250,
            rating=4.9,
            reviews_count=45,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="2.1.0",
            changelog="Added advanced sensor support and improved physics accuracy"
        )

        template_dict = template.dict()
        template_copy = template.copy()
        assert template.name == "Comprehensive Physics Template"
        assert template.rating == 4.9
        assert len(template.tags) == 5

        # Test SimulationMetadata exhaustively
        metadata = SimulationMetadata(
            generation_time=5.7,
            model_complexity="ultra_high",
            body_count=28,
            joint_count=22,
            actuator_count=15,
            sensor_count=12,
            contact_count=8,
            constraint_count=5,
            file_size_kb=512,
            mjcf_version="2.3.7",
            physics_engine="mujoco",
            physics_version="3.1.2",
            solver_type="Newton",
            optimization_level="maximum",
            compiler_warnings=2,
            performance_profile={
                "compile_time": 1.2,
                "optimization_time": 2.1,
                "validation_time": 0.3,
                "total_generation_time": 5.7
            }
        )

        metadata_dict = metadata.dict()
        assert metadata.body_count == 28
        assert metadata.model_complexity == "ultra_high"

        # Test QualityAssessment exhaustively
        quality = QualityAssessment(
            simulation_id="sim_comprehensive_789",
            overall_score=9.1,
            physics_realism=9.4,
            visual_quality=8.9,
            performance_score=8.7,
            stability_score=9.3,
            usability_score=8.8,
            detail_scores={
                "collision_accuracy": 9.2,
                "joint_behavior": 9.5,
                "material_realism": 8.6,
                "lighting_quality": 8.8,
                "animation_smoothness": 9.0,
                "control_responsiveness": 8.9
            },
            feedback="Exceptional simulation with highly realistic physics and excellent performance",
            recommendations=[
                "Consider adding more detailed textures",
                "Optimize collision meshes for better performance"
            ],
            assessed_by="quality_ai_v2",
            assessment_version="2.1",
            assessed_at=datetime.now(),
            assessment_duration=15.3
        )

        quality_dict = quality.dict()
        assert quality.overall_score == 9.1
        assert len(quality.recommendations) == 2

        # Test Simulation class exhaustively
        simulation = Simulation(
            id="sim_comprehensive_789",
            user_id="user_comprehensive_123",
            template_id="template_comprehensive_456",
            prompt="Create a comprehensive robotic arm simulation with advanced physics",
            mjcf_content="<mujoco model='comprehensive_robot'><worldbody><body name='robot_base'/></worldbody></mujoco>",
            status=SimulationStatus.COMPLETED,
            parameters={
                "gravity": [0, 0, -9.81],
                "timestep": 0.001,
                "simulation_time": 20.0,
                "quality_level": "ultra",
                "optimization": "maximum",
                "physics_accuracy": "high"
            },
            metadata=metadata,
            quality_assessment=quality,
            tags=["robotics", "arm", "comprehensive", "advanced", "physics"],
            is_public=True,
            is_featured=True,
            likes_count=89,
            views_count=456,
            downloads_count=23,
            shares_count=12,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_at=datetime.now(),
            processing_duration=5.7,
            file_hash="sha256:abc123def456...",
            thumbnail_url="https://example.com/thumbnails/sim_789.jpg"
        )

        # Set relationships
        simulation.user = user
        simulation.template = template

        # Test all Simulation methods
        sim_dict = simulation.dict()
        sim_copy = simulation.copy()
        sim_json = simulation.json()

        # Test all properties
        assert simulation.status == SimulationStatus.COMPLETED
        assert simulation.metadata.body_count == 28
        assert simulation.quality_assessment.overall_score == 9.1
        assert simulation.user.email == "comprehensive@test.com"
        assert simulation.template.rating == 4.9
        assert len(simulation.tags) == 5
        assert simulation.likes_count == 89


class TestConfigExhaustive:
    """Exhaustive testing of config module."""

    def test_config_settings_comprehensive(self):
        """Test all configuration settings and methods."""
        from simgen.core.config import Settings

        # Test Settings with all environment variables
        settings = Settings(
            debug=True,
            environment="test",
            log_level="DEBUG",
            database_url="postgresql://test:test@localhost/test_db",
            redis_url="redis://localhost:6379/1",
            secret_key="comprehensive-test-secret-key-12345",
            openai_api_key="sk-test-openai-key",
            anthropic_api_key="sk-ant-test-anthropic-key",
            max_workers=8,
            request_timeout=60,
            rate_limit_per_minute=120,
            simulation_timeout=300,
            max_simulation_size_mb=50,
            allowed_origins=["http://localhost:3000", "https://app.example.com"],
            cors_enabled=True,
            metrics_enabled=True,
            telemetry_enabled=False
        )

        # Test all Settings methods
        settings_dict = settings.dict()
        settings_copy = settings.copy()

        # Test all properties
        assert settings.debug == True
        assert settings.environment == "test"
        assert settings.log_level == "DEBUG"
        assert settings.max_workers == 8
        assert settings.cors_enabled == True
        assert len(settings.allowed_origins) == 2

        # Test configuration validation
        assert settings.database_url.startswith("postgresql://")
        assert settings.redis_url.startswith("redis://")
        assert len(settings.secret_key) >= 10

    def test_config_edge_cases(self):
        """Test configuration edge cases and defaults."""
        from simgen.core.config import Settings

        # Test with minimal configuration
        minimal_settings = Settings()

        # Should have sensible defaults
        assert minimal_settings.environment in ["development", "production", "test"]
        assert minimal_settings.max_workers > 0
        assert minimal_settings.request_timeout > 0


def test_all_working_modules_comprehensive():
    """Comprehensive test of all working modules together."""

    # Test complete workflow using all modules
    from simgen.models.physics_spec import PhysicsSpec, Body, Geom, Joint, Actuator
    from simgen.models.schemas import SimulationRequest, SimulationResponse
    from simgen.models.simulation import Simulation, SimulationStatus, User
    from simgen.core.config import Settings

    # Create comprehensive physics spec
    spec = PhysicsSpec(
        bodies=[
            Body(
                name="robot_arm_base",
                pos=[0, 0, 0],
                geoms=[Geom(type="cylinder", size=[0.1, 0.05], rgba=[0.5, 0.5, 0.5, 1])],
                joints=[Joint(type="free")]
            ),
            Body(
                name="robot_arm_link1",
                pos=[0, 0, 0.1],
                geoms=[Geom(type="capsule", size=[0.03, 0.2], rgba=[0.8, 0.2, 0.2, 1])],
                joints=[Joint(type="hinge", axis=[0, 0, 1], range=[-3.14, 3.14])]
            ),
            Body(
                name="robot_arm_link2",
                pos=[0, 0, 0.3],
                geoms=[Geom(type="capsule", size=[0.025, 0.15], rgba=[0.2, 0.8, 0.2, 1])],
                joints=[Joint(type="hinge", axis=[0, 1, 0], range=[-1.57, 1.57])]
            )
        ],
        actuators=[
            Actuator(name="joint1_motor", joint="robot_arm_link1_joint", gear=[50]),
            Actuator(name="joint2_motor", joint="robot_arm_link2_joint", gear=[40])
        ]
    )

    # Generate MJCF
    mjcf = spec.to_mjcf()
    assert "robot_arm_base" in mjcf
    assert "robot_arm_link1" in mjcf
    assert "robot_arm_link2" in mjcf
    assert "joint1_motor" in mjcf

    # Create request using schema
    request = SimulationRequest(
        prompt="Create a comprehensive 3-link robot arm simulation",
        parameters={
            "gravity": -9.81,
            "timestep": 0.001,
            "joint_damping": 0.1,
            "control_frequency": 100
        },
        quality_level="ultra"
    )

    # Create response using schema
    response = SimulationResponse(
        simulation_id="comprehensive_robot_001",
        mjcf_content=mjcf,
        status="completed",
        metadata={
            "generation_time": 3.2,
            "body_count": 3,
            "actuator_count": 2
        }
    )

    # Create user
    user = User(
        id="comprehensive_user",
        email="test@comprehensive.com",
        username="comp_user"
    )

    # Create simulation model
    simulation = Simulation(
        id="comprehensive_robot_001",
        user_id="comprehensive_user",
        prompt=request.prompt,
        mjcf_content=mjcf,
        status=SimulationStatus.COMPLETED,
        parameters=request.parameters
    )

    # Create settings
    settings = Settings(debug=True, environment="test")

    # Verify all integrations work
    assert request.parameters["gravity"] == -9.81
    assert response.metadata["body_count"] == 3
    assert simulation.status == SimulationStatus.COMPLETED
    assert settings.debug == True
    assert len(mjcf) > 100  # Substantial MJCF content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/simgen", "--cov-report=term"])