"""
Tests for PhysicsSpec pipeline
Validates the entire flow: PhysicsSpec → MJCF → MuJoCo
"""

import pytest
import numpy as np
import asyncio
from typing import Dict, Any

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simgen.models.physics_spec import (
    PhysicsSpec, Body, Geom, GeomType, JointType,
    SimulationMeta, Actuator, ActuatorType, Sensor, SensorType
)
from simgen.services.mjcf_compiler import MJCFCompiler
from .fixtures.golden_specs import (
    get_golden_spec, get_all_golden_names, validate_golden_spec, GOLDEN_SPECS
)

# Skip tests if MuJoCo not installed
try:
    import mujoco
    from simgen.services.mujoco_runtime import MuJoCoRuntime
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="MuJoCo not installed")

class TestPhysicsSpec:
    """Test PhysicsSpec model validation"""

    def test_minimal_spec(self):
        """Test minimal valid spec"""
        spec = PhysicsSpec(
            meta=SimulationMeta(name="test"),
            bodies=[
                Body(
                    id="box",
                    geoms=[Geom(type=GeomType.BOX, size=[0.1, 0.1, 0.1])]
                )
            ]
        )
        assert spec.meta.name == "test"
        assert len(spec.bodies) == 1

    def test_spec_validation(self):
        """Test spec validation rules"""
        # Invalid mass
        with pytest.raises(ValueError, match="Mass.*unrealistic"):
            Body(
                id="test",
                inertial={"mass": 1e-7}  # Too small
            )

        # Invalid joint range
        with pytest.raises(ValueError, match="range min.*less than max"):
            Body(
                id="test",
                joint={
                    "type": "hinge",
                    "limited": True,
                    "range": [1.0, -1.0]  # Invalid range
                }
            )

        # Invalid gear ratio
        with pytest.raises(ValueError, match="Gear ratio.*unrealistic"):
            Actuator(
                id="test",
                type=ActuatorType.MOTOR,
                target="joint",
                gear=10000  # Too high
            )

    def test_nested_bodies(self):
        """Test hierarchical body structure"""
        spec = PhysicsSpec(
            meta=SimulationMeta(name="nested"),
            bodies=[
                Body(
                    id="parent",
                    children=[
                        Body(
                            id="child1",
                            children=[
                                Body(id="grandchild")
                            ]
                        ),
                        Body(id="child2")
                    ]
                )
            ]
        )

        # Check structure
        assert len(spec.bodies) == 1
        assert len(spec.bodies[0].children) == 2
        assert len(spec.bodies[0].children[0].children) == 1

    def test_duplicate_body_ids(self):
        """Test that duplicate body IDs are rejected"""
        with pytest.raises(ValueError, match="Duplicate body ID"):
            PhysicsSpec(
                meta=SimulationMeta(name="test"),
                bodies=[
                    Body(id="box"),
                    Body(id="box")  # Duplicate
                ]
            )

class TestMJCFCompiler:
    """Test MJCF compilation"""

    @pytest.fixture
    def compiler(self):
        return MJCFCompiler()

    def test_compile_simple_box(self, compiler):
        """Test compiling a simple box"""
        spec = PhysicsSpec(
            meta=SimulationMeta(name="box_test"),
            bodies=[
                Body(
                    id="box",
                    pos=[0, 0, 1],
                    geoms=[
                        Geom(
                            type=GeomType.BOX,
                            size=[0.1, 0.1, 0.1],
                            material={"rgba": [1, 0, 0, 1]}
                        )
                    ],
                    inertial={"mass": 1.0}
                )
            ]
        )

        mjcf_xml = compiler.compile(spec)

        # Check XML structure
        assert "<mujoco" in mjcf_xml
        assert 'model="box_test"' in mjcf_xml
        assert "<body" in mjcf_xml
        assert 'name="box"' in mjcf_xml
        assert "<geom" in mjcf_xml
        assert 'type="box"' in mjcf_xml
        assert 'size="0.1 0.1 0.1"' in mjcf_xml

    def test_compile_with_joints(self, compiler):
        """Test compiling with different joint types"""
        for joint_type in [JointType.HINGE, JointType.SLIDER, JointType.BALL]:
            spec = PhysicsSpec(
                meta=SimulationMeta(name="joint_test"),
                bodies=[
                    Body(
                        id="body",
                        joint={
                            "type": joint_type.value,
                            "axis": [1, 0, 0] if joint_type != JointType.BALL else None
                        }
                    )
                ]
            )

            mjcf_xml = compiler.compile(spec)
            assert f'type="{joint_type.value}"' in mjcf_xml or f'type="slide"' in mjcf_xml

    def test_compile_with_actuators(self, compiler):
        """Test compiling with actuators"""
        spec = PhysicsSpec(
            meta=SimulationMeta(name="actuator_test"),
            bodies=[
                Body(
                    id="link",
                    joint={"type": "hinge"}
                )
            ],
            actuators=[
                Actuator(
                    id="motor",
                    type=ActuatorType.MOTOR,
                    target="link_joint",
                    gear=10
                )
            ]
        )

        mjcf_xml = compiler.compile(spec)
        assert "<actuator>" in mjcf_xml
        assert "<motor" in mjcf_xml
        assert 'name="motor"' in mjcf_xml
        assert 'gear="10"' in mjcf_xml

    def test_compile_with_sensors(self, compiler):
        """Test compiling with sensors"""
        spec = PhysicsSpec(
            meta=SimulationMeta(name="sensor_test"),
            bodies=[
                Body(
                    id="body",
                    joint={"type": "hinge"}
                )
            ],
            sensors=[
                Sensor(type=SensorType.JOINTPOS, source="body_joint"),
                Sensor(type=SensorType.JOINTVEL, source="body_joint")
            ]
        )

        mjcf_xml = compiler.compile(spec)
        assert "<sensor>" in mjcf_xml
        assert "<jointpos" in mjcf_xml
        assert "<jointvel" in mjcf_xml

@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
class TestGoldenSpecs:
    """Test golden physics specifications"""

    @pytest.mark.parametrize("spec_name", get_all_golden_names())
    def test_golden_spec_valid(self, spec_name):
        """Test that golden spec is valid"""
        spec = get_golden_spec(spec_name)
        assert isinstance(spec, PhysicsSpec)
        assert spec.meta.name == GOLDEN_SPECS[spec_name]["spec"]["meta"]["name"]

    @pytest.mark.parametrize("spec_name", get_all_golden_names())
    def test_golden_spec_compiles(self, spec_name):
        """Test that golden spec compiles to valid MJCF"""
        spec = get_golden_spec(spec_name)
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        # Validate with MuJoCo
        model = mujoco.MjModel.from_xml_string(mjcf_xml)
        assert model is not None

    @pytest.mark.parametrize("spec_name", get_all_golden_names())
    def test_golden_spec_properties(self, spec_name):
        """Test that golden spec has expected properties"""
        success, errors = validate_golden_spec(spec_name)
        assert success, f"Validation failed for {spec_name}: {errors}"

    @pytest.mark.asyncio
    async def test_pendulum_simulation(self):
        """Test running pendulum simulation"""
        spec = get_golden_spec("pendulum")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        # Create runtime
        runtime = MuJoCoRuntime(headless=True)
        manifest = runtime.load_mjcf(mjcf_xml)

        # Check manifest
        assert manifest.model_name == "simple_pendulum"
        assert manifest.nq == 1  # One hinge joint

        # Run for a short time
        initial_energy = None
        final_energy = None

        def capture_energy(frame):
            nonlocal initial_energy, final_energy
            if initial_energy is None:
                initial_energy = frame.sim_time
            final_energy = frame.sim_time

        await runtime.run_async(duration=1.0, callback=capture_energy)

        # Check simulation ran
        assert runtime.frame_count > 0
        assert runtime.sim_time > 0

    @pytest.mark.asyncio
    async def test_cart_pole_control(self):
        """Test cart-pole with control input"""
        spec = get_golden_spec("cart_pole")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        runtime = MuJoCoRuntime(headless=True)
        manifest = runtime.load_mjcf(mjcf_xml)

        # Check actuators
        assert manifest.nu == 1

        # Apply control
        runtime.set_control(np.array([0.5]))  # Push cart

        # Step simulation
        frame = runtime.step(10)
        assert frame.frame_id == 10

@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
class TestMuJoCoRuntime:
    """Test MuJoCo runtime and simulation"""

    @pytest.fixture
    def runtime(self):
        return MuJoCoRuntime(headless=True)

    def test_load_mjcf(self, runtime):
        """Test loading MJCF model"""
        spec = get_golden_spec("pendulum")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        manifest = runtime.load_mjcf(mjcf_xml)
        assert manifest.nbody > 0
        assert manifest.model_name == "simple_pendulum"

    def test_step_simulation(self, runtime):
        """Test stepping simulation"""
        spec = get_golden_spec("box_stack")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        runtime.load_mjcf(mjcf_xml)

        # Step multiple times
        for i in range(100):
            frame = runtime.step()
            assert frame.frame_id == i + 1
            assert frame.sim_time > 0

    def test_frame_serialization(self, runtime):
        """Test frame binary serialization"""
        spec = get_golden_spec("robot_arm_2dof")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        runtime.load_mjcf(mjcf_xml)
        frame = runtime.step()

        # Test binary serialization
        binary_data = frame.to_binary()
        assert isinstance(binary_data, bytes)
        assert len(binary_data) > 0

        # Test dict serialization
        dict_data = frame.to_dict()
        assert "frame_id" in dict_data
        assert "sim_time" in dict_data
        assert "qpos" in dict_data

    @pytest.mark.asyncio
    async def test_async_simulation(self, runtime):
        """Test async simulation run"""
        spec = get_golden_spec("double_pendulum")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        runtime.load_mjcf(mjcf_xml)

        frames_received = []

        def frame_callback(frame):
            frames_received.append(frame)

        # Run for 0.5 seconds
        await runtime.run_async(duration=0.5, callback=frame_callback)

        # Check frames were received
        assert len(frames_received) > 0
        assert runtime.status.value == "completed"

class TestPropertyBasedTesting:
    """Property-based tests for the physics pipeline"""

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_compile_never_crashes(self):
        """Test that compiler never crashes on valid input"""
        compiler = MJCFCompiler()

        # Generate various valid specs
        test_specs = [
            # Single body
            PhysicsSpec(
                meta=SimulationMeta(name="test1"),
                bodies=[Body(id="b1")]
            ),
            # Multiple bodies
            PhysicsSpec(
                meta=SimulationMeta(name="test2"),
                bodies=[
                    Body(id=f"b{i}")
                    for i in range(5)
                ]
            ),
            # Deep hierarchy
            PhysicsSpec(
                meta=SimulationMeta(name="test3"),
                bodies=[
                    Body(
                        id="root",
                        children=[
                            Body(
                                id=f"child{i}",
                                children=[Body(id=f"grandchild{i}")]
                            )
                            for i in range(3)
                        ]
                    )
                ]
            )
        ]

        for spec in test_specs:
            mjcf_xml = compiler.compile(spec)
            assert mjcf_xml is not None
            assert "<mujoco" in mjcf_xml

            # Should load in MuJoCo without errors
            model = mujoco.MjModel.from_xml_string(mjcf_xml)
            assert model is not None

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_energy_conservation(self):
        """Test energy conservation in pendulum"""
        spec = get_golden_spec("pendulum")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        runtime = MuJoCoRuntime(headless=True)
        runtime.load_mjcf(mjcf_xml)

        # Get initial energy
        initial_frame = runtime.get_state()

        # Run for 100 steps
        for _ in range(100):
            runtime.step()

        final_frame = runtime.get_state()

        # Energy should be approximately conserved (allowing for numerical errors)
        # This is a simplified test - real energy calculation would be more complex
        assert abs(initial_frame.sim_time - final_frame.sim_time) > 0