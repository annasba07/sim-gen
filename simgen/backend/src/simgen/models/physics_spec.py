"""
PhysicsSpec: The source of truth for physics simulations
A typed, validated intermediate representation between AI and MuJoCo
"""

from pydantic import BaseModel, Field, conlist, validator
from typing import List, Literal, Optional, Dict, Any, Union
from enum import Enum

# Type aliases for clarity
Vec3 = conlist(float, min_length=3, max_length=3)
Vec4 = conlist(float, min_length=4, max_length=4)
Vec6 = conlist(float, min_length=6, max_length=6)

class PhysicsSpecVersion(str, Enum):
    """Semantic versioning for schema evolution"""
    V1_0_0 = "1.0.0"
    V1_1_0 = "1.1.0"  # Future: adds soft bodies
    V2_0_0 = "2.0.0"  # Future: breaking changes

class JointType(str, Enum):
    """Supported joint types in MuJoCo"""
    HINGE = "hinge"      # 1 DOF rotation
    SLIDER = "slide"     # 1 DOF translation
    BALL = "ball"        # 3 DOF rotation
    FREE = "free"        # 6 DOF free joint

class GeomType(str, Enum):
    """Supported geometry primitives"""
    BOX = "box"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    ELLIPSOID = "ellipsoid"
    MESH = "mesh"
    PLANE = "plane"

class ActuatorType(str, Enum):
    """Actuator types"""
    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    CYLINDER = "cylinder"  # Pneumatic/hydraulic
    MUSCLE = "muscle"      # Hill-type muscle model

class SensorType(str, Enum):
    """Sensor types"""
    JOINTPOS = "jointpos"
    JOINTVEL = "jointvel"
    TENDONPOS = "tendonpos"
    TENDONVEL = "tendonvel"
    ACTUATORPOS = "actuatorpos"
    ACTUATORVEL = "actuatorvel"
    ACTUATORFRC = "actuatorfrc"
    ACCELEROMETER = "accelerometer"
    VELOCIMETER = "velocimeter"
    GYRO = "gyro"
    FORCE = "force"
    TORQUE = "torque"
    MAGNETOMETER = "magnetometer"
    RANGEFINDER = "rangefinder"
    FRAMEPOS = "framepos"
    FRAMEQUAT = "framequat"

class Material(BaseModel):
    """Material properties for rendering and physics"""
    rgba: Optional[Vec4] = Field(default=[0.7, 0.7, 0.7, 1.0], description="RGBA color")
    texture: Optional[str] = Field(default=None, description="Texture name")
    emission: Optional[float] = Field(default=0.0, ge=0, le=1, description="Light emission")
    specular: Optional[float] = Field(default=0.5, ge=0, le=1, description="Specularity")
    shininess: Optional[float] = Field(default=0.5, ge=0, le=1, description="Shininess")
    reflectance: Optional[float] = Field(default=0.0, ge=0, le=1, description="Reflectance")

class Friction(BaseModel):
    """Friction parameters (slide, spin, roll)"""
    slide: float = Field(default=1.0, ge=0, description="Sliding friction")
    spin: float = Field(default=0.005, ge=0, description="Spinning friction")
    roll: float = Field(default=0.0001, ge=0, description="Rolling friction")

class Joint(BaseModel):
    """Joint specification"""
    type: JointType
    axis: Optional[Vec3] = Field(default=[0, 0, 1], description="Joint axis in local frame")
    limited: bool = Field(default=True, description="Whether joint has limits")
    range: Optional[conlist(float, min_length=2, max_length=2)] = Field(
        default=None, description="Joint limits [min, max] in radians or meters"
    )
    damping: Optional[float] = Field(default=0.0, ge=0, description="Joint damping")
    stiffness: Optional[float] = Field(default=0.0, ge=0, description="Joint stiffness")
    armature: Optional[float] = Field(default=0.0, ge=0, description="Rotor inertia")

    @validator('range')
    def validate_range(cls, v, values):
        if v and values.get('limited'):
            if v[0] >= v[1]:
                raise ValueError(f"Joint range min {v[0]} must be less than max {v[1]}")
        return v

class Geom(BaseModel):
    """Geometry specification"""
    type: GeomType
    size: Optional[List[float]] = Field(default=None, description="Size parameters")
    fromto: Optional[Vec6] = Field(default=None, description="Capsule/cylinder endpoints")
    pos: Optional[Vec3] = Field(default=[0, 0, 0], description="Local position")
    quat: Optional[Vec4] = Field(default=None, description="Local orientation quaternion")
    mesh: Optional[str] = Field(default=None, description="Mesh filename")
    material: Optional[Material] = Field(default=None)
    friction: Optional[Friction] = Field(default=None)
    density: Optional[float] = Field(default=1000.0, gt=0, description="Density in kg/m³")

    @validator('size')
    def validate_size(cls, v, values):
        geom_type = values.get('type')
        if v:
            if geom_type == GeomType.BOX and len(v) != 3:
                raise ValueError("Box requires 3 size parameters [x, y, z]")
            elif geom_type == GeomType.SPHERE and len(v) != 1:
                raise ValueError("Sphere requires 1 size parameter [radius]")
            elif geom_type == GeomType.CAPSULE and len(v) != 1:
                raise ValueError("Capsule requires 1 size parameter [radius], use fromto for length")
            elif geom_type == GeomType.CYLINDER and len(v) != 1:
                raise ValueError("Cylinder requires 1 size parameter [radius], use fromto for length")
        return v

class Inertial(BaseModel):
    """Inertial properties"""
    mass: float = Field(gt=0, description="Mass in kg")
    pos: Optional[Vec3] = Field(default=[0, 0, 0], description="Center of mass")
    diaginertia: Optional[Vec3] = Field(default=None, description="Diagonal inertia")
    fullinertia: Optional[Vec6] = Field(default=None, description="Full inertia matrix")

    @validator('mass')
    def validate_mass(cls, v):
        if v < 1e-6 or v > 1e6:
            raise ValueError(f"Mass {v} kg is unrealistic (must be between 1e-6 and 1e6 kg)")
        return v

class Body(BaseModel):
    """Rigid body specification"""
    id: str = Field(description="Unique identifier")
    pos: Optional[Vec3] = Field(default=[0, 0, 0], description="Position in parent frame")
    quat: Optional[Vec4] = Field(default=None, description="Orientation quaternion")
    joint: Optional[Joint] = Field(default=None)
    geoms: List[Geom] = Field(default_factory=list, description="Geometries attached to body")
    inertial: Optional[Inertial] = Field(default=None, description="Inertial properties")
    children: List["Body"] = Field(default_factory=list, description="Child bodies")

    @validator('id')
    def validate_id(cls, v):
        if not v or not v.replace('_', '').isalnum():
            raise ValueError(f"Body ID '{v}' must be alphanumeric with underscores")
        return v

# Update forward references
Body.model_rebuild()

class Actuator(BaseModel):
    """Actuator specification"""
    id: str
    type: ActuatorType
    target: str = Field(description="Joint or tendon to actuate")
    gear: float = Field(default=1.0, description="Gear ratio")
    ctrlrange: Optional[conlist(float, min_length=2, max_length=2)] = Field(
        default=None, description="Control limits"
    )
    forcerange: Optional[conlist(float, min_length=2, max_length=2)] = Field(
        default=None, description="Force/torque limits"
    )

    @validator('gear')
    def validate_gear(cls, v):
        if abs(v) > 1000:
            raise ValueError(f"Gear ratio {v} is unrealistic (max ±1000)")
        return v

class Sensor(BaseModel):
    """Sensor specification"""
    id: Optional[str] = None
    type: SensorType
    source: str = Field(description="What to sense (body/joint/actuator name)")
    noise: Optional[float] = Field(default=0.0, ge=0, description="Sensor noise std dev")

class Contact(BaseModel):
    """Contact pair override"""
    pair: conlist(str, min_length=2, max_length=2) = Field(description="Geom pair names")
    exclude: bool = Field(default=False, description="Exclude this contact pair")
    friction: Optional[Friction] = None
    solref: Optional[conlist(float, min_length=2, max_length=2)] = Field(
        default=None, description="Solver reference parameters"
    )

class Equality(BaseModel):
    """Equality constraint"""
    type: Literal["joint", "tendon", "distance", "weld"]
    body1: str
    body2: Optional[str] = None
    joint: Optional[str] = None
    polycoef: Optional[List[float]] = Field(default=None, description="Polynomial coefficients")

class DefaultSettings(BaseModel):
    """Default settings for the simulation"""
    material: Optional[Material] = None
    friction: Optional[Friction] = None
    joint_damping: float = Field(default=0.0, ge=0)
    joint_stiffness: float = Field(default=0.0, ge=0)
    geom_density: float = Field(default=1000.0, gt=0)

class SimulationMeta(BaseModel):
    """Simulation metadata"""
    name: str = Field(default="simgen", description="Model name")
    version: PhysicsSpecVersion = Field(default=PhysicsSpecVersion.V1_0_0)
    author: Optional[str] = None
    description: Optional[str] = None
    gravity: Vec3 = Field(default=[0, 0, -9.81], description="Gravity vector in m/s²")
    timestep: float = Field(default=0.002, gt=0, le=0.1, description="Simulation timestep")
    integrator: Literal["Euler", "RK4", "implicit"] = Field(default="RK4")

    @validator('gravity')
    def validate_gravity(cls, v):
        mag = sum(x**2 for x in v) ** 0.5
        if mag > 100:
            raise ValueError(f"Gravity magnitude {mag} m/s² is unrealistic (Earth is 9.81)")
        return v

class PhysicsSpec(BaseModel):
    """
    Complete physics specification
    This is the contract between AI and the physics engine
    """
    meta: SimulationMeta = Field(default_factory=SimulationMeta)
    defaults: Optional[DefaultSettings] = None
    bodies: List[Body] = Field(description="Root bodies in worldbody")
    actuators: List[Actuator] = Field(default_factory=list)
    sensors: List[Sensor] = Field(default_factory=list)
    contacts: List[Contact] = Field(default_factory=list)
    equality: List[Equality] = Field(default_factory=list)

    @validator('bodies')
    def validate_bodies(cls, v):
        if not v:
            raise ValueError("At least one body is required")
        # Check for duplicate IDs
        def collect_ids(bodies: List[Body], ids: set = None):
            if ids is None:
                ids = set()
            for body in bodies:
                if body.id in ids:
                    raise ValueError(f"Duplicate body ID: {body.id}")
                ids.add(body.id)
                collect_ids(body.children, ids)
            return ids
        collect_ids(v)
        return v

    def to_mjcf(self) -> str:
        """Convert to MJCF XML (delegates to compiler)"""
        from ..services.mjcf_compiler import MJCFCompiler
        compiler = MJCFCompiler()
        return compiler.compile(self)

    class Config:
        json_encoders = {
            PhysicsSpecVersion: lambda v: v.value,
        }
        json_schema_extra = {
            "example": {
                "meta": {
                    "name": "pendulum",
                    "gravity": [0, 0, -9.81]
                },
                "bodies": [
                    {
                        "id": "pendulum",
                        "pos": [0, 0, 1],
                        "joint": {
                            "type": "hinge",
                            "axis": [0, 1, 0],
                            "limited": False
                        },
                        "geoms": [
                            {
                                "type": "capsule",
                                "fromto": [0, 0, 0, 0, 0, -0.5],
                                "size": [0.05]
                            }
                        ],
                        "inertial": {
                            "mass": 1.0
                        }
                    }
                ]
            }
        }