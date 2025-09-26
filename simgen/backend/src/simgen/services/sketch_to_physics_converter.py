"""
Sketch to PhysicsSpec Converter
Converts computer vision analysis results into structured PhysicsSpec objects
that can be compiled into MuJoCo simulations.
"""

import logging
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models.physics_spec import (
    PhysicsSpec, Body, Geom, GeomType, JointType, Joint, Actuator, ActuatorType,
    Sensor, SensorType, SimulationMeta, Inertial, Material, Friction, DefaultSettings
)
from .computer_vision_pipeline import (
    CVAnalysisResult, DetectedShape, DetectedConnection, DetectedText, ShapeType, ConnectionType
)

logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    """Result of sketch to PhysicsSpec conversion"""
    success: bool
    physics_spec: Optional[PhysicsSpec]
    confidence_score: float
    conversion_notes: List[str]
    error_message: Optional[str] = None

class SketchToPhysicsConverter:
    """
    Converts CV analysis results to PhysicsSpec objects
    """

    def __init__(self):
        self.shape_counter = 0
        self.joint_counter = 0
        self.actuator_counter = 0

    async def convert_cv_to_physics_spec(
        self,
        cv_result: CVAnalysisResult,
        user_prompt: Optional[str] = None,
        include_actuators: bool = True,
        include_sensors: bool = True
    ) -> ConversionResult:
        """
        Convert computer vision analysis to PhysicsSpec

        Args:
            cv_result: Computer vision analysis result
            user_prompt: Optional user text prompt for context
            include_actuators: Whether to add actuators to the spec
            include_sensors: Whether to add sensors to the spec

        Returns:
            Conversion result with PhysicsSpec
        """
        try:
            if not cv_result.shapes:
                return ConversionResult(
                    success=False,
                    physics_spec=None,
                    confidence_score=0.0,
                    conversion_notes=[],
                    error_message="No shapes detected in sketch"
                )

            # Reset counters
            self.shape_counter = 0
            self.joint_counter = 0
            self.actuator_counter = 0

            conversion_notes = []

            # Create simulation metadata
            meta = self._create_simulation_meta(cv_result, user_prompt)

            # Convert shapes to bodies
            bodies, body_conversion_notes = self._convert_shapes_to_bodies(cv_result.shapes, cv_result.text_annotations)
            conversion_notes.extend(body_conversion_notes)

            # Apply connections as joints
            self._apply_connections_to_bodies(bodies, cv_result.connections)

            # Create default settings
            defaults = self._create_default_settings(cv_result)

            # Create actuators if requested
            actuators = []
            if include_actuators:
                actuators, actuator_notes = self._create_actuators(bodies, cv_result)
                conversion_notes.extend(actuator_notes)

            # Create sensors if requested
            sensors = []
            if include_sensors:
                sensors, sensor_notes = self._create_sensors(bodies, cv_result)
                conversion_notes.extend(sensor_notes)

            # Create PhysicsSpec
            physics_spec = PhysicsSpec(
                meta=meta,
                defaults=defaults,
                bodies=bodies,
                actuators=actuators,
                sensors=sensors
            )

            # Calculate confidence score
            confidence = self._calculate_conversion_confidence(cv_result, len(bodies), len(actuators))

            conversion_notes.append(f"Successfully converted {len(cv_result.shapes)} shapes to {len(bodies)} bodies")

            return ConversionResult(
                success=True,
                physics_spec=physics_spec,
                confidence_score=confidence,
                conversion_notes=conversion_notes
            )

        except Exception as e:
            logger.error(f"Sketch to PhysicsSpec conversion failed: {e}")
            return ConversionResult(
                success=False,
                physics_spec=None,
                confidence_score=0.0,
                conversion_notes=[],
                error_message=str(e)
            )

    def _create_simulation_meta(self, cv_result: CVAnalysisResult, user_prompt: Optional[str]) -> SimulationMeta:
        """Create simulation metadata from CV analysis"""

        # Determine name from user prompt or default
        name = "sketch_simulation"
        if user_prompt:
            # Extract potential name from prompt
            words = user_prompt.lower().split()
            physics_words = ['pendulum', 'robot', 'arm', 'cart', 'pole', 'ball', 'box', 'spring', 'lever']
            for word in physics_words:
                if word in words:
                    name = word + "_sketch"
                    break

        # Determine gravity from environment or text annotations
        gravity = [0.0, 0.0, -9.81]  # Default Earth gravity

        # Check for gravity hints in text
        all_text = ' '.join([text.text.lower() for text in cv_result.text_annotations])
        if 'space' in all_text or 'zero gravity' in all_text:
            gravity = [0.0, 0.0, 0.0]
        elif 'moon' in all_text:
            gravity = [0.0, 0.0, -1.62]
        elif cv_result.physics_interpretation.get('environment', {}).get('gravity'):
            env_gravity = cv_result.physics_interpretation['environment']['gravity']
            gravity = env_gravity

        return SimulationMeta(
            name=name,
            description=f"Physics simulation generated from hand-drawn sketch{' with prompt: ' + user_prompt if user_prompt else ''}",
            gravity=gravity,
            timestep=0.002  # Good default for hand-drawn physics
        )

    def _convert_shapes_to_bodies(self, shapes: List[DetectedShape], text_annotations: List[DetectedText]) -> Tuple[List[Body], List[str]]:
        """Convert detected shapes to Body objects"""
        bodies = []
        notes = []

        # Convert image coordinates to physics coordinates (assume sketch is ~2m x 2m in physics)
        # Find image bounds from shapes
        if not shapes:
            return bodies, notes

        all_centers = [(shape.center.x, shape.center.y) for shape in shapes]
        min_x = min(x for x, y in all_centers)
        max_x = max(x for x, y in all_centers)
        min_y = min(y for x, y in all_centers)
        max_y = max(y for x, y in all_centers)

        # Scale factor to fit in ~2m x 2m physics space
        scale_factor = 2.0 / max(max_x - min_x, max_y - min_y, 100)  # Minimum 100px reference
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        for shape in shapes:
            body = self._convert_shape_to_body(shape, scale_factor, center_x, center_y)
            if body:
                bodies.append(body)
                notes.append(f"Converted {shape.shape_type.value} to body '{body.id}'")
            self.shape_counter += 1

        return bodies, notes

    def _convert_shape_to_body(self, shape: DetectedShape, scale_factor: float, center_x: float, center_y: float) -> Optional[Body]:
        """Convert a single detected shape to a Body"""

        # Calculate physics position (centered at origin, Y up)
        physics_x = (shape.center.x - center_x) * scale_factor
        physics_y = -(shape.center.y - center_y) * scale_factor  # Flip Y for physics
        physics_z = 0.0  # 2D sketch assumption

        # Determine if this should be a free-floating body or have a joint
        joint = None
        if self._shape_needs_joint(shape):
            joint = self._create_joint_for_shape(shape)

        # Create geometry
        geoms = self._create_geoms_for_shape(shape, scale_factor)
        if not geoms:
            return None

        # Create inertial properties
        inertial = self._create_inertial_for_shape(shape, scale_factor)

        # Create body ID
        body_id = f"{shape.shape_type.value}_{self.shape_counter}"

        return Body(
            id=body_id,
            pos=[physics_x, physics_y, physics_z],
            joint=joint,
            geoms=geoms,
            inertial=inertial
        )

    def _create_geoms_for_shape(self, shape: DetectedShape, scale_factor: float) -> List[Geom]:
        """Create geometry objects for a detected shape"""
        geoms = []

        # Determine material based on annotations
        material = self._create_material_from_shape(shape)

        if shape.shape_type == ShapeType.CIRCLE:
            radius = shape.parameters.get('radius', 25) * scale_factor
            geom = Geom(
                type=GeomType.SPHERE,
                size=[radius],
                material=material
            )
            geoms.append(geom)

        elif shape.shape_type == ShapeType.RECTANGLE:
            width = shape.parameters.get('width', 50) * scale_factor
            height = shape.parameters.get('height', 50) * scale_factor
            depth = min(width, height) * 0.1  # Thin box for 2D sketch

            geom = Geom(
                type=GeomType.BOX,
                size=[width/2, height/2, depth/2],  # MuJoCo uses half-sizes
                material=material
            )
            geoms.append(geom)

        elif shape.shape_type == ShapeType.LINE:
            length = shape.parameters.get('length', 100) * scale_factor
            radius = 0.01  # Thin rod

            # Use capsule for line/rod
            geom = Geom(
                type=GeomType.CAPSULE,
                fromto=[0, 0, -length/2, 0, 0, length/2],  # Vertical rod
                size=[radius],
                material=material
            )
            geoms.append(geom)

        else:
            # Default to small sphere for unknown shapes
            geom = Geom(
                type=GeomType.SPHERE,
                size=[0.05],
                material=material
            )
            geoms.append(geom)

        return geoms

    def _create_material_from_shape(self, shape: DetectedShape) -> Optional[Material]:
        """Create material properties based on shape annotations"""
        material = Material()

        # Check annotations for material hints
        annotations = shape.physics_properties.get('annotations', [])
        all_text = ' '.join(annotations).lower()

        # Color based on material type
        if 'metal' in all_text or 'steel' in all_text:
            material.rgba = [0.7, 0.7, 0.8, 1.0]  # Metallic gray
        elif 'wood' in all_text:
            material.rgba = [0.8, 0.6, 0.4, 1.0]  # Brown
        elif 'red' in all_text:
            material.rgba = [0.8, 0.2, 0.2, 1.0]
        elif 'blue' in all_text:
            material.rgba = [0.2, 0.2, 0.8, 1.0]
        elif 'green' in all_text:
            material.rgba = [0.2, 0.8, 0.2, 1.0]
        else:
            # Default colors based on shape type
            if shape.shape_type == ShapeType.CIRCLE:
                material.rgba = [0.8, 0.3, 0.3, 1.0]  # Red for balls
            elif shape.shape_type == ShapeType.RECTANGLE:
                material.rgba = [0.3, 0.3, 0.8, 1.0]  # Blue for boxes
            else:
                material.rgba = [0.6, 0.6, 0.6, 1.0]  # Gray default

        return material

    def _create_inertial_for_shape(self, shape: DetectedShape, scale_factor: float) -> Inertial:
        """Create inertial properties for a shape"""

        # Default mass based on volume and material density
        density = 1000.0  # Default density (kg/m³)

        # Adjust density based on annotations
        annotations = shape.physics_properties.get('annotations', [])
        all_text = ' '.join(annotations).lower()

        if 'heavy' in all_text:
            density *= 3
        elif 'light' in all_text:
            density *= 0.3
        elif 'metal' in all_text:
            density = 7800  # Steel
        elif 'wood' in all_text:
            density = 600
        elif 'plastic' in all_text:
            density = 1200

        # Calculate volume and mass
        if shape.shape_type == ShapeType.CIRCLE:
            radius = shape.parameters.get('radius', 25) * scale_factor
            volume = (4/3) * math.pi * radius**3
        elif shape.shape_type == ShapeType.RECTANGLE:
            width = shape.parameters.get('width', 50) * scale_factor
            height = shape.parameters.get('height', 50) * scale_factor
            depth = min(width, height) * 0.1
            volume = width * height * depth
        elif shape.shape_type == ShapeType.LINE:
            length = shape.parameters.get('length', 100) * scale_factor
            radius = 0.01
            volume = math.pi * radius**2 * length
        else:
            volume = 0.001  # Default small volume

        mass = volume * density

        return Inertial(
            mass=max(0.01, mass),  # Minimum mass to avoid numerical issues
            pos=[0, 0, 0]  # Center of mass at body origin
        )

    def _shape_needs_joint(self, shape: DetectedShape) -> bool:
        """Determine if a shape should have a joint (not be fixed to world)"""

        # Check annotations for hints about movement
        annotations = shape.physics_properties.get('annotations', [])
        all_text = ' '.join(annotations).lower()

        # Keywords that suggest the object should be moveable
        moveable_keywords = ['swing', 'rotate', 'move', 'slide', 'roll', 'fall', 'drop']
        fixed_keywords = ['fixed', 'static', 'ground', 'wall', 'base', 'anchor']

        if any(word in all_text for word in fixed_keywords):
            return False

        if any(word in all_text for word in moveable_keywords):
            return True

        # Default heuristics based on shape type
        if shape.shape_type == ShapeType.CIRCLE:
            return True  # Balls typically move
        elif shape.shape_type == ShapeType.RECTANGLE:
            # Large rectangles might be static, small ones might move
            width = shape.parameters.get('width', 50)
            height = shape.parameters.get('height', 50)
            if max(width, height) > 100:  # Large objects tend to be static
                return False
            return True
        elif shape.shape_type == ShapeType.LINE:
            return True  # Rods typically swing or move

        return True  # Default to moveable

    def _create_joint_for_shape(self, shape: DetectedShape) -> Joint:
        """Create an appropriate joint for a shape"""

        # Check annotations for joint type hints
        annotations = shape.physics_properties.get('annotations', [])
        all_text = ' '.join(annotations).lower()

        # Determine joint type from context
        if 'swing' in all_text or 'pendulum' in all_text or 'rotate' in all_text:
            return Joint(
                type=JointType.HINGE,
                axis=[0, 1, 0],  # Rotate around Y axis (into screen)
                limited=False,  # Free rotation
                damping=0.1
            )
        elif 'slide' in all_text or 'slider' in all_text:
            return Joint(
                type=JointType.SLIDER,
                axis=[1, 0, 0],  # Slide along X axis
                limited=True,
                range=[-1.0, 1.0],  # 2m range
                damping=0.05
            )
        else:
            # Default joint based on shape type
            if shape.shape_type == ShapeType.CIRCLE:
                # Free joint for balls (can move and rotate freely)
                return Joint(
                    type=JointType.FREE,
                    damping=0.01
                )
            elif shape.shape_type == ShapeType.LINE:
                # Hinge joint for rods
                return Joint(
                    type=JointType.HINGE,
                    axis=[0, 1, 0],
                    limited=False,
                    damping=0.1
                )
            else:
                # Free joint default
                return Joint(
                    type=JointType.FREE,
                    damping=0.05
                )

    def _apply_connections_to_bodies(self, bodies: List[Body], connections: List[DetectedConnection]):
        """Apply detected connections as parent-child relationships"""

        # For now, we'll handle connections by creating joint constraints
        # In a more advanced version, we could restructure the body hierarchy

        for connection in connections:
            # Find bodies that match the connection
            body1 = None
            body2 = None

            for body in bodies:
                if connection.shape1_id in body.id or any(connection.shape1_id in stroke_id for stroke_id in getattr(body, 'source_strokes', [])):
                    body1 = body
                if connection.shape2_id in body.id or any(connection.shape2_id in stroke_id for stroke_id in getattr(body, 'source_strokes', [])):
                    body2 = body

            if body1 and body2:
                # Modify joints to reflect connection
                if connection.connection_type == ConnectionType.HINGE_JOINT:
                    # Both bodies should have hinge joints
                    if body1.joint and body1.joint.type != JointType.HINGE:
                        body1.joint.type = JointType.HINGE
                    if body2.joint and body2.joint.type != JointType.HINGE:
                        body2.joint.type = JointType.HINGE
                elif connection.connection_type == ConnectionType.FIXED_JOINT:
                    # One body should be fixed
                    if body2.joint:
                        body2.joint = None  # Remove joint to fix to world

    def _create_actuators(self, bodies: List[Body], cv_result: CVAnalysisResult) -> Tuple[List[Actuator], List[str]]:
        """Create actuators for bodies that have joints"""
        actuators = []
        notes = []

        for body in bodies:
            if body.joint and body.joint.type in [JointType.HINGE, JointType.SLIDER]:
                # Check if body annotations suggest it should be actuated
                actuator_keywords = ['motor', 'drive', 'control', 'actuate', 'powered']

                # For now, add motors to hinge joints (typical for robot arms, pendulums)
                if body.joint.type == JointType.HINGE:
                    actuator = Actuator(
                        id=f"{body.id}_motor",
                        type=ActuatorType.MOTOR,
                        target=f"{body.id}_joint",  # Joint name convention
                        gear=20.0,  # Reasonable gear ratio
                        ctrlrange=[-2.0, 2.0]  # Control range in Nm
                    )
                    actuators.append(actuator)
                    notes.append(f"Added motor to {body.id}")
                    self.actuator_counter += 1

        return actuators, notes

    def _create_sensors(self, bodies: List[Body], cv_result: CVAnalysisResult) -> Tuple[List[Sensor], List[str]]:
        """Create sensors for physics monitoring"""
        sensors = []
        notes = []

        for body in bodies:
            if body.joint:
                # Add position sensor for joint
                pos_sensor = Sensor(
                    type=SensorType.JOINTPOS,
                    source=f"{body.id}_joint"
                )
                sensors.append(pos_sensor)

                # Add velocity sensor for joint
                vel_sensor = Sensor(
                    type=SensorType.JOINTVEL,
                    source=f"{body.id}_joint"
                )
                sensors.append(vel_sensor)

                notes.append(f"Added position and velocity sensors to {body.id}")

        return sensors, notes

    def _create_default_settings(self, cv_result: CVAnalysisResult) -> DefaultSettings:
        """Create default settings based on CV analysis"""

        # Analyze environment from CV result
        env_info = cv_result.physics_interpretation.get('environment', {})

        # Default friction based on environment
        friction_slide = 0.8  # Default
        if 'ice' in str(env_info).lower():
            friction_slide = 0.05
        elif 'rough' in str(env_info).lower():
            friction_slide = 1.2

        friction = Friction(
            slide=friction_slide,
            spin=0.005,
            roll=0.0001
        )

        return DefaultSettings(
            friction=friction,
            joint_damping=0.1,
            geom_density=1000.0
        )

    def _calculate_conversion_confidence(self, cv_result: CVAnalysisResult, num_bodies: int, num_actuators: int) -> float:
        """Calculate confidence score for the conversion"""

        base_confidence = cv_result.confidence_score

        # Boost confidence based on successful conversions
        if num_bodies > 0:
            base_confidence += 0.1

        if num_actuators > 0:
            base_confidence += 0.05

        # Check if we have good shape variety
        shape_types = set()
        for shape in cv_result.shapes:
            shape_types.add(shape.shape_type)

        if len(shape_types) > 1:
            base_confidence += 0.1

        # Check for connections
        if cv_result.connections:
            base_confidence += 0.15

        # Check for text annotations
        if cv_result.text_annotations:
            base_confidence += 0.05

        return min(1.0, base_confidence)


# Factory function
def create_sketch_to_physics_converter() -> SketchToPhysicsConverter:
    """Create a sketch to PhysicsSpec converter instance"""
    return SketchToPhysicsConverter()