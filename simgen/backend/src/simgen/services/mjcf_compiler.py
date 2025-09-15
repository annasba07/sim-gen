"""
MJCF Compiler: Deterministic PhysicsSpec → MuJoCo XML conversion
Produces valid, predictable MJCF output from validated PhysicsSpec
"""

import logging
from xml.etree import ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, List, Optional
from ..models.physics_spec import (
    PhysicsSpec, Body, Geom, Joint, Actuator, Sensor, Contact,
    JointType, GeomType, ActuatorType, SensorType, Material, Friction
)

logger = logging.getLogger(__name__)

class MJCFCompiler:
    """
    Compiles PhysicsSpec to MuJoCo XML format
    Deterministic, validating, with sensible defaults
    """

    def __init__(self):
        self.body_map: Dict[str, ET.Element] = {}
        self.joint_map: Dict[str, ET.Element] = {}
        self.geom_map: Dict[str, ET.Element] = {}

    def compile(self, spec: PhysicsSpec) -> str:
        """
        Main entry point: PhysicsSpec → MJCF XML string
        """
        try:
            # Create root element
            mujoco = ET.Element("mujoco", model=spec.meta.name)

            # Add compiler settings for better debugging
            compiler = ET.SubElement(mujoco, "compiler")
            compiler.set("angle", "radian")
            compiler.set("coordinate", "local")
            compiler.set("inertiafromgeom", "true")  # Auto-compute inertia if missing
            compiler.set("inertiagrouprange", "0 5")

            # Set options
            self._add_options(mujoco, spec)

            # Add defaults
            self._add_defaults(mujoco, spec)

            # Add assets (textures, meshes)
            self._add_assets(mujoco, spec)

            # Add worldbody
            worldbody = ET.SubElement(mujoco, "worldbody")
            self._add_ground_plane(worldbody)

            # Recursively add bodies
            for body in spec.bodies:
                self._add_body(worldbody, body, spec)

            # Add actuators
            if spec.actuators:
                self._add_actuators(mujoco, spec.actuators)

            # Add sensors
            if spec.sensors:
                self._add_sensors(mujoco, spec.sensors)

            # Add contact pairs
            if spec.contacts:
                self._add_contacts(mujoco, spec.contacts)

            # Add equality constraints
            if spec.equality:
                self._add_equality(mujoco, spec.equality)

            # Convert to pretty-printed string
            return self._prettify_xml(mujoco)

        except Exception as e:
            logger.error(f"MJCF compilation failed: {e}")
            raise

    def _add_options(self, mujoco: ET.Element, spec: PhysicsSpec):
        """Add simulation options"""
        option = ET.SubElement(mujoco, "option")

        # Gravity
        gravity = spec.meta.gravity
        option.set("gravity", f"{gravity[0]} {gravity[1]} {gravity[2]}")

        # Timestep
        option.set("timestep", str(spec.meta.timestep))

        # Integrator
        integrator_map = {"Euler": "Euler", "RK4": "RK4", "implicit": "implicit"}
        option.set("integrator", integrator_map.get(spec.meta.integrator, "RK4"))

        # Solver settings (collision is not a valid option attribute)
        # These go in a different section or are defaults

    def _add_defaults(self, mujoco: ET.Element, spec: PhysicsSpec):
        """Add default settings"""
        default = ET.SubElement(mujoco, "default")

        if spec.defaults:
            # Joint defaults
            joint_default = ET.SubElement(default, "joint")
            joint_default.set("damping", str(spec.defaults.joint_damping))
            joint_default.set("stiffness", str(spec.defaults.joint_stiffness))

            # Geom defaults
            geom_default = ET.SubElement(default, "geom")
            geom_default.set("density", str(spec.defaults.geom_density))

            # Material defaults
            if spec.defaults.material:
                mat = spec.defaults.material
                geom_default.set("rgba", " ".join(map(str, mat.rgba)))

            # Friction defaults
            if spec.defaults.friction:
                fric = spec.defaults.friction
                geom_default.set("friction", f"{fric.slide} {fric.spin} {fric.roll}")
        else:
            # Sensible defaults if not specified
            joint_default = ET.SubElement(default, "joint")
            joint_default.set("damping", "0.1")

            geom_default = ET.SubElement(default, "geom")
            geom_default.set("rgba", "0.7 0.7 0.7 1")
            geom_default.set("friction", "1.0 0.005 0.0001")
            geom_default.set("density", "1000")

    def _add_assets(self, mujoco: ET.Element, spec: PhysicsSpec):
        """Add assets (textures, materials, meshes)"""
        # Collect unique materials and meshes
        materials = set()
        meshes = set()

        def collect_from_body(body: Body):
            for geom in body.geoms:
                if geom.material and geom.material.texture:
                    materials.add(geom.material.texture)
                if geom.type == GeomType.MESH and geom.mesh:
                    meshes.add(geom.mesh)
            for child in body.children:
                collect_from_body(child)

        for body in spec.bodies:
            collect_from_body(body)

        if materials or meshes:
            asset = ET.SubElement(mujoco, "asset")

            # Add textures
            for tex_name in materials:
                texture = ET.SubElement(asset, "texture")
                texture.set("name", tex_name)
                texture.set("type", "2d")
                texture.set("builtin", "checker")  # Default checker pattern
                texture.set("rgb1", "0.8 0.8 0.8")
                texture.set("rgb2", "0.6 0.6 0.6")
                texture.set("width", "512")
                texture.set("height", "512")

            # Add materials
            for tex_name in materials:
                material = ET.SubElement(asset, "material")
                material.set("name", f"mat_{tex_name}")
                material.set("texture", tex_name)

            # Add meshes (placeholder - actual mesh loading would go here)
            for mesh_name in meshes:
                mesh = ET.SubElement(asset, "mesh")
                mesh.set("name", mesh_name)
                mesh.set("file", f"meshes/{mesh_name}.stl")

    def _add_ground_plane(self, worldbody: ET.Element):
        """Add a ground plane with checkerboard pattern"""
        ground = ET.SubElement(worldbody, "geom")
        ground.set("name", "ground")
        ground.set("type", "plane")
        ground.set("size", "10 10 0.1")
        ground.set("rgba", "0.8 0.8 0.8 1")
        ground.set("friction", "1.0 0.005 0.0001")

    def _add_body(self, parent: ET.Element, body: Body, spec: PhysicsSpec):
        """Recursively add body and its children"""
        # Create body element
        body_elem = ET.SubElement(parent, "body")
        body_elem.set("name", body.id)

        # Set position
        if body.pos:
            body_elem.set("pos", " ".join(map(str, body.pos)))

        # Set orientation (quaternion)
        if body.quat:
            body_elem.set("quat", " ".join(map(str, body.quat)))

        # Store in map for actuator/sensor references
        self.body_map[body.id] = body_elem

        # Add joint if specified
        if body.joint:
            self._add_joint(body_elem, body.joint, body.id)

        # Add geometries
        for i, geom in enumerate(body.geoms):
            self._add_geom(body_elem, geom, f"{body.id}_geom_{i}")

        # Add inertial properties
        if body.inertial:
            inertial = ET.SubElement(body_elem, "inertial")
            inertial.set("mass", str(body.inertial.mass))

            if body.inertial.pos:
                inertial.set("pos", " ".join(map(str, body.inertial.pos)))

            if body.inertial.diaginertia:
                inertial.set("diaginertia", " ".join(map(str, body.inertial.diaginertia)))
            elif body.inertial.fullinertia:
                inertial.set("fullinertia", " ".join(map(str, body.inertial.fullinertia)))

        # Recursively add children
        for child in body.children:
            self._add_body(body_elem, child, spec)

    def _add_joint(self, body_elem: ET.Element, joint: Joint, body_id: str):
        """Add joint to body"""
        # Map joint types
        type_map = {
            JointType.HINGE: "hinge",
            JointType.SLIDER: "slide",
            JointType.BALL: "ball",
            JointType.FREE: "free"
        }

        joint_elem = ET.SubElement(body_elem, "joint")
        joint_elem.set("name", f"{body_id}_joint")
        joint_elem.set("type", type_map[joint.type])

        # Set axis (except for ball and free joints)
        if joint.type not in [JointType.BALL, JointType.FREE] and joint.axis:
            joint_elem.set("axis", " ".join(map(str, joint.axis)))

        # Set limits
        if joint.limited:
            joint_elem.set("limited", "true")
            if joint.range:
                joint_elem.set("range", " ".join(map(str, joint.range)))
        else:
            joint_elem.set("limited", "false")

        # Set damping and stiffness
        if joint.damping:
            joint_elem.set("damping", str(joint.damping))
        if joint.stiffness:
            joint_elem.set("stiffness", str(joint.stiffness))
        if joint.armature:
            joint_elem.set("armature", str(joint.armature))

        # Store in map
        self.joint_map[f"{body_id}_joint"] = joint_elem

    def _add_geom(self, body_elem: ET.Element, geom: Geom, geom_id: str):
        """Add geometry to body"""
        geom_elem = ET.SubElement(body_elem, "geom")
        geom_elem.set("name", geom_id)
        geom_elem.set("type", geom.type.value)

        # Set position
        if geom.pos:
            geom_elem.set("pos", " ".join(map(str, geom.pos)))

        # Set orientation
        if geom.quat:
            geom_elem.set("quat", " ".join(map(str, geom.quat)))

        # Set size parameters based on type
        if geom.type == GeomType.BOX and geom.size:
            geom_elem.set("size", " ".join(map(str, geom.size)))
        elif geom.type == GeomType.SPHERE and geom.size:
            geom_elem.set("size", str(geom.size[0]))
        elif geom.type in [GeomType.CAPSULE, GeomType.CYLINDER]:
            if geom.fromto:
                geom_elem.set("fromto", " ".join(map(str, geom.fromto)))
            if geom.size:
                geom_elem.set("size", str(geom.size[0]))
        elif geom.type == GeomType.ELLIPSOID and geom.size:
            geom_elem.set("size", " ".join(map(str, geom.size)))
        elif geom.type == GeomType.MESH and geom.mesh:
            geom_elem.set("mesh", geom.mesh)

        # Set material properties
        if geom.material:
            mat = geom.material
            if mat.rgba:
                geom_elem.set("rgba", " ".join(map(str, mat.rgba)))
            if mat.texture:
                geom_elem.set("material", f"mat_{mat.texture}")

        # Set friction
        if geom.friction:
            fric = geom.friction
            geom_elem.set("friction", f"{fric.slide} {fric.spin} {fric.roll}")

        # Set density
        if geom.density:
            geom_elem.set("density", str(geom.density))

        # Store in map
        self.geom_map[geom_id] = geom_elem

    def _add_actuators(self, mujoco: ET.Element, actuators: List[Actuator]):
        """Add actuators section"""
        actuator_elem = ET.SubElement(mujoco, "actuator")

        for actuator in actuators:
            # Map actuator types
            if actuator.type == ActuatorType.MOTOR:
                act = ET.SubElement(actuator_elem, "motor")
            elif actuator.type == ActuatorType.POSITION:
                act = ET.SubElement(actuator_elem, "position")
            elif actuator.type == ActuatorType.VELOCITY:
                act = ET.SubElement(actuator_elem, "velocity")
            else:
                act = ET.SubElement(actuator_elem, "general")

            act.set("name", actuator.id)
            act.set("joint", actuator.target)
            act.set("gear", str(actuator.gear))

            if actuator.ctrlrange:
                act.set("ctrlrange", " ".join(map(str, actuator.ctrlrange)))
            if actuator.forcerange:
                act.set("forcerange", " ".join(map(str, actuator.forcerange)))

    def _add_sensors(self, mujoco: ET.Element, sensors: List[Sensor]):
        """Add sensors section"""
        sensor_elem = ET.SubElement(mujoco, "sensor")

        for sensor in sensors:
            # Create sensor based on type
            sens = ET.SubElement(sensor_elem, sensor.type.value)

            if sensor.id:
                sens.set("name", sensor.id)
            else:
                sens.set("name", f"{sensor.type.value}_{sensor.source}")

            # Set source based on sensor type
            if sensor.type in [SensorType.JOINTPOS, SensorType.JOINTVEL]:
                sens.set("joint", sensor.source)
            elif sensor.type in [SensorType.ACTUATORPOS, SensorType.ACTUATORVEL, SensorType.ACTUATORFRC]:
                sens.set("actuator", sensor.source)
            elif sensor.type in [SensorType.FRAMEPOS, SensorType.FRAMEQUAT,
                                SensorType.ACCELEROMETER, SensorType.VELOCIMETER, SensorType.GYRO]:
                sens.set("body", sensor.source)

            # Add noise if specified
            if sensor.noise:
                sens.set("noise", str(sensor.noise))

    def _add_contacts(self, mujoco: ET.Element, contacts: List[Contact]):
        """Add contact pair overrides"""
        contact_elem = ET.SubElement(mujoco, "contact")

        for contact in contacts:
            if contact.exclude:
                exclude = ET.SubElement(contact_elem, "exclude")
                exclude.set("body1", contact.pair[0])
                exclude.set("body2", contact.pair[1])
            else:
                pair = ET.SubElement(contact_elem, "pair")
                pair.set("geom1", contact.pair[0])
                pair.set("geom2", contact.pair[1])

                if contact.friction:
                    fric = contact.friction
                    pair.set("friction", f"{fric.slide} {fric.spin} {fric.roll}")

                if contact.solref:
                    pair.set("solref", " ".join(map(str, contact.solref)))

    def _add_equality(self, mujoco: ET.Element, equality: List):
        """Add equality constraints"""
        equality_elem = ET.SubElement(mujoco, "equality")

        for eq in equality:
            if eq.type == "joint":
                joint_eq = ET.SubElement(equality_elem, "joint")
                joint_eq.set("joint1", eq.body1)
                if eq.body2:
                    joint_eq.set("joint2", eq.body2)
                if eq.polycoef:
                    joint_eq.set("polycoef", " ".join(map(str, eq.polycoef)))
            elif eq.type == "weld":
                weld = ET.SubElement(equality_elem, "weld")
                weld.set("body1", eq.body1)
                weld.set("body2", eq.body2)

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Convert XML element to pretty-printed string"""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty.split('\n') if line.strip()]
        # Skip XML declaration
        if lines[0].startswith('<?xml'):
            lines = lines[1:]

        return '\n'.join(lines)