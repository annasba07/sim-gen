"""
Models package for VirtualForge
Exports all model classes for easy importing
"""

from .physics_spec import (
    PhysicsSpec,
    PhysicsSpecVersion,
    JointType,
    GeomType,
    Material,
    Body,
    Joint,
    Actuator,
    Sensor,
    Geom
)

# Define CVResult here (avoiding import of heavy CV dependencies for testing)
from pydantic import BaseModel
from typing import List, Any

class CVResult(BaseModel):
    """Computer Vision analysis result"""
    objects: List[Any] = []
    text_annotations: List[str] = []
    confidence: float = 0.0

__all__ = [
    'PhysicsSpec',
    'PhysicsSpecVersion',
    'JointType',
    'GeomType',
    'Material',
    'Body',
    'Joint',
    'Actuator',
    'Sensor',
    'Geom',
    'CVResult',
]
