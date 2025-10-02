"""
Mode Configuration System for VirtualForge
Manages different creation modes (physics, games, VR, etc.)
"""

from enum import Enum
from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass
from pydantic import BaseModel


class CreationMode(str, Enum):
    """Available creation modes"""
    PHYSICS = "physics"
    GAMES = "games"
    VR = "vr"  # Future


@dataclass
class ModeConfig:
    """Configuration for a creation mode"""
    id: str
    name: str
    description: str
    icon: str
    color: str
    enabled: bool
    beta: bool
    engines: List[str]
    features: List[str]
    target_users: List[str]


class ModeCompiler(Protocol):
    """Interface that all mode compilers must implement"""

    async def compile(self, spec: dict, options: dict = None) -> dict:
        """
        Compile a creation specification to mode-specific output.

        Args:
            spec: Unified creation specification
            options: Mode-specific compilation options

        Returns:
            Compiled output (MJCF, Phaser code, etc.)
        """
        ...

    async def validate(self, spec: dict) -> tuple[bool, List[str]]:
        """
        Validate a creation specification.

        Returns:
            (is_valid, errors)
        """
        ...


class ModeRegistry:
    """
    Central registry for all creation modes.
    Manages mode discovery, configuration, and compiler routing.
    """

    def __init__(self):
        self._modes: Dict[str, ModeConfig] = {}
        self._compilers: Dict[str, ModeCompiler] = {}
        self._initialize_modes()

    def _initialize_modes(self):
        """Initialize built-in modes"""

        # Physics Mode
        self.register_mode(ModeConfig(
            id="physics",
            name="Physics Lab",
            description="Scientific simulations & education",
            icon="🔬",
            color="blue",
            enabled=True,
            beta=False,
            engines=["mujoco"],
            features=[
                "sketch_analysis",
                "physics_compilation",
                "3d_visualization",
                "educational_templates",
                "real_time_simulation"
            ],
            target_users=[
                "educators",
                "students",
                "researchers",
                "engineers"
            ]
        ))

        # Games Mode
        self.register_mode(ModeConfig(
            id="games",
            name="Game Studio",
            description="60-second game creation",
            icon="🎮",
            color="purple",
            enabled=True,
            beta=True,  # New feature
            engines=["phaser", "babylon"],
            features=[
                "sketch_analysis",
                "game_compilation",
                "instant_preview",
                "remix_system",
                "template_library",
                "multi_engine_export"
            ],
            target_users=[
                "content_creators",
                "streamers",
                "kids",
                "hobbyists",
                "game_developers"
            ]
        ))

        # VR Mode (Future)
        self.register_mode(ModeConfig(
            id="vr",
            name="VR Worlds",
            description="Immersive virtual experiences",
            icon="🌐",
            color="green",
            enabled=False,  # Coming soon
            beta=True,
            engines=["babylonjs", "aframe", "threejs"],
            features=[
                "3d_modeling",
                "vr_interactions",
                "spatial_audio",
                "multiplayer_spaces"
            ],
            target_users=[
                "vr_creators",
                "educators",
                "artists",
                "architects"
            ]
        ))

    def register_mode(self, config: ModeConfig):
        """Register a new creation mode"""
        self._modes[config.id] = config

    def register_compiler(self, mode_id: str, compiler: ModeCompiler):
        """Register a compiler for a mode"""
        if mode_id not in self._modes:
            raise ValueError(f"Mode {mode_id} not registered")
        self._compilers[mode_id] = compiler

    def get_mode(self, mode_id: str) -> Optional[ModeConfig]:
        """Get mode configuration"""
        return self._modes.get(mode_id)

    def get_compiler(self, mode_id: str) -> Optional[ModeCompiler]:
        """Get compiler for a mode"""
        return self._compilers.get(mode_id)

    def get_enabled_modes(self) -> List[ModeConfig]:
        """Get all enabled modes"""
        return [m for m in self._modes.values() if m.enabled]

    def get_all_modes(self) -> List[ModeConfig]:
        """Get all modes (including disabled)"""
        return list(self._modes.values())

    def is_mode_available(self, mode_id: str) -> bool:
        """Check if a mode is available"""
        mode = self.get_mode(mode_id)
        return mode is not None and mode.enabled


# Global registry instance
mode_registry = ModeRegistry()


# Pydantic models for API
class ModeInfo(BaseModel):
    """Mode information for API responses"""
    id: str
    name: str
    description: str
    icon: str
    color: str
    beta: bool
    engines: List[str]
    features: List[str]
    available: bool


class CreationRequest(BaseModel):
    """Unified creation request for all modes"""
    mode: CreationMode
    prompt: str
    sketch_data: Optional[str] = None  # Base64 encoded
    options: Optional[Dict] = None


class CreationResponse(BaseModel):
    """Unified creation response"""
    success: bool
    mode: str
    creation_id: str
    output: Dict  # Mode-specific output
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None


def get_mode_info(mode_id: str) -> Optional[ModeInfo]:
    """Get mode information for API response"""
    mode = mode_registry.get_mode(mode_id)
    if not mode:
        return None

    return ModeInfo(
        id=mode.id,
        name=mode.name,
        description=mode.description,
        icon=mode.icon,
        color=mode.color,
        beta=mode.beta,
        engines=mode.engines,
        features=mode.features,
        available=mode.enabled
    )


def get_all_mode_info() -> List[ModeInfo]:
    """Get information for all modes"""
    return [get_mode_info(m.id) for m in mode_registry.get_all_modes()]
