"""
Pydantic models for game specification (JSON DSL).
Defines the schema that LLMs generate and the compiler consumes.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Asset Models
# ============================================================================

class SpriteAsset(BaseModel):
    """Sprite or spritesheet asset definition."""
    id: str = Field(..., description="Unique identifier for this sprite")
    type: Literal["sprite", "spritesheet"] = Field(default="sprite")
    source: Literal["placeholder", "generated", "url"] = Field(default="placeholder")
    url: Optional[str] = Field(None, description="URL if source is 'url'")
    width: int = Field(default=32, description="Sprite width in pixels")
    height: int = Field(default=32, description="Sprite height in pixels")
    frameWidth: Optional[int] = Field(None, description="Frame width for spritesheets")
    frameHeight: Optional[int] = Field(None, description="Frame height for spritesheets")


class SoundAsset(BaseModel):
    """Sound effect or music asset."""
    id: str
    source: Literal["placeholder", "url"] = "placeholder"
    url: Optional[str] = None
    volume: float = Field(default=1.0, ge=0.0, le=1.0)
    loop: bool = False


class Assets(BaseModel):
    """Collection of game assets."""
    sprites: List[SpriteAsset] = Field(default_factory=list)
    sounds: List[SoundAsset] = Field(default_factory=list)


# ============================================================================
# World Configuration
# ============================================================================

class CameraConfig(BaseModel):
    """Camera configuration."""
    follow: Optional[str] = Field(None, description="Entity ID to follow")
    bounds: bool = Field(default=True, description="Constrain camera to world bounds")


class WorldConfig(BaseModel):
    """Game world configuration."""
    width: int = Field(default=800, description="World width in pixels")
    height: int = Field(default=600, description="World height in pixels")
    gravity: int = Field(default=0, description="Physics gravity (0 = no gravity)")
    backgroundColor: str = Field(default="#87CEEB", description="Hex color")
    camera: Optional[CameraConfig] = None


# ============================================================================
# Entity Models
# ============================================================================

class PhysicsConfig(BaseModel):
    """Physics configuration for an entity."""
    enabled: bool = True
    static: bool = Field(default=False, description="Immovable object (platforms)")
    bounce: float = Field(default=0.0, ge=0.0, le=1.0)
    mass: float = Field(default=1.0, gt=0.0)
    friction: float = Field(default=0.0, ge=0.0, le=1.0)


class Entity(BaseModel):
    """Game entity (player, enemy, item, etc.)."""
    id: str = Field(..., description="Unique identifier")
    type: Literal["player", "enemy", "platform", "item", "decoration", "projectile"]
    sprite: str = Field(..., description="Sprite asset ID")
    x: float = Field(..., description="X position")
    y: float = Field(..., description="Y position")
    width: Optional[int] = None
    height: Optional[int] = None
    physics: Optional[PhysicsConfig] = None
    properties: Dict[str, Any] = Field(default_factory=dict, description="Custom properties")


# ============================================================================
# Behavior Models
# ============================================================================

class KeyboardMovementConfig(BaseModel):
    """Configuration for keyboard movement behavior."""
    keys: Literal["arrows", "wasd"] = "arrows"
    speed: int = Field(default=200, description="Movement speed in pixels/second")
    acceleration: bool = Field(default=False, description="Use acceleration instead of instant velocity")


class JumpConfig(BaseModel):
    """Configuration for jump behavior."""
    key: Literal["space", "up", "w"] = "space"
    velocity: int = Field(default=-400, description="Jump velocity (negative = up)")
    doubleJump: bool = Field(default=False, description="Allow double jump")


class CollectConfig(BaseModel):
    """Configuration for collection behavior."""
    targets: List[str] = Field(..., description="List of entity IDs to collect")
    scoreValue: int = Field(default=10, description="Points awarded")
    destroyOnCollect: bool = Field(default=True, description="Destroy collected items")
    sound: Optional[str] = Field(None, description="Sound to play on collect")


class ShootConfig(BaseModel):
    """Configuration for shooting behavior."""
    key: Literal["space", "z", "x"] = "space"
    projectileSprite: str = Field(..., description="Sprite ID for projectiles")
    projectileSpeed: int = Field(default=500, description="Projectile speed")
    cooldown: int = Field(default=500, description="Cooldown in milliseconds")
    direction: Literal["right", "left", "mouse", "facing"] = "facing"


class FollowConfig(BaseModel):
    """Configuration for follow/chase behavior."""
    target: str = Field(..., description="Entity ID to follow")
    speed: int = Field(default=100, description="Follow speed")
    minDistance: int = Field(default=50, description="Minimum distance to maintain")


class PatrolConfig(BaseModel):
    """Configuration for patrol behavior."""
    points: List[Dict[str, float]] = Field(..., description="List of {x, y} waypoints")
    speed: int = Field(default=100, description="Patrol speed")
    loop: bool = Field(default=True, description="Loop back to start")


class Behavior(BaseModel):
    """Behavior attached to an entity."""
    id: str = Field(..., description="Unique behavior identifier")
    type: Literal[
        "movement_keyboard",
        "movement_mouse",
        "jump",
        "shoot",
        "follow",
        "patrol",
        "collect",
        "damage",
        "animate",
        "destroy_offscreen"
    ]
    entityId: str = Field(..., description="Entity this behavior is attached to")
    config: Dict[str, Any] = Field(default_factory=dict, description="Behavior-specific config")


# ============================================================================
# Mechanic Models
# ============================================================================

class ScoreSystemConfig(BaseModel):
    """Score system configuration."""
    initialScore: int = 0
    displayPosition: Dict[str, int] = Field(default={"x": 20, "y": 20})


class HealthSystemConfig(BaseModel):
    """Health system configuration."""
    maxHealth: int = 100
    displayType: Literal["text", "bar", "hearts"] = "bar"
    displayPosition: Dict[str, int] = Field(default={"x": 20, "y": 50})


class TimerConfig(BaseModel):
    """Timer mechanic configuration."""
    duration: int = Field(..., description="Timer duration in seconds")
    countDown: bool = Field(default=True, description="Count down vs count up")
    displayPosition: Dict[str, int] = Field(default={"x": 400, "y": 20})


class SpawnSystemConfig(BaseModel):
    """Spawn system configuration."""
    entityTemplate: str = Field(..., description="Entity ID to use as template")
    interval: int = Field(..., description="Spawn interval in milliseconds")
    maxCount: int = Field(default=10, description="Maximum concurrent spawns")
    spawnPoints: List[Dict[str, float]] = Field(..., description="List of {x, y} spawn locations")


class Mechanic(BaseModel):
    """Game mechanic (global system)."""
    type: Literal["score_system", "health_system", "timer", "spawn_system", "level_progression"]
    config: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Rule Models
# ============================================================================

class RuleCondition(BaseModel):
    """Condition for win/lose rules."""
    type: Literal[
        "score_reaches",
        "time_reaches",
        "entity_collected",
        "entity_destroyed",
        "health_zero",
        "all_enemies_defeated"
    ]
    value: Optional[int] = None
    entityId: Optional[str] = None


class Rule(BaseModel):
    """Win or lose rule."""
    type: Literal["win", "lose"]
    condition: RuleCondition
    action: Literal["restart", "nextLevel", "showMessage", "returnToMenu"] = "showMessage"
    message: Optional[str] = None


# ============================================================================
# UI Models
# ============================================================================

class TextStyle(BaseModel):
    """Text styling configuration."""
    fontSize: str = "24px"
    fontFamily: str = "Arial"
    fill: str = "#ffffff"
    stroke: Optional[str] = None
    strokeThickness: int = 0


class UIElement(BaseModel):
    """UI element (text, sprite, bar)."""
    type: Literal["text", "sprite", "bar"]
    id: str
    x: int
    y: int
    content: str = Field(..., description="Text content or binding like 'Score: {score}'")
    style: Optional[TextStyle] = None
    scrollFactor: float = Field(default=0, description="0 = fixed to camera, 1 = moves with world")


# ============================================================================
# Main Game Specification
# ============================================================================

class GameSpec(BaseModel):
    """
    Complete game specification.
    This is the root model that defines an entire game.
    """
    version: str = Field(default="1.0", description="Spec version")
    gameType: Literal["platformer", "topdown", "puzzle", "shooter"]
    title: str = Field(..., description="Game title")
    description: Optional[str] = Field(None, description="Game description")

    world: WorldConfig
    assets: Assets = Field(default_factory=Assets)
    entities: List[Entity] = Field(default_factory=list)
    behaviors: List[Behavior] = Field(default_factory=list)
    mechanics: List[Mechanic] = Field(default_factory=list)
    rules: List[Rule] = Field(default_factory=list)
    ui: List[UIElement] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "version": "1.0",
                "gameType": "platformer",
                "title": "Jump Quest",
                "world": {
                    "width": 800,
                    "height": 600,
                    "gravity": 800,
                    "backgroundColor": "#87CEEB"
                },
                "assets": {
                    "sprites": [
                        {"id": "player", "type": "sprite", "source": "placeholder", "width": 32, "height": 48}
                    ]
                },
                "entities": [
                    {"id": "player", "type": "player", "sprite": "player", "x": 100, "y": 400}
                ]
            }
        }


# ============================================================================
# Compilation Result
# ============================================================================

class CompilationResult(BaseModel):
    """Result of game compilation."""
    success: bool
    code: str = Field(default="", description="Generated JavaScript code")
    html: str = Field(default="", description="Complete HTML page")
    assets: List[Dict[str, str]] = Field(default_factory=list, description="Asset URLs")
    errors: List[str] = Field(default_factory=list, description="Compilation errors")
    warnings: List[str] = Field(default_factory=list, description="Compilation warnings")
