"""
Starter game templates for VirtualForge.
Pre-built game specs that users can customize.
"""

from typing import List, Dict
from .models import GameSpec, SpriteAsset, GameEntity, GameBehavior, GameMechanic, GameRule, UIElement, Assets, WorldConfig, PhysicsConfig

# ============================================================================
# Template 1: Coin Collector (Platformer)
# ============================================================================

COIN_COLLECTOR_TEMPLATE = GameSpec(
    version="1.0",
    gameType="platformer",
    title="Coin Collector",
    description="Jump around platforms and collect all the coins!",

    world=WorldConfig(
        width=800,
        height=600,
        gravity=800,
        backgroundColor="#87CEEB",
        camera={"follow": "player", "bounds": True}
    ),

    assets=Assets(
        sprites=[
            SpriteAsset(id="player", type="sprite", source="placeholder", width=32, height=48),
            SpriteAsset(id="platform", type="sprite", source="placeholder", width=200, height=32),
            SpriteAsset(id="coin", type="sprite", source="placeholder", width=24, height=24),
        ]
    ),

    entities=[
        GameEntity(
            id="player",
            type="player",
            sprite="player",
            x=100,
            y=400,
            physics=PhysicsConfig(enabled=True, bounce=0.2)
        ),
        GameEntity(
            id="ground",
            type="platform",
            sprite="platform",
            x=400,
            y=568,
            width=800,
            height=32,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="platform1",
            type="platform",
            sprite="platform",
            x=300,
            y=450,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="platform2",
            type="platform",
            sprite="platform",
            x=500,
            y=350,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="coin1",
            type="item",
            sprite="coin",
            x=300,
            y=400,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="coin2",
            type="item",
            sprite="coin",
            x=500,
            y=300,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="coin3",
            type="item",
            sprite="coin",
            x=700,
            y=500,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
    ],

    behaviors=[
        GameBehavior(
            id="player_movement",
            type="movement_keyboard",
            entityId="player",
            config={"keys": "arrows", "speed": 200}
        ),
        GameBehavior(
            id="player_jump",
            type="jump",
            entityId="player",
            config={"key": "space", "velocity": -400, "doubleJump": False}
        ),
        GameBehavior(
            id="collect_coins",
            type="collect",
            entityId="player",
            config={
                "targets": ["coin1", "coin2", "coin3"],
                "scoreValue": 10,
                "destroyOnCollect": True
            }
        ),
    ],

    mechanics=[
        GameMechanic(
            type="score_system",
            config={"initialScore": 0, "displayPosition": {"x": 20, "y": 20}}
        )
    ],

    rules=[
        GameRule(
            type="win",
            condition={"type": "score_reaches", "value": 30},
            action="showMessage"
        )
    ],

    ui=[]
)


# ============================================================================
# Template 2: Dungeon Explorer (Top-down)
# ============================================================================

DUNGEON_EXPLORER_TEMPLATE = GameSpec(
    version="1.0",
    gameType="topdown",
    title="Dungeon Explorer",
    description="Navigate the dungeon and find the treasure!",

    world=WorldConfig(
        width=800,
        height=600,
        gravity=0,  # No gravity in top-down
        backgroundColor="#2d3436",
        camera={"follow": "player", "bounds": True}
    ),

    assets=Assets(
        sprites=[
            SpriteAsset(id="player", type="sprite", source="placeholder", width=32, height=32),
            SpriteAsset(id="wall", type="sprite", source="placeholder", width=32, height=32),
            SpriteAsset(id="key", type="sprite", source="placeholder", width=24, height=24),
            SpriteAsset(id="treasure", type="sprite", source="placeholder", width=48, height=48),
        ]
    ),

    entities=[
        GameEntity(
            id="player",
            type="player",
            sprite="player",
            x=100,
            y=100,
            physics=PhysicsConfig(enabled=True)
        ),
        # Walls
        GameEntity(
            id="wall_top",
            type="platform",
            sprite="wall",
            x=400,
            y=16,
            width=800,
            height=32,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="wall_bottom",
            type="platform",
            sprite="wall",
            x=400,
            y=584,
            width=800,
            height=32,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="wall_left",
            type="platform",
            sprite="wall",
            x=16,
            y=300,
            width=32,
            height=600,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        GameEntity(
            id="wall_right",
            type="platform",
            sprite="wall",
            x=784,
            y=300,
            width=32,
            height=600,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        # Key
        GameEntity(
            id="key",
            type="item",
            sprite="key",
            x=400,
            y=300,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
        # Treasure
        GameEntity(
            id="treasure",
            type="item",
            sprite="treasure",
            x=700,
            y=500,
            physics=PhysicsConfig(enabled=True, static=True)
        ),
    ],

    behaviors=[
        GameBehavior(
            id="player_movement",
            type="movement_keyboard",
            entityId="player",
            config={"keys": "wasd", "speed": 150}
        ),
        GameBehavior(
            id="collect_key",
            type="collect",
            entityId="player",
            config={
                "targets": ["key"],
                "scoreValue": 50,
                "destroyOnCollect": True
            }
        ),
        GameBehavior(
            id="collect_treasure",
            type="collect",
            entityId="player",
            config={
                "targets": ["treasure"],
                "scoreValue": 100,
                "destroyOnCollect": False
            }
        ),
    ],

    mechanics=[
        GameMechanic(
            type="score_system",
            config={"initialScore": 0, "displayPosition": {"x": 20, "y": 20}}
        )
    ],

    rules=[
        GameRule(
            type="win",
            condition={"type": "score_reaches", "value": 150},
            action="showMessage"
        )
    ],

    ui=[]
)


# ============================================================================
# Template 3: Space Shooter (Shooter)
# ============================================================================

SPACE_SHOOTER_TEMPLATE = GameSpec(
    version="1.0",
    gameType="shooter",
    title="Space Defender",
    description="Defend Earth from alien invaders!",

    world=WorldConfig(
        width=800,
        height=600,
        gravity=0,
        backgroundColor="#0a0e27",
        camera=None
    ),

    assets=Assets(
        sprites=[
            SpriteAsset(id="player", type="sprite", source="placeholder", width=48, height=48),
            SpriteAsset(id="enemy", type="sprite", source="placeholder", width=32, height=32),
            SpriteAsset(id="bullet", type="sprite", source="placeholder", width=8, height=16),
        ]
    ),

    entities=[
        GameEntity(
            id="player",
            type="player",
            sprite="player",
            x=400,
            y=500,
            physics=PhysicsConfig(enabled=True)
        ),
        GameEntity(
            id="enemy1",
            type="enemy",
            sprite="enemy",
            x=200,
            y=100,
            physics=PhysicsConfig(enabled=True)
        ),
        GameEntity(
            id="enemy2",
            type="enemy",
            sprite="enemy",
            x=400,
            y=100,
            physics=PhysicsConfig(enabled=True)
        ),
        GameEntity(
            id="enemy3",
            type="enemy",
            sprite="enemy",
            x=600,
            y=100,
            physics=PhysicsConfig(enabled=True)
        ),
    ],

    behaviors=[
        GameBehavior(
            id="player_movement",
            type="movement_keyboard",
            entityId="player",
            config={"keys": "arrows", "speed": 250}
        ),
        GameBehavior(
            id="player_shoot",
            type="shoot",
            entityId="player",
            config={
                "key": "space",
                "projectileSprite": "bullet",
                "projectileSpeed": 500,
                "cooldown": 300,
                "direction": "right"
            }
        ),
        GameBehavior(
            id="enemy1_patrol",
            type="patrol",
            entityId="enemy1",
            config={
                "points": [{"x": 200, "y": 100}, {"x": 200, "y": 300}],
                "speed": 100,
                "loop": True
            }
        ),
    ],

    mechanics=[
        GameMechanic(
            type="score_system",
            config={"initialScore": 0, "displayPosition": {"x": 20, "y": 20}}
        ),
        GameMechanic(
            type="health_system",
            config={"maxHealth": 100, "displayPosition": {"x": 20, "y": 50}}
        ),
    ],

    rules=[
        GameRule(
            type="win",
            condition={"type": "score_reaches", "value": 300},
            action="showMessage"
        ),
        GameRule(
            type="lose",
            condition={"type": "health_zero"},
            action="showMessage"
        ),
    ],

    ui=[]
)


# ============================================================================
# Template Registry
# ============================================================================

TEMPLATES: Dict[str, GameSpec] = {
    "coin-collector": COIN_COLLECTOR_TEMPLATE,
    "dungeon-explorer": DUNGEON_EXPLORER_TEMPLATE,
    "space-shooter": SPACE_SHOOTER_TEMPLATE,
}


def get_template(template_id: str) -> GameSpec | None:
    """Get a template by ID."""
    return TEMPLATES.get(template_id)


def get_all_templates() -> List[Dict]:
    """Get all templates with metadata."""
    return [
        {
            "id": "coin-collector",
            "name": "Coin Collector",
            "description": "A simple platformer where you collect coins",
            "gameType": "platformer",
            "difficulty": "beginner",
            "thumbnail": None,
        },
        {
            "id": "dungeon-explorer",
            "name": "Dungeon Explorer",
            "description": "Navigate a dungeon and find the treasure",
            "gameType": "topdown",
            "difficulty": "beginner",
            "thumbnail": None,
        },
        {
            "id": "space-shooter",
            "name": "Space Defender",
            "description": "Shoot aliens and defend Earth",
            "gameType": "shooter",
            "difficulty": "intermediate",
            "thumbnail": None,
        },
    ]


def get_templates_by_type(game_type: str) -> List[Dict]:
    """Get templates filtered by game type."""
    all_templates = get_all_templates()
    return [t for t in all_templates if t["gameType"] == game_type]
