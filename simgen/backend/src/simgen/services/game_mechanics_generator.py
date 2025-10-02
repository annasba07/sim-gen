"""
Game Mechanics Generator - Evolution from Physics to Full Games
Extends SimGen AI to generate interactive game mechanics beyond physics
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class GameType(Enum):
    """Types of games we can generate"""
    PLATFORMER = "platformer"
    PUZZLE = "puzzle"
    RACING = "racing"
    SURVIVAL = "survival"
    SANDBOX = "sandbox"
    EDUCATIONAL = "educational"
    RPG = "rpg"
    SPORTS = "sports"


class EntityType(Enum):
    """Building block entities"""
    PLAYER = "player"
    ENEMY = "enemy"
    COLLECTIBLE = "collectible"
    OBSTACLE = "obstacle"
    PLATFORM = "platform"
    TRIGGER = "trigger"
    NPC = "npc"
    VEHICLE = "vehicle"
    PROJECTILE = "projectile"


class MechanicType(Enum):
    """Game mechanics that can be applied"""
    MOVEMENT = "movement"
    JUMPING = "jumping"
    SHOOTING = "shooting"
    COLLECTING = "collecting"
    HEALTH = "health"
    SCORING = "scoring"
    INVENTORY = "inventory"
    DIALOGUE = "dialogue"
    CRAFTING = "crafting"
    BUILDING = "building"


@dataclass
class Entity:
    """Represents a game entity"""
    id: str
    type: EntityType
    position: Tuple[float, float, float]
    properties: Dict[str, Any]
    behaviors: List[str]
    physics_enabled: bool = True


@dataclass
class Mechanic:
    """Represents a game mechanic"""
    id: str
    type: MechanicType
    trigger: str  # What activates this mechanic
    action: str  # What happens
    parameters: Dict[str, Any]


@dataclass
class GameRule:
    """Represents a game rule or win condition"""
    id: str
    condition: str
    action: str
    priority: int = 0


@dataclass
class Level:
    """Represents a game level"""
    id: str
    name: str
    entities: List[Entity]
    mechanics: List[Mechanic]
    rules: List[GameRule]
    environment: Dict[str, Any]
    objectives: List[str]


@dataclass
class GameSpec:
    """Complete game specification"""
    name: str
    type: GameType
    description: str
    levels: List[Level]
    global_mechanics: List[Mechanic]
    global_rules: List[GameRule]
    player_config: Dict[str, Any]
    ui_elements: List[str]
    assets_needed: List[str]


class GameMechanicsGenerator:
    """
    Generates complete game specifications from prompts and sketches.
    Extends SimGen's physics capabilities to full game creation.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.template_library = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, GameSpec]:
        """Initialize game templates as building blocks"""
        return {
            "simple_platformer": self._create_platformer_template(),
            "physics_puzzle": self._create_puzzle_template(),
            "survival_game": self._create_survival_template(),
            "educational_sim": self._create_educational_template()
        }

    async def generate_from_prompt(
        self,
        prompt: str,
        sketch_analysis: Optional[Dict] = None,
        game_type: Optional[GameType] = None
    ) -> GameSpec:
        """
        Generate a complete game specification from natural language.

        Examples:
        - "Create a platformer where you collect gems and avoid enemies"
        - "Make a puzzle game with physics and magnets"
        - "Build an educational game about ecosystems"
        """
        # Analyze prompt to determine game type and requirements
        if not game_type:
            game_type = await self._identify_game_type(prompt)

        # Extract key elements from prompt
        entities = await self._extract_entities(prompt, sketch_analysis)
        mechanics = await self._identify_mechanics(prompt, game_type)
        objectives = await self._determine_objectives(prompt, game_type)

        # Generate game specification
        game_spec = await self._build_game_spec(
            prompt=prompt,
            game_type=game_type,
            entities=entities,
            mechanics=mechanics,
            objectives=objectives,
            sketch_analysis=sketch_analysis
        )

        return game_spec

    async def _identify_game_type(self, prompt: str) -> GameType:
        """Identify the type of game from the prompt"""
        prompt_lower = prompt.lower()

        # Simple keyword matching (would use LLM in production)
        if any(word in prompt_lower for word in ["platform", "jump", "mario", "sonic"]):
            return GameType.PLATFORMER
        elif any(word in prompt_lower for word in ["puzzle", "solve", "physics", "angry birds"]):
            return GameType.PUZZLE
        elif any(word in prompt_lower for word in ["race", "racing", "drive", "car"]):
            return GameType.RACING
        elif any(word in prompt_lower for word in ["survive", "survival", "zombie", "craft"]):
            return GameType.SURVIVAL
        elif any(word in prompt_lower for word in ["learn", "education", "teach", "student"]):
            return GameType.EDUCATIONAL
        elif any(word in prompt_lower for word in ["rpg", "quest", "adventure", "story"]):
            return GameType.RPG
        else:
            return GameType.SANDBOX

    async def _extract_entities(
        self,
        prompt: str,
        sketch_analysis: Optional[Dict]
    ) -> List[Entity]:
        """Extract game entities from prompt and sketch"""
        entities = []

        # Always add a player
        entities.append(Entity(
            id="player_1",
            type=EntityType.PLAYER,
            position=(0, 0, 0),
            properties={"health": 100, "speed": 5},
            behaviors=["controllable", "physics"],
            physics_enabled=True
        ))

        # Parse prompt for entity mentions
        if "enemy" in prompt.lower() or "enemies" in prompt.lower():
            for i in range(3):  # Add some enemies
                entities.append(Entity(
                    id=f"enemy_{i}",
                    type=EntityType.ENEMY,
                    position=(10 + i * 5, 0, 0),
                    properties={"health": 50, "damage": 10, "ai": "patrol"},
                    behaviors=["ai_controlled", "hostile"],
                    physics_enabled=True
                ))

        if "collect" in prompt.lower() or "gem" in prompt.lower() or "coin" in prompt.lower():
            for i in range(5):  # Add collectibles
                entities.append(Entity(
                    id=f"collectible_{i}",
                    type=EntityType.COLLECTIBLE,
                    position=(i * 3, 2, 0),
                    properties={"value": 10, "respawns": False},
                    behaviors=["collectable", "spinning"],
                    physics_enabled=False
                ))

        # Add entities from sketch analysis if available
        if sketch_analysis and "objects" in sketch_analysis:
            for obj in sketch_analysis.get("objects", []):
                if obj.get("type") == "platform":
                    entities.append(Entity(
                        id=f"platform_{obj.get('id', '')}",
                        type=EntityType.PLATFORM,
                        position=tuple(obj.get("position", [0, 0, 0])),
                        properties={"solid": True, "material": "stone"},
                        behaviors=["static"],
                        physics_enabled=True
                    ))

        return entities

    async def _identify_mechanics(
        self,
        prompt: str,
        game_type: GameType
    ) -> List[Mechanic]:
        """Identify game mechanics from prompt and game type"""
        mechanics = []

        # Base mechanics for game type
        if game_type == GameType.PLATFORMER:
            mechanics.extend([
                Mechanic(
                    id="jump",
                    type=MechanicType.JUMPING,
                    trigger="space_key",
                    action="apply_impulse",
                    parameters={"force": [0, 10, 0], "cooldown": 0.5}
                ),
                Mechanic(
                    id="move",
                    type=MechanicType.MOVEMENT,
                    trigger="arrow_keys",
                    action="set_velocity",
                    parameters={"speed": 5, "acceleration": 10}
                )
            ])

        # Add mechanics from prompt
        if "shoot" in prompt.lower() or "fire" in prompt.lower():
            mechanics.append(Mechanic(
                id="shoot",
                type=MechanicType.SHOOTING,
                trigger="mouse_click",
                action="spawn_projectile",
                parameters={
                    "projectile_type": "bullet",
                    "speed": 20,
                    "damage": 10,
                    "cooldown": 0.3
                }
            ))

        if "collect" in prompt.lower():
            mechanics.append(Mechanic(
                id="collect",
                type=MechanicType.COLLECTING,
                trigger="collision",
                action="add_to_inventory",
                parameters={"auto_collect": True, "sound": "coin.wav"}
            ))

        # Always add scoring
        mechanics.append(Mechanic(
            id="scoring",
            type=MechanicType.SCORING,
            trigger="event",
            action="update_score",
            parameters={"display": True, "save_highscore": True}
        ))

        return mechanics

    async def _determine_objectives(
        self,
        prompt: str,
        game_type: GameType
    ) -> List[str]:
        """Determine game objectives from prompt"""
        objectives = []

        # Parse prompt for objective keywords
        if "collect all" in prompt.lower():
            objectives.append("Collect all items in the level")
        if "defeat" in prompt.lower() or "kill" in prompt.lower():
            objectives.append("Defeat all enemies")
        if "reach" in prompt.lower() or "get to" in prompt.lower():
            objectives.append("Reach the end of the level")
        if "survive" in prompt.lower():
            objectives.append("Survive for as long as possible")
        if "solve" in prompt.lower():
            objectives.append("Solve the puzzle")

        # Default objective if none found
        if not objectives:
            if game_type == GameType.PLATFORMER:
                objectives.append("Reach the goal")
            elif game_type == GameType.PUZZLE:
                objectives.append("Complete the puzzle")
            else:
                objectives.append("Complete the level")

        return objectives

    async def _build_game_spec(
        self,
        prompt: str,
        game_type: GameType,
        entities: List[Entity],
        mechanics: List[Mechanic],
        objectives: List[str],
        sketch_analysis: Optional[Dict]
    ) -> GameSpec:
        """Build complete game specification"""

        # Create initial level
        level = Level(
            id="level_1",
            name="Level 1",
            entities=entities,
            mechanics=mechanics,
            rules=[
                GameRule(
                    id="win_condition",
                    condition="all_objectives_complete",
                    action="next_level",
                    priority=10
                ),
                GameRule(
                    id="game_over",
                    condition="player_health <= 0",
                    action="restart_level",
                    priority=20
                )
            ],
            environment={
                "gravity": -9.8 if game_type != GameType.PLATFORMER else -20,
                "bounds": "finite",
                "theme": "default",
                "lighting": "dynamic"
            },
            objectives=objectives
        )

        # Create game specification
        game_spec = GameSpec(
            name=self._generate_game_name(prompt, game_type),
            type=game_type,
            description=prompt,
            levels=[level],
            global_mechanics=mechanics,
            global_rules=[
                GameRule(
                    id="pause",
                    condition="escape_key",
                    action="pause_game",
                    priority=100
                )
            ],
            player_config={
                "lives": 3,
                "start_health": 100,
                "respawn": True,
                "camera": "follow" if game_type == GameType.PLATFORMER else "free"
            },
            ui_elements=["score", "health", "lives", "objectives"],
            assets_needed=self._determine_required_assets(entities, mechanics)
        )

        return game_spec

    def _generate_game_name(self, prompt: str, game_type: GameType) -> str:
        """Generate a game name from prompt"""
        # Simple extraction (would use LLM in production)
        words = prompt.split()[:3]
        return f"{game_type.value.title()}: {' '.join(words)}"

    def _determine_required_assets(
        self,
        entities: List[Entity],
        mechanics: List[Mechanic]
    ) -> List[str]:
        """Determine what assets are needed"""
        assets = set()

        # Add assets for entities
        for entity in entities:
            assets.add(f"model_{entity.type.value}")
            assets.add(f"texture_{entity.type.value}")

        # Add assets for mechanics
        for mechanic in mechanics:
            if mechanic.type == MechanicType.SHOOTING:
                assets.add("projectile_model")
                assets.add("muzzle_flash_effect")
            elif mechanic.type == MechanicType.COLLECTING:
                assets.add("collect_sound")
                assets.add("sparkle_effect")

        return list(assets)

    def _create_platformer_template(self) -> GameSpec:
        """Create a simple platformer template"""
        return GameSpec(
            name="Classic Platformer",
            type=GameType.PLATFORMER,
            description="A classic platform jumping game",
            levels=[],
            global_mechanics=[],
            global_rules=[],
            player_config={"lives": 3, "camera": "follow"},
            ui_elements=["score", "lives", "timer"],
            assets_needed=["player_sprite", "platform_tiles", "background"]
        )

    def _create_puzzle_template(self) -> GameSpec:
        """Create a physics puzzle template"""
        return GameSpec(
            name="Physics Puzzle",
            type=GameType.PUZZLE,
            description="A physics-based puzzle game",
            levels=[],
            global_mechanics=[],
            global_rules=[],
            player_config={"moves_limit": 10, "camera": "fixed"},
            ui_elements=["moves", "score", "reset"],
            assets_needed=["physics_objects", "goal_marker"]
        )

    def _create_survival_template(self) -> GameSpec:
        """Create a survival game template"""
        return GameSpec(
            name="Survival Challenge",
            type=GameType.SURVIVAL,
            description="Survive waves of enemies",
            levels=[],
            global_mechanics=[],
            global_rules=[],
            player_config={"health": 100, "inventory_size": 20},
            ui_elements=["health", "inventory", "wave_counter"],
            assets_needed=["player_model", "enemy_models", "weapons"]
        )

    def _create_educational_template(self) -> GameSpec:
        """Create an educational game template"""
        return GameSpec(
            name="Learning Adventure",
            type=GameType.EDUCATIONAL,
            description="Learn through interactive gameplay",
            levels=[],
            global_mechanics=[],
            global_rules=[],
            player_config={"hints_enabled": True, "progress_tracking": True},
            ui_elements=["progress", "hints", "score", "achievements"],
            assets_needed=["educational_content", "reward_animations"]
        )

    async def export_to_engine(
        self,
        game_spec: GameSpec,
        engine: str
    ) -> Dict[str, Any]:
        """Export game spec to specific engine format"""
        if engine == "mujoco":
            return await self._export_to_mujoco(game_spec)
        elif engine == "unity":
            return await self._export_to_unity(game_spec)
        elif engine == "babylon":
            return await self._export_to_babylon(game_spec)
        elif engine == "roblox":
            return await self._export_to_roblox(game_spec)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    async def _export_to_mujoco(self, game_spec: GameSpec) -> Dict[str, Any]:
        """Export to MuJoCo (physics only)"""
        # Extract physics entities
        physics_bodies = []
        for level in game_spec.levels:
            for entity in level.entities:
                if entity.physics_enabled:
                    physics_bodies.append({
                        "name": entity.id,
                        "type": "box" if entity.type == EntityType.PLATFORM else "sphere",
                        "pos": entity.position,
                        "mass": entity.properties.get("mass", 1.0)
                    })

        return {
            "format": "mjcf",
            "bodies": physics_bodies,
            "gravity": game_spec.levels[0].environment.get("gravity", -9.8)
        }

    async def _export_to_unity(self, game_spec: GameSpec) -> Dict[str, Any]:
        """Export to Unity format (C# scripts + scene)"""
        return {
            "format": "unity",
            "scene": asdict(game_spec),
            "scripts": self._generate_unity_scripts(game_spec),
            "prefabs": self._generate_unity_prefabs(game_spec)
        }

    async def _export_to_babylon(self, game_spec: GameSpec) -> Dict[str, Any]:
        """Export to Babylon.js format"""
        return {
            "format": "babylon",
            "scene": self._convert_to_babylon_scene(game_spec),
            "scripts": self._generate_babylon_scripts(game_spec)
        }

    async def _export_to_roblox(self, game_spec: GameSpec) -> Dict[str, Any]:
        """Export to Roblox Lua scripts"""
        return {
            "format": "roblox",
            "workspace": self._generate_roblox_workspace(game_spec),
            "scripts": self._generate_roblox_scripts(game_spec)
        }

    def _generate_unity_scripts(self, game_spec: GameSpec) -> List[str]:
        """Generate C# scripts for Unity"""
        # Placeholder - would generate actual C# code
        return ["PlayerController.cs", "GameManager.cs", "Enemy.cs"]

    def _generate_unity_prefabs(self, game_spec: GameSpec) -> List[Dict]:
        """Generate Unity prefab definitions"""
        # Placeholder - would generate actual prefab data
        return []

    def _convert_to_babylon_scene(self, game_spec: GameSpec) -> Dict:
        """Convert to Babylon.js scene format"""
        # Placeholder - would generate actual Babylon scene
        return {"meshes": [], "lights": [], "cameras": []}

    def _generate_babylon_scripts(self, game_spec: GameSpec) -> List[str]:
        """Generate JavaScript for Babylon.js"""
        # Placeholder - would generate actual JS code
        return []

    def _generate_roblox_workspace(self, game_spec: GameSpec) -> Dict:
        """Generate Roblox workspace structure"""
        # Placeholder - would generate actual Roblox workspace
        return {"Workspace": {}, "ServerScriptService": {}, "StarterGui": {}}

    def _generate_roblox_scripts(self, game_spec: GameSpec) -> List[str]:
        """Generate Lua scripts for Roblox"""
        # Placeholder - would generate actual Lua code
        return []


# Factory function
def create_game_generator(llm_client=None) -> GameMechanicsGenerator:
    """Create a game mechanics generator instance"""
    return GameMechanicsGenerator(llm_client=llm_client)