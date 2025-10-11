"""
Phaser 3 code generator.
Generates complete JavaScript code from parsed game specifications.
"""

import random
import logging
from typing import List, Dict, Any
from .models import GameSpec, Entity, Behavior, Mechanic, Rule, UIElement
from .behavior_templates.behaviors import BehaviorTemplates

logger = logging.getLogger(__name__)


class PhaserCodeGenerator:
    """Generates Phaser 3 game code from game specifications."""

    def __init__(self):
        self.behavior_templates = BehaviorTemplates()
        self.indent = "    "  # 4 spaces

    async def generate(self, spec: GameSpec) -> str:
        """
        Generate complete Phaser 3 game code.

        Args:
            spec: Validated game specification

        Returns:
            Complete JavaScript code as string
        """
        sections = []

        # Generate each section
        sections.append(self._generate_config(spec))
        sections.append(self._generate_scene_class(spec))
        sections.append(self._generate_preload(spec))
        sections.append(self._generate_create(spec))
        sections.append(self._generate_update(spec))
        sections.append(self._generate_helpers(spec))
        sections.append(self._generate_init())

        # Join with double newlines
        return "\n\n".join(sections)

    def _generate_config(self, spec: GameSpec) -> str:
        """Generate Phaser game configuration."""
        world = spec.world

        # Determine physics type based on game type
        physics_config = ""
        if spec.gameType in ["platformer", "topdown", "shooter"]:
            physics_config = f"""    physics: {{
        default: 'arcade',
        arcade: {{
            gravity: {{ y: {world.gravity} }},
            debug: false
        }}
    }},"""

        return f"""// ============================================================================
// Game Configuration
// ============================================================================
const config = {{
    type: Phaser.AUTO,
    width: {world.width},
    height: {world.height},
    backgroundColor: '{world.backgroundColor}',
{physics_config}
    scene: GameScene
}};"""

    def _generate_scene_class(self, spec: GameSpec) -> str:
        """Generate main game scene class declaration."""
        return f"""// ============================================================================
// Main Game Scene
// ============================================================================
class GameScene extends Phaser.Scene {{
    constructor() {{
        super('GameScene');

        // Game state
        this.score = 0;
        this.health = 100;
        this.gameState = 'playing'; // 'playing', 'won', 'lost'

        // Game objects will be added in create()
    }}

    preload() {{
        // Asset loading (implemented below)
    }}

    create() {{
        // Game setup (implemented below)
    }}

    update(time, delta) {{
        // Game loop (implemented below)
    }}
}}"""

    def _generate_preload(self, spec: GameSpec) -> str:
        """Generate asset preloading code."""
        lines = ["""    preload() {
        // Load game assets"""]

        # Generate placeholder sprites
        for sprite in spec.assets.sprites:
            if sprite.source == "placeholder":
                color = self._random_hex_color()
                lines.append(f"""
        // Placeholder for '{sprite.id}'
        const {sprite.id}Graphics = this.add.graphics();
        {sprite.id}Graphics.fillStyle(0x{color}, 1);
        {sprite.id}Graphics.fillRect(0, 0, {sprite.width}, {sprite.height});
        {sprite.id}Graphics.generateTexture('{sprite.id}', {sprite.width}, {sprite.height});
        {sprite.id}Graphics.destroy();""")

            elif sprite.source == "url":
                if sprite.type == "spritesheet":
                    lines.append(f"""
        this.load.spritesheet('{sprite.id}', '{sprite.url}', {{
            frameWidth: {sprite.frameWidth},
            frameHeight: {sprite.frameHeight}
        }});""")
                else:
                    lines.append(f"""
        this.load.image('{sprite.id}', '{sprite.url}');""")

        lines.append("    }")
        return "\n".join(lines)

    def _generate_create(self, spec: GameSpec) -> str:
        """Generate scene creation code."""
        lines = ["""    create() {
        // Initialize game world"""]

        # Camera setup
        if spec.world.camera:
            if spec.world.camera.bounds:
                lines.append(f"""
        this.cameras.main.setBounds(0, 0, {spec.world.width}, {spec.world.height});""")
            if spec.world.camera.follow:
                # Will be added after entity creation
                pass

        # Physics world bounds
        if spec.gameType in ["platformer", "topdown", "shooter"]:
            lines.append(f"""
        this.physics.world.setBounds(0, 0, {spec.world.width}, {spec.world.height});""")

        # Create entities
        lines.append("\n        // ========== Create Entities ==========")
        for entity in spec.entities:
            entity_code = self._generate_entity(entity, spec)
            lines.append(f"        {entity_code}")

        # Apply behaviors
        lines.append("\n        // ========== Setup Behaviors ==========")
        for behavior in spec.behaviors:
            behavior_code = self.behavior_templates.generate_setup(behavior, spec)
            if behavior_code:
                lines.append(behavior_code)

        # Setup mechanics
        lines.append("\n        // ========== Setup Mechanics ==========")
        for mechanic in spec.mechanics:
            mechanic_code = self._generate_mechanic(mechanic, spec)
            if mechanic_code:
                lines.append(mechanic_code)

        # Setup UI
        lines.append("\n        // ========== Setup UI ==========")
        for ui in spec.ui:
            ui_code = self._generate_ui(ui)
            if ui_code:
                lines.append(ui_code)

        # Camera follow (after entities created)
        if spec.world.camera and spec.world.camera.follow:
            lines.append(f"""
        this.cameras.main.startFollow(this.{spec.world.camera.follow});""")

        # Setup collisions
        lines.append(self._generate_collisions(spec))

        lines.append("    }")
        return "\n".join(lines)

    def _generate_entity(self, entity: Entity, spec: GameSpec) -> str:
        """Generate code for creating a single entity."""
        entity_id = entity.id
        sprite = entity.sprite
        x = entity.x
        y = entity.y

        if entity.physics and entity.physics.enabled:
            # Physics-enabled sprite
            code = f"this.{entity_id} = this.physics.add.sprite({x}, {y}, '{sprite}');"

            # Apply physics properties
            if entity.physics.static:
                code += f"\n        this.{entity_id}.setImmovable(true);"
                code += f"\n        this.{entity_id}.body.allowGravity = false;"

            if entity.physics.bounce > 0:
                code += f"\n        this.{entity_id}.setBounce({entity.physics.bounce});"

            if entity.physics.friction > 0:
                code += f"\n        this.{entity_id}.setFriction({entity.physics.friction});"

            if entity.physics.mass != 1.0:
                code += f"\n        this.{entity_id}.body.mass = {entity.physics.mass};"

            # Collide with world bounds for non-static entities
            if not entity.physics.static:
                code += f"\n        this.{entity_id}.setCollideWorldBounds(true);"

            return code
        else:
            # Regular sprite (no physics)
            return f"this.{entity_id} = this.add.sprite({x}, {y}, '{sprite}');"

    def _generate_collisions(self, spec: GameSpec) -> str:
        """Generate collision setup between entities."""
        lines = ["\n        // ========== Setup Collisions =========="]

        # Find player and platforms
        player_entities = [e for e in spec.entities if e.type == "player"]
        platform_entities = [e for e in spec.entities if e.type == "platform"]

        if player_entities and platform_entities:
            player = player_entities[0]
            for platform in platform_entities:
                lines.append(f"        this.physics.add.collider(this.{player.id}, this.{platform.id});")

        # Enemy collisions with platforms
        enemy_entities = [e for e in spec.entities if e.type == "enemy"]
        for enemy in enemy_entities:
            for platform in platform_entities:
                lines.append(f"        this.physics.add.collider(this.{enemy.id}, this.{platform.id});")

        return "\n".join(lines)

    def _generate_mechanic(self, mechanic: Mechanic, spec: GameSpec) -> str:
        """Generate code for game mechanics."""
        if mechanic.type == "score_system":
            config = mechanic.config
            pos = config.get("displayPosition", {"x": 20, "y": 20})
            return f"""        // Score system
        this.scoreText = this.add.text({pos['x']}, {pos['y']}, 'Score: 0', {{
            fontSize: '24px',
            fill: '#ffffff',
            fontFamily: 'Arial'
        }});
        this.scoreText.setScrollFactor(0);"""

        elif mechanic.type == "health_system":
            config = mechanic.config
            max_health = config.get("maxHealth", 100)
            pos = config.get("displayPosition", {"x": 20, "y": 50})
            self.health = max_health

            return f"""        // Health system
        this.health = {max_health};
        this.healthText = this.add.text({pos['x']}, {pos['y']}, 'Health: {max_health}', {{
            fontSize: '20px',
            fill: '#ff0000',
            fontFamily: 'Arial'
        }});
        this.healthText.setScrollFactor(0);"""

        elif mechanic.type == "timer":
            config = mechanic.config
            duration = config.get("duration", 60)
            count_down = config.get("countDown", True)
            pos = config.get("displayPosition", {"x": 400, "y": 20})

            return f"""        // Timer
        this.timeRemaining = {duration};
        this.timerText = this.add.text({pos['x']}, {pos['y']}, 'Time: {duration}', {{
            fontSize: '24px',
            fill: '#ffff00',
            fontFamily: 'Arial'
        }});
        this.timerText.setScrollFactor(0);

        this.timerEvent = this.time.addEvent({{
            delay: 1000,
            callback: () => {{
                this.timeRemaining--;
                this.timerText.setText('Time: ' + this.timeRemaining);
                if (this.timeRemaining <= 0) {{
                    this.gameLose();
                }}
            }},
            loop: true
        }});"""

        return ""

    def _generate_ui(self, ui: UIElement) -> str:
        """Generate UI element code."""
        if ui.type == "text":
            style = ui.style
            font_size = style.fontSize if style else "24px"
            fill = style.fill if style else "#ffffff"
            font_family = style.fontFamily if style else "Arial"

            return f"""        this.{ui.id} = this.add.text({ui.x}, {ui.y}, '{ui.content}', {{
            fontSize: '{font_size}',
            fill: '{fill}',
            fontFamily: '{font_family}'
        }});
        this.{ui.id}.setScrollFactor({ui.scrollFactor});"""

        return ""

    def _generate_update(self, spec: GameSpec) -> str:
        """Generate update loop code."""
        lines = ["""    update(time, delta) {
        // Exit if game is over
        if (this.gameState !== 'playing') return;"""]

        # Add behavior update code
        lines.append("\n        // ========== Behavior Updates ==========")
        for behavior in spec.behaviors:
            update_code = self.behavior_templates.generate_update(behavior, spec)
            if update_code:
                lines.append(update_code)

        # Check win/lose conditions
        lines.append("\n        // ========== Check Win/Lose Conditions ==========")
        for rule in spec.rules:
            rule_code = self._generate_rule_check(rule, spec)
            if rule_code:
                lines.append(rule_code)

        lines.append("    }")
        return "\n".join(lines)

    def _generate_rule_check(self, rule: Rule, spec: GameSpec) -> str:
        """Generate win/lose condition checking code."""
        condition = rule.condition
        rule_type = rule.type

        if condition.type == "score_reaches":
            target = condition.value
            return f"""        if (this.score >= {target}) {{
            this.game{rule_type.capitalize()}();
        }}"""

        elif condition.type == "health_zero":
            return f"""        if (this.health <= 0) {{
            this.game{rule_type.capitalize()}();
        }}"""

        elif condition.type == "time_reaches":
            target = condition.value
            return f"""        if (this.timeRemaining <= {target}) {{
            this.game{rule_type.capitalize()}();
        }}"""

        return ""

    def _generate_helpers(self, spec: GameSpec) -> str:
        """Generate helper functions."""
        return """    // ========================================================================
    // Helper Functions
    // ========================================================================

    gameWin() {
        this.gameState = 'won';

        const winText = this.add.text(
            this.cameras.main.centerX,
            this.cameras.main.centerY,
            'YOU WIN!',
            {
                fontSize: '64px',
                fill: '#00ff00',
                fontFamily: 'Arial',
                stroke: '#000000',
                strokeThickness: 6
            }
        );
        winText.setOrigin(0.5);
        winText.setScrollFactor(0);

        // Optional: Restart on click
        this.time.delayedCall(2000, () => {
            this.input.on('pointerdown', () => {
                this.scene.restart();
            });
        });
    }

    gameLose() {
        this.gameState = 'lost';

        const loseText = this.add.text(
            this.cameras.main.centerX,
            this.cameras.main.centerY,
            'GAME OVER',
            {
                fontSize: '64px',
                fill: '#ff0000',
                fontFamily: 'Arial',
                stroke: '#000000',
                strokeThickness: 6
            }
        );
        loseText.setOrigin(0.5);
        loseText.setScrollFactor(0);

        // Restart on click
        this.time.delayedCall(2000, () => {
            this.input.on('pointerdown', () => {
                this.scene.restart();
            });
        });
    }

    addScore(points) {
        this.score += points;
        if (this.scoreText) {
            this.scoreText.setText('Score: ' + this.score);
        }
    }

    takeDamage(amount) {
        this.health -= amount;
        if (this.healthText) {
            this.healthText.setText('Health: ' + this.health);
        }
        if (this.health <= 0) {
            this.gameLose();
        }
    }"""

    def _generate_init(self) -> str:
        """Generate game initialization code."""
        return """// ============================================================================
// Initialize Game
// ============================================================================
const game = new Phaser.Game(config);"""

    def _random_hex_color(self) -> str:
        """Generate a random hex color for placeholder sprites."""
        colors = [
            "FF6B6B",  # Red
            "4ECDC4",  # Teal
            "45B7D1",  # Blue
            "FFA07A",  # Orange
            "98D8C8",  # Mint
            "F7DC6F",  # Yellow
            "BB8FCE",  # Purple
            "85C1E2",  # Sky blue
        ]
        return random.choice(colors)
