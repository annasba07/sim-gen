"""
Behavior code generation templates.
Generates Phaser 3 code for different game behaviors.
"""

import json
from typing import Dict, Any, Optional
from ..models import Behavior, GameSpec


class BehaviorTemplates:
    """Generates Phaser code for game behaviors."""

    def generate_setup(self, behavior: Behavior, spec: GameSpec) -> str:
        """
        Generate setup code for a behavior (runs in create()).

        Args:
            behavior: Behavior specification
            spec: Complete game spec for context

        Returns:
            JavaScript code string
        """
        method_name = f"_setup_{behavior.type}"
        method = getattr(self, method_name, None)

        if method:
            return method(behavior, spec)
        else:
            return f"        // Unknown behavior: {behavior.type}"

    def generate_update(self, behavior: Behavior, spec: GameSpec) -> str:
        """
        Generate update code for a behavior (runs in update() loop).

        Args:
            behavior: Behavior specification
            spec: Complete game spec for context

        Returns:
            JavaScript code string
        """
        method_name = f"_update_{behavior.type}"
        method = getattr(self, method_name, None)

        if method:
            return method(behavior, spec)
        else:
            return ""

    # ========================================================================
    # Movement Behaviors
    # ========================================================================

    def _setup_movement_keyboard(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup keyboard movement controls."""
        config = behavior.config
        keys = config.get("keys", "arrows")

        if keys == "arrows":
            return """        // Keyboard controls (arrows)
        this.cursors = this.input.keyboard.createCursorKeys();"""
        elif keys == "wasd":
            return """        // Keyboard controls (WASD)
        this.wasd = this.input.keyboard.addKeys({
            up: 'W',
            down: 'S',
            left: 'A',
            right: 'D'
        });"""
        return ""

    def _update_movement_keyboard(self, behavior: Behavior, spec: GameSpec) -> str:
        """Update keyboard movement logic."""
        entity_id = behavior.entityId
        config = behavior.config
        speed = config.get("speed", 200)
        keys = config.get("keys", "arrows")
        acceleration = config.get("acceleration", False)

        key_var = "cursors" if keys == "arrows" else "wasd"

        if spec.gameType == "platformer":
            # Horizontal movement only (vertical handled by jump)
            if acceleration:
                return f"""        // Keyboard movement for {entity_id} (with acceleration)
        if (this.{key_var}.left.isDown) {{
            this.{entity_id}.setAccelerationX(-{speed});
        }} else if (this.{key_var}.right.isDown) {{
            this.{entity_id}.setAccelerationX({speed});
        }} else {{
            this.{entity_id}.setAccelerationX(0);
            this.{entity_id}.setVelocityX(this.{entity_id}.body.velocity.x * 0.9);
        }}"""
            else:
                return f"""        // Keyboard movement for {entity_id}
        if (this.{key_var}.left.isDown) {{
            this.{entity_id}.setVelocityX(-{speed});
        }} else if (this.{key_var}.right.isDown) {{
            this.{entity_id}.setVelocityX({speed});
        }} else {{
            this.{entity_id}.setVelocityX(0);
        }}"""
        else:
            # Top-down: all four directions
            return f"""        // Keyboard movement for {entity_id} (top-down)
        let velocityX = 0;
        let velocityY = 0;

        if (this.{key_var}.left.isDown) {{
            velocityX = -{speed};
        }} else if (this.{key_var}.right.isDown) {{
            velocityX = {speed};
        }}

        if (this.{key_var}.up.isDown) {{
            velocityY = -{speed};
        }} else if (this.{key_var}.down.isDown) {{
            velocityY = {speed};
        }}

        this.{entity_id}.setVelocity(velocityX, velocityY);"""

    # ========================================================================
    # Jump Behavior
    # ========================================================================

    def _setup_jump(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup jump controls."""
        entity_id = behavior.entityId
        config = behavior.config
        key = config.get("key", "space")
        double_jump = config.get("doubleJump", False)

        key_code = {
            "space": "SPACE",
            "up": "UP",
            "w": "W"
        }.get(key, "SPACE")

        extra_vars = ""
        if double_jump:
            extra_vars = f"\n        this.{entity_id}_jumpsRemaining = 2;"

        return f"""        // Jump controls for {entity_id}
        this.jumpKey = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.{key_code});{extra_vars}"""

    def _update_jump(self, behavior: Behavior, spec: GameSpec) -> str:
        """Update jump logic."""
        entity_id = behavior.entityId
        config = behavior.config
        velocity = config.get("velocity", -400)
        double_jump = config.get("doubleJump", False)

        if double_jump:
            return f"""        // Jump logic for {entity_id} (double jump enabled)
        if (Phaser.Input.Keyboard.JustDown(this.jumpKey)) {{
            if (this.{entity_id}.body.touching.down) {{
                this.{entity_id}.setVelocityY({velocity});
                this.{entity_id}_jumpsRemaining = 1;
            }} else if (this.{entity_id}_jumpsRemaining > 0) {{
                this.{entity_id}.setVelocityY({velocity});
                this.{entity_id}_jumpsRemaining--;
            }}
        }}

        // Reset jumps when landing
        if (this.{entity_id}.body.touching.down) {{
            this.{entity_id}_jumpsRemaining = 2;
        }}"""
        else:
            return f"""        // Jump logic for {entity_id}
        if (Phaser.Input.Keyboard.JustDown(this.jumpKey) && this.{entity_id}.body.touching.down) {{
            this.{entity_id}.setVelocityY({velocity});
        }}"""

    # ========================================================================
    # Collection Behavior
    # ========================================================================

    def _setup_collect(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup collection collision detection."""
        entity_id = behavior.entityId
        config = behavior.config
        targets = config.get("targets", [])
        score_value = config.get("scoreValue", 10)
        destroy = config.get("destroyOnCollect", True)
        sound = config.get("sound")

        code_lines = [f"        // Collection behavior for {entity_id}"]

        for target in targets:
            destroy_code = "item.destroy();" if destroy else "item.setVisible(false);"
            sound_code = f"this.sound.play('{sound}');" if sound else ""

            code_lines.append(f"""        this.physics.add.overlap(this.{entity_id}, this.{target}, (player, item) => {{
            this.addScore({score_value});
            {destroy_code}
            {sound_code}
        }}, null, this);""")

        return "\n".join(code_lines)

    # ========================================================================
    # Shooting Behavior
    # ========================================================================

    def _setup_shoot(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup shooting behavior."""
        entity_id = behavior.entityId
        config = behavior.config
        key = config.get("key", "space")
        cooldown = config.get("cooldown", 500)

        key_code = {
            "space": "SPACE",
            "z": "Z",
            "x": "X"
        }.get(key, "SPACE")

        return f"""        // Shooting behavior for {entity_id}
        this.shootKey = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.{key_code});
        this.projectiles = this.physics.add.group();
        this.canShoot = true;
        this.shootCooldown = {cooldown};"""

    def _update_shoot(self, behavior: Behavior, spec: GameSpec) -> str:
        """Update shooting logic."""
        entity_id = behavior.entityId
        config = behavior.config
        projectile_sprite = config.get("projectileSprite", "bullet")
        speed = config.get("projectileSpeed", 500)
        direction = config.get("direction", "facing")

        return f"""        // Shooting logic for {entity_id}
        if (Phaser.Input.Keyboard.JustDown(this.shootKey) && this.canShoot) {{
            const projectile = this.projectiles.create(
                this.{entity_id}.x + 20,
                this.{entity_id}.y,
                '{projectile_sprite}'
            );
            projectile.setVelocityX({speed});

            this.canShoot = false;
            this.time.delayedCall(this.shootCooldown, () => {{
                this.canShoot = true;
            }});
        }}"""

    # ========================================================================
    # Follow/Chase Behavior
    # ========================================================================

    def _setup_follow(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup follow/chase behavior."""
        # No setup needed, all in update
        return f"        // Follow behavior will run in update loop"

    def _update_follow(self, behavior: Behavior, spec: GameSpec) -> str:
        """Update follow/chase logic."""
        entity_id = behavior.entityId
        config = behavior.config
        target = config.get("target", "player")
        speed = config.get("speed", 100)
        min_distance = config.get("minDistance", 50)

        return f"""        // Follow behavior for {entity_id}
        const distance = Phaser.Math.Distance.Between(
            this.{entity_id}.x, this.{entity_id}.y,
            this.{target}.x, this.{target}.y
        );

        if (distance > {min_distance}) {{
            const angle = Phaser.Math.Angle.Between(
                this.{entity_id}.x, this.{entity_id}.y,
                this.{target}.x, this.{target}.y
            );
            this.{entity_id}.setVelocity(
                Math.cos(angle) * {speed},
                Math.sin(angle) * {speed}
            );
        }} else {{
            this.{entity_id}.setVelocity(0, 0);
        }}"""

    # ========================================================================
    # Patrol Behavior
    # ========================================================================

    def _setup_patrol(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup patrol behavior."""
        entity_id = behavior.entityId
        config = behavior.config
        points = config.get("points", [])

        points_str = json.dumps(points) if points else "[]"

        return f"""        // Patrol behavior for {entity_id}
        this.{entity_id}_patrolPoints = {points_str};
        this.{entity_id}_currentPatrolIndex = 0;"""

    def _update_patrol(self, behavior: Behavior, spec: GameSpec) -> str:
        """Update patrol logic."""
        entity_id = behavior.entityId
        config = behavior.config
        speed = config.get("speed", 100)
        loop = config.get("loop", True)

        loop_code = "this.{entity_id}_currentPatrolIndex = 0;" if loop else "return;"

        return f"""        // Patrol logic for {entity_id}
        if (this.{entity_id}_patrolPoints.length > 0) {{
            const targetPoint = this.{entity_id}_patrolPoints[this.{entity_id}_currentPatrolIndex];
            const distance = Phaser.Math.Distance.Between(
                this.{entity_id}.x, this.{entity_id}.y,
                targetPoint.x, targetPoint.y
            );

            if (distance < 10) {{
                this.{entity_id}_currentPatrolIndex++;
                if (this.{entity_id}_currentPatrolIndex >= this.{entity_id}_patrolPoints.length) {{
                    {loop_code}
                }}
            }} else {{
                const angle = Phaser.Math.Angle.Between(
                    this.{entity_id}.x, this.{entity_id}.y,
                    targetPoint.x, targetPoint.y
                );
                this.{entity_id}.setVelocity(
                    Math.cos(angle) * {speed},
                    Math.sin(angle) * {speed}
                );
            }}
        }}"""

    # ========================================================================
    # Utility Behaviors
    # ========================================================================

    def _setup_destroy_offscreen(self, behavior: Behavior, spec: GameSpec) -> str:
        """Setup destroy when offscreen."""
        # Handled in update
        return ""

    def _update_destroy_offscreen(self, behavior: Behavior, spec: GameSpec) -> str:
        """Destroy entity when it goes offscreen."""
        entity_id = behavior.entityId

        return f"""        // Destroy {entity_id} when offscreen
        if (this.{entity_id}.x < -100 || this.{entity_id}.x > {spec.world.width + 100} ||
            this.{entity_id}.y < -100 || this.{entity_id}.y > {spec.world.height + 100}) {{
            this.{entity_id}.destroy();
        }}"""
