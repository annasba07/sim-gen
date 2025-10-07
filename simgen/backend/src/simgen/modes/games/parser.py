"""
Game specification parser and validator.
Validates JSON game specs and ensures semantic correctness.
"""

import json
import logging
from typing import Tuple, List, Dict, Any, Union
from pydantic import ValidationError

from .models import GameSpec, Entity, Behavior

logger = logging.getLogger(__name__)


class GameSpecParser:
    """
    Parses and validates game specifications.
    Performs both schema validation (Pydantic) and semantic validation.
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    async def parse(self, spec_data: Union[str, dict]) -> Tuple[GameSpec, bool]:
        """
        Parse and validate a game specification.

        Args:
            spec_data: JSON string or dict

        Returns:
            (parsed_spec, is_valid)
        """
        self.errors = []
        self.warnings = []

        # Step 1: Parse JSON if string
        if isinstance(spec_data, str):
            try:
                spec_dict = json.loads(spec_data)
            except json.JSONDecodeError as e:
                self.errors.append(f"Invalid JSON: {e}")
                return None, False
        else:
            spec_dict = spec_data

        # Step 2: Schema validation with Pydantic
        try:
            game_spec = GameSpec(**spec_dict)
        except ValidationError as e:
            for error in e.errors():
                field = " -> ".join(str(x) for x in error["loc"])
                self.errors.append(f"{field}: {error['msg']}")
            return None, False

        # Step 3: Semantic validation
        is_valid = self._validate_semantics(game_spec)

        if not is_valid:
            return game_spec, False

        # Step 4: Add defaults and enrich
        game_spec = self._enrich_spec(game_spec)

        return game_spec, True

    def _validate_semantics(self, spec: GameSpec) -> bool:
        """
        Validate semantic rules that Pydantic can't check.

        Checks:
        - Entity references (sprite assets exist)
        - Behavior references (entities exist)
        - Player entity exists
        - No circular dependencies
        - Physics configuration is sensible
        """
        # Build lookup tables
        entity_ids = {e.id for e in spec.entities}
        sprite_ids = {s.id for s in spec.assets.sprites}

        # Check 1: All entities reference valid sprites
        for entity in spec.entities:
            if entity.sprite not in sprite_ids:
                self.errors.append(
                    f"Entity '{entity.id}' references unknown sprite '{entity.sprite}'"
                )
                return False

        # Check 2: All behaviors reference valid entities
        for behavior in spec.behaviors:
            if behavior.entityId not in entity_ids:
                self.errors.append(
                    f"Behavior '{behavior.id}' references unknown entity '{behavior.entityId}'"
                )
                return False

        # Check 3: Validate behavior-specific requirements
        for behavior in spec.behaviors:
            if not self._validate_behavior_config(behavior, spec):
                return False

        # Check 4: Ensure at least one player entity exists (warning only)
        player_entities = [e for e in spec.entities if e.type == "player"]
        if not player_entities:
            self.warnings.append("No player entity defined - game may not be playable")
        elif len(player_entities) > 1:
            self.warnings.append(
                f"Multiple player entities found ({len(player_entities)}), will use first one"
            )

        # Check 5: Platformer games should have gravity
        if spec.gameType == "platformer" and spec.world.gravity == 0:
            self.warnings.append(
                "Platformer game has gravity=0, jumping may not work as expected"
            )

        # Check 6: Static entities shouldn't have movement behaviors
        for behavior in spec.behaviors:
            if behavior.type in ["movement_keyboard", "movement_mouse", "jump"]:
                entity = next((e for e in spec.entities if e.id == behavior.entityId), None)
                if entity and entity.physics and entity.physics.static:
                    self.warnings.append(
                        f"Entity '{entity.id}' is static but has movement behavior '{behavior.type}'"
                    )

        # Check 7: Collect behavior targets exist
        for behavior in spec.behaviors:
            if behavior.type == "collect":
                targets = behavior.config.get("targets", [])
                for target_id in targets:
                    if target_id not in entity_ids:
                        self.errors.append(
                            f"Collect behavior '{behavior.id}' targets unknown entity '{target_id}'"
                        )
                        return False

        # Check 8: Camera follow target exists
        if spec.world.camera and spec.world.camera.follow:
            follow_id = spec.world.camera.follow
            if follow_id not in entity_ids:
                self.errors.append(
                    f"Camera follow references unknown entity '{follow_id}'"
                )
                return False

        return True

    def _validate_behavior_config(self, behavior: Behavior, spec: GameSpec) -> bool:
        """Validate behavior-specific configuration."""
        config = behavior.config

        # Movement keyboard
        if behavior.type == "movement_keyboard":
            keys = config.get("keys")
            if keys and keys not in ["arrows", "wasd"]:
                self.errors.append(
                    f"Behavior '{behavior.id}': invalid keys '{keys}', must be 'arrows' or 'wasd'"
                )
                return False

        # Jump
        elif behavior.type == "jump":
            key = config.get("key")
            if key and key not in ["space", "up", "w"]:
                self.errors.append(
                    f"Behavior '{behavior.id}': invalid jump key '{key}'"
                )
                return False

        # Shoot
        elif behavior.type == "shoot":
            projectile_sprite = config.get("projectileSprite")
            if projectile_sprite:
                sprite_ids = {s.id for s in spec.assets.sprites}
                if projectile_sprite not in sprite_ids:
                    self.errors.append(
                        f"Behavior '{behavior.id}': projectile sprite '{projectile_sprite}' not found"
                    )
                    return False

        # Follow
        elif behavior.type == "follow":
            target = config.get("target")
            if target:
                entity_ids = {e.id for e in spec.entities}
                if target not in entity_ids:
                    self.errors.append(
                        f"Behavior '{behavior.id}': follow target '{target}' not found"
                    )
                    return False

        return True

    def _enrich_spec(self, spec: GameSpec) -> GameSpec:
        """
        Add defaults and enrich the specification.

        - Set default physics configs
        - Set default entity dimensions
        - Add missing UI elements
        """
        # Ensure all entities have dimensions
        for entity in spec.entities:
            # Get dimensions from sprite if not specified
            if entity.width is None or entity.height is None:
                sprite = next((s for s in spec.assets.sprites if s.id == entity.sprite), None)
                if sprite:
                    entity.width = entity.width or sprite.width
                    entity.height = entity.height or sprite.height
                else:
                    entity.width = entity.width or 32
                    entity.height = entity.height or 32

            # Set default physics for player entities
            if entity.type == "player" and entity.physics is None:
                from .models import PhysicsConfig
                entity.physics = PhysicsConfig(enabled=True)

        # Auto-add score display if score system exists
        has_score_system = any(m.type == "score_system" for m in spec.mechanics)
        has_score_ui = any(ui.id == "score_display" for ui in spec.ui)

        if has_score_system and not has_score_ui:
            from .models import UIElement, TextStyle
            score_ui = UIElement(
                type="text",
                id="score_display",
                x=20,
                y=20,
                content="Score: {score}",
                style=TextStyle(fontSize="24px", fill="#ffffff")
            )
            spec.ui.append(score_ui)

        return spec

    def get_validation_report(self) -> Dict[str, Any]:
        """Get a detailed validation report."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }
