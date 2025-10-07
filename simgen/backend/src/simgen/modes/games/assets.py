"""
Asset management for games.
MVP: Use colored rectangles as placeholders.
Future: AI-generated sprites, asset library, CDN upload.
"""

import logging
from typing import List, Dict
from .models import GameSpec

logger = logging.getLogger(__name__)


class AssetManager:
    """
    Manages game assets (sprites, sounds, etc.).

    For MVP: Generates placeholder sprites as colored rectangles.
    Future phases: AI sprite generation, asset library, CDN hosting.
    """

    def __init__(self):
        self.asset_cache: Dict[str, str] = {}

    async def process(self, spec: GameSpec) -> List[Dict[str, str]]:
        """
        Process all assets in the game spec.

        For MVP, this mainly validates that assets exist and returns metadata.
        Actual asset loading happens in Phaser's preload() function.

        Args:
            spec: Game specification

        Returns:
            List of asset metadata dicts
        """
        asset_urls = []

        # Process sprites
        for sprite in spec.assets.sprites:
            if sprite.source == "placeholder":
                # Placeholder sprites are generated in Phaser preload()
                asset_urls.append({
                    "id": sprite.id,
                    "type": "sprite",
                    "source": "placeholder",
                    "width": sprite.width,
                    "height": sprite.height,
                    "url": None  # Generated client-side
                })

            elif sprite.source == "url":
                # External URL
                asset_urls.append({
                    "id": sprite.id,
                    "type": "sprite",
                    "source": "url",
                    "url": sprite.url,
                    "width": sprite.width,
                    "height": sprite.height
                })

            elif sprite.source == "generated":
                # Future: AI-generated sprites
                logger.warning(f"AI sprite generation not yet implemented for {sprite.id}")
                asset_urls.append({
                    "id": sprite.id,
                    "type": "sprite",
                    "source": "placeholder",  # Fallback to placeholder
                    "width": sprite.width,
                    "height": sprite.height,
                    "url": None
                })

        # Process sounds
        for sound in spec.assets.sounds:
            if sound.source == "placeholder":
                # Skip for MVP (no sound)
                logger.info(f"Sound '{sound.id}' using placeholder (silent)")
            elif sound.source == "url":
                asset_urls.append({
                    "id": sound.id,
                    "type": "sound",
                    "source": "url",
                    "url": sound.url
                })

        logger.info(f"Processed {len(asset_urls)} assets")
        return asset_urls

    async def generate_placeholder_sprite(self, width: int, height: int, color: str = "FF6B6B") -> str:
        """
        Generate a placeholder sprite as data URL (future enhancement).

        For MVP, placeholders are created in Phaser code.
        This method is reserved for future server-side generation.

        Args:
            width: Sprite width
            height: Sprite height
            color: Hex color (without #)

        Returns:
            Data URL or empty string
        """
        # TODO: Generate actual PNG data URL
        # For now, return empty - Phaser handles it
        return ""

    async def upload_to_cdn(self, asset_data: bytes, filename: str) -> str:
        """
        Upload asset to CDN (future enhancement).

        Args:
            asset_data: Binary asset data
            filename: Target filename

        Returns:
            CDN URL
        """
        # TODO: Implement CDN upload (Cloudflare R2, S3, etc.)
        logger.warning("CDN upload not yet implemented")
        return ""

    def validate_asset_references(self, spec: GameSpec) -> tuple[bool, List[str]]:
        """
        Validate that all asset references in the spec are valid.

        Args:
            spec: Game specification

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        sprite_ids = {s.id for s in spec.assets.sprites}

        # Check that all entities reference valid sprites
        for entity in spec.entities:
            if entity.sprite not in sprite_ids:
                errors.append(
                    f"Entity '{entity.id}' references unknown sprite '{entity.sprite}'"
                )

        # Check that shoot behaviors reference valid projectile sprites
        for behavior in spec.behaviors:
            if behavior.type == "shoot":
                projectile_sprite = behavior.config.get("projectileSprite")
                if projectile_sprite and projectile_sprite not in sprite_ids:
                    errors.append(
                        f"Shoot behavior '{behavior.id}' references unknown sprite '{projectile_sprite}'"
                    )

        return len(errors) == 0, errors
