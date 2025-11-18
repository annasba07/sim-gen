"""
Social sharing service for physics simulations and games.
Generates shareable links with social media metadata.
"""

import logging
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SharingService:
    """Service for creating shareable simulation/game links."""

    def __init__(self, base_url: str = "https://simgen.ai"):
        """
        Initialize sharing service.

        Args:
            base_url: Base URL for shareable links
        """
        self.base_url = base_url.rstrip('/')
        self._cache: Dict[str, Dict[str, Any]] = {}

    def create_share_link(
        self,
        content_type: str,  # 'simulation' or 'game'
        content_data: Dict[str, Any],
        title: Optional[str] = None,
        description: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        expires_in_days: int = 30
    ) -> Dict[str, Any]:
        """
        Create a shareable link with metadata.

        Args:
            content_type: Type of content ('simulation' or 'game')
            content_data: The actual content data (MJCF, game spec, etc.)
            title: Title for social media
            description: Description for social media
            thumbnail_url: URL to preview image
            expires_in_days: Days until link expires

        Returns:
            Dict with share_id, url, and social media metadata
        """
        # Generate unique share ID
        content_str = json.dumps(content_data, sort_keys=True)
        share_id = hashlib.sha256(content_str.encode()).hexdigest()[:12]

        # Create share URL
        share_url = f"{self.base_url}/share/{content_type}/{share_id}"

        # Default metadata
        if title is None:
            title = f"Check out my {content_type}!"
        if description is None:
            description = f"Created with SimGen AI - {content_type} generation powered by AI"

        # Calculate expiry
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Store in cache (in production, save to database)
        self._cache[share_id] = {
            "content_type": content_type,
            "content_data": content_data,
            "title": title,
            "description": description,
            "thumbnail_url": thumbnail_url,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "view_count": 0
        }

        # Generate social media metadata
        social_meta = self._generate_social_meta(
            share_url=share_url,
            title=title,
            description=description,
            thumbnail_url=thumbnail_url,
            content_type=content_type
        )

        return {
            "share_id": share_id,
            "share_url": share_url,
            "short_url": f"{self.base_url}/s/{share_id}",  # Shortened version
            "expires_at": expires_at.isoformat(),
            "social_meta": social_meta,
            "embed_code": self._generate_embed_code(share_id, content_type)
        }

    def _generate_social_meta(
        self,
        share_url: str,
        title: str,
        description: str,
        thumbnail_url: Optional[str],
        content_type: str
    ) -> Dict[str, str]:
        """Generate Open Graph and Twitter Card metadata."""
        meta = {
            # Open Graph (Facebook, LinkedIn, etc.)
            "og:url": share_url,
            "og:type": "website",
            "og:title": title,
            "og:description": description,
            "og:site_name": "SimGen AI",

            # Twitter Card
            "twitter:card": "summary_large_image",
            "twitter:title": title,
            "twitter:description": description,
            "twitter:site": "@simgenai",  # Update with actual Twitter handle
        }

        if thumbnail_url:
            meta["og:image"] = thumbnail_url
            meta["og:image:width"] = "1200"
            meta["og:image:height"] = "630"
            meta["twitter:image"] = thumbnail_url

        return meta

    def _generate_embed_code(self, share_id: str, content_type: str) -> str:
        """Generate HTML embed code for sharing."""
        embed_url = f"{self.base_url}/embed/{content_type}/{share_id}"

        if content_type == "simulation":
            width, height = 800, 600
        else:  # game
            width, height = 800, 600

        return f'''<iframe
    src="{embed_url}"
    width="{width}"
    height="{height}"
    frameborder="0"
    allowfullscreen
    title="SimGen AI - {content_type.title()}"
></iframe>'''

    def get_share_data(self, share_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve shared content by ID.

        Args:
            share_id: The share identifier

        Returns:
            Share data if found and not expired, None otherwise
        """
        if share_id not in self._cache:
            return None

        share_data = self._cache[share_id]

        # Check expiry
        expires_at = datetime.fromisoformat(share_data["expires_at"])
        if datetime.utcnow() > expires_at:
            logger.info(f"Share {share_id} has expired")
            del self._cache[share_id]
            return None

        # Increment view count
        share_data["view_count"] += 1

        return share_data

    def generate_social_share_links(self, share_url: str, title: str, description: str) -> Dict[str, str]:
        """
        Generate direct social media sharing links.

        Args:
            share_url: The URL to share
            title: Title of the content
            description: Description of the content

        Returns:
            Dict of social platform -> share URL
        """
        import urllib.parse

        encoded_url = urllib.parse.quote(share_url)
        encoded_title = urllib.parse.quote(title)
        encoded_desc = urllib.parse.quote(description)

        return {
            "twitter": f"https://twitter.com/intent/tweet?url={encoded_url}&text={encoded_title}",
            "facebook": f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}",
            "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}",
            "reddit": f"https://reddit.com/submit?url={encoded_url}&title={encoded_title}",
            "email": f"mailto:?subject={encoded_title}&body={encoded_desc}%0A%0A{encoded_url}",
            "copy": share_url  # For copy to clipboard
        }

    def generate_thumbnail(
        self,
        content_type: str,
        content_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a thumbnail image for social sharing.

        Args:
            content_type: Type of content
            content_data: The content data
            output_path: Optional output path

        Returns:
            Path to generated thumbnail
        """
        # TODO: Implement actual thumbnail generation
        # For simulations: render first frame
        # For games: capture game screenshot
        # For now, return placeholder
        logger.warning("Thumbnail generation not yet implemented, using placeholder")
        return f"{self.base_url}/static/placeholder-{content_type}.png"


# Singleton instance
_sharing_service = None

def get_sharing_service() -> SharingService:
    """Get singleton sharing service instance."""
    global _sharing_service
    if _sharing_service is None:
        _sharing_service = SharingService()
    return _sharing_service
