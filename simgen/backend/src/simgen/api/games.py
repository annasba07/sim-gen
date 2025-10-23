"""
Games API endpoints for VirtualForge.
Handles game generation, compilation, and template management.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..modes.games import PhaserCompiler
from ..modes.games.templates import get_template, get_all_templates, get_templates_by_type
from ..core.container import container
from ..core.interfaces import ILLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/games", tags=["games"])

# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateGameRequest(BaseModel):
    """Request to generate a game from prompt."""
    prompt: str
    sketch_data: Optional[str] = None
    gameType: Optional[str] = "platformer"
    complexity: Optional[str] = "simple"


class CompileGameRequest(BaseModel):
    """Request to compile a game spec."""
    spec: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None


class GameResponse(BaseModel):
    """Response for game operations."""
    success: bool
    html: Optional[str] = None
    code: Optional[str] = None
    game_spec: Optional[Dict] = None
    assets: Optional[List[Dict]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/generate", response_model=GameResponse)
async def generate_game(request: GenerateGameRequest):
    """
    Generate a game from a text prompt and optional sketch.

    This uses the LLM to create a game specification from the user's description.
    The spec is then compiled to a playable Phaser game.
    """
    try:
        # Get LLM client
        llm_client = container.get(ILLMClient)

        # Build robust LLM prompt with clear examples
        system_prompt = f"""You are a game design AI. Generate a VALID JSON game specification.

CRITICAL: Respond with ONLY the JSON object. No explanations, no markdown, just JSON.

Required JSON structure:
{{
  "version": "1.0",
  "gameType": "{request.gameType}",
  "title": "Game Name",
  "description": "Brief description",
  "world": {{
    "width": 800,
    "height": 600,
    "gravity": 800,
    "backgroundColor": "#87CEEB"
  }},
  "assets": {{
    "sprites": [
      {{"id": "player", "type": "sprite", "source": "placeholder", "width": 32, "height": 48}}
    ]
  }},
  "entities": [
    {{"id": "player", "type": "player", "sprite": "player", "x": 100, "y": 400, "physics": {{"enabled": true}}}}
  ],
  "behaviors": [
    {{"id": "move", "type": "movement_keyboard", "entityId": "player", "config": {{"keys": "arrows", "speed": 200}}}}
  ],
  "mechanics": [
    {{"type": "score_system", "config": {{"initialScore": 0}}}}
  ],
  "rules": [],
  "ui": []
}}

Behavior types: movement_keyboard, jump, collect, shoot, follow, patrol
Mechanic types: score_system, health_system, timer
Use "placeholder" for sprite source. Keep complexity {request.complexity}."""

        # Combine system and user prompts
        combined_prompt = f"{system_prompt}\n\nUser request: Create {request.gameType}: {request.prompt}"

        # Try to call LLM (with fallback to template on any error)
        import json
        import re
        game_spec = None

        try:
            # Call LLM
            response = await llm_client.complete(
                prompt=combined_prompt,
                max_tokens=2500
            )

            # Parse JSON response with multiple fallback strategies
            try:
                # Strategy 1: Direct parse
                game_spec = json.loads(response.strip())
            except:
                try:
                    # Strategy 2: Extract from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                    if json_match:
                        game_spec = json.loads(json_match.group(1))
                except:
                    try:
                        # Strategy 3: Find first { to last }
                        start = response.find('{')
                        end = response.rfind('}') + 1
                        if start != -1 and end > start:
                            game_spec = json.loads(response[start:end])
                    except:
                        pass
        except Exception as llm_error:
            logger.warning(f"LLM call failed: {llm_error}, will use template fallback")

        if not game_spec:
            # Fallback to template if LLM fails
            logger.warning(f"LLM failed to generate valid JSON, using template")
            from ..modes.games.templates import get_template

            template_map = {
                "platformer": "coin-collector",
                "topdown": "dungeon-explorer",
                "shooter": "space-shooter",
                "puzzle": "coin-collector"  # Use platformer as fallback
            }

            template_id = template_map.get(request.gameType, "coin-collector")
            template = get_template(template_id)

            if template:
                game_spec = template.model_dump()
                game_spec["title"] = f"Custom {request.gameType.title()}"
                game_spec["description"] = request.prompt[:100]
            else:
                return GameResponse(
                    success=False,
                    errors=["Failed to generate game and no template available"]
                )

        # Compile the game
        compiler = PhaserCompiler()
        result = await compiler.compile(game_spec)

        return GameResponse(
            success=result["success"],
            html=result.get("html"),
            code=result.get("code"),
            game_spec=game_spec,
            assets=result.get("assets"),
            errors=result.get("errors"),
            warnings=result.get("warnings")
        )

    except Exception as e:
        logger.error(f"Game generation error: {e}", exc_info=True)
        return GameResponse(
            success=False,
            errors=[str(e)]
        )


@router.post("/compile", response_model=GameResponse)
async def compile_game(request: CompileGameRequest):
    """
    Compile a game specification to Phaser HTML/JavaScript.

    Takes a complete game spec and generates playable HTML.
    """
    try:
        compiler = PhaserCompiler()
        result = await compiler.compile(
            spec=request.spec,
            options=request.options or {}
        )

        return GameResponse(
            success=result["success"],
            html=result.get("html"),
            code=result.get("code"),
            assets=result.get("assets"),
            errors=result.get("errors"),
            warnings=result.get("warnings")
        )

    except Exception as e:
        logger.error(f"Compilation error: {e}", exc_info=True)
        return GameResponse(
            success=False,
            errors=[str(e)]
        )


@router.get("/templates")
async def list_templates():
    """Get all available game templates."""
    return get_all_templates()


@router.get("/templates/{template_id}")
async def get_template_by_id(template_id: str):
    """Get a specific template by ID."""
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return {
        "id": template_id,
        "spec": template.model_dump()
    }


@router.get("/templates/type/{game_type}")
async def get_templates_by_game_type(game_type: str):
    """Get templates filtered by game type."""
    templates = get_templates_by_type(game_type)
    return templates


@router.post("/validate")
async def validate_spec(spec: Dict[str, Any]):
    """Validate a game specification without compiling."""
    try:
        compiler = PhaserCompiler()
        is_valid, errors = await compiler.validate(spec)

        return {
            "valid": is_valid,
            "errors": errors
        }

    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }
