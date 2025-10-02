"""
Game Generation API - From Prompts to Playable Games
Evolution of SimGen AI to create full interactive experiences
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel

from ..services.game_mechanics_generator import (
    GameMechanicsGenerator,
    GameType,
    GameSpec,
    create_game_generator
)
from ..core.container import container
from ..core.interfaces import IComputerVisionPipeline, ILLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/games", tags=["game-generation"])


class GameGenerationRequest(BaseModel):
    """Request for game generation"""
    prompt: str
    game_type: Optional[str] = None
    sketch_data: Optional[str] = None  # Base64 encoded sketch
    complexity: str = "simple"  # simple, medium, complex
    target_engine: str = "babylon"  # babylon, unity, roblox, mujoco
    include_tutorial: bool = False
    multiplayer: bool = False


class GameGenerationResponse(BaseModel):
    """Response with generated game"""
    success: bool
    game_spec: Optional[Dict[str, Any]] = None
    export_data: Optional[Dict[str, Any]] = None
    playable_url: Optional[str] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None


@router.post("/generate")
async def generate_game(
    request: GameGenerationRequest,
    llm_client: ILLMClient = Depends(lambda: container.get(ILLMClient)),
    cv_pipeline: IComputerVisionPipeline = Depends(lambda: container.get(IComputerVisionPipeline))
):
    """
    Generate a complete game from natural language prompt and optional sketch.

    Examples:
    - "Create a platformer where you collect gems and avoid enemies"
    - "Make a puzzle game with physics and magnets"
    - "Build an educational game about ecosystems"
    """
    try:
        # Initialize game generator
        game_generator = create_game_generator(llm_client)

        # Analyze sketch if provided
        sketch_analysis = None
        if request.sketch_data:
            try:
                # Decode base64 sketch
                import base64
                sketch_bytes = base64.b64decode(request.sketch_data)

                # Analyze with CV pipeline
                cv_result = await cv_pipeline.analyze_sketch(sketch_bytes)

                # Convert CV result to sketch analysis
                sketch_analysis = {
                    "objects": [
                        {
                            "type": obj.object_type.value,
                            "position": list(obj.center) + [0],  # Add Z coordinate
                            "size": obj.size,
                            "confidence": obj.confidence
                        }
                        for obj in cv_result.objects
                    ],
                    "annotations": cv_result.text_annotations,
                    "confidence": cv_result.confidence
                }
            except Exception as e:
                logger.warning(f"Sketch analysis failed: {e}, continuing without sketch")

        # Determine game type
        game_type = None
        if request.game_type:
            try:
                game_type = GameType(request.game_type.lower())
            except ValueError:
                logger.warning(f"Invalid game type: {request.game_type}")

        # Generate game specification
        game_spec = await game_generator.generate_from_prompt(
            prompt=request.prompt,
            sketch_analysis=sketch_analysis,
            game_type=game_type
        )

        # Export to target engine
        export_data = await game_generator.export_to_engine(
            game_spec=game_spec,
            engine=request.target_engine
        )

        # Generate playable URL (would deploy to CDN in production)
        playable_url = await _deploy_game(game_spec, export_data, request.target_engine)

        # Generate suggestions for improvements
        suggestions = _generate_suggestions(game_spec, request.prompt)

        return GameGenerationResponse(
            success=True,
            game_spec=game_spec.__dict__,
            export_data=export_data,
            playable_url=playable_url,
            suggestions=suggestions
        )

    except Exception as e:
        logger.error(f"Game generation failed: {e}")
        return GameGenerationResponse(
            success=False,
            error=str(e),
            suggestions=[
                "Try being more specific about game mechanics",
                "Include details about the player character",
                "Describe the game's objectives clearly",
                "Consider starting with a simpler concept"
            ]
        )


@router.get("/templates")
async def get_game_templates():
    """Get available game templates"""
    templates = {
        "platformer": {
            "name": "Classic Platformer",
            "description": "Jump and run through levels collecting items",
            "prompt_example": "Create a Mario-style platformer with coins and enemies",
            "mechanics": ["jumping", "running", "collecting", "enemies"],
            "difficulty": "beginner"
        },
        "puzzle": {
            "name": "Physics Puzzle",
            "description": "Solve puzzles using physics interactions",
            "prompt_example": "Make an Angry Birds style game with destructible structures",
            "mechanics": ["projectiles", "physics", "destruction", "scoring"],
            "difficulty": "intermediate"
        },
        "survival": {
            "name": "Survival Game",
            "description": "Survive waves of challenges",
            "prompt_example": "Create a zombie survival game with crafting",
            "mechanics": ["health", "combat", "crafting", "waves"],
            "difficulty": "advanced"
        },
        "educational": {
            "name": "Educational Game",
            "description": "Learn through interactive gameplay",
            "prompt_example": "Build a game that teaches kids about the solar system",
            "mechanics": ["quizzes", "exploration", "rewards", "progress"],
            "difficulty": "beginner"
        },
        "racing": {
            "name": "Racing Game",
            "description": "Race against time or opponents",
            "prompt_example": "Make a racing game with power-ups and obstacles",
            "mechanics": ["driving", "speed", "obstacles", "power-ups"],
            "difficulty": "intermediate"
        }
    }
    return templates


@router.get("/engines")
async def get_supported_engines():
    """Get information about supported game engines"""
    engines = {
        "babylon": {
            "name": "Babylon.js",
            "description": "Web-based 3D games, instant play in browser",
            "features": ["no_download", "cross_platform", "webgl", "mobile_friendly"],
            "best_for": ["web_games", "instant_play", "simple_3d"],
            "export_format": "javascript"
        },
        "unity": {
            "name": "Unity WebGL",
            "description": "Professional game engine with WebGL export",
            "features": ["advanced_graphics", "complex_mechanics", "asset_store"],
            "best_for": ["complex_games", "commercial_projects"],
            "export_format": "unity_package"
        },
        "roblox": {
            "name": "Roblox",
            "description": "Social gaming platform with built-in monetization",
            "features": ["multiplayer", "monetization", "huge_audience", "social"],
            "best_for": ["multiplayer_games", "kids_games", "monetization"],
            "export_format": "lua_scripts"
        },
        "mujoco": {
            "name": "MuJoCo",
            "description": "Advanced physics simulation",
            "features": ["realistic_physics", "robotics", "research_grade"],
            "best_for": ["physics_simulations", "educational", "research"],
            "export_format": "mjcf_xml"
        }
    }
    return engines


@router.post("/remix")
async def remix_game(
    game_id: str,
    modifications: str,
    llm_client: ILLMClient = Depends(lambda: container.get(ILLMClient))
):
    """
    Remix an existing game with new features or modifications.

    Example modifications:
    - "Add boss battles at the end of each level"
    - "Make it multiplayer"
    - "Change the theme to underwater"
    - "Add a crafting system"
    """
    # This would load the existing game and apply modifications
    return {
        "success": True,
        "message": "Game remix feature coming soon",
        "original_game_id": game_id,
        "modifications": modifications
    }


@router.get("/analytics/{game_id}")
async def get_game_analytics(game_id: str):
    """Get analytics for a generated game"""
    # Placeholder for game analytics
    return {
        "game_id": game_id,
        "plays": 0,
        "unique_players": 0,
        "average_play_time": 0,
        "completion_rate": 0,
        "user_ratings": [],
        "feedback": []
    }


async def _deploy_game(
    game_spec: GameSpec,
    export_data: Dict[str, Any],
    engine: str
) -> Optional[str]:
    """Deploy game and return playable URL"""
    # In production, this would:
    # 1. Upload to CDN
    # 2. Generate unique URL
    # 3. Set up game hosting

    # For now, return a mock URL
    if engine == "babylon":
        return f"https://play.simgen.ai/games/preview/{game_spec.name.replace(' ', '-').lower()}"
    elif engine == "mujoco":
        return f"https://simgen.ai/simulations/{game_spec.name.replace(' ', '-').lower()}"
    else:
        return None


def _generate_suggestions(game_spec: GameSpec, original_prompt: str) -> List[str]:
    """Generate suggestions for improving the game"""
    suggestions = []

    # Analyze game complexity
    if len(game_spec.levels) == 1:
        suggestions.append("Add more levels for progression")

    if len(game_spec.global_mechanics) < 3:
        suggestions.append("Consider adding more game mechanics for variety")

    # Check for common features
    has_enemies = any(
        entity.type.value == "enemy"
        for level in game_spec.levels
        for entity in level.entities
    )

    if not has_enemies and game_spec.type != GameType.PUZZLE:
        suggestions.append("Add enemies or obstacles for challenge")

    has_collectibles = any(
        entity.type.value == "collectible"
        for level in game_spec.levels
        for entity in level.entities
    )

    if not has_collectibles:
        suggestions.append("Add collectible items for rewards")

    # Check for multiplayer potential
    if "friend" in original_prompt.lower() or "multiplayer" in original_prompt.lower():
        suggestions.append("Enable multiplayer features for social gameplay")

    return suggestions


@router.get("/showcase")
async def get_showcase_games():
    """Get showcase of games created with the system"""
    return {
        "featured": [
            {
                "id": "gem-collector-adventure",
                "name": "Gem Collector Adventure",
                "description": "A platformer where you collect gems and avoid enemies",
                "thumbnail": "/images/gem-collector-thumb.png",
                "play_url": "https://play.simgen.ai/games/gem-collector-adventure",
                "creator": "AI Generated",
                "plays": 1234,
                "rating": 4.5
            },
            {
                "id": "physics-puzzle-master",
                "name": "Physics Puzzle Master",
                "description": "Solve challenging physics puzzles",
                "thumbnail": "/images/physics-puzzle-thumb.png",
                "play_url": "https://play.simgen.ai/games/physics-puzzle-master",
                "creator": "AI Generated",
                "plays": 892,
                "rating": 4.8
            },
            {
                "id": "ecosystem-builder",
                "name": "Ecosystem Builder",
                "description": "Educational game about building balanced ecosystems",
                "thumbnail": "/images/ecosystem-thumb.png",
                "play_url": "https://play.simgen.ai/games/ecosystem-builder",
                "creator": "AI Generated",
                "plays": 567,
                "rating": 4.7
            }
        ],
        "categories": {
            "platformers": 15,
            "puzzles": 23,
            "educational": 18,
            "survival": 7,
            "racing": 9
        }
    }