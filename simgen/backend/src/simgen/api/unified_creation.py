"""
Unified Creation API - Single endpoint for all creation modes
Routes to appropriate compiler based on mode
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from ..core.modes import (
    mode_registry,
    get_all_mode_info,
    get_mode_info,
    CreationRequest,
    CreationResponse,
    CreationMode
)
from ..core.container import container
from ..core.interfaces import ILLMClient, IComputerVisionPipeline, ICacheService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["unified-creation"])


@router.get("/modes")
async def list_modes():
    """
    List all available creation modes.

    Returns mode information including:
    - Name, description, icon
    - Available features
    - Beta status
    - Supported engines
    """
    modes = get_all_mode_info()

    return {
        "modes": [m.dict() for m in modes],
        "default_mode": "physics",  # Default to physics for existing users
        "total": len(modes)
    }


@router.get("/modes/{mode_id}")
async def get_mode_details(mode_id: str):
    """Get detailed information about a specific mode"""
    mode_info = get_mode_info(mode_id)

    if not mode_info:
        raise HTTPException(status_code=404, detail=f"Mode '{mode_id}' not found")

    if not mode_info.available:
        raise HTTPException(status_code=403, detail=f"Mode '{mode_id}' is not yet available")

    return mode_info.dict()


def get_optional_service(interface):
    """Get service from container, return None if not available"""
    def dependency():
        try:
            return container.get(interface)
        except ValueError:
            return None
    return dependency

@router.post("/create", response_model=CreationResponse)
async def create(
    request: CreationRequest,
    background_tasks: BackgroundTasks,
    llm_client: ILLMClient = Depends(lambda: container.get(ILLMClient)),
    cv_pipeline: Optional[IComputerVisionPipeline] = Depends(get_optional_service(IComputerVisionPipeline)),
    cache: Optional[ICacheService] = Depends(get_optional_service(ICacheService))
):
    """
    Unified creation endpoint for all modes.

    Routes to the appropriate compiler based on the mode:
    - physics → MuJoCo compiler
    - games → Phaser compiler
    - vr → Babylon/Three.js compiler (future)

    Workflow:
    1. Validate mode availability
    2. Analyze sketch (if provided)
    3. Generate specification using LLM
    4. Compile to mode-specific output
    5. Return playable/runnable result
    """
    try:
        # Validate mode
        if not mode_registry.is_mode_available(request.mode.value):
            raise HTTPException(
                status_code=400,
                detail=f"Mode '{request.mode}' is not available. Try: physics, games"
            )

        # Get mode compiler
        compiler = mode_registry.get_compiler(request.mode.value)
        if not compiler:
            raise HTTPException(
                status_code=500,
                detail=f"Compiler not configured for mode '{request.mode}'"
            )

        logger.info(f"Creating {request.mode} with prompt: {request.prompt[:100]}...")

        # Step 1: Analyze sketch if provided
        sketch_analysis = None
        if request.sketch_data:
            try:
                import base64
                sketch_bytes = base64.b64decode(request.sketch_data)

                # Use CV pipeline
                cv_result = await cv_pipeline.analyze_sketch(sketch_bytes)

                sketch_analysis = {
                    "objects": [
                        {
                            "type": obj.object_type.value,
                            "position": list(obj.center) + [0],
                            "size": obj.size,
                            "confidence": obj.confidence
                        }
                        for obj in cv_result.objects
                    ],
                    "annotations": cv_result.text_annotations,
                    "confidence": cv_result.confidence
                }
            except Exception as e:
                logger.warning(f"Sketch analysis failed: {e}, continuing without")

        # Step 2: Generate specification using LLM
        # This creates a unified intermediate representation
        spec = await generate_creation_spec(
            mode=request.mode.value,
            prompt=request.prompt,
            sketch_analysis=sketch_analysis,
            llm_client=llm_client
        )

        # Step 3: Validate specification
        is_valid, errors = await compiler.validate(spec)
        if not is_valid:
            return CreationResponse(
                success=False,
                mode=request.mode.value,
                creation_id="",
                output={},
                errors=errors
            )

        # Step 4: Compile to mode-specific output
        output = await compiler.compile(spec, request.options or {})

        # Step 5: Save to database (background task)
        creation_id = f"{request.mode.value}_{generate_id()}"
        background_tasks.add_task(
            save_creation,
            creation_id=creation_id,
            mode=request.mode.value,
            spec=spec,
            output=output
        )

        return CreationResponse(
            success=True,
            mode=request.mode.value,
            creation_id=creation_id,
            output=output,
            suggestions=generate_suggestions(request.mode.value, spec)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def generate_creation_spec(
    mode: str,
    prompt: str,
    sketch_analysis: Optional[dict],
    llm_client: ILLMClient
) -> dict:
    """
    Generate unified creation specification using LLM.

    This creates an intermediate representation that can be
    compiled to any mode-specific format.
    """
    # Mode-specific system prompts
    system_prompts = {
        "physics": """You are a physics simulation expert. Convert user ideas into
physics specifications with bodies, joints, forces, and constraints.""",

        "games": """You are a game design expert. Convert user ideas into
game specifications with entities, behaviors, mechanics, and rules.""",

        "vr": """You are a VR experience designer. Convert user ideas into
immersive 3D environments with interactions and spatial elements."""
    }

    system_prompt = system_prompts.get(mode, "You are a creative AI assistant.")

    # Build context
    context_parts = [f"User prompt: {prompt}"]

    if sketch_analysis:
        context_parts.append(f"Sketch analysis: {sketch_analysis}")

    context = "\n\n".join(context_parts)

    # Generate specification
    # Combine system and user prompts into single prompt
    full_prompt = f"{system_prompt}\n\n{context}\n\nGenerate a detailed specification as JSON."
    response = await llm_client.complete(prompt=full_prompt)

    # Parse response (assuming JSON)
    import json
    try:
        spec = json.loads(response)
    except:
        # Fallback: extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            spec = json.loads(json_match.group())
        else:
            spec = {"raw_response": response}

    return spec


def generate_suggestions(mode: str, spec: dict) -> list[str]:
    """Generate helpful suggestions based on the creation"""
    suggestions = []

    if mode == "physics":
        suggestions.extend([
            "Add sensors to track object positions",
            "Adjust gravity for different effects",
            "Try adding actuators for control"
        ])
    elif mode == "games":
        suggestions.extend([
            "Add power-ups for more variety",
            "Try the remix feature to modify",
            "Export to multiple game engines"
        ])

    return suggestions[:3]  # Return top 3


async def save_creation(creation_id: str, mode: str, spec: dict, output: dict):
    """Save creation to database (background task)"""
    # TODO: Implement database save
    logger.info(f"Saved creation {creation_id} ({mode})")


def generate_id() -> str:
    """Generate unique creation ID"""
    import uuid
    return str(uuid.uuid4())[:8]


@router.get("/creations/{creation_id}")
async def get_creation(creation_id: str):
    """Retrieve a saved creation"""
    # TODO: Implement database retrieval
    return {
        "creation_id": creation_id,
        "status": "placeholder",
        "message": "Database integration coming soon"
    }


@router.get("/creations")
async def list_creations(
    mode: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List user's creations, optionally filtered by mode"""
    # TODO: Implement database query
    return {
        "creations": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
        "mode_filter": mode
    }
