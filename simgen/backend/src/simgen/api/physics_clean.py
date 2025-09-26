"""
Clean Physics API endpoints with proper separation of concerns.
Thin controllers that delegate to services.
"""

from fastapi import APIRouter, HTTPException, Depends, Response
from typing import Optional, Dict, Any

from ..core.interfaces import IPhysicsCompiler, IWebSocketManager, ICacheService
from ..core.container import container
from ..core.validation import ValidatedPhysicsRequest, RequestValidator
from ..models.physics_spec import PhysicsSpec
from ..models.schemas import CompileResponse, SimulationRequest

router = APIRouter(prefix="/api/v2/physics", tags=["physics"])


def get_physics_compiler() -> IPhysicsCompiler:
    """Dependency injection for physics compiler."""
    return container.get(IPhysicsCompiler)


def get_websocket_manager() -> IWebSocketManager:
    """Dependency injection for WebSocket manager."""
    return container.get(IWebSocketManager)


def get_cache_service() -> ICacheService:
    """Dependency injection for cache service."""
    return container.get(ICacheService)


@router.post("/compile", response_model=CompileResponse)
async def compile_physics_spec(
    request: ValidatedPhysicsRequest,
    compiler: IPhysicsCompiler = Depends(get_physics_compiler),
    cache: ICacheService = Depends(get_cache_service)
) -> CompileResponse:
    """
    Compile PhysicsSpec to MJCF XML.

    This is a thin controller that:
    1. Validates input (via ValidatedPhysicsRequest)
    2. Checks cache
    3. Delegates to compiler service
    4. Caches result
    5. Returns response
    """
    try:
        # Generate cache key
        cache_key = f"mjcf:{request.physics_spec.meta.name}"

        # Check cache
        cached_mjcf = await cache.get(cache_key)
        if cached_mjcf:
            return CompileResponse(
                success=True,
                mjcf_xml=cached_mjcf,
                warnings=["Retrieved from cache"]
            )

        # Delegate to service
        mjcf_xml = await compiler.compile(request.physics_spec)

        # Cache result
        await cache.set(cache_key, mjcf_xml, ttl=3600)

        return CompileResponse(
            success=True,
            mjcf_xml=mjcf_xml
        )

    except ValueError as e:
        # Domain validation error
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected error
        raise HTTPException(status_code=500, detail=f"Compilation failed: {str(e)}")


@router.post("/generate")
async def generate_from_prompt(
    request: ValidatedPhysicsRequest,
    compiler: IPhysicsCompiler = Depends(get_physics_compiler)
) -> Dict[str, Any]:
    """
    Generate PhysicsSpec from natural language prompt.

    Thin controller that delegates to AI service.
    """
    # This would delegate to IAIService
    # For now, return mock response
    return {
        "success": True,
        "physics_spec": {
            "meta": {"name": "generated"},
            "bodies": []
        }
    }


@router.get("/templates")
async def list_templates(
    cache: ICacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """
    List available physics templates.

    Thin controller that retrieves templates.
    """
    # Check cache first
    cached_templates = await cache.get("templates:all")
    if cached_templates:
        return {"templates": cached_templates}

    # In production, this would query the repository
    templates = [
        {"name": "pendulum", "description": "Simple pendulum"},
        {"name": "double_pendulum", "description": "Chaotic double pendulum"},
        {"name": "cart_pole", "description": "Classic control problem"},
    ]

    # Cache for 5 minutes
    await cache.set("templates:all", templates, ttl=300)

    return {"templates": templates}


@router.get("/health")
async def physics_health_check() -> Dict[str, str]:
    """
    Health check for physics service.

    Simple endpoint that verifies service is responsive.
    """
    return {
        "status": "healthy",
        "service": "physics"
    }


# WebSocket endpoint would be in a separate file for clarity
# but included here for completeness

from fastapi import WebSocket, WebSocketDisconnect
import uuid


@router.websocket("/ws/stream")
async def physics_stream(
    websocket: WebSocket,
    ws_manager: IWebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for real-time physics streaming.

    Thin controller that:
    1. Accepts connection
    2. Delegates to WebSocket manager
    3. Handles disconnection
    """
    await websocket.accept()

    # Generate client ID
    client_id = str(uuid.uuid4())

    try:
        # Register session with manager
        session_id = await ws_manager.connect_session(websocket, client_id)

        # Keep connection alive
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Delegate message handling to service
            # This is where you'd handle simulation commands
            if data.get("type") == "start_simulation":
                await ws_manager.send_to_session(
                    session_id,
                    {"type": "status", "message": "Simulation started"}
                )

    except WebSocketDisconnect:
        # Clean disconnection
        await ws_manager.disconnect_session(session_id)

    except Exception as e:
        # Error disconnection
        await ws_manager.disconnect_session(session_id)
        raise