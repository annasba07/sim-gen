"""
Clean main application entry point with dependency injection.
Properly initializes services and registers dependencies.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Configuration
from .core.config_clean import settings

# Dependency injection
from .core.container import container
from .core.interfaces import (
    IPhysicsCompiler,
    ICacheService,
    IWebSocketManager,
    ILLMClient,
    IComputerVisionPipeline
)

# Service implementations
from .services.mjcf_compiler import MJCFCompiler
from .services.llm_client import LLMClient
from .services.cache_service import CacheService
from .services.websocket_session_manager import RedisWebSocketManager
from .services.cv_simplified import SimplifiedCVPipeline

# API routes
from .api import health, physics_clean, sketch_clean, realtime_feedback, sketch_templates, error_feedback, unified_creation, games

# Middleware
from .core.validation import validate_request_middleware, rate_limiter
from .core.resource_manager import resource_manager

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown properly.
    """
    # Startup
    logger.info("Starting SimGen AI application...")

    try:
        # Initialize services
        await initialize_services()

        # Register dependencies
        await register_dependencies()

        # Start background tasks
        await start_background_tasks()

        # Initialize CV pipeline models asynchronously
        cv_pipeline = container.get(IComputerVisionPipeline)
        await cv_pipeline.initialize()

        logger.info("Application started successfully")
        yield

    finally:
        # Shutdown
        logger.info("Shutting down SimGen AI application...")

        # Stop background tasks
        await stop_background_tasks()

        # Cleanup resources
        await cleanup_resources()

        logger.info("Application shutdown complete")


async def initialize_services():
    """Initialize all services with proper configuration."""
    logger.info("Initializing services...")

    # Initialize cache service
    cache_service = CacheService(
        memory_cache_size=settings.cache_memory_size,
        default_ttl=settings.cache_ttl
    )
    await cache_service.initialize(settings.redis_url)

    # Initialize WebSocket manager
    import redis.asyncio as redis
    redis_client = redis.from_url(settings.redis_url)
    ws_manager = RedisWebSocketManager(
        redis_client=redis_client,
        server_id=settings.server_id,
        heartbeat_interval=30,
        session_ttl=3600
    )
    await ws_manager.start()

    # Store initialized services
    container._cache_service = cache_service
    container._ws_manager = ws_manager

    logger.info("Services initialized")


async def register_dependencies():
    """Register all dependencies in the container."""
    logger.info("Registering dependencies...")

    # Register service implementations
    container.register(IPhysicsCompiler, MJCFCompiler())

    # LLM client
    llm_client = LLMClient()
    container.register(ILLMClient, llm_client)

    # Simplified CV pipeline (replaces 1,118-line custom implementation)
    cv_pipeline = SimplifiedCVPipeline()
    container.register(IComputerVisionPipeline, cv_pipeline)

    # Infrastructure services
    container.register(ICacheService, container._cache_service)
    container.register(IWebSocketManager, container._ws_manager)

    # Register mode compilers for VirtualForge
    from .core.modes import mode_registry
    from .modes.games import PhaserCompiler

    # Physics mode compiler
    physics_compiler = container.get(IPhysicsCompiler)
    mode_registry.register_compiler('physics', physics_compiler)

    # Games mode compiler
    games_compiler = PhaserCompiler()
    mode_registry.register_compiler('games', games_compiler)

    logger.info("Dependencies registered (including mode compilers: physics, games)")


async def start_background_tasks():
    """Start background tasks."""
    logger.info("Starting background tasks...")

    # Start resource manager cleanup
    asyncio.create_task(resource_manager.periodic_cleanup(interval=300))

    # Start rate limiter cleanup
    await rate_limiter.start()

    logger.info("Background tasks started")


async def stop_background_tasks():
    """Stop background tasks."""
    logger.info("Stopping background tasks...")

    # Stop rate limiter
    await rate_limiter.stop()

    # Stop WebSocket manager
    if hasattr(container, '_ws_manager'):
        await container._ws_manager.stop()

    logger.info("Background tasks stopped")


async def cleanup_resources():
    """Clean up all resources."""
    logger.info("Cleaning up resources...")

    # Clean up resource manager
    await resource_manager.cleanup_all()

    # Close cache connections
    if hasattr(container, '_cache_service'):
        await container._cache_service.close()

    # Clear dependency container
    container.clear()

    logger.info("Resources cleaned up")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="SimGen AI",
        description="Clean Architecture Implementation - Sketch to Physics Simulation",
        version="2.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.middleware("http")(validate_request_middleware)

    # Register routers
    app.include_router(health.router)
    app.include_router(physics_clean.router)
    app.include_router(sketch_clean.router)
    app.include_router(realtime_feedback.router)
    app.include_router(sketch_templates.router)
    app.include_router(error_feedback.router)
    app.include_router(unified_creation.router)  # VirtualForge unified API
    app.include_router(games.router)  # Games API (Phaser compiler)

    # Exception handlers
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": "Validation Error", "detail": str(exc)}
        )

    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
        )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "SimGen AI API",
            "version": "2.0.0",
            "status": "running",
            "architecture": "clean",
            "health_check": "/health",
            "documentation": "/docs"
        }

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "simgen.main_clean:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )