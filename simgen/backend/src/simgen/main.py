import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .core.config import settings
from .models.schemas import HealthCheck
from .api import simulation, templates


# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    
    # Initialize services
    try:
        # Test database connection - temporarily disabled for testing
        # from .db.base import async_engine
        # async with async_engine.begin() as conn:
        #     await conn.execute("SELECT 1")
        logger.info("Database connection: SKIPPED (testing mode)")
        
        # Test Redis connection
        try:
            import redis
            r = redis.from_url(settings.redis_url)
            r.ping()
            logger.info("Redis connection: OK")
        except Exception as e:
            logger.warning(f"Redis connection failed (optional): {e}")
            logger.info("Redis connection: SKIPPED (testing mode)")
        
        # Test LLM API
        from .services.llm_client import LLMClient
        llm_client = LLMClient()
        await llm_client.test_connection()
        logger.info("LLM API connection: OK")

        # Initialize DI container for VirtualForge unified API
        from .core.container import container
        from .core.interfaces import ILLMClient, IComputerVisionPipeline, IPhysicsCompiler
        from .services.mjcf_compiler import MJCFCompiler
        from .core.modes import mode_registry

        # Register services in DI container
        container.register(ILLMClient, llm_client)

        # CV pipeline - make optional for testing (requires heavy dependencies)
        try:
            from .services.cv_simplified import SimplifiedCVPipeline
            cv_pipeline = SimplifiedCVPipeline()
            await cv_pipeline.initialize()
            container.register(IComputerVisionPipeline, cv_pipeline)
            logger.info("CV pipeline initialized")
        except ImportError as e:
            logger.warning(f"CV pipeline not available (missing dependencies): {e}")
            logger.info("CV pipeline: SKIPPED (testing mode)")

        physics_compiler = MJCFCompiler()
        container.register(IPhysicsCompiler, physics_compiler)

        # Register physics compiler in mode registry for VirtualForge
        mode_registry.register_compiler('physics', physics_compiler)

        logger.info("DI container initialized with all services")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield

    # Shutdown
    logger.info("Shutting down application")

    # Clean up DI container
    try:
        from .core.container import container
        container.clear()
        logger.info("DI container cleaned up")
    except Exception as e:
        logger.warning(f"DI container cleanup failed: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered physics simulation generation using MuJoCo and advanced LLMs",
    lifespan=lifespan,
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    services = {}
    
    try:
        # Check database
        from .db.base import async_engine
        async with async_engine.begin() as conn:
            await conn.execute("SELECT 1")
        services["database"] = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        services["database"] = "error"
    
    try:
        # Check Redis
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        services["redis"] = "ok"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        services["redis"] = "error"
    
    try:
        # Check LLM API
        from .services.llm_client import LLMClient
        llm_client = LLMClient()
        await llm_client.test_connection()
        services["llm_api"] = "ok"
    except Exception as e:
        logger.error(f"LLM API health check failed: {e}")
        services["llm_api"] = "error"
    
    return HealthCheck(
        version=settings.version,
        timestamp=datetime.utcnow(),
        services=services
    )


# Include API routers
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["simulation"])
app.include_router(templates.router, prefix="/api/v1/templates", tags=["templates"])

# Include new physics pipeline router
from .api import physics, unified_creation
app.include_router(physics.router, prefix="/api/v2/physics", tags=["physics"])

# Include VirtualForge unified creation API
app.include_router(unified_creation.router)  # Already has /api/v2 prefix


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )