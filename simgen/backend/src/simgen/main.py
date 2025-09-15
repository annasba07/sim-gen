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
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        logger.info("Redis connection: OK")
        
        # Test LLM API
        from .services.llm_client import LLMClient
        llm_client = LLMClient()
        await llm_client.test_connection()
        logger.info("LLM API connection: OK")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


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
from .api import physics
app.include_router(physics.router, prefix="/api/v2/physics", tags=["physics"])


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