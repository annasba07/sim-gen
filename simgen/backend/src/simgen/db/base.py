from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from ..core.config_clean import settings

# Convert postgres:// to postgresql:// for SQLAlchemy 2.0
DATABASE_URL = settings.database_url
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Legacy engines for migrations and backwards compatibility
async_engine = create_async_engine(
    DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Sync engine for migrations
sync_engine = create_engine(DATABASE_URL, echo=settings.debug)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

Base = declarative_base()


async def get_async_session():
    """Legacy session getter - use get_optimized_session from database.connection_pool for better performance."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_session():
    """Legacy session getter for sync operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Import optimized database components
try:
    from ..database.connection_pool import get_optimized_session
    from ..database.service import get_database_service
    
    # Provide optimized session as the default
    get_db = get_optimized_session
    
except ImportError:
    # Fallback to legacy if optimized components not available
    get_db = get_async_session