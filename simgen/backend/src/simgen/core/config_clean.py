"""
Centralized configuration management.
All settings in one place, loaded from environment variables.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Provides defaults for development, overridden in production.
    """

    # Application
    app_name: str = "SimGen AI"
    app_version: str = "2.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    server_id: str = Field(default="server-1", env="SERVER_ID")

    # API Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:80",
        env="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins_str.split(",")]

    # Database
    database_url: str = Field(
        default="postgresql://simgen:simgen@localhost:5432/simgen",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_memory_size: int = Field(default=100, env="CACHE_MEMORY_SIZE")

    # LLM APIs
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    llm_timeout: int = Field(default=30, env="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")

    # Computer Vision
    cv_max_image_size: int = Field(default=10485760, env="CV_MAX_IMAGE_SIZE")  # 10MB
    cv_confidence_threshold: float = Field(default=0.3, env="CV_CONFIDENCE_THRESHOLD")
    cv_enable_gpu: bool = Field(default=False, env="CV_ENABLE_GPU")
    cv_service_url: Optional[str] = Field(default=None, env="CV_SERVICE_URL")

    # Physics Engine
    physics_timestep: float = Field(default=0.002, env="PHYSICS_TIMESTEP")
    physics_max_simulation_time: int = Field(default=30, env="PHYSICS_MAX_SIM_TIME")
    physics_headless: bool = Field(default=True, env="PHYSICS_HEADLESS")

    # WebSocket
    ws_heartbeat_interval: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    ws_session_ttl: int = Field(default=3600, env="WS_SESSION_TTL")
    ws_max_sessions: int = Field(default=1000, env="WS_MAX_SESSIONS")

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=600, env="RATE_LIMIT_PER_HOUR")
    rate_limit_burst_size: int = Field(default=10, env="RATE_LIMIT_BURST_SIZE")

    # Request Limits
    max_request_size: int = Field(default=52428800, env="MAX_REQUEST_SIZE")  # 50MB
    max_image_size: int = Field(default=10485760, env="MAX_IMAGE_SIZE")  # 10MB
    max_text_length: int = Field(default=10000, env="MAX_TEXT_LENGTH")
    max_mjcf_size: int = Field(default=5242880, env="MAX_MJCF_SIZE")  # 5MB

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(default=5, env="CB_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(default=60, env="CB_RECOVERY_TIMEOUT")

    # Performance
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")

    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=1440, env="JWT_EXPIRATION_MINUTES")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )

    # File Storage
    upload_dir: str = Field(default="/tmp/simgen/uploads", env="UPLOAD_DIR")
    max_upload_size: int = Field(default=104857600, env="MAX_UPLOAD_SIZE")  # 100MB

    # Feature Flags
    enable_websockets: bool = Field(default=True, env="ENABLE_WEBSOCKETS")
    enable_cv_pipeline: bool = Field(default=True, env="ENABLE_CV_PIPELINE")
    enable_llm_fallback: bool = Field(default=True, env="ENABLE_LLM_FALLBACK")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'  # Ignore extra fields from .env that don't have corresponding model fields
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v, info):
        """Ensure secret key is changed in production."""
        if info.data.get("environment") == "production" and v == "dev-secret-key-change-in-production":
            raise ValueError("Secret key must be changed in production")
        return v

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

    def get_database_url_async(self) -> str:
        """Get async database URL."""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        return self.database_url

    def get_redis_url_with_db(self, db: int = 0) -> str:
        """Get Redis URL with specific database."""
        if self.redis_url.endswith("/0"):
            return self.redis_url[:-1] + str(db)
        return f"{self.redis_url}/{db}"


# Global settings instance (singleton)
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


# Export commonly used settings
DEBUG = settings.debug
ENVIRONMENT = settings.environment
DATABASE_URL = settings.database_url
REDIS_URL = settings.redis_url