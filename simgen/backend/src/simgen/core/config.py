from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    app_name: str = "SimGen - AI Physics Simulation Generator"
    version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Redis Configuration
    redis_url: str = Field(..., env="REDIS_URL")
    
    # AI API Keys
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # MuJoCo Configuration
    mujoco_gl: str = "egl"  # Headless rendering
    mujoco_max_threads: int = 4
    
    # Simulation Limits
    max_simulation_duration: float = 30.0
    max_concurrent_simulations: int = 10
    max_refinement_iterations: int = 5
    generation_timeout: float = 120.0
    
    # File Storage
    storage_backend: str = "local"  # or "s3"
    local_storage_path: str = "./storage"
    aws_bucket_name: Optional[str] = Field(None, env="AWS_BUCKET_NAME")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = 60 * 24 * 8  # 8 days
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    
    @property
    def cors_origins_list(self) -> list:
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()