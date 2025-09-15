from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = "Fiberwise Common"
    secret_key: str = "change-me-in-production"
    debug: bool = False
    
    # Optional fields from environment
    gemini_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    app_env: str = "development"
    fiberwise_app_id: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields

settings = Settings()
