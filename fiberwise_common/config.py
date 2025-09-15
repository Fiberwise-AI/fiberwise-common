"""
Shared configuration management for FiberWise applications.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Go up until we find a directory with pyproject.toml or setup.py
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    return current.parent


def load_environment_files(base_path: Optional[Path] = None) -> None:
    """Load environment variables from .env files."""
    if base_path is None:
        base_path = get_project_root()
    
    # Check for .fiberwise directory in user home first (highest priority)
    fiberwise_home = Path.home() / ".fiberwise"
    
    env_files = []
    
    # Priority 1: .fiberwise directory in user home
    if fiberwise_home.exists():
        env_files.extend([
            fiberwise_home / ".env",
            fiberwise_home / f".env.{os.getenv('ENVIRONMENT', 'development')}"
        ])
    
    # Priority 2: Project directory
    env_files.extend([
        base_path / ".env",
        base_path / ".env.local",
        base_path / f".env.{os.getenv('ENVIRONMENT', 'development')}"
    ])
    
    for env_file in env_files:
        if env_file.exists():
            # Load environment file if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
            except ImportError:
                pass


class BaseWebSettings(BaseSettings):
    """Base settings class for FiberWise web applications."""
    
    # Application info
    PROJECT_NAME: str = "FiberWise"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "FiberWise Application"
    API_PREFIX: str = "/api"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Server settings
    DEFAULT_HOST: str = "127.0.0.1"
    DEFAULT_PORT: int = 8000
    
    # Database settings (can be overridden in subclasses)
    DATABASE_URL: str = "sqlite:///./fiberwise.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = "change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Worker settings
    WORKER_ENABLED: bool = False  # Disabled by default
    WORKER_COUNT: int = 1
    WORKER_HEARTBEAT_INTERVAL: int = 30  # seconds
    
    # Additional worker configuration for refactored system
    WORKER_POLL_INTERVAL: int = 5
    WORKER_MAX_CONCURRENT_JOBS: int = 1
    WORKER_RETRY_ATTEMPTS: int = 3
    WORKER_RETRY_DELAY: int = 60
    WORKER_TIMEOUT: int = 300
    WORKER_QUEUE_NAME: str = "fiberwise_activations"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Web-specific settings
    ALGORITHM: str = "HS256"
    BASE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: List[str] = ["*"]  # Alias for BACKEND_CORS_ORIGINS for compatibility
    
    # Storage settings
    STORAGE_PROVIDER: str = "local"  # Options: local, s3, azure, gcp
    UPLOADS_DIR: str = "app_uploads"
    APP_BUNDLES_DIR: str = "app_bundles/apps"
    ENTITY_BUNDLES_DIR: str = "app_entity_bundles"
    
    # AWS S3 configuration
    S3_BUCKET_NAME: str = ""
    S3_REGION: str = "us-east-1"
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_ENDPOINT_URL: Optional[str] = None
    
    # Azure Blob Storage configuration
    AZURE_CONNECTION_STRING: str = ""
    AZURE_CONTAINER_NAME: str = ""
    
    # Google Cloud Storage configuration  
    GCP_BUCKET_NAME: str = ""
    GCP_CREDENTIALS_FILE: str = ""
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields for flexibility
        
    def __init__(self, **kwargs):
        # Load environment files before initializing
        load_environment_files()
        super().__init__(**kwargs)
        
        # Set default DATABASE_URL to use home folder if not already set via environment
        if not os.getenv('DATABASE_URL'):
            from pathlib import Path
            home_db_path = Path.home() / '.fiberwise' / 'fiberwise.db'
            self.DATABASE_URL = f"sqlite:///{home_db_path}"
        
        # Override settings from environment variables if they exist
        self.STORAGE_PROVIDER = os.getenv('STORAGE_PROVIDER', self.STORAGE_PROVIDER)
        self.UPLOADS_DIR = os.getenv('UPLOADS_DIR', self.UPLOADS_DIR)
        self.APP_BUNDLES_DIR = os.getenv('APP_BUNDLES_DIR', self.APP_BUNDLES_DIR)
        self.ENTITY_BUNDLES_DIR = os.getenv('ENTITY_BUNDLES_DIR', self.ENTITY_BUNDLES_DIR)
        self.S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', self.S3_BUCKET_NAME)
        self.S3_REGION = os.getenv('S3_REGION', self.S3_REGION)
        self.S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID', self.S3_ACCESS_KEY_ID)
        self.S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY', self.S3_SECRET_ACCESS_KEY)
        self.S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', self.S3_ENDPOINT_URL)
        self.BASE_URL = os.getenv('BASE_URL', self.BASE_URL)
        self.ALGORITHM = os.getenv('ALGORITHM', self.ALGORITHM)
        self.WORKER_ENABLED = os.getenv('WORKER_ENABLED', 'false').lower() == 'true'
        
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "echo": self.DATABASE_ECHO
        }
        
    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.BACKEND_CORS_ORIGINS,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }
        
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT.lower() in ["development", "dev", "local"]
        
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() in ["production", "prod"]
        
    def get_database_provider(self) -> str:
        """Get the database provider from the DATABASE_URL."""
        if self.DATABASE_URL.startswith("sqlite"):
            return "sqlite"
        elif self.DATABASE_URL.startswith(("postgresql", "postgres")):
            return "postgresql"
        elif self.DATABASE_URL.startswith("mysql"):
            return "mysql"
        else:
            return "unknown"