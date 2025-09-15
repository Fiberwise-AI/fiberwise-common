"""
Base worker provider interface and configuration.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

from ..database.base import DatabaseProvider

# Optional imports for provider types
try:
    from ..llm.base import LLMProvider
except ImportError:
    LLMProvider = None

try:
    from ..storage.base import StorageProvider  
except ImportError:
    StorageProvider = None


class WorkerType(Enum):
    """Types of worker implementations."""
    LOCAL = "local"
    CELERY = "celery"
    RABBITMQ = "rabbitmq"
    AWS_SQS = "aws_sqs"
    REDIS = "redis"
    EXTERNAL = "external"


@dataclass
class WorkerConfig:
    """Configuration for worker providers."""
    worker_type: WorkerType
    poll_interval: int = 5
    max_concurrent_jobs: int = 1
    retry_attempts: int = 3
    retry_delay: int = 60
    timeout: int = 300
    
    # Provider-specific settings
    broker_url: Optional[str] = None
    queue_name: str = "fiberwise_activations"
    
    # External worker settings
    webhook_url: Optional[str] = None
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.worker_type, str):
            self.worker_type = WorkerType(self.worker_type)


class WorkerProvider(ABC):
    """Base interface for all worker providers."""
    
    def __init__(
        self, 
        config: WorkerConfig,
        db_provider: DatabaseProvider,
        llm_provider: Optional[Any] = None,
        storage_provider: Optional[Any] = None
    ):
        self.config = config
        self.db_provider = db_provider
        self.llm_provider = llm_provider
        self.storage_provider = storage_provider
        self.running = False
    
    @abstractmethod
    async def start(self) -> None:
        """Start the worker to process activations."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        pass
    
    @abstractmethod
    async def submit_activation(self, activation_id: str, priority: int = 1) -> bool:
        """Submit an activation for processing."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current worker status and statistics."""
        pass
    
    @property
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self.running
    
    async def health_check(self) -> bool:
        """Perform health check on worker."""
        try:
            status = await self.get_status()
            return status.get('healthy', False)
        except Exception:
            return False


class WorkerJob:
    """Represents a single activation job."""
    
    def __init__(self, activation_id: str, activation_data: Dict[str, Any]):
        self.activation_id = activation_id
        self.activation_data = activation_data
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.retry_count: int = 0
    
    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None
    
    @property
    def is_failed(self) -> bool:
        return self.error is not None
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
