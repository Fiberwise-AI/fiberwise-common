"""
Worker factory for creating worker providers based on configuration.
"""

import os
from typing import Optional, Dict, Any

from .base import WorkerProvider, WorkerConfig, WorkerType
from .local import LocalWorker
from ..database.base import DatabaseProvider


class WorkerFactory:
    """Factory for creating worker providers."""
    
    @staticmethod
    def create_worker(
        config: WorkerConfig,
        db_provider: DatabaseProvider,
        llm_provider = None,
        storage_provider = None,
        base_url: Optional[str] = None
    ) -> WorkerProvider:
        """Create a worker provider based on configuration."""
        
        if config.worker_type == WorkerType.LOCAL:
            return LocalWorker(config, db_provider, llm_provider, storage_provider, base_url=base_url)
        
        elif config.worker_type == WorkerType.CELERY:
            try:
                from .celery_worker import CeleryWorker
                return CeleryWorker(config, db_provider, llm_provider, storage_provider)
            except ImportError:
                raise ImportError("Celery is required for celery worker type. Install with: pip install celery")
        
        elif config.worker_type == WorkerType.RABBITMQ:
            try:
                from .rabbitmq_worker import RabbitMQWorker
                return RabbitMQWorker(config, db_provider, llm_provider, storage_provider)
            except ImportError:
                raise ImportError("pika is required for RabbitMQ worker type. Install with: pip install pika")
        
        elif config.worker_type == WorkerType.AWS_SQS:
            try:
                from .aws_sqs_worker import AWSSQSWorker
                return AWSSQSWorker(config, db_provider, llm_provider, storage_provider)
            except ImportError:
                raise ImportError("boto3 is required for AWS SQS worker type. Install with: pip install boto3")
        
        elif config.worker_type == WorkerType.REDIS:
            try:
                from .redis_worker import RedisWorker
                return RedisWorker(config, db_provider, llm_provider, storage_provider)
            except ImportError:
                raise ImportError("redis is required for Redis worker type. Install with: pip install redis")
        
        elif config.worker_type == WorkerType.EXTERNAL:
            from .external_worker import ExternalWorker
            return ExternalWorker(config, db_provider, llm_provider, storage_provider)
        
        else:
            raise ValueError(f"Unsupported worker type: {config.worker_type}")


def get_worker_provider(
    worker_type: Optional[str] = None,
    broker_url: Optional[str] = None,
    poll_interval: int = 5,
    max_concurrent_jobs: int = 1,
    db_provider: Optional[DatabaseProvider] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> WorkerProvider:
    """
    Convenience function to create a worker provider from environment variables or parameters.
    
    Environment variables:
    - FIBERWISE_WORKER_TYPE: Type of worker (local, celery, rabbitmq, aws_sqs, redis, external)
    - FIBERWISE_WORKER_BROKER_URL: Broker URL for message queue workers
    - FIBERWISE_WORKER_POLL_INTERVAL: Polling interval in seconds
    - FIBERWISE_WORKER_MAX_CONCURRENT: Maximum concurrent jobs
    - FIBERWISE_BASE_URL: Base URL for activation service
    """
    
    # Get configuration from environment or parameters
    worker_type = worker_type or os.getenv('FIBERWISE_WORKER_TYPE', 'local')
    broker_url = broker_url or os.getenv('FIBERWISE_WORKER_BROKER_URL')
    poll_interval = int(os.getenv('FIBERWISE_WORKER_POLL_INTERVAL', poll_interval))
    max_concurrent_jobs = int(os.getenv('FIBERWISE_WORKER_MAX_CONCURRENT', max_concurrent_jobs))
    base_url = base_url or os.getenv('FIBERWISE_BASE_URL') or os.getenv('BASE_URL')
    
    # Create configuration
    config = WorkerConfig(
        worker_type=WorkerType(worker_type),
        broker_url=broker_url,
        poll_interval=poll_interval,
        max_concurrent_jobs=max_concurrent_jobs,
        **kwargs
    )
    
    # Get database provider if not provided
    if db_provider is None:
        from ..database.factory import get_database_provider
        db_provider = get_database_provider()
    
    # Create worker
    return WorkerFactory.create_worker(config, db_provider, base_url=base_url)
