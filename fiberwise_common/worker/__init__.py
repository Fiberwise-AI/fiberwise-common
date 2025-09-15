"""
Worker system for processing activations with provider pattern support.

Supports local database workers and external job runners (Celery, RabbitMQ, AWS SQS, etc.)
"""

from .base import WorkerProvider, WorkerConfig, WorkerType
from .local import LocalWorker
from .factory import WorkerFactory, get_worker_provider

# AWS SQS worker (optional import)
try:
    from .aws_sqs_worker import AWSSQSWorker
    __all__ = [
        'WorkerProvider',
        'WorkerConfig',
        'WorkerType', 
        'LocalWorker',
        'AWSSQSWorker',
        'WorkerFactory',
        'get_worker_provider'
    ]
except ImportError:
    # boto3 not installed
    __all__ = [
        'WorkerProvider',
        'WorkerConfig',
        'WorkerType', 
        'LocalWorker',
        'WorkerFactory',
        'get_worker_provider'
    ]
