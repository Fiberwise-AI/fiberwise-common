# Fiberwise Worker Configuration Examples

## FastAPI Web Application Integration

The FiberWise worker system is now fully integrated with FastAPI applications and supports background processing of agent activations.

### Quick Start - FastAPI Integration

```python
# main.py - FastAPI Application with Worker
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import os

from worker.worker_service_adapter import WorkerServiceAdapter
from api.core.config import CoreWebSettings

settings = CoreWebSettings()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Initialize worker if enabled
    if settings.WORKER_ENABLED:
        logger.info("Starting background worker...")
        worker_adapter = WorkerServiceAdapter(app.state.db_manager.provider)
        app.state.worker_adapter = worker_adapter
        await worker_adapter.start()
    
    yield
    
    # Shutdown - Stop worker gracefully
    if settings.WORKER_ENABLED and hasattr(app.state, 'worker_adapter'):
        try:
            await app.state.worker_adapter.stop()
            logger.info("Worker service stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping worker service: {e}")

app = FastAPI(lifespan=lifespan)

# Worker is now running in background and processing activations!
```

### Environment Configuration

```bash
# Enable worker processing
WORKER_ENABLED=true

# Worker configuration (optional - defaults shown)
WORKER_POLL_INTERVAL=5
WORKER_MAX_CONCURRENT_JOBS=1
WORKER_RETRY_ATTEMPTS=3
WORKER_RETRY_DELAY=60
WORKER_TIMEOUT=300
WORKER_QUEUE_NAME=fiberwise_activations
```

### HTTP API Endpoints

The worker system provides comprehensive HTTP API endpoints:

```bash
# Check worker status
curl http://localhost:6307/api/v1/worker/status

# Response:
{
  "enabled": true,
  "status": "running",
  "worker_type": "local",
  "running": true,
  "active_jobs": 0,
  "max_concurrent_jobs": 1,
  "poll_interval": 5
}

# Health check
curl http://localhost:6307/api/v1/worker/health

# Control operations
curl -X POST http://localhost:6307/api/v1/worker/start
curl -X POST http://localhost:6307/api/v1/worker/stop
curl -X POST http://localhost:6307/api/v1/worker/restart
```

## Local Database Worker (Default)

The default worker uses database polling with `SELECT FOR UPDATE SKIP LOCKED`:

```python
from fiberwise_common.worker import get_worker_provider, WorkerConfig, WorkerType

# Local worker using database polling
worker = get_worker_provider(
    worker_type="local",
    poll_interval=5,
    max_concurrent_jobs=2,
    db_provider=db_provider
)

# Start worker (runs in background task)
await worker.start()
```

**Features:**
- ✅ No external dependencies
- ✅ Automatic retry logic  
- ✅ Health checks and status monitoring
- ✅ Graceful shutdown with job completion
- ✅ Background task execution (non-blocking)

## Advanced Worker Types (Future Support)

### Celery with Redis
```python
# Environment variables
FIBERWISE_WORKER_TYPE=celery
FIBERWISE_WORKER_BROKER_URL=redis://localhost:6379/0

# Or programmatically
worker = get_worker_provider(
    worker_type="celery",
    broker_url="redis://localhost:6379/0",
    max_concurrent_jobs=10
)
```

### AWS SQS
```python
# Environment variables
FIBERWISE_WORKER_TYPE=aws_sqs
FIBERWISE_WORKER_BROKER_URL=https://sqs.us-west-2.amazonaws.com/123456789012/fiberwise-queue

# Or programmatically
worker = get_worker_provider(
    worker_type="aws_sqs",
    broker_url="https://sqs.us-west-2.amazonaws.com/123456789012/fiberwise-queue",
    max_concurrent_jobs=20
)
```

### RabbitMQ
```python
# Environment variables
FIBERWISE_WORKER_TYPE=rabbitmq
FIBERWISE_WORKER_BROKER_URL=amqp://guest:guest@localhost:5672/

# Or programmatically
worker = get_worker_provider(
    worker_type="rabbitmq",
    broker_url="amqp://guest:guest@localhost:5672/",
    queue_name="fiberwise_activations"
)
```

## Real-World Usage Examples

### Development Environment
```bash
# Fast polling for development
export WORKER_ENABLED=true
export WORKER_POLL_INTERVAL=2
export WORKER_MAX_CONCURRENT_JOBS=1

# Start server with worker
python -m uvicorn main:app --port 6307 --host 0.0.0.0
```

### Production Environment
```bash
# Production configuration
export WORKER_ENABLED=true
export WORKER_POLL_INTERVAL=5
export WORKER_MAX_CONCURRENT_JOBS=4
export WORKER_RETRY_ATTEMPTS=5

# Start with multiple processes
python -m uvicorn main:app --port 6307 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Configure worker
ENV WORKER_ENABLED=true
ENV WORKER_POLL_INTERVAL=5
ENV WORKER_MAX_CONCURRENT_JOBS=2

# Health check using worker API
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:6307/api/v1/worker/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6307"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fiberwise-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: fiberwise/app:latest
        env:
        - name: WORKER_ENABLED
          value: "true"
        - name: WORKER_MAX_CONCURRENT_JOBS
          value: "4"
        ports:
        - containerPort: 6307
        livenessProbe:
          httpGet:
            path: /api/v1/worker/health
            port: 6307
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/worker/status
            port: 6307
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Monitoring and Observability

### Structured Logging
```
2025-08-08 12:47:58 - worker.worker_service_adapter - INFO - Worker service started successfully
2025-08-08 12:47:58 - fiberwise_common.worker.local - INFO - Starting local worker with 5s polling interval
2025-08-08 12:47:58 - fiberwise_common.worker.local - INFO - Max concurrent jobs: 1
```

### Monitoring Script
```python
import requests
import time
import logging

def monitor_worker():
    """Simple worker monitoring script"""
    while True:
        try:
            response = requests.get("http://localhost:6307/api/v1/worker/status")
            status = response.json()
            
            if status['running']:
                logging.info(f"✅ Worker healthy - {status['active_jobs']} active jobs")
            else:
                logging.error("❌ Worker not running")
                
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
        
        time.sleep(30)

if __name__ == "__main__":
    monitor_worker()
```

## Integration with Agent Activation System

The worker processes agent activations seamlessly:

```python
# Agent activation triggers worker processing
from api.routes.agents import activate_agent

# When an activation is created:
activation_response = await activate_agent(
    app_id="my-app",
    agent_id="my-agent", 
    inputs={"user_message": "Hello AI!"}
)

# The worker automatically:
# 1. Picks up the activation from database
# 2. Loads the agent and its configuration
# 3. Initializes LLM services and tools
# 4. Processes the activation asynchronously  
# 5. Updates status in real-time via WebSocket
# 6. Stores results and handles errors
```

## Worker Architecture

```
FastAPI Application
├── WorkerServiceAdapter 
│   └── fiberwise_common.worker.LocalWorker
│       ├── Database polling (SELECT FOR UPDATE SKIP LOCKED)
│       ├── ActivationProcessor (from common)
│       ├── Service injection (LLM, Storage, OAuth)
│       └── Background task execution
└── HTTP API (/api/v1/worker/*)
    ├── Status monitoring
    ├── Health checks  
    ├── Control operations
    └── Configuration details
```

## Worker Types Comparison

| Type | Use Case | Requirements | Scalability | Status |
|------|----------|--------------|-------------|---------|
| `local` | Development, single instance | Database only | Low-Medium | ✅ **Active** |
| `celery` | Production, distributed | Redis/RabbitMQ | High | 🔄 Future |
| `rabbitmq` | Production, message queue | RabbitMQ server | High | 🔄 Future |
| `aws_sqs` | Cloud, serverless | AWS credentials | Very High | 🔄 Future |
| `redis` | Production, simple queue | Redis server | Medium | 🔄 Future |
| `external` | Custom job runners | Webhook endpoint | Variable | 🔄 Future |

## Troubleshooting

### Common Issues

**Worker Won't Start**
```bash
# Check environment variables
echo $WORKER_ENABLED

# Check logs
docker logs your-container | grep -i worker

# Verify status
curl http://localhost:6307/api/v1/worker/status
```

**No Activations Processing**  
```bash
# Check LLM providers
curl http://localhost:6307/api/v1/llm-providers

# Check database for stuck activations
SELECT * FROM activations WHERE status = 'pending' AND created_at < NOW() - INTERVAL '5 minutes';

# Restart worker
curl -X POST http://localhost:6307/api/v1/worker/restart
```

**Performance Issues**
```bash
# Increase concurrent jobs
export WORKER_MAX_CONCURRENT_JOBS=4

# Reduce polling frequency 
export WORKER_POLL_INTERVAL=10

# Monitor active jobs
curl http://localhost:6307/api/v1/worker/jobs
```

## Migration from Custom Workers

The worker system has been migrated from custom implementations to the common provider pattern:

**Before (Custom):**
- ❌ 1,243 lines of duplicate code
- ❌ Inconsistent implementations
- ❌ Limited scalability options

**After (Common Provider):**
- ✅ Single source of truth in `fiberwise_common`
- ✅ Consistent across CLI and web
- ✅ Multiple worker types supported
- ✅ Background task execution
- ✅ Comprehensive HTTP API
- ✅ Production-ready monitoring

This migration provides a robust foundation for scaling from development to enterprise deployments.
