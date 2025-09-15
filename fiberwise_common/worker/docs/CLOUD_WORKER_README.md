# FiberWise Cloud Worker System

A distributed worker system for processing AI agent activations across multiple cloud platforms with high availability, scalability, and fault tolerance.

## Overview

The FiberWise Cloud Worker System enables distributed processing of agent activations across various cloud platforms and environments. It supports multiple worker types, from simple local processing to sophisticated cloud-based distributed systems.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FiberWise     │    │    Message       │    │  Cloud Workers  │
│   Core Web      │    │     Broker       │    │   (Multiple)    │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Activation  │ │───►│ │ Job Queue    │ │◄──►│ │  Worker 1   │ │
│ │  Service    │ │    │ │              │ │    │ │ (Cloudflare)│ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Job Router  │ │───►│ │ Worker       │ │◄──►│ │  Worker 2   │ │
│ │  Service    │ │    │ │ Registry     │ │    │ │   (AWS)     │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Supported Worker Types

### 1. **Local Worker** (Development)
- **Use Case**: Development and testing
- **Performance**: Single-threaded, local processing
- **Setup**: No additional dependencies

### 2. **External Worker** (HTTP-based)
- **Use Case**: Custom worker implementations via HTTP webhooks
- **Performance**: Network-dependent, flexible
- **Setup**: Webhook URL configuration

### 3. **AWS SQS Worker** (Cloud Queue)
- **Use Case**: AWS-native distributed processing
- **Performance**: High throughput, fault-tolerant
- **Setup**: AWS credentials and SQS queue

### 4. **Cloudflare Worker** (Edge Computing) ⭐
- **Use Case**: Global edge processing, ultra-low latency
- **Performance**: Sub-100ms response times worldwide
- **Setup**: Cloudflare account and API tokens

### 5. **Redis Worker** (In-Memory Queue)
- **Use Case**: High-speed processing with Redis
- **Performance**: Very fast, memory-based
- **Setup**: Redis instance

### 6. **Celery Worker** (Python-native)
- **Use Case**: Python ecosystem integration
- **Performance**: Scalable, mature
- **Setup**: Celery broker (Redis/RabbitMQ)

## 📦 Installation

### Core Installation
```bash
# Install base worker system
pip install fiberwise-common

# Or with specific worker extras
pip install "fiberwise-common[aws]"        # AWS SQS worker
pip install "fiberwise-common[redis]"      # Redis worker  
pip install "fiberwise-common[celery]"     # Celery worker
pip install "fiberwise-common[workers]"    # All worker types
```

### Cloud-Specific Setup

#### Cloudflare Workers
```bash
npm install -g wrangler
wrangler login
```

#### AWS SQS
```bash
pip install boto3
aws configure  # or set environment variables
```

#### Redis
```bash
pip install redis
# Start Redis server
redis-server
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FIBERWISE_WORKER_TYPE` | Worker type (`local`, `aws_sqs`, `redis`, etc.) | `local` | Yes |
| `FIBERWISE_WORKER_BROKER_URL` | Message broker URL | None | For queue-based workers |
| `FIBERWISE_WORKER_MAX_CONCURRENT` | Max concurrent jobs | `1` | No |
| `FIBERWISE_WORKER_POLL_INTERVAL` | Polling interval (seconds) | `5` | No |
| `FIBERWISE_WORKER_RETRY_ATTEMPTS` | Max retry attempts | `3` | No |
| `FIBERWISE_WORKER_TIMEOUT` | Job timeout (seconds) | `300` | No |

### Configuration Examples

#### Local Development
```bash
export FIBERWISE_WORKER_TYPE=local
export FIBERWISE_WORKER_MAX_CONCURRENT=2
```

#### AWS SQS Production
```bash
export FIBERWISE_WORKER_TYPE=aws_sqs
export FIBERWISE_WORKER_BROKER_URL=https://sqs.us-east-1.amazonaws.com/123456789012/fiberwise-activations
export FIBERWISE_WORKER_MAX_CONCURRENT=10
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

#### Redis High-Performance
```bash
export FIBERWISE_WORKER_TYPE=redis
export FIBERWISE_WORKER_BROKER_URL=redis://localhost:6379/0
export FIBERWISE_WORKER_MAX_CONCURRENT=20
```

## 🛠️ Usage

### 1. **Programmatic Usage**

```python
import asyncio
from fiberwise_common.worker import WorkerFactory, WorkerConfig, WorkerType
from fiberwise_common.database.factory import get_database_provider

async def main():
    # Get database provider
    db_provider = get_database_provider()
    
    # Create worker configuration
    config = WorkerConfig(
        worker_type=WorkerType.AWS_SQS,
        broker_url="https://sqs.us-east-1.amazonaws.com/123456789012/fiberwise-activations",
        max_concurrent_jobs=5,
        retry_attempts=3
    )
    
    # Create and start worker
    worker = WorkerFactory.create_worker(config, db_provider)
    await worker.start()
    
    # Submit an activation for processing
    success = await worker.submit_activation("activation-123", priority=1)
    print(f"Activation submitted: {success}")
    
    # Monitor worker status
    status = await worker.get_status()
    print(f"Worker status: {status}")
    
    # Stop worker gracefully
    await worker.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. **Environment-based Factory**

```python
from fiberwise_common.worker import get_worker_provider

# Creates worker based on environment variables
worker = get_worker_provider()

# Start processing
await worker.start()
```

### 3. **Integration with FiberWise Core**

```python
# In your FiberWise application
from fiberwise_common.worker import get_worker_provider

class MyApplication:
    def __init__(self):
        self.worker = get_worker_provider(
            worker_type="aws_sqs",
            max_concurrent_jobs=10
        )
    
    async def startup(self):
        await self.worker.start()
    
    async def process_activation(self, activation_id: str):
        return await self.worker.submit_activation(activation_id)
```

## 🌍 Cloud Deployments

### 1. **Cloudflare Workers** (Recommended for Edge)

#### Create Worker Script
```javascript
// cloudflare-worker.js
export default {
    async fetch(request, env) {
        const worker = new FiberWiseWorker({
            apiKey: env.FIBERWISE_API_KEY,
            databaseUrl: env.FIBERWISE_DB_URL,
            capabilities: ['python-execution', 'web-scraping']
        });
        
        const url = new URL(request.url);
        
        switch (url.pathname) {
            case '/register':
                return await worker.register();
            case '/poll':
                return await worker.pollForJobs();
            case '/execute':
                return await worker.executeJob(await request.json());
            case '/health':
                return new Response('OK', { status: 200 });
            default:
                return new Response('FiberWise Worker', { status: 200 });
        }
    }
};
```

#### Deploy to Cloudflare
```bash
# Configure wrangler
wrangler init fiberwise-worker
cd fiberwise-worker

# Add secrets
wrangler secret put FIBERWISE_API_KEY
wrangler secret put FIBERWISE_DB_URL

# Deploy
wrangler deploy
```

### 2. **AWS Lambda + SQS**

#### Lambda Function
```python
# lambda_worker.py
import json
import asyncio
from fiberwise_common.worker import get_worker_provider

def lambda_handler(event, context):
    """AWS Lambda handler for SQS worker"""
    
    async def process_messages():
        worker = get_worker_provider(worker_type="aws_sqs")
        await worker.start()
        
        # Process SQS messages
        for record in event.get('Records', []):
            activation_id = json.loads(record['body'])['activation_id']
            await worker.submit_activation(activation_id)
    
    # Run async code in Lambda
    asyncio.run(process_messages())
    
    return {
        'statusCode': 200,
        'body': json.dumps('Worker processed successfully')
    }
```

#### Deploy with AWS SAM
```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  FiberWiseWorker:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: worker/
      Handler: lambda_worker.lambda_handler
      Runtime: python3.9
      Events:
        SQSTrigger:
          Type: SQS
          Properties:
            Queue: !GetAtt FiberWiseQueue.Arn
            BatchSize: 10

  FiberWiseQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: fiberwise-activations
      VisibilityTimeoutSeconds: 300
```

### 3. **Google Cloud Run** (Container-based)

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV FIBERWISE_WORKER_TYPE=redis
ENV FIBERWISE_WORKER_BROKER_URL=redis://redis-service:6379/0

CMD ["python", "cloud_worker.py"]
```

#### Deploy to Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/fiberwise-worker
gcloud run deploy fiberwise-worker \
  --image gcr.io/PROJECT_ID/fiberwise-worker \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 4. **Kubernetes** (Enterprise Scale)

#### Deployment Configuration
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fiberwise-workers
spec:
  replicas: 10
  selector:
    matchLabels:
      app: fiberwise-worker
  template:
    metadata:
      labels:
        app: fiberwise-worker
    spec:
      containers:
      - name: worker
        image: fiberwise/cloud-worker:latest
        env:
        - name: FIBERWISE_WORKER_TYPE
          value: "redis"
        - name: FIBERWISE_WORKER_BROKER_URL
          value: "redis://redis-service:6379/0"
        - name: FIBERWISE_WORKER_MAX_CONCURRENT
          value: "5"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fiberwise-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fiberwise-workers
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📊 Monitoring and Observability

### Health Check Endpoint
```python
from fastapi import FastAPI
from fiberwise_common.worker import get_worker_provider

app = FastAPI()
worker = get_worker_provider()

@app.get("/health")
async def health_check():
    healthy = await worker.health_check()
    status = await worker.get_status()
    
    return {
        "healthy": healthy,
        "worker_type": worker.config.worker_type.value,
        "active_jobs": status.get("active_jobs", 0),
        "uptime": status.get("uptime", 0)
    }

@app.get("/metrics")
async def get_metrics():
    status = await worker.get_status()
    return {
        "worker_metrics": status,
        "timestamp": "2025-08-14T12:00:00Z"
    }
```

### Logging Configuration
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "worker_type": "%(worker_type)s"}'
)

logger = logging.getLogger(__name__)
logger = logging.LoggerAdapter(logger, {"worker_type": "aws_sqs"})
```

## 🔧 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run worker tests
pytest tests/worker/ -v

# Run specific worker type tests
pytest tests/worker/test_aws_sqs_worker.py -v
```

### Local Development Setup
```bash
# Start local dependencies
docker-compose up -d redis

# Set up environment
export FIBERWISE_WORKER_TYPE=local
export FIBERWISE_WORKER_MAX_CONCURRENT=2

# Run worker
python -m fiberwise_common.worker.cli start
```

### Testing Different Worker Types
```python
import pytest
from fiberwise_common.worker import WorkerFactory, WorkerConfig, WorkerType

@pytest.mark.asyncio
async def test_local_worker():
    config = WorkerConfig(worker_type=WorkerType.LOCAL)
    worker = WorkerFactory.create_worker(config, db_provider)
    
    await worker.start()
    success = await worker.submit_activation("test-123")
    assert success
    await worker.stop()

@pytest.mark.asyncio
async def test_redis_worker():
    config = WorkerConfig(
        worker_type=WorkerType.REDIS,
        broker_url="redis://localhost:6379/0"
    )
    worker = WorkerFactory.create_worker(config, db_provider)
    
    await worker.start()
    success = await worker.submit_activation("test-456")
    assert success
    await worker.stop()
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **Worker Not Starting**
```
Error: Worker failed to start
```

**Check:**
- Environment variables are set correctly
- Required dependencies are installed
- Message broker is accessible

```bash
# Debug worker configuration
python -c "
from fiberwise_common.worker import get_worker_provider
worker = get_worker_provider()
print(f'Worker type: {worker.config.worker_type}')
print(f'Broker URL: {worker.config.broker_url}')
"
```

#### 2. **Connection Issues**
```
Error: Failed to connect to broker
```

**Solutions:**
- Verify broker URL is correct
- Check network connectivity
- Ensure authentication credentials are valid

```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Test AWS SQS access
aws sqs list-queues
```

#### 3. **High Memory Usage**
```
Warning: Worker using excessive memory
```

**Solutions:**
- Reduce `max_concurrent_jobs`
- Implement job batching
- Monitor for memory leaks

```python
# Monitor worker memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {get_memory_usage():.2f} MB")
```

### Debug Mode
```python
import logging
logging.getLogger('fiberwise_common.worker').setLevel(logging.DEBUG)

# Enable debug mode
worker = get_worker_provider()
await worker.start(debug=True)
```

## 🎯 Performance Optimization

### Optimal Configurations

| Worker Type | Recommended Concurrent Jobs | Use Case |
|-------------|---------------------------|----------|
| **Local** | 1-2 | Development |
| **Redis** | 10-50 | High-speed processing |
| **AWS SQS** | 5-20 | Distributed processing |
| **Cloudflare** | 100+ | Edge computing |
| **External** | 1-10 | Custom integrations |

### Scaling Strategies

#### Horizontal Scaling
```python
# Deploy multiple worker instances
worker_configs = [
    WorkerConfig(worker_type=WorkerType.AWS_SQS, max_concurrent_jobs=5),
    WorkerConfig(worker_type=WorkerType.REDIS, max_concurrent_jobs=10),
    WorkerConfig(worker_type=WorkerType.CLOUDFLARE, max_concurrent_jobs=50)
]

workers = [WorkerFactory.create_worker(config, db_provider) for config in worker_configs]

# Start all workers
for worker in workers:
    await worker.start()
```

#### Auto-scaling with Kubernetes
```yaml
# Use HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fiberwise-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fiberwise-workers
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: External
    external:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "10"
```

## 🔐 Security Best Practices

### 1. **API Key Management**
```bash
# Use environment variables, never hardcode
export FIBERWISE_API_KEY="secure-api-key"
export AWS_ACCESS_KEY_ID="access-key"

# Or use cloud secret managers
aws secretsmanager get-secret-value --secret-id fiberwise/api-key
```

### 2. **Network Security**
```python
# Configure secure connections
config = WorkerConfig(
    worker_type=WorkerType.REDIS,
    broker_url="rediss://username:password@secure-redis.example.com:6380/0",  # SSL
    timeout=30
)
```

### 3. **Authentication**
```python
# Implement worker authentication
class SecureWorker(WorkerProvider):
    def __init__(self, config, db_provider):
        super().__init__(config, db_provider)
        self.api_key = os.getenv('FIBERWISE_API_KEY')
        
    async def authenticate(self):
        # Verify API key with central service
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # ... authentication logic
```

## 🚀 Roadmap

### Current Features ✅
- Multiple worker types (Local, External, AWS SQS, Redis, Celery)
- Configurable concurrency and retry logic
- Health checks and monitoring
- Environment-based configuration

### Planned Features 🔄
- **Cloudflare Workers integration** - Edge computing support
- **Worker capability registration** - Skill-based job routing  
- **Advanced load balancing** - Smart job distribution
- **Real-time metrics** - Performance dashboards
- **Auto-scaling integration** - Dynamic worker scaling

### Future Enhancements 📋
- **Multi-cloud failover** - Automatic provider switching
- **Job prioritization** - Priority-based processing
- **Worker pools** - Grouped worker management
- **Custom worker types** - Plugin architecture

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/fiberwise/fiberwise-common.git
cd fiberwise-common

# Install development dependencies
pip install -e ".[dev,workers]"

# Run tests
pytest tests/worker/ -v

# Run type checks
mypy fiberwise_common/worker/
```

### Adding New Worker Types
1. Create new worker class inheriting from `WorkerProvider`
2. Implement required abstract methods
3. Add to `WorkerFactory` 
4. Add configuration options
5. Write comprehensive tests
6. Update documentation

## 📚 Additional Resources

- **[FiberWise Documentation](https://docs.fiberwise.ai/workers)**
- **[API Reference](https://docs.fiberwise.ai/api/workers)**
- **[Examples Repository](https://github.com/fiberwise/examples)**
- **[Discord Community](https://discord.gg/gUb9zKxAdv)**

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Quick Start:**
1. `pip install "fiberwise-common[workers]"`
2. Set environment variables
3. `python -c "from fiberwise_common.worker import get_worker_provider; worker = get_worker_provider()"`
4. Deploy to your preferred cloud platform

**Need Help?** Join our [Discord community](https://discord.gg/gUb9zKxAdv) for support and discussions!
