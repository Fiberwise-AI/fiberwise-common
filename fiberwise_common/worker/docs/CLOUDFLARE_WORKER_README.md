# FiberWise Cloudflare Worker

Deploy AI agent processing at the edge with Cloudflare Workers for ultra-low latency and global distribution.

## 🌍 Overview

The FiberWise Cloudflare Worker enables distributed processing of AI agent activations directly at Cloudflare's edge locations worldwide. This provides sub-100ms response times and eliminates cold starts for AI processing.

### **Why Cloudflare Workers?**

- ✅ **Global Edge Network** - 200+ locations worldwide
- ✅ **Zero Cold Starts** - Always-warm execution environment
- ✅ **Ultra-Low Latency** - Sub-100ms response times
- ✅ **Auto-Scaling** - Handles millions of requests automatically
- ✅ **Cost-Effective** - Pay per request, no idle costs
- ✅ **Built-in Security** - DDoS protection and WAF included

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FiberWise     │    │   Cloudflare     │    │   Edge Workers  │
│   Core Web      │    │     Global       │    │  (200+ cities)  │
│                 │    │    Network       │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Activation  │ │───►│ │ Load         │ │───►│ │ Worker NYC  │ │
│ │  Service    │ │    │ │ Balancer     │ │    │ │ (Python VM) │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Job Queue   │ │◄──►│ │ D1 Database  │ │◄──►│ │ Worker LON  │ │
│ │  Service    │ │    │ │ (SQLite)     │ │    │ │ (Python VM) │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. **Prerequisites**

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Verify installation
wrangler whoami
```

### 2. **Project Setup**

```bash
# Create new Cloudflare Worker project
wrangler init fiberwise-worker
cd fiberwise-worker

# Install FiberWise SDK for Workers
npm install fiberwise-worker-sdk
```

### 3. **Basic Worker Implementation**

Create `src/worker.js`:

```javascript
import { FiberWiseWorker } from 'fiberwise-worker-sdk';
import { WorkerEntrypoint } from 'cloudflare:workers';

export default class FiberWiseEdgeWorker extends WorkerEntrypoint {
    async fetch(request, env) {
        const worker = new FiberWiseWorker({
            apiKey: env.FIBERWISE_API_KEY,
            databaseUrl: env.FIBERWISE_DB_URL,
            capabilities: [
                'python-execution',
                'web-scraping',
                'api-calls',
                'text-processing'
            ],
            workerId: `cf-${env.CF_REGION}-${crypto.randomUUID().substring(0, 8)}`,
            region: env.CF_REGION || 'global'
        });

        const url = new URL(request.url);
        
        try {
            switch (url.pathname) {
                case '/register':
                    return await this.handleRegister(worker, request);
                    
                case '/poll':
                    return await this.handlePoll(worker, request);
                    
                case '/execute':
                    return await this.handleExecute(worker, request);
                    
                case '/heartbeat':
                    return await this.handleHeartbeat(worker, request);
                    
                case '/health':
                    return this.handleHealth();
                    
                case '/metrics':
                    return await this.handleMetrics(worker, request);
                    
                default:
                    return new Response('FiberWise Edge Worker v1.0', { 
                        status: 200,
                        headers: { 'Content-Type': 'text/plain' }
                    });
            }
        } catch (error) {
            console.error('Worker error:', error);
            return new Response(JSON.stringify({ 
                error: error.message,
                timestamp: new Date().toISOString()
            }), { 
                status: 500,
                headers: { 'Content-Type': 'application/json' }
            });
        }
    }

    async handleRegister(worker, request) {
        const registration = await worker.register();
        
        return new Response(JSON.stringify({
            status: 'success',
            worker_id: worker.workerId,
            capabilities: worker.capabilities,
            region: worker.region,
            registered_at: new Date().toISOString()
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async handlePoll(worker, request) {
        const jobs = await worker.pollForJobs();
        
        return new Response(JSON.stringify({
            status: 'success',
            jobs: jobs,
            poll_timestamp: new Date().toISOString()
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async handleExecute(worker, request) {
        const jobData = await request.json();
        const result = await worker.executeJob(jobData);
        
        return new Response(JSON.stringify({
            status: 'completed',
            job_id: jobData.job_id,
            result: result,
            execution_time_ms: result.execution_time_ms,
            completed_at: new Date().toISOString()
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async handleHeartbeat(worker, request) {
        const heartbeat = await worker.sendHeartbeat();
        
        return new Response(JSON.stringify({
            status: 'alive',
            worker_id: worker.workerId,
            heartbeat_at: new Date().toISOString(),
            uptime_ms: heartbeat.uptime_ms
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    handleHealth() {
        return new Response(JSON.stringify({
            status: 'healthy',
            service: 'fiberwise-edge-worker',
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            region: this.env?.CF_REGION || 'unknown'
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async handleMetrics(worker, request) {
        const metrics = await worker.getMetrics();
        
        return new Response(JSON.stringify({
            worker_id: worker.workerId,
            region: worker.region,
            capabilities: worker.capabilities,
            metrics: metrics,
            timestamp: new Date().toISOString()
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}
```

### 4. **Configuration**

Create `wrangler.toml`:

```toml
name = "fiberwise-worker"
main = "src/worker.js"
compatibility_date = "2024-08-01"

# Cloudflare Workers AI binding
[ai]
binding = "AI"

# D1 Database binding
[[d1_databases]]
binding = "DB"
database_name = "fiberwise-worker-db"
database_id = "your-d1-database-id"

# KV namespace for caching
[[kv_namespaces]]
binding = "CACHE"
id = "your-kv-namespace-id"

# Environment variables
[vars]
ENVIRONMENT = "production"
WORKER_VERSION = "1.0.0"

# Secrets (set with wrangler secret put)
# FIBERWISE_API_KEY
# FIBERWISE_DB_URL
```

### 5. **Deploy Worker**

```bash
# Set secrets
wrangler secret put FIBERWISE_API_KEY
wrangler secret put FIBERWISE_DB_URL

# Deploy to Cloudflare
wrangler deploy

# Test deployment
curl https://fiberwise-worker.your-subdomain.workers.dev/health
```

## 🛠️ Advanced Implementation

### **Python Execution Support**

Using Pyodide for Python runtime in Workers:

```javascript
import { loadPyodide } from 'pyodide/pyodide.js';

export default class PythonEnabledWorker extends WorkerEntrypoint {
    constructor(env) {
        super();
        this.pyodide = null;
    }

    async initPython() {
        if (!this.pyodide) {
            this.pyodide = await loadPyodide({
                indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
            });
            
            // Install common packages
            await this.pyodide.loadPackage(['numpy', 'pandas', 'requests']);
        }
        return this.pyodide;
    }

    async executePython(code, context = {}) {
        const pyodide = await this.initPython();
        
        // Set context variables
        for (const [key, value] of Object.entries(context)) {
            pyodide.globals.set(key, value);
        }
        
        // Execute Python code
        const result = pyodide.runPython(code);
        
        return {
            result: result,
            type: typeof result,
            execution_time_ms: Date.now() - startTime
        };
    }

    async handlePythonExecution(request) {
        const { code, context } = await request.json();
        const startTime = Date.now();
        
        try {
            const result = await this.executePython(code, context);
            
            return new Response(JSON.stringify({
                status: 'success',
                result: result.result,
                execution_time_ms: result.execution_time_ms
            }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            });
        } catch (error) {
            return new Response(JSON.stringify({
                status: 'error',
                error: error.message,
                execution_time_ms: Date.now() - startTime
            }), {
                status: 500,
                headers: { 'Content-Type': 'application/json' }
            });
        }
    }
}
```

### **Cloudflare AI Integration**

Leverage Cloudflare's built-in AI models:

```javascript
export default class AIEnabledWorker extends WorkerEntrypoint {
    async handleAICompletion(request, env) {
        const { prompt, model } = await request.json();
        
        // Use Cloudflare Workers AI
        const response = await env.AI.run('@cf/meta/llama-2-7b-chat-fp16', {
            messages: [
                { role: 'user', content: prompt }
            ]
        });
        
        return new Response(JSON.stringify({
            status: 'success',
            text: response.response,
            model: model,
            provider: 'cloudflare-ai'
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async handleEmbeddings(request, env) {
        const { text, model } = await request.json();
        
        const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
            text: text
        });
        
        return new Response(JSON.stringify({
            status: 'success',
            embeddings: embeddings.data[0].embedding,
            dimensions: embeddings.data[0].embedding.length
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}
```

### **D1 Database Integration**

Store worker data in Cloudflare D1:

```javascript
export default class DatabaseWorker extends WorkerEntrypoint {
    async handleDatabaseOperation(request, env) {
        const { query, params } = await request.json();
        
        try {
            const result = await env.DB.prepare(query).bind(...params).all();
            
            return new Response(JSON.stringify({
                status: 'success',
                data: result.results,
                meta: result.meta
            }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            });
        } catch (error) {
            return new Response(JSON.stringify({
                status: 'error',
                error: error.message
            }), {
                status: 500,
                headers: { 'Content-Type': 'application/json' }
            });
        }
    }

    async logActivation(env, activationId, result) {
        await env.DB.prepare(`
            INSERT INTO activation_logs (activation_id, result, worker_id, created_at)
            VALUES (?, ?, ?, ?)
        `).bind(activationId, JSON.stringify(result), this.workerId, new Date().toISOString()).run();
    }
}
```

## 📊 Monitoring and Analytics

### **Custom Analytics**

```javascript
export default class AnalyticsWorker extends WorkerEntrypoint {
    async logMetrics(env, metrics) {
        // Send to Cloudflare Analytics
        await fetch('https://api.cloudflare.com/client/v4/accounts/{account_id}/analytics/logs', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${env.CF_API_TOKEN}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                timestamp: new Date().toISOString(),
                worker_id: this.workerId,
                metrics: metrics
            })
        });
    }

    async getAnalytics(request, env) {
        // Query Cloudflare Analytics API
        const analytics = await fetch(`https://api.cloudflare.com/client/v4/accounts/{account_id}/analytics/dashboard`, {
            headers: {
                'Authorization': `Bearer ${env.CF_API_TOKEN}`
            }
        });

        const data = await analytics.json();
        
        return new Response(JSON.stringify({
            status: 'success',
            analytics: data.result
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}
```

### **Performance Monitoring**

```javascript
export default class MonitoredWorker extends WorkerEntrypoint {
    async fetch(request, env) {
        const startTime = Date.now();
        
        try {
            const response = await super.fetch(request, env);
            
            // Log performance metrics
            await this.logPerformance(env, {
                path: new URL(request.url).pathname,
                method: request.method,
                status: response.status,
                duration_ms: Date.now() - startTime,
                region: env.CF_REGION
            });
            
            return response;
        } catch (error) {
            // Log error metrics
            await this.logError(env, {
                error: error.message,
                path: new URL(request.url).pathname,
                duration_ms: Date.now() - startTime
            });
            
            throw error;
        }
    }

    async logPerformance(env, metrics) {
        // Store in D1 database
        await env.DB.prepare(`
            INSERT INTO performance_logs (path, method, status, duration_ms, region, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        `).bind(
            metrics.path,
            metrics.method,
            metrics.status,
            metrics.duration_ms,
            metrics.region,
            new Date().toISOString()
        ).run();
    }
}
```

## 🚀 Deployment Strategies

### **Multi-Region Deployment**

```bash
# Deploy to multiple regions
wrangler deploy --name fiberwise-worker-us --env production
wrangler deploy --name fiberwise-worker-eu --env production --region eu
wrangler deploy --name fiberwise-worker-asia --env production --region asia
```

### **Canary Deployments**

```toml
# wrangler.toml
[env.canary]
name = "fiberwise-worker-canary"
routes = [
    { pattern = "worker-canary.fiberwise.ai/*", zone_name = "fiberwise.ai" }
]

[env.production]
name = "fiberwise-worker-prod"
routes = [
    { pattern = "worker.fiberwise.ai/*", zone_name = "fiberwise.ai" }
]
```

### **Traffic Splitting**

```javascript
// Route based on traffic percentage
export default class LoadBalancedWorker extends WorkerEntrypoint {
    async fetch(request, env) {
        const trafficPercent = Math.random() * 100;
        
        if (trafficPercent < 10) {
            // Route 10% to canary
            return await this.handleCanaryTraffic(request, env);
        } else {
            // Route 90% to production
            return await this.handleProductionTraffic(request, env);
        }
    }
}
```

## 🔧 Development and Testing

### **Local Development**

```bash
# Start local development server
wrangler dev

# Test endpoints locally
curl http://localhost:8787/health
curl http://localhost:8787/register
```

### **Unit Testing**

Create `test/worker.test.js`:

```javascript
import { unstable_dev } from 'wrangler';

describe('FiberWise Worker', () => {
    let worker;

    beforeAll(async () => {
        worker = await unstable_dev('src/worker.js', {
            experimental: { disableExperimentalWarning: true }
        });
    });

    afterAll(async () => {
        await worker.stop();
    });

    it('should return health status', async () => {
        const response = await worker.fetch('/health');
        const result = await response.json();
        
        expect(response.status).toBe(200);
        expect(result.status).toBe('healthy');
    });

    it('should handle worker registration', async () => {
        const response = await worker.fetch('/register');
        const result = await response.json();
        
        expect(response.status).toBe(200);
        expect(result.status).toBe('success');
        expect(result.worker_id).toBeDefined();
    });

    it('should execute Python code', async () => {
        const response = await worker.fetch('/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: 'test-123',
                code: 'result = 2 + 2',
                context: {}
            })
        });
        
        const result = await response.json();
        expect(result.status).toBe('completed');
        expect(result.result.result).toBe(4);
    });
});
```

### **Integration Testing**

```javascript
// test/integration.test.js
describe('FiberWise Worker Integration', () => {
    it('should integrate with FiberWise Core', async () => {
        // Test full activation flow
        const activation = await createTestActivation();
        const worker = await unstable_dev('src/worker.js');
        
        // Submit activation to worker
        const response = await worker.fetch('/execute', {
            method: 'POST',
            body: JSON.stringify({
                activation_id: activation.id,
                job_data: activation.data
            })
        });
        
        expect(response.status).toBe(200);
        
        // Verify result in database
        const result = await checkActivationResult(activation.id);
        expect(result.status).toBe('completed');
    });
});
```

## 🔐 Security and Best Practices

### **Secure Configuration**

```javascript
export default class SecureWorker extends WorkerEntrypoint {
    async fetch(request, env) {
        // Validate API key
        if (!this.validateApiKey(request, env)) {
            return new Response('Unauthorized', { status: 401 });
        }

        // Rate limiting
        if (await this.isRateLimited(request, env)) {
            return new Response('Rate limited', { status: 429 });
        }

        // CORS headers
        const response = await super.fetch(request, env);
        response.headers.set('Access-Control-Allow-Origin', 'https://app.fiberwise.ai');
        response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        response.headers.set('Access-Control-Allow-Headers', 'Authorization, Content-Type');

        return response;
    }

    validateApiKey(request, env) {
        const authHeader = request.headers.get('Authorization');
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            return false;
        }
        
        const token = authHeader.substring(7);
        return token === env.FIBERWISE_API_KEY;
    }

    async isRateLimited(request, env) {
        const clientIP = request.headers.get('CF-Connecting-IP');
        const key = `rate_limit:${clientIP}`;
        
        const current = await env.CACHE.get(key);
        if (current && parseInt(current) > 100) {
            return true;
        }
        
        await env.CACHE.put(key, (parseInt(current) || 0) + 1, { expirationTtl: 60 });
        return false;
    }
}
```

### **Environment Security**

```bash
# Never commit secrets - use wrangler secrets
wrangler secret put FIBERWISE_API_KEY
wrangler secret put DATABASE_URL
wrangler secret put ENCRYPTION_KEY

# Use environment-specific configurations
wrangler deploy --env production
```

## 📈 Performance Optimization

### **Caching Strategies**

```javascript
export default class CachedWorker extends WorkerEntrypoint {
    async handleCachedRequest(request, env) {
        const cacheKey = this.generateCacheKey(request);
        
        // Check cache first
        let response = await env.CACHE.get(cacheKey);
        if (response) {
            return new Response(response, {
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Cache': 'HIT'
                }
            });
        }
        
        // Generate response
        const result = await this.processRequest(request);
        
        // Cache for 5 minutes
        await env.CACHE.put(cacheKey, JSON.stringify(result), { 
            expirationTtl: 300 
        });
        
        return new Response(JSON.stringify(result), {
            headers: { 
                'Content-Type': 'application/json',
                'X-Cache': 'MISS'
            }
        });
    }
}
```

### **Resource Optimization**

```javascript
export default class OptimizedWorker extends WorkerEntrypoint {
    constructor(env) {
        super();
        // Initialize expensive resources once
        this.pythonRuntime = null;
        this.dbConnection = null;
    }

    async getOrInitPython() {
        if (!this.pythonRuntime) {
            this.pythonRuntime = await loadPyodide();
        }
        return this.pythonRuntime;
    }

    async cleanup() {
        // Clean up resources when worker is done
        if (this.pythonRuntime) {
            this.pythonRuntime.destroy();
        }
    }
}
```

## 🚨 Troubleshooting

### **Common Issues**

#### 1. **Worker Not Responding**
```
Error: Worker exceeded CPU time limit
```

**Solutions:**
- Break down large operations into smaller chunks
- Use async/await properly
- Optimize Python execution time
- Implement timeout handling

```javascript
// Add timeout to long-running operations
const timeoutPromise = new Promise((_, reject) => 
    setTimeout(() => reject(new Error('Operation timed out')), 30000)
);

const result = await Promise.race([
    this.longRunningOperation(),
    timeoutPromise
]);
```

#### 2. **Memory Limits**
```
Error: Worker exceeded memory limit
```

**Solutions:**
- Optimize memory usage in Python code
- Clear variables after use
- Use streaming for large data

```javascript
// Clean up large objects
async executePython(code) {
    const pyodide = await this.getOrInitPython();
    try {
        const result = pyodide.runPython(code);
        return result;
    } finally {
        // Clean up Python namespace
        pyodide.runPython('del globals()["__name__"]');
    }
}
```

#### 3. **Database Connection Issues**
```
Error: Database connection failed
```

**Solutions:**
- Implement retry logic
- Use connection pooling
- Handle D1 rate limits

```javascript
async executeQuery(query, params, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            return await this.env.DB.prepare(query).bind(...params).all();
        } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
        }
    }
}
```

### **Debug Mode**

```javascript
export default class DebugWorker extends WorkerEntrypoint {
    async fetch(request, env) {
        const isDebug = env.ENVIRONMENT === 'development';
        
        if (isDebug) {
            console.log('Request:', {
                url: request.url,
                method: request.method,
                headers: Object.fromEntries(request.headers)
            });
        }
        
        const response = await super.fetch(request, env);
        
        if (isDebug) {
            console.log('Response:', {
                status: response.status,
                headers: Object.fromEntries(response.headers)
            });
        }
        
        return response;
    }
}
```

## 📚 Resources

### **Documentation Links**
- [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)
- [Cloudflare D1 Database](https://developers.cloudflare.com/d1/)
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/)

### **Example Projects**
- [Basic FiberWise Worker](https://github.com/fiberwise/examples/cloudflare-basic)
- [AI-Enabled Worker](https://github.com/fiberwise/examples/cloudflare-ai)
- [Multi-Region Deployment](https://github.com/fiberwise/examples/cloudflare-multiregion)

### **Community**
- [FiberWise Discord](https://discord.gg/gUb9zKxAdv)
- [Cloudflare Discord](https://discord.gg/cloudflaredev)

## 🎯 Performance Benchmarks

| Metric | Cloudflare Worker | AWS Lambda | Google Cloud Functions |
|--------|------------------|------------|----------------------|
| **Cold Start** | <1ms | 100-3000ms | 200-2000ms |
| **Warm Latency** | 10-50ms | 10-100ms | 20-150ms |
| **Global Edge** | 200+ locations | Regional | Regional |
| **Concurrent Requests** | Unlimited | 15,000 | 1,000 |
| **Memory Limit** | 128MB | 10GB | 8GB |
| **Execution Time** | 30s (CPU time) | 15 minutes | 60 minutes |

## 🚀 Roadmap

### **Current Capabilities** ✅
- Python code execution with Pyodide
- Cloudflare AI model integration
- D1 database connectivity
- Global edge deployment
- Real-time monitoring

### **Coming Soon** 🔄
- Enhanced Python package support
- WebAssembly runtime optimization
- Advanced caching strategies
- Multi-worker orchestration
- Custom model deployment

### **Future Vision** 📋
- GPU acceleration for AI workloads
- Streaming processing capabilities
- Advanced analytics dashboard
- Auto-scaling based on queue depth
- Cross-platform worker migration

---

## 🚀 Get Started

1. **Install Wrangler**: `npm install -g wrangler`
2. **Clone template**: `git clone https://github.com/fiberwise/cloudflare-worker-template`
3. **Configure secrets**: `wrangler secret put FIBERWISE_API_KEY`
4. **Deploy**: `wrangler deploy`
5. **Monitor**: Visit your worker URL + `/health`

**Need help?** Join our [Discord community](https://discord.gg/gUb9zKxAdv) for support!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
