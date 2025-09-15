# Cloudflare Workers AI - LLM Service Provider

Integrate Cloudflare Workers AI models into FiberWise for edge-based AI inference with ultra-low latency and global distribution.

## 🌍 Overview

Cloudflare Workers AI provides serverless AI inference at the edge through Cloudflare's global network. This integration enables FiberWise to leverage high-performance AI models with sub-100ms response times from over 200 global locations.

### **Why Cloudflare Workers AI?**

- ✅ **Edge-First AI** - Run inference at 200+ global edge locations
- ✅ **Ultra-Low Latency** - Sub-100ms response times worldwide
- ✅ **No Cold Starts** - Always-warm AI inference infrastructure
- ✅ **Cost-Effective** - Competitive pricing with pay-per-request model
- ✅ **High Availability** - Built-in redundancy and failover
- ✅ **Easy Integration** - Simple REST API with standardized responses

## 🧠 Supported AI Models

### **Text Generation Models**

| Model | Description | Context Length | Use Case | Performance |
|-------|-------------|----------------|----------|-------------|
| `@cf/meta/llama-2-7b-chat-fp16` | Llama 2 7B Chat (16-bit) | 4,096 tokens | General conversation, Q&A | High quality |
| `@cf/meta/llama-2-7b-chat-int8` | Llama 2 7B Chat (8-bit) | 4,096 tokens | Faster inference, good quality | Very fast |
| `@cf/meta/llama-2-13b-chat-fp16` | Llama 2 13B Chat | 4,096 tokens | Complex reasoning, detailed responses | Highest quality |
| `@cf/mistral/mistral-7b-instruct-v0.1` | Mistral 7B Instruct | 8,192 tokens | Code generation, instruction following | Fast + accurate |
| `@cf/mistral/mistral-7b-instruct-v0.2` | Mistral 7B v0.2 | 32,768 tokens | Long-context tasks | Extended context |
| `@cf/microsoft/dialoGPT-medium` | DialoGPT Medium | 1,024 tokens | Conversational AI | Optimized for chat |
| `@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0` | TinyLlama 1.1B | 2,048 tokens | Lightweight, fast responses | Ultra-fast |

### **Embedding Models**

| Model | Description | Dimensions | Use Case | Languages |
|-------|-------------|------------|----------|-----------|
| `@cf/baai/bge-base-en-v1.5` | BGE Base English v1.5 | 768 | General purpose embeddings | English |
| `@cf/baai/bge-small-en-v1.5` | BGE Small English v1.5 | 384 | Fast, smaller embeddings | English |
| `@cf/baai/bge-large-en-v1.5` | BGE Large English v1.5 | 1,024 | High-quality embeddings | English |
| `@cf/baai/bge-m3` | BGE-M3 Multilingual | 1,024 | Multilingual embeddings | 100+ languages |

### **Specialized Models**

| Model | Description | Input/Output | Use Case |
|-------|-------------|--------------|----------|
| `@cf/microsoft/resnet-50` | ResNet-50 Image Classification | Image → Labels | Image recognition |
| `@cf/meta/m2m100-1.2b` | M2M100 Translation | Text → Text | Language translation |
| `@cf/huggingface/distilbert-sst-2-int8` | DistilBERT Sentiment | Text → Sentiment | Sentiment analysis |
| `@cf/openai/whisper` | Whisper ASR | Audio → Text | Speech recognition |

## 🚀 Setup and Configuration

### **1. Get Cloudflare Credentials**

#### **Create Cloudflare Account**
1. Sign up at [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to Workers & Pages
3. Enable Workers AI (may require payment method)

#### **Get Account ID**
```bash
# From dashboard: Right sidebar under "Account ID"
# Example: 1234567890abcdef1234567890abcdef
```

#### **Create API Token**
1. Go to [API Tokens](https://dash.cloudflare.com/profile/api-tokens)
2. Click "Create Token" → "Custom token"
3. **Token name**: `fiberwise-ai-token`
4. **Permissions**: 
   - `Cloudflare AI:Edit` (for model inference)
   - `Account:Read` (for account access)
5. **Account resources**: Include your account
6. **Zone resources**: All zones (or specific zones if preferred)

### **2. Environment Setup**

```bash
# Required environment variables
export CLOUDFLARE_API_TOKEN="your-api-token-here"
export CLOUDFLARE_ACCOUNT_ID="your-account-id-here"

# Optional - custom endpoint (usually not needed)
export CLOUDFLARE_API_ENDPOINT="https://api.cloudflare.com/client/v4"

# Test connectivity
curl -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
     "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/models/list"
```

### **3. Add to FiberWise**

#### **Via CLI (Recommended)**
```bash
# Add Cloudflare Workers AI provider
fiber provider add cloudflare \
  --name "Cloudflare Workers AI" \
  --api-key "$CLOUDFLARE_API_TOKEN" \
  --account-id "$CLOUDFLARE_ACCOUNT_ID" \
  --model "@cf/meta/llama-2-7b-chat-fp16" \
  --embedding-model "@cf/baai/bge-base-en-v1.5" \
  --set-default

# Verify provider
fiber provider test cloudflare
fiber provider list
```

#### **Via Database**
```sql
INSERT INTO llm_providers (
    provider_id, 
    name, 
    provider_type, 
    api_endpoint, 
    configuration, 
    is_active, 
    is_default
) VALUES (
    'cloudflare-workers-ai',
    'Cloudflare Workers AI',
    'cloudflare',
    'https://api.cloudflare.com/client/v4',
    JSON_OBJECT(
        'api_key', 'your-cloudflare-api-token',
        'account_id', 'your-cloudflare-account-id',
        'default_model', '@cf/meta/llama-2-7b-chat-fp16',
        'embedding_model', '@cf/baai/bge-base-en-v1.5',
        'temperature', 0.7,
        'max_tokens', 2048
    ),
    TRUE,
    FALSE
);
```

#### **Via Python SDK**
```python
from fiberwise_sdk import FiberWiseConfig

config = FiberWiseConfig()
config.add_llm_provider(
    provider_id="cloudflare-ai",
    provider_type="cloudflare",
    name="Cloudflare Workers AI",
    api_key="your-cloudflare-api-token",
    account_id="your-cloudflare-account-id",
    default_model="@cf/meta/llama-2-7b-chat-fp16",
    embedding_model="@cf/baai/bge-base-en-v1.5"
)
```

## 🛠️ Usage Examples

### **1. Basic Text Generation**

```python
from fiberwise_sdk import FiberAgent

class CloudflareTextAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', 'Hello, how are you?')
        model = input_data.get('model', '@cf/meta/llama-2-7b-chat-fp16')
        
        # Generate text using Cloudflare Workers AI
        response = llm_provider.complete(
            prompt=prompt,
            provider="cloudflare-ai",
            model=model,
            temperature=0.7,
            max_tokens=1000
        )
        
        return {
            'status': 'success',
            'generated_text': response.get('text', ''),
            'model_used': response.get('model', ''),
            'provider': 'cloudflare-workers-ai',
            'finish_reason': response.get('finish_reason', ''),
            'token_count': len(response.get('text', '').split())
        }
```

### **2. Multi-Model Comparison**

```python
class CloudflareModelComparison(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', 'Explain quantum computing in simple terms')
        
        # Test different Cloudflare AI models
        models = [
            "@cf/meta/llama-2-7b-chat-fp16",      # High quality
            "@cf/meta/llama-2-7b-chat-int8",       # Fast inference
            "@cf/mistral/mistral-7b-instruct-v0.1", # Good at instructions
            "@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0" # Ultra fast
        ]
        
        results = {}
        
        for model in models:
            try:
                start_time = time.time()
                
                response = llm_provider.complete(
                    prompt=prompt,
                    provider="cloudflare-ai",
                    model=model,
                    temperature=0.7,
                    max_tokens=500
                )
                
                execution_time = time.time() - start_time
                
                results[model] = {
                    'text': response.get('text', ''),
                    'execution_time_ms': int(execution_time * 1000),
                    'status': 'success',
                    'token_count': len(response.get('text', '').split()),
                    'finish_reason': response.get('finish_reason', '')
                }
                
            except Exception as e:
                results[model] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Analyze results
        fastest_model = min(
            [r for r in results.values() if r.get('status') == 'success'],
            key=lambda x: x['execution_time_ms'],
            default=None
        )
        
        return {
            'status': 'success',
            'prompt': prompt,
            'model_results': results,
            'analysis': {
                'fastest_model': fastest_model,
                'total_models_tested': len(models),
                'successful_responses': len([r for r in results.values() if r.get('status') == 'success'])
            }
        }
```

### **3. Code Generation Agent**

```python
class CloudflareCodeGenerator(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        task = input_data.get('task', 'Create a Python function to sort a list')
        language = input_data.get('language', 'python')
        
        # Use Mistral for code generation (better at instructions)
        prompt = f"""
Generate a {language} solution for the following task:
{task}

Requirements:
- Include proper error handling
- Add clear comments
- Follow best practices
- Provide usage examples

Code:
"""
        
        response = llm_provider.complete(
            prompt=prompt,
            provider="cloudflare-ai",
            model="@cf/mistral/mistral-7b-instruct-v0.1",
            temperature=0.3,  # Lower temperature for code
            max_tokens=1500
        )
        
        generated_code = response.get('text', '')
        
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', generated_code, re.DOTALL)
        
        return {
            'status': 'success',
            'task': task,
            'language': language,
            'generated_code': generated_code,
            'code_blocks': code_blocks,
            'model_used': '@cf/mistral/mistral-7b-instruct-v0.1',
            'provider': 'cloudflare-workers-ai'
        }
```

### **4. Embedding Generation for Search**

```python
class CloudflareEmbeddingAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider',
            'database': 'Database'
        }
    
    def run_agent(self, input_data, llm_provider=None, database=None):
        documents = input_data.get('documents', [])
        embedding_model = input_data.get('model', '@cf/baai/bge-base-en-v1.5')
        
        embeddings_results = []
        
        for doc in documents:
            try:
                # Generate embeddings using Cloudflare
                embedding_response = llm_provider.get_embeddings(
                    text=doc['content'],
                    provider="cloudflare-ai",
                    model=embedding_model
                )
                
                embedding = embedding_response.get('embeddings', [])
                
                if embedding:
                    embeddings_results.append({
                        'document_id': doc['id'],
                        'embedding': embedding,
                        'dimensions': len(embedding),
                        'status': 'success'
                    })
                    
                    # Store in database for search (pseudo-code)
                    # await database.store_embedding(doc['id'], embedding)
                    
                else:
                    embeddings_results.append({
                        'document_id': doc['id'],
                        'error': 'No embeddings returned',
                        'status': 'failed'
                    })
                    
            except Exception as e:
                embeddings_results.append({
                    'document_id': doc['id'],
                    'error': str(e),
                    'status': 'failed'
                })
        
        successful_embeddings = [r for r in embeddings_results if r.get('status') == 'success']
        
        return {
            'status': 'success',
            'processed_documents': len(documents),
            'successful_embeddings': len(successful_embeddings),
            'embedding_model': embedding_model,
            'embedding_dimensions': successful_embeddings[0]['dimensions'] if successful_embeddings else 0,
            'results': embeddings_results
        }
```

### **5. Conversational Chat Agent**

```python
class CloudflareChatAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        message = input_data.get('message', '')
        conversation_history = input_data.get('history', [])
        system_prompt = input_data.get('system_prompt', 'You are a helpful AI assistant.')
        
        # Build conversation context
        context = f"System: {system_prompt}\n\n"
        
        # Add conversation history
        for exchange in conversation_history[-5:]:  # Keep last 5 exchanges
            context += f"Human: {exchange['human']}\n"
            context += f"Assistant: {exchange['assistant']}\n\n"
        
        # Add current message
        context += f"Human: {message}\nAssistant:"
        
        # Generate response using Llama 2 Chat model
        response = llm_provider.complete(
            prompt=context,
            provider="cloudflare-ai",
            model="@cf/meta/llama-2-7b-chat-fp16",
            temperature=0.8,
            max_tokens=800
        )
        
        assistant_reply = response.get('text', '').strip()
        
        # Update conversation history
        updated_history = conversation_history + [{
            'human': message,
            'assistant': assistant_reply,
            'timestamp': datetime.now().isoformat()
        }]
        
        return {
            'status': 'success',
            'reply': assistant_reply,
            'conversation_history': updated_history,
            'model_used': '@cf/meta/llama-2-7b-chat-fp16',
            'provider': 'cloudflare-workers-ai',
            'context_length': len(context.split()),
            'response_length': len(assistant_reply.split())
        }
```

## 🔧 Advanced Configuration

### **Model-Specific Parameters**

Different Cloudflare AI models support various parameters:

```python
# Llama 2 models configuration
llama_config = {
    "temperature": 0.7,          # Randomness (0.0-1.0)
    "top_p": 0.9,               # Nucleus sampling
    "max_tokens": 2048,         # Response length
    "repetition_penalty": 1.1,   # Reduce repetition
    "frequency_penalty": 0.0,    # Frequency-based penalty
    "presence_penalty": 0.0      # Presence-based penalty
}

# Mistral models configuration
mistral_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 8192,         # Higher context window
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

# TinyLlama configuration (optimized for speed)
tiny_llama_config = {
    "temperature": 0.8,
    "max_tokens": 1024,         # Shorter responses for speed
    "top_p": 0.9
}

# Use configuration
response = llm_provider.complete(
    prompt="Write a technical blog post about serverless AI",
    provider="cloudflare-ai",
    model="@cf/mistral/mistral-7b-instruct-v0.1",
    **mistral_config
)
```

### **Custom Provider Configuration**

```python
class CustomCloudflareAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def get_model_config(self, task_type):
        """Get optimized configuration based on task type"""
        configs = {
            'creative': {
                'model': '@cf/meta/llama-2-7b-chat-fp16',
                'temperature': 0.9,
                'top_p': 0.95,
                'max_tokens': 1500
            },
            'analytical': {
                'model': '@cf/mistral/mistral-7b-instruct-v0.1',
                'temperature': 0.3,
                'top_p': 0.8,
                'max_tokens': 2000
            },
            'conversational': {
                'model': '@cf/microsoft/dialoGPT-medium',
                'temperature': 0.8,
                'max_tokens': 800
            },
            'fast': {
                'model': '@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0',
                'temperature': 0.7,
                'max_tokens': 500
            }
        }
        
        return configs.get(task_type, configs['analytical'])
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        task_type = input_data.get('task_type', 'analytical')
        
        config = self.get_model_config(task_type)
        
        response = llm_provider.complete(
            prompt=prompt,
            provider="cloudflare-ai",
            **config
        )
        
        return {
            'status': 'success',
            'text': response.get('text', ''),
            'task_type': task_type,
            'model_used': config['model'],
            'config_used': config
        }
```

## 📊 Performance Optimization

### **Model Selection Guide**

| Task Type | Recommended Model | Reasoning | Expected Latency |
|-----------|------------------|-----------|------------------|
| **Creative Writing** | `@cf/meta/llama-2-7b-chat-fp16` | Best creativity and coherence | 2-5 seconds |
| **Code Generation** | `@cf/mistral/mistral-7b-instruct-v0.1` | Excellent instruction following | 2-4 seconds |
| **Quick Q&A** | `@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0` | Ultra-fast responses | 0.5-1.5 seconds |
| **Detailed Analysis** | `@cf/meta/llama-2-13b-chat-fp16` | Highest quality reasoning | 4-8 seconds |
| **Chat/Conversation** | `@cf/microsoft/dialoGPT-medium` | Optimized for dialogue | 1-3 seconds |
| **Long Context** | `@cf/mistral/mistral-7b-instruct-v0.2` | 32k context window | 3-6 seconds |

### **Performance Monitoring**

```python
import time
from datetime import datetime

class PerformanceMonitoredAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', '@cf/meta/llama-2-7b-chat-fp16')
        
        # Track performance metrics
        start_time = time.time()
        request_timestamp = datetime.utcnow().isoformat()
        
        try:
            response = llm_provider.complete(
                prompt=prompt,
                provider="cloudflare-ai",
                model=model,
                temperature=0.7
            )
            
            end_time = time.time()
            
            # Calculate metrics
            total_time_ms = int((end_time - start_time) * 1000)
            tokens_generated = len(response.get('text', '').split())
            tokens_per_second = tokens_generated / (total_time_ms / 1000) if total_time_ms > 0 else 0
            
            # Log performance (replace with your monitoring system)
            performance_metrics = {
                'provider': 'cloudflare-workers-ai',
                'model': model,
                'total_time_ms': total_time_ms,
                'tokens_generated': tokens_generated,
                'tokens_per_second': round(tokens_per_second, 2),
                'prompt_length': len(prompt.split()),
                'timestamp': request_timestamp,
                'status': 'success'
            }
            
            print(f"Performance: {total_time_ms}ms, {tokens_per_second:.1f} tokens/sec")
            
            return {
                'status': 'success',
                'text': response.get('text', ''),
                'performance': performance_metrics
            }
            
        except Exception as e:
            end_time = time.time()
            error_time_ms = int((end_time - start_time) * 1000)
            
            error_metrics = {
                'provider': 'cloudflare-workers-ai',
                'model': model,
                'error_time_ms': error_time_ms,
                'error': str(e),
                'timestamp': request_timestamp,
                'status': 'failed'
            }
            
            return {
                'status': 'failed',
                'error': str(e),
                'performance': error_metrics
            }
```

### **Caching Strategy**

```python
import hashlib
import json
from typing import Dict, Any, Optional

class CachedCloudflareAgent(FiberAgent):
    def __init__(self):
        self._response_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def _generate_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for request"""
        cache_data = {
            'prompt': prompt,
            'model': model,
            'provider': 'cloudflare-ai',
            **{k: v for k, v in kwargs.items() if k not in ['timestamp']}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', '@cf/meta/llama-2-7b-chat-fp16')
        use_cache = input_data.get('use_cache', True)
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(prompt, model)
            cached_response = self._response_cache.get(cache_key)
            
            if cached_response and time.time() - cached_response['timestamp'] < self._cache_ttl:
                cached_response['from_cache'] = True
                return cached_response['data']
        
        # Generate new response
        response = llm_provider.complete(
            prompt=prompt,
            provider="cloudflare-ai",
            model=model,
            temperature=0.7
        )
        
        result = {
            'status': 'success',
            'text': response.get('text', ''),
            'model': model,
            'provider': 'cloudflare-workers-ai',
            'from_cache': False
        }
        
        # Cache the response
        if use_cache:
            cache_key = self._generate_cache_key(prompt, model)
            self._response_cache[cache_key] = {
                'data': result.copy(),
                'timestamp': time.time()
            }
        
        return result
```

## 💰 Cost Optimization

### **Pricing Structure** (as of August 2025)

| Model Type | Pricing | Notes |
|------------|---------|-------|
| **Text Generation** | ~$0.01 per 1K tokens | Very competitive |
| **Embeddings** | ~$0.0001 per 1K tokens | Extremely affordable |
| **Image Processing** | ~$0.005 per image | Image classification |
| **Audio Processing** | ~$0.02 per minute | Speech recognition |

### **Cost Optimization Strategies**

#### **1. Smart Model Selection**
```python
class CostOptimizedAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def select_optimal_model(self, task_complexity, speed_priority):
        """Select model based on complexity and speed requirements"""
        if speed_priority == 'ultra_fast':
            return '@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0'  # Cheapest, fastest
        elif task_complexity == 'simple':
            return '@cf/meta/llama-2-7b-chat-int8'  # Good balance
        elif task_complexity == 'complex':
            return '@cf/meta/llama-2-13b-chat-fp16'  # Best quality
        else:
            return '@cf/meta/llama-2-7b-chat-fp16'  # Default
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        task_complexity = input_data.get('complexity', 'medium')  # simple, medium, complex
        speed_priority = input_data.get('speed', 'normal')  # normal, fast, ultra_fast
        
        # Select optimal model for cost/performance balance
        model = self.select_optimal_model(task_complexity, speed_priority)
        
        # Optimize max_tokens based on task
        max_tokens_map = {
            'simple': 500,
            'medium': 1000,
            'complex': 2000
        }
        max_tokens = max_tokens_map.get(task_complexity, 1000)
        
        response = llm_provider.complete(
            prompt=prompt,
            provider="cloudflare-ai",
            model=model,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            'status': 'success',
            'text': response.get('text', ''),
            'model_selected': model,
            'optimization_strategy': {
                'complexity': task_complexity,
                'speed_priority': speed_priority,
                'max_tokens': max_tokens
            }
        }
```

#### **2. Token Usage Optimization**
```python
def optimize_prompt_length(self, prompt: str, max_length: int = 2000) -> str:
    """Optimize prompt length to reduce token costs"""
    if len(prompt.split()) <= max_length:
        return prompt
    
    # Truncate while preserving important context
    words = prompt.split()
    truncated = words[:max_length]
    return ' '.join(truncated) + "... [truncated for efficiency]"

def batch_process_prompts(self, prompts: list, llm_provider) -> list:
    """Batch process multiple prompts efficiently"""
    results = []
    
    # Group similar prompts for better caching
    grouped_prompts = {}
    for i, prompt in enumerate(prompts):
        key = f"batch_{len(prompt.split())//100}"  # Group by approximate length
        if key not in grouped_prompts:
            grouped_prompts[key] = []
        grouped_prompts[key].append((i, prompt))
    
    # Process each group
    for group_prompts in grouped_prompts.values():
        for i, prompt in group_prompts:
            result = llm_provider.complete(
                prompt=prompt,
                provider="cloudflare-ai",
                model="@cf/meta/llama-2-7b-chat-int8",  # Use efficient model
                max_tokens=800  # Limit response length
            )
            results.append((i, result))
    
    # Sort results back to original order
    results.sort(key=lambda x: x[0])
    return [result[1] for result in results]
```

## 🚨 Error Handling and Reliability

### **Comprehensive Error Handling**

```python
import asyncio
import time
from typing import Dict, Any, Optional

class ReliableCloudflareAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    async def run_agent_with_retries(
        self, 
        input_data, 
        llm_provider=None, 
        max_retries=3,
        base_delay=1
    ):
        """Run agent with exponential backoff retry logic"""
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', '@cf/meta/llama-2-7b-chat-fp16')
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await llm_provider.complete_async(
                    prompt=prompt,
                    provider="cloudflare-ai",
                    model=model,
                    temperature=0.7,
                    timeout=30  # 30 second timeout
                )
                
                end_time = time.time()
                
                return {
                    'status': 'success',
                    'text': response.get('text', ''),
                    'model': model,
                    'attempt': attempt + 1,
                    'execution_time_ms': int((end_time - start_time) * 1000)
                }
                
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                
                # Determine if error is retryable
                retryable_errors = [
                    'TimeoutError',
                    'ConnectionError', 
                    'HTTPError',
                    'RateLimitError'
                ]
                
                is_retryable = any(err in error_type for err in retryable_errors)
                
                if not is_retryable or attempt == max_retries - 1:
                    return {
                        'status': 'failed',
                        'error': error_message,
                        'error_type': error_type,
                        'attempts': attempt + 1,
                        'final_attempt': True
                    }
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed: {error_message}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        return {
            'status': 'failed',
            'error': 'Max retries exceeded',
            'attempts': max_retries
        }
    
    def run_agent(self, input_data, llm_provider=None):
        """Synchronous wrapper for async retry logic"""
        return asyncio.run(self.run_agent_with_retries(input_data, llm_provider))
```

### **Fallback Strategy**

```python
class FallbackCloudflareAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        
        # Define fallback chain (best to worst performance)
        fallback_models = [
            '@cf/meta/llama-2-7b-chat-fp16',    # Primary choice
            '@cf/meta/llama-2-7b-chat-int8',     # Faster fallback
            '@cf/mistral/mistral-7b-instruct-v0.1', # Different architecture
            '@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0' # Last resort
        ]
        
        last_error = None
        
        for i, model in enumerate(fallback_models):
            try:
                print(f"Trying model {i+1}/{len(fallback_models)}: {model}")
                
                response = llm_provider.complete(
                    prompt=prompt,
                    provider="cloudflare-ai",
                    model=model,
                    temperature=0.7,
                    timeout=20  # Shorter timeout for fallbacks
                )
                
                return {
                    'status': 'success',
                    'text': response.get('text', ''),
                    'model_used': model,
                    'fallback_level': i,
                    'is_fallback': i > 0
                }
                
            except Exception as e:
                last_error = e
                print(f"Model {model} failed: {str(e)}")
                
                if i < len(fallback_models) - 1:
                    print("Trying next fallback model...")
                    continue
        
        # All models failed
        return {
            'status': 'failed',
            'error': f"All fallback models failed. Last error: {str(last_error)}",
            'fallback_attempts': len(fallback_models)
        }
```

## 🔍 Monitoring and Analytics

### **Comprehensive Monitoring**

```python
import logging
from datetime import datetime
import json

class MonitoredCloudflareAgent(FiberAgent):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost_usd': 0.0,
            'average_latency_ms': 0.0
        }
    
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def calculate_cost(self, model: str, tokens: int) -> float:
        """Calculate approximate cost based on model and tokens"""
        cost_per_1k_tokens = {
            '@cf/meta/llama-2-7b-chat-fp16': 0.010,
            '@cf/meta/llama-2-7b-chat-int8': 0.008,
            '@cf/meta/llama-2-13b-chat-fp16': 0.015,
            '@cf/mistral/mistral-7b-instruct-v0.1': 0.010,
            '@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0': 0.005,
        }
        
        base_cost = cost_per_1k_tokens.get(model, 0.010)
        return (tokens / 1000) * base_cost
    
    def log_request_metrics(
        self, 
        success: bool, 
        model: str, 
        latency_ms: int, 
        tokens: int,
        error: str = None
    ):
        """Log comprehensive request metrics"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
            self.metrics['total_tokens'] += tokens
            cost = self.calculate_cost(model, tokens)
            self.metrics['total_cost_usd'] += cost
            
            # Update average latency
            current_avg = self.metrics['average_latency_ms']
            self.metrics['average_latency_ms'] = (
                (current_avg * (self.metrics['successful_requests'] - 1) + latency_ms) 
                / self.metrics['successful_requests']
            )
            
            self.logger.info(json.dumps({
                'event': 'llm_request_success',
                'provider': 'cloudflare-workers-ai',
                'model': model,
                'latency_ms': latency_ms,
                'tokens': tokens,
                'cost_usd': round(cost, 6),
                'timestamp': datetime.utcnow().isoformat()
            }))
        else:
            self.metrics['failed_requests'] += 1
            
            self.logger.error(json.dumps({
                'event': 'llm_request_failure',
                'provider': 'cloudflare-workers-ai',
                'model': model,
                'latency_ms': latency_ms,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            }))
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', '@cf/meta/llama-2-7b-chat-fp16')
        
        start_time = time.time()
        
        try:
            response = llm_provider.complete(
                prompt=prompt,
                provider="cloudflare-ai",
                model=model,
                temperature=0.7
            )
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            text = response.get('text', '')
            tokens = len(text.split()) + len(prompt.split())  # Approximate
            
            # Log success metrics
            self.log_request_metrics(
                success=True,
                model=model,
                latency_ms=latency_ms,
                tokens=tokens
            )
            
            return {
                'status': 'success',
                'text': text,
                'model': model,
                'metrics': {
                    'latency_ms': latency_ms,
                    'tokens': tokens,
                    'estimated_cost_usd': round(self.calculate_cost(model, tokens), 6)
                }
            }
            
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Log failure metrics
            self.log_request_metrics(
                success=False,
                model=model,
                latency_ms=latency_ms,
                tokens=0,
                error=str(e)
            )
            
            return {
                'status': 'failed',
                'error': str(e),
                'model': model,
                'metrics': {
                    'latency_ms': latency_ms
                }
            }
    
    def get_summary_metrics(self):
        """Get comprehensive metrics summary"""
        success_rate = (
            self.metrics['successful_requests'] / self.metrics['total_requests'] * 100
            if self.metrics['total_requests'] > 0 else 0
        )
        
        return {
            'provider': 'cloudflare-workers-ai',
            'total_requests': self.metrics['total_requests'],
            'success_rate_percent': round(success_rate, 2),
            'total_tokens_processed': self.metrics['total_tokens'],
            'total_cost_usd': round(self.metrics['total_cost_usd'], 4),
            'average_latency_ms': round(self.metrics['average_latency_ms'], 1),
            'cost_per_request_usd': round(
                self.metrics['total_cost_usd'] / max(self.metrics['successful_requests'], 1), 
                6
            )
        }
```

## 🚀 Migration and Integration

### **Migrating from Other Providers**

#### **From OpenAI**
```python
# Before (OpenAI)
response = llm_provider.complete(
    prompt=prompt,
    provider="openai",
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

# After (Cloudflare Workers AI)
response = llm_provider.complete(
    prompt=prompt,
    provider="cloudflare-ai",
    model="@cf/meta/llama-2-7b-chat-fp16",  # Similar capability
    temperature=0.7,
    max_tokens=1000
)
```

#### **From Anthropic**
```python
# Before (Anthropic)
response = llm_provider.complete(
    prompt=prompt,
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    temperature=0.7
)

# After (Cloudflare Workers AI)
response = llm_provider.complete(
    prompt=prompt,
    provider="cloudflare-ai",
    model="@cf/mistral/mistral-7b-instruct-v0.1",  # Good instruction following
    temperature=0.7
)
```

### **Hybrid Multi-Provider Setup**

```python
class HybridLLMAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        task_type = input_data.get('task_type', 'general')
        
        # Route to optimal provider based on task
        provider_routing = {
            'creative': ('cloudflare-ai', '@cf/meta/llama-2-7b-chat-fp16'),
            'analytical': ('openai', 'gpt-4'),
            'code': ('cloudflare-ai', '@cf/mistral/mistral-7b-instruct-v0.1'),
            'fast': ('cloudflare-ai', '@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0'),
            'general': ('cloudflare-ai', '@cf/meta/llama-2-7b-chat-fp16')
        }
        
        provider, model = provider_routing.get(task_type, provider_routing['general'])
        
        try:
            response = llm_provider.complete(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=0.7
            )
            
            return {
                'status': 'success',
                'text': response.get('text', ''),
                'provider_used': provider,
                'model_used': model,
                'task_type': task_type,
                'routing_decision': 'primary'
            }
            
        except Exception as e:
            # Fallback to Cloudflare if primary provider fails
            if provider != 'cloudflare-ai':
                try:
                    fallback_response = llm_provider.complete(
                        prompt=prompt,
                        provider="cloudflare-ai",
                        model="@cf/meta/llama-2-7b-chat-fp16",
                        temperature=0.7
                    )
                    
                    return {
                        'status': 'success',
                        'text': fallback_response.get('text', ''),
                        'provider_used': 'cloudflare-ai',
                        'model_used': '@cf/meta/llama-2-7b-chat-fp16',
                        'task_type': task_type,
                        'routing_decision': 'fallback',
                        'primary_error': str(e)
                    }
                except Exception as fallback_error:
                    return {
                        'status': 'failed',
                        'error': f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
                    }
            else:
                return {
                    'status': 'failed',
                    'error': str(e)
                }
```

## 📚 Best Practices

### **1. Model Selection**
- Use `@cf/meta/llama-2-7b-chat-fp16` for general tasks
- Use `@cf/mistral/mistral-7b-instruct-v0.1` for code generation
- Use `@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0` for speed-critical applications
- Use `@cf/baai/bge-base-en-v1.5` for embeddings

### **2. Prompt Engineering**
- Be specific and clear in your prompts
- Use system messages for context setting
- Include examples for better results
- Keep prompts concise to reduce costs

### **3. Error Handling**
- Implement retry logic with exponential backoff
- Use fallback models for reliability
- Monitor and log all requests
- Handle rate limiting gracefully

### **4. Performance Optimization**
- Cache responses when appropriate
- Use appropriate timeout values
- Monitor latency and adjust models as needed
- Batch similar requests when possible

### **5. Cost Management**
- Set appropriate max_tokens limits
- Use smaller models for simple tasks
- Implement request deduplication
- Monitor costs regularly

## 🛣️ Roadmap

### **Current Features** ✅
- Text generation with multiple models
- Embeddings generation  
- Error handling and retries
- Performance monitoring
- Cost tracking

### **Coming Soon** 🔄
- Streaming responses
- Fine-tuning support (when available)
- Advanced function calling
- Multi-modal capabilities

### **Future Enhancements** 📋
- Custom model deployment
- Advanced analytics dashboard
- Auto-scaling integration
- Cross-model ensemble responses

---

## 🚀 Quick Start Checklist

- [ ] Get Cloudflare account and enable Workers AI
- [ ] Create API token with appropriate permissions
- [ ] Set environment variables (`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`)
- [ ] Add provider to FiberWise: `fiber provider add cloudflare`
- [ ] Test with simple agent: `fiber provider test cloudflare`
- [ ] Deploy your first Cloudflare AI-powered agent
- [ ] Monitor performance and costs

**Need Help?**
- [Cloudflare Workers AI Docs](https://developers.cloudflare.com/workers-ai/)
- [FiberWise Documentation](https://docs.fiberwise.ai)
- [Discord Community](https://discord.gg/gUb9zKxAdv)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
