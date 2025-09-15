# Hugging Face - LLM Service Provider

Access thousands of open-source AI models through Hugging Face Inference API with FiberWise integration.

## 🌍 Overview

Hugging Face provides the world's largest collection of open-source machine learning models through their Inference API. This integration enables FiberWise to leverage thousands of pre-trained models for text generation, embeddings, classification, and specialized tasks.

### **Why Hugging Face?**

- ✅ **Massive Model Library** - 200,000+ models across all domains
- ✅ **Open Source First** - Community-driven development
- ✅ **Cost Effective** - Free tier + pay-per-use pricing
- ✅ **Specialized Models** - Domain-specific fine-tuned models
- ✅ **Easy Integration** - Simple REST API with consistent format
- ✅ **No Vendor Lock-in** - Open models you can self-host

## 🧠 Supported Model Categories

### **Text Generation Models**

| Model | Description | Use Case | Performance |
|-------|-------------|----------|-------------|
| `microsoft/DialoGPT-medium` | Conversational AI model | Chat, dialogue | Fast |
| `google/flan-t5-large` | Instruction-following model | Q&A, summarization | High quality |
| `bigscience/bloom-560m` | Multilingual text generation | Creative writing | Balanced |
| `EleutherAI/gpt-neo-2.7B` | GPT-style text generation | Content creation | Good |
| `microsoft/DialoGPT-large` | Large conversational model | Advanced chat | High quality |
| `facebook/blenderbot-400M-distill` | Blend of skills chatbot | Conversational AI | Fast |

### **Code Generation Models**

| Model | Description | Languages | Use Case |
|-------|-------------|-----------|----------|
| `microsoft/CodeBERT-base-mlm` | Code understanding | Multiple | Code analysis |
| `Salesforce/codet5-base` | Text-to-code generation | Python, Java, etc | Code generation |
| `microsoft/codebert-base` | Code-text understanding | Multiple | Code documentation |

### **Embedding Models**

| Model | Description | Dimensions | Use Case |
|-------|-------------|------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | General purpose embeddings | 384 | Semantic search |
| `sentence-transformers/all-mpnet-base-v2` | High-quality embeddings | 768 | Advanced search |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | Paraphrase detection | 384 | Similarity matching |

### **Specialized Models**

| Model | Description | Input/Output | Use Case |
|-------|-------------|--------------|----------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment analysis | Text → Sentiment | Social media analysis |
| `facebook/bart-large-cnn` | Text summarization | Long text → Summary | Document processing |
| `google/pegasus-xsum` | Abstractive summarization | Article → Summary | News summarization |
| `Helsinki-NLP/opus-mt-en-fr` | Translation EN→FR | English → French | Language translation |

## 🚀 Setup and Configuration

### **1. Get Hugging Face Account**

#### **Create Account**
1. Sign up at [Hugging Face](https://huggingface.co/)
2. Go to Settings → Access Tokens
3. Create a new token with "Read" permissions
4. Copy your token (starts with `hf_...`)

#### **Get API Token**
```bash
# From your Hugging Face settings
# Navigate to: https://huggingface.co/settings/tokens
# Token example: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### **2. Environment Setup**

```bash
# Required environment variables
export HUGGINGFACE_API_TOKEN="hf_your-token-here"

# Optional - custom endpoint (usually not needed)
export HUGGINGFACE_API_ENDPOINT="https://api-inference.huggingface.co"

# Test connectivity
curl -H "Authorization: Bearer $HUGGINGFACE_API_TOKEN" \
     "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium" \
     -d '{"inputs": "Hello, how are you?"}'
```

### **3. Add to FiberWise**

#### **Via CLI (Recommended)**
```bash
# Add Hugging Face provider
fiber provider add huggingface \
  --name "Hugging Face Models" \
  --api-key "$HUGGINGFACE_API_TOKEN" \
  --model "microsoft/DialoGPT-medium" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --set-default

# Verify provider
fiber provider test huggingface
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
    'huggingface-models',
    'Hugging Face Models',
    'huggingface',
    'https://api-inference.huggingface.co',
    JSON_OBJECT(
        'api_key', 'hf_your-token-here',
        'default_model', 'microsoft/DialoGPT-medium',
        'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2',
        'temperature', 0.7,
        'max_tokens', 1024
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
    provider_id="huggingface-models",
    provider_type="huggingface",
    name="Hugging Face Models",
    api_key="hf_your-token-here",
    default_model="microsoft/DialoGPT-medium",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

## 🛠️ Usage Examples

### **1. Basic Text Generation**

```python
from fiberwise_sdk import FiberAgent

class HuggingFaceTextAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', 'Hello, how are you today?')
        model = input_data.get('model', 'microsoft/DialoGPT-medium')
        
        # Generate text using Hugging Face
        response = llm_provider.complete(
            prompt=prompt,
            provider="huggingface-models",
            model=model,
            temperature=0.7,
            max_tokens=512
        )
        
        return {
            'status': 'success',
            'generated_text': response.get('text', ''),
            'model_used': response.get('model', ''),
            'provider': 'huggingface',
            'finish_reason': response.get('finish_reason', ''),
            'token_count': len(response.get('text', '').split())
        }
```

### **2. Multi-Model Comparison**

```python
class HuggingFaceModelComparison(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', 'Explain quantum computing in simple terms')
        
        # Test different Hugging Face models
        models = [
            "microsoft/DialoGPT-medium",        # Conversational
            "google/flan-t5-large",            # Instruction following
            "bigscience/bloom-560m",           # Multilingual
            "EleutherAI/gpt-neo-2.7B"         # General text generation
        ]
        
        results = {}
        
        for model in models:
            try:
                start_time = time.time()
                
                response = llm_provider.complete(
                    prompt=prompt,
                    provider="huggingface-models",
                    model=model,
                    temperature=0.7,
                    max_tokens=300
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
        
        return {
            'status': 'success',
            'prompt': prompt,
            'model_results': results,
            'total_models_tested': len(models),
            'successful_responses': len([r for r in results.values() if r.get('status') == 'success'])
        }
```

### **3. Sentiment Analysis Agent**

```python
class HuggingFaceSentimentAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        text = input_data.get('text', 'I love using FiberWise!')
        
        # Use specialized sentiment analysis model
        response = llm_provider.complete(
            prompt=text,
            provider="huggingface-models",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            temperature=0.1  # Low temperature for consistent classification
        )
        
        sentiment_text = response.get('text', '')
        
        # Parse sentiment from response
        sentiment_score = 'neutral'
        confidence = 0.5
        
        try:
            # Hugging Face sentiment models often return labels
            if 'positive' in sentiment_text.lower() or 'pos' in sentiment_text.lower():
                sentiment_score = 'positive'
                confidence = 0.8
            elif 'negative' in sentiment_text.lower() or 'neg' in sentiment_text.lower():
                sentiment_score = 'negative'
                confidence = 0.8
        except:
            pass
        
        return {
            'status': 'success',
            'text': text,
            'sentiment': sentiment_score,
            'confidence': confidence,
            'raw_response': sentiment_text,
            'model_used': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'provider': 'huggingface'
        }
```

### **4. Text Summarization Agent**

```python
class HuggingFaceSummarizationAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        text = input_data.get('text', '')
        max_length = input_data.get('max_length', 100)
        
        if not text:
            return {'status': 'failed', 'error': 'No text provided for summarization'}
        
        # Use BART model for summarization
        response = llm_provider.complete(
            prompt=text,
            provider="huggingface-models",
            model="facebook/bart-large-cnn",
            max_tokens=max_length
        )
        
        summary = response.get('text', '')
        
        return {
            'status': 'success',
            'original_text': text,
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()),
            'compression_ratio': round(len(summary.split()) / len(text.split()), 2),
            'model_used': 'facebook/bart-large-cnn',
            'provider': 'huggingface'
        }
```

### **5. Embedding Generation for Search**

```python
class HuggingFaceEmbeddingAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider',
            'database': 'Database'
        }
    
    def run_agent(self, input_data, llm_provider=None, database=None):
        documents = input_data.get('documents', [])
        embedding_model = input_data.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        embeddings_results = []
        
        for doc in documents:
            try:
                # Generate embeddings using Hugging Face
                embedding_response = llm_provider.get_embeddings(
                    text=doc['content'],
                    provider="huggingface-models",
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

## 🔧 Advanced Configuration

### **Model-Specific Parameters**

Different Hugging Face models support various parameters:

```python
# Conversational models
dialog_config = {
    "temperature": 0.8,          # Creativity (0.0-1.0)
    "max_new_tokens": 512,       # Response length
    "do_sample": True,           # Enable sampling
    "top_p": 0.95,              # Nucleus sampling
    "repetition_penalty": 1.1    # Reduce repetition
}

# Instruction-following models
instruction_config = {
    "temperature": 0.3,          # More deterministic
    "max_new_tokens": 800,       # Longer responses
    "do_sample": True,
    "top_k": 50                  # Top-k sampling
}

# Classification/analysis models
classification_config = {
    "temperature": 0.1,          # Very deterministic
    "max_new_tokens": 100,       # Short outputs
    "do_sample": False           # Greedy decoding
}

# Use configuration
response = llm_provider.complete(
    prompt="Analyze the sentiment of this review: The product is amazing!",
    provider="huggingface-models",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    **classification_config
)
```

### **Custom Model Selection**

```python
class AdaptiveHuggingFaceAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def get_best_model(self, task_type, text_length):
        """Select optimal model based on task and content length"""
        models = {
            'conversation': {
                'short': 'microsoft/DialoGPT-medium',
                'long': 'microsoft/DialoGPT-large'
            },
            'summarization': {
                'news': 'facebook/bart-large-cnn',
                'general': 'google/pegasus-xsum'
            },
            'sentiment': {
                'social': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'general': 'nlptown/bert-base-multilingual-uncased-sentiment'
            },
            'code': {
                'generation': 'Salesforce/codet5-base',
                'analysis': 'microsoft/CodeBERT-base-mlm'
            },
            'creative': {
                'short': 'bigscience/bloom-560m',
                'long': 'EleutherAI/gpt-neo-2.7B'
            }
        }
        
        task_models = models.get(task_type, models['conversation'])
        
        if text_length > 1000:
            return task_models.get('long', list(task_models.values())[0])
        else:
            return task_models.get('short', list(task_models.values())[0])
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        task_type = input_data.get('task_type', 'conversation')
        
        # Select optimal model
        model = self.get_best_model(task_type, len(prompt))
        
        # Get task-specific configuration
        config = self.get_task_config(task_type)
        
        response = llm_provider.complete(
            prompt=prompt,
            provider="huggingface-models",
            model=model,
            **config
        )
        
        return {
            'status': 'success',
            'text': response.get('text', ''),
            'model_selected': model,
            'task_type': task_type,
            'config_used': config
        }
    
    def get_task_config(self, task_type):
        """Get configuration optimized for specific task"""
        configs = {
            'conversation': {'temperature': 0.8, 'max_tokens': 512},
            'summarization': {'temperature': 0.3, 'max_tokens': 150},
            'sentiment': {'temperature': 0.1, 'max_tokens': 50},
            'code': {'temperature': 0.2, 'max_tokens': 800},
            'creative': {'temperature': 0.9, 'max_tokens': 1000}
        }
        
        return configs.get(task_type, configs['conversation'])
```

## 📊 Performance Optimization

### **Model Selection Guide**

| Task Type | Recommended Model | Reasoning | Expected Speed |
|-----------|------------------|-----------|----------------|
| **Chat/Conversation** | `microsoft/DialoGPT-medium` | Optimized for dialogue | Fast |
| **Instruction Following** | `google/flan-t5-large` | Excellent at following instructions | Medium |
| **Creative Writing** | `EleutherAI/gpt-neo-2.7B` | Good at creative tasks | Slow |
| **Summarization** | `facebook/bart-large-cnn` | Specialized for summarization | Medium |
| **Sentiment Analysis** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Accurate sentiment detection | Fast |
| **Code Tasks** | `Salesforce/codet5-base` | Code-specific training | Medium |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Balanced speed/quality | Fast |

### **Caching Strategy**

```python
import hashlib
import json
from typing import Dict, Any, Optional

class CachedHuggingFaceAgent(FiberAgent):
    def __init__(self):
        self._response_cache = {}
        self._cache_ttl = 600  # 10 minutes
    
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def _generate_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for request"""
        cache_data = {
            'prompt': prompt,
            'model': model,
            'provider': 'huggingface',
            **{k: v for k, v in kwargs.items() if k not in ['timestamp']}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', 'microsoft/DialoGPT-medium')
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
            provider="huggingface-models",
            model=model,
            temperature=0.7
        )
        
        result = {
            'status': 'success',
            'text': response.get('text', ''),
            'model': model,
            'provider': 'huggingface',
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

| Usage Tier | Price | Monthly Limit | Notes |
|------------|-------|---------------|-------|
| **Free Tier** | $0 | 30,000 characters/month | Rate limited |
| **Pro Tier** | $9/month | 1M characters/month | Higher rate limits |
| **Pay-per-use** | $0.60 per 1M characters | Unlimited | No rate limits |

### **Cost Optimization Strategies**

#### **1. Smart Model Selection**
```python
class CostOptimizedHuggingFaceAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def select_cost_effective_model(self, task_type, quality_requirement):
        """Select model based on cost/quality balance"""
        models = {
            'conversation': {
                'basic': 'microsoft/DialoGPT-medium',     # Cheaper, good quality
                'premium': 'microsoft/DialoGPT-large'     # More expensive, better
            },
            'text_generation': {
                'basic': 'bigscience/bloom-560m',         # Smaller, faster, cheaper
                'premium': 'EleutherAI/gpt-neo-2.7B'     # Larger, better, pricier
            }
        }
        
        task_models = models.get(task_type, models['conversation'])
        return task_models.get(quality_requirement, task_models['basic'])
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        task_type = input_data.get('task_type', 'conversation')
        quality = input_data.get('quality', 'basic')  # basic or premium
        
        model = self.select_cost_effective_model(task_type, quality)
        
        # Optimize token usage
        max_tokens = 256 if quality == 'basic' else 512
        
        response = llm_provider.complete(
            prompt=prompt,
            provider="huggingface-models",
            model=model,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            'status': 'success',
            'text': response.get('text', ''),
            'model_used': model,
            'optimization_strategy': {
                'quality': quality,
                'max_tokens': max_tokens,
                'estimated_cost': self.estimate_cost(prompt, response.get('text', ''))
            }
        }
    
    def estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate cost based on character count"""
        total_chars = len(prompt) + len(response)
        cost_per_million_chars = 0.60
        return (total_chars / 1_000_000) * cost_per_million_chars
```

## 🚨 Error Handling and Reliability

### **Comprehensive Error Handling**

```python
import asyncio
import time
from typing import Dict, Any, Optional

class ReliableHuggingFaceAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    async def run_agent_with_retries(
        self, 
        input_data, 
        llm_provider=None, 
        max_retries=3,
        base_delay=2
    ):
        """Run agent with exponential backoff retry logic"""
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', 'microsoft/DialoGPT-medium')
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await llm_provider.complete_async(
                    prompt=prompt,
                    provider="huggingface-models",
                    model=model,
                    temperature=0.7,
                    timeout=60  # HF models can be slow when cold
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
                    'ServiceUnavailableError',  # HF models loading
                    'ModelLoadingError'
                ]
                
                is_retryable = any(err in error_type for err in retryable_errors)
                
                # Special handling for model loading
                if 'loading' in error_message.lower() or 'warming up' in error_message.lower():
                    is_retryable = True
                    base_delay = max(base_delay, 10)  # Longer delay for model loading
                
                if not is_retryable or attempt == max_retries - 1:
                    return {
                        'status': 'failed',
                        'error': error_message,
                        'error_type': error_type,
                        'attempts': attempt + 1,
                        'final_attempt': True
                    }
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
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

## 📚 Best Practices

### **1. Model Selection**
- Use `microsoft/DialoGPT-medium` for conversational tasks
- Use `google/flan-t5-large` for instruction-following
- Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Use specialized models for specific tasks (sentiment, summarization, etc.)

### **2. Prompt Engineering**
- Be specific and clear in your prompts
- Include examples for better results with instruction-following models
- Keep prompts concise to reduce costs
- Test different prompt formats for optimal results

### **3. Error Handling**
- Implement retry logic with exponential backoff
- Handle model loading delays gracefully
- Use fallback models for reliability
- Monitor and log all requests

### **4. Performance Optimization**
- Cache responses when appropriate
- Use smaller models for simple tasks
- Monitor response times and adjust models as needed
- Batch similar requests when possible

### **5. Cost Management**
- Set appropriate max_tokens limits
- Use free tier for development and testing
- Monitor character usage regularly
- Implement request deduplication

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🚀 Quick Start Checklist

- [ ] Create Hugging Face account and get API token
- [ ] Set environment variable: `HUGGINGFACE_API_TOKEN`
- [ ] Add provider to FiberWise: `fiber provider add huggingface`
- [ ] Test with simple agent: `fiber provider test huggingface`
- [ ] Explore different models for your use case
- [ ] Monitor costs and optimize model selection

**Need Help?**
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Inference API Guide](https://huggingface.co/docs/api-inference/)
- [Model Hub](https://huggingface.co/models)
- [FiberWise Discord](https://discord.gg/gUb9zKxAdv)
