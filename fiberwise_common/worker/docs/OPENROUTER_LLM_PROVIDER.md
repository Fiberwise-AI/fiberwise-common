# OpenRouter - LLM Service Provider

Access 100+ AI models from multiple providers through a unified API with competitive pricing and intelligent routing.

## 🌍 Overview

OpenRouter provides access to the best AI models from multiple providers including OpenAI, Anthropic, Meta, Google, Mistral, and more through a single unified API. With intelligent routing, competitive pricing, and transparent cost tracking, OpenRouter simplifies multi-model AI integration.

### **Why OpenRouter?**

- ✅ **Multi-Provider Access** - 100+ models from 15+ providers
- ✅ **Unified API** - OpenAI-compatible interface for all models
- ✅ **Competitive Pricing** - Often 10-50% cheaper than direct APIs
- ✅ **Intelligent Routing** - Automatic fallback and load balancing
- ✅ **Transparent Costs** - Real-time cost tracking and limits
- ✅ **No Rate Limits** - Higher throughput than many direct APIs

## 🧠 Supported Models and Providers

### **Featured Models**

| Model | Provider | Description | Cost per 1M tokens | Use Case |
|-------|----------|-------------|-------------------|----------|
| `meta-llama/llama-3.1-8b-instruct:free` | Meta | Free Llama 3.1 8B | **Free** | Development, testing |
| `meta-llama/llama-3.1-70b-instruct` | Meta | Large Llama 3.1 70B | $0.88 | Complex reasoning |
| `anthropic/claude-3.5-sonnet` | Anthropic | Latest Claude model | $3.00 | Advanced analysis |
| `openai/gpt-4o` | OpenAI | GPT-4 Omni | $5.00 | Best overall quality |
| `google/gemini-pro` | Google | Gemini Pro | $0.50 | Fast, high-quality |
| `mistralai/mistral-7b-instruct` | Mistral | Instruction-tuned 7B | $0.25 | Efficient reasoning |
| `microsoft/wizardlm-2-8x22b` | Microsoft | Mixture of Experts | $1.00 | Specialized tasks |
| `qwen/qwen-2-72b-instruct` | Alibaba | Multilingual model | $0.90 | International support |

### **Provider Breakdown**

#### **OpenAI Models**
- `openai/gpt-4o` - Latest GPT-4 with vision
- `openai/gpt-4-turbo` - Fast GPT-4 variant
- `openai/gpt-3.5-turbo` - Balanced cost/performance
- `openai/gpt-4o-mini` - Lightweight GPT-4

#### **Anthropic Models**
- `anthropic/claude-3.5-sonnet` - Latest Claude model
- `anthropic/claude-3-opus` - Highest quality Claude
- `anthropic/claude-3-sonnet` - Balanced Claude model
- `anthropic/claude-3-haiku` - Fastest Claude model

#### **Meta Models**
- `meta-llama/llama-3.1-405b-instruct` - Largest Llama model
- `meta-llama/llama-3.1-70b-instruct` - Large Llama model
- `meta-llama/llama-3.1-8b-instruct` - Efficient Llama model
- `meta-llama/llama-3.1-8b-instruct:free` - **Free** Llama model

#### **Google Models**
- `google/gemini-pro` - Gemini Pro
- `google/gemini-pro-vision` - Gemini with vision
- `google/gemma-7b-it` - Open Gemma model

#### **Mistral Models**
- `mistralai/mistral-large` - Largest Mistral model
- `mistralai/mistral-medium` - Balanced Mistral model
- `mistralai/mistral-7b-instruct` - Efficient instruction model
- `mistralai/mixtral-8x7b-instruct` - Mixture of experts

### **Specialized Models**

| Model | Provider | Specialty | Use Case |
|-------|----------|-----------|----------|
| `perplexity/llama-3.1-sonar-large-128k-online` | Perplexity | Web search + generation | Research, current events |
| `cohere/command-r-plus` | Cohere | RAG and citations | Enterprise search |
| `01-ai/yi-large` | 01.AI | Multilingual reasoning | Global applications |
| `inflection/inflection-2.5` | Inflection | Conversational AI | Customer service |

## 🚀 Setup and Configuration

### **1. Get OpenRouter Account**

#### **Create Account**
1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Go to Keys section in your dashboard
3. Create a new API key
4. Copy your key (starts with `sk-or-...`)

#### **Add Credits**
```bash
# OpenRouter uses credit-based pricing
# Add credits via dashboard: https://openrouter.ai/credits
# $10 minimum, credits never expire
```

### **2. Environment Setup**

```bash
# Required environment variables
export OPENROUTER_API_KEY="sk-or-your-key-here"

# Optional configuration
export OPENROUTER_SITE_URL="https://yourapp.com"    # Your app URL for credits
export OPENROUTER_APP_NAME="Your App Name"          # Your app name

# Test connectivity
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     -H "HTTP-Referer: https://yourapp.com" \
     -H "X-Title: Your App Name" \
     "https://openrouter.ai/api/v1/models"
```

### **3. Add to FiberWise**

#### **Via CLI (Recommended)**
```bash
# Add OpenRouter provider
fiber provider add openrouter \
  --name "OpenRouter Multi-Model" \
  --api-key "$OPENROUTER_API_KEY" \
  --model "meta-llama/llama-3.1-8b-instruct:free" \
  --site-url "https://yourapp.com" \
  --app-name "Your App Name" \
  --set-default

# Verify provider
fiber provider test openrouter
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
    'openrouter-multi',
    'OpenRouter Multi-Model',
    'openrouter',
    'https://openrouter.ai/api/v1',
    JSON_OBJECT(
        'api_key', 'sk-or-your-key-here',
        'default_model', 'meta-llama/llama-3.1-8b-instruct:free',
        'site_url', 'https://yourapp.com',
        'app_name', 'Your App Name',
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
    provider_id="openrouter-multi",
    provider_type="openrouter",
    name="OpenRouter Multi-Model",
    api_key="sk-or-your-key-here",
    default_model="meta-llama/llama-3.1-8b-instruct:free",
    site_url="https://yourapp.com",
    app_name="Your App Name"
)
```

## 🛠️ Usage Examples

### **1. Basic Multi-Model Agent**

```python
from fiberwise_sdk import FiberAgent

class OpenRouterMultiModelAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', 'Hello, how can you help me?')
        model = input_data.get('model', 'meta-llama/llama-3.1-8b-instruct:free')
        
        # Generate response using OpenRouter
        response = llm_provider.complete(
            prompt=prompt,
            provider="openrouter-multi",
            model=model,
            temperature=0.7,
            max_tokens=1500
        )
        
        return {
            'status': 'success',
            'generated_text': response.get('text', ''),
            'model_used': response.get('model', ''),
            'provider': 'openrouter',
            'finish_reason': response.get('finish_reason', ''),
            'estimated_cost': self.estimate_cost(prompt, response.get('text', ''), model)
        }
    
    def estimate_cost(self, prompt: str, response: str, model: str) -> float:
        """Estimate cost based on model pricing"""
        # Approximate token count
        total_tokens = len((prompt + response).split()) * 1.3  # Rough token estimation
        
        # Model pricing per 1M tokens (approximate)
        pricing = {
            'meta-llama/llama-3.1-8b-instruct:free': 0.00,  # Free!
            'meta-llama/llama-3.1-70b-instruct': 0.88,
            'anthropic/claude-3.5-sonnet': 3.00,
            'openai/gpt-4o': 5.00,
            'google/gemini-pro': 0.50,
            'mistralai/mistral-7b-instruct': 0.25
        }
        
        cost_per_million = pricing.get(model, 1.00)  # Default estimate
        return (total_tokens / 1_000_000) * cost_per_million
```

### **2. Cost-Optimized Model Selection**

```python
class CostOptimizedOpenRouterAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def select_optimal_model(self, task_complexity, budget_priority, quality_priority):
        """Select model based on complexity, budget, and quality requirements"""
        
        models = {
            'free': [
                'meta-llama/llama-3.1-8b-instruct:free'  # Always free option
            ],
            'budget': [
                'mistralai/mistral-7b-instruct',          # $0.25/1M tokens
                'google/gemini-pro',                      # $0.50/1M tokens
                'meta-llama/llama-3.1-8b-instruct'       # $0.18/1M tokens
            ],
            'balanced': [
                'meta-llama/llama-3.1-70b-instruct',     # $0.88/1M tokens
                'microsoft/wizardlm-2-8x22b',            # $1.00/1M tokens
                'qwen/qwen-2-72b-instruct'               # $0.90/1M tokens
            ],
            'premium': [
                'anthropic/claude-3.5-sonnet',           # $3.00/1M tokens
                'openai/gpt-4o',                         # $5.00/1M tokens
                'anthropic/claude-3-opus'                # $15.00/1M tokens
            ]
        }
        
        # Select tier based on priorities
        if budget_priority == 'free':
            tier = 'free'
        elif budget_priority == 'low' and task_complexity == 'simple':
            tier = 'budget'
        elif quality_priority == 'high' or task_complexity == 'complex':
            tier = 'premium'
        else:
            tier = 'balanced'
        
        # Return first model from selected tier
        return models[tier][0]
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        task_complexity = input_data.get('complexity', 'medium')  # simple, medium, complex
        budget_priority = input_data.get('budget', 'balanced')    # free, low, balanced, high
        quality_priority = input_data.get('quality', 'balanced')  # low, balanced, high
        
        # Select optimal model
        model = self.select_optimal_model(task_complexity, budget_priority, quality_priority)
        
        response = llm_provider.complete(
            prompt=prompt,
            provider="openrouter-multi",
            model=model,
            temperature=0.7,
            max_tokens=self.get_max_tokens(task_complexity)
        )
        
        return {
            'status': 'success',
            'text': response.get('text', ''),
            'model_selected': model,
            'selection_criteria': {
                'complexity': task_complexity,
                'budget_priority': budget_priority,
                'quality_priority': quality_priority
            }
        }
    
    def get_max_tokens(self, complexity):
        """Get appropriate token limit based on task complexity"""
        limits = {
            'simple': 512,
            'medium': 1024,
            'complex': 2048
        }
        return limits.get(complexity, 1024)
```

### **3. Multi-Model Comparison Agent**

```python
class OpenRouterComparisonAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', 'Explain quantum computing in simple terms')
        comparison_type = input_data.get('type', 'quality')  # quality, speed, cost
        
        # Define model sets for different comparisons
        model_sets = {
            'quality': [
                'anthropic/claude-3.5-sonnet',
                'openai/gpt-4o', 
                'meta-llama/llama-3.1-70b-instruct'
            ],
            'speed': [
                'anthropic/claude-3-haiku',
                'openai/gpt-3.5-turbo',
                'google/gemini-pro'
            ],
            'cost': [
                'meta-llama/llama-3.1-8b-instruct:free',
                'mistralai/mistral-7b-instruct',
                'google/gemini-pro'
            ],
            'diverse': [
                'anthropic/claude-3.5-sonnet',    # Anthropic
                'openai/gpt-4o',                 # OpenAI
                'meta-llama/llama-3.1-70b-instruct',  # Meta
                'google/gemini-pro',             # Google
                'mistralai/mistral-7b-instruct'  # Mistral
            ]
        }
        
        models = model_sets.get(comparison_type, model_sets['quality'])
        results = {}
        
        for model in models:
            try:
                start_time = time.time()
                
                response = llm_provider.complete(
                    prompt=prompt,
                    provider="openrouter-multi",
                    model=model,
                    temperature=0.7,
                    max_tokens=800
                )
                
                execution_time = time.time() - start_time
                text = response.get('text', '')
                
                results[model] = {
                    'text': text,
                    'execution_time_ms': int(execution_time * 1000),
                    'status': 'success',
                    'word_count': len(text.split()),
                    'estimated_cost': self.estimate_cost(prompt, text, model),
                    'quality_score': self.assess_quality(text),
                    'provider': model.split('/')[0]
                }
                
            except Exception as e:
                results[model] = {
                    'error': str(e),
                    'status': 'failed',
                    'provider': model.split('/')[0]
                }
        
        # Analyze results
        successful_results = [r for r in results.values() if r.get('status') == 'success']
        
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['execution_time_ms'])
            cheapest = min(successful_results, key=lambda x: x['estimated_cost'])
            highest_quality = max(successful_results, key=lambda x: x['quality_score'])
        else:
            fastest = cheapest = highest_quality = None
        
        return {
            'status': 'success',
            'prompt': prompt,
            'comparison_type': comparison_type,
            'model_results': results,
            'analysis': {
                'fastest_model': fastest,
                'cheapest_model': cheapest,
                'highest_quality_model': highest_quality,
                'total_models_tested': len(models),
                'successful_responses': len(successful_results)
            }
        }
    
    def estimate_cost(self, prompt: str, response: str, model: str) -> float:
        """Estimate cost based on model and usage"""
        total_tokens = len((prompt + response).split()) * 1.3
        
        pricing = {
            'meta-llama/llama-3.1-8b-instruct:free': 0.00,
            'mistralai/mistral-7b-instruct': 0.25,
            'google/gemini-pro': 0.50,
            'meta-llama/llama-3.1-70b-instruct': 0.88,
            'anthropic/claude-3.5-sonnet': 3.00,
            'openai/gpt-4o': 5.00
        }
        
        cost_per_million = pricing.get(model, 1.00)
        return (total_tokens / 1_000_000) * cost_per_million
    
    def assess_quality(self, text: str) -> float:
        """Simple quality assessment based on text characteristics"""
        # Basic heuristics for quality assessment
        score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(text.split())
        if 100 <= word_count <= 300:
            score += 0.2
        
        # Coherence indicators
        if '.' in text and len(text.split('.')) > 2:
            score += 0.1
        
        # Explanation quality
        explanation_words = ['because', 'therefore', 'however', 'for example', 'such as']
        if any(word in text.lower() for word in explanation_words):
            score += 0.1
        
        # Structure indicators
        if any(marker in text for marker in ['1.', '2.', '-', '*']):
            score += 0.1
        
        return min(score, 1.0)
```

### **4. Free Tier Development Agent**

```python
class FreeOpenRouterAgent(FiberAgent):
    """Agent optimized for free tier development and testing"""
    
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        use_free_only = input_data.get('free_only', True)
        
        if use_free_only:
            # Use only free models
            model = 'meta-llama/llama-3.1-8b-instruct:free'
        else:
            # Use most cost-effective paid model
            model = 'mistralai/mistral-7b-instruct'
        
        try:
            response = llm_provider.complete(
                prompt=prompt,
                provider="openrouter-multi",
                model=model,
                temperature=0.7,
                max_tokens=1000  # Reasonable limit for development
            )
            
            return {
                'status': 'success',
                'text': response.get('text', ''),
                'model': model,
                'cost': 0.0 if use_free_only else self.estimate_cost(prompt, response.get('text', ''), model),
                'is_free': use_free_only,
                'development_mode': True
            }
            
        except Exception as e:
            # Fallback to free model if paid model fails
            if not use_free_only:
                try:
                    fallback_response = llm_provider.complete(
                        prompt=prompt,
                        provider="openrouter-multi",
                        model='meta-llama/llama-3.1-8b-instruct:free',
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    return {
                        'status': 'success',
                        'text': fallback_response.get('text', ''),
                        'model': 'meta-llama/llama-3.1-8b-instruct:free',
                        'cost': 0.0,
                        'is_free': True,
                        'fallback_used': True,
                        'original_error': str(e)
                    }
                except Exception as fallback_error:
                    return {
                        'status': 'failed',
                        'error': f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
                    }
            
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def estimate_cost(self, prompt: str, response: str, model: str) -> float:
        """Estimate cost for paid models"""
        total_tokens = len((prompt + response).split()) * 1.3
        
        pricing = {
            'mistralai/mistral-7b-instruct': 0.25,
            'google/gemini-pro': 0.50
        }
        
        cost_per_million = pricing.get(model, 0.25)
        return (total_tokens / 1_000_000) * cost_per_million
```

### **5. Smart Routing Agent**

```python
class SmartRoutingOpenRouterAgent(FiberAgent):
    """Agent that intelligently routes to different models based on task type"""
    
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def detect_task_type(self, prompt: str) -> str:
        """Detect task type from prompt to select optimal model"""
        prompt_lower = prompt.lower()
        
        # Code-related tasks
        code_keywords = ['code', 'programming', 'function', 'debug', 'syntax', 'python', 'javascript']
        if any(keyword in prompt_lower for keyword in code_keywords):
            return 'code'
        
        # Creative writing
        creative_keywords = ['story', 'creative', 'poem', 'write', 'imagine', 'character']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return 'creative'
        
        # Analysis and reasoning
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'explain', 'why', 'how']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return 'analysis'
        
        # Math and science
        math_keywords = ['calculate', 'solve', 'equation', 'math', 'science', 'formula']
        if any(keyword in prompt_lower for keyword in math_keywords):
            return 'math'
        
        # Conversation/chat
        chat_keywords = ['hello', 'hi', 'how are you', 'what do you think']
        if any(keyword in prompt_lower for keyword in chat_keywords):
            return 'chat'
        
        return 'general'
    
    def select_model_for_task(self, task_type: str, budget_level: str = 'balanced') -> str:
        """Select optimal model based on task type and budget"""
        
        model_recommendations = {
            'code': {
                'free': 'meta-llama/llama-3.1-8b-instruct:free',
                'budget': 'mistralai/mistral-7b-instruct',
                'balanced': 'meta-llama/llama-3.1-70b-instruct',
                'premium': 'anthropic/claude-3.5-sonnet'
            },
            'creative': {
                'free': 'meta-llama/llama-3.1-8b-instruct:free',
                'budget': 'meta-llama/llama-3.1-8b-instruct',
                'balanced': 'meta-llama/llama-3.1-70b-instruct',
                'premium': 'openai/gpt-4o'
            },
            'analysis': {
                'free': 'meta-llama/llama-3.1-8b-instruct:free',
                'budget': 'google/gemini-pro',
                'balanced': 'meta-llama/llama-3.1-70b-instruct',
                'premium': 'anthropic/claude-3.5-sonnet'
            },
            'math': {
                'free': 'meta-llama/llama-3.1-8b-instruct:free',
                'budget': 'mistralai/mistral-7b-instruct',
                'balanced': 'microsoft/wizardlm-2-8x22b',
                'premium': 'openai/gpt-4o'
            },
            'chat': {
                'free': 'meta-llama/llama-3.1-8b-instruct:free',
                'budget': 'google/gemini-pro',
                'balanced': 'anthropic/claude-3-haiku',
                'premium': 'anthropic/claude-3.5-sonnet'
            },
            'general': {
                'free': 'meta-llama/llama-3.1-8b-instruct:free',
                'budget': 'mistralai/mistral-7b-instruct',
                'balanced': 'meta-llama/llama-3.1-70b-instruct',
                'premium': 'anthropic/claude-3.5-sonnet'
            }
        }
        
        task_models = model_recommendations.get(task_type, model_recommendations['general'])
        return task_models.get(budget_level, task_models['balanced'])
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        budget_level = input_data.get('budget', 'balanced')  # free, budget, balanced, premium
        
        # Detect task type
        task_type = self.detect_task_type(prompt)
        
        # Select optimal model
        model = self.select_model_for_task(task_type, budget_level)
        
        # Get task-specific configuration
        config = self.get_task_config(task_type)
        
        try:
            response = llm_provider.complete(
                prompt=prompt,
                provider="openrouter-multi",
                model=model,
                **config
            )
            
            return {
                'status': 'success',
                'text': response.get('text', ''),
                'routing_decision': {
                    'detected_task': task_type,
                    'selected_model': model,
                    'budget_level': budget_level,
                    'config_used': config
                },
                'estimated_cost': self.estimate_cost(prompt, response.get('text', ''), model)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'routing_decision': {
                    'detected_task': task_type,
                    'attempted_model': model,
                    'budget_level': budget_level
                }
            }
    
    def get_task_config(self, task_type: str) -> dict:
        """Get optimal configuration for each task type"""
        configs = {
            'code': {'temperature': 0.1, 'max_tokens': 2048},      # Deterministic
            'creative': {'temperature': 0.9, 'max_tokens': 1500},  # High creativity
            'analysis': {'temperature': 0.3, 'max_tokens': 1200},  # Balanced
            'math': {'temperature': 0.0, 'max_tokens': 800},       # Very deterministic
            'chat': {'temperature': 0.8, 'max_tokens': 600},       # Conversational
            'general': {'temperature': 0.7, 'max_tokens': 1000}    # Default
        }
        
        return configs.get(task_type, configs['general'])
    
    def estimate_cost(self, prompt: str, response: str, model: str) -> float:
        """Estimate cost based on model pricing"""
        total_tokens = len((prompt + response).split()) * 1.3
        
        pricing = {
            'meta-llama/llama-3.1-8b-instruct:free': 0.00,
            'mistralai/mistral-7b-instruct': 0.25,
            'google/gemini-pro': 0.50,
            'meta-llama/llama-3.1-8b-instruct': 0.18,
            'meta-llama/llama-3.1-70b-instruct': 0.88,
            'microsoft/wizardlm-2-8x22b': 1.00,
            'anthropic/claude-3-haiku': 0.50,
            'anthropic/claude-3.5-sonnet': 3.00,
            'openai/gpt-4o': 5.00
        }
        
        cost_per_million = pricing.get(model, 1.00)
        return (total_tokens / 1_000_000) * cost_per_million
```

## 📊 Performance and Cost Analytics

### **Real-Time Cost Tracking**

```python
import time
from datetime import datetime
from typing import Dict, List

class OpenRouterAnalyticsAgent(FiberAgent):
    def __init__(self):
        self.usage_tracking = {
            'total_requests': 0,
            'total_cost': 0.0,
            'model_usage': {},
            'daily_costs': {},
            'request_history': []
        }
    
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        model = input_data.get('model', 'meta-llama/llama-3.1-8b-instruct:free')
        
        start_time = time.time()
        
        try:
            response = llm_provider.complete(
                prompt=prompt,
                provider="openrouter-multi",
                model=model,
                temperature=0.7,
                max_tokens=1500
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate costs
            cost = self.calculate_precise_cost(prompt, response.get('text', ''), model)
            
            # Track usage
            self.track_usage(model, cost, execution_time, len(prompt), len(response.get('text', '')))
            
            return {
                'status': 'success',
                'text': response.get('text', ''),
                'model': model,
                'analytics': {
                    'execution_time_ms': int(execution_time * 1000),
                    'cost_usd': cost,
                    'input_tokens': self.estimate_tokens(prompt),
                    'output_tokens': self.estimate_tokens(response.get('text', '')),
                    'cost_per_token': cost / max(self.estimate_tokens(prompt + response.get('text', '')), 1)
                },
                'usage_summary': self.get_usage_summary()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'analytics': {
                    'execution_time_ms': int((time.time() - start_time) * 1000)
                }
            }
    
    def calculate_precise_cost(self, prompt: str, response: str, model: str) -> float:
        """Calculate precise cost based on actual token counts and current pricing"""
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = self.estimate_tokens(response)
        
        # OpenRouter pricing (input/output may differ for some models)
        model_pricing = {
            'meta-llama/llama-3.1-8b-instruct:free': {'input': 0.0, 'output': 0.0},
            'mistralai/mistral-7b-instruct': {'input': 0.25, 'output': 0.25},
            'google/gemini-pro': {'input': 0.125, 'output': 0.375},  # Different input/output pricing
            'meta-llama/llama-3.1-70b-instruct': {'input': 0.88, 'output': 0.88},
            'anthropic/claude-3.5-sonnet': {'input': 3.0, 'output': 15.0},  # Higher output cost
            'openai/gpt-4o': {'input': 5.0, 'output': 15.0}
        }
        
        pricing = model_pricing.get(model, {'input': 1.0, 'output': 1.0})
        
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        return input_cost + output_cost
    
    def estimate_tokens(self, text: str) -> int:
        """More accurate token estimation"""
        # Rough approximation: 1 token ≈ 0.75 words for most models
        word_count = len(text.split())
        return int(word_count / 0.75)
    
    def track_usage(self, model: str, cost: float, execution_time: float, input_length: int, output_length: int):
        """Track detailed usage statistics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Update totals
        self.usage_tracking['total_requests'] += 1
        self.usage_tracking['total_cost'] += cost
        
        # Update model usage
        if model not in self.usage_tracking['model_usage']:
            self.usage_tracking['model_usage'][model] = {
                'requests': 0,
                'total_cost': 0.0,
                'avg_execution_time': 0.0,
                'total_input_chars': 0,
                'total_output_chars': 0
            }
        
        model_stats = self.usage_tracking['model_usage'][model]
        model_stats['requests'] += 1
        model_stats['total_cost'] += cost
        model_stats['avg_execution_time'] = (
            (model_stats['avg_execution_time'] * (model_stats['requests'] - 1) + execution_time) / 
            model_stats['requests']
        )
        model_stats['total_input_chars'] += input_length
        model_stats['total_output_chars'] += output_length
        
        # Update daily costs
        if today not in self.usage_tracking['daily_costs']:
            self.usage_tracking['daily_costs'][today] = 0.0
        self.usage_tracking['daily_costs'][today] += cost
        
        # Add to request history (keep last 100)
        self.usage_tracking['request_history'].append({
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'cost': cost,
            'execution_time': execution_time
        })
        
        if len(self.usage_tracking['request_history']) > 100:
            self.usage_tracking['request_history'].pop(0)
    
    def get_usage_summary(self) -> Dict:
        """Get comprehensive usage summary"""
        return {
            'session_totals': {
                'requests': self.usage_tracking['total_requests'],
                'cost_usd': round(self.usage_tracking['total_cost'], 4),
                'avg_cost_per_request': round(
                    self.usage_tracking['total_cost'] / max(self.usage_tracking['total_requests'], 1), 4
                )
            },
            'model_breakdown': {
                model: {
                    'requests': stats['requests'],
                    'cost_usd': round(stats['total_cost'], 4),
                    'avg_execution_ms': int(stats['avg_execution_time'] * 1000),
                    'avg_cost_per_request': round(stats['total_cost'] / stats['requests'], 4)
                }
                for model, stats in self.usage_tracking['model_usage'].items()
            },
            'daily_spending': self.usage_tracking['daily_costs']
        }
```

## 💰 Cost Optimization Strategies

### **1. Free Tier Maximization**
```python
# Always use free model for development
model = 'meta-llama/llama-3.1-8b-instruct:free'

# Batch requests to minimize API overhead
batch_prompts = [
    "Explain concept A",
    "Explain concept B", 
    "Explain concept C"
]

# Process in single session to maintain context
```

### **2. Smart Model Selection**
```python
def select_cost_optimal_model(task_complexity: str, quality_needed: str) -> str:
    """Select most cost-effective model for requirements"""
    
    if quality_needed == 'basic':
        return 'meta-llama/llama-3.1-8b-instruct:free'  # Free!
    
    elif task_complexity == 'simple' and quality_needed == 'good':
        return 'mistralai/mistral-7b-instruct'  # $0.25/1M tokens
    
    elif task_complexity == 'medium':
        return 'google/gemini-pro'  # $0.50/1M tokens
    
    elif quality_needed == 'high':
        return 'anthropic/claude-3.5-sonnet'  # $3.00/1M tokens
    
    else:
        return 'meta-llama/llama-3.1-70b-instruct'  # $0.88/1M tokens
```

### **3. Token Optimization**
```python
def optimize_prompt_for_cost(prompt: str, max_cost_cents: int = 5) -> tuple:
    """Optimize prompt length and select model to stay under budget"""
    
    # Estimate tokens
    estimated_tokens = len(prompt.split()) / 0.75
    
    # Calculate costs for different models
    model_costs = {
        'meta-llama/llama-3.1-8b-instruct:free': 0,
        'mistralai/mistral-7b-instruct': (estimated_tokens / 1_000_000) * 0.25,
        'google/gemini-pro': (estimated_tokens / 1_000_000) * 0.50
    }
    
    # Find most capable model within budget
    budget_usd = max_cost_cents / 100
    affordable_models = [(model, cost) for model, cost in model_costs.items() if cost <= budget_usd]
    
    if affordable_models:
        # Select best affordable model
        best_model = min(affordable_models, key=lambda x: x[1])[0]
        return prompt, best_model
    else:
        # Truncate prompt to fit budget with cheapest paid model
        cheapest_paid = min(model_costs.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))
        model = cheapest_paid[0]
        cost_per_token = cheapest_paid[1] / estimated_tokens
        
        max_tokens = int(budget_usd / cost_per_token)
        max_words = int(max_tokens * 0.75)
        
        truncated_prompt = ' '.join(prompt.split()[:max_words])
        return truncated_prompt, model
```

## 🚨 Error Handling and Reliability

### **Advanced Error Handling with Fallbacks**

```python
class RobustOpenRouterAgent(FiberAgent):
    def get_dependencies(self):
        return {
            'llm_provider': 'LLMProvider'
        }
    
    def run_agent(self, input_data, llm_provider=None):
        prompt = input_data.get('prompt', '')
        preferred_model = input_data.get('model', 'meta-llama/llama-3.1-70b-instruct')
        
        # Define fallback chain (preferred to most reliable)
        fallback_chain = [
            preferred_model,
            'google/gemini-pro',                    # Fast, reliable
            'mistralai/mistral-7b-instruct',        # Cheaper fallback
            'meta-llama/llama-3.1-8b-instruct:free'  # Free, always available
        ]
        
        last_error = None
        
        for i, model in enumerate(fallback_chain):
            try:
                response = llm_provider.complete(
                    prompt=prompt,
                    provider="openrouter-multi",
                    model=model,
                    temperature=0.7,
                    max_tokens=1500,
                    timeout=30  # Reasonable timeout
                )
                
                return {
                    'status': 'success',
                    'text': response.get('text', ''),
                    'model_used': model,
                    'fallback_level': i,
                    'is_fallback': i > 0,
                    'available_credit': self.check_remaining_credit()
                }
                
            except Exception as e:
                last_error = e
                error_message = str(e).lower()
                
                # Check for specific error types
                if 'insufficient credit' in error_message or 'quota exceeded' in error_message:
                    # Skip to free model immediately
                    if model != 'meta-llama/llama-3.1-8b-instruct:free':
                        try:
                            response = llm_provider.complete(
                                prompt=prompt,
                                provider="openrouter-multi", 
                                model='meta-llama/llama-3.1-8b-instruct:free',
                                temperature=0.7,
                                max_tokens=1500
                            )
                            
                            return {
                                'status': 'success',
                                'text': response.get('text', ''),
                                'model_used': 'meta-llama/llama-3.1-8b-instruct:free',
                                'fallback_reason': 'insufficient_credit',
                                'original_error': str(e)
                            }
                        except Exception as free_error:
                            return {
                                'status': 'failed',
                                'error': f"Credit exhausted and free model failed: {str(free_error)}"
                            }
                
                elif 'rate limit' in error_message:
                    # Wait and retry once
                    time.sleep(2)
                    continue
                
                # Log and continue to next fallback
                print(f"Model {model} failed: {str(e)}")
                continue
        
        return {
            'status': 'failed',
            'error': f"All fallback models failed. Last error: {str(last_error)}",
            'attempted_models': fallback_chain
        }
    
    def check_remaining_credit(self) -> float:
        """Check remaining OpenRouter credit (would need API implementation)"""
        # This would require calling OpenRouter's API to check credit balance
        # For now, return a placeholder
        return 10.0  # Placeholder credit amount
```

## 📚 Best Practices

### **1. Model Selection Strategy**
- Start with `meta-llama/llama-3.1-8b-instruct:free` for development
- Use `google/gemini-pro` for balanced cost/performance
- Use `anthropic/claude-3.5-sonnet` for highest quality needs
- Use `mistralai/mistral-7b-instruct` for cost-sensitive production

### **2. Cost Management**
- Monitor daily spending with analytics agents
- Set budget alerts and limits
- Use free models for development and testing
- Optimize prompts to reduce token usage

### **3. Performance Optimization**
- Cache responses for repeated queries
- Use appropriate timeout values (30s recommended)
- Implement smart fallback chains
- Monitor response times by model

### **4. Error Handling**
- Always implement fallback to free models
- Handle credit exhaustion gracefully
- Retry with exponential backoff for rate limits
- Log errors for monitoring and debugging

### **5. Development Workflow**
- Use free tier for initial development
- Test with multiple models before production
- Monitor costs in staging environment
- Set up alerting for unexpected cost spikes

---

## 🚀 Quick Start Checklist

- [ ] Create OpenRouter account at [openrouter.ai](https://openrouter.ai)
- [ ] Generate API key from dashboard
- [ ] Add initial credits ($10 minimum)
- [ ] Set environment variables: `OPENROUTER_API_KEY`
- [ ] Add provider to FiberWise: `fiber provider add openrouter`
- [ ] Test free model: `meta-llama/llama-3.1-8b-instruct:free`
- [ ] Set up cost monitoring and alerts
- [ ] Deploy with smart routing and fallbacks

**Need Help?**
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Model Comparison](https://openrouter.ai/models)
- [Pricing Calculator](https://openrouter.ai/pricing)
- [FiberWise Discord](https://discord.gg/gUb9zKxAdv)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
