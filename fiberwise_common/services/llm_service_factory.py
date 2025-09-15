"""
LLM Provider service for FiberWise SDK.
Manages LLM provider connections and standardizes responses.
"""

import json
import logging
import httpx
import asyncio
from typing import Dict, Any, List, Optional, Union, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator

from .fiberwise_config import FiberWiseConfig

logger = logging.getLogger(__name__)

class BaseLLMService(ABC):
    """Base class for LLM service providers"""
    
    @abstractmethod
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion from the LLM provider"""
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate embeddings from the LLM provider"""
        pass

class OpenAIService(BaseLLMService):
    """Service for OpenAI API calls"""
    
    def __init__(self, api_key: str, api_endpoint: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using OpenAI"""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ['temperature', 'max_tokens']:
                payload[key] = value
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/chat/completions", 
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "text": result["choices"][0]["message"]["content"],
                    "provider": "openai",
                    "model": model,
                    "finish_reason": result["choices"][0].get("finish_reason", ""),
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002", **kwargs) -> List[float]:
        """Generate embeddings using OpenAI"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "input": text
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/embeddings", 
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return result["data"][0]["embedding"]
            except Exception as e:
                logger.error(f"OpenAI embedding API error: {str(e)}")
                raise

class AnthropicService(BaseLLMService):
    """Service for Anthropic API calls"""
    
    def __init__(self, api_key: str, api_endpoint: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using Anthropic"""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/messages", 
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "text": result["content"][0]["text"],
                    "provider": "anthropic",
                    "model": model,
                    "finish_reason": result.get("stop_reason", ""),
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"Anthropic API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        """Anthropic currently doesn't support embeddings directly"""
        raise NotImplementedError("Anthropic doesn't support embeddings API")

class GoogleAIService(BaseLLMService):
    """Service for Google AI API calls"""
    
    def __init__(self, api_key: str, api_endpoint: str = "https://generativelanguage.googleapis.com/v1"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using Google AI"""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/models/{model}:generateContent?key={self.api_key}",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "text": result["candidates"][0]["content"]["parts"][0]["text"],
                    "provider": "google",
                    "model": model,
                    "finish_reason": result["candidates"][0].get("finishReason", ""),
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"Google AI API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate embeddings using Google AI"""
        payload = {
            "model": model,
            "text": text
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/models/{model}:embedContent?key={self.api_key}",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return result["embedding"]["values"]
            except Exception as e:
                logger.error(f"Google AI embedding API error: {str(e)}")
                raise

class OllamaService(BaseLLMService):
    """Service for Ollama API calls"""
    
    def __init__(self, api_endpoint: str = "http://localhost:11434/api"):
        self.api_endpoint = api_endpoint
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using Ollama"""
        temperature = kwargs.get('temperature', 0.7)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/generate",
                    json=payload,
                    timeout=60.0  # Longer timeout for local inference
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "text": result["response"],
                    "provider": "ollama",
                    "model": model,
                    "finish_reason": "stop",  # Ollama doesn't provide this
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"Ollama API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate embeddings using Ollama"""
        payload = {
            "model": model,
            "prompt": text
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/embeddings",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return result["embedding"]
            except Exception as e:
                logger.error(f"Ollama embedding API error: {str(e)}")
                raise

class HuggingFaceService(BaseLLMService):
    """Service for Hugging Face Inference API calls"""
    
    def __init__(self, api_key: str, api_endpoint: str = "https://api-inference.huggingface.co"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using Hugging Face Inference API"""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 1024)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Hugging Face API format
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            if key not in ['temperature', 'max_tokens']:
                payload["parameters"][key] = value
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/models/{model}",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        text = result[0]['generated_text']
                    elif 'text' in result[0]:
                        text = result[0]['text']
                    else:
                        text = str(result[0])
                else:
                    text = str(result)
                
                return {
                    "text": text,
                    "provider": "huggingface",
                    "model": model,
                    "finish_reason": "stop",  # HF doesn't provide this consistently
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"Hugging Face API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs) -> List[float]:
        """Generate embeddings using Hugging Face"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/models/{model}",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Hugging Face embeddings return a list of floats directly
                if isinstance(result, list) and len(result) > 0:
                    return result
                else:
                    raise ValueError("Invalid embedding response format")
                    
            except Exception as e:
                logger.error(f"Hugging Face embedding API error: {str(e)}")
                raise

class OpenRouterService(BaseLLMService):
    """Service for OpenRouter API calls"""
    
    def __init__(self, api_key: str, api_endpoint: str = "https://openrouter.ai/api/v1", site_url: str = "https://fiberwise.ai", app_name: str = "FiberWise"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.site_url = site_url
        self.app_name = app_name
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using OpenRouter"""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,  # Required by OpenRouter
            "X-Title": self.app_name         # Required by OpenRouter
        }
        
        # OpenRouter uses OpenAI-compatible format
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            if key not in ['temperature', 'max_tokens']:
                payload[key] = value
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "text": result["choices"][0]["message"]["content"],
                    "provider": "openrouter",
                    "model": model,
                    "finish_reason": result["choices"][0].get("finish_reason", ""),
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"OpenRouter API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002", **kwargs) -> List[float]:
        """Generate embeddings using OpenRouter (if the model supports it)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name
        }
        
        payload = {
            "model": model,
            "input": text
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                return result["data"][0]["embedding"]
            except Exception as e:
                logger.error(f"OpenRouter embedding API error: {str(e)}")
                raise

class CloudflareWorkersAIService(BaseLLMService):
    """Service for Cloudflare Workers AI API calls"""
    
    def __init__(self, api_key: str, account_id: str, api_endpoint: str = "https://api.cloudflare.com/client/v4"):
        self.api_key = api_key
        self.account_id = account_id
        self.api_endpoint = api_endpoint
        
    async def generate_completion(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion using Cloudflare Workers AI"""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Cloudflare Workers AI format
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/accounts/{self.account_id}/ai/run/{model}",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract text from Cloudflare response
                text = ""
                if result.get("success") and "result" in result:
                    if "response" in result["result"]:
                        text = result["result"]["response"]
                    elif isinstance(result["result"], str):
                        text = result["result"]
                
                return {
                    "text": text,
                    "provider": "cloudflare",
                    "model": model,
                    "finish_reason": "stop",  # Cloudflare doesn't provide this
                    "raw_response": result
                }
            except Exception as e:
                logger.error(f"Cloudflare Workers AI API error: {str(e)}")
                raise
    
    async def generate_embedding(self, text: str, model: str = "@cf/baai/bge-base-en-v1.5", **kwargs) -> List[float]:
        """Generate embeddings using Cloudflare Workers AI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_endpoint}/accounts/{self.account_id}/ai/run/{model}",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract embedding from Cloudflare response
                if result.get("success") and "result" in result:
                    if "data" in result["result"] and len(result["result"]["data"]) > 0:
                        return result["result"]["data"][0]
                    elif "embedding" in result["result"]:
                        return result["result"]["embedding"]
                    elif isinstance(result["result"], list):
                        return result["result"]
                
                raise ValueError("Invalid embedding response format from Cloudflare")
                
            except Exception as e:
                logger.error(f"Cloudflare Workers AI embedding API error: {str(e)}")
                raise

class LLMServiceFactory:
    """Factory for creating LLM services based on provider type"""
    
    @staticmethod
    def create_service(provider_type: str, api_key: str = None, api_endpoint: str = None, **kwargs) -> BaseLLMService:
        """Create an LLM service based on provider type"""
        if provider_type == "openai":
            return OpenAIService(api_key, api_endpoint or "https://api.openai.com/v1")
        elif provider_type == "anthropic":
            return AnthropicService(api_key, api_endpoint or "https://api.anthropic.com/v1")
        elif provider_type in ["google", "gemini"]:
            return GoogleAIService(api_key, api_endpoint or "https://generativelanguage.googleapis.com/v1")
        elif provider_type == "ollama":
            return OllamaService(api_endpoint or "http://localhost:11434/api")
        elif provider_type == "huggingface":
            return HuggingFaceService(api_key, api_endpoint or "https://api-inference.huggingface.co")
        elif provider_type == "openrouter":
            site_url = kwargs.get('site_url', 'https://fiberwise.ai')
            app_name = kwargs.get('app_name', 'FiberWise')
            return OpenRouterService(api_key, api_endpoint or "https://openrouter.ai/api/v1", site_url, app_name)
        elif provider_type == "cloudflare":
            account_id = kwargs.get('account_id')
            if not account_id:
                raise ValueError("account_id is required for Cloudflare Workers AI provider")
            return CloudflareWorkersAIService(api_key, account_id, api_endpoint or "https://api.cloudflare.com/client/v4")
        elif provider_type == "custom-openai":
            return OpenAIService(api_key, api_endpoint)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

class ProviderConfiguration(BaseModel):
    """Configuration for a specific LLM provider"""
    api_key: Optional[str] = None
    default_model: str = Field(..., description="Default model to use for completions")
    embedding_model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, gt=0)
    
    class Config:
        extra = "allow"  # Allow additional fields for provider-specific settings

class ProviderModel(BaseModel):
    """Pydantic model for LLM provider"""
    provider_type: str = Field(..., description="Type of provider (openai, anthropic, etc.)")
    api_endpoint: str = Field(..., description="API endpoint URL")
    configuration: ProviderConfiguration
    
    @validator('provider_type')
    def validate_provider_type(cls, v):
        valid_types = [
            'openai', 'anthropic', 'google', 'gemini', 'ollama', 'huggingface', 
            'openrouter', 'cloudflare', 'custom-openai', 'fiberwise'
        ]
        if v not in valid_types and not v.startswith('custom-'):
            raise ValueError(f"Provider type must be one of {valid_types} or start with 'custom-'")
        return v
    
    @validator('api_endpoint')
    def validate_api_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("API endpoint must be a valid URL starting with http:// or https://")
        return v

class LLMProviderService(BaseLLMService):
    """
    Service for managing LLM provider connections and standardizing responses.
    
    This class implements BaseLLMService and can be used directly as an LLM service,
    while also managing multiple provider configurations internally.
    """
    
    def __init__(self, config: FiberWiseConfig, providers: Dict[str, ProviderModel], default_provider_id: Optional[str] = None):
        """
        Initialize the LLM Provider Service with specific provider configurations.
        
        Args:
            config: FiberWise configuration object
            providers: Dictionary mapping provider IDs to their complete configurations
            default_provider_id: The default provider ID to use (uses first provider if None)
        """
        self.config = config
        
        # Store providers as Pydantic models
        self._providers = {}
        for provider_id, provider_config in providers.items():
            # Convert dict to ProviderModel if needed
            if isinstance(provider_config, dict):
                self._providers[provider_id] = ProviderModel(**provider_config)
            else:
                self._providers[provider_id] = provider_config
        
        # Set default provider ID
        self._default_provider_id = default_provider_id or (next(iter(self._providers.keys())) if self._providers else None)
        
        # Store custom provider implementations
        self._custom_provider_implementations = {}
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests with proper authentication."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Add API key using prefixed format
        api_key = self.config.get('api_key')
        agent_key = self.config.get('agent_api_key')
        
        # Prioritize agent key if available, otherwise use regular API key
        if agent_key:
            # Agent API keys should already have agent_ prefix
            if agent_key.startswith("agent_"):
                headers['Authorization'] = f"Bearer {agent_key}"
            else:
                headers['Authorization'] = f"Bearer agent_{agent_key}"
        elif api_key:
            # Regular API keys need api_ prefix if not already present
            if api_key.startswith("api_"):
                headers['Authorization'] = f"Bearer {api_key}"
            else:
                headers['Authorization'] = f"Bearer api_{api_key}"
            
        return headers
    
    @property
    def provider_ids(self) -> List[str]:
        """Get the list of provider IDs (read-only)."""
        return list(self._providers.keys())  # Return a copy to prevent modification
    
    def get_provider(self, provider_id: str) -> Optional[ProviderModel]:
        """
        Get provider details by ID.
        
        Args:
            provider_id: The ID of the provider to get
            
        Returns:
            Provider model or None if not found
        """
        return self._providers.get(provider_id)
    
    @property
    def default_provider_id(self) -> Optional[str]:
        """Get the default provider ID."""
        return self._default_provider_id
    
    @default_provider_id.setter
    def default_provider_id(self, provider_id: str) -> None:
        """Set the default provider ID if it exists."""
        if provider_id in self._providers:
            self._default_provider_id = provider_id
        else:
            raise ValueError(f"Provider ID '{provider_id}' does not exist")
    
    async def execute_llm_request(
        self,
        prompt: str, 
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an LLM request with the specified provider using direct provider implementations.
        
        Args:
            prompt: The text prompt to send to the model
            provider_id: The ID of the provider to use
            model_id: Optional model ID (defaults to provider's default model)
            temperature: Optional temperature parameter (0.0 to 1.0)
            max_tokens: Optional maximum tokens to generate
            output_schema: Optional JSON schema to structure the response
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the model's response
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return {
                "status": "failed",
                "error": f"Provider {provider_id} not found",
                "text": ""
            }
        
        try:
            # Get provider configuration
            provider_type = provider.provider_type
            api_endpoint = provider.api_endpoint
            config = provider.configuration
            
            # Get credentials and settings
            api_key = config.api_key
            default_model = config.default_model
            default_temperature = config.temperature
            default_max_tokens = config.max_tokens
            
            # Use provided values or fall back to defaults
            model = model_id or default_model
            temp = temperature if temperature is not None else default_temperature
            tokens = max_tokens if max_tokens is not None else default_max_tokens
            
            if not model:
                return {
                    "status": "failed",
                    "error": "No model specified and no default model configured for provider",
                    "text": ""
                }
            
            # Check for custom implementation first
            if provider_type in self._custom_provider_implementations:
                # Use custom implementation
                custom_class = self._custom_provider_implementations[provider_type]
                llm_service = custom_class(api_key, api_endpoint)
            else:
                # Fall back to built-in implementation
                llm_service = LLMServiceFactory.create_service(
                    provider_type=provider_type,
                    api_key=api_key,
                    api_endpoint=api_endpoint
                )
            
            # Make the direct API call using the service implementation
            raw_response = await llm_service.generate_completion(
                prompt=prompt,
                model=model,
                temperature=temp,
                max_tokens=tokens,
                **kwargs
            )
            
            # Process and standardize the response
            standardized_response = standardize_response_impl(
                raw_response, 
                provider_type, 
                model,
                output_schema
            )
            
            return {
                "status": "completed",
                "text": standardized_response.get("text", ""),
                "model": model,
                "provider": provider_type,
                "finish_reason": standardized_response.get("finish_reason", ""),
                # Include raw_response for debugging/advanced use
                "_raw_response": raw_response
            }
            
        except Exception as e:
            logger.error(f"Error executing LLM request with provider {provider_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "text": ""
            }
    
    async def generate_completion(
        self,
        prompt: str,
        provider_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a text completion using the specified or default provider.
        
        Args:
            prompt: The text prompt to send to the model
            provider_id: The ID of the provider to use (uses first available if None)
            **kwargs: Additional parameters to pass to execute_llm_request
            
        Returns:
            Dictionary containing the model's response
        """
        # If no provider specified, use the first available
        if provider_id is None:
            if not self._providers:
                return {
                    "status": "failed",
                    "error": "No providers initialized",
                    "text": ""
                }
            provider_id = next(iter(self._providers.keys()))
        
        return await self.execute_llm_request(
            prompt=prompt,
            provider_id=provider_id,
            **kwargs
        )
    
    async def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        provider_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a structured output from the LLM using a JSON schema.
        
        Args:
            prompt: The text prompt to send to the model
            schema: JSON schema definition for the expected output
            provider_id: The ID of the provider to use (uses first available if None)
            **kwargs: Additional parameters to pass to execute_llm_request
            
        Returns:
            Dictionary containing the structured data if successful
        """
        # If no provider specified, use the first available
        if provider_id is None:
            if not self._providers:
                return {
                    "status": "failed",
                    "error": "No providers initialized",
                    "text": ""
                }
            provider_id = next(iter(self._providers.keys()))
        
        result = await self.execute_llm_request(
            prompt=prompt,
            provider_id=provider_id,
            output_schema=schema,
            **kwargs
        )
        
        # Return the structured data if available
        if result.get("status") == "completed" and "structured_data" in result:
            return {
                "status": "completed",
                "data": result["structured_data"],
                "text": result.get("text", "")
            }
        
        return result
    
    async def generate_embedding(
        self,
        text: str,
        provider_id: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings for the specified text.
        
        Args:
            text: The text to generate embeddings for
            provider_id: The ID of the provider to use
            model_id: Optional model ID (defaults to provider's default model)
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the embedding vector and metadata
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return {
                "status": "failed",
                "error": f"Provider {provider_id} not found",
                "embedding": []
            }
        
        try:
            # Get provider configuration
            provider_type = provider.provider_type
            api_endpoint = provider.api_endpoint
            config = provider.configuration
            
            # Get credentials and settings
            api_key = config.api_key
            default_embedding_model = config.embedding_model or config.default_model
            
            # Use provided model or fall back to default
            model = model_id or default_embedding_model
            
            if not model:
                return {
                    "status": "failed",
                    "error": "No model specified and no default embedding model configured",
                    "embedding": []
                }
            
            # Create LLM service with provider-specific implementation
            llm_service = LLMServiceFactory.create_service(
                provider_type=provider_type,
                api_key=api_key,
                api_endpoint=api_endpoint
            )
            
            # Generate the embedding
            embedding = await llm_service.generate_embedding(
                text=text,
                model=model,
                **kwargs
            )
            
            return {
                "status": "completed",
                "embedding": embedding,
                "model": model,
                "provider": provider_type,
                "dimension": len(embedding)
            }
            
        except NotImplementedError:
            return {
                "status": "failed",
                "error": f"Provider {provider_id} does not support embeddings",
                "embedding": []
            }
        except Exception as e:
            logger.error(f"Error generating embedding with provider {provider_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "embedding": []
            }
    
    def register_provider(self, provider_id: str, provider_config: Union[Dict[str, Any], ProviderModel]) -> bool:
        """
        Register a new LLM provider with the service.
        
        This allows agents to bring their own provider configurations at runtime
        without needing to recreate the entire service.
        
        Args:
            provider_id: Unique identifier for the provider
            provider_config: Provider configuration as dict or ProviderModel
                
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Don't allow overwriting existing providers for security
            if provider_id in self._providers:
                logger.warning(f"Provider {provider_id} already exists, not overwriting")
                return False
            
            # Convert dict to ProviderModel for validation
            if isinstance(provider_config, dict):
                provider_model = ProviderModel(**provider_config)
            else:
                provider_model = provider_config
                
            # Store the provider model
            self._providers[provider_id] = provider_model
            logger.info(f"Registered provider {provider_id} of type {provider_model.provider_type}")
            
            # If this is the first provider, set it as default
            if len(self._providers) == 1:
                self._default_provider_id = provider_id
                
            return True
        except Exception as e:
            logger.error(f"Error registering provider {provider_id}: {str(e)}")
            return False
    
    def register_custom_provider_implementation(self, 
                                              provider_type: str, 
                                              implementation_class: type) -> bool:
        """
        Register a custom provider implementation class.
        
        This allows extending the service with new provider types beyond the built-in ones.
        The implementation class must extend BaseLLMService.
        
        Args:
            provider_type: The provider type identifier (e.g., "custom-llama")
            implementation_class: Class that extends BaseLLMService
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate that the class is a subclass of BaseLLMService
            if not issubclass(implementation_class, BaseLLMService):
                logger.error(f"Implementation class must extend BaseLLMService")
                return False
                
            # Store the implementation class
            self._custom_provider_implementations[provider_type] = implementation_class
            logger.info(f"Registered custom provider implementation for {provider_type}")
            return True
        except Exception as e:
            logger.error(f"Error registering custom provider implementation: {str(e)}")
            return False
    
    # Implement required BaseLLMService abstract methods
    
    async def generate_completion(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a completion using the default provider.
        Implements the BaseLLMService interface.
        
        Args:
            prompt: The text prompt to send to the model
            model: Optional model ID (defaults to provider's default model)
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the model's response
        """
        if not self._default_provider_id:
            raise ValueError("No default provider set and no providers available")
        
        provider_id = self._default_provider_id
        provider = self._providers.get(provider_id)
        
        # Get provider configuration for default model if none specified
        if model is None:
            config = provider.configuration
            model = config.default_model
        
        # Use the execute_llm_request method internally
        # Remove any duplicate parameters from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['prompt', 'provider_id', 'model', 'model_id']}
        
        result = await self.execute_llm_request(
            prompt=prompt,
            provider_id=provider_id,
            model_id=model,
            **filtered_kwargs
        )
        
        # If the request fails, raise an exception to be consistent with BaseLLMService
        if result.get("status") != "completed":
            raise Exception(result.get("error", "Unknown error"))
        
        return result
    
    async def generate_embedding(self, text: str, model: str = None, **kwargs) -> List[float]:
        """
        Generate embeddings using the default provider.
        Implements the BaseLLMService interface.
        
        Args:
            text: The text to generate embeddings for
            model: Optional model ID
            **kwargs: Additional parameters
            
        Returns:
            List of embedding values
        """
        if not self._default_provider_id:
            raise ValueError("No default provider set and no providers available")
        
        provider_id = self._default_provider_id
        provider = self._providers.get(provider_id)
        
        # Get provider configuration for default model if none specified
        if model is None:
            config = provider.configuration
            model = config.embedding_model or config.default_model
        
        # Get the embedding result
        result = await self.generate_embedding_result(
            text=text,
            provider_id=provider_id,
            model_id=model,
            **kwargs
        )
        
        # If the request fails, raise an exception to be consistent with BaseLLMService
        if result.get("status") != "completed":
            raise Exception(result.get("error", "Unknown error"))
        
        return result["embedding"]
    
    # Rename the existing generate_embedding method to avoid conflict with the abstract method
    async def generate_embedding_result(
        self,
        text: str,
        provider_id: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings result for the specified text.
        
        Args:
            text: The text to generate embeddings for
            provider_id: The ID of the provider to use
            model_id: Optional model ID (defaults to provider's default model)
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            Dictionary containing the embedding vector and metadata
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return {
                "status": "failed",
                "error": f"Provider {provider_id} not found",
                "embedding": []
            }
        
        try:
            # Get provider configuration
            provider_type = provider.provider_type
            api_endpoint = provider.api_endpoint
            config = provider.configuration
            
            # Get credentials and settings
            api_key = config.api_key
            default_embedding_model = config.embedding_model or config.default_model
            
            # Use provided model or fall back to default
            model = model_id or default_embedding_model
            
            if not model:
                return {
                    "status": "failed",
                    "error": "No model specified and no default embedding model configured",
                    "embedding": []
                }
            
            # Create LLM service with provider-specific implementation
            llm_service = LLMServiceFactory.create_service(
                provider_type=provider_type,
                api_key=api_key,
                api_endpoint=api_endpoint
            )
            
            # Generate the embedding
            embedding = await llm_service.generate_embedding(
                text=text,
                model=model,
                **kwargs
            )
            
            return {
                "status": "completed",
                "embedding": embedding,
                "model": model,
                "provider": provider_type,
                "dimension": len(embedding)
            }
            
        except NotImplementedError:
            return {
                "status": "failed",
                "error": f"Provider {provider_id} does not support embeddings",
                "embedding": []
            }
        except Exception as e:
            logger.error(f"Error generating embedding with provider {provider_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "embedding": []
            }

# Factory function to create LLMProviderService instances
def create_llm_provider_service(
    providers: Dict[str, Union[Dict[str, Any], ProviderModel]] = None,
    config_options: Dict[str, Any] = None,
    default_provider_id: Optional[str] = None
) -> LLMProviderService:
    """
    Create a new LLMProviderService instance with the specified providers.
    
    Args:
        providers: Dictionary mapping provider IDs to their complete configurations.
                  Each configuration should be a dict or ProviderModel.
        config_options: Configuration options including base_url, api_key, etc.
        default_provider_id: The default provider ID to use (uses first provider if None)
        
    Returns:
        A new LLMProviderService instance with the provided configurations
    """
    # Default to empty dicts if None provided
    providers = providers or {}
    config_options = config_options or {}
    
    # Create config with the provided options
    config = FiberWiseConfig(initial_values=config_options)
    
    # Make certain keys read-only to prevent modification
    read_only_keys = ['base_url', 'api_key', 'agent_api_key', 'app_id']
    config.read_only_keys = read_only_keys
    
    # Create and return the service instance with provided providers
    return LLMProviderService(config, providers, default_provider_id)
