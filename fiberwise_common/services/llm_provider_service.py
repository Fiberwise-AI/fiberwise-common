"""
LLM Provider Service - migrated from fiberwise-core-web/worker
Manages LLM provider connections and standardizes responses.
Provides a unified interface for different LLM providers.
"""

import json
import logging
from typing import Dict, Any, Optional
import asyncpg
import aiohttp
import os

from .base_service import BaseService

logger = logging.getLogger(__name__)


class LLMProviderService(BaseService):
    """Service for managing LLM provider connections and standardizing responses"""
    
    def __init__(self, db_provider, user_id: Optional[Any] = None, llm_service_factory: Optional[Any] = None):
        """Initialize LLM Provider Service with database provider."""
        super().__init__(db_provider)
        self.user_id = user_id
        self.llm_service_factory = llm_service_factory
    
    async def get_provider_by_id(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provider details from the database by provider_id with user scoping
        
        Returns:
            Provider details including configuration and credentials
        """
        try:
            # Query with user scoping - show system providers and user's own providers
            if self.user_id:
                query = """
                    SELECT provider_id, name, provider_type, api_endpoint, configuration 
                    FROM llm_providers
                    WHERE provider_id = $1 AND (is_active = 1 OR is_active = true OR is_active = 'true') 
                    AND (is_system = true OR is_system = 1 OR created_by = $2)
                """
                provider = await self._fetch_one(query, (provider_id, self.user_id))
            else:
                # No user scoping - only show system providers
                query = """
                    SELECT provider_id, name, provider_type, api_endpoint, configuration 
                    FROM llm_providers
                    WHERE provider_id = $1 AND (is_active = 1 OR is_active = true OR is_active = 'true') 
                    AND (is_system = true OR is_system = 1)
                """
                provider = await self._fetch_one(query, (provider_id,))
            
            if not provider:
                logger.warning(f"Provider {provider_id} not found or not active")
                return None
            
            # Debug: Log the raw provider data
            logger.info(f"Raw provider data for {provider_id}: {dict(provider)}")
            
            # Ensure configuration is properly parsed
            if isinstance(provider.get('configuration'), str):
                try:
                    config = json.loads(provider['configuration'])
                    
                    # Handle double-serialized JSON
                    if isinstance(config, str):
                        try:
                            config = json.loads(config)
                        except json.JSONDecodeError:
                            pass
                            
                    provider['configuration'] = config
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse configuration for provider {provider_id}: {e}")
                    logger.info(f"Raw configuration value: {repr(provider.get('configuration'))}")
                    provider['configuration'] = {}
            
            return provider
            
        except Exception as e:
            logger.error(f"Error fetching provider {provider_id}: {str(e)}")
            return None
    
    async def get_default_provider(self) -> Optional[Dict[str, Any]]:
        """
        Gets the default LLM provider with user scoping.
        """
        if self.user_id:
            # User's explicit default
            query = """SELECT * FROM llm_providers 
                      WHERE (is_active = 1 OR is_active = true OR is_active = 'true') 
                      AND (is_default = 1 OR is_default = true OR is_default = 'true') 
                      AND created_by = $1"""
            provider = await self._fetch_one(query, (self.user_id,))
        else:
            # System default
            query = """SELECT * FROM llm_providers 
                      WHERE (is_active = 1 OR is_active = true OR is_active = 'true') 
                      AND (is_default = 1 OR is_default = true OR is_default = 'true') 
                      AND (is_system = 1 OR is_system = true OR is_system = 'true')"""
            provider = await self._fetch_one(query)

        if provider:
            # Ensure configuration is properly parsed
            if isinstance(provider.get('configuration'), str):
                try:
                    config = json.loads(provider['configuration'])
                    
                    # Handle double-serialized JSON
                    if isinstance(config, str):
                        try:
                            config = json.loads(config)
                        except json.JSONDecodeError:
                            pass
                            
                    provider['configuration'] = config
                except json.JSONDecodeError as e:
                    provider_id = provider.get('provider_id', 'N/A')
                    logger.warning(f"Could not parse configuration for provider {provider_id}: {e}")
                    logger.info(f"Raw configuration value: {repr(provider.get('configuration'))}")
                    provider['configuration'] = {}
            return provider
            
        return None

    async def generate_completion(
        self,
        prompt: str,
        provider_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Alternative method name for execute_llm_request for convenience
        
        Args:
            prompt: The text prompt to send to the model
            provider_id: The ID of the provider to use
            **kwargs: Additional parameters to pass to execute_llm_request
            
        Returns:
            Same as execute_llm_request
        """
        return await self.execute_llm_request(
            provider_id=provider_id,
            prompt=prompt,
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
        Generate a structured output from the LLM using a JSON schema
        
        Args:
            prompt: The text prompt to send to the model
            schema: JSON schema definition for the expected output
            provider_id: The ID of the provider to use
            **kwargs: Additional parameters to pass to execute_llm_request
            
        Returns:
            Dictionary containing the structured data if successful
        """
        result = await self.execute_llm_request(
            provider_id=provider_id,
            prompt=prompt,
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

    async def execute_llm_request(
        self,
        provider_id: Optional[str], 
        prompt: str, 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an LLM request with the specified provider
        
        Args:
            provider_id: The ID of the provider to use. If None, uses the default provider.
            prompt: The text prompt to send
            model: Optional model ID to use (overrides provider default)
            temperature: Optional temperature setting
            max_tokens: Optional max tokens setting
            output_schema: Optional schema to format the response
            
        Returns:
            Standardized response with model output
        """
        try:
            provider = None
            if not provider_id:
                default_provider = await self.get_default_provider()
                if default_provider:
                    provider_id = default_provider['provider_id']
                    provider = default_provider
                else:
                    return {
                        "status": "failed",
                        "error": "No default LLM provider found and no provider_id specified."
                    }
            else:
                provider = await self.get_provider_by_id(provider_id)

            if not provider:
                return {
                    "status": "failed",
                    "error": f"Provider {provider_id} not found or not active"
                }
            
            # Get provider configuration
            provider_type = provider['provider_type']
            api_endpoint = provider['api_endpoint']
            config = provider['configuration']
            
            # Get credentials and settings
            api_key = config.get('api_key')
            default_model = config.get('default_model')
            default_temperature = config.get('temperature', 0.7)
            default_max_tokens = config.get('max_tokens', 2048)
            
            # Debug: Log what we extracted from config
            logger.info(f"Provider {provider_id} config parsed - api_key: {'***' if api_key else None}, default_model: {default_model}")
            
            # Use provided values or fall back to defaults
            final_model = model or default_model
            temp = temperature if temperature is not None else default_temperature
            tokens = max_tokens if max_tokens is not None else default_max_tokens
            
            if not final_model:
                logger.error(f"Provider {provider_id}: No model specified (model={model}) and no default model configured (default_model={default_model})")
                return {
                    "status": "failed",
                    "error": "No model specified and no default model configured for provider"
                }
            
            
            if not self.llm_service_factory:
                raise ValueError("LLMProviderService was not created with an LLMServiceFactory.")

            # Create and use LLM service
            llm_service = self.llm_service_factory.create_service(
                provider_type=provider_type,
                api_key=api_key,
                api_endpoint=api_endpoint
            )
            
            # Make the actual API call
            raw_response = await llm_service.generate_completion(
                prompt=prompt,
                model=final_model,
                temperature=temp,
                max_tokens=tokens
            )
            
            # Process and standardize the response
            from ..utils.llm_response_utils import standardize_response as standardize_response_impl
            standardized_response = standardize_response_impl(
                raw_response, 
                provider_type, 
                final_model,
                output_schema
            )
            
            return {
                "status": "completed",
                "text": standardized_response.get("text", ""),
                "model": final_model,
                "provider": provider_type,
                "finish_reason": standardized_response.get("finish_reason", ""),
                # Include raw_response for debugging/advanced use but don't store in DB
                "_raw_response": raw_response
            }
            
        except Exception as e:
            logger.error(f"Error executing LLM request with provider {provider_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
