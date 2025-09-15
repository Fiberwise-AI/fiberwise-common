"""
LLM Provider schemas for the FiberWise platform.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class LLMProviderConfig(BaseModel):
    """Configuration model for LLM providers."""
    provider_type: str = Field(..., description="Provider type (openai, anthropic, etc.)")
    api_key: Optional[str] = Field(None, description="API key for the provider (masked in responses)")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    model: Optional[str] = Field(None, description="Default model to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens per request")
    temperature: Optional[float] = Field(None, description="Temperature setting")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError('Temperature must be between 0 and 2')
        return v
    
    model_config = ConfigDict(extra="allow")  # Allow additional provider-specific config


class LLMProvider(BaseModel):
    """Model for LLM provider instances."""
    provider_id: str = Field(..., description="Unique provider ID")
    name: Optional[str] = Field(None, description="Provider display name")
    provider_type: Optional[str] = Field(None, description="Provider type (openai, anthropic, etc.)")
    api_endpoint: Optional[str] = Field(None, description="API endpoint URL")
    is_active: bool = Field(True, description="Whether provider is active")
    is_system: bool = Field(False, description="Whether provider is system-managed")
    configuration: Optional[LLMProviderConfig] = Field(None, description="Provider configuration")
    api_key_masked: Optional[str] = Field(None, description="Masked API key for display")