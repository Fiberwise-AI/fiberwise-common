"""
LLM Response Utilities

This module provides utilities for standardizing LLM responses from various providers.
Consolidates the duplicate standardize_response implementations found in 
llm_provider_service.py and llm_service_factory.py.
"""

from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _extract_text_and_finish_reason(raw_response: Any, provider_type: str) -> Tuple[str, str]:
    """
    Helper to extract text and finish reason from a raw provider response.
    This function isolates the provider-specific parsing logic.
    
    Args:
        raw_response: Raw response from the LLM provider
        provider_type: Provider type identifier
        
    Returns:
        Tuple of (text, finish_reason)
    """
    text = ""
    finish_reason = ""
    
    if provider_type in ("openai", "openrouter"):  # Consolidated duplicate case
        if isinstance(raw_response, dict) and 'choices' in raw_response:
            choices = raw_response.get('choices', [])
            if choices and len(choices) > 0:
                message = choices[0].get('message', {})
                text = message.get('content', "")
                finish_reason = choices[0].get('finish_reason', "")
    
    elif provider_type == "anthropic":
        if isinstance(raw_response, dict) and 'content' in raw_response:
            content = raw_response.get('content', [])
            if content and len(content) > 0:
                text_parts = [part.get('text', "") for part in content if part.get('type') == 'text']
                text = " ".join(text_parts)
            finish_reason = raw_response.get('stop_reason', "")
    
    elif provider_type == "google":
        if isinstance(raw_response, dict) and 'candidates' in raw_response:
            candidates = raw_response.get('candidates', [])
            if candidates and len(candidates) > 0:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts and len(parts) > 0:
                    text = parts[0].get('text', "")
                finish_reason = candidates[0].get('finishReason', "")
        else:
            text = str(raw_response.get('text', ""))
    
    elif provider_type == "ollama":
        if isinstance(raw_response, dict):
            text = raw_response.get('response', "")
            finish_reason = "STOP" if raw_response.get('done', False) else ""
    
    elif provider_type == "huggingface":
        if isinstance(raw_response, list) and len(raw_response) > 0:
            if 'generated_text' in raw_response[0]:
                text = raw_response[0]['generated_text']
            elif 'text' in raw_response[0]:
                text = raw_response[0]['text']
            else:
                text = str(raw_response[0])
        elif isinstance(raw_response, dict):
            text = raw_response.get('generated_text') or raw_response.get('text') or str(raw_response)
        else:
            text = str(raw_response)
        finish_reason = "stop"
    
    elif provider_type == "cloudflare":
        if isinstance(raw_response, dict) and raw_response.get("success"):
            result = raw_response.get("result", {})
            if isinstance(result, str):
                text = result
            elif isinstance(result, dict) and "response" in result:
                text = result["response"]
            else:
                text = str(result)
            finish_reason = "stop"
        else:
            text = str(raw_response)
    
    else:  # Generic fallback
        if isinstance(raw_response, dict):
            text = (raw_response.get('text') or 
                   raw_response.get('output') or 
                   raw_response.get('content') or 
                   raw_response.get('message') or 
                   raw_response.get('response') or 
                   "")  # Return empty string if nothing is found
            # This also handles the use case of Function 2, where 'text' is already present
            if 'finish_reason' in raw_response:
                 finish_reason = raw_response.get('finish_reason', "")
        
        if not text:  # Final fallback if all else fails
             text = str(raw_response)

    return text, finish_reason


def standardize_response(
    raw_response: Any, 
    provider_type: str, 
    model: str,
    output_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standardizes the response from any LLM provider into a consistent format.

    This single function handles both raw API responses and pre-standardized
    dictionaries by delegating parsing to an internal helper.
    
    Args:
        raw_response: The raw response from the provider or a pre-parsed dict.
        provider_type: The type of provider (e.g., 'openai', 'anthropic').
        model: The model used for the request.
        output_schema: Optional schema to format the response into structured data.
        
    Returns:
        A standardized response dictionary with keys: text, model, provider, 
        finish_reason, and optionally structured_data and raw_response.
    """
    try:
        # 1. Delegate the messy parsing to a dedicated helper function
        text, finish_reason = _extract_text_and_finish_reason(raw_response, provider_type)

        # 2. Apply output schema if provided
        structured_data = None
        if output_schema:
            from .schema_utils import _apply_output_schema
            structured_data = _apply_output_schema(text, output_schema)
        
        # 3. Assemble the final standardized response
        response = {
            "text": text,
            "model": model,
            "provider": provider_type,
            "finish_reason": finish_reason
        }
        
        if structured_data is not None:
            response["structured_data"] = structured_data
            
        # Include raw response for debugging purposes
        response["raw_response"] = raw_response
            
        return response
        
    except Exception as e:
        logger.error(f"Error standardizing response for provider '{provider_type}': {str(e)}")
        return {
            "text": str(raw_response),
            "error": str(e),
            "model": model,
            "provider": provider_type
        }