"""
Unit tests for fiberwise_common.utils.llm_response_utils module.

Tests LLM response standardization across different providers.
"""
import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

from fiberwise_common.utils.llm_response_utils import (
    standardize_response,
    _extract_text_and_finish_reason
)


class TestExtractTextAndFinishReason:
    """Test _extract_text_and_finish_reason helper function."""

    def test_extract_openai_response(self):
        """Test extracting text from OpenAI response format."""
        openai_response = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response from OpenAI."
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        text, finish_reason = _extract_text_and_finish_reason(openai_response, "openai")
        
        assert text == "This is a test response from OpenAI."
        assert finish_reason == "stop"

    def test_extract_openrouter_response(self):
        """Test extracting text from OpenRouter response (same format as OpenAI)."""
        openrouter_response = {
            "choices": [
                {
                    "message": {
                        "content": "OpenRouter response text."
                    },
                    "finish_reason": "length"
                }
            ]
        }
        
        text, finish_reason = _extract_text_and_finish_reason(openrouter_response, "openrouter")
        
        assert text == "OpenRouter response text."
        assert finish_reason == "length"

    def test_extract_anthropic_response(self):
        """Test extracting text from Anthropic response format."""
        anthropic_response = {
            "content": [
                {
                    "type": "text",
                    "text": "This is Claude's response."
                }
            ],
            "stop_reason": "end_turn"
        }
        
        text, finish_reason = _extract_text_and_finish_reason(anthropic_response, "anthropic")
        
        assert text == "This is Claude's response."
        assert finish_reason == "end_turn"

    def test_extract_anthropic_multiple_content_blocks(self):
        """Test extracting text from Anthropic response with multiple content blocks."""
        anthropic_response = {
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
                {"type": "image", "source": "image_data"}  # Should be ignored
            ],
            "stop_reason": "end_turn"
        }
        
        text, finish_reason = _extract_text_and_finish_reason(anthropic_response, "anthropic")
        
        assert text == "First part. Second part."
        assert finish_reason == "end_turn"

    def test_extract_google_response_candidates_format(self):
        """Test extracting text from Google response with candidates format."""
        google_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Google AI response text."}
                        ]
                    },
                    "finishReason": "STOP"
                }
            ]
        }
        
        text, finish_reason = _extract_text_and_finish_reason(google_response, "google")
        
        assert text == "Google AI response text."
        assert finish_reason == "STOP"

    def test_extract_google_response_direct_text_format(self):
        """Test extracting text from Google response with direct text format."""
        google_response = {
            "text": "Direct Google response."
        }
        
        text, finish_reason = _extract_text_and_finish_reason(google_response, "google")
        
        assert text == "Direct Google response."
        assert finish_reason == ""

    def test_extract_ollama_response(self):
        """Test extracting text from Ollama response format."""
        ollama_response = {
            "response": "This is an Ollama response.",
            "done": True
        }
        
        text, finish_reason = _extract_text_and_finish_reason(ollama_response, "ollama")
        
        assert text == "This is an Ollama response."
        assert finish_reason == "STOP"

    def test_extract_ollama_response_not_done(self):
        """Test extracting text from Ollama response that's not done."""
        ollama_response = {
            "response": "Partial response.",
            "done": False
        }
        
        text, finish_reason = _extract_text_and_finish_reason(ollama_response, "ollama")
        
        assert text == "Partial response."
        assert finish_reason == ""

    def test_extract_huggingface_response_generated_text(self):
        """Test extracting text from HuggingFace response with generated_text."""
        hf_response = [
            {
                "generated_text": "HuggingFace generated response."
            }
        ]
        
        text, finish_reason = _extract_text_and_finish_reason(hf_response, "huggingface")
        
        assert text == "HuggingFace generated response."
        assert finish_reason == "stop"

    def test_extract_huggingface_response_text_field(self):
        """Test extracting text from HuggingFace response with text field."""
        hf_response = [
            {
                "text": "HuggingFace text response."
            }
        ]
        
        text, finish_reason = _extract_text_and_finish_reason(hf_response, "huggingface")
        
        assert text == "HuggingFace text response."
        assert finish_reason == "stop"

    def test_extract_huggingface_response_dict_format(self):
        """Test extracting text from HuggingFace response in dict format."""
        hf_response = {
            "generated_text": "HuggingFace dict response."
        }
        
        text, finish_reason = _extract_text_and_finish_reason(hf_response, "huggingface")
        
        assert text == "HuggingFace dict response."
        assert finish_reason == "stop"

    def test_extract_cloudflare_response_success(self):
        """Test extracting text from Cloudflare response with success."""
        cf_response = {
            "success": True,
            "result": {
                "response": "Cloudflare AI response."
            }
        }
        
        text, finish_reason = _extract_text_and_finish_reason(cf_response, "cloudflare")
        
        assert text == "Cloudflare AI response."
        assert finish_reason == "stop"

    def test_extract_cloudflare_response_string_result(self):
        """Test extracting text from Cloudflare response with string result."""
        cf_response = {
            "success": True,
            "result": "Direct string response."
        }
        
        text, finish_reason = _extract_text_and_finish_reason(cf_response, "cloudflare")
        
        assert text == "Direct string response."
        assert finish_reason == "stop"

    def test_extract_generic_provider_response(self):
        """Test extracting text from generic provider response."""
        generic_response = {
            "text": "Generic provider response.",
            "finish_reason": "completed"
        }
        
        text, finish_reason = _extract_text_and_finish_reason(generic_response, "unknown_provider")
        
        assert text == "Generic provider response."
        assert finish_reason == "completed"

    def test_extract_empty_response(self):
        """Test extracting text from empty or invalid response."""
        empty_response = {}
        
        text, finish_reason = _extract_text_and_finish_reason(empty_response, "openai")
        
        assert text == ""
        assert finish_reason == ""

    def test_extract_malformed_response(self):
        """Test extracting text from malformed response."""
        malformed_response = "This is not a dict"
        
        text, finish_reason = _extract_text_and_finish_reason(malformed_response, "openai")
        
        assert text == ""
        assert finish_reason == ""

    @pytest.mark.parametrize("provider,response,expected_text,expected_reason", [
        ("openai", {"choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]}, "test", "stop"),
        ("anthropic", {"content": [{"type": "text", "text": "test"}], "stop_reason": "end"}, "test", "end"),
        ("google", {"candidates": [{"content": {"parts": [{"text": "test"}]}, "finishReason": "STOP"}]}, "test", "STOP"),
        ("ollama", {"response": "test", "done": True}, "test", "STOP"),
        ("huggingface", [{"generated_text": "test"}], "test", "stop"),
        ("cloudflare", {"success": True, "result": {"response": "test"}}, "test", "stop"),
    ])
    def test_extract_parametrized(self, provider: str, response: Dict[str, Any], expected_text: str, expected_reason: str):
        """Test text extraction with various provider formats."""
        text, finish_reason = _extract_text_and_finish_reason(response, provider)
        assert text == expected_text
        assert finish_reason == expected_reason


class TestStandardizeResponse:
    """Test standardize_response main function."""

    def test_standardize_openai_response(self):
        """Test standardizing OpenAI response."""
        raw_response = {
            "choices": [
                {
                    "message": {"content": "OpenAI response"},
                    "finish_reason": "stop"
                }
            ]
        }
        
        result = standardize_response(raw_response, "openai", "gpt-3.5-turbo")
        
        assert result["text"] == "OpenAI response"
        assert result["model"] == "gpt-3.5-turbo"
        assert result["provider"] == "openai"
        assert result["finish_reason"] == "stop"
        assert result["raw_response"] == raw_response

    def test_standardize_anthropic_response(self):
        """Test standardizing Anthropic response."""
        raw_response = {
            "content": [{"type": "text", "text": "Claude response"}],
            "stop_reason": "end_turn"
        }
        
        result = standardize_response(raw_response, "anthropic", "claude-3-sonnet")
        
        assert result["text"] == "Claude response"
        assert result["model"] == "claude-3-sonnet"
        assert result["provider"] == "anthropic"
        assert result["finish_reason"] == "end_turn"

    @patch('fiberwise_common.utils.schema_utils._apply_output_schema')
    def test_standardize_response_with_schema(self, mock_apply_schema):
        """Test standardizing response with output schema."""
        mock_apply_schema.return_value = {"parsed": "data"}
        
        raw_response = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]
        }
        output_schema = {"type": "json"}
        
        result = standardize_response(raw_response, "openai", "gpt-4", output_schema)
        
        mock_apply_schema.assert_called_once_with("test", output_schema)
        assert result["structured_data"] == {"parsed": "data"}
        assert "text" in result
        assert "raw_response" in result

    def test_standardize_response_error_handling(self):
        """Test error handling in standardize_response."""
        # None input should be handled gracefully, not cause an error
        invalid_response = None
        
        result = standardize_response(invalid_response, "openai", "gpt-4")
        
        # Function handles None gracefully by returning empty text
        assert result["text"] == ""
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["finish_reason"] == ""
        assert result["raw_response"] is None

    def test_standardize_response_all_fields(self):
        """Test that standardized response contains all expected fields."""
        raw_response = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]
        }
        
        result = standardize_response(raw_response, "openai", "gpt-4")
        
        required_fields = {"text", "model", "provider", "finish_reason", "raw_response"}
        assert all(field in result for field in required_fields)

    @pytest.mark.parametrize("provider_type", [
        "openai", "openrouter", "anthropic", "google", 
        "ollama", "huggingface", "cloudflare", "unknown"
    ])
    def test_standardize_response_all_providers(self, provider_type: str):
        """Test standardizing responses from all supported providers."""
        # Use generic response format that should work with fallback
        raw_response = {"text": "test response", "finish_reason": "stop"}
        
        result = standardize_response(raw_response, provider_type, "test-model")
        
        assert "text" in result
        assert result["provider"] == provider_type
        assert result["model"] == "test-model"


class TestLLMResponseUtilsIntegration:
    """Integration tests for LLM response utilities."""

    def test_full_response_processing_pipeline(self):
        """Test complete response processing from raw to structured."""
        # Simulate various provider responses in sequence
        provider_responses = [
            ("openai", {
                "choices": [{"message": {"content": "OpenAI result"}, "finish_reason": "stop"}]
            }),
            ("anthropic", {
                "content": [{"type": "text", "text": "Anthropic result"}],
                "stop_reason": "end_turn"
            }),
            ("google", {
                "candidates": [{"content": {"parts": [{"text": "Google result"}]}, "finishReason": "STOP"}]
            })
        ]
        
        results = []
        for provider, response in provider_responses:
            result = standardize_response(response, provider, f"{provider}-model")
            results.append(result)
        
        # Verify all responses were standardized correctly
        assert len(results) == 3
        assert results[0]["text"] == "OpenAI result"
        assert results[1]["text"] == "Anthropic result"
        assert results[2]["text"] == "Google result"
        
        # Verify all have consistent structure
        for result in results:
            assert all(field in result for field in ["text", "model", "provider", "finish_reason"])

    def test_response_standardization_with_error_recovery(self):
        """Test response standardization with various error conditions."""
        error_cases = [
            ("corrupted_json", {"choices": "not a list"}),
            ("missing_fields", {"choices": [{"message": {}}]}),
            ("null_response", None),
            ("empty_response", {}),
            ("string_response", "just a string")
        ]
        
        for case_name, raw_response in error_cases:
            result = standardize_response(raw_response, "openai", "test-model")
            
            # Should always return a valid response structure
            assert isinstance(result, dict)
            assert "model" in result
            assert "provider" in result
            
            # May have either text or error field
            assert "text" in result or "error" in result

    @patch('fiberwise_common.utils.schema_utils._apply_output_schema')
    def test_schema_integration_workflow(self, mock_apply_schema):
        """Test integration of schema processing with response standardization."""
        mock_apply_schema.return_value = {
            "intent": "greeting",
            "entities": ["user"],
            "confidence": 0.95
        }
        
        raw_response = {
            "choices": [{"message": {"content": "Hello there, user!"}, "finish_reason": "stop"}]
        }
        
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "entities": {"type": "array"},
                "confidence": {"type": "number"}
            }
        }
        
        result = standardize_response(raw_response, "openai", "gpt-4", schema)
        
        # Check that both raw text and structured data are present
        assert result["text"] == "Hello there, user!"
        assert result["structured_data"]["intent"] == "greeting"
        assert result["structured_data"]["confidence"] == 0.95
        assert len(result["structured_data"]["entities"]) == 1

    def test_cross_provider_consistency(self):
        """Test that different providers return consistent standardized format."""
        # Same content, different provider formats
        test_content = "Consistent response text"
        
        responses = {
            "openai": {
                "choices": [{"message": {"content": test_content}, "finish_reason": "stop"}]
            },
            "anthropic": {
                "content": [{"type": "text", "text": test_content}],
                "stop_reason": "end_turn"
            },
            "huggingface": [{"generated_text": test_content}]
        }
        
        standardized_responses = {}
        for provider, raw_response in responses.items():
            standardized_responses[provider] = standardize_response(
                raw_response, provider, f"{provider}-model"
            )
        
        # All should have the same text content
        for provider, result in standardized_responses.items():
            assert result["text"] == test_content
            assert result["provider"] == provider
            assert "finish_reason" in result
            assert "model" in result
        
        # Structure should be consistent across providers
        field_sets = [set(result.keys()) for result in standardized_responses.values()]
        common_fields = set.intersection(*field_sets)
        expected_common_fields = {"text", "model", "provider", "finish_reason", "raw_response"}
        
        assert common_fields >= expected_common_fields