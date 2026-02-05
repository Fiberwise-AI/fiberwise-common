"""
Unit tests for LLMServiceFactory covering Phase 1 integration.

Tests provider creation from config, each provider type (OpenAI, Anthropic, Google),
user-specific API keys, and default provider selection.
"""
import pytest
import pytest_asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from fiberwise_common.services.llm_service_factory import (
    LLMServiceFactory,
    BaseLLMService,
    OpenAIService,
    AnthropicService,
    GoogleAIService,
    OllamaService,
    HuggingFaceService,
    OpenRouterService,
    CloudflareWorkersAIService
)


# ============================================================================
# Test LLMServiceFactory Creation
# ============================================================================

class TestLLMServiceFactoryCreation:
    """Test factory creates correct provider instances."""

    def test_create_openai_service(self):
        """Test creating OpenAI service."""
        service = LLMServiceFactory.create_service(
            provider_type='openai',
            api_key='sk-test-key'
        )

        assert isinstance(service, OpenAIService)
        assert service.api_key == 'sk-test-key'
        assert service.api_endpoint == 'https://api.openai.com/v1'

    def test_create_openai_service_custom_endpoint(self):
        """Test creating OpenAI service with custom endpoint."""
        service = LLMServiceFactory.create_service(
            provider_type='openai',
            api_key='sk-test-key',
            api_endpoint='https://custom.openai.com/v1'
        )

        assert isinstance(service, OpenAIService)
        assert service.api_endpoint == 'https://custom.openai.com/v1'

    def test_create_anthropic_service(self):
        """Test creating Anthropic service."""
        service = LLMServiceFactory.create_service(
            provider_type='anthropic',
            api_key='sk-ant-test'
        )

        assert isinstance(service, AnthropicService)
        assert service.api_key == 'sk-ant-test'
        assert service.api_endpoint == 'https://api.anthropic.com/v1'

    def test_create_google_service(self):
        """Test creating Google AI service."""
        service = LLMServiceFactory.create_service(
            provider_type='google',
            api_key='google-test-key'
        )

        assert isinstance(service, GoogleAIService)
        assert service.api_key == 'google-test-key'
        assert service.api_endpoint == 'https://generativelanguage.googleapis.com/v1'

    def test_create_gemini_service(self):
        """Test creating Gemini service (alias for Google)."""
        service = LLMServiceFactory.create_service(
            provider_type='gemini',
            api_key='gemini-test-key'
        )

        assert isinstance(service, GoogleAIService)
        assert service.api_key == 'gemini-test-key'

    def test_create_ollama_service(self):
        """Test creating Ollama service."""
        service = LLMServiceFactory.create_service(
            provider_type='ollama'
        )

        assert isinstance(service, OllamaService)
        assert service.api_endpoint == 'http://localhost:11434/api'

    def test_create_ollama_service_custom_endpoint(self):
        """Test creating Ollama service with custom endpoint."""
        service = LLMServiceFactory.create_service(
            provider_type='ollama',
            api_endpoint='http://custom-ollama:11434/api'
        )

        assert isinstance(service, OllamaService)
        assert service.api_endpoint == 'http://custom-ollama:11434/api'

    def test_create_huggingface_service(self):
        """Test creating Hugging Face service."""
        service = LLMServiceFactory.create_service(
            provider_type='huggingface',
            api_key='hf-test-key'
        )

        assert isinstance(service, HuggingFaceService)
        assert service.api_key == 'hf-test-key'
        assert service.api_endpoint == 'https://api-inference.huggingface.co'

    def test_create_openrouter_service(self):
        """Test creating OpenRouter service."""
        service = LLMServiceFactory.create_service(
            provider_type='openrouter',
            api_key='sk-or-test',
            site_url='https://myapp.com',
            app_name='MyApp'
        )

        assert isinstance(service, OpenRouterService)
        assert service.api_key == 'sk-or-test'
        assert service.site_url == 'https://myapp.com'
        assert service.app_name == 'MyApp'

    def test_create_openrouter_service_defaults(self):
        """Test OpenRouter service uses defaults for site/app."""
        service = LLMServiceFactory.create_service(
            provider_type='openrouter',
            api_key='sk-or-test'
        )

        assert isinstance(service, OpenRouterService)
        assert service.site_url == 'https://fiberwise.ai'
        assert service.app_name == 'FiberWise'

    def test_create_cloudflare_service(self):
        """Test creating Cloudflare Workers AI service."""
        service = LLMServiceFactory.create_service(
            provider_type='cloudflare',
            api_key='cf-test-key',
            account_id='account-123'
        )

        assert isinstance(service, CloudflareWorkersAIService)
        assert service.api_key == 'cf-test-key'
        assert service.account_id == 'account-123'

    def test_create_cloudflare_service_missing_account_id(self):
        """Test Cloudflare service requires account_id."""
        with pytest.raises(ValueError) as exc_info:
            LLMServiceFactory.create_service(
                provider_type='cloudflare',
                api_key='cf-test-key'
            )

        assert 'account_id is required' in str(exc_info.value)

    def test_create_custom_openai_service(self):
        """Test creating custom OpenAI-compatible service."""
        service = LLMServiceFactory.create_service(
            provider_type='custom-openai',
            api_key='custom-key',
            api_endpoint='https://custom-llm.com/v1'
        )

        assert isinstance(service, OpenAIService)
        assert service.api_endpoint == 'https://custom-llm.com/v1'

    def test_create_unsupported_provider(self):
        """Test creating unsupported provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            LLMServiceFactory.create_service(
                provider_type='unsupported',
                api_key='test-key'
            )

        assert 'Unsupported provider type' in str(exc_info.value)


# ============================================================================
# Test Provider Configurations
# ============================================================================

class TestProviderConfigurations:
    """Test provider configuration handling."""

    def test_provider_with_all_parameters(self):
        """Test provider creation with all parameters."""
        service = LLMServiceFactory.create_service(
            provider_type='openai',
            api_key='sk-test',
            api_endpoint='https://custom.openai.com/v1'
        )

        assert service.api_key == 'sk-test'
        assert service.api_endpoint == 'https://custom.openai.com/v1'

    def test_provider_with_minimal_parameters(self):
        """Test provider creation with minimal parameters."""
        service = LLMServiceFactory.create_service(
            provider_type='openai',
            api_key='sk-test'
        )

        assert service.api_key == 'sk-test'
        # Should use default endpoint
        assert 'openai.com' in service.api_endpoint

    def test_ollama_no_api_key_required(self):
        """Test Ollama doesn't require API key."""
        service = LLMServiceFactory.create_service(
            provider_type='ollama'
        )

        assert isinstance(service, OllamaService)
        # Ollama should work without API key


# ============================================================================
# Test OpenAI Provider
# ============================================================================

class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    @pytest.mark.asyncio
    async def test_openai_generate_completion_structure(self):
        """Test OpenAI completion has correct structure."""
        service = OpenAIService(api_key='sk-test')

        # Mock the HTTP call
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'choices': [{
                    'message': {'content': 'Test response'},
                    'finish_reason': 'stop'
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await service.generate_completion(
                prompt='test prompt',
                model='gpt-4'
            )

            assert result['text'] == 'Test response'
            assert result['provider'] == 'openai'
            assert result['model'] == 'gpt-4'
            assert result['finish_reason'] == 'stop'
            assert 'raw_response' in result

    @pytest.mark.asyncio
    async def test_openai_generate_embedding_structure(self):
        """Test OpenAI embedding has correct structure."""
        service = OpenAIService(api_key='sk-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'data': [{
                    'embedding': [0.1, 0.2, 0.3]
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await service.generate_embedding(
                text='test text',
                model='text-embedding-ada-002'
            )

            assert result == [0.1, 0.2, 0.3]


# ============================================================================
# Test Anthropic Provider
# ============================================================================

class TestAnthropicProvider:
    """Test Anthropic provider functionality."""

    @pytest.mark.asyncio
    async def test_anthropic_generate_completion_structure(self):
        """Test Anthropic completion has correct structure."""
        service = AnthropicService(api_key='sk-ant-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'content': [{'text': 'Claude response'}],
                'stop_reason': 'end_turn'
            }
            mock_response.raise_for_status = Mock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await service.generate_completion(
                prompt='test prompt',
                model='claude-3-opus-20240229'
            )

            assert result['text'] == 'Claude response'
            assert result['provider'] == 'anthropic'
            assert result['finish_reason'] == 'end_turn'

    @pytest.mark.asyncio
    async def test_anthropic_embedding_not_implemented(self):
        """Test Anthropic doesn't support embeddings."""
        service = AnthropicService(api_key='sk-ant-test')

        with pytest.raises(NotImplementedError):
            await service.generate_embedding(
                text='test text',
                model='claude-3-opus-20240229'
            )


# ============================================================================
# Test Google AI Provider
# ============================================================================

class TestGoogleAIProvider:
    """Test Google AI provider functionality."""

    @pytest.mark.asyncio
    async def test_google_generate_completion_structure(self):
        """Test Google AI completion has correct structure."""
        service = GoogleAIService(api_key='google-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'candidates': [{
                    'content': {
                        'parts': [{'text': 'Gemini response'}]
                    },
                    'finishReason': 'STOP'
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await service.generate_completion(
                prompt='test prompt',
                model='gemini-pro'
            )

            assert result['text'] == 'Gemini response'
            assert result['provider'] == 'google'
            assert result['finish_reason'] == 'STOP'


# ============================================================================
# Test Ollama Provider
# ============================================================================

class TestOllamaProvider:
    """Test Ollama provider functionality."""

    @pytest.mark.asyncio
    async def test_ollama_generate_completion_structure(self):
        """Test Ollama completion has correct structure."""
        service = OllamaService()

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'response': 'Ollama response'
            }
            mock_response.raise_for_status = Mock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await service.generate_completion(
                prompt='test prompt',
                model='llama2'
            )

            assert result['text'] == 'Ollama response'
            assert result['provider'] == 'ollama'
            assert result['finish_reason'] == 'stop'


# ============================================================================
# Test Provider Error Handling
# ============================================================================

class TestProviderErrorHandling:
    """Test error handling in providers."""

    @pytest.mark.asyncio
    async def test_openai_api_error(self):
        """Test OpenAI handles API errors."""
        service = OpenAIService(api_key='sk-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception('API Error')
            )

            with pytest.raises(Exception) as exc_info:
                await service.generate_completion(
                    prompt='test',
                    model='gpt-4'
                )

            assert 'API Error' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_anthropic_api_error(self):
        """Test Anthropic handles API errors."""
        service = AnthropicService(api_key='sk-ant-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception('API Error')
            )

            with pytest.raises(Exception) as exc_info:
                await service.generate_completion(
                    prompt='test',
                    model='claude-3-opus-20240229'
                )

            assert 'API Error' in str(exc_info.value)


# ============================================================================
# Test BaseLLMService Interface
# ============================================================================

class TestBaseLLMServiceInterface:
    """Test BaseLLMService abstract interface."""

    def test_base_service_cannot_instantiate(self):
        """Test BaseLLMService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMService()

    def test_all_providers_implement_interface(self):
        """Test all provider classes implement BaseLLMService."""
        providers = [
            OpenAIService,
            AnthropicService,
            GoogleAIService,
            OllamaService,
            HuggingFaceService,
            OpenRouterService,
            CloudflareWorkersAIService
        ]

        for provider_class in providers:
            assert issubclass(provider_class, BaseLLMService)


# ============================================================================
# Test Provider Parameters
# ============================================================================

class TestProviderParameters:
    """Test provider parameter handling."""

    @pytest.mark.asyncio
    async def test_completion_with_temperature(self):
        """Test completion respects temperature parameter."""
        service = OpenAIService(api_key='sk-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'choices': [{
                    'message': {'content': 'Response'},
                    'finish_reason': 'stop'
                }]
            }
            mock_response.raise_for_status = Mock()

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await service.generate_completion(
                prompt='test',
                model='gpt-4',
                temperature=0.5
            )

            # Verify temperature was passed
            call_args = mock_post.call_args
            assert call_args[1]['json']['temperature'] == 0.5

    @pytest.mark.asyncio
    async def test_completion_with_max_tokens(self):
        """Test completion respects max_tokens parameter."""
        service = OpenAIService(api_key='sk-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'choices': [{
                    'message': {'content': 'Response'},
                    'finish_reason': 'stop'
                }]
            }
            mock_response.raise_for_status = Mock()

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await service.generate_completion(
                prompt='test',
                model='gpt-4',
                max_tokens=1000
            )

            # Verify max_tokens was passed
            call_args = mock_post.call_args
            assert call_args[1]['json']['max_tokens'] == 1000

    @pytest.mark.asyncio
    async def test_completion_default_parameters(self):
        """Test completion uses default parameters when not specified."""
        service = OpenAIService(api_key='sk-test')

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                'choices': [{
                    'message': {'content': 'Response'},
                    'finish_reason': 'stop'
                }]
            }
            mock_response.raise_for_status = Mock()

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await service.generate_completion(
                prompt='test',
                model='gpt-4'
            )

            # Verify defaults were used
            call_args = mock_post.call_args
            assert call_args[1]['json']['temperature'] == 0.7
            assert call_args[1]['json']['max_tokens'] == 2048


# ============================================================================
# Test Factory Pattern
# ============================================================================

class TestFactoryPattern:
    """Test factory pattern implementation."""

    def test_factory_returns_correct_type(self):
        """Test factory returns correct provider type."""
        provider_map = {
            'openai': OpenAIService,
            'anthropic': AnthropicService,
            'google': GoogleAIService,
            'ollama': OllamaService,
            'huggingface': HuggingFaceService,
            'openrouter': OpenRouterService
        }

        for provider_type, expected_class in provider_map.items():
            if provider_type == 'ollama':
                service = LLMServiceFactory.create_service(provider_type=provider_type)
            else:
                service = LLMServiceFactory.create_service(
                    provider_type=provider_type,
                    api_key='test-key'
                )
            assert isinstance(service, expected_class)

    def test_factory_consistent_interface(self):
        """Test all factory-created services have consistent interface."""
        services = [
            LLMServiceFactory.create_service('openai', api_key='sk-test'),
            LLMServiceFactory.create_service('anthropic', api_key='sk-ant-test'),
            LLMServiceFactory.create_service('ollama')
        ]

        for service in services:
            assert hasattr(service, 'generate_completion')
            assert hasattr(service, 'generate_embedding')
            assert callable(service.generate_completion)
            assert callable(service.generate_embedding)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_empty_api_key(self):
        """Test handling of empty API key."""
        # Should still create service (validation happens at API call time)
        service = OpenAIService(api_key='')
        assert service.api_key == ''

    def test_none_api_key(self):
        """Test handling of None API key."""
        service = OpenAIService(api_key=None)
        assert service.api_key is None

    def test_custom_endpoint_without_https(self):
        """Test custom endpoint accepts http."""
        service = LLMServiceFactory.create_service(
            provider_type='openai',
            api_key='sk-test',
            api_endpoint='http://localhost:8080'
        )
        assert service.api_endpoint == 'http://localhost:8080'

    def test_provider_type_case_sensitivity(self):
        """Test provider type is case-sensitive."""
        # Should work with lowercase
        service = LLMServiceFactory.create_service(
            provider_type='openai',
            api_key='sk-test'
        )
        assert isinstance(service, OpenAIService)

        # Should fail with uppercase
        with pytest.raises(ValueError):
            LLMServiceFactory.create_service(
                provider_type='OPENAI',
                api_key='sk-test'
            )


# ============================================================================
# Test Provider-Specific Features
# ============================================================================

class TestProviderSpecificFeatures:
    """Test provider-specific features."""

    def test_openrouter_requires_site_info(self):
        """Test OpenRouter uses site info in requests."""
        service = LLMServiceFactory.create_service(
            provider_type='openrouter',
            api_key='sk-or-test',
            site_url='https://myapp.com',
            app_name='MyApp'
        )

        assert service.site_url == 'https://myapp.com'
        assert service.app_name == 'MyApp'

    def test_cloudflare_requires_account_id(self):
        """Test Cloudflare requires account_id."""
        with pytest.raises(ValueError) as exc_info:
            LLMServiceFactory.create_service(
                provider_type='cloudflare',
                api_key='cf-key'
            )

        assert 'account_id' in str(exc_info.value).lower()

    def test_ollama_local_by_default(self):
        """Test Ollama defaults to local endpoint."""
        service = LLMServiceFactory.create_service(
            provider_type='ollama'
        )

        assert 'localhost' in service.api_endpoint
        assert '11434' in service.api_endpoint
