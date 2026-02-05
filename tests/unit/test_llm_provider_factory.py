"""
Unit tests for LLM Provider Factory.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from fiberwise_common.llm import LLMProviderFactory


@pytest.mark.asyncio
async def test_create_from_db_basic():
    """Test creating provider from DB with basic config."""
    # Mock database
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value={
        'provider_id': 'openai-default',
        'provider_type': 'openai',
        'configuration': json.dumps({
            'default_model': 'gpt-4',
            'api_key': 'sk-test-key',
            'base_url': None
        }),
        'is_active': True,
        'is_system': True
    })

    # Create provider
    service = await LLMProviderFactory.create_from_db(
        db=db,
        provider_id='openai-default'
    )

    # Verify
    assert service is not None
    assert len(service._providers) == 1
    assert 'openai-default' in service._providers
    provider_config = service._providers['openai-default']
    assert provider_config['model'] == 'gpt-4'
    assert provider_config['api_key'] == 'sk-test-key'


@pytest.mark.asyncio
async def test_create_from_db_with_user_scoping():
    """Test creating provider with user scoping."""
    # Mock database
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value={
        'provider_id': 'user-anthropic',
        'provider_type': 'anthropic',
        'configuration': json.dumps({
            'default_model': 'claude-sonnet-4-5-20250929',
            'api_key': 'sk-ant-test',
        }),
        'is_active': True,
        'created_by': 123
    })

    # Create provider with user_id
    service = await LLMProviderFactory.create_from_db(
        db=db,
        provider_id='user-anthropic',
        user_id=123
    )

    # Verify query was called with user scoping
    db.fetch_one.assert_called_once()
    call_args = db.fetch_one.call_args
    assert 'uid' in call_args[0][1]
    assert call_args[0][1]['uid'] == 123


@pytest.mark.asyncio
async def test_create_from_db_provider_not_found():
    """Test error handling when provider not found."""
    # Mock database returning None
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value=None)

    # Attempt to create provider
    with pytest.raises(ValueError, match="not found or not accessible"):
        await LLMProviderFactory.create_from_db(
            db=db,
            provider_id='nonexistent'
        )


@pytest.mark.asyncio
async def test_create_from_db_no_model_configured():
    """Test error handling when no model is configured."""
    # Mock database with no model
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value={
        'provider_id': 'broken-provider',
        'provider_type': 'openai',
        'configuration': json.dumps({
            'api_key': 'sk-test-key',
            # No model specified
        }),
        'is_active': True
    })

    # Attempt to create provider
    with pytest.raises(ValueError, match="no default_model configured"):
        await LLMProviderFactory.create_from_db(
            db=db,
            provider_id='broken-provider'
        )


@pytest.mark.asyncio
async def test_create_default_system():
    """Test creating default system provider."""
    # Mock database
    db = AsyncMock()
    db.fetch_one = AsyncMock(side_effect=[
        # First call: find default provider
        {'provider_id': 'system-default'},
        # Second call: get provider details
        {
            'provider_id': 'system-default',
            'provider_type': 'openai',
            'configuration': json.dumps({
                'default_model': 'gpt-4',
                'api_key': 'sk-system-key',
            }),
            'is_active': True,
            'is_system': True
        }
    ])

    # Create default provider
    service = await LLMProviderFactory.create_default(db=db)

    # Verify
    assert service is not None
    assert 'system-default' in service._providers


@pytest.mark.asyncio
async def test_create_default_user():
    """Test creating default user-scoped provider."""
    # Mock database
    db = AsyncMock()
    db.fetch_one = AsyncMock(side_effect=[
        # First call: find user's default provider
        {'provider_id': 'user-default'},
        # Second call: get provider details
        {
            'provider_id': 'user-default',
            'provider_type': 'anthropic',
            'configuration': json.dumps({
                'default_model': 'claude-3-5-sonnet-20241022',
                'api_key': 'sk-user-key',
            }),
            'is_active': True,
            'created_by': 456
        }
    ])

    # Create default provider for user
    service = await LLMProviderFactory.create_default(db=db, user_id=456)

    # Verify
    assert service is not None
    assert 'user-default' in service._providers


@pytest.mark.asyncio
async def test_create_default_not_found():
    """Test error when no default provider exists."""
    # Mock database returning None
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value=None)

    # Attempt to create default provider
    with pytest.raises(ValueError, match="No default LLM provider found"):
        await LLMProviderFactory.create_default(db=db)


@pytest.mark.asyncio
async def test_parse_configuration_json_string():
    """Test parsing configuration when it's a JSON string."""
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value={
        'provider_id': 'test-provider',
        'provider_type': 'openai',
        'configuration': '{"default_model": "gpt-3.5-turbo", "api_key": "sk-test"}',
        'is_active': True
    })

    service = await LLMProviderFactory.create_from_db(db, 'test-provider')

    assert service is not None
    assert service._providers['test-provider']['model'] == 'gpt-3.5-turbo'


@pytest.mark.asyncio
async def test_parse_configuration_dict():
    """Test parsing configuration when it's already a dict."""
    db = AsyncMock()
    db.fetch_one = AsyncMock(return_value={
        'provider_id': 'test-provider',
        'provider_type': 'openai',
        'configuration': {
            'default_model': 'gpt-3.5-turbo',
            'api_key': 'sk-test'
        },
        'is_active': True
    })

    service = await LLMProviderFactory.create_from_db(db, 'test-provider')

    assert service is not None
    assert service._providers['test-provider']['model'] == 'gpt-3.5-turbo'
