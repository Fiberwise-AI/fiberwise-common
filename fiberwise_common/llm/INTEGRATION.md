# LLM Provider Factory Integration Guide

This guide shows how to integrate the LLMProviderFactory into FiberWise's activation system and agents.

## Overview

The LLMProviderFactory is a tiny shim (~100 lines) that creates `ia_modules` LLM providers directly from FiberWise database configuration. This eliminates the need for a complex adapter layer.

## Integration Points

### 1. Activation Processor

Update `fiberwise_common/activation/activation_processor.py` to use the factory:

```python
from ..llm import LLMProviderFactory

class ActivationProcessor:

    async def _execute_llm_agent(self, agent_config, input_data, activation):
        """Execute LLM_CHAT agent using ia_modules LLM provider"""
        from ..entities.llm_chat_agent import LLMChatAgent
        from ia_modules.agents import StateManager

        # Get provider config
        provider_id = agent_config.get('provider_id')
        user_id = activation.get('created_by')

        # Create ia_modules provider directly (tiny shim)
        llm_provider = await LLMProviderFactory.create_from_db(
            self.db,
            provider_id=provider_id,
            user_id=user_id
        )

        # Create agent
        state_manager = StateManager()
        agent = LLMChatAgent(config=agent_config, state_manager=state_manager)

        # Inject ia_modules provider
        agent.inject_services(llm_service=llm_provider)

        # Execute
        result = await agent.execute(input_data)

        return result

    async def _execute_custom_agent(self, file_path, input_data, activation):
        """Execute CLASS/FUNCTION agent"""
        from ia_modules.agents import StateManager

        # Load agent...
        agent = load_agent(file_path)

        # Get dependencies
        dependencies = agent.get_dependencies()

        # Create services
        services = {}
        for dep in dependencies:
            if dep == 'llm_service':
                # Get provider from agent config or activation metadata
                provider_id = (
                    activation.get('metadata', {}).get('provider_id') or
                    agent.config.get('provider_id')
                )

                # Create ia_modules provider (tiny shim)
                services['llm_service'] = await LLMProviderFactory.create_from_db(
                    self.db,
                    provider_id=provider_id,
                    user_id=activation.get('created_by')
                )

            elif dep == 'fiber_app':
                services['fiber_app'] = self._create_fiber_app(activation)

            elif dep == 'db':
                services['db'] = self.db

        # Inject services
        agent.inject_services(**services)

        # Execute
        result = await agent.execute(input_data)

        return result
```

### 2. LLM Chat Agent

Update `fiberwise_common/entities/llm_chat_agent.py` to use ia_modules API:

```python
from fiberwise_common.entities.fiber_agent import FiberAgent

class LLMChatAgent(FiberAgent):
    """Config-only agent using ia_modules LLM provider"""

    def get_dependencies(self):
        return ['llm_service']  # ia_modules LLM provider

    async def run_agent_async(self, input_data, **kwargs):
        llm_service = kwargs.get('llm_service')  # ia_modules provider!

        # Build prompt
        system_message = self.config.get('system_message', 'You are a helpful assistant.')
        user_message = input_data.get('prompt', '')

        # Use ia_modules API directly (LiteLLM format)
        response = await llm_service.generate_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=self.config.get('temperature', 0.7),
            max_tokens=self.config.get('max_tokens', 2048)
        )

        return {
            "response": response['content'],
            "model": response['model'],
            "usage": response['usage']
        }
```

### 3. Custom Agent

Example custom agent using the factory:

```python
from fiberwise_common.entities.fiber_agent import FiberAgent

class MyCustomAgent(FiberAgent):
    """Custom agent using ia_modules LLM provider"""

    def get_dependencies(self):
        return ['llm_service', 'db']

    async def run_agent_async(self, input_data, **kwargs):
        llm_service = kwargs.get('llm_service')  # ia_modules provider!
        db = kwargs.get('db')

        # Get context from database
        session_id = kwargs.get('context', {}).get('session_id')
        history = await db.fetch_all(
            "SELECT input_data, output_data FROM agent_activations WHERE session_id = ?",
            session_id
        )

        # Build messages with history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # Add history
        for record in history:
            messages.append({"role": "user", "content": record['input_data']})
            messages.append({"role": "assistant", "content": record['output_data']})

        # Add current input
        messages.append({"role": "user", "content": input_data.get('prompt')})

        # Use ia_modules API directly
        response = await llm_service.generate_completion(
            messages=messages,
            temperature=0.7
        )

        return {
            "result": response['content'],
            "cost": response['usage']['cost_usd']
        }
```

### 4. Worker Integration

For background workers that need LLM access:

```python
from fiberwise_common.llm import LLMProviderFactory

class BackgroundWorker:

    async def process_job(self, job_data):
        # Get database connection
        db = await self.get_db()

        # Create provider
        llm_service = await LLMProviderFactory.create_from_db(
            db=db,
            provider_id=job_data.get('provider_id', 'openai-default'),
            user_id=job_data.get('user_id')
        )

        # Process with LLM
        response = await llm_service.generate_completion(
            messages=[{"role": "user", "content": job_data['prompt']}]
        )

        return response
```

### 5. API Endpoints

For REST API endpoints that need LLM access:

```python
from fastapi import APIRouter, Depends
from fiberwise_common.llm import LLMProviderFactory
from fiberwise_common.database import get_db

router = APIRouter()

@router.post("/chat/completion")
async def chat_completion(
    request: ChatRequest,
    db = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Generate chat completion using user's configured provider"""

    # Create provider from user's configuration
    llm_service = await LLMProviderFactory.create_from_db(
        db=db,
        provider_id=request.provider_id,
        user_id=current_user.id
    )

    # Generate completion
    response = await llm_service.generate_completion(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    return {
        "content": response['content'],
        "model": response['model'],
        "usage": response['usage']
    }
```

## Migration Strategy

### Phase 1: Side-by-side (Current Phase)

- New code uses LLMProviderFactory
- Old code continues using LLMProviderService (deprecated)
- Both work simultaneously

```python
# Old (deprecated, but still works)
from fiberwise_common.services.llm_provider_service import LLMProviderService

# New (recommended)
from fiberwise_common.llm import LLMProviderFactory
```

### Phase 2: Gradual Migration

1. Update activation_processor to use factory
2. Update llm_chat_agent to use ia_modules API
3. Migrate existing custom agents one by one
4. Update documentation and examples

### Phase 3: Full Migration

1. Remove old LLMProviderService
2. Update all references to use factory
3. Clean up deprecated imports

## Database Requirements

The factory requires the `llm_providers` table with the following structure:

```sql
CREATE TABLE llm_providers (
    provider_id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL,
    configuration TEXT NOT NULL,  -- JSON with api_key, default_model, etc.
    is_active BOOLEAN DEFAULT true,
    is_system BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### Example Configuration Data

```sql
-- System provider (available to all users)
INSERT INTO llm_providers (provider_id, provider_type, configuration, is_active, is_system, is_default)
VALUES (
    'openai-default',
    'openai',
    '{"default_model": "gpt-4", "api_key": "sk-..."}',
    true,
    true,
    true
);

-- User-scoped provider (private to user 123)
INSERT INTO llm_providers (provider_id, provider_type, configuration, is_active, created_by)
VALUES (
    'user-anthropic',
    'anthropic',
    '{"default_model": "claude-sonnet-4-5-20250929", "api_key": "sk-ant-..."}',
    true,
    123
);
```

## Error Handling Best Practices

Always handle provider creation errors gracefully:

```python
from fiberwise_common.llm import LLMProviderFactory

async def get_llm_service(db, provider_id=None, user_id=None):
    """Get LLM service with fallback to default."""
    try:
        if provider_id:
            return await LLMProviderFactory.create_from_db(
                db, provider_id, user_id
            )
        else:
            return await LLMProviderFactory.create_default(db, user_id)
    except ValueError as e:
        logger.warning(f"Provider creation failed: {e}, using default")
        return await LLMProviderFactory.create_default(db)
```

## Testing Integration

When testing code that uses the factory:

```python
import pytest
from unittest.mock import AsyncMock
from fiberwise_common.llm import LLMProviderFactory

@pytest.mark.asyncio
async def test_agent_with_llm(mock_db):
    """Test agent with mocked LLM provider."""

    # Mock the database response
    mock_db.fetch_one = AsyncMock(return_value={
        'provider_id': 'test-provider',
        'provider_type': 'openai',
        'configuration': '{"default_model": "gpt-4", "api_key": "sk-test"}',
        'is_active': True
    })

    # Create provider
    llm_service = await LLMProviderFactory.create_from_db(
        mock_db, 'test-provider'
    )

    # Test with agent
    agent = MyAgent()
    agent.inject_services(llm_service=llm_service)

    # ... rest of test
```

## Performance Considerations

### Caching Providers

For high-throughput scenarios, consider caching provider instances:

```python
from functools import lru_cache
from typing import Optional

class CachedLLMFactory:
    """Factory with provider caching."""

    _cache = {}

    @classmethod
    async def get_provider(cls, db, provider_id: str, user_id: Optional[int] = None):
        """Get provider with caching."""
        cache_key = f"{provider_id}:{user_id}"

        if cache_key not in cls._cache:
            cls._cache[cache_key] = await LLMProviderFactory.create_from_db(
                db, provider_id, user_id
            )

        return cls._cache[cache_key]

    @classmethod
    def clear_cache(cls):
        """Clear provider cache."""
        cls._cache.clear()
```

## Monitoring and Logging

The factory logs important events:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fiberwise_common.llm')

# Factory logs:
# - INFO: Provider creation success
# - ERROR: Configuration parsing errors
# - WARNING: Provider not found
```

## Support for Multiple Providers

To use multiple providers in a single agent:

```python
class MultiProviderAgent(FiberAgent):
    """Agent that uses multiple LLM providers."""

    async def run_agent_async(self, input_data, **kwargs):
        db = kwargs.get('db')
        user_id = kwargs.get('context', {}).get('user_id')

        # Create multiple providers
        openai = await LLMProviderFactory.create_from_db(
            db, 'openai-default', user_id
        )
        anthropic = await LLMProviderFactory.create_from_db(
            db, 'anthropic-default', user_id
        )

        # Use different providers for different tasks
        summary = await openai.generate_completion(
            messages=[{"role": "user", "content": "Summarize: " + input_data['text']}]
        )

        analysis = await anthropic.generate_completion(
            messages=[{"role": "user", "content": "Analyze: " + input_data['text']}]
        )

        return {
            "summary": summary['content'],
            "analysis": analysis['content']
        }
```

## Conclusion

The LLMProviderFactory provides a simple, lightweight integration between FiberWise's database configuration and ia_modules LLM providers. By using this factory:

- Agents get direct access to ia_modules API (no adapter overhead)
- User-scoped providers work automatically
- Configuration is centralized in the database
- Cost tracking is built-in (via LiteLLM)
- No complex translation layer to maintain
