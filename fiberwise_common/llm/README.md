# LLM Provider Factory

A tiny shim to create `ia_modules` LLM providers from FiberWise database configuration.

## Overview

This module provides a simple factory pattern to bridge FiberWise's database-stored LLM provider configuration with the `ia_modules` LLMProviderService. It's a lightweight integration layer (~100 lines) that:

- Reads provider config from FiberWise `llm_providers` table
- Creates `ia_modules.pipeline.llm_provider_service.LLMProviderService` instances
- Handles user-scoped providers and default provider selection
- No adapter, no wrapping, no translation - just a simple factory

## Architecture

```
FiberWise DB → LLMProviderFactory (tiny shim) → ia_modules LLMProviderService → Agent
                     ↑
                 ~100 lines
```

## Usage

### Basic Usage

```python
from fiberwise_common.llm import LLMProviderFactory

# Create provider from DB
llm_service = await LLMProviderFactory.create_from_db(
    db=db,
    provider_id="openai-default"
)

# Use ia_modules API directly
response = await llm_service.generate_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)
```

### User-Scoped Providers

```python
# Create user's configured provider
llm_service = await LLMProviderFactory.create_from_db(
    db=db,
    provider_id="user-anthropic",
    user_id=123
)
```

### Default Provider

```python
# Use default provider (system or user-scoped)
llm_service = await LLMProviderFactory.create_default(
    db=db,
    user_id=123  # Optional
)
```

### Agent Integration

```python
# In activation_processor.py
from fiberwise_common.llm import LLMProviderFactory

async def _execute_custom_agent(self, agent, activation):
    # Get provider config
    provider_id = agent.config.get('provider_id')
    user_id = activation.get('created_by')

    # Create ia_modules provider (tiny shim!)
    llm_service = await LLMProviderFactory.create_from_db(
        self.db,
        provider_id=provider_id,
        user_id=user_id
    )

    # Inject services
    agent.inject_services(llm_service=llm_service)

    # Execute agent
    result = await agent.execute(input_data)
```

## Database Schema

The factory reads from the `llm_providers` table:

```sql
CREATE TABLE llm_providers (
    provider_id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL,      -- 'openai', 'anthropic', 'google', etc.
    configuration TEXT NOT NULL,      -- JSON: {"default_model": "...", "api_key": "..."}
    is_active BOOLEAN DEFAULT true,
    is_system BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,
    created_by INTEGER,               -- User ID for user-scoped providers
    created_at TEXT,
    updated_at TEXT
);
```

### Configuration JSON Format

```json
{
    "default_model": "gpt-4",
    "api_key": "sk-...",
    "base_url": null,
    "temperature": 0.7,
    "max_tokens": 2048
}
```

## Provider Scoping

The factory supports two scoping modes:

1. **System Providers** (`is_system = true`): Available to all users
2. **User Providers** (`created_by = user_id`): Private to specific user

When `user_id` is provided, the factory will search for:
- User's private providers (`created_by = user_id`)
- OR system providers (`is_system = true`)

## Error Handling

```python
try:
    llm_service = await LLMProviderFactory.create_from_db(
        db, "nonexistent-provider"
    )
except ValueError as e:
    # Provider not found or not accessible
    print(f"Error: {e}")

    # Fall back to default
    llm_service = await LLMProviderFactory.create_default(db)
```

## ia_modules API

The returned `LLMProviderService` provides the standard ia_modules API:

```python
# Generate completion
response = await llm_service.generate_completion(
    messages=[{"role": "user", "content": "..."}],
    temperature=0.7,
    max_tokens=2048
)

# Returns:
{
    "content": "Generated text...",
    "model": "gpt-4",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "cost_usd": 0.003
    },
    "metadata": {
        "finish_reason": "stop",
        "provider_name": "openai-default"
    }
}
```

## Benefits

- **Simple**: ~100 lines, no complex adapter logic
- **Direct**: Agents use ia_modules API natively
- **Consistent**: Same API across all providers (via LiteLLM)
- **Cost tracking**: Automatic via ia_modules
- **User-scoped**: Supports per-user provider configuration
- **No translation**: No API mapping/conversion layer

## Migration from Old LLMProviderService

The factory replaces the old `fiberwise_common.services.llm_provider_service.LLMProviderService` with direct `ia_modules` usage:

**Before:**
```python
from fiberwise_common.services.llm_provider_service import LLMProviderService

llm_service = LLMProviderService(db, user_id=123)
result = await llm_service.generate_completion(
    provider_id="openai-default",
    prompt="Hello"
)
```

**After:**
```python
from fiberwise_common.llm import LLMProviderFactory

llm_service = await LLMProviderFactory.create_from_db(
    db, "openai-default", user_id=123
)
result = await llm_service.generate_completion(
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Implementation Details

- Located at: `fiberwise-common/fiberwise_common/llm/llm_provider_factory.py`
- Dependencies: `ia_modules.pipeline.llm_provider_service`
- Size: ~100 lines of code (excluding docstrings)
- Database interface: Uses FiberWise `BaseDbProvider`
- Async: Fully async/await compatible

## Testing

See `tests/unit/test_llm_provider_factory.py` for comprehensive unit tests covering:
- Basic provider creation
- User scoping
- Default provider selection
- Error handling
- Configuration parsing

## Examples

See `examples/llm_provider_factory_example.py` for complete usage examples.
