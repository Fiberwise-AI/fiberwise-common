# LLM Provider Factory

A factory to create `ia_modules` LLM provider instances from FiberWise database configuration.

## Overview

This module provides `LLMProviderFactory` — a simple factory that reads provider config from the `llm_providers` database table and creates `ia_modules.pipeline.llm_provider_service.LLMProviderService` instances.

```
FiberWise DB → LLMProviderFactory → ia_modules LLMProviderService
```

## Current Role

The factory is used **internally** by platform infrastructure (e.g., `LLMStep` in `ActivationProcessor`) to resolve provider API keys and model settings from the database. It is **not** the primary way agents make LLM calls.

### How Agents Access LLM

- **`agent_type_id: llm`** — Config-only. No code needed. Platform handles dispatch via `LLMStep`.
- **`agent_type_id: processor`** — Pipeline steps use `SubprocessAgentAdapter` or `litellm.acompletion()`.
- **`agent_type_id: custom`** — FiberAgent classes use `SubprocessAgentAdapter` from `ia_modules.utils.llm_adapters`.

See the [Agent Development Guide](../../sites/site-docs-fiberwise-ai/docs-bin/AGENT_DEVELOPMENT_GUIDE.md) for details.

## Usage (Internal)

```python
from fiberwise_common.llm import LLMProviderFactory

# Create provider from DB config
llm_service = await LLMProviderFactory.create_from_db(
    db=db,
    provider_id="openai-default"
)

# Use ia_modules API
response = await llm_service.generate_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)
```

### User-Scoped Providers

```python
llm_service = await LLMProviderFactory.create_from_db(
    db=db,
    provider_id="user-anthropic",
    user_id=123
)
```

### Default Provider

```python
llm_service = await LLMProviderFactory.create_default(
    db=db,
    user_id=123  # Optional
)
```

## Database Schema

Reads from the `llm_providers` table:

```sql
CREATE TABLE llm_providers (
    provider_id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL,
    configuration TEXT NOT NULL,  -- JSON: {"default_model": "...", "api_key": "..."}
    is_active BOOLEAN DEFAULT true,
    is_system BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,
    created_by INTEGER,
    created_at TEXT,
    updated_at TEXT
);
```

## Provider Scoping

1. **System Providers** (`is_system = true`): Available to all users
2. **User Providers** (`created_by = user_id`): Private to specific user

When `user_id` is provided, the factory searches for user-private OR system providers.

## Implementation

- Located at: `fiberwise_common/llm/llm_provider_factory.py`
- Dependencies: `ia_modules.pipeline.llm_provider_service`
- Async: Fully async/await compatible
