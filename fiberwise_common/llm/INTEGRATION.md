# LLM Provider Factory Integration Guide

This guide describes how the `LLMProviderFactory` integrates with FiberWise's activation system.

## Overview

The `LLMProviderFactory` creates `ia_modules` LLM provider instances from database-stored configuration. It is used **internally** by the platform (e.g., `LLMStep`) to resolve API keys and model settings.

> **Note**: Agent developers should not use `LLMProviderFactory` directly. For LLM access from agent code, use `SubprocessAgentAdapter` from `ia_modules.utils.llm_adapters`. For config-only LLM agents, use `agent_type_id: llm` in the manifest.

## Integration Points

### 1. ActivationProcessor (Internal)

The `ActivationProcessor` uses the factory when dispatching `agent_type_id: llm` activations:

```python
from ..llm import LLMProviderFactory

class ActivationProcessor:
    async def _execute_llm_step(self, agent_config, input_data, activation):
        """Internal: Execute LLM agent via LLMStep"""
        # LLMStep handles provider resolution internally
        # using LLMProviderFactory under the hood
        ...
```

### 2. Agent LLM Access (Developer-Facing)

Agents access LLM capabilities through `SubprocessAgentAdapter`, **not** through injected services:

```python
from fiberwise_sdk import FiberAgent
from ia_modules.utils.llm_adapters import SubprocessAgentAdapter
import os
from typing import Dict, Any

class MyAgent(FiberAgent):
    async def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        fiber = kwargs.get('fiber')

        # LLM call via SubprocessAgentAdapter
        adapter = SubprocessAgentAdapter(cwd=os.getcwd(), timeout_seconds=120.0)
        result = await adapter.generate(
            prompt=f"Analyze: {input_data.get('text', '')}"
        )

        return {"status": "success", "analysis": result}
```

### 3. Pipeline Steps (Developer-Facing)

Pipeline steps can use `SubprocessAgentAdapter` or `litellm.acompletion()` directly:

```python
from ia_modules.pipeline.core import Step
from ia_modules.utils.llm_adapters import SubprocessAgentAdapter
import os

class AnalysisStep(Step):
    async def run(self, data: dict) -> dict:
        adapter = SubprocessAgentAdapter(cwd=os.getcwd(), timeout_seconds=120.0)
        result = await adapter.generate(
            prompt=f"Analyze: {data.get('text', '')}"
        )
        return {"analysis": result}
```

## Database Requirements

The factory requires the `llm_providers` table:

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

## Agent Type Summary

| Agent Type | LLM Access Method | Uses Factory? |
|------------|------------------|---------------|
| `llm` (config-only) | Automatic via LLMStep | Yes (internal) |
| `processor` (pipeline) | SubprocessAgentAdapter or litellm in steps | No |
| `custom` (FiberAgent) | SubprocessAgentAdapter | No |
| `a2a` (remote) | Remote agent handles it | No |
