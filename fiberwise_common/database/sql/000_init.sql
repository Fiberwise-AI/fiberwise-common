-- Fiberwise Database Schema
-- ========================
-- 
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    hashed_password TEXT,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    is_superuser BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT,
    avatar_url TEXT,
    timezone TEXT DEFAULT 'UTC',
    locale TEXT DEFAULT 'en-US',
    global_role TEXT DEFAULT 'user', -- system_admin, instance_admin, user
    setup_completed BOOLEAN DEFAULT false, -- Track if user completed initial setup
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS apps (
    app_id TEXT PRIMARY KEY,
    creator_user_id INTEGER NOT NULL REFERENCES users(id),
    app_slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    entry_point_url TEXT,
    current_live_app_version_id TEXT,
    marketplace_status TEXT NOT NULL DEFAULT 'draft',
    is_featured BOOLEAN DEFAULT false,
    featured_order INTEGER DEFAULT 0,
    icon_class TEXT DEFAULT 'fas fa-cube',
    tags TEXT, -- JSON array of tags
    stats_json TEXT, -- JSON object for dynamic stats
    install_command TEXT, -- Custom install command
    tutorial_url TEXT,
    source_url TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS app_versions (
    app_version_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    description TEXT,
    manifest_yaml TEXT,
    entry_point_url TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    bundle_path TEXT,
    bundle_size INTEGER,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    submitted_at TEXT,
    reviewed_at TEXT,
    reviewer_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    review_comments TEXT,
    UNIQUE (app_id, version)
);

CREATE TABLE IF NOT EXISTS agent_types (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    capabilities TEXT DEFAULT '{}',
    default_config TEXT DEFAULT '{}',
    icon_url TEXT,
    version TEXT DEFAULT '1.0.0',
    is_system BOOLEAN DEFAULT false,
    is_enabled BOOLEAN DEFAULT true,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    agent_type_id TEXT NOT NULL,
    agent_slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    config TEXT DEFAULT '{}',
    state_data TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    is_enabled BOOLEAN DEFAULT false,
    agent_code TEXT,
    app_id TEXT,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT,
    input_schema TEXT,
    output_schema TEXT,
    dependencies TEXT,
    FOREIGN KEY (agent_type_id) REFERENCES agent_types(id),
    UNIQUE(app_id, agent_slug)  -- Composite unique constraint allows same slug across different apps
);

CREATE TABLE IF NOT EXISTS agent_versions (
    version_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    manifest_yaml TEXT,  -- Agent-specific manifest storage
    config TEXT DEFAULT '{}',
    agent_code TEXT,
    file_path TEXT,
    checksum TEXT,
    input_schema TEXT,
    output_schema TEXT,
    metadata TEXT DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'draft',  -- Added status column
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- LLM Provider Defaults
CREATE TABLE IF NOT EXISTS llm_provider_defaults (
    default_id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL UNIQUE,
    default_name TEXT NOT NULL,
    default_api_endpoint TEXT,
    default_configuration TEXT NOT NULL DEFAULT '{}',
    form_schema TEXT NOT NULL DEFAULT '{}',
    supports_streaming BOOLEAN DEFAULT false,
    supports_functions BOOLEAN DEFAULT false,
    supports_tools BOOLEAN DEFAULT false,
    supports_vision BOOLEAN DEFAULT false,
    request_template TEXT DEFAULT '{}',
    response_template TEXT DEFAULT '{}',
    documentation_url TEXT,
    help_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- LLM Providers (user-configured providers)
CREATE TABLE IF NOT EXISTS llm_providers (
    provider_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    api_endpoint TEXT,
    is_active BOOLEAN DEFAULT true,
    is_system BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,  -- Add is_default column
    configuration TEXT NOT NULL DEFAULT '{}',
    form_schema TEXT DEFAULT '{}',
    provider_description TEXT,
    default_model TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);


-- Dynamic App Platform Tables

-- Stores the definition of data models within an app
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    model_slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (app_id, model_slug)
);

-- Stores the definition of fields within a model
CREATE TABLE IF NOT EXISTS fields (
    field_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    field_column TEXT NOT NULL,
    name TEXT NOT NULL,
    is_primary_key BOOLEAN NOT NULL DEFAULT false,
    data_type TEXT NOT NULL,
    is_required BOOLEAN NOT NULL DEFAULT false,
    is_unique BOOLEAN NOT NULL DEFAULT false,
    default_value_json TEXT,
    validations_json TEXT,
    related_model_id TEXT REFERENCES models(model_id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    relation_details_json TEXT,
    UNIQUE (model_id, field_column)
);

-- Stores the actual instance data for all apps/models
CREATE TABLE IF NOT EXISTS app_model_items (
    item_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    created_by INTEGER REFERENCES users(id),
    data TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Agent Activations (Web platform-specific, more comprehensive)
CREATE TABLE IF NOT EXISTS agent_activations (
    activation_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    agent_type_id TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER,
    input_data TEXT DEFAULT '{}',
    output_data TEXT,
    input_summary TEXT,
    output_summary TEXT,
    error TEXT,
    notes TEXT,
    metadata TEXT DEFAULT '{}',
    context TEXT DEFAULT '{}',
    created_by TEXT,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    dependencies TEXT,
    input_schema TEXT,
    output_schema TEXT,
    execution_metadata TEXT
);

-- Agent Activation Logs
CREATE TABLE IF NOT EXISTS agent_activation_logs (
    activation_id TEXT PRIMARY KEY,
    logs TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (activation_id) REFERENCES agent_activations(activation_id) ON DELETE CASCADE
);

-- OAuth authenticators table
CREATE TABLE IF NOT EXISTS oauth_authenticators (
    id INTEGER PRIMARY KEY,
    authenticator_id TEXT UNIQUE, -- External identifier (e.g., 'google', 'microsoft')
    authenticator_name TEXT NOT NULL,
    authenticator_type TEXT NOT NULL, -- OAuth authenticator type 
    client_id TEXT NOT NULL,
    client_secret TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    authorize_url TEXT, -- OAuth authorization URL
    token_url TEXT, -- OAuth token exchange URL
    scopes TEXT NOT NULL, -- JSON array
    config TEXT, -- JSON object for additional authenticator-specific config
    configuration TEXT, -- Alias for config to match credentials service
    app_id TEXT REFERENCES apps(app_id), -- Associate authenticator with specific app
    is_active BOOLEAN DEFAULT true,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- OAuth sessions table for managing OAuth flow state
CREATE TABLE IF NOT EXISTS oauth_sessions (
    id INTEGER PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    authenticator_name TEXT NOT NULL,
    user_id INTEGER,
    state_token TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    scopes TEXT, -- JSON array
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- OAuth token grants table for storing user OAuth tokens
CREATE TABLE IF NOT EXISTS oauth_token_grants (
    grant_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    authenticator_id TEXT NOT NULL, -- References oauth_authenticators.authenticator_id
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_type TEXT DEFAULT 'Bearer',
    expires_at TEXT,
    scopes TEXT NOT NULL, -- JSON array of granted scopes
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    refreshed_at TEXT,
    is_revoked BOOLEAN DEFAULT false,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (authenticator_id) REFERENCES oauth_authenticators (authenticator_id) ON DELETE CASCADE
);

-- OAuth user app authentications table for tracking which apps users have authenticated
CREATE TABLE IF NOT EXISTS user_app_oauth_authentications (
    auth_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    app_id TEXT NOT NULL, -- References apps.app_id
    authenticator_id TEXT NOT NULL, -- References oauth_authenticators.authenticator_id
    grant_id TEXT NOT NULL, -- References oauth_token_grants.grant_id
    auth_status TEXT DEFAULT 'active', -- active, revoked, expired
    is_active BOOLEAN DEFAULT true,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (app_id) REFERENCES apps (app_id) ON DELETE CASCADE,
    FOREIGN KEY (authenticator_id) REFERENCES oauth_authenticators (authenticator_id) ON DELETE CASCADE,
    FOREIGN KEY (grant_id) REFERENCES oauth_token_grants (grant_id) ON DELETE CASCADE,
    UNIQUE(user_id, app_id, authenticator_id) -- One authentication per user/app/authenticator combo
);


-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_uuid ON users(uuid);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Apps table indexes
CREATE INDEX IF NOT EXISTS idx_apps_creator ON apps(creator_user_id);
CREATE INDEX IF NOT EXISTS idx_apps_slug ON apps(app_slug);
CREATE INDEX IF NOT EXISTS idx_apps_featured ON apps(is_featured) WHERE is_featured = 1;
CREATE INDEX IF NOT EXISTS idx_apps_marketplace_status ON apps(marketplace_status);

-- App versions indexes
CREATE INDEX IF NOT EXISTS idx_app_versions_app_id ON app_versions(app_id);
CREATE INDEX IF NOT EXISTS idx_app_versions_status ON app_versions(status);

-- Agent versions indexes
CREATE INDEX IF NOT EXISTS idx_agent_versions_agent_id ON agent_versions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_versions_version ON agent_versions(version);
CREATE INDEX IF NOT EXISTS idx_agent_versions_status ON agent_versions(status);
CREATE INDEX IF NOT EXISTS idx_agent_versions_is_active ON agent_versions(is_active);

-- Agent activations indexes
CREATE INDEX IF NOT EXISTS idx_agent_activations_agent_id ON agent_activations(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_activations_status ON agent_activations(status);
CREATE INDEX IF NOT EXISTS idx_agent_activations_started_at ON agent_activations(started_at);
CREATE INDEX IF NOT EXISTS idx_agent_activations_organization_id ON agent_activations(organization_id);
CREATE INDEX IF NOT EXISTS idx_agent_activations_dependencies ON agent_activations(dependencies) WHERE dependencies IS NOT NULL;

-- Agent table indexes
CREATE INDEX IF NOT EXISTS idx_agents_checksum ON agents(checksum);
CREATE INDEX IF NOT EXISTS idx_agents_agent_type_id ON agents(agent_type_id);
CREATE INDEX IF NOT EXISTS idx_agents_slug ON agents(agent_slug);

-- OAuth authenticators indexes
CREATE INDEX IF NOT EXISTS idx_oauth_authenticators_active ON oauth_authenticators(is_active) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_oauth_authenticators_app_id ON oauth_authenticators(app_id);

-- OAuth sessions indexes
CREATE INDEX IF NOT EXISTS idx_oauth_sessions_expires ON oauth_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_oauth_sessions_state ON oauth_sessions(state_token);

-- OAuth token grants indexes
CREATE INDEX IF NOT EXISTS idx_oauth_token_grants_user_id ON oauth_token_grants(user_id);
CREATE INDEX IF NOT EXISTS idx_oauth_token_grants_authenticator_id ON oauth_token_grants(authenticator_id);
CREATE INDEX IF NOT EXISTS idx_oauth_token_grants_expires_at ON oauth_token_grants(expires_at);
CREATE INDEX IF NOT EXISTS idx_oauth_token_grants_is_revoked ON oauth_token_grants(is_revoked) WHERE is_revoked = 0;

-- OAuth user app authentications indexes
CREATE INDEX IF NOT EXISTS idx_user_app_oauth_authentications_user_id ON user_app_oauth_authentications(user_id);
CREATE INDEX IF NOT EXISTS idx_user_app_oauth_authentications_app_id ON user_app_oauth_authentications(app_id);
CREATE INDEX IF NOT EXISTS idx_user_app_oauth_authentications_authenticator_id ON user_app_oauth_authentications(authenticator_id);
CREATE INDEX IF NOT EXISTS idx_user_app_oauth_authentications_is_active ON user_app_oauth_authentications(is_active) WHERE is_active = 1;

-- LLM providers indexes
CREATE INDEX IF NOT EXISTS idx_llm_providers_type ON llm_providers(provider_type);
CREATE INDEX IF NOT EXISTS idx_llm_providers_active ON llm_providers(is_active) WHERE is_active = 1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_providers_default ON llm_providers(is_default) WHERE is_default = 1;

-- User providers indexes
CREATE INDEX IF NOT EXISTS idx_user_providers_user_id ON user_providers(user_id);
CREATE INDEX IF NOT EXISTS idx_user_providers_provider_type ON user_providers(provider_type);
CREATE INDEX IF NOT EXISTS idx_user_providers_active ON user_providers(is_active) WHERE is_active = 1;


-- Insert default agent types
INSERT INTO agent_types (id, name, description, capabilities, default_config, is_system, is_enabled) VALUES ('llm', 'LLM', 'Large Language Model agents for chat and text generation', '{}', '{}', true, true) ON CONFLICT (id) DO NOTHING;
INSERT INTO agent_types (id, name, description, capabilities, default_config, is_system, is_enabled) VALUES ('function', 'Function', 'Function-based agents for specific tasks', '{}', '{}', true, true) ON CONFLICT (id) DO NOTHING;
INSERT INTO agent_types (id, name, description, capabilities, default_config, is_system, is_enabled) VALUES ('data-processing', 'Data Processing', 'Agents for data transformation and analysis', '{}', '{}', true, true) ON CONFLICT (id) DO NOTHING;
INSERT INTO agent_types (id, name, description, capabilities, default_config, is_system, is_enabled) VALUES ('integration', 'Integration', 'Agents for connecting external services', '{}', '{}', true, true) ON CONFLICT (id) DO NOTHING;
INSERT INTO agent_types (id, name, description, capabilities, default_config, is_system, is_enabled) VALUES ('custom', 'Custom', 'Custom agents from Python files and user code', '{}', '{}', true, true) ON CONFLICT (id) DO NOTHING;

-- Insert default LLM provider configurations
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('openai-default', 'openai', 'OpenAI', 'https://api.openai.com/v1', '{"default_model": "gpt-3.5-turbo", "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"], "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Key", "description": "Your OpenAI API key"}, "default_model": {"type": "string", "title": "Default Model", "default": "gpt-3.5-turbo", "enum": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 2.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}}, "required": ["api_key"]}', true, true, true, true, '{"model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}], "temperature": {{temperature}}, "max_tokens": {{max_tokens}}}', '{"text": "{{choices.0.message.content}}", "model": "{{model}}", "finish_reason": "{{choices.0.finish_reason}}"}', 'https://platform.openai.com/docs/api-reference', 'Configure your OpenAI API credentials to access GPT models for text generation, chat, and function calling.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('anthropic-default', 'anthropic', 'Anthropic Claude', 'https://api.anthropic.com/v1', '{"default_model": "claude-3-sonnet-20240229", "models": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"], "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Key", "description": "Your Anthropic API key"}, "default_model": {"type": "string", "title": "Default Model", "default": "claude-3-sonnet-20240229", "enum": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 1.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 4096}}, "required": ["api_key"]}', true, false, false, true, '{"model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}], "temperature": {{temperature}}, "max_tokens": {{max_tokens}}}', '{"text": "{{content.0.text}}", "model": "{{model}}", "finish_reason": "{{stop_reason}}"}', 'https://docs.anthropic.com/claude/reference/', 'Configure your Anthropic API credentials to access Claude models for advanced reasoning and analysis.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('cloudflare-default', 'cloudflare', 'Cloudflare Workers AI', 'https://api.cloudflare.com/client/v4', '{"default_model": "@cf/meta/llama-2-7b-chat-fp16", "models": ["@cf/meta/llama-2-7b-chat-fp16", "@cf/meta/llama-2-7b-chat-int8", "@cf/meta/llama-2-13b-chat-fp16", "@cf/mistral/mistral-7b-instruct-v0.1", "@cf/mistral/mistral-7b-instruct-v0.2"], "embedding_model": "@cf/baai/bge-base-en-v1.5", "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Token", "description": "Your Cloudflare API token with Workers AI permissions"}, "account_id": {"type": "string", "title": "Account ID", "description": "Your Cloudflare account ID"}, "default_model": {"type": "string", "title": "Default Text Model", "default": "@cf/meta/llama-2-7b-chat-fp16", "enum": ["@cf/meta/llama-2-7b-chat-fp16", "@cf/meta/llama-2-7b-chat-int8", "@cf/meta/llama-2-13b-chat-fp16", "@cf/mistral/mistral-7b-instruct-v0.1", "@cf/mistral/mistral-7b-instruct-v0.2", "@cf/microsoft/dialoGPT-medium", "@cf/tinyLlama/tinyLlama-1.1b-chat-v1.0"]}, "embedding_model": {"type": "string", "title": "Default Embedding Model", "default": "@cf/baai/bge-base-en-v1.5", "enum": ["@cf/baai/bge-base-en-v1.5", "@cf/baai/bge-small-en-v1.5", "@cf/baai/bge-large-en-v1.5", "@cf/baai/bge-m3"]}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 1.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}}, "required": ["api_key", "account_id"]}', false, false, false, false, '{"model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}], "temperature": {{temperature}}, "max_tokens": {{max_tokens}}}', '{"text": "{{result.response}}", "model": "{{model}}", "finish_reason": "stop"}', 'https://developers.cloudflare.com/workers-ai/', 'Configure Cloudflare Workers AI for edge-based AI inference with ultra-low latency. Supports Llama 2, Mistral, and embedding models running at 200+ global locations.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('huggingface-default', 'huggingface', 'Hugging Face', 'https://api-inference.huggingface.co', '{"default_model": "microsoft/DialoGPT-medium", "models": ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large", "microsoft/DialoGPT-small"], "temperature": 0.7, "max_tokens": 1024}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Token", "description": "Your Hugging Face API token"}, "default_model": {"type": "string", "title": "Default Model", "default": "microsoft/DialoGPT-medium", "description": "Model ID from Hugging Face Hub"}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 1.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 1024, "minimum": 1, "maximum": 2048}}, "required": ["api_key"]}', false, false, false, false, '{"inputs": "{{prompt}}", "parameters": {"temperature": {{temperature}}, "max_new_tokens": {{max_tokens}}}}', '{"text": "{{generated_text}}", "model": "{{model}}"}', 'https://huggingface.co/docs/api-inference/', 'Configure Hugging Face Inference API to access thousands of open-source models including text generation, embeddings, and specialized models.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('openrouter-default', 'openrouter', 'OpenRouter', 'https://openrouter.ai/api/v1', '{"default_model": "meta-llama/llama-3.1-8b-instruct:free", "models": ["meta-llama/llama-3.1-8b-instruct:free", "meta-llama/llama-3.1-70b-instruct", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-pro", "mistralai/mistral-7b-instruct"], "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Key", "description": "Your OpenRouter API key"}, "default_model": {"type": "string", "title": "Default Model", "default": "meta-llama/llama-3.1-8b-instruct:free", "enum": ["meta-llama/llama-3.1-8b-instruct:free", "meta-llama/llama-3.1-70b-instruct", "anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-pro", "mistralai/mistral-7b-instruct", "microsoft/wizardlm-2-8x22b", "qwen/qwen-2-72b-instruct"]}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 2.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}, "site_url": {"type": "string", "title": "Site URL", "description": "Your app/site URL for OpenRouter credits", "default": "https://fiberwise.ai"}, "app_name": {"type": "string", "title": "App Name", "description": "Your app name for OpenRouter credits", "default": "FiberWise"}}, "required": ["api_key"]}', true, true, false, false, '{"model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}], "temperature": {{temperature}}, "max_tokens": {{max_tokens}}}', '{"text": "{{choices.0.message.content}}", "model": "{{model}}", "finish_reason": "{{choices.0.finish_reason}}"}', 'https://openrouter.ai/docs', 'Configure OpenRouter to access 100+ AI models from multiple providers including OpenAI, Anthropic, Meta, Google, and Mistral with unified pricing and API.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('local-default', 'local', 'Local LLM Server', 'http://localhost:8080', '{"default_model": "llama-2-7b-chat", "models": ["llama-2-7b-chat", "llama-2-13b-chat", "codellama-7b-instruct", "mistral-7b-instruct"], "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_endpoint": {"type": "string", "title": "API Endpoint", "default": "http://localhost:8080", "description": "Local LLM server endpoint"}, "api_key": {"type": "string", "title": "API Key", "description": "API key if required by local server"}, "default_model": {"type": "string", "title": "Default Model", "default": "llama-2-7b-chat", "description": "Model name used by your local server"}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 2.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}}, "required": ["api_endpoint"]}', false, false, false, false, '{"model": "{{model}}", "prompt": "{{prompt}}", "temperature": {{temperature}}, "max_tokens": {{max_tokens}}}', '{"text": "{{text}}", "model": "{{model}}"}', 'https://github.com/ggerganov/llama.cpp', 'Configure a local LLM server (Ollama, llama.cpp, text-generation-webui, etc.) for private, self-hosted AI inference.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('gemini-default', 'gemini', 'Google Gemini', 'https://generativelanguage.googleapis.com/v1beta', '{"default_model": "gemini-1.5-pro", "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision"], "temperature": 0.8, "max_tokens": 2048}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Key", "description": "Your Google AI API key"}, "default_model": {"type": "string", "title": "Default Model", "default": "gemini-1.5-pro", "enum": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision"]}, "temperature": {"type": "number", "title": "Temperature", "default": 0.8, "minimum": 0.0, "maximum": 2.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}}, "required": ["api_key"]}', true, false, false, true, '{"model": "{{model}}", "contents": [{"parts": [{"text": "{{prompt}}"}]}], "generationConfig": {"temperature": {{temperature}}, "maxOutputTokens": {{max_tokens}}}}', '{"text": "{{candidates.0.content.parts.0.text}}", "model": "{{model}}", "finish_reason": "{{candidates.0.finishReason}}"}', 'https://ai.google.dev/docs', 'Configure your Google AI API credentials to access Gemini models for multimodal AI capabilities including text, vision, and code generation.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('groq-default', 'groq', 'Groq', 'https://api.groq.com/openai/v1', '{"default_model": "llama3-70b-8192", "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"], "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_key": {"type": "string", "title": "API Key", "description": "Your Groq API key"}, "default_model": {"type": "string", "title": "Default Model", "default": "llama3-70b-8192", "enum": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"]}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 2.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}}, "required": ["api_key"]}', true, true, true, false, '{"model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}], "temperature": {{temperature}}, "max_tokens": {{max_tokens}}}', '{"text": "{{choices.0.message.content}}", "model": "{{model}}", "finish_reason": "{{choices.0.finish_reason}}"}', 'https://console.groq.com/docs/', 'Configure your Groq API credentials for ultra-fast inference with open-source models like Llama 3, Mixtral, and Gemma.') ON CONFLICT (default_id) DO NOTHING;
INSERT INTO llm_provider_defaults (default_id, provider_type, default_name, default_api_endpoint, default_configuration, form_schema, supports_streaming, supports_functions, supports_tools, supports_vision, request_template, response_template, documentation_url, help_text) VALUES ('ollama-default', 'ollama', 'Ollama', 'http://localhost:11434/v1', '{"default_model": "llama3", "models": ["llama3", "llama2", "codellama", "mistral", "gemma"], "temperature": 0.7, "max_tokens": 2048}', '{"type": "object", "properties": {"api_endpoint": {"type": "string", "title": "Ollama Server URL", "default": "http://localhost:11434", "description": "Your Ollama server endpoint"}, "default_model": {"type": "string", "title": "Default Model", "default": "llama3", "description": "Model name available in your Ollama installation"}, "temperature": {"type": "number", "title": "Temperature", "default": 0.7, "minimum": 0.0, "maximum": 2.0}, "max_tokens": {"type": "integer", "title": "Max Tokens", "default": 2048, "minimum": 1, "maximum": 8192}}, "required": ["api_endpoint"]}', true, false, false, false, '{"model": "{{model}}", "messages": [{"role": "user", "content": "{{prompt}}"}], "stream": false, "options": {"temperature": {{temperature}}, "num_predict": {{max_tokens}}}}', '{"text": "{{message.content}}", "model": "{{model}}", "done": true}', 'https://ollama.ai/docs', 'Configure Ollama for local LLM inference with privacy and control. Supports Llama 3, Code Llama, Mistral, and many other open-source models.') ON CONFLICT (default_id) DO NOTHING;

-- Dynamic app platform indexes
CREATE INDEX IF NOT EXISTS idx_models_app ON models(app_id);
CREATE INDEX IF NOT EXISTS idx_fields_model ON fields(model_id);
CREATE INDEX IF NOT EXISTS idx_fields_related_model ON fields(related_model_id);
CREATE INDEX IF NOT EXISTS idx_app_model_items_model ON app_model_items(app_id, model_id);
CREATE INDEX IF NOT EXISTS idx_app_model_items_owner ON app_model_items(created_by);

-- Organizations table (for multi-tenancy)
CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    display_name TEXT,
    description TEXT,
    slug TEXT NOT NULL UNIQUE,
    website TEXT,
    logo_url TEXT,
    billing_email TEXT,
    settings TEXT DEFAULT '{}', -- JSON for organization settings
    subscription_tier TEXT DEFAULT 'free', -- free, starter, pro, enterprise
    max_users INTEGER DEFAULT 5,
    max_apps INTEGER DEFAULT 10,
    max_storage_gb INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Organization Members table (many-to-many between users and organizations)
CREATE TABLE IF NOT EXISTS organization_members (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member', -- owner, admin, member, viewer
    status TEXT DEFAULT 'active', -- active, inactive, pending, suspended
    invited_by INTEGER,
    invited_at TEXT,
    joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (invited_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(organization_id, user_id)
);

-- Agent API Keys (for agent authentication with FiberApp)
CREATE TABLE IF NOT EXISTS agent_api_keys (
    key_id TEXT PRIMARY KEY,
    key_value TEXT NOT NULL UNIQUE,
    app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    description TEXT,
    scopes TEXT DEFAULT '["read", "write"]', -- JSON array of allowed scopes
    expiration TEXT, -- ISO datetime string, NULL for no expiration
    resource_pattern TEXT DEFAULT '*', -- Resource pattern for limiting access
    metadata TEXT DEFAULT '{}', -- JSON metadata
    is_active BOOLEAN DEFAULT true,
    is_revoked BOOLEAN DEFAULT false,
    revoked_at TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Agent API keys indexes
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_app_id ON agent_api_keys(app_id);
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_organization_id ON agent_api_keys(organization_id);
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_agent_id ON agent_api_keys(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_key_value ON agent_api_keys(key_value);
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_expiration ON agent_api_keys(expiration);
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_active ON agent_api_keys(is_active) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_agent_api_keys_revoked ON agent_api_keys(is_revoked) WHERE is_revoked = 0;

-- Teams table (for organizing users within organizations)
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    color TEXT DEFAULT '#3B82F6', -- Hex color for UI
    is_default BOOLEAN DEFAULT false, -- Whether this is the default team for new members
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(organization_id, name)
);

-- Team Members table (many-to-many between users and teams)
CREATE TABLE IF NOT EXISTS team_members (
    id INTEGER PRIMARY KEY,
    team_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT DEFAULT 'member', -- lead, member
    added_by INTEGER,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (added_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(team_id, user_id)
);

-- User Roles table (comprehensive role system)
CREATE TABLE IF NOT EXISTS user_roles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    organization_id INTEGER, -- NULL for global roles
    role_type TEXT NOT NULL, -- global, organization, team, resource
    role_name TEXT NOT NULL, -- system_admin, org_admin, billing_admin, user, viewer, etc.
    resource_type TEXT, -- apps, agents, functions, billing, etc.
    resource_id TEXT, -- Specific resource ID (optional)
    permissions TEXT DEFAULT '[]', -- JSON array of specific permissions
    granted_by INTEGER, -- User who granted this role
    granted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT, -- Optional expiration
    is_active BOOLEAN DEFAULT true,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(user_id, organization_id, role_type, role_name, resource_type, resource_id)
);

-- Role Definitions table (defines what each role can do)
CREATE TABLE IF NOT EXISTS role_definitions (
    id INTEGER PRIMARY KEY,
    role_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    description TEXT,
    role_type TEXT NOT NULL, -- global, organization, team, resource
    permissions TEXT NOT NULL DEFAULT '[]', -- JSON array of permissions
    is_system_role BOOLEAN DEFAULT false, -- System-defined roles vs custom roles
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- User Permissions table (for fine-grained permissions)
CREATE TABLE IF NOT EXISTS user_permissions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    organization_id INTEGER, -- NULL for global permissions
    permission_name TEXT NOT NULL, -- read:apps, write:agents, admin:billing, etc.
    resource_type TEXT, -- apps, agents, functions, billing, users, etc.
    resource_id TEXT, -- Specific resource ID (optional)
    granted_by INTEGER,
    granted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT, -- Optional expiration
    is_active BOOLEAN DEFAULT true,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(user_id, organization_id, permission_name, resource_type, resource_id)
);

-- Invitations table (for managing user invitations to organizations)
CREATE TABLE IF NOT EXISTS invitations (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    team_id INTEGER, -- Optional team assignment
    token TEXT NOT NULL UNIQUE,
    status TEXT DEFAULT 'pending', -- pending, accepted, expired, revoked
    message TEXT, -- Optional invitation message
    invited_by INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    accepted_at TEXT,
    accepted_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL,
    FOREIGN KEY (invited_by) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (accepted_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Permissions table (defines available permissions)
CREATE TABLE IF NOT EXISTS permissions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    resource_type TEXT NOT NULL, -- app, agent, function, user, organization, team
    action TEXT NOT NULL, -- create, read, update, delete, execute, manage
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Role Permissions table (many-to-many between roles and permissions)
CREATE TABLE IF NOT EXISTS role_permissions (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL, -- owner, admin, member, viewer
    permission_id INTEGER NOT NULL,
    resource_scope TEXT DEFAULT 'organization', -- organization, team, self
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE,
    UNIQUE(role, permission_id, resource_scope)
);

-- Resource Permissions table (specific permissions for resources like apps, agents)
CREATE TABLE IF NOT EXISTS resource_permissions (
    id INTEGER PRIMARY KEY,
    resource_type TEXT NOT NULL, -- app, agent, function, etc.
    resource_id TEXT NOT NULL,
    organization_id INTEGER,
    team_id INTEGER,
    user_id INTEGER,
    permission TEXT NOT NULL, -- read, write, admin, execute
    granted_by INTEGER,
    granted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (granted_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Audit Log table (for tracking organization activities)
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER,
    user_id INTEGER,
    action TEXT NOT NULL, -- user.invited, user.added, app.created, etc.
    resource_type TEXT,
    resource_id TEXT,
    details TEXT DEFAULT '{}', -- JSON with additional details
    ip_address TEXT,
    user_agent TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Identity Providers table (for external identity systems like Keycloak, ZITADEL, Auth0)
CREATE TABLE IF NOT EXISTS identity_providers (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL, -- keycloak, zitadel, auth0, azure-ad, okta, generic-oauth
    base_url TEXT NOT NULL, -- Base URL of the identity provider
    realm_or_tenant TEXT, -- Keycloak realm, ZITADEL instance, Auth0 tenant, etc.
    client_id TEXT NOT NULL,
    client_secret TEXT NOT NULL,
    discovery_url TEXT, -- OpenID Connect discovery endpoint
    auth_url TEXT, -- Authorization endpoint (if not using discovery)
    token_url TEXT, -- Token endpoint (if not using discovery)
    userinfo_url TEXT, -- User info endpoint (if not using discovery)
    jwks_url TEXT, -- JWKS endpoint for token validation
    issuer TEXT, -- Token issuer
    scopes TEXT DEFAULT '["openid", "profile", "email"]', -- JSON array of default scopes
    config TEXT DEFAULT '{}', -- Additional provider-specific configuration
    user_mapping TEXT DEFAULT '{}', -- JSON mapping for user attributes
    organization_id INTEGER, -- Organization-specific identity provider
    is_enabled BOOLEAN DEFAULT true,
    is_system_default BOOLEAN DEFAULT false, -- Default provider for the entire system
    is_org_default BOOLEAN DEFAULT false, -- Default provider for the organization
    auto_create_users BOOLEAN DEFAULT true, -- Auto-create users on first login
    auto_assign_role TEXT DEFAULT 'member', -- Auto-assign role to new users
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Identity Provider Groups table (for mapping external groups to internal roles/teams)
CREATE TABLE IF NOT EXISTS identity_provider_groups (
    id INTEGER PRIMARY KEY,
    identity_provider_id INTEGER NOT NULL,
    external_group_name TEXT NOT NULL, -- Group/role name from external provider
    organization_id INTEGER,
    team_id INTEGER, -- Map to internal team
    role TEXT, -- Map to internal role (owner, admin, member, viewer)
    auto_add_to_team BOOLEAN DEFAULT true, -- Automatically add users to team
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (identity_provider_id) REFERENCES identity_providers(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL,
    UNIQUE(identity_provider_id, external_group_name)
);

-- External User Identities table (links local users to external identity provider users)
CREATE TABLE IF NOT EXISTS external_user_identities (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    identity_provider_id INTEGER NOT NULL,
    external_user_id TEXT NOT NULL, -- User ID from external provider
    external_username TEXT, -- Username from external provider
    external_email TEXT, -- Email from external provider
    external_groups TEXT DEFAULT '[]', -- JSON array of groups/roles from external provider
    last_login_at TEXT,
    token_data TEXT DEFAULT '{}', -- Cached token information
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (identity_provider_id) REFERENCES identity_providers(id) ON DELETE CASCADE,
    UNIQUE(identity_provider_id, external_user_id)
);

-- SSO Sessions table (for tracking SSO login sessions)
CREATE TABLE IF NOT EXISTS sso_sessions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL,
    identity_provider_id INTEGER NOT NULL,
    external_session_id TEXT, -- Session ID from external provider
    state_token TEXT NOT NULL,
    nonce TEXT, -- OIDC nonce for security
    redirect_url TEXT,
    access_token_hash TEXT, -- Hash of access token for security
    refresh_token_hash TEXT, -- Hash of refresh token for security
    expires_at TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (identity_provider_id) REFERENCES identity_providers(id) ON DELETE CASCADE
);

-- Usage Metrics table (for tracking resource consumption and billing)
CREATE TABLE IF NOT EXISTS usage_metrics (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    user_id INTEGER,
    team_id INTEGER,
    resource_type TEXT NOT NULL, -- agent, function, app, storage, api_call, etc.
    resource_id TEXT, -- ID of the specific resource (agent_id, function_id, etc.)
    metric_type TEXT NOT NULL, -- execution, duration, storage_gb, api_calls, tokens, etc.
    metric_value REAL NOT NULL, -- Numeric value (count, seconds, bytes, etc.)
    metric_unit TEXT NOT NULL, -- count, seconds, bytes, tokens, etc.
    billing_tier TEXT, -- Subscription tier at time of usage
    cost_cents INTEGER DEFAULT 0, -- Cost in cents for this usage
    metadata TEXT DEFAULT '{}', -- Additional metadata (model used, execution context, etc.)
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    date_bucket TEXT NOT NULL, -- YYYY-MM-DD for daily aggregation
    hour_bucket INTEGER NOT NULL, -- 0-23 for hourly aggregation
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL
);

-- Usage Quotas table (for setting usage limits per organization/user)
CREATE TABLE IF NOT EXISTS usage_quotas (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER,
    user_id INTEGER,
    team_id INTEGER,
    resource_type TEXT NOT NULL, -- agent, function, storage, api_calls, etc.
    quota_type TEXT NOT NULL, -- daily, monthly, total
    quota_limit REAL NOT NULL, -- Maximum allowed usage
    quota_unit TEXT NOT NULL, -- count, seconds, gb, tokens, etc.
    current_usage REAL DEFAULT 0, -- Current usage towards quota
    reset_period TEXT, -- daily, monthly, never
    last_reset_at TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
);

-- Usage Aggregations table (for pre-computed usage summaries)
CREATE TABLE IF NOT EXISTS usage_aggregations (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    user_id INTEGER,
    team_id INTEGER,
    resource_type TEXT NOT NULL,
    aggregation_type TEXT NOT NULL, -- hourly, daily, monthly
    aggregation_date TEXT NOT NULL, -- Date/hour this aggregation covers
    total_executions INTEGER DEFAULT 0,
    total_duration_seconds REAL DEFAULT 0,
    total_cost_cents INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    unique_resources INTEGER DEFAULT 0, -- Number of distinct resources used
    peak_concurrent INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}', -- Additional aggregated metrics
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL,
    UNIQUE(organization_id, user_id, team_id, resource_type, aggregation_type, aggregation_date)
);

-- Rate Limits table (for controlling API usage rates)
CREATE TABLE IF NOT EXISTS rate_limits (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER,
    user_id INTEGER,
    team_id INTEGER,
    resource_type TEXT NOT NULL, -- agent, function, api, activation, etc.
    limit_type TEXT NOT NULL, -- per_minute, per_hour, per_day, concurrent
    limit_value INTEGER NOT NULL, -- Maximum requests/executions allowed
    window_seconds INTEGER NOT NULL, -- Time window for the limit
    current_count INTEGER DEFAULT 0,
    window_start_at TEXT,
    burst_allowance INTEGER DEFAULT 0, -- Additional burst capacity
    action_on_limit TEXT DEFAULT 'block', -- block, throttle, warn
    is_active BOOLEAN DEFAULT true,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
);

-- Rate Limit Violations table (for tracking when limits are exceeded)
CREATE TABLE IF NOT EXISTS rate_limit_violations (
    id INTEGER PRIMARY KEY,
    rate_limit_id INTEGER NOT NULL,
    organization_id INTEGER,
    user_id INTEGER,
    team_id INTEGER,
    resource_type TEXT NOT NULL,
    resource_id TEXT, -- Specific agent/function ID
    attempted_action TEXT NOT NULL, -- create_activation, execute_function, etc.
    violation_count INTEGER DEFAULT 1,
    user_ip TEXT,
    user_agent TEXT,
    blocked_until TEXT, -- When user can try again
    violation_reason TEXT, -- Exceeded daily limit, too many concurrent, etc.
    metadata TEXT DEFAULT '{}', -- Additional context
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rate_limit_id) REFERENCES rate_limits(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL
);

-- User Suspensions table (for temporarily blocking users who abuse limits)
CREATE TABLE IF NOT EXISTS user_suspensions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    organization_id INTEGER NOT NULL,
    suspension_type TEXT NOT NULL, -- rate_limit, security, admin, billing
    resource_types TEXT DEFAULT '["all"]', -- JSON array of blocked resources
    reason TEXT NOT NULL,
    suspended_until TEXT, -- NULL for indefinite
    is_active BOOLEAN DEFAULT true,
    can_appeal BOOLEAN DEFAULT true,
    violation_count INTEGER DEFAULT 1,
    created_by INTEGER, -- Admin who created suspension
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Billing Events table (for tracking billable events and invoicing)
CREATE TABLE IF NOT EXISTS billing_events (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    event_type TEXT NOT NULL, -- usage, subscription_change, credit_purchase, etc.
    resource_type TEXT, -- What resource was used
    resource_id TEXT, -- Specific resource identifier
    user_id INTEGER, -- User who triggered the event
    team_id INTEGER, -- Team context if applicable
    quantity REAL NOT NULL, -- Amount of resource consumed
    unit_price_cents INTEGER NOT NULL, -- Price per unit in cents
    total_cost_cents INTEGER NOT NULL, -- Total cost for this event
    billing_period TEXT, -- YYYY-MM for monthly billing
    metadata TEXT DEFAULT '{}', -- Additional billing details
    invoiced_at TEXT, -- When this was included in an invoice
    invoice_id TEXT, -- Reference to external invoice system
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL
);

-- Subscription Tiers table (for defining subscription plans and limits)
CREATE TABLE IF NOT EXISTS subscription_tiers (
    id INTEGER PRIMARY KEY,
    tier_name TEXT NOT NULL UNIQUE, -- free, starter, pro, enterprise
    display_name TEXT NOT NULL,
    description TEXT,
    monthly_price_cents INTEGER NOT NULL,
    annual_price_cents INTEGER,
    max_users INTEGER DEFAULT -1, -- -1 for unlimited
    max_apps INTEGER DEFAULT -1,
    max_storage_gb INTEGER DEFAULT -1,
    max_monthly_executions INTEGER DEFAULT -1,
    max_concurrent_executions INTEGER DEFAULT 10,
    max_execution_duration_seconds INTEGER DEFAULT 300, -- 5 minutes
    includes_priority_support BOOLEAN DEFAULT false,
    includes_sso BOOLEAN DEFAULT false,
    includes_advanced_analytics BOOLEAN DEFAULT false,
    includes_compliance_tools BOOLEAN DEFAULT false,
    includes_white_labeling BOOLEAN DEFAULT false,
    includes_dedicated_support BOOLEAN DEFAULT false,
    includes_custom_integrations BOOLEAN DEFAULT false,
    rate_limit_per_minute INTEGER DEFAULT 60,
    features TEXT DEFAULT '[]', -- JSON array of feature flags
    is_active BOOLEAN DEFAULT true,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Security Policies table (for enterprise security configuration)
CREATE TABLE IF NOT EXISTS security_policies (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    policy_type TEXT NOT NULL, -- password, session, mfa, ip_restriction, etc.
    policy_name TEXT NOT NULL,
    policy_config TEXT NOT NULL, -- JSON configuration
    is_enforced BOOLEAN DEFAULT true,
    enforcement_level TEXT DEFAULT 'required', -- required, recommended, optional
    applies_to_roles TEXT DEFAULT '["member", "admin", "owner"]', -- JSON array
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(organization_id, policy_type, policy_name)
);

-- IP Restrictions table (for allowing/blocking specific IP addresses or ranges)
CREATE TABLE IF NOT EXISTS ip_restrictions (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    ip_address TEXT NOT NULL, -- Single IP or CIDR range
    restriction_type TEXT NOT NULL, -- allow, deny
    description TEXT,
    applies_to TEXT DEFAULT 'all', -- all, admin_only, api_only
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- MFA Settings table (for multi-factor authentication configuration)
CREATE TABLE IF NOT EXISTS mfa_settings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    mfa_type TEXT NOT NULL, -- totp, sms, email, hardware_key
    secret_key TEXT, -- Encrypted secret for TOTP
    phone_number TEXT, -- For SMS MFA
    backup_codes TEXT, -- JSON array of backup codes
    is_enabled BOOLEAN DEFAULT true,
    is_primary BOOLEAN DEFAULT false,
    last_used_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Data Retention Policies table (for compliance and data governance)
CREATE TABLE IF NOT EXISTS data_retention_policies (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    resource_type TEXT NOT NULL, -- logs, user_data, audit_logs, etc.
    retention_days INTEGER NOT NULL,
    auto_delete BOOLEAN DEFAULT true,
    archive_before_delete BOOLEAN DEFAULT false,
    archive_location TEXT, -- S3 bucket, etc.
    policy_reason TEXT, -- Legal compliance, storage costs, etc.
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Compliance Reports table (for generating compliance and audit reports)
CREATE TABLE IF NOT EXISTS compliance_reports (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    report_type TEXT NOT NULL, -- gdpr, hipaa, sox, custom
    report_period_start TEXT NOT NULL,
    report_period_end TEXT NOT NULL,
    status TEXT DEFAULT 'pending', -- pending, generating, completed, failed
    report_data TEXT, -- JSON report data
    file_path TEXT, -- Path to generated report file
    generated_by INTEGER,
    requested_by INTEGER,
    generated_at TEXT,
    expires_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (generated_by) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (requested_by) REFERENCES users(id) ON DELETE SET NULL
);

-- White Label Settings table (for enterprise branding)
CREATE TABLE IF NOT EXISTS white_label_settings (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL UNIQUE,
    company_name TEXT,
    company_logo_url TEXT,
    primary_color TEXT DEFAULT '#3B82F6',
    secondary_color TEXT DEFAULT '#EF4444',
    favicon_url TEXT,
    custom_domain TEXT,
    support_email TEXT,
    support_url TEXT,
    terms_url TEXT,
    privacy_url TEXT,
    custom_css TEXT, -- Custom CSS for further branding
    email_template_header TEXT, -- Custom email header
    email_template_footer TEXT, -- Custom email footer
    is_enabled BOOLEAN DEFAULT false,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE
);

-- Custom Fields table (for enterprise custom data requirements)
CREATE TABLE IF NOT EXISTS custom_fields (
    id INTEGER PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL, -- user, team, app, agent, function
    field_name TEXT NOT NULL,
    field_type TEXT NOT NULL, -- text, number, boolean, date, select, multi_select
    field_options TEXT, -- JSON array for select fields
    is_required BOOLEAN DEFAULT false,
    is_searchable BOOLEAN DEFAULT true,
    display_order INTEGER DEFAULT 0,
    validation_rules TEXT DEFAULT '{}', -- JSON validation rules
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE(organization_id, entity_type, field_name)
);

-- Custom Field Values table (for storing custom field data)
CREATE TABLE IF NOT EXISTS custom_field_values (
    id INTEGER PRIMARY KEY,
    custom_field_id INTEGER NOT NULL,
    entity_id TEXT NOT NULL, -- ID of the user, team, app, etc.
    field_value TEXT, -- Stored as text, parsed based on field_type
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (custom_field_id) REFERENCES custom_fields(id) ON DELETE CASCADE,
    UNIQUE(custom_field_id, entity_id)
);

-- Enterprise Integrations table (for custom enterprise system integrations)
CREATE TABLE IF NOT EXISTS enterprise_integrations (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    integration_type TEXT NOT NULL, -- ldap, active_directory, okta, salesforce, etc.
    name TEXT NOT NULL,
    config TEXT NOT NULL, -- JSON configuration
    sync_settings TEXT DEFAULT '{}', -- User/group sync settings
    is_enabled BOOLEAN DEFAULT true,
    last_sync_at TEXT,
    sync_status TEXT DEFAULT 'never', -- never, success, error, in_progress
    sync_error TEXT,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Scheduled Tasks table (for enterprise automation and maintenance)
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER,
    task_type TEXT NOT NULL, -- backup, cleanup, sync, report_generation, etc.
    task_name TEXT NOT NULL,
    schedule_cron TEXT NOT NULL, -- Cron expression
    task_config TEXT DEFAULT '{}', -- JSON task configuration
    is_enabled BOOLEAN DEFAULT true,
    last_run_at TEXT,
    next_run_at TEXT,
    run_status TEXT DEFAULT 'never', -- never, success, error, running
    run_result TEXT, -- JSON result data
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- API Webhooks table (for enterprise event notifications)
CREATE TABLE IF NOT EXISTS api_webhooks (
    id INTEGER PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    organization_id INTEGER NOT NULL,
    webhook_url TEXT NOT NULL,
    webhook_secret TEXT, -- For webhook signature verification
    event_types TEXT NOT NULL, -- JSON array of events to listen for
    is_enabled BOOLEAN DEFAULT true,
    timeout_seconds INTEGER DEFAULT 30,
    retry_attempts INTEGER DEFAULT 3,
    last_triggered_at TEXT,
    total_deliveries INTEGER DEFAULT 0,
    successful_deliveries INTEGER DEFAULT 0,
    failed_deliveries INTEGER DEFAULT 0,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Webhook Deliveries table (for tracking webhook delivery status)
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id INTEGER PRIMARY KEY,
    webhook_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL, -- JSON payload sent
    response_status INTEGER, -- HTTP response status
    response_body TEXT,
    delivery_duration_ms INTEGER,
    attempt_count INTEGER DEFAULT 1,
    delivered_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (webhook_id) REFERENCES api_webhooks(id) ON DELETE CASCADE
);

-- API Keys table (for CLI authentication)
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    organization_id INTEGER,
    scopes TEXT DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    expires_at TEXT,
    last_used_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_organizations_uuid ON organizations(uuid);
CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);
CREATE INDEX IF NOT EXISTS idx_organizations_created_by ON organizations(created_by);
CREATE INDEX IF NOT EXISTS idx_organizations_active ON organizations(is_active) WHERE is_active = 1;

-- Organization members indexes
CREATE INDEX IF NOT EXISTS idx_organization_members_org_id ON organization_members(organization_id);
CREATE INDEX IF NOT EXISTS idx_organization_members_user_id ON organization_members(user_id);
CREATE INDEX IF NOT EXISTS idx_organization_members_role ON organization_members(role);
CREATE INDEX IF NOT EXISTS idx_organization_members_status ON organization_members(status);

-- Teams indexes
CREATE INDEX IF NOT EXISTS idx_teams_uuid ON teams(uuid);
CREATE INDEX IF NOT EXISTS idx_teams_organization_id ON teams(organization_id);
CREATE INDEX IF NOT EXISTS idx_teams_created_by ON teams(created_by);
CREATE INDEX IF NOT EXISTS idx_teams_default ON teams(is_default) WHERE is_default = 1;

-- Team members indexes
CREATE INDEX IF NOT EXISTS idx_team_members_team_id ON team_members(team_id);
CREATE INDEX IF NOT EXISTS idx_team_members_user_id ON team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_team_members_role ON team_members(role);

-- Invitations indexes
CREATE INDEX IF NOT EXISTS idx_invitations_uuid ON invitations(uuid);
CREATE INDEX IF NOT EXISTS idx_invitations_organization_id ON invitations(organization_id);
CREATE INDEX IF NOT EXISTS idx_invitations_email ON invitations(email);
CREATE INDEX IF NOT EXISTS idx_invitations_token ON invitations(token);
CREATE INDEX IF NOT EXISTS idx_invitations_status ON invitations(status);
CREATE INDEX IF NOT EXISTS idx_invitations_expires_at ON invitations(expires_at);

-- Permissions indexes
CREATE INDEX IF NOT EXISTS idx_permissions_name ON permissions(name);
CREATE INDEX IF NOT EXISTS idx_permissions_resource_type ON permissions(resource_type);
CREATE INDEX IF NOT EXISTS idx_permissions_action ON permissions(action);

-- Role permissions indexes
CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role);
CREATE INDEX IF NOT EXISTS idx_role_permissions_permission_id ON role_permissions(permission_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_scope ON role_permissions(resource_scope);

-- Resource permissions indexes
CREATE INDEX IF NOT EXISTS idx_resource_permissions_resource ON resource_permissions(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_resource_permissions_org_id ON resource_permissions(organization_id);
CREATE INDEX IF NOT EXISTS idx_resource_permissions_team_id ON resource_permissions(team_id);
CREATE INDEX IF NOT EXISTS idx_resource_permissions_user_id ON resource_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_resource_permissions_permission ON resource_permissions(permission);

-- Audit logs indexes
CREATE INDEX IF NOT EXISTS idx_audit_logs_uuid ON audit_logs(uuid);
CREATE INDEX IF NOT EXISTS idx_audit_logs_org_id ON audit_logs(organization_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Identity providers indexes
CREATE INDEX IF NOT EXISTS idx_identity_providers_uuid ON identity_providers(uuid);
CREATE INDEX IF NOT EXISTS idx_identity_providers_type ON identity_providers(provider_type);
CREATE INDEX IF NOT EXISTS idx_identity_providers_org_id ON identity_providers(organization_id);
CREATE INDEX IF NOT EXISTS idx_identity_providers_enabled ON identity_providers(is_enabled) WHERE is_enabled = 1;
CREATE INDEX IF NOT EXISTS idx_identity_providers_system_default ON identity_providers(is_system_default) WHERE is_system_default = 1;
CREATE INDEX IF NOT EXISTS idx_identity_providers_org_default ON identity_providers(is_org_default, organization_id) WHERE is_org_default = 1;

-- Identity provider groups indexes
CREATE INDEX IF NOT EXISTS idx_identity_provider_groups_provider_id ON identity_provider_groups(identity_provider_id);
CREATE INDEX IF NOT EXISTS idx_identity_provider_groups_org_id ON identity_provider_groups(organization_id);
CREATE INDEX IF NOT EXISTS idx_identity_provider_groups_team_id ON identity_provider_groups(team_id);
CREATE INDEX IF NOT EXISTS idx_identity_provider_groups_external_name ON identity_provider_groups(external_group_name);

-- External user identities indexes
CREATE INDEX IF NOT EXISTS idx_external_user_identities_user_id ON external_user_identities(user_id);
CREATE INDEX IF NOT EXISTS idx_external_user_identities_provider_id ON external_user_identities(identity_provider_id);
CREATE INDEX IF NOT EXISTS idx_external_user_identities_external_user_id ON external_user_identities(external_user_id);
CREATE INDEX IF NOT EXISTS idx_external_user_identities_external_email ON external_user_identities(external_email);
CREATE INDEX IF NOT EXISTS idx_external_user_identities_last_login ON external_user_identities(last_login_at);

-- SSO sessions indexes
CREATE INDEX IF NOT EXISTS idx_sso_sessions_session_id ON sso_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sso_sessions_user_id ON sso_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sso_sessions_provider_id ON sso_sessions(identity_provider_id);
CREATE INDEX IF NOT EXISTS idx_sso_sessions_expires_at ON sso_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sso_sessions_state_token ON sso_sessions(state_token);

-- Usage metrics indexes
CREATE INDEX IF NOT EXISTS idx_usage_metrics_uuid ON usage_metrics(uuid);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_org_id ON usage_metrics(organization_id);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_user_id ON usage_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_team_id ON usage_metrics(team_id);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_resource ON usage_metrics(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_date_bucket ON usage_metrics(date_bucket);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_recorded_at ON usage_metrics(recorded_at);

-- Usage quotas indexes
CREATE INDEX IF NOT EXISTS idx_usage_quotas_org_id ON usage_quotas(organization_id);
CREATE INDEX IF NOT EXISTS idx_usage_quotas_user_id ON usage_quotas(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_quotas_team_id ON usage_quotas(team_id);
CREATE INDEX IF NOT EXISTS idx_usage_quotas_resource_type ON usage_quotas(resource_type);
CREATE INDEX IF NOT EXISTS idx_usage_quotas_active ON usage_quotas(is_active) WHERE is_active = 1;

-- Usage aggregations indexes
CREATE INDEX IF NOT EXISTS idx_usage_aggregations_org_id ON usage_aggregations(organization_id);
CREATE INDEX IF NOT EXISTS idx_usage_aggregations_user_id ON usage_aggregations(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_aggregations_team_id ON usage_aggregations(team_id);
CREATE INDEX IF NOT EXISTS idx_usage_aggregations_resource_type ON usage_aggregations(resource_type);
CREATE INDEX IF NOT EXISTS idx_usage_aggregations_date ON usage_aggregations(aggregation_date);

-- Rate limits indexes
CREATE INDEX IF NOT EXISTS idx_rate_limits_org_id ON rate_limits(organization_id);
CREATE INDEX IF NOT EXISTS idx_rate_limits_user_id ON rate_limits(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limits_team_id ON rate_limits(team_id);
CREATE INDEX IF NOT EXISTS idx_rate_limits_resource_type ON rate_limits(resource_type);
CREATE INDEX IF NOT EXISTS idx_rate_limits_active ON rate_limits(is_active) WHERE is_active = 1;

-- Billing events indexes
CREATE INDEX IF NOT EXISTS idx_billing_events_uuid ON billing_events(uuid);
CREATE INDEX IF NOT EXISTS idx_billing_events_org_id ON billing_events(organization_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_user_id ON billing_events(user_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_billing_period ON billing_events(billing_period);
CREATE INDEX IF NOT EXISTS idx_billing_events_recorded_at ON billing_events(recorded_at);

-- Security policies indexes
CREATE INDEX IF NOT EXISTS idx_security_policies_org_id ON security_policies(organization_id);
CREATE INDEX IF NOT EXISTS idx_security_policies_type ON security_policies(policy_type);
CREATE INDEX IF NOT EXISTS idx_security_policies_enforced ON security_policies(is_enforced) WHERE is_enforced = 1;

-- IP restrictions indexes
CREATE INDEX IF NOT EXISTS idx_ip_restrictions_org_id ON ip_restrictions(organization_id);
CREATE INDEX IF NOT EXISTS idx_ip_restrictions_active ON ip_restrictions(is_active) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_ip_restrictions_ip_address ON ip_restrictions(ip_address);

-- MFA settings indexes
CREATE INDEX IF NOT EXISTS idx_mfa_settings_user_id ON mfa_settings(user_id);
CREATE INDEX IF NOT EXISTS idx_mfa_settings_enabled ON mfa_settings(is_enabled) WHERE is_enabled = 1;
CREATE INDEX IF NOT EXISTS idx_mfa_settings_primary ON mfa_settings(is_primary) WHERE is_primary = 1;

-- Data retention policies indexes
CREATE INDEX IF NOT EXISTS idx_data_retention_policies_org_id ON data_retention_policies(organization_id);
CREATE INDEX IF NOT EXISTS idx_data_retention_policies_resource_type ON data_retention_policies(resource_type);
CREATE INDEX IF NOT EXISTS idx_data_retention_policies_active ON data_retention_policies(is_active) WHERE is_active = 1;

-- Compliance reports indexes
CREATE INDEX IF NOT EXISTS idx_compliance_reports_uuid ON compliance_reports(uuid);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_org_id ON compliance_reports(organization_id);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_type ON compliance_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_status ON compliance_reports(status);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_period ON compliance_reports(report_period_start, report_period_end);

-- White label settings indexes
CREATE INDEX IF NOT EXISTS idx_white_label_settings_org_id ON white_label_settings(organization_id);
CREATE INDEX IF NOT EXISTS idx_white_label_settings_enabled ON white_label_settings(is_enabled) WHERE is_enabled = 1;
CREATE INDEX IF NOT EXISTS idx_white_label_settings_custom_domain ON white_label_settings(custom_domain);

-- Custom fields indexes
CREATE INDEX IF NOT EXISTS idx_custom_fields_org_id ON custom_fields(organization_id);
CREATE INDEX IF NOT EXISTS idx_custom_fields_entity_type ON custom_fields(entity_type);
CREATE INDEX IF NOT EXISTS idx_custom_fields_searchable ON custom_fields(is_searchable) WHERE is_searchable = 1;

-- Custom field values indexes
CREATE INDEX IF NOT EXISTS idx_custom_field_values_field_id ON custom_field_values(custom_field_id);
CREATE INDEX IF NOT EXISTS idx_custom_field_values_entity_id ON custom_field_values(entity_id);

-- Enterprise integrations indexes
CREATE INDEX IF NOT EXISTS idx_enterprise_integrations_uuid ON enterprise_integrations(uuid);
CREATE INDEX IF NOT EXISTS idx_enterprise_integrations_org_id ON enterprise_integrations(organization_id);
CREATE INDEX IF NOT EXISTS idx_enterprise_integrations_type ON enterprise_integrations(integration_type);
CREATE INDEX IF NOT EXISTS idx_enterprise_integrations_enabled ON enterprise_integrations(is_enabled) WHERE is_enabled = 1;

-- Scheduled tasks indexes
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_uuid ON scheduled_tasks(uuid);
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_org_id ON scheduled_tasks(organization_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_enabled ON scheduled_tasks(is_enabled) WHERE is_enabled = 1;
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run ON scheduled_tasks(next_run_at);

-- API webhooks indexes
CREATE INDEX IF NOT EXISTS idx_api_webhooks_uuid ON api_webhooks(uuid);
CREATE INDEX IF NOT EXISTS idx_api_webhooks_org_id ON api_webhooks(organization_id);
CREATE INDEX IF NOT EXISTS idx_api_webhooks_enabled ON api_webhooks(is_enabled) WHERE is_enabled = 1;

-- Webhook deliveries indexes
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_id ON webhook_deliveries(webhook_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_delivered_at ON webhook_deliveries(delivered_at);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_response_status ON webhook_deliveries(response_status);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_organization_id ON api_keys(organization_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON api_keys(key_prefix);

-- App Installations table (for CLI app management)  
CREATE TABLE IF NOT EXISTS app_installations (
    id INTEGER PRIMARY KEY,
    app_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    installation_id TEXT UNIQUE NOT NULL,
    organization_id INTEGER,
    install_path TEXT,
    install_at_root BOOLEAN DEFAULT false,
    installed_version TEXT,
    config TEXT DEFAULT '{}',
    status TEXT DEFAULT 'active',
    installed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (app_id) REFERENCES apps(app_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_app_installations_app_id ON app_installations(app_id);
CREATE INDEX IF NOT EXISTS idx_app_installations_user_id ON app_installations(user_id);
CREATE INDEX IF NOT EXISTS idx_app_installations_organization_id ON app_installations(organization_id);
CREATE INDEX IF NOT EXISTS idx_app_installations_installation_id ON app_installations(installation_id);

-- App Routes table (for storing routes from app manifests)
CREATE TABLE IF NOT EXISTS app_routes (
    id INTEGER PRIMARY KEY,
    route_id TEXT NOT NULL UNIQUE, -- GUID for route identification
    app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    app_version_id TEXT REFERENCES app_versions(app_version_id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    title TEXT,
    icon TEXT,
    component TEXT,
    description TEXT,
    route_type TEXT DEFAULT 'page', -- 'page', 'api', 'modal', etc.
    is_active BOOLEAN DEFAULT true,
    sort_order INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}', -- JSON for additional route metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for fast route lookups
CREATE INDEX IF NOT EXISTS idx_app_routes_app_id ON app_routes(app_id);
CREATE INDEX IF NOT EXISTS idx_app_routes_path ON app_routes(path);
CREATE INDEX IF NOT EXISTS idx_app_routes_active ON app_routes(is_active) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_app_routes_app_version ON app_routes(app_version_id);
CREATE INDEX IF NOT EXISTS idx_app_routes_app_path ON app_routes(app_id, path);

-- Functions table
CREATE TABLE IF NOT EXISTS functions (
    function_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    function_type TEXT NOT NULL DEFAULT 'utility',
    tags TEXT DEFAULT '[]', -- JSON array of tags
    input_schema TEXT DEFAULT '{}', -- JSON schema for input validation
    output_schema TEXT DEFAULT '{}', -- JSON schema for output validation
    implementation TEXT, -- Function implementation code (nullable for file-based implementations)
    config TEXT DEFAULT '{}', -- JSON configuration for the function
    is_async BOOLEAN NOT NULL DEFAULT false,
    is_system BOOLEAN NOT NULL DEFAULT false,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Function Executions table
CREATE TABLE IF NOT EXISTS function_executions (
    execution_id TEXT PRIMARY KEY,
    function_id TEXT NOT NULL REFERENCES functions(function_id),
    status TEXT NOT NULL, -- queued, running, completed, failed
    input_data TEXT DEFAULT '{}', -- JSON input data
    output_data TEXT, -- JSON output data
    error TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Function Versions table
CREATE TABLE IF NOT EXISTS function_versions (
    version_id TEXT PRIMARY KEY,
    function_id TEXT NOT NULL REFERENCES functions(function_id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    description TEXT,
    manifest_yaml TEXT, -- Function-specific manifest storage
    implementation TEXT,
    input_schema TEXT,
    output_schema TEXT,
    config TEXT DEFAULT '{}', -- JSON configuration for the function version
    status TEXT NOT NULL DEFAULT 'draft',
    is_active BOOLEAN NOT NULL DEFAULT false,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Function Implementations table (for file-based implementations)
CREATE TABLE IF NOT EXISTS function_implementations (
    implementation_id TEXT PRIMARY KEY,
    function_id TEXT NOT NULL REFERENCES functions(function_id) ON DELETE CASCADE,
    implementation_path TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'python',
    entrypoint_file TEXT NOT NULL,
    function_name TEXT,
    version TEXT DEFAULT '1.0.0',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Functions App Association table (many-to-many)
CREATE TABLE IF NOT EXISTS functions_app (
    function_id TEXT NOT NULL REFERENCES functions(function_id) ON DELETE CASCADE,
    app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (function_id, app_id)
);



-- Note: Agent implementations are now stored directly in agents.agent_code field
-- agent_implementations table removed - agents follow same pattern as functions

-- Executor Types table (for execution API keys)
CREATE TABLE IF NOT EXISTS executor_types (
    type_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Execution API Keys table (for temporary function/agent/pipeline execution keys)
CREATE TABLE IF NOT EXISTS execution_api_keys (
    key_id TEXT PRIMARY KEY,
    key_value TEXT NOT NULL UNIQUE,
    app_id TEXT REFERENCES apps(app_id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    executor_type_id TEXT NOT NULL REFERENCES executor_types(type_id),
    executor_id TEXT NOT NULL, -- function_id, agent_id, or pipeline_id
    created_by INTEGER REFERENCES users(id),
    scopes TEXT DEFAULT '[]', -- JSON array of scopes
    expiration TEXT, -- ISO datetime string
    resource_pattern TEXT DEFAULT '*',
    metadata TEXT DEFAULT '{}', -- JSON metadata
    is_revoked INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Pipelines table
CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id TEXT PRIMARY KEY,
    pipeline_slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    file_path TEXT, -- For file-based pipelines
    definition TEXT DEFAULT '{}', -- For UI-defined pipelines
    config TEXT DEFAULT '{}',
    app_id TEXT REFERENCES apps(app_id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(app_id, pipeline_slug)
);

-- Pipeline Executions table
CREATE TABLE IF NOT EXISTS pipeline_executions (
    execution_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 10,
    input_data TEXT DEFAULT '{}',
    context TEXT DEFAULT '{}',
    results TEXT, -- Final execution result as JSON
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- HITL (Human-in-the-Loop) support columns
    human_input_config TEXT, -- JSON configuration for human input UI when pipeline is paused
    human_input_data TEXT, -- JSON data submitted by human when resuming pipeline  
    waiting_step_id TEXT -- ID of the step waiting for human input
);

-- Pipeline Step Executions table (for tracking individual step executions)
CREATE TABLE IF NOT EXISTS pipeline_step_executions (
    step_execution_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    step_id TEXT NOT NULL, -- Step ID from pipeline definition
    step_name TEXT,
    step_type TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    input_data TEXT DEFAULT '{}',
    output_data TEXT,
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline Versions table
CREATE TABLE IF NOT EXISTS pipeline_versions (
    version_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    description TEXT,
    manifest_yaml TEXT,
    config TEXT DEFAULT '{}',
    file_path TEXT,
    checksum TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    is_active BOOLEAN DEFAULT true,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline Activations table (unified execution system)
CREATE TABLE IF NOT EXISTS pipeline_activations (
    activation_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    pipeline_slug TEXT,
    input_data TEXT DEFAULT '{}',
    output_data TEXT DEFAULT '{}', 
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    created_by INTEGER NOT NULL REFERENCES users(id),
    app_id TEXT REFERENCES apps(app_id) ON DELETE CASCADE,
    context TEXT DEFAULT '{}', -- Additional execution context
    priority INTEGER DEFAULT 10,
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for pipeline_activations
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_status ON pipeline_activations(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_created_by ON pipeline_activations(created_by);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_pipeline_id ON pipeline_activations(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_app_id ON pipeline_activations(app_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_priority_created ON pipeline_activations(priority DESC, created_at ASC);

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config TEXT DEFAULT '{}',
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Versions table
CREATE TABLE IF NOT EXISTS workflow_versions (
    version_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    config TEXT DEFAULT '{}',
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for function tables
CREATE INDEX IF NOT EXISTS idx_functions_function_type ON functions(function_type);
CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
CREATE INDEX IF NOT EXISTS idx_functions_created_by ON functions(created_by);

CREATE INDEX IF NOT EXISTS idx_function_executions_function_id ON function_executions(function_id);
CREATE INDEX IF NOT EXISTS idx_function_executions_status ON function_executions(status);
CREATE INDEX IF NOT EXISTS idx_function_executions_created_by ON function_executions(created_by);
CREATE INDEX IF NOT EXISTS idx_function_executions_started_at ON function_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_function_versions_function_id ON function_versions(function_id);
CREATE INDEX IF NOT EXISTS idx_function_versions_status ON function_versions(status);
CREATE INDEX IF NOT EXISTS idx_function_versions_version ON function_versions(version);

CREATE INDEX IF NOT EXISTS idx_function_implementations_function_id ON function_implementations(function_id);
CREATE INDEX IF NOT EXISTS idx_function_implementations_active ON function_implementations(is_active) WHERE is_active = 1;

CREATE INDEX IF NOT EXISTS idx_functions_app_function_id ON functions_app(function_id);
CREATE INDEX IF NOT EXISTS idx_functions_app_app_id ON functions_app(app_id);

-- Note: agent_implementations indexes removed since table was removed

CREATE INDEX IF NOT EXISTS idx_execution_api_keys_app_id ON execution_api_keys(app_id);
CREATE INDEX IF NOT EXISTS idx_execution_api_keys_organization_id ON execution_api_keys(organization_id);
CREATE INDEX IF NOT EXISTS idx_execution_api_keys_executor ON execution_api_keys(executor_type_id, executor_id);
CREATE INDEX IF NOT EXISTS idx_execution_api_keys_expiration ON execution_api_keys(expiration);
CREATE INDEX IF NOT EXISTS idx_execution_api_keys_revoked ON execution_api_keys(is_revoked) WHERE is_revoked = 0;

CREATE INDEX IF NOT EXISTS idx_pipeline_versions_pipeline_id ON pipeline_versions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_versions_status ON pipeline_versions(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_versions_version ON pipeline_versions(version);



CREATE INDEX IF NOT EXISTS idx_workflow_versions_workflow_id ON workflow_versions(workflow_id);

-- Pipeline execution indexes
CREATE INDEX IF NOT EXISTS idx_pipelines_app_id ON pipelines(app_id);
CREATE INDEX IF NOT EXISTS idx_pipelines_slug ON pipelines(app_id, pipeline_slug);
CREATE INDEX IF NOT EXISTS idx_pipelines_file_path ON pipelines(file_path);
CREATE INDEX IF NOT EXISTS idx_pipelines_active ON pipelines(is_active) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_pipelines_created_by ON pipelines(created_by);

CREATE INDEX IF NOT EXISTS idx_pipeline_executions_pipeline_id ON pipeline_executions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status ON pipeline_executions(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_started_at ON pipeline_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_created_by ON pipeline_executions(created_by);
-- HITL support index
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status_waiting ON pipeline_executions(status, waiting_step_id) WHERE status = 'paused_for_input';

CREATE INDEX IF NOT EXISTS idx_pipeline_step_executions_execution_id ON pipeline_step_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_step_executions_step_id ON pipeline_step_executions(step_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_step_executions_status ON pipeline_step_executions(status);

-- Insert default permissions
INSERT INTO permissions (name, resource_type, action, description) VALUES ('organizations.create', 'organization', 'create', 'Create new organizations') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('organizations.read', 'organization', 'read', 'View organization details') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('organizations.update', 'organization', 'update', 'Update organization settings') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('organizations.delete', 'organization', 'delete', 'Delete organizations') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('organizations.manage', 'organization', 'manage', 'Full organization management') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('users.invite', 'user', 'create', 'Invite new users to organization') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('users.read', 'user', 'read', 'View user profiles and lists') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('users.update', 'user', 'update', 'Update user roles and permissions') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('users.remove', 'user', 'delete', 'Remove users from organization') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('users.manage', 'user', 'manage', 'Full user management') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('teams.create', 'team', 'create', 'Create new teams') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('teams.read', 'team', 'read', 'View team details and members') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('teams.update', 'team', 'update', 'Update team settings and membership') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('teams.delete', 'team', 'delete', 'Delete teams') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('teams.manage', 'team', 'manage', 'Full team management') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('apps.create', 'app', 'create', 'Create new applications') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('apps.read', 'app', 'read', 'View applications and their details') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('apps.update', 'app', 'update', 'Update application settings and code') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('apps.delete', 'app', 'delete', 'Delete applications') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('apps.execute', 'app', 'execute', 'Execute and use applications') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('apps.manage', 'app', 'manage', 'Full application management') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('agents.create', 'agent', 'create', 'Create new agents') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('agents.read', 'agent', 'read', 'View agents and their configurations') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('agents.update', 'agent', 'update', 'Update agent settings and code') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('agents.delete', 'agent', 'delete', 'Delete agents') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('agents.execute', 'agent', 'execute', 'Execute agents') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('agents.manage', 'agent', 'manage', 'Full agent management') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('functions.create', 'function', 'create', 'Create new functions') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('functions.read', 'function', 'read', 'View functions and their code') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('functions.update', 'function', 'update', 'Update function code and settings') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('functions.delete', 'function', 'delete', 'Delete functions') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('functions.execute', 'function', 'execute', 'Execute functions') ON CONFLICT (name) DO NOTHING;
INSERT INTO permissions (name, resource_type, action, description) VALUES ('functions.manage', 'function', 'manage', 'Full function management') ON CONFLICT (name) DO NOTHING;

-- Insert default role permissions
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('owner', (SELECT id FROM permissions WHERE name = 'organizations.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('owner', (SELECT id FROM permissions WHERE name = 'users.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('owner', (SELECT id FROM permissions WHERE name = 'teams.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('owner', (SELECT id FROM permissions WHERE name = 'apps.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('owner', (SELECT id FROM permissions WHERE name = 'agents.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('owner', (SELECT id FROM permissions WHERE name = 'functions.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'organizations.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'organizations.update'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'users.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'teams.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'apps.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'agents.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('admin', (SELECT id FROM permissions WHERE name = 'functions.manage'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'organizations.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'users.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'teams.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'apps.create'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'apps.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'apps.update'), 'team') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'apps.execute'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'agents.create'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'agents.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'agents.update'), 'team') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'agents.execute'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'functions.create'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'functions.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'functions.update'), 'team') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('member', (SELECT id FROM permissions WHERE name = 'functions.execute'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'organizations.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'users.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'teams.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'apps.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'apps.execute'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'agents.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'agents.execute'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'functions.read'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;
INSERT INTO role_permissions (role, permission_id, resource_scope) VALUES ('viewer', (SELECT id FROM permissions WHERE name = 'functions.execute'), 'organization') ON CONFLICT (role, permission_id, resource_scope) DO NOTHING;

-- Insert default subscription tiers
INSERT INTO subscription_tiers (tier_name, display_name, description, monthly_price_cents, annual_price_cents, max_users, max_apps, max_storage_gb, max_monthly_executions, max_concurrent_executions, max_execution_duration_seconds, includes_priority_support, includes_sso, includes_advanced_analytics, includes_compliance_tools, includes_white_labeling, includes_dedicated_support, includes_custom_integrations, rate_limit_per_minute, features) VALUES ('free', 'Free', 'Perfect for getting started with basic AI automation', 0, 0, 3, 5, 1, 1000, 2, 60, false, false, false, false, false, false, false, 10, '["basic_analytics", "community_support"]') ON CONFLICT (tier_name) DO NOTHING;
INSERT INTO subscription_tiers (tier_name, display_name, description, monthly_price_cents, annual_price_cents, max_users, max_apps, max_storage_gb, max_monthly_executions, max_concurrent_executions, max_execution_duration_seconds, includes_priority_support, includes_sso, includes_advanced_analytics, includes_compliance_tools, includes_white_labeling, includes_dedicated_support, includes_custom_integrations, rate_limit_per_minute, features) VALUES ('starter', 'Starter', 'For small teams building AI-powered applications', 2900, 29000, 10, 25, 10, 10000, 5, 300, false, false, true, false, false, false, false, 60, '["basic_analytics", "email_support", "api_access"]') ON CONFLICT (tier_name) DO NOTHING;
INSERT INTO subscription_tiers (tier_name, display_name, description, monthly_price_cents, annual_price_cents, max_users, max_apps, max_storage_gb, max_monthly_executions, max_concurrent_executions, max_execution_duration_seconds, includes_priority_support, includes_sso, includes_advanced_analytics, includes_compliance_tools, includes_white_labeling, includes_dedicated_support, includes_custom_integrations, rate_limit_per_minute, features) VALUES ('pro', 'Professional', 'For growing teams with advanced automation needs', 9900, 99000, 50, 100, 100, 100000, 25, 1800, true, true, true, true, false, false, true, 300, '["advanced_analytics", "priority_support", "api_access", "custom_integrations", "team_management"]') ON CONFLICT (tier_name) DO NOTHING;
INSERT INTO subscription_tiers (tier_name, display_name, description, monthly_price_cents, annual_price_cents, max_users, max_apps, max_storage_gb, max_monthly_executions, max_concurrent_executions, max_execution_duration_seconds, includes_priority_support, includes_sso, includes_advanced_analytics, includes_compliance_tools, includes_white_labeling, includes_dedicated_support, includes_custom_integrations, rate_limit_per_minute, features) VALUES ('enterprise', 'Enterprise', 'For large organizations requiring enterprise features', 29900, 299000, -1, -1, -1, -1, -1, -1, true, true, true, true, true, true, true, 1000, '["advanced_analytics", "dedicated_support", "api_access", "custom_integrations", "team_management", "white_labeling", "compliance_tools", "custom_domains", "priority_execution", "advanced_security"]') ON CONFLICT (tier_name) DO NOTHING;

-- Insert default executor types
INSERT INTO executor_types (type_id, name, description) VALUES ('agent', 'Agent', 'Agent execution keys') ON CONFLICT (type_id) DO NOTHING;
INSERT INTO executor_types (type_id, name, description) VALUES ('function', 'Function', 'Function execution keys') ON CONFLICT (type_id) DO NOTHING;
INSERT INTO executor_types (type_id, name, description) VALUES ('pipeline', 'Pipeline', 'Pipeline execution keys') ON CONFLICT (type_id) DO NOTHING;
INSERT INTO executor_types (type_id, name, description) VALUES ('workflow', 'Workflow', 'Workflow execution keys') ON CONFLICT (type_id) DO NOTHING;

-- Insert default role definitions
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('system_admin', 'System Administrator', 'Full system access across all instances and organizations', 'global', '["admin:all", "read:all", "write:all", "delete:all", "manage:users", "manage:organizations", "manage:billing", "manage:system"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('instance_admin', 'Instance Administrator', 'Administrator for this FiberWise instance', 'global', '["admin:instance", "read:all", "write:all", "manage:users", "manage:organizations", "view:billing", "manage:settings"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('org_owner', 'Organization Owner', 'Full control over the organization', 'organization', '["admin:organization", "manage:members", "manage:teams", "manage:billing", "manage:settings", "read:all", "write:all", "delete:all"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('org_admin', 'Organization Administrator', 'Administrative access within the organization', 'organization', '["admin:organization", "manage:members", "manage:teams", "read:all", "write:all", "view:billing"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('billing_admin', 'Billing Administrator', 'Manages billing, invoices, and usage for the organization', 'organization', '["admin:billing", "read:billing", "write:billing", "manage:invoices", "view:usage", "manage:quotas"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('developer', 'Developer', 'Can create and manage apps, agents, and functions', 'organization', '["read:apps", "write:apps", "read:agents", "write:agents", "read:functions", "write:functions", "execute:all"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('analyst', 'Analyst', 'Can view data and run analysis but not modify resources', 'organization', '["read:all", "execute:agents", "execute:functions", "view:analytics", "export:data"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('member', 'Member', 'Basic organization member with read access', 'organization', '["read:apps", "read:agents", "read:functions", "execute:allowed"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('viewer', 'Viewer', 'Read-only access to organization resources', 'organization', '["read:apps", "read:agents", "read:functions", "view:analytics"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('team_lead', 'Team Lead', 'Leads a team within an organization', 'team', '["manage:team", "read:all", "write:all", "execute:all"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('team_member', 'Team Member', 'Standard team member', 'team', '["read:team", "write:team", "execute:team"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('app_admin', 'App Administrator', 'Full control over specific apps', 'resource', '["admin:app", "read:app", "write:app", "delete:app", "manage:app_users"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('agent_admin', 'Agent Administrator', 'Full control over specific agents', 'resource', '["admin:agent", "read:agent", "write:agent", "delete:agent", "execute:agent"]', true) ON CONFLICT (role_name) DO NOTHING;
INSERT INTO role_definitions (role_name, display_name, description, role_type, permissions, is_system_role) VALUES ('function_admin', 'Function Administrator', 'Full control over specific functions', 'resource', '["admin:function", "read:function", "write:function", "delete:function", "execute:function"]', true) ON CONFLICT (role_name) DO NOTHING;

-- Set the first user as system admin when they complete setup
-- This will be handled by the CLI setup command
