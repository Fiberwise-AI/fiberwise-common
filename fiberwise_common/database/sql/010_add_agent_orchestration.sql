-- Multi-agent orchestration workflows
CREATE TABLE IF NOT EXISTS agent_workflows (
    workflow_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    workflow_config TEXT NOT NULL,
    app_id TEXT REFERENCES apps(app_id) ON DELETE SET NULL,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agent_roles (
    role_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT,
    allowed_tools TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agent_workflow_executions (
    execution_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES agent_workflows(workflow_id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending',
    input_data TEXT DEFAULT '{}',
    output_data TEXT,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_workflows_app ON agent_workflows(app_id);
CREATE INDEX IF NOT EXISTS idx_agent_wf_exec_workflow ON agent_workflow_executions(workflow_id);
