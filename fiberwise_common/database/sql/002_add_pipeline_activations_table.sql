-- Migration: Add pipeline_activations table for unified execution system
-- This creates a parallel activation system for pipelines similar to agent_activations

CREATE TABLE IF NOT EXISTS pipeline_activations (
    activation_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    pipeline_slug VARCHAR(255),
    input_data TEXT,
    output_data TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_by INTEGER NOT NULL,
    app_id TEXT,
    context TEXT DEFAULT '{}',
    priority INTEGER DEFAULT 10,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (pipeline_id) REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    FOREIGN KEY (app_id) REFERENCES apps(app_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_status ON pipeline_activations(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_created_by ON pipeline_activations(created_by);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_pipeline_id ON pipeline_activations(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_app_id ON pipeline_activations(app_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_activations_priority_created ON pipeline_activations(priority DESC, created_at ASC);

-- Comments (PostgreSQL COMMENT ON not supported in SQLite)