-- Scheduled Jobs for pipeline cron/interval execution
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id TEXT PRIMARY KEY,
    job_name TEXT NOT NULL,
    pipeline_id TEXT NOT NULL REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    app_id TEXT REFERENCES apps(app_id) ON DELETE SET NULL,
    cron_expression TEXT,
    interval_seconds INTEGER,
    input_data TEXT DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT TRUE,
    last_run_at TEXT,
    next_run_at TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scheduled_job_executions (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE,
    execution_id TEXT,
    triggered_by TEXT DEFAULT 'scheduler',
    status TEXT DEFAULT 'pending',
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_pipeline ON scheduled_jobs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_enabled ON scheduled_jobs(is_enabled);
CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_run ON scheduled_jobs(next_run_at);
CREATE INDEX IF NOT EXISTS idx_sched_job_exec_job ON scheduled_job_executions(job_id);
