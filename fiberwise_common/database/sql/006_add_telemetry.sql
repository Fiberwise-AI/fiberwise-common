-- Lightweight execution metrics for historical queries.
-- Real-time metrics use OpenTelemetry/Prometheus exporters from ia_modules.
CREATE TABLE IF NOT EXISTS execution_metrics (
    id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    pipeline_id TEXT,
    step_id TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT DEFAULT 'gauge',
    labels TEXT DEFAULT '{}',
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_exec_metrics_execution ON execution_metrics(execution_id);
CREATE INDEX IF NOT EXISTS idx_exec_metrics_pipeline ON execution_metrics(pipeline_id);
