-- Reliability metrics and alerts
CREATE TABLE IF NOT EXISTS reliability_metrics (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    window_start TEXT,
    window_end TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    pipeline_id TEXT,
    alert_type TEXT NOT NULL,
    severity TEXT DEFAULT 'warning',
    message TEXT NOT NULL,
    context TEXT DEFAULT '{}',
    is_resolved BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_reliability_pipeline ON reliability_metrics(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_alerts_pipeline ON alerts(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON alerts(is_resolved);
