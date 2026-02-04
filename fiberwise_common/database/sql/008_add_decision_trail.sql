-- Decision trail for execution audit
CREATE TABLE IF NOT EXISTS decision_trail_nodes (
    node_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    step_id TEXT,
    decision_type TEXT,
    decision TEXT,
    rationale TEXT,
    confidence REAL DEFAULT 1.0,
    evidence TEXT DEFAULT '[]',
    alternatives TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decision_trail_edges (
    id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    label TEXT,
    condition TEXT,
    weight REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_dt_nodes_execution ON decision_trail_nodes(execution_id);
CREATE INDEX IF NOT EXISTS idx_dt_edges_execution ON decision_trail_edges(execution_id);
