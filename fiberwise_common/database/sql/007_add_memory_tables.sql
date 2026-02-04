-- Conversation memory for pipeline/agent sessions
CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    thread_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    app_id TEXT REFERENCES apps(app_id) ON DELETE SET NULL,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conv_messages_session ON conversation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_messages_thread ON conversation_messages(thread_id);
