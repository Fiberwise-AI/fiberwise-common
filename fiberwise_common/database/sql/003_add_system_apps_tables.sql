-- Migration: Add system_apps and system_app_deployments tables
-- This creates tables for managing system app source code catalog and per-organization deployments

CREATE TABLE IF NOT EXISTS system_apps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    source_path TEXT NOT NULL,
    source_version TEXT,
    current_manifest_hash TEXT,
    current_models_hash TEXT,
    category TEXT,
    icon_class TEXT,
    publisher TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_scanned_at TIMESTAMP
);

-- Indexes for system_apps
CREATE INDEX IF NOT EXISTS idx_system_apps_slug ON system_apps(app_slug);

-- Comments (PostgreSQL COMMENT ON not supported in SQLite)


CREATE TABLE IF NOT EXISTS system_app_deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_app_id INTEGER NOT NULL REFERENCES system_apps(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    deployed_app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    deployed_version_id TEXT,
    deployed_version TEXT,
    deployed_manifest_hash TEXT,
    deployed_models_hash TEXT,
    deploy_status TEXT NOT NULL DEFAULT 'deploying',
    last_deploy_error TEXT,
    deploy_state_json TEXT DEFAULT '{}',
    has_update_available BOOLEAN DEFAULT false,
    models_migration_needed BOOLEAN DEFAULT false,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP,
    UNIQUE(system_app_id, organization_id)
);

-- Indexes for system_app_deployments
CREATE INDEX IF NOT EXISTS idx_system_app_deployments_org ON system_app_deployments(organization_id);
CREATE INDEX IF NOT EXISTS idx_system_app_deployments_app ON system_app_deployments(deployed_app_id);
CREATE INDEX IF NOT EXISTS idx_system_app_deployments_status ON system_app_deployments(deploy_status);

-- Comments (PostgreSQL COMMENT ON not supported in SQLite)
