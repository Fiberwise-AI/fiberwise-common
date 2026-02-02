-- Migration: Add system_apps and system_app_deployments tables
-- This creates tables for managing system app source code catalog and per-organization deployments

CREATE TABLE IF NOT EXISTS system_apps (
    id SERIAL PRIMARY KEY,
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

-- Comments for system_apps documentation
COMMENT ON TABLE system_apps IS 'Catalog of system app source code available on disk for deployment';
COMMENT ON COLUMN system_apps.app_slug IS 'Unique identifier slug for the system app';
COMMENT ON COLUMN system_apps.name IS 'Human-readable name of the system app';
COMMENT ON COLUMN system_apps.source_path IS 'File system path to the app source code';
COMMENT ON COLUMN system_apps.source_version IS 'Version of the source code';
COMMENT ON COLUMN system_apps.current_manifest_hash IS 'Hash of the current manifest file for change detection';
COMMENT ON COLUMN system_apps.current_models_hash IS 'Hash of the current models for change detection';
COMMENT ON COLUMN system_apps.category IS 'Category or type of the system app';
COMMENT ON COLUMN system_apps.icon_class IS 'CSS class for app icon display';
COMMENT ON COLUMN system_apps.publisher IS 'Publisher or author of the system app';
COMMENT ON COLUMN system_apps.last_scanned_at IS 'Timestamp of last filesystem scan for this app';


CREATE TABLE IF NOT EXISTS system_app_deployments (
    id SERIAL PRIMARY KEY,
    system_app_id INTEGER NOT NULL REFERENCES system_apps(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    deployed_app_id TEXT NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    deployed_version_id TEXT,
    deployed_version TEXT,
    deployed_manifest_hash TEXT,
    deployed_models_hash TEXT,
    deploy_status TEXT NOT NULL DEFAULT 'deploying',
    last_deploy_error TEXT,
    deploy_state_json JSONB DEFAULT '{}',
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

-- Comments for system_app_deployments documentation
COMMENT ON TABLE system_app_deployments IS 'Per-organization deployment tracking for system apps with version and status management';
COMMENT ON COLUMN system_app_deployments.system_app_id IS 'Reference to the system app being deployed';
COMMENT ON COLUMN system_app_deployments.organization_id IS 'Organization where the app is deployed';
COMMENT ON COLUMN system_app_deployments.deployed_app_id IS 'Reference to the deployed app instance';
COMMENT ON COLUMN system_app_deployments.deployed_version_id IS 'Version ID of the deployed app';
COMMENT ON COLUMN system_app_deployments.deployed_version IS 'Human-readable version of the deployed app';
COMMENT ON COLUMN system_app_deployments.deployed_manifest_hash IS 'Hash of the manifest at deployment time';
COMMENT ON COLUMN system_app_deployments.deployed_models_hash IS 'Hash of the models at deployment time';
COMMENT ON COLUMN system_app_deployments.deploy_status IS 'Current deployment status: deploying, deployed, failed, update_available';
COMMENT ON COLUMN system_app_deployments.last_deploy_error IS 'Last error message encountered during deployment';
COMMENT ON COLUMN system_app_deployments.deploy_state_json IS 'JSON state data for deployment process and tracking';
COMMENT ON COLUMN system_app_deployments.has_update_available IS 'Flag indicating if a newer version of the app is available';
COMMENT ON COLUMN system_app_deployments.models_migration_needed IS 'Flag indicating if model migration is required';
COMMENT ON COLUMN system_app_deployments.last_updated_at IS 'Timestamp of the last status update for this deployment';
