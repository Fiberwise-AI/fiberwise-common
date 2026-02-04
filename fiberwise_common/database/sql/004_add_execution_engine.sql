-- Add execution engine support to pipelines
-- Allows pipelines to use different execution engines (fiber-default, ia_modules, etc.)

ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS execution_engine TEXT DEFAULT 'fiber-default';
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS pipeline_definition_file TEXT;

ALTER TABLE pipeline_versions ADD COLUMN IF NOT EXISTS execution_engine TEXT DEFAULT 'fiber-default';

CREATE INDEX IF NOT EXISTS idx_pipelines_execution_engine ON pipelines(execution_engine);
