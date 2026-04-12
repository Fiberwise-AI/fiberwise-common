-- Add IDP client columns to agent_api_keys for OIDC-backed agent auth.
-- idp_client_id: the client_id registered with the IDP (or mini OIDC)
-- idp_client_secret_hash: SHA256 hash of client_secret (mini OIDC validates locally)
-- a2a_permissions: JSON dict of a2a permissions baked into JWT tokens

ALTER TABLE agent_api_keys ADD COLUMN idp_client_id TEXT;
ALTER TABLE agent_api_keys ADD COLUMN idp_client_secret_hash TEXT;
ALTER TABLE agent_api_keys ADD COLUMN a2a_permissions TEXT DEFAULT '{}';
