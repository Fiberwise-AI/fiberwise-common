-- Migration: Add OIDC identity linking table
-- Supports generic OIDC providers (Keycloak, Authentik, Cognito, Azure AD, etc.)

CREATE TABLE IF NOT EXISTS oidc_identities (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    issuer TEXT NOT NULL,
    subject TEXT NOT NULL,
    email TEXT,
    last_login_at TEXT DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(issuer, subject)
);

CREATE INDEX IF NOT EXISTS idx_oidc_identities_user ON oidc_identities(user_id);
CREATE INDEX IF NOT EXISTS idx_oidc_identities_lookup ON oidc_identities(issuer, subject);
