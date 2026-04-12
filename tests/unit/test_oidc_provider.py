"""Unit tests for MiniOIDCProvider — actual key generation, JWKS, and token signing."""

import pytest
from jose import jwt

from fiberwise_common.oidc_provider.provider import MiniOIDCProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """Create a MiniOIDCProvider using a temp directory for keys."""
    monkeypatch.setenv("MINI_OIDC_KEY_DIR", str(tmp_path / "oidc_keys"))
    return MiniOIDCProvider(issuer="http://localhost:5555/oidc")


class TestMiniOIDCProviderInit:

    def test_generates_keys(self, provider, tmp_path):
        key_dir = tmp_path / "oidc_keys"
        assert (key_dir / "private.pem").exists()
        assert (key_dir / "public.pem").exists()

    def test_loads_existing_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINI_OIDC_KEY_DIR", str(tmp_path / "oidc_keys"))
        p1 = MiniOIDCProvider(issuer="http://localhost:5555/oidc")
        p2 = MiniOIDCProvider(issuer="http://localhost:5555/oidc")
        assert p1._kid == p2._kid

    def test_kid_is_set(self, provider):
        assert provider._kid is not None
        assert len(provider._kid) == 16


class TestDiscoveryAndJWKS:

    def test_discovery_document(self, provider):
        doc = provider.get_discovery()
        assert doc["issuer"] == "http://localhost:5555/oidc"
        assert doc["jwks_uri"] == "http://localhost:5555/oidc/jwks"
        assert "client_credentials" in doc["grant_types_supported"]

    def test_jwks_has_key(self, provider):
        jwks = provider.get_jwks()
        assert len(jwks["keys"]) == 1
        key = jwks["keys"][0]
        assert key["kty"] == "RSA"
        assert key["alg"] == "RS256"
        assert key["kid"] == provider._kid
        assert "n" in key
        assert "e" in key


class TestTokenIssuance:

    def test_issue_token_returns_jwt(self, provider):
        result = provider.issue_token(
            subject="agent-123",
            audience="a2a-server",
            scopes=["data:read"],
        )
        assert "access_token" in result
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] > 0

    def test_token_is_valid_jwt(self, provider):
        result = provider.issue_token(
            subject="agent-123",
            audience="a2a-server",
            scopes=["data:read", "data:write"],
            claims={"org_id": 1, "app_id": "app-abc"},
        )
        # Decode without verification to inspect claims
        payload = jwt.get_unverified_claims(result["access_token"])
        assert payload["sub"] == "agent-123"
        assert payload["aud"] == "a2a-server"
        assert payload["iss"] == "http://localhost:5555/oidc"
        assert payload["scope"] == "data:read data:write"
        assert payload["org_id"] == 1
        assert payload["app_id"] == "app-abc"

    def test_token_has_kid_header(self, provider):
        result = provider.issue_token(
            subject="agent-123",
            audience="a2a-server",
            scopes=[],
        )
        headers = jwt.get_unverified_header(result["access_token"])
        assert headers["kid"] == provider._kid
        assert headers["alg"] == "RS256"
