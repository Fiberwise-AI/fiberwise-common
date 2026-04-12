"""Unit tests for OIDC adapter layer — MiniOIDCAdapter and KeycloakAdapter."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fiberwise_common.oidc_provider.adapter import IDPAdapter, ClientCredentials
from fiberwise_common.oidc_provider.mini_oidc_adapter import MiniOIDCAdapter
from fiberwise_common.oidc_provider.defaults import get_default_permissions, DEFAULT_A2A_PERMISSIONS


# ---------------------------------------------------------------------------
# Mock DB helpers
# ---------------------------------------------------------------------------

class MockRow:
    """Dict-like row that supports dict() conversion."""
    def __init__(self, data):
        self._data = data
    def __getitem__(self, key):
        return self._data[key]
    def get(self, key, default=None):
        return self._data.get(key, default)
    def keys(self):
        return self._data.keys()
    def values(self):
        return self._data.values()
    def items(self):
        return self._data.items()
    def __iter__(self):
        return iter(self._data)
    def __len__(self):
        return len(self._data)


def make_mock_db(fetch_one_return=None):
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetch_one = AsyncMock(return_value=fetch_one_return)
    return db


# ---------------------------------------------------------------------------
# IDPAdapter contract tests
# ---------------------------------------------------------------------------

class TestIDPAdapterContract:
    """Verify both adapters implement the abstract interface."""

    def test_mini_oidc_adapter_is_idp_adapter(self):
        adapter = MiniOIDCAdapter(db=None)
        assert isinstance(adapter, IDPAdapter)

    def test_keycloak_adapter_requires_env_vars(self):
        with patch.dict(os.environ, {"IDP_ADMIN_URL": "", "IDP_TOKEN_URL": ""}):
            # Need to reimport to pick up empty env vars
            import importlib
            import fiberwise_common.oidc_provider.keycloak_adapter as kc_mod
            importlib.reload(kc_mod)
            with pytest.raises(ValueError, match="requires IDP_ADMIN_URL"):
                kc_mod.KeycloakAdapter(db=None)


# ---------------------------------------------------------------------------
# MiniOIDCAdapter tests
# ---------------------------------------------------------------------------

class TestMiniOIDCAdapterRegisterClient:

    @pytest.mark.asyncio
    async def test_register_client_returns_credentials(self):
        db = make_mock_db()
        adapter = MiniOIDCAdapter(db=db)

        creds = await adapter.register_client("agent-123", org_id=1, permissions={"a": 1})

        assert isinstance(creds, ClientCredentials)
        assert creds.client_id.startswith("agent-agent-123-")
        assert len(creds.client_secret) > 30
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_client_stores_hash_not_plaintext(self):
        db = make_mock_db()
        adapter = MiniOIDCAdapter(db=db)

        creds = await adapter.register_client("agent-abc", org_id=1, permissions={})

        call_args = db.execute.call_args
        params = call_args[0][1]
        # secret_hash should be a SHA256 hex digest (64 chars)
        assert len(params["secret_hash"]) == 64
        # The plaintext secret should NOT be stored
        assert params["secret_hash"] != creds.client_secret

    @pytest.mark.asyncio
    async def test_register_client_no_db(self):
        adapter = MiniOIDCAdapter(db=None)
        creds = await adapter.register_client("agent-no-db", org_id=1, permissions={})
        assert creds.client_id.startswith("agent-agent-no-db-")


class TestMiniOIDCAdapterGetToken:

    @pytest.mark.asyncio
    async def test_get_token_valid_credentials(self):
        import hashlib
        secret = "test-secret-value"

        row = MockRow({
            "agent_id": "agent-xyz",
            "organization_id": 1,
            "app_id": "app-1",
            "scopes": '["data:read"]',
            "a2a_permissions": '{"allowed_modes": ["research"]}',
        })
        db = make_mock_db(fetch_one_return=row)
        adapter = MiniOIDCAdapter(db=db)

        # Mock the provider so we don't need cryptography installed
        mock_provider = MagicMock()
        mock_provider.issue_token.return_value = {"access_token": "mock-jwt-token-xyz", "token_type": "Bearer", "expires_in": 300}
        adapter._provider = mock_provider

        token = await adapter.get_token("client-id", secret)
        assert token == "mock-jwt-token-xyz"
        mock_provider.issue_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_token_invalid_credentials_raises(self):
        db = make_mock_db(fetch_one_return=None)
        adapter = MiniOIDCAdapter(db=db)

        with pytest.raises(ValueError, match="Invalid client credentials"):
            await adapter.get_token("bad-client", "bad-secret")


class TestMiniOIDCAdapterIssueTokenForAgent:

    @pytest.mark.asyncio
    async def test_issue_token_for_agent_success(self):
        row = MockRow({
            "agent_id": "agent-xyz",
            "organization_id": 1,
            "app_id": "app-1",
            "scopes": '[]',
            "a2a_permissions": '{"allowed_modes": ["research"]}',
        })
        db = make_mock_db(fetch_one_return=row)
        adapter = MiniOIDCAdapter(db=db)

        # Mock the provider so we don't need cryptography installed
        mock_provider = MagicMock()
        mock_provider.issue_token.return_value = {"access_token": "mock-jwt-for-agent", "token_type": "Bearer", "expires_in": 300}
        adapter._provider = mock_provider

        token = await adapter.issue_token_for_agent("agent-xyz")
        assert token == "mock-jwt-for-agent"
        mock_provider.issue_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_issue_token_for_agent_no_key_raises(self):
        db = make_mock_db(fetch_one_return=None)
        adapter = MiniOIDCAdapter(db=db)

        with pytest.raises(ValueError, match="No IDP-registered key found"):
            await adapter.issue_token_for_agent("nonexistent")


class TestMiniOIDCAdapterValidateToken:
    """validate_token — real issue → validate round-trip (no mocks on provider)."""

    @pytest.fixture
    def real_adapter(self, tmp_path, monkeypatch):
        """Adapter backed by a real MiniOIDCProvider with temp RSA keys."""
        monkeypatch.setenv("MINI_OIDC_KEY_DIR", str(tmp_path / "oidc_keys"))
        return MiniOIDCAdapter(db=None)

    @pytest.mark.asyncio
    async def test_validate_token_round_trip(self, real_adapter):
        """Issue a token via provider, validate it, check all claims."""
        provider = real_adapter._get_provider()
        result = provider.issue_token(
            subject="agent_test-agent",
            audience="a2a-server",
            scopes=["data:read", "data:write"],
            claims={"org_id": 1, "app_id": "app-abc", "agent_id": "test-agent", "a2a": {"allowed_modes": ["research"]}},
        )
        token = result["access_token"]

        claims = await real_adapter.validate_token(token, audience="a2a-server")

        assert claims["sub"] == "agent_test-agent"
        assert claims["aud"] == "a2a-server"
        assert claims["iss"] == provider.issuer
        assert claims["org_id"] == 1
        assert claims["app_id"] == "app-abc"
        assert claims["agent_id"] == "test-agent"
        assert claims["a2a"] == {"allowed_modes": ["research"]}
        assert "data:read" in claims["scope"]
        assert "data:write" in claims["scope"]

    @pytest.mark.asyncio
    async def test_validate_token_wrong_audience_raises(self, real_adapter):
        """Token issued for one audience must fail validation for another."""
        provider = real_adapter._get_provider()
        result = provider.issue_token(
            subject="agent_x", audience="a2a-server", scopes=[],
        )

        with pytest.raises(ValueError, match="Token validation failed"):
            await real_adapter.validate_token(result["access_token"], audience="wrong-audience")

    @pytest.mark.asyncio
    async def test_validate_token_tampered_raises(self, real_adapter):
        """A tampered token must fail signature verification."""
        provider = real_adapter._get_provider()
        result = provider.issue_token(
            subject="agent_x", audience="a2a-server", scopes=[],
        )
        # Flip a character in the middle of the token
        token = result["access_token"]
        mid = len(token) // 2
        bad_char = "A" if token[mid] != "A" else "B"
        tampered = token[:mid] + bad_char + token[mid + 1:]

        with pytest.raises(ValueError, match="Token validation failed"):
            await real_adapter.validate_token(tampered, audience="a2a-server")

    @pytest.mark.asyncio
    async def test_validate_token_expired_raises(self, real_adapter, monkeypatch):
        """An expired token must fail validation."""
        import time
        provider = real_adapter._get_provider()

        # Issue token with TTL=0 by patching time to be in the past
        from jose import jwt as jose_jwt
        from cryptography.hazmat.primitives import serialization

        now = int(time.time())
        payload = {
            "iss": provider.issuer,
            "sub": "agent_expired",
            "aud": "a2a-server",
            "iat": now - 600,
            "exp": now - 300,  # expired 5 minutes ago
            "scope": "",
        }
        priv_pem = provider._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        expired_token = jose_jwt.encode(payload, priv_pem, algorithm="RS256", headers={"kid": provider._kid})

        with pytest.raises(ValueError, match="Token validation failed"):
            await real_adapter.validate_token(expired_token, audience="a2a-server")

    @pytest.mark.asyncio
    async def test_validate_token_different_key_raises(self, tmp_path):
        """Token signed by a different key pair must fail."""
        from fiberwise_common.oidc_provider.provider import MiniOIDCProvider

        # Create two providers with separate key dirs directly
        provider_a = MiniOIDCProvider.__new__(MiniOIDCProvider)
        provider_a.issuer = "http://localhost:5555/oidc"
        provider_a._key_dir = tmp_path / "keys_a"
        provider_a._private_key = None
        provider_a._public_key = None
        provider_a._kid = None
        provider_a._jwks_json = None
        provider_a._load_or_generate_keys()

        provider_b = MiniOIDCProvider.__new__(MiniOIDCProvider)
        provider_b.issuer = "http://localhost:5555/oidc"
        provider_b._key_dir = tmp_path / "keys_b"
        provider_b._private_key = None
        provider_b._public_key = None
        provider_b._kid = None
        provider_b._jwks_json = None
        provider_b._load_or_generate_keys()

        # Issue with provider_a's keys
        result = provider_a.issue_token(
            subject="agent_x", audience="a2a-server", scopes=[],
        )

        # Validate with adapter_b using provider_b's keys — should fail
        adapter_b = MiniOIDCAdapter(db=None)
        adapter_b._provider = provider_b

        with pytest.raises(ValueError, match="Token validation failed"):
            await adapter_b.validate_token(result["access_token"], audience="a2a-server")


class TestMiniOIDCAdapterUpdateAndDelete:

    @pytest.mark.asyncio
    async def test_update_client_claims(self):
        db = make_mock_db()
        adapter = MiniOIDCAdapter(db=db)

        await adapter.update_client_claims("client-id", {"allowed_modes": ["execute"]})
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_client(self):
        db = make_mock_db()
        adapter = MiniOIDCAdapter(db=db)

        await adapter.delete_client("client-id")
        db.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Default permissions
# ---------------------------------------------------------------------------

class TestDefaultPermissions:

    def test_llm_defaults(self):
        perms = get_default_permissions("llm")
        assert perms["allowed_modes"] == ["research"]
        assert "Read" in perms["allowed_tools"]

    def test_processor_defaults(self):
        perms = get_default_permissions("processor")
        assert "plan" in perms["allowed_modes"]

    def test_custom_defaults(self):
        perms = get_default_permissions("custom")
        assert "execute" in perms["allowed_modes"]
        assert "Bash" in perms["allowed_tools"]

    def test_unknown_type_falls_back_to_llm(self):
        perms = get_default_permissions("unknown_type_xyz")
        assert perms == DEFAULT_A2A_PERMISSIONS["llm"]

    def test_returns_copy_not_reference(self):
        perms1 = get_default_permissions("llm")
        perms2 = get_default_permissions("llm")
        perms1["allowed_modes"].append("MUTATED")
        assert "MUTATED" not in perms2["allowed_modes"]


# ---------------------------------------------------------------------------
# get_adapter factory
# ---------------------------------------------------------------------------

class TestGetAdapterFactory:

    def test_default_returns_mini_oidc(self):
        with patch.dict(os.environ, {"AGENT_AUTH_MODE": "local"}):
            from fiberwise_common.oidc_provider import get_adapter
            adapter = get_adapter(db=None)
            assert isinstance(adapter, MiniOIDCAdapter)

    def test_oidc_mode_returns_keycloak(self):
        with patch.dict(os.environ, {
            "AGENT_AUTH_MODE": "oidc",
            "IDP_ADMIN_URL": "http://localhost:8080/admin/realms/test",
            "IDP_TOKEN_URL": "http://localhost:8080/realms/test/protocol/openid-connect/token",
        }):
            import importlib
            # Reload the ia_modules source (where env vars are read at module level)
            import ia_modules.agents.auth.keycloak_adapter as kc_src
            importlib.reload(kc_src)
            import ia_modules.agents.auth as auth_pkg
            importlib.reload(auth_pkg)
            # Reload the fiberwise re-export shims
            import fiberwise_common.oidc_provider.keycloak_adapter as kc_mod
            importlib.reload(kc_mod)
            import fiberwise_common.oidc_provider as oidc_pkg
            importlib.reload(oidc_pkg)
            adapter = oidc_pkg.get_adapter(db=MagicMock())
            assert isinstance(adapter, kc_src.KeycloakAdapter)
