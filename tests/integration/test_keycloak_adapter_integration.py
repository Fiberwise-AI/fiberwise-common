"""Integration tests for KeycloakAdapter against a real Keycloak instance.

A fresh Keycloak container is started per test session and torn down after.

Usage:
    pytest tests/integration/test_keycloak_adapter_integration.py -v
"""

import json
import os
import socket
import subprocess
import time
import pytest
import pytest_asyncio
import httpx

# ---------------------------------------------------------------------------
# Container lifecycle — session-scoped: one fresh Keycloak per test run
# ---------------------------------------------------------------------------

_CONTAINER_NAME = "keycloak-integration-test"
_KC_PORT = 18080
_REALM_JSON = os.path.join(os.path.dirname(__file__), "..", "fixtures", "keycloak-test-realm.json")


def _run(cmd: str, check=True, timeout=30):
    """Run a shell command."""
    return subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True, text=True, check=check, timeout=timeout,
    )


def _keycloak_ready() -> bool:
    """Check if Keycloak is responding."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect(("localhost", _KC_PORT))
        sock.close()
        return True
    except OSError:
        return False


def _wait_for_keycloak(max_wait=120):
    """Wait for Keycloak realm to be available."""
    import httpx as hx
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = hx.get(f"http://localhost:{_KC_PORT}/realms/fiberwise/.well-known/openid-configuration", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _start_keycloak():
    """Start a fresh Keycloak container via podman."""
    # Remove any leftover container
    _run(f"podman rm -f {_CONTAINER_NAME}", check=False)

    realm_path = os.path.realpath(_REALM_JSON)
    _run(
        f"podman run -d --name {_CONTAINER_NAME} "
        f"-p {_KC_PORT}:8080 "
        f"-e KEYCLOAK_ADMIN=admin "
        f"-e KEYCLOAK_ADMIN_PASSWORD=admin123 "
        f"-e KC_HEALTH_ENABLED=true "
        f"-v {realm_path}:/opt/keycloak/data/import/realm.json:ro "
        f"quay.io/keycloak/keycloak:latest start-dev --import-realm",
        timeout=120,
    )

    if not _wait_for_keycloak():
        # Grab logs for debugging
        result = _run(f"podman logs {_CONTAINER_NAME}", check=False, timeout=10)
        raise RuntimeError(f"Keycloak did not start in time.\nLogs:\n{result.stdout}\n{result.stderr}")


def _stop_keycloak():
    """Stop and remove the Keycloak container."""
    _run(f"podman rm -f {_CONTAINER_NAME}", check=False)


@pytest.fixture(scope="session", autouse=True)
def keycloak_container():
    """Session fixture: start a fresh Keycloak, tear it down when done."""
    _start_keycloak()
    yield
    _stop_keycloak()


# ---------------------------------------------------------------------------
# Env vars and adapter import — after container fixture is defined
# ---------------------------------------------------------------------------

os.environ["IDP_ADMIN_URL"] = f"http://localhost:{_KC_PORT}/admin/realms/fiberwise"
os.environ["IDP_TOKEN_URL"] = f"http://localhost:{_KC_PORT}/realms/fiberwise/protocol/openid-connect/token"
os.environ["IDP_ADMIN_CLIENT_ID"] = "fiberwise-admin"
os.environ["IDP_ADMIN_CLIENT_SECRET"] = "fiberwise_admin_secret"
os.environ["OIDC_DISCOVERY_URL"] = f"http://localhost:{_KC_PORT}/realms/fiberwise/.well-known/openid-configuration"
os.environ["A2A_AUDIENCE"] = "a2a-server"

import importlib
import fiberwise_common.oidc_provider.keycloak_adapter as kc_mod
importlib.reload(kc_mod)
from fiberwise_common.oidc_provider.keycloak_adapter import KeycloakAdapter

pytestmark = [pytest.mark.integration]


class MockDBRow:
    """Minimal mock that behaves like a DB row (supports dict() conversion)."""
    def __init__(self, data: dict):
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


class MockDB:
    """In-memory mock DB that stores agent_api_keys rows."""

    def __init__(self):
        self._rows = []

    async def execute(self, query: str, params: dict = None):
        # Handle UPDATE for register_client (sets idp_client_id)
        if "UPDATE agent_api_keys" in query and "idp_client_id" in query:
            agent_id = params.get("agent_id")
            for row in self._rows:
                if row["agent_id"] == agent_id and row["is_active"]:
                    if "idp_client_id = :client_id" in query:
                        row["idp_client_id"] = params.get("client_id")
                        row["idp_client_secret_hash"] = params.get("secret_hash", "")
                        if "a2a_permissions" in params:
                            row["a2a_permissions"] = params["permissions"]

    async def fetch_one(self, query: str, params: dict = None):
        agent_id = params.get("agent_id")
        client_id = params.get("client_id")
        for row in self._rows:
            if not row.get("is_active", True):
                continue
            if agent_id and row.get("agent_id") == agent_id:
                if "idp_client_id IS NOT NULL" in query and not row.get("idp_client_id"):
                    continue
                return MockDBRow(row)
            if client_id and row.get("idp_client_id") == client_id:
                return MockDBRow(row)
        return None

    def insert_row(self, data: dict):
        """Helper — insert a fake agent_api_keys row."""
        self._rows.append(data)


@pytest.fixture
def mock_db():
    return MockDB()


@pytest_asyncio.fixture
async def adapter():
    """Create a KeycloakAdapter (no DB — for tests that don't need issue_token_for_agent)."""
    return KeycloakAdapter(db=None)


@pytest_asyncio.fixture
async def adapter_with_db(mock_db):
    """Create a KeycloakAdapter with a mock DB."""
    return KeycloakAdapter(db=mock_db)




class TestKeycloakAdapterAdminToken:
    """Test admin token acquisition."""

    @pytest.mark.asyncio
    async def test_get_admin_token(self, adapter):
        token = await adapter._get_admin_token()
        assert token
        assert isinstance(token, str)
        assert len(token) > 50  # JWTs are long


class TestKeycloakAdapterRegisterClient:
    """Test client registration."""

    @pytest.mark.asyncio
    async def test_register_client_creates_client_with_secret(self, adapter):
        creds = await adapter.register_client(
            agent_id="test-agent-001",
            org_id=42,
            permissions={"allowed_modes": ["research"], "allowed_tools": ["Read"]},
        )


        assert creds.client_id == "agent-test-agent-001"
        assert creds.client_secret
        assert len(creds.client_secret) > 10

    @pytest.mark.asyncio
    async def test_register_client_has_protocol_mappers(self, adapter):
        creds = await adapter.register_client(
            agent_id="test-agent-002",
            org_id=7,
            permissions={"allowed_modes": ["research", "plan"]},
        )


        # Verify the client exists and has our custom mappers
        admin_token = await adapter._get_admin_token()
        kc_client = await adapter._find_client(admin_token, creds.client_id)

        mapper_names = {m["name"] for m in kc_client.get("protocolMappers", [])}
        assert "a2a-permissions" in mapper_names
        assert "a2a-org-id" in mapper_names
        assert "a2a-agent-id" in mapper_names

        # Verify attributes
        attrs = kc_client.get("attributes", {})
        assert attrs.get("a2a.agent_id") == "test-agent-002"
        assert attrs.get("a2a.org_id") == "7"

    @pytest.mark.asyncio
    async def test_register_client_service_accounts_enabled(self, adapter):
        creds = await adapter.register_client(
            agent_id="test-agent-003",
            org_id=1,
            permissions={},
        )


        admin_token = await adapter._get_admin_token()
        kc_client = await adapter._find_client(admin_token, creds.client_id)
        assert kc_client["serviceAccountsEnabled"] is True


class TestKeycloakAdapterGetToken:
    """Test client_credentials token acquisition."""

    @pytest.mark.asyncio
    async def test_get_token_returns_jwt(self, adapter):
        # Register a client first
        creds = await adapter.register_client(
            agent_id="test-agent-token-001",
            org_id=99,
            permissions={"allowed_modes": ["research"]},
        )


        # Get a token
        token = await adapter.get_token(creds.client_id, creds.client_secret)
        assert token
        assert isinstance(token, str)

        # Decode the JWT (without verification) to check claims
        import base64
        parts = token.split(".")
        assert len(parts) == 3  # header.payload.signature

        # Decode payload
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        assert payload["iss"].endswith("/realms/fiberwise")
        assert "a2a" in payload  # Our custom claim
        assert payload["a2a"] == {"allowed_modes": ["research"]}
        assert payload["agent_id"] == "test-agent-token-001"
        assert payload["org_id"] == 99

    @pytest.mark.asyncio
    async def test_get_token_invalid_credentials_raises(self, adapter):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.get_token("nonexistent-client", "bad-secret")


class TestKeycloakAdapterUpdateClaims:
    """Test updating client claims."""

    @pytest.mark.asyncio
    async def test_update_client_claims(self, adapter):
        creds = await adapter.register_client(
            agent_id="test-agent-update-001",
            org_id=10,
            permissions={"allowed_modes": ["research"]},
        )


        # Update permissions
        new_perms = {"allowed_modes": ["research", "plan", "execute"], "allowed_tools": ["Read", "Edit"]}
        await adapter.update_client_claims(creds.client_id, new_perms)

        # Verify the update
        admin_token = await adapter._get_admin_token()
        kc_client = await adapter._find_client(admin_token, creds.client_id)

        attrs = kc_client.get("attributes", {})
        assert json.loads(attrs["a2a.permissions"]) == new_perms

        # Verify the protocol mapper was also updated
        for mapper in kc_client.get("protocolMappers", []):
            if mapper["name"] == "a2a-permissions":
                assert json.loads(mapper["config"]["claim.value"]) == new_perms

    @pytest.mark.asyncio
    async def test_update_nonexistent_client_raises(self, adapter):
        with pytest.raises(ValueError, match="not found"):
            await adapter.update_client_claims("nonexistent-client-id", {})


class TestKeycloakAdapterDeleteClient:
    """Test client deletion."""

    @pytest.mark.asyncio
    async def test_delete_client(self, adapter):
        creds = await adapter.register_client(
            agent_id="test-agent-delete-001",
            org_id=1,
            permissions={},
        )

        await adapter.delete_client(creds.client_id)

        # Verify it's gone
        admin_token = await adapter._get_admin_token()
        with pytest.raises(ValueError, match="not found"):
            await adapter._find_client(admin_token, creds.client_id)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_client_is_noop(self, adapter):
        # Should not raise
        await adapter.delete_client("nonexistent-client-xyz")


class TestKeycloakAdapterIssueTokenForAgent:
    """Test issue_token_for_agent — the internal token issuance path."""

    @pytest.mark.asyncio
    async def test_issue_token_for_agent(self, adapter_with_db, mock_db):
        # Register a real Keycloak client
        creds = await adapter_with_db.register_client(
            agent_id="test-agent-issue-001",
            org_id=55,
            permissions={"allowed_modes": ["research"]},
        )


        # Insert a mock DB row that issue_token_for_agent will look up
        mock_db.insert_row({
            "agent_id": "test-agent-issue-001",
            "idp_client_id": creds.client_id,
            "is_active": True,
        })

        # Issue token
        token = await adapter_with_db.issue_token_for_agent("test-agent-issue-001")
        assert token
        assert isinstance(token, str)

        # Decode and verify claims
        import base64
        parts = token.split(".")
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        assert payload["iss"].endswith("/realms/fiberwise")
        assert "a2a" in payload
        assert payload["agent_id"] == "test-agent-issue-001"

    @pytest.mark.asyncio
    async def test_issue_token_for_agent_no_db_raises(self):
        adapter = KeycloakAdapter(db=None)
        with pytest.raises(RuntimeError, match="requires db"):
            await adapter.issue_token_for_agent("any-agent")

    @pytest.mark.asyncio
    async def test_issue_token_for_agent_unknown_agent_raises(self, adapter_with_db):
        with pytest.raises(ValueError, match="No IDP-registered key found"):
            await adapter_with_db.issue_token_for_agent("nonexistent-agent")


class TestKeycloakAdapterDiscovery:
    """Test discovery URL."""

    def test_get_discovery_url(self, adapter):
        url = adapter.get_discovery_url()
        assert "realms/fiberwise" in url
        assert "openid-configuration" in url


class TestKeycloakAdapterTokenClaimsEndToEnd:
    """End-to-end: register client, get token, verify all a2a claims in JWT."""

    @pytest.mark.asyncio
    async def test_full_flow_claims_in_token(self, adapter):
        permissions = {
            "allowed_modes": ["research", "execute"],
            "allowed_tools": ["Read", "Glob", "Grep", "Edit", "Bash"],
            "limits": {"max_turns": 50, "max_duration_seconds": 600},
        }

        creds = await adapter.register_client(
            agent_id="test-agent-e2e-001",
            org_id=123,
            permissions=permissions,
        )


        # Get token using the client credentials
        token = await adapter.get_token(creds.client_id, creds.client_secret)

        # Decode JWT payload
        import base64
        parts = token.split(".")
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        # Verify standard OIDC claims
        assert "iss" in payload
        assert "sub" in payload
        assert "exp" in payload
        assert "iat" in payload

        # Verify a2a custom claims from protocol mappers
        assert payload["a2a"] == permissions
        assert payload["agent_id"] == "test-agent-e2e-001"
        assert payload["org_id"] == 123
