"""
Integration tests for ApiKeyService against real SQLite and PostgreSQL databases.

Tests the full stack: ApiKeyService → NexusQLProvider → real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/
"""
import json
import uuid
import pytest
import pytest_asyncio

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.api_keys_service import ApiKeyService, APIKeyData

# Pre-computed bcrypt hash for "testpassword"
_TEST_HASH = "$2b$12$Al0VrEkSchsrQTQ1HfC4EuTT7C47LHoffd6uX35yiVW5QoH68JM86"


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

USERS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE,
    username TEXT UNIQUE,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    hashed_password TEXT,
    is_active BOOLEAN DEFAULT 1,
    is_admin BOOLEAN DEFAULT 0,
    is_superuser BOOLEAN DEFAULT 0,
    is_verified BOOLEAN DEFAULT 0,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT,
    avatar_url TEXT,
    timezone TEXT DEFAULT 'UTC',
    locale TEXT DEFAULT 'en',
    global_role TEXT DEFAULT 'user',
    setup_completed BOOLEAN DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

API_KEYS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    organization_id INTEGER,
    scopes TEXT DEFAULT '[]',
    is_active BOOLEAN DEFAULT 1,
    expires_at TEXT,
    last_used_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

EXECUTION_API_KEYS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS execution_api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL UNIQUE,
    key_value TEXT NOT NULL,
    app_id TEXT NOT NULL,
    organization_id INTEGER NOT NULL,
    executor_type_id TEXT NOT NULL,
    executor_id TEXT NOT NULL,
    created_by INTEGER,
    scopes TEXT DEFAULT '[]',
    expiration TEXT,
    resource_pattern TEXT DEFAULT '*',
    metadata TEXT DEFAULT '{}',
    is_revoked INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

AGENT_API_KEYS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS agent_api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL UNIQUE,
    app_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    organization_id INTEGER NOT NULL,
    key_value TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    is_revoked INTEGER DEFAULT 0,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

USERS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    username TEXT UNIQUE,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    hashed_password TEXT,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    is_superuser BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT,
    avatar_url TEXT,
    timezone TEXT DEFAULT 'UTC',
    locale TEXT DEFAULT 'en',
    global_role TEXT DEFAULT 'user',
    setup_completed BOOLEAN DEFAULT false,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

API_KEYS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    organization_id INTEGER,
    scopes TEXT DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    expires_at TEXT,
    last_used_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

EXECUTION_API_KEYS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS execution_api_keys (
    id SERIAL PRIMARY KEY,
    key_id TEXT NOT NULL UNIQUE,
    key_value TEXT NOT NULL,
    app_id TEXT NOT NULL,
    organization_id INTEGER NOT NULL,
    executor_type_id TEXT NOT NULL,
    executor_id TEXT NOT NULL,
    created_by INTEGER,
    scopes TEXT DEFAULT '[]',
    expiration TEXT,
    resource_pattern TEXT DEFAULT '*',
    metadata TEXT DEFAULT '{}',
    is_revoked BOOLEAN DEFAULT false,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

AGENT_API_KEYS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS agent_api_keys (
    id SERIAL PRIMARY KEY,
    key_id TEXT NOT NULL UNIQUE,
    app_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    organization_id INTEGER NOT NULL,
    key_value TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_revoked BOOLEAN DEFAULT false,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# Schema setup helpers
# ---------------------------------------------------------------------------

async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Set up tables for SQLite integration tests."""
    await provider.execute(USERS_TABLE_SQLITE)
    await provider.execute(API_KEYS_TABLE_SQLITE)
    await provider.execute(EXECUTION_API_KEYS_TABLE_SQLITE)
    await provider.execute(AGENT_API_KEYS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Set up tables for PostgreSQL integration tests."""
    await provider.execute(USERS_TABLE_PG)
    await provider.execute(API_KEYS_TABLE_PG)
    await provider.execute(EXECUTION_API_KEYS_TABLE_PG)
    await provider.execute(AGENT_API_KEYS_TABLE_PG)


async def _insert_user_directly(provider: NexusQLProvider, **overrides) -> dict:
    """Insert a user directly for tests that need a pre-existing user."""
    defaults = {
        "uuid": str(uuid.uuid4()),
        "email": f"test-{uuid.uuid4().hex[:8]}@example.com",
        "username": f"user_{uuid.uuid4().hex[:8]}",
        "hashed_password": "fakehash",
        "first_name": "Test",
        "last_name": "User",
        "full_name": "Test User",
        "is_active": True,
        "is_superuser": False,
        "is_verified": False,
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO users (uuid, email, username, hashed_password,
                           first_name, last_name, full_name,
                           is_active, is_superuser, is_verified)
        VALUES (:uuid, :email, :username, :hashed_password,
                :first_name, :last_name, :full_name,
                :is_active, :is_superuser, :is_verified)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM users WHERE uuid = :uuid", {"uuid": defaults["uuid"]}
    )


async def _insert_api_key_directly(provider: NexusQLProvider, user_id: int, **overrides) -> dict:
    """Insert an API key directly into the database for read/validate tests."""
    import hashlib
    raw_key = str(uuid.uuid4())
    key_prefix = raw_key[:8]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    defaults = {
        "name": "test-key",
        "key_prefix": key_prefix,
        "key_hash": key_hash,
        "user_id": user_id,
        "organization_id": None,
        "scopes": "[]",
        "is_active": True,
        "expires_at": None,
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO api_keys (name, key_prefix, key_hash, user_id, organization_id, scopes, is_active, expires_at)
        VALUES (:name, :key_prefix, :key_hash, :user_id, :organization_id, :scopes, :is_active, :expires_at)
    """, defaults)

    row = await provider.fetch_one(
        "SELECT * FROM api_keys WHERE key_hash = :key_hash", {"key_hash": key_hash}
    )
    # Return both the row and the raw key for validation tests
    result = dict(row)
    result["_raw_key"] = raw_key
    return result


# ===========================================================================
# ApiKeyService integration tests -- SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestApiKeyServiceSQLite:
    """ApiKeyService → NexusQLProvider → SQLite.

    Note: create_api_key and delete_api_key use RETURNING clauses which are
    not supported by SQLite. Those methods are tested only on PostgreSQL.
    """

    @pytest_asyncio.fixture
    async def provider(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return ApiKeyService(provider)

    @pytest_asyncio.fixture
    async def user(self, provider):
        return await _insert_user_directly(provider, email="apiuser@example.com", username="apiuser")

    @pytest_asyncio.fixture
    async def user2(self, provider):
        return await _insert_user_directly(provider, email="apiuser2@example.com", username="apiuser2")

    # -- get_keys_for_user -------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_keys_for_user_empty(self, svc, user):
        """No keys exist yet — should return empty list."""
        keys = await svc.get_keys_for_user(user["id"])
        assert keys == []

    @pytest.mark.asyncio
    async def test_get_keys_for_user_returns_inserted_keys(self, svc, provider, user):
        """Keys inserted directly are returned by get_keys_for_user."""
        await _insert_api_key_directly(provider, user["id"], name="key-alpha", scopes='["read"]')
        await _insert_api_key_directly(provider, user["id"], name="key-beta", scopes='["read","write"]')

        keys = await svc.get_keys_for_user(user["id"])
        assert len(keys) == 2
        names = {k.name for k in keys}
        assert names == {"key-alpha", "key-beta"}

    @pytest.mark.asyncio
    async def test_get_keys_for_user_isolates_users(self, svc, provider, user, user2):
        """Keys for user A are not visible to user B."""
        await _insert_api_key_directly(provider, user["id"], name="user1-key")
        await _insert_api_key_directly(provider, user2["id"], name="user2-key")

        keys1 = await svc.get_keys_for_user(user["id"])
        keys2 = await svc.get_keys_for_user(user2["id"])
        assert len(keys1) == 1
        assert keys1[0].name == "user1-key"
        assert len(keys2) == 1
        assert keys2[0].name == "user2-key"

    @pytest.mark.asyncio
    async def test_get_keys_parses_scopes_json(self, svc, provider, user):
        """Scopes stored as JSON string are parsed into a list."""
        await _insert_api_key_directly(provider, user["id"], scopes='["read","write","admin"]')
        keys = await svc.get_keys_for_user(user["id"])
        assert keys[0].scopes == ["read", "write", "admin"]

    # -- validate_api_key --------------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, svc, provider, user):
        """A valid raw key resolves to the correct APIKeyInfo."""
        inserted = await _insert_api_key_directly(provider, user["id"], name="valid-key")
        raw_key = inserted["_raw_key"]

        info = await svc.validate_api_key(raw_key)
        assert info is not None
        assert info.user_id == user["id"]
        assert info.name == "valid-key"

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid_returns_none(self, svc):
        """An unknown key returns None."""
        info = await svc.validate_api_key("nonexistent-key-value")
        assert info is None

    @pytest.mark.asyncio
    async def test_validate_api_key_empty_returns_none(self, svc):
        """Empty string returns None."""
        assert await svc.validate_api_key("") is None
        assert await svc.validate_api_key(None) is None

    @pytest.mark.asyncio
    async def test_validate_api_key_expired_returns_none(self, svc, provider, user):
        """An expired key returns None."""
        inserted = await _insert_api_key_directly(
            provider, user["id"], name="expired-key", expires_at="2000-01-01T00:00:00+00:00"
        )
        info = await svc.validate_api_key(inserted["_raw_key"])
        assert info is None

    @pytest.mark.asyncio
    async def test_validate_api_key_updates_last_used(self, svc, provider, user):
        """Validating a key should update last_used_at."""
        inserted = await _insert_api_key_directly(provider, user["id"])
        raw_key = inserted["_raw_key"]

        # Initially last_used_at is NULL
        row_before = await provider.fetch_one(
            "SELECT last_used_at FROM api_keys WHERE id = :id", {"id": inserted["id"]}
        )
        assert row_before["last_used_at"] is None

        await svc.validate_api_key(raw_key)

        row_after = await provider.fetch_one(
            "SELECT last_used_at FROM api_keys WHERE id = :id", {"id": inserted["id"]}
        )
        assert row_after["last_used_at"] is not None

    # -- create_pipeline_execution_key -------------------------------------

    @pytest.mark.asyncio
    async def test_create_pipeline_execution_key(self, svc, provider, user):
        """Creates an execution key and stores it in execution_api_keys."""
        key = await svc.create_pipeline_execution_key(
            app_id="app-1", pipeline_id="pipe-1",
            created_by=user["id"], organization_id=1,
        )
        assert key is not None
        assert key.startswith("exec_")

        # Verify stored in DB
        row = await provider.fetch_one(
            "SELECT * FROM execution_api_keys WHERE key_value = :kv",
            {"kv": key},
        )
        assert row is not None
        assert row["app_id"] == "app-1"
        assert row["executor_type_id"] == "pipeline"
        assert row["executor_id"] == "pipe-1"

    # -- create_agent_api_key (SQLite) -------------------------------------
    # Note: create_agent_api_key uses `is_active = 1 AND is_revoked = 0` which
    # works on SQLite (INTEGER columns). It also uses NOW() which does NOT work
    # on SQLite for the INSERT, so we skip the create-new-key path on SQLite.

    @pytest.mark.asyncio
    async def test_create_agent_api_key_returns_existing(self, svc, provider, user):
        """If an active agent key already exists, it is returned."""
        # Insert an existing active agent key directly
        existing_key = "agent_existingkey123"
        await provider.execute("""
            INSERT INTO agent_api_keys (key_id, app_id, agent_id, organization_id, key_value, is_active, is_revoked, created_by)
            VALUES (:key_id, :app_id, :agent_id, :org_id, :key_value, 1, 0, :created_by)
        """, {
            "key_id": str(uuid.uuid4()),
            "app_id": "app-1",
            "agent_id": "agent-1",
            "org_id": 1,
            "key_value": existing_key,
            "created_by": user["id"],
        })

        result = await svc.create_agent_api_key(
            app_id="app-1", agent_id="agent-1",
            created_by=user["id"], organization_id=1,
        )
        assert result == existing_key


# ===========================================================================
# ApiKeyService integration tests -- PostgreSQL
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestApiKeyServicePostgreSQL:
    """ApiKeyService → NexusQLProvider → PostgreSQL.

    Tests all methods including those using RETURNING and NOW().
    """

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        await _setup_pg_schema(pg_provider)
        return pg_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return ApiKeyService(provider)

    @pytest_asyncio.fixture
    async def user(self, provider):
        return await _insert_user_directly(provider, email="pgapiuser@example.com", username="pgapiuser")

    @pytest_asyncio.fixture
    async def user2(self, provider):
        return await _insert_user_directly(provider, email="pgapiuser2@example.com", username="pgapiuser2")

    # -- create_api_key ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_api_key(self, svc, user):
        """create_api_key returns a raw key and an APIKeyResponse."""
        key_data = APIKeyData(name="pg-key", scopes=["read", "write"], expires_in_days=30)
        raw_key, response = await svc.create_api_key(user["id"], key_data)

        assert raw_key is not None
        assert len(raw_key) > 0
        assert response.name == "pg-key"
        assert response.scopes == ["read", "write"]
        assert response.id is not None
        assert response.key_prefix == raw_key[:8]

    @pytest.mark.asyncio
    async def test_create_api_key_no_expiry(self, svc, user):
        """create_api_key without expiration works."""
        key_data = APIKeyData(name="no-expiry-key")
        raw_key, response = await svc.create_api_key(user["id"], key_data)
        assert raw_key is not None
        assert response.expires_at is None

    @pytest.mark.asyncio
    async def test_create_api_key_with_scopes_string(self, svc, user):
        """Scopes passed as JSON string are accepted."""
        key_data = APIKeyData(name="str-scopes-key")
        key_data.scopes = '["admin"]'
        raw_key, response = await svc.create_api_key(user["id"], key_data)
        assert response.scopes == ["admin"]

    @pytest.mark.asyncio
    async def test_create_and_validate_roundtrip(self, svc, user):
        """Created key can be validated immediately."""
        key_data = APIKeyData(name="roundtrip-key", scopes=["read"])
        raw_key, response = await svc.create_api_key(user["id"], key_data)

        info = await svc.validate_api_key(raw_key)
        assert info is not None
        assert info.user_id == user["id"]
        assert info.name == "roundtrip-key"
        assert info.scopes == ["read"]

    # -- get_keys_for_user -------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_keys_for_user_empty(self, svc, user):
        keys = await svc.get_keys_for_user(user["id"])
        assert keys == []

    @pytest.mark.asyncio
    async def test_get_keys_for_user_after_create(self, svc, user):
        """Created keys appear in get_keys_for_user."""
        await svc.create_api_key(user["id"], APIKeyData(name="k1"))
        await svc.create_api_key(user["id"], APIKeyData(name="k2"))

        keys = await svc.get_keys_for_user(user["id"])
        assert len(keys) == 2
        names = {k.name for k in keys}
        assert names == {"k1", "k2"}

    @pytest.mark.asyncio
    async def test_get_keys_isolates_users(self, svc, user, user2):
        await svc.create_api_key(user["id"], APIKeyData(name="u1-key"))
        await svc.create_api_key(user2["id"], APIKeyData(name="u2-key"))

        keys1 = await svc.get_keys_for_user(user["id"])
        keys2 = await svc.get_keys_for_user(user2["id"])
        assert len(keys1) == 1
        assert keys1[0].name == "u1-key"
        assert len(keys2) == 1
        assert keys2[0].name == "u2-key"

    # -- validate_api_key --------------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_invalid_key(self, svc):
        assert await svc.validate_api_key("bogus") is None

    @pytest.mark.asyncio
    async def test_validate_empty_key(self, svc):
        assert await svc.validate_api_key("") is None
        assert await svc.validate_api_key(None) is None

    @pytest.mark.asyncio
    async def test_validate_expired_key(self, svc, provider, user):
        """Expired key returns None even though hash matches."""
        inserted = await _insert_api_key_directly(
            provider, user["id"], name="expired", expires_at="2000-01-01T00:00:00+00:00"
        )
        assert await svc.validate_api_key(inserted["_raw_key"]) is None

    # -- delete_api_key ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_api_key(self, svc, user):
        """Deleted key no longer validates."""
        key_data = APIKeyData(name="to-delete")
        raw_key, response = await svc.create_api_key(user["id"], key_data)

        deleted = await svc.delete_api_key(response.id, user["id"])
        assert deleted is True

        info = await svc.validate_api_key(raw_key)
        assert info is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, svc, user):
        """Deleting a key that does not exist returns False."""
        result = await svc.delete_api_key(99999, user["id"])
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_key_wrong_user(self, svc, user, user2):
        """Cannot delete another user's key."""
        key_data = APIKeyData(name="owned-by-u1")
        _, response = await svc.create_api_key(user["id"], key_data)

        result = await svc.delete_api_key(response.id, user2["id"])
        assert result is False

    # -- create_pipeline_execution_key -------------------------------------

    @pytest.mark.asyncio
    async def test_create_pipeline_execution_key(self, svc, user):
        key = await svc.create_pipeline_execution_key(
            app_id="app-pg", pipeline_id="pipe-pg",
            created_by=user["id"], organization_id=1,
        )
        assert key is not None
        assert key.startswith("exec_")

    # -- create_agent_api_key ----------------------------------------------

    @pytest.mark.asyncio
    async def test_create_agent_api_key_new(self, svc, user):
        """Creates a new agent key when none exists."""
        key = await svc.create_agent_api_key(
            app_id="app-pg", agent_id="agent-pg",
            created_by=user["id"], organization_id=1,
        )
        assert key is not None
        assert key.startswith("agent_")

    @pytest.mark.asyncio
    async def test_create_agent_api_key_reuses_existing(self, svc, user):
        """Second call for same agent returns the same key."""
        key1 = await svc.create_agent_api_key(
            app_id="app-pg", agent_id="agent-pg2",
            created_by=user["id"], organization_id=1,
        )
        key2 = await svc.create_agent_api_key(
            app_id="app-pg", agent_id="agent-pg2",
            created_by=user["id"], organization_id=1,
        )
        assert key1 == key2

    # -- full CRUD flow ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_api_key_lifecycle(self, svc, user):
        """Create → list → validate → delete → verify gone."""
        # Create
        key_data = APIKeyData(name="lifecycle-key", scopes=["read", "write"])
        raw_key, response = await svc.create_api_key(user["id"], key_data)
        assert response.id is not None

        # List
        keys = await svc.get_keys_for_user(user["id"])
        assert len(keys) == 1
        assert keys[0].name == "lifecycle-key"

        # Validate
        info = await svc.validate_api_key(raw_key)
        assert info is not None
        assert info.scopes == ["read", "write"]

        # Delete
        deleted = await svc.delete_api_key(response.id, user["id"])
        assert deleted is True

        # Verify gone
        keys_after = await svc.get_keys_for_user(user["id"])
        assert len(keys_after) == 0
        assert await svc.validate_api_key(raw_key) is None
