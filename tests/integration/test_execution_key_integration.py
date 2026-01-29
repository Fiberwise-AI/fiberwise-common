"""
Integration tests for ExecutionKeyService against real databases.

Tests the full stack: ExecutionKeyService → NexusQLProvider → real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/
"""
import json
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.execution_key_service import ExecutionKeyService


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

EXECUTION_KEYS_TABLE_SQLITE = """
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

EXECUTION_KEYS_TABLE_PG = """
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
    is_revoked INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Set up tables for SQLite integration tests."""
    await provider.execute(EXECUTION_KEYS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Set up tables for PostgreSQL integration tests."""
    await provider.execute(EXECUTION_KEYS_TABLE_PG)


async def _insert_expired_key(provider: NexusQLProvider, svc: ExecutionKeyService,
                               app_id: str = "app1", org_id: int = 1,
                               executor_type_id: str = "function",
                               executor_id: str = "func1") -> dict:
    """Create a key then manually set its expiration to the past."""
    key = await svc.create_execution_key(
        app_id=app_id,
        organization_id=org_id,
        executor_type_id=executor_type_id,
        executor_id=executor_id,
        expiration_minutes=60,
    )
    # Backdate expiration to the past
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await provider.execute(
        "UPDATE execution_api_keys SET expiration = :exp WHERE key_id = :kid",
        {"exp": past, "kid": key["key_id"]},
    )
    return key


# ===========================================================================
# ExecutionKeyService integration tests — SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestExecutionKeyServiceSQLite:
    """ExecutionKeyService → NexusQLProvider → SQLite."""

    @pytest_asyncio.fixture
    async def provider(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return ExecutionKeyService(provider)

    @pytest.mark.asyncio
    async def test_create_key_and_validate(self, svc):
        """create_execution_key returns key data; validate_execution_key succeeds."""
        key = await svc.create_execution_key(
            app_id="myapp",
            organization_id=1,
            executor_type_id="function",
            executor_id="func_abc",
            scopes=["read", "write"],
            resource_pattern="org/1/*",
            metadata={"env": "test"},
        )
        assert key is not None
        assert key["app_id"] == "myapp"
        assert key["executor_type_id"] == "function"
        assert key["executor_id"] == "func_abc"
        assert key["scopes"] == ["read", "write"]
        assert key["resource_pattern"] == "org/1/*"
        assert key["metadata"] == {"env": "test"}
        assert key["key_id"].startswith("exec_")
        assert key["key_value"].startswith("exec_")

        # Validate the key
        validated = await svc.validate_execution_key(key["key_value"])
        assert validated is not None
        assert validated["key_id"] == key["key_id"]
        assert validated["app_id"] == "myapp"
        assert validated["scopes"] == ["read", "write"]

    @pytest.mark.asyncio
    async def test_create_key_and_revoke_then_validation_fails(self, svc):
        """After revoking, validate_execution_key returns None."""
        key = await svc.create_execution_key(
            app_id="myapp", organization_id=1,
            executor_type_id="agent", executor_id="agent_1",
        )
        assert key is not None

        revoked = await svc.revoke_execution_key(key["key_id"])
        assert revoked is True

        validated = await svc.validate_execution_key(key["key_value"])
        assert validated is None

    @pytest.mark.asyncio
    async def test_expired_key_validation_fails(self, svc, provider):
        """An expired key should not validate."""
        key = await _insert_expired_key(provider, svc)
        validated = await svc.validate_execution_key(key["key_value"])
        assert validated is None

    @pytest.mark.asyncio
    async def test_validate_nonexistent_key_returns_none(self, svc):
        result = await svc.validate_execution_key("exec_doesnotexist")
        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key_returns_false(self, svc):
        result = await svc.revoke_execution_key("exec_doesnotexist")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_execution_keys_no_filters(self, svc):
        """get_execution_keys returns all non-revoked keys when no filters given."""
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.create_execution_key(
            app_id="app2", organization_id=2,
            executor_type_id="agent", executor_id="a1",
        )
        keys = await svc.get_execution_keys()
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_get_execution_keys_filter_by_app_id(self, svc):
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.create_execution_key(
            app_id="app2", organization_id=1,
            executor_type_id="function", executor_id="f2",
        )
        keys = await svc.get_execution_keys(app_id="app1")
        assert len(keys) == 1
        assert keys[0]["app_id"] == "app1"

    @pytest.mark.asyncio
    async def test_get_execution_keys_filter_by_executor_type(self, svc):
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="agent", executor_id="a1",
        )
        keys = await svc.get_execution_keys(executor_type_id="agent")
        assert len(keys) == 1
        assert keys[0]["executor_type_id"] == "agent"

    @pytest.mark.asyncio
    async def test_get_execution_keys_excludes_revoked_by_default(self, svc):
        key = await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.revoke_execution_key(key["key_id"])

        keys = await svc.get_execution_keys()
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_get_execution_keys_include_revoked(self, svc):
        key = await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.revoke_execution_key(key["key_id"])

        keys = await svc.get_execution_keys(include_revoked=True)
        assert len(keys) == 1
        assert keys[0]["is_revoked"] is True

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys(self, svc, provider):
        """cleanup_expired_keys marks expired keys as revoked."""
        # Create one expired key and one valid key
        await _insert_expired_key(provider, svc, app_id="expired_app")
        await svc.create_execution_key(
            app_id="valid_app", organization_id=1,
            executor_type_id="function", executor_id="f_valid",
        )

        count = await svc.cleanup_expired_keys()
        assert count >= 1

        # The expired key should now be revoked
        keys = await svc.get_execution_keys(include_revoked=False)
        app_ids = [k["app_id"] for k in keys]
        assert "expired_app" not in app_ids
        assert "valid_app" in app_ids

    @pytest.mark.asyncio
    async def test_get_execution_keys_filter_by_executor_id(self, svc):
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f2",
        )
        keys = await svc.get_execution_keys(executor_id="f2")
        assert len(keys) == 1
        assert keys[0]["executor_id"] == "f2"

    @pytest.mark.asyncio
    async def test_get_execution_keys_filter_by_created_by(self, svc):
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f1",
            created_by=42,
        )
        await svc.create_execution_key(
            app_id="app1", organization_id=1,
            executor_type_id="function", executor_id="f2",
            created_by=99,
        )
        keys = await svc.get_execution_keys(created_by=42)
        assert len(keys) == 1
        assert keys[0]["created_by"] == 42


# ===========================================================================
# ExecutionKeyService integration tests — PostgreSQL
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestExecutionKeyServicePostgreSQL:
    """ExecutionKeyService → NexusQLProvider → PostgreSQL."""

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        await _setup_pg_schema(pg_provider)
        return pg_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return ExecutionKeyService(provider)

    @pytest.mark.asyncio
    async def test_create_key_and_validate(self, svc):
        key = await svc.create_execution_key(
            app_id="pgapp",
            organization_id=1,
            executor_type_id="function",
            executor_id="pg_func",
            scopes=["execute"],
            metadata={"env": "pg_test"},
        )
        assert key is not None
        assert key["app_id"] == "pgapp"
        assert key["key_id"].startswith("exec_")

        validated = await svc.validate_execution_key(key["key_value"])
        assert validated is not None
        assert validated["key_id"] == key["key_id"]
        assert validated["scopes"] == ["execute"]

    @pytest.mark.asyncio
    async def test_revoke_key_then_validation_fails(self, svc):
        key = await svc.create_execution_key(
            app_id="pgapp", organization_id=1,
            executor_type_id="agent", executor_id="pg_agent",
        )
        assert await svc.revoke_execution_key(key["key_id"]) is True
        assert await svc.validate_execution_key(key["key_value"]) is None

    @pytest.mark.asyncio
    async def test_expired_key_validation_fails(self, svc, provider):
        key = await _insert_expired_key(provider, svc, app_id="pg_expired")
        assert await svc.validate_execution_key(key["key_value"]) is None

    @pytest.mark.asyncio
    async def test_get_keys_with_filters(self, svc):
        await svc.create_execution_key(
            app_id="pgapp1", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.create_execution_key(
            app_id="pgapp2", organization_id=1,
            executor_type_id="agent", executor_id="a1",
        )
        keys = await svc.get_execution_keys(app_id="pgapp1")
        assert len(keys) == 1
        assert keys[0]["app_id"] == "pgapp1"

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys(self, svc, provider):
        await _insert_expired_key(provider, svc, app_id="pg_expired")
        await svc.create_execution_key(
            app_id="pg_valid", organization_id=1,
            executor_type_id="function", executor_id="f_valid",
        )
        count = await svc.cleanup_expired_keys()
        assert count >= 1

        keys = await svc.get_execution_keys(include_revoked=False)
        app_ids = [k["app_id"] for k in keys]
        assert "pg_expired" not in app_ids
        assert "pg_valid" in app_ids

    @pytest.mark.asyncio
    async def test_include_revoked(self, svc):
        key = await svc.create_execution_key(
            app_id="pgapp", organization_id=1,
            executor_type_id="function", executor_id="f1",
        )
        await svc.revoke_execution_key(key["key_id"])

        keys = await svc.get_execution_keys(include_revoked=True)
        assert len(keys) == 1
        assert keys[0]["is_revoked"] is True
