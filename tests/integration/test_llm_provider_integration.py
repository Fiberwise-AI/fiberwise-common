"""
Integration tests for LLMProviderService against real databases.

Tests the DB-layer methods (get_provider_by_id, get_default_provider) only.
Does NOT test execute_llm_request or generate_completion (those require external API calls).

SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.
"""
import json
import pytest
import pytest_asyncio

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.llm_provider_service import LLMProviderService


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

LLM_PROVIDERS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS llm_providers (
    provider_id TEXT PRIMARY KEY,
    name TEXT,
    provider_type TEXT,
    api_endpoint TEXT,
    configuration TEXT,
    is_active INTEGER DEFAULT 1,
    is_default INTEGER DEFAULT 0,
    is_system INTEGER DEFAULT 0,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

LLM_PROVIDERS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS llm_providers (
    provider_id TEXT PRIMARY KEY,
    name TEXT,
    provider_type TEXT,
    api_endpoint TEXT,
    configuration TEXT,
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    is_system BOOLEAN DEFAULT false,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Create llm_providers table for SQLite."""
    await provider.execute(LLM_PROVIDERS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Create llm_providers table for PostgreSQL."""
    await provider.execute(LLM_PROVIDERS_TABLE_PG)


async def _insert_provider(provider: NexusQLProvider, **overrides) -> None:
    """Insert a test LLM provider row directly."""
    defaults = {
        "provider_id": "test-provider",
        "name": "Test Provider",
        "provider_type": "openai",
        "api_endpoint": "https://api.openai.com/v1",
        "configuration": json.dumps({"api_key": "sk-test", "default_model": "gpt-4"}),
        "is_active": True,
        "is_default": False,
        "is_system": True,
        "created_by": None,
    }
    defaults.update(overrides)
    await provider.execute("""
        INSERT INTO llm_providers
            (provider_id, name, provider_type, api_endpoint, configuration,
             is_active, is_default, is_system, created_by)
        VALUES
            (:provider_id, :name, :provider_type, :api_endpoint, :configuration,
             :is_active, :is_default, :is_system, :created_by)
    """, defaults)


# ===========================================================================
# SQLite tests
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestLLMProviderServiceSQLite:
    """LLMProviderService → NexusQLProvider → SQLite."""

    @pytest_asyncio.fixture
    async def db(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    # -- get_provider_by_id --------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_system_provider_without_user_scoping(self, db):
        """System provider is visible when no user_id is set."""
        await _insert_provider(db, provider_id="sys-openai", is_system=True)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("sys-openai")
        assert result is not None
        assert result["provider_id"] == "sys-openai"

    @pytest.mark.asyncio
    async def test_get_user_provider_with_user_scoping(self, db):
        """User-created provider is visible when user_id matches created_by."""
        await _insert_provider(db, provider_id="user-custom", is_system=False, created_by=42)
        svc = LLMProviderService(db, user_id=42)
        result = await svc.get_provider_by_id("user-custom")
        assert result is not None
        assert result["provider_id"] == "user-custom"

    @pytest.mark.asyncio
    async def test_user_provider_hidden_without_user_scoping(self, db):
        """User-created (non-system) provider is NOT visible without user_id."""
        await _insert_provider(db, provider_id="user-only", is_system=False, created_by=42)
        svc = LLMProviderService(db)  # no user_id
        result = await svc.get_provider_by_id("user-only")
        assert result is None

    @pytest.mark.asyncio
    async def test_user_provider_hidden_from_other_user(self, db):
        """User-created provider is NOT visible to a different user."""
        await _insert_provider(db, provider_id="private", is_system=False, created_by=42)
        svc = LLMProviderService(db, user_id=99)
        result = await svc.get_provider_by_id("private")
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_not_found_returns_none(self, db):
        """Non-existent provider_id returns None."""
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_inactive_provider_not_returned(self, db):
        """Inactive provider is not returned."""
        await _insert_provider(db, provider_id="inactive", is_active=False)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("inactive")
        assert result is None

    # -- get_default_provider ------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_default_provider_system(self, db):
        """System default provider is returned when no user_id is set."""
        await _insert_provider(db, provider_id="default-sys", is_default=True, is_system=True)
        svc = LLMProviderService(db)
        result = await svc.get_default_provider()
        assert result is not None
        assert result["provider_id"] == "default-sys"

    @pytest.mark.asyncio
    async def test_get_default_provider_user_specific(self, db):
        """User's own default provider is returned when user_id is set."""
        await _insert_provider(db, provider_id="default-user", is_default=True, is_system=False, created_by=7)
        svc = LLMProviderService(db, user_id=7)
        result = await svc.get_default_provider()
        assert result is not None
        assert result["provider_id"] == "default-user"

    @pytest.mark.asyncio
    async def test_get_default_provider_none_when_no_default(self, db):
        """Returns None when no default provider exists."""
        await _insert_provider(db, provider_id="not-default", is_default=False)
        svc = LLMProviderService(db)
        result = await svc.get_default_provider()
        assert result is None

    # -- Configuration JSON parsing ------------------------------------------

    @pytest.mark.asyncio
    async def test_configuration_json_parsed(self, db):
        """Configuration string is parsed into a dict."""
        config = {"api_key": "sk-abc", "default_model": "gpt-4", "temperature": 0.5}
        await _insert_provider(db, provider_id="json-test", configuration=json.dumps(config))
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("json-test")
        assert result is not None
        assert isinstance(result["configuration"], dict)
        assert result["configuration"]["api_key"] == "sk-abc"
        assert result["configuration"]["default_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_double_serialized_json_parsed(self, db):
        """Double-serialized JSON configuration is unwrapped correctly."""
        inner = {"api_key": "sk-double", "default_model": "gpt-3.5-turbo"}
        double_serialized = json.dumps(json.dumps(inner))
        await _insert_provider(db, provider_id="double-json", configuration=double_serialized)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("double-json")
        assert result is not None
        assert isinstance(result["configuration"], dict)
        assert result["configuration"]["api_key"] == "sk-double"

    # -- Cross-DB boolean handling -------------------------------------------

    @pytest.mark.asyncio
    async def test_boolean_handling_sqlite_integers(self, db):
        """SQLite stores booleans as 0/1 integers; service queries handle this."""
        await _insert_provider(db, provider_id="bool-test", is_active=True, is_system=True, is_default=True)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("bool-test")
        assert result is not None
        default = await svc.get_default_provider()
        assert default is not None
        assert default["provider_id"] == "bool-test"


# ===========================================================================
# PostgreSQL tests — only run when TEST_POSTGRESQL_URL is set
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestLLMProviderServicePostgreSQL:
    """LLMProviderService → NexusQLProvider → PostgreSQL."""

    @pytest_asyncio.fixture
    async def db(self, pg_provider):
        # Clean up before creating
        try:
            await pg_provider.execute("DROP TABLE IF EXISTS llm_providers CASCADE")
        except Exception:
            pass
        await _setup_pg_schema(pg_provider)
        yield pg_provider
        # Clean up after
        try:
            await pg_provider.execute("DROP TABLE IF EXISTS llm_providers CASCADE")
        except Exception:
            pass

    # -- get_provider_by_id --------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_system_provider_without_user_scoping(self, db):
        """System provider is visible when no user_id is set."""
        await _insert_provider(db, provider_id="pg-sys", is_system=True)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-sys")
        assert result is not None
        assert result["provider_id"] == "pg-sys"

    @pytest.mark.asyncio
    async def test_get_user_provider_with_user_scoping(self, db):
        """User-created provider is visible when user_id matches."""
        await _insert_provider(db, provider_id="pg-user", is_system=False, created_by=42)
        svc = LLMProviderService(db, user_id=42)
        result = await svc.get_provider_by_id("pg-user")
        assert result is not None
        assert result["provider_id"] == "pg-user"

    @pytest.mark.asyncio
    async def test_user_provider_hidden_without_user_scoping(self, db):
        """User-created provider is NOT visible without user_id."""
        await _insert_provider(db, provider_id="pg-priv", is_system=False, created_by=42)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-priv")
        assert result is None

    @pytest.mark.asyncio
    async def test_user_provider_hidden_from_other_user(self, db):
        """User-created provider is NOT visible to a different user."""
        await _insert_provider(db, provider_id="pg-other", is_system=False, created_by=42)
        svc = LLMProviderService(db, user_id=99)
        result = await svc.get_provider_by_id("pg-other")
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_not_found_returns_none(self, db):
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_inactive_provider_not_returned(self, db):
        await _insert_provider(db, provider_id="pg-inactive", is_active=False)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-inactive")
        assert result is None

    # -- get_default_provider ------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_default_provider_system(self, db):
        await _insert_provider(db, provider_id="pg-def-sys", is_default=True, is_system=True)
        svc = LLMProviderService(db)
        result = await svc.get_default_provider()
        assert result is not None
        assert result["provider_id"] == "pg-def-sys"

    @pytest.mark.asyncio
    async def test_get_default_provider_user_specific(self, db):
        await _insert_provider(db, provider_id="pg-def-user", is_default=True, is_system=False, created_by=7)
        svc = LLMProviderService(db, user_id=7)
        result = await svc.get_default_provider()
        assert result is not None
        assert result["provider_id"] == "pg-def-user"

    @pytest.mark.asyncio
    async def test_get_default_provider_none_when_no_default(self, db):
        await _insert_provider(db, provider_id="pg-nodef", is_default=False)
        svc = LLMProviderService(db)
        result = await svc.get_default_provider()
        assert result is None

    # -- Configuration JSON parsing ------------------------------------------

    @pytest.mark.asyncio
    async def test_configuration_json_parsed(self, db):
        config = {"api_key": "sk-pg", "default_model": "gpt-4"}
        await _insert_provider(db, provider_id="pg-json", configuration=json.dumps(config))
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-json")
        assert result is not None
        assert isinstance(result["configuration"], dict)
        assert result["configuration"]["api_key"] == "sk-pg"

    @pytest.mark.asyncio
    async def test_double_serialized_json_parsed(self, db):
        inner = {"api_key": "sk-pg-double", "default_model": "gpt-3.5-turbo"}
        double_serialized = json.dumps(json.dumps(inner))
        await _insert_provider(db, provider_id="pg-dbl", configuration=double_serialized)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-dbl")
        assert result is not None
        assert isinstance(result["configuration"], dict)
        assert result["configuration"]["api_key"] == "sk-pg-double"

    # -- Cross-DB boolean handling -------------------------------------------

    @pytest.mark.asyncio
    async def test_boolean_handling_pg_native(self, db):
        """PostgreSQL stores booleans natively; service queries handle this."""
        await _insert_provider(db, provider_id="pg-bool", is_active=True, is_system=True, is_default=True)
        svc = LLMProviderService(db)
        result = await svc.get_provider_by_id("pg-bool")
        assert result is not None
        default = await svc.get_default_provider()
        assert default is not None
        assert default["provider_id"] == "pg-bool"
