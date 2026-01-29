"""
Integration tests for AccountService against real databases.

Tests the full stack: AccountService -> NexusQLProvider -> real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/

In CI (Gitea), the pipeline sets TEST_POSTGRESQL_URL pointing at the service container.
"""
import json
import pytest
import pytest_asyncio

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.account_service import AccountService


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

ACCOUNT_CONFIGS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS account_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    provider TEXT,
    api_key TEXT,
    base_url TEXT,
    user_id INTEGER,
    config_data TEXT,
    is_default INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
"""

ACCOUNT_CONFIGS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS account_configs (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE,
    provider TEXT,
    api_key TEXT,
    base_url TEXT,
    user_id INTEGER,
    config_data TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TEXT,
    updated_at TEXT
);
"""


async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Set up tables for SQLite integration tests."""
    await provider.execute(ACCOUNT_CONFIGS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Set up tables for PostgreSQL integration tests."""
    await provider.execute(ACCOUNT_CONFIGS_TABLE_PG)


# Fixtures (sqlite_provider, pg_provider, etc.) are provided by conftest.py


# ===========================================================================
# AccountService integration tests — SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestAccountServiceSQLite:
    """AccountService -> NexusQLProvider -> SQLite."""

    @pytest_asyncio.fixture
    async def svc(self, sqlite_provider, tmp_path):
        await _setup_sqlite_schema(sqlite_provider)
        config_dir = str(tmp_path / "configs")
        return AccountService(sqlite_provider, config_dir=config_dir)

    @pytest.mark.asyncio
    async def test_add_config_and_retrieve(self, svc):
        """add_account_config persists a row and get_account_config retrieves it."""
        config = await svc.add_account_config(
            name="my-openai",
            provider="openai",
            api_key="sk-test-key-123",
            base_url="https://api.openai.com/v1",
        )
        assert config["name"] == "my-openai"
        assert config["provider"] == "openai"
        assert config["api_key"] == "sk-test-key-123"
        assert config["base_url"] == "https://api.openai.com/v1"

        fetched = await svc.get_account_config("my-openai")
        assert fetched is not None
        assert fetched["name"] == "my-openai"
        assert fetched["provider"] == "openai"
        assert fetched["api_key"] == "sk-test-key-123"

    @pytest.mark.asyncio
    async def test_add_config_saves_file(self, svc, tmp_path):
        """add_account_config also writes a JSON file to config_dir."""
        await svc.add_account_config(
            name="file-check",
            provider="anthropic",
            api_key="sk-ant-test",
        )
        config_file = tmp_path / "configs" / "file-check.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_list_configs_with_provider_filter(self, svc):
        """get_account_configs filters by provider."""
        await svc.add_account_config(name="oai-1", provider="openai", api_key="sk-1")
        await svc.add_account_config(name="oai-2", provider="openai", api_key="sk-2")
        await svc.add_account_config(name="anth-1", provider="anthropic", api_key="sk-ant-1")

        openai_configs = await svc.get_account_configs(provider="openai")
        assert len(openai_configs) == 2
        assert all(c["provider"] == "openai" for c in openai_configs)

        anthropic_configs = await svc.get_account_configs(provider="anthropic")
        assert len(anthropic_configs) == 1
        assert anthropic_configs[0]["name"] == "anth-1"

    @pytest.mark.asyncio
    async def test_list_configs_with_user_id_filter(self, svc):
        """get_account_configs filters by user_id."""
        await svc.add_account_config(name="u1-cfg", provider="openai", api_key="sk-1", user_id=1)
        await svc.add_account_config(name="u2-cfg", provider="openai", api_key="sk-2", user_id=2)

        user1_configs = await svc.get_account_configs(user_id=1)
        assert len(user1_configs) == 1
        assert user1_configs[0]["name"] == "u1-cfg"

    @pytest.mark.asyncio
    async def test_set_and_get_default_config(self, svc):
        """set_default_config marks one config as default; get_default_config retrieves it."""
        await svc.add_account_config(name="cfg-a", provider="openai", api_key="sk-a")
        await svc.add_account_config(name="cfg-b", provider="openai", api_key="sk-b")

        result = await svc.set_default_config("cfg-b")
        assert result is True

        default = await svc.get_default_config()
        assert default is not None
        assert default["name"] == "cfg-b"

    @pytest.mark.asyncio
    async def test_set_default_clears_previous_default(self, svc):
        """Setting a new default clears the previous default."""
        await svc.add_account_config(name="first", provider="openai", api_key="sk-1")
        await svc.add_account_config(name="second", provider="openai", api_key="sk-2")

        await svc.set_default_config("first")
        await svc.set_default_config("second")

        default = await svc.get_default_config()
        assert default["name"] == "second"

    @pytest.mark.asyncio
    async def test_get_default_config_with_provider_filter(self, svc):
        """get_default_config can filter by provider."""
        await svc.add_account_config(name="oai-default", provider="openai", api_key="sk-oai")
        await svc.add_account_config(name="anth-default", provider="anthropic", api_key="sk-ant")

        await svc.set_default_config("oai-default")

        default_openai = await svc.get_default_config(provider="openai")
        assert default_openai is not None
        assert default_openai["name"] == "oai-default"

        # No default set for anthropic specifically (oai-default is global default)
        # but provider filter should not return a non-matching provider
        default_anth = await svc.get_default_config(provider="anthropic")
        # Falls back to first anthropic config
        assert default_anth is not None
        assert default_anth["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_delete_config(self, svc):
        """delete_account_config removes from database and filesystem."""
        await svc.add_account_config(name="to-delete", provider="openai", api_key="sk-del")

        result = await svc.delete_account_config("to-delete")
        assert result is True

        fetched = await svc.get_account_config("to-delete")
        # After delete, DB returns None; file fallback also removed
        assert fetched is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_config_returns_none(self, svc):
        """get_account_config returns None for missing name."""
        result = await svc.get_account_config("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, svc):
        """validate_config accepts valid config data."""
        result = await svc.validate_config({
            "name": "test",
            "provider": "openai",
            "api_key": "sk-valid-key",
        })
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_fields(self, svc):
        """validate_config rejects config missing required fields."""
        result = await svc.validate_config({})
        assert result["valid"] is False
        assert len(result["errors"]) == 3  # name, provider, api_key

    @pytest.mark.asyncio
    async def test_validate_config_bad_url(self, svc):
        """validate_config rejects invalid base_url."""
        result = await svc.validate_config({
            "name": "test",
            "provider": "openai",
            "api_key": "sk-key",
            "base_url": "not-a-url",
        })
        assert result["valid"] is False
        assert any("Base URL" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_config_openai_key_warning(self, svc):
        """validate_config warns if OpenAI key lacks sk- prefix."""
        result = await svc.validate_config({
            "name": "test",
            "provider": "openai",
            "api_key": "bad-prefix",
        })
        assert result["valid"] is True
        assert any("sk-" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_validate_config_anthropic_key_warning(self, svc):
        """validate_config warns if Anthropic key lacks sk-ant- prefix."""
        result = await svc.validate_config({
            "name": "test",
            "provider": "anthropic",
            "api_key": "bad-prefix",
        })
        assert result["valid"] is True
        assert any("sk-ant-" in w for w in result["warnings"])


# ===========================================================================
# AccountService integration tests — PostgreSQL
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestAccountServicePostgreSQL:
    """AccountService -> NexusQLProvider -> PostgreSQL."""

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        # Clean up account_configs table before and after
        try:
            await pg_provider.execute("DROP TABLE IF EXISTS account_configs CASCADE")
        except Exception:
            pass
        await _setup_pg_schema(pg_provider)
        yield pg_provider
        try:
            await pg_provider.execute("DROP TABLE IF EXISTS account_configs CASCADE")
        except Exception:
            pass

    @pytest_asyncio.fixture
    async def svc(self, provider, tmp_path):
        config_dir = str(tmp_path / "configs")
        return AccountService(provider, config_dir=config_dir)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="INSERT OR REPLACE is SQLite-only syntax; fails on PostgreSQL")
    async def test_add_config_and_retrieve(self, svc):
        """add_account_config uses INSERT OR REPLACE which is not supported on PostgreSQL."""
        config = await svc.add_account_config(
            name="pg-openai",
            provider="openai",
            api_key="sk-pg-test",
        )
        assert config["name"] == "pg-openai"

        fetched = await svc.get_account_config("pg-openai")
        assert fetched is not None
        assert fetched["provider"] == "openai"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="INSERT OR REPLACE is SQLite-only syntax; fails on PostgreSQL")
    async def test_list_configs_with_provider_filter(self, svc):
        """get_account_configs after add_account_config — xfail due to INSERT OR REPLACE."""
        await svc.add_account_config(name="pg-oai", provider="openai", api_key="sk-1")
        await svc.add_account_config(name="pg-anth", provider="anthropic", api_key="sk-ant-1")

        openai_configs = await svc.get_account_configs(provider="openai")
        assert len(openai_configs) == 1

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="INSERT OR REPLACE is SQLite-only syntax; fails on PostgreSQL")
    async def test_set_and_get_default_config(self, svc):
        """set_default_config + get_default_config — xfail due to INSERT OR REPLACE in setup."""
        await svc.add_account_config(name="pg-default", provider="openai", api_key="sk-d")
        await svc.set_default_config("pg-default")

        default = await svc.get_default_config()
        assert default is not None
        assert default["name"] == "pg-default"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="INSERT OR REPLACE is SQLite-only syntax; fails on PostgreSQL")
    async def test_delete_config(self, svc):
        """delete_account_config — xfail due to INSERT OR REPLACE in setup."""
        await svc.add_account_config(name="pg-del", provider="openai", api_key="sk-del")
        result = await svc.delete_account_config("pg-del")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_config_returns_none(self, svc):
        """get_account_config returns None for missing name (no INSERT OR REPLACE needed)."""
        result = await svc.get_account_config("pg-does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_empty_configs_list(self, svc):
        """get_account_configs returns empty list when no configs exist."""
        configs = await svc.get_account_configs(provider="openai")
        assert configs == []

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, svc):
        """validate_config works without any DB interaction."""
        result = await svc.validate_config({
            "name": "pg-test",
            "provider": "openai",
            "api_key": "sk-valid",
        })
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_fields(self, svc):
        """validate_config rejects missing required fields (no DB needed)."""
        result = await svc.validate_config({})
        assert result["valid"] is False
        assert len(result["errors"]) == 3
