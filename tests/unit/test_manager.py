"""
Unit tests for fiberwise_common.database.manager module.

Tests DatabaseManager initialization, migration tracking, and lifecycle
using real SQLite databases (no mocking needed since SQLite is always available).
"""
import pytest
from unittest.mock import MagicMock, patch

from fiberwise_common.database.manager import DatabaseManager


class TestDatabaseManagerInit:
    """Test DatabaseManager construction."""

    def test_init_stores_url(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'test.db'}"
        mgr = DatabaseManager(url)
        assert mgr.database_url == url
        assert mgr.provider is not None

    def test_default_migrations_dir(self, tmp_path):
        mgr = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
        assert mgr.migrations_dir.name == "sql"

    def test_custom_migrations_dir(self, tmp_path):
        custom = tmp_path / "my_migrations"
        custom.mkdir()
        mgr = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}", migrations_dir=custom)
        assert mgr.migrations_dir == custom

    def test_create_from_settings(self, tmp_path):
        settings = MagicMock()
        settings.DATABASE_URL = f"sqlite:///{tmp_path / 'test.db'}"
        mgr = DatabaseManager.create_from_settings(settings)
        assert mgr.database_url == settings.DATABASE_URL

    def test_create_from_settings_missing_url(self):
        settings = MagicMock(spec=[])
        with pytest.raises(ValueError, match="DATABASE_URL"):
            DatabaseManager.create_from_settings(settings)

    def test_get_provider(self, tmp_path):
        mgr = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
        assert mgr.get_provider() is mgr.provider


class TestDatabaseManagerLifecycle:
    """Test initialize, shutdown, and health_check."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, tmp_path):
        mgr = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
        result = await mgr.initialize()
        assert result is True
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_bad_url_fails(self):
        mgr = DatabaseManager("postgresql://bad:bad@nonexistent:9999/nope")
        result = await mgr.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown(self, tmp_path):
        mgr = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
        await mgr.initialize()
        await mgr.shutdown()
        # No error means success

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, tmp_path):
        mgr = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
        await mgr.initialize()
        assert await mgr.health_check() is True
        await mgr.shutdown()


class TestDatabaseManagerMigrations:
    """Test migration application and tracking."""

    @pytest.mark.asyncio
    async def test_creates_schema_migrations_table(self, tmp_path):
        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=tmp_path / "empty_dir",
        )
        (tmp_path / "empty_dir").mkdir()
        await mgr.initialize()
        await mgr.apply_migrations()

        row = await mgr.provider.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        assert row is not None
        await mgr.shutdown()

    @pytest.mark.asyncio
    @patch("fiberwise_common.database.manager.load_sql_script", side_effect=FileNotFoundError)
    async def test_applies_numbered_migrations(self, _mock_load, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_create_items.sql").write_text(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);"
        )
        (migrations / "002_add_price.sql").write_text(
            "ALTER TABLE items ADD COLUMN price REAL DEFAULT 0;"
        )

        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=migrations,
        )
        await mgr.initialize()
        result = await mgr.apply_migrations()
        assert result is True

        # Verify schema was created
        await mgr.provider.execute(
            "INSERT INTO items (name, price) VALUES (:p1, :p2)",
            {"p1": "test", "p2": 1.99},
        )
        row = await mgr.provider.fetch_one("SELECT name, price FROM items")
        assert row["name"] == "test"
        assert row["price"] == 1.99

        # Verify tracking
        rows = await mgr.provider.fetch_all(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        versions = [r["version"] for r in rows]
        assert "001_create_items" in versions
        assert "002_add_price" in versions
        await mgr.shutdown()

    @pytest.mark.asyncio
    @patch("fiberwise_common.database.manager.load_sql_script", side_effect=FileNotFoundError)
    async def test_skips_already_applied(self, _mock_load, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_create_t.sql").write_text(
            "CREATE TABLE t (id INTEGER PRIMARY KEY);"
        )

        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=migrations,
        )
        await mgr.initialize()

        # First run
        assert await mgr.apply_migrations() is True
        # Second run — should not fail or duplicate
        assert await mgr.apply_migrations() is True

        rows = await mgr.provider.fetch_all("SELECT version FROM schema_migrations")
        assert len(rows) == 1
        await mgr.shutdown()

    @pytest.mark.asyncio
    @patch("fiberwise_common.database.manager.load_sql_script", side_effect=FileNotFoundError)
    async def test_migration_failure_returns_false(self, _mock_load, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_bad.sql").write_text("THIS IS NOT VALID SQL;")

        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=migrations,
        )
        await mgr.initialize()
        result = await mgr.apply_migrations()
        assert result is False
        await mgr.shutdown()

    @pytest.mark.asyncio
    @patch("fiberwise_common.database.manager.load_sql_script", side_effect=FileNotFoundError)
    async def test_applies_in_sorted_order(self, _mock_load, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        # Write out of order to make sure sort matters
        (migrations / "003_third.sql").write_text(
            "CREATE TABLE third (id INTEGER PRIMARY KEY);"
        )
        (migrations / "001_first.sql").write_text(
            "CREATE TABLE first (id INTEGER PRIMARY KEY);"
        )
        (migrations / "002_second.sql").write_text(
            "CREATE TABLE second (id INTEGER PRIMARY KEY);"
        )

        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=migrations,
        )
        await mgr.initialize()
        assert await mgr.apply_migrations() is True

        rows = await mgr.provider.fetch_all(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        versions = [r["version"] for r in rows]
        assert versions == ["001_first", "002_second", "003_third"]
        await mgr.shutdown()
