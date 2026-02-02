"""
Integration tests for DatabaseManager migrations via NexusQL MigrationRunner.

Tests the full migration lifecycle using real SQLite databases.
"""
import pytest
import pytest_asyncio

from fiberwise_common.database.manager import DatabaseManager


class TestMigrationIntegration:
    """Integration tests for the MigrationRunner-backed migration system."""

    @pytest_asyncio.fixture
    async def make_manager(self, tmp_path):
        """Factory fixture: creates a DatabaseManager with a temp migrations dir."""
        managers = []

        def _make(migrations_dir=None):
            mdir = migrations_dir or (tmp_path / "migrations")
            mdir.mkdir(exist_ok=True)
            mgr = DatabaseManager(
                f"sqlite:///{tmp_path / 'test.db'}",
                migrations_dir=mdir,
            )
            managers.append(mgr)
            return mgr, mdir

        yield _make

        for m in managers:
            try:
                await m.shutdown()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_creates_tracking_table(self, make_manager):
        """apply_migrations() creates the ia_migrations tracking table."""
        mgr, _ = make_manager()
        await mgr.initialize()
        await mgr.apply_migrations()

        row = await mgr.provider.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ia_migrations'"
        )
        assert row is not None
        assert row["name"] == "ia_migrations"

    @pytest.mark.asyncio
    async def test_applies_migrations_in_sorted_order(self, make_manager):
        """Migration files are applied in alphabetical order by stem."""
        mgr, mdir = make_manager()
        # Write out of order
        (mdir / "002_add_col.sql").write_text(
            "ALTER TABLE items ADD COLUMN price REAL DEFAULT 0;"
        )
        (mdir / "001_create_table.sql").write_text(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);"
        )

        await mgr.initialize()
        assert await mgr.apply_migrations() is True

        # Verify both tables/columns exist
        await mgr.provider.execute(
            "INSERT INTO items (name, price) VALUES (:n, :p)",
            {"n": "widget", "p": 9.99},
        )
        row = await mgr.provider.fetch_one("SELECT name, price FROM items")
        assert row["name"] == "widget"
        assert row["price"] == 9.99

        # Verify tracking order
        rows = await mgr.provider.fetch_all(
            "SELECT version FROM ia_migrations ORDER BY version"
        )
        versions = [r["version"] for r in rows]
        assert versions == ["001_create_table", "002_add_col"]

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, make_manager):
        """Running apply_migrations() twice doesn't duplicate or fail."""
        mgr, mdir = make_manager()
        (mdir / "001_create_t.sql").write_text(
            "CREATE TABLE t (id INTEGER PRIMARY KEY);"
        )

        await mgr.initialize()
        assert await mgr.apply_migrations() is True
        assert await mgr.apply_migrations() is True

        rows = await mgr.provider.fetch_all(
            "SELECT version FROM ia_migrations WHERE migration_type = 'system'"
        )
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_failed_migration_stops_chain(self, make_manager):
        """A failing migration stops execution and returns False."""
        mgr, mdir = make_manager()
        (mdir / "001_good.sql").write_text(
            "CREATE TABLE good (id INTEGER PRIMARY KEY);"
        )
        (mdir / "002_bad.sql").write_text("THIS IS NOT VALID SQL;")
        (mdir / "003_never.sql").write_text(
            "CREATE TABLE never_created (id INTEGER PRIMARY KEY);"
        )

        await mgr.initialize()
        result = await mgr.apply_migrations()
        assert result is False

        # 001 should have been applied before the failure
        row = await mgr.provider.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='good'"
        )
        assert row is not None

        # 003 should NOT have been applied
        row = await mgr.provider.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='never_created'"
        )
        assert row is None

    @pytest.mark.asyncio
    async def test_new_migration_picked_up_on_rerun(self, make_manager):
        """Adding a new migration file after initial run gets picked up."""
        mgr, mdir = make_manager()
        (mdir / "001_first.sql").write_text(
            "CREATE TABLE first (id INTEGER PRIMARY KEY);"
        )

        await mgr.initialize()
        assert await mgr.apply_migrations() is True

        # Add a new migration
        (mdir / "002_second.sql").write_text(
            "CREATE TABLE second (id INTEGER PRIMARY KEY);"
        )
        assert await mgr.apply_migrations() is True

        rows = await mgr.provider.fetch_all(
            "SELECT version FROM ia_migrations ORDER BY version"
        )
        versions = [r["version"] for r in rows]
        assert "001_first" in versions
        assert "002_second" in versions

        # Verify second table exists
        row = await mgr.provider.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='second'"
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_tracking_record_fields(self, make_manager):
        """Migration tracking records have correct version, filename, and type."""
        mgr, mdir = make_manager()
        (mdir / "001_users.sql").write_text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        )

        await mgr.initialize()
        await mgr.apply_migrations()

        row = await mgr.provider.fetch_one(
            "SELECT version, filename, migration_type FROM ia_migrations WHERE version = '001_users'"
        )
        assert row is not None
        assert row["version"] == "001_users"
        assert row["filename"] == "001_users.sql"
        assert row["migration_type"] == "system"

    @pytest.mark.asyncio
    async def test_empty_migrations_dir_succeeds(self, make_manager):
        """An empty migrations directory is fine — no-op success."""
        mgr, _ = make_manager()
        await mgr.initialize()
        assert await mgr.apply_migrations() is True

    @pytest.mark.asyncio
    async def test_real_migrations_apply_successfully(self, tmp_path):
        """Run the actual SQL migrations from fiberwise_common/database/sql/ against SQLite."""
        from pathlib import Path

        sql_dir = Path(__file__).resolve().parents[2] / "fiberwise_common" / "database" / "sql"
        assert sql_dir.exists(), f"SQL directory not found: {sql_dir}"

        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=sql_dir,
        )
        await mgr.initialize()
        result = await mgr.apply_migrations()
        assert result is True, "Real migrations failed — check SQL compatibility"

        # Verify key tables exist
        for table in ["users", "organizations", "agents", "pipelines", "apps"]:
            row = await mgr.provider.fetch_one(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            assert row is not None, f"Expected table '{table}' was not created"

        # Verify all migration files were tracked
        rows = await mgr.provider.fetch_all(
            "SELECT version FROM ia_migrations ORDER BY version"
        )
        versions = [r["version"] for r in rows]
        assert "000_init" in versions
        assert "001_add_oidc_identities" in versions
        assert "002_add_pipeline_activations_table" in versions

        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_real_migrations_idempotent(self, tmp_path):
        """Running real migrations twice succeeds (idempotent)."""
        from pathlib import Path

        sql_dir = Path(__file__).resolve().parents[2] / "fiberwise_common" / "database" / "sql"

        mgr = DatabaseManager(
            f"sqlite:///{tmp_path / 'test.db'}",
            migrations_dir=sql_dir,
        )
        await mgr.initialize()
        assert await mgr.apply_migrations() is True
        assert await mgr.apply_migrations() is True
        await mgr.shutdown()
