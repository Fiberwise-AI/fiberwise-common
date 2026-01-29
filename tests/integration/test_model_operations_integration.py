"""
Integration tests for ModelOperationsService against real databases.

Tests the full stack: ModelOperationsService → NexusQLProvider → real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/
"""
import os
import asyncio
import json
import uuid
import pytest
import pytest_asyncio
from pathlib import Path
from uuid import uuid4, UUID

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.model_operations_service import ModelOperationsService


# ---------------------------------------------------------------------------
# Schema helpers — create only the tables each test class needs
# ---------------------------------------------------------------------------

APPS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS apps (
    app_id TEXT PRIMARY KEY,
    app_slug TEXT,
    name TEXT,
    description TEXT,
    version TEXT,
    creator_user_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

MODELS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    app_id TEXT,
    model_slug TEXT,
    name TEXT,
    description TEXT,
    table_name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

FIELDS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS fields (
    field_id TEXT PRIMARY KEY,
    model_id TEXT,
    field_column TEXT,
    name TEXT,
    is_primary_key BOOLEAN DEFAULT 0,
    data_type TEXT,
    is_required BOOLEAN DEFAULT 0,
    is_unique BOOLEAN DEFAULT 0,
    default_value_json TEXT,
    validations_json TEXT,
    relation_details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# PostgreSQL variants (uses SERIAL, BOOLEAN literals, etc.)
APPS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS apps (
    app_id TEXT PRIMARY KEY,
    app_slug TEXT,
    name TEXT,
    description TEXT,
    version TEXT,
    creator_user_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

MODELS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    app_id TEXT,
    model_slug TEXT,
    name TEXT,
    description TEXT,
    table_name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

FIELDS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS fields (
    field_id TEXT PRIMARY KEY,
    model_id TEXT,
    field_column TEXT,
    name TEXT,
    is_primary_key BOOLEAN DEFAULT false,
    data_type TEXT,
    is_required BOOLEAN DEFAULT false,
    is_unique BOOLEAN DEFAULT false,
    default_value_json TEXT,
    validations_json TEXT,
    relation_details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


# Fixtures (sqlite_provider, pg_provider, etc.) are provided by conftest.py


async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Set up tables for SQLite integration tests."""
    await provider.execute(APPS_TABLE_SQLITE)
    await provider.execute(MODELS_TABLE_SQLITE)
    await provider.execute(FIELDS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Set up tables for PostgreSQL integration tests."""
    await provider.execute(APPS_TABLE_PG)
    await provider.execute(MODELS_TABLE_PG)
    await provider.execute(FIELDS_TABLE_PG)


async def _insert_app_directly(provider: NexusQLProvider, **overrides) -> dict:
    """Insert an app directly for tests that need a pre-existing app."""
    defaults = {
        "app_id": str(uuid4()),
        "app_slug": f"app-{uuid4().hex[:8]}",
        "name": "Test App",
        "description": "A test application",
        "version": "1.0.0",
        "creator_user_id": 1,
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO apps (app_id, app_slug, name, description, version, creator_user_id)
        VALUES (:app_id, :app_slug, :name, :description, :version, :creator_user_id)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM apps WHERE app_id = :app_id", {"app_id": defaults["app_id"]}
    )


async def _insert_model_directly(provider: NexusQLProvider, app_id: str, **overrides) -> dict:
    """Insert a model directly for tests that need a pre-existing model."""
    defaults = {
        "model_id": str(uuid4()),
        "app_id": app_id,
        "model_slug": f"model-{uuid4().hex[:8]}",
        "name": "Test Model",
        "description": "A test model",
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO models (model_id, app_id, model_slug, name, description)
        VALUES (:model_id, :app_id, :model_slug, :name, :description)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM models WHERE model_id = :model_id", {"model_id": defaults["model_id"]}
    )


async def _insert_field_directly(provider: NexusQLProvider, model_id: str, **overrides) -> dict:
    """Insert a field directly for tests that need a pre-existing field."""
    defaults = {
        "field_id": str(uuid4()),
        "model_id": model_id,
        "field_column": f"col_{uuid4().hex[:8]}",
        "name": "Test Field",
        "data_type": "text",
        "is_required": False,
        "is_unique": False,
        "is_primary_key": False,
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO fields (field_id, model_id, field_column, name, data_type, is_required, is_unique, is_primary_key)
        VALUES (:field_id, :model_id, :field_column, :name, :data_type, :is_required, :is_unique, :is_primary_key)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM fields WHERE field_id = :field_id", {"field_id": defaults["field_id"]}
    )


# ===========================================================================
# ModelOperationsService integration tests — SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestModelOperationsServiceSQLite:
    """ModelOperationsService → NexusQLProvider → SQLite."""

    @pytest_asyncio.fixture
    async def provider(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return ModelOperationsService(provider)

    @pytest_asyncio.fixture
    async def app(self, provider):
        return await _insert_app_directly(provider, name="My App", app_slug="my-app")

    # -- get_current_models -------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_current_models_empty(self, svc, app):
        """get_current_models returns empty list when app has no models."""
        result = await svc.get_current_models(UUID(app["app_id"]))
        assert result == []

    @pytest.mark.asyncio
    async def test_get_current_models_with_data(self, svc, provider, app):
        """get_current_models returns models with their fields."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="customers", name="Customers")
        await _insert_field_directly(provider, model["model_id"], field_column="email", name="Email", data_type="text")
        await _insert_field_directly(provider, model["model_id"], field_column="age", name="Age", data_type="integer")

        result = await svc.get_current_models(UUID(app["app_id"]))
        assert len(result) == 1
        assert result[0]["model_slug"] == "customers"
        assert len(result[0]["fields"]) == 2

        field_columns = {f["field_column"] for f in result[0]["fields"]}
        assert field_columns == {"email", "age"}

    @pytest.mark.asyncio
    async def test_get_current_models_multiple_models(self, svc, provider, app):
        """get_current_models returns all models for the app."""
        await _insert_model_directly(provider, app["app_id"], model_slug="orders", name="Orders")
        await _insert_model_directly(provider, app["app_id"], model_slug="products", name="Products")

        result = await svc.get_current_models(UUID(app["app_id"]))
        assert len(result) == 2
        slugs = {m["model_slug"] for m in result}
        assert slugs == {"orders", "products"}

    @pytest.mark.asyncio
    async def test_get_current_models_does_not_return_other_app(self, svc, provider, app):
        """get_current_models only returns models belonging to the given app."""
        other_app = await _insert_app_directly(provider, name="Other App", app_slug="other-app")
        await _insert_model_directly(provider, app["app_id"], model_slug="mine", name="Mine")
        await _insert_model_directly(provider, other_app["app_id"], model_slug="theirs", name="Theirs")

        result = await svc.get_current_models(UUID(app["app_id"]))
        assert len(result) == 1
        assert result[0]["model_slug"] == "mine"

    # -- compare_models -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_compare_models_no_changes(self, svc, provider, app):
        """compare_models reports no changes when models match."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [{"model_slug": "tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"}
        ]}]

        changes = await svc.compare_models(current, new_manifest)
        assert changes["has_changes"] is False
        assert changes["new_models"] == []
        assert changes["new_fields"] == []

    @pytest.mark.asyncio
    async def test_compare_models_detects_new_model(self, svc, provider, app):
        """compare_models identifies a brand-new model."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [
            {"model_slug": "tasks", "name": "Tasks", "fields": []},
            {"model_slug": "notes", "name": "Notes", "fields": []},
        ]

        changes = await svc.compare_models(current, new_manifest)
        assert changes["has_changes"] is True
        assert len(changes["new_models"]) == 1
        assert changes["new_models"][0]["model_slug"] == "notes"

    @pytest.mark.asyncio
    async def test_compare_models_detects_new_optional_field(self, svc, provider, app):
        """compare_models identifies a new optional field on an existing model."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [{"model_slug": "tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"},
            {"field_column": "priority", "name": "Priority", "data_type": "integer", "is_required": False},
        ]}]

        changes = await svc.compare_models(current, new_manifest)
        assert changes["has_changes"] is True
        assert len(changes["new_fields"]) == 1
        assert changes["new_fields"][0]["model_slug"] == "tasks"
        assert changes["new_fields"][0]["field"]["field_column"] == "priority"

    @pytest.mark.asyncio
    async def test_compare_models_skips_new_required_field(self, svc, provider, app):
        """compare_models does NOT include new required fields (unsafe change)."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [{"model_slug": "tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"},
            {"field_column": "owner", "name": "Owner", "data_type": "text", "is_required": True},
        ]}]

        changes = await svc.compare_models(current, new_manifest)
        # Required field addition is skipped as unsafe
        assert changes["new_fields"] == []

    # -- create_models_from_manifest ----------------------------------------

    @pytest.mark.asyncio
    async def test_create_models_from_manifest(self, svc, provider, app):
        """create_models_from_manifest inserts a model with fields."""
        manifest_models = [{
            "model_slug": "invoices",
            "name": "Invoices",
            "description": "Invoice records",
            "fields": [
                {"field_column": "invoice_number", "name": "Invoice Number", "data_type": "text", "is_required": True, "is_unique": True, "is_primary_key": False},
                {"field_column": "amount", "name": "Amount", "data_type": "decimal", "is_required": False, "is_unique": False, "is_primary_key": False},
            ],
        }]

        result = await svc.create_models_from_manifest(
            UUID(app["app_id"]), manifest_models, connection=provider
        )
        assert result["success"] is True
        assert result["model_count"] == 1
        assert result["total_fields"] == 2

        # Verify persisted
        models = await svc.get_current_models(UUID(app["app_id"]))
        assert len(models) == 1
        assert models[0]["model_slug"] == "invoices"
        assert len(models[0]["fields"]) == 2

    @pytest.mark.asyncio
    async def test_create_models_from_manifest_multiple(self, svc, provider, app):
        """create_models_from_manifest handles multiple models at once."""
        manifest_models = [
            {"model_slug": "orders", "name": "Orders", "description": "Order records", "fields": [
                {"field_column": "order_id", "name": "Order ID", "data_type": "text", "is_primary_key": True},
            ]},
            {"model_slug": "line_items", "name": "Line Items", "description": "Line item records", "fields": [
                {"field_column": "product", "name": "Product", "data_type": "text"},
                {"field_column": "quantity", "name": "Quantity", "data_type": "integer"},
            ]},
        ]

        result = await svc.create_models_from_manifest(
            UUID(app["app_id"]), manifest_models, connection=provider
        )
        assert result["success"] is True
        assert result["model_count"] == 2
        assert result["total_fields"] == 3

    @pytest.mark.asyncio
    async def test_create_models_requires_connection(self, svc, app):
        """create_models_from_manifest raises ValueError without a connection."""
        with pytest.raises(ValueError, match="connection is required"):
            await svc.create_models_from_manifest(UUID(app["app_id"]), [], connection=None)

    # -- add_fields_to_existing_models --------------------------------------

    @pytest.mark.asyncio
    async def test_add_fields_to_existing_models(self, svc, provider, app):
        """add_fields_to_existing_models inserts new fields on existing models."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="contacts", name="Contacts")
        await _insert_field_directly(provider, model["model_id"], field_column="name", name="Name", data_type="text")

        new_fields = [{
            "model_slug": "contacts",
            "field": {"field_column": "phone", "name": "Phone", "data_type": "text", "is_required": False},
        }]

        result = await svc.add_fields_to_existing_models(
            UUID(app["app_id"]), new_fields, connection=provider
        )
        assert result["success"] is True
        assert result["field_count"] == 1

        # Verify persisted
        models = await svc.get_current_models(UUID(app["app_id"]))
        assert len(models[0]["fields"]) == 2
        columns = {f["field_column"] for f in models[0]["fields"]}
        assert "phone" in columns

    @pytest.mark.asyncio
    async def test_add_fields_skips_missing_model(self, svc, provider, app):
        """add_fields_to_existing_models silently skips when the model slug is not found."""
        new_fields = [{
            "model_slug": "nonexistent",
            "field": {"field_column": "foo", "name": "Foo", "data_type": "text"},
        }]

        result = await svc.add_fields_to_existing_models(
            UUID(app["app_id"]), new_fields, connection=provider
        )
        assert result["success"] is True
        assert result["field_count"] == 0

    @pytest.mark.asyncio
    async def test_add_fields_requires_connection(self, svc, app):
        """add_fields_to_existing_models raises ValueError without a connection."""
        with pytest.raises(ValueError, match="connection is required"):
            await svc.add_fields_to_existing_models(UUID(app["app_id"]), [], connection=None)

    # -- process_model_updates (end-to-end) ---------------------------------

    @pytest.mark.asyncio
    async def test_process_model_updates_no_changes(self, svc, provider, app):
        """process_model_updates reports no changes when manifest matches DB."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        new_manifest = [{"model_slug": "tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"},
        ]}]

        result = await svc.process_model_updates(
            UUID(app["app_id"]), new_manifest, connection=provider
        )
        assert result["success"] is True
        assert result["new_models"] == 0
        assert result["new_fields"] == 0

    @pytest.mark.asyncio
    async def test_process_model_updates_adds_model_and_field(self, svc, provider, app):
        """process_model_updates creates new models AND adds optional fields."""
        # Existing model with one field
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        new_manifest = [
            # Existing model with a new optional field
            {"model_slug": "tasks", "name": "Tasks", "fields": [
                {"field_column": "title", "name": "Title", "data_type": "text"},
                {"field_column": "status", "name": "Status", "data_type": "text", "is_required": False},
            ]},
            # Brand-new model
            {"model_slug": "comments", "name": "Comments", "description": "Comment records", "fields": [
                {"field_column": "body", "name": "Body", "data_type": "text"},
            ]},
        ]

        result = await svc.process_model_updates(
            UUID(app["app_id"]), new_manifest, connection=provider
        )
        assert result["success"] is True
        assert result["new_models"] == 1
        assert result["new_fields"] == 1

        # Verify both models exist
        models = await svc.get_current_models(UUID(app["app_id"]))
        slugs = {m["model_slug"] for m in models}
        assert slugs == {"tasks", "comments"}

        # Verify the new field on tasks
        tasks_model = next(m for m in models if m["model_slug"] == "tasks")
        task_columns = {f["field_column"] for f in tasks_model["fields"]}
        assert "status" in task_columns

    # -- get_model_attr -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_model_attr_dict(self, svc):
        """get_model_attr works with dict inputs."""
        d = {"name": "hello", "value": 42}
        assert svc.get_model_attr(d, "name") == "hello"
        assert svc.get_model_attr(d, "value") == 42
        assert svc.get_model_attr(d, "missing") is None

    @pytest.mark.asyncio
    async def test_get_model_attr_object(self, svc):
        """get_model_attr works with object inputs."""
        class Obj:
            name = "hello"
            value = 42

        obj = Obj()
        assert svc.get_model_attr(obj, "name") == "hello"
        assert svc.get_model_attr(obj, "value") == 42
        assert svc.get_model_attr(obj, "missing") is None

    # -- concurrent reads ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_model_reads(self, svc, provider, app):
        """Multiple concurrent get_current_models calls should all succeed."""
        await _insert_model_directly(provider, app["app_id"], model_slug="alpha", name="Alpha")
        await _insert_model_directly(provider, app["app_id"], model_slug="beta", name="Beta")

        results = await asyncio.gather(
            svc.get_current_models(UUID(app["app_id"])),
            svc.get_current_models(UUID(app["app_id"])),
            svc.get_current_models(UUID(app["app_id"])),
        )
        assert all(len(r) == 2 for r in results)


# ===========================================================================
# PostgreSQL variants — only run when TEST_POSTGRESQL_URL is set
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestModelOperationsServicePostgreSQL:
    """ModelOperationsService → NexusQLProvider → PostgreSQL."""

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        # Drop tables in dependency order before creating
        for table in ["fields", "models", "apps"]:
            try:
                await pg_provider.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            except Exception:
                pass
        await _setup_pg_schema(pg_provider)
        return pg_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return ModelOperationsService(provider)

    @pytest_asyncio.fixture
    async def app(self, provider):
        return await _insert_app_directly(provider, name="PG App", app_slug="pg-app")

    # -- get_current_models -------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_current_models_empty(self, svc, app):
        """get_current_models returns empty list when app has no models."""
        result = await svc.get_current_models(UUID(app["app_id"]))
        assert result == []

    @pytest.mark.asyncio
    async def test_get_current_models_with_data(self, svc, provider, app):
        """get_current_models returns models with their fields."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="pg-customers", name="Customers")
        await _insert_field_directly(provider, model["model_id"], field_column="email", name="Email", data_type="text")
        await _insert_field_directly(provider, model["model_id"], field_column="age", name="Age", data_type="integer")

        result = await svc.get_current_models(UUID(app["app_id"]))
        assert len(result) == 1
        assert result[0]["model_slug"] == "pg-customers"
        assert len(result[0]["fields"]) == 2

    # -- compare_models -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_compare_models_detects_new_model(self, svc, provider, app):
        """compare_models identifies a brand-new model."""
        await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [
            {"model_slug": "tasks", "name": "Tasks", "fields": []},
            {"model_slug": "notes", "name": "Notes", "fields": []},
        ]

        changes = await svc.compare_models(current, new_manifest)
        assert changes["has_changes"] is True
        assert len(changes["new_models"]) == 1
        assert changes["new_models"][0]["model_slug"] == "notes"

    @pytest.mark.asyncio
    async def test_compare_models_detects_new_optional_field(self, svc, provider, app):
        """compare_models identifies a new optional field on an existing model."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [{"model_slug": "tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"},
            {"field_column": "priority", "name": "Priority", "data_type": "integer", "is_required": False},
        ]}]

        changes = await svc.compare_models(current, new_manifest)
        assert changes["has_changes"] is True
        assert len(changes["new_fields"]) == 1

    @pytest.mark.asyncio
    async def test_compare_models_skips_required_field(self, svc, provider, app):
        """compare_models skips new required fields as unsafe."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        current = await svc.get_current_models(UUID(app["app_id"]))
        new_manifest = [{"model_slug": "tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"},
            {"field_column": "owner", "name": "Owner", "data_type": "text", "is_required": True},
        ]}]

        changes = await svc.compare_models(current, new_manifest)
        assert changes["new_fields"] == []

    # -- create_models_from_manifest ----------------------------------------

    @pytest.mark.asyncio
    async def test_create_models_from_manifest(self, svc, provider, app):
        """create_models_from_manifest inserts a model with fields."""
        manifest_models = [{
            "model_slug": "pg-invoices",
            "name": "Invoices",
            "description": "Invoice records",
            "fields": [
                {"field_column": "invoice_number", "name": "Invoice Number", "data_type": "text", "is_required": True, "is_unique": True, "is_primary_key": False},
                {"field_column": "amount", "name": "Amount", "data_type": "decimal", "is_required": False, "is_unique": False, "is_primary_key": False},
            ],
        }]

        result = await svc.create_models_from_manifest(
            UUID(app["app_id"]), manifest_models, connection=provider
        )
        assert result["success"] is True
        assert result["model_count"] == 1
        assert result["total_fields"] == 2

        # Verify persisted
        models = await svc.get_current_models(UUID(app["app_id"]))
        assert len(models) == 1
        assert models[0]["model_slug"] == "pg-invoices"

    @pytest.mark.asyncio
    async def test_create_models_requires_connection(self, svc, app):
        """create_models_from_manifest raises ValueError without a connection."""
        with pytest.raises(ValueError, match="connection is required"):
            await svc.create_models_from_manifest(UUID(app["app_id"]), [], connection=None)

    # -- add_fields_to_existing_models --------------------------------------

    @pytest.mark.asyncio
    async def test_add_fields_to_existing_models(self, svc, provider, app):
        """add_fields_to_existing_models inserts new fields on existing models."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="pg-contacts", name="Contacts")
        await _insert_field_directly(provider, model["model_id"], field_column="name", name="Name", data_type="text")

        new_fields = [{
            "model_slug": "pg-contacts",
            "field": {"field_column": "phone", "name": "Phone", "data_type": "text", "is_required": False},
        }]

        result = await svc.add_fields_to_existing_models(
            UUID(app["app_id"]), new_fields, connection=provider
        )
        assert result["success"] is True
        assert result["field_count"] == 1

        models = await svc.get_current_models(UUID(app["app_id"]))
        columns = {f["field_column"] for f in models[0]["fields"]}
        assert "phone" in columns

    @pytest.mark.asyncio
    async def test_add_fields_requires_connection(self, svc, app):
        """add_fields_to_existing_models raises ValueError without a connection."""
        with pytest.raises(ValueError, match="connection is required"):
            await svc.add_fields_to_existing_models(UUID(app["app_id"]), [], connection=None)

    # -- process_model_updates (end-to-end) ---------------------------------

    @pytest.mark.asyncio
    async def test_process_model_updates_no_changes(self, svc, provider, app):
        """process_model_updates reports no changes when manifest matches DB."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="pg-tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        new_manifest = [{"model_slug": "pg-tasks", "name": "Tasks", "fields": [
            {"field_column": "title", "name": "Title", "data_type": "text"},
        ]}]

        result = await svc.process_model_updates(
            UUID(app["app_id"]), new_manifest, connection=provider
        )
        assert result["success"] is True
        assert result["new_models"] == 0
        assert result["new_fields"] == 0

    @pytest.mark.asyncio
    async def test_process_model_updates_adds_model_and_field(self, svc, provider, app):
        """process_model_updates creates new models AND adds optional fields."""
        model = await _insert_model_directly(provider, app["app_id"], model_slug="pg-tasks", name="Tasks")
        await _insert_field_directly(provider, model["model_id"], field_column="title", name="Title", data_type="text")

        new_manifest = [
            {"model_slug": "pg-tasks", "name": "Tasks", "fields": [
                {"field_column": "title", "name": "Title", "data_type": "text"},
                {"field_column": "status", "name": "Status", "data_type": "text", "is_required": False},
            ]},
            {"model_slug": "pg-comments", "name": "Comments", "description": "Comment records", "fields": [
                {"field_column": "body", "name": "Body", "data_type": "text"},
            ]},
        ]

        result = await svc.process_model_updates(
            UUID(app["app_id"]), new_manifest, connection=provider
        )
        assert result["success"] is True
        assert result["new_models"] == 1
        assert result["new_fields"] == 1

        models = await svc.get_current_models(UUID(app["app_id"]))
        slugs = {m["model_slug"] for m in models}
        assert slugs == {"pg-tasks", "pg-comments"}

    # -- concurrent reads ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_model_reads(self, svc, provider, app):
        """Multiple concurrent get_current_models calls should all succeed."""
        await _insert_model_directly(provider, app["app_id"], model_slug="pg-alpha", name="Alpha")
        await _insert_model_directly(provider, app["app_id"], model_slug="pg-beta", name="Beta")

        results = await asyncio.gather(
            svc.get_current_models(UUID(app["app_id"])),
            svc.get_current_models(UUID(app["app_id"])),
            svc.get_current_models(UUID(app["app_id"])),
        )
        assert all(len(r) == 2 for r in results)
