"""
Integration tests for AppService against real databases.

Tests the full stack: AppService -> NexusQLProvider -> real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/
"""
import json
import uuid
import pytest
import pytest_asyncio

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.app_service import AppService


# ---------------------------------------------------------------------------
# Schema helpers — SQLite variants
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

ORGANIZATIONS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    display_name TEXT,
    description TEXT,
    slug TEXT NOT NULL UNIQUE,
    website TEXT,
    logo_url TEXT,
    billing_email TEXT,
    settings TEXT DEFAULT '{}',
    subscription_tier TEXT DEFAULT 'free',
    max_users INTEGER DEFAULT 5,
    max_apps INTEGER DEFAULT 10,
    max_storage_gb INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT 1,
    is_verified BOOLEAN DEFAULT 0,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);
"""

ORG_MEMBERS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS organization_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    status TEXT DEFAULT 'active',
    invited_by INTEGER,
    invited_at TEXT,
    joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(organization_id, user_id)
);
"""

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
    is_primary_key BOOLEAN,
    data_type TEXT,
    is_required BOOLEAN,
    is_unique BOOLEAN,
    default_value_json TEXT,
    validations_json TEXT,
    relation_details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

APP_INSTALLATIONS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS app_installations (
    installation_id TEXT PRIMARY KEY,
    app_id TEXT,
    user_id INTEGER,
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

APP_VERSIONS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS app_versions (
    app_version_id TEXT PRIMARY KEY,
    app_id TEXT,
    version TEXT,
    description TEXT,
    manifest_yaml TEXT,
    status TEXT,
    entry_point_url TEXT,
    created_by TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# ---------------------------------------------------------------------------
# Schema helpers — PostgreSQL variants
# ---------------------------------------------------------------------------

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

ORGANIZATIONS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS organizations (
    id SERIAL PRIMARY KEY,
    uuid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    display_name TEXT,
    description TEXT,
    slug TEXT NOT NULL UNIQUE,
    website TEXT,
    logo_url TEXT,
    billing_email TEXT,
    settings TEXT DEFAULT '{}',
    subscription_tier TEXT DEFAULT 'free',
    max_users INTEGER DEFAULT 5,
    max_apps INTEGER DEFAULT 10,
    max_storage_gb INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);
"""

ORG_MEMBERS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS organization_members (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    status TEXT DEFAULT 'active',
    invited_by INTEGER,
    invited_at TEXT,
    joined_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(organization_id, user_id)
);
"""

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

APP_INSTALLATIONS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS app_installations (
    installation_id TEXT PRIMARY KEY,
    app_id TEXT,
    user_id INTEGER,
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

APP_VERSIONS_TABLE_PG = """
CREATE TABLE IF NOT EXISTS app_versions (
    app_version_id TEXT PRIMARY KEY,
    app_id TEXT,
    version TEXT,
    description TEXT,
    manifest_yaml TEXT,
    status TEXT,
    entry_point_url TEXT,
    created_by TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# Schema setup helpers
# ---------------------------------------------------------------------------

async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Set up all tables for SQLite integration tests."""
    await provider.execute(USERS_TABLE_SQLITE)
    await provider.execute(ORGANIZATIONS_TABLE_SQLITE)
    await provider.execute(ORG_MEMBERS_TABLE_SQLITE)
    await provider.execute(APPS_TABLE_SQLITE)
    await provider.execute(MODELS_TABLE_SQLITE)
    await provider.execute(FIELDS_TABLE_SQLITE)
    await provider.execute(APP_INSTALLATIONS_TABLE_SQLITE)
    await provider.execute(APP_VERSIONS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Set up all tables for PostgreSQL integration tests."""
    await provider.execute(USERS_TABLE_PG)
    await provider.execute(ORGANIZATIONS_TABLE_PG)
    await provider.execute(ORG_MEMBERS_TABLE_PG)
    await provider.execute(APPS_TABLE_PG)
    await provider.execute(MODELS_TABLE_PG)
    await provider.execute(FIELDS_TABLE_PG)
    await provider.execute(APP_INSTALLATIONS_TABLE_PG)
    await provider.execute(APP_VERSIONS_TABLE_PG)


# ---------------------------------------------------------------------------
# Test data insertion helpers
# ---------------------------------------------------------------------------

async def _insert_user(provider: NexusQLProvider, **overrides) -> dict:
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


async def _insert_app(provider: NexusQLProvider, **overrides) -> dict:
    """Insert an app directly."""
    defaults = {
        "app_id": str(uuid.uuid4()),
        "app_slug": f"test-app-{uuid.uuid4().hex[:8]}",
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


async def _insert_model(provider: NexusQLProvider, **overrides) -> dict:
    """Insert a model directly."""
    defaults = {
        "model_id": str(uuid.uuid4()),
        "app_id": str(uuid.uuid4()),
        "model_slug": f"test-model-{uuid.uuid4().hex[:8]}",
        "name": "Test Model",
        "description": "A test model",
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO models (model_id, app_id, model_slug, name, description)
        VALUES (:model_id, :app_id, :model_slug, :name, :description)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM models WHERE model_id = :model_id",
        {"model_id": defaults["model_id"]},
    )


async def _insert_field(provider: NexusQLProvider, **overrides) -> dict:
    """Insert a field directly."""
    defaults = {
        "field_id": str(uuid.uuid4()),
        "model_id": str(uuid.uuid4()),
        "field_column": "test_column",
        "name": "Test Column",
        "is_primary_key": False,
        "data_type": "string",
        "is_required": False,
        "is_unique": False,
        "default_value_json": None,
        "validations_json": None,
        "relation_details_json": None,
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO fields (field_id, model_id, field_column, name, is_primary_key,
                            data_type, is_required, is_unique, default_value_json,
                            validations_json, relation_details_json)
        VALUES (:field_id, :model_id, :field_column, :name, :is_primary_key,
                :data_type, :is_required, :is_unique, :default_value_json,
                :validations_json, :relation_details_json)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM fields WHERE field_id = :field_id",
        {"field_id": defaults["field_id"]},
    )


async def _insert_installation(provider: NexusQLProvider, **overrides) -> dict:
    """Insert an app installation directly."""
    defaults = {
        "installation_id": str(uuid.uuid4()),
        "app_id": str(uuid.uuid4()),
        "user_id": 1,
        "status": "active",
    }
    defaults.update(overrides)

    await provider.execute("""
        INSERT INTO app_installations (installation_id, app_id, user_id, status)
        VALUES (:installation_id, :app_id, :user_id, :status)
    """, defaults)

    return await provider.fetch_one(
        "SELECT * FROM app_installations WHERE installation_id = :installation_id",
        {"installation_id": defaults["installation_id"]},
    )


# ===========================================================================
# AppService integration tests -- SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestAppServiceSQLite:
    """AppService -> NexusQLProvider -> SQLite."""

    @pytest_asyncio.fixture
    async def provider(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return AppService(provider)

    @pytest_asyncio.fixture
    async def user(self, provider):
        return await _insert_user(provider, email="appowner@example.com", username="appowner")

    @pytest_asyncio.fixture
    async def app(self, provider, user):
        return await _insert_app(
            provider,
            app_slug="my-app",
            name="My App",
            description="Integration test app",
            version="1.0.0",
            creator_user_id=user["id"],
        )

    @pytest_asyncio.fixture
    async def model(self, provider, app):
        return await _insert_model(
            provider,
            app_id=app["app_id"],
            model_slug="tasks",
            name="Tasks",
            description="Task model",
        )

    @pytest_asyncio.fixture
    async def fields(self, provider, model):
        """Insert a set of fields for the test model."""
        f1 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="id", name="ID",
                                  is_primary_key=True, data_type="integer",
                                  is_required=True, is_unique=True)
        f2 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="title", name="Title",
                                  data_type="string", is_required=True,
                                  validations_json='{"max_length": 100}')
        f3 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="done", name="Done",
                                  data_type="boolean", is_required=False,
                                  default_value_json='false')
        f4 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="priority", name="Priority",
                                  data_type="integer", is_required=False,
                                  default_value_json='1',
                                  validations_json='{}')
        return [f1, f2, f3, f4]

    # -- get_app_by_id -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_app_by_id(self, svc, app):
        """get_app_by_id returns the app dict for a valid ID."""
        result = await svc.get_app_by_id(app["app_id"])
        assert result["app_id"] == app["app_id"]
        assert result["name"] == "My App"
        assert result["app_slug"] == "my-app"

    @pytest.mark.asyncio
    async def test_get_app_by_id_not_found(self, svc):
        """get_app_by_id raises ValueError for a missing ID."""
        with pytest.raises(ValueError, match="not found"):
            await svc.get_app_by_id("nonexistent-id")

    # -- get_app_by_slug -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_app_by_slug(self, svc, app):
        """get_app_by_slug returns the app dict for a valid slug."""
        result = await svc.get_app_by_slug("my-app")
        assert result["app_id"] == app["app_id"]
        assert result["name"] == "My App"

    @pytest.mark.asyncio
    async def test_get_app_by_slug_not_found(self, svc):
        """get_app_by_slug raises ValueError for a missing slug."""
        with pytest.raises(ValueError, match="not found"):
            await svc.get_app_by_slug("no-such-slug")

    # -- get_model_by_slug ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_model_by_slug(self, svc, app, model):
        """get_model_by_slug returns the model dict."""
        result = await svc.get_model_by_slug(app["app_id"], "tasks")
        assert result["model_id"] == model["model_id"]
        assert result["model_slug"] == "tasks"
        assert result["name"] == "Tasks"

    @pytest.mark.asyncio
    async def test_get_model_by_slug_not_found(self, svc, app):
        """get_model_by_slug raises ValueError for a missing slug."""
        with pytest.raises(ValueError, match="not found"):
            await svc.get_model_by_slug(app["app_id"], "nonexistent-model")

    @pytest.mark.asyncio
    async def test_get_model_by_slug_wrong_app(self, svc, model):
        """get_model_by_slug raises ValueError when app_id does not match."""
        with pytest.raises(ValueError, match="not found"):
            await svc.get_model_by_slug("wrong-app-id", "tasks")

    # -- get_fields_for_model ------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_fields_for_model(self, svc, model, fields):
        """get_fields_for_model returns all fields with parsed JSON."""
        result = await svc.get_fields_for_model(model["model_id"])
        assert len(result) == 4
        columns = [f["field_column"] for f in result]
        assert "id" in columns
        assert "title" in columns
        assert "done" in columns
        assert "priority" in columns

    @pytest.mark.asyncio
    async def test_get_fields_for_model_parses_json(self, svc, model, fields):
        """get_fields_for_model parses JSON string fields into dicts/values."""
        result = await svc.get_fields_for_model(model["model_id"])
        title_field = next(f for f in result if f["field_column"] == "title")
        # validations_json should be parsed from string to dict
        assert isinstance(title_field["validations_json"], dict)
        assert title_field["validations_json"]["max_length"] == 100

    @pytest.mark.asyncio
    async def test_get_fields_for_model_parses_default_value(self, svc, model, fields):
        """get_fields_for_model parses default_value_json."""
        result = await svc.get_fields_for_model(model["model_id"])
        done_field = next(f for f in result if f["field_column"] == "done")
        # 'false' JSON string should parse to Python False
        assert done_field["default_value_json"] is False

    @pytest.mark.asyncio
    async def test_get_fields_for_model_empty(self, svc):
        """get_fields_for_model returns empty list for unknown model_id."""
        result = await svc.get_fields_for_model("nonexistent-model-id")
        assert result == []

    # -- validate_data_against_fields ----------------------------------------

    @pytest.mark.asyncio
    async def test_validate_data_valid(self, svc, model, fields):
        """validate_data_against_fields passes for valid data."""
        field_defs = await svc.get_fields_for_model(model["model_id"])
        result = await svc.validate_data_against_fields(
            {"id": 1, "title": "Do laundry"},
            field_defs,
            is_create=False,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_missing_required(self, svc, model, fields):
        """validate_data_against_fields raises for missing required fields."""
        field_defs = await svc.get_fields_for_model(model["model_id"])
        with pytest.raises(ValueError, match="Validation failed"):
            # 'title' is required but missing; 'id' is required but also missing
            await svc.validate_data_against_fields(
                {"done": True},
                field_defs,
                is_create=False,
            )

    @pytest.mark.asyncio
    async def test_validate_data_skip_pk_on_create(self, svc, model, fields):
        """validate_data_against_fields skips PK required check during create."""
        field_defs = await svc.get_fields_for_model(model["model_id"])
        # 'id' is PK + required, but is_create=True should skip it
        result = await svc.validate_data_against_fields(
            {"title": "New task"},
            field_defs,
            is_create=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_wrong_type(self, svc, model, fields):
        """validate_data_against_fields raises for wrong data type."""
        field_defs = await svc.get_fields_for_model(model["model_id"])
        with pytest.raises(ValueError, match="Validation failed"):
            await svc.validate_data_against_fields(
                {"id": 1, "title": 12345},  # title should be a string
                field_defs,
                is_create=False,
            )

    @pytest.mark.asyncio
    async def test_validate_data_undefined_field(self, svc, model, fields):
        """validate_data_against_fields rejects fields not in the model."""
        field_defs = await svc.get_fields_for_model(model["model_id"])
        with pytest.raises(ValueError, match="Validation failed"):
            await svc.validate_data_against_fields(
                {"id": 1, "title": "Ok", "ghost_field": "boo"},
                field_defs,
                is_create=False,
            )

    @pytest.mark.asyncio
    async def test_validate_data_max_length(self, svc, model, fields):
        """validate_data_against_fields enforces max_length from validations_json."""
        field_defs = await svc.get_fields_for_model(model["model_id"])
        with pytest.raises(ValueError, match="Validation failed"):
            await svc.validate_data_against_fields(
                {"id": 1, "title": "x" * 101},  # exceeds max_length=100
                field_defs,
                is_create=False,
            )

    # -- check_app_access ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_app_access_creator(self, svc, app, user):
        """check_app_access returns True for the app creator."""
        result = await svc.check_app_access(app["app_id"], user["id"])
        assert result is True

    @pytest.mark.asyncio
    async def test_check_app_access_installed_user(self, svc, provider, app):
        """check_app_access returns True for a user who installed the app."""
        other_user = await _insert_user(provider, email="installer@example.com", username="installer")
        await _insert_installation(
            provider,
            app_id=app["app_id"],
            user_id=other_user["id"],
            status="active",
        )
        result = await svc.check_app_access(app["app_id"], other_user["id"])
        assert result is True

    @pytest.mark.asyncio
    async def test_check_app_access_denied(self, svc, provider, app):
        """check_app_access raises ValueError for a user without access."""
        stranger = await _insert_user(provider, email="stranger@example.com", username="stranger")
        with pytest.raises(ValueError, match="don't have access"):
            await svc.check_app_access(app["app_id"], stranger["id"])

    @pytest.mark.asyncio
    async def test_check_app_access_inactive_installation(self, svc, provider, app):
        """check_app_access raises ValueError when installation is not active."""
        other_user = await _insert_user(provider, email="inactive@example.com", username="inactive")
        await _insert_installation(
            provider,
            app_id=app["app_id"],
            user_id=other_user["id"],
            status="revoked",
        )
        with pytest.raises(ValueError, match="don't have access"):
            await svc.check_app_access(app["app_id"], other_user["id"])

    # -- fetch_app_data ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_app_data(self, svc, app, model, fields):
        """fetch_app_data returns full app structure with models and fields."""
        result = await svc.fetch_app_data(app["app_id"])
        assert result is not None
        assert result["app_id"] == app["app_id"]
        assert result["name"] == "My App"
        assert "models" in result
        assert len(result["models"]) == 1

        m = result["models"][0]
        assert m["model_slug"] == "tasks"
        assert "fields" in m
        assert len(m["fields"]) == 4

    @pytest.mark.asyncio
    async def test_fetch_app_data_parses_json_fields(self, svc, app, model, fields):
        """fetch_app_data parses JSON columns in nested fields."""
        result = await svc.fetch_app_data(app["app_id"])
        task_model = result["models"][0]
        title_field = next(f for f in task_model["fields"] if f["field_column"] == "title")
        assert isinstance(title_field["validations_json"], dict)
        assert title_field["validations_json"]["max_length"] == 100

    @pytest.mark.asyncio
    async def test_fetch_app_data_not_found(self, svc):
        """fetch_app_data returns None for a missing app."""
        result = await svc.fetch_app_data("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_app_data_no_models(self, svc, provider, user):
        """fetch_app_data returns app with empty models list when none exist."""
        bare_app = await _insert_app(
            provider,
            app_slug="bare-app",
            name="Bare App",
            creator_user_id=user["id"],
        )
        result = await svc.fetch_app_data(bare_app["app_id"])
        assert result is not None
        assert result["models"] == []

    @pytest.mark.asyncio
    async def test_fetch_app_data_multiple_models(self, svc, provider, app):
        """fetch_app_data returns all models belonging to the app."""
        await _insert_model(provider, app_id=app["app_id"],
                            model_slug="projects", name="Projects")
        await _insert_model(provider, app_id=app["app_id"],
                            model_slug="comments", name="Comments")
        # The fixture already inserted "tasks" model
        result = await svc.fetch_app_data(app["app_id"])
        # We do not have the 'model' fixture here, so only 2 models
        assert len(result["models"]) == 2

    # -- create_new_app (uses RETURNING) -------------------------------------

    @pytest.mark.asyncio
    async def test_create_new_app(self, svc, provider, user):
        """create_new_app inserts an app and returns its ID."""

        class FakeManifest:
            app_slug = "new-app"
            name = "New App"
            description = "Created via test"
            version = "0.1.0"

        created_id = await svc.create_new_app(FakeManifest(), str(user["id"]))
        assert created_id is not None

        # Verify it was persisted
        row = await provider.fetch_one(
            "SELECT * FROM apps WHERE app_id = :app_id",
            {"app_id": str(created_id)},
        )
        assert row is not None
        assert row["name"] == "New App"
        assert row["app_slug"] == "new-app"


# ===========================================================================
# AppService integration tests -- PostgreSQL
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestAppServicePostgreSQL:
    """AppService -> NexusQLProvider -> PostgreSQL."""

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        # Drop app-related tables that conftest does not know about
        for table in [
            "app_installations", "app_versions", "fields", "models", "apps",
            "organization_members", "organizations", "users",
        ]:
            try:
                await pg_provider.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            except Exception:
                pass
        await _setup_pg_schema(pg_provider)
        return pg_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return AppService(provider)

    @pytest_asyncio.fixture
    async def user(self, provider):
        return await _insert_user(provider, email="pgappowner@example.com", username="pgappowner")

    @pytest_asyncio.fixture
    async def app(self, provider, user):
        return await _insert_app(
            provider,
            app_slug="pg-app",
            name="PG App",
            description="PostgreSQL integration test app",
            version="1.0.0",
            creator_user_id=user["id"],
        )

    @pytest_asyncio.fixture
    async def model(self, provider, app):
        return await _insert_model(
            provider,
            app_id=app["app_id"],
            model_slug="tickets",
            name="Tickets",
            description="Ticket model",
        )

    @pytest_asyncio.fixture
    async def fields(self, provider, model):
        f1 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="id", name="ID",
                                  is_primary_key=True, data_type="integer",
                                  is_required=True, is_unique=True)
        f2 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="summary", name="Summary",
                                  data_type="string", is_required=True,
                                  validations_json='{"max_length": 200}')
        f3 = await _insert_field(provider, model_id=model["model_id"],
                                  field_column="resolved", name="Resolved",
                                  data_type="boolean", is_required=False,
                                  default_value_json='false')
        return [f1, f2, f3]

    # -- get_app_by_id -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_app_by_id(self, svc, app):
        result = await svc.get_app_by_id(app["app_id"])
        assert result["app_id"] == app["app_id"]
        assert result["name"] == "PG App"

    @pytest.mark.asyncio
    async def test_get_app_by_id_not_found(self, svc):
        with pytest.raises(ValueError, match="not found"):
            await svc.get_app_by_id("pg-nonexistent")

    # -- get_app_by_slug -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_app_by_slug(self, svc, app):
        result = await svc.get_app_by_slug("pg-app")
        assert result["app_id"] == app["app_id"]

    @pytest.mark.asyncio
    async def test_get_app_by_slug_not_found(self, svc):
        with pytest.raises(ValueError, match="not found"):
            await svc.get_app_by_slug("pg-no-slug")

    # -- get_model_by_slug ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_model_by_slug(self, svc, app, model):
        result = await svc.get_model_by_slug(app["app_id"], "tickets")
        assert result["model_id"] == model["model_id"]
        assert result["name"] == "Tickets"

    @pytest.mark.asyncio
    async def test_get_model_by_slug_not_found(self, svc, app):
        with pytest.raises(ValueError, match="not found"):
            await svc.get_model_by_slug(app["app_id"], "pg-ghost-model")

    # -- get_fields_for_model ------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_fields_for_model(self, svc, model, fields):
        result = await svc.get_fields_for_model(model["model_id"])
        assert len(result) == 3
        columns = {f["field_column"] for f in result}
        assert columns == {"id", "summary", "resolved"}

    @pytest.mark.asyncio
    async def test_get_fields_for_model_parses_json(self, svc, model, fields):
        result = await svc.get_fields_for_model(model["model_id"])
        summary = next(f for f in result if f["field_column"] == "summary")
        assert isinstance(summary["validations_json"], dict)
        assert summary["validations_json"]["max_length"] == 200

    # -- validate_data_against_fields ----------------------------------------

    @pytest.mark.asyncio
    async def test_validate_data_valid(self, svc, model, fields):
        field_defs = await svc.get_fields_for_model(model["model_id"])
        result = await svc.validate_data_against_fields(
            {"id": 1, "summary": "Fix bug"},
            field_defs,
            is_create=False,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_missing_required(self, svc, model, fields):
        field_defs = await svc.get_fields_for_model(model["model_id"])
        with pytest.raises(ValueError, match="Validation failed"):
            await svc.validate_data_against_fields(
                {"resolved": True},
                field_defs,
                is_create=False,
            )

    @pytest.mark.asyncio
    async def test_validate_data_skip_pk_on_create(self, svc, model, fields):
        field_defs = await svc.get_fields_for_model(model["model_id"])
        result = await svc.validate_data_against_fields(
            {"summary": "New ticket"},
            field_defs,
            is_create=True,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_wrong_type(self, svc, model, fields):
        field_defs = await svc.get_fields_for_model(model["model_id"])
        with pytest.raises(ValueError, match="Validation failed"):
            await svc.validate_data_against_fields(
                {"id": 1, "summary": 999},
                field_defs,
                is_create=False,
            )

    # -- check_app_access ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_app_access_creator(self, svc, app, user):
        result = await svc.check_app_access(app["app_id"], user["id"])
        assert result is True

    @pytest.mark.asyncio
    async def test_check_app_access_installed_user(self, svc, provider, app):
        other = await _insert_user(provider, email="pginstaller@example.com", username="pginstaller")
        await _insert_installation(provider, app_id=app["app_id"],
                                   user_id=other["id"], status="active")
        result = await svc.check_app_access(app["app_id"], other["id"])
        assert result is True

    @pytest.mark.asyncio
    async def test_check_app_access_denied(self, svc, provider, app):
        stranger = await _insert_user(provider, email="pgstranger@example.com", username="pgstranger")
        with pytest.raises(ValueError, match="don't have access"):
            await svc.check_app_access(app["app_id"], stranger["id"])

    # -- fetch_app_data ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_app_data(self, svc, app, model, fields):
        result = await svc.fetch_app_data(app["app_id"])
        assert result is not None
        assert result["app_id"] == app["app_id"]
        assert len(result["models"]) == 1
        assert len(result["models"][0]["fields"]) == 3

    @pytest.mark.asyncio
    async def test_fetch_app_data_not_found(self, svc):
        result = await svc.fetch_app_data("pg-nonexistent")
        assert result is None

    # -- create_new_app ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_new_app(self, svc, provider, user):
        class FakeManifest:
            app_slug = "pg-new-app"
            name = "PG New App"
            description = "Created in PG test"
            version = "0.2.0"

        created_id = await svc.create_new_app(FakeManifest(), str(user["id"]))
        assert created_id is not None

        row = await provider.fetch_one(
            "SELECT * FROM apps WHERE app_id = :app_id",
            {"app_id": str(created_id)},
        )
        assert row is not None
        assert row["name"] == "PG New App"
        assert row["app_slug"] == "pg-new-app"
        assert row["version"] == "0.2.0"
