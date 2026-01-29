"""
Integration tests for fiberwise-common service layer against real databases.

Tests the full stack: Service → NexusQLProvider → real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/

In CI (Gitea), the pipeline sets TEST_POSTGRESQL_URL pointing at the service container.
"""
import os
import asyncio
import json
import uuid
import pytest
import pytest_asyncio
from pathlib import Path

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.base_service import ServiceError
from fiberwise_common.services.user_service import UserService
from fiberwise_common.services.organization_service import OrganizationService
from fiberwise_common.services.api_keys_service import ApiKeyService, APIKeyData

# Pre-computed bcrypt hash for "testpassword" — avoids passlib backend issues in tests
_TEST_HASH = "$2b$12$Al0VrEkSchsrQTQ1HfC4EuTT7C47LHoffd6uX35yiVW5QoH68JM86"


# ---------------------------------------------------------------------------
# Schema helpers — create only the tables each test class needs
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
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL
);
"""

# PostgreSQL variants (uses SERIAL, BOOLEAN literals, etc.)
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
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL
);
"""


# Fixtures (sqlite_provider, pg_provider, etc.) are provided by conftest.py


async def _setup_sqlite_schema(provider: NexusQLProvider):
    """Set up tables for SQLite integration tests."""
    await provider.execute(USERS_TABLE_SQLITE)
    await provider.execute(ORGANIZATIONS_TABLE_SQLITE)
    await provider.execute(ORG_MEMBERS_TABLE_SQLITE)
    await provider.execute(API_KEYS_TABLE_SQLITE)


async def _setup_pg_schema(provider: NexusQLProvider):
    """Set up tables for PostgreSQL integration tests."""
    await provider.execute(USERS_TABLE_PG)
    await provider.execute(ORGANIZATIONS_TABLE_PG)
    await provider.execute(ORG_MEMBERS_TABLE_PG)
    await provider.execute(API_KEYS_TABLE_PG)


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


# ===========================================================================
# UserService integration tests — SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestUserServiceSQLite:
    """UserService → NexusQLProvider → SQLite."""

    @pytest_asyncio.fixture
    async def svc(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return UserService(sqlite_provider)

    @pytest.mark.asyncio
    async def test_create_user_and_get_by_email(self, svc, sqlite_provider):
        """create_user persists a row and get_user_by_email retrieves it."""
        user = await svc.create_user({
            "email": "alice@example.com",
            "username": "alice",
            "hashed_password": _TEST_HASH,
            "first_name": "Alice",
            "last_name": "Wonder",
        })
        assert user["email"] == "alice@example.com"
        assert user["username"] == "alice"
        assert user["full_name"] == "Alice Wonder"

        fetched = await svc.get_user_by_email("alice@example.com")
        assert fetched is not None
        assert fetched["id"] == user["id"]

    @pytest.mark.asyncio
    async def test_create_user_also_creates_default_organization(self, svc, sqlite_provider):
        """Creating a user should auto-create a default organization."""
        user = await svc.create_user({
            "email": "bob@example.com",
            "username": "bob",
            "hashed_password": _TEST_HASH,
        })

        # The default org is created with the user as owner
        orgs = await sqlite_provider.fetch_all(
            "SELECT * FROM organizations WHERE created_by = :uid",
            {"uid": user["id"]},
        )
        assert len(orgs) >= 1
        org = orgs[0]
        assert "bob" in org["name"].lower() or "organization" in org["name"].lower()

        # User should be an 'owner' member of that org
        membership = await sqlite_provider.fetch_one(
            "SELECT * FROM organization_members WHERE user_id = :uid AND organization_id = :oid",
            {"uid": user["id"], "oid": org["id"]},
        )
        assert membership is not None
        assert membership["role"] == "owner"

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, svc, sqlite_provider):
        user = await svc.create_user({
            "email": "charlie@example.com",
            "username": "charlie",
            "hashed_password": _TEST_HASH,
        })
        fetched = await svc.get_user_by_id(user["id"])
        assert fetched["username"] == "charlie"

    @pytest.mark.asyncio
    async def test_get_user_by_username(self, svc):
        await svc.create_user({
            "email": "dave@example.com",
            "username": "dave",
            "hashed_password": _TEST_HASH,
        })
        fetched = await svc.get_user_by_username("dave")
        assert fetched is not None
        assert fetched["email"] == "dave@example.com"

    @pytest.mark.asyncio
    async def test_get_nonexistent_user_returns_none(self, svc):
        assert await svc.get_user_by_id(99999) is None
        assert await svc.get_user_by_email("noone@nowhere.com") is None

    @pytest.mark.asyncio
    async def test_duplicate_email_raises_validation_error(self, svc):
        from fiberwise_common.services.base_service import ValidationError
        await svc.create_user({
            "email": "dup@example.com", "username": "dup1", "hashed_password": _TEST_HASH,
        })
        with pytest.raises(ValidationError, match="already registered"):
            await svc.create_user({
                "email": "dup@example.com", "username": "dup2", "hashed_password": _TEST_HASH,
            })

    @pytest.mark.asyncio
    async def test_update_user(self, svc):
        user = await svc.create_user({
            "email": "eve@example.com", "username": "eve", "hashed_password": _TEST_HASH,
        })
        updated = await svc.update_user(user["id"], {"first_name": "Eve", "last_name": "Star"})
        assert updated["first_name"] == "Eve"
        assert updated["last_name"] == "Star"

    @pytest.mark.asyncio
    async def test_delete_user_soft_deletes(self, svc):
        user = await svc.create_user({
            "email": "frank@example.com", "username": "frank", "hashed_password": _TEST_HASH,
        })
        result = await svc.delete_user(user["id"])
        assert result is True

        deleted = await svc.get_user_by_id(user["id"])
        # Soft delete sets is_active = 0/False
        assert deleted["is_active"] in (0, False, "0")

    @pytest.mark.asyncio
    async def test_concurrent_user_lookups(self, svc):
        """Multiple concurrent reads should all succeed."""
        await svc.create_user({
            "email": "grace@example.com", "username": "grace", "hashed_password": _TEST_HASH,
        })
        results = await asyncio.gather(
            svc.get_user_by_email("grace@example.com"),
            svc.get_user_by_username("grace"),
            svc.get_user_by_email("grace@example.com"),
        )
        assert all(r is not None for r in results)
        assert all(r["username"] == "grace" for r in results)


# ===========================================================================
# OrganizationService integration tests — SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestOrganizationServiceSQLite:
    """OrganizationService → NexusQLProvider → SQLite."""

    @pytest_asyncio.fixture
    async def provider(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return OrganizationService(provider)

    @pytest_asyncio.fixture
    async def owner(self, provider):
        return await _insert_user_directly(provider, email="owner@example.com", username="owner")

    @pytest_asyncio.fixture
    async def member_user(self, provider):
        return await _insert_user_directly(provider, email="member@example.com", username="member")

    @pytest.mark.asyncio
    async def test_create_organization(self, svc, owner):
        org = await svc.create_organization(
            name="Acme Corp",
            display_name="Acme Corporation",
            created_by=owner["id"],
        )
        assert org is not None
        assert org["name"] == "Acme Corp"
        assert org["slug"] == "acme-corp"

    @pytest.mark.asyncio
    async def test_get_organization(self, svc, owner):
        org = await svc.create_organization(name="TestOrg", created_by=owner["id"])
        fetched = await svc.get_organization(org["id"])
        assert fetched is not None
        assert fetched["name"] == "TestOrg"

    @pytest.mark.asyncio
    async def test_update_organization(self, svc, owner):
        org = await svc.create_organization(name="OldName", created_by=owner["id"])
        updated = await svc.update_organization(org["id"], {"name": "NewName", "description": "Updated"})
        assert updated["name"] == "NewName"
        assert updated["description"] == "Updated"

    @pytest.mark.asyncio
    async def test_add_member_and_get_user_orgs(self, svc, owner, member_user):
        org = await svc.create_organization(name="TeamOrg", created_by=owner["id"])

        # Add owner as member first
        await svc.add_member(org["id"], owner["id"], "owner", invited_by=owner["id"])

        # Add another member
        await svc.add_member(org["id"], member_user["id"], "member", invited_by=owner["id"])

        # Member should see the org
        orgs = await svc.get_user_organizations(member_user["id"])
        assert len(orgs) == 1
        assert orgs[0]["name"] == "TeamOrg"

    @pytest.mark.asyncio
    async def test_slug_uniqueness(self, svc, owner):
        """Creating orgs with the same name should generate unique slugs."""
        org1 = await svc.create_organization(name="Duplicate", created_by=owner["id"])
        org2 = await svc.create_organization(name="Duplicate", created_by=owner["id"])
        assert org1["slug"] != org2["slug"]
        assert org2["slug"].startswith("duplicate-")


# ===========================================================================
# Full user-creation flow — SQLite
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestUserCreationFlowSQLite:
    """Test the Joab-style flow: create user → org → membership."""

    @pytest_asyncio.fixture
    async def provider(self, sqlite_provider):
        await _setup_sqlite_schema(sqlite_provider)
        return sqlite_provider

    @pytest.mark.asyncio
    async def test_full_user_onboarding_flow(self, provider):
        """
        Mimics what happens when a new user signs up:
        1. UserService.create_user is called
        2. A default organization is created automatically
        3. The user becomes the owner of that organization
        4. Another user can be added to the organization
        """
        user_svc = UserService(provider)
        org_svc = OrganizationService(provider)

        # Step 1: Create the first user (like Joab signing up)
        joab = await user_svc.create_user({
            "email": "joab@fiberwise.ai",
            "username": "joab",
            "hashed_password": _TEST_HASH,
            "first_name": "Joab",
            "last_name": "Agent",
        })
        assert joab["email"] == "joab@fiberwise.ai"
        assert joab["full_name"] == "Joab Agent"

        # Step 2: Verify the default org was created
        orgs = await provider.fetch_all(
            "SELECT * FROM organizations WHERE created_by = :uid",
            {"uid": joab["id"]},
        )
        assert len(orgs) >= 1
        default_org = orgs[0]

        # Step 3: Verify Joab is the owner
        membership = await provider.fetch_one(
            "SELECT * FROM organization_members WHERE user_id = :uid AND organization_id = :oid",
            {"uid": joab["id"], "oid": default_org["id"]},
        )
        assert membership is not None
        assert membership["role"] == "owner"
        assert membership["status"] == "active"

        # Step 4: Create a second user and add them to Joab's org
        team_member = await user_svc.create_user({
            "email": "colleague@fiberwise.ai",
            "username": "colleague",
            "hashed_password": _TEST_HASH,
        })

        await org_svc.add_member(
            default_org["id"],
            team_member["id"],
            "member",
            invited_by=joab["id"],
        )

        # Step 5: Verify the colleague can see Joab's org
        colleague_orgs = await org_svc.get_user_organizations(team_member["id"])
        org_ids = [o["id"] for o in colleague_orgs]
        assert default_org["id"] in org_ids

        # Step 6: Verify user lookup works
        found = await user_svc.get_user_by_email_or_username("joab")
        assert found is not None
        assert found["id"] == joab["id"]

        found2 = await user_svc.get_user_by_email_or_username("joab@fiberwise.ai")
        assert found2 is not None
        assert found2["id"] == joab["id"]


# ===========================================================================
# PostgreSQL variants — only run when TEST_POSTGRESQL_URL is set
# ===========================================================================

@pytest.mark.integration
@pytest.mark.database
class TestUserServicePostgreSQL:
    """UserService → NexusQLProvider → PostgreSQL."""

    @pytest_asyncio.fixture
    async def svc(self, pg_provider):
        await _setup_pg_schema(pg_provider)
        return UserService(pg_provider)

    @pytest.mark.asyncio
    async def test_create_user_and_get_by_email(self, svc):
        user = await svc.create_user({
            "email": "pgalice@example.com",
            "username": "pgalice",
            "hashed_password": _TEST_HASH,
            "first_name": "Alice",
            "last_name": "PG",
        })
        assert user["email"] == "pgalice@example.com"
        fetched = await svc.get_user_by_email("pgalice@example.com")
        assert fetched is not None
        assert fetched["id"] == user["id"]

    @pytest.mark.asyncio
    async def test_create_user_creates_org(self, svc, pg_provider):
        user = await svc.create_user({
            "email": "pgbob@example.com",
            "username": "pgbob",
            "hashed_password": _TEST_HASH,
        })
        orgs = await pg_provider.fetch_all(
            "SELECT * FROM organizations WHERE created_by = :uid",
            {"uid": user["id"]},
        )
        assert len(orgs) >= 1

    @pytest.mark.asyncio
    async def test_duplicate_email_raises(self, svc):
        from fiberwise_common.services.base_service import ValidationError
        await svc.create_user({
            "email": "pgdup@example.com", "username": "pgdup1", "hashed_password": _TEST_HASH,
        })
        with pytest.raises(ValidationError, match="already registered"):
            await svc.create_user({
                "email": "pgdup@example.com", "username": "pgdup2", "hashed_password": _TEST_HASH,
            })

    @pytest.mark.asyncio
    async def test_update_and_delete_user(self, svc):
        user = await svc.create_user({
            "email": "pgeve@example.com", "username": "pgeve", "hashed_password": _TEST_HASH,
        })
        updated = await svc.update_user(user["id"], {"first_name": "Eve"})
        assert updated["first_name"] == "Eve"

        await svc.delete_user(user["id"])
        deleted = await svc.get_user_by_id(user["id"])
        assert deleted["is_active"] in (0, False, "0")


@pytest.mark.integration
@pytest.mark.database
class TestOrganizationServicePostgreSQL:
    """OrganizationService → NexusQLProvider → PostgreSQL."""

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        await _setup_pg_schema(pg_provider)
        return pg_provider

    @pytest_asyncio.fixture
    async def svc(self, provider):
        return OrganizationService(provider)

    @pytest_asyncio.fixture
    async def owner(self, provider):
        return await _insert_user_directly(provider, email="pgowner@example.com", username="pgowner")

    @pytest.mark.asyncio
    async def test_create_and_add_member(self, svc, owner, provider):
        member = await _insert_user_directly(provider, email="pgmember@example.com", username="pgmember")
        org = await svc.create_organization(name="PG Corp", created_by=owner["id"])
        await svc.add_member(org["id"], owner["id"], "owner", invited_by=owner["id"])
        await svc.add_member(org["id"], member["id"], "member", invited_by=owner["id"])

        orgs = await svc.get_user_organizations(member["id"])
        assert len(orgs) == 1
        assert orgs[0]["name"] == "PG Corp"


@pytest.mark.integration
@pytest.mark.database
class TestFullFlowPostgreSQL:
    """Full user onboarding flow against PostgreSQL."""

    @pytest_asyncio.fixture
    async def provider(self, pg_provider):
        await _setup_pg_schema(pg_provider)
        return pg_provider

    @pytest.mark.asyncio
    async def test_joab_onboarding(self, provider):
        user_svc = UserService(provider)
        org_svc = OrganizationService(provider)

        joab = await user_svc.create_user({
            "email": "pgjoab@fiberwise.ai",
            "username": "pgjoab",
            "hashed_password": _TEST_HASH,
            "first_name": "Joab",
            "last_name": "Agent",
        })

        orgs = await provider.fetch_all(
            "SELECT * FROM organizations WHERE created_by = :uid",
            {"uid": joab["id"]},
        )
        assert len(orgs) >= 1

        membership = await provider.fetch_one(
            "SELECT * FROM organization_members WHERE user_id = :uid",
            {"uid": joab["id"]},
        )
        assert membership is not None
        assert membership["role"] == "owner"

        colleague = await user_svc.create_user({
            "email": "pgcolleague@fiberwise.ai",
            "username": "pgcolleague",
            "hashed_password": _TEST_HASH,
        })

        await org_svc.add_member(
            orgs[0]["id"], colleague["id"], "member", invited_by=joab["id"],
        )

        colleague_orgs = await org_svc.get_user_organizations(colleague["id"])
        assert len(colleague_orgs) >= 1
