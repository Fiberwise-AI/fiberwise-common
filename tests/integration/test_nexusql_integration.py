"""
Integration tests for the fiberwise-common service layer.

Tests the full stack: BaseService → NexusQLProvider → real database.
SQLite tests always run. PostgreSQL tests run when TEST_POSTGRESQL_URL is set.

To run with PostgreSQL:
    TEST_POSTGRESQL_URL=postgresql://user:pass@localhost/test_fiberwise pytest tests/integration/
"""
import os
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.services.base_service import BaseService, ServiceError

# Fixtures (sqlite_provider, pg_provider, etc.) are provided by conftest.py


# ---------------------------------------------------------------------------
# Example service used by tests
# ---------------------------------------------------------------------------

class UserService(BaseService):
    """Example service for integration testing."""

    async def setup_schema(self):
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)

    async def setup_schema_pg(self):
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)

    async def create_user(self, username: str, email: str) -> dict:
        await self._execute(
            "INSERT INTO users (username, email) VALUES (:username, :email)",
            {"username": username, "email": email},
        )
        return await self._fetch_one(
            "SELECT * FROM users WHERE username = :username", {"username": username}
        )

    async def get_user(self, user_id: int) -> dict:
        return await self._fetch_one(
            "SELECT * FROM users WHERE id = :id", {"id": user_id}
        )

    async def list_users(self) -> list:
        return await self._fetch_all("SELECT * FROM users ORDER BY id")

    async def update_email(self, user_id: int, email: str):
        await self._execute(
            "UPDATE users SET email = :email WHERE id = :id",
            {"email": email, "id": user_id},
        )

    async def delete_user(self, user_id: int):
        await self._execute(
            "DELETE FROM users WHERE id = :id", {"id": user_id}
        )


# ---------------------------------------------------------------------------
# BaseService integration tests (SQLite — always run)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.database
class TestBaseServiceSQLite:
    """End-to-end: BaseService → NexusQLProvider → SQLite."""

    @pytest_asyncio.fixture
    async def user_service(self, sqlite_provider):
        svc = UserService(sqlite_provider)
        await svc.setup_schema()
        return svc

    @pytest.mark.asyncio
    async def test_create_and_get_user(self, user_service):
        created = await user_service.create_user("alice", "alice@example.com")
        assert created["username"] == "alice"
        assert created["email"] == "alice@example.com"

        fetched = await user_service.get_user(created["id"])
        assert fetched["username"] == "alice"

    @pytest.mark.asyncio
    async def test_list_users(self, user_service):
        await user_service.create_user("alice", "a@test.com")
        await user_service.create_user("bob", "b@test.com")
        await user_service.create_user("charlie", "c@test.com")

        users = await user_service.list_users()
        assert len(users) == 3
        assert [u["username"] for u in users] == ["alice", "bob", "charlie"]

    @pytest.mark.asyncio
    async def test_update_email(self, user_service):
        user = await user_service.create_user("alice", "old@test.com")
        await user_service.update_email(user["id"], "new@test.com")
        updated = await user_service.get_user(user["id"])
        assert updated["email"] == "new@test.com"

    @pytest.mark.asyncio
    async def test_delete_user(self, user_service):
        user = await user_service.create_user("alice", "a@test.com")
        await user_service.delete_user(user["id"])
        assert await user_service.get_user(user["id"]) is None

    @pytest.mark.asyncio
    async def test_unique_constraint_raises_service_error(self, user_service):
        await user_service.create_user("alice", "a@test.com")
        with pytest.raises(ServiceError):
            await user_service.create_user("alice", "different@test.com")

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, user_service):
        await user_service.create_user("alice", "a@test.com")
        results = await asyncio.gather(
            user_service.list_users(),
            user_service.list_users(),
            user_service.list_users(),
        )
        assert all(len(r) == 1 for r in results)


# ---------------------------------------------------------------------------
# BaseService integration tests (PostgreSQL — only when env var is set)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.database
class TestBaseServicePostgreSQL:
    """End-to-end: BaseService → NexusQLProvider → PostgreSQL."""

    @pytest_asyncio.fixture
    async def user_service(self, pg_provider):
        svc = UserService(pg_provider)
        await svc.setup_schema_pg()
        return svc

    @pytest.mark.asyncio
    async def test_create_and_get_user(self, user_service):
        created = await user_service.create_user("alice", "alice@example.com")
        assert created["username"] == "alice"
        assert created["email"] == "alice@example.com"

        fetched = await user_service.get_user(created["id"])
        assert fetched["username"] == "alice"

    @pytest.mark.asyncio
    async def test_list_users(self, user_service):
        await user_service.create_user("alice", "a@test.com")
        await user_service.create_user("bob", "b@test.com")

        users = await user_service.list_users()
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_unique_constraint_raises_service_error(self, user_service):
        await user_service.create_user("alice", "a@test.com")
        with pytest.raises(ServiceError):
            await user_service.create_user("alice", "different@test.com")
