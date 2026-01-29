"""
Shared fixtures for integration tests.

Database URLs default to non-default ports matching docker-compose.test.yml.
Tests auto-skip when a database is not reachable.

Local usage:
    docker compose -f tests/docker-compose.test.yml up -d
    pytest tests/integration/ -v

CI usage (Gitea):
    The pipeline overrides TEST_POSTGRESQL_URL to point at the service container.
"""
import os
import pytest
import pytest_asyncio
from pathlib import Path

from fiberwise_common.database.provider import NexusQLProvider

# ---------------------------------------------------------------------------
# Default connection URLs  (non-default ports from docker-compose.test.yml)
# ---------------------------------------------------------------------------

DEFAULT_POSTGRESQL_URL = "postgresql://fiberwise_test:fiberwise_test@localhost:15432/fiberwise_test"


def _get_pg_url() -> str:
    """Return PostgreSQL URL from env or default (non-default port)."""
    return os.getenv("TEST_POSTGRESQL_URL", DEFAULT_POSTGRESQL_URL)


# ---------------------------------------------------------------------------
# SQLite fixtures  (always available)
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    """Fresh SQLite database URL per test."""
    return f"sqlite:///{tmp_path / 'integration_test.db'}"


@pytest_asyncio.fixture
async def sqlite_provider(sqlite_url: str):
    """Connected NexusQLProvider backed by a temp SQLite file."""
    provider = NexusQLProvider(sqlite_url)
    await provider.connect()
    yield provider
    await provider.disconnect()


# ---------------------------------------------------------------------------
# PostgreSQL fixtures  (skip when unreachable)
# ---------------------------------------------------------------------------

@pytest.fixture
def postgresql_url():
    """PostgreSQL connection URL; skips if the server is unreachable."""
    url = _get_pg_url()
    # Quick TCP probe to avoid slow timeouts in test collection
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        sock.close()
    except OSError:
        pytest.skip(f"PostgreSQL not reachable at {host}:{port}")
    return url


@pytest_asyncio.fixture
async def pg_provider(postgresql_url: str):
    """Connected NexusQLProvider backed by PostgreSQL.

    Cleans up common test tables before and after each test.
    """
    provider = NexusQLProvider(postgresql_url)
    await provider.connect()

    # Tables that integration tests may create — drop in dependency order
    _tables = [
        "agent_api_keys",
        "execution_api_keys",
        "organization_members",
        "api_keys",
        "organizations",
        "users",
        "schema_migrations",
    ]
    for table in _tables:
        try:
            await provider.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        except Exception:
            pass

    yield provider

    for table in _tables:
        try:
            await provider.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        except Exception:
            pass
    await provider.disconnect()
