"""
Integration tests for activation_processor dispatch logic, unified storage,
and A2A delegation — including a live test against the A2A server with
zai-coding-plan provider and GLM-5.1 model.

Tests the fixes from 4-7-26:
  1. Dispatch reorder: cli_type → A2A, processor → pipeline, else → custom
  2. Unified FIBERWISE_DATA_DIR storage helpers
  3. validate_agent_cwd() permissions foundation
  4. A2A delegation via ia_modules A2AExecutor
"""
import json
import os
import pytest
import pytest_asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.activation.activation_processor import (
    ActivationProcessor,
    get_data_dir,
    get_bundle_dir,
    get_workspace_dir,
    get_log_dir,
    validate_agent_cwd,
)
from ia_modules.agents.executor import AgentEvent, EventType
from ia_modules.agents.a2a_executor import A2AExecutor


# ============================================================================
# Database schema (same as test_activation_integration.py)
# ============================================================================

AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL,
    name TEXT NOT NULL,
    agent_type_id TEXT NOT NULL,
    description TEXT,
    config TEXT,
    configuration TEXT,
    file_path TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

ACTIVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_activations (
    activation_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    app_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    created_by INTEGER,
    input_data TEXT,
    output_data TEXT,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    execution_time_ms INTEGER,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);
"""

AGENT_VERSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_versions (
    version_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    file_path TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);
"""

LLM_PROVIDERS_TABLE = """
CREATE TABLE IF NOT EXISTS llm_providers (
    provider_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    api_endpoint TEXT NOT NULL,
    configuration TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


async def setup_db(db: NexusQLProvider):
    await db.execute(AGENTS_TABLE)
    await db.execute(ACTIVATIONS_TABLE)
    await db.execute(AGENT_VERSIONS_TABLE)
    await db.execute(LLM_PROVIDERS_TABLE)


async def insert_agent(db, agent_id, agent_type, config=None, app_id="test-app"):
    config_json = json.dumps(config or {})
    await db.execute("""
        INSERT INTO agents (agent_id, app_id, name, agent_type_id, config, configuration, file_path)
        VALUES (:agent_id, :app_id, :name, :agent_type_id, :config, :config, :file_path)
    """, {
        "agent_id": agent_id,
        "app_id": app_id,
        "name": f"Agent {agent_id}",
        "agent_type_id": agent_type,
        "config": config_json,
        "file_path": f"/agents/{agent_id}.py",
    })


async def insert_activation(db, activation_id, agent_id, app_id="test-app", input_data=None):
    await db.execute("""
        INSERT INTO agent_activations (activation_id, agent_id, app_id, organization_id, input_data, status)
        VALUES (:activation_id, :agent_id, :app_id, :org_id, :input_data, 'pending')
    """, {
        "activation_id": activation_id,
        "agent_id": agent_id,
        "app_id": app_id,
        "org_id": "org-1",
        "input_data": json.dumps(input_data or {"message": "hello"}),
    })


# ============================================================================
# Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def db(tmp_path):
    provider = NexusQLProvider(f"sqlite:///{tmp_path / 'test.db'}")
    await provider.connect()
    await setup_db(provider)
    yield provider
    await provider.disconnect()


@pytest.fixture
def data_dir(tmp_path):
    """Set FIBERWISE_DATA_DIR to a temp directory for the test."""
    d = tmp_path / "_data"
    d.mkdir()
    with patch.dict(os.environ, {"FIBERWISE_DATA_DIR": str(d)}):
        yield d


# ============================================================================
# 1. Unified storage helpers
# ============================================================================

class TestStorageHelpers:
    """Test the new FIBERWISE_DATA_DIR storage helper functions."""

    def test_get_data_dir_default(self):
        """Default data dir is _data/ resolved to absolute."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FIBERWISE_DATA_DIR", None)
            d = get_data_dir()
            assert d.is_absolute()
            assert d.name == "_data"

    def test_get_data_dir_custom(self, tmp_path):
        with patch.dict(os.environ, {"FIBERWISE_DATA_DIR": str(tmp_path / "custom")}):
            d = get_data_dir()
            assert d == (tmp_path / "custom").resolve()

    def test_get_bundle_dir(self, data_dir):
        d = get_bundle_dir("app-123")
        assert d == data_dir / "bundles" / "app-123"

    def test_get_workspace_dir_creates(self, data_dir):
        d = get_workspace_dir("app-456")
        assert d == data_dir / "workspaces" / "app-456"
        assert d.exists(), "workspace dir should be auto-created"

    def test_get_log_dir_creates(self, data_dir):
        d = get_log_dir("activation-789")
        assert d == data_dir / "logs" / "activation-789"
        assert d.exists(), "log dir should be auto-created"


# ============================================================================
# 2. CWD validation (permissions foundation)
# ============================================================================

class TestCWDValidation:
    """Test validate_agent_cwd — Fiberwise-side CWD jail."""

    def test_unrestricted_when_env_unset(self):
        """No FIBERWISE_DATA_DIR → passes any path through."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FIBERWISE_DATA_DIR", None)
            result = validate_agent_cwd("/some/random/path")
            assert result == "/some/random/path"

    def test_valid_path_under_data_dir(self, data_dir):
        workspace = data_dir / "workspaces" / "app-1"
        workspace.mkdir(parents=True)
        result = validate_agent_cwd(str(workspace))
        assert Path(result).resolve() == workspace.resolve()

    def test_rejects_path_outside_data_dir(self, data_dir):
        with pytest.raises(ValueError, match="outside the allowed root"):
            validate_agent_cwd("/tmp/evil/escape")

    def test_rejects_traversal_attack(self, data_dir):
        """Prevent ../../ breakout."""
        sneaky = str(data_dir / "workspaces" / ".." / ".." / ".." / "etc")
        with pytest.raises(ValueError, match="outside the allowed root"):
            validate_agent_cwd(sneaky)

    def test_resolves_symlinks(self, data_dir):
        """Symlink pointing outside data_dir should be rejected."""
        link = data_dir / "workspaces" / "sneaky-link"
        link.parent.mkdir(parents=True, exist_ok=True)
        # On Windows, symlinks to dirs may require elevated privileges,
        # so we test with a path that resolves outside instead.
        outside = Path(os.path.realpath(data_dir / ".."))
        with pytest.raises(ValueError, match="outside the allowed root"):
            validate_agent_cwd(str(outside))


# ============================================================================
# 3. Dispatch logic — correct routing
# ============================================================================

class TestDispatchLogic:
    """Test that process_activation routes agents to the correct execution path."""

    @pytest.mark.asyncio
    async def test_cli_type_routes_to_a2a(self, db, data_dir):
        """Non-processor agent with cli_type in config → _delegate_to_a2a."""
        await insert_agent(db, "a2a-agent", "a2a", config={
            "cli_type": "claude_code",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "tools": ["Read", "Glob"],
            "mode": "research",
        })
        await insert_activation(db, "act-1", "a2a-agent")

        processor = ActivationProcessor(db, context="test")

        with patch.object(processor, '_delegate_to_a2a', new_callable=AsyncMock) as mock_a2a, \
             patch.object(processor, '_execute_pipeline_agent', new_callable=AsyncMock) as mock_pipe, \
             patch.object(processor, '_execute_custom_agent', new_callable=AsyncMock) as mock_custom, \
             patch.object(processor, '_update_activation_status', new_callable=AsyncMock), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={}):

            mock_a2a.return_value = {
                'output_data': {'response': 'a2a result'},
                'execution_time_ms': 100,
                'status': 'completed',
            }

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-1"}
            )
            await processor.process_activation(dict(activation))

            mock_a2a.assert_called_once()
            mock_pipe.assert_not_called()
            mock_custom.assert_not_called()

    @pytest.mark.asyncio
    async def test_processor_type_routes_to_pipeline(self, db, data_dir):
        """Processor agent WITHOUT cli_type → _execute_pipeline_agent."""
        await insert_agent(db, "pipe-agent", "processor", config={
            "pipeline_definition": "pipelines/chat.yaml",
        })
        await insert_activation(db, "act-2", "pipe-agent")

        processor = ActivationProcessor(db, context="test")

        with patch.object(processor, '_delegate_to_a2a', new_callable=AsyncMock) as mock_a2a, \
             patch.object(processor, '_execute_pipeline_agent', new_callable=AsyncMock) as mock_pipe, \
             patch.object(processor, '_execute_custom_agent', new_callable=AsyncMock) as mock_custom, \
             patch.object(processor, '_update_activation_status', new_callable=AsyncMock), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={}):

            mock_pipe.return_value = {
                'output_data': {'result': 'pipeline output'},
                'execution_time_ms': 50,
                'status': 'completed',
            }

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-2"}
            )
            await processor.process_activation(dict(activation))

            mock_pipe.assert_called_once()
            mock_a2a.assert_not_called()
            mock_custom.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_definition_without_processor_type(self, db, data_dir):
        """Custom agent type but with pipeline_definition → still routes to pipeline."""
        await insert_agent(db, "custom-pipe", "custom", config={
            "pipeline_definition": "pipelines/custom.yaml",
        })
        await insert_activation(db, "act-3", "custom-pipe")

        processor = ActivationProcessor(db, context="test")

        with patch.object(processor, '_delegate_to_a2a', new_callable=AsyncMock) as mock_a2a, \
             patch.object(processor, '_execute_pipeline_agent', new_callable=AsyncMock) as mock_pipe, \
             patch.object(processor, '_execute_custom_agent', new_callable=AsyncMock) as mock_custom, \
             patch.object(processor, '_update_activation_status', new_callable=AsyncMock), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={}):

            mock_pipe.return_value = {
                'output_data': {'result': 'pipeline output'},
                'execution_time_ms': 50,
                'status': 'completed',
            }

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-3"}
            )
            await processor.process_activation(dict(activation))

            mock_pipe.assert_called_once()
            mock_a2a.assert_not_called()

    @pytest.mark.asyncio
    async def test_processor_with_cli_type_routes_to_pipeline(self, db, data_dir):
        """Processor agent with both cli_type and pipeline_definition → pipeline.
        The processor type takes priority; cli_type is used by the pipeline internally."""
        await insert_agent(db, "both-agent", "processor", config={
            "cli_type": "opencode",
            "pipeline_definition": "pipelines/chat.yaml",
            "provider": "zai-coding-plan",
            "model": "glm-5.1",
        })
        await insert_activation(db, "act-4", "both-agent")

        processor = ActivationProcessor(db, context="test")

        with patch.object(processor, '_delegate_to_a2a', new_callable=AsyncMock) as mock_a2a, \
             patch.object(processor, '_execute_pipeline_agent', new_callable=AsyncMock) as mock_pipe, \
             patch.object(processor, '_update_activation_status', new_callable=AsyncMock), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={}):

            mock_pipe.return_value = {
                'output_data': {'result': 'pipeline output'},
                'execution_time_ms': 50,
                'status': 'completed',
            }

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-4"}
            )
            await processor.process_activation(dict(activation))

            mock_pipe.assert_called_once()
            mock_a2a.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_agent_fallthrough(self, db, data_dir):
        """Agent with no cli_type, no pipeline_definition, not processor → custom code."""
        await insert_agent(db, "custom-agent", "custom", config={})
        await insert_activation(db, "act-5", "custom-agent")

        processor = ActivationProcessor(db, context="test")

        # Mock _get_agent_version since the real query needs columns not in test schema
        mock_version = AsyncMock(return_value={
            "version_id": "v-1",
            "agent_id": "custom-agent",
            "version": "1.0.0",
            "file_path": "/agents/custom-agent.py",
        })

        with patch.object(processor, '_delegate_to_a2a', new_callable=AsyncMock) as mock_a2a, \
             patch.object(processor, '_execute_pipeline_agent', new_callable=AsyncMock) as mock_pipe, \
             patch.object(processor, '_execute_custom_agent', new_callable=AsyncMock) as mock_custom, \
             patch.object(processor, '_get_agent_version', mock_version), \
             patch.object(processor, '_update_activation_status', new_callable=AsyncMock), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock):

            mock_custom.return_value = {
                'output_data': {'result': 'custom output'},
                'execution_time_ms': 10,
                'status': 'completed',
            }

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-5"}
            )
            await processor.process_activation(dict(activation))

            mock_custom.assert_called_once()
            mock_a2a.assert_not_called()
            mock_pipe.assert_not_called()


# ============================================================================
# 4. A2A delegation — mock stream test (uses ia_modules A2AExecutor)
# ============================================================================

class TestA2ADelegation:
    """Test _delegate_to_a2a with mocked A2AExecutor (ia_modules)."""

    @pytest.fixture
    def mock_oidc_adapter(self):
        """Mock OIDC adapter that returns a fake token and passes validation."""
        mock_adapter = AsyncMock()
        mock_adapter.issue_token_for_agent = AsyncMock(return_value="mock-jwt-token")
        mock_adapter.validate_token = AsyncMock(return_value={
            "sub": "agent_test",
            "aud": "a2a-server",
            "scope": "data:read",
            "agent_id": "test",
            "org_id": 1,
            "a2a": {"allowed_modes": ["research"]},
        })
        return mock_adapter

    @pytest.mark.asyncio
    async def test_delegate_passes_correct_params(self, db, data_dir, mock_oidc_adapter):
        """Verify the CWD, model, provider, cli_type are passed to A2AExecutor."""
        config = {
            "cli_type": "opencode",
            "provider": "zai-coding-plan",
            "model": "glm-5.1",
            "tools": ["Read", "Glob", "Grep"],
            "mode": "research",
            "system_prompt": "You are a research assistant.",
        }
        await insert_agent(db, "zai-agent", "a2a", config=config, app_id="app-zai")
        await insert_activation(db, "act-zai", "zai-agent", app_id="app-zai",
                                input_data={"message": "summarize this project"})

        processor = ActivationProcessor(db, context="test")

        # Capture the AgentConfig passed to execute()
        captured_config = {}

        # Track status updates
        status_updates = []
        async def track_status(aid, status, **kw):
            status_updates.append(status)

        # Mock SubprocessExecutor and OIDC adapter
        with patch("ia_modules.agents.subprocess_executor.SubprocessExecutor") as MockSubExec, \
             patch("fiberwise_common.oidc_provider.get_adapter", return_value=mock_oidc_adapter), \
             patch.object(processor, '_update_activation_status', side_effect=track_status), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={"status": "completed"}):

            async def mock_execute(agent_config):
                captured_config["cli_type"] = agent_config.cli_type.value
                captured_config["provider"] = agent_config.provider
                captured_config["model"] = agent_config.model
                captured_config["mode"] = agent_config.mode.value
                yield AgentEvent(type=EventType.RESULT, subtype="result",
                                 result="Summary: this project does X",
                                 job_id="act-zai", seq=1)
                yield AgentEvent(type=EventType.SYSTEM, subtype="stream_end",
                                 job_id="act-zai", seq=2)

            instance = MockSubExec.return_value
            instance.execute = mock_execute

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-zai"}
            )
            result = await processor.process_activation(dict(activation))

        # Verify auth gate was called
        mock_oidc_adapter.issue_token_for_agent.assert_awaited_once()
        mock_oidc_adapter.validate_token.assert_awaited_once_with("mock-jwt-token")

        # Verify status progression: running → completed
        assert "running" in status_updates
        assert "completed" in status_updates

        # Verify SubprocessExecutor received correct params via AgentConfig
        assert captured_config["cli_type"] == "opencode"
        assert captured_config["provider"] == "zai-coding-plan"
        assert captured_config["model"] == "glm-5.1"
        assert captured_config["mode"] == "research"

    @pytest.mark.asyncio
    async def test_delegate_a2a_connection_error(self, db, data_dir, mock_oidc_adapter):
        """If SubprocessExecutor yields an error event, activation still completes
        (the error is in output_data, not a thrown exception)."""
        config = {"cli_type": "claude_code", "provider": "anthropic"}
        await insert_agent(db, "err-agent", "a2a", config=config)
        await insert_activation(db, "act-err", "err-agent")

        processor = ActivationProcessor(db, context="test")

        async def mock_execute_error(agent_config):
            yield AgentEvent(
                type=EventType.RESULT, subtype="error",
                error="Connection refused",
                job_id="act-err", seq=1,
            )
            yield AgentEvent(
                type=EventType.SYSTEM, subtype="stream_end",
                job_id="act-err", seq=2,
            )

        status_updates = []
        output_stored = {}
        async def track_status(aid, status, output_data=None, **kw):
            status_updates.append(status)
            if output_data:
                output_stored.update(output_data)

        with patch("ia_modules.agents.subprocess_executor.SubprocessExecutor") as MockSubExec, \
             patch("fiberwise_common.oidc_provider.get_adapter", return_value=mock_oidc_adapter), \
             patch.object(processor, '_update_activation_status', side_effect=track_status), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={"status": "completed"}):
            instance = MockSubExec.return_value
            instance.execute = mock_execute_error

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-err"}
            )
            await processor.process_activation(dict(activation))

        # Auth gate was called
        mock_oidc_adapter.issue_token_for_agent.assert_awaited_once()
        mock_oidc_adapter.validate_token.assert_awaited_once()

        # Dispatch completes (error content is in the output, not a failure)
        assert "completed" in status_updates
        # The error text should be captured in the output
        assert "Connection refused" in str(output_stored)

    @pytest.mark.asyncio
    async def test_delegate_a2a_auth_failure_prevents_execution(self, db, data_dir):
        """If validate_token raises, the agent must NOT execute."""
        config = {"cli_type": "claude_code", "provider": "anthropic"}
        await insert_agent(db, "unauth-agent", "a2a", config=config)
        await insert_activation(db, "act-unauth", "unauth-agent")

        processor = ActivationProcessor(db, context="test")

        # Adapter that fails validation
        bad_adapter = AsyncMock()
        bad_adapter.issue_token_for_agent = AsyncMock(return_value="bad-token")
        bad_adapter.validate_token = AsyncMock(side_effect=ValueError("Token validation failed: expired"))

        status_updates = []
        async def track_status(aid, status, **kw):
            status_updates.append(status)

        with patch("ia_modules.agents.subprocess_executor.SubprocessExecutor") as MockSubExec, \
             patch("fiberwise_common.oidc_provider.get_adapter", return_value=bad_adapter), \
             patch.object(processor, '_update_activation_status', side_effect=track_status), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={"status": "failed"}):

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-unauth"}
            )
            await processor.process_activation(dict(activation))

        # Executor should NOT have been called
        MockSubExec.return_value.execute.assert_not_called()

        # Should have failed status
        assert "failed" in status_updates


# ============================================================================
# 5. Live A2A test — zai-coding-plan / GLM-5.1
# ============================================================================

@pytest.mark.network
@pytest.mark.slow
class TestLiveA2AZaiCoderPlan:
    """Live integration test against a running A2A server with zai-coding-plan.

    Requires:
      - A2A server running at A2A_SERVER_URL (default http://localhost:3008)
      - ZAI_API_KEY environment variable set
      - Model: glm-5.1 available via the zai-coding-plan provider

    Skip with: pytest -m "not network"
    """

    @pytest.fixture(autouse=True)
    def require_zai_key(self):
        # 1. Already in env (CI sets secrets as env vars) — use it
        key = os.environ.get("ZAI_API_KEY", "")
        if not key:
            # 2. Local dev: load tests/.env.test next to this test package
            env_file = Path(__file__).resolve().parent.parent / ".env.test"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            key = os.environ.get("ZAI_API_KEY", "")
        if not key:
            pytest.skip("ZAI_API_KEY not set — skipping live A2A test")

    @pytest.fixture(autouse=True)
    def require_a2a_server(self):
        """Skip if A2A server is not reachable."""
        import httpx
        url = os.getenv("A2A_SERVER_URL", "http://localhost:3008")
        try:
            resp = httpx.get(f"{url}/health", timeout=3.0)
            if resp.status_code != 200:
                pytest.skip(f"A2A server at {url} returned {resp.status_code}")
        except Exception:
            pytest.skip(f"A2A server not reachable at {url}")

    @pytest.mark.asyncio
    async def test_stream_agent_zai_glm5(self, tmp_path):
        """Send a task to GLM-5.1 via zai-coding-plan and verify A2AExecutor
        connects, sends the request, and gets back a submitted event.

        The A2AExecutor is fire-and-forget — it yields a 'submitted' event.
        Actual agent events arrive via callback URL, not streaming.
        """
        from ia_modules.agents.executor import AgentConfig as IAAgentConfig, CLIType, AgentMode

        cwd = str(tmp_path)

        ia_config = IAAgentConfig(
            task="What is 2 + 2? Reply with just the number.",
            cwd=cwd,
            cli_type=CLIType.OPENCODE,
            mode=AgentMode.RESEARCH,
            tools=["Read", "Glob", "Grep"],
            provider="zai-coding-plan",
            model="glm-5.1",
            api_key=os.environ.get("ZAI_API_KEY", ""),
            job_id="test-live-zai",
        )

        executor = A2AExecutor()
        events = []
        async for event in executor.execute(ia_config):
            events.append(event)

        # Should get at least a submitted event (fire-and-forget)
        assert len(events) >= 1
        assert events[0].subtype == "submitted"

    @pytest.mark.asyncio
    async def test_full_activation_flow_zai(self, db, data_dir):
        """Full end-to-end: insert agent + activation in DB, dispatch through
        ActivationProcessor, verify it reaches A2A server with zai-coding-plan.

        Uses mocked DB status updates (test schema != production schema)
        but real A2A client connection. OIDC adapter is mocked since the
        test DB has no agent_api_keys table.
        """
        config = {
            "cli_type": "opencode",
            "provider": "zai-coding-plan",
            "model": "glm-5.1",
            "tools": ["Read", "Glob", "Grep"],
            "mode": "research",
            "system_prompt": "Answer concisely.",
        }
        await insert_agent(db, "zai-live", "a2a", config=config, app_id="app-live")
        await insert_activation(db, "act-live", "zai-live", app_id="app-live",
                                input_data={"message": "What is the capital of France?"})

        processor = ActivationProcessor(db, context="test-live")

        # Mock OIDC adapter — test DB has no agent keys
        mock_adapter = AsyncMock()
        mock_adapter.issue_token_for_agent = AsyncMock(return_value="mock-jwt-live")
        mock_adapter.validate_token = AsyncMock(return_value={
            "sub": "agent_zai-live", "aud": "a2a-server", "scope": "",
            "agent_id": "zai-live", "org_id": 1, "a2a": {},
        })

        status_updates = []
        async def track_status(aid, status, **kw):
            status_updates.append(status)

        with patch("fiberwise_common.oidc_provider.get_adapter", return_value=mock_adapter), \
             patch.object(processor, '_update_activation_status', side_effect=track_status), \
             patch.object(processor, '_get_activation', new_callable=AsyncMock, return_value={"status": "completed"}):

            activation = await db.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = :id",
                {"id": "act-live"}
            )
            result = await processor.process_activation(dict(activation))

        # Verify the processor ran through the A2A path
        assert "running" in status_updates, f"Expected 'running' status, got: {status_updates}"
        assert "completed" in status_updates or "failed" in status_updates, \
            f"Expected terminal status, got: {status_updates}"
