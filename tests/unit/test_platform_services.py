"""
Unit tests for platform services (scheduler, telemetry, memory, checkpoints,
decision trail, reliability, agent orchestration, tool registry).
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fiberwise_common.database.provider import NexusQLProvider


# ---------------------------------------------------------------------------
# Shared mock DB fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_db():
    mock = Mock(spec=NexusQLProvider)
    mock.fetch_all = AsyncMock(return_value=[])
    mock.fetch_one = AsyncMock(return_value=None)
    mock.fetch_val = AsyncMock(return_value=None)
    mock.execute = AsyncMock()
    mock.execute_many = AsyncMock()
    return mock


# ===========================================================================
# SchedulerService
# ===========================================================================
class TestSchedulerService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.scheduler_service import SchedulerService
        return SchedulerService(mock_db)

    @pytest.mark.asyncio
    async def test_create_job(self, service, mock_db):
        mock_db.execute.return_value = None
        job_id = await service.create_job(
            job_name="nightly-sync",
            pipeline_id="pipe-1",
            created_by="user-1",
            cron_expression="0 0 * * *",
        )
        assert job_id is not None
        mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        jobs = await service.list_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_list_jobs_with_pipeline_filter(self, service, mock_db):
        mock_db.fetch_all.return_value = [
            {"job_id": "j1", "job_name": "job1", "pipeline_id": "pipe-1"}
        ]
        jobs = await service.list_jobs(pipeline_id="pipe-1")
        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.get_job("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_job_found(self, service, mock_db):
        mock_db.fetch_one.return_value = {"job_id": "j1", "job_name": "test"}
        result = await service.get_job("j1")
        assert result["job_id"] == "j1"

    @pytest.mark.asyncio
    async def test_update_job(self, service, mock_db):
        mock_db.fetch_one.return_value = {"job_id": "j1"}
        result = await service.update_job("j1", job_name="renamed")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_job(self, service, mock_db):
        mock_db.fetch_one.return_value = {"job_id": "j1"}
        result = await service.delete_job("j1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_job_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.delete_job("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_job_history(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        history = await service.get_job_history("j1")
        assert history == []

    @pytest.mark.asyncio
    async def test_run_job_now_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        with pytest.raises(ValueError):
            await service.run_job_now("nonexistent")


# ===========================================================================
# TelemetryService
# ===========================================================================
class TestTelemetryService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.telemetry_service import TelemetryService
        return TelemetryService(mock_db)

    @pytest.mark.asyncio
    async def test_get_execution_metrics(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        result = await service.get_execution_metrics("exec-1")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_execution_spans_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        spans = await service.get_execution_spans("exec-1")
        assert spans == []

    @pytest.mark.asyncio
    async def test_get_span_timeline(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        timeline = await service.get_span_timeline("exec-1")
        assert isinstance(timeline, list)

    @pytest.mark.asyncio
    async def test_record_step_metrics(self, service, mock_db):
        await service.record_step_metrics(
            execution_id="exec-1",
            pipeline_id="pipe-1",
            step_id="step-1",
            metrics={"latency_ms": 100, "tokens_used": 50},
        )
        mock_db.execute.assert_called()


# ===========================================================================
# MemoryService
# ===========================================================================
class TestMemoryService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.memory_service import MemoryService
        return MemoryService(mock_db)

    @pytest.mark.asyncio
    async def test_save_conversation_turn(self, service, mock_db):
        msg_id = await service.save_conversation_turn(
            session_id="sess-1",
            role="user",
            content="Hello",
            created_by="user-1",
        )
        assert msg_id is not None
        mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        history = await service.get_conversation_history("sess-1")
        assert history == []

    @pytest.mark.asyncio
    async def test_search_memory(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        results = await service.search_memory("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        mock_db.fetch_val.return_value = 0
        stats = await service.get_memory_stats("sess-1")
        assert isinstance(stats, dict)


# ===========================================================================
# CheckpointService
# ===========================================================================
class TestCheckpointService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.checkpoint_service import CheckpointService
        return CheckpointService(mock_db)

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        cps = await service.list_checkpoints("exec-1")
        assert cps == []

    @pytest.mark.asyncio
    async def test_get_checkpoint_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.get_checkpoint("cp-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_checkpoint_state_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.get_checkpoint_state("cp-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, service, mock_db):
        mock_db.fetch_one.return_value = {
            "checkpoint_id": "cp-1",
            "pipeline_id": "pipe-1",
            "step_index": 2,
            "state_data": "{}",
            "execution_id": "exec-1",
        }
        result = await service.resume_from_checkpoint("cp-1")
        assert isinstance(result, dict)


# ===========================================================================
# DecisionTrailService
# ===========================================================================
class TestDecisionTrailService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.decision_trail_service import DecisionTrailService
        return DecisionTrailService(mock_db)

    @pytest.mark.asyncio
    async def test_get_decision_trail_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        trail = await service.get_decision_trail("exec-1")
        assert isinstance(trail, dict)

    @pytest.mark.asyncio
    async def test_get_decision_node_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.get_decision_node("exec-1", "node-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_execution_path(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        path = await service.get_execution_path("exec-1")
        assert isinstance(path, list)

    @pytest.mark.asyncio
    async def test_export_trail_json(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        result = await service.export_trail("exec-1", fmt="json")
        assert result is not None


# ===========================================================================
# ReliabilityService
# ===========================================================================
class TestReliabilityService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.reliability_service import ReliabilityService
        return ReliabilityService(mock_db)

    @pytest.mark.asyncio
    async def test_get_metrics(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        mock_db.fetch_one.return_value = None
        metrics = await service.get_metrics()
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_get_alerts_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        alerts = await service.get_alerts()
        assert alerts == []

    @pytest.mark.asyncio
    async def test_create_alert(self, service, mock_db):
        await service.create_alert(
            pipeline_id="pipe-1",
            alert_type="anomaly",
            severity="warning",
            message="High latency detected",
        )
        mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_get_cost_report(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        mock_db.fetch_one.return_value = None
        report = await service.get_cost_report()
        assert isinstance(report, dict)


# ===========================================================================
# AgentOrchestrationService
# ===========================================================================
class TestAgentOrchestrationService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.agent_orchestration_service import AgentOrchestrationService
        return AgentOrchestrationService(mock_db)

    @pytest.mark.asyncio
    async def test_create_workflow(self, service, mock_db):
        wf_id = await service.create_workflow(
            name="multi-agent-debate",
            description="Debate workflow",
            workflow_config={"pattern": "debate", "agents": []},
            created_by="user-1",
        )
        assert wf_id is not None
        mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_list_workflows_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        workflows = await service.list_workflows()
        assert workflows == []

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.get_workflow("wf-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_role(self, service, mock_db):
        role_id = await service.create_role(
            name="researcher",
            description="Research role",
            system_prompt="You are a researcher.",
        )
        assert role_id is not None

    @pytest.mark.asyncio
    async def test_list_roles_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        roles = await service.list_roles()
        assert roles == []

    @pytest.mark.asyncio
    async def test_list_patterns(self, service, mock_db):
        patterns = await service.list_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0  # Should return hardcoded patterns

    @pytest.mark.asyncio
    async def test_get_workflow_status_not_found(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service.get_workflow_status("exec-1")
        assert result is None


# ===========================================================================
# ToolRegistryService
# ===========================================================================
class TestToolRegistryService:
    @pytest.fixture
    def service(self, mock_db):
        from fiberwise_common.services.tool_registry_service import ToolRegistryService
        return ToolRegistryService(mock_db)

    @pytest.mark.asyncio
    async def test_list_tools_fallback(self, service):
        """When ia_modules is not available, returns built-in defaults."""
        tools = await service.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert tools[0]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_builtin_tools_structure(self, service):
        tools = service._builtin_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "category" in tool
            assert "parameters" in tool
