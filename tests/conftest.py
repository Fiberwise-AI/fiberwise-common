"""
Fiberwise-common specific test fixtures and configuration.
"""
import os
import tempfile
import pytest
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, AsyncMock

from fiberwise_common.database.provider import NexusQLProvider, DatabaseProvider


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Sample JSON data for testing."""
    return {
        "name": "test_agent",
        "version": "1.0.0",
        "description": "Test agent for unit tests",
        "type": "function",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 100
        },
        "tags": ["test", "sample"]
    }


@pytest.fixture
def sample_manifest_data() -> Dict[str, Any]:
    """Sample manifest data for testing."""
    return {
        "name": "test-app",
        "version": "1.0.0",
        "description": "Test application manifest",
        "agents": [
            {
                "name": "test_agent",
                "file": "agents/test_agent.py",
                "description": "A test agent"
            }
        ],
        "functions": [
            {
                "name": "test_function",
                "file": "functions/test_function.py",
                "description": "A test function"
            }
        ]
    }


@pytest.fixture
def mock_database_provider() -> Mock:
    """Mock database provider for testing."""
    mock_db = Mock(spec=NexusQLProvider)
    mock_db.connect = AsyncMock(return_value=True)
    mock_db.disconnect = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch_all = AsyncMock(return_value=[])
    mock_db.fetch_one = AsyncMock(return_value=None)
    mock_db.fetch_val = AsyncMock(return_value=None)
    mock_db.execute_many = AsyncMock()
    mock_db.execute_script = AsyncMock()
    mock_db.is_healthy = AsyncMock(return_value=True)
    mock_db.table_exists = Mock(return_value=False)
    mock_db.database_url = "sqlite:///test.db"
    return mock_db


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    """SQLite database URL for testing."""
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def sqlite_provider(sqlite_url: str) -> NexusQLProvider:
    """NexusQL provider backed by SQLite for testing."""
    return NexusQLProvider(sqlite_url)


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
def test_function():
    """A simple test function."""
    return "Hello, World!"

class TestAgent:
    """A simple test agent."""
    
    def __init__(self):
        self.name = "test_agent"
    
    def run_agent(self, input_data):
        """Run the agent with input data."""
        return {"status": "success", "data": input_data}
'''


@pytest.fixture
def sample_agent_file(temp_dir: Path, sample_python_code: str) -> Path:
    """Create a sample agent file for testing."""
    agent_file = temp_dir / "test_agent.py"
    agent_file.write_text(sample_python_code)
    return agent_file


@pytest.fixture
def sample_json_file(temp_dir: Path, sample_json_data: Dict[str, Any]) -> Path:
    """Create a sample JSON file for testing."""
    import json
    json_file = temp_dir / "test.json"
    json_file.write_text(json.dumps(sample_json_data, indent=2))
    return json_file


@pytest.fixture  
def sample_yaml_file(temp_dir: Path, sample_manifest_data: Dict[str, Any]) -> Path:
    """Create a sample YAML file for testing."""
    import yaml
    yaml_file = temp_dir / "test.yaml"
    yaml_file.write_text(yaml.dump(sample_manifest_data, default_flow_style=False))
    return yaml_file


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Mock LLM response data for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the LLM."
                },
                "finish_reason": "stop"
            }
        ],
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup logic can be added here if needed


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment configuration."""
    # Set test environment variables
    os.environ["FIBERWISE_TEST_MODE"] = "true"
    os.environ["FIBERWISE_LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup environment variables
    os.environ.pop("FIBERWISE_TEST_MODE", None)
    os.environ.pop("FIBERWISE_LOG_LEVEL", None)


# Custom markers for parametrized tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration directory  
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)