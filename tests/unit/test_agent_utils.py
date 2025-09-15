"""
Unit tests for fiberwise_common.utils.agent_utils module.

Tests the MetadataMixin class and extract_agent_metadata function.
"""
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from fiberwise_common.utils.agent_utils import (
    MetadataMixin,
    extract_agent_metadata
)


class TestMetadataMixin:
    """Test MetadataMixin class functionality."""

    def test_metadata_mixin_initialization(self):
        """Test that MetadataMixin initializes metadata dictionary."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        assert hasattr(agent, 'metadata')
        assert isinstance(agent.metadata, dict)
        assert len(agent.metadata) == 0

    def test_metadata_mixin_with_existing_metadata(self):
        """Test MetadataMixin with pre-existing metadata."""
        class TestAgent(MetadataMixin):
            def __init__(self):
                self.metadata = {"existing": "value"}
                super().__init__()
        
        agent = TestAgent()
        assert agent.metadata == {"existing": "value"}

    def test_set_agent_metadata(self):
        """Test setting agent metadata."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        agent.set_agent_metadata(name="test_agent", version="1.0.0")
        
        assert agent.metadata["name"] == "test_agent"
        assert agent.metadata["version"] == "1.0.0"

    def test_set_agent_metadata_multiple_calls(self):
        """Test multiple calls to set_agent_metadata."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        agent.set_agent_metadata(name="test_agent")
        agent.set_agent_metadata(version="1.0.0", author="Test Author")
        agent.set_agent_metadata(name="updated_agent")  # Should overwrite
        
        expected = {
            "name": "updated_agent",
            "version": "1.0.0", 
            "author": "Test Author"
        }
        assert agent.metadata == expected

    def test_get_agent_metadata(self):
        """Test getting agent metadata."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        agent.set_agent_metadata(name="test", version="1.0")
        
        metadata_copy = agent.get_agent_metadata()
        assert metadata_copy == {"name": "test", "version": "1.0"}
        
        # Ensure it's a copy, not the original
        metadata_copy["new_key"] = "new_value"
        assert "new_key" not in agent.metadata

    def test_clear_agent_metadata(self):
        """Test clearing agent metadata."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        agent.set_agent_metadata(name="test", version="1.0", tags=["test"])
        
        assert len(agent.metadata) > 0
        agent.clear_agent_metadata()
        assert len(agent.metadata) == 0

    def test_remove_metadata_key_existing(self):
        """Test removing existing metadata key."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        agent.set_agent_metadata(name="test", version="1.0", deprecated=True)
        
        result = agent.remove_metadata_key("deprecated")
        assert result is True
        assert "deprecated" not in agent.metadata
        assert "name" in agent.metadata
        assert "version" in agent.metadata

    def test_remove_metadata_key_nonexistent(self):
        """Test removing nonexistent metadata key."""
        class TestAgent(MetadataMixin):
            pass
        
        agent = TestAgent()
        agent.set_agent_metadata(name="test")
        
        result = agent.remove_metadata_key("nonexistent")
        assert result is False
        assert agent.metadata == {"name": "test"}

    def test_metadata_mixin_with_inheritance(self):
        """Test MetadataMixin with class inheritance."""
        class BaseAgent:
            def __init__(self):
                self.base_attr = "base"
        
        class TestAgent(MetadataMixin, BaseAgent):  # MetadataMixin first for MRO
            def __init__(self):
                super().__init__()
        
        agent = TestAgent()
        assert hasattr(agent, 'base_attr')
        assert hasattr(agent, 'metadata')
        assert agent.base_attr == "base"
        
        agent.set_agent_metadata(type="hybrid")
        assert agent.metadata["type"] == "hybrid"


class TestExtractAgentMetadata:
    """Test extract_agent_metadata function."""

    def test_extract_metadata_function_based_agent(self, temp_dir: Path):
        """Test extracting metadata from function-based agent."""
        agent_code = '''
def run_agent(input_data):
    """This is a test agent that processes input data."""
    return {"status": "success", "data": input_data}
'''
        agent_file = temp_dir / "function_agent.py"
        agent_file.write_text(agent_code)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(agent_file.absolute())
        assert "test agent that processes input data" in metadata["description"]

    def test_extract_metadata_class_based_agent(self, temp_dir: Path):
        """Test extracting metadata from class-based agent."""
        agent_code = '''
class TestAgent:
    """A test agent for unit testing."""
    
    def __init__(self):
        self.name = "test_agent"
    
    def run_agent(self, input_data):
        return {"status": "success"}
'''
        agent_file = temp_dir / "class_agent.py"
        agent_file.write_text(agent_code)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert "TestAgent" in metadata["name"]
        assert str(agent_file.absolute()) in metadata["name"]
        assert "test agent for unit testing" in metadata["description"].lower()

    def test_extract_metadata_class_with_description_attribute(self, temp_dir: Path):
        """Test extracting metadata from class with description attribute."""
        agent_code = '''
class TestAgent:
    def __init__(self):
        self._description = "Agent with internal description"
    
    def run_agent(self, input_data):
        return input_data
'''
        agent_file = temp_dir / "desc_agent.py"
        agent_file.write_text(agent_code)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert metadata["description"] == "Agent with internal description"

    def test_extract_metadata_class_with_execute_method(self, temp_dir: Path):
        """Test extracting metadata from class with execute method."""
        agent_code = '''
class ExecuteAgent:
    """Agent that uses execute method."""
    
    def execute(self, input_data):
        return {"result": input_data}
'''
        agent_file = temp_dir / "execute_agent.py" 
        agent_file.write_text(agent_code)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert "ExecuteAgent" in metadata["name"]

    def test_extract_metadata_no_agent_found(self, temp_dir: Path):
        """Test extracting metadata when no agent is found."""
        regular_code = '''
def regular_function():
    return "not an agent"

class RegularClass:
    def regular_method(self):
        pass
'''
        regular_file = temp_dir / "regular.py"
        regular_file.write_text(regular_code)
        
        metadata = extract_agent_metadata(str(regular_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(regular_file.absolute())
        assert metadata["description"] == ""

    def test_extract_metadata_nonexistent_file(self, temp_dir: Path):
        """Test extracting metadata from nonexistent file."""
        nonexistent = temp_dir / "does_not_exist.py"
        
        metadata = extract_agent_metadata(str(nonexistent))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(nonexistent.absolute())
        assert metadata["description"] == ""

    def test_extract_metadata_invalid_python_file(self, temp_dir: Path):
        """Test extracting metadata from invalid Python file."""
        invalid_code = '''
def invalid_syntax(
    # Missing closing parenthesis and colon
'''
        invalid_file = temp_dir / "invalid.py"
        invalid_file.write_text(invalid_code)
        
        metadata = extract_agent_metadata(str(invalid_file))
        
        # Should fallback gracefully
        assert metadata["type"] == "function"
        assert metadata["name"] == str(invalid_file.absolute())
        assert metadata["description"] == ""

    def test_extract_metadata_class_instantiation_fails(self, temp_dir: Path):
        """Test extracting metadata when class instantiation fails."""
        agent_code = '''
class FailingAgent:
    """Agent that fails to instantiate."""
    
    def __init__(self):
        raise Exception("Cannot instantiate")
    
    def run_agent(self, input_data):
        return input_data
'''
        agent_file = temp_dir / "failing_agent.py"
        agent_file.write_text(agent_code)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert "FailingAgent" in metadata["name"]
        assert "agent that fails to instantiate" in metadata["description"].lower()

    def test_extract_metadata_multiple_classes(self, temp_dir: Path):
        """Test extracting metadata from file with multiple classes."""
        agent_code = '''
class RegularClass:
    def regular_method(self):
        pass

class Agent:  # Should be ignored (reserved name)
    def run_agent(self, input_data):
        return input_data

class TestAgent:
    """The actual agent class."""
    
    def run_agent(self, input_data):
        return input_data

class AnotherAgent:
    """Another agent class."""
    
    def execute(self, input_data):
        return input_data
'''
        agent_file = temp_dir / "multi_agent.py"
        agent_file.write_text(agent_code)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        # Should find the first valid agent (TestAgent)
        assert metadata["type"] == "class"
        assert "TestAgent" in metadata["name"]
        assert "actual agent class" in metadata["description"].lower()

    @pytest.mark.parametrize("agent_type,expected_type", [
        ("function", "function"),
        ("class_with_run_agent", "class"),
        ("class_with_execute", "class"),
        ("no_agent", "function")
    ])
    def test_extract_metadata_parametrized(self, temp_dir: Path, agent_type: str, expected_type: str):
        """Test various agent types."""
        code_templates = {
            "function": 'def run_agent(input_data): return input_data',
            "class_with_run_agent": '''
class TestAgent:
    def run_agent(self, input_data): return input_data
''',
            "class_with_execute": '''
class TestAgent:
    def execute(self, input_data): return input_data
''',
            "no_agent": 'def regular_function(): pass'
        }
        
        agent_file = temp_dir / f"{agent_type}_agent.py"
        agent_file.write_text(code_templates[agent_type])
        
        metadata = extract_agent_metadata(str(agent_file))
        assert metadata["type"] == expected_type


class TestIntegration:
    """Integration tests combining MetadataMixin and extract_agent_metadata."""

    def test_agent_with_metadata_mixin_introspection(self, temp_dir: Path):
        """Test agent using MetadataMixin with metadata extraction."""
        # This simulates how an agent might use both features
        agent_code = '''
from fiberwise_common.utils.agent_utils import MetadataMixin

class SmartAgent(MetadataMixin):
    """A smart agent that manages its own metadata."""
    
    def __init__(self):
        super().__init__()
        self.set_agent_metadata(
            name="smart_agent",
            version="1.0.0",
            capabilities=["data_processing", "analysis"]
        )
    
    def run_agent(self, input_data):
        metadata = self.get_agent_metadata()
        return {
            "status": "success",
            "agent_info": metadata,
            "result": input_data
        }
'''
        agent_file = temp_dir / "smart_agent.py"
        agent_file.write_text(agent_code)
        
        # Test metadata extraction (file introspection)
        file_metadata = extract_agent_metadata(str(agent_file))
        assert file_metadata["type"] == "class"
        assert "SmartAgent" in file_metadata["name"]
        
        # Note: We can't actually import and test the runtime metadata
        # without proper module loading, but this tests the file analysis
        assert "smart agent that manages its own metadata" in file_metadata["description"].lower()

    def test_metadata_workflow_example(self, temp_dir: Path):
        """Example workflow combining file analysis and runtime metadata."""
        # Create multiple agent files
        agents = {
            "processor": '''
def run_agent(input_data):
    """Processes incoming data."""
    return {"processed": input_data}
''',
            "analyzer": '''
class AnalyzerAgent:
    """Analyzes processed data."""
    
    def run_agent(self, data):
        return {"analysis": "complete"}
''',
            "reporter": '''
class ReporterAgent:
    """Generates reports from analysis."""
    
    def execute(self, analysis):
        return {"report": "generated"}
'''
        }
        
        agent_metadata = {}
        for name, code in agents.items():
            agent_file = temp_dir / f"{name}_agent.py"
            agent_file.write_text(code)
            
            metadata = extract_agent_metadata(str(agent_file))
            agent_metadata[name] = {
                "file_path": str(agent_file),
                "type": metadata["type"],
                "description": metadata["description"]
            }
        
        # Verify we found all agents with correct types
        assert agent_metadata["processor"]["type"] == "function"
        assert agent_metadata["analyzer"]["type"] == "class"
        assert agent_metadata["reporter"]["type"] == "class"
        
        # Verify descriptions were extracted
        assert "processes" in agent_metadata["processor"]["description"].lower()
        assert "analyzes" in agent_metadata["analyzer"]["description"].lower()
        assert "generates" in agent_metadata["reporter"]["description"].lower()