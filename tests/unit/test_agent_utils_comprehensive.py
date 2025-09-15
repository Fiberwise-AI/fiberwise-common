"""
Comprehensive unit tests for fiberwise_common.utils.agent_utils module.

This module tests the agent utility functions including metadata management
and agent introspection capabilities.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fiberwise_common.utils.agent_utils import MetadataMixin, extract_agent_metadata


class TestAgent(MetadataMixin):
    """Test agent class for testing MetadataMixin."""
    
    def __init__(self):
        super().__init__()


class TestMetadataMixin:
    """Test suite for MetadataMixin class."""
    
    def test_mixin_initialization(self):
        """Test that MetadataMixin initializes metadata properly."""
        agent = TestAgent()
        assert hasattr(agent, 'metadata')
        assert isinstance(agent.metadata, dict)
        assert len(agent.metadata) == 0
        
    def test_mixin_with_existing_metadata(self):
        """Test mixin behavior when metadata already exists."""
        class AgentWithMetadata(MetadataMixin):
            def __init__(self):
                self.metadata = {"existing": "value"}
                super().__init__()
        
        agent = AgentWithMetadata()
        assert agent.metadata == {"existing": "value"}
        
    def test_set_agent_metadata_basic(self):
        """Test setting basic metadata."""
        agent = TestAgent()
        
        agent.set_agent_metadata(name="test_agent", version="1.0.0")
        
        assert agent.metadata["name"] == "test_agent"
        assert agent.metadata["version"] == "1.0.0"
        
    def test_set_agent_metadata_overwrite(self):
        """Test that setting metadata overwrites existing keys."""
        agent = TestAgent()
        
        agent.set_agent_metadata(name="old_name")
        assert agent.metadata["name"] == "old_name"
        
        agent.set_agent_metadata(name="new_name", description="test")
        assert agent.metadata["name"] == "new_name"
        assert agent.metadata["description"] == "test"
        
    def test_set_agent_metadata_complex_values(self):
        """Test setting metadata with complex values."""
        agent = TestAgent()
        
        complex_data = {
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "data"},
            "tuple_value": (1, 2),
            "none_value": None,
            "bool_value": True
        }
        
        agent.set_agent_metadata(**complex_data)
        
        for key, value in complex_data.items():
            assert agent.metadata[key] == value
            
    def test_get_agent_metadata_copy(self):
        """Test that get_agent_metadata returns a copy."""
        agent = TestAgent()
        agent.set_agent_metadata(name="test")
        
        metadata1 = agent.get_agent_metadata()
        metadata2 = agent.get_agent_metadata()
        
        # Should be equal but not the same object
        assert metadata1 == metadata2
        assert metadata1 is not metadata2
        assert metadata1 is not agent.metadata
        
    def test_get_agent_metadata_modification_safety(self):
        """Test that modifying returned metadata doesn't affect original."""
        agent = TestAgent()
        agent.set_agent_metadata(name="test")
        
        metadata = agent.get_agent_metadata()
        metadata["modified"] = "value"
        
        # Original metadata should be unchanged
        assert "modified" not in agent.metadata
        assert agent.metadata == {"name": "test"}
        
    def test_clear_agent_metadata(self):
        """Test clearing all metadata."""
        agent = TestAgent()
        agent.set_agent_metadata(name="test", version="1.0.0", description="test")
        
        assert len(agent.metadata) == 3
        
        agent.clear_agent_metadata()
        
        assert len(agent.metadata) == 0
        assert agent.metadata == {}
        
    def test_remove_metadata_key_existing(self):
        """Test removing existing metadata key."""
        agent = TestAgent()
        agent.set_agent_metadata(name="test", version="1.0.0", description="test")
        
        result = agent.remove_metadata_key("version")
        
        assert result is True
        assert "version" not in agent.metadata
        assert agent.metadata == {"name": "test", "description": "test"}
        
    def test_remove_metadata_key_non_existing(self):
        """Test removing non-existing metadata key."""
        agent = TestAgent()
        agent.set_agent_metadata(name="test")
        
        result = agent.remove_metadata_key("non_existing")
        
        assert result is False
        assert agent.metadata == {"name": "test"}
        
    def test_remove_metadata_key_empty_metadata(self):
        """Test removing key from empty metadata."""
        agent = TestAgent()
        
        result = agent.remove_metadata_key("any_key")
        
        assert result is False
        assert agent.metadata == {}


class TestExtractAgentMetadata:
    """Test suite for extract_agent_metadata function."""
    
    def test_extract_function_agent_with_docstring(self, temp_dir):
        """Test extracting metadata from function-based agent with docstring."""
        agent_content = '''
def run_agent(input_data):
    """This is a test agent that processes input data."""
    return {"result": "processed"}
'''
        agent_file = temp_dir / "function_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(agent_file.resolve())
        assert "test agent that processes input data" in metadata["description"]
        
    def test_extract_function_agent_no_docstring(self, temp_dir):
        """Test extracting metadata from function-based agent without docstring."""
        agent_content = '''
def run_agent(input_data):
    return {"result": "processed"}
'''
        agent_file = temp_dir / "function_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(agent_file.resolve())
        assert metadata["description"] == ""
        
    def test_extract_class_agent_with_run_agent(self, temp_dir):
        """Test extracting metadata from class-based agent with run_agent method."""
        agent_content = '''
class TestAgent:
    """A test agent class."""
    
    def __init__(self):
        self._description = "Agent with description attribute"
        
    def run_agent(self, input_data):
        return {"result": "processed"}
'''
        agent_file = temp_dir / "class_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert metadata["name"] == f"{agent_file.resolve()}::TestAgent"
        assert metadata["description"] == "Agent with description attribute"
        
    def test_extract_class_agent_with_execute(self, temp_dir):
        """Test extracting metadata from class-based agent with execute method."""
        agent_content = '''
class ProcessorAgent:
    """An agent that processes data."""
    
    def __init__(self):
        self.description = "Processor description"
        
    def execute(self, data):
        return data
'''
        agent_file = temp_dir / "processor_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert metadata["name"] == f"{agent_file.resolve()}::ProcessorAgent"
        assert metadata["description"] == "Processor description"
        
    def test_extract_class_agent_fallback_docstring(self, temp_dir):
        """Test extracting metadata falls back to class docstring."""
        agent_content = '''
class DocumentedAgent:
    """This agent is well documented.
    
    It performs various operations on data.
    """
    
    def run_agent(self, data):
        return data
'''
        agent_file = temp_dir / "documented_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert metadata["description"] == "This agent is well documented."
        
    def test_extract_class_agent_instantiation_error(self, temp_dir):
        """Test handling class that can't be instantiated."""
        agent_content = '''
class ProblematicAgent:
    """An agent that has instantiation issues."""
    
    def __init__(self):
        raise ValueError("Cannot instantiate")
        
    def run_agent(self, data):
        return data
'''
        agent_file = temp_dir / "problematic_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert metadata["name"] == f"{agent_file.resolve()}::ProblematicAgent"
        assert metadata["description"] == "An agent that has instantiation issues."
        
    def test_extract_no_suitable_agent(self, temp_dir):
        """Test file with no suitable agent (fallback to function type)."""
        content = '''
def some_function():
    return "not an agent"
    
class NotAnAgent:
    def regular_method(self):
        return "nothing"
'''
        agent_file = temp_dir / "not_agent.py"
        agent_file.write_text(content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(agent_file.resolve())
        assert metadata["description"] == ""
        
    def test_extract_invalid_file(self, temp_dir):
        """Test handling invalid Python file."""
        invalid_file = temp_dir / "invalid.py"
        invalid_file.write_text("invalid python syntax !")
        
        metadata = extract_agent_metadata(str(invalid_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(invalid_file.resolve())
        assert metadata["description"] == ""
        
    def test_extract_nonexistent_file(self, temp_dir):
        """Test handling non-existent file."""
        nonexistent = temp_dir / "does_not_exist.py"
        
        metadata = extract_agent_metadata(str(nonexistent))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(nonexistent.resolve())
        assert metadata["description"] == ""
        
    def test_extract_multiple_classes(self, temp_dir):
        """Test file with multiple agent classes (should find the first suitable one)."""
        agent_content = '''
class FirstAgent:
    """First agent class."""
    
    def run_agent(self, data):
        return data

class SecondAgent:
    """Second agent class."""
    
    def execute(self, data):
        return data
'''
        agent_file = temp_dir / "multi_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        # Should get the first suitable class found
        assert "Agent" in metadata["name"]
        
    def test_extract_excluded_base_classes(self, temp_dir):
        """Test that base classes are excluded from detection."""
        agent_content = '''
class Agent:
    """Base agent class - should be excluded."""
    
    def run_agent(self, data):
        return data
        
class FiberAgent:
    """FiberAgent base class - should be excluded."""
    
    def execute(self, data):
        return data
        
class MyCustomAgent:
    """Custom agent - should be detected."""
    
    def run_agent(self, data):
        return data
'''
        agent_file = temp_dir / "base_classes.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert "MyCustomAgent" in metadata["name"]
        assert metadata["description"] == "Custom agent - should be detected."
        
    @patch('importlib.util.spec_from_file_location', return_value=None)
    def test_extract_import_spec_failure(self, mock_spec, temp_dir):
        """Test handling import spec creation failure."""
        agent_file = temp_dir / "agent.py"
        agent_file.write_text("def run_agent(): pass")
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "function"
        assert metadata["name"] == str(agent_file.resolve())
        assert metadata["description"] == ""
        
    def test_extract_complex_class_hierarchy(self, temp_dir):
        """Test extracting from complex class hierarchy."""
        agent_content = '''
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """Abstract base processor."""
    
    @abstractmethod
    def process(self, data):
        pass

class ConcreteAgent(BaseProcessor):
    """A concrete implementation of the processor."""
    
    def __init__(self):
        self._description = "Processes data concretely"
    
    def process(self, data):
        return data
        
    def run_agent(self, data):
        return self.process(data)
'''
        agent_file = temp_dir / "hierarchy_agent.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["type"] == "class"
        assert "ConcreteAgent" in metadata["name"]
        assert metadata["description"] == "Processes data concretely"


@pytest.mark.parametrize("description_attr,expected", [
    ("_description", "private description"),
    ("description", "public description"),
    (None, "Fallback to docstring"),
])
class TestMetadataExtractionPriority:
    """Test the priority order for description extraction."""
    
    def test_description_priority(self, temp_dir, description_attr, expected):
        """Test description extraction priority."""
        if description_attr:
            agent_content = f'''
class TestAgent:
    """Fallback to docstring"""
    
    def __init__(self):
        self.{description_attr} = "{expected}"
        
    def run_agent(self, data):
        return data
'''
        else:
            agent_content = '''
class TestAgent:
    """Fallback to docstring"""
    
    def run_agent(self, data):
        return data
'''
        
        agent_file = temp_dir / "priority_test.py"
        agent_file.write_text(agent_content)
        
        metadata = extract_agent_metadata(str(agent_file))
        
        assert metadata["description"] == expected


if __name__ == "__main__":
    pytest.main([__file__])