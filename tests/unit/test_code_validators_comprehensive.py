"""
Comprehensive unit tests for fiberwise_common.utils.code_validators module.

This module tests the code validation functions including input validation
and code snippet validation.
"""

import pytest
from typing import Any, List

from fiberwise_common.utils.code_validators import validate_input, validate_code_snippet


class TestValidateInput:
    """Test suite for validate_input function."""
    
    def test_validate_none_input(self):
        """Test that None input is invalid."""
        assert validate_input(None) is False
        
    def test_validate_empty_string(self):
        """Test that empty string is invalid."""
        assert validate_input("") is False
        
    def test_validate_non_empty_string(self):
        """Test that non-empty string is valid."""
        assert validate_input("hello") is True
        assert validate_input("a") is True
        assert validate_input("  ") is True  # Whitespace counts as non-empty
        
    def test_validate_empty_list(self):
        """Test that empty list is invalid."""
        assert validate_input([]) is False
        
    def test_validate_non_empty_list(self):
        """Test that non-empty list is valid."""
        assert validate_input([1, 2, 3]) is True
        assert validate_input([""]) is True  # List with empty string is non-empty list
        assert validate_input([None]) is True
        
    def test_validate_empty_dict(self):
        """Test that empty dictionary is invalid."""
        assert validate_input({}) is False
        
    def test_validate_non_empty_dict(self):
        """Test that non-empty dictionary is valid."""
        assert validate_input({"key": "value"}) is True
        assert validate_input({"": ""}) is True  # Dict with empty strings is non-empty dict
        assert validate_input({None: None}) is True
        
    def test_validate_numeric_values(self):
        """Test validation of numeric values."""
        assert validate_input(0) is True
        assert validate_input(42) is True
        assert validate_input(-1) is True
        assert validate_input(3.14) is True
        assert validate_input(0.0) is True
        
    def test_validate_boolean_values(self):
        """Test validation of boolean values."""
        assert validate_input(True) is True
        assert validate_input(False) is True
        
    def test_validate_complex_objects(self):
        """Test validation of complex objects."""
        # Tuples
        assert validate_input(()) is False  # Empty tuple
        assert validate_input((1, 2)) is True  # Non-empty tuple
        
        # Sets
        assert validate_input(set()) is False  # Empty set
        assert validate_input({1, 2, 3}) is True  # Non-empty set
        
        # Custom objects
        class CustomObject:
            pass
        
        assert validate_input(CustomObject()) is True
        
    def test_validate_nested_structures(self):
        """Test validation of nested data structures."""
        # List of lists
        assert validate_input([[]]) is True  # Non-empty list containing empty list
        assert validate_input([[], []]) is True
        
        # Dict with empty values
        assert validate_input({"key": []}) is True  # Non-empty dict with empty list value
        assert validate_input({"key": {}}) is True  # Non-empty dict with empty dict value
        
        # Complex nested structure
        complex_data = {
            "users": [
                {"name": "Alice", "scores": [85, 92]},
                {"name": "Bob", "scores": []}
            ],
            "metadata": {}
        }
        assert validate_input(complex_data) is True


@pytest.mark.parametrize("test_input,expected", [
    # String tests
    ("hello", True),
    ("", False),
    ("  ", True),
    
    # List tests
    ([1, 2, 3], True),
    ([], False),
    ([None], True),
    
    # Dict tests
    ({"a": 1}, True),
    ({}, False),
    
    # Other types
    (42, True),
    (0, True),
    (None, False),
    (True, True),
    (False, True),
])
class TestValidateInputParametrized:
    """Parametrized tests for validate_input function."""
    
    def test_validate_input_parametrized(self, test_input: Any, expected: bool):
        """Test validate_input with various inputs."""
        assert validate_input(test_input) == expected


class TestValidateCodeSnippet:
    """Test suite for validate_code_snippet function."""
    
    def test_validate_empty_code(self):
        """Test validation of empty code."""
        code, warnings = validate_code_snippet("", "python")
        
        assert code == ""
        assert len(warnings) == 1
        assert "Code is empty" in warnings
        
    def test_validate_whitespace_only_code(self):
        """Test validation of whitespace-only code."""
        code, warnings = validate_code_snippet("   \n\t  ", "python")
        
        assert code == "   \n\t  "
        assert len(warnings) == 1
        assert "Code is empty" in warnings
        
    def test_validate_valid_code(self):
        """Test validation of valid code."""
        valid_code = "def hello():\n    return 'world'"
        code, warnings = validate_code_snippet(valid_code, "python")
        
        assert code == valid_code
        assert len(warnings) == 0
        
    def test_validate_simple_code(self):
        """Test validation of simple code."""
        simple_code = "print('hello')"
        code, warnings = validate_code_snippet(simple_code, "python")
        
        assert code == simple_code
        assert len(warnings) == 0
        
    def test_validate_code_with_comments(self):
        """Test validation of code with comments."""
        commented_code = "# This is a comment\nprint('hello')  # Another comment"
        code, warnings = validate_code_snippet(commented_code, "python")
        
        assert code == commented_code
        assert len(warnings) == 0
        
    def test_validate_multiline_code(self):
        """Test validation of multiline code."""
        multiline_code = """
def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    return a + b

result = calculate_sum(5, 3)
print(f"Result: {result}")
"""
        code, warnings = validate_code_snippet(multiline_code, "python")
        
        assert code == multiline_code
        assert len(warnings) == 0
        
    def test_validate_code_with_special_characters(self):
        """Test validation of code with special characters."""
        special_code = "text = 'Hello, World! @#$%^&*()'"
        code, warnings = validate_code_snippet(special_code, "python")
        
        assert code == special_code
        assert len(warnings) == 0
        
    def test_language_parameter_unused(self):
        """Test that language parameter is currently unused."""
        test_code = "console.log('hello');"
        
        # Should work the same regardless of language
        python_result = validate_code_snippet(test_code, "python")
        javascript_result = validate_code_snippet(test_code, "javascript")
        unknown_result = validate_code_snippet(test_code, "unknown_language")
        
        assert python_result == javascript_result == unknown_result
        assert all(len(result[1]) == 0 for result in [python_result, javascript_result, unknown_result])
        
    def test_validate_code_return_types(self):
        """Test that function returns correct types."""
        code, warnings = validate_code_snippet("print('test')", "python")
        
        assert isinstance(code, str)
        assert isinstance(warnings, list)
        assert all(isinstance(warning, str) for warning in warnings)
        
    def test_validate_code_preserves_original(self):
        """Test that original code is preserved unchanged."""
        original_code = "  def test():\n      return True  \n"
        code, warnings = validate_code_snippet(original_code, "python")
        
        assert code == original_code  # Should be exactly the same
        assert code is not original_code  # But not the same object
        
    def test_validate_code_edge_cases(self):
        """Test edge cases for code validation."""
        # Single character
        code, warnings = validate_code_snippet("x", "python")
        assert code == "x"
        assert len(warnings) == 0
        
        # Only newlines
        code, warnings = validate_code_snippet("\n\n", "python")
        assert code == "\n\n"
        assert len(warnings) == 1
        
        # Mixed whitespace
        code, warnings = validate_code_snippet(" \t\n ", "python")
        assert code == " \t\n "
        assert len(warnings) == 1


@pytest.mark.parametrize("code_input,expected_warnings_count", [
    ("print('hello')", 0),
    ("", 1),
    ("   ", 1),
    ("\n\t", 1),
    ("x", 0),
    ("def func(): pass", 0),
    ("# just a comment", 0),
    (" # comment with spaces ", 0),
])
class TestValidateCodeSnippetParametrized:
    """Parametrized tests for validate_code_snippet function."""
    
    def test_validate_code_snippet_parametrized(self, code_input: str, expected_warnings_count: int):
        """Test validate_code_snippet with various code inputs."""
        code, warnings = validate_code_snippet(code_input, "python")
        
        assert code == code_input
        assert len(warnings) == expected_warnings_count
        
        if expected_warnings_count > 0:
            assert any("empty" in warning.lower() for warning in warnings)


class TestValidateCodeSnippetLanguages:
    """Test validate_code_snippet with different languages."""
    
    @pytest.mark.parametrize("language", [
        "python",
        "javascript", 
        "java",
        "cpp",
        "c",
        "rust",
        "go",
        "typescript",
        "unknown_language",
        "",
        None
    ])
    def test_different_languages_same_behavior(self, language):
        """Test that different languages produce the same validation behavior."""
        test_code = "function test() { return true; }"
        
        # Handle None case
        if language is None:
            code, warnings = validate_code_snippet(test_code, language)
        else:
            code, warnings = validate_code_snippet(test_code, language)
        
        assert code == test_code
        assert len(warnings) == 0


class TestValidationIntegration:
    """Integration tests combining both validation functions."""
    
    def test_validate_input_then_code(self):
        """Test workflow of validating input then code."""
        # Valid input
        code_input = "def hello(): return 'world'"
        
        # First validate as input
        input_valid = validate_input(code_input)
        assert input_valid is True
        
        # Then validate as code
        code, warnings = validate_code_snippet(code_input, "python")
        assert code == code_input
        assert len(warnings) == 0
        
    def test_validate_empty_input_and_code(self):
        """Test validation of empty input through both functions."""
        empty_input = ""
        
        # Validate as input
        input_valid = validate_input(empty_input)
        assert input_valid is False
        
        # Validate as code
        code, warnings = validate_code_snippet(empty_input, "python")
        assert code == empty_input
        assert len(warnings) == 1
        assert "Code is empty" in warnings[0]
        
    def test_validate_none_input_vs_code(self):
        """Test behavior difference between validating None as input vs code."""
        # None as input
        input_valid = validate_input(None)
        assert input_valid is False
        
        # None as code (should be converted to string)
        with pytest.raises(AttributeError):
            # This should fail because None.strip() will raise AttributeError
            validate_code_snippet(None, "python")


class TestValidationErrorHandling:
    """Test error handling in validation functions."""
    
    def test_validate_input_with_non_standard_types(self):
        """Test validate_input with non-standard types."""
        # Generator
        def gen():
            yield 1
        
        generator = gen()
        assert validate_input(generator) is True
        
        # Lambda function
        func = lambda x: x
        assert validate_input(func) is True
        
        # Module
        import os
        assert validate_input(os) is True
        
    def test_validate_code_snippet_type_safety(self):
        """Test that validate_code_snippet handles string input safely."""
        # Should work with any string
        test_cases = [
            "normal code",
            "code with unicode: 你好",
            "code\nwith\nnewlines",
            "code\twith\ttabs",
            r"code with raw strings \n \t",
            "code with quotes: 'single' and \"double\"",
        ]
        
        for test_code in test_cases:
            code, warnings = validate_code_snippet(test_code, "python")
            assert isinstance(code, str)
            assert isinstance(warnings, list)
            assert code == test_code


if __name__ == "__main__":
    pytest.main([__file__])