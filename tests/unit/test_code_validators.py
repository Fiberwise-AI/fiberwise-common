"""
Unit tests for fiberwise_common.utils.code_validators module.

Tests input validation and code snippet validation functions.
"""
import pytest
from typing import List, Tuple, Any

from fiberwise_common.utils.code_validators import (
    validate_input,
    validate_code_snippet
)


class TestValidateInput:
    """Test validate_input function."""

    def test_validate_input_valid_string(self):
        """Test validation of valid non-empty string."""
        assert validate_input("hello world") is True
        assert validate_input("a") is True
        assert validate_input("123") is True

    def test_validate_input_empty_string(self):
        """Test validation of empty string."""
        assert validate_input("") is False

    def test_validate_input_whitespace_string(self):
        """Test validation of whitespace-only string."""
        assert validate_input("   ") is True  # Non-empty string
        assert validate_input("\n\t") is True  # Non-empty string

    def test_validate_input_none(self):
        """Test validation of None value."""
        assert validate_input(None) is False

    def test_validate_input_valid_list(self):
        """Test validation of valid non-empty list."""
        assert validate_input([1, 2, 3]) is True
        assert validate_input(["a"]) is True
        assert validate_input([None]) is True  # Non-empty list

    def test_validate_input_empty_list(self):
        """Test validation of empty list."""
        assert validate_input([]) is False

    def test_validate_input_valid_dict(self):
        """Test validation of valid non-empty dictionary."""
        assert validate_input({"key": "value"}) is True
        assert validate_input({"a": 1, "b": 2}) is True
        assert validate_input({1: "test"}) is True

    def test_validate_input_empty_dict(self):
        """Test validation of empty dictionary."""
        assert validate_input({}) is False

    def test_validate_input_numeric_types(self):
        """Test validation of numeric types."""
        assert validate_input(42) is True
        assert validate_input(0) is True
        assert validate_input(-1) is True
        assert validate_input(3.14) is True
        assert validate_input(0.0) is True

    def test_validate_input_boolean_types(self):
        """Test validation of boolean types."""
        assert validate_input(True) is True
        assert validate_input(False) is True

    def test_validate_input_other_types(self):
        """Test validation of other types."""
        assert validate_input(object()) is True
        assert validate_input(lambda x: x) is True
        assert validate_input(set()) is True  # Empty set but not list/dict

    @pytest.mark.parametrize("input_value,expected", [
        ("non-empty", True),
        ("", False),
        ([1, 2], True),
        ([], False),
        ({"key": "value"}, True),
        ({}, False),
        (None, False),
        (42, True),
        (0, True),
        (True, True),
        (False, True),
    ])
    def test_validate_input_parametrized(self, input_value: Any, expected: bool):
        """Test various input validation scenarios."""
        assert validate_input(input_value) == expected


class TestValidateCodeSnippet:
    """Test validate_code_snippet function."""

    def test_validate_code_snippet_valid_python(self):
        """Test validation of valid Python code."""
        code = """
def hello_world():
    return "Hello, World!"
        """.strip()
        
        result_code, warnings = validate_code_snippet(code, "python")
        
        assert result_code == code  # Code should be returned unchanged
        assert isinstance(warnings, list)
        assert len(warnings) == 0  # No warnings for valid code

    def test_validate_code_snippet_valid_javascript(self):
        """Test validation of JavaScript code."""
        code = """
function helloWorld() {
    return "Hello, World!";
}
        """.strip()
        
        result_code, warnings = validate_code_snippet(code, "javascript")
        
        assert result_code == code
        assert isinstance(warnings, list)
        assert len(warnings) == 0

    def test_validate_code_snippet_empty_code(self):
        """Test validation of empty code."""
        result_code, warnings = validate_code_snippet("", "python")
        
        assert result_code == ""
        assert isinstance(warnings, list)
        assert len(warnings) == 1
        assert "Code is empty" in warnings[0]

    def test_validate_code_snippet_whitespace_only(self):
        """Test validation of whitespace-only code."""
        result_code, warnings = validate_code_snippet("   \n\t  ", "python")
        
        assert result_code == "   \n\t  "
        assert isinstance(warnings, list)
        assert len(warnings) == 1
        assert "Code is empty" in warnings[0]

    def test_validate_code_snippet_language_parameter_unused(self):
        """Test that language parameter doesn't affect basic validation."""
        code = "print('hello')"
        
        # Test with different languages - should behave the same
        result1, warnings1 = validate_code_snippet(code, "python")
        result2, warnings2 = validate_code_snippet(code, "javascript")
        result3, warnings3 = validate_code_snippet(code, "go")
        result4, warnings4 = validate_code_snippet(code, "unknown")
        
        assert result1 == result2 == result3 == result4 == code
        assert warnings1 == warnings2 == warnings3 == warnings4 == []

    def test_validate_code_snippet_return_types(self):
        """Test that function returns correct types."""
        code = "test code"
        result = validate_code_snippet(code, "python")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        result_code, warnings = result
        assert isinstance(result_code, str)
        assert isinstance(warnings, list)

    @pytest.mark.parametrize("code,expected_warning_count", [
        ("def test(): pass", 0),
        ("", 1),
        ("   ", 1),
        ("\n\n\n", 1),
        ("print('hello')", 0),
        ("// JavaScript comment", 0),
        ("# Python comment", 0)
    ])
    def test_validate_code_snippet_parametrized(self, code: str, expected_warning_count: int):
        """Test various code validation scenarios."""
        result_code, warnings = validate_code_snippet(code, "python")
        
        assert result_code == code
        assert len(warnings) == expected_warning_count


class TestCodeValidatorsIntegration:
    """Integration tests combining validation functions."""

    def test_validate_input_then_code_snippet(self):
        """Test using validate_input before validate_code_snippet."""
        code_samples = [
            "def valid_function(): return True",
            "",
            "   ",
            None,
            ["not", "code"],
            42
        ]
        
        for code in code_samples:
            if validate_input(code):
                if isinstance(code, str):
                    result_code, warnings = validate_code_snippet(code, "python")
                    assert isinstance(result_code, str)
                    assert isinstance(warnings, list)
                # Non-string valid inputs can't be validated as code
            else:
                # Invalid inputs shouldn't be processed as code
                assert code is None or code == "" or code == []

    def test_validation_pipeline_example(self):
        """Example of a validation pipeline using both functions."""
        def validate_user_code(user_input: Any, language: str = "python") -> dict:
            """Example validation pipeline."""
            result = {
                "valid_input": False,
                "code": "",
                "warnings": [],
                "errors": []
            }
            
            # First, validate the input
            if not validate_input(user_input):
                result["errors"].append("Invalid input: empty or None")
                return result
            
            # Then validate as string for code processing
            if not isinstance(user_input, str):
                result["errors"].append("Input must be a string")
                return result
            
            result["valid_input"] = True
            
            # Finally, validate the code snippet
            code, warnings = validate_code_snippet(user_input, language)
            result["code"] = code
            result["warnings"] = warnings
            
            return result
        
        # Test the pipeline with various inputs
        test_cases = [
            ("def test(): pass", True, 0, 0),
            ("", False, 1, 0),  # Empty string fails input validation
            (None, False, 1, 0),
            (42, False, 1, 0),
            ("   ", True, 0, 1)  # Valid input but empty code triggers warning
        ]
        
        for input_data, should_be_valid_input, expected_errors, expected_warnings in test_cases:
            result = validate_user_code(input_data)
            
            assert result["valid_input"] == should_be_valid_input
            assert len(result["errors"]) == expected_errors
            
            if should_be_valid_input and isinstance(input_data, str):
                assert len(result["warnings"]) == expected_warnings