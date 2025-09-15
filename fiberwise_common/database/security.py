"""
SQL Security Layer - Input sanitization and SQL injection protection.
Integrates with QueryAdapter to provide automatic security checks.
"""

import re
import logging
import json
import sqlparse
import bleach
from typing import Any, List, Tuple, Union, Dict, Optional
from enum import Enum
from sqlparse import sql, tokens
import html
import urllib.parse

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security enforcement levels."""
    STRICT = "strict"      # Block dangerous operations
    MODERATE = "moderate"  # Log warnings, allow with sanitization
    PERMISSIVE = "permissive"  # Log only, minimal sanitization

class SQLSecurityError(Exception):
    """Raised when dangerous SQL patterns are detected."""
    pass

class InputSecurityError(Exception):
    """Raised when dangerous input values are detected."""
    pass

class SQLSecurityValidator:
    """Validates SQL queries for security threats."""
    
    # Dangerous SQL keywords that require special handling
    DANGEROUS_KEYWORDS = {
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'GRANT', 'REVOKE',
        'INSERT', 'UPDATE', 'EXEC', 'EXECUTE', 'DECLARE', 'BULK'
    }
    
    # SQL functions that can be dangerous
    DANGEROUS_FUNCTIONS = {
        'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE', 'SYSTEM', 'SHELL',
        'EXEC', 'xp_cmdshell', 'sp_executesql'
    }
    
    # Patterns that indicate SQL injection attempts
    INJECTION_PATTERNS = [
        r"(?i)(union\s+select)",           # UNION SELECT
        r"(?i)(;\s*drop\s+table)",        # ; DROP TABLE
        r"(?i)(;\s*delete\s+from)",       # ; DELETE FROM
        r"(?i)(;\s*insert\s+into)",       # ; INSERT INTO
        r"(?i)(;\s*update\s+\w+\s+set)",  # ; UPDATE ... SET
        r"(?i)(\bor\s+1\s*=\s*1\b)",      # OR 1=1
        r"(?i)(\band\s+1\s*=\s*2\b)",     # AND 1=2
        r"(?i)(\/\*.*\*\/)",              # SQL comments
        r"(?i)(--\s*.*$)",                # SQL line comments
        r"(?i)(\bconcat\s*\()",           # String concatenation
        r"(?i)(\bchar\s*\(\d+\))",        # CHAR() function
    ]
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MODERATE):
        self.security_level = security_level
    
    def validate_query(self, query: str, operation_type: str = "unknown") -> str:
        """
        Validate SQL query for security threats.
        
        Args:
            query: SQL query to validate
            operation_type: Type of operation (SELECT, INSERT, etc.)
            
        Returns:
            Validated query string
            
        Raises:
            SQLSecurityError: If dangerous patterns are detected
        """
        try:
            # Parse the SQL query
            parsed = sqlparse.parse(query)
            if not parsed:
                logger.warning(f"Could not parse SQL query: {query[:100]}...")
                return query
            
            parsed_statement = parsed[0]
            
            # Check for dangerous keywords
            self._check_dangerous_keywords(parsed_statement, query)
            
            # Check for injection patterns
            self._check_injection_patterns(query)
            
            # Check for dangerous functions
            self._check_dangerous_functions(parsed_statement, query)
            
            # Validate query structure
            self._validate_query_structure(parsed_statement, operation_type)
            
            logger.debug(f"SQL query validated successfully: {operation_type}")
            return query
            
        except sqlparse.exceptions.SQLParseError as e:
            logger.warning(f"SQL parse error: {e}, query: {query[:100]}...")
            if self.security_level == SecurityLevel.STRICT:
                raise SQLSecurityError(f"Invalid SQL syntax: {e}")
            return query
        except SQLSecurityError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in SQL validation: {e}")
            if self.security_level == SecurityLevel.STRICT:
                raise SQLSecurityError(f"SQL validation failed: {e}")
            return query
    
    def _check_dangerous_keywords(self, parsed_statement: sql.Statement, query: str):
        """Check for dangerous SQL keywords."""
        found_dangerous = []
        
        for token in parsed_statement.flatten():
            if token.ttype is tokens.Keyword:
                keyword = token.value.upper()
                if keyword in self.DANGEROUS_KEYWORDS:
                    found_dangerous.append(keyword)
        
        if found_dangerous:
            message = f"Dangerous SQL keywords detected: {', '.join(found_dangerous)}"
            logger.warning(f"{message} in query: {query[:100]}...")
            
            if self.security_level == SecurityLevel.STRICT:
                # In strict mode, block all dangerous operations
                raise SQLSecurityError(message)
            elif self.security_level == SecurityLevel.MODERATE:
                # In moderate mode, allow SELECT but be cautious with others
                if any(kw in ['DROP', 'TRUNCATE', 'ALTER', 'CREATE'] for kw in found_dangerous):
                    raise SQLSecurityError(f"Blocked dangerous operation: {message}")
    
    def _check_injection_patterns(self, query: str):
        """Check for SQL injection patterns."""
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                message = f"Potential SQL injection pattern detected: {pattern}"
                logger.warning(f"{message} in query: {query[:100]}...")
                
                if self.security_level == SecurityLevel.STRICT:
                    raise SQLSecurityError(message)
                elif self.security_level == SecurityLevel.MODERATE:
                    # Block obvious injection attempts
                    if re.search(r"(?i)(union\s+select|;\s*drop|or\s+1\s*=\s*1)", query):
                        raise SQLSecurityError(message)
    
    def _check_dangerous_functions(self, parsed_statement: sql.Statement, query: str):
        """Check for dangerous SQL functions."""
        for token in parsed_statement.flatten():
            if token.ttype is tokens.Name:
                func_name = token.value.upper()
                if func_name in self.DANGEROUS_FUNCTIONS:
                    message = f"Dangerous SQL function detected: {func_name}"
                    logger.warning(f"{message} in query: {query[:100]}...")
                    
                    if self.security_level in [SecurityLevel.STRICT, SecurityLevel.MODERATE]:
                        raise SQLSecurityError(message)
    
    def _validate_query_structure(self, parsed_statement: sql.Statement, operation_type: str):
        """Validate overall query structure."""
        # Check for multiple statements (potential injection)
        if ';' in str(parsed_statement) and operation_type.upper() in ['SELECT', 'INSERT', 'UPDATE']:
            if self.security_level in [SecurityLevel.STRICT, SecurityLevel.MODERATE]:
                raise SQLSecurityError("Multiple SQL statements not allowed in single query")

class InputSanitizer:
    """Sanitizes input values to prevent XSS and injection attacks."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MODERATE):
        self.security_level = security_level
        
        # Configure bleach for HTML sanitization
        self.allowed_tags = []  # No HTML tags allowed by default
        self.allowed_attributes = {}
    
    def sanitize_value(self, value: Any) -> Any:
        """
        Sanitize individual input values.
        
        Args:
            value: Input value to sanitize
            
        Returns:
            Sanitized value
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, (list, tuple)):
            return type(value)(self.sanitize_value(v) for v in value)
        elif isinstance(value, dict):
            return {k: self.sanitize_value(v) for k, v in value.items()}
        else:
            return value
    
    def sanitize_parameters(self, params: Union[tuple, list, dict]) -> Union[tuple, list, dict]:
        """
        Sanitize query parameters.
        
        Args:
            params: Parameters to sanitize
            
        Returns:
            Sanitized parameters
        """
        if isinstance(params, (tuple, list)):
            return type(params)(self.sanitize_value(param) for param in params)
        elif isinstance(params, dict):
            return {key: self.sanitize_value(value) for key, value in params.items()}
        else:
            return self.sanitize_value(params)
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string values."""
        if not value:
            return value
        
        # Check for potential XSS
        if self._contains_xss_patterns(value):
            message = f"Potential XSS pattern detected in input: {value[:50]}..."
            logger.warning(message)
            
            if self.security_level == SecurityLevel.STRICT:
                raise InputSecurityError(message)
        
        # Check if this looks like JSON data or simple identifiers - if so, skip HTML escaping
        if self._is_likely_json(value) or self._is_simple_identifier(value):
            sanitized = value
        else:
            # HTML entity encoding for complex strings that might contain XSS
            sanitized = html.escape(value, quote=True)
        
        # Remove potentially dangerous HTML tags
        sanitized = bleach.clean(
            sanitized,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        # URL decode to prevent double-encoding attacks
        try:
            decoded = urllib.parse.unquote(sanitized)
            if decoded != sanitized:
                # If URL decoding changed the string, it might be encoded
                logger.debug(f"URL decoded input: {sanitized} -> {decoded}")
                sanitized = decoded
        except Exception:
            pass  # If decoding fails, use original
        
        return sanitized
    
    def _is_likely_json(self, value: str) -> bool:
        """
        Check if a string is likely to be JSON data.
        
        Args:
            value: String to check
            
        Returns:
            True if the string appears to be JSON
        """
        if not value:
            return False
        
        # Trim whitespace
        trimmed = value.strip()
        
        # Check if it starts and ends with JSON object/array delimiters
        if (trimmed.startswith('{') and trimmed.endswith('}')) or \
           (trimmed.startswith('[') and trimmed.endswith(']')):
            # Quick validation - try to parse as JSON
            try:
                json.loads(trimmed)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        
        return False
    
    def _is_simple_identifier(self, value: str) -> bool:
        """
        Check if a string is a simple identifier that doesn't need HTML escaping.
        
        This includes:
        - UUIDs (with or without hyphens)
        - Simple alphanumeric strings
        - Database table/column names
        - Numbers
        
        Args:
            value: String to check
            
        Returns:
            True if the string is a simple identifier
        """
        if not value:
            return False
        
        # Remove whitespace
        trimmed = value.strip()
        
        # Check for UUIDs (with or without hyphens)
        uuid_pattern = r'^[a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12}$'
        if re.match(uuid_pattern, trimmed):
            return True
        
        # Check for simple alphanumeric identifiers (table names, column names, etc.)
        # Allow letters, numbers, underscores, hyphens
        simple_pattern = r'^[a-zA-Z0-9_-]+$'
        if re.match(simple_pattern, trimmed):
            return True
        
        # Check for numbers (integers or floats)
        try:
            float(trimmed)
            return True
        except ValueError:
            pass
        
        return False
    
    def _contains_xss_patterns(self, value: str) -> bool:
        """Check for common XSS patterns."""
        xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"<iframe[^>]*>",
            r"eval\s*\(",
            r"expression\s*\(",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False

class SecureQueryAdapter:
    """Enhanced QueryAdapter with built-in security validation."""
    
    def __init__(self, target_style, security_level: SecurityLevel = SecurityLevel.MODERATE):
        # Import the factory function to create query adapter properly
        from .query_adapter import create_query_adapter
        self.query_adapter = create_query_adapter(target_style)
        
        # Initialize security components
        self.security_level = security_level
        self.sql_validator = SQLSecurityValidator(security_level)
        self.input_sanitizer = InputSanitizer(security_level)
    
    def adapt_query_and_params(self, query: str, params: Any = None, 
                              source_style=None) -> Tuple[str, Any]:
        """
        Secure version of adapt_query_and_params.
        
        Args:
            query: SQL query to adapt
            params: Query parameters
            source_style: Source parameter style
            
        Returns:
            Tuple of (adapted_query, sanitized_params)
        """
        # Determine operation type from query
        operation_type = self._extract_operation_type(query)
        
        # Validate SQL query for security
        validated_query = self.sql_validator.validate_query(query, operation_type)
        
        # Sanitize input parameters
        sanitized_params = None
        if params is not None:
            sanitized_params = self.input_sanitizer.sanitize_parameters(params)
        
        # Apply original query adaptation
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(
            validated_query, sanitized_params, source_style
        )
        
        logger.debug(f"Secure query adaptation completed: {operation_type}")
        return adapted_query, adapted_params
    
    def _extract_operation_type(self, query: str) -> str:
        """Extract SQL operation type from query."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith('SELECT'):
            return 'SELECT'
        elif query_upper.startswith('INSERT'):
            return 'INSERT'
        elif query_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif query_upper.startswith('DELETE'):
            return 'DELETE'
        elif query_upper.startswith('CREATE'):
            return 'CREATE'
        elif query_upper.startswith('DROP'):
            return 'DROP'
        elif query_upper.startswith('ALTER'):
            return 'ALTER'
        else:
            return 'UNKNOWN'
    
    # Delegate other methods to the original adapter
    def __getattr__(self, name):
        return getattr(self.query_adapter, name)

def create_secure_query_adapter(target_style, security_level: SecurityLevel = SecurityLevel.MODERATE):
    """Factory function to create secure query adapter."""
    return SecureQueryAdapter(target_style, security_level)