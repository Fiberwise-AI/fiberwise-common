"""
Security Configuration for Database Operations.
Provides environment-based configuration for SQL security levels.
"""

import os
import logging
from typing import Dict, List, Set
from .security import SecurityLevel

logger = logging.getLogger(__name__)

class DatabaseSecurityConfig:
    """Configuration class for database security settings."""
    
    def __init__(self):
        # Get security level from environment
        env_level = os.getenv("DB_SECURITY_LEVEL", "moderate").lower()
        self.security_level = self._parse_security_level(env_level)
        
        # Configure allowed operations per security level
        self.allowed_operations = self._get_allowed_operations()
        
        # Configure dangerous keywords based on environment
        self.additional_dangerous_keywords = self._get_additional_keywords()
        
        # Configure allowed HTML tags for input sanitization
        self.allowed_html_tags = self._get_allowed_html_tags()
        
        # Log current configuration
        logger.info(f"Database security configured: level={self.security_level.value}")
    
    def _parse_security_level(self, level_str: str) -> SecurityLevel:
        """Parse security level from string."""
        level_mapping = {
            "strict": SecurityLevel.STRICT,
            "moderate": SecurityLevel.MODERATE,
            "permissive": SecurityLevel.PERMISSIVE,
        }
        
        return level_mapping.get(level_str, SecurityLevel.MODERATE)
    
    def _get_allowed_operations(self) -> Dict[str, Set[str]]:
        """Get allowed SQL operations per security level."""
        base_read_ops = {"SELECT", "EXPLAIN", "DESCRIBE", "SHOW"}
        
        if self.security_level == SecurityLevel.STRICT:
            return {
                "development": base_read_ops | {"INSERT", "UPDATE"},
                "production": base_read_ops,
            }
        elif self.security_level == SecurityLevel.MODERATE:
            return {
                "development": base_read_ops | {"INSERT", "UPDATE", "DELETE"},
                "production": base_read_ops | {"INSERT", "UPDATE", "DELETE"},
            }
        else:  # PERMISSIVE
            return {
                "development": base_read_ops | {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"},
                "production": base_read_ops | {"INSERT", "UPDATE", "DELETE"},
            }
    
    def _get_additional_keywords(self) -> Set[str]:
        """Get additional dangerous keywords from environment."""
        env_keywords = os.getenv("DB_DANGEROUS_KEYWORDS", "")
        if env_keywords:
            return set(keyword.strip().upper() for keyword in env_keywords.split(","))
        return set()
    
    def _get_allowed_html_tags(self) -> List[str]:
        """Get allowed HTML tags for input sanitization."""
        if self.security_level == SecurityLevel.STRICT:
            return []  # No HTML tags allowed
        elif self.security_level == SecurityLevel.MODERATE:
            return ["b", "i", "em", "strong", "p", "br"]  # Basic formatting
        else:  # PERMISSIVE
            return ["b", "i", "em", "strong", "p", "br", "a", "ul", "ol", "li"]
    
    def is_operation_allowed(self, operation: str, environment: str = None) -> bool:
        """Check if SQL operation is allowed in current environment."""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        allowed_ops = self.allowed_operations.get(environment, set())
        return operation.upper() in allowed_ops
    
    def get_max_query_length(self) -> int:
        """Get maximum allowed query length."""
        if self.security_level == SecurityLevel.STRICT:
            return 1000
        elif self.security_level == SecurityLevel.MODERATE:
            return 5000
        else:  # PERMISSIVE
            return 10000
    
    def get_max_parameter_count(self) -> int:
        """Get maximum allowed parameter count."""
        if self.security_level == SecurityLevel.STRICT:
            return 50
        elif self.security_level == SecurityLevel.MODERATE:
            return 100
        else:  # PERMISSIVE
            return 200

# Global configuration instance
security_config = DatabaseSecurityConfig()

# Environment-based factory functions
def get_security_level_for_environment() -> SecurityLevel:
    """Get security level based on current environment."""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # Production defaults to stricter security
        return SecurityLevel.STRICT if security_config.security_level == SecurityLevel.PERMISSIVE else security_config.security_level
    else:
        # Development can use configured level
        return security_config.security_level

def create_environment_aware_security_config() -> DatabaseSecurityConfig:
    """Create security config that adapts to environment."""
    config = DatabaseSecurityConfig()
    config.security_level = get_security_level_for_environment()
    return config