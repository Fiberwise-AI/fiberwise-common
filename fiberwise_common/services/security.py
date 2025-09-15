"""
Shared Security Module - Consolidated password utilities
Provides secure password hashing and verification functions.
"""

import logging
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Initialize password context at module level for efficiency
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hashed version.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to verify against

    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a plain password using bcrypt.

    Args:
        password: The plain text password to hash

    Returns:
        str: The hashed password
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        raise 