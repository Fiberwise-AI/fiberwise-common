"""
Constants for FiberWise CLI application.
"""

import os
import getpass

# CLI App Constants - Static FIBERWISE_APP_ID for CLI (not from environment)
CLI_APP_UUID = "F1BE8ACD-E0F1-4A2B-915E-C3D4E5F6A7B8"  # Static FIBERWISE_APP_ID constant
FIBERWISE_APP_ID = CLI_APP_UUID  # Alias for consistency
CLI_APP_SLUG = "fiberwise-cli"
CLI_APP_NAME_TEMPLATE = "FiberWise - {hostname}"
CLI_APP_DESCRIPTION_TEMPLATE = "FiberWise application on {hostname} (user: {username})"
CLI_APP_VERSION = "1.0.0"
# Note: app_type removed - core web schema doesn't use it

def get_cli_app_name() -> str:
    """Get the FiberWise app name with hostname."""
    hostname = os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))
    return CLI_APP_NAME_TEMPLATE.format(hostname=hostname)

def get_cli_app_description() -> str:
    """Get the FiberWise app description with hostname and username."""
    hostname = os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))
    username = getpass.getuser()
    return CLI_APP_DESCRIPTION_TEMPLATE.format(hostname=hostname, username=username)

def get_cli_user_email() -> str:
    """Get the CLI user email based on username and hostname."""
    hostname = os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))
    username = getpass.getuser()
    return f"{username}@{hostname}.fiberwise.dev"
