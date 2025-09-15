"""
User management utilities for FiberWise CLI operations.
"""

import uuid
import getpass
import os
from typing import Optional, Dict, Any


def get_system_user_info() -> tuple[str, str, str]:
    """Get current system user information."""
    system_username = getpass.getuser()
    hostname = os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))
    # Create email without using "admin" - use actual username
    user_email = f"{system_username}@{hostname}.fiberwise.dev"
    return system_username, hostname, user_email


def create_cli_default_user(db_service, verbose: bool = False) -> Optional[int]:
    """
    Create a default user based on current logged-in user and generate app ID.
    
    Args:
        db_service: Database service instance
        verbose: Whether to enable verbose output
        
    Returns:
        User ID if successful, None if failed
    """
    system_username, hostname, user_email = get_system_user_info()
    
    # Check if any user already exists
    existing_users = db_service.fetch_all("SELECT id, username FROM users")
    
    if not existing_users:
        # No users exist, create the current system user
        user_uuid = str(uuid.uuid4())
        
        if verbose:
            print(f"Creating user: {system_username} ({user_email})")
        
        # Create user - use actual system username, not "admin"
        success = db_service.execute(
            """INSERT INTO users (uuid, username, email, display_name, is_active, is_admin, created_at, updated_at) 
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))""",
            (user_uuid, system_username, user_email, 
             system_username.title(), True, True)
        )
        
        if not success:
            if verbose:
                print(f"[ERROR] Failed to create user: {system_username}")
            return None
        
        # Get the user ID properly
        created_user = db_service.fetch_one(
            "SELECT id FROM users WHERE username = ?", 
            (system_username,)
        )
        
        if created_user:
            user_id = created_user['id']
            if verbose:
                print(f"[OK] Created user ID: {user_id}")
            return user_id
        else:
            if verbose:
                print(f"[ERROR] Could not retrieve created user ID")
            return None
            
    elif len(existing_users) == 1:
        # Exactly one user exists - use it regardless of username
        user_id = existing_users[0]['id']
        existing_username = existing_users[0]['username']
        if verbose:
            print(f"Using existing user: {existing_username} (ID: {user_id})")
        return user_id
            
    else:
        # Multiple users exist - this shouldn't happen in CLI setup
        # Use the first user
        user_id = existing_users[0]['id']
        if verbose:
            print(f"Multiple users found, using first: {existing_users[0]['username']} (ID: {user_id})")
        return user_id


def create_cli_default_app(db_service, user_id: int, verbose: bool = False) -> Optional[str]:
    """
    Create a default FiberWise app for CLI operations.
    
    Args:
        db_service: Database service instance
        user_id: ID of the user who will own the app
        verbose: Whether to enable verbose output
        
    Returns:
        App ID if successful, None if failed
    """
    # Import CLI constants - need to add path for CLI constants
    import sys
    from pathlib import Path
    
    # Add the fiberwise package to path to import CLI constants
    # This assumes we're running from within the fiberwise environment
    try:
        from fiberwise.common.constants.cli import (
            CLI_APP_UUID, 
            CLI_APP_SLUG,
            get_cli_app_name, 
            get_cli_app_description,
            CLI_APP_VERSION
        )
    except ImportError:
        # Fallback constants if import fails
        CLI_APP_UUID = "fiberwise-cli-default"
        CLI_APP_SLUG = "fiberwise-cli"
        CLI_APP_VERSION = "1.0.0"
        
        system_username, hostname, _ = get_system_user_info()
        
        def get_cli_app_name():
            return f"FiberWise - {hostname.upper()}"
        
        def get_cli_app_description():
            return f"FiberWise CLI application for {system_username} on {hostname}"
    
    # Check if any FiberWise apps already exist
    existing_apps = db_service.fetch_all("SELECT app_id, name FROM apps")
    
    if not existing_apps:
        # No CLI apps exist, create the hostname-based app
        app_name = get_cli_app_name()  # This will be "FiberWise - HOSTNAME"
        app_description = get_cli_app_description()  # Includes hostname and username
        
        if verbose:
            print(f"Creating FiberWise app: {app_name}")
        
        success = db_service.execute(
            """INSERT INTO apps (app_id, app_slug, name, description, version, creator_user_id, created_at, updated_at) 
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))""",
            (CLI_APP_UUID, CLI_APP_SLUG, app_name, app_description, 
             CLI_APP_VERSION, user_id)
        )
        
        if success:
            # Get the app ID properly
            created_app = db_service.fetch_one(
                "SELECT app_id FROM apps WHERE app_id = ?", 
                (CLI_APP_UUID,)
            )
            
            if created_app:
                app_id = created_app['app_id']
                if verbose:
                    print(f"[OK] Created app ID: {app_id}")
                return app_id
            else:
                if verbose:
                    print(f"[ERROR] Could not retrieve created app ID")
                return None
        else:
            if verbose:
                print(f"[ERROR] Failed to create CLI app")
            return None
                
    elif len(existing_apps) == 1:
        # Exactly one CLI app exists - use it
        app_id = existing_apps[0]['app_id']
        if verbose:
            print(f"Using existing FiberWise app: {existing_apps[0]['name']} (ID: {app_id})")
        return app_id
    else:
        # Multiple CLI apps exist - use the first one
        app_id = existing_apps[0]['app_id']
        if verbose:
            print(f"Multiple FiberWise apps found, using first: {existing_apps[0]['name']} (ID: {app_id})")
        return app_id


def setup_cli_user_and_app(db_service, verbose: bool = False) -> tuple[Optional[int], Optional[str]]:
    """
    Set up default CLI user and app in one operation.
    
    Args:
        db_service: Database service instance
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (user_id, app_id) if successful, (None, None) if failed
    """
    # Create or get user
    user_id = create_cli_default_user(db_service, verbose)
    if not user_id:
        return None, None
    
    # Create or get app
    app_id = create_cli_default_app(db_service, user_id, verbose)
    if not app_id:
        return user_id, None
    
    return user_id, app_id
