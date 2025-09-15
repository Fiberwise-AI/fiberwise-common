"""
User Service - extracted from fiberwise-core-web
Manages user accounts, authentication, and user-related operations.
"""

import logging
import re
import secrets
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import uuid

from .base_service import BaseService, ServiceError, NotFoundError, ValidationError
from .security import get_password_hash, verify_password
from ..database.query_adapter import QueryAdapter, ParameterStyle

logger = logging.getLogger(__name__)


class UserService(BaseService):
    """
    Service for managing users and user-related operations.
    Handles user CRUD operations, authentication helpers, and user data management.
    """
    
    def __init__(self, db_provider):
        super().__init__(db_provider)
        # Initialize query adapter for SQLite (since we're using SQLite)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)

    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user by ID (excluding password).
        
        Args:
            user_id: User ID
            
        Returns:
            User record or None if not found
        """
        query = """
            SELECT id, uuid, email, username, is_active, is_superuser, is_verified,
                   first_name, last_name, full_name, avatar_url, 
                   timezone, locale, created_at, updated_at
            FROM users WHERE id = $1
        """
        converted_query = self.query_adapter.convert_query(query)
        return await self.db.fetch_one(converted_query, user_id)

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email (excluding password).
        
        Args:
            email: User email
            
        Returns:
            User record or None if not found
        """
        query = """
            SELECT id, email, username, is_active, is_superuser, is_verified,
                   first_name, last_name, full_name, avatar_url,
                   timezone, locale, created_at, updated_at
            FROM users WHERE email = $1
        """
        converted_query = self.query_adapter.convert_query(query)
        return await self.db.fetch_one(converted_query, email)

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username (excluding password).
        
        Args:
            username: Username
            
        Returns:
            User record or None if not found
        """
        query = """
            SELECT id, username, email, is_active, is_superuser, is_verified,
                   first_name, last_name, full_name, avatar_url,
                   timezone, locale, created_at, updated_at
            FROM users WHERE username = $1
        """
        converted_query = self.query_adapter.convert_query(query)
        return await self.db.fetch_one(converted_query, username)

    async def get_user_by_email_or_username(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email or username (excluding password).
        
        Args:
            identifier: Email or username
            
        Returns:
            User record or None if not found
        """
        # First try email
        user = await self.get_user_by_email(identifier)
        if user:
            return user
        
        # Then try username
        user = await self.get_user_by_username(identifier)
        if user:
            return user
            
        return None

    async def get_user_by_email_or_username_with_password(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email or username including hashed password (for authentication).
        This is the method the web auth system expects for login.
        
        Args:
            identifier: Email or username
            
        Returns:
            User record with password or None if not found
        """
        # First try email
        user = await self.get_user_with_password(identifier)
        if user:
            return user
        
        # Then try username
        user = await self.get_user_with_password_by_username(identifier)
        if user:
            return user
            
        return None

    async def get_user_with_password(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email including hashed password (for authentication).
        
        Args:
            email: User email
            
        Returns:
            User record with password or None if not found
        """
        query = "SELECT * FROM users WHERE email = $1"
        converted_query = self.query_adapter.convert_query(query)
        return await self.db.fetch_one(converted_query, email)

    async def get_user_with_password_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username including hashed password (for authentication).
        
        Args:
            username: Username
            
        Returns:
            User record with password or None if not found
        """
        query = "SELECT * FROM users WHERE username = $1"
        converted_query = self.query_adapter.convert_query(query)
        return await self.db.fetch_one(converted_query, username)

    async def get_user_with_password_by_email_or_username(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email or username including hashed password (for authentication).
        
        Args:
            identifier: Email or username
            
        Returns:
            User record with password or None if not found
        """
        # First try email
        user = await self.get_user_with_password(identifier)
        if user:
            return user
        
        # Then try username
        user = await self.get_user_with_password_by_username(identifier)
        if user:
            return user
            
        return None

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            user_data: User data dictionary containing email, hashed_password, etc.
            
        Returns:
            Created user record
            
        Raises:
            ValidationError: If user already exists or validation fails
        """
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data['email'])
        if existing_user:
            raise ValidationError("Email already registered")
        
        # Validate email format
        if not self._is_valid_email(user_data['email']):
            raise ValidationError("Invalid email format")
        
        # Hash password if provided as plain text
        if 'password' in user_data and 'hashed_password' not in user_data:
            user_data['hashed_password'] = get_password_hash(user_data['password'])
        elif 'password' in user_data and 'hashed_password' in user_data:
            # If both are provided, prefer the hashed version but ensure password is hashed
            if not user_data['hashed_password']:
                user_data['hashed_password'] = get_password_hash(user_data['password'])
        
        # Generate full_name if not provided
        if not user_data.get('full_name'):
            first = user_data.get('first_name', '')
            last = user_data.get('last_name', '')
            user_data['full_name'] = f"{first} {last}".strip()
        
        now = datetime.now().isoformat()
        user_id = None  # Let database auto-increment
        user_uuid = str(uuid.uuid4())  # Generate UUID for the user
        
        query = """
            INSERT INTO users (
                uuid, email, username, hashed_password, first_name, last_name, full_name,
                is_active, is_superuser, is_verified, avatar_url,
                timezone, locale, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """
        
        converted_query = self.query_adapter.convert_query(query)
        await self.db.execute(
            converted_query,
            user_uuid,
            user_data['email'],
            user_data.get('username'),
            user_data.get('hashed_password'),
            user_data.get('first_name'),
            user_data.get('last_name'),
            user_data['full_name'],
            user_data.get('is_active', True),
            user_data.get('is_superuser', False),
            user_data.get('is_verified', False),
            user_data.get('avatar_url'),
            user_data.get('timezone', 'UTC'),
            user_data.get('locale', 'en'),
            now,
            now
        )
        
        # Return the created user (without password)
        created_user = await self.get_user_by_email(user_data['email'])
        if not created_user:
            raise ServiceError("Failed to create user")
        
        # Create a default organization for the user
        await self._create_default_organization(created_user)
        
        return created_user

    async def update_user(self, user_id: int, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing user.
        
        Args:
            user_id: User ID to update
            user_data: Updated user data
            
        Returns:
            Updated user record
            
        Raises:
            NotFoundError: If user not found
        """
        # Check if user exists
        existing_user = await self.get_user_by_id(user_id)
        if not existing_user:
            raise NotFoundError(f"User with id {user_id} not found")
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        updateable_fields = [
            'first_name', 'last_name', 'full_name', 'avatar_url',
            'timezone', 'locale', 'is_active', 'is_verified'
        ]
        
        for field in updateable_fields:
            if field in user_data:
                update_fields.append(f"{field} = ?")
                params.append(user_data[field])
        
        if update_fields:
            update_fields.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(user_id)
            
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            await self._execute_query(query, tuple(params))
        
        return await self.get_user_by_id(user_id)

    async def update_user_password(self, user_id: int, hashed_password: str) -> None:
        """
        Update user's password.
        
        Args:
            user_id: User ID
            hashed_password: New hashed password
            
        Raises:
            NotFoundError: If user not found
        """
        # Check if user exists
        existing_user = await self.get_user_by_id(user_id)
        if not existing_user:
            raise NotFoundError(f"User with id {user_id} not found")
        
        query = "UPDATE users SET hashed_password = ?, updated_at = ? WHERE id = ?"
        await self._execute_query(query, (
            hashed_password, 
            datetime.now().isoformat(), 
            user_id
        ))

    async def delete_user(self, user_id: int) -> bool:
        """
        Soft delete a user (set is_active = False).
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            NotFoundError: If user not found
        """
        # Check if user exists
        existing_user = await self.get_user_by_id(user_id)
        if not existing_user:
            raise NotFoundError(f"User with id {user_id} not found")
        
        query = "UPDATE users SET is_active = 0, updated_at = ? WHERE id = ?"
        await self._execute_query(query, (datetime.now().isoformat(), user_id))
        return True

    async def get_all_users(
        self, 
        limit: int = 50, 
        offset: int = 0,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all users with pagination.
        
        Args:
            limit: Maximum number of users to return
            offset: Pagination offset
            include_inactive: Whether to include inactive users
            
        Returns:
            List of user records
        """
        query = """
            SELECT id, email, is_active, is_superuser, is_verified,
                   first_name, last_name, full_name, avatar_url,
                   timezone, locale, created_at, updated_at
            FROM users 
        """
        params = []
        
        if not include_inactive:
            query += " WHERE is_active = 1"
        
        query += f" ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
        
        return await self._fetch_all(query, tuple(params))

    async def search_users(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search users by email, name, or full name.
        
        Args:
            search_term: Search term
            limit: Maximum results
            
        Returns:
            List of matching user records
        """
        query = """
            SELECT id, email, is_active, is_superuser, is_verified,
                   first_name, last_name, full_name, avatar_url,
                   timezone, locale, created_at, updated_at
            FROM users 
            WHERE is_active = 1 
            AND (
                email LIKE ? OR 
                first_name LIKE ? OR 
                last_name LIKE ? OR 
                full_name LIKE ?
            )
            ORDER BY full_name
            LIMIT ?
        """
        
        search_pattern = f"%{search_term}%"
        return await self._fetch_all(query, (
            search_pattern, search_pattern, search_pattern, search_pattern, limit
        ))

    async def get_user_apps(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all apps accessible to a user (created or installed).
        
        Args:
            user_id: User ID
            
        Returns:
            List of app records
        """
        query = """
            SELECT DISTINCT a.app_id, a.app_slug, a.name, a.description, 
                   a.version, a.created_at, a.updated_at,
                   CASE WHEN a.creator_user_id = ? THEN 'creator' ELSE 'installed' END as access_type
            FROM apps a
            LEFT JOIN app_installations ai ON a.app_id = ai.app_id
            WHERE a.creator_user_id = ? OR (ai.installed_by_user_id = ? AND ai.is_active = 1)
            ORDER BY a.name
        """
        return await self._fetch_all(query, (user_id, user_id, user_id))

    async def get_user_apps_and_routes(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all apps available to the user with their route information.
        Similar to get_all_apps in users.py but designed to be used as a service.
        """
        try:
            # Fetch apps for the user, including installed apps
            query = """
            SELECT a.app_id, a.app_slug, a.name, a.description, a.version, 
                   a.entry_point_url, av.manifest_yaml 
            FROM apps a
            LEFT JOIN app_versions av ON a.app_id = av.app_id 
                AND av.app_version_id = (
                    SELECT av2.app_version_id 
                    FROM app_versions av2 
                    WHERE av2.app_id = a.app_id 
                    ORDER BY av2.created_at DESC 
                    LIMIT 1
                )
            ORDER BY a.name
            """
            apps_records = await self._fetch_all(query)
            
            result_apps = []
            for app_record in apps_records:
                app_dict = dict(app_record)
                
                # Extract route information from manifest if available
                routes = []
                if app_dict.get("manifest_yaml"):
                    import yaml
                    from ..models import UnifiedManifest
                    
                    manifest_data = yaml.safe_load(app_dict["manifest_yaml"])
                    if manifest_data is None:
                        print(f"Warning: yaml.safe_load returned None for manifest_yaml in app {app_dict.get('app_id')}")
                        continue
                    
                    # Parse with UnifiedManifest model
                    unified_manifest = UnifiedManifest.model_validate(manifest_data)
                    routes = unified_manifest.app.routes or []
                
                # Add routes to app info
                app_info = {
                    "app_id": app_dict["app_id"],
                    "app_slug": app_dict["app_slug"],
                    "name": app_dict["name"],
                    "description": app_dict["description"],
                    "version": app_dict["version"],
                    "entry_point_url": app_dict["entry_point_url"],
                    "routes": routes
                }
                result_apps.append(app_info)
                
            return result_apps
            
        except Exception as e:
            logger.error(f"Error retrieving user apps: {e}", exc_info=True)
            return []

    async def is_path_app_route(self, path: str, user_id: int) -> bool:
        """
        Check if the given path corresponds to a route in one of the user's installed apps.
        
        Args:
            path: The URL path to check
            user_id: The user's ID
            
        Returns:
            True if the path matches an app route, False otherwise
        """
        # Normalize path (remove leading/trailing slashes, handle empty path)
        normalized_path = path.strip('/')
        if not normalized_path:
            normalized_path = '/'
        
        # Get all user's apps with their routes
        user_apps = await self.get_user_apps_and_routes(user_id)
        
        # Check each app's routes for a match
        for app in user_apps:
            app_routes = app.get('routes', [])
            app_slug = app.get('app_slug', '')
            
            # Check if path starts with app slug (app_slug/...)
            app_slug_prefix = f"{app_slug}/"
            is_app_slug_path = False
            remaining_path = normalized_path
            
            if normalized_path.startswith(app_slug_prefix):
                is_app_slug_path = True
                # Get the part of the path after the app_slug/
                remaining_path = normalized_path[len(app_slug_prefix):]
                if not remaining_path:
                    remaining_path = '/'
            
            for route in app_routes:
                try:
                    # Properly access attributes for both dicts and Pydantic models
                    if hasattr(route, 'path'):
                        # It's a Pydantic model
                        route_path = route.path or ''
                    else:
                        # It's a dictionary
                        route_path = route.get('path', '')
                    
                    # Normalize route path
                    route_path = route_path.strip('/')
                    if not route_path:
                        route_path = '/'
                    
                    # Direct 1:1 comparison of paths (both with and without app slug prefix)
                    if normalized_path == route_path or (is_app_slug_path and remaining_path == route_path):
                        logger.info(f"Path '{path}' matches app route '{route_path}' for app '{app['name']}'")
                        return True
                    
                    # Also check for dynamic routes with parameters
                    if ':' in route_path:
                        # Convert route path to regex pattern
                        # Fix the invalid escape sequence by using re.escape for the whole pattern
                        pattern = re.escape(route_path)
                        # Then replace the escaped colon and parameter part with the capture group
                        pattern = re.sub(re.escape(':') + r'([^/]+)', r'([^/]+)', pattern)
                        pattern = f"^{pattern}$"
                        
                        # Check both with and without app slug prefix
                        if re.match(pattern, normalized_path) or (is_app_slug_path and re.match(pattern, remaining_path)):
                            logger.info(f"Path '{path}' matches dynamic app route '{route_path}' for app '{app['name']}'")
                            return True
                except Exception as e:
                    logger.error(f"Error checking route for app {app['app_id']}: {e}", exc_info=True)
        
        return False

    async def verify_user(self, user_id: int) -> Dict[str, Any]:
        """
        Mark a user as verified.
        
        Args:
            user_id: User ID to verify
            
        Returns:
            Updated user record
        """
        query = "UPDATE users SET is_verified = 1, updated_at = ? WHERE id = ?"
        await self._execute_query(query, (datetime.now().isoformat(), user_id))
        
        user = await self.get_user_by_id(user_id)
        if not user:
            raise NotFoundError(f"User with id {user_id} not found")
        
        return user

    async def register_user(self, registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user with password hashing and validation.
        
        Args:
            registration_data: Registration data including email, password, etc.
            
        Returns:
            Created user record (without password)
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate required fields
        required_fields = ['email', 'password']
        for field in required_fields:
            if not registration_data.get(field):
                raise ValidationError(f"Field '{field}' is required")
        
        # Validate password strength
        password = registration_data['password']
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters")
        
        # Validate email format
        if not self._is_valid_email(registration_data['email']):
            raise ValidationError("Invalid email format")
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(registration_data['email'])
        if existing_user:
            raise ValidationError("Email already registered")
        
        # Hash the password
        hashed_password = get_password_hash(password)
        
        # Prepare user data
        user_data = {
            'email': registration_data['email'],
            'hashed_password': hashed_password,
            'first_name': registration_data.get('first_name'),
            'last_name': registration_data.get('last_name'),
            'username': registration_data.get('username'),
            'is_active': True,
            'is_verified': False,  # New users start unverified
            'is_superuser': False
        }
        
        # Create the user
        created_user = await self.create_user(user_data)
        
        logger.info(f"New user registered: {created_user['email']} (ID: {created_user['id']})")
        return created_user

    async def authenticate_user(self, identifier: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with email/username and password.
        
        Args:
            identifier: Email or username
            password: Plain text password
            
        Returns:
            User record if authentication successful, None otherwise
        """
        # Get user with password
        user = await self.get_user_by_email_or_username_with_password(identifier)
        
        if not user:
            logger.warning(f"Authentication failed: user not found for identifier {identifier}")
            return None
        
        # Check if user is active
        if not user.get('is_active'):
            logger.warning(f"Authentication failed: inactive user {identifier}")
            return None
        
        # Verify password
        if not verify_password(password, user.get('hashed_password')):
            logger.warning(f"Authentication failed: incorrect password for {identifier}")
            return None
        
        # Remove password from response
        user_without_password = {k: v for k, v in user.items() if k != 'hashed_password'}
        
        logger.info(f"User authenticated successfully: {user['email']} (ID: {user['id']})")
        return user_without_password

    async def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """
        Change a user's password after verifying current password.
        
        Args:
            user_id: User ID
            current_password: Current plain text password
            new_password: New plain text password
            
        Returns:
            True if password changed successfully
            
        Raises:
            NotFoundError: If user not found
            ValidationError: If current password is incorrect or new password is weak
        """
        # Get user with password
        user = await self.get_user_by_id(user_id)
        if not user:
            raise NotFoundError(f"User with id {user_id} not found")
        
        # Get user with password for verification
        user_with_password = await self.get_user_with_password(user['email'])
        
        # Verify current password
        if not verify_password(current_password, user_with_password.get('hashed_password')):
            raise ValidationError("Current password is incorrect")
        
        # Validate new password
        if len(new_password) < 8:
            raise ValidationError("New password must be at least 8 characters")
        
        # Hash new password and update
        new_hashed_password = get_password_hash(new_password)
        await self.update_user_password(user_id, new_hashed_password)
        
        logger.info(f"Password changed for user: {user['email']} (ID: {user_id})")
        return True

    async def request_password_reset(self, email: str) -> Optional[str]:
        """
        Request a password reset token for a user.
        
        Args:
            email: User email
            
        Returns:
            Reset token if user exists, None otherwise (don't reveal if user exists)
        """
        user = await self.get_user_by_email(email)
        
        if not user:
            # Don't reveal whether user exists
            logger.info(f"Password reset requested for non-existent email: {email}")
            return None
        
        # Generate secure random token
        reset_token = secrets.token_urlsafe(32)
        
        # In a full implementation, you would store this token in a database table
        # with an expiration time. For now, we'll just return it.
        # TODO: Implement password_reset_tokens table
        
        logger.info(f"Password reset requested for user: {email} (ID: {user['id']})")
        return reset_token

    async def confirm_password_reset(self, token: str, new_password: str, user_id: int) -> bool:
        """
        Confirm password reset with token and set new password.
        
        Args:
            token: Reset token
            new_password: New plain text password
            user_id: User ID (in practice, this would be looked up by token)
            
        Returns:
            True if password reset successfully
            
        Raises:
            ValidationError: If token is invalid or password is weak
            NotFoundError: If user not found
        """
        # In a full implementation, you would:
        # 1. Look up the token in password_reset_tokens table
        # 2. Check if token is expired
        # 3. Get the associated user_id
        # For now, we'll just validate the user_id and update the password
        
        user = await self.get_user_by_id(user_id)
        if not user:
            raise NotFoundError(f"User with id {user_id} not found")
        
        # Validate new password
        if len(new_password) < 8:
            raise ValidationError("New password must be at least 8 characters")
        
        # Hash new password and update
        new_hashed_password = get_password_hash(new_password)
        await self.update_user_password(user_id, new_hashed_password)
        
        # In a full implementation, you would also delete the used token
        
        logger.info(f"Password reset completed for user: {user['email']} (ID: {user_id})")
        return True

    async def get_user_from_token(self, token: str, settings=None) -> Optional[Dict[str, Any]]:
        """
        Get user from JWT token.
        
        Args:
            token: JWT token string
            settings: Settings object with SECRET_KEY and ALGORITHM (optional)
            
        Returns:
            User record or None if token is invalid
        """
        try:
            if not token:
                return None
                
            # Import JWT libraries here to avoid dependency issues
            from jose import JWTError, jwt
            
            # Use settings if provided, otherwise fall back to defaults
            if settings:
                SECRET_KEY = settings.SECRET_KEY
                ALGORITHM = getattr(settings, 'ALGORITHM', 'HS256')
            else:
                # Import config for backward compatibility
                from ..config import BaseWebSettings
                config = BaseWebSettings()
                SECRET_KEY = config.SECRET_KEY
                ALGORITHM = getattr(config, 'ALGORITHM', 'HS256')
            
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            user_id = payload.get("sub")
            if not user_id:
                return None
                
            # Get user from database
            user = await self.get_user_by_id(int(user_id))
            return user
            
        except JWTError:
            return None
        except Exception as e:
            logger.error(f"Error getting user from token: {e}")
            return None

    async def _create_default_organization(self, user: Dict[str, Any]) -> None:
        """
        Create a default organization for a new user and make them the owner.
        
        Args:
            user: User record
        """
        try:
            user_id = user['id']
            user_name = user.get('full_name') or user.get('username') or user['email'].split('@')[0]
            
            # Create organization
            org_uuid = str(uuid.uuid4())
            org_name = f"{user_name}'s Organization"
            org_slug = await self._generate_org_slug(org_name)
            
            query = """
                INSERT INTO organizations (
                    uuid, name, display_name, slug, subscription_tier, 
                    is_active, created_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, 'free', 1, $5, $6, $7)
                RETURNING id
            """
            
            converted_query = self.query_adapter.convert_query(query)
            now = datetime.now().isoformat()
            
            org_id = await self.db.fetch_val(
                converted_query, 
                org_uuid, org_name, org_name, org_slug, user_id, now, now
            )
            
            if not org_id:
                logger.warning(f"Failed to create default organization for user {user_id}")
                return
                
            # Add user as owner of the organization
            member_query = """
                INSERT INTO organization_members (
                    organization_id, user_id, role, status, invited_by, joined_at
                ) VALUES ($1, $2, 'owner', 'active', $3, $4)
            """
            
            converted_member_query = self.query_adapter.convert_query(member_query)
            await self.db.execute(converted_member_query, org_id, user_id, user_id, now)
            
            # No need to update users table - membership is tracked in organization_members
            
            logger.info(f"Created default organization '{org_name}' (ID: {org_id}) for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to create default organization for user {user_id}: {e}", exc_info=True)
            # Don't raise - user creation should still succeed even if org creation fails
    
    async def _generate_org_slug(self, name: str) -> str:
        """Generate a unique slug for organization."""
        slug = name.lower().replace(' ', '-').replace("'", '').replace('_', '-')
        slug = ''.join(c for c in slug if c.isalnum() or c == '-')
        slug = slug[:50]  # Limit length
        
        # Check uniqueness
        counter = 1
        original_slug = slug
        while True:
            query = "SELECT id FROM organizations WHERE slug = $1"
            converted_query = self.query_adapter.convert_query(query)
            existing = await self.db.fetch_one(converted_query, slug)
            if not existing:
                break
            slug = f"{original_slug}-{counter}"
            counter += 1
        
        return slug

    def _is_valid_email(self, email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email to validate
            
        Returns:
            True if valid email format
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
