"""
Integration tests for database providers and services working together.

This module tests the complete integration between database providers,
base services, and the service registry to ensure they work together correctly.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from fiberwise_common.database.providers import SQLiteProvider
from fiberwise_common.services.base_service import BaseService, ServiceRegistry, ServiceError


class UserService(BaseService):
    """Example user service for integration testing."""
    
    async def create_user(self, username: str, email: str) -> int:
        """Create a new user and return the user ID."""
        await self._execute(
            "INSERT INTO users (username, email, created_at) VALUES (?, ?, datetime('now'))",
            (username, email)
        )
        
        # Get the last inserted row ID
        result = await self._fetch_one("SELECT last_insert_rowid()")
        return result[0] if result else 0
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        result = await self._fetch_one(
            "SELECT id, username, email, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        
        if result:
            return {
                "id": result[0],
                "username": result[1], 
                "email": result[2],
                "created_at": result[3]
            }
        return None
    
    async def list_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all users with optional limit."""
        results = await self._fetch_all(
            "SELECT id, username, email, created_at FROM users ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        return [
            {
                "id": row[0],
                "username": row[1],
                "email": row[2], 
                "created_at": row[3]
            }
            for row in results
        ]
    
    async def update_user(self, user_id: int, username: str = None, email: str = None) -> bool:
        """Update user information."""
        updates = []
        params = []
        
        if username:
            updates.append("username = ?")
            params.append(username)
        
        if email:
            updates.append("email = ?")
            params.append(email)
        
        if not updates:
            return False
        
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        
        await self._execute(query, tuple(params))
        return True
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID."""
        await self._execute("DELETE FROM users WHERE id = ?", (user_id,))
        return True
    
    async def initialize_schema(self):
        """Initialize the database schema."""
        await self._execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


class PostService(BaseService):
    """Example post service for integration testing."""
    
    async def create_post(self, user_id: int, title: str, content: str) -> int:
        """Create a new post."""
        await self._execute(
            "INSERT INTO posts (user_id, title, content, created_at) VALUES (?, ?, ?, datetime('now'))",
            (user_id, title, content)
        )
        
        result = await self._fetch_one("SELECT last_insert_rowid()")
        return result[0] if result else 0
    
    async def get_post(self, post_id: int) -> Optional[Dict[str, Any]]:
        """Get post by ID with user information."""
        result = await self._fetch_one("""
            SELECT p.id, p.title, p.content, p.created_at,
                   u.username, u.email
            FROM posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
        """, (post_id,))
        
        if result:
            return {
                "id": result[0],
                "title": result[1],
                "content": result[2],
                "created_at": result[3],
                "author": {
                    "username": result[4],
                    "email": result[5]
                }
            }
        return None
    
    async def list_posts_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """List all posts by a specific user."""
        results = await self._fetch_all("""
            SELECT id, title, content, created_at
            FROM posts 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        """, (user_id,))
        
        return [
            {
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "created_at": row[3]
            }
            for row in results
        ]
    
    async def initialize_schema(self):
        """Initialize the database schema."""
        await self._execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)


@pytest.mark.integration
class TestDatabaseServiceIntegration:
    """Integration tests for database and service components."""
    
    @pytest.fixture
    async def database_provider(self, temp_dir):
        """Create and initialize database provider."""
        db_path = temp_dir / "integration_test.db"
        provider = SQLiteProvider(str(db_path))
        await provider.connect()
        return provider
    
    @pytest.fixture
    async def user_service(self, database_provider):
        """Create user service with real database."""
        service = UserService(database_provider)
        await service.initialize_schema()
        return service
    
    @pytest.fixture
    async def post_service(self, database_provider):
        """Create post service with real database."""
        service = PostService(database_provider)
        await service.initialize_schema()
        return service
    
    @pytest.fixture
    def service_registry(self):
        """Create service registry for testing."""
        registry = ServiceRegistry()
        yield registry
        registry.clear()
    
    async def test_basic_user_crud_operations(self, user_service):
        """Test basic CRUD operations for users."""
        # Create user
        user_id = await user_service.create_user("testuser", "test@example.com")
        assert user_id > 0
        
        # Read user
        user = await user_service.get_user(user_id)
        assert user is not None
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["id"] == user_id
        
        # Update user
        success = await user_service.update_user(user_id, username="updateduser")
        assert success is True
        
        updated_user = await user_service.get_user(user_id)
        assert updated_user["username"] == "updateduser"
        assert updated_user["email"] == "test@example.com"  # Unchanged
        
        # Delete user
        success = await user_service.delete_user(user_id)
        assert success is True
        
        deleted_user = await user_service.get_user(user_id)
        assert deleted_user is None
    
    async def test_user_service_with_multiple_users(self, user_service):
        """Test user service with multiple users."""
        # Create multiple users
        users_data = [
            ("alice", "alice@example.com"),
            ("bob", "bob@example.com"),
            ("charlie", "charlie@example.com")
        ]
        
        user_ids = []
        for username, email in users_data:
            user_id = await user_service.create_user(username, email)
            user_ids.append(user_id)
        
        # List users
        users = await user_service.list_users()
        assert len(users) == 3
        
        # Verify users are ordered by creation time (newest first)
        usernames = [user["username"] for user in users]
        assert "charlie" in usernames
        assert "alice" in usernames
        assert "bob" in usernames
        
        # Test limit
        limited_users = await user_service.list_users(limit=2)
        assert len(limited_users) == 2
    
    async def test_user_and_post_services_integration(self, user_service, post_service):
        """Test integration between user and post services."""
        # Create a user
        user_id = await user_service.create_user("blogger", "blogger@example.com")
        
        # Create posts for the user
        post1_id = await post_service.create_post(user_id, "First Post", "This is my first post!")
        post2_id = await post_service.create_post(user_id, "Second Post", "Another great post!")
        
        # Get posts by user
        user_posts = await post_service.list_posts_by_user(user_id)
        assert len(user_posts) == 2
        
        # Posts should be ordered by creation time (newest first)
        assert user_posts[0]["title"] == "Second Post"
        assert user_posts[1]["title"] == "First Post"
        
        # Get post with user information
        post = await post_service.get_post(post1_id)
        assert post is not None
        assert post["title"] == "First Post"
        assert post["author"]["username"] == "blogger"
        assert post["author"]["email"] == "blogger@example.com"
    
    async def test_services_with_registry(self, database_provider, service_registry):
        """Test services working with service registry."""
        # Create services
        user_service = UserService(database_provider)
        post_service = PostService(database_provider)
        
        # Initialize schemas
        await user_service.initialize_schema()
        await post_service.initialize_schema()
        
        # Register services
        service_registry.register("user_service", user_service)
        service_registry.register("post_service", post_service)
        
        # Use services through registry
        user_svc = service_registry.get("user_service")
        post_svc = service_registry.get("post_service")
        
        # Create user through registry
        user_id = await user_svc.create_user("registry_user", "registry@example.com")
        
        # Create post through registry
        post_id = await post_svc.create_post(user_id, "Registry Post", "Posted via registry!")
        
        # Verify integration works
        post = await post_svc.get_post(post_id)
        assert post["author"]["username"] == "registry_user"
    
    async def test_database_transactions_with_services(self, database_provider, user_service):
        """Test database transactions with services."""
        await user_service.initialize_schema()
        
        # Test successful transaction
        async with database_provider.transaction():
            user1_id = await user_service.create_user("user1", "user1@example.com")
            user2_id = await user_service.create_user("user2", "user2@example.com")
        
        # Both users should exist
        user1 = await user_service.get_user(user1_id)
        user2 = await user_service.get_user(user2_id)
        assert user1 is not None
        assert user2 is not None
        
        # Test failed transaction (rollback)
        try:
            async with database_provider.transaction():
                user3_id = await user_service.create_user("user3", "user3@example.com")
                # This should fail due to duplicate email
                await user_service.create_user("user4", "user1@example.com")  # Duplicate email
        except Exception:
            pass  # Expected to fail
        
        # User3 should not exist due to rollback
        user3 = await user_service.get_user(3)  # Assuming incremental IDs
        assert user3 is None
    
    async def test_concurrent_service_operations(self, user_service):
        """Test concurrent operations through services."""
        await user_service.initialize_schema()
        
        async def create_user_batch(batch_id: int, count: int):
            user_ids = []
            for i in range(count):
                username = f"batch{batch_id}_user{i}"
                email = f"batch{batch_id}_user{i}@example.com"
                user_id = await user_service.create_user(username, email)
                user_ids.append(user_id)
            return user_ids
        
        # Create multiple batches concurrently
        tasks = [
            create_user_batch(1, 3),
            create_user_batch(2, 3),
            create_user_batch(3, 3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all users were created
        all_user_ids = [user_id for batch in results for user_id in batch]
        assert len(all_user_ids) == 9
        
        # Verify all users exist in database
        users = await user_service.list_users()
        assert len(users) == 9
    
    async def test_service_error_handling(self, user_service):
        """Test error handling in service operations."""
        await user_service.initialize_schema()
        
        # Create a user
        user_id = await user_service.create_user("testuser", "test@example.com")
        
        # Try to create duplicate user (should fail)
        with pytest.raises(Exception):  # SQLite constraint error
            await user_service.create_user("testuser", "different@example.com")
        
        # Try to create user with duplicate email
        with pytest.raises(Exception):  # SQLite constraint error
            await user_service.create_user("differentuser", "test@example.com")
        
        # Original user should still exist
        user = await user_service.get_user(user_id)
        assert user is not None
        assert user["username"] == "testuser"
    
    async def test_database_connection_lifecycle(self, temp_dir):
        """Test complete database connection lifecycle."""
        db_path = temp_dir / "lifecycle_test.db"
        
        # Create provider and connect
        provider = SQLiteProvider(str(db_path))
        await provider.connect()
        
        # Create and use service
        service = UserService(provider)
        await service.initialize_schema()
        
        user_id = await service.create_user("lifecycle_user", "lifecycle@example.com")
        
        # Disconnect
        await provider.disconnect()
        
        # Reconnect
        provider2 = SQLiteProvider(str(db_path))
        await provider2.connect()
        
        # Use new service with same database
        service2 = UserService(provider2)
        
        # Data should persist
        user = await service2.get_user(user_id)
        assert user is not None
        assert user["username"] == "lifecycle_user"
        
        await provider2.disconnect()
    
    async def test_service_with_complex_queries(self, user_service, post_service):
        """Test services with more complex database queries."""
        await user_service.initialize_schema()
        await post_service.initialize_schema()
        
        # Create users and posts
        users_data = [
            ("alice", "alice@example.com"),
            ("bob", "bob@example.com")
        ]
        
        user_ids = []
        for username, email in users_data:
            user_id = await user_service.create_user(username, email)
            user_ids.append(user_id)
        
        # Create posts for each user
        for user_id in user_ids:
            user = await user_service.get_user(user_id)
            username = user["username"]
            
            for i in range(3):
                title = f"{username}'s Post {i+1}"
                content = f"Content for {title}"
                await post_service.create_post(user_id, title, content)
        
        # Test complex queries through services
        alice_id = user_ids[0]
        alice_posts = await post_service.list_posts_by_user(alice_id)
        assert len(alice_posts) == 3
        
        # Get a post with author info
        post = await post_service.get_post(alice_posts[0]["id"])
        assert post["author"]["username"] == "alice"
    
    async def test_service_registry_error_handling(self, service_registry):
        """Test error handling in service registry."""
        # Try to get non-existent service
        with pytest.raises(ServiceError, match="Service not found"):
            service_registry.get("nonexistent_service")
        
        # Register and retrieve service
        mock_service = UserService(None)  # Passing None for testing
        service_registry.register("test_service", mock_service)
        
        retrieved = service_registry.get("test_service")
        assert retrieved is mock_service
        
        # Clear and verify
        service_registry.clear()
        
        with pytest.raises(ServiceError, match="Service not found"):
            service_registry.get("test_service")


@pytest.mark.integration
class TestDatabaseServiceWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.fixture
    async def complete_system(self, temp_dir):
        """Set up complete system with database, services, and registry."""
        # Database
        db_path = temp_dir / "workflow_test.db"
        provider = SQLiteProvider(str(db_path))
        await provider.connect()
        
        # Services
        user_service = UserService(provider)
        post_service = PostService(provider)
        
        await user_service.initialize_schema()
        await post_service.initialize_schema()
        
        # Registry
        registry = ServiceRegistry()
        registry.register("users", user_service)
        registry.register("posts", post_service)
        registry.register("database", provider)
        
        yield {
            "database": provider,
            "services": {"users": user_service, "posts": post_service},
            "registry": registry
        }
        
        # Cleanup
        await provider.disconnect()
        registry.clear()
    
    async def test_blog_application_workflow(self, complete_system):
        """Test complete blog application workflow."""
        registry = complete_system["registry"]
        user_service = registry.get("users")
        post_service = registry.get("posts")
        
        # User registration workflow
        alice_id = await user_service.create_user("alice", "alice@blog.com")
        bob_id = await user_service.create_user("bob", "bob@blog.com")
        
        # Content creation workflow
        posts_data = [
            (alice_id, "Alice's First Post", "Welcome to my blog!"),
            (alice_id, "Alice's Second Post", "My thoughts on technology"),
            (bob_id, "Bob's Introduction", "Hello everyone!"),
            (bob_id, "Bob's Recipe", "How to make great coffee"),
        ]
        
        post_ids = []
        for user_id, title, content in posts_data:
            post_id = await post_service.create_post(user_id, title, content)
            post_ids.append(post_id)
        
        # Content retrieval workflow
        all_users = await user_service.list_users()
        assert len(all_users) == 2
        
        alice_posts = await post_service.list_posts_by_user(alice_id)
        bob_posts = await post_service.list_posts_by_user(bob_id)
        
        assert len(alice_posts) == 2
        assert len(bob_posts) == 2
        
        # Featured content workflow (get specific post with author)
        featured_post = await post_service.get_post(post_ids[0])
        assert featured_post["title"] == "Alice's First Post"
        assert featured_post["author"]["username"] == "alice"
        
        # User profile update workflow
        await user_service.update_user(alice_id, email="alice.updated@blog.com")
        
        updated_alice = await user_service.get_user(alice_id)
        assert updated_alice["email"] == "alice.updated@blog.com"
        
        # Content moderation workflow (delete user and cascade posts)
        await user_service.delete_user(bob_id)
        
        # Bob should be gone
        deleted_bob = await user_service.get_user(bob_id)
        assert deleted_bob is None
        
        # Bob's posts should still be queryable individually
        # (Note: This depends on foreign key constraints and cascade behavior)
        remaining_users = await user_service.list_users()
        assert len(remaining_users) == 1
        assert remaining_users[0]["username"] == "alice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])