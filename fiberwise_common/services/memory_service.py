"""
Memory Service — conversation history and semantic search.

Wraps ia_modules memory backends for the FiberWise platform.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)


class MemoryService(BaseService):
    """Service for conversation memory management."""

    def __init__(self, database_provider: DatabaseProvider):
        super().__init__(database_provider)
        self.db = database_provider

    async def save_conversation_turn(self, session_id: str, role: str, content: str,
                                      created_by: int, thread_id: Optional[str] = None,
                                      app_id: Optional[str] = None, metadata: Optional[dict] = None):
        """Save a conversation message."""
        msg_id = str(uuid.uuid4())
        await self.db.execute(
            """INSERT INTO conversation_messages
               (message_id, session_id, thread_id, role, content, metadata, app_id, created_by)
               VALUES (:id, :session_id, :thread_id, :role, :content, :metadata, :app_id, :created_by)""",
            {"id": msg_id, "session_id": session_id, "thread_id": thread_id or session_id,
             "role": role, "content": content, "metadata": json.dumps(metadata or {}),
             "app_id": app_id, "created_by": created_by}
        )
        return msg_id

    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        rows = await self.db.fetch_all(
            "SELECT * FROM conversation_messages WHERE session_id = :sid ORDER BY created_at ASC LIMIT :limit",
            {"sid": session_id, "limit": limit}
        )
        return [dict(r) for r in rows] if rows else []

    async def search_memory(self, query: str, session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Keyword search over conversation messages. For semantic search, ia_modules vector backend is used."""
        conditions = ["content LIKE :query"]
        params = {"query": f"%{query}%", "limit": limit}
        if session_id:
            conditions.append("session_id = :sid")
            params["sid"] = session_id

        rows = await self.db.fetch_all(
            f"SELECT * FROM conversation_messages WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT :limit",
            params
        )
        return [dict(r) for r in rows] if rows else []

    async def get_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about memory usage for a session."""
        count = await self.db.fetch_val(
            "SELECT COUNT(*) FROM conversation_messages WHERE session_id = :sid", {"sid": session_id}
        )
        first = await self.db.fetch_one(
            "SELECT created_at FROM conversation_messages WHERE session_id = :sid ORDER BY created_at ASC LIMIT 1",
            {"sid": session_id}
        )
        last = await self.db.fetch_one(
            "SELECT created_at FROM conversation_messages WHERE session_id = :sid ORDER BY created_at DESC LIMIT 1",
            {"sid": session_id}
        )
        return {
            "session_id": session_id,
            "message_count": count or 0,
            "first_message": dict(first)["created_at"] if first else None,
            "last_message": dict(last)["created_at"] if last else None
        }
