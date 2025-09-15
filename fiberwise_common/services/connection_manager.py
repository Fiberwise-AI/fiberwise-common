import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import WebSocket

from fiberwise_common.entities.user import User


logger = logging.getLogger(__name__)

# Connection manager for WebSockets - implemented as a singleton
class ConnectionManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ConnectionManager"""
        if cls._instance is None:
            cls._instance = ConnectionManager()
        return cls._instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            # Initialize instance attributes here, not in __init__
            cls._instance.active_connections = {}  # Key: (app_id, organization_id) -> [websockets]
            cls._instance.connection_ids = {}      # websocket -> user_id
            cls._instance.connection_apps = {}     # websocket -> app_id
            cls._instance.connection_orgs = {}     # websocket -> organization_id
            cls._instance.connection_users = {}    # websocket -> user object
            cls._instance.initialized = True
            logger.info("Created new ConnectionManager singleton instance")
        else:
            logger.debug("Returning existing ConnectionManager singleton")
        return cls._instance

    def __init__(self):
        # No initialization here - it's all done in __new__
        # This prevents re-initialization if the constructor is called again
        pass

    async def connect(self, websocket: WebSocket, app_id: str, organization_id: int, user: User):
        """Connect a client and store the connection"""
        try:
            logger.info(f"WebSocket connection ACCEPTED for org {organization_id} app {app_id} user {user.id}")
            
            connection_key = (app_id, organization_id)

            # Initialize list for app if it doesn't exist
            if connection_key not in self.active_connections:
                self.active_connections[connection_key] = []
            
            # Store the connection
            self.active_connections[connection_key].append(websocket)
            self.connection_ids[websocket] = user.id
            self.connection_apps[websocket] = app_id
            self.connection_orgs[websocket] = organization_id
            
            # Store user info
            self.connection_users[websocket] = user
            
            # Send welcome message
            welcome_message = {
                "type": "connection_established",
                "app_id": app_id,
                "organization_id": organization_id,
                "user_id": user.id,
                "username": user.username if hasattr(user, 'username') else user.email,
                "message": "Connected to FiberWise real-time service",
                "timestamp": datetime.now().isoformat()
            }
            
            await self.send_personal_message(welcome_message, websocket)
            
            logger.info(f"Client {user.id} connected to app {app_id}")
            return user.id
        
        except Exception as e:
            logger.error(f"WebSocket accept FAILED for app {app_id}: {e}", exc_info=True)
            # Handle disconnect or re-raise
            raise

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a client and clean up"""
        # Get app_id and organization_id for this connection
        app_id = self.connection_apps.get(websocket)
        organization_id = self.connection_orgs.get(websocket)
        client_id = self.connection_ids.get(websocket)
        
        if app_id and organization_id:
            connection_key = (app_id, organization_id)
            if connection_key in self.active_connections:
                if websocket in self.active_connections[connection_key]:
                    self.active_connections[connection_key].remove(websocket)
                
                # Clean up empty app lists
                if not self.active_connections[connection_key]:
                    del self.active_connections[connection_key]
        
        # Clean up client ID and app mappings
        if websocket in self.connection_ids:
            del self.connection_ids[websocket]
        
        if websocket in self.connection_apps:
            del self.connection_apps[websocket]
            
        if websocket in self.connection_orgs:
            del self.connection_orgs[websocket]

        # Clean up user mapping if exists
        if websocket in self.connection_users:
            del self.connection_users[websocket]
            
        logger.info(f"Client {client_id} disconnected from org {organization_id} app {app_id}")

    async def send_personal_message(self, message: Any, websocket: WebSocket):
        """Send a message to a specific client"""
        if isinstance(message, dict):
            await websocket.send_json(message)
        elif isinstance(message, str):
            await websocket.send_text(message)
        else:
            await websocket.send_json({"data": str(message)})

    async def broadcast_to_app(self, message: Any, app_id: str, exclude: Optional[WebSocket] = None):
        """Broadcast a message to all clients connected to an app across all organizations."""
        app_id_str = str(app_id)
        logger.debug(f"Broadcasting message to app {app_id_str} across all orgs, active connections: {list(self.active_connections.keys())}")

        all_connections_for_app = []
        for (conn_app_id, _), connections in self.active_connections.items():
            if conn_app_id == app_id_str:
                all_connections_for_app.extend(connections)

        if not all_connections_for_app:
            return

        closed_connections = []
        success_count = 0
        for connection in all_connections_for_app:
            if connection != exclude:
                try:
                    await self.send_personal_message(message, connection)
                    success_count += 1
                except RuntimeError as e:
                    if "after sending" in str(e) or "response already completed" in str(e):
                        logger.debug(f"Connection already closed, will remove: {str(e)[:50]}...")
                        closed_connections.append(connection)
                    else:
                        logger.error(f"RuntimeError sending message to connection: {e}")
                except Exception as e:
                    logger.error(f"Failed to send message to connection: {e}")
                    if "closed" in str(e).lower() or "completed" in str(e).lower():
                        closed_connections.append(connection)

        # Clean up closed connections
        for conn in closed_connections:
            await self.disconnect(conn)

        logger.debug(f"Successfully sent message to {success_count} clients for app {app_id_str}")
    
    async def broadcast_to_org_app(self, message: Any, app_id: str, organization_id: int, exclude: Optional[WebSocket] = None):
        """
        Broadcast a message to all users in an organization connected to an app.
        """
        app_id_str = str(app_id)
        connection_key = (app_id_str, organization_id)
        
        logger.debug(f"Broadcasting to org {organization_id} app {app_id_str}")
        
        if connection_key not in self.active_connections:
            return

        # Get a copy of the list to iterate over, to avoid issues if the list is modified during iteration
        org_connections = list(self.active_connections.get(connection_key, []))
        closed_connections = []
        
        success_count = 0
        for connection in org_connections:
            if connection == exclude:
                continue
            try:
                await self.send_personal_message(message, connection)
                success_count += 1
            except RuntimeError as e:
                if "after sending" in str(e) or "response already completed" in str(e):
                    logger.debug(f"Connection already closed, will remove: {str(e)[:50]}...")
                    closed_connections.append(connection)
                else:
                    logger.error(f"RuntimeError sending message to connection: {e}")
            except Exception as e:
                logger.error(f"Failed to send message to connection: {e}")
                if "closed" in str(e).lower() or "completed" in str(e).lower():
                    closed_connections.append(connection)
        
        # Clean up closed connections
        for conn in closed_connections:
            await self.disconnect(conn)
        
        logger.debug(f"Successfully sent org message to {success_count}/{len(org_connections)} organization members")
    
    async def broadcast_to_user_app(self, message: Any, app_id: str, user_id: str, exclude: Optional[WebSocket] = None):
        """
        Broadcast a message to a specific user in an app (for backward compatibility with user-scoped systems).
        This method finds the user's connections across all organizations.
        """
        app_id_str = str(app_id)
        user_id_str = str(user_id)
        
        logger.debug(f"Broadcasting user-scoped message to user {user_id_str} in app {app_id_str}")
        
        # Find all connections for this user in this app across all organizations
        user_connections = []
        for (conn_app_id, org_id), connections in self.active_connections.items():
            if conn_app_id == app_id_str:
                # Check each connection to see if it belongs to the target user
                for conn in connections:
                    if self.connection_ids.get(conn) == user_id_str:
                        user_connections.append(conn)
        
        if not user_connections:
            logger.debug(f"No connections found for user {user_id_str} in app {app_id_str}")
            return 0
        
        closed_connections = []
        success_count = 0
        
        for connection in user_connections:
            if connection != exclude:
                try:
                    await self.send_personal_message(message, connection)
                    success_count += 1
                except RuntimeError as e:
                    if "after sending" in str(e) or "response already completed" in str(e):
                        logger.debug(f"Connection already closed, will remove: {str(e)[:50]}...")
                        closed_connections.append(connection)
                    else:
                        logger.error(f"RuntimeError sending message to connection: {e}")
                except Exception as e:
                    logger.error(f"Failed to send message to connection: {e}")
                    if "closed" in str(e).lower() or "completed" in str(e).lower():
                        closed_connections.append(connection)
        
        # Clean up closed connections
        for conn in closed_connections:
            await self.disconnect(conn)
        
        logger.debug(f"Successfully sent user-scoped message to {success_count} connections for user {user_id_str} in app {app_id_str}")
        return success_count
    
    def _user_belongs_to_org(self, user, organization_id: int) -> bool:
        """Check if user belongs to the specified organization."""
        # This is a simple check - in production you might want to cache this
        user_org_id = getattr(user, 'organization_id', None)
        return user_org_id == organization_id
    
    async def get_api_key_organization(self, api_key: str) -> Optional[int]:
        """Get the organization_id for an API key (must be stored on the key)."""
        try:
            # Get the organization_id directly from the API key
            key_query = "SELECT organization_id FROM agent_api_keys WHERE key_value = $1"
            key_result = await self.db.fetch_one(key_query, api_key)
            
            if not key_result:
                logger.warning(f"API key not found: {api_key[:10]}...")
                return None
            
            if not key_result['organization_id']:
                logger.error(f"API key {api_key[:10]}... has no organization_id - run migration script!")
                return None
            
            return key_result['organization_id']
            
        except Exception as e:
            logger.error(f"Error getting organization for API key: {e}")
            return None
    
    async def send_to_client(self, message: Any, app_id: str, client_id: str):
        """Send a message to a specific client by client_id"""
        if app_id not in self.active_connections:
            return False
            
        for connection in self.active_connections[app_id]:
            if self.connection_ids.get(connection) == client_id:
                await self.send_personal_message(message, connection)
                return True
                
        return False
        
    def get_connected_clients(self, app_id: str, organization_id: Optional[int] = None) -> List[str]:
        """
        Get list of client IDs connected to an app.
        If organization_id is provided, it filters by organization.
        Otherwise, it returns clients from all organizations for that app.
        """
        clients = []
        if organization_id:
            connection_key = (app_id, organization_id)
            if connection_key in self.active_connections:
                for conn in self.active_connections[connection_key]:
                    if conn in self.connection_ids:
                        clients.append(self.connection_ids[conn])
        else:
            for (conn_app_id, _), connections in self.active_connections.items():
                if conn_app_id == app_id:
                    for conn in connections:
                        if conn in self.connection_ids:
                            clients.append(self.connection_ids[conn])
        return clients
