"""
WebSocket connection manager for real-time verification progress updates.

Handles WebSocket connections, authentication, broadcasting, and connection lifecycle.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from src.api.auth import get_db, User, UserRole
from src.api.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()


class WebSocketConnection:
    """Represents a single WebSocket connection with metadata."""
    
    def __init__(self, websocket: WebSocket, user: User, connection_id: str):
        self.websocket = websocket
        self.user = user
        self.connection_id = connection_id
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.subscriptions: Set[str] = set()  # Set of task IDs user is subscribed to
        self.is_active = True
    
    async def send_message(self, message: dict) -> bool:
        """Send a message to the WebSocket client."""
        try:
            await self.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            self.is_active = False
            return False
    
    async def send_ping(self) -> bool:
        """Send a ping message to keep connection alive."""
        return await self.send_message({
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def update_ping(self):
        """Update last ping timestamp."""
        self.last_ping = datetime.utcnow()
    
    def is_stale(self, timeout_minutes: int = 5) -> bool:
        """Check if connection is stale based on last ping."""
        return datetime.utcnow() - self.last_ping > timedelta(minutes=timeout_minutes)


class WebSocketManager:
    """Manages all WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[int, List[str]] = {}  # user_id -> connection_ids
        self.task_subscribers: Dict[str, Set[str]] = {}  # task_id -> connection_ids
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self):
        """Start the WebSocket manager background tasks."""
        if not self.is_running:
            self.is_running = True
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop the WebSocket manager and cleanup."""
        self.is_running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self.disconnect(connection.connection_id)
        
        logger.info("WebSocket manager stopped")
    
    async def authenticate_websocket(self, websocket: WebSocket, token: str) -> Optional[User]:
        """Authenticate WebSocket connection using JWT token."""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            token_type = payload.get("type")
            
            if not user_id or token_type != "access":
                return None
            
            # Get user from database
            db = next(get_db())
            user = db.query(User).filter(User.id == int(user_id)).first()
            
            if not user or user.status.value != "ACTIVE":
                return None
            
            return user
            
        except (JWTError, ValueError, AttributeError) as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            return None
    
    async def connect(self, websocket: WebSocket, user: User) -> str:
        """Add a new WebSocket connection."""
        connection_id = str(uuid4())
        
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(websocket, user, connection_id)
            self.connections[connection_id] = connection
            
            # Track user connections
            if user.id not in self.user_connections:
                self.user_connections[user.id] = []
            self.user_connections[user.id].append(connection_id)
            
            # Send connection confirmation
            await connection.send_message({
                "type": "connection_established",
                "connection_id": connection_id,
                "user_id": user.id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"WebSocket connected: {connection_id} for user {user.id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            if connection_id in self.connections:
                del self.connections[connection_id]
            raise
    
    async def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_id = connection.user.id
        
        # Remove from subscriptions
        for task_id in list(connection.subscriptions):
            await self.unsubscribe_from_task(connection_id, task_id)
        
        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id] = [
                cid for cid in self.user_connections[user_id] 
                if cid != connection_id
            ]
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception:
            pass  # Connection might already be closed
        
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def subscribe_to_task(self, connection_id: str, task_id: str) -> bool:
        """Subscribe a connection to task updates."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.add(task_id)
        
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = set()
        self.task_subscribers[task_id].add(connection_id)
        
        # Send subscription confirmation
        await connection.send_message({
            "type": "subscribed",
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.debug(f"Connection {connection_id} subscribed to task {task_id}")
        return True
    
    async def unsubscribe_from_task(self, connection_id: str, task_id: str) -> bool:
        """Unsubscribe a connection from task updates."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.discard(task_id)
        
        if task_id in self.task_subscribers:
            self.task_subscribers[task_id].discard(connection_id)
            if not self.task_subscribers[task_id]:
                del self.task_subscribers[task_id]
        
        # Send unsubscription confirmation
        await connection.send_message({
            "type": "unsubscribed",
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.debug(f"Connection {connection_id} unsubscribed from task {task_id}")
        return True
    
    async def broadcast_task_update(self, task_id: str, update_data: dict):
        """Broadcast an update to all subscribers of a task."""
        if task_id not in self.task_subscribers:
            return
        
        message = {
            "type": "task_update",
            "task_id": task_id,
            "data": update_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all subscribers
        failed_connections = []
        for connection_id in self.task_subscribers[task_id].copy():
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        logger.debug(f"Broadcasted update for task {task_id} to {len(self.task_subscribers[task_id])} subscribers")
    
    async def send_to_user(self, user_id: int, message: dict):
        """Send a message to all connections for a specific user."""
        if user_id not in self.user_connections:
            return
        
        failed_connections = []
        for connection_id in self.user_connections[user_id].copy():
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming message from WebSocket client."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                connection.update_ping()
                await connection.send_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "subscribe":
                task_id = data.get("task_id")
                if task_id:
                    await self.subscribe_to_task(connection_id, task_id)
            
            elif message_type == "unsubscribe":
                task_id = data.get("task_id")
                if task_id:
                    await self.unsubscribe_from_task(connection_id, task_id)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from {connection_id}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def _heartbeat_loop(self):
        """Background task to send periodic pings."""
        while self.is_running:
            try:
                failed_connections = []
                
                for connection_id, connection in self.connections.items():
                    if connection.is_stale():
                        failed_connections.append(connection_id)
                    else:
                        success = await connection.send_ping()
                        if not success:
                            failed_connections.append(connection_id)
                
                # Clean up failed connections
                for connection_id in failed_connections:
                    await self.disconnect(connection_id)
                
                await asyncio.sleep(30)  # Ping every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        while self.is_running:
            try:
                stale_connections = [
                    connection_id for connection_id, connection in self.connections.items()
                    if connection.is_stale(timeout_minutes=10)
                ]
                
                for connection_id in stale_connections:
                    await self.disconnect(connection_id)
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.connections),
            "unique_users": len(self.user_connections),
            "active_subscriptions": sum(len(subs) for subs in self.task_subscribers.values()),
            "tasks_with_subscribers": len(self.task_subscribers),
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager() 