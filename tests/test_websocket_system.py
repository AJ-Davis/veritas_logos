"""
Test suite for WebSocket system.

Tests WebSocket connections, authentication, broadcasting, and event handling.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from jose import jwt

from src.api.websocket_manager import (
    WebSocketManager, WebSocketConnection, websocket_manager
)
from src.api.verification_events import (
    VerificationEventHandler, VerificationEventType,
    emit_task_created, emit_task_progress, emit_task_completed
)
from src.api.config import settings


class TestWebSocketConnection:
    """Test WebSocketConnection class."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        websocket = MagicMock(spec=WebSocket)
        websocket.send_text = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = MagicMock()
        user.id = 1
        user.username = "testuser"
        user.status.value = "ACTIVE"
        return user
    
    def test_connection_initialization(self, mock_websocket, mock_user):
        """Test WebSocketConnection initialization."""
        connection_id = str(uuid4())
        connection = WebSocketConnection(mock_websocket, mock_user, connection_id)
        
        assert connection.websocket == mock_websocket
        assert connection.user == mock_user
        assert connection.connection_id == connection_id
        assert connection.is_active is True
        assert len(connection.subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_websocket, mock_user):
        """Test successful message sending."""
        connection = WebSocketConnection(mock_websocket, mock_user, "test-id")
        message = {"type": "test", "data": "hello"}
        
        result = await connection.send_message(message)
        
        assert result is True
        mock_websocket.send_text.assert_called_once_with(json.dumps(message))
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, mock_websocket, mock_user):
        """Test message sending failure."""
        connection = WebSocketConnection(mock_websocket, mock_user, "test-id")
        mock_websocket.send_text.side_effect = Exception("Connection closed")
        
        result = await connection.send_message({"type": "test"})
        
        assert result is False
        assert connection.is_active is False
    
    @pytest.mark.asyncio
    async def test_send_ping(self, mock_websocket, mock_user):
        """Test ping message sending."""
        connection = WebSocketConnection(mock_websocket, mock_user, "test-id")
        
        result = await connection.send_ping()
        
        assert result is True
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == "ping"
        assert "timestamp" in message
    
    def test_is_stale(self, mock_websocket, mock_user):
        """Test stale connection detection."""
        connection = WebSocketConnection(mock_websocket, mock_user, "test-id")
        
        # Fresh connection should not be stale
        assert connection.is_stale() is False
        
        # Manually set old timestamp
        connection.last_ping = datetime.utcnow() - timedelta(minutes=10)
        assert connection.is_stale(timeout_minutes=5) is True


class TestWebSocketManager:
    """Test WebSocketManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh WebSocket manager for testing."""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        websocket = MagicMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = MagicMock()
        user.id = 1
        user.username = "testuser"
        user.status.value = "ACTIVE"
        return user
    
    @pytest.mark.asyncio
    async def test_manager_start_stop(self, manager):
        """Test WebSocket manager startup and shutdown."""
        assert manager.is_running is False
        
        await manager.start()
        assert manager.is_running is True
        assert manager.heartbeat_task is not None
        assert manager.cleanup_task is not None
        
        await manager.stop()
        assert manager.is_running is False
    
    @patch('src.api.websocket_manager.get_db')
    @pytest.mark.asyncio
    async def test_authenticate_websocket_success(self, mock_get_db, manager, mock_user):
        """Test successful WebSocket authentication."""
        # Mock database session
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        # Create valid JWT token
        token_data = {"sub": str(mock_user.id), "type": "access"}
        token = jwt.encode(token_data, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
        
        user = await manager.authenticate_websocket(MagicMock(), token)
        
        assert user == mock_user
    
    @pytest.mark.asyncio
    async def test_authenticate_websocket_invalid_token(self, manager):
        """Test WebSocket authentication with invalid token."""
        user = await manager.authenticate_websocket(MagicMock(), "invalid-token")
        assert user is None
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, manager, mock_websocket, mock_user):
        """Test WebSocket connection establishment."""
        connection_id = await manager.connect(mock_websocket, mock_user)
        
        assert connection_id in manager.connections
        assert mock_user.id in manager.user_connections
        assert connection_id in manager.user_connections[mock_user.id]
        
        # Verify welcome message was sent
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, manager, mock_websocket, mock_user):
        """Test WebSocket disconnection."""
        connection_id = await manager.connect(mock_websocket, mock_user)
        
        # Add subscription to test cleanup
        await manager.subscribe_to_task(connection_id, "test-task")
        
        await manager.disconnect(connection_id)
        
        assert connection_id not in manager.connections
        assert mock_user.id not in manager.user_connections or not manager.user_connections[mock_user.id]
        assert "test-task" not in manager.task_subscribers or connection_id not in manager.task_subscribers["test-task"]
    
    @pytest.mark.asyncio
    async def test_subscribe_to_task(self, manager, mock_websocket, mock_user):
        """Test task subscription."""
        connection_id = await manager.connect(mock_websocket, mock_user)
        task_id = "test-task-123"
        
        result = await manager.subscribe_to_task(connection_id, task_id)
        
        assert result is True
        assert task_id in manager.connections[connection_id].subscriptions
        assert task_id in manager.task_subscribers
        assert connection_id in manager.task_subscribers[task_id]
    
    @pytest.mark.asyncio
    async def test_broadcast_task_update(self, manager, mock_websocket, mock_user):
        """Test broadcasting task updates."""
        # Connect and subscribe
        connection_id = await manager.connect(mock_websocket, mock_user)
        task_id = "test-task-123"
        await manager.subscribe_to_task(connection_id, task_id)
        
        # Reset mock to ignore setup calls
        mock_websocket.send_text.reset_mock()
        
        # Broadcast update
        update_data = {"status": "running", "progress": 50}
        await manager.broadcast_task_update(task_id, update_data)
        
        # Verify message was sent
        assert mock_websocket.send_text.call_count >= 1
        last_call = mock_websocket.send_text.call_args[0][0]
        message = json.loads(last_call)
        assert message["type"] == "task_update"
        assert message["task_id"] == task_id
        assert message["data"] == update_data
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, manager, mock_websocket, mock_user):
        """Test handling ping messages."""
        connection_id = await manager.connect(mock_websocket, mock_user)
        
        # Reset mock to ignore setup calls
        mock_websocket.send_text.reset_mock()
        
        # Handle ping message
        ping_message = json.dumps({"type": "ping"})
        await manager.handle_message(connection_id, ping_message)
        
        # Verify pong response
        mock_websocket.send_text.assert_called_once()
        response = json.loads(mock_websocket.send_text.call_args[0][0])
        assert response["type"] == "pong"
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, manager, mock_websocket, mock_user):
        """Test handling subscribe messages."""
        connection_id = await manager.connect(mock_websocket, mock_user)
        task_id = "test-task-456"
        
        # Handle subscribe message
        subscribe_message = json.dumps({"type": "subscribe", "task_id": task_id})
        await manager.handle_message(connection_id, subscribe_message)
        
        # Verify subscription
        assert task_id in manager.connections[connection_id].subscriptions
        assert task_id in manager.task_subscribers
    
    def test_get_connection_stats(self, manager):
        """Test connection statistics."""
        stats = manager.get_connection_stats()
        
        expected_keys = [
            "total_connections", "unique_users", "active_subscriptions",
            "tasks_with_subscribers", "connections_by_user"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["total_connections"] == 0
        assert stats["unique_users"] == 0


class TestVerificationEventHandler:
    """Test VerificationEventHandler class."""
    
    @pytest.fixture
    def event_handler(self):
        """Create a fresh event handler for testing."""
        return VerificationEventHandler()
    
    @pytest.mark.asyncio
    async def test_emit_task_created_event(self, event_handler):
        """Test emitting task created event."""
        task_id = "test-task-789"
        user_id = 1
        
        with patch.object(websocket_manager, 'broadcast_task_update') as mock_broadcast:
            await event_handler.emit_event(
                task_id=task_id,
                event_type=VerificationEventType.TASK_CREATED,
                data={"document_name": "test.pdf"},
                user_id=user_id
            )
        
        # Verify event was stored
        assert task_id in event_handler.event_history
        assert len(event_handler.event_history[task_id]) == 1
        
        # Verify task state was created
        assert task_id in event_handler.active_tasks
        task_state = event_handler.active_tasks[task_id]
        assert task_state["status"] == "created"
        assert task_state["progress"] == 0
        assert task_state["user_id"] == user_id
        
        # Verify broadcast was called
        mock_broadcast.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_emit_task_progress_event(self, event_handler):
        """Test emitting task progress event."""
        task_id = "test-task-progress"
        
        # Create initial task
        await event_handler.emit_event(
            task_id=task_id,
            event_type=VerificationEventType.TASK_CREATED,
            data={},
            user_id=1
        )
        
        with patch.object(websocket_manager, 'broadcast_task_update'):
            await event_handler.emit_event(
                task_id=task_id,
                event_type=VerificationEventType.TASK_PROGRESS,
                data={"progress": 25, "step": "Analyzing document"}
            )
        
        # Verify progress was updated
        task_state = event_handler.active_tasks[task_id]
        assert task_state["progress"] == 25
        assert task_state["current_step"] == "Analyzing document"
    
    @pytest.mark.asyncio
    async def test_emit_task_completed_event(self, event_handler):
        """Test emitting task completed event."""
        task_id = "test-task-completed"
        user_id = 1
        
        # Create initial task
        await event_handler.emit_event(
            task_id=task_id,
            event_type=VerificationEventType.TASK_CREATED,
            data={},
            user_id=user_id
        )
        
        with patch.object(websocket_manager, 'broadcast_task_update'), \
             patch.object(websocket_manager, 'send_to_user') as mock_send_user:
            
            await event_handler.emit_event(
                task_id=task_id,
                event_type=VerificationEventType.TASK_COMPLETED,
                data={"results": {"total_issues": 3}},
                user_id=user_id
            )
        
        # Verify task completion
        task_state = event_handler.active_tasks[task_id]
        assert task_state["status"] == "completed"
        assert task_state["progress"] == 100
        assert "completed_at" in task_state
        
        # Verify user notification was sent
        mock_send_user.assert_called_once_with(user_id, unittest.mock.ANY)
    
    def test_get_task_history(self, event_handler):
        """Test retrieving task history."""
        task_id = "test-task-history"
        
        # No history initially
        history = event_handler.get_task_history(task_id)
        assert history == []
        
        # Add some events
        event_handler.event_history[task_id] = [
            {"event_type": "task_created", "timestamp": "2024-01-01T00:00:00"},
            {"event_type": "task_completed", "timestamp": "2024-01-01T00:05:00"}
        ]
        
        history = event_handler.get_task_history(task_id)
        assert len(history) == 2
        assert history[0]["event_type"] == "task_created"
    
    def test_get_task_state(self, event_handler):
        """Test retrieving task state."""
        task_id = "test-task-state"
        
        # No state initially
        state = event_handler.get_task_state(task_id)
        assert state is None
        
        # Add task state
        event_handler.active_tasks[task_id] = {
            "status": "running",
            "progress": 50,
            "user_id": 1
        }
        
        state = event_handler.get_task_state(task_id)
        assert state["status"] == "running"
        assert state["progress"] == 50
    
    def test_get_active_tasks_for_user(self, event_handler):
        """Test retrieving active tasks for a user."""
        user_id = 1
        
        # Add tasks for different users
        event_handler.active_tasks["task1"] = {"user_id": 1, "status": "running"}
        event_handler.active_tasks["task2"] = {"user_id": 2, "status": "running"}
        event_handler.active_tasks["task3"] = {"user_id": 1, "status": "completed"}
        
        user_tasks = event_handler.get_active_tasks_for_user(user_id)
        
        assert len(user_tasks) == 2
        task_ids = [task["task_id"] for task in user_tasks]
        assert "task1" in task_ids
        assert "task3" in task_ids
        assert "task2" not in task_ids


class TestConvenienceFunctions:
    """Test convenience functions for event emission."""
    
    @pytest.mark.asyncio
    async def test_emit_task_created(self):
        """Test emit_task_created convenience function."""
        with patch('src.api.verification_events.verification_event_handler.emit_event') as mock_emit:
            await emit_task_created(
                task_id="test-123",
                user_id=1,
                document_name="test.pdf",
                verification_type="full"
            )
        
        mock_emit.assert_called_once()
        args, kwargs = mock_emit.call_args
        assert kwargs["task_id"] == "test-123"
        assert kwargs["event_type"] == VerificationEventType.TASK_CREATED
        assert kwargs["user_id"] == 1
        assert kwargs["data"]["document_name"] == "test.pdf"
    
    @pytest.mark.asyncio
    async def test_emit_task_progress(self):
        """Test emit_task_progress convenience function."""
        with patch('src.api.verification_events.verification_event_handler.emit_event') as mock_emit:
            await emit_task_progress(
                task_id="test-456",
                progress=75,
                step="Finalizing report",
                details="Almost done"
            )
        
        mock_emit.assert_called_once()
        args, kwargs = mock_emit.call_args
        assert kwargs["task_id"] == "test-456"
        assert kwargs["event_type"] == VerificationEventType.TASK_PROGRESS
        assert kwargs["data"]["progress"] == 75
        assert kwargs["data"]["step"] == "Finalizing report"
        assert kwargs["data"]["details"] == "Almost done"
    
    @pytest.mark.asyncio
    async def test_emit_task_completed(self):
        """Test emit_task_completed convenience function."""
        with patch('src.api.verification_events.verification_event_handler.emit_event') as mock_emit:
            await emit_task_completed(
                task_id="test-789",
                user_id=2,
                results={"total_issues": 5, "passes_completed": 8},
                duration=120.5
            )
        
        mock_emit.assert_called_once()
        args, kwargs = mock_emit.call_args
        assert kwargs["task_id"] == "test-789"
        assert kwargs["event_type"] == VerificationEventType.TASK_COMPLETED
        assert kwargs["user_id"] == 2
        assert kwargs["data"]["results"]["total_issues"] == 5
        assert kwargs["data"]["duration_seconds"] == 120.5


if __name__ == "__main__":
    pytest.main([__file__]) 