"""
Test suite for WebSocket API functionality.

Tests WebSocket connection management, authentication, and event broadcasting.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from src.api.websocket_manager import WebSocketManager, WebSocketConnection
from src.api.verification_events import VerificationEventHandler, VerificationEventType


class TestWebSocketManager:
    """Test WebSocket manager functionality."""
    
    def test_manager_initialization(self):
        """Test WebSocket manager initialization."""
        manager = WebSocketManager()
        assert manager.connections == {}
        assert manager.user_connections == {}
        assert manager.task_subscribers == {}
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_manager_start_stop(self):
        """Test manager startup and shutdown."""
        manager = WebSocketManager()
        
        await manager.start()
        assert manager.is_running is True
        
        await manager.stop()
        assert manager.is_running is False
    
    def test_connection_stats(self):
        """Test connection statistics."""
        manager = WebSocketManager()
        stats = manager.get_connection_stats()
        
        assert "total_connections" in stats
        assert "unique_users" in stats
        assert "active_subscriptions" in stats
        assert stats["total_connections"] == 0


class TestWebSocketConnection:
    """Test WebSocket connection functionality."""
    
    def test_connection_creation(self):
        """Test WebSocket connection creation."""
        mock_websocket = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 1
        
        connection = WebSocketConnection(mock_websocket, mock_user, "test-id")
        
        assert connection.websocket == mock_websocket
        assert connection.user == mock_user
        assert connection.connection_id == "test-id"
        assert connection.is_active is True
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending messages through WebSocket."""
        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()
        mock_user = MagicMock()
        
        connection = WebSocketConnection(mock_websocket, mock_user, "test-id")
        message = {"type": "test", "data": "hello"}
        
        result = await connection.send_message(message)
        
        assert result is True
        mock_websocket.send_text.assert_called_once_with(json.dumps(message))


class TestVerificationEventHandler:
    """Test verification event handler."""
    
    def test_event_handler_initialization(self):
        """Test event handler initialization."""
        handler = VerificationEventHandler()
        assert handler.event_history == {}
        assert handler.active_tasks == {}
    
    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test event emission."""
        handler = VerificationEventHandler()
        task_id = "test-task"
        
        with patch('src.api.verification_events.websocket_manager') as mock_manager:
            mock_manager.broadcast_task_update = AsyncMock()
            
            await handler.emit_event(
                task_id=task_id,
                event_type=VerificationEventType.TASK_CREATED,
                data={"test": "data"},
                user_id=1
            )
        
        assert task_id in handler.event_history
        assert task_id in handler.active_tasks
        mock_manager.broadcast_task_update.assert_called_once()
    
    def test_get_task_history(self):
        """Test retrieving task history."""
        handler = VerificationEventHandler()
        task_id = "test-task"
        
        # Empty history initially
        history = handler.get_task_history(task_id)
        assert history == []
        
        # Add some history
        handler.event_history[task_id] = [{"event": "test"}]
        history = handler.get_task_history(task_id)
        assert len(history) == 1


class TestWebSocketAuthentication:
    """Test WebSocket authentication."""
    
    @pytest.mark.asyncio
    async def test_authentication_with_invalid_token(self):
        """Test authentication with invalid token."""
        manager = WebSocketManager()
        
        user = await manager.authenticate_websocket(MagicMock(), "invalid-token")
        assert user is None
    
    @patch('src.api.websocket_manager.get_db')
    @patch('src.api.websocket_manager.jwt.decode')
    @pytest.mark.asyncio
    async def test_authentication_success(self, mock_jwt_decode, mock_get_db):
        """Test successful authentication."""
        manager = WebSocketManager()
        
        # Mock JWT decode
        mock_jwt_decode.return_value = {"sub": "1", "type": "access"}
        
        # Mock database query
        mock_user = MagicMock()
        mock_user.status.value = "ACTIVE"
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_get_db.return_value = iter([mock_db])
        
        user = await manager.authenticate_websocket(MagicMock(), "valid-token")
        assert user == mock_user


if __name__ == "__main__":
    pytest.main([__file__]) 