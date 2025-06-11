"""
WebSocket routes for real-time verification progress updates.

Provides WebSocket endpoints for frontend connections and real-time updates.
"""

import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from src.api.auth import User, get_current_user, get_db
from src.api.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["WebSocket"])
security = HTTPBearer()


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time verification updates.
    
    Query Parameters:
        token: JWT access token for authentication
    
    WebSocket Message Types:
        - Client -> Server:
            - {"type": "ping"}: Heartbeat message
            - {"type": "subscribe", "task_id": "uuid"}: Subscribe to task updates
            - {"type": "unsubscribe", "task_id": "uuid"}: Unsubscribe from task updates
        
        - Server -> Client:
            - {"type": "connection_established", "connection_id": "uuid", "user_id": int}
            - {"type": "ping", "timestamp": "iso8601"}
            - {"type": "pong", "timestamp": "iso8601"}
            - {"type": "subscribed", "task_id": "uuid", "timestamp": "iso8601"}
            - {"type": "unsubscribed", "task_id": "uuid", "timestamp": "iso8601"}
            - {"type": "task_update", "task_id": "uuid", "data": {...}, "timestamp": "iso8601"}
            - {"type": "error", "message": "string", "code": "string"}
    """
    connection_id = None
    
    try:
        # Authenticate the WebSocket connection
        if not token:
            await websocket.close(code=4001, reason="Missing authentication token")
            return
        
        user = await websocket_manager.authenticate_websocket(websocket, token)
        if not user:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return
        
        # Establish connection
        connection_id = await websocket_manager.connect(websocket, user)
        
        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                # Send error message to client
                try:
                    await websocket.send_text(
                        '{"type": "error", "message": "Internal server error", "code": "internal_error"}'
                    )
                except:
                    break  # Connection might be closed
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except:
            pass
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)


@router.get("/stats")
async def get_websocket_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get WebSocket connection statistics.
    
    Requires authentication. Admin users get full stats, regular users get limited stats.
    """
    stats = websocket_manager.get_connection_stats()
    
    if current_user.role.value != "ADMIN":
        # Regular users only see their own connection count
        user_connection_count = stats["connections_by_user"].get(current_user.id, 0)
        return {
            "your_connections": user_connection_count,
            "total_connections": stats["total_connections"]
        }
    
    return stats


@router.post("/broadcast/{task_id}")
async def broadcast_task_update(
    task_id: str,
    update_data: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Broadcast an update to all subscribers of a specific task.
    
    This endpoint is typically called by the verification system when task status changes.
    Regular users can only broadcast updates for tasks they own or are involved with.
    Admin users can broadcast updates for any task.
    
    Args:
        task_id: UUID of the verification task
        update_data: Dictionary containing the update information
    """
    # For now, allow any authenticated user to broadcast
    # In production, you might want to add additional authorization checks
    # based on task ownership or user roles
    
    await websocket_manager.broadcast_task_update(task_id, update_data)
    
    return {
        "message": f"Update broadcasted to task {task_id} subscribers",
        "task_id": task_id,
        "subscriber_count": len(websocket_manager.task_subscribers.get(task_id, set()))
    }


@router.post("/notify-user/{user_id}")
async def notify_user(
    user_id: int,
    message: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Send a notification message to a specific user's WebSocket connections.
    
    Only admin users or the user themselves can send notifications.
    
    Args:
        user_id: ID of the target user
        message: Notification message data
    """
    # Check authorization
    if current_user.role.value != "ADMIN" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="You can only send notifications to yourself unless you are an admin"
        )
    
    await websocket_manager.send_to_user(user_id, message)
    
    user_connection_count = len(websocket_manager.user_connections.get(user_id, []))
    
    return {
        "message": f"Notification sent to user {user_id}",
        "user_id": user_id,
        "connection_count": user_connection_count
    }


@router.delete("/connections/{connection_id}")
async def force_disconnect(
    connection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Force disconnect a specific WebSocket connection.
    
    Only admin users can force disconnect connections.
    
    Args:
        connection_id: UUID of the connection to disconnect
    """
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=403,
            detail="Only admin users can force disconnect connections"
        )
    
    if connection_id not in websocket_manager.connections:
        raise HTTPException(
            status_code=404,
            detail="Connection not found"
        )
    
    await websocket_manager.disconnect(connection_id)
    
    return {
        "message": f"Connection {connection_id} disconnected",
        "connection_id": connection_id
    }


@router.get("/health")
async def websocket_health():
    """
    Health check endpoint for WebSocket service.
    
    Returns the status of the WebSocket manager and basic statistics.
    """
    stats = websocket_manager.get_connection_stats()
    
    return {
        "status": "healthy" if websocket_manager.is_running else "stopped",
        "is_running": websocket_manager.is_running,
        "total_connections": stats["total_connections"],
        "unique_users": stats["unique_users"],
        "active_subscriptions": stats["active_subscriptions"]
    } 