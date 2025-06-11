"""
Verification event handling for real-time WebSocket updates.

This module provides event handlers that integrate with the ACVF verification pipeline
to broadcast real-time progress updates to connected clients.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from uuid import UUID

from src.api.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class VerificationEventType(Enum):
    """Types of verification events that can be broadcasted."""
    
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    
    # Detailed progress events
    DOCUMENT_PARSED = "document_parsed"
    CHAIN_STARTED = "chain_started"
    CHAIN_COMPLETED = "chain_completed"
    PASS_STARTED = "pass_started"
    PASS_COMPLETED = "pass_completed"
    ISSUE_DETECTED = "issue_detected"
    
    # Output generation events
    OUTPUT_GENERATION_STARTED = "output_generation_started"
    OUTPUT_GENERATION_COMPLETED = "output_generation_completed"


class VerificationEventHandler:
    """Handles verification events and broadcasts them to WebSocket clients."""
    
    def __init__(self):
        self.event_history: Dict[str, List[Dict[str, Any]]] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def emit_event(
        self,
        task_id: str,
        event_type: VerificationEventType,
        data: Dict[str, Any],
        user_id: Optional[int] = None
    ):
        """
        Emit a verification event to all subscribers.
        
        Args:
            task_id: UUID of the verification task
            event_type: Type of event being emitted
            data: Event-specific data
            user_id: Optional user ID for user-specific events
        """
        try:
            event_data = {
                "event_type": event_type.value,
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Store event in history
            if task_id not in self.event_history:
                self.event_history[task_id] = []
            self.event_history[task_id].append(event_data)
            
            # Update active task state
            if task_id not in self.active_tasks:
                self.active_tasks[task_id] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "unknown",
                    "progress": 0,
                    "user_id": user_id
                }
            
            # Update task state based on event type
            self._update_task_state(task_id, event_type, data)
            
            # Broadcast to WebSocket subscribers
            await websocket_manager.broadcast_task_update(task_id, event_data)
            
            # Send user-specific notification if needed
            if user_id and event_type in [
                VerificationEventType.TASK_COMPLETED,
                VerificationEventType.TASK_FAILED,
                VerificationEventType.TASK_CANCELLED
            ]:
                await self._send_user_notification(user_id, event_data)
            
            logger.debug(f"Emitted event {event_type.value} for task {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to emit event {event_type.value} for task {task_id}: {e}")
    
    def _update_task_state(self, task_id: str, event_type: VerificationEventType, data: Dict[str, Any]):
        """Update internal task state based on event."""
        task_state = self.active_tasks[task_id]
        
        if event_type == VerificationEventType.TASK_CREATED:
            task_state["status"] = "created"
            task_state["progress"] = 0
        
        elif event_type == VerificationEventType.TASK_STARTED:
            task_state["status"] = "running"
            task_state["progress"] = 5
            task_state["started_at"] = datetime.utcnow().isoformat()
        
        elif event_type == VerificationEventType.TASK_PROGRESS:
            task_state["progress"] = data.get("progress", task_state["progress"])
            task_state["current_step"] = data.get("step", "")
        
        elif event_type == VerificationEventType.DOCUMENT_PARSED:
            task_state["progress"] = 10
            task_state["document_info"] = data.get("document_info", {})
        
        elif event_type == VerificationEventType.CHAIN_STARTED:
            task_state["progress"] = 15
            task_state["current_chain"] = data.get("chain_name", "")
        
        elif event_type == VerificationEventType.PASS_STARTED:
            task_state["current_pass"] = data.get("pass_name", "")
            # Increment progress for each pass
            task_state["progress"] = min(task_state["progress"] + 10, 85)
        
        elif event_type == VerificationEventType.PASS_COMPLETED:
            task_state["completed_passes"] = task_state.get("completed_passes", 0) + 1
        
        elif event_type == VerificationEventType.ISSUE_DETECTED:
            task_state["issues_found"] = task_state.get("issues_found", 0) + 1
        
        elif event_type == VerificationEventType.OUTPUT_GENERATION_STARTED:
            task_state["progress"] = 90
            task_state["status"] = "generating_output"
        
        elif event_type == VerificationEventType.TASK_COMPLETED:
            task_state["status"] = "completed"
            task_state["progress"] = 100
            task_state["completed_at"] = datetime.utcnow().isoformat()
            task_state["results"] = data.get("results", {})
        
        elif event_type == VerificationEventType.TASK_FAILED:
            task_state["status"] = "failed"
            task_state["failed_at"] = datetime.utcnow().isoformat()
            task_state["error"] = data.get("error", "Unknown error")
        
        elif event_type == VerificationEventType.TASK_CANCELLED:
            task_state["status"] = "cancelled"
            task_state["cancelled_at"] = datetime.utcnow().isoformat()
    
    async def _send_user_notification(self, user_id: int, event_data: Dict[str, Any]):
        """Send a notification to a specific user."""
        notification = {
            "type": "task_notification",
            "notification_type": event_data["event_type"],
            "task_id": event_data["task_id"],
            "timestamp": event_data["timestamp"],
            "message": self._get_notification_message(event_data),
            "data": event_data["data"]
        }
        
        await websocket_manager.send_to_user(user_id, notification)
    
    def _get_notification_message(self, event_data: Dict[str, Any]) -> str:
        """Generate a user-friendly notification message."""
        event_type = event_data["event_type"]
        task_id = event_data["task_id"][:8]  # Short task ID for display
        
        messages = {
            "task_completed": f"Verification task {task_id} has completed successfully",
            "task_failed": f"Verification task {task_id} has failed",
            "task_cancelled": f"Verification task {task_id} was cancelled"
        }
        
        return messages.get(event_type, f"Task {task_id} event: {event_type}")
    
    def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Get event history for a specific task."""
        return self.event_history.get(task_id, [])
    
    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a task."""
        return self.active_tasks.get(task_id)
    
    def cleanup_completed_task(self, task_id: str):
        """Clean up data for a completed task (optional)."""
        # Keep history but remove from active tasks after some time
        if task_id in self.active_tasks:
            task_state = self.active_tasks[task_id]
            if task_state["status"] in ["completed", "failed", "cancelled"]:
                # Could implement a cleanup strategy here
                pass
    
    def get_active_tasks_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all active tasks for a specific user."""
        user_tasks = []
        for task_id, task_state in self.active_tasks.items():
            if task_state.get("user_id") == user_id:
                user_tasks.append({
                    "task_id": task_id,
                    **task_state
                })
        return user_tasks


# Global event handler instance
verification_event_handler = VerificationEventHandler()


# Convenience functions for common verification events
async def emit_task_created(task_id: str, user_id: int, document_name: str, verification_type: str):
    """Emit task created event."""
    await verification_event_handler.emit_event(
        task_id=task_id,
        event_type=VerificationEventType.TASK_CREATED,
        data={
            "document_name": document_name,
            "verification_type": verification_type,
            "user_id": user_id
        },
        user_id=user_id
    )


async def emit_task_started(task_id: str, chains: List[str]):
    """Emit task started event."""
    await verification_event_handler.emit_event(
        task_id=task_id,
        event_type=VerificationEventType.TASK_STARTED,
        data={
            "chains": chains,
            "total_chains": len(chains)
        }
    )


async def emit_task_progress(task_id: str, progress: int, step: str, details: Optional[str] = None):
    """Emit task progress update."""
    await verification_event_handler.emit_event(
        task_id=task_id,
        event_type=VerificationEventType.TASK_PROGRESS,
        data={
            "progress": progress,
            "step": step,
            "details": details
        }
    )


async def emit_task_completed(
    task_id: str,
    user_id: int,
    results: Dict[str, Any],
    duration: float
):
    """Emit task completed event."""
    await verification_event_handler.emit_event(
        task_id=task_id,
        event_type=VerificationEventType.TASK_COMPLETED,
        data={
            "results": results,
            "duration_seconds": duration,
            "issues_found": results.get("total_issues", 0),
            "passes_completed": results.get("passes_completed", 0)
        },
        user_id=user_id
    )


async def emit_task_failed(task_id: str, user_id: int, error: str, step: str):
    """Emit task failed event."""
    await verification_event_handler.emit_event(
        task_id=task_id,
        event_type=VerificationEventType.TASK_FAILED,
        data={
            "error": error,
            "failed_step": step
        },
        user_id=user_id
    )


async def emit_issue_detected(
    task_id: str,
    issue_type: str,
    severity: str,
    location: str,
    description: str
):
    """Emit issue detected event."""
    await verification_event_handler.emit_event(
        task_id=task_id,
        event_type=VerificationEventType.ISSUE_DETECTED,
        data={
            "issue_type": issue_type,
            "severity": severity,
            "location": location,
            "description": description
        }
    ) 