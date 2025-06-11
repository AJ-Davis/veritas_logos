"""
Metrics Collector for real-time analytics data collection.

This module provides comprehensive metrics collection capabilities for the
Veritas Logos verification system, capturing events, performance data,
and user interactions for analytics and monitoring.
"""

import uuid
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
from contextlib import asynccontextmanager

from .clickhouse_client import get_clickhouse_client

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be collected."""
    VERIFICATION_START = "verification_start"
    VERIFICATION_COMPLETE = "verification_complete"
    VERIFICATION_FAILED = "verification_failed"
    ISSUE_DETECTED = "issue_detected"
    ACVF_TRIGGERED = "acvf_triggered"
    USER_ACTIVITY = "user_activity"
    API_REQUEST = "api_request"
    DEBATE_VIEW = "debate_view"
    DOCUMENT_UPLOAD = "document_upload"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_HEALTH = "system_health"


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"          # Incrementing value
    GAUGE = "gauge"              # Point-in-time value
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Rate of events per time period


@dataclass
class Event:
    """Represents an event to be collected."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Represents a metric to be collected."""
    metric_id: str
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Real-time metrics collection system for comprehensive analytics.
    
    Collects events, metrics, and performance data from the verification
    system and stores them in ClickHouse for analysis and monitoring.
    """
    
    def __init__(self, batch_size: int = 100, flush_interval: int = 30):
        """
        Initialize the metrics collector.
        
        Args:
            batch_size: Number of events to batch before flushing
            flush_interval: Seconds between automatic flushes
        """
        self.logger = logging.getLogger("analytics.metrics_collector")
        self.clickhouse_client = get_clickhouse_client()
        
        # Configuration
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Buffers for batching
        self.event_buffer: List[Event] = []
        self.metric_buffer: List[Metric] = []
        self.buffer_lock = threading.Lock()
        
        # Background processing
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance tracking
        self.collection_stats = {
            "events_collected": 0,
            "metrics_collected": 0,
            "flushes_completed": 0,
            "errors": 0,
            "last_flush": None
        }
    
    async def start(self) -> None:
        """Start the metrics collector background processing."""
        if self.running:
            return
        
        self.running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        self.logger.info("Metrics collector started")
    
    async def stop(self) -> None:
        """Stop the metrics collector and flush remaining data."""
        if not self.running:
            return
        
        self.running = False
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        self.logger.info("Metrics collector stopped")
    
    def collect_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: Optional[bool] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Collect an event for analytics.
        
        Args:
            event_type: Type of event
            user_id: ID of the user (if applicable)
            document_id: ID of the document (if applicable)
            session_id: Session identifier
            properties: Event-specific properties
            duration_ms: Duration of the event in milliseconds
            success: Whether the event was successful
            error_message: Error message if unsuccessful
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            document_id=document_id,
            session_id=session_id,
            properties=properties or {},
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        with self.buffer_lock:
            self.event_buffer.append(event)
            self.collection_stats["events_collected"] += 1
        
        # Trigger flush if buffer is full
        if len(self.event_buffer) >= self.batch_size:
            asyncio.create_task(self.flush())
        
        return event.event_id
    
    def collect_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "count",
        dimensions: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Collect a metric for analytics.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric
            unit: Unit of measurement
            dimensions: Metric dimensions for grouping
            user_id: ID of the user (if applicable)
            document_id: ID of the document (if applicable)
            tags: Additional tags
            
        Returns:
            Metric ID
        """
        metric = Metric(
            metric_id=str(uuid.uuid4()),
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(timezone.utc),
            dimensions=dimensions or {},
            user_id=user_id,
            document_id=document_id,
            tags=tags or {}
        )
        
        with self.buffer_lock:
            self.metric_buffer.append(metric)
            self.collection_stats["metrics_collected"] += 1
        
        # Trigger flush if buffer is full
        if len(self.metric_buffer) >= self.batch_size:
            asyncio.create_task(self.flush())
        
        return metric.metric_id
    
    async def flush(self) -> None:
        """Flush buffered events and metrics to ClickHouse."""
        events_to_flush = []
        metrics_to_flush = []
        
        # Get items from buffers
        with self.buffer_lock:
            if self.event_buffer:
                events_to_flush = self.event_buffer.copy()
                self.event_buffer.clear()
            
            if self.metric_buffer:
                metrics_to_flush = self.metric_buffer.copy()
                self.metric_buffer.clear()
        
        # Process events
        if events_to_flush:
            try:
                await self._flush_events(events_to_flush)
            except Exception as e:
                self.logger.error(f"Failed to flush events: {e}")
                self.collection_stats["errors"] += 1
                # Re-queue events for retry
                with self.buffer_lock:
                    self.event_buffer.extend(events_to_flush)
        
        # Process metrics
        if metrics_to_flush:
            try:
                await self._flush_metrics(metrics_to_flush)
            except Exception as e:
                self.logger.error(f"Failed to flush metrics: {e}")
                self.collection_stats["errors"] += 1
                # Re-queue metrics for retry
                with self.buffer_lock:
                    self.metric_buffer.extend(metrics_to_flush)
        
        if events_to_flush or metrics_to_flush:
            self.collection_stats["flushes_completed"] += 1
            self.collection_stats["last_flush"] = datetime.now(timezone.utc)
    
    async def _flush_events(self, events: List[Event]) -> None:
        """Flush events to ClickHouse."""
        for event in events:
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "document_id": event.document_id or "",
                "user_id": event.user_id or "",
                "verification_type": event.properties.get("verification_type", ""),
                "duration_ms": event.duration_ms or 0.0,
                "success": event.success if event.success is not None else True,
                "error_message": event.error_message or "",
                "metadata": str(event.metadata)
            }
            
            await self.clickhouse_client.insert_verification_event(event_data)
    
    async def _flush_metrics(self, metrics: List[Metric]) -> None:
        """Flush metrics to ClickHouse."""
        for metric in metrics:
            # Convert dimensions to string
            dimensions_str = ",".join([f"{k}:{v}" for k, v in metric.dimensions.items()])
            
            metric_data = {
                "metric_id": metric.metric_id,
                "timestamp": metric.timestamp,
                "metric_name": metric.metric_name,
                "metric_value": metric.value,
                "metric_unit": metric.unit,
                "dimensions": dimensions_str,
                "document_id": metric.document_id or "",
                "user_id": metric.user_id or ""
            }
            
            await self.clickhouse_client.insert_kpi_metric(metric_data)
    
    async def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    @asynccontextmanager
    async def timed_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for timing events.
        
        Usage:
            async with collector.timed_event(EventType.VERIFICATION_COMPLETE, user_id="123"):
                # Do verification work
                pass
        """
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            self.collect_event(
                event_type=event_type,
                user_id=user_id,
                document_id=document_id,
                session_id=session_id,
                properties=properties,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                metadata=metadata
            )
    
    def increment_counter(
        self,
        metric_name: str,
        value: float = 1.0,
        dimensions: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Convenience method for incrementing counters."""
        return self.collect_metric(
            metric_name=metric_name,
            value=value,
            metric_type=MetricType.COUNTER,
            unit="count",
            dimensions=dimensions,
            user_id=user_id,
            document_id=document_id
        )
    
    def record_gauge(
        self,
        metric_name: str,
        value: float,
        unit: str = "value",
        dimensions: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Convenience method for recording gauge values."""
        return self.collect_metric(
            metric_name=metric_name,
            value=value,
            metric_type=MetricType.GAUGE,
            unit=unit,
            dimensions=dimensions,
            user_id=user_id,
            document_id=document_id
        )
    
    def record_timer(
        self,
        metric_name: str,
        duration_ms: float,
        dimensions: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Convenience method for recording timing data."""
        return self.collect_metric(
            metric_name=metric_name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            unit="milliseconds",
            dimensions=dimensions,
            user_id=user_id,
            document_id=document_id
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.collection_stats.copy()


# Global metrics collector instance
metrics_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global metrics_collector_instance
    if metrics_collector_instance is None:
        metrics_collector_instance = MetricsCollector()
    return metrics_collector_instance


# Convenience functions for common metrics
async def track_verification_event(
    event_type: EventType,
    user_id: Optional[str] = None,
    document_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    success: Optional[bool] = None,
    error_message: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None
) -> str:
    """Track a verification-related event."""
    collector = get_metrics_collector()
    return collector.collect_event(
        event_type=event_type,
        user_id=user_id,
        document_id=document_id,
        duration_ms=duration_ms,
        success=success,
        error_message=error_message,
        properties=properties
    )


async def track_user_activity(
    activity_type: str,
    user_id: str,
    duration_ms: Optional[float] = None,
    properties: Optional[Dict[str, Any]] = None
) -> str:
    """Track user activity events."""
    collector = get_metrics_collector()
    return collector.collect_event(
        event_type=EventType.USER_ACTIVITY,
        user_id=user_id,
        duration_ms=duration_ms,
        properties={"activity_type": activity_type, **(properties or {})}
    )


async def track_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None
) -> str:
    """Track API request metrics."""
    collector = get_metrics_collector()
    return collector.collect_event(
        event_type=EventType.API_REQUEST,
        user_id=user_id,
        duration_ms=duration_ms,
        success=200 <= status_code < 400,
        properties={
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code
        }
    ) 