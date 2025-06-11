"""
ClickHouse client for analytics data storage and retrieval.

This module provides the interface for storing and querying analytics data
in ClickHouse, optimized for high-volume time-series data and fast analytics.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import clickhouse_connect
from clickhouse_connect.driver import Client
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ClickHouseConfig:
    """Configuration for ClickHouse connection."""
    host: str = "localhost"
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "veritas_analytics"
    secure: bool = False
    verify: bool = True
    compress: bool = True


class ClickHouseClient:
    """
    Client for interacting with ClickHouse for analytics data storage.
    
    Provides high-performance analytics storage for metrics, events,
    and time-series data with optimized queries for dashboard visualization.
    """
    
    def __init__(self, config: Optional[ClickHouseConfig] = None):
        """Initialize ClickHouse client with configuration."""
        self.config = config or self._load_config_from_env()
        self.client: Optional[Client] = None
        self.logger = logging.getLogger("analytics.clickhouse")
        
    def _load_config_from_env(self) -> ClickHouseConfig:
        """Load ClickHouse configuration from environment variables."""
        return ClickHouseConfig(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            username=os.getenv("CLICKHOUSE_USERNAME", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            database=os.getenv("CLICKHOUSE_DATABASE", "veritas_analytics"),
            secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
            verify=os.getenv("CLICKHOUSE_VERIFY", "true").lower() == "true",
            compress=os.getenv("CLICKHOUSE_COMPRESS", "true").lower() == "true"
        )
    
    async def connect(self) -> None:
        """Establish connection to ClickHouse."""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                secure=self.config.secure,
                verify=self.config.verify,
                compress=self.config.compress
            )
            
            # Test connection
            result = self.client.command("SELECT 1")
            self.logger.info("Successfully connected to ClickHouse")
            
            # Initialize database schema
            await self._initialize_schema()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to ClickHouse: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close ClickHouse connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.logger.info("Disconnected from ClickHouse")
    
    async def _initialize_schema(self) -> None:
        """Initialize ClickHouse database schema for analytics."""
        schemas = [
            # Verification events table
            """
            CREATE TABLE IF NOT EXISTS verification_events (
                event_id UUID,
                timestamp DateTime64(3) DEFAULT now64(),
                event_type String,
                document_id String,
                user_id String,
                verification_type String,
                duration_ms Float64,
                success Boolean,
                error_message String,
                metadata String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY date
            ORDER BY (timestamp, event_type, document_id)
            """,
            
            # KPI metrics table
            """
            CREATE TABLE IF NOT EXISTS kpi_metrics (
                metric_id UUID,
                timestamp DateTime64(3) DEFAULT now64(),
                metric_name String,
                metric_value Float64,
                metric_unit String,
                dimensions String,
                document_id String,
                user_id String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY date
            ORDER BY (timestamp, metric_name)
            """,
            
            # ACVF debate events table
            """
            CREATE TABLE IF NOT EXISTS acvf_events (
                event_id UUID,
                timestamp DateTime64(3) DEFAULT now64(),
                debate_round_id String,
                event_type String,
                participant_role String,
                participant_model String,
                argument_length Int32,
                confidence_score Float64,
                processing_time_ms Float64,
                verdict String,
                metadata String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY date
            ORDER BY (timestamp, debate_round_id, event_type)
            """,
            
            # User activity table
            """
            CREATE TABLE IF NOT EXISTS user_activity (
                activity_id UUID,
                timestamp DateTime64(3) DEFAULT now64(),
                user_id String,
                activity_type String,
                resource_id String,
                resource_type String,
                session_id String,
                ip_address String,
                user_agent String,
                metadata String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY date
            ORDER BY (timestamp, user_id, activity_type)
            """,
            
            # System performance metrics table
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id UUID,
                timestamp DateTime64(3) DEFAULT now64(),
                metric_name String,
                metric_value Float64,
                service_name String,
                instance_id String,
                tags String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY date
            ORDER BY (timestamp, metric_name, service_name)
            """,
            
            # A/B testing events table
            """
            CREATE TABLE IF NOT EXISTS ab_test_events (
                event_id UUID,
                timestamp DateTime64(3) DEFAULT now64(),
                test_name String,
                variant String,
                user_id String,
                event_type String,
                conversion_value Float64,
                metadata String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY date
            ORDER BY (timestamp, test_name, variant)
            """
        ]
        
        for schema in schemas:
            try:
                self.client.command(schema)
                self.logger.debug(f"Created/verified table schema")
            except Exception as e:
                self.logger.error(f"Failed to create schema: {e}")
                raise
    
    async def insert_verification_event(self, event_data: Dict[str, Any]) -> None:
        """Insert a verification event into ClickHouse."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        try:
            self.client.insert(
                "verification_events",
                [event_data],
                column_names=[
                    "event_id", "timestamp", "event_type", "document_id",
                    "user_id", "verification_type", "duration_ms", "success",
                    "error_message", "metadata"
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to insert verification event: {e}")
            raise
    
    async def insert_kpi_metric(self, metric_data: Dict[str, Any]) -> None:
        """Insert a KPI metric into ClickHouse."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        try:
            self.client.insert(
                "kpi_metrics",
                [metric_data],
                column_names=[
                    "metric_id", "timestamp", "metric_name", "metric_value",
                    "metric_unit", "dimensions", "document_id", "user_id"
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to insert KPI metric: {e}")
            raise
    
    async def insert_acvf_event(self, event_data: Dict[str, Any]) -> None:
        """Insert an ACVF debate event into ClickHouse."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        try:
            self.client.insert(
                "acvf_events",
                [event_data],
                column_names=[
                    "event_id", "timestamp", "debate_round_id", "event_type",
                    "participant_role", "participant_model", "argument_length",
                    "confidence_score", "processing_time_ms", "verdict", "metadata"
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to insert ACVF event: {e}")
            raise
    
    async def insert_user_activity(self, activity_data: Dict[str, Any]) -> None:
        """Insert user activity data into ClickHouse."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        try:
            self.client.insert(
                "user_activity",
                [activity_data],
                column_names=[
                    "activity_id", "timestamp", "user_id", "activity_type",
                    "resource_id", "resource_type", "session_id", "ip_address",
                    "user_agent", "metadata"
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to insert user activity: {e}")
            raise
    
    async def query_verification_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        group_by: str = "toStartOfDay(timestamp)",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query verification metrics with time range and grouping."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        where_clause = "timestamp BETWEEN %(start_time)s AND %(end_time)s"
        params = {
            "start_time": start_time,
            "end_time": end_time
        }
        
        # Add additional filters
        if filters:
            for key, value in filters.items():
                where_clause += f" AND {key} = %({key})s"
                params[key] = value
        
        query = f"""
        SELECT
            {group_by} as time_bucket,
            count(*) as total_events,
            countIf(success = true) as successful_events,
            countIf(success = false) as failed_events,
            avg(duration_ms) as avg_duration_ms,
            uniq(document_id) as unique_documents,
            uniq(user_id) as unique_users
        FROM verification_events
        WHERE {where_clause}
        GROUP BY time_bucket
        ORDER BY time_bucket
        """
        
        try:
            result = self.client.query(query, parameters=params)
            return result.result_rows
        except Exception as e:
            self.logger.error(f"Failed to query verification metrics: {e}")
            raise
    
    async def query_kpi_trends(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
        group_by: str = "toStartOfDay(timestamp)"
    ) -> List[Dict[str, Any]]:
        """Query KPI metric trends over time."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        metric_names_str = "', '".join(metric_names)
        
        query = f"""
        SELECT
            {group_by} as time_bucket,
            metric_name,
            avg(metric_value) as avg_value,
            min(metric_value) as min_value,
            max(metric_value) as max_value,
            count(*) as sample_count
        FROM kpi_metrics
        WHERE timestamp BETWEEN %(start_time)s AND %(end_time)s
        AND metric_name IN ('{metric_names_str}')
        GROUP BY time_bucket, metric_name
        ORDER BY time_bucket, metric_name
        """
        
        try:
            result = self.client.query(query, parameters={
                "start_time": start_time,
                "end_time": end_time
            })
            return result.result_rows
        except Exception as e:
            self.logger.error(f"Failed to query KPI trends: {e}")
            raise
    
    async def query_acvf_statistics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Query ACVF debate statistics."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        query = """
        SELECT
            uniq(debate_round_id) as total_debates,
            countIf(event_type = 'debate_completed') as completed_debates,
            avg(processing_time_ms) as avg_processing_time,
            avgIf(confidence_score, confidence_score > 0) as avg_confidence,
            count(*) as total_events,
            groupUniqArray(verdict) as verdicts
        FROM acvf_events
        WHERE timestamp BETWEEN %(start_time)s AND %(end_time)s
        """
        
        try:
            result = self.client.query(query, parameters={
                "start_time": start_time,
                "end_time": end_time
            })
            return dict(zip(result.column_names, result.result_rows[0])) if result.result_rows else {}
        except Exception as e:
            self.logger.error(f"Failed to query ACVF statistics: {e}")
            raise
    
    async def query_user_engagement(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Query user engagement metrics."""
        if not self.client:
            raise RuntimeError("ClickHouse client not connected")
        
        query = """
        SELECT
            uniq(user_id) as active_users,
            count(*) as total_activities,
            uniq(session_id) as total_sessions,
            avg(countIf(activity_type = 'document_upload')) as avg_uploads_per_user,
            countIf(activity_type = 'debate_view') as debate_views,
            countIf(activity_type = 'dashboard_view') as dashboard_views
        FROM user_activity
        WHERE timestamp BETWEEN %(start_time)s AND %(end_time)s
        """
        
        try:
            result = self.client.query(query, parameters={
                "start_time": start_time,
                "end_time": end_time
            })
            return dict(zip(result.column_names, result.result_rows[0])) if result.result_rows else {}
        except Exception as e:
            self.logger.error(f"Failed to query user engagement: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check ClickHouse connection health."""
        try:
            if not self.client:
                return False
            result = self.client.command("SELECT 1")
            return result == 1
        except Exception:
            return False


# Global ClickHouse client instance
clickhouse_client: Optional[ClickHouseClient] = None


def get_clickhouse_client() -> ClickHouseClient:
    """Get the global ClickHouse client instance."""
    global clickhouse_client
    if clickhouse_client is None:
        clickhouse_client = ClickHouseClient()
    return clickhouse_client


async def initialize_clickhouse() -> None:
    """Initialize the global ClickHouse client."""
    client = get_clickhouse_client()
    await client.connect()


async def cleanup_clickhouse() -> None:
    """Cleanup the global ClickHouse client."""
    global clickhouse_client
    if clickhouse_client:
        await clickhouse_client.disconnect()
        clickhouse_client = None 