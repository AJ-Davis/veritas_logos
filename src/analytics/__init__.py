"""
Analytics and Metrics Collection Module for Veritas Logos

This module provides comprehensive analytics and metrics collection capabilities
for tracking KPIs, monitoring system performance, and measuring verification
effectiveness including the Adversarial Recall Metric.
"""

from .metrics_collector import MetricsCollector
from .kpi_tracker import KPITracker
from .adversarial_metrics import AdversarialRecallMetric
from .clickhouse_client import ClickHouseClient
from .dashboard_service import DashboardService
from .ab_testing import ABTestingFramework

__all__ = [
    "MetricsCollector",
    "KPITracker", 
    "AdversarialRecallMetric",
    "ClickHouseClient",
    "DashboardService",
    "ABTestingFramework"
] 