"""
Dashboard Data Visualization Structures for the Veritas Logos verification system.

This module provides comprehensive data structures and aggregation methods
specifically designed for dashboard visualization, analytics, and reporting.

It builds upon the existing verification and output models while providing
dashboard-specific features like time-series analysis, heatmaps, comparisons,
and export functionality.
"""

import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Literal
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from pydantic import BaseModel, Field

from src.verification import VerificationChainResult, VerificationResult, VerificationStatus, VerificationPassType
from src.issues import UnifiedIssue, IssueSeverity, IssueType, IssueRegistry
from src.output import DashboardDataPoint, DashboardVisualizationData, OutputVerificationResult
from src.acvf import ACVFResult
from src.verification.pipeline.aggregators import AggregatedScore


class MetricType(str, Enum):
    """Types of metrics for dashboard visualization."""
    COUNT = "count"
    PERCENTAGE = "percentage"
    SCORE = "score"
    DURATION = "duration"
    RATE = "rate"
    RATIO = "ratio"
    TREND = "trend"
    DISTRIBUTION = "distribution"


class AggregationLevel(str, Enum):
    """Time aggregation levels for metrics."""
    HOURLY = "hourly"
    DAILY = "daily" 
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class VisualizationType(str, Enum):
    """Types of dashboard visualizations."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    TABLE = "table"
    METRIC_CARD = "metric_card"
    TREEMAP = "treemap"
    NETWORK_GRAPH = "network_graph"


class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PDF = "pdf"
    PNG = "png"
    SVG = "svg"


@dataclass
class TimeRange:
    """Time range for data filtering."""
    start: datetime
    end: datetime
    
    @classmethod
    def last_n_days(cls, days: int) -> 'TimeRange':
        """Create time range for the last N days."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        return cls(start=start, end=end)
    
    @classmethod
    def current_month(cls) -> 'TimeRange':
        """Create time range for current month."""
        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            end = start.replace(year=now.year + 1, month=1) - timedelta(seconds=1)
        else:
            end = start.replace(month=now.month + 1) - timedelta(seconds=1)
        return cls(start=start, end=end)


class MetricDefinition(BaseModel):
    """Definition of a dashboard metric."""
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    metric_type: MetricType
    unit: Optional[str] = None
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    higher_is_better: bool = True
    calculation_method: str = Field(description="How the metric is calculated")
    dependencies: List[str] = Field(default_factory=list)


class TimeSeriesDataPoint(BaseModel):
    """A single data point in a time series."""
    timestamp: datetime
    value: Union[int, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def is_numeric(self) -> bool:
        """Check if value is numeric."""
        return isinstance(self.value, (int, float))


class TimeSeriesData(BaseModel):
    """Time series data structure."""
    series_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    metric_type: MetricType
    aggregation_level: AggregationLevel
    data_points: List[TimeSeriesDataPoint] = Field(default_factory=list)
    time_range: Optional[TimeRange] = None
    
    def add_data_point(self, timestamp: datetime, value: Union[int, float], 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a data point to the series."""
        point = TimeSeriesDataPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata or {}
        )
        self.data_points.append(point)
        
        # Update time range
        if not self.time_range:
            self.time_range = TimeRange(start=timestamp, end=timestamp)
        else:
            if timestamp < self.time_range.start:
                self.time_range.start = timestamp
            if timestamp > self.time_range.end:
                self.time_range.end = timestamp
    
    def get_trend(self) -> Literal["increasing", "decreasing", "stable"]:
        """Calculate trend direction."""
        if len(self.data_points) < 2:
            return "stable"
        
        numeric_points = [p for p in self.data_points if p.is_numeric]
        if len(numeric_points) < 2:
            return "stable"
        
        # Simple trend calculation using first and last values
        first_value = numeric_points[0].value
        last_value = numeric_points[-1].value
        
        change_percent = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of the series."""
        numeric_values = [p.value for p in self.data_points if p.is_numeric]
        
        if not numeric_values:
            return {}
        
        return {
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": sum(numeric_values) / len(numeric_values),
            "count": len(numeric_values),
            "std_dev": np.std(numeric_values) if len(numeric_values) > 1 else 0.0
        }


class HeatmapCell(BaseModel):
    """A single cell in a heatmap."""
    x_coordinate: Union[str, int]
    y_coordinate: Union[str, int]
    value: float
    intensity: float = Field(ge=0.0, le=1.0, description="Normalized intensity (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def create(cls, x: Union[str, int], y: Union[str, int], value: float,
              min_value: float, max_value: float, **metadata) -> 'HeatmapCell':
        """Create a heatmap cell with calculated intensity."""
        if max_value == min_value:
            intensity = 0.5
        else:
            intensity = (value - min_value) / (max_value - min_value)
        
        return cls(
            x_coordinate=x,
            y_coordinate=y,
            value=value,
            intensity=intensity,
            metadata=metadata
        )


class HeatmapData(BaseModel):
    """Heatmap visualization data structure."""
    heatmap_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    x_axis_label: str
    y_axis_label: str
    cells: List[HeatmapCell] = Field(default_factory=list)
    
    # Value range information
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Color scheme
    color_scheme: str = Field(default="viridis", description="Color scheme for heatmap")
    
    def add_cell(self, x: Union[str, int], y: Union[str, int], value: float,
                **metadata) -> None:
        """Add a cell to the heatmap."""
        # Update min/max values
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value
        
        # Create cell with calculated intensity
        cell = HeatmapCell.create(
            x=x, y=y, value=value,
            min_value=self.min_value, max_value=self.max_value,
            **metadata
        )
        self.cells.append(cell)
        
        # Recalculate intensities for all cells
        self._recalculate_intensities()
    
    def _recalculate_intensities(self) -> None:
        """Recalculate intensities for all cells."""
        if self.min_value is None or self.max_value is None:
            return
        
        for cell in self.cells:
            if self.max_value == self.min_value:
                cell.intensity = 0.5
            else:
                cell.intensity = (cell.value - self.min_value) / (self.max_value - self.min_value)
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get dimensions of the heatmap (width, height)."""
        if not self.cells:
            return (0, 0)
        
        x_values = set(str(cell.x_coordinate) for cell in self.cells)
        y_values = set(str(cell.y_coordinate) for cell in self.cells)
        
        return (len(x_values), len(y_values))


class GraphNode(BaseModel):
    """Node in a relationship graph."""
    node_id: str
    label: str
    node_type: str
    weight: float = Field(default=1.0, description="Node importance/weight")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Visual properties
    color: Optional[str] = None
    size: Optional[float] = None
    position: Optional[Tuple[float, float]] = None


class GraphEdge(BaseModel):
    """Edge in a relationship graph."""
    source_node: str
    target_node: str
    edge_type: str
    weight: float = Field(default=1.0, description="Connection strength")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Visual properties
    color: Optional[str] = None
    thickness: Optional[float] = None


class RelationshipGraph(BaseModel):
    """Relationship graph data structure."""
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    layout_algorithm: str = Field(default="force_directed")
    
    def add_node(self, node_id: str, label: str, node_type: str,
                weight: float = 1.0, **metadata) -> None:
        """Add a node to the graph."""
        node = GraphNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            weight=weight,
            metadata=metadata
        )
        self.nodes.append(node)
    
    def add_edge(self, source: str, target: str, edge_type: str,
                weight: float = 1.0, **metadata) -> None:
        """Add an edge to the graph."""
        edge = GraphEdge(
            source_node=source,
            target_node=target,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata
        )
        self.edges.append(edge)
    
    def get_node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) for a node."""
        return sum(1 for edge in self.edges 
                  if edge.source_node == node_id or edge.target_node == node_id)
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get all nodes connected to the given node."""
        connected = set()
        for edge in self.edges:
            if edge.source_node == node_id:
                connected.add(edge.target_node)
            elif edge.target_node == node_id:
                connected.add(edge.source_node)
        return list(connected)


class ComparisonData(BaseModel):
    """Data structure for comparing document versions or metrics."""
    comparison_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    baseline_label: str
    comparison_label: str
    
    # Metric comparisons
    metric_comparisons: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Overall summary
    improvement_areas: List[str] = Field(default_factory=list)
    regression_areas: List[str] = Field(default_factory=list)
    overall_change_percent: Optional[float] = None
    
    def add_metric_comparison(self, metric_name: str, baseline_value: float,
                            comparison_value: float, unit: str = "",
                            higher_is_better: bool = True) -> None:
        """Add a metric comparison."""
        change = comparison_value - baseline_value
        change_percent = (change / baseline_value * 100) if baseline_value != 0 else 0
        
        is_improvement = (change > 0 and higher_is_better) or (change < 0 and not higher_is_better)
        
        self.metric_comparisons[metric_name] = {
            "baseline_value": baseline_value,
            "comparison_value": comparison_value,
            "change": change,
            "change_percent": change_percent,
            "unit": unit,
            "is_improvement": is_improvement,
            "higher_is_better": higher_is_better
        }
        
        # Update improvement/regression tracking
        if is_improvement and metric_name not in self.improvement_areas:
            self.improvement_areas.append(metric_name)
        elif not is_improvement and change != 0 and metric_name not in self.regression_areas:
            self.regression_areas.append(metric_name)
    
    def get_overall_score(self) -> float:
        """Calculate overall comparison score."""
        if not self.metric_comparisons:
            return 0.0
        
        improvement_count = len(self.improvement_areas)
        regression_count = len(self.regression_areas)
        total_metrics = len(self.metric_comparisons)
        
        # Simple scoring: (improvements - regressions) / total
        return (improvement_count - regression_count) / total_metrics if total_metrics > 0 else 0.0


class DashboardAggregator:
    """Main aggregator for creating dashboard visualizations from verification data."""
    
    def __init__(self):
        """Initialize the dashboard aggregator."""
        self.metric_definitions = self._initialize_metric_definitions()
    
    def aggregate_verification_metrics(
        self,
        results: List[OutputVerificationResult],
        time_range: Optional[TimeRange] = None,
        aggregation_level: AggregationLevel = AggregationLevel.DAILY
    ) -> Dict[str, TimeSeriesData]:
        """
        Aggregate verification results into time-series metrics.
        
        Args:
            results: List of verification results to aggregate
            time_range: Optional time range filter
            aggregation_level: Time aggregation level
            
        Returns:
            Dictionary of metric name to time series data
        """
        metrics = {}
        
        # Filter results by time range if specified
        if time_range:
            results = [r for r in results if time_range.start <= r.generated_at <= time_range.end]
        
        # Group results by time bucket
        time_buckets = self._group_by_time_bucket(results, aggregation_level)
        
        for bucket_time, bucket_results in time_buckets.items():
            # Calculate metrics for this time bucket
            bucket_metrics = self._calculate_bucket_metrics(bucket_results)
            
            # Add to time series
            for metric_name, value in bucket_metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = TimeSeriesData(
                        name=metric_name,
                        metric_type=self._get_metric_type(metric_name),
                        aggregation_level=aggregation_level
                    )
                
                metrics[metric_name].add_data_point(
                    timestamp=bucket_time,
                    value=value,
                    metadata={"bucket_size": len(bucket_results)}
                )
        
        return metrics
    
    def create_issue_severity_heatmap(
        self,
        results: List[OutputVerificationResult],
        x_axis: str = "document_section",
        y_axis: str = "issue_type"
    ) -> HeatmapData:
        """
        Create a heatmap showing issue density by various dimensions.
        
        Args:
            results: List of verification results
            x_axis: X-axis dimension (document_section, time_period, etc.)
            y_axis: Y-axis dimension (issue_type, severity, etc.)
            
        Returns:
            Heatmap data structure
        """
        heatmap = HeatmapData(
            title=f"Issue Density: {y_axis} vs {x_axis}",
            x_axis_label=x_axis.replace("_", " ").title(),
            y_axis_label=y_axis.replace("_", " ").title()
        )
        
        # Count issues by the specified dimensions
        issue_counts = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            for issue in result.issue_registry.issues:
                x_value = self._extract_dimension_value(issue, result, x_axis)
                y_value = self._extract_dimension_value(issue, result, y_axis)
                
                issue_counts[x_value][y_value] += 1
        
        # Add cells to heatmap
        for x_value, y_counts in issue_counts.items():
            for y_value, count in y_counts.items():
                heatmap.add_cell(
                    x=x_value,
                    y=y_value,
                    value=count,
                    issue_count=count
                )
        
        return heatmap
    
    def create_issue_relationship_graph(
        self,
        results: List[OutputVerificationResult],
        include_acvf_connections: bool = True
    ) -> RelationshipGraph:
        """
        Create a graph showing relationships between issues.
        
        Args:
            results: List of verification results
            include_acvf_connections: Whether to include ACVF debate connections
            
        Returns:
            Relationship graph data structure
        """
        graph = RelationshipGraph(title="Issue Relationship Network")
        
        # Track added nodes to avoid duplicates
        added_nodes = set()
        
        for result in results:
            # Add issue nodes
            for issue in result.issue_registry.issues:
                if issue.issue_id not in added_nodes:
                    graph.add_node(
                        node_id=issue.issue_id,
                        label=issue.title[:50] + "..." if len(issue.title) > 50 else issue.title,
                        node_type="issue",
                        weight=issue.calculate_priority_score(),
                        severity=issue.severity.value,
                        issue_type=issue.issue_type.value
                    )
                    added_nodes.add(issue.issue_id)
                
                # Add edges for related issues
                for related_id in issue.metadata.related_issue_ids:
                    if related_id in added_nodes:
                        graph.add_edge(
                            source=issue.issue_id,
                            target=related_id,
                            edge_type="related",
                            weight=0.5
                        )
            
            # Add ACVF connections if requested
            if include_acvf_connections:
                for acvf_result in result.acvf_results:
                    debate_node_id = f"debate_{acvf_result.session_id}"
                    if debate_node_id not in added_nodes:
                        graph.add_node(
                            node_id=debate_node_id,
                            label=f"ACVF Debate: {acvf_result.subject_id}",
                            node_type="debate",
                            weight=1.0,
                            verdict=acvf_result.final_verdict.value if acvf_result.final_verdict else None
                        )
                        added_nodes.add(debate_node_id)
                    
                    # Connect debate to related issue
                    if acvf_result.subject_id in added_nodes:
                        graph.add_edge(
                            source=acvf_result.subject_id,
                            target=debate_node_id,
                            edge_type="acvf_escalation",
                            weight=1.0
                        )
        
        return graph
    
    def create_document_comparison(
        self,
        baseline_result: OutputVerificationResult,
        comparison_result: OutputVerificationResult
    ) -> ComparisonData:
        """
        Create comparison data between two document verification results.
        
        Args:
            baseline_result: Baseline verification result
            comparison_result: Comparison verification result
            
        Returns:
            Comparison data structure
        """
        comparison = ComparisonData(
            title=f"Document Comparison: {baseline_result.document_id} vs {comparison_result.document_id}",
            baseline_label=f"Baseline ({baseline_result.document_id})",
            comparison_label=f"Comparison ({comparison_result.document_id})"
        )
        
        # Compare key metrics
        metrics_to_compare = [
            ("total_issues", "Total Issues", "", False),
            ("critical_issues", "Critical Issues", "", False),
            ("overall_confidence", "Overall Confidence", "%", True),
            ("processing_time", "Processing Time", "s", False),
            ("acvf_debates", "ACVF Debates", "", False)
        ]
        
        for metric_key, metric_name, unit, higher_is_better in metrics_to_compare:
            baseline_value = self._extract_result_metric(baseline_result, metric_key)
            comparison_value = self._extract_result_metric(comparison_result, metric_key)
            
            comparison.add_metric_comparison(
                metric_name=metric_name,
                baseline_value=baseline_value,
                comparison_value=comparison_value,
                unit=unit,
                higher_is_better=higher_is_better
            )
        
        return comparison
    
    def export_data(
        self,
        data: Union[TimeSeriesData, HeatmapData, RelationshipGraph, ComparisonData],
        format: ExportFormat,
        filename: str
    ) -> str:
        """
        Export dashboard data to various formats.
        
        Args:
            data: Data structure to export
            format: Export format
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if isinstance(data, TimeSeriesData):
            return self._export_time_series(data, format, filename)
        elif isinstance(data, HeatmapData):
            return self._export_heatmap(data, format, filename)
        elif isinstance(data, RelationshipGraph):
            return self._export_graph(data, format, filename)
        elif isinstance(data, ComparisonData):
            return self._export_comparison(data, format, filename)
        else:
            raise ValueError(f"Unsupported data type for export: {type(data)}")
    
    def _initialize_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Initialize standard metric definitions."""
        return {
            "total_documents": MetricDefinition(
                name="Total Documents",
                description="Total number of documents processed",
                metric_type=MetricType.COUNT,
                calculation_method="count_documents"
            ),
            "issue_detection_rate": MetricDefinition(
                name="Issue Detection Rate",
                description="Percentage of documents with detected issues",
                metric_type=MetricType.PERCENTAGE,
                unit="%",
                calculation_method="(documents_with_issues / total_documents) * 100"
            ),
            "average_confidence": MetricDefinition(
                name="Average Confidence",
                description="Average verification confidence score",
                metric_type=MetricType.SCORE,
                unit="score",
                target_value=0.8,
                warning_threshold=0.6,
                critical_threshold=0.4,
                calculation_method="mean(confidence_scores)"
            ),
            "acvf_escalation_rate": MetricDefinition(
                name="ACVF Escalation Rate",
                description="Percentage of verifications requiring ACVF",
                metric_type=MetricType.PERCENTAGE,
                unit="%",
                calculation_method="(acvf_escalations / total_verifications) * 100"
            ),
            "processing_time": MetricDefinition(
                name="Average Processing Time",
                description="Average time to complete verification",
                metric_type=MetricType.DURATION,
                unit="seconds",
                higher_is_better=False,
                calculation_method="mean(processing_times)"
            )
        }
    
    def _group_by_time_bucket(
        self,
        results: List[OutputVerificationResult],
        aggregation_level: AggregationLevel
    ) -> Dict[datetime, List[OutputVerificationResult]]:
        """Group results by time buckets based on aggregation level."""
        buckets = defaultdict(list)
        
        for result in results:
            bucket_time = self._get_time_bucket(result.generated_at, aggregation_level)
            buckets[bucket_time].append(result)
        
        return dict(buckets)
    
    def _get_time_bucket(self, timestamp: datetime, level: AggregationLevel) -> datetime:
        """Get the time bucket for a timestamp based on aggregation level."""
        if level == AggregationLevel.HOURLY:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.DAILY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.WEEKLY:
            days_since_monday = timestamp.weekday()
            monday = timestamp - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.MONTHLY:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.QUARTERLY:
            quarter_month = ((timestamp.month - 1) // 3) * 3 + 1
            return timestamp.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif level == AggregationLevel.YEARLY:
            return timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp
    
    def _calculate_bucket_metrics(
        self,
        results: List[OutputVerificationResult]
    ) -> Dict[str, float]:
        """Calculate metrics for a time bucket of results."""
        if not results:
            return {}
        
        metrics = {}
        
        # Basic counts
        metrics["total_documents"] = len(results)
        metrics["documents_with_issues"] = sum(1 for r in results if r.total_issues > 0)
        
        # Issue detection rate
        metrics["issue_detection_rate"] = (metrics["documents_with_issues"] / metrics["total_documents"]) * 100
        
        # Average confidence
        confidences = [r.overall_confidence for r in results if r.overall_confidence is not None]
        metrics["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Processing times
        processing_times = [
            r.chain_result.total_execution_time_seconds 
            for r in results 
            if r.chain_result.total_execution_time_seconds is not None
        ]
        metrics["processing_time"] = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # ACVF escalations
        acvf_escalations = sum(1 for r in results if r.acvf_results)
        metrics["acvf_escalation_rate"] = (acvf_escalations / len(results)) * 100
        
        # Issue severity distribution
        severity_counts = defaultdict(int)
        for result in results:
            for issue in result.issue_registry.issues:
                severity_counts[issue.severity.value] += 1
        
        for severity, count in severity_counts.items():
            metrics[f"issues_{severity}"] = count
        
        return metrics
    
    def _get_metric_type(self, metric_name: str) -> MetricType:
        """Get the metric type for a metric name."""
        if metric_name in self.metric_definitions:
            return self.metric_definitions[metric_name].metric_type
        
        # Default inference
        if "rate" in metric_name or "percentage" in metric_name:
            return MetricType.PERCENTAGE
        elif "time" in metric_name or "duration" in metric_name:
            return MetricType.DURATION
        elif "confidence" in metric_name or "score" in metric_name:
            return MetricType.SCORE
        else:
            return MetricType.COUNT
    
    def _extract_dimension_value(
        self,
        issue: UnifiedIssue,
        result: OutputVerificationResult,
        dimension: str
    ) -> str:
        """Extract a dimension value from an issue or result."""
        if dimension == "issue_type":
            return issue.issue_type.value
        elif dimension == "severity":
            return issue.severity.value
        elif dimension == "document_section":
            return issue.location.section or "unknown"
        elif dimension == "detection_pass":
            return issue.metadata.detected_by.value
        elif dimension == "time_period":
            return result.generated_at.strftime("%Y-%m-%d")
        else:
            return "unknown"
    
    def _extract_result_metric(self, result: OutputVerificationResult, metric_key: str) -> float:
        """Extract a metric value from a verification result."""
        if metric_key == "total_issues":
            return float(result.total_issues)
        elif metric_key == "critical_issues":
            return float(len(result.critical_issues))
        elif metric_key == "overall_confidence":
            return result.overall_confidence or 0.0
        elif metric_key == "processing_time":
            return result.chain_result.total_execution_time_seconds or 0.0
        elif metric_key == "acvf_debates":
            return float(len(result.acvf_results))
        else:
            return 0.0
    
    def _export_time_series(self, data: TimeSeriesData, format: ExportFormat, filename: str) -> str:
        """Export time series data."""
        if format == ExportFormat.CSV:
            df = pd.DataFrame([
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                    **point.metadata
                }
                for point in data.data_points
            ])
            filepath = f"{filename}.csv"
            df.to_csv(filepath, index=False)
            return filepath
        
        elif format == ExportFormat.JSON:
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                f.write(data.model_dump_json(indent=2))
            return filepath
        
        else:
            raise ValueError(f"Unsupported export format for time series: {format}")
    
    def _export_heatmap(self, data: HeatmapData, format: ExportFormat, filename: str) -> str:
        """Export heatmap data."""
        if format == ExportFormat.CSV:
            df = pd.DataFrame([
                {
                    "x": cell.x_coordinate,
                    "y": cell.y_coordinate,
                    "value": cell.value,
                    "intensity": cell.intensity,
                    **cell.metadata
                }
                for cell in data.cells
            ])
            filepath = f"{filename}.csv"
            df.to_csv(filepath, index=False)
            return filepath
        
        elif format == ExportFormat.JSON:
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                f.write(data.model_dump_json(indent=2))
            return filepath
        
        else:
            raise ValueError(f"Unsupported export format for heatmap: {format}")
    
    def _export_graph(self, data: RelationshipGraph, format: ExportFormat, filename: str) -> str:
        """Export graph data."""
        if format == ExportFormat.JSON:
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                f.write(data.model_dump_json(indent=2))
            return filepath
        
        elif format == ExportFormat.CSV:
            # Export as two CSV files: nodes and edges
            nodes_df = pd.DataFrame([
                {
                    "node_id": node.node_id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "weight": node.weight,
                    **node.metadata
                }
                for node in data.nodes
            ])
            edges_df = pd.DataFrame([
                {
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                    **edge.metadata
                }
                for edge in data.edges
            ])
            
            nodes_filepath = f"{filename}_nodes.csv"
            edges_filepath = f"{filename}_edges.csv"
            nodes_df.to_csv(nodes_filepath, index=False)
            edges_df.to_csv(edges_filepath, index=False)
            
            return f"Exported to {nodes_filepath} and {edges_filepath}"
        
        else:
            raise ValueError(f"Unsupported export format for graph: {format}")
    
    def _export_comparison(self, data: ComparisonData, format: ExportFormat, filename: str) -> str:
        """Export comparison data."""
        if format == ExportFormat.CSV:
            df = pd.DataFrame([
                {
                    "metric": metric_name,
                    "baseline_value": comparison["baseline_value"],
                    "comparison_value": comparison["comparison_value"],
                    "change": comparison["change"],
                    "change_percent": comparison["change_percent"],
                    "unit": comparison["unit"],
                    "is_improvement": comparison["is_improvement"]
                }
                for metric_name, comparison in data.metric_comparisons.items()
            ])
            filepath = f"{filename}.csv"
            df.to_csv(filepath, index=False)
            return filepath
        
        elif format == ExportFormat.JSON:
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                f.write(data.model_dump_json(indent=2))
            return filepath
        
        else:
            raise ValueError(f"Unsupported export format for comparison: {format}")


# Factory functions for easy creation

def create_dashboard_visualization(
    data: Union[TimeSeriesData, HeatmapData, RelationshipGraph],
    title: str,
    visualization_type: VisualizationType,
    **config_options
) -> DashboardVisualizationData:
    """Create a DashboardVisualizationData from dashboard data structures."""
    viz = DashboardVisualizationData(
        title=title,
        visualization_type=visualization_type.value
    )
    
    if isinstance(data, TimeSeriesData):
        # Convert time series to data points
        for point in data.data_points:
            dashboard_point = DashboardDataPoint(
                timestamp=point.timestamp,
                metric_name=data.name,
                metric_value=point.value,
                metric_type=data.metric_type.value,
                metadata=point.metadata
            )
            viz.add_data_point(dashboard_point)
        
        # Add summary metrics
        stats = data.get_statistics()
        viz.summary_metrics.update(stats)
        viz.summary_metrics["trend"] = data.get_trend()
    
    elif isinstance(data, HeatmapData):
        # Convert heatmap to data points
        for cell in data.cells:
            dashboard_point = DashboardDataPoint(
                timestamp=datetime.now(timezone.utc),  # Heatmaps are typically snapshot data
                metric_name=f"{cell.x_coordinate}_{cell.y_coordinate}",
                metric_value=cell.value,
                metric_type=MetricType.COUNT.value,
                category=str(cell.x_coordinate),
                subcategory=str(cell.y_coordinate),
                metadata={"intensity": cell.intensity, **cell.metadata}
            )
            viz.add_data_point(dashboard_point)
        
        # Add summary metrics
        viz.summary_metrics["total_cells"] = len(data.cells)
        viz.summary_metrics["min_value"] = data.min_value or 0
        viz.summary_metrics["max_value"] = data.max_value or 0
    
    # Apply additional config options
    for key, value in config_options.items():
        if hasattr(viz, key):
            setattr(viz, key, value)
    
    return viz


def create_time_series_from_issues(
    issue_registries: List[IssueRegistry],
    metric_name: str = "issue_count",
    aggregation_level: AggregationLevel = AggregationLevel.DAILY
) -> TimeSeriesData:
    """Create time series data from issue registries."""
    time_series = TimeSeriesData(
        name=metric_name,
        metric_type=MetricType.COUNT,
        aggregation_level=aggregation_level
    )
    
    # Group issues by time bucket
    time_buckets = defaultdict(int)
    
    for registry in issue_registries:
        for issue in registry.issues:
            bucket_time = DashboardAggregator()._get_time_bucket(
                issue.metadata.detection_timestamp,
                aggregation_level
            )
            time_buckets[bucket_time] += 1
    
    # Add data points
    for timestamp, count in sorted(time_buckets.items()):
        time_series.add_data_point(
            timestamp=timestamp,
            value=count,
            metadata={"aggregation_level": aggregation_level.value}
        )
    
    return time_series 