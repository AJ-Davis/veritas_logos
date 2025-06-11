"""
Unit tests for Dashboard Data Visualization Structures.

This module tests the dashboard models to ensure proper aggregation,
time-series tracking, comparison capabilities, heatmap generation,
relationship graphs, and export functionality.
"""

import pytest
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any

from src.models.dashboard import (
    MetricType,
    AggregationLevel,
    VisualizationType,
    ExportFormat,
    TimeRange,
    MetricDefinition,
    TimeSeriesDataPoint,
    TimeSeriesData,
    HeatmapCell,
    HeatmapData,
    GraphNode,
    GraphEdge,
    RelationshipGraph,
    ComparisonData,
    DashboardAggregator,
    create_dashboard_visualization,
    create_time_series_from_issues
)
from src.models.verification import (
    VerificationChainResult,
    VerificationResult,
    VerificationStatus,
    VerificationPassType
)
from src.models.issues import (
    UnifiedIssue,
    IssueSeverity,
    IssueType,
    IssueStatus,
    IssueLocation,
    IssueMetadata,
    IssueRegistry
)
from src.models.document import DocumentSection, ParsedDocument, DocumentMetadata, DocumentFormat, ExtractionMethod
from src.models.output import DashboardDataPoint, OutputVerificationResult


class TestTimeSeriesData:
    """Test time-series data structures and calculations."""
    
    def test_add_data_point(self):
        """Test adding data points to time series."""
        ts = TimeSeriesData(
            name="test_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        now = datetime.now(timezone.utc)
        ts.add_data_point(now, 10, {"source": "test"})
        
        assert len(ts.data_points) == 1
        assert ts.data_points[0].timestamp == now
        assert ts.data_points[0].value == 10
        assert ts.data_points[0].metadata["source"] == "test"
    
    def test_time_range_update(self):
        """Test that time range is updated correctly."""
        ts = TimeSeriesData(
            name="test_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        base_time = datetime.now(timezone.utc)
        ts.add_data_point(base_time + timedelta(hours=2), 30)
        ts.add_data_point(base_time, 10)
        ts.add_data_point(base_time + timedelta(hours=1), 20)
        
        assert ts.time_range is not None
        assert ts.time_range.start == base_time
        assert ts.time_range.end == base_time + timedelta(hours=2)
    
    def test_trend_calculation(self):
        """Test trend calculation for time series."""
        ts = TimeSeriesData(
            name="test_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        base_time = datetime.now(timezone.utc)
        # Add increasing values
        for i in range(5):
            ts.add_data_point(base_time + timedelta(days=i), (i + 1) * 10)
        
        trend = ts.get_trend()
        assert trend == "increasing"
    
    def test_statistics_calculation(self):
        """Test statistics calculation for time series."""
        ts = TimeSeriesData(
            name="test_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        values = [10, 20, 30, 40, 50]
        base_time = datetime.now(timezone.utc)
        
        for i, value in enumerate(values):
            ts.add_data_point(base_time + timedelta(days=i), value)
        
        stats = ts.get_statistics()
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30
        assert stats["count"] == 5


class TestHeatmapData:
    """Test heatmap data structures and calculations."""
    
    def test_add_cell(self):
        """Test adding cells to heatmap."""
        heatmap = HeatmapData(
            title="Test Heatmap",
            x_axis_label="X Axis",
            y_axis_label="Y Axis"
        )
        
        heatmap.add_cell("section1", "type1", 10.0, test="value")
        
        assert len(heatmap.cells) == 1
        cell = heatmap.cells[0]
        assert cell.x_coordinate == "section1"
        assert cell.y_coordinate == "type1"
        assert cell.value == 10.0
        assert cell.metadata["test"] == "value"
    
    def test_value_range_tracking(self):
        """Test that min/max values are tracked correctly."""
        heatmap = HeatmapData(
            title="Test Heatmap",
            x_axis_label="X Axis",
            y_axis_label="Y Axis"
        )
        
        heatmap.add_cell("a", "1", 5.0)
        heatmap.add_cell("b", "2", 15.0)
        heatmap.add_cell("c", "3", 10.0)
        
        assert heatmap.min_value == 5.0
        assert heatmap.max_value == 15.0
    
    def test_dimensions_calculation(self):
        """Test heatmap dimensions calculation."""
        heatmap = HeatmapData(
            title="Test Heatmap",
            x_axis_label="X Axis",
            y_axis_label="Y Axis"
        )
        
        heatmap.add_cell("a", "1", 5.0)
        heatmap.add_cell("b", "1", 10.0)
        heatmap.add_cell("a", "2", 15.0)
        
        width, height = heatmap.get_dimensions()
        assert width == 2  # "a", "b"
        assert height == 2  # "1", "2"


class TestRelationshipGraph:
    """Test relationship graph data structures."""
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = RelationshipGraph(title="Test Graph")
        
        graph.add_node("node1", "Node 1", "issue", weight=1.5, severity="high")
        
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert node.node_id == "node1"
        assert node.label == "Node 1"
        assert node.node_type == "issue"
        assert node.weight == 1.5
        assert node.metadata["severity"] == "high"
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = RelationshipGraph(title="Test Graph")
        
        graph.add_node("node1", "Node 1", "issue")
        graph.add_node("node2", "Node 2", "issue")
        graph.add_edge("node1", "node2", "related", weight=0.8, relationship="causes")
        
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.source_node == "node1"
        assert edge.target_node == "node2"
        assert edge.edge_type == "related"
        assert edge.weight == 0.8
        assert edge.metadata["relationship"] == "causes"
    
    def test_node_degree_calculation(self):
        """Test node degree calculation."""
        graph = RelationshipGraph(title="Test Graph")
        
        graph.add_node("node1", "Node 1", "issue")
        graph.add_node("node2", "Node 2", "issue")
        graph.add_node("node3", "Node 3", "issue")
        
        graph.add_edge("node1", "node2", "related")
        graph.add_edge("node1", "node3", "related")
        
        assert graph.get_node_degree("node1") == 2
        assert graph.get_node_degree("node2") == 1
        assert graph.get_node_degree("node3") == 1


class TestComparisonData:
    """Test comparison data structures."""
    
    def test_add_metric_comparison(self):
        """Test adding metric comparisons."""
        comparison = ComparisonData(
            title="Test Comparison",
            baseline_label="Version A",
            comparison_label="Version B"
        )
        
        comparison.add_metric_comparison(
            "accuracy", 0.8, 0.9, "%", higher_is_better=True
        )
        
        assert "accuracy" in comparison.metric_comparisons
        metric_data = comparison.metric_comparisons["accuracy"]
        assert metric_data["baseline_value"] == 0.8
        assert metric_data["comparison_value"] == 0.9
        # Use approximate comparison for floating point
        assert abs(metric_data["change"] - 0.1) < 0.0001
    
    def test_overall_score_calculation(self):
        """Test overall comparison score calculation."""
        comparison = ComparisonData(
            title="Test Comparison",
            baseline_label="Version A",
            comparison_label="Version B"
        )
        
        comparison.add_metric_comparison("metric1", 0.8, 0.9, "%", higher_is_better=True)
        comparison.add_metric_comparison("metric2", 0.7, 0.6, "%", higher_is_better=True)
        
        score = comparison.get_overall_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestDashboardAggregator:
    """Test dashboard aggregator functionality."""
    
    def create_test_issue(self, severity: IssueSeverity, issue_type: IssueType) -> UnifiedIssue:
        """Create a test issue with all required fields."""
        return UnifiedIssue(
            issue_id=str(uuid.uuid4()),
            title="Test Issue",
            description="Test description",
            issue_type=issue_type,
            severity=severity,
            severity_score=0.7,  # Required field
            confidence_score=0.8,  # Required field
            impact_score=0.6,  # Required field
            status=IssueStatus.DETECTED,
            location=IssueLocation(start_position=0, end_position=10),
            text_excerpt="test text",
            metadata=IssueMetadata(detected_by=VerificationPassType.LOGIC_ANALYSIS)
        )
    
    def create_test_output_result(self, timestamp: datetime, 
                                 issues: List[UnifiedIssue]) -> OutputVerificationResult:
        """Create a test output verification result."""
        # Create a mock chain result
        chain_result = VerificationChainResult(
            chain_id="test_chain",
            document_id="test_doc",
            status=VerificationStatus.COMPLETED,
            started_at=timestamp,
            total_execution_time_seconds=10.0
        )
        
        # Create issue registry with required document_id
        issue_registry = IssueRegistry(document_id="test_doc")
        for issue in issues:
            issue_registry.add_issue(issue)
        
        # Create a mock document
        document = ParsedDocument(
            content="Test document content",
            sections=[
                DocumentSection(
                    content="Test section",
                    section_type="paragraph",
                    page_number=1,
                    position=0
                )
            ],
            metadata=DocumentMetadata(
                filename="test_doc.txt",
                file_size_bytes=100,
                format=DocumentFormat.TXT,
                extraction_method=ExtractionMethod.DIRECT
            )
        )
        
        return OutputVerificationResult(
            chain_result=chain_result,
            issue_registry=issue_registry,
            document=document,
            generated_at=timestamp
        )
    
    def test_aggregate_verification_metrics(self):
        """Test verification metrics aggregation."""
        aggregator = DashboardAggregator()
        
        base_time = datetime.now(timezone.utc)
        results = []
        
        # Create test results
        for i in range(5):
            issues = [
                self.create_test_issue(IssueSeverity.HIGH, IssueType.LOGICAL_FALLACY),
                self.create_test_issue(IssueSeverity.LOW, IssueType.BIAS_DETECTION)
            ]
            result = self.create_test_output_result(
                base_time + timedelta(days=i), issues
            )
            results.append(result)
        
        metrics = aggregator.aggregate_verification_metrics(
            results, aggregation_level=AggregationLevel.DAILY
        )
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check that we have expected metrics
        assert "total_documents" in metrics
        assert "issue_detection_rate" in metrics
    
    def test_create_issue_severity_heatmap(self):
        """Test issue severity heatmap creation."""
        aggregator = DashboardAggregator()
        
        # Create test results with issues
        issues = [
            self.create_test_issue(IssueSeverity.HIGH, IssueType.LOGICAL_FALLACY),
            self.create_test_issue(IssueSeverity.LOW, IssueType.BIAS_DETECTION),
            self.create_test_issue(IssueSeverity.MEDIUM, IssueType.LOGICAL_FALLACY)
        ]
        
        base_time = datetime.now(timezone.utc)
        result = self.create_test_output_result(base_time, issues)
        
        heatmap = aggregator.create_issue_severity_heatmap([result])
        
        assert isinstance(heatmap, HeatmapData)
        assert heatmap.title.startswith("Issue Density:")
        assert len(heatmap.cells) > 0
    
    def test_create_document_comparison(self):
        """Test document comparison creation."""
        aggregator = DashboardAggregator()
        
        base_time = datetime.now(timezone.utc)
        
        # Create baseline result
        baseline_issues = [
            self.create_test_issue(IssueSeverity.HIGH, IssueType.LOGICAL_FALLACY)
        ]
        baseline_result = self.create_test_output_result(base_time, baseline_issues)
        
        # Create comparison result
        comparison_issues = [
            self.create_test_issue(IssueSeverity.HIGH, IssueType.LOGICAL_FALLACY),
            self.create_test_issue(IssueSeverity.MEDIUM, IssueType.BIAS_DETECTION)
        ]
        comparison_result = self.create_test_output_result(base_time, comparison_issues)
        
        comparison = aggregator.create_document_comparison(
            baseline_result, comparison_result
        )
        
        assert isinstance(comparison, ComparisonData)
        assert comparison.title.startswith("Document Comparison:")
        assert len(comparison.metric_comparisons) > 0


class TestVisualizationFactories:
    """Test visualization factory functions."""
    
    def test_create_dashboard_visualization_from_time_series(self):
        """Test creating dashboard visualization from time series data."""
        ts = TimeSeriesData(
            name="test_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        base_time = datetime.now(timezone.utc)
        for i in range(5):
            ts.add_data_point(base_time + timedelta(days=i), (i + 1) * 10)
        
        viz = create_dashboard_visualization(
            ts, "Test Visualization", VisualizationType.LINE_CHART,
            description="Test description"
        )
        
        assert viz.title == "Test Visualization"
        assert viz.visualization_type == VisualizationType.LINE_CHART.value
        assert viz.description == "Test description"
        assert len(viz.data_points) == 5
        assert "trend" in viz.summary_metrics
    
    def test_create_time_series_from_issues(self):
        """Test creating time series from issue registries."""
        registries = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(3):
            registry = IssueRegistry(document_id=f"test_doc_{i}")
            # Add issues with timestamps
            for j in range(i + 1):  # Increasing number of issues
                issue = UnifiedIssue(
                    issue_id=f"issue_{i}_{j}",
                    title="Test Issue",
                    description="Test description",
                    issue_type=IssueType.LOGICAL_FALLACY,
                    severity=IssueSeverity.MEDIUM,
                    severity_score=0.5,
                    confidence_score=0.8,
                    impact_score=0.6,
                    status=IssueStatus.DETECTED,
                    location=IssueLocation(start_position=0, end_position=10),
                    text_excerpt="test text",
                    metadata=IssueMetadata(
                        detected_by=VerificationPassType.LOGIC_ANALYSIS,
                        detection_timestamp=base_time + timedelta(days=i)
                    )
                )
                registry.add_issue(issue)
            registries.append(registry)
        
        ts = create_time_series_from_issues(registries)
        
        assert isinstance(ts, TimeSeriesData)
        assert ts.name == "issue_count"
        assert ts.metric_type == MetricType.COUNT
        assert len(ts.data_points) > 0


class TestTimeRange:
    """Test time range utilities."""
    
    def test_last_n_days(self):
        """Test creating time range for last N days."""
        time_range = TimeRange.last_n_days(7)
        
        now = datetime.now(timezone.utc)
        assert time_range.end <= now
        assert (time_range.end - time_range.start).days == 7
    
    def test_current_month(self):
        """Test creating time range for current month."""
        time_range = TimeRange.current_month()
        
        now = datetime.now(timezone.utc)
        assert time_range.start.day == 1
        assert time_range.start.month == now.month
        assert time_range.start.year == now.year


class TestExportFunctionality:
    """Test data export functionality."""
    
    def test_export_time_series_to_json(self):
        """Test exporting time series to JSON."""
        aggregator = DashboardAggregator()
        
        ts = TimeSeriesData(
            name="test_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        base_time = datetime.now(timezone.utc)
        for i in range(3):
            ts.add_data_point(base_time + timedelta(days=i), (i + 1) * 10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = str(Path(temp_dir) / "test_export")
            exported_path = aggregator.export_data(ts, ExportFormat.JSON, filename)
            
            assert Path(exported_path).exists()
            assert exported_path.endswith('.json')


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_time_series_statistics(self):
        """Test statistics calculation with empty time series."""
        ts = TimeSeriesData(
            name="empty_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        stats = ts.get_statistics()
        assert stats == {}
    
    def test_empty_time_series_trend(self):
        """Test trend calculation with empty or single-point time series."""
        ts = TimeSeriesData(
            name="trend_metric",
            metric_type=MetricType.COUNT,
            aggregation_level=AggregationLevel.DAILY
        )
        
        # Empty series
        assert ts.get_trend() == "stable"
        
        # Single point
        ts.add_data_point(datetime.now(timezone.utc), 10)
        assert ts.get_trend() == "stable"
    
    def test_heatmap_with_no_cells(self):
        """Test heatmap operations with no cells."""
        heatmap = HeatmapData(
            title="Empty Heatmap",
            x_axis_label="X",
            y_axis_label="Y"
        )
        
        width, height = heatmap.get_dimensions()
        assert width == 0
        assert height == 0
        assert heatmap.min_value is None
        assert heatmap.max_value is None 