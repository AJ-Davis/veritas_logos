"""
Output data structures for the Veritas Logos verification system.

This module provides data structures specifically designed for generating
actionable outputs including annotated documents, API responses, and dashboard
visualizations. It builds upon the existing verification models while adding
output-specific functionality.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass

from .verification import VerificationChainResult, VerificationResult
from .issues import UnifiedIssue, IssueRegistry, IssueSeverity
from .acvf import ACVFResult, DebateRound, DebateArgument, JudgeVerdict
from .document import ParsedDocument


class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    EXCEL = "excel"


class AnnotationType(str, Enum):
    """Types of annotations that can be applied to documents."""
    HIGHLIGHT = "highlight"
    COMMENT = "comment"
    STRIKETHROUGH = "strikethrough"
    UNDERLINE = "underline"
    SIDEBAR_NOTE = "sidebar_note"
    TOOLTIP = "tooltip"
    LINK = "link"


class HighlightStyle(str, Enum):
    """Predefined highlight styles based on issue severity."""
    CRITICAL = "critical"      # Red background, white text
    HIGH = "high"             # Orange background, black text
    MEDIUM = "medium"         # Yellow background, black text
    LOW = "low"               # Light blue background, black text
    INFO = "info"             # Light gray background, black text


@dataclass
class ColorScheme:
    """Color scheme for annotations."""
    critical: str = "#FF4444"    # Red
    high: str = "#FF8C00"        # Orange
    medium: str = "#FFD700"      # Gold
    low: str = "#87CEEB"         # Sky blue
    info: str = "#E6E6FA"        # Lavender
    text_light: str = "#FFFFFF"  # White
    text_dark: str = "#000000"   # Black


class DocumentAnnotation(BaseModel):
    """
    Represents a single annotation within a document.
    
    Annotations can be highlights, comments, links, or other markup
    applied to specific text ranges within a document.
    """
    annotation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    annotation_type: AnnotationType
    
    # Location within document
    start_position: int = Field(ge=0, description="Character start position")
    end_position: int = Field(ge=0, description="Character end position") 
    text_excerpt: str = Field(description="The text being annotated")
    
    # Annotation content
    title: Optional[str] = Field(None, description="Annotation title")
    content: str = Field(description="Annotation content/comment")
    
    # Styling
    style: HighlightStyle = Field(description="Visual style for the annotation")
    custom_color: Optional[str] = Field(None, description="Custom color override")
    
    # Related data
    related_issue_id: Optional[str] = Field(None, description="ID of related issue")
    related_debate_id: Optional[str] = Field(None, description="ID of related ACVF debate")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(default="system", description="Who/what created this annotation")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def length(self) -> int:
        """Get the length of the annotated text."""
        return self.end_position - self.start_position
    
    def overlaps_with(self, other: 'DocumentAnnotation') -> bool:
        """Check if this annotation overlaps with another."""
        return not (self.end_position <= other.start_position or 
                   self.start_position >= other.end_position)
    
    def contains_position(self, position: int) -> bool:
        """Check if a position is within this annotation."""
        return self.start_position <= position < self.end_position


class AnnotationLayer(BaseModel):
    """
    A layer of annotations for a document.
    
    Different types of annotations (highlights, comments, etc.) can be 
    organized into separate layers for better management and rendering.
    """
    layer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    layer_name: str = Field(description="Human-readable layer name")
    layer_type: AnnotationType = Field(description="Type of annotations in this layer")
    annotations: List[DocumentAnnotation] = Field(default_factory=list)
    visible: bool = Field(True, description="Whether this layer is visible")
    z_index: int = Field(0, description="Rendering order (higher = on top)")
    
    def add_annotation(self, annotation: DocumentAnnotation) -> None:
        """Add an annotation to this layer."""
        if annotation.annotation_type != self.layer_type:
            raise ValueError(f"Annotation type {annotation.annotation_type} doesn't match layer type {self.layer_type}")
        self.annotations.append(annotation)
    
    def get_annotations_at_position(self, position: int) -> List[DocumentAnnotation]:
        """Get all annotations that contain the given position."""
        return [ann for ann in self.annotations if ann.contains_position(position)]
    
    def get_overlapping_annotations(self, start: int, end: int) -> List[DocumentAnnotation]:
        """Get all annotations that overlap with the given range."""
        return [ann for ann in self.annotations 
                if not (ann.end_position <= start or ann.start_position >= end)]


class DebateEntryOutput(BaseModel):
    """
    Output-formatted representation of a debate entry from ACVF.
    
    This extends the basic DebateArgument with output-specific formatting
    and presentation options for displaying debates in various formats.
    """
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core debate information
    debate_round_id: str
    participant_role: str = Field(description="challenger, defender, or judge")
    participant_model: str = Field(description="Model name/identifier")
    
    # Argument content
    argument_text: str
    position: str = Field(description="The position taken in this argument")
    evidence: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    
    # Scoring and confidence
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    strength_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Timing
    timestamp: datetime
    
    # Output formatting
    summary: Optional[str] = Field(None, description="Brief summary for display")
    formatted_content: Optional[str] = Field(None, description="Pre-formatted content for output")
    
    # Context and relationships
    responds_to: Optional[str] = Field(None, description="ID of argument this responds to")
    related_document_sections: List[str] = Field(default_factory=list)
    
    @classmethod
    def from_debate_argument(cls, argument: DebateArgument, round_id: str, 
                           model_name: str) -> 'DebateEntryOutput':
        """Convert a DebateArgument to output format."""
        return cls(
            debate_round_id=round_id,
            participant_role=argument.role.value,
            participant_model=model_name,
            argument_text=argument.content,
            position=argument.position or "N/A",
            evidence=argument.evidence,
            reasoning=argument.reasoning,
            confidence_score=argument.confidence_score,
            timestamp=argument.timestamp,
            responds_to=argument.responds_to
        )


class DebateViewOutput(BaseModel):
    """
    Complete debate view output for displaying ACVF debates.
    
    Provides a structured representation of entire debates with
    proper threading, formatting, and navigation support.
    """
    debate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    subject_type: str
    subject_id: str
    subject_content: str
    
    # Debate structure
    rounds: List[Dict[str, Any]] = Field(default_factory=list)
    entries: List[DebateEntryOutput] = Field(default_factory=list)
    
    # Results
    final_verdict: Optional[JudgeVerdict] = None
    final_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    consensus_achieved: bool = False
    
    # Metadata
    total_rounds: int = Field(0, ge=0)
    total_arguments: int = Field(0, ge=0)
    debate_duration_seconds: Optional[float] = None
    
    # Output configuration
    show_evidence: bool = Field(True, description="Show evidence in output")
    show_reasoning: bool = Field(True, description="Show reasoning chains")
    show_confidence: bool = Field(True, description="Show confidence scores")
    threaded_view: bool = Field(True, description="Use threaded conversation view")
    
    @classmethod
    def from_acvf_result(cls, acvf_result: ACVFResult, 
                        output_config: Optional[Dict[str, Any]] = None) -> 'DebateViewOutput':
        """Convert ACVFResult to output format."""
        config = output_config or {}
        
        debate_view = cls(
            session_id=acvf_result.session_id,
            subject_type=acvf_result.subject_type,
            subject_id=acvf_result.subject_id,
            subject_content="",  # Will need to be provided separately
            final_verdict=acvf_result.final_verdict,
            final_confidence=acvf_result.final_confidence,
            consensus_achieved=acvf_result.consensus_achieved,
            total_rounds=acvf_result.total_rounds,
            total_arguments=acvf_result.total_arguments,
            debate_duration_seconds=acvf_result.total_duration_seconds,
            show_evidence=config.get("show_evidence", True),
            show_reasoning=config.get("show_reasoning", True),
            show_confidence=config.get("show_confidence", True),
            threaded_view=config.get("threaded_view", True)
        )
        
        # Convert debate rounds and arguments
        for round_data in acvf_result.debate_rounds:
            round_info = {
                "round_number": round_data.round_number,
                "status": round_data.status.value,
                "verdict": round_data.final_verdict.value if round_data.final_verdict else None,
                "consensus_confidence": round_data.consensus_confidence
            }
            debate_view.rounds.append(round_info)
            
            # Convert arguments
            for arg in round_data.arguments:
                entry = DebateEntryOutput.from_debate_argument(
                    arg, round_data.round_id, 
                    f"{arg.role.value}_model"  # Simplified model name
                )
                debate_view.entries.append(entry)
        
        return debate_view


class OutputVerificationResult(BaseModel):
    """
    Specialized wrapper around VerificationChainResult for output generation.
    
    This class provides a view of verification results optimized for
    generating various output formats while maintaining compatibility
    with the existing verification system.
    """
    # Core verification data
    chain_result: VerificationChainResult
    issue_registry: IssueRegistry
    
    # Output context
    document: ParsedDocument
    output_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Annotation data
    annotation_layers: List[AnnotationLayer] = Field(default_factory=list)
    
    # ACVF debate data (if available)
    acvf_results: List[ACVFResult] = Field(default_factory=list)
    debate_views: List[DebateViewOutput] = Field(default_factory=list)
    
    # Output metadata
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generator_version: str = Field(default="1.0.0")
    
    @property
    def document_id(self) -> str:
        """Get the document ID."""
        return self.chain_result.document_id
    
    @property
    def verification_status(self) -> str:
        """Get the overall verification status."""
        return self.chain_result.status.value
    
    @property
    def total_issues(self) -> int:
        """Get the total number of issues found."""
        return len(self.issue_registry.issues)
    
    @property
    def critical_issues(self) -> List[UnifiedIssue]:
        """Get all critical severity issues."""
        return self.issue_registry.get_issues_by_severity(IssueSeverity.CRITICAL)
    
    @property
    def high_priority_issues(self) -> List[UnifiedIssue]:
        """Get all high severity issues."""
        return self.issue_registry.get_issues_by_severity(IssueSeverity.HIGH)
    
    @property
    def overall_confidence(self) -> Optional[float]:
        """Get the overall confidence score."""
        return self.chain_result.overall_confidence
    
    def get_issues_by_location(self, start: int, end: int) -> List[UnifiedIssue]:
        """Get all issues that overlap with the specified text range."""
        overlapping_issues = []
        for issue in self.issue_registry.issues:
            if (issue.location.start_position is not None and 
                issue.location.end_position is not None):
                if not (issue.location.end_position <= start or 
                       issue.location.start_position >= end):
                    overlapping_issues.append(issue)
        return overlapping_issues
    
    def get_annotations_for_issue(self, issue: UnifiedIssue) -> List[DocumentAnnotation]:
        """Get all annotations related to a specific issue."""
        related_annotations = []
        for layer in self.annotation_layers:
            for annotation in layer.annotations:
                if annotation.related_issue_id == issue.issue_id:
                    related_annotations.append(annotation)
        return related_annotations
    
    def add_annotation_layer(self, layer: AnnotationLayer) -> None:
        """Add an annotation layer to the result."""
        self.annotation_layers.append(layer)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the verification results."""
        issues_by_severity = {}
        for issue in self.issue_registry.issues:
            severity = issue.severity.value
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        return {
            "document_id": self.document_id,
            "verification_status": self.verification_status,
            "overall_confidence": self.overall_confidence,
            "total_issues": self.total_issues,
            "issues_by_severity": issues_by_severity,
            "verification_passes_completed": len(self.chain_result.pass_results),
            "acvf_debates_conducted": len(self.acvf_results),
            "processing_time_seconds": self.chain_result.total_execution_time_seconds,
            "generated_at": self.generated_at.isoformat()
        }


class DashboardDataPoint(BaseModel):
    """A single data point for dashboard visualizations."""
    timestamp: datetime
    metric_name: str
    metric_value: Union[int, float, str]
    metric_type: str = Field(description="count, percentage, score, etc.")
    category: Optional[str] = None
    subcategory: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DashboardVisualizationData(BaseModel):
    """
    Data structure for dashboard visualizations.
    
    Provides aggregated and formatted data optimized for
    dashboard charts, graphs, and summary displays.
    """
    visualization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    visualization_type: str = Field(description="chart, graph, table, metric, etc.")
    
    # Data
    data_points: List[DashboardDataPoint] = Field(default_factory=list)
    summary_metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    
    # Visualization config
    title: str
    description: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    color_scheme: Optional[ColorScheme] = None
    
    # Time range
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    
    # Filtering and grouping
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    group_by: Optional[str] = None
    
    def add_data_point(self, data_point: DashboardDataPoint) -> None:
        """Add a data point to the visualization."""
        self.data_points.append(data_point)
    
    def get_data_by_category(self, category: str) -> List[DashboardDataPoint]:
        """Get all data points for a specific category."""
        return [dp for dp in self.data_points if dp.category == category]


class APIResponseEnvelope(BaseModel):
    """
    Standard envelope for API responses.
    
    Provides consistent structure for all API responses with
    metadata, pagination, and error handling support.
    """
    # Request metadata
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="v1")
    
    # Response status
    success: bool = Field(True)
    status_code: int = Field(200)
    message: Optional[str] = None
    
    # Data payload
    data: Optional[Any] = None
    
    # Pagination (when applicable)
    pagination: Optional[Dict[str, Any]] = None
    
    # Error information
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    cached: bool = Field(False)
    
    @classmethod
    def success_response(cls, data: Any, message: Optional[str] = None,
                        pagination: Optional[Dict[str, Any]] = None) -> 'APIResponseEnvelope':
        """Create a successful response."""
        return cls(
            success=True,
            status_code=200,
            message=message,
            data=data,
            pagination=pagination
        )
    
    @classmethod
    def error_response(cls, errors: List[str], status_code: int = 400,
                      message: Optional[str] = None) -> 'APIResponseEnvelope':
        """Create an error response."""
        return cls(
            success=False,
            status_code=status_code,
            message=message or "Request failed",
            errors=errors
        )


class OutputGenerationConfig(BaseModel):
    """
    Configuration for output generation.
    
    Controls various aspects of how outputs are generated
    including styling, content inclusion, and format options.
    """
    # Format options
    output_format: OutputFormat
    include_annotations: bool = Field(True)
    include_debates: bool = Field(True)
    include_confidence_scores: bool = Field(True)
    include_metadata: bool = Field(True)
    
    # Content filtering
    minimum_issue_severity: IssueSeverity = Field(IssueSeverity.LOW)
    maximum_issues_per_document: Optional[int] = None
    
    # Styling
    color_scheme: ColorScheme = Field(default_factory=ColorScheme)
    font_family: str = Field(default="Arial, sans-serif")
    font_size: int = Field(default=12)
    
    # Document options
    include_summary: bool = Field(True)
    include_table_of_contents: bool = Field(True)
    include_appendices: bool = Field(True)
    
    # API options
    include_raw_data: bool = Field(False)
    compact_format: bool = Field(False)
    
    # Dashboard options
    time_range_days: int = Field(default=30)
    aggregation_level: str = Field(default="daily")  # hourly, daily, weekly, monthly
    
    # Custom options
    custom_options: Dict[str, Any] = Field(default_factory=dict)


# Utility functions for creating output data structures

def create_annotation_from_issue(issue: UnifiedIssue, 
                                annotation_type: AnnotationType = AnnotationType.HIGHLIGHT) -> DocumentAnnotation:
    """Create a DocumentAnnotation from a UnifiedIssue."""
    # Determine highlight style based on severity
    style_map = {
        IssueSeverity.CRITICAL: HighlightStyle.CRITICAL,
        IssueSeverity.HIGH: HighlightStyle.HIGH,
        IssueSeverity.MEDIUM: HighlightStyle.MEDIUM,
        IssueSeverity.LOW: HighlightStyle.LOW,
        IssueSeverity.INFORMATIONAL: HighlightStyle.INFO
    }
    
    return DocumentAnnotation(
        annotation_type=annotation_type,
        start_position=issue.location.start_position or 0,
        end_position=issue.location.end_position or 0,
        text_excerpt=issue.text_excerpt,
        title=issue.title,
        content=issue.description,
        style=style_map.get(issue.severity, HighlightStyle.MEDIUM),
        related_issue_id=issue.issue_id
    )


def create_output_result(chain_result: VerificationChainResult,
                        issue_registry: IssueRegistry,
                        document: ParsedDocument,
                        acvf_results: Optional[List[ACVFResult]] = None,
                        config: Optional[OutputGenerationConfig] = None) -> OutputVerificationResult:
    """Create an OutputVerificationResult from verification components."""
    
    output_result = OutputVerificationResult(
        chain_result=chain_result,
        issue_registry=issue_registry,
        document=document,
        acvf_results=acvf_results or [],
        output_config=config.dict() if config else {}
    )
    
    # Create annotation layers for each issue type
    highlight_layer = AnnotationLayer(
        layer_name="Issue Highlights",
        layer_type=AnnotationType.HIGHLIGHT,
        z_index=1
    )
    
    comment_layer = AnnotationLayer(
        layer_name="Issue Comments", 
        layer_type=AnnotationType.COMMENT,
        z_index=2
    )
    
    # Generate annotations from issues
    for issue in issue_registry.issues:
        if config and issue.severity.value < config.minimum_issue_severity.value:
            continue
            
        # Create highlight annotation
        highlight_annotation = create_annotation_from_issue(issue, AnnotationType.HIGHLIGHT)
        highlight_layer.add_annotation(highlight_annotation)
        
        # Create comment annotation if issue has substantial content
        if len(issue.description) > 50:
            comment_annotation = create_annotation_from_issue(issue, AnnotationType.COMMENT)
            comment_layer.add_annotation(comment_annotation)
    
    output_result.add_annotation_layer(highlight_layer)
    output_result.add_annotation_layer(comment_layer)
    
    # Create debate views from ACVF results
    for acvf_result in (acvf_results or []):
        debate_view = DebateViewOutput.from_acvf_result(acvf_result)
        output_result.debate_views.append(debate_view)
    
    return output_result 