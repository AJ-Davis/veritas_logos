"""
JSON API Response Structures for the Veritas Logos verification system.

This module provides standardized JSON response structures for API endpoints
that return verification results. It builds upon the existing verification
and output models while providing API-specific features like pagination,
filtering, versioning, and different detail levels.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field

from .verification import VerificationChainResult, VerificationResult, VerificationStatus
from .issues import UnifiedIssue, IssueSeverity, IssueType, IssueStatus
from .output import OutputVerificationResult, DebateViewOutput, DashboardVisualizationData
from .acvf import ACVFResult


class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"  # Future version


class ResponseFormat(str, Enum):
    """Response detail levels."""
    SUMMARY = "summary"      # Minimal information
    STANDARD = "standard"    # Standard detail level
    DETAILED = "detailed"    # Full information
    COMPACT = "compact"      # Compressed format for bandwidth efficiency


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class ResponseMetadata(BaseModel):
    """Metadata included in all API responses."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: APIVersion = Field(APIVersion.V1)
    format: ResponseFormat = Field(ResponseFormat.STANDARD)
    processing_time_ms: Optional[float] = None
    cached: bool = Field(False)
    cache_expires_at: Optional[datetime] = None


class PaginationInfo(BaseModel):
    """Pagination information for paginated responses."""
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, le=1000, description="Items per page")
    total_items: int = Field(ge=0, description="Total number of items")
    total_pages: int = Field(ge=0, description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_previous: bool = Field(description="Whether there are previous pages")
    next_page: Optional[int] = Field(None, description="Next page number if available")
    previous_page: Optional[int] = Field(None, description="Previous page number if available")


class FilterOptions(BaseModel):
    """Available filtering options for API responses."""
    severity: Optional[List[IssueSeverity]] = None
    issue_types: Optional[List[IssueType]] = None
    verification_status: Optional[List[VerificationStatus]] = None
    date_range: Optional[Dict[str, datetime]] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    has_acvf_debate: Optional[bool] = None


class SortOptions(BaseModel):
    """Sorting options for API responses."""
    field: str = Field(description="Field to sort by")
    order: SortOrder = Field(SortOrder.DESC)


class BaseAPIResponse(BaseModel):
    """Base structure for all API responses."""
    success: bool = Field(True)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# Issue-related responses
class IssueResponseSummary(BaseModel):
    """Summary representation of an issue for API responses."""
    issue_id: str
    issue_type: IssueType
    title: str
    severity: IssueSeverity
    severity_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    status: IssueStatus
    location: Optional[Dict[str, Any]] = None  # Simplified location info
    has_acvf_debate: bool = Field(False)


class IssueResponseStandard(BaseModel):
    """Standard representation of an issue for API responses."""
    issue_id: str
    issue_type: IssueType
    title: str
    description: str
    severity: IssueSeverity
    severity_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    status: IssueStatus
    
    # Location and context
    location: Dict[str, Any]
    text_excerpt: str
    context: Optional[str] = None
    
    # Metadata
    detected_by: str
    detection_timestamp: datetime
    
    # Evidence and recommendations (limited for standard view)
    evidence_count: int = Field(0, ge=0)
    recommendations_count: int = Field(0, ge=0)
    
    # ACVF information
    has_acvf_debate: bool = Field(False)
    acvf_sessions_count: int = Field(0, ge=0)


class IssueResponseDetailed(BaseModel):
    """Detailed representation of an issue for API responses."""
    issue_id: str
    issue_type: IssueType
    title: str
    description: str
    severity: IssueSeverity
    severity_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    status: IssueStatus
    
    # Location and context
    location: Dict[str, Any]
    text_excerpt: str
    context: Optional[str] = None
    
    # Full metadata
    metadata: Dict[str, Any]
    
    # Evidence and recommendations
    evidence: List[str]
    recommendations: List[str]
    alternative_interpretations: List[str]
    
    # ACVF information
    has_acvf_debate: bool = Field(False)
    acvf_session_ids: List[str] = Field(default_factory=list)
    
    # Resolution information
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


# Verification result responses
class VerificationResultSummary(BaseModel):
    """Summary representation of verification results."""
    document_id: str
    execution_id: str
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Issue summary
    total_issues: int = Field(0, ge=0)
    critical_issues: int = Field(0, ge=0)
    high_severity_issues: int = Field(0, ge=0)
    
    # Overall scores
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    document_credibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # ACVF summary
    has_acvf_debates: bool = Field(False)
    total_debates: int = Field(0, ge=0)


class VerificationResultStandard(BaseModel):
    """Standard representation of verification results."""
    document_id: str
    execution_id: str
    chain_id: str
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_execution_time_seconds: Optional[float] = None
    
    # Issues (summary format)
    issues: List[IssueResponseSummary] = Field(default_factory=list)
    issue_summary: Dict[str, int] = Field(default_factory=dict)  # Issues by type/severity
    
    # Pass results summary
    pass_results: List[Dict[str, Any]] = Field(default_factory=list)  # Simplified pass info
    
    # Overall scores
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    document_credibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # ACVF summary
    has_acvf_debates: bool = Field(False)
    total_debates: int = Field(0, ge=0)
    debate_summaries: List[Dict[str, Any]] = Field(default_factory=list)


class VerificationResultDetailed(BaseModel):
    """Detailed representation of verification results."""
    document_id: str
    execution_id: str
    chain_id: str
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_execution_time_seconds: Optional[float] = None
    
    # Full chain result data
    chain_result: VerificationChainResult
    
    # Issues (detailed format based on request)
    issues: List[Union[IssueResponseSummary, IssueResponseStandard, IssueResponseDetailed]] = Field(default_factory=list)
    issue_registry: Dict[str, Any]  # Full issue registry data
    
    # ACVF results
    acvf_results: List[ACVFResult] = Field(default_factory=list)
    debate_views: List[DebateViewOutput] = Field(default_factory=list)
    
    # Document information
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing metadata
    processing_summary: Dict[str, Any] = Field(default_factory=dict)


# Paginated responses
class PaginatedIssuesResponse(BaseAPIResponse):
    """Paginated response for issues."""
    data: List[Union[IssueResponseSummary, IssueResponseStandard, IssueResponseDetailed]] = Field(default_factory=list)
    pagination: PaginationInfo
    filters_applied: FilterOptions = Field(default_factory=FilterOptions)
    sort_options: SortOptions = Field(default_factory=SortOptions)


class PaginatedVerificationResultsResponse(BaseAPIResponse):
    """Paginated response for verification results."""
    data: List[Union[VerificationResultSummary, VerificationResultStandard, VerificationResultDetailed]] = Field(default_factory=list)
    pagination: PaginationInfo
    filters_applied: FilterOptions = Field(default_factory=FilterOptions)
    sort_options: SortOptions = Field(default_factory=SortOptions)


# Single item responses
class SingleIssueResponse(BaseAPIResponse):
    """Response for a single issue."""
    data: Union[IssueResponseSummary, IssueResponseStandard, IssueResponseDetailed]


class SingleVerificationResultResponse(BaseAPIResponse):
    """Response for a single verification result."""
    data: Union[VerificationResultSummary, VerificationResultStandard, VerificationResultDetailed]


# Dashboard and analytics responses
class DashboardResponse(BaseAPIResponse):
    """Response for dashboard data."""
    data: List[DashboardVisualizationData]
    summary_metrics: Dict[str, Union[int, float, str]]
    time_range: Dict[str, datetime]


class AnalyticsResponse(BaseAPIResponse):
    """Response for analytics data."""
    data: Dict[str, Any]
    aggregation_period: str
    metric_definitions: Dict[str, str] = Field(default_factory=dict)


# Debate view responses
class DebateResponse(BaseAPIResponse):
    """Response for ACVF debate data."""
    data: DebateViewOutput


class DebateListResponse(BaseAPIResponse):
    """Response for multiple debates."""
    data: List[DebateViewOutput]
    filters_applied: Dict[str, Any] = Field(default_factory=dict)


# Utility response classes
class ErrorResponse(BaseAPIResponse):
    """Standardized error response."""
    success: bool = Field(False)
    error_code: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)


class SuccessResponse(BaseAPIResponse):
    """Simple success response with optional message."""
    message: Optional[str] = None


# Response factory functions
def create_success_response(
    data: Any = None,
    message: Optional[str] = None,
    format: ResponseFormat = ResponseFormat.STANDARD,
    processing_time_ms: Optional[float] = None
) -> BaseAPIResponse:
    """Create a standardized success response."""
    metadata = ResponseMetadata(format=format, processing_time_ms=processing_time_ms)
    
    if data is None:
        return SuccessResponse(metadata=metadata, message=message)
    
    # Determine response type based on data
    if isinstance(data, list):
        if len(data) > 0 and hasattr(data[0], 'issue_id'):
            # List of issues
            return PaginatedIssuesResponse(
                data=data,
                metadata=metadata,
                pagination=PaginationInfo(page=1, page_size=len(data), total_items=len(data), total_pages=1, has_next=False, has_previous=False)
            )
        elif len(data) > 0 and hasattr(data[0], 'execution_id'):
            # List of verification results
            return PaginatedVerificationResultsResponse(
                data=data,
                metadata=metadata,
                pagination=PaginationInfo(page=1, page_size=len(data), total_items=len(data), total_pages=1, has_next=False, has_previous=False)
            )
    
    # Single item responses
    if hasattr(data, 'issue_id'):
        return SingleIssueResponse(data=data, metadata=metadata)
    elif hasattr(data, 'execution_id'):
        return SingleVerificationResultResponse(data=data, metadata=metadata)
    
    # Generic success response
    return SuccessResponse(data=data, metadata=metadata, message=message)


def create_error_response(
    errors: List[str],
    error_code: Optional[str] = None,
    error_details: Optional[Dict[str, Any]] = None,
    status_code: int = 400
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        errors=errors,
        error_code=error_code,
        error_details=error_details or {},
        metadata=ResponseMetadata()
    )


def create_paginated_response(
    data: List[Any],
    page: int,
    page_size: int,
    total_items: int,
    filters: Optional[FilterOptions] = None,
    sort_options: Optional[SortOptions] = None,
    format: ResponseFormat = ResponseFormat.STANDARD
) -> Union[PaginatedIssuesResponse, PaginatedVerificationResultsResponse]:
    """Create a paginated response with proper pagination metadata."""
    total_pages = (total_items + page_size - 1) // page_size
    has_next = page < total_pages
    has_previous = page > 1
    
    pagination = PaginationInfo(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=has_next,
        has_previous=has_previous,
        next_page=page + 1 if has_next else None,
        previous_page=page - 1 if has_previous else None
    )
    
    metadata = ResponseMetadata(format=format)
    
    # Determine response type
    if len(data) > 0 and hasattr(data[0], 'issue_id'):
        return PaginatedIssuesResponse(
            data=data,
            pagination=pagination,
            metadata=metadata,
            filters_applied=filters or FilterOptions(),
            sort_options=sort_options or SortOptions(field="created_at")
        )
    else:
        return PaginatedVerificationResultsResponse(
            data=data,
            pagination=pagination,
            metadata=metadata,
            filters_applied=filters or FilterOptions(),
            sort_options=sort_options or SortOptions(field="created_at")
        )


# Serialization utilities
class ResponseSerializer:
    """Utility class for converting domain models to API response models."""
    
    @staticmethod
    def serialize_issue(
        issue: UnifiedIssue,
        format: ResponseFormat = ResponseFormat.STANDARD,
        include_acvf: bool = True
    ) -> Union[IssueResponseSummary, IssueResponseStandard, IssueResponseDetailed]:
        """Convert a UnifiedIssue to the appropriate response format."""
        has_acvf_debate = len(issue.metadata.acvf_session_ids) > 0
        
        if format == ResponseFormat.SUMMARY or format == ResponseFormat.COMPACT:
            return IssueResponseSummary(
                issue_id=issue.issue_id,
                issue_type=issue.issue_type,
                title=issue.title,
                severity=issue.severity,
                severity_score=issue.severity_score,
                confidence_score=issue.confidence_score,
                status=issue.status,
                location={
                    "start_position": issue.location.start_position,
                    "end_position": issue.location.end_position,
                    "section": issue.location.section
                } if issue.location else None,
                has_acvf_debate=has_acvf_debate
            )
        
        elif format == ResponseFormat.STANDARD:
            return IssueResponseStandard(
                issue_id=issue.issue_id,
                issue_type=issue.issue_type,
                title=issue.title,
                description=issue.description,
                severity=issue.severity,
                severity_score=issue.severity_score,
                confidence_score=issue.confidence_score,
                impact_score=issue.impact_score,
                status=issue.status,
                location=issue.location.dict() if issue.location else {},
                text_excerpt=issue.text_excerpt,
                context=issue.context,
                detected_by=issue.metadata.detected_by.value,
                detection_timestamp=issue.metadata.detection_timestamp,
                evidence_count=len(issue.evidence),
                recommendations_count=len(issue.recommendations),
                has_acvf_debate=has_acvf_debate,
                acvf_sessions_count=len(issue.metadata.acvf_session_ids)
            )
        
        else:  # DETAILED
            return IssueResponseDetailed(
                issue_id=issue.issue_id,
                issue_type=issue.issue_type,
                title=issue.title,
                description=issue.description,
                severity=issue.severity,
                severity_score=issue.severity_score,
                confidence_score=issue.confidence_score,
                impact_score=issue.impact_score,
                status=issue.status,
                location=issue.location.dict() if issue.location else {},
                text_excerpt=issue.text_excerpt,
                context=issue.context,
                metadata=issue.metadata.dict(),
                evidence=issue.evidence,
                recommendations=issue.recommendations,
                alternative_interpretations=issue.alternative_interpretations,
                has_acvf_debate=has_acvf_debate,
                acvf_session_ids=issue.metadata.acvf_session_ids,
                resolution_notes=issue.resolution_notes,
                resolved_at=issue.resolved_at,
                resolved_by=issue.resolved_by
            )
    
    @staticmethod
    def serialize_verification_result(
        output_result: OutputVerificationResult,
        format: ResponseFormat = ResponseFormat.STANDARD,
        include_issues_format: ResponseFormat = ResponseFormat.SUMMARY
    ) -> Union[VerificationResultSummary, VerificationResultStandard, VerificationResultDetailed]:
        """Convert an OutputVerificationResult to the appropriate response format."""
        chain_result = output_result.chain_result
        issues = output_result.issue_registry.issues
        
        # Common calculations
        total_issues = len(issues)
        critical_issues = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high_severity_issues = len([i for i in issues if i.severity == IssueSeverity.HIGH])
        has_acvf_debates = len(output_result.acvf_results) > 0
        
        if format == ResponseFormat.SUMMARY or format == ResponseFormat.COMPACT:
            return VerificationResultSummary(
                document_id=chain_result.document_id,
                execution_id=chain_result.execution_id,
                status=chain_result.status,
                started_at=chain_result.started_at,
                completed_at=chain_result.completed_at,
                total_issues=total_issues,
                critical_issues=critical_issues,
                high_severity_issues=high_severity_issues,
                overall_confidence=chain_result.overall_confidence,
                has_acvf_debates=has_acvf_debates,
                total_debates=len(output_result.acvf_results)
            )
        
        elif format == ResponseFormat.STANDARD:
            # Serialize issues in summary format for standard response
            serialized_issues = [
                ResponseSerializer.serialize_issue(issue, ResponseFormat.SUMMARY)
                for issue in issues
            ]
            
            # Create issue summary
            issue_summary = {}
            for issue in issues:
                issue_summary[issue.issue_type.value] = issue_summary.get(issue.issue_type.value, 0) + 1
                issue_summary[issue.severity.value] = issue_summary.get(issue.severity.value, 0) + 1
            
            # Simplified pass results
            pass_results = [
                {
                    "pass_id": result.pass_id,
                    "pass_type": result.pass_type.value,
                    "status": result.status.value,
                    "execution_time_seconds": result.execution_time_seconds,
                    "confidence_score": result.confidence_score
                }
                for result in chain_result.pass_results
            ]
            
            return VerificationResultStandard(
                document_id=chain_result.document_id,
                execution_id=chain_result.execution_id,
                chain_id=chain_result.chain_id,
                status=chain_result.status,
                started_at=chain_result.started_at,
                completed_at=chain_result.completed_at,
                total_execution_time_seconds=chain_result.total_execution_time_seconds,
                issues=serialized_issues,
                issue_summary=issue_summary,
                pass_results=pass_results,
                overall_confidence=chain_result.overall_confidence,
                has_acvf_debates=has_acvf_debates,
                total_debates=len(output_result.acvf_results)
            )
        
        else:  # DETAILED
            # Serialize issues in the requested format
            serialized_issues = [
                ResponseSerializer.serialize_issue(issue, include_issues_format)
                for issue in issues
            ]
            
            return VerificationResultDetailed(
                document_id=chain_result.document_id,
                execution_id=chain_result.execution_id,
                chain_id=chain_result.chain_id,
                status=chain_result.status,
                started_at=chain_result.started_at,
                completed_at=chain_result.completed_at,
                total_execution_time_seconds=chain_result.total_execution_time_seconds,
                chain_result=chain_result,
                issues=serialized_issues,
                issue_registry=output_result.issue_registry.dict(),
                acvf_results=output_result.acvf_results,
                debate_views=output_result.debate_views,
                document_metadata=output_result.document.metadata if output_result.document else {},
                processing_summary=output_result.generate_summary()
            ) 