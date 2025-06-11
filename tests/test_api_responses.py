"""
Unit tests for JSON API Response Structures.

This module tests the API response models to ensure proper serialization,
deserialization, and structure validation for all response types.
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import List

from src.models.api_responses import (
    APIVersion,
    ResponseFormat,
    ResponseMetadata,
    PaginationInfo,
    FilterOptions,
    SortOptions,
    SortOrder,
    BaseAPIResponse,
    IssueResponseSummary,
    IssueResponseStandard,
    IssueResponseDetailed,
    VerificationResultSummary,
    VerificationResultStandard,
    VerificationResultDetailed,
    PaginatedIssuesResponse,
    PaginatedVerificationResultsResponse,
    SingleIssueResponse,
    SingleVerificationResultResponse,
    DashboardResponse,
    ErrorResponse,
    SuccessResponse,
    ResponseSerializer,
    create_success_response,
    create_error_response,
    create_paginated_response
)
from src.models.issues import (
    UnifiedIssue,
    IssueType,
    IssueSeverity,
    IssueStatus,
    IssueLocation,
    IssueMetadata,
    IssueRegistry
)
from src.models.verification import (
    VerificationChainResult,
    VerificationResult,
    VerificationStatus,
    VerificationPassType
)
from src.models.output import OutputVerificationResult
from src.models.document import ParsedDocument


class TestResponseMetadata:
    """Test the ResponseMetadata model."""
    
    def test_default_metadata_creation(self):
        """Test creating metadata with default values."""
        metadata = ResponseMetadata()
        
        assert metadata.version == APIVersion.V1
        assert metadata.format == ResponseFormat.STANDARD
        assert metadata.cached is False
        assert metadata.cache_expires_at is None
        assert isinstance(metadata.request_id, str)
        assert isinstance(metadata.timestamp, datetime)
        
    def test_custom_metadata_creation(self):
        """Test creating metadata with custom values."""
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        metadata = ResponseMetadata(
            request_id=request_id,
            timestamp=timestamp,
            version=APIVersion.V2,
            format=ResponseFormat.DETAILED,
            processing_time_ms=150.5,
            cached=True
        )
        
        assert metadata.request_id == request_id
        assert metadata.timestamp == timestamp
        assert metadata.version == APIVersion.V2
        assert metadata.format == ResponseFormat.DETAILED
        assert metadata.processing_time_ms == 150.5
        assert metadata.cached is True
    
    def test_metadata_serialization(self):
        """Test metadata serialization to JSON."""
        metadata = ResponseMetadata(
            processing_time_ms=100.0,
            cached=True
        )
        
        json_data = metadata.dict()
        
        assert "request_id" in json_data
        assert "timestamp" in json_data
        assert json_data["version"] == "v1"
        assert json_data["format"] == "standard"
        assert json_data["processing_time_ms"] == 100.0
        assert json_data["cached"] is True


class TestPaginationInfo:
    """Test the PaginationInfo model."""
    
    def test_valid_pagination_creation(self):
        """Test creating valid pagination info."""
        pagination = PaginationInfo(
            page=2,
            page_size=20,
            total_items=100,
            total_pages=5,
            has_next=True,
            has_previous=True,
            next_page=3,
            previous_page=1
        )
        
        assert pagination.page == 2
        assert pagination.page_size == 20
        assert pagination.total_items == 100
        assert pagination.total_pages == 5
        assert pagination.has_next is True
        assert pagination.has_previous is True
        assert pagination.next_page == 3
        assert pagination.previous_page == 1
    
    def test_pagination_validation(self):
        """Test pagination validation constraints."""
        # Test minimum values
        with pytest.raises(ValueError):
            PaginationInfo(page=0, page_size=10, total_items=50, total_pages=5, has_next=False, has_previous=False)
        
        with pytest.raises(ValueError):
            PaginationInfo(page=1, page_size=0, total_items=50, total_pages=5, has_next=False, has_previous=False)
        
        with pytest.raises(ValueError):
            PaginationInfo(page=1, page_size=1001, total_items=50, total_pages=5, has_next=False, has_previous=False)


class TestFilterOptions:
    """Test the FilterOptions model."""
    
    def test_empty_filters(self):
        """Test creating empty filter options."""
        filters = FilterOptions()
        
        assert filters.severity is None
        assert filters.issue_types is None
        assert filters.verification_status is None
        assert filters.date_range is None
        assert filters.confidence_threshold is None
        assert filters.has_acvf_debate is None
    
    def test_populated_filters(self):
        """Test creating populated filter options."""
        start_date = datetime.now(timezone.utc)
        end_date = datetime.now(timezone.utc)
        
        filters = FilterOptions(
            severity=[IssueSeverity.CRITICAL, IssueSeverity.HIGH],
            issue_types=[IssueType.LOGICAL_FALLACY, IssueType.BIAS_DETECTION],
            verification_status=[VerificationStatus.COMPLETED],
            date_range={"start": start_date, "end": end_date},
            confidence_threshold=0.8,
            has_acvf_debate=True
        )
        
        assert len(filters.severity) == 2
        assert IssueSeverity.CRITICAL in filters.severity
        assert len(filters.issue_types) == 2
        assert IssueType.LOGICAL_FALLACY in filters.issue_types
        assert filters.confidence_threshold == 0.8
        assert filters.has_acvf_debate is True


class TestIssueResponses:
    """Test issue response models."""
    
    @pytest.fixture
    def sample_issue(self):
        """Create a sample UnifiedIssue for testing."""
        location = IssueLocation(
            start_position=100,
            end_position=150,
            line_number=5,
            section="Introduction",
            paragraph=2
        )
        
        metadata = IssueMetadata(
            detected_by=VerificationPassType.LOGIC_ANALYSIS,
            detection_model="test-model",
            acvf_session_ids=["session-123"]
        )
        
        return UnifiedIssue(
            issue_id="issue-123",
            issue_type=IssueType.LOGICAL_FALLACY,
            title="Ad Hominem Attack",
            description="Personal attack detected instead of addressing the argument",
            location=location,
            text_excerpt="You're wrong because you're not smart enough",
            context="In the context of discussing economic policy...",
            severity=IssueSeverity.HIGH,
            severity_score=0.8,
            confidence_score=0.9,
            impact_score=0.7,
            metadata=metadata,
            evidence=["Evidence 1", "Evidence 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            alternative_interpretations=["Alternative 1"]
        )
    
    def test_issue_summary_serialization(self, sample_issue):
        """Test serializing an issue to summary format."""
        summary = ResponseSerializer.serialize_issue(sample_issue, ResponseFormat.SUMMARY)
        
        assert isinstance(summary, IssueResponseSummary)
        assert summary.issue_id == "issue-123"
        assert summary.issue_type == IssueType.LOGICAL_FALLACY
        assert summary.title == "Ad Hominem Attack"
        assert summary.severity == IssueSeverity.HIGH
        assert summary.severity_score == 0.8
        assert summary.confidence_score == 0.9
        assert summary.has_acvf_debate is True  # Has session IDs
        assert summary.location is not None
        assert summary.location["start_position"] == 100
    
    def test_issue_standard_serialization(self, sample_issue):
        """Test serializing an issue to standard format."""
        standard = ResponseSerializer.serialize_issue(sample_issue, ResponseFormat.STANDARD)
        
        assert isinstance(standard, IssueResponseStandard)
        assert standard.issue_id == "issue-123"
        assert standard.title == "Ad Hominem Attack"
        assert standard.description == "Personal attack detected instead of addressing the argument"
        assert standard.text_excerpt == "You're wrong because you're not smart enough"
        assert standard.context == "In the context of discussing economic policy..."
        assert standard.detected_by == "logic_analysis"
        assert standard.evidence_count == 2
        assert standard.recommendations_count == 2
        assert standard.acvf_sessions_count == 1
    
    def test_issue_detailed_serialization(self, sample_issue):
        """Test serializing an issue to detailed format."""
        detailed = ResponseSerializer.serialize_issue(sample_issue, ResponseFormat.DETAILED)
        
        assert isinstance(detailed, IssueResponseDetailed)
        assert detailed.issue_id == "issue-123"
        assert len(detailed.evidence) == 2
        assert "Evidence 1" in detailed.evidence
        assert len(detailed.recommendations) == 2
        assert "Recommendation 1" in detailed.recommendations
        assert len(detailed.alternative_interpretations) == 1
        assert "Alternative 1" in detailed.alternative_interpretations
        assert len(detailed.acvf_session_ids) == 1
        assert "session-123" in detailed.acvf_session_ids
        assert isinstance(detailed.metadata, dict)


class TestVerificationResultResponses:
    """Test verification result response models."""
    
    @pytest.fixture
    def sample_chain_result(self):
        """Create a sample VerificationChainResult for testing."""
        pass_result = VerificationResult(
            pass_id="logic-analysis-1",
            pass_type=VerificationPassType.LOGIC_ANALYSIS,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            confidence_score=0.85
        )
        
        return VerificationChainResult(
            chain_id="chain-123",
            execution_id="exec-456",
            document_id="doc-789",
            status=VerificationStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            total_execution_time_seconds=45.5,
            pass_results=[pass_result],
            overall_confidence=0.82
        )
    
    @pytest.fixture
    def sample_output_result(self, sample_chain_result):
        """Create a sample OutputVerificationResult for testing."""
        # Create a sample issue registry
        issue_registry = IssueRegistry(document_id="doc-789")
        
        # Create a sample document
        document = ParsedDocument(
            document_id="doc-789",
            content="Sample document content",
            metadata={"title": "Test Document"}
        )
        
        return OutputVerificationResult(
            chain_result=sample_chain_result,
            issue_registry=issue_registry,
            document=document
        )
    
    def test_verification_result_summary_serialization(self, sample_output_result):
        """Test serializing verification result to summary format."""
        summary = ResponseSerializer.serialize_verification_result(
            sample_output_result, 
            ResponseFormat.SUMMARY
        )
        
        assert isinstance(summary, VerificationResultSummary)
        assert summary.document_id == "doc-789"
        assert summary.execution_id == "exec-456"
        assert summary.status == VerificationStatus.COMPLETED
        assert summary.total_issues == 0  # Empty issue registry
        assert summary.critical_issues == 0
        assert summary.overall_confidence == 0.82
        assert summary.has_acvf_debates is False
    
    def test_verification_result_standard_serialization(self, sample_output_result):
        """Test serializing verification result to standard format."""
        standard = ResponseSerializer.serialize_verification_result(
            sample_output_result,
            ResponseFormat.STANDARD
        )
        
        assert isinstance(standard, VerificationResultStandard)
        assert standard.document_id == "doc-789"
        assert standard.chain_id == "chain-123"
        assert len(standard.issues) == 0  # Empty issue registry
        assert len(standard.pass_results) == 1
        assert standard.pass_results[0]["pass_id"] == "logic-analysis-1"
        assert standard.pass_results[0]["pass_type"] == "logic_analysis"
        assert standard.pass_results[0]["status"] == "completed"
    
    def test_verification_result_detailed_serialization(self, sample_output_result):
        """Test serializing verification result to detailed format."""
        detailed = ResponseSerializer.serialize_verification_result(
            sample_output_result,
            ResponseFormat.DETAILED
        )
        
        assert isinstance(detailed, VerificationResultDetailed)
        assert detailed.document_id == "doc-789"
        assert detailed.chain_result is not None
        assert detailed.chain_result.execution_id == "exec-456"
        assert isinstance(detailed.issue_registry, dict)
        assert len(detailed.acvf_results) == 0
        assert isinstance(detailed.document_metadata, dict)
        assert detailed.document_metadata.get("title") == "Test Document"


class TestPaginatedResponses:
    """Test paginated response models."""
    
    def test_paginated_issues_response_creation(self):
        """Test creating a paginated issues response."""
        issues = [
            IssueResponseSummary(
                issue_id="issue-1",
                issue_type=IssueType.LOGICAL_FALLACY,
                title="Test Issue 1",
                severity=IssueSeverity.HIGH,
                severity_score=0.8,
                confidence_score=0.9,
                status=IssueStatus.DETECTED
            ),
            IssueResponseSummary(
                issue_id="issue-2",
                issue_type=IssueType.BIAS_DETECTION,
                title="Test Issue 2",
                severity=IssueSeverity.MEDIUM,
                severity_score=0.6,
                confidence_score=0.7,
                status=IssueStatus.DETECTED
            )
        ]
        
        pagination = PaginationInfo(
            page=1,
            page_size=10,
            total_items=2,
            total_pages=1,
            has_next=False,
            has_previous=False
        )
        
        response = PaginatedIssuesResponse(
            data=issues,
            pagination=pagination
        )
        
        assert response.success is True
        assert len(response.data) == 2
        assert response.pagination.total_items == 2
        assert isinstance(response.metadata, ResponseMetadata)
    
    def test_paginated_verification_results_response_creation(self):
        """Test creating a paginated verification results response."""
        results = [
            VerificationResultSummary(
                document_id="doc-1",
                execution_id="exec-1",
                status=VerificationStatus.COMPLETED,
                started_at=datetime.now(timezone.utc)
            )
        ]
        
        pagination = PaginationInfo(
            page=1,
            page_size=10,
            total_items=1,
            total_pages=1,
            has_next=False,
            has_previous=False
        )
        
        response = PaginatedVerificationResultsResponse(
            data=results,
            pagination=pagination
        )
        
        assert response.success is True
        assert len(response.data) == 1
        assert response.data[0].document_id == "doc-1"


class TestResponseFactoryFunctions:
    """Test response factory functions."""
    
    def test_create_success_response_simple(self):
        """Test creating a simple success response."""
        response = create_success_response(
            message="Operation completed successfully",
            processing_time_ms=50.0
        )
        
        assert isinstance(response, SuccessResponse)
        assert response.success is True
        assert response.message == "Operation completed successfully"
        assert response.metadata.processing_time_ms == 50.0
    
    def test_create_error_response(self):
        """Test creating an error response."""
        errors = ["Invalid input parameter", "Missing required field"]
        response = create_error_response(
            errors=errors,
            error_code="VALIDATION_ERROR",
            error_details={"field": "email"}
        )
        
        assert isinstance(response, ErrorResponse)
        assert response.success is False
        assert len(response.errors) == 2
        assert "Invalid input parameter" in response.errors
        assert response.error_code == "VALIDATION_ERROR"
        assert response.error_details["field"] == "email"
    
    def test_create_paginated_response_with_issues(self):
        """Test creating a paginated response with issues."""
        # Create mock issue objects with issue_id attribute
        class MockIssue:
            def __init__(self, issue_id: str):
                self.issue_id = issue_id
        
        issues = [MockIssue("issue-1"), MockIssue("issue-2")]
        
        response = create_paginated_response(
            data=issues,
            page=1,
            page_size=10,
            total_items=2
        )
        
        assert isinstance(response, PaginatedIssuesResponse)
        assert len(response.data) == 2
        assert response.pagination.total_items == 2
        assert response.pagination.has_next is False
    
    def test_create_paginated_response_with_verification_results(self):
        """Test creating a paginated response with verification results."""
        # Create mock verification result objects with execution_id attribute
        class MockVerificationResult:
            def __init__(self, execution_id: str):
                self.execution_id = execution_id
        
        results = [MockVerificationResult("exec-1")]
        
        response = create_paginated_response(
            data=results,
            page=2,
            page_size=5,
            total_items=8
        )
        
        assert isinstance(response, PaginatedVerificationResultsResponse)
        assert len(response.data) == 1
        assert response.pagination.page == 2
        assert response.pagination.total_pages == 2
        assert response.pagination.has_next is False
        assert response.pagination.has_previous is True


class TestResponseSerialization:
    """Test JSON serialization of response models."""
    
    def test_base_api_response_serialization(self):
        """Test serializing base API response to JSON."""
        response = BaseAPIResponse()
        json_data = response.dict()
        
        assert json_data["success"] is True
        assert "metadata" in json_data
        assert "errors" in json_data
        assert "warnings" in json_data
        assert isinstance(json_data["errors"], list)
        assert len(json_data["errors"]) == 0
    
    def test_issue_response_json_schema(self):
        """Test that issue responses have valid JSON schema."""
        issue_summary = IssueResponseSummary(
            issue_id="test-123",
            issue_type=IssueType.LOGICAL_FALLACY,
            title="Test Issue",
            severity=IssueSeverity.HIGH,
            severity_score=0.8,
            confidence_score=0.9,
            status=IssueStatus.DETECTED
        )
        
        json_data = issue_summary.dict()
        
        # Verify all required fields are present
        required_fields = [
            "issue_id", "issue_type", "title", "severity", 
            "severity_score", "confidence_score", "status"
        ]
        
        for field in required_fields:
            assert field in json_data
        
        # Verify enum values are serialized as strings
        assert json_data["issue_type"] == "logical_fallacy"
        assert json_data["severity"] == "high"
        assert json_data["status"] == "detected"
    
    def test_complex_response_serialization(self):
        """Test serializing complex nested response structures."""
        # Create a complex paginated response
        issue = IssueResponseSummary(
            issue_id="test-123",
            issue_type=IssueType.BIAS_DETECTION,
            title="Test Bias Issue",
            severity=IssueSeverity.MEDIUM,
            severity_score=0.6,
            confidence_score=0.8,
            status=IssueStatus.UNDER_REVIEW
        )
        
        pagination = PaginationInfo(
            page=1,
            page_size=20,
            total_items=1,
            total_pages=1,
            has_next=False,
            has_previous=False
        )
        
        response = PaginatedIssuesResponse(
            data=[issue],
            pagination=pagination
        )
        
        json_data = response.dict()
        
        # Verify structure
        assert "success" in json_data
        assert "metadata" in json_data
        assert "data" in json_data
        assert "pagination" in json_data
        assert "filters_applied" in json_data
        assert "sort_options" in json_data
        
        # Verify nested data
        assert len(json_data["data"]) == 1
        assert json_data["data"][0]["issue_id"] == "test-123"
        assert json_data["pagination"]["total_items"] == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data_responses(self):
        """Test responses with empty data."""
        response = PaginatedIssuesResponse(
            data=[],
            pagination=PaginationInfo(
                page=1,
                page_size=10,
                total_items=0,
                total_pages=0,
                has_next=False,
                has_previous=False
            )
        )
        
        assert response.success is True
        assert len(response.data) == 0
        assert response.pagination.total_items == 0
    
    def test_response_with_warnings(self):
        """Test responses that include warnings."""
        response = BaseAPIResponse(
            warnings=["This is a deprecation warning", "Performance may be degraded"]
        )
        
        assert len(response.warnings) == 2
        assert "deprecation warning" in response.warnings[0]
    
    def test_response_with_errors_but_partial_success(self):
        """Test responses that have errors but still succeed partially."""
        response = BaseAPIResponse(
            success=True,
            errors=["Non-critical error occurred"],
            warnings=["Some data may be incomplete"]
        )
        
        assert response.success is True
        assert len(response.errors) == 1
        assert len(response.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__]) 