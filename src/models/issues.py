"""
Unified issue models for the verification system.

This module provides a comprehensive issue representation that can handle
issues from all verification passes with proper metadata, severity scoring,
confidence aggregation, and escalation tracking.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

from .verification import VerificationPassType
from .logic_bias import LogicalIssue, BiasIssue
from .citations import CitationIssue, VerifiedCitation


class IssueType(str, Enum):
    """Types of issues that can be detected."""
    LOGICAL_FALLACY = "logical_fallacy"
    BIAS_DETECTION = "bias_detection"
    CITATION_PROBLEM = "citation_problem"
    EVIDENCE_MISSING = "evidence_missing"
    CLAIM_UNSUPPORTED = "claim_unsupported"
    FACTUAL_ERROR = "factual_error"
    PLAGIARISM = "plagiarism"
    FORMATTING_ERROR = "formatting_error"
    CREDIBILITY_CONCERN = "credibility_concern"
    CONSISTENCY_ISSUE = "consistency_issue"
    AMBIGUITY = "ambiguity"
    MISSING_CONTEXT = "missing_context"
    VERIFICATION_FAILURE = "verification_failure"


class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"      # 0.8-1.0: Document integrity compromised
    HIGH = "high"             # 0.6-0.8: Significant credibility impact
    MEDIUM = "medium"         # 0.4-0.6: Moderate concerns
    LOW = "low"               # 0.2-0.4: Minor issues
    INFORMATIONAL = "info"    # 0.0-0.2: Observations, style suggestions


class IssueStatus(str, Enum):
    """Status of issue processing."""
    DETECTED = "detected"         # Initial detection
    UNDER_REVIEW = "under_review" # Being processed/escalated
    ACVF_ESCALATED = "acvf_escalated"  # Sent to ACVF for debate
    RESOLVED = "resolved"         # Issue addressed
    DISMISSED = "dismissed"       # Determined to be false positive
    REQUIRES_HUMAN = "requires_human"  # Needs human review


class EscalationPath(str, Enum):
    """Available escalation paths for issues."""
    NONE = "none"                 # No escalation needed
    ACVF_DEBATE = "acvf_debate"   # Send to ACVF for adversarial validation
    HUMAN_REVIEW = "human_review" # Requires human expert review
    AUTOMATED_RETRY = "automated_retry"  # Retry with different approach
    EXTERNAL_FACT_CHECK = "external_fact_check"  # External verification needed


class IssueLocation(BaseModel):
    """Location information for an issue within a document."""
    start_position: Optional[int] = Field(None, description="Character start position")
    end_position: Optional[int] = Field(None, description="Character end position")
    line_number: Optional[int] = Field(None, description="Line number if available")
    section: Optional[str] = Field(None, description="Document section")
    paragraph: Optional[int] = Field(None, description="Paragraph number")
    sentence: Optional[int] = Field(None, description="Sentence number within paragraph")


class IssueMetadata(BaseModel):
    """Comprehensive metadata for issue tracking."""
    detected_by: VerificationPassType = Field(..., description="Which pass detected this issue")
    detection_model: Optional[str] = Field(None, description="Specific model/method used")
    detection_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Related entities
    related_claim_ids: List[str] = Field(default_factory=list)
    related_citation_ids: List[str] = Field(default_factory=list)
    related_issue_ids: List[str] = Field(default_factory=list)
    
    # Processing history
    escalation_history: List[Dict[str, Any]] = Field(default_factory=list)
    acvf_session_ids: List[str] = Field(default_factory=list)
    review_attempts: int = Field(0, description="Number of review attempts")
    
    # Verification lineage
    source_verification_id: Optional[str] = Field(None, description="Source verification result ID")
    contributing_passes: List[VerificationPassType] = Field(default_factory=list)
    
    # Additional context
    document_context: Dict[str, Any] = Field(default_factory=dict)
    processing_notes: List[str] = Field(default_factory=list)


class UnifiedIssue(BaseModel):
    """
    Unified issue representation that can handle issues from all verification passes.
    
    This model provides a standardized way to represent issues with comprehensive
    metadata, severity scoring, confidence tracking, and escalation management.
    """
    # Core identification
    issue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issue_type: IssueType = Field(..., description="Type of issue detected")
    title: str = Field(..., description="Brief descriptive title")
    description: str = Field(..., description="Detailed description of the issue")
    
    # Location and content
    location: IssueLocation = Field(..., description="Where the issue occurs")
    text_excerpt: str = Field(..., description="Relevant text excerpt")
    context: Optional[str] = Field(None, description="Surrounding context")
    
    # Severity and confidence
    severity: IssueSeverity = Field(..., description="Severity level")
    severity_score: float = Field(..., ge=0.0, le=1.0, description="Numeric severity (0.0-1.0)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Potential impact on document credibility")
    
    # Status and escalation
    status: IssueStatus = Field(IssueStatus.DETECTED, description="Current processing status")
    escalation_path: EscalationPath = Field(EscalationPath.NONE, description="Recommended escalation")
    requires_escalation: bool = Field(False, description="Whether escalation is needed")
    
    # Metadata and tracking
    metadata: IssueMetadata = Field(..., description="Comprehensive metadata")
    
    # Evidence and recommendations
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    recommendations: List[str] = Field(default_factory=list, description="Suggested actions")
    alternative_interpretations: List[str] = Field(default_factory=list, description="Alternative viewpoints")
    
    # Resolution information
    resolution_notes: Optional[str] = Field(None, description="Notes on how issue was resolved")
    resolved_at: Optional[datetime] = Field(None, description="When issue was resolved")
    resolved_by: Optional[str] = Field(None, description="Who/what resolved the issue")
    
    # Aggregation support
    source_issues: List[str] = Field(default_factory=list, description="Original issue IDs that were merged")
    is_aggregated: bool = Field(False, description="Whether this is an aggregated issue")
    aggregation_method: Optional[str] = Field(None, description="How issues were aggregated")

    def add_escalation_event(self, escalation_type: str, details: Dict[str, Any]) -> None:
        """Add an escalation event to the history."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "escalation_type": escalation_type,
            "details": details
        }
        self.metadata.escalation_history.append(event)
    
    def add_acvf_session(self, session_id: str) -> None:
        """Add an ACVF session ID to the tracking list."""
        if session_id not in self.metadata.acvf_session_ids:
            self.metadata.acvf_session_ids.append(session_id)
    
    def update_status(self, new_status: IssueStatus, notes: Optional[str] = None) -> None:
        """Update issue status with optional notes."""
        self.status = new_status
        if notes:
            self.metadata.processing_notes.append(f"{datetime.now(timezone.utc).isoformat()}: {notes}")
    
    def calculate_priority_score(self) -> float:
        """Calculate a priority score for issue ordering."""
        # Combine severity, confidence, and impact with weights
        priority = (
            self.severity_score * 0.4 +      # Severity is most important
            self.confidence_score * 0.3 +    # Confidence in detection
            self.impact_score * 0.3          # Potential impact
        )
        
        # Boost priority for certain issue types
        critical_types = {IssueType.FACTUAL_ERROR, IssueType.PLAGIARISM, IssueType.CREDIBILITY_CONCERN}
        if self.issue_type in critical_types:
            priority *= 1.2
        
        # Boost priority if multiple passes detected related issues
        if len(self.metadata.contributing_passes) > 1:
            priority *= 1.1
        
        return min(1.0, priority)  # Cap at 1.0


class IssueRegistry(BaseModel):
    """
    Registry for managing and tracking issues across verification passes.
    
    Provides centralized issue management with deduplication, aggregation,
    prioritization, and escalation routing capabilities.
    """
    document_id: str = Field(..., description="Document being analyzed")
    issues: List[UnifiedIssue] = Field(default_factory=list)
    
    # Tracking metrics
    total_issues_detected: int = Field(0, description="Total issues found")
    issues_by_type: Dict[str, int] = Field(default_factory=dict)
    issues_by_severity: Dict[str, int] = Field(default_factory=dict)
    issues_by_pass: Dict[str, int] = Field(default_factory=dict)
    
    # Registry metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_summary: Dict[str, Any] = Field(default_factory=dict)

    def add_issue(self, issue: UnifiedIssue) -> str:
        """Add an issue to the registry."""
        self.issues.append(issue)
        self.total_issues_detected += 1
        self.last_updated = datetime.now(timezone.utc)
        
        # Update tracking metrics
        self._update_metrics()
        
        return issue.issue_id
    
    def get_issues_by_type(self, issue_type: IssueType) -> List[UnifiedIssue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]
    
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[UnifiedIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_requiring_escalation(self) -> List[UnifiedIssue]:
        """Get all issues that require escalation."""
        return [issue for issue in self.issues if issue.requires_escalation]
    
    def get_prioritized_issues(self) -> List[UnifiedIssue]:
        """Get issues sorted by priority score."""
        return sorted(self.issues, key=lambda x: x.calculate_priority_score(), reverse=True)
    
    def find_similar_issues(self, issue: UnifiedIssue, similarity_threshold: float = 0.8) -> List[UnifiedIssue]:
        """Find issues similar to the given issue."""
        similar = []
        
        for existing_issue in self.issues:
            if existing_issue.issue_id == issue.issue_id:
                continue
            
            # Simple similarity based on type, location, and text overlap
            type_match = existing_issue.issue_type == issue.issue_type
            location_match = self._locations_overlap(existing_issue.location, issue.location)
            text_similarity = self._calculate_text_similarity(
                existing_issue.text_excerpt, 
                issue.text_excerpt
            )
            
            if type_match and (location_match or text_similarity >= similarity_threshold):
                similar.append(existing_issue)
        
        return similar
    
    def aggregate_similar_issues(self, similarity_threshold: float = 0.8) -> int:
        """Aggregate similar issues and return count of aggregations performed."""
        aggregations = 0
        processed_ids = set()
        
        for issue in self.issues[:]:  # Create a copy to iterate over
            if issue.issue_id in processed_ids or issue.is_aggregated:
                continue
            
            similar_issues = self.find_similar_issues(issue, similarity_threshold)
            
            if similar_issues:
                # Create aggregated issue
                aggregated = self._create_aggregated_issue(issue, similar_issues)
                
                # Remove original issues
                for similar_issue in similar_issues:
                    if similar_issue in self.issues:
                        self.issues.remove(similar_issue)
                        processed_ids.add(similar_issue.issue_id)
                
                # Replace original issue with aggregated version
                issue_index = self.issues.index(issue)
                self.issues[issue_index] = aggregated
                processed_ids.add(issue.issue_id)
                aggregations += 1
        
        if aggregations > 0:
            self._update_metrics()
        
        return aggregations
    
    def _update_metrics(self) -> None:
        """Update tracking metrics."""
        self.issues_by_type = {}
        self.issues_by_severity = {}
        self.issues_by_pass = {}
        
        for issue in self.issues:
            # Count by type
            issue_type_str = issue.issue_type.value
            self.issues_by_type[issue_type_str] = self.issues_by_type.get(issue_type_str, 0) + 1
            
            # Count by severity
            severity_str = issue.severity.value
            self.issues_by_severity[severity_str] = self.issues_by_severity.get(severity_str, 0) + 1
            
            # Count by detection pass
            pass_str = issue.metadata.detected_by.value
            self.issues_by_pass[pass_str] = self.issues_by_pass.get(pass_str, 0) + 1
    
    def _locations_overlap(self, loc1: IssueLocation, loc2: IssueLocation) -> bool:
        """Check if two locations overlap."""
        if (loc1.start_position is not None and loc1.end_position is not None and
            loc2.start_position is not None and loc2.end_position is not None):
            
            # Check for character position overlap
            return not (loc1.end_position < loc2.start_position or loc2.end_position < loc1.start_position)
        
        # Fallback to section/paragraph matching
        return (loc1.section == loc2.section and loc1.paragraph == loc2.paragraph)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text excerpts."""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _create_aggregated_issue(self, primary_issue: UnifiedIssue, similar_issues: List[UnifiedIssue]) -> UnifiedIssue:
        """Create an aggregated issue from similar issues."""
        all_issues = [primary_issue] + similar_issues
        
        # Use the highest severity and confidence
        max_severity_score = max(issue.severity_score for issue in all_issues)
        max_confidence = max(issue.confidence_score for issue in all_issues)
        max_impact = max(issue.impact_score for issue in all_issues)
        
        # Determine severity enum from score
        if max_severity_score >= 0.8:
            severity = IssueSeverity.CRITICAL
        elif max_severity_score >= 0.6:
            severity = IssueSeverity.HIGH
        elif max_severity_score >= 0.4:
            severity = IssueSeverity.MEDIUM
        elif max_severity_score >= 0.2:
            severity = IssueSeverity.LOW
        else:
            severity = IssueSeverity.INFORMATIONAL
        
        # Combine evidence and recommendations
        all_evidence = []
        all_recommendations = []
        all_contributing_passes = set()
        
        for issue in all_issues:
            all_evidence.extend(issue.evidence)
            all_recommendations.extend(issue.recommendations)
            all_contributing_passes.add(issue.metadata.detected_by)
            all_contributing_passes.update(issue.metadata.contributing_passes)
        
        # Create aggregated metadata
        aggregated_metadata = IssueMetadata(
            detected_by=primary_issue.metadata.detected_by,
            detection_model=f"aggregated_{len(all_issues)}_issues",
            contributing_passes=list(all_contributing_passes),
            related_claim_ids=list(set(
                claim_id 
                for issue in all_issues 
                for claim_id in issue.metadata.related_claim_ids
            )),
            related_citation_ids=list(set(
                citation_id 
                for issue in all_issues 
                for citation_id in issue.metadata.related_citation_ids
            )),
            processing_notes=[f"Aggregated from {len(all_issues)} similar issues"]
        )
        
        return UnifiedIssue(
            issue_type=primary_issue.issue_type,
            title=f"Aggregated: {primary_issue.title}",
            description=f"Combined issue from {len(all_issues)} similar detections: {primary_issue.description}",
            location=primary_issue.location,
            text_excerpt=primary_issue.text_excerpt,
            context=primary_issue.context,
            severity=severity,
            severity_score=max_severity_score,
            confidence_score=max_confidence,
            impact_score=max_impact,
            metadata=aggregated_metadata,
            evidence=list(set(all_evidence)),
            recommendations=list(set(all_recommendations)),
            source_issues=[issue.issue_id for issue in all_issues],
            is_aggregated=True,
            aggregation_method="similarity_based"
        )


# Conversion functions for existing issue types
def convert_logical_issue(logical_issue: LogicalIssue, verification_pass: VerificationPassType = VerificationPassType.LOGIC_ANALYSIS) -> UnifiedIssue:
    """Convert a LogicalIssue to UnifiedIssue."""
    # Map LogicalIssue to IssueType
    issue_type = IssueType.LOGICAL_FALLACY
    
    # Map severity score to severity enum
    if logical_issue.severity_score >= 0.8:
        severity = IssueSeverity.CRITICAL
    elif logical_issue.severity_score >= 0.6:
        severity = IssueSeverity.HIGH
    elif logical_issue.severity_score >= 0.4:
        severity = IssueSeverity.MEDIUM
    elif logical_issue.severity_score >= 0.2:
        severity = IssueSeverity.LOW
    else:
        severity = IssueSeverity.INFORMATIONAL
    
    location = IssueLocation(
        start_position=logical_issue.start_position,
        end_position=logical_issue.end_position
    )
    
    metadata = IssueMetadata(
        detected_by=verification_pass,
        related_claim_ids=logical_issue.affected_claims
    )
    
    return UnifiedIssue(
        issue_type=issue_type,
        title=logical_issue.title,
        description=logical_issue.description,
        location=location,
        text_excerpt=logical_issue.text_excerpt,
        context=logical_issue.context,
        severity=severity,
        severity_score=logical_issue.severity_score,
        confidence_score=logical_issue.confidence_score,
        impact_score=logical_issue.impact_score,
        metadata=metadata,
        recommendations=logical_issue.suggestions
    )


def convert_bias_issue(bias_issue: BiasIssue, verification_pass: VerificationPassType = VerificationPassType.BIAS_SCAN) -> UnifiedIssue:
    """Convert a BiasIssue to UnifiedIssue."""
    issue_type = IssueType.BIAS_DETECTION
    
    # Map bias severity to issue severity
    severity_mapping = {
        "critical": IssueSeverity.CRITICAL,
        "high": IssueSeverity.HIGH,
        "moderate": IssueSeverity.MEDIUM,
        "low": IssueSeverity.LOW,
        "minimal": IssueSeverity.INFORMATIONAL
    }
    severity = severity_mapping.get(bias_issue.severity.value, IssueSeverity.MEDIUM)
    
    location = IssueLocation(
        start_position=bias_issue.start_position,
        end_position=bias_issue.end_position
    )
    
    metadata = IssueMetadata(
        detected_by=verification_pass,
        related_claim_ids=bias_issue.affected_claims
    )
    
    return UnifiedIssue(
        issue_type=issue_type,
        title=bias_issue.title,
        description=bias_issue.description,
        location=location,
        text_excerpt=bias_issue.text_excerpt,
        severity=severity,
        severity_score=bias_issue.impact_score,  # Use impact as severity
        confidence_score=bias_issue.confidence_score,
        impact_score=bias_issue.impact_score,
        metadata=metadata,
        evidence=bias_issue.evidence,
        recommendations=bias_issue.mitigation_suggestions,
        alternative_interpretations=bias_issue.alternative_perspectives
    )


def convert_citation_issue(verified_citation: VerifiedCitation, verification_pass: VerificationPassType = VerificationPassType.CITATION_CHECK) -> List[UnifiedIssue]:
    """Convert citation issues to UnifiedIssues."""
    issues = []
    
    for citation_issue_type in verified_citation.identified_issues:
        # Map citation issue to general issue type
        issue_type_mapping = {
            CitationIssue.BROKEN_LINK: IssueType.CITATION_PROBLEM,
            CitationIssue.CONTENT_MISMATCH: IssueType.EVIDENCE_MISSING,
            CitationIssue.UNRELIABLE_SOURCE: IssueType.CREDIBILITY_CONCERN,
            CitationIssue.INSUFFICIENT_EVIDENCE: IssueType.EVIDENCE_MISSING,
            CitationIssue.FORMATTING_ERROR: IssueType.FORMATTING_ERROR,
            CitationIssue.PLAGIARISM_DETECTED: IssueType.PLAGIARISM
        }
        
        issue_type = issue_type_mapping.get(citation_issue_type, IssueType.CITATION_PROBLEM)
        
        # Determine severity based on citation confidence and issue type
        if verified_citation.confidence_score <= 0.3:
            severity = IssueSeverity.HIGH
        elif verified_citation.confidence_score <= 0.6:
            severity = IssueSeverity.MEDIUM
        else:
            severity = IssueSeverity.LOW
        
        # Critical issues get upgraded
        if citation_issue_type in [CitationIssue.PLAGIARISM_DETECTED, CitationIssue.UNRELIABLE_SOURCE]:
            severity = IssueSeverity.CRITICAL
        
        location = IssueLocation(
            start_position=verified_citation.location.start_position,
            end_position=verified_citation.location.end_position,
            line_number=verified_citation.location.line_number
        )
        
        metadata = IssueMetadata(
            detected_by=verification_pass,
            related_claim_ids=[verified_citation.claim_id],
            related_citation_ids=[verified_citation.citation_id]
        )
        
        unified_issue = UnifiedIssue(
            issue_type=issue_type,
            title=f"Citation Issue: {citation_issue_type.value.replace('_', ' ').title()}",
            description=f"Citation verification found: {citation_issue_type.value}",
            location=location,
            text_excerpt=verified_citation.citation_text,
            severity=severity,
            severity_score=1.0 - verified_citation.confidence_score,  # Invert confidence for severity
            confidence_score=verified_citation.confidence_score,
            impact_score=1.0 - verified_citation.confidence_score,
            metadata=metadata,
            recommendations=verified_citation.issue_descriptions
        )
        
        issues.append(unified_issue)
    
    return issues
