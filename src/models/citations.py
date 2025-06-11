"""
Citation verification models for representing citation analysis results.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class CitationStatus(str, Enum):
    """Status of citation verification."""
    VALID = "valid"
    INVALID = "invalid"
    PARTIALLY_VALID = "partially_valid"
    INACCESSIBLE = "inaccessible"
    NOT_FOUND = "not_found"
    PENDING = "pending"


class CitationType(str, Enum):
    """Types of citations."""
    ACADEMIC_PAPER = "academic_paper"
    BOOK = "book"
    WEBSITE = "website"
    NEWS_ARTICLE = "news_article"
    GOVERNMENT_DOCUMENT = "government_document"
    REPORT = "report"
    DATASET = "dataset"
    LEGAL_DOCUMENT = "legal_document"
    INTERVIEW = "interview"
    PERSONAL_COMMUNICATION = "personal_communication"
    OTHER = "other"


class SupportLevel(str, Enum):
    """Level of support a citation provides for a claim."""
    STRONG_SUPPORT = "strong_support"
    MODERATE_SUPPORT = "moderate_support"
    WEAK_SUPPORT = "weak_support"
    NO_SUPPORT = "no_support"
    CONTRADICTS = "contradicts"
    IRRELEVANT = "irrelevant"


class CitationIssue(str, Enum):
    """Types of issues found with citations."""
    BROKEN_LINK = "broken_link"
    CONTENT_MISMATCH = "content_mismatch"
    OUTDATED_SOURCE = "outdated_source"
    UNRELIABLE_SOURCE = "unreliable_source"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FORMATTING_ERROR = "formatting_error"
    MISSING_DETAILS = "missing_details"
    CIRCULAR_REFERENCE = "circular_reference"
    PLAGIARISM_DETECTED = "plagiarism_detected"


class SourceCredibility(BaseModel):
    """Assessment of source credibility."""
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Overall credibility score")
    domain_authority: Optional[float] = Field(None, ge=0.0, le=1.0, description="Domain authority score")
    author_expertise: Optional[float] = Field(None, ge=0.0, le=1.0, description="Author expertise score")
    publication_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Publication quality score")
    peer_review_status: Optional[bool] = Field(None, description="Whether source is peer-reviewed")
    publication_date: Optional[datetime] = Field(None, description="Original publication date")
    last_updated: Optional[datetime] = Field(None, description="Last update date")
    
    # Assessment details
    assessment_method: str = Field(..., description="Method used for credibility assessment")
    assessment_details: Dict[str, Any] = Field(default_factory=dict)
    

class CitationLocation(BaseModel):
    """Location information for where a citation appears."""
    start_position: int = Field(..., description="Character start position in document")
    end_position: int = Field(..., description="Character end position in document")
    page_number: Optional[int] = Field(None, description="Page number")
    section_type: Optional[str] = Field(None, description="Type of section")
    context_before: Optional[str] = Field(None, max_length=200, description="Text before citation")
    context_after: Optional[str] = Field(None, max_length=200, description="Text after citation")


class CitationContent(BaseModel):
    """Content retrieved from a citation source."""
    content_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    retrieved_content: str = Field(..., description="Content retrieved from source")
    content_snippet: Optional[str] = Field(None, description="Relevant snippet supporting/refuting claim")
    retrieval_method: str = Field(..., description="Method used to retrieve content")
    retrieval_timestamp: datetime = Field(default_factory=datetime.utcnow)
    retrieval_success: bool = Field(..., description="Whether content retrieval was successful")
    retrieval_errors: List[str] = Field(default_factory=list)


class VerifiedCitation(BaseModel):
    """A citation that has been verified for accuracy and relevance."""
    citation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Citation details
    citation_text: str = Field(..., description="Original citation text from document")
    formatted_citation: Optional[str] = Field(None, description="Properly formatted citation")
    url: Optional[str] = Field(None, description="URL if available")
    doi: Optional[str] = Field(None, description="DOI if available")
    isbn: Optional[str] = Field(None, description="ISBN if applicable")
    
    # Classification
    citation_type: CitationType = Field(..., description="Type of citation")
    
    # Related claim
    claim_id: str = Field(..., description="ID of claim this citation supports")
    claim_text: str = Field(..., description="Text of the claim")
    
    # Location information
    location: CitationLocation = Field(..., description="Where citation appears")
    document_id: str = Field(..., description="Source document ID")
    
    # Verification results
    verification_status: CitationStatus = Field(CitationStatus.PENDING, description="Verification status")
    support_level: SupportLevel = Field(..., description="How well citation supports claim")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in verification")
    
    # Content analysis
    retrieved_content: Optional[CitationContent] = Field(None, description="Content from citation source")
    source_credibility: Optional[SourceCredibility] = Field(None, description="Credibility assessment")
    
    # Issues and flags
    identified_issues: List[CitationIssue] = Field(default_factory=list)
    issue_descriptions: List[str] = Field(default_factory=list)
    requires_manual_review: bool = Field(False, description="Whether manual review is needed")
    
    # Processing metadata
    verified_by: str = Field(..., description="Model/method used for verification")
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_seconds: Optional[float] = Field(None, description="Time taken for verification")
    verification_metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationVerificationResult(BaseModel):
    """Result of citation verification process."""
    verification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(..., description="Source document ID")
    
    # Verified citations
    verified_citations: List[VerifiedCitation] = Field(default_factory=list)
    total_citations_found: int = Field(0, description="Total citations identified")
    total_citations_verified: int = Field(0, description="Total citations successfully verified")
    
    # Summary statistics
    valid_citations: int = Field(0, description="Number of valid citations")
    invalid_citations: int = Field(0, description="Number of invalid citations")
    inaccessible_citations: int = Field(0, description="Number of inaccessible citations")
    
    # Quality metrics
    average_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_credibility: Optional[float] = Field(None, ge=0.0, le=1.0)
    citation_accuracy_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Distribution analysis
    support_level_distribution: Dict[str, int] = Field(default_factory=dict)
    citation_type_distribution: Dict[str, int] = Field(default_factory=dict)
    issue_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Processing details
    model_used: str = Field(..., description="LLM model used for verification")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
    
    # Issues and warnings
    verification_warnings: List[str] = Field(default_factory=list)
    verification_errors: List[str] = Field(default_factory=list)
    
    # Claims analysis
    claims_with_citations: int = Field(0, description="Number of claims that have citations")
    claims_without_citations: int = Field(0, description="Number of claims lacking citations")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claim IDs lacking proper citation support")
    
    # Metadata
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    parameters_used: Dict[str, Any] = Field(default_factory=dict)


class CitationSearchQuery(BaseModel):
    """Query model for searching verified citations."""
    query_text: Optional[str] = Field(None, description="Text to search for")
    citation_types: Optional[List[CitationType]] = Field(None, description="Filter by citation types")
    verification_statuses: Optional[List[CitationStatus]] = Field(None, description="Filter by verification status")
    support_levels: Optional[List[SupportLevel]] = Field(None, description="Filter by support level")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    claim_ids: Optional[List[str]] = Field(None, description="Filter by claim IDs")
    has_issues: Optional[bool] = Field(None, description="Filter citations with/without issues")
    requires_review: Optional[bool] = Field(None, description="Filter citations requiring manual review")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Filter by verification date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class CitationReport(BaseModel):
    """Comprehensive report on citation verification results."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(..., description="Source document ID")
    
    # Summary
    total_claims: int = Field(0, description="Total claims analyzed")
    total_citations: int = Field(0, description="Total citations found")
    verification_summary: CitationVerificationResult = Field(..., description="Verification results")
    
    # Issues analysis
    critical_issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Credibility assessment
    overall_credibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_diversity: int = Field(0, description="Number of unique sources cited")
    peer_reviewed_percentage: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Generated at
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_metadata: Dict[str, Any] = Field(default_factory=dict) 