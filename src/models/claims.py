"""
Claim models for representing extracted claims from documents.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ClaimType(str, Enum):
    """Types of claims that can be extracted."""
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    DEFINITIONAL = "definitional"
    EVALUATIVE = "evaluative"
    POLICY = "policy"
    EXISTENTIAL = "existential"
    TEMPORAL = "temporal"


class ClaimCategory(str, Enum):
    """High-level categories for claims."""
    SCIENTIFIC = "scientific"
    HISTORICAL = "historical"
    POLITICAL = "political"
    ECONOMIC = "economic"
    SOCIAL = "social"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MEDICAL = "medical"
    ENVIRONMENTAL = "environmental"
    GENERAL = "general"


class ClaimEvidence(BaseModel):
    """Evidence supporting or refuting a claim."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The evidence text")
    source: Optional[str] = Field(None, description="Source of the evidence")
    location_in_document: Optional[Dict[str, Any]] = Field(None, description="Location where evidence appears")
    support_type: str = Field(..., description="Type of support: supporting, refuting, neutral")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in this evidence")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class ClaimLocation(BaseModel):
    """Location information for where a claim appears in a document."""
    start_position: int = Field(..., description="Character start position in document")
    end_position: int = Field(..., description="Character end position in document")
    page_number: Optional[int] = Field(None, description="Page number (for paginated documents)")
    section_type: Optional[str] = Field(None, description="Type of section (paragraph, heading, etc.)")
    section_index: Optional[int] = Field(None, description="Index of section in document")
    line_number: Optional[int] = Field(None, description="Line number in document")
    context_before: Optional[str] = Field(None, max_length=200, description="Text before the claim")
    context_after: Optional[str] = Field(None, max_length=200, description="Text after the claim")


class ExtractedClaim(BaseModel):
    """A claim extracted from a document with metadata."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core claim content
    claim_text: str = Field(..., description="The extracted claim text")
    normalized_claim: Optional[str] = Field(None, description="Normalized/cleaned version of the claim")
    
    # Classification
    claim_type: ClaimType = Field(..., description="Type of claim")
    category: ClaimCategory = Field(ClaimCategory.GENERAL, description="High-level category")
    
    # Location and context
    location: ClaimLocation = Field(..., description="Where the claim appears in the document")
    document_id: str = Field(..., description="ID of the source document")
    
    # Confidence and metadata
    extraction_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in claim extraction")
    clarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="How clearly the claim is stated")
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Estimated importance of the claim")
    
    # Relationships
    related_claims: List[str] = Field(default_factory=list, description="IDs of related claims")
    contradicts_claims: List[str] = Field(default_factory=list, description="IDs of contradicting claims")
    
    # Evidence and sources
    supporting_evidence: List[ClaimEvidence] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list, description="Citations associated with this claim")
    
    # Processing metadata
    extracted_by: str = Field(..., description="Name/ID of the extraction model/method")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Verification status
    verification_status: str = Field("pending", description="Status of claim verification")
    verification_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Flags and issues
    flagged_issues: List[str] = Field(default_factory=list, description="Any issues flagged with this claim")
    requires_fact_check: bool = Field(True, description="Whether this claim needs fact-checking")
    complexity_level: Optional[str] = Field(None, description="Simple, moderate, complex")


class ClaimExtractionResult(BaseModel):
    """Result of claim extraction process for a document."""
    document_id: str = Field(..., description="ID of the processed document")
    extraction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Extracted claims
    claims: List[ExtractedClaim] = Field(default_factory=list)
    total_claims_found: int = Field(0, description="Total number of claims extracted")
    
    # Processing statistics
    processing_time_seconds: Optional[float] = Field(None, description="Time taken for extraction")
    model_used: str = Field(..., description="LLM model used for extraction")
    prompt_version: str = Field(..., description="Version of extraction prompt used")
    
    # Quality metrics
    average_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    claim_type_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Document analysis
    document_length: int = Field(..., description="Length of source document in characters")
    claims_per_1000_chars: Optional[float] = Field(None, description="Claim density metric")
    
    # Issues and warnings
    extraction_warnings: List[str] = Field(default_factory=list)
    extraction_errors: List[str] = Field(default_factory=list)
    
    # Metadata
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    parameters_used: Dict[str, Any] = Field(default_factory=dict)


class ClaimValidationResult(BaseModel):
    """Result of validating extracted claims."""
    claim_id: str = Field(..., description="ID of the validated claim")
    is_valid: bool = Field(..., description="Whether the claim is valid")
    validation_score: float = Field(..., ge=0.0, le=1.0, description="Validation confidence score")
    
    validation_issues: List[str] = Field(default_factory=list)
    suggested_improvements: List[str] = Field(default_factory=list)
    
    validated_by: str = Field(..., description="Validator model/method")
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class ClaimSearchQuery(BaseModel):
    """Query model for searching extracted claims."""
    query_text: Optional[str] = Field(None, description="Text to search for")
    claim_types: Optional[List[ClaimType]] = Field(None, description="Filter by claim types")
    categories: Optional[List[ClaimCategory]] = Field(None, description="Filter by categories")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Filter by extraction date")
    has_citations: Optional[bool] = Field(None, description="Filter claims with/without citations")
    verification_status: Optional[str] = Field(None, description="Filter by verification status")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class ClaimStatistics(BaseModel):
    """Statistics about extracted claims."""
    total_claims: int = Field(0)
    claims_by_type: Dict[str, int] = Field(default_factory=dict)
    claims_by_category: Dict[str, int] = Field(default_factory=dict)
    claims_by_confidence_range: Dict[str, int] = Field(default_factory=dict)
    claims_by_document: Dict[str, int] = Field(default_factory=dict)
    
    average_confidence: float = Field(0.0)
    average_claims_per_document: float = Field(0.0)
    most_common_claim_type: Optional[str] = Field(None)
    most_common_category: Optional[str] = Field(None)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)