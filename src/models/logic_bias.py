"""
Models for logic analysis and bias detection verification passes.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class LogicalFallacyType(str, Enum):
    """Types of logical fallacies that can be detected."""
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_DILEMMA = "false_dilemma"
    SLIPPERY_SLOPE = "slippery_slope"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    APPEAL_TO_EMOTION = "appeal_to_emotion"
    APPEAL_TO_POPULARITY = "appeal_to_popularity"
    CIRCULAR_REASONING = "circular_reasoning"
    HASTY_GENERALIZATION = "hasty_generalization"
    POST_HOC = "post_hoc"
    FALSE_CAUSE = "false_cause"
    RED_HERRING = "red_herring"
    BEGGING_THE_QUESTION = "begging_the_question"
    EQUIVOCATION = "equivocation"
    COMPOSITION_FALLACY = "composition_fallacy"
    DIVISION_FALLACY = "division_fallacy"
    APPEAL_TO_IGNORANCE = "appeal_to_ignorance"
    BURDEN_OF_PROOF = "burden_of_proof"
    NO_TRUE_SCOTSMAN = "no_true_scotsman"
    TEXAS_SHARPSHOOTER = "texas_sharpshooter"
    BANDWAGON = "bandwagon"
    APPEAL_TO_TRADITION = "appeal_to_tradition"
    APPEAL_TO_NOVELTY = "appeal_to_novelty"
    GENETIC_FALLACY = "genetic_fallacy"
    TU_QUOQUE = "tu_quoque"
    LOADED_QUESTION = "loaded_question"
    MIDDLE_GROUND = "middle_ground"
    CHERRY_PICKING = "cherry_picking"
    ANECDOTAL_EVIDENCE = "anecdotal_evidence"


class ReasoningIssueType(str, Enum):
    """Types of reasoning issues beyond formal fallacies."""
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"
    INVALID_INFERENCE = "invalid_inference"
    CONTRADICTION = "contradiction"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    WEAK_EVIDENCE = "weak_evidence"
    MISSING_PREMISES = "missing_premises"
    UNJUSTIFIED_ASSUMPTION = "unjustified_assumption"
    OVERCONFIDENT_CLAIM = "overconfident_claim"
    SCOPE_CONFUSION = "scope_confusion"
    TEMPORAL_CONFUSION = "temporal_confusion"


class LogicalIssue(BaseModel):
    """Represents a logical issue found in the text."""
    issue_id: str = Field(description="Unique identifier for this issue")
    issue_type: str = Field(description="Type of logical issue (fallacy or reasoning error)")
    fallacy_type: Optional[LogicalFallacyType] = Field(None, description="Specific fallacy type if applicable")
    reasoning_type: Optional[ReasoningIssueType] = Field(None, description="Specific reasoning issue type if applicable")
    
    title: str = Field(description="Brief title of the issue")
    description: str = Field(description="Detailed description of the logical problem")
    explanation: str = Field(description="Explanation of why this is problematic")
    
    # Location information
    text_excerpt: str = Field(description="The specific text containing the issue")
    start_position: Optional[int] = Field(None, description="Character position where issue starts")
    end_position: Optional[int] = Field(None, description="Character position where issue ends")
    context: Optional[str] = Field(None, description="Surrounding context for the issue")
    
    # Assessment scores
    severity_score: float = Field(ge=0.0, le=1.0, description="How severe is this logical issue (0.0-1.0)")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the detection (0.0-1.0)")
    impact_score: float = Field(ge=0.0, le=1.0, description="Impact on overall argument validity (0.0-1.0)")
    
    # Additional metadata
    affected_claims: List[str] = Field(default_factory=list, description="Claims affected by this logical issue")
    related_issues: List[str] = Field(default_factory=list, description="IDs of related logical issues")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for addressing the issue")


class LogicAnalysisResult(BaseModel):
    """Result of logic analysis on a document."""
    document_id: str = Field(description="ID of the analyzed document")
    analysis_id: str = Field(description="Unique ID for this analysis")
    analyzed_at: datetime = Field(description="When the analysis was performed")
    
    # Overall assessment
    overall_logic_score: float = Field(ge=0.0, le=1.0, description="Overall logical quality score")
    total_issues_found: int = Field(description="Total number of logical issues detected")
    severity_distribution: Dict[str, int] = Field(
        default_factory=dict, 
        description="Distribution of issues by severity level"
    )
    
    # Detected issues
    logical_issues: List[LogicalIssue] = Field(
        default_factory=list, 
        description="List of logical issues found"
    )
    
    # Statistics and metadata
    fallacy_counts: Dict[LogicalFallacyType, int] = Field(
        default_factory=dict,
        description="Count of each type of fallacy detected"
    )
    reasoning_issue_counts: Dict[ReasoningIssueType, int] = Field(
        default_factory=dict,
        description="Count of each type of reasoning issue detected"
    )
    
    # Analysis parameters
    model_used: str = Field(description="LLM model used for analysis")
    prompt_version: str = Field(description="Version of prompt template used")
    analysis_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for analysis"
    )
    
    # Quality metrics
    average_confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Average confidence score across all detected issues"
    )
    average_severity: float = Field(
        ge=0.0, le=1.0,
        description="Average severity score across all detected issues"
    )
    
    # Warnings and errors
    analysis_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during analysis"
    )
    analysis_errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during analysis"
    )


class BiasType(str, Enum):
    """Types of bias that can be detected."""
    SELECTION_BIAS = "selection_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    CULTURAL_BIAS = "cultural_bias"
    DEMOGRAPHIC_BIAS = "demographic_bias"
    POLITICAL_BIAS = "political_bias"
    IDEOLOGICAL_BIAS = "ideological_bias"
    STATISTICAL_BIAS = "statistical_bias"
    FRAMING_BIAS = "framing_bias"
    ATTRIBUTION_BIAS = "attribution_bias"
    ANCHORING_BIAS = "anchoring_bias"
    AVAILABILITY_BIAS = "availability_bias"
    REPRESENTATIVENESS_BIAS = "representativeness_bias"
    RECENCY_BIAS = "recency_bias"
    PUBLICATION_BIAS = "publication_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    FUNDING_BIAS = "funding_bias"
    OBSERVER_BIAS = "observer_bias"
    REPORTING_BIAS = "reporting_bias"
    LANGUAGE_BIAS = "language_bias"
    TEMPORAL_BIAS = "temporal_bias"
    GEOGRAPHIC_BIAS = "geographic_bias"
    SOCIOECONOMIC_BIAS = "socioeconomic_bias"
    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    AGE_BIAS = "age_bias"


class BiasSeverity(str, Enum):
    """Severity levels for detected bias."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class BiasIssue(BaseModel):
    """Represents a bias issue found in the text."""
    issue_id: str = Field(description="Unique identifier for this bias issue")
    bias_type: BiasType = Field(description="Type of bias detected")
    
    title: str = Field(description="Brief title of the bias issue")
    description: str = Field(description="Detailed description of the bias")
    explanation: str = Field(description="Explanation of how this bias affects the content")
    
    # Location information
    text_excerpt: str = Field(description="The specific text exhibiting bias")
    start_position: Optional[int] = Field(None, description="Character position where bias starts")
    end_position: Optional[int] = Field(None, description="Character position where bias ends")
    context: Optional[str] = Field(None, description="Surrounding context for the bias")
    
    # Assessment scores
    severity: BiasSeverity = Field(description="Severity level of the bias")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the detection (0.0-1.0)")
    impact_score: float = Field(ge=0.0, le=1.0, description="Impact on content reliability (0.0-1.0)")
    
    # Evidence and examples
    evidence: List[str] = Field(
        default_factory=list,
        description="Specific evidence supporting the bias detection"
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Examples of the bias in the text"
    )
    
    # Mitigation
    affected_claims: List[str] = Field(
        default_factory=list,
        description="Claims affected by this bias"
    )
    mitigation_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for mitigating the bias"
    )
    alternative_perspectives: List[str] = Field(
        default_factory=list,
        description="Alternative perspectives that should be considered"
    )


class BiasAnalysisResult(BaseModel):
    """Result of bias analysis on a document."""
    document_id: str = Field(description="ID of the analyzed document")
    analysis_id: str = Field(description="Unique ID for this analysis")
    analyzed_at: datetime = Field(description="When the analysis was performed")
    
    # Overall assessment
    overall_bias_score: float = Field(ge=0.0, le=1.0, description="Overall bias level score (higher = more biased)")
    total_issues_found: int = Field(description="Total number of bias issues detected")
    severity_distribution: Dict[BiasSeverity, int] = Field(
        default_factory=dict,
        description="Distribution of issues by severity level"
    )
    
    # Detected issues
    bias_issues: List[BiasIssue] = Field(
        default_factory=list,
        description="List of bias issues found"
    )
    
    # Statistics and metadata
    bias_type_counts: Dict[BiasType, int] = Field(
        default_factory=dict,
        description="Count of each type of bias detected"
    )
    
    # Bias categories analysis
    political_leaning: Optional[str] = Field(None, description="Detected political leaning if any")
    demographic_representation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of demographic representation in the content"
    )
    source_diversity: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of source diversity and representation"
    )
    
    # Analysis parameters
    model_used: str = Field(description="LLM model used for analysis")
    prompt_version: str = Field(description="Version of prompt template used")
    analysis_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for analysis"
    )
    
    # Quality metrics
    average_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Average confidence score across all detected issues"
    )
    average_impact: float = Field(
        ge=0.0, le=1.0,
        description="Average impact score across all detected issues"
    )
    
    # Recommendations
    overall_recommendations: List[str] = Field(
        default_factory=list,
        description="Overall recommendations for addressing detected bias"
    )
    
    # Warnings and errors
    analysis_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during analysis"
    )
    analysis_errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during analysis"
    ) 