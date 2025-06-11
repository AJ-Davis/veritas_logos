"""
Adversarial Cross-Validation Framework (ACVF) data models.

This module defines the data structures for the ACVF system where Challenger models
debate Defender models with Judge adjudication.
"""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import uuid


class ACVFRole(str, Enum):
    """Roles in the ACVF system."""
    CHALLENGER = "challenger"
    DEFENDER = "defender"
    JUDGE = "judge"


class DebateStatus(str, Enum):
    """Status of a debate round."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class JudgeVerdict(str, Enum):
    """Possible judge verdicts."""
    CHALLENGER_WINS = "challenger_wins"
    DEFENDER_WINS = "defender_wins"
    TIE = "tie"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_DEBATE = "invalid_debate"


class ConfidenceLevel(str, Enum):
    """Confidence levels for judgments."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


class ModelAssignment(BaseModel):
    """Assignment of a model to a specific role in the debate."""
    model_id: str = Field(description="Identifier for the LLM model")
    provider: str = Field(description="LLM provider (openai, anthropic, etc.)")
    role: ACVFRole = Field(description="Role assigned to this model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    system_prompt_override: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DebateArgument(BaseModel):
    """A single argument in the debate."""
    argument_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: ACVFRole = Field(description="Role of the model making this argument")
    content: str = Field(description="The argument content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_number: int = Field(ge=1, description="Round number when this argument was made")
    references: List[str] = Field(default_factory=list, description="References to claims, citations, or evidence")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JudgeScore(BaseModel):
    """Detailed scoring from a judge."""
    judge_id: str = Field(description="Identifier for the judge model")
    verdict: JudgeVerdict = Field(description="The judge's verdict")
    confidence: float = Field(ge=0.0, le=1.0, description="Judge's confidence in the verdict")
    confidence_level: ConfidenceLevel = Field(description="Categorical confidence level")
    
    # Detailed scoring breakdown
    challenger_score: float = Field(ge=0.0, le=1.0, description="Score for challenger arguments")
    defender_score: float = Field(ge=0.0, le=1.0, description="Score for defender arguments")
    
    # Qualitative assessments
    reasoning: str = Field(description="Judge's reasoning for the verdict")
    key_points_challenger: List[str] = Field(default_factory=list)
    key_points_defender: List[str] = Field(default_factory=list)
    critical_weaknesses: List[str] = Field(default_factory=list)
    
    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('confidence_level', mode='before')
    @classmethod
    def set_confidence_level(cls, v, info):
        """Automatically set confidence level based on numeric confidence."""
        if info.data and 'confidence' in info.data:
            confidence = info.data['confidence']
            if confidence < 0.2:
                return ConfidenceLevel.VERY_LOW
            elif confidence < 0.4:
                return ConfidenceLevel.LOW
            elif confidence < 0.6:
                return ConfidenceLevel.MEDIUM
            elif confidence < 0.8:
                return ConfidenceLevel.HIGH
            else:
                return ConfidenceLevel.VERY_HIGH
        return v


class DebateRound(BaseModel):
    """A complete debate round in the ACVF system."""
    round_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    verification_task_id: str = Field(description="ID of the verification task this debate relates to")
    subject_type: str = Field(description="Type of subject being debated (claim, citation, etc.)")
    subject_id: str = Field(description="ID of the specific subject being debated")
    subject_content: str = Field(description="Content of the subject being debated")
    
    # Model assignments
    challenger_model: ModelAssignment
    defender_model: ModelAssignment
    judge_models: List[ModelAssignment] = Field(min_length=1)
    
    # Debate flow
    status: DebateStatus = DebateStatus.PENDING
    round_number: int = Field(ge=1, description="Round number in the overall debate")
    max_rounds: int = Field(default=3, ge=1, le=10)
    arguments: List[DebateArgument] = Field(default_factory=list)
    
    # Results
    judge_scores: List[JudgeScore] = Field(default_factory=list)
    final_verdict: Optional[JudgeVerdict] = None
    consensus_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Configuration
    debate_config: Dict[str, Any] = Field(default_factory=dict)
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('judge_models')
    @classmethod
    def validate_judge_models(cls, v):
        """Ensure all judge models have the correct role."""
        for judge in v:
            if judge.role != ACVFRole.JUDGE:
                raise ValueError("All judge models must have role 'judge'")
        return v
    
    @field_validator('challenger_model')
    @classmethod
    def validate_challenger_model(cls, v):
        """Ensure challenger model has the correct role."""
        if v.role != ACVFRole.CHALLENGER:
            raise ValueError("Challenger model must have role 'challenger'")
        return v
    
    @field_validator('defender_model')
    @classmethod
    def validate_defender_model(cls, v):
        """Ensure defender model has the correct role."""
        if v.role != ACVFRole.DEFENDER:
            raise ValueError("Defender model must have role 'defender'")
        return v
    
    def add_argument(self, role: ACVFRole, content: str, round_number: int, 
                    references: Optional[List[str]] = None, confidence_score: Optional[float] = None) -> DebateArgument:
        """Add an argument to the debate."""
        argument = DebateArgument(
            role=role,
            content=content,
            round_number=round_number,
            references=references or [],
            confidence_score=confidence_score
        )
        self.arguments.append(argument)
        return argument
    
    def get_arguments_by_role(self, role: ACVFRole) -> List[DebateArgument]:
        """Get all arguments from a specific role."""
        return [arg for arg in self.arguments if arg.role == role]
    
    def get_arguments_by_round(self, round_number: int) -> List[DebateArgument]:
        """Get all arguments from a specific round."""
        return [arg for arg in self.arguments if arg.round_number == round_number]
    
    def is_complete(self) -> bool:
        """Check if the debate round is complete."""
        return self.status == DebateStatus.COMPLETED and self.final_verdict is not None
    
    def calculate_consensus(self) -> Optional[float]:
        """Calculate consensus confidence across all judges."""
        if not self.judge_scores:
            return None
        
        verdicts = [score.verdict for score in self.judge_scores]
        confidences = [score.confidence for score in self.judge_scores]
        
        # Find most common verdict
        verdict_counts = {}
        for verdict in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        if not verdict_counts:
            return 0.0
        
        max_count = max(verdict_counts.values())
        total_judges = len(self.judge_scores)
        
        # Calculate consensus as weighted average of confidence for majority verdict
        majority_verdicts = [v for v, count in verdict_counts.items() if count == max_count]
        
        if len(majority_verdicts) > 1:  # Tie between verdicts
            return 0.0
        
        majority_verdict = majority_verdicts[0]
        majority_confidences = [
            score.confidence for score in self.judge_scores 
            if score.verdict == majority_verdict
        ]
        
        # Weight by proportion of judges agreeing and average confidence
        agreement_ratio = max_count / total_judges
        avg_confidence = sum(majority_confidences) / len(majority_confidences)
        
        return agreement_ratio * avg_confidence


class ACVFConfiguration(BaseModel):
    """Configuration for ACVF debates."""
    config_id: str = Field(description="Unique identifier for this configuration")
    name: str = Field(description="Human-readable name for this configuration")
    description: Optional[str] = None
    
    # Model pools
    challenger_models: List[ModelAssignment] = Field(min_length=1)
    defender_models: List[ModelAssignment] = Field(min_length=1) 
    judge_models: List[ModelAssignment] = Field(min_length=1)
    
    # Debate parameters
    max_rounds_per_debate: int = Field(default=3, ge=1, le=10)
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    consensus_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    
    # When to trigger ACVF
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Advanced settings
    allow_model_self_assignment: bool = Field(default=False)
    require_unanimous_consensus: bool = Field(default=False)
    enable_meta_judging: bool = Field(default=False)  # Judges judging other judges
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('challenger_models')
    @classmethod
    def validate_challenger_models(cls, v):
        """Ensure challenger models have the correct roles."""
        for model in v:
            if model.role != ACVFRole.CHALLENGER:
                raise ValueError(f"All challenger_models must have role '{ACVFRole.CHALLENGER.value}'")
        return v
    
    @field_validator('defender_models')
    @classmethod
    def validate_defender_models(cls, v):
        """Ensure defender models have the correct roles."""
        for model in v:
            if model.role != ACVFRole.DEFENDER:
                raise ValueError(f"All defender_models must have role '{ACVFRole.DEFENDER.value}'")
        return v
    
    @field_validator('judge_models')
    @classmethod
    def validate_judge_models_config(cls, v):
        """Ensure judge models have the correct role."""
        for model in v:
            if model.role != ACVFRole.JUDGE:
                raise ValueError("All judge models must have role 'judge'")
        return v


class ACVFResult(BaseModel):
    """Final result from an ACVF debate session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    verification_task_id: str
    subject_type: str
    subject_id: str
    
    # All debate rounds
    debate_rounds: List[DebateRound] = Field(default_factory=list)
    
    # Final outcome
    final_verdict: Optional[JudgeVerdict] = None
    final_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    escalated: bool = Field(default=False)
    
    # Summary statistics
    total_rounds: int = Field(default=0, ge=0)
    total_arguments: int = Field(default=0, ge=0)
    average_judge_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    consensus_achieved: bool = Field(default=False)
    
    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Configuration used
    acvf_config_id: str
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_debate_round(self, debate_round: DebateRound):
        """Add a debate round to the session."""
        self.debate_rounds.append(debate_round)
        self.total_rounds = len(self.debate_rounds)
        self.total_arguments = sum(len(round.arguments) for round in self.debate_rounds)
    
    def calculate_final_metrics(self):
        """Calculate final metrics from all debate rounds."""
        if not self.debate_rounds:
            return
        
        # Calculate average judge confidence
        all_confidences = []
        for round in self.debate_rounds:
            for score in round.judge_scores:
                all_confidences.append(score.confidence)
        
        if all_confidences:
            self.average_judge_confidence = sum(all_confidences) / len(all_confidences)
        
        # Determine final verdict (could be from last round or overall consensus)
        if self.debate_rounds:
            last_round = self.debate_rounds[-1]
            if last_round.final_verdict:
                self.final_verdict = last_round.final_verdict
                self.final_confidence = last_round.consensus_confidence
                self.consensus_achieved = (last_round.consensus_confidence or 0) > 0.7
        
        # Set completion time
        if not self.completed_at:
            self.completed_at = datetime.now(timezone.utc)
            
        if self.started_at and self.completed_at:
            self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds() 