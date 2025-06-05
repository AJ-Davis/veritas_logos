"""
Verification models for the chain framework.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


class VerificationStatus(str, Enum):
    """Status of verification tasks and chains."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


class VerificationPassType(str, Enum):
    """Types of verification passes."""
    CLAIM_EXTRACTION = "claim_extraction"
    CITATION_CHECK = "citation_check"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    LOGIC_ANALYSIS = "logic_analysis"
    BIAS_SCAN = "bias_scan"
    ADVERSARIAL_VALIDATION = "adversarial_validation"


class Priority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Pydantic models for API serialization
class VerificationPassConfig(BaseModel):
    """Configuration for a single verification pass."""
    pass_type: VerificationPassType
    name: str
    description: Optional[str] = None
    enabled: bool = True
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 10
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)


class VerificationChainConfig(BaseModel):
    """Configuration for a complete verification chain."""
    chain_id: str
    name: str
    description: Optional[str] = None
    passes: List[VerificationPassConfig]
    global_timeout_seconds: int = 3600
    parallel_execution: bool = False
    stop_on_failure: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result from a single verification pass."""
    pass_id: str
    pass_type: VerificationPassType
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    result_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationChainResult(BaseModel):
    """Result from a complete verification chain execution."""
    chain_id: str
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    status: VerificationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_execution_time_seconds: Optional[float] = None
    pass_results: List[VerificationResult] = Field(default_factory=list)
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class VerificationTask(BaseModel):
    """A verification task to be processed."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chain_config: VerificationChainConfig
    priority: Priority = Priority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    status: VerificationStatus = VerificationStatus.PENDING
    result: Optional[VerificationChainResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Dataclasses for internal processing
@dataclass
class VerificationContext:
    """Context information passed between verification passes."""
    document_id: str
    document_content: str
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    chain_config: Optional[VerificationChainConfig] = None
    previous_results: List[VerificationResult] = field(default_factory=list)
    global_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_previous_result(self, pass_type: VerificationPassType) -> Optional[VerificationResult]:
        """Get the result from a previous pass by type."""
        for result in self.previous_results:
            if result.pass_type == pass_type:
                return result
        return None
    
    def get_all_results_by_type(self, pass_type: VerificationPassType) -> List[VerificationResult]:
        """Get all results from a specific pass type."""
        return [result for result in self.previous_results if result.pass_type == pass_type]


@dataclass 
class PassExecutionConfig:
    """Configuration for executing a verification pass."""
    pass_config: VerificationPassConfig
    context: VerificationContext
    max_retries: int = 3
    timeout_seconds: int = 300
    retry_delay_seconds: int = 10


# Exception classes
class VerificationError(Exception):
    """Base exception for verification errors."""
    pass


class VerificationTimeoutError(VerificationError):
    """Raised when verification pass times out."""
    pass


class VerificationConfigError(VerificationError):
    """Raised when there's an error in verification configuration."""
    pass


class PassDependencyError(VerificationError):
    """Raised when pass dependencies are not satisfied."""
    pass


# Metrics and monitoring
class VerificationMetrics(BaseModel):
    """Metrics for verification performance monitoring."""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time_seconds: float = 0.0
    pass_success_rates: Dict[str, float] = Field(default_factory=dict)
    error_frequencies: Dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)