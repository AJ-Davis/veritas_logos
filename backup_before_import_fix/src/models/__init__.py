"""Models package for document structures and data classes."""

from .document import (
    DocumentFormat,
    ExtractionMethod,
    DocumentMetadata,
    DocumentSection,
    ParsedDocument
)

from .verification import (
    VerificationStatus,
    VerificationPassType,
    Priority,
    VerificationPassConfig,
    VerificationChainConfig,
    VerificationResult,
    VerificationChainResult,
    VerificationTask,
    VerificationContext,
    PassExecutionConfig,
    VerificationError,
    VerificationTimeoutError,
    VerificationConfigError,
    PassDependencyError,
    VerificationMetrics
)

from .logic_bias import (
    LogicalFallacyType,
    ReasoningIssueType,
    LogicalIssue,
    LogicAnalysisResult,
    BiasType,
    BiasSeverity,
    BiasIssue,
    BiasAnalysisResult
)

from .acvf import (
    ACVFRole,
    DebateStatus,
    JudgeVerdict,
    ConfidenceLevel,
    ModelAssignment,
    DebateArgument,
    JudgeScore,
    DebateRound,
    ACVFConfiguration,
    ACVFResult
)

__all__ = [
    # Document models
    'DocumentFormat',
    'ExtractionMethod', 
    'DocumentMetadata',
    'DocumentSection',
    'ParsedDocument',
    
    # Verification models
    'VerificationStatus',
    'VerificationPassType',
    'Priority',
    'VerificationPassConfig',
    'VerificationChainConfig',
    'VerificationResult',
    'VerificationChainResult',
    'VerificationTask',
    'VerificationContext',
    'PassExecutionConfig',
    'VerificationError',
    'VerificationTimeoutError',
    'VerificationConfigError',
    'PassDependencyError',
    'VerificationMetrics',
    
    # Logic and bias models
    'LogicalFallacyType',
    'ReasoningIssueType',
    'LogicalIssue',
    'LogicAnalysisResult',
    'BiasType',
    'BiasSeverity',
    'BiasIssue',
    'BiasAnalysisResult',
    
    # ACVF models
    'ACVFRole',
    'DebateStatus',
    'JudgeVerdict',
    'ConfidenceLevel',
    'ModelAssignment',
    'DebateArgument',
    'JudgeScore',
    'DebateRound',
    'ACVFConfiguration',
    'ACVFResult'
]