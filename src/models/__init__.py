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
    'VerificationMetrics'
]