"""
Base verification pass interface.
"""

from abc import ABC, abstractmethod
import asyncio
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.models.verification import (
    VerificationPassConfig,
    VerificationResult,
    VerificationContext,
    VerificationStatus,
    VerificationPassType,
    VerificationError,
    VerificationTimeoutError
)


class BaseVerificationPass(ABC):
    """
    Abstract base class for all verification passes.
    
    This class defines the interface that all verification passes must implement
    and provides common functionality like timeout handling, error management,
    and result generation.
    """
    
    def __init__(self, pass_type: VerificationPassType):
        """
        Initialize the verification pass.
        
        Args:
            pass_type: The type of verification this pass performs
        """
        self.pass_type = pass_type
        self.logger = logging.getLogger(f"verification.{pass_type.value}")
    
    @abstractmethod
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """
        Execute the verification pass.
        
        This method must be implemented by all verification passes.
        
        Args:
            context: Context containing document and previous results
            config: Configuration for this specific pass execution
            
        Returns:
            VerificationResult containing the pass results
            
        Raises:
            VerificationError: If verification fails
            VerificationTimeoutError: If verification times out
        """
        pass
    
    @abstractmethod
    def get_required_dependencies(self) -> list[VerificationPassType]:
        """
        Get the list of verification pass types this pass depends on.
        
        Returns:
            List of required pass types
        """
        pass
    
    def can_execute(self, context: VerificationContext, config: VerificationPassConfig) -> bool:
        """
        Check if this pass can execute given the current context.
        
        Args:
            context: Current verification context
            config: Pass configuration
            
        Returns:
            True if pass can execute, False otherwise
        """
        # Check if all dependencies are satisfied
        required_deps = self.get_required_dependencies()
        
        for required_type in required_deps:
            if not context.get_previous_result(required_type):
                self.logger.warning(f"Missing required dependency: {required_type.value}")
                return False
        
        # Check if pass is enabled
        if not config.enabled:
            self.logger.info(f"Pass {config.name} is disabled")
            return False
        
        return True
    
    async def execute_with_timeout(self, context: VerificationContext, 
                                 config: VerificationPassConfig) -> VerificationResult:
        """
        Execute the verification pass with timeout and error handling.
        
        Args:
            context: Verification context
            config: Pass configuration
            
        Returns:
            VerificationResult with execution results
        """
        start_time = time.time()
        pass_id = f"{config.name}_{int(start_time)}"
        
        # Create initial result
        result = VerificationResult(
            pass_id=pass_id,
            pass_type=self.pass_type,
            status=VerificationStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Check if pass can execute
            if not self.can_execute(context, config):
                result.status = VerificationStatus.FAILED
                result.error_message = "Dependencies not satisfied or pass disabled"
                return result
            
            self.logger.info(f"Starting verification pass: {config.name}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute(context, config),
                timeout=config.timeout_seconds
            )
            
            # Update timing information
            execution_time = time.time() - start_time
            result.execution_time_seconds = execution_time
            result.completed_at = datetime.utcnow()
            
            if result.status == VerificationStatus.RUNNING:
                result.status = VerificationStatus.COMPLETED
            
            self.logger.info(f"Completed verification pass: {config.name} in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            result.status = VerificationStatus.FAILED
            result.error_message = f"Pass timed out after {config.timeout_seconds} seconds"
            result.execution_time_seconds = execution_time
            result.completed_at = datetime.utcnow()
            
            self.logger.error(f"Pass {config.name} timed out after {config.timeout_seconds}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.status = VerificationStatus.FAILED
            result.error_message = str(e)
            result.execution_time_seconds = execution_time
            result.completed_at = datetime.utcnow()
            
            self.logger.error(f"Pass {config.name} failed with error: {str(e)}")
        
        return result
    
    def validate_config(self, config: VerificationPassConfig) -> bool:
        """
        Validate the configuration for this pass.
        
        Args:
            config: Pass configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            VerificationError: If configuration is invalid
        """
        if config.pass_type != self.pass_type:
            raise VerificationError(
                f"Config pass type {config.pass_type} doesn't match pass type {self.pass_type}"
            )
        
        if config.timeout_seconds <= 0:
            raise VerificationError("Timeout must be positive")
        
        if config.max_retries < 0:
            raise VerificationError("Max retries cannot be negative")
        
        return True
    
    def extract_parameters(self, config: VerificationPassConfig) -> Dict[str, Any]:
        """
        Extract and validate parameters from configuration.
        
        Args:
            config: Pass configuration
            
        Returns:
            Dictionary of validated parameters
        """
        return config.parameters.copy()
    
    def create_result(self, pass_id: str, status: VerificationStatus = VerificationStatus.RUNNING,
                     result_data: Optional[Dict[str, Any]] = None,
                     confidence_score: Optional[float] = None,
                     error_message: Optional[str] = None) -> VerificationResult:
        """
        Helper method to create a verification result.
        
        Args:
            pass_id: Unique identifier for this pass execution
            status: Verification status
            result_data: Result data from the pass
            confidence_score: Confidence in the results
            error_message: Error message if failed
            
        Returns:
            VerificationResult instance
        """
        return VerificationResult(
            pass_id=pass_id,
            pass_type=self.pass_type,
            status=status,
            started_at=datetime.utcnow(),
            result_data=result_data or {},
            confidence_score=confidence_score,
            error_message=error_message
        )


class MockVerificationPass(BaseVerificationPass):
    """
    Mock verification pass for testing purposes.
    """
    
    def __init__(self, pass_type: VerificationPassType, delay: float = 0.1):
        """
        Initialize mock pass.
        
        Args:
            pass_type: Type of verification pass to mock
            delay: Artificial delay to simulate processing time
        """
        super().__init__(pass_type)
        self.delay = delay
    
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """Mock execution that simulates verification work."""
        
        # Simulate processing time
        await asyncio.sleep(self.delay)
        
        # Create mock result
        result_data = {
            "mock_result": True,
            "pass_type": self.pass_type.value,
            "document_id": context.document_id,
            "content_length": len(context.document_content),
            "parameters": config.parameters
        }
        
        return self.create_result(
            pass_id=f"mock_{self.pass_type.value}_{int(time.time())}",
            status=VerificationStatus.COMPLETED,
            result_data=result_data,
            confidence_score=0.8
        )
    
    def get_required_dependencies(self) -> list[VerificationPassType]:
        """Mock pass has no dependencies."""
        return []