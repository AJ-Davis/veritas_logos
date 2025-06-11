"""
Verification adapters for different pipeline execution modes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...models.verification import (
    VerificationChainResult,
    VerificationResult,
    VerificationContext,
    VerificationStatus,
    VerificationPassType,
    VerificationChainConfig,
    VerificationPassConfig
)
from ..passes.base_pass import BaseVerificationPass
from ..workers.verification_worker import VerificationWorker

logger = logging.getLogger(__name__)


class IVerificationAdapter(ABC):
    """Interface for verification adapters."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the adapter."""
        pass
    
    @abstractmethod
    async def process_content(self, context: VerificationContext) -> VerificationChainResult:
        """Process content through the verification system."""
        pass
    
    @abstractmethod
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the adapter."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        pass


class StandardVerificationAdapter(IVerificationAdapter):
    """Adapter for standard verification pipeline using the existing VerificationWorker."""
    
    def __init__(self, pass_registry: Dict[VerificationPassType, BaseVerificationPass], config):
        """
        Initialize the standard verification adapter.
        
        Args:
            pass_registry: Registry of verification passes
            config: Pipeline configuration
        """
        self.pass_registry = pass_registry
        self.config = config
        self.worker = None
        self.chain_config = self._create_chain_config()
    
    def _create_chain_config(self) -> VerificationChainConfig:
        """Create a verification chain configuration for the enabled passes."""
        passes = []
        
        # Create pass configurations for enabled passes
        for pass_type in self.config.enabled_passes:
            pass_config = VerificationPassConfig(
                pass_type=pass_type,
                pass_id=f"{pass_type.value}_standard",
                name=f"{pass_type.value}_pass",
                description=f"Standard {pass_type.value} verification",
                enabled=True,
                timeout_seconds=300,
                max_retries=self.config.max_retries,
                parameters=self._get_pass_parameters(pass_type)
            )
            passes.append(pass_config)
        
        # Set up dependencies
        self._configure_dependencies(passes)
        
        return VerificationChainConfig(
            chain_id="standard_verification_pipeline",
            name="Standard Verification Pipeline",
            description="Unified pipeline for standard verification passes",
            passes=passes,
            global_timeout_seconds=self.config.timeout_seconds,
            parallel_execution=self.config.parallel_execution,
            stop_on_failure=self.config.stop_on_failure
        )
    
    def _get_pass_parameters(self, pass_type: VerificationPassType) -> Dict[str, Any]:
        """Get parameters for a specific pass type."""
        parameters = {}
        
        if pass_type == VerificationPassType.LOGIC_ANALYSIS:
            parameters.update({
                "analyzer_type": self.config.logic_analyzer.type,
                "confidence_threshold": self.config.logic_analyzer.confidence_threshold,
                "use_ensemble": self.config.logic_analyzer.use_ensemble
            })
            parameters.update(self.config.logic_analyzer.parameters)
        
        elif pass_type == VerificationPassType.BIAS_SCAN:
            parameters.update({
                "analyzer_type": self.config.bias_analyzer.type,
                "confidence_threshold": self.config.bias_analyzer.confidence_threshold,
                "use_ensemble": self.config.bias_analyzer.use_ensemble
            })
            parameters.update(self.config.bias_analyzer.parameters)
        
        return parameters
    
    def _configure_dependencies(self, passes: List[VerificationPassConfig]) -> None:
        """Configure dependencies between passes."""
        pass_by_type = {p.pass_type: p for p in passes}
        
        # Set up typical dependencies
        if VerificationPassType.CITATION_CHECK in pass_by_type:
            if VerificationPassType.CLAIM_EXTRACTION in pass_by_type:
                pass_by_type[VerificationPassType.CITATION_CHECK].depends_on = [
                    VerificationPassType.CLAIM_EXTRACTION
                ]
        
        if VerificationPassType.LOGIC_ANALYSIS in pass_by_type:
            if VerificationPassType.CLAIM_EXTRACTION in pass_by_type:
                pass_by_type[VerificationPassType.LOGIC_ANALYSIS].depends_on = [
                    VerificationPassType.CLAIM_EXTRACTION
                ]
        
        if VerificationPassType.BIAS_SCAN in pass_by_type:
            if VerificationPassType.CLAIM_EXTRACTION in pass_by_type:
                pass_by_type[VerificationPassType.BIAS_SCAN].depends_on = [
                    VerificationPassType.CLAIM_EXTRACTION
                ]
    
    async def initialize(self) -> None:
        """Initialize the adapter."""
        # Create a worker with our pass registry
        self.worker = VerificationWorker()
        
        # Register our passes
        for pass_type, pass_instance in self.pass_registry.items():
            if pass_type in self.config.enabled_passes:
                self.worker.register_pass(pass_type, pass_instance)
        
        logger.info("StandardVerificationAdapter initialized")
    
    async def process_content(self, context: VerificationContext) -> VerificationChainResult:
        """Process content through the standard verification pipeline."""
        if not self.worker:
            await self.initialize()
        
        # Set the chain config in the context
        context.chain_config = self.chain_config
        
        # Create a verification task
        from ...models.verification import VerificationTask, Priority
        verification_task = VerificationTask(
            document_id=context.document_id,
            chain_config=self.chain_config,
            priority=Priority.MEDIUM,
            metadata=context.global_context
        )
        
        # Execute the verification chain
        result = await self.worker.execute_verification_chain(verification_task)
        
        # Update chain ID to indicate this came from the pipeline
        result.chain_id = "standard_pipeline"
        
        return result
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the adapter."""
        health = {
            "status": "healthy",
            "enabled_passes": [pt.value for pt in self.config.enabled_passes],
            "chain_config": {
                "parallel_execution": self.config.parallel_execution,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
        
        if self.worker:
            health["registered_passes"] = len(self.worker.pass_registry)
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        # No specific cleanup needed for the standard adapter
        logger.info("StandardVerificationAdapter shutdown")


class ACVFAdapter(IVerificationAdapter):
    """Adapter for ACVF verification pipeline."""
    
    def __init__(self, pass_registry: Dict[VerificationPassType, BaseVerificationPass], config):
        """
        Initialize the ACVF adapter.
        
        Args:
            pass_registry: Registry of verification passes
            config: Pipeline configuration
        """
        self.pass_registry = pass_registry
        self.config = config
        self.acvf_controller = None
    
    async def initialize(self) -> None:
        """Initialize the adapter."""
        try:
            # Import ACVF controller
            from ...verification.acvf_controller import ACVFController
            self.acvf_controller = ACVFController()
            logger.info("ACVFAdapter initialized")
        except ImportError as e:
            logger.error(f"Failed to initialize ACVF controller: {e}")
            raise RuntimeError("ACVF system not available") from e
    
    async def process_content(self, context: VerificationContext) -> VerificationChainResult:
        """Process content through the ACVF verification pipeline."""
        if not self.acvf_controller:
            await self.initialize()
        
        # Create a basic result structure
        result = VerificationChainResult(
            chain_id="acvf_pipeline",
            document_id=context.document_id,
            status=VerificationStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # For now, we'll use a placeholder implementation
            # In a full implementation, this would:
            # 1. Extract claims from the document
            # 2. Run ACVF debates on controversial claims
            # 3. Aggregate results from the debates
            
            # Placeholder: run standard verification first, then ACVF on flagged content
            standard_adapter = StandardVerificationAdapter(self.pass_registry, self.config)
            await standard_adapter.initialize()
            standard_result = await standard_adapter.process_content(context)
            
            # If there are issues found, run ACVF debates
            if self._has_verification_issues(standard_result):
                logger.info(f"Running ACVF debates for document {context.document_id}")
                acvf_results = await self._run_acvf_debates(context, standard_result)
                
                # Merge standard and ACVF results
                result.pass_results = standard_result.pass_results + acvf_results
            else:
                result.pass_results = standard_result.pass_results
            
            result.status = VerificationStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            # Calculate overall confidence
            confidence_scores = [r.confidence_score for r in result.pass_results 
                               if r.confidence_score is not None]
            if confidence_scores:
                result.overall_confidence = sum(confidence_scores) / len(confidence_scores)
            
        except Exception as e:
            logger.error(f"ACVF processing failed: {e}")
            result.status = VerificationStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.utcnow()
        
        return result
    
    def _has_verification_issues(self, result: VerificationChainResult) -> bool:
        """Check if the standard verification found issues that warrant ACVF debate."""
        # Check for failed passes
        failed_passes = [r for r in result.pass_results 
                        if r.status == VerificationStatus.FAILED]
        if failed_passes:
            return True
        
        # Check for low confidence
        low_confidence_passes = [r for r in result.pass_results 
                               if r.confidence_score and r.confidence_score < 0.7]
        if low_confidence_passes:
            return True
        
        return False
    
    async def _run_acvf_debates(self, context: VerificationContext, 
                              standard_result: VerificationChainResult) -> List[VerificationResult]:
        """Run ACVF debates on problematic content."""
        # This is a placeholder implementation
        # In practice, this would:
        # 1. Identify specific claims/issues that need debate
        # 2. Configure challenger/defender/judge models
        # 3. Run the actual debates
        # 4. Return debate results as VerificationResult objects
        
        debate_results = []
        
        # For now, create a mock ACVF result
        acvf_result = VerificationResult(
            pass_id="acvf_debate_1",
            pass_type=VerificationPassType.ACVF_ESCALATION,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            confidence_score=0.8,
            execution_time_seconds=30.0,
            result_data={
                "debate_rounds": 3,
                "challenger_arguments": ["Claim lacks sufficient evidence"],
                "defender_arguments": ["Citation supports the claim"],
                "judge_verdict": "Defendant position upheld",
                "final_confidence": 0.8
            }
        )
        
        debate_results.append(acvf_result)
        return debate_results
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status of the adapter."""
        health = {
            "status": "healthy" if self.acvf_controller else "not_initialized",
            "acvf_controller": bool(self.acvf_controller)
        }
        
        if self.acvf_controller:
            # Add ACVF-specific health checks
            health["acvf_status"] = "available"
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.acvf_controller:
            # Shutdown ACVF controller if it has cleanup methods
            pass
        
        logger.info("ACVFAdapter shutdown") 