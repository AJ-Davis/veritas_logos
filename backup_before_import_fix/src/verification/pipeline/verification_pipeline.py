"""
Unified verification pipeline for orchestrating multiple verification passes.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ...models.verification import (
    VerificationTask,
    VerificationChainResult,
    VerificationResult,
    VerificationContext,
    VerificationStatus,
    VerificationPassType
)
from ...models.document import ParsedDocument
from ..passes.base_pass import BaseVerificationPass
from .adapters import IVerificationAdapter, StandardVerificationAdapter, ACVFAdapter
from .aggregators import ResultAggregator
from .cache import VerificationCache

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Pipeline execution modes."""
    STANDARD = "standard"
    ACVF = "acvf"
    HYBRID = "hybrid"


@dataclass
class AnalyzerConfig:
    """Configuration for analyzers."""
    type: str = "basic"  # "basic" or "ml_enhanced"
    confidence_threshold: float = 0.75
    use_ensemble: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for the verification pipeline."""
    
    # Pipeline settings
    mode: PipelineMode = PipelineMode.STANDARD
    enabled_passes: Set[VerificationPassType] = field(default_factory=lambda: {
        VerificationPassType.CLAIM_EXTRACTION,
        VerificationPassType.CITATION_CHECK,
        VerificationPassType.LOGIC_ANALYSIS,
        VerificationPassType.BIAS_SCAN
    })
    
    # Execution settings
    parallel_execution: bool = False
    stop_on_failure: bool = False
    timeout_seconds: int = 3600
    max_retries: int = 3
    
    # Analyzer configurations
    logic_analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    bias_analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    
    # Result aggregation
    confidence_threshold: float = 0.7
    use_weighted_scoring: bool = True
    pass_weights: Dict[VerificationPassType, float] = field(default_factory=lambda: {
        VerificationPassType.CLAIM_EXTRACTION: 1.0,
        VerificationPassType.CITATION_CHECK: 1.2,
        VerificationPassType.LOGIC_ANALYSIS: 1.1,
        VerificationPassType.BIAS_SCAN: 1.0
    })
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_size_limit: int = 1000
    
    # ACVF settings (when mode is ACVF or HYBRID)
    acvf_trigger_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "low_confidence_threshold": 0.6,
        "conflicting_results": True,
        "high_stakes_content": True
    })


class VerificationPipeline:
    """
    Unified verification pipeline that orchestrates multiple verification passes
    with configurable adapters for different execution modes.
    """
    
    def __init__(self, config: PipelineConfig, pass_registry: Dict[VerificationPassType, BaseVerificationPass]):
        """
        Initialize the verification pipeline.
        
        Args:
            config: Pipeline configuration
            pass_registry: Registry of available verification passes
        """
        self.config = config
        self.pass_registry = pass_registry
        self.cache = VerificationCache(
            ttl_seconds=config.cache_ttl_seconds,
            size_limit=config.cache_size_limit
        ) if config.enable_caching else None
        
        # Initialize adapters based on configuration
        self.adapters = self._initialize_adapters()
        self.aggregator = ResultAggregator(config)
        
        logger.info(f"Initialized VerificationPipeline with mode: {config.mode}")
        logger.info(f"Enabled passes: {[pt.value for pt in config.enabled_passes]}")
    
    def _initialize_adapters(self) -> Dict[PipelineMode, IVerificationAdapter]:
        """Initialize verification adapters for different modes."""
        adapters = {}
        
        if self.config.mode in [PipelineMode.STANDARD, PipelineMode.HYBRID]:
            adapters[PipelineMode.STANDARD] = StandardVerificationAdapter(
                self.pass_registry, self.config
            )
        
        if self.config.mode in [PipelineMode.ACVF, PipelineMode.HYBRID]:
            adapters[PipelineMode.ACVF] = ACVFAdapter(
                self.pass_registry, self.config
            )
        
        return adapters
    
    async def process_document(self, document_id: str, content: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> VerificationChainResult:
        """
        Process a document through the verification pipeline.
        
        Args:
            document_id: Unique document identifier
            content: Document content to verify
            metadata: Additional metadata for processing
            
        Returns:
            VerificationChainResult with aggregated results
        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Check cache first
        if self.cache:
            cached_result = await self.cache.get(document_id, content)
            if cached_result:
                logger.info(f"Using cached result for document {document_id}")
                return cached_result
        
        # Create verification context
        context = VerificationContext(
            document_id=document_id,
            document_content=content,
            chain_config=None,  # Will be set by adapter
            global_context=metadata
        )
        
        try:
            # Execute based on pipeline mode
            if self.config.mode == PipelineMode.STANDARD:
                result = await self._execute_standard_pipeline(context)
            elif self.config.mode == PipelineMode.ACVF:
                result = await self._execute_acvf_pipeline(context)
            elif self.config.mode == PipelineMode.HYBRID:
                result = await self._execute_hybrid_pipeline(context)
            else:
                raise ValueError(f"Unknown pipeline mode: {self.config.mode}")
            
            # Finalize result
            result.total_execution_time_seconds = time.time() - start_time
            result.completed_at = datetime.utcnow()
            
            # Cache the result
            if self.cache:
                await self.cache.set(document_id, content, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed for document {document_id}: {str(e)}")
            # Create failed result
            result = VerificationChainResult(
                chain_id=f"{self.config.mode}_pipeline",
                document_id=document_id,
                status=VerificationStatus.FAILED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                total_execution_time_seconds=time.time() - start_time,
                errors=[str(e)]
            )
            return result
    
    async def _execute_standard_pipeline(self, context: VerificationContext) -> VerificationChainResult:
        """Execute the standard verification pipeline."""
        adapter = self.adapters[PipelineMode.STANDARD]
        return await adapter.process_content(context)
    
    async def _execute_acvf_pipeline(self, context: VerificationContext) -> VerificationChainResult:
        """Execute the ACVF verification pipeline."""
        adapter = self.adapters[PipelineMode.ACVF]
        return await adapter.process_content(context)
    
    async def _execute_hybrid_pipeline(self, context: VerificationContext) -> VerificationChainResult:
        """Execute the hybrid pipeline (standard first, then ACVF if triggered)."""
        # First, run standard pipeline
        standard_result = await self._execute_standard_pipeline(context)
        
        # Check if ACVF should be triggered
        if self._should_trigger_acvf(standard_result):
            logger.info(f"Triggering ACVF for document {context.document_id}")
            
            # Run ACVF pipeline
            acvf_result = await self._execute_acvf_pipeline(context)
            
            # Merge results
            merged_result = self.aggregator.merge_pipeline_results(
                standard_result, acvf_result
            )
            return merged_result
        
        return standard_result
    
    def _should_trigger_acvf(self, result: VerificationChainResult) -> bool:
        """
        Determine if ACVF should be triggered based on standard pipeline results.
        
        Args:
            result: Result from standard pipeline
            
        Returns:
            True if ACVF should be triggered
        """
        conditions = self.config.acvf_trigger_conditions
        
        # Check low confidence
        if (conditions.get("low_confidence_threshold") and 
            result.overall_confidence and 
            result.overall_confidence < conditions["low_confidence_threshold"]):
            return True
        
        # Check for conflicting results
        if conditions.get("conflicting_results"):
            if self._has_conflicting_results(result):
                return True
        
        # Check for high-stakes content
        if conditions.get("high_stakes_content"):
            if self._is_high_stakes_content(result):
                return True
        
        return False
    
    def _has_conflicting_results(self, result: VerificationChainResult) -> bool:
        """Check if verification results conflict with each other."""
        if len(result.pass_results) < 2:
            return False
        
        # Simple heuristic: significant difference in confidence scores
        confidence_scores = [r.confidence_score for r in result.pass_results 
                           if r.confidence_score is not None]
        
        if len(confidence_scores) >= 2:
            confidence_range = max(confidence_scores) - min(confidence_scores)
            return confidence_range > 0.3  # Arbitrary threshold for "conflicting"
        
        return False
    
    def _is_high_stakes_content(self, result: VerificationChainResult) -> bool:
        """Check if content is considered high-stakes and needs additional verification."""
        # This is a placeholder - in practice, this would check for:
        # - Medical/health claims
        # - Financial advice
        # - Legal information
        # - Political statements
        # - Scientific claims
        
        # For now, trigger on any failed verification
        failed_passes = [r for r in result.pass_results 
                        if r.status == VerificationStatus.FAILED]
        return len(failed_passes) > 0
    
    async def get_pipeline_health(self) -> Dict[str, Any]:
        """Get health status of the pipeline and its components."""
        health = {
            "pipeline": {
                "mode": self.config.mode,
                "enabled_passes": [pt.value for pt in self.config.enabled_passes],
                "status": "healthy"
            },
            "adapters": {},
            "cache": None
        }
        
        # Check adapter health
        for mode, adapter in self.adapters.items():
            try:
                adapter_health = await adapter.get_health()
                health["adapters"][mode.value] = adapter_health
            except Exception as e:
                health["adapters"][mode.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check cache health
        if self.cache:
            health["cache"] = await self.cache.get_health()
        
        return health
    
    async def shutdown(self):
        """Shutdown the pipeline and cleanup resources."""
        logger.info("Shutting down VerificationPipeline")
        
        # Shutdown adapters
        for adapter in self.adapters.values():
            try:
                await adapter.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down adapter: {e}")
        
        # Shutdown cache
        if self.cache:
            await self.cache.shutdown()
        
        logger.info("VerificationPipeline shutdown complete") 