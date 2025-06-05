"""
Verification worker for executing verification chains using Celery.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from celery import Celery
from celery.exceptions import Retry, WorkerLostError

from ...models.verification import (
    VerificationTask,
    VerificationChainResult,
    VerificationResult,
    VerificationContext,
    VerificationStatus,
    VerificationPassType,
    VerificationError
)
from ...models.document import ParsedDocument
from ...document_ingestion import document_service
from ..passes.base_pass import BaseVerificationPass, MockVerificationPass
from ..config.chain_loader import ChainConfigLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('verification_worker')
celery_app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    }
)


class VerificationWorker:
    """
    Core verification worker that orchestrates verification chain execution.
    """
    
    def __init__(self):
        """Initialize the verification worker."""
        self.config_loader = ChainConfigLoader()
        self.pass_registry: Dict[VerificationPassType, BaseVerificationPass] = {}
        self._initialize_pass_registry()
    
    def _initialize_pass_registry(self):
        """Initialize the registry of available verification passes."""
        # For now, register mock passes for all types
        # These will be replaced with actual implementations in subsequent tasks
        for pass_type in VerificationPassType:
            self.pass_registry[pass_type] = MockVerificationPass(pass_type)
        
        logger.info(f"Initialized {len(self.pass_registry)} verification passes")
    
    def register_pass(self, pass_type: VerificationPassType, pass_instance: BaseVerificationPass):
        """
        Register a verification pass implementation.
        
        Args:
            pass_type: Type of verification pass
            pass_instance: Implementation instance
        """
        self.pass_registry[pass_type] = pass_instance
        logger.info(f"Registered verification pass: {pass_type.value}")
    
    async def execute_verification_chain(self, verification_task: VerificationTask) -> VerificationChainResult:
        """
        Execute a complete verification chain.
        
        Args:
            verification_task: Task containing chain configuration and document info
            
        Returns:
            VerificationChainResult with results from all passes
        """
        start_time = time.time()
        chain_config = verification_task.chain_config
        
        # Initialize chain result
        chain_result = VerificationChainResult(
            chain_id=chain_config.chain_id,
            document_id=verification_task.document_id,
            status=VerificationStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Load document content
            document_content = await self._load_document_content(verification_task.document_id)
            
            # Create verification context
            context = VerificationContext(
                document_id=verification_task.document_id,
                document_content=document_content,
                chain_config=chain_config,
                global_context=verification_task.metadata
            )
            
            # Execute verification passes
            if chain_config.parallel_execution:
                chain_result.pass_results = await self._execute_passes_parallel(context, chain_config)
            else:
                chain_result.pass_results = await self._execute_passes_sequential(context, chain_config)
            
            # Calculate overall results
            chain_result = self._finalize_chain_result(chain_result, start_time)
            
        except Exception as e:
            logger.error(f"Chain execution failed: {str(e)}")
            chain_result.status = VerificationStatus.FAILED
            chain_result.errors.append(str(e))
            chain_result.completed_at = datetime.utcnow()
            chain_result.total_execution_time_seconds = time.time() - start_time
        
        return chain_result
    
    async def _load_document_content(self, document_id: str) -> str:
        """
        Load document content for verification.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document content as string
        """
        # This is a simplified implementation
        # In a real system, this would load from a document store
        
        # For now, assume document_id is a file path for demonstration
        try:
            parsed_doc = document_service.parse_document(document_id)
            if parsed_doc.is_valid:
                return parsed_doc.content
            else:
                raise VerificationError(f"Failed to parse document: {', '.join(parsed_doc.errors)}")
        except Exception as e:
            raise VerificationError(f"Failed to load document {document_id}: {str(e)}")
    
    async def _execute_passes_sequential(self, context: VerificationContext, 
                                       chain_config) -> List[VerificationResult]:
        """
        Execute verification passes sequentially, respecting dependencies.
        
        Args:
            context: Verification context
            chain_config: Chain configuration
            
        Returns:
            List of verification results
        """
        results = []
        executed_passes = set()
        
        # Sort passes by dependencies (topological sort)
        sorted_passes = self._sort_passes_by_dependencies(chain_config.passes)
        
        for pass_config in sorted_passes:
            if not pass_config.enabled:
                logger.info(f"Skipping disabled pass: {pass_config.name}")
                continue
            
            # Check dependencies
            if not self._check_pass_dependencies(pass_config, executed_passes):
                error_msg = f"Dependencies not satisfied for pass: {pass_config.name}"
                logger.error(error_msg)
                
                if chain_config.stop_on_failure:
                    raise VerificationError(error_msg)
                else:
                    # Create failed result and continue
                    failed_result = VerificationResult(
                        pass_id=f"{pass_config.name}_failed",
                        pass_type=pass_config.pass_type,
                        status=VerificationStatus.FAILED,
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                        error_message=error_msg
                    )
                    results.append(failed_result)
                    continue
            
            # Execute the pass
            try:
                # Update context with previous results
                context.previous_results = results.copy()
                
                # Get the pass implementation
                pass_impl = self.pass_registry.get(pass_config.pass_type)
                if not pass_impl:
                    raise VerificationError(f"No implementation found for pass type: {pass_config.pass_type}")
                
                # Execute with retries
                result = await self._execute_pass_with_retries(pass_impl, context, pass_config)
                results.append(result)
                executed_passes.add(pass_config.name)
                
                # Check if we should stop on failure
                if result.status == VerificationStatus.FAILED and chain_config.stop_on_failure:
                    logger.error(f"Stopping chain execution due to failed pass: {pass_config.name}")
                    break
                
            except Exception as e:
                error_msg = f"Pass {pass_config.name} failed with error: {str(e)}"
                logger.error(error_msg)
                
                # Create failed result
                failed_result = VerificationResult(
                    pass_id=f"{pass_config.name}_error",
                    pass_type=pass_config.pass_type,
                    status=VerificationStatus.FAILED,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    error_message=error_msg
                )
                results.append(failed_result)
                
                if chain_config.stop_on_failure:
                    break
        
        return results
    
    async def _execute_passes_parallel(self, context: VerificationContext,
                                     chain_config) -> List[VerificationResult]:
        """
        Execute verification passes in parallel where dependencies allow.
        
        Args:
            context: Verification context
            chain_config: Chain configuration
            
        Returns:
            List of verification results
        """
        # This is a simplified parallel implementation
        # A full implementation would build a dependency graph and execute in waves
        
        results = []
        tasks = []
        
        # For now, execute all enabled passes concurrently
        for pass_config in chain_config.passes:
            if pass_config.enabled:
                pass_impl = self.pass_registry.get(pass_config.pass_type)
                if pass_impl:
                    task = self._execute_pass_with_retries(pass_impl, context, pass_config)
                    tasks.append(task)
        
        # Wait for all tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_result = VerificationResult(
                        pass_id=f"parallel_error_{i}",
                        pass_type=chain_config.passes[i].pass_type,
                        status=VerificationStatus.FAILED,
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                        error_message=str(result)
                    )
                    processed_results.append(failed_result)
                else:
                    processed_results.append(result)
            
            results = processed_results
        
        return results
    
    async def _execute_pass_with_retries(self, pass_impl: BaseVerificationPass,
                                       context: VerificationContext,
                                       config) -> VerificationResult:
        """
        Execute a verification pass with retry logic.
        
        Args:
            pass_impl: Pass implementation
            context: Verification context
            config: Pass configuration
            
        Returns:
            VerificationResult
        """
        last_error = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying pass {config.name}, attempt {attempt + 1}")
                    await asyncio.sleep(config.retry_delay_seconds)
                
                result = await pass_impl.execute_with_timeout(context, config)
                
                if result.status != VerificationStatus.FAILED:
                    result.retry_count = attempt
                    return result
                
                last_error = result.error_message
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Pass {config.name} attempt {attempt + 1} failed: {last_error}")
        
        # All retries exhausted
        failed_result = VerificationResult(
            pass_id=f"{config.name}_max_retries",
            pass_type=config.pass_type,
            status=VerificationStatus.FAILED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            error_message=f"Max retries exceeded. Last error: {last_error}",
            retry_count=config.max_retries
        )
        
        return failed_result
    
    def _sort_passes_by_dependencies(self, passes) -> List:
        """
        Sort passes by their dependencies using topological sort.
        
        Args:
            passes: List of pass configurations
            
        Returns:
            Sorted list of passes
        """
        # Simple implementation - a full implementation would do proper topological sort
        pass_map = {p.name: p for p in passes}
        sorted_passes = []
        processed = set()
        
        def add_pass(pass_config):
            if pass_config.name in processed:
                return
            
            # Add dependencies first
            for dep_name in pass_config.depends_on:
                if dep_name in pass_map:
                    add_pass(pass_map[dep_name])
            
            sorted_passes.append(pass_config)
            processed.add(pass_config.name)
        
        for pass_config in passes:
            add_pass(pass_config)
        
        return sorted_passes
    
    def _check_pass_dependencies(self, pass_config, executed_passes: set) -> bool:
        """
        Check if pass dependencies are satisfied.
        
        Args:
            pass_config: Pass configuration
            executed_passes: Set of already executed pass names
            
        Returns:
            True if dependencies are satisfied
        """
        for dep_name in pass_config.depends_on:
            if dep_name not in executed_passes:
                return False
        return True
    
    def _finalize_chain_result(self, chain_result: VerificationChainResult, 
                             start_time: float) -> VerificationChainResult:
        """
        Finalize chain result with summary statistics.
        
        Args:
            chain_result: Chain result to finalize
            start_time: Chain start time
            
        Returns:
            Finalized chain result
        """
        chain_result.completed_at = datetime.utcnow()
        chain_result.total_execution_time_seconds = time.time() - start_time
        
        # Calculate overall status
        failed_count = sum(1 for r in chain_result.pass_results if r.status == VerificationStatus.FAILED)
        completed_count = sum(1 for r in chain_result.pass_results if r.status == VerificationStatus.COMPLETED)
        
        if failed_count == 0:
            chain_result.status = VerificationStatus.COMPLETED
        elif completed_count > 0:
            chain_result.status = VerificationStatus.COMPLETED  # Partial success
        else:
            chain_result.status = VerificationStatus.FAILED
        
        # Calculate overall confidence
        confidence_scores = [r.confidence_score for r in chain_result.pass_results 
                           if r.confidence_score is not None]
        if confidence_scores:
            chain_result.overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Generate summary
        chain_result.summary = {
            "total_passes": len(chain_result.pass_results),
            "completed_passes": completed_count,
            "failed_passes": failed_count,
            "average_confidence": chain_result.overall_confidence,
            "execution_time_seconds": chain_result.total_execution_time_seconds
        }
        
        return chain_result


# Global worker instance
verification_worker = VerificationWorker()


# Celery tasks
@celery_app.task(bind=True, name='verification.execute_chain')
def execute_verification_chain_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for executing verification chains.
    
    Args:
        task_data: Serialized verification task data
        
    Returns:
        Serialized verification chain result
    """
    try:
        # Deserialize task
        verification_task = VerificationTask(**task_data)
        
        # Execute chain asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                verification_worker.execute_verification_chain(verification_task)
            )
            return result.dict()
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Verification chain task failed: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(name='verification.health_check')
def health_check() -> Dict[str, Any]:
    """Health check task for monitoring worker status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker_id": "verification_worker",
        "registered_passes": len(verification_worker.pass_registry)
    }