"""
Integration tests for the unified verification pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.verification.pipeline import (
    VerificationPipeline,
    PipelineConfig,
    PipelineMode,
    AnalyzerConfig
)
from src.verification.passes.base_pass import BaseVerificationPass
from src.models.verification import (
    VerificationResult,
    VerificationChainResult,
    VerificationStatus,
    VerificationPassType
)


class MockVerificationPass(BaseVerificationPass):
    """Mock verification pass for testing."""
    
    def __init__(self, pass_type: VerificationPassType, should_fail: bool = False):
        super().__init__(pass_type)
        self.should_fail = should_fail
        self.call_count = 0
    
    async def execute(self, context) -> VerificationResult:
        """Mock execution that returns a test result."""
        self.call_count += 1
        
        if self.should_fail:
            status = VerificationStatus.FAILED
            confidence = 0.2
        else:
            status = VerificationStatus.COMPLETED
            confidence = 0.8
        
        return VerificationResult(
            pass_id=f"mock_{self.pass_type.value}",
            pass_type=self.pass_type,
            status=status,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            confidence_score=confidence,
            execution_time_seconds=1.0,
            result_data={
                "mock_data": f"Test result from {self.pass_type.value}",
                "call_count": self.call_count
            }
        )


@pytest.fixture
def mock_pass_registry():
    """Create a mock pass registry for testing."""
    return {
        VerificationPassType.CLAIM_EXTRACTION: MockVerificationPass(VerificationPassType.CLAIM_EXTRACTION),
        VerificationPassType.CITATION_CHECK: MockVerificationPass(VerificationPassType.CITATION_CHECK),
        VerificationPassType.LOGIC_ANALYSIS: MockVerificationPass(VerificationPassType.LOGIC_ANALYSIS),
        VerificationPassType.BIAS_SCAN: MockVerificationPass(VerificationPassType.BIAS_SCAN)
    }


@pytest.fixture
def basic_config():
    """Create a basic pipeline configuration for testing."""
    return PipelineConfig(
        mode=PipelineMode.STANDARD,
        parallel_execution=False,
        enable_caching=False  # Disable caching for tests
    )


@pytest.fixture
def pipeline_with_cache():
    """Create a pipeline configuration with caching enabled."""
    return PipelineConfig(
        mode=PipelineMode.STANDARD,
        enable_caching=True,
        cache_ttl_seconds=10,
        cache_size_limit=10
    )


class TestVerificationPipeline:
    """Test suite for the VerificationPipeline class."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_pass_registry, basic_config):
        """Test that the pipeline initializes correctly."""
        pipeline = VerificationPipeline(basic_config, mock_pass_registry)
        
        assert pipeline.config.mode == PipelineMode.STANDARD
        assert len(pipeline.pass_registry) == 4
        assert pipeline.cache is None  # Caching disabled
        assert PipelineMode.STANDARD in pipeline.adapters
        assert PipelineMode.ACVF not in pipeline.adapters  # Only standard mode
    
    @pytest.mark.asyncio
    async def test_standard_pipeline_execution(self, mock_pass_registry, basic_config):
        """Test execution of the standard pipeline."""
        pipeline = VerificationPipeline(basic_config, mock_pass_registry)
        
        test_content = "This is a test document with claims and citations."
        
        result = await pipeline.process_document(
            document_id="test_doc_1",
            content=test_content,
            metadata={"source": "test"}
        )
        
        # Verify result structure
        assert isinstance(result, VerificationChainResult)
        assert result.document_id == "test_doc_1"
        assert result.status in [VerificationStatus.COMPLETED, VerificationStatus.FAILED]
        assert result.total_execution_time_seconds > 0
        
        # Verify that all enabled passes were executed
        expected_passes = basic_config.enabled_passes
        result_pass_types = {r.pass_type for r in result.pass_results}
        assert result_pass_types == expected_passes
    
    @pytest.mark.asyncio
    async def test_pipeline_with_failed_passes(self, basic_config):
        """Test pipeline behavior when some passes fail."""
        # Create registry with one failing pass
        failing_registry = {
            VerificationPassType.CLAIM_EXTRACTION: MockVerificationPass(VerificationPassType.CLAIM_EXTRACTION),
            VerificationPassType.CITATION_CHECK: MockVerificationPass(VerificationPassType.CITATION_CHECK, should_fail=True),
            VerificationPassType.LOGIC_ANALYSIS: MockVerificationPass(VerificationPassType.LOGIC_ANALYSIS),
            VerificationPassType.BIAS_SCAN: MockVerificationPass(VerificationPassType.BIAS_SCAN)
        }
        
        pipeline = VerificationPipeline(basic_config, failing_registry)
        
        result = await pipeline.process_document(
            document_id="test_doc_2",
            content="Test content"
        )
        
        # Check that some passes succeeded and one failed
        failed_results = [r for r in result.pass_results if r.status == VerificationStatus.FAILED]
        successful_results = [r for r in result.pass_results if r.status == VerificationStatus.COMPLETED]
        
        assert len(failed_results) == 1
        assert len(successful_results) == 3
        assert failed_results[0].pass_type == VerificationPassType.CITATION_CHECK
    
    @pytest.mark.asyncio
    async def test_pipeline_caching(self, mock_pass_registry, pipeline_with_cache):
        """Test that caching works correctly."""
        pipeline = VerificationPipeline(pipeline_with_cache, mock_pass_registry)
        
        test_content = "This is a test document for caching."
        document_id = "cache_test_doc"
        
        # First execution - should miss cache
        result1 = await pipeline.process_document(document_id, test_content)
        
        # Verify passes were called
        for pass_instance in mock_pass_registry.values():
            assert pass_instance.call_count == 1
        
        # Second execution with same content - should hit cache
        result2 = await pipeline.process_document(document_id, test_content)
        
        # Verify passes were not called again (cached result used)
        for pass_instance in mock_pass_registry.values():
            assert pass_instance.call_count == 1  # Still 1, not 2
        
        # Results should be identical
        assert result1.document_id == result2.document_id
        assert len(result1.pass_results) == len(result2.pass_results)
    
    @pytest.mark.asyncio
    async def test_pipeline_cache_invalidation(self, mock_pass_registry, pipeline_with_cache):
        """Test cache invalidation when content changes."""
        pipeline = VerificationPipeline(pipeline_with_cache, mock_pass_registry)
        
        document_id = "invalidation_test_doc"
        
        # First execution
        result1 = await pipeline.process_document(document_id, "Original content")
        
        # Second execution with different content - should miss cache
        result2 = await pipeline.process_document(document_id, "Modified content")
        
        # Verify passes were called twice (once for each different content)
        for pass_instance in mock_pass_registry.values():
            assert pass_instance.call_count == 2
    
    @pytest.mark.asyncio
    async def test_hybrid_pipeline_mode(self, mock_pass_registry):
        """Test hybrid pipeline mode (standard + ACVF)."""
        config = PipelineConfig(
            mode=PipelineMode.HYBRID,
            enable_caching=False,
            acvf_trigger_conditions={
                "low_confidence_threshold": 0.5,
                "conflicting_results": True,
                "high_stakes_content": True
            }
        )
        
        # Create registry with low-confidence passes to trigger ACVF
        low_confidence_registry = {
            VerificationPassType.CLAIM_EXTRACTION: MockVerificationPass(VerificationPassType.CLAIM_EXTRACTION),
            VerificationPassType.CITATION_CHECK: MockVerificationPass(VerificationPassType.CITATION_CHECK, should_fail=True),
            VerificationPassType.LOGIC_ANALYSIS: MockVerificationPass(VerificationPassType.LOGIC_ANALYSIS),
            VerificationPassType.BIAS_SCAN: MockVerificationPass(VerificationPassType.BIAS_SCAN)
        }
        
        with patch('src.verification.acvf_controller.ACVFController') as mock_acvf:
            # Mock ACVF controller
            mock_acvf.return_value = MagicMock()
            
            pipeline = VerificationPipeline(config, low_confidence_registry)
            
            result = await pipeline.process_document(
                document_id="hybrid_test_doc",
                content="Test content for hybrid pipeline"
            )
            
            # Check that standard pipeline was executed
            standard_pass_types = {r.pass_type for r in result.pass_results 
                                 if r.pass_type != VerificationPassType.ACVF_ESCALATION}
            assert len(standard_pass_types) == 4
    
    @pytest.mark.asyncio
    async def test_pipeline_health_check(self, mock_pass_registry, basic_config):
        """Test pipeline health check functionality."""
        pipeline = VerificationPipeline(basic_config, mock_pass_registry)
        
        health = await pipeline.get_pipeline_health()
        
        assert "pipeline" in health
        assert "adapters" in health
        assert health["pipeline"]["mode"] == PipelineMode.STANDARD
        assert health["pipeline"]["status"] == "healthy"
        assert PipelineMode.STANDARD.value in health["adapters"]
    
    @pytest.mark.asyncio
    async def test_pipeline_shutdown(self, mock_pass_registry, basic_config):
        """Test pipeline shutdown process."""
        pipeline = VerificationPipeline(basic_config, mock_pass_registry)
        
        # Should not raise any exceptions
        await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_analyzer_configuration(self, mock_pass_registry):
        """Test that analyzer configurations are properly applied."""
        config = PipelineConfig(
            mode=PipelineMode.STANDARD,
            logic_analyzer=AnalyzerConfig(
                type="ml_enhanced",
                confidence_threshold=0.9,
                use_ensemble=True,
                parameters={"model_version": "v2.0"}
            ),
            bias_analyzer=AnalyzerConfig(
                type="basic",
                confidence_threshold=0.7,
                parameters={"language": "en"}
            )
        )
        
        pipeline = VerificationPipeline(config, mock_pass_registry)
        
        # Check that adapter was created with correct configuration
        standard_adapter = pipeline.adapters[PipelineMode.STANDARD]
        chain_config = standard_adapter.chain_config
        
        # Find logic analysis pass configuration
        logic_pass = next(
            (p for p in chain_config.passes if p.pass_type == VerificationPassType.LOGIC_ANALYSIS),
            None
        )
        
        assert logic_pass is not None
        assert logic_pass.parameters["analyzer_type"] == "ml_enhanced"
        assert logic_pass.parameters["confidence_threshold"] == 0.9
        assert logic_pass.parameters["use_ensemble"] is True
        assert logic_pass.parameters["model_version"] == "v2.0"
    
    @pytest.mark.asyncio
    async def test_weighted_scoring(self, mock_pass_registry):
        """Test weighted scoring functionality."""
        config = PipelineConfig(
            mode=PipelineMode.STANDARD,
            use_weighted_scoring=True,
            pass_weights={
                VerificationPassType.CLAIM_EXTRACTION: 1.0,
                VerificationPassType.CITATION_CHECK: 2.0,  # Higher weight
                VerificationPassType.LOGIC_ANALYSIS: 1.5,
                VerificationPassType.BIAS_SCAN: 1.0
            },
            enable_caching=False
        )
        
        pipeline = VerificationPipeline(config, mock_pass_registry)
        
        result = await pipeline.process_document(
            document_id="weighted_test_doc",
            content="Test content for weighted scoring"
        )
        
        # Verify that aggregator was created with weighted scoring
        assert pipeline.aggregator.scorer is not None
        assert pipeline.aggregator.config.use_weighted_scoring is True
        
        # Test aggregation
        aggregated = pipeline.aggregator.aggregate_results(result)
        
        assert "scoring" in aggregated
        assert "weighted_score" in aggregated["scoring"]
        assert "overall_score" in aggregated["scoring"] 