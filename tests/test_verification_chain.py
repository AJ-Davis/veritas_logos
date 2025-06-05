"""
Tests for verification chain framework.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from src.verification import VerificationWorker, ChainConfigLoader, create_default_chain_configs
from src.models.verification import (
    VerificationTask,
    VerificationChainConfig,
    VerificationPassConfig,
    VerificationPassType,
    VerificationStatus,
    Priority
)


class TestVerificationChainFramework:
    """Test cases for the verification chain framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = VerificationWorker()
        self.config_loader = ChainConfigLoader()
    
    def test_worker_initialization(self):
        """Test that the worker initializes correctly."""
        assert len(self.worker.pass_registry) > 0
        
        # Check that all pass types have implementations
        for pass_type in VerificationPassType:
            assert pass_type in self.worker.pass_registry
    
    def test_default_chain_configs(self):
        """Test creation of default chain configurations."""
        configs = create_default_chain_configs()
        
        assert len(configs) >= 2
        assert 'standard_verification' in configs
        assert 'fast_verification' in configs
        
        # Validate structure
        standard_config = configs['standard_verification']
        assert 'chain_id' in standard_config
        assert 'name' in standard_config
        assert 'passes' in standard_config
        assert len(standard_config['passes']) > 0
    
    def test_chain_config_parsing(self):
        """Test parsing of chain configuration."""
        config_data = {
            'chain_id': 'test_chain',
            'name': 'Test Chain',
            'description': 'Test verification chain',
            'passes': [
                {
                    'type': 'claim_extraction',
                    'name': 'extract_claims',
                    'description': 'Extract claims',
                    'timeout_seconds': 300,
                    'parameters': {'model': 'gpt-4'}
                }
            ]
        }
        
        chain_config = VerificationChainConfig(**config_data)
        
        assert chain_config.chain_id == 'test_chain'
        assert chain_config.name == 'Test Chain'
        assert len(chain_config.passes) == 1
        assert chain_config.passes[0].pass_type == VerificationPassType.CLAIM_EXTRACTION
    
    def test_verification_task_creation(self):
        """Test creation of verification tasks."""
        # Create a simple chain config
        chain_config = VerificationChainConfig(
            chain_id='test_chain',
            name='Test Chain',
            passes=[
                VerificationPassConfig(
                    pass_type=VerificationPassType.CLAIM_EXTRACTION,
                    name='extract_claims'
                )
            ]
        )
        
        # Create verification task
        task = VerificationTask(
            document_id='test_document.txt',
            chain_config=chain_config,
            priority=Priority.HIGH
        )
        
        assert task.document_id == 'test_document.txt'
        assert task.chain_config.chain_id == 'test_chain'
        assert task.priority == Priority.HIGH
        assert task.status == VerificationStatus.PENDING
        assert task.task_id is not None
    
    @pytest.mark.asyncio
    async def test_mock_verification_execution(self):
        """Test execution of verification chain with mock passes."""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for verification.\n\nIt contains claims and citations.")
            test_doc_path = f.name
        
        try:
            # Create chain config with mock passes
            chain_config = VerificationChainConfig(
                chain_id='mock_test_chain',
                name='Mock Test Chain',
                passes=[
                    VerificationPassConfig(
                        pass_type=VerificationPassType.CLAIM_EXTRACTION,
                        name='extract_claims',
                        timeout_seconds=30
                    ),
                    VerificationPassConfig(
                        pass_type=VerificationPassType.CITATION_CHECK,
                        name='check_citations',
                        timeout_seconds=30,
                        depends_on=['extract_claims']
                    )
                ]
            )
            
            # Create verification task
            task = VerificationTask(
                document_id=test_doc_path,
                chain_config=chain_config
            )
            
            # Execute the chain
            result = await self.worker.execute_verification_chain(task)
            
            # Verify results
            assert result is not None
            assert result.chain_id == 'mock_test_chain'
            assert result.document_id == test_doc_path
            assert result.status in [VerificationStatus.COMPLETED, VerificationStatus.FAILED]
            assert len(result.pass_results) == 2
            assert result.total_execution_time_seconds is not None
            assert result.summary is not None
            
            # Check that passes executed in correct order
            pass_names = [r.pass_id.split('_')[0] for r in result.pass_results]
            # extract should come before check (due to dependency)
            extract_index = next(i for i, name in enumerate(pass_names) if 'extract' in name)
            check_index = next(i for i, name in enumerate(pass_names) if 'check' in name)
            assert extract_index < check_index
            
        finally:
            # Clean up
            os.unlink(test_doc_path)
    
    @pytest.mark.asyncio 
    async def test_parallel_execution(self):
        """Test parallel execution of verification passes."""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document for parallel verification.")
            test_doc_path = f.name
        
        try:
            # Create chain config with parallel execution
            chain_config = VerificationChainConfig(
                chain_id='parallel_test_chain',
                name='Parallel Test Chain',
                parallel_execution=True,
                passes=[
                    VerificationPassConfig(
                        pass_type=VerificationPassType.CLAIM_EXTRACTION,
                        name='extract_claims',
                        timeout_seconds=30
                    ),
                    VerificationPassConfig(
                        pass_type=VerificationPassType.BIAS_SCAN,
                        name='scan_bias',
                        timeout_seconds=30
                    )
                ]
            )
            
            task = VerificationTask(
                document_id=test_doc_path,
                chain_config=chain_config
            )
            
            # Execute the chain
            result = await self.worker.execute_verification_chain(task)
            
            # Verify results
            assert result is not None
            assert len(result.pass_results) == 2
            
        finally:
            os.unlink(test_doc_path)
    
    def test_dependency_validation(self):
        """Test validation of pass dependencies."""
        # Create chain config with invalid dependency
        with pytest.raises(Exception):  # Should raise validation error
            chain_config = VerificationChainConfig(
                chain_id='invalid_chain',
                name='Invalid Chain',
                passes=[
                    VerificationPassConfig(
                        pass_type=VerificationPassType.CITATION_CHECK,
                        name='check_citations',
                        depends_on=['nonexistent_pass']
                    )
                ]
            )
            self.config_loader._validate_chain_config(chain_config)
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        # This would need more sophisticated testing
        # For now, just test that the validation method exists
        passes = [
            VerificationPassConfig(
                pass_type=VerificationPassType.CLAIM_EXTRACTION,
                name='pass_a',
                depends_on=['pass_b']
            ),
            VerificationPassConfig(
                pass_type=VerificationPassType.CITATION_CHECK,
                name='pass_b',
                depends_on=['pass_a']
            )
        ]
        
        with pytest.raises(Exception):  # Should detect circular dependency
            self.config_loader._check_circular_dependencies(passes)


if __name__ == '__main__':
    pytest.main([__file__])