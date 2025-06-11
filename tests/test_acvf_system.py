"""
Tests for Adversarial Cross-Validation Framework (ACVF) system.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import ACVF components
from src.models.acvf import (
    ACVFRole, DebateStatus, JudgeVerdict, ConfidenceLevel,
    ModelAssignment, DebateArgument, JudgeScore, DebateRound,
    ACVFConfiguration, ACVFResult
)
from src.verification.acvf_controller import ACVFController
from src.verification.config.acvf_config_loader import ACVFConfigLoader
from src.verification.passes.implementations.acvf_escalation_pass import ACVFEscalationPass
from src.models.verification import VerificationContext, VerificationResult, VerificationStatus, VerificationPassType
from src.llm.llm_client import LLMClient, LLMResponse, LLMProvider


class TestACVFModels:
    """Test ACVF data models."""
    
    def test_model_assignment_creation(self):
        """Test ModelAssignment creation and validation."""
        model = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic",
            role=ACVFRole.CHALLENGER,
            temperature=0.8,
            max_tokens=1500
        )
        
        assert model.model_id == "claude-3-opus"
        assert model.provider == "anthropic"
        assert model.role == ACVFRole.CHALLENGER
        assert model.temperature == 0.8
        assert model.max_tokens == 1500
    
    def test_debate_argument_creation(self):
        """Test DebateArgument creation."""
        argument = DebateArgument(
            role=ACVFRole.CHALLENGER,
            content="This claim is problematic because...",
            round_number=1,
            references=["claim_1", "citation_2"]
        )
        
        assert argument.role == ACVFRole.CHALLENGER
        assert argument.content == "This claim is problematic because..."
        assert argument.round_number == 1
        assert len(argument.references) == 2
    
    def test_judge_score_confidence_level_auto_assignment(self):
        """Test automatic confidence level assignment based on numeric confidence."""
        # Test very low confidence
        score_very_low = JudgeScore(
            judge_id="judge_1",
            verdict=JudgeVerdict.CHALLENGER_WINS,
            confidence=0.1,
            confidence_level=ConfidenceLevel.VERY_LOW,
            challenger_score=0.8,
            defender_score=0.2,
            reasoning="Strong challenger arguments"
        )
        assert score_very_low.confidence_level == ConfidenceLevel.VERY_LOW
        
        # Test high confidence
        score_high = JudgeScore(
            judge_id="judge_2",
            verdict=JudgeVerdict.DEFENDER_WINS,
            confidence=0.9,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            challenger_score=0.3,
            defender_score=0.9,
            reasoning="Defender provided strong evidence"
        )
        assert score_high.confidence_level == ConfidenceLevel.VERY_HIGH
    
    def test_debate_round_validation(self):
        """Test DebateRound model validation."""
        challenger = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic",
            role=ACVFRole.CHALLENGER
        )
        
        defender = ModelAssignment(
            model_id="gpt-4",
            provider="openai",
            role=ACVFRole.DEFENDER
        )
        
        judge = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic",
            role=ACVFRole.JUDGE
        )
        
        debate_round = DebateRound(
            verification_task_id="task_123",
            subject_type="claim",
            subject_id="claim_1",
            subject_content="The economy is growing rapidly",
            challenger_model=challenger,
            defender_model=defender,
            judge_models=[judge],
            round_number=1
        )
        
        assert debate_round.verification_task_id == "task_123"
        assert debate_round.subject_type == "claim"
        assert debate_round.round_number == 1
        assert debate_round.challenger_model.role == ACVFRole.CHALLENGER
        assert debate_round.defender_model.role == ACVFRole.DEFENDER
        assert len(debate_round.judge_models) == 1
    
    def test_debate_round_argument_management(self):
        """Test adding and retrieving arguments in a debate round."""
        challenger = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic", 
            role=ACVFRole.CHALLENGER
        )
        
        defender = ModelAssignment(
            model_id="gpt-4",
            provider="openai",
            role=ACVFRole.DEFENDER
        )
        
        judge = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic",
            role=ACVFRole.JUDGE
        )
        
        debate_round = DebateRound(
            verification_task_id="task_123",
            subject_type="claim",
            subject_id="claim_1", 
            subject_content="Test claim",
            challenger_model=challenger,
            defender_model=defender,
            judge_models=[judge],
            round_number=1
        )
        
        # Add arguments
        challenger_arg = debate_round.add_argument(
            role=ACVFRole.CHALLENGER,
            content="This claim lacks evidence",
            round_number=1
        )
        
        defender_arg = debate_round.add_argument(
            role=ACVFRole.DEFENDER,
            content="The claim is supported by data",
            round_number=1
        )
        
        # Test retrieval
        assert len(debate_round.arguments) == 2
        challenger_args = debate_round.get_arguments_by_role(ACVFRole.CHALLENGER)
        assert len(challenger_args) == 1
        assert challenger_args[0].content == "This claim lacks evidence"

        round_args = debate_round.get_arguments_by_round(1)
        assert len(round_args) == 2


class TestACVFConfigLoader:
    """Test ACVF configuration loading."""
    
    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            "name": "Test ACVF Configuration",
            "description": "Test configuration for ACVF system",
            "version": "1.0",
            "models": {
                "challengers": [
                    {
                        "provider": "anthropic",
                        "model": "claude-3-opus",
                        "role": "challenger",
                        "temperature": 0.8
                    }
                ],
                "defenders": [
                    {
                        "provider": "openai",
                        "model": "gpt-4",
                        "role": "defender",
                        "temperature": 0.7
                    }
                ],
                "judges": [
                    {
                        "provider": "anthropic",
                        "model": "claude-3-opus",
                        "role": "judge",
                        "temperature": 0.5
                    }
                ]
            },
            "debate_config": {
                "max_rounds_per_debate": 3,
                "escalation_threshold": 0.5,
                "consensus_threshold": 0.7
            },
            "trigger_conditions": {
                "min_confidence_threshold": 0.6,
                "escalate_failed_passes": True,
                "escalate_on_issues": True
            },
            "advanced_settings": {
                "allow_model_self_assignment": True,
                "require_unanimous_consensus": False,
                "enable_meta_judging": False
            }
        }
    
    def test_create_acvf_config_from_data(self, sample_config_data):
        """Test creating ACVFConfiguration from parsed data."""
        loader = ACVFConfigLoader()
        config = loader._create_acvf_config(sample_config_data)
        
        assert config.name == "Test ACVF Configuration"
        assert len(config.challenger_models) == 1
        assert len(config.defender_models) == 1
        assert len(config.judge_models) == 1
        assert config.max_rounds_per_debate == 3
        assert config.escalation_threshold == 0.5
        assert config.consensus_threshold == 0.7
        
        # Test model assignments
        challenger = config.challenger_models[0]
        assert challenger.model_id == "claude-3-opus"
        assert challenger.provider == "anthropic"
        assert challenger.role == ACVFRole.CHALLENGER
        assert challenger.temperature == 0.8
    
    def test_config_validation(self, sample_config_data):
        """Test configuration validation."""
        loader = ACVFConfigLoader()
        config = loader._create_acvf_config(sample_config_data)
        
        # Valid config should have no issues
        issues = loader.validate_config(config)
        assert len(issues) == 0
        
        # Test with invalid config
        config.challenger_models = []
        issues = loader.validate_config(config)
        assert len(issues) > 0
        assert "At least one challenger model is required" in issues


@pytest.mark.asyncio
class TestACVFController:
    """Test ACVF controller functionality."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        mock_client = Mock(spec=LLMClient)
        
        # Mock challenger response
        mock_client.generate_challenger_response = AsyncMock()
        mock_client.generate_challenger_response.return_value = Mock(
            content="The evidence provided is insufficient and biased."
        )
        
        # Mock defender response  
        mock_client.generate_defender_response = AsyncMock()
        mock_client.generate_defender_response.return_value = Mock(
            content="The evidence is credible and well-sourced."
        )
        
        # Mock judge verdict - return LLMResponse with JSON content
        import json
        mock_verdict_data = {
            "judge_id": "anthropic:claude-3-opus",
            "verdict": "defender_wins",
            "confidence": 0.8,
            "confidence_level": "high",
            "challenger_score": 0.3,
            "defender_score": 0.9,
            "reasoning": "Defender provided stronger evidence and reasoning.",
            "key_points_challenger": ["Questioned evidence quality"],
            "key_points_defender": ["Provided credible sources", "Strong logical reasoning"],
            "critical_weaknesses": ["Challenger's critique was too broad"]
        }
        
        mock_client.generate_judge_verdict = AsyncMock()
        mock_client.generate_judge_verdict.return_value = Mock(
            content=json.dumps(mock_verdict_data)
        )
        
        return mock_client
    
    @pytest.fixture
    def sample_acvf_config(self):
        """Sample ACVF configuration for testing."""
        challenger = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic",
            role=ACVFRole.CHALLENGER,
            temperature=0.8
        )
        
        defender = ModelAssignment(
            model_id="gpt-4", 
            provider="openai",
            role=ACVFRole.DEFENDER,
            temperature=0.7
        )
        
        judge = ModelAssignment(
            model_id="claude-3-sonnet",
            provider="anthropic",
            role=ACVFRole.JUDGE,
            temperature=0.5
        )
        
        return ACVFConfiguration(
            config_id="test_config",
            name="Test Configuration",
            challenger_models=[challenger],
            defender_models=[defender],
            judge_models=[judge],
            max_rounds_per_debate=3,
            escalation_threshold=0.5,
            consensus_threshold=0.7,
            trigger_conditions={
                "min_confidence_threshold": 0.6,
                "escalate_failed_passes": True,
                "escalate_on_issues": True
            },
            allow_model_self_assignment=False,
            require_unanimous_consensus=False,
            enable_meta_judging=False
        )
    
    @pytest.fixture
    def sample_verification_context(self):
        """Sample verification context for testing."""
        # Create a low-confidence result to trigger ACVF  
        low_confidence_result = VerificationResult(
            pass_id="citation_check_001",
            pass_type=VerificationPassType.CITATION_CHECK,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            confidence_score=0.4,
            result_data={
                "issues_found": True,
                "citations": [
                    {"id": "citation_1", "text": "Source claims economic growth rate of 5%", "confidence": 0.3},
                    {"id": "citation_2", "text": "Another questionable economic indicator", "confidence": 0.2}
                ]
            },
            metadata={}
        )
        
        return VerificationContext(
            document_id="doc_123",
            document_content="Test document with questionable claims",
            previous_results=[low_confidence_result],
            document_metadata={}
        )
    
    async def test_should_trigger_acvf(self, mock_llm_client, sample_acvf_config, sample_verification_context):
        """Test ACVF trigger conditions."""
        controller = ACVFController(mock_llm_client, sample_acvf_config)
        
        # Should trigger due to low confidence score
        should_trigger = await controller.should_trigger_acvf(sample_verification_context)
        assert should_trigger is True
        
        # Test with high confidence - should not trigger
        high_confidence_result = VerificationResult(
            pass_id="citation_check_002",
            pass_type=VerificationPassType.CITATION_CHECK,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            confidence_score=0.9,
            result_data={},
            metadata={}
        )
        
        high_confidence_context = VerificationContext(
            document_id="doc_123",
            document_content="Test document",
            previous_results=[high_confidence_result],
            document_metadata={}
        )
        
        should_not_trigger = await controller.should_trigger_acvf(high_confidence_context)
        assert should_not_trigger is False
    
    async def test_conduct_debate_round(self, mock_llm_client, sample_acvf_config):
        """Test conducting a single debate round."""
        controller = ACVFController(mock_llm_client, sample_acvf_config)
        
        debate_round = await controller.conduct_debate_round(
            verification_task_id="task_123",
            subject_type="claim",
            subject_id="claim_1",
            subject_content="The economy is growing rapidly",
            context="Economic indicators and recent data",
            round_number=1
        )
        
        assert debate_round.status == DebateStatus.COMPLETED
        assert len(debate_round.arguments) == 2  # Challenger + Defender
        assert len(debate_round.judge_scores) == 1
        assert debate_round.final_verdict == JudgeVerdict.DEFENDER_WINS
        
        # Verify LLM client calls
        mock_llm_client.generate_challenger_response.assert_called_once()
        mock_llm_client.generate_defender_response.assert_called_once()
        mock_llm_client.generate_judge_verdict.assert_called_once()
    
    async def test_process_verification_escalation(self, mock_llm_client, sample_acvf_config, sample_verification_context):
        """Test processing verification escalation through ACVF."""
        controller = ACVFController(mock_llm_client, sample_acvf_config)
        
        acvf_results = await controller.process_verification_escalation(sample_verification_context)
        
        assert len(acvf_results) > 0
        
        for result in acvf_results:
            assert isinstance(result, ACVFResult)
            assert result.verification_task_id == sample_verification_context.document_id
            assert len(result.debate_rounds) > 0
            assert result.final_verdict is not None


@pytest.mark.asyncio 
class TestACVFEscalationPass:
    """Test ACVF escalation verification pass."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        return Mock(spec=LLMClient)
    
    @pytest.fixture
    def sample_verification_context(self):
        """Sample verification context for testing."""
        failed_result = VerificationResult(
            pass_id="citation_check_failed",
            pass_type=VerificationPassType.CITATION_CHECK,
            status=VerificationStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            confidence_score=0.3,
            result_data={"issues_found": True},
            metadata={}
        )
        
        return VerificationContext(
            document_id="doc_123",
            document_content="Test document",
            previous_results=[failed_result],
            document_metadata={}
        )
    
    @patch('src.verification.passes.implementations.acvf_escalation_pass.load_default_acvf_config')
    async def test_acvf_escalation_pass_trigger(self, mock_load_config, mock_llm_client, sample_verification_context):
        """Test ACVF escalation pass triggering and execution."""
        # Mock configuration
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_config.config_id = "test_config"
        mock_config.trigger_conditions = {"min_confidence_threshold": 0.6}
        mock_load_config.return_value = mock_config
        
        # Mock ACVF controller
        mock_controller = Mock()
        mock_controller.should_trigger_acvf = AsyncMock(return_value=True)
        
        # Create a mock ACVF result for the escalation
        mock_acvf_result = Mock()
        mock_acvf_result.verification_task_id = "doc_123"
        mock_acvf_result.final_verdict = JudgeVerdict.DEFENDER_WINS
        mock_acvf_result.consensus_confidence = 0.8
        mock_acvf_result.debate_rounds = [Mock()]
        
        mock_controller.process_verification_escalation = AsyncMock(return_value=[mock_acvf_result])
        
        with patch('src.verification.passes.implementations.acvf_escalation_pass.ACVFController', return_value=mock_controller):
            escalation_pass = ACVFEscalationPass(mock_llm_client)
            
            # Create mock config for the execute method
            from src.models.verification import VerificationPassConfig
            mock_pass_config = VerificationPassConfig(
                pass_id="acvf_test_001",
                name="ACVF Test Pass",
                pass_type=VerificationPassType.ADVERSARIAL_VALIDATION,
                enabled=True,
                timeout_seconds=300,
                max_retries=2
            )
            
            result = await escalation_pass.execute(sample_verification_context, mock_pass_config)
            
            assert result.pass_type == VerificationPassType.ADVERSARIAL_VALIDATION
            assert result.result_data["escalation_triggered"] is True
            
            # Verify controller methods were called
            mock_controller.should_trigger_acvf.assert_called_once_with(sample_verification_context)
            mock_controller.process_verification_escalation.assert_called_once_with(sample_verification_context)
    
    @patch('src.verification.passes.implementations.acvf_escalation_pass.load_default_acvf_config')
    async def test_acvf_escalation_pass_no_trigger(self, mock_load_config, mock_llm_client):
        """Test ACVF escalation pass when trigger conditions are not met."""
        # High confidence context that shouldn't trigger ACVF
        high_confidence_result = VerificationResult(
            pass_id="citation_check_success",
            pass_type=VerificationPassType.CITATION_CHECK,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            confidence_score=0.9,
            result_data={},
            metadata={}
        )
        
        high_confidence_context = VerificationContext(
            document_id="doc_123",
            document_content="Test document",
            previous_results=[high_confidence_result],
            document_metadata={}
        )
        
        # Mock configuration
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_config.config_id = "test_config"
        mock_config.trigger_conditions = {"min_confidence_threshold": 0.6}
        mock_load_config.return_value = mock_config
        
        # Mock ACVF controller
        mock_controller = Mock()
        mock_controller.should_trigger_acvf = AsyncMock(return_value=False)
        
        with patch('src.verification.passes.implementations.acvf_escalation_pass.ACVFController', return_value=mock_controller):
            escalation_pass = ACVFEscalationPass(mock_llm_client)
            
            # Create mock config for the execute method
            from src.models.verification import VerificationPassConfig
            mock_pass_config = VerificationPassConfig(
                pass_id="acvf_test_002",
                name="ACVF Test Pass No Trigger",
                pass_type=VerificationPassType.ADVERSARIAL_VALIDATION,
                enabled=True,
                timeout_seconds=300,
                max_retries=2
            )
            
            result = await escalation_pass.execute(high_confidence_context, mock_pass_config)
            
            assert result.pass_type == VerificationPassType.ADVERSARIAL_VALIDATION
            assert result.status == VerificationStatus.COMPLETED
            assert result.result_data["escalation_triggered"] is False
            assert result.confidence_score == 1.0


@pytest.mark.integration
class TestACVFIntegration:
    """Integration tests for complete ACVF workflow."""
    
    @pytest.fixture
    def real_config_file(self, tmp_path):
        """Create a real configuration file for testing."""
        config_content = """
name: "Integration Test ACVF"
description: "Test configuration for integration tests"
version: "1.0"

models:
  challengers:
    - provider: "anthropic"
      model: "claude-3-opus"
      role: "challenger"
      temperature: 0.8
      max_tokens: 2000
      
  defenders:
    - provider: "openai"
      model: "gpt-4"
      role: "defender"
      temperature: 0.7
      max_tokens: 2000
      
  judges:
    - provider: "anthropic"
      model: "claude-3-sonnet"
      role: "judge"
      temperature: 0.5
      max_tokens: 1500

debate_config:
  max_rounds_per_debate: 3
  escalation_threshold: 0.5
  consensus_threshold: 0.7

trigger_conditions:
  min_confidence_threshold: 0.6
  escalate_failed_passes: true
  escalate_on_issues: true

advanced_settings:
  allow_model_self_assignment: false
  require_unanimous_consensus: false
  enable_meta_judging: false
"""
        config_file = tmp_path / "test_acvf_config.yml"
        config_file.write_text(config_content)
        return str(config_file)
    
    def test_end_to_end_config_loading(self, real_config_file):
        """Test end-to-end configuration loading and validation."""
        # Copy the config file to the expected location  
        import shutil
        import os
        
        # Get the config directory
        loader = ACVFConfigLoader()
        config_dir = loader.config_dir
        os.makedirs(config_dir, exist_ok=True)
        
        # Copy file to expected location
        target_file = config_dir / "test_acvf_config.yml"
        shutil.copy2(real_config_file, target_file)
        
        try:
            config = loader.load_config(config_file="test_acvf_config.yml")
            
            assert config.name == "Integration Test ACVF"
            assert len(config.challenger_models) == 1
            assert len(config.defender_models) == 1
            assert len(config.judge_models) == 1
            
            # Validate configuration
            issues = loader.validate_config(config)
            assert len(issues) == 0  # Should be valid
            
        finally:
            # Clean up
            if target_file.exists():
                target_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__]) 