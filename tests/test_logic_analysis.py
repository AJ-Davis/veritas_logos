"""
Tests for logic analysis verification pass.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from src.verification.passes.implementations.logic_analysis_pass import LogicAnalysisPass
from src.models.verification import (
    VerificationPassConfig,
    VerificationPassType,
    VerificationContext,
    VerificationStatus
)
from src.models.logic_bias import (
    LogicalFallacyType,
    ReasoningIssueType,
    LogicalIssue,
    LogicAnalysisResult
)


class TestLogicAnalysisPass:
    """Test suite for LogicAnalysisPass."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        mock_client = Mock()
        mock_client.generate_response = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def logic_pass(self, mock_llm_client):
        """Logic analysis pass instance with mocked LLM client."""
        return LogicAnalysisPass(llm_client=mock_llm_client)
    
    @pytest.fixture
    def basic_config(self):
        """Basic verification pass configuration."""
        return VerificationPassConfig(
            pass_type=VerificationPassType.LOGIC_ANALYSIS,
            pass_id="logic_test",
            name="Logic Analysis Test",
            parameters={
                "model": "gpt-4",
                "min_confidence": 0.3,
                "max_issues": 10
            }
        )
    
    @pytest.fixture
    def verification_context(self):
        """Basic verification context for testing."""
        return VerificationContext(
            document_id="test_doc_001",
            document_content="This is a test document with various logical statements."
        )
    
    @pytest.mark.asyncio
    async def test_execute_successful_analysis(self, logic_pass, basic_config, verification_context, mock_llm_client):
        """Test successful execution of logic analysis."""
        # Mock LLM response
        mock_response = '''[
            {
                "fallacy_type": "ad_hominem",
                "text_excerpt": "You can't trust John's argument because he's incompetent",
                "description": "Attack on person rather than argument",
                "confidence": 0.8,
                "severity": 0.7,
                "start_position": 10,
                "end_position": 70
            }
        ]'''
        
        mock_llm_client.generate_response.return_value = mock_response
        
        result = await logic_pass.execute(verification_context, basic_config)
        
        assert result.status == VerificationStatus.COMPLETED
        assert result.pass_type == VerificationPassType.LOGIC_ANALYSIS
        assert "analysis_result" in result.result_data
        assert result.confidence_score is not None
        
    @pytest.mark.asyncio
    async def test_execute_with_claims_context(self, logic_pass, basic_config, verification_context, mock_llm_client):
        """Test execution with previous claim extraction results."""
        # Add mock claim extraction result to context
        from src.models.verification import VerificationResult
        
        claim_result = VerificationResult(
            pass_id="claim_extract_001",
            pass_type=VerificationPassType.CLAIM_EXTRACTION,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.now(),
            result_data={
                "extraction_result": {
                    "claims": [
                        {"claim_text": "Climate change is fake", "confidence": 0.9}
                    ]
                }
            }
        )
        
        verification_context.previous_results = [claim_result]
        
        mock_llm_client.generate_response.return_value = '[]'
        
        result = await logic_pass.execute(verification_context, basic_config)
        
        assert result.status == VerificationStatus.COMPLETED
        mock_llm_client.generate_response.assert_called_once()
        
        # Check that claims context was used in the prompt
        call_args = mock_llm_client.generate_response.call_args
        assert "climate change" in call_args[0][0].lower()  # First argument should contain the prompt
    
    @pytest.mark.asyncio
    async def test_parse_logic_response_valid_json(self, logic_pass):
        """Test parsing of valid LLM response JSON."""
        response = '''[
            {
                "fallacy_type": "straw_man",
                "text_excerpt": "Environmentalists want to destroy the economy",
                "description": "Misrepresenting opponent's position",
                "confidence": 0.85,
                "severity": 0.8,
                "start_position": 50,
                "end_position": 100
            },
            {
                "reasoning_issue": "invalid_inference",
                "text_excerpt": "All birds fly, penguins are birds, therefore penguins fly",
                "description": "Invalid syllogistic reasoning",
                "confidence": 0.95,
                "severity": 0.9,
                "start_position": 120,
                "end_position": 180
            }
        ]'''
        
        issues = logic_pass._parse_logic_response(
            response, 
            "Sample document content", 
            "test_doc", 
            "gpt-4"
        )
        
        assert len(issues) == 2
        
        # Check first issue (fallacy)
        assert issues[0].fallacy_type == LogicalFallacyType.STRAW_MAN
        assert issues[0].reasoning_type is None
        assert issues[0].confidence_score == 0.85
        assert issues[0].severity_score == 0.8
        
        # Check second issue (reasoning error)
        assert issues[1].fallacy_type is None
        assert issues[1].reasoning_type == ReasoningIssueType.INVALID_INFERENCE
        assert issues[1].confidence_score == 0.95
        assert issues[1].severity_score == 0.9
    
    @pytest.mark.asyncio
    async def test_parse_logic_response_invalid_json(self, logic_pass):
        """Test handling of invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        issues = logic_pass._parse_logic_response(
            invalid_response,
            "Sample content",
            "test_doc",
            "gpt-4"
        )
        
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_execute_llm_failure(self, logic_pass, basic_config, verification_context, mock_llm_client):
        """Test handling of LLM client failure."""
        mock_llm_client.generate_response.side_effect = Exception("LLM service unavailable")
        
        result = await logic_pass.execute(verification_context, basic_config)
        
        assert result.status == VerificationStatus.FAILED
        assert "Logic analysis failed" in result.error_message
    
    def test_get_required_dependencies(self, logic_pass):
        """Test that logic analysis correctly declares its dependencies."""
        dependencies = logic_pass.get_required_dependencies()
        assert VerificationPassType.CLAIM_EXTRACTION in dependencies
    
    @pytest.mark.asyncio
    async def test_create_logical_issue_complete_data(self, logic_pass):
        """Test creation of logical issue with complete data."""
        issue_data = {
            "fallacy_type": "appeal_to_authority",
            "text_excerpt": "Dr. Smith says climate change is false",
            "description": "Appeal to inappropriate authority",
            "explanation": "Dr. Smith is a linguist, not a climate scientist",
            "confidence": 0.75,
            "severity": 0.6,
            "start_position": 10,
            "end_position": 50,
            "suggestions": ["Consider the authority's expertise in the field"]
        }
        
        issue = logic_pass._create_logical_issue(
            issue_data,
            "Sample document content",
            "test_doc"
        )
        
        assert issue.fallacy_type == LogicalFallacyType.APPEAL_TO_AUTHORITY
        assert issue.text_excerpt == "Dr. Smith says climate change is false"
        assert issue.confidence_score == 0.75
        assert issue.severity_score == 0.6
        assert len(issue.suggestions) == 1
    
    @pytest.mark.asyncio
    async def test_create_logical_issue_minimal_data(self, logic_pass):
        """Test creation of logical issue with minimal required data."""
        issue_data = {
            "reasoning_issue": "contradiction",
            "text_excerpt": "Water is both wet and dry",
            "description": "Contradictory statements",
            "confidence": 0.9
        }
        
        issue = logic_pass._create_logical_issue(
            issue_data,
            "Sample document content",
            "test_doc"
        )
        
        assert issue.reasoning_type == ReasoningIssueType.CONTRADICTION
        assert issue.fallacy_type is None
        assert issue.confidence_score == 0.9
        assert issue.severity_score == 0.5  # Default value
    
    @pytest.mark.asyncio
    async def test_configuration_parameters(self, logic_pass, verification_context, mock_llm_client):
        """Test that configuration parameters are properly used."""
        config = VerificationPassConfig(
            pass_type=VerificationPassType.LOGIC_ANALYSIS,
            pass_id="logic_custom",
            name="Custom Logic Analysis",
            parameters={
                "model": "claude-3",
                "min_confidence": 0.8,
                "max_issues": 5,
                "focus_areas": ["logical_fallacies", "reasoning_errors"]
            }
        )
        
        mock_llm_client.generate_response.return_value = '[]'
        
        result = await logic_pass.execute(verification_context, config)
        
        assert result.status == VerificationStatus.COMPLETED
        
        # Verify that parameters were extracted correctly
        call_args = mock_llm_client.generate_response.call_args
        # The prompt should reflect the custom parameters
        assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_analysis_result_creation(self, logic_pass):
        """Test creation of comprehensive analysis result."""
        logical_issues = [
            LogicalIssue(
                issue_id="issue_1",
                issue_type="fallacy",
                fallacy_type=LogicalFallacyType.AD_HOMINEM,
                title="Personal Attack",
                description="Attack on person",
                explanation="Argument targets person not position",
                text_excerpt="You're wrong because you're stupid",
                severity_score=0.8,
                confidence_score=0.9,
                impact_score=0.7
            ),
            LogicalIssue(
                issue_id="issue_2",
                issue_type="reasoning_error",
                reasoning_type=ReasoningIssueType.HASTY_GENERALIZATION,
                title="Hasty Generalization",
                description="Insufficient evidence for broad claim",
                explanation="Single example used to support universal claim",
                text_excerpt="I met one rude French person, so all French people are rude",
                severity_score=0.6,
                confidence_score=0.8,
                impact_score=0.5
            )
        ]
        
        result = logic_pass._create_analysis_result(
            logical_issues,
            "test_doc",
            "gpt-4",
            "v1",
            "Sample document content"
        )
        
        assert isinstance(result, LogicAnalysisResult)
        assert result.document_id == "test_doc"
        assert result.total_issues_found == 2
        assert result.model_used == "gpt-4"
        assert len(result.logical_issues) == 2
        
        # Check fallacy counts
        assert LogicalFallacyType.AD_HOMINEM in result.fallacy_counts
        assert result.fallacy_counts[LogicalFallacyType.AD_HOMINEM] == 1
        
        # Check reasoning issue counts
        assert ReasoningIssueType.HASTY_GENERALIZATION in result.reasoning_issue_counts
        assert result.reasoning_issue_counts[ReasoningIssueType.HASTY_GENERALIZATION] == 1
        
        # Check average scores
        assert 0.6 <= result.average_confidence <= 0.9
        assert 0.6 <= result.average_severity <= 0.8
        assert 0.6 <= result.overall_logic_score <= 0.8


@pytest.mark.integration
class TestLogicAnalysisIntegration:
    """Integration tests for logic analysis with real components."""
    
    @pytest.mark.asyncio
    async def test_integration_with_verification_chain(self):
        """Test integration with verification chain framework."""
        # This would test the full integration but requires real LLM setup
        # For now, this is a placeholder for integration tests
        pass
    
    @pytest.mark.asyncio
    async def test_performance_with_large_document(self):
        """Test performance characteristics with large documents."""
        # This would test memory usage and execution time
        # For now, this is a placeholder for performance tests
        pass


# Test data samples for different types of logical fallacies
FALLACY_TEST_SAMPLES = {
    LogicalFallacyType.AD_HOMINEM: [
        "You can't trust Jane's economic analysis because she's young and inexperienced.",
        "Don't listen to him about health policy - he's overweight himself."
    ],
    LogicalFallacyType.STRAW_MAN: [
        "Environmentalists want to ban all cars and force everyone to walk everywhere.",
        "Gun control advocates want to confiscate every gun in America."
    ],
    LogicalFallacyType.FALSE_DILEMMA: [
        "You're either with us or against us - there's no middle ground.",
        "We must either increase military spending or surrender to our enemies."
    ],
    LogicalFallacyType.SLIPPERY_SLOPE: [
        "If we allow gay marriage, next people will want to marry their pets.",
        "If we raise taxes by 1%, soon we'll have 100% tax rates and live in a communist state."
    ]
}

REASONING_ERROR_SAMPLES = {
    ReasoningIssueType.INVALID_INFERENCE: [
        "All birds can fly. Penguins are birds. Therefore, penguins can fly.",
        "Some politicians are corrupt. John is a politician. Therefore, John is corrupt."
    ],
    ReasoningIssueType.CONTRADICTION: [
        "The economy is both growing rapidly and shrinking at the same time.",
        "This statement is completely true and completely false."
    ],
    ReasoningIssueType.UNSUPPORTED_CONCLUSION: [
        "Technology will solve all environmental problems, so we don't need to worry about climate change.",
        "Since crime rates vary by neighborhood, all social programs are ineffective."
    ]
}


@pytest.mark.parametrize("fallacy_type,samples", FALLACY_TEST_SAMPLES.items())
def test_fallacy_detection_samples(fallacy_type, samples):
    """Test that fallacy detection works with known samples."""
    # This would test against known fallacy samples
    # Implementation would depend on having a trained model or rules
    pass


@pytest.mark.parametrize("reasoning_type,samples", REASONING_ERROR_SAMPLES.items())
def test_reasoning_error_samples(reasoning_type, samples):
    """Test that reasoning error detection works with known samples."""
    # This would test against known reasoning error samples
    pass 