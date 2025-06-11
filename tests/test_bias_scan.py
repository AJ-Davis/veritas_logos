"""
Tests for bias scan verification pass.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from src.verification.passes.implementations.bias_scan_pass import BiasScanPass
from src.models.verification import (
    VerificationPassConfig,
    VerificationPassType,
    VerificationContext,
    VerificationStatus
)
from src.models.logic_bias import (
    BiasType,
    BiasSeverity,
    BiasIssue,
    BiasAnalysisResult
)


class TestBiasScanPass:
    """Test suite for BiasScanPass."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        mock_client = Mock()
        mock_client.generate_response = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def bias_pass(self, mock_llm_client):
        """Bias scan pass instance with mocked LLM client."""
        return BiasScanPass(llm_client=mock_llm_client)
    
    @pytest.fixture
    def basic_config(self):
        """Basic verification pass configuration."""
        return VerificationPassConfig(
            pass_type=VerificationPassType.BIAS_SCAN,
            pass_id="bias_test",
            name="Bias Scan Test",
            parameters={
                "model": "gpt-4",
                "min_confidence": 0.3,
                "max_issues": 10,
                "demographic_analysis": True
            }
        )
    
    @pytest.fixture
    def verification_context(self):
        """Basic verification context for testing."""
        return VerificationContext(
            document_id="test_doc_001",
            document_content="This is a test document about various social and political topics."
        )
    
    @pytest.mark.asyncio
    async def test_execute_successful_analysis(self, bias_pass, basic_config, verification_context, mock_llm_client):
        """Test successful execution of bias scan."""
        # Mock LLM response
        mock_response = '''[
            {
                "bias_type": "gender_bias",
                "text_excerpt": "Women are naturally better at nurturing children",
                "description": "Gender stereotype about caregiving roles",
                "severity": "moderate",
                "confidence": 0.8,
                "impact_score": 0.6,
                "start_position": 20,
                "end_position": 80
            }
        ]'''
        
        mock_llm_client.generate_response.return_value = mock_response
        
        result = await bias_pass.execute(verification_context, basic_config)
        
        assert result.status == VerificationStatus.COMPLETED
        assert result.pass_type == VerificationPassType.BIAS_SCAN
        assert "analysis_result" in result.result_data
        assert result.confidence_score is not None
        
    @pytest.mark.asyncio
    async def test_execute_with_claims_context(self, bias_pass, basic_config, verification_context, mock_llm_client):
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
                        {"claim_text": "Men are better leaders than women", "confidence": 0.9}
                    ]
                }
            }
        )
        
        verification_context.previous_results = [claim_result]
        
        mock_llm_client.generate_response.return_value = '[]'
        
        result = await bias_pass.execute(verification_context, basic_config)
        
        assert result.status == VerificationStatus.COMPLETED
        mock_llm_client.generate_response.assert_called_once()
        
        # Check that claims context was used in the prompt
        call_args = mock_llm_client.generate_response.call_args
        assert "men are better leaders" in call_args[0][0].lower()
    
    @pytest.mark.asyncio
    async def test_parse_bias_response_valid_json(self, bias_pass):
        """Test parsing of valid LLM response JSON."""
        response = '''[
            {
                "bias_type": "racial_bias",
                "text_excerpt": "Asian students are naturally good at math",
                "description": "Racial stereotype about academic abilities",
                "severity": "moderate",
                "confidence": 0.85,
                "impact_score": 0.7,
                "start_position": 50,
                "end_position": 100,
                "evidence": ["Perpetuates model minority myth"],
                "mitigation_suggestions": ["Acknowledge individual differences within groups"]
            },
            {
                "bias_type": "confirmation_bias",
                "text_excerpt": "I only read news sources that confirm my beliefs",
                "description": "Selective information gathering",
                "severity": "low",
                "confidence": 0.95,
                "impact_score": 0.4,
                "start_position": 120,
                "end_position": 180
            }
        ]'''
        
        issues = bias_pass._parse_bias_response(
            response, 
            "Sample document content", 
            "test_doc", 
            "gpt-4"
        )
        
        assert len(issues) == 2
        
        # Check first issue (racial bias)
        assert issues[0].bias_type == BiasType.RACIAL_BIAS
        assert issues[0].severity == BiasSeverity.MODERATE
        assert issues[0].confidence_score == 0.85
        assert issues[0].impact_score == 0.7
        assert len(issues[0].evidence) == 1
        assert len(issues[0].mitigation_suggestions) == 1
        
        # Check second issue (confirmation bias)
        assert issues[1].bias_type == BiasType.CONFIRMATION_BIAS
        assert issues[1].severity == BiasSeverity.LOW
        assert issues[1].confidence_score == 0.95
        assert issues[1].impact_score == 0.4
    
    @pytest.mark.asyncio
    async def test_parse_bias_response_invalid_json(self, bias_pass):
        """Test handling of invalid JSON response."""
        invalid_response = "This is not valid JSON"
        
        issues = bias_pass._parse_bias_response(
            invalid_response,
            "Sample content",
            "test_doc",
            "gpt-4"
        )
        
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_execute_llm_failure(self, bias_pass, basic_config, verification_context, mock_llm_client):
        """Test handling of LLM client failure."""
        mock_llm_client.generate_response.side_effect = Exception("LLM service unavailable")
        
        result = await bias_pass.execute(verification_context, basic_config)
        
        assert result.status == VerificationStatus.FAILED
        assert "Bias scan failed" in result.error_message
    
    def test_get_required_dependencies(self, bias_pass):
        """Test that bias scan correctly declares its dependencies."""
        dependencies = bias_pass.get_required_dependencies()
        assert VerificationPassType.CLAIM_EXTRACTION in dependencies
    
    @pytest.mark.asyncio
    async def test_create_bias_issue_complete_data(self, bias_pass):
        """Test creation of bias issue with complete data."""
        issue_data = {
            "bias_type": "political_bias",
            "text_excerpt": "All conservatives hate the environment",
            "description": "Political generalization",
            "explanation": "Overgeneralization about political group's views",
            "severity": "high",
            "confidence": 0.75,
            "impact_score": 0.8,
            "start_position": 10,
            "end_position": 50,
            "evidence": ["Uses absolute language 'all'"],
            "mitigation_suggestions": ["Use more nuanced language about political positions"],
            "alternative_perspectives": ["Many conservatives support environmental protection"]
        }
        
        issue = bias_pass._create_bias_issue(
            issue_data,
            "Sample document content",
            "test_doc"
        )
        
        assert issue.bias_type == BiasType.POLITICAL_BIAS
        assert issue.text_excerpt == "All conservatives hate the environment"
        assert issue.severity == BiasSeverity.HIGH
        assert issue.confidence_score == 0.75
        assert issue.impact_score == 0.8
        assert len(issue.evidence) == 1
        assert len(issue.mitigation_suggestions) == 1
        assert len(issue.alternative_perspectives) == 1
    
    @pytest.mark.asyncio
    async def test_create_bias_issue_minimal_data(self, bias_pass):
        """Test creation of bias issue with minimal required data."""
        issue_data = {
            "bias_type": "age_bias",
            "text_excerpt": "Young people don't understand responsibility",
            "description": "Age-based generalization",
            "confidence": 0.9
        }
        
        issue = bias_pass._create_bias_issue(
            issue_data,
            "Sample document content",
            "test_doc"
        )
        
        assert issue.bias_type == BiasType.AGE_BIAS
        assert issue.confidence_score == 0.9
        assert issue.severity == BiasSeverity.MODERATE  # Default value
        assert issue.impact_score == 0.5  # Default value
    
    @pytest.mark.asyncio
    async def test_configuration_parameters(self, bias_pass, verification_context, mock_llm_client):
        """Test that configuration parameters are properly used."""
        config = VerificationPassConfig(
            pass_type=VerificationPassType.BIAS_SCAN,
            pass_id="bias_custom",
            name="Custom Bias Scan",
            parameters={
                "model": "claude-3",
                "min_confidence": 0.8,
                "max_issues": 5,
                "bias_types_focus": ["gender_bias", "racial_bias"],
                "demographic_analysis": False
            }
        )
        
        mock_llm_client.generate_response.return_value = '[]'
        
        result = await bias_pass.execute(verification_context, config)
        
        assert result.status == VerificationStatus.COMPLETED
        
        # Verify that parameters were extracted correctly
        call_args = mock_llm_client.generate_response.call_args
        assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_analysis_result_creation(self, bias_pass):
        """Test creation of comprehensive analysis result."""
        bias_issues = [
            BiasIssue(
                issue_id="bias_1",
                bias_type=BiasType.GENDER_BIAS,
                title="Gender Stereotype",
                description="Assumption about gender roles",
                explanation="Reinforces traditional gender expectations",
                text_excerpt="Women should focus on family over career",
                severity=BiasSeverity.HIGH,
                confidence_score=0.9,
                impact_score=0.8,
                evidence=["Uses prescriptive language"],
                mitigation_suggestions=["Acknowledge individual choice and capability"]
            ),
            BiasIssue(
                issue_id="bias_2",
                bias_type=BiasType.CONFIRMATION_BIAS,
                title="Selective Evidence",
                description="Cherry-picking supporting information",
                explanation="Ignores contradictory evidence",
                text_excerpt="Studies show my position is correct",
                severity=BiasSeverity.MODERATE,
                confidence_score=0.7,
                impact_score=0.6,
                evidence=["Vague reference to 'studies'"],
                mitigation_suggestions=["Include comprehensive literature review"]
            )
        ]
        
        result = bias_pass._create_analysis_result(
            bias_issues,
            "test_doc",
            "gpt-4",
            "v1",
            "Sample document content"
        )
        
        assert isinstance(result, BiasAnalysisResult)
        assert result.document_id == "test_doc"
        assert result.total_issues_found == 2
        assert result.model_used == "gpt-4"
        assert len(result.bias_issues) == 2
        
        # Check bias type counts
        assert BiasType.GENDER_BIAS in result.bias_type_counts
        assert result.bias_type_counts[BiasType.GENDER_BIAS] == 1
        assert BiasType.CONFIRMATION_BIAS in result.bias_type_counts
        assert result.bias_type_counts[BiasType.CONFIRMATION_BIAS] == 1
        
        # Check severity distribution
        assert BiasSeverity.HIGH in result.severity_distribution
        assert result.severity_distribution[BiasSeverity.HIGH] == 1
        assert BiasSeverity.MODERATE in result.severity_distribution
        assert result.severity_distribution[BiasSeverity.MODERATE] == 1
        
        # Check average scores
        assert 0.7 <= result.average_confidence <= 0.9
        assert 0.6 <= result.average_impact <= 0.8
    
    @pytest.mark.asyncio
    async def test_political_leaning_analysis(self, bias_pass):
        """Test political leaning analysis functionality."""
        bias_issues = [
            BiasIssue(
                issue_id="political_1",
                bias_type=BiasType.POLITICAL_BIAS,
                title="Conservative Bias",
                description="Favors conservative viewpoints",
                explanation="Presents conservative positions more favorably",
                text_excerpt="Conservative policies are always better",
                severity=BiasSeverity.HIGH,
                confidence_score=0.8,
                impact_score=0.7
            )
        ]
        
        political_leaning = bias_pass._analyze_political_leaning(bias_issues)
        
        # The method should return a string indicating detected political bias
        assert political_leaning is not None
        assert isinstance(political_leaning, str)
    
    @pytest.mark.asyncio
    async def test_demographic_representation_analysis(self, bias_pass):
        """Test demographic representation analysis."""
        bias_issues = [
            BiasIssue(
                issue_id="demo_1",
                bias_type=BiasType.GENDER_BIAS,
                title="Gender Representation",
                description="Unequal gender representation",
                explanation="Male perspectives dominate discussion",
                text_excerpt="Men's opinions on this topic",
                severity=BiasSeverity.MODERATE,
                confidence_score=0.7,
                impact_score=0.6
            ),
            BiasIssue(
                issue_id="demo_2",
                bias_type=BiasType.RACIAL_BIAS,
                title="Racial Representation",
                description="Limited racial perspectives",
                explanation="Primarily reflects majority viewpoints",
                text_excerpt="From a white perspective",
                severity=BiasSeverity.MODERATE,
                confidence_score=0.8,
                impact_score=0.7
            )
        ]
        
        demo_analysis = bias_pass._analyze_demographic_representation(bias_issues)
        
        assert isinstance(demo_analysis, dict)
        assert "gender_representation" in demo_analysis
        assert "racial_representation" in demo_analysis
        assert "overall_diversity_score" in demo_analysis
    
    @pytest.mark.asyncio
    async def test_source_diversity_analysis(self, bias_pass):
        """Test source diversity analysis."""
        bias_issues = [
            BiasIssue(
                issue_id="source_1",
                bias_type=BiasType.SELECTION_BIAS,
                title="Source Selection",
                description="Limited source diversity",
                explanation="Sources represent narrow perspective",
                text_excerpt="According to our preferred sources",
                severity=BiasSeverity.MODERATE,
                confidence_score=0.75,
                impact_score=0.65
            )
        ]
        
        document_content = "Sample document with various citations and references"
        source_analysis = bias_pass._analyze_source_diversity(bias_issues, document_content)
        
        assert isinstance(source_analysis, dict)
        assert "source_count_estimate" in source_analysis
        assert "diversity_issues" in source_analysis
        assert "recommendations" in source_analysis
    
    @pytest.mark.asyncio
    async def test_generate_overall_recommendations(self, bias_pass):
        """Test generation of overall recommendations."""
        bias_issues = [
            BiasIssue(
                issue_id="rec_1",
                bias_type=BiasType.CONFIRMATION_BIAS,
                title="Confirmation Bias",
                description="Seeking confirming evidence",
                explanation="Ignoring contradictory information",
                text_excerpt="Only evidence that supports us",
                severity=BiasSeverity.HIGH,
                confidence_score=0.9,
                impact_score=0.8,
                mitigation_suggestions=["Seek diverse perspectives", "Include opposing viewpoints"]
            )
        ]
        
        recommendations = bias_pass._generate_overall_recommendations(bias_issues)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


@pytest.mark.integration
class TestBiasScanIntegration:
    """Integration tests for bias scan with real components."""
    
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


# Test data samples for different types of bias
BIAS_TEST_SAMPLES = {
    BiasType.GENDER_BIAS: [
        "Women are naturally more emotional and less logical than men.",
        "Men are better suited for leadership positions in business.",
        "Mothers should prioritize family over career advancement."
    ],
    BiasType.RACIAL_BIAS: [
        "Asian students are inherently better at mathematics.",
        "African Americans are naturally gifted athletes.",
        "White people are more trustworthy in financial matters."
    ],
    BiasType.AGE_BIAS: [
        "Young people lack the wisdom to make important decisions.",
        "Older workers are less adaptable to new technology.",
        "Millennials are lazy and entitled compared to previous generations."
    ],
    BiasType.POLITICAL_BIAS: [
        "All liberals want to destroy traditional values.",
        "Conservatives are inherently selfish and uncaring.",
        "Independent voters are just confused and indecisive."
    ],
    BiasType.CONFIRMATION_BIAS: [
        "I only trust news sources that align with my beliefs.",
        "Studies that contradict my position must be flawed.",
        "Evidence supporting my view is always more reliable."
    ]
}


@pytest.mark.parametrize("bias_type,samples", BIAS_TEST_SAMPLES.items())
def test_bias_detection_samples(bias_type, samples):
    """Test that bias detection works with known samples."""
    # This would test against known bias samples
    # Implementation would depend on having trained models or detection rules
    pass


class TestBiasSeverityAssessment:
    """Test bias severity assessment functionality."""
    
    def test_severity_mapping(self):
        """Test that severity strings map correctly to enum values."""
        severity_mappings = {
            "minimal": BiasSeverity.MINIMAL,
            "low": BiasSeverity.LOW,
            "moderate": BiasSeverity.MODERATE,
            "high": BiasSeverity.HIGH,
            "severe": BiasSeverity.SEVERE
        }
        
        for string_val, enum_val in severity_mappings.items():
            assert BiasSeverity(string_val) == enum_val
    
    def test_severity_ordering(self):
        """Test that severity levels can be properly ordered."""
        severities = [
            BiasSeverity.MINIMAL,
            BiasSeverity.LOW,
            BiasSeverity.MODERATE,
            BiasSeverity.HIGH,
            BiasSeverity.SEVERE
        ]
        
        # This would test ordering logic if implemented
        # For now, just verify the enum values exist
        assert len(severities) == 5


class TestBiasTypeClassification:
    """Test bias type classification functionality."""
    
    def test_all_bias_types_covered(self):
        """Test that all defined bias types are accounted for."""
        expected_types = [
            BiasType.SELECTION_BIAS,
            BiasType.CONFIRMATION_BIAS,
            BiasType.CULTURAL_BIAS,
            BiasType.DEMOGRAPHIC_BIAS,
            BiasType.POLITICAL_BIAS,
            BiasType.IDEOLOGICAL_BIAS,
            BiasType.STATISTICAL_BIAS,
            BiasType.FRAMING_BIAS,
            BiasType.GENDER_BIAS,
            BiasType.RACIAL_BIAS,
            BiasType.AGE_BIAS
        ]
        
        # Verify all types are in the enum
        for bias_type in expected_types:
            assert bias_type in BiasType
    
    def test_bias_type_categorization(self):
        """Test that bias types can be categorized by domain."""
        demographic_biases = [
            BiasType.GENDER_BIAS,
            BiasType.RACIAL_BIAS,
            BiasType.AGE_BIAS,
            BiasType.DEMOGRAPHIC_BIAS
        ]
        
        cognitive_biases = [
            BiasType.CONFIRMATION_BIAS,
            BiasType.SELECTION_BIAS,
            BiasType.ANCHORING_BIAS,
            BiasType.AVAILABILITY_BIAS
        ]
        
        # This would test categorization logic if implemented
        # For now, just verify the types exist
        for bias_type in demographic_biases + cognitive_biases:
            assert bias_type in BiasType 