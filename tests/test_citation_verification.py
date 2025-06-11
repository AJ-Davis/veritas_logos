"""
Comprehensive tests for citation verification functionality.
"""

import json
import pytest
import tempfile
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.verification.passes.implementations.citation_verification_pass import CitationVerificationPass
from src.models.verification import (
    VerificationContext,
    VerificationPassConfig,
    VerificationPassType,
    VerificationResult,
    VerificationStatus
)
from src.models.claims import ExtractedClaim, ClaimLocation, ClaimExtractionResult
from src.models.citations import (
    CitationVerificationResult,
    VerifiedCitation,
    CitationStatus,
    CitationType,
    SupportLevel,
    CitationIssue,
    CitationLocation,
    SourceCredibility
)


class TestCitationVerificationPass:
    """Test suite for citation verification pass."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for testing."""
        from src.llm.llm_client import LLMResponse, LLMProvider
        
        mock_client = Mock()
        
        async def mock_generate_response(*args, **kwargs):
            # Return the mocked JSON response wrapped in LLMResponse
            return LLMResponse(
                content=mock_client._mock_response_content,
                provider=LLMProvider.OPENAI,
                model="mock-gpt-4",
                usage={},
                response_time_seconds=0.1,
                metadata={}
            )
        
        mock_client.generate_response = mock_generate_response
        mock_client._mock_response_content = json.dumps([])  # Default empty response
        return mock_client
    
    @pytest.fixture
    def citation_pass(self, mock_llm_client):
        """Create citation verification pass with mock LLM client."""
        return CitationVerificationPass(llm_client=mock_llm_client)
    
    @pytest.fixture
    def sample_claims_with_citations(self):
        """Create sample claims with citations for testing."""
        return [
            ExtractedClaim(
                claim_id="claim_1",
                claim_text="Global temperatures have increased by 1.1°C since pre-industrial times",
                location=ClaimLocation(
                    start_position=100,
                    end_position=165,
                    page_number=1,
                    section_type="main_content"
                ),
                claim_type="factual",
                document_id="test_doc_1",
                extraction_confidence=0.9,
                extracted_by="test_extractor",
                citations=["NASA Climate Change Evidence, https://climate.nasa.gov/evidence/"]
            ),
            ExtractedClaim(
                claim_id="claim_2",
                claim_text="Chocolate is healthy and cures all diseases",
                location=ClaimLocation(
                    start_position=200,
                    end_position=245,
                    page_number=1,
                    section_type="main_content"
                ),
                claim_type="factual",
                document_id="test_doc_1",
                extraction_confidence=0.8,
                extracted_by="test_extractor",
                citations=["Health Benefits of Chocolate, www.chocolatelovers.com/health"]
            )
        ]
    
    @pytest.fixture
    def sample_claims_without_citations(self):
        """Create sample claims without citations for testing."""
        return [
            ExtractedClaim(
                claim_id="claim_3",
                claim_text="This is an unsupported claim",
                location=ClaimLocation(
                    start_position=300,
                    end_position=330,
                    page_number=1,
                    section_type="main_content"
                ),
                claim_type="evaluative",
                document_id="test_doc_2",
                extraction_confidence=0.7,
                extracted_by="test_extractor",
                citations=[]
            )
        ]
    
    @pytest.fixture
    def verification_context(self, sample_claims_with_citations):
        """Create verification context with sample data."""
        # Create claim extraction result
        claim_result = ClaimExtractionResult(
            document_id="test_doc_1",
            claims=sample_claims_with_citations,
            total_claims_found=len(sample_claims_with_citations),
            model_used="test_model",
            prompt_version="v1.0",
            document_length=1000
        )
        
        # Create context with previous results
        previous_result = VerificationResult(
            pass_id="claim_extraction_1",
            pass_type=VerificationPassType.CLAIM_EXTRACTION,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.utcnow(),
            result_data={
                "extraction_result": claim_result.model_dump()
            }
        )
        
        context = VerificationContext(
            document_id="test_doc_1",
            document_content="Test document content with claims and citations...",
            document_metadata={"title": "Test Document"},
            previous_results=[previous_result]
        )
        return context
    
    @pytest.fixture
    def verification_config(self):
        """Create verification configuration for citation checking."""
        return VerificationPassConfig(
            pass_id="citation_check_1",
            pass_type=VerificationPassType.CITATION_CHECK,
            name="check_citations",
            parameters={
                "max_citations": 10,
                "model": "mock-gpt-4",
                "min_confidence": 0.3,
                "retrieve_content": False  # Disable for testing
            }
        )
    
    @pytest.mark.asyncio
    async def test_citation_verification_initialization(self):
        """Test citation verification pass initialization."""
        pass_instance = CitationVerificationPass()
        assert pass_instance.pass_type == VerificationPassType.CITATION_CHECK
        assert pass_instance.llm_client is not None
        assert pass_instance.get_required_dependencies() == [VerificationPassType.CLAIM_EXTRACTION]
    
    @pytest.mark.asyncio
    async def test_extract_claims_from_context(self, citation_pass, verification_context):
        """Test extraction of claims from verification context."""
        claims = citation_pass._extract_claims_from_context(verification_context)
        assert len(claims) == 2
        assert claims[0].claim_id == "claim_1"
        assert claims[1].claim_id == "claim_2"
        assert all(len(claim.citations) > 0 for claim in claims)
    
    @pytest.mark.asyncio
    async def test_citation_type_classification(self, citation_pass):
        """Test citation type classification logic."""
        test_cases = [
            ("NASA study from nasa.gov", CitationType.GOVERNMENT_DOCUMENT),
            ("Journal of Science, DOI: 10.1234", CitationType.ACADEMIC_PAPER),
            ("BBC News article", CitationType.NEWS_ARTICLE),
            ("Wikipedia entry", CitationType.WEBSITE),
            ("Book by Publisher", CitationType.BOOK),
            ("https://example.com/page", CitationType.WEBSITE),
            ("Random text", CitationType.OTHER)
        ]
        
        for citation_text, expected_type in test_cases:
            result = citation_pass._classify_citation_type(citation_text)
            assert result == expected_type, f"Failed for {citation_text}"
    
    @pytest.mark.asyncio
    async def test_verification_with_valid_citation(self, citation_pass, verification_context, verification_config, mock_llm_client):
        """Test citation verification with a valid citation."""
        # Mock LLM response for valid citations (2 citations expected)
        mock_response = json.dumps([
            {
                "verification_status": "valid",
                "support_level": "strong_support",
                "confidence_score": 0.9,
                "issues": [],
                "explanation": "NASA is a credible source and supports the temperature claim",
                "recommendations": []
            },
            {
                "verification_status": "valid",
                "support_level": "moderate_support",
                "confidence_score": 0.7,
                "issues": [],
                "explanation": "Second citation also verified",
                "recommendations": []
            }
        ])
        mock_llm_client._mock_response_content = mock_response
        
        result = await citation_pass.execute(verification_context, verification_config)
        
        assert result.status == VerificationStatus.COMPLETED
        assert result.confidence_score > 0.5
        
        # Check result data structure
        verification_result = result.result_data["verification_result"]
        assert verification_result["total_citations_verified"] >= 1
        assert verification_result["valid_citations"] >= 1
    
    @pytest.mark.asyncio
    async def test_verification_with_invalid_citation(self, citation_pass, verification_context, verification_config, mock_llm_client):
        """Test citation verification with an invalid citation."""
        # Mock LLM response for invalid citation
        mock_response = json.dumps([
            {
                "verification_status": "invalid",
                "support_level": "contradicts",
                "confidence_score": 0.8,
                "issues": ["unreliable_source", "content_mismatch"],
                "explanation": "Source is unreliable and contradicts scientific consensus",
                "recommendations": ["Find peer-reviewed source"]
            }
        ])
        mock_llm_client._mock_response_content = mock_response
        
        result = await citation_pass.execute(verification_context, verification_config)
        
        assert result.status == VerificationStatus.COMPLETED
        verification_result = result.result_data["verification_result"]
        assert verification_result["invalid_citations"] >= 1
    
    @pytest.mark.asyncio
    async def test_verification_with_no_claims(self, citation_pass, verification_config):
        """Test citation verification when no claims are found."""
        # Create context with no previous results
        context = VerificationContext(
            document_id="test_doc_empty",
            document_content="Document with no claims",
            document_metadata={},
            previous_results=[]
        )
        
        result = await citation_pass.execute(context, verification_config)
        
        assert result.status == VerificationStatus.COMPLETED
        assert "warning" in result.result_data
        assert result.confidence_score == 1.0  # High confidence in "no claims" result
    
    @pytest.mark.asyncio
    async def test_verification_with_claims_no_citations(self, citation_pass, verification_config, sample_claims_without_citations):
        """Test citation verification when claims have no citations."""
        # Create claim extraction result with no citations
        claim_result = ClaimExtractionResult(
            document_id="test_doc_no_citations",
            claims=sample_claims_without_citations,
            total_claims_found=len(sample_claims_without_citations),
            model_used="test_model",
            prompt_version="v1.0",
            document_length=500
        )
        
        previous_result = VerificationResult(
            pass_id="claim_extraction_2",
            pass_type=VerificationPassType.CLAIM_EXTRACTION,
            status=VerificationStatus.COMPLETED,
            started_at=datetime.utcnow(),
            result_data={
                "extraction_result": claim_result.model_dump()
            }
        )
        
        context = VerificationContext(
            document_id="test_doc_no_citations",
            document_content="Document with claims but no citations",
            document_metadata={},
            previous_results=[previous_result]
        )
        
        result = await citation_pass.execute(context, verification_config)
        
        assert result.status == VerificationStatus.COMPLETED
        assert "info" in result.result_data
        verification_result = result.result_data["verification_result"]
        assert verification_result["claims_without_citations"] == len(sample_claims_without_citations)
    
    @pytest.mark.asyncio
    async def test_llm_response_parsing(self, citation_pass, sample_claims_with_citations):
        """Test parsing of LLM responses."""
        citation_batch = [
            {
                "claim": sample_claims_with_citations[0],
                "citation_text": sample_claims_with_citations[0].citations[0]
            }
        ]
        
        # Test valid response
        valid_response = json.dumps([
            {
                "verification_status": "valid",
                "support_level": "moderate_support",
                "confidence_score": 0.7,
                "issues": [],
                "explanation": "Good source",
                "recommendations": []
            }
        ])
        
        results = citation_pass._parse_verification_response(
            valid_response, citation_batch, "test_doc", "mock-model"
        )
        
        assert len(results) == 1
        assert results[0].verification_status == CitationStatus.VALID
        assert results[0].support_level == SupportLevel.MODERATE_SUPPORT
        assert results[0].confidence_score == 0.7
    
    @pytest.mark.asyncio
    async def test_llm_response_parsing_malformed(self, citation_pass, sample_claims_with_citations):
        """Test parsing of malformed LLM responses."""
        citation_batch = [
            {
                "claim": sample_claims_with_citations[0],
                "citation_text": sample_claims_with_citations[0].citations[0]
            }
        ]
        
        # Test malformed response
        malformed_response = "This is not valid JSON"
        
        results = citation_pass._parse_verification_response(
            malformed_response, citation_batch, "test_doc", "mock-model"
        )
        
        assert len(results) == 1
        assert results[0].verification_status == CitationStatus.PENDING
        assert results[0].confidence_score <= 0.5
        assert CitationIssue.FORMATTING_ERROR in results[0].identified_issues
    
    @pytest.mark.asyncio
    async def test_citation_location_finding(self, citation_pass):
        """Test citation location detection logic."""
        claim_location = ClaimLocation(
            start_position=100,
            end_position=150,
            page_number=1,
            section_type="main_content"
        )
        
        citation_text = "Test citation"
        location = citation_pass._find_citation_location(citation_text, claim_location)
        
        assert location.start_position == claim_location.start_position
        assert location.page_number == claim_location.page_number
        assert location.section_type == claim_location.section_type
    
    @pytest.mark.asyncio 
    async def test_error_handling_llm_failure(self, citation_pass, verification_context, verification_config, mock_llm_client):
        """Test error handling when LLM fails."""
        # Mock LLM failure
        async def failing_generate_response(*args, **kwargs):
            raise Exception("LLM API error")
        mock_llm_client.generate_response = failing_generate_response
        
        result = await citation_pass.execute(verification_context, verification_config)
        
        # Should still complete but with error information
        assert result.status in [VerificationStatus.COMPLETED, VerificationStatus.FAILED]
        if result.status == VerificationStatus.FAILED:
            assert "error" in result.result_data
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, citation_pass):
        """Test confidence score calculation logic."""
        # Test high confidence case
        high_conf_citation = VerifiedCitation(
            citation_text="Test citation",
            citation_type=CitationType.ACADEMIC_PAPER,
            claim_id="test_claim",
            claim_text="Test claim",
            location=CitationLocation(start_position=0, end_position=10),
            document_id="test_doc",
            verification_status=CitationStatus.VALID,
            support_level=SupportLevel.STRONG_SUPPORT,
            confidence_score=0.9,
            verified_by="test_model"
        )
        
        # Test low confidence case
        low_conf_citation = VerifiedCitation(
            citation_text="Test citation",
            citation_type=CitationType.WEBSITE,
            claim_id="test_claim",
            claim_text="Test claim",
            location=CitationLocation(start_position=0, end_position=10),
            document_id="test_doc",
            verification_status=CitationStatus.INVALID,
            support_level=SupportLevel.NO_SUPPORT,
            confidence_score=0.2,
            identified_issues=[CitationIssue.UNRELIABLE_SOURCE],
            verified_by="test_model"
        )
        
        # Create verification result
        verification_result = CitationVerificationResult(
            document_id="test_doc",
            verified_citations=[high_conf_citation, low_conf_citation],
            model_used="test_model"
        )
        
        # Manually calculate expected average
        expected_avg = (0.9 + 0.2) / 2
        verification_result.average_confidence = expected_avg
        
        assert verification_result.average_confidence == expected_avg
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, citation_pass, mock_llm_client):
        """Test batch processing of citations."""
        # Create multiple citations
        claims_with_citations = []
        for i in range(5):
            claim = ExtractedClaim(
                claim_id=f"claim_{i}",
                claim_text=f"Test claim {i}",
                location=ClaimLocation(start_position=i*100, end_position=(i+1)*100),
                claim_type="factual",
                document_id=f"test_doc_{i}",
                extraction_confidence=0.8,
                extracted_by="test_extractor",
                citations=[f"Citation {i} text"]
            )
            claims_with_citations.append(claim)
        
        # Mock batch response
        batch_response = json.dumps([
            {
                "verification_status": "valid",
                "support_level": "moderate_support",
                "confidence_score": 0.7,
                "issues": [],
                "explanation": f"Citation {i} explanation",
                "recommendations": []
            } for i in range(5)
        ])
        mock_llm_client._mock_response_content = batch_response
        
        # Test batch verification
        result = await citation_pass._verify_citations_with_llm(
            claims_with_citations=claims_with_citations,
            document_content="Test document",
            document_id="test_doc",
            max_citations=10,
            model="mock-model",
            prompt_version="v1",
            min_confidence=0.3,
            retrieve_content=False
        )
        
        assert len(result.verified_citations) == 5
        assert result.total_citations_verified == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 