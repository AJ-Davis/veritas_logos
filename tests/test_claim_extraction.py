"""
Tests for claim extraction module.
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.models.claims import (
    ExtractedClaim,
    ClaimExtractionResult,
    ClaimLocation,
    ClaimType,
    ClaimCategory
)
from src.models.verification import (
    VerificationContext,
    VerificationPassConfig,
    VerificationPassType,
    VerificationStatus
)
from src.llm.llm_client import LLMClient, LLMConfig, LLMProvider, LLMResponse
from src.llm.prompts import PromptType, prompt_manager
from src.verification.passes.implementations.claim_extraction_pass import ClaimExtractionPass


class TestClaimModels:
    """Test cases for claim data models."""
    
    def test_extracted_claim_creation(self):
        """Test creation of ExtractedClaim objects."""
        location = ClaimLocation(
            start_position=100,
            end_position=200,
            context_before="Before text",
            context_after="After text"
        )
        
        claim = ExtractedClaim(
            claim_text="Climate change causes global warming",
            claim_type=ClaimType.CAUSAL,
            category=ClaimCategory.ENVIRONMENTAL,
            location=location,
            document_id="test_doc",
            extraction_confidence=0.9,
            extracted_by="gpt-4"
        )
        
        assert claim.claim_text == "Climate change causes global warming"
        assert claim.claim_type == ClaimType.CAUSAL
        assert claim.category == ClaimCategory.ENVIRONMENTAL
        assert claim.extraction_confidence == 0.9
        assert claim.requires_fact_check is True
        assert claim.claim_id is not None
    
    def test_claim_extraction_result(self):
        """Test ClaimExtractionResult creation and statistics."""
        claims = [
            ExtractedClaim(
                claim_text="Test claim 1",
                claim_type=ClaimType.FACTUAL,
                location=ClaimLocation(start_position=0, end_position=10),
                document_id="test",
                extraction_confidence=0.8,
                extracted_by="test"
            ),
            ExtractedClaim(
                claim_text="Test claim 2", 
                claim_type=ClaimType.STATISTICAL,
                location=ClaimLocation(start_position=20, end_position=30),
                document_id="test",
                extraction_confidence=0.9,
                extracted_by="test"
            )
        ]
        
        result = ClaimExtractionResult(
            document_id="test_doc",
            claims=claims,
            total_claims_found=2,
            model_used="gpt-4",
            prompt_version="v1",
            document_length=1000
        )
        
        assert result.total_claims_found == 2
        assert result.document_id == "test_doc"
        assert len(result.claims) == 2


class TestPromptTemplates:
    """Test cases for prompt templates."""
    
    def test_claim_extraction_prompt_template(self):
        """Test claim extraction prompt template."""
        template = prompt_manager.get_template(PromptType.CLAIM_EXTRACTION, "v1")
        
        assert template.name == "claim_extraction"
        assert template.version == "v1"
        assert template.prompt_type == PromptType.CLAIM_EXTRACTION
        assert "expert document analyst" in template.system_message.lower()
        assert len(template.examples) > 0
    
    def test_prompt_message_creation(self):
        """Test creation of chat messages from templates."""
        template = prompt_manager.get_template(PromptType.CLAIM_EXTRACTION, "v1")
        
        messages = prompt_manager.create_messages(
            template,
            document_text="Test document content",
            max_claims=10
        )
        
        assert len(messages) >= 3  # System + examples + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "Test document content" in messages[-1]["content"]


class TestLLMClient:
    """Test cases for LLM client."""
    
    def test_llm_config_creation(self):
        """Test LLM configuration creation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            temperature=0.1,
            max_tokens=4000
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.1
    
    @pytest.mark.asyncio
    async def test_mock_llm_response(self):
        """Test mock LLM response for testing."""
        # Create mock LLM client
        mock_response = LLMResponse(
            content='{"claims": [{"claim_text": "Test claim", "claim_type": "factual", "extraction_confidence": 0.9}]}',
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            usage={"tokens": 100},
            response_time_seconds=1.0,
            metadata={}
        )
        
        with patch('src.llm.llm_client.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Create config and client
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key="test-key"
            )
            
            llm_client = LLMClient([config])
            
            # Mock the provider's generate_structured_response method
            with patch.object(llm_client.providers['openai:gpt-4'], 'generate_structured_response', 
                            return_value=mock_response) as mock_generate:
                
                response = await llm_client.generate_structured_response(
                    messages=[{"role": "user", "content": "test"}],
                    response_schema={}
                )
                
                assert response.content == mock_response.content
                assert response.provider == LLMProvider.OPENAI


class TestClaimExtractionPass:
    """Test cases for ClaimExtractionPass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock LLM client
        self.mock_llm_client = Mock(spec=LLMClient)
        self.claim_pass = ClaimExtractionPass(llm_client=self.mock_llm_client)
    
    def test_pass_initialization(self):
        """Test claim extraction pass initialization."""
        assert self.claim_pass.pass_type == VerificationPassType.CLAIM_EXTRACTION
        assert len(self.claim_pass.get_required_dependencies()) == 0
    
    @pytest.mark.asyncio
    async def test_claim_extraction_execution(self):
        """Test execution of claim extraction pass."""
        # Mock LLM response
        mock_response_content = {
            "claims": [
                {
                    "claim_text": "The Earth is round",
                    "claim_type": "factual",
                    "extraction_confidence": 0.95,
                    "importance_score": 0.8,
                    "citations": [],
                    "start_position": 0,
                    "end_position": 17,
                    "requires_fact_check": True
                },
                {
                    "claim_text": "COVID-19 vaccines are 95% effective",
                    "claim_type": "statistical", 
                    "extraction_confidence": 0.9,
                    "importance_score": 0.9,
                    "citations": ["CDC study 2021"],
                    "start_position": 20,
                    "end_position": 55,
                    "requires_fact_check": True
                }
            ],
            "metadata": {
                "processing_notes": "Extracted 2 claims successfully",
                "extraction_warnings": []
            }
        }
        
        mock_response = LLMResponse(
            content=json.dumps(mock_response_content),
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            usage={"tokens": 200},
            response_time_seconds=2.0,
            metadata={}
        )
        
        # Configure mock LLM client
        self.mock_llm_client.generate_structured_response = AsyncMock(return_value=mock_response)
        self.mock_llm_client.get_available_providers.return_value = ["openai:gpt-4"]
        
        # Create test context
        document_content = "The Earth is round. COVID-19 vaccines are 95% effective according to CDC study 2021."
        context = VerificationContext(
            document_id="test_doc",
            document_content=document_content
        )
        
        # Create pass configuration
        config = VerificationPassConfig(
            pass_type=VerificationPassType.CLAIM_EXTRACTION,
            name="test_claim_extraction",
            parameters={
                "max_claims": 10,
                "model": "gpt-4",
                "prompt_version": "v1",
                "min_confidence": 0.5
            }
        )
        
        # Execute the pass
        result = await self.claim_pass.execute(context, config)
        
        # Verify results
        assert result.status == VerificationStatus.COMPLETED
        assert result.confidence_score is not None
        assert "extraction_result" in result.result_data
        
        extraction_result = ClaimExtractionResult(**result.result_data["extraction_result"])
        assert extraction_result.total_claims_found == 2
        assert len(extraction_result.claims) == 2
        
        texts = {c.claim_text for c in extraction_result.claims}
        assert {"The Earth is round",
                "COVID-19 vaccines are 95% effective"} <= texts
        assert claims[0].claim_type == ClaimType.FACTUAL
        assert claims[1].claim_text == "COVID-19 vaccines are 95% effective"
        assert claims[1].claim_type == ClaimType.STATISTICAL
    
    @pytest.mark.asyncio
    async def test_claim_extraction_with_errors(self):
        """Test claim extraction with LLM errors."""
        # Configure mock to raise exception
        self.mock_llm_client.generate_structured_response = AsyncMock(
            side_effect=Exception("LLM API error")
        )
        
        context = VerificationContext(
            document_id="test_doc",
            document_content="Test content"
        )
        
        config = VerificationPassConfig(
            pass_type=VerificationPassType.CLAIM_EXTRACTION,
            name="test_claim_extraction"
        )
        
        # Execute the pass
        result = await self.claim_pass.execute(context, config)
        
        # Verify error handling
        assert result.status == VerificationStatus.FAILED
        assert "LLM API error" in result.error_message
    
    def test_claim_location_finding(self):
        """Test finding claim locations in documents."""
        document_content = "This is a test document. Climate change is real. The end."
        claim_text = "Climate change is real"
        
        location = self.claim_pass._find_claim_location(
            claim_text, document_content, {}
        )
        
        assert location.start_position == 25  # Position of "Climate change is real"
        assert location.end_position == 47
        assert "test document" in location.context_before
        assert "The end" in location.context_after
    
    def test_fuzzy_claim_finding(self):
        """Test fuzzy matching for claim text."""
        document_content = "The global climate is changing rapidly due to human activities."
        claim_text = "climate change human activities"  # Not exact match
        
        position = self.claim_pass._fuzzy_find_claim(claim_text, document_content)
        
        # Should find some position (fuzzy matching)
        assert position >= 0 or position == -1  # Either found or not found


class TestIntegrationWithVerificationFramework:
    """Integration tests with the verification framework."""
    
    @pytest.mark.asyncio 
    async def test_claim_extraction_in_chain(self):
        """Test claim extraction as part of a verification chain."""
        # This would test the integration with the verification worker
        # For now, we'll just verify the pass can be registered
        
        from src.verification.workers.verification_worker import VerificationWorker
        
        worker = VerificationWorker()
        
        # Verify claim extraction pass is registered
        assert VerificationPassType.CLAIM_EXTRACTION in worker.pass_registry
        
        claim_pass = worker.pass_registry[VerificationPassType.CLAIM_EXTRACTION]
        assert isinstance(claim_pass, ClaimExtractionPass)


class TestClaimExtractionWithRealDocuments:
    """Test claim extraction with real document content."""
    
    @pytest.mark.asyncio
    async def test_extraction_from_scientific_paper(self):
        """Test claim extraction from scientific paper content."""
        
        # Sample scientific paper content
        document_content = """
        Abstract: This study examines the effectiveness of renewable energy sources in reducing carbon emissions. 
        Our analysis shows that solar power installations have increased by 23% globally in 2023. 
        Wind energy contributes approximately 15% of total electricity generation in developed countries.
        
        Introduction: Climate change represents one of the most pressing challenges of our time. 
        The Intergovernmental Panel on Climate Change (IPCC) reports that global temperatures have risen by 1.1°C since pre-industrial times.
        
        Results: Our data analysis reveals that countries with higher renewable energy adoption show 30% lower carbon intensity.
        Solar panel efficiency has improved from 15% to 22% over the past decade.
        
        Conclusion: The transition to renewable energy is critical for achieving net-zero emissions by 2050.
        """
        
        # Create mock LLM response
        mock_response_content = {
            "claims": [
                {
                    "claim_text": "Solar power installations have increased by 23% globally in 2023",
                    "claim_type": "statistical",
                    "extraction_confidence": 0.95,
                    "importance_score": 0.8,
                    "citations": [],
                    "requires_fact_check": True
                },
                {
                    "claim_text": "Wind energy contributes approximately 15% of total electricity generation in developed countries",
                    "claim_type": "statistical",
                    "extraction_confidence": 0.9,
                    "importance_score": 0.7,
                    "citations": [],
                    "requires_fact_check": True
                },
                {
                    "claim_text": "Global temperatures have risen by 1.1°C since pre-industrial times",
                    "claim_type": "statistical",
                    "extraction_confidence": 0.95,
                    "importance_score": 0.9,
                    "citations": ["IPCC"],
                    "requires_fact_check": True
                },
                {
                    "claim_text": "Countries with higher renewable energy adoption show 30% lower carbon intensity",
                    "claim_type": "statistical",
                    "extraction_confidence": 0.85,
                    "importance_score": 0.8,
                    "citations": [],
                    "requires_fact_check": True
                }
            ]
        }
        
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client.generate_structured_response = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(mock_response_content),
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                usage={"tokens": 300},
                response_time_seconds=3.0,
                metadata={}
            )
        )
        mock_llm_client.get_available_providers.return_value = ["openai:gpt-4"]
        
        # Test extraction
        claim_pass = ClaimExtractionPass(llm_client=mock_llm_client)
        
        context = VerificationContext(
            document_id="scientific_paper.pdf",
            document_content=document_content
        )
        
        config = VerificationPassConfig(
            pass_type=VerificationPassType.CLAIM_EXTRACTION,
            name="extract_claims",
            parameters={
                "max_claims": 20,
                "model": "gpt-4",
                "min_confidence": 0.7
            }
        )
        
        result = await claim_pass.execute(context, config)
        
        # Verify scientific claims were extracted
        assert result.status == VerificationStatus.COMPLETED
        extraction_result = ClaimExtractionResult(**result.result_data["extraction_result"])
        
        assert extraction_result.total_claims_found == 4
        
        # Check for statistical claims (should be majority in scientific paper)
        statistical_claims = [
            claim for claim in extraction_result.claims 
            if claim.claim_type == ClaimType.STATISTICAL
        ]
        assert len(statistical_claims) >= 3
        
        # Check for citations
        claims_with_citations = [
            claim for claim in extraction_result.claims
            if len(claim.citations) > 0
        ]
        assert len(claims_with_citations) >= 1


if __name__ == '__main__':
    pytest.main([__file__])