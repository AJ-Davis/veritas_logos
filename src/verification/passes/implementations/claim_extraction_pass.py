"""
Claim extraction verification pass implementation.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.verification.passes.base_pass import BaseVerificationPass
from src.models.verification import (
    VerificationPassConfig,
    VerificationResult,
    VerificationContext,
    VerificationPassType,
    VerificationStatus,
    VerificationError
)
from src.models.claims import (
    ExtractedClaim,
    ClaimExtractionResult,
    ClaimLocation,
    ClaimType,
    ClaimCategory
)
from src.llm.llm_client import LLMClient, LLMConfig, LLMProvider
from src.llm.prompts import PromptType, prompt_manager


logger = logging.getLogger(__name__)


class ClaimExtractionPass(BaseVerificationPass):
    """
    Verification pass for extracting claims from documents using LLMs.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the claim extraction pass.
        
        Args:
            llm_client: Optional LLM client. If None, will create default client.
        """
        super().__init__(VerificationPassType.CLAIM_EXTRACTION)
        self.llm_client = llm_client or self._create_default_llm_client()
    
    def _create_default_llm_client(self) -> LLMClient:
        """Create default LLM client from environment variables."""
        configs = []
        
        # OpenAI configuration
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            configs.append(LLMConfig(
                provider=LLMProvider.OPENAI,
                model=os.getenv('OPENAI_MODEL', 'gpt-4'),
                api_key=openai_key,
                temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '4000'))
            ))
        
        # Anthropic configuration
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            configs.append(LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model=os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'),
                api_key=anthropic_key,
                temperature=float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('ANTHROPIC_MAX_TOKENS', '4000'))
            ))
        
        if not configs:
            # Create mock configuration for testing
            logger.warning("No LLM API keys found, creating mock client for testing")
            configs.append(LLMConfig(
                provider=LLMProvider.OPENAI,
                model='gpt-4',
                api_key='mock-key-for-testing',
                temperature=0.1,
                max_tokens=4000
            ))
        
        return LLMClient(configs)
    
    def get_required_dependencies(self) -> List[VerificationPassType]:
        """Claim extraction has no dependencies."""
        return []
    
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """
        Execute claim extraction on the document.
        
        Args:
            context: Verification context containing document content
            config: Pass configuration
            
        Returns:
            VerificationResult with extracted claims
        """
        start_time = time.time()
        pass_id = f"claim_extraction_{uuid.uuid4().hex}"
        
        try:
            self.logger.info(f"Starting claim extraction for document {context.document_id}")
            
            # Extract parameters from config
            parameters = self.extract_parameters(config)
            max_claims = parameters.get('max_claims', 50)
            model = parameters.get('model', 'gpt-4')
            prompt_version = parameters.get('prompt_version', 'v1')
            min_confidence = parameters.get('min_confidence', 0.3)
            
            # Extract claims using LLM
            extraction_result = await self._extract_claims_with_llm(
                document_content=context.document_content,
                document_id=context.document_id,
                max_claims=max_claims,
                model=model,
                prompt_version=prompt_version,
                min_confidence=min_confidence
            )
            
            # Calculate confidence score
            if extraction_result.claims:
                confidence = extraction_result.average_confidence
            else:
                confidence = 0.5  # Neutral confidence when no claims found
            
            # Prepare result data
            result_data = {
                "extraction_result": extraction_result.dict(),
                "claims_count": extraction_result.total_claims_found,
                "average_confidence": extraction_result.average_confidence,
                "model_used": extraction_result.model_used,
                "prompt_version": extraction_result.prompt_version,
                "processing_statistics": {
                    "document_length": extraction_result.document_length,
                    "claims_per_1000_chars": extraction_result.claims_per_1000_chars,
                    "claim_type_distribution": extraction_result.claim_type_distribution,
                    "confidence_distribution": extraction_result.confidence_distribution
                }
            }
            
            # Add warnings and errors
            if extraction_result.extraction_warnings:
                result_data["warnings"] = extraction_result.extraction_warnings
            if extraction_result.extraction_errors:
                result_data["errors"] = extraction_result.extraction_errors
            
            self.logger.info(f"Extracted {extraction_result.total_claims_found} claims from document {context.document_id}")
            
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.COMPLETED,
                result_data=result_data,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Claim extraction failed: {str(e)}")
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.FAILED,
                error_message=f"Claim extraction failed: {str(e)}"
            )
    
    async def _extract_claims_with_llm(self, document_content: str, document_id: str,
                                     max_claims: int, model: str, prompt_version: str,
                                     min_confidence: float) -> ClaimExtractionResult:
        """
        Extract claims from document using LLM.
        
        Args:
            document_content: Content of the document
            document_id: Document identifier
            max_claims: Maximum number of claims to extract
            model: LLM model to use
            prompt_version: Version of prompt template
            min_confidence: Minimum confidence threshold
            
        Returns:
            ClaimExtractionResult with extracted claims
        """
        start_time = time.time()
        
        # Get prompt template
        template = prompt_manager.get_template(PromptType.CLAIM_EXTRACTION, prompt_version)
        
        # Prepare prompt parameters
        prompt_kwargs = {
            'document_text': document_content,
            'max_claims': max_claims
        }
        
        # Create messages
        messages = prompt_manager.create_messages(template, **prompt_kwargs)
        
        # Define JSON schema for structured response
        response_schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_text": {"type": "string"},
                            "normalized_claim": {"type": "string"},
                            "claim_type": {"type": "string"},
                            "category": {"type": "string"},
                            "start_position": {"type": "integer"},
                            "end_position": {"type": "integer"},
                            "extraction_confidence": {"type": "number"},
                            "importance_score": {"type": "number"},
                            "clarity_score": {"type": "number"},
                            "citations": {"type": "array", "items": {"type": "string"}},
                            "context_before": {"type": "string"},
                            "context_after": {"type": "string"},
                            "requires_fact_check": {"type": "boolean"}
                        },
                        "required": ["claim_text", "claim_type", "extraction_confidence"]
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "processing_notes": {"type": "string"},
                        "extraction_warnings": {"type": "array"},
                        "document_summary": {"type": "string"}
                    }
                }
            },
            "required": ["claims"]
        }
        
        try:
            # Make LLM API call
            response = await self.llm_client.generate_structured_response(
                messages=messages,
                response_schema=response_schema,
                provider=self._get_provider_key(model)
            )
            
            processing_time = time.time() - start_time
            
            # Parse response
            response_data = json.loads(response.content)
            
            # Convert to ExtractedClaim objects
            claims = []
            for claim_data in response_data.get('claims', []):
                try:
                    claim = self._create_extracted_claim(claim_data, document_content, document_id, model)
                    
                    # Filter by confidence
                    if claim.extraction_confidence >= min_confidence:
                        claims.append(claim)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create claim object: {str(e)}")
                    continue
            
            # Calculate statistics
            total_claims = len(claims)
            average_confidence = sum(c.extraction_confidence for c in claims) / total_claims if claims else 0.0
            claims_per_1000_chars = (total_claims / len(document_content)) * 1000 if document_content else 0.0
            
            # Type distribution
            type_distribution = {}
            confidence_distribution = {"high": 0, "medium": 0, "low": 0}
            
            for claim in claims:
                # Type distribution
                claim_type = claim.claim_type.value
                type_distribution[claim_type] = type_distribution.get(claim_type, 0) + 1
                
                # Confidence distribution
                if claim.extraction_confidence >= 0.8:
                    confidence_distribution["high"] += 1
                elif claim.extraction_confidence >= 0.5:
                    confidence_distribution["medium"] += 1
                else:
                    confidence_distribution["low"] += 1
            
            # Extract warnings and errors from metadata
            metadata = response_data.get('metadata', {})
            warnings = metadata.get('extraction_warnings', [])
            errors = []
            
            # Create extraction result
            extraction_result = ClaimExtractionResult(
                document_id=document_id,
                claims=claims,
                total_claims_found=total_claims,
                processing_time_seconds=processing_time,
                model_used=model,
                prompt_version=prompt_version,
                average_confidence=average_confidence,
                confidence_distribution=confidence_distribution,
                claim_type_distribution=type_distribution,
                document_length=len(document_content),
                claims_per_1000_chars=claims_per_1000_chars,
                extraction_warnings=warnings,
                extraction_errors=errors,
                parameters_used={
                    "max_claims": max_claims,
                    "min_confidence": min_confidence,
                    "model": model,
                    "prompt_version": prompt_version
                }
            )
            
            return extraction_result
            
        except json.JSONDecodeError as e:
            raise VerificationError(f"Invalid JSON response from LLM: {str(e)}")
        except Exception as e:
            raise VerificationError(f"LLM claim extraction failed: {str(e)}")
    
    def _create_extracted_claim(self, claim_data: Dict[str, Any], document_content: str,
                              document_id: str, model: str) -> ExtractedClaim:
        """
        Create an ExtractedClaim object from LLM response data.
        
        Args:
            claim_data: Claim data from LLM response
            document_content: Full document content
            document_id: Document identifier
            model: Model used for extraction
            
        Returns:
            ExtractedClaim object
        """
        claim_text = claim_data['claim_text']
        
        # Find claim location in document
        location = self._find_claim_location(claim_text, document_content, claim_data)
        
        # Parse claim type
        try:
            claim_type = ClaimType(claim_data['claim_type'].lower())
        except ValueError:
            claim_type = ClaimType.FACTUAL  # Default fallback
        
        # Parse category
        try:
            category = ClaimCategory(claim_data.get('category', 'general').lower())
        except ValueError:
            category = ClaimCategory.GENERAL
        
        return ExtractedClaim(
            claim_text=claim_text,
            normalized_claim=claim_data.get('normalized_claim'),
            claim_type=claim_type,
            category=category,
            location=location,
            document_id=document_id,
            extraction_confidence=claim_data['extraction_confidence'],
            clarity_score=claim_data.get('clarity_score'),
            importance_score=claim_data.get('importance_score'),
            citations=claim_data.get('citations', []),
            extracted_by=model,
            requires_fact_check=claim_data.get('requires_fact_check', True)
        )
    
    def _find_claim_location(self, claim_text: str, document_content: str,
                           claim_data: Dict[str, Any]) -> ClaimLocation:
        """
        Find the location of a claim in the document.
        
        Args:
            claim_text: The claim text to find
            document_content: Full document content
            claim_data: Additional claim data that might contain position info
            
        Returns:
            ClaimLocation object
        """
        # Use provided positions if available
        start_pos = claim_data.get('start_position')
        end_pos = claim_data.get('end_position')
        
        if start_pos is not None and end_pos is not None:
            # Use provided positions
            context_before = claim_data.get('context_before', '')
            context_after = claim_data.get('context_after', '')
        else:
            # Search for the claim in the document
            start_pos = document_content.find(claim_text)
            if start_pos == -1:
                # Try fuzzy matching for similar text
                start_pos = self._fuzzy_find_claim(claim_text, document_content)
            
            if start_pos != -1:
                end_pos = start_pos + len(claim_text)
                
                # Extract context
                context_start = max(0, start_pos - 100)
                context_end = min(len(document_content), end_pos + 100)
                context_before = document_content[context_start:start_pos]
                context_after = document_content[end_pos:context_end]
            else:
                # Claim not found in document (might be paraphrased)
                start_pos = 0
                end_pos = 0
                context_before = ""
                context_after = ""
        
        return ClaimLocation(
            start_position=start_pos,
            end_position=end_pos,
            context_before=context_before,
            context_after=context_after
        )
    
    def _fuzzy_find_claim(self, claim_text: str, document_content: str) -> int:
        """
        Attempt to find claim text using fuzzy matching.
        
        Args:
            claim_text: Claim text to find
            document_content: Document content to search
            
        Returns:
            Position of best match or -1 if not found
        """
        # Simple approach: look for overlapping words
        claim_words = set(claim_text.lower().split())
        if len(claim_words) < 3:
            return -1
        
        # Search in chunks with safe bounds
        chunk_size = max(6, len(claim_text) * 2)
        step = max(1, chunk_size // 2)
        best_match_pos = -1
        best_match_score = 0
        
        for i in range(0, len(document_content) - chunk_size + 1, step):
            chunk = document_content[i:i + chunk_size]
            chunk_words = set(chunk.lower().split())
            
            # Calculate overlap score
            overlap = len(claim_words & chunk_words)
            score = overlap / len(claim_words)
            
            if score > best_match_score and score > 0.5:
                best_match_score = score
                best_match_pos = i
        
        return best_match_pos
    
    def _get_provider_key(self, model: str) -> str:
        """
        Get provider key for a given model.
        
        Args:
            model: Model name
            
        Returns:
            Provider key string
            
        Raises:
            VerificationError: If no LLM providers are available
        """
        if model.startswith('gpt-'):
            return f"openai:{model}"
        elif model.startswith('claude-'):
            return f"anthropic:{model}"
        else:
            # Return first available provider
            available = self.llm_client.get_available_providers()
            if not available:
                raise VerificationError("No LLM providers available")
            return available[0]