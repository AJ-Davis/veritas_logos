"""
Citation verification pass implementation.
"""

import json
import logging
import os
import re
import time
import uuid
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

from src.verification.passes.base_pass import BaseVerificationPass
from src.models.verification import (
    VerificationPassConfig,
    VerificationResult,
    VerificationContext,
    VerificationPassType,
    VerificationStatus,
    VerificationError
)
from src.models.claims import ExtractedClaim, ClaimExtractionResult
from src.models.citations import (
    VerifiedCitation,
    CitationVerificationResult,
    CitationStatus,
    CitationType,
    SupportLevel,
    CitationIssue,
    CitationLocation,
    CitationContent,
    SourceCredibility
)
from src.llm.llm_client import LLMClient, LLMConfig, LLMProvider
from src.llm.prompts import PromptType, prompt_manager


logger = logging.getLogger(__name__)


class CitationVerificationPass(BaseVerificationPass):
    """
    Verification pass for checking citations against their associated claims using LLMs.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the citation verification pass.
        
        Args:
            llm_client: Optional LLM client. If None, will create default client.
        """
        super().__init__(VerificationPassType.CITATION_CHECK)
        self.llm_client = llm_client or self._create_default_llm_client()
        self.http_timeout = aiohttp.ClientTimeout(total=30)
    
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
        """Citation verification depends on claim extraction."""
        return [VerificationPassType.CLAIM_EXTRACTION]
    
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """
        Execute citation verification on extracted claims.
        
        Args:
            context: Verification context containing document content and previous results
            config: Pass configuration
            
        Returns:
            VerificationResult with citation verification results
        """
        start_time = time.time()
        pass_id = f"citation_verification_{uuid.uuid4().hex}"
        
        try:
            self.logger.info(f"Starting citation verification for document {context.document_id}")
            
            # Extract parameters from config
            parameters = self.extract_parameters(config)
            max_citations = parameters.get('max_citations', 20)
            model = parameters.get('model', 'gpt-4')
            prompt_version = parameters.get('prompt_version', 'v1')
            min_confidence = parameters.get('min_confidence', 0.3)
            retrieve_content = parameters.get('retrieve_content', True)
            
            # Get claims from previous pass results
            claims = self._extract_claims_from_context(context)
            if not claims:
                logger.warning(f"No claims found in context for document {context.document_id}")
                return self.create_result(
                    pass_id=pass_id,
                    status=VerificationStatus.COMPLETED,
                    result_data={
                        "verification_result": CitationVerificationResult(
                            document_id=context.document_id,
                            model_used=model
                        ).model_dump(),
                        "warning": "No claims with citations found to verify"
                    },
                    confidence_score=1.0
                )
            
            # Filter claims that have citations
            claims_with_citations = [claim for claim in claims if claim.citations]
            if not claims_with_citations:
                logger.info(f"No claims with citations found for document {context.document_id}")
                return self.create_result(
                    pass_id=pass_id,
                    status=VerificationStatus.COMPLETED,
                    result_data={
                        "verification_result": CitationVerificationResult(
                            document_id=context.document_id,
                            model_used=model,
                            total_claims_found=len(claims),
                            claims_without_citations=len(claims)
                        ).model_dump(),
                        "info": f"Found {len(claims)} claims but none have citations to verify"
                    },
                    confidence_score=1.0
                )
            
            # Verify citations using LLM
            verification_result = await self._verify_citations_with_llm(
                claims_with_citations=claims_with_citations,
                document_content=context.document_content,
                document_id=context.document_id,
                max_citations=max_citations,
                model=model,
                prompt_version=prompt_version,
                min_confidence=min_confidence,
                retrieve_content=retrieve_content
            )
            
            # Calculate overall confidence score
            if verification_result.verified_citations:
                confidence = verification_result.average_confidence or 0.7
            else:
                confidence = 0.5  # Neutral confidence when no citations verified
            
            # Prepare result data
            result_data = {
                "verification_result": verification_result.model_dump(),
                "citations_verified": verification_result.total_citations_verified,
                "valid_citations": verification_result.valid_citations,
                "invalid_citations": verification_result.invalid_citations,
                "average_confidence": verification_result.average_confidence,
                "model_used": verification_result.model_used,
                "processing_statistics": {
                    "total_claims": len(claims),
                    "claims_with_citations": len(claims_with_citations),
                    "claims_without_citations": verification_result.claims_without_citations,
                    "citation_accuracy_rate": verification_result.citation_accuracy_rate,
                    "support_level_distribution": verification_result.support_level_distribution,
                    "issue_distribution": verification_result.issue_distribution
                }
            }
            
            # Add warnings and errors
            if verification_result.verification_warnings:
                result_data["warnings"] = verification_result.verification_warnings
            if verification_result.verification_errors:
                result_data["errors"] = verification_result.verification_errors
            
            self.logger.info(f"Verified {verification_result.total_citations_verified} citations from document {context.document_id}")
            
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.COMPLETED,
                result_data=result_data,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Citation verification failed: {str(e)}")
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.FAILED,
                error_message=f"Citation verification failed: {str(e)}"
            )
    
    def _extract_claims_from_context(self, context: VerificationContext) -> List[ExtractedClaim]:
        """
        Extract claims from previous verification pass results in the context.
        
        Args:
            context: Verification context
            
        Returns:
            List of ExtractedClaim objects
        """
        claims = []
        
        # Look for claim extraction results in previous pass results
        for pass_result in context.previous_results:
            if pass_result.pass_type == VerificationPassType.CLAIM_EXTRACTION:
                if pass_result.result_data and "extraction_result" in pass_result.result_data:
                    extraction_data = pass_result.result_data["extraction_result"]
                    if "claims" in extraction_data:
                        for claim_data in extraction_data["claims"]:
                            try:
                                claim = ExtractedClaim(**claim_data)
                                claims.append(claim)
                            except Exception as e:
                                logger.warning(f"Failed to parse claim data: {e}")
                                continue
                break
        
        return claims
    
    async def _verify_citations_with_llm(self, claims_with_citations: List[ExtractedClaim],
                                       document_content: str, document_id: str,
                                       max_citations: int, model: str, prompt_version: str,
                                       min_confidence: float, retrieve_content: bool) -> CitationVerificationResult:
        """
        Verify citations using LLM analysis.
        
        Args:
            claims_with_citations: Claims that have citations to verify
            document_content: Content of the document
            document_id: Document identifier
            max_citations: Maximum number of citations to verify
            model: LLM model to use
            prompt_version: Version of prompt template
            min_confidence: Minimum confidence threshold
            retrieve_content: Whether to attempt content retrieval
            
        Returns:
            CitationVerificationResult with verification results
        """
        start_time = time.time()
        
        # Prepare citations for verification
        citations_to_verify = []
        total_citations_found = 0
        
        for claim in claims_with_citations:
            for citation_text in claim.citations:
                total_citations_found += 1
                if len(citations_to_verify) < max_citations:
                    citations_to_verify.append({
                        "claim": claim,
                        "citation_text": citation_text
                    })
        
        verified_citations = []
        verification_warnings = []
        verification_errors = []
        
        # Process citations in batches to avoid overwhelming the LLM
        batch_size = 5
        for i in range(0, len(citations_to_verify), batch_size):
            batch = citations_to_verify[i:i + batch_size]
            
            try:
                batch_results = await self._verify_citation_batch(
                    citation_batch=batch,
                    document_content=document_content,
                    document_id=document_id,
                    model=model,
                    prompt_version=prompt_version,
                    retrieve_content=retrieve_content
                )
                verified_citations.extend(batch_results)
                
                # Add small delay between batches to be respectful to APIs
                if i + batch_size < len(citations_to_verify):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                error_msg = f"Failed to verify citation batch {i//batch_size + 1}: {str(e)}"
                verification_errors.append(error_msg)
                logger.error(error_msg)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        
        valid_count = sum(1 for c in verified_citations if c.verification_status == CitationStatus.VALID)
        invalid_count = sum(1 for c in verified_citations if c.verification_status == CitationStatus.INVALID)
        inaccessible_count = sum(1 for c in verified_citations if c.verification_status == CitationStatus.INACCESSIBLE)
        
        # Calculate average confidence
        confidences = [c.confidence_score for c in verified_citations if c.confidence_score is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Calculate citation accuracy rate
        verifiable_citations = valid_count + invalid_count
        accuracy_rate = valid_count / verifiable_citations if verifiable_citations > 0 else None
        
        # Calculate distributions
        support_distribution = {}
        for citation in verified_citations:
            level = citation.support_level.value
            support_distribution[level] = support_distribution.get(level, 0) + 1
        
        issue_distribution = {}
        for citation in verified_citations:
            for issue in citation.identified_issues:
                issue_name = issue.value
                issue_distribution[issue_name] = issue_distribution.get(issue_name, 0) + 1
        
        return CitationVerificationResult(
            document_id=document_id,
            verified_citations=verified_citations,
            total_citations_found=total_citations_found,
            total_citations_verified=len(verified_citations),
            valid_citations=valid_count,
            invalid_citations=invalid_count,
            inaccessible_citations=inaccessible_count,
            average_confidence=avg_confidence,
            citation_accuracy_rate=accuracy_rate,
            support_level_distribution=support_distribution,
            issue_distribution=issue_distribution,
            model_used=model,
            processing_time_seconds=processing_time,
            verification_warnings=verification_warnings,
            verification_errors=verification_errors,
            claims_with_citations=len(claims_with_citations),
            claims_without_citations=len([c for c in claims_with_citations if not c.citations]),
            parameters_used={
                "max_citations": max_citations,
                "model": model,
                "prompt_version": prompt_version,
                "min_confidence": min_confidence,
                "retrieve_content": retrieve_content
            }
        )
    
    async def _verify_citation_batch(self, citation_batch: List[Dict[str, Any]],
                                   document_content: str, document_id: str,
                                   model: str, prompt_version: str,
                                   retrieve_content: bool) -> List[VerifiedCitation]:
        """
        Verify a batch of citations using LLM.
        
        Args:
            citation_batch: Batch of citations to verify
            document_content: Document content
            document_id: Document ID
            model: LLM model to use
            prompt_version: Prompt version
            retrieve_content: Whether to retrieve content
            
        Returns:
            List of VerifiedCitation objects
        """
        # Prepare claims and citations data for the prompt
        claims_citations_data = []
        for item in citation_batch:
            claim = item["claim"]
            citation_text = item["citation_text"]
            
            # Attempt content retrieval if enabled
            retrieved_content = None
            if retrieve_content:
                retrieved_content = await self._attempt_content_retrieval(citation_text)
            
            claims_citations_data.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "citation_text": citation_text,
                "retrieved_content": retrieved_content.retrieved_content if retrieved_content else None,
                "retrieval_success": retrieved_content.retrieval_success if retrieved_content else False
            })
        
        # Get prompt template
        template = prompt_manager.get_template(PromptType.CITATION_CHECK, prompt_version)
        
        # Format the claims and citations for the prompt
        formatted_data = "\n\n".join([
            f"Claim ID: {item['claim_id']}\n"
            f"Claim: {item['claim_text']}\n"
            f"Citation: {item['citation_text']}\n"
            f"Retrieved Content: {item['retrieved_content'] or 'Not available'}\n"
            f"Content Retrieved Successfully: {item['retrieval_success']}"
            for item in claims_citations_data
        ])
        
        # Prepare prompt parameters
        prompt_kwargs = {
            'claims_citations_data': formatted_data,
            'max_citations': len(citation_batch),
            'min_confidence': 0.3
        }
        
        try:
            # Create messages for the LLM
            messages = prompt_manager.create_messages(template, **prompt_kwargs)
            
            # Call LLM
            llm_response = await self.llm_client.generate_response(
                messages=messages,
                model=model,
                temperature=0.1,
                max_tokens=4000
            )
            response = llm_response.content
            
            # Parse LLM response
            verification_results = self._parse_verification_response(response, citation_batch, document_id, model)
            
            return verification_results
            
        except Exception as e:
            logger.error(f"LLM verification failed for batch: {str(e)}")
            # Return default failed verifications for the batch
            return [
                self._create_failed_verification(
                    item["claim"],
                    item["citation_text"],
                    document_id,
                    model,
                    f"LLM verification failed: {str(e)}"
                )
                for item in citation_batch
            ]
    
    async def _attempt_content_retrieval(self, citation_text: str) -> Optional[CitationContent]:
        """
        Attempt to retrieve content from a citation source.
        
        Args:
            citation_text: Citation text that may contain URLs
            
        Returns:
            CitationContent if retrieval was attempted, None otherwise
        """
        # Extract URLs from citation text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, citation_text)
        
        if not urls:
            return CitationContent(
                retrieved_content="",
                retrieval_method="no_url_found",
                retrieval_success=False,
                retrieval_errors=["No URL found in citation text"]
            )
        
        # Try to retrieve content from the first URL
        url = urls[0]
        if not url.startswith('http'):
            url = 'https://' + url
        
        try:
            async with aiohttp.ClientSession(timeout=self.http_timeout) as session:
                async with session.get(url, headers={'User-Agent': 'Veritas-Logos-Citation-Checker/1.0'}) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Truncate content to avoid overwhelming the LLM
                        truncated_content = content[:2000] + "..." if len(content) > 2000 else content
                        return CitationContent(
                            retrieved_content=truncated_content,
                            retrieval_method="http_get",
                            retrieval_success=True
                        )
                    else:
                        return CitationContent(
                            retrieved_content="",
                            retrieval_method="http_get",
                            retrieval_success=False,
                            retrieval_errors=[f"HTTP {response.status}: {response.reason}"]
                        )
        except Exception as e:
            return CitationContent(
                retrieved_content="",
                retrieval_method="http_get",
                retrieval_success=False,
                retrieval_errors=[f"Content retrieval failed: {str(e)}"]
            )
    
    def _parse_verification_response(self, response: str, citation_batch: List[Dict[str, Any]],
                                   document_id: str, model: str) -> List[VerifiedCitation]:
        """
        Parse LLM response into VerifiedCitation objects.
        
        Args:
            response: LLM response
            citation_batch: Original citation batch
            document_id: Document ID
            model: Model used
            
        Returns:
            List of VerifiedCitation objects
        """
        try:
            # Try to parse as JSON
            if response.strip().startswith('['):
                results_data = json.loads(response)
            else:
                # Look for JSON array in the response
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    results_data = json.loads(json_match.group())
                else:
                    # Fallback: create default results
                    return [self._create_default_verification(item["claim"], item["citation_text"], document_id, model)
                           for item in citation_batch]
            
            verified_citations = []
            for i, result_data in enumerate(results_data):
                if i >= len(citation_batch):
                    break
                    
                citation_item = citation_batch[i]
                claim = citation_item["claim"]
                citation_text = citation_item["citation_text"]
                
                verified_citation = self._create_verified_citation_from_result(
                    result_data, claim, citation_text, document_id, model
                )
                verified_citations.append(verified_citation)
            
            # Fill in any missing citations with defaults
            while len(verified_citations) < len(citation_batch):
                missing_item = citation_batch[len(verified_citations)]
                default_citation = self._create_default_verification(
                    missing_item["claim"], missing_item["citation_text"], document_id, model
                )
                verified_citations.append(default_citation)
            
            return verified_citations
            
        except Exception as e:
            logger.error(f"Failed to parse verification response: {str(e)}")
            # Return default verifications for all items in batch
            return [self._create_default_verification(item["claim"], item["citation_text"], document_id, model)
                   for item in citation_batch]
    
    def _create_verified_citation_from_result(self, result_data: Dict[str, Any],
                                            claim: ExtractedClaim, citation_text: str,
                                            document_id: str, model: str) -> VerifiedCitation:
        """
        Create a VerifiedCitation from LLM result data.
        
        Args:
            result_data: LLM result data
            claim: Associated claim
            citation_text: Citation text
            document_id: Document ID
            model: Model used
            
        Returns:
            VerifiedCitation object
        """
        # Parse verification status
        status_str = result_data.get("verification_status", "pending").lower()
        try:
            verification_status = CitationStatus(status_str)
        except ValueError:
            verification_status = CitationStatus.PENDING
        
        # Parse support level
        support_str = result_data.get("support_level", "no_support").lower()
        try:
            support_level = SupportLevel(support_str)
        except ValueError:
            support_level = SupportLevel.NO_SUPPORT
        
        # Parse confidence score
        confidence = float(result_data.get("confidence_score", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        # Parse issues
        issues = []
        issue_strs = result_data.get("issues", [])
        for issue_str in issue_strs:
            try:
                issue = CitationIssue(issue_str.lower())
                issues.append(issue)
            except ValueError:
                continue
        
        # Determine citation type (basic heuristics)
        citation_type = self._classify_citation_type(citation_text)
        
        # Find citation location in document
        location = self._find_citation_location(citation_text, claim.location)
        
        # Create source credibility if provided
        source_credibility = None
        if "source_credibility" in result_data:
            cred_data = result_data["source_credibility"]
            source_credibility = SourceCredibility(
                credibility_score=float(cred_data.get("credibility_score", 0.5)),
                domain_authority=cred_data.get("domain_authority"),
                author_expertise=cred_data.get("author_expertise"),
                publication_quality=cred_data.get("publication_quality"),
                peer_review_status=cred_data.get("peer_review_status"),
                assessment_method="llm_analysis",
                assessment_details=cred_data
            )
        
        return VerifiedCitation(
            citation_text=citation_text,
            citation_type=citation_type,
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            location=location,
            document_id=document_id,
            verification_status=verification_status,
            support_level=support_level,
            confidence_score=confidence,
            source_credibility=source_credibility,
            identified_issues=issues,
            issue_descriptions=result_data.get("recommendations", []),
            requires_manual_review=confidence < 0.5 or len(issues) > 0,
            verified_by=model,
            verification_metadata={
                "llm_explanation": result_data.get("explanation", ""),
                "raw_result": result_data
            }
        )
    
    def _create_default_verification(self, claim: ExtractedClaim, citation_text: str,
                                   document_id: str, model: str) -> VerifiedCitation:
        """Create a default verification when parsing fails."""
        return VerifiedCitation(
            citation_text=citation_text,
            citation_type=self._classify_citation_type(citation_text),
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            location=self._find_citation_location(citation_text, claim.location),
            document_id=document_id,
            verification_status=CitationStatus.PENDING,
            support_level=SupportLevel.NO_SUPPORT,
            confidence_score=0.1,
            identified_issues=[CitationIssue.FORMATTING_ERROR],
            requires_manual_review=True,
            verified_by=model,
            verification_metadata={"error": "Failed to parse LLM response"}
        )
    
    def _create_failed_verification(self, claim: ExtractedClaim, citation_text: str,
                                  document_id: str, model: str, error_msg: str) -> VerifiedCitation:
        """Create a failed verification result."""
        return VerifiedCitation(
            citation_text=citation_text,
            citation_type=self._classify_citation_type(citation_text),
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            location=self._find_citation_location(citation_text, claim.location),
            document_id=document_id,
            verification_status=CitationStatus.PENDING,
            support_level=SupportLevel.NO_SUPPORT,
            confidence_score=0.0,
            identified_issues=[CitationIssue.FORMATTING_ERROR],
            requires_manual_review=True,
            verified_by=model,
            verification_metadata={"error": error_msg}
        )
    
    def _classify_citation_type(self, citation_text: str) -> CitationType:
        """
        Classify citation type based on text content.
        
        Args:
            citation_text: Citation text
            
        Returns:
            CitationType
        """
        citation_lower = citation_text.lower()
        
        if any(domain in citation_lower for domain in ['.gov', '.edu', 'government', 'nasa', 'cdc']):
            return CitationType.GOVERNMENT_DOCUMENT
        elif any(term in citation_lower for term in ['journal', 'doi:', 'pubmed', 'arxiv']):
            return CitationType.ACADEMIC_PAPER
        elif any(term in citation_lower for term in ['news', 'reuters', 'ap news', 'bbc', 'cnn']):
            return CitationType.NEWS_ARTICLE
        elif 'wikipedia' in citation_lower:
            return CitationType.WEBSITE
        elif any(term in citation_lower for term in ['book', 'isbn', 'publisher']):
            return CitationType.BOOK
        elif 'http' in citation_lower or 'www.' in citation_lower:
            return CitationType.WEBSITE
        else:
            return CitationType.OTHER
    
    def _find_citation_location(self, citation_text: str, claim_location) -> CitationLocation:
        """
        Find citation location relative to claim location.
        
        Args:
            citation_text: Citation text
            claim_location: Location of the associated claim
            
        Returns:
            CitationLocation
        """
        # For now, use the claim location as approximation
        # In a more sophisticated implementation, we would search for the citation in the document
        return CitationLocation(
            start_position=claim_location.start_position,
            end_position=claim_location.end_position + len(citation_text),
            page_number=claim_location.page_number,
            section_type=claim_location.section_type,
            context_before=claim_location.context_before,
            context_after=citation_text[:200]
        ) 