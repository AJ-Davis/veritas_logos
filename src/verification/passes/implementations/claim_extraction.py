"""
Claim Extraction Pass

Extracts key claims and assertions from documents for verification.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ..base_pass import BaseVerificationPass
from ....models.verification import VerificationResult, VerificationStatus

# Use the new enhanced LLM service
try:
    from ....llm.enhanced_llm_service import get_enhanced_llm_service
    ENHANCED_LLM_AVAILABLE = True
except ImportError:
    ENHANCED_LLM_AVAILABLE = False
    # Fallback to old service
    from ....llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class ClaimExtractionPass(BaseVerificationPass):
    """
    Pass that extracts key claims and assertions from documents.
    """
    
    def __init__(self):
        super().__init__()
        self.pass_name = "claim_extraction"
        self.description = "Extract key claims and assertions from the document"
        
        # Initialize LLM service
        if ENHANCED_LLM_AVAILABLE:
            self.llm_service = get_enhanced_llm_service()
            self.use_enhanced_llm = True
        else:
            self.llm_service = LLMService()
            self.use_enhanced_llm = False
    
    async def execute(self, 
                     document_text: str, 
                     context: Dict[str, Any]) -> VerificationResult:
        """
        Extract claims from the document.
        
        Args:
            document_text: The text to analyze
            context: Additional context for verification
            
        Returns:
            VerificationResult with extracted claims
        """
        try:
            logger.info(f"Starting claim extraction for document")
            
            # Enhanced prompt for better claim extraction
            system_prompt = """You are an expert fact-checker and claim analyst. Your task is to carefully analyze documents and extract specific, verifiable claims.

Instructions:
1. Identify factual assertions, statistics, quotes, and claims made in the document
2. Focus on statements that can be fact-checked against external sources
3. Extract the most important and significant claims (limit to 10-15 key claims)
4. For each claim, provide the exact text and context where it appears
5. Categorize claims by type (factual, statistical, expert opinion, prediction, etc.)
6. Avoid extracting obvious facts, definitions, or generally accepted knowledge

Format your response as a structured analysis with:
- Total number of claims found
- List of extracted claims with categories
- Assessment of claim significance
- Recommendations for verification priority"""
            
            user_prompt = f"""Please extract and analyze the key claims from this document:

{document_text}

Provide a comprehensive claim extraction analysis following the specified format."""
            
            # Generate response using appropriate LLM service
            if self.use_enhanced_llm:
                # Use enhanced service with provider fallback
                response = await self.llm_service.generate_text(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=2048,
                    temperature=0.3,
                    preferred_provider=None  # Let it auto-select best provider
                )
                
                # Enhanced response includes provider info
                provider_info = response.provider.value if hasattr(response, 'provider') else "unknown"
                logger.info(f"Claim extraction completed using {provider_info}")
                analysis_text = response.content
                
            else:
                # Use legacy service
                analysis_text = await self.llm_service.generate_response(
                    f"{system_prompt}\n\n{user_prompt}"
                )
                provider_info = "legacy"
            
            # Extract structured data from the analysis
            claims = self._parse_claims_from_analysis(analysis_text)
            
            # Create result
            result = VerificationResult(
                pass_name=self.pass_name,
                status=VerificationStatus.COMPLETED,
                score=1.0,  # Claim extraction is binary - either works or fails
                details={
                    "claims_extracted": len(claims),
                    "claims": claims,
                    "analysis": analysis_text,
                    "provider_used": provider_info,
                    "timestamp": datetime.now().isoformat()
                },
                evidence=[],
                metadata={
                    "document_length": len(document_text),
                    "processing_method": "enhanced_llm" if self.use_enhanced_llm else "legacy_llm"
                }
            )
            
            logger.info(f"Claim extraction completed: {len(claims)} claims found")
            return result
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return VerificationResult(
                pass_name=self.pass_name,
                status=VerificationStatus.FAILED,
                score=0.0,
                details={
                    "error": str(e),
                    "claims_extracted": 0,
                    "analysis": "Failed to extract claims due to error",
                    "provider_used": "none",
                    "timestamp": datetime.now().isoformat()
                },
                evidence=[],
                metadata={"error_type": type(e).__name__}
            )
    
    def _parse_claims_from_analysis(self, analysis_text: str) -> List[Dict[str, Any]]:
        """
        Parse claims from the LLM analysis text.
        
        Args:
            analysis_text: The analysis from the LLM
            
        Returns:
            List of extracted claims with metadata
        """
        claims = []
        
        try:
            # Look for patterns in the analysis that indicate claims
            lines = analysis_text.split('\n')
            current_claim = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for numbered claims or bullet points
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')) or
                    line.startswith(('•', '-', '*')) or
                    'claim' in line.lower()):
                    
                    if current_claim:
                        claims.append(current_claim)
                    
                    # Extract claim text (remove numbering/bullets)
                    claim_text = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '•', '-', '*']:
                        if claim_text.startswith(prefix):
                            claim_text = claim_text[len(prefix):].strip()
                            break
                    
                    current_claim = {
                        "text": claim_text,
                        "category": self._categorize_claim(claim_text),
                        "priority": "medium",  # Default priority
                        "verifiable": True
                    }
                
                elif current_claim and line:
                    # Continue previous claim
                    current_claim["text"] += " " + line
            
            # Add the last claim
            if current_claim:
                claims.append(current_claim)
            
            # If no structured claims found, try to extract from free text
            if not claims:
                claims = self._extract_claims_from_free_text(analysis_text)
            
        except Exception as e:
            logger.warning(f"Failed to parse claims from analysis: {e}")
            # Fallback: create a generic claim from the analysis
            claims = [{
                "text": "Document contains claims requiring verification",
                "category": "general",
                "priority": "medium",
                "verifiable": True
            }]
        
        return claims[:15]  # Limit to 15 claims max
    
    def _categorize_claim(self, claim_text: str) -> str:
        """Categorize a claim based on its content"""
        claim_lower = claim_text.lower()
        
        if any(word in claim_lower for word in ['%', 'percent', 'statistics', 'study', 'research', 'data']):
            return "statistical"
        elif any(word in claim_lower for word in ['said', 'stated', 'according to', 'quote', 'expert']):
            return "expert_opinion"
        elif any(word in claim_lower for word in ['will', 'predict', 'forecast', 'expect', 'future']):
            return "prediction"
        elif any(word in claim_lower for word in ['date', 'year', 'time', 'when', 'occurred']):
            return "historical"
        else:
            return "factual"
    
    def _extract_claims_from_free_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims from free-form text as fallback"""
        # Simple extraction: look for sentences with strong assertions
        sentences = text.split('.')
        claims = []
        
        assertion_words = ['is', 'are', 'was', 'were', 'has', 'have', 'shows', 'demonstrates', 'proves']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(word in sentence.lower() for word in assertion_words) and
                not sentence.lower().startswith(('i', 'you', 'we', 'they', 'this analysis', 'the analysis'))):
                
                claims.append({
                    "text": sentence + ".",
                    "category": self._categorize_claim(sentence),
                    "priority": "medium",
                    "verifiable": True
                })
        
        return claims[:10]  # Limit to 10 fallback claims 