"""
Logic analysis verification pass implementation.
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..base_pass import BaseVerificationPass
from ....models.verification import (
    VerificationPassConfig,
    VerificationResult,
    VerificationContext,
    VerificationPassType,
    VerificationStatus,
    VerificationError
)
from ....models.logic_bias import (
    LogicalIssue,
    LogicAnalysisResult,
    LogicalFallacyType,
    ReasoningIssueType
)
from ....llm.llm_client import LLMClient, LLMConfig, LLMProvider
from ....llm.prompts import PromptType, prompt_manager
from .ml_enhanced_logic import MLEnhancedLogicAnalyzer, MLLogicConfig


logger = logging.getLogger(__name__)


class LogicAnalysisPass(BaseVerificationPass):
    """
    Verification pass for analyzing logical structure and detecting fallacies.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, enable_ml_enhancement: bool = True):
        """
        Initialize the logic analysis pass.
        
        Args:
            llm_client: Optional LLM client. If None, will create default client.
            enable_ml_enhancement: Whether to enable ML-enhanced analysis
        """
        super().__init__(VerificationPassType.LOGIC_ANALYSIS)
        self.llm_client = llm_client or self._create_default_llm_client()
        self.enable_ml_enhancement = enable_ml_enhancement
        self.ml_analyzer = None
        
        if self.enable_ml_enhancement:
            try:
                self.ml_analyzer = MLEnhancedLogicAnalyzer()
                logger.info("ML-enhanced logic analysis enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize ML enhancement: {e}")
                self.enable_ml_enhancement = False
    
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
                model=os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
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
        """Logic analysis can depend on claim extraction for better analysis."""
        return [VerificationPassType.CLAIM_EXTRACTION]
    
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """
        Execute logic analysis on the document.
        
        Args:
            context: Verification context containing document content
            config: Pass configuration
            
        Returns:
            VerificationResult with logic analysis results
        """
        start_time = time.time()
        pass_id = f"logic_analysis_{uuid.uuid4().hex}"
        
        try:
            self.logger.info(f"Starting logic analysis for document {context.document_id}")
            
            # Extract parameters from config
            parameters = self.extract_parameters(config)
            model = parameters.get('model', 'gpt-4')
            prompt_version = parameters.get('prompt_version', 'v1')
            min_confidence = parameters.get('min_confidence', 0.3)
            max_issues = parameters.get('max_issues', 50)
            focus_areas = parameters.get('focus_areas', [])
            
            # Get any previously extracted claims for context
            claims_context = None
            claim_extraction_result = context.get_previous_result(VerificationPassType.CLAIM_EXTRACTION)
            if claim_extraction_result and claim_extraction_result.result_data:
                claims_context = claim_extraction_result.result_data.get('extraction_result')
            
            # Perform logic analysis using LLM
            llm_analysis_result = await self._analyze_logic_with_llm(
                document_content=context.document_content,
                document_id=context.document_id,
                model=model,
                prompt_version=prompt_version,
                min_confidence=min_confidence,
                max_issues=max_issues,
                focus_areas=focus_areas,
                claims_context=claims_context
            )
            
            # Perform ML-enhanced analysis if enabled
            ml_issues = []
            if self.enable_ml_enhancement and self.ml_analyzer:
                try:
                    ml_issues = self.ml_analyzer.analyze_text(
                        context.document_content, 
                        context.document_id
                    )
                    logger.info(f"ML analysis found {len(ml_issues)} additional issues")
                except Exception as e:
                    logger.warning(f"ML analysis failed: {e}")
            
            # Combine LLM and ML results
            all_issues = list(llm_analysis_result.logical_issues) + ml_issues
            
            # Create combined analysis result
            analysis_result = self._create_combined_analysis_result(
                llm_analysis_result, ml_issues, context.document_id, model, prompt_version, context.document_content
            )
            
            # Calculate overall confidence score
            if analysis_result.logical_issues:
                confidence = analysis_result.average_confidence
            else:
                confidence = 0.8  # High confidence when no issues found
            
            # Prepare result data
            result_data = {
                "analysis_result": analysis_result.dict(),
                "total_issues": analysis_result.total_issues_found,
                "overall_logic_score": analysis_result.overall_logic_score,
                "fallacy_counts": {k.value: v for k, v in analysis_result.fallacy_counts.items()},
                "reasoning_issue_counts": {k.value: v for k, v in analysis_result.reasoning_issue_counts.items()},
                "model_used": analysis_result.model_used,
                "prompt_version": analysis_result.prompt_version,
                "processing_statistics": {
                    "average_confidence": analysis_result.average_confidence,
                    "average_severity": analysis_result.average_severity,
                    "severity_distribution": analysis_result.severity_distribution
                }
            }
            
            # Add warnings and errors
            if analysis_result.analysis_warnings:
                result_data["warnings"] = analysis_result.analysis_warnings
            if analysis_result.analysis_errors:
                result_data["errors"] = analysis_result.analysis_errors
            
            self.logger.info(f"Found {analysis_result.total_issues_found} logical issues in document {context.document_id}")
            
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.COMPLETED,
                result_data=result_data,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Logic analysis failed: {str(e)}")
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.FAILED,
                error_message=f"Logic analysis failed: {str(e)}"
            )
    
    async def _analyze_logic_with_llm(self, document_content: str, document_id: str,
                                    model: str, prompt_version: str, min_confidence: float,
                                    max_issues: int, focus_areas: List[str],
                                    claims_context: Optional[Dict[str, Any]]) -> LogicAnalysisResult:
        """
        Analyze document logic using LLM.
        
        Args:
            document_content: Content of the document
            document_id: Document identifier
            model: LLM model to use
            prompt_version: Version of prompt template
            min_confidence: Minimum confidence threshold
            max_issues: Maximum number of issues to find
            focus_areas: Specific areas to focus analysis on
            claims_context: Previously extracted claims for context
            
        Returns:
            LogicAnalysisResult with analysis findings
        """
        start_time = time.time()
        
        # Get prompt template
        template = prompt_manager.get_template(PromptType.LOGIC_ANALYSIS, prompt_version)
        
        # Prepare prompt parameters
        prompt_kwargs = {
            'text': document_content,
            'max_issues': max_issues,
            'min_confidence': min_confidence
        }
        
        # Add claims context if available
        if claims_context:
            claims_summary = self._create_claims_summary(claims_context)
            prompt_kwargs['claims_context'] = claims_summary
        
        # Add focus areas if specified
        if focus_areas:
            prompt_kwargs['focus_areas'] = ', '.join(focus_areas)
        
        # Create messages for LLM
        messages = prompt_manager.create_messages(template, **prompt_kwargs)
        
        # Call LLM
        try:
            response = await self.llm_client.generate_response(
                messages=messages,
                model=model,
                temperature=0.1
            )
            
            # Parse response
            logical_issues = self._parse_logic_response(response.content, document_content, document_id, model)
            
            # Create analysis result
            analysis_result = self._create_analysis_result(
                logical_issues=logical_issues,
                document_id=document_id,
                model=model,
                prompt_version=prompt_version,
                document_content=document_content
            )
            
            return analysis_result
        except Exception as e:
            self.logger.error(f"Logic analysis failed: {str(e)}")
            # Return empty result with error
            return LogicAnalysisResult(
                document_id=document_id,
                analysis_id=f"logic_analysis_{uuid.uuid4().hex}",
                analyzed_at=datetime.utcnow(),
                overall_logic_score=0.5,  # Neutral score on error
                total_issues_found=0,
                logical_issues=[],
                model_used=model,
                prompt_version=prompt_version,
                analysis_parameters={},
                average_confidence=0.0,
                average_severity=0.0,
                analysis_errors=[f"Logic analysis failed: {str(e)}"]
            )
    
    def _create_combined_analysis_result(self, llm_result: LogicAnalysisResult, 
                                       ml_issues: List[LogicalIssue], 
                                       document_id: str, model: str, 
                                       prompt_version: str, document_content: str) -> LogicAnalysisResult:
        """
        Combine LLM and ML analysis results.
        
        Args:
            llm_result: Results from LLM analysis
            ml_issues: Issues detected by ML analysis
            document_id: Document identifier
            model: Model used for analysis
            prompt_version: Prompt version used
            document_content: Original document content
            
        Returns:
            Combined LogicAnalysisResult
        """
        # Combine all issues
        all_issues = list(llm_result.logical_issues) + ml_issues
        
        # Remove duplicates based on similar text spans and fallacy types
        deduplicated_issues = self._deduplicate_logical_issues(all_issues)
        
        # Create new combined result
        return self._create_analysis_result(
            deduplicated_issues, document_id, model, prompt_version, document_content
        )
    
    def _deduplicate_logical_issues(self, issues: List[LogicalIssue]) -> List[LogicalIssue]:
        """
        Remove duplicate logical issues based on content similarity.
        
        Args:
            issues: List of logical issues to deduplicate
            
        Returns:
            Deduplicated list of issues
        """
        if not issues:
            return []
        
        # Group issues by fallacy type
        grouped_issues = defaultdict(list)
        for issue in issues:
            grouped_issues[issue.fallacy_type].append(issue)
        
        deduplicated = []
        for fallacy_type, type_issues in grouped_issues.items():
            # Sort by confidence score (highest first)
            type_issues.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Keep track of used text spans to avoid duplicates
            used_spans = set()
            
            for issue in type_issues:
                # Create a normalized version of the text excerpt for comparison
                normalized_text = issue.text_excerpt.lower().strip()
                
                # Check if this text span is too similar to an already included one
                is_duplicate = False
                for used_span in used_spans:
                    if self._text_similarity(normalized_text, used_span) > 0.8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    used_spans.add(normalized_text)
                    deduplicated.append(issue)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple similarity based on word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    

    
    def _create_claims_summary(self, claims_context: Dict[str, Any]) -> str:
        """Create a summary of extracted claims for context."""
        try:
            claims = claims_context.get('claims', [])
            if not claims:
                return "No claims were extracted from this document."
            
            summary_parts = [f"Previously extracted {len(claims)} claims:"]
            for i, claim in enumerate(claims[:10], 1):  # Limit to first 10 claims
                claim_text = claim.get('claim_text', 'Unknown claim')
                claim_type = claim.get('claim_type', 'unknown')
                summary_parts.append(f"{i}. [{claim_type}] {claim_text}")
            
            if len(claims) > 10:
                summary_parts.append(f"... and {len(claims) - 10} more claims")
            
            return '\n'.join(summary_parts)
        except Exception as e:
            self.logger.warning(f"Failed to create claims summary: {str(e)}")
            return "Claims context unavailable."
    
    def _parse_logic_response(self, response: str, document_content: str, 
                            document_id: str, model: str) -> List[LogicalIssue]:
        """Parse LLM response and extract logical issues."""
        logical_issues = []
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{') or response.strip().startswith('['):
                parsed_response = json.loads(response)
            else:
                # If not JSON, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group())
                else:
                    # Fallback: treat as text and create a single issue
                    parsed_response = {"issues": [{"description": response}]}
            
            # Handle different response formats
            issues_data = []
            if isinstance(parsed_response, list):
                issues_data = parsed_response
            elif isinstance(parsed_response, dict):
                issues_data = parsed_response.get('issues', [parsed_response])
            
            # Create LogicalIssue objects
            for i, issue_data in enumerate(issues_data):
                if isinstance(issue_data, str):
                    # Simple string format
                    issue = LogicalIssue(
                        issue_id=f"logic_issue_{uuid.uuid4().hex}",
                        issue_type="reasoning_error",
                        title=f"Logic Issue {i+1}",
                        description=issue_data,
                        explanation="Detected by automated analysis",
                        text_excerpt=issue_data[:200] + "..." if len(issue_data) > 200 else issue_data,
                        severity_score=0.5,
                        confidence_score=0.5,
                        impact_score=0.5
                    )
                else:
                    # Structured format
                    issue = self._create_logical_issue(issue_data, document_content, document_id)
                
                logical_issues.append(issue)
        
        except Exception as e:
            self.logger.error(f"Failed to parse logic analysis response: {str(e)}")
            # Create a fallback issue
            fallback_issue = LogicalIssue(
                issue_id=f"logic_issue_{uuid.uuid4().hex}",
                issue_type="parsing_error",
                title="Analysis Parsing Error",
                description=f"Failed to parse logic analysis results: {str(e)}",
                explanation="The automated analysis encountered a parsing error",
                text_excerpt=response[:200] + "..." if len(response) > 200 else response,
                severity_score=0.3,
                confidence_score=0.2,
                impact_score=0.2
            )
            logical_issues.append(fallback_issue)
        
        return logical_issues
    
    def _create_logical_issue(self, issue_data: Dict[str, Any], document_content: str, 
                            document_id: str) -> LogicalIssue:
        """Create a LogicalIssue from parsed data."""
        
        # Extract basic information
        issue_id = issue_data.get('id', f"logic_issue_{uuid.uuid4().hex}")
        title = issue_data.get('title', 'Logical Issue')
        description = issue_data.get('description', 'No description provided')
        explanation = issue_data.get('explanation', 'No explanation provided')
        text_excerpt = issue_data.get('text', issue_data.get('excerpt', description))
        
        # Determine issue type and specific type
        issue_type = issue_data.get('type', 'reasoning_error')
        fallacy_type = None
        reasoning_type = None
        
        # Try to map to specific types
        if issue_type in [ft.value for ft in LogicalFallacyType]:
            fallacy_type = LogicalFallacyType(issue_type)
            issue_type = "fallacy"
        elif issue_type in [rt.value for rt in ReasoningIssueType]:
            reasoning_type = ReasoningIssueType(issue_type)
            issue_type = "reasoning_error"
        
        # Extract scores with defaults
        severity_score = float(issue_data.get('severity', issue_data.get('severity_score', 0.5)))
        confidence_score = float(issue_data.get('confidence', issue_data.get('confidence_score', 0.5)))
        impact_score = float(issue_data.get('impact', issue_data.get('impact_score', 0.5)))
        
        # Ensure scores are in valid range
        severity_score = max(0.0, min(1.0, severity_score))
        confidence_score = max(0.0, min(1.0, confidence_score))
        impact_score = max(0.0, min(1.0, impact_score))
        
        # Find position in document if possible
        start_pos = None
        end_pos = None
        if text_excerpt and text_excerpt in document_content:
            start_pos = document_content.find(text_excerpt)
            end_pos = start_pos + len(text_excerpt) if start_pos != -1 else None
        
        # Extract additional fields
        affected_claims = issue_data.get('affected_claims', [])
        suggestions = issue_data.get('suggestions', issue_data.get('recommendations', []))
        
        return LogicalIssue(
            issue_id=issue_id,
            issue_type=issue_type,
            fallacy_type=fallacy_type,
            reasoning_type=reasoning_type,
            title=title,
            description=description,
            explanation=explanation,
            text_excerpt=text_excerpt,
            start_position=start_pos,
            end_position=end_pos,
            severity_score=severity_score,
            confidence_score=confidence_score,
            impact_score=impact_score,
            affected_claims=affected_claims,
            suggestions=suggestions
        )
    
    def _create_analysis_result(self, logical_issues: List[LogicalIssue], document_id: str,
                              model: str, prompt_version: str, document_content: str) -> LogicAnalysisResult:
        """Create LogicAnalysisResult from logical issues."""
        
        # Calculate statistics
        total_issues = len(logical_issues)
        
        # Calculate averages
        if logical_issues:
            average_confidence = sum(issue.confidence_score for issue in logical_issues) / total_issues
            average_severity = sum(issue.severity_score for issue in logical_issues) / total_issues
            
            # Calculate overall logic score (lower is better for logic quality)
            # This is based on severity and impact of issues found
            overall_impact = sum(issue.severity_score * issue.impact_score for issue in logical_issues) / total_issues
            overall_logic_score = max(0.0, 1.0 - overall_impact)
        else:
            average_confidence = 1.0
            average_severity = 0.0
            overall_logic_score = 1.0  # Perfect score when no issues found
        
        # Count fallacy types
        fallacy_counts = {}
        reasoning_issue_counts = {}
        
        for issue in logical_issues:
            if issue.fallacy_type:
                fallacy_counts[issue.fallacy_type] = fallacy_counts.get(issue.fallacy_type, 0) + 1
            if issue.reasoning_type:
                reasoning_issue_counts[issue.reasoning_type] = reasoning_issue_counts.get(issue.reasoning_type, 0) + 1
        
        # Create severity distribution
        severity_ranges = {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "high": (0.7, 1.0)
        }
        
        severity_distribution = {}
        for level, (min_val, max_val) in severity_ranges.items():
            count = sum(1 for issue in logical_issues 
                       if min_val <= issue.severity_score < max_val)
            if level == "high":  # Include 1.0 in high range
                count += sum(1 for issue in logical_issues if issue.severity_score == 1.0)
            severity_distribution[level] = count
        
        return LogicAnalysisResult(
            document_id=document_id,
            analysis_id=f"logic_analysis_{uuid.uuid4().hex}",
            analyzed_at=datetime.utcnow(),
            overall_logic_score=overall_logic_score,
            total_issues_found=total_issues,
            severity_distribution=severity_distribution,
            logical_issues=logical_issues,
            fallacy_counts=fallacy_counts,
            reasoning_issue_counts=reasoning_issue_counts,
            model_used=model,
            prompt_version=prompt_version,
            analysis_parameters={},
            average_confidence=average_confidence,
            average_severity=average_severity
        ) 