"""
Bias detection verification pass implementation.
"""

import json
import logging
import os
import time
import uuid
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
    BiasIssue,
    BiasAnalysisResult,
    BiasType,
    BiasSeverity
)
from ....llm.llm_client import LLMClient, LLMConfig, LLMProvider
from ....llm.prompts import PromptType, prompt_manager
from .ml_enhanced_bias import MLEnhancedBiasAnalyzer, MLBiasConfig


logger = logging.getLogger(__name__)


class BiasScanPass(BaseVerificationPass):
    """
    Verification pass for detecting various types of bias in documents.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, enable_ml_enhancement: bool = True):
        """
        Initialize the bias scan pass.
        
        Args:
            llm_client: Optional LLM client. If None, will create default client.
            enable_ml_enhancement: Whether to enable ML-enhanced analysis
        """
        super().__init__(VerificationPassType.BIAS_SCAN)
        self.llm_client = llm_client or self._create_default_llm_client()
        self.enable_ml_enhancement = enable_ml_enhancement
        self.ml_analyzer = None
        
        if self.enable_ml_enhancement:
            try:
                self.ml_analyzer = MLEnhancedBiasAnalyzer()
                logger.info("ML-enhanced bias analysis enabled")
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
        """Bias scan can depend on claim extraction for better analysis."""
        return [VerificationPassType.CLAIM_EXTRACTION]
    
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """
        Execute bias scanning on the document.
        
        Args:
            context: Verification context containing document content
            config: Pass configuration
            
        Returns:
            VerificationResult with bias analysis results
        """
        start_time = time.time()
        pass_id = f"bias_scan_{uuid.uuid4().hex}"
        
        try:
            self.logger.info(f"Starting bias scan for document {context.document_id}")
            
            # Extract parameters from config
            parameters = self.extract_parameters(config)
            model = parameters.get('model', 'gpt-4')
            prompt_version = parameters.get('prompt_version', 'v1')
            min_confidence = parameters.get('min_confidence', 0.3)
            max_issues = parameters.get('max_issues', 50)
            bias_types_focus = parameters.get('bias_types_focus', [])
            demographic_analysis = parameters.get('demographic_analysis', True)
            
            # Get any previously extracted claims for context
            claims_context = None
            claim_extraction_result = context.get_previous_result(VerificationPassType.CLAIM_EXTRACTION)
            if claim_extraction_result and claim_extraction_result.result_data:
                claims_context = claim_extraction_result.result_data.get('extraction_result')
            
            # Perform bias analysis using LLM
            llm_analysis_result = await self._analyze_bias_with_llm(
                document_content=context.document_content,
                document_id=context.document_id,
                model=model,
                prompt_version=prompt_version,
                min_confidence=min_confidence,
                max_issues=max_issues,
                bias_types_focus=bias_types_focus,
                demographic_analysis=demographic_analysis,
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
                    logger.info(f"ML bias analysis found {len(ml_issues)} additional issues")
                except Exception as e:
                    logger.warning(f"ML bias analysis failed: {e}")
            
            # Combine LLM and ML results
            all_issues = list(llm_analysis_result.bias_issues) + ml_issues
            
            # Create combined analysis result
            analysis_result = self._create_combined_analysis_result(
                llm_analysis_result, ml_issues, context.document_id, model, prompt_version, context.document_content
            )
            
            # Calculate overall confidence score
            if analysis_result.bias_issues:
                confidence = analysis_result.average_confidence
            else:
                confidence = 0.8  # High confidence when no bias detected
            
            # Prepare result data
            result_data = {
                "analysis_result": analysis_result.dict(),
                "total_issues": analysis_result.total_issues_found,
                "overall_bias_score": analysis_result.overall_bias_score,
                "bias_type_counts": {k.value: v for k, v in analysis_result.bias_type_counts.items()},
                "severity_distribution": {k.value: v for k, v in analysis_result.severity_distribution.items()},
                "model_used": analysis_result.model_used,
                "prompt_version": analysis_result.prompt_version,
                "processing_statistics": {
                    "average_confidence": analysis_result.average_confidence,
                    "average_impact": analysis_result.average_impact,
                    "political_leaning": analysis_result.political_leaning,
                    "demographic_representation": analysis_result.demographic_representation,
                    "source_diversity": analysis_result.source_diversity
                }
            }
            
            # Add warnings and errors
            if analysis_result.analysis_warnings:
                result_data["warnings"] = analysis_result.analysis_warnings
            if analysis_result.analysis_errors:
                result_data["errors"] = analysis_result.analysis_errors
            
            self.logger.info(f"Found {analysis_result.total_issues_found} bias issues in document {context.document_id}")
            
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.COMPLETED,
                result_data=result_data,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Bias scan failed: {str(e)}")
            return self.create_result(
                pass_id=pass_id,
                status=VerificationStatus.FAILED,
                error_message=f"Bias scan failed: {str(e)}"
            )
    
    async def _analyze_bias_with_llm(self, document_content: str, document_id: str,
                                   model: str, prompt_version: str, min_confidence: float,
                                   max_issues: int, bias_types_focus: List[str],
                                   demographic_analysis: bool,
                                   claims_context: Optional[Dict[str, Any]]) -> BiasAnalysisResult:
        """
        Analyze document for bias using LLM.
        
        Args:
            document_content: Content of the document
            document_id: Document identifier
            model: LLM model to use
            prompt_version: Version of prompt template
            min_confidence: Minimum confidence threshold
            max_issues: Maximum number of issues to find
            bias_types_focus: Specific bias types to focus on
            demographic_analysis: Whether to perform demographic analysis
            claims_context: Previously extracted claims for context
            
        Returns:
            BiasAnalysisResult with analysis findings
        """
        start_time = time.time()
        
        # Get prompt template
        template = prompt_manager.get_template(PromptType.BIAS_SCAN, prompt_version)
        
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
        
        # Add bias type focus if specified
        if bias_types_focus:
            prompt_kwargs['bias_focus'] = ', '.join(bias_types_focus)
        
        # Add demographic analysis flag
        prompt_kwargs['demographic_analysis'] = demographic_analysis
        
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
            bias_issues = self._parse_bias_response(response, document_content, document_id, model)
            
            # Create analysis result
            analysis_result = self._create_analysis_result(
                bias_issues=bias_issues,
                document_id=document_id,
                model=model,
                prompt_version=prompt_version,
                document_content=document_content
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"LLM call failed for bias analysis: {str(e)}")
            # Return empty result with error
            return BiasAnalysisResult(
                document_id=document_id,
                analysis_id=f"bias_analysis_{uuid.uuid4().hex}",
                analyzed_at=datetime.utcnow(),
                overall_bias_score=0.5,  # Neutral score on error
                total_issues_found=0,
                bias_issues=[],
                model_used=model,
                prompt_version=prompt_version,
                analysis_parameters={},
                average_confidence=0.0,
                average_impact=0.0,
                analysis_errors=[f"LLM analysis failed: {str(e)}"]
            )
    
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
    
    def _parse_bias_response(self, response: str, document_content: str, 
                           document_id: str, model: str) -> List[BiasIssue]:
        """Parse LLM response and extract bias issues."""
        bias_issues = []
        
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
                issues_data = parsed_response.get('issues', parsed_response.get('bias_issues', [parsed_response]))
            
            # Create BiasIssue objects
            for i, issue_data in enumerate(issues_data):
                if isinstance(issue_data, str):
                    # Simple string format
                    issue = BiasIssue(
                        issue_id=f"bias_issue_{uuid.uuid4().hex}",
                        bias_type=BiasType.CULTURAL_BIAS,  # Default type
                        title=f"Bias Issue {i+1}",
                        description=issue_data,
                        explanation="Detected by automated analysis",
                        text_excerpt=issue_data[:200] + "..." if len(issue_data) > 200 else issue_data,
                        severity=BiasSeverity.MODERATE,
                        confidence_score=0.5,
                        impact_score=0.5
                    )
                else:
                    # Structured format
                    issue = self._create_bias_issue(issue_data, document_content, document_id)
                
                bias_issues.append(issue)
        
        except Exception as e:
            self.logger.error(f"Failed to parse bias analysis response: {str(e)}")
            # Create a fallback issue
            fallback_issue = BiasIssue(
                issue_id=f"bias_issue_{uuid.uuid4().hex}",
                bias_type=BiasType.CULTURAL_BIAS,
                title="Analysis Parsing Error",
                description=f"Failed to parse bias analysis results: {str(e)}",
                explanation="The automated analysis encountered a parsing error",
                text_excerpt=response[:200] + "..." if len(response) > 200 else response,
                severity=BiasSeverity.LOW,
                confidence_score=0.2,
                impact_score=0.2
            )
            bias_issues.append(fallback_issue)
        
        return bias_issues
    
    def _create_bias_issue(self, issue_data: Dict[str, Any], document_content: str, 
                         document_id: str) -> BiasIssue:
        """Create a BiasIssue from parsed data."""
        
        # Extract basic information
        issue_id = issue_data.get('id', f"bias_issue_{uuid.uuid4().hex}")
        title = issue_data.get('title', 'Bias Issue')
        description = issue_data.get('description', 'No description provided')
        explanation = issue_data.get('explanation', 'No explanation provided')
        text_excerpt = issue_data.get('text', issue_data.get('excerpt', description))
        
        # Determine bias type
        bias_type_str = issue_data.get('type', issue_data.get('bias_type', 'cultural_bias'))
        try:
            bias_type = BiasType(bias_type_str)
        except ValueError:
            # Default to cultural bias if type not recognized
            bias_type = BiasType.CULTURAL_BIAS
        
        # Determine severity
        severity_str = issue_data.get('severity', 'moderate')
        try:
            severity = BiasSeverity(severity_str.lower())
        except ValueError:
            # Default to moderate if severity not recognized
            severity = BiasSeverity.MODERATE
        
        # Extract scores with defaults
        confidence_score = float(issue_data.get('confidence', issue_data.get('confidence_score', 0.5)))
        impact_score = float(issue_data.get('impact', issue_data.get('impact_score', 0.5)))
        
        # Ensure scores are in valid range
        confidence_score = max(0.0, min(1.0, confidence_score))
        impact_score = max(0.0, min(1.0, impact_score))
        
        # Find position in document if possible
        start_pos = None
        end_pos = None
        if text_excerpt and text_excerpt in document_content:
            start_pos = document_content.find(text_excerpt)
            end_pos = start_pos + len(text_excerpt) if start_pos != -1 else None
        
        # Extract additional fields
        evidence = issue_data.get('evidence', [])
        examples = issue_data.get('examples', [])
        affected_claims = issue_data.get('affected_claims', [])
        mitigation_suggestions = issue_data.get('mitigation_suggestions', issue_data.get('suggestions', []))
        alternative_perspectives = issue_data.get('alternative_perspectives', [])
        
        return BiasIssue(
            issue_id=issue_id,
            bias_type=bias_type,
            title=title,
            description=description,
            explanation=explanation,
            text_excerpt=text_excerpt,
            start_position=start_pos,
            end_position=end_pos,
            severity=severity,
            confidence_score=confidence_score,
            impact_score=impact_score,
            evidence=evidence,
            examples=examples,
            affected_claims=affected_claims,
            mitigation_suggestions=mitigation_suggestions,
            alternative_perspectives=alternative_perspectives
        )
    
    def _create_analysis_result(self, bias_issues: List[BiasIssue], document_id: str,
                              model: str, prompt_version: str, document_content: str) -> BiasAnalysisResult:
        """Create BiasAnalysisResult from bias issues."""
        
        # Calculate statistics
        total_issues = len(bias_issues)
        
        # Calculate averages
        if bias_issues:
            average_confidence = sum(issue.confidence_score for issue in bias_issues) / total_issues
            average_impact = sum(issue.impact_score for issue in bias_issues) / total_issues
            
            # Calculate overall bias score (higher means more biased)
            # This is based on severity and impact of issues found
            overall_bias_score = average_impact
        else:
            average_confidence = 1.0
            average_impact = 0.0
            overall_bias_score = 0.0  # No bias when no issues found
        
        # Count bias types
        bias_type_counts = {}
        for issue in bias_issues:
            bias_type_counts[issue.bias_type] = bias_type_counts.get(issue.bias_type, 0) + 1
        
        # Create severity distribution
        severity_distribution = {}
        for severity in BiasSeverity:
            severity_distribution[severity] = sum(1 for issue in bias_issues if issue.severity == severity)
        
        # Analyze political leaning and demographic representation
        political_leaning = self._analyze_political_leaning(bias_issues)
        demographic_representation = self._analyze_demographic_representation(bias_issues)
        source_diversity = self._analyze_source_diversity(bias_issues, document_content)
        
        # Generate overall recommendations
        overall_recommendations = self._generate_overall_recommendations(bias_issues)
        
        return BiasAnalysisResult(
            document_id=document_id,
            analysis_id=f"bias_analysis_{uuid.uuid4().hex}",
            analyzed_at=datetime.utcnow(),
            overall_bias_score=overall_bias_score,
            total_issues_found=total_issues,
            severity_distribution=severity_distribution,
            bias_issues=bias_issues,
            bias_type_counts=bias_type_counts,
            political_leaning=political_leaning,
            demographic_representation=demographic_representation,
            source_diversity=source_diversity,
            model_used=model,
            prompt_version=prompt_version,
            analysis_parameters={},
            average_confidence=average_confidence,
            average_impact=average_impact,
            overall_recommendations=overall_recommendations
        )
    
    def _analyze_political_leaning(self, bias_issues: List[BiasIssue]) -> Optional[str]:
        """Analyze political leaning based on bias issues."""
        political_bias_count = sum(1 for issue in bias_issues 
                                 if issue.bias_type == BiasType.POLITICAL_BIAS)
        
        if political_bias_count > 0:
            # Simple heuristic - could be enhanced with more sophisticated analysis
            return "Potential political bias detected"
        return None
    
    def _analyze_demographic_representation(self, bias_issues: List[BiasIssue]) -> Dict[str, Any]:
        """Analyze demographic representation based on bias issues."""
        demographic_issues = [
            issue for issue in bias_issues 
            if issue.bias_type in [BiasType.DEMOGRAPHIC_BIAS, BiasType.GENDER_BIAS, 
                                 BiasType.RACIAL_BIAS, BiasType.AGE_BIAS]
        ]
        
        return {
            "demographic_bias_count": len(demographic_issues),
            "has_representation_issues": len(demographic_issues) > 0,
            "affected_groups": list(set(
                group for issue in demographic_issues 
                for group in issue.examples if isinstance(group, str)
            ))
        }
    
    def _analyze_source_diversity(self, bias_issues: List[BiasIssue], document_content: str) -> Dict[str, Any]:
        """Analyze source diversity based on bias issues and content."""
        selection_bias_count = sum(1 for issue in bias_issues 
                                 if issue.bias_type == BiasType.SELECTION_BIAS)
        
        return {
            "selection_bias_count": selection_bias_count,
            "diversity_concerns": selection_bias_count > 0,
            "recommendations": [
                "Consider diversifying sources",
                "Include multiple perspectives",
                "Address potential selection bias"
            ] if selection_bias_count > 0 else []
        }
    
    def _generate_overall_recommendations(self, bias_issues: List[BiasIssue]) -> List[str]:
        """Generate overall recommendations based on detected bias issues."""
        if not bias_issues:
            return ["No significant bias detected. Continue monitoring for bias in future content."]
        
        recommendations = []
        
        # High severity issues
        high_severity_count = sum(1 for issue in bias_issues 
                                if issue.severity in [BiasSeverity.HIGH, BiasSeverity.SEVERE])
        if high_severity_count > 0:
            recommendations.append(f"Address {high_severity_count} high-severity bias issues immediately")
        
        # Common bias types
        bias_type_counts = {}
        for issue in bias_issues:
            bias_type_counts[issue.bias_type] = bias_type_counts.get(issue.bias_type, 0) + 1
        
        most_common_bias = max(bias_type_counts, key=bias_type_counts.get)
        recommendations.append(f"Focus on addressing {most_common_bias.value} (most frequent bias type)")
        
        # General recommendations
        recommendations.extend([
            "Review content for balanced representation",
            "Consider multiple perspectives on controversial topics",
            "Implement bias review processes for future content"
        ])
        
        return recommendations
    
    def _create_combined_analysis_result(self, llm_result: BiasAnalysisResult, ml_issues: List[BiasIssue], 
                                       document_id: str, model: str, prompt_version: str, 
                                       document_content: str) -> BiasAnalysisResult:
        """
        Create combined analysis result from LLM and ML findings.
        
        Args:
            llm_result: Result from LLM-based analysis
            ml_issues: Issues found by ML-enhanced analysis
            document_id: Document identifier
            model: LLM model used
            prompt_version: Prompt version used
            document_content: Original document content
            
        Returns:
            Combined BiasAnalysisResult
        """
        # Combine all issues, avoiding duplicates
        all_issues = list(llm_result.bias_issues)
        
        # Add ML issues that don't duplicate LLM findings
        for ml_issue in ml_issues:
            # Simple deduplication based on text similarity
            is_duplicate = False
            for existing_issue in all_issues:
                if (ml_issue.text_excerpt and existing_issue.text_excerpt and
                    self._text_similarity(ml_issue.text_excerpt, existing_issue.text_excerpt) > 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_issues.append(ml_issue)
        
        # Remove duplicates and sort by confidence
        deduplicated_issues = self._deduplicate_bias_issues(all_issues)
        
        # Recalculate statistics with combined issues
        total_issues = len(deduplicated_issues)
        
        if deduplicated_issues:
            average_confidence = sum(issue.confidence_score for issue in deduplicated_issues) / total_issues
            average_impact = sum(issue.impact_score for issue in deduplicated_issues) / total_issues
            overall_bias_score = average_impact
        else:
            average_confidence = 1.0
            average_impact = 0.0
            overall_bias_score = 0.0
        
        # Recalculate bias type counts
        bias_type_counts = {}
        for issue in deduplicated_issues:
            bias_type_counts[issue.bias_type] = bias_type_counts.get(issue.bias_type, 0) + 1
        
        # Recalculate severity distribution
        severity_distribution = {}
        for severity in BiasSeverity:
            severity_distribution[severity] = sum(1 for issue in deduplicated_issues if issue.severity == severity)
        
        # Preserve LLM analysis metadata but update with combined results
        return BiasAnalysisResult(
            document_id=document_id,
            analysis_id=f"combined_bias_analysis_{uuid.uuid4().hex}",
            analyzed_at=datetime.utcnow(),
            overall_bias_score=overall_bias_score,
            total_issues_found=total_issues,
            severity_distribution=severity_distribution,
            bias_issues=deduplicated_issues,
            bias_type_counts=bias_type_counts,
            political_leaning=self._analyze_political_leaning(deduplicated_issues),
            demographic_representation=self._analyze_demographic_representation(deduplicated_issues),
            source_diversity=self._analyze_source_diversity(deduplicated_issues, document_content),
            model_used=f"{model} + ML",
            prompt_version=prompt_version,
            analysis_parameters=llm_result.analysis_parameters,
            average_confidence=average_confidence,
            average_impact=average_impact,
            overall_recommendations=self._generate_overall_recommendations(deduplicated_issues),
            analysis_warnings=llm_result.analysis_warnings,
            analysis_errors=llm_result.analysis_errors
        )
    
    def _deduplicate_bias_issues(self, issues: List[BiasIssue]) -> List[BiasIssue]:
        """
        Remove duplicate bias issues based on text similarity and bias type.
        
        Args:
            issues: List of bias issues to deduplicate
            
        Returns:
            Deduplicated list of bias issues
        """
        if not issues:
            return []
        
        # Sort by confidence score (descending) to keep highest confidence issues
        sorted_issues = sorted(issues, key=lambda x: x.confidence_score, reverse=True)
        
        unique_issues = []
        for issue in sorted_issues:
            is_duplicate = False
            
            for existing_issue in unique_issues:
                # Check if same bias type and similar text
                if (issue.bias_type == existing_issue.bias_type and 
                    issue.text_excerpt and existing_issue.text_excerpt):
                    
                    similarity = self._text_similarity(issue.text_excerpt, existing_issue.text_excerpt)
                    if similarity > 0.75:  # 75% similarity threshold
                        is_duplicate = True
                        break
                        
                # Also check for overlapping positions
                elif (issue.start_position is not None and existing_issue.start_position is not None and
                      issue.end_position is not None and existing_issue.end_position is not None):
                    
                    # Check for significant overlap (more than 50% of either span)
                    overlap_start = max(issue.start_position, existing_issue.start_position)
                    overlap_end = min(issue.end_position, existing_issue.end_position)
                    
                    if overlap_end > overlap_start:
                        overlap_length = overlap_end - overlap_start
                        issue_length = issue.end_position - issue.start_position
                        existing_length = existing_issue.end_position - existing_issue.start_position
                        
                        overlap_ratio = overlap_length / min(issue_length, existing_length)
                        if overlap_ratio > 0.5:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_issues.append(issue)
        
        return unique_issues
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0 