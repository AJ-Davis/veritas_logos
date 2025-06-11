"""
Issue Detection Engine for centralizing issue collection and management.

This module provides the core IssueDetectionEngine that orchestrates issue
collection from all verification passes, applies unified scoring, handles
deduplication, and integrates with ACVF escalation system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict

from src.models.issues import (
    UnifiedIssue, IssueRegistry, IssueType, IssueSeverity, IssueStatus,
    EscalationPath, IssueLocation, IssueMetadata,
    convert_logical_issue, convert_bias_issue, convert_citation_issue
)
from src.models.verification import (
    VerificationChainResult, VerificationResult, VerificationPassType, VerificationContext
)
from src.models.logic_bias import LogicalIssue, BiasIssue, LogicAnalysisResult, BiasAnalysisResult
from src.models.citations import VerifiedCitation, CitationVerificationResult
from src.verification.acvf_controller import ACVFController

logger = logging.getLogger(__name__)


@dataclass
class IssueCollectionConfig:
    """Configuration for issue collection and processing."""
    # Severity thresholds for different actions
    escalation_threshold: float = 0.7
    auto_resolve_threshold: float = 0.2
    
    # Deduplication settings
    similarity_threshold: float = 0.8
    location_overlap_threshold: float = 0.5
    
    # ACVF escalation settings
    enable_acvf_escalation: bool = True
    acvf_threshold: float = 0.6
    max_escalation_attempts: int = 3
    
    # Priority calculation weights
    severity_weight: float = 0.4
    confidence_weight: float = 0.3
    impact_weight: float = 0.3
    
    # Cross-pass correlation settings
    enable_cross_pass_correlation: bool = True
    correlation_window_chars: int = 500


@dataclass
class IssueCollectionStats:
    """Statistics from issue collection process."""
    total_issues_found: int = 0
    issues_by_pass: Dict[VerificationPassType, int] = None
    issues_by_severity: Dict[IssueSeverity, int] = None
    issues_by_type: Dict[IssueType, int] = None
    duplicates_removed: int = 0
    issues_escalated: int = 0
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.issues_by_pass is None:
            self.issues_by_pass = defaultdict(int)
        if self.issues_by_severity is None:
            self.issues_by_severity = defaultdict(int)
        if self.issues_by_type is None:
            self.issues_by_type = defaultdict(int)


class IssueDetectionEngine:
    """
    Central engine for issue detection and management across verification passes.
    
    This engine collects issues from all verification passes, applies unified
    scoring, handles deduplication, manages escalation to ACVF, and provides
    comprehensive issue tracking and analytics.
    """
    
    def __init__(self, config: IssueCollectionConfig, acvf_controller: Optional[ACVFController] = None):
        """
        Initialize the issue detection engine.
        
        Args:
            config: Configuration for issue collection and processing
            acvf_controller: ACVF controller for escalation (optional)
        """
        self.config = config
        self.acvf_controller = acvf_controller
        self.logger = logging.getLogger("issue_detection.engine")
        
        # Tracking for analytics
        self._session_stats = IssueCollectionStats()
        self._escalation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def collect_issues_from_chain_result(self, 
                                             chain_result: VerificationChainResult,
                                             document_id: str,
                                             document_content: str) -> IssueRegistry:
        """
        Collect and process issues from a complete verification chain result.
        
        Args:
            chain_result: Result from verification chain execution
            document_id: Unique identifier for the document
            document_content: Full document content for context
            
        Returns:
            IssueRegistry with all collected and processed issues
        """
        start_time = datetime.now()
        
        try:
            # Initialize registry
            registry = IssueRegistry(document_id=document_id)
            
            # Reset session stats
            self._session_stats = IssueCollectionStats()
            
            # Collect issues from each verification pass
            await self._collect_from_all_passes(chain_result, registry, document_content)
            
            # Apply cross-pass correlation analysis
            if self.config.enable_cross_pass_correlation:
                await self._analyze_cross_pass_correlations(registry)
            
            # Deduplicate similar issues
            duplicates_removed = registry.aggregate_similar_issues(self.config.similarity_threshold)
            self._session_stats.duplicates_removed = duplicates_removed
            
            # Apply unified scoring and prioritization
            await self._apply_unified_scoring(registry)
            
            # Handle escalation for high-priority issues
            if self.config.enable_acvf_escalation and self.acvf_controller:
                await self._handle_escalations(registry, chain_result, document_content)
            
            # Update final statistics
            self._update_final_stats(registry)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._session_stats.processing_time_ms = processing_time
            
            self.logger.info(f"Issue collection completed: {self._session_stats.total_issues_found} issues, "
                           f"{duplicates_removed} duplicates removed, "
                           f"{self._session_stats.issues_escalated} escalated, "
                           f"processed in {processing_time:.2f}ms")
            
            return registry
            
        except Exception as e:
            self.logger.error(f"Error during issue collection: {str(e)}")
            raise
    
    async def _collect_from_all_passes(self, 
                                     chain_result: VerificationChainResult,
                                     registry: IssueRegistry,
                                     document_content: str) -> None:
        """Collect issues from all verification pass results."""
        for pass_result in chain_result.pass_results:
            try:
                await self._collect_from_single_pass(pass_result, registry, document_content)
                self._session_stats.issues_by_pass[pass_result.pass_type] += len(registry.issues)
            except Exception as e:
                self.logger.error(f"Error collecting issues from {pass_result.pass_type}: {str(e)}")
                continue
    
    async def _collect_from_single_pass(self, 
                                      pass_result: VerificationResult,
                                      registry: IssueRegistry,
                                      document_content: str) -> None:
        """Collect issues from a single verification pass result."""
        pass_type = pass_result.pass_type
        
        if not pass_result.result_data:
            return
        
        if pass_type == VerificationPassType.LOGIC_ANALYSIS:
            await self._collect_logic_issues(pass_result, registry)
        elif pass_type == VerificationPassType.BIAS_SCAN:
            await self._collect_bias_issues(pass_result, registry)
        elif pass_type == VerificationPassType.CITATION_CHECK:
            await self._collect_citation_issues(pass_result, registry)
        elif pass_type == VerificationPassType.CLAIM_EXTRACTION:
            await self._collect_claim_issues(pass_result, registry)
        else:
            await self._collect_generic_issues(pass_result, registry)
    
    async def _collect_logic_issues(self, pass_result: VerificationResult, registry: IssueRegistry) -> None:
        """Collect issues from logic analysis pass."""
        try:
            # Handle both direct LogicAnalysisResult and result_data dict
            if isinstance(pass_result.result_data, dict):
                if 'issues' in pass_result.result_data:
                    logical_issues = pass_result.result_data['issues']
                elif 'logical_fallacies' in pass_result.result_data:
                    logical_issues = pass_result.result_data['logical_fallacies']
                else:
                    logical_issues = []
            else:
                # Assume it's a LogicAnalysisResult object
                logical_issues = getattr(pass_result.result_data, 'issues', [])
            
            for issue_data in logical_issues:
                # Convert to LogicalIssue if it's a dict
                if isinstance(issue_data, dict):
                    logical_issue = LogicalIssue(**issue_data)
                else:
                    logical_issue = issue_data
                
                # Convert to unified issue
                unified_issue = convert_logical_issue(logical_issue, pass_result.pass_type)
                
                # Add additional metadata
                unified_issue.metadata.source_verification_id = pass_result.result_id
                unified_issue.metadata.contributing_passes = [pass_result.pass_type]
                
                registry.add_issue(unified_issue)
                
        except Exception as e:
            self.logger.error(f"Error collecting logic issues: {str(e)}")
    
    async def _collect_bias_issues(self, pass_result: VerificationResult, registry: IssueRegistry) -> None:
        """Collect issues from bias scan pass."""
        try:
            # Handle both direct BiasDetectionResult and result_data dict
            if isinstance(pass_result.result_data, dict):
                if 'issues' in pass_result.result_data:
                    bias_issues = pass_result.result_data['issues']
                elif 'detected_biases' in pass_result.result_data:
                    bias_issues = pass_result.result_data['detected_biases']
                else:
                    bias_issues = []
            else:
                # Assume it's a BiasDetectionResult object
                bias_issues = getattr(pass_result.result_data, 'issues', [])
            
            for issue_data in bias_issues:
                # Convert to BiasIssue if it's a dict
                if isinstance(issue_data, dict):
                    bias_issue = BiasIssue(**issue_data)
                else:
                    bias_issue = issue_data
                
                # Convert to unified issue
                unified_issue = convert_bias_issue(bias_issue, pass_result.pass_type)
                
                # Add additional metadata
                unified_issue.metadata.source_verification_id = pass_result.result_id
                unified_issue.metadata.contributing_passes = [pass_result.pass_type]
                
                registry.add_issue(unified_issue)
                
        except Exception as e:
            self.logger.error(f"Error collecting bias issues: {str(e)}")
    
    async def _collect_citation_issues(self, pass_result: VerificationResult, registry: IssueRegistry) -> None:
        """Collect issues from citation check pass."""
        try:
            # Handle both direct CitationVerificationResult and result_data dict
            if isinstance(pass_result.result_data, dict):
                if 'verified_citations' in pass_result.result_data:
                    citations = pass_result.result_data['verified_citations']
                elif 'invalid_citations' in pass_result.result_data:
                    # Handle legacy format
                    invalid_citations = pass_result.result_data['invalid_citations']
                    citations = [{'citation_id': i, 'is_valid': False, 'issues': []}
                               for i in range(len(invalid_citations))]
                else:
                    citations = []
            else:
                # Assume it's a CitationCheckResult object
                citations = getattr(pass_result.result_data, 'verified_citations', [])
            
            for citation_data in citations:
                # Convert to VerifiedCitation if it's a dict
                if isinstance(citation_data, dict):
                    verified_citation = VerifiedCitation(**citation_data)
                else:
                    verified_citation = citation_data
                
                # Convert to unified issues (may return multiple issues per citation)
                unified_issues = convert_citation_issue(verified_citation, pass_result.pass_type)
                
                for unified_issue in unified_issues:
                    # Add additional metadata
                    unified_issue.metadata.source_verification_id = pass_result.result_id
                    unified_issue.metadata.contributing_passes = [pass_result.pass_type]
                    
                    registry.add_issue(unified_issue)
                    
        except Exception as e:
            self.logger.error(f"Error collecting citation issues: {str(e)}")
    
    async def _collect_claim_issues(self, pass_result: VerificationResult, registry: IssueRegistry) -> None:
        """Collect issues from claim extraction pass."""
        try:
            # Look for claims that couldn't be extracted or verified
            if isinstance(pass_result.result_data, dict):
                unverifiable_claims = pass_result.result_data.get('unverifiable_claims', [])
                failed_extractions = pass_result.result_data.get('failed_extractions', [])
                
                # Create issues for unverifiable claims
                for claim_id, claim_data in enumerate(unverifiable_claims):
                    unified_issue = UnifiedIssue(
                        issue_type=IssueType.CLAIM_UNSUPPORTED,
                        title="Unverifiable Claim",
                        description=f"Claim could not be verified: {claim_data.get('text', 'Unknown claim')}",
                        location=IssueLocation(
                            start_position=claim_data.get('start_position'),
                            end_position=claim_data.get('end_position'),
                            section=claim_data.get('section')
                        ),
                        text_excerpt=claim_data.get('text', ''),
                        severity=IssueSeverity.MEDIUM,
                        severity_score=0.5,
                        confidence_score=0.8,
                        impact_score=0.6,
                        metadata=IssueMetadata(
                            detected_by=pass_result.pass_type,
                            source_verification_id=pass_result.result_id,
                            contributing_passes=[pass_result.pass_type]
                        ),
                        evidence=[f"Claim extraction marked as unverifiable: {claim_data.get('reason', 'No reason provided')}"],
                        recommendations=["Review claim for factual accuracy", "Consider adding supporting evidence"]
                    )
                    
                    registry.add_issue(unified_issue)
                
                # Create issues for failed extractions
                for extraction_data in failed_extractions:
                    unified_issue = UnifiedIssue(
                        issue_type=IssueType.VERIFICATION_FAILURE,
                        title="Claim Extraction Failed",
                        description=f"Could not extract claims from section: {extraction_data.get('reason', 'Unknown error')}",
                        location=IssueLocation(
                            section=extraction_data.get('section'),
                            start_position=extraction_data.get('start_position'),
                            end_position=extraction_data.get('end_position')
                        ),
                        text_excerpt=extraction_data.get('text', ''),
                        severity=IssueSeverity.LOW,
                        severity_score=0.3,
                        confidence_score=0.9,
                        impact_score=0.2,
                        metadata=IssueMetadata(
                            detected_by=pass_result.pass_type,
                            source_verification_id=pass_result.result_id,
                            contributing_passes=[pass_result.pass_type]
                        ),
                        evidence=[f"Extraction failed: {extraction_data.get('error', 'Unknown error')}"],
                        recommendations=["Review document structure", "Consider manual claim identification"]
                    )
                    
                    registry.add_issue(unified_issue)
                    
        except Exception as e:
            self.logger.error(f"Error collecting claim issues: {str(e)}")
    
    async def _collect_generic_issues(self, pass_result: VerificationResult, registry: IssueRegistry) -> None:
        """Collect issues from other verification passes using generic approach."""
        try:
            # Look for common issue indicators in result_data
            if isinstance(pass_result.result_data, dict):
                # Check for errors or failures
                if 'errors' in pass_result.result_data:
                    for error in pass_result.result_data['errors']:
                        unified_issue = UnifiedIssue(
                            issue_type=IssueType.VERIFICATION_FAILURE,
                            title=f"{pass_result.pass_type.value} Error",
                            description=f"Error in {pass_result.pass_type.value}: {error}",
                            location=IssueLocation(),
                            text_excerpt="",
                            severity=IssueSeverity.MEDIUM,
                            severity_score=0.5,
                            confidence_score=0.9,
                            impact_score=0.4,
                            metadata=IssueMetadata(
                                detected_by=pass_result.pass_type,
                                source_verification_id=pass_result.result_id,
                                contributing_passes=[pass_result.pass_type]
                            ),
                            evidence=[f"Pass error: {error}"],
                            recommendations=[f"Review {pass_result.pass_type.value} configuration"]
                        )
                        
                        registry.add_issue(unified_issue)
                
                # Check for low confidence results
                if pass_result.confidence_score and pass_result.confidence_score < 0.5:
                    unified_issue = UnifiedIssue(
                        issue_type=IssueType.VERIFICATION_FAILURE,
                        title=f"Low Confidence in {pass_result.pass_type.value}",
                        description=f"Verification pass {pass_result.pass_type.value} completed with low confidence",
                        location=IssueLocation(),
                        text_excerpt="",
                        severity=IssueSeverity.MEDIUM,
                        severity_score=0.6,
                        confidence_score=pass_result.confidence_score,
                        impact_score=0.5,
                        metadata=IssueMetadata(
                            detected_by=pass_result.pass_type,
                            source_verification_id=pass_result.result_id,
                            contributing_passes=[pass_result.pass_type]
                        ),
                        evidence=[f"Low confidence score: {pass_result.confidence_score}"],
                        recommendations=[f"Consider manual review of {pass_result.pass_type.value} results"]
                    )
                    
                    registry.add_issue(unified_issue)
                    
        except Exception as e:
            self.logger.error(f"Error collecting generic issues: {str(e)}")
    
    async def _analyze_cross_pass_correlations(self, registry: IssueRegistry) -> None:
        """Analyze correlations between issues from different passes."""
        try:
            # Group issues by location proximity
            location_groups = self._group_issues_by_location(registry.issues)
            
            for location_key, issues in location_groups.items():
                if len(issues) > 1:
                    # Multiple issues in same location - analyze for correlation
                    await self._correlate_location_issues(issues, registry)
            
            # Analyze thematic correlations (e.g., bias + logical fallacy patterns)
            await self._correlate_thematic_issues(registry)
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-pass correlations: {str(e)}")
    
    def _group_issues_by_location(self, issues: List[UnifiedIssue]) -> Dict[str, List[UnifiedIssue]]:
        """Group issues by location proximity."""
        location_groups = defaultdict(list)
        
        for issue in issues:
            # Create location key based on position ranges
            if issue.location.start_position is not None:
                # Group by character position ranges
                range_start = (issue.location.start_position // self.config.correlation_window_chars) * self.config.correlation_window_chars
                location_key = f"chars_{range_start}-{range_start + self.config.correlation_window_chars}"
            elif issue.location.section:
                location_key = f"section_{issue.location.section}"
            elif issue.location.paragraph is not None:
                location_key = f"paragraph_{issue.location.paragraph}"
            else:
                location_key = "unknown_location"
            
            location_groups[location_key].append(issue)
        
        return location_groups
    
    async def _correlate_location_issues(self, issues: List[UnifiedIssue], registry: IssueRegistry) -> None:
        """Correlate issues that occur in the same location."""
        # Update metadata to indicate correlation
        for issue in issues:
            related_ids = [other.issue_id for other in issues if other.issue_id != issue.issue_id]
            issue.metadata.related_issue_ids.extend(related_ids)
            
            # Boost impact score for clustered issues
            if len(issues) > 2:
                issue.impact_score = min(1.0, issue.impact_score * 1.2)
                issue.metadata.processing_notes.append(
                    f"Impact boosted due to {len(issues)} related issues in same location"
                )
    
    async def _correlate_thematic_issues(self, registry: IssueRegistry) -> None:
        """Correlate issues based on thematic relationships."""
        # Example: Bias issues + logical fallacies often occur together
        bias_issues = registry.get_issues_by_type(IssueType.BIAS_DETECTION)
        logic_issues = registry.get_issues_by_type(IssueType.LOGICAL_FALLACY)
        
        for bias_issue in bias_issues:
            for logic_issue in logic_issues:
                # Check if they're in similar locations or have similar text
                if self._issues_are_thematically_related(bias_issue, logic_issue):
                    bias_issue.metadata.related_issue_ids.append(logic_issue.issue_id)
                    logic_issue.metadata.related_issue_ids.append(bias_issue.issue_id)
                    
                    # Note the correlation
                    correlation_note = f"Thematically related to {logic_issue.issue_type.value} issue {logic_issue.issue_id}"
                    bias_issue.metadata.processing_notes.append(correlation_note)
    
    def _issues_are_thematically_related(self, issue1: UnifiedIssue, issue2: UnifiedIssue) -> bool:
        """Determine if two issues are thematically related."""
        # Simple implementation - can be enhanced with ML approaches
        
        # Check location proximity
        if (issue1.location.start_position is not None and 
            issue2.location.start_position is not None):
            distance = abs(issue1.location.start_position - issue2.location.start_position)
            if distance <= self.config.correlation_window_chars:
                return True
        
        # Check text similarity
        similarity = self._calculate_text_similarity(issue1.text_excerpt, issue2.text_excerpt)
        if similarity > 0.6:
            return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text excerpts."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _apply_unified_scoring(self, registry: IssueRegistry) -> None:
        """Apply unified scoring and prioritization to all issues."""
        for issue in registry.issues:
            # Recalculate priority score with engine-specific logic
            priority = self._calculate_enhanced_priority(issue)
            
            # Store original priority and update with enhanced version
            original_priority = issue.calculate_priority_score()
            issue.metadata.processing_notes.append(
                f"Priority enhanced: {original_priority:.3f} -> {priority:.3f}"
            )
            
            # Update severity based on correlation analysis
            if len(issue.metadata.related_issue_ids) > 2:
                self._boost_severity_for_clustering(issue)
    
    def _calculate_enhanced_priority(self, issue: UnifiedIssue) -> float:
        """Calculate enhanced priority score using engine-specific logic."""
        base_priority = issue.calculate_priority_score()
        
        # Apply engine-specific enhancements
        enhanced_priority = base_priority
        
        # Boost for cross-pass correlations
        if len(issue.metadata.contributing_passes) > 1:
            enhanced_priority *= 1.15
        
        # Boost for multiple related issues
        if len(issue.metadata.related_issue_ids) > 1:
            boost_factor = 1 + (len(issue.metadata.related_issue_ids) * 0.1)
            enhanced_priority *= boost_factor
        
        # Boost for issues in critical document sections
        if issue.location.section and any(keyword in issue.location.section.lower() 
                                        for keyword in ['conclusion', 'summary', 'abstract', 'introduction']):
            enhanced_priority *= 1.1
        
        return min(1.0, enhanced_priority)
    
    def _boost_severity_for_clustering(self, issue: UnifiedIssue) -> None:
        """Boost severity for issues that are part of clusters."""
        cluster_size = len(issue.metadata.related_issue_ids) + 1  # +1 for the issue itself
        
        if cluster_size >= 3:
            # Boost severity level if multiple issues cluster together
            current_score = issue.severity_score
            boosted_score = min(1.0, current_score * (1 + cluster_size * 0.05))
            
            if boosted_score != current_score:
                issue.severity_score = boosted_score
                
                # Update severity enum if needed
                if boosted_score >= 0.8 and issue.severity != IssueSeverity.CRITICAL:
                    issue.severity = IssueSeverity.CRITICAL
                elif boosted_score >= 0.6 and issue.severity == IssueSeverity.MEDIUM:
                    issue.severity = IssueSeverity.HIGH
                
                issue.metadata.processing_notes.append(
                    f"Severity boosted due to clustering: {current_score:.3f} -> {boosted_score:.3f}"
                )
    
    async def _handle_escalations(self, registry: IssueRegistry, 
                                chain_result: VerificationChainResult,
                                document_content: str) -> None:
        """Handle escalation for high-priority issues."""
        if not self.acvf_controller:
            return
        
        escalation_candidates = [
            issue for issue in registry.issues
            if (issue.severity_score >= self.config.escalation_threshold and
                not issue.requires_escalation and
                issue.status == IssueStatus.DETECTED)
        ]
        
        for issue in escalation_candidates:
            try:
                # Check if this issue type should be escalated
                if await self._should_escalate_issue(issue):
                    await self._escalate_issue_to_acvf(issue, chain_result, document_content)
                    self._session_stats.issues_escalated += 1
                    
            except Exception as e:
                self.logger.error(f"Error escalating issue {issue.issue_id}: {str(e)}")
                continue
    
    async def _should_escalate_issue(self, issue: UnifiedIssue) -> bool:
        """Determine if an issue should be escalated to ACVF."""
        # Check escalation history to prevent loops
        escalation_count = len([event for event in issue.metadata.escalation_history
                              if event.get('escalation_type') == 'acvf_debate'])
        
        if escalation_count >= self.config.max_escalation_attempts:
            self.logger.info(f"Issue {issue.issue_id} has reached max escalation attempts")
            return False
        
        # Check issue type suitability for ACVF
        acvf_suitable_types = {
            IssueType.LOGICAL_FALLACY,
            IssueType.BIAS_DETECTION,
            IssueType.FACTUAL_ERROR,
            IssueType.CLAIM_UNSUPPORTED,
            IssueType.CREDIBILITY_CONCERN
        }
        
        if issue.issue_type not in acvf_suitable_types:
            return False
        
        # Check confidence threshold
        if issue.confidence_score < self.config.acvf_threshold:
            return False
        
        return True
    
    async def _escalate_issue_to_acvf(self, issue: UnifiedIssue, 
                                    chain_result: VerificationChainResult,
                                    document_content: str) -> None:
        """Escalate an issue to ACVF for adversarial validation."""
        try:
            # Create verification context for ACVF
            verification_context = VerificationContext(
                document_id=issue.metadata.source_verification_id or "unknown",
                document_content=document_content,
                previous_results=chain_result.pass_results,
                additional_context={
                    "escalated_issue_id": issue.issue_id,
                    "escalated_issue_type": issue.issue_type.value,
                    "escalated_issue_description": issue.description
                }
            )
            
            # Trigger ACVF debate
            subject_content = issue.text_excerpt or issue.description
            acvf_result = await self.acvf_controller.conduct_full_debate(
                verification_context=verification_context,
                subject_type="issue_validation",
                subject_id=issue.issue_id,
                subject_content=subject_content
            )
            
            # Update issue with ACVF results
            issue.add_escalation_event("acvf_debate", {
                "acvf_session_id": acvf_result.session_id,
                "final_verdict": acvf_result.final_verdict.value if acvf_result.final_verdict else None,
                "consensus_confidence": acvf_result.consensus_confidence,
                "debate_rounds": len(acvf_result.debate_rounds)
            })
            
            issue.add_acvf_session(acvf_result.session_id)
            issue.update_status(IssueStatus.ACVF_ESCALATED, 
                              f"Escalated to ACVF with {len(acvf_result.debate_rounds)} rounds")
            
            # Update escalation path and requirements
            issue.escalation_path = EscalationPath.ACVF_DEBATE
            issue.requires_escalation = False
            
            self.logger.info(f"Successfully escalated issue {issue.issue_id} to ACVF")
            
        except Exception as e:
            self.logger.error(f"Failed to escalate issue {issue.issue_id} to ACVF: {str(e)}")
            issue.add_escalation_event("acvf_escalation_failed", {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def _update_final_stats(self, registry: IssueRegistry) -> None:
        """Update final statistics after processing."""
        self._session_stats.total_issues_found = len(registry.issues)
        
        # Count by severity
        for issue in registry.issues:
            self._session_stats.issues_by_severity[issue.severity] += 1
            self._session_stats.issues_by_type[issue.issue_type] += 1
    
    def get_session_stats(self) -> IssueCollectionStats:
        """Get statistics from the current session."""
        return self._session_stats
    
    def get_escalation_history(self, issue_id: str) -> List[Dict[str, Any]]:
        """Get escalation history for a specific issue."""
        return self._escalation_history.get(issue_id, []) 