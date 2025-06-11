"""
Test suite for IssueDetectionEngine.

Tests the core functionality of issue collection, aggregation, scoring,
and escalation across multiple verification passes.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.verification.pipeline.issue_detection_engine import (
    IssueDetectionEngine, IssueCollectionConfig, IssueCollectionStats
)
from src.models.issues import (
    UnifiedIssue, IssueRegistry, IssueType, IssueSeverity, IssueStatus,
    EscalationPath, IssueLocation, IssueMetadata
)
from src.models.verification import (
    VerificationChainResult, VerificationResult, VerificationPassType, 
    VerificationStatus, VerificationContext
)
from src.models.logic_bias import LogicalIssue, BiasIssue, LogicalFallacyType, BiasType
from src.models.citations import VerifiedCitation, CitationIssue
from src.verification.acvf_controller import ACVFController


@pytest.fixture
def issue_config():
    """Default configuration for issue detection engine."""
    return IssueCollectionConfig(
        escalation_threshold=0.7,
        auto_resolve_threshold=0.2,
        similarity_threshold=0.8,
        acvf_threshold=0.6,
        enable_cross_pass_correlation=True,
        correlation_window_chars=500
    )


@pytest.fixture
def mock_acvf_controller():
    """Mock ACVF controller for testing escalation."""
    controller = Mock(spec=ACVFController)
    controller.conduct_full_debate = AsyncMock()
    return controller


@pytest.fixture
def sample_verification_chain_result():
    """Sample verification chain result with multiple passes."""
    logic_result = VerificationResult(
        pass_type=VerificationPassType.LOGIC_ANALYSIS,
        status=VerificationStatus.COMPLETED,
        confidence_score=0.8,
        result_data={
            'issues': [
                {
                    'fallacy_type': LogicalFallacyType.AD_HOMINEM,
                    'description': 'Attack on person rather than argument',
                    'text_excerpt': 'John is wrong because he is stupid',
                    'start_position': 100,
                    'end_position': 135,
                    'confidence_score': 0.9,
                    'severity_score': 0.7
                }
            ]
        }
    )
    
    bias_result = VerificationResult(
        pass_type=VerificationPassType.BIAS_SCAN,
        status=VerificationStatus.COMPLETED,
        confidence_score=0.75,
        result_data={
            'issues': [
                {
                    'bias_type': BiasType.GENDER,
                    'description': 'Gender-biased language detected',
                    'text_excerpt': 'Women are naturally worse at math',
                    'start_position': 200,
                    'end_position': 235,
                    'confidence_score': 0.85,
                    'severity_score': 0.8
                }
            ]
        }
    )
    
    citation_result = VerificationResult(
        pass_type=VerificationPassType.CITATION_CHECK,
        status=VerificationStatus.COMPLETED,
        confidence_score=0.6,
        result_data={
            'verified_citations': [
                {
                    'citation_id': 'cite_1',
                    'original_citation': 'Smith et al. 2020',
                    'is_valid': False,
                    'issues': [
                        {
                            'issue_type': 'missing_source',
                            'description': 'Citation source could not be verified',
                            'severity': 'high'
                        }
                    ]
                }
            ]
        }
    )
    
    return VerificationChainResult(
        chain_id="test_chain",
        document_id="test_doc",
        status=VerificationStatus.COMPLETED,
        pass_results=[logic_result, bias_result, citation_result],
        overall_confidence=0.7
    )


@pytest.fixture
def issue_detection_engine(issue_config, mock_acvf_controller):
    """Issue detection engine with mocked dependencies."""
    return IssueDetectionEngine(config=issue_config, acvf_controller=mock_acvf_controller)


class TestIssueDetectionEngine:
    """Test cases for IssueDetectionEngine."""
    
    @pytest.mark.asyncio
    async def test_collect_issues_from_chain_result(self, issue_detection_engine, sample_verification_chain_result):
        """Test collecting issues from a complete verification chain result."""
        document_content = "Sample document content with various issues to detect."
        
        registry = await issue_detection_engine.collect_issues_from_chain_result(
            chain_result=sample_verification_chain_result,
            document_id="test_doc",
            document_content=document_content
        )
        
        # Verify registry was created
        assert registry is not None
        assert registry.document_id == "test_doc"
        assert len(registry.issues) > 0
        
        # Verify issues from different passes were collected
        pass_types = {issue.metadata.detected_by for issue in registry.issues}
        assert VerificationPassType.LOGIC_ANALYSIS in pass_types
        assert VerificationPassType.BIAS_SCAN in pass_types
        assert VerificationPassType.CITATION_CHECK in pass_types
        
        # Verify statistics
        stats = issue_detection_engine.get_session_stats()
        assert stats.total_issues_found > 0
        assert stats.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_collect_logic_issues(self, issue_detection_engine):
        """Test collection of logical fallacy issues."""
        pass_result = VerificationResult(
            pass_type=VerificationPassType.LOGIC_ANALYSIS,
            status=VerificationStatus.COMPLETED,
            confidence_score=0.8,
            result_data={
                'issues': [
                    {
                        'fallacy_type': LogicalFallacyType.STRAW_MAN,
                        'description': 'Misrepresentation of opponent argument',
                        'text_excerpt': 'Test excerpt',
                        'start_position': 50,
                        'end_position': 80,
                        'confidence_score': 0.9,
                        'severity_score': 0.75
                    }
                ]
            }
        )
        
        registry = IssueRegistry(document_id="test")
        await issue_detection_engine._collect_logic_issues(pass_result, registry)
        
        assert len(registry.issues) == 1
        issue = registry.issues[0]
        assert issue.issue_type == IssueType.LOGICAL_FALLACY
        assert issue.description == 'Misrepresentation of opponent argument'
        assert issue.metadata.detected_by == VerificationPassType.LOGIC_ANALYSIS
    
    @pytest.mark.asyncio
    async def test_collect_bias_issues(self, issue_detection_engine):
        """Test collection of bias detection issues."""
        pass_result = VerificationResult(
            pass_type=VerificationPassType.BIAS_SCAN,
            status=VerificationStatus.COMPLETED,
            confidence_score=0.7,
            result_data={
                'issues': [
                    {
                        'bias_type': BiasType.RACIAL,
                        'description': 'Racial bias detected',
                        'text_excerpt': 'Biased text excerpt',
                        'start_position': 100,
                        'end_position': 130,
                        'confidence_score': 0.85,
                        'severity_score': 0.9
                    }
                ]
            }
        )
        
        registry = IssueRegistry(document_id="test")
        await issue_detection_engine._collect_bias_issues(pass_result, registry)
        
        assert len(registry.issues) == 1
        issue = registry.issues[0]
        assert issue.issue_type == IssueType.BIAS_DETECTION
        assert issue.description == 'Racial bias detected'
        assert issue.metadata.detected_by == VerificationPassType.BIAS_SCAN
    
    @pytest.mark.asyncio
    async def test_collect_citation_issues(self, issue_detection_engine):
        """Test collection of citation verification issues."""
        pass_result = VerificationResult(
            pass_type=VerificationPassType.CITATION_CHECK,
            status=VerificationStatus.COMPLETED,
            confidence_score=0.6,
            result_data={
                'verified_citations': [
                    {
                        'citation_id': 'cite_1',
                        'original_citation': 'Invalid citation',
                        'is_valid': False,
                        'issues': [
                            {
                                'issue_type': 'broken_link',
                                'description': 'Citation link is broken',
                                'severity': 'medium'
                            }
                        ]
                    }
                ]
            }
        )
        
        registry = IssueRegistry(document_id="test")
        await issue_detection_engine._collect_citation_issues(pass_result, registry)
        
        assert len(registry.issues) >= 1
        # At least one citation issue should be created
        citation_issues = [issue for issue in registry.issues if issue.issue_type == IssueType.CITATION_PROBLEM]
        assert len(citation_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_cross_pass_correlation(self, issue_detection_engine):
        """Test cross-pass correlation analysis."""
        # Create registry with issues from different passes in similar locations
        registry = IssueRegistry(document_id="test")
        
        logic_issue = UnifiedIssue(
            issue_type=IssueType.LOGICAL_FALLACY,
            title="Logic Issue",
            description="Logical fallacy detected",
            location=IssueLocation(start_position=100, end_position=130),
            text_excerpt="problematic text",
            severity=IssueSeverity.HIGH,
            severity_score=0.8,
            confidence_score=0.9,
            impact_score=0.7,
            metadata=IssueMetadata(detected_by=VerificationPassType.LOGIC_ANALYSIS)
        )
        
        bias_issue = UnifiedIssue(
            issue_type=IssueType.BIAS_DETECTION,
            title="Bias Issue",
            description="Bias detected",
            location=IssueLocation(start_position=110, end_position=140),
            text_excerpt="problematic text",
            severity=IssueSeverity.HIGH,
            severity_score=0.8,
            confidence_score=0.85,
            impact_score=0.75,
            metadata=IssueMetadata(detected_by=VerificationPassType.BIAS_SCAN)
        )
        
        registry.add_issue(logic_issue)
        registry.add_issue(bias_issue)
        
        await issue_detection_engine._analyze_cross_pass_correlations(registry)
        
        # Verify correlation was detected
        assert len(logic_issue.metadata.related_issue_ids) > 0
        assert bias_issue.issue_id in logic_issue.metadata.related_issue_ids
        assert logic_issue.issue_id in bias_issue.metadata.related_issue_ids
    
    @pytest.mark.asyncio
    async def test_unified_scoring_enhancement(self, issue_detection_engine):
        """Test unified scoring and priority enhancement."""
        registry = IssueRegistry(document_id="test")
        
        issue = UnifiedIssue(
            issue_type=IssueType.LOGICAL_FALLACY,
            title="Test Issue",
            description="Test description",
            location=IssueLocation(start_position=100, end_position=130),
            text_excerpt="test text",
            severity=IssueSeverity.MEDIUM,
            severity_score=0.6,
            confidence_score=0.8,
            impact_score=0.7,
            metadata=IssueMetadata(
                detected_by=VerificationPassType.LOGIC_ANALYSIS,
                contributing_passes=[VerificationPassType.LOGIC_ANALYSIS, VerificationPassType.BIAS_SCAN]
            )
        )
        
        registry.add_issue(issue)
        
        original_priority = issue.calculate_priority_score()
        await issue_detection_engine._apply_unified_scoring(registry)
        
        # Verify scoring enhancement was applied
        assert len(issue.metadata.processing_notes) > 0
        
        # Check if priority was enhanced for cross-pass detection
        enhanced_priority = issue_detection_engine._calculate_enhanced_priority(issue)
        assert enhanced_priority >= original_priority
    
    @pytest.mark.asyncio
    async def test_escalation_to_acvf(self, issue_detection_engine, mock_acvf_controller):
        """Test escalation of high-priority issues to ACVF."""
        # Mock ACVF result
        from src.models.acvf import ACVFResult, JudgeVerdict
        mock_acvf_result = ACVFResult(
            session_id="test_session",
            final_verdict=JudgeVerdict.CHALLENGER_WINS,
            consensus_confidence=0.85,
            debate_rounds=[]
        )
        mock_acvf_controller.conduct_full_debate.return_value = mock_acvf_result
        
        registry = IssueRegistry(document_id="test")
        
        # Create high-priority issue that should trigger escalation
        high_priority_issue = UnifiedIssue(
            issue_type=IssueType.FACTUAL_ERROR,
            title="High Priority Issue",
            description="Critical factual error",
            location=IssueLocation(start_position=100, end_position=130),
            text_excerpt="factual error text",
            severity=IssueSeverity.CRITICAL,
            severity_score=0.9,
            confidence_score=0.8,
            impact_score=0.85,
            metadata=IssueMetadata(detected_by=VerificationPassType.CITATION_CHECK)
        )
        
        registry.add_issue(high_priority_issue)
        
        # Mock chain result
        chain_result = VerificationChainResult(
            chain_id="test",
            document_id="test",
            status=VerificationStatus.COMPLETED,
            pass_results=[]
        )
        
        await issue_detection_engine._handle_escalations(registry, chain_result, "test content")
        
        # Verify escalation was triggered
        assert mock_acvf_controller.conduct_full_debate.called
        assert high_priority_issue.status == IssueStatus.ACVF_ESCALATED
        assert len(high_priority_issue.metadata.escalation_history) > 0
        assert len(high_priority_issue.metadata.acvf_session_ids) > 0
    
    @pytest.mark.asyncio
    async def test_issue_clustering_severity_boost(self, issue_detection_engine):
        """Test severity boost for clustered issues."""
        registry = IssueRegistry(document_id="test")
        
        # Create multiple related issues
        base_issue = UnifiedIssue(
            issue_type=IssueType.LOGICAL_FALLACY,
            title="Base Issue",
            description="Base issue",
            location=IssueLocation(start_position=100, end_position=130),
            text_excerpt="issue text",
            severity=IssueSeverity.MEDIUM,
            severity_score=0.5,
            confidence_score=0.8,
            impact_score=0.6,
            metadata=IssueMetadata(detected_by=VerificationPassType.LOGIC_ANALYSIS)
        )
        
        # Add related issues
        for i in range(3):
            related_issue_id = f"related_{i}"
            base_issue.metadata.related_issue_ids.append(related_issue_id)
        
        registry.add_issue(base_issue)
        
        original_severity_score = base_issue.severity_score
        issue_detection_engine._boost_severity_for_clustering(base_issue)
        
        # Verify severity was boosted due to clustering
        assert base_issue.severity_score > original_severity_score
        
        # Check if severity level was updated
        if base_issue.severity_score >= 0.6:
            assert base_issue.severity in [IssueSeverity.HIGH, IssueSeverity.CRITICAL]
    
    def test_issue_similarity_calculation(self, issue_detection_engine):
        """Test text similarity calculation for issue correlation."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox leaps over the lazy dog"
        text3 = "Completely different text with no overlap"
        
        similarity_high = issue_detection_engine._calculate_text_similarity(text1, text2)
        similarity_low = issue_detection_engine._calculate_text_similarity(text1, text3)
        
        assert similarity_high > similarity_low
        assert similarity_high > 0.8  # High similarity
        assert similarity_low < 0.2   # Low similarity
    
    def test_session_stats_tracking(self, issue_detection_engine):
        """Test session statistics tracking."""
        stats = issue_detection_engine.get_session_stats()
        
        assert isinstance(stats, IssueCollectionStats)
        assert stats.total_issues_found == 0  # Initial state
        assert isinstance(stats.issues_by_pass, dict)
        assert isinstance(stats.issues_by_severity, dict)
        assert isinstance(stats.issues_by_type, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_collection(self, issue_detection_engine):
        """Test error handling during issue collection."""
        # Create malformed pass result
        malformed_result = VerificationResult(
            pass_type=VerificationPassType.LOGIC_ANALYSIS,
            status=VerificationStatus.COMPLETED,
            confidence_score=0.8,
            result_data={"invalid_data": "malformed"}
        )
        
        registry = IssueRegistry(document_id="test")
        
        # Should not raise exception, should handle gracefully
        await issue_detection_engine._collect_logic_issues(malformed_result, registry)
        
        # Registry should remain in valid state
        assert len(registry.issues) == 0  # No issues collected due to malformed data
    
    def test_escalation_history_tracking(self, issue_detection_engine):
        """Test escalation history tracking functionality."""
        issue_id = "test_issue_123"
        
        # Initially empty
        history = issue_detection_engine.get_escalation_history(issue_id)
        assert len(history) == 0
        
        # Add history (this would normally be done during escalation)
        issue_detection_engine._escalation_history[issue_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "escalation_type": "acvf_debate",
            "result": "successful"
        })
        
        # Verify history is tracked
        updated_history = issue_detection_engine.get_escalation_history(issue_id)
        assert len(updated_history) == 1
        assert updated_history[0]["escalation_type"] == "acvf_debate"


@pytest.mark.integration
class TestIssueDetectionEngineIntegration:
    """Integration tests for IssueDetectionEngine with real verification results."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, issue_config):
        """Test complete integration with realistic verification data."""
        # This would test with actual verification pass outputs
        # For now, we'll create realistic mock data
        
        engine = IssueDetectionEngine(config=issue_config)
        
        # Create realistic verification chain result
        chain_result = VerificationChainResult(
            chain_id="integration_test",
            document_id="integration_doc",
            status=VerificationStatus.COMPLETED,
            pass_results=[
                VerificationResult(
                    pass_type=VerificationPassType.LOGIC_ANALYSIS,
                    status=VerificationStatus.COMPLETED,
                    confidence_score=0.75,
                    result_data={
                        'issues': [
                            {
                                'fallacy_type': LogicalFallacyType.FALSE_DICHOTOMY,
                                'description': 'False dichotomy presented',
                                'text_excerpt': 'Either you are with us or against us',
                                'start_position': 250,
                                'end_position': 285,
                                'confidence_score': 0.9,
                                'severity_score': 0.8
                            }
                        ]
                    }
                ),
                VerificationResult(
                    pass_type=VerificationPassType.BIAS_SCAN,
                    status=VerificationStatus.COMPLETED,
                    confidence_score=0.8,
                    result_data={
                        'issues': [
                            {
                                'bias_type': BiasType.POLITICAL,
                                'description': 'Political bias detected',
                                'text_excerpt': 'All conservatives are wrong',
                                'start_position': 300,
                                'end_position': 325,
                                'confidence_score': 0.85,
                                'severity_score': 0.75
                            }
                        ]
                    }
                )
            ]
        )
        
        document_content = "Full document content for integration testing..."
        
        registry = await engine.collect_issues_from_chain_result(
            chain_result=chain_result,
            document_id="integration_doc",
            document_content=document_content
        )
        
        # Verify comprehensive integration
        assert registry is not None
        assert len(registry.issues) >= 2  # At least logic and bias issues
        
        # Verify stats were calculated
        stats = engine.get_session_stats()
        assert stats.total_issues_found >= 2
        assert stats.processing_time_ms > 0
        
        # Verify issues have proper metadata
        for issue in registry.issues:
            assert issue.issue_id is not None
            assert issue.metadata.detected_by is not None
            assert issue.severity_score > 0
            assert issue.confidence_score > 0 