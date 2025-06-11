"""
Adversarial Recall Metric implementation for measuring verification system effectiveness.

This module implements the Adversarial Recall Metric (ARM) to measure how well
the verification system performs against adversarial examples and challenging
content that is designed to fool verification systems.
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field

from ..models.verification import VerificationResult, VerificationStatus
from ..models.issues import UnifiedIssue, IssueSeverity
from ..models.acvf import ACVFResult
from .clickhouse_client import get_clickhouse_client

logger = logging.getLogger(__name__)


class AdversarialCategory(str, Enum):
    """Categories of adversarial examples for testing."""
    DEEPFAKE_DETECTION = "deepfake_detection"
    MISINFORMATION_INJECTION = "misinformation_injection"
    CONTEXT_MANIPULATION = "context_manipulation"
    SOURCE_SPOOFING = "source_spoofing"
    STATISTICAL_MANIPULATION = "statistical_manipulation"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    TEMPORAL_MISDIRECTION = "temporal_misdirection"
    CAUSAL_CONFUSION = "causal_confusion"
    LOGICAL_FALLACIES = "logical_fallacies"


class AdversarialDifficulty(str, Enum):
    """Difficulty levels for adversarial examples."""
    TRIVIAL = "trivial"      # Easy to detect, basic errors
    EASY = "easy"            # Detectable with standard verification
    MEDIUM = "medium"        # Requires careful analysis
    HARD = "hard"            # Subtle manipulation, requires expertise
    EXPERT = "expert"        # Near-perfect adversarial examples


@dataclass
class AdversarialExample:
    """An adversarial example for testing verification systems."""
    example_id: str
    category: AdversarialCategory
    difficulty: AdversarialDifficulty
    original_content: str
    adversarial_content: str
    expected_issues: List[str]  # List of issue types that should be detected
    ground_truth_label: bool    # True if content is actually true/valid
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AdversarialTestResult:
    """Result of testing verification system against adversarial example."""
    test_id: str
    example_id: str
    verification_result: VerificationResult
    detected_correctly: bool
    false_positive: bool
    false_negative: bool
    confidence_score: float
    processing_time_ms: float
    issues_detected: List[str]
    issues_missed: List[str]
    acvf_triggered: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    tested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdversarialRecallMetric:
    """
    Implementation of the Adversarial Recall Metric for measuring verification effectiveness.
    
    The ARM measures how well the verification system performs against adversarial
    examples designed to fool verification systems. This includes detection rates,
    false positive/negative rates, and confidence calibration.
    """
    
    def __init__(self):
        """Initialize the Adversarial Recall Metric calculator."""
        self.logger = logging.getLogger("analytics.adversarial_metrics")
        self.clickhouse_client = get_clickhouse_client()
        
        # Standard adversarial example library
        self.adversarial_examples: Dict[str, AdversarialExample] = {}
        self.test_results: Dict[str, AdversarialTestResult] = {}
        
        # Performance tracking
        self.category_performance: Dict[AdversarialCategory, Dict[str, float]] = {}
        self.difficulty_performance: Dict[AdversarialDifficulty, Dict[str, float]] = {}
        
    async def initialize_adversarial_library(self) -> None:
        """Initialize the library of adversarial examples for testing."""
        
        # Load pre-defined adversarial examples
        examples = [
            # Deepfake detection examples
            AdversarialExample(
                example_id="deepfake_001",
                category=AdversarialCategory.DEEPFAKE_DETECTION,
                difficulty=AdversarialDifficulty.MEDIUM,
                original_content="Original video of political statement",
                adversarial_content="AI-generated deepfake with subtle facial inconsistencies",
                expected_issues=["visual_inconsistency", "deepfake_detection"],
                ground_truth_label=False,
                metadata={"video_quality": "high", "manipulation_type": "face_swap"}
            ),
            
            # Misinformation injection examples
            AdversarialExample(
                example_id="misinfo_001",
                category=AdversarialCategory.MISINFORMATION_INJECTION,
                difficulty=AdversarialDifficulty.HARD,
                original_content="Accurate scientific study results",
                adversarial_content="Study with subtly altered statistics and cherry-picked data",
                expected_issues=["statistical_manipulation", "selective_reporting"],
                ground_truth_label=False,
                metadata={"manipulation_subtlety": "high", "domain": "medical"}
            ),
            
            # Context manipulation examples
            AdversarialExample(
                example_id="context_001",
                category=AdversarialCategory.CONTEXT_MANIPULATION,
                difficulty=AdversarialDifficulty.MEDIUM,
                original_content="Photo from natural disaster response",
                adversarial_content="Same photo presented as from different event/location",
                expected_issues=["context_mismatch", "temporal_inconsistency"],
                ground_truth_label=False,
                metadata={"image_authentic": True, "context_authentic": False}
            ),
            
            # Source spoofing examples
            AdversarialExample(
                example_id="source_001",
                category=AdversarialCategory.SOURCE_SPOOFING,
                difficulty=AdversarialDifficulty.HARD,
                original_content="News article from reputable source",
                adversarial_content="Article with forged byline and fake credentials",
                expected_issues=["source_verification", "credential_mismatch"],
                ground_truth_label=False,
                metadata={"spoofing_sophistication": "high"}
            ),
            
            # Statistical manipulation examples
            AdversarialExample(
                example_id="stats_001",
                category=AdversarialCategory.STATISTICAL_MANIPULATION,
                difficulty=AdversarialDifficulty.EXPERT,
                original_content="Research paper with proper methodology",
                adversarial_content="Paper with p-hacking and misleading visualizations",
                expected_issues=["statistical_bias", "methodology_flaws"],
                ground_truth_label=False,
                metadata={"manipulation_techniques": ["p_hacking", "cherry_picking"]}
            )
        ]
        
        for example in examples:
            self.adversarial_examples[example.example_id] = example
            
        self.logger.info(f"Initialized adversarial library with {len(examples)} examples")
    
    async def test_verification_system(
        self,
        example_id: str,
        verification_function: callable
    ) -> AdversarialTestResult:
        """
        Test the verification system against a specific adversarial example.
        
        Args:
            example_id: ID of the adversarial example to test
            verification_function: Function that performs verification
            
        Returns:
            Test result with detailed metrics
        """
        if example_id not in self.adversarial_examples:
            raise ValueError(f"Adversarial example {example_id} not found")
        
        example = self.adversarial_examples[example_id]
        start_time = datetime.now(timezone.utc)
        
        try:
            # Run verification on adversarial content
            verification_result = await verification_function(example.adversarial_content)
            
            # Calculate processing time
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Analyze results
            test_result = await self._analyze_test_result(
                example=example,
                verification_result=verification_result,
                processing_time_ms=processing_time_ms
            )
            
            # Store result
            self.test_results[test_result.test_id] = test_result
            
            # Log to ClickHouse for analytics
            await self._log_test_result(test_result)
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Error testing adversarial example {example_id}: {e}")
            raise
    
    async def _analyze_test_result(
        self,
        example: AdversarialExample,
        verification_result: VerificationResult,
        processing_time_ms: float
    ) -> AdversarialTestResult:
        """Analyze the verification result against the adversarial example."""
        
        # Extract detected issues
        detected_issues = []
        if hasattr(verification_result, 'issue_registry'):
            detected_issues = [issue.issue_type.value for issue in verification_result.issue_registry.issues]
        
        # Calculate detection metrics
        expected_issues_set = set(example.expected_issues)
        detected_issues_set = set(detected_issues)
        
        # Check if correctly detected as problematic
        should_flag = not example.ground_truth_label  # If ground truth is False, should be flagged
        was_flagged = len(detected_issues) > 0 or (
            hasattr(verification_result, 'overall_confidence') and 
            verification_result.overall_confidence < 0.5
        )
        
        detected_correctly = should_flag == was_flagged
        false_positive = was_flagged and example.ground_truth_label
        false_negative = not was_flagged and not example.ground_truth_label
        
        # Issues analysis
        issues_detected_correctly = list(expected_issues_set.intersection(detected_issues_set))
        issues_missed = list(expected_issues_set - detected_issues_set)
        
        # Check if ACVF was triggered
        acvf_triggered = hasattr(verification_result, 'acvf_results') and verification_result.acvf_results
        
        # Get confidence score
        confidence_score = getattr(verification_result, 'overall_confidence', 0.0)
        
        return AdversarialTestResult(
            test_id=str(uuid.uuid4()),
            example_id=example.example_id,
            verification_result=verification_result,
            detected_correctly=detected_correctly,
            false_positive=false_positive,
            false_negative=false_negative,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            issues_detected=issues_detected_correctly,
            issues_missed=issues_missed,
            acvf_triggered=acvf_triggered,
            metadata={
                "category": example.category.value,
                "difficulty": example.difficulty.value,
                "expected_issues_count": len(example.expected_issues),
                "detected_issues_count": len(detected_issues),
                "ground_truth": example.ground_truth_label
            }
        )
    
    async def _log_test_result(self, test_result: AdversarialTestResult) -> None:
        """Log test result to ClickHouse for analytics."""
        try:
            event_data = {
                "event_id": str(uuid.uuid4()),
                "timestamp": test_result.tested_at,
                "event_type": "adversarial_test",
                "document_id": test_result.example_id,
                "user_id": "system",
                "verification_type": "adversarial_recall",
                "duration_ms": test_result.processing_time_ms,
                "success": test_result.detected_correctly,
                "error_message": "" if test_result.detected_correctly else "Detection failure",
                "metadata": str(test_result.metadata)
            }
            
            await self.clickhouse_client.insert_verification_event(event_data)
            
            # Log specific ARM metrics
            metrics = [
                {
                    "metric_id": str(uuid.uuid4()),
                    "timestamp": test_result.tested_at,
                    "metric_name": "adversarial_recall_accuracy",
                    "metric_value": 1.0 if test_result.detected_correctly else 0.0,
                    "metric_unit": "ratio",
                    "dimensions": f"category:{test_result.metadata['category']},difficulty:{test_result.metadata['difficulty']}",
                    "document_id": test_result.example_id,
                    "user_id": "system"
                },
                {
                    "metric_id": str(uuid.uuid4()),
                    "timestamp": test_result.tested_at,
                    "metric_name": "adversarial_confidence_score",
                    "metric_value": test_result.confidence_score,
                    "metric_unit": "score",
                    "dimensions": f"category:{test_result.metadata['category']},difficulty:{test_result.metadata['difficulty']}",
                    "document_id": test_result.example_id,
                    "user_id": "system"
                }
            ]
            
            for metric in metrics:
                await self.clickhouse_client.insert_kpi_metric(metric)
                
        except Exception as e:
            self.logger.error(f"Failed to log test result: {e}")
    
    async def calculate_arm_score(
        self,
        time_period_days: int = 30,
        category_filter: Optional[AdversarialCategory] = None,
        difficulty_filter: Optional[AdversarialDifficulty] = None
    ) -> Dict[str, Any]:
        """
        Calculate the overall Adversarial Recall Metric score.
        
        Args:
            time_period_days: Number of days to look back for test results
            category_filter: Optional filter by adversarial category
            difficulty_filter: Optional filter by difficulty level
            
        Returns:
            Dictionary containing ARM score and breakdown metrics
        """
        # Get test results for the specified period
        results = await self._get_test_results(
            time_period_days=time_period_days,
            category_filter=category_filter,
            difficulty_filter=difficulty_filter
        )
        
        if not results:
            return {
                "arm_score": 0.0,
                "total_tests": 0,
                "message": "No test results found for specified period"
            }
        
        # Calculate metrics
        total_tests = len(results)
        correct_detections = sum(1 for r in results if r.detected_correctly)
        false_positives = sum(1 for r in results if r.false_positive)
        false_negatives = sum(1 for r in results if r.false_negative)
        
        # ARM Score Components
        accuracy = correct_detections / total_tests if total_tests > 0 else 0.0
        precision = correct_detections / (correct_detections + false_positives) if (correct_detections + false_positives) > 0 else 0.0
        recall = correct_detections / (correct_detections + false_negatives) if (correct_detections + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Confidence calibration
        confidence_scores = [r.confidence_score for r in results]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        confidence_std = np.std(confidence_scores) if len(confidence_scores) > 1 else 0.0
        
        # Processing efficiency
        processing_times = [r.processing_time_ms for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # ACVF escalation rate
        acvf_escalations = sum(1 for r in results if r.acvf_triggered)
        acvf_escalation_rate = acvf_escalations / total_tests if total_tests > 0 else 0.0
        
        # Calculate weighted ARM score
        # Weight accuracy heavily, but also consider precision, recall, and confidence calibration
        arm_score = (
            0.4 * accuracy +
            0.25 * precision +
            0.25 * recall +
            0.1 * (1.0 - min(confidence_std / avg_confidence, 1.0) if avg_confidence > 0 else 0.0)
        )
        
        # Performance by category and difficulty
        category_breakdown = await self._calculate_category_performance(results)
        difficulty_breakdown = await self._calculate_difficulty_performance(results)
        
        return {
            "arm_score": round(arm_score, 4),
            "total_tests": total_tests,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "false_positive_rate": round(false_positives / total_tests, 4) if total_tests > 0 else 0.0,
            "false_negative_rate": round(false_negatives / total_tests, 4) if total_tests > 0 else 0.0,
            "average_confidence": round(avg_confidence, 4),
            "confidence_std": round(confidence_std, 4),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "acvf_escalation_rate": round(acvf_escalation_rate, 4),
            "category_performance": category_breakdown,
            "difficulty_performance": difficulty_breakdown,
            "recommendations": self._generate_recommendations(results)
        }
    
    async def _get_test_results(
        self,
        time_period_days: int,
        category_filter: Optional[AdversarialCategory] = None,
        difficulty_filter: Optional[AdversarialDifficulty] = None
    ) -> List[AdversarialTestResult]:
        """Get test results for the specified criteria."""
        # Filter results from memory (in production, query from ClickHouse)
        cutoff_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=time_period_days)
        
        filtered_results = []
        for result in self.test_results.values():
            if result.tested_at < cutoff_date:
                continue
                
            if category_filter and result.metadata.get("category") != category_filter.value:
                continue
                
            if difficulty_filter and result.metadata.get("difficulty") != difficulty_filter.value:
                continue
                
            filtered_results.append(result)
        
        return filtered_results
    
    async def _calculate_category_performance(
        self, 
        results: List[AdversarialTestResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance breakdown by adversarial category."""
        category_stats = {}
        
        for category in AdversarialCategory:
            category_results = [r for r in results if r.metadata.get("category") == category.value]
            
            if category_results:
                correct = sum(1 for r in category_results if r.detected_correctly)
                total = len(category_results)
                avg_confidence = np.mean([r.confidence_score for r in category_results])
                avg_processing_time = np.mean([r.processing_time_ms for r in category_results])
                
                category_stats[category.value] = {
                    "accuracy": round(correct / total, 4),
                    "total_tests": total,
                    "avg_confidence": round(avg_confidence, 4),
                    "avg_processing_time_ms": round(avg_processing_time, 2)
                }
        
        return category_stats
    
    async def _calculate_difficulty_performance(
        self, 
        results: List[AdversarialTestResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance breakdown by difficulty level."""
        difficulty_stats = {}
        
        for difficulty in AdversarialDifficulty:
            difficulty_results = [r for r in results if r.metadata.get("difficulty") == difficulty.value]
            
            if difficulty_results:
                correct = sum(1 for r in difficulty_results if r.detected_correctly)
                total = len(difficulty_results)
                avg_confidence = np.mean([r.confidence_score for r in difficulty_results])
                
                difficulty_stats[difficulty.value] = {
                    "accuracy": round(correct / total, 4),
                    "total_tests": total,
                    "avg_confidence": round(avg_confidence, 4)
                }
        
        return difficulty_stats
    
    def _generate_recommendations(self, results: List[AdversarialTestResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not results:
            return ["Insufficient test data for recommendations"]
        
        # Analyze performance patterns
        false_negatives = [r for r in results if r.false_negative]
        false_positives = [r for r in results if r.false_positive]
        
        # High false negative rate
        if len(false_negatives) / len(results) > 0.2:
            recommendations.append(
                "High false negative rate detected. Consider improving detection sensitivity "
                "and adding more verification passes."
            )
        
        # High false positive rate
        if len(false_positives) / len(results) > 0.15:
            recommendations.append(
                "High false positive rate detected. Consider refining verification criteria "
                "and improving confidence scoring."
            )
        
        # Low ACVF usage on difficult examples
        hard_examples = [r for r in results if r.metadata.get("difficulty") in ["hard", "expert"]]
        if hard_examples:
            acvf_usage = sum(1 for r in hard_examples if r.acvf_triggered) / len(hard_examples)
            if acvf_usage < 0.3:
                recommendations.append(
                    "Low ACVF escalation rate on difficult examples. Consider lowering "
                    "escalation thresholds for complex content."
                )
        
        # Category-specific recommendations
        category_performance = {}
        for result in results:
            category = result.metadata.get("category")
            if category not in category_performance:
                category_performance[category] = {"correct": 0, "total": 0}
            
            category_performance[category]["total"] += 1
            if result.detected_correctly:
                category_performance[category]["correct"] += 1
        
        for category, stats in category_performance.items():
            accuracy = stats["correct"] / stats["total"]
            if accuracy < 0.7:
                recommendations.append(
                    f"Low performance on {category} adversarial examples ({accuracy:.2%}). "
                    f"Consider specialized training or verification passes for this category."
                )
        
        return recommendations if recommendations else ["System performance is satisfactory"]


# Global ARM instance
arm_instance: Optional[AdversarialRecallMetric] = None


def get_adversarial_metric() -> AdversarialRecallMetric:
    """Get the global Adversarial Recall Metric instance."""
    global arm_instance
    if arm_instance is None:
        arm_instance = AdversarialRecallMetric()
    return arm_instance 