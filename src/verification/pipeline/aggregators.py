"""
Result aggregation and scoring components for verification pipeline.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.models.verification import (
    VerificationChainResult,
    VerificationResult,
    VerificationStatus,
    VerificationPassType
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatedScore:
    """Aggregated scoring result."""
    overall_score: float
    weighted_score: float
    confidence_score: float
    pass_scores: Dict[VerificationPassType, float]
    flags: List[str]
    recommendations: List[str]


class WeightedScorer:
    """Weighted scoring system for verification results."""
    
    def __init__(self, pass_weights: Dict[VerificationPassType, float]):
        """
        Initialize the weighted scorer.
        
        Args:
            pass_weights: Weights for different verification pass types
        """
        self.pass_weights = pass_weights
    
    def calculate_weighted_score(self, results: List[VerificationResult]) -> AggregatedScore:
        """
        Calculate weighted score from verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            AggregatedScore with calculated metrics
        """
        if not results:
            return AggregatedScore(
                overall_score=0.0,
                weighted_score=0.0,
                confidence_score=0.0,
                pass_scores={},
                flags=["No results available"],
                recommendations=["Unable to verify - no passes executed"]
            )
        
        pass_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_scores = []
        flags = []
        recommendations = []
        
        for result in results:
            pass_type = result.pass_type
            weight = self.pass_weights.get(pass_type, 1.0)
            
            # Calculate pass score based on status and confidence
            if result.status == VerificationStatus.COMPLETED:
                pass_score = result.confidence_score or 0.8  # Default confidence if not provided
            elif result.status == VerificationStatus.FAILED:
                pass_score = 0.0
                flags.append(f"{pass_type.value} verification failed")
                recommendations.append(f"Review {pass_type.value} issues")
            else:
                pass_score = 0.5  # Partial completion
            
            pass_scores[pass_type] = pass_score
            weighted_sum += pass_score * weight
            total_weight += weight
            
            if result.confidence_score:
                confidence_scores.append(result.confidence_score)
            
            # Add pass-specific flags and recommendations
            self._add_pass_specific_insights(result, flags, recommendations)
        
        # Calculate final scores
        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_score = sum(pass_scores.values()) / len(pass_scores)
        confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Add overall recommendations
        if weighted_score < 0.5:
            recommendations.append("Document requires significant review")
        elif weighted_score < 0.7:
            recommendations.append("Document needs minor improvements")
        else:
            recommendations.append("Document meets verification standards")
        
        return AggregatedScore(
            overall_score=overall_score,
            weighted_score=weighted_score,
            confidence_score=confidence_score,
            pass_scores=pass_scores,
            flags=flags,
            recommendations=recommendations
        )
    
    def _add_pass_specific_insights(self, result: VerificationResult, 
                                  flags: List[str], recommendations: List[str]) -> None:
        """Add pass-specific insights to flags and recommendations."""
        if not result.result_data:
            return
        
        pass_type = result.pass_type
        
        if pass_type == VerificationPassType.CLAIM_EXTRACTION:
            claims_count = result.result_data.get('total_claims', 0)
            if claims_count == 0:
                flags.append("No verifiable claims found")
            elif claims_count > 50:
                flags.append(f"High number of claims ({claims_count}) detected")
                recommendations.append("Consider breaking document into smaller sections")
        
        elif pass_type == VerificationPassType.CITATION_CHECK:
            invalid_citations = result.result_data.get('invalid_citations', [])
            if invalid_citations:
                flags.append(f"{len(invalid_citations)} invalid citations found")
                recommendations.append("Review and update citation sources")
        
        elif pass_type == VerificationPassType.LOGIC_ANALYSIS:
            fallacies = result.result_data.get('logical_fallacies', [])
            if fallacies:
                flags.append(f"{len(fallacies)} logical fallacies detected")
                recommendations.append("Review argument structure and reasoning")
        
        elif pass_type == VerificationPassType.BIAS_SCAN:
            biases = result.result_data.get('detected_biases', [])
            if biases:
                flags.append(f"{len(biases)} potential biases detected")
                recommendations.append("Review content for balanced perspective")


class ResultAggregator:
    """Aggregates verification results from multiple passes and pipelines."""
    
    def __init__(self, config):
        """
        Initialize the result aggregator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.scorer = WeightedScorer(config.pass_weights) if config.use_weighted_scoring else None
    
    def aggregate_results(self, chain_result: VerificationChainResult) -> Dict[str, Any]:
        """
        Aggregate results from a verification chain.
        
        Args:
            chain_result: Chain result to aggregate
            
        Returns:
            Aggregated result summary
        """
        if not chain_result.pass_results:
            return {
                "status": "no_results",
                "summary": "No verification passes were executed",
                "recommendations": ["Check verification configuration"]
            }
        
        # Calculate scores if weighted scoring is enabled
        scores = None
        if self.scorer:
            scores = self.scorer.calculate_weighted_score(chain_result.pass_results)
        
        # Basic aggregation
        total_passes = len(chain_result.pass_results)
        completed_passes = sum(1 for r in chain_result.pass_results 
                             if r.status == VerificationStatus.COMPLETED)
        failed_passes = sum(1 for r in chain_result.pass_results 
                          if r.status == VerificationStatus.FAILED)
        
        # Determine overall status
        if failed_passes == 0:
            overall_status = "passed"
        elif completed_passes > failed_passes:
            overall_status = "passed_with_issues"
        else:
            overall_status = "failed"
        
        # Create summary
        summary = {
            "overall_status": overall_status,
            "execution_summary": {
                "total_passes": total_passes,
                "completed_passes": completed_passes,
                "failed_passes": failed_passes,
                "success_rate": completed_passes / total_passes if total_passes > 0 else 0.0
            },
            "confidence": chain_result.overall_confidence,
            "execution_time": chain_result.total_execution_time_seconds
        }
        
        # Add scoring information if available
        if scores:
            summary["scoring"] = {
                "overall_score": scores.overall_score,
                "weighted_score": scores.weighted_score,
                "confidence_score": scores.confidence_score,
                "pass_scores": {pt.value: score for pt, score in scores.pass_scores.items()}
            }
            summary["flags"] = scores.flags
            summary["recommendations"] = scores.recommendations
        else:
            summary["flags"] = self._extract_basic_flags(chain_result)
            summary["recommendations"] = self._generate_basic_recommendations(chain_result)
        
        # Add pass-specific details
        summary["pass_details"] = {}
        for result in chain_result.pass_results:
            summary["pass_details"][result.pass_type.value] = {
                "status": result.status.value,
                "confidence": result.confidence_score,
                "execution_time": result.execution_time_seconds,
                "has_data": bool(result.result_data)
            }
        
        return summary
    
    def merge_pipeline_results(self, standard_result: VerificationChainResult,
                             acvf_result: VerificationChainResult) -> VerificationChainResult:
        """
        Merge results from standard and ACVF pipelines.
        
        Args:
            standard_result: Result from standard pipeline
            acvf_result: Result from ACVF pipeline
            
        Returns:
            Merged verification chain result
        """
        # Create a new merged result
        merged_result = VerificationChainResult(
            chain_id="hybrid_pipeline",
            document_id=standard_result.document_id,
            status=VerificationStatus.COMPLETED,
            started_at=min(standard_result.started_at, acvf_result.started_at),
            completed_at=max(
                standard_result.completed_at or datetime.utcnow(),
                acvf_result.completed_at or datetime.utcnow()
            )
        )
        
        # Combine pass results
        merged_result.pass_results = standard_result.pass_results + acvf_result.pass_results
        
        # Calculate merged confidence
        all_confidences = []
        for result in merged_result.pass_results:
            if result.confidence_score:
                all_confidences.append(result.confidence_score)
        
        if all_confidences:
            merged_result.overall_confidence = sum(all_confidences) / len(all_confidences)
        
        # Combine errors and warnings
        merged_result.errors = standard_result.errors + acvf_result.errors
        merged_result.warnings = standard_result.warnings + acvf_result.warnings
        
        # Calculate total execution time
        standard_time = standard_result.total_execution_time_seconds or 0
        acvf_time = acvf_result.total_execution_time_seconds or 0
        merged_result.total_execution_time_seconds = standard_time + acvf_time
        
        # Create merged summary
        merged_result.summary = {
            "standard_pipeline": standard_result.summary,
            "acvf_pipeline": acvf_result.summary,
            "total_passes": len(merged_result.pass_results),
            "total_execution_time": merged_result.total_execution_time_seconds
        }
        
        return merged_result
    
    def _extract_basic_flags(self, chain_result: VerificationChainResult) -> List[str]:
        """Extract basic flags from chain result."""
        flags = []
        
        for result in chain_result.pass_results:
            if result.status == VerificationStatus.FAILED:
                flags.append(f"{result.pass_type.value} verification failed")
            elif result.confidence_score and result.confidence_score < 0.5:
                flags.append(f"Low confidence in {result.pass_type.value}")
        
        if chain_result.errors:
            flags.append(f"{len(chain_result.errors)} execution errors occurred")
        
        return flags
    
    def _generate_basic_recommendations(self, chain_result: VerificationChainResult) -> List[str]:
        """Generate basic recommendations from chain result."""
        recommendations = []
        
        failed_passes = [r for r in chain_result.pass_results 
                        if r.status == VerificationStatus.FAILED]
        
        if failed_passes:
            recommendations.append("Review and address failed verification passes")
        
        low_confidence_passes = [r for r in chain_result.pass_results 
                               if r.confidence_score and r.confidence_score < 0.7]
        
        if low_confidence_passes:
            recommendations.append("Consider additional review for low-confidence results")
        
        if not recommendations:
            recommendations.append("Document appears to meet verification standards")
        
        return recommendations 