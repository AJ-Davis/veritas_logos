"""
ACVF Escalation Verification Pass.

This pass triggers the Adversarial Cross-Validation Framework when verification
results require additional scrutiny or when confidence levels are low.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from src.verification.passes.base_pass import BaseVerificationPass
from src.models.verification import VerificationContext, VerificationResult, VerificationStatus, VerificationPassType, VerificationPassConfig
from src.models.acvf import ACVFResult, DebateStatus
from src.verification.config.acvf_config_loader import load_default_acvf_config
from src.verification.acvf_controller import ACVFController
from src.llm.llm_client import LLMClient


class ACVFEscalationPass(BaseVerificationPass):
    """Verification pass that escalates uncertain results to ACVF for adversarial validation."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config_file: Optional[str] = None):
        """
        Initialize ACVF escalation pass.
        
        Args:
            llm_client: LLM client for ACVF debates (optional for base class compatibility)
            config_file: Optional path to ACVF configuration file
        """
        super().__init__(VerificationPassType.ADVERSARIAL_VALIDATION)
        
        self.llm_client = llm_client
        self.logger = logging.getLogger("verification.acvf_escalation")
        
        # Load ACVF configuration
        try:
            if config_file:
                from src.verification.config.acvf_config_loader import load_acvf_config_from_file
                self.acvf_config = load_acvf_config_from_file(config_file)
            else:
                self.acvf_config = load_default_acvf_config()
            
            self.logger.info(f"Loaded ACVF configuration: {self.acvf_config.name}")
        except Exception as e:
            self.logger.error(f"Failed to load ACVF configuration: {str(e)}")
            raise
        
        # Initialize ACVF controller (only if llm_client is provided)
        if self.llm_client:
            self.acvf_controller = ACVFController(self.llm_client, self.acvf_config)
        else:
            self.acvf_controller = None
    
    def get_required_dependencies(self) -> list[VerificationPassType]:
        """Get the list of verification pass types this pass depends on."""
        # ACVF can run after any verification pass but typically after initial passes
        return [VerificationPassType.CLAIM_EXTRACTION, VerificationPassType.CITATION_CHECK]
    
    async def execute(self, context: VerificationContext, config: VerificationPassConfig) -> VerificationResult:
        """
        Execute ACVF escalation based on verification context.
        
        Args:
            context: Verification context with previous results
            config: Pass configuration
            
        Returns:
            VerificationResult with ACVF debate outcomes
        """
        from datetime import datetime, timezone
        
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting ACVF escalation for document: {context.document_id}")
            
            # Check if ACVF controller is initialized
            if not self.acvf_controller:
                raise ValueError("ACVF controller not initialized - LLM client required")
            
            # Check if ACVF should be triggered
            should_escalate = await self.acvf_controller.should_trigger_acvf(context)
            
            if not should_escalate:
                self.logger.info("ACVF escalation not required based on trigger conditions")
                return VerificationResult(
                    pass_id=config.pass_id,
                    pass_type=self.pass_type,
                    status=VerificationStatus.COMPLETED,
                    started_at=start_time,
                    confidence_score=1.0,
                    result_data={
                        "escalation_triggered": False,
                        "reason": "Trigger conditions not met",
                        "trigger_conditions_checked": list(self.acvf_config.trigger_conditions.keys())
                    },
                    metadata={
                        "processing_time_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                        "acvf_config_id": self.acvf_config.config_id
                    }
                )
            
            self.logger.info("ACVF escalation triggered - conducting adversarial debates")
            
            # Conduct ACVF debates
            acvf_results = await self.acvf_controller.process_verification_escalation(context)
            
            # Aggregate results
            aggregated_result = self._aggregate_acvf_results(acvf_results, context)
            
            # Calculate overall confidence based on ACVF outcomes
            final_confidence = self._calculate_final_confidence(acvf_results)
            
            # Determine final status
            final_status = self._determine_final_status(acvf_results)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return VerificationResult(
                pass_id=config.pass_id,
                pass_type=self.pass_type,
                status=final_status,
                started_at=start_time,
                confidence_score=final_confidence,
                result_data=aggregated_result,
                metadata={
                    "processing_time_seconds": processing_time,
                    "acvf_config_id": self.acvf_config.config_id,
                    "debates_conducted": len(acvf_results),
                    "escalation_triggered": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"ACVF escalation failed: {str(e)}")
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return VerificationResult(
                pass_id=config.pass_id,
                pass_type=self.pass_type,
                status=VerificationStatus.FAILED,
                started_at=start_time,
                confidence_score=0.0,
                result_data={
                    "error": str(e),
                    "escalation_triggered": True,
                    "debates_conducted": 0
                },
                metadata={
                    "processing_time_seconds": processing_time,
                    "error_type": type(e).__name__
                }
            )
    
    def _aggregate_acvf_results(self, acvf_results: List[ACVFResult], 
                               context: VerificationContext) -> Dict[str, Any]:
        """Aggregate multiple ACVF debate results into a summary."""
        
        if not acvf_results:
            return {
                "debates_conducted": 0,
                "summary": "No debates were conducted"
            }
        
        # Aggregate statistics
        total_debates = len(acvf_results)
        total_rounds = sum(result.total_rounds for result in acvf_results)
        total_arguments = sum(result.total_arguments for result in acvf_results)
        
        # Count verdict outcomes
        verdict_counts = {}
        confidence_scores = []
        escalated_count = 0
        consensus_achieved_count = 0
        
        debate_summaries = []
        
        for result in acvf_results:
            # Track verdict distribution
            if result.final_verdict:
                verdict_counts[result.final_verdict.value] = verdict_counts.get(result.final_verdict.value, 0) + 1
            
            # Track confidence scores
            if result.final_confidence is not None:
                confidence_scores.append(result.final_confidence)
            
            # Track escalation and consensus
            if result.escalated:
                escalated_count += 1
            
            if result.consensus_achieved:
                consensus_achieved_count += 1
            
            # Create debate summary
            debate_summary = {
                "session_id": result.session_id,
                "subject_type": result.subject_type,
                "subject_id": result.subject_id,
                "final_verdict": result.final_verdict.value if result.final_verdict else None,
                "final_confidence": result.final_confidence,
                "total_rounds": result.total_rounds,
                "total_arguments": result.total_arguments,
                "consensus_achieved": result.consensus_achieved,
                "escalated": result.escalated,
                "duration_seconds": result.total_duration_seconds
            }
            debate_summaries.append(debate_summary)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine overall assessment
        overall_assessment = self._determine_overall_assessment(verdict_counts, avg_confidence, consensus_achieved_count, total_debates)
        
        return {
            "debates_conducted": total_debates,
            "total_rounds": total_rounds,
            "total_arguments": total_arguments,
            "verdict_distribution": verdict_counts,
            "average_confidence": avg_confidence,
            "consensus_rate": consensus_achieved_count / total_debates if total_debates > 0 else 0.0,
            "escalation_rate": escalated_count / total_debates if total_debates > 0 else 0.0,
            "overall_assessment": overall_assessment,
            "debate_summaries": debate_summaries,
            "recommendations": self._generate_recommendations(acvf_results, context)
        }
    
    def _calculate_final_confidence(self, acvf_results: List[ACVFResult]) -> float:
        """Calculate final confidence score based on ACVF results."""
        if not acvf_results:
            return 0.0
        
        confidence_scores = [result.final_confidence for result in acvf_results 
                           if result.final_confidence is not None]
        
        if not confidence_scores:
            return 0.5  # Neutral confidence if no scores available
        
        # Weight by consensus achievement
        weighted_scores = []
        for result in acvf_results:
            if result.final_confidence is not None:
                weight = 1.0
                if result.consensus_achieved:
                    weight += 0.2  # Boost confidence for consensus
                if result.escalated:
                    weight *= 0.9  # Slight penalty for escalation needed
                
                weighted_scores.append(result.final_confidence * weight)
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.5
    
    def _determine_final_status(self, acvf_results: List[ACVFResult]) -> VerificationStatus:
        """Determine final verification status based on ACVF results."""
        if not acvf_results:
            return VerificationStatus.FAILED
        
        # Check if any debates failed to complete
        failed_debates = [result for result in acvf_results 
                         if not result.debate_rounds or 
                         any(round.status == DebateStatus.FAILED for round in result.debate_rounds)]
        
        if failed_debates:
            return VerificationStatus.FAILED
        
        # Check consensus achievement rate
        consensus_count = sum(1 for result in acvf_results if result.consensus_achieved)
        consensus_rate = consensus_count / len(acvf_results)
        
        if consensus_rate >= 0.8:
            return VerificationStatus.COMPLETED
        elif consensus_rate >= 0.5:
            return VerificationStatus.PARTIAL_SUCCESS
        else:
            return VerificationStatus.FAILED
    
    def _determine_overall_assessment(self, verdict_counts: Dict[str, int], 
                                    avg_confidence: float, consensus_count: int, 
                                    total_debates: int) -> str:
        """Determine overall assessment of verification after ACVF."""
        
        if total_debates == 0:
            return "No debates conducted"
        
        consensus_rate = consensus_count / total_debates
        
        # Determine dominant verdict
        if verdict_counts:
            dominant_verdict = max(verdict_counts.keys(), key=lambda k: verdict_counts[k])
            dominant_count = verdict_counts[dominant_verdict]
            dominant_rate = dominant_count / total_debates
        else:
            dominant_verdict = "unknown"
            dominant_rate = 0.0
        
        # Generate assessment
        if consensus_rate >= 0.8 and avg_confidence >= 0.7:
            if dominant_verdict == "defender_wins":
                return "High confidence validation - original claims strongly supported"
            elif dominant_verdict == "challenger_wins":
                return "High confidence rejection - significant issues identified"
            else:
                return "High confidence mixed results - requires human review"
        
        elif consensus_rate >= 0.5 and avg_confidence >= 0.5:
            return f"Moderate confidence assessment - {dominant_verdict.replace('_', ' ')} in {dominant_rate:.1%} of debates"
        
        else:
            return "Low confidence assessment - requires additional verification or human review"
    
    def _generate_recommendations(self, acvf_results: List[ACVFResult], 
                                context: VerificationContext) -> List[str]:
        """Generate actionable recommendations based on ACVF results."""
        recommendations = []
        
        if not acvf_results:
            recommendations.append("Investigate why ACVF debates could not be conducted")
            return recommendations
        
        # Analyze verdict patterns
        verdict_counts = {}
        for result in acvf_results:
            if result.final_verdict:
                verdict_counts[result.final_verdict.value] = verdict_counts.get(result.final_verdict.value, 0) + 1
        
        consensus_rate = sum(1 for result in acvf_results if result.consensus_achieved) / len(acvf_results)
        avg_confidence = sum(result.final_confidence for result in acvf_results 
                           if result.final_confidence is not None) / len(acvf_results)
        
        # Generate specific recommendations
        if consensus_rate < 0.5:
            recommendations.append("Low consensus rate indicates document requires human expert review")
        
        if avg_confidence < 0.6:
            recommendations.append("Low average confidence suggests additional evidence gathering needed")
        
        if verdict_counts.get("challenger_wins", 0) > len(acvf_results) * 0.6:
            recommendations.append("Multiple challenges suggest significant content issues requiring revision")
        
        if verdict_counts.get("insufficient_evidence", 0) > 0:
            recommendations.append("Insufficient evidence verdicts indicate need for additional source verification")
        
        if any(result.escalated for result in acvf_results):
            recommendations.append("Escalated debates suggest complex issues requiring specialized expertise")
        
        # Add context-specific recommendations
        failed_passes = [result for result in context.previous_results 
                        if result.status == VerificationStatus.FAILED]
        
        if failed_passes:
            recommendations.append(f"Address specific failures in: {', '.join(p.pass_name for p in failed_passes)}")
        
        return recommendations or ["Continue with standard verification workflow"] 