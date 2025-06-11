"""
Adversarial Cross-Validation Framework (ACVF) Controller.

This module implements the core ACVF system that pits Challenger models against
Defender models with Judge adjudication for enhanced verification accuracy.
"""

import asyncio
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from src.models.acvf import (
    ACVFRole, DebateStatus, JudgeVerdict, ConfidenceLevel,
    ModelAssignment, DebateArgument, JudgeScore, DebateRound,
    ACVFConfiguration, ACVFResult
)
from src.models.verification import VerificationContext, VerificationResult
from src.llm.llm_client import LLMClient, LLMResponse
from src.verification.acvf_repository import ACVFRepository

logger = logging.getLogger(__name__)


class ACVFController:
    """Controller for orchestrating ACVF debates."""
    
    def __init__(self, llm_client: LLMClient, config: ACVFConfiguration, 
                 repository: Optional[ACVFRepository] = None):
        """
        Initialize ACVF controller.
        
        Args:
            llm_client: LLM client for generating responses
            config: ACVF configuration with model assignments and parameters
            repository: ACVF repository for database persistence (optional)
        """
        self.llm_client = llm_client
        self.config = config
        self.repository = repository or ACVFRepository()
        self.logger = logging.getLogger("acvf.controller")
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate ACVF configuration."""
        if not self.config.challenger_models:
            raise ValueError("At least one challenger model must be configured")
        if not self.config.defender_models:
            raise ValueError("At least one defender model must be configured")
        if not self.config.judge_models:
            raise ValueError("At least one judge model must be configured")
        
        # Check that models are properly configured for their roles
        for model in self.config.challenger_models:
            if model.role != ACVFRole.CHALLENGER:
                raise ValueError(f"Challenger model {model.model_id} has incorrect role: {model.role}")
        
        for model in self.config.defender_models:
            if model.role != ACVFRole.DEFENDER:
                raise ValueError(f"Defender model {model.model_id} has incorrect role: {model.role}")
        
        for model in self.config.judge_models:
            if model.role != ACVFRole.JUDGE:
                raise ValueError(f"Judge model {model.model_id} has incorrect role: {model.role}")
    
    async def should_trigger_acvf(self, verification_context: VerificationContext) -> bool:
        """
        Determine if ACVF should be triggered based on verification results.
        
        Args:
            verification_context: Context from verification chain
            
        Returns:
            True if ACVF should be triggered
        """
        trigger_conditions = self.config.trigger_conditions
        
        # Default trigger conditions if none specified
        if not trigger_conditions:
            # Trigger if any verification pass has low confidence
            for result in verification_context.previous_results:
                if result.confidence_score and result.confidence_score < 0.6:
                    self.logger.info(f"Triggering ACVF due to low confidence ({result.confidence_score}) in {result.pass_type}")
                    return True
            return False
        
        # Check configured trigger conditions
        if "min_confidence_threshold" in trigger_conditions:
            threshold = trigger_conditions["min_confidence_threshold"]
            for result in verification_context.previous_results:
                if result.confidence_score and result.confidence_score < threshold:
                    self.logger.info(f"Triggering ACVF: confidence {result.confidence_score} below threshold {threshold}")
                    return True
        
        if "escalate_failed_passes" in trigger_conditions and trigger_conditions["escalate_failed_passes"]:
            for result in verification_context.previous_results:
                if result.status.value == "failed":
                    self.logger.info(f"Triggering ACVF due to failed pass: {result.pass_type}")
                    return True
        
        if "escalate_on_issues" in trigger_conditions and trigger_conditions["escalate_on_issues"]:
            for result in verification_context.previous_results:
                # Check if result data indicates issues found
                if "issues_found" in result.result_data and result.result_data["issues_found"]:
                    self.logger.info(f"Triggering ACVF due to issues found in {result.pass_type}")
                    return True
        
        return False
    
    def _select_models(self) -> Tuple[ModelAssignment, ModelAssignment, List[ModelAssignment]]:
        """
        Select challenger, defender, and judge models for the debate.
        
        Returns:
            Tuple of (challenger, defender, judges)
        """
        # Randomly select from available models
        challenger = random.choice(self.config.challenger_models)
        defender = random.choice(self.config.defender_models)
        
        # For judges, we can use multiple for consensus
        if len(self.config.judge_models) == 1:
            judges = self.config.judge_models.copy()
        else:
            # Use all judges or select subset based on configuration
            num_judges = min(len(self.config.judge_models), 3)  # Max 3 judges for efficiency
            judges = random.sample(self.config.judge_models, num_judges)
        
        # Ensure no model conflicts (same model playing multiple roles)
        if not self.config.allow_model_self_assignment:
            challenger_key = f"{challenger.provider}:{challenger.model_id}"
            defender_key = f"{defender.provider}:{defender.model_id}"
            
            if challenger_key == defender_key:
                # Select different defender
                available_defenders = [m for m in self.config.defender_models 
                                     if f"{m.provider}:{m.model_id}" != challenger_key]
                if available_defenders:
                    defender = random.choice(available_defenders)
                else:
                    self.logger.warning("No alternative defender models available, allowing self-assignment")
            
            # Remove any judges that conflict with challenger/defender
            judges = [j for j in judges 
                     if f"{j.provider}:{j.model_id}" not in [challenger_key, defender_key]]
            
            if not judges:
                self.logger.warning("No non-conflicting judges available, using original selection")
                judges = random.sample(self.config.judge_models, 1)
        
        return challenger, defender, judges
    
    async def _generate_challenger_argument(self, debate_round: DebateRound, 
                                          context: str) -> DebateArgument:
        """Generate challenger argument for the debate."""
        try:
            provider_key = f"{debate_round.challenger_model.provider}:{debate_round.challenger_model.model_id}"
            
            response = await self.llm_client.generate_challenger_response(
                subject_content=debate_round.subject_content,
                context=context,
                provider=provider_key,
                temperature=debate_round.challenger_model.temperature,
                max_tokens=debate_round.challenger_model.max_tokens
            )
            
            argument = debate_round.add_argument(
                role=ACVFRole.CHALLENGER,
                content=response.content,
                round_number=debate_round.round_number,
                confidence_score=None  # Could extract from response if available
            )
            
            self.logger.info(f"Generated challenger argument for round {debate_round.round_number}")
            return argument
            
        except Exception as e:
            self.logger.error(f"Failed to generate challenger argument: {str(e)}")
            raise
    
    async def _generate_defender_argument(self, debate_round: DebateRound, 
                                        challenger_arguments: str, context: str) -> DebateArgument:
        """Generate defender argument for the debate."""
        try:
            provider_key = f"{debate_round.defender_model.provider}:{debate_round.defender_model.model_id}"
            
            response = await self.llm_client.generate_defender_response(
                subject_content=debate_round.subject_content,
                challenger_arguments=challenger_arguments,
                context=context,
                provider=provider_key,
                temperature=debate_round.defender_model.temperature,
                max_tokens=debate_round.defender_model.max_tokens
            )
            
            argument = debate_round.add_argument(
                role=ACVFRole.DEFENDER,
                content=response.content,
                round_number=debate_round.round_number,
                confidence_score=None
            )
            
            self.logger.info(f"Generated defender argument for round {debate_round.round_number}")
            return argument
            
        except Exception as e:
            self.logger.error(f"Failed to generate defender argument: {str(e)}")
            raise
    
    async def _generate_judge_scores(self, debate_round: DebateRound, 
                                   challenger_arguments: str, defender_arguments: str,
                                   context: str) -> List[JudgeScore]:
        """Generate judge scores for the debate round."""
        judge_scores = []
        
        for judge_model in debate_round.judge_models:
            try:
                provider_key = f"{judge_model.provider}:{judge_model.model_id}"
                
                start_time = time.time()
                response = await self.llm_client.generate_judge_verdict(
                    subject_content=debate_round.subject_content,
                    challenger_arguments=challenger_arguments,
                    defender_arguments=defender_arguments,
                    context=context,
                    provider=provider_key,
                    temperature=judge_model.temperature,
                    max_tokens=judge_model.max_tokens
                )
                processing_time = time.time() - start_time
                
                # Parse JSON response
                try:
                    verdict_data = json.loads(response.content)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse judge verdict JSON: {str(e)}")
                    continue
                
                # Create JudgeScore object
                judge_score = JudgeScore(
                    judge_id=f"{judge_model.provider}:{judge_model.model_id}",
                    verdict=JudgeVerdict(verdict_data["verdict"]),
                    confidence=verdict_data["confidence"],
                    confidence_level=verdict_data.get("confidence_level"),
                    challenger_score=verdict_data["challenger_score"],
                    defender_score=verdict_data["defender_score"],
                    reasoning=verdict_data["reasoning"],
                    key_points_challenger=verdict_data.get("key_points_challenger", []),
                    key_points_defender=verdict_data.get("key_points_defender", []),
                    critical_weaknesses=verdict_data.get("critical_weaknesses", []),
                    processing_time_seconds=processing_time
                )
                
                judge_scores.append(judge_score)
                self.logger.info(f"Judge {judge_score.judge_id} verdict: {judge_score.verdict} "
                               f"(confidence: {judge_score.confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"Failed to generate judge score for {judge_model.model_id}: {str(e)}")
                continue
        
        return judge_scores
    
    def _calculate_round_verdict(self, judge_scores: List[JudgeScore]) -> Tuple[Optional[JudgeVerdict], float]:
        """
        Calculate the final verdict for a debate round based on judge scores.
        
        Args:
            judge_scores: List of judge scores
            
        Returns:
            Tuple of (verdict, consensus_confidence)
        """
        if not judge_scores:
            return None, 0.0
        
        # Count verdicts
        verdict_counts = {}
        total_confidence = 0.0
        
        for score in judge_scores:
            verdict = score.verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            total_confidence += score.confidence
        
        # Find majority verdict
        if not verdict_counts:
            return None, 0.0
        
        max_count = max(verdict_counts.values())
        total_judges = len(judge_scores)
        majority_verdicts = [v for v, count in verdict_counts.items() if count == max_count]
        
        if len(majority_verdicts) > 1:
            # Tie - use average scores to break tie
            avg_challenger_score = sum(s.challenger_score for s in judge_scores) / len(judge_scores)
            avg_defender_score = sum(s.defender_score for s in judge_scores) / len(judge_scores)
            
            if abs(avg_challenger_score - avg_defender_score) < 0.1:
                final_verdict = JudgeVerdict.TIE
            elif avg_challenger_score > avg_defender_score:
                final_verdict = JudgeVerdict.CHALLENGER_WINS
            else:
                final_verdict = JudgeVerdict.DEFENDER_WINS
        else:
            final_verdict = majority_verdicts[0]
        
        # Calculate consensus confidence
        agreement_ratio = max_count / total_judges
        avg_confidence = total_confidence / total_judges
        consensus_confidence = agreement_ratio * avg_confidence
        
        return final_verdict, consensus_confidence
    
    async def conduct_debate_round(self, verification_task_id: str, subject_type: str,
                                 subject_id: str, subject_content: str, 
                                 context: str, round_number: int = 1) -> DebateRound:
        """
        Conduct a single round of ACVF debate.
        
        Args:
            verification_task_id: ID of the verification task
            subject_type: Type of subject being debated
            subject_id: ID of the subject being debated
            subject_content: Content of the subject being debated
            context: Context for the debate
            round_number: Round number
            
        Returns:
            Completed DebateRound object
        """
        self.logger.info(f"Starting ACVF debate round {round_number} for {subject_type}:{subject_id}")
        
        # Select models for this debate
        challenger, defender, judges = self._select_models()
        
        # Create debate round
        debate_round = DebateRound(
            verification_task_id=verification_task_id,
            subject_type=subject_type,
            subject_id=subject_id,
            subject_content=subject_content,
            challenger_model=challenger,
            defender_model=defender,
            judge_models=judges,
            round_number=round_number,
            max_rounds=self.config.max_rounds_per_debate,
            escalation_threshold=self.config.escalation_threshold,
            started_at=datetime.now(timezone.utc)
        )
        
        try:
            debate_round.status = DebateStatus.IN_PROGRESS
            
            # Step 1: Generate challenger argument
            challenger_arg = await self._generate_challenger_argument(debate_round, context)
            challenger_content = challenger_arg.content
            
            # Step 2: Generate defender response
            defender_arg = await self._generate_defender_argument(
                debate_round, challenger_content, context
            )
            defender_content = defender_arg.content
            
            # Step 3: Generate judge verdicts
            judge_scores = await self._generate_judge_scores(
                debate_round, challenger_content, defender_content, context
            )
            
            if not judge_scores:
                self.logger.error("No valid judge scores generated")
                debate_round.status = DebateStatus.FAILED
                return debate_round
            
            # Step 4: Calculate final verdict and consensus
            final_verdict, consensus_confidence = self._calculate_round_verdict(judge_scores)
            
            # Update debate round with results
            debate_round.judge_scores = judge_scores
            debate_round.final_verdict = final_verdict
            debate_round.consensus_confidence = consensus_confidence
            debate_round.status = DebateStatus.COMPLETED
            debate_round.completed_at = datetime.now(timezone.utc)
            
            if debate_round.started_at and debate_round.completed_at:
                debate_round.total_duration_seconds = (
                    debate_round.completed_at - debate_round.started_at
                ).total_seconds()
            
            # Save debate round to database
            try:
                db_round_id = self.repository.save_debate_round(debate_round)
                self.logger.info(f"Saved ACVF debate round {round_number} to database with ID: {db_round_id}")
            except Exception as e:
                self.logger.error(f"Failed to save debate round to database: {str(e)}")
                # Don't fail the operation if database save fails
            
            self.logger.info(f"Completed ACVF debate round {round_number}. "
                           f"Verdict: {final_verdict}, Consensus: {consensus_confidence:.2f}")
            
            return debate_round
            
        except Exception as e:
            self.logger.error(f"ACVF debate round {round_number} failed: {str(e)}")
            debate_round.status = DebateStatus.FAILED
            debate_round.completed_at = datetime.now(timezone.utc)
            
            # Save failed debate round to database for debugging
            try:
                db_round_id = self.repository.save_debate_round(debate_round)
                self.logger.info(f"Saved failed debate round to database with ID: {db_round_id}")
            except Exception as db_e:
                self.logger.error(f"Failed to save failed debate round to database: {str(db_e)}")
            
            raise
    
    async def conduct_full_debate(self, verification_context: VerificationContext,
                                subject_type: str, subject_id: str, 
                                subject_content: str) -> ACVFResult:
        """
        Conduct a complete ACVF debate session with multiple rounds if needed.
        
        Args:
            verification_context: Context from verification chain
            subject_type: Type of subject being debated
            subject_id: ID of the subject being debated  
            subject_content: Content of the subject being debated
            
        Returns:
            Complete ACVFResult with all debate rounds
        """
        self.logger.info(f"Starting full ACVF debate for {subject_type}:{subject_id}")
        
        # Create ACVF result object
        acvf_result = ACVFResult(
            verification_task_id=verification_context.document_id,
            subject_type=subject_type,
            subject_id=subject_id,
            acvf_config_id=self.config.config_id
        )
        
        # Build context string from verification results
        context_parts = [f"Document: {verification_context.document_id}"]
        for result in verification_context.previous_results:
            context_parts.append(f"{result.pass_type}: {result.result_data}")
        context = "\n".join(context_parts)
        
        try:
            current_round = 1
            consensus_achieved = False
            
            while current_round <= self.config.max_rounds_per_debate and not consensus_achieved:
                # Conduct debate round
                debate_round = await self.conduct_debate_round(
                    verification_task_id=verification_context.document_id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    subject_content=subject_content,
                    context=context,
                    round_number=current_round
                )
                
                acvf_result.add_debate_round(debate_round)
                
                # Check if consensus achieved
                if (debate_round.consensus_confidence and 
                    debate_round.consensus_confidence >= self.config.consensus_threshold):
                    consensus_achieved = True
                    self.logger.info(f"Consensus achieved in round {current_round} "
                                   f"with confidence {debate_round.consensus_confidence:.2f}")
                
                current_round += 1
            
            # Calculate final metrics
            acvf_result.calculate_final_metrics()
            acvf_result.consensus_achieved = consensus_achieved
            
            if not consensus_achieved:
                self.logger.warning(f"ACVF debate completed without consensus after "
                                  f"{self.config.max_rounds_per_debate} rounds")
                acvf_result.escalated = True
            
            # Save ACVF session result to database
            try:
                session_db_id = self.repository.save_acvf_session(acvf_result)
                self.logger.info(f"Saved ACVF session to database with ID: {session_db_id}")
            except Exception as e:
                self.logger.error(f"Failed to save ACVF session to database: {str(e)}")
                # Don't fail the operation if database save fails
            
            return acvf_result
            
        except Exception as e:
            self.logger.error(f"ACVF full debate failed: {str(e)}")
            acvf_result.calculate_final_metrics()
            
            # Save failed session to database for debugging
            try:
                session_db_id = self.repository.save_acvf_session(acvf_result)
                self.logger.info(f"Saved failed ACVF session to database with ID: {session_db_id}")
            except Exception as db_e:
                self.logger.error(f"Failed to save failed ACVF session to database: {str(db_e)}")
            
            raise
    
    async def process_verification_escalation(self, verification_context: VerificationContext) -> List[ACVFResult]:
        """
        Process verification results that need ACVF escalation.
        
        Args:
            verification_context: Context from verification chain
            
        Returns:
            List of ACVF results for escalated items
        """
        if not await self.should_trigger_acvf(verification_context):
            return []
        
        acvf_results = []
        
        # For each verification result that needs escalation
        for result in verification_context.previous_results:
            if result.confidence_score and result.confidence_score < self.config.escalation_threshold:
                # Extract subjects for debate (claims, citations, etc.)
                subjects = self._extract_debate_subjects(result)
                
                for subject_type, subject_id, subject_content in subjects:
                    try:
                        acvf_result = await self.conduct_full_debate(
                            verification_context=verification_context,
                            subject_type=subject_type,
                            subject_id=subject_id,
                            subject_content=subject_content
                        )
                        acvf_results.append(acvf_result)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to conduct ACVF debate for {subject_type}:{subject_id}: {str(e)}")
                        continue
        
        return acvf_results
    
    def _extract_debate_subjects(self, verification_result: VerificationResult) -> List[Tuple[str, str, str]]:
        """
        Extract subjects for debate from verification results.
        
        Args:
            verification_result: Result from verification pass
            
        Returns:
            List of (subject_type, subject_id, subject_content) tuples
        """
        subjects = []
        result_data = verification_result.result_data
        
        # Extract based on pass type
        if verification_result.pass_type.value == "claim_extraction" and "claims" in result_data:
            for claim in result_data["claims"]:
                if isinstance(claim, dict) and "text" in claim:
                    claim_id = claim.get("id", str(len(subjects)))
                    subjects.append(("claim", claim_id, claim["text"]))
        
        elif verification_result.pass_type.value == "citation_check" and "citations" in result_data:
            for citation in result_data["citations"]:
                if isinstance(citation, dict) and "text" in citation:
                    citation_id = citation.get("id", str(len(subjects)))
                    subjects.append(("citation", citation_id, citation["text"]))
        
        # Add more extraction logic for other pass types as needed
        
        return subjects 