"""
ACVF Repository for database persistence operations.

This module provides a repository layer for storing and retrieving ACVF debate data
using SQLAlchemy ORM models while working with Pydantic data models.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from ..models.database import (
    DatabaseSession, DBModelAssignment, DBDebateRound, DBDebateArgument, 
    DBJudgeScore, DBACVFSession, DatabaseManager
)
from ..models.acvf import (
    ModelAssignment, DebateRound, DebateArgument, JudgeScore, ACVFResult,
    ACVFRole, DebateStatus, JudgeVerdict, ConfidenceLevel
)


class ACVFRepository:
    """Repository for ACVF database operations."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    # Model Assignment Operations
    
    def save_model_assignment(self, assignment: ModelAssignment) -> str:
        """Save a model assignment to the database."""
        with DatabaseSession() as session:
            db_assignment = DBModelAssignment(
                model_id=assignment.model_id,
                provider=assignment.provider,
                role=assignment.role.value,
                temperature=assignment.temperature,
                max_tokens=assignment.max_tokens,
                system_prompt_override=assignment.system_prompt_override,
                metadata=assignment.metadata
            )
            session.add(db_assignment)
            session.flush()
            return db_assignment.id
    
    def get_model_assignment(self, assignment_id: str) -> Optional[ModelAssignment]:
        """Get a model assignment by ID."""
        with DatabaseSession() as session:
            db_assignment = session.query(DBModelAssignment).filter(
                DBModelAssignment.id == assignment_id
            ).first()
            
            if not db_assignment:
                return None
            
            return ModelAssignment(
                model_id=db_assignment.model_id,
                provider=db_assignment.provider,
                role=ACVFRole(db_assignment.role),
                temperature=db_assignment.temperature,
                max_tokens=db_assignment.max_tokens,
                system_prompt_override=db_assignment.system_prompt_override,
                metadata=db_assignment.metadata
            )
    
    def find_model_assignments_by_role(self, role: ACVFRole) -> List[ModelAssignment]:
        """Find all model assignments for a specific role."""
        with DatabaseSession() as session:
            db_assignments = session.query(DBModelAssignment).filter(
                DBModelAssignment.role == role.value
            ).all()
            
            return [
                ModelAssignment(
                    model_id=db_assignment.model_id,
                    provider=db_assignment.provider,
                    role=ACVFRole(db_assignment.role),
                    temperature=db_assignment.temperature,
                    max_tokens=db_assignment.max_tokens,
                    system_prompt_override=db_assignment.system_prompt_override,
                    metadata=db_assignment.metadata
                )
                for db_assignment in db_assignments
            ]
    
    # Debate Round Operations
    
    def save_debate_round(self, debate_round: DebateRound) -> str:
        """Save a debate round to the database."""
        with DatabaseSession() as session:
            # Save model assignments first
            challenger_id = self.save_model_assignment(debate_round.challenger_model)
            defender_id = self.save_model_assignment(debate_round.defender_model)
            
            # Create debate round
            db_round = DBDebateRound(
                round_id=debate_round.round_id,
                verification_task_id=debate_round.verification_task_id,
                subject_type=debate_round.subject_type,
                subject_id=debate_round.subject_id,
                subject_content=debate_round.subject_content,
                challenger_model_id=challenger_id,
                defender_model_id=defender_id,
                round_number=debate_round.round_number,
                max_rounds=debate_round.max_rounds,
                status=debate_round.status.value,
                final_verdict=debate_round.final_verdict.value if debate_round.final_verdict else None,
                consensus_confidence=debate_round.consensus_confidence,
                started_at=debate_round.started_at,
                completed_at=debate_round.completed_at,
                total_duration_seconds=debate_round.total_duration_seconds,
                debate_config=debate_round.debate_config,
                escalation_threshold=debate_round.escalation_threshold,
                metadata=debate_round.metadata
            )
            session.add(db_round)
            session.flush()
            
            # Save arguments
            for argument in debate_round.arguments:
                self._save_debate_argument_to_session(session, argument, db_round.id)
            
            # Save judge assignments and scores
            for judge_model in debate_round.judge_models:
                judge_id = self.save_model_assignment(judge_model)
            
            for judge_score in debate_round.judge_scores:
                self._save_judge_score_to_session(session, judge_score, db_round.id)
            
            return db_round.id
    
    def _save_debate_argument_to_session(self, session: Session, argument: DebateArgument, round_id: str):
        """Save a debate argument within an existing session."""
        db_argument = DBDebateArgument(
            argument_id=argument.argument_id,
            debate_round_id=round_id,
            role=argument.role.value,
            content=argument.content,
            round_number=argument.round_number,
            references=argument.references,
            confidence_score=argument.confidence_score,
            metadata=argument.metadata,
            timestamp=argument.timestamp
        )
        session.add(db_argument)
    
    def _save_judge_score_to_session(self, session: Session, judge_score: JudgeScore, round_id: str):
        """Save a judge score within an existing session."""
        # Find the judge model ID
        judge_assignment = session.query(DBModelAssignment).filter(
            and_(
                DBModelAssignment.model_id == judge_score.judge_model_id,
                DBModelAssignment.role == "judge"
            )
        ).first()
        
        if not judge_assignment:
            # Create a basic judge assignment if not found
            judge_assignment = DBModelAssignment(
                model_id=judge_score.judge_model_id,
                provider="unknown",  # Will need to be updated when we have more info
                role="judge"
            )
            session.add(judge_assignment)
            session.flush()
        
        db_score = DBJudgeScore(
            score_id=judge_score.score_id,
            debate_round_id=round_id,
            judge_model_id=judge_assignment.id,
            challenger_score=judge_score.challenger_score,
            defender_score=judge_score.defender_score,
            confidence_level=judge_score.confidence_level.value,
            verdict=judge_score.verdict.value,
            reasoning=judge_score.reasoning,
            key_points=judge_score.key_points,
            evidence_assessment=judge_score.evidence_assessment,
            argument_quality_challenger=judge_score.argument_quality_challenger,
            argument_quality_defender=judge_score.argument_quality_defender,
            evidence_strength_challenger=judge_score.evidence_strength_challenger,
            evidence_strength_defender=judge_score.evidence_strength_defender,
            logical_consistency_challenger=judge_score.logical_consistency_challenger,
            logical_consistency_defender=judge_score.logical_consistency_defender,
            processing_time_seconds=judge_score.processing_time_seconds,
            response_metadata=judge_score.response_metadata
        )
        session.add(db_score)
    
    def get_debate_round(self, round_id: str) -> Optional[DebateRound]:
        """Get a debate round by ID."""
        with DatabaseSession() as session:
            db_round = session.query(DBDebateRound).filter(
                DBDebateRound.round_id == round_id
            ).first()
            
            if not db_round:
                return None
            
            return self._convert_db_round_to_pydantic(db_round)
    
    def _convert_db_round_to_pydantic(self, db_round: DBDebateRound) -> DebateRound:
        """Convert a database debate round to a Pydantic model."""
        # Get model assignments
        challenger_model = self.get_model_assignment(db_round.challenger_model_id)
        defender_model = self.get_model_assignment(db_round.defender_model_id)
        
        # Get judge models (from judge scores)
        judge_models = []
        for score in db_round.judge_scores:
            judge_model = self.get_model_assignment(score.judge_model_id)
            if judge_model and judge_model not in judge_models:
                judge_models.append(judge_model)
        
        # Convert arguments
        arguments = [
            DebateArgument(
                argument_id=arg.argument_id,
                role=ACVFRole(arg.role),
                content=arg.content,
                timestamp=arg.timestamp,
                round_number=arg.round_number,
                references=arg.references,
                confidence_score=arg.confidence_score,
                metadata=arg.metadata
            )
            for arg in db_round.arguments
        ]
        
        # Convert judge scores
        judge_scores = [
            JudgeScore(
                score_id=score.score_id,
                judge_model_id=score.judge_model.model_id,
                challenger_score=score.challenger_score,
                defender_score=score.defender_score,
                confidence_level=ConfidenceLevel(score.confidence_level),
                verdict=JudgeVerdict(score.verdict),
                reasoning=score.reasoning,
                key_points=score.key_points,
                evidence_assessment=score.evidence_assessment,
                argument_quality_challenger=score.argument_quality_challenger,
                argument_quality_defender=score.argument_quality_defender,
                evidence_strength_challenger=score.evidence_strength_challenger,
                evidence_strength_defender=score.evidence_strength_defender,
                logical_consistency_challenger=score.logical_consistency_challenger,
                logical_consistency_defender=score.logical_consistency_defender,
                processing_time_seconds=score.processing_time_seconds,
                response_metadata=score.response_metadata
            )
            for score in db_round.judge_scores
        ]
        
        return DebateRound(
            round_id=db_round.round_id,
            verification_task_id=db_round.verification_task_id,
            subject_type=db_round.subject_type,
            subject_id=db_round.subject_id,
            subject_content=db_round.subject_content,
            challenger_model=challenger_model,
            defender_model=defender_model,
            judge_models=judge_models,
            status=DebateStatus(db_round.status),
            round_number=db_round.round_number,
            max_rounds=db_round.max_rounds,
            arguments=arguments,
            judge_scores=judge_scores,
            final_verdict=JudgeVerdict(db_round.final_verdict) if db_round.final_verdict else None,
            consensus_confidence=db_round.consensus_confidence,
            started_at=db_round.started_at,
            completed_at=db_round.completed_at,
            total_duration_seconds=db_round.total_duration_seconds,
            debate_config=db_round.debate_config,
            escalation_threshold=db_round.escalation_threshold,
            metadata=db_round.metadata
        )
    
    def find_debate_rounds_by_task(self, verification_task_id: str) -> List[DebateRound]:
        """Find all debate rounds for a verification task."""
        with DatabaseSession() as session:
            db_rounds = session.query(DBDebateRound).filter(
                DBDebateRound.verification_task_id == verification_task_id
            ).order_by(asc(DBDebateRound.created_at)).all()
            
            return [
                self._convert_db_round_to_pydantic(db_round)
                for db_round in db_rounds
            ]
    
    def find_debate_rounds_by_subject(self, subject_type: str, subject_id: str) -> List[DebateRound]:
        """Find all debate rounds for a specific subject."""
        with DatabaseSession() as session:
            db_rounds = session.query(DBDebateRound).filter(
                and_(
                    DBDebateRound.subject_type == subject_type,
                    DBDebateRound.subject_id == subject_id
                )
            ).order_by(asc(DBDebateRound.created_at)).all()
            
            return [
                self._convert_db_round_to_pydantic(db_round)
                for db_round in db_rounds
            ]
    
    def update_debate_round_status(self, round_id: str, status: DebateStatus, 
                                 final_verdict: Optional[JudgeVerdict] = None,
                                 consensus_confidence: Optional[float] = None):
        """Update the status and results of a debate round."""
        with DatabaseSession() as session:
            db_round = session.query(DBDebateRound).filter(
                DBDebateRound.round_id == round_id
            ).first()
            
            if db_round:
                db_round.status = status.value
                if final_verdict:
                    db_round.final_verdict = final_verdict.value
                if consensus_confidence is not None:
                    db_round.consensus_confidence = consensus_confidence
                if status == DebateStatus.COMPLETED:
                    db_round.completed_at = datetime.now(timezone.utc)
                    if db_round.started_at:
                        db_round.total_duration_seconds = (
                            db_round.completed_at - db_round.started_at
                        ).total_seconds()
    
    # ACVF Session Operations
    
    def save_acvf_session(self, result: ACVFResult) -> str:
        """Save an ACVF session result."""
        with DatabaseSession() as session:
            db_session = DBACVFSession(
                session_id=result.session_id,
                verification_task_id=result.verification_task_id,
                subject_type=result.subject_type,
                subject_id=result.subject_id,
                final_verdict=result.final_verdict.value if result.final_verdict else None,
                final_confidence=result.final_confidence,
                escalated=result.escalated,
                consensus_achieved=result.consensus_achieved,
                total_rounds=result.total_rounds,
                total_arguments=result.total_arguments,
                average_judge_confidence=result.average_judge_confidence,
                acvf_config_id=result.acvf_config_id,
                started_at=result.started_at,
                completed_at=result.completed_at,
                total_duration_seconds=result.total_duration_seconds,
                metadata=result.metadata
            )
            session.add(db_session)
            session.flush()
            
            # Save all debate rounds associated with this session
            for debate_round in result.debate_rounds:
                self.save_debate_round(debate_round)
            
            return db_session.id
    
    def get_acvf_session(self, session_id: str) -> Optional[ACVFResult]:
        """Get an ACVF session by ID."""
        with DatabaseSession() as session:
            db_session = session.query(DBACVFSession).filter(
                DBACVFSession.session_id == session_id
            ).first()
            
            if not db_session:
                return None
            
            # Get associated debate rounds
            debate_rounds = self.find_debate_rounds_by_task(db_session.verification_task_id)
            
            return ACVFResult(
                session_id=db_session.session_id,
                verification_task_id=db_session.verification_task_id,
                subject_type=db_session.subject_type,
                subject_id=db_session.subject_id,
                debate_rounds=debate_rounds,
                final_verdict=JudgeVerdict(db_session.final_verdict) if db_session.final_verdict else None,
                final_confidence=db_session.final_confidence,
                escalated=db_session.escalated,
                total_rounds=db_session.total_rounds,
                total_arguments=db_session.total_arguments,
                average_judge_confidence=db_session.average_judge_confidence,
                consensus_achieved=db_session.consensus_achieved,
                started_at=db_session.started_at,
                completed_at=db_session.completed_at,
                total_duration_seconds=db_session.total_duration_seconds,
                acvf_config_id=db_session.acvf_config_id,
                metadata=db_session.metadata
            )
    
    # Query and Analytics Operations
    
    def get_debate_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get debate statistics for the last N days."""
        with DatabaseSession() as session:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Total debates
            total_debates = session.query(DBDebateRound).filter(
                DBDebateRound.created_at >= cutoff_date
            ).count()
            
            # Completed debates
            completed_debates = session.query(DBDebateRound).filter(
                and_(
                    DBDebateRound.created_at >= cutoff_date,
                    DBDebateRound.status == "completed"
                )
            ).count()
            
            # Average duration
            avg_duration = session.query(func.avg(DBDebateRound.total_duration_seconds)).filter(
                and_(
                    DBDebateRound.created_at >= cutoff_date,
                    DBDebateRound.status == "completed"
                )
            ).scalar()
            
            # Verdict distribution
            verdict_counts = session.query(
                DBDebateRound.final_verdict,
                func.count(DBDebateRound.id)
            ).filter(
                and_(
                    DBDebateRound.created_at >= cutoff_date,
                    DBDebateRound.final_verdict.isnot(None)
                )
            ).group_by(DBDebateRound.final_verdict).all()
            
            return {
                "total_debates": total_debates,
                "completed_debates": completed_debates,
                "completion_rate": completed_debates / total_debates if total_debates > 0 else 0,
                "average_duration_seconds": avg_duration,
                "verdict_distribution": dict(verdict_counts)
            }
    
    def find_recent_debates(self, limit: int = 10) -> List[DebateRound]:
        """Find the most recent debate rounds."""
        with DatabaseSession() as session:
            db_rounds = session.query(DBDebateRound).order_by(
                desc(DBDebateRound.created_at)
            ).limit(limit).all()
            
            return [
                self._convert_db_round_to_pydantic(db_round)
                for db_round in db_rounds
            ]
    
    def cleanup_old_debates(self, days: int = 90) -> int:
        """Clean up old debate records (older than N days)."""
        with DatabaseSession() as session:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Count records to be deleted
            count = session.query(DBDebateRound).filter(
                DBDebateRound.created_at < cutoff_date
            ).count()
            
            # Delete old records (cascades to arguments and scores)
            session.query(DBDebateRound).filter(
                DBDebateRound.created_at < cutoff_date
            ).delete()
            
            return count 