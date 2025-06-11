"""
Database models for VeritasLogos using SQLAlchemy ORM.

This module provides persistent storage for ACVF debate rounds, judge scores,
and other verification data that needs to be stored across sessions.
"""

import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, DateTime, 
    Text, JSON, ForeignKey, Enum as SQLEnum, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///veritas_logos.db")

# Create database engine
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    pool_pre_ping=True
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all database models
Base = declarative_base()


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Session will be closed by caller


# ACVF Database Models

class DBModelAssignment(Base):
    """Database model for storing model assignments in debates."""
    __tablename__ = "model_assignments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False)
    role = Column(SQLEnum("challenger", "defender", "judge", name="acvf_role"), nullable=False)
    temperature = Column(Float, nullable=False, default=0.7)
    max_tokens = Column(Integer, nullable=False, default=2000)
    system_prompt_override = Column(Text, nullable=True)
    model_metadata = Column(JSON, nullable=False, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    challenger_debates = relationship("DBDebateRound", foreign_keys="DBDebateRound.challenger_model_id", back_populates="challenger_model")
    defender_debates = relationship("DBDebateRound", foreign_keys="DBDebateRound.defender_model_id", back_populates="defender_model")
    judge_scores = relationship("DBJudgeScore", back_populates="judge_model")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(temperature >= 0.0, name='check_temperature_positive'),
        CheckConstraint(temperature <= 2.0, name='check_temperature_max'),
        CheckConstraint(max_tokens > 0, name='check_max_tokens_positive'),
        Index('idx_model_assignments_model_role', 'model_id', 'role'),
    )


class DBDebateRound(Base):
    """Database model for storing ACVF debate rounds."""
    __tablename__ = "debate_rounds"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    round_id = Column(String, nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    verification_task_id = Column(String, nullable=False, index=True)
    subject_type = Column(String, nullable=False)  # 'claim', 'citation', etc.
    subject_id = Column(String, nullable=False, index=True)
    subject_content = Column(Text, nullable=False)
    
    # Model assignments (foreign keys)
    challenger_model_id = Column(String, ForeignKey("model_assignments.id"), nullable=False)
    defender_model_id = Column(String, ForeignKey("model_assignments.id"), nullable=False)
    
    # Debate configuration
    round_number = Column(Integer, nullable=False, default=1)
    max_rounds = Column(Integer, nullable=False, default=3)
    status = Column(SQLEnum("pending", "in_progress", "completed", "failed", name="debate_status"), 
                   nullable=False, default="pending")
    
    # Results
    final_verdict = Column(SQLEnum("challenger_wins", "defender_wins", "inconclusive", name="judge_verdict"), nullable=True)
    consensus_confidence = Column(Float, nullable=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_duration_seconds = Column(Float, nullable=True)
    
    # Configuration and metadata
    debate_config = Column(JSON, nullable=False, default=dict)
    escalation_threshold = Column(Float, nullable=False, default=0.5)
    debate_metadata = Column(JSON, nullable=False, default=dict)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    challenger_model = relationship("DBModelAssignment", foreign_keys=[challenger_model_id], back_populates="challenger_debates")
    defender_model = relationship("DBModelAssignment", foreign_keys=[defender_model_id], back_populates="defender_debates")
    arguments = relationship("DBDebateArgument", back_populates="debate_round", cascade="all, delete-orphan")
    judge_scores = relationship("DBJudgeScore", back_populates="debate_round", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(round_number >= 1, name='check_round_number_positive'),
        CheckConstraint(max_rounds >= 1, name='check_max_rounds_positive'),
        CheckConstraint(max_rounds <= 10, name='check_max_rounds_limit'),
        CheckConstraint(escalation_threshold >= 0.0, name='check_escalation_threshold_min'),
        CheckConstraint(escalation_threshold <= 1.0, name='check_escalation_threshold_max'),
        CheckConstraint(consensus_confidence.is_(None) | 
                       ((consensus_confidence >= 0.0) & (consensus_confidence <= 1.0)), 
                       name='check_consensus_confidence_range'),
        Index('idx_debate_rounds_task_subject', 'verification_task_id', 'subject_id'),
        Index('idx_debate_rounds_status_created', 'status', 'created_at'),
    )


class DBDebateArgument(Base):
    """Database model for storing individual debate arguments."""
    __tablename__ = "debate_arguments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    argument_id = Column(String, nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    debate_round_id = Column(String, ForeignKey("debate_rounds.id"), nullable=False)
    
    role = Column(SQLEnum("challenger", "defender", "judge", name="acvf_role"), nullable=False)
    content = Column(Text, nullable=False)
    round_number = Column(Integer, nullable=False)
    
    # Optional metadata
    references = Column(JSON, nullable=False, default=list)  # List of reference IDs
    confidence_score = Column(Float, nullable=True)
    argument_metadata = Column(JSON, nullable=False, default=dict)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    debate_round = relationship("DBDebateRound", back_populates="arguments")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(round_number >= 1, name='check_argument_round_positive'),
        CheckConstraint(confidence_score.is_(None) | 
                       ((confidence_score >= 0.0) & (confidence_score <= 1.0)), 
                       name='check_argument_confidence_range'),
        Index('idx_debate_arguments_round_role', 'debate_round_id', 'role'),
        Index('idx_debate_arguments_timestamp', 'timestamp'),
    )


class DBJudgeScore(Base):
    """Database model for storing judge scores and verdicts."""
    __tablename__ = "judge_scores"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    score_id = Column(String, nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    debate_round_id = Column(String, ForeignKey("debate_rounds.id"), nullable=False)
    judge_model_id = Column(String, ForeignKey("model_assignments.id"), nullable=False)
    
    # Scoring
    challenger_score = Column(Float, nullable=False)
    defender_score = Column(Float, nullable=False)
    confidence_level = Column(SQLEnum("very_low", "low", "medium", "high", "very_high", name="confidence_level"), 
                             nullable=False)
    verdict = Column(SQLEnum("challenger_wins", "defender_wins", "inconclusive", name="judge_verdict"), nullable=False)
    
    # Detailed feedback
    reasoning = Column(Text, nullable=False)
    key_points = Column(JSON, nullable=False, default=list)  # List of key reasoning points
    evidence_assessment = Column(JSON, nullable=False, default=dict)
    
    # Quality metrics
    argument_quality_challenger = Column(Float, nullable=True)
    argument_quality_defender = Column(Float, nullable=True)
    evidence_strength_challenger = Column(Float, nullable=True)
    evidence_strength_defender = Column(Float, nullable=True)
    logical_consistency_challenger = Column(Float, nullable=True)
    logical_consistency_defender = Column(Float, nullable=True)
    
    # Processing metadata
    processing_time_seconds = Column(Float, nullable=True)
    response_metadata = Column(JSON, nullable=False, default=dict)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    debate_round = relationship("DBDebateRound", back_populates="judge_scores")
    judge_model = relationship("DBModelAssignment", back_populates="judge_scores")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(challenger_score >= 0.0, name='check_challenger_score_min'),
        CheckConstraint(challenger_score <= 1.0, name='check_challenger_score_max'),
        CheckConstraint(defender_score >= 0.0, name='check_defender_score_min'),
        CheckConstraint(defender_score <= 1.0, name='check_defender_score_max'),
        CheckConstraint(argument_quality_challenger.is_(None) | 
                       ((argument_quality_challenger >= 0.0) & (argument_quality_challenger <= 1.0)), 
                       name='check_arg_quality_challenger_range'),
        CheckConstraint(argument_quality_defender.is_(None) | 
                       ((argument_quality_defender >= 0.0) & (argument_quality_defender <= 1.0)), 
                       name='check_arg_quality_defender_range'),
        CheckConstraint(evidence_strength_challenger.is_(None) | 
                       ((evidence_strength_challenger >= 0.0) & (evidence_strength_challenger <= 1.0)), 
                       name='check_evidence_challenger_range'),
        CheckConstraint(evidence_strength_defender.is_(None) | 
                       ((evidence_strength_defender >= 0.0) & (evidence_strength_defender <= 1.0)), 
                       name='check_evidence_defender_range'),
        CheckConstraint(logical_consistency_challenger.is_(None) | 
                       ((logical_consistency_challenger >= 0.0) & (logical_consistency_challenger <= 1.0)), 
                       name='check_logic_challenger_range'),
        CheckConstraint(logical_consistency_defender.is_(None) | 
                       ((logical_consistency_defender >= 0.0) & (logical_consistency_defender <= 1.0)), 
                       name='check_logic_defender_range'),
        Index('idx_judge_scores_round_model', 'debate_round_id', 'judge_model_id'),
        Index('idx_judge_scores_verdict_confidence', 'verdict', 'confidence_level'),
    )


class DBACVFSession(Base):
    """Database model for storing ACVF session results."""
    __tablename__ = "acvf_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    verification_task_id = Column(String, nullable=False, index=True)
    subject_type = Column(String, nullable=False)
    subject_id = Column(String, nullable=False, index=True)
    
    # Session results
    final_verdict = Column(SQLEnum("challenger_wins", "defender_wins", "inconclusive", name="judge_verdict"), nullable=True)
    final_confidence = Column(Float, nullable=True)
    escalated = Column(Boolean, nullable=False, default=False)
    consensus_achieved = Column(Boolean, nullable=False, default=False)
    
    # Statistics
    total_rounds = Column(Integer, nullable=False, default=0)
    total_arguments = Column(Integer, nullable=False, default=0)
    average_judge_confidence = Column(Float, nullable=True)
    
    # Configuration
    acvf_config_id = Column(String, nullable=False)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_duration_seconds = Column(Float, nullable=True)
    
    # Metadata
    session_metadata = Column(JSON, nullable=False, default=dict)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(final_confidence.is_(None) | 
                       ((final_confidence >= 0.0) & (final_confidence <= 1.0)), 
                       name='check_final_confidence_range'),
        CheckConstraint(total_rounds >= 0, name='check_total_rounds_positive'),
        CheckConstraint(total_arguments >= 0, name='check_total_arguments_positive'),
        CheckConstraint(average_judge_confidence.is_(None) | 
                       ((average_judge_confidence >= 0.0) & (average_judge_confidence <= 1.0)), 
                       name='check_avg_confidence_range'),
        Index('idx_acvf_sessions_task_subject', 'verification_task_id', 'subject_id'),
        Index('idx_acvf_sessions_verdict_confidence', 'final_verdict', 'final_confidence'),
    )


# Database initialization
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables (use with caution!)."""
    Base.metadata.drop_all(bind=engine)


# Database utility functions
class DatabaseManager:
    """Database management utilities."""
    
    @staticmethod
    def get_session() -> Session:
        """Get a new database session."""
        return SessionLocal()
    
    @staticmethod
    def close_session(session: Session):
        """Close a database session."""
        session.close()
    
    @staticmethod
    def commit_and_close(session: Session):
        """Commit changes and close session."""
        try:
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @staticmethod
    def rollback_and_close(session: Session):
        """Rollback changes and close session."""
        session.rollback()
        session.close()
    
    @staticmethod
    def initialize_database():
        """Initialize the database with tables."""
        create_tables()
        print(f"Database initialized at: {DATABASE_URL}")
    
    @staticmethod
    def reset_database():
        """Reset the database (drop and recreate all tables)."""
        drop_tables()
        create_tables()
        print(f"Database reset at: {DATABASE_URL}")


# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions with automatic cleanup."""
    
    def __init__(self):
        self.session = None
    
    def __enter__(self) -> Session:
        self.session = DatabaseManager.get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # No exception, commit the transaction
            try:
                self.session.commit()
            except Exception:
                self.session.rollback()
                raise
        else:
            # Exception occurred, rollback
            self.session.rollback()
        
        self.session.close()


if __name__ == "__main__":
    # Initialize database when run as script
    DatabaseManager.initialize_database() 