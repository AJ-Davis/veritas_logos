"""
Tests for ACVF database functionality.

This module tests the database persistence layer for ACVF including
the repository pattern and SQLAlchemy models.
"""

import pytest
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

# Set up test database
TEST_DB_URL = "sqlite:///:memory:"
os.environ["DATABASE_URL"] = TEST_DB_URL

from src.models.database import DatabaseManager, create_tables, drop_tables
from src.models.acvf import (
    ACVFRole, DebateStatus, JudgeVerdict, ConfidenceLevel,
    ModelAssignment, DebateArgument, JudgeScore, DebateRound,
    ACVFConfiguration, ACVFResult
)
from src.verification.acvf_repository import ACVFRepository
from src.verification.acvf_controller import ACVFController
from src.llm.llm_client import LLMClient


@pytest.fixture
def setup_database():
    """Set up test database."""
    # Create tables
    create_tables()
    yield
    # Clean up
    drop_tables()


@pytest.fixture
def repository(setup_database):
    """Create ACVF repository for testing."""
    return ACVFRepository()


@pytest.fixture
def sample_model_assignment():
    """Create a sample model assignment."""
    return ModelAssignment(
        model_id="gpt-4",
        provider="openai",
        role=ACVFRole.CHALLENGER,
        temperature=0.7,
        max_tokens=2000,
        metadata={"test": "data"}
    )


@pytest.fixture
def sample_debate_round():
    """Create a sample debate round."""
    challenger = ModelAssignment(
        model_id="gpt-4", provider="openai", role=ACVFRole.CHALLENGER
    )
    defender = ModelAssignment(
        model_id="claude-3", provider="anthropic", role=ACVFRole.DEFENDER
    )
    judge = ModelAssignment(
        model_id="gpt-4-judge", provider="openai", role=ACVFRole.JUDGE
    )
    
    return DebateRound(
        verification_task_id="test-task-123",
        subject_type="claim",
        subject_id="claim-456",
        subject_content="The earth is round.",
        challenger_model=challenger,
        defender_model=defender,
        judge_models=[judge],
        status=DebateStatus.PENDING
    )


@pytest.fixture
def sample_acvf_config():
    """Create a sample ACVF configuration."""
    challenger = ModelAssignment(
        model_id="gpt-4", provider="openai", role=ACVFRole.CHALLENGER
    )
    defender = ModelAssignment(
        model_id="claude-3", provider="anthropic", role=ACVFRole.DEFENDER
    )
    judge = ModelAssignment(
        model_id="gpt-4-judge", provider="openai", role=ACVFRole.JUDGE
    )
    
    return ACVFConfiguration(
        config_id="test-config",
        name="Test ACVF Configuration",
        challenger_models=[challenger],
        defender_models=[defender],
        judge_models=[judge],
        escalation_threshold=0.5,
        consensus_threshold=0.7
    )


class TestACVFRepository:
    """Test ACVF repository operations."""
    
    def test_save_and_get_model_assignment(self, repository, sample_model_assignment):
        """Test saving and retrieving model assignments."""
        # Save model assignment
        assignment_id = repository.save_model_assignment(sample_model_assignment)
        assert assignment_id is not None
        
        # Retrieve model assignment
        retrieved = repository.get_model_assignment(assignment_id)
        assert retrieved is not None
        assert retrieved.model_id == sample_model_assignment.model_id
        assert retrieved.provider == sample_model_assignment.provider
        assert retrieved.role == sample_model_assignment.role
        assert retrieved.temperature == sample_model_assignment.temperature
        assert retrieved.max_tokens == sample_model_assignment.max_tokens
    
    def test_find_model_assignments_by_role(self, repository):
        """Test finding model assignments by role."""
        # Create multiple model assignments
        challenger1 = ModelAssignment(
            model_id="gpt-4", provider="openai", role=ACVFRole.CHALLENGER
        )
        challenger2 = ModelAssignment(
            model_id="claude-3", provider="anthropic", role=ACVFRole.CHALLENGER
        )
        judge1 = ModelAssignment(
            model_id="gpt-4-judge", provider="openai", role=ACVFRole.JUDGE
        )
        
        # Save them
        repository.save_model_assignment(challenger1)
        repository.save_model_assignment(challenger2)
        repository.save_model_assignment(judge1)
        
        # Find challengers
        challengers = repository.find_model_assignments_by_role(ACVFRole.CHALLENGER)
        assert len(challengers) == 2
        assert all(m.role == ACVFRole.CHALLENGER for m in challengers)
        
        # Find judges
        judges = repository.find_model_assignments_by_role(ACVFRole.JUDGE)
        assert len(judges) == 1
        assert judges[0].role == ACVFRole.JUDGE
    
    def test_save_and_get_debate_round(self, repository, sample_debate_round):
        """Test saving and retrieving debate rounds."""
        # Add some arguments and scores
        sample_debate_round.add_argument(
            role=ACVFRole.CHALLENGER,
            content="I challenge this claim!",
            round_number=1
        )
        sample_debate_round.add_argument(
            role=ACVFRole.DEFENDER,
            content="I defend this claim!",
            round_number=1
        )
        
        # Add judge score
        judge_score = JudgeScore(
            judge_model_id="gpt-4-judge",
            challenger_score=0.6,
            defender_score=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            verdict=JudgeVerdict.DEFENDER_WINS,
            reasoning="The defender provided better evidence."
        )
        sample_debate_round.judge_scores.append(judge_score)
        sample_debate_round.status = DebateStatus.COMPLETED
        
        # Save debate round
        round_id = repository.save_debate_round(sample_debate_round)
        assert round_id is not None
        
        # Retrieve debate round
        retrieved = repository.get_debate_round(sample_debate_round.round_id)
        assert retrieved is not None
        assert retrieved.verification_task_id == sample_debate_round.verification_task_id
        assert retrieved.subject_type == sample_debate_round.subject_type
        assert retrieved.subject_id == sample_debate_round.subject_id
        assert retrieved.subject_content == sample_debate_round.subject_content
        assert len(retrieved.arguments) == 2
        assert len(retrieved.judge_scores) == 1
    
    def test_find_debate_rounds_by_task(self, repository, sample_debate_round):
        """Test finding debate rounds by verification task."""
        # Save debate round
        repository.save_debate_round(sample_debate_round)
        
        # Create another round for same task
        round2 = DebateRound(
            verification_task_id=sample_debate_round.verification_task_id,
            subject_type="citation",
            subject_id="citation-789",
            subject_content="Source: Wikipedia",
            challenger_model=sample_debate_round.challenger_model,
            defender_model=sample_debate_round.defender_model,
            judge_models=sample_debate_round.judge_models
        )
        repository.save_debate_round(round2)
        
        # Find rounds by task
        rounds = repository.find_debate_rounds_by_task(sample_debate_round.verification_task_id)
        assert len(rounds) == 2
        assert all(r.verification_task_id == sample_debate_round.verification_task_id for r in rounds)
    
    def test_update_debate_round_status(self, repository, sample_debate_round):
        """Test updating debate round status."""
        # Save debate round
        repository.save_debate_round(sample_debate_round)
        
        # Update status
        repository.update_debate_round_status(
            sample_debate_round.round_id,
            DebateStatus.COMPLETED,
            JudgeVerdict.CHALLENGER_WINS,
            0.85
        )
        
        # Verify update
        retrieved = repository.get_debate_round(sample_debate_round.round_id)
        assert retrieved.status == DebateStatus.COMPLETED
        assert retrieved.final_verdict == JudgeVerdict.CHALLENGER_WINS
        assert retrieved.consensus_confidence == 0.85
    
    def test_save_and_get_acvf_session(self, repository, sample_debate_round):
        """Test saving and retrieving ACVF sessions."""
        # Create ACVF result
        acvf_result = ACVFResult(
            verification_task_id="test-task-123",
            subject_type="claim",
            subject_id="claim-456",
            acvf_config_id="test-config"
        )
        acvf_result.add_debate_round(sample_debate_round)
        acvf_result.calculate_final_metrics()
        
        # Save session
        session_id = repository.save_acvf_session(acvf_result)
        assert session_id is not None
        
        # Retrieve session
        retrieved = repository.get_acvf_session(acvf_result.session_id)
        assert retrieved is not None
        assert retrieved.verification_task_id == acvf_result.verification_task_id
        assert retrieved.subject_type == acvf_result.subject_type
        assert retrieved.subject_id == acvf_result.subject_id
        assert retrieved.total_rounds == acvf_result.total_rounds


class TestACVFControllerWithDatabase:
    """Test ACVF controller with database integration."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock(spec=LLMClient)
        
        # Mock challenger response
        challenger_response = Mock()
        challenger_response.content = "I challenge this claim because..."
        client.generate_challenger_response = AsyncMock(return_value=challenger_response)
        
        # Mock defender response
        defender_response = Mock()
        defender_response.content = "I defend this claim because..."
        client.generate_defender_response = AsyncMock(return_value=defender_response)
        
        # Mock judge response
        judge_response = Mock()
        judge_response.content = '{"verdict": "defender_wins", "challenger_score": 0.4, "defender_score": 0.8, "confidence_level": "high", "reasoning": "Defender provided better evidence"}'
        client.generate_judge_verdict = AsyncMock(return_value=judge_response)
        
        return client
    
    @pytest.mark.asyncio
    async def test_acvf_controller_database_integration(self, setup_database, mock_llm_client, sample_acvf_config):
        """Test that ACVF controller properly saves data to database."""
        # Create controller with repository
        repository = ACVFRepository()
        controller = ACVFController(
            llm_client=mock_llm_client,
            config=sample_acvf_config,
            repository=repository
        )
        
        # Conduct debate round
        debate_round = await controller.conduct_debate_round(
            verification_task_id="test-task-123",
            subject_type="claim",
            subject_id="claim-456",
            subject_content="The earth is round.",
            context="Testing ACVF database integration"
        )
        
        # Verify debate round was saved to database
        retrieved = repository.get_debate_round(debate_round.round_id)
        assert retrieved is not None
        assert retrieved.verification_task_id == "test-task-123"
        assert retrieved.subject_type == "claim"
        assert retrieved.subject_id == "claim-456"
        assert retrieved.status == DebateStatus.COMPLETED
        
        # Verify arguments were saved
        assert len(retrieved.arguments) >= 2  # At least challenger and defender
        
        # Verify judge scores were saved
        assert len(retrieved.judge_scores) >= 1


class TestDatabaseManager:
    """Test database manager functionality."""
    
    def test_database_initialization(self):
        """Test database initialization."""
        # This should not raise an exception
        db_manager = DatabaseManager()
        db_manager.initialize_database()
        
        # Verify we can create a session
        session = db_manager.get_session()
        assert session is not None
        db_manager.close_session(session)
    
    def test_database_session_context_manager(self, setup_database):
        """Test database session context manager."""
        from src.models.database import DatabaseSession, DBModelAssignment
        
        # Test successful transaction
        with DatabaseSession() as session:
            assignment = DBModelAssignment(
                model_id="test-model",
                provider="test-provider",
                role="challenger"
            )
            session.add(assignment)
            # Session should commit automatically
        
        # Verify data was saved
        with DatabaseSession() as session:
            saved_assignment = session.query(DBModelAssignment).filter(
                DBModelAssignment.model_id == "test-model"
            ).first()
            assert saved_assignment is not None
    
    def test_database_session_rollback_on_exception(self, setup_database):
        """Test that database session rolls back on exception."""
        from src.models.database import DatabaseSession, DBModelAssignment
        
        # Test failed transaction
        try:
            with DatabaseSession() as session:
                assignment = DBModelAssignment(
                    model_id="test-model-fail",
                    provider="test-provider",
                    role="challenger"
                )
                session.add(assignment)
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected
        
        # Verify data was not saved
        with DatabaseSession() as session:
            saved_assignment = session.query(DBModelAssignment).filter(
                DBModelAssignment.model_id == "test-model-fail"
            ).first()
            assert saved_assignment is None


if __name__ == "__main__":
    pytest.main([__file__]) 