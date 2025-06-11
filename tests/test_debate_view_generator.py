"""
Unit tests for the debate view output generator.

This module tests the DebateViewGenerator class and related functionality
for creating structured, navigable debate views from ACVF results.
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from src.models.debate_view_generator import (
    DebateViewGenerator, DebateViewFormat, ParticipantStyle, 
    SectionType, DebateTheme, DebateSection, ThreadedConversation, 
    DebateNavigation
)
from src.models.acvf import (
    ACVFResult, DebateRound, DebateArgument, JudgeScore, ModelAssignment,
    ACVFRole, JudgeVerdict, ConfidenceLevel, DebateStatus
)
from src.models.output import DebateViewOutput, DebateEntryOutput
from src.models.document import ParsedDocument, DocumentSection


class TestDebateViewGenerator:
    """Test cases for DebateViewGenerator."""
    
    @pytest.fixture
    def sample_model_assignments(self):
        """Create sample model assignments."""
        challenger = ModelAssignment(
            model_id="gpt-4",
            provider="openai",
            role=ACVFRole.CHALLENGER,
            temperature=0.8,
            max_tokens=2000
        )
        
        defender = ModelAssignment(
            model_id="claude-3-opus",
            provider="anthropic", 
            role=ACVFRole.DEFENDER,
            temperature=0.7,
            max_tokens=2000
        )
        
        judge = ModelAssignment(
            model_id="gpt-4-turbo",
            provider="openai",
            role=ACVFRole.JUDGE,
            temperature=0.3,
            max_tokens=1500
        )
        
        return challenger, defender, judge
    
    @pytest.fixture
    def sample_debate_arguments(self):
        """Create sample debate arguments."""
        challenger_arg = DebateArgument(
            role=ACVFRole.CHALLENGER,
            content="The claim about climate change impacts is overstated. The data shows inconsistencies.",
            round_number=1,
            references=["Figure 2", "Table 1"],
            confidence_score=0.8
        )
        
        defender_arg = DebateArgument(
            role=ACVFRole.DEFENDER,
            content="The challenger's analysis is flawed. The methodology accounts for these variations.",
            round_number=1,
            references=["Section 3.2", "Appendix A"],
            confidence_score=0.9
        )
        
        return challenger_arg, defender_arg
    
    @pytest.fixture
    def sample_judge_score(self):
        """Create a sample judge score."""
        return JudgeScore(
            judge_id="gpt-4-turbo",
            verdict=JudgeVerdict.DEFENDER_WINS,
            confidence=0.85,
            challenger_score=0.7,
            defender_score=0.9,
            reasoning="The defender provided more comprehensive evidence and addressed the challenger's concerns effectively.",
            key_points_challenger=["Data inconsistencies noted", "Methodology questioned"],
            key_points_defender=["Addressed methodology concerns", "Provided additional context"],
            critical_weaknesses=["Challenger's analysis was superficial"]
        )
    
    @pytest.fixture
    def sample_debate_round(self, sample_model_assignments, sample_debate_arguments, sample_judge_score):
        """Create a sample debate round."""
        challenger, defender, judge = sample_model_assignments
        challenger_arg, defender_arg = sample_debate_arguments
        
        round_data = DebateRound(
            verification_task_id="task_123",
            subject_type="claim",
            subject_id="claim_456",
            subject_content="Climate change will cause significant economic disruption by 2030.",
            challenger_model=challenger,
            defender_model=defender,
            judge_models=[judge],
            status=DebateStatus.COMPLETED,
            round_number=1,
            max_rounds=3,
            final_verdict=JudgeVerdict.DEFENDER_WINS,
            consensus_confidence=0.85
        )
        
        round_data.arguments = [challenger_arg, defender_arg]
        round_data.judge_scores = [sample_judge_score]
        
        return round_data
    
    @pytest.fixture
    def sample_acvf_result(self, sample_debate_round):
        """Create a sample ACVF result."""
        result = ACVFResult(
            verification_task_id="task_123",
            subject_type="claim",
            subject_id="claim_456",
            final_verdict=JudgeVerdict.DEFENDER_WINS,
            final_confidence=0.85,
            consensus_achieved=True,
            total_rounds=1,
            total_arguments=2,
            acvf_config_id="config_789"
        )
        
        result.debate_rounds = [sample_debate_round]
        
        return result
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample parsed document."""
        section1 = DocumentSection(
            section_id="sec_1",
            title="Introduction",
            content="This document discusses climate change impacts...",
            section_type="introduction",
            level=1,
            order_index=0
        )
        
        section2 = DocumentSection(
            section_id="sec_2", 
            title="Methodology",
            content="Our analysis methodology includes...",
            section_type="methodology",
            level=1,
            order_index=1
        )
        
        document = ParsedDocument(
            document_id="doc_123",
            filename="climate_report.pdf",
            content_type="application/pdf",
            raw_content="Full document content...",
            sections=[section1, section2]
        )
        
        return document
    
    @pytest.fixture
    def generator(self):
        """Create a DebateViewGenerator instance."""
        return DebateViewGenerator()
    
    @pytest.fixture
    def custom_theme_generator(self):
        """Create a generator with custom theme."""
        theme = DebateTheme(
            challenger_color="#FF0000",
            defender_color="#0000FF", 
            judge_color="#808080"
        )
        return DebateViewGenerator(theme=theme)
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.theme is not None
        assert generator.section_counter == 0
        assert isinstance(generator.theme, DebateTheme)
    
    def test_custom_theme_initialization(self, custom_theme_generator):
        """Test generator with custom theme."""
        assert custom_theme_generator.theme.challenger_color == "#FF0000"
        assert custom_theme_generator.theme.defender_color == "#0000FF"
        assert custom_theme_generator.theme.judge_color == "#808080"
    
    def test_generate_threaded_debate_view(self, generator, sample_acvf_result, sample_document):
        """Test generating a threaded debate view."""
        debate_view = generator.generate_debate_view(
            sample_acvf_result,
            document=sample_document,
            format_type=DebateViewFormat.THREADED,
            config={"show_evidence": True, "show_reasoning": True}
        )
        
        assert isinstance(debate_view, DebateViewOutput)
        assert debate_view.session_id == sample_acvf_result.session_id
        assert debate_view.threaded_view is True
        assert len(debate_view.entries) > 0
        assert len(debate_view.rounds) == 1
        
        # Check that entries include both arguments and verdict summary
        entry_roles = [entry.participant_role for entry in debate_view.entries]
        assert "challenger" in entry_roles
        assert "defender" in entry_roles
    
    def test_generate_chronological_debate_view(self, generator, sample_acvf_result):
        """Test generating a chronological debate view."""
        debate_view = generator.generate_debate_view(
            sample_acvf_result,
            format_type=DebateViewFormat.CHRONOLOGICAL
        )
        
        assert debate_view.threaded_view is False
        assert len(debate_view.entries) >= 2  # At least challenger and defender
        
        # Check that entries are sorted by timestamp
        timestamps = [entry.timestamp for entry in debate_view.entries]
        assert timestamps == sorted(timestamps)
    
    def test_generate_structured_debate_view(self, generator, sample_acvf_result):
        """Test generating a structured debate view."""
        debate_view = generator.generate_debate_view(
            sample_acvf_result,
            format_type=DebateViewFormat.STRUCTURED
        )
        
        assert debate_view.threaded_view is False
        
        # Should include round header
        entry_roles = [entry.participant_role for entry in debate_view.entries]
        assert "system" in entry_roles  # Round header
        assert "challenger" in entry_roles
        assert "defender" in entry_roles
        assert "judge" in entry_roles
    
    def test_generate_summary_debate_view(self, generator, sample_acvf_result):
        """Test generating a summary debate view."""
        debate_view = generator.generate_debate_view(
            sample_acvf_result,
            format_type=DebateViewFormat.SUMMARY
        )
        
        assert debate_view.threaded_view is False
        
        # Should include summary and verdict entries
        entry_roles = [entry.participant_role for entry in debate_view.entries]
        assert "system" in entry_roles
        
        # Check that at least one entry is a summary
        summaries = [entry for entry in debate_view.entries if "summary" in entry.position.lower()]
        assert len(summaries) > 0
    
    def test_create_debate_entry_from_argument(self, generator, sample_debate_round, sample_document):
        """Test creating debate entry from argument."""
        argument = sample_debate_round.arguments[0]  # Challenger argument
        config = {"show_evidence": True, "show_reasoning": True}
        
        entry = generator._create_debate_entry_from_argument(
            argument, sample_debate_round, sample_document, config
        )
        
        assert isinstance(entry, DebateEntryOutput)
        assert entry.participant_role == "challenger"
        assert entry.participant_model == sample_debate_round.challenger_model.model_id
        assert entry.argument_text == argument.content
        assert entry.confidence_score == argument.confidence_score
        assert len(entry.evidence) == len(argument.references)
        assert entry.formatted_content is not None
        assert entry.summary is not None
    
    def test_create_judge_entry_from_score(self, generator, sample_debate_round, sample_judge_score):
        """Test creating judge entry from score."""
        config = {"show_confidence": True}
        
        entry = generator._create_judge_entry_from_score(
            sample_judge_score, sample_debate_round, config
        )
        
        assert isinstance(entry, DebateEntryOutput)
        assert entry.participant_role == "judge"
        assert entry.participant_model == sample_judge_score.judge_id
        assert "verdict" in entry.position.lower()
        assert entry.confidence_score == sample_judge_score.confidence
        assert entry.strength_score is not None
        assert entry.formatted_content is not None
    
    def test_format_argument_content(self, generator, sample_debate_arguments):
        """Test argument content formatting."""
        argument = sample_debate_arguments[0]  # Challenger
        position = "Climate data is inconsistent"
        reasoning = ["Data gaps noted", "Methodology questioned"]
        config = {"show_evidence": True, "show_reasoning": True}
        
        formatted = generator._format_argument_content(argument, position, reasoning, config)
        
        assert "challenger" in formatted.lower()
        assert position in formatted
        assert "reasoning" in formatted.lower()
        assert "evidence" in formatted.lower()
        assert generator.theme.challenger_color in formatted
    
    def test_format_judge_verdict(self, generator, sample_judge_score):
        """Test judge verdict formatting."""
        config = {}
        
        formatted = generator._format_judge_verdict(sample_judge_score, config)
        
        assert "judge verdict" in formatted.lower()
        assert sample_judge_score.verdict.value in formatted
        assert str(sample_judge_score.confidence) in formatted
        assert "score breakdown" in formatted.lower()
        assert sample_judge_score.reasoning in formatted
        assert generator.theme.verdict_color in formatted
    
    def test_parse_argument_content(self, generator):
        """Test parsing argument content."""
        content = """The main claim is incorrect.
        
        First, the data is incomplete.
        Second, the methodology is flawed.
        Third, the conclusions are overstated."""
        
        position, reasoning = generator._parse_argument_content(content)
        
        assert "main claim is incorrect" in position.lower()
        assert len(reasoning) >= 3
        assert "data is incomplete" in reasoning[0]
        assert "methodology is flawed" in reasoning[1]
    
    def test_find_document_links(self, generator, sample_document):
        """Test finding document links from references."""
        references = ["Introduction", "methodology", "section 2"]
        
        links = generator._find_document_links(references, sample_document)
        
        assert len(links) > 0
        # Should find links to sections with matching titles/content
        section_ids = [section.section_id for section in sample_document.sections]
        for link in links:
            assert link in section_ids
    
    def test_group_arguments_by_thread(self, generator, sample_debate_arguments):
        """Test grouping arguments into threads."""
        # Create more arguments for better testing
        arg1, arg2 = sample_debate_arguments
        arg3 = DebateArgument(
            role=ACVFRole.CHALLENGER,
            content="Follow-up point",
            round_number=1,
            timestamp=arg2.timestamp + timedelta(seconds=30)
        )
        
        arguments = [arg1, arg2, arg3]
        threads = generator._group_arguments_by_thread(arguments)
        
        assert len(threads) > 0
        assert all(isinstance(thread, list) for thread in threads)
        assert all(len(thread) <= 3 for thread in threads)  # Max 3 args per thread
    
    def test_generate_argument_summary(self, generator):
        """Test generating argument summaries."""
        content = "This is a detailed argument about climate change. It has multiple sentences explaining the position."
        
        summary = generator._generate_argument_summary(content, max_length=50)
        
        assert len(summary) <= 50
        assert "climate change" in summary.lower()
    
    def test_generate_verdict_summary(self, generator, sample_acvf_result, sample_judge_score):
        """Test generating verdict summary."""
        all_scores = [sample_judge_score]
        
        verdict_summary = generator._generate_verdict_summary(sample_acvf_result, all_scores)
        
        assert "text" in verdict_summary
        assert "summary" in verdict_summary
        assert sample_acvf_result.final_verdict.value in verdict_summary["text"]
        assert str(sample_acvf_result.final_confidence) in verdict_summary["text"]
    
    def test_generate_round_summaries(self, generator, sample_acvf_result):
        """Test generating round summaries."""
        summaries = generator._generate_round_summaries(sample_acvf_result)
        
        assert len(summaries) == len(sample_acvf_result.debate_rounds)
        
        summary = summaries[0]
        assert "round_number" in summary
        assert "arguments_count" in summary
        assert "challenger_arguments" in summary
        assert "defender_arguments" in summary
        assert summary["round_number"] == 1
    
    def test_truncate_content(self, generator):
        """Test content truncation."""
        long_content = "A" * 300
        
        truncated = generator._truncate_content(long_content, max_length=100)
        
        assert len(truncated) <= 103  # 100 + "..."
        assert truncated.endswith("...")
        
        # Test content that doesn't need truncation
        short_content = "Short text"
        result = generator._truncate_content(short_content, max_length=100)
        assert result == short_content
    
    def test_extract_key_arguments(self, generator, sample_acvf_result):
        """Test extracting key arguments for summary."""
        key_arguments = generator._extract_key_arguments(sample_acvf_result)
        
        assert len(key_arguments) > 0
        assert all(isinstance(arg, DebateEntryOutput) for arg in key_arguments)
        
        # Should include key arguments from different roles
        roles = [arg.participant_role for arg in key_arguments]
        assert "challenger" in roles
        assert "defender" in roles
    
    def test_configuration_options(self, generator, sample_acvf_result):
        """Test various configuration options."""
        config = {
            "show_evidence": False,
            "show_reasoning": False,
            "show_confidence": False,
            "subject_content_max_length": 50
        }
        
        debate_view = generator.generate_debate_view(
            sample_acvf_result,
            config=config
        )
        
        assert debate_view.show_evidence is False
        assert debate_view.show_reasoning is False
        assert debate_view.show_confidence is False
        assert len(debate_view.subject_content) <= 53  # 50 + "..."
    
    def test_empty_acvf_result(self, generator):
        """Test handling empty ACVF result."""
        empty_result = ACVFResult(
            verification_task_id="empty_task",
            subject_type="claim",
            subject_id="empty_claim",
            acvf_config_id="config_empty"
        )
        
        debate_view = generator.generate_debate_view(empty_result)
        
        assert isinstance(debate_view, DebateViewOutput)
        assert len(debate_view.entries) == 0
        assert len(debate_view.rounds) == 0
        assert debate_view.total_rounds == 0
        assert debate_view.total_arguments == 0


class TestDebateTheme:
    """Test cases for DebateTheme."""
    
    def test_default_theme(self):
        """Test default theme values."""
        theme = DebateTheme()
        
        assert theme.challenger_color == "#D32F2F"
        assert theme.defender_color == "#1976D2"
        assert theme.judge_color == "#616161"
        assert theme.evidence_color == "#4CAF50"
        assert theme.reasoning_color == "#FF9800"
        assert theme.verdict_color == "#9C27B0"
    
    def test_custom_theme(self):
        """Test custom theme values."""
        theme = DebateTheme(
            challenger_color="#FF0000",
            defender_color="#0000FF",
            background_color="#FFFFFF"
        )
        
        assert theme.challenger_color == "#FF0000"
        assert theme.defender_color == "#0000FF"
        assert theme.background_color == "#FFFFFF"
        # Other colors should remain default
        assert theme.judge_color == "#616161"


class TestDebateSection:
    """Test cases for DebateSection."""
    
    def test_section_creation(self):
        """Test creating a debate section."""
        section = DebateSection(
            section_type=SectionType.ARGUMENT,
            title="Challenger Argument 1",
            content="This is the argument content...",
            participant_style=ParticipantStyle.CHALLENGER,
            collapsible=True
        )
        
        assert section.section_type == SectionType.ARGUMENT
        assert section.title == "Challenger Argument 1"
        assert section.participant_style == ParticipantStyle.CHALLENGER
        assert section.collapsible is True
        assert section.expanded is True  # Default
        assert section.level == 0  # Default
    
    def test_section_with_hierarchy(self):
        """Test section with parent-child hierarchy."""
        parent_section = DebateSection(
            section_type=SectionType.DEBATE_HEADER,
            title="Round 1",
            content="Round 1 header",
            level=0
        )
        
        child_section = DebateSection(
            section_type=SectionType.ARGUMENT,
            title="Challenger Argument",
            content="Argument content",
            level=1,
            parent_section_id=parent_section.section_id
        )
        
        parent_section.child_sections.append(child_section.section_id)
        
        assert child_section.parent_section_id == parent_section.section_id
        assert child_section.section_id in parent_section.child_sections
        assert child_section.level > parent_section.level


class TestThreadedConversation:
    """Test cases for ThreadedConversation."""
    
    def test_thread_creation(self):
        """Test creating a threaded conversation."""
        thread = ThreadedConversation(
            root_argument_id="arg_1",
            arguments=["arg_1", "arg_2", "arg_3"],
            depth=2,
            thread_topic="Climate data validity",
            dominant_participant=ACVFRole.CHALLENGER
        )
        
        assert thread.root_argument_id == "arg_1"
        assert len(thread.arguments) == 3
        assert thread.depth == 2
        assert thread.thread_topic == "Climate data validity"
        assert thread.dominant_participant == ACVFRole.CHALLENGER


class TestDebateNavigation:
    """Test cases for DebateNavigation."""
    
    def test_navigation_creation(self):
        """Test creating debate navigation."""
        nav = DebateNavigation(
            total_rounds=3,
            current_round=2,
            participants=[
                {"role": "challenger", "model": "gpt-4"},
                {"role": "defender", "model": "claude-3-opus"}
            ],
            key_moments=[
                {"type": "verdict", "description": "Judge ruled in favor of defender"}
            ]
        )
        
        assert nav.total_rounds == 3
        assert nav.current_round == 2
        assert len(nav.participants) == 2
        assert len(nav.key_moments) == 1
        assert nav.participants[0]["role"] == "challenger"


class TestEnums:
    """Test cases for enum classes."""
    
    def test_debate_view_format_enum(self):
        """Test DebateViewFormat enum."""
        assert DebateViewFormat.THREADED == "threaded"
        assert DebateViewFormat.CHRONOLOGICAL == "chronological"
        assert DebateViewFormat.STRUCTURED == "structured"
        assert DebateViewFormat.SUMMARY == "summary"
    
    def test_participant_style_enum(self):
        """Test ParticipantStyle enum."""
        assert ParticipantStyle.CHALLENGER == "challenger"
        assert ParticipantStyle.DEFENDER == "defender"
        assert ParticipantStyle.JUDGE == "judge"
    
    def test_section_type_enum(self):
        """Test SectionType enum."""
        assert SectionType.DEBATE_HEADER == "debate_header"
        assert SectionType.ARGUMENT == "argument"
        assert SectionType.VERDICT_SUMMARY == "verdict_summary"
        assert SectionType.NAVIGATION == "navigation"


if __name__ == "__main__":
    pytest.main([__file__]) 