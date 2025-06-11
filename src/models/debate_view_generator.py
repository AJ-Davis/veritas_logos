"""
Debate View Output Generator for the Veritas Logos verification system.

This module implements the specialized output format that shows the ACVF debate 
process with Challenger critiques, Defender rebuttals, and Judge rulings in 
a structured, navigable format.

Key Features:
- Threaded conversation view structure for debates
- Formatting for different debate participants (Challenger, Defender, Judge)
- Collapsible/expandable sections for detailed arguments
- Evidence linking between debate points and document sections
- Verdict highlighting and summary sections
- Integration with document annotation system
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from src.models.acvf import (
    ACVFResult, DebateRound, DebateArgument, JudgeScore, 
    ACVFRole, JudgeVerdict, ConfidenceLevel, DebateStatus
)
from src.models.output import (
    DebateViewOutput, DebateEntryOutput, OutputFormat,
    DocumentAnnotation, AnnotationType, HighlightStyle
)
from src.models.verification import VerificationChainResult
from src.models.document import DocumentSection, ParsedDocument


class DebateViewFormat(str, Enum):
    """Output formats for debate views."""
    THREADED = "threaded"        # Conversation-style threading
    CHRONOLOGICAL = "chronological"  # Time-ordered linear view
    STRUCTURED = "structured"    # Formal debate structure
    SUMMARY = "summary"          # Condensed overview


class ParticipantStyle(str, Enum):
    """Visual styling for different participants."""
    CHALLENGER = "challenger"    # Red/aggressive styling
    DEFENDER = "defender"        # Blue/defensive styling
    JUDGE = "judge"             # Gray/neutral styling


class SectionType(str, Enum):
    """Types of sections in the debate view."""
    DEBATE_HEADER = "debate_header"
    PARTICIPANT_INFO = "participant_info"
    ROUND_SUMMARY = "round_summary"
    ARGUMENT = "argument"
    EVIDENCE = "evidence"
    REASONING = "reasoning"
    JUDGE_ANALYSIS = "judge_analysis"
    VERDICT_SUMMARY = "verdict_summary"
    NAVIGATION = "navigation"


@dataclass
class DebateTheme:
    """Styling theme for debate views."""
    challenger_color: str = "#D32F2F"      # Red
    defender_color: str = "#1976D2"        # Blue
    judge_color: str = "#616161"           # Gray
    evidence_color: str = "#4CAF50"        # Green
    reasoning_color: str = "#FF9800"       # Orange
    verdict_color: str = "#9C27B0"         # Purple
    background_color: str = "#FAFAFA"      # Light gray
    text_color: str = "#212121"            # Dark gray
    border_color: str = "#E0E0E0"          # Light gray border


class DebateSection(BaseModel):
    """A section within the debate view."""
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section_type: SectionType
    title: str
    content: str
    
    # Hierarchy and navigation
    level: int = Field(default=0, ge=0, le=5)
    parent_section_id: Optional[str] = None
    child_sections: List[str] = Field(default_factory=list)
    
    # Interactivity
    collapsible: bool = Field(default=False)
    expanded: bool = Field(default=True)
    
    # Styling
    participant_style: Optional[ParticipantStyle] = None
    custom_style: Dict[str, str] = Field(default_factory=dict)
    
    # Links and references
    document_links: List[str] = Field(default_factory=list)
    evidence_links: List[str] = Field(default_factory=list)
    related_arguments: List[str] = Field(default_factory=list)
    
    # Metadata
    timestamp: Optional[datetime] = None
    participant_id: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThreadedConversation(BaseModel):
    """Represents a threaded conversation structure."""
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    root_argument_id: str
    arguments: List[str] = Field(default_factory=list)  # IDs in thread order
    depth: int = Field(default=0, ge=0)
    
    # Thread metadata
    thread_topic: Optional[str] = None
    dominant_participant: Optional[ACVFRole] = None
    resolution_status: Optional[str] = None


class DebateNavigation(BaseModel):
    """Navigation structure for complex debates."""
    nav_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Round navigation
    total_rounds: int = Field(default=0, ge=0)
    current_round: int = Field(default=1, ge=1)
    
    # Participant navigation
    participants: List[Dict[str, str]] = Field(default_factory=list)
    
    # Topic navigation
    topics: List[Dict[str, str]] = Field(default_factory=list)
    
    # Verdict navigation
    verdicts_by_round: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Quick links
    key_moments: List[Dict[str, str]] = Field(default_factory=list)
    evidence_summary: List[Dict[str, str]] = Field(default_factory=list)


class DebateViewGenerator:
    """
    Main generator class for creating debate view outputs.
    
    Transforms ACVF results into structured, navigable debate views
    with proper formatting, threading, and interactivity.
    """
    
    def __init__(self, theme: Optional[DebateTheme] = None):
        """Initialize the generator with optional custom theme."""
        self.theme = theme or DebateTheme()
        self.section_counter = 0
    
    def generate_debate_view(self, 
                           acvf_result: ACVFResult,
                           document: Optional[ParsedDocument] = None,
                           format_type: DebateViewFormat = DebateViewFormat.THREADED,
                           config: Optional[Dict[str, Any]] = None) -> DebateViewOutput:
        """
        Generate a complete debate view from an ACVF result.
        
        Args:
            acvf_result: The ACVF result containing debate data
            document: Optional source document for linking
            format_type: The output format type
            config: Additional configuration options
            
        Returns:
            A structured DebateViewOutput object
        """
        config = config or {}
        
        # Create base debate view
        debate_view = DebateViewOutput(
            session_id=acvf_result.session_id,
            subject_type=acvf_result.subject_type,
            subject_id=acvf_result.subject_id,
            subject_content=self._truncate_content(
                getattr(acvf_result.debate_rounds[0], 'subject_content', '') if acvf_result.debate_rounds else '',
                max_length=config.get('subject_content_max_length', 200)
            ),
            final_verdict=acvf_result.final_verdict,
            final_confidence=acvf_result.final_confidence,
            consensus_achieved=acvf_result.consensus_achieved,
            total_rounds=acvf_result.total_rounds,
            total_arguments=acvf_result.total_arguments,
            debate_duration_seconds=acvf_result.total_duration_seconds,
            show_evidence=config.get('show_evidence', True),
            show_reasoning=config.get('show_reasoning', True),
            show_confidence=config.get('show_confidence', True),
            threaded_view=format_type == DebateViewFormat.THREADED
        )
        
        # Generate debate entries based on format
        if format_type == DebateViewFormat.THREADED:
            debate_view.entries = self._generate_threaded_entries(acvf_result, document, config)
        elif format_type == DebateViewFormat.CHRONOLOGICAL:
            debate_view.entries = self._generate_chronological_entries(acvf_result, document, config)
        elif format_type == DebateViewFormat.STRUCTURED:
            debate_view.entries = self._generate_structured_entries(acvf_result, document, config)
        else:  # SUMMARY
            debate_view.entries = self._generate_summary_entries(acvf_result, document, config)
        
        # Generate round summaries
        debate_view.rounds = self._generate_round_summaries(acvf_result)
        
        return debate_view
    
    def _generate_threaded_entries(self, 
                                 acvf_result: ACVFResult,
                                 document: Optional[ParsedDocument],
                                 config: Dict[str, Any]) -> List[DebateEntryOutput]:
        """Generate entries in threaded conversation format."""
        entries = []
        
        # Create threads for each debate round
        for round_data in acvf_result.debate_rounds:
            thread_entries = self._create_round_thread(round_data, document, config)
            entries.extend(thread_entries)
        
        # Add judge summary entries
        if acvf_result.final_verdict:
            summary_entry = self._create_verdict_summary_entry(acvf_result, config)
            entries.append(summary_entry)
        
        return entries
    
    def _generate_chronological_entries(self,
                                      acvf_result: ACVFResult,
                                      document: Optional[ParsedDocument],
                                      config: Dict[str, Any]) -> List[DebateEntryOutput]:
        """Generate entries in chronological order."""
        all_arguments = []
        
        # Collect all arguments from all rounds
        for round_data in acvf_result.debate_rounds:
            for argument in round_data.arguments:
                entry = self._create_debate_entry_from_argument(
                    argument, round_data, document, config
                )
                all_arguments.append(entry)
        
        # Sort by timestamp
        all_arguments.sort(key=lambda x: x.timestamp)
        
        return all_arguments
    
    def _generate_structured_entries(self,
                                   acvf_result: ACVFResult,
                                   document: Optional[ParsedDocument],
                                   config: Dict[str, Any]) -> List[DebateEntryOutput]:
        """Generate entries in formal debate structure."""
        entries = []
        
        for round_idx, round_data in enumerate(acvf_result.debate_rounds):
            # Round header
            round_header = self._create_round_header_entry(round_data, round_idx + 1)
            entries.append(round_header)
            
            # Challenger arguments
            challenger_args = [arg for arg in round_data.arguments 
                             if arg.role == ACVFRole.CHALLENGER]
            if challenger_args:
                entries.extend([
                    self._create_debate_entry_from_argument(arg, round_data, document, config)
                    for arg in challenger_args
                ])
            
            # Defender arguments
            defender_args = [arg for arg in round_data.arguments 
                           if arg.role == ACVFRole.DEFENDER]
            if defender_args:
                entries.extend([
                    self._create_debate_entry_from_argument(arg, round_data, document, config)
                    for arg in defender_args
                ])
            
            # Judge scores for this round
            if round_data.judge_scores:
                judge_entries = [
                    self._create_judge_entry_from_score(score, round_data, config)
                    for score in round_data.judge_scores
                ]
                entries.extend(judge_entries)
        
        return entries
    
    def _generate_summary_entries(self,
                                acvf_result: ACVFResult,
                                document: Optional[ParsedDocument],
                                config: Dict[str, Any]) -> List[DebateEntryOutput]:
        """Generate condensed summary entries."""
        entries = []
        
        # Overall summary
        summary_entry = self._create_debate_summary_entry(acvf_result, config)
        entries.append(summary_entry)
        
        # Key arguments summary
        key_arguments = self._extract_key_arguments(acvf_result)
        for arg_summary in key_arguments:
            entries.append(arg_summary)
        
        # Final verdict
        if acvf_result.final_verdict:
            verdict_entry = self._create_verdict_summary_entry(acvf_result, config)
            entries.append(verdict_entry)
        
        return entries
    
    def _create_round_thread(self,
                           round_data: DebateRound,
                           document: Optional[ParsedDocument],
                           config: Dict[str, Any]) -> List[DebateEntryOutput]:
        """Create a threaded conversation for a single round."""
        entries = []
        
        # Group arguments by their logical flow
        grouped_args = self._group_arguments_by_thread(round_data.arguments)
        
        for thread_group in grouped_args:
            for argument in thread_group:
                entry = self._create_debate_entry_from_argument(
                    argument, round_data, document, config
                )
                # Add thread information
                if len(thread_group) > 1:
                    entry.responds_to = self._find_parent_argument_id(argument, thread_group)
                entries.append(entry)
        
        return entries
    
    def _create_debate_entry_from_argument(self,
                                         argument: DebateArgument,
                                         round_data: DebateRound,
                                         document: Optional[ParsedDocument],
                                         config: Dict[str, Any]) -> DebateEntryOutput:
        """Create a DebateEntryOutput from a DebateArgument."""
        
        # Determine participant model name
        if argument.role == ACVFRole.CHALLENGER:
            model_name = round_data.challenger_model.model_id
        elif argument.role == ACVFRole.DEFENDER:
            model_name = round_data.defender_model.model_id
        else:  # JUDGE
            # Find the judge model (assuming single judge for now)
            model_name = round_data.judge_models[0].model_id if round_data.judge_models else "unknown"
        
        # Extract position and reasoning from content
        position, reasoning = self._parse_argument_content(argument.content)
        
        # Link to document sections if available
        doc_links = []
        if document and argument.references:
            doc_links = self._find_document_links(argument.references, document)
        
        # Create formatted content
        formatted_content = self._format_argument_content(
            argument, position, reasoning, config
        )
        
        # Generate summary
        summary = self._generate_argument_summary(argument.content, max_length=150)
        
        return DebateEntryOutput(
            debate_round_id=round_data.round_id,
            participant_role=argument.role.value,
            participant_model=model_name,
            argument_text=argument.content,
            position=position,
            evidence=argument.references,
            reasoning=reasoning,
            confidence_score=argument.confidence_score,
            timestamp=argument.timestamp,
            summary=summary,
            formatted_content=formatted_content,
            related_document_sections=doc_links
        )
    
    def _create_judge_entry_from_score(self,
                                     score: JudgeScore,
                                     round_data: DebateRound,
                                     config: Dict[str, Any]) -> DebateEntryOutput:
        """Create a DebateEntryOutput from a JudgeScore."""
        
        # Create comprehensive judge analysis content
        analysis_content = self._format_judge_analysis(score, config)
        
        # Extract key points for structured display
        all_key_points = score.key_points_challenger + score.key_points_defender
        reasoning_points = [score.reasoning] + score.critical_weaknesses
        
        return DebateEntryOutput(
            debate_round_id=round_data.round_id,
            participant_role="judge",
            participant_model=score.judge_id,
            argument_text=analysis_content,
            position=f"Verdict: {score.verdict.value}",
            evidence=all_key_points,
            reasoning=reasoning_points,
            confidence_score=score.confidence,
            strength_score=max(score.challenger_score, score.defender_score),
            timestamp=score.timestamp,
            summary=f"Judge verdict: {score.verdict.value} (confidence: {score.confidence:.2f})",
            formatted_content=self._format_judge_verdict(score, config)
        )
    
    def _create_round_header_entry(self,
                                 round_data: DebateRound,
                                 round_number: int) -> DebateEntryOutput:
        """Create a header entry for a debate round."""
        
        header_content = f"""
        <div class="debate-round-header">
            <h3>Round {round_number}</h3>
            <div class="round-info">
                <p><strong>Subject:</strong> {round_data.subject_content[:200]}...</p>
                <p><strong>Status:</strong> {round_data.status.value}</p>
                <p><strong>Arguments:</strong> {len(round_data.arguments)}</p>
            </div>
        </div>
        """
        
        return DebateEntryOutput(
            debate_round_id=round_data.round_id,
            participant_role="system",
            participant_model="system",
            argument_text=f"Round {round_number} - {round_data.subject_type}",
            position="Round Header",
            timestamp=round_data.started_at or datetime.now(timezone.utc),
            summary=f"Round {round_number}: {round_data.subject_type}",
            formatted_content=header_content
        )
    
    def _create_verdict_summary_entry(self,
                                    acvf_result: ACVFResult,
                                    config: Dict[str, Any]) -> DebateEntryOutput:
        """Create a summary entry for the final verdict."""
        
        # Calculate consensus statistics
        all_scores = []
        for round_data in acvf_result.debate_rounds:
            all_scores.extend(round_data.judge_scores)
        
        verdict_summary = self._generate_verdict_summary(acvf_result, all_scores)
        formatted_summary = self._format_verdict_summary(verdict_summary, config)
        
        return DebateEntryOutput(
            debate_round_id="final",
            participant_role="system",
            participant_model="system",
            argument_text=verdict_summary["text"],
            position=f"Final Verdict: {acvf_result.final_verdict.value if acvf_result.final_verdict else 'None'}",
            confidence_score=acvf_result.final_confidence,
            timestamp=datetime.now(timezone.utc),
            summary=verdict_summary["summary"],
            formatted_content=formatted_summary
        )
    
    def _create_debate_summary_entry(self,
                                   acvf_result: ACVFResult,
                                   config: Dict[str, Any]) -> DebateEntryOutput:
        """Create an overall debate summary entry."""
        
        summary_stats = {
            "total_rounds": acvf_result.total_rounds,
            "total_arguments": acvf_result.total_arguments,
            "duration": acvf_result.total_duration_seconds,
            "consensus": acvf_result.consensus_achieved,
            "final_verdict": acvf_result.final_verdict,
            "confidence": acvf_result.final_confidence
        }
        
        summary_text = self._generate_debate_overview(summary_stats)
        formatted_overview = self._format_debate_overview(summary_stats, config)
        
        return DebateEntryOutput(
            debate_round_id="summary",
            participant_role="system",
            participant_model="system",
            argument_text=summary_text,
            position="Debate Overview",
            timestamp=acvf_result.completed_at or datetime.now(timezone.utc),
            summary=f"Debate summary: {acvf_result.total_rounds} rounds, {acvf_result.total_arguments} arguments",
            formatted_content=formatted_overview
        )
    
    def _generate_round_summaries(self, acvf_result: ACVFResult) -> List[Dict[str, Any]]:
        """Generate summary data for each round."""
        summaries = []
        
        for round_idx, round_data in enumerate(acvf_result.debate_rounds):
            summary = {
                "round_number": round_idx + 1,
                "round_id": round_data.round_id,
                "status": round_data.status.value,
                "arguments_count": len(round_data.arguments),
                "challenger_arguments": len([a for a in round_data.arguments if a.role == ACVFRole.CHALLENGER]),
                "defender_arguments": len([a for a in round_data.arguments if a.role == ACVFRole.DEFENDER]),
                "judge_scores_count": len(round_data.judge_scores),
                "duration_seconds": round_data.total_duration_seconds,
                "final_verdict": round_data.final_verdict.value if round_data.final_verdict else None,
                "consensus_confidence": round_data.consensus_confidence
            }
            summaries.append(summary)
        
        return summaries
    
    # Helper methods for content processing and formatting
    
    def _truncate_content(self, content: str, max_length: int = 200) -> str:
        """Truncate content to specified length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length - 3] + "..."
    
    def _parse_argument_content(self, content: str) -> Tuple[str, List[str]]:
        """Parse argument content to extract position and reasoning."""
        lines = content.split('\n')
        
        # Simple heuristic: first substantial line is the position
        position = next((line.strip() for line in lines if len(line.strip()) > 10), content[:100])
        
        # Remaining lines are reasoning
        reasoning = [line.strip() for line in lines[1:] if len(line.strip()) > 5]
        
        return position, reasoning
    
    def _find_document_links(self, references: List[str], document: ParsedDocument) -> List[str]:
        """Find links to document sections based on references."""
        links = []
        
        for ref in references:
            # Simple matching based on section titles or content
            for section in document.sections:
                if ref.lower() in section.title.lower() or ref.lower() in section.content[:200].lower():
                    links.append(section.section_id)
        
        return links
    
    def _format_argument_content(self,
                               argument: DebateArgument,
                               position: str,
                               reasoning: List[str],
                               config: Dict[str, Any]) -> str:
        """Format argument content for display."""
        
        role_colors = {
            ACVFRole.CHALLENGER: self.theme.challenger_color,
            ACVFRole.DEFENDER: self.theme.defender_color,
            ACVFRole.JUDGE: self.theme.judge_color
        }
        
        color = role_colors.get(argument.role, self.theme.text_color)
        
        formatted = f"""
        <div class="debate-argument" style="border-left: 4px solid {color}; padding-left: 15px;">
            <div class="argument-header">
                <span class="participant-role" style="color: {color}; font-weight: bold;">
                    {argument.role.value.title()}
                </span>
                <span class="timestamp">{argument.timestamp.strftime('%H:%M:%S')}</span>
                {f'<span class="confidence">Confidence: {argument.confidence_score:.2f}</span>' if argument.confidence_score else ''}
            </div>
            <div class="argument-position">
                <strong>Position:</strong> {position}
            </div>
        """
        
        if reasoning and config.get('show_reasoning', True):
            formatted += '<div class="argument-reasoning"><strong>Reasoning:</strong><ul>'
            for point in reasoning:
                formatted += f'<li>{point}</li>'
            formatted += '</ul></div>'
        
        if argument.references and config.get('show_evidence', True):
            formatted += '<div class="argument-evidence"><strong>Evidence:</strong><ul>'
            for ref in argument.references:
                formatted += f'<li>{ref}</li>'
            formatted += '</ul></div>'
        
        formatted += '</div>'
        
        return formatted
    
    def _format_judge_analysis(self, score: JudgeScore, config: Dict[str, Any]) -> str:
        """Format judge analysis for display."""
        return f"""
        Judge Analysis:
        
        Verdict: {score.verdict.value}
        Confidence: {score.confidence:.2f} ({score.confidence_level.value})
        
        Scores:
        - Challenger: {score.challenger_score:.2f}
        - Defender: {score.defender_score:.2f}
        
        Reasoning: {score.reasoning}
        
        Key Points - Challenger: {', '.join(score.key_points_challenger)}
        Key Points - Defender: {', '.join(score.key_points_defender)}
        
        Critical Weaknesses: {', '.join(score.critical_weaknesses)}
        """
    
    def _format_judge_verdict(self, score: JudgeScore, config: Dict[str, Any]) -> str:
        """Format judge verdict with styling."""
        verdict_color = self.theme.verdict_color
        
        return f"""
        <div class="judge-verdict" style="border: 2px solid {verdict_color}; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <div class="verdict-header" style="color: {verdict_color}; font-weight: bold; font-size: 1.2em;">
                ⚖️ Judge Verdict: {score.verdict.value.replace('_', ' ').title()}
            </div>
            <div class="confidence-bar" style="margin: 10px 0;">
                <div style="background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                    <div style="background: {verdict_color}; height: 20px; width: {score.confidence * 100}%; 
                         display: flex; align-items: center; justify-content: center; color: white;">
                        {score.confidence:.1%} confidence
                    </div>
                </div>
            </div>
            <div class="score-breakdown">
                <strong>Score Breakdown:</strong>
                <div>Challenger: {score.challenger_score:.2f} | Defender: {score.defender_score:.2f}</div>
            </div>
            <div class="reasoning" style="margin-top: 10px;">
                <strong>Reasoning:</strong> {score.reasoning}
            </div>
        </div>
        """
    
    def _generate_argument_summary(self, content: str, max_length: int = 150) -> str:
        """Generate a brief summary of an argument."""
        # Simple extractive summary - take first sentence or up to max_length
        sentences = content.split('. ')
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0] + '.'
        return self._truncate_content(content, max_length)
    
    def _generate_verdict_summary(self, acvf_result: ACVFResult, all_scores: List[JudgeScore]) -> Dict[str, str]:
        """Generate a comprehensive verdict summary."""
        
        verdict_counts = {}
        total_confidence = 0
        
        for score in all_scores:
            verdict_counts[score.verdict] = verdict_counts.get(score.verdict, 0) + 1
            total_confidence += score.confidence
        
        avg_confidence = total_confidence / len(all_scores) if all_scores else 0
        
        summary_text = f"""
        Final Verdict Analysis:
        
        Overall Verdict: {acvf_result.final_verdict.value if acvf_result.final_verdict else 'No consensus'}
        Final Confidence: {acvf_result.final_confidence:.2f if acvf_result.final_confidence else 'N/A'}
        Consensus Achieved: {'Yes' if acvf_result.consensus_achieved else 'No'}
        
        Verdict Distribution:
        {chr(10).join([f'- {verdict.value}: {count}' for verdict, count in verdict_counts.items()])}
        
        Average Judge Confidence: {avg_confidence:.2f}
        Total Rounds: {acvf_result.total_rounds}
        Total Arguments: {acvf_result.total_arguments}
        """
        
        brief_summary = f"Final verdict: {acvf_result.final_verdict.value if acvf_result.final_verdict else 'No consensus'} " \
                       f"(confidence: {acvf_result.final_confidence:.2f if acvf_result.final_confidence else 'N/A'})"
        
        return {
            "text": summary_text,
            "summary": brief_summary
        }
    
    def _format_verdict_summary(self, verdict_summary: Dict[str, str], config: Dict[str, Any]) -> str:
        """Format verdict summary with styling."""
        return f"""
        <div class="verdict-summary" style="background: {self.theme.background_color}; 
             border-radius: 8px; padding: 20px; border: 1px solid {self.theme.border_color};">
            <h3 style="color: {self.theme.verdict_color}; margin-top: 0;">🏛️ Final Verdict Summary</h3>
            <pre style="white-space: pre-wrap; font-family: {config.get('font_family', 'Arial, sans-serif')};">
{verdict_summary['text']}
            </pre>
        </div>
        """
    
    def _generate_debate_overview(self, stats: Dict[str, Any]) -> str:
        """Generate an overview of the entire debate."""
        duration_str = f"{stats['duration']:.1f} seconds" if stats['duration'] else "Unknown duration"
        
        return f"""
        Debate Overview:
        
        📊 Statistics:
        - Total Rounds: {stats['total_rounds']}
        - Total Arguments: {stats['total_arguments']}
        - Duration: {duration_str}
        - Consensus Achieved: {'✅' if stats['consensus'] else '❌'}
        
        🎯 Final Outcome:
        - Verdict: {stats['final_verdict'].value if stats['final_verdict'] else 'No verdict'}
        - Confidence: {stats['confidence']:.2f if stats['confidence'] else 'N/A'}
        
        This debate examined the specified subject through adversarial cross-validation,
        with challenger and defender models presenting arguments, and judge models
        providing independent evaluation and verdicts.
        """
    
    def _format_debate_overview(self, stats: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Format debate overview with styling."""
        return f"""
        <div class="debate-overview" style="background: linear-gradient(135deg, {self.theme.background_color}, #ffffff); 
             border-radius: 12px; padding: 25px; border: 2px solid {self.theme.border_color}; 
             box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: {self.theme.text_color}; margin-top: 0; text-align: center;">
                🎭 ACVF Debate Analysis
            </h2>
            <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                <div class="stat-item" style="text-align: center; padding: 15px; border-radius: 8px; background: white;">
                    <div style="font-size: 2em; color: {self.theme.challenger_color};">{stats['total_rounds']}</div>
                    <div>Rounds</div>
                </div>
                <div class="stat-item" style="text-align: center; padding: 15px; border-radius: 8px; background: white;">
                    <div style="font-size: 2em; color: {self.theme.defender_color};">{stats['total_arguments']}</div>
                    <div>Arguments</div>
                </div>
                <div class="stat-item" style="text-align: center; padding: 15px; border-radius: 8px; background: white;">
                    <div style="font-size: 2em; color: {self.theme.judge_color};">{'✅' if stats['consensus'] else '❌'}</div>
                    <div>Consensus</div>
                </div>
            </div>
            <div class="final-verdict" style="text-align: center; margin-top: 20px; padding: 15px; 
                 background: {self.theme.verdict_color}; color: white; border-radius: 8px;">
                <strong>Final Verdict: {stats['final_verdict'].value if stats['final_verdict'] else 'No Verdict'}</strong>
                <br>
                <span>Confidence: {stats['confidence']:.1%} if stats['confidence'] else 'N/A'</span>
            </div>
        </div>
        """
    
    def _group_arguments_by_thread(self, arguments: List[DebateArgument]) -> List[List[DebateArgument]]:
        """Group arguments into conversation threads."""
        # Simple grouping by round and role alternation
        # More sophisticated threading could be implemented based on content similarity
        
        threads = []
        current_thread = []
        
        sorted_args = sorted(arguments, key=lambda x: x.timestamp)
        
        for arg in sorted_args:
            if not current_thread:
                current_thread = [arg]
            elif len(current_thread) >= 3:  # Start new thread after 3 arguments
                threads.append(current_thread)
                current_thread = [arg]
            else:
                current_thread.append(arg)
        
        if current_thread:
            threads.append(current_thread)
        
        return threads
    
    def _find_parent_argument_id(self, argument: DebateArgument, thread: List[DebateArgument]) -> Optional[str]:
        """Find the parent argument ID for threading."""
        idx = thread.index(argument)
        if idx > 0:
            return thread[idx - 1].argument_id
        return None
    
    def _extract_key_arguments(self, acvf_result: ACVFResult) -> List[DebateEntryOutput]:
        """Extract key arguments for summary view."""
        key_entries = []
        
        # Find highest confidence arguments from each role
        for round_data in acvf_result.debate_rounds:
            role_args = {}
            
            for arg in round_data.arguments:
                if arg.role not in role_args or (arg.confidence_score or 0) > (role_args[arg.role].confidence_score or 0):
                    role_args[arg.role] = arg
            
            # Create summary entries for top arguments
            for role, arg in role_args.items():
                entry = self._create_debate_entry_from_argument(arg, round_data, None, {})
                entry.summary = f"Key {role.value} argument: " + self._generate_argument_summary(arg.content, 100)
                key_entries.append(entry)
        
        return key_entries


# Export the main generator class and related models
__all__ = [
    'DebateViewGenerator',
    'DebateViewFormat',
    'ParticipantStyle',
    'SectionType',
    'DebateTheme',
    'DebateSection',
    'ThreadedConversation',
    'DebateNavigation'
] 