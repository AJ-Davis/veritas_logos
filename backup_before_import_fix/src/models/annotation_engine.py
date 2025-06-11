"""
Document Annotation Engine for Veritas Logos verification system.

This module provides the core engine for annotating documents with issue highlights,
comments, and references. It maps verification results to document locations and 
creates appropriate annotations with configurable styles.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
import difflib
from collections import defaultdict

from .output import (
    DocumentAnnotation, AnnotationLayer, AnnotationType, HighlightStyle,
    OutputVerificationResult, ColorScheme, OutputGenerationConfig
)
from .verification import VerificationChainResult, VerificationResult
from .issues import UnifiedIssue, IssueSeverity
from .document import ParsedDocument, DocumentSection
from .acvf import ACVFResult, DebateRound


logger = logging.getLogger(__name__)


class PositionMapping(Enum):
    """Methods for mapping issue locations to document positions."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    CONTEXTUAL_SEARCH = "contextual_search"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"


@dataclass
class TextMatch:
    """Represents a matched text location in a document."""
    start_position: int
    end_position: int
    matched_text: str
    confidence: float
    method: PositionMapping
    context_before: str = ""
    context_after: str = ""


@dataclass
class AnnotationConflict:
    """Represents a conflict between overlapping annotations."""
    annotation1: DocumentAnnotation
    annotation2: DocumentAnnotation
    overlap_start: int
    overlap_end: int
    conflict_type: str  # "full_overlap", "partial_overlap", "nested"


class DocumentAnnotationEngine:
    """
    Core engine for annotating documents with verification issues.
    
    This engine takes verification results and maps them to specific locations
    in the original document, creating appropriate annotations with styling
    based on issue severity and type.
    """
    
    def __init__(self, config: Optional[OutputGenerationConfig] = None):
        """
        Initialize the annotation engine.
        
        Args:
            config: Output generation configuration
        """
        self.config = config or OutputGenerationConfig(output_format="json")
        self.color_scheme = self.config.color_scheme
        self._position_cache: Dict[str, List[TextMatch]] = {}
        self._annotation_cache: Dict[str, DocumentAnnotation] = {}
    
    def annotate_document(self, verification_result: OutputVerificationResult) -> OutputVerificationResult:
        """
        Annotate a document with verification results.
        
        Args:
            verification_result: The verification result containing document and issues
            
        Returns:
            Updated verification result with annotations
        """
        logger.info(f"Starting annotation of document {verification_result.document_id}")
        
        # Clear any existing annotation layers
        verification_result.annotation_layers.clear()
        
        # Create annotation layers for different types
        layers = self._create_annotation_layers()
        
        # Process each issue and create annotations
        for issue in verification_result.issue_registry.get_all_issues():
            if self._should_annotate_issue(issue):
                annotations = self._create_annotations_for_issue(
                    issue, verification_result.document
                )
                self._add_annotations_to_layers(annotations, layers)
        
        # Process ACVF debates if available
        if verification_result.acvf_results:
            for acvf_result in verification_result.acvf_results:
                debate_annotations = self._create_debate_annotations(
                    acvf_result, verification_result.document
                )
                self._add_annotations_to_layers(debate_annotations, layers)
        
        # Resolve annotation conflicts
        for layer in layers.values():
            self._resolve_annotation_conflicts(layer)
        
        # Add layers to result
        for layer in layers.values():
            if layer.annotations:  # Only add layers with annotations
                verification_result.add_annotation_layer(layer)
        
        logger.info(f"Created {sum(len(layer.annotations) for layer in layers.values())} annotations")
        return verification_result
    
    def _create_annotation_layers(self) -> Dict[AnnotationType, AnnotationLayer]:
        """Create the standard annotation layers."""
        layers = {}
        
        # Highlight layer for issue locations
        layers[AnnotationType.HIGHLIGHT] = AnnotationLayer(
            layer_name="Issue Highlights",
            layer_type=AnnotationType.HIGHLIGHT,
            z_index=1
        )
        
        # Comment layer for detailed issue information
        layers[AnnotationType.COMMENT] = AnnotationLayer(
            layer_name="Issue Comments",
            layer_type=AnnotationType.COMMENT,
            z_index=2
        )
        
        # Sidebar notes for additional context
        layers[AnnotationType.SIDEBAR_NOTE] = AnnotationLayer(
            layer_name="Sidebar Notes",
            layer_type=AnnotationType.SIDEBAR_NOTE,
            z_index=3
        )
        
        # Links to related content
        layers[AnnotationType.LINK] = AnnotationLayer(
            layer_name="Reference Links",
            layer_type=AnnotationType.LINK,
            z_index=4
        )
        
        return layers
    
    def _should_annotate_issue(self, issue: UnifiedIssue) -> bool:
        """Determine if an issue should be annotated based on configuration."""
        # Check minimum severity
        if issue.severity < self.config.minimum_issue_severity:
            return False
        
        # Check if issue has location information
        if not issue.context or not issue.context.get("text_location"):
            logger.debug(f"Skipping issue {issue.issue_id} - no location information")
            return False
        
        return True
    
    def _create_annotations_for_issue(self, issue: UnifiedIssue, 
                                    document: ParsedDocument) -> List[DocumentAnnotation]:
        """
        Create annotations for a specific issue.
        
        Args:
            issue: The issue to annotate
            document: The document being annotated
            
        Returns:
            List of annotations created for this issue
        """
        annotations = []
        
        # Find text location in document
        text_matches = self._find_text_location(issue, document)
        
        if not text_matches:
            logger.warning(f"Could not find location for issue {issue.issue_id}")
            return annotations
        
        # Create annotations for each match
        for match in text_matches:
            # Create highlight annotation
            highlight = self._create_highlight_annotation(issue, match)
            annotations.append(highlight)
            
            # Create comment annotation if detailed information is available
            if issue.description or issue.recommendation:
                comment = self._create_comment_annotation(issue, match)
                annotations.append(comment)
            
            # Create sidebar note for additional context
            if issue.context and len(str(issue.context)) > 100:
                sidebar = self._create_sidebar_annotation(issue, match)
                annotations.append(sidebar)
        
        return annotations
    
    def _find_text_location(self, issue: UnifiedIssue, 
                          document: ParsedDocument) -> List[TextMatch]:
        """
        Find the location of an issue's text within the document.
        
        Args:
            issue: The issue containing location information
            document: The document to search
            
        Returns:
            List of text matches found
        """
        # Check cache first
        cache_key = f"{issue.issue_id}_{document.document_id}"
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]
        
        text_location = issue.context.get("text_location", {})
        target_text = text_location.get("text", "")
        
        if not target_text:
            return []
        
        matches = []
        document_text = document.full_text
        
        # Try different matching strategies
        strategies = [
            (PositionMapping.EXACT_MATCH, self._exact_match),
            (PositionMapping.FUZZY_MATCH, self._fuzzy_match),
            (PositionMapping.CONTEXTUAL_SEARCH, self._contextual_search),
            (PositionMapping.SENTENCE_BOUNDARY, self._sentence_boundary_match),
            (PositionMapping.PARAGRAPH_BOUNDARY, self._paragraph_boundary_match)
        ]
        
        for strategy_name, strategy_func in strategies:
            strategy_matches = strategy_func(target_text, document_text, text_location)
            for match in strategy_matches:
                match.method = strategy_name
                matches.append(match)
            
            # If we found high-confidence matches, we can stop
            if matches and any(m.confidence > 0.9 for m in matches):
                break
        
        # Cache results
        self._position_cache[cache_key] = matches
        return matches
    
    def _exact_match(self, target_text: str, document_text: str, 
                    location_info: Dict[str, Any]) -> List[TextMatch]:
        """Find exact matches of the target text."""
        matches = []
        start = 0
        
        while True:
            pos = document_text.find(target_text, start)
            if pos == -1:
                break
            
            match = TextMatch(
                start_position=pos,
                end_position=pos + len(target_text),
                matched_text=target_text,
                confidence=1.0,
                method=PositionMapping.EXACT_MATCH
            )
            
            # Add context
            context_size = 50
            match.context_before = document_text[max(0, pos - context_size):pos]
            match.context_after = document_text[pos + len(target_text):pos + len(target_text) + context_size]
            
            matches.append(match)
            start = pos + 1
        
        return matches
    
    def _fuzzy_match(self, target_text: str, document_text: str,
                    location_info: Dict[str, Any]) -> List[TextMatch]:
        """Find fuzzy matches using sequence matching."""
        matches = []
        
        # Use difflib to find similar sequences
        matcher = difflib.SequenceMatcher(None, target_text.lower(), document_text.lower())
        
        for match in matcher.get_matching_blocks():
            if match.size < len(target_text) * 0.6:  # Require at least 60% match
                continue
            
            # Map back to original text positions
            start_pos = match.b
            end_pos = match.b + match.size
            matched_text = document_text[start_pos:end_pos]
            
            # Calculate confidence based on similarity
            similarity = match.size / len(target_text)
            
            text_match = TextMatch(
                start_position=start_pos,
                end_position=end_pos,
                matched_text=matched_text,
                confidence=similarity,
                method=PositionMapping.FUZZY_MATCH
            )
            
            matches.append(text_match)
        
        return matches
    
    def _contextual_search(self, target_text: str, document_text: str,
                          location_info: Dict[str, Any]) -> List[TextMatch]:
        """Find matches using contextual information."""
        matches = []
        
        # Extract key phrases from target text
        words = target_text.split()
        if len(words) < 3:
            return matches
        
        # Look for sequences of words
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            start = 0
            
            while True:
                pos = document_text.lower().find(phrase.lower(), start)
                if pos == -1:
                    break
                
                # Expand to find full sentence or clause
                expanded_start = pos
                expanded_end = pos + len(phrase)
                
                # Look for sentence boundaries
                while expanded_start > 0 and document_text[expanded_start - 1] not in '.!?':
                    expanded_start -= 1
                
                while expanded_end < len(document_text) and document_text[expanded_end] not in '.!?':
                    expanded_end += 1
                
                matched_text = document_text[expanded_start:expanded_end]
                confidence = len(phrase) / len(matched_text)
                
                text_match = TextMatch(
                    start_position=expanded_start,
                    end_position=expanded_end,
                    matched_text=matched_text,
                    confidence=confidence,
                    method=PositionMapping.CONTEXTUAL_SEARCH
                )
                
                matches.append(text_match)
                start = pos + 1
        
        return matches
    
    def _sentence_boundary_match(self, target_text: str, document_text: str,
                                location_info: Dict[str, Any]) -> List[TextMatch]:
        """Find matches at sentence boundaries."""
        # Split document into sentences
        sentences = re.split(r'[.!?]+', document_text)
        matches = []
        
        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if target text is similar to this sentence
            similarity = difflib.SequenceMatcher(None, target_text.lower(), sentence.lower()).ratio()
            
            if similarity > 0.6:
                start_pos = document_text.find(sentence, current_pos)
                if start_pos != -1:
                    text_match = TextMatch(
                        start_position=start_pos,
                        end_position=start_pos + len(sentence),
                        matched_text=sentence,
                        confidence=similarity,
                        method=PositionMapping.SENTENCE_BOUNDARY
                    )
                    matches.append(text_match)
            
            current_pos += len(sentence) + 1
        
        return matches
    
    def _paragraph_boundary_match(self, target_text: str, document_text: str,
                                 location_info: Dict[str, Any]) -> List[TextMatch]:
        """Find matches at paragraph boundaries."""
        # Split document into paragraphs
        paragraphs = document_text.split('\n\n')
        matches = []
        
        current_pos = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if target text is contained in this paragraph
            if target_text.lower() in paragraph.lower():
                start_pos = document_text.find(paragraph, current_pos)
                if start_pos != -1:
                    text_match = TextMatch(
                        start_position=start_pos,
                        end_position=start_pos + len(paragraph),
                        matched_text=paragraph,
                        confidence=0.7,  # Lower confidence for paragraph matches
                        method=PositionMapping.PARAGRAPH_BOUNDARY
                    )
                    matches.append(text_match)
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        return matches
    
    def _create_highlight_annotation(self, issue: UnifiedIssue, 
                                   match: TextMatch) -> DocumentAnnotation:
        """Create a highlight annotation for an issue."""
        # Determine highlight style based on severity
        style_mapping = {
            IssueSeverity.CRITICAL: HighlightStyle.CRITICAL,
            IssueSeverity.HIGH: HighlightStyle.HIGH,
            IssueSeverity.MEDIUM: HighlightStyle.MEDIUM,
            IssueSeverity.LOW: HighlightStyle.LOW,
            IssueSeverity.INFO: HighlightStyle.INFO
        }
        
        style = style_mapping.get(issue.severity, HighlightStyle.INFO)
        
        return DocumentAnnotation(
            annotation_type=AnnotationType.HIGHLIGHT,
            start_position=match.start_position,
            end_position=match.end_position,
            text_excerpt=match.matched_text,
            title=f"{issue.issue_type.value}: {issue.title}",
            content=issue.description or f"Issue detected: {issue.title}",
            style=style,
            related_issue_id=issue.issue_id,
            metadata={
                "match_confidence": match.confidence,
                "match_method": match.method.value,
                "issue_severity": issue.severity.value,
                "issue_type": issue.issue_type.value
            }
        )
    
    def _create_comment_annotation(self, issue: UnifiedIssue,
                                 match: TextMatch) -> DocumentAnnotation:
        """Create a comment annotation for an issue."""
        content_parts = []
        
        if issue.description:
            content_parts.append(f"Description: {issue.description}")
        
        if issue.recommendation:
            content_parts.append(f"Recommendation: {issue.recommendation}")
        
        if issue.context:
            context_str = str(issue.context)
            if len(context_str) <= 200:
                content_parts.append(f"Context: {context_str}")
        
        content = "\n\n".join(content_parts)
        
        return DocumentAnnotation(
            annotation_type=AnnotationType.COMMENT,
            start_position=match.start_position,
            end_position=match.end_position,
            text_excerpt=match.matched_text,
            title=f"Issue Details: {issue.title}",
            content=content,
            style=HighlightStyle.INFO,
            related_issue_id=issue.issue_id,
            metadata={
                "comment_type": "issue_details",
                "issue_severity": issue.severity.value
            }
        )
    
    def _create_sidebar_annotation(self, issue: UnifiedIssue,
                                 match: TextMatch) -> DocumentAnnotation:
        """Create a sidebar annotation for additional context."""
        return DocumentAnnotation(
            annotation_type=AnnotationType.SIDEBAR_NOTE,
            start_position=match.start_position,
            end_position=match.end_position,
            text_excerpt=match.matched_text,
            title=f"Additional Context: {issue.title}",
            content=str(issue.context),
            style=HighlightStyle.INFO,
            related_issue_id=issue.issue_id,
            metadata={
                "note_type": "additional_context"
            }
        )
    
    def _create_debate_annotations(self, acvf_result: ACVFResult,
                                 document: ParsedDocument) -> List[DocumentAnnotation]:
        """Create annotations for ACVF debate results."""
        annotations = []
        
        for round_data in acvf_result.rounds:
            debate_round = round_data.get("round")
            if not isinstance(debate_round, DebateRound):
                continue
            
            # Find subject text in document if available
            subject_text = debate_round.subject_content
            if subject_text and len(subject_text) > 10:
                matches = self._find_debate_text_location(subject_text, document)
                
                for match in matches:
                    # Create link annotation to debate view
                    debate_annotation = DocumentAnnotation(
                        annotation_type=AnnotationType.LINK,
                        start_position=match.start_position,
                        end_position=match.end_position,
                        text_excerpt=match.matched_text,
                        title="View ACVF Debate",
                        content=f"This content was subject to adversarial debate. Click to view the full debate.",
                        style=HighlightStyle.INFO,
                        related_debate_id=debate_round.round_id,
                        metadata={
                            "debate_type": "acvf",
                            "total_rounds": len(acvf_result.rounds),
                            "final_verdict": str(acvf_result.final_verdict) if acvf_result.final_verdict else None
                        }
                    )
                    annotations.append(debate_annotation)
        
        return annotations
    
    def _find_debate_text_location(self, subject_text: str,
                                 document: ParsedDocument) -> List[TextMatch]:
        """Find locations of debate subject text in the document."""
        # Similar to _find_text_location but optimized for debate content
        document_text = document.full_text
        matches = []
        
        # Try exact match first
        exact_matches = self._exact_match(subject_text, document_text, {})
        if exact_matches:
            return exact_matches
        
        # Try fuzzy match
        fuzzy_matches = self._fuzzy_match(subject_text, document_text, {})
        return fuzzy_matches
    
    def _add_annotations_to_layers(self, annotations: List[DocumentAnnotation],
                                 layers: Dict[AnnotationType, AnnotationLayer]) -> None:
        """Add annotations to appropriate layers."""
        for annotation in annotations:
            layer = layers.get(annotation.annotation_type)
            if layer:
                layer.add_annotation(annotation)
    
    def _resolve_annotation_conflicts(self, layer: AnnotationLayer) -> None:
        """
        Resolve conflicts between overlapping annotations in a layer.
        
        Args:
            layer: The annotation layer to process
        """
        if len(layer.annotations) <= 1:
            return
        
        # Sort annotations by position
        layer.annotations.sort(key=lambda a: (a.start_position, a.end_position))
        
        # Find conflicts
        conflicts = []
        for i, ann1 in enumerate(layer.annotations):
            for j, ann2 in enumerate(layer.annotations[i+1:], i+1):
                if ann1.overlaps_with(ann2):
                    overlap_start = max(ann1.start_position, ann2.start_position)
                    overlap_end = min(ann1.end_position, ann2.end_position)
                    
                    conflict = AnnotationConflict(
                        annotation1=ann1,
                        annotation2=ann2,
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        conflict_type=self._classify_overlap(ann1, ann2)
                    )
                    conflicts.append(conflict)
        
        # Resolve conflicts
        for conflict in conflicts:
            self._resolve_conflict(conflict, layer)
    
    def _classify_overlap(self, ann1: DocumentAnnotation, 
                         ann2: DocumentAnnotation) -> str:
        """Classify the type of overlap between two annotations."""
        if (ann1.start_position <= ann2.start_position and 
            ann1.end_position >= ann2.end_position):
            return "nested"  # ann1 contains ann2
        elif (ann2.start_position <= ann1.start_position and 
              ann2.end_position >= ann1.end_position):
            return "nested"  # ann2 contains ann1
        elif (ann1.start_position == ann2.start_position and 
              ann1.end_position == ann2.end_position):
            return "full_overlap"
        else:
            return "partial_overlap"
    
    def _resolve_conflict(self, conflict: AnnotationConflict,
                         layer: AnnotationLayer) -> None:
        """
        Resolve a specific annotation conflict.
        
        Args:
            conflict: The conflict to resolve
            layer: The layer containing the conflicting annotations
        """
        ann1, ann2 = conflict.annotation1, conflict.annotation2
        
        if conflict.conflict_type == "full_overlap":
            # Keep the annotation with higher priority (based on severity or type)
            if self._get_annotation_priority(ann1) >= self._get_annotation_priority(ann2):
                # Merge information from ann2 into ann1
                self._merge_annotations(ann1, ann2)
                layer.annotations.remove(ann2)
            else:
                self._merge_annotations(ann2, ann1)
                layer.annotations.remove(ann1)
        
        elif conflict.conflict_type == "nested":
            # For nested annotations, we can keep both but adjust styling
            # The outer annotation becomes a subtle background
            # The inner annotation remains prominent
            pass  # Keep both for now
        
        elif conflict.conflict_type == "partial_overlap":
            # Split overlapping annotations at boundaries
            self._split_overlapping_annotations(ann1, ann2, layer)
    
    def _get_annotation_priority(self, annotation: DocumentAnnotation) -> int:
        """Get priority score for an annotation (higher = more important)."""
        priority = 0
        
        # Priority based on annotation type
        type_priorities = {
            AnnotationType.HIGHLIGHT: 5,
            AnnotationType.COMMENT: 4,
            AnnotationType.SIDEBAR_NOTE: 3,
            AnnotationType.LINK: 2,
            AnnotationType.TOOLTIP: 1
        }
        priority += type_priorities.get(annotation.annotation_type, 0)
        
        # Priority based on severity (if available)
        if annotation.metadata.get("issue_severity"):
            severity_priorities = {
                "critical": 10,
                "high": 8,
                "medium": 6,
                "low": 4,
                "info": 2
            }
            priority += severity_priorities.get(
                annotation.metadata["issue_severity"], 0
            )
        
        return priority
    
    def _merge_annotations(self, target: DocumentAnnotation, 
                          source: DocumentAnnotation) -> None:
        """Merge information from source annotation into target."""
        # Combine content
        if source.content and source.content not in target.content:
            target.content += f"\n\n{source.content}"
        
        # Merge metadata
        for key, value in source.metadata.items():
            if key not in target.metadata:
                target.metadata[key] = value
        
        # Add reference to merged annotation
        if source.related_issue_id and source.related_issue_id != target.related_issue_id:
            merged_issues = target.metadata.get("merged_issues", [])
            merged_issues.append(source.related_issue_id)
            target.metadata["merged_issues"] = merged_issues
    
    def _split_overlapping_annotations(self, ann1: DocumentAnnotation,
                                     ann2: DocumentAnnotation,
                                     layer: AnnotationLayer) -> None:
        """Split partially overlapping annotations."""
        # For now, we'll keep both annotations as-is
        # In a more sophisticated implementation, we could split them
        # at the overlap boundaries and create separate annotations
        # for the non-overlapping parts
        pass
    
    def get_annotations_at_position(self, position: int, 
                                   layers: List[AnnotationLayer]) -> List[DocumentAnnotation]:
        """Get all annotations at a specific position across all layers."""
        annotations = []
        for layer in layers:
            if layer.visible:
                annotations.extend(layer.get_annotations_at_position(position))
        return annotations
    
    def get_annotation_statistics(self, layers: List[AnnotationLayer]) -> Dict[str, Any]:
        """Get statistics about annotations across all layers."""
        stats = {
            "total_annotations": 0,
            "annotations_by_type": defaultdict(int),
            "annotations_by_severity": defaultdict(int),
            "average_annotation_length": 0,
            "layers": {}
        }
        
        total_length = 0
        
        for layer in layers:
            layer_stats = {
                "count": len(layer.annotations),
                "visible": layer.visible,
                "type": layer.layer_type.value
            }
            
            for annotation in layer.annotations:
                stats["total_annotations"] += 1
                stats["annotations_by_type"][annotation.annotation_type.value] += 1
                
                severity = annotation.metadata.get("issue_severity")
                if severity:
                    stats["annotations_by_severity"][severity] += 1
                
                total_length += annotation.length
            
            stats["layers"][layer.layer_name] = layer_stats
        
        if stats["total_annotations"] > 0:
            stats["average_annotation_length"] = total_length / stats["total_annotations"]
        
        return dict(stats) 