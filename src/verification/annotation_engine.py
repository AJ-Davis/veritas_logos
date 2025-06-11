"""
Document Annotation Engine for the Veritas Logos verification system.

This module provides the core engine for creating document annotations
from verification results, including highlights, comments, and references.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from src.models.verification import VerificationChainResult, VerificationResult
from src.models.issues import UnifiedIssue, IssueRegistry, IssueSeverity
from src.models.document import ParsedDocument, DocumentSection
from src.models.output import (
    DocumentAnnotation, AnnotationLayer, AnnotationType, HighlightStyle, 
    OutputVerificationResult, OutputGenerationConfig, ColorScheme,
    create_annotation_from_issue, create_output_result
)

logger = logging.getLogger(__name__)


class TextLocationStrategy(str, Enum):
    """Strategies for mapping issue locations to document text."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SECTION_BASED = "section_based"
    HYBRID = "hybrid"


@dataclass
class AnnotationContext:
    """Context information for annotation generation."""
    document: ParsedDocument
    chain_result: VerificationChainResult
    issue_registry: IssueRegistry
    config: OutputGenerationConfig
    location_strategy: TextLocationStrategy = TextLocationStrategy.HYBRID
    
    # Caching
    _text_positions: Optional[Dict[str, Tuple[int, int]]] = None
    _section_map: Optional[Dict[str, DocumentSection]] = None


class DocumentAnnotationEngine:
    """
    Core engine for generating document annotations from verification results.
    
    This engine takes verification results and creates structured annotations
    that can be applied to documents in various output formats.
    """
    
    def __init__(self, config: Optional[OutputGenerationConfig] = None):
        """
        Initialize the annotation engine.
        
        Args:
            config: Configuration for output generation
        """
        self.config = config or OutputGenerationConfig(output_format="json")
        self.logger = logger
        
        # Text matching configuration
        self.fuzzy_threshold = 0.8
        self.context_window = 100  # Characters before/after for context
        self.max_annotation_length = 1000  # Maximum annotation text length
        
    def generate_annotations(self, context: AnnotationContext) -> OutputVerificationResult:
        """
        Generate complete annotations for a document with verification results.
        
        Args:
            context: Annotation context containing document and verification data
            
        Returns:
            Complete output verification result with annotations
        """
        self.logger.info(f"Generating annotations for document {context.document.document_id}")
        
        # Create the output result structure
        output_result = create_output_result(
            chain_result=context.chain_result,
            issue_registry=context.issue_registry,
            document=context.document,
            config=context.config
        )
        
        # Clear default annotation layers to rebuild them properly
        output_result.annotation_layers = []
        
        # Initialize text mapping
        self._initialize_text_mapping(context)
        
        # Create annotation layers by type
        annotation_layers = self._create_annotation_layers(context)
        
        # Generate annotations for each issue
        for issue in context.issue_registry.issues:
            if self._should_include_issue(issue, context.config):
                annotations = self._create_annotations_for_issue(issue, context)
                self._add_annotations_to_layers(annotations, annotation_layers)
        
        # Add reference links between related annotations
        self._add_reference_links(annotation_layers, context)
        
        # Sort and optimize annotation layers
        self._optimize_annotation_layers(annotation_layers)
        
        # Add layers to output result
        for layer in annotation_layers.values():
            output_result.add_annotation_layer(layer)
        
        self.logger.info(f"Generated {sum(len(layer.annotations) for layer in annotation_layers.values())} annotations across {len(annotation_layers)} layers")
        
        return output_result
    
    def _initialize_text_mapping(self, context: AnnotationContext) -> None:
        """Initialize text position mapping for efficient location lookup."""
        if context._text_positions is None:
            context._text_positions = {}
            context._section_map = {}
            
            # Build section position map
            current_pos = 0
            full_text = context.document.content
            
            for section in context.document.sections:
                if section.content:
                    # Find section in full text
                    section_start = full_text.find(section.content, current_pos)
                    if section_start != -1:
                        section_end = section_start + len(section.content)
                        context._text_positions[section.section_id] = (section_start, section_end)
                        context._section_map[section.section_id] = section
                        current_pos = section_end
    
    def _create_annotation_layers(self, context: AnnotationContext) -> Dict[str, AnnotationLayer]:
        """Create annotation layers for different annotation types."""
        layers = {}
        
        # Define layer configurations
        layer_configs = [
            ("highlights", AnnotationType.HIGHLIGHT, "Issue Highlights", 1),
            ("comments", AnnotationType.COMMENT, "Issue Comments", 2),
            ("sidebar_notes", AnnotationType.SIDEBAR_NOTE, "Detailed Notes", 3),
            ("links", AnnotationType.LINK, "Reference Links", 4)
        ]
        
        for layer_key, annotation_type, layer_name, z_index in layer_configs:
            layers[layer_key] = AnnotationLayer(
                layer_name=layer_name,
                layer_type=annotation_type,
                z_index=z_index,
                visible=True
            )
        
        return layers
    
    def _should_include_issue(self, issue: UnifiedIssue, config: OutputGenerationConfig) -> bool:
        """Check if an issue should be included based on configuration."""
        # Check severity threshold
        if issue.severity.value < config.minimum_issue_severity.value:
            return False
        
        # Check if we have valid location information
        if not issue.location or (not issue.location.start_position and not issue.location.section_id):
            self.logger.warning(f"Issue {issue.issue_id} has no valid location information")
            return False
        
        return True
    
    def _create_annotations_for_issue(self, issue: UnifiedIssue, context: AnnotationContext) -> List[DocumentAnnotation]:
        """Create all annotations for a specific issue."""
        annotations = []
        
        # Get text location for the issue
        text_location = self._resolve_text_location(issue, context)
        
        if not text_location:
            self.logger.warning(f"Could not resolve text location for issue {issue.issue_id}")
            return annotations
        
        start_pos, end_pos, text_excerpt = text_location
        
        # Create highlight annotation
        highlight = DocumentAnnotation(
            annotation_type=AnnotationType.HIGHLIGHT,
            start_position=start_pos,
            end_position=end_pos,
            text_excerpt=text_excerpt,
            title=issue.title,
            content=issue.description,
            style=self._get_highlight_style(issue.severity),
            related_issue_id=issue.issue_id,
            metadata={
                "issue_type": issue.issue_type,
                "severity": issue.severity.value,
                "confidence": issue.confidence,
                "source_pass": issue.source_pass
            }
        )
        annotations.append(highlight)
        
        # Create comment annotation for detailed information
        if len(issue.description) > 50 or issue.remediation_suggestion:
            comment_content = self._build_comment_content(issue)
            comment = DocumentAnnotation(
                annotation_type=AnnotationType.COMMENT,
                start_position=start_pos,
                end_position=end_pos,
                text_excerpt=text_excerpt,
                title=f"Details: {issue.title}",
                content=comment_content,
                style=self._get_highlight_style(issue.severity),
                related_issue_id=issue.issue_id,
                metadata={
                    "annotation_subtype": "detailed_comment",
                    "has_remediation": bool(issue.remediation_suggestion)
                }
            )
            annotations.append(comment)
        
        # Create sidebar note for complex issues
        if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH] and issue.remediation_suggestion:
            sidebar_content = self._build_sidebar_content(issue)
            sidebar = DocumentAnnotation(
                annotation_type=AnnotationType.SIDEBAR_NOTE,
                start_position=start_pos,
                end_position=end_pos,
                text_excerpt=text_excerpt,
                title=f"Action Required: {issue.title}",
                content=sidebar_content,
                style=self._get_highlight_style(issue.severity),
                related_issue_id=issue.issue_id,
                metadata={
                    "annotation_subtype": "action_item",
                    "priority": "high" if issue.severity == IssueSeverity.CRITICAL else "medium"
                }
            )
            annotations.append(sidebar)
        
        return annotations
    
    def _resolve_text_location(self, issue: UnifiedIssue, context: AnnotationContext) -> Optional[Tuple[int, int, str]]:
        """
        Resolve the text location for an issue using various strategies.
        
        Returns:
            Tuple of (start_position, end_position, text_excerpt) or None if not found
        """
        document_text = context.document.content
        
        # Strategy 1: Use exact positions if available
        if issue.location.start_position is not None and issue.location.end_position is not None:
            start_pos = max(0, issue.location.start_position)
            end_pos = min(len(document_text), issue.location.end_position)
            
            if start_pos < end_pos:
                text_excerpt = document_text[start_pos:end_pos]
                return (start_pos, end_pos, text_excerpt)
        
        # Strategy 2: Use section-based mapping
        if issue.location.section_id and context._text_positions:
            section_positions = context._text_positions.get(issue.location.section_id)
            if section_positions:
                section_start, section_end = section_positions
                
                # If we have specific text, try to find it within the section
                if issue.text_excerpt:
                    section_text = document_text[section_start:section_end]
                    local_pos = self._find_text_in_section(issue.text_excerpt, section_text)
                    if local_pos:
                        local_start, local_end = local_pos
                        global_start = section_start + local_start
                        global_end = section_start + local_end
                        return (global_start, global_end, issue.text_excerpt)
                
                # Fallback: annotate a reasonable portion of the section
                excerpt_length = min(200, section_end - section_start)
                return (section_start, section_start + excerpt_length, 
                       document_text[section_start:section_start + excerpt_length])
        
        # Strategy 3: Fuzzy text matching if we have text excerpt
        if issue.text_excerpt:
            match_location = self._fuzzy_find_text(issue.text_excerpt, document_text)
            if match_location:
                start_pos, end_pos = match_location
                actual_text = document_text[start_pos:end_pos]
                return (start_pos, end_pos, actual_text)
        
        # Strategy 4: Use line/paragraph information if available
        if hasattr(issue.location, 'line_number') and issue.location.line_number:
            line_location = self._find_line_position(issue.location.line_number, document_text)
            if line_location:
                return line_location
        
        return None
    
    def _find_text_in_section(self, text: str, section_text: str) -> Optional[Tuple[int, int]]:
        """Find text within a specific section using fuzzy matching."""
        # Try exact match first
        pos = section_text.find(text)
        if pos != -1:
            return (pos, pos + len(text))
        
        # Try fuzzy matching for small variations
        words = text.split()
        if len(words) > 2:
            # Try matching with first and last words
            pattern = f"{re.escape(words[0])}.*{re.escape(words[-1])}"
            match = re.search(pattern, section_text, re.IGNORECASE | re.DOTALL)
            if match:
                return (match.start(), match.end())
        
        return None
    
    def _fuzzy_find_text(self, target_text: str, document_text: str) -> Optional[Tuple[int, int]]:
        """Find text in document using fuzzy matching techniques."""
        # Normalize whitespace
        normalized_target = ' '.join(target_text.split())
        normalized_document = ' '.join(document_text.split())
        
        # Try exact match on normalized text
        pos = normalized_document.find(normalized_target)
        if pos != -1:
            # Map back to original positions
            return self._map_normalized_to_original(pos, pos + len(normalized_target), 
                                                  document_text, normalized_document)
        
        # Try partial matching with keywords
        words = normalized_target.split()
        if len(words) >= 3:
            # Create pattern with key words
            key_words = [w for w in words if len(w) > 3][:3]  # Take up to 3 key words
            if key_words:
                pattern = '.*'.join(re.escape(word) for word in key_words)
                match = re.search(pattern, normalized_document, re.IGNORECASE)
                if match:
                    return self._map_normalized_to_original(match.start(), match.end(),
                                                          document_text, normalized_document)
        
        return None
    
    def _map_normalized_to_original(self, norm_start: int, norm_end: int, 
                                   original_text: str, normalized_text: str) -> Tuple[int, int]:
        """Map positions from normalized text back to original text."""
        # This is a simplified mapping - for production, you'd want more sophisticated handling
        char_ratio = len(original_text) / len(normalized_text) if normalized_text else 1
        
        orig_start = int(norm_start * char_ratio)
        orig_end = int(norm_end * char_ratio)
        
        # Ensure we don't go out of bounds
        orig_start = max(0, min(orig_start, len(original_text)))
        orig_end = max(orig_start, min(orig_end, len(original_text)))
        
        return (orig_start, orig_end)
    
    def _find_line_position(self, line_number: int, document_text: str) -> Optional[Tuple[int, int, str]]:
        """Find position of a specific line in the document."""
        lines = document_text.split('\n')
        if 0 <= line_number - 1 < len(lines):
            line_text = lines[line_number - 1]
            
            # Calculate position
            start_pos = sum(len(line) + 1 for line in lines[:line_number - 1])  # +1 for newline
            end_pos = start_pos + len(line_text)
            
            return (start_pos, end_pos, line_text)
        
        return None
    
    def _get_highlight_style(self, severity: IssueSeverity) -> HighlightStyle:
        """Get appropriate highlight style for issue severity."""
        style_map = {
            IssueSeverity.CRITICAL: HighlightStyle.CRITICAL,
            IssueSeverity.HIGH: HighlightStyle.HIGH,
            IssueSeverity.MEDIUM: HighlightStyle.MEDIUM,
            IssueSeverity.LOW: HighlightStyle.LOW,
            IssueSeverity.INFORMATIONAL: HighlightStyle.INFO
        }
        return style_map.get(severity, HighlightStyle.MEDIUM)
    
    def _build_comment_content(self, issue: UnifiedIssue) -> str:
        """Build comprehensive comment content for an issue."""
        parts = []
        
        # Issue description
        if issue.description:
            parts.append(f"Issue: {issue.description}")
        
        # Impact information
        if issue.impact:
            parts.append(f"Impact: {issue.impact}")
        
        # Confidence information
        if issue.confidence:
            parts.append(f"Confidence: {issue.confidence:.2f}")
        
        # Source pass information
        if issue.source_pass:
            parts.append(f"Detected by: {issue.source_pass}")
        
        # Remediation suggestion
        if issue.remediation_suggestion:
            parts.append(f"Suggested action: {issue.remediation_suggestion}")
        
        return "\n\n".join(parts)
    
    def _build_sidebar_content(self, issue: UnifiedIssue) -> str:
        """Build sidebar content for critical issues."""
        parts = []
        
        # Priority indicator
        priority = "🔴 CRITICAL" if issue.severity == IssueSeverity.CRITICAL else "🟠 HIGH PRIORITY"
        parts.append(priority)
        
        # Quick summary
        parts.append(f"Summary: {issue.title}")
        
        # Action needed
        if issue.remediation_suggestion:
            parts.append(f"Action needed: {issue.remediation_suggestion}")
        else:
            parts.append("Action needed: Review and address this issue")
        
        # Additional context
        if issue.impact:
            parts.append(f"Potential impact: {issue.impact}")
        
        return "\n\n".join(parts)
    
    def _add_annotations_to_layers(self, annotations: List[DocumentAnnotation], 
                                  layers: Dict[str, AnnotationLayer]) -> None:
        """Add annotations to appropriate layers."""
        layer_map = {
            AnnotationType.HIGHLIGHT: "highlights",
            AnnotationType.COMMENT: "comments",
            AnnotationType.SIDEBAR_NOTE: "sidebar_notes",
            AnnotationType.LINK: "links"
        }
        
        for annotation in annotations:
            layer_key = layer_map.get(annotation.annotation_type)
            if layer_key and layer_key in layers:
                layers[layer_key].add_annotation(annotation)
    
    def _add_reference_links(self, layers: Dict[str, AnnotationLayer], 
                           context: AnnotationContext) -> None:
        """Add reference links between related annotations."""
        # Find related issues and create link annotations
        issue_positions = {}
        
        # Build map of issue positions
        for layer in layers.values():
            for annotation in layer.annotations:
                if annotation.related_issue_id:
                    issue_positions[annotation.related_issue_id] = (
                        annotation.start_position, annotation.end_position
                    )
        
        # Create links between related issues
        links_layer = layers.get("links")
        if links_layer:
            for issue in context.issue_registry.issues:
                if hasattr(issue, 'related_issues') and issue.related_issues:
                    source_pos = issue_positions.get(issue.issue_id)
                    if source_pos:
                        for related_id in issue.related_issues:
                            target_pos = issue_positions.get(related_id)
                            if target_pos:
                                link = DocumentAnnotation(
                                    annotation_type=AnnotationType.LINK,
                                    start_position=source_pos[0],
                                    end_position=source_pos[1],
                                    text_excerpt="",
                                    title=f"Related to issue {related_id}",
                                    content=f"See related issue at position {target_pos[0]}-{target_pos[1]}",
                                    style=HighlightStyle.INFO,
                                    related_issue_id=issue.issue_id,
                                    metadata={
                                        "link_type": "related_issue",
                                        "target_issue_id": related_id,
                                        "target_position": target_pos
                                    }
                                )
                                links_layer.add_annotation(link)
    
    def _optimize_annotation_layers(self, layers: Dict[str, AnnotationLayer]) -> None:
        """Optimize annotation layers for better performance and rendering."""
        for layer in layers.values():
            # Sort annotations by position
            layer.annotations.sort(key=lambda a: (a.start_position, a.end_position))
            
            # Remove duplicate annotations
            unique_annotations = []
            seen_positions = set()
            
            for annotation in layer.annotations:
                position_key = (annotation.start_position, annotation.end_position, 
                              annotation.annotation_type, annotation.related_issue_id)
                if position_key not in seen_positions:
                    unique_annotations.append(annotation)
                    seen_positions.add(position_key)
            
            layer.annotations = unique_annotations
            
            # Log layer statistics
            self.logger.debug(f"Layer '{layer.layer_name}': {len(layer.annotations)} annotations")


# Utility functions for coordinate system conversion

def document_to_section_coordinates(document_pos: int, section_map: Dict[str, Tuple[int, int]]) -> Optional[Tuple[str, int]]:
    """Convert document position to section-relative position."""
    for section_id, (start, end) in section_map.items():
        if start <= document_pos < end:
            return (section_id, document_pos - start)
    return None


def section_to_document_coordinates(section_id: str, section_pos: int, 
                                  section_map: Dict[str, Tuple[int, int]]) -> Optional[int]:
    """Convert section-relative position to document position."""
    if section_id in section_map:
        section_start, section_end = section_map[section_id]
        document_pos = section_start + section_pos
        if document_pos < section_end:
            return document_pos
    return None


def merge_overlapping_annotations(annotations: List[DocumentAnnotation]) -> List[DocumentAnnotation]:
    """Merge overlapping annotations of the same type to avoid visual conflicts."""
    if not annotations:
        return []
    
    # Group by annotation type
    by_type = {}
    for annotation in annotations:
        key = annotation.annotation_type
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(annotation)
    
    merged = []
    
    for annotation_type, type_annotations in by_type.items():
        # Sort by position
        type_annotations.sort(key=lambda a: (a.start_position, a.end_position))
        
        current_group = [type_annotations[0]]
        
        for annotation in type_annotations[1:]:
            last_annotation = current_group[-1]
            
            # Check if annotations overlap
            if annotation.start_position <= last_annotation.end_position:
                # Merge overlapping annotations
                current_group.append(annotation)
            else:
                # Finalize current group and start new one
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    merged_annotation = _merge_annotation_group(current_group)
                    merged.append(merged_annotation)
                current_group = [annotation]
        
        # Handle final group
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged_annotation = _merge_annotation_group(current_group)
            merged.append(merged_annotation)
    
    return merged


def _merge_annotation_group(annotations: List[DocumentAnnotation]) -> DocumentAnnotation:
    """Merge a group of overlapping annotations into a single annotation."""
    if len(annotations) == 1:
        return annotations[0]
    
    # Calculate merged boundaries
    start_pos = min(a.start_position for a in annotations)
    end_pos = max(a.end_position for a in annotations)
    
    # Combine content
    titles = [a.title for a in annotations if a.title]
    contents = [a.content for a in annotations if a.content]
    related_issues = [a.related_issue_id for a in annotations if a.related_issue_id]
    
    # Use the highest severity style
    severity_order = [HighlightStyle.CRITICAL, HighlightStyle.HIGH, HighlightStyle.MEDIUM, 
                     HighlightStyle.LOW, HighlightStyle.INFO]
    best_style = HighlightStyle.INFO
    for style in severity_order:
        if any(a.style == style for a in annotations):
            best_style = style
            break
    
    return DocumentAnnotation(
        annotation_type=annotations[0].annotation_type,
        start_position=start_pos,
        end_position=end_pos,
        text_excerpt=annotations[0].text_excerpt,  # Use first excerpt as representative
        title=f"Multiple issues: {', '.join(titles)}" if titles else "Multiple issues",
        content="\n---\n".join(contents) if contents else "Multiple issues detected",
        style=best_style,
        related_issue_id=related_issues[0] if related_issues else None,
        metadata={
            "merged_annotation": True,
            "merged_count": len(annotations),
            "related_issue_ids": related_issues
        }
    ) 