"""
Markdown document parser for .md and .markdown files.
"""

import re
from pathlib import Path
from typing import List
from markdown_it import MarkdownIt

from .base_parser import BaseDocumentParser
from ..models.document import ParsedDocument, DocumentSection, DocumentFormat


class MarkdownParser(BaseDocumentParser):
    """Parser for Markdown files."""
    
    def __init__(self):
        super().__init__(supported_extensions={'.md', '.markdown'})
        self.md = MarkdownIt()
    
    def get_format(self) -> DocumentFormat:
        """Return the document format this parser handles."""
        return DocumentFormat.MARKDOWN
    
    def _parse_content(self, file_path: str) -> ParsedDocument:
        """
        Parse Markdown file content.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedDocument instance
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Parse markdown and extract sections
            sections = self._create_sections(content)
            
            # Convert to plain text for the main content
            plain_text = self._markdown_to_text(content)
            
            return ParsedDocument(
                content=plain_text,
                sections=sections,
                raw_data={"markdown_content": content}
            )
            
        except Exception as e:
            return ParsedDocument(
                content="",
                errors=[f"Failed to parse Markdown file: {str(e)}"]
            )
    
    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Convert Markdown to plain text.
        
        Args:
            markdown_content: Markdown content
            
        Returns:
            Plain text version
        """
        # Parse markdown to tokens
        tokens = self.md.parse(markdown_content)
        
        # Extract text content
        text_parts = []
        
        for token in tokens:
            if hasattr(token, 'content') and token.content:
                text_parts.append(token.content)
            elif hasattr(token, 'children') and token.children:
                for child in token.children:
                    if hasattr(child, 'content') and child.content:
                        text_parts.append(child.content)
        
        # Join and clean up
        text = '\n'.join(text_parts)
        
        # Use markdown_it's renderer for clean text extraction
        html = self.md.render(markdown_content)
        # Then strip HTML tags for plain text
        import re
        text = re.sub('<[^<]+?>', '', html)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip()
    
    def _create_sections(self, content: str) -> List[DocumentSection]:
        """
        Create sections from Markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        lines = content.split('\n')
        current_section = []
        current_type = "paragraph"
        position = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Determine section type
            section_type = self._determine_section_type(line)
            
            # If section type changes or we hit a header, start new section
            if section_type != current_type or section_type == "heading":
                # Save previous section if it has content
                if current_section:
                    section_content = '\n'.join(current_section).strip()
                    if section_content:
                        sections.append(DocumentSection(
                            content=section_content,
                            section_type=current_type,
                            position=position
                        ))
                        position += len(section_content) + 1
                
                # Start new section
                current_section = [line] if line_stripped else []
                current_type = section_type
            else:
                # Continue current section
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                sections.append(DocumentSection(
                    content=section_content,
                    section_type=current_type,
                    position=position
                ))
        
        return sections
    
    def _determine_section_type(self, line: str) -> str:
        """
        Determine the type of a Markdown line.
        
        Args:
            line: Line of text
            
        Returns:
            Section type
        """
        line = line.strip()
        
        # Headers
        if line.startswith('#'):
            return "heading"
        
        # Alternative header syntax (must be only equals or dashes)
        if len(line) >= 3 and (set(line) == {'='} or set(line) == {'-'}):
            return "heading"
        
        # Code blocks
        if line.startswith('```') or line.startswith('~~~'):
            return "code"
        
        # Lists
        if re.match(r'^\s*[-*+]\s+', line):
            return "list_item"
        
        # Numbered lists
        if re.match(r'^\s*\d+\.\s+', line):
            return "list_item"
        
        # Blockquotes
        if line.startswith('>'):
            return "blockquote"
        
        # Tables
        if '|' in line and line.count('|') >= 2:
            return "table"
        
        # Horizontal rules
        if re.match(r'^[-*_]{3,}$', line):
            return "horizontal_rule"
        
        # Empty lines
        if not line:
            return "empty"
        
        # Default to paragraph
        return "paragraph"