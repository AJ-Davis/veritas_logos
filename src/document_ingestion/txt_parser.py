"""
TXT document parser for plain text files.
"""

import chardet
from pathlib import Path
from typing import List

from .base_parser import BaseDocumentParser
from ..models.document import ParsedDocument, DocumentSection, DocumentFormat


class TxtParser(BaseDocumentParser):
    """Parser for plain text files."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.txt'}
    
    def get_format(self) -> DocumentFormat:
        """Return the document format this parser handles."""
        return DocumentFormat.TXT
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _parse_content(self, file_path: str) -> ParsedDocument:
        """
        Parse text file content.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedDocument instance
        """
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Split into sections (paragraphs)
            sections = self._create_sections(content)
            
            return ParsedDocument(
                content=content,
                sections=sections
            )
            
        except Exception as e:
            return ParsedDocument(
                content="",
                errors=[f"Failed to parse TXT file: {str(e)}"]
            )
    
    def _create_sections(self, content: str) -> List[DocumentSection]:
        """
        Create sections from text content.
        
        Args:
            content: Full text content
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        # Split by double newlines to identify paragraphs
        paragraphs = content.split('\n\n')
        position = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:
                # Determine section type
                section_type = self._determine_section_type(paragraph)
                
                sections.append(DocumentSection(
                    content=paragraph,
                    section_type=section_type,
                    position=position
                ))
                
                # Update position for next section
                position += len(paragraph) + 2  # +2 for the double newline
        
        return sections
    
    def _determine_section_type(self, text: str) -> str:
        """
        Determine the type of a text section.
        
        Args:
            text: Section text
            
        Returns:
            Section type
        """
        text = text.strip()
        
        # Check for headings (lines that are short and followed by content)
        lines = text.split('\n')
        if len(lines) == 1 and len(text) < 100:
            # Could be a heading if it's short and doesn't end with punctuation
            if not text.endswith(('.', '!', '?', ':')):
                return "heading"
        
        # Check for list items
        if text.startswith(('- ', '* ', '• ')):
            return "list_item"
        
        # Check for numbered lists
        if len(lines) > 0 and lines[0].strip() and lines[0].strip()[0].isdigit():
            if '. ' in lines[0] or ') ' in lines[0]:
                return "list_item"
        
        # Default to paragraph
        return "paragraph"