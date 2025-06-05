"""
Base parser class for document ingestion.
"""

from abc import ABC, abstractmethod
import time
import hashlib
from pathlib import Path
from typing import Optional

from ..models.document import ParsedDocument, DocumentMetadata, DocumentFormat, ExtractionMethod
from ..utils.validation import validate_file_before_processing, validate_token_count


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    def __init__(self):
        self.supported_extensions = set()
        self.default_extraction_method = ExtractionMethod.DIRECT
    
    @abstractmethod
    def _parse_content(self, file_path: str) -> ParsedDocument:
        """
        Parse document content. Must be implemented by subclasses.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedDocument instance
        """
        pass
    
    @abstractmethod
    def get_format(self) -> DocumentFormat:
        """Return the document format this parser handles."""
        pass
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if parser can handle this file type
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document with validation and metadata generation.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedDocument instance with metadata
        """
        start_time = time.time()
        
        # Pre-processing validation
        is_valid, error = validate_file_before_processing(file_path)
        if not is_valid:
            return ParsedDocument(
                content="",
                errors=[f"Validation failed: {error}"]
            )
        
        try:
            # Parse the document
            parsed_doc = self._parse_content(file_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata if not already set
            if parsed_doc.metadata is None:
                file_path_obj = Path(file_path)
                file_size = file_path_obj.stat().st_size
                
                # Generate checksum
                with open(file_path, 'rb') as f:
                    content = f.read()
                    checksum = hashlib.sha256(content).hexdigest()
                
                parsed_doc.metadata = DocumentMetadata(
                    filename=file_path_obj.name,
                    file_size_bytes=file_size,
                    format=self.get_format(),
                    extraction_method=self.default_extraction_method,
                    processing_time_seconds=processing_time,
                    checksum=checksum,
                    character_count=len(parsed_doc.content),
                    word_count=len(parsed_doc.content.split())
                )
            else:
                # Update existing metadata
                parsed_doc.metadata.processing_time_seconds = processing_time
                if parsed_doc.metadata.character_count is None:
                    parsed_doc.metadata.character_count = len(parsed_doc.content)
                if parsed_doc.metadata.word_count is None:
                    parsed_doc.metadata.word_count = len(parsed_doc.content.split())
            
            # Validate token count
            is_valid, error, token_count = validate_token_count(parsed_doc.content)
            parsed_doc.metadata.token_count = token_count
            
            if not is_valid:
                parsed_doc.errors.append(error)
            
            return parsed_doc
            
        except Exception as e:
            return ParsedDocument(
                content="",
                errors=[f"Parsing failed: {str(e)}"],
                metadata=DocumentMetadata(
                    filename=Path(file_path).name,
                    file_size_bytes=Path(file_path).stat().st_size,
                    format=self.get_format(),
                    extraction_method=self.default_extraction_method,
                    processing_time_seconds=time.time() - start_time
                )
            )
    
    def _create_section(self, content: str, section_type: str = "paragraph", 
                       page_number: Optional[int] = None, position: Optional[int] = None,
                       confidence: Optional[float] = None) -> dict:
        """
        Helper method to create a section dictionary.
        
        Args:
            content: Section content
            section_type: Type of section
            page_number: Page number if applicable
            position: Character position in document
            confidence: OCR confidence if applicable
            
        Returns:
            Section dictionary
        """
        return {
            "content": content,
            "section_type": section_type,
            "page_number": page_number,
            "position": position,
            "confidence": confidence,
            "metadata": {}
        }