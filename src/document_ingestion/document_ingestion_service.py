"""
Document Ingestion Service - Unified API for document parsing.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import mimetypes

from .base_parser import BaseDocumentParser
from .txt_parser import TxtParser
from .markdown_parser import MarkdownParser
from .pdf_parser import PdfParser
from .docx_parser import DocxParser
from ..models.document import ParsedDocument, DocumentFormat
from ..utils.validation import validate_file_before_processing


class DocumentIngestionService:
    """
    Unified service for document parsing and ingestion.
    
    This service provides a clean API that automatically selects the appropriate
    parser based on file type and handles validation and error handling.
    """
    
    def __init__(self):
        """Initialize the document ingestion service with all parsers."""
        self.parsers: List[BaseDocumentParser] = [
            TxtParser(),
            MarkdownParser(),
            PdfParser(),
            DocxParser()
        ]
        
        # Create mapping of extensions to parsers for quick lookup
        self._parser_map: Dict[str, BaseDocumentParser] = {}
        for parser in self.parsers:
            for ext in parser.supported_extensions:
                self._parser_map[ext] = parser
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return list(self._parser_map.keys())
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if a file can be parsed by this service.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file can be parsed
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self._parser_map
    
    def get_parser_for_file(self, file_path: str) -> Optional[BaseDocumentParser]:
        """
        Get the appropriate parser for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser instance or None if no suitable parser found
        """
        file_ext = Path(file_path).suffix.lower()
        return self._parser_map.get(file_ext)
    
    def parse_document(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        Parse a document file and return structured content.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional arguments to pass to the parser
            
        Returns:
            ParsedDocument instance with content, metadata, and any errors
        """
        # Quick validation
        is_valid, error = validate_file_before_processing(file_path)
        if not is_valid:
            return ParsedDocument(
                content="",
                errors=[f"Pre-processing validation failed: {error}"]
            )
        
        # Get appropriate parser
        parser = self.get_parser_for_file(file_path)
        if not parser:
            supported = ', '.join(sorted(self.get_supported_formats()))
            return ParsedDocument(
                content="",
                errors=[f"No parser available for file type. Supported formats: {supported}"]
            )
        
        # Parse the document
        try:
            return parser.parse(file_path)
        except Exception as e:
            return ParsedDocument(
                content="",
                errors=[f"Unexpected error during parsing: {str(e)}"]
            )
    
    def parse_multiple_documents(self, file_paths: List[str]) -> List[ParsedDocument]:
        """
        Parse multiple documents.
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            List of ParsedDocument instances
        """
        results = []
        for file_path in file_paths:
            result = self.parse_document(file_path)
            results.append(result)
        return results
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a document without fully parsing it.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with basic document information
        """
        file_path_obj = Path(file_path)
        
        # Basic file information
        info = {
            "filename": file_path_obj.name,
            "file_extension": file_path_obj.suffix.lower(),
            "file_size_bytes": file_path_obj.stat().st_size if file_path_obj.exists() else 0,
            "can_parse": self.can_parse(file_path),
            "parser_type": None
        }
        
        # Add parser information if available
        parser = self.get_parser_for_file(file_path)
        if parser:
            info["parser_type"] = parser.__class__.__name__
            info["document_format"] = parser.get_format().value
        
        # Add MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        info["mime_type"] = mime_type
        
        return info
    
    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a document for parsing without actually parsing it.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with validation results
        """
        is_valid, error = validate_file_before_processing(file_path)
        
        validation_result = {
            "is_valid": is_valid,
            "error": error,
            "can_parse": self.can_parse(file_path) if is_valid else False,
            "file_info": self.get_document_info(file_path) if is_valid else None
        }
        
        return validation_result
    
    def get_parser_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available parsers.
        
        Returns:
            Dictionary with parser statistics
        """
        return {
            "total_parsers": len(self.parsers),
            "supported_extensions": list(self._parser_map.keys()),
            "parser_types": [parser.__class__.__name__ for parser in self.parsers],
            "format_mapping": {
                ext: parser.__class__.__name__ 
                for ext, parser in self._parser_map.items()
            }
        }


# Global service instance for convenience
document_service = DocumentIngestionService()