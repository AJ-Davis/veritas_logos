"""
Document models for unified representation of parsed documents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"


class ExtractionMethod(str, Enum):
    """Methods used for text extraction."""
    DIRECT = "direct"  # Text directly extracted
    OCR = "ocr"  # Optical Character Recognition
    HYBRID = "hybrid"  # Combination of methods


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""
    filename: str
    file_size_bytes: int
    format: DocumentFormat
    extraction_method: ExtractionMethod
    processed_at: datetime = field(default_factory=datetime.utcnow)
    processing_time_seconds: Optional[float] = None
    checksum: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    token_count: Optional[int] = None
    language: Optional[str] = None
    encoding: Optional[str] = None
    
    def __post_init__(self):
        """Generate checksum if not provided."""
        if self.checksum is None and hasattr(self, '_file_content'):
            self.checksum = hashlib.sha256(self._file_content).hexdigest()


@dataclass
class DocumentSection:
    """A section or paragraph within a document."""
    content: str
    section_type: str = "paragraph"  # paragraph, heading, list_item, table, etc.
    page_number: Optional[int] = None
    position: Optional[int] = None  # Character position in document
    confidence: Optional[float] = None  # OCR confidence if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Unified representation of a parsed document."""
    content: str  # Full text content
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
    raw_data: Optional[Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if the document was parsed successfully."""
        return len(self.content.strip()) > 0 and len(self.errors) == 0
    
    @property
    def token_count(self) -> Optional[int]:
        """Get token count from metadata."""
        return self.metadata.token_count if self.metadata else None
    
    @property
    def word_count(self) -> int:
        """Calculate word count from content."""
        return len(self.content.split())
    
    def get_sections_by_type(self, section_type: str) -> List[DocumentSection]:
        """Get all sections of a specific type."""
        return [section for section in self.sections if section.section_type == section_type]
    
    def get_content_by_page(self, page_number: int) -> str:
        """Get content from a specific page."""
        page_sections = [
            section for section in self.sections 
            if section.page_number == page_number
        ]
        return "\n".join(section.content for section in page_sections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "sections": [
                {
                    "content": section.content,
                    "section_type": section.section_type,
                    "page_number": section.page_number,
                    "position": section.position,
                    "confidence": section.confidence,
                    "metadata": section.metadata
                }
                for section in self.sections
            ],
            "metadata": {
                "filename": self.metadata.filename,
                "file_size_bytes": self.metadata.file_size_bytes,
                "format": self.metadata.format.value,
                "extraction_method": self.metadata.extraction_method.value,
                "processed_at": self.metadata.processed_at.isoformat(),
                "processing_time_seconds": self.metadata.processing_time_seconds,
                "checksum": self.metadata.checksum,
                "page_count": self.metadata.page_count,
                "word_count": self.metadata.word_count,
                "character_count": self.metadata.character_count,
                "token_count": self.metadata.token_count,
                "language": self.metadata.language,
                "encoding": self.metadata.encoding
            } if self.metadata else None,
            "raw_data": self.raw_data,
            "errors": self.errors,
            "warnings": self.warnings
        }