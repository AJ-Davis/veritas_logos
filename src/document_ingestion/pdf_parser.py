"""
PDF document parser using pdfplumber and pytesseract for OCR.
"""

import pdfplumber
import pytesseract
from PIL import Image
import io
from pathlib import Path
from typing import List, Tuple, Optional

from .base_parser import BaseDocumentParser
from ..models.document import ParsedDocument, DocumentSection, DocumentFormat, ExtractionMethod


class PdfParser(BaseDocumentParser):
    """Parser for PDF files with OCR fallback."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.pdf'}
        self.ocr_threshold = 50  # Minimum characters per page before trying OCR
    
    def get_format(self) -> DocumentFormat:
        """Return the document format this parser handles."""
        return DocumentFormat.PDF
    
    def _parse_content(self, file_path: str) -> ParsedDocument:
        """
        Parse PDF file content with OCR fallback.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedDocument instance
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                # First try direct text extraction
                text_content, sections, extraction_method = self._extract_text_content(pdf)
                
                # If direct extraction yields little text, try OCR
                if len(text_content.strip()) < self.ocr_threshold * len(pdf.pages):
                    ocr_content, ocr_sections, _ = self._extract_with_ocr(pdf)
                    if len(ocr_content.strip()) > len(text_content.strip()):
                        text_content = ocr_content
                        sections = ocr_sections
                        extraction_method = ExtractionMethod.OCR
                    else:
                        extraction_method = ExtractionMethod.HYBRID
                
                # Set extraction method for this parser
                self.default_extraction_method = extraction_method
                
                return ParsedDocument(
                    content=text_content,
                    sections=sections,
                    raw_data={
                        "page_count": len(pdf.pages),
                        "extraction_method": extraction_method.value
                    }
                )
                
        except Exception as e:
            return ParsedDocument(
                content="",
                errors=[f"Failed to parse PDF file: {str(e)}"]
            )
    
    def _extract_text_content(self, pdf) -> Tuple[str, List[DocumentSection], ExtractionMethod]:
        """
        Extract text content directly from PDF.
        
        Args:
            pdf: pdfplumber PDF object
            
        Returns:
            Tuple of (content, sections, extraction_method)
        """
        all_text = []
        sections = []
        position = 0
        
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                # Extract text from page
                page_text = page.extract_text()
                
                if page_text:
                    all_text.append(page_text)
                    
                    # Create sections for this page
                    page_sections = self._create_page_sections(page_text, page_num, position)
                    sections.extend(page_sections)
                    
                    position += len(page_text) + 1
                    
            except Exception as e:
                # Add error section for failed page
                sections.append(DocumentSection(
                    content=f"[Error extracting page {page_num}: {str(e)}]",
                    section_type="error",
                    page_number=page_num,
                    position=position
                ))
        
        content = '\n\n'.join(all_text)
        return content, sections, ExtractionMethod.DIRECT
    
    def _extract_with_ocr(self, pdf) -> Tuple[str, List[DocumentSection], ExtractionMethod]:
        """
        Extract text content using OCR.
        
        Args:
            pdf: pdfplumber PDF object
            
        Returns:
            Tuple of (content, sections, extraction_method)
        """
        all_text = []
        sections = []
        position = 0
        
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                # Convert page to image
                image = page.to_image(resolution=300)
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(image.original)
                
                if ocr_text.strip():
                    all_text.append(ocr_text)
                    
                    # Create sections for this page with OCR confidence
                    page_sections = self._create_page_sections_with_ocr(
                        page, ocr_text, page_num, position
                    )
                    sections.extend(page_sections)
                    
                    position += len(ocr_text) + 1
                    
            except Exception as e:
                # Add error section for failed OCR
                sections.append(DocumentSection(
                    content=f"[OCR failed for page {page_num}: {str(e)}]",
                    section_type="error",
                    page_number=page_num,
                    position=position
                ))
        
        content = '\n\n'.join(all_text)
        return content, sections, ExtractionMethod.OCR
    
    def _create_page_sections(self, page_text: str, page_num: int, start_position: int) -> List[DocumentSection]:
        """
        Create sections from page text.
        
        Args:
            page_text: Text content of the page
            page_num: Page number
            start_position: Starting character position
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        # Split into paragraphs
        paragraphs = page_text.split('\n\n')
        position = start_position
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                # Determine section type
                section_type = self._determine_section_type(paragraph)
                
                sections.append(DocumentSection(
                    content=paragraph,
                    section_type=section_type,
                    page_number=page_num,
                    position=position
                ))
                
                position += len(paragraph) + 2
        
        return sections
    
    def _create_page_sections_with_ocr(self, page, ocr_text: str, page_num: int, 
                                      start_position: int) -> List[DocumentSection]:
        """
        Create sections from OCR text with confidence scores.
        
        Args:
            page: pdfplumber page object
            ocr_text: OCR extracted text
            page_num: Page number
            start_position: Starting character position
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        try:
            # Get OCR data with confidence scores
            image = page.to_image(resolution=300)
            ocr_data = pytesseract.image_to_data(image.original, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence for the page
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
        except Exception:
            avg_confidence = None
        
        # Split into paragraphs
        paragraphs = ocr_text.split('\n\n')
        position = start_position
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                # Determine section type
                section_type = self._determine_section_type(paragraph)
                
                sections.append(DocumentSection(
                    content=paragraph,
                    section_type=section_type,
                    page_number=page_num,
                    position=position,
                    confidence=avg_confidence / 100 if avg_confidence else None
                ))
                
                position += len(paragraph) + 2
        
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
        lines = text.split('\n')
        
        # Check for headings (short lines, often centered or bold-formatted)
        if len(lines) == 1 and len(text) < 100:
            # Check if it looks like a heading
            if text.isupper() or not text.endswith('.'):
                return "heading"
        
        # Check for lists
        if any(line.strip().startswith(('•', '-', '*', '◦')) for line in lines):
            return "list_item"
        
        # Check for numbered lists
        if any(line.strip() and line.strip()[0].isdigit() and 
               ('.' in line[:10] or ')' in line[:10]) for line in lines):
            return "list_item"
        
        # Check for tables (multiple columns separated by spaces)
        if len(lines) > 1:
            # Simple heuristic: if multiple lines have similar structure, might be a table
            space_counts = [line.count('  ') for line in lines if line.strip()]
            if space_counts and max(space_counts) >= 2 and len(set(space_counts)) <= 2:
                return "table"
        
        # Default to paragraph
        return "paragraph"