"""
Tests for document ingestion system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.document_ingestion import document_service, DocumentIngestionService
from src.models.document import DocumentFormat


class TestDocumentIngestionService:
    """Test cases for DocumentIngestionService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DocumentIngestionService()
    
    def test_service_initialization(self):
        """Test that the service initializes correctly."""
        assert len(self.service.parsers) >= 4
        assert len(self.service.get_supported_formats()) >= 4
        
        # Check that we have parsers for all expected formats
        supported = self.service.get_supported_formats()
        assert '.txt' in supported
        assert '.md' in supported
        assert '.pdf' in supported
        assert '.docx' in supported
    
    def test_can_parse_supported_formats(self):
        """Test that service correctly identifies supported formats."""
        assert self.service.can_parse('test.txt')
        assert self.service.can_parse('test.md')
        assert self.service.can_parse('test.markdown')
        assert self.service.can_parse('test.pdf')
        assert self.service.can_parse('test.docx')
        
        # Unsupported formats
        assert not self.service.can_parse('test.xlsx')
        assert not self.service.can_parse('test.pptx')
        assert not self.service.can_parse('test.jpg')
    
    def test_get_parser_for_file(self):
        """Test parser selection based on file extension."""
        txt_parser = self.service.get_parser_for_file('test.txt')
        assert txt_parser is not None
        assert txt_parser.get_format() == DocumentFormat.TXT
        
        md_parser = self.service.get_parser_for_file('test.md')
        assert md_parser is not None
        assert md_parser.get_format() == DocumentFormat.MARKDOWN
        
        # Unsupported format
        unknown_parser = self.service.get_parser_for_file('test.unknown')
        assert unknown_parser is None
    
    def test_parse_text_document(self):
        """Test parsing a simple text document."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nEnd of document.")
            temp_path = f.name
        
        try:
            # Parse the document
            result = self.service.parse_document(temp_path)
            
            # Verify results
            assert result.is_valid
            assert len(result.errors) == 0
            assert "test document" in result.content
            assert len(result.sections) > 0
            assert result.metadata is not None
            assert result.metadata.format == DocumentFormat.TXT
            assert result.metadata.file_size_bytes > 0
            assert result.metadata.token_count > 0
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_parse_markdown_document(self):
        """Test parsing a Markdown document."""
        markdown_content = """# Test Document

This is a **test** markdown document.

## Section 1

- Item 1
- Item 2
- Item 3

### Subsection

Some more content here.

```python
def hello():
    print("Hello, world!")
```

The end.
"""
        
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            temp_path = f.name
        
        try:
            # Parse the document
            result = self.service.parse_document(temp_path)
            
            # Verify results
            assert result.is_valid
            assert len(result.errors) == 0
            assert "Test Document" in result.content
            assert len(result.sections) > 0
            assert result.metadata is not None
            assert result.metadata.format == DocumentFormat.MARKDOWN
            assert 'This is a **test** markdown document.' in result.raw_data['markdown_content']
            
            # Check that sections are properly identified
            headings = result.get_sections_by_type("heading")
            assert len(headings) > 0
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = self.service.parse_document('/nonexistent/file.txt')
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "does not exist" in result.errors[0].lower()
    
    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            result = self.service.parse_document(temp_path)
            
            assert not result.is_valid
            assert len(result.errors) > 0
            assert "Unsupported file extension" in result.errors[0]
            
        finally:
            os.unlink(temp_path)
    
    def test_get_document_info(self):
        """Test getting document information without parsing."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            info = self.service.get_document_info(temp_path)
            
            assert info['filename'] == Path(temp_path).name
            assert info['file_extension'] == '.txt'
            assert info['file_size_bytes'] > 0
            assert info['can_parse'] is True
            assert info['parser_type'] == 'TxtParser'
            assert info['document_format'] == 'txt'
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_document(self):
        """Test document validation."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            validation = self.service.validate_document(temp_path)
            
            assert validation['is_valid'] is True
            assert validation['error'] is None
            assert validation['can_parse'] is True
            assert validation['file_info'] is not None
            
        finally:
            os.unlink(temp_path)
    
    def test_parser_statistics(self):
        """Test getting parser statistics."""
        stats = self.service.get_parser_statistics()
        
        assert stats['total_parsers'] == 4
        assert len(stats['supported_extensions']) >= 4
        assert len(stats['parser_types']) == 4
        assert '.txt' in stats['format_mapping']
        assert stats['format_mapping']['.txt'] == 'TxtParser'
    
    def test_global_service_instance(self):
        """Test that the global service instance works."""
        assert document_service is not None
        assert isinstance(document_service, DocumentIngestionService)
        assert len(document_service.get_supported_formats()) >= 4


if __name__ == '__main__':
    pytest.main([__file__])